"""Conformal cascade router (benchmarks/suites/conformal_cascade_router.py).

Covers: theoretical coverage of a single-tier split-conformal calibration, low-cost early
commit on confident Tier-1 samples, escalation of ambiguous/adversarial and out-of-distribution
(empty-set) samples up the cascade, safe terminal abstention (never a guessed label, including
on aegis_safety), fail-closed rejection of malformed input, and monotone/consistent expected
cost and coverage across alpha.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suites"))

import conformal_cascade_router as ccr  # noqa: E402
from conformal_cascade_router import (  # noqa: E402
    CascadeEvaluationReport,
    ConformalCascadeRouter,
    Verdict,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _make_probs(rng, labels, n_classes, signal, noise_scale):
    n = labels.shape[0]
    logits = rng.normal(0.0, noise_scale, size=(n, n_classes))
    logits[np.arange(n), labels] += signal
    return _softmax(logits)


def _labels(rng, n, n_classes):
    return rng.integers(0, n_classes, size=n)


# ------------------------------------------------------------------ theoretical coverage

def test_single_tier_calibration_achieves_theoretical_coverage():
    rng = np.random.default_rng(20260927)
    n_classes = 4
    alpha = 0.1
    n_calib, n_test = 3000, 4000

    calib_labels = _labels(rng, n_calib, n_classes)
    calib_probs = _make_probs(rng, calib_labels, n_classes, signal=2.0, noise_scale=1.0)
    test_labels = _labels(rng, n_test, n_classes)
    test_probs = _make_probs(rng, test_labels, n_classes, signal=2.0, noise_scale=1.0)

    router = ConformalCascadeRouter().calibrate([calib_probs], calib_labels, alpha=alpha)
    report = router.evaluate_cascade([test_probs], test_labels, costs=[1.0], alpha=alpha)

    coverage = report.empirical_coverage_by_tier["tier_1"]
    assert coverage >= 1.0 - alpha - 0.03, coverage


@pytest.mark.parametrize("alpha", [0.2, 0.1, 0.05, 0.01])
def test_coverage_tracks_1_minus_alpha_across_settings(alpha):
    rng = np.random.default_rng(7)
    n_classes = 5
    n_calib, n_test = 4000, 4000

    calib_labels = _labels(rng, n_calib, n_classes)
    calib_probs = _make_probs(rng, calib_labels, n_classes, signal=1.5, noise_scale=1.2)
    test_labels = _labels(rng, n_test, n_classes)
    test_probs = _make_probs(rng, test_labels, n_classes, signal=1.5, noise_scale=1.2)

    router = ConformalCascadeRouter().calibrate([calib_probs], calib_labels, alpha=alpha)
    report = router.evaluate_cascade([test_probs], test_labels, costs=[1.0], alpha=alpha)
    assert report.empirical_coverage_by_tier["tier_1"] >= 1.0 - alpha - 0.03


# ------------------------------------------------------------------ early commit / escalation / abstain

def _three_tier_router(rng, n_calib=2000, n_classes=3, alpha=0.1, score_method="margin"):
    calib_labels = _labels(rng, n_calib, n_classes)
    tier1 = _make_probs(rng, calib_labels, n_classes, signal=1.5, noise_scale=1.0)
    tier2a = _make_probs(rng, calib_labels, n_classes, signal=1.5, noise_scale=1.0)
    tier2b = _make_probs(rng, calib_labels, n_classes, signal=1.5, noise_scale=1.0)
    router = ConformalCascadeRouter(
        tier_names=("tier1_70b_72b", "tier2a_123b", "tier2b_405b"),
        costs=(1.0, 3.0, 8.0),
        score_method=score_method,
    ).calibrate([tier1, tier2a, tier2b], calib_labels, alpha=alpha)
    return router


def test_confident_sample_commits_at_tier1_with_single_tier_cost():
    rng = np.random.default_rng(11)
    router = _three_tier_router(rng)
    # Top class comfortably clears tier1's q_hat (~0.96) while top+second would not, so tier1's
    # conformal set is the singleton {0}: no need to ever look at tier2a/tier2b's probabilities.
    confident = np.array([0.93, 0.05, 0.02])
    unused = np.full(3, 1.0 / 3.0)

    decision = router.route_sample([confident, unused, unused], task_name="demo")
    assert decision.verdict == Verdict.COMMIT.value
    assert decision.committed is True
    assert decision.label == 0
    assert decision.prediction_set == (0,)
    assert decision.committed_tier == "tier1_70b_72b"
    assert decision.tiers_visited == ("tier1_70b_72b",)
    assert decision.n_escalations == 0
    assert decision.cost == pytest.approx(1.0)  # only tier1's cost, tier2a/2b never run
    assert decision.reason == "tier1_70b_72b_SINGLETON"


def test_very_confident_sample_commits_at_tier1_under_either_score_method():
    # Regression guard: a naive "score(x, c) <= q_hat for every c independently" APS variant
    # excludes the top class whenever p_top(x) > q_hat, turning the MOST confident, correct
    # samples into empty sets that force a terminal abstain. Both supported score methods must
    # instead commit a near-certain sample at tier 1.
    very_confident = np.array([0.999, 0.0005, 0.0005])
    unused = np.full(3, 1.0 / 3.0)
    for method in ("margin", "aps"):
        rng = np.random.default_rng(11)
        router = _three_tier_router(rng, score_method=method)
        decision = router.route_sample([very_confident, unused, unused], task_name="demo")
        assert decision.verdict == Verdict.COMMIT.value, method
        assert decision.label == 0, method
        assert decision.tiers_visited == ("tier1_70b_72b",), method


def test_ambiguous_tier1_escalates_and_tied_tier2a_escalates_to_tier2b():
    rng = np.random.default_rng(12)
    router = _three_tier_router(rng)
    # No class clearly separates from the pack -> |C| == 2 (genuine ambiguity) at tier1, and an
    # exact top-two tie -> |C| == 2 at tier2a too, forcing two escalations before tier2b's
    # confident distribution can commit.
    ambiguous = np.array([0.45, 0.40, 0.15])
    tied = np.array([0.49, 0.49, 0.02])
    confident = np.array([0.02, 0.03, 0.95])

    decision = router.route_sample([ambiguous, tied, confident], task_name="demo")
    assert decision.verdict == Verdict.COMMIT.value
    assert decision.committed_tier == "tier2b_405b"
    assert decision.tiers_visited == ("tier1_70b_72b", "tier2a_123b", "tier2b_405b")
    assert decision.n_escalations == 2
    assert decision.label == 2
    assert decision.cost == pytest.approx(1.0 + 3.0 + 8.0)


def test_terminal_ambiguity_safely_abstains_and_never_guesses_on_aegis_safety():
    rng = np.random.default_rng(13)
    router = _three_tier_router(rng)
    ambiguous = np.array([0.45, 0.40, 0.15])

    decision = router.route_sample([ambiguous, ambiguous, ambiguous], task_name="aegis_safety")
    assert decision.verdict == Verdict.ABSTAIN.value
    assert decision.abstained is True
    assert decision.committed is False
    assert decision.label is None  # never a silent guess, especially on a safety task
    assert decision.committed_tier is None
    assert decision.reason == "TERMINAL_ABSTAIN_AMBIGUOUS"
    assert len(decision.prediction_set) >= 2
    assert decision.tiers_visited == ("tier1_70b_72b", "tier2a_123b", "tier2b_405b")
    assert decision.n_escalations == 2
    assert decision.task_name == "aegis_safety"


def test_out_of_distribution_empty_set_escalates_and_can_still_abstain():
    # Under the default "margin" score, an empty set means "no class cleared the confidence
    # bar" -- reachable only when the calibrated threshold sits below 1 - 1/n_classes, so use
    # enough classes that a flat/near-uniform point genuinely fails every one of them.
    rng = np.random.default_rng(21)
    n_classes = 6
    n_calib = 3000
    calib_labels = _labels(rng, n_calib, n_classes)
    calib_probs = _make_probs(rng, calib_labels, n_classes, signal=2.5, noise_scale=1.0)
    router = ConformalCascadeRouter(costs=(1.0,)).calibrate([calib_probs], calib_labels, alpha=0.1)
    q_hat = router.tier_calibrations[0].q_hat
    assert q_hat < 1.0 - 1.0 / n_classes  # precondition: uniform-over-K is excludable at all

    flat = np.full(n_classes, 1.0 / n_classes)
    decision = router.route_sample([flat], task_name="demo")
    assert decision.prediction_set == ()
    assert decision.verdict == Verdict.ABSTAIN.value
    assert decision.label is None
    assert decision.reason == "TERMINAL_ABSTAIN_EMPTY"


def test_calibration_set_too_small_for_alpha_is_fail_closed():
    # n=10 calibration points cannot support alpha=0.05 (needs ceil((n+1)*0.95) <= n, i.e. n >=
    # 19): silently returning an infinite threshold would make every sample escalate forever
    # with no error. That must raise instead.
    rng = np.random.default_rng(99)
    labels = _labels(rng, 10, 3)
    probs = _make_probs(rng, labels, 3, signal=1.0, noise_scale=1.0)
    with pytest.raises(ValueError, match="too small for alpha"):
        ConformalCascadeRouter().calibrate([probs], labels, alpha=0.05)
    # A larger calibration set at the same alpha must calibrate fine.
    labels2 = _labels(rng, 50, 3)
    probs2 = _make_probs(rng, labels2, 3, signal=1.0, noise_scale=1.0)
    ConformalCascadeRouter().calibrate([probs2], labels2, alpha=0.05)


# ------------------------------------------------------------------ fail-closed

def test_calibrate_rejects_nan_inf_and_unnormalized_rows():
    rng = np.random.default_rng(1)
    labels = _labels(rng, 10, 3)
    probs = _make_probs(rng, labels, 3, signal=1.0, noise_scale=1.0)

    bad_nan = probs.copy()
    bad_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        ConformalCascadeRouter().calibrate([bad_nan], labels)

    bad_inf = probs.copy()
    bad_inf[1, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        ConformalCascadeRouter().calibrate([bad_inf], labels)

    bad_sum = probs.copy()
    bad_sum[2] = np.array([0.9, 0.9, 0.9])
    with pytest.raises(ValueError, match="sum to 1"):
        ConformalCascadeRouter().calibrate([bad_sum], labels)


def test_calibrate_rejects_label_and_shape_mismatches():
    rng = np.random.default_rng(2)
    labels = _labels(rng, 10, 3)
    probs = _make_probs(rng, labels, 3, signal=1.0, noise_scale=1.0)

    with pytest.raises(ValueError, match="labels must have shape"):
        ConformalCascadeRouter().calibrate([probs], labels[:5])

    with pytest.raises(ValueError, match="labels must be in"):
        ConformalCascadeRouter().calibrate([probs], labels + 10)

    tier2 = probs[:-1]
    with pytest.raises(ValueError, match="expected 10"):
        ConformalCascadeRouter().calibrate([probs, tier2], labels)

    with pytest.raises(ValueError, match="non-empty sequence"):
        ConformalCascadeRouter().calibrate([], labels)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])
def test_calibrate_rejects_bad_alpha(bad_alpha):
    rng = np.random.default_rng(3)
    labels = _labels(rng, 10, 3)
    probs = _make_probs(rng, labels, 3, signal=1.0, noise_scale=1.0)
    with pytest.raises(ValueError, match="alpha"):
        ConformalCascadeRouter().calibrate([probs], labels, alpha=bad_alpha)


def test_route_sample_requires_calibration_and_matching_tier_count():
    with pytest.raises(ValueError, match="not calibrated"):
        ConformalCascadeRouter().route_sample([np.array([0.5, 0.5])])

    rng = np.random.default_rng(4)
    labels = _labels(rng, 60, 3)
    probs = _make_probs(rng, labels, 3, signal=1.0, noise_scale=1.0)
    router = ConformalCascadeRouter().calibrate([probs], labels)
    with pytest.raises(ValueError, match="tiers"):
        router.route_sample([probs[0], probs[1]])


def test_route_sample_rejects_bad_probability_vector():
    rng = np.random.default_rng(5)
    labels = _labels(rng, 60, 3)
    probs = _make_probs(rng, labels, 3, signal=1.0, noise_scale=1.0)
    router = ConformalCascadeRouter().calibrate([probs], labels)

    with pytest.raises(ValueError, match="NaN or Inf"):
        router.route_sample([np.array([np.nan, 0.5, 0.5])])
    with pytest.raises(ValueError, match="sum to 1"):
        router.route_sample([np.array([0.9, 0.9, 0.9])])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        router.route_sample([np.array([1.5, -0.5, 0.0])])


def test_evaluate_cascade_requires_matching_alpha_and_shapes():
    rng = np.random.default_rng(6)
    labels = _labels(rng, 200, 3)
    probs = _make_probs(rng, labels, 3, signal=1.5, noise_scale=1.0)
    router = ConformalCascadeRouter().calibrate([probs], labels, alpha=0.1)

    with pytest.raises(ValueError, match="does not match the calibrated alpha"):
        router.evaluate_cascade([probs], labels, costs=[1.0], alpha=0.2)

    with pytest.raises(ValueError, match="costs has"):
        router.evaluate_cascade([probs], labels, costs=[1.0, 2.0], alpha=0.1)

    with pytest.raises(ValueError, match="cost must be finite"):
        router.evaluate_cascade([probs], labels, costs=[-1.0], alpha=0.1)

    with pytest.raises(ValueError, match="tiers_probs has"):
        router.evaluate_cascade([probs, probs], labels, costs=[1.0], alpha=0.1)


def test_evaluate_cascade_rejects_bad_thresholds():
    rng = np.random.default_rng(9)
    labels = _labels(rng, 200, 3)
    probs = _make_probs(rng, labels, 3, signal=1.5, noise_scale=1.0)
    router = ConformalCascadeRouter().calibrate([probs], labels, alpha=0.1)

    with pytest.raises(ValueError, match="entropy_threshold"):
        router.evaluate_cascade([probs], labels, costs=[1.0], alpha=0.1, entropy_threshold=0.0)
    with pytest.raises(ValueError, match="max_prob_threshold"):
        router.evaluate_cascade([probs], labels, costs=[1.0], alpha=0.1, max_prob_threshold=1.5)


def test_bad_score_method_rejected():
    with pytest.raises(ValueError, match="score_method"):
        ConformalCascadeRouter(score_method="bogus")


def test_tier_names_and_costs_length_must_match_tier_count():
    rng = np.random.default_rng(10)
    labels = _labels(rng, 50, 3)
    probs = _make_probs(rng, labels, 3, signal=1.0, noise_scale=1.0)

    with pytest.raises(ValueError, match="tier_names has"):
        ConformalCascadeRouter(tier_names=("a", "b")).calibrate([probs], labels)
    with pytest.raises(ValueError, match="costs has"):
        ConformalCascadeRouter(costs=(1.0, 2.0)).calibrate([probs], labels)


# ------------------------------------------------------------------ expected cost / coverage monotonicity

def test_expected_cost_and_coverage_are_monotone_in_alpha_and_deterministic():
    rng = np.random.default_rng(2026)
    n_classes = 4
    n_calib, n_test = 3000, 3000
    calib_labels = _labels(rng, n_calib, n_classes)
    tier1_calib = _make_probs(rng, calib_labels, n_classes, signal=1.2, noise_scale=1.3)
    tier2_calib = _make_probs(rng, calib_labels, n_classes, signal=1.2, noise_scale=1.3)
    test_labels = _labels(rng, n_test, n_classes)
    tier1_test = _make_probs(rng, test_labels, n_classes, signal=1.2, noise_scale=1.3)
    tier2_test = _make_probs(rng, test_labels, n_classes, signal=1.2, noise_scale=1.3)
    costs = [1.0, 5.0]

    reports = {}
    for alpha in (0.3, 0.1, 0.02):
        router = ConformalCascadeRouter().calibrate([tier1_calib, tier2_calib], calib_labels, alpha=alpha)
        reports[alpha] = router.evaluate_cascade(
            [tier1_test, tier2_test], test_labels, costs=costs, alpha=alpha)

    # Smaller alpha -> larger q_hat -> prediction sets only grow (never shrink) on this fixed
    # test set, so both expected cost (more escalation) and empirical coverage (per tier) are
    # EXACTLY non-decreasing as alpha shrinks -- not just above their theoretical floor.
    alphas_desc = [0.3, 0.1, 0.02]
    costs_seq = [reports[a].expected_cost for a in alphas_desc]
    assert costs_seq == sorted(costs_seq)
    for tier_name in ("tier_1", "tier_2"):
        cov_seq = [reports[a].empirical_coverage_by_tier[tier_name] for a in alphas_desc]
        assert cov_seq == sorted(cov_seq), tier_name
    for a in alphas_desc:
        assert reports[a].empirical_coverage_by_tier["tier_1"] >= 1.0 - a - 0.03

    # Determinism / internal consistency: re-running with identical inputs reproduces the report.
    router = ConformalCascadeRouter().calibrate([tier1_calib, tier2_calib], calib_labels, alpha=0.1)
    r1 = router.evaluate_cascade([tier1_test, tier2_test], test_labels, costs=costs, alpha=0.1)
    r2 = router.evaluate_cascade([tier1_test, tier2_test], test_labels, costs=costs, alpha=0.1)
    assert r1 == r2
    # Selective accuracy on committed samples should never be lower than plain overall accuracy,
    # since abstained samples are counted as failures in overall_accuracy but excluded from selective.
    assert r1.selective_accuracy >= r1.overall_accuracy


def test_baseline_reports_are_present_and_shaped_like_the_conformal_report():
    rng = np.random.default_rng(2027)
    n_classes = 3
    labels = _labels(rng, 500, n_classes)
    tier1 = _make_probs(rng, labels, n_classes, signal=1.5, noise_scale=1.0)
    tier2 = _make_probs(rng, labels, n_classes, signal=1.5, noise_scale=1.0)
    router = ConformalCascadeRouter().calibrate([tier1, tier2], labels, alpha=0.1)
    report = router.evaluate_cascade([tier1, tier2], labels, costs=[1.0, 2.0], alpha=0.1)

    assert isinstance(report, CascadeEvaluationReport)
    for baseline in (report.baseline_entropy, report.baseline_max_prob):
        assert 0.0 <= baseline.overall_accuracy <= 1.0
        assert 0.0 <= baseline.abstention_rate <= 1.0
        assert baseline.expected_cost > 0.0
        assert set(baseline.escalation_rate_by_tier) == {"tier_1", "tier_2"}


@pytest.mark.parametrize("method", ["margin", "aps"])
def test_extreme_binary_confidence_commits_at_tier1(method):
    probs = np.tile([0.8, 0.2], (19, 1))
    router = ConformalCascadeRouter(costs=(1.0, 3.0), score_method=method).calibrate(
        [probs, probs], np.zeros(19, dtype=int), alpha=0.05)

    decision = router.route_sample([[0.999, 0.001], [0.5, 0.5]])
    assert decision.verdict == Verdict.COMMIT.value
    assert decision.label == 0
    assert decision.prediction_set == (0,)
    assert decision.committed_tier == "tier_1"
    assert decision.tiers_visited == ("tier_1",)
    assert decision.n_escalations == 0
    assert decision.cost == 1.0


def test_calibration_at_minimum_size_uses_maximum_score():
    p_true = np.linspace(0.6, 0.9, 19)
    probs = np.column_stack((p_true, 1.0 - p_true))
    labels = np.zeros(19, dtype=int)
    scores = 1.0 - p_true

    assert ccr._min_calibration_n(0.05) == 19
    router = ConformalCascadeRouter().calibrate([probs], labels, alpha=0.05)
    assert router.tier_calibrations[0].q_hat == scores.max()
    assert ccr._quantile_threshold(scores, 0.05) == scores.max()
    with pytest.raises(ValueError, match="need at least 19 calibration samples"):
        ConformalCascadeRouter().calibrate([probs[:18]], labels[:18], alpha=0.05)


@pytest.mark.parametrize("bad_row, message", [
    ([1.00005, -0.00005], "values must lie in"),
    ([0.8, 0.20005], "rows must sum to 1"),
])
def test_small_probability_violations_are_rejected(bad_row, message):
    probs = np.tile([0.8, 0.2], (19, 1))
    labels = np.zeros(19, dtype=int)
    router = ConformalCascadeRouter().calibrate([probs], labels)
    with pytest.raises(ValueError, match=message):
        router.route_sample([bad_row])
    probs[0] = bad_row
    with pytest.raises(ValueError, match=message):
        ConformalCascadeRouter().calibrate([probs], labels)


@pytest.mark.parametrize("row", [[1.0 + 5e-8, -5e-8], [0.8, 0.200005]])
def test_machine_rounding_is_accepted_and_normalized(row):
    normalized = ccr._validate_prob_matrix([row], "probs")
    assert (normalized >= 0.0).all()
    assert (normalized <= 1.0).all()
    assert normalized.sum() == pytest.approx(1.0, abs=1e-15)
