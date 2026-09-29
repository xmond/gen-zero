"""Tests for AdaptiveMahalanobisETFChoiceHead
(docs/zero/31-multiscale-dense-resonance-etf-dual-process-plan.md S5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "suites"))

from adaptive_mahalanobis_etf_choice_head import (  # noqa: E402
    ORDINAL_TASK_NAMES,
    AdaptiveMahalanobisETFChoiceHead,
    expected_calibration_error,
)


def _make_classification_data(rng, n_per_class, d, k, scale=3.0, noise=1.0):
    means = rng.normal(scale=scale, size=(k, d))
    y = np.repeat(np.arange(k), n_per_class)
    rng.shuffle(y)
    Z = means[y] + rng.normal(scale=noise, size=(y.size, d))
    return Z, y


# --------------------------------------------------------------------------- fit / predict basics

@pytest.mark.parametrize("k", list(range(2, 11)))
@pytest.mark.parametrize("d", [128, 8192])
def test_fit_predict_shapes_across_k_and_d(k, d):
    rng = np.random.default_rng(1000 + k * 100 + d)
    n_per_class = 12 if d == 8192 else 40
    Z, y = _make_classification_data(rng, n_per_class, d, k)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=0.5)
    probs = head.predict_proba(Z)
    assert probs.shape == (Z.shape[0], k)
    assert np.all(probs >= 0.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)
    preds = head.predict(Z)
    assert preds.shape == (Z.shape[0],)
    assert preds.max() < k and preds.min() >= 0
    # well-separated synthetic classes: the head should mostly recover the generating label
    assert (preds == y).mean() > 0.7


# --------------------------------------------------------------------------- lambda=0 exactness

@pytest.mark.parametrize("k", [2, 3, 5, 10])
def test_lambda_zero_matches_plain_mahalanobis_lda_exactly(k):
    rng = np.random.default_rng(2000 + k)
    Z, y = _make_classification_data(rng, 30, 128, k)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=0.0)

    class_means = np.stack([Z[y == c].mean(axis=0) for c in range(k)])
    direct_whitened_means = head._whiten(class_means)

    np.testing.assert_allclose(head._proto_final_w, direct_whitened_means, atol=1e-8)
    # Gram of the lambda=0 prototypes equals the raw (data-only) Gram, not the ETF one.
    np.testing.assert_allclose(head.G_target_, head.G_raw_, atol=1e-8)


def test_truncated_metric_rank_overstates_precision():
    """When max_metric_rank truncates the true metric rank, the discarded directions are
    scored with variance rho*nu, but their true Sigma_LW eigenvalue is
    (1-rho)*lam_i + rho*nu >= rho*nu. Assumed variance <= true variance means assumed
    precision (1/variance) >= true precision, so truncated squared distances must be >= the
    exact dense-Sigma_LW Mahalanobis distance (truncation overstates confidence, it does not
    understate it -- see the `max_metric_rank` comment in `fit`).
    """
    rng = np.random.default_rng(9501)
    n, d, k = 300, 300, 3  # rank of R is up to min(n, d) - ish, comfortably > 64
    Z, y = _make_classification_data(rng, n // k, d, k, scale=2.0, noise=1.0)

    head_full = AdaptiveMahalanobisETFChoiceHead(max_metric_rank=4096).fit(Z, y, lambda_etf=0.0)
    head_trunc = AdaptiveMahalanobisETFChoiceHead(max_metric_rank=64).fit(Z, y, lambda_etf=0.0)
    assert head_full._metric_truncated is False
    assert head_trunc._metric_truncated is True

    dist2_full = head_full._squared_distances(Z[:20])
    dist2_trunc = head_trunc._squared_distances(Z[:20])
    assert np.all(dist2_trunc >= dist2_full - 1e-6)
    assert np.any(dist2_trunc > dist2_full + 1e-3)  # truncation actually changes something

    probs_trunc = head_trunc.predict_proba(Z)
    np.testing.assert_allclose(probs_trunc.sum(axis=1), 1.0, atol=1e-8)


def test_lambda_zero_scores_equal_negative_mahalanobis_distance():
    rng = np.random.default_rng(3001)
    Z, y = _make_classification_data(rng, 40, 64, 4)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=0.0)

    # Reference Mahalanobis distance computed directly from a dense (unshrunk-inverse) Sigma_LW,
    # built the slow O(D^3) way, independent of the head's whitening trick.
    class_means = np.stack([Z[y == c].mean(axis=0) for c in range(4)])
    R = np.concatenate([Z[y == c] - class_means[c] for c in range(4)], axis=0)
    n, d = Z.shape
    S = (R.T @ R) / n
    rho, nu = head.rho_, head.nu_
    Sigma_lw = (1.0 - rho) * S + rho * nu * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma_lw)

    z0 = Z[0]
    dists_ref = np.array([
        (z0 - class_means[c]) @ Sigma_inv @ (z0 - class_means[c])
        for c in range(4)
    ])
    logits_ref = -dists_ref  # T = 1.0 (default, uncalibrated)
    probs_ref = np.exp(logits_ref - logits_ref.max())
    probs_ref /= probs_ref.sum()

    probs_head = head.predict_proba(z0[None, :])[0]
    np.testing.assert_allclose(probs_head, probs_ref, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- ETF pull

@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_lambda_pulls_gram_toward_etf_monotonically(k):
    rng = np.random.default_rng(4000 + k)
    Z, y = _make_classification_data(rng, 30, 128, k)

    lambdas = [0.0, 0.1, 1.0, 10.0, 1000.0, 1e8]
    distances = [
        np.linalg.norm(
            AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=lam).G_target_
            - AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=lam).G_etf_
        )
        for lam in lambdas
    ]
    assert all(a >= b - 1e-9 for a, b in zip(distances, distances[1:])), distances
    assert distances[0] > distances[-1]
    assert distances[-1] < 1e-4  # effectively hard ETF at lambda=1e8


def test_hard_etf_limit_gram_is_exact():
    rng = np.random.default_rng(4321)
    k = 6
    Z, y = _make_classification_data(rng, 25, 128, k)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=1e10)
    np.testing.assert_allclose(head.G_target_, head.G_etf_, atol=1e-5)
    # The reconstruction is exact by construction (fit() itself Fail-Closes if it weren't);
    # re-derive the achieved Gram directly from the fitted prototypes as an outside check.
    proto_mean = head._proto_final_w.mean(axis=0)
    centered = head._proto_final_w - proto_mean
    achieved = centered @ centered.T
    achieved_normalized = achieved / np.mean(np.diag(achieved))
    np.testing.assert_allclose(achieved_normalized, head.G_etf_, atol=1e-4)


# --------------------------------------------------------------------------- permutation equivariance

@pytest.mark.parametrize("k", [2, 3, 5, 7])
@pytest.mark.parametrize("lambda_etf", [0.0, 1.0, 50.0])
def test_permutation_equivariance(k, lambda_etf):
    rng = np.random.default_rng(5000 + k * 10 + int(lambda_etf))
    Z, y = _make_classification_data(rng, 30, 128, k)

    perm = rng.permutation(k)
    y_perm = perm[y]

    head_orig = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=lambda_etf)
    head_perm = AdaptiveMahalanobisETFChoiceHead().fit(Z, y_perm, lambda_etf=lambda_etf)

    probs_orig = head_orig.predict_proba(Z)
    probs_perm = head_perm.predict_proba(Z)

    # class k under the permuted labeling is the same physical class as perm^{-1}(k) originally
    inv = np.argsort(perm)
    np.testing.assert_allclose(probs_perm, probs_orig[:, inv], atol=1e-6, rtol=1e-5)

    preds_orig = head_orig.predict(Z)
    preds_perm = head_perm.predict(Z)
    np.testing.assert_array_equal(preds_perm, perm[preds_orig])


def test_permutation_equivariance_with_candidate_reps():
    rng = np.random.default_rng(6001)
    k, d = 4, 96
    Z, y = _make_classification_data(rng, 30, d, k)
    candidate_reps = rng.normal(size=(k, d))

    perm = rng.permutation(k)
    y_perm = perm[y]
    candidate_reps_perm = candidate_reps[np.argsort(perm)]  # row k now describes class perm^{-1}(k)...
    # Row for new label `k` must be the reps of whichever physical class is now called k, i.e.
    # candidate_reps_perm[k] = candidate_reps[perm^{-1}(k)] = candidate_reps[inv[k]].
    inv = np.argsort(perm)
    candidate_reps_perm = candidate_reps[inv]

    head_orig = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, candidate_reps=candidate_reps, lambda_etf=3.0)
    head_perm = AdaptiveMahalanobisETFChoiceHead().fit(Z, y_perm, candidate_reps=candidate_reps_perm, lambda_etf=3.0)

    probs_orig = head_orig.predict_proba(Z)
    probs_perm = head_perm.predict_proba(Z)
    np.testing.assert_allclose(probs_perm, probs_orig[:, inv], atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- temperature calibration

def test_temperature_calibration_lowers_ece_on_overconfident_head():
    rng = np.random.default_rng(7001)
    k, d = 4, 64
    # Overlapping classes (small scale, large noise) -> a genuinely imperfect classifier.
    Z, y = _make_classification_data(rng, 150, d, k, scale=1.0, noise=1.8)
    Z_val, y_val = _make_classification_data(rng, 150, d, k, scale=1.0, noise=1.8)

    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=0.0)
    head.temperature_ = 0.05  # force overconfidence relative to true accuracy
    probs_before = head.predict_proba(Z_val)
    ece_before = expected_calibration_error(probs_before, y_val)

    acc = (head.predict(Z_val) == y_val).mean()
    assert acc < 0.98  # sanity: task is not perfectly separable, calibration is meaningful

    t = head.calibrate_temperature(Z_val, y_val)
    assert t > 0.0
    probs_after = head.predict_proba(Z_val)
    ece_after = expected_calibration_error(probs_after, y_val)

    assert ece_after < ece_before


def test_calibrate_temperature_requires_fit_first():
    head = AdaptiveMahalanobisETFChoiceHead()
    with pytest.raises(RuntimeError):
        head.calibrate_temperature(np.zeros((5, 3)), np.array([0, 1, 0, 1, 0]))


def test_scores_matches_existing_head_scores_convention():
    """`scores(X).argmax(axis=1)` is the calling convention used by
    `spec21_advanced_heads.LedoitWolfLDAHead` / `BBPAdaptiveProbe` and dispatched on by
    `evaluate_spec21_scorecard.py` / `evaluate_dual_70b_72b_advanced_ensemble.py`; this head
    must support the same convention to be a real (not zero-call) alternative there.
    """
    rng = np.random.default_rng(7501)
    Z, y = _make_classification_data(rng, 30, 64, 4)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=0.5)
    scores = head.scores(Z)
    assert scores.shape == (Z.shape[0], 4)
    np.testing.assert_array_equal(scores.argmax(axis=1), head.predict(Z))
    # softmax(scores) must reproduce predict_proba exactly: scores are the same logits.
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    softmax_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(softmax_scores, head.predict_proba(Z), atol=1e-10)


def test_predict_proba_requires_fit_first():
    head = AdaptiveMahalanobisETFChoiceHead()
    with pytest.raises(RuntimeError):
        head.predict_proba(np.zeros((5, 3)))


# --------------------------------------------------------------------------- Fail-Closed: fit inputs

def test_fit_rejects_nan_in_features():
    rng = np.random.default_rng(8001)
    Z, y = _make_classification_data(rng, 20, 32, 3)
    Z[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z, y)


def test_fit_rejects_inf_in_features():
    rng = np.random.default_rng(8002)
    Z, y = _make_classification_data(rng, 20, 32, 3)
    Z[3, 5] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z, y)


def test_fit_rejects_n_less_than_k():
    # K is inferred as max(y) + 1 (fit() takes no explicit num_classes), so a genuine N < K
    # case needs a sparse label set: y only touches classes {0, 4}, implying K=5, with just
    # N=2 rows -- far fewer rows than the K the labels imply.
    rng = np.random.default_rng(8003)
    d = 16
    Z = rng.normal(size=(2, d))
    y = np.array([0, 4])
    with pytest.raises(ValueError, match=r"N=2 samples but K=5 classes"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z, y)


def test_fit_rejects_single_sample_class():
    rng = np.random.default_rng(8004)
    Z, y = _make_classification_data(rng, 20, 32, 3)
    # Collapse class 1 down to a single row.
    keep = np.concatenate([
        np.flatnonzero(y == 0),
        np.flatnonzero(y == 1)[:1],
        np.flatnonzero(y == 2),
    ])
    Z2, y2 = Z[keep], y[keep]
    with pytest.raises(ValueError, match="fewer than 2 training rows"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z2, y2)


def test_fit_rejects_missing_class():
    rng = np.random.default_rng(8005)
    Z, y = _make_classification_data(rng, 20, 32, 3)
    keep = y != 1
    Z2, y2 = Z[keep], y[keep]  # labels are now {0, 2}, so class 1 has zero rows
    with pytest.raises(ValueError, match="no training rows"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z2, y2)


def test_fit_rejects_negative_lambda():
    rng = np.random.default_rng(8006)
    Z, y = _make_classification_data(rng, 20, 32, 3)
    with pytest.raises(ValueError, match="lambda_etf"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=-1.0)


def test_fit_rejects_mismatched_candidate_reps_shape():
    rng = np.random.default_rng(8007)
    Z, y = _make_classification_data(rng, 20, 32, 3)
    bad_reps = rng.normal(size=(4, 32))  # 4 rows, but K=3
    with pytest.raises(ValueError, match="candidate_reps"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z, y, candidate_reps=bad_reps)


def test_predict_proba_rejects_dimension_mismatch():
    rng = np.random.default_rng(8008)
    Z, y = _make_classification_data(rng, 20, 32, 3)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y)
    with pytest.raises(ValueError):
        head.predict_proba(rng.normal(size=(5, 31)))


# --------------------------------------------------------------------------- ordinal task gate

@pytest.mark.parametrize("task_name", sorted(ORDINAL_TASK_NAMES))
def test_ordinal_task_gate_blocks_construction(task_name):
    with pytest.raises(ValueError, match="ordinal"):
        AdaptiveMahalanobisETFChoiceHead(task_name=task_name)


def test_non_ordinal_task_name_is_allowed():
    head = AdaptiveMahalanobisETFChoiceHead(task_name="boolq")
    assert head.task_name == "boolq"


# --------------------------------------------------------------------------- ECE helper

def test_expected_calibration_error_perfect_calibration_is_zero():
    rng = np.random.default_rng(9001)
    n = 200
    y = rng.integers(0, 2, size=n)
    probs = np.zeros((n, 2))
    probs[np.arange(n), y] = 1.0
    ece = expected_calibration_error(probs, y)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_shape_mismatch_raises():
    with pytest.raises(ValueError):
        expected_calibration_error(np.zeros((5, 2)), np.zeros(4, dtype=int))


@pytest.mark.parametrize("lambda_etf", [0.0, 0.5])
def test_large_coordinate_shift_preserves_distances(lambda_etf):
    Z, y = _make_classification_data(np.random.default_rng(10001), 20, 8, 3,
                                     scale=0.7)
    base = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=lambda_etf)
    shifted = AdaptiveMahalanobisETFChoiceHead().fit(Z + 1e10, y, lambda_etf=lambda_etf)
    np.testing.assert_allclose(shifted._squared_distances(Z + 1e10),
                               base._squared_distances(Z), atol=1e-3, rtol=1e-4)
    np.testing.assert_allclose(shifted.predict_proba(Z + 1e10),
                               base.predict_proba(Z), atol=1e-4)


@pytest.mark.parametrize("magnitude", [1e200, np.finfo(float).max])
def test_extreme_finite_queries_return_uniform_probabilities(magnitude):
    Z, y = _make_classification_data(np.random.default_rng(10002), 20, 8, 3)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y)
    queries = np.vstack([Z[:1], np.full((2, 8), magnitude)])
    with np.errstate(all="raise"):
        probs = head.predict_proba(queries)
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs[1:], 1.0 / 3)
    np.testing.assert_allclose(probs[:1], head.predict_proba(Z[:1]))


def test_chunked_distances_match_individual_queries():
    Z, y = _make_classification_data(np.random.default_rng(10003), 20, 4, 3)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y)
    queries = np.tile(Z[:1], (5001, 1))
    np.testing.assert_allclose(head._squared_distances(queries),
                               np.repeat(head._squared_distances(Z[:1]), 5001, axis=0))


@pytest.mark.parametrize("already_fitted", [False, True])
@pytest.mark.parametrize("failure", ["coincident", "rank"])
def test_failed_fit_preserves_entire_previous_state(already_fitted, failure):
    Z, y = _make_classification_data(np.random.default_rng(10004), 20, 8, 3)
    head = AdaptiveMahalanobisETFChoiceHead()
    if already_fitted:
        head.fit(Z, y)
        head.temperature_ = 2.5
        before_probs = head.predict_proba(Z)
    before = head.__dict__.copy()
    new_Z, new_y = _make_classification_data(np.random.default_rng(10005), 20, 4, 3)
    reps = np.zeros((3, 4))
    if failure == "rank":
        reps[:, 0] = [0, 1, 3]  # rank one cannot reconstruct a three-class ETF
    with pytest.raises(ValueError, match="across-class mean|Gram reconstruction"):
        head.fit(new_Z, new_y, candidate_reps=reps, lambda_etf=1.0)
    assert head.__dict__.keys() == before.keys()
    for key, value in before.items():
        assert head.__dict__[key] is value
    if already_fitted:
        np.testing.assert_array_equal(head.predict_proba(Z), before_probs)


@pytest.mark.parametrize("bad_input", ["features", "candidates", "labels_complex",
                                       "labels_bool", "labels_string", "labels_object"])
def test_fit_rejects_unsupported_input_types(bad_input):
    Z, y = _make_classification_data(np.random.default_rng(10006), 20, 8, 3)
    kwargs = {}
    if bad_input == "features":
        Z = Z.astype(complex)
    elif bad_input == "candidates":
        kwargs["candidate_reps"] = np.ones((3, 8), dtype=complex)
    else:
        dtype = {"labels_complex": complex, "labels_bool": bool,
                 "labels_string": str, "labels_object": object}[bad_input]
        y = y.astype(dtype)
    with pytest.raises(ValueError, match="not supported"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z, y, **kwargs)


@pytest.mark.parametrize("value", [True, np.bool_(True), "1", 1j, np.nan, np.inf])
def test_fit_rejects_invalid_lambda_types_and_values(value):
    Z, y = _make_classification_data(np.random.default_rng(10007), 20, 8, 3)
    with pytest.raises(ValueError, match="lambda_etf"):
        AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=value)


def test_fit_accepts_numpy_integer_lambda():
    Z, y = _make_classification_data(np.random.default_rng(10008), 20, 8, 3)
    head = AdaptiveMahalanobisETFChoiceHead().fit(Z, y, lambda_etf=np.int64(1))
    assert head.lambda_etf_ == 1.0


@pytest.mark.parametrize("task_name", ["HelpSteer2", " HELPSTEER2 ",
                                       " Summeval_Relevance ", "SUMMEVAL_CONSISTENCY"])
def test_ordinal_task_gate_normalizes_name(task_name):
    with pytest.raises(ValueError, match="ordinal"):
        AdaptiveMahalanobisETFChoiceHead(task_name=task_name)


@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True])
def test_ece_rejects_invalid_bin_count(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(np.array([[0.2, 0.8]]), np.array([1]), n_bins=n_bins)


@pytest.mark.parametrize("probs, labels", [
    ([[1.2, -0.2]], [0]),
    ([[0.2, 0.2]], [0]),
    ([[np.nan, 0.5]], [0]),
    ([[np.inf, 0.5]], [0]),
    ([0.2, 0.8], [1]),
    (np.empty((0, 2)), []),
    (np.empty((1, 0)), [0]),
    ([[0.2, 0.8]], [2]),
    ([[0.2, 0.8]], [-1]),
    ([[0.2, 0.8]], [0, 1]),
])
def test_ece_rejects_invalid_probabilities_and_labels(probs, labels):
    with pytest.raises(ValueError):
        expected_calibration_error(probs, labels)
