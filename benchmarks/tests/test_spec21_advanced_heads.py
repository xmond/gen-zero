"""Spec 21 phase 1 (benchmarks/suites/spec21_advanced_heads.py). Covers:
  1. priors + prior-collapse gate: below-majority-prior fake highs are caught even when the
     prediction distribution is NOT concentrated; train-prior vs eval-label prior semantics;
  2. logit adjustment (Menon et al. 2021): identity at tau=0, bias folding == logit adjusting,
     90:10 data: minority recall up, majority bias down (external sklearn head and our LDA head);
  3. Gavish-Donoho threshold (unknown sigma): omega(beta) against independent references
     (known beta=1 constant, Monte-Carlo MP median), exact rank recovery on spiked-covariance
     data for beta<1 and beta>1, scale invariance in sigma, no-signal refusal;
  4. BBPAdaptiveProbe: fold to one (W, b) equals the explicit project-then-LR path;
  5. LedoitWolfLDAHead: rho matches sklearn's ledoit_wolf_shrinkage, covariance is PD while the
     sample covariance is singular (D > n), folded scores equal explicit Mahalanobis
     discrimination and sklearn's lsqr LDA at the same rho;
  6. SplitConformalPredictor: exact order-statistic quantile, empirical coverage over many random
     splits, abstain semantics (|C| = 0 and |C| > 1), trivial q_hat when n is too small.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES))

import spec21_advanced_heads as s21  # noqa: E402


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- priors + gate

def test_class_priors_sum_to_one_and_keep_absent_class_at_zero():
    y = np.array([0, 0, 0, 1, 3])
    pi = s21.compute_class_priors(y, 4)
    np.testing.assert_allclose(pi, [0.6, 0.2, 0.0, 0.2])
    assert pi.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("y,k", [(np.array([0, 1, 4]), 4), (np.array([-1, 0]), 2),
                                 (np.array([], dtype=int), 2), (np.array([0.5, 1.0]), 2)])
def test_class_priors_reject_bad_labels(y, k):
    with pytest.raises(ValueError):
        s21.compute_class_priors(y, k)


def test_gate_catches_below_majority_prior_without_concentrated_predictions():
    # helpsteer2-like failure: predictions are spread over all classes (max frac 0.35, no
    # "collapse") yet accuracy 0.30 < majority prior 0.5. The old 0.95 gate cannot see this.
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 500 + [1] * 250 + [2] * 250)
    y_pred = rng.integers(0, 3, size=1000)
    g = s21.check_prior_collapse_gate(y_true, y_pred)
    assert g["max_pred_class_frac"] < 0.95 and g["collapsed"] is False
    assert g["accuracy"] < g["majority_prior"] == pytest.approx(0.5)
    assert g["below_prior"] is True and g["passed"] is False


def test_gate_flags_constant_predictor_as_collapsed_and_not_beating_prior():
    y_true = np.array([0] * 92 + [1] * 8)
    g = s21.check_prior_collapse_gate(y_true, np.zeros(100, dtype=int))
    assert g["collapsed"] is True and g["passed"] is False
    assert g["accuracy"] == pytest.approx(0.92) == pytest.approx(g["majority_prior"])
    assert g["balanced_accuracy"] == pytest.approx(0.5)


def test_gate_passes_genuinely_better_head():
    y_true = np.array([0] * 60 + [1] * 40)
    y_pred = y_true.copy()
    y_pred[:5] = 1
    g = s21.check_prior_collapse_gate(y_true, y_pred)
    assert g["passed"] is True and g["below_prior"] is False and g["collapsed"] is False
    assert g["prior_source"] == "eval_labels"


def test_gate_uses_train_prior_when_given_and_it_differs_from_eval_prior():
    # eval set is balanced (eval majority 0.5), but the TRAIN majority is 0.9. Accuracy 0.85 beats
    # the eval prior yet is below what "always predict the train majority" would score on train.
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = y_true.copy()
    y_pred[:15] = 1 - y_pred[:15]
    assert s21.check_prior_collapse_gate(y_true, y_pred)["below_prior"] is False
    g = s21.check_prior_collapse_gate(y_true, y_pred, train_priors=np.array([0.9, 0.1]))
    assert g["prior_source"] == "train_priors"
    assert g["majority_prior"] == pytest.approx(0.9)
    assert g["accuracy"] == pytest.approx(0.85)
    assert g["below_prior"] is True and g["passed"] is False


def test_gate_counts_abstain_as_wrong_and_validates_input():
    y_true = np.array([0, 1, 1, 1])
    g = s21.check_prior_collapse_gate(y_true, np.array([0, -1, -1, 1]))
    assert g["accuracy"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        s21.check_prior_collapse_gate(y_true, np.array([0, 1]))
    with pytest.raises(ValueError):
        s21.check_prior_collapse_gate(np.array([], dtype=int), np.array([], dtype=int))
    with pytest.raises(ValueError):
        s21.check_prior_collapse_gate(y_true, y_true, train_priors=np.array([0.5, 0.6]))


# --------------------------------------------------------------------------- logit adjustment

def _imbalanced_gaussians(rng, n_major, n_minor, d=4, sep=1.5):
    mu = np.zeros(d)
    mu[0] = sep
    X = np.vstack([rng.standard_normal((n_major, d)), rng.standard_normal((n_minor, d)) + mu])
    y = np.array([0] * n_major + [1] * n_minor)
    return X, y


def test_logit_adjust_tau_zero_is_identity_and_formula_is_exact():
    logits = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]])
    pi = np.array([0.7, 0.2, 0.1])
    np.testing.assert_array_equal(s21.logit_adjust(logits, pi, tau=0.0), logits)
    out = s21.logit_adjust(logits, pi, tau=1.5, eps=1e-12)
    np.testing.assert_allclose(out, logits - 1.5 * np.log(pi + 1e-12))
    assert not np.shares_memory(out, logits)


def test_fold_logit_adjustment_into_bias_equals_adjusting_logits():
    rng = np.random.default_rng(1)
    W, b, X = rng.standard_normal((3, 6)), rng.standard_normal(3), rng.standard_normal((10, 6))
    pi = np.array([0.6, 0.3, 0.1])
    W2, b2 = s21.fold_logit_adjustment(W, b, pi, tau=0.8)
    np.testing.assert_array_equal(W2, W)
    np.testing.assert_allclose(X @ W2.T + b2, s21.logit_adjust(X @ W.T + b, pi, tau=0.8), atol=1e-12)
    assert not np.shares_memory(b2, b)


def test_logit_adjust_rejects_bad_shapes_and_tau():
    with pytest.raises(ValueError):
        s21.logit_adjust(np.zeros((2, 3)), np.array([0.5, 0.5]), tau=1.0)
    with pytest.raises(ValueError):
        s21.logit_adjust(np.zeros((2, 2)), np.array([0.5, 0.5]), tau=-1.0)


def test_logit_adjustment_lifts_minority_recall_and_cuts_majority_bias_on_90_10():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(7)
    Xtr, ytr = _imbalanced_gaussians(rng, 9000, 1000)
    Xte, yte = _imbalanced_gaussians(rng, 5000, 5000)      # balanced test: balanced-error target
    pi = s21.compute_class_priors(ytr, 2)
    clf = LogisticRegression(C=1e4, max_iter=2000).fit(Xtr, ytr)   # external head, weak reg
    logits = clf.decision_function(Xte)
    logits = np.stack([-logits / 2, logits / 2], 1)               # 2-class logit form
    base = logits.argmax(1)
    adj = s21.logit_adjust(logits, pi, tau=1.0).argmax(1)

    def recall1(p): return float((p[yte == 1] == 1).mean())
    def maj_frac(p): return float((p == 0).mean())
    def bal(p): return 0.5 * (recall1(p) + float((p[yte == 0] == 0).mean()))

    assert recall1(adj) > recall1(base) + 0.15
    assert maj_frac(base) > 0.6 and maj_frac(adj) < maj_frac(base) - 0.1
    assert bal(adj) > bal(base) + 0.05
    # Menon et al. tradeoff: majority recall is paid for it, not free.
    assert float((adj[yte == 0] == 0).mean()) < float((base[yte == 0] == 0).mean())


def test_logit_adjustment_on_ledoit_wolf_lda_head_lifts_minority_recall_and_lowers_majority_share():
    rng = np.random.default_rng(8)
    Xtr, ytr = _imbalanced_gaussians(rng, 9000, 1000, d=6)
    Xte, yte = _imbalanced_gaussians(rng, 4000, 4000, d=6)
    head = s21.LedoitWolfLDAHead.fit(Xtr, ytr, 2, standardize=False)
    base = head.predict(Xte)
    W, b = s21.fold_logit_adjustment(head.W_fold, head.b_fold, head.priors, tau=1.0)
    adj = (Xte @ W.T + b).argmax(1)
    assert (adj[yte == 1] == 1).mean() > (base[yte == 1] == 1).mean() + 0.1
    assert (adj == 0).mean() < (base == 0).mean()


# --------------------------------------------------------------------------- Gavish-Donoho

def test_gd_omega_matches_known_constant_and_polynomial_approximation():
    assert s21.gavish_donoho_omega(1.0) == pytest.approx(2.858, abs=2e-3)       # GD 2014, Fig./Sec. 5
    for beta in (0.1, 0.25, 0.5, 0.75, 1.0):
        poly = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43          # GD 2014 eq. (5)
        assert s21.gavish_donoho_omega(beta) == pytest.approx(poly, rel=0.02)


def test_gd_omega_matches_montecarlo_mp_median():
    # independent route: median eigenvalue of a big Wishart matrix estimates the MP median mu_beta.
    rng = np.random.default_rng(3)
    m, n = 600, 1200
    beta = m / n
    Y = rng.standard_normal((m, n))
    mu_mc = float(np.median(np.linalg.eigvalsh(Y @ Y.T / n)))
    lam_star = math.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) + math.sqrt(beta ** 2 + 14 * beta + 1)))
    assert s21.gavish_donoho_omega(beta) == pytest.approx(lam_star / math.sqrt(mu_mc), rel=0.01)


@pytest.mark.parametrize("beta", [0.0, -0.1, 1.5, float("nan")])
def test_gd_omega_domain_errors(beta):
    with pytest.raises(ValueError):
        s21.gavish_donoho_omega(beta)


def _spiked(rng, n, d, r, sigma, strength):
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V, _ = np.linalg.qr(rng.standard_normal((d, r)))
    sv = strength * sigma * (math.sqrt(n) + math.sqrt(d)) * np.linspace(1.0, 2.0, r)
    return (U * sv) @ V.T + sigma * rng.standard_normal((n, d))


@pytest.mark.parametrize("n,d", [(400, 1000), (1000, 400), (600, 600)])
def test_gd_recovers_true_rank_and_cuts_white_noise_spiked_model(n, d):
    rng = np.random.default_rng(11)
    ranks = []
    for r in (3, 8, 15):
        X = _spiked(rng, n, d, r, sigma=1.0, strength=1.0)
        est = s21.estimate_gd_rank(X, standardize=False, center=False)
        ranks.append(est["rank"])
        # every retained value is signal, every noise value is cut
        assert est["rank"] == r
        assert est["tau"] == pytest.approx(est["omega"] * est["y_med"])
        assert est["beta"] == pytest.approx(min(n, d) / max(n, d))
        assert est["n_above_tau"] == r
    assert ranks == [3, 8, 15]


def test_gd_rank_is_invariant_to_unknown_noise_scale():
    rng = np.random.default_rng(12)
    base = _spiked(rng, 500, 800, 6, sigma=1.0, strength=1.0)
    r1 = s21.estimate_gd_rank(base, standardize=False, center=False)["rank"]
    r2 = s21.estimate_gd_rank(base * 37.0, standardize=False, center=False)["rank"]
    assert r1 == r2 == 6


def test_gd_pure_noise_keeps_nothing_and_fit_refuses_zero_rank():
    rng = np.random.default_rng(13)
    X = rng.standard_normal((400, 900))
    est = s21.estimate_gd_rank(X, standardize=False, center=False)
    assert est["rank"] == 0 and est["n_singular_values"] == 400
    with pytest.raises(ValueError, match="no singular value"):
        s21.BBPAdaptiveProbe.fit(X, rng.integers(0, 2, size=400), 2, standardize=False)


def test_gd_threshold_uses_centered_effective_sample_count():
    rng = np.random.default_rng(14)
    X = rng.standard_normal((50, 200)) + 5.0                   # big mean offset must not become "signal"
    est = s21.estimate_gd_rank(X, standardize=True, center=True)
    assert est["rank"] <= 1
    assert est["n_effective"] == 49


# --------------------------------------------------------------------------- BBPAdaptiveProbe

def test_bbp_probe_selects_rank_without_grid_and_folds_to_single_gemv():
    k = 4
    # train and test share one class-mean basis V
    V, _ = np.linalg.qr(np.random.default_rng(23).standard_normal((1000, k)))

    def make(seed, n):
        r = np.random.default_rng(seed)
        y = np.arange(n) % k
        r.shuffle(y)
        return 6.0 * np.eye(k)[y] @ V.T + r.standard_normal((n, 1000)), y
    Xtr, ytr = make(31, 400)
    Xte, yte = make(32, 400)
    probe = s21.BBPAdaptiveProbe.fit(Xtr, ytr, k)             # C chosen by inner CV
    assert probe.rank == k - 1                                # centred class means span k-1 dims
    assert probe.C_ in s21.BBP_C_GRID
    assert probe.W_fold.shape == (k, 1000) and probe.b_fold.shape == (k,)
    pred = probe.predict(Xte)
    assert (pred == yte).mean() > 0.9
    # folded single GEMV == explicit standardize -> project -> linear head
    Z = (Xte - probe.mean_) / probe.scale_
    explicit = (Z @ probe.components_.T) @ probe.coef_.T + probe.intercept_
    np.testing.assert_allclose(probe.scores(Xte), explicit, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(Xte @ probe.W_fold.T + probe.b_fold, probe.scores(Xte), atol=1e-12)


def test_bbp_probe_binary_fold_keeps_sigmoid_probabilities():
    rng = np.random.default_rng(24)
    V, _ = np.linalg.qr(rng.standard_normal((300, 2)))
    y = np.arange(300) % 2
    X = 5.0 * np.eye(2)[y] @ V.T + rng.standard_normal((300, 300))
    probe = s21.BBPAdaptiveProbe.fit(X, y, 2, C=1.0)
    sc = probe.scores(X)
    assert sc.shape == (300, 2)
    Z = (X - probe.mean_) / probe.scale_
    margin = (Z @ probe.components_.T) @ probe.coef_binary_.ravel() + probe.intercept_binary_
    np.testing.assert_allclose(sc[:, 1] - sc[:, 0], margin, atol=1e-8)


def test_bbp_probe_rejects_missing_class_and_bad_shapes():
    rng = np.random.default_rng(25)
    X = rng.standard_normal((30, 60))
    with pytest.raises(ValueError):
        s21.BBPAdaptiveProbe.fit(X, np.zeros(30, dtype=int), 2)
    with pytest.raises(ValueError):
        s21.BBPAdaptiveProbe.fit(X, np.zeros(29, dtype=int), 2)
    with pytest.raises(ValueError):
        s21.BBPAdaptiveProbe.fit(np.full((30, 60), np.nan), np.arange(30) % 2, 2)


# --------------------------------------------------------------------------- Ledoit-Wolf LDA

def _lda_data(rng, n, d, k, sep=1.2):
    y = np.arange(n) % k
    rng.shuffle(y)
    means = sep * rng.standard_normal((k, d)) / math.sqrt(d) * 3
    A = rng.standard_normal((d, d)) / math.sqrt(d)
    X = means[y] + rng.standard_normal((n, d)) @ (A + np.eye(d))
    return X, y


def test_lw_rho_matches_sklearn_reference_and_lies_in_unit_interval():
    from sklearn.covariance import ledoit_wolf_shrinkage
    rng = np.random.default_rng(41)
    X, y = _lda_data(rng, 90, 40, 3)
    head = s21.LedoitWolfLDAHead.fit(X, y, 3, standardize=False)
    Xc = X - np.stack([X[y == c].mean(0) for c in range(3)])[y]
    assert 0.0 < head.rho <= 1.0
    assert head.rho == pytest.approx(ledoit_wolf_shrinkage(Xc, assume_centered=True), rel=1e-9)


def test_lw_covariance_positive_definite_when_D_exceeds_n_while_sample_cov_is_singular():
    rng = np.random.default_rng(42)
    X, y = _lda_data(rng, 60, 150, 3)                          # D=150 > n=60
    head = s21.LedoitWolfLDAHead.fit(X, y, 3, standardize=False)
    Sigma = head.covariance_matrix()
    Xc = X - np.stack([X[y == c].mean(0) for c in range(3)])[y]
    S = Xc.T @ Xc / len(X)
    assert np.linalg.eigvalsh(S)[0] < 1e-9                     # sample covariance is singular
    ev = np.linalg.eigvalsh(Sigma)
    assert ev[0] > 0 and np.isfinite(np.linalg.cond(Sigma))
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)
    # unit interval and the analytic form (1-rho) S + rho nu I with nu = tr(S)/D
    nu = np.trace(S) / S.shape[0]
    np.testing.assert_allclose(Sigma, (1 - head.rho) * S + head.rho * nu * np.eye(150), atol=1e-10)


@pytest.mark.parametrize("n,d,standardize", [(60, 150, False), (60, 150, True), (200, 30, False), (200, 30, True)])
def test_lw_folded_scores_equal_explicit_mahalanobis_discriminant(n, d, standardize):
    rng = np.random.default_rng(43)
    X, y = _lda_data(rng, n, d, 4)
    Xte, _ = _lda_data(np.random.default_rng(44), 50, d, 4)
    head = s21.LedoitWolfLDAHead.fit(X, y, 4, standardize=standardize)
    # explicit reference: dense Sigma^{-1} in the (optionally standardized) feature space
    if standardize:
        m, sd = X.mean(0), X.std(0)
    else:
        m, sd = np.zeros(d), np.ones(d)
    Z, Zte = (X - m) / sd, (Xte - m) / sd
    mu = np.stack([Z[y == c].mean(0) for c in range(4)])
    pi = np.bincount(y, minlength=4) / n
    Zc = Z - mu[y]
    S = Zc.T @ Zc / n
    nu = np.trace(S) / d
    Sigma = (1 - head.rho) * S + head.rho * nu * np.eye(d)
    P = np.linalg.inv(Sigma)
    maha = np.stack([-0.5 * np.einsum("nd,de,ne->n", Zte - mu[k], P, Zte - mu[k]) + np.log(pi[k])
                     for k in range(4)], 1)
    got = head.scores(Xte)
    # equal up to the class-independent term -0.5 z^T P z  ->  compare after per-row centering
    ref_c = maha - maha.mean(1, keepdims=True)
    got_c = got - got.mean(1, keepdims=True)
    np.testing.assert_allclose(got_c, ref_c, rtol=1e-7, atol=1e-7)
    assert (got.argmax(1) == maha.argmax(1)).all()
    # and the folded operator is literally one matmul
    np.testing.assert_allclose(Xte @ head.W_fold.T + head.b_fold, got, atol=1e-12)


def test_lw_head_equals_sklearn_lsqr_lda_at_same_shrinkage():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.default_rng(45)
    X, y = _lda_data(rng, 80, 120, 3)
    Xte, _ = _lda_data(np.random.default_rng(46), 40, 120, 3)
    head = s21.LedoitWolfLDAHead.fit(X, y, 3, standardize=False)
    ref = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=float(head.rho)).fit(X, y)
    np.testing.assert_allclose(head.scores(Xte), ref.decision_function(Xte), rtol=1e-6, atol=1e-6)


def test_lw_head_beats_chance_in_D_gt_n_regime_and_rejects_bad_input():
    # class means must be shared, so train and test come from one draw
    Xall, yall = _lda_data(np.random.default_rng(49), 420, 400, 3, sep=2.0)
    head = s21.LedoitWolfLDAHead.fit(Xall[:120], yall[:120], 3)
    assert (head.predict(Xall[120:]) == yall[120:]).mean() > 0.5      # chance 1/3
    with pytest.raises(ValueError):
        s21.LedoitWolfLDAHead.fit(Xall[:120], np.zeros(120, dtype=int), 3)
    with pytest.raises(ValueError):
        s21.LedoitWolfLDAHead.fit(np.zeros((6, 4)), np.arange(6) % 2, 2, standardize=False)  # zero variance


# --------------------------------------------------------------------------- split conformal

def test_conformal_quantile_is_exact_order_statistic():
    # scores 1-p_y engineered as 0.05..0.45 (n=9): alpha=0.1 -> k=ceil(10*0.9)=9 -> the max score.
    n = 9
    p_true = np.linspace(0.95, 0.55, n)
    logits = np.log(np.stack([p_true, 1 - p_true], 1))
    cp = s21.SplitConformalPredictor().calibrate(logits, np.zeros(n, dtype=int), alpha=0.1)
    assert cp.q_hat == pytest.approx(1 - p_true.min())
    assert cp.k_index == 9 and cp.n_cal == 9
    cp = s21.SplitConformalPredictor().calibrate(logits, np.zeros(n, dtype=int), alpha=0.5)
    assert cp.k_index == 5                                    # ceil(10*0.5)
    assert cp.q_hat == pytest.approx(np.sort(1 - p_true)[4])


def test_conformal_trivial_quantile_when_calibration_set_too_small():
    logits = np.log(np.array([[0.9, 0.1]] * 5))
    cp = s21.SplitConformalPredictor().calibrate(logits, np.zeros(5, dtype=int), alpha=0.1)
    assert cp.k_index == 6 > cp.n_cal and math.isinf(cp.q_hat) and cp.is_trivial
    assert cp.predict_set(np.log(np.array([[0.99, 0.01]]))) == [[0, 1]]   # never claims a false guarantee


def test_conformal_set_definition_and_abstain_semantics():
    cal = np.log(np.array([[0.9, 0.1]] * 19 + [[0.5, 0.5]]))              # scores 0.1 x19, 0.5 x1
    cp = s21.SplitConformalPredictor().calibrate(cal, np.zeros(20, dtype=int), alpha=0.1)
    assert cp.k_index == math.ceil(21 * 0.9) == 19 and cp.q_hat == pytest.approx(0.1)
    test = np.log(np.array([
        [0.95, 0.05],     # p0 >= 0.9 only            -> {0}
        [0.05, 0.95],     # p1 >= 0.9 only            -> {1}
        [0.6, 0.4],       # nobody >= 0.9             -> {}   (empty)
    ]))
    sets = cp.predict_set(test)
    assert sets == [[0], [1], []]
    pred, size = cp.predict_with_abstain(test)
    np.testing.assert_array_equal(pred, [0, 1, -1])
    np.testing.assert_array_equal(size, [1, 1, 0])
    # ambiguous set (|C| > 1): threshold 1-q_hat = 0.4 admits both classes of a 0.5/0.5 row
    cp2 = s21.SplitConformalPredictor().calibrate(
        np.log(np.array([[0.4, 0.6]] * 19 + [[0.9, 0.1]])), np.zeros(20, dtype=int), alpha=0.1)
    pred2, size2 = cp2.predict_with_abstain(np.log(np.array([[0.5, 0.5], [0.99, 0.01]])))
    assert cp2.predict_set(np.log(np.array([[0.5, 0.5]]))) == [[0, 1]]
    np.testing.assert_array_equal(size2, [2, 1])
    np.testing.assert_array_equal(pred2, [-1, 0])


def test_conformal_requires_calibration_and_validates_input():
    cp = s21.SplitConformalPredictor()
    with pytest.raises(RuntimeError):
        cp.predict_set(np.zeros((1, 2)))
    for alpha in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            cp.calibrate(np.zeros((4, 2)), np.zeros(4, dtype=int), alpha=alpha)
    with pytest.raises(ValueError):
        cp.calibrate(np.zeros((4, 2)), np.array([0, 1, 2, 0]))                # label out of range
    with pytest.raises(ValueError):
        cp.calibrate(np.array([[np.nan, 0.0]] * 4), np.zeros(4, dtype=int))
    cp.calibrate(np.zeros((30, 3)), np.zeros(30, dtype=int))
    with pytest.raises(ValueError):
        cp.predict_set(np.zeros((2, 4)))                                      # wrong K


def _pool(rng, n, k, sharp):
    """Exchangeable pool: labels drawn from the TRUE softmax; model logits scaled by `sharp`
    (sharp != 1 -> deliberately miscalibrated, coverage must still hold: distribution-free)."""
    z = 2.0 * rng.standard_normal((n, k))
    y = np.array([rng.choice(k, p=p) for p in _softmax(z)])
    return sharp * z, y


@pytest.mark.parametrize("sharp,alpha", [(1.0, 0.1), (3.0, 0.1), (0.3, 0.1), (1.0, 0.2)])
def test_conformal_empirical_coverage_over_random_splits(sharp, alpha):
    rng = np.random.default_rng(51)
    logits, y = _pool(rng, 4000, 5, sharp)
    n_cal, trials, eps = 200, 300, 0.01
    covs = []
    for t in range(trials):
        perm = np.random.default_rng(1000 + t).permutation(len(y))
        ci, ti = perm[:n_cal], perm[n_cal:]
        cp = s21.SplitConformalPredictor().calibrate(logits[ci], y[ci], alpha=alpha)
        sets = cp.predict_set(logits[ti])
        covs.append(np.mean([y[i] in s for i, s in zip(ti, sets)]))
    mean_cov = float(np.mean(covs))
    assert mean_cov >= 1 - alpha - eps, mean_cov
    assert mean_cov <= 1 - alpha + 1.0 / (n_cal + 1) + eps              # Lei et al. 2018 upper bound


def test_conformal_abstained_predictions_are_more_accurate_than_unfiltered():
    rng = np.random.default_rng(52)
    logits, y = _pool(rng, 6000, 4, 1.0)
    cp = s21.SplitConformalPredictor().calibrate(logits[:1000], y[:1000], alpha=0.1)
    pred, size = cp.predict_with_abstain(logits[1000:])
    yt = y[1000:]
    kept = pred != -1
    assert 0.05 < kept.mean() < 1.0
    assert (pred[kept] == yt[kept]).mean() > (logits[1000:].argmax(1) == yt).mean()
    np.testing.assert_array_equal(kept, size == 1)


def test_conformal_predict_set_matches_direct_probability_threshold():
    rng = np.random.default_rng(53)
    logits, y = _pool(rng, 800, 6, 1.0)
    cp = s21.SplitConformalPredictor().calibrate(logits[:300], y[:300], alpha=0.15)
    P = _softmax(logits[300:])
    ref = [np.flatnonzero(row >= 1 - cp.q_hat).tolist() for row in P]
    assert cp.predict_set(logits[300:]) == ref
