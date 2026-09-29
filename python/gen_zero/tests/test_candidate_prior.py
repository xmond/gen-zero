"""Tests for CandidateSemanticPrior: ESZSL closed-form bilinear compatibility and Bayesian
prototype shrinkage over candidate-text embeddings.

Verifies:
1. `compute_w0_prior`'s closed-form A satisfies the exact first-order (gradient = 0) condition
   of the full 4-term ridge objective, and that the objective is not lower at any perturbation
   (convexity check), so the closed form really is the minimizer, not just a plausible formula.
2. `score_candidates` on genuinely-unseen candidate embeddings (never used to fit A) matches the
   raw `Z A E_new^T` formula and recovers the correct class on a synthetic zero-shot problem.
3. `compute_prototype_prior`'s two limits: n_k = 0 collapses exactly to the semantic prototype
   `A @ E[k]`; n_k -> large collapses to the empirical mean.
4. `shuffle_control` actually permutes (never silently returns the identity), and an ablation
   showing the zero-shot accuracy gain comes from real candidate semantics, not merely from the
   ridge regularization shared by both the real and shuffled fits.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gen_zero.manifold import CandidateSemanticPrior


def _make_candidate_embeddings(rng, k, q):
    """k x q orthonormal rows (k must equal q here), so every pair of candidates is maximally
    separated in semantic space and classification accuracy isolates the fit quality rather
    than an unrelated embedding-geometry confound.
    """
    raw = rng.normal(size=(q, q))
    orthonormal, _ = np.linalg.qr(raw)
    return orthonormal


def _synthetic_zero_shot_problem(rng, E, A_true, k=6, d=8, n_per_class=60, noise=0.15, signal_scale=3.0):
    """Z rows cluster around signal_scale * A_true @ E[k] for their class k, so real E carries
    the exact signal a correctly-fit A should recover; a class-index-scrambled E carries none.
    """
    true_prototypes = signal_scale * (E @ A_true.T)  # k x d
    Z, y = [], []
    for cls in range(k):
        Z.append(true_prototypes[cls] + noise * rng.normal(size=(n_per_class, d)))
        y.append(np.full(n_per_class, cls))
    Z = np.concatenate(Z, axis=0)
    y = np.concatenate(y, axis=0)
    Y = np.eye(k)[y]
    return Z, Y, y, true_prototypes


# ---------------------------------------------------------------------------
# 1. Closed-form first-order condition + convexity
# ---------------------------------------------------------------------------

def test_w0_prior_satisfies_first_order_condition():
    rng = np.random.default_rng(0)
    n, d, k, q = 50, 6, 5, 4
    Z = rng.normal(size=(n, d))
    Y = rng.normal(size=(n, k))
    E = rng.normal(size=(k, q))
    gamma, delta = 3.0, 7.0

    prior = CandidateSemanticPrior()
    W0, A = prior.compute_w0_prior(Z, Y, E, gamma=gamma, delta=delta)
    assert_allclose(W0, A @ E.T)

    # Gradient of the full 4-term objective, evaluated at all four terms (not just the data term).
    residual = Z @ A @ E.T - Y
    grad = 2.0 * (
        Z.T @ residual @ E
        + gamma * A @ (E.T @ E)
        + delta * (Z.T @ Z) @ A
        + gamma * delta * A
    )
    scale = np.linalg.norm(Z.T @ Y @ E)
    assert np.max(np.abs(grad)) / scale < 1e-8

    def loss(A_):
        r = Z @ A_ @ E.T - Y
        return (
            np.sum(r**2)
            + gamma * np.sum((A_ @ E.T) ** 2)
            + delta * np.sum((Z @ A_) ** 2)
            + gamma * delta * np.sum(A_**2)
        )

    base = loss(A)
    for eps in (1e-4, 1e-3, 1e-2):
        direction = rng.normal(size=A.shape)
        assert loss(A + eps * direction) >= base - 1e-6 * max(base, 1.0)


def test_w0_prior_rejects_bad_shapes_and_nonfinite():
    prior = CandidateSemanticPrior()
    Z = np.ones((4, 3))
    Y = np.ones((4, 2))
    E = np.ones((2, 5))
    with pytest.raises(ValueError):
        prior.compute_w0_prior(Z, np.ones((3, 2)), E)  # N mismatch
    with pytest.raises(ValueError):
        prior.compute_w0_prior(Z, Y, np.ones((3, 5)))  # K mismatch
    with pytest.raises(ValueError):
        prior.compute_w0_prior(Z, Y, E, gamma=-1.0)
    bad_Z = Z.copy()
    bad_Z[0, 0] = np.nan
    with pytest.raises(ValueError):
        prior.compute_w0_prior(bad_Z, Y, E)


# ---------------------------------------------------------------------------
# 2. Zero-shot scoring on genuinely unseen candidate embeddings
# ---------------------------------------------------------------------------

def test_score_candidates_matches_raw_formula():
    rng = np.random.default_rng(1)
    Z_test = rng.normal(size=(7, 5))
    E_new = rng.normal(size=(3, 4))
    A = rng.normal(size=(5, 4))

    prior = CandidateSemanticPrior()
    scores = prior.score_candidates(Z_test, E_new, A)
    assert_allclose(scores, Z_test @ A @ E_new.T)
    assert scores.shape == (7, 3)


def test_score_candidates_recovers_class_on_held_out_inputs():
    """Same candidate set E used for fitting; only the *inputs* (Z rows) are held out."""
    rng = np.random.default_rng(2)
    k, d, q = 6, 8, 6
    E = _make_candidate_embeddings(rng, k, q)
    A_true = rng.normal(size=(d, q))
    Z, Y, y, _ = _synthetic_zero_shot_problem(rng, E, A_true, k=k, d=d)
    prior = CandidateSemanticPrior()
    _, A = prior.compute_w0_prior(Z, Y, E, gamma=0.1, delta=0.1)

    Z_new, _, y_new, _ = _synthetic_zero_shot_problem(rng, E, A_true, k=k, d=d, n_per_class=15)
    scores = prior.score_candidates(Z_new, E, A)
    predicted = np.argmax(scores, axis=1)
    assert (predicted == y_new).mean() >= 0.95


def test_score_candidates_transfers_to_genuinely_unseen_candidate_embeddings():
    """A is fit on 5 seen candidates only; scoring uses 3 *unseen* candidate embeddings that
    A never saw during the fit (`E_new` rows are never a column of the training E). Zero-shot
    transfer is only well-posed when the unseen embeddings live in the span of the seen ones
    (the standard ZSL assumption that seen/unseen classes share one attribute space) -- an
    E_new orthogonal to every seen row would carry no signal A could have learned to use, so
    E_new here is built from combinations of the seen rows.
    """
    rng = np.random.default_rng(2)
    q = 6
    E_full = _make_candidate_embeddings(rng, q, q)  # 6 orthonormal candidate rows in R^6
    k_train = 5
    E_train = E_full[:k_train]  # only 5 of the 6 are ever seen while fitting A
    d = 8
    A_true = rng.normal(size=(d, q))
    signal_scale, noise = 3.0, 0.15

    def make_Z(local_rng, E_rows, n_per_class):
        prototypes = signal_scale * (E_rows @ A_true.T)
        Z, y = [], []
        for i in range(E_rows.shape[0]):
            Z.append(prototypes[i] + noise * local_rng.normal(size=(n_per_class, d)))
            y.append(np.full(n_per_class, i))
        return np.concatenate(Z, axis=0), np.concatenate(y, axis=0)

    Z_train, y_train = make_Z(rng, E_train, 60)
    Y_train = np.eye(k_train)[y_train]

    prior = CandidateSemanticPrior()
    _, A = prior.compute_w0_prior(Z_train, Y_train, E_train, gamma=0.1, delta=0.1)

    # Unseen candidates: pairwise combinations of seen rows, never presented to compute_w0_prior.
    E_new = np.stack([E_train[0] + E_train[1], E_train[1] + E_train[2], E_train[2] + E_train[3]])
    E_new /= np.linalg.norm(E_new, axis=1, keepdims=True)
    assert not any(np.allclose(row, seen) for row in E_new for seen in E_train)

    Z_new, y_new = make_Z(rng, E_new, 20)
    predicted = np.argmax(prior.score_candidates(Z_new, E_new, A), axis=1)
    assert (predicted == y_new).mean() >= 0.95


def test_score_candidates_rejects_dimension_mismatch():
    prior = CandidateSemanticPrior()
    with pytest.raises(ValueError):
        prior.score_candidates(np.ones((3, 5)), np.ones((2, 4)), np.ones((6, 4)))
    with pytest.raises(ValueError):
        prior.score_candidates(np.ones((3, 5)), np.ones((2, 3)), np.ones((5, 4)))


# ---------------------------------------------------------------------------
# 3. Prototype shrinkage limits
# ---------------------------------------------------------------------------

def test_prototype_prior_collapses_to_semantic_prior_when_zero_shot():
    rng = np.random.default_rng(3)
    d, q, k = 4, 3, 5
    n_per_present = 20
    E = rng.normal(size=(k, q))
    # Class k-1 has zero training samples: only labels 0..k-2 appear in y.
    Z = rng.normal(size=(n_per_present * (k - 1), d))
    y = np.repeat(np.arange(k - 1), n_per_present)

    prior = CandidateSemanticPrior()
    kappa = 5.0
    prototypes = prior.compute_prototype_prior(Z, y, E, kappa=kappa, gamma=2.0, delta=2.0)

    Y = np.eye(k)[y]
    _, A = prior.compute_w0_prior(Z, Y, E, gamma=2.0, delta=2.0)
    expected_zero_shot_prototype = A @ E[k - 1]
    assert_allclose(prototypes[k - 1], expected_zero_shot_prototype, rtol=1e-10, atol=1e-10)


def test_prototype_prior_collapses_to_empirical_mean_when_data_rich():
    rng = np.random.default_rng(4)
    d, q, k = 3, 4, 3
    E = rng.normal(size=(k, q))
    kappa = 5.0

    n_big = 100_000
    true_mean = np.array([2.0, -1.0, 0.5])
    Z_big = true_mean + 0.1 * rng.normal(size=(n_big, d))
    y_big = np.zeros(n_big, dtype=int)
    # A couple of far-away, low-count classes so the fit is well-posed (E has 3 rows).
    Z_small = rng.normal(size=(4, d)) + 50.0
    y_small = np.array([1, 1, 2, 2])

    Z = np.concatenate([Z_big, Z_small], axis=0)
    y = np.concatenate([y_big, y_small], axis=0)

    prior = CandidateSemanticPrior()
    prototypes = prior.compute_prototype_prior(Z, y, E, kappa=kappa, gamma=2.0, delta=2.0)

    # The formula's own bound: |mu_k - z_bar_k| = kappa/(n_k+kappa) * |p_k - z_bar_k|. Checking
    # against this (not just against true_mean) means an implementation that silently drops
    # kappa's shrinkage term entirely would also pass a bare "close to true_mean" check but
    # fails this one, since it would put z_bar_k exactly at prototypes[0] every time.
    z_bar_0 = Z[y == 0].mean(axis=0)
    Y = np.eye(k)[y]
    _, A = prior.compute_w0_prior(Z, Y, E, gamma=2.0, delta=2.0)
    p_0 = A @ E[0]
    shrink_weight = kappa / (n_big + kappa)
    assert shrink_weight < 1e-4
    expected = (n_big * z_bar_0 + kappa * p_0) / (n_big + kappa)
    assert_allclose(prototypes[0], expected, rtol=1e-10, atol=1e-10)
    # And that expected value really is within the formula's shrink bound of the raw mean,
    # i.e. the limit claim (mu_k -> z_bar_k as n_k -> infinity) actually holds numerically.
    assert np.linalg.norm(prototypes[0] - z_bar_0) <= shrink_weight * np.linalg.norm(p_0 - z_bar_0) + 1e-9
    assert_allclose(prototypes[0], true_mean, atol=1e-2)


def test_prototype_prior_rejects_out_of_range_labels():
    prior = CandidateSemanticPrior()
    Z = np.ones((3, 2))
    y = np.array([0, 1, 2])
    E = np.ones((2, 4))  # only 2 candidates, but y references label 2
    with pytest.raises(ValueError):
        prior.compute_prototype_prior(Z, y, E)


# ---------------------------------------------------------------------------
# 4. shuffle_control: genuine permutation + semantics-vs-regularization ablation
# ---------------------------------------------------------------------------

def test_shuffle_control_is_a_genuine_deterministic_permutation():
    rng = np.random.default_rng(5)
    E = rng.normal(size=(6, 4))

    shuffled = CandidateSemanticPrior.shuffle_control(E, seed=42)
    shuffled_again = CandidateSemanticPrior.shuffle_control(E, seed=42)
    assert_allclose(shuffled, shuffled_again)  # deterministic

    assert not np.array_equal(shuffled, E)  # not the identity permutation
    # Same multiset of rows, just reordered.
    assert_allclose(sorted(shuffled.tolist()), sorted(E.tolist()))


def test_shuffle_control_ablation_isolates_semantic_gain_from_regularization():
    """Fits two classifiers with identical gamma/delta (identical regularization strength):
    one on the real E, one on `shuffle_control(E)`. Both are then scored against the *real*
    E on held-out data. If the gain were only from the ridge terms, shuffling E would not
    matter since gamma/delta are unchanged; instead the shuffled fit should perform far worse,
    showing the gain is tied to the real semantic alignment between Y's columns and E's rows.
    """
    rng = np.random.default_rng(6)
    k, d, q = 6, 8, 6
    E = _make_candidate_embeddings(rng, k, q)
    A_true = rng.normal(size=(d, q))
    Z, Y, y, _ = _synthetic_zero_shot_problem(rng, E, A_true, k=k, d=d, n_per_class=60)
    Z_test, _, y_test, _ = _synthetic_zero_shot_problem(rng, E, A_true, k=k, d=d, n_per_class=20)

    prior = CandidateSemanticPrior()
    gamma, delta = 0.1, 0.1

    _, A_real = prior.compute_w0_prior(Z, Y, E, gamma=gamma, delta=delta)
    real_accuracy = (np.argmax(prior.score_candidates(Z_test, E, A_real), axis=1) == y_test).mean()

    E_shuffled = CandidateSemanticPrior.shuffle_control(E, seed=42)
    _, A_shuffled = prior.compute_w0_prior(Z, Y, E_shuffled, gamma=gamma, delta=delta)
    shuffled_accuracy = (np.argmax(prior.score_candidates(Z_test, E, A_shuffled), axis=1) == y_test).mean()

    assert real_accuracy >= 0.95
    assert real_accuracy - shuffled_accuracy >= 0.3
