"""Tests for ordinal-aware soft targets and expected-value decoding.

Covers ordinal_soft_targets and ordinal_expected_decode in
gen_zero.manifold.master_objective: monotone target shape, a real
closed-form ridge-regression comparison against one-hot targets on
simulated ordinal data, decoder boundary/shape behaviour, and
fail-closed parameter validation.
"""

import numpy as np
import pytest
from scipy.linalg import solve
from scipy.special import softmax

from gen_zero.manifold.master_objective import (
    ordinal_expected_decode,
    ordinal_soft_targets,
)


# --------------------------------------------------------------- monotonicity
@pytest.mark.parametrize("y_i", [0, 1, 2, 3, 4])
def test_soft_targets_are_monotone_in_rank_distance(y_i):
    """Row i peaks at y_i and strictly decays as |j - y_i| grows."""
    k = 5
    row = ordinal_soft_targets(np.array([y_i]), k, tau=1.0)[0]
    order = sorted(range(k), key=lambda j: abs(j - y_i))
    values_by_distance = [row[j] for j in order]
    # Ties only occur between two columns at the same distance (e.g. y_i=2,
    # j=1 and j=3 are both distance 1); strictly decreasing only holds
    # between distinct distances, so compare distance-group maxima.
    by_distance = {}
    for j in range(k):
        by_distance.setdefault(abs(j - y_i), []).append(row[j])
    distances = sorted(by_distance)
    group_values = [max(by_distance[d]) for d in distances]
    assert all(a > b for a, b in zip(group_values, group_values[1:])), (
        f"y_i={y_i}: expected strictly decreasing peak per rank-distance, got {row}"
    )
    assert np.argmax(row) == y_i


def test_soft_targets_rows_sum_to_one_and_shape():
    y = np.array([0, 2, 4, 1, 3])
    k = 5
    Y = ordinal_soft_targets(y, k, tau=0.7)
    assert Y.shape == (5, 5)
    assert Y.dtype == np.float64
    assert np.allclose(Y.sum(axis=1), 1.0)
    assert np.all(Y > 0)


def test_soft_targets_larger_tau_flattens_distribution():
    y = np.array([2])
    k = 5
    sharp = ordinal_soft_targets(y, k, tau=0.3)[0]
    flat = ordinal_soft_targets(y, k, tau=5.0)[0]
    # A larger tau should spread mass further from the peak.
    assert sharp[2] > flat[2]
    assert sharp[0] < flat[0]


# ------------------------------------------------- ridge simulation: MAE gain
def _fit_ridge_head(Z_train, target, lam):
    d = Z_train.shape[1] - 1
    reg = np.eye(d + 1) * lam
    reg[-1, -1] = 0.0
    return solve(Z_train.T @ Z_train + reg, Z_train.T @ target, assume_a="pos")


def _simulate_ordinal_regression(seed, n_train=150, n_test=2000, k=5, d=20,
                                  noise=8.0, tau=1.0, lam=30.0):
    """Deterministic synthetic ordinal-rank scenario: a noisy linear signal
    carries the true rank, fit with a plain closed-form ridge head twice
    (one-hot target decoded by argmax, ordinal-soft target decoded by
    expected-value), and return held-out (MAE, accuracy) for each."""
    rng = np.random.default_rng(seed)
    y_train = rng.integers(0, k, size=n_train)
    y_test = rng.integers(0, k, size=n_test)
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)

    def make_x(y):
        signal = np.outer(y.astype(float), w_true)
        return signal + noise * rng.normal(size=(len(y), d))

    Z_train = np.column_stack([make_x(y_train), np.ones(n_train)])
    Z_test = np.column_stack([make_x(y_test), np.ones(n_test)])

    W_onehot = _fit_ridge_head(Z_train, np.eye(k)[y_train], lam)
    W_soft = _fit_ridge_head(Z_train, ordinal_soft_targets(y_train, k, tau), lam)

    pred_onehot = (Z_test @ W_onehot).argmax(1)
    probs_soft = softmax(Z_test @ W_soft, axis=1)
    pred_soft = ordinal_expected_decode(probs_soft)

    mae_onehot = float(np.mean(np.abs(pred_onehot - y_test)))
    mae_soft = float(np.mean(np.abs(pred_soft - y_test)))
    acc_onehot = float(np.mean(pred_onehot == y_test))
    acc_soft = float(np.mean(pred_soft == y_test))
    return mae_onehot, mae_soft, acc_onehot, acc_soft


def test_ordinal_soft_target_head_has_lower_mean_absolute_rank_error():
    """Load-bearing claim: on noisy ordinal data, a ridge head trained with
    ordinal_soft_targets and decoded via ordinal_expected_decode achieves
    materially lower mean absolute rank error than a one-hot/argmax head
    fit on the exact same features and folds. Fixed seed => deterministic."""
    mae_onehot, mae_soft, acc_onehot, acc_soft = _simulate_ordinal_regression(seed=3)
    assert mae_soft < 0.8 * mae_onehot, (
        f"expected ordinal MAE ({mae_soft:.3f}) < 0.8x one-hot MAE ({mae_onehot:.3f})"
    )
    # Informational: exact-match accuracy is not the load-bearing claim here
    # (a well-separated one-hot fit can already get argmax right despite
    # optimizing the wrong loss); just sanity-check neither head degenerates.
    assert acc_onehot > 0.0 and acc_soft > 0.0


# --------------------------------------------------------- decoder shape/edge
def test_expected_decode_one_hot_peak():
    proba = np.zeros((1, 5))
    proba[0, 2] = 1.0
    out = ordinal_expected_decode(proba)
    assert out.shape == (1,)
    assert np.issubdtype(out.dtype, np.integer)
    assert out[0] == 2


def test_expected_decode_uniform_rounds_to_middle():
    proba = np.full((1, 5), 1.0 / 5)
    out = ordinal_expected_decode(proba)
    assert out[0] == 2  # round(2.0) == 2


def test_expected_decode_always_within_bounds():
    k = 5
    rng = np.random.default_rng(42)
    rows = []
    # Adversarial rows concentrated at the extreme boundary columns.
    rows.append([0.5, 0.5, 0.0, 0.0, 0.0])
    rows.append([0.0, 0.0, 0.0, 0.5, 0.5])
    rows.append([1.0, 0.0, 0.0, 0.0, 0.0])
    rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
    for _ in range(20):
        v = rng.uniform(size=k)
        v = v / v.sum()
        rows.append(v.tolist())
    proba = np.array(rows)
    out = ordinal_expected_decode(proba)
    assert out.shape == (len(rows),)
    assert np.issubdtype(out.dtype, np.integer)
    assert np.all(out >= 0) and np.all(out <= k - 1)
    # 0.5/0.5 split between columns 0 and 1: numpy round-half-to-even on 0.5 -> 0.
    assert out[0] == 0
    # 0.5/0.5 split between columns 3 and 4: expected value 3.5 -> round-half-to-even -> 4.
    assert out[1] == 4


# --------------------------------------------------------------- fail closed
@pytest.mark.parametrize("k, tau, y, match", [
    (1, 1.0, np.array([0]), "k must be"),
    (0, 1.0, np.array([0]), "k must be"),
    (5, 0.0, np.array([0]), "tau must be"),
    (5, -1.0, np.array([0]), "tau must be"),
    (5, np.inf, np.array([0]), "tau must be"),
    (5, np.nan, np.array([0]), "tau must be"),
    (5, 1.0, np.array([5]), r"outside \[0, 5\)"),
    (5, 1.0, np.array([-1]), r"outside \[0, 5\)"),
])
def test_ordinal_soft_targets_fails_closed(k, tau, y, match):
    with pytest.raises(ValueError, match=match):
        ordinal_soft_targets(y, k, tau)


def test_ordinal_soft_targets_rejects_non_integer_y():
    with pytest.raises(ValueError, match="integer"):
        ordinal_soft_targets(np.array([0.5, 1.5]), 5, 1.0)


def test_ordinal_soft_targets_rejects_non_1d_y():
    with pytest.raises(ValueError, match="1-D"):
        ordinal_soft_targets(np.array([[0, 1], [2, 3]]), 5, 1.0)


def test_ordinal_expected_decode_fails_closed():
    with pytest.raises(ValueError, match="2-D"):
        ordinal_expected_decode(np.array([0.2, 0.3, 0.5]))
    with pytest.raises(ValueError, match="non-negative"):
        ordinal_expected_decode(np.array([[0.5, -0.5, 1.0]]))
    with pytest.raises(ValueError, match="sum to 1"):
        ordinal_expected_decode(np.array([[0.2, 0.1]]))  # sums to 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        ordinal_expected_decode(np.array([[0.8, 0.7]]))  # sums to 1.5
