"""Tests for the dual-manifold state-action head (Spec 23 / Spec 24).

All synthetic: random seeded arrays, no task ids, no natural-language data.
NumPy only, float64 throughout.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from gen_zero.causal.dual_manifold_head import (
    TIERS,
    DecisionResult,
    DualManifoldHead,
    Status,
    simplex_etf,
)


def fixed_elapsed_clock(elapsed_ns: int):
    """A clock() stand-in: first call returns 0, second returns elapsed_ns."""
    values = iter([0, elapsed_ns])
    return lambda: next(values)


def identity_head(d_s, d_a, seed=0, with_sigma=False, sigma_scale=1.0):
    rng = np.random.default_rng(seed)
    P_s = np.eye(d_s)
    mu_s = np.zeros(d_s)
    P_a = np.eye(d_a)
    mu_a = np.zeros(d_a)
    M = rng.standard_normal((d_s, d_a))
    b = float(rng.standard_normal())
    Sigma_s = Sigma_a = None
    if with_sigma:
        Sigma_s = sigma_scale * np.eye(d_s)
        Sigma_a = sigma_scale * np.eye(d_a)
    return DualManifoldHead(P_s, mu_s, P_a, mu_a, M, b, Sigma_s, Sigma_a)


# ---------------------------------------------------------------------------
# a. simplex_etf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K,d", [(3, 8), (13, 64), (18, 896)])
def test_simplex_etf_geometry(K, d):
    rows = simplex_etf(K, d, seed=0)
    assert rows.shape == (K, d)
    norms = np.linalg.norm(rows, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)
    gram = rows @ rows.T
    target = -1.0 / (K - 1)
    off_diag = gram[~np.eye(K, dtype=bool)]
    assert np.allclose(off_diag, target, atol=1e-10)


def test_simplex_etf_k_greater_than_d_raises():
    with pytest.raises(ValueError):
        simplex_etf(10, 5)


def test_simplex_etf_k_less_than_2_raises():
    with pytest.raises(ValueError):
        simplex_etf(1, 5)


# ---------------------------------------------------------------------------
# b. bilinear equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds,da", [(64, 64), (64, 32), (896, 896)])
def test_bilinear_equivalence(ds, da):
    rng = np.random.default_rng(42)
    K = 16
    head = identity_head(ds, da, seed=1)
    z_s = rng.standard_normal(ds)
    Z_a = rng.standard_normal((K, da))

    direct = head.score_bilinear(z_s, Z_a)
    W, c = head.precompile(Z_a)
    precompiled = head.score_precompiled(z_s, W, c)
    assert np.max(np.abs(direct - precompiled)) < 1e-6

    explicit = np.array([z_s @ head.M @ Z_a[k] + head.b for k in range(K)])
    assert np.max(np.abs(direct - explicit)) < 1e-6


def test_precompile_folded_matches_raw_input():
    rng = np.random.default_rng(7)
    D_s, d_s = 1024, 896
    D_a = d_a = 896
    K = 16

    basis, _ = np.linalg.qr(rng.standard_normal((D_s, d_s)))
    P_s = basis  # orthonormal columns
    mu_s = rng.standard_normal(D_s)
    P_a = np.eye(D_a)
    mu_a = np.zeros(D_a)
    M = rng.standard_normal((d_s, d_a))
    b = float(rng.standard_normal())
    head = DualManifoldHead(P_s, mu_s, P_a, mu_a, M, b)

    x_s = rng.standard_normal(D_s)
    Z_a = rng.standard_normal((K, d_a))

    z_s = head.project_state(x_s)
    baseline = head.score_bilinear(z_s, Z_a)

    W_fold, c_fold = head.precompile_folded(Z_a)
    folded = W_fold @ x_s + c_fold
    assert np.max(np.abs(baseline - folded)) < 1e-6


# ---------------------------------------------------------------------------
# c. mask three-state
# ---------------------------------------------------------------------------

def _scored_head_k8():
    """Head with d_s=d_a=1 so an explicit W,c bypass fully controls scores."""
    head = DualManifoldHead(np.eye(1), np.zeros(1), np.eye(1), np.zeros(1), np.zeros((1, 1)), 0.0)
    scores = np.array([10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 0.0])
    z_s = np.array([1.0])
    Z_a = np.zeros((8, 1))
    W = scores.reshape(8, 1)
    c = np.zeros(8)
    return head, z_s, Z_a, W, c, scores


def test_mask_single_support_overrides_raw_argmax():
    head, z_s, Z_a, W, c, scores = _scored_head_k8()
    mask = np.zeros(8)
    mask[3] = 1
    result = head.predict_with_sla(z_s, Z_a, tier="tier2", axiom_mask=mask, W=W, c=c)
    assert result.status == Status.ACCEPT
    assert result.action_index == 3
    assert int(np.argmax(scores)) == 0  # raw argmax would have been 0


def test_mask_all_zero_refuses_empty_support():
    head, z_s, Z_a, W, c, scores = _scored_head_k8()
    mask = np.zeros(8)
    result = head.predict_with_sla(z_s, Z_a, tier="tier2", axiom_mask=mask, W=W, c=c)
    assert result.status == Status.REFUSE_EMPTY_SUPPORT
    assert result.action_index is None


def test_mask_multi_support_picks_argmax_within_support():
    head, z_s, Z_a, W, c, scores = _scored_head_k8()
    mask = np.zeros(8)
    for i in (2, 5, 7):
        mask[i] = 1
    result = head.predict_with_sla(z_s, Z_a, tier="tier2", axiom_mask=mask, W=W, c=c)
    assert result.status == Status.ACCEPT
    support = [2, 5, 7]
    expected = support[int(np.argmax(scores[support]))]
    assert result.action_index == expected
    assert result.action_index == 5
    assert result.action_index != 0


def test_mask_soft_values_raise():
    head, z_s, Z_a, W, c, scores = _scored_head_k8()
    mask = np.full(8, 0.5)
    with pytest.raises(ValueError):
        head.predict_with_sla(z_s, Z_a, tier="tier2", axiom_mask=mask, W=W, c=c)


def test_mask_wrong_shape_raises():
    head, z_s, Z_a, W, c, scores = _scored_head_k8()
    mask = np.zeros(7)
    with pytest.raises(ValueError):
        head.predict_with_sla(z_s, Z_a, tier="tier2", axiom_mask=mask, W=W, c=c)


# ---------------------------------------------------------------------------
# d. variance gate
# ---------------------------------------------------------------------------

def _variance_head(sigma_scale):
    d_s = d_a = 2
    head = DualManifoldHead(
        np.eye(d_s), np.zeros(d_s), np.eye(d_a), np.zeros(d_a),
        np.zeros((d_s, d_a)), 0.0,
        sigma_scale * np.eye(d_s), sigma_scale * np.eye(d_a),
    )
    return head


def _variance_scene():
    z_s = np.array([1.0, 0.0])
    Z_a = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    scores = np.array([10.0, 1.0, 0.0, -1.0])
    W = np.column_stack([scores, np.zeros(4)])
    c = np.zeros(4)
    return z_s, Z_a, W, c


def test_variance_gate_accepts_with_small_sigma():
    head = _variance_head(1e-6)
    z_s, Z_a, W, c = _variance_scene()
    result = head.predict_with_sla(z_s, Z_a, tier="tier2", W=W, c=c)
    assert result.status == Status.ACCEPT
    assert result.action_index == 0
    assert result.sigma2 is not None
    assert result.sigma2.shape == (4,)
    assert np.all(result.sigma2 > 0)


def test_variance_gate_refuses_with_large_sigma():
    head = _variance_head(10.0)
    z_s, Z_a, W, c = _variance_scene()
    result = head.predict_with_sla(z_s, Z_a, tier="tier2", W=W, c=c)
    assert result.status == Status.REFUSE_UNCERTAIN
    assert result.action_index is None
    assert result.sigma2 is not None
    assert result.sigma2.shape == (4,)


def test_variance_still_computed_for_single_support():
    head = _variance_head(1e-6)
    z_s, Z_a, W, c = _variance_scene()
    mask = np.zeros(4)
    mask[2] = 1
    result = head.predict_with_sla(z_s, Z_a, tier="tier2", axiom_mask=mask, W=W, c=c)
    assert result.status == Status.ACCEPT
    assert result.action_index == 2
    assert result.sigma2 is not None
    assert result.sigma2.shape == (4,)


def test_variance_raises_without_sigma():
    head = identity_head(4, 4, seed=3, with_sigma=False)
    z_s = np.zeros(4)
    Z_a = np.zeros((5, 4))
    with pytest.raises(ValueError):
        head.variance(z_s, Z_a)


# ---------------------------------------------------------------------------
# e. SLA with a fake clock
# ---------------------------------------------------------------------------

def _sla_head():
    return identity_head(4, 4, seed=9)


def _sla_scene():
    rng = np.random.default_rng(9)
    z_s = rng.standard_normal(4)
    Z_a = rng.standard_normal((2, 4))
    return z_s, Z_a


@pytest.mark.parametrize(
    "tier,elapsed_ns,expect_timeout",
    [
        ("tier0", 49_900, False),
        ("tier0", 50_000, True),
        ("tier1", 1_000_000, False),
        ("tier1", 1_000_100, True),
        ("tier2", 20_000_000, False),
        ("tier2", 20_001_000, True),
    ],
)
def test_sla_fake_clock_boundaries(tier, elapsed_ns, expect_timeout):
    head = _sla_head()
    z_s, Z_a = _sla_scene()
    clock = fixed_elapsed_clock(elapsed_ns)
    result = head.predict_with_sla(z_s, Z_a, tier=tier, clock=clock)
    if expect_timeout:
        assert result.status == Status.TIMEOUT
        assert result.action_index is None
        assert result.scores is not None
    else:
        assert result.status != Status.TIMEOUT


def test_sla_unknown_tier_raises():
    head = _sla_head()
    z_s, Z_a = _sla_scene()
    with pytest.raises(ValueError):
        head.predict_with_sla(z_s, Z_a, tier="tier3")


# ---------------------------------------------------------------------------
# f. real wall clock
# ---------------------------------------------------------------------------

def test_sla_real_clock_tier2_never_times_out():
    rng = np.random.default_rng(123)
    d_s = d_a = 32
    K = 8
    head = DualManifoldHead(
        np.eye(d_s), np.zeros(d_s), np.eye(d_a), np.zeros(d_a),
        rng.standard_normal((d_s, d_a)), 0.1,
        1e-4 * np.eye(d_s), 1e-4 * np.eye(d_a),
    )
    z_s = rng.standard_normal(d_s)
    Z_a = rng.standard_normal((K, d_a))
    mask = np.ones(K)
    for _ in range(200):
        result = head.predict_with_sla(z_s, Z_a, tier="tier2", axiom_mask=mask)
        assert result.status != Status.TIMEOUT


def _measure_tier(head, z_s, Z_a, tier, n):
    elapsed = np.empty(n)
    timeouts = 0
    for i in range(n):
        result = head.predict_with_sla(z_s, Z_a, tier=tier)
        elapsed[i] = result.elapsed_us
        if result.status == Status.TIMEOUT:
            timeouts += 1
    p50 = float(np.percentile(elapsed, 50))
    p99 = float(np.percentile(elapsed, 99))
    mx = float(np.max(elapsed))
    return p50, p99, mx, timeouts


def test_sla_real_clock_measured_tier0_and_tier1():
    rng = np.random.default_rng(321)
    d_s = d_a = 32
    K = 16
    head = identity_head(d_s, d_a, seed=321)
    z_s = rng.standard_normal(d_s)
    Z_a = rng.standard_normal((K, d_a))

    n = 1000
    for tier in ("tier0", "tier1"):
        p50, p99, mx, timeouts = _measure_tier(head, z_s, Z_a, tier, n)
        frac = timeouts / n
        print(
            f"SLA_MEASURED {tier} K={K} d=32 p50={p50:.2f} p99={p99:.2f} "
            f"max={mx:.2f} n={n} timeout_frac={frac:.3f}"
        )
    # Deliberately no assertions on the numbers above: Spec 24 measured
    # tier0 fails on shared/unpinned hardware; asserting would be flaky.


# ---------------------------------------------------------------------------
# g. invalid inputs
# ---------------------------------------------------------------------------

def test_non_psd_sigma_raises():
    d = 3
    bad = np.eye(d)
    bad[0, 0] = -1.0
    with pytest.raises(ValueError):
        DualManifoldHead(np.eye(d), np.zeros(d), np.eye(d), np.zeros(d), np.zeros((d, d)), 0.0, bad, np.eye(d))


def test_one_sigma_without_other_raises():
    d = 3
    with pytest.raises(ValueError):
        DualManifoldHead(np.eye(d), np.zeros(d), np.eye(d), np.zeros(d), np.zeros((d, d)), 0.0, np.eye(d), None)


def test_nonfinite_z_s_raises():
    head = identity_head(4, 4, seed=5)
    z_s = np.array([1.0, np.nan, 0.0, 0.0])
    Z_a = np.zeros((3, 4))
    with pytest.raises(ValueError):
        head.predict_with_sla(z_s, Z_a, tier="tier2")


def test_wrong_z_s_dim_raises():
    head = identity_head(4, 4, seed=5)
    z_s = np.zeros(5)
    Z_a = np.zeros((3, 4))
    with pytest.raises(ValueError):
        head.predict_with_sla(z_s, Z_a, tier="tier2")
