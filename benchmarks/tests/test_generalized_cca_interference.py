"""Generalized CCA manifold interference operator (benchmarks/suites/generalized_cca_manifold_interference.py).

Covers: identical views align perfectly (rho = 1), exactly orthogonal views give rho = 0 (built with
n >= d_1 + d_2, the only shape where that is possible), the sample-space solver against the primal
regularized generalized eigenproblem, Procrustes orthogonality and recovery of a known orthogonal
map, cross-check against the existing ``procrustes_residual``, fail-closed input validation, and the
n = 100, d = 128 / 8192 CPU regime, including the proof that in-sample rho is 1 there even for
shuffled rows (so only held-out rho against a permutation null may be quoted).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.stats import ortho_group

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suites"))

import generalized_cca_manifold_interference as gcm  # noqa: E402

OP = gcm.GeneralizedCCAManifoldInterference


@pytest.fixture(autouse=True)
def _single_blas_thread():
    # Measured on a box at load ~66 / 24 cores: one 100 x 8192 fit takes 0.7 s on 1 BLAS thread and
    # 46 s on 24 (oversubscription). Pin to 1 so the timing assertion measures this code, not the box.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(1):
        yield


def _rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ perfect alignment

def test_identical_views_give_all_ones_without_ridge():
    x = _rng().standard_normal((200, 12))
    res = OP(reg=0.0).fit([x, x.copy()])
    rho = res.canonical_correlations[(0, 1)]
    assert rho.shape == (12,)
    np.testing.assert_allclose(rho, 1.0, atol=1e-10)
    # M = 2, no ridge: resonance spectrum equals the canonical correlations.
    np.testing.assert_allclose(res.resonance_spectrum[:12], rho, atol=1e-10)
    np.testing.assert_allclose(res.orthogonal_energy_ratio, 0.0, atol=1e-10)
    assert res.participation_ratio == pytest.approx(12.0)
    assert res.entropy_effective_rank == pytest.approx(12.0)


def test_identical_views_with_ridge_shrink_by_the_analytic_factor():
    x = _rng(1).standard_normal((150, 6)) * np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    res = OP(reg=0.1).fit([x, x.copy()])
    sig2 = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2 / (x.shape[0] - 1)
    ridge = 0.1 * sig2.mean()
    assert res.ridges[0] == pytest.approx(ridge)
    np.testing.assert_allclose(res.canonical_correlations[(0, 1)], sig2 / (sig2 + ridge), rtol=1e-10)


def test_orthogonal_transform_invariance_of_rho():
    rng = _rng(2)
    x1 = rng.standard_normal((300, 8))
    x2 = x1[:, :5] @ rng.standard_normal((5, 10)) + 0.5 * rng.standard_normal((300, 10))
    base = OP(reg=0.0).fit([x1, x2]).canonical_correlations[(0, 1)]
    q1, q2 = ortho_group.rvs(8, random_state=3), ortho_group.rvs(10, random_state=4)
    rot = OP(reg=0.0).fit([x1 @ q1, x2 @ q2]).canonical_correlations[(0, 1)]
    np.testing.assert_allclose(rot, base, atol=1e-10)
    assert np.all(np.diff(base) <= 1e-12)


# ------------------------------------------------------------------ uncorrelated manifolds

def test_exactly_orthogonal_views_give_zero_rho():
    # n >= d1 + d2 is required: with fewer rows two views cannot be uncorrelated in-sample.
    n, d1, d2 = 120, 10, 15
    base = _rng(5).standard_normal((n, d1 + d2))
    q, _ = np.linalg.qr(base - base.mean(0))  # centered, mutually orthogonal columns
    x1, x2 = q[:, :d1] @ _rng(6).standard_normal((d1, d1)), q[:, d1:] @ _rng(7).standard_normal((d2, d2))
    res = OP(reg=0.0).fit([x1, x2])
    rho = res.canonical_correlations[(0, 1)]
    np.testing.assert_allclose(rho, 0.0, atol=1e-10)
    assert res.participation_ratio == 0.0 and res.entropy_effective_rank == 0.0


def test_independent_noise_with_many_rows_is_near_zero():
    rng = _rng(8)
    res = OP(reg=0.0).fit([rng.standard_normal((20000, 5)), rng.standard_normal((20000, 5))])
    assert res.canonical_correlations[(0, 1)].max() < 0.05


# ------------------------------------------------------------------ dual vs primal

def test_sample_space_solver_matches_primal_generalized_eigenproblem():
    rng = _rng(9)
    n, d1, d2, reg = 250, 5, 7, 0.05
    z = rng.standard_normal((n, 3))
    x1 = z @ rng.standard_normal((3, d1)) + rng.standard_normal((n, d1))
    x2 = z @ rng.standard_normal((3, d2)) + rng.standard_normal((n, d2))
    res = OP(reg=reg).fit([x1, x2])
    c = np.cov(np.hstack([x1, x2]), rowvar=False)
    c11, c12, c22 = c[:d1, :d1], c[:d1, d1:], c[d1:, d1:]
    r1, r2 = res.ridges
    a = np.block([[np.zeros((d1, d1)), c12], [c12.T, np.zeros((d2, d2))]])
    b = np.block([[c11 + r1 * np.eye(d1), np.zeros((d1, d2))], [np.zeros((d2, d1)), c22 + r2 * np.eye(d2)]])
    primal = np.sort(eigh(a, b, eigvals_only=True))[::-1][:min(d1, d2)]
    np.testing.assert_allclose(res.canonical_correlations[(0, 1)], primal, atol=1e-10)
    np.testing.assert_allclose(res.ridges[0], reg * np.trace(c11) / d1, rtol=1e-10)


def test_maxvar_three_views_and_pairwise_keys():
    rng = _rng(10)
    z = rng.standard_normal((400, 2))
    views = [z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((400, d)) for d in (6, 8, 4)]
    res = OP(reg=0.0, n_shared=2).fit(views)
    assert set(res.canonical_correlations) == {(0, 1), (0, 2), (1, 2)}
    assert res.maxvar_eigenvalues[0] <= 3.0 + 1e-10
    assert res.resonance_spectrum[:2].min() > 0.9  # the two planted shared directions
    assert res.shared_basis.shape == (400, 2)
    np.testing.assert_allclose(res.shared_basis.T @ res.shared_basis, np.eye(2), atol=1e-10)


# ------------------------------------------------------------------ Procrustes

def test_procrustes_recovers_known_orthogonal_map_and_is_orthogonal():
    rng = _rng(11)
    x1 = rng.standard_normal((200, 32))
    q = ortho_group.rvs(32, random_state=12)
    pr = OP().procrustes(x1, x1 @ q)
    r = pr.rotation
    np.testing.assert_allclose(r.T @ r, np.eye(32), atol=1e-10)
    np.testing.assert_allclose(r, q, atol=1e-8)  # n > d: the minimizer is unique
    assert pr.residual_ratio < 1e-6               # closed form is quantized near 1e-7
    np.testing.assert_allclose(pr.left @ pr.right.T, q, atol=1e-8)


def test_procrustes_unequal_dims_zero_pads_and_stays_orthogonal():
    rng = _rng(13)
    x1 = rng.standard_normal((80, 10))
    x2 = np.hstack([x1, np.zeros((80, 6))]) @ ortho_group.rvs(16, random_state=14)
    pr = OP().procrustes(x1, x2)
    assert pr.padded_dim == 16 and pr.rotation.shape == (16, 16)
    np.testing.assert_allclose(pr.rotation.T @ pr.rotation, np.eye(16), atol=1e-10)
    assert pr.residual_ratio < 1e-6


def test_procrustes_residual_matches_existing_module_and_brute_force():
    import cross_model_manifold_alignment as cmma

    rng = _rng(15)
    x1, x2 = rng.standard_normal((60, 9)), rng.standard_normal((60, 14))
    pr = OP().procrustes(x1, x2)
    assert np.sqrt(pr.residual_ratio) == pytest.approx(cmma.procrustes_residual(x1, x2), abs=1e-9)
    a = x1 - x1.mean(0); a /= np.linalg.norm(a)
    b = x2 - x2.mean(0); b /= np.linalg.norm(b)
    ap = np.hstack([a, np.zeros((60, 5))])
    direct = np.linalg.norm(ap @ pr.rotation - b) ** 2
    assert direct == pytest.approx(pr.residual_ratio, abs=1e-10)


def test_procrustes_refuses_dense_rotation_above_limit_but_gives_factors():
    rng = _rng(16)
    pr = OP().procrustes(rng.standard_normal((20, 300)), rng.standard_normal((20, 300)), dense_limit=100)
    assert pr.rotation is None
    np.testing.assert_allclose(pr.left.T @ pr.left, np.eye(pr.left.shape[1]), atol=1e-10)
    np.testing.assert_allclose(pr.right.T @ pr.right, np.eye(pr.right.shape[1]), atol=1e-10)


# ------------------------------------------------------------------ fail-closed

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_raise(bad):
    x = _rng().standard_normal((10, 4))
    y = x.copy(); y[3, 2] = bad
    with pytest.raises(ValueError, match="non-finite"):
        OP().fit([x, y])
    with pytest.raises(ValueError, match="non-finite"):
        OP().procrustes(y, x)


@pytest.mark.parametrize("views, match", [
    (lambda r: [r.standard_normal((10, 4)), r.standard_normal((11, 4))], "same number of samples"),
    (lambda r: [r.standard_normal((1, 4)), r.standard_normal((1, 4))], "at least 2 samples"),
    (lambda r: [r.standard_normal(10), r.standard_normal((10, 3))], "2D"),
    (lambda r: [r.standard_normal((10, 3, 1)), r.standard_normal((10, 3))], "2D"),
    (lambda r: [r.standard_normal((10, 0)), r.standard_normal((10, 3))], "at least 1 feature"),
    (lambda r: [r.standard_normal((10, 3))], "at least 2"),
    (lambda r: [np.ones((10, 3)), r.standard_normal((10, 3))], "zero variance"),
    (lambda r: [np.array([["a"] * 3] * 10), r.standard_normal((10, 3))], "floating-point"),
    (lambda r: [np.ones((10, 3), dtype=bool), r.standard_normal((10, 3))], "floating-point"),
    (lambda r: [r.standard_normal((10, 3)) + 1j, r.standard_normal((10, 3))], "floating-point"),
])
def test_bad_inputs_raise(views, match):
    with pytest.raises(ValueError, match=match):
        OP().fit(views(_rng()))


@pytest.mark.parametrize("kwargs", [{"reg": -1.0}, {"reg": float("nan")}, {"max_rank": 0}, {"n_shared": 0},
                                     {"max_rank": True}, {"n_shared": True}])
def test_bad_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        OP(**kwargs)


def test_constant_view_raises_even_with_floating_point_noise():
    # np.full((10, 3), 0.1) must not slip through as a rank-1 view via round-off in centering.
    with pytest.raises(ValueError, match="zero variance"):
        OP().fit([np.full((10, 3), 0.1), _rng().standard_normal((10, 3))])


def test_split_rows_rejects_bool_and_negative_max_rows():
    with pytest.raises(ValueError, match="max_rows"):
        gcm._split_rows(10, -1, 0.5, 0)
    with pytest.raises(ValueError, match="max_rows"):
        gcm._split_rows(10, True, 0.5, 0)


def test_n_shared_larger_than_available_raises():
    rng = _rng()
    with pytest.raises(ValueError, match="n_shared"):
        OP(n_shared=50).fit([rng.standard_normal((10, 3)), rng.standard_normal((10, 3))])


# ------------------------------------------------------------------ n = 100, d = 128 / 8192 on CPU

@pytest.mark.parametrize("d1, d2", [(128, 8192), (8192, 8192)])
def test_large_dim_regime_is_stable_fast_and_in_sample_rho_is_uninformative(d1, d2):
    rng = _rng(17)
    n = 100
    x1 = rng.standard_normal((n, d1)).astype(np.float32)
    x2 = rng.standard_normal((n, d2)).astype(np.float32)
    op = OP(reg=0.0)
    started = time.perf_counter()
    res = op.fit([x1, x2])
    pr = op.procrustes(x1, x2, dense_limit=0)
    elapsed = time.perf_counter() - started
    rho = res.canonical_correlations[(0, 1)]
    assert rho.shape == (n - 1,)  # rank support after centering, never zero-padded
    assert np.all(np.isfinite(rho)) and np.all(np.diff(rho) <= 1e-12)
    assert np.all((rho >= 0) & (rho <= 1))
    assert np.isfinite(pr.residual_ratio) and pr.rotation is None
    # Independent random views, yet every in-sample rho is 1: n <= d makes both row spaces the same
    # (n-1)-dim space. This is the artifact the module docstring warns about, pinned here.
    np.testing.assert_allclose(rho, 1.0, atol=1e-6)
    shuffled = op.fit([x1, x2[rng.permutation(n)]]).canonical_correlations[(0, 1)]
    np.testing.assert_allclose(shuffled, 1.0, atol=1e-6)
    assert elapsed < 20.0, f"took {elapsed:.1f}s on CPU"  # ~1-2 s measured single-threaded


def test_heldout_rho_raises_on_zero_variance_eval_set():
    rng = _rng(19)
    x1, x2 = rng.standard_normal((60, 10)), rng.standard_normal((60, 10))
    op = OP(reg=1e-3, max_rank=5)
    const_eval = [np.tile(x1[0], (10, 1)), np.tile(x2[0], (10, 1))]
    with pytest.raises(ValueError, match="zero variance"):
        op.heldout_canonical_correlations([x1, x2], const_eval)


def test_heldout_rho_raises_on_numerical_overflow_instead_of_returning_nan():
    rng = _rng(20)
    x1, x2 = rng.standard_normal((60, 10)), rng.standard_normal((60, 10))
    op = OP(reg=1e-3, max_rank=5)
    huge_eval = [rng.standard_normal((10, 10)) * 1e200, rng.standard_normal((10, 10)) * 1e200]
    with pytest.raises(ValueError, match="overflow"):
        op.heldout_canonical_correlations([x1, x2], huge_eval)


def test_heldout_rho_separates_planted_signal_from_permutation_null_at_n_below_d():
    rng = _rng(18)
    n, d, k = 200, 8192, 3
    z = rng.standard_normal((n, k))
    x1 = z @ rng.standard_normal((k, d)) + 2.0 * rng.standard_normal((n, d))
    x2 = z @ rng.standard_normal((k, d)) + 2.0 * rng.standard_normal((n, d))
    fit, ev = slice(0, 100), slice(100, 200)
    op = OP(reg=1e-3, max_rank=16)
    heldout = op.heldout_canonical_correlations([x1[fit], x2[fit]], [x1[ev], x2[ev]])
    null = op.permutation_null([x1[fit], x2[fit]], [x1[ev], x2[ev]], permutations=10, seed=0)
    assert heldout.shape == (16,)
    assert np.all(heldout[:k] > 0.8)
    assert np.all(heldout[:k] > null["per_direction"][:k])
    assert np.all(np.abs(null["mean"]) < 0.3)
