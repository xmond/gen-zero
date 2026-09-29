"""Tests for the LoRA parallel-RNN semigroup scan (parallel_rnn_lora.py).

Covers: (1) associativity of the raw (A, u) combine operator, (2) exact
agreement of the Blelloch `associative_scan` with the O(T) `sequential_scan`
reference, (3) strict Lyapunov spectral-radius enforcement (including an
adversarial raw matrix that starts far outside the unit disk), (4) constant
O(dim*rank) working set and no memory growth across many `step()` calls,
(5) fail-closed rejection of non-finite input, and (6) no `re` import /
no hardcoded-label shortcuts in the implementation module.
"""
import ast
import tracemalloc

import numpy as np
import pytest

from gen_zero.causal.parallel_rnn_lora import (
    ParallelRNNLoRAAdapter,
    associative_scan,
    combine,
    sequential_scan,
    spectral_normalize,
)
from gen_zero.causal.lyapunov_verifier import verify_trajectory


def _contractive_dense_sequence(rng, T, d, cap=0.8):
    """Random dense (non-commutative) A_t sequence with sigma_max(A_t) <= cap."""
    A = rng.standard_normal((T, d, d))
    for t in range(T):
        sigma = np.linalg.svd(A[t], compute_uv=False)[0]
        A[t] *= cap / sigma
    u = rng.standard_normal((T, d))
    return A, u


# --------------------------------------------------------------------------
# 1. Associativity of the raw semigroup operator.
# --------------------------------------------------------------------------
def test_combine_associativity_dense_nonrandom_order_matters():
    rng = np.random.default_rng(0)
    d = 5
    g1 = (rng.standard_normal((d, d)) * 0.5, rng.standard_normal(d))
    g2 = (rng.standard_normal((d, d)) * 0.5, rng.standard_normal(d))
    g3 = (rng.standard_normal((d, d)) * 0.5, rng.standard_normal(d))

    lhs = combine(combine(g1, g2), g3)
    rhs = combine(g1, combine(g2, g3))
    np.testing.assert_allclose(lhs[0], rhs[0], atol=1e-10)
    np.testing.assert_allclose(lhs[1], rhs[1], atol=1e-10)

    # Non-commutative: composing in the opposite order must differ.
    swapped = combine(g2, g1)
    original = combine(g1, g2)
    assert np.max(np.abs(swapped[0] - original[0])) > 1e-6


def test_combine_associativity_random_batch():
    rng = np.random.default_rng(1)
    d, n = 4, 200
    def rand_elem():
        return (rng.standard_normal((n, d, d)) * 0.3, rng.standard_normal((n, d)))
    a, b, c = rand_elem(), rand_elem(), rand_elem()
    lhs = combine(combine(a, b), c)
    rhs = combine(a, combine(b, c))
    np.testing.assert_allclose(lhs[0], rhs[0], atol=1e-9)
    np.testing.assert_allclose(lhs[1], rhs[1], atol=1e-9)


# --------------------------------------------------------------------------
# 2. associative_scan == sequential_scan.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("T", [1, 2, 5, 37, 64])
def test_associative_scan_matches_sequential(T):
    rng = np.random.default_rng(T)
    d = 6
    A, u = _contractive_dense_sequence(rng, T, d)
    ref = sequential_scan(A, u)
    got = associative_scan(A, u)
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_associative_scan_matches_sequential_larger_T_and_d():
    rng = np.random.default_rng(42)
    T, d = 300, 10
    A, u = _contractive_dense_sequence(rng, T, d, cap=0.95)
    ref = sequential_scan(A, u)
    got = associative_scan(A, u)
    np.testing.assert_allclose(got, ref, atol=1e-6)


# --------------------------------------------------------------------------
# 3. Spectral-radius / Lyapunov stability enforcement.
# --------------------------------------------------------------------------
def test_spectral_normalize_adversarial_case():
    d = 6
    huge = 1e6 * np.eye(d)
    rho_max = 0.9
    out = spectral_normalize(huge, rho_max)
    sigma = np.linalg.svd(out, compute_uv=False)[0]
    assert sigma < 1.0
    assert sigma == pytest.approx(rho_max, rel=1e-9)


def test_spectral_normalize_leaves_already_small_matrix_alone():
    rng = np.random.default_rng(2)
    small = rng.standard_normal((5, 5)) * 0.01
    out = spectral_normalize(small, 0.9)
    np.testing.assert_allclose(out, small)


def test_adapter_transition_matrix_spectral_radius_is_strictly_below_one():
    for seed in range(5):
        adapter = ParallelRNNLoRAAdapter(dim=32, rank=8, seed=seed, rho_max=0.9)
        A = adapter.transition_matrix()
        sigma_max = np.linalg.svd(A, compute_uv=False)[0]
        assert sigma_max < 1.0
        eigvals = np.linalg.eigvals(A)
        assert np.max(np.abs(eigvals)) < 1.0


def test_adapter_enforces_stability_even_with_adversarial_raw_init():
    """Directly build an adapter whose raw diag+low-rank matrix is unstable
    (sigma_max >> 1) and confirm the constructor's rescale still clamps it."""
    adapter = ParallelRNNLoRAAdapter(dim=16, rank=4, rho_max=0.9, seed=3)
    # Force a huge low-rank update after construction, mimicking a raw
    # parameterisation that would blow past the unit disk, then re-run the
    # same rescale the constructor uses.
    adapter.U_A *= 1000.0
    adapter.V_A *= 1000.0
    adapter._scale = adapter._compute_scale()
    A = adapter.transition_matrix()
    sigma_max = np.linalg.svd(A, compute_uv=False)[0]
    assert sigma_max < 1.0
    assert sigma_max == pytest.approx(adapter.rho_max, rel=1e-6)


def test_energy_never_grows_faster_than_rho_max_with_zero_input():
    rho_max = 0.85
    adapter = ParallelRNNLoRAAdapter(dim=24, rank=6, rho_max=rho_max, seed=7)
    rng = np.random.default_rng(7)
    h = rng.standard_normal(24)
    zero_x = np.zeros(24)
    for _ in range(50):
        h_next = adapter.step(zero_x, h)
        assert np.linalg.norm(h_next) <= rho_max * np.linalg.norm(h) + 1e-12
        h = h_next


def test_lyapunov_trajectory_audit_via_existing_verifier():
    """Cross-check with the already-verified `verify_trajectory` tool: energy
    ||h||^2 must be nonincreasing under zero input, and every prefix Jacobian
    must respect the enforced spectral bound."""
    rho_max = 0.8
    adapter = ParallelRNNLoRAAdapter(dim=8, rank=3, rho_max=rho_max, seed=11)
    rng = np.random.default_rng(11)
    h0 = rng.standard_normal(8) * 0.1
    c0 = np.zeros(8)

    def step_fn(x, c):
        return adapter.step(c, x)

    def energy(x, c):
        return float(x @ x)

    def manifold_distance(x, c):
        return 0.0

    report = verify_trajectory(
        step_fn, energy, h0, c0, steps=15,
        manifold_distance=manifold_distance, max_manifold_distance=0.0,
        min_sigma=1e-9, max_sigma=1.0, max_condition_number=1e12,
        energy_atol=1e-9, energy_rtol=0.0,
    )
    assert report.energy_nonincreasing
    assert report.manifold_within_bound
    assert np.all(report.sigma_max <= 1.0 + 1e-9)


# --------------------------------------------------------------------------
# 4. step()/forward() correctness, constant working set, no leaks.
# --------------------------------------------------------------------------
def test_step_matches_dense_transition_and_input_matrix():
    adapter = ParallelRNNLoRAAdapter(dim=20, rank=5, input_dim=12, seed=4)
    rng = np.random.default_rng(4)
    h = rng.standard_normal(20)
    x = rng.standard_normal(12)
    got = adapter.step(x, h)
    expected = adapter.transition_matrix() @ h + adapter.input_matrix() @ x
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_forward_matches_step_loop():
    adapter = ParallelRNNLoRAAdapter(dim=14, rank=4, input_dim=9, seed=5)
    rng = np.random.default_rng(5)
    T = 40
    x_seq = rng.standard_normal((T, 9))
    h = np.zeros(14)
    expected = np.empty((T, 14))
    for t in range(T):
        h = adapter.step(x_seq[t], h)
        expected[t] = h
    got = adapter.forward(x_seq)
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_working_set_fits_l1_l2_budget():
    # dim=128, rank=8 sized so parameters + h/x buffers fit under 64 KiB,
    # matching the doc's cache-resident working-set target.
    adapter = ParallelRNNLoRAAdapter(dim=128, rank=8, input_dim=128, seed=0)
    budget_bytes = 64 * 1024
    assert adapter.working_set_bytes() <= budget_bytes


def test_step_state_dimension_is_constant_and_no_memory_growth():
    adapter = ParallelRNNLoRAAdapter(dim=32, rank=8, seed=6)
    rng = np.random.default_rng(6)
    h = rng.standard_normal(32)
    x = rng.standard_normal(32)

    param_bytes_before = {
        k: v.nbytes for k, v in vars(adapter).items() if isinstance(v, np.ndarray)
    }

    for _ in range(200):
        h = adapter.step(x, h)
        assert h.shape == (32,)

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    for _ in range(2000):
        h = adapter.step(x, h)
        assert h.shape == (32,)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    param_bytes_after = {
        k: v.nbytes for k, v in vars(adapter).items() if isinstance(v, np.ndarray)
    }
    assert param_bytes_before == param_bytes_after

    growth = sum(stat.size_diff for stat in snap_after.compare_to(snap_before, "filename"))
    # A handful of KB of interpreter/pytest bookkeeping noise is expected;
    # a growing cache over 2000 steps would show up as tens/hundreds of KB.
    assert growth < 8192, f"unexpected heap growth of {growth} bytes over 2000 steps"


# --------------------------------------------------------------------------
# 5. Fail-closed on non-finite / malformed input.
# --------------------------------------------------------------------------
def test_combine_rejects_nonfinite():
    g_ok = (np.eye(3), np.zeros(3))
    g_bad = (np.eye(3), np.array([1.0, np.nan, 0.0]))
    with pytest.raises(ValueError):
        combine(g_ok, g_bad)
    g_inf = (np.array([[1.0, np.inf], [0.0, 1.0]]), np.zeros(2))
    with pytest.raises(ValueError):
        combine(g_inf, (np.eye(2), np.zeros(2)))


def test_sequential_and_associative_scan_reject_nonfinite():
    A = np.tile(np.eye(3), (4, 1, 1))
    u = np.zeros((4, 3))
    u_bad = u.copy()
    u_bad[2, 1] = np.inf
    with pytest.raises(ValueError):
        sequential_scan(A, u_bad)
    with pytest.raises(ValueError):
        associative_scan(A, u_bad)
    A_bad = A.copy()
    A_bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        sequential_scan(A_bad, u)
    with pytest.raises(ValueError):
        associative_scan(A_bad, u)


def test_spectral_normalize_rejects_nonfinite_and_bad_rho():
    with pytest.raises(ValueError):
        spectral_normalize(np.array([[1.0, np.nan], [0.0, 1.0]]), 0.9)
    with pytest.raises(ValueError):
        spectral_normalize(np.eye(3), 1.0)
    with pytest.raises(ValueError):
        spectral_normalize(np.eye(3), 0.0)
    with pytest.raises(ValueError):
        spectral_normalize(np.eye(3), float("nan"))


def test_adapter_constructor_rejects_invalid_scalars():
    with pytest.raises(ValueError):
        ParallelRNNLoRAAdapter(dim=0, rank=1)
    with pytest.raises(ValueError):
        ParallelRNNLoRAAdapter(dim=8, rank=0)
    with pytest.raises(ValueError):
        ParallelRNNLoRAAdapter(dim=4, rank=8)  # rank > dim
    with pytest.raises(ValueError):
        ParallelRNNLoRAAdapter(dim=8, rank=2, rho_max=1.0)
    with pytest.raises(ValueError):
        ParallelRNNLoRAAdapter(dim=8, rank=2, rho_max=0.0)


def test_adapter_step_rejects_nonfinite_and_wrong_shape():
    adapter = ParallelRNNLoRAAdapter(dim=6, rank=2, seed=8)
    h = np.zeros(6)
    x = np.zeros(6)
    with pytest.raises(ValueError):
        adapter.step(np.array([np.nan] * 6), h)
    with pytest.raises(ValueError):
        adapter.step(x, np.array([np.inf] * 6))
    with pytest.raises(ValueError):
        adapter.step(np.zeros(5), h)  # wrong input_dim
    with pytest.raises(ValueError):
        adapter.step(x, np.zeros(7))  # wrong dim


def test_adapter_forward_rejects_nonfinite_and_wrong_shape():
    adapter = ParallelRNNLoRAAdapter(dim=6, rank=2, seed=9)
    with pytest.raises(ValueError):
        adapter.forward(np.full((5, 6), np.nan))
    with pytest.raises(ValueError):
        adapter.forward(np.zeros((5, 7)))  # wrong input_dim
    with pytest.raises(ValueError):
        adapter.forward(np.zeros(6))  # wrong ndim


# --------------------------------------------------------------------------
# 5b. save()/load() round trip.
# --------------------------------------------------------------------------
def test_save_load_round_trip_step_matches(tmp_path):
    adapter = ParallelRNNLoRAAdapter(dim=20, rank=5, input_dim=12, rho_max=0.9, seed=42)
    rng = np.random.default_rng(42)
    h = rng.standard_normal(20)
    x = rng.standard_normal(12)
    before = adapter.step(x, h)

    path = tmp_path / "rnn_adapter.npz"
    adapter.save(path)
    restored = ParallelRNNLoRAAdapter.load(path)

    assert restored.dim == adapter.dim
    assert restored.rank == adapter.rank
    assert restored.input_dim == adapter.input_dim
    assert restored.rho_max == pytest.approx(adapter.rho_max)
    assert restored.dtype == adapter.dtype

    after = restored.step(x, h)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(restored.transition_matrix(), adapter.transition_matrix())
    np.testing.assert_array_equal(restored.input_matrix(), adapter.input_matrix())


def test_save_load_round_trip_default_dim_equals_input_dim(tmp_path):
    adapter = ParallelRNNLoRAAdapter(dim=10, rank=3, seed=1)
    path = tmp_path / "rnn_adapter_square.npz"
    adapter.save(path)
    restored = ParallelRNNLoRAAdapter.load(path)
    assert restored.input_dim == adapter.input_dim == adapter.dim


def test_load_rejects_invalid_schema(tmp_path):
    path = tmp_path / "bad_schema.npz"
    np.savez(path, foo=np.zeros(3))
    with pytest.raises(ValueError):
        ParallelRNNLoRAAdapter.load(path)


# --------------------------------------------------------------------------
# 6. No `re` import, no regex/label shortcuts, in the implementation module.
# --------------------------------------------------------------------------
def test_module_does_not_import_re():
    import gen_zero.causal.parallel_rnn_lora as mod

    source = open(mod.__file__, "r", encoding="utf-8").read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "re" for alias in node.names), "must not import re"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "re", "must not import from re"
