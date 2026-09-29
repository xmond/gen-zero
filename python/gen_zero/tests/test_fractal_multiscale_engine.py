"""Tests for the cache-oblivious fractal multiscale dynamical thinking engine.

Covers: (1) multiscale vs. single-scale convergence equivalence with a large
step-count saving, (2) exact orthogonal-equivariance of the decision and the
trajectory under any Q with Q^T Q = I, (3) L1 (32KB) working-set bound
verification and cache-oblivious block partitioning, (4) fail-closed rejection
of malformed inputs and parameters.
"""
from __future__ import annotations

import numpy as np
import pytest

from gen_zero.causal.fractal_multiscale_engine import (
    FractalMultiscaleEngine,
    counterfactual_repulsion_score,
    recursive_cache_oblivious_partition,
    verify_cache_bound,
)

DIM = 6


def _orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    assert np.allclose(q.T @ q, np.eye(dim), atol=1e-10)
    return q


def _engine(**overrides) -> FractalMultiscaleEngine:
    # gamma = 2*omega is critical damping (fastest non-oscillatory decay, the
    # Lyapunov-stable regime of design-doc section 5.3); tau_macro=8 is dyadic
    # (2^3, matching the design doc's own "tau=4 or tau=8" macro scale) and
    # keeps dt_macro=1.0, comfortably inside the dt*omega<2 stability bound.
    params = dict(dim=DIM, tau_macro=8, dt_micro=0.125, omega=1.0,
                  gamma_macro=2.0, gamma_micro=2.0, repulsion_threshold=0.5,
                  residual_tol=1e-5, max_macro_steps=2, max_micro_steps=300)
    params.update(overrides)
    return FractalMultiscaleEngine(**params)


# ---------------------------------------------------------------------------
# Cache-oblivious L1 (32KB) working-set bound
# ---------------------------------------------------------------------------

class TestCacheBound:
    def test_small_batch_fits_within_l1_bound(self):
        report = verify_cache_bound(DIM, 8)
        assert report.within_bound
        assert report.working_set_bytes <= 32768

    def test_huge_batch_exceeds_l1_bound(self):
        report = verify_cache_bound(DIM, 100_000)
        assert not report.within_bound
        assert report.working_set_bytes > report.l1_bytes

    def test_recursive_partition_bounds_every_block(self):
        blocks = recursive_cache_oblivious_partition(DIM, 100_000)
        assert sum(blocks) == 100_000
        for size in blocks:
            report = verify_cache_bound(DIM, size)
            assert report.within_bound, f"block of size {size} exceeds the L1 bound"

    def test_partition_is_a_binary_recursive_bisection(self):
        # Every block size must be reachable by repeated halving of the batch.
        blocks = recursive_cache_oblivious_partition(DIM, 37)
        assert sum(blocks) == 37
        assert all(size >= 1 for size in blocks)

    def test_partition_fails_closed_when_dimension_alone_exceeds_bound(self):
        with pytest.raises(ValueError):
            recursive_cache_oblivious_partition(dim=10_000_000, batch_size=4)

    def test_cache_bound_rejects_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            verify_cache_bound(0, 8)
        with pytest.raises(ValueError):
            verify_cache_bound(DIM, 0)
        with pytest.raises(ValueError):
            verify_cache_bound(DIM, -3)


# ---------------------------------------------------------------------------
# Fail-closed construction and input validation
# ---------------------------------------------------------------------------

class TestFailClosed:
    @pytest.mark.parametrize("bad_kwargs", [
        dict(dim=0),
        dict(dt_micro=0.0),
        dict(dt_micro=-1.0),
        dict(omega=0.0),
        dict(gamma_macro=-1.0),
        dict(gamma_micro=-1.0),
        dict(repulsion_threshold=0.0),
        dict(repulsion_threshold=1.5),
        dict(residual_tol=0.0),
        dict(max_macro_steps=3),
        dict(max_macro_steps=0),
        dict(max_micro_steps=0),
        dict(tau_macro=16, dt_micro=0.5, omega=1.0),  # dt_macro*omega=8 >= 2 -> unstable
        dict(tau_macro=10),  # not a power of two -> not a dyadic scale
        dict(tau_macro=1),  # 2^0 collapses macro into the micro scale
    ])
    def test_rejects_invalid_construction_params(self, bad_kwargs):
        with pytest.raises(ValueError):
            _engine(**bad_kwargs)

    def test_rejects_nonfinite_state(self):
        engine = _engine()
        c = np.zeros(DIM)
        with pytest.raises(ValueError):
            engine.think(np.full(DIM, np.nan), np.zeros(DIM), c)
        with pytest.raises(ValueError):
            engine.think(np.zeros(DIM), np.full(DIM, np.inf), c)

    def test_rejects_shape_mismatch(self):
        engine = _engine()
        c = np.zeros(DIM)
        with pytest.raises(ValueError):
            engine.think(np.zeros(DIM + 1), np.zeros(DIM), c)
        with pytest.raises(ValueError):
            engine.think(np.zeros(DIM), np.zeros(DIM), np.zeros(DIM + 1))

    def test_rejects_malformed_repulsors(self):
        engine = _engine()
        c = np.zeros(DIM)
        q0, p0 = np.ones(DIM) * 3.0, np.zeros(DIM)
        with pytest.raises(ValueError):
            engine.think(q0, p0, c, repulsors=np.zeros((0, DIM)))
        with pytest.raises(ValueError):
            engine.think(q0, p0, c, repulsors=np.zeros((2, DIM + 1)))
        with pytest.raises(ValueError):
            engine.think(q0, p0, c, repulsors=np.full((2, DIM), np.nan))

    def test_repulsion_score_rejects_malformed_repulsors(self):
        with pytest.raises(ValueError):
            counterfactual_repulsion_score(np.zeros(DIM), np.zeros((0, DIM)))
        with pytest.raises(ValueError):
            counterfactual_repulsion_score(np.zeros(DIM), np.zeros((2, DIM + 1)))


# ---------------------------------------------------------------------------
# Two-phase macro pruning / micro relaxation semantics
# ---------------------------------------------------------------------------

class TestTwoPhaseThinking:
    def test_no_repulsors_converges_in_micro_phase(self):
        engine = _engine()
        c = np.zeros(DIM)
        q0 = np.array([5.0, -3.0, 2.0, 1.0, -1.0, 0.5])
        p0 = np.zeros(DIM)
        result = engine.think(q0, p0, c)
        assert not result.pruned
        assert result.micro is not None
        assert result.micro.converged
        assert result.residual < 1e-5
        # Lyapunov dissipation: H must never rise across a damped leapfrog step,
        # at either scale (section 5.3's H_dot <= 0 contract).
        assert np.all(np.diff(result.macro.energy_trace) <= 1e-9)
        assert np.all(np.diff(result.micro.energy_trace) <= 1e-9)

    def test_aligned_repulsor_triggers_backtrack_pruning_within_two_steps(self):
        engine = _engine()
        c = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q0 = c.copy()
        p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        repulsors = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        result = engine.think(q0, p0, c, repulsors=repulsors)
        assert result.pruned
        assert result.macro.steps <= 2
        assert result.macro.repulsion_score > engine.repulsion_threshold
        assert result.micro is None

    def test_orthogonal_repulsor_does_not_trigger_pruning(self):
        engine = _engine()
        c = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q0 = c.copy()
        p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        repulsors = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        result = engine.think(q0, p0, c, repulsors=repulsors)
        assert not result.pruned
        assert result.micro is not None


# ---------------------------------------------------------------------------
# Multiscale convergence equivalence with far fewer total steps
# ---------------------------------------------------------------------------

class TestMultiscaleSpeedup:
    def test_multiscale_matches_single_scale_equilibrium_with_fewer_steps(self):
        engine = _engine()
        c = np.array([20.0, -15.0, 10.0, -5.0, 8.0, -3.0])
        q0 = np.zeros(DIM)
        p0 = np.zeros(DIM)

        multiscale = engine.think(q0, p0, c)
        baseline = engine.single_scale_baseline(q0, p0, c, max_steps=100_000)

        assert not multiscale.pruned
        assert multiscale.micro.converged
        assert baseline.converged
        # Same dynamical fixed point: both settle at the frozen context c.
        assert np.allclose(multiscale.q, c, atol=1e-3)
        assert np.allclose(baseline.q, c, atol=1e-3)
        assert multiscale.residual < 1e-5
        assert baseline.residual < 1e-5

        assert multiscale.total_steps < baseline.steps
        # "Significantly" fewer: the macro phase's two large-tau steps buy a real,
        # reproducible >= 15% cut in total integration steps (empirically ~30-40%
        # across random contexts; 15% is a safety margin, not the measured value).
        assert multiscale.total_steps <= 0.85 * baseline.steps
        # Lyapunov dissipation must hold in both phases, not just at the endpoints.
        assert np.all(np.diff(multiscale.macro.energy_trace) <= 1e-9)
        assert np.all(np.diff(multiscale.micro.energy_trace) <= 1e-9)


# ---------------------------------------------------------------------------
# Orthogonal rotation equivariance (Q^T Q = I)
# ---------------------------------------------------------------------------

class TestRotationEquivariance:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_free_convergence_is_equivariant(self, seed):
        q_mat = _orthogonal_matrix(DIM, seed)
        engine = _engine()
        c = np.array([6.0, -4.0, 3.0, -2.0, 1.0, -1.0])
        q0 = np.array([1.0, 2.0, -1.0, 0.5, -0.5, 0.25])
        p0 = np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.0])

        base = engine.think(q0, p0, c)
        rotated = engine.think(q_mat @ q0, q_mat @ p0, q_mat @ c)

        assert rotated.pruned == base.pruned
        assert rotated.total_steps == base.total_steps
        assert np.allclose(rotated.q, q_mat @ base.q, atol=1e-8)
        assert np.allclose(rotated.p, q_mat @ base.p, atol=1e-8)
        assert rotated.residual == pytest.approx(base.residual, abs=1e-8)

    @pytest.mark.parametrize("seed", [3, 4])
    def test_pruning_decision_is_equivariant(self, seed):
        q_mat = _orthogonal_matrix(DIM, seed)
        engine = _engine()
        c = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q0 = c.copy()
        p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        repulsors = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

        base = engine.think(q0, p0, c, repulsors=repulsors)
        rotated = engine.think(q_mat @ q0, q_mat @ p0, q_mat @ c,
                               repulsors=repulsors @ q_mat.T)

        assert rotated.pruned == base.pruned
        assert rotated.macro.steps == base.macro.steps
        assert rotated.macro.repulsion_score == pytest.approx(base.macro.repulsion_score, abs=1e-8)
        assert np.allclose(rotated.q, q_mat @ base.q, atol=1e-8)
        assert np.allclose(rotated.p, q_mat @ base.p, atol=1e-8)


# ---------------------------------------------------------------------------
# Batched, cache-oblivious block processing
# ---------------------------------------------------------------------------

class TestThinkBatch:
    def test_batch_matches_individual_calls_and_respects_cache_bound(self):
        engine = _engine(l1_bytes=1024)  # force a small L1 bound -> forces splitting
        c = np.zeros(DIM)
        rng = np.random.default_rng(42)
        q0_batch = rng.normal(size=(9, DIM))
        p0_batch = np.zeros((9, DIM))

        blocks = recursive_cache_oblivious_partition(DIM, 9, l1_bytes=1024)
        assert len(blocks) > 1  # confirms the small bound actually forced splitting
        for size in blocks:
            assert verify_cache_bound(DIM, size, l1_bytes=1024).within_bound

        results = engine.think_batch(q0_batch, p0_batch, c)
        assert len(results) == 9
        for i, result in enumerate(results):
            individual = engine.think(q0_batch[i], p0_batch[i], c)
            assert result.total_steps == individual.total_steps
            assert np.allclose(result.q, individual.q)

    def test_batch_rejects_shape_mismatch(self):
        engine = _engine()
        c = np.zeros(DIM)
        with pytest.raises(ValueError):
            engine.think_batch(np.zeros((3, DIM)), np.zeros((4, DIM)), c)
        with pytest.raises(ValueError):
            engine.think_batch(np.zeros((0, DIM)), np.zeros((0, DIM)), c)
