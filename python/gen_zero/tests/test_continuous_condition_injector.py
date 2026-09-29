"""Tests for Route 2 (Iterative External Forcing): continuous condition
reinjection layered on top of `FractalMultiscaleEngine`.

Covers: (1) condition-cache 16KB byte-budget accounting, (2) the
fixed-vs-data-dependent schedule distinction that determines
`associative_scan` compatibility, (3) the PAWS/VitaminC repulsor bridge,
(4) the per-segment Lyapunov (energy-dissipation) guarantee under reinjection,
(5) the macro-scale bifurcation effect: a repulsor reinjected mid-relaxation
keeps the trajectory off an attractor the unforced run settles into exactly,
(6) fail-closed rejection of malformed schedules and budgets.
"""
from __future__ import annotations

import numpy as np
import pytest

from gen_zero.causal.continuous_condition_injector import (
    L2_CONDITION_CACHE_BYTES_DEFAULT,
    MACRO_TAU,
    MESO_TAU,
    MICRO_TAU,
    ContinuousConditionInjector,
    LyapunovViolation,
    ReinjectionEvent,
    build_fixed_reinjection_schedule,
    is_schedule_parallel_scan_compatible,
    macro_repulsor_from_counterfactual,
    symbolic_violation_to_repulsor,
    verify_condition_cache_bound,
)
from gen_zero.causal.counterfactual_constraint_verifier import CounterfactualConstraintVerifier
from gen_zero.causal.fractal_multiscale_engine import FractalMultiscaleEngine

DIM = 16


def _engine(**overrides) -> FractalMultiscaleEngine:
    params = dict(dim=DIM, tau_macro=8, dt_micro=0.125, omega=1.0,
                  gamma_macro=2.0, gamma_micro=2.0, repulsion_threshold=0.5,
                  residual_tol=1e-5, max_macro_steps=2, max_micro_steps=300)
    params.update(overrides)
    return FractalMultiscaleEngine(**params)


# ---------------------------------------------------------------------------
# Condition cache byte budget
# ---------------------------------------------------------------------------

class TestConditionCacheBudget:
    def test_worked_example_fits_exactly_16kb(self):
        # dim=256, 8 slots (3 scale centers + up to 5 repulsor rows), float64:
        # 256 * 8 * 8 = 16384 bytes = the default L2 budget, exactly.
        report = verify_condition_cache_bound(256, 8)
        assert report.bytes_used == L2_CONDITION_CACHE_BYTES_DEFAULT
        assert report.within_bound

    def test_one_more_slot_exceeds_the_budget(self):
        report = verify_condition_cache_bound(256, 9)
        assert not report.within_bound
        assert report.bytes_used > report.l2_bytes

    def test_rejects_non_positive_dim(self):
        with pytest.raises(ValueError):
            verify_condition_cache_bound(0, 4)

    def test_rejects_non_positive_slots(self):
        with pytest.raises(ValueError):
            verify_condition_cache_bound(64, 0)


# ---------------------------------------------------------------------------
# Schedule construction and parallel-scan compatibility
# ---------------------------------------------------------------------------

class TestSchedule:
    def test_fixed_schedule_cycles_macro_meso_micro_in_order(self):
        rng = np.random.default_rng(0)
        centers = [rng.normal(size=DIM) for _ in range(5)]
        schedule = build_fixed_reinjection_schedule(DIM, centers)
        assert [e.scale for e in schedule] == ["macro", "meso", "micro", "macro", "meso"]
        assert [e.tau for e in schedule] == [MACRO_TAU, MESO_TAU, MICRO_TAU, MACRO_TAU, MESO_TAU]
        assert [e.step_index for e in schedule] == [0, 1, 2, 3, 4]

    def test_fixed_schedule_is_parallel_scan_compatible(self):
        assert is_schedule_parallel_scan_compatible(schedule_is_data_dependent=False)

    def test_residual_triggered_schedule_is_not_parallel_scan_compatible(self):
        assert not is_schedule_parallel_scan_compatible(schedule_is_data_dependent=True)

    def test_empty_centers_rejected(self):
        with pytest.raises(ValueError):
            build_fixed_reinjection_schedule(DIM, [])

    def test_mismatched_repulsor_sets_length_rejected(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            build_fixed_reinjection_schedule(DIM, [rng.normal(size=DIM)], repulsor_sets=[])

    def test_event_digest_changes_when_content_changes(self):
        rng = np.random.default_rng(0)
        c1 = rng.normal(size=DIM)
        c2 = rng.normal(size=DIM)
        e1 = ReinjectionEvent.make("macro", MACRO_TAU, 0, c1)
        e2 = ReinjectionEvent.make("macro", MACRO_TAU, 0, c2)
        assert e1.digest != e2.digest

    def test_scale_validation_is_fail_closed(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            ReinjectionEvent.make("nano", 1, 0, rng.normal(size=DIM))


# ---------------------------------------------------------------------------
# PAWS / VitaminC repulsor bridge
# ---------------------------------------------------------------------------

class TestCounterfactualBridge:
    def test_macro_repulsor_shape_matches_engine_expectation(self):
        rng = np.random.default_rng(2)
        verifier = CounterfactualConstraintVerifier()
        premise, hyp = rng.normal(size=DIM), rng.normal(size=DIM)
        cf_premise, cf_hyp = rng.normal(size=DIM), rng.normal(size=DIM)
        delta_perp, penalty, neg = verifier.repulsion(premise, hyp, cf_premise, cf_hyp)
        repulsor = macro_repulsor_from_counterfactual(delta_perp)
        assert repulsor.shape == (1, DIM)
        # Must be directly usable as `repulsors` in macro_prune without further work.
        engine = _engine()
        result = engine.macro_prune(rng.normal(size=DIM) * 0.1, delta_perp, delta_perp, repulsors=repulsor)
        assert isinstance(result.repulsion_score, float)

    def test_symbolic_violation_with_zero_penalty_yields_no_repulsor(self):
        rng = np.random.default_rng(0)
        assert symbolic_violation_to_repulsor(rng.normal(size=DIM), penalty=0.0) is None

    def test_symbolic_violation_with_positive_penalty_yields_repulsor_row(self):
        rng = np.random.default_rng(0)
        r = symbolic_violation_to_repulsor(rng.normal(size=DIM), penalty=1.0)
        assert r.shape == (1, DIM)

    def test_symbolic_violation_rejects_negative_penalty(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            symbolic_violation_to_repulsor(rng.normal(size=DIM), penalty=-0.1)


# ---------------------------------------------------------------------------
# Per-segment Lyapunov guarantee under reinjection
# ---------------------------------------------------------------------------

class TestLyapunovUnderReinjection:
    @pytest.mark.parametrize("seed", range(5))
    def test_every_segment_is_non_increasing_against_its_own_center(self, seed):
        rng = np.random.default_rng(seed)
        engine = _engine()
        injector = ContinuousConditionInjector(engine)
        q0, p0 = rng.normal(size=DIM), rng.normal(size=DIM)
        centers = [rng.normal(size=DIM) * 2 for _ in range(3)]
        schedule = build_fixed_reinjection_schedule(DIM, centers)
        # Raises LyapunovViolation internally if any segment's own H rises;
        # reaching a normal return is the assertion.
        result = injector.think_with_reinjection(q0, p0, schedule)
        assert result.total_steps > 0
        assert result.cache_budget.within_bound

    def test_converges_to_the_last_reinjected_center(self):
        rng = np.random.default_rng(7)
        engine = _engine(residual_tol=1e-6)
        injector = ContinuousConditionInjector(engine)
        q0, p0 = rng.normal(size=DIM) * 0.1, rng.normal(size=DIM) * 0.1
        centers = [rng.normal(size=DIM), rng.normal(size=DIM), rng.normal(size=DIM)]
        schedule = build_fixed_reinjection_schedule(DIM, centers)
        result = injector.think_with_reinjection(q0, p0, schedule, max_steps_per_segment=2000)
        assert result.final_residual < 1e-4
        assert np.linalg.norm(result.q - centers[-1]) < 1e-2


# ---------------------------------------------------------------------------
# Macro-scale bifurcation: reinjected repulsor prunes the wrong attractor
# ---------------------------------------------------------------------------

class TestBifurcation:
    def test_unforced_run_settles_exactly_at_the_wrong_attractor(self):
        rng = np.random.default_rng(1)
        engine = _engine()
        c_wrong = rng.normal(size=DIM)
        c_wrong /= np.linalg.norm(c_wrong)
        q0 = rng.normal(size=DIM) * 0.1
        p0 = c_wrong * 1.5
        baseline = engine.think(q0, p0, c_wrong, repulsors=None)
        assert not baseline.pruned
        assert baseline.residual < 1e-4

    def test_reinjected_repulsor_prunes_before_settling_at_the_same_attractor(self):
        rng = np.random.default_rng(1)
        engine = _engine()
        c_wrong = rng.normal(size=DIM)
        c_wrong /= np.linalg.norm(c_wrong)
        q0 = rng.normal(size=DIM) * 0.1
        p0 = c_wrong * 1.5
        repulsors = np.stack([c_wrong])
        forced = engine.think(q0, p0, c_wrong, repulsors=repulsors)
        assert forced.pruned
        assert forced.residual > 0.1
        assert np.linalg.norm(forced.q - c_wrong) > 0.1


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_empty_schedule_rejected(self):
        engine = _engine()
        injector = ContinuousConditionInjector(engine)
        with pytest.raises(ValueError):
            injector.think_with_reinjection(np.zeros(DIM), np.zeros(DIM), [])

    def test_dim_mismatch_between_schedule_and_engine_rejected(self):
        engine = _engine()
        injector = ContinuousConditionInjector(engine)
        wrong_dim_event = ReinjectionEvent.make("macro", MACRO_TAU, 0, np.zeros(DIM + 1))
        with pytest.raises(ValueError):
            injector.think_with_reinjection(np.zeros(DIM), np.zeros(DIM), [wrong_dim_event])

    def test_schedule_exceeding_condition_cache_budget_rejected(self):
        engine = _engine(dim=4096)
        injector = ContinuousConditionInjector(engine, l2_bytes=1024, max_slots=64)
        rng = np.random.default_rng(0)
        centers = [rng.normal(size=4096) for _ in range(3)]
        schedule = build_fixed_reinjection_schedule(4096, centers)
        with pytest.raises(ValueError):
            injector.think_with_reinjection(np.zeros(4096), np.zeros(4096), schedule)
