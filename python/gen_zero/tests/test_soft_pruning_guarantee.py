"""Regression tests for the catastrophic over-pruning fix.

Diagnosed in `benchmarks/results_track_c/cpu_dynamics_clean_eval.json`
(2026-09-23 run): with `continuous_causal_reasoning_expert` mounted, 67.33%
of true MultiNLI candidates were hard-pruned by
`BifurcatedFractalEngine.race()`'s repulsion-threshold gate and their score
was floored to a fixed non-physical constant, collapsing MultiNLI accuracy
from 83.33% (`no_expert`) to 4.33% (`full_calibrated_dynamics`).

Two independent things are covered here:

1. `BifurcatedFractalEngine.race()` never lets a branch pool with at least one
   finite-cost branch go extinct: a hard repulsion-threshold wipeout is
   rescued by the least-cost branch instead of reporting zero survivors.
2. `continuous_causal_reasoning_expert`'s defense-in-depth fallback for the
   (now much rarer) case where `race_trace.pruned_all` is still True never
   assigns a fixed floor score: it prices the failure as a bounded,
   continuous Lyapunov-style energy penalty and keeps running the micro
   projection step, so one degenerate candidate can never swamp a K-ary
   comparison the way a `-100`/`-1e9` floor did.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from gen_zero.causal.bifurcated_fractal_engine import (
    BifurcatedFractalEngine,
    BranchPool,
    RaceTrace,
)
from gen_zero.causal.continuous_causal_reasoning_expert import (
    RACE_PENALTY_SCALE,
    STATUS_PRUNED,
    continuous_causal_reasoning_expert,
)

DIM = 16


def _competing_candidates(rng, k=3, eps=0.02):
    """K candidates whose directions from q0 are within a few degrees of each
    other -- the "origin saddle" / near-duplicate-candidate geometry that
    `continuous_causal_reasoning_expert`'s own docstring identifies as the
    case plain cosine similarity (and the old hard repulsion gate) cannot
    resolve cleanly. Every candidate registers every other as a repulsor
    that is almost perfectly aligned with its own direction, so the hard
    repulsion threshold (default 0.5 on cos^2, i.e. ~45 degrees) is
    exceeded on essentially every macro branch.
    """
    q0 = rng.normal(size=DIM)
    base = rng.normal(size=DIM)
    base /= np.linalg.norm(base)
    cands = np.stack([q0 + base + eps * rng.normal(size=DIM) for _ in range(k)])
    return q0, cands


class TestEngineNeverExtinguishesOnFiniteCost:
    def test_high_mutual_repulsion_pool_is_rescued_not_wiped_out(self):
        rng = np.random.default_rng(11)
        q0, cands = _competing_candidates(rng, k=3)
        from gen_zero.causal.parallel_rnn_lora import ParallelRNNLoRAAdapter
        adapter = ParallelRNNLoRAAdapter(DIM, 2, rho_max=0.5, seed=0)
        eng = BifurcatedFractalEngine(DIM, adapter, repulsion_threshold=0.5)

        c0 = cands[0]
        p0 = c0 - q0
        repulsors = cands[1:] - q0
        pool, _decision = eng.bifurcate(q0, p0, c0, repulsors=repulsors)
        out, trace = eng.race(pool, c0, repulsors=repulsors)

        # The construction must actually exercise the hard gate, or this test
        # is vacuous.
        assert trace.rescued, (
            "test construction did not trigger the hard repulsion gate; "
            f"alive_history={trace.alive_history}"
        )
        assert not trace.pruned_all
        assert int(out.alive.sum()) == 1


class TestExpertScoresStayFiniteAndBounded:
    def test_no_candidate_ever_gets_a_fixed_floor_score(self):
        rng = np.random.default_rng(7)
        q0, cands = _competing_candidates(rng, k=3)
        result = continuous_causal_reasoning_expert(q0, cands, seed=1)

        assert np.all(np.isfinite(result.scores)), result.scores
        # The old bug: a hard-pruned candidate's score was exactly -100.0
        # (previously -1e9 per the diagnosis), regardless of how close its
        # micro-refined state actually landed to its target.
        assert not np.any(np.isclose(result.scores, -100.0))
        assert not np.any(np.isclose(result.scores, -1e9))
        for score, trace in zip(result.scores, result.traces):
            # Physical bound: score is -residual, minus at most a bounded
            # penalty (0 in the normal path, < RACE_PENALTY_SCALE in the
            # race-failure fallback). Never an unrelated fixed constant.
            assert score >= -(trace.micro_residual_to_target + RACE_PENALTY_SCALE) - 1e-9
            assert np.isfinite(trace.score)

    def test_repulsion_ablation_still_finite_and_bounded(self):
        # Same competing geometry, but with the macro repulsion gate disabled
        # entirely -- a different code path (identity condition set), must
        # be just as finite and bounded.
        rng = np.random.default_rng(23)
        q0, cands = _competing_candidates(rng, k=4)
        result = continuous_causal_reasoning_expert(q0, cands, seed=2, enable_repulsion=False)
        assert np.all(np.isfinite(result.scores))
        assert not np.any(np.isclose(result.scores, -100.0))

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_many_high_repulsion_draws_never_floor(self, seed):
        rng = np.random.default_rng(1000 + seed)
        q0, cands = _competing_candidates(rng, k=3, eps=0.01)
        result = continuous_causal_reasoning_expert(q0, cands, seed=seed)
        assert np.all(np.isfinite(result.scores))
        assert np.all(np.abs(result.scores) < 1e6)


class TestScoreContinuityAcrossThresholdCrossing:
    def test_score_has_no_100_scale_discontinuity_as_repulsion_varies(self):
        """Sweep the angular separation between two candidates from wide (no
        hard prune) to near-zero (hard prune on essentially every branch).
        The old code jumped from a `-residual` score (O(0.01-1)) straight to
        the `-100` floor the instant the hard gate tripped: a ~100-unit
        discontinuity. The fix must keep the adjacent-step score change
        small and bounded everywhere on the sweep.
        """
        rng = np.random.default_rng(99)
        q0 = rng.normal(size=DIM)
        base = rng.normal(size=DIM)
        base /= np.linalg.norm(base)
        # An orthogonal wobble direction so the angle to `base` is controllable.
        wobble = rng.normal(size=DIM)
        wobble -= float(wobble @ base) * base
        wobble /= np.linalg.norm(wobble)

        scores_of_candidate_0 = []
        angles_deg = np.linspace(0.5, 60.0, 40)
        for deg in angles_deg:
            theta = np.deg2rad(deg)
            c0 = q0 + base
            c1 = q0 + np.cos(theta) * base + np.sin(theta) * wobble
            cands = np.stack([c0, c1])
            result = continuous_causal_reasoning_expert(q0, cands, seed=0)
            assert np.all(np.isfinite(result.scores))
            scores_of_candidate_0.append(float(result.scores[0]))

        deltas = np.abs(np.diff(np.array(scores_of_candidate_0)))
        assert deltas.max() < 5.0, (
            f"max adjacent score jump {deltas.max():.3f} across the sweep is on the "
            f"same order as the old -100 floor discontinuity"
        )


class TestPrunedAllFallbackIsHonest:
    """Directly exercises `continuous_causal_reasoning_expert`'s handling of
    `race_trace.pruned_all` by forcing `BifurcatedFractalEngine.race` to
    report a genuine, no-finite-cost wipeout. This path is defense-in-depth:
    after the engine-level rescue, `pruned_all` should be unreachable through
    ordinary finite inputs (see TestEngineNeverExtinguishesOnFiniteCost), so
    it has to be forced here rather than hit naturally.
    """

    def test_pruned_all_gets_bounded_continuous_score_not_a_floor(self):
        rng = np.random.default_rng(5)
        q0 = rng.normal(size=DIM)
        cands = rng.normal(size=(2, DIM)) + q0

        real_race = BifurcatedFractalEngine.race

        def fake_race(self, pool, c, repulsors=None):
            out_pool, trace = real_race(self, pool, c, repulsors)
            # Force the same "no branch has a finite cost" outcome the
            # genuine all-non-finite case produces, regardless of what the
            # real race computed.
            bad_cost = np.full_like(out_pool.cost, np.nan)
            bad_alive = np.zeros_like(out_pool.alive)
            forced_pool = BranchPool(out_pool.q, out_pool.p, bad_cost, bad_alive, out_pool.seed_dirs)
            forced_trace = RaceTrace(trace.steps, trace.alive_history, trace.cost_history,
                                     True, False)
            return forced_pool, forced_trace

        with patch.object(BifurcatedFractalEngine, "race", fake_race):
            result = continuous_causal_reasoning_expert(q0, cands, seed=3)

        assert all(t.status == STATUS_PRUNED for t in result.traces)
        assert np.all(np.isfinite(result.scores))
        assert not np.any(np.isclose(result.scores, -100.0))
        assert not np.any(np.isclose(result.scores, -1e9))
        for score, trace in zip(result.scores, result.traces):
            # Bounded: never worse than -(residual + RACE_PENALTY_SCALE).
            assert score >= -(trace.micro_residual_to_target + RACE_PENALTY_SCALE) - 1e-9
            assert score <= -trace.micro_residual_to_target + 1e-9
