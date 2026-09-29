"""Tests for the bifurcated fractal engine (design doc 04, P1).

Covers: constructive >=50% prune per macro step, K selection from repulsion
entropy, orthogonal equivariance of the bifurcate+race stages, Woodbury
closed-form fixed point vs dense solve, iterative micro path within 20 steps
under the spectral-radius contract, fail-closed rejection, byte ledger.
"""
from __future__ import annotations

import numpy as np
import pytest

from gen_zero.causal.bifurcated_fractal_engine import (
    STATUS_CONVERGED,
    BifurcatedFractalEngine,
    BranchPool,
    gram_schmidt,
    normalized_entropy,
    pick_k,
    repulsion_scores_batch,
    spectral_radius,
    woodbury_fixed_point,
    working_set_bytes,
)
from gen_zero.causal.parallel_rnn_lora import ParallelRNNLoRAAdapter

DIM = 8


def _adapter(dim=DIM, seed=0, rho_max=0.95):
    return ParallelRNNLoRAAdapter(dim, 2, rho_max=rho_max, seed=seed)


def _engine(**kw):
    return BifurcatedFractalEngine(DIM, _adapter(), **kw)


def _orthogonal(dim, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q


def _problem(seed=0, m=3):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=DIM)
    q0 = c + rng.normal(size=DIM)
    p0 = rng.normal(size=DIM)
    r = rng.normal(size=(m, DIM))
    return q0, p0, c, r


class TestBifurcation:
    def test_no_repulsors_gives_two_branches(self):
        q0, p0, c, _ = _problem()
        pool, dec = _engine().bifurcate(q0, p0, c)
        assert dec.K == 2 and pool.q.shape == (2, DIM) and pool.p.shape == (2, DIM)
        assert np.array_equal(pool.p[0], p0)

    def test_equal_scores_give_max_entropy_and_four_branches(self):
        assert normalized_entropy(np.array([0.3, 0.3, 0.3, 0.3])) == pytest.approx(1.0)
        assert pick_k(1.0, u_lo=0.33, u_hi=0.66) == 4
        assert pick_k(0.0, u_lo=0.33, u_hi=0.66) == 2
        assert pick_k(0.5, u_lo=0.33, u_hi=0.66) == 3

    def test_seed_directions_are_orthonormal_and_never_padded(self):
        q0, p0, c, r = _problem(m=4)
        eng = _engine(u_lo=0.01, u_hi=0.02)  # force K=4 whenever U > 0.02
        pool, dec = eng.bifurcate(q0, p0, c, r)
        d = pool.seed_dirs
        assert d.shape == (dec.K - 1, DIM)
        assert np.allclose(d @ d.T, np.eye(dec.K - 1), atol=1e-10)
        # Rank-deficient candidates: only one independent direction exists.
        dirs = gram_schmidt([np.ones(DIM), 2 * np.ones(DIM), 3 * np.ones(DIM)], max_dirs=3)
        assert dirs.shape == (1, DIM)


class TestBatchScoring:
    def test_batch_repulsion_matches_scalar_helper_including_zero_rows(self):
        from gen_zero.causal.fractal_multiscale_engine import counterfactual_repulsion_score
        rng = np.random.default_rng(7)
        d = rng.normal(size=(5, DIM)); d[2] = 0.0
        r = rng.normal(size=(3, DIM)); r[1] = 0.0
        got = repulsion_scores_batch(d, r)
        want = np.array([counterfactual_repulsion_score(row, r) for row in d])
        assert np.allclose(got, want, atol=1e-14)
        assert got[2] == 0.0

    def test_per_repulsor_scores_feed_the_decision(self):
        q0, p0, c, r = _problem(m=3)
        _, dec = _engine().bifurcate(q0, p0, c, r)
        assert len(dec.repulsion_scores) == 3
        assert max(dec.repulsion_scores) == pytest.approx(repulsion_scores_batch(p0[None, :], r)[0])


class TestRaceAndPrune:
    @pytest.mark.parametrize("k_expected,u_lo,u_hi", [(4, 0.01, 0.02), (3, 0.01, 0.9999999)])
    def test_prunes_at_least_half_every_step(self, k_expected, u_lo, u_hi):
        q0, p0, c, r = _problem(m=4)
        eng = _engine(u_lo=u_lo, u_hi=u_hi, repulsion_threshold=1.0)
        res = eng.think(q0, p0, c, r)
        assert res.bifurcation.K == k_expected
        hist = res.race.alive_history
        assert hist[0] == k_expected
        for before, after in zip(hist, hist[1:]):
            assert after <= before // 2 or after <= 1
        assert hist[-1] == 1 and res.survivor is not None

    def test_two_branches_resolve_in_one_step(self):
        q0, p0, c, _ = _problem()
        res = _engine().think(q0, p0, c)
        assert res.race.alive_history == (2, 1) and res.race.steps == 1

    def test_hard_repulsion_wipeout_is_rescued_by_least_cost_branch(self):
        # Design change (67.3% true-candidate loss diagnosed in
        # continuous_causal_reasoning_expert): a hard repulsion threshold that
        # would kill every branch in a step no longer reports zero survivors.
        # `_cost` already prices repulsion in softly, so the pool falls back
        # to the single least-cost branch instead of going extinct.
        q0, p0, c, _ = _problem()
        r = np.eye(DIM)  # every direction is forbidden by the hard gate
        eng = _engine(repulsion_threshold=1e-9)
        pool, decision = eng.bifurcate(q0, p0, c, r)
        out, trace = eng.race(pool, c, r)
        assert trace.rescued and not trace.pruned_all
        assert int(out.alive.sum()) == 1
        finite = np.isfinite(out.cost)
        assert finite.any()
        want = int(np.argmin(np.where(finite, out.cost, np.inf)))
        assert int(np.flatnonzero(out.alive)[0]) == want

        res = eng.think(q0, p0, c, r)
        assert res.status == STATUS_CONVERGED
        assert res.survivor is not None and res.q is not None and res.micro is not None
        assert res.race.rescued and not res.race.pruned_all

    def test_genuinely_no_finite_cost_branch_stays_pruned(self):
        # The rescue only has a state to fall back to when at least one branch
        # has a finite cost. If literally every branch is non-finite there is
        # nothing to rescue from, and PRUNED (no fabricated survivor) is still
        # the correct, honest result.
        q0, p0, c, _ = _problem()
        eng = _engine()
        pool, _ = eng.bifurcate(q0, p0, c)
        bad_q = np.full_like(pool.q, np.nan)
        bad_pool = BranchPool(bad_q, pool.p, pool.cost, pool.alive, pool.seed_dirs)
        out, trace = eng.race(bad_pool, c)
        assert trace.pruned_all and not trace.rescued
        assert int(out.alive.sum()) == 0

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_bifurcate_and_race_are_orthogonally_equivariant(self, seed):
        q0, p0, c, r = _problem(seed, m=3)
        Q = _orthogonal(DIM, seed + 10)
        eng = _engine(u_lo=0.01, u_hi=0.02, repulsion_threshold=1.0)
        pool_a, dec_a = eng.bifurcate(q0, p0, c, r)
        pool_b, dec_b = eng.bifurcate(Q @ q0, Q @ p0, Q @ c, r @ Q.T)
        assert dec_a.K == dec_b.K
        assert dec_a.uncertainty == pytest.approx(dec_b.uncertainty, abs=1e-12)
        assert np.allclose(pool_b.p, pool_a.p @ Q.T, atol=1e-9)
        out_a, tr_a = eng.race(pool_a, c, r)
        out_b, tr_b = eng.race(pool_b, Q @ c, r @ Q.T)
        assert tr_a.alive_history == tr_b.alive_history
        assert np.array_equal(out_a.alive, out_b.alive)
        assert np.allclose(out_b.q, out_a.q @ Q.T, atol=1e-9)
        assert np.allclose(out_a.cost, out_b.cost, atol=1e-9)

    def test_pool_step_matches_scalar_leapfrog(self):
        from gen_zero.causal.fractal_multiscale_engine import leapfrog_step
        q0, p0, c, _ = _problem()
        eng = _engine(max_macro_steps=1)
        pool, _ = eng.bifurcate(q0, p0, c)
        out, _ = eng.race(pool, c)
        for i in range(2):
            q_ref, p_ref = leapfrog_step(pool.q[i], pool.p[i], c, omega=eng.omega,
                                         dt=eng.dt_macro, gamma=eng.gamma_macro)
            assert np.allclose(out.q[i], q_ref, atol=1e-12)
            assert np.allclose(out.p[i], p_ref, atol=1e-12)


class TestMicro:
    def test_woodbury_matches_dense_solve(self):
        ad = _adapter(dim=32, seed=3)
        rng = np.random.default_rng(5)
        x = rng.normal(size=32)
        h, resid = woodbury_fixed_point(ad, x)
        dense = np.linalg.solve(np.eye(32) - ad.transition_matrix(), ad.input_matrix() @ x)
        assert np.allclose(h, dense, atol=1e-9)
        assert resid < 1e-9

    def test_iterative_path_converges_within_budget_under_spectral_contract(self):
        q0, p0, c, _ = _problem()
        eng = _engine(micro_path="iterative", micro_max_steps=20, micro_tol=1e-5)
        res = eng.think(q0, p0, c)
        assert res.status == STATUS_CONVERGED
        assert res.micro.path == "iterative" and res.micro.steps <= 20
        rho = eng.adapter_spectral_radius
        assert rho <= 0.55
        h_star, _ = woodbury_fixed_point(eng.adapter, c)
        assert np.linalg.norm(res.q - h_star) < 1e-3

    def test_untrained_adapter_fixed_point_gap_is_reported_not_hidden(self):
        q0, p0, c, _ = _problem()
        res = _engine().think(q0, p0, c)
        assert res.status == STATUS_CONVERGED and res.micro.path == "closed_form"
        # An untrained adapter's fixed point is not the well minimum c: the gap is visible.
        assert res.micro.fixed_point_gap > 1e-3


class TestFailClosed:
    def test_rejects_adapter_above_spectral_radius_contract(self):
        ad = _adapter()
        assert spectral_radius(ad.transition_matrix()) <= 0.55
        with pytest.raises(ValueError, match="spectral radius"):
            BifurcatedFractalEngine(DIM, ad, rho_spec=0.01)

    def test_rejects_nonfinite_and_mismatched_inputs(self):
        q0, p0, c, r = _problem()
        eng = _engine()
        bad = q0.copy(); bad[0] = np.nan
        with pytest.raises(ValueError):
            eng.think(bad, p0, c, r)
        with pytest.raises(ValueError):
            eng.think(q0[:-1], p0, c, r)
        with pytest.raises(ValueError):
            eng.think(q0, p0, c, r[:, :-1])

    def test_rejects_unstable_macro_step_and_bad_tau(self):
        with pytest.raises(ValueError):
            _engine(dt_micro=0.3)  # 0.3 * 8 >= 2
        with pytest.raises(ValueError):
            _engine(tau_macro=6)

    def test_rejects_working_set_over_budget(self):
        q0, p0, c, r = _problem()
        with pytest.raises(ValueError, match="working set"):
            _engine(l2_bytes=64).think(q0, p0, c, r)


class TestLedger:
    @pytest.mark.parametrize("dim,k,rank,m,expected", [
        (16, 4, 4, 4, 4804), (64, 4, 4, 4, 19012), (128, 4, 4, 8, 42052),
        (192, 4, 4, 8, 63044), (256, 4, 4, 8, 84036),
    ])
    def test_ledger_matches_design_doc_table(self, dim, k, rank, m, expected):
        assert working_set_bytes(dim, k, rank, m) == expected
        assert (expected <= 65536) == (dim <= 192)
