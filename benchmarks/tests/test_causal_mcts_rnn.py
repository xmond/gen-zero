"""Tests for CausalMCTSRNN (RFC-072): Lie latent dynamics, Lyapunov damper, ACT,
counterfactual sensitivity, adaptive fast/MCTS decisions.

Pure NumPy, CPU only. Random runtimes are built directly from arrays so the
tests need neither torch nor any trained artifact; the tests that use the
trained grid prior or the cached Qwen features skip when those files are absent.
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

from gen_zero.causal.causal_mcts_rnn import (CausalMCTSRNN, ContractionDamper, DualManifoldPrior,  # noqa: E402
                                             LatentReadout, LieLatentDynamics, certified_contraction)
from gen_zero.causal.dual_manifold_head import DualManifoldHead, TIERS  # noqa: E402
from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime, _softmax  # noqa: E402


def _runtime(D=16, d=8, r=2, T=6, heads=2, L=1, seed=0, ub_scale=0.5) -> RNNSetAdapterRuntime:
    rng = np.random.default_rng(seed)
    F = 2 * d
    g = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    arr = {"mu": 0.1 * g(D), "W_in": g(D, d) / math.sqrt(D), "lam": g(d),
           "U_A": 0.5 * g(d, r), "V_A": 0.5 * g(d, r), "U_B": ub_scale * g(d, r), "V_B": ub_scale * g(d, r),
           "W_s": np.eye(d, dtype=np.float32) + 0.2 * g(d, d),
           "wq": 0.4 * g(L, d, d), "bq": 0.05 * g(L, d), "wk": 0.4 * g(L, d, d), "bk": 0.05 * g(L, d),
           "wv": 0.4 * g(L, d, d), "bv": 0.05 * g(L, d), "wo": 0.4 * g(L, d, d), "bo": 0.05 * g(L, d),
           "w1": 0.4 * g(L, F, d), "b1": 0.05 * g(L, F), "w2": 0.4 * g(L, d, F), "b2": 0.05 * g(L, d)}
    a_scale, _ = certified_contraction(arr["lam"], arr["U_A"], arr["V_A"], 0.95)
    cfg = {"in_dim": D, "d": d, "rank": r, "think_steps": T, "n_heads": heads, "n_layers": L, "ffn_dim": F}
    return RNNSetAdapterRuntime(arr, cfg, a_scale, 0.95, 1e-6, {"test": True})


def _perm_world(n_states=24, dim=28, n_actions=3, seed=0):
    """Random permutation dynamics on orthonormal codes; action 2 is made of 4-cycles (eigenvalue -1)."""
    rng = np.random.default_rng(seed)
    codes = np.linalg.qr(rng.standard_normal((dim, n_states)))[0].T
    perms = [rng.permutation(n_states) for _ in range(n_actions - 1)]
    perms.append(np.roll(np.arange(n_states).reshape(-1, 4), 1, axis=1).ravel())
    Z = np.vstack([codes] * n_actions)
    Zn = np.vstack([codes[p] for p in perms])
    A = np.repeat(np.arange(n_actions), n_states)
    return codes, perms, Z, A, Zn


def _readout(codes, D, seed, fail_states=(), success_states=()):
    rng = np.random.default_rng(seed)
    n = codes.shape[0]
    G = rng.standard_normal((n, D))
    fail = np.zeros(n)
    fail[list(fail_states)] = 1.0
    succ = np.zeros(n)
    succ[list(success_states)] = 1.0
    pot = -rng.random(n)
    return LatentReadout(G.T @ codes, codes.T @ fail, codes.T @ succ, codes.T @ pot)


# --------------------------------------------------------------------------- Lie dynamics

def test_lie_fit_recovers_permutation_dynamics_exactly():
    codes, perms, Z, A, Zn = _perm_world()
    dyn = LieLatentDynamics.fit(Z, A, Zn, 3)
    for a, p in enumerate(perms):
        assert np.abs(codes @ dyn.R64[a].T - codes[p]).max() < 1e-10
        G = dyn.generators[a]
        assert np.abs(G + G.T).max() < 1e-12                       # Omega in so(d)
    assert dyn.orthogonality_error < 1e-10
    z = codes[3].astype(np.float32)
    for a in range(3):
        assert abs(np.linalg.norm(dyn.step(z, a)) - 1.0) < 1e-5     # norm preserved


def test_lie_fractional_step_composes():
    codes, _, Z, A, Zn = _perm_world(seed=1)
    full = LieLatentDynamics.fit(Z, A, Zn, 3)
    half = LieLatentDynamics(full.generators, dt=0.5)
    for a in range(3):
        np.testing.assert_allclose(half.R64[a] @ half.R64[a], full.R64[a], atol=1e-10)


def test_lie_rejects_bad_inputs():
    codes, _, Z, A, Zn = _perm_world()
    with pytest.raises(ValueError):
        LieLatentDynamics(np.ones((2, 4, 4)))                      # not skew
    bad = Z.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        LieLatentDynamics.fit(bad, A, Zn, 3)
    with pytest.raises(ValueError):
        LieLatentDynamics.fit(Z, A, Zn, 4)                         # action 3 never observed


# --------------------------------------------------------------------------- ACT + prior

def test_act_truncated_reproduces_runtime_score_exactly():
    rt = _runtime(seed=3)
    eng = CausalMCTSRNN(rt, act_eps=0.0, act_max_steps=rt.cfg["think_steps"])
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.standard_normal(16).astype(np.float32)
        C = rng.standard_normal((5, 16)).astype(np.float32)
        logits, steps = eng.prior_logits(q, C)
        assert steps == rt.cfg["think_steps"]
        np.testing.assert_allclose(logits, rt.score(q, C), atol=1e-5, rtol=0)


def test_act_halts_early_at_the_fixed_point():
    rt = _runtime(seed=4, ub_scale=2.0)
    eng = CausalMCTSRNN(rt, act_eps=1e-4, act_max_steps=64)
    zq = rt._encode(np.random.default_rng(1).standard_normal(16).astype(np.float32))
    q_out, steps = eng.act_think(zq)
    assert 1 <= steps < 64
    p = rt.p
    A = rt.a_scale * (np.diag(rt._dvec) + p["U_A"] @ p["V_A"].T)
    u = p["U_B"] @ (p["V_B"].T @ zq)
    h_star = np.linalg.solve(np.eye(A.shape[0]) - A.astype(np.float64), u.astype(np.float64))
    # geometric tail after the halting step: ||h - h*|| <= rho/(1-rho) * eps*||u||
    bound = 0.95 / 0.05 * 1e-4 * np.linalg.norm(u) + 1e-5
    assert np.linalg.norm((q_out - zq) - h_star) <= bound
    # a looser tolerance halts no later
    eng.act_eps = 1e-2
    assert eng.act_think(zq)[1] <= steps


# --------------------------------------------------------------------------- equivariance

def test_prior_and_classify_permutation_equivariant():
    rt = _runtime(seed=5)
    eng = CausalMCTSRNN(rt, temperature=1.3)
    rng = np.random.default_rng(2)
    q = rng.standard_normal(16).astype(np.float32)
    C = rng.standard_normal((6, 16)).astype(np.float32)
    perm = rng.permutation(6)
    p, _, _ = eng.prior(q, C)
    pp, _, _ = eng.prior(q, C[perm])
    np.testing.assert_allclose(pp, p[perm], atol=1e-6)
    d1, d2 = eng.classify(q, C), eng.classify(q, C[perm])
    assert perm[d2.action] == d1.action


def test_mcts_decision_equivariant_under_action_relabeling():
    codes, _, Z, A, Zn = _perm_world(n_states=32, dim=36, seed=6)
    dyn = LieLatentDynamics.fit(Z, A, Zn, 3)
    ro = _readout(codes, 16, seed=7, fail_states=(1, 5, 9, 13), success_states=(20,))
    rt = _runtime(seed=8)
    perm = np.array([2, 0, 1])
    dyn_p = LieLatentDynamics(dyn.generators[perm])
    for s in (0, 3, 7):
        z = codes[s].astype(np.float32)
        # clock=lambda: 0 freezes the tier SLA (always "instant"): this test asserts an exact
        # search outcome and must not be at the mercy of wall-clock jitter on a shared host.
        e1 = CausalMCTSRNN(rt, dyn, top_k=3, sims_min=16, sims_max=16, clock=lambda: 0)
        e2 = CausalMCTSRNN(rt, dyn_p, top_k=3, sims_min=16, sims_max=16, clock=lambda: 0)
        d1, d2 = e1.decide(z, ro, mode="mcts"), e2.decide(z, ro, mode="mcts")
        assert perm[d2.action] == d1.action
        np.testing.assert_array_equal(d2.raw_policy, d1.raw_policy[perm])   # visit counts: exact
        np.testing.assert_allclose(d2.probs, d1.probs[perm], atol=1e-6)     # cf: QR round-off only


# --------------------------------------------------------------------------- fast vs MCTS consistency

def test_fast_path_equals_softmax_of_runtime_scores():
    codes, _, Z, A, Zn = _perm_world(seed=9)
    dyn = LieLatentDynamics.fit(Z, A, Zn, 3)
    ro = _readout(codes, 16, seed=10)
    rt = _runtime(seed=11)
    eng = CausalMCTSRNN(rt, dyn, temperature=2.0, act_eps=0.0, act_max_steps=rt.cfg["think_steps"])
    z = codes[4].astype(np.float32)
    d = eng.decide(z, ro, mode="fast")
    q = ro.features @ z
    C = (ro.features @ dyn.step_all(z).T).T
    ref = _softmax(rt.score(q, C) / np.float32(2.0))
    np.testing.assert_allclose(d.prior, ref, atol=1e-6)
    assert d.mode == "fast" and d.action == int(np.argmax(ref))


def test_mcts_without_rewards_follows_the_prior():
    """No terminals, zero shaping: every Q is 0, so PUCT visits follow the (top-k) prior."""
    codes, _, Z, A, Zn = _perm_world(seed=12)
    dyn = LieLatentDynamics.fit(Z, A, Zn, 3)
    ro = _readout(codes, 16, seed=13)
    rt = _runtime(seed=14)
    # clock=lambda: 0 freezes the tier SLA: this test asserts an exact sim count and search
    # outcome, unrelated to the (real-clock) tier ceiling feature, so it must not be flaky
    # under host load.
    eng = CausalMCTSRNN(rt, dyn, top_k=2, sims_min=16, sims_max=16, shaping=0.0, clock=lambda: 0)
    for s in range(0, 24, 5):
        d = eng.decide(codes[s].astype(np.float32), ro, mode="mcts")
        assert d.mode == "mcts" and d.n_simulations == 16
        assert d.action == int(np.argmax(d.prior))
        pruned = np.argsort(-d.prior)[2:]
        assert np.all(d.raw_policy[pruned] == 0.0)                  # top-2 pruning
        np.testing.assert_allclose(d.root_q[np.isfinite(d.root_q)], 0.0, atol=1e-12)


def test_mcts_avoids_a_terminal_that_the_prior_prefers():
    codes, perms, Z, A, Zn = _perm_world(seed=15)
    dyn = LieLatentDynamics.fit(Z, A, Zn, 3)
    rt = _runtime(seed=16)
    z0 = codes[0].astype(np.float32)
    ro0 = _readout(codes, 16, seed=17)
    top = int(np.argmax(CausalMCTSRNN(rt, dyn).decide(z0, ro0, mode="fast").prior))
    doomed = int(perms[top][0])                                     # the state the prior's move enters
    ro = _readout(codes, 16, seed=17, fail_states=(doomed,))
    # clock=lambda: 0 freezes the tier SLA so the "adaptive" escalation below always gets its
    # full planned simulation budget, not a host-load-dependent truncation.
    eng = CausalMCTSRNN(rt, dyn, top_k=3, clock=lambda: 0)
    assert eng.decide(z0, ro, mode="fast").action == top            # System 1 walks into it
    d = eng.decide(z0, ro, mode="adaptive")
    assert d.triggers["probe_conflict"] and d.mode == "mcts"
    assert d.action != top


# --------------------------------------------------------------------------- Lyapunov contraction

def test_damper_is_certified_and_contracts_off_manifold_drift():
    codes, _, _, _, _ = _perm_world(n_states=24, dim=40, seed=18)
    damper = ContractionDamper.random(codes.T, rank=4, rho_max=0.95, seed=0)
    assert damper.sigma_max_A <= 0.95 + 1e-9
    rng = np.random.default_rng(0)
    e = rng.standard_normal(40)
    e -= codes.T @ (codes @ e)                                       # purely off-manifold
    z = (codes[2] + e).astype(np.float32)
    n0 = damper.off_manifold_norm(z)
    for k in (1, 5, 20):
        assert damper.off_manifold_norm(damper.apply(z, k)) <= 0.95 ** k * n0 * (1 + 1e-4)
    on = codes[2].astype(np.float32)
    np.testing.assert_allclose(damper.apply(on, 3), on, atol=1e-6)   # manifold points untouched


def test_long_rollout_energy_neither_explodes_nor_collapses():
    codes, _, Z, A, Zn = _perm_world(n_states=24, dim=40, seed=19)
    dyn = LieLatentDynamics.fit(Z, A, Zn, 3)
    damper = ContractionDamper.random(codes.T, rank=4, rho_max=0.95, seed=1)
    rng = np.random.default_rng(2)
    z_d = z_u = codes[0].astype(np.float32)
    norms, off_d, off_u = [], [], []
    for t in range(3000):
        a = int(rng.integers(3))
        noise = (1e-3 * rng.standard_normal(40)).astype(np.float32)  # injected drift each step
        z_u = dyn.step(z_u, a) + noise
        z_d = dyn.step(z_d, a) + noise
        if t % 2 == 1:
            z_d = damper.apply(z_d)
        norms.append(float(np.linalg.norm(z_d)))
        off_d.append(damper.off_manifold_norm(z_d))
        off_u.append(damper.off_manifold_norm(z_u))
    assert 0.9 < min(norms) and max(norms) < 1.1
    # stationary bound for e <- A(e + n): ||e|| <= ||n|| / (1 - 0.95), with ||n|| ~ 1e-3*sqrt(16)
    assert max(off_d[100:]) < 1e-3 * math.sqrt(16) * 2 / (1 - 0.95 ** 2)
    assert np.mean(off_u[-500:]) > 3 * np.mean(off_d[-500:])         # undamped drift keeps growing
    # a pure rotation keeps the norm to float32 precision over 3000 steps
    z = codes[1].astype(np.float32)
    for t in range(3000):
        z = dyn.step(z, t % 3)
    assert abs(float(np.linalg.norm(z)) - 1.0) < 1e-3


# --------------------------------------------------------------------------- counterfactual

def test_counterfactual_perturbation_is_orthogonal_and_deterministic():
    rt = _runtime(seed=20)
    eng = CausalMCTSRNN(rt, cf_eps=0.0)
    rng = np.random.default_rng(3)
    q = rng.standard_normal(16).astype(np.float32)
    C = rng.standard_normal((3, 16)).astype(np.float32)
    p, _, _ = eng.prior(q, C)
    assert eng.counterfactual_sensitivity(q, C, p) == (0.0, 0.0)     # eps = 0: do(X) = X
    eng.cf_eps = 0.3
    a = eng.counterfactual_sensitivity(q, C, p)
    assert a == eng.counterfactual_sensitivity(q, C, p)             # seeded
    assert 0.0 <= a[0] <= 1.0 and 0.0 <= a[1] <= 1.0
    assert a[0] > 0.0                                               # the RMS encoder does react
    eng.cf_eps = 3.0
    assert eng.counterfactual_sensitivity(q, C, p)[0] >= a[0] - 1e-9


# --------------------------------------------------------------------------- fail closed

def test_decide_fails_closed():
    codes, _, Z, A, Zn = _perm_world()
    dyn = LieLatentDynamics.fit(Z, A, Zn, 3)
    ro = _readout(codes, 16, seed=0)
    eng = CausalMCTSRNN(_runtime(), dyn)
    z = codes[0].astype(np.float32).copy()
    z[0] = np.inf
    with pytest.raises(ValueError):
        eng.decide(z, ro)
    with pytest.raises(ValueError):
        eng.decide(codes[0][:10], ro)
    with pytest.raises(ValueError):
        eng.decide(codes[0], ro, mode="nope")
    with pytest.raises(ValueError):
        CausalMCTSRNN(_runtime()).decide(codes[0], ro)               # no dynamics


# --------------------------------------------------------------------------- real Qwen features

QWEN_ADAPTER = REPO / "artifacts" / "qwen35_9b" / "zero_rnn_set_adapter_qwen35_9b.npz"
QWEN_PARITY = REPO / "artifacts" / "qwen35_9b" / "parity_val200.npz"


@pytest.mark.skipif(not (QWEN_ADAPTER.exists() and QWEN_PARITY.exists()), reason="Qwen artifacts missing")
def test_real_qwen_adapter_classify_matches_runtime():
    rt = RNNSetAdapterRuntime.from_npz(QWEN_ADAPTER)
    eng = CausalMCTSRNN(rt, act_eps=0.0, act_max_steps=rt.cfg["think_steps"], cf_samples=1)
    with np.load(QWEN_PARITY, allow_pickle=False) as z:
        Q, Cs, off, torch_scores = z["q"], z["cands"], z["offsets"], z["torch_scores"]
    for i in range(0, 200, 10):
        C = Cs[off[i]:off[i + 1]]
        s = rt.score(Q[i], C)
        d = eng.classify(Q[i], C)
        assert d.action == int(np.argmax(s))
        np.testing.assert_allclose(s, torch_scores[off[i]:off[i + 1]], atol=5e-3, rtol=0)
