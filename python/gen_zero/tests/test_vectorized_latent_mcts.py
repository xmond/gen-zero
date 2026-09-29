"""Tests for the vectorized batch latent MCTS planner and batch transition.

Covers: (1) batch transition outputs match single-instance trajectories within
numerical tolerance; (2) spherical norm conservation and Lyapunov dissipation of
the vectorized flow; (3) execution speedup of batch planning over sequential
planning for ``B >= 16``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gen_zero.causal.latent_mcts import (
    ContractiveHamiltonianTransition,
    LatentMctsPlanner,
)
from gen_zero.causal.vectorized_latent_mcts import (
    BatchContractiveHamiltonianTransition,
    BatchLatentMctsPlanner,
    benchmark_batch_vs_sequential,
)

CANDS = ["a", "b", "c"]
EMB = np.eye(3, 4)


# ---------------------------------------------------------------------------
# 1. Batch outputs match single-instance trajectories.
# ---------------------------------------------------------------------------

def _random_setup(rng, num_actions, dim):
    emb = rng.normal(size=(num_actions, dim))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def test_batch_transition_all_matches_single_instance_per_action():
    rng = np.random.default_rng(7)
    for dim, k in [(8, 4), (16, 6), (12, 3)]:
        emb = _random_setup(rng, k, dim)
        states = rng.normal(size=(20, dim))
        single = ContractiveHamiltonianTransition(emb, gamma=0.3)
        batch = BatchContractiveHamiltonianTransition(emb, gamma=0.3)
        expected = np.stack([
            np.stack([single(states[i], a) for a in range(k)])
            for i in range(20)
        ])  # (20, k, dim)
        got = batch.transition_all(states)
        assert got.shape == (20, k, dim)
        assert np.allclose(got, expected, atol=1e-10)


def test_batch_transition_with_generators_matches_single():
    rng = np.random.default_rng(11)
    dim, k = 8, 5
    emb = _random_setup(rng, k, dim)
    generators = rng.normal(size=(k, dim, dim)) * 5.0
    states = rng.normal(size=(15, dim))
    single = ContractiveHamiltonianTransition(emb, gamma=0.25, generators=generators)
    batch = BatchContractiveHamiltonianTransition(
        emb, gamma=0.25, generators=generators
    )
    expected = np.stack([
        np.stack([single(states[i], a) for a in range(k)])
        for i in range(15)
    ])
    got = batch.transition_all(states)
    assert np.allclose(got, expected, atol=1e-9)


def test_batch_transition_indexed_matches_single_per_element():
    rng = np.random.default_rng(13)
    emb = _random_setup(rng, 5, 10)
    states = rng.normal(size=(12, 10))
    actions = rng.integers(0, 5, size=12)
    single = ContractiveHamiltonianTransition(emb, gamma=0.4)
    batch = BatchContractiveHamiltonianTransition(emb, gamma=0.4)
    expected = np.stack([single(states[i], int(actions[i])) for i in range(12)])
    got = batch.transition_indexed(states, actions)
    assert np.allclose(got, expected, atol=1e-10)


def test_batch_planner_single_simulation_matches_single_instance():
    """B==1, num_simulations==1: equals MAP argmax for both planners."""
    rng = np.random.default_rng(21)
    for _ in range(10):
        emb = _random_setup(rng, 6, 12)
        z = rng.normal(size=12)
        sp = LatentMctsPlanner(num_simulations=1, temperature=0.1)
        bp = BatchLatentMctsPlanner(num_simulations=1, temperature=0.1)
        sb, spol, sinfo = sp.plan(z, [str(i) for i in range(6)], emb)
        bb, bpol, binfo = bp.plan(z[None], [str(i) for i in range(6)], emb)
        assert sb == int(bb[0])
        assert np.allclose(spol, bpol[0], atol=1e-9)
        assert sinfo["nodes"] == int(binfo["nodes"][0])


def test_batch_planner_matches_single_instance_for_various_simulations():
    """B==1 batch planner reproduces single-instance policy within tolerance."""
    rng = np.random.default_rng(31)
    emb = _random_setup(rng, 5, 10)
    z = rng.normal(size=10)
    cands = [f"c{i}" for i in range(5)]
    for nsims in (5, 20, 60):
        sp = LatentMctsPlanner(num_simulations=nsims, temperature=0.1, max_depth=4)
        bp = BatchLatentMctsPlanner(num_simulations=nsims, temperature=0.1, max_depth=4)
        sb, spol, sinfo = sp.plan(z, cands, emb)
        bb, bpol, binfo = bp.plan(z[None], cands, emb)
        assert sb == int(bb[0]), f"best mismatch at nsims={nsims}"
        assert np.allclose(spol, bpol[0], atol=2e-2), f"policy mismatch at nsims={nsims}"
        assert int(binfo["nodes"][0]) == sinfo["nodes"]


def test_batch_planner_multi_instance_independent_and_valid():
    rng = np.random.default_rng(41)
    emb = _random_setup(rng, 6, 12)
    B = 8
    states = rng.normal(size=(B, 12))
    cands = [f"c{i}" for i in range(6)]
    bp = BatchLatentMctsPlanner(num_simulations=50, max_depth=4)
    best, policy, info = bp.plan(states, cands, emb)
    assert best.shape == (B,)
    assert policy.shape == (B, 6)
    assert np.allclose(policy.sum(axis=1), 1.0)
    # Each instance must match its own single-instance run.
    sp = LatentMctsPlanner(num_simulations=50, max_depth=4)
    for i in range(B):
        sb, spol, _ = sp.plan(states[i], cands, emb)
        assert sb == int(best[i])
        assert np.allclose(spol, policy[i], atol=2e-2)


# ---------------------------------------------------------------------------
# 2. Spherical norm conservation and Lyapunov dissipation.
# ---------------------------------------------------------------------------

def test_norm_conservation_under_batch_flow():
    rng = np.random.default_rng(71)
    targets = np.eye(4) - np.ones((4, 4)) / 4
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    batch = BatchContractiveHamiltonianTransition(targets, gamma=0.3)
    states = rng.normal(size=(50, 4))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    z = states.copy()
    for _ in range(100):
        nxt = batch.transition_all(z)  # (B, num_actions, dim)
        norms = np.linalg.norm(nxt, axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        z = nxt[:, 0, :]  # follow action 0


def test_norm_conservation_with_generators():
    rng = np.random.default_rng(73)
    dim, k = 6, 4
    emb = _random_setup(rng, k, dim)
    generators = rng.normal(size=(k, dim, dim)) * 8.0
    batch = BatchContractiveHamiltonianTransition(
        emb, gamma=0.2, generators=generators, max_angle=0.4
    )
    states = rng.normal(size=(30, dim))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    for _ in range(50):
        nxt = batch.transition_all(states)
        assert np.allclose(np.linalg.norm(nxt, axis=-1), 1.0, atol=1e-5)
        states = nxt[:, 1, :]


def test_lyapunov_dissipation_potential_nonincreasing():
    rng = np.random.default_rng(91)
    dim, k = 8, 5
    emb = _random_setup(rng, k, dim)
    batch = BatchContractiveHamiltonianTransition(emb, gamma=0.3)
    states = rng.normal(size=(40, dim))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    eps = 1e-6
    for _ in range(80):
        nxt = batch.transition_all(states)  # (B, k, dim)
        for a in range(k):
            target = emb[a]
            v_before = 1.0 - states @ target          # (B,)
            v_after = 1.0 - nxt[:, a, :] @ target       # (B,)
            assert np.all(v_after <= v_before + eps), (
                f"action {a}: V increased (max delta {np.max(v_after - v_before)})"
            )
        states = nxt[:, 0, :]


def test_lyapunov_dissipation_batch_per_action():
    """V(z_{t+1}) <= V(z_t) + eps for every (state, action) in the batch."""
    rng = np.random.default_rng(97)
    dim, k = 10, 6
    emb = _random_setup(rng, k, dim)
    batch = BatchContractiveHamiltonianTransition(emb, gamma=0.35, max_angle=0.5)
    states = rng.normal(size=(25, dim))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    targets = batch.targets  # (k, dim) unit
    eps = 1e-6
    for _ in range(60):
        nxt = batch.transition_all(states)  # (B, k, dim)
        # V for every (instance, action): 1 - target_a . z_next
        v_before = 1.0 - states @ targets.T          # (B, k)
        v_after = 1.0 - np.einsum("Bkd,kd->Bk", nxt, targets)
        assert np.all(v_after <= v_before + eps)
        states = nxt[:, 2, :]


# ---------------------------------------------------------------------------
# 3. Execution speedup on batch queries.
# ---------------------------------------------------------------------------

def test_batch_planner_faster_than_sequential_for_B_ge_16():
    result = benchmark_batch_vs_sequential(
        B=64, num_actions=6, dim=16, num_simulations=120, max_depth=4, repeats=5
    )
    assert result["batch_ms"] < result["sequential_ms"], (
        f"batch not faster: batch={result['batch_ms']:.2f}ms "
        f"seq={result['sequential_ms']:.2f}ms"
    )
    assert result["speedup"] > 1.2, f"speedup too small: {result['speedup']:.2f}"


def test_batch_planner_speedup_grows_with_B():
    r16 = benchmark_batch_vs_sequential(B=16, repeats=4)
    r64 = benchmark_batch_vs_sequential(B=64, repeats=4)
    # Both must show a speedup, and the larger batch must not be slower.
    assert r16["batch_ms"] < r16["sequential_ms"]
    assert r64["batch_ms"] < r64["sequential_ms"]
    assert r64["speedup"] >= r16["speedup"] * 0.8  # monotonic-ish


def test_transition_batch_faster_than_per_element_loop():
    rng = np.random.default_rng(5)
    dim, k = 16, 8
    emb = _random_setup(rng, k, dim)
    single = ContractiveHamiltonianTransition(emb, gamma=0.3)
    batch = BatchContractiveHamiltonianTransition(emb, gamma=0.3)
    B = 256
    states = rng.normal(size=(B, dim))

    import time as _t
    for _ in range(3):  # warm up
        batch.transition_all(states)
        for i in range(0, B, 16):
            for a in range(k):
                single(states[i], a)

    t0 = _t.perf_counter()
    for _ in range(5):
        for i in range(B):
            for a in range(k):
                single(states[i], a)
    seq = _t.perf_counter() - t0

    t0 = _t.perf_counter()
    for _ in range(5):
        batch.transition_all(states)
    bat = _t.perf_counter() - t0

    assert bat < seq, f"batch transition not faster: batch={bat:.4f}s seq={seq:.4f}s"
