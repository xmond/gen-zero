"""Tests for the 0-token latent MCTS planner."""

import numpy as np
import pytest

from gen_zero.causal.latent_mcts import LatentMctsNode, LatentMctsPlanner

CANDS = ["a", "b", "c"]
EMB = np.eye(3, 4)  # three orthonormal candidates; dim 3 is a free "depth" channel


def _map_argmax(z, emb, temperature):
    return int(np.argmax(np.exp(emb @ z / temperature)))


def test_single_step_agrees_with_etf_map_argmax():
    rng = np.random.default_rng(0)
    for _ in range(50):
        emb = rng.normal(size=(6, 16))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        z = rng.normal(size=16)
        planner = LatentMctsPlanner(num_simulations=1, temperature=0.1)
        best, policy, info = planner.plan(z, [str(i) for i in range(6)], emb)
        assert best == _map_argmax(z, emb, 0.1)
        assert info["agrees_with_map"]
        assert policy[best] == 1.0 and policy.sum() == pytest.approx(1.0)
        assert info["tokens_generated"] == 0


def test_backtracks_when_top_prior_leads_to_bad_transition():
    z0 = np.array([0.6, 0.5, 0.1, 0.0])

    def transition(z, a):
        if a == 0 or np.allclose(z[:3], z[0]):  # top prior: absorbing dead end
            return np.array([0.5, 0.5, 0.5, 0.0])
        if a == 1:  # runner-up: confident state
            return np.array([0.0, 1.0, 0.0, 0.0])
        return np.zeros(4)

    planner = LatentMctsPlanner(num_simulations=40)
    best, policy, info = planner.plan(z0, CANDS, EMB, transition)
    assert info["map_action"] == 0
    assert best == 1
    assert not info["agrees_with_map"]
    assert info["root_q"][0] < info["root_q"][1]
    assert info["root_visits"][1] > info["root_visits"][0]
    assert policy.sum() == pytest.approx(1.0)


def test_converges_to_optimal_two_step_branch():
    z0 = np.array([0.6, 0.5, 0.1, 0.0])
    ambiguous = np.array([0.5, 0.5, 0.5])

    def transition(z, a):
        depth = z[3]
        if z[2] > 0.9:
            nxt = np.array([0.0, 0.0, 1.0])  # absorbing payoff
        elif np.allclose(z[:3], z[0]):
            nxt = ambiguous  # absorbing trap
        elif depth == 0:
            if a == 0:
                nxt = ambiguous  # trap: no way out
            elif a == 1:
                nxt = np.array([0.4, 0.5, 0.3])  # mid confidence, opens a path
            else:
                nxt = np.zeros(3)
        else:
            if z[1] > 0.45 and a == 2:
                nxt = np.array([0.0, 0.0, 1.0])  # payoff two steps out
            else:
                nxt = ambiguous
        return np.append(nxt, depth + 1)

    planner = LatentMctsPlanner(num_simulations=200)
    best, policy, info = planner.plan(z0, CANDS, EMB, transition)
    assert best == 1
    assert policy[1] > 0.6
    assert info["max_depth"] >= 2
    assert info["nodes"] > 3


def test_default_contact_transition_is_deterministic_and_valid():
    z0 = np.array([0.3, 0.29, 0.0, 0.0])
    planner = LatentMctsPlanner(num_simulations=30)
    a1, p1, _ = planner.plan(z0, CANDS, EMB)
    a2, p2, _ = planner.plan(z0, CANDS, EMB)
    assert a1 == a2 == 0
    assert np.array_equal(p1, p2)
    assert p1.shape == (3,)
    assert isinstance(planner.last_root, LatentMctsNode)
    assert planner.last_root.total_visits == 30


def test_single_candidate_and_input_validation():
    planner = LatentMctsPlanner(num_simulations=5)
    best, policy, _ = planner.plan(np.ones(4), ["only"], np.ones((1, 4)))
    assert best == 0 and policy.tolist() == [1.0]
    with pytest.raises(ValueError):
        planner.plan(np.ones(4), [], np.ones((0, 4)))
    with pytest.raises(ValueError):
        planner.plan(np.ones(4), CANDS, np.ones((2, 4)))
    with pytest.raises(ValueError):
        planner.plan(np.ones(4), CANDS, EMB, lambda z, a: np.ones(2))
    with pytest.raises(ValueError):
        LatentMctsPlanner(temperature=0.0)
    with pytest.raises(ValueError):
        LatentMctsPlanner(num_simulations=0)


def test_latency_under_10ms_for_100_simulations():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(8, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    z = rng.normal(size=16)
    planner = LatentMctsPlanner(num_simulations=100)
    planner.plan(z, list("abcdefgh"), emb)  # warm-up
    best_ms = min(
        planner.plan(z, list("abcdefgh"), emb)[2]["latency_ms"] for _ in range(5)
    )
    assert best_ms < 10.0


def test_spherical_flow_dissipates_potential_and_bounds_motion():
    from gen_zero.causal.latent_mcts import ContractiveHamiltonianTransition

    # Rows of a three-vertex Simplex ETF in three dimensions.
    targets = np.eye(3) - np.ones((3, 3)) / 3
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    rng = np.random.default_rng(71)
    generators = rng.normal(size=(3, 3, 3)) * 10
    for fitted in (None, generators):
        step = ContractiveHamiltonianTransition(targets, generators=fitted)
        for action in range(3):
            z = rng.normal(size=3)
            z /= np.linalg.norm(z)
            for _ in range(100):
                nxt = step(z, action)
                assert np.linalg.norm(nxt) == pytest.approx(1, abs=1e-12)
                assert targets[action] @ nxt >= targets[action] @ z - 1e-12
                assert np.linalg.norm(nxt - z) <= 2 * np.sin(0.15) + 1e-12
                z = nxt


def test_exact_gradient_flow_and_stationary_points():
    from gen_zero.causal.latent_mcts import ContractiveHamiltonianTransition

    step = ContractiveHamiltonianTransition(np.eye(3), gamma=0.2, max_angle=1)
    z = np.array([0., 1., 0.])
    expected_angle = 2 * np.arctan(np.exp(-0.2))
    assert np.allclose(step(z, 0), [np.cos(expected_angle), np.sin(expected_angle), 0])
    assert np.array_equal(step(np.eye(3)[0], 0), np.eye(3)[0])
    assert np.array_equal(step(-np.eye(3)[0], 0), -np.eye(3)[0])
    assert np.array_equal(step(np.zeros(3), 0), np.eye(3)[0])
    assert np.allclose(step(z * 1e300, 0), step(z, 0))


def test_fit_action_generators_recovers_derivatives_and_unseen_actions():
    from gen_zero.causal.latent_mcts import fit_action_dynamics

    matrices = np.array([[[0., -2.], [2., 0.]], [[-1., 0.], [0., -3.]]])
    records = []
    for action in range(2):
        for z, dt in zip(np.eye(2), [0.001, 0.003]):
            records.append((z, action, z + dt * matrices[action] @ z, dt))
    result = fit_action_dynamics(records, 2, 3)
    assert np.allclose(result[:2], matrices)
    assert np.array_equal(result[2], np.zeros((2, 2)))
    assert np.array_equal(fit_action_dynamics([], 2, 1), np.zeros((1, 2, 2)))
    for bad in [([1, 0], -1, [1, 1]), ([1, 0], 0, [1, 1], 0),
                ([1, np.nan], 0, [1, 1]), ([1], 0, [1])]:
        with pytest.raises(ValueError):
            fit_action_dynamics([bad], 2, 2)


def test_relational_violation_cannot_be_redeemed_by_confidence():
    planner = LatentMctsPlanner(num_simulations=80)
    planner.plan(np.array([0.6, 0.5, 0.1, 0.]), CANDS, EMB,
                 lambda z, a: EMB[a].copy(),
                 relational_constraint=lambda root, nxt, a: float(a != 0))
    root = planner.last_root
    assert root.q[0] == 0
    assert root.q[1] > 0
    assert all(child.evidence == 0 for child in root.children[0].children.values())


def test_leaf_value_penalizes_instability_and_invalid_evidence():
    from gen_zero.causal.latent_mcts import _leaf_value

    confidence = np.array([0.999, 0.001])
    assert _leaf_value(confidence, stability=0) == 0
    assert _leaf_value(confidence, relational=0) == 0
    assert _leaf_value(confidence, stability=0.2) < _leaf_value(confidence)
    planner = LatentMctsPlanner(num_simulations=1)
    with pytest.raises(ValueError):
        planner.plan(np.ones(4), CANDS, EMB, lambda z, a: z * np.nan)
    with pytest.raises(ValueError):
        planner.plan(np.ones(4), CANDS, EMB, relational_constraint=lambda *args: np.nan)


def test_fitted_operator_integrates_with_planner():
    from gen_zero.causal.latent_mcts import (
        ContractiveHamiltonianTransition, fit_action_dynamics,
    )

    z = np.array([0.3, 0.7, 0.2, 0.1])
    generators = fit_action_dynamics([(z, 1, np.roll(z, 1), 0.1)], 4, 3)
    transition = ContractiveHamiltonianTransition(EMB, generators=generators)
    planner = LatentMctsPlanner(num_simulations=10)
    _, policy, _ = planner.plan(z, CANDS, EMB, transition)
    assert policy.sum() == pytest.approx(1)
    assert all(np.linalg.norm(child.z) == pytest.approx(1)
               for child in planner.last_root.children.values())
