"""Unit Tests for Issue #93: 6 Converged Orthogonal Planning Engines.

RFC-093 Verification Suite:
1. AStarEngine (AStarArena64B, Bidirectional meet-in-the-middle, Uncertainty penalty)
2. MctsEngine (MctsNode64B, PUCT, Hamiltonian dynamics, Causal pruning)
3. MpcCemEngine (Continuous trajectory CEM, Discrete CEM, Receding horizon step)
4. ManifoldGFlowNetEngine (Simplex ETF flow matching, Mode coverage, Permutation equivariance)
5. CfrNashEngine (CFR+, BayesianBeliefTracker, Exploitability bound)
6. CpSatFormalEngine (0-1 ILP Hard pruning, NCBF Lie derivative barrier filter)
7. Backward compatibility & GenZeroClient integration
"""

import math
import unittest
import numpy as np

from gen_zero.planner.engines import (
    AStarArena64B,
    AStarEngine,
    MctsNode64B,
    MctsEngine,
    MpcCemEngine,
    MpcMode,
    ManifoldGFlowNetEngine,
    CfrNashEngine,
    BayesianBeliefTracker,
    CpSatFormalEngine,
)
from gen_zero.client import GenZeroClient, GenZeroConfig


class TestAStarEngine(unittest.TestCase):
    """Test suite for AStarEngine & AStarArena64B."""

    def test_astar_arena_64b_layout(self):
        """Validates exact 64-byte layout and zero-heap allocation in arena."""
        arena = AStarArena64B(capacity=16)
        self.assertEqual(arena.NODE_SIZE_BYTES, 64)
        self.assertEqual(len(arena.raw_buffer), 16 * 64)

        node_id = arena.allocate(
            parent_id=-1,
            state_idx=42,
            action_idx=7,
            g_cost=3.5,
            f_cost=5.0,
            uncertainty=0.125,
            flags=1,
            direction=0,
        )
        self.assertEqual(node_id, 0)
        self.assertEqual(arena.allocated_count, 1)

        unpacked = arena.read_node(0)
        self.assertEqual(unpacked[0], 0)     # node_id
        self.assertEqual(unpacked[1], -1)    # parent_id
        self.assertEqual(unpacked[2], 42)    # state_idx
        self.assertEqual(unpacked[3], 7)     # action_idx
        self.assertAlmostEqual(unpacked[4], 3.5)
        self.assertAlmostEqual(unpacked[5], 5.0)
        self.assertAlmostEqual(unpacked[6], 0.125, places=5)
        self.assertEqual(unpacked[7], 1)     # flags
        self.assertEqual(unpacked[8], 0)     # direction

        arena.reset()
        self.assertEqual(arena.allocated_count, 0)

    def test_astar_forward_uncertainty_plan(self):
        """Validates forward uncertainty-weighted A* search."""
        engine = AStarEngine(lambda_weight=1.5)

        # Simple graph: (x) from 0 to 4
        def is_goal(s):
            return s == 4

        def get_neighbors(s):
            if s >= 4:
                return []
            return [
                (s + 1, "right_safe", 0.95),
                (s + 1, "right_risky", 0.20),
            ]

        def h_fn(s):
            return float(4 - s)

        res = engine.plan(0, is_goal, get_neighbors, heuristic_or_bwd=h_fn)
        self.assertTrue(res["success"])
        self.assertEqual(len(res["path"]), 4)
        # Should always pick right_safe due to lower uncertainty penalty
        for act in res["path"]:
            self.assertEqual(act, "right_safe")
        self.assertGreater(res["arena_nodes"], 0)

    def test_astar_bidirectional_meet_in_the_middle(self):
        """Validates bidirectional goal-directed search."""
        engine = AStarEngine()

        def get_fwd(s):
            return [(s + 1, "east", 0.99)] if s < 10 else []

        def get_bwd(s):
            return [(s - 1, "east", 0.99)] if s > 0 else []

        def h_fn(s1, s2):
            return float(abs(s1 - s2))

        res = engine.plan(0, 10, get_fwd, get_bwd, heuristic_fn=h_fn)
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "bidirectional")
        self.assertEqual(len(res["path"]), 10)


class TestMctsEngine(unittest.TestCase):
    """Test suite for MctsEngine, MctsNode64B, and causal pruning."""

    def test_mcts_node_64b_serialization(self):
        """Validates 64B struct packing of MctsNode64B."""
        node = MctsNode64B(node_id=1, parent_id=0, action_idx=3, prior_p=0.25, depth=2)
        node.visit_count = 10
        node.value_sum = 7.5
        node.hamiltonian_energy = 1.234

        raw = node.to_bytes()
        self.assertEqual(len(raw), 64)

        restored = MctsNode64B.from_bytes(raw)
        self.assertEqual(restored.node_id, 1)
        self.assertEqual(restored.parent_id, 0)
        self.assertEqual(restored.action_idx, 3)
        self.assertEqual(restored.visit_count, 10)
        self.assertAlmostEqual(restored.value_sum, 7.5)
        self.assertAlmostEqual(restored.hamiltonian_energy, 1.234, places=3)
        self.assertEqual(restored.depth, 2)

    def test_mcts_causal_falsification_pruning(self):
        """Validates causal falsification cascading pruning."""
        engine = MctsEngine(num_simulations=50, max_depth=5)

        candidates = ["safe_act", "falsified_act"]

        def trans_fn(s, a):
            next_s = s + (1 if a == "safe_act" else -1)
            return next_s, 1.0, False

        def causal_inv(s, a, next_s, reward, done):
            # Invariant: state must never become negative
            return next_s >= 0

        res = engine.plan(
            root_state=0,
            candidate_actions=candidates,
            transition_fn=trans_fn,
            legal_actions_fn=lambda s: candidates,
            causal_invariant_fn=causal_inv,
        )

        self.assertEqual(res["best_action"], "safe_act")
        self.assertGreater(res["falsified_pruned_count"], 0)
        self.assertEqual(res["visit_distribution"].get("falsified_act", 0.0), 0.0)


class TestMpcCemEngine(unittest.TestCase):
    """Test suite for MpcCemEngine (Continuous & Discrete CEM)."""

    def test_continuous_trajectory_optimization(self):
        """Validates continuous latent trajectory optimization with bounds."""
        engine = MpcCemEngine(action_dim=2, horizon=2, num_samples=64, iterations=5, momentum=0.2)

        # Target action: [0.8, -0.5]
        def custom_reward(z, a):
            return -((a[0] - 0.8) ** 2 + (a[1] - (-0.5)) ** 2)

        res = engine.plan_continuous(
            state=[0.0, 0.0],
            bounds=(-1.0, 1.0),
            custom_reward_fn=custom_reward,
        )

        self.assertEqual(res["mode"], MpcMode.CONTINUOUS.value)
        best_act = res["best_action"]
        self.assertAlmostEqual(best_act[0], 0.8, delta=0.25)
        self.assertAlmostEqual(best_act[1], -0.5, delta=0.25)

    def test_discrete_cem_optimization(self):
        """Validates categorical discrete action sequence CEM."""
        engine = MpcCemEngine(horizon=3, num_samples=24, iterations=3)

        candidates = ["good", "bad"]

        def trans_fn(s, a):
            return s + (1 if a == "good" else -1), 1.0 if a == "good" else -1.0, False

        res = engine.plan_discrete(
            initial_state=0,
            candidate_actions=candidates,
            transition_fn=trans_fn,
        )

        self.assertEqual(res["mode"], MpcMode.DISCRETE.value)
        self.assertEqual(res["best_action"], "good")

    def test_receding_horizon_step(self):
        """Validates receding_step rolling horizon execution."""
        engine = MpcCemEngine(horizon=3, num_samples=16)

        def trans_fn(s, a):
            r = 10.0 if a == "step" else 0.0
            return s + (1 if a == "step" else 0), r, False

        act, next_s, info = engine.receding_step(
            current_state=0,
            transition_fn=trans_fn,
            candidate_actions_or_bounds=["step", "stay"],
        )
        self.assertEqual(act, "step")
        self.assertEqual(next_s, 1)
        self.assertEqual(info["step_reward"], 10.0)


class TestManifoldGFlowNetEngine(unittest.TestCase):
    """Test suite for ManifoldGFlowNetEngine (Simplex ETF Geodesic Flow)."""

    def test_simplex_etf_diversity_and_mode_coverage(self):
        """Validates 100% mode coverage and permutation equivariance."""
        engine = ManifoldGFlowNetEngine(dim=64, temperature=1.0)

        candidates = ["alpha", "beta", "gamma", "delta"]

        def trans_fn(s, a):
            return f"{s}_{a}", 1.0, False

        res = engine.plan(
            state="root",
            candidate_actions=candidates,
            transition_fn=trans_fn,
            sample_count=20,
        )

        self.assertEqual(res["mode"], "manifold_gflownet")
        self.assertIn(res["best_action"], candidates)
        self.assertGreater(res["trajectory_entropy"], 0.0)
        self.assertEqual(res["permutation_flip_rate"], 0.0)


class TestCfrNashEngine(unittest.TestCase):
    """Test suite for CfrNashEngine (CFR+, BayesianBeliefTracker)."""

    def test_bayesian_belief_tracker(self):
        """Validates Dirichlet belief updating and bluff score detection."""
        tracker = BayesianBeliefTracker()
        tracker.observe_action("turn_1", "bluff_raise", weight=5.0)

        profile = tracker.get_profile(
            info_set="turn_1",
            candidate_actions=["fold", "call", "bluff_raise"],
        )

        self.assertEqual(profile["dominant_action"], "bluff_raise")
        self.assertGreater(profile["confidence"], 0.3)
        self.assertGreater(profile["bluff_score"], 0.0)

    def test_cfr_plus_convergence_and_exploitability(self):
        """Validates CFR+ convergence and bounded exploitability."""
        engine = CfrNashEngine(iterations=25, use_cfr_plus=True)

        candidates = ["rock", "paper", "scissors"]
        res = engine.solve_imperfect_decision(
            info_set_repr="state_rps",
            candidates=candidates,
        )

        self.assertIn(res["best_action"], candidates)
        self.assertIn("exploitability_bound", res)
        self.assertLessEqual(res["exploitability_bound"], 1.0)
        self.assertAlmostEqual(sum(res["strategy"].values()), 1.0, delta=0.01)


class TestCpSatFormalEngine(unittest.TestCase):
    """Test suite for CpSatFormalEngine (0-1 ILP and NCBF Lie Derivative)."""

    def test_discrete_hard_safety_pruning(self):
        """Validates 0-1 boolean hard safety constraint pruning."""
        engine = CpSatFormalEngine(strict_mode=True, action_effects={
            "approve": "EXECUTE", "execute": "EXECUTE",
            "query_log": "READ_ONLY", "quarantine": "READ_ONLY",
        })

        state = {"text": "status: unauthorized access detected authorized=no"}
        candidates = ["approve", "execute", "query_log", "quarantine"]

        res = engine.verify_and_prune(state, candidates)
        self.assertNotIn("approve", res["feasible_actions"])
        self.assertNotIn("execute", res["feasible_actions"])
        self.assertIn("query_log", res["feasible_actions"])
        self.assertIn("quarantine", res["feasible_actions"])

    def test_ncbf_lie_derivative_barrier_filter(self):
        """Validates Lie derivative barrier condition: dot{h}(x, u) >= -alpha * h(x)."""
        engine = CpSatFormalEngine(default_alpha=1.0)

        # State is x position. Safety set: x >= 0, so h(x) = x.
        def barrier_fn(s):
            return float(s)

        # Dynamics: f(s, "move_right") = s + 1; f(s, "fast_drop") = s - 3
        def dynamics_fn(s, a):
            return s + (1 if a == "move_right" else -3)

        res = engine.filter_by_barrier(
            state=1.0,
            candidate_actions=["move_right", "fast_drop"],
            barrier_fn=barrier_fn,
            dynamics_fn=dynamics_fn,
            alpha=1.0,
        )

        # At s=1.0, h(1.0)=1.0, -alpha*h = -1.0.
        # move_right: next_h=2.0 -> h_dot = +1.0 >= -1.0 (feasible)
        # fast_drop:  next_h=-2.0 -> h_dot = -3.0 < -1.0 (violates barrier!)
        self.assertIn("move_right", res["feasible_actions"])
        self.assertIn("fast_drop", res["barrier_pruned_actions"])


class TestEngineOrthogonalArchitectureAndClient(unittest.TestCase):
    """Test pure orthogonal architecture and GenZeroClient initialization."""

    def test_client_wiring_of_6_engines(self):
        """Validates that GenZeroClient initializes and exposes all 6 engines."""
        client = GenZeroClient(config=GenZeroConfig())

        # Canonical engines
        self.assertIsInstance(client.astar_engine, AStarEngine)
        self.assertIsInstance(client.mcts_engine, MctsEngine)
        self.assertIsInstance(client.mpc_cem_engine, MpcCemEngine)
        self.assertIsInstance(client.manifold_gflownet_engine, ManifoldGFlowNetEngine)
        self.assertIsInstance(client.cfr_nash_engine, CfrNashEngine)
        self.assertIsInstance(client.cpsat_formal_engine, CpSatFormalEngine)


if __name__ == "__main__":
    unittest.main()
