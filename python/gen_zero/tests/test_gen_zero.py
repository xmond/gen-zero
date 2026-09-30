"""Unit tests for Gen-Zero Core Components."""

import unittest
import math
import copy
from gen_zero.config import GenZeroConfig
from gen_zero.planner.engines import AStarEngine, MctsEngine
from gen_zero.world_model import GenZeroTextWorldModel
from gen_zero.rollout.hard_miner import HardSampleMiner
from gen_zero.gate.safety_gate import SafetyGate, GateVerdict


class TestGenZeroComponents(unittest.TestCase):
    def test_astar_planner(self):
        planner = AStarEngine(lambda_weight=1.0)
        # Simple graph from 0 to 3: 0 -> 1 -> 3 (p=0.9 each), 0 -> 2 -> 3 (p=0.1 each)
        def get_neighbors(node):
            if node == 0:
                return [(1, "to_1", 0.95), (2, "to_2", 0.05)]
            elif node == 1:
                return [(3, "to_3", 0.95)]
            elif node == 2:
                return [(3, "to_3", 0.05)]
            return []

        res = planner.plan(
            start_state=0,
            target_or_is_goal=lambda s: s == 3,
            get_neighbors_or_fwd=get_neighbors,
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["path"], ["to_1", "to_3"])

    def test_text_world_model_requires_uncalibrated_acknowledgement(self):
        with self.assertRaises(ValueError):
            GenZeroTextWorldModel()

    def test_text_world_model(self):
        wm = GenZeroTextWorldModel(acknowledge_uncalibrated=True)
        state = {
            "size": 8,
            "body": [[4, 4], [4, 3], [4, 2]],
            "direction": "east",
            "food": [4, 5]
        }
        # Moving east into food
        conseq = wm.predict_consequences(state, "east")
        self.assertLess(conseq["p_fail"], 0.5)
        self.assertGreater(conseq["p_reward"], 0.5)

        # Virtual step forward
        next_s, r, term = wm.virtual_step(state, "east")
        self.assertFalse(term)
        self.assertEqual(next_s["body"][0], [4, 5])
        self.assertGreater(r, 1.0)


    def test_safety_gate(self):
        gate = SafetyGate(min_acc_retention=0.99, min_score_gain=0.0)
        base = {"accuracy": 85.0, "mean_score": 20.0, "collision_rate": 0.10, "is_valid": True}
        
        # Candidate 1: Better score and accuracy
        cand1 = {"accuracy": 86.0, "mean_score": 22.0, "collision_rate": 0.05, "is_valid": True}
        verdict1 = gate.evaluate_candidate(base, cand1)
        self.assertTrue(verdict1.passed)
        self.assertEqual(verdict1.action, "DEPLOY_HOT_UPDATE")

        # Candidate 2: Severe accuracy regression
        cand2 = {"accuracy": 70.0, "mean_score": 21.0, "collision_rate": 0.15, "is_valid": True}
        verdict2 = gate.evaluate_candidate(base, cand2)
        self.assertFalse(verdict2.passed)
        self.assertEqual(verdict2.action, "ROLLBACK_ADJUST_HYPERPARAMS")


    def test_adaptive_parameter_engine(self):
        from gen_zero.adaptive_engine import AdaptiveParameterEngine
        # 1. Entropy calculation
        ent_low = AdaptiveParameterEngine.get_normalized_entropy({"a": 0.99, "b": 0.01})
        ent_high = AdaptiveParameterEngine.get_normalized_entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
        self.assertLess(ent_low, 0.2)
        self.assertGreater(ent_high, 0.95)

        # 2. Dynamic MCTS budget
        budget_low = AdaptiveParameterEngine.dynamic_mcts_budget({}, policy_entropy=0.1)
        budget_high = AdaptiveParameterEngine.dynamic_mcts_budget({}, policy_entropy=0.9)
        self.assertLess(budget_low, budget_high)
        self.assertGreaterEqual(budget_low, 8)
        self.assertLessEqual(budget_high, 256)

        # 3. Dynamic A* lambda
        lam_low = AdaptiveParameterEngine.dynamic_astar_lambda({}, policy_entropy=0.9)
        lam_high = AdaptiveParameterEngine.dynamic_astar_lambda({}, policy_entropy=0.1)
        self.assertLess(lam_low, lam_high)

        # 4. Dynamic MPC
        h_calm, s_calm = AdaptiveParameterEngine.dynamic_mpc_params(volatility=0.001)
        h_vol, s_vol = AdaptiveParameterEngine.dynamic_mpc_params(volatility=0.05)
        self.assertLessEqual(h_calm, h_vol)
        self.assertLessEqual(s_calm, s_vol)

        # 5. Dynamic ATR risk
        highs = [100.0 + i for i in range(25)]
        lows = [98.0 + i for i in range(25)]
        closes = [99.0 + i for i in range(25)]
        stop, tp = AdaptiveParameterEngine.dynamic_quant_risk_atr(highs, lows, closes)
        self.assertGreater(tp, stop)
        self.assertGreater(stop, 0.0)

    def test_dynamic_moe_decide(self):
        from gen_zero import GenZero
        engine = GenZero()
        state = {"size": 8, "body": [[2, 2]], "food": [2, 3]}
        candidates = ["north", "east", "south", "west"]
        res = engine.decide(state, candidates, mode="auto")
        self.assertIn("action", res)
        self.assertIn("adaptive_params", res)
        self.assertIn("k_experts", res["adaptive_params"])
        self.assertIn(res["action"], candidates)



    def test_opponent_belief_tracker(self):
        from gen_zero.planner.engines import BayesianBeliefTracker
        tracker = BayesianBeliefTracker(prior_weight=1.0)
        cands = ["fold", "call", "raise"]

        # Initial profile with 0 observations: uniform posterior & 0 confidence
        p0 = tracker.get_profile("river_check", cands)
        self.assertEqual(p0["sample_count"], 0.0)
        self.assertEqual(p0["confidence"], 0.0)

        # Feed 15 consecutive "fold" observations (revealing a tight/fearful opponent)
        for _ in range(15):
            tracker.observe_action("river_check", "fold")

        p1 = tracker.get_profile("river_check", cands)
        self.assertEqual(p1["sample_count"], 15.0)
        self.assertGreater(p1["confidence"], 0.5)
        self.assertGreater(p1["bias_kl"], 0.3)
        self.assertEqual(p1["dominant_action"], "fold")
        self.assertGreater(p1["posterior"]["fold"], 0.8)

    def test_safe_cfr_exploitation(self):
        from gen_zero.planner.engines import CfrNashEngine
        expert = CfrNashEngine(iterations=50)
        info_set = "blind_turn_bluff"
        cands = ["call", "fold", "raise"]

        # 1. Round 1 without opponent data: plays pure safe Nash equilibrium (beta = 1.0)
        res1 = expert.solve_imperfect_decision(info_set, cands)
        self.assertFalse(res1["is_exploiting"])
        self.assertAlmostEqual(res1["exploitation_beta"], 1.0, places=2)

        # 2. Feed 20 observed opponent "fold" mistakes
        for _ in range(20):
            expert.record_opponent_action(info_set, "fold")

        # 3. Round 2: detects exploitable bias and activates safe best response exploitation
        res2 = expert.solve_imperfect_decision(info_set, cands)
        self.assertTrue(res2["is_exploiting"])
        self.assertLess(res2["exploitation_beta"], 0.50)
        # Exploitation strategy significantly raises aggression to exploit opponent folds
        self.assertGreater(res2["strategy"]["raise"], res1["strategy"]["raise"])


    def test_cpsat_infeasible_abstain(self):
        from gen_zero.planner.engines import CpSatFormalEngine
        from gen_zero import GenZero
        solver = CpSatFormalEngine()
        solver.register_hard_rule(lambda s, a: False)  # Rule rejects everything
        res = solver.verify_and_prune({"pos": [0, 0]}, ["left", "right"])
        self.assertEqual(res["feasible_actions"], [])
        self.assertEqual(res["status"], "INFEASIBLE_ABSTAIN")

        # In client
        engine = GenZero()
        engine.cpsat_formal_engine.register_hard_rule(lambda s, a: False)
        dec = engine.decide({"pos": [0, 0]}, ["left", "right"], mode="cp_sat")
        self.assertEqual(dec["action"], "ABSTAIN")
        self.assertEqual(dec["status"], "INFEASIBLE_ABSTAIN")



    def test_client_astar_routing(self):
        from gen_zero import GenZero
        engine = GenZero()
        state = {
            "size": 8,
            "body": [[2, 2]],
            "food": [2, 3]  # Food is east of head
        }
        res = engine.decide(state, ["north", "east", "south", "west"], mode="astar")
        self.assertEqual(res["action"], "east")
        self.assertGreater(res["confidence"], 0.5)

    def test_latent_world_model(self):
        from gen_zero.world_model import LatentTransitionModel
        model = LatentTransitionModel(latent_dim=16, action_dim=3, hidden_dim=32)
        z0 = [1.0] * 16
        self.assertEqual(len(z0), 16)

        # Single step in latent space
        z1, r, v = model.step(z0, [0.1, -0.2, 0.5])
        self.assertEqual(len(z1), 16)
        self.assertIsInstance(r, float)
        self.assertIsInstance(v, float)

        # Trajectory rollout in latent space
        acts = [[0.1, 0.2, 0.3] for _ in range(5)]
        roll = model.rollout(z0, acts)
        self.assertEqual(roll["horizon"], 5)
        self.assertEqual(len(roll["latent_trajectory"]), 6)
        self.assertIn("cumulative_return", roll)

    def test_continuous_mpc_decision(self):
        from gen_zero import GenZero
        engine = GenZero()

        # 1. Simplex action (e.g. portfolio weights sum to 1.0, non-negative)
        res_simplex = engine.decide_continuous(
            state={"volatility": 0.02, "feature_1": 1.5},
            action_dim=4,
            simplex=True,
            horizon=5,
            num_samples=24
        )
        self.assertEqual(len(res_simplex["action"]), 4)
        self.assertAlmostEqual(sum(res_simplex["action"]), 1.0, places=2)
        for w in res_simplex["action"]:
            self.assertGreaterEqual(w, 0.0)

        # 2. Box-bounded continuous action
        res_box = engine.decide_continuous(
            state=[0.5, -0.5, 1.0],
            action_dim=3,
            bounds=(-2.0, 2.0),
            simplex=False,
            horizon=4,
            num_samples=20
        )
        self.assertEqual(len(res_box["action"]), 3)
        for val in res_box["action"]:
            self.assertGreaterEqual(val, -2.0)
            self.assertLessEqual(val, 2.0)

    def test_prm_verification_and_prior_modulation(self):
        from gen_zero.model.prm import ProcessRewardModel
        prm = ProcessRewardModel()

        # 1. Healthy step
        res_ok = prm.verify_step({"pos": [0, 0]}, "east", {"pos": [0, 1]}, is_done=False, step_reward=1.0)
        self.assertFalse(res_ok["should_prune"])
        self.assertGreater(res_ok["viability"], 0.8)

        # 2. Fatal collision step
        res_fatal = prm.verify_step({"pos": [0, 0]}, "west", {"pos": [0, 0]}, is_done=True, step_reward=-10.0)
        self.assertTrue(res_fatal["should_prune"])
        self.assertEqual(res_fatal["deadlock_risk"], 1.0)

        # 3. Pocket trap deadlock step in grid (enclosed corner)
        deadlock_s = {
            "size": 6,
            "body": [[0, 0], [0, 1], [1, 0], [1, 1]], # head at [0, 0] enclosed by body and walls
            "food": [5, 5]
        }
        res_trap = prm.verify_step({"size": 6, "body": [[0, 0]]}, "south", deadlock_s)
        self.assertTrue(res_trap["deadlock_risk"] > 0.5)

        # 4. Modulate priors
        priors = {"safe": 0.5, "trap": 0.5}
        evals = {
            "safe": {"viability": 1.0, "should_prune": False},
            "trap": {"viability": 0.05, "should_prune": True}
        }
        mod_priors = prm.modulate_priors(priors, evals)
        self.assertGreater(mod_priors["safe"], 0.95)
        self.assertLess(mod_priors["trap"], 0.05)

    def test_prm_guided_mcts_pruning(self):
        from gen_zero.planner.engines import MctsEngine
        from gen_zero.model.prm import ProcessRewardModel
        planner = MctsEngine(num_simulations=32, max_depth=5)
        prm = ProcessRewardModel()

        def trans_fn(s, a):
            if a == "fatal":
                return {"status": "fatal"}, -10.0, True
            return s, 1.0, False

        def inv_fn(s, a, next_s, reward, done):
            v = prm.verify_step(s, a, next_s, is_done=done, step_reward=reward)
            return not v.get("should_prune", False)

        res = planner.plan(
            root_state={"status": "normal"},
            candidate_actions=["safe", "fatal"],
            transition_fn=trans_fn,
            legal_actions_fn=lambda s: ["safe", "fatal"],
            causal_invariant_fn=inv_fn,
        )
        self.assertEqual(res["best_action"], "safe")
        self.assertGreater(res["falsified_pruned_count"], 0)


if __name__ == "__main__":
    unittest.main()
