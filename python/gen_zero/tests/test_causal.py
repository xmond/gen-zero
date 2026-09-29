"""Unit Tests for Gen-Zero Causal Reasoning & Counterfactual Inference."""

import unittest
from gen_zero.causal.counterfactual_engine import (
    StructuralCausalModel,
    CounterfactualEngine,
    CounterfactualResult,
    CausalAttribution
)
from gen_zero.client import GenZero
from gen_zero.rollout.hard_miner import HardSampleMiner


class TestCausalCounterfactual(unittest.TestCase):
    def setUp(self):
        self.scm = StructuralCausalModel()
        self.engine = CounterfactualEngine(scm=self.scm)

    def test_abduction_and_intervention(self):
        # Initial state: head at [5, 5]
        s = {"head": [5, 5], "food": [6, 5]}
        # Action taken: 'east'
        # Observed next state: drifted/slipped to [6, 6] (drift of (0, 1))
        obs_next = {"head": [6, 6], "food": [6, 5]}

        # 1. Abduction: Noise should be (0, 1)
        noise = self.scm.abduce_exogenous_noise(s, "east", obs_next)
        self.assertEqual(noise["coord_drift"], (0, 1))
        self.assertAlmostEqual(noise["norm"], 1.0)
        self.assertTrue(noise["slip_occurred"])

        # 2. Intervention: What if action was 'north'?
        # Nominal north from [5, 5] is [5, 4]. With locked noise (0, 1), it becomes [5, 5]!
        cf_next, r = self.scm.intervene_and_predict(s, "north", noise)
        self.assertEqual(cf_next["head"], [5, 5])

    def test_counterfactual_decision_error_attribution(self):
        # Scenario: Agent at [2, 2], food at [1, 2].
        # Agent chose 'east' -> ended up at [3, 2] moving further away or towards danger.
        # Alternative 'west' -> moves directly towards [1, 2].
        s = {"head": [2, 2], "food": [1, 2]}
        factual_a = "east"
        obs_next = {"head": [3, 2], "food": [1, 2]}

        def evaluator(state):
            hx, hy = state["head"]
            fx, fy = state["food"]
            return -float(abs(hx - fx) + abs(hy - fy))

        cf_res = self.engine.compute_counterfactuals(
            state=s,
            factual_action=factual_a,
            observed_next_state=obs_next,
            candidate_actions=["north", "south", "east", "west"],
            value_evaluator=evaluator,
            factual_return=-2.0
        )

        self.assertTrue(cf_res.is_action_culprit)
        self.assertEqual(cf_res.best_action, "west")
        self.assertGreater(cf_res.max_ite, 0.0)

    def test_trajectory_attribution_and_mining(self):
        traj = [
            {
                "step_idx": 0,
                "state": {"head": [2, 2], "food": [1, 2]},
                "action": "east",
                "next_state": {"head": [3, 2], "food": [1, 2]},
                "candidates": ["north", "south", "east", "west"]
            },
            {
                "step_idx": 1,
                "state": {"head": [3, 2], "food": [1, 2]},
                "action": "east",
                "next_state": {"head": [4, 2], "food": [1, 2]},
                "candidates": ["north", "south", "east", "west"]
            }
        ]

        attrs = self.engine.attribute_trajectory_failures(
            trajectory=traj,
            final_outcome="collision",
            final_score=0.0
        )
        self.assertEqual(len(attrs), 2)
        root_cause = self.engine.find_root_cause_step(attrs)
        self.assertIsNotNone(root_cause)
        self.assertTrue(root_cause.is_decision_culprit)

        # Verify integration with GenZero client
        client = GenZero()
        res = client.attribute_counterfactual(traj, final_outcome="collision", final_score=0.0)
        self.assertTrue(res["has_decision_culprit"])
        self.assertIsNotNone(res["root_cause_step"])

    def test_mcts_metacognitive_reflection(self):
        from gen_zero.planner.engines.mcts_engine import MctsEngine as GenZeroMCTS

        mcts = GenZeroMCTS(simulations=32, max_depth=5, enable_reflection=True)

        # Environment where 'east' leads to immediate trap/deadlock at step 2,
        # but 'north' leads to safety.
        def get_legal(s):
            return ["north", "east"]

        def transition(s, a):
            x, y = s["pos"]
            if a == "east":
                nx, ny = x + 1, y
                done = (nx >= 3)
                r = -10.0 if done else 0.0
                return {"pos": [nx, ny]}, r, done
            else:
                nx, ny = x, y + 1
                done = (ny >= 5)
                r = 1.0 if done else 0.1
                return {"pos": [nx, ny]}, r, done

        def eval_fn(s, actions):
            # Prior biased towards east (deceptive trap)
            priors = {"east": 0.8, "north": 0.2}
            return priors, 0.0

        root_s = {"pos": [0, 0]}
        res = mcts.search(
            root_state=root_s,
            get_legal_actions_fn=get_legal,
            transition_fn=transition,
            eval_fn=eval_fn,
            simulations=32
        )

        # Reflection should trigger, penalize east, and successfully choose north
        self.assertGreater(res["reflections_triggered"], 0)
        self.assertEqual(res["best_action"], "north")
        self.assertIn("reflections_log", res)
        self.assertTrue(len(res["reflections_log"]) > 0)
        self.assertEqual(res["reflections_log"][0]["culprit_action"], "east")
        self.assertEqual(res["reflections_log"][0]["recommended_action"], "north")


if __name__ == "__main__":
    unittest.main()
