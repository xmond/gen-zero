"""Unit Tests for Phase 1 (IRM Causal Distillation) and Phase 2 (Counterfactual Synthetic Augmentation)."""

import unittest
from gen_zero.client import GenZero
from gen_zero.train.replay_buffer import StabilityReplayBuffer
from gen_zero.train.invariant_distiller import InvariantCausalDistiller
from gen_zero.causal.synthetic_generator import CounterfactualSyntheticGenerator


class TestIRMandSyntheticAugmentation(unittest.TestCase):
    def setUp(self):
        self.client = GenZero()
        self.buffer = self.client.replay_buffer

    def test_counterfactual_synthetic_augmentation(self):
        # Create a sample trajectory with 3 steps
        traj = [
            {
                "step_idx": 0,
                "state": {"head": [3, 3], "food": [6, 3]},
                "action": "east",
                "next_state": {"head": [4, 3], "food": [6, 3]},
                "candidates": ["north", "south", "east", "west"]
            },
            {
                "step_idx": 1,
                "state": {"head": [4, 3], "food": [6, 3]},
                "action": "east",
                "next_state": {"head": [5, 3], "food": [6, 3]},
                "candidates": ["north", "south", "east", "west"]
            },
            {
                "step_idx": 2,
                "state": {"head": [5, 3], "food": [6, 3]},
                "action": "north", # Suboptimal turn away from food
                "next_state": {"head": [5, 2], "food": [6, 3]},
                "candidates": ["north", "south", "east", "west"]
            }
        ]

        generator = CounterfactualSyntheticGenerator()
        twins = generator.augment_trajectory(traj, final_outcome="horizon_survived", final_score=5.0)

        # In each step, alternative actions (north, south, west for step 0) produce valid twins
        self.assertGreater(len(twins), 0)
        for twin in twins:
            self.assertTrue(twin["is_synthetic"])
            self.assertIn("soft_target", twin)
            self.assertIn("value_target", twin)
            self.assertIn(twin["environment_id"], "env_synthetic_counterfactual")

        # Test buffer injection via client
        res = self.client.augment_synthetic_counterfactuals(traj, final_outcome="horizon_survived", final_score=5.0)
        self.assertGreater(res["synthetic_twins_generated"], 0)
        self.assertGreaterEqual(res["sample_efficiency_multiplier"], 1.5)
        self.assertGreater(len(self.client.replay_buffer), 0)

    def test_invariant_causal_distiller(self):
        # Populate buffer with gold anchor, hard mined, and synthetic samples
        gold_samples = [
            {
                "id": f"gold:{i}",
                "type": "choice",
                "state": {"pos": [i, i]},
                "candidate_ids": ["north", "east"],
                "soft_target": {"north": 0.8, "east": 0.2},
                "value_target": 1.0,
                "environment_id": "env_gold_anchor"
            }
            for i in range(10)
        ]
        self.buffer.load_gold_samples(gold_samples)

        distiller = InvariantCausalDistiller(
            model=self.client.model,
            replay_buffer=self.buffer,
            irm_lambda=0.1
        )

        batch = self.buffer.sample_batch(batch_size=8)
        envs = distiller.partition_into_environments(batch)
        self.assertGreater(len(envs), 0)

        step_res = distiller.train_step(batch)
        self.assertIn("loss", step_res)
        self.assertIn("irm_penalty", step_res)
        self.assertIn("invariance_score", step_res)
        self.assertGreaterEqual(step_res["invariance_score"], 0.0)

        # Test iteration via client
        iter_res = self.client.run_invariant_distillation(steps=5, batch_size=8)
        self.assertEqual(iter_res["steps_trained"], 5)
        self.assertIn("mean_loss", iter_res)
        self.assertIn("invariance_score", iter_res)


if __name__ == "__main__":
    unittest.main()
