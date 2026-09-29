"""Unit tests for Gen-Zero Direction 1: Streaming Spatial-Temporal World Model."""

import unittest
import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from gen_zero.world_model.rolling_kv_cache import RollingVisionKVCache
from gen_zero.world_model.latent_dynamics import LatentTransitionModel
from gen_zero.world_model.imagination_planner import ImaginationMCTSPlanner
from gen_zero.world_model.streaming_engine import StreamingWorldModelEngine
from gen_zero.client import GenZero


class TestStreamingWorldModel(unittest.TestCase):

    def setUp(self):
        self.feature_dim = 1024

    def test_rolling_kv_cache_constant_memory(self):
        """Verify Attention Sinks retain initial tokens and memory strictly bounds to O(K+W)."""
        cache = RollingVisionKVCache(
            num_sink_tokens=4,
            window_size=8,
            feature_dim=self.feature_dim
        )

        # Feed 100 consecutive frames
        for i in range(100):
            frame_vec = np.full((1, self.feature_dim), float(i), dtype=np.float32)
            cache.append(key=frame_vec, value=frame_vec)

        stats = cache.get_temporal_context()
        self.assertEqual(stats["total_frames_processed"], 100)
        self.assertEqual(stats["sink_tokens"], 4)
        self.assertEqual(stats["window_frames"], 8)
        self.assertEqual(stats["active_tokens"], 12)  # 4 sinks + 8 window

        # Verify Attention Sinks are exactly the first 4 frames [0, 1, 2, 3]
        sink_k, _ = cache.sink_keys, cache.sink_values
        if HAS_TORCH and isinstance(sink_k, torch.Tensor):
            sink_vals = [int(sink_k[idx, 0].item()) for idx in range(4)]
        else:
            sink_vals = [int(sink_k[idx, 0]) for idx in range(4)]
        self.assertEqual(sink_vals, [0, 1, 2, 3])

        # Verify Window retains the most recent 8 frames [92 .. 99]
        k_full, _ = cache.get_kv()
        if HAS_TORCH and isinstance(k_full, torch.Tensor):
            last_val = int(k_full[-1, 0].item())
        else:
            last_val = int(k_full[-1, 0])
        self.assertEqual(last_val, 99)

        # Verify reset
        cache.reset()
        self.assertEqual(cache.get_temporal_context()["active_tokens"], 0)

    def test_latent_transition_dynamics(self):
        """Verify 1-step latent residual transition and causal shock attribution."""
        model = LatentTransitionModel(
            latent_dim=self.feature_dim,
            action_dim=16
        )

        z0 = np.random.randn(self.feature_dim).astype(np.float32)
        next_z, reward, var = model.step(z0, action=2)

        self.assertEqual(len(next_z), self.feature_dim)
        self.assertTrue(-1.0 <= reward <= 1.0)
        self.assertGreater(var, 0.0)

        # Test multi-step rollout
        rollout_res = model.rollout(initial_latent=z0, action_sequence=[0, 1, 2, 3])
        self.assertEqual(rollout_res["horizon"], 4)
        self.assertEqual(len(rollout_res["latent_states"]), 5)
        self.assertEqual(len(rollout_res["rewards"]), 4)

        # Test causal shock: expected state vs shocked state
        pred_z, _, _ = model.step(z0, action=1)
        # 1. Zero shock on perfect match
        shock_zero, _ = model.compute_causal_shock(z0, action=1, real_next_latent=pred_z)
        self.assertAlmostEqual(shock_zero, 0.0, places=4)

        # 2. Positive shock on unmodeled disturbance
        disturbed_z = pred_z + 2.5
        shock_pos, _ = model.compute_causal_shock(z0, action=1, real_next_latent=disturbed_z)
        self.assertGreater(shock_pos, 1.0)

    def test_imagination_mcts_planner(self):
        """Verify PUCT Monte Carlo Tree Search execution entirely in latent space."""
        transition_model = LatentTransitionModel(
            latent_dim=self.feature_dim,
            action_dim=8
        )
        planner = ImaginationMCTSPlanner(
            transition_model=transition_model,
            max_simulations=32,
            max_depth=3
        )

        root_z = np.random.randn(self.feature_dim).astype(np.float32)
        candidates = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "BRAKE"]

        # Warm-up pass to initialize PyTorch allocator and dynamic graphs
        _ = planner.plan(root_latent=root_z, candidate_actions=candidates)

        res = planner.plan(
            root_latent=root_z,
            candidate_actions=candidates,
            action_priors={"FORWARD": 0.5, "TURN_LEFT": 0.2, "TURN_RIGHT": 0.2, "BRAKE": 0.1}
        )

        self.assertIn(res["best_action"], candidates)
        self.assertEqual(len(res["action_probabilities"]), 4)
        prob_sum = sum(res["action_probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)
        self.assertGreater(len(res["imagined_trajectory"]), 0)
        # Latent MCTS is fast: should complete well within reasonable CI threshold
        self.assertLess(res["planning_time_ms"], 150.0)

    def test_streaming_engine_end_to_end(self):
        """Verify StreamingWorldModelEngine continuous multi-frame execution loop."""
        engine = StreamingWorldModelEngine(
            latent_dim=self.feature_dim,
            num_sink_tokens=4,
            window_size=8,
            max_simulations=32
        )

        candidates = ["ACTION_A", "ACTION_B", "ACTION_C"]

        # Run 5 continuous streaming frames
        prev_shock = None
        for step_idx in range(1, 6):
            frame_input = np.random.randn(self.feature_dim).astype(np.float32)
            res = engine.step_stream(
                observation=frame_input,
                candidate_actions=candidates
            )

            self.assertEqual(res["step"], step_idx)
            self.assertIn(res["selected_action"], candidates)
            self.assertEqual(res["kv_cache_stats"]["total_frames_processed"], step_idx)
            self.assertIn("imagination_mcts_ms", res["latency_breakdown_ms"])

            if step_idx > 1:
                # Causal shock is computed from step 2 onwards
                self.assertIsNotNone(res["causal_shock_norm"])

        # Reset stream
        engine.reset_stream()
        self.assertEqual(engine.step_count, 0)
        self.assertEqual(engine.kv_cache.get_temporal_context()["active_tokens"], 0)

    def test_client_decide_streaming_integration(self):
        """Verify GenZero top-level client exposes streaming decision API."""
        client = GenZero()
        candidates = ["SORT_RED", "SORT_BLUE", "HOLD"]

        # Frame 1
        res1 = client.decide_streaming(
            observation="camera_frame_001.jpg",
            candidates=candidates
        )
        self.assertIn(res1["selected_action"], candidates)
        self.assertEqual(res1["step"], 1)

        # Frame 2
        res2 = client.decide_streaming(
            observation="camera_frame_002.jpg",
            candidates=candidates
        )
        self.assertIn(res2["selected_action"], candidates)
        self.assertEqual(res2["step"], 2)

        client.reset_stream()


if __name__ == "__main__":
    unittest.main()
