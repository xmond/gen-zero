"""Unit tests for Gen-Zero Direction 2: Pure CPU INT8 Extreme Quantized Runtime."""

import unittest
import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from gen_zero.runtime.nano_core import NanoGenZeroCore
from gen_zero.runtime.quantized_engine import QuantizedTensor, QuantizedCandidateScorer
from gen_zero.client import GenZero


class TestQuantizedRuntime(unittest.TestCase):

    def setUp(self):
        self.state_dim = 1024
        self.candidate_dim = 1024
        self.embed_dim = 128

    def test_quantized_tensor_int8_accuracy(self):
        """Verify INT8 symmetric quantization preserves dynamic range and achieves 4x compression."""
        fp32_data = np.random.randn(128, 1024).astype(np.float32)
        q_tensor = QuantizedTensor.quantize(fp32_data)

        self.assertEqual(q_tensor.data.dtype, np.int8)
        # Size in memory should be ~1/4 of float32
        self.assertLess(q_tensor.data.nbytes, fp32_data.nbytes * 0.3)

        deq_data = q_tensor.dequantize()
        # Max quantization error bounded by scale / 2
        max_error = np.max(np.abs(fp32_data - deq_data))
        self.assertLess(max_error, q_tensor.scale)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for NanoGenZeroCore weight export test")
    def test_nano_core_export_to_quantized_engine(self):
        """Verify PyTorch NanoCore parameters can be exported and ingested by the INT8 engine."""
        core = NanoGenZeroCore(
            state_dim=self.state_dim,
            candidate_dim=self.candidate_dim,
            embed_dim=self.embed_dim
        )
        core.eval()

        weight_dict = core.export_weight_dict()
        self.assertIn("state_proj.weight", weight_dict)
        self.assertIn("score_mlp.0.weight", weight_dict)

        scorer = QuantizedCandidateScorer(
            state_dim=self.state_dim,
            candidate_dim=self.candidate_dim,
            embed_dim=self.embed_dim
        )
        scorer.load_from_weight_dict(weight_dict)

        # Ensure memory footprint is minimal
        mem_info = scorer.get_memory_footprint()
        self.assertLess(mem_info["weights_memory_mb"], 35.0)

    def test_quantized_candidate_scorer_sub_4ms_latency(self):
        """Verify Pure CPU candidate scoring executes in < 4.0ms without PyTorch runtime dependency."""
        scorer = QuantizedCandidateScorer(
            state_dim=self.state_dim,
            candidate_dim=self.candidate_dim,
            embed_dim=self.embed_dim
        )

        state = np.random.randn(self.state_dim).astype(np.float32)
        candidates = ["BUY_CALL", "BUY_PUT", "SELL_STRADDLE", "HOLD_CASH", "DELTA_HEDGE"]

        # Warm up
        _ = scorer.score_candidates(state, candidates)

        # Measure 10 rounds
        latencies = []
        for _ in range(10):
            res = scorer.score_candidates(state, candidates)
            latencies.append(res["scoring_latency_ms"])

        mean_latency = sum(latencies) / len(latencies)
        self.assertIn(res["best_action"], candidates)
        self.assertEqual(len(res["probabilities"]), 5)
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=4)
        self.assertTrue(-1.0 <= res["expected_value"] <= 1.0)
        self.assertEqual(res["device"], "CPU_extreme")

        # Crucial KPI assertion: sub-4ms on standard CPU
        self.assertLess(mean_latency, 4.0, f"Mean latency {mean_latency}ms exceeded 4.0ms KPI")

    def test_client_decide_cpu_extreme_integration(self):
        """Verify GenZero top-level client exposes decide_cpu_extreme API."""
        client = GenZero()
        candidates = ["LANE_1", "LANE_2", "LANE_3", "EMERGENCY_STOP"]

        # State as list
        state_list = [0.1] * 1024

        # Warm up
        _ = client.decide_cpu_extreme(state_repr=state_list, candidates=candidates, temperature=1.0)

        res = client.decide_cpu_extreme(
            state_repr=state_list,
            candidates=candidates,
            temperature=1.0
        )

        self.assertIn(res["best_action"], candidates)
        self.assertGreaterEqual(res["confidence"], 0.25)
        self.assertGreater(res["scoring_latency_ms"], 0.0)
        self.assertLess(res["scoring_latency_ms"], 500.0)

        # State as dict
        state_dict = {"features": [0.2] * 1024, "tag": "sensor_frame"}
        res2 = client.decide_cpu_extreme(
            state_repr=state_dict,
            candidates=candidates
        )
        self.assertIn(res2["best_action"], candidates)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for multi-dimension hidden_size tests")
    def test_dynamic_hidden_size_adaptation_across_model_families(self):
        """Verifies seamless initialization and execution across 896, 1024, 2048, 3584, 4096 dimensions."""
        model_family_dimensions = [
            (896, "Qwen2.5-0.5B"),
            (1024, "Qwen3.5-0.8B"),
            (2048, "Qwen3.5-4B"),
            (3584, "Qwen2.5-7B"),
            (4096, "Qwen3.5-9B"),
        ]

        for hidden_size, family_name in model_family_dimensions:
            with self.subTest(family=family_name, hidden_size=hidden_size):
                # 1. Initialize PyTorch NanoCore with dynamic state_dim
                core = NanoGenZeroCore(state_dim=hidden_size, candidate_dim=hidden_size, embed_dim=128)
                state_t = torch.randn(2, hidden_size)
                cand_t = torch.randn(2, 4, hidden_size)
                logits, value = core(state_t, cand_t)

                self.assertEqual(logits.shape, (2, 4))
                self.assertEqual(value.shape, (2, 1))

                # 2. Export with dimension metadata
                weight_dict = core.export_weight_dict()
                self.assertIn("__metadata__", weight_dict)
                meta = weight_dict["__metadata__"]
                self.assertEqual(meta[0], hidden_size)
                self.assertEqual(meta[1], hidden_size)

                # 3. Ingest into INT8 QuantizedCandidateScorer
                scorer = QuantizedCandidateScorer(state_dim=512)  # different initial state_dim
                scorer.load_from_weight_dict(weight_dict)
                self.assertEqual(scorer.state_dim, hidden_size)

                # 4. Score candidates on pure CPU with new dimension
                state_arr = np.random.randn(hidden_size).astype(np.float32)
                res = scorer.score_candidates(state_repr=state_arr, candidates=["ACT_A", "ACT_B", "ACT_C"])
                self.assertEqual(len(res["probabilities"]), 3)
                self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=4)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for dimension mismatch assertion test")
    def test_forward_dimension_mismatch_raises_assertion(self):
        """Verifies that passing a mismatched state representation raises ValueError instead of crashing."""
        core = NanoGenZeroCore(state_dim=3584, candidate_dim=3584)
        mismatched_state = torch.randn(1, 1024)
        valid_cands = torch.randn(1, 3, 3584)

        with self.assertRaises(ValueError) as ctx:
            core(mismatched_state, valid_cands)
        self.assertIn("State dimension mismatch: expected 3584, got 1024", str(ctx.exception))

    def test_from_config_factory_resolution(self):
        """Verifies from_config correctly extracts hidden_size from mock HF config."""
        class MockConfig:
            hidden_size = 3584

        config = MockConfig()
        scorer = QuantizedCandidateScorer.from_config(config)
        self.assertEqual(scorer.state_dim, 3584)
        self.assertEqual(scorer.candidate_dim, 3584)

        if HAS_TORCH:
            core = NanoGenZeroCore.from_config(config)
            self.assertEqual(core.state_dim, 3584)
            self.assertEqual(core.candidate_dim, 3584)


if __name__ == "__main__":
    unittest.main()

