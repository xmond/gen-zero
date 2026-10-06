"""Tests for the GGUFCompilationPipeline: VRAM budgeting and deployment manifest generation.

The contrastive data pipeline, Qwen post-trainer and decision-foundation benchmark tests
moved to gen-zero-research with the code they tested.
"""

import unittest
import os
import json
import math

from gen_zero.gateway.gguf_pipeline import GGUFCompilationPipeline






class TestGGUFCompilationPipeline(unittest.TestCase):
    """Verifies Milestone 3: GGUF Quantization & llama-server Deployment."""

    def setUp(self):
        self.pipeline = GGUFCompilationPipeline(
            base_model="Qwen/Qwen3.5-9B",
            server_endpoint="http://127.0.0.1:8080"
        )

    def test_hardware_budget_estimation(self):
        budget_q4 = self.pipeline.estimate_hardware_budget("q4_k_m")
        self.assertEqual(budget_q4["quant_type"], "q4_k_m")
        self.assertLess(budget_q4["weight_vram_gb"], 6.0)
        self.assertLess(budget_q4["total_recommended_vram_gb"], 8.0)
        self.assertTrue(budget_q4["within_100ms_sla"])

        budget_q8 = self.pipeline.estimate_hardware_budget("q8_0")
        self.assertEqual(budget_q8["quant_type"], "q8_0")
        self.assertGreater(budget_q8["weight_vram_gb"], 8.0)

    def test_deployment_manifest_generation(self):
        manifest = self.pipeline.generate_deployment_manifest(
            lora_checkpoint_path="checkpoints/qwen_lora_epoch3.pt",
            quant_type="q4_k_m"
        )
        self.assertEqual(manifest["quant_type"], "q4_k_m")
        self.assertEqual(manifest["compatible_endpoint"], "POST /v1/score")
        self.assertIn("llama-server", manifest["launch_command"])
        self.assertIn("qwen3.5-9b-decision.q4_k_m.gguf", manifest["launch_command"])




if __name__ == "__main__":
    unittest.main()
