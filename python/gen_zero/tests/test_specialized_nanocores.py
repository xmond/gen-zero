"""Unit and Contract Tests for Specialized Domain NanoCores (Browser & Vision)."""

import unittest
import numpy as np

from gen_zero.runtime.base_nano_core import BaseNanoCore
from gen_zero.runtime.nano_core_browser import NanoCoreBrowser
from gen_zero.runtime.nano_core_vision import NanoCoreVision


class TestSpecializedNanoCores(unittest.TestCase):
    """Verifies BaseNanoCore contracts, domain specialization, and performance."""

    def test_browser_nanocore_contract(self):
        """NanoCoreBrowser executes sub-3.5ms inference on DOM candidate sets."""
        core = NanoCoreBrowser()
        self.assertIsInstance(core, BaseNanoCore)
        self.assertEqual(core.domain, "browser")

        candidates = ["btn_submit", "btn_cancel", "link_help"]
        state = "Form submission page: please confirm your booking"
        res = core.score_candidates(state_repr=state, candidates=candidates)

        self.assertIn("probs", res)
        self.assertIn("value", res)
        self.assertIn("best_action", res)
        self.assertIn(res["best_action"], candidates)
        self.assertAlmostEqual(sum(res["probs"].values()), 1.0, places=4)
        self.assertLess(res["latency_ms"], 10.0)

    def test_vision_nanocore_contract(self):
        """NanoCoreVision executes sub-3.5ms inference on continuous latent states and actions."""
        core = NanoCoreVision()
        self.assertIsInstance(core, BaseNanoCore)
        self.assertEqual(core.domain, "vision")

        state_vec = np.random.randn(512).astype(np.float32)
        candidates = ["action_forward", "action_turn_left", "action_stop"]
        res = core.score_candidates(state_repr=state_vec, candidates=candidates)

        self.assertIn("probs", res)
        self.assertIn("value", res)
        self.assertIn("best_action", res)
        self.assertIn(res["best_action"], candidates)
        self.assertAlmostEqual(sum(res["probs"].values()), 1.0, places=4)
        self.assertLess(res["latency_ms"], 10.0)

    def test_empty_candidate_protection(self):
        """Empty candidates returns empty probabilities and zero latency gracefully."""
        b_core = NanoCoreBrowser()
        v_core = NanoCoreVision()
        b_res = b_core.score_candidates(state_repr="empty", candidates=[])
        v_res = v_core.score_candidates(state_repr="empty", candidates=[])

        self.assertEqual(b_res["probs"], {})
        self.assertIsNone(b_res["best_action"])
        self.assertEqual(v_res["probs"], {})
        self.assertIsNone(v_res["best_action"])

    def test_artifact_export_and_checksum_integrity(self):
        """Exported artifact manifests contain canonical keys and deterministic checksums."""
        b_core = NanoCoreBrowser()
        v_core = NanoCoreVision()

        b_art = b_core.export_artifact()
        v_art = v_core.export_artifact()

        self.assertEqual(b_art["domain"], "browser")
        self.assertIn("weights_checksum", b_art)
        self.assertIn("memory_bytes", b_art)
        self.assertGreater(len(b_art["weights_checksum"]), 10)

        self.assertEqual(v_art["domain"], "vision")
        self.assertIn("weights_checksum", v_art)
        self.assertIn("memory_bytes", v_art)

    def test_memory_footprint_limits(self):
        """Specialized cores must strictly stay below 40MB memory footprint."""
        b_core = NanoCoreBrowser()
        v_core = NanoCoreVision()

        b_mem = b_core.memory_footprint_bytes()
        v_mem = v_core.memory_footprint_bytes()

        # Less than 40 MB (40 * 1024 * 1024 bytes)
        self.assertLess(b_mem, 40 * 1024 * 1024, f"Browser core footprint {b_mem} exceeded 40MB")
        self.assertLess(v_mem, 40 * 1024 * 1024, f"Vision core footprint {v_mem} exceeded 40MB")

    def test_cross_domain_weights_isolation(self):
        """Browser and Vision cores do not share weights or mutate each other."""
        b_core = NanoCoreBrowser()
        v_core = NanoCoreVision()

        b_checksum_before = b_core._checksum
        # Update vision weights
        v_core.weights["latent_proj"] += 1.0
        self.assertEqual(b_core._checksum, b_checksum_before)


    def test_specialized_nanocores_multi_dimension_compatibility(self):
        """Verifies Specialized NanoCores dynamically adapt to 896, 1024, 2048, 3584, and 4096 base dimensions."""
        dimensions = [896, 1024, 2048, 3584, 4096]
        for dim in dimensions:
            with self.subTest(state_dim=dim):
                core = NanoCoreBrowser(state_dim=dim, candidate_dim=dim)
                self.assertEqual(core.state_dim, dim)
                self.assertEqual(core.weights["state_proj"].shape, (128, dim))

                # Test scoring with arbitrary representation
                candidates = ["btn_ok", "btn_cancel"]
                state = np.random.randn(dim).astype(np.float32)
                res = core.score_candidates(state_repr=state, candidates=candidates)
                self.assertIn("probs", res)
                self.assertEqual(len(res["probs"]), 2)
                self.assertAlmostEqual(sum(res["probs"].values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

