"""Unit and Contract Tests for Mixture of Vectors (MoV) Dynamic Modality Routing."""

import unittest
import numpy as np

from gen_zero.gateway.modality_router import (
    MoVVectorRouter,
    RouteDecision,
    ModalityType,
    AdaptiveModalityRouter
)
from gen_zero.train.replay_buffer import (
    DomainExperienceBuffer,
    BrowserExperienceBuffer,
    VisionExperienceBuffer,
    IngestionReceipt,
    SampledBatch
)


class TestMoVVectorRouting(unittest.TestCase):
    """Verifies geometric MoV vector routing guarantees and edge cases."""

    def setUp(self):
        # Construct orthogonal 4-dim prototypes for deterministic testing
        self.prototypes = {
            "browser": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "vision": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        }
        self.router = MoVVectorRouter(
            prototypes=self.prototypes,
            temperature=0.10,
            confidence_threshold=0.60,
            min_cosine_threshold=0.10,
            margin_threshold=0.05,
            fallback_domain="fallback"
        )

    def test_exact_prototype_routing(self):
        """State aligned with browser prototype routes to 'browser' with high confidence."""
        vec = [1.0, 0.0, 0.0, 0.0]
        decision = self.router.route(vec)
        self.assertEqual(decision.selected_domain, "browser")
        self.assertEqual(decision.status, "ROUTED")
        self.assertGreaterEqual(decision.confidence, 0.90)
        self.assertAlmostEqual(decision.cosine_scores["browser"], 1.0, places=4)
        self.assertAlmostEqual(decision.cosine_scores["vision"], 0.0, places=4)
        self.assertGreater(decision.margin, 0.9)

    def test_vision_prototype_routing(self):
        """State aligned with vision prototype routes to 'vision'."""
        vec = [0.0, 2.5, 0.0, 0.0]  # Non-unit length must be normalized correctly
        decision = self.router.route(vec)
        self.assertEqual(decision.selected_domain, "vision")
        self.assertEqual(decision.status, "ROUTED")
        self.assertGreaterEqual(decision.confidence, 0.90)

    def test_positive_scale_invariance(self):
        """Multiplying input vector by positive scalar produces identical routing probabilities."""
        v1 = [0.2, 0.8, 0.0, 0.0]
        v2 = [2.0, 8.0, 0.0, 0.0]
        d1 = self.router.route(v1)
        d2 = self.router.route(v2)
        self.assertEqual(d1.selected_domain, d2.selected_domain)
        self.assertAlmostEqual(d1.confidence, d2.confidence, places=5)
        self.assertAlmostEqual(d1.margin, d2.margin, places=5)

    def test_uncertainty_threshold_trigger(self):
        """Equal 45-degree vector between browser and vision triggers UNCERTAIN."""
        vec = [1.0, 1.0, 0.0, 0.0]  # Ties between browser and vision
        decision = self.router.route(vec)
        self.assertEqual(decision.selected_domain, "fallback")
        self.assertEqual(decision.status, "UNCERTAIN")
        self.assertAlmostEqual(decision.margin, 0.0, places=4)

    def test_ood_rejection(self):
        """Orthogonal vector completely outside prototype subspace triggers OOD status."""
        vec = [0.0, 0.0, 1.0, 0.0]  # Orthogonal to both browser and vision
        decision = self.router.route(vec)
        self.assertEqual(decision.selected_domain, "fallback")
        self.assertEqual(decision.status, "OOD")

    def test_invalid_input_rejection(self):
        """NaN, Inf, near-zero norm, and dimension mismatch are rejected safely."""
        # NaN
        d_nan = self.router.route([float("nan"), 0.0, 0.0, 0.0])
        self.assertEqual(d_nan.status, "REJECTED")

        # Inf
        d_inf = self.router.route([float("inf"), 0.0, 0.0, 0.0])
        self.assertEqual(d_inf.status, "REJECTED")

        # Zero norm
        d_zero = self.router.route([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(d_zero.status, "REJECTED")

        # Dimension mismatch
        d_dim = self.router.route([1.0, 0.0])
        self.assertEqual(d_dim.status, "REJECTED")

    def test_projection_matrix(self):
        """Linear projection W maps 6-dim inputs to 4-dim prototype space."""
        # W: 6 -> 4 mapping first 4 dims
        W = np.zeros((6, 4), dtype=np.float32)
        for i in range(4):
            W[i, i] = 1.0

        proj_router = MoVVectorRouter(
            prototypes=self.prototypes,
            projection_matrix=W,
            temperature=0.15
        )
        vec_6d = [1.0, 0.0, 0.0, 0.0, 0.99, 0.88]
        decision = proj_router.route(vec_6d)
        self.assertEqual(decision.selected_domain, "browser")
        self.assertEqual(decision.status, "ROUTED")


class TestDomainExperienceBuffers(unittest.TestCase):
    """Verifies domain experience buffers, physical ownership, receipts, and 1:3 ratio math."""

    def test_browser_buffer_domain_validation(self):
        """BrowserExperienceBuffer accepts browser records and rejects foreign domains."""
        buf = BrowserExperienceBuffer(capacity=100)
        rec = {"state": "click button", "candidate_ids": ["btn_a", "btn_b"]}
        receipt = buf.append([rec], partition="gold")
        self.assertIsInstance(receipt, IngestionReceipt)
        self.assertEqual(receipt.domain, "browser")
        self.assertEqual(receipt.count, 1)
        self.assertEqual(len(buf), 1)

        # Rejection of foreign domain
        foreign_rec = {"state": "click", "domain": "vision"}
        with self.assertRaises(ValueError):
            buf.append([foreign_rec], partition="gold")

    def test_receipt_based_rollback(self):
        """Receipt-based rollback purges exact ingested samples under concurrent additions."""
        buf = BrowserExperienceBuffer(capacity=100)
        batch_1 = [{"state": f"sample_{i}", "candidate_ids": ["a"]} for i in range(5)]
        batch_2 = [{"state": f"sample_{i+5}", "candidate_ids": ["a"]} for i in range(3)]

        r1 = buf.append(batch_1, partition="hard")
        r2 = buf.append(batch_2, partition="hard")
        self.assertEqual(len(buf), 8)

        # Rollback batch 1 without affecting batch 2
        removed = buf.rollback_ingestion(r1)
        self.assertEqual(removed, 5)
        self.assertEqual(len(buf), 3)
        remaining_ids = [s["id"] for s in buf.hard_buffer]
        self.assertEqual(remaining_ids, r2.sample_ids)

    def test_strict_one_to_three_ratio_sampling(self):
        """Strict sampling math: m = min(B/4, |H_D|, floor(|G_D|/3)), n_H = m, n_G = 3m."""
        buf = BrowserExperienceBuffer(capacity=100)
        # Add 5 hard, 9 gold
        buf.append([{"state": f"h_{i}", "candidate_ids": ["a"]} for i in range(5)], partition="hard")
        buf.append([{"state": f"g_{i}", "candidate_ids": ["a"]} for i in range(9)], partition="gold")

        # Request B=16 -> B/4 = 4. len(H)=5, len(G)/3 = 3. So m = min(4, 5, 3) = 3.
        # Should return 4 * 3 = 12 samples (3 hard, 9 gold)
        batch_res = buf.sample_training_batch(batch_size=16, strict_ratio=True)
        self.assertTrue(batch_res.ready)
        self.assertEqual(batch_res.m, 3)
        self.assertEqual(batch_res.n_hard, 3)
        self.assertEqual(batch_res.n_gold, 9)
        self.assertEqual(len(batch_res), 12)

        # If zero hard samples, m=0 -> ready=False
        buf_empty_h = BrowserExperienceBuffer(capacity=100)
        buf_empty_h.append([{"state": "g_0", "candidate_ids": ["a"]}], partition="gold")
        res_not_ready = buf_empty_h.sample_training_batch(batch_size=4, strict_ratio=True)
        self.assertFalse(res_not_ready.ready)
        self.assertEqual(len(res_not_ready), 0)


if __name__ == "__main__":
    unittest.main()
