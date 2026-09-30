"""Unit and Contract Tests for Mixture of Vectors (MoV) Dynamic Modality Routing."""

import unittest
import numpy as np

from gen_zero.gateway.modality_router import (
    MoVVectorRouter,
    RouteDecision,
    ModalityType,
    AdaptiveModalityRouter
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




if __name__ == "__main__":
    unittest.main()
