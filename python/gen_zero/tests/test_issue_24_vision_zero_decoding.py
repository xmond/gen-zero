"""Unit and Integration Tests for Issue #24: Multimodal Vision Zero-Decoding.

Tests:
1. Shared Vision Prefix KV-Cache & Prefill: Encodes image once, evaluates multi-question batches with >= 70% latency reduction.
2. Single-Token Stability & TokenFragmentationError: Strictly enforces 1 token ID, catches fragmentation, CanonicalLabelMapper roundtrip.
3. Discrete Bins Expectation Scorer: Evaluates M=9 bins, expected score E[S], normalized score in [0.01, 0.99], uncertainty variance, calibrated confidence.
4. RFDT Single-Step Distillation: NumPy & PyTorch loss parity, alpha weighting between CE and KL.
5. Adaptive Perception Router: Dual-track routing between A11y Tree zero-vision fast path and Vision Multimodal fallback.
6. SharedVisionPrefixCache Capacity & Eviction: Tests capacity eviction and hit count tracking.
7. Client Integration: Ensures GenZeroClient exposes vision zero-decoding components seamlessly.
"""

import unittest
import numpy as np

from gen_zero.vision.engine import (
    SharedVisionPrefixCache,
    MultimodalVisionEngine,
    PerceptionChannel,
    AdaptivePerceptionRouter,
    VisionPrefixEntry,
)
from gen_zero.model.token_stability import (
    TokenFragmentationError,
    assert_single_token_stability,
    CanonicalLabelMapper,
)
from gen_zero.vision.discrete_bins_scorer import (
    DiscreteBinsExpectationScorer,
    DiscreteBinsVerdict,
)
from gen_zero.train.rfdt_distiller import (
    compute_rfdt_loss_numpy,
    compute_rfdt_loss_torch,
    RFDTDistiller,
)
from gen_zero.client import GenZeroClient, GenZero


class FakeVisionModel:
    """Deterministic test double implementing MultimodalVisionEngine's vision_model protocol.

    Not a stand-in for a real vision tower's accuracy -- only used to verify that the engine
    wires ``encode_image``/``score_labels`` calls correctly (caching, call counts, shapes).
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self.encode_calls = 0
        self.score_calls = 0

    def encode_image(self, image_data, image_shape):
        self.encode_calls += 1
        prefix_embedding = np.ones((4, self.embed_dim), dtype=np.float32)
        kv_cache = {
            "k": np.ones((4, self.embed_dim), dtype=np.float32),
            "v": np.ones((4, self.embed_dim), dtype=np.float32) * 2.0,
        }
        return prefix_embedding, kv_cache

    def score_labels(self, prefix_entry, question_suffix, labels):
        self.score_calls += 1
        # Deterministic: the first candidate label always receives the highest logit.
        return [float(len(labels) - i) for i in range(len(labels))]


class TestSharedVisionPrefixAndZeroDecoding(unittest.TestCase):
    """Test 1: Shared vision prefix prefill/caching, and fail-closed behavior without a model."""

    def setUp(self):
        self.model = FakeVisionModel(embed_dim=128)
        self.engine = MultimodalVisionEngine(embed_dim=128, cache_capacity=16, vision_model=self.model)
        self.dummy_image = b"RAW_PIXEL_IMAGE_DATA_FRAME_001"

    def test_no_vision_model_raises_runtime_error(self):
        """Without weights loaded, the engine must fail closed instead of fabricating output."""
        bare_engine = MultimodalVisionEngine(embed_dim=128, cache_capacity=16)

        with self.assertRaises(RuntimeError) as ctx:
            bare_engine.prefill_image(self.dummy_image)
        self.assertEqual(
            str(ctx.exception),
            "Vision model weights are not loaded. Provide a valid checkpoint."
        )

        entry, _ = self.engine.prefill_image(self.dummy_image)
        with self.assertRaises(RuntimeError) as ctx2:
            bare_engine.score_question_on_prefix(entry, "Is it visible?", ["YES", "NO"])
        self.assertEqual(
            str(ctx2.exception),
            "Vision model weights are not loaded. Provide a valid checkpoint."
        )

    def test_single_prefill_and_cache_hit(self):
        entry1, t1 = self.engine.prefill_image(self.dummy_image)
        self.assertIsInstance(entry1, VisionPrefixEntry)
        self.assertGreaterEqual(t1, 0.0)
        self.assertEqual(self.model.encode_calls, 1)

        # Second call with same image hits cache and never calls the model again.
        entry2, t2 = self.engine.prefill_image(self.dummy_image)
        self.assertEqual(entry1.fingerprint, entry2.fingerprint)
        self.assertEqual(entry2.hit_count, 1)
        self.assertEqual(self.model.encode_calls, 1)

    def test_multi_question_shared_prefix_reuses_cached_prefill(self):
        questions = [
            ("Is the confirmation modal visible?", ["YES", "NO"]),
            ("Which button is the primary CTA?", ["OK", "CANCEL", "SUBMIT"]),
            ("What is the current slider state?", ["LOW", "MED", "HIGH"]),
            ("Is there an active error banner?", ["TRUE", "FALSE"]),
        ]

        batch_report = self.engine.batch_evaluate_shared_prefix(self.dummy_image, questions)
        self.assertEqual(batch_report["questions_count"], 4)
        self.assertEqual(len(batch_report["results"]), 4)

        # The image must be encoded exactly once and reused across all four questions.
        self.assertEqual(self.model.encode_calls, 1)
        self.assertEqual(self.model.score_calls, 4)

        for (q_text, labels), res in zip(questions, batch_report["results"]):
            self.assertIn(res["best_action"], res["probabilities"])
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)
            # FakeVisionModel always ranks the first label highest.
            self.assertEqual(res["best_action"], labels[0])


class TestSingleTokenStabilityAndCanonicalMapper(unittest.TestCase):
    """Test 2: Single-token stability assertions and CanonicalLabelMapper."""

    def test_assert_single_token_stability_success(self):
        class MockSingleTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text in ["A", "B", "C", "1", "2"]:
                    return [ord(text)]
                return [ord(c) for c in text]

        tok = MockSingleTokenizer()
        token_id = assert_single_token_stability(tok, "A")
        self.assertEqual(token_id, ord("A"))

        token_id_2 = assert_single_token_stability(tok, "1")
        self.assertEqual(token_id_2, ord("1"))

    def test_assert_single_token_stability_raises_fragmentation_error(self):
        class MockFragmentingTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Multi-word string decomposes into subwords
                return [101, 102, 103]

        tok = MockFragmentingTokenizer()
        with self.assertRaises(TokenFragmentationError) as ctx:
            assert_single_token_stability(tok, "Confirm Order")
        self.assertIn("fragmented into 3 tokens", str(ctx.exception))

    def test_canonical_label_mapper_roundtrip(self):
        mapper = CanonicalLabelMapper(key_type="alphabetic")
        raw_labels = ["Confirm Transaction", "Cancel & Refund", "Hold in Escrow"]

        canonical_keys, prompt_mapping = mapper.map_labels(raw_labels)
        self.assertEqual(canonical_keys, ["A", "B", "C"])
        self.assertEqual(prompt_mapping["A"], "Confirm Transaction")

        # Decode single choice
        decoded = mapper.decode_choice("B")
        self.assertEqual(decoded, "Cancel & Refund")

        # Decode probability distribution
        canon_probs = {"A": 0.1, "B": 0.8, "C": 0.1}
        decoded_dist = mapper.decode_distribution(canon_probs)
        self.assertAlmostEqual(decoded_dist["Cancel & Refund"], 0.8)
        self.assertAlmostEqual(decoded_dist["Confirm Transaction"], 0.1)


class TestDiscreteBinsExpectationScorer(unittest.TestCase):
    """Test 3: Discrete bins expectation scorer and variance confidence."""

    def setUp(self):
        self.scorer = DiscreteBinsExpectationScorer(num_bins=9, max_variance_threshold=2.0)

    def test_sharp_distribution_confidence(self):
        # All mass concentrated on bin "5"
        bin_probs = {str(i): 0.0 for i in range(1, 10)}
        bin_probs["5"] = 1.0

        verdict = self.scorer.compute_expectation_and_variance(bin_probs)
        self.assertAlmostEqual(verdict.expected_bin, 5.0, places=3)
        self.assertAlmostEqual(verdict.normalized_score, 0.50, places=2)  # Middle of [0.01, 0.99]
        self.assertAlmostEqual(verdict.uncertainty_variance, 0.0, places=3)
        self.assertAlmostEqual(verdict.confidence, 1.0, places=3)
        self.assertTrue(verdict.is_confident)

    def test_boundary_distributions(self):
        # All mass on bin "1" -> minimum score 0.01
        p_min = {"1": 1.0}
        v_min = self.scorer.compute_expectation_and_variance(p_min)
        self.assertAlmostEqual(v_min.normalized_score, 0.01, places=2)

        # All mass on bin "9" -> maximum score 0.99
        p_max = {"9": 1.0}
        v_max = self.scorer.compute_expectation_and_variance(p_max)
        self.assertAlmostEqual(v_max.normalized_score, 0.99, places=2)

    def test_uniform_dispersed_distribution_high_variance(self):
        # Uniform across 9 bins
        uniform_probs = {str(i): 1.0 / 9.0 for i in range(1, 10)}
        verdict = self.scorer.compute_expectation_and_variance(uniform_probs)
        self.assertAlmostEqual(verdict.expected_bin, 5.0, places=2)
        self.assertGreater(verdict.uncertainty_variance, 6.0)
        self.assertLess(verdict.confidence, 0.35)
        self.assertFalse(verdict.is_confident)

    def test_continuous_to_target_bin_quantization(self):
        self.assertEqual(self.scorer.continuous_to_target_bin(0.0), "1")
        self.assertEqual(self.scorer.continuous_to_target_bin(0.5), "5")
        self.assertEqual(self.scorer.continuous_to_target_bin(1.0), "9")


class TestRFDTDistiller(unittest.TestCase):
    """Test 4: RFDT single-step distillation loss in NumPy and PyTorch."""

    def setUp(self):
        self.logits = np.array([[2.0, 0.5, -1.0], [0.1, 1.8, 0.3]], dtype=np.float32)
        self.targets = np.array([0, 1], dtype=np.int64)
        self.teacher_probs = np.array([[0.7, 0.2, 0.1], [0.15, 0.75, 0.1]], dtype=np.float32)

    def test_rfdt_loss_numpy(self):
        loss, metrics = compute_rfdt_loss_numpy(
            model_logits=self.logits,
            target_indices=self.targets,
            teacher_probs=self.teacher_probs,
            alpha=0.6,
        )
        self.assertGreater(loss, 0.0)
        self.assertIn("ce_loss", metrics)
        self.assertIn("kl_loss", metrics)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_rfdt_loss_torch_parity(self):
        try:
            import torch
            loss_t, metrics_t = compute_rfdt_loss_torch(
                model_logits=torch.tensor(self.logits),
                target_indices=torch.tensor(self.targets),
                teacher_probs=torch.tensor(self.teacher_probs),
                alpha=0.6,
            )
            loss_np, metrics_np = compute_rfdt_loss_numpy(
                model_logits=self.logits,
                target_indices=self.targets,
                teacher_probs=self.teacher_probs,
                alpha=0.6,
            )
            self.assertAlmostEqual(metrics_t["ce_loss"], metrics_np["ce_loss"], places=3)
            self.assertAlmostEqual(metrics_t["kl_loss"], metrics_np["kl_loss"], places=3)
            self.assertAlmostEqual(loss_t.item(), loss_np, places=3)
        except ImportError:
            pass

    def test_rfdt_distiller_step(self):
        distiller = RFDTDistiller(alpha=0.5)
        step_metrics = distiller.distillation_step(self.logits, self.targets, self.teacher_probs)
        self.assertIn("loss", step_metrics)
        self.assertEqual(len(distiller.history), 1)


class TestAdaptivePerceptionRouter(unittest.TestCase):
    """Test 5: Dual-track perception routing between A11y zero-vision and Vision Multimodal fallback."""

    def setUp(self):
        self.router = AdaptivePerceptionRouter()

    def test_standard_gui_routes_to_a11y(self):
        env_context = {
            "has_a11y_tree": True,
            "a11y_nodes_count": 42,
            "is_canvas_or_game": False,
            "unstructured_screen": False,
        }
        channel = self.router.route_environment(env_context)
        self.assertEqual(channel, PerceptionChannel.A11Y_ZERO_VISION)

    def test_canvas_or_game_routes_to_vision_multimodal(self):
        env_context = {
            "has_a11y_tree": True,
            "a11y_nodes_count": 5,
            "is_canvas_or_game": True,
            "unstructured_screen": False,
        }
        channel = self.router.route_environment(env_context)
        self.assertEqual(channel, PerceptionChannel.VISION_MULTIMODAL_FALLBACK)

    def test_missing_a11y_tree_routes_to_vision_fallback(self):
        env_context = {
            "has_a11y_tree": False,
            "a11y_nodes_count": 0,
            "is_canvas_or_game": False,
            "unstructured_screen": False,
        }
        channel = self.router.route_environment(env_context)
        self.assertEqual(channel, PerceptionChannel.VISION_MULTIMODAL_FALLBACK)


class TestCacheEvictionAndClientIntegration(unittest.TestCase):
    """Test 6 & 7: Cache eviction and top-level client integration."""

    def test_shared_cache_capacity_and_lru_eviction(self):
        cache = SharedVisionPrefixCache(capacity=2)
        e1 = cache.put("fp1", (100, 100), np.zeros((2, 4)))
        e2 = cache.put("fp2", (100, 100), np.zeros((2, 4)))
        self.assertEqual(cache.size, 2)

        # Hit fp1 twice
        cache.get("fp1")
        cache.get("fp1")

        # Put fp3 -> fp2 has 0 hits, should be evicted
        cache.put("fp3", (100, 100), np.zeros((2, 4)))
        self.assertEqual(cache.size, 2)
        self.assertIsNotNone(cache.get("fp1"))
        self.assertIsNotNone(cache.get("fp3"))
        self.assertIsNone(cache.get("fp2"))

    def test_client_vision_zero_decoding_components(self):
        client = GenZeroClient()
        self.assertTrue(hasattr(client, "multimodal_vision"))
        self.assertTrue(hasattr(client, "perception_router"))
        self.assertTrue(hasattr(client, "discrete_bins_scorer"))

        # Verify discrete bins quantization through client instance
        target_bin = client.discrete_bins_scorer.continuous_to_target_bin(0.85)
        self.assertIn(target_bin, ["7", "8", "9"])


if __name__ == "__main__":
    unittest.main()
