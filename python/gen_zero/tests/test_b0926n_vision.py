"""Regression tests for the b0926n vision fail-closed contracts."""

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal installs
    torch = None

from gen_zero.vision.discrete_bins_scorer import DiscreteBinsExpectationScorer
from gen_zero.vision.engine import (
    MultimodalVisionEngine,
    PyTorchVisualDecisionEngine,
)


class _VisionModel:
    def __init__(self, logits):
        self.logits = logits
        self.encode_calls = 0

    def encode_image(self, image_data, image_shape):
        self.encode_calls += 1
        return np.ones((1, 2), dtype=np.float32), {}

    def score_labels(self, prefix_entry, question_suffix, labels):
        return self.logits


class TestB0926NVision(unittest.TestCase):
    def test_pytorch_engine_has_no_unloaded_mock_paths(self):
        engine = PyTorchVisualDecisionEngine(device="cpu")
        expected = "Vision model weights are not loaded. Provide a valid checkpoint."

        self.assertFalse(hasattr(engine, "_mock_prefill"))
        self.assertFalse(hasattr(engine, "_mock_score_candidates"))
        with self.assertRaisesRegex(RuntimeError, expected):
            engine.prefill_visual_context("frame.png")
        with self.assertRaisesRegex(RuntimeError, expected):
            engine.score_candidates_direct({}, ["LEFT"])

    @unittest.skipUnless(torch is not None, "PyTorch is required for direct-logit validation")
    def test_nonfinite_pytorch_direct_logits_are_rejected(self):
        class _Tokenizer:
            def encode(self, text, add_special_tokens=False, return_tensors=None):
                return torch.tensor([[0]], dtype=torch.long)

        class _Model:
            def __init__(self, logit):
                self.logit = logit

            def __call__(self, input_ids, past_key_values=None, use_cache=False):
                return type("_Output", (), {
                    "logits": torch.tensor([[[self.logit]]], dtype=torch.float32),
                })()

        engine = PyTorchVisualDecisionEngine(device="cpu")
        engine._is_loaded = True
        engine.model = _Model(2.0)
        engine.tokenizer = _Tokenizer()
        finite = engine.score_candidates_direct({"past_key_values": object()}, ["LEFT"])
        self.assertEqual(finite["best_action"], "LEFT")
        self.assertEqual(finite["probs"], {"LEFT": 1.0})

        for logit in (np.nan, np.inf, -np.inf):
            with self.subTest(logit=logit):
                engine.model = _Model(logit)
                with self.assertRaisesRegex(ValueError, "finite"):
                    engine.score_candidates_direct({"past_key_values": object()}, ["LEFT"])

    def test_prefill_does_not_use_stale_cache_without_model(self):
        image = b"frame-001"
        model = _VisionModel([1.0, 0.0])
        engine = MultimodalVisionEngine(embed_dim=2, vision_model=model)
        entry, _ = engine.prefill_image(image)
        self.assertEqual(model.encode_calls, 1)

        # Removing the model must invalidate the cache path as well.
        engine.vision_model = None
        with self.assertRaisesRegex(
            RuntimeError,
            "Vision model weights are not loaded\\. Provide a valid checkpoint\\.",
        ):
            engine.prefill_image(image)
        self.assertEqual(entry.hit_count, 0)

    def test_nonfinite_multimodal_logits_are_rejected(self):
        image = b"frame-002"
        for logits in ([np.nan, 0.0], [np.inf, 0.0], [-np.inf, 0.0]):
            with self.subTest(logits=logits):
                model = _VisionModel(logits)
                engine = MultimodalVisionEngine(embed_dim=2, vision_model=model)
                entry, _ = engine.prefill_image(image)
                with self.assertRaisesRegex(ValueError, "finite"):
                    engine.score_question_on_prefix(entry, "Which?", ["A", "B"])

    def test_empty_bins_are_not_confident(self):
        verdict = DiscreteBinsExpectationScorer().compute_expectation_and_variance({})

        self.assertEqual(verdict.confidence, 0.0)
        self.assertFalse(verdict.is_confident)
        self.assertEqual(verdict.expected_bin, 0.0)
        self.assertTrue(all(value == 0.0 for value in verdict.bin_probabilities.values()))


class TestVisionAuditEdgeCases(unittest.TestCase):
    def test_bins_reject_nonfinite_mass_and_targets(self):
        scorer = DiscreteBinsExpectationScorer()
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    scorer.compute_expectation_and_variance({"1": value, "2": 1.0})
                with self.assertRaisesRegex(ValueError, "finite"):
                    scorer.continuous_to_target_bin(value)

    def test_bins_large_weights_remain_normalized(self):
        verdict = DiscreteBinsExpectationScorer().compute_expectation_and_variance(
            {"1": 1e308, "9": 1e308}
        )
        self.assertAlmostEqual(sum(verdict.bin_probabilities.values()), 1.0)
        self.assertEqual(verdict.expected_bin, 5.0)
        self.assertEqual(verdict.uncertainty_variance, 16.0)
        self.assertFalse(verdict.is_confident)

    def test_bins_configuration_and_single_bin(self):
        for bins in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                DiscreteBinsExpectationScorer(num_bins=bins)
        for threshold in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                DiscreteBinsExpectationScorer(max_variance_threshold=threshold)
        verdict = DiscreteBinsExpectationScorer(num_bins=1).compute_expectation_and_variance({"1": 2})
        self.assertEqual(verdict.expected_bin, 1)
        self.assertEqual(verdict.normalized_score, 0.5)
        self.assertEqual(verdict.confidence, 1)

    def test_extreme_finite_logits_remain_valid(self):
        model = _VisionModel([1e308, -1e308])
        engine = MultimodalVisionEngine(embed_dim=2, vision_model=model)
        entry, _ = engine.prefill_image(b"extreme")
        with np.errstate(all="raise"):
            best, probs, confidence, _ = engine.score_question_on_prefix(entry, "?", ["A", "B"])
        self.assertEqual(best, "A")
        self.assertAlmostEqual(probs["A"], 1.0)
        self.assertEqual(probs["B"], 0)
        self.assertTrue(np.isfinite(confidence))

    @unittest.skipUnless(torch is not None, "PyTorch required")
    def test_prefill_matches_model_dtype_preserving_token_ids(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))

            def forward(self, input_ids, pixel_values, **kwargs):
                assert input_ids.dtype == torch.long
                assert pixel_values.dtype == self.weight.dtype
                assert pixel_values.device == self.weight.device
                return type("Output", (), {"past_key_values": object(), "hidden_states": [pixel_values]})()

        engine = PyTorchVisualDecisionEngine(device="cpu")
        engine.model = Model()
        engine.device = "meta"  # Stale configuration must not override loaded parameters.
        engine._is_loaded = True
        engine.processor = lambda **kwargs: {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "pixel_values": torch.ones(1, 1, 1, dtype=torch.float32),
        }
        bundle = engine.prefill_visual_context(b"image")
        self.assertEqual(bundle["last_hidden_state"].dtype, torch.float64)

    @unittest.skipUnless(torch is not None, "PyTorch required")
    def test_invalid_temperature_and_empty_candidate_tokens(self):
        engine = PyTorchVisualDecisionEngine(device="cpu")
        engine._is_loaded = True
        engine.model = object()
        for temperature in (0, -1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "Temperature"):
                engine.score_candidates_direct({"past_key_values": object()}, ["A"], temperature)
        engine.tokenizer = type("Tokenizer", (), {
            "encode": lambda *args, **kwargs: torch.empty((1, 0), dtype=torch.long)
        })()
        with self.assertRaisesRegex(ValueError, "at least one token"):
            engine.score_candidates_direct({"past_key_values": object()}, [""])


if __name__ == "__main__":
    unittest.main()
