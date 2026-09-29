"""Unit tests for Gen-Zero Gateway Layer (Adaptive Modality Router)."""

import unittest
from gen_zero.gateway.modality_router import (
    AdaptiveModalityRouter,
    ModalityType,
    IngestedState,
)
from gen_zero.adaptive_engine import AdaptiveParameterEngine
from gen_zero.client import GenZero


class TestAdaptiveModalityGateway(unittest.TestCase):
    def setUp(self):
        self.router = AdaptiveModalityRouter(feature_dim=128)

    def test_pure_numerical_classification(self):
        # Case 1: Dictionary of floats (e.g. TSLA quant bar)
        tsla_state = {
            "price": 242.50,
            "ret_5m": 0.0082,
            "ret_15m": -0.0014,
            "volatility": 0.0145,
            "position": 1,
            "unrealized_pnl": 0.021,
            "drawdown": 0.003
        }
        mod = self.router.classify_modality(tsla_state)
        self.assertEqual(mod, ModalityType.PURE_NUMERICAL)

        # Ingestion should NOT invoke LLM
        ingested = self.router.ingest(tsla_state)
        self.assertFalse(ingested.llm_invoked)
        self.assertEqual(ingested.modality, ModalityType.PURE_NUMERICAL)
        self.assertIn("bypass_llm", ingested.metadata["reason"])

    def test_unstructured_text_classification(self):
        # Case 2: Natural language prompt string
        prompt = "Turn off living room air conditioner and lock the front door."
        mod = self.router.classify_modality(prompt)
        self.assertEqual(mod, ModalityType.UNSTRUCTURED_TEXT)

        # No llm_extractor is mounted: the hash projection is not an LLM call, so llm_invoked is False
        ingested = self.router.ingest(prompt)
        self.assertFalse(ingested.llm_invoked)
        self.assertEqual(ingested.metadata["extractor"], "cpu_deterministic_projection")
        self.assertEqual(ingested.modality, ModalityType.UNSTRUCTURED_TEXT)
        self.assertIsNotNone(ingested.dense_repr)
        self.assertTrue(ingested.normalized_state.get("is_semantic"))

    def test_multimodal_hybrid_classification(self):
        # Case 3: Mixed text instruction with numerical telemetry
        hybrid_input = {
            "instruction": "Execute dynamic hedging if drawdown exceeds threshold",
            "current_drawdown": 0.035,
            "portfolio_value": 150000.0,
            "delta_exposure": 0.42
        }
        mod = self.router.classify_modality(hybrid_input)
        self.assertEqual(mod, ModalityType.MULTIMODAL_HYBRID)

        ingested = self.router.ingest(hybrid_input)
        self.assertFalse(ingested.llm_invoked)  # no real extractor mounted
        self.assertEqual(ingested.modality, ModalityType.MULTIMODAL_HYBRID)
        self.assertTrue(ingested.normalized_state.get("is_hybrid"))
        self.assertIn("text_embedding", ingested.normalized_state)
        self.assertEqual(ingested.normalized_state["current_drawdown"], 0.035)

    def test_adaptive_parameter_formula17(self):
        # Test Formula 17 in AdaptiveParameterEngine
        m1, invoke1, r1 = AdaptiveParameterEngine.dynamic_modality_decision({"price": 250.0, "pnl": 0.05})
        self.assertEqual(m1, "pure_numerical")
        self.assertFalse(invoke1)

        m2, invoke2, r2 = AdaptiveParameterEngine.dynamic_modality_decision("Summarize user intent")
        self.assertEqual(m2, "unstructured_text")
        self.assertTrue(invoke2)

    def test_client_end_to_end_auto_routing(self):
        # Test that GenZero client seamlessly passes modality info and makes decisions
        client = GenZero()

        # Pure quant decision
        quant_state = {"price": 100.0, "momentum": 0.05}
        quant_res = client.decide(
            state=quant_state,
            candidates=["BUY", "HOLD", "SELL"],
            mode="auto"
        )
        self.assertIn("modality", quant_res)
        self.assertEqual(quant_res["modality"]["modality"], "pure_numerical")
        self.assertFalse(quant_res["modality"]["llm_invoked"])

        # Natural language text decision
        text_state = "Please confirm the transaction order for customer A"
        text_res = client.decide(
            state=text_state,
            candidates=["APPROVE", "REJECT", "ESCALATE"],
            mode="auto"
        )
        self.assertIn("modality", text_res)
        self.assertEqual(text_res["modality"]["modality"], "unstructured_text")
        # GenZeroDualHeadModel has no extract_hidden_state, so no LLM extractor is mounted:
        # the hash projection must not be reported as an LLM call
        self.assertFalse(text_res["modality"]["llm_invoked"])


class _LoadedVision:
    is_loaded = True

    def __call__(self, img, prompt=""):
        return [0.5] * 16


class TestVisionExtractorReadiness(unittest.TestCase):
    """llm_invoked must reflect a loaded vision model, not merely a mounted object."""

    MIXED = {"frame": "frame_001.png", "prompt": "which region has the ball?"}

    def test_placeholder_callable_is_not_llm_invoked(self):
        calls = []
        router = AdaptiveModalityRouter(
            vision_extractor=lambda img, prompt="": calls.append(img) or [1.0], feature_dim=16
        )
        for raw, modality in ((self.MIXED, "multimodal_vision_text"), ("frame_001.png", "vision_image")):
            with self.assertLogs("gen_zero.gateway.modality_router", level="WARNING"):
                ing = router.ingest(raw)
            self.assertEqual(ing.modality.value, modality)
            self.assertFalse(ing.llm_invoked)
            self.assertTrue(ing.metadata["degraded"])
            self.assertEqual(ing.metadata["reason"], "vision_engine_not_loaded_heuristic_fallback")
            self.assertEqual(ing.metadata["extractor"], "cpu_deterministic_visual_projection")
        self.assertEqual(calls, [], "placeholder extractor must not be called")

    def test_unloaded_genzero_engine_is_not_llm_invoked(self):
        client = GenZero()
        self.assertFalse(client.vision_engine.is_loaded)
        ing = client.modality_router.ingest(self.MIXED)
        self.assertFalse(ing.llm_invoked)
        self.assertTrue(ing.metadata["degraded"])

    def test_is_loaded_method_is_called_not_trusted(self):
        class MethodFlag:
            def is_loaded(self):
                return False

            def __call__(self, img, prompt=""):
                raise AssertionError("must not run")

        router = AdaptiveModalityRouter(vision_extractor=MethodFlag(), feature_dim=16)
        with self.assertLogs("gen_zero.gateway.modality_router", level="WARNING"):
            ing = router.ingest(self.MIXED)
        self.assertFalse(ing.llm_invoked)

    def test_loaded_extractor_is_llm_invoked(self):
        router = AdaptiveModalityRouter(vision_extractor=_LoadedVision(), feature_dim=16)
        ing = router.ingest(self.MIXED)
        self.assertTrue(ing.llm_invoked)
        self.assertNotIn("degraded", ing.metadata)
        self.assertEqual(ing.metadata["extractor"], "qwen_vlm_joint_prefill_extractor")
        self.assertEqual(ing.metadata["reason"], "joint_vision_language_multimodal_prefill")


if __name__ == "__main__":
    unittest.main()
