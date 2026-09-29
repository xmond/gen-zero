"""Unit tests for Gen-Zero Pure PyTorch Vision Decision Pipeline."""

import unittest
from gen_zero.gateway.modality_router import (
    AdaptiveModalityRouter,
    ModalityType,
)
from gen_zero.adaptive_engine import AdaptiveParameterEngine
from gen_zero.vision.engine import PyTorchVisualDecisionEngine
from gen_zero.client import GenZero


class TestVisionDecisionPipeline(unittest.TestCase):
    def setUp(self):
        self.router = AdaptiveModalityRouter(feature_dim=128)
        self.vision_engine = PyTorchVisualDecisionEngine(feature_dim=128)

    def test_vision_modality_classification(self):
        # 1. Image path string
        img_path = "assets/game_screenshot_frame01.png"
        mod = self.router.classify_modality(img_path)
        self.assertEqual(mod, ModalityType.VISION_IMAGE)

        # 2. Image object with size and mode (mocking PIL Image)
        class MockPILImage:
            size = (224, 224)
            mode = "RGB"
            def save(self, *args): pass

        mock_img = MockPILImage()
        mod_pil = self.router.classify_modality(mock_img)
        self.assertEqual(mod_pil, ModalityType.VISION_IMAGE)

        # 3. Multimodal vision + text dict
        multi_input = {
            "image": mock_img,
            "instruction": "Identify the paddle position and avoid ball collision",
            "score": 450
        }
        mod_multi = self.router.classify_modality(multi_input)
        self.assertEqual(mod_multi, ModalityType.MULTIMODAL_VISION_TEXT)

    def test_adaptive_formula_17_vision(self):
        # Test Formula 17 with image input
        mod, invoke, rat = AdaptiveParameterEngine.dynamic_modality_decision("robot_camera.jpg")
        self.assertEqual(mod, "vision_image")
        self.assertTrue(invoke)
        self.assertIn("vlm", rat)

    def test_unloaded_vision_engine_fails_closed(self):
        expected = "Vision model weights are not loaded. Provide a valid checkpoint."
        with self.assertRaisesRegex(RuntimeError, expected):
            self.vision_engine.prefill_visual_context(
                image="screenshot.png",
                prompt="Which action keeps the ball alive?",
            )

        with self.assertRaisesRegex(RuntimeError, expected):
            self.vision_engine.score_candidates_direct(
                prefill_bundle={},
                candidates=["LEFT", "STAY", "RIGHT"],
                temperature=0.8,
            )

    def test_client_decide_visual_requires_checkpoint(self):
        client = GenZero()
        with self.assertRaisesRegex(
            RuntimeError,
            "Vision model weights are not loaded\\. Provide a valid checkpoint\\.",
        ):
            client.decide_visual(
                image="breakout_frame.png",
                candidates=["REGION_1", "REGION_2", "REGION_3", "REGION_4", "REGION_5"],
                prompt="Which numbered region contains the moving ball?",
                mode="auto",
            )

    def test_windows_cross_platform_compatibility(self):
        # 1. Windows path detection (e.g. C:\zero\models\Qwen3.5-0.8B)
        win_path = r"C:\zero\models\Qwen3.5-0.8B"
        norm_path = self.vision_engine._normalize_model_path(win_path)
        self.assertTrue("Qwen3.5-0.8B" in norm_path)

        # 2. Windows image path input (e.g. C:\zero\data\frame_001.png)
        win_img_path = r"C:\zero\data\frame_001.png"
        mod = self.router.classify_modality(win_img_path)
        self.assertEqual(mod, ModalityType.VISION_IMAGE)


if __name__ == "__main__":
    unittest.main()
