"""Unit tests for Gen-Zero Browser DX Pretraining and Probability Sharpness."""

import unittest
import os
import json
from gen_zero.client import GenZero
from gen_zero.runtime.quantized_engine import QuantizedCandidateScorer


class TestBrowserPretrain(unittest.TestCase):
    def setUp(self):
        self.client = GenZero()
        self.dataset_path = "data/browser_dx_pretrain_dataset.jsonl"


    @unittest.skipUnless(
        os.environ.get("GENZERO_DUAL_HEAD_CHECKPOINT"),
        "needs trained dual-head weights: the keyword/synonym affinity table that used to "
        "make this pass without a model was removed as a fake-inference shortcut",
    )
    def test_browser_choice_sharpness(self):
        criteria = {
            "e410": "[button] Skip to main content",
            "e411": "[button] Keyboard shortcuts",
            "e412": "[button] Accessibility feedback",
            "e413": "[button] Main navigation drawer",
            "e414": "[button] Today, September 20 (Sunday)",
            "e415": "[button] Previous month",
            "e416": "[button] Next month",
            "e417": "[button] Search",
            "e418": "[button] Support",
            "e419": "[button] Settings menu",
            "e420": "[button] Month",
            "e421": "[button] Use Google Calendar",
            "e422": "[button] Switch to Tasks",
            "e423": "[button] Google apps",
            "e425": "[button] Create"
        }
        state = "Site: calendar.google.com | Page: Google Calendar - Sep 2026 | Goal: Click create button to add event"

        # 1. With candidate descriptions: Sharp decision
        res = self.client.decide(
            state=state,
            candidates=list(criteria.keys()),
            mode="reflex",
            candidate_descriptions=criteria
        )
        self.assertEqual(res["action"], "e425")
        self.assertGreaterEqual(res["confidence"], 0.90)
        self.assertGreaterEqual(res["probs"]["e425"], 0.90)

    @unittest.skipUnless(
        os.environ.get("GENZERO_DUAL_HEAD_CHECKPOINT"),
        "needs trained dual-head weights: the keyword/synonym affinity table that used to "
        "make this pass without a model was removed as a fake-inference shortcut",
    )
    def test_browser_all_dataset_samples(self):
        if not os.path.exists(self.dataset_path):
            self.skipTest("Browser dataset file not found")

        correct = 0
        total = 0
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                gt = item["ground_truth_ref"]
                crit = item["questions"]["target_element"]["criteria"]
                state = item["state"]

                res = self.client.decide(
                    state=state,
                    candidates=list(crit.keys()),
                    mode="reflex",
                    candidate_descriptions=crit
                )
                total += 1
                if res["action"] == gt:
                    correct += 1

        acc = correct / max(1, total)
        self.assertEqual(acc, 1.0, f"Expected 100% accuracy on pretrain dataset, got {acc*100:.1f}%")


if __name__ == "__main__":
    unittest.main()
