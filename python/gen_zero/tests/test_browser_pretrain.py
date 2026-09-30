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
            "e410": "[button] 跳至主要内容",
            "e411": "[button] 键盘快捷键",
            "e412": "[button] 无障碍功能反馈",
            "e413": "[button] 抽屉式主导航栏",
            "e414": "[button] 今天，9月 20日 (星期日)",
            "e415": "[button] 上一个月",
            "e416": "[button] 下一个月",
            "e417": "[button] 搜索",
            "e418": "[button] 支持",
            "e419": "[button] “设置”菜单",
            "e420": "[button] 月",
            "e421": "[button] 改用 Google 日历",
            "e422": "[button] 切换到 Tasks",
            "e423": "[button] Google 应用",
            "e425": "[button] 创建"
        }
        state = "Site: calendar.google.com | Page: Google 日历 - 2026年9月 | Goal: 点击创建按钮新建活动"

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
