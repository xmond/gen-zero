"""Unit & Integration Tests for Issue #17: Qwen3.5-9B Open-Source Decision Foundation.

Verifies:
1. ContrastiveMutationEngine, VerifierFeedbackMiner, and ToneInvariancePairGenerator.
2. QwenPostTrainer with margin loss, tone invariance KL penalty, and sensitivity evaluation.
3. GGUFCompilationPipeline with VRAM budgeting and deployment manifest generation.
4. DecisionFoundationBenchmark verifying 11 domains, >90% accuracy, and 10-bin ECE.
"""

import unittest
import os
import json
import math

from gen_zero.train.contrastive_data_pipeline import (
    ContrastiveMutationEngine,
    VerifierFeedbackMiner,
    ToneInvariancePairGenerator,
    ContrastiveDataPipeline,
    CORE_DECISION_DOMAINS,
)
from gen_zero.train.qwen_post_trainer import (
    QwenPostTrainer,
    QwenPostTrainerConfig,
)
from gen_zero.gateway.gguf_pipeline import GGUFCompilationPipeline
from gen_zero.evaluate.decision_foundation_benchmark import DecisionFoundationBenchmark


class TestContrastiveDataPipeline(unittest.TestCase):
    """Verifies Milestone 1: Contrastive Data Pipeline & Verifier Feedback Mining."""

    def setUp(self):
        self.mutator = ContrastiveMutationEngine()
        self.miner = VerifierFeedbackMiner()
        self.tone_gen = ToneInvariancePairGenerator()
        self.pipeline = ContrastiveDataPipeline()

    def test_condition_inversion_mutation(self):
        prompt = "Cluster alert: CPU > 95% on worker-node-2"
        mutated, flipped_target, m_type = self.mutator.mutate_fact(prompt, "SCALE_UP")
        self.assertEqual(m_type, "condition_inversion")
        self.assertIn("CPU <= 95%", mutated)
        self.assertEqual(flipped_target, "MAINTAIN")

    def test_temporal_flip_mutation(self):
        prompt = "User requested: refund processed before item return"
        mutated, flipped_target, m_type = self.mutator.mutate_fact(prompt, "APPROVE")
        self.assertEqual(m_type, "temporal_flip")
        self.assertIn("refund requested after item return", mutated)
        self.assertEqual(flipped_target, "REJECT")

    def test_entity_substitution_mutation(self):
        prompt = "Access request to /api/v1/delete_user with user_role: admin"
        mutated, flipped_target, m_type = self.mutator.mutate_fact(prompt, "ALLOW")
        self.assertEqual(m_type, "entity_substitution")
        self.assertIn("user_role: guest", mutated)
        self.assertEqual(flipped_target, "DENY")

    def test_verifier_feedback_mining(self):
        # Case: Issue #13 Verifier catches false completion with failed tests
        verifier_record = {
            "task": "Fix race condition in database pool",
            "claimed_completion": "COMPLETE",
            "test_exit_code": 1,
            "error_trace": "AssertionError: Connection timed out at pool.py:88",
            "failed_criteria": ["exit_code == 0", "all unit tests passing"],
        }
        mined_sample = self.miner.mine_from_verifier_log(verifier_record)
        self.assertIsNotNone(mined_sample)
        self.assertEqual(mined_sample["domain"], "verifier_audit")
        self.assertEqual(mined_sample["target_choice"], "REVISE")
        self.assertTrue(mined_sample["is_hard_negative"])
        self.assertIn("AssertionError", mined_sample["prompt"])

    def test_tone_invariance_generation(self):
        prompt = "Database replication lag exceeds 30 seconds"
        candidates = ["ROUTE_PRIMARY", "WAIT"]
        pairs = self.tone_gen.generate_tone_invariance_pair(prompt, candidates, "ROUTE_PRIMARY")
        self.assertEqual(len(pairs), 2)

        neutral_item, emotional_item = pairs[0], pairs[1]
        self.assertEqual(neutral_item["tone"], "neutral")
        self.assertEqual(emotional_item["tone"], "emotional_adversarial")
        # Invariance contract: ground truth MUST be identical
        self.assertEqual(neutral_item["target_choice"], emotional_item["target_choice"])
        self.assertEqual(neutral_item["pair_id"], emotional_item["pair_id"])

    def test_full_corpus_generation_11_domains(self):
        corpus = self.pipeline.generate_benchmark_corpus(samples_per_domain=3)
        self.assertGreater(len(corpus), 0)

        # Check coverage of all 11 domains
        domains_present = {item["domain"] for item in corpus}
        for d in CORE_DECISION_DOMAINS:
            self.assertIn(d, domains_present)


class TestQwenPostTrainer(unittest.TestCase):
    """Verifies Milestone 2: Qwen3.5-9B Post-Training & Tone Decoupling."""

    def setUp(self):
        self.config = QwenPostTrainerConfig(contrastive_margin=1.0)
        self.trainer = QwenPostTrainer(self.config)

    def test_contrastive_loss_computation(self):
        # Case 1: Target logit strictly beats distractors by margin -> Loss = 0
        logits_good = [3.5, 0.5, -1.0]
        loss_good = self.trainer.compute_contrastive_loss(logits_good, target_index=0, margin=1.0)
        self.assertEqual(loss_good, 0.0)

        # Case 2: Target logit beaten by distractor -> Positive hinge loss
        logits_bad = [0.5, 2.0, -1.0]
        loss_bad = self.trainer.compute_contrastive_loss(logits_bad, target_index=0, margin=1.0)
        self.assertGreater(loss_bad, 0.0)

    def test_tone_invariance_kl(self):
        probs_a = [0.8, 0.15, 0.05]
        probs_b = [0.78, 0.17, 0.05]
        kl = self.trainer.compute_tone_invariance_kl(probs_a, probs_b)
        self.assertGreaterEqual(kl, 0.0)
        self.assertLess(kl, 0.1)

    def test_train_step(self):
        neutral_sample = {
            "prompt": "Cluster CPU > 95% on node-1",
            "candidates": ["SCALE_UP", "MAINTAIN"],
            "target_choice": "SCALE_UP"
        }
        perturbed_sample = {
            "prompt": "OMG disaster! Cluster CPU > 95% on node-1",
            "candidates": ["SCALE_UP", "MAINTAIN"],
            "target_choice": "SCALE_UP"
        }
        step_metrics = self.trainer.train_step(neutral_sample, perturbed_sample)
        self.assertIn("step", step_metrics)
        self.assertIn("contrastive_loss", step_metrics)
        self.assertIn("tone_kl_loss", step_metrics)
        self.assertIn("total_loss", step_metrics)

    def test_evaluate_tone_sensitivity_drop_rate(self):
        eval_pairs = [
            (
                {"prompt": "Disk usage > 90%", "candidates": ["CLEAN", "WAIT"], "target_choice": "CLEAN"},
                {"prompt": "Panic! Disk usage > 90%", "candidates": ["CLEAN", "WAIT"], "target_choice": "CLEAN"}
            ),
            (
                {"prompt": "Memory usage <= 30%", "candidates": ["CLEAN", "WAIT"], "target_choice": "WAIT"},
                {"prompt": "So broken! Memory usage <= 30%", "candidates": ["CLEAN", "WAIT"], "target_choice": "WAIT"}
            )
        ]
        res = self.trainer.evaluate_tone_sensitivity(eval_pairs)
        self.assertIn("drop_rate", res)
        self.assertLessEqual(res["drop_rate"], 0.02)
        self.assertTrue(res["passes_sla"])


class TestGGUFCompilationPipeline(unittest.TestCase):
    """Verifies Milestone 3: GGUF Quantization & ai-server Deployment."""

    def setUp(self):
        self.pipeline = GGUFCompilationPipeline(
            base_model="Qwen/Qwen3.5-9B",
            server_endpoint="http://ai-server:8080"
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


class TestDecisionFoundationBenchmark(unittest.TestCase):
    """Verifies Milestone 4: 11-Domain Benchmark & 10-Bin ECE Calibration Suite."""

    def setUp(self):
        self.benchmark = DecisionFoundationBenchmark()

    def test_full_benchmark_run_and_reporting(self):
        results = self.benchmark.run_full_benchmark(samples_per_domain=5)

        # 1. Overall Accuracy Target >= 90%
        self.assertIn("overall_accuracy", results)
        self.assertGreaterEqual(results["overall_accuracy"], 0.90)
        self.assertTrue(results["meets_90pct_target"])

        # 2. 11 Domain Coverage
        self.assertEqual(len(results["domain_accuracies"]), 11)
        for d in CORE_DECISION_DOMAINS:
            self.assertIn(d, results["domain_accuracies"])

        # 3. 10-Bin ECE Calibration
        cal = results["calibration"]
        self.assertIn("ece_10bin", cal)
        self.assertLessEqual(cal["ece_10bin"], 0.35)
        self.assertTrue(cal["passed_safety_red_line"])

        # 4. Tone Robustness SLA (drop <= 2%)
        tone = results["tone_invariance"]
        self.assertLessEqual(tone["drop_rate"], 0.02)
        self.assertTrue(tone["meets_2pct_drop_sla"])

        # 5. Permutation Invariance (flip rate == 0.0)
        perm = results["permutation_invariance"]
        self.assertEqual(perm["argmax_flip_rate"], 0.0)
        self.assertTrue(perm["is_permutation_equivariant"])

        # 6. Artifact persistence
        self.assertTrue(os.path.exists("results/gen_zero/qwen_foundation_benchmark_results.json"))
        self.assertTrue(os.path.exists("results/gen_zero/qwen_foundation_benchmark_report.md"))


if __name__ == "__main__":
    unittest.main()
