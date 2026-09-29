"""Unit and Contract Tests for 4-Dimensional Perturbation Stability Gate."""

import unittest
from typing import Dict, List, Any

from gen_zero.gate.safety_gate import SafetyGate, PerturbationStabilityGate, GateVerdict
from gen_zero.gate.evaluate_perturbations import PerturbationEvaluator


class TestPerturbationStabilityGate(unittest.TestCase):
    """Verifies 4-dimensional invariance constraints and safety gate integration."""

    def setUp(self):
        self.gate = PerturbationStabilityGate()
        self.safety_gate = SafetyGate()
        self.base_metrics = {
            "accuracy": 85.0,
            "mean_score": 0.75,
            "collision_rate": 0.02,
            "is_valid": True
        }
        self.cand_metrics = {
            "accuracy": 86.0,
            "mean_score": 0.78,
            "collision_rate": 0.015,
            "is_valid": True
        }

    def test_ideal_candidate_passes_all_dimensions(self):
        """Candidate with perfect permutation equivariance and abstain recall passes."""
        metrics = {
            "dim1_dfr": 0.0,
            "dim1_tvd": 0.0,
            "dim2_dfr": 0.005,
            "dim2_tvd": 0.02,
            "dim3_dfr": 0.003,
            "dim3_tvd": 0.015,
            "dim4_abstain_recall": 1.0,
            "dim4_max_confidence": 0.0,
            "is_valid": True
        }
        verdict = self.gate.evaluate(metrics)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.action, "DEPLOY_HOT_UPDATE")
        self.assertNotIn("warning", verdict.details)

        # Integrated with SafetyGate
        full_verdict = self.safety_gate.evaluate_candidate(
            self.base_metrics,
            self.cand_metrics,
            perturbation_metrics=metrics
        )
        self.assertTrue(full_verdict.passed)
        self.assertEqual(full_verdict.action, "DEPLOY_HOT_UPDATE")

    def test_dim1_order_reversal_flip_triggers_hard_rollback(self):
        """Any decision flip (DFR > 0.0%) under candidate order reversal triggers immediate hard rollback."""
        metrics = {
            "dim1_dfr": 0.02,  # 2% flip rate violation
            "dim1_tvd": 0.005,
            "dim2_dfr": 0.0,
            "dim2_tvd": 0.0,
            "dim3_dfr": 0.0,
            "dim3_tvd": 0.0,
            "dim4_abstain_recall": 1.0,
            "dim4_max_confidence": 0.0,
            "is_valid": True
        }
        verdict = self.gate.evaluate(metrics)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.action, "ROLLBACK_ADJUST_HYPERPARAMS")
        self.assertFalse(verdict.details["dim1_order_invariance"]["passed"])

        # Integrated SafetyGate also triggers rollback
        full_verdict = self.safety_gate.evaluate_candidate(
            self.base_metrics,
            self.cand_metrics,
            perturbation_metrics=metrics
        )
        self.assertFalse(full_verdict.passed)
        self.assertEqual(full_verdict.action, "ROLLBACK_ADJUST_HYPERPARAMS")

    def test_dim4_missing_evidence_failure_triggers_hard_rollback(self):
        """Failure to actively abstain (< 98.0%) or high confidence (>= 0.70) triggers rollback."""
        # Low recall
        metrics_low_recall = {
            "dim1_dfr": 0.0,
            "dim1_tvd": 0.0,
            "dim4_abstain_recall": 0.90,  # Below 98%
            "dim4_max_confidence": 0.40,
            "is_valid": True
        }
        v_low = self.gate.evaluate(metrics_low_recall)
        self.assertFalse(v_low.passed)
        self.assertEqual(v_low.action, "ROLLBACK_ADJUST_HYPERPARAMS")

        # High confidence on hallucinated action
        metrics_high_conf = {
            "dim1_dfr": 0.0,
            "dim1_tvd": 0.0,
            "dim4_abstain_recall": 1.0,
            "dim4_max_confidence": 0.85,  # >= 0.70 violation
            "is_valid": True
        }
        v_conf = self.gate.evaluate(metrics_high_conf)
        self.assertFalse(v_conf.passed)
        self.assertEqual(v_conf.action, "ROLLBACK_ADJUST_HYPERPARAMS")

    def test_dim2_dim3_degraded_warning_alert(self):
        """Dim 2 (semantic wrapper) or Dim 3 (noise injection) degradation emits warning without hard rollback."""
        metrics_degraded = {
            "dim1_dfr": 0.0,
            "dim1_tvd": 0.0,
            "dim2_dfr": 0.025,  # Exceeds 1.5%
            "dim2_tvd": 0.06,   # Exceeds 0.05
            "dim3_dfr": 0.005,
            "dim3_tvd": 0.01,
            "dim4_abstain_recall": 1.0,
            "dim4_max_confidence": 0.20,
            "is_valid": True
        }
        verdict = self.gate.evaluate(metrics_degraded)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.action, "DEPLOY_HOT_UPDATE")
        self.assertEqual(verdict.details.get("warning"), "ALERT_DEGRADED_STABILITY")

    def test_missing_or_invalid_flag_rejection(self):
        """Metrics dict without is_valid=True or with non-finite values is rejected."""
        v_no_flag = self.gate.evaluate({"dim1_dfr": 0.0})
        self.assertFalse(v_no_flag.passed)
        self.assertEqual(v_no_flag.action, "ROLLBACK_ADJUST_HYPERPARAMS")

        v_nan = self.gate.evaluate({
            "dim1_dfr": float("nan"),
            "dim1_tvd": 0.0,
            "dim4_abstain_recall": 1.0,
            "dim4_max_confidence": 0.0,
            "is_valid": True
        })
        self.assertFalse(v_nan.passed)
        self.assertEqual(v_nan.action, "ROLLBACK_ADJUST_HYPERPARAMS")

    def test_perturbation_evaluator_harness_execution(self):
        """PerturbationEvaluator executes over the 108-case blind suite."""
        evaluator = PerturbationEvaluator()
        self.assertEqual(len(evaluator.cases), 108)

        # Mock model decision function with perfect equivariance and abstain
        def mock_decide(state: str, candidates: List[str]) -> Dict[str, Any]:
            if "CORRUPTED_STREAM" in state or "ambiguous request" in state:
                return {"action": "ABSTAIN", "confidence": 0.0, "probs": {c: 0.0 for c in candidates}}
            # Deterministic choice based on candidate hash (order invariant)
            # Pick candidate with lowest string sort
            best = sorted(candidates)[0]
            probs = {c: 0.8 if c == best else 0.1 for c in candidates}
            return {"action": best, "confidence": 0.8, "probs": probs}

        # Tier 1 (Fast In-loop)
        t1_res = evaluator.evaluate_model(mock_decide, tier=1)
        self.assertEqual(t1_res["dim1_dfr"], 0.0)
        self.assertEqual(t1_res["dim4_abstain_recall"], 1.0)
        self.assertEqual(t1_res["tier"], 1)

        # Tier 2 (Full 108 cases)
        t2_res = evaluator.evaluate_model(mock_decide, tier=2)
        self.assertEqual(t2_res["total_evaluated"], 108)
        self.assertEqual(t2_res["dim1_dfr"], 0.0)
        self.assertEqual(t2_res["dim4_abstain_recall"], 1.0)

    def test_backwards_compatibility_without_perturbation_metrics(self):
        """SafetyGate evaluates normally when perturbation_metrics is None."""
        verdict = self.safety_gate.evaluate_candidate(
            self.base_metrics,
            self.cand_metrics
        )
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.action, "DEPLOY_HOT_UPDATE")
        self.assertNotIn("perturbation_check", verdict.details)


if __name__ == "__main__":
    unittest.main()
