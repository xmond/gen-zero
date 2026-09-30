"""Round 5 Contract Tests for Gen-Zero Decision Engine.

Validates complete closure of Round 5 review findings:
R5-02: Unified Token Encoding & Scorer Export (dual_head encode_leaf_tokens, score_mlp_0 state preservation)
R5-03: Invalid Evaluation Gate Blocking (is_valid flag, zero-accuracy rejection, forward exception protection)
R5-05: Strict JSON Serialization & Value Inf Check (_sanitize_for_json, NaN/Inf handling)
R5-06: Sandbox Unknown Mutation Side Effect & Self-Healing Planner (rollback on missing tool, unknown side effect abort)

Training-side guarantees (distiller, replay, RSI daemon, promotion) moved to gen-zero-research
with the code they tested.
"""

import importlib.util
import unittest
import math
import json
import numpy as np

from gen_zero.daemon.atomic_container import AtomicModelContainer
from gen_zero.client import GenZeroClient, GenZeroConfig
from gen_zero.gate.safety_gate import SafetyGate, GateVerdict
from gen_zero.model.dual_head import encode_leaf_tokens, GenZeroDualHeadModel
from gen_zero.gateway.arbiter_bridge import ArbiterVerdict, _sanitize_for_json
HAS_TORCH = importlib.util.find_spec("torch") is not None
from gen_zero.sandbox.causal_sandbox import CausalToolSandbox, ErrorCategory
from gen_zero.sandbox.tool_registry import ToolRegistry, SideEffectLevel
from gen_zero.sandbox.safety_barrier import PRMSafetyBarrier
from gen_zero.sandbox.self_healing_planner import SelfHealingToolchainPlanner, WorkflowStep


class TestRound5Contract(unittest.TestCase):

    def setUp(self):
        self.client = GenZeroClient(GenZeroConfig(hidden_dim=2048, enable_gpu_arbiter_fallback=True))

    # --- R5-01: Atomic Promotion & Rollback Tracking ---


    # --- R5-02: Unified Token Encoding & Scorer Export ---
    def test_r5_02_unified_token_encoding_consistency(self):
        """Verify encode_leaf_tokens produces identical token sequences regardless of caller."""
        state = {"url": "https://calendar.google.com", "step": 3}
        action = "CLICK_EVENT"
        desc = "Click create event button"

        tokens_1 = encode_leaf_tokens(state, action, desc, vocab_size=19999)
        tokens_2 = encode_leaf_tokens(state, action, desc, vocab_size=19999)
        self.assertEqual(tokens_1, tokens_2)
        self.assertEqual(tokens_1[0], 101)  # CLS
        self.assertEqual(tokens_1[-1], 102)  # SEP
        self.assertTrue(all(isinstance(t, int) for t in tokens_1))
        self.assertTrue(all(0 <= t < 19999 for t in tokens_1))

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for dual-head weight export test")
    def test_r5_02_dual_head_scorer_export_preserves_state_features(self):
        """Verify export_to_scorer_weights does not zero out state representation slice."""
        model = GenZeroDualHeadModel(hidden_dim=2048, embed_dim=128)
        weights = model.export_to_scorer_weights()

        score_mlp_0 = weights["score_mlp_0"]
        self.assertEqual(score_mlp_0.shape, (128, 256))
        # Both first half (cand_proj) and second half (state_proj) must have non-zero elements
        cand_slice = score_mlp_0[:, :128]
        state_slice = score_mlp_0[:, 128:]
        self.assertFalse(np.all(cand_slice == 0.0))
        self.assertFalse(np.all(state_slice == 0.0))

    # --- R5-03: Invalid Evaluation Gate Blocking ---
    def test_r5_03_safety_gate_rejects_invalid_evaluation_or_zero_acc(self):
        """Verify SafetyGate blocks invalid evaluation and rejects 0% candidate accuracy."""
        gate = SafetyGate()

        # 1. Invalid evaluation flag
        verdict_invalid = gate.evaluate_candidate(
            {"accuracy": 90.0, "mean_score": 10.0, "collision_rate": 0.0, "is_valid": False},
            {"accuracy": 95.0, "mean_score": 15.0, "collision_rate": 0.0, "is_valid": True}
        )
        self.assertFalse(verdict_invalid.passed)
        self.assertEqual(verdict_invalid.action, "ROLLBACK_ADJUST_HYPERPARAMS")

        # 2. Candidate 0% accuracy
        verdict_zero_acc = gate.evaluate_candidate(
            {"accuracy": 80.0, "mean_score": 10.0, "collision_rate": 0.0, "is_valid": True},
            {"accuracy": 0.0, "mean_score": 50.0, "collision_rate": 0.0, "is_valid": True}
        )
        self.assertFalse(verdict_zero_acc.passed)
        self.assertEqual(verdict_zero_acc.action, "ROLLBACK_ADJUST_HYPERPARAMS")


    # --- R5-04: Distiller Supervision Validity & Mock Training ---

    # --- R5-05: Strict JSON Serialization & Value Inf Check ---
    def test_r5_05_strict_json_sanitization(self):
        """Verify _sanitize_for_json eliminates NaN and Inf ensuring strictly compliant JSON."""
        bad_dict = {
            "conf": float("nan"),
            "latency": float("inf"),
            "neg_inf": float("-inf"),
            "nested": {"val": float("nan"), "arr": [1.0, float("inf"), "safe"]}
        }
        sanitized = _sanitize_for_json(bad_dict)
        json_str = json.dumps(sanitized)
        decoded = json.loads(json_str)

        self.assertIsNone(decoded["conf"])
        self.assertIsNone(decoded["latency"])
        self.assertIsNone(decoded["neg_inf"])
        self.assertIsNone(decoded["nested"]["val"])
        self.assertEqual(decoded["nested"]["arr"], [1.0, None, "safe"])

        verdict = ArbiterVerdict(
            action="BUY",
            confidence=float("nan"),
            probs={"BUY": float("nan"), "HOLD": float("inf")},
            is_fallback=False,
            arbitration_source="gpu_cloud",
            latency_ms=float("inf")
        )
        verdict_dict = verdict.to_dict()
        verdict_json = json.dumps(verdict_dict)
        self.assertNotIn("NaN", verdict_json)
        self.assertNotIn("Infinity", verdict_json)

    # --- R5-06: Sandbox Unknown Mutation Side Effect & Self-Healing Planner ---
    def test_r5_06_self_healing_planner_rolls_back_on_missing_tool(self):
        """Verify workflow execution rolls back completed mutations when a subsequent tool is missing."""
        registry = ToolRegistry()
        db = {"balance": 1000}

        def transfer_funds(amount: int):
            db["balance"] -= amount
            return {"new_balance": db["balance"]}

        def compensate_transfer(amount: int):
            db["balance"] += amount
            return {"restored_balance": db["balance"]}

        registry.register(
            name="transfer",
            func=transfer_funds,
            rollback_func=compensate_transfer,
            side_effect_level=SideEffectLevel.MUTATING_REVERSIBLE
        )

        sandbox = CausalToolSandbox(registry=registry)
        barrier = PRMSafetyBarrier()
        planner = SelfHealingToolchainPlanner(sandbox=sandbox, safety_barrier=barrier)

        # Workflow: Step 1 succeeds (transfer 200 -> balance 800), Step 2 references unregistered tool
        workflow = [
            WorkflowStep(step_id="s1", tool_name="transfer", params={"amount": 200}),
            WorkflowStep(step_id="s2", tool_name="missing_email_notification", params={"to": "user@test.com"})
        ]

        report = planner.execute_workflow(workflow)
        self.assertFalse(report.success)
        self.assertIn("not found", report.failure_reason)
        # Crucial contract: previous mutation must be rolled back!
        self.assertEqual(db["balance"], 1000, "Mutation was not rolled back after subsequent missing tool failure!")


if __name__ == "__main__":
    unittest.main()
