"""Round 5 Contract Tests for Gen-Zero Decision Engine.

Validates complete closure of Round 5 review findings:
R5-01: Atomic Promotion & Rollback Tracking (no premature container pop, ROLLBACK_INCOMPLETE reporting)
R5-02: Unified Token Encoding & Scorer Export (dual_head encode_leaf_tokens, score_mlp_0 state preservation)
R5-03: Invalid Evaluation Gate Blocking (is_valid flag, zero-accuracy rejection, forward exception protection)
R5-04: Distiller Supervision Validity & Mock Training (MISSING_SUPERVISION, NEGATIVE_TARGET_PROBABILITIES, SKIPPED_NO_TRAINING_BACKEND)
R5-05: Strict JSON Serialization & Value Inf Check (_sanitize_for_json, NaN/Inf handling)
R5-06: Sandbox Unknown Mutation Side Effect & Self-Healing Planner (rollback on missing tool, unknown side effect abort)
"""

import unittest
import math
import json
import numpy as np

from gen_zero.daemon.atomic_container import AtomicModelContainer
from gen_zero.daemon.daemon_engine import GenZeroRSIDaemon
from gen_zero.client import GenZeroClient, GenZeroConfig
from gen_zero.gate.safety_gate import SafetyGate, GateVerdict
from gen_zero.model.dual_head import encode_leaf_tokens, GenZeroDualHeadModel
from gen_zero.gateway.arbiter_bridge import ArbiterVerdict, _sanitize_for_json
from gen_zero.train.distiller import GenZeroDistiller, HAS_TORCH
from gen_zero.train.replay_buffer import StabilityReplayBuffer
from gen_zero.sandbox.causal_sandbox import CausalToolSandbox, ErrorCategory
from gen_zero.sandbox.tool_registry import ToolRegistry, SideEffectLevel
from gen_zero.sandbox.safety_barrier import PRMSafetyBarrier
from gen_zero.sandbox.self_healing_planner import SelfHealingToolchainPlanner, WorkflowStep


class TestRound5Contract(unittest.TestCase):

    def setUp(self):
        self.client = GenZeroClient(GenZeroConfig(hidden_dim=2048, enable_gpu_arbiter_fallback=True))

    # --- R5-01: Atomic Promotion & Rollback Tracking ---
    def test_r5_01_no_premature_rollback_when_no_swap_occurred(self):
        """Verify container.rollback() is NOT called if failure happens before swap_model."""
        daemon = self.client.rsi_daemon
        container = daemon.container
        init_version = container.get_status()["active_version"]

        # If gate rejects candidate, swap_model is never called and active_version remains untouched
        mock_trainer = type("MockTrainer", (), {"run_iteration": lambda self, **k: {"steps_trained": 1, "mean_loss": 0.05}})()
        orig_clone = daemon.client.distiller.clone_for_candidate
        daemon.client.distiller.clone_for_candidate = lambda m: mock_trainer

        # Force gate failure
        orig_eval = daemon.evaluate_model_on_benchmark
        daemon.evaluate_model_on_benchmark = lambda m, s: {"accuracy": 0.0, "mean_score": 0.0, "collision_rate": 1.0, "is_valid": True}

        try:
            report = daemon.run_single_evolution_cycle()
            self.assertFalse(report["gate_passed"])
            self.assertEqual(report["reload_status"], "REJECTED_ROLLED_BACK")
            # Crucial check: container history was not popped
            self.assertEqual(container.get_status()["active_version"], init_version)
        finally:
            daemon.client.distiller.clone_for_candidate = orig_clone
            daemon.evaluate_model_on_benchmark = orig_eval

    def test_r5_01_rollback_incomplete_reported_when_recovery_sync_fails(self):
        """Verify reload_status is marked ROLLBACK_INCOMPLETE when scorer recovery sync fails."""
        daemon = self.client.rsi_daemon
        orig_sync = getattr(daemon.client, "sync_model_to_scorer", None)
        orig_eval = daemon.evaluate_model_on_benchmark
        orig_clone = daemon.client.distiller.clone_for_candidate

        # sync_model_to_scorer always returns False
        daemon.client.sync_model_to_scorer = lambda model=None, target_scorer=None: False
        daemon.client.distiller.clone_for_candidate = lambda m: type("MockTrainer", (), {"run_iteration": lambda self, **k: {"steps_trained": 1, "mean_loss": 0.05}})()

        live_model = daemon.container.get_model()
        daemon.evaluate_model_on_benchmark = lambda m, s: {
            "accuracy": 85.0 if m is live_model else 95.0,
            "mean_score": 20.0 if m is live_model else 30.0,
            "collision_rate": 0.0,
            "is_valid": True
        }

        try:
            report = daemon.run_single_evolution_cycle()
            self.assertFalse(report["gate_passed"])
            self.assertEqual(report["reload_status"], "ROLLBACK_INCOMPLETE")
            self.assertIn("sync_model_to_scorer returned False", report["gate_reason"])
        finally:
            daemon.client.distiller.clone_for_candidate = orig_clone
            daemon.evaluate_model_on_benchmark = orig_eval
            if orig_sync is not None:
                daemon.client.sync_model_to_scorer = orig_sync

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

    def test_r5_03_benchmark_eval_handles_forward_exception(self):
        """Verify evaluate_model_on_benchmark marks is_valid=False when model raises during eval."""
        daemon = self.client.rsi_daemon
        class BrokenModel:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("Hardware/CUDA OOM during evaluation")
        broken_model = BrokenModel()

        res = daemon.evaluate_model_on_benchmark(broken_model, [{"state_repr": [0.1]*10, "candidate_actions": ["a"], "ground_truth_safe_action": "a"}])
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["accuracy"], 0.0)
        self.assertIn("EVALUATION_EXCEPTION", res["status"])

    # --- R5-04: Distiller Supervision Validity & Mock Training ---
    def test_r5_04_distiller_rejects_invalid_supervision(self):
        """Verify distiller rejects missing supervision and negative probabilities."""
        buf = StabilityReplayBuffer(capacity=10)

        if HAS_TORCH:
            import torch
            import torch.nn as nn
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(4, 2)
                    self.scalar = self.linear
                def forward(self, *a, **k):
                    b = len(a[0])
                    return self.linear(torch.ones((b, 4))), torch.ones((b, 2), dtype=torch.bool), torch.zeros((b, 1))

            distiller = GenZeroDistiller(model=DummyModel(), replay_buffer=buf)

            # Missing supervision
            res_no_sup = distiller.train_step([{"id": "ex1", "state": {}}])
            self.assertFalse(res_no_sup["optimized"])
            self.assertEqual(res_no_sup["status"], "MISSING_SUPERVISION")

            # Negative target probability
            bad_batch = [{
                "id": "ex2", "state": {}, "candidate_ids": ["c1", "c2"],
                "pi_target": {"c1": -0.5, "c2": 1.5}, "leaf_tokens": [[101, 102], [101, 102]]
            }]
            res_neg = distiller.train_step(bad_batch)
            self.assertFalse(res_neg["optimized"])
            self.assertEqual(res_neg["status"], "NEGATIVE_TARGET_PROBABILITIES")
        else:
            distiller_mock = GenZeroDistiller(model=None, replay_buffer=buf)
            res_no_torch = distiller_mock.train_step([{"id": "ex1", "state": {}}])
            self.assertFalse(res_no_torch["optimized"])
            self.assertEqual(res_no_torch["status"], "SKIPPED_NO_TRAINING_BACKEND")

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
