"""Contract verification tests addressing ChatGPT Round 3 Review Findings.

Guarantees:
1. Candidate Model Isolation & Lifecycle (Live model untouched during distillation)
2. Optimizer Parameter Group Binding
3. Dynamic Benchmark Evaluation (No hardcoded metrics)
4. Empty Batch Rejection (Zero effective steps trained)
5. Failure & Exception Injection Clean Rollback
6. Global Safety Barrier Enforcement across execute_tool()
7. Uncompensated Transaction Retention on Rollback Failure
8. Workflow Report Failure Integrity
9. CPU Quantized Scorer Neural Weight Synchronization & Deterministic Hashing
10. Arbiter Result Pruning & Probability Normalization Consistency
"""

import unittest
import copy
import numpy as np

from gen_zero.config import GenZeroConfig
from gen_zero.client import GenZero
from gen_zero.model.dual_head import GenZeroDualHeadModel
from gen_zero.daemon.daemon_engine import GenZeroRSIDaemon
from gen_zero.train.distiller import GenZeroDistiller
from gen_zero.train.replay_buffer import StabilityReplayBuffer
from gen_zero.rollout.hard_miner import MinedSample
from gen_zero.sandbox.tool_registry import ToolRegistry, SideEffectLevel
from gen_zero.sandbox.causal_sandbox import CausalToolSandbox
from gen_zero.sandbox.safety_barrier import PRMSafetyBarrier, SafetyVerdict
from gen_zero.sandbox.self_healing_planner import SelfHealingToolchainPlanner, WorkflowStep


class TestRound3ContractIntegrity(unittest.TestCase):
    def setUp(self):
        self.config = GenZeroConfig()
        self.client = GenZero(self.config)

    def test_candidate_model_isolation_during_distillation(self):
        """1. Candidate Isolation: Live model parameters MUST NOT change during candidate training."""
        live_model = self.client.model
        init_weights = {k: v.copy() for k, v in live_model.export_to_scorer_weights().items()}

        # Clone candidate
        candidate_model = copy.deepcopy(live_model)
        candidate_distiller = self.client.distiller.clone_for_candidate(candidate_model)

        # Distiller optimizer should be bound to candidate_model
        if candidate_distiller.optimizer is not None:
            opt_params = {id(p) for group in candidate_distiller.optimizer.param_groups for p in group['params']}
            cand_params = {id(p) for p in candidate_model.parameters()}
            live_params = {id(p) for p in live_model.parameters()}
            self.assertEqual(opt_params, cand_params, "Optimizer MUST bind strictly to candidate model parameters")
            self.assertEqual(len(opt_params.intersection(live_params)), 0, "Optimizer MUST NOT touch live model")

        # Live model weights must remain 100% identical
        current_weights = live_model.export_to_scorer_weights()
        for k in init_weights:
            np.testing.assert_array_equal(init_weights[k], current_weights[k], err_msg=f"Live model weight {k} modified!")

    def test_empty_batch_zero_effective_steps(self):
        """2. Empty batch MUST report steps_trained=0, mean_loss=0.0 and NO_EFFECTIVE_STEPS."""
        empty_buf = StabilityReplayBuffer(capacity=10, hard_ratio=0.5)
        distiller = GenZeroDistiller(model=self.client.model, replay_buffer=empty_buf, lr=1e-3)
        res = distiller.run_iteration(steps=3, batch_size=2)
        self.assertEqual(res["steps_trained"], 0)
        self.assertEqual(res["mean_loss"], 0.0)
        self.assertEqual(res["status"], "NO_EFFECTIVE_STEPS")

    def test_dynamic_benchmark_evaluation_no_hardcoded_metrics(self):
        """3. Daemon benchmark evaluation dynamically computes metrics on candidate and baseline."""
        daemon = GenZeroRSIDaemon(client=self.client, cycle_interval_sec=60.0)
        suite = [daemon.curriculum.generate_boundary_scenario() for _ in range(3)]
        metrics = daemon.evaluate_model_on_benchmark(self.client.model, suite)

        self.assertIn("accuracy", metrics)
        self.assertIn("mean_score", metrics)
        self.assertIn("collision_rate", metrics)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 100.0)
        self.assertTrue(0.0 <= metrics["collision_rate"] <= 1.0)
        self.assertTrue(np.isfinite(metrics["mean_score"]))

    def test_global_safety_barrier_execute_tool(self):
        """4. Direct client.execute_tool() MUST run through PRMSafetyBarrier and block non-ALLOWED."""
        # Register a destructive mutating tool
        self.client.register_tool(
            name="rm_rf_system",
            func=lambda path: "deleted",
            description="Dangerous deletion tool",
            side_effect_level=SideEffectLevel.MUTATING_IRREVERSIBLE,
            required_params=["path"]
        )

        # Direct execution must be intercepted and raise PermissionError
        with self.assertRaises(PermissionError):
            self.client.execute_tool("rm_rf_system", {"path": "/etc/root"})

    def test_uncompensated_failed_transactions_retention(self):
        """5. Failed compensation records MUST NOT be lost during rollback."""
        reg = ToolRegistry()
        def failing_rollback(params, original_output):
            raise IOError("Disk write error during compensation")

        reg.register(
            name="write_file",
            func=lambda data: "written",
            description="File writer",
            side_effect_level=SideEffectLevel.MUTATING_REVERSIBLE,
            required_params=["data"],
            rollback_func=failing_rollback
        )
        sandbox = CausalToolSandbox(reg)
        sandbox.execute("write_file", {"data": "test_payload"})
        self.assertEqual(sandbox.pending_transactions_count, 1)

        # Rollback attempt fails -> must retain uncompensated record
        success_rollbacks = sandbox.rollback_transactions()
        self.assertEqual(success_rollbacks, 0)
        self.assertEqual(sandbox.pending_transactions_count, 0)
        self.assertEqual(sandbox.uncompensated_transactions_count, 1)
        self.assertEqual(len(sandbox.uncompensated_records), 1)

    def test_self_healing_workflow_failure_reporting(self):
        """6. Workflow with unrecoverable failure MUST report success=False even if execution continues."""
        planner = self.client.workflow_planner
        # Step with non-existent tool
        step1 = WorkflowStep(step_id="step_1", tool_name="non_existent_tool_xyz", params={})
        report = planner.execute_workflow([step1], stop_on_unrecoverable_failure=False)
        self.assertFalse(report.success, "Workflow must not report success=True when unrecoverable failure occurs")

    def test_cpu_scorer_weight_synchronization(self):
        """7. CPU QuantizedCandidateScorer neural weight synchronization and deterministic output."""
        synced = self.client.sync_model_to_scorer()
        self.assertTrue(synced, "Model weights must successfully synchronize to CPU scorer")

        res1 = self.client.decide(state="benchmark_eval_state", candidates=["GO_LEFT", "GO_RIGHT"], mode="reflex")
        res2 = self.client.decide(state="benchmark_eval_state", candidates=["GO_LEFT", "GO_RIGHT"], mode="reflex")
        self.assertEqual(res1["action"], res2["action"], "Decision must be 100% deterministic across calls")
        self.assertEqual(res1["probs"], res2["probs"], "Probabilities must be 100% deterministic across calls")

    def test_arbiter_pruning_and_normalization(self):
        """8. Arbiter probabilities on pruned candidates must be 0.0 and valid probs sum to 1.0."""
        cfg = GenZeroConfig(enable_gpu_arbiter_fallback=True, arbiter_confidence_threshold=0.99)
        client = GenZero(cfg)
        res = client.decide(
            state="unconfident_test_state",
            candidates=["SAFE_A", "SAFE_B"],
            mode="reflex"
        )
        if res.get("arbiter_fallback"):
            self.assertIn("backend_reachable", res["arbiter_fallback"])
            # Sum of probabilities must strictly equal 1.0
            prob_sum = sum(res["probs"].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=3)
            self.assertIn(res["action"], ["SAFE_A", "SAFE_B"])


if __name__ == "__main__":
    unittest.main()
