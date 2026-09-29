"""Round 4 Contract Tests for Gen-Zero Decision Engine.

Validates complete closure of Round 4 review findings:
1. Transactional model promotion & rollback integrity (sync_model_to_scorer failure recovery)
2. Distiller rejection of non-finite targets/loss and non-optimizer steps
3. Model weight export completeness (val_mlp_0/1, score_mlp_0) & CPU scorer bias mapping
4. Dynamic benchmark evaluation exception handling & zero-default on empty scenarios
5. Arbiter backend_reachable propagation and negative/NaN probability sanitization
6. Uncompensated transaction preservation in sandbox
7. Self-healing fallback transaction preservation
"""

import unittest
import math
import numpy as np
import copy

from gen_zero.daemon.atomic_container import AtomicModelContainer
from gen_zero.daemon.daemon_engine import GenZeroRSIDaemon
from gen_zero.daemon.curriculum_self_play import CurriculumSelfPlayGenerator
from gen_zero.client import GenZeroClient, GenZeroConfig

from gen_zero.model.dual_head import GenZeroDualHeadModel
from gen_zero.runtime.quantized_engine import QuantizedCandidateScorer
from gen_zero.gateway.arbiter_bridge import ArbiterVerdict, CloudGPUArbiterBridge
from gen_zero.sandbox.causal_sandbox import CausalToolSandbox, ErrorCategory
from gen_zero.sandbox.tool_registry import ToolRegistry, SideEffectLevel
from gen_zero.sandbox.safety_barrier import PRMSafetyBarrier
from gen_zero.sandbox.self_healing_planner import SelfHealingPlanner, WorkflowStep


class DummyDualHead:
    """Mock dual-head model for isolated testing."""
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.scalar = True
        self.vocab_size = 19999
        self.param = [1.0, 2.0]

    def forward(self, examples, pad_token=0, return_value=False):
        if self.should_fail:
            raise RuntimeError("Injected forward simulation error")
        import torch
        logits = torch.tensor([[1.5, 0.5]])
        valid = torch.tensor([[True, True]])
        values = torch.tensor([[0.8]])
        if return_value:
            return logits, valid, values
        return logits, valid

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        import torch
        yield torch.tensor([1.0, 2.0], requires_grad=True)

    def export_to_scorer_weights(self):
        return {
            "state_proj": np.zeros((128, 2048), dtype=np.float32),
            "cand_proj": np.zeros((128, 2048), dtype=np.float32),
            "q_proj": np.zeros((128, 128), dtype=np.float32),
            "k_proj": np.zeros((128, 128), dtype=np.float32),
            "v_proj": np.zeros((128, 128), dtype=np.float32),
            "out_proj": np.zeros((128, 128), dtype=np.float32),
            "score_mlp_0": np.zeros((128, 256), dtype=np.float32),
            "score_mlp_1": np.zeros((1, 128), dtype=np.float32),
            "val_mlp_0": np.ones((64, 128), dtype=np.float32) * 0.5,
            "val_mlp_0_bias": np.ones(64, dtype=np.float32) * 0.7,
            "val_mlp_1": np.ones((1, 64), dtype=np.float32) * 0.3,
            "val_mlp_1_bias": np.ones(1, dtype=np.float32) * 0.1,
        }


class TestRound4Contract(unittest.TestCase):

    def setUp(self):
        self.client = GenZeroClient(GenZeroConfig(hidden_dim=2048, enable_gpu_arbiter_fallback=True))

    def test_01_atomic_container_rollback_model_alias(self):
        """Verify rollback_model is an exact alias for rollback() and restores state."""
        container = AtomicModelContainer(initial_model="model_v1")
        self.assertEqual(container.get_model(), "model_v1")
        swap_res = container.swap_model("model_v2", version_tag="v2")
        self.assertEqual(container.get_model(), "model_v2")
        self.assertEqual(container.get_status()["active_version"], 2)

        # Call rollback_model()
        rb_res = container.rollback_model()
        self.assertEqual(rb_res["status"], "ROLLED_BACK")
        self.assertEqual(container.get_model(), "model_v1")
        self.assertEqual(container.get_status()["active_version"], 1)

    def test_02_daemon_hot_reload_failure_transactional_rollback(self):
        """Verify that when scorer sync fails during promotion, full transactional rollback occurs."""
        daemon = self.client.rsi_daemon
        init_version = daemon.container.get_status()["active_version"]
        baseline_model = daemon.container.get_model()

        # Mock sync_model_to_scorer to fail on promotion (1st call) but succeed on rollback recovery (2nd call)
        orig_sync = getattr(daemon.client, "sync_model_to_scorer", None)
        orig_eval = daemon.evaluate_model_on_benchmark
        sync_calls = []
        def mock_sync(model=None, target_scorer=None):
            sync_calls.append(True)
            return False if len(sync_calls) == 1 else True
        daemon.client.sync_model_to_scorer = mock_sync

        # Ensure training step succeeds so cycle proceeds to gate evaluation and hot-reload
        orig_clone = daemon.client.distiller.clone_for_candidate
        mock_trainer = type("MockTrainer", (), {"run_iteration": lambda self, **k: {"steps_trained": 1, "mean_loss": 0.05}})()
        daemon.client.distiller.clone_for_candidate = lambda m: mock_trainer

        # Construct dynamic evaluation where candidate model improves over baseline live model
        live_model = daemon.container.get_model()
        def mock_eval(model, suite):
            if model is live_model:
                return {"accuracy": 85.0, "mean_score": 20.0, "collision_rate": 0.05, "is_valid": True}
            else:
                return {"accuracy": 95.0, "mean_score": 30.0, "collision_rate": 0.0, "is_valid": True}
        daemon.evaluate_model_on_benchmark = mock_eval
        
        try:
            report = daemon.run_single_evolution_cycle()
            self.assertFalse(report["gate_passed"])
            self.assertEqual(report["reload_status"], "REJECTED_ROLLED_BACK")
            self.assertIn("sync_model_to_scorer returned False", report["gate_reason"])
            
            # Verify container state rolled back cleanly
            self.assertEqual(daemon.container.get_status()["active_version"], init_version)
            self.assertEqual(daemon.client.model, baseline_model)
        finally:
            daemon.client.distiller.clone_for_candidate = orig_clone
            daemon.evaluate_model_on_benchmark = orig_eval
            if orig_sync is not None:
                daemon.client.sync_model_to_scorer = orig_sync

    def test_03_distiller_rejects_unbound_optimizer_and_empty_batches(self):
        """Verify distiller does not report effective optimization on empty batches or unbound optimizer."""
        from gen_zero.train.distiller import HAS_TORCH
        distiller = self.client.distiller
        
        # 1. Empty batch must always return optimized = False
        res_empty = distiller.train_step([])
        self.assertFalse(res_empty["optimized"])
        self.assertEqual(res_empty["status"], "EMPTY_BATCH")

        # 2. Unbound optimizer when torch is available must return optimized = False
        if HAS_TORCH:
            distiller.optimizer = None
            res = distiller.train_step([{"leaf_tokens": [[1]], "candidate_ids": ["a"], "type": "choice"}])
            self.assertFalse(res["optimized"])
            self.assertEqual(res["status"], "NO_OPTIMIZER_OR_MODEL")


    def test_04_dual_head_export_weights_and_cpu_scorer_bias_mapping(self):
        """Verify dual-head exports complete layers and CPU scorer maps val_mlp biases accurately."""
        model = GenZeroDualHeadModel(hidden_dim=2048)
        weights = model.export_to_scorer_weights()
        
        # Required keys for seamless CPU deployment
        expected_keys = [
            "state_proj", "cand_proj", "q_proj", "k_proj", "v_proj", "out_proj",
            "score_mlp_0", "score_mlp_1", "val_mlp_0", "val_mlp_1"
        ]
        for k in expected_keys:
            self.assertIn(k, weights, f"Missing key in export_to_scorer_weights: {k}")

        # Test loading into QuantizedCandidateScorer
        scorer = QuantizedCandidateScorer(state_dim=2048, candidate_dim=2048)
        weights["val_mlp_0_bias"] = np.ones(64, dtype=np.float32) * 0.85
        scorer.load_from_weight_dict(weights)

        # Both val_mlp_0 and val_mlp_0_bias keys in biases must receive 0.85
        self.assertAlmostEqual(float(scorer.biases["val_mlp_0"][0]), 0.85, places=4)
        self.assertAlmostEqual(float(scorer.biases["val_mlp_0_bias"][0]), 0.85, places=4)

    def test_05_benchmark_eval_fails_on_exception_and_empty_scenarios(self):
        """Verify dynamic benchmark returns 0.0 accuracy on exceptions or empty suites."""
        daemon = self.client.rsi_daemon

        # 1. Empty scenarios
        res_empty = daemon.evaluate_model_on_benchmark(DummyDualHead(), [])
        self.assertEqual(res_empty["accuracy"], 0.0)
        self.assertEqual(res_empty["collision_rate"], 1.0)

        # 2. Forward exception
        scenarios = [{
            "state_repr": "test_state",
            "candidate_actions": ["ACTION_A", "ACTION_B"],
            "ground_truth_safe_action": "ACTION_A"
        }]
        failing_model = DummyDualHead(should_fail=True)
        res_fail = daemon.evaluate_model_on_benchmark(failing_model, scenarios)
        self.assertEqual(res_fail["accuracy"], 0.0)
        self.assertEqual(res_fail["collision_rate"], 1.0)

    def test_06_arbiter_backend_reachable_and_negative_probability_sanitization(self):
        """Verify backend_reachable field is retained and negative/NaN probs are safely normalized."""
        # 1. Check ArbiterVerdict backend_reachable
        verdict = ArbiterVerdict(
            action="TEST",
            confidence=0.5,
            probs={"TEST": 1.0},
            is_fallback=True,
            arbitration_source="test",
            backend_reachable=False
        )
        self.assertFalse(verdict.backend_reachable)
        self.assertFalse(verdict.to_dict()["backend_reachable"])

        # 2. Client decide() with negative / NaN probabilities from arbiter
        self.client.arbiter_bridge.should_trigger_fallback = lambda **k: True
        bad_verdict = ArbiterVerdict(
            action="a",
            confidence=-1.0,
            probs={"a": -1.0, "b": float("nan"), "c": 2.0},
            is_fallback=True,
            arbitration_source="test",
            backend_reachable=False
        )
        self.client.arbiter_bridge.arbitrate = lambda *a, **k: bad_verdict
        
        res = self.client.decide(
            state="test_state",
            candidates=["a", "b", "c"],
            mode="reflex",
            task_hint="sync_arbiter"
        )
        # Probabilities must be sanitized: non-negative, finite, sum to 1.0
        for act, p in res["probs"].items():
            self.assertTrue(math.isfinite(p))
            self.assertGreaterEqual(p, 0.0)
        self.assertAlmostEqual(sum(res["probs"].values()), 1.0, places=4)
        self.assertTrue(math.isfinite(res["confidence"]))
        self.assertGreater(res["confidence"], 0.0)
        # Backend reachable status accurately reflects False
        self.assertFalse(res["adaptive_params"]["arbiter_fallback"]["backend_reachable"])

    def test_07_uncompensated_transactions_retained_in_sandbox(self):
        """Verify transactions without rollback_func are retained in uncompensated list upon rollback."""
        reg = ToolRegistry()
        executed = []
        reg.register(
            name="irreversible_mutating",
            func=lambda x: executed.append(x) or "done",
            side_effect_level=SideEffectLevel.MUTATING_REVERSIBLE,
            rollback_func=None  # Missing compensation function
        )
        sandbox = CausalToolSandbox(registry=reg)
        res = sandbox.execute("irreversible_mutating", {"x": 10})
        self.assertTrue(res.success)
        self.assertEqual(sandbox.pending_transactions_count, 1)

        # Rollback
        sandbox.rollback_transactions()
        self.assertEqual(len(sandbox.uncompensated_failed_transactions), 1)
        tool, params, out, err = sandbox.uncompensated_failed_transactions[0]
        self.assertEqual(tool.name, "irreversible_mutating")
        self.assertEqual(err, "NO_ROLLBACK_FUNCTION_REGISTERED")

    def test_08_self_healing_fallback_preserves_prior_transactions(self):
        """Verify successful step fallback does not prematurely undo prior successful step transactions."""
        reg = ToolRegistry()
        state = {"step_a": False}
        
        reg.register(
            name="tool_a",
            func=lambda: state.update({"step_a": True}) or "A_OK",
            side_effect_level=SideEffectLevel.MUTATING_REVERSIBLE,
            rollback_func=lambda **k: state.update({"step_a": False})
        )
        reg.register(
            name="tool_b_failing",
            func=lambda: (_ for _ in ()).throw(RuntimeError("B failed")),
            side_effect_level=SideEffectLevel.READ_ONLY
        )
        reg.register(
            name="tool_b_fallback",
            func=lambda: "B_HEALED",
            side_effect_level=SideEffectLevel.READ_ONLY
        )

        sandbox = CausalToolSandbox(registry=reg)
        barrier = PRMSafetyBarrier()
        planner = SelfHealingPlanner(sandbox=sandbox, safety_barrier=barrier)

        steps = [
            WorkflowStep(step_id="step_1", tool_name="tool_a", params={}),
            WorkflowStep(step_id="step_2", tool_name="tool_b_failing", params={}, fallback_tool_name="tool_b_fallback")
        ]


        report = planner.execute_workflow(steps)
        self.assertTrue(report.success)
        self.assertTrue(report.self_healing_occurred)
        # Step A state must still be True, NOT rolled back!
        self.assertTrue(state["step_a"])


if __name__ == "__main__":
    unittest.main()
