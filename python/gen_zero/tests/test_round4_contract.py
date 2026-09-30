"""Round 4 Contract Tests for Gen-Zero Decision Engine.

Validates complete closure of Round 4 review findings:
3. Model weight export completeness (val_mlp_0/1, score_mlp_0) & CPU scorer bias mapping
5. Arbiter backend_reachable propagation and negative/NaN probability sanitization
6. Uncompensated transaction preservation in sandbox
7. Self-healing fallback transaction preservation

Training-side guarantees (distiller, replay, RSI daemon, promotion) moved to gen-zero-research
with the code they tested.
"""

import unittest
import math
import numpy as np
import copy

from gen_zero.daemon.atomic_container import AtomicModelContainer
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
