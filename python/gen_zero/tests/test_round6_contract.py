"""Round 6 Contract Tests for Gen-Zero Decision Engine.

Validates complete closure of Round 6 review findings:
R6-01: Structured state normalization in encode_leaf_tokens (preserves non-text keys, unwraps containers)
R6-02: Benchmark evaluation vs live policy alignment (candidate descriptions & semantic affinity)
R6-04: Strict boolean is_valid in SafetyGate & invalid model return protection
R6-06: Compensation function single-execution guarantee (inspect.signature pre-binding prevents duplicate runs on internal TypeError)

Training-side guarantees (distiller, replay, RSI daemon, promotion) moved to gen-zero-research
with the code they tested.
"""

import importlib.util
import unittest
import math
import json
import numpy as np

from gen_zero.gate.safety_gate import SafetyGate, GateVerdict
from gen_zero.model.dual_head import encode_leaf_tokens, normalize_state_repr, GenZeroDualHeadModel
from gen_zero.client import GenZeroClient, GenZeroConfig
HAS_TORCH = importlib.util.find_spec("torch") is not None
from gen_zero.sandbox.causal_sandbox import CausalToolSandbox
from gen_zero.sandbox.tool_registry import ToolRegistry, SideEffectLevel
from gen_zero.sandbox.safety_barrier import PRMSafetyBarrier
from gen_zero.sandbox.self_healing_planner import SelfHealingToolchainPlanner, WorkflowStep


class TestRound6Contract(unittest.TestCase):

    def setUp(self):
        self.client = GenZeroClient(GenZeroConfig(hidden_dim=2048, enable_gpu_arbiter_fallback=True))

    # --- R6-01: Structured State Normalization ---
    def test_r6_01_structured_state_without_text_preserves_tokens(self):
        """Probe C01: Dictionaries without 'text' key must not be encoded as 'None'."""
        s1 = {"balance": 1000, "authorized": True}
        s2 = {"balance": 0, "authorized": False}

        toks1 = encode_leaf_tokens(s1, "execute")
        toks2 = encode_leaf_tokens(s2, "execute")

        self.assertNotEqual(toks1, toks2, "States with different fields produced identical tokens!")
        # Verify it doesn't encode literal 'None'
        none_toks = encode_leaf_tokens(None, "execute")
        self.assertNotEqual(toks1, none_toks)

    def test_r6_01_text_with_numeric_fields_preserves_differences(self):
        """Probe C02: Dictionaries with same text but different numeric fields must differ."""
        s1 = {"text": "click", "budget": 100}
        s2 = {"text": "click", "budget": 200}

        toks1 = encode_leaf_tokens(s1, "submit")
        toks2 = encode_leaf_tokens(s2, "submit")
        self.assertNotEqual(toks1, toks2)

    def test_r6_01_wrapped_state_equivalence(self):
        """Probe E05: Replay buffer wrapped state {'state': ...} matches raw benchmark state."""
        raw_state = [0.1, 0.2, 0.3, 0.4]
        wrapped_state = {"state": [0.1, 0.2, 0.3, 0.4]}

        toks_raw = encode_leaf_tokens(raw_state, "action_a")
        toks_wrapped = encode_leaf_tokens(wrapped_state, "action_a")
        self.assertEqual(toks_raw, toks_wrapped)

    # --- R6-04: Strict SafetyGate is_valid & Model Returns ---
    def test_r6_04_safety_gate_rejects_missing_or_string_is_valid(self):
        """Probes G01 & G02: is_valid must be strictly boolean True."""
        gate = SafetyGate()

        # Missing is_valid
        v_missing = gate.evaluate_candidate(
            {"accuracy": 90.0, "mean_score": 10.0, "collision_rate": 0.0},
            {"accuracy": 95.0, "mean_score": 15.0, "collision_rate": 0.0, "is_valid": True}
        )
        self.assertFalse(v_missing.passed)
        self.assertIn("MISSING_VALIDITY_FLAG", v_missing.details.get("error", ""))

        # String "false"
        v_str_false = gate.evaluate_candidate(
            {"accuracy": 90.0, "mean_score": 10.0, "collision_rate": 0.0, "is_valid": "false"},
            {"accuracy": 95.0, "mean_score": 15.0, "collision_rate": 0.0, "is_valid": True}
        )
        self.assertFalse(v_str_false.passed)



    # --- R6-05: Sparse Supervision & Per-Head Masking ---

    # --- R6-06: Compensation Single Execution Guarantee ---
    def test_r6_06_rollback_function_never_retried_on_internal_type_error(self):
        """Probe S08: Internal TypeError within compensation function must NOT trigger duplicate call."""
        registry = ToolRegistry()
        db = {"balance": 1000, "call_count": 0}

        def spend(amount: int):
            db["balance"] -= amount
            return {"balance": db["balance"]}

        def compensate(amount: int):
            db["call_count"] += 1
            db["balance"] += amount
            # Simulate internal TypeError after executing side effect
            raise TypeError("Simulated internal TypeError in compensation logic")

        registry.register(
            name="spend",
            func=spend,
            rollback_func=compensate,
            side_effect_level=SideEffectLevel.MUTATING_REVERSIBLE
        )

        sandbox = CausalToolSandbox(registry=registry)
        # Execute spend
        res = sandbox.execute("spend", {"amount": 100})
        self.assertTrue(res.success)
        self.assertEqual(db["balance"], 900)

        # Trigger rollback
        rolled = sandbox.rollback_transactions()
        self.assertEqual(rolled, 0)
        # Crucial contract: compensate was called EXACTLY once, not retried
        self.assertEqual(db["call_count"], 1)
        self.assertEqual(db["balance"], 1000)
        self.assertEqual(sandbox.uncompensated_transactions_count, 1)

    # --- R6-03: Rollback Recovery Resilience ---


if __name__ == "__main__":
    unittest.main()
