"""Unit Tests for Phase 3: Causal Tool Sandbox, Safety Barrier & Self-Healing Planner."""

import unittest
from gen_zero.client import GenZero
from gen_zero.sandbox.tool_registry import ToolRegistry, SideEffectLevel
from gen_zero.sandbox.causal_sandbox import CausalToolSandbox, ErrorCategory
from gen_zero.sandbox.safety_barrier import PRMSafetyBarrier, SafetyVerdict
from gen_zero.sandbox.self_healing_planner import SelfHealingToolchainPlanner, WorkflowStep


class TestCausalSandboxAndSelfHealing(unittest.TestCase):
    def setUp(self):
        self.engine = GenZero()
        self.registry = self.engine.tool_registry
        self.sandbox = self.engine.sandbox
        self.barrier = self.engine.safety_barrier
        self.planner = self.engine.workflow_planner

    def test_tool_registration_and_validation(self):
        def my_read(user_id: str):
            return {"user_id": user_id, "balance": 100.0}

        self.registry.register(
            name="get_user",
            func=my_read,
            side_effect_level=SideEffectLevel.READ_ONLY,
            required_params=["user_id"]
        )

        # Missing required parameter -> validation fails
        res_fail = self.sandbox.execute("get_user", {})
        self.assertFalse(res_fail.success)
        self.assertEqual(res_fail.error_category, ErrorCategory.ACTION_DECISION_ERROR)

        # Valid call
        res_ok = self.sandbox.execute("get_user", {"user_id": "u123"})
        self.assertTrue(res_ok.success)
        self.assertEqual(res_ok.output["balance"], 100.0)

    def test_exogenous_shock_disentanglement_and_retry(self):
        call_count = 0

        def flaky_api(query: str):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionResetError("503 Service Unavailable: Upstream gateway timeout")
            return {"status": "ok", "result": f"Answer for {query}"}

        self.registry.register(
            name="flaky_service",
            func=flaky_api,
            required_params=["query"]
        )

        res = self.sandbox.execute("flaky_service", {"query": "weather"}, auto_retry_exogenous=True)
        self.assertTrue(res.success)
        self.assertTrue(res.exogenous_noise_detected)
        self.assertEqual(res.retries_attempted, 1)

    def test_transactional_rollback(self):
        db_state = {"account_a": 500.0}

        def debit(account: str, amount: float):
            db_state[account] -= amount
            return {"new_balance": db_state[account]}

        def rollback_debit(params, original_output):
            db_state[params["account"]] += params["amount"]

        self.registry.register(
            name="debit_account",
            func=debit,
            side_effect_level=SideEffectLevel.MUTATING_REVERSIBLE,
            required_params=["account", "amount"],
            rollback_func=rollback_debit
        )

        res = self.sandbox.execute("debit_account", {"account": "account_a", "amount": 100.0})
        self.assertTrue(res.success)
        self.assertEqual(db_state["account_a"], 400.0)
        self.assertEqual(self.sandbox.pending_transactions_count, 1)

        # Rollback
        rolled = self.sandbox.rollback_transactions()
        self.assertEqual(rolled, 1)
        self.assertEqual(db_state["account_a"], 500.0)
        self.assertEqual(self.sandbox.pending_transactions_count, 0)

    def test_prm_safety_barrier_invariants(self):
        def run_sql(query: str):
            return "Executed: " + query

        tool = self.registry.register(
            name="execute_sql",
            func=run_sql,
            side_effect_level=SideEffectLevel.MUTATING_IRREVERSIBLE,
            required_params=["query"]
        )

        # Safe SQL
        audit_ok = self.barrier.audit(tool, {"query": "SELECT * FROM users;"})
        self.assertEqual(audit_ok.verdict, SafetyVerdict.ALLOWED)

        # Destructive SQL -> must be blocked by formal barrier
        audit_blocked = self.barrier.audit(tool, {"query": "DROP TABLE users;"})
        self.assertEqual(audit_blocked.verdict, SafetyVerdict.BLOCKED)
        self.assertEqual(audit_blocked.rule_violated, "NO_DROP_TABLE")

        # Exceeding financial ceiling ($100k)
        def transfer(amount: float):
            return f"Transferred ${amount}"

        t_tool = self.registry.register(
            name="wire_funds",
            func=transfer,
            side_effect_level=SideEffectLevel.MUTATING_IRREVERSIBLE,
            required_params=["amount"]
        )
        audit_wire_blocked = self.barrier.audit(t_tool, {"amount": 250_000.0})
        self.assertEqual(audit_wire_blocked.verdict, SafetyVerdict.BLOCKED)

    def test_self_healing_workflow_planner(self):
        # Scenario: Primary API fails permanently with action logic error,
        # but secondary fallback API succeeds.
        def primary_pay(order_id: str):
            raise ValueError("Invalid merchant terminal ID")

        def secondary_pay(order_id: str):
            return {"status": "paid", "order_id": order_id, "gateway": "secondary"}

        self.registry.register("primary_pay", primary_pay, required_params=["order_id"])
        self.registry.register("secondary_pay", secondary_pay, required_params=["order_id"])

        steps = [
            WorkflowStep(
                step_id="step_1",
                tool_name="primary_pay",
                params={"order_id": "ord_999"},
                fallback_tool_name="secondary_pay"
            )
        ]

        report = self.planner.execute_workflow(steps)
        self.assertTrue(report.success)
        self.assertTrue(report.self_healing_occurred)
        self.assertEqual(report.final_output["gateway"], "secondary")
        self.assertEqual(len(report.reflections), 1)


if __name__ == "__main__":
    unittest.main()
