"""Unit tests for Issue #76: Automated Natural Language / AST Constraint Linear Projection Compiler."""

import unittest
import numpy as np

from gen_zero.gate.constraint_compiler import (
    ConstraintLinearProjectionCompiler,
    CompiledConstraintRule,
    CompilationReport,
    ASTConstraintParser,
    Tristate,
)
from gen_zero.nanocore.world_model_orchestrator import WorldModelNanoCoreOrchestrator
from gen_zero.client import GenZeroClient


class TestConstraintLinearProjectionCompiler(unittest.TestCase):
    """Verifies natural language / AST constraint compilation, W_sat projection, and CP-SAT blocking."""

    def setUp(self):
        self.latent_dim = 128
        self.compiler = ConstraintLinearProjectionCompiler(
            latent_dim=self.latent_dim,
            seed=42,
            # Semantic tests have a generous budget; the benchmark separately enforces 2ms.
            hard_timeout_ms=100.0,
        )

    def test_ast_and_nl_rule_parsing(self):
        """Verify parsing of declarative natural language and AST comparison expressions."""
        rules = [
            "FORBID RESTART IF memory_usage > 0.90",
            "ALLOW SCALE_UP ONLY IF cpu_load >= 0.75",
            "REQUIRE HOLD WHEN error_rate > 0.10",
            "disk_full > 0.98 -> BAN WRITE_DATA",
            "MUTUAL_EXCLUSIVE REBOOT SHUTDOWN",
        ]

        report = self.compiler.compile_rules(rules)
        self.assertEqual(report.total_rules_compiled, 5)
        self.assertLessEqual(report.compilation_time_ms, 50.0)
        self.assertTrue(report.is_orthonormal)

        # Check rule 1
        r1 = report.rules[0]
        self.assertEqual(r1.rule_type, "FORBID_IF")
        self.assertEqual(r1.target_action, "RESTART")
        self.assertEqual(r1.variable_name, "memory_usage")
        self.assertEqual(r1.comparator, ">")
        self.assertAlmostEqual(r1.threshold, 0.90)

        # Check rule 2
        r2 = report.rules[1]
        self.assertEqual(r2.rule_type, "ALLOW_ONLY_IF")
        self.assertEqual(r2.target_action, "SCALE_UP")
        self.assertEqual(r2.comparator, ">=")
        self.assertAlmostEqual(r2.threshold, 0.75)

    def test_orthonormal_projection_matrix_properties(self):
        """Verify that projection matrix W_sat is strictly orthonormal."""
        rules = [
            "FORBID FLUSH IF buffer_size < 10.0",
            "FORBID DRAIN IF queue_depth >= 50.0",
            "FORBID OFFLINE IF active_connections > 0.0",
        ]
        report = self.compiler.compile_rules(rules)
        w_sat = self.compiler.projection_matrix
        self.assertIsNotNone(w_sat)
        self.assertEqual(w_sat.shape, (report.num_propositions, self.latent_dim))

        # Check orthonormality: W_sat * W_sat.T == Eye
        gram = np.dot(w_sat, w_sat.T)
        np.testing.assert_allclose(gram, np.eye(report.num_propositions), atol=1e-4)

    def test_latent_projection_to_boolean_propositions(self):
        """Verify continuous latent vector projection to discrete boolean propositions."""
        rules = [
            "FORBID PURGE IF memory_ratio > 0.85",
            "FORBID COMPACT IF io_wait >= 0.40",
        ]
        self.compiler.compile_rules(rules)

        # Evaluate deterministic projection
        rng = np.random.RandomState(123)
        z = rng.randn(self.latent_dim).astype(np.float32)
        prop_map = self.compiler.project_latent_propositions(z, schema_fingerprint=self.compiler.schema_fingerprint)
        self.assertEqual(len(prop_map), 2)
        for val in prop_map.values():
            self.assertIsInstance(val, Tristate)
            self.assertIn(val, (Tristate.TRUE, Tristate.FALSE))

    def test_cpsat_neuro_symbolic_hard_blocking(self):
        """Verify that CP-SAT solver strictly vetoes high-utility neural proposals that violate rules."""
        rules = [
            "FORBID RESTART IF memory_ratio > 0.90",
        ]
        self.compiler.compile_rules(rules)

        # Neural proposal: RESTART is favored with 0.92 utility, WAIT has 0.10, SAFE_SCALE has 0.50
        candidate_utilities = {
            "RESTART": 0.92,
            "SAFE_SCALE": 0.50,
            "WAIT": 0.10,
        }

        # Scenario A: memory_ratio = 0.50 (SAFE -> RESTART is allowed)
        verdict_safe = self.compiler.solve_safest_action(
            candidate_utilities=candidate_utilities,
            current_metrics={"memory_ratio": 0.50},
        )
        self.assertTrue(verdict_safe.is_safe)
        self.assertEqual(verdict_safe.selected_action, "RESTART")

        # Scenario B: memory_ratio = 0.95 (HAZARD -> RESTART MUST be hard blocked!)
        verdict_blocked = self.compiler.solve_safest_action(
            candidate_utilities=candidate_utilities,
            current_metrics={"memory_ratio": 0.95},
        )
        self.assertTrue(verdict_blocked.is_safe)
        self.assertEqual(verdict_blocked.selected_action, "SAFE_SCALE")  # Second highest safe action
        self.assertIn("RESTART", verdict_blocked.applied_constraints[0])

    def test_dynamic_runtime_update_zero_fine_tuning(self):
        """Verify that rules can be updated dynamically at runtime without model retraining."""
        # Initial rule
        self.compiler.compile_rules(["FORBID ACT_A IF error_rate > 0.05"])
        candidate_utilities = {"ACT_A": 0.8, "ACT_B": 0.6}

        # Initially ACT_B is not forbidden
        v1 = self.compiler.solve_safest_action(
            candidate_utilities, current_metrics={"error_rate": 0.01, "latency": 150}
        )
        self.assertEqual(v1.selected_action, "ACT_A")

        # Dynamically append rule banning ACT_A under latency > 100
        self.compiler.compile_rules([
            "FORBID ACT_A IF error_rate > 0.05",
            "FORBID ACT_A IF latency > 100",
        ])
        v2 = self.compiler.solve_safest_action(
            candidate_utilities, current_metrics={"error_rate": 0.01, "latency": 150}
        )
        # Immediately blocked without any retraining
        self.assertEqual(v2.selected_action, "ACT_B")

    def test_orchestrator_integration(self):
        """Verify integration with WorldModelNanoCoreOrchestrator."""
        orchestrator = WorldModelNanoCoreOrchestrator(
            latent_dim=self.latent_dim,
            action_dim=4,
        )

        # Compile safety rule: FORBID HAZARD_ACT IF hazard_flag > 0.5
        report = orchestrator.compile_safety_rules([
            "FORBID ACT_0 IF hazard_flag > 0.5",
        ])
        self.assertEqual(report.total_rules_compiled, 1)

        # Run step with candidate actions
        rng = np.random.RandomState(456)
        z = rng.randn(self.latent_dim).astype(np.float32)
        # ACT_1 fails the safety NanoCore; ACT_2 is caller-forbidden. The compiled-rule
        # solver only knows its own rules, so both must be filtered out before it runs.
        res = orchestrator.step(
            state=z,
            candidate_actions=["ACT_1", "ACT_2"],
            enforce_cpsat=True,
            forbidden_actions={"ACT_2"},
            safety_evaluator=lambda z_, act: 0.1 if act == "ACT_1" else 0.9,
        )
        self.assertEqual(res.selected_action, "HOLD")
        # Every candidate was blocked, so the decision is interlocked. The compiled rules were
        # still evaluated against HOLD itself (single-candidate solve), never NO_CANDIDATES.
        self.assertEqual(res.decision_status, "SAFETY_INTERLOCKED")
        self.assertIn("ALL_CANDIDATES_BLOCKED", res.degradations)
        self.assertIn(res.nanocore_status.cpsat_status, ("ORTOOLS_UNAVAILABLE_FALLBACK", "CP_SAT_OPTIMAL", "CP_SAT_DEADLINE_EXCEEDED", "CPSAT_NO_OPTIMUM:UNKNOWN"))
        # A fallback-only solve is never "verified", even when OR-Tools solved it.
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIn(f"CPSAT_NOT_VERIFIED:{res.nanocore_status.cpsat_status}", res.degradations)

    def test_orchestrator_rule_forbidding_fallback_raises_interlock(self):
        """A compiled rule that forbids HOLD must block the fallback too, not release it."""
        from gen_zero.nanocore.world_model_orchestrator import SafetyInterlockError
        orchestrator = WorldModelNanoCoreOrchestrator(latent_dim=self.latent_dim, action_dim=4)
        report = orchestrator.compile_safety_rules(["FORBID HOLD IF hazard_flag > 0.5"])
        self.assertEqual(report.total_rules_compiled, 1)
        # Force the rule active; the latent projection would otherwise decide at random.
        orchestrator.constraint_compiler.evaluate_condition = lambda *a, **k: True
        z = np.random.RandomState(7).randn(self.latent_dim).astype(np.float32)
        with self.assertRaises(SafetyInterlockError):
            orchestrator.step(
                state=z,
                candidate_actions=["ACT_1"],
                enforce_cpsat=True,
                safety_evaluator=lambda z_, act: 0.1,
            )

    def test_client_integration(self):
        """Verify GenZeroClient factory method for constraint compiler."""
        client = GenZeroClient()
        compiler = client.create_constraint_compiler(
            latent_dim=64,
            hard_timeout_ms=1.5,
        )
        self.assertIsInstance(compiler, ConstraintLinearProjectionCompiler)
        self.assertEqual(compiler.latent_dim, 64)

    def test_s08_s10_variable_aligned_projection(self):
        """Verify S08 and S10: Single-variable projection aligns with declared coordinate dimensions and boundary semantics."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=16, seed=42)
        rules = [
            "FORBID EXECUTE IF x > 0",
            "FORBID EXECUTE_TEN IF x > 10",
        ]
        compiler.compile_rules(rules)

        # In state z where x=1.0 at coordinate 0 (z[0] = 1.0)
        z1 = np.zeros(16, dtype=np.float32)
        z1[0] = 1.0
        prop_map1 = compiler.project_latent_propositions(z1, schema_fingerprint=compiler.schema_fingerprint)
        self.assertIs(prop_map1["x > 0"], Tristate.TRUE)  # S08: Direct predicate True == Proj True
        self.assertIs(prop_map1["x > 10"], Tristate.FALSE)

        # In state z where x=11.0 at coordinate 0 (z[0] = 11.0)
        z11 = np.zeros(16, dtype=np.float32)
        z11[0] = 11.0
        prop_map11 = compiler.project_latent_propositions(z11, schema_fingerprint=compiler.schema_fingerprint)
        self.assertIs(prop_map11["x > 0"], Tristate.TRUE)
        self.assertIs(prop_map11["x > 10"], Tristate.TRUE)  # S10: Threshold 10.0 correctly active for x=11

        # Strict open boundary test: x > 0 at zero vector (z=0) MUST be False
        z0 = np.zeros(16, dtype=np.float32)
        prop_map0 = compiler.project_latent_propositions(z0, schema_fingerprint=compiler.schema_fingerprint)
        self.assertIs(prop_map0["x > 0"], Tristate.FALSE)

    def test_s17_complementary_conditions_mutex(self):
        """Verify S17: Complementary conditions (x > 10 vs x <= 10) are mutually exclusive under all latent states."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=16, seed=42)
        rules = [
            "FORBID ACT_A IF x > 10",
            "FORBID ACT_B IF x <= 10",
        ]
        compiler.compile_rules(rules)

        # Test across various points around threshold 10
        test_vals = [-100.0, 0.0, 9.99, 10.0, 10.001, 50.0]
        for val in test_vals:
            z = np.zeros(16, dtype=np.float32)
            z[0] = val
            prop_map = compiler.project_latent_propositions(z, schema_fingerprint=compiler.schema_fingerprint)
            p_gt = prop_map["x > 10"]
            p_le = prop_map["x <= 10"]
            # Both conditions can NEVER be True simultaneously!
            self.assertFalse(p_gt is Tristate.TRUE and p_le is Tristate.TRUE, f"Both True at x={val}")
            # Exactly one must be True
            self.assertTrue(p_gt is Tristate.TRUE or p_le is Tristate.TRUE, f"Neither True at x={val}")

    def test_s12_s13_infinite_and_nan_fail_closed(self):
        """Verify S12 and S13: Non-finite values (-inf, +inf, NaN) are strictly blocked via Fail-Closed."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=16, seed=42)
        compiler.compile_rules([
            "FORBID EXECUTE IF temp > 10",
            "ALLOW PROCEED ONLY IF budget >= 10",
        ])

        # S12: temp = -inf must Fail-Closed and forbid EXECUTE
        v1 = compiler.solve_safest_action(
            candidate_utilities={"EXECUTE": 1.0, "WAIT": 0.5},
            current_metrics={"temp": -float("inf")},
        )
        self.assertNotEqual(v1.selected_action, "EXECUTE")

        # S13: budget = +inf must Fail-Closed and deny PROCEED authorization
        v2 = compiler.solve_safest_action(
            candidate_utilities={"PROCEED": 1.0, "WAIT": 0.5},
            current_metrics={"budget": float("inf")},
        )
        self.assertNotEqual(v2.selected_action, "PROCEED")

        # NaN must also Fail-Closed
        v3 = compiler.solve_safest_action(
            candidate_utilities={"EXECUTE": 1.0, "WAIT": 0.5},
            current_metrics={"temp": float("nan")},
        )
        self.assertNotEqual(v3.selected_action, "EXECUTE")

    def test_s07_conflicting_requirements(self):
        """Verify S07: Multiple conflicting requirements return is_safe=False and CONFLICTING_REQUIREMENTS."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=16, seed=42)
        compiler.compile_rules([
            "REQUIRE ACTION_A WHEN flag > 0",
            "REQUIRE ACTION_B WHEN flag > 0",
        ])
        v = compiler.solve_safest_action(
            candidate_utilities={"ACTION_A": 1.0, "ACTION_B": 0.8},
            current_metrics={"flag": 1.0},
        )
        self.assertFalse(v.is_safe)
        self.assertEqual(v.solver_status, "CONFLICTING_REQUIREMENTS")

    def test_s14_fallback_action_validation(self):
        """Verify S14: When candidates are forbidden and fallback does not satisfy REQUIRE, is_safe=False."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=16, seed=42)
        compiler.compile_rules([
            "REQUIRE STOP WHEN alert > 0",
        ])
        # Candidates only have EXECUTE (which is not STOP)
        v = compiler.solve_safest_action(
            candidate_utilities={"EXECUTE": 1.0},
            current_metrics={"alert": 1.0},
            fallback_safe_action="HOLD",
        )
        # Fallback HOLD does not satisfy REQUIRE STOP
        self.assertFalse(v.is_safe)
        self.assertEqual(v.solver_status, "UNSAFE_NO_FEASIBLE_ACTIONS")

    def test_s15_colon_action_regex_and_unsupported_syntax(self):
        """Verify S15: Colon-namespaced actions are parsed correctly, and unsupported syntax fails closed."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=16, seed=42)
        compiler.compile_rules([
            "FORBID tool:execute IF danger > 0",
        ])
        v = compiler.solve_safest_action(
            candidate_utilities={"tool:execute": 1.0, "safe_read": 0.5},
            current_metrics={"danger": 1.0},
        )
        self.assertEqual(v.selected_action, "safe_read")
        self.assertTrue(v.is_safe)

        # Malformed unsupported rule fails closed
        bad_compiler = ConstraintLinearProjectionCompiler(latent_dim=16, seed=42)
        bad_compiler.compile_rules([
            "THIS IS NOT A VALID RULE SYNTAX",
        ])
        v_bad = bad_compiler.solve_safest_action(
            candidate_utilities={"tool:execute": 1.0},
            fallback_safe_action="HOLD",
        )
        self.assertFalse(v_bad.is_safe)
        self.assertEqual(v_bad.solver_status, "UNSUPPORTED_SYNTAX_FAIL_CLOSED")


if __name__ == "__main__":
    unittest.main()
