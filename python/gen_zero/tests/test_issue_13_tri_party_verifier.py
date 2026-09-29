"""Comprehensive Test Suite for Issue #13:
Tri-Party Continuous Verifier Harness, In-Loop Self-Healing, Tone Invariance,
Session Compaction, and RSI Flywheel Benchmark.
"""

import math
import unittest
from typing import Any, Dict, List

from gen_zero.client import GenZero
from gen_zero.harness.context_compactor import (
    CompactionResult,
    SessionContextCompactor,
)
from gen_zero.harness.evidence_sanitizer import (
    ObjectiveEvidence,
    ObjectiveEvidenceSanitizer,
)
from gen_zero.harness.goal_verifier import (
    GoalVerificationVerdict,
    GoalVerifier,
    VerifierStatus,
)
from gen_zero.harness.rsi_benchmark import (
    ContinuousVerificationMetrics,
    TriPartyVerificationBenchmark,
)
from gen_zero.harness.self_healing import (
    InLoopSelfHealingTrigger,
    SelfHealingTicket,
)
from gen_zero.harness.tri_party_harness import (
    TriPartyHarness,
    TriPartyStepResult,
)


class TestMilestone1GoalVerifier(unittest.TestCase):
    """Milestone 1: Goal Verifier Protocol & Multi-Criteria Verification."""

    def setUp(self):
        self.client = GenZero()
        self.verifier = GoalVerifier(client=self.client)

    def test_01_verify_step_all_tests_passed(self):
        goal = "Implement fast token encoder and pass regression tests"
        criteria = [
            "All regression tests must pass 100%",
            "Exit code must be 0"
        ]
        evidence = "Ran 24 tests in 0.45s\n\nOK"
        verdict = self.verifier.verify_step(goal, criteria, evidence, explicit_exit_code=0)

        self.assertEqual(verdict.status, VerifierStatus.GOAL_MET.value)
        self.assertTrue(verdict.is_goal_met)
        self.assertGreaterEqual(verdict.confidence, 0.70)
        self.assertEqual(verdict.failed_criteria, [])
        self.assertGreater(verdict.timing_ms, 0.0)

    def test_02_verify_step_tests_failed_hard_invariant(self):
        goal = "Refactor database pool"
        criteria = [
            "All regression tests must pass 100%",
            "Zero memory leaks"
        ]
        evidence = "Ran 20 tests in 0.52s\n\nFAILED (failures=2, errors=0)"
        verdict = self.verifier.verify_step(goal, criteria, evidence, explicit_exit_code=1)

        self.assertEqual(verdict.status, VerifierStatus.NOT_YET.value)
        self.assertFalse(verdict.is_goal_met)
        self.assertEqual(verdict.details["All regression tests must pass 100%"], 0.0)
        self.assertIn("All regression tests must pass 100%", verdict.failed_criteria)

    def test_03_verify_step_empty_criteria(self):
        verdict = self.verifier.verify_step(
            goal="Simple ping",
            criteria=[],
            evidence="ping ok"
        )
        self.assertEqual(verdict.status, VerifierStatus.GOAL_MET.value)
        self.assertEqual(verdict.confidence, 1.0)


class TestMilestone2InLoopSelfHealing(unittest.TestCase):
    """Milestone 2: In-Loop Self-Healing Trigger & Narrow Replanning Assembler."""

    def test_01_execution_success_no_healing(self):
        needs_healing, ticket = InLoopSelfHealingTrigger.analyze_execution(
            command="python3 -m unittest test_auth.py",
            exit_code=0,
            stdout="Ran 5 tests in 0.1s\n\nOK",
            stderr=""
        )
        self.assertFalse(needs_healing)
        self.assertIsNone(ticket)

    def test_02_execution_failure_with_traceback(self):
        mock_stderr = (
            'Traceback (most recent call last):\n'
            '  File "gen_zero/auth/token.py", line 42, in verify_token\n'
            '    assert token.is_valid(), "Token has expired"\n'
            'AssertionError: Token has expired\n'
        )
        needs_healing, ticket = InLoopSelfHealingTrigger.analyze_execution(
            command="pytest gen_zero/tests/test_auth.py",
            exit_code=1,
            stdout="FAILED gen_zero/tests/test_auth.py::test_verify",
            stderr=mock_stderr
        )
        self.assertTrue(needs_healing)
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket.error_type, "AssertionError")
        self.assertEqual(ticket.target_file, "gen_zero/auth/token.py")
        self.assertEqual(ticket.line_number, 42)
        self.assertIn("gen_zero/auth/token.py", ticket.narrow_instruction)

        prompt = ticket.format_orchestrator_prompt()
        self.assertIn("IN-LOOP SELF-HEALING TICKET", prompt)
        self.assertIn("gen_zero/auth/token.py:42", prompt)
        self.assertIn("Constraint: Focus strictly on repairing this localized error", prompt)


class TestMilestone3ToneInvarianceAndSanitization(unittest.TestCase):
    """Milestone 3: Objective Evidence Sanitizer & Tone Invariant Anchoring."""

    def test_01_strip_subjective_noise(self):
        panicky_text = "Oh god this is a terrible disaster!! Please please fix it honestly????"
        cleaned = ObjectiveEvidenceSanitizer.strip_subjective_noise(panicky_text)
        self.assertNotIn("disaster", cleaned.lower())
        self.assertNotIn("please please", cleaned.lower())
        self.assertNotIn("!!", cleaned)

    def test_02_extract_objective_evidence_unittest(self):
        raw_output = (
            "I hope this works finally...\n"
            "Ran 35 tests in 1.42s\n\n"
            "OK (skipped=2)\n"
            "ExitCode: 0"
        )
        evidence = ObjectiveEvidenceSanitizer.extract_objective_evidence(raw_output)
        self.assertEqual(evidence.total_tests, 35)
        self.assertEqual(evidence.passed_tests, 33)
        self.assertEqual(evidence.skipped_tests, 2)
        self.assertEqual(evidence.failed_tests, 0)
        self.assertEqual(evidence.exit_code, 0)
        self.assertTrue(evidence.is_success)

    def test_03_tone_invariance_verification(self):
        """Identical underlying facts with extreme emotional variations yield 100% invariant verdicts."""
        client = GenZero()
        verifier = GoalVerifier(client=client)

        base_fact = "Ran 16 tests in 0.3s\n\nOK\nExitCode: 0"
        variants = [
            f"URGENT DISASTER CRAP CODE!!\n{base_fact}",
            f"Genius magnificent phenomenal masterpiece!\n{base_fact}",
            f"To be honest I think maybe it might pass cross fingers.\n{base_fact}",
            base_fact
        ]

        verdicts = [
            verifier.verify_step(
                goal="Pass all 16 tests",
                criteria=["All regression tests must pass 100%"],
                evidence=v
            ).status
            for v in variants
        ]

        # 100% invariant
        self.assertEqual(len(set(verdicts)), 1)
        self.assertEqual(verdicts[0], VerifierStatus.GOAL_MET.value)


class TestMilestone4ContextCompactor(unittest.TestCase):
    """Milestone 4: Session Pruning & Context Compaction Manager."""

    def test_01_compaction_on_long_session_achieves_target_ratio(self):
        compactor = SessionContextCompactor(keep_recent_turns=3)

        history = [
            {"role": "system", "content": "You are a specialized coding agent."}
        ]
        for i in range(12):
            history.append({"role": "assistant", "content": f"Step {i}: executing database migration script"})
            # Massive tool output (1200+ characters)
            big_output = (
                f"Migration log for batch {i}:\n" +
                "Applying schema update: 001_create_user_table ... OK\n" * 20 +
                "Ran 25 tests in 0.5s\n\nOK\nExitCode: 0"
            )
            history.append({"role": "tool", "content": big_output})

        res = compactor.compact_history(history)
        self.assertIsInstance(res, CompactionResult)

        # Milestone requirement: >= 60% compression ratio
        self.assertGreaterEqual(res.compression_ratio, 0.60)
        self.assertGreater(res.compacted_turns_count, 0)

        # Invariant checks:
        # 1. System prompt preserved
        self.assertEqual(res.compacted_history[0]["role"], "system")
        self.assertEqual(res.compacted_history[0]["content"], "You are a specialized coding agent.")

        # 2. Recent turns preserved without compaction
        recent_turns = res.compacted_history[-3:]
        for turn in recent_turns:
            self.assertFalse(turn.get("_was_compacted", False))


class TestMilestone5TriPartyHarnessAndRSIN(unittest.TestCase):
    """Milestone 5: Tri-Party Continuous Verifier Harness & RSI Flywheel Benchmark."""

    def setUp(self):
        self.client = GenZero()

    def test_01_tri_party_harness_step_and_false_completion_interception(self):
        harness = TriPartyHarness(client=self.client)

        # Step 1: Normal command execution
        step1 = harness.step(
            action="git status",
            executor_fn=lambda: (0, "On branch main\nnothing to commit", ""),
            current_goal="Verify clean working directory",
            criteria=["Working directory must be clean"],
            agent_claims_completed=False
        )
        self.assertEqual(step1.exit_code, 0)
        self.assertFalse(step1.is_healed)
        self.assertFalse(step1.false_completion_intercepted)

        # Step 2: Agent attempts false completion while tests are failing
        failing_log = "Ran 10 tests in 0.2s\n\nFAILED (failures=3)"
        step2 = harness.step(
            action="pytest tests/",
            executor_fn=lambda: (1, failing_log, ""),
            current_goal="All tests must pass",
            criteria=["All regression tests must pass 100%"],
            agent_claims_completed=True  # Asymmetric deception attempt
        )
        self.assertEqual(step2.exit_code, 1)
        self.assertTrue(step2.is_healed)
        self.assertTrue(step2.false_completion_intercepted)
        self.assertIn("COMPLETION REJECTED BY VERIFIER", step2.narrow_feedback)
        self.assertEqual(len(harness.intercepted_false_completions), 1)

    def test_02_tri_party_verification_benchmark_suite(self):
        bench = TriPartyVerificationBenchmark(client=self.client)
        metrics = bench.run_benchmark_suite()

        self.assertIsInstance(metrics, ContinuousVerificationMetrics)
        self.assertEqual(metrics.false_completion_interception_rate, 1.0)
        self.assertGreaterEqual(metrics.self_healing_recovery_rate, 1.0)
        self.assertGreaterEqual(metrics.avg_context_compression_ratio, 0.60)
        self.assertEqual(metrics.tone_invariance_consistency, 1.0)
        self.assertGreater(metrics.mined_counterfactual_samples, 0)

        report = metrics.to_dict()
        self.assertIn("success_rate", report)
        self.assertIn("false_completion_interception_rate", report)
        self.assertIn("avg_context_compression_ratio", report)


if __name__ == "__main__":
    unittest.main()
