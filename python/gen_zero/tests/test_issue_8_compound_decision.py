"""Comprehensive Test Suite for Issue #8:
Composite Decision Bundling, Dual-Track Policy Gate, Stagnation Breaker,
Candidate Governance, and Honest Telemetry.
"""

import unittest
from typing import Any, Dict, List, Optional

from gen_zero.runtime.composite_decision import (
    CompositeStepDecision,
    CompositeDecisionEngine,
    DEFAULT_ACTION_TYPES
)
from gen_zero.gate.policy_gate import (
    DecisionPolicyGate,
    PolicyGateVerdict,
    PolicyVerdictAction,
    DomainRiskProfile,
    RiskLevel
)
from gen_zero.runtime.loop_state_machine import (
    StagnationCircuitBreaker,
    CompositeExecutionLoop,
    LoopExecutionStatus,
    compute_perceptual_fingerprint
)
from gen_zero.runtime.candidate_governance import (
    AntiTruncationRanker,
    GovernedCandidatePool
)
from gen_zero.evaluate.snapshot_benchmark import (
    SnapshotItem,
    HonestTelemetryTracker,
    FrozenSnapshotBenchmark
)
from gen_zero.client import GenZero


class TestCompositeDecisionBundling(unittest.TestCase):
    """Milestone 1: Step Decision 4-Tuple Protocol."""

    def test_01_engine_decide_step_structure(self):
        engine = CompositeDecisionEngine()
        affordances = ["#btn-submit", "#btn-cancel", "#input-search"]
        decision = engine.decide_step(
            state={"page": "checkout", "total": 50},
            affordances=affordances,
            goal="Submit order"
        )
        self.assertIn(decision.target, affordances)
        self.assertIn(decision.action, DEFAULT_ACTION_TYPES)
        self.assertGreaterEqual(decision.target_confidence, 0.0)
        self.assertLessEqual(decision.target_confidence, 1.0)
        self.assertGreaterEqual(decision.done_prob, 0.0)
        self.assertGreaterEqual(decision.risk_prob, 0.0)
        self.assertGreater(decision.timing_ms, 0.0)

    def test_02_empty_affordances_safe_finish(self):
        engine = CompositeDecisionEngine()
        decision = engine.decide_step(
            state="no elements available",
            affordances=[]
        )
        self.assertIsNone(decision.target)
        self.assertEqual(decision.action, "finish")
        self.assertEqual(decision.done_prob, 1.0)

    def test_03_client_decide_step_integration(self):
        client = GenZero()
        res = client.decide_step(
            state="Order summary page",
            affordances=["#confirm-btn", "#back-btn"],
            goal="Confirm purchase",
            apply_policy_gate=True
        )
        self.assertIn("target", res)
        self.assertIn("action", res)
        self.assertIn("policy_gate", res)
        self.assertIn("candidate_pool", res)
        self.assertEqual(res["candidate_pool"]["total_candidates"], 2)


class TestDualTrackPolicyGate(unittest.TestCase):
    """Milestone 2: Dual-Track Policy Gate & Tiered Escalation."""

    def setUp(self):
        self.gate = DecisionPolicyGate()

    def test_01_destructive_action_requires_stop(self):
        decision = CompositeStepDecision(
            target="table_users",
            target_confidence=0.95,
            action="drop",
            action_confidence=0.95,
            done_prob=0.0,
            risk_prob=0.01  # Even with low self-reported risk, hard rule catches it!
        )
        verdict = self.gate.evaluate_policy(decision, state="Postgres DB shell")
        self.assertEqual(verdict.action, PolicyVerdictAction.STOP)
        self.assertFalse(verdict.requires_confirmation)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("DESTRUCTIVE_OPERATION" in r for r in verdict.triggered_rules))

    def test_02_high_risk_requires_stop(self):
        decision = CompositeStepDecision(
            target="#btn-update-config",
            target_confidence=0.85,
            action="click",
            action_confidence=0.90,
            done_prob=0.0,
            risk_prob=0.35  # Exceeds default tau_risk = 0.20
        )
        verdict = self.gate.evaluate_policy(decision, state="Settings page")
        self.assertEqual(verdict.action, PolicyVerdictAction.STOP)
        self.assertFalse(verdict.requires_confirmation)
        self.assertTrue(any("SOFT_RISK_BREACH" in r for r in verdict.triggered_rules))

    def test_03_confidence_three_tiers(self):
        # Tier 1: Blind Guess (< 0.30) -> STOP
        blind_decision = CompositeStepDecision(
            target="button_a",
            target_confidence=0.25,
            action="click",
            action_confidence=0.80,
            done_prob=0.0,
            risk_prob=0.01
        )
        v_stop = self.gate.evaluate_policy(blind_decision, state={})
        self.assertEqual(v_stop.action, PolicyVerdictAction.STOP)
        self.assertFalse(v_stop.passed)

        # Tier 2: Ambiguity (0.30 <= conf < 0.50) -> ESCALATE
        ambiguous_decision = CompositeStepDecision(
            target="button_b",
            target_confidence=0.42,
            action="click",
            action_confidence=0.80,
            done_prob=0.0,
            risk_prob=0.01
        )
        v_escalate = self.gate.evaluate_policy(ambiguous_decision, state={})
        self.assertEqual(v_escalate.action, PolicyVerdictAction.ESCALATE)
        self.assertFalse(v_escalate.passed)

        # Tier 3: Certainty (>= 0.50) with no risk -> PROCEED
        safe_decision = CompositeStepDecision(
            target="button_c",
            target_confidence=0.88,
            action="click",
            action_confidence=0.92,
            done_prob=0.0,
            risk_prob=0.02
        )
        v_proceed = self.gate.evaluate_policy(safe_decision, state={})
        self.assertEqual(v_proceed.action, PolicyVerdictAction.PROCEED)
        self.assertTrue(v_proceed.passed)

    def test_04_domain_risk_profiles(self):
        # Read-only profile allows 0.42 confidence to proceed
        ro_prof = DomainRiskProfile.read_only()
        decision = CompositeStepDecision(
            target="search_result_1",
            target_confidence=0.45,
            action="click",
            action_confidence=0.90,
            done_prob=0.0,
            risk_prob=0.01
        )
        v_ro = self.gate.evaluate_policy(decision, state={}, profile=ro_prof)
        self.assertEqual(v_ro.action, PolicyVerdictAction.PROCEED)

        # Critical profile stops / escalates 0.45 confidence
        crit_prof = DomainRiskProfile.critical()
        v_crit = self.gate.evaluate_policy(decision, state={}, profile=crit_prof)
        self.assertEqual(v_crit.action, PolicyVerdictAction.ESCALATE)


class TestLoopStateMachineAndBreakers(unittest.TestCase):
    """Milestone 3: Loop State Machine, Stagnation Breaker & Dry-Run."""

    def test_01_stagnation_breaker_trips_on_nochange(self):
        breaker = StagnationCircuitBreaker(max_consecutive_stagnations=2)
        state_static = {"url": "https://site.test", "text": "Unchanged screen"}

        # Step 1: initial action
        ok1 = breaker.record_step("click:#broken-btn", state_static)
        self.assertTrue(ok1)
        self.assertFalse(breaker.is_tripped)

        # Step 2: same action, same state fingerprint -> 1 stagnation
        ok2 = breaker.record_step("click:#broken-btn", state_static)
        self.assertTrue(ok2)
        self.assertFalse(breaker.is_tripped)

        # Step 3: same action, same state fingerprint -> 2nd stagnation -> TRIPS!
        ok3 = breaker.record_step("click:#broken-btn", state_static)
        self.assertFalse(ok3)
        self.assertTrue(breaker.is_tripped)
        self.assertIn("STATE_STAGNATION_DETECTED", breaker.trip_reason)

    def test_02_asymmetric_oracle_verification_catches_hallucination(self):
        # Explicit mock engine where model hallucinates high confidence done (P(done) = 0.98)
        mock_engine = CompositeDecisionEngine()
        mock_engine.decide_step = lambda **kwargs: CompositeStepDecision(
            target="#finish-btn",
            target_confidence=0.95,
            action="finish",
            action_confidence=0.95,
            done_prob=0.98,
            risk_prob=0.01
        )
        loop = CompositeExecutionLoop(decision_engine=mock_engine, max_steps=5)

        # Oracle returns False (task is actually NOT done!)
        oracle_called = False
        def mock_oracle(s):
            nonlocal oracle_called
            oracle_called = True
            return False

        report = loop.execute_loop(
            initial_state={"step": 1},
            get_affordances_fn=lambda s: ["#finish-btn"],
            actuate_fn=lambda a, t: {"step": 2},
            verify_fn=mock_oracle,
            goal="Complete all forms"
        )
        self.assertTrue(oracle_called)
        # Must NOT be SUCCESS! Must be ESCALATED due to asymmetric trust breach
        self.assertEqual(report.status, LoopExecutionStatus.ESCALATED)
        self.assertIn("ASYMMETRIC_TRUST_BREACH", report.verdict_reason)

    def test_03_native_dry_run_contract(self):
        loop = CompositeExecutionLoop(max_steps=5)
        actuation_called = False
        def mock_actuate(a, t):
            nonlocal actuation_called
            actuation_called = True
            return {}

        report = loop.execute_loop(
            initial_state={"page": "dashboard"},
            get_affordances_fn=lambda s: ["#btn-save"],
            actuate_fn=mock_actuate,
            dry_run=True
        )
        self.assertEqual(report.status, LoopExecutionStatus.DRY_RUN)
        self.assertFalse(actuation_called)  # Actuator never touched!
        self.assertEqual(len(report.history), 1)
        self.assertTrue(report.history[0].dry_run)


class TestCandidateGovernance(unittest.TestCase):
    """Milestone 4: Candidate Governance & Anti-Truncation Ranking."""

    def test_01_anti_truncation_protects_bottom_action_buttons(self):
        ranker = AntiTruncationRanker(max_candidates=5)

        # 10 candidates where the critical '#btn-submit-order' is at index 9 (very bottom)
        candidates = [
            f"#unimportant-link-{i}" for i in range(9)
        ] + ["#btn-submit-order"]

        pool = ranker.govern_candidates(
            raw_candidates=candidates,
            goal="Submit my current purchase"
        )
        self.assertTrue(pool.clipped)
        self.assertEqual(pool.total_candidates, 10)
        self.assertEqual(pool.retained_count, 5)
        # Naive slicing [:5] would drop #btn-submit-order. AntiTruncation MUST keep it!
        self.assertIn("#btn-submit-order", pool.candidates)

    def test_02_state_feedback_decoupling(self):
        ranker = AntiTruncationRanker(max_candidates=10)
        feedback = {"display_reading": "42.0", "focused_id": "#input-val"}
        pool = ranker.govern_candidates(
            raw_candidates=["#btn-1", "#btn-2"],
            state_feedback=feedback
        )
        self.assertFalse(pool.clipped)
        self.assertEqual(pool.state_feedback["display_reading"], "42.0")
        self.assertNotIn("display_reading", pool.candidates)


class TestSnapshotBenchmarkAndHonestTelemetry(unittest.TestCase):
    """Milestone 5: Frozen Snapshot Benchmark & Honest Telemetry."""

    def test_01_snapshot_benchmark_execution(self):
        snapshots = [
            SnapshotItem(
                id="snap_01",
                state="Shopping cart page with item $10",
                affordances=["#btn-checkout", "#btn-cancel"],
                goal="Proceed to checkout",
                expected_target="#btn-checkout",
                expected_action="click"
            ),
            SnapshotItem(
                id="snap_02",
                state="Search box page",
                affordances=["#input-query", "#btn-help"],
                goal="Search for flight",
                expected_target="#input-query",
                expected_action="input_text"
            )
        ]
        benchmark = FrozenSnapshotBenchmark(snapshots)
        report = benchmark.run_benchmark()

        self.assertEqual(report.total_snapshots, 2)
        self.assertGreaterEqual(report.target_accuracy, 0.0)
        self.assertIn("ece_10bin", report.calibration)
        self.assertIn("autonomous_rate", report.telemetry)
        self.assertGreater(report.latency_p50_ms, 0.0)

    def test_02_honest_telemetry_never_counts_takeover_as_autonomous(self):
        tracker = HonestTelemetryTracker()

        # Step 1: Autonomous model decision
        v_proc = PolicyGateVerdict(
            action=PolicyVerdictAction.PROCEED,
            passed=True,
            requires_confirmation=False,
            confidence=0.9,
            risk_score=0.0
        )
        tracker.record_step(v_proc, was_bypassed_by_planner=False)

        # Step 2: System 2 planner bypass
        tracker.record_step(v_proc, was_bypassed_by_planner=True)

        # Step 3: Human confirmation required
        v_conf = PolicyGateVerdict(
            action=PolicyVerdictAction.CONFIRM,
            passed=False,
            requires_confirmation=True,
            confidence=0.8,
            risk_score=0.5
        )
        tracker.record_step(v_conf)

        # Step 4: Supervisor escalation
        v_esc = PolicyGateVerdict(
            action=PolicyVerdictAction.ESCALATE,
            passed=False,
            requires_confirmation=False,
            confidence=0.4,
            risk_score=0.1
        )
        tracker.record_step(v_esc)

        summary = tracker.get_summary()
        self.assertEqual(summary.total_steps, 4)
        self.assertEqual(summary.autonomous_decisions, 1)  # Only step 1!
        self.assertEqual(summary.bypassed_steps, 1)
        self.assertEqual(summary.confirmed_steps, 1)
        self.assertEqual(summary.escalated_steps, 1)
        self.assertEqual(summary.autonomous_rate, 0.25)
        self.assertEqual(summary.takeover_rate, 0.75)


class TestReviewerEdgeCases(unittest.TestCase):
    """Specific regression tests for Reviewer 1 & 2 edge cases."""

    def test_01_nan_inf_sanitization_stops_corrupted_decisions(self):
        gate = DecisionPolicyGate()

        # NaN confidence must NOT fall through to PROCEED; must trigger STOP
        nan_decision = CompositeStepDecision(
            target="#btn-ok",
            target_confidence=float("nan"),
            action="click",
            action_confidence=float("nan"),
            done_prob=0.0,
            risk_prob=0.0
        )
        v_nan = gate.evaluate_policy(nan_decision, state={})
        self.assertEqual(v_nan.action, PolicyVerdictAction.STOP)
        self.assertFalse(v_nan.passed)

        # NaN risk must be rejected before policy arbitration.
        nan_risk_decision = CompositeStepDecision(
            target="#btn-ok",
            target_confidence=0.9,
            action="click",
            action_confidence=0.9,
            done_prob=0.0,
            risk_prob=float("nan")
        )
        v_risk = gate.evaluate_policy(nan_risk_decision, state={})
        self.assertEqual(v_risk.action, PolicyVerdictAction.STOP)
        self.assertFalse(v_risk.requires_confirmation)

    def test_02_scoped_regex_avoids_context_false_positives(self):
        gate = DecisionPolicyGate()
        # Context mentions 'payment' or 'delete', but the prospective action is harmless FAQ navigation
        state = "User ticket: Why did my payment fail and delete my cart?"
        decision = CompositeStepDecision(
            target="#help-faq",
            target_confidence=0.92,
            action="click",
            action_confidence=0.95,
            done_prob=0.0,
            risk_prob=0.01
        )
        verdict = gate.evaluate_policy(decision, state=state)
        # Should PROCEED because prospective action 'click #help-faq' contains no sensitive pattern
        self.assertEqual(verdict.action, PolicyVerdictAction.PROCEED)
        self.assertTrue(verdict.passed)

    def test_03_whitelist_still_obeys_confidence_stop(self):
        prof = DomainRiskProfile(
            whitelisted_targets={"#btn-pay"},
            sensitive_patterns=[r"\bpay\b"]
        )
        gate = DecisionPolicyGate()
        # Whitelisted target with blind confidence 0.15 must still STOP
        decision = CompositeStepDecision(
            target="#btn-pay",
            target_confidence=0.15,
            action="click",
            action_confidence=0.15,
            done_prob=0.0,
            risk_prob=0.01
        )
        verdict = gate.evaluate_policy(decision, state={}, profile=prof)
        self.assertEqual(verdict.action, PolicyVerdictAction.STOP)
        self.assertFalse(verdict.passed)

    def test_04_candidate_role_word_boundary_no_display_boost(self):
        ranker = AntiTruncationRanker(max_candidates=5)
        # '#display-panel' should NOT trigger 'pay' boost (+2.5)
        score_display = ranker._score_candidate("#display-panel", set())
        score_pay = ranker._score_candidate("#btn-pay-now", set())
        self.assertEqual(score_display, 1.0)
        self.assertGreater(score_pay, score_display)


if __name__ == "__main__":
    unittest.main()
