"""Comprehensive Unit & Integration Tests for Issue #20.

Validates:
1. Milestone 1: FactoryObservation bounded serialization, smart diff truncation (<= 50KB),
   and CoalescedDebouncer with debounce floor and periodic pulse.
2. Milestone 2: 10-Dimensional atomic semantic assessment and deterministic policy state machine.
3. Milestone 3: Two-Tier intervention (Tier 1 soft steer, Tier 2 circuit breaker stop)
   and 30-second steering grace period with recovery vs timeout.
4. Milestone 4: Architecture Contract Drift Gate preventing unauthorized modifications.
"""

import unittest
import time

from gen_zero.runtime.debouncer import CoalescedDebouncer, DebounceEvent
from gen_zero.runtime.supervisor_observation import (
    FactoryObservation,
    smart_truncate_text,
)
from gen_zero.runtime.supervisor_assessment import (
    SupervisorAssessment,
    SemanticAssessmentAdapter,
)
from gen_zero.runtime.supervisor_policy import (
    SupervisorPolicyStateMachine,
    SupervisorAction,
    SupervisorState,
    SteeringGuidance,
)
from gen_zero.runtime.async_supervisor import (
    AsyncSemanticSupervisor,
    SupervisorTelemetry,
)
from gen_zero.gate.contract_drift_gate import (
    ContractDriftGate,
    ContractGateVerdict,
)


class TestSupervisorObservationAndDebouncer(unittest.TestCase):
    """Tests for Milestone 1: Bounded Observation & Coalesced Debouncer."""

    def test_smart_truncate_text_boundary_enforcement(self):
        # 1. Short text remains untruncated
        short_text = "diff --git a/foo.py b/foo.py\n+print('hello')\n"
        self.assertEqual(smart_truncate_text(short_text, max_bytes=1000), short_text)

        # 2. Giant text (> 100KB) is strictly truncated <= 51,200 bytes
        giant_text = "\n".join([f"+ line number {i} with substantial padding text" for i in range(5000)])
        self.assertGreater(len(giant_text.encode("utf-8")), 100000)

        truncated = smart_truncate_text(giant_text, max_bytes=51200)
        self.assertLessEqual(len(truncated.encode("utf-8")), 51200)
        self.assertIn("DIFF TRUNCATED", truncated)
        self.assertTrue(truncated.startswith("+ line number 0"))

    def test_factory_observation_initialization_and_fingerprint(self):
        obs = FactoryObservation(
            workspace_root=".",
            worker_id="worker_42",
            step_index=1,
            git_status={"modified": ["gen_zero/model/dual_head.py"]},
            git_diff="+ some valid diff lines\n",
            recent_logs="Ran 1 test in 0.01s: OK\n",
            test_results={"passed": 1, "failed": 0, "errors": 0},
            contract_rules=["forbidden: /etc/"],
        )
        self.assertEqual(obs.worker_id, "worker_42")
        self.assertTrue(len(obs.fingerprint) > 0)
        summary = obs.to_compact_summary()
        self.assertIn("worker=worker_42", summary)
        self.assertIn("tests(passed=1, failed=0)", summary)

    def test_factory_observation_json_roundtrip(self):
        obs = FactoryObservation(
            workspace_root=".",
            worker_id="worker_json",
            step_index=2,
            git_status={"modified": ["a.py"], "untracked": ["b.py"]},
            git_diff="+ changes\n",
            recent_logs="log output\n",
            test_results={"passed": 2, "failed": 0},
            contract_rules=["rule: 1"],
            metadata={"source": "test"},
        )
        json_str = obs.to_json()
        reconstructed = FactoryObservation.from_json(json_str)
        self.assertEqual(reconstructed.worker_id, obs.worker_id)
        self.assertEqual(reconstructed.git_status, obs.git_status)
        self.assertEqual(reconstructed.fingerprint, obs.fingerprint)
        self.assertEqual(reconstructed.metadata, obs.metadata)

    def test_coalesced_debouncer_thread_safety(self):
        import threading
        debouncer = CoalescedDebouncer(debounce_floor_seconds=1.0, periodic_interval_seconds=5.0)

        def worker(idx):
            for i in range(50):
                debouncer.record_event(f"event_{idx}_{i}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(debouncer.pending_count, 200)
        drained = debouncer.mark_assessed()
        self.assertEqual(len(drained), 200)
        self.assertEqual(debouncer.pending_count, 0)

    def test_coalesced_debouncer_coalescing_and_periodic_pulse(self):
        debouncer = CoalescedDebouncer(
            debounce_floor_seconds=0.1,  # Fast for unit tests
            periodic_interval_seconds=0.3,
        )

        # 1. Initially clean, no events -> should not assess
        should, reason = debouncer.should_assess()
        self.assertFalse(should)

        # 2. Record event -> coalescing window begins
        debouncer.record_event("file_edited", {"path": "main.py"})
        self.assertEqual(debouncer.pending_count, 1)
        # Immediately checking: floor hasn't elapsed
        should, reason = debouncer.should_assess()
        self.assertFalse(should)

        # 3. Wait for debounce floor (0.12s) -> should trigger assessment
        time.sleep(0.12)
        should, reason = debouncer.should_assess()
        self.assertTrue(should)
        self.assertIn("DEBOUNCE_FLOOR_QUIET", reason)

        # Drain
        drained = debouncer.mark_assessed()
        self.assertEqual(len(drained), 1)
        self.assertEqual(debouncer.pending_count, 0)

        # 4. Silent execution: wait for periodic interval (0.32s) -> should trigger periodic pulse
        time.sleep(0.32)
        should_periodic, reason_periodic = debouncer.should_assess()
        self.assertTrue(should_periodic)
        self.assertEqual(reason_periodic, "PERIODIC_INTERVAL_ELAPSED")


class TestSemanticAssessmentAndPolicyStateMachine(unittest.TestCase):
    """Tests for Milestone 2: 10-D Atomic Assessment & Deterministic Policy."""

    def test_supervisor_assessment_finite_clamping(self):
        # Verify non-finite and out-of-range floats are safely clamped to [0.0, 1.0]
        assessment = SupervisorAssessment(
            implementation_complete=1.5,
            tests_sufficient=-0.5,
            contract_drift=float("nan"),
            worker_stuck=0.85,
        )
        self.assertEqual(assessment.implementation_complete, 1.0)
        self.assertEqual(assessment.tests_sufficient, 0.0)
        self.assertEqual(assessment.contract_drift, 0.0)
        self.assertEqual(assessment.worker_stuck, 0.85)

    def test_semantic_assessment_adapter_evaluation(self):
        adapter = SemanticAssessmentAdapter()

        # Case 1: Healthy progress with passing tests
        obs_healthy = FactoryObservation(
            workspace_root=".",
            worker_id="w1",
            step_index=2,
            git_status={"modified": ["file1.py"]},
            git_diff="+ def test(): pass\n",
            test_results={"passed": 5, "failed": 0, "errors": 0},
        )
        ass_healthy = adapter.evaluate(obs_healthy)
        self.assertGreaterEqual(ass_healthy.tests_sufficient, 0.80)
        self.assertGreaterEqual(ass_healthy.meaningful_progress, 0.70)
        self.assertLess(ass_healthy.contract_drift, 0.20)
        self.assertTrue(ass_healthy.is_completion_ready)

        # Case 2: Broken test execution
        obs_broken = FactoryObservation(
            workspace_root=".",
            worker_id="w1",
            step_index=3,
            git_status={"modified": ["file1.py"]},
            git_diff="+ syntax error here",
            test_results={"passed": 0, "failed": 3, "errors": 1},
        )
        ass_broken = adapter.evaluate(obs_broken)
        self.assertEqual(ass_broken.tests_sufficient, 0.0)
        self.assertGreaterEqual(ass_broken.needs_verification, 0.90)

    def test_policy_state_machine_deterministic_transitions(self):
        sm = SupervisorPolicyStateMachine(
            max_steers_per_worker=1,
            max_retries=3,
            steering_grace_seconds=10.0,
            drift_warn_threshold=0.70,
        )

        # Step 1: Normal progress -> CONTINUE_WORKER
        ass_normal = SupervisorAssessment(meaningful_progress=0.80)
        act, reason = sm.evaluate_step(ass_normal, now_monotonic=100.0)
        self.assertEqual(act, SupervisorAction.CONTINUE_WORKER)
        self.assertEqual(sm.state, SupervisorState.MONITORING)

        # Step 2: First warning (contract_drift = 0.75) -> STEER_WORKER (Tier 1) & enters STEERING_GRACE
        ass_drift = SupervisorAssessment(contract_drift=0.75)
        act, reason = sm.evaluate_step(ass_drift, now_monotonic=105.0)
        self.assertEqual(act, SupervisorAction.STEER_WORKER)
        self.assertEqual(sm.state, SupervisorState.STEERING_GRACE)
        self.assertEqual(sm.steer_count, 1)

        # Step 3: During grace period (t=110.0, grace is 10s until t=115.0) -> protected, CONTINUE_WORKER
        act_grace, _ = sm.evaluate_step(ass_drift, now_monotonic=110.0)
        self.assertEqual(act_grace, SupervisorAction.CONTINUE_WORKER)
        self.assertEqual(sm.state, SupervisorState.STEERING_GRACE)

        # Step 4: Grace expired (t=120.0) and still drifting -> STOP_WORKER (Tier 2)
        act_timeout, reason_to = sm.evaluate_step(ass_drift, now_monotonic=120.0)
        self.assertEqual(act_timeout, SupervisorAction.STOP_WORKER)
        self.assertEqual(sm.state, SupervisorState.STOPPED)
        self.assertIn("grace expired", reason_to.lower())

        # Step 5: Terminal state retention -> STOPPED state must not return CONTINUE_WORKER
        act_terminal, _ = sm.evaluate_step(ass_normal, now_monotonic=125.0)
        self.assertEqual(act_terminal, SupervisorAction.STOP_WORKER)
        self.assertEqual(sm.state, SupervisorState.STOPPED)

    def test_supervisor_assessment_properties(self):
        ass = SupervisorAssessment(
            ready_to_finish=0.90,
            tests_sufficient=0.90,
            contract_drift=0.10,
            worker_stuck=0.75,
        )
        self.assertTrue(ass.is_completion_ready)
        self.assertFalse(ass.is_drifting)
        self.assertTrue(ass.is_stuck)

        ass_drifting = SupervisorAssessment(contract_drift=0.75)
        self.assertTrue(ass_drifting.is_drifting)
        self.assertFalse(ass_drifting.is_stuck)

    def test_semantic_assessment_stagnation_history(self):
        adapter = SemanticAssessmentAdapter()
        obs1 = FactoryObservation(workspace_root=".", worker_id="w1", step_index=1)
        obs2 = FactoryObservation(workspace_root=".", worker_id="w1", step_index=2)
        obs3 = FactoryObservation(workspace_root=".", worker_id="w1", step_index=3)

        ass = adapter.evaluate(obs3, history=[obs1, obs2])
        self.assertGreater(ass.worker_stuck, 0.60)
        self.assertTrue(ass.is_stuck)
        self.assertEqual(ass.meaningful_progress, 0.10)


class TestTwoTierInterventionAndGracePeriod(unittest.TestCase):
    """Tests for Milestone 3: Live Steering Grace Period & Recovery."""

    def test_worker_recovery_during_grace_period(self):
        sm = SupervisorPolicyStateMachine(
            max_steers_per_worker=1,
            steering_grace_seconds=10.0,
        )

        # 1. Trigger Tier 1 Steer
        ass_stuck = SupervisorAssessment(worker_stuck=0.80)
        act, _ = sm.evaluate_step(ass_stuck, now_monotonic=100.0)
        self.assertEqual(act, SupervisorAction.STEER_WORKER)
        self.assertEqual(sm.state, SupervisorState.STEERING_GRACE)

        # 2. Worker acts on guidance and recovers (stuck drops to 0.1, progress increases)
        ass_recovered = SupervisorAssessment(worker_stuck=0.10, meaningful_progress=0.85)
        act_rec, reason_rec = sm.evaluate_step(ass_recovered, now_monotonic=104.0)
        self.assertEqual(act_rec, SupervisorAction.CONTINUE_WORKER)
        self.assertEqual(sm.state, SupervisorState.MONITORING)
        self.assertIn("recovered", reason_rec.lower())

    def test_async_supervisor_end_to_end(self):
        guidance_received = []

        def on_guidance(g):
            guidance_received.append(g)

        supervisor = AsyncSemanticSupervisor(
            debounce_floor_seconds=0.0,
            periodic_interval_seconds=10.0,
            steering_grace_seconds=5.0,
            on_guidance_callback=on_guidance,
        )

        # Worker emits event
        supervisor.on_worker_event("tool_call", {"tool": "edit_file"})

        # Step 1: Off-track observation -> triggers Tier 1 steer
        obs_drift = FactoryObservation(
            workspace_root=".",
            worker_id="w_test",
            step_index=1,
            git_status={"modified": ["bad_file.py"]},
            git_diff="+ rm -rf /\n",
            contract_rules=["forbidden: rm -rf"],
        )
        action, ass, reason = supervisor.step(obs_drift, force_assess=True)
        self.assertEqual(action, SupervisorAction.STOP_WORKER)
        self.assertEqual(supervisor.current_state, SupervisorState.STOPPED)


class TestContractDriftGate(unittest.TestCase):
    """Tests for Milestone 4: Architecture Contract Drift Gate."""

    def setUp(self):
        self.gate = ContractDriftGate(drift_threshold=0.75)

    def test_permitted_normal_modification(self):
        obs = FactoryObservation(
            workspace_root=".",
            worker_id="w1",
            step_index=1,
            git_status={"modified": ["gen_zero/model/dual_head.py"]},
            git_diff="+ def test(): return True\n",
        )
        verdict = self.gate.evaluate(obs)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.verdict_status, "PERMITTED")
        self.assertLess(verdict.contract_drift, 0.75)

    def test_blocked_contract_breach(self):
        # Modifying forbidden protected path /etc/ or .env
        obs = FactoryObservation(
            workspace_root=".",
            worker_id="w1",
            step_index=1,
            git_status={"modified": [".env", "secrets.key"]},
            git_diff="+ DB_PASSWORD=leaked\n",
            contract_rules=["do not edit: .env"],
        )
        verdict = self.gate.evaluate(obs)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.verdict_status, "BLOCKED_CONTRACT_DRIFT")
        self.assertGreaterEqual(verdict.contract_drift, 0.75)
        self.assertIn(".env", verdict.violating_files)


if __name__ == "__main__":
    unittest.main()
