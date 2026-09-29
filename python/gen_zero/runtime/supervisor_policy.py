"""Deterministic Policy State Machine for Supervisor Governance.

Implements Milestone 2 & Milestone 3 of Issue #20:
1. Deterministic Control:
   Translates 10-dimensional probabilities into deterministic policy actions:
   - CONTINUE_WORKER
   - STEER_WORKER (Tier 1 Soft Guidance)
   - STOP_WORKER (Tier 2 Hard Circuit Breaker)
   - RETRY_WORKER
   - TRIGGER_VERIFIER (Awakening Issue #13 TriParty Verifier)
   - ESCALATE_HUMAN
   - FINISH_WORKFLOW
2. Two-Tier Intervention & Monotonic Steering Grace Period:
   Enforces 30.0s grace period during which worker is protected from hard stops
   to allow self-healing, followed by deterministic circuit breaker on timeout.
3. Hysteresis & Invariant Priority:
   Ensures hard safety bounds cannot be overridden by probability noise.
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import dataclasses
import time

from .supervisor_assessment import SupervisorAssessment


class SupervisorAction(str, Enum):
    CONTINUE_WORKER = "continue_worker"
    STEER_WORKER = "steer_worker"
    STOP_WORKER = "stop_worker"
    RETRY_WORKER = "retry_worker"
    TRIGGER_VERIFIER = "trigger_verifier"
    ESCALATE_HUMAN = "escalate_human"
    FINISH_WORKFLOW = "finish_workflow"


class SupervisorState(str, Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    STEERING_GRACE = "steering_grace"
    VERIFYING = "verifying"
    STOPPED = "stopped"
    ESCALATED = "escalated"
    FINISHED = "finished"


@dataclasses.dataclass
class SteeringGuidance:
    """Structured in-turn guidance payload injected into worker channel."""
    guidance_id: str
    drift_category: str
    severity: float
    detected_issue: str
    corrective_instruction: str
    timestamp: float = dataclasses.field(default_factory=time.time)


class SupervisorPolicyStateMachine:
    """State machine governing two-tier intervention, grace periods, and invariant safety."""

    def __init__(
        self,
        max_steers_per_worker: int = 1,
        max_retries: int = 3,
        steering_grace_seconds: float = 30.0,
        drift_warn_threshold: float = 0.70,
        drift_stop_threshold: float = 0.75,
        stuck_warn_threshold: float = 0.70,
        stuck_stop_threshold: float = 0.85,
        finish_threshold: float = 0.85,
    ):
        self.max_steers_per_worker = max_steers_per_worker
        self.max_retries = max_retries
        self.steering_grace_seconds = steering_grace_seconds
        self.drift_warn_threshold = drift_warn_threshold
        self.drift_stop_threshold = drift_stop_threshold
        self.stuck_warn_threshold = stuck_warn_threshold
        self.stuck_stop_threshold = stuck_stop_threshold
        self.finish_threshold = finish_threshold

        self.state: SupervisorState = SupervisorState.MONITORING
        self.steer_count: int = 0
        self.retry_count: int = 0
        self.grace_start_time: Optional[float] = None
        self.last_guidance: Optional[SteeringGuidance] = None

    def evaluate_step(
        self,
        assessment: SupervisorAssessment,
        now_monotonic: Optional[float] = None,
    ) -> Tuple[SupervisorAction, str]:
        """Evaluates policy state machine transition given current assessment probabilities."""
        now = now_monotonic if now_monotonic is not None else time.monotonic()

        # Invariant 1: Fatal unrecoverable breach (> 0.95)
        if assessment.contract_drift >= 0.95:
            self.state = SupervisorState.STOPPED
            return SupervisorAction.STOP_WORKER, f"Fatal contract breach detected ({assessment.contract_drift:.2f})."

        # Invariant 2: Human escalation or exhausted retries
        if assessment.needs_human >= 0.90 or self.retry_count >= self.max_retries:
            self.state = SupervisorState.ESCALATED
            return SupervisorAction.ESCALATE_HUMAN, "Max retries exceeded or critical human intervention needed."

        # State Handling: STEERING_GRACE
        if self.state == SupervisorState.STEERING_GRACE:
            grace_elapsed = now - (self.grace_start_time or now)
            has_recovered = (
                assessment.contract_drift < self.drift_warn_threshold
                and assessment.worker_stuck < self.stuck_warn_threshold
                and assessment.meaningful_progress >= 0.40
            )

            if has_recovered:
                self.state = SupervisorState.MONITORING
                self.grace_start_time = None
                return SupervisorAction.CONTINUE_WORKER, "Worker recovered within steering grace period."

            if grace_elapsed < self.steering_grace_seconds:
                # Still within grace window: protect from hard stop
                remaining = self.steering_grace_seconds - grace_elapsed
                return SupervisorAction.CONTINUE_WORKER, f"Steering grace active ({remaining:.1f}s remaining)."
            else:
                # Grace period expired and still degraded: Tier 2 Hard Circuit Breaker
                self.state = SupervisorState.STOPPED
                self.grace_start_time = None
                return SupervisorAction.STOP_WORKER, f"Steering grace expired after {grace_elapsed:.1f}s without recovery."

        # State Handling: MONITORING
        if self.state in (SupervisorState.MONITORING, SupervisorState.IDLE):
            # 1. Check completion ready
            if assessment.is_completion_ready:
                if assessment.needs_verification >= 0.70:
                    self.state = SupervisorState.VERIFYING
                    return SupervisorAction.TRIGGER_VERIFIER, "High completion confidence warrants independent verification."
                self.state = SupervisorState.FINISHED
                return SupervisorAction.FINISH_WORKFLOW, "Workflow goals verified and ready to finalize."

            # 2. Check drift or stuck
            is_drifting = assessment.contract_drift >= self.drift_warn_threshold
            is_stuck = assessment.worker_stuck >= self.stuck_warn_threshold
            is_off_track = assessment.work_off_track >= self.drift_warn_threshold

            if is_drifting or is_stuck or is_off_track:
                # Check if eligible for Tier 1 Soft Guidance
                if self.steer_count < self.max_steers_per_worker:
                    self.steer_count += 1
                    self.state = SupervisorState.STEERING_GRACE
                    self.grace_start_time = now

                    cat = "contract_drift" if is_drifting else ("worker_stuck" if is_stuck else "work_off_track")
                    sev = max(assessment.contract_drift, assessment.worker_stuck, assessment.work_off_track)
                    self.last_guidance = SteeringGuidance(
                        guidance_id=f"steer_{int(time.time()*1000)%1000000}",
                        drift_category=cat,
                        severity=sev,
                        detected_issue=f"Degraded {cat} metric ({sev:.2f})",
                        corrective_instruction=f"Please refocus on task specification and resolve {cat}.",
                    )
                    return SupervisorAction.STEER_WORKER, f"Tier 1 steering guidance emitted; starting {self.steering_grace_seconds}s grace."
                else:
                    # Steer quota exhausted: Tier 2 Hard Stop
                    self.state = SupervisorState.STOPPED
                    return SupervisorAction.STOP_WORKER, f"Degraded metrics with steering quota exhausted ({self.steer_count}/{self.max_steers_per_worker})."

            return SupervisorAction.CONTINUE_WORKER, "Normal execution within tolerance bounds."

        # State Handling: VERIFYING
        if self.state == SupervisorState.VERIFYING:
            if assessment.requirements_satisfied >= 0.80 and assessment.tests_sufficient >= 0.80:
                self.state = SupervisorState.FINISHED
                return SupervisorAction.FINISH_WORKFLOW, "Verification passed; concluding workflow."
            else:
                self.state = SupervisorState.MONITORING
                return SupervisorAction.CONTINUE_WORKER, "Verification incomplete; continuing iteration."

        # State Handling: Terminal States
        if self.state == SupervisorState.STOPPED:
            return SupervisorAction.STOP_WORKER, "Worker is stopped."
        if self.state == SupervisorState.FINISHED:
            return SupervisorAction.FINISH_WORKFLOW, "Workflow is finished."
        if self.state == SupervisorState.ESCALATED:
            return SupervisorAction.ESCALATE_HUMAN, "Workflow is escalated."

        return SupervisorAction.CONTINUE_WORKER, f"State: {self.state}"

    def reset_for_new_worker(self, worker_id: str) -> None:
        """Resets steering tracking for a new worker invocation."""
        self.state = SupervisorState.MONITORING
        self.steer_count = 0
        self.grace_start_time = None
        self.last_guidance = None

    def record_retry(self) -> int:
        """Records a retry attempt."""
        self.retry_count += 1
        return self.retry_count
