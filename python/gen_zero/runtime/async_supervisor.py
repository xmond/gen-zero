"""Async Semantic Supervisor Runtime Orchestrator.

Implements Milestone 3 of Issue #20:
1. Dual-Loop Decoupling:
   Asynchronously observes worker progress without blocking worker tool actuation.
2. Coalesced Assessment & Two-Tier Interventions:
   Manages debounced sampling, runs 10-D semantic assessment, applies two-tier
   interventions (Tier 1 soft guidance with 30s grace, Tier 2 circuit breaker stops).
3. Telemetry & Audit Stream:
   Records observation snapshots, probability trajectories, and policy transitions.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import dataclasses
import time

from .debouncer import CoalescedDebouncer
from .supervisor_observation import FactoryObservation
from .supervisor_assessment import SupervisorAssessment, SemanticAssessmentAdapter
from .supervisor_policy import (
    SupervisorPolicyStateMachine,
    SupervisorAction,
    SupervisorState,
    SteeringGuidance,
)


@dataclasses.dataclass
class SupervisorTelemetry:
    """Telemetry record for each supervisor inspection tick."""
    timestamp: float
    worker_id: str
    step_index: int
    action: SupervisorAction
    assessment: SupervisorAssessment
    reason: str
    guidance: Optional[SteeringGuidance] = None


class AsyncSemanticSupervisor:
    """Orchestrates asynchronous semantic supervision over worker agents."""

    def __init__(
        self,
        debounce_floor_seconds: float = 5.0,
        periodic_interval_seconds: float = 30.0,
        steering_grace_seconds: float = 30.0,
        max_steers_per_worker: int = 1,
        max_retries: int = 3,
        on_guidance_callback: Optional[Callable[[SteeringGuidance], None]] = None,
        on_action_callback: Optional[Callable[[SupervisorAction, str], None]] = None,
    ):
        self.debouncer = CoalescedDebouncer(
            debounce_floor_seconds=debounce_floor_seconds,
            periodic_interval_seconds=periodic_interval_seconds,
        )
        self.assessment_adapter = SemanticAssessmentAdapter()
        self.policy = SupervisorPolicyStateMachine(
            max_steers_per_worker=max_steers_per_worker,
            max_retries=max_retries,
            steering_grace_seconds=steering_grace_seconds,
        )
        self.on_guidance_callback = on_guidance_callback
        self.on_action_callback = on_action_callback

        self.observation_history: List[FactoryObservation] = []
        self.telemetry_history: List[SupervisorTelemetry] = []

    def on_worker_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Informs supervisor of a worker event (e.g. file edit, test run, bash execution)."""
        self.debouncer.record_event(event_type, payload)

    def step(
        self,
        obs: FactoryObservation,
        now_monotonic: Optional[float] = None,
        force_assess: bool = False,
    ) -> Tuple[SupervisorAction, Optional[SupervisorAssessment], str]:
        """Runs supervisor inspection tick.

        Returns:
            Tuple of (SupervisorAction, SupervisorAssessment or None, explanation string).
        """
        now = now_monotonic if now_monotonic is not None else time.monotonic()

        # Check if debouncer warrants an assessment
        should_run, trigger_reason = self.debouncer.should_assess()
        if not should_run and not force_assess:
            return SupervisorAction.CONTINUE_WORKER, None, f"Coalescing ({trigger_reason})"

        # 1. Store observation in bounded history
        self.observation_history.append(obs)
        if len(self.observation_history) > 50:
            self.observation_history = self.observation_history[-50:]

        # 2. Run 10-D semantic probability evaluation
        assessment = self.assessment_adapter.evaluate(obs, self.observation_history[:-1])

        # 3. Drain debounced events
        self.debouncer.mark_assessed()

        # 4. Evaluate deterministic state machine
        action, reason = self.policy.evaluate_step(assessment, now_monotonic=now)

        # 5. Handle guidance injection or action dispatch
        guidance = None
        if action == SupervisorAction.STEER_WORKER and self.policy.last_guidance:
            guidance = self.policy.last_guidance
            if self.on_guidance_callback:
                try:
                    self.on_guidance_callback(guidance)
                except Exception:
                    pass

        if self.on_action_callback:
            try:
                self.on_action_callback(action, reason)
            except Exception:
                pass

        # 6. Record telemetry
        telemetry = SupervisorTelemetry(
            timestamp=time.time(),
            worker_id=obs.worker_id,
            step_index=obs.step_index,
            action=action,
            assessment=assessment,
            reason=reason,
            guidance=guidance,
        )
        self.telemetry_history.append(telemetry)
        if len(self.telemetry_history) > 500:
            self.telemetry_history = self.telemetry_history[-500:]

        return action, assessment, reason

    def reset_for_worker(self, worker_id: str) -> None:
        """Resets state machine and debouncer for a fresh worker process."""
        self.debouncer.reset()
        self.policy.reset_for_new_worker(worker_id)

    @property
    def current_state(self) -> SupervisorState:
        return self.policy.state

    @property
    def total_inspections(self) -> int:
        return len(self.telemetry_history)
