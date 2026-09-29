"""Gen-Zero Layer 2: Robust Loop State Machine, Stagnation Breaker & Dry-Run Contract.

RFC Implementation for Issue #8 (Module 3):
1. Asymmetric Oracle Verification:
   - Independent verification callback verify(observation).
   - Model self-reported done (P(done) >= 0.90) alone is never trusted as proof of success.
   - Missing verifier yields UNVERIFIED_FINISH; rejected verification yields ESCALATED.
2. State Stagnation Circuit Breaker:
   - Perceptual diff hashing across observations.
   - Detects consecutive identical actions causing zero environmental change (noChange).
   - Immediately trips circuit breaker, preventing infinite blind loops.
3. Native Dry-Run / Shadow Execution:
   - Evaluates pipeline and returns structured dry-run report without executing physical actuation.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .composite_decision import CompositeStepDecision, CompositeDecisionEngine
from ..gate.policy_gate import DecisionPolicyGate, PolicyGateVerdict, PolicyVerdictAction, DomainRiskProfile
from ..model.dual_head import normalize_state_repr


class LoopExecutionStatus(str, Enum):
    INIT = "init"
    RUNNING = "running"
    DRY_RUN = "dry_run"
    SUCCESS = "success"
    UNVERIFIED_FINISH = "unverified_finish"
    STAGNATION_TRIPPED = "stagnation_tripped"
    POLICY_STOPPED = "policy_stopped"
    CONFIRMATION_REQUIRED = "confirmation_required"
    ESCALATED = "escalated"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"


def compute_perceptual_fingerprint(observation: Any) -> str:
    """Computes deterministic SHA-256 perceptual fingerprint of environment observation."""
    canon_str = normalize_state_repr(observation)
    return hashlib.sha256(canon_str.encode("utf-8")).hexdigest()[:16]


class StagnationCircuitBreaker:
    """Circuit breaker detecting environmental stagnation and infinite loop blind trials."""

    def __init__(self, max_consecutive_stagnations: int = 2):
        self.max_consecutive_stagnations = max_consecutive_stagnations
        self.recent_actions: List[str] = []
        self.recent_fingerprints: List[str] = []
        self.stagnation_count: int = 0
        self.is_tripped: bool = False
        self.trip_reason: Optional[str] = None

    def record_step(self, action_signature: str, post_observation: Any) -> bool:
        """Records executed action and post-action observation fingerprint.

        Returns:
            True if circuit breaker is intact, False if breaker has TRIPPED.
        """
        if self.is_tripped:
            return False

        fp = compute_perceptual_fingerprint(post_observation)
        self.recent_actions.append(action_signature)
        self.recent_fingerprints.append(fp)

        # Need at least two steps to evaluate stagnation
        if len(self.recent_fingerprints) >= 2:
            prev_fp = self.recent_fingerprints[-2]
            prev_act = self.recent_actions[-2]

            # Check: Did the same action fail to produce any change in state fingerprint?
            if action_signature == prev_act and fp == prev_fp:
                self.stagnation_count += 1
                if self.stagnation_count >= self.max_consecutive_stagnations:
                    self.is_tripped = True
                    self.trip_reason = (
                        f"STATE_STAGNATION_DETECTED: Action '{action_signature}' executed "
                        f"{self.stagnation_count} times consecutively without environmental state change (FP={fp})."
                    )
                    return False
            else:
                self.stagnation_count = 0

        return True

    def reset(self) -> None:
        """Resets breaker state for a fresh execution episode."""
        self.recent_actions.clear()
        self.recent_fingerprints.clear()
        self.stagnation_count = 0
        self.is_tripped = False
        self.trip_reason = None


@dataclass
class LoopStepRecord:
    step_idx: int
    pre_fingerprint: str
    decision: CompositeStepDecision
    gate_verdict: PolicyGateVerdict
    action_signature: str
    post_fingerprint: Optional[str] = None
    observation_changed: bool = False
    dry_run: bool = False


@dataclass
class LoopExecutionReport:
    status: LoopExecutionStatus
    total_steps: int
    final_observation: Any
    history: List[LoopStepRecord] = field(default_factory=list)
    verdict_reason: str = ""
    timing_ms: float = 0.0
    agent_finished: bool = False
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "agent_finished": self.agent_finished,
            "verified": self.verified,
            "total_steps": self.total_steps,
            "verdict_reason": self.verdict_reason,
            "timing_ms": round(self.timing_ms, 2),
            "step_count": len(self.history)
        }


class CompositeExecutionLoop:
    """Robust loop state machine enforcing asymmetric trust, stagnation breakers, and dry-run."""

    def __init__(
        self,
        decision_engine: Optional[CompositeDecisionEngine] = None,
        policy_gate: Optional[DecisionPolicyGate] = None,
        circuit_breaker: Optional[StagnationCircuitBreaker] = None,
        max_steps: int = 25
    ):
        self.decision_engine = decision_engine or CompositeDecisionEngine()
        self.policy_gate = policy_gate or DecisionPolicyGate()
        self.circuit_breaker = circuit_breaker or StagnationCircuitBreaker()
        self.max_steps = max_steps

    def execute_loop(
        self,
        initial_state: Any,
        get_affordances_fn: Callable[[Any], List[str]],
        actuate_fn: Callable[[str, Optional[str]], Any],
        verify_fn: Optional[Callable[[Any], bool]] = None,
        goal: Optional[str] = None,
        dry_run: bool = False,
        profile: Optional[DomainRiskProfile] = None
    ) -> LoopExecutionReport:
        """Runs the robust execution state machine."""
        t0 = time.perf_counter()
        current_state = initial_state
        self.circuit_breaker.reset()
        history: List[LoopStepRecord] = []

        status = LoopExecutionStatus.RUNNING

        for step_idx in range(self.max_steps):
            pre_fp = compute_perceptual_fingerprint(current_state)
            affordances = get_affordances_fn(current_state)

            # 1. Decision Step
            decision = self.decision_engine.decide_step(
                state=current_state,
                affordances=affordances,
                goal=goal
            )
            act_sig = f"{decision.action}:{decision.target}"

            # 2. Dual-Track Policy Gate Check
            gate = self.policy_gate.evaluate_policy(
                decision=decision,
                state=current_state,
                profile=profile
            )

            # 3. Dry-Run Handling
            if dry_run:
                step_record = LoopStepRecord(
                    step_idx=step_idx,
                    pre_fingerprint=pre_fp,
                    decision=decision,
                    gate_verdict=gate,
                    action_signature=act_sig,
                    post_fingerprint=pre_fp,
                    observation_changed=False,
                    dry_run=True
                )
                history.append(step_record)
                elapsed = (time.perf_counter() - t0) * 1000.0
                return LoopExecutionReport(
                    status=LoopExecutionStatus.DRY_RUN,
                    total_steps=1,
                    final_observation=current_state,
                    history=history,
                    verdict_reason="Dry-run simulation completed successfully without physical actuation.",
                    timing_ms=elapsed
                )

            # 4. Handle Policy Gate Verdicts
            if gate.action == PolicyVerdictAction.STOP:
                step_record = LoopStepRecord(
                    step_idx=step_idx,
                    pre_fingerprint=pre_fp,
                    decision=decision,
                    gate_verdict=gate,
                    action_signature=act_sig
                )
                history.append(step_record)
                return LoopExecutionReport(
                    status=LoopExecutionStatus.POLICY_STOPPED,
                    total_steps=step_idx + 1,
                    final_observation=current_state,
                    history=history,
                    verdict_reason=f"Policy gate STOP triggered: {gate.reason}",
                    timing_ms=(time.perf_counter() - t0) * 1000.0
                )

            if gate.action == PolicyVerdictAction.CONFIRM:
                step_record = LoopStepRecord(
                    step_idx=step_idx,
                    pre_fingerprint=pre_fp,
                    decision=decision,
                    gate_verdict=gate,
                    action_signature=act_sig
                )
                history.append(step_record)
                return LoopExecutionReport(
                    status=LoopExecutionStatus.CONFIRMATION_REQUIRED,
                    total_steps=step_idx + 1,
                    final_observation=current_state,
                    history=history,
                    verdict_reason=f"High risk operation requires approval: {gate.reason}",
                    timing_ms=(time.perf_counter() - t0) * 1000.0
                )

            if gate.action == PolicyVerdictAction.ESCALATE:
                step_record = LoopStepRecord(
                    step_idx=step_idx,
                    pre_fingerprint=pre_fp,
                    decision=decision,
                    gate_verdict=gate,
                    action_signature=act_sig
                )
                history.append(step_record)
                return LoopExecutionReport(
                    status=LoopExecutionStatus.ESCALATED,
                    total_steps=step_idx + 1,
                    final_observation=current_state,
                    history=history,
                    verdict_reason=f"Ambiguity escalated to supervisor: {gate.reason}",
                    timing_ms=(time.perf_counter() - t0) * 1000.0
                )

            # 5. Check Asymmetric Oracle Termination Verification
            # If model self-reports done, or finish action chosen:
            if decision.done_prob >= 0.90 or decision.action == "finish":
                history.append(LoopStepRecord(
                    step_idx=step_idx, pre_fingerprint=pre_fp, decision=decision,
                    gate_verdict=gate, action_signature=act_sig,
                ))
                oracle_passed = False
                if not callable(verify_fn):
                    finish_status = LoopExecutionStatus.UNVERIFIED_FINISH
                    reason = "Model reported completion, but no independent verifier is available."
                else:
                    try:
                        # The verifier contract is bool; truthy strings/objects
                        # are not evidence of a successful independent check.
                        oracle_passed = verify_fn(current_state) is True
                        reason = ("Task successfully completed and verified." if oracle_passed
                                  else "ASYMMETRIC_TRUST_BREACH: External oracle did not verify completion.")
                    except Exception:
                        reason = "Independent verifier failed; completion remains unverified."
                    finish_status = (LoopExecutionStatus.SUCCESS if oracle_passed
                                     else LoopExecutionStatus.ESCALATED)
                return LoopExecutionReport(
                    status=finish_status,
                    total_steps=step_idx + 1,
                    final_observation=current_state,
                    history=history,
                    verdict_reason=reason,
                    timing_ms=(time.perf_counter() - t0) * 1000.0,
                    agent_finished=True,
                    verified=oracle_passed,
                )

            # 6. Physical Actuation
            next_state = actuate_fn(decision.action, decision.target)
            post_fp = compute_perceptual_fingerprint(next_state)
            has_changed = (post_fp != pre_fp)

            step_record = LoopStepRecord(
                step_idx=step_idx,
                pre_fingerprint=pre_fp,
                decision=decision,
                gate_verdict=gate,
                action_signature=act_sig,
                post_fingerprint=post_fp,
                observation_changed=has_changed
            )
            history.append(step_record)

            # 7. Stagnation Circuit Breaker Check
            is_intact = self.circuit_breaker.record_step(act_sig, next_state)
            if not is_intact:
                return LoopExecutionReport(
                    status=LoopExecutionStatus.STAGNATION_TRIPPED,
                    total_steps=step_idx + 1,
                    final_observation=next_state,
                    history=history,
                    verdict_reason=self.circuit_breaker.trip_reason or "State stagnation tripped.",
                    timing_ms=(time.perf_counter() - t0) * 1000.0
                )

            current_state = next_state

            # Periodic Oracle Check
            periodic_verified = False
            if callable(verify_fn):
                try:
                    periodic_verified = verify_fn(current_state) is True
                except Exception:
                    return LoopExecutionReport(
                        status=LoopExecutionStatus.ESCALATED,
                        total_steps=step_idx + 1, final_observation=current_state,
                        history=history, verdict_reason="Independent verifier failed.",
                        timing_ms=(time.perf_counter() - t0) * 1000.0,
                    )
            if periodic_verified:
                return LoopExecutionReport(
                    status=LoopExecutionStatus.SUCCESS,
                    total_steps=step_idx + 1,
                    final_observation=current_state,
                    history=history,
                    verdict_reason="External oracle verified task completion.",
                    verified=True,
                    timing_ms=(time.perf_counter() - t0) * 1000.0
                )

        return LoopExecutionReport(
            status=LoopExecutionStatus.MAX_STEPS_EXCEEDED,
            total_steps=self.max_steps,
            final_observation=current_state,
            history=history,
            verdict_reason=f"Exceeded maximum step budget of {self.max_steps}.",
            timing_ms=(time.perf_counter() - t0) * 1000.0
        )
