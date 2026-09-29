"""Tri-Party Continuous Verifier Harness Architecture (Issue #13).

Separates powers into three distinct entities:
1. Orchestrator: High-level planning and sub-goal breakdown.
2. Executor: Code editing, shell commands, test runs (yields pure objective facts).
3. Verifier: Objective evidence sanitization, in-loop self-healing triggers,
   continuous multi-criteria verification, and session context compaction.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gen_zero.client import GenZero
from .context_compactor import CompactionResult, SessionContextCompactor
from .evidence_sanitizer import ObjectiveEvidence, ObjectiveEvidenceSanitizer
from .goal_verifier import GoalVerificationVerdict, GoalVerifier, VerifierStatus
from .self_healing import InLoopSelfHealingTrigger, SelfHealingTicket


@dataclass
class TriPartyStepResult:
    """Outcome of a single step in the Tri-Party loop."""
    step_index: int
    action_type: str
    command_executed: Optional[str]
    exit_code: int
    is_healed: bool
    self_healing_ticket: Optional[SelfHealingTicket]
    verdict: Optional[GoalVerificationVerdict]
    false_completion_intercepted: bool = False
    timing_ms: float = 0.0
    narrow_feedback: Optional[str] = None


class TriPartyHarness:
    """Continuous Tri-Party Execution Harness with System 1 Verification."""

    def __init__(
        self,
        client: Optional[Any] = None,
        goal_verifier: Optional[GoalVerifier] = None,
        compactor: Optional[SessionContextCompactor] = None,
        auto_compact_interval: int = 5
    ):
        if client is None and goal_verifier is None:
            from gen_zero.client import GenZero
            client = GenZero()
        self.client = client or (goal_verifier.client if goal_verifier else None)
        self.goal_verifier = goal_verifier or GoalVerifier(client=self.client)
        self.verifier = self.goal_verifier
        self.compactor = compactor or SessionContextCompactor()
        self.auto_compact_interval = auto_compact_interval

        # Tracking state
        self.history: List[Dict[str, Any]] = []
        self.intercepted_false_completions: List[Dict[str, Any]] = []
        self.self_healing_history: List[SelfHealingTicket] = []
        self.active_step_count: int = 0

    def step(
        self,
        action: str,
        executor_fn: Callable[[], Tuple[int, str, str]],
        current_goal: str,
        criteria: List[str],
        agent_claims_completed: bool = False
    ) -> TriPartyStepResult:
        """Executes a single step under Tri-Party separation of powers."""
        t0 = time.perf_counter()
        self.active_step_count += 1
        step_idx = self.active_step_count

        # Record action in history
        self.history.append({"role": "assistant", "content": action, "step": step_idx})

        # 1. Execute tool via Executor (objective facts only)
        try:
            exit_code, stdout, stderr = executor_fn()
        except Exception as e:
            exit_code, stdout, stderr = 1, "", f"Executor execution exception: {type(e).__name__}: {str(e)}"
        raw_output = f"{stdout}\n{stderr}".strip()
        self.history.append({"role": "tool", "content": raw_output, "exit_code": exit_code, "step": step_idx})

        # 2. In-Loop Self-Healing Trigger
        needs_healing, ticket = InLoopSelfHealingTrigger.analyze_execution(
            command=action,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr
        )
        if needs_healing and ticket:
            self.self_healing_history.append(ticket)

        # 3. Continuous Verification check
        verdict = self.verifier.verify_step(
            goal=current_goal,
            criteria=criteria,
            evidence=raw_output,
            explicit_exit_code=exit_code
        )

        # 4. Asymmetric False-Completion Interception:
        # If the agent claims it completed the task, but Verifier rejects it:
        false_completion_intercepted = False
        if agent_claims_completed and not verdict.is_goal_met:
            false_completion_intercepted = True
            self.intercepted_false_completions.append({
                "step": step_idx,
                "action": action,
                "goal": current_goal,
                "verdict": verdict.to_dict(),
                "evidence_snippet": raw_output[:200]
            })

        # 5. Periodic Context Compaction
        if self.active_step_count % self.auto_compact_interval == 0:
            comp_res = self.compactor.compact_history(self.history)
            self.history = comp_res.compacted_history

        timing_ms = (time.perf_counter() - t0) * 1000.0

        # Narrow feedback generation
        rejection_msg = (
            f"[COMPLETION REJECTED BY VERIFIER]: You declared completion, but goal criteria were NOT met.\n"
            f"Failing Criteria: {', '.join(verdict.failed_criteria)}\n"
            f"Action Required: Resolve the failing criteria before concluding the task."
        ) if false_completion_intercepted else None

        healing_msg = ticket.format_orchestrator_prompt() if (needs_healing and ticket) else None

        if rejection_msg and healing_msg:
            narrow_feedback = f"{rejection_msg}\n\n{healing_msg}"
        elif rejection_msg:
            narrow_feedback = rejection_msg
        elif healing_msg:
            narrow_feedback = healing_msg
        else:
            narrow_feedback = None

        return TriPartyStepResult(
            step_index=step_idx,
            action_type=action,
            command_executed=action,
            exit_code=exit_code,
            is_healed=needs_healing,
            self_healing_ticket=ticket,
            verdict=verdict,
            false_completion_intercepted=false_completion_intercepted,
            timing_ms=timing_ms,
            narrow_feedback=narrow_feedback
        )

    def compact_context_now(self) -> CompactionResult:
        """Forces an immediate session compaction."""
        res = self.compactor.compact_history(self.history)
        self.history = res.compacted_history
        return res
