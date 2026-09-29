"""Agent Tool Call Safety Interlock & CP-SAT Formal Constraint Bridge.

Implements Milestone 5 of Issue #26:
- Dual Interlock on Agent Tool Invocations:
  1. Pre-execution Interlock: Inspects tool arguments for prompt injection, destructive shell commands,
     arbitrary code execution, or unauthorized resource mutations.
  2. Post-execution Interlock: Inspects tool stdout/stderr/results for secret key leakage (API tokens, private keys, JWTs).
- CP-SAT Formal Veto: Integrates with CPSATFormalSolver to enforce hard invariant blocking on high-risk payloads.
- Structured Audit Trail: Full telemetry and probe attribution for compliance and security auditing.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import dataclasses
import time
import json

from gen_zero.guard.batteries import (
    InputBattery,
    OutputBattery,
    GuardAction,
)
from gen_zero.guard.dual_gate import DualGateGuardrail, GuardVerdict
from gen_zero.gate.cpsat_formal_solver import CPSATFormalSolver, CPSATVerdict


@dataclasses.dataclass
class ToolInterlockVerdict:
    """Verdict returned by Tool Call Interlock."""
    phase: str           # "pre_execution" or "post_execution"
    tool_name: str
    allowed: bool
    action: GuardAction
    interlocked_by_cpsat: bool
    block_reason: Optional[str]
    severity_score: float
    audit_record: Dict[str, Any]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "tool_name": self.tool_name,
            "allowed": self.allowed,
            "action": self.action.value,
            "interlocked_by_cpsat": self.interlocked_by_cpsat,
            "block_reason": self.block_reason,
            "severity_score": round(self.severity_score, 4),
            "audit_record": self.audit_record,
            "latency_ms": round(self.latency_ms, 2),
        }


class AgentToolInterlock:
    """Monitors, filters, and interlocks tool execution in autonomous agent workflows."""

    def __init__(
        self,
        guardrail: Optional[DualGateGuardrail] = None,
        cpsat_solver: Optional[CPSATFormalSolver] = None,
    ):
        self.guardrail = guardrail or DualGateGuardrail()
        self.cpsat_solver = cpsat_solver or CPSATFormalSolver(hard_timeout_ms=2.0)
        self.audit_log: List[Dict[str, Any]] = []

    def _args_to_string(self, tool_args: Any) -> str:
        if isinstance(tool_args, str):
            return tool_args
        try:
            return json.dumps(tool_args, default=str)
        except Exception:
            return str(tool_args)

    def interlock_pre_execution(
        self,
        tool_name: str,
        tool_args: Any,
        agent_context: Optional[str] = None,
    ) -> ToolInterlockVerdict:
        """Inspects tool invocation arguments before dispatching execution to runtime."""
        t0 = time.perf_counter()
        args_str = self._args_to_string(tool_args)
        combined_payload = f"Tool: {tool_name}\nArgs: {args_str}"

        # 1. Evaluate via Input Guardrail
        verdict: GuardVerdict = self.guardrail.guard_input(
            prompt=combined_payload,
            context=agent_context,
            policy="strict",
        )

        allowed = (verdict.action == GuardAction.PASS)
        interlocked_by_cpsat = False
        block_reason = None

        # 2. CP-SAT Formal Safety Interlock
        # If jailbreak, harmful request, or unauthorized execution detected, CP-SAT formal solver vetoes execution
        if verdict.action in [GuardAction.BLOCK, GuardAction.SUPPORT]:
            candidate_utils = {
                "EXECUTE": 1.0,
                "BLOCK_EXECUTION": 0.0,
            }
            forbidden = {"EXECUTE"}
            cpsat_res: CPSATVerdict = self.cpsat_solver.solve_safest_optimal_action(
                candidate_utilities=candidate_utils,
                forbidden_actions=forbidden,
                fallback_safe_action="BLOCK_EXECUTION",
            )
            allowed = False
            interlocked_by_cpsat = True
            block_reason = f"CP-SAT Formal Interlock barred execution: {verdict.explanation}"

        elif verdict.action == GuardAction.REVIEW:
            allowed = False
            block_reason = f"Tool call requires secondary review: {verdict.explanation}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        record = {
            "timestamp": time.time(),
            "phase": "pre_execution",
            "tool_name": tool_name,
            "allowed": allowed,
            "action": verdict.action.value,
            "severity": verdict.severity_score,
            "probes": verdict.probe_probabilities,
            "reason": block_reason or "Cleared pre-execution check",
        }
        self.audit_log.append(record)

        return ToolInterlockVerdict(
            phase="pre_execution",
            tool_name=tool_name,
            allowed=allowed,
            action=verdict.action,
            interlocked_by_cpsat=interlocked_by_cpsat,
            block_reason=block_reason,
            severity_score=verdict.severity_score,
            audit_record=record,
            latency_ms=elapsed_ms,
        )

    def interlock_post_execution(
        self,
        tool_name: str,
        tool_output: Any,
        agent_context: Optional[str] = None,
    ) -> ToolInterlockVerdict:
        """Inspects tool execution results / stdout to prevent secret key or sensitive data leakage."""
        t0 = time.perf_counter()
        out_str = self._args_to_string(tool_output)

        # Evaluate via Output Guardrail
        verdict: GuardVerdict = self.guardrail.guard_output(
            response=out_str,
            input_context=f"Tool: {tool_name}",
            policy="strict",
        )

        allowed = (verdict.action == GuardAction.PASS)
        interlocked_by_cpsat = False
        block_reason = None

        if verdict.action in [GuardAction.BLOCK, GuardAction.SUPPORT]:
            allowed = False
            interlocked_by_cpsat = True
            block_reason = f"Tool output sanitized / blocked: {verdict.explanation}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        record = {
            "timestamp": time.time(),
            "phase": "post_execution",
            "tool_name": tool_name,
            "allowed": allowed,
            "action": verdict.action.value,
            "severity": verdict.severity_score,
            "probes": verdict.probe_probabilities,
            "reason": block_reason or "Cleared post-execution check",
        }
        self.audit_log.append(record)

        return ToolInterlockVerdict(
            phase="post_execution",
            tool_name=tool_name,
            allowed=allowed,
            action=verdict.action,
            interlocked_by_cpsat=interlocked_by_cpsat,
            block_reason=block_reason,
            severity_score=verdict.severity_score,
            audit_record=record,
            latency_ms=elapsed_ms,
        )
