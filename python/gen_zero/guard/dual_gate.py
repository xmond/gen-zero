"""Bidirectional Dual-Gate Guardrail Interface (Input Gate + Output Gate).

Implements Module 1 & 2 of Issue #26:
- DualGateGuardrail:
  - guard_input(prompt, context=None, policy="strict") -> GuardVerdict
  - guard_output(response, input_context=None, policy="strict") -> GuardVerdict
- Prompt injection immunity: zero token generation, pure non-autoregressive logit difference.
- Four-tier arbitration: support / block / review / pass.
"""

from typing import Dict, List, Any, Optional, Union
import dataclasses
import time

from gen_zero.guard.batteries import (
    InputBattery,
    OutputBattery,
    GuardAction,
    BatteryEvaluationResult,
)
from gen_zero.guard.routing_algebra import (
    RoutingPolicy,
    PrecedenceRoutingAlgebra,
    RoutingDecision,
)
from gen_zero.guard.cascaded_guard import (
    CascadedGuardEngine,
    CascadedGuardResult,
)


@dataclasses.dataclass
class GuardVerdict:
    """Consolidated safety verdict returned by DualGateGuardrail."""
    side: str  # "input" or "output"
    action: GuardAction
    is_safe: bool  # True if action == PASS
    severity_score: float
    highest_hazard: Optional[str]
    escalated: bool
    fast_path_used: bool
    latency_ms: float
    policy_applied: str
    probe_probabilities: Dict[str, float]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "action": self.action.value,
            "is_safe": self.is_safe,
            "severity_score": round(self.severity_score, 4),
            "highest_hazard": self.highest_hazard,
            "escalated": self.escalated,
            "fast_path_used": self.fast_path_used,
            "latency_ms": round(self.latency_ms, 2),
            "policy_applied": self.policy_applied,
            "probe_probabilities": {k: round(v, 4) for k, v in self.probe_probabilities.items()},
            "explanation": self.explanation,
        }


class DualGateGuardrail:
    """Bidirectional non-autoregressive guardrail for prompt input and LLM response output."""

    def __init__(
        self,
        input_battery: Optional[InputBattery] = None,
        output_battery: Optional[OutputBattery] = None,
        default_policy: Optional[RoutingPolicy] = None,
    ):
        self.input_battery = input_battery or InputBattery()
        self.output_battery = output_battery or OutputBattery()
        self.default_policy = default_policy or RoutingPolicy.strict()

        self.input_engine = CascadedGuardEngine(battery=self.input_battery)
        self.output_engine = CascadedGuardEngine(battery=self.output_battery)

    def _resolve_policy(self, policy: Optional[Union[str, RoutingPolicy]]) -> RoutingPolicy:
        if isinstance(policy, RoutingPolicy):
            return policy
        if isinstance(policy, str):
            if policy.lower() == "permissive":
                return RoutingPolicy.permissive()
            return RoutingPolicy.strict()
        return self.default_policy

    def guard_input(
        self,
        prompt: str,
        context: Optional[str] = None,
        policy: Optional[Union[str, RoutingPolicy]] = None,
    ) -> GuardVerdict:
        """Inspects user input prompt before forwarding to model / agent."""
        pol = self._resolve_policy(policy)
        res: CascadedGuardResult = self.input_engine.evaluate(prompt, context=context, policy=pol)

        action = res.action
        is_safe = (action == GuardAction.PASS)

        if action == GuardAction.PASS:
            explanation = "Input cleared all security probes. Fast-pass enabled." if res.fast_path_used else "Input cleared all security probes."
        elif action == GuardAction.SUPPORT:
            explanation = "Self-harm or psychological distress detected. Diverted to crisis support channels."
        elif action == GuardAction.BLOCK:
            explanation = f"Security violation detected: {res.highest_hazard or 'prohibited content'}. Execution blocked."
        else:
            explanation = f"Potential sensitive inquiry ({res.highest_hazard or 'review required'}). Queued for secondary review."

        return GuardVerdict(
            side="input",
            action=action,
            is_safe=is_safe,
            severity_score=res.severity_score,
            highest_hazard=res.highest_hazard,
            escalated=res.escalated,
            fast_path_used=res.fast_path_used,
            latency_ms=res.total_latency_ms,
            policy_applied=pol.name,
            probe_probabilities=res.battery_result.probe_probabilities,
            explanation=explanation,
        )

    def guard_output(
        self,
        response: str,
        input_context: Optional[str] = None,
        policy: Optional[Union[str, RoutingPolicy]] = None,
    ) -> GuardVerdict:
        """Inspects LLM / Agent response output before transmitting to user or downstream consumers."""
        pol = self._resolve_policy(policy)
        res: CascadedGuardResult = self.output_engine.evaluate(response, context=input_context, policy=pol)

        action = res.action
        is_safe = (action == GuardAction.PASS)

        if action == GuardAction.PASS:
            explanation = "Output cleared all security probes."
        elif action == GuardAction.SUPPORT:
            explanation = "Output contains dangerous self-harm validation. Overridden by crisis response."
        elif action == GuardAction.BLOCK:
            explanation = f"Generated output contains forbidden content ({res.highest_hazard or 'critical violation'}). Blocked from delivery."
        else:
            explanation = f"Output contains sensitive information ({res.highest_hazard or 'review'}). Suspended for review."

        return GuardVerdict(
            side="output",
            action=action,
            is_safe=is_safe,
            severity_score=res.severity_score,
            highest_hazard=res.highest_hazard,
            escalated=res.escalated,
            fast_path_used=res.fast_path_used,
            latency_ms=res.total_latency_ms,
            policy_applied=pol.name,
            probe_probabilities=res.battery_result.probe_probabilities,
            explanation=explanation,
        )
