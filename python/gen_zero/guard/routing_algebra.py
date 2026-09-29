"""Precedence Routing Algebra & Severity Dynamic Escalation.

Implements Milestone 2 of Issue #26:
- Four-level precedence arbitration stack: PRECEDENCE = [support, block, review, pass]
- Triggered action determination:
  a_i = Act(h_i) if p(h_i) >= tau_action
  a_i = review   if tau_review <= p(h_i) < tau_action
  a_i = empty    if p(h_i) < tau_review
- Severity Escalation Operator:
  if a_i == review and Severity >= tau_sev_block -> a_i = block
- Global Decision:
  Decision = argmax_Precedence ({a_i} U {pass})
- Decoupled named policies: 'strict', 'permissive', or custom RoutingPolicy.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import dataclasses

from gen_zero.guard.batteries import GuardAction, BatteryEvaluationResult, ProbeDefinition


@dataclasses.dataclass
class RoutingPolicy:
    """Configurable routing thresholds for business logic decoupling."""
    name: str = "strict"
    tau_review: float = 0.35
    tau_action: float = 0.70
    tau_sev_block: float = 2.0

    @classmethod
    def strict(cls) -> "RoutingPolicy":
        return cls(name="strict", tau_review=0.35, tau_action=0.70, tau_sev_block=2.0)

    @classmethod
    def permissive(cls) -> "RoutingPolicy":
        return cls(name="permissive", tau_review=0.35, tau_action=0.85, tau_sev_block=2.0)


# Precedence order: index 0 has highest precedence
PRECEDENCE_ORDER: List[GuardAction] = [
    GuardAction.SUPPORT,  # Rank 0 (highest)
    GuardAction.BLOCK,    # Rank 1
    GuardAction.REVIEW,   # Rank 2
    GuardAction.PASS,     # Rank 3 (lowest)
]

PRECEDENCE_RANK: Dict[GuardAction, int] = {
    action: rank for rank, action in enumerate(PRECEDENCE_ORDER)
}


@dataclasses.dataclass
class RoutingDecision:
    """Final routing decision produced by the algebra."""
    action: GuardAction
    escalated: bool
    escalation_reason: Optional[str]
    triggered_probes: Dict[str, GuardAction]
    severity_score: float
    highest_hazard: Optional[str]
    policy_applied: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "triggered_probes": {k: v.value for k, v in self.triggered_probes.items()},
            "severity_score": round(self.severity_score, 4),
            "highest_hazard": self.highest_hazard,
            "policy_applied": self.policy_applied,
        }


class PrecedenceRoutingAlgebra:
    """Evaluates four-tier precedence algebra and severity escalation over battery results."""

    def __init__(self, policy: Optional[RoutingPolicy] = None):
        self.policy = policy or RoutingPolicy.strict()

    def route(
        self,
        battery_result: BatteryEvaluationResult,
        probe_definitions: Dict[str, ProbeDefinition],
        override_policy: Optional[RoutingPolicy] = None,
    ) -> RoutingDecision:
        """Computes global decision via Precedence Hierarchy and Severity Escalation.

        1. For each probe h_i with probability p_i:
           - if p_i >= tau_action: a_i = probe.default_action
           - if tau_review <= p_i < tau_action: a_i = REVIEW
           - if p_i < tau_review: None
        2. Escalation:
           - if a_i == REVIEW and Severity >= tau_sev_block: a_i = BLOCK
        3. Global decision:
           - argmin rank (highest priority) in triggered actions U {PASS}
        """
        pol = override_policy or self.policy
        probs = battery_result.probe_probabilities
        sev = battery_result.severity_score

        triggered_actions: Dict[str, GuardAction] = {}
        escalated = False
        escalation_reason = None

        for probe_name, p_val in probs.items():
            probe_def = probe_definitions.get(probe_name)
            default_act = probe_def.default_action if probe_def else GuardAction.BLOCK

            # 1. Base threshold mapping
            if p_val >= pol.tau_action:
                act = default_act
            elif p_val >= pol.tau_review:
                act = GuardAction.REVIEW
            else:
                act = None

            # 2. Severity Escalation Operator
            # When severity exceeds tau_sev_block, escalate review -> block
            if act == GuardAction.REVIEW and sev >= pol.tau_sev_block:
                act = GuardAction.BLOCK
                escalated = True
                escalation_reason = f"Severity ({sev:.2f}) >= tau_sev_block ({pol.tau_sev_block:.2f}) triggered review -> block escalation."

            if act is not None:
                triggered_actions[probe_name] = act

        # 3. Global Precedence Arbitration
        if not triggered_actions:
            final_action = GuardAction.PASS
            highest_hazard = None
        else:
            # Pick action with minimum rank index (highest precedence)
            final_action = min(triggered_actions.values(), key=lambda a: PRECEDENCE_RANK[a])
            # Find probe responsible for this action
            highest_hazard = max(
                (name for name, act in triggered_actions.items() if act == final_action),
                key=lambda name: probs[name]
            )

        return RoutingDecision(
            action=final_action,
            escalated=escalated,
            escalation_reason=escalation_reason,
            triggered_probes=triggered_actions,
            severity_score=sev,
            highest_hazard=highest_hazard,
            policy_applied=pol.name,
        )
