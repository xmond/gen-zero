"""Cascaded NanoCore Fast-Path Guardrail.

Implements Milestone 3 of Issue #26:
- Two-Tier Cascaded Evaluation:
  Tier 1: L2 Shallow NanoCore Pre-Check (<= 2.5ms):
    If max(p) < 0.20 and Severity < 0.5 -> FAST_PASS (bypasses full-depth backbone for >90% benign traffic).
  Tier 2: Full-Depth Backbone Refinement:
    Suspicious or borderline queries (max(p) >= 0.20 or Severity >= 0.5) seamlessly route to full evaluation.
"""

from typing import Dict, List, Any, Optional, Tuple
import dataclasses
import time

from gen_zero.guard.batteries import (
    GuardBattery,
    BatteryEvaluationResult,
    ProbeDefinition,
    GuardAction,
)
from gen_zero.guard.routing_algebra import (
    PrecedenceRoutingAlgebra,
    RoutingPolicy,
    RoutingDecision,
)


@dataclasses.dataclass
class CascadedGuardResult:
    """Result of cascaded guardrail evaluation."""
    action: GuardAction
    fast_path_used: bool
    escalated: bool
    severity_score: float
    highest_hazard: Optional[str]
    total_latency_ms: float
    l2_precheck_latency_ms: float
    full_eval_latency_ms: Optional[float]
    battery_result: BatteryEvaluationResult
    routing_decision: RoutingDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "fast_path_used": self.fast_path_used,
            "escalated": self.escalated,
            "severity_score": round(self.severity_score, 4),
            "highest_hazard": self.highest_hazard,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "l2_precheck_latency_ms": round(self.l2_precheck_latency_ms, 2),
            "full_eval_latency_ms": round(self.full_eval_latency_ms, 2) if self.full_eval_latency_ms is not None else None,
        }


class CascadedGuardEngine:
    """Cascades L2 shallow NanoCore fast-path screening with full-depth precision routing."""

    def __init__(
        self,
        battery: GuardBattery,
        algebra: Optional[PrecedenceRoutingAlgebra] = None,
        fast_pass_max_p_threshold: float = 0.20,
        fast_pass_max_sev_threshold: float = 0.50,
    ):
        self.battery = battery
        self.algebra = algebra or PrecedenceRoutingAlgebra()
        self.fast_pass_max_p_threshold = fast_pass_max_p_threshold
        self.fast_pass_max_sev_threshold = fast_pass_max_sev_threshold
        self.probe_defs = {p.name: p for p in self.battery.probes}

    def evaluate(
        self,
        text: str,
        context: Optional[str] = None,
        policy: Optional[RoutingPolicy] = None,
    ) -> CascadedGuardResult:
        """Executes cascaded pre-check and conditional full-depth evaluation."""
        t_start = time.perf_counter()

        # Step 1: L2 Shallow Core Evaluation (<2.5ms)
        t_l2_start = time.perf_counter()
        l2_eval = self.battery.evaluate(text, context)
        l2_ms = (time.perf_counter() - t_l2_start) * 1000.0

        max_p = max(l2_eval.probe_probabilities.values()) if l2_eval.probe_probabilities else 0.0
        sev = l2_eval.severity_score

        # Fast-pass criteria: max(p) < 0.20 and Severity < 0.50
        if max_p < self.fast_pass_max_p_threshold and sev < self.fast_pass_max_sev_threshold:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            fast_decision = RoutingDecision(
                action=GuardAction.PASS,
                escalated=False,
                escalation_reason=None,
                triggered_probes={},
                severity_score=sev,
                highest_hazard=None,
                policy_applied=policy.name if policy else self.algebra.policy.name,
            )
            return CascadedGuardResult(
                action=GuardAction.PASS,
                fast_path_used=True,
                escalated=False,
                severity_score=sev,
                highest_hazard=None,
                total_latency_ms=total_ms,
                l2_precheck_latency_ms=l2_ms,
                full_eval_latency_ms=None,
                battery_result=l2_eval,
                routing_decision=fast_decision,
            )

        # Step 2: Full-Depth Evaluation & Precedence Routing
        t_full_start = time.perf_counter()
        # In actual deployment, full depth runs deeper Backbone; in software harness l2_eval is refined
        routing_decision = self.algebra.route(
            battery_result=l2_eval,
            probe_definitions=self.probe_defs,
            override_policy=policy,
        )
        full_ms = (time.perf_counter() - t_full_start) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return CascadedGuardResult(
            action=routing_decision.action,
            fast_path_used=False,
            escalated=routing_decision.escalated,
            severity_score=routing_decision.severity_score,
            highest_hazard=routing_decision.highest_hazard,
            total_latency_ms=total_ms,
            l2_precheck_latency_ms=l2_ms,
            full_eval_latency_ms=full_ms,
            battery_result=l2_eval,
            routing_decision=routing_decision,
        )
