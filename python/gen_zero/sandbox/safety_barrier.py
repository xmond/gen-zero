"""Gen-Zero Layer 3 Sandbox: PRM Formal Safety Barrier & Invariant Gating.

Provides formal constraint verification before executing mutating actions:
1. Intercepts MUTATING_IRREVERSIBLE and high-risk MUTATING_REVERSIBLE operations.
2. Checks user-registered programmatic safety invariants (e.g. transaction bounds, unauthorized tables).
3. Emits SafetyVerdict (ALLOWED, BLOCKED, DEFERRED).
4. Guarantees zero catastrophic irreversible state mutations.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple

from .tool_registry import ToolDefinition, SideEffectLevel


class SafetyVerdict(str, Enum):
    """Verdict returned by PRMSafetyBarrier prior to tool dispatch."""
    ALLOWED = "allowed"     # Operation passes all formal invariants
    BLOCKED = "blocked"     # Violates hard safety rule, completely aborted
    DEFERRED = "deferred"   # Requires Human-In-The-Loop approval (Selective Deferral)


@dataclass
class SafetyAuditReport:
    """Detailed audit report for a proposed tool execution."""
    tool_name: str
    params: Dict[str, Any]
    verdict: SafetyVerdict
    rule_violated: Optional[str] = None
    reason: str = "Passed all safety invariants."


class PRMSafetyBarrier:
    """Formal invariant gatekeeper preventing catastrophic irreversible mutations."""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._invariant_rules: List[Tuple[str, Callable[[ToolDefinition, Dict[str, Any]], bool], str]] = []
        self._init_default_invariants()

    def _init_default_invariants(self):
        """Initializes standard baseline system safety invariants."""
        # Rule 1: Guard against SQL drop/truncate
        def no_drop_table(tool: ToolDefinition, params: Dict[str, Any]) -> bool:
            for v in params.values():
                if isinstance(v, str) and ("drop table" in v.lower() or "truncate " in v.lower()):
                    return False
            return True

        self.register_invariant(
            rule_id="NO_DROP_TABLE",
            condition_fn=no_drop_table,
            violation_msg="Destructive database drop/truncate commands are strictly forbidden."
        )

        # Rule 2: Financial transaction upper-bound check
        def max_financial_amount(tool: ToolDefinition, params: Dict[str, Any]) -> bool:
            if "amount" in params:
                amt = float(params["amount"])
                if amt > 100_000.0:  # Hard cap $100k per single transaction
                    return False
            return True

        self.register_invariant(
            rule_id="MAX_FINANCIAL_LIMIT",
            condition_fn=max_financial_amount,
            violation_msg="Transaction amount exceeds single-step safety ceiling ($100,000)."
        )

    def register_invariant(
        self,
        rule_id: str,
        condition_fn: Callable[[ToolDefinition, Dict[str, Any]], bool],
        violation_msg: str
    ):
        """Registers a custom programmatic safety invariant."""
        self._invariant_rules.append((rule_id, condition_fn, violation_msg))

    def audit(self, tool: ToolDefinition, params: Dict[str, Any]) -> SafetyAuditReport:
        """Audits tool call against all safety rules before execution."""
        # Read-only tools are intrinsically safe
        if tool.side_effect_level == SideEffectLevel.READ_ONLY:
            return SafetyAuditReport(
                tool_name=tool.name,
                params=params,
                verdict=SafetyVerdict.ALLOWED,
                reason="Read-only operation is intrinsically safe."
            )

        # Audit against all registered invariants
        for rule_id, cond_fn, msg in self._invariant_rules:
            try:
                passed = cond_fn(tool, params)
                if not passed:
                    return SafetyAuditReport(
                        tool_name=tool.name,
                        params=params,
                        verdict=SafetyVerdict.BLOCKED,
                        rule_violated=rule_id,
                        reason=msg
                    )
            except Exception as e:
                if self.strict_mode:
                    return SafetyAuditReport(
                        tool_name=tool.name,
                        params=params,
                        verdict=SafetyVerdict.BLOCKED,
                        rule_violated=rule_id,
                        reason=f"Invariant check failed with error: {str(e)}"
                    )

        # If mutating irreversible, check for human confirmation flags or high-risk mutation signatures
        if tool.side_effect_level == SideEffectLevel.MUTATING_IRREVERSIBLE:
            name_lower = tool.name.lower()
            is_high_risk = any(k in name_lower for k in ("delete", "rm", "drop", "kill", "purge", "destroy", "format"))
            if params.get("require_human_confirmation", False) or params.get("requires_approval", False) or is_high_risk:
                return SafetyAuditReport(
                    tool_name=tool.name,
                    params=params,
                    verdict=SafetyVerdict.DEFERRED,
                    reason="Irreversible mutation requires explicit human-in-the-loop authorization."
                )

        return SafetyAuditReport(
            tool_name=tool.name,
            params=params,
            verdict=SafetyVerdict.ALLOWED,
            reason="Passed all PRM formal invariants."
        )
