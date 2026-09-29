"""Architecture Contract Drift Gate.

Implements Milestone 4 of Issue #20:
1. Specification Injection:
   Ingests architectural constraints and safety rules from AGENTS.md and project rules.
2. Micro-Level Anti-Vandalism Circuit Breaker:
   Blocks workflow execution and commits if contract_drift >= 0.75.
3. Pinpointed Diagnostic Feedback:
   Highlights exact violating file paths and breached architectural invariants.
"""

from typing import Dict, List, Any, Optional, Tuple
import dataclasses
import re

from gen_zero.runtime.supervisor_observation import FactoryObservation
from gen_zero.runtime.supervisor_assessment import SupervisorAssessment


@dataclasses.dataclass
class ContractGateVerdict:
    """Verdict rendered by the Contract Drift Gate."""
    passed: bool
    contract_drift: float
    threshold: float = 0.75
    violating_rules: List[str] = dataclasses.field(default_factory=list)
    violating_files: List[str] = dataclasses.field(default_factory=list)
    verdict_status: str = "PERMITTED"
    explanation: str = "All architectural contract invariants satisfied."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "contract_drift": round(self.contract_drift, 4),
            "threshold": self.threshold,
            "violating_rules": self.violating_rules,
            "violating_files": self.violating_files,
            "verdict_status": self.verdict_status,
            "explanation": self.explanation,
        }


class ContractDriftGate:
    """Guards repository against architecture contract breaches and architectural drift."""

    def __init__(
        self,
        drift_threshold: float = 0.75,
        default_rules: Optional[List[str]] = None,
    ):
        self.drift_threshold = drift_threshold
        self.default_rules = default_rules or [
            "forbidden: .git/",
            "forbidden: /etc/",
            "do not edit: .env",
            "forbidden: rm -rf",
        ]

    def evaluate(
        self,
        observation: FactoryObservation,
        assessment: Optional[SupervisorAssessment] = None,
    ) -> ContractGateVerdict:
        """Evaluates whether the workspace state breaches contract rules."""
        active_rules = list(self.default_rules) + observation.contract_rules
        violating_rules = []
        violating_files = []

        all_modified = (
            observation.git_status.get("modified", [])
            + observation.git_status.get("untracked", [])
            + observation.git_status.get("deleted", [])
        )
        diff_lower = observation.git_diff.lower()

        # Check explicit rules
        for rule in active_rules:
            rule_lower = rule.lower()
            if "forbidden:" in rule_lower:
                pattern = rule_lower.split("forbidden:", 1)[1].strip()
                # Check path match
                for path in all_modified:
                    if pattern in path.lower():
                        violating_rules.append(rule)
                        violating_files.append(path)
                # Check content in diff
                if pattern in diff_lower and pattern not in violating_rules:
                    violating_rules.append(f"Diff content violated: {rule}")

            elif "do not edit:" in rule_lower:
                pattern = rule_lower.split("do not edit:", 1)[1].strip()
                for path in all_modified:
                    if pattern in path.lower():
                        violating_rules.append(rule)
                        violating_files.append(path)

        # Deduplicate
        violating_rules = sorted(list(set(violating_rules)))
        violating_files = sorted(list(set(violating_files)))

        # Determine drift score
        if assessment is not None:
            drift_score = assessment.contract_drift
        else:
            drift_score = 0.90 if violating_rules else 0.05

        if violating_rules and drift_score < self.drift_threshold:
            drift_score = max(drift_score, 0.85)

        passed = drift_score < self.drift_threshold
        if passed:
            verdict_status = "PERMITTED"
            explanation = f"Contract check passed (drift={drift_score:.2f} < {self.drift_threshold:.2f})."
        else:
            verdict_status = "BLOCKED_CONTRACT_DRIFT"
            explanation = (
                f"Blocked: Contract drift {drift_score:.2f} >= {self.drift_threshold:.2f}. "
                f"Violations: {', '.join(violating_rules) if violating_rules else 'High semantic contract drift detected'}."
            )

        return ContractGateVerdict(
            passed=passed,
            contract_drift=drift_score,
            threshold=self.drift_threshold,
            violating_rules=violating_rules,
            violating_files=violating_files,
            verdict_status=verdict_status,
            explanation=explanation,
        )
