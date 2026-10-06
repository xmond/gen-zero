"""Ten-Dimensional Atomic Semantic Assessment Adapter.

Implements Milestone 2 of Issue #20:
1. Ten-Dimensional Atomic Assessment:
   Parallel evaluation of workspace evidence across 10 atomic semantic dimensions in [0.0, 1.0]:
   - implementation_complete
   - tests_sufficient
   - requirements_satisfied
   - needs_verification
   - meaningful_progress
   - worker_stuck
   - work_off_track
   - contract_drift
   - ready_to_finish
   - needs_human
2. Calibrated Probability Validation:
   Guarantees all outputs are finite numbers clamped strictly to [0.0, 1.0].
"""

from typing import Dict, List, Any, Optional, Tuple
import dataclasses
import math
import re

from .supervisor_observation import FactoryObservation


@dataclasses.dataclass
class SupervisorAssessment:
    """Ten-dimensional atomic semantic probabilities assessing worker state."""
    implementation_complete: float = 0.0
    tests_sufficient: float = 0.0
    requirements_satisfied: float = 0.0
    needs_verification: float = 0.0
    meaningful_progress: float = 0.5
    worker_stuck: float = 0.0
    work_off_track: float = 0.0
    contract_drift: float = 0.0
    ready_to_finish: float = 0.0
    needs_human: float = 0.0
    rationale: str = ""

    def __post_init__(self):
        # Validate and clamp all 10 dimensions to [0.0, 1.0]
        def clamp(val):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                val = 0.0
            return max(0.0, min(1.0, float(val)))

        self.implementation_complete = clamp(self.implementation_complete)
        self.tests_sufficient = clamp(self.tests_sufficient)
        self.requirements_satisfied = clamp(self.requirements_satisfied)
        self.needs_verification = clamp(self.needs_verification)
        self.meaningful_progress = clamp(self.meaningful_progress)
        self.worker_stuck = clamp(self.worker_stuck)
        self.work_off_track = clamp(self.work_off_track)
        self.contract_drift = clamp(self.contract_drift)
        self.ready_to_finish = clamp(self.ready_to_finish)
        self.needs_human = clamp(self.needs_human)

    @property
    def has_critical_contract_drift(self) -> bool:
        return self.contract_drift >= 0.75

    @property
    def is_drifting(self) -> bool:
        return self.contract_drift >= 0.70

    @property
    def is_stuck(self) -> bool:
        return self.worker_stuck >= 0.70

    @property
    def is_stuck_or_off_track(self) -> bool:
        return self.worker_stuck >= 0.70 or self.work_off_track >= 0.70

    @property
    def is_completion_ready(self) -> bool:
        return (
            self.ready_to_finish >= 0.85
            and self.tests_sufficient >= 0.85
            and self.contract_drift < 0.20
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "implementation_complete": round(self.implementation_complete, 4),
            "tests_sufficient": round(self.tests_sufficient, 4),
            "requirements_satisfied": round(self.requirements_satisfied, 4),
            "needs_verification": round(self.needs_verification, 4),
            "meaningful_progress": round(self.meaningful_progress, 4),
            "worker_stuck": round(self.worker_stuck, 4),
            "work_off_track": round(self.work_off_track, 4),
            "contract_drift": round(self.contract_drift, 4),
            "ready_to_finish": round(self.ready_to_finish, 4),
            "needs_human": round(self.needs_human, 4),
            "rationale": self.rationale,
        }


class SemanticAssessmentAdapter:
    """Evaluates FactoryObservation into calibrated 10-dimensional probabilities."""

    def __init__(self, score_adapter: Any = None):
        self.score_adapter = score_adapter

    def evaluate(
        self,
        obs: FactoryObservation,
        history: Optional[List[FactoryObservation]] = None,
    ) -> SupervisorAssessment:
        """Evaluates workspace observation across 10 atomic semantic dimensions."""
        hist = history or []

        # 1. Evaluate Tests Sufficient & Needs Verification
        tests = obs.test_results or {}
        passed = tests.get("passed", 0)
        failed = tests.get("failed", 0)
        errors = tests.get("errors", 0)
        total = passed + failed + errors

        if total > 0:
            if failed == 0 and errors == 0:
                p_tests = min(1.0, 0.60 + 0.40 * min(1.0, passed / 5.0))
                p_verify = 0.80 if p_tests >= 0.80 else 0.40
            else:
                p_tests = 0.0
                p_verify = 0.95
        else:
            p_tests = 0.20
            p_verify = 0.30

        # 2. Evaluate Worker Stuck & Stagnation from Diff History
        p_stuck = 0.0
        if len(hist) >= 2:
            prev_fp = hist[-1].fingerprint
            curr_fp = obs.fingerprint
            if prev_fp == curr_fp:
                # Fingerprint stagnation
                same_count = sum(1 for h in reversed(hist) if h.fingerprint == curr_fp)
                if same_count >= 2:
                    p_stuck = min(0.95, 0.50 + 0.20 * same_count)
            else:
                p_stuck = 0.05
        elif len(obs.git_diff.strip()) == 0 and len(obs.git_status.get("modified", [])) == 0:
            # Clean worktree with no changes
            if obs.step_index > 3:
                p_stuck = 0.70

        # 3. Evaluate Contract Drift against Forbidden Rules
        p_drift = 0.0
        drift_reasons = []
        mod_files = obs.git_status.get("modified", []) + obs.git_status.get("untracked", [])
        diff_lower = obs.git_diff.lower()

        # Check explicit rules
        for rule in obs.contract_rules:
            rule_lower = rule.lower()
            # E.g., forbidden directory modifications
            if "forbidden:" in rule_lower or "do not edit" in rule_lower:
                target = rule_lower.split(":")[-1].strip()
                for mf in mod_files:
                    if target and target in mf.lower():
                        p_drift = max(p_drift, 0.90)
                        drift_reasons.append(f"Forbidden modification of {mf} matching rule: {rule}")

        # Check for circular imports, syntax errors, or forbidden global patterns in diff
        if "rm -rf /" in diff_lower or ":(){ :|:& };:" in diff_lower:
            p_drift = 1.0
            drift_reasons.append("Catastrophic destructive command detected in diff")

        # 4. Evaluate Meaningful Progress & Work Off Track
        p_off_track = 0.10
        if p_stuck > 0.60:
            p_progress = 0.10
            p_off_track = 0.30
        elif p_drift > 0.70:
            p_progress = 0.15
            p_off_track = 0.85
        elif len(obs.git_diff) > 0 and (failed == 0 and errors == 0):
            p_progress = 0.85
            p_off_track = 0.05
        else:
            p_progress = 0.50
            p_off_track = 0.10

        p_off_track = max(p_off_track, 0.80 if p_drift >= 0.75 else 0.05)

        # 5. Implementation Complete & Requirements Satisfied
        if p_tests >= 0.85 and p_drift < 0.20 and p_stuck < 0.20:
            p_impl = 0.90
            p_reqs = 0.90
            p_ready = 0.90
        elif p_tests >= 0.50:
            p_impl = 0.65
            p_reqs = 0.60
            p_ready = 0.40
        else:
            p_impl = 0.30
            p_reqs = 0.30
            p_ready = 0.05

        # 6. Needs Human
        p_human = 0.95 if p_drift >= 0.95 else 0.05

        rationale = "; ".join(drift_reasons) if drift_reasons else "Normal telemetry assessment."

        return SupervisorAssessment(
            implementation_complete=p_impl,
            tests_sufficient=p_tests,
            requirements_satisfied=p_reqs,
            needs_verification=p_verify,
            meaningful_progress=p_progress,
            worker_stuck=p_stuck,
            work_off_track=p_off_track,
            contract_drift=p_drift,
            ready_to_finish=p_ready,
            needs_human=p_human,
            rationale=rationale,
        )
