"""Goal Verifier Protocol & Verification Middleware (Issue #13).

Provides deterministic and calibrated goal verification across multi-criteria objectives:
1. Strips subjective noise and extracts objective evidence anchors.
2. Evaluates each criterion against evidence using non-autoregressive decision model.
3. Enforces hard safety invariants (e.g. test failure clamps pass probability to 0.0).
4. Produces structured verdict with fine-grained failed_criteria attribution for narrow replanning.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from gen_zero.client import GenZero
from gen_zero.logic.boolean_engine import BooleanEngine, BooleanSemantics
from .evidence_sanitizer import ObjectiveEvidence, ObjectiveEvidenceSanitizer


class VerifierStatus(str, Enum):
    GOAL_MET = "GOAL_MET"
    NOT_YET = "NOT_YET"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass
class GoalVerificationVerdict:
    """Structured verdict output from GoalVerifier."""
    status: str
    confidence: float
    details: Dict[str, float]
    failed_criteria: List[str]
    goal: str
    sanitized_evidence: str
    timing_ms: float
    objective_facts: Optional[Dict[str, Any]] = None
    narrow_replanning_hints: List[str] = field(default_factory=list)

    @property
    def is_goal_met(self) -> bool:
        return self.status == VerifierStatus.GOAL_MET.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "details": {k: round(v, 4) for k, v in self.details.items()},
            "failed_criteria": list(self.failed_criteria),
            "goal": self.goal,
            "sanitized_evidence": self.sanitized_evidence,
            "timing_ms": round(self.timing_ms, 2),
            "objective_facts": self.objective_facts or {},
            "narrow_replanning_hints": list(self.narrow_replanning_hints)
        }


class GoalVerifier:
    """Continuous System 1 Verification Engine for Long-Horizon Agent Tasks."""

    def __init__(
        self,
        client: Optional[Any] = None,
        default_threshold: float = 0.70,
        semantics: BooleanSemantics = BooleanSemantics.ZADEH
    ):
        if client is None:
            from gen_zero.client import GenZero
            client = GenZero()
        self.client = client
        self.default_threshold = default_threshold
        self.boolean_engine = BooleanEngine(default_semantics=semantics)

    def _check_hard_invariants(
        self,
        criterion: str,
        evidence_obj: ObjectiveEvidence
    ) -> Optional[float]:
        """Checks non-negotiable hard invariants (test pass/fail, exit codes)."""
        crit_lower = criterion.lower()

        # Invariant 1: If criterion requires tests passing and tests failed or exit_code != 0
        test_related = any(k in crit_lower for k in ("test", "单测", "测试", "regression", "回归", "unittest", "pytest"))
        pass_related = any(k in crit_lower for k in ("pass", "通过", "100%", "成功", "all green", "全绿"))

        if test_related and pass_related:
            if not evidence_obj.is_success or evidence_obj.failed_tests > 0 or evidence_obj.error_tests > 0 or evidence_obj.exit_code != 0:
                return 0.0
            if evidence_obj.is_success and evidence_obj.failed_tests == 0 and evidence_obj.error_tests == 0 and (evidence_obj.passed_tests > 0 or evidence_obj.total_tests > 0):
                return 1.0

        # Invariant 2: If criterion requires zero exceptions / zero errors
        no_err_related = any(k in crit_lower for k in ("no exception", "无异常", "无新增未捕获异常", "no error", "0 error"))
        if no_err_related:
            if evidence_obj.detected_exceptions or evidence_obj.error_tests > 0 or evidence_obj.exit_code != 0:
                return 0.0
            if evidence_obj.is_success and not evidence_obj.detected_exceptions and evidence_obj.error_tests == 0:
                return 1.0

        # Invariant 3: If exit code must be 0 / clean status
        if any(k in crit_lower for k in ("exit code must be 0", "exit code 0", "退出码 0", "returncode 0")):
            return 1.0 if evidence_obj.exit_code == 0 else 0.0

        if evidence_obj.exit_code != 0 and any(k in crit_lower for k in ("success", "成功", "exit 0", "退出码 0")):
            return 0.0

        return None

    def verify_step(
        self,
        goal: str,
        criteria: List[str],
        evidence: Union[str, Dict[str, Any], ObjectiveEvidence],
        threshold: Optional[float] = None,
        explicit_exit_code: Optional[int] = None
    ) -> GoalVerificationVerdict:
        """Verifies goal satisfaction against multi-criteria evidence."""
        t0 = time.perf_counter()
        active_threshold = threshold if threshold is not None else self.default_threshold

        # 1. Sanitize evidence and extract hard facts
        if isinstance(evidence, ObjectiveEvidence):
            evidence_obj = evidence
        else:
            evidence_obj = ObjectiveEvidenceSanitizer.extract_objective_evidence(
                evidence,
                explicit_exit_code=explicit_exit_code
            )
        sanitized_str = evidence_obj.to_canonical_string()

        if not criteria:
            # Trivial satisfaction if no criteria specified
            timing_ms = (time.perf_counter() - t0) * 1000.0
            return GoalVerificationVerdict(
                status=VerifierStatus.GOAL_MET.value,
                confidence=1.0,
                details={},
                failed_criteria=[],
                goal=goal,
                sanitized_evidence=sanitized_str,
                timing_ms=timing_ms,
                objective_facts={"is_success": evidence_obj.is_success, "exit_code": evidence_obj.exit_code}
            )

        # 2. Evaluate criteria probabilities
        details: Dict[str, float] = {}
        failed_criteria: List[str] = []
        narrow_hints: List[str] = []

        # Construct batch queries for neural scoring
        queries_to_eval = []
        criterion_map = {}

        for idx, crit in enumerate(criteria):
            hard_prob = self._check_hard_invariants(crit, evidence_obj)
            if hard_prob is not None:
                details[crit] = hard_prob
                if hard_prob < active_threshold:
                    failed_criteria.append(crit)
                    narrow_hints.append(f"Hard invariant violated for criterion: '{crit}'")
            else:
                q_prompt = f"Goal: {goal}\nCriterion: {crit}\nEvidence: {sanitized_str}\nQ: Does the evidence satisfy this criterion?\nTrue: Satisfied\nFalse: Not satisfied"
                queries_to_eval.append(q_prompt)
                criterion_map[len(queries_to_eval) - 1] = crit

        # Batch evaluation via decision client
        if queries_to_eval:
            batch_res = self.client.decide_batch(
                queries_to_eval,
                candidates=["true", "false"],
                mode="reflex"
            )
            for q_idx, res in enumerate(batch_res):
                c_name = criterion_map[q_idx]
                prob = res.get("probs", {}).get("true", 0.5)
                # Bound check
                if not math.isfinite(prob):
                    prob = 0.0
                prob = max(0.0, min(1.0, float(prob)))
                details[c_name] = round(prob, 4)
                if prob < active_threshold:
                    failed_criteria.append(c_name)
                    narrow_hints.append(f"Model scored criterion '{c_name}' at P={prob:.2f} (< {active_threshold:.2f})")

        # 3. Aggregate joint confidence across all criteria (Zadeh conjunction: min(P_i))
        if details:
            joint_confidence = min(details.values())
        else:
            joint_confidence = 0.0

        is_met = (len(failed_criteria) == 0 and joint_confidence >= active_threshold)
        status = VerifierStatus.GOAL_MET.value if is_met else VerifierStatus.NOT_YET.value

        timing_ms = (time.perf_counter() - t0) * 1000.0

        return GoalVerificationVerdict(
            status=status,
            confidence=round(joint_confidence, 4),
            details=details,
            failed_criteria=failed_criteria,
            goal=goal,
            sanitized_evidence=sanitized_str,
            timing_ms=timing_ms,
            objective_facts={
                "exit_code": evidence_obj.exit_code,
                "total_tests": evidence_obj.total_tests,
                "passed_tests": evidence_obj.passed_tests,
                "failed_tests": evidence_obj.failed_tests,
                "error_tests": evidence_obj.error_tests,
                "is_success": evidence_obj.is_success,
                "exceptions": evidence_obj.detected_exceptions
            },
            narrow_replanning_hints=narrow_hints
        )
