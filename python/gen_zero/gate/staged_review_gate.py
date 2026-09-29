"""Staged Code-Review Funnel and Semantic Risk Gate.

Implements Issues #22:
- 6-Stage Hierarchical Review Funnel:
  Stage 1: 5-D Risk Matrix Screening (Screen threshold: 0.70, sub-10ms bypass).
  Stage 2: Change Profiling & Review Priority (0~3 scale).
  Stage 3: Evidence Hunk Selection with explicit noMatch and min confidence 0.55 cutoff.
  Stage 4: Structured Mechanism Classification with explicit noIssue rejection.
  Stage 5: Production Impact Severity Scoring (0~3 scale).
  Stage 6: Admission Action & Domain Routing (Severity >= 1.5 routes to owner, >= 2.0 blocks merge).
- Dual-Context evaluation for testGap dimension.
"""

from typing import Dict, List, Any, Optional, Tuple
import dataclasses
import time
import math
import re

from gen_zero.gate.diff_parser import DiffHunk, FileDiff, DualContext, DualContextCollator, DiffHunkParser
from gen_zero.gate.review_taxonomy import (
    RiskDimension,
    ChangeProfile,
    ReviewAction,
    OwnerDomain,
    MECHANISM_TAXONOMY,
    NO_MATCH,
    NO_ISSUE,
    map_dimension_to_owner,
)


@dataclasses.dataclass
class ReviewIssue:
    """Represents a validated, localized, and classified code review defect."""
    dimension: str
    profile: str
    review_priority: float
    hunk_id: str
    location_confidence: float
    mechanism: str
    severity: float
    action: str
    routed_owner: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "profile": self.profile,
            "review_priority": round(self.review_priority, 2),
            "hunk_id": self.hunk_id,
            "location_confidence": round(self.location_confidence, 4),
            "mechanism": self.mechanism,
            "severity": round(self.severity, 2),
            "action": self.action,
            "routed_owner": self.routed_owner,
            "explanation": self.explanation,
        }


@dataclasses.dataclass
class ReviewGateVerdict:
    """Verdict emitted by the 6-stage semantic code review gate."""
    passed: bool
    screening_scores: Dict[str, float]
    profile: str
    priority: float
    issues: List[ReviewIssue]
    blocking_issues: List[ReviewIssue]
    action: str
    routed_owners: Dict[str, List[Dict[str, Any]]]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "screening_scores": {k: round(v, 4) for k, v in self.screening_scores.items()},
            "profile": self.profile,
            "priority": round(self.priority, 2),
            "issues": [i.to_dict() for i in self.issues],
            "blocking_issues": [i.to_dict() for i in self.blocking_issues],
            "action": self.action,
            "routed_owners": self.routed_owners,
            "latency_ms": round(self.latency_ms, 2),
        }


class StagedReviewGate:
    """Evaluates git diffs through a 6-stage discrete non-autoregressive review funnel."""

    def __init__(
        self,
        screen_threshold: float = 0.70,
        min_location_confidence: float = 0.55,
        owner_routing_severity: float = 1.5,
        blocking_severity: float = 2.0,
        score_adapter: Any = None,
    ):
        self.screen_threshold = screen_threshold
        self.min_location_confidence = min_location_confidence
        self.owner_routing_severity = owner_routing_severity
        self.blocking_severity = blocking_severity
        self.score_adapter = score_adapter

    def review(self, diff_text: str, custom_rules: Optional[List[str]] = None) -> ReviewGateVerdict:
        """Executes full 6-stage semantic code review on given diff text."""
        start_time = time.monotonic()

        # Step 0: Parse Diff and Dual Context
        dual_ctx = DualContextCollator.collate(diff_text)
        total_hunks = len(dual_ctx.all_hunks)

        # Handle trivial / empty diffs
        if not diff_text.strip() or total_hunks == 0:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            return ReviewGateVerdict(
                passed=True,
                screening_scores={d.value: 0.0 for d in RiskDimension},
                profile=ChangeProfile.ROUTINE.value,
                priority=0.0,
                issues=[],
                blocking_issues=[],
                action=ReviewAction.APPROVE.value,
                routed_owners={},
                latency_ms=elapsed_ms,
            )

        # Stage 1: Five-Dimension Risk Screening
        screening_scores = self._stage_1_screen(dual_ctx)
        active_signals = [
            dim for dim, score in screening_scores.items()
            if score >= self.screen_threshold
        ]

        # Stage 1 Fast-Path Bypass: 90% harmless changes pass in sub-10ms
        if not active_signals:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            return ReviewGateVerdict(
                passed=True,
                screening_scores=screening_scores,
                profile=ChangeProfile.ROUTINE.value,
                priority=0.2,
                issues=[],
                blocking_issues=[],
                action=ReviewAction.APPROVE.value,
                routed_owners={},
                latency_ms=elapsed_ms,
            )

        # Stage 2: Profiling & Review Priority
        profile, priority = self._stage_2_profile(dual_ctx, active_signals)

        # Stage 3 ~ Stage 6: Process each active signal through Hunk Selection, Mechanism, Severity & Routing
        validated_issues: List[ReviewIssue] = []

        for signal_dim in active_signals:
            # Stage 3: Hunk Selection with noMatch
            hunk, location_conf = self._stage_3_select_hunk(signal_dim, dual_ctx)

            # Rejection Gate 1: Model selected noMatch or confidence below threshold
            if hunk is None or hunk.hunk_id == NO_MATCH or location_conf < self.min_location_confidence:
                continue

            # Stage 4: Mechanism Classification with noIssue
            mechanism, mech_conf = self._stage_4_classify_mechanism(signal_dim, hunk, dual_ctx)

            # Rejection Gate 2: Model selected noIssue or other ungrounded response
            if mechanism == NO_ISSUE:
                continue

            # Stage 5: Severity Scoring (0 ~ 3 scale)
            severity = self._stage_5_score_severity(signal_dim, mechanism, hunk, dual_ctx)

            # Stage 6: Action & Routing
            owner = map_dimension_to_owner(signal_dim, mechanism).value
            issue_action = (
                ReviewAction.REQUEST_CHANGES.value
                if severity >= self.blocking_severity
                else ReviewAction.COMMENT.value
            )

            explanation = (
                f"[{signal_dim.upper()} / {mechanism}] Detected in {hunk.hunk_id} "
                f"(severity: {severity:.1f}, conf: {location_conf:.2f})."
            )

            validated_issues.append(ReviewIssue(
                dimension=signal_dim,
                profile=profile.value,
                review_priority=priority,
                hunk_id=hunk.hunk_id,
                location_confidence=location_conf,
                mechanism=mechanism,
                severity=severity,
                action=issue_action,
                routed_owner=owner,
                explanation=explanation,
            ))

        # Determine Gate Verdict
        blocking = [i for i in validated_issues if i.severity >= self.blocking_severity]
        passed = (len(blocking) == 0)

        top_action = ReviewAction.APPROVE.value
        if blocking:
            top_action = ReviewAction.REQUEST_CHANGES.value
        elif validated_issues:
            top_action = ReviewAction.COMMENT.value

        # Group routed owners for issues with severity >= owner_routing_severity (1.5)
        routed_owners: Dict[str, List[Dict[str, Any]]] = {}
        for iss in validated_issues:
            if iss.severity >= self.owner_routing_severity:
                routed_owners.setdefault(iss.routed_owner, []).append(iss.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return ReviewGateVerdict(
            passed=passed,
            screening_scores=screening_scores,
            profile=profile.value,
            priority=priority,
            issues=validated_issues,
            blocking_issues=blocking,
            action=top_action,
            routed_owners=routed_owners,
            latency_ms=elapsed_ms,
        )

    # -------------------------------------------------------------------------
    # Stage 1: Screening
    # -------------------------------------------------------------------------
    def _stage_1_screen(self, ctx: DualContext) -> Dict[str, float]:
        """Parallel 5-D screening risk matrix evaluation."""
        diff_text = ctx.file_patch.lower()
        test_text = ctx.changed_tests.lower()

        scores: Dict[str, float] = {d.value: 0.05 for d in RiskDimension}

        # 1. Correctness indicators
        correctness_patterns = [
            r"\bif\s+.*\bnone\b.*:", r"\bwhile\s+true\b", r"\breturn\s+none\b",
            r"\boffbyone\b", r"\bindexerror\b", r"\bkeyerror\b",
            r"\bTODO\b", r"\bFIXME\b", r"\braise\s+notimplementederror\b"
        ]
        corr_hits = sum(1 for p in correctness_patterns if re.search(p, diff_text, re.IGNORECASE))
        if corr_hits > 0:
            scores[RiskDimension.CORRECTNESS.value] = min(0.95, 0.45 + 0.25 * corr_hits)

        # 2. Security indicators
        security_patterns = [
            r"eval\(", r"exec\(", r"os\.system\(", r"subprocess\.call\(",
            r"chmod\s+777", r"password\s*=\s*['\"][^'\"]+", r"token\s*=\s*['\"][^'\"]+",
            r"select\s+.*\s+from\s+.*%", r"shell\s*=\s*true", r"verify\s*=\s*false",
            r"rm\s+-rf\s+/"
        ]
        sec_hits = sum(1 for p in security_patterns if re.search(p, diff_text, re.IGNORECASE))
        if sec_hits > 0:
            scores[RiskDimension.SECURITY.value] = min(0.98, 0.60 + 0.25 * sec_hits)

        # 3. Reliability indicators
        reliability_patterns = [
            r"\bexcept:\s*pass\b", r"\bthreading\.lock\b", r"\btime\.sleep\([5-9]\d*\)",
            r"\bopen\(", r"\bclose\(\)", r"\bwhile\s+not\b", r"\bnonlocal\b",
            r"\brecursionerror\b", r"\bmemory\b"
        ]
        rel_hits = sum(1 for p in reliability_patterns if re.search(p, diff_text, re.IGNORECASE))
        if rel_hits > 0:
            scores[RiskDimension.RELIABILITY.value] = min(0.95, 0.50 + 0.25 * rel_hits)

        # 4. Compatibility indicators
        compat_patterns = [
            r"\bdef\s+[a-zA-Z0-9_]+\([^)]*\):",  # API signature change
            r"\bclass\s+[a-zA-Z0-9_]+",
            r"@deprecated", r"schema_version", r"\bprotocol\b"
        ]
        comp_hits = sum(1 for p in compat_patterns if re.search(p, diff_text, re.IGNORECASE))
        if comp_hits > 0:
            scores[RiskDimension.COMPATIBILITY.value] = min(0.90, 0.50 + 0.20 * comp_hits)

        # 5. Dual-Context testGap indicators:
        # If business files changed significant logic but changed_tests is empty or lacking assertions
        if ctx.business_files and len(diff_text.strip()) > 100:
            if not ctx.test_files or len(test_text.strip()) == 0:
                scores[RiskDimension.TEST_GAP.value] = 0.88
            else:
                # Tests changed, but check assertion density
                assert_count = len(re.findall(r"\bassert[a-zA-Z0-9_]*\(", test_text))
                new_funcs = len(re.findall(r"\+.*def\s+[a-zA-Z0-9_]+", ctx.file_patch))
                if new_funcs > 0 and assert_count < new_funcs:
                    scores[RiskDimension.TEST_GAP.value] = 0.78
                else:
                    scores[RiskDimension.TEST_GAP.value] = 0.20

        return scores

    # -------------------------------------------------------------------------
    # Stage 2: Profiling & Priority
    # -------------------------------------------------------------------------
    def _stage_2_profile(self, ctx: DualContext, active_signals: List[str]) -> Tuple[ChangeProfile, float]:
        """Categorizes change profile and computes review priority (0~3 scale)."""
        diff_text = ctx.file_patch.lower()

        if any(f.startswith(".github/") or f.endswith(".yml") or "docker" in f for f in ctx.business_files):
            profile = ChangeProfile.INFRA
        elif any("logging" in diff_text or "telemetry" in diff_text or "metric" in diff_text for _ in [1]):
            profile = ChangeProfile.OBSERVABILITY
        elif RiskDimension.COMPATIBILITY.value in active_signals or "def " in diff_text:
            profile = ChangeProfile.INTERFACE
        elif RiskDimension.CORRECTNESS.value in active_signals or RiskDimension.SECURITY.value in active_signals:
            profile = ChangeProfile.BEHAVIOR
        elif "refactor" in diff_text:
            profile = ChangeProfile.REFACTOR
        else:
            profile = ChangeProfile.ROUTINE

        # Base priority from active risk count and lines
        line_count = len(ctx.file_patch.splitlines())
        priority = min(3.0, 0.5 * len(active_signals) + (1.0 if line_count > 100 else 0.3))
        if RiskDimension.SECURITY.value in active_signals:
            priority = max(priority, 2.5)

        return profile, priority

    # -------------------------------------------------------------------------
    # Stage 3: Hunk Selection with noMatch
    # -------------------------------------------------------------------------
    def _stage_3_select_hunk(self, dimension: str, ctx: DualContext) -> Tuple[Optional[DiffHunk], float]:
        """Selects the candidate hunk supporting the concern, with explicit noMatch sentinel."""
        if not ctx.all_hunks:
            return None, 0.0

        # Score candidate hunks against the dimension
        hunk_scores: List[Tuple[Optional[DiffHunk], float]] = []

        for hunk in ctx.all_hunks:
            content_lower = hunk.content.lower()
            score = 0.10

            if dimension == RiskDimension.SECURITY.value:
                if any(w in content_lower for w in ["eval", "exec", "os.system", "password", "token", "rm -rf", "chmod"]):
                    score = 0.95
                elif "auth" in content_lower or "secret" in content_lower:
                    score = 0.80
            elif dimension == RiskDimension.CORRECTNESS.value:
                if any(w in content_lower for w in ["none", "while true", "indexerror", "keyerror", "todo"]):
                    score = 0.85
                elif "+" in hunk.content:
                    score = 0.50
            elif dimension == RiskDimension.RELIABILITY.value:
                if any(w in content_lower for w in ["except:", "lock", "sleep", "open(", "close()"]):
                    score = 0.90
            elif dimension == RiskDimension.COMPATIBILITY.value:
                if any(w in content_lower for w in ["def ", "class ", "deprecated", "schema"]):
                    score = 0.85
            elif dimension == RiskDimension.TEST_GAP.value:
                # For testGap, we point to the business hunk lacking test coverage
                if not hunk.file_path.endswith("_test.py") and "test" not in hunk.file_path:
                    if len(hunk.added_lines) > 3:
                        score = 0.80

            hunk_scores.append((hunk, score))

        # Add noMatch candidate with a baseline ambiguity threshold (0.45)
        # If no hunk scores well, noMatch wins!
        best_hunk, best_score = max(hunk_scores, key=lambda x: x[1])

        # If best hunk score is below 0.50, noMatch dominates
        if best_score < 0.50:
            return None, 0.20

        # Compute closed-form confidence for the chosen hunk
        confidence = min(1.0, max(0.0, best_score))
        return best_hunk, confidence

    # -------------------------------------------------------------------------
    # Stage 4: Mechanism Classification with noIssue
    # -------------------------------------------------------------------------
    def _stage_4_classify_mechanism(self, dimension: str, hunk: DiffHunk, ctx: DualContext) -> Tuple[str, float]:
        """Classifies the defect mechanism against the standardized taxonomy with noIssue fallback."""
        candidates = MECHANISM_TAXONOMY.get(dimension, ["other", NO_ISSUE])
        content_lower = hunk.content.lower()

        # Deterministic semantic matching per dimension taxonomy
        if dimension == RiskDimension.CORRECTNESS.value:
            if any(w in content_lower for w in ["if ", "elif ", "else", "not ", "=="]):
                return "condition", 0.90
            if any(w in content_lower for w in ["state", "cache", "self.", "global"]):
                return "state", 0.85
            if any(w in content_lower for w in ["return", "transform", "map", "data"]):
                return "dataFlow", 0.80
            if any(w in content_lower for w in ["async", "await", "coroutine", "thread"]):
                return "asyncControl", 0.85

        elif dimension == RiskDimension.SECURITY.value:
            if any(w in content_lower for w in ["auth", "permission", "role", "token", "bearer"]):
                return "authorization", 0.95
            if any(w in content_lower for w in ["eval", "exec", "os.system", "format", "%s", "rm -rf", "subprocess", "shell", "bash"]):
                return "injection", 0.95
            if any(w in content_lower for w in ["password", "secret", "private_key", "leak"]):
                return "exposure", 0.90
            if any(w in content_lower for w in ["default", "allow_all", "0.0.0.0", "verify=false"]):
                return "unsafeDefault", 0.85

        elif dimension == RiskDimension.RELIABILITY.value:
            if any(w in content_lower for w in ["open(", "close()", "finally:", "leak", "free"]):
                return "cleanup", 0.90
            if any(w in content_lower for w in ["lock", "race", "mutex", "deadlock"]):
                return "concurrency", 0.90
            if any(w in content_lower for w in ["except", "retry", "fallback", "catch"]):
                return "recovery", 0.85
            if any(w in content_lower for w in ["panic", "raise", "error", "crash"]):
                return "crash", 0.90

        elif dimension == RiskDimension.COMPATIBILITY.value:
            if any(w in content_lower for w in ["def ", "interface", "signature"]):
                return "api", 0.90
            if any(w in content_lower for w in ["behavior", "return", "output"]):
                return "behavior", 0.85
            if any(w in content_lower for w in ["json", "proto", "format", "schema"]):
                return "dataFormat", 0.85
            if any(w in content_lower for w in ["protocol", "http", "grpc"]):
                return "protocol", 0.85

        elif dimension == RiskDimension.TEST_GAP.value:
            if any(w in content_lower for w in ["if ", "else", "case", "switch"]):
                return "branch", 0.85
            if any(w in content_lower for w in ["except", "fail", "raise", "error"]):
                return "failure", 0.90
            if any(w in content_lower for w in ["max", "min", "0", "-1", "boundary", "len"]):
                return "boundary", 0.85
            if any(w in content_lower for w in ["call", "service", "client", "request"]):
                return "integration", 0.85

        # Fallback to noIssue if nothing concrete matched
        return NO_ISSUE, 0.40

    # -------------------------------------------------------------------------
    # Stage 5: Severity Scoring (0 ~ 3 scale)
    # -------------------------------------------------------------------------
    def _stage_5_score_severity(
        self,
        dimension: str,
        mechanism: str,
        hunk: DiffHunk,
        ctx: DualContext,
    ) -> float:
        """Assigns production impact severity on a 0~3 scale.

        0: No impact (cosmetic, documentation)
        1: Minor impact (non-critical, routine warning)
        2: Significant impact (blocks merge, potential bug or data corruption)
        3: Fatal impact (destructive command, remote execution, security hole)
        """
        content_lower = hunk.content.lower()

        # Level 3: Fatal breaches
        if "rm -rf /" in content_lower or ":(){ :|:& };:" in content_lower:
            return 3.0
        if dimension == RiskDimension.SECURITY.value and mechanism in ("injection", "authorization"):
            if "os.system" in content_lower or "eval(" in content_lower:
                return 3.0
            return 2.5

        # Level 2: Significant bugs / data corruption / missing tests on critical logic
        if dimension == RiskDimension.CORRECTNESS.value and mechanism in ("condition", "state"):
            return 2.0
        if dimension == RiskDimension.RELIABILITY.value and mechanism in ("concurrency", "cleanup"):
            return 2.0
        if dimension == RiskDimension.COMPATIBILITY.value and mechanism == "api":
            return 2.0
        if dimension == RiskDimension.TEST_GAP.value and mechanism in ("failure", "branch"):
            # Significant test gap on core business code
            return 2.0

        # Level 1: Minor
        return 1.0
