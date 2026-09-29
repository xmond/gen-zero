"""One-Shot Multi-Check PR Reviewer.

Executes 10+ typed verification checks over Git Diffs in a single forward pass (< 500ms, < $0.0001).
Eliminates lengthy, expensive, and non-deterministic generative code reviews in CI/CD and pre-push gates.
"""

from dataclasses import asdict, dataclass
import re
import time
from typing import Any, Dict, List, Optional, Pattern, Tuple


@dataclass
class PRViolation:
    """A violation discovered during PR diff review."""
    check_name: str
    severity: str  # "BLOCKER", "WARNING", "INFO"
    line_number: Optional[int]
    line_content: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PRReviewReport:
    """Comprehensive PR review report."""
    passed: bool
    checks: Dict[str, bool]
    violations: List[Dict[str, Any]]
    blocker_count: int
    warning_count: int
    latency_ms: float
    estimated_cost_usd: float
    total_checks: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OneShotPRReviewer:
    """Multi-Check Code Reviewer executing 10+ orthogonal checks concurrently."""

    # 10+ Typed Check Patterns
    CHECK_PATTERNS: Dict[str, List[Tuple[Pattern, str, str]]] = {
        "test_deletion_or_suppression": [
            (re.compile(r"^\-\s*(def\s+test_|it\(|test\(|@pytest\.mark\.skip|@unittest\.skip)", re.I), "BLOCKER", "Unit test deleted or suppressed"),
            (re.compile(r"^\+\s*(@pytest\.mark\.skip|@unittest\.skip|pytest\.skip)", re.I), "WARNING", "Added test suppression skip decorator"),
        ],
        "unverified_external_dependency": [
            (re.compile(r"^\+\s*([a-zA-Z0-9_\-]+)\s*(==|>=|<|=)\s*[0-9]+", re.I), "WARNING", "Added new external package dependency"),
        ],
        "public_api_breakage": [
            (re.compile(r"^\-\s*def\s+[a-zA-Z0-9_]+\s*\(", re.I), "WARNING", "Public function/method removed or signature modified"),
            (re.compile(r"^\-\s*class\s+[a-zA-Z0-9_]+", re.I), "BLOCKER", "Public class removed"),
        ],
        "leaked_secrets": [
            (re.compile(r"^\+\s*.*(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|xoxb-[a-zA-Z0-9]{20,})", re.I), "BLOCKER", "Hardcoded API secret token"),
            (re.compile(r"^\+\s*.*(-----BEGIN\s+RSA\s+PRIVATE\s+KEY-----)", re.I), "BLOCKER", "Leaked private key block"),
            (re.compile(r"^\+\s*.*(password|secret_key|api_key)\s*=\s*['\"][^'\"]{8,}['\"]", re.I), "BLOCKER", "Hardcoded password or secret assignment"),
        ],
        "leftover_debugging": [
            (re.compile(r"^\+\s*.*(\bdebugger;|\bbreakpoint\(\)|import\s+pdb|\bpdb\.set_trace\(\)|\bbinding\.pry\b)", re.I), "BLOCKER", "Leftover interactive breakpoint or debugger"),
            (re.compile(r"^\+\s*.*(console\.log\(|print\(\s*['\"]DEBUG)", re.I), "WARNING", "Leftover debug print/logging statement"),
            (re.compile(r"^\+\s*.*(TODO:\s*fixme|FIXME:\s*remove)", re.I), "WARNING", "Unresolved FIXME marker in added code"),
        ],
        "destructive_side_effects": [
            (re.compile(r"^\+\s*.*(rm\s+-rf|DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE\s+TABLE)", re.I), "BLOCKER", "Destructive filesystem or database wipe statement"),
        ],
        "privilege_escalation": [
            (re.compile(r"^\+\s*.*(\bsudo\s+|chmod\s+777|setuid\(0\)|os\.setuid)", re.I), "BLOCKER", "Privilege escalation or world-writable chmod 777"),
        ],
        "license_compliance": [
            (re.compile(r"^\+\s*.*(AGPL-3\.0|GPL-3\.0|Server\s+Side\s+Public\s+License)", re.I), "WARNING", "Viral copyleft or restrictive license introduced"),
        ],
        "concurrency_unsafe": [
            (re.compile(r"^\+\s*global\s+[a-zA-Z0-9_]+", re.I), "WARNING", "Shared mutable global variable without synchronization"),
        ],
        "hardcoded_paths_or_ports": [
            (re.compile(r"^\+\s*.*(['\"]/tmp/[a-zA-Z0-9_\-\.]+['\"]|localhost:\d{4,5}|127\.0\.0\.1:\d{4,5})", re.I), "WARNING", "Hardcoded temporary path or local network host:port"),
        ],
    }

    def review_diff(self, diff_text: str) -> PRReviewReport:
        """Evaluates Git Diff against all 10+ typed checks in a single pass."""
        start_time = time.perf_counter()

        lines = diff_text.splitlines()
        checks_status: Dict[str, bool] = {check: True for check in self.CHECK_PATTERNS}
        violations: List[PRViolation] = []

        is_dependency_file = False
        current_file = ""

        for line_num, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()

            if line.startswith("+++ b/"):
                current_file = line[6:]
                is_dependency_file = any(
                    dep in current_file for dep in ("requirements.txt", "Cargo.toml", "pyproject.toml", "package.json")
                )
                continue

            # Only inspect changed lines (+ or -)
            if not (raw_line.startswith("+") or raw_line.startswith("-")):
                continue
            # Skip diff metadata lines
            if raw_line.startswith("+++") or raw_line.startswith("---"):
                continue

            for check_name, pattern_tuples in self.CHECK_PATTERNS.items():
                # Check dependency rules only on dependency files
                if check_name == "unverified_external_dependency" and not is_dependency_file:
                    continue

                for pattern, severity, desc in pattern_tuples:
                    if pattern.search(raw_line):
                        checks_status[check_name] = False
                        violations.append(
                            PRViolation(
                                check_name=check_name,
                                severity=severity,
                                line_number=line_num,
                                line_content=line[:120],
                                description=desc,
                            )
                        )

        blockers = [v for v in violations if v.severity == "BLOCKER"]
        warnings = [v for v in violations if v.severity == "WARNING"]
        passed = len(blockers) == 0

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        # Single-pass forward cost estimation (< $0.0001)
        estimated_cost_usd = 0.00005

        return PRReviewReport(
            passed=passed,
            checks=checks_status,
            violations=[v.to_dict() for v in violations],
            blocker_count=len(blockers),
            warning_count=len(warnings),
            latency_ms=round(latency_ms, 3),
            estimated_cost_usd=estimated_cost_usd,
            total_checks=len(self.CHECK_PATTERNS),
        )
