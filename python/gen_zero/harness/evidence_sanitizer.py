"""Objective Evidence Sanitizer & Tone Invariant Anchoring (Issue #13).

Defends verification and decision layers against subjective emotional bias, developer panic,
hyperbole, and tone variations. Extracts deterministic ground-truth system facts
(exit codes, test statistics, stack frames, metrics) and produces canonical representations.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple, Union


# Regex patterns matching subjective emotional adjectives and filler
TONE_NOISE_PATTERNS = [
    r"(?i)\b(?:please\s+please|i\s+am\s+begging\s+you|for\s+the\s+love\s+of\s+god|oh\s+god|damn\s+it|holy\s+shit)\b",
    r"(?i)\b(?:disaster|catastrophe|horrible|terrible|stupid|awful|crap|nightmare|insane)\b",
    r"(?i)\b(?:miracle|incredible|magnificent|flawless|genius|phenomenal|wonderful)\b",
    r"(?i)\b(?:i\s+think\s+maybe|probably\s+works|looks\s+fine\s+to\s+me|hope\s+it\s+works|cross\s+fingers)\b",
    r"(?i)\b(?:as\s+per\s+my\s+previous\s+email|frankly\s+speaking|to\s+be\s+honest|honestly)\b",
    r"[!]{2,}",  # Multiple exclamation marks
    r"[\?]{2,}",  # Multiple question marks
]

# Patterns for extracting structured metrics from test and command outputs
RE_EXIT_CODE = re.compile(r"(?i)(?:exit\s*code|returncode|status\s*code)\s*[:=]?\s*([-\d]+)")
RE_UNITTEST_STATS = re.compile(r"(?i)Ran\s+(\d+)\s+tests?\s+in\s+([\d\.]+)s")
RE_UNITTEST_FAILED = re.compile(r"(?i)FAILED\s*\((?:failures=(\d+))?(?:,\s*)?(?:errors=(\d+))?(?:,\s*)?(?:skipped=(\d+))?\)")
RE_UNITTEST_OK = re.compile(r"(?i)\bOK\b(?:\s*\((?:skipped=(\d+))?\))?")
RE_PYTEST_SUMMARY = re.compile(r"(?i)=(\s*(?:\d+\s+passed|\d+\s+failed|\d+\s+error|\d+\s+skipped)[^=]*?)=")


@dataclass
class ObjectiveEvidence:
    """Canonical representation of objective system evidence."""
    exit_code: int = 0
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    duration_sec: float = 0.0
    detected_exceptions: List[str] = field(default_factory=list)
    raw_output_snippet: str = ""
    is_success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_canonical_string(self) -> str:
        """Serializes hard evidence to deterministic canonical string."""
        parts = [
            f"EXIT_CODE={self.exit_code}",
            f"TESTS_TOTAL={self.total_tests}",
            f"TESTS_PASSED={self.passed_tests}",
            f"TESTS_FAILED={self.failed_tests}",
            f"TESTS_ERRORS={self.error_tests}",
            f"SUCCESS={'TRUE' if self.is_success else 'FALSE'}"
        ]
        if self.duration_sec > 0:
            parts.append(f"DURATION={self.duration_sec:.2f}s")
        if self.detected_exceptions:
            unique_exc = sorted(set(self.detected_exceptions))
            parts.append(f"EXCEPTIONS=[{', '.join(unique_exc)}]")
        if self.raw_output_snippet:
            clean_snip = re.sub(r"\s+", " ", self.raw_output_snippet.strip())
            parts.append(f"SNIPPET={clean_snip[:160]}")
        return " | ".join(parts)


class ObjectiveEvidenceSanitizer:
    """Sanitizes raw human/tool outputs into objective, tone-invariant evidence."""

    @classmethod
    def strip_subjective_noise(cls, text: str) -> str:
        """Strips emotional bias, hyperbolic language, and conversational fluff."""
        cleaned = text
        for pat in TONE_NOISE_PATTERNS:
            cleaned = re.sub(pat, " ", cleaned)
        # Normalize redundant whitespaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def extract_objective_evidence(
        cls,
        raw_evidence: Union[str, Dict[str, Any]],
        explicit_exit_code: Optional[int] = None
    ) -> ObjectiveEvidence:
        """Extracts structured objective facts from raw evidence string or dict."""
        evidence = ObjectiveEvidence()

        if isinstance(raw_evidence, dict):
            evidence.exit_code = int(raw_evidence.get("exit_code", explicit_exit_code if explicit_exit_code is not None else 0))
            evidence.total_tests = int(raw_evidence.get("total_tests", 0))
            evidence.passed_tests = int(raw_evidence.get("passed_tests", 0))
            evidence.failed_tests = int(raw_evidence.get("failed_tests", 0))
            evidence.error_tests = int(raw_evidence.get("error_tests", 0))
            evidence.skipped_tests = int(raw_evidence.get("skipped_tests", 0))
            evidence.duration_sec = float(raw_evidence.get("duration_sec", 0.0))
            evidence.detected_exceptions = list(raw_evidence.get("exceptions", []))
            evidence.is_success = bool(raw_evidence.get("is_success", (evidence.exit_code == 0 and evidence.failed_tests == 0 and evidence.error_tests == 0)))
            evidence.metadata = dict(raw_evidence)
            evidence.raw_output_snippet = str(raw_evidence.get("stdout", raw_evidence.get("output", "")))[:200]
            return evidence

        raw_str = str(raw_evidence)
        cleaned_text = cls.strip_subjective_noise(raw_str)

        # 1. Exit code extraction
        if explicit_exit_code is not None:
            evidence.exit_code = explicit_exit_code
        else:
            m_code = RE_EXIT_CODE.search(raw_str)
            if m_code:
                try:
                    evidence.exit_code = int(m_code.group(1))
                except ValueError:
                    evidence.exit_code = 0
            else:
                evidence.exit_code = 0

        # 2. Python unittest statistics
        m_ran = RE_UNITTEST_STATS.search(raw_str)
        if m_ran:
            evidence.total_tests = int(m_ran.group(1))
            try:
                evidence.duration_sec = float(m_ran.group(2))
            except ValueError:
                evidence.duration_sec = 0.0

            m_fail = RE_UNITTEST_FAILED.search(raw_str)
            if m_fail:
                evidence.failed_tests = int(m_fail.group(1) or 0)
                evidence.error_tests = int(m_fail.group(2) or 0)
                evidence.skipped_tests = int(m_fail.group(3) or 0)
                evidence.passed_tests = max(0, evidence.total_tests - evidence.failed_tests - evidence.error_tests - evidence.skipped_tests)
                evidence.is_success = False
            elif RE_UNITTEST_OK.search(raw_str):
                m_ok = RE_UNITTEST_OK.search(raw_str)
                evidence.skipped_tests = int(m_ok.group(1) or 0) if m_ok else 0
                evidence.passed_tests = max(0, evidence.total_tests - evidence.skipped_tests)
                evidence.failed_tests = 0
                evidence.error_tests = 0
                evidence.is_success = (evidence.exit_code == 0)

        # 3. Pytest summary parsing
        m_pytest = RE_PYTEST_SUMMARY.search(raw_str)
        if m_pytest:
            summary_content = m_pytest.group(1)
            p_pass = re.search(r"(\d+)\s+passed", summary_content)
            p_fail = re.search(r"(\d+)\s+failed", summary_content)
            p_err = re.search(r"(\d+)\s+error", summary_content)
            p_skip = re.search(r"(\d+)\s+skipped", summary_content)

            passed = int(p_pass.group(1)) if p_pass else 0
            failed = int(p_fail.group(1)) if p_fail else 0
            errors = int(p_err.group(1)) if p_err else 0
            skipped = int(p_skip.group(1)) if p_skip else 0

            evidence.passed_tests = passed
            evidence.failed_tests = failed
            evidence.error_tests = errors
            evidence.skipped_tests = skipped
            evidence.total_tests = passed + failed + errors + skipped
            evidence.is_success = (failed == 0 and errors == 0 and evidence.exit_code == 0)

        # 4. Extract standard exception classes
        exc_matches = re.findall(r"\b([A-Z][a-zA-Z0-9]*(?:Error|Exception|Warning|Failure))\b", raw_str)
        if exc_matches:
            evidence.detected_exceptions = list(dict.fromkeys(exc_matches))

        # Overall success arbitration
        if evidence.failed_tests > 0 or evidence.error_tests > 0 or evidence.exit_code != 0:
            evidence.is_success = False

        evidence.raw_output_snippet = cleaned_text[:200]
        return evidence

    @classmethod
    def sanitize(
        cls,
        evidence_input: Union[str, Dict[str, Any]],
        explicit_exit_code: Optional[int] = None
    ) -> str:
        """Converts raw evidence with subjective noise into canonical objective string."""
        obj = cls.extract_objective_evidence(evidence_input, explicit_exit_code=explicit_exit_code)
        return obj.to_canonical_string()
