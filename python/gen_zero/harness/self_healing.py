"""In-Loop Self-Healing Trigger & Narrow Replanning Assembler (Issue #13).

Provides immediate interruption and focused remediation when executor steps fail:
1. Intercepts command exit codes, test failures, and tracebacks.
2. Extracts target file, line number, exception type, and failing assertion.
3. Generates structured SelfHealingTicket with narrow, non-divergent fix instructions.
4. Directly feeds bounded repair guidance into Orchestrator for instant remediation.
"""

from dataclasses import dataclass, field
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union


# Regex parsers for stack traces and test failures
RE_PYTHON_TRACEBACK_LINE = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\w+))?')
RE_PYTHON_EXCEPTION = re.compile(r'\b([A-Za-z0-9_]*(?:Error|Exception|Failure))(?:\s*:\s*(.*))?')
RE_UNITTEST_FAIL_LINE = re.compile(r'(?:FAIL|ERROR):\s+([a-zA-Z0-9_]+)\s+\((?:[\w\.]+)\.([a-zA-Z0-9_]+)\)')
RE_PYTEST_FAIL_LINE = re.compile(r'FAILED\s+([^:]+)::([^\s]+)')


@dataclass
class SelfHealingTicket:
    """Structured remediation ticket emitted upon execution failure."""
    ticket_id: str
    error_type: str
    error_message: str
    target_file: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    failed_tests: List[str] = field(default_factory=list)
    traceback_snippet: str = ""
    reproduction_command: Optional[str] = None
    narrow_instruction: str = ""
    is_critical: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "target_file": self.target_file,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "failed_tests": list(self.failed_tests),
            "traceback_snippet": self.traceback_snippet,
            "reproduction_command": self.reproduction_command,
            "narrow_instruction": self.narrow_instruction,
            "is_critical": self.is_critical,
            "metadata": dict(self.metadata)
        }

    def format_orchestrator_prompt(self) -> str:
        """Formats a high-density, narrow remediation prompt for the orchestrator."""
        lines = [
            f"=== [IN-LOOP SELF-HEALING TICKET: {self.ticket_id}] ===",
            f"Error Category: {self.error_type}",
            f"Root Message: {self.error_message}"
        ]
        if self.target_file:
            loc = f"{self.target_file}:{self.line_number}" if self.line_number else self.target_file
            if self.function_name:
                loc += f" (in {self.function_name})"
            lines.append(f"Target Location: {loc}")
        if self.failed_tests:
            lines.append(f"Failed Tests ({len(self.failed_tests)}): {', '.join(self.failed_tests[:5])}")
        if self.reproduction_command:
            lines.append(f"Reproduction: `{self.reproduction_command}`")
        lines.append(f"Narrow Fix Instruction: {self.narrow_instruction}")
        if self.traceback_snippet:
            lines.append(f"Traceback Snippet:\n{self.traceback_snippet.strip()}")
        lines.append("Constraint: Focus strictly on repairing this localized error. Do not refactor unrelated modules.")
        return "\n".join(lines)


class InLoopSelfHealingTrigger:
    """Detects in-flight execution anomalies and synthesizes self-healing tickets."""

    @classmethod
    def analyze_execution(
        cls,
        command: Optional[str],
        exit_code: int,
        stdout: str,
        stderr: str = ""
    ) -> Tuple[bool, Optional[SelfHealingTicket]]:
        """Analyzes command outcome and generates SelfHealingTicket if failed."""
        combined_output = f"{stdout}\n{stderr}".strip()

        # Check if success
        has_failed = (exit_code != 0)
        has_fail_indicator = any(
            ind in combined_output
            for ind in ("FAILED (failures=", "FAILED (errors=", "Traceback (most recent call last):", "SyntaxError:", "AssertionError:")
        )

        if not has_failed and not has_fail_indicator:
            return False, None

        ticket_id = f"heal-{uuid.uuid4().hex[:8]}"

        # 1. Extract traceback file & line (filtering internal stdlib assertion frames)
        target_file = None
        line_num = None
        func_name = None
        tb_matches = RE_PYTHON_TRACEBACK_LINE.findall(combined_output)
        if tb_matches:
            selected_frame = tb_matches[-1]
            for frame in reversed(tb_matches):
                f_path = frame[0]
                if not any(ign in f_path for ign in ("/unittest/case.py", "/lib/python", "site-packages/pytest")):
                    selected_frame = frame
                    break
            target_file = selected_frame[0]
            try:
                line_num = int(selected_frame[1])
            except ValueError:
                line_num = None
            func_name = selected_frame[2] if len(selected_frame) > 2 and selected_frame[2] else None

        # 2. Extract exception class and message
        error_type = "CommandExecutionError"
        error_msg = f"Process exited with non-zero status code {exit_code}"
        exc_match = RE_PYTHON_EXCEPTION.search(combined_output)
        if exc_match:
            error_type = exc_match.group(1).strip()
            msg_part = exc_match.group(2)
            error_msg = msg_part.strip() if msg_part else f"{error_type} raised"

        # 3. Extract failed test names
        failed_tests = []
        for m in RE_UNITTEST_FAIL_LINE.finditer(combined_output):
            test_method, test_class = m.groups()
            failed_tests.append(f"{test_class}.{test_method}")
        for m in RE_PYTEST_FAIL_LINE.finditer(combined_output):
            file_part, test_part = m.groups()
            failed_tests.append(f"{file_part}::{test_part}")

        # 4. Extract compact traceback snippet
        snippet_lines = []
        if "Traceback (most recent call last):" in combined_output:
            tb_part = combined_output.split("Traceback (most recent call last):")[-1]
            for l in tb_part.strip().split("\n")[:10]:
                snippet_lines.append(l)
        elif failed_tests:
            snippet_lines = [f"Failed test: {t}" for t in failed_tests[:5]]
        else:
            # Last 5 lines of combined output
            snippet_lines = combined_output.split("\n")[-5:]
        tb_snippet = "\n".join(snippet_lines)

        # 5. Synthesize narrow instruction
        if target_file and line_num:
            narrow_instruction = f"Fix {error_type} in {target_file} around line {line_num} ('{error_msg}')."
        elif failed_tests:
            narrow_instruction = f"Fix assertion in test suite: {', '.join(failed_tests[:3])} ('{error_msg}')."
        else:
            narrow_instruction = f"Remediate command failure: '{error_msg}'."

        ticket = SelfHealingTicket(
            ticket_id=ticket_id,
            error_type=error_type,
            error_message=error_msg,
            target_file=target_file,
            line_number=line_num,
            function_name=func_name,
            failed_tests=failed_tests,
            traceback_snippet=tb_snippet,
            reproduction_command=command,
            narrow_instruction=narrow_instruction,
            is_critical=True,
            metadata={"exit_code": exit_code}
        )

        return True, ticket
