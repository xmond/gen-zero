"""FactoryObservation & Bounded Workspace Evidence Snapshot.

Implements Milestone 1 of Issue #20:
1. Bounded Evidence Sampling:
   Captures read-only, sanitized, compact representations of workspace status.
2. Smart Diff Truncation:
   Limits git diff strictly to <= 50KB (51,200 bytes) while preserving critical
   head and tail lines with clear omission markers.
3. Test Telemetry & Contract Rule Ingestion:
   Aggregates test outcomes and project contract rules (AGENTS.md) for supervision.
"""

from typing import Dict, List, Any, Optional, Tuple
import dataclasses
import hashlib
import json
import time
import os


def smart_truncate_text(
    text: str,
    max_bytes: int = 51200,
    head_ratio: float = 0.5,
) -> str:
    """Truncates text strictly within max_bytes preserving head and tail lines."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    budget = max(100, max_bytes - 120)  # Reserve space for truncation banner
    head_budget = int(budget * head_ratio)
    tail_budget = budget - head_budget

    head_part = encoded[:head_budget].decode("utf-8", errors="ignore")
    tail_part = encoded[-tail_budget:].decode("utf-8", errors="ignore")

    # Clean partial lines
    if "\n" in head_part:
        head_part = head_part.rsplit("\n", 1)[0]
    if "\n" in tail_part:
        tail_part = tail_part.split("\n", 1)[1]

    omitted_bytes = len(encoded) - len(head_part.encode("utf-8")) - len(tail_part.encode("utf-8"))
    banner = f"\n... [DIFF TRUNCATED: {omitted_bytes} BYTES OMITTED (LIMIT: {max_bytes} BYTES)] ...\n"

    return f"{head_part}{banner}{tail_part}"


@dataclasses.dataclass
class FactoryObservation:
    """Bounded, read-only evidence snapshot captured from the worker workspace."""
    workspace_root: str
    worker_id: str
    step_index: int
    timestamp: float = dataclasses.field(default_factory=time.time)
    git_status: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    git_diff: str = ""
    recent_logs: str = ""
    test_results: Optional[Dict[str, Any]] = None
    contract_rules: List[str] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Enforce strict 50KB diff boundary
        if self.git_diff:
            self.git_diff = smart_truncate_text(self.git_diff, max_bytes=51200)
        # Enforce strict 10KB logs boundary
        if self.recent_logs:
            self.recent_logs = smart_truncate_text(self.recent_logs, max_bytes=10240)

    @property
    def fingerprint(self) -> str:
        """Deterministic fingerprint of modified files, untracked, deleted, and truncated diff."""
        mod_files = sorted(self.git_status.get("modified", []))
        untracked = sorted(self.git_status.get("untracked", []))
        deleted = sorted(self.git_status.get("deleted", []))
        diff_hash = hashlib.sha256(self.git_diff.encode("utf-8")).hexdigest()[:16]
        raw = f"m:{','.join(mod_files)}|u:{','.join(untracked)}|d:{','.join(deleted)}::{diff_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_compact_summary(self) -> str:
        """Constructs a compact token-efficient state summary for scoring."""
        mod = len(self.git_status.get("modified", []))
        untracked = len(self.git_status.get("untracked", []))
        test_info = "no_tests_run"
        if self.test_results:
            passed = self.test_results.get("passed", 0)
            failed = self.test_results.get("failed", 0)
            test_info = f"tests(passed={passed}, failed={failed})"

        diff_lines = len(self.git_diff.splitlines())
        rules_cnt = len(self.contract_rules)

        return (
            f"WorkerObservation[worker={self.worker_id}, step={self.step_index}] | "
            f"Git: mod={mod}, untracked={untracked}, diff_lines={diff_lines} | "
            f"Test: {test_info} | Contracts: {rules_cnt} rules | "
            f"FP: {self.fingerprint}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "worker_id": self.worker_id,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
            "recent_logs": self.recent_logs,
            "test_results": self.test_results,
            "contract_rules": self.contract_rules,
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serializes observation to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactoryObservation":
        """Reconstructs FactoryObservation from dict, stripping computed fields."""
        clean_data = dict(data)
        clean_data.pop("fingerprint", None)
        return cls(**clean_data)

    @classmethod
    def from_json(cls, json_str: str) -> "FactoryObservation":
        """Reconstructs FactoryObservation from JSON string."""
        return cls.from_dict(json.loads(json_str))
