"""Canonical loader for the frozen Qwen3.5-9B baseline results.

This ensures Qwen-9B raw performance is loaded from the persistent canonical
archive, rather than recomputed or re-parsed from raw prediction files every time.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASELINE_JSON = ROOT / "benchmarks" / "results" / "canonical_qwen9b_baseline.json"

_CACHED_BASELINE: Dict[str, Any] | None = None


def load_qwen9b_baseline(force_reload: bool = False) -> Dict[str, Any]:
    """Load the permanently frozen Qwen3.5-9B baseline data.

    Returns:
        dict containing 'summary', 'per_task', 'hardware', and 'provenance_sha256'.
    """
    global _CACHED_BASELINE
    if _CACHED_BASELINE is not None and not force_reload:
        return _CACHED_BASELINE

    if not BASELINE_JSON.exists():
        raise FileNotFoundError(
            f"Canonical baseline file not found at {BASELINE_JSON}. "
            "Run export script to generate it."
        )

    with open(BASELINE_JSON, "r", encoding="utf-8") as fh:
        _CACHED_BASELINE = json.load(fh)

    return _CACHED_BASELINE


def get_task_baseline(task_name: str) -> Dict[str, Any]:
    """Retrieve raw Qwen-9B baseline numbers for a specific task."""
    baseline = load_qwen9b_baseline()
    tasks = baseline.get("per_task", {})
    if task_name not in tasks:
        raise KeyError(f"Task '{task_name}' not found in canonical baseline.")
    return tasks[task_name]
