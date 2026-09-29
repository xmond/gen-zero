#!/usr/bin/env python3
"""Shared physical anti-leak gate for benchmark train-split converters.

Imports the exact `text_hash` function from build_diversity_train.py (Unicode
casefold + whitespace-normalized SHA256 of `context`) instead of redefining it,
so every converter excludes the frozen 930-question eval set
(benchmarks/data/manifest.json) and benchmarks/data/calibration_clean_16.jsonl
under one shared definition rather than two copies that could drift apart.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

CONVERTERS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = CONVERTERS_DIR.parent
BENCHMARKS_DIR = DATASETS_DIR.parent
DATA_DIR = BENCHMARKS_DIR / "data"

_SPEC = importlib.util.spec_from_file_location(
    "build_diversity_train", DATASETS_DIR / "build_diversity_train.py"
)
_bdt = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _bdt
_SPEC.loader.exec_module(_bdt)
text_hash = _bdt.text_hash


def _rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_excluded_hashes(data_dir: Path = DATA_DIR) -> Set[str]:
    """Content hashes for every frozen eval sample plus the calibration split.

    Reads benchmarks/data/manifest.json (930-question eval set) and
    calibration_clean_16.jsonl if present. Any row whose context hashes to one
    of these values must never enter a train pool.
    """
    excluded: Set[str] = set()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["tasks"].values():
        for row in _rows(data_dir / entry["file"]):
            excluded.add(text_hash(row["context"]))
    calibration_path = data_dir / "calibration_clean_16.jsonl"
    if calibration_path.exists():
        for row in _rows(calibration_path):
            excluded.add(text_hash(row["context"]))
    return excluded


def filter_leak_free(records: List[dict], excluded: Set[str]) -> Tuple[List[dict], Dict[str, int]]:
    """Drop records whose context hash is in `excluded`, or repeats within `records`."""
    kept: List[dict] = []
    seen_ids: Set[str] = set()
    seen_contexts: Set[str] = set()
    stats = {"input": len(records), "leaked": 0, "duplicate_in_batch": 0, "duplicate_id": 0, "kept": 0}
    for record in records:
        key = text_hash(record["context"])
        if key in excluded:
            stats["leaked"] += 1
            continue
        if key in seen_contexts:
            stats["duplicate_in_batch"] += 1
            continue
        if record["id"] in seen_ids:
            stats["duplicate_id"] += 1
            continue
        seen_ids.add(record["id"])
        seen_contexts.add(key)
        kept.append(record)
    stats["kept"] = len(kept)
    return kept, stats
