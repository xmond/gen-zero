"""Shared schema validation and physical evaluation isolation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

EVALUATION = Path(__file__).resolve().parents[2] / "data" / "all_benchmarks.jsonl"
KEYS = {"id", "task", "context", "candidates", "ground_truth", "metadata"}


def text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def excluded_contexts(path: Path = EVALUATION) -> set[str]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {text_hash(row["context"]) for row in rows}


def shuffled(candidates: list[str], key: str) -> list[str]:
    """Deterministic order that depends only on the sample id and each candidate's text.

    Sorting by a per-candidate hash, not permuting by a seeded shuffle: a seeded
    permutation of ``[answer, *distractors]`` can be recomputed from the id and
    points straight at the answer. Here the order of a candidate set is the same
    whichever member is the answer, so the id reveals nothing about it.
    """
    return sorted(candidates, key=lambda text: exact_hash(f"{key}\0{text}"))


def isolated(rows, evaluation_path: Path = EVALUATION, *, exact: bool = False, stats: Counter | None = None):
    """Validate rows, drop evaluation overlap and duplicate contexts.

    Evaluation exclusion always uses the casefolded, whitespace-collapsed hash:
    a looser match excludes more, which is the safe side for leakage. Within-set
    deduplication uses that same key for natural language, but ``exact=True``
    (code) compares raw text, because case and indentation change a program.
    """
    excluded = excluded_contexts(evaluation_path)
    stats = Counter() if stats is None else stats
    seen_ids, seen_text = set(), set()
    for row in rows:
        if set(row) != KEYS or any(not isinstance(row[k], str) or not row[k] for k in ("id", "task", "context", "ground_truth")):
            raise ValueError("invalid six-field schema")
        candidates = row["candidates"]
        if not isinstance(candidates, list) or len(candidates) < 2 or any(not isinstance(c, str) or not c.strip() for c in candidates):
            raise ValueError("invalid candidates")
        if len(set(candidates)) != len(candidates) or row["ground_truth"] not in candidates or not isinstance(row["metadata"], dict):
            raise ValueError("invalid ground truth or metadata")
        if row["id"] in seen_ids:
            raise ValueError("duplicate source id")
        seen_ids.add(row["id"])
        if text_hash(row["context"]) in excluded:
            stats["excluded_eval_overlap"] += 1
            continue
        digest = exact_hash(row["context"]) if exact else text_hash(row["context"])
        if digest in seen_text:
            stats["duplicate_context"] += 1
            continue
        seen_text.add(digest)
        stats["kept"] += 1
        yield row
