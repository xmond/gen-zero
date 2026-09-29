#!/usr/bin/env python3
"""Rebuild the open training pool with natural-language candidates and ground truth.

ARC-Challenge, ARC-Easy and MMLU-Pro ship `candidates` as bare option labels
("A".."D"/"A".."J", or in 62 ARC rows "1".."4") lifted off the frozen-eval
answer format; the candidate hidden states Zero's runtime encodes therefore
carry no information about what the option actually says
(docs/zero/10-trunk-unfreeze-and-latent-recurrence-engineering-spec.md,
section 0 and 1.3: MMLU-Pro's task-head cross-validation is 11.3%, no better
than the 10% chance level, because of exactly this). Each record's `context`
already embeds the option text as "(label) text" lines; this script parses
those lines back out and replaces `candidates`/`ground_truth` with the text
itself. APPS execution-prediction and Banking77 already carry natural-language
candidates and pass through byte-identical (asserted below, not assumed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "benchmarks" / "artifacts" / "verified_datasets" / "open_training_pool_5k.jsonl"
DST = REPO / "benchmarks" / "artifacts" / "verified_datasets" / "open_training_pool_natural_5k.jsonl"

LABEL_TASKS = {"arc_challenge_train", "arc_easy_train", "mmlu_pro_test"}
PASSTHROUGH_TASKS = {"apps_execution_prediction", "banking77"}

SELECT_SUFFIX = "Select the single correct option."

# The frozen 930-question eval manifest (benchmarks/data/manifest.json, tasks[]).
# Neither "mmlu" nor "mmlu_pro" is in it, so using MMLU-Pro's test-split label
# here is not a direct eval leak. It is still a *test*-split label (MMLU-Pro
# has no upstream train split), so records are flagged as unsupervised-only.
FROZEN_EVAL_TASKS = {
    "massive_en", "massive_de", "vitaminc", "boolq", "squad2", "paws",
    "civil_comments", "aegis_safety", "multinli", "pubmedqa", "summeval",
    "arc_challenge", "gsm8k",
}
assert not ({"mmlu", "mmlu_pro"} & FROZEN_EVAL_TASKS)


def parse_labeled_options(context: str, labels: list[str], record_id: str) -> list[str]:
    """Pull each "(label) text" option's text out of a "(A) ...\\n(B) ..." context block.

    Anchored on the record's own candidate labels, not a fixed alphabet: 62
    ARC rows in this pool ship digit labels ("1".."4") instead of letters, so
    the parse target is whatever `candidates` already holds for this record.
    Raises ValueError (caught by the caller, which drops-and-counts) rather
    than ever silently guessing at a boundary.
    """
    if not context.endswith(SELECT_SUFFIX):
        raise ValueError("context does not end with the expected instruction suffix")
    body = context[: -len(SELECT_SUFFIX)]
    markers = [f"({label}) " for label in labels]
    positions = []
    for marker in markers:
        idx = body.find("\n" + marker)
        if idx == -1:
            raise ValueError(f"option marker {marker!r} not found in context")
        positions.append(idx + 1)  # position of the marker text, past the leading newline
    if positions != sorted(positions):
        raise ValueError("option markers are out of order in context")
    options = []
    for i, (pos, marker) in enumerate(zip(positions, markers)):
        start = pos + len(marker)
        end = positions[i + 1] if i + 1 < len(positions) else len(body)
        text = body[start:end].strip()
        if not text:
            raise ValueError(f"option {marker!r} parsed to empty text")
        options.append(text)
    return options


def rebuild(record: dict, stats: dict) -> Optional[dict]:
    task = record["task"]
    if task in PASSTHROUGH_TASKS:
        stats["passthrough"] += 1
        return record
    if task not in LABEL_TASKS:
        raise ValueError(f"unrecognized task {task!r}; add it to LABEL_TASKS or PASSTHROUGH_TASKS")

    labels = record["candidates"]
    gt_label = record["ground_truth"]
    if gt_label not in labels:
        stats["dropped_gt_not_in_labels"] += 1
        print(f"[drop] {record['id']}: ground_truth {gt_label!r} not in candidates {labels}", file=sys.stderr)
        return None

    try:
        options = parse_labeled_options(record["context"], labels, record["id"])
    except ValueError as error:
        stats["dropped_parse_error"] += 1
        print(f"[drop] {record['id']}: {error}", file=sys.stderr)
        return None

    if len(set(options)) != len(options):
        stats["dropped_duplicate_option_text"] += 1
        print(f"[drop] {record['id']}: duplicate option text among {options}", file=sys.stderr)
        return None

    new_record = dict(record)
    new_record["candidates"] = options
    new_record["ground_truth"] = options[labels.index(gt_label)]
    metadata = dict(record.get("metadata", {}))
    metadata["candidates_source"] = "parsed_from_context_natural_text"
    metadata["original_candidate_labels"] = labels
    if metadata.get("split") == "test":
        metadata["supervised_training_eligible"] = False
        metadata["leak_check"] = (
            "mmlu_pro is not among the 13 tasks in the frozen 930-question eval manifest "
            "(benchmarks/data/manifest.json: " + ", ".join(sorted(FROZEN_EVAL_TASKS)) + "); "
            "TIGER-Lab/MMLU-Pro has no upstream train split, so ground_truth here is a "
            "test-split label. Use for unsupervised state collection (manifold fitting) "
            "only; do not feed as a supervised task-head label."
        )
    new_record["metadata"] = metadata
    stats["rebuilt"] += 1
    return new_record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--max-drops", type=int, default=0,
                     help="Fail closed (nonzero exit) if more than this many records are "
                          "dropped for parse error, ground_truth-not-in-candidates, or "
                          "duplicate option text. Default 0: any real drop fails the run "
                          "instead of silently shipping a smaller pool.")
    args = ap.parse_args()

    stats = {"rebuilt": 0, "passthrough": 0, "dropped_gt_not_in_labels": 0,
              "dropped_parse_error": 0, "dropped_duplicate_option_text": 0}
    input_records = 0
    out_lines = []
    passthrough_unchanged = 0
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            input_records += 1
            record = json.loads(line)
            new_record = rebuild(record, stats)
            if new_record is not None:
                if record["task"] in PASSTHROUGH_TASKS:
                    assert new_record == record, f"{record['id']}: passthrough record was mutated"
                    passthrough_unchanged += 1
                out_lines.append(json.dumps(new_record, ensure_ascii=False))

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    real_drops = (stats["dropped_gt_not_in_labels"] + stats["dropped_parse_error"]
                  + stats["dropped_duplicate_option_text"])
    if real_drops > args.max_drops:
        print(f"[fail-closed] {real_drops} record(s) dropped, exceeds --max-drops={args.max_drops}; "
              "refusing to ship a silently-shrunk pool", file=sys.stderr)
        return 1

    tmp_dst = args.dst.with_name(f".{args.dst.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_dst, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        tmp_dst.replace(args.dst)
    except Exception:
        if tmp_dst.exists():
            tmp_dst.unlink()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
