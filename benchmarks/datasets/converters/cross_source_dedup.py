#!/usr/bin/env python3
"""Cross-source dedup for train-split converters that draw from overlapping
upstream corpora.

cais/mmlu's own data card documents `auxiliary_train` as assembled from other
multiple-choice exam sources, ARC included. Verified directly against the
live parquet, running this module in the same order the real pipeline does
(each source already put through its own arc_converter.py / mmlu_converter.py
`main()`, which already applies leak_gate.filter_leak_free per source before
this module ever sees the data): ARC train converts+leak-gate-filters to
3,369 rows across both configs, MMLU auxiliary_train converts+leak-gate-
filters to 98,589 rows, and comparing exact `text_hash(context)` values finds
3,359 of those MMLU rows are a verbatim duplicate of an ARC train row (all
counted here as `cross_source_duplicates`, since the per-source leak_gate
pass already absorbed MMLU's internal exact duplicates as
`duplicate_in_batch` before this module runs). Final kept MMLU row count:
95,230.

An earlier count of "4,306" was reported before two things were both true:
ARC's 124 digit-labeled rows ("(1)..(4)") had not yet been normalized to
letters ("(A)..(D)", see arc_converter.py's `_normalize_label`), and the
comparison was run against each source's *raw* converted output rather than
its leak_gate-filtered output. Normalizing those 124 rows makes ~248 of them
newly detectable as MMLU duplicates (MMLU always shipped letter labels), and
running after per-source leak_gate dedup reassigns some rows from
"cross-source duplicate" to "already removed as an internal MMLU
duplicate" -- the two effects roughly offset, but the honest, currently-true
number for `cross_source_duplicates` in the real pipeline order is 3,359, not
4,306. Whichever order this module is invoked in, the final *kept* MMLU count
is the same 95,230 -- that invariant is verified by
`test_live_full_scale_cross_source_dedup_finds_the_same_kept_count_either_order`
in benchmarks/tests/test_arc_mmlu_converters.py.

This is a *within-corpus* concern, distinct from the physical leak_gate
(which only guards against the frozen 930-question eval set and the
calibration split). A row can be perfectly leak-free against the eval set
and still be a same-content duplicate of another train-source row -- feeding
both copies into one fused training pool would silently double-weight that
question. This module removes that duplication with one designated source
treated as primary (ARC, since it is the original author of these questions)
and the other treated as secondary (MMLU auxiliary_train), using the exact
same `text_hash` function as leak_gate so a row counted as "the same
question" here means the same thing everywhere else in this pipeline.

No language-specific heuristics: dedup key is the SHA256 of the fully
formatted context, not a keyword or language-specific match.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leak_gate import text_hash  # noqa: E402


def cross_source_deduplicate(
    primary_records: List[dict], secondary_records: List[dict]
) -> Tuple[List[dict], Dict[str, int]]:
    """Drop rows from `secondary_records` whose context duplicates a row in
    `primary_records` (or a row already kept from `secondary_records`).

    `primary_records` is never filtered or reordered: ties always resolve in
    its favor. Returns the deduplicated secondary list plus a stats dict.
    """
    primary_hashes: Set[str] = {text_hash(r["context"]) for r in primary_records}
    kept: List[dict] = []
    seen_secondary_hashes: Set[str] = set()
    stats = {
        "primary_count": len(primary_records),
        "secondary_input": len(secondary_records),
        "cross_source_duplicates": 0,
        "secondary_internal_duplicates": 0,
        "kept": 0,
    }
    for record in secondary_records:
        key = text_hash(record["context"])
        if key in primary_hashes:
            stats["cross_source_duplicates"] += 1
            continue
        if key in seen_secondary_hashes:
            stats["secondary_internal_duplicates"] += 1
            continue
        seen_secondary_hashes.add(key)
        kept.append(record)
    stats["kept"] = len(kept)
    return kept, stats


def deduplicate_mmlu_against_arc(
    arc_records: List[dict], mmlu_records: List[dict]
) -> Tuple[List[dict], Dict[str, int]]:
    """ARC-vs-MMLU-auxiliary_train convenience wrapper: ARC is primary because
    it is the original author of the New York Regents / ARC-Easy questions
    that MMLU auxiliary_train re-packaged; MMLU is the secondary pool that
    gets its overlapping rows dropped.
    """
    return cross_source_deduplicate(arc_records, mmlu_records)


def _rows(path: Path) -> List[dict]:
    import json

    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    import argparse
    import json as json_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True, help="primary-source JSONL (e.g. ARC train), never filtered")
    parser.add_argument("--secondary", type=Path, required=True, help="secondary-source JSONL (e.g. MMLU auxiliary_train), gets duplicates removed")
    parser.add_argument("--output", type=Path, required=True, help="path to write the deduplicated secondary JSONL")
    args = parser.parse_args()

    primary_records = _rows(args.primary)
    secondary_records = _rows(args.secondary)
    kept, stats = cross_source_deduplicate(primary_records, secondary_records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json_module.dumps(r, ensure_ascii=False) + "\n")

    print(json_module.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
