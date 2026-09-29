#!/usr/bin/env python3
"""Builds benchmarks/data/calibration_clean_16.jsonl.

Draws 16-32 calibration samples per task from the SAME raw Hugging Face
parquet caches used by fetch_real_datasets.py, but strictly AFTER the row
index already consumed by the frozen 930-sample test manifest
(benchmarks/data/manifest.json).

How disjointness is guaranteed:
  For 11 of 13 tasks (all except paws and gsm8k), the frozen test file was
  built by fetch_real_datasets.py's plain scan: parquet rows 0, 1, 2, ...
  in order through spec.extractor(), keeping the first N valid extractions
  (N = manifest.tasks[task].samples_count). This script replays that exact
  scan to find `boundary`, the row index right after the last row the test
  set consumed, then resumes from `boundary + GAP`. For those 11 tasks this
  is a provable, row-index-level disjointness guarantee: verified by
  replaying rows [0, boundary) with the current extractor and diffing
  against benchmarks/data/{task}.jsonl (exact match, see build log).

  paws and gsm8k are NOT built by that plain scan -- benchmarks/data/
  {paws,gsm8k}.jsonl come from rebuild_gsm8k_paws.py, which does seeded
  stratified sampling (by lexical-overlap bucket for paws, by solution
  length for gsm8k) over the FULL split, not a first-N-rows scan. Replaying
  the plain-scan boundary for these two tasks does NOT reproduce their real
  test rows (confirmed: content mismatches at index 0 for both). paws also
  reads from a different cached parquet split (`validation`) than its real
  test set (`test`), a stronger disjointness signal by construction. gsm8k
  reads the same `test` split, non-contiguously sampled, so the boundary
  heuristic here is not provable a priori.

  The actual, authoritative guarantee for ALL 13 tasks -- including paws
  and gsm8k -- is benchmarks/tests/test_calibration_split_isolation.py,
  which diffs this file's normalized `context` text directly against the
  real, on-disk benchmarks/data/{task}.jsonl files and asserts zero
  intersection. That test is what must pass, not this script's heuristics.

Usage:
    python3 benchmarks/datasets/build_calibration_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_real_datasets import BENCHMARK_SPECS, get_default_external_cache_dir  # noqa: E402

BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BENCHMARKS_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
OUTPUT_PATH = DATA_DIR / "calibration_clean_16.jsonl"

SAMPLES_PER_TASK = 24  # within the requested 16-32 band
GAP = 20  # extra safety margin of rows skipped past the test-set boundary


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _load_excluded_contexts(task_name: str) -> set[str]:
    """Normalized contexts already used by the frozen test file for this task.
    This is the hard, construction-time guarantee: independent of whether the
    row-index boundary heuristic below is accurate for this task (it is not,
    for paws/gsm8k -- see module docstring), no row whose extracted content
    matches an existing test row is ever emitted as a calibration sample."""
    task_file = DATA_DIR / f"{task_name}.jsonl"
    excluded = set()
    with open(task_file, encoding="utf-8") as f:
        for line in f:
            excluded.add(_normalize(json.loads(line)["context"]))
    return excluded


def find_boundary_and_calibration(
    spec, cache_file: Path, target_test_count: int, calibration_count: int
) -> tuple[int, List[Dict[str, Any]]]:
    """Replays the fetch_real_datasets.py plain scan to find a starting point
    past the rows the test set is expected to have consumed (exact for 11/13
    tasks, a best-effort head start for paws/gsm8k -- see module docstring),
    then collects fresh calibration samples, hard-skipping any row whose
    extracted content collides with the real on-disk test file."""
    tbl = pq.read_table(cache_file)
    num_rows = len(tbl)
    col_names = tbl.column_names
    excluded = _load_excluded_contexts(spec.name)

    consumed = 0
    boundary = None
    for i in range(num_rows):
        row_dict = {c: tbl[c][i].as_py() for c in col_names}
        extracted = spec.extractor(row_dict)
        if extracted and extracted["ground_truth"] in spec.candidates:
            consumed += 1
            if consumed >= target_test_count:
                boundary = i + 1
                break
    if boundary is None:
        raise RuntimeError(
            f"[{spec.name}] could not replay test-set boundary: only "
            f"{consumed}/{target_test_count} valid rows found in {num_rows} total rows"
        )

    cal_start = boundary + GAP
    calibration: List[Dict[str, Any]] = []
    skipped_collisions = 0
    for i in range(cal_start, num_rows):
        row_dict = {c: tbl[c][i].as_py() for c in col_names}
        extracted = spec.extractor(row_dict)
        if not extracted or extracted["ground_truth"] not in spec.candidates:
            continue
        normalized = _normalize(extracted["context"])
        if normalized in excluded:
            skipped_collisions += 1
            continue
        sample_id = f"{spec.name}-cal-{len(calibration) + 1:04d}"
        calibration.append(
            {
                "id": sample_id,
                "task": spec.name,
                "context": extracted["context"],
                "candidates": spec.candidates,
                "ground_truth": extracted["ground_truth"],
                "metadata": {
                    **extracted.get("metadata", {}),
                    "source": "huggingface",
                    "repo": spec.hf_repo,
                    "split": spec.hf_split,
                    "raw_row_index": i,
                    "purpose": "calibration",
                },
            }
        )
        if len(calibration) >= calibration_count:
            break

    if skipped_collisions:
        print(f"  [!] {spec.name}: skipped {skipped_collisions} rows that collided with test-set content")

    if len(calibration) < calibration_count:
        raise RuntimeError(
            f"[{spec.name}] only found {len(calibration)}/{calibration_count} "
            f"calibration rows after boundary {boundary} in {num_rows} total rows"
        )
    return boundary, calibration


def main() -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    raw_cache_dir = get_default_external_cache_dir()
    all_calibration: List[Dict[str, Any]] = []

    for task_name, spec in BENCHMARK_SPECS.items():
        task_info = manifest["tasks"][task_name]
        target_test_count = task_info["samples_count"]
        cache_file = raw_cache_dir / f"{spec.name}_{spec.hf_config}_{spec.hf_split}.parquet"
        if not cache_file.exists():
            raise FileNotFoundError(f"Missing raw cache for {task_name}: {cache_file}")

        boundary, calibration = find_boundary_and_calibration(
            spec, cache_file, target_test_count, SAMPLES_PER_TASK
        )
        print(
            f"  -> {task_name}: test consumed rows [0,{boundary}), "
            f"calibration rows start at {boundary + GAP}, "
            f"collected {len(calibration)} samples"
        )
        all_calibration.extend(calibration)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in all_calibration:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n[OK] wrote {len(all_calibration)} calibration samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
