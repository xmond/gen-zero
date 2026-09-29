# Calibration split: `calibration_clean_16.jsonl`

Purpose: legitimate few-shot calibration signal for the Continuous Causal
Reasoning Expert's dynamics matrix. Separate from, and zero-overlap with,
the 930-sample frozen test set in `benchmarks/data/manifest.json`.

## Locked hash

```
sha256(calibration_clean_16.jsonl) = bd45f4df430ee7c74440afca44b7880ac033e24013a34345ce4e54f1583efa4a
```

Enforced by `benchmarks/tests/test_calibration_split_isolation.py::test_calibration_sha256_locked`.
If the file is regenerated on purpose, recompute with `sha256sum` and update
both this doc and `EXPECTED_SHA256` in that test deliberately.

## Contents

312 samples total, 24 per task, across the same 13 tasks as the test manifest:

| task | test rows (frozen, on disk) | calibration source |
|---|---|---|
| massive_en | parquet rows `[0, 30)` | same `validation` split, rows `50+` (plain scan replay, exact match verified) |
| massive_de | parquet rows `[0, 30)` | same `validation` split, rows `50+` (exact match verified) |
| vitaminc | parquet rows `[0, 30)` | same `test` split, rows `50+` (exact match verified) |
| boolq | parquet rows `[0, 30)` | same `validation` split, rows `50+` (exact match verified) |
| squad2 | parquet rows `[0, 30)` | same `validation` split, rows `50+` (exact match verified) |
| paws | `rebuild_gsm8k_paws.py`, stratified sample of the `test` split (400 rows) | **different official split** (`validation`, 8000 rows), rows `420+` |
| civil_comments | parquet rows `[0, 30)` | same `test` split, rows `50+` (exact match verified) |
| aegis_safety | parquet rows `[0, 30)` | same `test` split, rows `50+` (exact match verified) |
| multinli | parquet rows `[0, 30)` | same `validation_matched` split, rows `50+` (exact match verified) |
| pubmedqa | parquet rows `[0, 30)` | same `train` split, rows `50+` (exact match verified) |
| summeval | parquet rows `[0, 30)` | same `test` split, rows `50+` (exact match verified) |
| arc_challenge | parquet rows `[0, 30)` | same `test` split, rows `50+` (exact match verified) |
| gsm8k | `rebuild_gsm8k_paws.py`, stratified-by-solution-length sample of the `test` split (200 rows) | same `test` split, rows `220+` (non-contiguous origin, no row-index proof) |

## How zero-overlap is actually guaranteed

Two independent mechanisms, not one:

1. **Row-index disjointness** (`benchmarks/datasets/build_calibration_split.py`).
   For 11 of 13 tasks, the frozen test file was built by
   `fetch_real_datasets.py`'s plain scan (first N valid rows, in parquet
   order). This script replays that exact scan with the current extractor,
   finds the boundary row, and starts calibration sampling 20 rows past it.
   Verified by direct replay: re-extracting rows `[0, boundary)` with the
   current code and diffing against the on-disk `{task}.jsonl` files gives
   an **exact match** for all 11 (see build log below).

   `paws` and `gsm8k` do **not** follow that plain scan — their test rows
   come from `rebuild_gsm8k_paws.py`, which does seeded stratified sampling
   over the full split (by lexical-overlap bucket for paws, by solution
   length for gsm8k), not a first-N scan. Replaying the plain-scan boundary
   for these two does not reproduce their real test rows (confirmed
   mismatch at row 0 for both). For `paws`, calibration instead draws from
   a different official HF split (`validation` vs. the test set's `test`),
   which is disjoint by construction. For `gsm8k`, both draw from `test`,
   and the row-index heuristic here is a head start, not a proof.

2. **Explicit content exclusion** (hard runtime check, all 13 tasks). Before
   accepting any calibration row, the builder loads every normalized
   `context` string already present in the real, on-disk `{task}.jsonl` test
   file and skips any candidate row that collides. This makes zero-overlap
   a construction-time guarantee for every task, independent of whether the
   row-index reasoning above holds.

3. **Test-time verification** (the actual gate):
   `benchmarks/tests/test_calibration_split_isolation.py` independently
   diffs `calibration_clean_16.jsonl`'s normalized contexts against all 930
   on-disk test contexts and asserts zero intersection, by ID and by text.
   This is what must pass in CI — mechanisms 1 and 2 are how the builder
   avoids relying on luck to satisfy it, not a substitute for it.

Known residual gap (documented, not silently ignored): the content check is
exact-string match on the full normalized context. It does not catch
sub-document reuse — e.g. a VitaminC calibration row citing the same
underlying claim as a test row with different phrasing, or a PAWS
calibration pair sharing one sentence with a test pair. Not observed in a
manual spot-check of the 312 rows, but not exhaustively verified either.

## Regeneration

```bash
python3 benchmarks/datasets/build_calibration_split.py
python3 -m pytest benchmarks/tests/test_calibration_split_isolation.py -v
```

Re-running the builder twice in a row (no source data changes) produced a
byte-identical file and SHA-256 both times, confirming the process is
deterministic.
