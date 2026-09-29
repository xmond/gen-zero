import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
"""Checks for the real-source expansion and its on-disk integrity contract."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.datasets.fetch_real_datasets import (
    DEFAULT_DATA_DIR, DatasetFetcher, verify_generated,
)


def test_committed_manifest_and_aggregate():
    verify_generated(DEFAULT_DATA_DIR)
    manifest = json.loads((DEFAULT_DATA_DIR / "manifest.json").read_text())
    assert manifest["total_samples"] == 1700
    all_ids = [sample_id for entry in manifest["tasks"].values() for sample_id in entry["sample_ids"]]
    assert len(all_ids) == len(set(all_ids))
    for name, entry in manifest["tasks"].items():
        assert entry["samples_count"] == (400 if name == "paws" else 200 if name == "gsm8k" else 100)
        assert entry["sha256"] == hashlib.sha256((DEFAULT_DATA_DIR / entry["file"]).read_bytes()).hexdigest()
        assert sum(entry["label_counts"].values()) == entry["samples_count"]
        if name not in ("paws", "gsm8k"):
            assert max(entry["label_counts"].values()) - min(entry["label_counts"].values()) <= 1


def test_real_cache_deterministic_at_two_limits(tmp_path):
    cache = Path(os.environ["GEN_ZERO_EVAL_DATA_DIR"]) / "raw_datasets" if "GEN_ZERO_EVAL_DATA_DIR" in os.environ else DEFAULT_DATA_DIR.parents[3] / "gen-zero-eval-data" / "raw_datasets"
    required = cache / "boolq_default_validation.parquet"
    if not required.exists():
        pytest.skip(f"real parquet cache unavailable: {required}")
    fetcher = DatasetFetcher(output_dir=tmp_path, raw_cache_dir=cache)
    for count in (30, 100):
        first = fetcher.fetch_task_samples("boolq", count)
        second = fetcher.fetch_task_samples("boolq", count)
        assert first == second
        assert len(first) == count
        assert len({r["id"] for r in first}) == count
        counts = Counter(r["ground_truth"] for r in first)
        assert abs(counts["yes"] - counts["no"]) <= 1


def test_generation_two_sizes_and_task_override(tmp_path):
    cache = Path(os.environ["GEN_ZERO_EVAL_DATA_DIR"]) / "raw_datasets" if "GEN_ZERO_EVAL_DATA_DIR" in os.environ else DEFAULT_DATA_DIR.parents[3] / "gen-zero-eval-data" / "raw_datasets"
    if not (cache / "boolq_default_validation.parquet").exists():
        pytest.skip(f"real parquet cache unavailable: {cache}")
    for limit in (30, 40):
        output = tmp_path / str(limit)
        fetcher = DatasetFetcher(output_dir=output, raw_cache_dir=cache)
        manifest = fetcher.generate_all(samples_per_task=limit, task_limits={"boolq": limit + 2})
        verify_generated(output)
        assert manifest["tasks"]["boolq"]["samples_count"] == limit + 2
        assert manifest["tasks"]["squad2"]["samples_count"] == limit
        before = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
        fetcher.generate_all(samples_per_task=limit, task_limits={"boolq": limit + 2})
        assert hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest() == before


def test_verify_detects_tampered_hash(tmp_path):
    manifest = json.loads((DEFAULT_DATA_DIR / "manifest.json").read_text())
    for entry in manifest["tasks"].values():
        (tmp_path / entry["file"]).write_bytes((DEFAULT_DATA_DIR / entry["file"]).read_bytes())
    (tmp_path / "all_benchmarks.jsonl").write_bytes((DEFAULT_DATA_DIR / "all_benchmarks.jsonl").read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    verify_generated(tmp_path)
    path = tmp_path / manifest["tasks"]["boolq"]["file"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_generated(tmp_path)
