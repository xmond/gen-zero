"""Mutation tests for the manifest registry and both evaluation entry points."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "datasets"))
sys.path.insert(0, str(BENCH / "suites"))
from dataset_registry import DatasetError, check_file, load_registry, registered_tasks
from clean_evaluation_harness import load_registered_dataset, ContractViolation


def fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    row = {"id": "sample-1", "task": "example", "context": "question",
           "candidates": ["yes", "no"], "ground_truth": "yes", "metadata": {}}
    raw = json.dumps(row) + "\n"
    (data / "example.jsonl").write_text(raw)
    (data / "all_benchmarks.jsonl").write_text(raw)
    entry = {"file": "example.jsonl", "sha256": hashlib.sha256(raw.encode()).hexdigest(),
             "samples_count": 1, "sample_ids": ["sample-1"]}
    manifest = {"task_groups": {"core_13": {"example": entry}}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data, row, entry


def test_registered_pass_and_harness(tmp_path):
    data, _, _ = fixture(tmp_path)
    _, rows = load_registry(data)
    assert rows["example"][0]["ground_truth"] == "yes"
    split = load_registered_dataset(data, "example")
    assert split.samples[0].label == "yes"


@pytest.mark.parametrize("field,value,reason", [
    ("ground_truth", "maybe", "ground_truth"),
    ("task", "other", "task"),
    ("candidates", ["yes", 3], "schema"),
    ("metadata", None, "schema"),
])
def test_invalid_row_with_recomputed_hash_fails(tmp_path, field, value, reason):
    data, row, entry = fixture(tmp_path)
    row[field] = value
    raw = json.dumps(row) + "\n"
    (data / "example.jsonl").write_text(raw)
    entry["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    with pytest.raises(DatasetError, match=reason):
        check_file(data, "example", entry)


def test_hash_count_ids_and_stray_fail(tmp_path):
    data, row, entry = fixture(tmp_path)
    (data / "example.jsonl").write_text("{}\n")
    with pytest.raises(DatasetError, match="sha256"):
        load_registry(data)
    raw = json.dumps(row) + "\n"
    (data / "example.jsonl").write_text(raw)
    entry["samples_count"] = 2
    with pytest.raises(DatasetError, match="count"):
        check_file(data, "example", entry)
    entry["samples_count"] = 1
    entry["sample_ids"] = ["wrong"]
    with pytest.raises(DatasetError, match="sample_ids"):
        check_file(data, "example", entry)
    (data / "rogue.jsonl").write_text(raw)
    with pytest.raises(DatasetError, match="unregistered"):
        load_registry(data)


def test_suite_selection_and_unknown(tmp_path):
    data, _, _ = fixture(tmp_path)
    manifest, rows = load_registry(data, "core_13")
    assert list(rows) == ["example"]
    with pytest.raises(DatasetError, match="unknown suite"):
        registered_tasks(manifest, "missing")
    with pytest.raises(ContractViolation, match="unregistered task"):
        load_registered_dataset(data, "missing")


def test_real_registry_has_all_extra_suites():
    manifest, rows = load_registry(BENCH / "data")
    assert {"bbh_boolean_expressions", "hans", "bespoke"} <= rows.keys()
    assert len(rows["bespoke"]) == 324
