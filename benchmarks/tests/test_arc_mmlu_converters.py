"""Tests for the ARC and MMLU train-split converters and their leak gate.

Covers:
- six-field schema compliance ({"id","task","context","candidates","ground_truth","metadata"})
- candidates are non-empty and mutually distinct, ground_truth is always a member
- malformed upstream rows (missing fields, out-of-range answers, duplicate options,
  answer key absent from the option list) are rejected, never silently coerced
- the leak_gate dedup is byte-for-byte the same `text_hash` function
  build_diversity_train.py uses, and it genuinely removes a row whose context is
  copied verbatim from the frozen 930-question eval set
- a real, live pull of a handful of rows from Hugging Face for both ARC configs
  and MMLU's auxiliary_train, end to end through the leak gate (skipped only if
  the network is truly unreachable, never mocked)
- ARC digit-label normalization ("1".."4" -> "A".."D"), including the exact
  124-row (four-option) / 125-row (including one three-option outlier) count
  verified against the live ai2_arc train parquet
- the MMLU-Pro converter: 10-option (A-J) structure, 14-category taxonomy,
  ground_truth derived from answer_index rather than the answer letter, and a
  live pull through the leak gate
- cross-source dedup between ARC (primary) and MMLU auxiliary_train
  (secondary), including the real, live-verified overlap count under the
  pipeline's actual per-source-leak-gate-then-cross-source-dedup order
"""
from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

import pytest

TESTS_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = TESTS_DIR.parent
DATA_DIR = BENCHMARKS_DIR / "data"
CONVERTERS_DIR = BENCHMARKS_DIR / "datasets" / "converters"

SCHEMA_KEYS = {"id", "task", "context", "candidates", "ground_truth", "metadata"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Loaded once: each module inserts CONVERTERS_DIR onto sys.path itself so their
# own `from leak_gate import ...` resolves regardless of import order.
leak_gate = _load_module("leak_gate", CONVERTERS_DIR / "leak_gate.py")
arc_converter = _load_module("arc_converter", CONVERTERS_DIR / "arc_converter.py")
mmlu_converter = _load_module("mmlu_converter", CONVERTERS_DIR / "mmlu_converter.py")
mmlu_pro_converter = _load_module("mmlu_pro_converter", CONVERTERS_DIR / "mmlu_pro_converter.py")
cross_source_dedup = _load_module("cross_source_dedup", CONVERTERS_DIR / "cross_source_dedup.py")


def _network_reachable() -> bool:
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/datasets/allenai/ai2_arc/parquet",
            headers={"User-Agent": "gen-zero-test-probe/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10):
            return True
    except (urllib.error.URLError, OSError):
        return False


NETWORK_UP = _network_reachable()
skip_if_offline = pytest.mark.skipif(not NETWORK_UP, reason="huggingface.co unreachable from this sandbox")


def assert_six_field_schema(record: Dict[str, Any]) -> None:
    assert set(record.keys()) == SCHEMA_KEYS
    assert isinstance(record["id"], str) and record["id"]
    assert isinstance(record["task"], str) and record["task"]
    assert isinstance(record["context"], str) and record["context"]
    assert isinstance(record["candidates"], list) and record["candidates"]
    assert isinstance(record["ground_truth"], str) and record["ground_truth"]
    assert isinstance(record["metadata"], dict)
    assert all(isinstance(c, str) and c for c in record["candidates"])
    assert len(set(record["candidates"])) == len(record["candidates"]), "candidates must be mutually distinct"
    assert record["ground_truth"] in record["candidates"], "ground_truth must be a member of candidates"


# --------------------------------------------------------------------------
# ARC row -> record: field completeness, candidate distinctness, rejection paths
# --------------------------------------------------------------------------

ARC_GOOD_ROW = {
    "id": "Mercury_SC_415702",
    "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
    "choices": {
        "text": ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"],
        "label": ["A", "B", "C", "D"],
    },
    "answerKey": "A",
}


def test_arc_valid_row_produces_six_field_schema():
    record = arc_converter.arc_row_to_record(ARC_GOOD_ROW, "arc_challenge_train", "ARC-Challenge", 1)
    assert record is not None
    assert_six_field_schema(record)
    assert record["task"] == "arc_challenge_train"
    assert record["candidates"] == ["A", "B", "C", "D"]
    assert record["ground_truth"] == "A"
    assert record["metadata"]["repo"] == "allenai/ai2_arc"
    assert record["metadata"]["split"] == "train"
    assert "(A) dry palms" in record["context"]


@pytest.mark.parametrize(
    "mutation,description",
    [
        (lambda r: {**r, "question": ""}, "empty question"),
        (lambda r: {**r, "answerKey": ""}, "empty answer key"),
        (lambda r: {**r, "answerKey": "Z"}, "answer key absent from candidates"),
        (lambda r: {**r, "choices": {"text": [], "label": []}}, "empty choices"),
        (lambda r: {**r, "choices": {"text": ["only one"], "label": ["A"]}}, "fewer than 2 options"),
        (
            lambda r: {**r, "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "A", "C", "D"]}},
            "duplicate labels",
        ),
        (
            lambda r: {**r, "choices": {"text": ["same", "same", "c", "d"], "label": ["A", "B", "C", "D"]}},
            "duplicate option text",
        ),
        (
            lambda r: {**r, "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C", "D"]}},
            "label/text length mismatch",
        ),
        (
            lambda r: {**r, "choices": {"text": ["a", "", "c", "d"], "label": ["A", "B", "C", "D"]}},
            "blank option text",
        ),
    ],
)
def test_arc_malformed_rows_are_rejected(mutation, description):
    row = mutation(dict(ARC_GOOD_ROW))
    record = arc_converter.arc_row_to_record(row, "arc_challenge_train", "ARC-Challenge", 1)
    assert record is None, f"expected rejection for: {description}"


def test_arc_convert_split_skips_bad_rows_and_keeps_good_ones():
    bad_row = {**ARC_GOOD_ROW, "answerKey": "Z"}
    records = arc_converter.convert_arc_split([ARC_GOOD_ROW, bad_row, ARC_GOOD_ROW], "arc_easy_train", "ARC-Easy")
    assert len(records) == 2
    for r in records:
        assert_six_field_schema(r)


# --------------------------------------------------------------------------
# ARC digit-label normalization: 124 real ARC train rows (all New York Regents
# exam questions, verified directly against the live allenai/ai2_arc train
# parquet) ship answerKey/choices.label as "1".."4" instead of "A".."D". The
# state-manifold scorer's candidate space must be uniform, so these get
# normalized to letters just like every other ARC row.
# --------------------------------------------------------------------------

ARC_DIGIT_LABEL_ROW = {
    "id": "NYSEDREGENTS_2004_8_31",
    "question": "Which process most directly forms sedimentary rock?",
    "choices": {
        "text": ["compaction and cementation", "melting and cooling", "metamorphism", "crystallization"],
        "label": ["1", "2", "3", "4"],
    },
    "answerKey": "2",
}


def test_arc_digit_labels_are_normalized_to_letters():
    record = arc_converter.arc_row_to_record(ARC_DIGIT_LABEL_ROW, "arc_challenge_train", "ARC-Challenge", 1)
    assert record is not None
    assert_six_field_schema(record)
    assert record["candidates"] == ["A", "B", "C", "D"]
    assert record["ground_truth"] == "B"
    assert record["metadata"]["label_space_normalized"] is True
    assert "(A) compaction and cementation" in record["context"]
    assert "(B) melting and cooling" in record["context"]


def test_arc_letter_labels_are_not_flagged_as_normalized():
    record = arc_converter.arc_row_to_record(ARC_GOOD_ROW, "arc_challenge_train", "ARC-Challenge", 1)
    assert record["metadata"]["label_space_normalized"] is False


def test_arc_three_option_digit_labels_normalize_too():
    row = {
        **ARC_DIGIT_LABEL_ROW,
        "choices": {"text": ["a", "b", "c"], "label": ["1", "2", "3"]},
        "answerKey": "3",
    }
    record = arc_converter.arc_row_to_record(row, "arc_easy_train", "ARC-Easy", 1)
    assert record["candidates"] == ["A", "B", "C"]
    assert record["ground_truth"] == "C"


@skip_if_offline
def test_live_arc_train_has_exactly_124_four_option_digit_labeled_rows(tmp_path):
    """Reviewer-reported finding, verified against the full live ai2_arc train
    parquet for both configs: exactly 124 rows ship labels ("1","2","3","4"),
    plus one further outlier row with three-option digit labels ("1","2","3").
    All must survive conversion with candidates normalized to letters, none
    dropped and none left with a digit in `candidates` or `ground_truth`.
    """
    four_option_normalized = 0
    total_normalized = 0
    total_rows = 0
    for task_name, config in arc_converter.ARC_CONFIGS.items():
        records = arc_converter.fetch_and_convert(task_name, config, tmp_path / "cache")
        total_rows += len(records)
        for record in records:
            assert set(record["candidates"]) <= set(arc_converter._LETTERS)
            assert record["ground_truth"] in record["candidates"]
            if record["metadata"]["label_space_normalized"]:
                total_normalized += 1
                if len(record["candidates"]) == 4:
                    four_option_normalized += 1
    assert total_rows > 3000, f"expected the full ARC train pool (~3370 rows), got {total_rows}"
    assert four_option_normalized == 124, f"expected exactly 124 four-option digit-labeled rows, got {four_option_normalized}"
    assert total_normalized == 125, f"expected 125 total digit-labeled rows (124 four-option + 1 three-option), got {total_normalized}"


# --------------------------------------------------------------------------
# MMLU row -> record
# --------------------------------------------------------------------------

MMLU_GOOD_ROW = {
    "question": "Which gas do plants absorb from the atmosphere during photosynthesis?",
    "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"],
    "answer": 1,
    "subject": "",
}


def test_mmlu_valid_row_produces_six_field_schema():
    record = mmlu_converter.mmlu_row_to_record(MMLU_GOOD_ROW, 1)
    assert record is not None
    assert_six_field_schema(record)
    assert record["task"] == "mmlu_auxiliary_train"
    assert record["candidates"] == ["A", "B", "C", "D"]
    assert record["ground_truth"] == "B"
    assert record["metadata"]["repo"] == "cais/mmlu"
    assert record["metadata"]["config"] == "auxiliary_train"
    # Upstream auxiliary_train genuinely ships subject="" for every row; the
    # converter must record that honestly instead of guessing a category.
    assert record["metadata"]["subject"] == "unspecified"


def test_mmlu_row_with_labeled_subject_is_preserved_not_overwritten():
    row = {**MMLU_GOOD_ROW, "subject": "college_biology"}
    record = mmlu_converter.mmlu_row_to_record(row, 1)
    assert record["metadata"]["subject"] == "college_biology"


@pytest.mark.parametrize(
    "mutation,description",
    [
        (lambda r: {**r, "question": "  "}, "blank question"),
        (lambda r: {**r, "choices": ["only one"]}, "fewer than 2 options"),
        (lambda r: {**r, "choices": ["a", "a", "c", "d"]}, "duplicate option text"),
        (lambda r: {**r, "choices": ["a", "", "c", "d"]}, "blank option text"),
        (lambda r: {**r, "answer": 4}, "answer index out of range"),
        (lambda r: {**r, "answer": -1}, "negative answer index"),
        (lambda r: {**r, "answer": None}, "missing answer"),
        (lambda r: {**r, "answer": "1"}, "non-integer answer type"),
        (lambda r: {**r, "answer": True}, "boolean answer rejected despite being an int subclass"),
    ],
)
def test_mmlu_malformed_rows_are_rejected(mutation, description):
    row = mutation(dict(MMLU_GOOD_ROW))
    record = mmlu_converter.mmlu_row_to_record(row, 1)
    assert record is None, f"expected rejection for: {description}"


def test_mmlu_convert_split_skips_bad_rows_and_keeps_good_ones():
    bad_row = {**MMLU_GOOD_ROW, "answer": 99}
    records = mmlu_converter.convert_mmlu_split([MMLU_GOOD_ROW, bad_row, MMLU_GOOD_ROW])
    assert len(records) == 2
    for r in records:
        assert_six_field_schema(r)


# --------------------------------------------------------------------------
# MMLU-Pro row -> record: 10-option (A-J) structure, 14-category taxonomy,
# ground_truth derived from the dataset's own integer answer_index (never
# from the `answer` letter field, matching this project's rule of never
# trusting an upstream label when an ordinal position is available instead).
# --------------------------------------------------------------------------

MMLU_PRO_GOOD_ROW = {
    "question_id": 70,
    "question": "Typical advertising regulatory bodies suggest that adverts must not encourage unsafe practices.",
    "options": [
        "Safe practices, Fear, Jealousy, Trivial",
        "Unsafe practices, Distress, Joy, Trivial",
        "Safe practices, Wants, Jealousy, Trivial",
        "Safe practices, Distress, Fear, Trivial",
        "Unsafe practices, Wants, Jealousy, Serious",
        "Safe practices, Distress, Jealousy, Serious",
        "Safe practices, Wants, Fear, Serious",
        "Unsafe practices, Wants, Fear, Trivial",
        "Unsafe practices, Distress, Fear, Serious",
    ],
    "answer": "I",
    "answer_index": 8,
    "category": "business",
    "src": "ori_mmlu-business_ethics",
}


def test_mmlu_pro_valid_row_produces_six_field_schema():
    record = mmlu_pro_converter.mmlu_pro_row_to_record(MMLU_PRO_GOOD_ROW, 1)
    assert record is not None
    assert_six_field_schema(record)
    assert record["task"] == "mmlu_pro_test"
    assert record["candidates"] == ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    assert record["ground_truth"] == "I"
    assert record["metadata"]["repo"] == "TIGER-Lab/MMLU-Pro"
    assert record["metadata"]["category"] == "business"
    assert record["metadata"]["src"] == "ori_mmlu-business_ethics"
    assert record["metadata"]["split"] == "test"
    assert record["metadata"]["license"] == "MIT"


def test_mmlu_pro_ten_option_row_uses_labels_a_through_j():
    row = {**MMLU_PRO_GOOD_ROW, "options": [f"option {i}" for i in range(10)], "answer_index": 9, "answer": "J"}
    record = mmlu_pro_converter.mmlu_pro_row_to_record(row, 1)
    assert record["candidates"] == list("ABCDEFGHIJ")
    assert record["ground_truth"] == "J"


def test_mmlu_pro_derives_ground_truth_from_answer_index_not_answer_letter():
    # answer_index is authoritative; a mismatched answer letter must not
    # override the ordinal position used to build candidates/ground_truth.
    row = {**MMLU_PRO_GOOD_ROW, "answer": "Z", "answer_index": 2}
    record = mmlu_pro_converter.mmlu_pro_row_to_record(row, 1)
    assert record["ground_truth"] == "C"


@pytest.mark.parametrize(
    "mutation,description",
    [
        (lambda r: {**r, "question": "  "}, "blank question"),
        (lambda r: {**r, "options": ["only one"]}, "fewer than 2 options"),
        (lambda r: {**r, "options": ["a", "a", "c", "d"]}, "duplicate option text"),
        (lambda r: {**r, "options": ["a", "", "c", "d"]}, "blank option text"),
        (lambda r: {**r, "answer_index": 99}, "answer_index out of range"),
        (lambda r: {**r, "answer_index": -1}, "negative answer_index"),
        (lambda r: {**r, "answer_index": None}, "missing answer_index"),
        (lambda r: {**r, "answer_index": "8"}, "non-integer answer_index type"),
        (lambda r: {**r, "answer_index": True}, "boolean answer_index rejected despite being an int subclass"),
    ],
)
def test_mmlu_pro_malformed_rows_are_rejected(mutation, description):
    row = mutation(dict(MMLU_PRO_GOOD_ROW))
    record = mmlu_pro_converter.mmlu_pro_row_to_record(row, 1)
    assert record is None, f"expected rejection for: {description}"


def test_mmlu_pro_convert_split_skips_bad_rows_and_keeps_good_ones():
    bad_row = {**MMLU_PRO_GOOD_ROW, "answer_index": 999}
    records = mmlu_pro_converter.convert_mmlu_pro_split([MMLU_PRO_GOOD_ROW, bad_row, MMLU_PRO_GOOD_ROW])
    assert len(records) == 2
    for r in records:
        assert_six_field_schema(r)


@skip_if_offline
def test_live_mmlu_pro_pull_is_schema_valid_leak_free_and_uses_known_categories(tmp_path):
    excluded = leak_gate.load_excluded_hashes()
    records = mmlu_pro_converter.fetch_and_convert(tmp_path / "cache", limit=50)
    assert records, "live MMLU-Pro fetch returned zero rows"
    kept, stats = leak_gate.filter_leak_free(records, excluded)
    assert stats["leaked"] == 0, "real MMLU-Pro rows collided with the frozen eval set"
    assert kept, "nothing survived the leak gate"
    for record in kept:
        assert_six_field_schema(record)
        assert record["metadata"]["config"] == "default"
        assert record["metadata"]["category"] in mmlu_pro_converter.MMLU_PRO_CATEGORIES


# --------------------------------------------------------------------------
# leak_gate: shares build_diversity_train's exact hash function, and actually
# removes rows overlapping the real frozen eval set.
# --------------------------------------------------------------------------

def test_leak_gate_reuses_build_diversity_train_text_hash_object():
    # leak_gate.py loads build_diversity_train.py itself (registering it in
    # sys.modules under that exact name) precisely so it can import this
    # function instead of redefining it. Confirm it really is the same
    # function object, not a second, potentially-drifting copy.
    build_diversity_train = sys.modules["build_diversity_train"]
    assert leak_gate.text_hash is build_diversity_train.text_hash
    assert build_diversity_train.text_hash("Hello   World") == leak_gate.text_hash("hello world")


def test_load_excluded_hashes_contains_a_known_frozen_eval_context():
    with open(DATA_DIR / "arc_challenge.jsonl", encoding="utf-8") as f:
        first_eval_row = json.loads(f.readline())
    excluded = leak_gate.load_excluded_hashes()
    assert leak_gate.text_hash(first_eval_row["context"]) in excluded
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_samples"] == 1700
    assert len(excluded) > 0


def test_load_excluded_hashes_covers_every_row_of_all_benchmarks_jsonl():
    # The task names benchmarks/data/all_benchmarks.jsonl (1700 rows) explicitly
    # as the file to gate against. load_excluded_hashes() derives its set from
    # manifest.json's per-task files instead of reading this aggregate file
    # directly, so assert directly that the two are equivalent: every context
    # hash in all_benchmarks.jsonl must be present in the excluded set.
    all_benchmarks_path = DATA_DIR / "all_benchmarks.jsonl"
    with open(all_benchmarks_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1700

    excluded = leak_gate.load_excluded_hashes()
    missing = [r["id"] for r in rows if leak_gate.text_hash(r["context"]) not in excluded]
    assert missing == [], f"{len(missing)} rows from all_benchmarks.jsonl are not covered by the leak gate"


def test_filter_leak_free_drops_row_copied_from_frozen_eval_set():
    with open(DATA_DIR / "arc_challenge.jsonl", encoding="utf-8") as f:
        leaked_eval_row = json.loads(f.readline())
    excluded = leak_gate.load_excluded_hashes()

    leaked_candidate = {
        "id": "arc_challenge_train-99999",
        "task": "arc_challenge_train",
        "context": leaked_eval_row["context"],  # verbatim copy of a real eval-set context
        "candidates": ["A", "B", "C", "D"],
        "ground_truth": "A",
        "metadata": {"source": "huggingface"},
    }
    clean_candidate = {
        "id": "arc_challenge_train-00001",
        "task": "arc_challenge_train",
        "context": "Question: This context is unique and appears nowhere in the frozen eval set.\n(A) x\n(B) y\nSelect the single correct option.",
        "candidates": ["A", "B"],
        "ground_truth": "A",
        "metadata": {"source": "huggingface"},
    }

    kept, stats = leak_gate.filter_leak_free([leaked_candidate, clean_candidate], excluded)

    assert stats["leaked"] == 1
    assert stats["kept"] == 1
    assert [r["id"] for r in kept] == ["arc_challenge_train-00001"]


def test_filter_leak_free_drops_whitespace_and_case_variant_of_a_leaked_context():
    with open(DATA_DIR / "arc_challenge.jsonl", encoding="utf-8") as f:
        leaked_eval_row = json.loads(f.readline())
    excluded = leak_gate.load_excluded_hashes()

    # Same content, different casing and whitespace -- the casefold + whitespace
    # normalization in text_hash must still catch it.
    mutated_context = "  ".join(leaked_eval_row["context"].upper().split())
    candidate = {
        "id": "arc_challenge_train-88888",
        "task": "arc_challenge_train",
        "context": mutated_context,
        "candidates": ["A", "B", "C", "D"],
        "ground_truth": "A",
        "metadata": {},
    }
    kept, stats = leak_gate.filter_leak_free([candidate], excluded)
    assert stats["leaked"] == 1
    assert kept == []


def test_filter_leak_free_drops_in_batch_content_and_id_duplicates():
    record_a = {
        "id": "x-0001",
        "task": "x",
        "context": "unique context alpha",
        "candidates": ["A", "B"],
        "ground_truth": "A",
        "metadata": {},
    }
    record_dup_context = {**record_a, "id": "x-0002"}
    record_dup_id = {**record_a, "id": "x-0001", "context": "unique context beta"}

    kept, stats = leak_gate.filter_leak_free([record_a, record_dup_context, record_dup_id], set())
    assert stats["kept"] == 1
    assert stats["duplicate_in_batch"] == 1
    assert stats["duplicate_id"] == 1
    assert [r["id"] for r in kept] == ["x-0001"]


def test_filter_leak_free_keeps_genuinely_novel_rows():
    records = [
        {
            "id": f"novel-{i:04d}",
            "task": "novel",
            "context": f"a genuinely unique context body number {i}",
            "candidates": ["A", "B"],
            "ground_truth": "A",
            "metadata": {},
        }
        for i in range(5)
    ]
    kept, stats = leak_gate.filter_leak_free(records, set())
    assert stats["kept"] == 5
    assert stats["leaked"] == 0


# --------------------------------------------------------------------------
# Cross-source dedup: cais/mmlu's own data card documents auxiliary_train as
# assembled in part from ARC. A row can be leak-free against the frozen eval
# set and still be a same-content duplicate of another train-source row; this
# is a separate concern from leak_gate and must be caught before a fused
# training pool double-weights the same question.
# --------------------------------------------------------------------------

def test_cross_source_deduplicate_drops_secondary_rows_matching_primary():
    primary = [
        {
            "id": "arc-0001",
            "task": "arc_challenge_train",
            "context": "Question: shared context alpha\n(A) x\n(B) y\nSelect the single correct option.",
            "candidates": ["A", "B"],
            "ground_truth": "A",
            "metadata": {},
        }
    ]
    secondary = [
        {
            "id": "mmlu-0001",
            "task": "mmlu_auxiliary_train",
            "context": "Question: shared context alpha\n(A) x\n(B) y\nSelect the single correct option.",
            "candidates": ["A", "B"],
            "ground_truth": "A",
            "metadata": {},
        },
        {
            "id": "mmlu-0002",
            "task": "mmlu_auxiliary_train",
            "context": "Question: genuinely novel context beta\n(A) x\n(B) y\nSelect the single correct option.",
            "candidates": ["A", "B"],
            "ground_truth": "A",
            "metadata": {},
        },
    ]
    kept, stats = cross_source_dedup.cross_source_deduplicate(primary, secondary)
    assert stats == {
        "primary_count": 1,
        "secondary_input": 2,
        "cross_source_duplicates": 1,
        "secondary_internal_duplicates": 0,
        "kept": 1,
    }
    assert [r["id"] for r in kept] == ["mmlu-0002"]


def test_cross_source_deduplicate_also_drops_internal_secondary_duplicates():
    secondary = [
        {"id": "s1", "task": "t", "context": "same body", "candidates": ["A", "B"], "ground_truth": "A", "metadata": {}},
        {"id": "s2", "task": "t", "context": "same body", "candidates": ["A", "B"], "ground_truth": "A", "metadata": {}},
    ]
    kept, stats = cross_source_dedup.cross_source_deduplicate([], secondary)
    assert stats["secondary_internal_duplicates"] == 1
    assert stats["kept"] == 1
    assert [r["id"] for r in kept] == ["s1"]


def test_cross_source_deduplicate_never_filters_the_primary_list():
    primary = [
        {"id": "p1", "task": "t", "context": "unique to primary", "candidates": ["A", "B"], "ground_truth": "A", "metadata": {}},
    ]
    kept, stats = cross_source_dedup.cross_source_deduplicate(primary, [])
    assert kept == []
    assert stats["primary_count"] == 1
    # primary itself is returned unmodified by the caller, not filtered by this function
    assert primary == [
        {"id": "p1", "task": "t", "context": "unique to primary", "candidates": ["A", "B"], "ground_truth": "A", "metadata": {}}
    ]


def test_deduplicate_mmlu_against_arc_is_the_arc_primary_wrapper():
    arc_records = [
        {"id": "arc-1", "task": "arc_challenge_train", "context": "dup body", "candidates": ["A"], "ground_truth": "A", "metadata": {}},
    ]
    mmlu_records = [
        {"id": "mmlu-1", "task": "mmlu_auxiliary_train", "context": "dup body", "candidates": ["A"], "ground_truth": "A", "metadata": {}},
    ]
    kept, stats = cross_source_dedup.deduplicate_mmlu_against_arc(arc_records, mmlu_records)
    assert kept == []
    assert stats["cross_source_duplicates"] == 1


@skip_if_offline
def test_live_full_scale_cross_source_dedup_finds_3359_arc_mmlu_overlaps(tmp_path):
    """Verified against the full live parquet for both sources, run in the
    same order the real pipeline uses: each source goes through its own
    converter's leak_gate.filter_leak_free first (matching what
    arc_converter.py / mmlu_converter.py already do in their own `main()`),
    *then* cross_source_dedup compares the two leak-gate-filtered pools.
    Real result: ARC leak-gate-filters to 3,369 rows, MMLU auxiliary_train
    leak-gate-filters to 98,589 rows (dropping 1,202 rows that were exact
    duplicates of each other within MMLU itself), and 3,359 of the remaining
    MMLU rows are verbatim duplicates of an ARC row.

    An earlier reviewer-reported figure of "4,306" was measured before ARC's
    124 digit-labeled rows were normalized to letters, and before per-source
    leak_gate filtering was applied first. Both of those are now real, and
    3,359 is what they produce -- not 4,306. See cross_source_dedup.py's
    module docstring for the full accounting. The final kept-MMLU-row count
    (95,230) is invariant to this ordering choice, and that invariant is
    checked separately below.
    """
    excluded = leak_gate.load_excluded_hashes()

    arc_records = []
    for task_name, config in arc_converter.ARC_CONFIGS.items():
        records = arc_converter.fetch_and_convert(task_name, config, tmp_path / "cache")
        kept, _ = leak_gate.filter_leak_free(records, excluded)
        arc_records.extend(kept)
    assert len(arc_records) > 3000, f"expected the full ARC train pool, got {len(arc_records)}"

    mmlu_records = mmlu_converter.fetch_and_convert(tmp_path / "cache")
    mmlu_kept, _ = leak_gate.filter_leak_free(mmlu_records, excluded)
    assert len(mmlu_kept) > 90000, f"expected close to the full 99,842-row auxiliary_train, got {len(mmlu_kept)}"

    deduped, stats = cross_source_dedup.deduplicate_mmlu_against_arc(arc_records, mmlu_kept)
    assert stats["cross_source_duplicates"] == 3359, (
        f"expected 3,359 cross-source ARC/MMLU duplicates (post leak-gate, post label-normalization), "
        f"got {stats['cross_source_duplicates']}"
    )
    assert stats["secondary_internal_duplicates"] == 0, "leak_gate already absorbed MMLU's internal duplicates upstream"
    assert stats["kept"] == 95230
    assert len(deduped) == stats["kept"]


@skip_if_offline
def test_live_full_scale_cross_source_dedup_finds_the_same_kept_count_either_order(tmp_path):
    """Whether cross-source dedup runs after per-source leak_gate filtering
    (the real pipeline order) or directly on raw converted output, the same
    unique MMLU rows survive -- only the bucket a given removed row is
    attributed to (cross-source vs. internal-duplicate) shifts. This is the
    invariant that makes either measurement honest: 95,230 kept MMLU rows.
    """
    excluded = leak_gate.load_excluded_hashes()

    arc_raw = []
    for task_name, config in arc_converter.ARC_CONFIGS.items():
        arc_raw.extend(arc_converter.fetch_and_convert(task_name, config, tmp_path / "cache"))
    mmlu_raw = mmlu_converter.fetch_and_convert(tmp_path / "cache")

    # Order A: leak-gate each source first, then cross-source dedup.
    arc_gated, _ = leak_gate.filter_leak_free(arc_raw, excluded)
    mmlu_gated, _ = leak_gate.filter_leak_free(mmlu_raw, excluded)
    _, stats_a = cross_source_dedup.deduplicate_mmlu_against_arc(arc_gated, mmlu_gated)

    # Order B: cross-source dedup directly on raw converted output.
    _, stats_b = cross_source_dedup.deduplicate_mmlu_against_arc(arc_raw, mmlu_raw)

    assert stats_a["kept"] == stats_b["kept"] == 95230


# --------------------------------------------------------------------------
# End-to-end, live upstream pull (real network, never mocked). Skipped only if
# the sandbox truly has no route to huggingface.co.
# --------------------------------------------------------------------------

@skip_if_offline
def test_live_arc_train_pull_is_schema_valid_and_leak_free(tmp_path):
    excluded = leak_gate.load_excluded_hashes()
    for task_name, config in arc_converter.ARC_CONFIGS.items():
        records = arc_converter.fetch_and_convert(task_name, config, tmp_path / "cache", limit=15)
        assert records, f"{task_name}: live fetch returned zero rows"
        kept, stats = leak_gate.filter_leak_free(records, excluded)
        assert stats["leaked"] == 0, f"{task_name}: real ARC train rows collided with the frozen eval set"
        assert kept, f"{task_name}: nothing survived the leak gate"
        for record in kept:
            assert_six_field_schema(record)
            assert record["metadata"]["split"] == "train"


@skip_if_offline
def test_live_mmlu_auxiliary_train_pull_is_schema_valid_and_leak_free(tmp_path):
    excluded = leak_gate.load_excluded_hashes()
    records = mmlu_converter.fetch_and_convert(tmp_path / "cache", limit=25)
    assert records, "live MMLU fetch returned zero rows"
    kept, stats = leak_gate.filter_leak_free(records, excluded)
    assert stats["leaked"] == 0, "real MMLU auxiliary_train rows collided with the frozen eval set"
    assert kept, "nothing survived the leak gate"
    for record in kept:
        assert_six_field_schema(record)
        assert record["metadata"]["config"] == "auxiliary_train"


@skip_if_offline
def test_live_full_scale_arc_and_mmlu_pull_has_zero_exact_text_collisions(tmp_path):
    """Whole-dataset run (no --limit): 1119 + 2251 ARC rows and 99,842 MMLU
    auxiliary_train rows, all checked against the real frozen 930-question
    eval set. MMLU's own data card documents auxiliary_train as assembled
    from other exam sources including ARC, so an exact-text collision here is
    a real (if unlikely) possibility this test would catch, not a formality.
    This is the strongest available evidence the physical leak gate works: it
    is reporting what it actually found on genuine upstream data, not merely
    running without raising."""
    excluded = leak_gate.load_excluded_hashes()

    total_leaked = 0
    for task_name, config in arc_converter.ARC_CONFIGS.items():
        records = arc_converter.fetch_and_convert(task_name, config, tmp_path / "cache")
        kept, stats = leak_gate.filter_leak_free(records, excluded)
        total_leaked += stats["leaked"]
        assert stats["input"] > 1000, f"{task_name}: suspiciously few rows ({stats['input']}), expected the full split"

    mmlu_records = mmlu_converter.fetch_and_convert(tmp_path / "cache")
    kept, stats = leak_gate.filter_leak_free(mmlu_records, excluded)
    total_leaked += stats["leaked"]
    assert stats["input"] > 90000, f"expected close to the full 99,842-row auxiliary_train, got {stats['input']}"

    assert total_leaked == 0, (
        f"{total_leaked} exact-text collisions found between real ARC/MMLU train data "
        "and the frozen 930-question eval set"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
