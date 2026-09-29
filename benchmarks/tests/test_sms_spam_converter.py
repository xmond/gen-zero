import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.datasets.converters.common import KEYS
from benchmarks.datasets.converters.sms_spam_converter import LABELS, convert, parse


def evaluation(tmp_path, contexts=()):
    path = tmp_path / "evaluation.jsonl"
    path.write_text("".join(json.dumps({"context": text}) + "\n" for text in contexts), encoding="utf-8")
    return path


def valid(rows):
    for row in rows:
        assert set(row) == KEYS
        assert all(isinstance(row[key], str) and row[key] for key in ("id", "task", "context", "ground_truth"))
        assert len(row["candidates"]) >= 2
        assert len(row["candidates"]) == len(set(row["candidates"]))
        assert all(candidate.strip() for candidate in row["candidates"])
        assert row["ground_truth"] in row["candidates"]
        assert isinstance(row["metadata"], dict)


def test_parse_splits_on_first_tab_only():
    raw = "ham\tCall me at 5\t30pm\nspam\tWIN a prize now"
    rows = list(parse(raw))
    assert rows == [{"label": "ham", "text": "Call me at 5\t30pm"}, {"label": "spam", "text": "WIN a prize now"}]


def test_parse_skips_blank_lines_and_rejects_lines_without_a_tab():
    rows = list(parse("ham\thello\n\nspam\twin now\n"))
    assert len(rows) == 2
    with pytest.raises(ValueError, match="malformed"):
        list(parse("ham\thello\nno tab here\n"))


def test_convert_schema_candidates_and_ground_truth(tmp_path):
    rows = [{"label": "ham", "text": "Ok lar joking wif u oni"},
             {"label": "spam", "text": "Free entry in 2 a wkly comp"}]
    converted = convert(rows, split="train", evaluation_path=evaluation(tmp_path))
    valid(converted)
    assert len(converted) == 2
    assert [row["ground_truth"] for row in converted] == ["ham", "spam"]
    assert all(row["candidates"] == LABELS for row in converted)
    assert all(row["task"] == "sms_spam" for row in converted)
    assert converted[0]["id"] == "sms_spam-train-00000"
    assert converted[1]["id"] == "sms_spam-train-00001"
    assert all(row["metadata"]["source"] == "uciml/sms-spam-collection-dataset" for row in converted)
    assert all(row["metadata"]["split"] == "train" for row in converted)
    assert [row["metadata"]["source_index"] for row in converted] == [0, 1]


def test_convert_ids_are_unique_and_sequential(tmp_path):
    rows = [{"label": "ham", "text": f"message number {i}"} for i in range(25)]
    converted = convert(rows, split="train", evaluation_path=evaluation(tmp_path))
    ids = [row["id"] for row in converted]
    assert len(ids) == len(set(ids)) == 25
    assert ids == [f"sms_spam-train-{i:05d}" for i in range(25)]


def test_convert_rejects_non_train_split(tmp_path):
    with pytest.raises(ValueError, match="train"):
        convert([], split="test", evaluation_path=evaluation(tmp_path))


def test_convert_rejects_invalid_label_or_empty_text(tmp_path):
    with pytest.raises(ValueError, match="invalid"):
        convert([{"label": "unknown", "text": "hello"}], split="train", evaluation_path=evaluation(tmp_path))
    with pytest.raises(ValueError, match="invalid"):
        convert([{"label": "ham", "text": "   "}], split="train", evaluation_path=evaluation(tmp_path))


def test_convert_drops_evaluation_overlap_case_and_whitespace_insensitive(tmp_path):
    rows = [{"label": "spam", "text": "  WIN a FREE prize now "},
             {"label": "ham", "text": "call me later"}]
    converted = convert(rows, split="train", evaluation_path=evaluation(tmp_path, ["win a free prize now"]))
    assert len(converted) == 1
    assert converted[0]["context"] == "call me later"


def test_convert_drops_duplicate_context_within_set(tmp_path):
    rows = [{"label": "ham", "text": "Ok lar joking wif u oni"},
             {"label": "ham", "text": "OK LAR   joking wif u oni"},
             {"label": "spam", "text": "unique spam message"}]
    stats = Counter()
    converted = convert(rows, split="train", evaluation_path=evaluation(tmp_path), stats=stats)
    assert len(converted) == 2
    assert stats["duplicate_context"] == 1
    assert stats["kept"] == 2


def test_convert_label_hit_rate_matches_source(tmp_path):
    rows = [{"label": "ham", "text": f"ham message {i}"} for i in range(10)] + \
           [{"label": "spam", "text": f"spam message {i}"} for i in range(5)]
    converted = convert(rows, split="train", evaluation_path=evaluation(tmp_path))
    assert sum(1 for row in converted if row["ground_truth"] == "ham") == 10
    assert sum(1 for row in converted if row["ground_truth"] == "spam") == 5
    for row in converted:
        expected_label = "ham" if "ham message" in row["context"] else "spam"
        assert row["ground_truth"] == expected_label
