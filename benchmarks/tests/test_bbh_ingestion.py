"""Tests for BIG-Bench Hard (BBH) ingestion: benchmarks/datasets/convert_new_benchmarks.py.

Two layers, per the module's own docstring contract:
  (a) invariant tests over the committed benchmarks/data/bbh_*.jsonl fixtures --
      all 27 tasks present, strict 6-key schema, ground_truth in candidates,
      no empty/duplicate candidates, answer_mode matches the task's own design.
  (b) parser unit tests against inline fixture strings (no /ebs/tmp dependency,
      no network) -- lettered options, dashed options (including the
      formal_fallacies "valid " trailing-space quirk), boolean/yes-no/valid-
      invalid tasks with no Options: block, the free-form exact_match
      allow-list, and the failure paths: unrecognized layout must raise,
      known upstream annotation noise (comma-split titles) must skip-and-log
      rather than either crash the whole file or silently fake a match.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.datasets.convert_new_benchmarks import (  # noqa: E402
    BBH_TASK_NAMES,
    BBHFormatError,
    BBHGroundTruthMismatch,
    _normalize_bbh_example,
    _split_bbh_options,
    convert_bbh_task,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SIX_KEYS = {"id", "task", "context", "candidates", "ground_truth", "metadata"}
FREE_FORM_TASKS = {"dyck_languages", "multistep_arithmetic_two", "object_counting", "word_sorting"}

assert len(BBH_TASK_NAMES) == 27, "BBH has exactly 27 tasks"
assert FREE_FORM_TASKS <= set(BBH_TASK_NAMES)


def _read_rows(task_name: str) -> list[dict]:
    path = DATA_DIR / f"bbh_{task_name}.jsonl"
    assert path.exists(), f"missing committed fixture: {path}"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, f"{path} is empty"
    return rows


# ---------------------------------------------------------------- (a) fixtures on disk

def test_all_27_bbh_task_files_exist():
    missing = [t for t in BBH_TASK_NAMES if not (DATA_DIR / f"bbh_{t}.jsonl").exists()]
    assert not missing, f"missing bbh_<task>.jsonl for: {missing}"


@pytest.mark.parametrize("task_name", BBH_TASK_NAMES)
def test_schema_is_exactly_six_keys_correct_types(task_name):
    for row in _read_rows(task_name):
        assert set(row.keys()) == SIX_KEYS, f"{task_name}/{row.get('id')}: dirty/missing keys {set(row) ^ SIX_KEYS}"
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["task"], str) and row["task"] == f"bbh_{task_name}"
        assert isinstance(row["context"], str) and row["context"]
        assert isinstance(row["candidates"], list)
        assert isinstance(row["ground_truth"], str) and row["ground_truth"]
        assert isinstance(row["metadata"], dict)


@pytest.mark.parametrize("task_name", BBH_TASK_NAMES)
def test_ground_truth_always_in_candidates(task_name):
    for row in _read_rows(task_name):
        assert row["ground_truth"] in row["candidates"], f"{task_name}/{row['id']}: gt not in candidates"


@pytest.mark.parametrize("task_name", BBH_TASK_NAMES)
def test_candidates_no_empty_no_duplicates(task_name):
    for row in _read_rows(task_name):
        cands = row["candidates"]
        assert cands, f"{task_name}/{row['id']}: empty candidates list"
        assert all(isinstance(c, str) and c.strip() for c in cands), f"{task_name}/{row['id']}: blank candidate"
        assert len(set(cands)) == len(cands), f"{task_name}/{row['id']}: duplicate candidates {cands}"


@pytest.mark.parametrize("task_name", BBH_TASK_NAMES)
def test_ids_unique_within_task(task_name):
    rows = _read_rows(task_name)
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), f"{task_name}: duplicate ids"


@pytest.mark.parametrize("task_name", BBH_TASK_NAMES)
def test_answer_mode_matches_task_design(task_name):
    expected = "exact_match" if task_name in FREE_FORM_TASKS else "choice"
    for row in _read_rows(task_name):
        assert row["metadata"]["answer_mode"] == expected, (
            f"{task_name}/{row['id']}: expected answer_mode={expected}, "
            f"got {row['metadata']['answer_mode']}"
        )
        assert row["metadata"]["task_name"] == task_name
        assert row["metadata"]["source"] == "bbh"


def test_exact_match_tasks_are_honestly_single_candidate_not_fabricated_choice():
    """The 4 free-form tasks must not be dressed up as a fake multi-way choice:
    candidates is exactly [ground_truth], never invented distractors."""
    for task_name in FREE_FORM_TASKS:
        for row in _read_rows(task_name):
            assert row["candidates"] == [row["ground_truth"]]


def test_choice_tasks_have_real_discriminative_candidate_sets():
    """Every non-free-form task must offer at least 2 real candidates -- this
    is the regression test for the original bug (single-candidate cheat)."""
    for task_name in BBH_TASK_NAMES:
        if task_name in FREE_FORM_TASKS:
            continue
        for row in _read_rows(task_name):
            assert len(row["candidates"]) >= 2, f"{task_name}/{row['id']}: not a real choice task"


def test_total_sample_count_and_known_upstream_noise_exclusions():
    """Locks in the full-corpus regeneration: 6507 records total, with exactly
    4 known-bad upstream records excluded -- 3 comma-split titles
    (movie_recommendation x1, ruin_names x2) and 1 truncated input
    (snarks[88], only "(A) The NB" with no (B) printed) -- see
    BBHGroundTruthMismatch's docstring."""
    counts = {t: len(_read_rows(t)) for t in BBH_TASK_NAMES}
    assert sum(counts.values()) == 6507
    assert counts["movie_recommendation"] == 249
    assert counts["ruin_names"] == 248
    assert counts["penguins_in_a_table"] == 146
    assert counts["causal_judgement"] == 187
    assert counts["snarks"] == 177


# ---------------------------------------------------------------- (b) parser unit tests

def test_lettered_options_parsed_to_bare_uppercase_letters():
    inp = "Which is correct?\nOptions:\n(A) midsize old grey sweater\n(B) midsize grey old sweater"
    cands = _split_bbh_options(inp)
    assert cands == ["A", "B"]


def test_dashed_options_with_trailing_space_parsed():
    """formal_fallacies' own raw data has a literal trailing space after 'valid '."""
    inp = "Is it valid?\nOptions:\n- valid \n- invalid"
    cands = _split_bbh_options(inp)
    assert cands == ["valid", "invalid"]


def test_no_options_marker_returns_none():
    assert _split_bbh_options("Evaluate: ((1 + 2)) =") is None


def test_options_marker_with_non_uniform_lines_raises():
    inp = "Pick one.\nOptions:\n(A) first\nnot an option line at all\n(B) second"
    with pytest.raises(BBHFormatError):
        _split_bbh_options(inp)


def test_options_marker_with_no_lines_raises():
    inp = "Pick one.\nOptions:\n"
    with pytest.raises(BBHFormatError):
        _split_bbh_options(inp)


def test_normalize_lettered_example():
    inp = "Which is correct?\nOptions:\n(A) midsize old grey sweater\n(B) midsize grey old sweater"
    cands, gt, mode = _normalize_bbh_example("hyperbaton", 0, inp, "(A)")
    assert (cands, gt, mode) == (["A", "B"], "A", "choice")


def test_normalize_dashed_example():
    inp = "Is it valid?\nOptions:\n- valid \n- invalid"
    cands, gt, mode = _normalize_bbh_example("formal_fallacies", 0, inp, "invalid")
    assert (cands, gt, mode) == (["valid", "invalid"], "invalid", "choice")


def test_normalize_boolean_expressions_no_options_block():
    cands, gt, mode = _normalize_bbh_example("boolean_expressions", 0, "not True and False =", "False")
    assert (cands, gt, mode) == (["True", "False"], "False", "choice")


def test_normalize_free_form_allowlisted_task_is_exact_match_not_fabricated():
    cands, gt, mode = _normalize_bbh_example("object_counting", 0, "How many fruits?", "3")
    assert (cands, gt, mode) == (["3"], "3", "exact_match")


def test_normalize_unknown_free_form_task_raises_instead_of_guessing():
    """A task with no Options: block, a target that isn't a known binary
    keyword, and that is NOT on the free-form allow-list must fail loud --
    never silently fall back to a fabricated single-candidate list."""
    with pytest.raises(BBHFormatError):
        _normalize_bbh_example("some_new_bbh_task_nobody_reviewed", 0, "Do the thing.", "some free text answer")


def test_normalize_ground_truth_not_in_candidates_raises_the_mismatch_subclass():
    inp = "Which is correct?\nOptions:\n(A) foo\n(B) bar"
    with pytest.raises(BBHGroundTruthMismatch):
        _normalize_bbh_example("hyperbaton", 0, inp, "totally, unrelated, text")


def test_normalize_single_option_choice_raises_mismatch_not_a_fake_choice():
    """Regression test for snarks[88] in the real raw corpus: input truncated
    to only "(A) The NB" with no (B). A 1-candidate 'choice' record is the
    exact single-candidate cheat this rewrite removes -- must not slip through
    just because ground_truth trivially equals its lone candidate."""
    inp = "Which statement is sarcastic?\nOptions:\n(A) The NB"
    with pytest.raises(BBHGroundTruthMismatch):
        _normalize_bbh_example("snarks", 88, inp, "(A)")


def test_convert_bbh_task_skips_ground_truth_mismatch_and_logs(tmp_path, capsys):
    good = {"input": "Which is correct?\nOptions:\n(A) foo\n(B) bar", "target": "(A)"}
    bad = {"input": "Which is correct?\nOptions:\n(A) foo\n(B) bar", "target": "not an option at all"}
    # 1 bad in 100 keeps the skip rate under _BBH_MAX_SKIP_RATE so this test
    # exercises skip-and-continue rather than the abort-on-high-skip-rate path.
    raw = {"examples": [good] * 99 + [bad]}
    task_file = tmp_path / "hyperbaton.json"
    task_file.write_text(json.dumps(raw), encoding="utf-8")
    out_path = tmp_path / "out.jsonl"

    written, skipped = convert_bbh_task(task_file, out_path)

    assert (written, skipped) == (99, 1)
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    assert all(r["ground_truth"] == "A" for r in rows)
    assert "WARNING" in capsys.readouterr().err


def test_convert_bbh_task_aborts_when_skip_rate_looks_like_a_parser_bug(tmp_path):
    """A handful of upstream-noise skips is tolerated; a systemic mismatch rate
    must raise rather than silently keep thinning the dataset."""
    bad = {"input": "Which is correct?\nOptions:\n(A) foo\n(B) bar", "target": "garbage"}
    good = {"input": "Which is correct?\nOptions:\n(A) foo\n(B) bar", "target": "(A)"}
    raw = {"examples": [bad] * 5 + [good] * 5}
    task_file = tmp_path / "hyperbaton.json"
    task_file.write_text(json.dumps(raw), encoding="utf-8")
    out_path = tmp_path / "out.jsonl"

    with pytest.raises(BBHFormatError, match="skip rate"):
        convert_bbh_task(task_file, out_path)


def test_convert_bbh_task_raises_on_duplicate_candidates(tmp_path):
    raw = {"examples": [{"input": "Which?\nOptions:\n- same\n- same", "target": "same"}]}
    task_file = tmp_path / "some_dashed_task.json"
    task_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BBHFormatError):
        convert_bbh_task(task_file, tmp_path / "out.jsonl")


def test_convert_bbh_task_respects_limit(tmp_path):
    raw = {
        "examples": [
            {"input": "Which is correct?\nOptions:\n(A) foo\n(B) bar", "target": "(A)"}
            for _ in range(10)
        ]
    }
    task_file = tmp_path / "hyperbaton.json"
    task_file.write_text(json.dumps(raw), encoding="utf-8")
    out_path = tmp_path / "out.jsonl"

    written, skipped = convert_bbh_task(task_file, out_path, limit=3)

    assert (written, skipped) == (3, 0)
    assert len(out_path.read_text().splitlines()) == 3
