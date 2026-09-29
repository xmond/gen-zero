"""HANS and Bespoke Labs ingestion: six-key schema, candidate integrity, stratified sampling,
fail-closed parsing and no private paths in the output.

Synthetic fixtures cover the parser branches. The real-data tests run only when
GEN_ZERO_BENCHMARK_RAW_DIR points at a raw root that holds the sources; otherwise they skip
with an explicit reason.
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "datasets"))

import convert_new_benchmarks as cnb  # noqa: E402

KEYS = {"id", "task", "context", "candidates", "ground_truth", "metadata"}
HEURISTICS = ("lexical_overlap", "subsequence", "constituent")
# Absolute POSIX homes/mounts, Windows drive paths, file URLs.
PRIVATE_PATH_RE = re.compile(r"(/ebs/|/home/|/Users/|/tmp/|/mnt/|[A-Za-z]:\\\\|file://)")
HANS_HEADER = ["gold_label", "sentence1_binary_parse", "sentence2_binary_parse", "sentence1_parse",
               "sentence2_parse", "sentence1", "sentence2", "pairID", "heuristic", "subcase", "template"]


def _hans_tsv(path: Path, per_stratum: int = 4, extra_rows=()) -> Path:
    lines = ["\t".join(HANS_HEADER)]
    n = 0
    for h in HEURISTICS:
        for gold in ("entailment", "non-entailment"):
            for _ in range(per_stratum):
                lines.append("\t".join([gold, "( a )", "( b )", "(ROOT a)", "(ROOT b)",
                                        f"The doctor saw the actor {n} .", f"The actor saw the doctor {n} .",
                                        f"ex{n}", h, f"{h}_case", "temp1"]))
                n += 1
    lines.extend(extra_rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x]


def _assert_six_key(rows):
    for r in rows:
        assert set(r) == KEYS, r.keys()
        assert isinstance(r["id"], str) and r["id"]
        assert isinstance(r["task"], str) and r["task"]
        assert isinstance(r["context"], str) and r["context"].strip()
        assert isinstance(r["candidates"], list) and len(r["candidates"]) >= 2
        assert all(isinstance(c, str) for c in r["candidates"])
        assert len(set(r["candidates"])) == len(r["candidates"])
        assert isinstance(r["ground_truth"], str) and r["ground_truth"] in r["candidates"]
        assert isinstance(r["metadata"], dict)
    assert len({r["id"] for r in rows}) == len(rows)


# ---------------------------------------------------------------- HANS


def test_hans_full_mode_schema_candidates_and_metadata(tmp_path):
    src = _hans_tsv(tmp_path / "hans.txt")
    out = tmp_path / "hans.jsonl"
    assert cnb.convert_hans(src, out, limit=None) == 24
    rows = _read(out)
    _assert_six_key(rows)
    for r in rows:
        assert r["task"] == "hans"
        assert r["candidates"] == ["entailment", "non-entailment"]
        assert set(r["metadata"]) == {"heuristic", "subcase", "template", "source"}
        assert r["metadata"]["source"] == "hans"
        assert r["metadata"]["heuristic"] in HEURISTICS
        assert r["metadata"]["subcase"] == f"{r['metadata']['heuristic']}_case"
        assert r["metadata"]["template"] == "temp1"
        assert r["context"].startswith("Premise: ") and "\nHypothesis: " in r["context"]
    assert rows[0]["id"] == "hans-ex0"


def test_hans_stratified_sample_is_equal_per_heuristic_and_label(tmp_path):
    src = _hans_tsv(tmp_path / "hans.txt", per_stratum=10)
    out = tmp_path / "hans.jsonl"
    assert cnb.convert_hans(src, out, limit=12, stratify=True, seed=3) == 12
    rows = _read(out)
    _assert_six_key(rows)
    assert Counter(r["metadata"]["heuristic"] for r in rows) == {h: 4 for h in HEURISTICS}
    assert Counter((r["metadata"]["heuristic"], r["ground_truth"]) for r in rows) == {
        (h, g): 2 for h in HEURISTICS for g in ("entailment", "non-entailment")}


def test_hans_stratified_sample_is_seeded_and_deterministic(tmp_path):
    src = _hans_tsv(tmp_path / "hans.txt", per_stratum=10)
    ids = []
    for seed in (5, 5, 6):
        out = tmp_path / f"hans_{len(ids)}.jsonl"
        cnb.convert_hans(src, out, limit=18, seed=seed)
        ids.append([r["id"] for r in _read(out)])
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


def test_hans_unstratified_limit_takes_file_prefix(tmp_path):
    src = _hans_tsv(tmp_path / "hans.txt")
    out = tmp_path / "hans.jsonl"
    assert cnb.convert_hans(src, out, limit=5, stratify=False) == 5
    assert [r["id"] for r in _read(out)] == [f"hans-ex{i}" for i in range(5)]


@pytest.mark.parametrize("limit", [7, 100])
def test_hans_stratified_limit_that_cannot_split_evenly_raises(tmp_path, limit):
    src = _hans_tsv(tmp_path / "hans.txt", per_stratum=20)
    with pytest.raises(ValueError, match="multiple of 6"):
        cnb.convert_hans(src, tmp_path / "o.jsonl", limit=limit)


def test_hans_short_stratum_raises_instead_of_undersampling(tmp_path):
    # 2 rows per stratum plus 6 extra in one stratum: 18 rows total, but limit 18 needs 3 in each.
    extra = [f"entailment\ta\tb\tc\td\tP .\tH .\textra{i}\tlexical_overlap\tx\ttemp1" for i in range(6)]
    src = _hans_tsv(tmp_path / "hans.txt", per_stratum=2, extra_rows=extra)
    with pytest.raises(ValueError, match="stratum .* has 2 rows, need 3"):
        cnb.convert_hans(src, tmp_path / "o.jsonl", limit=18)


@pytest.mark.parametrize("bad_row, message", [
    ("contradiction\ta\tb\tc\td\tP .\tH .\tbad1\tlexical_overlap\tx\ttemp1", "gold_label"),
    ("entailment\ta\tb\tc\td\tP .\tH .\tbad2\tnegation\tx\ttemp1", "heuristic"),
    ("entailment\ta\tb\tc\td\t\tH .\tbad3\tlexical_overlap\tx\ttemp1", "empty fields"),
    ("entailment\ta\tb\tc\td\tP .\tH .\tex0\tlexical_overlap\tx\ttemp1", "duplicate pairID"),
    ("entailment\ta\tb\tc\td\tP .\tH .\tbad5\tlexical_overlap\tx\ttemp1\tspill", "more fields"),
])
def test_hans_malformed_row_raises_not_skipped(tmp_path, bad_row, message):
    src = _hans_tsv(tmp_path / "hans.txt", extra_rows=[bad_row])
    with pytest.raises(ValueError, match=message):
        cnb.convert_hans(src, tmp_path / "o.jsonl")


def test_hans_missing_column_raises(tmp_path):
    src = tmp_path / "hans.txt"
    src.write_text("gold_label\tsentence1\tsentence2\npairID\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks columns"):
        cnb.convert_hans(src, tmp_path / "o.jsonl")


# ---------------------------------------------------------------- Bespoke


def _bespoke_row(row_id, dtype, criteria, target, state, domain="commerce", variant="base"):
    return {
        "id": row_id, "domain": domain, "family": "fam-1", "source_family": f"{domain}-01",
        "split": "validation", "variant": variant, "method": "c2d", "quality_status": "checked",
        "provenance": {"source_is_synthetic": True},
        "evidence_certificate": {"fact_states": {"a1": "supported"}, "necessity_checks_passed": True},
        "input": {"state": state, "questions": {"decision": {
            "criteria": criteria, "instructions": "Choose exactly one option.", "type": dtype}}},
        "reference": {"human_reviewed": False, "source": "audited_rule", "target": target},
    }


def _bespoke_file(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def _mixed_bespoke_rows():
    choice = {"ready_publish": "Ready.", "major_fix": "Major defect; route to seller."}
    noul = {"false": "No, not ready.", "true": "Yes, ready."}
    return [
        _bespoke_row("case-001-base", "choice", choice, "major_fix",
                     {"context": "A listing was submitted.",
                      "timeline": ["At 16:10 UTC the packing sheet excluded the charger.",
                                   "At 16:42 UTC the listing claimed a charger."],
                      "request": "Assign severity."}),
        _bespoke_row("case-001-counterfactual", "choice", choice, "ready_publish",
                     {"context": "A listing was submitted.", "timeline": ["Packing sheet includes charger."],
                      "request": "Assign severity."}, variant="counterfactual"),
        _bespoke_row("case-002-base", "noul", noul, True,
                     "All required fields match the latest attachment.", domain="travel"),
        _bespoke_row("case-003-base", "score",
                     ["0 — Complete: nothing missing.", "1 — Minor: optional field missing.",
                      "2 — Major: required field missing."], 1,
                     [{"speaker": "Reviewer", "text": "One optional field is blank."}], domain="media"),
        _bespoke_row("case-004-base", "score",
                     ["No release: claim withdrawn.", "Hold—weak match: generic only.",
                      "Hold—strong match: one check remains."], 2,
                     {"context": "Backpack T-884 was found.",
                      "price_records": {"list": "12.00", "sale": "9.50"}}, domain="public_services"),
    ]


def test_bespoke_causal_decision_schema_and_metadata(tmp_path):
    out = tmp_path / "bespoke.jsonl"
    assert cnb.convert_bespoke(_bespoke_file(tmp_path / "eval.jsonl", _mixed_bespoke_rows()), out) == 5
    rows = {r["id"]: r for r in _read(out)}
    _assert_six_key(list(rows.values()))
    for r in rows.values():
        assert r["task"] == "bespoke"
        assert set(r["metadata"]) == {"source", "policy_domain", "decision_type", "variant", "pair_id",
                                      "family", "source_family", "split", "human_reviewed"}
        assert r["metadata"]["source"] == "bespoke_labs"
        assert "Instructions: Choose exactly one option." in r["context"]
        assert "Candidate Options:\n- " in r["context"]

    base, cf = rows["bespoke-case-001-base"], rows["bespoke-case-001-counterfactual"]
    assert base["candidates"] == cf["candidates"] == ["ready_publish", "major_fix"]
    assert (base["ground_truth"], cf["ground_truth"]) == ("major_fix", "ready_publish")
    assert base["metadata"]["pair_id"] == cf["metadata"]["pair_id"] == "case-001"
    assert (base["metadata"]["variant"], cf["metadata"]["variant"]) == ("base", "counterfactual")
    assert base["metadata"]["policy_domain"] == "commerce"
    assert base["metadata"]["decision_type"] == "choice"

    noul = rows["bespoke-case-002-base"]
    assert noul["candidates"] == ["false", "true"] and noul["ground_truth"] == "true"
    assert noul["context"].startswith("All required fields match")

    numbered = rows["bespoke-case-003-base"]
    assert numbered["candidates"] == ["0", "1", "2"] and numbered["ground_truth"] == "1"
    assert "Reviewer: One optional field is blank." in numbered["context"]
    assert "- 1 — Minor: optional field missing." in numbered["context"]

    named = rows["bespoke-case-004-base"]
    assert named["candidates"] == ["0", "1", "2"] and named["ground_truth"] == "2"
    assert "- 2: Hold—strong match: one check remains." in named["context"]
    assert "Price records:\n  List: 12.00\n  Sale: 9.50" in named["context"]


def test_bespoke_renders_every_state_field(tmp_path):
    """The old whitelist dropped `timeline`; every field must now reach the context."""
    out = tmp_path / "bespoke.jsonl"
    cnb.convert_bespoke(_bespoke_file(tmp_path / "eval.jsonl", _mixed_bespoke_rows()[:1]), out)
    ctx = _read(out)[0]["context"]
    assert "Timeline:\n  1. At 16:10 UTC the packing sheet excluded the charger." in ctx
    assert "  2. At 16:42 UTC the listing claimed a charger." in ctx
    assert "Context: A listing was submitted." in ctx and "Request: Assign severity." in ctx


def test_bespoke_never_leaks_certificate_or_reference(tmp_path):
    out = tmp_path / "bespoke.jsonl"
    cnb.convert_bespoke(_bespoke_file(tmp_path / "eval.jsonl", _mixed_bespoke_rows()), out)
    raw = out.read_text(encoding="utf-8")
    for needle in ("fact_states", "necessity_checks_passed", "evidence_certificate", "audited_rule",
                   "provenance", "source_is_synthetic"):
        assert needle not in raw


def test_bespoke_embedded_questions_copy_is_unwrapped_when_identical(tmp_path):
    row = _mixed_bespoke_rows()[0]
    inner = row["input"]["state"]
    row["input"]["state"] = {"questions": row["input"]["questions"], "state": inner}
    out = tmp_path / "bespoke.jsonl"
    cnb.convert_bespoke(_bespoke_file(tmp_path / "eval.jsonl", [row]), out)
    ctx = _read(out)[0]["context"]
    assert ctx.startswith("Context: A listing was submitted.")
    assert "Questions" not in ctx


def test_bespoke_embedded_questions_that_differ_raise(tmp_path):
    row = _mixed_bespoke_rows()[0]
    row["input"]["state"]["questions"] = {"decision": {"criteria": {"x": 1, "y": 2}}}
    with pytest.raises(ValueError, match="differs"):
        cnb.convert_bespoke(_bespoke_file(tmp_path / "eval.jsonl", [row]), tmp_path / "o.jsonl")


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["input"].__setitem__("state", "   "), "empty context"),
    (lambda r: r["input"].__setitem__("state", {}), "empty context"),
    (lambda r: r["input"].__setitem__("state", 42), "unsupported state type"),
    (lambda r: r["reference"].__setitem__("target", "not_an_option"), "including the target"),
    (lambda r: r["reference"].pop("target"), "missing reference.target"),
    (lambda r: r["reference"].__setitem__("target", True), "boolean target"),
    (lambda r: r["reference"].__setitem__("target", 1), "integer target"),
    (lambda r: r["input"]["questions"]["decision"].__setitem__("instructions", ""), "instructions"),
    (lambda r: r.pop("id"), "missing id"),
])
def test_bespoke_malformed_row_raises_not_skipped(tmp_path, mutate, message):
    row = _mixed_bespoke_rows()[0]
    mutate(row)
    with pytest.raises(ValueError, match=message):
        cnb.convert_bespoke(_bespoke_file(tmp_path / "eval.jsonl", [row]), tmp_path / "o.jsonl")


def test_bespoke_score_label_that_disagrees_with_position_raises(tmp_path):
    row = _mixed_bespoke_rows()[3]
    row["input"]["questions"]["decision"]["criteria"] = ["1 — A.", "0 — B.", "2 — C."]
    with pytest.raises(ValueError, match="labelled"):
        cnb.convert_bespoke(_bespoke_file(tmp_path / "eval.jsonl", [row]), tmp_path / "o.jsonl")


def test_bespoke_duplicate_id_raises(tmp_path):
    rows = _mixed_bespoke_rows()[:1] * 2
    with pytest.raises(ValueError, match="duplicate id"):
        cnb.convert_bespoke(_bespoke_file(tmp_path / "eval.jsonl", rows), tmp_path / "o.jsonl")


def test_bespoke_empty_source_raises(tmp_path):
    src = tmp_path / "eval.jsonl"
    src.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        cnb.convert_bespoke(src, tmp_path / "o.jsonl")


# ---------------------------------------------------------------- CLI and path hygiene


def test_cli_converts_and_never_writes_private_paths(tmp_path, capsys):
    raw = tmp_path / "raw"
    (raw / "hans").mkdir(parents=True)
    (raw / "bespoke").mkdir()
    _hans_tsv(raw / "hans" / "heuristics_evaluation_set.txt", per_stratum=3)
    _bespoke_file(raw / "bespoke" / "eval.jsonl", _mixed_bespoke_rows())
    out = tmp_path / "out"
    rc = cnb.main(["--raw-dir", str(raw), "--out-dir", str(out), "--only", "hans", "--only", "bespoke",
                   "--hans-limit", "12", "--bespoke-limit", "all"])
    assert rc == 0
    captured = capsys.readouterr()
    assert str(tmp_path) not in captured.out + captured.err
    for name, n in (("hans.jsonl", 12), ("bespoke.jsonl", 5)):
        text = (out / name).read_text(encoding="utf-8")
        assert len(_read(out / name)) == n
        assert str(tmp_path) not in text
        assert not PRIVATE_PATH_RE.search(text)


def test_cli_missing_requested_source_fails_loudly(tmp_path, capsys):
    rc = cnb.main(["--raw-dir", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "out"), "--only", "hans"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "[SKIP] missing raw source <raw-dir>/hans/heuristics_evaluation_set.txt" in err
    assert str(tmp_path) not in err


def test_cli_nothing_converted_is_an_error(tmp_path, capsys):
    rc = cnb.main(["--raw-dir", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "out")])
    assert rc == 2
    assert "[ERROR] no source converted" in capsys.readouterr().err


def test_converter_source_has_no_hardcoded_private_paths():
    src = (BENCH / "datasets" / "convert_new_benchmarks.py").read_text(encoding="utf-8")
    assert not PRIVATE_PATH_RE.search(src)


# ---------------------------------------------------------------- Real data (opt-in)

RAW_ENV = os.environ.get("GEN_ZERO_BENCHMARK_RAW_DIR")
REAL_HANS = Path(RAW_ENV) / "hans" / "heuristics_evaluation_set.txt" if RAW_ENV else None
REAL_BESPOKE = Path(RAW_ENV) / "bespoke" / "eval.jsonl" if RAW_ENV else None


@pytest.mark.skipif(not (REAL_HANS and REAL_HANS.exists()),
                    reason="set GEN_ZERO_BENCHMARK_RAW_DIR to a root with hans/heuristics_evaluation_set.txt")
def test_real_hans_full_and_stratified(tmp_path):
    full = tmp_path / "full.jsonl"
    assert cnb.convert_hans(REAL_HANS, full, limit=None) == 30000
    rows = _read(full)
    _assert_six_key(rows)
    assert Counter((r["metadata"]["heuristic"], r["ground_truth"]) for r in rows) == {
        (h, g): 5000 for h in HEURISTICS for g in ("entailment", "non-entailment")}
    assert not PRIVATE_PATH_RE.search(full.read_text(encoding="utf-8"))

    strat = tmp_path / "strat.jsonl"
    assert cnb.convert_hans(REAL_HANS, strat, limit=600, seed=0) == 600
    assert Counter(r["metadata"]["heuristic"] for r in _read(strat)) == {h: 200 for h in HEURISTICS}


@pytest.mark.skipif(not (REAL_BESPOKE and REAL_BESPOKE.exists()),
                    reason="set GEN_ZERO_BENCHMARK_RAW_DIR to a root with bespoke/eval.jsonl")
def test_real_bespoke_full(tmp_path):
    out = tmp_path / "bespoke.jsonl"
    n_source = sum(1 for line in REAL_BESPOKE.read_text(encoding="utf-8").split("\n") if line.strip())
    assert cnb.convert_bespoke(REAL_BESPOKE, out, limit=None) == n_source
    rows = _read(out)
    _assert_six_key(rows)
    text = out.read_text(encoding="utf-8")
    assert not PRIVATE_PATH_RE.search(text)
    assert "fact_states" not in text and "necessity_checks_passed" not in text
    assert all(r["metadata"]["decision_type"] in {"choice", "noul", "score"} for r in rows)
    assert all(len(r["context"]) > 200 for r in rows)
    pairs = Counter(r["metadata"]["pair_id"] for r in rows)
    assert set(pairs.values()) == {2}
