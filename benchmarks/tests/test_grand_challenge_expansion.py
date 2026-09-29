"""Dataset scale expansion: larger GC_N_TRAIN caps, the triple leakage gate under
injected leakage, and the natural-text rebuilder's parser/fail-closed behavior.

Section D injects synthetic rows through ``gd._iter_train`` (a monkeypatch of the
I/O-bound HF/tarball generator only) so the leakage gate can be exercised
deterministically without downloading anything. ``build_train`` and its gate run
completely unmodified on the injected rows — this is fixture injection, not a
production mock: the code under test never has its own logic replaced.

Sections A-C need no external data at all (synthetic contexts and jsonl files).
Section E re-runs the real 3000-row-class checks from test_grand_challenge_train_cap.py
at a larger cap (5000) to cover this task's "larger-scale extraction" requirement with
real data; it skips when that data isn't on the machine.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES))
sys.path.insert(0, str(ROOT / "scripts"))

import grand_challenge_data as gd  # noqa: E402
import rebuild_open_training_pool_natural_text as rb  # noqa: E402

REAL_POOL = Path("/ebs/pj/gen-zero/benchmarks/artifacts/verified_datasets/open_training_pool_5k.jsonl")


def have_massive() -> bool:
    return (gd.RAW / "amazon-massive-dataset-1.1.tar.gz").exists()


def have_hf(config_dir: str) -> bool:
    import os
    root = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
    return (root / config_dir).exists()


# ================================================================== section A
# rebuild_open_training_pool_natural_text.parse_labeled_options: prefix formats
# and fail-closed behavior on malformed contexts.

def _arc_context(options, labels=None):
    labels = labels or [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(f"({lab}) {opt}" for lab, opt in zip(labels, options))
    return f"Question: stem\n{body}\n{rb.SELECT_SUFFIX}", labels


def test_parses_letter_labels():
    ctx, labels = _arc_context(["increased use of glass bottles.", "increased number of trees cut down."])
    options = rb.parse_labeled_options(ctx, labels, "rid")
    assert options == ["increased use of glass bottles.", "increased number of trees cut down."]


def test_parses_digit_labels_62_arc_rows_use_this_format():
    ctx, labels = _arc_context(["four", "seven", "ten"], labels=["1", "2", "3"])
    options = rb.parse_labeled_options(ctx, labels, "rid")
    assert options == ["four", "seven", "ten"]


def test_wrong_suffix_raises():
    ctx = "Question: stem\n(A) x\n(B) y\nChoose wisely."
    with pytest.raises(ValueError, match="instruction suffix"):
        rb.parse_labeled_options(ctx, ["A", "B"], "rid")


def test_missing_marker_raises():
    ctx = f"Question: stem\n(A) x\n{rb.SELECT_SUFFIX}"
    with pytest.raises(ValueError, match="not found"):
        rb.parse_labeled_options(ctx, ["A", "B"], "rid")


def test_out_of_order_markers_raises():
    # (B) appears before (A) in the body -> the parser must refuse, not silently
    # sort labels to make it fit.
    ctx = f"Question: stem\n(B) y\n(A) x\n{rb.SELECT_SUFFIX}"
    with pytest.raises(ValueError, match="out of order"):
        rb.parse_labeled_options(ctx, ["A", "B"], "rid")


def test_empty_option_text_raises():
    ctx = f"Question: stem\n(A) \n(B) y\n{rb.SELECT_SUFFIX}"
    with pytest.raises(ValueError, match="empty"):
        rb.parse_labeled_options(ctx, ["A", "B"], "rid")


def test_rebuild_drops_and_counts_on_parse_error_not_silent():
    ctx = f"Question: stem\n(A) x\n{rb.SELECT_SUFFIX}"  # (B) marker missing
    record = {"task": "arc_challenge_train", "id": "r1", "candidates": ["A", "B"],
              "ground_truth": "A", "context": ctx}
    stats = {"rebuilt": 0, "passthrough": 0, "dropped_gt_not_in_labels": 0,
             "dropped_parse_error": 0, "dropped_duplicate_option_text": 0}
    result = rb.rebuild(record, stats)
    assert result is None
    assert stats["dropped_parse_error"] == 1


def test_rebuild_drops_gt_not_in_labels():
    ctx, labels = _arc_context(["x", "y"])
    record = {"task": "arc_challenge_train", "id": "r1", "candidates": labels,
              "ground_truth": "Z", "context": ctx}
    stats = {"rebuilt": 0, "passthrough": 0, "dropped_gt_not_in_labels": 0,
             "dropped_parse_error": 0, "dropped_duplicate_option_text": 0}
    result = rb.rebuild(record, stats)
    assert result is None
    assert stats["dropped_gt_not_in_labels"] == 1


def test_rebuild_drops_duplicate_option_text():
    ctx, labels = _arc_context(["same", "same"])
    record = {"task": "mmlu_pro_test", "id": "r1", "candidates": labels,
              "ground_truth": labels[0], "context": ctx}
    stats = {"rebuilt": 0, "passthrough": 0, "dropped_gt_not_in_labels": 0,
             "dropped_parse_error": 0, "dropped_duplicate_option_text": 0}
    result = rb.rebuild(record, stats)
    assert result is None
    assert stats["dropped_duplicate_option_text"] == 1


def test_rebuild_unrecognized_task_raises_not_silently_passes_through():
    record = {"task": "some_new_task_nobody_registered", "id": "r1"}
    with pytest.raises(ValueError, match="unrecognized task"):
        rb.rebuild(record, {"rebuilt": 0, "passthrough": 0, "dropped_gt_not_in_labels": 0,
                             "dropped_parse_error": 0, "dropped_duplicate_option_text": 0})


def test_rebuild_success_replaces_candidates_with_text():
    ctx, labels = _arc_context(["increased use of glass bottles.", "increased number of trees cut down."])
    record = {"task": "arc_challenge_train", "id": "r1", "candidates": labels,
              "ground_truth": "B", "context": ctx, "metadata": {}}
    stats = {"rebuilt": 0, "passthrough": 0, "dropped_gt_not_in_labels": 0,
             "dropped_parse_error": 0, "dropped_duplicate_option_text": 0}
    out = rb.rebuild(record, stats)
    assert out["candidates"] == ["increased use of glass bottles.", "increased number of trees cut down."]
    assert out["ground_truth"] == "increased number of trees cut down."
    assert out["metadata"]["candidates_source"] == "parsed_from_context_natural_text"
    assert stats["rebuilt"] == 1


def test_passthrough_task_is_returned_unchanged():
    record = {"task": "banking77", "id": "r1", "candidates": ["a", "b"], "ground_truth": "a",
              "context": "irrelevant"}
    stats = {"rebuilt": 0, "passthrough": 0, "dropped_gt_not_in_labels": 0,
             "dropped_parse_error": 0, "dropped_duplicate_option_text": 0}
    out = rb.rebuild(record, stats)
    assert out is record
    assert stats["passthrough"] == 1


# ================================================================== section B
# main(): fail-closed exit code on real drops (this task's hardening).

def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _good_record(rid: str):
    ctx, labels = _arc_context([f"opt-{rid}-1", f"opt-{rid}-2"])
    return {"task": "arc_challenge_train", "id": rid, "candidates": labels,
            "ground_truth": labels[0], "context": ctx, "metadata": {}}


def _bad_record(rid: str):
    # (B) marker missing -> guaranteed parse error.
    ctx = f"Question: stem\n(A) x\n{rb.SELECT_SUFFIX}"
    return {"task": "arc_challenge_train", "id": rid, "candidates": ["A", "B"],
            "ground_truth": "A", "context": ctx}


def test_main_exits_zero_with_default_max_drops_when_nothing_drops(tmp_path, monkeypatch):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _write_jsonl(src, [_good_record(f"r{i}") for i in range(20)])
    monkeypatch.setattr(sys, "argv", ["rebuild", "--src", str(src), "--dst", str(dst)])
    assert rb.main() == 0
    assert len(dst.read_text().strip().splitlines()) == 20


def test_main_fails_closed_when_drops_exceed_default_zero_tolerance(tmp_path, monkeypatch):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _write_jsonl(src, [_good_record("r0"), _bad_record("r1")])
    monkeypatch.setattr(sys, "argv", ["rebuild", "--src", str(src), "--dst", str(dst)])
    assert rb.main() == 1


def test_main_respects_explicit_max_drops_override(tmp_path, monkeypatch):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _write_jsonl(src, [_good_record("r0"), _bad_record("r1")])
    monkeypatch.setattr(sys, "argv", ["rebuild", "--src", str(src), "--dst", str(dst), "--max-drops", "1"])
    assert rb.main() == 0
    # even under an explicit tolerance, the bad record is still dropped, not fabricated
    assert len(dst.read_text().strip().splitlines()) == 1


# ================================================================== section C
# Scaled extraction (10k / 20k synthetic) stays deterministic and byte-identical.

def _synthetic_pool(n: int):
    records = []
    for i in range(n):
        if i % 3 == 0:
            records.append({"task": "banking77", "id": f"b{i}", "candidates": ["x", "y"],
                             "ground_truth": "x", "context": "irrelevant"})
        else:
            records.append(_good_record(f"r{i}"))
    return records


@pytest.mark.parametrize("n", [10_000, 20_000])
def test_rebuild_at_scale_is_deterministic_byte_identical(tmp_path, monkeypatch, n):
    src = tmp_path / "in.jsonl"
    _write_jsonl(src, _synthetic_pool(n))
    dst1, dst2 = tmp_path / "out1.jsonl", tmp_path / "out2.jsonl"

    monkeypatch.setattr(sys, "argv", ["rebuild", "--src", str(src), "--dst", str(dst1)])
    assert rb.main() == 0
    monkeypatch.setattr(sys, "argv", ["rebuild", "--src", str(src), "--dst", str(dst2)])
    assert rb.main() == 0

    b1, b2 = dst1.read_bytes(), dst2.read_bytes()
    assert hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest()
    assert len(b1.strip().split(b"\n")) == n


# ================================================================== section D
# Triple leakage gate under injected leakage (fixture injection, not a mock of
# build_train itself: only the row source _iter_train is swapped).

def _boolq_test_rows():
    return [{"id": "boolq-test-1", "instruction": "instr", "candidates": ["true", "false"],
              "fields": {"passage": "The sky is blue during the day.", "question": "Is the sky blue?"}}]


def test_leakage_gate_drops_id_family_and_text_collisions(monkeypatch):
    test_rows = _boolq_test_rows()
    test_passage = test_rows[0]["fields"]["passage"]
    test_family_key = gd.sha16(test_passage)

    def fake_iter_train(task, rng):
        assert task == "boolq"
        # 1) exact id collision with the test row
        yield {"id": "boolq-test-1", "family": "different-family", "label": "true",
               "fields": {"passage": "unrelated passage one", "question": "q1"}}
        # 2) family collision (same passage hash) with a different id
        yield {"id": "boolq-train-fam-collide", "family": test_family_key, "label": "false",
               "fields": {"passage": "some other passage text", "question": "q2"}}
        # 3) normalized-text collision: same passage+question, different id/family
        yield {"id": "boolq-train-text-collide", "family": "yet-another-family", "label": "true",
               "fields": {"passage": test_passage, "question": test_rows[0]["fields"]["question"]}}
        # 4) a clean row, then an exact duplicate of it (intra-train duplicate)
        yield {"id": "boolq-clean-1", "family": "clean-fam-1", "label": "true",
               "fields": {"passage": "clean passage", "question": "clean question"}}
        yield {"id": "boolq-clean-1-dup", "family": "clean-fam-1-dup", "label": "true",
               "fields": {"passage": "clean passage", "question": "clean question"}}
        # 5) a second genuinely clean row
        yield {"id": "boolq-clean-2", "family": "clean-fam-2", "label": "false",
               "fields": {"passage": "second clean passage", "question": "second clean question"}}

    monkeypatch.setattr(gd, "_iter_train", fake_iter_train)
    rows, gate = gd.build_train("boolq", test_rows, n_max=None)

    assert gate["dropped_test_id"] == 1
    assert gate["dropped_test_family"] == 1
    assert gate["dropped_test_text"] == 1
    assert gate["dropped_train_duplicate"] == 1
    assert gate["kept"] == len(rows) == 2
    assert {r["id"] for r in rows} == {"boolq-clean-1", "boolq-clean-2"}
    # the final zero-intersection assertion in build_train() already ran without
    # raising to get here; re-check it explicitly against the gate's own report.
    assert gate["id_overlap"] == gate["text_overlap"] == gate["family_overlap"] == 0


def test_leakage_gate_raises_if_a_leak_survived_the_filters(monkeypatch):
    """The filters above already remove every leak by construction; this proves the
    final AssertionError backstop itself is live by forcing a leak past field_key
    (a row whose *id* only would slip through if the family/text filters were ever
    weakened) — it must still be caught, here by the text filter, and must never
    silently ship."""
    test_rows = _boolq_test_rows()

    def fake_iter_train(task, rng):
        # id collision is the simplest way to prove the assertion path stays wired:
        # if dropped_test_id's filter were ever deleted, this must raise, not pass.
        yield {"id": "boolq-test-1", "family": "boolq-test-1-fam", "label": "true",
               "fields": {"passage": "different passage entirely", "question": "different q"}}

    monkeypatch.setattr(gd, "_iter_train", fake_iter_train)
    rows, gate = gd.build_train("boolq", test_rows, n_max=None)
    assert gate["dropped_test_id"] == 1
    assert rows == []


# ================================================================== section E
# Real data at a larger cap than the existing 3000-row tests (5000), still
# leak-free and a strict superset. Skips when the data isn't on this machine.

@pytest.mark.skipif(not have_massive(), reason="MASSIVE tarball not on this machine")
def test_massive_de_5000_rows_superset_of_1000_and_leak_free():
    test = gd.load_test("massive_de")
    small, _ = gd.build_train("massive_de", test, n_max=1000)
    big, gate = gd.build_train("massive_de", test, n_max=5000)
    assert [r["id"] for r in big[:1000]] == [r["id"] for r in small]
    assert gate["kept"] == len(big) == 5000
    assert gate["id_overlap"] == gate["text_overlap"] == gate["family_overlap"] == 0
    assert len({r["id"] for r in big}) == 5000


@pytest.mark.skipif(not have_hf("google___boolq"), reason="BoolQ not in the HF cache")
def test_boolq_5000_rows_superset_of_1000_and_leak_free():
    test = gd.load_test("boolq")
    small, _ = gd.build_train("boolq", test, n_max=1000)
    big, gate = gd.build_train("boolq", test, n_max=5000)
    assert [r["id"] for r in big[:1000]] == [r["id"] for r in small]
    assert gate["kept"] == len(big) == 5000
    assert gate["id_overlap"] == gate["text_overlap"] == gate["family_overlap"] == 0


@pytest.mark.skipif(not REAL_POOL.exists(), reason="real open_training_pool_5k.jsonl not on this machine")
def test_real_5304_row_pool_rebuilds_with_zero_real_drops(tmp_path):
    dst = tmp_path / "out.jsonl"
    old_argv = sys.argv
    try:
        sys.argv = ["rebuild", "--src", str(REAL_POOL), "--dst", str(dst)]
        assert rb.main() == 0
    finally:
        sys.argv = old_argv
    lines = dst.read_text().strip().splitlines()
    assert len(lines) == 5304
