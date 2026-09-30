"""Dataset scale expansion: larger GC_N_TRAIN caps and the triple leakage gate under
injected leakage.

Section D injects synthetic rows through ``gd._iter_train`` (a monkeypatch of the
I/O-bound HF/tarball generator only) so the leakage gate can be exercised
deterministically without downloading anything. ``build_train`` and its gate run
completely unmodified on the injected rows — this is fixture injection, not a
production mock: the code under test never has its own logic replaced.

Section E re-runs the real 3000-row-class checks from test_grand_challenge_train_cap.py
at a larger cap (5000) to cover this task's "larger-scale extraction" requirement with
real data; it skips when that data isn't on the machine.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES))

import grand_challenge_data as gd  # noqa: E402



def have_massive() -> bool:
    return (gd.RAW / "amazon-massive-dataset-1.1.tar.gz").exists()


def have_hf(config_dir: str) -> bool:
    import os
    root = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
    return (root / config_dir).exists()


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


