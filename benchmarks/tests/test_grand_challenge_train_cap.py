"""Train-row cap (GC_N_TRAIN / GC_N_TRAIN_TASKS), pqa_artificial fill, and the
guards that keep a larger cap from silently mixing with 1000-row artifacts.

The data tests read the real public train partitions (MASSIVE tarball, BoolQ and
PubMedQA from the local HF cache) and the real 01.PNG test files. They skip only
when that data is absent on the machine.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

SUITES = Path(__file__).resolve().parents[1] / "suites"
sys.path.insert(0, str(SUITES))

import grand_challenge_data as gd  # noqa: E402


def have_massive() -> bool:
    return (gd.RAW / "amazon-massive-dataset-1.1.tar.gz").exists()


def have_hf(config_dir: str) -> bool:
    root = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
    return (root / config_dir).exists()


# ------------------------------------------------------------- n_train_for

def test_default_cap_is_1000_so_old_caches_stay_valid():
    assert gd.n_train_for("boolq", env={}) == 1000


def test_global_and_per_task_caps():
    env = {"GC_N_TRAIN": "1500", "GC_N_TRAIN_TASKS": "massive_de=3000, boolq=2500"}
    assert gd.n_train_for("massive_de", env) == 3000
    assert gd.n_train_for("boolq", env) == 2500
    assert gd.n_train_for("pubmedqa", env) == 1500


@pytest.mark.parametrize("env", [
    {"GC_N_TRAIN": "0"}, {"GC_N_TRAIN": "-5"}, {"GC_N_TRAIN": "2k"},
    {"GC_N_TRAIN_TASKS": "boolq"}, {"GC_N_TRAIN_TASKS": "boolqq=2000"}, {"GC_N_TRAIN_TASKS": "boolq=abc"},
])
def test_bad_cap_fails_loudly(env):
    with pytest.raises(ValueError):
        gd.n_train_for("boolq", env)


def test_zero_is_not_a_max_alias():
    """0 stays a rejected config, not a synonym for unlimited: overloading it would
    turn a typo'd GC_N_TRAIN=0 into a silent multi-hour full-pool run."""
    with pytest.raises(ValueError, match="positive integer"):
        gd.n_train_for("boolq", {"GC_N_TRAIN": "0"})


def test_max_means_unlimited():
    assert gd.n_train_for("boolq", {"GC_N_TRAIN": "max"}) is None
    assert gd.n_train_for("boolq", {"GC_N_TRAIN": "MAX"}) is None
    assert gd.n_train_for("boolq", {"GC_N_TRAIN": " Max "}) is None


def test_per_task_max_overrides_global_number():
    env = {"GC_N_TRAIN": "500", "GC_N_TRAIN_TASKS": "boolq=max"}
    assert gd.n_train_for("boolq", env) is None
    assert gd.n_train_for("massive_de", env) == 500


def test_global_max_with_per_task_number_override():
    env = {"GC_N_TRAIN": "max", "GC_N_TRAIN_TASKS": "boolq=500"}
    assert gd.n_train_for("boolq", env) == 500
    assert gd.n_train_for("massive_de", env) is None


def test_bad_pubmedqa_extra_fails_loudly(monkeypatch):
    monkeypatch.setenv("GC_PUBMEDQA_EXTRA", "unlabeled")
    with pytest.raises(ValueError):
        gd.pubmedqa_extra()


# ------------------------------------------------------------ build_train

def _check_superset_and_gate(task: str, small: int, big: int):
    test = gd.load_test(task)
    rows_s, gate_s = gd.build_train(task, test, n_max=small)
    rows_b, gate_b = gd.build_train(task, test, n_max=big)
    # One fixed permutation: the larger set starts with the smaller one, row for row.
    assert [r["id"] for r in rows_b[:small]] == [r["id"] for r in rows_s]
    assert gate_b["kept"] == len(rows_b) == big and not gate_b["pool_exhausted"]
    assert gate_b["id_overlap"] == gate_b["text_overlap"] == gate_b["family_overlap"] == 0
    assert len({r["id"] for r in rows_b}) == big
    return rows_b, gate_b


@pytest.mark.skipif(not have_massive(), reason="MASSIVE tarball not on this machine")
def test_massive_de_3000_rows_superset_of_1000_and_leak_free():
    rows, gate = _check_superset_and_gate("massive_de", 1000, 3000)
    labels = Counter(r["ground_truth"] for r in rows)
    test_labels = set(gd.load_test("massive_de")[0]["candidates"])
    assert set(labels) <= test_labels


@pytest.mark.skipif(not have_hf("google___boolq"), reason="BoolQ not in the HF cache")
def test_boolq_3000_rows_superset_of_1000_and_leak_free():
    rows, _ = _check_superset_and_gate("boolq", 1000, 3000)
    test_passages = {gd.sha16(r["fields"]["passage"]) for r in gd.load_test("boolq")}
    assert not test_passages & {r["family"] for r in rows}


# ------------------------------------------------- GC_N_TRAIN=max (unlimited)

def _check_unlimited_is_leak_free_superset(task: str, small: int, expected_kept: int):
    test = gd.load_test(task)
    rows_s, _ = gd.build_train(task, test, n_max=small)
    rows_u, gate_u = gd.build_train(task, test, n_max=None)
    assert gate_u["pool_exhausted"] is True
    assert gate_u["kept"] == len(rows_u) == expected_kept
    assert gate_u["id_overlap"] == gate_u["text_overlap"] == gate_u["family_overlap"] == 0
    # One fixed permutation: unlimited starts with the same prefix as any smaller cap.
    assert [r["id"] for r in rows_u[:small]] == [r["id"] for r in rows_s]
    assert len({r["id"] for r in rows_u}) == expected_kept          # no intra-train duplicate ids
    return rows_u, gate_u


@pytest.mark.skipif(not have_massive(), reason="MASSIVE tarball not on this machine")
def test_massive_de_unlimited_is_full_leak_free_pool():
    # Measured against the real MASSIVE de-DE train split + full_13 test file on
    # 2026-09-24 (see /tmp/verify_max_cap_report.json in the task report).
    _check_unlimited_is_leak_free_superset("massive_de", 1000, expected_kept=11247)


@pytest.mark.skipif(not have_hf("google___boolq"), reason="BoolQ not in the HF cache")
def test_boolq_unlimited_is_full_leak_free_pool():
    _check_unlimited_is_leak_free_superset("boolq", 1000, expected_kept=9264)


@pytest.mark.skipif(not have_hf("qiaojin___pub_med_qa/pqa_artificial"), reason="pqa_artificial not in the HF cache")
def test_pubmedqa_unlimited_artificial_is_full_leak_free_pool(monkeypatch):
    monkeypatch.setenv("GC_PUBMEDQA_EXTRA", "artificial")
    test = gd.load_test("pubmedqa")
    rows, gate = gd.build_train("pubmedqa", test, n_max=None)
    assert gate["pool_exhausted"] is True
    assert gate["kept"] == len(rows) == 211963
    assert gate["id_overlap"] == gate["text_overlap"] == gate["family_overlap"] == 0
    assert {r["ground_truth"] for r in rows[750:]} <= {"yes", "no"}  # pqa_artificial has no "maybe"


@pytest.mark.skipif(not have_hf("qiaojin___pub_med_qa/pqa_labeled"), reason="PubMedQA not in the HF cache")
def test_pubmedqa_labeled_pool_is_750_and_a_bigger_cap_cannot_grow_it(monkeypatch):
    monkeypatch.delenv("GC_PUBMEDQA_EXTRA", raising=False)
    rows, gate = gd.build_train("pubmedqa", gd.load_test("pubmedqa"), n_max=3000)
    assert gate["kept"] == len(rows) == 750
    assert gate["pool_exhausted"]
    assert gate["family_overlap"] == 0


@pytest.mark.skipif(not have_hf("qiaojin___pub_med_qa/pqa_artificial"), reason="pqa_artificial not in the HF cache")
def test_pubmedqa_artificial_fills_after_all_labeled_rows(monkeypatch):
    test = gd.load_test("pubmedqa")
    monkeypatch.delenv("GC_PUBMEDQA_EXTRA", raising=False)
    labeled, _ = gd.build_train("pubmedqa", test, n_max=3000)
    monkeypatch.setenv("GC_PUBMEDQA_EXTRA", "artificial")
    rows, gate = gd.build_train("pubmedqa", test, n_max=3000)
    assert [r["id"] for r in rows[:750]] == [r["id"] for r in labeled]
    extra = rows[750:]
    assert len(extra) == 2250 and all(r["id"].startswith("pubmedqa-art-") for r in extra)
    assert {r["ground_truth"] for r in extra} <= {"yes", "no"}      # pqa_artificial has no "maybe"
    test_fams = {gd.test_family("pubmedqa", r) for r in test}
    assert not test_fams & {r["family"] for r in rows}
    assert gate["id_overlap"] == gate["text_overlap"] == gate["family_overlap"] == 0


# ------------------------------------------- guards against mixed row counts

def test_encode_cache_refuses_features_built_with_another_cap(tmp_path, monkeypatch):
    import benchmark_01png_grand_challenge as gc
    feat = tmp_path / "features"
    feat.mkdir()
    # A pre-change cache: no n_train_max key, so it was built with 1000 rows.
    np.savez(feat / "massive_de.npz", info_json=np.array(json.dumps({"encoder": gc.ENCODER})))
    monkeypatch.setattr(gc, "FEAT_DIR", feat)
    monkeypatch.setattr(gc, "make_encoder", lambda threads: None)
    monkeypatch.delenv("GC_N_TRAIN", raising=False)
    monkeypatch.delenv("GC_N_TRAIN_TASKS", raising=False)
    gc.stage_encode(["massive_de"], 1)                      # same cap: reused
    monkeypatch.setenv("GC_N_TRAIN_TASKS", "massive_de=3000")
    with pytest.raises(SystemExit, match="n_train_max"):
        gc.stage_encode(["massive_de"], 1)


def test_cached_train_spec_distinguishes_legacy_int_and_unlimited(tmp_path):
    legacy = tmp_path / "legacy.npz"      # pre-GC_N_TRAIN cache: no n_train_max key at all
    np.savez(legacy, info_json=np.array(json.dumps({"encoder": "x"})))
    assert gd.cached_train_spec(legacy) == (1000, "")

    explicit = tmp_path / "explicit.npz"
    np.savez(explicit, info_json=np.array(json.dumps({"n_train_max": 3000, "pubmedqa_extra": ""})))
    assert gd.cached_train_spec(explicit) == (3000, "")

    unlimited = tmp_path / "unlimited.npz"
    np.savez(unlimited, info_json=np.array(json.dumps({"n_train_max": None, "pubmedqa_extra": "artificial"})))
    assert gd.cached_train_spec(unlimited) == (None, "artificial")


def test_encode_cache_accepts_max_after_max_but_rejects_shrinking_back(tmp_path, monkeypatch):
    import benchmark_01png_grand_challenge as gc
    feat = tmp_path / "features"
    feat.mkdir()
    monkeypatch.setattr(gc, "FEAT_DIR", feat)
    monkeypatch.setattr(gc, "make_encoder", lambda threads: None)
    monkeypatch.delenv("GC_N_TRAIN", raising=False)
    monkeypatch.setenv("GC_N_TRAIN_TASKS", "massive_de=max")
    np.savez(feat / "massive_de.npz",
             info_json=np.array(json.dumps({"encoder": gc.ENCODER, "n_train_max": None, "pubmedqa_extra": ""})))
    gc.stage_encode(["massive_de"], 1)                       # same (unlimited) cap: reused
    monkeypatch.setenv("GC_N_TRAIN_TASKS", "massive_de=1000")
    with pytest.raises(SystemExit, match="n_train_max"):
        gc.stage_encode(["massive_de"], 1)                   # asking for less than the max cache: refuse


def test_reused_head_must_match_feature_rows(tmp_path):
    import benchmark_sota_ensemble as se
    head = tmp_path / "boolq_baseline.npz"
    head.write_bytes(b"")
    assert se.head_rows_match(head, 1000) is False            # no train log: reported, not assumed
    (tmp_path / "boolq_baseline_train.json").write_text(json.dumps({"n_train": 850, "n_early_stop": 150}))
    assert se.head_rows_match(head, 1000) is True
    with pytest.raises(ValueError, match="1000 rows"):
        se.head_rows_match(head, 3000)


def test_rnn_heads_override_rejects_two_rnn_sources(monkeypatch, tmp_path):
    import benchmark_sota_ensemble as se
    argv = ["benchmark_sota_ensemble.py", "combine", "--tasks", "boolq",
            "--source", f"q05={tmp_path}", "--source", f"qwen27={tmp_path}",
            "--rnn-source", "q05,qwen27", "--rnn-heads", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="exactly one"):
        se.main()


def test_two_rnn_sources_each_use_their_own_heads(monkeypatch, tmp_path):
    import benchmark_sota_ensemble as se
    a, b = tmp_path / "a", tmp_path / "b"
    argv = ["benchmark_sota_ensemble.py", "combine", "--tasks", "boolq",
            "--source", f"q05={a}", "--source", f"qwen27={b}", "--rnn-source", "q05,qwen27"]
    monkeypatch.setattr(sys, "argv", argv)
    # main() rebinds module globals; monkeypatch restores them after the test.
    monkeypatch.setattr(se, "SOURCES", {})
    monkeypatch.setattr(se, "RNN_SOURCES", [])
    monkeypatch.setattr(se, "RNN_HEADS", {})
    for name in ("OUT_ART", "RESULTS", "PRIOR_REPORT"):
        monkeypatch.setattr(se, name, getattr(se, name))
    monkeypatch.setattr(se.gd, "TEST_DIR", se.gd.TEST_DIR)
    monkeypatch.setattr(se, "stage_combine", lambda *args, **kwargs: None)
    se.main()
    assert se.RNN_SOURCES == ["q05", "qwen27"]
    assert se.RNN_HEADS == {"q05": a / "heads", "qwen27": b / "heads"}
