"""Data contract test for the real Qwen2.5-72B 13-task feature extraction.

Unlike test_gpu_extract_qwen72b.py (which exercises the extractor's logic against a fake
llama-server and never touches real weights), this file inspects the actual files copied from
the AI box's run (D:\\genz\\features_uncap_v1\\q72b_last, 2026-09-25) into
data/extracted_features/qwen72b/. It says nothing about the extractor code; it says whether the
bytes that arrived are what a downstream consumer (cross_model_manifold_alignment.py,
sota_ensemble_experts.py) can safely load.

Skipped entirely (not "passed") when the data directory is absent, so this file does not fail
CI on a checkout that never ran the transfer.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "extracted_features" / "qwen72b"
FEATURES_DIR = DATA_DIR / "features"
GTE7B_MASSIVE_EN = ROOT / "data" / "extracted_features" / "gte7b_cpu" / "massive_en.npz"

sys.path.insert(0, str(ROOT / "benchmarks" / "suites"))
import grand_challenge_data as gd  # noqa: E402
from cross_model_manifold_alignment import load_features  # noqa: E402

pytestmark = pytest.mark.skipif(not FEATURES_DIR.is_dir(), reason="qwen72b feature data not present locally")

EXPECTED_DIM = 8192
NPZ_KEYS = {"train_full", "test_full", "cands", "train_label", "train_ids", "test_ids", "info_json"}
TASKS = gd.TASKS
# Sane, data-driven bounds for a raw (embd_normalize=-1) last-token hidden state at this model's
# scale: observed rows across all 13 tasks sit in 265-409; this is a much looser envelope meant to
# catch collapse (~0) or a runaway/garbage decode (~1e5+), not to pin the exact observed range.
MIN_SANE_NORM, MAX_SANE_NORM = 10.0, 5_000.0


def npz_path(task: str) -> Path:
    return FEATURES_DIR / f"{task}.npz"


def load_raw(task: str) -> dict:
    with np.load(npz_path(task), allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


# ------------------------------------------------------------------ presence

def test_all_13_grand_challenge_tasks_are_present():
    assert len(TASKS) == 13
    missing = [t for t in TASKS if not npz_path(t).is_file()]
    assert missing == [], f"missing npz for tasks: {missing}"
    extra = [p.stem for p in FEATURES_DIR.glob("*.npz") if p.stem not in TASKS]
    assert extra == [], f"unexpected extra npz files not in the 13-task list: {extra}"


def test_manifest_json_describes_the_qwen72b_run():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    assert manifest["feature_dim"] == EXPECTED_DIM
    assert "qwen2.5-72b" in manifest["encoder"].lower()
    assert set(manifest["budgets"].keys()) == set(TASKS)


# ------------------------------------------------------------------ per-task contract

@pytest.mark.parametrize("task", TASKS)
def test_npz_has_exactly_the_contract_keys(task):
    with np.load(npz_path(task), allow_pickle=False) as z:
        assert set(z.files) == NPZ_KEYS, f"{task}: key set {set(z.files)} != {NPZ_KEYS}"


@pytest.mark.parametrize("task", TASKS)
def test_feature_dim_is_exactly_8192(task):
    z = load_raw(task)
    for block in ("train_full", "test_full", "cands"):
        assert z[block].ndim == 2, f"{task}/{block}: expected 2D, got {z[block].shape}"
        assert z[block].shape[1] == EXPECTED_DIM, f"{task}/{block}: dim {z[block].shape[1]} != {EXPECTED_DIM}"


@pytest.mark.parametrize("task", TASKS)
def test_no_nan_or_inf_anywhere_in_the_feature_blocks(task):
    z = load_raw(task)
    for block in ("train_full", "test_full", "cands"):
        assert np.all(np.isfinite(z[block])), f"{task}/{block}: contains NaN or Inf"


@pytest.mark.parametrize("task", TASKS)
def test_row_norms_are_in_a_sane_range_no_collapse_no_blowup(task):
    z = load_raw(task)
    for block in ("train_full", "test_full"):
        norms = np.linalg.norm(z[block], axis=1)
        assert norms.min() > MIN_SANE_NORM, f"{task}/{block}: near-zero row norm {norms.min()} (collapsed encoder?)"
        assert norms.max() < MAX_SANE_NORM, f"{task}/{block}: runaway row norm {norms.max()}"
        # a real encoder should not emit an (almost) constant vector for every distinct input
        assert norms.std() > 1e-3, f"{task}/{block}: row norms have ~zero variance ({norms.std()}), looks collapsed"


@pytest.mark.parametrize("task", TASKS)
def test_row_counts_are_internally_consistent(task):
    """train_full/train_ids/train_label agree; test_full/test_ids agree. Reuses the same loader
    cross_model_manifold_alignment.py and sota_ensemble_experts.py both trust, so this test and
    the downstream analysis in this suite check the identical contract."""
    data = load_features(npz_path(task))  # raises ValueError internally on any row-count disagreement
    assert data["train_full"].shape[0] == data["train_ids"].shape[0] == data["train_label"].shape[0]
    assert data["test_full"].shape[0] == data["test_ids"].shape[0]
    assert data["train_full"].shape[0] > 0 and data["test_full"].shape[0] > 0


@pytest.mark.parametrize("task", TASKS)
def test_train_ids_are_unique_and_disjoint_from_test_ids(task):
    z = load_raw(task)
    train_ids, test_ids = z["train_ids"], z["test_ids"]
    assert len(set(train_ids.tolist())) == len(train_ids), f"{task}: duplicate train_ids"
    assert len(set(test_ids.tolist())) == len(test_ids), f"{task}: duplicate test_ids"
    assert set(train_ids.tolist()).isdisjoint(test_ids.tolist()), f"{task}: train/test id overlap"


@pytest.mark.parametrize("task", TASKS)
def test_info_json_records_the_q72b_last_variant_and_this_dim(task):
    z = load_raw(task)
    info = json.loads(str(z["info_json"]))
    assert info["task"] == task
    assert info["feature_dim"] == EXPECTED_DIM
    assert info["variant"] == "q72b_last"


# ------------------------------------------------------------------ cross-model reference row count

def test_massive_en_row_and_label_counts_match_the_gte7b_reference_run():
    """The only task for which a second model's real extraction exists locally
    (data/extracted_features/gte7b_cpu/massive_en.npz). If qwen72b's massive_en did not draw the
    exact same train/test rows in the exact same order, every downstream CKA/Procrustes number
    computed against it would be comparing unrelated examples."""
    assert GTE7B_MASSIVE_EN.is_file(), "reference file missing; cannot cross-check row identity"
    ours = load_raw("massive_en")
    with np.load(GTE7B_MASSIVE_EN, allow_pickle=False) as z:
        ref = {k: np.array(z[k]) for k in z.files}
    assert ours["train_ids"].shape[0] == ref["train_ids"].shape[0] == 1000
    assert ours["test_ids"].shape[0] == ref["test_ids"].shape[0] == 350
    assert np.array_equal(ours["train_ids"], ref["train_ids"])
    assert np.array_equal(ours["test_ids"], ref["test_ids"])
    assert np.array_equal(ours["train_label"], ref["train_label"])


@pytest.mark.parametrize("task", TASKS)
def test_test_set_ground_truth_is_loadable_and_row_count_matches(task):
    """Cross-check against grand_challenge_data's own task loader (the same source the extractor
    built its rows from): the number of test rows in the npz must equal what gd.load_test(task)
    returns today, and every test id in the npz must have a matching row."""
    rows = gd.load_test(task)
    z = load_raw(task)
    assert len(rows) == z["test_ids"].shape[0], (
        f"{task}: gd.load_test returns {len(rows)} rows, npz has {z['test_ids'].shape[0]} test_ids"
    )
    assert {r["id"] for r in rows} == set(z["test_ids"].tolist())
