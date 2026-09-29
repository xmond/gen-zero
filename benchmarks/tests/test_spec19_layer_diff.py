"""Spec 19 Phase 4 direction 2: layer-difference feature synthesis (synthesize_layer_diff_features).

All data is synthetic: two fake q9b_mid / q9b_late sources with a small even width whose
halves hold known values, so slice/diff correctness is checked bit-exactly (values are
small integers, exact in float32). Covers every mode, zero/constant difference, the
rejections (ids, labels, rows, width, block order, fingerprint), .npz round-trip, that the
result passes benchmark_sota_ensemble.load_sources' cross-source check next to the
original, and that a 12288-D source still registers lin_full under DEFAULT_RAW_MAX_DIM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(SUITES))

import benchmark_sota_ensemble as bse  # noqa: E402
import synthesize_layer_diff_features as sl  # noqa: E402

ENC = "Qwen3.5-9B:bf16:text-backbone"
D, HALF, N_TR, N_TE, K = 8, 4, 6, 3, 2


def _info(variant, blocks, **over):
    info = {"task": "boolq", "gate": "x", "n_train_max": None, "pubmedqa_extra": "", "max_tok": 1536,
            "head_tok": 256, "encoder": f"{ENC}:{'+'.join(blocks)}", "variant": variant, "feature_dim": D,
            "train_full_seconds": 1.0, "train_full_tokens": 10}
    info.update(over)
    return info


def _source(variant, blocks, first, second, pair=False, seed=0):
    """A fake feature file: every array's first half is `first`, second half `second` (per-row offsets)."""
    rng = np.random.default_rng(seed)

    def mat(n):
        rows = np.arange(n, dtype=np.float32)[:, None]
        a = np.full((n, HALF), first, np.float32) + rows
        b = np.full((n, HALF), second, np.float32) + rows
        return np.concatenate([a, b], axis=1)

    f = {"train_full": mat(N_TR), "test_full": mat(N_TE), "cands": mat(K),
         "train_ids": np.array([f"tr{i}" for i in range(N_TR)]), "test_ids": np.array([f"te{i}" for i in range(N_TE)]),
         "train_label": rng.integers(0, K, N_TR).astype(np.int64), "info": _info(variant, blocks)}
    if pair:
        f.update(train_a=mat(N_TR), train_b=mat(N_TR), test_a=mat(N_TE), test_b=mat(N_TE))
    return f


def _mid(first=1.0, second=2.0, **kw):
    return _source("q9b_mid", sl.MID_BLOCKS, first, second, **kw)


def _late(first=5.0, second=9.0, **kw):
    return _source("q9b_late", sl.LATE_BLOCKS, first, second, **kw)


def _rows(base, n):
    """The (n, HALF) block a fake source holds: `base` plus the row index, exact in float32."""
    return np.broadcast_to(base + np.arange(n, dtype=np.float32)[:, None], (n, HALF)).astype(np.float32)


def _same_labels(mid, late):
    late["train_label"] = mid["train_label"].copy()
    return mid, late


# ------------------------------------------------------------- slicing and differences

def test_block_slices_read_order_from_info_json():
    assert sl.block_slices(_info("q9b_mid", sl.MID_BLOCKS), sl.MID_BLOCKS) == {"mean@16": slice(0, 4), "last@16": slice(4, 8)}
    assert sl.block_slices(_info("q9b_late", sl.LATE_BLOCKS), sl.LATE_BLOCKS) == {"last@24": slice(0, 4), "last@final": slice(4, 8)}
    with pytest.raises(sl.LayerDiffError, match="in that order"):
        sl.block_slices(_info("q9b_late", ("last@final", "last@24")), sl.LATE_BLOCKS)
    with pytest.raises(sl.LayerDiffError, match="even"):
        sl.block_slices(_info("q9b_mid", sl.MID_BLOCKS, feature_dim=7), sl.MID_BLOCKS)


def test_full_mode_is_h16_h24_delta():
    mid, late = _same_labels(_mid(1.0, 2.0), _late(5.0, 9.0))     # h16 = 2+row, h24 = 5+row, hfinal = 9+row
    out = sl.synthesize(mid, late, "full")
    X = out["train_full"]
    assert X.shape == (N_TR, 3 * HALF) and X.dtype == np.float32
    assert np.array_equal(X[:, :HALF], _rows(2.0, N_TR))                 # h16 = mid second half
    assert np.array_equal(X[:, HALF:2 * HALF], _rows(5.0, N_TR))   # h24 = late first half
    assert np.array_equal(X[:, 2 * HALF:], np.full((N_TR, HALF), 3.0, np.float32))   # h24 - h16 = 3
    assert out["info"]["feature_dim"] == 3 * HALF
    assert out["info"]["variant"] == "q9b_diff_16_24"
    assert out["info"]["encoder"] == f"{ENC}:diff_16_24:last@16+last@24+last@24-last@16"
    assert out["info"]["layer_diff"]["blocks"] == ["last@16", "last@24", "last@24-last@16"]
    assert "regularisation" in out["info"]["layer_diff"]["linear_redundancy_note"]
    for k in ("n_train_max", "pubmedqa_extra", "max_tok", "head_tok", "task"):
        assert out["info"][k] == late["info"][k]
    assert "train_full_seconds" not in out["info"]                  # timings are the source's, not ours


@pytest.mark.parametrize("mode,width,blocks", [
    ("compact", 2 * HALF, ["last@24", "last@24-last@16"]),
    ("pure_diff", HALF, ["last@24-last@16"]),
    ("late_diff", 3 * HALF, ["last@24", "last@final", "last@final-last@24"]),
])
def test_other_modes_widths_and_values(mode, width, blocks):
    mid, late = _same_labels(_mid(1.0, 2.0), _late(5.0, 9.0))
    out = sl.synthesize(mid, late, mode)
    X = out["test_full"]
    assert X.shape == (N_TE, width)
    assert out["info"]["layer_diff"]["blocks"] == blocks
    if mode == "compact":
        assert np.array_equal(X[:, :HALF], _rows(5.0, N_TE)) and np.array_equal(X[:, HALF:], np.full((N_TE, HALF), 3.0, np.float32))
    elif mode == "pure_diff":
        assert np.array_equal(X, np.full((N_TE, HALF), 3.0, np.float32))
    else:
        assert np.array_equal(X[:, :HALF], _rows(5.0, N_TE)) and np.array_equal(X[:, HALF:2 * HALF], _rows(9.0, N_TE))
        assert np.array_equal(X[:, 2 * HALF:], np.full((N_TE, HALF), 4.0, np.float32))


def test_late_diff_needs_no_mid_but_full_does():
    out = sl.synthesize(None, _late(), "late_diff")
    assert out["train_full"].shape == (N_TR, 3 * HALF)
    assert list(out["info"]["layer_diff"]["sources"]) == ["late"]
    with pytest.raises(sl.LayerDiffError, match="needs the mid source"):
        sl.synthesize(None, _late(), "full")
    with pytest.raises(sl.LayerDiffError, match="unknown mode"):
        sl.synthesize(_mid(), _late(), "bogus")


def test_zero_difference_when_h24_equals_h16():
    mid, late = _same_labels(_mid(1.0, 7.0), _late(7.0, 9.0))     # h16 == h24 == 7+row
    for mode in ("full", "compact", "pure_diff"):
        X = sl.synthesize(mid, late, mode)["train_full"]
        delta = X[:, -HALF:]
        assert np.array_equal(delta, np.zeros((N_TR, HALF), np.float32))
        assert not np.signbit(delta).any()                          # +0.0, not -0.0


def test_constant_difference_every_row_equals_c():
    c = -2.5                                                        # exact in float32
    mid, late = _same_labels(_mid(0.0, 4.0), _late(4.0 + c, 0.0))
    for key in ("train_full", "test_full", "cands"):
        delta = sl.synthesize(mid, late, "pure_diff")[key]
        assert np.array_equal(delta, np.full(delta.shape, c, np.float32))


def test_pair_arrays_get_the_same_transform_and_key_sets_must_match():
    mid, late = _same_labels(_mid(pair=True), _late(pair=True))
    out = sl.synthesize(mid, late, "full")
    for k in ("train_a", "train_b", "test_a", "test_b", "cands"):
        assert out[k].shape[1] == 3 * HALF
        assert np.array_equal(out[k][:, -HALF:], np.full((out[k].shape[0], HALF), 3.0, np.float32))
    mid_only_pair, late_no_pair = _same_labels(_mid(pair=True), _late())
    with pytest.raises(sl.LayerDiffError, match="different feature arrays"):
        sl.synthesize(mid_only_pair, late_no_pair, "full")


# ------------------------------------------------------------- rejections

def test_rejects_id_mismatch():
    mid, late = _same_labels(_mid(), _late())
    late["test_ids"] = late["test_ids"].copy()
    late["test_ids"][1] = "other"
    with pytest.raises(sl.LayerDiffError, match=r"test_ids: 1 rows differ, first at row 1"):
        sl.synthesize(mid, late, "full")
    mid, late = _same_labels(_mid(), _late())
    mid["train_ids"] = mid["train_ids"][::-1].copy()
    with pytest.raises(sl.LayerDiffError, match="train_ids"):
        sl.synthesize(mid, late, "compact")


def test_rejects_label_mismatch_and_row_count_mismatch():
    mid, late = _same_labels(_mid(), _late())
    late["train_label"] = (late["train_label"] + 1) % K
    with pytest.raises(sl.LayerDiffError, match="train_label"):
        sl.synthesize(mid, late, "full")
    mid, late = _same_labels(_mid(), _late())
    late["train_full"] = late["train_full"][:-1]
    with pytest.raises(sl.LayerDiffError, match=r"late/train_full: 5 rows, ids say 6"):
        sl.synthesize(mid, late, "full")
    mid, late = _same_labels(_mid(), _late())
    mid["train_ids"] = mid["train_ids"][:-1]
    mid["train_label"] = mid["train_label"][:-1]
    mid["train_full"] = mid["train_full"][:-1]
    with pytest.raises(sl.LayerDiffError, match=r"train_ids: mid has \(5,\), late has \(6,\)"):
        sl.synthesize(mid, late, "full")


def test_rejects_width_mismatch_and_bad_block_order_and_nonfinite():
    mid, late = _same_labels(_mid(), _late())
    late["test_full"] = np.concatenate([late["test_full"], late["test_full"]], axis=1)
    with pytest.raises(sl.LayerDiffError, match=r"late/test_full: shape \(3, 16\), need \(rows, 8\)"):
        sl.synthesize(mid, late, "full")
    mid, late = _same_labels(_mid(), _late())
    late["info"]["encoder"] = f"{ENC}:last@final+last@24"
    with pytest.raises(sl.LayerDiffError, match="in that order"):
        sl.synthesize(mid, late, "full")
    mid, late = _same_labels(_mid(), _late())
    mid["train_full"][2, 0] = np.nan
    with pytest.raises(sl.LayerDiffError, match="mid/train_full: non-finite"):
        sl.synthesize(mid, late, "full")


def test_rejects_fingerprint_mismatch():
    for key, val in (("n_train_max", 1000), ("max_tok", 256), ("task", "paws"), ("pubmedqa_extra", "artificial")):
        mid, late = _same_labels(_mid(), _late())
        mid["info"][key] = val
        with pytest.raises(sl.LayerDiffError, match=f"info_json\\['{key}'\\] differs"):
            sl.synthesize(mid, late, "full")
    mid, late = _same_labels(_mid(), _late())
    mid["info"]["encoder"] = "OtherModel:bf16:text-backbone:mean@16+last@16"
    with pytest.raises(sl.LayerDiffError, match="encoder prefix differs"):
        sl.synthesize(mid, late, "full")


# ------------------------------------------------------------- npz round trip and downstream fit

def _write(f, path):
    arrays = {k: v for k, v in f.items() if k != "info"}
    np.savez(path, info_json=np.array(json.dumps(f["info"])), **arrays)


def test_npz_round_trip_is_bit_exact_and_refuses_silent_overwrite(tmp_path):
    mid, late = _same_labels(_mid(pair=True), _late(pair=True))
    out = sl.synthesize(mid, late, "full")
    dst = tmp_path / "features" / "boolq.npz"
    sl.save_npz(out, dst)
    assert dst.exists() and not dst.with_suffix(".tmp.npz").exists()
    back = sl.load_npz(dst)
    for k in sl.FEATURE_KEYS + sl.ALIGN_KEYS:
        assert np.array_equal(back[k], out[k]) and back[k].dtype == out[k].dtype, k
    assert back["info"] == out["info"]
    with pytest.raises(FileExistsError):
        sl.save_npz(out, dst)
    sl.save_npz(out, dst, overwrite=True)


def test_run_end_to_end_and_load_sources_next_to_original(tmp_path, monkeypatch):
    mid, late = _same_labels(_mid(pair=True), _late(pair=True))
    for name, f in (("mid", mid), ("late", late)):
        (tmp_path / name).mkdir()
        _write(f, tmp_path / name / "boolq.npz")
    out_root = tmp_path / "diff"
    manifest = sl.run(tmp_path / "mid", tmp_path / "late", out_root, "compact", ["boolq"])
    assert manifest["boolq"]["feature_dim"] == 2 * HALF
    assert (out_root / "layer_diff_manifest.json").exists()
    # The mid source, laid out as an ART root, and the new source sit side by side in the ensemble loader.
    (tmp_path / "mid_art" / "features").mkdir(parents=True)
    _write(mid, tmp_path / "mid_art" / "features" / "boolq.npz")
    monkeypatch.setattr(bse, "SOURCES", {"q9b_mid": tmp_path / "mid_art", "q9b_diff": out_root})
    fs = bse.load_sources("boolq")
    assert fs["q9b_diff"]["X_train"].shape == (N_TR, 2 * HALF)
    assert fs["q9b_diff"]["info"]["variant"] == "q9b_diff_16_24_compact"
    wrong = tmp_path / "late" / "paws.npz"                          # boolq info_json under a paws file name
    _write(late, wrong)
    with pytest.raises(sl.LayerDiffError, match="info_json task"):
        sl.load_npz(wrong)


def test_cli_rejects_missing_mid_dir_and_unknown_task(tmp_path):
    with pytest.raises(SystemExit, match="needs --mid-dir"):
        sl.main(["--late-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--mode", "full"])
    with pytest.raises(SystemExit, match="unknown tasks"):
        sl.main(["--late-dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--mode", "late_diff", "--tasks", "nope"])


def test_12288_wide_source_registers_lin_full_under_default_raw_max_dim(monkeypatch):
    """Task premise check: RAW_MAX_DIM is 48000 by default, not 12000, so the full 12288-D probe is kept."""
    assert 3 * 4096 <= bse.DEFAULT_RAW_MAX_DIM
    monkeypatch.setattr(bse, "RAW_MAX_DIM", bse.DEFAULT_RAW_MAX_DIM)
    fs = {"q9b_diff_16_24": {"train_full": np.zeros((2, 3 * 4096), np.float32)}}
    names = [n for n, _, _ in bse.expert_specs("boolq", fs)]
    assert "q9b_diff_16_24:lin_full" in names and "q9b_diff_16_24:lin_full_pca" in names
    fs_pair = {"q9b_diff_16_24": {"train_full": np.zeros((2, 3 * 4096), np.float32)}}
    names_pair = [n for n, _, _ in bse.expert_specs("paws", fs_pair)]
    assert "q9b_diff_16_24:lin_full" in names_pair and "q9b_diff_16_24:lin_pair_pca" in names_pair
    # 4 * 12288 = 49152 > 48000: the raw pair map on a 12288-D source is skipped, only its PCA variant runs.
    assert "q9b_diff_16_24:lin_pair" not in names_pair
