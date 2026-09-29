#!/usr/bin/env python3
"""Spec 19 Phase 4 / Spec 20 S5: layer-difference feature sources from cached Qwen3.5-9B blocks.

No GPU, no re-extraction. The two cached variants of
gpu_extract_qwen35_9b_grand_challenge_a100.py share the same rows, so the
per-layer 4096-D blocks can be sliced back out and recombined:

    q9b_mid  = [ mean@16 | last@16    ]   ->  h16    = mid[:, D/2:]
    q9b_late = [ last@24 | last@final ]   ->  h24    = late[:, :D/2]
                                              hfinal = late[:, D/2:]

Modes (fixed before any score is seen):

    full       [h16; h24; h24 - h16]          3 blocks (12288-D)
    compact    [h24; h24 - h16]               2 blocks (8192-D)
    pure_diff  [h24 - h16]                    1 block  (4096-D)
    late_diff  [h24; hfinal; hfinal - h24]    3 blocks (12288-D), late file only

Every feature array of the source (train_full, test_full, cands, and the pair
arrays train_a/train_b/test_a/test_b when present) gets the same transform, so
the output is a complete --source for benchmark_sota_ensemble.py
(<out>/features/<task>.npz) and a --features-dir for the grand scorecard.

Guards (each raises before anything is written): block order is read from
info_json["encoder"], never assumed; train_ids, test_ids and train_label must
be identical row for row; row counts, widths and key sets must match; the two
files must carry the same task / n_train_max / pubmedqa_extra / max_tok /
head_tok / encoder prefix; all values must be finite.

Honesty note carried into info_json (Spec 20 S5): [h16; h24; h24-h16] adds no
linearly expressible information over [h16; h24]; any gain is a change of
regularisation, and the matched-capacity control for late+delta is
[h24; hfinal; h16], not the diff alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grand_challenge_data as gd  # noqa: E402

MID_BLOCKS = ("mean@16", "last@16")
LATE_BLOCKS = ("last@24", "last@final")
FEATURE_KEYS = ("train_full", "test_full", "cands", "train_a", "train_b", "test_a", "test_b")
ALIGN_KEYS = ("train_ids", "test_ids", "train_label")
FINGERPRINT_KEYS = ("task", "n_train_max", "pubmedqa_extra", "max_tok", "head_tok")
# mode -> (variant suffix, ordered output blocks). A block is a name or (a, b) meaning a - b.
MODES: Dict[str, Tuple[str, Tuple[object, ...]]] = {
    "full": ("diff_16_24", ("last@16", "last@24", ("last@24", "last@16"))),
    "compact": ("diff_16_24_compact", ("last@24", ("last@24", "last@16"))),
    "pure_diff": ("diff_16_24_pure", (("last@24", "last@16"),)),
    "late_diff": ("diff_24_final", ("last@24", "last@final", ("last@final", "last@24"))),
}
LINEAR_REDUNDANCY_NOTE = ("[h_a; h_b; h_b - h_a] spans the same linear space as [h_a; h_b]: any gain over the "
                          "two-block source is a regularisation change, not new information (Spec 20 S5). "
                          "Matched-capacity control for late+delta is [h24; hfinal; h16].")


class LayerDiffError(ValueError):
    """A source pair that must not be sliced (misaligned rows, wrong block order, mixed caps)."""


# ---------------------------------------------------------------- pure numpy core

def parse_blocks(encoder: str) -> Tuple[str, Tuple[str, ...]]:
    """('<model>:bf16:text-backbone', ('mean@16', 'last@16')) from the extractor's encoder string.

    The prefix itself contains colons, so only the last ':' separates the block list."""
    if ":" not in encoder:
        raise LayerDiffError(f"encoder {encoder!r} carries no block list")
    prefix, blocks = encoder.rsplit(":", 1)
    names = tuple(b for b in blocks.split("+") if b)
    if len(names) != 2:
        raise LayerDiffError(f"encoder {encoder!r}: expected exactly two blocks, got {names}")
    return prefix, names


def block_slices(info: dict, expected: Sequence[str]) -> Dict[str, slice]:
    """block name -> column slice, from info_json. Refuses any block order other than `expected`."""
    prefix, names = parse_blocks(str(info.get("encoder", "")))
    if names != tuple(expected):
        raise LayerDiffError(f"encoder {info.get('encoder')!r} lists blocks {names}, need {tuple(expected)} in that order")
    D = int(info.get("feature_dim", 0))
    if D <= 0 or D % 2:
        raise LayerDiffError(f"feature_dim {D} is not a positive even width")
    half = D // 2
    return {names[0]: slice(0, half), names[1]: slice(half, D)}


def _check_arrays(name: str, f: dict, info: dict) -> None:
    D = int(info["feature_dim"])
    n_tr, n_te = len(f["train_ids"]), len(f["test_ids"])
    if len(f["train_label"]) != n_tr:
        raise LayerDiffError(f"{name}: train_label has {len(f['train_label'])} rows, train_ids {n_tr}")
    rows = {"train_full": n_tr, "test_full": n_te, "train_a": n_tr, "train_b": n_tr, "test_a": n_te, "test_b": n_te}
    for k in FEATURE_KEYS:
        if k not in f:
            continue
        a = f[k]
        if a.ndim != 2 or a.shape[1] != D:
            raise LayerDiffError(f"{name}/{k}: shape {a.shape}, need (rows, {D})")
        if k in rows and a.shape[0] != rows[k]:
            raise LayerDiffError(f"{name}/{k}: {a.shape[0]} rows, ids say {rows[k]}")
        if not np.isfinite(a).all():
            raise LayerDiffError(f"{name}/{k}: non-finite values")


def check_alignment(mid: Optional[dict], late: dict) -> None:
    """Row-for-row id and label identity plus the extraction fingerprint. mid may be None (late_diff)."""
    for name, f in (("mid", mid), ("late", late)):
        if f is None:
            continue
        missing = [k for k in ("train_full", "test_full", "cands", *ALIGN_KEYS, "info") if k not in f]
        if missing:
            raise LayerDiffError(f"{name}: missing keys {missing}")
        _check_arrays(name, f, f["info"])
    if mid is None:
        return
    if set(k for k in FEATURE_KEYS if k in mid) != set(k for k in FEATURE_KEYS if k in late):
        raise LayerDiffError("mid and late carry different feature arrays "
                             f"({sorted(k for k in FEATURE_KEYS if k in mid)} vs {sorted(k for k in FEATURE_KEYS if k in late)})")
    for k in ALIGN_KEYS:
        a, b = np.asarray(mid[k]), np.asarray(late[k])
        if a.shape != b.shape:
            raise LayerDiffError(f"{k}: mid has {a.shape}, late has {b.shape}")
        bad = np.flatnonzero(a != b)
        if bad.size:
            i = int(bad[0])
            raise LayerDiffError(f"{k}: {bad.size} rows differ, first at row {i}: mid {a[i]!r} vs late {b[i]!r}")
    for k in FINGERPRINT_KEYS:
        if mid["info"].get(k) != late["info"].get(k):
            raise LayerDiffError(f"info_json[{k!r}] differs: mid {mid['info'].get(k)!r}, late {late['info'].get(k)!r}")
    if parse_blocks(mid["info"]["encoder"])[0] != parse_blocks(late["info"]["encoder"])[0]:
        raise LayerDiffError(f"encoder prefix differs: {mid['info']['encoder']!r} vs {late['info']['encoder']!r}")
    if int(mid["info"]["feature_dim"]) != int(late["info"]["feature_dim"]):
        raise LayerDiffError(f"feature_dim differs: mid {mid['info']['feature_dim']}, late {late['info']['feature_dim']}")


def compose(blocks: Dict[str, np.ndarray], layout: Sequence[object]) -> np.ndarray:
    """Concatenate named blocks and (a, b) differences, computed in float32."""
    parts = []
    for item in layout:
        if isinstance(item, tuple):
            a, b = item
            parts.append(blocks[a].astype(np.float32) - blocks[b].astype(np.float32))
        else:
            parts.append(blocks[item].astype(np.float32))
    return np.concatenate(parts, axis=1)


def synthesize(mid: Optional[dict], late: dict, mode: str) -> dict:
    """The new source (feature arrays, ids, labels, info) for one task. Pure NumPy, no I/O.

    `mid` / `late` are dicts of npz arrays plus "info" (parsed info_json), as load_npz returns."""
    if mode not in MODES:
        raise LayerDiffError(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
    suffix, layout = MODES[mode]
    needs_mid = any(("last@16" in (item if isinstance(item, tuple) else (item,))) for item in layout)
    if needs_mid and mid is None:
        raise LayerDiffError(f"mode {mode!r} needs the mid source")
    check_alignment(mid, late)
    late_sl = block_slices(late["info"], LATE_BLOCKS)
    mid_sl = block_slices(mid["info"], MID_BLOCKS) if mid is not None else {}
    out: Dict[str, object] = {k: np.asarray(late[k]).copy() for k in ALIGN_KEYS}
    for k in FEATURE_KEYS:
        if k not in late:
            continue
        blocks = {name: late[k][:, sl] for name, sl in late_sl.items()}
        if mid is not None:
            blocks.update({name: mid[k][:, sl] for name, sl in mid_sl.items()})
        out[k] = compose(blocks, layout)
    prefix = parse_blocks(late["info"]["encoder"])[0]
    names = [f"{item[0]}-{item[1]}" if isinstance(item, tuple) else item for item in layout]
    base = str(late["info"].get("variant", "q9b_late")).split("_")[0]   # "q9b_late" -> "q9b"
    info = {k: v for k, v in late["info"].items() if not k.endswith("_seconds") and not k.endswith("_tokens")}
    info.update(
        encoder=f"{prefix}:{suffix}:{'+'.join(names)}",
        variant=f"{base}_{suffix}",
        feature_dim=int(out["cands"].shape[1]),
        layer_diff={
            "mode": mode, "blocks": names, "block_width": int(late["info"]["feature_dim"]) // 2,
            "dtype": "float32", "linear_redundancy_note": LINEAR_REDUNDANCY_NOTE,
            "sources": {name: {"variant": f["info"].get("variant"), "encoder": f["info"].get("encoder"),
                               "feature_dim": int(f["info"]["feature_dim"]), "sha256": f.get("sha256"),
                               "path": f.get("path"), "train_full_shape": list(f["train_full"].shape)}
                        for name, f in (("mid", mid), ("late", late)) if f is not None},
            "synthesized_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    out["info"] = info
    return out


# ---------------------------------------------------------------- I/O

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        f = {k: z[k] for k in z.files}
    f["info"] = json.loads(str(f.pop("info_json")))
    f["path"], f["sha256"] = str(path), sha256_of(path)
    if f["info"].get("task") != path.stem:
        raise LayerDiffError(f"{path}: info_json task {f['info'].get('task')!r} != file stem {path.stem!r}")
    return f


def save_npz(out: dict, path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in out.items() if k != "info"}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, info_json=np.array(json.dumps(out["info"])), **arrays)
    tmp.replace(path)


def run(mid_dir: Optional[Path], late_dir: Path, out_dir: Path, mode: str, tasks: List[str],
        overwrite: bool = False) -> Dict[str, dict]:
    feats = out_dir / "features"
    manifest = {}
    for task in tasks:
        late = load_npz(late_dir / f"{task}.npz")
        mid = load_npz(mid_dir / f"{task}.npz") if mid_dir is not None else None
        out = synthesize(mid, late, mode)
        dst = feats / f"{task}.npz"
        save_npz(out, dst, overwrite=overwrite)
        manifest[task] = {"path": str(dst), "feature_dim": out["info"]["feature_dim"],
                          "train_rows": int(out["train_full"].shape[0]), "test_rows": int(out["test_full"].shape[0]),
                          "variant": out["info"]["variant"], "sources": out["info"]["layer_diff"]["sources"]}
        print(f"[synth] {task}: {out['info']['variant']} {out['train_full'].shape} -> {dst}", flush=True)
    (out_dir / "layer_diff_manifest.json").write_text(
        json.dumps({"mode": mode, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "command": " ".join(sys.argv), "tasks": manifest}, indent=2), encoding="utf-8")
    return manifest


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mid-dir", type=Path, default=None, help="dir of q9b_mid <task>.npz (required unless --mode late_diff)")
    ap.add_argument("--late-dir", type=Path, required=True, help="dir of q9b_late <task>.npz")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="source root; files land in <out-dir>/features/<task>.npz "
                         "(--source name=<out-dir> for the ensemble, --features-dir <out-dir>/features for the scorecard)")
    ap.add_argument("--mode", choices=sorted(MODES), default="full")
    ap.add_argument("--tasks", default="", help="comma list (default: all 13)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(gd.TASKS)
    bad = [t for t in tasks if t not in gd.TASKS]
    if bad:
        raise SystemExit(f"unknown tasks {bad}; valid: {list(gd.TASKS)}")
    if args.mode != "late_diff" and args.mid_dir is None:
        raise SystemExit(f"--mode {args.mode} needs --mid-dir")
    run(args.mid_dir, args.late_dir, args.out_dir, args.mode, tasks, overwrite=args.overwrite)
    print(f"[synth] done: point the scorecard at {args.out_dir / 'features'}", flush=True)


if __name__ == "__main__":
    main()
