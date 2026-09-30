"""Offline compiler: pack per-task CounterfactualDriftDynamics artifacts into one
mmap-able binary manifold file, `causal_codebook.bin`.

Honesty note (read before trusting anything this module produces as "knowledge"):
  The 13 tasks this repo tracks have real frozen Qwen3.5-9B hidden states for only
  30 ids each (`benchmarks/results/v5_hidden_features.npz`, extracted A100
  2026-09-22), and those 30 ids are the FROZEN TEST rows, not the family-disjoint
  calibration split (`benchmarks/data/calibration_clean_16.jsonl`). Fitting the
  *shipped* codebook on test-split features would be exactly the "29-shot
  transductive" leak `docs/architecture/15-offline-knowledge-compiler-and-cpu-runtime.md`
  §1.2 rejects. This host has no GPU and no 9B weights, so the true calibration
  split has never had an encoder pass and no real held-out-clean fit is possible
  here today.

  So the artifact this module ships is fit on SYNTHETIC feature vectors at the
  real production shapes (4096-dim raw encoder space, r=16 PCA components,
  8 counterfactual components, real per-task class counts read from each task's
  jsonl schema -- candidate *count*, never a label). Its `provenance.source` is
  literally "synthetic-shape-golden-v1" in every section, and `manifest["notes"]`
  repeats this in the file itself so nobody downstream can cite it as a trained
  model. This is scaffolding for the binary format, the runtime, and the
  latency/size budget -- not a claim about task accuracy. See
  `python/gen_zero/tests/test_compiled_manifold_runtime.py` for the one place a
  *real* fit (on real 9B features, calibration-fold only, never persisted) is
  used, to check the compiled runtime agrees with the reference Python dynamics.

Binary layout (`causal_codebook.bin`):
    0   8    magic b"GZCBK001"
    8   4    format_version (u32 LE)
    12  4    num_tasks (u32 LE)
    16  8    created_unix_ts (u64 LE)
    24  8    manifest_len (u64 LE)          bytes of the UTF-8 JSON manifest (unpadded)
    32  8    arrays_base_offset (u64 LE)    absolute file offset of the array section,
                                            64B aligned
    40  8    payload_len (u64 LE)           bytes of the payload, offset [80, 80+payload_len)
    48  32   sha256 of the payload (the exact bytes at [80, 80+payload_len))
    80  ...  payload: manifest JSON, zero-padded to 64B alignment, then the raw
             array section; each array's byte offset (relative to
             `arrays_base_offset`) is given in the manifest

Every task section stores A, its precomputed (I-A)^-1, B, W_c (zero matrix if
the task has no counterfactual signal), and the codebook as exact float32 (a
few KB total across all 13 tasks -- the dynamics carry no quantization error).
The 4096-dim projection bases (x_basis, c_basis) and means are the only large
arrays; bases are int8-quantized with a per-column scale (means stay float32 --
they are subtracted before the basis matmul and dominate accuracy far more than
the basis itself). The runtime's hot path never touches the bases: per the task
contract it receives already-projected `x_projected`/`c_projected` vectors, so
quantizing them only affects the *audit trail*, never inference numerics.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .counterfactual_drift_dynamics import (
    CounterfactualDriftDynamics,
    fit_counterfactual_drift_dynamics,
)

__all__ = [
    "MAGIC",
    "FORMAT_VERSION",
    "MAX_FILE_BYTES",
    "SYNTHETIC_SOURCE_TAG",
    "ALIGNMENT",
    "WORKING_SET_BUDGET_BYTES",
    "choose_compact_dims",
    "synthetic_shape_golden_dynamics",
    "compile_manifold",
    "compile_from_task_schemas",
    "read_task_schema",
]

MAGIC = b"GZCBK001"
FORMAT_VERSION = 1
MAX_FILE_BYTES = 2 * 1024 * 1024
ALIGNMENT = 64
SYNTHETIC_SOURCE_TAG = "synthetic-shape-golden-v1"
HEADER = struct.Struct("<8sIIQQQQ32s")
assert HEADER.size == 80


def _align_up(n: int, align: int = ALIGNMENT) -> int:
    return ((n + align - 1) // align) * align


def _quantize_columns(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column symmetric int8 quantization of a (rows, cols) float64/32 array."""
    mat = np.asarray(mat, dtype=np.float64)
    scale = np.max(np.abs(mat), axis=0)
    scale = np.where(scale > 0, scale, 1.0) / 127.0
    q = np.round(mat / scale[None, :]).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def read_task_schema(jsonl_path: Path) -> Tuple[int, Sequence[str]]:
    """Read only the *shape* of a task from its first record: candidate count and
    the candidate label strings themselves (fixed multiple-choice symbols such as
    "A".."D", not natural-language content). No ground-truth label is read."""
    with open(jsonl_path, encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    cands = first["candidates"]
    if not isinstance(cands, list) or len(cands) < 2:
        raise ValueError(f"{jsonl_path}: candidates must be a list of >=2 entries")
    return len(cands), list(cands)


WORKING_SET_BUDGET_BYTES = 4096  # 4.0 KiB, the online runtime's per-inference cache budget


def choose_compact_dims(k: int, *, base_dim: int = 16, base_n_components: int = 16,
                         base_cf_components: int = 8,
                         budget_bytes: int = WORKING_SET_BUDGET_BYTES) -> Tuple[int, int, int]:
    """Pick (dim, n_components, cf_components) for a task with `k` classes.

    `dim` must satisfy `dim >= k` (one codebook vertex per class, see
    `simplex_codebook`). For k <= base_dim this is exactly (16, 16, 8): the
    online working set (`TaskRuntimeSection.working_set_bytes`) comes out to
    2.8-3.2 KB, comfortably under budget. Tasks with more classes than
    `base_dim` (only massive_en/massive_de, k=18, in this repo's 13 tasks)
    need a larger `dim`, which grows the dynamics matrices quadratically; to
    stay under `budget_bytes` this shrinks `n_components`/`cf_components`
    (the compact-space rank) rather than dropping classes or lying about the
    working set.
    """
    dim = max(base_dim, k)
    n_components, cf_components = base_n_components, base_cf_components

    def working_set(d, p, pc, k_):
        dynamics = 4 * (d * d + d * p + d * pc + 3 * d + p + pc)
        codebook = 4 * (k_ * d)
        return dynamics + codebook

    while working_set(dim, n_components, cf_components, k) > budget_bytes and \
            (n_components > 2 or cf_components > 1):
        if n_components > 2:
            n_components -= 1
        if cf_components > 1:
            cf_components -= 1
    return dim, n_components, cf_components


def synthetic_shape_golden_dynamics(
    task_id: str, k: int, *, paired: bool, raw_dim: int = 4096, dim: int = 16,
    n_components: int = 16, cf_components: int = 8, n_samples: int = 48,
    seed: int = 0,
) -> Tuple[CounterfactualDriftDynamics, dict]:
    """Fit a CounterfactualDriftDynamics artifact on synthetic vectors that share
    the *shape* of the real pipeline (raw_dim, n_components, cf_components, k)
    but carry no real-world signal. Explicitly not a trained model; see module
    docstring. `use_counterfactual=paired` matches the real pipeline's rule that
    only paired-field tasks (multinli/paws/vitaminc) have a counterfactual view.
    """
    dim = max(dim, k)
    rng = np.random.default_rng(seed)
    dirs_x = rng.standard_normal((k, raw_dim))
    y = np.array([i % k for i in range(n_samples)], dtype=np.int64)
    rng.shuffle(y)
    x = rng.standard_normal((n_samples, raw_dim)) + 0.8 * dirs_x[y]
    if paired:
        dirs_c = rng.standard_normal((k, raw_dim))
        c = rng.standard_normal((n_samples, raw_dim)) + 0.5 * dirs_c[y]
    else:
        c = rng.standard_normal((n_samples, raw_dim))
    ids = [f"{task_id}-synthetic-{i}" for i in range(n_samples)]
    dyn, info = fit_counterfactual_drift_dynamics(
        x, c, y, sample_ids=ids, source=SYNTHETIC_SOURCE_TAG, split="calibration",
        encoder_id="synthetic-identity-v1", n_classes=k, dim=dim,
        n_components=min(n_components, raw_dim), cf_components=min(cf_components, raw_dim),
        use_counterfactual=paired)
    return dyn, info


def _pack_task(task_id: str, dyn: CounterfactualDriftDynamics, paired: bool) -> Tuple[dict, Dict[str, bytes]]:
    d, p, pc, k = dyn.dim, dyn.B.shape[1], dyn.W_c.shape[1], dyn.codebook.shape[0]
    inv_i_minus_a = np.linalg.inv(np.eye(d) - dyn.A)
    x_q, x_scale = _quantize_columns(dyn.x_basis)
    blobs: Dict[str, bytes] = {
        "A": dyn.A.astype(np.float32).tobytes(),
        "inv_i_minus_a": inv_i_minus_a.astype(np.float32).tobytes(),
        "B": dyn.B.astype(np.float32).tobytes(),
        "W_c": dyn.W_c.astype(np.float32).tobytes(),
        "codebook": dyn.codebook.astype(np.float32).tobytes(),
        "x_mean": dyn.x_mean.astype(np.float32).tobytes(),
        "x_basis_q": x_q.tobytes(),
        "x_basis_scale": x_scale.tobytes(),
    }
    arrays_meta = {
        "A": {"shape": [d, d], "dtype": "f32"},
        "inv_i_minus_a": {"shape": [d, d], "dtype": "f32"},
        "B": {"shape": [d, p], "dtype": "f32"},
        "W_c": {"shape": [d, pc], "dtype": "f32"},
        "codebook": {"shape": [k, d], "dtype": "f32"},
        "x_mean": {"shape": list(dyn.x_mean.shape), "dtype": "f32"},
        "x_basis_q": {"shape": list(dyn.x_basis.shape), "dtype": "i8", "scale": "x_basis_scale"},
        "x_basis_scale": {"shape": list(x_scale.shape), "dtype": "f32"},
    }
    if paired:
        c_q, c_scale = _quantize_columns(dyn.c_basis)
        blobs["c_mean"] = dyn.c_mean.astype(np.float32).tobytes()
        blobs["c_basis_q"] = c_q.tobytes()
        blobs["c_basis_scale"] = c_scale.tobytes()
        arrays_meta["c_mean"] = {"shape": list(dyn.c_mean.shape), "dtype": "f32"}
        arrays_meta["c_basis_q"] = {"shape": list(dyn.c_basis.shape), "dtype": "i8", "scale": "c_basis_scale"}
        arrays_meta["c_basis_scale"] = {"shape": list(c_scale.shape), "dtype": "f32"}
    section = {
        "task_id": task_id, "k": k, "paired": paired, "dim": d,
        "n_components": p, "cf_components": pc,
        "rho_a": float(np.max(np.abs(np.linalg.eigvals(dyn.A)))),
        "provenance": dyn.provenance,
        "arrays": arrays_meta,
    }
    return section, blobs


def compile_manifold(
    dynamics_by_task: Mapping[str, Tuple[CounterfactualDriftDynamics, bool]],
    out_path: Path, *, notes: str,
) -> dict:
    """Pack a `{task_id: (dynamics, paired)}` map into `out_path`. Asserts the
    resulting file is under `MAX_FILE_BYTES`; refuses to write an oversized file."""
    task_ids = sorted(dynamics_by_task)
    sections = []
    all_blobs: Dict[str, bytes] = {}
    order: list[str] = []
    for task_id in task_ids:
        dyn, paired = dynamics_by_task[task_id]
        section, blobs = _pack_task(task_id, dyn, paired)
        for name, blob in blobs.items():
            key = f"{task_id}/{name}"
            all_blobs[key] = blob
            order.append(key)
        for arr_name, meta in section["arrays"].items():
            meta["_blob_key"] = f"{task_id}/{arr_name}"
        sections.append(section)

    array_offsets: Dict[str, int] = {}
    cursor = 0
    array_bytes: list[bytes] = []
    for key in order:
        blob = all_blobs[key]
        start = _align_up(cursor)
        pad = start - cursor
        if pad:
            array_bytes.append(b"\x00" * pad)
        array_offsets[key] = start
        array_bytes.append(blob)
        cursor = start + len(blob)
    arrays_blob = b"".join(array_bytes)
    for section in sections:
        for arr_name, meta in section["arrays"].items():
            del meta["_blob_key"]
            meta["offset_in_arrays"] = array_offsets[f"{section['task_id']}/{arr_name}"]
            meta["nbytes"] = len(all_blobs[f"{section['task_id']}/{arr_name}"])

    manifest = {
        "format_version": FORMAT_VERSION,
        "notes": notes,
        "tasks": sections,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_padded_len = _align_up(len(manifest_bytes))
    manifest_section = manifest_bytes + b"\x00" * (manifest_padded_len - len(manifest_bytes))

    payload_body = manifest_section + arrays_blob
    payload_sha256 = hashlib.sha256(payload_body).digest()

    arrays_base_offset = HEADER.size + manifest_padded_len
    header = HEADER.pack(
        MAGIC, FORMAT_VERSION, len(sections), int(time.time()),
        len(manifest_bytes), arrays_base_offset, len(payload_body), payload_sha256,
    )
    file_bytes = header + payload_body

    if len(file_bytes) >= MAX_FILE_BYTES:
        raise ValueError(
            f"compiled manifold is {len(file_bytes)} bytes, must be < {MAX_FILE_BYTES} "
            f"(2 MiB); refusing to write an oversized codebook")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(file_bytes)
    assert out_path.stat().st_size < MAX_FILE_BYTES

    return {
        "path": str(out_path),
        "file_size_bytes": len(file_bytes),
        "num_tasks": len(sections),
        "task_ids": task_ids,
        "payload_sha256": payload_sha256.hex(),
        "manifest": manifest,
    }


PAIRED_TASKS = ("multinli", "paws", "vitaminc")


def compile_from_task_schemas(
    data_dir: Path, out_path: Path, *, seed: int = 0,
    notes: Optional[str] = None,
) -> dict:
    """Compile the 13-task codebook from real task *schemas* (candidate counts
    only) in `data_dir`, using synthetic-shape-golden dynamics for every task.
    See module docstring for exactly why this is not a trained/calibrated model.
    """
    data_dir = Path(data_dir)
    task_files = sorted(
        p for p in data_dir.glob("*.jsonl")
        if p.stem not in {"all_benchmarks", "bespoke", "hans", "calibration_clean_16"}
        and not p.stem.startswith("bbh_")
    )
    if len(task_files) != 13:
        raise ValueError(f"expected 13 task files in {data_dir}, found {len(task_files)}: "
                          f"{[p.name for p in task_files]}")
    dynamics_by_task: Dict[str, Tuple[CounterfactualDriftDynamics, bool]] = {}
    for i, path in enumerate(task_files):
        task_id = path.stem
        k, _cands = read_task_schema(path)
        paired = task_id in PAIRED_TASKS
        dim, n_components, cf_components = choose_compact_dims(k)
        dyn, _info = synthetic_shape_golden_dynamics(
            task_id, k, paired=paired, dim=dim, n_components=n_components,
            cf_components=cf_components, seed=seed + i)
        dynamics_by_task[task_id] = (dyn, paired)
    notes = notes or (
        "SYNTHETIC-SHAPE-GOLDEN artifact. Every task section is fit on synthetic "
        "vectors at real production shapes (raw_dim=4096, r=16, cf=8, real "
        "per-task class count read from schema only). This is NOT a trained "
        "model and carries no natural-language knowledge. See "
        "python/gen_zero/causal/knowledge_compiler.py module docstring for why: "
        "the true family-disjoint calibration split has no extracted encoder "
        "features on this host (no GPU, no 9B weights), and the only real "
        "features available (benchmarks/results/v5_hidden_features.npz) are "
        "frozen TEST rows, which the compiler refuses to fit the shipped "
        "binary on to avoid the exact LOO/transductive leak "
        "docs/architecture/15-offline-knowledge-compiler-and-cpu-runtime.md "
        "%c1.2 rejects." % chr(0xa7)
    )
    return compile_manifold(dynamics_by_task, out_path, notes=notes)


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "benchmarks" / "data"
    out_path = root / "benchmarks" / "results" / "causal_codebook.bin"
    report = compile_from_task_schemas(data_dir, out_path)
    print(json.dumps({
        "path": report["path"],
        "file_size_bytes": report["file_size_bytes"],
        "max_file_bytes": MAX_FILE_BYTES,
        "num_tasks": report["num_tasks"],
        "task_ids": report["task_ids"],
        "payload_sha256": report["payload_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
