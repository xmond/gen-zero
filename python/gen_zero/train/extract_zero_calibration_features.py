#!/usr/bin/env python3
"""Extract Zero's own label-free 64-D manifold states for the labeled calibration split.

Runs the real ``ZeroStandaloneRuntime`` (Qwen2.5-0.5B trunk, INT8 weight-only,
single core, CPU only) over every record of ``benchmarks/data/calibration_clean_16.jsonl``,
projects prompt and candidate hidden states through the already-fit
``zero_manifold_v1.npz`` (label-free ZCA), and writes the resulting vectors plus
the calibration split's own ``ground_truth`` label (the *only* place a label is
read; the encoder and manifold see nothing but ``context``/``candidates`` text).

This calibration split is proven disjoint from the frozen 930-record test set
(see benchmarks/data/CALIBRATION_SPLIT.md); nothing here touches
``benchmarks/data/all_benchmarks.jsonl``.

Output schema (``.npz``, no pickle): candidate blocks are ragged (K varies
2..18 across the 13 tasks), so they are packed CSR-style --
``candidates_flat`` is (sum_K, dim) and ``offsets``/``counts`` locate each
record's own block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.zero_runtime import (  # noqa: E402
    ZeroStandaloneRuntime,
    build_int8_artifact,
    find_local_snapshot,
)

CALIBRATION = REPO / "benchmarks" / "data" / "calibration_clean_16.jsonl"
CALIBRATION_SHA256 = "bd45f4df430ee7c74440afca44b7880ac033e24013a34345ce4e54f1583efa4a"
ARTIFACTS = REPO / "benchmarks" / "artifacts" / "zero"
INT8_ARTIFACT = ARTIFACTS / "zero_int8_v2.safetensors"
MANIFOLD = ARTIFACTS / "zero_manifold_v1.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("int8", "bf16", "fp32"), default="int8")
    parser.add_argument("--out", type=Path, default=REPO / "benchmarks" / "artifacts" / "zero" /
                        "zero_calibration_features_v1.npz")
    args = parser.parse_args()

    observed = sha256_file(CALIBRATION)
    if observed != CALIBRATION_SHA256:
        raise SystemExit(f"calibration split hash mismatch: {observed} != {CALIBRATION_SHA256}")
    if not MANIFOLD.exists():
        raise SystemExit(f"no manifold at {MANIFOLD}; run benchmark_zero_cpu.py fit-manifold first")
    manifold_sha256 = sha256_file(MANIFOLD)

    if args.precision == "int8" and not INT8_ARTIFACT.exists():
        build_int8_artifact(find_local_snapshot(), INT8_ARTIFACT)
    runtime = ZeroStandaloneRuntime(
        precision=args.precision,
        int8_artifact=INT8_ARTIFACT if args.precision == "int8" else None,
        manifold_path=MANIFOLD,
    )

    records = []
    with open(CALIBRATION, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))

    z0s, candidates_flat, offsets, counts, positive_indices = [], [], [], [], []
    sample_ids, tasks = [], []
    cursor = 0
    for i, record in enumerate(records):
        candidates = list(record["candidates"])
        if record["ground_truth"] not in candidates:
            raise ValueError(f"{record['id']}: ground_truth not among its own candidates")
        q0, c_states, _info = runtime.encode_prompt_with_candidates(record["context"], candidates)
        z0 = runtime.manifold.project(q0)
        zc = runtime.manifold.project(c_states)
        z0s.append(z0)
        candidates_flat.append(zc)
        offsets.append(cursor)
        counts.append(len(candidates))
        positive_indices.append(candidates.index(record["ground_truth"]))
        sample_ids.append(record["id"])
        tasks.append(record["task"])
        cursor += len(candidates)
        if (i + 1) % 25 == 0 or i + 1 == len(records):
            print(f"[extract] {i + 1}/{len(records)}", flush=True)

    metadata = {
        "encoder_id": runtime.encoder_id,
        "manifold_sha256": manifold_sha256,
        "calibration_file": str(CALIBRATION),
        "calibration_sha256": observed,
        "source": f"{CALIBRATION.name}:{CALIBRATION_SHA256[:12]}",
        "split": "calibration",
        "label_free_encoder_input": True,
        "records": len(records),
        "manifold_dim": int(z0s[0].shape[0]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as stream:
        np.savez(
            stream,
            metadata=json.dumps(metadata),
            z0=np.stack(z0s).astype(np.float32),
            candidates_flat=np.concatenate(candidates_flat, axis=0).astype(np.float32),
            offsets=np.asarray(offsets, dtype=np.int64),
            counts=np.asarray(counts, dtype=np.int64),
            positive_indices=np.asarray(positive_indices, dtype=np.int64),
            sample_ids=np.asarray(sample_ids, dtype="<U32"),
            tasks=np.asarray(tasks, dtype="<U32"),
        )
    print(f"[extract] wrote {len(records)} records -> {args.out}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
