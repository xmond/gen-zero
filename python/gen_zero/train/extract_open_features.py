#!/usr/bin/env python3
"""Extract Zero's label-free 64-D manifold states and features for open training datasets.

Parallel, multi-process feature extraction across CPU cores.
1. Reads clean verified training records (ARC, MMLU-Pro, APPS, Banking77, etc.);
2. Dispatches encoding across multiple worker processes using ZeroStandaloneRuntime (INT8/BF16/FP32);
3. Fits a label-free ZCA manifold; test-split inputs in the pool make this
   transductive for those sources, which downstream reports must disclose;
4. Projects prompt states and candidate states to the 64-D unit sphere;
5. Saves CSR-packed projected features and real 896-D hidden states. Supervised
   trainers must honor each source record's supervised_training_eligible flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.zero_runtime import (  # noqa: E402
    MANIFOLD_DIM,
    ZeroManifold,
    ZeroStandaloneRuntime,
    build_int8_artifact,
    find_local_snapshot,
)

ARTIFACTS = REPO / "benchmarks" / "artifacts" / "zero"
INT8_ARTIFACT = ARTIFACTS / "zero_int8_v2.safetensors"


def _worker_init(precision: str, int8_path: Optional[Path], max_length: int):
    global _RUNTIME
    torch.set_num_threads(1)
    _RUNTIME = ZeroStandaloneRuntime(
        precision=precision,
        int8_artifact=int8_path,
        max_length=max_length,
    )


def _encode_single(record: dict) -> Tuple[str, str, int, int, np.ndarray, np.ndarray]:
    global _RUNTIME
    candidates = list(record["candidates"])
    gt = record["ground_truth"]
    if gt not in candidates:
        raise ValueError(f"{record['id']}: ground_truth not in candidates")
    pos_idx = candidates.index(gt)
    q0, c_states, _ = _RUNTIME.encode_prompt_with_candidates(record["context"], candidates)
    return (record["id"], record["task"], pos_idx, len(candidates), q0, c_states)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=REPO / "benchmarks" / "artifacts" / "verified_datasets"
                        / "open_training_pool_natural_5k.jsonl")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to encode (default: no cap, extract the full dataset)")
    parser.add_argument("--max-length", type=int, default=8192, help="Maximum full prompt plus candidate token length; overflow fails the run")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes")
    parser.add_argument("--precision", choices=("int8", "bf16", "fp32"), default="int8")
    parser.add_argument("--fit-manifold", action="store_true", default=True, help="Fit a new manifold from train states")
    parser.add_argument("--out-manifold", type=Path, default=ARTIFACTS / "zero_manifold_open_v1.npz")
    parser.add_argument("--out-features", type=Path, default=ARTIFACTS / "zero_open_features_v1.npz")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.precision == "int8" and not INT8_ARTIFACT.exists():
        print(f"[extract] Building INT8 artifact -> {INT8_ARTIFACT}...", flush=True)
        build_int8_artifact(find_local_snapshot(), INT8_ARTIFACT)

    print(f"[extract] Loading dataset from {args.dataset}...", flush=True)
    records = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    filtered = records[:args.max_samples] if args.max_samples else records

    total_n = len(filtered)
    print(f"[extract] Dispatching {total_n} records across {args.workers} worker processes...", flush=True)
    t0 = time.time()
    int8_arg = INT8_ARTIFACT if args.precision == "int8" else None
    valid_results = []
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=(args.precision, int8_arg, args.max_length)) as pool:
        for idx, res in enumerate(pool.imap(_encode_single, filtered, chunksize=4)):
            valid_results.append(res)
            if (idx + 1) % 50 == 0 or (idx + 1) == total_n:
                now = time.time() - t0
                rate = (idx + 1) / max(now, 1e-3)
                eta = (total_n - (idx + 1)) / max(rate, 1e-3)
                print(f"[extract] Processed {idx + 1}/{total_n} records ({len(valid_results)} valid) in {now:.1f}s "
                      f"({rate:.1f} rec/s, ETA: {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"[extract] Successfully encoded {len(valid_results)}/{total_n} records in {elapsed:.1f}s "
          f"({len(valid_results)/elapsed:.1f} records/s)", flush=True)

    if len(valid_results) != total_n:
        raise RuntimeError(f"incomplete extraction: {len(valid_results)}/{total_n}")

    # Collect all hidden states for unsupervised manifold fitting
    all_states = []
    for _, _, _, _, q0, c_states in valid_results:
        all_states.append(q0)
        all_states.extend(c_states)
    matrix = np.stack(all_states)
    print(f"[extract] Collected {len(all_states)} total hidden states (matrix shape: {matrix.shape})", flush=True)

    # Step 1: Fit or load manifold
    snapshot = find_local_snapshot()
    encoder_id = f"zero-qwen2.5-0.5b-trunk:{args.precision}:last-token"
    if args.fit_manifold:
        print(f"[extract] Fitting new 64-D ZCA manifold from open train states...", flush=True)
        dataset_name = args.dataset.name
        manifold = ZeroManifold.fit(
            matrix,
            dim=MANIFOLD_DIM,
            encoder_id=encoder_id,
            source=f"open_train:{dataset_name}",
            split="train",
        )
        args.out_manifold.parent.mkdir(parents=True, exist_ok=True)
        manifold.save(args.out_manifold)
        print(f"[extract] Saved new open train manifold -> {args.out_manifold} "
              f"(energy kept: {manifold.provenance['energy_kept']:.3%})", flush=True)
    else:
        manifold = ZeroManifold.load(args.out_manifold)

    manifold_sha256 = hashlib.sha256(args.out_manifold.read_bytes()).hexdigest()

    # Step 2: Project states to 64-D unit sphere vectors
    print("[extract] Projecting hidden states onto 64-D unit sphere...", flush=True)
    z0s, candidates_flat, raw_h, raw_candidates_h, offsets, counts, positive_indices = [], [], [], [], [], [], []
    sample_ids, tasks = [], []
    cursor = 0
    for sample_id, task, pos_idx, n_cands, q0, c_states in valid_results:
        z0 = manifold.project(q0)
        zc = manifold.project(c_states)
        z0s.append(z0)
        candidates_flat.append(zc)
        raw_h.append(q0)
        raw_candidates_h.append(c_states)
        offsets.append(cursor)
        counts.append(n_cands)
        positive_indices.append(pos_idx)
        sample_ids.append(sample_id)
        tasks.append(task)
        cursor += n_cands

    metadata = {
        "encoder_id": encoder_id,
        "manifold_sha256": manifold_sha256,
        "source": f"open_train:{args.dataset.name}",
        "split": "train",
        "label_free_encoder_input": True,
        "records": len(valid_results),
        "manifold_dim": MANIFOLD_DIM,
    }

    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_features, "wb") as stream:
        np.savez(
            stream,
            metadata=json.dumps(metadata),
            z0=np.stack(z0s).astype(np.float32),
            candidates_flat=np.concatenate(candidates_flat, axis=0).astype(np.float32),
            h=np.stack(raw_h).astype(np.float32),
            candidates_h=np.concatenate(raw_candidates_h, axis=0).astype(np.float32),
            offsets=np.asarray(offsets, dtype=np.int64),
            counts=np.asarray(counts, dtype=np.int64),
            positive_indices=np.asarray(positive_indices, dtype=np.int64),
            sample_ids=np.asarray(sample_ids, dtype="<U32"),
            tasks=np.asarray(tasks, dtype="<U32"),
        )

    print(f"[extract] Successfully wrote {len(valid_results)} records -> {args.out_features}", flush=True)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
