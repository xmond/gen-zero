#!/usr/bin/env python3
"""Cumulative PCA energy of real 896-D last-token states at 64/128/256 dims.

Behind docs/zero/11-unbound-memory-and-multistep-reasoning-cpu-spec.md section 1.
Encodes the first --n contexts of the open training pool (labels never read),
centers them, runs an SVD and reports the energy fraction kept by the top-d
principal directions. Real INT8 trunk, no mock, CUDA must stay uninitialized.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
ART = REPO / "benchmarks" / "artifacts" / "zero"
POOL = REPO / "benchmarks" / "artifacts" / "verified_datasets" / "open_training_pool_5k.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    from gen_zero.causal.zero_runtime import ZeroStandaloneRuntime
    rt = ZeroStandaloneRuntime(precision="int8", int8_artifact=ART / "zero_int8_v2.safetensors",
                               single_core=False)
    texts, tasks, skipped = [], [], 0
    per_task: dict = {}
    for line in open(POOL, encoding="utf-8"):
        rec = json.loads(line)
        if per_task.get(rec["task"], 0) >= args.n // 5:
            continue
        try:
            n_tok = len(rt.token_ids(rec["context"]))
        except ValueError:  # over max_length; the runtime refuses to truncate
            n_tok = args.max_tokens + 1
        if n_tok > args.max_tokens:
            skipped += 1
            continue
        per_task[rec["task"]] = per_task.get(rec["task"], 0) + 1
        texts.append(rec["context"])
        tasks.append(rec["task"])
        if len(texts) >= args.n:
            break
    t0 = time.perf_counter()
    states = []
    for i in range(0, len(texts), args.batch):
        s, _ = rt.encode(texts[i:i + args.batch])
        states.append(s)
    x = np.concatenate(states).astype(np.float64)
    elapsed = time.perf_counter() - t0
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eig = singular ** 2
    cum = np.cumsum(eig) / eig.sum()
    out = {"n_states": int(x.shape[0]), "hidden": int(x.shape[1]), "per_task": per_task,
           "skipped_over_max_tokens": skipped, "encode_seconds": elapsed,
           "torch_threads": torch.get_num_threads(),
           "energy_kept": {str(d): float(cum[d - 1]) for d in (32, 64, 128, 256, 512, 896)},
           "dims_for_energy": {str(t): int(np.searchsorted(cum, t) + 1) for t in (0.90, 0.95, 0.99, 0.999)}}
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized; not a CPU-only run")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
