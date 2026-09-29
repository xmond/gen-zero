#!/usr/bin/env python3
"""Single-core CPU cost probes behind docs/zero/10-trunk-unfreeze-and-latent-recurrence-engineering-spec.md.

  thought   cost of one cached single-token trunk step (the per-step price of a
            Coconut-style latent thought) versus a full prompt forward
  peak      VmHWM after one decide() on a record with K candidates taken from the
            open training pool (label is never read; only context + candidates)

Real INT8 runtime, real weights, no mock. Missing artifacts abort.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.zero_runtime import ZeroStandaloneRuntime, enforce_single_core  # noqa: E402

ART = REPO / "benchmarks" / "artifacts" / "zero"
POOL = REPO / "benchmarks" / "artifacts" / "verified_datasets" / "open_training_pool_5k.jsonl"


def vm_hwm_bytes() -> int:
    for line in open("/proc/self/status", encoding="utf-8"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM missing")


def load() -> ZeroStandaloneRuntime:
    enforce_single_core()
    return ZeroStandaloneRuntime(precision="int8", int8_artifact=ART / "zero_int8_v2.safetensors",
                                 manifold_path=ART / "zero_manifold_open_v1.npz",
                                 task_head_path=ART / "zero_task_head_open_v1.npz")


def probe_thought(rt: ZeroStandaloneRuntime, text: str, reps: int) -> dict:
    ids = rt.token_ids(text)

    def fwd(tokens, past=None, past_mask=None, cache=False):
        p = torch.tensor([tokens])
        m = torch.ones_like(p)
        t = time.perf_counter()
        with torch.inference_mode():
            _, c = rt.model(p, m, past=past, past_mask=past_mask, return_cache=cache)
        return (time.perf_counter() - t) * 1000.0, c

    for _ in range(2):
        fwd(ids)
    full = [fwd(ids)[0] for _ in range(reps)]
    _, cache = fwd(ids, cache=True)
    pm = torch.ones(1, len(ids), dtype=torch.long)
    one = [fwd([ids[-1]], past=cache, past_mask=pm)[0] for _ in range(reps)]
    chained = {}
    for k in (2, 4):
        t = time.perf_counter()
        cur, mask = cache, pm
        with torch.inference_mode():
            for _ in range(k):
                _, cur = rt.model(torch.tensor([[ids[-1]]]), torch.ones(1, 1, dtype=torch.long),
                                  past=cur, past_mask=mask, return_cache=True)
                mask = torch.ones(1, cur[0][0].shape[2], dtype=torch.long)
        chained[f"k{k}_chained_ms"] = (time.perf_counter() - t) * 1000.0
    return {"prompt_tokens": len(ids), "full_forward_ms_p50": float(np.median(full)),
            "full_forward_ms_min": float(min(full)), "single_token_step_ms_p50": float(np.median(one)),
            "single_token_step_ms_min": float(min(one)), **chained, "torch_threads": torch.get_num_threads()}


def probe_peak(rt: ZeroStandaloneRuntime, task: str) -> dict:
    record = next(json.loads(l) for l in open(POOL, encoding="utf-8") if json.loads(l)["task"] == task)
    before = vm_hwm_bytes()
    t = time.perf_counter()
    d = rt.decide(record["context"], record["candidates"])
    return {"task": task, "k": len(record["candidates"]), "prompt_tokens": d.prompt_tokens,
            "candidate_tokens": d.candidate_tokens, "decide_ms": (time.perf_counter() - t) * 1000.0,
            "vm_hwm_before_decide_bytes": before, "vm_hwm_after_decide_bytes": vm_hwm_bytes()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("thought", "peak"))
    ap.add_argument("--task", default="banking77")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    rt = load()
    out = {"mode": args.mode, "vm_hwm_after_load_bytes": vm_hwm_bytes(), "tensor_bytes": rt.tensor_bytes()}
    if args.mode == "thought":
        out.update(probe_thought(rt, "Which of the following best explains why the sky appears blue "
                                     "during the day on Earth?", args.reps))
    else:
        out.update(probe_peak(rt, args.task))
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized; not a CPU-only run")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
