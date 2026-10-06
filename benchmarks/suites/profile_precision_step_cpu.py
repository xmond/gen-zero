#!/usr/bin/env python3
"""CPU cost of one cached single-token trunk step per weight precision and thread count.

Behind docs/zero/11-unbound-memory-and-multistep-reasoning-cpu-spec.md.

  step      p50 of a full prompt forward, one cached single-token step, and
            chained K in {2,4,8} steps, for --precision in {int8,bf16,fp32}
            and --threads N. Also reports VmHWM after load and tensor bytes.
  fidelity  last-token cosine of --precision against fp32 on a few prompts
            (both trunks loaded in one process; memory is not the point here).

Real weights, real runtime, no mock. Missing artifacts abort. CUDA must stay
uninitialized. Thread count is set before any torch op and reported back.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

ART = REPO / "benchmarks" / "artifacts" / "zero"
PROMPTS = [
    "Which of the following best explains why the sky appears blue during the day on Earth?",
    "If an object's mass doubles while the force stays constant, how does its acceleration change?",
    "def f(xs):\n    return sorted(xs)[-1]\nprint(f([3, 9, 2]))",
]


def vm_rss_bytes() -> int:
    for line in open("/proc/self/status", encoding="utf-8"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS missing")


def loadavg() -> list:
    """1/5/15-minute load average: timing on a loaded box is not a quiet-box number."""
    return [float(v) for v in open("/proc/loadavg", encoding="utf-8").read().split()[:3]]


def vm_hwm_bytes() -> int:
    for line in open("/proc/self/status", encoding="utf-8"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM missing")


def cpu_flags() -> dict:
    flags = ""
    for line in open("/proc/cpuinfo", encoding="utf-8"):
        if line.startswith("flags"):
            flags = line.split(":", 1)[1]
            break
    toks = set(flags.split())
    return {"nproc": os.cpu_count(),
            "avx512f": "avx512f" in toks, "avx512_bf16": "avx512_bf16" in toks,
            "avx512_vnni": "avx512_vnni" in toks, "amx_bf16": "amx_bf16" in toks,
            "amx_int8": "amx_int8" in toks}


def set_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    import torch
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load(precision: str):
    from gen_zero.causal.zero_runtime import ZeroStandaloneRuntime
    kw = {"int8_artifact": ART / "zero_int8_v2.safetensors"} if precision == "int8" else {}
    t = time.perf_counter()
    rt = ZeroStandaloneRuntime(precision=precision, single_core=False, **kw)
    return rt, (time.perf_counter() - t) * 1000.0


def probe_step(rt, text: str, reps: int) -> dict:
    import torch
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
    for k in (2, 4, 8):
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
            "single_token_step_ms_min": float(min(one)), **chained}


def probe_fidelity(precision: str) -> dict:
    ref, _ = load("fp32")
    other, _ = load(precision)
    cos = []
    for p in PROMPTS:
        a, _ = ref.encode([p])
        b, _ = other.encode([p])
        a, b = a[0].astype(np.float64), b[0].astype(np.float64)
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return {"precision": precision, "cosine_vs_fp32": cos, "cosine_min": min(cos)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("step", "fidelity"))
    ap.add_argument("--precision", default="int8", choices=("int8", "bf16", "fp32"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    set_threads(args.threads)
    import torch
    out = {"mode": args.mode, "precision": args.precision, "torch_threads": torch.get_num_threads(),
           "torch_version": torch.__version__, "cpu": cpu_flags(), "python": platform.python_version(), "loadavg_start": loadavg()}
    if args.mode == "step":
        rt, load_ms = load(args.precision)
        out.update({"load_ms": load_ms, "vm_hwm_after_load_bytes": vm_hwm_bytes(),
                    "vm_rss_after_load_bytes": vm_rss_bytes(),
                    "tensor_bytes": rt.tensor_bytes()})
        out.update(probe_step(rt, PROMPTS[0], args.reps))
        out["vm_hwm_after_probe_bytes"] = vm_hwm_bytes()
        out["vm_rss_after_probe_bytes"] = vm_rss_bytes()
        out["loadavg_end"] = loadavg()
    else:
        out.update(probe_fidelity(args.precision))
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized; not a CPU-only run")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
