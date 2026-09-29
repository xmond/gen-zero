#!/usr/bin/env python3
"""Bandwidth-wall probes behind docs/zero/15-cpu-compute-hardware-and-architecture-deep-survey.md.

Reproduces, with the trunk's real matrix shapes (Qwen2.5-0.5B: 24 layers x
{q,k,v,o,gate,up,down}), the per-token cost of one decode step under three
weight formats and splits the INT8 path into its two halves:

  dequant   int8 -> fp32 copy_ into the shared scratch (zero_trunk.py:125-127)
  gemv      fp32 (1,1,in) @ W^T on the freshly written scratch

Also measures: single-thread streaming copy bandwidth, mmap page-touch cost
(pages are warm in page cache; this is the minor-fault floor, not disk), and
the same GEMV at 4 threads. Synthetic weights (random), timing only; no model
file is read. Machine load is recorded so a reader can judge the noise.
"""
from __future__ import annotations

import json
import mmap
import os
import platform
import sys
import tempfile
import time

import numpy as np
import torch

H, I, NH, NKV, HD, L = 896, 4864, 14, 2, 64, 24
SHAPES = [(H, NH * HD), (H, NKV * HD), (H, NKV * HD), (NH * HD, H), (H, I), (H, I), (I, H)]  # (in, out)


def cpu_flags() -> dict:
    flags = ""
    for line in open("/proc/cpuinfo", encoding="utf-8"):
        if line.startswith("flags"):
            flags = line
            break
    return {k: (k in flags.split()) for k in ("avx512f", "avx512_vnni", "avx512_bf16", "amx_bf16", "amx_int8", "avx_vnni")}


def bench(fn, reps: int, warm: int = 2) -> dict:
    for _ in range(warm):
        fn()
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000.0)
    return {"p50_ms": float(np.median(t)), "min_ms": float(min(t)), "reps": reps}


def build(dtype: torch.dtype):
    g = torch.Generator().manual_seed(0)
    return [[torch.randn(o, i, generator=g).to(dtype) for (i, o) in SHAPES] for _ in range(L)]


def main() -> int:
    out = {"is_synthetic": True, "torch": torch.__version__, "python": platform.python_version(), "cpu": cpu_flags(),
           "loadavg_start": list(os.getloadavg()), "layers": L, "shapes_in_out": SHAPES}
    lin_params = L * sum(i * o for i, o in SHAPES)
    out["linear_params"] = lin_params
    out["bytes_per_step"] = {"int8": lin_params, "bf16": 2 * lin_params, "fp32": 4 * lin_params}
    torch.set_num_threads(1)
    reps = 5

    # --- streaming copy bandwidth, one thread
    src = torch.empty(256 << 20, dtype=torch.uint8).random_()
    dst = torch.empty_like(src)
    r = bench(lambda: dst.copy_(src), reps)
    out["memcpy_1thread"] = {**r, "GB_per_s": (2 * src.numel()) / (r["min_ms"] * 1e6)}

    # --- INT8 path split, exactly as Int8WeightOnlyLinear.forward does it
    w8 = build(torch.int8)
    scales = [[torch.rand(o) for (i, o) in SHAPES] for _ in range(L)]
    scratch = torch.empty(H * I, dtype=torch.float32)
    xs = [torch.randn(1, 1, i) for (i, o) in SHAPES]

    def int8_dequant_only():
        for layer in w8:
            for w in layer:
                scratch[: w.numel()].view_as(w).copy_(w)

    def int8_full_step():
        for layer, sc in zip(w8, scales):
            for j, w in enumerate(layer):
                d = scratch[: w.numel()].view_as(w)
                d.copy_(w)
                y = torch.nn.functional.linear(xs[j], d)
                y.mul_(sc[j])

    with torch.inference_mode():
        out["int8_dequant_only_1t"] = bench(int8_dequant_only, reps)
        out["int8_step_1t"] = bench(int8_full_step, reps)
    del w8, scales

    # --- fp32 / bf16 direct GEMV
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        ws = build(dtype)
        xd = [x.to(dtype) for x in xs]

        def step():
            for layer in ws:
                for j, w in enumerate(layer):
                    torch.nn.functional.linear(xd[j], w)

        with torch.inference_mode():
            r1 = bench(step, reps)
            torch.set_num_threads(4)
            r4 = bench(step, reps)
            torch.set_num_threads(1)
        nbytes = out["bytes_per_step"][name]
        out[f"{name}_step_1t"] = {**r1, "effective_GB_per_s": nbytes / (r1["min_ms"] * 1e6)}
        out[f"{name}_step_4t"] = {**r4, "effective_GB_per_s": nbytes / (r4["min_ms"] * 1e6)}
        del ws

    # --- mmap page-touch floor (warm page cache): the cost model for MoE paging
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(os.urandom(64 << 20))
        path = f.name
    try:
        fd = os.open(path, os.O_RDONLY)
        m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        pages = (64 << 20) // 4096
        s = time.perf_counter()
        acc = 0
        for p in range(pages):
            acc += m[p * 4096]
        t_first = time.perf_counter() - s
        s = time.perf_counter()
        for p in range(pages):
            acc += m[p * 4096]
        t_second = time.perf_counter() - s
        m.close(); os.close(fd)
        out["mmap_touch"] = {"pages": pages, "first_touch_us_per_page": t_first / pages * 1e6,
                             "second_touch_us_per_page": t_second / pages * 1e6, "sink": acc & 1,
                             "note": "file was just written so pages are in page cache; first touch = minor fault + python overhead"}
    finally:
        os.unlink(path)

    out["loadavg_end"] = list(os.getloadavg())
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
