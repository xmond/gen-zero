#!/usr/bin/env python3
"""Fused INT8 GEMV probe (torch dynamic quantization, fbgemm) with the trunk's real shapes.

Question: on this CPU (AVX-512F, no VNNI/AMX), how fast is one decode step when
int8 weights are consumed directly by the kernel, versus Int8WeightOnlyLinear's
dequantize-to-fp32-scratch path (zero_trunk.py:125-127)? Synthetic random
weights, timing only. Records the quantization backend actually used.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
from torch import nn

H, I, NH, NKV, HD, L = 896, 4864, 14, 2, 64, 24
SHAPES = [(H, NH * HD), (H, NKV * HD), (H, NKV * HD), (NH * HD, H), (H, I), (H, I), (I, H)]

def bench(fn, reps=7, warm=2):
    for _ in range(warm): fn()
    t = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1e3)
    return {"p50_ms": float(np.median(t)), "min_ms": float(min(t)), "reps": reps}

def main():
    torch.set_num_threads(1)
    out = {"is_synthetic": True, "torch": torch.__version__, "loadavg_start": list(os.getloadavg()),
           "supported_engines": list(torch.backends.quantized.supported_engines),
           "engine": torch.backends.quantized.engine}
    torch.manual_seed(0)
    mods = [nn.Sequential(*[nn.Linear(i, o, bias=False) for (i, o) in SHAPES]) for _ in range(L)]
    xs = [torch.randn(1, i) for (i, o) in SHAPES]
    def run(stack):
        for layer in stack:
            for j, lin in enumerate(layer):
                lin(xs[j])
    with torch.inference_mode():
        out["fp32_nn_linear_1t"] = bench(lambda: run(mods))
    q = [torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8) for m in mods]
    out["quantized_module_repr"] = repr(q[0][0])
    with torch.inference_mode():
        out["fused_int8_dynamic_1t"] = bench(lambda: run(q))
        torch.set_num_threads(4)
        out["fused_int8_dynamic_4t"] = bench(lambda: run(q))
        torch.set_num_threads(1)
    params = L * sum(i * o for i, o in SHAPES)
    out["int8_bytes_per_step"] = params
    out["fused_int8_effective_GB_per_s_1t"] = params / (out["fused_int8_dynamic_1t"]["min_ms"] * 1e6)
    # numeric sanity: cosine of quantized vs fp32 output for one layer stack
    with torch.inference_mode():
        a = mods[0][4](xs[4]); b = q[0][4](xs[4])
    out["cosine_fp32_vs_int8_gate_proj"] = float(nn.functional.cosine_similarity(a, b).item())
    out["loadavg_end"] = list(os.getloadavg())
    print(json.dumps(out, indent=2)); return 0

if __name__ == "__main__":
    sys.exit(main())
