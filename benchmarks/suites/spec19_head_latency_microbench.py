"""CPU latency microbenchmark for the Spec 19 head candidates (docs/zero/19-*.md).

Measures batch-1 decision latency of each head form on random float32 inputs,
plus the per-step training cost of the residual adapter. No dataset, no label,
no accuracy: this file only backs the latency budget table in Spec 19 §6.3.
Random weights have the same cost as trained ones for dense kernels.

Gotcha: the host is shared. loadavg is recorded at start and end; distrust the
multi-thread rows when load is high.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPS = 2000
WARM = 200


def timeit(fn, reps: int = REPS) -> dict:
    for _ in range(WARM):
        fn()
    ts = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        fn()
        ts[i] = (time.perf_counter() - t0) * 1e6
    return {"median_us": round(float(np.median(ts)), 2), "p95_us": round(float(np.percentile(ts, 95)), 2)}


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def bench_heads(D: int, K: int, rng) -> dict:
    f32 = np.float32
    x = rng.standard_normal(D).astype(f32)
    W = rng.standard_normal((K, D)).astype(f32) * f32(0.01)
    b = np.zeros(K, f32)
    r = 512
    Wd = rng.standard_normal((r, D)).astype(f32) * f32(0.01)
    Wu = rng.standard_normal((D, r)).astype(f32) * f32(0.01)
    M = 16                                           # heads in the bank
    bank = rng.standard_normal((M * K, D)).astype(f32) * f32(0.01)
    m, rk = 1024, 256                                # Nystrom landmarks, rank
    Z = rng.standard_normal((m, D)).astype(f32)
    zn = (Z * Z).sum(1)
    P = rng.standard_normal((m, rk)).astype(f32) * f32(0.01)
    Wk = rng.standard_normal((K, rk)).astype(f32) * f32(0.01)
    gamma = f32(1.0 / D)
    Pw = rng.standard_normal((D, 64)).astype(f32) * f32(0.01)   # projection to the Poincare ball
    protos = rng.standard_normal((K, 64)).astype(f32)
    protos *= f32(0.5) / np.linalg.norm(protos, axis=1, keepdims=True)   # inside the unit ball
    x_ln = rng.standard_normal(D).astype(f32)                  # deep layer, for the layer differential
    Wdiff = rng.standard_normal((K, 3 * D)).astype(f32) * f32(0.01)

    def linear():
        return x @ W.T + b

    def adapter_linear():
        h = x + Wu @ gelu(Wd @ x)
        return h @ W.T + b

    WWu = W @ Wu                                     # (K, r): head folded into the up-projection

    def adapter_folded():
        # logits = W(x + Wu g) + b = Wx + (W Wu) g + b, g = GELU(Wd x); exact, not an approximation
        return x @ W.T + WWu @ gelu(Wd @ x) + b

    def bank_gemv():
        return bank @ x

    def nystrom():
        k = np.exp(-gamma * (zn - 2.0 * (Z @ x) + x @ x))
        return (k @ P) @ Wk.T

    def poincare():
        v = x @ Pw
        n = np.linalg.norm(v)
        u = np.tanh(n) * v / (n + 1e-9)             # exp map at the origin, radius < 1
        diff = ((protos - u) ** 2).sum(1)
        den = (1 - (u * u).sum()) * (1 - (protos * protos).sum(1))
        return -np.arccosh(1 + 2 * diff / den)

    def layer_diff():
        f = np.concatenate([x, x_ln, x_ln - x])
        return f @ Wdiff.T

    fold_err = float(np.max(np.abs(adapter_linear() - adapter_folded())))
    out = {name: timeit(fn) for name, fn in (
        ("linear_probe", linear), ("residual_adapter_r512_plus_linear", adapter_linear),
        ("residual_adapter_folded_into_head", adapter_folded),
        ("bank_gemv_16_heads", bank_gemv), ("nystrom_m1024_r256", nystrom),
        ("poincare_64d_protos", poincare), ("layer_diff_linear_3D", layer_diff))}
    out["fold_max_abs_err"] = fold_err
    for rr in (64, 128):                             # the rank sweep Spec 19 §2.5 recommends
        Wd_r = rng.standard_normal((rr, D)).astype(f32) * f32(0.01)
        WWu_r = rng.standard_normal((K, rr)).astype(f32) * f32(0.01)
        out[f"residual_adapter_folded_r{rr}"] = timeit(lambda: x @ W.T + WWu_r @ gelu(Wd_r @ x) + b)
    return out


def bench_adapter_training(D: int, N_batch: int = 256) -> dict:
    import torch
    torch.manual_seed(0)
    r = 512
    down, up = torch.nn.Linear(D, r), torch.nn.Linear(r, D)
    torch.nn.init.zeros_(up.weight)
    torch.nn.init.zeros_(up.bias)
    head = torch.nn.Linear(D, 18)
    params = list(down.parameters()) + list(up.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    x = torch.randn(N_batch, D)
    y = torch.randint(0, 18, (N_batch,))
    # Zero-init check: the adapter is the identity map before the first step.
    with torch.no_grad():
        ident_err = float((x + up(torch.nn.functional.gelu(down(x))) - x).abs().max())

    def step(dtype):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == "bf16"):
            h = x + up(torch.nn.functional.gelu(down(x)))
            loss = torch.nn.functional.cross_entropy(head(h).float(), y)
        loss.backward()
        opt.step()

    out = {"identity_max_abs_err_at_init": ident_err, "batch": N_batch}
    for dtype in ("fp32", "bf16"):
        for _ in range(3):
            step(dtype)
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            step(dtype)
        out[f"{dtype}_ms_per_step"] = round((time.perf_counter() - t0) * 1e3 / n, 2)
    return out


def main() -> None:
    threads = os.environ.get("OMP_NUM_THREADS")
    la0 = os.getloadavg()
    rng = np.random.default_rng(0)
    res = {"host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "loadavg_start": la0,
                    "omp_threads": threads, "python": sys.version.split()[0], "numpy": np.__version__},
           "note": "random weights; latency only; batch 1; float32", "heads": {}}
    for D in (896 * 2, 5120, 8192):
        res["heads"][f"D{D}_K18"] = bench_heads(D, 18, rng)
    import torch
    res["host"]["torch"] = torch.__version__
    res["host"]["torch_threads"] = torch.get_num_threads()
    res["adapter_training"] = {f"D{D}": bench_adapter_training(D) for D in (5120, 8192)}
    res["host"]["loadavg_end"] = os.getloadavg()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("spec19_head_latency_microbench.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
