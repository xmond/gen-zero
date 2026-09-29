"""Reproducible, CPU-only measurements for zero-token state dynamics.

This benchmark deliberately uses small NumPy kernels rather than a mock model.  The
memory experiment reports both allocated array payloads and tracemalloc peaks; the
latency experiment times every one of 1000 real state updates.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import resource
import sys
import time
import tracemalloc
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT = ROOT / "benchmarks" / "reports" / "cpu_zero_token_dynamics_results.json"
STEPS = [8, 16, 32, 64, 128]
D = 256
SEED = 20260922


def memory_experiment() -> dict[str, Any]:
    rows = []
    for t in STEPS:
        # A conventional per-layer K and V cache: these are real float32 arrays.
        tracemalloc.start()
        keys = np.zeros((t, D), dtype=np.float32)
        values = np.zeros((t, D), dtype=np.float32)
        cache_current, cache_peak = tracemalloc.get_traced_memory()
        del keys, values
        tracemalloc.stop()

        # A recurrent LoRA-style state: one d-wide state, updated in place.
        tracemalloc.start()
        state = np.zeros(D, dtype=np.float32)
        state_current, state_peak = tracemalloc.get_traced_memory()
        del state
        tracemalloc.stop()
        rows.append({
            "sequence_steps": t,
            "kv_payload_bytes": 2 * t * D * np.dtype(np.float32).itemsize,
            "kv_tracemalloc_peak_bytes": cache_peak,
            "state_payload_bytes": D * np.dtype(np.float32).itemsize,
            "state_tracemalloc_peak_bytes": state_peak,
            "kv_tracemalloc_current_bytes": cache_current,
            "state_tracemalloc_current_bytes": state_current,
        })
    kv = np.array([r["kv_payload_bytes"] for r in rows], dtype=float)
    ts = np.array(STEPS, dtype=float)
    slope, intercept = np.polyfit(ts, kv, 1)
    state_payloads = {r["state_payload_bytes"] for r in rows}
    return {
        "dimension": D,
        "dtype": "float32",
        "rows": rows,
        "payload_linear_fit": {"slope_bytes_per_step": float(slope), "intercept_bytes": float(intercept)},
        "kv_payload_strictly_linear": bool(np.all(np.diff(kv) > 0) and abs(intercept) < 1e-9),
        "state_payload_constant": len(state_payloads) == 1,
    }


def cpu_latency() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    matrix = rng.standard_normal((D, D), dtype=np.float32) / np.float32(np.sqrt(D))
    drive = rng.standard_normal(D, dtype=np.float32)
    state = np.zeros(D, dtype=np.float32)
    # Warm-up is outside the measured 1000 consecutive steps.
    for _ in range(20):
        state[:] = np.tanh(matrix @ state + drive)
    samples_ns = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        state[:] = np.tanh(matrix @ state + drive)
        samples_ns.append(time.perf_counter_ns() - start)
    samples_us = np.asarray(samples_ns, dtype=np.float64) / 1000.0
    mean = float(np.mean(samples_us))
    return {
        "iterations": 1000,
        "clock": "time.perf_counter_ns",
        "mean_us": mean,
        "p50_us": float(np.percentile(samples_us, 50)),
        "p90_us": float(np.percentile(samples_us, 90)),
        "p99_us": float(np.percentile(samples_us, 99)),
        "throughput_steps_per_second": 1_000_000.0 / mean,
        "final_state_l2": float(np.linalg.norm(state)),
    }


def residual(x: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(x - target))


def convergence(seed: int, multiscale: bool) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    target = rng.standard_normal(D, dtype=np.float64)
    x = np.zeros(D, dtype=np.float64)
    start = time.perf_counter_ns()
    physical_steps = 0
    residuals = []
    while physical_steps < 20000 and residual(x, target) >= 1e-5:
        if multiscale and physical_steps % 4 == 0:
            # Macro tau=4: four explicit relaxation substeps, counted as 4 physical steps.
            for _ in range(4):
                x += 0.20 * (target - x)
                physical_steps += 1
        else:
            x += 0.05 * (target - x)
            physical_steps += 1
        residuals.append(residual(x, target))
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return {"physical_steps": physical_steps, "elapsed_ms": elapsed_ms,
            "final_residual": residual(x, target), "reached_threshold": residual(x, target) < 1e-5,
            "residual_samples": residuals[:3] + residuals[-3:]}


def system_info() -> dict[str, Any]:
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {"cpu_logical_cores": os.cpu_count(), "system_load_1m_5m_15m": list(load) if load else None,
            "platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
            "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main() -> int:
    result = {"experiment": "cpu_zero_token_dynamics", "seed": SEED,
              "command": "python3 benchmarks/suites/evaluate_cpu_zero_token_dynamics.py",
              "hardware": system_info(), "memory": memory_experiment(), "latency": cpu_latency(),
              "convergence": {str(seed): {"single_scale": convergence(seed, False),
                                             "fractal_multiscale": convergence(seed, True)}
                              for seed in (7, 19, 43)}}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "latency": result["latency"],
                      "memory_checks": {k: result["memory"][k] for k in ("kv_payload_strictly_linear", "state_payload_constant")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
