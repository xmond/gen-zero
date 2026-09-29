#!/usr/bin/env python3
"""Single-core CPU forward latency for three Zero presets: Zero-Lite Dense, Zero-Lite Sparse
MoE (top-1), and Zero-Compact 220M (factorized embedding + sparse MoE).

Real forward passes on random token IDs, one model resident at a time (they total ~2.7GB
BF16 combined, so each is freed with del + gc.collect() before the next loads). No mocked
timings, no synthetic parameter counts: every number in the JSON output comes from either
torch's own tensor bookkeeping (parameter_count()/storage_bytes()) or a wall-clock
time.perf_counter() measurement around model(ids).
"""
from __future__ import annotations

import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

from gen_zero.model.zero_model import ZeroConfig, ZeroModel  # noqa: E402

RESULTS = REPO / "benchmarks" / "results"
BATCH_SIZE = 1
SEQ_LEN = 16
WARMUP_ITERS = 2
MEASURE_ITERS = 5


def rss_bytes() -> int:
    with open("/proc/self/status", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS not available")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary_ms(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, .9),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def bf16_supported() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return False
    return "avx512_bf16" in flags or "amx_bf16" in flags


def environment() -> dict:
    return {
        "hostname": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": __import__("os").cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
        "cpu_has_native_bf16": bf16_supported(),
    }


def bench_one(name: str, config: ZeroConfig, *, dtype=torch.bfloat16) -> dict:
    torch.manual_seed(0)
    rss_before = rss_bytes()
    t_load = time.perf_counter()
    model = ZeroModel(config, dtype=dtype).eval()
    load_seconds = time.perf_counter() - t_load
    rss_after_load = rss_bytes()

    torch.manual_seed(1)
    ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN))

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            out = model(ids)
    assert out.shape == (BATCH_SIZE, 64) and torch.isfinite(out.float()).all(), \
        f"{name}: non-finite or wrong-shaped forward output"

    per_batch_ms = []
    with torch.inference_mode():
        for _ in range(MEASURE_ITERS):
            t0 = time.perf_counter()
            model(ids)
            per_batch_ms.append((time.perf_counter() - t0) * 1000.0)
    per_sample_ms = [t / BATCH_SIZE for t in per_batch_ms]
    rss_after_run = rss_bytes()

    active = config.active_parameter_count()
    total = config.parameter_count()
    result = {
        "name": name,
        "config": {
            "vocab_size": config.vocab_size, "hidden_size": config.hidden_size,
            "embedding_dim": config.embedding_dim if config.factorized_embedding else None,
            "factorized_embedding": config.factorized_embedding,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "sparse_moe": config.sparse_moe,
            "sparse_layer_start": config.sparse_layer_start if config.sparse_moe else None,
            "num_experts": config.num_experts if config.sparse_moe else None,
            "experts_per_token": config.experts_per_token if config.sparse_moe else None,
            "expert_intermediate_size": config.effective_expert_intermediate_size if config.sparse_moe else None,
        },
        "active_parameter_count": active,
        "parameter_count": total,
        "storage_bytes_bf16": config.storage_bytes(),
        "storage_within_1gb": config.storage_bytes() <= 1_000_000_000,
        "active_within_230m": active <= 230_000_000,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQ_LEN,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "load_seconds": load_seconds,
        "forward_ms_per_batch": summary_ms(per_batch_ms),
        "forward_ms_per_sample": summary_ms(per_sample_ms),
        "throughput_samples_per_sec": BATCH_SIZE / (statistics.fmean(per_batch_ms) / 1000.0),
        "memory": {
            "rss_before_load_bytes": rss_before,
            "rss_after_load_bytes": rss_after_load,
            "rss_after_run_bytes": rss_after_run,
            "model_footprint_bytes": rss_after_load - rss_before,
        },
    }
    print(f"[{name}] active={active:,} total={total:,} bf16_bytes={config.storage_bytes():,} "
          f"mean={result['forward_ms_per_sample']['mean_ms']:.2f}ms/sample "
          f"throughput={result['throughput_samples_per_sec']:.2f} samples/s", flush=True)
    del model
    gc.collect()
    return result


def main() -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1:
        raise SystemExit(f"failed to pin to a single thread: torch.get_num_threads()={torch.get_num_threads()}")

    presets = [
        ("Zero-Lite Dense", ZeroConfig.zero_lite()),
        ("Zero-Lite Sparse MoE (Top-1)", ZeroConfig.zero_lite(sparse_moe=True)),
        ("Zero-Compact 220M", ZeroConfig.zero_compact_220m()),
    ]

    results = []
    for name, config in presets:
        results.append(bench_one(name, config))

    dense_ms = results[0]["forward_ms_per_sample"]["mean_ms"]
    for entry in results:
        entry["speedup_vs_zero_lite_dense"] = dense_ms / entry["forward_ms_per_sample"]["mean_ms"]

    report = {
        "description": "Real single-core CPU forward latency, BF16, random token inputs. "
                        "No synthetic timings: every ms figure is a wall-clock "
                        "time.perf_counter() measurement around a real ZeroModel forward pass.",
        "environment": environment(),
        "results": results,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "zero_cpu_latency_benchmark.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
