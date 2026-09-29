#!/usr/bin/env python3
"""Profile ``NanocoreAnchorBridge`` end-to-end: real 8192-d features -> 128-d
``nanocore_state`` MCP payload for ``nanocore_ask`` (zero.rs:2246-2360).

This is the real consumption chain, not a synthetic one: it loads the actual
fitted ``ManifoldAnchorDistiller`` artifact and actual extracted LLaMA-70B
BoolQ hidden representations, projects each row through the bridge exactly
as a live caller would before sending an MCP ``tools/call zero`` request,
and reports single-step wall-clock latency (mean, median, P99) over
``--iters`` repetitions, as two separate blocks: ``projection_ms`` (the
budget-gated projection step alone) and ``payload_generation_ms``
(projection + MCP payload assembly, reported for transparency, not gated).

Requires a version 2 bound artifact, source space metadata in the feature
store, and an independently supplied target core manifest.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

SUITE_DIR = Path(__file__).resolve().parent
BENCH_DIR = SUITE_DIR.parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from gen_zero.causal.nanocore_bridge import NanocoreAnchorBridge  # noqa: E402

DEFAULT_ARTIFACT = REPO_ROOT / "benchmarks" / "results" / "manifold" / "distilled_128d_llama70b_boolq.npz"
DEFAULT_FEATURES = Path("/ebs/data/extracted_features/llama70b/boolq.npz")
DEFAULT_ITERS = 100
LATENCY_BUDGET_MS = 1.0


def _stats_ms(samples_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "iters": int(arr.size),
    }


def load_feature_rows(features_path: Path, n_rows: int) -> np.ndarray:
    with np.load(str(features_path), allow_pickle=False) as data:
        block = np.asarray(data["test_full"], dtype=np.float64)
    if block.shape[0] < n_rows:
        raise ValueError(
            f"{features_path} has only {block.shape[0]} test rows, need at least {n_rows}"
        )
    with np.load(str(features_path), allow_pickle=False) as data:
        space = json.loads(str(data["space"]))
    return [{"values": row, "space": space} for row in block[:n_rows]]


def run_profile(artifact_path: Path, features_path: Path, iters: int, core_manifest: dict) -> Dict[str, Any]:
    if iters < 1:
        raise ValueError("iters must be positive")
    bridge = NanocoreAnchorBridge(artifact_path, core_manifest=core_manifest)
    rows = load_feature_rows(features_path, iters)

    # Warm up (import/JIT/cache effects) before the timed loop, real data only.
    bridge.project_to_nanocore_state(rows[0])

    projection_latencies_ms: List[float] = []
    for i in range(iters):
        row = rows[i]
        t0 = time.perf_counter()
        state = bridge.project_to_nanocore_state(row)
        projection_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if len(state) != 128 or not all(np.isfinite(v) for v in state):
            raise ValueError(f"iteration {i}: bridge produced an invalid nanocore_state")

    # payload_generation_ms includes the same projection step as projection_ms, plus
    # request validation and dict assembly; it does not include MCP transport or Rust
    # engine execution -- it measures projection + payload assembly only, not
    # end-to-end MCP round-trip. Timed in its own loop (separate warm-up, own samples)
    # so it never contaminates the projection_ms budget measurement above.
    bridge.generate_mcp_ask_payload(rows[0], bridge.space["domain_id"], ["proceed"])

    payload_latencies_ms: List[float] = []
    for i in range(iters):
        row = rows[i]
        t0 = time.perf_counter()
        payload = bridge.generate_mcp_ask_payload(row, bridge.space["domain_id"], ["proceed"])
        payload_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        state = payload["nanocore_state"]
        if len(state) != 128 or not all(np.isfinite(v) for v in state):
            raise ValueError(f"iteration {i}: bridge produced an invalid nanocore_state")

    projection_stats = _stats_ms(projection_latencies_ms)
    payload_stats = _stats_ms(payload_latencies_ms)
    return {
        "suite": "profile_nanocore_latency",
        "artifact": str(artifact_path),
        "features": str(features_path),
        "iters": iters,
        "loadavg": os.getloadavg(),
        "projection_ms": projection_stats,
        "payload_generation_ms": payload_stats,
        "budget_ms": LATENCY_BUDGET_MS,
        "within_budget": projection_stats["p99_ms"] < LATENCY_BUDGET_MS,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=str, default=str(DEFAULT_ARTIFACT),
                        help="path to a fitted ManifoldAnchorDistiller .npz artifact")
    parser.add_argument("--features", type=str, default=str(DEFAULT_FEATURES),
                        help="path to an .npz feature store with a 'test_full' block")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS,
                        help=f"number of single-step projections to time (default {DEFAULT_ITERS})")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="output directory override (default: benchmarks/results/manifold)")
    parser.add_argument("--core-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_profile(Path(args.artifact), Path(args.features), args.iters,
                         json.loads(args.core_manifest.read_text()))
    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "benchmarks" / "results" / "manifold"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "nanocore_latency_profile.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not report["within_budget"]:
        print(
            f"[FAIL] p99 projection latency {report['projection_ms']['p99_ms']:.4f}ms exceeds "
            f"the {LATENCY_BUDGET_MS}ms budget (loadavg={report['loadavg']})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
