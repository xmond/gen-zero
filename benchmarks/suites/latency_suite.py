"""Latency and Throughput Benchmark Suite for Gen-Zero.

Measures (all numbers come from timed `gen-zero reflex` subprocess calls):
- CLI reflex decision latency (Mean, P50, P90, P95, P99 in microseconds, includes process spawn)
- Sequential CLI throughput (decisions / second, single caller, no concurrency)

Anything this suite does not time (in-engine reflex latency, MCTS latency, accuracy, ECE)
is reported as "not_measured". No constant is substituted for a missing measurement.
Competitor metrics require measured evidence; no static leaderboard rows are emitted.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


NOT_MEASURED = "not_measured"


class LatencyBenchmarkSuite:
    """Evaluates sub-millisecond execution latencies and throughput."""

    def __init__(self, binary_path: str = "./target/release/gen-zero", iterations: int = 100):
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self.binary_path = Path(binary_path)
        self.iterations = iterations

    def _reflex(self, context: str) -> None:
        """One CLI reflex call. A refused or failed call is not a decision, so it raises instead of being timed."""
        res = subprocess.run(
            [str(self.binary_path), "reflex", "--context", context],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"reflex call exited {res.returncode}, refusing to time a non-decision: "
                f"stderr={res.stderr.strip()[:200]!r} stdout_head={res.stdout.strip()[:300]!r}"
            )

        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("reflex response is not JSON; refusing to time a non-decision") from exc
        meta = data.get("meta") if isinstance(data, dict) else None
        chosen = meta.get("chosen_action") if isinstance(meta, dict) else None
        if (not isinstance(chosen, str) or not chosen.strip()
                or data.get("is_error") or chosen == "ABSTAIN"):
            raise RuntimeError("reflex response has no successful decision; refusing to time a non-decision")

    def run(self) -> Dict[str, Any]:
        print("  Running System 1 Reflex Latency & Throughput Benchmark...")
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"gen-zero binary not found: {self.binary_path}; build it, no numbers are invented")
        latencies_us: List[float] = []

        # Warmup
        for _ in range(5):
            self._reflex("User wants to query account balance")

        # Benchmark runs
        start_total = time.perf_counter()
        for i in range(self.iterations):
            context = f"Autonomous navigation step {i}: obstacle at 45 degrees, speed 12m/s"
            t0 = time.perf_counter()
            self._reflex(context)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1_000_000.0)

        total_duration = time.perf_counter() - start_total
        arr = np.array(latencies_us)

        mean_us = float(np.mean(arr))
        p50_us = float(np.percentile(arr, 50))
        p90_us = float(np.percentile(arr, 90))
        p95_us = float(np.percentile(arr, 95))
        p99_us = float(np.percentile(arr, 99))
        throughput_qps = self.iterations / total_duration

        # Gen-Zero row: only what this run timed. CLI latency includes fork/exec, so it is
        # an upper bound on in-engine latency, never a stand-in for it.
        gen_zero_row = {
            "model": "Gen-Zero (Rust CLI, measured)",
            "throughput_qps": round(throughput_qps, 2),
            "latency_ms": round(mean_us / 1000.0, 3),
            "acc": NOT_MEASURED,
            "ece": NOT_MEASURED,
            "size": f"{self.binary_path.stat().st_size / 1e6:.1f}MB binary",
            "provenance": "measured_this_run_cli_subprocess",
        }
        competitors = [gen_zero_row]

        return {
            "is_synthetic": True,
            "measurement_scope": "real CLI timings on generated contexts; no task accuracy measured",
            "iterations": self.iterations,
            "mean_latency_us": round(mean_us, 2),
            "p50_latency_us": round(p50_us, 2),
            "p90_latency_us": round(p90_us, 2),
            "p95_latency_us": round(p95_us, 2),
            "p99_latency_us": round(p99_us, 2),
            "cli_throughput_qps": round(throughput_qps, 2),
            "in_engine_reflex_latency_us": NOT_MEASURED,
            "in_engine_mcts_latency_ms": NOT_MEASURED,
            "competitors_comparison": competitors,
        }
