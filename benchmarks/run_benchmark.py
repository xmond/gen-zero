#!/usr/bin/env python3
"""Gen-Zero CLI latency and candidate-order benchmark runner.

Executes 2 benchmark dimensions:
1. CLI reflex latency & sequential throughput (microseconds & decisions/s, measured)
2. Candidate-order flip rate (empirical count; error samples are reported, never hidden)

Unmeasured fields (in-engine latency, MCTS latency, accuracy, ECE) are "not_measured".

The former safety and cross-domain accuracy suites were deprecated and isolated on
2026-09-24 (see benchmarks/deprecated_unverified/README.md). They are not run here.

Outputs:
- benchmarks/results/latest_report.json
- benchmarks/results/latest_report.md
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure paths
BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCHMARK_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from suites.latency_suite import LatencyBenchmarkSuite
from suites.equivariance_suite import PermutationEquivarianceSuite
from suites.error_analysis import ErrorAnalysisSuite


def main():
    parser = argparse.ArgumentParser(description="Measure Rust CLI reflex latency and candidate-order flips on generated workloads")
    parser.add_argument("--binary", default=str(ROOT_DIR / "target" / "release" / "gen-zero"), help="Path to gen-zero release binary (default: repository target/release/gen-zero)")
    parser.add_argument("--latency-runs", type=int, default=50, help="Number of latency test iterations")
    parser.add_argument("--equivariance-trials", type=int, default=30, help="Number of candidate shuffle trials")
    parser.add_argument("--output-dir", default=str(BENCHMARK_DIR / "results"), help="Directory for latest_report.json and latest_report.md (default: benchmarks/results)")
    parser.add_argument("--error-analysis", action="store_true", help="Analyze real predictions supplied with --predictions")
    parser.add_argument("--predictions", type=Path, help="Path to real predictions JSONL outside the repository (requires --error-analysis)")
    args = parser.parse_args()
    if args.error_analysis and args.predictions is None:
        parser.error("--error-analysis requires --predictions")
    if args.predictions is not None and not args.error_analysis:
        parser.error("--predictions requires --error-analysis")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("           GEN-ZERO COMPREHENSIVE INDUSTRIAL BENCHMARK SUITE             ")
    print("==========================================================================")
    print(f" Timestamp: {datetime.datetime.now().isoformat()}")
    print(f" Target Binary: {args.binary}")
    print(f" Output Directory: {output_path}")
    print("==========================================================================\n")

    t_start = time.time()

    # 1. Latency & Throughput
    latency_suite = LatencyBenchmarkSuite(binary_path=args.binary, iterations=args.latency_runs)
    latency_results = latency_suite.run()

    # 2. Permutation Equivariance
    equiv_suite = PermutationEquivarianceSuite(binary_path=args.binary, num_trials=args.equivariance_trials)
    equiv_results = equiv_suite.run()

    # 3. Optional Deep Error Diagnostics & Taxonomy
    error_results = None
    if args.error_analysis:
        error_suite = ErrorAnalysisSuite(predictions_path=args.predictions)
        error_results = error_suite.run(output_md=output_path / "error_analysis.md")

    total_time = time.time() - t_start

    binary = Path(args.binary)
    full_results = {
        "is_synthetic": True,
        "measurement_scope": "real CLI measurements on generated workloads",
        "timestamp": datetime.datetime.now().isoformat(),
        "provenance": {
            "repo_head": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, stdout=subprocess.PIPE, text=True, check=True
            ).stdout.strip(),
            "binary_path": str(binary),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "binary_mtime": datetime.datetime.fromtimestamp(binary.stat().st_mtime).isoformat(),
            "loadavg_1m_5m_15m": os.getloadavg(),
        },
        "total_elapsed_seconds": round(total_time, 2),
        "latency_throughput": latency_results,
        "permutation_equivariance": equiv_results,
    }

    # Save JSON report
    json_file = output_path / "latest_report.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Raw metrics saved to: {json_file}")

    # Generate Markdown report
    md_file = output_path / "latest_report.md"
    generate_markdown_report(full_results, md_file)
    print(f"[+] Formatted report generated at: {md_file}")
    print("\n==========================================================================")
    print("                       BENCHMARK RUN COMPLETED                            ")
    print("==========================================================================")

    n_err = equiv_results["error_samples"]
    if n_err:
        print(f"[!] FAIL-CLOSED: {n_err} equivariance error samples (see error_details in the report); exiting 2", file=sys.stderr)
        sys.exit(2)


def generate_markdown_report(data: dict, out_file: Path):
    lat = data["latency_throughput"]
    eq = data["permutation_equivariance"]

    md = f"""# Gen-Zero Comprehensive Benchmark & Competitive Analysis Report

> **Execution Timestamp**: `{data["timestamp"]}`  
> **Binary**: `{data["provenance"]["binary_path"]}` sha256 `{data["provenance"]["binary_sha256"][:16]}` (built `{data["provenance"]["binary_mtime"]}`), repo HEAD `{data["provenance"]["repo_head"]}`, loadavg `{data["provenance"]["loadavg_1m_5m_15m"]}`  
> **Evaluation Profile**: Dual-Engine (Rust Bare-Metal Engine + Python Algorithmic Suite)  
> **License**: Apache-2.0

---

## ⚡ 1. Latency & Throughput Benchmark

Every number below comes from timed `gen-zero reflex` CLI subprocess calls ({lat["iterations"]} iterations, sequential, single caller).

- **CLI reflex call (process spawn + IO included)**:
  - Mean Latency: `{lat["mean_latency_us"]} µs` (~{lat["mean_latency_us"]/1000.0:.2f} ms)
  - P50 Latency: `{lat["p50_latency_us"]} µs`
  - P90 Latency: `{lat["p90_latency_us"]} µs`
  - P95 Latency: `{lat["p95_latency_us"]} µs`
  - P99 Latency: `{lat["p99_latency_us"]} µs`
  - Sequential CLI throughput: `{lat["cli_throughput_qps"]} decisions/s`
- **In-engine reflex latency**: `{lat["in_engine_reflex_latency_us"]}`
- **In-engine MCTS latency**: `{lat["in_engine_mcts_latency_ms"]}`

No competitor performance is measured by this run.

---

## ⚖️ 2. Candidate-Order Flip Rate (empirical)

Counts how often the chosen action changes when the candidate list is shuffled.

- **Valid Comparisons**: `{eq["valid_comparisons"]}`
- **Observed Decision Flips**: `{eq["observed_flips"]}`
- **Candidate Order Flip Rate**: `{eq["flip_rate_percent"]}%` (over valid comparisons only)
- **Error Samples**: `{eq["error_samples"]}` (status: `{eq["status"]}`)
- **Scope**: {eq["scope"]}

This is an observed count, not a proof, and no LLM baseline was measured here.
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()

