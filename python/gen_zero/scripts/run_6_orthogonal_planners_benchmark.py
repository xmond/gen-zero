"""Benchmark Suite for Issue #93: 6 Converged Orthogonal Planning Engines.

Evaluates latency (Mean, P95, P99), success rate, and throughput across:
1. AStarEngine (64B Arena, Bidirectional Meet-in-the-Middle)
2. MctsEngine (64B Node, PUCT, Hamiltonian Dynamics, Causal Pruning)
3. MpcCemEngine (Continuous Latent Trajectory CEM & Receding Horizon)
4. ManifoldGFlowNetEngine (Simplex ETF Geodesic Flow Matching)
5. CfrNashEngine (CFR+, Bayesian Opponent Profiler & Bluff Detection)
6. CpSatFormalEngine (0-1 ILP Hard Safety Masking & NCBF Lie Barrier Filter)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Ensure gen_zero importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gen_zero.planner.engines import (
    AStarArena64B,
    AStarEngine,
    MctsEngine,
    MpcCemEngine,
    ManifoldGFlowNetEngine,
    CfrNashEngine,
    CpSatFormalEngine,
)


class SixOrthogonalPlannersBenchmark:
    """Comprehensive benchmark engine for the 6 converged planning engines."""

    def __init__(self, iterations: int = 50):
        self.iterations = iterations
        self.results: Dict[str, Dict[str, Any]] = {}

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        print("=================================================================")
        print("   GEN-ZERO 6 CONVERGED ORTHOGONAL PLANNERS BENCHMARK (RFC-093) ")
        print("=================================================================")
        print(f"  Iterations per Engine: {self.iterations}")
        print("=================================================================\n")

        self.bench_astar_engine()
        self.bench_mcts_engine()
        self.bench_mpc_cem_engine()
        self.bench_manifold_gflownet_engine()
        self.bench_cfr_nash_engine()
        self.bench_cpsat_formal_engine()

        self._print_summary()
        self._save_results()
        return self.results

    def bench_astar_engine(self) -> None:
        print("[1/6] Benchmarking AStarEngine (64B Arena & Bidirectional)...")
        engine = AStarEngine(lambda_weight=1.2, use_arena=True)
        latencies = []

        def get_fwd(s):
            return [(s + 1, "fwd", 0.98)] if s < 25 else []

        def get_bwd(s):
            return [(s - 1, "fwd", 0.98)] if s > 0 else []

        def h_fn(s1, s2):
            return float(abs(s1 - s2))

        successes = 0
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            res = engine.plan(0, 25, get_fwd, get_bwd, heuristic_fn=h_fn)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            if res.get("success"):
                successes += 1

        self.results["AStarEngine"] = self._compute_metrics(latencies, successes)

    def bench_mcts_engine(self) -> None:
        print("[2/6] Benchmarking MctsEngine (64B Node, PUCT & Causal Pruning)...")
        engine = MctsEngine(num_simulations=40, max_depth=6)
        latencies = []
        candidates = ["step_a", "step_b", "step_c", "risky_step"]

        def trans_fn(s, a):
            next_s = s + (1 if a != "risky_step" else -2)
            return next_s, 1.0 if next_s > s else -1.0, False

        def causal_inv(s, a, next_s, reward, done):
            return next_s >= 0

        successes = 0
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            res = engine.plan(0, candidates, trans_fn, legal_actions_fn=lambda s: candidates,
                              causal_invariant_fn=causal_inv)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            if res.get("best_action") in candidates:
                successes += 1

        self.results["MctsEngine"] = self._compute_metrics(latencies, successes)

    def bench_mpc_cem_engine(self) -> None:
        print("[3/6] Benchmarking MpcCemEngine (Continuous Latent Trajectory CEM)...")
        engine = MpcCemEngine(action_dim=4, horizon=4, num_samples=32, iterations=4)
        latencies = []

        def custom_reward(z, a):
            return -float(np.sum(np.array(a) ** 2))

        successes = 0
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            res = engine.plan_continuous(
                state=[0.0] * 4,
                bounds=(-1.0, 1.0),
                custom_reward_fn=custom_reward,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            if res.get("best_action") is not None:
                successes += 1

        self.results["MpcCemEngine"] = self._compute_metrics(latencies, successes)

    def bench_manifold_gflownet_engine(self) -> None:
        print("[4/6] Benchmarking ManifoldGFlowNetEngine (Simplex ETF Geodesic Flow)...")
        engine = ManifoldGFlowNetEngine(dim=64, temperature=1.0)
        latencies = []
        candidates = ["mode_1", "mode_2", "mode_3", "mode_4", "mode_5"]

        def trans_fn(s, a):
            return f"{s}_{a}", 1.0, False

        successes = 0
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            res = engine.plan(
                state="start",
                candidate_actions=candidates,
                transition_fn=trans_fn,
                sample_count=16,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            if res.get("best_action") in candidates:
                successes += 1

        self.results["ManifoldGFlowNetEngine"] = self._compute_metrics(latencies, successes)

    def bench_cfr_nash_engine(self) -> None:
        print("[5/6] Benchmarking CfrNashEngine (CFR+ & BayesianBeliefTracker)...")
        engine = CfrNashEngine(iterations=20, use_cfr_plus=True)
        latencies = []
        candidates = ["raise_small", "raise_pot", "call", "fold"]

        # Prime opponent tracker
        engine.record_opponent_action("river_turn", "raise_small")
        engine.record_opponent_action("river_turn", "raise_pot")

        successes = 0
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            res = engine.solve_imperfect_decision("river_turn", candidates)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            if res.get("best_action") in candidates:
                successes += 1

        self.results["CfrNashEngine"] = self._compute_metrics(latencies, successes)

    def bench_cpsat_formal_engine(self) -> None:
        print("[6/6] Benchmarking CpSatFormalEngine (0-1 ILP & NCBF Lie Derivative)...")
        engine = CpSatFormalEngine(strict_mode=True, default_alpha=1.0)
        latencies = []
        candidates = ["exec_normal", "drop_speed", "emergency_thrust"]

        def barrier_fn(s):
            return float(s)

        def dynamics_fn(s, a):
            return s + (1.0 if a == "exec_normal" else (-2.0 if a == "drop_speed" else 5.0))

        successes = 0
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            res = engine.plan(
                state=2.0,
                candidates=candidates,
                barrier_fn=barrier_fn,
                dynamics_fn=dynamics_fn,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            if res.get("status") == "OPTIMAL_SATISFIED":
                successes += 1

        self.results["CpSatFormalEngine"] = self._compute_metrics(latencies, successes)

    def _compute_metrics(self, latencies: List[float], successes: int) -> Dict[str, Any]:
        arr = np.array(latencies)
        return {
            "success_rate_pct": round(100.0 * successes / len(latencies), 2),
            "mean_latency_ms": round(float(np.mean(arr)), 4),
            "p50_latency_ms": round(float(np.percentile(arr, 50)), 4),
            "p95_latency_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_latency_ms": round(float(np.percentile(arr, 99)), 4),
            "min_latency_ms": round(float(np.min(arr)), 4),
            "max_latency_ms": round(float(np.max(arr)), 4),
            "throughput_qps": round(1000.0 / max(1e-4, float(np.mean(arr))), 1),
        }

    def _print_summary(self) -> None:
        print("\n" + "=" * 80)
        print(f"{'ENGINE':<26} | {'SUCCESS':<8} | {'MEAN (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10} | {'QPS':<8}")
        print("-" * 80)
        for name, m in self.results.items():
            print(
                f"{name:<26} | {m['success_rate_pct']:>6.1f}% | {m['mean_latency_ms']:>10.4f} | "
                f"{m['p95_latency_ms']:>10.4f} | {m['p99_latency_ms']:>10.4f} | {m['throughput_qps']:>8.1f}"
            )
        print("=" * 80 + "\n")

    def _save_results(self) -> None:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "gen_zero"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "issue_93_6_planners_benchmark_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Report successfully written to {out_path}")


if __name__ == "__main__":
    bench = SixOrthogonalPlannersBenchmark(iterations=50)
    bench.run_all()
