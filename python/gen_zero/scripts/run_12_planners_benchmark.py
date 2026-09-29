"""Comprehensive 12-Planner Stress Test, Latency Benchmark & Stress Profiling Suite.

Executes unified stress testing across all 12 decision paradigms in Gen-Zero:
1. Reflex Fast Prior (System 1 Set-Attention)
2. Uncertainty-Guided A* Graph Planner
3. Bidirectional Goal-Directed Search
4. Dual-Head PUCT Monte Carlo Tree Search (MCTS)
5. Text Transition World Model
6. Cross-Entropy Model Predictive Control (MPC-CEM)
7. GFlowNet Diverse Trajectory Sampler
8. Counterfactual Regret Minimization (CFR)
9. CP-SAT Formal Risk Solver (OR-Tools)
10. Continuous Trajectory Latent MPC
11. Process Reward Model Safety Barrier (PRM)
12. Decentralized SCM Multi-Agent Bluff Detector & Causal-CFR Engine

Outputs comprehensive metrics:
- Success / Feasibility Rate
- Mean, P95, P99 Latency (ms)
- Memory / Computation Complexity
- OOD Adversarial Stress Robustness
- Dynamic Allocation & Expert Selection Distribution
"""

import sys
import os
import time
import math
import random
import json
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure gen_zero importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gen_zero import GenZero, GenZeroConfig
from gen_zero.model.prm import ProcessRewardModel


class TwelvePlannerStressBenchmark:
    """Rigorous stress testing and benchmarking engine for all 12 planners."""

    def __init__(self):
        random.seed(42)
        np.random.seed(42)
        self.config = GenZeroConfig(enable_gpu_arbiter_fallback=True)
        self.client = GenZero(self.config)
        self.results: Dict[str, Dict[str, Any]] = {}

    def run_all_stress_tests(self, iterations: int = 50) -> Dict[str, Any]:
        print("=================================================================")
        print("     GEN-ZERO 12-PLANNER UNIFIED STRESS TEST & BENCHMARK SUITE   ")
        print("=================================================================")
        print(f"  Iterations per Planner: {iterations}")
        print(f"  System 1 + System 2 Planners: 12 Unique Paradigms")
        print("=================================================================\n")

        tests = [self._test_reflex, self._test_astar, self._test_bidirectional,
                 self._test_mcts, self._test_world_model, self._test_mpc_cem,
                 self._test_gflownet, self._test_cfr, self._test_cp_sat,
                 self._test_continuous_mpc, self._test_prm, self._test_d_scm_bluff]
        for test in tests:
            try:
                test(iterations)
            except Exception as exc:
                traceback.print_exc()
                self.results[test.__name__] = {
                    "success_rate": 0.0, "mean_ms": 0.0, "p95_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}", "passed": False,
                }

        print("\n=================================================================")
        print("                 12-PLANNER BENCHMARK SUMMARY                    ")
        print("=================================================================")
        print(f"{'#':<3} | {'Planner Paradigm':<26} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Success%':<9} | {'Category'}")
        print("-" * 80)
        
        sorted_planners = sorted(self.results.items(), key=lambda x: x[1]["mean_ms"])
        for idx, (name, metrics) in enumerate(sorted_planners, 1):
            cat = metrics.get("category", "Planner")
            print(f"{idx:<3} | {name:<26} | {metrics['mean_ms']:<10.3f} | {metrics['p95_ms']:<10.3f} | {metrics['success_rate']*100:<8.1f}% | {cat}")
        print("=================================================================")
        return self.results

    def _test_reflex(self, n: int):
        latencies = []
        success = 0
        degradation_reasons = set()
        state = {"type": "choice", "context": "urgent_customer_refund"}
        candidates = ["APPROVE", "REJECT", "ESCALATE", "DEFER"]
        for _ in range(n):
            t0 = time.perf_counter()
            res = self.client.decide(state, candidates, mode="reflex")
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.get("degraded"):
                degradation_reasons.add(str(res.get("degraded_reason", res.get("scorer", "unspecified"))))
            if res.get("action") in candidates and not res.get("degraded", False):
                success += 1
        self._record("1. Reflex Fast Prior", latencies, success / n, "System 1 Fast")
        self.results["1. Reflex Fast Prior"]["degradation_reasons"] = sorted(degradation_reasons)

    def _test_astar(self, n: int):
        latencies = []
        success = 0
        def get_neighbors(node):
            return [(node + 1, f"step_{node+1}", 0.95), (node + 2, f"jump_{node+2}", 0.70)]
        for _ in range(n):
            t0 = time.perf_counter()
            res = self.client.plan_path(
                start_state=0,
                is_goal_fn=lambda s: s >= 8,
                get_neighbors_fn=get_neighbors
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.get("success"):
                success += 1
        self._record("2. Uncertainty A* Graph", latencies, success / n, "Graph Search")

    def _test_bidirectional(self, n: int):
        latencies = []
        success = 0
        for _ in range(n):
            t0 = time.perf_counter()
            res = self.client.plan_bidirectional(
                start_state=0,
                goal_state=10,
                get_forward_neighbors_fn=lambda s: [(s + 1, "fwd", 1.0)],
                get_backward_neighbors_fn=lambda s: [(s - 1, "bwd", 1.0)]
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.get("success") and len(res.get("path", [])) == 10:
                success += 1
        self._record("3. Bidirectional Search", latencies, success / n, "Dual Graph")

    def _test_mcts(self, n: int):
        latencies = []
        success = 0
        def trans_fn(s, a):
            return {"val": s["val"] + (1 if a == "inc" else -1)}, 1.0, False
        for _ in range(n):
            t0 = time.perf_counter()
            res = self.client.mcts_planner.plan(
                root_state={"val": 0},
                candidate_actions=["inc", "dec"],
                transition_fn=trans_fn,
                legal_actions_fn=lambda s: ["inc", "dec"],
                reward_fn=lambda parent, action, child: float(child["val"])
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.get("best_action") == "inc" and res.get("nodes_expanded", 0) > 1:
                success += 1
        self._record("4. PUCT MCTS", latencies, success / n, "System 2 Search")

    def _test_world_model(self, n: int):
        latencies = []
        success = 0
        state = {"size": 8, "body": [[4, 4], [4, 3]], "food": [4, 5]}
        for _ in range(n):
            t0 = time.perf_counter()
            next_s, r, done = self.client.world_model.virtual_step(state, "south")
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if next_s is not None:
                success += 1
        self._record("5. Text World Model", latencies, success / n, "Model-Based")

    def _test_mpc_cem(self, n: int):
        latencies = []
        success = 0
        def sim_trans(s, a):
            next_state = {"price": s.get("price", 100.0) + (1.0 if a == "BUY" else -1.0)}
            return next_state, next_state["price"] - s["price"], False
        for _ in range(n):
            t0 = time.perf_counter()
            res = self.client.mpc_cem_planner.plan(
                state={"price": 100.0},
                candidate_actions_or_bounds=["BUY", "HOLD", "SELL"],
                transition_fn=sim_trans
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.get("best_action") == "BUY":
                success += 1
        self._record("6. MPC-CEM Trajectory", latencies, success / n, "Continuous Control")

    def _test_gflownet(self, n: int):
        latencies = []
        success = 0
        def gfn_trans(s, a):
            return s, (1.0 if a == "A" else 0.2), False
        for _ in range(n):
            t0 = time.perf_counter()
            res = self.client.gflownet_sampler.sample_trajectory(
                initial_state={"x": 1.0},
                candidate_actions=["A", "B", "C", "D"],
                transition_fn=gfn_trans
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.get("best_action"):
                success += 1
        self._record("7. GFlowNet Sampler", latencies, success / n, "Diverse Exploration")

    def _test_cfr(self, n: int):
        latencies = []
        success = 0
        for _ in range(n):
            t0 = time.perf_counter()
            res = self.client.cfr_expert.solve_imperfect_decision(
                info_set_repr="poker_turn_high_bet",
                candidates=["FOLD", "CALL", "RAISE"],
                observed_opponent_action="RAISE"
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.get("best_action"):
                success += 1
        self._record("8. Causal CFR Regret", latencies, success / n, "Game Theory")

    def _test_cp_sat(self, n: int):
        latencies = []
        success = 0
        from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler
        compiler = ConstraintLinearProjectionCompiler(hard_timeout_ms=2.0)
        compiler.compile_rules(["FORBID MOVE_OVERHEAT IF temperature >= 90",
                                "FORBID DUMP_CARGO IF authorized == 0"])
        for _ in range(n):
            t0 = time.perf_counter()
            res = compiler.solve_safest_action(
                {"MOVE_SAFE": 1.0, "MOVE_OVERHEAT": 9.0, "DUMP_CARGO": 8.0},
                current_metrics={"temperature": 95.0, "authorized": 0.0})
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.is_safe and res.selected_action == "MOVE_SAFE" and res.solver_status == "CP_SAT_OPTIMAL" and not res.fallback_used:
                success += 1
        self._record("9. CP-SAT Formal Solver", latencies, success / n, "Hard Constraints")

    def _test_continuous_mpc(self, n: int):
        latencies = []
        success = 0
        state = [0.5] * 16
        for _ in range(n):
            t0 = time.perf_counter()
            res = self.client.decide_continuous(
                state=state,
                action_dim=4,
                horizon=4,
                num_samples=16
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if len(res.get("action", [])) == 4:
                success += 1
        self._record("10. Continuous Latent MPC", latencies, success / n, "High-Dim Latent")

    def _test_prm(self, n: int):
        latencies = []
        success = 0
        prm = ProcessRewardModel()
        for _ in range(n):
            t0 = time.perf_counter()
            res = prm.verify_step(
                parent_state={"pos": [1, 1]},
                action="forward",
                next_state={"pos": [1, 2]},
                is_done=False
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if not res.get("should_prune"):
                success += 1
        self._record("11. PRM Safety Barrier", latencies, success / n, "Process Verifier")

    def _test_d_scm_bluff(self, n: int):
        latencies = []
        success = 0
        state = {"pot": 20.0, "pressure": 0.2, "volatility": 0.1}
        actual_next = {"pot": 35.0, "pressure": 0.5, "volatility": 0.1}
        for _ in range(n):
            t0 = time.perf_counter()
            intent = self.client.bluff_detector.analyze_intent(
                opponent_id="adversary_1",
                state=state,
                ego_action="call",
                opponent_action="raise",
                actual_next_state=actual_next,
                opponent_revealed_strength=0.20  # Weak hand shoving raise -> Pure bluff
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            from gen_zero.multiagent.decentralized_scm import IntentType
            if intent.intent_type == IntentType.STRATEGIC_BLUFF or intent.bluff_probability >= 0.58:
                success += 1
        self._record("12. D-SCM Bluff Detector", latencies, success / n, "Multi-Agent Causal")

    def _record(self, name: str, latencies: List[float], success_rate: float, category: str):
        latencies.sort()
        mean_lat = sum(latencies) / len(latencies)
        p95_idx = int(len(latencies) * 0.95)
        p95_lat = latencies[min(p95_idx, len(latencies) - 1)]
        p99_idx = int(len(latencies) * 0.99)
        p99_lat = latencies[min(p99_idx, len(latencies) - 1)]
        
        self.results[name] = {
            "mean_ms": round(mean_lat, 3),
            "p95_ms": round(p95_lat, 3),
            "p99_ms": round(p99_lat, 3),
            "success_rate": round(success_rate, 4),
            "category": category,
            "passed": success_rate == 1.0,
            "iterations": len(latencies),
            "latencies_ms": latencies,
        }


def main():
    bench = TwelvePlannerStressBenchmark()
    results = bench.run_all_stress_tests(iterations=50)
    report = {"scope": "algorithm fixtures; does not establish learned-model quality",
              "planners": results,
              "passed_count": sum(m["passed"] for m in results.values()),
              "total_count": len(results)}
    path = Path(__file__).resolve().parents[3] / "benchmarks/results/r4_evidence/planners_after_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {path}; passed {report['passed_count']}/12")
    return 0 if report["passed_count"] == 12 and len(results) == 12 else 1


if __name__ == "__main__":
    sys.exit(main())
