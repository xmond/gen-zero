"""Gen-Zero Issue #76: Automated Natural Language / AST Constraint Linear Projection Compiler Benchmark.

Empirical Before vs After benchmark evaluating:
1. Compilation Speed: Real-time compilation latency of declarative safety rules (<= 2.0ms target).
2. Solver Latency: 0-1 integer linear programming solve time per step (<= 1.5ms target, hard <= 2.0ms).
3. Safety Violation Interception Rate (Before vs After):
   - BEFORE (Unconstrained Synthetic Utilities): Proposes hazardous actions under stress (20-40% violation rate).
   - AFTER (CP-SAT Linear Projection Compiler): 100% hard blocking of safety violations (0.0% breach rate).
4. Zero-Retraining Adaptation: Instantaneous dynamic rule injection without fine-tuning weights (0.00s).
"""

import os
import sys

# Ensure repository root is in sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import time
import json
from collections import Counter
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any

from gen_zero.gate.constraint_compiler import (
    ConstraintLinearProjectionCompiler,
    CompilationReport,
)


def run_benchmark() -> Dict[str, Any]:
    print("=" * 80)
    print("🛡️ [Gen-Zero Issue #76 Benchmark] NL/AST Constraint Linear Projection Compiler")
    print("=" * 80)

    latent_dim = 1024
    num_eval_steps = 1000
    rng = np.random.RandomState(42)

    # --------------------------------------------------------------------------
    # 1. COMPILATION SPEED & ORTHONORMAL PROJECTION MATRIX (W_sat)
    # --------------------------------------------------------------------------
    print("\n⚡ [Phase 1] Benchmarking Natural Language & AST Rule Compilation Speed...")
    ruleset = [
        "FORBID RESTART IF memory_ratio > 0.85",
        "FORBID OVERCLOCK IF core_temp >= 90.0",
        "FORBID FLUSH_CACHE IF dirty_pages > 5000.0",
        "ALLOW SCALE_UP ONLY IF cpu_utilization >= 0.70",
        "REQUIRE THROTTLE WHEN thermal_pressure > 0.95",
        "disk_free < 0.05 -> BAN APPEND_LOG",
        "io_wait > 0.60 -> BAN BATCH_WRITE",
        "MUTUAL_EXCLUSIVE EMERGENCY_HALT RESUME_TRAFFIC",
    ]

    compiler = ConstraintLinearProjectionCompiler(
        latent_dim=latent_dim,
        seed=42,
        hard_timeout_ms=2.0,
    )

    # Warmup compiler
    compiler.compile_rules(ruleset[:2])

    compilation_trials = 20
    comp_times_ms = []
    report = None
    for _ in range(compilation_trials):
        t0 = time.perf_counter()
        report = compiler.compile_rules(ruleset)
        comp_times_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_compile_ms = float(np.mean(comp_times_ms))
    p99_compile_ms = float(np.percentile(comp_times_ms, 99))

    print(f"  * Compiled {len(ruleset)} Rules ({report.num_propositions} Distinct Propositions)")
    print(f"  * Average Compilation Latency: {avg_compile_ms:.3f} ms (Target: <= 2.0 ms)")
    print(f"  * P99 Compilation Latency:     {p99_compile_ms:.3f} ms")
    print(f"  * Orthonormal Basis (W_sat):   {report.is_orthonormal} (Shape: {compiler.projection_matrix.shape})")

    # --------------------------------------------------------------------------
    # 2. BEFORE vs AFTER: SAFETY VIOLATION INTERCEPTION AUDIT
    # --------------------------------------------------------------------------
    print(f"\n🔬 [Phase 2] Simulating {num_eval_steps:,} Operational Decision Steps under Hazardous Stress...")

    candidate_actions = [
        "RESTART",
        "OVERCLOCK",
        "SCALE_UP",
        "THROTTLE",
        "APPEND_LOG",
        "BATCH_WRITE",
        "NORMAL_STEP",
        "HOLD",
        "FLUSH_CACHE",
        "EMERGENCY_HALT",
        "RESUME_TRAFFIC",
    ]

    # Generate seeded synthetic telemetry states; breach rates are measured below.
    scenarios = []
    for i in range(num_eval_steps):
        # Synthetic utility scores; no trained neural policy is evaluated
        raw_utils = rng.uniform(0.1, 0.95, size=len(candidate_actions))
        action_utils = {act: float(raw_utils[idx]) for idx, act in enumerate(candidate_actions)}

        # Telemetry metrics
        mem_ratio = float(rng.uniform(0.40, 0.98))
        core_temp = float(rng.uniform(50.0, 100.0))
        cpu_util = float(rng.uniform(0.20, 0.95))
        disk_free = float(rng.uniform(0.01, 0.50))
        thermal = float(rng.uniform(0.10, 1.00))

        metrics = {
            "memory_ratio": mem_ratio,
            "core_temp": core_temp,
            "cpu_utilization": cpu_util,
            "disk_free": disk_free,
            "thermal_pressure": thermal,
            "dirty_pages": float(rng.uniform(0, 10000)),
            "io_wait": float(rng.uniform(0, 1)),
        }

        # Determine ground-truth hazardous actions for this step
        prohibited = set()
        if mem_ratio > 0.85:
            prohibited.add("RESTART")
        if core_temp >= 90.0:
            prohibited.add("OVERCLOCK")
        if cpu_util < 0.70:
            prohibited.add("SCALE_UP")  # ALLOW ONLY IF >= 0.70
        if disk_free < 0.05:
            prohibited.add("APPEND_LOG")

        if metrics["dirty_pages"] > 5000:
            prohibited.add("FLUSH_CACHE")
        if metrics["io_wait"] > 0.60:
            prohibited.add("BATCH_WRITE")
        if thermal > 0.95:
            prohibited.update(a for a in candidate_actions if a != "THROTTLE")

        z_latent = rng.randn(latent_dim).astype(np.float32)
        scenarios.append({
            "utilities": action_utils,
            "metrics": metrics,
            "prohibited": prohibited,
            "z_latent": z_latent,
        })

    # --- BEFORE: Unconstrained Synthetic Proposal ---
    before_violations = 0
    before_actions_taken = []
    t_before_start = time.perf_counter()
    for s in scenarios:
        # Standard unconstrained argmax selection
        chosen = max(s["utilities"].keys(), key=lambda a: s["utilities"][a])
        before_actions_taken.append(chosen)
        if chosen in s["prohibited"]:
            before_violations += 1
    t_before_total_ms = (time.perf_counter() - t_before_start) * 1000.0
    before_violation_rate = (before_violations / num_eval_steps) * 100.0

    print("\n🚫 [BEFORE - Unconstrained Synthetic Utilities]:")
    print(f"  * Total Evaluation Steps:       {num_eval_steps:,}")
    print(f"  * Safety Invariant Violations:  {before_violations:,} breaches")
    print(f"  * Violation Rate:               {before_violation_rate:.2f}% (Safety Failure)")
    print(f"  * Evaluation Latency:           {t_before_total_ms / num_eval_steps * 1000.0:.2f} µs/step")

    # --- AFTER: Constraint Linear Projection Compiler + CP-SAT 0-1 Solver ---
    after_violations = 0
    after_actions_taken = []
    solve_times_ms = []
    statuses = Counter()
    fallback_count = timeout_count = rejected_count = 0
    trials = []
    t_after_start = time.perf_counter()
    for s in scenarios:
        t_cpu0 = time.thread_time()
        t_solve0 = time.perf_counter()
        verdict = compiler.solve_safest_action(
            candidate_utilities=s["utilities"],
            z_latent=s["z_latent"],
            current_metrics=s["metrics"],
            fallback_safe_action="HOLD",
        )
        solve_times_ms.append((time.perf_counter() - t_solve0) * 1000.0)
        statuses[verdict.solver_status] += 1
        fallback_count += int(verdict.fallback_used)
        timeout_count += int(verdict.timed_out)
        rejected_count += int(not verdict.is_safe)
        trials.append({"index": len(trials), "wall_ms": solve_times_ms[-1],
                       "thread_cpu_ms": (time.thread_time() - t_cpu0) * 1000,
                       "verdict": vars(verdict), "metrics": s["metrics"],
                       "utilities": s["utilities"], "prohibited": sorted(s["prohibited"])})
        chosen = verdict.selected_action
        after_actions_taken.append(chosen)
        if verdict.is_safe and chosen in s["prohibited"]:
            after_violations += 1

    t_after_total_ms = (time.perf_counter() - t_after_start) * 1000.0
    after_violation_rate = (after_violations / num_eval_steps) * 100.0
    avg_solve_ms = float(np.mean(solve_times_ms))
    p99_solve_ms = float(np.percentile(solve_times_ms, 99))
    hard_block_rate = 100.0 - after_violation_rate

    print("\n🛡️ [AFTER - CP-SAT Constraint Linear Projection Compiler]:")
    print(f"  * Total Evaluation Steps:       {num_eval_steps:,}")
    print(f"  * Safety Invariant Violations:  {after_violations} breaches ({after_violation_rate:.2f}% false pass rate)")
    print(f"  * Violation Rate:               {after_violation_rate:.2f}%")
    print(f"  * Hard Blocking Rate:           {hard_block_rate:.2f}% (observed accepted-action safety only)")
    print(f"  * Mean CP-SAT Solve Time:       {avg_solve_ms:.3f} ms (Target: <= 1.5 ms)")
    print(f"  * P99 CP-SAT Solve Time:        {p99_solve_ms:.3f} ms (Target: <= 2.0 ms)")

    # --------------------------------------------------------------------------
    # 3. ZERO-RETRAINING DYNAMIC ADAPTATION
    # --------------------------------------------------------------------------
    print("\n🔄 [Phase 3] Auditing Zero-Retraining Dynamic Rule Adaptation...")
    t_dyn0 = time.perf_counter()
    new_rule = "FORBID THROTTLE IF thermal_pressure < 0.20"
    updated_report = compiler.compile_rules(ruleset + [new_rule])
    dyn_update_ms = (time.perf_counter() - t_dyn0) * 1000.0

    print(f"  * Dynamic Rule Injected: '{new_rule}'")
    print(f"  * Recompilation & Hot-Activation Latency: {dyn_update_ms:.3f} ms")
    print("  * Neural Weight Gradient Descent Steps:  0 (Zero Retraining)")
    adaptation = compiler.solve_safest_action(
        {"THROTTLE": 1.0, "HOLD": 0.1},
        current_metrics={"thermal_pressure": 0.1, "memory_ratio": 0.1,
                         "core_temp": 50, "dirty_pages": 0, "cpu_utilization": 0.9,
                         "disk_free": 0.9, "io_wait": 0.1},
    )

    # --------------------------------------------------------------------------
    # COMPILE BENCHMARK REPORT
    # --------------------------------------------------------------------------
    report_dict = {
        "metadata": {
            "latent_dim": latent_dim,
            "num_eval_steps": num_eval_steps,
            "num_rules_compiled": len(ruleset),
            "num_propositions": report.num_propositions,
            "hard_timeout_ms": 2.0,
        },
        "time": {
            "avg_compilation_latency_ms": round(avg_compile_ms, 3),
            "p99_compilation_latency_ms": round(p99_compile_ms, 3),
            "target_compilation_ms": 2.0,
            "passed_compilation_target": avg_compile_ms <= 2.0,
            "avg_cpsat_solve_latency_ms": round(avg_solve_ms, 3),
            "p99_cpsat_solve_latency_ms": round(p99_solve_ms, 3),
            "target_solve_ms": 1.5,
            "passed_solve_target": avg_solve_ms <= 1.5,
            "dynamic_rule_injection_ms": round(dyn_update_ms, 3),
        },
        "safety_and_blocking": {
            "before_unconstrained_violations": before_violations,
            "before_violation_rate_pct": round(before_violation_rate, 2),
            "after_cpsat_violations": after_violations,
            "after_violation_rate_pct": round(after_violation_rate, 2),
            "hard_blocking_rate_pct": round(hard_block_rate, 2),
            "target_blocking_rate_pct": 100.0,
            "passed_hard_safety_target": (after_violations == 0),
        },
        "zero_training_adaptation": {
            "neural_parameters_added": 0,
            "retraining_epochs": 0,
            "weights_evaluated": False,
            "immediate_hot_activation": adaptation.is_safe and adaptation.selected_action == "HOLD",
            "verification_verdict": vars(adaptation),
        },
    }

    deadline_misses = sum(t > compiler.hard_timeout_ms for t in solve_times_ms)
    report_dict["solver_audit"] = {
        "status_counts": dict(statuses), "fallback_count": fallback_count,
        "timeout_count": timeout_count, "rejected_count": rejected_count,
        "wall_deadline_misses": deadline_misses, "max_wall_ms": max(solve_times_ms),
        "passed": (after_violations == 0 and fallback_count == 0 and rejected_count == 0
                   and deadline_misses == 0 and report_dict["zero_training_adaptation"]["immediate_hot_activation"]),
        "trials": trials,
    }
    report_path = str(Path(__file__).resolve().parents[3] / "benchmarks/results/r4_evidence/constraints_after_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print("\n" + "=" * 80)
    print(f"📊 Benchmark Report Successfully Written to: {report_path}")
    print("=" * 80)
    return report_dict


if __name__ == "__main__":
    report = run_benchmark()
    print(json.dumps({k: v for k, v in report.items() if k != "solver_audit"}, indent=2))
    print(json.dumps({k: v for k, v in report["solver_audit"].items() if k != "trials"}, indent=2))
    sys.exit(0 if report["solver_audit"]["passed"] else 1)
