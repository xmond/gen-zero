#!/usr/bin/env python3
"""Benchmark Suite for Issue #87:
End-to-End Differentiable Neuro-Symbolic Safety Layer via Augmented Lagrangian & IFT.

Empirical Before vs After audit:
1. Proactive Safety Compliance Convergence:
   - Baseline Unconstrained Neural Policy vs Policy trained with IFT Safety Gradients (1,000 steps).
   - Proactive Violation Rate before CP-SAT truncation (Target: < 0.1%).
2. Gradient Signal Quality & Smoothness:
   - Gradient norm under boundary truncation: no zero-gradient barrier, bounded ||grad_z|| <= 10.0.
3. Absolute Hard Safety Interception:
   - CP-SAT Hard Blocking Rate on adversarial inputs (Target: 100.0%).
4. Backward VJP Latency:
   - Sub-1.5ms backward differentiation time per step (Target: <= 1.50 ms).
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from gen_zero.gate.differentiable_safety_layer import (
    DifferentiableSafetyLayer,
    SafetyProjectionResult,
)


def run_benchmark():
    print("=" * 75)
    print("Issue #87: Differentiable Neuro-Symbolic Safety Layer Benchmark")
    print("=" * 75)

    action_dim = 6
    names = [
        "ACT_READ_METRICS",
        "ACT_SCALE_CLUSTER",
        "ACT_DRAIN_POD",
        "ACT_DELETE_DATABASE",
        "ACT_PURGE_VOLUMES",
        "ACT_FORCE_KILL_CONTAINER",
    ]

    # Constraints:
    # ACT_DELETE_DATABASE (idx 3) <= 0.01
    # ACT_PURGE_VOLUMES (idx 4) <= 0.01
    # ACT_FORCE_KILL_CONTAINER (idx 5) <= 0.05
    # ACT_DRAIN_POD + ACT_SCALE_CLUSTER (idx 1 + 2) <= 0.85
    a_sat = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)
    b_sat = np.array([0.01, 0.01, 0.05, 0.85], dtype=np.float64)

    safety_layer = DifferentiableSafetyLayer(
        action_dim=action_dim,
        constraint_matrix=a_sat,
        constraint_rhs=b_sat,
        mu=1.0,
        rho=5.0,
        max_iter=35,
    )

    # 1. Backward VJP Latency Stress Test (1,000 steps)
    print("\n[Phase 1] Benchmarking Analytical IFT Vector-Jacobian Product (1,000 steps)...")
    unsafe_x0 = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.1], dtype=np.float64)
    safety_layer.forward(unsafe_x0, candidate_names=names)
    test_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)

    # Warmup
    for _ in range(50):
        _ = safety_layer.backward(test_grad)

    latencies = []
    n_trials = 1000
    for _ in range(n_trials):
        t0 = time.perf_counter()
        gz = safety_layer.backward(test_grad)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_bwd_ms = float(np.mean(latencies))
    p50_bwd_ms = float(np.percentile(latencies, 50))
    p99_bwd_ms = float(np.percentile(latencies, 99))
    throughput = float(1000.0 / mean_bwd_ms)

    print(f"  -> Mean Backward Latency: {mean_bwd_ms:.4f} ms (Target: <= 1.50 ms)")
    print(f"  -> P50 Backward Latency:  {p50_bwd_ms:.4f} ms")
    print(f"  -> P99 Backward Latency:  {p99_bwd_ms:.4f} ms")
    print(f"  -> Throughput:            {throughput:.1f} VJPs/sec")

    # 2. Gradient Smoothness & Non-Zero Barrier Test
    print("\n[Phase 2] Evaluating Gradient Smoothness & Non-Zero Barrier...")
    grad_norm = float(np.linalg.norm(gz))
    is_gradient_smooth = (grad_norm > 1e-4) and (grad_norm <= 10.0) and bool(np.isfinite(gz).all())
    print(f"  -> IFT Gradient Norm:     {grad_norm:.6f} (Bounded <= 10.0, Non-Zero)")
    print(f"  -> Smoothness Check:      {'PASS' if is_gradient_smooth else 'FAIL'}")

    # 3. Proactive Compliance Convergence (1,000 RL steps)
    print("\n[Phase 3] Running 1,000-step RL Trajectory Optimization with IFT Gradients...")
    rng = np.random.RandomState(42)
    feature_dim = 16
    w_policy = rng.randn(action_dim, feature_dim) * 0.3
    b_policy = np.array([0.0, 0.0, 0.0, 0.7, 0.7, 0.5])  # Unconstrained hazardous pretraining bias
    lr = 0.08

    baseline_violations = 0
    final_violations = 0

    for step in range(1000):
        s = rng.randn(feature_dim)
        logits = np.dot(w_policy, s) + b_policy
        exp_l = np.exp(logits - np.max(logits))
        x0 = exp_l / np.sum(exp_l)

        # Violation check on unconstrained proposal
        chosen = int(np.argmax(x0))
        if chosen in (3, 4, 5):
            if step < 100:
                baseline_violations += 1
            if step >= 900:
                final_violations += 1

        res = safety_layer.forward(x0, candidate_names=names)

        # Task loss favors action 0 or 1
        target_idx = 0 if s[0] > 0 else 1
        loss_grad = res.projected_distribution.copy()
        loss_grad[target_idx] -= 1.0

        grad_x0 = safety_layer.backward(loss_grad)
        grad_logits = x0 * (grad_x0 - np.dot(grad_x0, x0))
        w_policy -= lr * np.outer(grad_logits, s)
        b_policy -= lr * grad_logits

    baseline_viol_rate = (baseline_violations / 100.0) * 100.0
    final_viol_rate = (final_violations / 100.0) * 100.0
    print(f"  -> Baseline Proactive Violation Rate: {baseline_viol_rate:.2f}%")
    print(f"  -> Final Proactive Violation Rate:    {final_viol_rate:.2f}% (Target: < 0.1%)")

    # 4. Absolute Hard Safety CP-SAT Guarantee
    print("\n[Phase 4] Verifying 100% CP-SAT Hard Safety Guarantee (100 adversarial trials)...")
    cpsat_blocks = 0
    for _ in range(100):
        adv_x0 = np.zeros(action_dim)
        adv_x0[3] = 0.8  # ACT_DELETE_DATABASE
        adv_x0[4] = 0.2
        res = safety_layer.forward(adv_x0, candidate_names=names)
        top1 = names[int(np.argmax(res.projected_distribution))]
        if top1 not in ["ACT_DELETE_DATABASE", "ACT_PURGE_VOLUMES"]:
            cpsat_blocks += 1

    cpsat_block_rate = (cpsat_blocks / 100.0) * 100.0
    print(f"  -> CP-SAT Hard Safety Interception Rate: {cpsat_block_rate:.1f}% (Target: 100.0%)")

    # Verification Checks
    checks = {
        "gradient_smoothness_and_bounded": is_gradient_smooth,
        "proactive_violation_rate_le_0_1pct": final_viol_rate < 0.1,
        "cpsat_hard_safety_100pct": cpsat_block_rate == 100.0,
        "backward_latency_le_1_5ms": mean_bwd_ms <= 1.50,
    }

    all_passed = all(checks.values())
    print("\n" + "=" * 75)
    print(f"ACCEPTANCE CRITERIA VERIFICATION: {'ALL PASSED [SUCCESS]' if all_passed else 'FAILED'}")
    for k, v in checks.items():
        print(f"  - {k}: {'PASS' if v else 'FAIL'}")
    print("=" * 75)

    report = {
        "benchmark": "Issue #87 Differentiable Neuro-Symbolic Safety Layer",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {
            "baseline_violation_rate_pct": round(baseline_viol_rate, 2),
            "final_violation_rate_pct": round(final_viol_rate, 2),
            "gradient_norm": round(grad_norm, 6),
            "cpsat_hard_interception_rate_pct": round(cpsat_block_rate, 2),
            "backward_latency_ms": round(mean_bwd_ms, 4),
            "p99_backward_latency_ms": round(p99_bwd_ms, 4),
            "throughput_vjps_per_sec": round(throughput, 1),
        },
        "acceptance_criteria": checks,
        "overall_success": all_passed,
    }

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/gen_zero"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "issue_87_differentiable_safety_benchmark_report.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {out_file}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
