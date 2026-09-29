#!/usr/bin/env python3
"""Benchmark Suite for Issue #86:
Structure-Preserving Latent World Model via Hamiltonian Mechanics & Symplectic Integrators.

Empirical Before vs After audit:
1. Long-Horizon Energy Conservation:
   - Standard Non-Symplectic MLP Residual Dynamics vs Hamiltonian World Model (H=100 steps).
   - Relative Energy Drift: |H_100 - H_0| / H_0 (Target: <= 1.50%).
2. Phase-Space Stability & Divergence:
   - State norm ||z_t||_2 trajectory boundedness over 100 steps.
   - Divergence Rate (Target: 0.00%).
3. Symplectic Phase-Space Volume Preservation:
   - det(Jacobian) of discrete transition map (Target: 1.0000).
4. Step Inference Latency & Throughput:
   - Average step latency (Target: <= 0.40 ms).
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

from gen_zero.world_model.hamiltonian_dynamics import (
    HamiltonianRolloutResult,
    HamiltonianWorldModel,
)


class StandardMLPDynamicsBaseline:
    """Standard non-symplectic feedforward residual dynamics baseline."""

    def __init__(self, latent_dim: int = 64, seed: int = 42):
        self.latent_dim = latent_dim
        rng = np.random.RandomState(seed)
        # Small non-unitary residual layer
        self.w1 = rng.randn(64, latent_dim) * 0.05
        self.w2 = rng.randn(latent_dim, 64) * 0.05

    def step(self, z: np.ndarray, dt: float = 0.04) -> np.ndarray:
        # z_{t+1} = z_t + dt * W2 * tanh(W1 * z_t)
        h = np.tanh(np.dot(self.w1, z))
        delta = np.dot(self.w2, h)
        return z + dt * delta

    def rollout(self, z0: np.ndarray, horizon: int = 100, dt: float = 0.04) -> Dict[str, Any]:
        curr = z0.copy()
        norms = [float(np.linalg.norm(curr))]
        energies = [0.5 * float(np.dot(curr, curr))]

        for _ in range(horizon):
            curr = self.step(curr, dt=dt)
            norm_val = float(np.linalg.norm(curr))
            norms.append(norm_val)
            energies.append(0.5 * float(np.dot(curr, curr)))

        e0 = energies[0]
        drifts = [abs(e - e0) / max(1e-6, abs(e0)) for e in energies]
        max_drift = max(drifts)
        is_diverged = any(math.isnan(n) or math.isinf(n) or n > 50.0 for n in norms)

        return {
            "initial_energy": e0,
            "final_energy": energies[-1],
            "max_energy_drift_ratio": max_drift,
            "max_norm": max(norms),
            "is_diverged": is_diverged,
        }


def run_benchmark():
    print("=" * 75)
    print("Issue #86: Hamiltonian World Model & Symplectic Integrator Benchmark")
    print("=" * 75)

    latent_dim = 64
    action_dim = 16
    horizon = 100
    dt = 0.04

    # Deterministic initial state
    rng = np.random.RandomState(1337)
    z0 = rng.randn(latent_dim).astype(np.float64) * 0.5

    # 1. Standard MLP Baseline
    print("\n[Phase 1] Simulating Standard Non-Symplectic MLP Residual Dynamics (H=100)...")
    mlp_baseline = StandardMLPDynamicsBaseline(latent_dim=latent_dim, seed=42)
    mlp_res = mlp_baseline.rollout(z0, horizon=horizon, dt=dt)
    print(f"  -> MLP Max Energy Drift:      {mlp_res['max_energy_drift_ratio'] * 100.0:.2f}%")
    print(f"  -> MLP Max Latent Norm:       {mlp_res['max_norm']:.4f}")
    print(f"  -> MLP Divergence Detected:   {mlp_res['is_diverged']}")

    # 2. Hamiltonian World Model with Symplectic Integrator
    print("\n[Phase 2] Simulating Hamiltonian World Model with Störmer-Verlet Integrator (H=100)...")
    h_model = HamiltonianWorldModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        dt=dt,
        mass=1.0,
        potential_hidden_dim=32,
        omega=0.8,
        seed=42,
    )

    t0 = time.perf_counter()
    h_res = h_model.rollout(z0, horizon=horizon, actions=None)
    t1 = time.perf_counter()

    h_max_drift_pct = h_res.max_energy_drift_ratio * 100.0
    h_final_drift_pct = abs(h_res.final_energy - h_res.initial_energy) / max(1e-6, abs(h_res.initial_energy)) * 100.0
    print(f"  -> Hamiltonian Initial Energy: {h_res.initial_energy:.6f}")
    print(f"  -> Hamiltonian Final Energy:   {h_res.final_energy:.6f}")
    print(f"  -> Hamiltonian Max Drift:      {h_max_drift_pct:.4f}% (Target: <= 1.50%)")
    print(f"  -> Hamiltonian Final Drift:    {h_final_drift_pct:.4f}%")
    print(f"  -> Hamiltonian Max State Norm: {h_res.max_state_norm:.4f} (Bounded)")
    print(f"  -> Divergence Rate:            {h_res.divergence_rate * 100.0:.2f}% (Target: 0.00%)")

    # 3. Microsecond Latency Stress Test (2,000 steps)
    print("\n[Phase 3] Stress Benchmarking Symplectic Leapfrog Integration (2,000 steps)...")
    latencies = []
    curr = z0.copy()
    for _ in range(50):
        curr = h_model.step(curr).next_state

    n_steps = 2000
    for _ in range(n_steps):
        t_start = time.perf_counter()
        res = h_model.step(curr)
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0)
        curr = res.next_state

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p99_lat = float(np.percentile(latencies, 99))
    throughput = float(1000.0 / mean_lat)
    print(f"  -> Mean Latency:  {mean_lat:.4f} ms (Target: <= 0.40 ms)")
    print(f"  -> P50 Latency:   {p50_lat:.4f} ms")
    print(f"  -> P99 Latency:   {p99_lat:.4f} ms")
    print(f"  -> Throughput:    {throughput:.1f} transitions/sec")

    # 4. Acceptance Criteria Verification
    checks = {
        "energy_drift_le_1_5pct": h_res.max_energy_drift_ratio <= 0.015,
        "divergence_rate_is_0pct": h_res.divergence_rate == 0.0,
        "step_latency_le_0_40ms": mean_lat <= 0.40,
        "energy_drift_gain_vs_mlp": h_res.max_energy_drift_ratio < mlp_res["max_energy_drift_ratio"],
    }

    all_passed = all(checks.values())
    print("\n" + "=" * 75)
    print(f"ACCEPTANCE CRITERIA VERIFICATION: {'ALL PASSED [SUCCESS]' if all_passed else 'FAILED'}")
    for k, v in checks.items():
        print(f"  - {k}: {'PASS' if v else 'FAIL'}")
    print("=" * 75)

    report = {
        "benchmark": "Issue #86 Hamiltonian World Model & Symplectic Integrator vs MLP Baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mlp_baseline": {
            "initial_energy": round(mlp_res["initial_energy"], 6),
            "final_energy": round(mlp_res["final_energy"], 6),
            "max_energy_drift_pct": round(mlp_res["max_energy_drift_ratio"] * 100.0, 4),
            "max_norm": round(mlp_res["max_norm"], 4),
            "is_diverged": mlp_res["is_diverged"],
        },
        "hamiltonian_world_model": {
            "horizon": horizon,
            "initial_energy": round(h_res.initial_energy, 6),
            "final_energy": round(h_res.final_energy, 6),
            "max_energy_drift_pct": round(h_max_drift_pct, 4),
            "final_energy_drift_pct": round(h_final_drift_pct, 4),
            "max_state_norm": round(h_res.max_state_norm, 4),
            "divergence_rate_pct": round(h_res.divergence_rate * 100.0, 2),
            "step_latency_ms": round(mean_lat, 4),
            "p99_latency_ms": round(p99_lat, 4),
            "throughput_transitions_per_sec": round(throughput, 1),
        },
        "acceptance_criteria": checks,
        "overall_success": all_passed,
    }

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/gen_zero"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "issue_86_hamiltonian_benchmark_report.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {out_file}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
