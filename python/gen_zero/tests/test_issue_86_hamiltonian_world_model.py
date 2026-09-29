"""Unit tests for RFC-086 / Issue #86:
Structure-Preserving Latent World Model via Hamiltonian Mechanics & Symplectic Integrators.

Acceptance Criteria:
1. Long-Horizon Energy Conservation:
   In H = 100 step closed-loop continuous simulation, relative energy drift
   |H_H - H_0| / H_0 <= 1.5% (measured standard-MLP baseline drift: 15.67%,
   see benchmark_issue_86_hamiltonian_world_model.py Phase 1).
2. Bounded Phase Space (No Divergence):
   State norm ||z_t||_2 stays strictly bounded; divergence rate is 0.00%.
3. Single-Step Latency:
   Symplectic update latency <= 0.40 ms.
4. Zero-Word Compliance:
   0 hits for historical deprecated keywords.
"""

import math
import os
import re
import time
import unittest
from typing import Dict, List, Tuple

import numpy as np

from gen_zero.world_model.hamiltonian_dynamics import (
    HamiltonianRolloutResult,
    HamiltonianStepResult,
    HamiltonianWorldModel,
    PotentialEnergyNetwork,
)
from gen_zero.client import GenZero
from gen_zero.config import GenZeroConfig
from unittest.mock import patch

try:
    import torch
    from gen_zero.world_model.hamiltonian_dynamics import PyTorchHamiltonianNeuralODE
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestIssue86HamiltonianWorldModel(unittest.TestCase):
    """Test suite for Issue #86 Hamiltonian World Model and Symplectic Integrators."""

    def setUp(self):
        self.latent_dim = 64
        self.action_dim = 16
        self.model = HamiltonianWorldModel(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            dt=0.04,
            mass=1.0,
            potential_hidden_dim=32,
            omega=0.8,
            seed=42,
        )

    def test_client_mounts_hamiltonian_model_and_uses_predicted_state(self):
        cfg = GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=2)
        cfg.use_hamiltonian_dynamics = True
        client = GenZero(cfg)
        self.assertIsInstance(client.mcts_engine.hamiltonian_model, HamiltonianWorldModel)
        state = np.array([0.5, -0.3, 0.2, 0.4], dtype=np.float64)
        predicted = []
        original_step = client.hamiltonian_model.step

        def recording_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            predicted.append(result.next_state)
            return result

        evaluated = []
        with patch.object(client.hamiltonian_model, "step", side_effect=recording_step):
            result = client.mcts_engine.plan(
                root_state=state,
                candidate_actions=["advance"],
                transition_fn=lambda s, a: (s.copy(), 0.0, False),
                legal_actions_fn=lambda s: ["advance"],
                reward_fn=lambda parent, action, leaf: evaluated.append(leaf.copy()) or float(leaf[0]),
            )
        self.assertEqual(result["best_action"], "advance")
        self.assertGreater(len(predicted), 0)
        np.testing.assert_allclose(evaluated[0], predicted[0])
        self.assertFalse(np.array_equal(evaluated[0], state))

    def test_hamiltonian_failure_is_not_silenced(self):
        client = GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=1),
                         hamiltonian_model=self.model)
        with self.assertRaisesRegex(TypeError, "numpy.ndarray"):
            client.mcts_engine.plan(
                root_state="not a latent state",
                candidate_actions=["advance"],
                transition_fn=lambda s, a: (s, 0.0, False),
                legal_actions_fn=lambda s: ["advance"],
            )

    def test_01_canonical_symplectic_decomposition(self):
        """Validates canonical coordinate splitting and exact reconstruction."""
        rng = np.random.RandomState(123)
        z = rng.randn(self.latent_dim).astype(np.float64)
        q, p = self.model.split_canonical_coordinates(z)

        self.assertEqual(len(q), self.latent_dim // 2)
        self.assertEqual(len(p), self.latent_dim // 2)
        reconstructed = self.model.join_canonical_coordinates(q, p)
        np.testing.assert_allclose(z, reconstructed, atol=1e-15)

    def test_02_potential_energy_analytical_gradients(self):
        """Validates that analytical gradients of V_theta(q) match finite differences."""
        pot = self.model.potential
        rng = np.random.RandomState(456)
        q = rng.randn(self.model.dim_q).astype(np.float64) * 0.5

        _, analytical_grad = pot.forward(q)

        # Finite difference gradient
        eps = 1e-6
        numerical_grad = np.zeros_like(q)
        for i in range(len(q)):
            q_plus = q.copy()
            q_plus[i] += eps
            v_plus, _ = pot.forward(q_plus)

            q_minus = q.copy()
            q_minus[i] -= eps
            v_minus, _ = pot.forward(q_minus)

            numerical_grad[i] = (v_plus - v_minus) / (2.0 * eps)

        np.testing.assert_allclose(
            analytical_grad,
            numerical_grad,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Analytical gradient deviates from finite difference.",
        )

    def test_03_stormer_verlet_symplectic_volume_preservation(self):
        """Validates that the Störmer-Verlet update map has Jacobian determinant det(J) = 1.0000."""
        # For a small canonical system (dim_q = 2, latent_dim = 4), numerically compute Jacobian
        micro_model = HamiltonianWorldModel(
            latent_dim=4,
            action_dim=2,
            dt=0.02,
            mass=1.0,
            potential_hidden_dim=8,
            omega=0.5,
            seed=789,
        )

        z0 = np.array([0.5, -0.3, 0.2, 0.4], dtype=np.float64)
        dim = len(z0)
        eps = 1e-6

        # Construct Jacobian matrix numerically: J_ij = d(z_next_i) / d(z0_j)
        jacobian = np.zeros((dim, dim), dtype=np.float64)
        for j in range(dim):
            z_plus = z0.copy()
            z_plus[j] += eps
            res_plus = micro_model.step(z_plus)

            z_minus = z0.copy()
            z_minus[j] -= eps
            res_minus = micro_model.step(z_minus)

            jacobian[:, j] = (res_plus.next_state - res_minus.next_state) / (2.0 * eps)

        det_j = float(np.linalg.det(jacobian))
        print(f"\n[Test Symplectic Volume] det(Jacobian) = {det_j:.6f}")
        # Symplectic transformation preserves phase space volume: det(J) == 1.0000
        self.assertAlmostEqual(det_j, 1.0, places=4, msg="Symplectic phase volume was not preserved!")

    def test_04_long_horizon_energy_conservation(self):
        """Validates that relative energy drift over H=100 steps is <= 1.5%."""
        rng = np.random.RandomState(999)
        z0 = rng.randn(self.latent_dim).astype(np.float64) * 0.5

        # 100-step free closed-loop simulation
        rollout_res = self.model.rollout(
            initial_state=z0,
            horizon=100,
            actions=None,  # Autonomous Hamiltonian evolution
        )

        drift_pct = rollout_res.max_energy_drift_ratio * 100.0
        print(f"\n[Test Energy Drift] H=100 Max Energy Drift: {drift_pct:.4f}% (Threshold: <= 1.50%)")
        self.assertLessEqual(
            rollout_res.max_energy_drift_ratio,
            0.015,
            f"Max energy drift {drift_pct:.4f}% exceeded 1.5% requirement!",
        )
        self.assertTrue(rollout_res.is_stable)

    def test_05_bounded_phase_space_no_divergence(self):
        """Validates that latent state norms remain bounded and divergence rate is 0.00%."""
        rng = np.random.RandomState(101)
        z0 = rng.randn(self.latent_dim).astype(np.float64) * 0.5

        rollout_res = self.model.rollout(
            initial_state=z0,
            horizon=150,
            actions=None,
            stability_norm_threshold=50.0,
        )

        self.assertEqual(rollout_res.divergence_rate, 0.00)
        self.assertLess(rollout_res.max_state_norm, 10.0)

    def test_06_sub_millisecond_step_latency(self):
        """Validates that single-step symplectic update latency is <= 0.40 ms."""
        z = np.ones(self.latent_dim, dtype=np.float64) * 0.2

        # Warmup
        for _ in range(50):
            res = self.model.step(z)
            z = res.next_state

        n_steps = 1000
        t0 = time.perf_counter()
        for _ in range(n_steps):
            res = self.model.step(z)
            z = res.next_state
        t1 = time.perf_counter()

        avg_latency_ms = ((t1 - t0) * 1000.0) / n_steps
        print(f"\n[Test Latency] Average symplectic step latency: {avg_latency_ms:.4f} ms")
        self.assertLessEqual(avg_latency_ms, 0.40, f"Step latency {avg_latency_ms:.4f} ms > 0.40 ms")

    def test_07_action_control_force_integration(self):
        """Validates that external action forces appropriately inject work into the system."""
        z0 = np.zeros(self.latent_dim, dtype=np.float64)
        action_names = ["ACCELERATE", "STABILIZE", "ROTATE", "DECELERATE"]

        curr = z0
        energies = []
        for act in action_names:
            res = self.model.step(curr, action=act)
            curr = res.next_state
            energies.append(res.hamiltonian_energy)

        # System starts from resting state (H=0); action force injects non-zero energy
        self.assertGreater(energies[-1], 0.0)
        self.assertFalse(np.allclose(curr, z0))

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for autograd Hamiltonian Neural ODE")
    def test_08_pytorch_hamiltonian_ode(self):
        """Validates PyTorch autograd differentiable Hamiltonian Neural ODE."""
        ode = PyTorchHamiltonianNeuralODE(latent_dim=32, action_dim=8, hidden_dim=16)
        z = torch.randn(2, 32, requires_grad=True)
        act = torch.randn(2, 8)

        z_next = ode.forward_symplectic_step(z, action=act, dt=0.05)
        self.assertEqual(z_next.shape, (2, 32))

        # Test backpropagation through symplectic step
        loss = z_next.sum()
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertFalse(torch.isnan(z.grad).any())

    def test_09_zero_word_compliance(self):
        """Validates zero hits for historical deprecated keywords."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        files_to_check = [
            os.path.join(repo_root, "gen_zero/world_model/hamiltonian_dynamics.py"),
            __file__,
        ]
        forbidden_token = "j" + "e" + "v"
        pattern = re.compile(rf"{forbidden_token}", re.IGNORECASE)
        for fpath in files_to_check:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            # Exclude current compliance check function
            body = content.split("def test_09_zero_word_compliance")[0]
            matches = pattern.findall(body)
            self.assertEqual(len(matches), 0, f"Found forbidden matches in {fpath}: {matches}")


if __name__ == "__main__":
    unittest.main()
