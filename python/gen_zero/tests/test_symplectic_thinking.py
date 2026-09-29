"""Unit tests for symplectic thinking: leapfrog integrator, neural potential gradient, Lyapunov exit.

Verifies:
1. Analytic potential gradient matches central finite differences at dim 2048.
2. Leapfrog is symplectic (J^T Omega J = Omega) and time-reversible when gamma = 0.
3. Undamped energy drift is bounded and shrinks ~4x when dt is halved (O(dt^2)).
4. Damped rollouts make H non-increasing and exit early (Converged / Stalled).
5. Bad steps are caught (LyapunovViolation, NonFinite) and bad inputs are rejected.
6. No private file-system paths leak into the module or this test.
"""

import pathlib
import unittest

import numpy as np

from gen_zero.causal.symplectic_thinking import (
    LATENT_DIM,
    ExitReason,
    NeuralPotential,
    SymplecticThinker,
)


class TestNeuralPotential(unittest.TestCase):
    def test_01_gradient_matches_finite_differences(self):
        pot = NeuralPotential(dim=LATENT_DIM, hidden_dim=128, seed=1)
        rng = np.random.default_rng(2)
        q = rng.normal(size=LATENT_DIM)
        _, grad = pot.energy_and_grad(q)
        eps = 1e-6
        for i in rng.choice(LATENT_DIM, size=12, replace=False):
            e = np.zeros(LATENT_DIM)
            e[i] = eps
            numeric = (pot.energy(q + e) - pot.energy(q - e)) / (2 * eps)
            self.assertAlmostEqual(grad[i], numeric, places=6)

    def test_02_extreme_inputs_stay_finite(self):
        pot = NeuralPotential(dim=64, hidden_dim=32, seed=3)
        v, g = pot.energy_and_grad(np.full(64, 1e4))
        self.assertTrue(np.isfinite(v) and np.all(np.isfinite(g)))

    def test_03_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            NeuralPotential(dim=0)
        with self.assertRaises(ValueError):
            NeuralPotential(omega=0.0)


class TestLeapfrog(unittest.TestCase):
    def test_04_default_state_is_2048_dim(self):
        thinker = SymplecticThinker()
        self.assertEqual(thinker.dim, 2048)
        q, p, g = thinker.step(np.zeros(2048), np.zeros(2048))
        self.assertEqual((q.shape, p.shape, g.shape), ((2048,),) * 3)

    def test_05_symplectic_jacobian(self):
        d = 4
        thinker = SymplecticThinker(NeuralPotential(dim=d, hidden_dim=8, seed=4), dt=0.1, gamma=0.0)
        rng = np.random.default_rng(5)
        z0 = rng.normal(size=2 * d)

        def flow(z):
            q, p, _ = thinker.step(z[:d], z[d:])
            return np.concatenate([q, p])

        eps = 1e-6
        jac = np.zeros((2 * d, 2 * d))
        for j in range(2 * d):
            e = np.zeros(2 * d)
            e[j] = eps
            jac[:, j] = (flow(z0 + e) - flow(z0 - e)) / (2 * eps)
        omega = np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])
        np.testing.assert_allclose(jac.T @ omega @ jac, omega, atol=1e-7)

    def test_06_time_reversible(self):
        thinker = SymplecticThinker(NeuralPotential(hidden_dim=64, seed=6), dt=0.05, gamma=0.0)
        rng = np.random.default_rng(7)
        q0, p0 = rng.normal(size=2048), rng.normal(size=2048)
        q, p = q0, p0
        for _ in range(50):
            q, p, _ = thinker.step(q, p)
        p = -p
        for _ in range(50):
            q, p, _ = thinker.step(q, p)
        np.testing.assert_allclose(q, q0, atol=1e-9)
        np.testing.assert_allclose(-p, p0, atol=1e-9)

    def test_07_energy_drift_is_second_order(self):
        pot = NeuralPotential(hidden_dim=64, seed=8)
        rng = np.random.default_rng(9)
        q0 = rng.normal(size=2048) * 0.2
        p0 = rng.normal(size=2048) * 0.2

        def max_drift(dt):
            thinker = SymplecticThinker(pot, dt=dt, gamma=0.0)
            q, p = q0, p0
            h0 = thinker.hamiltonian(q, p)
            worst = 0.0
            for _ in range(int(2.0 / dt)):
                q, p, _ = thinker.step(q, p)
                worst = max(worst, abs(thinker.hamiltonian(q, p) - h0))
            return worst / abs(h0)

        coarse, fine = max_drift(0.1), max_drift(0.05)
        self.assertLess(coarse, 1e-2)
        self.assertGreater(coarse / fine, 3.0)  # ideal ratio is 4


class TestLyapunovExit(unittest.TestCase):
    def test_08_damped_energy_is_non_increasing_and_exits_early(self):
        thinker = SymplecticThinker(NeuralPotential(hidden_dim=64, seed=10), dt=0.05,
                                    gamma=1.0, max_steps=2000)
        rng = np.random.default_rng(11)
        result = thinker.think(rng.normal(size=2048), rng.normal(size=2048))
        self.assertTrue(result.early_exit)
        self.assertIn(result.exit_reason, (ExitReason.Converged, ExitReason.Stalled))
        self.assertLess(result.steps, 2000)
        self.assertLess(result.final_energy, result.initial_energy)
        trace = np.array(result.energy_trace)
        self.assertTrue(np.all(np.diff(trace) <= 1e-6 * np.maximum(1.0, np.abs(trace[:-1]))))
        self.assertEqual(len(trace), result.steps + 1)

    def test_09_converged_state_is_an_equilibrium(self):
        # stall_tol=0 disables the Stalled exit so the equilibrium test decides.
        thinker = SymplecticThinker(NeuralPotential(hidden_dim=64, seed=12), dt=0.05,
                                    gamma=1.0, max_steps=3000, stall_tol=0.0)
        result = thinker.think(np.random.default_rng(13).normal(size=2048))
        self.assertEqual(result.exit_reason, ExitReason.Converged)
        self.assertLessEqual(np.linalg.norm(thinker.potential.grad(result.q)), 1e-4)
        self.assertLessEqual(np.linalg.norm(result.p), 1e-4)

    def test_09b_stalled_exit_is_near_equilibrium(self):
        thinker = SymplecticThinker(NeuralPotential(hidden_dim=64, seed=12), dt=0.05,
                                    gamma=1.0, max_steps=3000)
        result = thinker.think(np.random.default_rng(13).normal(size=2048))
        self.assertEqual(result.exit_reason, ExitReason.Stalled)
        self.assertLessEqual(np.linalg.norm(thinker.potential.grad(result.q)), 1e-2)
        self.assertLessEqual(np.linalg.norm(result.p), 1e-2)

    def test_10_undamped_runs_full_budget(self):
        thinker = SymplecticThinker(NeuralPotential(hidden_dim=64, seed=14), dt=0.05,
                                    gamma=0.0, max_steps=40)
        rng = np.random.default_rng(15)
        result = thinker.think(rng.normal(size=2048), rng.normal(size=2048))
        self.assertEqual(result.exit_reason, ExitReason.MaxSteps)
        self.assertEqual(result.steps, 40)
        self.assertFalse(result.early_exit)

    def test_11_unstable_step_size_is_caught(self):
        # dt * omega > 2 makes leapfrog unstable; the energy blows up.
        thinker = SymplecticThinker(NeuralPotential(hidden_dim=16, omega=2.0, seed=16), dt=1.5,
                                    gamma=0.1, max_steps=500)
        result = thinker.think(np.random.default_rng(17).normal(size=2048))
        self.assertIn(result.exit_reason, (ExitReason.LyapunovViolation, ExitReason.NonFinite))
        self.assertLess(result.steps, 500)

    def test_12_non_finite_input_exits_at_once(self):
        thinker = SymplecticThinker(NeuralPotential(dim=8, hidden_dim=4, seed=18))
        q = np.zeros(8)
        q[0] = np.nan
        result = thinker.think(q)
        self.assertEqual(result.exit_reason, ExitReason.NonFinite)
        self.assertEqual(result.steps, 0)

    def test_13_input_validation_and_to_dict(self):
        thinker = SymplecticThinker(NeuralPotential(dim=8, hidden_dim=4, seed=19), max_steps=5)
        with self.assertRaises(ValueError):
            thinker.think(np.zeros(7))
        with self.assertRaises(ValueError):
            thinker.think(np.zeros(8), np.zeros(9))
        with self.assertRaises(ValueError):
            SymplecticThinker(dt=0.0)
        with self.assertRaises(ValueError):
            SymplecticThinker(gamma=-1.0)
        report = thinker.think(np.ones(8)).to_dict()
        self.assertEqual(report["max_steps"], 5)
        self.assertIn(report["exit_reason"], {r.value for r in ExitReason})

    def test_14_deterministic(self):
        q0 = np.random.default_rng(20).normal(size=2048)
        runs = [SymplecticThinker(NeuralPotential(hidden_dim=32, seed=21)).think(q0) for _ in range(2)]
        np.testing.assert_array_equal(runs[0].q, runs[1].q)
        self.assertEqual(runs[0].steps, runs[1].steps)


class TestNoPrivatePathLeaks(unittest.TestCase):
    def test_15_no_private_paths(self):
        here = pathlib.Path(__file__).resolve()
        module = here.parent.parent / "causal" / "symplectic_thinking.py"
        # Needles are assembled so this file does not match itself.
        needles = ["/" + "home/", "/" + "ebs/", "/" + "Users/", "C:" + "\\", "/" + "root/", "/" + "mnt/"]
        for path in (module, here):
            text = path.read_text(encoding="utf-8")
            for needle in needles:
                self.assertNotIn(needle, text, f"{needle!r} leaked in {path.name}")


if __name__ == "__main__":
    unittest.main()
