"""Unit tests for continuous-time Koopman latent thinking.

Checks:
1. exp(t * A) matches closed forms and the semigroup law.
2. The generator is recovered from snapshot and derivative data.
3. Multi-step lookahead matches the closed-form flow and stays latent-only.
4. Unstable modes are bounded on request.
5. 0 private path leaks in the module and in this test file.
"""

import os
import re
import unittest

import numpy as np

from gen_zero.causal.koopman_thinking import (
    KoopmanGenerator,
    KoopmanThinker,
    expm,
    fit_generator_from_derivatives,
    fit_generator_from_snapshots,
    logm_real,
)

OMEGA = 0.7
ROTATION_GEN = np.array([[0.0, -OMEGA], [OMEGA, 0.0]])


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def damped_system(dim: int = 6, seed: int = 0) -> np.ndarray:
    """Random stable generator: skew-symmetric part minus a positive diagonal."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dim, dim))
    return 0.5 * (raw - raw.T) - np.diag(rng.uniform(0.05, 0.3, size=dim))


class TestMatrixExponential(unittest.TestCase):
    def test_zero_matrix_gives_identity(self):
        np.testing.assert_allclose(expm(np.zeros((4, 4))), np.eye(4), atol=1e-14)

    def test_rotation_closed_form(self):
        for t in (0.1, 1.0, 5.0, 40.0):
            np.testing.assert_allclose(expm(t * ROTATION_GEN), rotation(OMEGA * t), atol=1e-10)

    def test_diagonal_matches_scalar_exp(self):
        d = np.array([-1.5, 0.0, 0.3, 2.0])
        np.testing.assert_allclose(expm(np.diag(d)), np.diag(np.exp(d)), rtol=1e-12)

    def test_semigroup_law(self):
        a = damped_system()
        s, t = 0.37, 1.21
        np.testing.assert_allclose(expm((s + t) * a), expm(s * a) @ expm(t * a), atol=1e-11)

    def test_inverse_is_negative_time(self):
        a = damped_system()
        np.testing.assert_allclose(expm(a) @ expm(-a), np.eye(a.shape[0]), atol=1e-10)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            expm(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            expm(np.array([[np.nan, 0.0], [0.0, 1.0]]))


class TestMatrixLogarithm(unittest.TestCase):
    def test_log_inverts_exp(self):
        a = damped_system()
        np.testing.assert_allclose(logm_real(expm(a)), a, atol=1e-8)

    def test_negative_real_eigenvalue_is_rejected(self):
        with self.assertRaises(ValueError):
            logm_real(np.diag([-1.0, 2.0]))

    def test_singular_is_rejected(self):
        with self.assertRaises(ValueError):
            logm_real(np.zeros((3, 3)))


class TestGeneratorFitting(unittest.TestCase):
    def setUp(self):
        self.a = damped_system(dim=6, seed=3)
        rng = np.random.default_rng(11)
        self.z0 = rng.normal(size=(200, 6))
        self.dt = 0.1

    def test_recover_from_snapshots(self):
        z1 = self.z0 @ expm(self.dt * self.a).T
        fitted = fit_generator_from_snapshots(self.z0, z1, self.dt)
        np.testing.assert_allclose(fitted.matrix, self.a, atol=1e-7)

    def test_recover_from_derivatives(self):
        fitted = fit_generator_from_derivatives(self.z0, self.z0 @ self.a.T)
        np.testing.assert_allclose(fitted.matrix, self.a, atol=1e-9)

    def test_snapshot_noise_is_tolerated(self):
        rng = np.random.default_rng(5)
        z1 = self.z0 @ expm(self.dt * self.a).T + 1e-4 * rng.normal(size=self.z0.shape)
        fitted = fit_generator_from_snapshots(self.z0, z1, self.dt)
        self.assertLess(np.linalg.norm(fitted.matrix - self.a), 0.05)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            fit_generator_from_snapshots(self.z0, self.z0, dt=0.0)
        with self.assertRaises(ValueError):
            fit_generator_from_snapshots(self.z0, self.z0[:10], dt=0.1)
        with self.assertRaises(ValueError):
            fit_generator_from_derivatives(self.z0, self.z0, ridge=-1.0)


class TestKoopmanLookahead(unittest.TestCase):
    def setUp(self):
        self.thinker = KoopmanThinker(ROTATION_GEN)
        self.z0 = np.array([1.0, 0.0])

    def test_propagate_matches_closed_form(self):
        np.testing.assert_allclose(self.thinker.propagate(self.z0, 2.5), rotation(OMEGA * 2.5) @ self.z0, atol=1e-10)

    def test_multistep_matches_direct_jump(self):
        result = self.thinker.lookahead(self.z0, horizon=50, dt=0.2)
        self.assertEqual(result.states.shape, (51, 2))
        np.testing.assert_allclose(result.states[0], self.z0)
        for i in (1, 10, 50):
            np.testing.assert_allclose(result.states[i], self.thinker.propagate(self.z0, result.times[i]), atol=1e-9)

    def test_rotation_preserves_norm_over_long_horizon(self):
        result = self.thinker.lookahead(self.z0, horizon=1000, dt=0.05)
        norms = np.linalg.norm(result.states, axis=1)
        self.assertLess(np.max(np.abs(norms - 1.0)), 1e-8)

    def test_irregular_time_grid(self):
        grid = [0.0, 0.3, 0.31, 4.0]
        result = self.thinker.lookahead_at(self.z0, grid)
        for t, state in zip(grid, result.states):
            np.testing.assert_allclose(state, rotation(OMEGA * t) @ self.z0, atol=1e-10)

    def test_result_is_latent_only(self):
        result = self.thinker.lookahead(self.z0, horizon=3, dt=0.1)
        self.assertIsInstance(result.states, np.ndarray)
        self.assertEqual(result.states.dtype, np.float64)
        self.assertIsInstance(result.final_state, np.ndarray)

    def test_zero_horizon_returns_start_only(self):
        result = self.thinker.lookahead(self.z0, horizon=0, dt=0.1)
        self.assertEqual(result.states.shape, (1, 2))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.thinker.lookahead(self.z0, horizon=-1, dt=0.1)
        with self.assertRaises(ValueError):
            self.thinker.lookahead(self.z0, horizon=3, dt=0.0)
        with self.assertRaises(ValueError):
            self.thinker.lookahead(np.zeros(5), horizon=3, dt=0.1)
        with self.assertRaises(ValueError):
            self.thinker.lookahead_at(self.z0, [-1.0])

    def test_fit_then_think_end_to_end(self):
        a = damped_system(dim=8, seed=21)
        rng = np.random.default_rng(2)
        x0 = rng.normal(size=(300, 8))
        fitted = fit_generator_from_snapshots(x0, x0 @ expm(0.05 * a).T, dt=0.05)
        start = rng.normal(size=8)
        truth = KoopmanThinker(a).lookahead(start, horizon=100, dt=0.05).final_state
        pred = KoopmanThinker(fitted).lookahead(start, horizon=100, dt=0.05).final_state
        np.testing.assert_allclose(pred, truth, atol=1e-6)


class TestStability(unittest.TestCase):
    def test_spectral_abscissa(self):
        self.assertAlmostEqual(KoopmanGenerator(np.diag([-1.0, 0.4])).spectral_abscissa(), 0.4)

    def test_unstable_mode_is_clipped_by_think(self):
        gen = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        thinker = KoopmanThinker(gen)
        start = np.array([1.0, 1.0, 0.0])
        raw = thinker.lookahead(start, horizon=200, dt=0.1)
        bounded = thinker.think(start, horizon=200, dt=0.1)
        self.assertGreater(np.linalg.norm(raw.final_state), 1e3)
        self.assertLess(np.max(np.linalg.norm(bounded.states, axis=1)), 2.0)

    def test_stable_generator_is_left_unchanged(self):
        thinker = KoopmanThinker(damped_system())
        start = np.ones(6)
        np.testing.assert_allclose(
            thinker.think(start, 20, 0.1).states, thinker.lookahead(start, 20, 0.1).states
        )

    def test_generator_matrix_is_read_only(self):
        gen = KoopmanGenerator(np.eye(2))
        with self.assertRaises(ValueError):
            gen.matrix[0, 0] = 5.0


class TestPathLeakCompliance(unittest.TestCase):
    """0 private path leaks in the module and its test."""

    def test_no_private_paths(self):
        here = os.path.dirname(os.path.abspath(__file__))
        files = [
            os.path.join(here, "../causal/koopman_thinking.py"),
            os.path.join(here, "test_koopman_thinking.py"),
        ]
        # Fragments are split so this file does not match itself.
        markers = ["/ho" + "me/", "/eb" + "s/", "/Us" + "ers/", "C:" + "\\\\", "/ro" + "ot/", "/mn" + "t/"]
        pattern = re.compile("|".join(re.escape(m) for m in markers), re.IGNORECASE)
        for path in files:
            with open(path, encoding="utf-8") as handle:
                hits = pattern.findall(handle.read())
            self.assertEqual(hits, [], f"private path leak in {os.path.basename(path)}: {hits}")


if __name__ == "__main__":
    unittest.main()
