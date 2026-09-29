"""Unit tests for the LatentUpdater zero-token latent-space reasoning module.

Covers:
1. Shape consistency (batched and single-sample, torch and NumPy paths).
2. Convergence behavior (early stop vs. full budget exhaustion).
3. Damping effect of alpha.
4. Telemetry completeness.
5. Zero-token (G=0) / reflex-depth (R>0) contract.
6. Integration with SubSecondDecisionPipeline.process_tick.
"""

import unittest

import numpy as np
import torch

from gen_zero.model.latent_updater import (
    LatentUpdater,
    LatentUpdaterTelemetry,
    NumpyLatentUpdater,
    HAS_TORCH,
)
from gen_zero.model.market_state import L2OrderBookSnapshot
from gen_zero.runtime.subsecond_pipeline import SubSecondDecisionPipeline


class TestShapeConsistency(unittest.TestCase):
    """Task 4.1: output shape matches (batch, latent_dim); torch and NumPy paths."""

    def test_torch_batched_shape(self):
        self.assertTrue(HAS_TORCH)
        updater = LatentUpdater(input_dim=3, latent_dim=8, T=4)
        x = torch.randn(5, 3)
        z, telemetry = updater(x)
        self.assertEqual(tuple(z.shape), (5, 8))
        self.assertIsInstance(telemetry, LatentUpdaterTelemetry)

    def test_torch_single_sample_shape(self):
        updater = LatentUpdater(input_dim=3, latent_dim=8, T=4)
        x_1d = torch.randn(3)
        z_1d, _ = updater(x_1d)
        self.assertEqual(tuple(z_1d.shape), (8,))

        x_2d = torch.randn(1, 3)
        z_2d, _ = updater(x_2d)
        self.assertEqual(tuple(z_2d.shape), (1, 8))

    def test_torch_accepts_array_like_input(self):
        updater = LatentUpdater(input_dim=2, latent_dim=4, T=3)
        z, _ = updater([0.1, -0.2])
        self.assertEqual(tuple(z.shape), (4,))

    def test_numpy_fallback_batched_shape(self):
        """Directly exercises the NumPy code path regardless of torch availability."""
        updater = NumpyLatentUpdater(input_dim=3, latent_dim=8, T=4)
        x = np.random.randn(5, 3)
        z, telemetry = updater(x)
        self.assertEqual(z.shape, (5, 8))
        self.assertIsInstance(z, np.ndarray)
        self.assertIsInstance(telemetry, LatentUpdaterTelemetry)

    def test_numpy_fallback_single_sample_shape(self):
        updater = NumpyLatentUpdater(input_dim=3, latent_dim=8, T=4)
        x_1d = np.random.randn(3)
        z_1d, _ = updater(x_1d)
        self.assertEqual(z_1d.shape, (8,))

        x_2d = np.random.randn(1, 3)
        z_2d, _ = updater(x_2d)
        self.assertEqual(z_2d.shape, (1, 8))


class TestConstructorValidation(unittest.TestCase):
    """Fail-closed: invalid constructor args raise ValueError immediately."""

    def test_alpha_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            LatentUpdater(input_dim=2, latent_dim=4, alpha=0.0)
        with self.assertRaises(ValueError):
            LatentUpdater(input_dim=2, latent_dim=4, alpha=1.5)
        with self.assertRaises(ValueError):
            NumpyLatentUpdater(input_dim=2, latent_dim=4, alpha=-0.1)

    def test_non_positive_T_raises(self):
        with self.assertRaises(ValueError):
            LatentUpdater(input_dim=2, latent_dim=4, T=0)
        with self.assertRaises(ValueError):
            NumpyLatentUpdater(input_dim=2, latent_dim=4, T=-1)

    def test_non_positive_epsilon_raises(self):
        with self.assertRaises(ValueError):
            LatentUpdater(input_dim=2, latent_dim=4, epsilon=0.0)
        with self.assertRaises(ValueError):
            NumpyLatentUpdater(input_dim=2, latent_dim=4, epsilon=-1e-4)

    def test_non_positive_dims_raise(self):
        with self.assertRaises(ValueError):
            LatentUpdater(input_dim=0, latent_dim=4)
        with self.assertRaises(ValueError):
            LatentUpdater(input_dim=2, latent_dim=-3)


class TestConvergenceBehavior(unittest.TestCase):
    """Task 4.2: early stop below epsilon vs. exhausting the full budget T."""

    def test_converges_before_budget_exhausted(self):
        torch.manual_seed(0)
        updater = LatentUpdater(input_dim=2, latent_dim=4, T=6, epsilon=1e6)
        x = torch.tensor([0.1, -0.2])
        _, telemetry = updater(x)
        self.assertTrue(telemetry.converged)
        self.assertLess(telemetry.n_iterations, updater.T)

    def test_exhausts_budget_without_converging(self):
        torch.manual_seed(0)
        updater = LatentUpdater(input_dim=2, latent_dim=4, T=6, epsilon=1e-12)
        x = torch.tensor([0.1, -0.2])
        _, telemetry = updater(x)
        self.assertFalse(telemetry.converged)
        self.assertEqual(telemetry.n_iterations, updater.T)

    def test_numpy_converges_before_budget_exhausted(self):
        updater = NumpyLatentUpdater(input_dim=2, latent_dim=4, T=6, epsilon=1e6)
        x = np.array([0.1, -0.2])
        _, telemetry = updater(x)
        self.assertTrue(telemetry.converged)
        self.assertLess(telemetry.n_iterations, updater.T)

    def test_numpy_exhausts_budget_without_converging(self):
        updater = NumpyLatentUpdater(input_dim=2, latent_dim=4, T=6, epsilon=1e-12)
        x = np.array([0.1, -0.2])
        _, telemetry = updater(x)
        self.assertFalse(telemetry.converged)
        self.assertEqual(telemetry.n_iterations, updater.T)


class TestDampingEffect(unittest.TestCase):
    """Task 4.3: a smaller alpha damps the per-step residual more than a larger one."""

    def test_smaller_alpha_yields_smaller_first_step_residual_torch(self):
        torch.manual_seed(42)
        updater = LatentUpdater(input_dim=2, latent_dim=6, T=1, epsilon=1e-12, alpha=0.05)
        x = torch.tensor([0.3, 0.4])

        _, telemetry_small = updater(x)
        residual_small = telemetry_small.residual_history[0]

        # Same weights, larger alpha applied to the same first step.
        updater.alpha = 0.95
        _, telemetry_large = updater(x)
        residual_large = telemetry_large.residual_history[0]

        self.assertLess(residual_small, residual_large)

    def test_smaller_alpha_yields_smaller_first_step_residual_numpy(self):
        updater = NumpyLatentUpdater(input_dim=2, latent_dim=6, T=1, epsilon=1e-12, alpha=0.05)
        x = np.array([0.3, 0.4])

        _, telemetry_small = updater(x)
        residual_small = telemetry_small.residual_history[0]

        updater.alpha = 0.95
        _, telemetry_large = updater(x)
        residual_large = telemetry_large.residual_history[0]

        self.assertLess(residual_small, residual_large)


class TestTelemetryCompleteness(unittest.TestCase):
    """Task 4.4: telemetry fields are present, non-negative, and consistent."""

    def test_telemetry_fields_present_and_sane(self):
        torch.manual_seed(1)
        updater = LatentUpdater(input_dim=2, latent_dim=4, T=5, epsilon=1e-4)
        x = torch.tensor([0.2, -0.1])
        _, telemetry = updater(x)

        self.assertIsInstance(telemetry.n_iterations, int)
        self.assertIsInstance(telemetry.final_residual, float)
        self.assertIsInstance(telemetry.converged, bool)
        self.assertIsInstance(telemetry.residual_history, list)

        self.assertEqual(len(telemetry.residual_history), telemetry.n_iterations)
        self.assertTrue(all(r >= 0.0 for r in telemetry.residual_history))
        self.assertAlmostEqual(
            telemetry.residual_history[-1], telemetry.final_residual, places=8,
        )


class TestZeroTokenContract(unittest.TestCase):
    """Task 4.5: never emits token/text output; R>0 and G=0 are explicit."""

    def test_public_api_returns_only_tensor_and_telemetry(self):
        torch.manual_seed(2)
        updater = LatentUpdater(input_dim=2, latent_dim=4, T=3)
        x = torch.tensor([0.1, 0.1])
        outputs = updater(x)

        self.assertEqual(len(outputs), 2)
        z, telemetry = outputs
        self.assertIsInstance(z, torch.Tensor)
        self.assertIsInstance(telemetry, LatentUpdaterTelemetry)
        # No string/token-id output anywhere in the return value.
        self.assertNotIsInstance(z, str)
        for field_value in (telemetry.n_iterations, telemetry.final_residual,
                            telemetry.converged, telemetry.residual_history,
                            telemetry.zero_token, telemetry.reflex_depth):
            self.assertNotIsInstance(field_value, str)

    def test_reflex_depth_and_zero_token_flag(self):
        torch.manual_seed(3)
        updater = LatentUpdater(input_dim=2, latent_dim=4, T=3)
        x = torch.tensor([0.1, 0.1])
        _, telemetry = updater(x)

        self.assertTrue(telemetry.zero_token is True)
        self.assertGreaterEqual(telemetry.reflex_depth, 1)
        self.assertGreaterEqual(telemetry.n_iterations, 1)
        self.assertEqual(telemetry.reflex_depth, telemetry.n_iterations)


class TestPipelineIntegration(unittest.TestCase):
    """Task 4.6: LatentUpdater wired into SubSecondDecisionPipeline.process_tick."""

    def setUp(self):
        self.snap = L2OrderBookSnapshot(
            bids=[(100.0, 30.0), (99.95, 20.0)],
            asks=[(100.05, 10.0), (100.10, 10.0)],
            cvd=25.0,
            last_price=100.02,
        )

    def test_with_latent_updater_adds_stage_and_telemetry(self):
        torch.manual_seed(7)
        updater = LatentUpdater(input_dim=2, latent_dim=4, T=3)
        pipeline = SubSecondDecisionPipeline(safety_deadline_ms=100.0, latent_updater=updater)

        frame = pipeline.process_tick(self.snap)

        self.assertIn("latent_update", frame["stage_breakdown_ms"])
        self.assertIn("latent_update", frame)
        telem = frame["latent_update"]
        self.assertIn("n_iterations", telem)
        self.assertIn("final_residual", telem)
        self.assertIn("converged", telem)

        # Generous bound to avoid flakiness on shared/noisy CI runners.
        self.assertLess(frame["stage_breakdown_ms"]["latent_update"], 20.0)

    def test_without_latent_updater_matches_prior_contract(self):
        pipeline = SubSecondDecisionPipeline(safety_deadline_ms=100.0)
        frame = pipeline.process_tick(self.snap)

        self.assertNotIn("latent_update", frame["stage_breakdown_ms"])
        self.assertNotIn("latent_update", frame)

        # Existing contract: same top-level keys as before this change.
        expected_keys = {
            "tick_id", "timestamp_ms", "action", "probabilities", "confidence",
            "latency_ms", "late", "stage_breakdown_ms", "risk_guard",
        }
        self.assertEqual(set(frame.keys()), expected_keys)
        self.assertEqual(
            set(frame["stage_breakdown_ms"].keys()), {"ingest", "scoring", "risk_gate"},
        )


if __name__ == "__main__":
    unittest.main()
