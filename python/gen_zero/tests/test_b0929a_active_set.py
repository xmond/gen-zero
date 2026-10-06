"""Forward-only finite-difference regressions for activity at a strict simplex interior."""

import unittest

import numpy as np

from gen_zero.gate.differentiable_safety_layer import DifferentiableSafetyLayer


class TestActiveSetScale(unittest.TestCase):
    def check_gradient(self, x0, scale, step):
        layer = DifferentiableSafetyLayer(
            2, np.array([[scale, 0.0]]), np.array([scale])
        )
        utility = np.zeros(2)
        result = layer.forward(x0, utility)
        self.assertEqual(result.active_constraints_count, 0)
        self.assertEqual(layer._cached_lambda[0], 0.0)
        numeric = np.empty(2)
        for j in range(2):
            delta = np.zeros(2)
            delta[j] = step
            plus = layer.forward(x0, utility + delta).projected_distribution[0]
            minus = layer.forward(x0, utility - delta).projected_distribution[0]
            numeric[j] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(numeric, [0.5, -0.5], atol=1e-4, rtol=0)
        return numeric

    def test_r5_s01_near_interior(self):
        self.check_gradient(np.array([0.99999995, 0.00000005]), 1.0, 1e-9)

    def test_r5_s02_equivalent_constraint_scales(self):
        x0 = np.array([0.9999, 0.0001])
        reference = self.check_gradient(x0, 1.0, 1e-7)
        for scale in (1e-4, 1e2):
            np.testing.assert_allclose(
                self.check_gradient(x0, scale, 1e-7), reference, atol=1e-12, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
