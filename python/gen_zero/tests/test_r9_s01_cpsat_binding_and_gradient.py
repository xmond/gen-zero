"""R9-S01: CP-SAT verdict is bound to the output action; backward is a contraction."""

import unittest
from unittest import mock

import numpy as np

from gen_zero.gate.cpsat_formal_solver import CPSATVerdict
from gen_zero.gate.differentiable_safety_layer import DifferentiableSafetyLayer

A = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
B = np.array([0.0, 0.0])


def _verdict(action, status, fallback=False):
    return CPSATVerdict(action, True, 0.1, False, fallback, status, [])


class TestCpsatBinding(unittest.TestCase):
    def test_no_constraints_reports_not_invoked(self):
        res = DifferentiableSafetyLayer(4).forward(np.full(4, 0.25))
        self.assertFalse(res.cpsat_hard_verified)
        self.assertEqual(res.cpsat_solver_status, "NOT_INVOKED")
        self.assertIsNone(res.cpsat_selected_action)

    def test_verified_only_when_cpsat_picks_output_top1(self):
        layer = DifferentiableSafetyLayer(4, A, B)
        x0 = np.array([0.6, 0.2, 0.1, 0.1])
        with mock.patch.object(layer.cpsat_solver, "solve_safest_optimal_action",
                               return_value=_verdict("ACT_0", "OPTIMAL")):
            self.assertTrue(layer.forward(x0).cpsat_hard_verified)
        with mock.patch.object(layer.cpsat_solver, "solve_safest_optimal_action",
                               return_value=_verdict("ACT_1", "OPTIMAL")):
            res = layer.forward(x0)
            self.assertFalse(res.cpsat_hard_verified)
            self.assertEqual(res.cpsat_selected_action, "ACT_1")

    def test_shortcut_and_fallback_are_not_verified(self):
        layer = DifferentiableSafetyLayer(4, A, B)
        x0 = np.array([0.6, 0.2, 0.1, 0.1])
        for status, fallback in (("DETERMINISTIC_SAFE_SOLVED", False),
                                 ("ORTOOLS_UNAVAILABLE_FALLBACK", True),
                                 ("SOLVER_TIMEOUT_FALLBACK", True)):
            with mock.patch.object(layer.cpsat_solver, "solve_safest_optimal_action",
                                   return_value=_verdict("ACT_0", status, fallback)):
                res = layer.forward(x0)
                self.assertFalse(res.cpsat_hard_verified, status)
                self.assertEqual(res.cpsat_solver_status, status)


class TestBackwardContraction(unittest.TestCase):
    def test_projected_gradient_norm_never_exceeds_upstream(self):
        rng = np.random.RandomState(9)
        layer = DifferentiableSafetyLayer(4, A, B)
        for _ in range(200):
            x0 = rng.dirichlet(np.ones(4))
            layer.forward(x0)
            g_out = rng.normal(size=4) * rng.uniform(0.1, 100.0)
            g_in = layer.backward(g_out)
            self.assertLessEqual(np.linalg.norm(g_in), np.linalg.norm(g_out) * (1 + 1e-12))

    def test_gradient_can_be_zero(self):
        layer = DifferentiableSafetyLayer(4, A, B)
        layer.forward(np.array([0.5, 0.5, 0.0, 0.0]))
        # A uniform upstream gradient is normal to the simplex face.
        self.assertTrue(np.allclose(layer.backward(np.ones(4)), 0.0))


if __name__ == "__main__":
    unittest.main()
