"""R6-S01: rounded primal contact must not activate a zero-dual inequality."""

import numpy as np
import pytest

from gen_zero.gate.differentiable_safety_layer import DifferentiableSafetyLayer


def _layer():
    a = np.array([[1.0, 1.0 - 2**-14]], dtype=np.float64)
    b = np.array([1.0], dtype=np.float64)
    return DifferentiableSafetyLayer(2, a, b)


def _first_output(layer, proposal):
    return layer.forward(proposal, np.zeros(2)).projected_distribution[0]


def test_rounded_strict_interior_has_unconstrained_simplex_response():
    x0 = np.array([1.0 - 2**-39, 2**-39], dtype=np.float64)
    layer = _layer()
    result = layer.forward(x0, np.zeros(2))
    assert layer._cached_lambda[0] == 0.0
    assert (layer.b_sat - layer.a_sat @ result.projected_distribution)[0] == 0.0
    assert result.active_constraints_count == 0

    # Stay inside the local face: x0[1] is only 2**-39 from the simplex edge.
    h = 2**-41
    finite_difference = np.array([
        (_first_output(_layer(), x0 + h * np.eye(2)[i])
         - _first_output(_layer(), x0 - h * np.eye(2)[i])) / (2 * h)
        for i in range(2)
    ])
    np.testing.assert_allclose(finite_difference, [0.5, -0.5], atol=1e-6, rtol=0)


@pytest.mark.parametrize("second_coordinate", [2**-39, 2**-30, 1e-5])
def test_zero_dual_never_activates_at_or_near_contact(second_coordinate):
    x0 = np.array([1.0 - second_coordinate, second_coordinate], dtype=np.float64)
    layer = _layer()
    result = layer.forward(x0)
    assert layer._cached_lambda[0] == 0.0
    assert result.active_constraints_count == 0
