"""R8-S01: repeated rounded contacts cannot activate an interior constraint."""

import numpy as np
import pytest
import torch

from gen_zero.gate.differentiable_safety_layer import (
    DifferentiableSafetyLayer,
    PyTorchDifferentiableSafetyModule,
)


@pytest.mark.parametrize("max_iter", [200, 500])
def test_accumulated_dual_noise_preserves_free_simplex_response(max_iter):
    layer = DifferentiableSafetyLayer(
        2,
        np.array([[1.0, 1.0 - 2**-20]]),
        np.array([1.0]),
        mu=1.0,
        rho=10000.0,
        max_iter=max_iter,
        tolerance=1e-16,
    )
    module = PyTorchDifferentiableSafetyModule(layer)
    x0 = torch.tensor([1.0 - 2**-36, 2**-36], dtype=torch.float64)
    utility = torch.zeros(2, dtype=torch.float64)

    module(x0, utility)
    assert layer._cached_lambda[0] > 1e-12
    assert layer._cached_active_a.shape == (0, 2)

    for h in [2**-38, 2**-39, 2**-40]:
        finite_difference = np.empty(2)
        for index in range(2):
            delta = torch.zeros(2, dtype=torch.float64)
            delta[index] = h
            plus = module(x0.detach(), utility.detach() + delta)[0].item()
            minus = module(x0.detach(), utility.detach() - delta)[0].item()
            finite_difference[index] = (plus - minus) / (2 * h)
        np.testing.assert_array_equal(finite_difference, [0.5, -0.5])
