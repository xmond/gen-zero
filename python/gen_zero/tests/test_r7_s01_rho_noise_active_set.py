import numpy as np
import torch

from gen_zero.gate.differentiable_safety_layer import (
    DifferentiableSafetyLayer,
    PyTorchDifferentiableSafetyModule,
)


def test_r7_s01_roundoff_dual_does_not_activate_interior_constraint():
    layer = DifferentiableSafetyLayer(
        2,
        np.array([[1.0, 1.0 - 2**-20]]),
        np.array([1.0]),
        mu=1.0,
        rho=10000.0,
    )
    module = PyTorchDifferentiableSafetyModule(layer)
    x0 = torch.tensor([1.0 - 2**-36, 2**-36], dtype=torch.float64)
    utility = torch.zeros(2, dtype=torch.float64)

    module(x0, utility)
    assert layer._cached_lambda[0] > 1e-12  # Reproduces the old false-positive dual.
    assert layer._cached_active_a.shape == (0, 2)

    # Stay inside the simplex face: x0[1] is only 2**-36 above zero.
    epsilon = 1e-11
    finite_difference = np.empty(2)
    for index in range(2):
        delta = torch.zeros(2, dtype=torch.float64)
        delta[index] = epsilon
        plus = module(x0.detach(), utility.detach() + delta)[0].item()
        minus = module(x0.detach(), utility.detach() - delta)[0].item()
        finite_difference[index] = (plus - minus) / (2 * epsilon)

    np.testing.assert_allclose(finite_difference, [0.5, -0.5], rtol=0, atol=1e-6)
