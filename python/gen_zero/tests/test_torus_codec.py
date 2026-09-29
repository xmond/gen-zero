"""torus_codec lives in the package so wheel installs never need benchmarks/."""
import inspect

import numpy as np
import pytest

from gen_zero.world_model import neural_dynamics
from gen_zero.world_model.neural_dynamics import NeuralDynamicsWorldModel
from gen_zero.world_model.torus_codec import (
    TORUS_ACTION_DIM,
    TORUS_ACTION_NAMES,
    encode_torus_action,
    latent_codes,
)


def test_codes_are_orthonormal_and_deterministic():
    codes = latent_codes(3, TORUS_ACTION_DIM, 20260926)
    assert codes.shape == (3, TORUS_ACTION_DIM)
    np.testing.assert_allclose(codes @ codes.T, np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(codes, latent_codes(3, TORUS_ACTION_DIM, 20260926))


def test_latent_codes_rejects_too_small_dim():
    with pytest.raises(ValueError):
        latent_codes(4, 3, 0)


def test_encode_torus_action_shape_dtype_and_unknown_name():
    for name in TORUS_ACTION_NAMES:
        vec = encode_torus_action(name)
        assert vec.shape == (TORUS_ACTION_DIM,) and vec.dtype == np.float32
    with pytest.raises(KeyError):
        encode_torus_action("north")


def test_model_encode_action_uses_codec():
    model = NeuralDynamicsWorldModel.__new__(NeuralDynamicsWorldModel)
    model.action_dim, model.action_vocab = TORUS_ACTION_DIM, None
    np.testing.assert_array_equal(model.encode_action("straight"), encode_torus_action("straight"))


def test_neural_dynamics_has_no_benchmarks_import():
    assert "deadlock_torus_env" not in inspect.getsource(neural_dynamics)
