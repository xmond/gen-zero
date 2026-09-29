"""Tests for ``DeepProjectionAdapter`` (docs/zero/10-...-spec.md §3.3).

Loaded against the real ``zero_manifold_open_v1.npz`` artifact -- not a synthetic
manifold -- so the zero-init equivalence check is against the actual linear skip
the deep adapter is meant to extend, not a stand-in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.deep_projection_adapter import DeepProjectionAdapter  # noqa: E402
from gen_zero.causal.zero_runtime import ZeroManifold  # noqa: E402

MANIFOLD_PATH = REPO / "benchmarks" / "artifacts" / "zero" / "zero_manifold_open_v1.npz"
ENCODER_ID = "zero-qwen2.5-0.5b-trunk:int8:last-token"

EXPECTED_PARAM_COUNT = 303_424
EXPECTED_BYTES = EXPECTED_PARAM_COUNT * 4  # float32


@pytest.fixture(scope="module")
def manifold() -> ZeroManifold:
    return ZeroManifold.load(MANIFOLD_PATH, encoder_id=ENCODER_ID)


def test_parameter_count_and_byte_size():
    model = DeepProjectionAdapter(MANIFOLD_PATH, encoder_id=ENCODER_ID)
    total = sum(p.numel() for p in model.parameters())
    assert total == EXPECTED_PARAM_COUNT
    assert total * 4 == EXPECTED_BYTES == 1_213_696


def test_zero_init_equals_zero_manifold_project(manifold):
    """At construction, delta_z == 0 exactly, so the adapter must reproduce
    ``ZeroManifold.project`` bit-for-bit up to floating-point rounding.

    The task's numeric contract is "< 1e-7 logically, < 1e-6 enforced". Doing the
    forward pass in float64 (matching ``ZeroManifold.project``'s own float64
    computation, zero_runtime.py:185) measured max abs error ~1e-8, comfortably
    inside both bounds -- so no loosening of the enforced tolerance was needed.
    A float32 forward was also measured (~2.2e-7): still under the 1e-6 bound
    enforced here, but outside the tighter 1e-7 the doc states informally; this
    is a real float32-accumulation finding, not something to paper over, which
    is why the test pins the adapter to float64 rather than silently loosening
    the assertion.
    """
    model = DeepProjectionAdapter(MANIFOLD_PATH, encoder_id=ENCODER_ID).double()

    rng = np.random.default_rng(0)
    h = rng.normal(size=(32, 896)).astype(np.float64)

    expected = manifold.project(h)
    with torch.no_grad():
        actual = model(torch.from_numpy(h)).numpy()

    max_abs_err = np.abs(actual - expected).max()
    assert max_abs_err < 1e-6, f"max abs err {max_abs_err} exceeds 1e-6"


def test_output_is_unit_norm():
    model = DeepProjectionAdapter(MANIFOLD_PATH, encoder_id=ENCODER_ID)
    rng = np.random.default_rng(1)
    h = torch.from_numpy(rng.normal(size=(16, 896)).astype(np.float32))
    with torch.no_grad():
        z = model(h)
    norms = z.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)


def test_single_vector_input_is_supported():
    model = DeepProjectionAdapter(MANIFOLD_PATH, encoder_id=ENCODER_ID)
    h = torch.randn(896)
    with torch.no_grad():
        z = model(h)
    assert z.shape == (64,)
    assert torch.isfinite(z).all()


def test_wrong_last_dim_raises_value_error():
    model = DeepProjectionAdapter(MANIFOLD_PATH, encoder_id=ENCODER_ID)
    with pytest.raises(ValueError, match="896"):
        model(torch.randn(4, 100))


def test_nonfinite_input_raises_value_error():
    model = DeepProjectionAdapter(MANIFOLD_PATH, encoder_id=ENCODER_ID)
    bad = torch.randn(2, 896)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(bad)


def test_non_tensor_input_raises_type_error():
    model = DeepProjectionAdapter(MANIFOLD_PATH, encoder_id=ENCODER_ID)
    with pytest.raises(TypeError):
        model(np.random.randn(896).astype(np.float32))
