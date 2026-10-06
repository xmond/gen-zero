"""Mathematical calibration tests, NOT natural-language semantic evidence.

Numeric latent-variable data have independent nuisance dimensions. They verify
optimization generalization on that stated model, never stand in for NLI data.
"""
import numpy as np
import pytest

from gen_zero.causal.dynamics_calibrator import CausalDynamicsCalibrator, CalibratedDynamics


def numeric_problem(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    return x, y


def separation(z, y):
    # Scale-invariant between/within scatter; cannot pass by enlarging B.
    means = np.array([z[y == k].mean(0) for k in (0, 1)])
    return np.square(means[0] - means[1]).sum() / np.square(z - means[y]).sum(1).mean()


def static_factors(seed):
    """Fixed numeric factors: input axis 0 drives state axis 0."""
    rng = np.random.default_rng(seed)
    u_b = np.zeros((8, 2), np.float32)
    v_b = np.zeros((12, 2), np.float32)
    u_b[0, 0] = 1.0
    v_b[0, 0] = 1.0
    u_a = (.05 * rng.normal(size=(8, 2))).astype(np.float32)
    v_a = (.05 * rng.normal(size=(8, 2))).astype(np.float32)
    return u_a, v_a, u_b, v_b


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_static_parameters_roundtrip_and_frozen_diagonal(seed, tmp_path):
    eval_x, eval_y = numeric_problem(41, 256)
    c = CausalDynamicsCalibrator(8, 2, 12, seed=seed)
    diagonal = c.adapter.lambda_.copy()
    c.set_static_parameters(*static_factors(seed), source='mathematical-test-only',
                            split='train', encoder_id='numeric-identity')
    after = separation(c.adapter.fixed_points(eval_x), eval_y)
    assert after > 1.0
    np.testing.assert_array_equal(diagonal, c.adapter.lambda_)
    path = tmp_path / 'numeric-test-only.npz'
    c.adapter.save(path)
    restored = CalibratedDynamics.load(path, encoder_id='numeric-identity')
    np.testing.assert_array_equal(restored.transition_matrix(), c.adapter.transition_matrix())
    h = np.zeros(8, np.float32)
    target = restored.fixed_points(eval_x[:1])[0]
    for _ in range(400):
        h = restored.step(eval_x[0], h)
    np.testing.assert_allclose(h, target, atol=2e-5)
    with pytest.raises(ValueError, match='encoder'):
        CalibratedDynamics.load(path, encoder_id='different')


def test_projection_nonnormal_and_lyapunov():
    a = CalibratedDynamics(8, 2, dtype=np.float32, seed=0)
    a.U_A *= 1000
    a._scale = a._compute_scale()
    certificate = a.certify()
    assert certificate['rho'] <= .55 and certificate['sigma'] <= .95
    x = np.arange(8, dtype=np.float32)
    fixed = a.fixed_points(x[None])[0]
    h = np.ones(8, np.float32)
    for _ in range(30):
        hn = a.step(x, h)
        assert np.linalg.norm(hn - fixed) <= .95 * np.linalg.norm(h - fixed) + 2e-5
        h = hn


def test_budget_includes_buffers_and_rejects_large_encoder():
    a = CausalDynamicsCalibrator(64, 1, 64).adapter
    assert a.working_set_bytes() <= 5632
    assert a.working_set_bytes() > sum(v.nbytes for v in vars(a).values() if isinstance(v, np.ndarray))
    with pytest.raises(ValueError, match='5.5 KiB'):
        CausalDynamicsCalibrator(64, 2, 64)
    print('numeric working-set bound:', a.working_set_bytes())


@pytest.mark.parametrize('split', ['test', 'validation', 'validation_matched', ''])
def test_reject_eval_splits(split):
    c = CausalDynamicsCalibrator(8, 2, 12)
    with pytest.raises(ValueError, match='splits'):
        c.set_static_parameters(*static_factors(0), source='test', split=split,
                                encoder_id='numeric')


def test_fail_closed_inputs_and_export(tmp_path):
    c = CausalDynamicsCalibrator(8, 2, 12)
    kw = dict(source='math', split='train', encoder_id='numeric')
    with pytest.raises(ValueError, match='uncalibrated'):
        c.adapter.save(tmp_path / 'bad.npz')
    u_a, v_a, u_b, v_b = static_factors(0)
    u_b[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        c.set_static_parameters(u_a, v_a, u_b, v_b, **kw)
    with pytest.raises(ValueError, match='shape'):
        c.set_static_parameters(u_a[:4], v_a, *static_factors(0)[2:], **kw)


def test_projection_handles_large_nonnormal_transient():
    a = CalibratedDynamics(4, 1, dtype=np.float32, seed=0)
    a.lambda_[:] = -20
    a.U_A[:] = 0
    a.V_A[:] = 0
    a.U_A[0, 0] = 100
    a.V_A[1, 0] = 100
    a._scale = a._compute_scale()
    cert = a.certify()
    assert cert['rho'] < .01
    assert .94 < cert['sigma'] <= .95
