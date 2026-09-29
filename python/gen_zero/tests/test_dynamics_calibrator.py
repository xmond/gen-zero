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


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_fixed_point_learning_roundtrip_and_frozen_diagonal(seed, tmp_path):
    x, y = numeric_problem(40, 256)
    eval_x, eval_y = numeric_problem(41, 256)
    c = CausalDynamicsCalibrator(8, 2, 12, seed=seed)
    before = separation(c.adapter.fixed_points(eval_x), eval_y)
    diagonal = c.adapter.lambda_.copy()
    original_x = x.copy()
    history = c.fit(x, y, sample_ids=np.arange(len(x)), source='mathematical-test-only',
                    split='train', encoder_id='numeric-identity', epochs=1000)
    after = separation(c.adapter.fixed_points(eval_x), eval_y)
    print(f'seed={seed} numeric separation before={before:.6f} after={after:.6f}')
    assert after > 3 * before
    assert history[-1] < history[0]
    np.testing.assert_array_equal(diagonal, c.adapter.lambda_)
    np.testing.assert_array_equal(x, original_x)
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
    x, y = numeric_problem(0, 16)
    with pytest.raises(ValueError, match='splits'):
        c.fit(x, y, sample_ids=np.arange(16), source='test', split=split, encoder_id='numeric')


def test_fail_closed_inputs_and_export(tmp_path):
    c = CausalDynamicsCalibrator(8, 2, 12)
    x, y = numeric_problem(0, 16)
    kw = dict(sample_ids=np.arange(16), source='math', split='train', encoder_id='numeric')
    with pytest.raises(ValueError, match='uncalibrated'):
        c.adapter.save(tmp_path / 'bad.npz')
    with pytest.raises(ValueError, match='unique'):
        c.fit(x, y, **dict(kw, sample_ids=np.zeros(16)))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        c.fit(x, y, **kw)


def test_cli_emits_loadable_weights(tmp_path):
    import json
    import os
    import subprocess
    import sys
    x, y = numeric_problem(50, 32)
    features = tmp_path / 'mathematical-features.npz'
    artifact = tmp_path / 'mathematical-weights.npz'
    np.savez(features, features=x, labels=y, sample_ids=np.arange(32),
             metadata=json.dumps(dict(source='mathematical-test-only', split='train',
                                      encoder_id='numeric', label_free_encoder_input=True)))
    from pathlib import Path
    env = os.environ.copy()
    python_dir = str(Path(__file__).resolve().parent.parent.parent)
    env["PYTHONPATH"] = f"{python_dir}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run([sys.executable, '-m', 'gen_zero.train.calibrate_causal_dynamics',
                             '--features', str(features), '--out', str(artifact),
                             '--dim', '8', '--epochs', '2'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['provenance']['split'] == 'train'
    assert CalibratedDynamics.load(artifact, encoder_id='numeric').certify()['rho'] <= .55


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
