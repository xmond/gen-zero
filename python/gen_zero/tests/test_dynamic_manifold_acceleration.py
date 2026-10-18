"""Regression checks for batched manifold readout and centered PRESS."""
import numpy as np
import pytest
from time import perf_counter

from gen_zero.causal.compiled_manifold_runtime import CompiledManifoldRuntime, TaskRuntimeSection
from gen_zero.causal.dynamics_calibrator import (CentredRidgeGram, centred_ridge_exact_loo,
                                                 select_regularization)


def test_centred_press_matches_literal_refits():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(13, 4)) + 2
    y = rng.normal(size=(13, 3)) + 1
    lam = 0.7
    residual = centred_ridge_exact_loo(x, y, lam)
    expected = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        xc = x[keep] - x[keep].mean(axis=0)
        coef = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]),
                               xc.T @ (y[keep] - y[keep].mean(axis=0)))
        expected.append(y[i] - (y[keep].mean(axis=0) +
                                (x[i] - x[keep].mean(axis=0)) @ coef))
    np.testing.assert_allclose(residual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('fit_intercept', [True, False])
def test_closed_form_loo_matches_brute_force_refits_in_both_intercept_modes(fit_intercept):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(17, 5)) + 1.5
    y = rng.normal(size=(17, 3)) + 2.0
    lam = 0.9
    residual = centred_ridge_exact_loo(x, y, lam, fit_intercept=fit_intercept)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        xm = x[keep].mean(axis=0) if fit_intercept else np.zeros(x.shape[1])
        ym = y[keep].mean(axis=0) if fit_intercept else np.zeros(y.shape[1])
        xc, yc = x[keep] - xm, y[keep] - ym
        coef = np.linalg.solve(xc.T @ xc + lam * np.eye(x.shape[1]), xc.T @ yc)
        expected = y[i] - (ym + (x[i] - xm) @ coef)
        assert np.max(np.abs(residual[i] - expected)) < 1e-10
    # The two modes score different models on uncentred data.
    other = centred_ridge_exact_loo(x, y, lam, fit_intercept=not fit_intercept)
    assert np.max(np.abs(residual - other)) > 1e-3


def test_drift_dynamics_deploys_the_loo_selected_model():
    from gen_zero.causal.counterfactual_drift_dynamics import (
        fit_counterfactual_drift_dynamics, simplex_codebook)
    from gen_zero.manifold import MasterClosedFormSolver
    rng = np.random.default_rng(5)
    n, k = 30, 3
    y = np.arange(n) % k
    x = rng.normal(size=(n, 12)) + rng.normal(size=(k, 12))[y]
    c = rng.normal(size=(n, 6))
    grid = (.01, .1, 1., 10.)
    dyn, info = fit_counterfactual_drift_dynamics(
        x, c, y, sample_ids=[str(i) for i in range(n)], source='unit', split='calibration',
        encoder_id='identity', n_classes=k, dim=4, n_components=6, cf_components=3,
        ridge_grid=grid, use_counterfactual=True)
    zx = (x - dyn.x_mean) @ dyn.x_basis
    zc = (c - dyn.c_mean) @ dyn.c_basis
    z = np.concatenate([zx, zc], axis=1)
    t = simplex_codebook(k, 4)[y]
    # Deployed map M (dim, p) = (I - A)^-1 [B | W_c]; compare with the no-intercept
    # ridge at the selected effective penalty, the model whose LOO was scored. The
    # artifact stores float32 arrays, hence the float32-level tolerance.
    deployed = np.linalg.solve(np.eye(4) - dyn.A, np.concatenate([dyn.B, dyn.W_c], axis=1))
    lam_eff = info['ridge_lambda_effective']
    expected = np.linalg.solve(z.T @ z + lam_eff * np.eye(z.shape[1]), z.T @ t).T
    np.testing.assert_allclose(deployed, expected, atol=1e-5, rtol=1e-4)
    assert lam_eff == pytest.approx(info['ridge_lambda'] * np.linalg.norm(z.T @ z) / np.sqrt(z.shape[1]))
    assert dyn.provenance['ridge_fit_intercept'] is False
    # The reported LOO accuracy is the literal leave-one-out refit of that deployed model.
    hits = 0
    for i in range(n):
        keep = np.arange(n) != i
        m = MasterClosedFormSolver().fit(z[keep], t[keep], lambda_reg=lam_eff, fit_intercept=False,
                                         adaptive_spectral_scaling=False).T
        pred = int(np.argmin(np.linalg.norm(simplex_codebook(k, 4) - m @ z[i], axis=1)))
        hits += pred == y[i]
    assert info['ridge_loo_accuracy'] == pytest.approx(hits / n)


def test_batch_matches_scalar_without_broadcast_tensor():
    rng = np.random.default_rng(8)
    d, p, k = 7, 5, 4
    sec = TaskRuntimeSection('t', k, False, d, p, 0, .2, (.2*np.eye(d)).astype('f'),
                             (np.eye(d)/.8).astype('f'), rng.normal(size=(d, p)).astype('f'), None,
                             rng.normal(size=(k, d)).astype('f'), np.zeros(p),
                             np.zeros((p, p), dtype='i1'), np.ones(p), None, None, None)
    rt = CompiledManifoldRuntime(None, {'t': sec}, '', '')
    x = rng.normal(size=(23, p)).astype('f')
    batch = rt.infer_batch('t', x)
    for i, item in enumerate(batch):
        scalar = rt.infer('t', x[i])
        np.testing.assert_allclose(item.scores, scalar.scores, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(item.fixed_point, scalar.fixed_point, rtol=2e-5, atol=2e-5)
        assert item.prediction == scalar.prediction
    with pytest.raises(ValueError):
        rt.infer_batch('t', np.full((2, p), np.nan))
    with pytest.raises(ValueError, match='infer_batch requires at least one sample row'):
        rt.infer_batch('t', np.empty((0, p), dtype='f'))
    with pytest.raises(ValueError, match='non-finite fixed_point or scores in infer_batch'):
        rt.infer_batch('t', np.full((2, p), 3e38, dtype='f'))


def test_calibrator_selects_exact_loo_minimum():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(25, 3))
    y = x @ rng.normal(size=(3, 2)) + rng.normal(scale=.2, size=(25, 2))
    grid = (.01, .1, 1., 10.)
    choice = select_regularization(x, y, grid)
    expected = min(grid, key=lambda lam: np.mean(centred_ridge_exact_loo(x, y, lam) ** 2))
    assert choice['ridge_lambda'] == expected
    assert choice['loo_mse'] == pytest.approx(np.mean(centred_ridge_exact_loo(x, y, expected) ** 2))


def test_batch_and_gram_benchmark(capsys):
    rng = np.random.default_rng(21)
    d, p, k, n = 16, 16, 8, 512
    sec = TaskRuntimeSection('bench', k, False, d, p, 0, .1,
                             (.1*np.eye(d)).astype('f'), (np.eye(d)/.9).astype('f'),
                             rng.normal(size=(d, p)).astype('f'), None,
                             rng.normal(size=(k, d)).astype('f'), np.zeros(p),
                             np.zeros((p, p), dtype='i1'), np.ones(p), None, None, None)
    runtime = CompiledManifoldRuntime(None, {'bench': sec}, '', '')
    x = rng.normal(size=(n, p)).astype('f')
    runtime.infer_batch('bench', x)
    t0 = perf_counter()
    scalar = [runtime.infer('bench', row).scores.copy() for row in x]
    scalar_time = perf_counter() - t0
    t0 = perf_counter()
    batch = runtime.infer_batch('bench', x)
    batch_time = perf_counter() - t0
    np.testing.assert_allclose(np.array([item.scores for item in batch]), scalar,
                               atol=5e-4, rtol=5e-5)
    z = rng.normal(size=(160, 64))
    targets = rng.normal(size=(160, 4))
    grid = (.001, .003, .01, .03, .1, .3, 1., 3., 10., 30., 100.)
    t0 = perf_counter()
    reused = select_regularization(z, targets, grid)
    gram_time = perf_counter() - t0
    t0 = perf_counter()
    losses = [np.mean(centred_ridge_exact_loo(z, targets, lam) ** 2) for lam in grid]
    refit_time = perf_counter() - t0
    assert reused['ridge_lambda'] == grid[int(np.argmin(losses))]
    print(f'batch={batch_time:.6f}s scalar={scalar_time:.6f}s ratio={scalar_time/batch_time:.2f}x '
          f'gram_reused={gram_time:.6f}s repeated={refit_time:.6f}s ratio={refit_time/gram_time:.2f}x')


def test_loo_selection_reduces_heldout_mse():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(30, 20))
    heldout = rng.normal(size=(300, 20))
    truth = np.zeros((20, 2))
    truth[:2] = rng.normal(size=(2, 2))
    y_train = train @ truth + rng.normal(scale=1.5, size=(30, 2))
    y_test = heldout @ truth + rng.normal(scale=1.5, size=(300, 2))
    gram = CentredRidgeGram(train)
    chosen = select_regularization(train, y_train, (.01, .1, 1., 10., 100.))
    baseline_coef, baseline_intercept, _ = gram.fit_and_loo(y_train, .01)
    baseline = np.mean((y_test - heldout @ baseline_coef - baseline_intercept) ** 2)
    selected = np.mean((y_test - heldout @ chosen['coefficients'] - chosen['intercept']) ** 2)
    print(f'heldout_n={len(heldout)} fixed_lambda_mse={baseline:.6f} '
          f'loo_lambda={chosen["ridge_lambda"]} loo_mse={selected:.6f}')
    assert selected < baseline


def test_cli_manifold_infer_matches_scalar_runtime(tmp_path, capsys):
    import json
    from gen_zero import cli
    from gen_zero.causal.knowledge_compiler import compile_manifold, synthetic_shape_golden_dynamics
    dyn, _ = synthetic_shape_golden_dynamics('t_cli', 3, paired=True, raw_dim=64, seed=4)
    path = tmp_path / 'codebook.bin'
    compile_manifold({'t_cli': (dyn, True)}, path, notes='cli test')
    rng = np.random.default_rng(6)
    x = rng.normal(size=(9, 64)).astype('f')
    c = rng.normal(size=(9, 64)).astype('f')
    y = rng.integers(0, 3, size=9)
    feats = tmp_path / 'feats.npz'
    np.savez(feats, x=x, c=c, y=y)
    argv = ['--no-color', 'manifold-infer', '--codebook', str(path), '--task', 't_cli',
            '--features', str(feats), '--block', 'x', '--cf-block', 'c', '--label-key', 'y', '--json']
    assert cli.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    rt = CompiledManifoldRuntime.load(path)
    sec = rt.section('t_cli')
    xp = (x - sec.x_mean) @ sec.x_basis()
    cp = (c - sec.c_mean) @ sec.c_basis()
    expected = [rt.infer('t_cli', xp[i], cp[i]).prediction for i in range(len(x))]
    assert report['predictions'] == expected
    assert report['accuracy'] == pytest.approx(np.mean(np.asarray(expected) == y))
    bad = tmp_path / 'bad.npz'
    np.savez(bad, x=np.full_like(x, np.nan))
    assert cli.main(['--no-color', 'manifold-infer', '--codebook', str(path), '--task', 't_cli',
                     '--features', str(bad), '--block', 'x']) == 1
    empty = tmp_path / 'empty.npz'
    np.savez(empty, x=np.empty((0, 64), dtype='f'))
    assert cli.main(['--no-color', 'manifold-infer', '--codebook', str(path), '--task', 't_cli',
                     '--features', str(empty), '--block', 'x']) == 1
    overflow = tmp_path / 'overflow.npz'
    np.savez(overflow, x=np.full((9, 64), 3e38, dtype='f'))
    assert cli.main(['--no-color', 'manifold-infer', '--codebook', str(path), '--task', 't_cli',
                     '--features', str(overflow), '--block', 'x']) == 1
