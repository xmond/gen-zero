"""Algebra/property tests only; synthetic vectors are NOT an agent benchmark."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'suites'))
from state_action_disentangled_head import CachedCPUHead, StateActionDisentangledHead


@pytest.mark.parametrize('ds,da,k', [(1, 1, 1), (3, 5, 7), (11, 4, 19)])
@pytest.mark.parametrize('normalize', [False, True])
def test_cached_matches_independent_scalar_bilinear(ds, da, k, normalize):
    rng = np.random.default_rng(ds + da + k)
    m, s, a = rng.normal(size=(ds, da)), rng.normal(size=(6, ds)), rng.normal(size=(k, da))
    h = StateActionDisentangledHead(m, -0.7, normalize=normalize).cache_actions(a)
    xs = s / np.linalg.norm(s, axis=1, keepdims=True) if normalize else s
    xa = a / np.linalg.norm(a, axis=1, keepdims=True) if normalize else a
    # Independent scalar summation, not just two wrappers of the same GEMV.
    oracle = np.array([[sum(xs[n,i]*m[i,j]*xa[t,j] for i in range(ds) for j in range(da)) - .7
                        for t in range(k)] for n in range(len(s))])
    np.testing.assert_allclose(h.score(s, a), oracle, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.array([h.score_cached(z) for z in s]), oracle, rtol=1e-13, atol=1e-13)


def test_exact_binary_representable_equivalence():
    # Bitwise equality is only promised for exactly representable arithmetic.
    h = StateActionDisentangledHead([[1, 2], [3, 4]], .5)
    a, s = np.array([[2, 1], [0, 4]]), np.array([2, 3])
    h.cache_actions(a)
    np.testing.assert_array_equal(h.score(s[None], a)[0], h.score_cached(s))


@pytest.mark.parametrize('normalize', [False, True])
def test_export_reload_absolute_error_and_unicode_ids(tmp_path, normalize):
    rng = np.random.default_rng(10)
    h = StateActionDisentangledHead(rng.normal(size=(16, 7)), .123, normalize=normalize)
    a, s = rng.normal(size=(4, 7)), rng.normal(size=(8, 16))
    ids = ['action-alpha', 'acción β', 'تشغيل', '실행']
    h.cache_actions(a, ids)
    path = h.export_cpu(tmp_path / 'head.npz')
    cpu = CachedCPUHead.load_cpu(path)
    with np.load(path, allow_pickle=False) as artifact:
        assert artifact['weights'].dtype == np.float64
        assert artifact['weights'].flags.c_contiguous
        assert artifact['weights'].shape == (4, 16)
    assert cpu.action_ids.tolist() == ids
    expected, actual = h.score(s, a), np.array([cpu.score_cached(z) for z in s])
    assert np.max(np.abs(expected - actual)) < 1e-5
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_cosine_special_case():
    rng = np.random.default_rng(14)
    s, a = rng.normal(size=(3, 6)), rng.normal(size=(5, 6))
    h = StateActionDisentangledHead(np.eye(6), normalize=True).cache_actions(a)
    expected = s @ a.T / np.outer(np.linalg.norm(s, axis=1), np.linalg.norm(a, axis=1))
    np.testing.assert_allclose(h.score(s, a), expected, atol=1e-14)
    np.testing.assert_allclose(h.score_cached(s[0]), expected[0], atol=1e-14)
    np.testing.assert_allclose(h.score(s * 3, a * 9), expected, atol=1e-14)


def test_fit_recovers_metric_on_unseen_pairs_and_invalidates_cache():
    rng = np.random.default_rng(31)
    m, bias = rng.normal(size=(3, 2)), -1.7
    s, a = rng.normal(size=(100, 3)), rng.normal(size=(100, 2))
    y = np.array([s[i] @ m @ a[i] + bias for i in range(len(s))])
    h = StateActionDisentangledHead(np.zeros((3, 2))).cache_actions(a[:4])
    h.fit(s, a, y, ridge=0)
    np.testing.assert_allclose(h.metric, m, atol=1e-12)
    assert abs(h.bias - bias) < 1e-12
    with pytest.raises(RuntimeError):
        h.score_cached(s[0])
    unseen_s, unseen_a = rng.normal(size=(8, 3)), rng.normal(size=(5, 2))
    h.cache_actions(unseen_a)
    np.testing.assert_allclose(h.score(unseen_s, unseen_a), unseen_s @ m @ unseen_a.T + bias, atol=1e-12)


def test_ridge_matches_independent_normal_equations_and_permutation():
    rng = np.random.default_rng(45)
    s, a, y = rng.normal(size=(40, 3)), rng.normal(size=(40, 2)), rng.normal(size=40)
    design = np.array([np.outer(si, ai).ravel() for si, ai in zip(s, a)])
    design = np.column_stack((design, np.ones(40)))
    penalty = np.diag([.8] * 6 + [0])
    oracle = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    h = StateActionDisentangledHead(np.zeros((3, 2))).fit(s, a, y, ridge=.8)
    np.testing.assert_allclose(np.r_[h.metric.ravel(), h.bias], oracle, atol=1e-12)
    p = rng.permutation(40)
    other = StateActionDisentangledHead(np.zeros((3, 2))).fit(s[p], a[p], y[p], ridge=.8)
    np.testing.assert_allclose(other.metric, h.metric, atol=1e-12)


def test_normalized_fit_uses_same_features_as_scoring():
    rng = np.random.default_rng(23)
    s, a = rng.normal(size=(60, 2)), rng.normal(size=(60, 3))
    teacher = StateActionDisentangledHead(rng.normal(size=(2,3)), .3, normalize=True)
    y = teacher.score(s, a).diagonal()
    h = StateActionDisentangledHead(np.zeros((2,3)), normalize=True).fit(s,a,y,ridge=0)
    np.testing.assert_allclose(h.metric, teacher.metric, atol=1e-12)


def test_ownership_and_action_permutation():
    m, a, s = np.eye(3), np.eye(3), np.arange(1., 4.)
    h = StateActionDisentangledHead(m).cache_actions(a)
    before = h.score_cached(s)
    m[:] = 99; a[:] = 42
    np.testing.assert_array_equal(before, h.score_cached(s))
    with pytest.raises(ValueError):
        h.metric.setflags(write=True)
    h.cache_actions(np.eye(3)[[2, 0, 1]])
    np.testing.assert_array_equal(h.score_cached(s), before[[2, 0, 1]])


@pytest.mark.parametrize('metric', [[], [[np.nan]], [[np.inf]], [['1']], [[1j]]])
def test_reject_invalid_metric(metric):
    with pytest.raises(ValueError):
        StateActionDisentangledHead(metric)


@pytest.mark.parametrize('state', [[0., 0.], [np.nan, 1.], [1.], [[1., 2.]]])
def test_reject_invalid_normalized_state(state):
    h = StateActionDisentangledHead(np.eye(2), normalize=True).cache_actions(np.eye(2))
    with pytest.raises(ValueError):
        h.score_cached(state)


def test_failures_are_explicit_and_preserve_valid_model(tmp_path):
    h = StateActionDisentangledHead(np.eye(2))
    with pytest.raises(RuntimeError): h.score_cached([1, 2])
    with pytest.raises(RuntimeError): h.export_cpu(tmp_path/'missing.npz')
    h.cache_actions(np.eye(2))
    with pytest.raises(ValueError): h.cache_actions([[1, 2]], ['duplicate', 'duplicate'])
    with pytest.raises(ValueError): h.cache_actions(np.eye(2), ['same', 'same'])
    with pytest.raises(ValueError): h.cache_actions(np.zeros((0, 2)))
    with pytest.raises(ValueError): h.fit(np.eye(2), np.eye(2), [1, 2], ridge=-1)
    with pytest.raises(ValueError): h.fit(np.eye(2), np.eye(2), [1, 2], max_design_elements=1)
    with pytest.raises(ValueError): h.fit(np.eye(2), np.eye(2), [1])
    np.testing.assert_array_equal(h.score_cached([1, 2]), [1, 2])


def test_corrupt_export_is_rejected(tmp_path):
    path = tmp_path/'bad.npz'
    np.savez(path, version=2, weights=np.eye(2), bias=np.zeros(2), action_ids=['a', 'b'], normalize=False)
    with pytest.raises(ValueError): CachedCPUHead.load_cpu(path)
    np.savez(path, version=1, weights=np.eye(2), bias=np.zeros(2), action_ids=['a', 'b'], normalize='False')
    with pytest.raises(ValueError): CachedCPUHead.load_cpu(path)


def test_rank_deficiency_and_overflow_are_not_hidden():
    h = StateActionDisentangledHead(np.eye(2)).fit(np.ones((4,2)), np.ones((4,2)), np.full(4,3.), ridge=0)
    np.testing.assert_allclose(h.score(np.ones((1,2)), np.ones((1,2))), [[3.]])
    with pytest.raises(FloatingPointError):
        StateActionDisentangledHead([[1e308]]).cache_actions([[1e308]])
