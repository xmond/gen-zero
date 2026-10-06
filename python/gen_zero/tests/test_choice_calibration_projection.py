"""Numerical contracts, independent of language, labels, or external models."""
# anti-leakage: allow-mock-tensor
import math
import numpy as np
import pytest
from gen_zero.nanocore.choice_head import (
    FastSimplexETFProjection, calibrate_temperature_and_entropy as calibrate,
)


def entropy(p):
    positive = p[p > 0]
    return -np.sum(positive * np.log(positive)) / np.log(p.size) if p.size > 1 else 0.0


def test_temperature_matches_direct_softmax_and_offsets():
    x = np.array([0., 2., 4.])
    expected = np.exp(x / 2) / np.exp(x / 2).sum()
    for scores in (x, x + 1e16):
        p, confidence, h = calibrate(scores, temperature=2)
        np.testing.assert_allclose(p, expected, atol=1e-14, rtol=0)
        assert confidence == max(p)
        assert h == pytest.approx(entropy(p))


@pytest.mark.parametrize('scale', [1e-300, 1., 1e300])
@pytest.mark.parametrize('target', [0., .1, .5, .95, 1.])
def test_entropy_target_across_scales(scale, target):
    p, _, h = calibrate(np.array([-1., 0., 1.]) * scale, target)
    assert h == pytest.approx(target, abs=1e-9)
    assert entropy(p) == pytest.approx(target, abs=1e-9)
    assert np.all(np.diff(p) >= 0)
    assert p.sum() == pytest.approx(1)


def test_ties_unreachable_and_nats():
    lower = math.log(2) / math.log(3)
    p, _, h = calibrate([3., 3., -1.], lower)
    np.testing.assert_array_equal(p, [.5, .5, 0])
    with pytest.raises(ValueError):
        calibrate([3., 3., -1.], lower - .01)
    with pytest.raises(ValueError):
        calibrate([1., 1.], .5)
    np.testing.assert_allclose(calibrate([1., 2., 3.], .5)[0],
                               calibrate([1., 2., 3.], .5 * math.log(3), normalized=False)[0])


def test_extreme_finite_logits_and_temperature():
    with np.errstate(all='raise'):
        p, _, _ = calibrate([-1e308, 1e308], temperature=1e308)
    np.testing.assert_allclose(p, [1 / (1 + math.exp(2)), 1 / (1 + math.exp(-2))])
    p, _, _ = calibrate([-1e308, 1e308], temperature=1e-300)
    np.testing.assert_array_equal(p, [0., 1.])


def test_batch_and_permutation_equivariance():
    x = np.random.default_rng(41).normal(size=(4, 5))
    perm = np.array([3, 1, 4, 0, 2])
    p, conf, h = calibrate(x, .73)
    assert conf.shape == h.shape == (4,)
    np.testing.assert_allclose(h, .73, atol=1e-9)
    np.testing.assert_allclose(calibrate(x[:, perm], .73)[0], p[:, perm], atol=1e-12)


def test_probability_cap_and_uniform_mixing():
    p, c, h = calibrate([100., 0., 0., 0.], max_confidence_cap=.4)
    np.testing.assert_allclose(p, [.4, .2, .2, .2], atol=1e-14)
    assert c == pytest.approx(.4)
    assert h == pytest.approx(entropy(p))
    np.testing.assert_allclose(calibrate([3., 1., 0.], entropy_penalty=1)[0], np.ones(3) / 3)
    for cap in (.2, 0, float('nan')):
        with pytest.raises(ValueError):
            calibrate([1., 2.], max_confidence_cap=cap)


def test_singleton_and_conflicting_controls():
    assert calibrate([5.]) == (np.array([1.]), 1., 0.)
    with pytest.raises(ValueError):
        calibrate([5.], max_confidence_cap=.88)
    with pytest.raises(ValueError):
        calibrate([5.], .5)
    for options in ({'entropy_penalty': .2}, {'max_confidence_cap': .8}):
        with pytest.raises(ValueError):
            calibrate([0., 1.], .5, **options)
    with pytest.raises(FloatingPointError):
        calibrate([0., 1., 2.], .654321, max_iterations=1, tolerance=1e-14)


@pytest.mark.parametrize('scores', [[], 2., [float('nan'), 1.], [1., float('inf')]])
def test_invalid_logits(scores):
    with pytest.raises(ValueError):
        calibrate(scores)


@pytest.mark.parametrize('k,d', [(2, 1), (3, 2), (5, 4), (5, 128), (32, 64)])
def test_etf_geometry_and_tight_frame(k, d):
    projection = FastSimplexETFProjection(k, d)
    v = projection.matrix
    expected = (k / (k - 1)) * (np.eye(k) - np.ones((k, k)) / k)
    np.testing.assert_allclose(v @ v.T, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(v.sum(axis=0), 0, atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(v.T @ v)
    np.testing.assert_allclose(eigenvalues[-(k-1):], k / (k-1), atol=1e-14)
    np.testing.assert_allclose(eigenvalues[:d-k+1], 0, atol=1e-14)
    assert np.linalg.matrix_rank(v) == k - 1
    np.testing.assert_array_equal(v, FastSimplexETFProjection(k, d).matrix)
    x = np.random.default_rng(8).normal(size=(4, d))
    np.testing.assert_allclose(projection(x), x @ v.T)
    with pytest.raises(ValueError):
        projection.matrix[0, 0] = 42


@pytest.mark.parametrize('k,d', [(3, 1), (0, 3), (2, 0), (True, 3), (2.5, 3)])
def test_etf_invalid_dimensions(k, d):
    with pytest.raises((ValueError, TypeError)):
        FastSimplexETFProjection(k, d)


def test_cosine_projection_scaling_and_zero_rejection():
    projection = FastSimplexETFProjection(3, 4, normalize=True)
    x = np.array([1., 2., -1., 0.])
    np.testing.assert_allclose(projection(x * 1e300), projection(x), atol=1e-14)
    np.testing.assert_allclose(projection(x * 1e-300), projection(x), atol=1e-14)
    with pytest.raises(ValueError):
        projection(np.zeros(4))


def test_torch_projection_and_gradient():
    torch = pytest.importorskip('torch')
    projection = FastSimplexETFProjection(4, 6)
    x = torch.arange(12, dtype=torch.float64).reshape(2, 6).requires_grad_()
    result = projection(x)
    np.testing.assert_allclose(result.detach().numpy(), projection(x.detach().numpy()))
    result.square().sum().backward()
    expected_grad = 2 * x.detach().numpy() @ projection.matrix.T @ projection.matrix
    np.testing.assert_allclose(x.grad.numpy(), expected_grad, atol=1e-13)
    with pytest.raises(TypeError):
        projection(torch.ones(6, dtype=torch.int64))
    tensor = projection.torch_matrix()
    tensor.zero_()
    assert np.any(projection.matrix != 0)


def test_facade_exports_exist():
    from gen_zero.model import choice_head
    assert len(choice_head.__all__) == len(set(choice_head.__all__))
    for name in choice_head.__all__:
        assert hasattr(choice_head, name), name
