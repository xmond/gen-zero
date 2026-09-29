"""Numerical contracts for regularized ZCA and special orthogonal alignment."""

import numpy as np
import pytest

from gen_zero.nanocore.zca_whitening import (
    ZCAWhitening,
    orthogonal_procrustes,
    zca_whitening_matrix,
)


def test_closed_form_and_regularized_covariance():
    rng = np.random.default_rng(71)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    eigenvalues = np.array([0.0, 1e-12, 0.1, 2.0, 8.0])
    covariance = (q * eigenvalues) @ q.T
    epsilon = 1e-4
    w = zca_whitening_matrix(covariance, epsilon)
    expected = (q * (eigenvalues + epsilon) ** -0.5) @ q.T
    np.testing.assert_allclose(w, expected, atol=1e-8)
    np.testing.assert_allclose(w, w.T, atol=1e-12)
    np.testing.assert_allclose(w @ covariance @ w,
                               (q * (eigenvalues / (eigenvalues + epsilon))) @ q.T,
                               atol=1e-10)
    assert np.linalg.norm(w, 2) <= (1 + 1e-12) / np.sqrt(epsilon)


def test_fit_centering_rotation_and_sphere():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 4)) @ rng.normal(size=(4, 4)) + 7
    model = ZCAWhitening(dim=4, epsilon=1e-10)
    whitened = model.fit_transform(x)
    np.testing.assert_allclose(whitened.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(4), atol=1e-7)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    q[:, -1] *= np.linalg.det(q)
    target = whitened @ q
    aligned = model.fit_transform(x, target)
    np.testing.assert_allclose(aligned, target, atol=1e-12)
    np.testing.assert_allclose(model.R_, q, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(model.transform(x, normalize=True), axis=1), 1)
    np.testing.assert_array_equal(model.transform(model.mean_[None, :], normalize=True), 0)


def test_reflection_is_corrected_optimally():
    source = np.diag([3.0, 2.0, 1.0])
    target = source @ np.diag([1.0, 1.0, -1.0])
    rotation = orthogonal_procrustes(source, target)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)
    # SO constraint necessarily costs 4 times the smallest cross singular value.
    assert np.linalg.norm(source @ rotation - target) ** 2 == pytest.approx(4)


@pytest.mark.parametrize('x', [np.ones((3, 5)), np.zeros((2, 5)),
                                  np.arange(15).reshape(3, 5)])
def test_rank_deficiency_is_finite(x):
    model = ZCAWhitening(dim=5)
    result = model.fit_transform(x, np.zeros_like(x))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(model.R_.T @ model.R_, np.eye(5), atol=1e-12)
    assert np.linalg.det(model.R_) == pytest.approx(1)


@pytest.mark.parametrize('epsilon', [0, -1, np.nan, np.inf])
def test_invalid_regularization(epsilon):
    with pytest.raises(ValueError):
        zca_whitening_matrix(np.eye(2), epsilon)
    with pytest.raises(ValueError):
        ZCAWhitening(epsilon=epsilon)


@pytest.mark.parametrize('covariance', [np.ones((2, 3)), [[1, 1], [0, 1]],
    np.diag([1, -0.1]), [[np.nan]], [[np.inf]], [[1j]], np.empty((0, 0))])
def test_invalid_covariance(covariance):
    with pytest.raises(ValueError):
        zca_whitening_matrix(covariance)


def test_validation_and_failed_refit():
    model = ZCAWhitening(dim=2)
    with pytest.raises(RuntimeError):
        model.transform([[0, 0]])
    for x in ([[1, 2]], [[1, 2, 3], [4, 5, 6]], [[0, np.inf], [0, 1]]):
        with pytest.raises(ValueError):
            model.fit(x)
    x = np.array([[1., 2.], [3., 4.], [5., 7.]])
    expected = model.fit_transform(x)
    with pytest.raises(ValueError):
        model.fit(x, np.zeros((2, 2)))
    np.testing.assert_array_equal(model.transform(x), expected)
    with pytest.raises(ValueError):
        orthogonal_procrustes([[1j]], [[0]])
    with pytest.raises((FloatingPointError, ValueError)):
        model.fit([[1e308, 0], [-1e308, 0]])


def test_full_4096_dimension():
    # Exercise actual dense decompositions at production dimension (no mocks).
    assert ZCAWhitening().dim == 4096
    w = zca_whitening_matrix(np.eye(4096), epsilon=0.25)
    np.testing.assert_allclose(np.diag(w), 1 / np.sqrt(1.25))
    w.flat[::4097] = 0
    assert np.count_nonzero(w) == 0
    del w
    rotation = orthogonal_procrustes(np.zeros((1, 4096)), np.zeros((1, 4096)))
    assert rotation.shape == (4096, 4096)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(4096), atol=1e-12)
    assert np.linalg.slogdet(rotation)[0] == 1
