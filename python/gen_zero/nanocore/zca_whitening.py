"""Closed-form ZCA whitening and proper orthogonal alignment of row vectors.

All calculations use float64. For covariance C = U diag(lambda) U.T,
W_ZCA = U diag((max(lambda, 0) + epsilon)**-1/2) U.T. Thus the
spectral gain is at most 1/sqrt(epsilon), including singular covariances.
Regularized output covariance has eigenvalues lambda/(lambda + epsilon),
not exactly one. Unit-sphere normalization is a separate, optional step.

Dense fitting costs O(d**3) time and O(d**2) memory; d defaults to 4096.
Guarantees hold up to floating-point error. Unrepresentable intermediate
values raise an exception rather than returning NaN or infinity.
"""

import numpy as np


__all__ = ["zca_whitening_matrix", "orthogonal_procrustes", "ZCAWhitening"]


def _matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty 2-D matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _epsilon(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    return value


def zca_whitening_matrix(covariance, epsilon=1e-6):
    """Return the symmetric regularized inverse square root of a PSD matrix.

    Reject asymmetric or materially indefinite inputs. Negative eigenvalues
    within a dimension-scaled floating-point roundoff tolerance are clipped.
    Epsilon is an additive ridge in covariance units, not a variance floor.
    """
    epsilon = _epsilon(epsilon)
    covariance = _matrix(covariance, "covariance")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    tolerance = 100 * np.finfo(float).eps * covariance.shape[0] * scale
    if np.max(np.abs(covariance / scale - covariance.T / scale)) > tolerance / scale:
        raise ValueError("covariance must be symmetric")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance * 0.5 + covariance.T * 0.5)
        if eigenvalues[0] < -tolerance:
            raise ValueError("covariance must be positive semidefinite")
        gains = 1.0 / np.sqrt(np.maximum(eigenvalues, 0.0) + epsilon)
        whitening = (eigenvectors * gains) @ eigenvectors.T
    return whitening


def orthogonal_procrustes(source, target):
    """Minimize ||source @ R - target||_F over R in SO(d), including SO(4096).

    Rows are paired observations; no implicit centering or scaling is applied.
    If source.T @ target = U S V.T, return U D V.T, with D's final
    entry det(U V.T). This corrects reflections along the least singular
    direction. Rank-deficient problems admit multiple optimal rotations.
    """
    source = _matrix(source, "source")
    target = _matrix(target, "target")
    if source.shape != target.shape:
        raise ValueError("source and target must have matching shapes")
    with np.errstate(over="raise", invalid="raise"):
        cross_covariance = source.T @ target
        u, _, vt = np.linalg.svd(cross_covariance, full_matrices=False)
        # slogdet avoids determinant underflow; orthogonal factors have sign +/-1.
        sign = np.linalg.slogdet(u)[0] * np.linalg.slogdet(vt)[0]
        u[:, -1] *= sign
        return u @ vt


class ZCAWhitening:
    """Fit a training mean, ZCA matrix, and optional SO(d) target alignment.

    ``fit(X, target)`` aligns whitened training rows to target rows.
    ``transform(X)`` computes (X - mean_) @ W_ZCA_ @ R_. Targets are
    already in the desired coordinate system and are not centered implicitly.
    With no target, R_ is identity. Fit uses sample covariance (n - 1).
    """

    def __init__(self, dim=4096, epsilon=1e-6):
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 1:
            raise ValueError("dim must be a positive integer")
        self.dim = int(dim)
        self.epsilon = _epsilon(epsilon)
        self.mean_ = None
        self.W_ZCA_ = None
        self.R_ = None

    def _samples(self, samples):
        samples = _matrix(samples, "samples")
        if samples.shape[1] != self.dim:
            raise ValueError(f"samples must have {self.dim} features")
        return samples

    def fit(self, samples, target=None):
        """Fit using at least two rows; failed fits preserve previous state."""
        samples = self._samples(samples)
        if len(samples) < 2:
            raise ValueError("fit requires at least two samples")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = samples.mean(axis=0)
            centered = samples - mean
            covariance = (centered.T @ centered) / (len(samples) - 1)
            whitening = zca_whitening_matrix(covariance, self.epsilon)
            rotation = (np.eye(self.dim) if target is None else
                        orthogonal_procrustes(centered @ whitening, target))
        self.mean_, self.W_ZCA_, self.R_ = mean, whitening, rotation
        return self

    def transform(self, samples, *, normalize=False):
        """Transform rows; optional unit normalization leaves exact zeros zero."""
        if self.mean_ is None:
            raise RuntimeError("fit must be called before transform")
        samples = self._samples(samples)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = ((samples - self.mean_) @ self.W_ZCA_) @ self.R_
            if normalize:
                # Scaling first avoids overflow/underflow in the row norm.
                scale = np.max(np.abs(result), axis=1, keepdims=True)
                result = result / np.where(scale == 0, 1.0, scale)
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                result = result / np.where(norms == 0, 1.0, norms)
        return result

    def fit_transform(self, samples, target=None, *, normalize=False):
        """Fit and transform the same training rows."""
        return self.fit(samples, target).transform(samples, normalize=normalize)
