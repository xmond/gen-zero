"""Geometric latent fusion of two frozen-LLM feature views (thin SVD + orthogonal Procrustes).

Input: two row-aligned views X_q (n x d_q), X_l (n x d_l), e.g. Qwen-72B and Llama-70B last-token
states (8192-d each). Everything is fitted on TRAINING rows only; `transform` is a pure function
of the fitted attributes, so held-out and test rows never influence the geometry.

  1. Standardize each view with train mean / std, thin-SVD it, keep the Gavish-Donoho signal rank
     r_m (spec21_advanced_heads): coordinates C_m = Z_m V_m^T  (n x r_m).
  2. Cross-covariance of the two coordinate systems M = C_q^T C_l / n = A Sigma B^T (SVD). The
     orthogonal Procrustes map from Qwen coordinates to Llama coordinates is R = A B^T; the paired
     singular vectors (a_i, b_i) are the axes on which the two views agree best.
  3. Core = the first k axis pairs holding `core_energy` of ||M||_F^2. Per axis the two views are
     put on a common scale and averaged (the average denoises what both views share).
     Residual = the remaining orthogonal directions of EACH view (A[:, k:], B[:, k:]): what one
     model sees and the aligned shared subspace does not.
  4. Fused representation = [core | w * resid_q | w * resid_l].

Scope limits, stated so nobody reads more into this than it does:
  * With every direction kept (k small, w = 1, full rank) the fused matrix is an invertible linear
    re-parametrisation of [C_q | C_l]. A plain linear head fitted without regularisation would
    therefore give the same predictions as on the concatenation. The gain, if any, comes from
    (a) the Gavish-Donoho truncation, (b) the averaged, denoised core, and (c) the residual weight
    w acting through the head's regulariser. The evaluation suite therefore always reports a plain
    concatenation control next to it.
  * The rotation is estimated from second moments of two views of the SAME rows; it says nothing
    about rows outside the training distribution.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import svd as _svd

from spec21_advanced_heads import _gd_from_singular_values

_EPS = 1e-12


def orthogonal_procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R minimising ||a R - b||_F over R with orthonormal columns/rows: R = U V^T, U S V^T = a^T b.
    Shape (a.shape[1], b.shape[1]); a partial isometry when the two widths differ."""
    a, b = _finite_matrix(a, "a"), _finite_matrix(b, "b")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"a has {a.shape[0]} rows but b has {b.shape[0]}")
    u, _, vt = _svd(a.T @ b, full_matrices=False)
    return u @ vt


def _finite_matrix(x, name: str) -> np.ndarray:
    m = np.asarray(x, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError(f"{name} contains non-finite values")
    return m


class _View:
    """Train-fitted standardisation + Gavish-Donoho-truncated PCA of one feature view."""

    def __init__(self, x: np.ndarray, name: str, max_rank: Optional[int]) -> None:
        n, d = x.shape
        self.mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self.scale = np.where(sd > _EPS, sd, 1.0)
        z = (x - self.mean) / self.scale
        _, s, vt = _svd(z, full_matrices=False, overwrite_a=True)
        info = _gd_from_singular_values(s, n, d, center=True)
        r = info["rank"] if max_rank is None else min(info["rank"], max_rank)
        if r == 0:
            raise ValueError(f"view {name}: no singular value exceeds the Gavish-Donoho threshold")
        self.rank, self.gd_info = int(r), info
        self.components = np.ascontiguousarray(vt[:r].T)               # d x r

    def coords(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.components


class GeometricLatentFusion:
    """See module docstring. Build with `GeometricLatentFusion.fit(x_q, x_l)`."""

    def __init__(self) -> None:
        raise TypeError("use GeometricLatentFusion.fit(...)")

    @classmethod
    def fit(cls, x_q: np.ndarray, x_l: np.ndarray, core_energy: float = 0.9,
            max_rank: Optional[int] = None) -> "GeometricLatentFusion":
        x_q, x_l = _finite_matrix(x_q, "x_q"), _finite_matrix(x_l, "x_l")
        if x_q.shape[0] != x_l.shape[0]:
            raise ValueError(f"views have {x_q.shape[0]} vs {x_l.shape[0]} rows")
        if x_q.shape[0] < 3:
            raise ValueError("need at least 3 rows")
        if not 0.0 < core_energy <= 1.0:
            raise ValueError(f"core_energy must be in (0, 1], got {core_energy!r}")
        n = x_q.shape[0]
        vq, vl = _View(x_q, "q", max_rank), _View(x_l, "l", max_rank)
        cq, cl = vq.coords(x_q), vl.coords(x_l)
        cross = cq.T @ cl / n                                          # r_q x r_l
        a, sig, bt = _svd(cross, full_matrices=True)
        b = bt.T
        p = sig.size
        energy = np.cumsum(sig ** 2) / max(float((sig ** 2).sum()), _EPS)
        k = int(min(np.searchsorted(energy, core_energy - 1e-12) + 1, p))
        self = object.__new__(cls)
        self.core_energy, self.core_dim = float(core_energy), k
        self.view_q, self.view_l = vq, vl
        self.cross_singular_values_ = sig
        self.core_dirs_q_, self.resid_dirs_q_ = a[:, :k].copy(), a[:, k:].copy()
        self.core_dirs_l_, self.resid_dirs_l_ = b[:, :k].copy(), b[:, k:].copy()
        # Common per-axis scale so averaging two views does not let the louder one dominate.
        self.core_rms_q_ = np.maximum((cq @ self.core_dirs_q_).std(axis=0), _EPS)
        self.core_rms_l_ = np.maximum((cl @ self.core_dirs_l_).std(axis=0), _EPS)
        self.procrustes_ = a[:, :p] @ bt[:p]                           # R = A B^T, Qwen coords -> Llama coords
        return self

    @property
    def output_dim(self) -> int:
        return self.core_dim + self.resid_dirs_q_.shape[1] + self.resid_dirs_l_.shape[1]

    def core_views(self, x_q: np.ndarray, x_l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The two views' coordinates on the k shared axes (aligned pairs)."""
        cq = self.view_q.coords(_finite_matrix(x_q, "x_q")) @ self.core_dirs_q_
        cl = self.view_l.coords(_finite_matrix(x_l, "x_l")) @ self.core_dirs_l_
        return cq, cl

    def transform(self, x_q: np.ndarray, x_l: np.ndarray, residual_weight: float = 1.0) -> np.ndarray:
        if not residual_weight >= 0.0:
            raise ValueError(f"residual_weight must be >= 0, got {residual_weight!r}")
        x_q, x_l = _finite_matrix(x_q, "x_q"), _finite_matrix(x_l, "x_l")
        if x_q.shape[0] != x_l.shape[0]:
            raise ValueError(f"views have {x_q.shape[0]} vs {x_l.shape[0]} rows")
        coords_q, coords_l = self.view_q.coords(x_q), self.view_l.coords(x_l)
        cq, cl = coords_q @ self.core_dirs_q_, coords_l @ self.core_dirs_l_
        core = 0.5 * (cq / self.core_rms_q_ + cl / self.core_rms_l_) * np.sqrt(self.core_rms_q_ * self.core_rms_l_)
        res_q = coords_q @ self.resid_dirs_q_
        res_l = coords_l @ self.resid_dirs_l_
        return np.hstack([core, residual_weight * res_q, residual_weight * res_l])
