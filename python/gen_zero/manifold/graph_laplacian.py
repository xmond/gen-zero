"""Sparse multi-model kNN graph Laplacian and the compact Z^T L Z operator.

Each feature matrix in ``feature_list`` describes the same N samples as seen by
one model (for example two teacher encoders). Every model contributes a
symmetric kNN affinity graph with self-tuning heat-kernel weights
(Zelnik-Manor & Perona 2004):

    w_ij = exp(-d_ij^2 / (sigma_i sigma_j)),  sigma_i = distance to the k-th neighbour.

The combined Laplacian is L = sum_m a_m (D_m - W_m) with non-negative model
weights a_m summing to one. It is combinatorial (L 1 = 0), symmetric, and has
non-positive off-diagonals, which makes it positive semi-definite by
Gershgorin. The N x N matrix is only ever held in sparse form.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

SUPPORTED_METRICS = ("cosine", "euclidean")


class MultiModelGraphLaplacian:
    """Builds and checks sparse graph Laplacians over one sample set seen by several models."""

    def __init__(self):
        self.laplacian_: Optional[sp.csr_matrix] = None
        self.model_weights_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ build
    @staticmethod
    def knn_affinity(features, k_neighbors: int = 5, metric: str = "cosine") -> sp.csr_matrix:
        """Symmetric sparse kNN affinity matrix W (zero diagonal) for one model."""
        X = np.asarray(features, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("features contain NaN or Inf")
        n = X.shape[0]
        if not (isinstance(k_neighbors, (int, np.integer)) and 1 <= k_neighbors < n):
            raise ValueError(f"k_neighbors must be an int in [1, {n - 1}], got {k_neighbors}")
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}")
        if metric == "cosine":
            norms = np.linalg.norm(X, axis=1)
            if np.any(norms == 0.0):
                raise ValueError("cosine metric is undefined for zero-norm rows")
            # Euclidean distance between unit vectors is monotone in cosine distance.
            X = X / norms[:, None]

        dist, idx = cKDTree(X).query(X, k=k_neighbors + 1)
        dist, idx = dist[:, 1:], idx[:, 1:]  # drop the self match
        sigma = dist[:, -1]
        if np.any(sigma == 0.0):
            bad = int(np.flatnonzero(sigma == 0.0)[0])
            raise ValueError(
                f"row {bad} has more than k_neighbors={k_neighbors} exact duplicates; "
                "deduplicate the features or raise k_neighbors"
            )
        rows = np.repeat(np.arange(n), k_neighbors)
        cols = idx.ravel()
        w = np.exp(-(dist.ravel() ** 2) / (sigma[rows] * sigma[cols]))
        W = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
        # Symmetrise with max so that an edge found from either side keeps its weight.
        return W.maximum(W.T).tocsr()

    @staticmethod
    def laplacian_from_affinity(W: sp.spmatrix) -> sp.csr_matrix:
        W = sp.csr_matrix(W, dtype=np.float64)
        W = (W - sp.diags(W.diagonal())).tocsr()
        W.eliminate_zeros()
        deg = np.asarray(W.sum(axis=1)).ravel()
        return (sp.diags(deg) - W).tocsr()

    def build_sparse_laplacian(
        self,
        feature_list: Sequence,
        k_neighbors: int = 5,
        metric: str = "cosine",
        weights: Optional[Sequence[float]] = None,
    ) -> sp.csr_matrix:
        """Return the combined sparse Laplacian L = sum_m a_m L_m over all models."""
        if isinstance(feature_list, np.ndarray) and feature_list.ndim == 2:
            feature_list = [feature_list]
        if len(feature_list) == 0:
            raise ValueError("feature_list must contain at least one feature matrix")
        n = np.asarray(feature_list[0]).shape[0]
        for m, X in enumerate(feature_list):
            if np.asarray(X).shape[0] != n:
                raise ValueError(f"model {m} has {np.asarray(X).shape[0]} rows, expected {n}")
        if weights is None:
            a = np.full(len(feature_list), 1.0 / len(feature_list))
        else:
            a = np.asarray(weights, dtype=np.float64)
            if a.shape != (len(feature_list),):
                raise ValueError(f"weights must have length {len(feature_list)}")
            if not np.all(np.isfinite(a)) or np.any(a < 0) or a.sum() <= 0:
                raise ValueError("weights must be finite, non-negative and not all zero")
            a = a / a.sum()

        L = sp.csr_matrix((n, n))
        for a_m, X in zip(a, feature_list):
            if a_m == 0.0:
                continue
            L = L + a_m * self.laplacian_from_affinity(self.knn_affinity(X, k_neighbors, metric))
        L = L.tocsr()
        self.verify_laplacian(L, n_nodes=n)
        self.laplacian_ = L
        self.model_weights_ = a
        return L

    # --------------------------------------------------------------- operator
    @staticmethod
    def quadratic_operator(Z, L) -> np.ndarray:
        """Z^T L Z (d x d) via one sparse mat-mat product: O(nnz(L) d + N d^2), no N x N dense."""
        Z = np.asarray(Z, dtype=np.float64)
        LZ = L @ Z
        G = Z.T @ np.asarray(LZ)
        return 0.5 * (G + G.T)

    @staticmethod
    def pairwise_quadratic_operator(Z, L) -> np.ndarray:
        """Z^T L Z from edge differences: sum_{i<j} w_ij (z_i - z_j)(z_i - z_j)^T.

        Valid for combinatorial Laplacians, where w_ij = -L_ij and the diagonal is the degree.
        """
        Z = np.asarray(Z, dtype=np.float64)
        U = sp.triu(sp.csr_matrix(L), k=1).tocoo()
        w = -U.data
        D = Z[U.row] - Z[U.col]
        return (D * w[:, None]).T @ D

    @staticmethod
    def dirichlet_energy(F, L) -> float:
        """tr(F^T L F), the graph roughness of outputs F (N x k)."""
        F = np.asarray(F, dtype=np.float64)
        if F.ndim == 1:
            F = F[:, None]
        return float(np.sum(F * np.asarray(L @ F)))

    # ----------------------------------------------------------------- checks
    @staticmethod
    def verify_laplacian(L, n_nodes: Optional[int] = None, atol: float = 1e-9) -> None:
        """Raise ValueError unless L is a symmetric combinatorial Laplacian (hence PSD).

        Checks: square, finite, symmetric, off-diagonals <= 0, rows sum to 0. Together
        these give diagonal dominance with a non-negative diagonal, so x^T L x >= 0 for
        every x (Gershgorin). Cost is O(nnz); no eigendecomposition of the N x N matrix.
        """
        Ls = sp.csr_matrix(L, dtype=np.float64)
        if Ls.shape[0] != Ls.shape[1]:
            raise ValueError(f"L must be square, got {Ls.shape}")
        if n_nodes is not None and Ls.shape[0] != n_nodes:
            raise ValueError(f"L must be ({n_nodes}, {n_nodes}), got {Ls.shape}")
        if not np.all(np.isfinite(Ls.data)):
            raise ValueError("L contains NaN or Inf")
        scale = max(1.0, float(abs(Ls).max()) if Ls.nnz else 1.0)
        asym = abs(Ls - Ls.T)
        if asym.nnz and asym.max() > atol * scale:
            raise ValueError(f"L is not symmetric (max |L - L^T| = {asym.max():.3e})")
        off = Ls - sp.diags(Ls.diagonal())
        if off.nnz and off.max() > atol * scale:
            raise ValueError("L has positive off-diagonal entries; not a graph Laplacian")
        row_sums = np.asarray(Ls.sum(axis=1)).ravel()
        if np.any(np.abs(row_sums) > atol * scale * max(1, Ls.shape[0])):
            raise ValueError(
                f"L rows do not sum to zero (max |L 1| = {np.abs(row_sums).max():.3e}); "
                "only combinatorial Laplacians are accepted"
            )
