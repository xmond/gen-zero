"""Gen-Zero candidate semantic-embedding priors: closed-form zero-shot bilinear compatibility.

Given a K x q matrix ``E`` of candidate-text high-dimensional embeddings, this module fits a
linear map ``A`` (input-manifold space -> candidate-embedding space) so that the induced
per-class classifier ``W0 = A E^T`` scores a manifold point ``z`` against every candidate
purely through semantic similarity, with no per-candidate training example required.

ESZSL closed-form bilinear compatibility (Romera-Paredes & Torr, "An Embarrassingly Simple
Approach to Zero-Shot Learning", ICML 2015), specialized to candidate-text embeddings as the
"attribute" space:

    min_A || Z A E^T - Y ||_F^2 + gamma || A E^T ||_F^2 + delta || Z A ||_F^2 + gamma delta || A ||_F^2

Setting the gradient to zero and grouping the two ridge terms into ``(Z^T Z + gamma I)`` and
``(E^T E + delta I)`` gives the closed-form solution used here:

    A = (Z^T Z + gamma I)^{-1} Z^T Y E (E^T E + delta I)^{-1}

Because ``A`` never depends on the number or identity of the training classes beyond ``Z`` and
``Y``, a new candidate embedding row ``e_new`` can be scored zero-shot: ``s = z A e_new^T``,
with no re-fit.

``compute_prototype_prior`` complements the bilinear classifier with a per-class Bayesian
shrinkage prototype in the *same* d-dimensional manifold space as ``Z``: it blends the
empirical class mean (when samples exist) with the semantic prototype ``A @ E[k]`` (the ESZSL
classifier's own column k, so the two views of the same candidate stay consistent), so a class
with zero training samples degrades gracefully to the pure semantic prior instead of a
division-by-zero or a fabricated mean.
"""

from __future__ import annotations

import numpy as np


class CandidateSemanticPrior:
    """Closed-form ESZSL bilinear compatibility and Bayesian prototype shrinkage over candidate
    text embeddings. Every method is a pure function of its explicit arguments; the class holds
    no fitted state, so callers keep and pass back whatever ``A`` they get from
    ``compute_w0_prior``.
    """

    def compute_w0_prior(
        self,
        Z: np.ndarray,
        Y: np.ndarray,
        E: np.ndarray,
        gamma: float = 10.0,
        delta: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the ESZSL closed-form bilinear classifier.

        Args:
            Z: N x d manifold embeddings of the training inputs.
            Y: N x K one-hot (or real-valued) class-indicator matrix.
            E: K x q candidate-text high-dimensional embeddings, row k aligned with column k of Y.
            gamma: ridge weight on ``A E^T`` (the induced classifier).
            delta: ridge weight on ``Z A`` (the induced manifold projection).

        Returns:
            ``(W0, A)`` where ``W0 = A @ E.T`` is the d x K classifier and ``A`` is the d x q
            manifold-to-candidate-embedding map.
        """
        Z = np.asarray(Z, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        if Z.ndim != 2 or Y.ndim != 2 or E.ndim != 2:
            raise ValueError("Z, Y, E must all be 2-D arrays")
        n, d = Z.shape
        n_y, k = Y.shape
        k_e, q = E.shape
        if n_y != n:
            raise ValueError(f"Z has {n} rows but Y has {n_y} rows")
        if k_e != k:
            raise ValueError(f"Y has {k} columns (classes) but E has {k_e} rows")
        if gamma < 0 or delta < 0:
            raise ValueError("gamma and delta must be non-negative")
        if not (np.isfinite(Z).all() and np.isfinite(Y).all() and np.isfinite(E).all()):
            raise ValueError("Z, Y, E must be finite (no NaN/Inf)")

        left = Z.T @ Z + gamma * np.eye(d)
        mid = Z.T @ Y @ E
        right = E.T @ E + delta * np.eye(q)

        # A = left^{-1} mid right^{-1}. Solved in two triangular-free linear solves rather than
        # explicit inverses (numerically stabler); `right` is symmetric so `X @ right^{-1}` is
        # obtained via `solve(right, X.T).T`.
        step1 = np.linalg.solve(left, mid)
        A = np.linalg.solve(right, step1.T).T

        W0 = A @ E.T
        return W0, A

    def compute_prototype_prior(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        E: np.ndarray,
        kappa: float = 5.0,
        gamma: float = 10.0,
        delta: float = 10.0,
    ) -> np.ndarray:
        """Bayesian shrinkage of empirical class means toward the ESZSL semantic prototype.

            mu_k = (n_k * z_bar_k + kappa * p_k) / (n_k + kappa)

        ``p_k = A @ E[k]`` is column k of the ESZSL classifier fit on ``(Z, y, E)``, living in
        the same d-dimensional manifold space as ``z_bar_k`` (so the blend is dimensionally
        valid without requiring q == d). ``n_k = 0`` collapses the blend exactly to ``p_k``.

        Args:
            Z: N x d manifold embeddings of the training inputs.
            y: length-N integer class labels in [0, K), K = E.shape[0].
            E: K x q candidate-text embeddings.
            kappa: prior pseudo-count; larger values weight the semantic prior more.
            gamma, delta: ridge weights forwarded to ``compute_w0_prior``.

        Returns:
            K x d matrix of shrunk prototypes, one row per candidate class.
        """
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y)
        E = np.asarray(E, dtype=np.float64)
        if Z.ndim != 2:
            raise ValueError("Z must be a 2-D array")
        if y.ndim != 1 or y.shape[0] != Z.shape[0]:
            raise ValueError("y must be 1-D with the same length as Z's rows")
        if E.ndim != 2:
            raise ValueError("E must be a 2-D array")
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"y must hold integer class labels, got dtype {y.dtype}")

        n, d = Z.shape
        k = E.shape[0]
        if n and (y.min() < 0 or y.max() >= k):
            raise ValueError(f"y labels must lie in [0, {k}) to index E's {k} candidates")

        Y = np.zeros((n, k), dtype=np.float64)
        if n:
            Y[np.arange(n), y] = 1.0

        _, A = self.compute_w0_prior(Z, Y, E, gamma=gamma, delta=delta)
        semantic_prototypes = (A @ E.T).T  # K x d, row k = A @ E[k]

        prototypes = np.empty((k, d), dtype=np.float64)
        for cls in range(k):
            mask = y == cls
            n_k = int(mask.sum())
            p_k = semantic_prototypes[cls]
            z_bar_k = Z[mask].mean(axis=0) if n_k else np.zeros(d, dtype=np.float64)
            prototypes[cls] = (n_k * z_bar_k + kappa * p_k) / (n_k + kappa)
        return prototypes

    def score_candidates(self, Z_test: np.ndarray, E_candidates: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Zero-shot compatibility scores ``S = Z_test @ A @ E_candidates.T``.

        ``E_candidates`` need not be (and, for a genuine zero-shot test, should not be) the same
        candidate set ``A`` was fit on: any K_new x q embedding matrix scores directly, with no
        re-fit, because ``A`` only depends on the manifold and the training candidates' labels.
        """
        Z_test = np.asarray(Z_test, dtype=np.float64)
        E_candidates = np.asarray(E_candidates, dtype=np.float64)
        A = np.asarray(A, dtype=np.float64)
        if Z_test.ndim != 2 or E_candidates.ndim != 2 or A.ndim != 2:
            raise ValueError("Z_test, E_candidates, A must all be 2-D arrays")
        if Z_test.shape[1] != A.shape[0]:
            raise ValueError(f"Z_test has {Z_test.shape[1]} columns but A expects {A.shape[0]}")
        if E_candidates.shape[1] != A.shape[1]:
            raise ValueError(f"E_candidates has {E_candidates.shape[1]} columns but A expects {A.shape[1]}")
        return Z_test @ A @ E_candidates.T

    @staticmethod
    def shuffle_control(E: np.ndarray, seed: int = 42) -> np.ndarray:
        """Row-permute ``E`` deterministically, for an ablation that isolates whether a gain
        comes from real candidate semantics or merely from the ridge regularization: refit with
        ``shuffle_control(E)`` in place of ``E`` and compare. Guarantees a genuine permutation
        (never the identity) whenever there is more than one row to shuffle.
        """
        E = np.asarray(E, dtype=np.float64)
        if E.ndim != 2:
            raise ValueError("E must be a 2-D array")
        k = E.shape[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(k)
        if k > 1 and np.array_equal(perm, np.arange(k)):
            perm = np.roll(perm, 1)
        return E[perm]
