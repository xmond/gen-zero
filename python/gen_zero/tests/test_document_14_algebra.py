r"""Document 14 Mathematical and Algebraic Verification Suite.

Formally verifies the exact 14 algebraic checks from Appendix A of
docs/architecture/14-gen-zero-ultimate-cpu-lookup-engine.md, as well as
their integration with the production CPU lookup engine.
"""

import math
import re
from pathlib import Path
import numpy as np
import pytest

from gen_zero.nanocore.cpu_lookup_engine import (
    ExactMatchLookupTable,
    SimplexPrototypeCodebook,
)


class TestDocument14AppendixAChecks:
    """The exact 14 numerical checks defined in Document 14 Appendix A lines 387-502."""

    @pytest.mark.parametrize("k", [2, 3, 8, 32])
    def test_etf_multiple_arities_gram_zero_mean_and_rank(self, k: int):
        """Checks etf_k2, etf_k3, etf_k8, etf_k32 from Document 14."""
        h = np.zeros((k, k - 1))
        for j in range(1, k):
            h[:j, j - 1] = 1 / math.sqrt(j * (j + 1))
            h[j, j - 1] = -j / math.sqrt(j * (j + 1))
        v = math.sqrt(k / (k - 1)) * h.T
        gram = k / (k - 1) * (np.eye(k) - np.ones((k, k)) / k)

        assert np.allclose(v.T @ v, gram, atol=1e-12)
        assert np.linalg.norm(v.sum(axis=1)) < 1e-12
        assert np.linalg.matrix_rank(v) == k - 1

    def test_codebook_uses_equiangular_simplex_etf(self):
        """Verifies that SimplexPrototypeCodebook implements the Document 14 Helmert ETF."""
        dim = 8
        labels = tuple(f"action_{i}" for i in range(5))
        codebook = SimplexPrototypeCodebook(labels=labels, dim=dim)
        prototypes = codebook.prototypes  # shape (5, 8)
        K = len(labels)

        # Unit norm
        norms = np.linalg.norm(prototypes, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-12)

        # Mutual coherence is exactly 1 / (K - 1)
        cos_sim = prototypes @ prototypes.T
        off_diag = cos_sim[~np.eye(K, dtype=bool)]
        assert np.allclose(off_diag, -1.0 / (K - 1), atol=1e-12)

    def test_zca_regularized_singular_identity(self):
        """Check zca_regularized_singular from Document 14."""
        rng = np.random.default_rng(20260922)
        u, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        eigs = np.array([0, 0, 0.02, 0.1, 0.3, 1, 2, 5, 9.0])
        eps = 0.03
        cov = (u * eigs) @ u.T
        w = (u * (eigs + eps) ** -0.5) @ u.T
        expected = (u * (eigs / (eigs + eps))) @ u.T

        assert np.allclose(w @ cov @ w.T, expected, atol=1e-11)
        assert not np.allclose(expected, np.eye(9))

    def test_affine_fusion_exact_equivalence(self):
        """Check affine_fusion from Document 14."""
        rng = np.random.default_rng(20260922)
        u, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        eigs = np.array([0, 0, 0.02, 0.1, 0.3, 1, 2, 5, 9.0])
        w = (u * (eigs + 0.03) ** -0.5) @ u.T

        rmat = rng.normal(size=(4, 9))
        mu = rng.normal(size=9)
        x = rng.normal(size=(9, 41))
        centers = rng.normal(size=(4, 7))
        beta = rng.normal(size=7)
        a = rmat @ w
        b = -a @ mu
        scores = centers.T @ (rmat @ (w @ (x - mu[:, None]))) + beta[:, None]
        compact = centers.T @ (a @ x + b[:, None]) + beta[:, None]
        full = (centers.T @ a) @ x + (beta + centers.T @ b)[:, None]

        assert np.allclose(scores, compact, atol=1e-10)
        assert np.allclose(scores, full, atol=1e-10)

    def test_affine_exact_condition_and_collision_impossible(self):
        """Checks affine_exact_condition and affine_collision_impossible from Document 14."""
        rng = np.random.default_rng(20260922)
        s = rng.normal(size=(5, 40))
        aug = np.vstack((s, np.ones(40)))
        theta = rng.normal(size=(3, 6))
        y = theta @ aug
        fit = y @ np.linalg.pinv(aug)
        assert np.allclose(fit @ aug, y, atol=1e-11)

        collision_x = np.array([[2.0, 2.0], [1.0, 1.0]])
        collision_y = np.array([[0.0, 1.0]])
        residual = float(
            np.linalg.norm(
                collision_y @ np.linalg.pinv(collision_x) @ collision_x - collision_y
            )
        )
        assert abs(residual - math.sqrt(0.5)) < 1e-12

    def test_ridge_stationarity(self):
        """Check ridge_stationarity from Document 14."""
        rng = np.random.default_rng(20260922)
        s = rng.normal(size=(5, 40))
        theta = rng.normal(size=(3, 6))
        aug = np.vstack((s, np.ones(40)))
        y = theta @ aug

        xc = s - s.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        lam = 0.2
        g = xc @ xc.T + lam * np.eye(5)
        ar = np.linalg.solve(g, xc @ yc.T).T
        assert np.linalg.norm((ar @ xc - yc) @ xc.T + lam * ar) < 1e-10

    def test_centroid_argmin_argmax_and_pairwise_midpoint_identity(self):
        """Checks centroid_argmin_argmax and pairwise_midpoint_identity from Document 14."""
        rng = np.random.default_rng(20260922)
        f = rng.normal(size=(5, 5))
        m = f.T @ f
        c = rng.normal(size=(7, 5))
        queries = rng.normal(size=(31, 5))
        score = queries @ m @ c.T - 0.5 * np.einsum("ij,jk,ik->i", c, m, c)
        delta = queries[:, None, :] - c[None, :, :]
        dist = np.einsum("nki,ij,nkj->nk", delta, m, delta)

        assert np.array_equal(dist.argmin(1), score.argmax(1))

        err = 0.0
        for i in range(7):
            for j in range(7):
                normal = m @ (c[i] - c[j])
                pair = (queries - (c[i] + c[j]) / 2) @ normal
                err = max(err, float(np.max(np.abs(pair - (score[:, i] - score[:, j])))))
        assert err < 1e-10

    def test_independent_pair_psd_cycle(self):
        """Check independent_pair_psd_cycle from Document 14."""
        c3 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        q3 = np.array([0.0, 2.0])
        vs = [np.array([1.0, 0.5]), np.array([0.0, 1.0]), np.array([1.0, -1.0])]
        cycle = []
        for (i, j), vv in zip(((0, 1), (1, 2), (2, 0)), vs):
            mm = np.outer(vv, vv)  # PSD rank-one
            cycle.append((c3[i] - c3[j]) @ mm @ (q3 - (c3[i] + c3[j]) / 2))
        assert all(v > 0 for v in cycle)

    def test_margin_preservation_under_perturbation(self):
        """Check margin_preservation from Document 14."""
        rng = np.random.default_rng(20260922)
        cs = rng.normal(size=(12, 6))
        cs /= np.linalg.norm(cs, axis=1, keepdims=True)
        q = rng.normal(size=(200, 6))
        perturb = rng.normal(size=q.shape)
        perturb *= 0.001 / np.linalg.norm(perturb, axis=1, keepdims=True)
        s0 = q @ cs.T
        s1 = (q + perturb) @ cs.T
        sorted_s = np.sort(s0, axis=1)
        mask = sorted_s[:, -1] - sorted_s[:, -2] > 0.002
        assert mask.sum() > 100
        assert np.array_equal(s0[mask].argmax(1), s1[mask].argmax(1))

    def test_byte_accounting_formula(self):
        """Check byte_accounting from Document 14."""
        ds, r, k = 768, 128, 32
        unfused = 4 * ds * ds + 4 * r * ds + 4 * k * r + 4 * ds + 4 * k
        fused = 4 * r * ds + 4 * r + 4 * k * r + 4 * k
        full_bytes = 4 * k * ds + 4 * k
        assert (unfused, fused, full_bytes) == (2772096, 410240, 98432)
