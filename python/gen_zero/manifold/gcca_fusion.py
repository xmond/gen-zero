"""Regularized MAX-VAR GCCA shared/private mid-fusion for heterogeneous model features.

Derivation (regularized MAX-VAR GCCA):
    min_{G^T G = I, {A_m}} sum_m omega_m ( ||G - Z_m A_m||_F^2 + lambda_m ||A_m||_F^2 )

For fixed G, A_m is the ridge-regression solution A_m = (Z_m^T Z_m + lambda_m I)^{-1} Z_m^T G,
and the minimum residual of the ridge fit is a standard identity:
    ||G - Z_m A_m*||_F^2 + lambda_m ||A_m*||_F^2 = tr(G^T G) - tr(G^T P_m G)
where P_m = Z_m (Z_m^T Z_m + lambda_m I)^{-1} Z_m^T is the ridge hat matrix. Since
tr(G^T G) = r is constant under the G^T G = I constraint, minimizing the sum over m
is equivalent to maximizing sum_m omega_m tr(G^T P_m G) = ||B^T G||_F^2, where
B = [ sqrt(omega_m) * Z_m (Z_m^T Z_m + lambda_m I)^{-1/2} ]_m is the horizontal block
concatenation across views. This is maximized by taking G as the top-r left singular
vectors of B — an N x sum(d_m) matrix, so the N x N matrix B @ B^T is never formed.

Private residuals: A_m's column space (in the original d_m-dim feature space) captures
the directions of view m that explain the shared coordinate G. Projecting those directions
out of Z_m and retaining the top principal components of what remains gives each view a
residual channel disjoint from the shared bottleneck; whether that residual is
discriminative for any downstream task is untested here
and must be measured on real paired-model features, not assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

FORMAT_VERSION = 1


def _ridge_inverse_and_inverse_sqrt(gram: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose `gram + reg * I` and return (its inverse, its inverse square root)."""
    dim = gram.shape[0]
    eigvals, eigvecs = np.linalg.eigh(gram + reg * np.eye(dim))
    if np.any(eigvals <= 0):
        raise ValueError(
            "gram + reg * I is not positive definite (smallest eigenvalue "
            f"{eigvals.min():.3e} <= 0); increase reg"
        )
    inv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
    inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return inv, inv_sqrt


def _numeric_rank(singular_values: np.ndarray, ambient_dim: int) -> int:
    """Count singular values distinguishable from the zero/noise floor at working precision.

    Returns 0 for an all-zero (or empty) spectrum — a rank of 0 is the correct answer for a
    degenerate encoder or an already-fully-explained residual, and callers (`fit`) must accept
    that a view can legitimately contribute an empty shared basis or empty residual block.
    """
    if singular_values.size == 0 or singular_values[0] <= 0:
        return 0
    tol = singular_values[0] * ambient_dim * np.finfo(singular_values.dtype).eps
    return int(np.sum(singular_values > max(tol, 1e-10)))


def _validate_views(feature_list: Sequence[np.ndarray], label: str) -> int:
    if len(feature_list) == 0:
        raise ValueError(f"{label} must contain at least one view")
    n_samples = feature_list[0].shape[0]
    for i, z in enumerate(feature_list):
        if z.ndim != 2:
            raise ValueError(f"{label}[{i}] must be a 2D (n_samples, n_features) array, got shape {z.shape}")
        if z.shape[0] != n_samples:
            raise ValueError(
                f"all views must share the same number of aligned samples: view 0 has {n_samples}, "
                f"view {i} has {z.shape[0]}"
            )
        if not np.all(np.isfinite(z)):
            raise ValueError(f"{label}[{i}] contains non-finite values (NaN or Inf)")
    return n_samples


def _payload_sha256(payload: dict, meta: Optional[dict] = None) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(payload[name]).tobytes())
    if meta is not None:
        for k in sorted(meta):
            if k == "payload_sha256":
                continue
            v = meta[k]
            digest.update(f"{k}:{json.dumps(v, sort_keys=True)}".encode())
    return digest.hexdigest()


@dataclass
class _ViewState:
    mean: np.ndarray                  # (1, d_m)
    encoder: np.ndarray                # A_m, (d_m, shared_dim)
    shared_basis: np.ndarray           # U_m = orth(A_m), (d_m, rank_m)
    residual_directions: np.ndarray    # (d_m, residual_dim_m)
    weight: float


class GCCAMidFusion:
    """Regularized GCCA shared manifold + orthogonal private-residual mid-fusion.

    `fit` learns, per view m: a ridge encoder `A_m` into a shared `shared_dim`-dimensional
    space and a private-residual basis orthogonal to `A_m`'s column space (capped at
    `residual_dim` or at the residual's own rank, whichever is smaller — see the
    `_numeric_rank` cap in `fit`, which is what keeps this orthogonal to machine precision
    even when the caller requests more residual dimensions than the view has left after the
    shared directions are removed). `transform` projects new same-view-aligned samples into
    the fused space `[G_hat, r_1, ..., r_M]`.
    """

    def __init__(self) -> None:
        self.shared_dim: Optional[int] = None
        self.reg: Optional[float] = None
        self.views_: List[_ViewState] = []
        self._fitted = False

    def fit(
        self,
        feature_list_train: Sequence[np.ndarray],
        shared_dim: int = 64,
        residual_dim: int = 16,
        reg: float = 1e-3,
        weights: Optional[Sequence[float]] = None,
    ) -> "GCCAMidFusion":
        n_samples = _validate_views(feature_list_train, "feature_list_train")
        if shared_dim < 1:
            raise ValueError(f"shared_dim must be >= 1, got {shared_dim}")
        if residual_dim < 0:
            raise ValueError(f"residual_dim must be >= 0, got {residual_dim}")
        if reg <= 0:
            raise ValueError(f"reg must be > 0 (ridge regularizer; 0 can make the gram matrix singular), got {reg}")

        n_views = len(feature_list_train)
        if weights is None:
            weights = [1.0] * n_views
        if len(weights) != n_views:
            raise ValueError(f"weights must have one entry per view ({n_views}), got {len(weights)}")
        if any(w <= 0 for w in weights):
            raise ValueError(f"all weights must be > 0, got {list(weights)}")

        total_feature_dim = sum(z.shape[1] for z in feature_list_train)
        max_shared_dim = min(n_samples, total_feature_dim)
        if shared_dim > max_shared_dim:
            raise ValueError(
                f"shared_dim={shared_dim} exceeds the supported maximum {max_shared_dim} "
                f"(min(n_samples={n_samples}, sum of view feature dims={total_feature_dim}))"
            )

        means = [z.mean(axis=0, keepdims=True) for z in feature_list_train]
        centered = [z - m for z, m in zip(feature_list_train, means)]

        blocks = []
        gram_invs = []
        for z, w in zip(centered, weights):
            gram = z.T @ z
            inv, inv_sqrt = _ridge_inverse_and_inverse_sqrt(gram, reg)
            gram_invs.append(inv)
            blocks.append(np.sqrt(w) * (z @ inv_sqrt))

        # B is N x sum(d_m); its top singular vectors solve the MAX-VAR problem without
        # ever forming the N x N matrix B @ B^T (see module docstring for the derivation).
        block_matrix = np.concatenate(blocks, axis=1)
        left_singular_vectors, _singular_values, _ = np.linalg.svd(block_matrix, full_matrices=False)
        shared_coords = left_singular_vectors[:, :shared_dim]

        views: List[_ViewState] = []
        for z, mean, inv, w in zip(centered, means, gram_invs, weights):
            d_m = z.shape[1]
            encoder = inv @ (z.T @ shared_coords)  # A_m: (d_m, shared_dim)

            # Feature-space loading basis: Z_m^T G spans the direction of shared variation in feature space.
            # Projecting orthogonal to this basis eliminates shared signal leakage (R^2 ~ 0).
            loading_matrix = z.T @ shared_coords  # (d_m, shared_dim)
            basis_u, basis_s, _ = np.linalg.svd(loading_matrix, full_matrices=False)
            rank_m = _numeric_rank(basis_s, d_m)
            shared_basis = basis_u[:, :rank_m]

            # The dimension of the orthogonal complement to the shared representation
            # in view m cannot exceed d_m - rank_m.
            max_res_rank = max(0, d_m - rank_m)

            if max_res_rank == 0 or residual_dim == 0:
                residual_directions = np.zeros((d_m, 0), dtype=np.float64)
            else:
                # Regress out shared loading space in feature space
                residual_sample = z - z @ shared_basis @ shared_basis.T
                _, residual_s, residual_vt = np.linalg.svd(residual_sample, full_matrices=False)
                residual_rank = min(max_res_rank, _numeric_rank(residual_s, d_m))
                d_res = min(residual_dim, residual_rank)
                res_dirs = residual_vt[:d_res].T  # (d_m, d_res)

                # Ensure strict numerical orthogonality against shared loading basis
                if shared_basis.shape[1] > 0 and res_dirs.shape[1] > 0:
                    res_dirs = res_dirs - shared_basis @ (shared_basis.T @ res_dirs)
                    q, _ = np.linalg.qr(res_dirs)
                    res_dirs = q[:, :d_res]
                residual_directions = res_dirs

            views.append(_ViewState(
                mean=mean,
                encoder=encoder,
                shared_basis=shared_basis,
                residual_directions=residual_directions,
                weight=w,
            ))

        self.views_ = views
        self.shared_dim = shared_dim
        self.reg = reg
        self._fitted = True
        return self

    def transform(self, feature_list: Sequence[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GCCAMidFusion must be fit() before transform()")
        if len(feature_list) != len(self.views_):
            raise ValueError(
                f"feature_list must match the number of fitted views ({len(self.views_)}), got {len(feature_list)}"
            )

        weighted_sum = None
        weight_total = 0.0
        residual_parts = []
        for z, view in zip(feature_list, self.views_):
            if z.ndim != 2:
                raise ValueError(f"expected 2D feature matrix, got shape {z.shape}")
            if z.shape[1] != view.mean.shape[1]:
                raise ValueError(f"view feature dim {z.shape[1]} does not match the dim fit() saw ({view.mean.shape[1]})")
            if not np.all(np.isfinite(z)):
                raise ValueError("transform input contains non-finite values (NaN or Inf)")

            centered = z - view.mean
            g_estimate = centered @ view.encoder
            weighted_sum = (
                g_estimate * view.weight if weighted_sum is None else weighted_sum + g_estimate * view.weight
            )
            weight_total += view.weight

            if view.residual_directions.shape[1] > 0:
                residual = centered - centered @ view.shared_basis @ view.shared_basis.T
                residual_parts.append(residual @ view.residual_directions)
            else:
                residual_parts.append(np.zeros((z.shape[0], 0), dtype=np.float64))

        shared_coords = weighted_sum / weight_total
        res = np.concatenate([shared_coords] + residual_parts, axis=1)
        if not np.all(np.isfinite(res)):
            raise ValueError("fused representations contain non-finite values")
        return res

    def save(self, path) -> dict:
        if not self._fitted:
            raise RuntimeError("GCCAMidFusion must be fit() before save()")
        payload = {}
        for i, view in enumerate(self.views_):
            payload[f"mean_{i}"] = np.asarray(view.mean, dtype=np.float64)
            payload[f"encoder_{i}"] = np.asarray(view.encoder, dtype=np.float64)
            payload[f"shared_basis_{i}"] = np.asarray(view.shared_basis, dtype=np.float64)
            payload[f"residual_directions_{i}"] = np.asarray(view.residual_directions, dtype=np.float64)
        meta = dict(
            format_version=FORMAT_VERSION,
            n_views=len(self.views_),
            shared_dim=self.shared_dim,
            reg=self.reg,
            weights=[float(view.weight) for view in self.views_],
        )
        meta["payload_sha256"] = _payload_sha256(payload, meta)
        with open(path, "wb") as f:
            np.savez(f, metadata=json.dumps(meta), **payload)
        return meta

    @classmethod
    def load(cls, path) -> "GCCAMidFusion":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"GCCAMidFusion artifact format_version mismatch: expected {FORMAT_VERSION}, "
                    f"got {meta.get('format_version')!r}"
                )
            n_views = int(meta["n_views"])
            payload = {}
            for i in range(n_views):
                for key in ("mean", "encoder", "shared_basis", "residual_directions"):
                    payload[f"{key}_{i}"] = np.asarray(data[f"{key}_{i}"], dtype=np.float64)

        expected_meta = dict(
            format_version=meta["format_version"],
            n_views=meta["n_views"],
            shared_dim=meta["shared_dim"],
            reg=meta["reg"],
            weights=meta["weights"],
        )
        if _payload_sha256(payload, expected_meta) != meta.get("payload_sha256"):
            raise ValueError("GCCAMidFusion artifact payload_sha256 mismatch: corrupted or tampered file")

        model = cls()
        views = []
        for i, w in enumerate(meta["weights"]):
            views.append(_ViewState(
                mean=payload[f"mean_{i}"],
                encoder=payload[f"encoder_{i}"],
                shared_basis=payload[f"shared_basis_{i}"],
                residual_directions=payload[f"residual_directions_{i}"],
                weight=float(w),
            ))
        model.views_ = views
        model.shared_dim = int(meta["shared_dim"])
        model.reg = float(meta["reg"])
        model._fitted = True
        return model


@dataclass
class _AnchorViewState:
    mean: np.ndarray               # (1, d_m)
    anchors_centered: np.ndarray   # (n_anchors, d_m), h_m(a_j) - mu_m


class RelativeAnchorEncoder:
    """Relative anchor representation: phi_m(x)_j = cos(h_m(x) - mu_m, h_m(a_j) - mu_m).

    Maps heterogeneous per-view feature spaces onto a common `n_anchors`-dimensional
    coordinate system built from cosine similarity to a shared, sample-aligned set of
    anchor points. The raw cosine in [-1, 1] is affinely rescaled to [0, 1]
    (`(cos + 1) / 2`, gcca_fusion.py near the end of `transform`) so the output is a
    non-negative, bounded similarity coordinate rather than the signed cosine the
    formula in the module's design doc literally specifies; that rescaling is a
    deliberate, order-preserving reparameterization, not an independent algorithm.
    """

    def __init__(self) -> None:
        self.n_anchors: Optional[int] = None
        self.views_: List[_AnchorViewState] = []
        self.anchor_indices_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        feature_list_train: Sequence[np.ndarray],
        n_anchors: int = 128,
        seed: int = 42,
    ) -> "RelativeAnchorEncoder":
        n_samples = _validate_views(feature_list_train, "feature_list_train")
        if n_anchors < 1:
            raise ValueError(f"n_anchors must be >= 1, got {n_anchors}")
        if n_anchors > n_samples:
            raise ValueError(f"n_anchors={n_anchors} exceeds n_samples={n_samples}")

        rng = np.random.default_rng(seed)
        anchor_indices = np.sort(rng.choice(n_samples, size=n_anchors, replace=False))

        views = []
        for z in feature_list_train:
            mean = z.mean(axis=0, keepdims=True)
            anchors_centered = z[anchor_indices] - mean
            if np.any(np.linalg.norm(anchors_centered, axis=1) < 1e-12):
                raise ValueError(
                    "an anchor point is numerically identical to its view mean "
                    "(zero-norm after centering); cosine similarity is undefined for it"
                )
            views.append(_AnchorViewState(mean=mean, anchors_centered=anchors_centered))

        self.views_ = views
        self.anchor_indices_ = anchor_indices
        self.n_anchors = n_anchors
        self._fitted = True
        return self

    def transform(self, feature_list: Sequence[np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("RelativeAnchorEncoder must be fit() before transform()")
        if len(feature_list) != len(self.views_):
            raise ValueError(
                f"feature_list must match the number of fitted views ({len(self.views_)}), got {len(feature_list)}"
            )

        blocks = []
        for z, view in zip(feature_list, self.views_):
            if z.ndim != 2:
                raise ValueError(f"expected 2D feature matrix, got shape {z.shape}")
            if z.shape[1] != view.mean.shape[1]:
                raise ValueError(f"view feature dim {z.shape[1]} does not match the dim fit() saw ({view.mean.shape[1]})")
            if not np.all(np.isfinite(z)):
                raise ValueError("transform input contains non-finite values (NaN or Inf)")
            centered = z - view.mean
            x_norm = np.linalg.norm(centered, axis=1, keepdims=True)
            if np.any(x_norm < 1e-12):
                raise ValueError("a transform() row is numerically identical to its view mean; cosine is undefined")
            a_norm = np.linalg.norm(view.anchors_centered, axis=1, keepdims=True)
            cosine = (centered @ view.anchors_centered.T) / (x_norm @ a_norm.T)
            cosine = np.clip(cosine, -1.0, 1.0)
            blocks.append((cosine + 1.0) * 0.5)

        res = np.concatenate(blocks, axis=1)
        if not np.all(np.isfinite(res)):
            raise ValueError("encoded anchor representations contain non-finite values")
        return res

    def save(self, path) -> dict:
        if not self._fitted:
            raise RuntimeError("RelativeAnchorEncoder must be fit() before save()")
        payload = {"anchor_indices": np.asarray(self.anchor_indices_, dtype=np.int64)}
        for i, view in enumerate(self.views_):
            payload[f"mean_{i}"] = np.asarray(view.mean, dtype=np.float64)
            payload[f"anchors_centered_{i}"] = np.asarray(view.anchors_centered, dtype=np.float64)
        meta = dict(
            format_version=FORMAT_VERSION,
            n_views=len(self.views_),
            n_anchors=self.n_anchors,
        )
        meta["payload_sha256"] = _payload_sha256(payload, meta)
        with open(path, "wb") as f:
            np.savez(f, metadata=json.dumps(meta), **payload)
        return meta

    @classmethod
    def load(cls, path) -> "RelativeAnchorEncoder":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"RelativeAnchorEncoder artifact format_version mismatch: expected {FORMAT_VERSION}, "
                    f"got {meta.get('format_version')!r}"
                )
            n_views = int(meta["n_views"])
            payload = {"anchor_indices": np.asarray(data["anchor_indices"], dtype=np.int64)}
            for i in range(n_views):
                payload[f"mean_{i}"] = np.asarray(data[f"mean_{i}"], dtype=np.float64)
                payload[f"anchors_centered_{i}"] = np.asarray(data[f"anchors_centered_{i}"], dtype=np.float64)

        expected_meta = dict(
            format_version=meta["format_version"],
            n_views=meta["n_views"],
            n_anchors=meta["n_anchors"],
        )
        if _payload_sha256(payload, expected_meta) != meta.get("payload_sha256"):
            raise ValueError("RelativeAnchorEncoder artifact payload_sha256 mismatch: corrupted or tampered file")

        model = cls()
        views = []
        for i in range(n_views):
            views.append(_AnchorViewState(mean=payload[f"mean_{i}"], anchors_centered=payload[f"anchors_centered_{i}"]))
        model.views_ = views
        model.anchor_indices_ = payload["anchor_indices"]
        model.n_anchors = int(meta["n_anchors"])
        model._fitted = True
        return model
