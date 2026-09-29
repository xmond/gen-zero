"""Generalized CCA (MAXVAR) + orthogonal-Procrustes manifold interference operator.

Implements the shared-subspace step of Spec 31 section 4.2 (``docs/zero/31-multiscale-dense-resonance-
etf-dual-process-plan.md``): for M row-aligned feature views ``X_m`` (n x d_m), find the shared
coordinates ``G`` (n x s, ``G^T G = I``) as the top eigenvectors of ``sum_m P_m``, with the ridge
projector ``P_m = U_m (U_m^T U_m + lambda I)^{-1} U_m^T``. Reported per fit:

  * pairwise canonical correlations rho_1 >= rho_2 >= ... for every view pair;
  * the MAXVAR eigenvalues of ``sum_m P_m`` and a "resonance spectrum" ``(lambda_k - 1) / (M - 1)``
    clipped to [0, 1] (for M = 2 and no ridge this equals the canonical correlations exactly);
  * the participation ratio and the entropy effective rank of the squared resonance spectrum;
  * per view, the fraction of centered energy orthogonal to the top-s shared coordinates.

Solved in sample space, never with d x d covariances. With the thin SVD ``X_c = Q S V^T`` and
``sigma^2 = s^2 / (n - 1)``, the ridge-whitened cross-covariance
``(C_11 + r_1 I)^{-1/2} C_12 (C_22 + r_2 I)^{-1/2}`` equals ``V_1 (W_1^T W_2) V_2^T`` with
``W = Q diag(sigma / sqrt(sigma^2 + r))``, so the regularized generalized eigenproblem
``[0 C_12; C_21 0] v = rho diag(C_11 + r_1 I, C_22 + r_2 I) v`` has its positive eigenvalues equal to
the singular values of the small matrix ``W_1^T W_2`` (verified against the primal route in the
tests). ``sum_m W_m W_m^T = Z Z^T`` for ``Z = [W_1 | ... | W_M]``, so MAXVAR is one thin SVD of Z.
Cost is O(n^2 d) per view: seconds at n = 100, d = 8192, versus 288 s per block for a d x d SVD
(measured in ``cross_model_manifold_alignment.py``).

The ridge is scale-relative: ``r_m = reg * mean(retained sigma^2)``. Raw last-token LLM states carry
a few massive-activation dimensions, so an absolute ridge would mean something different per model.

WARNING, read before quoting any number from this module. When n <= min(d_1, d_2) and both
views span the centered sample space (the normal case here: n = 100..1000 rows against d = 8192),
unregularized in-sample canonical correlations are ALL exactly 1 for any two full-rank views,
paired or shuffled: the two row spaces are the same (n - 1)-dim space. In-sample rho therefore
measures nothing at that shape. Only held-out rho (``heldout_canonical_correlations``, fit on some
rows, evaluated on others) compared against a row-permutation null (``permutation_null``) carries
information, and ``max_rank`` (PCA truncation, Spec 31 section 4.1 step 2, with max_rank < n_fit / 4)
keeps the fit well conditioned.

Procrustes: ``R* = argmin ||A R - B||_F`` s.t. ``R^T R = I`` on centered, Frobenius-normalized
views. Unequal widths are zero-padded to max(d_1, d_2) so R is square. This matters: the rectangular
SVD formula is only the true minimizer when the narrower block is mapped into the wider space (see the
module docstring of ``cross_model_manifold_alignment.py``); padding makes the problem square and the
SVD solution exact. Reflections are allowed (R orthogonal, not necessarily det +1). The residual is
closed form, ``(||A||^2 + ||B||^2 - 2 ||A^T B||_*) / ||B||^2``, from the n x n core, and is quantized
by float64 round-off near 1e-7. A dense R is only materialized up to ``dense_limit`` columns; above
that the call returns the low-rank factors (``R`` restricted to the data row spaces is
``left @ right.T``) and refuses to build a dense matrix instead of silently approximating it.

Every input is validated fail-closed: fewer than 2 views, row counts that differ or are < 2,
non-2D arrays, non-real dtypes, NaN/Inf, or a zero-variance view raise ``ValueError``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent

Pair = Tuple[int, int]


def _validate_view(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_ or not (np.issubdtype(array.dtype, np.floating)
                                       or np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{name}: features must be real floating-point numbers, got dtype {array.dtype}")
    if array.ndim != 2:
        raise ValueError(f"{name}: must be a 2D (n_samples, n_features) array, got shape {array.shape}")
    if array.shape[0] < 2:
        raise ValueError(f"{name}: needs at least 2 samples, got {array.shape[0]}")
    if array.shape[1] < 1:
        raise ValueError(f"{name}: needs at least 1 feature, got {array.shape[1]}")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: contains non-finite (NaN/Inf) values")
    if np.all(np.var(array, axis=0) == 0.0) or np.max(np.abs(array - array[0])) < 1e-12:
        raise ValueError(f"{name}: zero variance (every row identical)")
    return array


def validate_views(views: Sequence, min_views: int = 2) -> List[np.ndarray]:
    """Validate a list of row-aligned feature matrices; raise ``ValueError`` on any defect."""
    if isinstance(views, np.ndarray) or len(views) < min_views:
        raise ValueError(f"need a sequence of at least {min_views} feature matrices")
    arrays = [_validate_view(v, f"view[{i}]") for i, v in enumerate(views)]
    counts = {a.shape[0] for a in arrays}
    if len(counts) != 1:
        raise ValueError(f"all views must have the same number of samples, got {[a.shape[0] for a in arrays]}")
    return arrays


def _thin_svd(centered: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Left singular vectors and singular values on the numerical rank support."""
    q, s, _ = np.linalg.svd(centered, full_matrices=False)
    tol = (s[0] if s.size else 0.0) * max(centered.shape) * np.finfo(np.float64).eps
    keep = s > tol
    if not np.any(keep):
        raise ValueError(f"{name}: zero variance after centering (every row identical)")
    return q[:, keep], s[keep]


# The resonance spectrum lives in [0, 1]; values below this are float64 round-off, not shared
# structure. Without the floor, exactly orthogonal views report an effective rank of ~6 made of 1e-16s.
SPECTRUM_FLOOR = 1e-10


def _effective_ranks(spectrum: np.ndarray) -> Tuple[float, float]:
    """Participation ratio and entropy effective rank of ``spectrum**2``; (0, 0) if it is all zero."""
    spectrum = np.where(spectrum > SPECTRUM_FLOOR, spectrum, 0.0)
    energy = np.square(spectrum)
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0, 0.0
    pr = total ** 2 / float(np.square(energy).sum())
    p = energy[energy > 0] / total
    return float(pr), float(np.exp(-np.sum(p * np.log(p))))


@dataclass(frozen=True)
class ProcrustesResult:
    residual_ratio: float          # ||A R - B||_F^2 / ||B||_F^2 on the preprocessed views
    singular_values: np.ndarray    # of A^T B, descending (their sum is the nuclear norm)
    left: np.ndarray               # (d, k): R restricted to the data row spaces is left @ right.T
    right: np.ndarray              # (d, k)
    padded_dim: int                # both views were zero-padded to this width
    rotation: Optional[np.ndarray] = None  # dense (d, d) orthogonal R, only if d <= dense_limit


@dataclass(frozen=True)
class GCCAResult:
    n_samples: int
    view_dims: Tuple[int, ...]
    view_ranks: Tuple[int, ...]
    ridges: Tuple[float, ...]
    canonical_correlations: Dict[Pair, np.ndarray]
    maxvar_eigenvalues: np.ndarray
    resonance_spectrum: np.ndarray
    participation_ratio: float
    entropy_effective_rank: float
    n_shared: int
    shared_basis: np.ndarray       # G, (n, n_shared), orthonormal columns
    orthogonal_energy_ratio: np.ndarray  # per view, energy outside span(G)
    canonical_weights: Dict[Pair, Tuple[np.ndarray, np.ndarray]] = field(repr=False, default_factory=dict)
    means: Tuple[np.ndarray, ...] = field(repr=False, default=())
    scales: Tuple[np.ndarray, ...] = field(repr=False, default=())

    maps: Tuple[np.ndarray, ...] = field(repr=False, default=())
    fit_data_sha256: Tuple[str, ...] = field(repr=False, default=())

    def transform(self, samples, view: int = 0) -> np.ndarray:
        if not 0 <= view < len(self.maps):
            raise ValueError("unknown GCCA view")
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.view_dims[view] or not np.isfinite(x).all():
            raise ValueError("invalid GCCA transform input")
        out = ((x - self.means[view]) / self.scales[view]) @ self.maps[view]
        if not np.isfinite(out).all():
            raise ValueError("non-finite GCCA transform output")
        return out

    def save_transform(self, path, view: int, *, source_model: str, layer: str, norm: str, source_data=None, fit_indices=None, eval_indices=None):
        if not all(isinstance(v, str) and v.strip() for v in (source_model, layer, norm)):
            raise ValueError("source_model, layer and norm are required")
        payload = {"W": self.maps[view], "mean": self.means[view], "scale": self.scales[view]}
        digest = hashlib.sha256()
        for key in sorted(payload):
            digest.update(np.ascontiguousarray(payload[key], dtype=np.float64).tobytes())
        manifest = dict(source_model=source_model, layer=layer, norm=norm,
                        GCCA_map=digest.hexdigest())
        if source_data is not None:
            rows = np.asarray(source_data, dtype=np.float64)
            fi, ei = np.asarray(fit_indices), np.asarray(eval_indices)
            if (fi.ndim != 1 or ei.ndim != 1 or fi.dtype.kind not in "iu" or ei.dtype.kind not in "iu"
                    or fi.size != self.n_samples or ei.size < 3
                    or np.any(fi < 0) or np.any(ei < 0) or np.any(fi >= len(rows)) or np.any(ei >= len(rows))
                    or len(np.unique(np.concatenate([fi, ei]))) != len(fi) + len(ei)):
                raise ValueError("invalid GCCA fit/eval partition")
            if hashlib.sha256(np.ascontiguousarray(rows[fi]).tobytes()).hexdigest() != self.fit_data_sha256[view]:
                raise ValueError("partition does not match GCCA fitted rows")
            payload.update(source_data_sha256=hashlib.sha256(np.ascontiguousarray(source_data, dtype=np.float64).tobytes()).hexdigest(),
                           fit_indices=np.asarray(fit_indices), eval_indices=np.asarray(eval_indices))
        manifest_text = json.dumps(manifest, sort_keys=True)
        artifact_digest = hashlib.sha256(manifest_text.encode())
        for key in sorted(payload):
            value = np.ascontiguousarray(payload[key])
            artifact_digest.update(key.encode())
            artifact_digest.update(str(value.dtype).encode())
            artifact_digest.update(str(value.shape).encode())
            artifact_digest.update(value.tobytes())
        np.savez(path, **payload, manifest=manifest_text, artifact_sha256=artifact_digest.hexdigest())
        return manifest

    def summary(self) -> Dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "view_dims": list(self.view_dims),
            "view_ranks": list(self.view_ranks),
            "ridges": list(self.ridges),
            "canonical_correlations": {f"{i}-{j}": rho.tolist() for (i, j), rho in self.canonical_correlations.items()},
            "maxvar_eigenvalues": self.maxvar_eigenvalues.tolist(),
            "resonance_spectrum": self.resonance_spectrum.tolist(),
            "participation_ratio": self.participation_ratio,
            "entropy_effective_rank": self.entropy_effective_rank,
            "n_shared": self.n_shared,
            "orthogonal_energy_ratio": self.orthogonal_energy_ratio.tolist(),
        }


class GeneralizedCCAManifoldInterference:
    """MAXVAR generalized CCA plus orthogonal Procrustes over row-aligned feature views.

    reg: scale-relative ridge (>= 0); r_m = reg * mean retained eigenvalue of view m.
    max_rank: keep only the top-r principal components per view (PCA pre-reduction); None = all.
    n_shared: width s of the shared basis G; None = min view rank.
    standardize: z-score every feature with fit-row statistics before centering (constant
        features are left centered, not divided by zero).
    """

    def __init__(self, reg: float = 1e-3, max_rank: Optional[int] = None,
                 n_shared: Optional[int] = None, standardize: bool = False):
        if not (isinstance(reg, (int, float)) and np.isfinite(reg) and reg >= 0):
            raise ValueError(f"reg must be a finite number >= 0, got {reg!r}")
        for name, value in (("max_rank", max_rank), ("n_shared", n_shared)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                                       or value < 1):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
        self.reg = float(reg)
        self.max_rank = max_rank
        self.n_shared = n_shared
        self.standardize = standardize

    # ------------------------------------------------------------------ preprocessing

    def _preprocess(self, arrays: Sequence[np.ndarray]):
        means, scales, centered = [], [], []
        for a in arrays:
            mean = a.mean(axis=0)
            scale = np.ones(a.shape[1])
            if self.standardize:
                std = a.std(axis=0, ddof=1)
                scale = np.where(std > 0, std, 1.0)
            means.append(mean)
            scales.append(scale)
            centered.append((a - mean) / scale)
        return means, scales, centered

    # ------------------------------------------------------------------ GCCA

    def fit(self, views: Sequence) -> GCCAResult:
        arrays = validate_views(views)
        n = arrays[0].shape[0]
        means, scales, centered = self._preprocess(arrays)

        full_svds, ws, ridges, vs, sig2s = [], [], [], [], []
        for i, xc in enumerate(centered):
            q_full, s_full = _thin_svd(xc, f"view[{i}]")
            full_svds.append((q_full, s_full))
            r = s_full.size if self.max_rank is None else min(self.max_rank, s_full.size)
            q, s = q_full[:, :r], s_full[:r]
            sig2 = s ** 2 / (n - 1)
            ridge = self.reg * float(sig2.mean())
            ws.append(q * np.sqrt(sig2 / (sig2 + ridge)))
            ridges.append(ridge)
            sig2s.append(sig2)
            # Primal loadings V = X_c^T Q S^{-1}; only needed for canonical weights (held-out scores).
            vs.append((xc.T @ q) / s)

        correlations: Dict[Pair, np.ndarray] = {}
        weights: Dict[Pair, Tuple[np.ndarray, np.ndarray]] = {}
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, rho, bt = np.linalg.svd(ws[i].T @ ws[j], full_matrices=False)
                correlations[(i, j)] = np.clip(rho, 0.0, 1.0)
                weights[(i, j)] = (vs[i] @ (a / np.sqrt(sig2s[i] + ridges[i])[:, None]),
                                   vs[j] @ (bt.T / np.sqrt(sig2s[j] + ridges[j])[:, None]))

        m = len(ws)
        g_full, z_s, _ = np.linalg.svd(np.hstack(ws), full_matrices=False)
        eigenvalues = z_s ** 2
        resonance = np.clip((eigenvalues - 1.0) / (m - 1), 0.0, 1.0)
        pr, erank = _effective_ranks(resonance)

        min_rank = min(w.shape[1] for w in ws)
        s_dim = min_rank if self.n_shared is None else self.n_shared
        if s_dim > min_rank:
            raise ValueError(f"n_shared={s_dim} exceeds the available shared dimensions {min_rank}")
        g = g_full[:, :s_dim]
        orth = []
        for q_full, s_full in full_svds:
            captured = float(np.sum(np.square(g.T @ (q_full * s_full))))
            orth.append(max(0.0, 1.0 - captured / float(np.sum(s_full ** 2))))

        return GCCAResult(
            n_samples=n,
            view_dims=tuple(a.shape[1] for a in arrays),
            view_ranks=tuple(w.shape[1] for w in ws),
            ridges=tuple(ridges),
            canonical_correlations=correlations,
            maxvar_eigenvalues=eigenvalues,
            resonance_spectrum=resonance,
            participation_ratio=pr,
            entropy_effective_rank=erank,
            n_shared=s_dim,
            shared_basis=g,
            orthogonal_energy_ratio=np.asarray(orth),
            canonical_weights=weights,
            means=tuple(means),
            scales=tuple(scales),
            fit_data_sha256=tuple(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in arrays),
            maps=tuple(v @ ((np.sqrt(sig2 * (n - 1)) /
                            ((sig2 + ridge) * (n - 1)))[:, None] * (q[:, :v.shape[1]].T @ g))
                       for v, sig2, ridge, (q, _) in zip(vs, sig2s, ridges, full_svds)),
        )

    def heldout_canonical_correlations(self, fit_views: Sequence, eval_views: Sequence,
                                       pair: Pair = (0, 1)) -> np.ndarray:
        """Correlation of the canonical scores on rows not used for fitting, per direction.

        This is the only rho that is informative when n_fit <= d. Order follows the in-sample
        directions, so the sequence is NOT guaranteed to be descending.
        """
        fit_arrays = validate_views(fit_views)
        eval_arrays = validate_views(eval_views)
        if [a.shape[1] for a in fit_arrays] != [a.shape[1] for a in eval_arrays]:
            raise ValueError("fit and eval views must have the same feature dimensions")
        result = self.fit(fit_arrays)
        if pair not in result.canonical_weights:
            raise ValueError(f"pair {pair} not available; views are indexed 0..{len(fit_arrays) - 1}")
        i, j = pair
        wa, wb = result.canonical_weights[pair]
        sa = ((eval_arrays[i] - result.means[i]) / result.scales[i]) @ wa
        sb = ((eval_arrays[j] - result.means[j]) / result.scales[j]) @ wb
        sa = sa - sa.mean(axis=0)
        sb = sb - sb.mean(axis=0)
        norm_a = np.linalg.norm(sa, axis=0)
        norm_b = np.linalg.norm(sb, axis=0)
        if np.any(norm_a <= 1e-12) or np.any(norm_b <= 1e-12):
            raise ValueError("eval views produced zero-variance canonical projection directions")
        denom = norm_a * norm_b
        if not np.all(np.isfinite(denom)):
            raise ValueError("overflow in canonical correlation norm calculation")
        with np.errstate(invalid="raise", divide="raise", over="raise"):
            try:
                rho = np.sum(sa * sb, axis=0) / denom
            except FloatingPointError as exc:
                raise ValueError(f"overflow/underflow in canonical correlation computation: {exc}") from exc
        if not np.all(np.isfinite(rho)):
            raise ValueError("heldout canonical correlations contain non-finite (NaN/Inf) values")
        return rho

    def permutation_null(self, fit_views: Sequence, eval_views: Sequence, permutations: int = 20,
                         seed: int = 0, pair: Pair = (0, 1), quantile: float = 0.95) -> Dict[str, object]:
        """Held-out rho after shuffling the rows of every view except view 0 (fit and eval rows
        independently). Breaks pairing, keeps every marginal statistic. Returns the per-direction
        ``quantile`` of the null, to be compared against ``heldout_canonical_correlations``."""
        if permutations < 1:
            raise ValueError("permutations must be >= 1")
        fit_arrays = validate_views(fit_views)
        eval_arrays = validate_views(eval_views)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(permutations):
            fp = [fit_arrays[0]] + [a[rng.permutation(a.shape[0])] for a in fit_arrays[1:]]
            ep = [eval_arrays[0]] + [a[rng.permutation(a.shape[0])] for a in eval_arrays[1:]]
            draws.append(self.heldout_canonical_correlations(fp, ep, pair))
        stacked = np.vstack(draws)
        return {"quantile": quantile, "per_direction": np.quantile(stacked, quantile, axis=0),
                "mean": stacked.mean(axis=0), "permutations": permutations}

    # ------------------------------------------------------------------ Procrustes

    def procrustes(self, source, target, dense_limit: int = 2048,
                   center: bool = True, normalize: bool = True) -> ProcrustesResult:
        """Orthogonal Procrustes ``argmin_R ||A R - B||_F`` s.t. ``R^T R = I`` (see module docstring)."""
        a, b = validate_views([source, target])
        if center:
            a = a - a.mean(axis=0)
            b = b - b.mean(axis=0)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            raise ValueError("Procrustes needs non-zero source and target after centering")
        if normalize:
            a, b = a / na, b / nb
            na = nb = 1.0
        d = max(a.shape[1], b.shape[1])
        qa, sa = _thin_svd(a, "source")
        qb, sb = _thin_svd(b, "target")
        va = (a.T @ qa) / sa   # (d_a, r_a), orthonormal columns
        vb = (b.T @ qb) / sb
        p, sv, qt = np.linalg.svd((sa[:, None] * (qa.T @ qb)) * sb[None, :], full_matrices=False)
        left = np.zeros((d, p.shape[1]))
        right = np.zeros((d, qt.shape[0]))
        left[: a.shape[1]] = va @ p
        right[: b.shape[1]] = vb @ qt.T
        residual = max(0.0, (na ** 2 + nb ** 2 - 2.0 * float(sv.sum())) / nb ** 2)
        rotation = None
        if d <= dense_limit:
            ap = np.zeros((a.shape[0], d)); ap[:, : a.shape[1]] = a
            bp = np.zeros((b.shape[0], d)); bp[:, : b.shape[1]] = b
            u, _, vt = np.linalg.svd(ap.T @ bp)
            rotation = u @ vt
        return ProcrustesResult(residual_ratio=residual, singular_values=sv, left=left, right=right,
                                padded_dim=d, rotation=rotation)


# ---------------------------------------------------------------------- CLI on extracted features

def _split_rows(n: int, max_rows: Optional[int], heldout_frac: float, seed: int):
    if not 0.0 < heldout_frac < 1.0:
        raise ValueError("heldout_frac must be in (0, 1)")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, (int, np.integer))
                                  or max_rows < 2):
        raise ValueError(f"max_rows must be an integer >= 2, got {max_rows!r}")
    order = np.random.default_rng(seed).permutation(n)
    if max_rows is not None:
        order = order[:max_rows]
    n_eval = int(round(len(order) * heldout_frac))
    if n_eval < 2 or len(order) - n_eval < 2:
        raise ValueError(f"too few rows for a fit/eval split: {len(order)} rows, heldout_frac={heldout_frac}")
    return np.sort(order[n_eval:]), np.sort(order[:n_eval])


def run_pair(path_a: Path, path_b: Path, block: str, reg: float, max_rank: Optional[int],
             max_rows: Optional[int], heldout_frac: float, permutations: int, seed: int,
             standardize: bool) -> Dict[str, object]:
    """Load two extractor ``<task>.npz`` files, re-verify row alignment, then report in-sample GCCA,
    held-out rho, the row-permutation null, the shared width s (directions above the null) and the
    Procrustes residual on the fit rows."""
    sys.path.insert(0, str(HERE))
    from cross_model_manifold_alignment import load_features, verify_id_alignment

    a, b = load_features(path_a), load_features(path_b)
    verify_id_alignment(a, str(path_a), b, str(path_b))
    xa, xb = a[block], b[block]
    fit_idx, eval_idx = _split_rows(xa.shape[0], max_rows, heldout_frac, seed)
    op = GeneralizedCCAManifoldInterference(reg=reg, max_rank=max_rank, standardize=standardize)
    started = time.perf_counter()
    fit = op.fit([xa[fit_idx], xb[fit_idx]])
    heldout = op.heldout_canonical_correlations([xa[fit_idx], xb[fit_idx]], [xa[eval_idx], xb[eval_idx]])
    null = op.permutation_null([xa[fit_idx], xb[fit_idx]], [xa[eval_idx], xb[eval_idx]],
                               permutations=permutations, seed=seed)
    above = heldout > null["per_direction"]
    # s counts the leading run of directions above the null, not scattered exceedances.
    n_shared = int(np.argmin(above)) if not above.all() else int(above.size)
    proc = op.procrustes(xa[fit_idx], xb[fit_idx], dense_limit=0)
    return {
        "path_a": str(path_a), "path_b": str(path_b), "block": block,
        "n_fit": int(fit_idx.size), "n_eval": int(eval_idx.size),
        "reg": reg, "max_rank": max_rank, "standardize": standardize, "seed": seed,
        "in_sample": fit.summary(),
        "heldout_rho": heldout.tolist(),
        "null_rho_q95": null["per_direction"].tolist(),
        "null_rho_mean": null["mean"].tolist(),
        "permutations": permutations,
        "n_shared_above_null": n_shared,
        "procrustes_residual_ratio_fit_rows": proc.residual_ratio,
        "seconds": time.perf_counter() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path_a", type=Path)
    parser.add_argument("path_b", type=Path)
    parser.add_argument("--block", default="train_full")
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--max-rank", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--heldout-frac", type=float, default=0.5)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--transform-prefix", type=Path)
    args = parser.parse_args(argv)
    report = run_pair(args.path_a, args.path_b, args.block, args.reg, args.max_rank, args.max_rows,
                      args.heldout_frac, args.permutations, args.seed, not args.no_standardize)
    if args.transform_prefix:
        from cross_model_manifold_alignment import load_features, verify_id_alignment
        sys.path.insert(0, str((HERE / ".." / ".." / "python").resolve()))
        from gen_zero.causal.feature_space import load_source_space
        a, b = load_features(args.path_a), load_features(args.path_b)
        verify_id_alignment(a, str(args.path_a), b, str(args.path_b))
        space_a, space_b = load_source_space(args.path_a), load_source_space(args.path_b)
        fit_idx, eval_idx = _split_rows(len(a[args.block]), args.max_rows, args.heldout_frac, args.seed)
        fitted = GeneralizedCCAManifoldInterference(reg=args.reg, max_rank=args.max_rank,
            n_shared=128, standardize=not args.no_standardize).fit([a[args.block][fit_idx], b[args.block][fit_idx]])
        for i, space in enumerate((space_a, space_b)):
            fitted.save_transform(str(args.transform_prefix) + f".view{i}.npz", i,
                source_model=space["source_model"], layer=space["layer"], norm=space["norm"],
                source_data=[a[args.block], b[args.block]][i], fit_indices=fit_idx, eval_indices=eval_idx)
    text = json.dumps(report, indent=2)
    if args.out_json:
        args.out_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
