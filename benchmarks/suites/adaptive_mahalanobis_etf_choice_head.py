"""Adaptive Mahalanobis / generalized-ETF choice head
(docs/zero/31-multiscale-dense-resonance-etf-dual-process-plan.md S5).

Fixed equiangular-tight-frame (ETF) geometry forces the same pairwise angle between all K
class prototypes regardless of what the data says, which floors the entropy at a value set
by K rather than by evidence (S2.1/S5.1 of the plan doc). This head instead:

  1. fits a Ledoit-Wolf (2004) shrunk within-class covariance Sigma_LW exactly like
     `spec21_advanced_heads.LedoitWolfLDAHead` (same closed-form rho, same thin-SVD trick so a
     D x D matrix is never formed -- cheap even at D=8192), then whitens class prototypes
     into that metric so that squared Euclidean distance in whitened space equals the
     Mahalanobis distance in the original space;
  2. centers the whitened prototypes across classes (so they sum to zero) and rescales all of
     them by one shared scalar (the RMS class-to-center distance, not a per-class norm -- a
     per-class rescaling would rotate the Gram matrix's null direction off the all-ones vector
     and break the exact reconstruction in step 3) into a matrix C_raw (D x K), then computes
     its Gram matrix G_raw = C_raw^T C_raw;
  3. blends G_raw toward the hard-ETF Gram G_etf = K/(K-1) I - 1/(K-1) 11^T with weight
     lambda_etf, and reconstructs actual prototype vectors whose Gram matches the blend
     exactly (closed form via symmetric matrix square roots -- no gradient descent).

lambda_etf=0 reproduces the plain Ledoit-Wolf Mahalanobis (LW-LDA) prototypes to numerical
precision (tested to ~1e-8), whenever the metric rank is not truncated by `max_metric_rank`;
lambda_etf -> inf drives the Gram matrix to G_etf exactly in the limit. Whenever
`max_metric_rank` truncates the true rank, both the lambda=0 exactness and the metric itself
become an approximation that OVERSTATES precision on the discarded directions (see the
`max_metric_rank` comment in `fit`) -- this is a known, documented cost/accuracy trade, not a
free truncation.
lambda_etf is a hyperparameter the caller selects by OOF cross-validation -- this module does
not pick it, so whether the ETF term helps is answered by data, not assumed.

Every step here operates symmetrically on the K classes (the Ledoit-Wolf fit pools residuals
over the full row set regardless of label naming, G_etf is invariant under conjugation by any
permutation matrix, and the Gram-matching reconstruction is itself equivariant under
conjugation), so permuting which physical class is called "0" vs "1" ... permutes the output
probability vector by exactly the same permutation. See
`benchmarks/tests/test_adaptive_mahalanobis_etf_choice_head.py::test_permutation_equivariance`
for a numeric check.

Ordinal-scored tasks (helpsteer2, summeval_relevance, summeval_consistency) violate the ETF
assumption that all classes are pairwise equidistant (|1-2| < |1-5|); the constructor refuses
to build a head for those task names (S5.4) and points callers at
`OrdinalCumulativeHead` (python/gen_zero/model/choice_head.py) instead.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-12

# S5.4: labels with a natural order (1 < 2 < ... < 5) where ETF's equidistance assumption is
# false. Kept as an explicit, non-overridable gate: silently applying ETF here would misreport
# a metric-topology violation as a modeling choice.
ORDINAL_TASK_NAMES: frozenset = frozenset({
    "helpsteer2",
    "summeval_relevance",
    "summeval_consistency",
})


def _as_matrix(x, name: str) -> np.ndarray:
    if np.iscomplexobj(x):
        raise ValueError("complex inputs not supported")
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or Inf values")
    return a


def _as_labels(y, name: str) -> np.ndarray:
    a = np.asarray(y)
    if np.iscomplexobj(a):
        raise ValueError("complex inputs not supported")
    if a.dtype == bool:
        raise ValueError("bool labels not supported")
    if a.dtype.kind in "USO":
        raise ValueError("string or object labels not supported")
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if a.dtype.kind not in "iu":
        af = np.asarray(a, dtype=np.float64)
        if not np.all(np.isfinite(af)):
            raise ValueError(f"{name} contains NaN or Inf values")
        if np.any(af != np.round(af)):
            raise ValueError(f"{name} must hold integer class ids")
        a = af.astype(np.int64)
    else:
        a = a.astype(np.int64)
    if a.min() < 0:
        raise ValueError(f"{name} has negative class ids")
    return a


def _sym_matrix_func(M: np.ndarray, func_pos: Callable[[np.ndarray], np.ndarray],
                      tol: float = 1e-10) -> np.ndarray:
    """f(M) for a symmetric matrix via eigendecomposition, applying `func_pos` only to the
    numerically-positive eigenvalues and zeroing the rest (Moore-Penrose-style truncation).
    Used for both the symmetric square root (func_pos = sqrt) and its pseudo-inverse square
    root (func_pos = 1/sqrt); both matrices here (Gram matrices of centered prototypes) are
    PSD with a null space of dimension >= 1 (the all-ones direction), so a plain inverse does
    not exist and must not be used.
    """
    w, V = np.linalg.eigh((M + M.T) / 2.0)
    thresh = tol * max(float(w.max()), _EPS)
    out = np.zeros_like(w)
    mask = w > thresh
    if np.any(mask):
        out[mask] = func_pos(w[mask])
    return (V * out) @ V.T


def _softmax(logits: np.ndarray) -> np.ndarray:
    # Overflowed distances carry no usable evidence when every class is infinite.
    all_negative_inf = np.all(np.isneginf(logits), axis=1, keepdims=True)
    safe_logits = np.where(all_negative_inf, 0.0, logits)
    with np.errstate(over="ignore"):
        z = safe_logits - safe_logits.max(axis=1, keepdims=True)
        e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(probs)):
        raise ValueError("softmax produced non-finite probabilities")
    return probs


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE (S5.3): mean, over `n_bins` equal-width confidence bins, of
    |accuracy(bin) - mean_confidence(bin)|, weighted by bin occupancy. `probs` is (N, K)
    predicted probabilities, `y` the true class ids. This is the standard measurable proxy
    the plan doc substitutes for the informal "entropy collapse" / "overconfidence" claim.
    """
    if (not isinstance(n_bins, (int, np.integer))
            or isinstance(n_bins, (bool, np.bool_)) or n_bins < 1):
        raise ValueError("n_bins must be a positive integer")
    probs = _as_matrix(probs, "probs")
    y = _as_labels(y, "y")
    if probs.shape[0] != y.size:
        raise ValueError(f"probs has {probs.shape[0]} rows but y has {y.size}")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probs must have values in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("probs rows must sum to 1")
    if np.any(y >= probs.shape[1]):
        raise ValueError("y contains out-of-range class ids")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = y.size
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


class AdaptiveMahalanobisETFChoiceHead:
    """Ledoit-Wolf Mahalanobis prototypes with an optional, data-selected pull toward a
    generalized (soft) ETF geometry. See module docstring for the construction.
    """

    def __init__(self, task_name: Optional[str] = None, max_metric_rank: int = 256):
        if task_name is not None and task_name.strip().lower() in ORDINAL_TASK_NAMES:
            raise ValueError(
                f"task {task_name!r} has ordinal (naturally-ordered) labels: ETF geometry "
                "assumes all classes are pairwise equidistant, which is false for an ordinal "
                "scale (|1-2| < |1-5|). Use OrdinalCumulativeHead "
                "(python/gen_zero/model/choice_head.py) for this task instead."
            )
        if not (isinstance(max_metric_rank, (int, np.integer)) and max_metric_rank >= 1):
            raise ValueError(f"max_metric_rank must be a positive int, got {max_metric_rank!r}")
        self.task_name = task_name
        self.max_metric_rank = int(max_metric_rank)
        self.temperature_ = 1.0
        self._fitted = False

    # ------------------------------------------------------------------ whitening

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        """Linear map A with A^T A = Sigma_LW^{-1}, applied to rows of X (m, D) -> (m, D)."""
        T = X @ self._Vt_r.T                      # m x r
        X_span = T @ self._Vt_r                   # m x D
        X_perp = X - X_span
        return X_perp * self._inv_sqrt_tail + (T * self._inv_sqrt_e) @ self._Vt_r

    # ------------------------------------------------------------------ fit

    def fit(self, Z: np.ndarray, y: np.ndarray,
            candidate_reps: Optional[np.ndarray] = None,
            lambda_etf: float = 0.0) -> "AdaptiveMahalanobisETFChoiceHead":
        Z = _as_matrix(Z, "Z")
        y = _as_labels(y, "y")
        if Z.shape[0] != y.size:
            raise ValueError(f"Z has {Z.shape[0]} rows but y has {y.size}")
        n, d = Z.shape
        num_classes = int(y.max()) + 1
        if num_classes < 2:
            raise ValueError(f"need at least 2 classes, found {num_classes}")
        if n < num_classes:
            raise ValueError(
                f"N={n} samples but K={num_classes} classes: need N >= K (Fail-Closed)"
            )
        if not (isinstance(lambda_etf, (int, float, np.floating, np.integer))
                and not isinstance(lambda_etf, (bool, np.bool_)) and np.isfinite(lambda_etf)
                and lambda_etf >= 0.0):
            raise ValueError(f"lambda_etf must be a finite number >= 0, got {lambda_etf!r}")

        counts = np.bincount(y, minlength=num_classes)
        absent = np.flatnonzero(counts == 0)
        if absent.size:
            raise ValueError(f"classes {absent.tolist()} have no training rows")
        singleton = np.flatnonzero(counts < 2)
        if singleton.size:
            raise ValueError(
                f"classes {singleton.tolist()} have fewer than 2 training rows: a singleton "
                "class contributes an exactly-zero within-class residual (its \"mean\" is the "
                "single point itself), so its true within-class spread is unobserved and would "
                "be silently borrowed from the pooled estimate over the other classes "
                "(Fail-Closed instead of pooling silently)"
            )

        if candidate_reps is not None:
            candidate_reps = _as_matrix(candidate_reps, "candidate_reps")
            if candidate_reps.shape != (num_classes, d):
                raise ValueError(
                    f"candidate_reps must have shape ({num_classes}, {d}), "
                    f"got {candidate_reps.shape}"
                )

        class_means = np.stack([Z[y == k].mean(axis=0) for k in range(num_classes)])
        R = np.concatenate([Z[y == k] - class_means[k] for k in range(num_classes)], axis=0)

        # Ledoit-Wolf (2004) closed-form shrinkage via the thin SVD of the within-class
        # residuals -- identical derivation to spec21_advanced_heads.LedoitWolfLDAHead, so it
        # never forms a D x D matrix (O(min(n,d)^2 max(n,d)) instead of O(d^3)).
        _, sv, Vt = np.linalg.svd(R, full_matrices=False)
        lam = sv ** 2 / n
        trace = float(lam.sum())
        if not trace > _EPS:
            raise ValueError("within-class covariance is (numerically) zero: features carry no variance")
        nu = trace / d
        sum_sq = float((lam ** 2).sum())
        d2 = (sum_sq - d * nu ** 2) / d
        row_sq = np.einsum("ij,ij->i", R, R)
        b2_bar = (float((row_sq ** 2).sum()) - n * sum_sq) / (n ** 2 * d)
        rho = 1.0 if d2 <= 1e-15 * nu ** 2 else min(max(b2_bar, 0.0), d2) / d2

        # S5.2 point 1: cap the retained metric rank at max_metric_rank to bound cost on very
        # high-D features (e.g. D=8192). Directions beyond the cap are folded into the
        # isotropic tail and scored with variance rho*nu, but their true Sigma_LW eigenvalue is
        # (1-rho)*lam_i + rho*nu >= rho*nu (lam_i >= 0). The assumed variance is therefore <=
        # the true variance, i.e. the assumed PRECISION (1/variance) is >= the true precision:
        # truncation OVERSTATES confidence along the discarded directions, it does not
        # understate it. See test_truncated_metric_rank_overstates_precision.

        r = min(lam.size, self.max_metric_rank)
        metric_truncated = r < lam.size
        lam_r = lam[:r]
        Vt_r = Vt[:r]
        if rho * nu <= 0.0:
            raise ValueError("Ledoit-Wolf shrinkage collapsed to a singular covariance (rho*nu <= 0)")
        e = (1.0 - rho) * lam_r + rho * nu
        if np.any(e <= 0.0):
            raise ValueError("Ledoit-Wolf covariance has a non-positive eigenvalue on the retained metric rank")

        inv_sqrt_e = 1.0 / np.sqrt(e)
        inv_sqrt_tail = 1.0 / math.sqrt(rho * nu)

        # Whitened class prototypes (Mahalanobis metric folded in): squared Euclidean distance
        # between whitened vectors equals the Mahalanobis distance in the original space.
        proto_source = candidate_reps if candidate_reps is not None else class_means
        T = proto_source @ Vt_r.T
        mu_w = ((proto_source - T @ Vt_r) * inv_sqrt_tail
                + (T * inv_sqrt_e) @ Vt_r)                      # K x D
        proto_mean = mu_w.mean(axis=0)                          # centers across classes
        p = mu_w - proto_mean                                   # sum_k p_k = 0 exactly
        # A single GLOBAL scale factor (not a per-class norm) keeps `sum_k (p_k / s) = 0`
        # exactly, so G_raw's null space stays exactly span{1} -- the same null space as
        # G_etf. Normalizing each column to its OWN norm would break that (the null direction
        # would rotate to align with the vector of per-class norms instead), which silently
        # makes G_target generically full rank for any lambda_etf > 0 and breaks the exact
        # Gram-matching reconstruction below. `s` is a symmetric function (a mean) of the
        # per-class magnitudes, so it does not depend on class label order either.
        rho_k = np.linalg.norm(p, axis=1)
        if np.any(rho_k <= _EPS):
            raise ValueError(
                "a class prototype coincides with the across-class mean after whitening: "
                "no direction to regularize toward ETF for this class"
            )
        s = math.sqrt(float(np.mean(rho_k ** 2)))
        if not s > _EPS:
            raise ValueError("class prototypes coincide with the across-class mean after whitening")
        C_raw = (p / s).T                                       # D x K, unit average norm
        G_raw = C_raw.T @ C_raw                                 # K x K, null space = span{1}

        K = num_classes
        G_etf = (K / (K - 1)) * np.eye(K) - (1.0 / (K - 1)) * np.ones((K, K))
        # Both weights computed directly from lambda_etf (never as `1.0 - other_weight`): for
        # large lambda_etf, w_raw = 1/(1+lambda_etf) is tiny and `1.0 - w_etf` would cancel
        # catastrophically against a w_etf that already rounds to ~1.0 in double precision.
        denom = 1.0 + lambda_etf
        w_raw = 1.0 / denom
        w_etf = lambda_etf / denom
        G_target = w_raw * G_raw + w_etf * G_etf                # null space = span{1} for any lambda_etf

        G_raw_pinv_sqrt = _sym_matrix_func(G_raw, lambda ev: 1.0 / np.sqrt(ev))
        G_target_sqrt = _sym_matrix_func(G_target, np.sqrt)
        A = G_raw_pinv_sqrt @ G_target_sqrt
        U_new = C_raw @ A                                       # D x K

        achieved = U_new.T @ U_new
        if not np.allclose(achieved, G_target, atol=1e-6, rtol=1e-4):
            raise ValueError(
                "ETF Gram reconstruction failed to match the target within tolerance: the "
                f"{K} class prototypes do not span a {K - 1}-dimensional subspace after "
                "whitening (too few distinct directions for this K). Fail-Closed rather than "
                "silently returning a mismatched geometry."
            )

        proto_final_w = proto_mean[None, :] + s * U_new.T
        if not np.all(np.isfinite(proto_final_w)):
            raise ValueError("fitted prototypes contain non-finite values")

        # Commit only after every calculation and validation has succeeded.
        self._d = d
        self._Vt_r = Vt_r
        self._inv_sqrt_e = inv_sqrt_e
        self._inv_sqrt_tail = inv_sqrt_tail
        self._metric_truncated = metric_truncated
        self.rho_, self.nu_ = float(rho), float(nu)
        self._proto_final_w = proto_final_w
        self.num_classes_ = num_classes
        self.lambda_etf_ = float(lambda_etf)
        self.G_raw_ = G_raw
        self.G_etf_ = G_etf
        self.G_target_ = G_target
        self.temperature_ = 1.0
        self._fitted = True
        return self

    # ------------------------------------------------------------------ scoring

    def _squared_distances(self, Z: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit(...) before scoring")
        Z = _as_matrix(Z, "Z")
        if Z.shape[1] != self._d:
            raise ValueError(f"Z has {Z.shape[1]} features, head was fitted on {self._d}")
        dist2 = np.empty((Z.shape[0], self.num_classes_))
        # Bound the broadcast temporary to roughly 64 MiB, and at most 5000 rows.
        chunk_size = max(1, min(5000, (64 * 1024 ** 2) //
                                (8 * self.num_classes_ * self._d)))
        with np.errstate(over="ignore", invalid="ignore"):
            for start in range(0, Z.shape[0], chunk_size):
                Zw = self._whiten(Z[start:start + chunk_size])
                diff = Zw[:, None, :] - self._proto_final_w[None, :, :]
                np.square(diff, out=diff)
                chunk = np.sum(diff, axis=-1)
                # Finite inputs may overflow even during whitening. Treat an
                # unrepresentable distance as infinity, never as NaN evidence.
                chunk[~np.isfinite(chunk)] = np.inf
                dist2[start:start + chunk_size] = chunk
        return dist2

    def scores(self, Z: np.ndarray) -> np.ndarray:
        """logit_k = -d(z, mu_k) / T, in the same `X -> (N, K) argmax-able logits` convention
        as `spec21_advanced_heads.LedoitWolfLDAHead.scores` / `BBPAdaptiveProbe.scores`
        (`.scores(Q).argmax(axis=1)`), so this head can be dispatched the same way those are in
        `evaluate_spec21_scorecard` / `evaluate_dual_70b_72b_advanced_ensemble`. The `fit(...)`
        signature differs (those are `.fit(X, y, num_classes)` classmethods; this is an
        instance method with `lambda_etf`/`candidate_reps`), so wiring this in as another
        scorecard family is a separate change, not automatic from this method existing.
        """
        return -self._squared_distances(Z) / self.temperature_

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        dist2 = self._squared_distances(Z)
        logits = -dist2 / self.temperature_
        return _softmax(logits)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(Z), axis=1)

    def calibrate_temperature(self, Z_val: np.ndarray, y_val: np.ndarray) -> float:
        """Sets `self.temperature_` to the value minimizing NLL on (Z_val, y_val); returns it."""
        if not self._fitted:
            raise RuntimeError("call fit(...) before calibrate_temperature")
        Z_val = _as_matrix(Z_val, "Z_val")
        y_val = _as_labels(y_val, "y_val")
        if Z_val.shape[0] != y_val.size:
            raise ValueError(f"Z_val has {Z_val.shape[0]} rows but y_val has {y_val.size}")
        if y_val.max() >= self.num_classes_:
            raise ValueError(f"y_val has label {int(y_val.max())} but head has {self.num_classes_} classes")
        dist2 = self._squared_distances(Z_val)
        idx = np.arange(y_val.size)

        def nll(log_t: float) -> float:
            t = math.exp(log_t)
            logits = -dist2 / t
            z = logits - logits.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            return float(-logp[idx, y_val].mean())

        result = minimize_scalar(nll, bounds=(math.log(1e-4), math.log(1e4)), method="bounded")
        self.temperature_ = math.exp(float(result.x))
        return self.temperature_


__all__ = [
    "AdaptiveMahalanobisETFChoiceHead",
    "ORDINAL_TASK_NAMES",
    "expected_calibration_error",
]
