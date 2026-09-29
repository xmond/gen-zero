"""Instance-Adaptive Gating with Log-linear Opinion Pool.

Per-sample dynamic expert weighting for a multi-model ensemble, following the
WebGPT-style instance-adaptive routing design:

1. Reliability/applicability features per expert per sample: normalized
   predictive entropy, top1-top2 confidence margin, and Jensen-Shannon
   divergence against the cross-expert mean distribution (plus an optional
   external distance feature).
2. A 5-fold-averaged closed-form ridge regression (numpy only, no external ML
   dependency), fit on out-of-fold expert predictions, maps those features to
   the expert's expected cross-entropy risk \\ell_m(x) = -log p_m(y | x).
3. Predicted risk converts into a softmax gate over experts, shrunk toward a
   global prior: w_m(x) = (1-eps) * softmax_m(log w_m0 - risk_m(x)/tau) + eps * w_m0.
4. Expert distributions fuse either by weighted arithmetic mean or by a
   log-linear (geometric mean) opinion pool, which sharpens the fused
   distribution when experts agree and lets any single confident veto
   dominate when they disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS_PROB = 1e-12
_DEFAULT_DELTA = 1e-6

__all__ = [
    "InstanceAdaptiveRouter",
    "GatingDecision",
    "LogLinearOpinionPool",
    "normalized_entropy",
    "top1_top2_margin",
    "jensen_shannon_divergence",
    "extract_reliability_features",
]


@dataclass
class GatingDecision:
    """Container for instance-adaptive gating outputs."""
    weights: np.ndarray                         # (N, M)
    predicted_risks: np.ndarray                 # (N, M)
    fused_probs: Optional[np.ndarray] = None    # (N, K)
    expert_names: Optional[List[str]] = None


class LogLinearOpinionPool:
    """Helper for log-linear (geometric mean) opinion pool fusion."""
    @staticmethod
    def fuse(
        expert_probs_list: Sequence[np.ndarray],
        weights: np.ndarray,
        delta: float = _DEFAULT_DELTA,
    ) -> np.ndarray:
        router = InstanceAdaptiveRouter(delta=delta)
        return router.fuse_predictions(
            expert_probs_list,
            pool_type="log_linear",
            dynamic_weights=weights,
        )


def _as_prob_array(probs) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a (N, K) probability array, got shape {arr.shape}")
    return arr


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, normalized to [0, 1] by log(K)."""
    probs = _as_prob_array(probs)
    k = probs.shape[1]
    if k <= 1:
        return np.zeros(probs.shape[0], dtype=np.float64)
    clipped = np.clip(probs, _EPS_PROB, 1.0)
    ent = -np.sum(np.where(probs > 0.0, probs * np.log(clipped), 0.0), axis=1)
    return np.clip(ent / np.log(k), 0.0, 1.0)


def top1_top2_margin(probs: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability per row."""
    probs = _as_prob_array(probs)
    sorted_p = np.sort(probs, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2] if probs.shape[1] >= 2 else np.zeros_like(top1)
    return top1 - top2


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence per row between two distributions, base-2
    normalized so the result falls in [0, 1]."""
    p = _as_prob_array(p)
    q = _as_prob_array(q)
    p_sum = p.sum(axis=1, keepdims=True)
    q_sum = q.sum(axis=1, keepdims=True)
    p = p / np.where(p_sum > 0.0, p_sum, 1.0)
    q = q / np.where(q_sum > 0.0, q_sum, 1.0)
    p = np.clip(p, _EPS_PROB, 1.0)
    q = np.clip(q, _EPS_PROB, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return np.clip(js / np.log(2.0), 0.0, 1.0)


def extract_reliability_features(
    probs: np.ndarray,
    mean_probs: np.ndarray,
    distance: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-sample reliability/applicability features for one expert.

    Columns: [normalized entropy, top1-top2 margin, JS divergence vs the
    cross-expert mean distribution], plus an optional external distance
    column when ``distance`` is given.
    """
    probs = _as_prob_array(probs)
    mean_probs = _as_prob_array(mean_probs)
    feats = [
        normalized_entropy(probs),
        top1_top2_margin(probs),
        jensen_shannon_divergence(probs, mean_probs),
    ]
    if distance is not None:
        dist = np.asarray(distance, dtype=np.float64).reshape(-1)
        if dist.shape[0] != probs.shape[0]:
            raise ValueError("distance must have one value per sample")
        feats.append(dist)
    return np.stack(feats, axis=1)


class InstanceAdaptiveRouter:
    """Sample-level dynamic gating and opinion-pool fusion for an ensemble of
    expert probability distributions."""

    def __init__(
        self,
        ridge_lambda: float = 1.0,
        n_folds: int = 5,
        random_state: int = 0,
        delta: float = _DEFAULT_DELTA,
    ):
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if delta <= 0:
            raise ValueError("delta must be > 0")
        self.ridge_lambda = float(ridge_lambda)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.delta = float(delta)

        self._risk_models: List[Tuple[np.ndarray, float]] = []
        self._expert_names: List[str] = []
        self._global_weights: Optional[np.ndarray] = None
        self._tau = 1.0
        self._epsilon = 0.2
        self._n_features: Optional[int] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def expert_names(self) -> List[str]:
        return list(self._expert_names)

    def fit(
        self,
        oof_probs_list: Sequence[np.ndarray],
        y_true,
        global_weights: Optional[Sequence[float]] = None,
        tau: float = 1.0,
        epsilon: float = 0.2,
        expert_names: Optional[Sequence[str]] = None,
        distances_list: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> "InstanceAdaptiveRouter":
        """Fit one 5-fold-averaged Ridge risk model per expert from
        out-of-fold predictions ``oof_probs_list`` (each shape (N, K)) and the
        true labels ``y_true`` (shape (N,), values in [0, K))."""
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be a finite positive number")
        if not np.isfinite(epsilon) or not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        num_experts = len(oof_probs_list)
        if num_experts == 0:
            raise ValueError("oof_probs_list must contain at least one expert")

        probs_arrs = [_as_prob_array(p) for p in oof_probs_list]
        n, k = probs_arrs[0].shape
        for arr in probs_arrs:
            if arr.shape != (n, k):
                raise ValueError("all experts must share the same (N, K) shape")

        y = np.asarray(y_true).reshape(-1).astype(np.int64)
        if y.shape[0] != n:
            raise ValueError("y_true length must match the number of samples")
        if n > 0 and (np.any(y < 0) or np.any(y >= k)):
            raise ValueError("y_true entries must be in [0, K)")

        if expert_names is None:
            expert_names = [str(i) for i in range(num_experts)]
        elif len(expert_names) != num_experts:
            raise ValueError("expert_names length must match the number of experts")

        if global_weights is None:
            w0 = np.full(num_experts, 1.0 / num_experts, dtype=np.float64)
        else:
            w0 = np.asarray(global_weights, dtype=np.float64)
            if w0.shape != (num_experts,):
                raise ValueError("global_weights length must match the number of experts")
            if not np.all(np.isfinite(w0)) or np.any(w0 <= 0.0):
                raise ValueError("global_weights must be finite and strictly positive")
            w0 = w0 / w0.sum()

        mean_probs = np.mean(np.stack(probs_arrs, axis=0), axis=0)

        risk_models = []
        for m in range(num_experts):
            dist_m = distances_list[m] if distances_list is not None else None
            feats = extract_reliability_features(probs_arrs[m], mean_probs, dist_m)
            clipped = np.clip(probs_arrs[m][np.arange(n), y], _EPS_PROB, 1.0)
            loss = -np.log(clipped)
            coef, intercept = self._fit_ridge_kfold(feats, loss)
            risk_models.append((coef, intercept))

        self._risk_models = risk_models
        self._expert_names = list(expert_names)
        self._global_weights = w0
        self._tau = float(tau)
        self._epsilon = float(epsilon)
        self._n_features = risk_models[0][0].shape[0]
        self._fitted = True
        return self

    @staticmethod
    def _ridge_closed_form(features: np.ndarray, target: np.ndarray, ridge_lambda: float) -> Tuple[np.ndarray, float]:
        """Closed-form ridge regression with an unregularized intercept:
        solve (X_d^T X_d + diag(lambda,...,lambda,0)) beta = X_d^T y, where
        X_d appends a bias column of ones."""
        n, f = features.shape
        design = np.concatenate([features, np.ones((n, 1), dtype=np.float64)], axis=1)
        reg = np.eye(f + 1, dtype=np.float64) * ridge_lambda
        reg[-1, -1] = 0.0
        gram = design.T @ design + reg
        rhs = design.T @ target
        beta = np.linalg.solve(gram, rhs)
        return beta[:-1], float(beta[-1])

    def _fit_ridge_kfold(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Average ridge coefficients across `n_folds` folds of the OOF data,
        so the tiny risk model does not overfit the handful of OOF samples."""
        n = features.shape[0]
        n_folds = min(self.n_folds, n) if n >= 2 else 1
        if n_folds < 2:
            return self._ridge_closed_form(features, target, self.ridge_lambda)

        rng = np.random.default_rng(self.random_state)
        shuffled = rng.permutation(n)
        folds = np.array_split(shuffled, n_folds)
        coefs = []
        intercepts = []
        for i in range(n_folds):
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            coef, intercept = self._ridge_closed_form(features[train_idx], target[train_idx], self.ridge_lambda)
            coefs.append(coef)
            intercepts.append(intercept)
        return np.mean(coefs, axis=0), float(np.mean(intercepts))

    def predict_weights(
        self,
        probs_list: Sequence[np.ndarray],
        distances_list: Optional[Sequence[Optional[np.ndarray]]] = None,
        expert_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Predict dynamic gate weights, shape (N, M), for each sample /
        expert pair from the experts' own current-sample distributions.

        ``expert_names``, when given, keys each row of ``probs_list`` to its
        fitted risk model by name instead of by position -- required because a
        production caller's active expert subset/order can vary call to call
        (e.g. dynamic-K MoE routing). Every name must have been seen at
        ``fit()`` time; an unknown name fails closed rather than silently
        guessing a risk model by position.
        """
        self._check_fitted()
        probs_arrs = [_as_prob_array(p) for p in probs_list]
        num_experts = len(probs_arrs)

        if expert_names is not None:
            if len(expert_names) != num_experts:
                raise ValueError("expert_names length must match probs_list length")
            try:
                model_idx = [self._expert_names.index(name) for name in expert_names]
            except ValueError as exc:
                raise ValueError(
                    f"unknown expert name(s) in {list(expert_names)!r}; "
                    f"router was fitted on {self._expert_names!r}"
                ) from exc
            active_w0 = self._global_weights[model_idx]
            active_w0 = active_w0 / active_w0.sum()
            active_models = [self._risk_models[i] for i in model_idx]
        else:
            if num_experts != len(self._expert_names):
                raise ValueError(
                    f"expected {len(self._expert_names)} experts, got {num_experts}; "
                    "pass expert_names to select a named subset"
                )
            active_w0 = self._global_weights
            active_models = self._risk_models

        n, k = probs_arrs[0].shape
        for arr in probs_arrs:
            if arr.shape != (n, k):
                raise ValueError("all experts must share the same (N, K) shape")

        mean_probs = np.mean(np.stack(probs_arrs, axis=0), axis=0)

        risk = np.zeros((n, num_experts), dtype=np.float64)
        for m in range(num_experts):
            dist_m = distances_list[m] if distances_list is not None else None
            feats = extract_reliability_features(probs_arrs[m], mean_probs, dist_m)
            coef, intercept = active_models[m]
            if feats.shape[1] != coef.shape[0]:
                name = expert_names[m] if expert_names is not None else self._expert_names[m]
                raise ValueError(
                    f"feature dimension mismatch for expert '{name}': "
                    f"model expects {coef.shape[0]}, got {feats.shape[1]} "
                    "(did you pass distances at fit time but not at predict time, or vice versa?)"
                )
            risk[:, m] = feats @ coef + intercept

        log_w0 = np.log(active_w0)
        logits = log_w0[None, :] - risk / self._tau
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        tilde_w = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        w = (1.0 - self._epsilon) * tilde_w + self._epsilon * active_w0[None, :]
        w = w / np.sum(w, axis=1, keepdims=True)
        return w

    def fuse_predictions(
        self,
        probs_list: Sequence[np.ndarray],
        pool_type: str = "log_linear",
        dynamic_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fuse M experts' (N, K) distributions into one (N, K) distribution.

        ``pool_type='arithmetic'`` is the weighted arithmetic mean
        (traditional consensus). ``pool_type='log_linear'`` is the weighted
        geometric mean (log-linear opinion pool): it sharpens the fused
        distribution when experts agree and lets a confident veto (a near-zero
        probability from any single expert) dominate when they disagree.
        """
        probs_arrs = [_as_prob_array(p) for p in probs_list]
        num_experts = len(probs_arrs)
        if num_experts == 0:
            raise ValueError("probs_list must contain at least one expert")
        n, k = probs_arrs[0].shape
        probs_arrs = [_as_prob_array(p) for p in probs_list]
        for arr in probs_arrs:
            if arr.shape != (n, k):
                raise ValueError("all experts must share the same (N, K) shape")
            if not np.all(np.isfinite(arr)) or np.any(arr < -1e-9):
                raise ValueError("expert probabilities must be finite and non-negative")

        if dynamic_weights is None:
            dynamic_weights = self.predict_weights(probs_list)
        w = np.asarray(dynamic_weights, dtype=np.float64)
        if w.shape != (n, num_experts):
            raise ValueError(f"dynamic_weights must have shape ({n}, {num_experts}), got {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < -1e-9):
            raise ValueError("dynamic_weights must be finite and non-negative")

        row_w_sums = w.sum(axis=1, keepdims=True)
        if np.any(row_w_sums <= 0.0):
            raise ValueError("dynamic_weights rows must not be all-zero")
        w = np.clip(w, 0.0, None)
        w = w / w.sum(axis=1, keepdims=True)

        stacked = np.stack(probs_arrs, axis=0)  # (M, N, K)
        stacked = np.clip(stacked, 0.0, 1.0)
        # A class only survives fusion if at least one expert assigns it mass;
        # this preserves the hard support-set semantics production callers rely on
        # (an expert reporting exactly 0.0 for a candidate declares it infeasible).
        support = np.any(stacked > 0.0, axis=0)  # (N, K)

        if pool_type == "arithmetic":
            fused = np.einsum("nm,mnk->nk", w, stacked)
        elif pool_type == "log_linear":
            log_p = np.log(stacked + self.delta)  # (M, N, K)
            weighted_log = np.einsum("nm,mnk->nk", w, log_p)
            weighted_log = weighted_log - np.max(weighted_log, axis=1, keepdims=True)
            fused = np.exp(weighted_log)
            fused = np.where(support, fused, 0.0)
        else:
            raise ValueError(f"unknown pool_type: {pool_type!r}")

        row_sums = fused.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] <= 0.0
        safe_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        fused = fused / safe_sums
        fused[zero_rows] = 0.0
        fused = np.clip(fused, 0.0, 1.0)
        return fused

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("InstanceAdaptiveRouter is not fitted; call fit() first")

    def save(self, path: str) -> None:
        """Serialize the fitted risk models and gate hyperparameters to an
        .npz file so a production caller can load a calibrated router without
        refitting on every process start."""
        self._check_fitted()
        coefs = np.stack([c for c, _ in self._risk_models], axis=0)
        intercepts = np.asarray([b for _, b in self._risk_models], dtype=np.float64)
        np.savez(
            path,
            coefs=coefs,
            intercepts=intercepts,
            expert_names=np.asarray(self._expert_names),
            global_weights=self._global_weights,
            tau=np.asarray(self._tau),
            epsilon=np.asarray(self._epsilon),
            ridge_lambda=np.asarray(self.ridge_lambda),
            n_folds=np.asarray(self.n_folds),
            delta=np.asarray(self.delta),
        )

    @classmethod
    def load(cls, path: str) -> "InstanceAdaptiveRouter":
        with np.load(path, allow_pickle=False) as data:
            tau = float(data["tau"])
            epsilon = float(data["epsilon"])
            if not np.isfinite(tau) or tau <= 0:
                raise ValueError(f"invalid tau in artifact: {tau}")
            if not np.isfinite(epsilon) or not (0.0 <= epsilon <= 1.0):
                raise ValueError(f"invalid epsilon in artifact: {epsilon}")

            coefs = np.asarray(data["coefs"], dtype=np.float64)
            intercepts = np.asarray(data["intercepts"], dtype=np.float64)
            if not np.all(np.isfinite(coefs)) or not np.all(np.isfinite(intercepts)):
                raise ValueError("risk model coefficients in artifact contain non-finite values")

            router = cls(
                ridge_lambda=float(data["ridge_lambda"]),
                n_folds=int(data["n_folds"]),
                delta=float(data["delta"]),
            )
            router._risk_models = [
                (coefs[m], float(intercepts[m]))
                for m in range(coefs.shape[0])
            ]
            router._expert_names = [str(x) for x in data["expert_names"]]
            router._global_weights = np.asarray(data["global_weights"], dtype=np.float64)
            if not np.all(np.isfinite(router._global_weights)) or np.any(router._global_weights < 0):
                raise ValueError("global_weights in artifact contains invalid values")

            router._tau = tau
            router._epsilon = epsilon
            router._n_features = router._risk_models[0][0].shape[0]
            router._fitted = True
        return router
