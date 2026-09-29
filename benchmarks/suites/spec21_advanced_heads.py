"""Spec 21 phase 1: statistical-physics / geometry operator library
(docs/zero/21-mathematical-foundations-and-advanced-architectures-spec.md S2, S4.3, S5.4, S6 A/A'/C/E).

Everything here is fitted offline (NumPy + scikit-learn for the logistic step of the probe) and
folds to a plain NumPy operator: `X @ W_fold.T + b_fold`, then O(K) work for conformal sets.

  Design C   compute_class_priors, check_prior_collapse_gate, logit_adjust, fold_logit_adjustment
  Design A   gavish_donoho_omega, estimate_gd_rank, BBPAdaptiveProbe
  Design A'  LedoitWolfLDAHead
  Design E   SplitConformalPredictor

Scope limits, stated so nobody reads more into this file than it does:
  * Gavish-Donoho is an asymptotic i.i.d.-noise threshold. Real LLM features have correlated noise;
    the retained rank is a principled default, not a proof that the rest is noise.
  * The GD-2017 optimal singular-value *shrinkage* (S2.3 theorem 2.5) is not implemented; the probe
    projects onto the retained right singular vectors without whitening (the S7.2 "unwhitened" variant).
  * Split conformal gives MARGINAL coverage under exchangeability of calibration and test rows.
    It is not per-class coverage (no Mondrian variant here) and not a guarantee under distribution shift.
  * Latency figures are measured by the `__main__` micro-benchmark, on whatever box runs it.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

_EPS = 1e-12
COLLAPSE_FRAC = 0.95            # same threshold as the legacy collapse_stats gate
MIN_CLASS_RECALL = 0.10         # K >= 3: a head that (almost) never predicts some class is not admissible
# Inner-CV grid for the BBP probe's L2 strength. It reaches 1e-5 because a chosen C sitting on the old
# lower end (1e-3) cannot show whether stronger regularisation would fit better. Reported for PubMedQA,
# not re-measured on real features in this change.
BBP_C_GRID: Tuple[float, ...] = tuple(float(c) for c in np.logspace(-5, 3, 9))


# ----------------------------------------------------------------------------- validation helpers

def _as_labels(y, name: str, min_value: int = 0) -> np.ndarray:
    a = np.asarray(y)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if a.dtype.kind not in "iu":
        if a.dtype.kind != "f" or not np.all(np.isfinite(a)) or np.any(a != np.round(a)):
            raise ValueError(f"{name} must hold integer class ids")
    a = a.astype(np.int64)
    if a.min() < min_value:
        raise ValueError(f"{name} has values < {min_value}")
    return a


def _as_matrix(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _check_priors(priors, k: Optional[int] = None) -> np.ndarray:
    p = np.asarray(priors, dtype=np.float64)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("priors must be a 1-D finite non-negative vector")
    if abs(p.sum() - 1.0) > 1e-6:
        raise ValueError(f"priors must sum to 1, got {p.sum()}")
    if k is not None and p.size != k:
        raise ValueError(f"priors has {p.size} entries, expected {k}")
    return p


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- Design C

def compute_class_priors(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Empirical class frequencies pi_c of the TRAINING labels; absent classes get exactly 0."""
    y = _as_labels(y, "y")
    if not isinstance(num_classes, (int, np.integer)) or num_classes < 1:
        raise ValueError(f"num_classes must be a positive int, got {num_classes!r}")
    if y.max() >= num_classes:
        raise ValueError(f"label {int(y.max())} out of range for num_classes={num_classes}")
    return np.bincount(y, minlength=num_classes).astype(np.float64) / y.size


def check_prior_collapse_gate(y_true: np.ndarray, y_pred: np.ndarray,
                              train_priors: Optional[np.ndarray] = None,
                              min_class_recall: float = MIN_CLASS_RECALL) -> Dict[str, object]:
    """Spec 21 S1.2 / S6 C gate. Three independent failure modes:

      collapsed    one class takes > 95% of the predictions (legacy gate);
      below_prior  accuracy < the majority-class prior, i.e. worse than the constant predictor.
                   This is the check the legacy gate lacks: helpsteer2 (32.1 vs 40.3) has diverse
                   predictions and still loses to "always say the majority".
      class_recall_below_min  K >= 3 and some class present in `y_true` has recall < `min_class_recall`.
                   A head can beat the prior and avoid collapse while never predicting a minority class;
                   that zero-recall head must not be admitted. K is len(train_priors) when given, else the
                   number of distinct labels in `y_true`. Binary tasks are exempt (the other two checks
                   already cover a binary head that ignores one class).

    `train_priors` is the TRAIN class distribution (what the spec compares against). Without it the
    prior is taken from `y_true` itself and `prior_source` says so; that is a weaker, eval-set
    baseline, so pass the train priors whenever they exist. `y_pred == -1` means abstain and counts
    as wrong. `passed` requires no failure; `reasons` names every failure that fired.
    """
    yt = _as_labels(y_true, "y_true")
    yp = _as_labels(y_pred, "y_pred", min_value=-1)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} differ in length")
    if not (isinstance(min_class_recall, (int, float, np.floating)) and math.isfinite(min_class_recall)
            and 0.0 <= min_class_recall <= 1.0):
        raise ValueError(f"min_class_recall must be in [0, 1], got {min_class_recall!r}")
    n = yt.size
    if train_priors is None:
        majority = float(np.bincount(yt).max()) / n
        source = "eval_labels"
    else:
        tp = _check_priors(train_priors)
        majority = float(tp.max())
        source = "train_priors"
    classes = np.unique(yt)
    num_classes = int(tp.size) if train_priors is not None else int(classes.size)
    correct = yp == yt
    accuracy = float(correct.mean())
    per_class = {int(c): float(correct[yt == c].mean()) for c in classes}
    recalls = list(per_class.values())
    min_recall = float(min(recalls))
    recall_low = num_classes >= 3 and min_recall < min_class_recall
    predicted = yp[yp >= 0]
    max_frac = float(np.bincount(predicted).max()) / n if predicted.size else 0.0
    collapsed = max_frac > COLLAPSE_FRAC
    below = accuracy < majority
    reasons = [name for name, hit in (("collapsed", collapsed), ("below_prior", below),
                                      ("class_recall_below_min", recall_low)) if hit]
    return {"n": int(n), "accuracy": accuracy, "balanced_accuracy": float(np.mean(recalls)),
            "majority_prior": majority, "prior_source": source,
            "max_pred_class_frac": max_frac, "collapsed": bool(collapsed),
            "below_prior": bool(below), "num_classes": num_classes,
            "per_class_recall": per_class, "min_class_recall": min_recall,
            "min_class_recall_threshold": float(min_class_recall),
            "class_recall_below_min": bool(recall_low), "reasons": reasons, "passed": not reasons}


def logit_adjust(logits: np.ndarray, priors: np.ndarray, tau: float = 1.0,
                 eps: float = _EPS) -> np.ndarray:
    """Post-hoc logit adjustment (Menon et al., ICLR 2021): logits - tau * log(pi + eps).
    tau = 0 returns an unchanged copy. Returns a new array."""
    z = _as_matrix(logits, "logits")
    p = _check_priors(priors, z.shape[1])
    if not (isinstance(tau, (int, float, np.floating)) and math.isfinite(tau) and tau >= 0):
        raise ValueError(f"tau must be a finite non-negative float, got {tau!r}")
    return z - float(tau) * np.log(p + eps)


def fold_logit_adjustment(W: np.ndarray, b: np.ndarray, priors: np.ndarray, tau: float = 1.0,
                          eps: float = _EPS) -> Tuple[np.ndarray, np.ndarray]:
    """Fold the adjustment into a linear head: X @ W.T + b' == logit_adjust(X @ W.T + b).
    Only the bias changes (zero inference cost). Inputs are not mutated."""
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if W.ndim != 2 or b.shape != (W.shape[0],):
        raise ValueError(f"W {W.shape} and b {b.shape} are not a (K,D)/(K,) head")
    adj = logit_adjust(b[None, :], priors, tau, eps)[0]
    return W.copy(), adj


# ----------------------------------------------------------------------------- Design A: Gavish-Donoho

@lru_cache(maxsize=256)
def _mp_median(beta: float) -> float:
    """Median mu_beta of the Marchenko-Pastur law (sigma = 1, ratio beta in (0, 1])."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    a, b = (1.0 - math.sqrt(beta)) ** 2, (1.0 + math.sqrt(beta)) ** 2

    def dens(x: float) -> float:
        return math.sqrt(max((b - x) * (x - a), 0.0)) / (2.0 * math.pi * beta * x)

    def cdf(x: float) -> float:
        return quad(dens, a, x, epsabs=1e-13, epsrel=1e-13, limit=400)[0]

    return float(brentq(lambda x: cdf(x) - 0.5, a + 1e-12, b, xtol=1e-13))


def gavish_donoho_omega(beta: float) -> float:
    """omega(beta) = lambda*(beta) / sqrt(mu_beta): optimal hard-threshold coefficient for UNKNOWN
    noise level (Gavish & Donoho 2014, S3.2), so that tau* = omega(beta) * median(singular values).
    beta = min(n, D) / max(n, D) in (0, 1]. omega(1) = 2.858."""
    if not (isinstance(beta, (int, float, np.floating)) and math.isfinite(beta) and 0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta!r}")
    beta = float(beta)
    lam_star = math.sqrt(2.0 * (beta + 1.0) + 8.0 * beta / ((beta + 1.0) + math.sqrt(beta ** 2 + 14.0 * beta + 1.0)))
    return lam_star / math.sqrt(_mp_median(beta))


def _prepare(X: np.ndarray, standardize: bool, center: bool):
    """Centre (and optionally unit-variance) the columns; constant columns keep scale 1."""
    X = _as_matrix(X, "X")
    n, d = X.shape
    if n < 3 or d < 2:
        raise ValueError(f"X must be at least 3 x 2, got {X.shape}")
    mean = X.mean(axis=0) if center else np.zeros(d)
    scale = np.ones(d)
    if standardize:
        sd = X.std(axis=0)
        scale = np.where(sd > _EPS, sd, 1.0)
    return (X - mean) / scale, mean, scale


def _gd_from_singular_values(s: np.ndarray, n: int, d: int, center: bool) -> Dict[str, object]:
    n_eff = n - 1 if center else n          # centring removes one degree of freedom (one zero singular value)
    m = min(n_eff, d)
    s = s[:m]
    beta = m / max(n_eff, d)
    omega = gavish_donoho_omega(beta)
    y_med = float(np.median(s))
    tau = omega * y_med
    rank = int(np.sum(s > tau))
    return {"rank": rank, "tau": float(tau), "omega": float(omega), "y_med": y_med, "beta": float(beta),
            "n_above_tau": rank, "n_effective": int(n_eff), "n_singular_values": int(m)}


def estimate_gd_rank(X: np.ndarray, standardize: bool = True, center: bool = True) -> Dict[str, object]:
    """Signal rank r* = #{s_i > tau*}, tau* = omega(beta) * y_med, no grid and no label use."""
    Z, _, _ = _prepare(X, standardize, center)
    s = np.linalg.svd(Z, compute_uv=False)
    return _gd_from_singular_values(s, Z.shape[0], Z.shape[1], center)


def _check_fit_inputs(X, y, num_classes) -> Tuple[np.ndarray, np.ndarray]:
    X = _as_matrix(X, "X")
    y = _as_labels(y, "y")
    if y.size != X.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size}")
    if not isinstance(num_classes, (int, np.integer)) or num_classes < 2:
        raise ValueError(f"num_classes must be an int >= 2, got {num_classes!r}")
    if y.max() >= num_classes:
        raise ValueError(f"label {int(y.max())} out of range for num_classes={num_classes}")
    absent = np.flatnonzero(np.bincount(y, minlength=num_classes) == 0)
    if absent.size:
        raise ValueError(f"classes {absent.tolist()} have no training rows")
    return X, y


def resolve_class_weight(class_weight, num_classes: int):
    """None | 'balanced' | per-class weights (dict {class: w} or a length-K sequence) -> the form
    LogisticRegression takes. Weights must be finite and > 0; anything else raises."""
    if class_weight is None or (isinstance(class_weight, str) and class_weight == "balanced"):
        return class_weight
    if isinstance(class_weight, str):
        raise ValueError(f"class_weight must be None, 'balanced' or per-class weights, got {class_weight!r}")
    if isinstance(class_weight, dict):
        items = [(int(k), float(v)) for k, v in class_weight.items()]
    else:
        w = np.asarray(class_weight, dtype=np.float64)
        if w.ndim != 1 or w.size != num_classes:
            raise ValueError(f"class_weight needs {num_classes} entries, got shape {w.shape}")
        items = list(enumerate(w.tolist()))
    if any(not 0 <= k < num_classes for k, _ in items):
        raise ValueError(f"class_weight has a class outside 0..{num_classes - 1}")
    if any(not (math.isfinite(v) and v > 0) for _, v in items):
        raise ValueError("class_weight values must be finite and > 0")
    return dict(items)


def _select_c_by_cv(F: np.ndarray, y: np.ndarray, num_classes: int, folds: int, seed: int,
                    grid: Tuple[float, ...] = BBP_C_GRID, class_weight=None) -> float:
    """Inner-CV choice of the L2 strength by held-out log-loss (ties -> smaller C, more regularised).
    With a class_weight the held-out log-loss is weighted the same way, so C is chosen for the objective
    that is actually fitted. Written out instead of LogisticRegressionCV so it behaves the same across
    sklearn versions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_sample_weight
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(F, y))
    labels = np.arange(num_classes)
    best_c, best_loss = float(grid[0]), math.inf
    for c in grid:
        loss = 0.0
        for tr, va in splits:
            m = LogisticRegression(C=float(c), max_iter=3000, class_weight=class_weight).fit(F[tr], y[tr])
            proba = np.zeros((va.size, num_classes))
            proba[:, m.classes_] = m.predict_proba(F[va])
            sw = None if class_weight is None else compute_sample_weight(class_weight, y[va])
            loss += log_loss(y[va], np.clip(proba, 1e-15, 1.0), labels=labels, sample_weight=sw) * va.size
        if loss < best_loss - 1e-12:
            best_c, best_loss = float(c), loss
    return best_c


class BBPAdaptiveProbe:
    """Design A: GD-truncated PCA + L2 multinomial logistic regression, folded to one GEMV.

    Fit: standardize -> SVD -> r* by Gavish-Donoho -> project on V_{r*} (no whitening) -> logistic
    regression (C by inner CV unless `C` is given). The rank needs no grid search.
    Fold: scores = X @ W_fold.T + b_fold with W_fold = coef V_r^T / scale, K x D.
    Binary tasks are stored as two logit rows [-m/2, +m/2] so softmax equals the sigmoid.
    `class_weight` (None | 'balanced' | per-class weights) reweights the logistic loss, in the inner CV
    and in the final fit. 'balanced' already pushes the head toward uniform priors; stacking logit
    adjustment (tau > 0) on top corrects for the prior twice.
    """

    def __init__(self) -> None:
        raise TypeError("use BBPAdaptiveProbe.fit(...)")

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, num_classes: int, C: Optional[float] = None,
            standardize: bool = True, cv_folds: int = 4, seed: int = 0,
            class_weight=None) -> "BBPAdaptiveProbe":
        from sklearn.linear_model import LogisticRegression
        X, y = _check_fit_inputs(X, y, num_classes)
        cw = resolve_class_weight(class_weight, num_classes)
        Z, mean, scale = _prepare(X, standardize, center=True)
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        info = _gd_from_singular_values(s, Z.shape[0], Z.shape[1], center=True)
        r = info["rank"]
        if r == 0:
            raise ValueError("no singular value exceeds the Gavish-Donoho threshold: the features carry "
                             "no signal above the estimated noise floor")
        comps = Vt[:r].copy()
        F = Z @ comps.T
        if C is None:
            folds = min(cv_folds, int(np.bincount(y).min()))
            if folds < 2:
                raise ValueError("a class has < 2 rows: pass an explicit C instead of inner CV")
            chosen_c = _select_c_by_cv(F, y, num_classes, folds, seed, class_weight=cw)
        else:
            chosen_c = float(C)
        clf = LogisticRegression(C=chosen_c, max_iter=3000, class_weight=cw).fit(F, y)
        coef, icpt = clf.coef_, clf.intercept_
        self = object.__new__(cls)
        self.coef_binary_ = self.intercept_binary_ = None
        if num_classes == 2:
            self.coef_binary_, self.intercept_binary_ = coef.copy(), icpt.copy()
            coef, icpt = np.vstack([-coef[0], coef[0]]) / 2.0, np.array([-icpt[0], icpt[0]]) / 2.0
        self.num_classes = int(num_classes)
        self.mean_, self.scale_, self.components_ = mean, scale, comps
        self.coef_, self.intercept_ = coef, icpt
        self.rank, self.tau, self.omega, self.beta = r, info["tau"], info["omega"], info["beta"]
        self.C_, self.class_weight_ = chosen_c, cw
        self.W_fold = (coef @ comps) / scale
        self.b_fold = icpt - self.W_fold @ mean
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        X = _as_matrix(X, "X")
        if X.shape[1] != self.W_fold.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, probe was fitted on {self.W_fold.shape[1]}")
        return X @ self.W_fold.T + self.b_fold

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.scores(X).argmax(axis=1)


# ----------------------------------------------------------------------------- Design A': LW-LDA

class LedoitWolfLDAHead:
    """Design A': LDA with the Ledoit-Wolf (2004) linear-shrinkage covariance, folded to one matmul.

    Sigma_LW = (1 - rho) S + rho nu I, S the pooled within-class covariance (divisor n), nu = tr(S)/D,
    rho the closed-form Ledoit-Wolf optimum in [0, 1]. Sigma^-1 is applied through the thin SVD of the
    centred rows (Sigma^-1 = I/(rho nu) + V diag(1/e_i - 1/(rho nu)) V^T), so no D x D matrix is built
    and D >> n is cheap. W = mu Sigma^-1, b_k = -mu_k^T Sigma^-1 mu_k / 2 + log pi_k. With
    standardize=True the per-feature (total) mean/std are folded into W_fold and b_fold.
    """

    def __init__(self) -> None:
        raise TypeError("use LedoitWolfLDAHead.fit(...)")

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, num_classes: int,
            standardize: bool = True) -> "LedoitWolfLDAHead":
        X, y = _check_fit_inputs(X, y, num_classes)
        n, d = X.shape
        if standardize:
            mean = X.mean(axis=0)
            sd = X.std(axis=0)
            scale = np.where(sd > _EPS, sd, 1.0)
        else:
            mean, scale = np.zeros(d), np.ones(d)
        Z = (X - mean) / scale
        mu = np.stack([Z[y == c].mean(axis=0) for c in range(num_classes)])
        pi = np.bincount(y, minlength=num_classes) / n
        R = Z - mu[y]                                    # within-class residuals
        _, sv, Vt = np.linalg.svd(R, full_matrices=False)
        lam = sv ** 2 / n                                # non-zero spectrum of S = R^T R / n
        trace = float(lam.sum())
        if not trace > _EPS:
            raise ValueError("within-class covariance is (numerically) zero: features carry no variance")
        nu = trace / d
        sum_sq = float((lam ** 2).sum())                 # ||S||_F^2
        d2 = (sum_sq - d * nu ** 2) / d                  # ||S - nu I||_F^2 / D
        row_sq = np.einsum("ij,ij->i", R, R)
        b2_bar = (float((row_sq ** 2).sum()) - n * sum_sq) / (n ** 2 * d)
        rho = 1.0 if d2 <= 1e-15 * nu ** 2 else min(max(b2_bar, 0.0), d2) / d2
        e = (1.0 - rho) * lam + rho * nu                 # Sigma eigenvalues on span(V)
        base = 1.0 / (rho * nu) if rho * nu > 0 else 0.0
        if rho * nu == 0 and (sv.size < d or np.any(e <= 0)):
            raise ValueError("Ledoit-Wolf rho is 0 and the sample covariance is singular")
        c = 1.0 / e - base
        W = base * mu + ((mu @ Vt.T) * c) @ Vt           # K x D  =  mu Sigma^-1
        b = -0.5 * np.einsum("kd,kd->k", mu, W) + np.log(pi)
        self = object.__new__(cls)
        self.num_classes = int(num_classes)
        self.priors, self.rho, self.nu = pi, float(rho), float(nu)
        self.mean_, self.scale_, self.class_means_ = mean, scale, mu
        self._lam, self._Vt = lam, Vt
        self.W_fold = W / scale
        self.b_fold = b - self.W_fold @ mean
        return self

    def covariance_matrix(self) -> np.ndarray:
        """Dense Sigma_LW in the (standardized) feature space. O(D^2) memory: for tests/diagnostics."""
        d = self._Vt.shape[1]
        S = (self._Vt.T * self._lam) @ self._Vt
        return (1.0 - self.rho) * S + self.rho * self.nu * np.eye(d)

    def scores(self, X: np.ndarray) -> np.ndarray:
        X = _as_matrix(X, "X")
        if X.shape[1] != self.W_fold.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, head was fitted on {self.W_fold.shape[1]}")
        return X @ self.W_fold.T + self.b_fold

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.scores(X).argmax(axis=1)


# ----------------------------------------------------------------------------- Design E: conformal

class SplitConformalPredictor:
    """Split conformal prediction with the THR/LAC score s = 1 - softmax(logits)[y]
    (Vovk et al. 2005; Angelopoulos & Bates 2023).

    Marginal guarantee: P(y in C(x)) >= 1 - alpha, over calibration + test draws, when the
    calibration rows are exchangeable with the test row and were NOT used to fit the model.
    q_hat is the k-th smallest calibration score, k = ceil((n + 1)(1 - alpha)); if k > n the
    calibration set is too small for this alpha, q_hat = +inf and every set is the full label set
    (`is_trivial`), which is the honest answer rather than a fake guarantee.
    """

    def __init__(self) -> None:
        self.alpha: Optional[float] = None
        self.q_hat: Optional[float] = None
        self.k_index: Optional[int] = None
        self.n_cal: Optional[int] = None
        self.num_classes: Optional[int] = None

    @property
    def is_trivial(self) -> bool:
        return self.q_hat is not None and math.isinf(self.q_hat)

    def calibrate(self, cal_logits: np.ndarray, cal_labels: np.ndarray,
                  alpha: float = 0.1) -> "SplitConformalPredictor":
        if not (isinstance(alpha, (int, float, np.floating)) and math.isfinite(alpha) and 0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        z = _as_matrix(cal_logits, "cal_logits")
        y = _as_labels(cal_labels, "cal_labels")
        if y.size != z.shape[0]:
            raise ValueError(f"cal_logits has {z.shape[0]} rows but cal_labels has {y.size}")
        if y.max() >= z.shape[1]:
            raise ValueError(f"label {int(y.max())} out of range for {z.shape[1]} classes")
        n = y.size
        scores = 1.0 - _softmax(z)[np.arange(n), y]
        k = max(math.ceil(round((n + 1) * (1.0 - alpha), 9)), 1)
        self.q_hat = math.inf if k > n else float(np.partition(scores, k - 1)[k - 1])
        self.alpha, self.k_index, self.n_cal, self.num_classes = float(alpha), int(k), int(n), z.shape[1]
        return self

    def _mask(self, test_logits: np.ndarray) -> np.ndarray:
        if self.q_hat is None:
            raise RuntimeError("call calibrate() first")
        z = _as_matrix(test_logits, "test_logits")
        if z.shape[1] != self.num_classes:
            raise ValueError(f"test_logits has {z.shape[1]} classes, calibrated for {self.num_classes}")
        return _softmax(z) >= 1.0 - self.q_hat            # s <= q_hat  <=>  p_k >= 1 - q_hat

    def predict_set(self, test_logits: np.ndarray) -> List[List[int]]:
        return [np.flatnonzero(row).tolist() for row in self._mask(test_logits)]

    def predict_with_abstain(self, test_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(pred, set_size). pred[i] is the single class when |C(x_i)| == 1, else -1 (abstain:
        ambiguous |C| > 1 or empty |C| == 0). Abstained rows are exactly `pred == -1`."""
        mask = self._mask(test_logits)
        size = mask.sum(axis=1)
        pred = np.where(size == 1, mask.argmax(axis=1), -1).astype(np.int64)
        return pred, size.astype(np.int64)


# ----------------------------------------------------------------------------- micro-benchmark

def _bench_us(fn, reps: int) -> float:
    import time
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6


if __name__ == "__main__":
    # Not an assertion, a measurement: numbers depend on the box and its load (see loadavg).
    rng = np.random.default_rng(0)
    K, D = 18, 8192
    W, b, x = rng.standard_normal((K, D)), rng.standard_normal(K), rng.standard_normal(D)
    print(f"loadavg={os.getloadavg()}")
    print(f"single-row GEMV  K={K} D={D} float64: {_bench_us(lambda: x @ W.T + b, 2000):8.2f} us/call")
    W32, b32, x32 = W.astype(np.float32), b.astype(np.float32), x.astype(np.float32)
    print(f"single-row GEMV  K={K} D={D} float32: {_bench_us(lambda: x32 @ W32.T + b32, 2000):8.2f} us/call")
    cal_z, cal_y = rng.standard_normal((500, K)), rng.integers(0, K, 500)
    cp = SplitConformalPredictor().calibrate(cal_z, cal_y, 0.1)
    one, many = rng.standard_normal((1, K)), rng.standard_normal((100000, K))
    print(f"conformal abstain, 1 row / call     : {_bench_us(lambda: cp.predict_with_abstain(one), 5000):8.2f} us/call")
    t = _bench_us(lambda: cp.predict_with_abstain(many), 20)
    print(f"conformal abstain, batch 100000 rows: {t / 100000:8.4f} us/row amortised")
