"""CALA: confidence-adaptive prior shift for post-hoc logit adjustment.

    Logit_y(x) = f_y(x) - tau * Phi(x) * log(pi_y)

pi is the TRAINING class prior, Phi(x) in [0, 1] is a per-row gate computed from the row's own
scores or features (never from labels): Phi -> 0 on confident rows (their sharp boundary is
untouched, so accuracy is protected), Phi -> 1 on ambiguous rows (the full Menon et al. 2021 prior
compensation, which lifts rare-class recall).

Sign convention. The task brief writes the term as `+ tau * Phi * log(pi_y)`. With log(pi) <= 0
that would push the RARE classes down and the majority up, the opposite of the stated goal (rescue
minority recall). This module therefore SUBTRACTS the term, exactly as
`spec21_advanced_heads.logit_adjust` does; Phi == 1 reproduces that function bit for bit.

Gates (all label-free at inference; gamma >= 0 sharpens the transition):
  entropy   Phi = (H(p) / log K) ** gamma            p = softmax(f / scale)
  margin    Phi = (1 - (p_(1) - p_(2))) ** gamma      top-two probability gap
  centroid  Phi = (d_(1) / d_(2)) ** gamma            nearest / second-nearest class-centroid distance
  static    Phi = 1                                   the old global shift, kept as the baseline
`scale` is the label-free logit scale (standard deviation of out-of-fold scores) that turns raw
head scores into comparable probabilities.

Scope limits: the gate only reshuffles decisions the head already scores as ambiguous. It cannot
create signal, and on a task whose classes are not separable in the head's scores it moves accuracy
by roughly nothing. Whether it helps on a task is an empirical, per-task question the suite answers
with out-of-fold selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

_EPS = 1e-12
GATE_MODES = ("static", "entropy", "margin", "centroid")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _scores(logits, name: str = "logits") -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError(f"{name} must be 2-D with >= 2 classes, got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError(f"{name} contains non-finite values")
    return z


def _check_gamma(gamma: float) -> float:
    if not (isinstance(gamma, (int, float, np.floating)) and math.isfinite(gamma) and gamma >= 0):
        raise ValueError(f"gamma must be a finite non-negative float, got {gamma!r}")
    return float(gamma)


def gate_from_logits(logits: np.ndarray, scale: float, mode: str, gamma: float = 1.0) -> np.ndarray:
    """Phi(x) from the head's own scores. `mode` in {'entropy', 'margin', 'static'}."""
    z = _scores(logits)
    gamma = _check_gamma(gamma)
    if mode == "static":
        return np.ones(z.shape[0])
    if mode not in ("entropy", "margin"):
        raise ValueError(f"unknown score gate {mode!r}; expected 'entropy', 'margin' or 'static'")
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be finite and > 0, got {scale!r}")
    p = _softmax(z / scale)
    if mode == "entropy":
        h = -(p * np.log(np.maximum(p, _EPS))).sum(axis=1) / math.log(z.shape[1])
        phi = h
    else:
        top2 = np.partition(p, -2, axis=1)[:, -2:]
        phi = 1.0 - (top2[:, 1] - top2[:, 0])
    return np.clip(phi, 0.0, 1.0) ** gamma


def gate_from_centroids(features: np.ndarray, class_means: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Phi(x) = (d1 / d2) ** gamma with d1, d2 the distances to the nearest and second-nearest class
    centroid in the representation the head is fitted on. Equidistant rows -> 1, on a centroid -> 0."""
    f = np.asarray(features, dtype=np.float64)
    m = np.asarray(class_means, dtype=np.float64)
    if f.ndim != 2 or m.ndim != 2 or f.shape[1] != m.shape[1] or m.shape[0] < 2:
        raise ValueError(f"features {f.shape} and class_means {m.shape} are incompatible")
    gamma = _check_gamma(gamma)
    d2 = (f ** 2).sum(1)[:, None] - 2.0 * f @ m.T + (m ** 2).sum(1)[None, :]
    d = np.sqrt(np.maximum(np.sort(d2, axis=1)[:, :2], 0.0))
    ratio = d[:, 0] / np.maximum(d[:, 1], _EPS)
    return np.clip(ratio, 0.0, 1.0) ** gamma


def cala_adjust(logits: np.ndarray, priors: np.ndarray, tau: float, phi: np.ndarray,
                eps: float = _EPS) -> np.ndarray:
    """f - tau * Phi * log(pi). Returns a new array; Phi == 0 or tau == 0 returns the input values."""
    z = _scores(logits)
    p = np.asarray(priors, dtype=np.float64)
    if p.ndim != 1 or p.size != z.shape[1] or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-6:
        raise ValueError("priors must be a finite non-negative vector of length K summing to 1")
    if not (isinstance(tau, (int, float, np.floating)) and math.isfinite(tau) and tau >= 0):
        raise ValueError(f"tau must be a finite non-negative float, got {tau!r}")
    g = np.asarray(phi, dtype=np.float64)
    if g.shape != (z.shape[0],):
        raise ValueError(f"phi must have shape ({z.shape[0]},), got {g.shape}")
    if not np.isfinite(g).all() or g.min() < 0.0 or g.max() > 1.0:
        raise ValueError("phi must lie in [0, 1]")
    return z - float(tau) * g[:, None] * np.log(p + eps)[None, :]


@dataclass(frozen=True)
class CalaConfig:
    """One point of the gate grid. tau == 0 is the untouched head."""
    mode: str = "static"
    gamma: float = 1.0
    tau: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in GATE_MODES:
            raise ValueError(f"unknown gate mode {self.mode!r}")
        _check_gamma(self.gamma)
        if not self.tau >= 0:
            raise ValueError(f"tau must be >= 0, got {self.tau!r}")

    @property
    def key(self) -> str:
        if self.tau == 0:
            return "raw"
        return f"static:t{self.tau:g}" if self.mode == "static" else f"{self.mode}:g{self.gamma:g}:t{self.tau:g}"

    def apply(self, logits: np.ndarray, priors: np.ndarray, scale: float,
              centroid_ratio: Optional[np.ndarray] = None) -> np.ndarray:
        """`centroid_ratio`: gate_from_centroids(..., gamma=1) output, required for mode 'centroid';
        this config's gamma is applied here so one precomputed ratio serves every gamma."""
        z = _scores(logits)
        if self.tau == 0:
            return z.copy()
        if self.mode == "centroid":
            if centroid_ratio is None:
                raise ValueError("mode 'centroid' needs a precomputed centroid_ratio")
            phi = np.clip(np.asarray(centroid_ratio, dtype=np.float64), 0.0, 1.0) ** self.gamma
        else:
            phi = gate_from_logits(z, scale, self.mode, self.gamma)
        return cala_adjust(z, priors, self.tau, phi)
