"""Contrastive representation probing and Helmert calibration for three-way NLI.

ContrastiveManifoldProbe learns a relational embedding-to-simplex map from
labeled pairs and classifies by squared distance to the semantic vertices.
HelmertNLIProbe instead calibrates already measured label logits.

Class order is always (entailment, neutral, contradiction). Here "3-simplex"
means three vertices: an equilateral triangle (a mathematical 2-simplex).
With the orthonormal Helmert contrast matrix H in R^(3 x 2),

    H.T H = I_2,  H H.T = I_3 - 11.T / 3,
    V = sqrt(3/2) H,  V V.T = (3/2) I_3 - 11.T / 2.

Thus the rows of V have unit length and pairwise cosine -1/2 (120 degrees).
They sum to zero: each vertex is antipodal to the sum of the other two;
individual vertex pairs are not antipodal.

For contextual label logits l and context-free label logits b, use

    r = l - b,  z = r H,  s = z H.T = r - mean(r),
    p = softmax(s / temperature).

Equivalently, p is proportional to (P(label|context)/P(label|null))**(1/T).
Any additive class bias shared by l and b cancels exactly. This removes the
vocabulary-frequency component represented by the supplied prior, not arbitrary
context-dependent bias. Geometry alone does not calibrate vocabulary scores or
align an arbitrary language-model hidden state with semantic labels.

Callers must measure b with the same model, label verbalizers, scoring method,
and instruction/template as l, with task context removed. Supply logits or
log-probabilities, never raw probabilities. Multi-token verbalizers must use the
same sequence-scoring rule in both calls. This module does not run a language
model or assume a dataset's numeric label encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["NLI_CLASSES", "NLIProbeResult", "HelmertNLIProbe", "ContrastiveManifoldProbe"]

NLI_CLASSES = ("entailment", "neutral", "contradiction")


def _label_scores(values: ArrayLike, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real scores")
    scores = np.asarray(raw, dtype=np.float64)
    if scores.ndim == 0 or scores.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3)")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} must contain finite scores")
    return scores


@dataclass(frozen=True)
class NLIProbeResult:
    """Arrays preserve input batch axes; coordinates have final dimension two.

    Predictions are class indices in NLI_CLASSES; ties select the first class.
    An exact null signal has uniform probabilities, not evidence of entailment.
    """

    coordinates: NDArray[np.float64]
    logits: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    predicted_indices: NDArray[np.intp]


class HelmertNLIProbe:
    """Project prior-subtracted label evidence onto the Helmert contrast plane.

    Accept a single score vector (3,) or batches (..., 3). The prior must
    have exactly the input shape or be a shared (3,) vector. It is required:
    no implicit uniform or heuristic vocabulary prior is substituted.

    Temperature scales the calibrated logits, not the prior independently.
    The two-dimensional projection preserves all three-way log-odds; it does
    not normalize the evidence vector, which would discard its magnitude.
    """

    classes = NLI_CLASSES

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    @property
    def helmert(self) -> NDArray[np.float64]:
        """Orthonormal contrasts (3, 2), returned as an independent array."""
        return np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(6)],
            [-1 / np.sqrt(2), 1 / np.sqrt(6)],
            [0, -2 / np.sqrt(6)],
        ])

    @property
    def frame(self) -> NDArray[np.float64]:
        """Unit simplex vertices (3, 2), ordered by classes."""
        return np.sqrt(1.5) * self.helmert

    def project(
        self, contextual_logits: ArrayLike, context_free_logits: ArrayLike,
    ) -> NLIProbeResult:
        """Subtract the measured prior, project, and return calibrated scores.

        ``logits`` are centered, untempered log-ratio scores. Non-finite inputs,
        mismatched batch axes, or arithmetic overflow raise ValueError.
        """
        contextual = _label_scores(contextual_logits, "contextual_logits")
        prior = _label_scores(context_free_logits, "context_free_logits")
        if prior.shape != (3,) and prior.shape != contextual.shape:
            raise ValueError("context_free_logits must have shape (3,) or match contextual_logits")
        h = self.helmert
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                residual = contextual - prior
                # Center before multiplication to suppress common-offset leakage.
                # Divide before summing so the mean of large finite scores is safe.
                residual = residual - np.sum(residual / 3, axis=-1, keepdims=True)
                coordinates = residual @ h
                logits = coordinates @ h.T
                # Shift before temperature scaling; extreme negative scores may
                # safely saturate to -inf and hence probability zero.
                with np.errstate(over="ignore"):
                    scaled = (logits - np.max(logits, axis=-1, keepdims=True)) / self.temperature
                weights = np.exp(scaled)
                probabilities = weights / np.sum(weights, axis=-1, keepdims=True)
        except FloatingPointError as exc:
            raise ValueError("scores exceed the supported floating-point range") from exc
        return NLIProbeResult(
            coordinates=coordinates,
            logits=logits,
            probabilities=probabilities,
            predicted_indices=np.asarray(np.argmax(probabilities, axis=-1), dtype=np.intp),
        )


class ContrastiveManifoldProbe(HelmertNLIProbe):
    """Supervised relational ridge map with squared-distance simplex decisions.

    Arbitrary hidden states have no intrinsic entailment axis: labeled training
    pairs align the relational manifold with the three semantic vertices.
    The lift A(d) = [d, -d]/sqrt(2) is orthogonal to every symmetric pair
    [u, u]. For d=|p-h| this is an antisymmetric *space*, not a swap-odd
    function; ordered p,h features retain the direction lost by absolute value.
    The lift reweights discrepancy in the ridge metric; it does not create
    semantic information absent from the input representations.
    Swapping equivalent sentences must not manufacture a contradiction.
    """

    def __init__(self, ridge: float = 1.0, temperature: float = 1.0) -> None:
        super().__init__(temperature)
        self.ridge = float(ridge)
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self._weights = None

    @staticmethod
    def _pair(premise: ArrayLike, hypothesis: ArrayLike):
        p, h = np.asarray(premise), np.asarray(hypothesis)
        if np.iscomplexobj(p) or np.iscomplexobj(h):
            raise ValueError("representations must be real")
        p, h = np.asarray(p, dtype=float), np.asarray(h, dtype=float)
        if p.ndim < 1 or p.shape != h.shape or p.shape[-1] == 0:
            raise ValueError("representations must have matching nonempty feature axes")
        if not (np.isfinite(p).all() and np.isfinite(h).all()):
            raise ValueError("representations must be finite")
        return p, h

    @staticmethod
    def orthogonal_complement(difference: ArrayLike) -> NDArray[np.float64]:
        """Isometric lift into the complement of the symmetric pair subspace."""
        d, _ = ContrastiveManifoldProbe._pair(difference, difference)
        return np.concatenate((d / np.sqrt(2), -d / np.sqrt(2)), axis=-1)

    @staticmethod
    def relational_features(premise: ArrayLike, hypothesis: ArrayLike) -> NDArray[np.float64]:
        """Four relational facets plus the orthogonal difference lift.

        A joint per-pair scale bounds arithmetic while retaining relative norms.
        No labels, text, regex, or batch-dependent statistics enter this map.
        """
        p, h = ContrastiveManifoldProbe._pair(premise, hypothesis)
        scale = np.maximum(np.maximum(np.max(np.abs(p), axis=-1, keepdims=True),
                                      np.max(np.abs(h), axis=-1, keepdims=True)), 1.0)
        p, h = p / scale, h / scale
        d = np.abs(p - h)
        return np.concatenate((p, h, d, p * h,
                               ContrastiveManifoldProbe.orthogonal_complement(d)), axis=-1)

    def fit(self, premise: ArrayLike, hypothesis: ArrayLike, labels: ArrayLike):
        """Fit only training pairs; numeric labels follow NLI_CLASSES."""
        x = self.relational_features(premise, hypothesis)
        y = np.asarray(labels)
        if x.ndim != 2 or len(x) == 0 or y.shape != (len(x),):
            raise ValueError("fit requires nonempty matrices and one label per pair")
        if y.dtype.kind not in 'iu' or np.any((y < 0) | (y >= 3)):
            raise ValueError("labels must be integer NLI class indices")
        self._mean = x.mean(axis=0)
        self._target_mean = self.frame[y].mean(axis=0)
        xc = x - self._mean
        targets = self.frame[y] - self._target_mean
        # Dual solve keeps the cost bounded by examples, not embedding width.
        alpha = np.linalg.solve(xc @ xc.T + self.ridge * np.eye(len(x)), targets)
        self._weights = xc.T @ alpha
        return self

    def _result(self, coordinates):
        distances = np.sum((coordinates[..., None, :] - self.frame) ** 2, axis=-1)
        logits = -distances
        logits -= logits.mean(axis=-1, keepdims=True)
        with np.errstate(over="ignore"):
            weights = np.exp((logits - logits.max(axis=-1, keepdims=True)) / self.temperature)
        return NLIProbeResult(coordinates, logits, weights / weights.sum(axis=-1, keepdims=True),
                              np.asarray(logits.argmax(axis=-1), dtype=np.intp))

    def project(self, premise: ArrayLike, hypothesis: ArrayLike) -> NLIProbeResult:
        if self._weights is None:
            raise ValueError("fit the probe before projection")
        x = self.relational_features(premise, hypothesis)
        if x.shape[-1] != self._mean.shape[-1]:
            raise ValueError("representation dimension differs from training")
        return self._result((x - self._mean) @ self._weights + self._target_mean)

    @classmethod
    def leave_one_out(cls, premise, hypothesis, labels, ridge=1.0):
        """Each prediction excludes its own label; -1 denotes unlabeled rows."""
        p, h = cls._pair(premise, hypothesis)
        y = np.asarray(labels)
        if p.ndim != 2 or y.shape != (len(p),) or y.dtype.kind not in 'iu':
            raise ValueError("leave_one_out requires matrices and integer labels")
        if np.any((y < -1) | (y >= 3)):
            raise ValueError("invalid NLI class index")
        probe = cls(ridge=ridge)
        coordinates = np.zeros((len(p), 2))
        missing = []
        for i in range(len(p)):
            train = (np.arange(len(p)) != i) & (y >= 0)
            missing.append(sorted(set(range(3)) - set(y[train].tolist())))
            if train.any():
                probe.fit(p[train], h[train], y[train])
                coordinates[i] = probe.project(p[i], h[i]).coordinates
        return probe._result(coordinates), missing
