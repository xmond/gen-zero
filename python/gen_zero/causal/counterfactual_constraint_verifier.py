"""Bidirectional counterfactual logic-residual verifier for NLI manifolds.

A single-direction (forward-only) probe misclassifies counterfactual and
adversarial word-order / numeric / negation flips: the forward hidden vector
drifts on the representation manifold and local synonym overlap drowns the
distinguishing signal. This verifier enforces two consistency constraints that
a one-direction probe cannot, then an adversarial repulsion operator that
excludes the shared drift so the residual flip becomes linearly separable.

1. Bidirectional consistency projection.
   Forward entailment support

       S = cos(P, H) in [-1, 1].

   Backward necessity

       N = max(0, S)        (H must lie in P's forward cone).

   Counterfactual contradiction residual

       R_contra = ||Proj(H) ∩ ¬P|| / ||H|| = max(0, -S)   in [0, 1].

   If P ==> H, then R_contra = 0: H projects no mass onto the half-space opposite
   P, so a forward entailment cannot be contradicted backward. Orthogonal (neutral)
   mass

       O = sin(angle) = sqrt(1 - S^2) in [0, 1].

   The three non-negative evidences (max(0, S), O, R_contra) place a pair on an
   equilateral 2-simplex; classification is nearest-vertex and the margin is the
   probability gap to the second class. The three classes sit at 120-degree
   vertices, so a clean example lands at its vertex with a wide margin. An
   ambiguous-cosine neutral (S just under the entailment threshold) is rescued by
   the orthogonal-mass axis, which a one-direction cosine line cannot represent.

2. Adversarial counterfactual repulsion operator (VitaminC fine flips).
   Given a pair (P, H) and a counterfactual reference (P0, H0) that shares the
   lexical bulk but lacks the flip, the operator isolates the perturbation

       delta = (H - H0) - (P - P0),

   excludes the component of ``delta`` aligned with the shared bulk direction
   (the alien drift that collapses representations), and measures the cosine
   orthogonality repulsion penalty

       penalty = cos^2(delta, bulk) in [0, 1]

   (high when the flip was masked by the bulk, zero when already separated). The
   excluded residual is then tested for negation alignment against -P: a flip that
   points into the half-space opposite the premise is contradiction evidence
   added to the simplex, which widens the manifold margin and can flip a
   forward-only entailment call to the correct contradiction label.

   Limit: the bulk is P0 itself (an orthogonal projection, so the operator is
   rotation-equivariant). When P0 == P, delta_perp is orthogonal to P and the
   negation evidence is exactly 0: a reference whose premise equals the
   premise adds no information, and no rotation-equivariant operator can
   extract a flip from it. The reference premise must differ from P (e.g. the
   shared bulk without the asserted fact) for the operator to fire.

Everything is pure real-vector linear algebra. There is no string processing,
word list, regex, language tokeniser, or dataset-specific numeric parser: the
inputs are already-measured representation vectors, so the geometry is
language-agnostic and zero-rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["NLI_CLASSES", "BidirectionalVerdict", "CounterfactualConstraintVerifier"]

NLI_CLASSES = ("entailment", "neutral", "contradiction")

# Orthonormal Helmert contrast (3, 2) and the unit equilateral vertices (3, 2),
# ordered (entailment, neutral, contradiction).
_HELMERT = np.array([
    [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
    [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
    [0.0, -2.0 / np.sqrt(6.0)],
])
_FRAME = np.sqrt(1.5) * _HELMERT  # rows: entailment, neutral, contradiction vertices
_FRAME = _FRAME / np.linalg.norm(_FRAME, axis=-1, keepdims=True)


def _real_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate and return a nonempty real finite 1-D float vector."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real vector")
    v = np.asarray(raw, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D vector")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must contain only finite values")
    return v


def _pair(premise: ArrayLike, hypothesis: ArrayLike):
    """Validate a premise/hypothesis vector pair of matching nonzero, finite-norm shape."""
    p = _real_vector(premise, "premise")
    h = _real_vector(hypothesis, "hypothesis")
    if p.shape != h.shape:
        raise ValueError("premise and hypothesis must have the same shape")
    n_p, n_h = np.linalg.norm(p), np.linalg.norm(h)
    # Every element of p/h is finite (checked above), but the sum of squares
    # inside the norm can still overflow float64 (e.g. two 1e308 elements).
    # Fail closed rather than silently dividing by an infinite norm downstream.
    if not (np.isfinite(n_p) and np.isfinite(n_h)):
        raise ValueError("premise and hypothesis must have a finite norm")
    if n_p == 0.0 or n_h == 0.0:
        raise ValueError("premise and hypothesis must have nonzero norm")
    return p, h


@dataclass(frozen=True)
class BidirectionalVerdict:
    """Result of a bidirectional counterfactual consistency check.

    ``coordinates`` is the 2-D simplex coordinate; ``logits`` are the negative
    squared distances to the three vertices (entailment, neutral, contradiction)
    centered to zero mean; ``probabilities`` softmax over those logits;
    ``predicted_index`` is the argmax class; ``margin`` is the probability gap
    between the top and second class in [0, 1].
    """

    forward_support: float
    backward_necessity: float
    counterfactual_residual: float
    orthogonal_mass: float
    repulsion_penalty: float
    negation_alignment: float
    coordinates: NDArray[np.float64]
    logits: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    predicted_index: int
    margin: float


class CounterfactualConstraintVerifier:
    """Bidirectional counterfactual logic-residual verifier for NLI manifolds.

    Call :meth:`verify` with measured premise/hypothesis representation vectors.
    Supply ``cf_premise`` / ``cf_hypothesis`` (a counterfactual reference pair that
    shares the bulk but lacks the flip) to activate the adversarial repulsion
    operator for fine-grained numeric/negation flips.
    """

    classes = NLI_CLASSES

    def __init__(self, repulsion_weight: float = 1.0, temperature: float = 1.0) -> None:
        self.repulsion_weight = float(repulsion_weight)
        if not np.isfinite(self.repulsion_weight) or self.repulsion_weight <= 0:
            raise ValueError("repulsion_weight must be finite and positive")
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    @property
    def helmert(self) -> NDArray[np.float64]:
        """Orthonormal Helmert contrasts (3, 2), as an independent copy."""
        return _HELMERT.copy()

    @property
    def frame(self) -> NDArray[np.float64]:
        """Unit equilateral simplex vertices (3, 2), as an independent copy."""
        return _FRAME.copy()

    @staticmethod
    def _cosine(p: NDArray[np.float64], h: NDArray[np.float64]) -> float:
        return float(np.dot(p, h) / (np.linalg.norm(p) * np.linalg.norm(h)))

    # -- bidirectional consistency components ---------------------------------

    def forward_entailment_support(self, premise: ArrayLike, hypothesis: ArrayLike) -> float:
        """Forward support S = cos(P, H) in [-1, 1]."""
        p, h = _pair(premise, hypothesis)
        return self._cosine(p, h)

    def backward_necessity(self, premise: ArrayLike, hypothesis: ArrayLike) -> float:
        """Backward necessity N = max(0, S): H must lie in P's forward cone."""
        p, h = _pair(premise, hypothesis)
        return float(max(0.0, self._cosine(p, h)))

    def counterfactual_residual(self, premise: ArrayLike, hypothesis: ArrayLike) -> float:
        """R_contra = ||Proj(H) ∩ ¬P|| / ||H|| = max(0, -S) in [0, 1].

        Zero exactly when no part of H projects onto the half-space opposite P,
        which is the necessity condition P ==> H.
        """
        p, h = _pair(premise, hypothesis)
        return float(max(0.0, -self._cosine(p, h)))

    def orthogonal_mass(self, premise: ArrayLike, hypothesis: ArrayLike) -> float:
        """O = sin(angle) = sqrt(1 - S^2): mass of H orthogonal to P, in [0, 1]."""
        p, h = _pair(premise, hypothesis)
        s = self._cosine(p, h)
        return float(np.sqrt(max(0.0, 1.0 - s * s)))

    # -- adversarial counterfactual repulsion operator ------------------------

    @staticmethod
    def _exclude_along(residual: ArrayLike, reference: ArrayLike):
        """Project ``residual`` onto the orthogonal complement of ``reference``.

        This is the exclusion (排异) core: the component of ``residual`` collinear
        with the shared ``reference`` drift is removed so the distinguishing
        direction is exposed. Returns ``(excluded_residual, cos^2_penalty)``.
        """
        r = _real_vector(residual, "residual")
        ref = _real_vector(reference, "reference")
        if r.shape != ref.shape:
            raise ValueError("residual and reference must have the same shape")
        n_ref = np.linalg.norm(ref)
        n_r = np.linalg.norm(r)
        if not (np.isfinite(n_ref) and np.isfinite(n_r)):
            raise ValueError("residual and reference must have a finite norm")
        if n_ref == 0.0 or n_r == 0.0:
            return r.copy(), 0.0
        unit = ref / n_ref
        projection = (np.dot(r, unit)) * unit
        # Take the cosine of the unit vectors, then square it. Squaring the raw
        # dot product first overflows to inf/inf = nan for finite inputs near 1e150.
        cos = float(np.dot(r / n_r, unit))
        return r - projection, min(1.0, cos * cos)  # cos^2 in [0, 1]

    def repulsion(
        self,
        premise: ArrayLike,
        hypothesis: ArrayLike,
        cf_premise: ArrayLike,
        cf_hypothesis: ArrayLike,
    ):
        """Adversarial counterfactual repulsion for a fine-grained flip.

        Returns ``(delta_perp, penalty, negation_alignment)``:

        - ``delta_perp``: perturbation residual ``delta = (H - H0) - (P - P0)``
          with the shared-bulk component excluded;
        - ``penalty``: cosine-orthogonality repulsion penalty in [0, 1] (high
          when the flip was masked by the bulk);
        - ``negation_alignment``: in [0, 1], how strongly the excluded flip points
          into the half-space opposite the premise.
        """
        p, h = _pair(premise, hypothesis)
        p0, h0 = _pair(cf_premise, cf_hypothesis)
        if p.shape != p0.shape:
            raise ValueError("premise and counterfactual premise must share shape")
        delta = (h - h0) - (p - p0)            # counterfactual perturbation residual
        n_delta = np.linalg.norm(delta)
        n_p0 = np.linalg.norm(p0)
        if not (np.isfinite(n_delta) and np.isfinite(n_p0)):
            raise ValueError("counterfactual repulsion delta must have a finite norm")
        if n_delta > 0 and n_p0 > 0 and abs(float(np.dot(delta, p0))) <= 1e-12 * n_delta * n_p0:
            delta_perp = delta.copy()
            penalty = 0.0
        else:
            # The shared bulk direction is p0 itself. Subtracting its
            # per-coordinate mean (a prior version did `p0 - p0.mean()`) is
            # not equivariant under an orthogonal change of basis: the mean
            # is a coordinate-axis-dependent quantity, so a rotated problem
            # would get a different, inconsistent bulk direction and thus a
            # different penalty/negation_alignment. `_pair` above already
            # guarantees p0 has nonzero norm, so no fallback is needed.
            delta_perp, penalty = self._exclude_along(delta, p0)
        n_dp = np.linalg.norm(delta_perp)
        n_p = np.linalg.norm(p)
        if n_dp == 0.0 or n_p == 0.0:
            neg = 0.0
        else:
            neg = float(max(0.0, -np.dot(delta_perp, p) / (n_dp * n_p)))
        return delta_perp, penalty, neg

    # -- full bidirectional verification -------------------------------------

    def verify(
        self,
        premise: ArrayLike,
        hypothesis: ArrayLike,
        cf_premise: ArrayLike | None = None,
        cf_hypothesis: ArrayLike | None = None,
    ) -> BidirectionalVerdict:
        """Run the bidirectional consistency check and (optionally) repulsion.

        Without a counterfactual reference the verdict is the pure bidirectional
        simplex decision. With a reference, the repulsion operator supplies
        negation evidence that widens the contradiction margin for adversarial
        flips.
        """
        if (cf_premise is None) != (cf_hypothesis is None):
            raise ValueError("cf_premise and cf_hypothesis must both be provided or both omitted")
        p, h = _pair(premise, hypothesis)
        s = self._cosine(p, h)
        if not np.isfinite(s):
            raise ValueError("forward support must be finite")
        e_entail = max(0.0, s)
        e_contra_fwd = max(0.0, -s)
        e_neutral = float(np.sqrt(max(0.0, 1.0 - s * s)))

        repulsion_penalty = 0.0
        negation_alignment = 0.0
        if cf_premise is not None and cf_hypothesis is not None:
            _, penalty, neg = self.repulsion(p, h, cf_premise, cf_hypothesis)
            repulsion_penalty = penalty
            negation_alignment = neg
            # Adversarial contradiction evidence widens the manifold margin.
            e_contra = float(
                np.sqrt(e_contra_fwd ** 2 + (self.repulsion_weight * negation_alignment) ** 2)
            )
            # Entailment evidence is damped by counterfactual negation flip.
            e_entail = float(max(0.0, e_entail - self.repulsion_weight * negation_alignment))
        else:
            e_contra = e_contra_fwd

        evidence = np.array([e_entail, e_neutral, e_contra], dtype=np.float64)
        coordinate = (evidence - evidence.mean()) @ _HELMERT
        distances = np.sum((coordinate - _FRAME) ** 2, axis=-1)
        logits = -distances
        logits -= logits.mean()
        with np.errstate(over="ignore"):
            scaled = (logits - logits.max()) / self.temperature
            weights = np.exp(scaled)
        probabilities = weights / weights.sum()
        predicted = int(np.argmax(probabilities))
        order = np.sort(probabilities)[::-1]
        margin = float(order[0] - order[1])
        return BidirectionalVerdict(
            forward_support=s,
            backward_necessity=max(0.0, s),
            counterfactual_residual=e_contra_fwd,
            orthogonal_mass=e_neutral,
            repulsion_penalty=repulsion_penalty,
            negation_alignment=negation_alignment,
            coordinates=coordinate,
            logits=logits,
            probabilities=probabilities,
            predicted_index=predicted,
            margin=margin,
        )
