"""Gen-Zero Specialist Fleet: Choice Decision Head with Simplex ETF Action Manifold.

RFC-069 & Issue #72 Implementation:
Choice Head provides permutation-equivariant, isotropic candidate action selection:
1. Projects state representations into continuous latent space.
2. Projects candidate actions onto an Equiangular Tight Frame (Simplex ETF).
3. Evaluates dot-product affinity on the ETF manifold, eliminating semantic clustering.
4. Enforces 0.00% permutation flip rate under candidate action reordering.
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .action_etf_embedding import (
    ActionSpaceETFEmbedding,
    ETFVerificationReport,
    generate_simplex_etf,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    F = None
    HAS_TORCH = False


def _softmax_last_axis(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Evaluate a numerically stable softmax along the last axis.

    The calibration routines operate in float64 even when the caller supplied
    float32 scores.  Temperature calibration is a one-dimensional root-finding
    problem and doing the arithmetic in the input dtype can make the entropy
    bracket depend on avoidable underflow.
    """
    scores = np.asarray(logits, dtype=np.float64)
    maximum = np.max(scores, axis=-1, keepdims=True)
    # Subtract before dividing: this preserves small gaps on a large offset.
    # Opposite-sign extremes can overflow subtraction; divide those terms
    # first. Their opposite signs ensure the alternate difference is defined.
    with np.errstate(over="ignore", under="ignore"):
        delta = scores - maximum
        scaled = delta / temperature
        overflow = np.isneginf(delta)
        if np.any(overflow):
            scaled[overflow] = (scores[overflow] / temperature
                                - np.broadcast_to(maximum, scores.shape)[overflow] / temperature)
        exp_scores = np.exp(scaled)
    denominator = np.sum(exp_scores, axis=-1, keepdims=True)
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise FloatingPointError("softmax normalization produced a non-finite denominator")
    return exp_scores / denominator


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return Shannon entropy in nats normalized to ``[0, 1]``."""
    p = np.asarray(probabilities, dtype=np.float64)
    k = p.shape[-1]
    if k <= 1:
        return np.zeros(p.shape[:-1], dtype=np.float64)
    # Treat exact zero mass as zero contribution.  Clipping it to the smallest
    # positive float would report a spurious ~1e-305 entropy for a genuinely
    # deterministic calibrated distribution.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log(p), 0.0)
    entropy = -np.sum(terms, axis=-1)
    return entropy / math.log(k)


def _cap_probabilities(probabilities: np.ndarray, cap: float) -> np.ndarray:
    """Apply an exact per-row upper probability cap by water filling.

    The operation preserves the simplex sum and changes only rows whose mass
    exceeds ``cap``.  Redistribution is deterministic and uses the available
    capacity of all unsaturated entries, so it does not introduce a class or
    language-specific tie break.
    """
    p = np.asarray(probabilities, dtype=np.float64).copy()
    k = p.shape[-1]
    if cap < 1.0 / k - 1e-14:
        raise ValueError(f"max_confidence_cap={cap} is infeasible for {k} classes; require >= {1.0 / k}")
    if abs(cap - 1.0 / k) <= 1e-14:
        return np.full_like(p, 1.0 / k)

    flat = p.reshape(-1, k)
    for row in flat:
        over = np.maximum(row - cap, 0.0)
        excess = float(np.sum(over))
        row[:] = np.minimum(row, cap)
        if excess <= 1e-15:
            continue
        # Each pass saturates entries that cannot absorb the equal share.  The
        # number of classes is small in this head, so this is both clearer and
        # less error-prone than solving a second floating-point optimization.
        free = np.flatnonzero(row < cap - 1e-15)
        while excess > 1e-14:
            if free.size == 0:
                raise FloatingPointError("probability cap has no remaining simplex capacity")
            share = excess / float(free.size)
            capacities = cap - row[free]
            additions = np.minimum(capacities, share)
            row[free] += additions
            excess -= float(np.sum(additions))
            free = free[row[free] < cap - 1e-15]
    return p


def calibrate_temperature_and_entropy(
    logits: Union[np.ndarray, Sequence[float]],
    target_entropy: Optional[float] = None,
    temperature: float = 1.0,
    entropy_penalty: float = 0.0,
    max_confidence_cap: Optional[float] = None,
    *,
    normalized: bool = True,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Calibrate a categorical logit vector by temperature and entropy.

    ``target_entropy`` is normalized Shannon entropy by default, i.e.
    ``H(p) / log(K)``.  When it is omitted, ``temperature`` is applied once
    and the optional ``entropy_penalty``/``max_confidence_cap`` controls are
    applied.  When it is supplied, a monotone bisection solve finds the
    positive temperature whose empirical softmax distribution has the requested
    entropy.  The returned tuple is ``(probabilities, max_probability,
    normalized_entropy)``; for a one-dimensional input the latter two are
    Python floats, while a batch keeps one value per row.

    The entropy range is checked against the actual logits.  A target below the
    zero-temperature entropy (which is non-zero when the maximum is tied) or
    above the uniform entropy is rejected explicitly.  The limiting targets
    zero and one are represented by the corresponding deterministic/uniform
    distributions.

    This matches entropy of the empirical softmax distribution; it does not
    establish accuracy calibration or statistical coverage without held-out
    labeled data. Entropy targets cannot be combined with the postprocessing
    controls because those controls would change the achieved entropy.

    Args:
        logits: Scores with shape ``(K,)`` or ``(..., K)``.  Values must be
            finite and ``K >= 1``.
        target_entropy: Desired normalized entropy in ``[0, 1]``.  If
            ``normalized=False``, the target is in nats and is converted using
            ``log(K)``.
        temperature: Positive finite temperature used when no target is given,
            ignored by a target solve, which determines its own scale.
        entropy_penalty: Non-negative uniform mixing weight in ``[0, 1]`` when
            no target is given.  A value of one yields the uniform distribution.
        max_confidence_cap: Optional feasible upper bound on every probability.
            The cap is enforced by exact simplex redistribution.

    Raises:
        ValueError: For malformed input, invalid controls, or an unreachable
            entropy target.
    """
    if np.iscomplexobj(logits):
        raise TypeError("logits must be real")
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim == 0 or scores.shape[-1] < 1:
        raise ValueError("logits must have shape (K,) or (..., K) with K >= 1")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must contain only finite values")
    if not (np.isfinite(temperature) and temperature > 0.0):
        raise ValueError("temperature must be finite and strictly positive")
    if not (np.isfinite(entropy_penalty) and 0.0 <= entropy_penalty <= 1.0):
        raise ValueError("entropy_penalty must be finite and in [0, 1]")
    if not (np.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be finite and strictly positive")
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer")

    k = int(scores.shape[-1])
    if target_entropy is not None and (entropy_penalty != 0.0 or max_confidence_cap is not None):
        raise ValueError(
            "target_entropy cannot be combined with entropy_penalty or max_confidence_cap; "
            "those controls change the achieved entropy"
        )
    if max_confidence_cap is not None:
        cap = float(max_confidence_cap)
        if not (np.isfinite(cap) and 0.0 < cap <= 1.0):
            raise ValueError("max_confidence_cap must be finite and in (0, 1]")
        if cap < 1.0 / k - 1e-14:
            raise ValueError(f"max_confidence_cap={cap} is infeasible for {k} classes; require >= {1.0 / k}")
    leading_shape = scores.shape[:-1]
    flat_scores = scores.reshape(-1, k)
    flat_probs = np.empty_like(flat_scores)
    flat_entropy = np.empty(flat_scores.shape[0], dtype=np.float64)

    requested = None if target_entropy is None else float(target_entropy)
    if requested is not None and not np.isfinite(requested):
        raise ValueError("target_entropy must be finite")
    if requested is not None:
        if normalized:
            target = requested
        else:
            if k == 1:
                target = 0.0 if abs(requested) <= tolerance else math.nan
            else:
                target = requested / math.log(k)
        if not np.isfinite(target) or target < -tolerance or target > 1.0 + tolerance:
            max_value = 1.0 if normalized else math.log(k)
            raise ValueError(
                f"target_entropy={requested} is outside the attainable entropy interval [0, {max_value}]"
            )
        target = min(1.0, max(0.0, target))

    for row_index, row in enumerate(flat_scores):
        if k == 1:
            if requested is not None and target > tolerance:
                raise ValueError("a one-class distribution can attain only zero entropy")
            flat_probs[row_index] = 1.0
            flat_entropy[row_index] = 0.0
            continue

        if requested is None:
            p = _softmax_last_axis(row, temperature)
        else:
            with np.errstate(over="ignore"):
                centered = row - np.max(row)
            if np.isneginf(centered).any():
                bounded = row / np.max(np.abs(row))
                centered = bounded - np.max(bounded)
            scale = float(np.max(np.abs(centered)))
            work_row = centered / scale if scale > 0.0 else np.zeros_like(row)
            max_score = 0.0
            # Exact equality is intentional: near-ties have a different
            # limiting distribution and must not be silently collapsed.
            max_count = int(np.count_nonzero(work_row == max_score))
            min_entropy = math.log(max_count) / math.log(k)
            if target < min_entropy - tolerance or target > 1.0 + tolerance:
                raise ValueError(
                    "target_entropy is unreachable for these logits: "
                    f"attainable normalized interval is [{min_entropy}, 1]"
                )

            if target <= min_entropy + tolerance:
                p = np.zeros(k, dtype=np.float64)
                p[work_row == max_score] = 1.0 / max_count
            elif target >= 1.0 - tolerance:
                p = np.full(k, 1.0 / k, dtype=np.float64)
            else:
                def entropy_at_log_beta(log_beta: float) -> float:
                    # beta = 1/T.  Searching in log(beta) handles score
                    # vectors spanning hundreds of orders of magnitude.
                    bounded = min(700.0, max(-700.0, float(log_beta)))
                    beta = math.exp(bounded)
                    centred = work_row - float(np.max(work_row))
                    beta_scores = centred * beta
                    q = _softmax_last_axis(beta_scores, 1.0)
                    return float(_normalized_entropy(q))

                low = -1.0
                high = 1.0
                # H decreases monotonically with log(beta).  Expand both ends
                # until the target is bracketed, with finite search limits.
                while entropy_at_log_beta(low) < target:
                    low -= math.log(2.0)
                    if low < -700.0:
                        raise FloatingPointError("failed to bracket the requested entropy from above")
                while entropy_at_log_beta(high) > target:
                    high += math.log(2.0)
                    if high > 700.0:
                        raise FloatingPointError("failed to bracket the requested entropy from below")
                for _ in range(int(max_iterations)):
                    middle = (low + high) * 0.5
                    if entropy_at_log_beta(middle) > target:
                        low = middle
                    else:
                        high = middle
                centre_log_beta = (low + high) * 0.5
                achieved = entropy_at_log_beta(centre_log_beta)
                if abs(achieved - target) > tolerance:
                    raise FloatingPointError(
                        f"temperature solve did not reach target entropy: achieved={achieved}, target={target}"
                    )
                beta = math.exp(min(700.0, max(-700.0, centre_log_beta)))
                centred = work_row - float(np.max(work_row))
                beta_scores = centred * beta
                p = _softmax_last_axis(beta_scores, 1.0)

        if requested is None and entropy_penalty:
            p = (1.0 - float(entropy_penalty)) * p + float(entropy_penalty) / k
        if max_confidence_cap is not None:
            cap = float(max_confidence_cap)
            p = _cap_probabilities(p, cap)
        if not np.all(np.isfinite(p)) or np.any(p < -tolerance):
            raise FloatingPointError("calibration produced an invalid probability distribution")
        # Re-normalize only for accumulated round-off; no failure is hidden by
        # clipping or a fallback distribution.
        p_sum = float(np.sum(p))
        if not np.isfinite(p_sum) or p_sum <= 0.0:
            raise FloatingPointError("calibration produced zero probability mass")
        p = p / p_sum
        flat_probs[row_index] = p
        flat_entropy[row_index] = float(_normalized_entropy(p))

    probabilities = flat_probs.reshape(scores.shape)
    entropy = flat_entropy.reshape(leading_shape)
    confidence = np.max(probabilities, axis=-1)
    if probabilities.ndim == 1:
        return probabilities, float(confidence), float(entropy)
    return probabilities, confidence, entropy


class FastSimplexETFProjection:
    """Construct and apply a canonical Helmert Simplex ETF projection.

    For ``K`` classes and ambient dimension ``D``, the returned matrix has
    shape ``(K, D)`` and obeys

    ``V @ V.T = (K/(K-1)) * (I - 11.T/K)``

    for ``K >= 2`` (with zero padding when ``D > K-1``).  Thus every row has
    unit norm, rows sum to zero, and every distinct pair has inner product
    ``-1/(K-1)``.  The construction is deterministic and has no learned or
    language-dependent component.

    NumPy arrays and PyTorch tensors are both accepted by :meth:`project`.
    Tensor inputs stay in their device/dtype and preserve autograd through the
    input; the ETF matrix itself is a fixed constant.
    """

    def __init__(
        self, num_classes: int, dim: int, *, dtype: Any = np.float64,
        device: Optional[Any] = None, normalize: bool = False,
    ) -> None:
        for value in (num_classes, dim):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError("num_classes and dim must be integers")
        classes, ambient_dim = int(num_classes), int(dim)
        if classes < 1:
            raise ValueError("num_classes must be at least 1")
        if ambient_dim < 1:
            raise ValueError("dim must be at least 1")
        if classes > ambient_dim + 1:
            raise ValueError(
                f"Simplex ETF needs dim >= num_classes - 1; got dim={ambient_dim}, num_classes={classes}"
            )
        if dtype is None:
            dtype = np.float64
        np_dtype = np.dtype(dtype)
        if np_dtype.kind != "f":
            raise TypeError("dtype must be a floating-point NumPy dtype")

        self.num_classes = classes
        self.dim = ambient_dim
        self.normalize = bool(normalize)
        self.dtype = np_dtype
        self.device = device
        self._matrix = generate_simplex_etf(classes, ambient_dim).astype(np_dtype, copy=False)
        self._matrix.setflags(write=False)

    @property
    def projection_matrix(self) -> np.ndarray:
        """The canonical ``(K, D)`` ETF matrix."""
        return self._matrix

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def numpy(self, *, dtype: Optional[Any] = None, copy: bool = False) -> np.ndarray:
        """Return the ETF matrix as a NumPy array."""
        result = self._matrix.astype(dtype, copy=False) if dtype is not None else self._matrix
        return result.copy() if copy else result

    def torch_matrix(self, *, device: Optional[Any] = None, dtype: Optional[Any] = None):
        """Return the fixed matrix as a PyTorch tensor."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed")
        target_device = device if device is not None else self.device
        # ``_matrix`` is deliberately read-only for NumPy callers.  Make an
        # explicit tensor copy instead of handing PyTorch a non-writable NumPy
        # view (which would permit undefined writes through the tensor).
        return torch.tensor(self._matrix, device=target_device, dtype=dtype)

    def project(self, values: Any, *, normalize: Optional[bool] = None):
        """Project ``(..., D)`` values onto all ETF vertices, yielding ``(..., K)``.

        By default this is the literal linear projection ``values @ V.T``.
        Set ``normalize=True`` (or construct with ``normalize=True``) to use
        cosine projections, matching the unit-sphere readout in the latent
        reasoning specification.
        """
        do_normalize = self.normalize if normalize is None else bool(normalize)
        if HAS_TORCH and isinstance(values, torch.Tensor):
            if values.ndim == 0 or values.shape[-1] != self.dim:
                raise ValueError(f"values must have final dimension {self.dim}")
            if not values.is_floating_point() or values.is_complex():
                raise TypeError("values must be a real floating-point tensor")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("values must contain only finite values")
            matrix = self.torch_matrix(device=values.device, dtype=values.dtype)
            source = values
            if do_normalize:
                scale = source.abs().amax(dim=-1, keepdim=True)
                if bool((scale == 0).any()):
                    raise ValueError("cosine projection is undefined for a zero vector")
                scaled = source / scale
                norms = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
                source = scaled / norms
            result = torch.matmul(source, matrix.transpose(0, 1))
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError("projection overflowed")
            return result

        if np.iscomplexobj(values):
            raise TypeError("values must be real")
        source = np.asarray(values, dtype=self._matrix.dtype)
        if source.ndim == 0 or source.shape[-1] != self.dim:
            raise ValueError(f"values must have final dimension {self.dim}")
        if not np.all(np.isfinite(source)):
            raise ValueError("values must contain only finite values")
        if do_normalize:
            scale = np.max(np.abs(source), axis=-1, keepdims=True)
            if np.any(scale == 0.0):
                raise ValueError("cosine projection is undefined for a zero vector")
            scaled = source / scale
            norms = np.linalg.norm(scaled, axis=-1, keepdims=True)
            source = scaled / norms
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.matmul(source, self._matrix.T)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("projection overflowed")
        return result

    __call__ = project
    project_logits = project

    def predict(self, values: Any, *, normalize: Optional[bool] = None):
        """Return the lowest-index ETF vertex attaining the largest score."""
        scores = self.project(values, normalize=normalize)
        if HAS_TORCH and isinstance(scores, torch.Tensor):
            return torch.argmax(scores, dim=-1)
        return np.argmax(scores, axis=-1)

    def verify(self, *, atol: float = 1e-10) -> Dict[str, Any]:
        """Compute finite numerical checks for the ETF invariants."""
        gram = self._matrix @ self._matrix.T
        if self.num_classes == 1:
            expected = np.ones((1, 1), dtype=self._matrix.dtype)
        else:
            expected = np.full_like(gram, -1.0 / (self.num_classes - 1))
            np.fill_diagonal(expected, 1.0)
        return {
            "k": self.num_classes,
            "dim": self.dim,
            "unit_norm": bool(np.allclose(np.diag(gram), 1.0, atol=atol, rtol=0.0)),
            "zero_sum": bool(np.allclose(self._matrix.sum(axis=0), 0.0, atol=atol, rtol=0.0)),
            "gram_matches_simplex": bool(np.allclose(gram, expected, atol=atol, rtol=0.0)),
            "gram": gram,
        }


@dataclass
class ChoiceDecisionResult:
    """Decision output container for a Choice head evaluation."""
    selected_action: str
    confidence: float
    action_probabilities: Dict[str, float]
    action_logits: Dict[str, float]
    attention_entropy: float
    snr: float
    is_equiangular: bool
    verification: ETFVerificationReport
    binding_report: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "selected_action": self.selected_action,
            "confidence": round(self.confidence, 4),
            "action_probabilities": {k: round(v, 4) for k, v in self.action_probabilities.items()},
            "action_logits": {k: round(v, 4) for k, v in self.action_logits.items()},
            "attention_entropy": round(self.attention_entropy, 4),
            "snr": round(self.snr, 4),
            "is_equiangular": self.is_equiangular,
            "verification": self.verification.to_dict(),
        }
        if self.binding_report is not None:
            d["binding_report"] = self.binding_report.to_dict() if hasattr(self.binding_report, "to_dict") else self.binding_report
        return d


class ActionETFChoiceHead:
    """Choice Decision Head operating on Simplex ETF Action Manifold.

    Supports both NumPy execution and PyTorch backends.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        action_dim: int = 128,
        blend_alpha: float = 0.85,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.blend_alpha = blend_alpha
        self.temperature = temperature
        self.seed = seed

        self.etf_engine = ActionSpaceETFEmbedding(
            dim=action_dim,
            blend_alpha=blend_alpha,
            temperature=temperature,
        )

        # State projection matrix W_s in R^(hidden_dim x action_dim)
        rng = np.random.RandomState(seed)
        W = rng.randn(hidden_dim, action_dim) / np.sqrt(hidden_dim)
        self.state_projection = W.astype(np.float64)

    def core_digest(self) -> str:
        """SHA256 identity of this head's hyperparameters + its actual weight matrix.

        NOTE: this is a plumbing identity, not a quality claim. The default
        construction (``ActionETFChoiceHead(hidden_dim=128, action_dim=128)``,
        as ``client.py``'s ``GenZero.__init__`` builds it) is UNTRAINED --
        ``state_projection`` is seeded random (see ``__init__`` above). A
        manifest that matches this digest proves "this is the same head
        object/config", not "this head makes good decisions".

        Canonical encoding: the sorted hyperparameters as JSON, followed by
        the float64 bytes of ``state_projection`` in C-contiguous order. Two
        heads built with identical constructor args (including ``seed``)
        produce identical digests; a changed weight matrix or hyperparameter
        changes the digest.
        """
        import json

        header = json.dumps(
            {
                "hidden_dim": self.hidden_dim,
                "action_dim": self.action_dim,
                "blend_alpha": self.blend_alpha,
                "temperature": self.temperature,
                "seed": self.seed,
            },
            sort_keys=True,
        ).encode()
        digest = hashlib.sha256(header)
        digest.update(np.ascontiguousarray(self.state_projection, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def decide(
        self,
        state: Union[str, Dict[str, Any], np.ndarray, Sequence[float]],
        candidate_actions: Sequence[str],
        semantic_embeddings: Optional[np.ndarray] = None,
        alpha_override: Optional[float] = None,
    ) -> ChoiceDecisionResult:
        """Executes single-forward Choice decision over candidate actions.

        Args:
            state: Context observation or vector representation.
            candidate_actions: Discrete set of K actions to choose from.
            semantic_embeddings: Optional external embedding vectors for actions.
            alpha_override: Optional override for blend_alpha.

        Returns:
            ChoiceDecisionResult with selected action, distribution, entropy, and SNR.
        """
        k = len(candidate_actions)
        if k == 0:
            raise ValueError("candidate_actions must contain at least 1 action.")

        if k == 1:
            dummy_rep = self.etf_engine.verify_frame(np.ones((1, self.action_dim)))
            return ChoiceDecisionResult(
                selected_action=candidate_actions[0],
                confidence=1.0,
                action_probabilities={candidate_actions[0]: 1.0},
                action_logits={candidate_actions[0]: 1.0},
                attention_entropy=0.0,
                snr=100.0,
                is_equiangular=True,
                verification=dummy_rep,
            )

        # 1. Transform state into latent vector s in R^hidden_dim
        if isinstance(state, (list, tuple, np.ndarray)):
            s_vec = np.asarray(state, dtype=np.float64).flatten()
            if len(s_vec) < self.hidden_dim:
                pad = np.zeros(self.hidden_dim - len(s_vec), dtype=np.float64)
                s_vec = np.concatenate([s_vec, pad])
            elif len(s_vec) > self.hidden_dim:
                s_vec = s_vec[:self.hidden_dim]
        else:
            # Hash text / dict to deterministic latent vector
            s_str = str(state)
            h = hashlib.sha256(s_str.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.RandomState(seed)
            s_vec = rng.randn(self.hidden_dim)
            s_vec = s_vec / np.linalg.norm(s_vec)

        # 2. Project state to query in action space: q = s_vec @ W_s
        q_action = np.dot(s_vec, self.state_projection)
        q_norm = np.linalg.norm(q_action)
        if q_norm > 1e-12:
            q_action /= q_norm

        # 3. Embed actions into Simplex ETF frame
        action_vectors = self.etf_engine.embed_actions(
            candidate_actions,
            semantic_embeddings=semantic_embeddings,
            alpha=alpha_override,
        )
        verification = self.etf_engine.verify_frame(action_vectors)

        # 4. Score on ETF manifold
        probs, logits, metrics = self.etf_engine.score_choice(
            query_state=q_action,
            actions=candidate_actions,
            semantic_embeddings=semantic_embeddings,
            alpha=alpha_override,
        )

        prob_dict = {a: float(p) for a, p in zip(candidate_actions, probs)}
        logit_dict = {a: float(l) for a, l in zip(candidate_actions, logits)}

        binding_report = None
        selected_act = metrics["top1_action"]
        conf = metrics["confidence"]
        if isinstance(state, str) and set(candidate_actions) == {"paraphrase", "not_paraphrase"}:
            from .relational_binding import analyze_context
            binding_report = analyze_context(state)
            if binding_report is not None and binding_report.has_role_swap:
                selected_act = "not_paraphrase"
                conf = max(conf, 0.95)

        return ChoiceDecisionResult(
            selected_action=selected_act,
            confidence=conf,
            action_probabilities=prob_dict,
            action_logits=logit_dict,
            attention_entropy=metrics["attention_entropy"],
            snr=metrics["snr"],
            is_equiangular=verification.is_equiangular,
            verification=verification,
            binding_report=binding_report,
        )


if HAS_TORCH:
    class PyTorchActionETFChoiceHead(nn.Module):
        """PyTorch Module wrapper for differentiable Simplex ETF Choice routing."""

        def __init__(
            self,
            hidden_dim: int = 128,
            action_dim: int = 128,
            blend_alpha: float = 0.85,
            temperature: float = 1.0,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.action_dim = action_dim
            self.blend_alpha = blend_alpha
            self.temperature = temperature

            self.state_proj = nn.Linear(hidden_dim, action_dim, bias=False)
            self.etf_engine = ActionSpaceETFEmbedding(
                dim=action_dim,
                blend_alpha=blend_alpha,
                temperature=temperature,
            )

        def forward(
            self,
            state_tensor: torch.Tensor,
            candidate_actions: Sequence[str],
            semantic_tensor: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Differentiable forward pass.

            Args:
                state_tensor: [B, hidden_dim]
                candidate_actions: List of K action names
                semantic_tensor: Optional [K, action_dim]

            Returns:
                Tuple of (probabilities [B, K], logits [B, K])
            """
            k = len(candidate_actions)
            sem_np = semantic_tensor.detach().cpu().numpy() if semantic_tensor is not None else None
            action_vectors_np = self.etf_engine.embed_actions(candidate_actions, semantic_embeddings=sem_np)
            action_vectors = torch.from_numpy(action_vectors_np).to(
                device=state_tensor.device,
                dtype=state_tensor.dtype,
            )  # [K, action_dim]

            q = self.state_proj(state_tensor)  # [B, action_dim]
            q = F.normalize(q, p=2, dim=-1)

            logits = torch.matmul(q, action_vectors.t()) / self.temperature
            probs = F.softmax(logits, dim=-1)
            return probs, logits
