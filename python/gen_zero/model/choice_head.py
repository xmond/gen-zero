"""Gen-Zero Layer 1: Choice Head Module & Ordinal Cumulative Head.

RFC-069 & Issue #72 Implementation:
Re-exports Choice Decision Head with Simplex ETF Action Manifold, Contrastive Verification,
and Temperature Calibration from gen_zero.nanocore.choice_head.

Includes OrdinalCumulativeHead:
Implements Proportional Odds monotonic cumulative logits model over continuous ordinal manifolds
(e.g. SummEval 1-5 quality evaluation), eliminating the categorical entropy explosion (1.56 bits, 27/30 STOP)
by respecting the continuous metric topology |1 - 2| < |1 - 5|.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from gen_zero.nanocore.choice_head import (
    ActionETFChoiceHead,
    ChoiceDecisionResult,
    FastSimplexETFProjection,
    calibrate_temperature_and_entropy,
)

try:
    from gen_zero.nanocore.choice_head import PyTorchActionETFChoiceHead
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    PyTorchActionETFChoiceHead = None
    torch = None
    nn = object
    F = None
    HAS_TORCH = False



@dataclass
class OrdinalEvaluationResult:
    """Result of ordinal cumulative evaluation under Proportional Odds model."""
    probabilities: List[float]
    cumulative_probs: List[float]
    expected_score: float
    quantized_decision: int
    ordinal_variance: float
    normalized_entropy: float
    categorical_entropy: float
    wasserstein_loss: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probabilities": [round(p, 4) for p in self.probabilities],
            "cumulative_probs": [round(p, 4) for p in self.cumulative_probs],
            "expected_score": round(self.expected_score, 3),
            "quantized_decision": self.quantized_decision,
            "ordinal_variance": round(self.ordinal_variance, 4),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "categorical_entropy": round(self.categorical_entropy, 4),
            "wasserstein_loss": round(self.wasserstein_loss, 4),
        }


class OrdinalCumulativeHead:
    """Ordinal Cumulative Decision Head using Proportional Odds Monotonic Logits.

    Eliminates the categorical entropy explosion on continuous rating benchmarks (e.g. SummEval)
    by respecting the metric topology |1 - 2| < |1 - 5|:
      P(Y > k | h) = \sigma(w^T h - \theta_k),  \theta_1 < \theta_2 < \theta_3 < \theta_4
    Discrete probabilities:
      P(Y = 1) = 1 - P(Y > 1)
      P(Y = k) = P(Y > k-1) - P(Y > k) for 1 < k < K
      P(Y = K) = P(Y > K-1)
    Expected score:
      E[Y] = 1 + \sum_{k=1}^{K-1} P(Y > k)
    Quantized decision:
      \hat{y} = \arg\max_{k \in 1..=K} P(Y = k)
    """

    def __init__(
        self,
        dimension: int = 1024,
        num_classes: int = 5,
        thresholds: Optional[Sequence[float]] = None,
        weights: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        alpha_penalty: float = 0.5,
    ):
        self.dimension = dimension
        self.num_classes = num_classes
        self.temperature = max(1e-6, float(temperature))
        self.alpha_penalty = float(alpha_penalty)

        if thresholds is not None:
            if len(thresholds) != num_classes - 1:
                raise ValueError(f"Expected {num_classes - 1} thresholds, got {len(thresholds)}")
            # Enforce strictly monotonic cutoffs
            self.thresholds = np.array(sorted(thresholds), dtype=np.float64)
        else:
            # Default symmetrically spaced cutpoints around 0.0: [-1.5, -0.5, 0.5, 1.5] for K=5
            half = (num_classes - 1) / 2.0
            self.thresholds = np.array(
                [float(k) - half + 0.5 for k in range(num_classes - 1)],
                dtype=np.float64,
            )

        if weights is not None:
            self.weights = np.array(weights, dtype=np.float64)
            if self.weights.shape[0] != dimension:
                raise ValueError(f"Weight dimension mismatch: expected {dimension}, got {self.weights.shape[0]}")
        else:
            self.weights = np.zeros(dimension, dtype=np.float64)
            self.weights[0] = 1.0  # Canonical probe vector

    def evaluate_scalar(self, s: float) -> OrdinalEvaluationResult:
        """Evaluates directly from latent scalar projection s = w^T h."""
        k_thresh = len(self.thresholds)
        k_classes = self.num_classes

        # 1. Monotonic cumulative logits: P(Y > k | h) = \sigma((s - \theta_k) / T)
        cum_probs = []
        for theta in self.thresholds:
            logit = (s - theta) / self.temperature
            # Stable sigmoid
            if logit >= 0:
                p = 1.0 / (1.0 + math.exp(-logit))
            else:
                z = math.exp(logit)
                p = z / (1.0 + z)
            cum_probs.append(p)

        # Enforce numerical monotonicity
        for i in range(1, k_thresh):
            if cum_probs[i] > cum_probs[i - 1]:
                cum_probs[i] = cum_probs[i - 1]

        # 2. Discrete probabilities
        probs = [0.0] * k_classes
        probs[0] = max(1e-12, 1.0 - cum_probs[0])
        for k in range(1, k_thresh):
            probs[k] = max(1e-12, cum_probs[k - 1] - cum_probs[k])
        probs[k_classes - 1] = max(1e-12, cum_probs[k_thresh - 1])

        # Normalize sum to 1.0
        sum_p = sum(probs)
        probs = [p / sum_p for p in probs]

        # 3. Expected continuous quality score: E[Y] = 1 + \sum P(Y > k)
        expected_score = 1.0 + sum(cum_probs)

        # 4. Quantized decision: \hat{y} = \arg\max P(Y = k) (1-indexed: 1..=K)
        quantized_decision = int(np.argmax(probs)) + 1

        # 5. Ordinal dispersion / variance: Var[Y] = \sum P(Y = k) * (k - E[Y])^2
        ordinal_variance = sum(
            p * ((idx + 1) - expected_score) ** 2
            for idx, p in enumerate(probs)
        )

        # 6. Categorical Shannon entropy
        categorical_entropy = -sum(p * math.log2(p + 1e-15) for p in probs if p > 0.0)

        # 7. Normalized metric entropy:
        # Respects metric topology |1 - 2| < |1 - 5|.
        max_sigma = (k_classes - 1) / 2.0
        normalized_entropy = min(1.0, max(0.0, math.sqrt(ordinal_variance) / max_sigma))

        return OrdinalEvaluationResult(
            probabilities=probs,
            cumulative_probs=cum_probs,
            expected_score=expected_score,
            quantized_decision=quantized_decision,
            ordinal_variance=ordinal_variance,
            normalized_entropy=normalized_entropy,
            categorical_entropy=categorical_entropy,
        )

    def evaluate(self, hidden_state: Union[np.ndarray, Sequence[float]]) -> OrdinalEvaluationResult:
        """Evaluates hidden representation h \in \mathbb{R}^D."""
        h = np.asarray(hidden_state, dtype=np.float64).ravel()
        n = min(len(self.weights), len(h))
        dot = float(np.dot(self.weights[:n], h[:n]))
        return self.evaluate_scalar(dot)

    def wasserstein_distance(self, probs: Sequence[float], target_y: int) -> float:
        """Earth Mover's Distance W_1(P, \delta_{y^*}) = |E[Y] - y^*|."""
        e_y = sum((idx + 1) * p for idx, p in enumerate(probs))
        return abs(e_y - float(target_y))

    def wasserstein_exponential_loss(self, probs: Sequence[float], target_y: int) -> float:
        """Exponential distance penalty loss: penalizes |1 - 5| exponentially harder than |4 - 5|."""
        y_star = float(target_y)
        loss = 0.0
        for idx, p in enumerate(probs):
            score = float(idx + 1)
            dist = abs(score - y_star)
            penalty = math.exp(self.alpha_penalty * dist) - 1.0
            loss += p * penalty
        return loss


if HAS_TORCH:
    class PyTorchOrdinalCumulativeHead(nn.Module):
        """PyTorch differentiable implementation of OrdinalCumulativeHead.

        Ensures strictly monotonic thresholds via cumulative sum of positive increments:
          \theta_k = \theta_1 + \sum_{j=2}^k \text{softplus}(\Delta_j).
        """

        def __init__(
            self,
            hidden_dim: int = 1024,
            num_classes: int = 5,
            temperature: float = 1.0,
            alpha_penalty: float = 0.5,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.temperature = temperature
            self.alpha_penalty = alpha_penalty

            # Linear projection: h -> scalar score s
            self.projection = nn.Linear(hidden_dim, 1, bias=False)

            # Monotonic cutpoints parametrization
            # threshold_0 is the base cutoff; threshold_deltas are positive step sizes
            self.base_threshold = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))
            # num_classes - 2 increments between the K-1 thresholds
            self.threshold_deltas = nn.Parameter(torch.ones(num_classes - 2, dtype=torch.float32) * 1.0)

        def get_thresholds(self) -> torch.Tensor:
            """Computes strictly monotonic thresholds \theta_1 < \theta_2 < ... < \theta_{K-1}."""
            increments = F.softplus(self.threshold_deltas)
            thresholds = torch.cat([
                self.base_threshold.unsqueeze(0),
                self.base_threshold + torch.cumsum(increments, dim=0),
            ])
            return thresholds

        def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Forward pass: (B, D) -> cumulative and discrete probabilities."""
            # Scalar latent projections: (B, 1)
            s = self.projection(h)
            thresholds = self.get_thresholds()  # (K - 1,)

            # Cumulative logits: (s - \theta_k) / T -> (B, K - 1)
            cum_logits = (s - thresholds.unsqueeze(0)) / self.temperature
            cum_probs = torch.sigmoid(cum_logits)  # P(Y > k | h)

            # Discrete probabilities: (B, K)
            b_size = h.size(0)
            p_first = 1.0 - cum_probs[:, 0:1]  # P(Y = 1)
            p_mid = cum_probs[:, :-1] - cum_probs[:, 1:]  # P(Y = k)
            p_last = cum_probs[:, -1:]  # P(Y = K)
            probs = torch.cat([p_first, p_mid, p_last], dim=-1)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

            # Expected score: E[Y] = 1 + \sum P(Y > k)
            expected_score = 1.0 + cum_probs.sum(dim=-1, keepdim=True)

            # Quantized decision: argmax P(Y = k) + 1
            quantized = torch.argmax(probs, dim=-1) + 1

            # Ordinal variance
            scores = torch.arange(1, self.num_classes + 1, device=h.device, dtype=torch.float32)
            diffs = scores.unsqueeze(0) - expected_score
            variance = torch.sum(probs * (diffs ** 2), dim=-1, keepdim=True)

            # Normalized metric entropy
            max_sigma = (self.num_classes - 1) / 2.0
            normalized_entropy = torch.clamp(torch.sqrt(variance) / max_sigma, 0.0, 1.0)

            return {
                "probabilities": probs,
                "cumulative_probs": cum_probs,
                "expected_score": expected_score,
                "quantized_decision": quantized,
                "ordinal_variance": variance,
                "normalized_entropy": normalized_entropy,
            }

        def compute_wasserstein_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """Wasserstein metric loss with exponential distance penalty.

            Args:
                probs: (B, K) predicted discrete probabilities
                targets: (B,) 1-indexed target labels in {1, ..., K}
            """
            b_size, k_classes = probs.shape
            scores = torch.arange(1, k_classes + 1, device=probs.device, dtype=torch.float32)  # (K,)
            # Distances: (B, K)
            distances = torch.abs(scores.unsqueeze(0) - targets.unsqueeze(1).float())
            penalties = torch.exp(self.alpha_penalty * distances) - 1.0
            loss = torch.sum(probs * penalties, dim=-1).mean()
            return loss
else:
    PyTorchOrdinalCumulativeHead = None


__all__ = [
    "ActionETFChoiceHead",
    "ChoiceDecisionResult",
    "FastSimplexETFProjection",
    "calibrate_temperature_and_entropy",
    "PyTorchActionETFChoiceHead",
    "OrdinalCumulativeHead",
    "OrdinalEvaluationResult",
    "PyTorchOrdinalCumulativeHead",
]
