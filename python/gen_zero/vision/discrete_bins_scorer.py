"""Discrete Bins Expectation Scorer for Continuous Metrics & Confidence Variance.

Implements Milestone 3 of Issue #24:
- Maps continuous scores / priority / risk metrics to M=9 discrete token bins ("1" to "9").
- Computes expected value E[S] = sum_{i=1}^M i * P(i).
- Linearly normalizes to [0.01, 0.99]: NormScore = 0.01 + 0.98 * (E[S] - 1) / (M - 1).
- Evaluates Uncertainty Variance = sum_{i=1}^M (i - E[S])^2 * P(i) as a calibrated confidence estimator.
"""

from typing import Dict, List, Any, Optional, Tuple
import dataclasses
import math
import numbers


@dataclasses.dataclass
class DiscreteBinsVerdict:
    expected_bin: float
    normalized_score: float
    uncertainty_variance: float
    bin_probabilities: Dict[str, float]
    confidence: float
    is_confident: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_bin": round(self.expected_bin, 4),
            "normalized_score": round(self.normalized_score, 4),
            "uncertainty_variance": round(self.uncertainty_variance, 4),
            "bin_probabilities": {k: round(v, 4) for k, v in self.bin_probabilities.items()},
            "confidence": round(self.confidence, 4),
            "is_confident": self.is_confident,
        }


class DiscreteBinsExpectationScorer:
    """Computes continuous metric expectation and uncertainty variance over discrete token bins."""

    def __init__(self, num_bins: int = 9, max_variance_threshold: float = 2.5):
        if isinstance(num_bins, bool) or not isinstance(num_bins, numbers.Integral) or num_bins < 1:
            raise ValueError("num_bins must be a positive integer.")
        if not math.isfinite(max_variance_threshold) or max_variance_threshold < 0:
            raise ValueError("max_variance_threshold must be finite and nonnegative.")
        self.num_bins = num_bins
        self.max_variance_threshold = max_variance_threshold
        self.bin_labels = [str(i) for i in range(1, num_bins + 1)]

    def compute_expectation_and_variance(
        self,
        bin_probabilities: Dict[str, float],
    ) -> DiscreteBinsVerdict:
        """Evaluates expected score and uncertainty variance given discrete bin probabilities."""
        M = float(self.num_bins)

        # Extract normalized probability vector
        raw_probs = [float(bin_probabilities.get(str(i), 0.0)) for i in range(1, self.num_bins + 1)]
        if not all(math.isfinite(p) for p in raw_probs):
            raise ValueError("Bin probabilities must be finite.")
        raw_probs = [max(0.0, p) for p in raw_probs]
        scale = max(raw_probs)
        if scale <= 0.0:
            # No bin carries any mass, so there is no supported expectation or
            # confident prediction to report.  Keep the vector shape stable for
            # callers while explicitly failing closed on confidence.
            return DiscreteBinsVerdict(
                expected_bin=0.0,
                normalized_score=0.01,
                uncertainty_variance=0.0,
                bin_probabilities={str(i): 0.0 for i in range(1, self.num_bins + 1)},
                confidence=0.0,
                is_confident=False,
            )
        # Scaling first avoids overflow when several finite weights approach float max.
        scaled_probs = [p / scale for p in raw_probs]
        sum_p = math.fsum(scaled_probs)
        probs = [p / sum_p for p in scaled_probs]

        # 1. Expected bin value E[S] = sum_{i=1}^M i * P(i)
        expected_s = sum(i * p for i, p in zip(range(1, self.num_bins + 1), probs))

        # 2. Normalized Score in [0.01, 0.99]
        if M > 1:
            norm_score = 0.01 + 0.98 * ((expected_s - 1.0) / (M - 1.0))
        else:
            norm_score = 0.50
        norm_score = max(0.01, min(0.99, float(norm_score)))

        # 3. Uncertainty Variance = sum_{i=1}^M (i - E[S])^2 * P(i)
        var_s = sum(((i - expected_s) ** 2) * p for i, p in zip(range(1, self.num_bins + 1), probs))

        # 4. Calibrated Confidence inversely related to variance
        # Standard deviation std = sqrt(var_s). When concentrated on 1 bin, var=0 -> conf=1.0.
        std_s = math.sqrt(max(0.0, var_s))
        # Max theoretical std for M=9 uniform is ~2.58
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + std_s)))
        is_confident = (var_s <= self.max_variance_threshold)

        prob_dict = {str(i): float(p) for i, p in zip(range(1, self.num_bins + 1), probs)}

        return DiscreteBinsVerdict(
            expected_bin=expected_s,
            normalized_score=norm_score,
            uncertainty_variance=var_s,
            bin_probabilities=prob_dict,
            confidence=confidence,
            is_confident=is_confident,
        )

    def continuous_to_target_bin(self, continuous_val: float) -> str:
        """Quantizes continuous scalar in [0.0, 1.0] into discrete bin token ("1".."9")."""
        continuous_val = float(continuous_val)
        if not math.isfinite(continuous_val):
            raise ValueError("Continuous value must be finite.")
        clipped = max(0.0, min(1.0, continuous_val))
        idx = int(round(clipped * (self.num_bins - 1))) + 1
        idx = max(1, min(self.num_bins, idx))
        return str(idx)
