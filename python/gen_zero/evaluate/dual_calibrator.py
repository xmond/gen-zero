"""Dual-Stage Calibrator (Temperature Scaling + Isotonic Regression) for NanoCore.

Implements Milestone 2 of Issue #23:
- Dual-Stage Calibration Pipeline:
  1. Parametric Temperature Scaling: z / T smoothing uncalibrated raw logits.
  2. Non-Parametric Isotonic Regression: Pool-Adjacent Violators Algorithm (PAVA) monotonic mapping.
- Calibration helpers for caller-supplied near-miss/OOD samples.
- 10-Bin ECE calculation.  The caller is responsible for evaluating on a
  held-out split; fitting and scoring on the same observations is optimistic.
"""

from typing import List, Dict, Any, Optional, Tuple
import dataclasses
import numpy as np


@dataclasses.dataclass
class CalibrationSample:
    sample_id: str
    domain: str
    raw_confidence: float
    true_label: int  # 1 if correct, 0 if near-miss / incorrect
    is_near_miss: bool = False


class IsotonicCalibrator:
    """Non-parametric monotonic probability calibration using Pool-Adjacent Violators Algorithm (PAVA)."""

    def __init__(self):
        self.x_thresholds: np.ndarray = np.array([0.0, 1.0])
        self.y_values: np.ndarray = np.array([0.0, 1.0])

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        """Fits non-parametric isotonic regression on (confidence, label) pairs."""
        x = np.asarray(confidences, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("confidences and labels must be one-dimensional")
        if len(x) == 0 or len(x) != len(y):
            raise ValueError(
                "isotonic regression requires equally sized, non-empty confidences and labels"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("confidences and labels must contain only finite values")
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("confidences must lie in [0, 1]")
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("labels must lie in [0, 1]")

        # Sort by x
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # Equal confidence values are one calibration point.  Keeping several
        # blocks at an identical x makes interpolation depend on sort order,
        # which can turn a tie into an arbitrary confidence jump.
        unique_x, inverse, counts = np.unique(
            x_sorted, return_inverse=True, return_counts=True
        )
        unique_y = np.zeros_like(unique_x, dtype=np.float32)
        for idx in range(len(unique_x)):
            unique_y[idx] = float(np.mean(y_sorted[inverse == idx]))

        # Simple PAVA implementation
        # Initialize blocks
        blocks = [
            [float(unique_y[i]), int(counts[i]), float(unique_x[i])]
            for i in range(len(unique_x))
        ]

        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                # Pool adjacent violators
                w1 = blocks[i][1]
                w2 = blocks[i + 1][1]
                v1 = blocks[i][0]
                v2 = blocks[i + 1][0]
                new_v = (w1 * v1 + w2 * v2) / (w1 + w2)
                new_w = w1 + w2
                new_x = (w1 * blocks[i][2] + w2 * blocks[i + 1][2]) / (w1 + w2)

                blocks[i] = [new_v, new_w, new_x]
                del blocks[i + 1]

                # Backtrack if needed
                if i > 0:
                    i -= 1
            else:
                i += 1

        self.x_thresholds = np.array([b[2] for b in blocks], dtype=np.float32)
        self.y_values = np.array([b[0] for b in blocks], dtype=np.float32)

        # Ensure boundaries [0.0, 1.0]
        if self.x_thresholds[0] > 0.0:
            self.x_thresholds = np.insert(self.x_thresholds, 0, 0.0)
            self.y_values = np.insert(self.y_values, 0, self.y_values[0])
        if self.x_thresholds[-1] < 1.0:
            self.x_thresholds = np.append(self.x_thresholds, 1.0)
            self.y_values = np.append(self.y_values, self.y_values[-1])

        return self

    def predict(self, confidences: np.ndarray) -> np.ndarray:
        """Applies monotonic calibrated mapping."""
        x = np.asarray(confidences, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("confidences must contain only finite values")
        # Monotonic piecewise linear interpolation
        return np.interp(x, self.x_thresholds, self.y_values)


class DualCalibrator:
    """Combines parametric Temperature Scaling with non-parametric Isotonic Regression."""

    def __init__(self, temperature: float = 1.25):
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")
        self.temperature = max(0.1, temperature)
        self.isotonic = IsotonicCalibrator()
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "DualCalibrator":
        """Calibrates temperature-scaled probabilities via Isotonic PAVA."""
        probs = np.asarray(raw_probs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if probs.ndim != 1 or labels.ndim != 1:
            raise ValueError("raw_probs and labels must be one-dimensional")
        if len(probs) == 0 or len(probs) != len(labels):
            raise ValueError("raw_probs and labels must be equally sized and non-empty")
        if not np.isfinite(probs).all() or not np.isfinite(labels).all():
            raise ValueError("raw_probs and labels must contain only finite values")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("raw_probs must lie in [0, 1]")
        if np.any((labels < 0.0) | (labels > 1.0)):
            raise ValueError("labels must lie in [0, 1]")
        # Stage 1: Temperature scaling in logit space
        eps = 1e-7
        clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped))
        temp_scaled_probs = 1.0 / (1.0 + np.exp(-logits / self.temperature))

        # Stage 2: Fit Isotonic Regression
        self.isotonic.fit(temp_scaled_probs, labels)
        self.is_fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrates inputs through temperature scaling and isotonic projection."""
        probs = np.asarray(raw_probs, dtype=np.float32)
        if probs.ndim != 1:
            raise ValueError("raw_probs must be one-dimensional")
        if not np.isfinite(probs).all():
            raise ValueError("raw_probs must contain only finite values")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("raw_probs must lie in [0, 1]")
        eps = 1e-7
        clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped))
        temp_scaled = 1.0 / (1.0 + np.exp(-logits / self.temperature))

        if not self.is_fitted:
            return temp_scaled

        calibrated = self.isotonic.predict(temp_scaled)
        return np.clip(calibrated, 0.0, 1.0)

    @staticmethod
    def compute_10bin_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
        """Computes 10-Bin Expected Calibration Error (ECE)."""
        p = np.asarray(probs, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        if p.ndim != 1 or y.ndim != 1:
            raise ValueError("probs and labels must be one-dimensional")
        if len(p) != len(y):
            raise ValueError("probs and labels must have equal length")
        if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins <= 0:
            raise ValueError("num_bins must be a positive integer")
        n = len(p)
        if n == 0:
            raise ValueError("ECE requires at least one prediction")
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError("probs and labels must contain only finite values")
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("probs must lie in [0, 1]")
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("labels must lie in [0, 1]")

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (p >= bin_lower) & (p < bin_upper if i < num_bins - 1 else p <= bin_upper)
            bin_count = int(np.sum(in_bin))

            if bin_count > 0:
                bin_acc = float(np.mean(y[in_bin]))
                bin_conf = float(np.mean(p[in_bin]))
                ece += (bin_count / n) * abs(bin_acc - bin_conf)

        return float(ece)
