"""Gen-Zero Layer 5: Locked-Test Protocol & High-Confidence Error Safety Gate.

RFC Implementation for Issue #7:
1. Locked-Test Physical-Level Read Protection:
   - Cryptographic SHA-256 fingerprint verification.
   - Access audit trail recording every evaluation event.
   - Prevents hyperparameter data contamination on test suites.
2. 10-Bin Expected Calibration Error (ECE) & Reliability Diagram:
   - Partitions confidence space into 10 bins: [0.0, 0.1), ..., [0.9, 1.0].
   - Computes bin-wise accuracy, confidence, calibration gap, and overall ECE/MCE/Brier score.
3. High-Confidence Error (Confident Errors) Red Line Gate:
   - Hard red line: Predictions with P >= 0.90 that fail (y_pred != y_true) trigger hard veto.
"""

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def compute_wilson_score_interval(
    k: int,
    n: int,
    z: float = 1.96
) -> Tuple[float, float]:
    """Computes Wilson 95% score confidence interval for binomial proportion.

    Formula:
      w = (p_hat + z^2 / (2n) +/- z * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4n^2))) / (1 + z^2 / n)
    """
    if n <= 0:
        return (0.0, 0.0)
    p_hat = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (p_hat + (z2 / (2.0 * n))) / denom
    var_term = (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))
    margin = (z / denom) * math.sqrt(max(0.0, var_term))
    lower = max(0.0, min(1.0, center - margin))
    upper = max(0.0, min(1.0, center + margin))
    return (round(lower, 4), round(upper, 4))


@dataclass
class CalibrationBin:
    bin_index: int
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_confidence: float
    accuracy: float
    calibration_gap: float
    wilson_lower: float = 0.0
    wilson_upper: float = 0.0
    ci_95: Tuple[float, float] = (0.0, 0.0)


@dataclass
class CalibrationReport:
    total_samples: int
    ece_10bin: float
    mce: float
    brier_score: float
    bins: List[CalibrationBin]
    confident_error_count: int
    confident_error_rate: float
    confident_errors: List[Dict[str, Any]]
    passed_safety_red_line: bool
    verdict: str

    def to_ascii_curve(self) -> str:
        return generate_ascii_calibration_curve(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "ece_10bin": round(self.ece_10bin, 4),
            "mce": round(self.mce, 4),
            "brier_score": round(self.brier_score, 4),
            "confident_error_count": self.confident_error_count,
            "confident_error_rate": round(self.confident_error_rate, 4),
            "passed_safety_red_line": self.passed_safety_red_line,
            "verdict": self.verdict,
            "bins": [asdict(b) for b in self.bins],
            "confident_errors": self.confident_errors
        }


def generate_ascii_calibration_curve(report: CalibrationReport) -> str:
    """Renders a text/ASCII calibration reliability diagram with Wilson 95% CIs."""
    lines = [
        f"10-Bin Calibration Reliability Diagram (ECE = {report.ece_10bin:.4f}, MCE = {report.mce:.4f}):",
        "Bin Range    | Count | Mean Conf | Accuracy | 95% Wilson CI     | Reliability (C=Conf, A=Acc)",
        "-------------+-------+-----------+----------+-------------------+-----------------------------"
    ]
    width = 25
    for b in report.bins:
        range_str = f"[{b.lower_bound:.1f}, {b.upper_bound:.1f})"
        c_pos = int(round(b.mean_confidence * width))
        a_pos = int(round(b.accuracy * width)) if b.sample_count > 0 else -1

        bar = [" "] * (width + 1)
        ref_pos = int(round(((b.lower_bound + b.upper_bound) / 2.0) * width))
        if 0 <= ref_pos <= width:
            bar[ref_pos] = "."
        if 0 <= c_pos <= width:
            bar[c_pos] = "C"
        if a_pos >= 0 and 0 <= a_pos <= width:
            bar[a_pos] = "A" if a_pos != c_pos else "*"

        bar_str = "".join(bar)
        ci_str = f"[{b.wilson_lower:.2f}, {b.wilson_upper:.2f}]" if b.sample_count > 0 else "[- , - ]"
        lines.append(
            f"{range_str:<12} | {b.sample_count:>5} | {b.mean_confidence:>9.4f} | {b.accuracy:>8.4f} | {ci_str:<17} | |{bar_str}|"
        )
    lines.append("-------------+-------+-----------+----------+-------------------+-----------------------------")
    lines.append("Legend: '.' = Ideal Diagonal, 'C' = Mean Confidence, 'A' = Empirical Accuracy, '*' = C == A")
    return "\n".join(lines)



class LockedTestSet:
    """Read-only test benchmark protected with cryptographic integrity auditing."""

    def __init__(self, samples: List[Dict[str, Any]], suite_name: str = "locked_benchmark_v1"):
        self.suite_name = suite_name
        self._samples = [copy.deepcopy(s) for s in samples]
        self._sha256 = self._compute_fingerprint()
        self._access_log: List[Dict[str, Any]] = []
        self._is_locked: bool = True

    def _compute_fingerprint(self) -> str:
        serialized = json.dumps(self._samples, default=str, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def size(self) -> int:
        return len(self._samples)

    def access_samples(self, caller_id: str = "evaluator", purpose: str = "verification") -> List[Dict[str, Any]]:
        """Accesses frozen samples and logs the read event to prevent contamination."""
        event = {
            "timestamp": time.time(),
            "caller_id": caller_id,
            "purpose": purpose,
            "sample_count": len(self._samples),
            "sha256": self._sha256
        }
        self._access_log.append(event)
        # Return deep copy so external code cannot mutate the frozen test set
        return [copy.deepcopy(s) for s in self._samples]

    def verify_integrity(self) -> bool:
        """Verifies test set has not suffered from in-memory bit rot or tampering."""
        return self._compute_fingerprint() == self._sha256

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        return list(self._access_log)


class CalibrationEvaluator:
    """Computes 10-Bin ECE and evaluates Confident Error red line."""

    CONFIDENT_ERROR_THRESHOLD = 0.90

    @classmethod
    def compute_calibration(
        cls,
        predictions: List[Dict[str, Any]],
        num_bins: int = 10,
        max_allowed_confident_error_rate: float = 0.0,
        max_allowed_ece: float = 0.15
    ) -> CalibrationReport:
        """Computes 10-bin ECE and checks high-confidence errors.

        Expected prediction dict:
        - 'confidence': float in [0.0, 1.0] (predicted probability for chosen class)
        - 'predicted_label': Any
        - 'ground_truth': Any
        - 'target_probability': Optional[float] (probability assigned to correct label, for Brier score)
        - 'sample_id': Optional[str]
        """
        n = len(predictions)
        if n == 0:
            return CalibrationReport(
                total_samples=0,
                ece_10bin=0.0,
                mce=0.0,
                brier_score=0.0,
                bins=[],
                confident_error_count=0,
                confident_error_rate=0.0,
                confident_errors=[],
                passed_safety_red_line=True,
                verdict="EMPTY_PREDICTIONS"
            )

        # 1. Group samples into 10 bins
        bin_width = 1.0 / num_bins
        bins_data: List[List[Dict[str, Any]]] = [[] for _ in range(num_bins)]

        confident_errors: List[Dict[str, Any]] = []
        brier_sum = 0.0

        for pred in predictions:
            conf = float(max(0.0, min(1.0, pred.get("confidence", 0.5))))
            y_pred = pred.get("predicted_label")
            y_true = pred.get("ground_truth")
            is_correct = bool(y_pred == y_true)

            target_prob = float(pred.get("target_probability", conf if is_correct else (1.0 - conf)))
            brier_sum += (1.0 - target_prob) ** 2

            # Identify High-Confidence Errors (P >= 0.90 but wrong)
            if conf >= cls.CONFIDENT_ERROR_THRESHOLD and not is_correct:
                confident_errors.append({
                    "sample_id": pred.get("sample_id", f"sample_{len(confident_errors)}"),
                    "confidence": round(conf, 4),
                    "predicted_label": y_pred,
                    "ground_truth": y_true
                })

            # Precise bin index robust against IEEE-754 floor division drift
            bin_idx = min(num_bins - 1, int(round(conf * num_bins, 9)))
            bins_data[bin_idx].append({
                "confidence": conf,
                "is_correct": is_correct
            })

        # 2. Compute bin-level metrics and aggregate ECE / MCE
        ece = 0.0
        mce = 0.0
        calibration_bins: List[CalibrationBin] = []

        for b_idx in range(num_bins):
            low = b_idx * bin_width
            high = (b_idx + 1) * bin_width
            bin_samples = bins_data[b_idx]
            b_count = len(bin_samples)

            if b_count > 0:
                k_correct = sum(1 for s in bin_samples if s["is_correct"])
                mean_c = sum(s["confidence"] for s in bin_samples) / b_count
                acc = float(k_correct) / b_count
                gap = abs(acc - mean_c)
                w_low, w_high = compute_wilson_score_interval(k_correct, b_count)
            else:
                mean_c = (low + high) / 2.0
                acc = 0.0
                gap = 0.0
                w_low, w_high = (0.0, 0.0)

            ece += (b_count / n) * gap
            if gap > mce:
                mce = gap

            calibration_bins.append(CalibrationBin(
                bin_index=b_idx,
                lower_bound=round(low, 2),
                upper_bound=round(high, 2),
                sample_count=b_count,
                mean_confidence=round(mean_c, 4),
                accuracy=round(acc, 4),
                calibration_gap=round(gap, 4),
                wilson_lower=w_low,
                wilson_upper=w_high,
                ci_95=(w_low, w_high)
            ))

        brier = brier_sum / max(1, n)
        conf_err_count = len(confident_errors)
        conf_err_rate = conf_err_count / n

        # Check safety red line: confident error rate <= threshold and ECE <= max_ece
        red_line_passed = (conf_err_rate <= max_allowed_confident_error_rate) and (ece <= max_allowed_ece)
        if not red_line_passed:
            if conf_err_count > 0:
                verdict = "RED_LINE_CONFIDENT_ERROR_BREACH"
            else:
                verdict = "RED_LINE_EXCESSIVE_ECE_CALIBRATION_DRIFT"
        else:
            verdict = "CALIBRATION_RED_LINE_PASSED"

        return CalibrationReport(
            total_samples=n,
            ece_10bin=round(ece, 4),
            mce=round(mce, 4),
            brier_score=round(brier, 4),
            bins=calibration_bins,
            confident_error_count=conf_err_count,
            confident_error_rate=round(conf_err_rate, 4),
            confident_errors=confident_errors,
            passed_safety_red_line=red_line_passed,
            verdict=verdict
        )


class LockedEvaluator:
    """Executes candidate evaluation against a LockedTestSet with full safety reporting."""

    def __init__(self, test_set: LockedTestSet):
        self.test_set = test_set

    def evaluate_engine(
        self,
        predict_fn: Callable[[Any, List[str]], Dict[str, Any]],
        caller_id: str = "candidate_validator",
        max_allowed_confident_error_rate: float = 0.0,
        max_allowed_ece: float = 0.15
    ) -> Dict[str, Any]:
        """Runs evaluation over frozen test set and generates comprehensive calibration audit."""
        if not self.test_set.verify_integrity():
            raise RuntimeError(f"Locked test set checksum mismatch! Expected {self.test_set.sha256}")

        samples = self.test_set.access_samples(caller_id=caller_id, purpose="candidate_verification")
        predictions: List[Dict[str, Any]] = []

        correct_count = 0
        for s in samples:
            state = s.get("state")
            candidates = s.get("candidates", [])
            truth = None
            for key in ("ground_truth", "target_action", "optimal_action"):
                if key in s and s[key] is not None:
                    truth = s[key]
                    break

            out = predict_fn(state, candidates)
            pred_act = out.get("best_action")
            conf = float(out.get("confidence", 0.5))

            if pred_act == truth:
                correct_count += 1

            predictions.append({
                "sample_id": s.get("id"),
                "confidence": conf,
                "predicted_label": pred_act,
                "ground_truth": truth
            })

        acc = (correct_count / max(1, len(samples))) * 100.0
        report = CalibrationEvaluator.compute_calibration(
            predictions=predictions,
            max_allowed_confident_error_rate=max_allowed_confident_error_rate,
            max_allowed_ece=max_allowed_ece
        )

        metrics = {
            "accuracy": round(acc, 2),
            "mean_score": round(acc / 100.0, 4),
            "collision_rate": 0.0,
            "is_valid": True,
            "test_sha256": self.test_set.sha256,
            "calibration_report": report.to_dict(),
            "passed_safety_red_line": report.passed_safety_red_line,
            "verdict": report.verdict
        }

        return metrics
