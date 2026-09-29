"""Shuffled-State Control Benchmark & Causal Gain Ratio (CGR) Measurement.

Implements Milestone 1 of Issue #19:
1. Cyclic state perturbation across evaluation batches:
   Batch_normal   = {(s_i, A_i, y_i)}_{i=1}^B
   Batch_shuffled = {(s_{(i mod B) + 1}, A_i, y_i)}_{i=1}^B
2. Metrics:
   - Causal Accuracy Gap: CAG = Top1Acc(normal) - Top1Acc(shuffled)
   - Causal Gain Ratio:   CGR = Top1Acc(normal) / max(Top1Acc(shuffled), 1e-4)
   - 10-Bin ECE calculation for both normal and shuffled arms
3. Quality Gate Hard Red Line:
   - CAG >= 20% (0.20)
   - CGR >= 3.0x
   Models failing this gate are flagged for causal degradation and blocked from deployment.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import dataclasses
import copy
import math

from gen_zero.gate.locked_evaluator import CalibrationEvaluator, CalibrationReport


def compute_cag_and_cgr(normal_acc: float, shuffled_acc: float, eps: float = 1e-4) -> Tuple[float, float]:
    """Computes Causal Accuracy Gap (CAG) and Causal Gain Ratio (CGR).

    Args:
        normal_acc: Top-1 accuracy on natural state-action pairs in [0.0, 1.0].
        shuffled_acc: Top-1 accuracy on mismatched/shuffled state-action pairs in [0.0, 1.0].
        eps: Floor epsilon to prevent zero division.

    Returns:
        Tuple of (CAG, CGR).
    """
    cag = float(normal_acc - shuffled_acc)
    denom = max(float(shuffled_acc), eps)
    cgr = float(normal_acc / denom)
    return cag, cgr


@dataclasses.dataclass
class ShuffledBenchmarkReport:
    """Quantitative report for Shuffled-State Control Benchmark."""
    normal_total: int
    normal_correct: int
    normal_accuracy: float
    shuffled_total: int
    shuffled_correct: int
    shuffled_accuracy: float
    cag: float                  # Causal Accuracy Gap: normal_acc - shuffled_acc
    cgr: float                  # Causal Gain Ratio: normal_acc / max(shuffled_acc, 1e-4)
    normal_ece: float
    shuffled_ece: float
    cag_threshold: float = 0.20
    cgr_threshold: float = 3.0
    passed_causal_gate: bool = False
    verdict: str = "PENDING"
    normal_predictions: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    shuffled_predictions: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normal_total": self.normal_total,
            "normal_correct": self.normal_correct,
            "normal_accuracy": round(self.normal_accuracy, 4),
            "shuffled_total": self.shuffled_total,
            "shuffled_correct": self.shuffled_correct,
            "shuffled_accuracy": round(self.shuffled_accuracy, 4),
            "cag": round(self.cag, 4),
            "cgr": round(self.cgr, 2),
            "cag_threshold": self.cag_threshold,
            "cgr_threshold": self.cgr_threshold,
            "normal_ece": round(self.normal_ece, 4),
            "shuffled_ece": round(self.shuffled_ece, 4),
            "passed_causal_gate": self.passed_causal_gate,
            "verdict": self.verdict,
        }

    def to_markdown(self) -> str:
        gate_status = "PASSED" if self.passed_causal_gate else "FAILED"
        return f"""### Shuffled-State Control Causal Benchmark Report

- **Natural State Top-1 Accuracy**: {self.normal_accuracy * 100:.2f}% ({self.normal_correct}/{self.normal_total})
- **Shuffled State Control Top-1 Accuracy**: {self.shuffled_accuracy * 100:.2f}% ({self.shuffled_correct}/{self.shuffled_total})
- **Causal Accuracy Gap (CAG)**: {self.cag * 100:+.2f}% (Threshold: >={self.cag_threshold * 100:.0f}%)
- **Causal Gain Ratio (CGR)**: {self.cgr:.2f}x (Threshold: >={self.cgr_threshold:.1f}x)
- **Natural Arm 10-Bin ECE**: {self.normal_ece:.4f}
- **Shuffled Arm 10-Bin ECE**: {self.shuffled_ece:.4f}
- **Causal Quality Gate**: **{gate_status}** ({self.verdict})
"""


class ShuffledStateBenchmark:
    """Executes Shuffled-State Control evaluation on decision benchmarks."""

    def __init__(
        self,
        cag_threshold: float = 0.20,
        cgr_threshold: float = 3.0,
        num_ece_bins: int = 10,
    ):
        self.cag_threshold = cag_threshold
        self.cgr_threshold = cgr_threshold
        self.num_ece_bins = num_ece_bins

    @staticmethod
    def create_shuffled_batch(
        batch: List[Dict[str, Any]],
        shift: int = 1,
    ) -> List[Dict[str, Any]]:
        """Constructs a shuffled control batch where state contexts are cyclically permuted.

        Retains original candidate actions and ground truth targets, replacing the state context
        with state_{(i + shift) % B}.

        Args:
            batch: List of sample dicts containing 'state' (or 'context'), 'candidate_ids', and 'gold'/'target'.
            shift: Cyclic offset (default: 1).

        Returns:
            Shuffled batch list of identical length.
        """
        n = len(batch)
        if n <= 1:
            # Cannot form mismatched pair with 1 sample; duplicate and mutate key marker
            shuffled = [copy.deepcopy(batch[0])] if n == 1 else []
            if shuffled:
                shuffled[0]["state"] = f"[SHUFFLED_CONTROL_PERTURBED] {shuffled[0].get('state', '')}"
            return shuffled

        shuffled = []
        for i in range(n):
            target_idx = (i + shift) % n
            donor_state = batch[target_idx].get("state", batch[target_idx].get("context", ""))

            rec = copy.deepcopy(batch[i])
            if "state" in rec:
                rec["state"] = donor_state
            if "context" in rec:
                rec["context"] = donor_state
            shuffled.append(rec)

        return shuffled

    def evaluate_scorer(
        self,
        scorer_fn: Callable[[Any, List[str]], Tuple[str, float]],
        dataset: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> ShuffledBenchmarkReport:
        """Evaluates a scoring function against natural and shuffled datasets.

        Args:
            scorer_fn: Callable(state, candidates) -> (predicted_action, confidence_prob)
            dataset: List of sample dicts containing:
                     'state' or 'context': state representation
                     'candidate_ids' or 'candidates': list of candidate actions
                     'target' or 'gold' or 'ground_truth': correct action string
            batch_size: Batch slicing size for cyclic permutations.

        Returns:
            ShuffledBenchmarkReport.
        """
        if not dataset:
            return ShuffledBenchmarkReport(
                normal_total=0,
                normal_correct=0,
                normal_accuracy=0.0,
                shuffled_total=0,
                shuffled_correct=0,
                shuffled_accuracy=0.0,
                cag=0.0,
                cgr=0.0,
                normal_ece=0.0,
                shuffled_ece=0.0,
                passed_causal_gate=False,
                verdict="NO_SAMPLES"
            )

        # 1. Normal Evaluation
        normal_preds: List[Dict[str, Any]] = []
        normal_correct = 0

        def _call_scorer(s: Any, c: List[str], itm: Dict[str, Any]) -> Tuple[str, float]:
            try:
                return scorer_fn(s, c, itm)
            except TypeError:
                return scorer_fn(s, c)

        for item in dataset:
            state = item.get("state", item.get("context", ""))
            cands = item.get("candidate_ids", item.get("candidates", []))
            target = item.get("target") or item.get("gold") or item.get("ground_truth")
            if isinstance(target, dict) and "action" in target:
                target = target["action"]

            pred_act, conf = _call_scorer(state, cands, item)
            is_correct = (pred_act == target) or (isinstance(target, list) and pred_act in target)
            if is_correct:
                normal_correct += 1

            normal_preds.append({
                "confidence": float(conf),
                "correct": is_correct,
                "predicted": pred_act,
                "ground_truth": target,
            })

        normal_total = len(dataset)
        normal_acc = normal_correct / max(1, normal_total)

        # 2. Shuffled Control Evaluation (in batches)
        shuffled_dataset: List[Dict[str, Any]] = []
        for i in range(0, normal_total, batch_size):
            chunk = dataset[i : i + batch_size]
            shuffled_chunk = self.create_shuffled_batch(chunk, shift=1)
            shuffled_dataset.extend(shuffled_chunk)

        shuffled_preds: List[Dict[str, Any]] = []
        shuffled_correct = 0

        for item in shuffled_dataset:
            state = item.get("state", item.get("context", ""))
            cands = item.get("candidate_ids", item.get("candidates", []))
            target = item.get("target") or item.get("gold") or item.get("ground_truth")
            if isinstance(target, dict) and "action" in target:
                target = target["action"]

            pred_act, conf = _call_scorer(state, cands, item)
            is_correct = (pred_act == target) or (isinstance(target, list) and pred_act in target)
            if is_correct:
                shuffled_correct += 1


            shuffled_preds.append({
                "confidence": float(conf),
                "correct": is_correct,
                "predicted": pred_act,
                "ground_truth": target,
            })

        shuffled_total = len(shuffled_dataset)
        shuffled_acc = shuffled_correct / max(1, shuffled_total)

        # 3. Compute CAG & CGR
        cag, cgr = compute_cag_and_cgr(normal_acc, shuffled_acc)

        # 4. Compute 10-Bin ECE
        normal_cal = CalibrationEvaluator.compute_calibration(normal_preds, num_bins=self.num_ece_bins)
        shuffled_cal = CalibrationEvaluator.compute_calibration(shuffled_preds, num_bins=self.num_ece_bins)

        passed_cag = cag >= self.cag_threshold
        passed_cgr = cgr >= self.cgr_threshold
        passed_causal_gate = passed_cag and passed_cgr

        if passed_causal_gate:
            verdict = "PASSED_CAUSAL_DEPENDENCY"
        else:
            reasons = []
            if not passed_cag:
                reasons.append(f"CAG {cag:.3f} < {self.cag_threshold:.3f}")
            if not passed_cgr:
                reasons.append(f"CGR {cgr:.2f}x < {self.cgr_threshold:.1f}x")
            verdict = f"FAILED_CAUSAL_DEGRADATION ({', '.join(reasons)})"

        return ShuffledBenchmarkReport(
            normal_total=normal_total,
            normal_correct=normal_correct,
            normal_accuracy=normal_acc,
            shuffled_total=shuffled_total,
            shuffled_correct=shuffled_correct,
            shuffled_accuracy=shuffled_acc,
            cag=cag,
            cgr=cgr,
            normal_ece=normal_cal.ece_10bin,
            shuffled_ece=shuffled_cal.ece_10bin,
            cag_threshold=self.cag_threshold,
            cgr_threshold=self.cgr_threshold,
            passed_causal_gate=passed_causal_gate,
            verdict=verdict,
            normal_predictions=normal_preds,
            shuffled_predictions=shuffled_preds,
        )
