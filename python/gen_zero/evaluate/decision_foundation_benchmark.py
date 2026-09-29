"""Comprehensive Evaluation & Calibration Benchmark for Open-Source Decision Foundation.

Implements Milestone 4 of Issue #17:
1. 11 Core Decision Domains Benchmark: Evaluates overall and domain-specific decision accuracy (> 90%).
2. 10-Bin ECE & Wilson 95% Confidence Interval Validation via CalibrationEvaluator.
3. Permutation Invariance Testing: Evaluates argmax flip rate under candidate order shuffling.
4. Tone Perturbation Sensitivity Testing: Measures accuracy drop under emotional adversarial prompts (<= 2%).
5. Generates structured JSON and Markdown audit reports in results/gen_zero/.
"""

from typing import Callable, Dict, List, Any, Optional, Tuple, Sequence
import os
import json
import time
import math

from gen_zero.gate.locked_evaluator import (
    CalibrationEvaluator,
    compute_wilson_score_interval,
)
from gen_zero.train.contrastive_data_pipeline import (
    ContrastiveDataPipeline,
    CORE_DECISION_DOMAINS,
)
from gen_zero.train.qwen_post_trainer import QwenPostTrainer


class DecisionFoundationBenchmark:
    """Benchmark suite auditing accuracy, calibration, permutation equivariance, and tone robustness.

    ``QwenPostTrainer.simulate_decision_forward`` is a deterministic template
    scorer used by the repository's CPU smoke tests.  It is deliberately kept
    usable for backwards compatibility, but it is not model inference.  Runs
    that use it are labelled as synthetic and their acceptance claims are
    withdrawn in the report.  A caller that has a real model must provide an
    ``inference_fn`` returning the same ``choice``/``probs`` payload.
    """

    def __init__(
        self,
        trainer: Optional[QwenPostTrainer] = None,
        inference_fn: Optional[Callable[[str, List[str]], Dict[str, Any]]] = None,
    ):
        self.trainer = trainer or QwenPostTrainer()
        self.data_pipeline = ContrastiveDataPipeline()
        self.inference_fn = inference_fn

    def _forward(self, prompt: str, candidates: List[str]) -> Dict[str, Any]:
        """Run the configured scorer and validate its minimal prediction contract."""
        if not candidates:
            raise ValueError("benchmark items must provide at least one candidate")
        if self.inference_fn is not None:
            result = self.inference_fn(prompt, candidates)
        else:
            # This is explicitly a smoke-test path; QwenPostTrainer's method is
            # named ``simulate`` because it does not load or execute Qwen.
            result = self.trainer.simulate_decision_forward(prompt, candidates)
        if not isinstance(result, dict):
            raise TypeError("inference_fn must return a prediction dictionary")
        choice = result.get("choice")
        probs = result.get("probs")
        if choice not in candidates:
            raise ValueError(f"prediction choice {choice!r} is not one of the candidates")
        if isinstance(probs, dict):
            probs = [probs.get(candidate, 0.0) for candidate in candidates]
        if not isinstance(probs, (list, tuple)) or len(probs) != len(candidates):
            raise ValueError("prediction probs must align one-to-one with candidates")
        probabilities = [float(value) for value in probs]
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("prediction probabilities must be finite and non-negative")
        total = sum(probabilities)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("prediction probabilities must have a positive finite sum")
        # The evaluator consumes a confidence, not a rounded probability
        # vector.  Normalize here so custom inference functions cannot inflate
        # confidence by returning unnormalised scores.
        probabilities = [value / total for value in probabilities]
        return {**result, "choice": choice, "probs": probabilities}

    def run_full_benchmark(
        self,
        samples_per_domain: int = 15,
        output_dir: str = "results/gen_zero",
    ) -> Dict[str, Any]:
        """Runs end-to-end evaluation suite across all 11 domains."""
        if not isinstance(samples_per_domain, int) or isinstance(samples_per_domain, bool) or samples_per_domain <= 0:
            raise ValueError("samples_per_domain must be a positive integer")
        corpus = self.data_pipeline.generate_benchmark_corpus(samples_per_domain=samples_per_domain)
        if not corpus:
            raise ValueError("benchmark corpus must not be empty")

        domain_results: Dict[str, Dict[str, Any]] = {
            domain: {"total": 0, "correct": 0, "samples": []}
            for domain in CORE_DECISION_DOMAINS
        }

        predictions: List[Dict[str, Any]] = []
        total_correct = 0
        unknown_domain_samples = 0
        tone_neutral_correct = 0
        tone_neutral_total = 0
        tone_emotional_correct = 0
        tone_emotional_total = 0

        # Permutation invariance tracker
        permutation_flips = 0
        permutation_tests = 0

        for item in corpus:
            domain = item["domain"]
            prompt = item["prompt"]
            candidates = item["candidates"]
            target = item["target_choice"]
            tone = item.get("tone", "neutral")

            # Model decision forward
            pred_res = self._forward(prompt, candidates)
            choice = pred_res["choice"]
            # Confidence is the probability assigned to the selected label.
            # Taking the vector maximum would hide an invalid/non-argmax choice
            # and inflate confidence in custom inference implementations.
            choice_index = candidates.index(choice)
            max_prob = pred_res["probs"][choice_index]
            is_correct = (choice == target)
            if is_correct:
                total_correct += 1

            if domain in domain_results:
                domain_results[domain]["total"] += 1
                if is_correct:
                    domain_results[domain]["correct"] += 1
            else:
                # Keep the overall denominator honest when a custom corpus
                # contains a domain outside the declared eleven-domain suite.
                unknown_domain_samples += 1

            # Track calibration sample
            predictions.append({
                "confidence": max_prob,
                "predicted_label": choice,
                "ground_truth": target,
            })

            # Track tone statistics
            if tone == "neutral":
                tone_neutral_total += 1
                if is_correct:
                    tone_neutral_correct += 1
            elif tone == "emotional_adversarial":
                tone_emotional_total += 1
                if is_correct:
                    tone_emotional_correct += 1

            # Test permutation invariance on neutral samples
            if tone == "neutral" and len(candidates) > 1:
                permutation_tests += 1
                reversed_candidates = list(reversed(candidates))
                perm_res = self._forward(prompt, reversed_candidates)
                if perm_res["choice"] != choice:
                    permutation_flips += 1

        # Calculate metrics
        total_samples = len(corpus)
        overall_accuracy = round(total_correct / total_samples, 4) if total_samples else 0.0

        domain_accuracies = {
            dom: round(d["correct"] / max(1, d["total"]), 4)
            for dom, d in domain_results.items()
        }

        # 10-Bin ECE
        cal_report = CalibrationEvaluator.compute_calibration(predictions, num_bins=10)

        # Tone sensitivity
        neutral_acc = tone_neutral_correct / tone_neutral_total if tone_neutral_total else None
        emotional_acc = tone_emotional_correct / tone_emotional_total if tone_emotional_total else None
        tone_drop_rate = (neutral_acc - emotional_acc) if neutral_acc is not None and emotional_acc is not None else None

        # Permutation invariance flip rate
        flip_rate = permutation_flips / permutation_tests if permutation_tests else None

        # Wilson 95% CI on overall accuracy
        ci_lower, ci_upper = compute_wilson_score_interval(total_correct, total_samples)

        inference_is_synthetic = self.inference_fn is None
        # ContrastiveDataPipeline.generate_benchmark_corpus is a generated
        # fixture, regardless of which inference function scores it.  Keep the
        # two provenance dimensions separate so a real model cannot turn a
        # synthetic corpus into a production benchmark claim.
        dataset_is_synthetic = True
        is_synthetic = dataset_is_synthetic or inference_is_synthetic
        calibration_passed = bool(predictions) and bool(cal_report.passed_safety_red_line)
        result_payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_synthetic": is_synthetic,
            "dataset_is_synthetic": dataset_is_synthetic,
            "inference_is_synthetic": inference_is_synthetic,
            "withdrawn_as_model_evidence": is_synthetic,
            "claim_scope": (
                "generated benchmark fixture; not production accuracy or deployment evidence"
                if dataset_is_synthetic else "caller supplied benchmark dataset"
            ),
            "unknown_domain_samples": unknown_domain_samples,
            "total_evaluated_samples": total_samples,
            "overall_accuracy": overall_accuracy,
            "accuracy_wilson_95_ci": (round(ci_lower, 4), round(ci_upper, 4)),
            "meets_90pct_target": bool(overall_accuracy >= 0.90),
            "acceptance_status": (
                "WITHHELD_SYNTHETIC" if is_synthetic else
                ("PASS" if total_samples and overall_accuracy >= 0.90 else "FAIL")
            ),
            "domain_accuracies": domain_accuracies,
            "calibration": {
                "ece_10bin": round(cal_report.ece_10bin, 4),
                "mce": round(cal_report.mce, 4),
                "brier_score": round(cal_report.brier_score, 4),
                "confident_error_count": cal_report.confident_error_count,
                "passed_safety_red_line": calibration_passed,
            },
            "tone_invariance": {
                "neutral_accuracy": round(neutral_acc, 4) if neutral_acc is not None else None,
                "emotional_accuracy": round(emotional_acc, 4) if emotional_acc is not None else None,
                "drop_rate": round(tone_drop_rate, 4) if tone_drop_rate is not None else None,
                "meets_2pct_drop_sla": bool(tone_drop_rate is not None and tone_drop_rate <= 0.02),
            },
            "permutation_invariance": {
                "tested_count": permutation_tests,
                "flip_count": permutation_flips,
                "argmax_flip_rate": round(flip_rate, 4) if flip_rate is not None else None,
                "is_permutation_equivariant": bool(flip_rate is not None and flip_rate == 0.0),
            },
        }

        # Write reports
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "qwen_foundation_benchmark_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

        md_path = os.path.join(output_dir, "qwen_foundation_benchmark_report.md")
        self._write_markdown_report(result_payload, md_path)

        return result_payload

    def _write_markdown_report(self, res: Dict[str, Any], filepath: str) -> None:
        """Writes audit report to GitHub Flavored Markdown."""
        tone_drop = res["tone_invariance"]["drop_rate"]
        tone_drop_display = f"{tone_drop * 100:.2f}%" if tone_drop is not None else "N/A"
        flip_rate = res["permutation_invariance"]["argmax_flip_rate"]
        flip_rate_display = f"{flip_rate * 100:.2f}%" if flip_rate is not None else "N/A"
        lines = [
            "# Qwen3.5-9B 开源决策基座后训练评测与校准报告",
            "",
            f"**评测时间**: {res['timestamp']} | **样本规模**: {res['total_evaluated_samples']} 对比样本",
            "",
            "## 1. 核心指标概览",
            "",
            "| 评测维度 | 验收目标 | 实测结果 | 状态 |",
            "| :--- | :--- | :--- | :---: |",
            f"| **全量决策准确率** | $\\ge 90.0\\%$ | **{res['overall_accuracy']*100:.2f}%** (95% CI: [{res['accuracy_wilson_95_ci'][0]:.3f}, {res['accuracy_wilson_95_ci'][1]:.3f}]) | {'WITHHELD' if res.get('is_synthetic') else ('PASS' if res['meets_90pct_target'] else 'FAIL')} |",
            f"| **10-Bin ECE 校准误差** | $\\le 0.35$ | **{res['calibration']['ece_10bin']:.4f}** (MCE: {res['calibration']['mce']:.4f}) | {'WITHHELD' if res.get('is_synthetic') else ('PASS' if res['calibration']['passed_safety_red_line'] else 'FAIL/UNAVAILABLE')} |",
            f"| **情绪语气扰动准确率回退** | $\\le 2.0\\%$ | **{tone_drop_display}** | {'WITHHELD' if res.get('is_synthetic') else ('PASS' if res['tone_invariance']['meets_2pct_drop_sla'] else 'FAIL/UNAVAILABLE')} |",
            f"| **选项置换等变性翻转率** | $0.0\\%$ | **{flip_rate_display}** (翻转次数: {res['permutation_invariance']['flip_count']}) | {'WITHHELD' if res.get('is_synthetic') else ('PASS' if res['permutation_invariance']['is_permutation_equivariant'] else 'FAIL/UNAVAILABLE')} |",
            "",
            "## 2. 11 大核心决策场景细分表现",
            "",
            "| 决策门类 | 样本类别 | 准确率 | 状态 |",
            "| :--- | :--- | :--- | :---: |",
        ]

        for dom, acc in res["domain_accuracies"].items():
            status_str = "WITHHELD" if res.get("is_synthetic") else ("PASS" if acc >= 0.85 else "WARN")
            lines.append(f"| `{dom}` | 生产决策与因果干预 | **{acc*100:.1f}%** | {status_str} |")

        lines.extend([
            "",
            "## 3. 结论与部署建议",
            "",
            (
                "- **证据范围**：本次结果使用生成式基准样本，仅验证评测流程；结果已撤回作为生产准确率、校准、硬件或部署证据。"
                if res.get("is_synthetic") else
                f"- **模型评测状态**：{'PASS' if res.get('acceptance_status') == 'PASS' else 'FAIL'}；硬件与部署 SLA 未在此评测中测量。"
            ),
            "",
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
