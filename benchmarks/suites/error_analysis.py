#!/usr/bin/env python3
"""Gen-Zero Error Analysis & Deep Diagnostic Failure Suite.

Ingests REAL benchmark prediction records (benchmarks/results/real_predictions.jsonl).
If that file is missing the suite raises FileNotFoundError; synthetic predictions are
strictly forbidden (anti-cheat policy).
It performs systematic diagnostic failure analysis comparing Gen-Zero against
dense LLM baselines (Qwen3.5-9B):
1. Error Taxonomy Classification:
   - Boundary Misclassification (High-Confidence Overconfidence vs Low-Confidence Epistemic Uncertainty)
   - Label Ambiguity (Subjective Human Annotator Disagreement & Noise)
   - Semantic Inversion & Negation Traps (Double Negatives, Polarity Reversals, Contrastive Refutation)
   - Multilingual Drift (en-US vs de-DE Compound Noun Fragmentation & Alignment Degradation)
   - Adversarial Syntactic Permutations & Lexical Overlap Traps
2. Confusion Matrix Computations (Global, Task-Specific, and Failure-Category).
3. Calibration & Confidence Decile Reliability Analysis.
4. Generates a markdown report of computed statistics only: benchmarks/results/error_analysis.md.
   Empty slices print 'not measured'; no narrative conclusions or example traces are emitted.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# ERROR TAXONOMY DEFINITIONS
# ============================================================================

TAXONOMY_CATEGORIES = {
    "boundary_misclassification": "Boundary Misclassification (High vs Low Confidence)",
    "label_ambiguity": "Label Ambiguity (Subjective Human Annotator Noise)",
    "semantic_inversion": "Semantic Inversion & Negation Traps",
    "multilingual_drift": "Multilingual Drift (de-DE vs en-US)",
    "syntactic_permutation": "Adversarial Syntactic Permutation (Lexical Overlap)",
    "borderline_threshold": "Borderline Threshold Classification",
    "domain_knowledge_gap": "Fine-Grained Domain Nuance Gap",
}


@dataclasses.dataclass
class ModelPrediction:
    prediction: str
    confidence: float
    latency_ms: float
    correct: bool
    system_mode: Optional[str] = None  # "reflex" or "mcts" for Gen-Zero


@dataclasses.dataclass
class BenchmarkRecord:
    id: str
    task: str
    category: str
    input_text: str
    ground_truth: str
    gen_zero: ModelPrediction
    qwen35_9b: ModelPrediction
    has_negation: bool
    is_borderline: bool
    annotator_agreement: Optional[float] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


def _pct(num: int, den: int) -> Optional[float]:
    """Percentage rounded to 2 dp, or None when the denominator is empty (not measured)."""
    return round(num / den * 100.0, 2) if den else None


def _mean(vals: List[float], nd: int = 3) -> Optional[float]:
    return round(sum(vals) / len(vals), nd) if vals else None


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else round(a - b, 2)


def _fmt(v: Optional[float], suffix: str = "", signed: bool = False) -> str:
    """Render a measured value, or an explicit 'not measured' marker for empty subsets."""
    if v is None:
        return "not measured"
    return f"{v:+}{suffix}" if signed else f"{v}{suffix}"


def _accuracy(subset: List["BenchmarkRecord"], side: str) -> Optional[float]:
    return _pct(sum(1 for r in subset if getattr(r, side).correct), len(subset))



# ============================================================================
# DIAGNOSTIC ENGINE & TAXONOMY EVALUATOR
# ============================================================================

class ErrorAnalysisSuite:
    """Performs deep failure diagnostic analysis on prediction records."""

    def __init__(self, predictions_path: Optional[Path] = None):
        self.predictions_path = predictions_path
        self.records: List[BenchmarkRecord] = []

    def load_dataset(self) -> int:
        """Loads the real benchmark predictions dataset. Fails closed when it is absent."""
        # Only the explicit path (or the repo default) is read. The former external-store search
        # paths (../gen-zero-eval-data, eval-data) were removed: the deleted synthesizer mirrored its
        # fake output there, so a stale fake file could load as "real".
        search_paths: List[Path] = [self.predictions_path or Path("benchmarks/results/real_predictions.jsonl")]

        target_file: Optional[Path] = None
        for p in search_paths:
            if p.exists() and p.stat().st_size > 0:
                target_file = p
                break

        if target_file is None:
            raise FileNotFoundError(
                "real_predictions.jsonl is required; synthetic predictions are strictly forbidden per anti-cheat policy "
                f"(searched: {[str(p) for p in search_paths]})"
            )

        # Parse JSONL records
        self.records = []
        records = []
        seen_ids = set()
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("is_synthetic") or d.get("metadata", {}).get("is_synthetic"):
                    raise ValueError("synthetic predictions are forbidden in real error analysis")
                if d["id"] in seen_ids:
                    raise ValueError(f"duplicate prediction id: {d['id']}")
                seen_ids.add(d["id"])
                for side in ("gen_zero", "qwen35_9b"):
                    pred = d[side]
                    # A producer's claimed success is never evaluation evidence.
                    pred["correct"] = pred["prediction"] == d["ground_truth"]
                    if (not math.isfinite(pred["confidence"])
                            or not 0 <= pred["confidence"] <= 1
                            or not math.isfinite(pred["latency_ms"])
                            or pred["latency_ms"] < 0):
                        raise ValueError(f"invalid confidence or latency for {side}")
                rec = BenchmarkRecord(
                    id=d["id"],
                    task=d["task"],
                    category=d.get("category", "standard"),
                    input_text=d["input_text"],
                    ground_truth=d["ground_truth"],
                    gen_zero=ModelPrediction(**d["gen_zero"]),
                    qwen35_9b=ModelPrediction(**d["qwen35_9b"]),
                    annotator_agreement=float(d["annotator_agreement"]) if d.get("annotator_agreement") is not None else None,
                    has_negation=bool(d.get("has_negation", False)),
                    is_borderline=bool(d.get("is_borderline", False)),
                    metadata=d.get("metadata", {}),
                )
                records.append(rec)

        self.records = records

        print(f"[+] Loaded {len(self.records)} records from {target_file}")
        return len(self.records)

    def analyze(self) -> Dict[str, Any]:
        """Executes systematic failure diagnostics across all dimensions."""
        if not self.records:
            raise ValueError("No records loaded. Call load_dataset() first.")

        total = len(self.records)

        # 1. Macro & Task-Level Summary
        task_stats: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: {
            "total": 0,
            "gz_correct": 0,
            "qw_correct": 0,
            "gz_conf_sum": 0.0,
            "qw_conf_sum": 0.0,
            "gz_lat_sum": 0.0,
            "qw_lat_sum": 0.0,
        })

        for r in self.records:
            t = task_stats[r.task]
            t["total"] += 1
            if r.gen_zero.correct:
                t["gz_correct"] += 1
            if r.qwen35_9b.correct:
                t["qw_correct"] += 1
            t["gz_conf_sum"] += r.gen_zero.confidence
            t["qw_conf_sum"] += r.qwen35_9b.confidence
            t["gz_lat_sum"] += r.gen_zero.latency_ms
            t["qw_lat_sum"] += r.qwen35_9b.latency_ms

        task_metrics: List[Dict[str, Any]] = []
        for task_name, ts in sorted(task_stats.items()):
            n = ts["total"]
            gz_acc = (ts["gz_correct"] / n) * 100.0
            qw_acc = (ts["qw_correct"] / n) * 100.0
            task_metrics.append({
                "task": task_name,
                "samples": n,
                "gen_zero_acc": round(gz_acc, 2),
                "qwen35_acc": round(qw_acc, 2),
                "delta": round(gz_acc - qw_acc, 2),
                "lead": "Gen-Zero" if gz_acc >= qw_acc else "Qwen3.5-9B",
                "gen_zero_avg_conf": round(ts["gz_conf_sum"] / n, 3),
                "qwen35_avg_conf": round(ts["qw_conf_sum"] / n, 3),
                "gen_zero_avg_lat_ms": round(ts["gz_lat_sum"] / n, 4),
                "qwen35_avg_lat_ms": round(ts["qw_lat_sum"] / n, 2),
            })

        # Overall summary
        gz_tot_corr = sum(1 for r in self.records if r.gen_zero.correct)
        qw_tot_corr = sum(1 for r in self.records if r.qwen35_9b.correct)
        overall_gz_acc = round((gz_tot_corr / total) * 100.0, 2)
        overall_qw_acc = round((qw_tot_corr / total) * 100.0, 2)

        # 2. Confusion Matrices (Global & Task Specific)
        confusion_matrices = self._compute_confusion_matrices()

        # 3. Deep Error Taxonomy:
        # 3a. Boundary Misclassification (High Confidence Error vs Low Confidence Uncertainty)
        boundary_analysis = self._compute_boundary_misclassification()

        # 3b. Label Ambiguity (Human Annotator Noise Analysis)
        ambiguity_analysis = self._compute_label_ambiguity_analysis()

        # 3c. Semantic Inversion & Negation Traps
        negation_analysis = self._compute_negation_inversion_analysis()

        # 3d. Multilingual Drift (MASSIVE de-DE vs en-US)
        multilingual_analysis = self._compute_multilingual_drift_analysis()

        # 3e. Syntactic Permutations & Lexical Overlap
        permutation_analysis = self._compute_syntactic_permutation_analysis()

        return {
            "dataset_summary": {
                "total_samples": total,
                "tasks_count": len(task_stats),
                "gen_zero_overall_acc": overall_gz_acc,
                "qwen35_overall_acc": overall_qw_acc,
                "gen_zero_wins": sum(1 for t in task_metrics if t["gen_zero_acc"] > t["qwen35_acc"]),
                "qwen35_wins": sum(1 for t in task_metrics if t["qwen35_acc"] > t["gen_zero_acc"]),
                "ties": sum(1 for t in task_metrics if t["gen_zero_acc"] == t["qwen35_acc"]),
                "gen_zero_mean_latency_ms": round(sum(r.gen_zero.latency_ms for r in self.records) / total, 4),
                "qwen35_mean_latency_ms": round(sum(r.qwen35_9b.latency_ms for r in self.records) / total, 2),
            },
            "task_metrics": task_metrics,
            "confusion_matrices": confusion_matrices,
            "boundary_misclassification": boundary_analysis,
            "label_ambiguity": ambiguity_analysis,
            "semantic_inversion": negation_analysis,
            "multilingual_drift": multilingual_analysis,
            "syntactic_permutation": permutation_analysis,
        }

    # ------------------------------------------------------------------------
    # SUB-DIAGNOSTIC 1: CONFUSION MATRICES
    # ------------------------------------------------------------------------
    UNMAPPED_LABEL = "(unmapped)"

    def _compute_confusion_matrices(self) -> Dict[str, Any]:
        """Calculates contingency tables and per-class F1 for key tasks.

        A prediction outside the ground-truth label set is counted in an explicit
        ``(unmapped)`` column, never coerced into a real class.
        """
        focal_tasks = ["Civil Comments", "PAWS", "VitaminC", "BoolQ", "Aegis 2.0"]
        results: Dict[str, Any] = {}

        for task in focal_tasks:
            task_recs = [r for r in self.records if r.task == task]
            if not task_recs:
                continue

            labels = sorted({r.ground_truth for r in task_recs})
            side_results: Dict[str, Any] = {}
            for side in ("gen_zero", "qwen35_9b"):
                preds = [getattr(r, side).prediction for r in task_recs]
                has_unmapped = any(p not in labels for p in preds)
                cols = labels + ([self.UNMAPPED_LABEL] if has_unmapped else [])
                matrix = {t: {c: 0 for c in cols} for t in labels}
                for r, p in zip(task_recs, preds):
                    matrix[r.ground_truth][p if p in labels else self.UNMAPPED_LABEL] += 1

                f1: Dict[str, Optional[float]] = {}
                for l in labels:
                    tp = matrix[l][l]
                    fp = sum(matrix[o][l] for o in labels if o != l)
                    fn = sum(matrix[l][c] for c in cols if c != l)
                    f1[l] = round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) else None
                side_results[side] = {"cols": cols, "matrix": matrix, "f1": f1}

            results[task] = {
                "labels": labels,
                "gen_zero_cols": side_results["gen_zero"]["cols"],
                "qwen35_cols": side_results["qwen35_9b"]["cols"],
                "gen_zero_matrix": side_results["gen_zero"]["matrix"],
                "qwen35_matrix": side_results["qwen35_9b"]["matrix"],
                "gen_zero_f1_scores": side_results["gen_zero"]["f1"],
                "qwen35_f1_scores": side_results["qwen35_9b"]["f1"],
            }

        return results

    # ------------------------------------------------------------------------
    # SUB-DIAGNOSTIC 2: BOUNDARY MISCLASSIFICATION (HIGH CONF VS LOW CONF)
    # ------------------------------------------------------------------------
    def _compute_boundary_misclassification(self) -> Dict[str, Any]:
        """Confidence-decile reliability plus high-confidence vs low-confidence errors."""
        edges = [(i / 10.0, (i + 1) / 10.0 if i < 9 else 1.0000001) for i in range(10)]
        labels = [f"{i / 10:.2f}-{(i + 1) / 10 - 0.01:.2f}" if i < 9 else "0.90-1.00" for i in range(10)]

        def bin_stats(side: str) -> Tuple[Dict[str, Dict[str, Any]], int]:
            stats = {lbl: {"total": 0, "correct": 0, "errors": 0, "conf_sum": 0.0} for lbl in labels}
            out_of_range = 0
            for r in self.records:
                pred = getattr(r, side)
                for (low, high), lbl in zip(edges, labels):
                    if low <= pred.confidence < high:
                        st = stats[lbl]
                        st["total"] += 1
                        st["conf_sum"] += pred.confidence
                        st["correct" if pred.correct else "errors"] += 1
                        break
                else:
                    out_of_range += 1
            return stats, out_of_range

        def ece(stats: Dict[str, Dict[str, Any]]) -> Optional[float]:
            n = sum(b["total"] for b in stats.values())
            if not n:
                return None
            return round(sum(
                b["total"] / n * abs(b["correct"] / b["total"] - b["conf_sum"] / b["total"])
                for b in stats.values() if b["total"]
            ), 4)

        gz_bins, gz_oor = bin_stats("gen_zero")
        qw_bins, qw_oor = bin_stats("qwen35_9b")

        gz_errors = [r for r in self.records if not r.gen_zero.correct]
        qw_errors = [r for r in self.records if not r.qwen35_9b.correct]
        gz_hi = sum(1 for r in gz_errors if r.gen_zero.confidence >= 0.85)
        gz_lo = sum(1 for r in gz_errors if r.gen_zero.confidence < 0.65)
        qw_hi = sum(1 for r in qw_errors if r.qwen35_9b.confidence >= 0.85)
        qw_lo = sum(1 for r in qw_errors if r.qwen35_9b.confidence < 0.65)

        return {
            "decile_labels": labels,
            "gen_zero_calibration_bins": gz_bins,
            "qwen35_calibration_bins": qw_bins,
            "gen_zero_confidence_out_of_range": gz_oor,
            "qwen35_confidence_out_of_range": qw_oor,
            "gen_zero_ece": ece(gz_bins),
            "qwen35_ece": ece(qw_bins),
            "gen_zero_total_errors": len(gz_errors),
            "gen_zero_high_conf_errors": gz_hi,
            "gen_zero_high_conf_error_pct": _pct(gz_hi, len(gz_errors)),
            "gen_zero_low_conf_uncertainty": gz_lo,
            "gen_zero_low_conf_uncertainty_pct": _pct(gz_lo, len(gz_errors)),
            "qwen35_total_errors": len(qw_errors),
            "qwen35_high_conf_errors": qw_hi,
            "qwen35_high_conf_error_pct": _pct(qw_hi, len(qw_errors)),
            "qwen35_low_conf_uncertainty": qw_lo,
            "qwen35_low_conf_uncertainty_pct": _pct(qw_lo, len(qw_errors)),
        }

    # ------------------------------------------------------------------------
    # SUB-DIAGNOSTIC 3: LABEL AMBIGUITY (HUMAN ANNOTATOR NOISE)
    # ------------------------------------------------------------------------
    def _compute_label_ambiguity_analysis(self) -> Dict[str, Any]:
        """Accuracy by annotator-agreement stratum (uses the record's own agreement field)."""
        valid_annotated = [r for r in self.records if r.annotator_agreement is not None]
        if not valid_annotated:
            return {
                "tier_breakdown": {},
                "status": "not_measured",
                "message": "No records contained annotator_agreement metadata",
                "gen_zero_errors_in_ambiguous_stratum_pct": None,
                "qwen35_errors_in_ambiguous_stratum_pct": None,
            }

        tiers = {
            "high_agreement": [r for r in valid_annotated if r.annotator_agreement >= 0.80],
            "moderate_agreement": [r for r in valid_annotated if 0.60 <= r.annotator_agreement < 0.80],
            "low_agreement_noise": [r for r in valid_annotated if r.annotator_agreement < 0.60],
        }

        tier_breakdown: Dict[str, Dict[str, Any]] = {}
        for name, subset in tiers.items():
            if not subset:
                continue
            tier_breakdown[name] = {
                "samples": len(subset),
                "pct_of_dataset": _pct(len(subset), len(valid_annotated)),
                "gen_zero_accuracy": _accuracy(subset, "gen_zero"),
                "qwen35_accuracy": _accuracy(subset, "qwen35_9b"),
                "gen_zero_mean_confidence": _mean([r.gen_zero.confidence for r in subset]),
                "qwen35_mean_confidence": _mean([r.qwen35_9b.confidence for r in subset]),
            }

        gz_errors = [r for r in valid_annotated if not r.gen_zero.correct]
        qw_errors = [r for r in valid_annotated if not r.qwen35_9b.correct]
        return {
            "tier_breakdown": tier_breakdown,
            "gen_zero_errors_in_ambiguous_stratum_pct": _pct(
                sum(1 for r in gz_errors if r.annotator_agreement < 0.70), len(gz_errors)),
            "qwen35_errors_in_ambiguous_stratum_pct": _pct(
                sum(1 for r in qw_errors if r.annotator_agreement < 0.70), len(qw_errors)),
        }

    # ------------------------------------------------------------------------
    # SUB-DIAGNOSTIC 4: SEMANTIC INVERSION & NEGATION TRAPS
    # ------------------------------------------------------------------------
    def _compute_negation_inversion_analysis(self) -> Dict[str, Any]:
        """Accuracy on records flagged negation / semantic_inversion vs the rest.

        Subsets come only from the record's own flags (``has_negation``, ``category``,
        ``metadata['double_negation']``); the input text is never pattern-matched.
        """
        negation_subset = [r for r in self.records if r.has_negation or r.category == "semantic_inversion"]
        control_subset = [r for r in self.records if not r.has_negation and r.category != "semantic_inversion"]
        double_neg = [r for r in negation_subset if r.metadata.get("double_negation") is True]

        gz_neg, gz_ctl = _accuracy(negation_subset, "gen_zero"), _accuracy(control_subset, "gen_zero")
        qw_neg, qw_ctl = _accuracy(negation_subset, "qwen35_9b"), _accuracy(control_subset, "qwen35_9b")
        gz_dn, qw_dn = _accuracy(double_neg, "gen_zero"), _accuracy(double_neg, "qwen35_9b")

        return {
            "negation_samples": len(negation_subset),
            "control_samples": len(control_subset),
            "gen_zero_negation_acc": gz_neg,
            "gen_zero_control_acc": gz_ctl,
            "gen_zero_negation_penalty": _diff(gz_ctl, gz_neg),
            "qwen35_negation_acc": qw_neg,
            "qwen35_control_acc": qw_ctl,
            "qwen35_negation_penalty": _diff(qw_ctl, qw_neg),
            "double_negation_samples": len(double_neg),
            "gen_zero_double_negation_acc": gz_dn,
            "qwen35_double_negation_acc": qw_dn,
            "gen_zero_double_negation_delta_vs_control": _diff(gz_dn, gz_ctl),
            "qwen35_double_negation_delta_vs_control": _diff(qw_dn, qw_ctl),
        }

    # ------------------------------------------------------------------------
    # SUB-DIAGNOSTIC 5: MULTILINGUAL DRIFT (de-DE vs en-US)
    # ------------------------------------------------------------------------
    def _compute_multilingual_drift_analysis(self) -> Dict[str, Any]:
        """English vs German intent-routing accuracy on the MASSIVE slices."""
        en_recs = [r for r in self.records if r.task == "MASSIVE en-US"]
        de_recs = [r for r in self.records if r.task == "MASSIVE de-DE"]
        compound_recs = [r for r in de_recs if r.category == "multilingual_drift"]

        gz_en, qw_en = _accuracy(en_recs, "gen_zero"), _accuracy(en_recs, "qwen35_9b")
        gz_de, qw_de = _accuracy(de_recs, "gen_zero"), _accuracy(de_recs, "qwen35_9b")
        gz_comp, qw_comp = _accuracy(compound_recs, "gen_zero"), _accuracy(compound_recs, "qwen35_9b")

        return {
            "en_samples": len(en_recs),
            "de_samples": len(de_recs),
            "gen_zero_en_acc": gz_en,
            "gen_zero_de_acc": gz_de,
            "gen_zero_cross_lingual_drop": _diff(gz_en, gz_de),
            "qwen35_en_acc": qw_en,
            "qwen35_de_acc": qw_de,
            "qwen35_cross_lingual_drop": _diff(qw_en, qw_de),
            "german_compound_samples": len(compound_recs),
            "gen_zero_compound_acc": gz_comp,
            "qwen35_compound_acc": qw_comp,
            "compound_acc_gap": _diff(gz_comp, qw_comp),
        }

    # ------------------------------------------------------------------------
    # SUB-DIAGNOSTIC 6: SYNTACTIC PERMUTATION & LEXICAL OVERLAP
    # ------------------------------------------------------------------------
    def _compute_syntactic_permutation_analysis(self) -> Dict[str, Any]:
        """PAWS accuracy on records categorised syntactic_permutation vs the other PAWS records."""
        perm_recs = [r for r in self.records if r.task == "PAWS"]
        traps = [r for r in perm_recs if r.category == "syntactic_permutation"]
        standards = [r for r in perm_recs if r.category != "syntactic_permutation"]

        gz_trap, qw_trap = _accuracy(traps, "gen_zero"), _accuracy(traps, "qwen35_9b")
        gz_std, qw_std = _accuracy(standards, "gen_zero"), _accuracy(standards, "qwen35_9b")

        return {
            "lexical_overlap_trap_samples": len(traps),
            "standard_paraphrase_samples": len(standards),
            "gen_zero_trap_acc": gz_trap,
            "qwen35_trap_acc": qw_trap,
            "gen_zero_standard_acc": gz_std,
            "qwen35_standard_acc": qw_std,
            "gen_zero_trap_drop": _diff(gz_std, gz_trap),
            "qwen35_trap_drop": _diff(qw_std, qw_trap),
        }

    # ------------------------------------------------------------------------
    # MARKDOWN REPORT GENERATOR
    # ------------------------------------------------------------------------
    def generate_markdown_report(self, analysis: Dict[str, Any], output_path: Path):
        """Writes a markdown report containing only statistics computed from the loaded records.

        Empty subsets are printed as 'not measured'. No narrative conclusions, example
        traces or mitigation claims are emitted: the report never asserts anything the
        input records do not show.
        """
        summary = analysis["dataset_summary"]
        tasks = analysis["task_metrics"]
        bm = analysis["boundary_misclassification"]
        la = analysis["label_ambiguity"]
        si = analysis["semantic_inversion"]
        md = analysis["multilingual_drift"]
        sp = analysis["syntactic_permutation"]
        cms = analysis["confusion_matrices"]

        gz_lat, qw_lat = summary["gen_zero_mean_latency_ms"], summary["qwen35_mean_latency_ms"]
        latency_line = (
            f"- **Mean recorded latency**: Gen-Zero `{gz_lat} ms` vs Qwen3.5-9B `{qw_lat} ms` "
            "(as recorded in the input file; not re-measured here)"
        )

        md_lines: List[str] = [
            "# Gen-Zero Diagnostic Failure Analysis Report",
            "",
            f"> Statistics computed from {summary['total_samples']:,} recorded predictions across "
            f"{summary['tasks_count']} tasks (loaded from real_predictions.jsonl). "
            "'not measured' means the input file has no records for that slice.",
            "",
            "---",
            "",
            "## 1. Summary",
            "",
            f"- **Total predictions**: `{summary['total_samples']:,}` across `{summary['tasks_count']}` tasks",
            f"- **Gen-Zero pooled accuracy**: `{summary['gen_zero_overall_acc']}%` (task wins: `{summary['gen_zero_wins']}/{summary['tasks_count']}`)",
            f"- **Qwen3.5-9B pooled accuracy**: `{summary['qwen35_overall_acc']}%` (task wins: `{summary['qwen35_wins']}/{summary['tasks_count']}`)",
            f"- **Ties**: `{summary['ties']}`",
            latency_line,
            "",
            "> Point accuracies only. No paired per-sample significance test is run here, so no difference below is claimed as significant.",
            "",
            "### Task-Level Accuracy",
            "",
            "| Task | Samples | Gen-Zero Acc | Qwen3.5-9B Acc | Δ (GZ − Qwen) | Gen-Zero Latency | Qwen Latency |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for t in tasks:
            md_lines.append(
                f"| {t['task']} | {t['samples']} | {t['gen_zero_acc']}% | {t['qwen35_acc']}% | "
                f"{t['delta']:+}% | {t['gen_zero_avg_lat_ms']} ms | {t['qwen35_avg_lat_ms']} ms |"
            )

        gz_err, qw_err = bm["gen_zero_total_errors"], bm["qwen35_total_errors"]
        md_lines.extend([
            "",
            "---",
            "",
            "## 2. Error Breakdown",
            "",
            "### 2.1 Confidence vs Correctness",
            "",
            "| Metric | Gen-Zero | Qwen3.5-9B |",
            "| :--- | :---: | :---: |",
            f"| Total errors | `{gz_err}` | `{qw_err}` |",
            f"| High-confidence errors (conf ≥ 0.85) | `{bm['gen_zero_high_conf_errors']}` ({_fmt(bm['gen_zero_high_conf_error_pct'], '%')} of errors) | `{bm['qwen35_high_conf_errors']}` ({_fmt(bm['qwen35_high_conf_error_pct'], '%')} of errors) |",
            f"| Low-confidence errors (conf < 0.65) | `{bm['gen_zero_low_conf_uncertainty']}` ({_fmt(bm['gen_zero_low_conf_uncertainty_pct'], '%')} of errors) | `{bm['qwen35_low_conf_uncertainty']}` ({_fmt(bm['qwen35_low_conf_uncertainty_pct'], '%')} of errors) |",
            f"| Expected Calibration Error (10 equal-width bins, mean-confidence vs accuracy) | `{_fmt(bm['gen_zero_ece'])}` | `{_fmt(bm['qwen35_ece'])}` |",
            f"| Records with confidence outside [0, 1] (excluded from bins) | `{bm['gen_zero_confidence_out_of_range']}` | `{bm['qwen35_confidence_out_of_range']}` |",
            "",
            "#### Confidence Decile Reliability Table",
            "",
            "| Confidence Bin | Gen-Zero Samples | Gen-Zero Acc | Gen-Zero Mean Conf | Qwen3.5 Samples | Qwen3.5 Acc | Qwen3.5 Mean Conf |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for lbl in bm["decile_labels"]:
            gz_b = bm["gen_zero_calibration_bins"][lbl]
            qw_b = bm["qwen35_calibration_bins"][lbl]
            gz_acc = _fmt(_pct(gz_b["correct"], gz_b["total"]), "%")
            qw_acc = _fmt(_pct(qw_b["correct"], qw_b["total"]), "%")
            gz_mc = _fmt(_mean([gz_b["conf_sum"] / gz_b["total"]]) if gz_b["total"] else None)
            qw_mc = _fmt(_mean([qw_b["conf_sum"] / qw_b["total"]]) if qw_b["total"] else None)
            md_lines.append(f"| `{lbl}` | {gz_b['total']} | {gz_acc} | {gz_mc} | {qw_b['total']} | {qw_acc} | {qw_mc} |")

        md_lines.extend([
            "",
            "---",
            "",
            "### 2.2 Annotator Agreement Strata",
            "",
            "| Agreement Stratum | Share of Records | Gen-Zero Acc | Qwen3.5 Acc | Gen-Zero Mean Conf | Qwen3.5 Mean Conf |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for tier_key, label_name in [
            ("high_agreement", "High (≥ 0.80)"),
            ("moderate_agreement", "Moderate (0.60-0.80)"),
            ("low_agreement_noise", "Low (< 0.60)"),
        ]:
            if tier_key in la["tier_breakdown"]:
                tb = la["tier_breakdown"][tier_key]
                md_lines.append(
                    f"| {label_name} | {tb['pct_of_dataset']}% ({tb['samples']} records) | {tb['gen_zero_accuracy']}% | "
                    f"{tb['qwen35_accuracy']}% | {tb['gen_zero_mean_confidence']} | {tb['qwen35_mean_confidence']} |"
                )
            else:
                md_lines.append(f"| {label_name} | not measured | not measured | not measured | not measured | not measured |")

        md_lines.extend([
            "",
            f"- Gen-Zero errors with annotator agreement < 0.70: `{_fmt(la['gen_zero_errors_in_ambiguous_stratum_pct'], '%')}`",
            f"- Qwen3.5-9B errors with annotator agreement < 0.70: `{_fmt(la['qwen35_errors_in_ambiguous_stratum_pct'], '%')}`",
            "",
            "---",
            "",
            "### 2.3 Negation / Semantic Inversion Slice",
            "",
            "Slice membership comes from each record's own `has_negation` / `category` / `metadata.double_negation` flags.",
            "",
            "| Slice | Samples | Gen-Zero Acc | Qwen3.5-9B Acc |",
            "| :--- | :---: | :---: | :---: |",
            f"| Control (no negation flag) | {si['control_samples']} | {_fmt(si['gen_zero_control_acc'], '%')} | {_fmt(si['qwen35_control_acc'], '%')} |",
            f"| Negation / inversion | {si['negation_samples']} | {_fmt(si['gen_zero_negation_acc'], '%')} | {_fmt(si['qwen35_negation_acc'], '%')} |",
            f"| Double negation (flagged) | {si['double_negation_samples']} | {_fmt(si['gen_zero_double_negation_acc'], '%')} | {_fmt(si['qwen35_double_negation_acc'], '%')} |",
            "",
            f"- Accuracy drop, control → negation: Gen-Zero `{_fmt(si['gen_zero_negation_penalty'], ' pp')}`, Qwen3.5-9B `{_fmt(si['qwen35_negation_penalty'], ' pp')}`",
            "",
            "---",
            "",
            "### 2.4 MASSIVE en-US vs de-DE",
            "",
            "| Slice | Samples | Gen-Zero Acc | Qwen3.5-9B Acc |",
            "| :--- | :---: | :---: | :---: |",
            f"| MASSIVE en-US | {md['en_samples']} | {_fmt(md['gen_zero_en_acc'], '%')} | {_fmt(md['qwen35_en_acc'], '%')} |",
            f"| MASSIVE de-DE | {md['de_samples']} | {_fmt(md['gen_zero_de_acc'], '%')} | {_fmt(md['qwen35_de_acc'], '%')} |",
            f"| de-DE records categorised multilingual_drift | {md['german_compound_samples']} | {_fmt(md['gen_zero_compound_acc'], '%')} | {_fmt(md['qwen35_compound_acc'], '%')} |",
            "",
            f"- Accuracy drop, en-US → de-DE: Gen-Zero `{_fmt(md['gen_zero_cross_lingual_drop'], ' pp')}`, Qwen3.5-9B `{_fmt(md['qwen35_cross_lingual_drop'], ' pp')}`",
            "",
            "---",
            "",
            "### 2.5 PAWS: syntactic_permutation vs other records",
            "",
            "| PAWS Slice | Samples | Gen-Zero Acc | Qwen3.5-9B Acc |",
            "| :--- | :---: | :---: | :---: |",
            f"| Other PAWS records | {sp['standard_paraphrase_samples']} | {_fmt(sp['gen_zero_standard_acc'], '%')} | {_fmt(sp['qwen35_standard_acc'], '%')} |",
            f"| Category syntactic_permutation | {sp['lexical_overlap_trap_samples']} | {_fmt(sp['gen_zero_trap_acc'], '%')} | {_fmt(sp['qwen35_trap_acc'], '%')} |",
            "",
            f"- Accuracy drop, other → syntactic_permutation: Gen-Zero `{_fmt(sp['gen_zero_trap_drop'], ' pp')}`, Qwen3.5-9B `{_fmt(sp['qwen35_trap_drop'], ' pp')}`",
            "",
            "---",
            "",
            "## 3. Confusion Matrices",
            "",
        ])

        if not cms:
            md_lines.extend(["not measured (no records for Civil Comments / PAWS / VitaminC / BoolQ / Aegis 2.0)", ""])

        for idx, (task_name, cm) in enumerate(cms.items(), 1):
            for title, cols_key, m_key, f_key in [
                ("Gen-Zero", "gen_zero_cols", "gen_zero_matrix", "gen_zero_f1_scores"),
                ("Qwen3.5-9B", "qwen35_cols", "qwen35_matrix", "qwen35_f1_scores"),
            ]:
                cols, matrix, f1 = cm[cols_key], cm[m_key], cm[f_key]
                if title == "Gen-Zero":
                    md_lines.extend([f"### 3.{idx} {task_name}", ""])
                md_lines.extend([
                    f"**{title}** (rows = ground truth, columns = prediction):",
                    "",
                    "| True \\ Pred | " + " | ".join(cols) + " | F1 |",
                    "| :--- | " + " | ".join([":---:"] * len(cols)) + " | :---: |",
                ])
                for l in cm["labels"]:
                    md_lines.append(
                        f"| **{l}** | " + " | ".join(str(matrix[l][c]) for c in cols) + f" | `{_fmt(f1[l])}` |"
                    )
                md_lines.append("")

        md_lines.extend([
            "---",
            "*Report automatically generated by `benchmarks/suites/error_analysis.py`.*",
        ])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"[+] Diagnostic report written to: {output_path}")

    def run(self, output_md: Optional[Path] = None, output_json: Optional[Path] = None) -> Dict[str, Any]:
        """Runs the entire pipeline."""
        print("==========================================================================")
        print("              GEN-ZERO DEEP ERROR ANALYSIS & DIAGNOSTICS                  ")
        print("==========================================================================")
        self.load_dataset()
        analysis = self.analyze()

        out_md = output_md or Path("benchmarks/results/error_analysis.md")
        self.generate_markdown_report(analysis, out_md)

        if output_json:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"[+] JSON metrics written to: {output_json}")

        print("==========================================================================")
        print("                 DIAGNOSTIC ANALYSIS COMPLETE                             ")
        print("==========================================================================")
        return analysis


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gen-Zero Error Analysis & Diagnostics Engine")
    parser.add_argument("--predictions", type=str, default="benchmarks/results/real_predictions.jsonl",
                        help="Path to real_predictions.jsonl")
    parser.add_argument("--output-md", type=str, default="benchmarks/results/error_analysis.md",
                        help="Path for generated Markdown failure report")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path for JSON metrics dump")
    args = parser.parse_args()

    pred_path = Path(args.predictions)

    suite = ErrorAnalysisSuite(predictions_path=pred_path)
    suite.run(
        output_md=Path(args.output_md) if args.output_md else None,
        output_json=Path(args.output_json) if args.output_json else None,
    )


if __name__ == "__main__":
    main()
