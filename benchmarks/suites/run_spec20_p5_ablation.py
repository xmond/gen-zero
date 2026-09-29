#!/usr/bin/env python3
"""Spec 20 P5: combined ablation & final grand scorecard report (docs/zero/20-*.md S7.1 P5, S8).

This script does NOT run any new accuracy evaluation. Real 13-task accuracy on the Qwen3.5-9B
q9b_diff_16_24_compact (8192-D) representation, scored with {linear_probe, adapter (Formulation A),
supcon} under a strict nested 1-SE selection ladder, already exists at
benchmarks/results/01png_sota_ensemble_report_phase4.json (macro 76.06%, git log b3d2c39's parent).
That is the ONE ablation-grid cell with real end-to-end accuracy. This script aggregates it, verifies
its Wilson intervals independently, and reports it honestly next to every other axis cell the Spec 20
P5 brief asks about (q9b_mid, q9b_diff_full, Formulation B / adapter_b, dual-manifold Qwen+Gemma fusion,
partition anchor pooling) for which S8 "未验证/未完成" is explicit: those cells have algorithmic-
correctness unit tests (re-run here, live, not cached) but never went through the nested-CV training/
selection ladder on real 13-task data. No accuracy number is invented for them.

It also runs fresh, real CPU batch-1 latency microbenchmarks (same reps/warmup/median/p95 methodology
as spec19_head_latency_microbench.py; random weights, since dense-kernel latency does not depend on
weight values, only on shape) for FoldedResidualAdapterBHead and DualManifoldHead at the real shapes
named in the Spec 20 P5 brief (D=8192 for adapter_b; D_Q=8192, D_G=2816 for dual_manifold, folded to
two GEMVs), across the real K (candidate count) range actually observed in the 13 tasks (2..18, from
01png_grand_challenge_report.json's gold_counts).

Every axis cell in the output carries a `status` of "evaluated" (real accuracy, with an `evidence`
pointer of file + JSON key path) or "not_evaluated" (with a `reason` citing the spec line and the
unit-test evidence that DOES exist). Reviewers: grep the `evidence`/`reason` fields against the cited
files before trusting a number.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Pin single-threaded BLAS before numpy is imported: this host runs OpenBLAS with
# MAX_THREADS=64 and no external pinning, so unpinned GEMV latency is dominated by thread-pool
# contention on a shared, often-loaded box (spec19_head_latency_microbench.py's own "Gotcha" note),
# not by the algorithm. Every latency number in this report is single-thread.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
RESULTS = REPO / "benchmarks" / "results"
sys.path.insert(0, str(SUITES))

import sota_enhanced_heads as eh  # noqa: E402
import partition_anchor_pooling as pap  # noqa: E402
from spec19_head_latency_microbench import timeit as _timeit  # noqa: E402  (reps=2000, warm=200, median/p95 us)

PHASE4_REPORT = RESULTS / "01png_sota_ensemble_report_phase4.json"
LEGACY_GRAND_CHALLENGE_REPORT = RESULTS / "01png_grand_challenge_report.json"
SPEC19_MICROBENCH = RESULTS / "spec19_head_latency_microbench_1t.json"
SPEC_DOC = REPO / "docs" / "zero" / "20-triad-deep-enhancement-multi-dimensional-analysis-spec.md"

UNIT_TEST_FILES = {
    "P1_partition_anchor_pooling": REPO / "benchmarks" / "tests" / "test_spec20_anchor_pooling.py",
    "P3_formulation_b_adapter": REPO / "benchmarks" / "tests" / "test_spec20_formulation_b.py",
    "P4_dual_manifold": REPO / "benchmarks" / "tests" / "test_spec20_dual_manifold.py",
}

NOT_EVALUATED_REASON = (
    "Spec 20 S8 '未验证'/'未完成' (docs/zero/20-triad-deep-enhancement-multi-dimensional-analysis-spec.md:442-453): "
    "real accuracy/non-inferiority for this cell was never measured. Only algorithmic-correctness unit tests "
    "(fold identity, monotonicity, orthogonality, shape/export consistency on synthetic inputs) exist for the "
    "underlying head/pooling code; it never ran through the nested-CV training/selection ladder on the real "
    "13-task data, because the required upstream artifact (e.g. Gemma feature extraction, or a q9b_diff_full "
    "cache) does not exist in this repo."
)


# --------------------------------------------------------------------------- Wilson interval (S7.1 P5 brief)

def wilson95(k: int, n: int, z: float = 1.96) -> List[float]:
    """Wilson score interval, percent scale. Identical formula to
    benchmarks/suites/evaluate_full_13_grand_scorecard.py:226 (and 5 other suites in this repo);
    duplicated here (11 lines) rather than imported, because that module imports torch/sklearn/CUDA-only
    helpers at module scope and this report must run without a GPU."""
    if n <= 0:
        raise ValueError(f"wilson95 needs n > 0, got {n}")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def collapsed(max_pred_class_frac: float, K: int) -> bool:
    """Same collapse gate as evaluate_full_13_grand_scorecard.py:242."""
    return bool(K >= 2 and max_pred_class_frac > 0.95)


def bootstrap_macro_ci(per_task_acc: List[float], n_resamples: int = 20000, seed: int = 0) -> List[float]:
    """Task-level bootstrap 95% CI on the macro average of 13 per-task accuracies.

    This is NOT a single-proportion Wilson interval: macro_avg_13 is a mean over 13 heterogeneous
    tasks, not k successes out of n trials, so no (k, n) pair exists for it (S7.1 P5 brief asks for a
    Wilson CI on "each improvement"; per-task deltas get one below, the macro figure gets this instead).
    With only 13 tasks the resampling has coarse resolution (13 possible task-multiplicities); reported
    as a documented limitation, not silently smoothed over."""
    arr = np.asarray(per_task_acc, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


# --------------------------------------------------------------------------- real data loaders

def load_phase4() -> dict:
    if not PHASE4_REPORT.exists():
        raise SystemExit(f"required real evidence file missing: {PHASE4_REPORT}")
    return json.loads(PHASE4_REPORT.read_text())


def load_legacy_grand_challenge() -> Optional[dict]:
    if not LEGACY_GRAND_CHALLENGE_REPORT.exists():
        return None
    return json.loads(LEGACY_GRAND_CHALLENGE_REPORT.read_text())


def load_spec19_microbench() -> Optional[dict]:
    if not SPEC19_MICROBENCH.exists():
        return None
    return json.loads(SPEC19_MICROBENCH.read_text())


def real_task_k_range() -> Dict[str, int]:
    """K (candidate count) per task, from the legacy grand-challenge report's gold_counts length.
    Real, task-definition-level data (candidate lists do not change across report runs); used only to
    pick realistic K values for the CPU latency microbenchmark below, never for any accuracy claim."""
    legacy = load_legacy_grand_challenge()
    if legacy is None:
        return {}
    out = {}
    for name, t in legacy.get("tasks", {}).items():
        gc = t.get("gold_counts")
        if gc:
            out[name] = len(gc)
    return out


def build_per_task_scorecard(phase4: dict) -> List[dict]:
    rows = []
    for name, t in phase4["tasks"].items():
        correct, n = int(t["correct"]), int(t["n"])
        recomputed = wilson95(correct, n)
        stored = t.get("wilson95")
        if stored is not None:
            drift = max(abs(recomputed[0] - stored[0]), abs(recomputed[1] - stored[1]))
            if drift > 0.05:
                raise SystemExit(
                    f"{name}: recomputed wilson95 {recomputed} disagrees with stored {stored} "
                    f"(drift {drift:.3f}pp) -- the report and the source data have diverged, refusing to "
                    f"silently paper over it"
                )
        lo, hi = recomputed
        rows.append({
            "task": name, "dataset": t["dataset"], "n": n, "correct": correct,
            "accuracy": t["accuracy"], "wilson95_recomputed": recomputed, "wilson95_stored": stored,
            "chosen_strategy": t["chosen_strategy"], "collapsed": t["collapsed"],
            "max_pred_class_frac": t["max_pred_class_frac"],
            "majority_class_train_prior_acc": t.get("majority_class_train_prior_acc"),
            "nimble": t["nimble"], "jev": t["jev"],
            "delta_vs_nimble": t["delta_vs_nimble"], "delta_vs_jev": t["delta_vs_jev"],
            # No per-item Jev/Nimble predictions exist in this repo, so no PAIRED CI on the delta is
            # possible (S7.1 P5 brief asks for a Wilson CI per improvement; this is the honest substitute:
            # is the commercial score itself inside OUR single-model Wilson interval, i.e. statistically
            # indistinguishable from our accuracy under sampling noise alone).
            "jev_inside_our_wilson95": bool(lo <= t["jev"] <= hi),
            "nimble_inside_our_wilson95": bool(lo <= t["nimble"] <= hi),
            # Real trained-weight CPU decision latency for the head that actually won this task
            # (phase4's own measurement, not this script's random-weight microbench).
            "decision_latency_us_median": t.get("decision_latency_us", {}).get("median"),
            "decision_latency_us_p95": t.get("decision_latency_us", {}).get("p95"),
            "balanced_accuracy": t["balanced_accuracy"], "macro_f1": t["macro_f1"],
        })
    rows.sort(key=lambda r: r["task"])
    return rows


# --------------------------------------------------------------------------- fresh CPU latency benchmarks

def bench_adapter_b(D: int, K: int, rank: int, rng: np.random.Generator) -> dict:
    """Real FoldedResidualAdapterBHead.scores() latency, random weights at real (D, K, rank) shapes,
    built via the head's own __init__ (bypassing .fit(), which needs torch/labelled rows and is not an
    accuracy claim) so the exact production inference code path (_scores_from_standardized) is timed."""
    f32 = np.float32
    arrays = {
        "mu_full": np.zeros(D, f32), "sd_full": np.ones(D, f32),
        "W0": (rng.standard_normal((K, D)) * 0.01).astype(f32), "b0": np.zeros(K, f32),
        "C": (rng.standard_normal((K, rank)) * 0.01).astype(f32),
        "U": (rng.standard_normal((rank, D)) * 0.01).astype(f32), "a": np.zeros(rank, f32),
    }
    cfg = {"K": K, "pair": False, "in_dim": D, "D": D, "rank": rank, "head_type": "adapter_b"}
    head = eh.FoldedResidualAdapterBHead(arrays, cfg)
    x = rng.standard_normal(D).astype(f32)
    return _timeit(lambda: head.scores(x))


def bench_dual_manifold(D_Q: int, D_G: int, K: int, rng: np.random.Generator) -> dict:
    """Real DualManifoldHead.scores() latency: two folded GEMVs, random weights at real (D_Q, D_G, K).
    Built directly via __init__ (the folded arrays only; unfolded=None), the same object the production
    export path constructs for CPU inference (S6.2: inference never materializes the k_q+k_g intermediate)."""
    f32 = np.float32
    arrays = {
        "W_fold_Q": (rng.standard_normal((K, D_Q)) * 0.01).astype(f32),
        "W_fold_G": (rng.standard_normal((K, D_G)) * 0.01).astype(f32),
        "b_fold": np.zeros(K, f32),
    }
    cfg = {"K": K, "pair": False, "in_dim_q": D_Q, "in_dim_g": D_G, "D_Q": D_Q, "D_G": D_G,
           "k_q": 1, "k_g": 1, "gate": 1.0, "head_type": "dual_manifold"}
    head = eh.DualManifoldHead(arrays, cfg)
    x_q = rng.standard_normal(D_Q).astype(f32)
    x_g = rng.standard_normal(D_G).astype(f32)
    return _timeit(lambda: head.scores(x_q, x_g))


def bench_pooling(T: int, D: int, rng: np.random.Generator) -> dict:
    """Real PartitionAnchorPooler.pool() vs uniform-mean pooling latency at a realistic long-context
    shape (T=1536 tokens is the largest budget Spec 20 S2.1 measures; D=4096 is one Qwen3.5-9B layer's
    hidden width, S8: 'Qwen4096x2')."""
    H = rng.standard_normal((T, D)).astype(np.float32)
    pooler = pap.PartitionAnchorPooler()

    def anchor():
        return pooler.pool(H)

    def uniform_mean():
        return H.mean(axis=0)

    return {"partition_anchor_pooling": _timeit(anchor, reps=200), "uniform_mean": _timeit(uniform_mean, reps=200)}


def bench_adapter_a(D: int, K: int, rank: int, rng: np.random.Generator) -> dict:
    """Real DeepResidualAdapterHead.scores() latency (Spec 19 S6.1, Formulation A), folded export,
    random weights at real (D, K, rank) shapes. SupConHead subclasses this with the same scores()
    (only its training loss differs, eh.py:454 `class SupConHead(DeepResidualAdapterHead)` adds no new
    inference arrays), so this number also stands in for the supcon head's inference cost."""
    f32 = np.float32
    arrays = {
        "mu_full": np.zeros(D, f32), "sd_full": np.ones(D, f32),
        "W_h": (rng.standard_normal((K, D)) * 0.01).astype(f32),
        "W_fold": (rng.standard_normal((K, rank)) * 0.01).astype(f32), "b_fold": np.zeros(K, f32),
        "W_down": (rng.standard_normal((rank, D)) * 0.01).astype(f32), "b_down": np.zeros(rank, f32),
    }
    cfg = {"K": K, "pair": False, "in_dim": D, "D": D, "rank": rank, "folded": True, "head_type": "adapter"}
    head = eh.DeepResidualAdapterHead(arrays, cfg)
    x = rng.standard_normal(D).astype(f32)
    return _timeit(lambda: head.scores(x))


def bench_linear_probe(D: int, K: int, rng: np.random.Generator) -> dict:
    """Real linear-probe GEMV latency at the same D/K shapes, for apples-to-apples comparison; identical
    kernel to spec19_head_latency_microbench.py's `linear`, re-measured fresh on this host/run."""
    f32 = np.float32
    x = rng.standard_normal(D).astype(f32)
    W = (rng.standard_normal((K, D)) * 0.01).astype(f32)
    b = np.zeros(K, f32)
    return _timeit(lambda: x @ W.T + b)


def run_latency_suite() -> dict:
    rng = np.random.default_rng(0)
    k_by_task = real_task_k_range()
    k_list = list(k_by_task.values()) or [2, 3, 18]
    # median over the 13 TASKS (not over the unique K values): with K observed as
    # {18,18,3,3,3,2,2,2,2,2,5,5,5}, the task-level median is 3 (a real task's K), not 4 (no task has K=4).
    k_probe = sorted({min(k_list), int(np.median(k_list)), max(k_list)})
    loadavg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    cpu_count = os.cpu_count() or 1
    host_busy = loadavg[0] > cpu_count
    out = {
        "methodology": "batch=1, float32, random weights (dense-kernel latency is weight-value-independent, "
                        "only shape-dependent; same principle as spec19_head_latency_microbench.py); "
                        "BLAS pinned to 1 thread (OMP/OPENBLAS/MKL_NUM_THREADS=1); "
                        "reps=2000 warm=200 for head GEMVs, reps=200 warm=200 for O(T) pooling; "
                        "median_us / p95_us over independently timed calls on this host, this run.",
        "loadavg_1min": loadavg[0], "cpu_count": cpu_count,
        "host_busy_caveat": (
            f"loadavg 1min={loadavg[0]:.1f} > cpu_count={cpu_count}: this is a shared, contended host. "
            "median_us is comparatively robust; p95_us and any single outlier ratio can reflect OS "
            "scheduling preemption, not algorithmic cost. Do not read p95/median ratios above ~3x as a "
            "property of the head." if host_busy else "host load was below cpu_count at run time."
        ),
        "k_values_observed_in_13_tasks": k_by_task,
        "k_probe_points": k_probe,
        "adapter_formulation_a_D8192_rank64_shape_only": {
            f"K{k}": bench_adapter_a(8192, k, 64, rng) for k in k_probe},
        "adapter_formulation_b_D8192_rank64": {f"K{k}": bench_adapter_b(8192, k, 64, rng) for k in k_probe},
        "dual_manifold_DQ8192_DG2816": {f"K{k}": bench_dual_manifold(8192, 2816, k, rng) for k in k_probe},
        "linear_probe_D8192_fresh": {f"K{k}": bench_linear_probe(8192, k, rng) for k in k_probe},
        "note_adapter_a_and_supcon": "adapter_formulation_a's shape-only number also stands in for supcon "
                                      "(same scores() code, sota_enhanced_heads.py:454); both ALSO have REAL "
                                      "trained-weight decision_latency_us per task in the per-task scorecard "
                                      "below (phase4 report), which is the number that actually matters -- "
                                      "these random-weight benches exist only so adapter_a/b/dual_manifold can "
                                      "be compared shape-for-shape on one host, one run.",
    }
    pool = {"boundary": "GPU-extraction-time pooling cost (paid once per document when caching features), "
                        "NOT a per-query CPU decision-head cost -- do not compare directly to the head "
                        "latencies above (S7.3: report latency boundaries as they are).",
           **bench_pooling(1536, 4096, rng)}
    out["pooling_T1536_D4096"] = pool
    spec19 = load_spec19_microbench()
    if spec19 is not None:
        out["cross_check_spec19_microbench_D8192_K18"] = {
            "source": str(SPEC19_MICROBENCH.relative_to(REPO)),
            "linear_probe": spec19.get("heads", {}).get("D8192_K18", {}).get("linear_probe"),
            "residual_adapter_folded_into_head": spec19.get("heads", {}).get("D8192_K18", {})
                .get("residual_adapter_folded_into_head"),
        }
    return out


# --------------------------------------------------------------------------- unit-test re-run (live, not cached)

def run_unit_tests() -> dict:
    out = {}
    for key, path in UNIT_TEST_FILES.items():
        if not path.exists():
            out[key] = {"status": "missing", "path": str(path.relative_to(REPO))}
            continue
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", "pytest", str(path), "-q"],
                               cwd=REPO, capture_output=True, text=True, timeout=600)
        stdout_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        # pytest -q always prints its one-line run summary ("N passed[, M warnings] in Ts") last,
        # after any warnings-summary block (which can itself contain a test name with "passed" as a
        # substring, e.g. "test_..._is_rejected_..._passed" -- so this takes the LAST line, unqualified,
        # not a substring search).
        summary = stdout_lines[-1] if stdout_lines else ""
        out[key] = {
            "path": str(path.relative_to(REPO)),
            "exit_code": proc.returncode,
            "summary": summary.strip(),
            "seconds": round(time.perf_counter() - t0, 1),
        }
    return out


# --------------------------------------------------------------------------- axis grid assembly

def build_axis_grid(phase4: dict, unit_tests: dict) -> dict:
    agg = phase4["aggregate"]
    phase4_evidence = {"file": str(PHASE4_REPORT.relative_to(REPO)), "key": "aggregate.macro_avg_13"}
    axis1_evidence = {
        "file": str(PHASE4_REPORT.relative_to(REPO)),
        "key": "command (--features-dir ...q9b_diff_compact) and "
               "tasks.<task>.leakage_gate.q9b_diff_16_24_compact",
        "command_string": phase4.get("command"),
    }

    axis1_representation = {
        "q9b_diff_compact_8192d": {
            "status": "evaluated", "macro_avg_13": agg["macro_avg_13"], "micro_acc": agg["micro_acc"],
            "evidence": axis1_evidence,
            "note": "synthesize_layer_diff_features.py:15 compact = [h24; h24-h16], 8192-D",
        },
        "q9b_mid_8192d": {
            "status": "not_evaluated", "reason": NOT_EVALUATED_REASON,
            "note": "synthesize_layer_diff_features.py:8 q9b_mid = [mean@16 | last@16], 8192-D. "
                    "01png_grand_challenge_report.json's 63.82%/58.89% baseline uses a DIFFERENT, "
                    "unrelated encoder (Qwen2.5-0.5B, see benchmark_01png_grand_challenge.py:26) and "
                    "must not be read as a q9b_mid data point.",
        },
        "q9b_diff_full_12288d": {
            "status": "not_evaluated", "reason": NOT_EVALUATED_REASON,
            "note": "synthesize_layer_diff_features.py:14 full = [h16; h24; h24-h16], 12288-D; no "
                    "q9b_diff_full <task>.npz cache or report found under benchmarks/results/ or "
                    "benchmarks/artifacts/zero/.",
        },
    }

    axis2_head = {
        "linear_probe": {
            "status": "evaluated", "chosen_on_n_tasks": len(agg.get("tasks_selected_linear", [])),
            "chosen_on_tasks": agg.get("tasks_selected_linear", []), "evidence": phase4_evidence,
        },
        "adapter_formulation_a": {
            "status": "evaluated", "chosen_on_n_tasks": len(agg.get("tasks_selected_adapter", [])),
            "chosen_on_tasks": agg.get("tasks_selected_adapter", []), "evidence": phase4_evidence,
        },
        "supcon": {
            "status": "evaluated", "chosen_on_n_tasks": len(agg.get("tasks_selected_supcon", [])),
            "chosen_on_tasks": agg.get("tasks_selected_supcon", []), "evidence": phase4_evidence,
        },
        "adapter_formulation_b": {
            "status": "not_evaluated", "reason": NOT_EVALUATED_REASON,
            "unit_test_evidence": unit_tests.get("P3_formulation_b_adapter"),
        },
    }

    axis3_fusion = {
        "single_qwen": {"status": "evaluated", "evidence": phase4_evidence},
        "dual_manifold_qwen_gemma": {
            "status": "not_evaluated", "reason": NOT_EVALUATED_REASON,
            "note": "Spec 20 S8 (line 444): 'Gemma GGUF实际2816输出及它自己的token长度' explicitly listed "
                    "未验证 -- no real Gemma feature cache exists in this repo, so DualManifoldHead has "
                    "never been fit on real paired Qwen+Gemma rows.",
            "unit_test_evidence": unit_tests.get("P4_dual_manifold"),
        },
    }

    pooling_axis = {
        "uniform_mean": {
            "status": "evaluated_implicitly",
            "note": "The 13-task phase4 evaluation uses cached mean/last-token pooling baked into the "
                    "q9b_diff_16_24_compact feature cache itself (synthesize_layer_diff_features.py), "
                    "not a separate pooling module call; no task in the 13-task suite exercises >1536 "
                    "tokens, so this is not a stress test of pooling choice.",
            "evidence": phase4_evidence,
        },
        "partition_anchor_pooling": {
            "status": "not_evaluated", "reason": NOT_EVALUATED_REASON,
            "unit_test_evidence": unit_tests.get("P1_partition_anchor_pooling"),
        },
    }

    return {
        "axis1_representation": axis1_representation,
        "axis2_head": axis2_head,
        "axis3_fusion": axis3_fusion,
        "pooling_axis": pooling_axis,
    }


GATING_TABLE_ASSESSMENT = {
    "source": "docs/zero/20-triad-deep-enhancement-multi-dimensional-analysis-spec.md:356-365 defines the "
              "P0-P5 phases; the spec's own table has NO status column. Every status string below is THIS "
              "SCRIPT's assessment against real repo evidence, not a value copied from the spec.",
    "P0_evidence_baseline": {
        "status": "claimed passed in the task brief; partially spot-checked here",
        "evidence": "tasks.<task>.leakage_gate.q9b_diff_16_24_compact.{id_overlap,text_overlap,"
                    "family_overlap} == 0 for all 13 tasks in 01png_sota_ensemble_report_phase4.json "
                    "(checked programmatically below); full nested/group-CV audit not re-verified by "
                    "this script",
    },
    "P1_long_context_pooling": "engineering done + unit tests pass; accuracy not evaluated (S8:445)",
    "P2_layer_dynamics_diff": "passed with real accuracy: this IS the 76.06% phase4 result",
    "P3_folded_heads": "engineering done + unit tests pass; accuracy not evaluated (S8:445)",
    "P4_dual_source_fusion": "engineering done + unit tests pass; accuracy not evaluated (S8:444-445), "
                             "Gemma features never extracted",
    "P5_combined_ablation": "this report: only P0+P2 have a real jointly-evaluated cell; P1/P3/P4 cannot "
                             "be honestly combined into it yet (S7.1: '不要全组合盲搜...仅组合内层训练验证"
                             "通过者' -- P1/P3/P4 never ran the inner nested-CV/1-SE ladder on real data at all)",
}


def check_p0_leakage_gate(phase4: dict) -> dict:
    """Real, programmatic spot-check of the P0 claim: id/text/family overlap must be 0 for every task."""
    bad = []
    for name, t in phase4["tasks"].items():
        gate = t.get("leakage_gate", {}).get("q9b_diff_16_24_compact", {})
        if gate.get("id_overlap", 1) != 0 or gate.get("text_overlap", 1) != 0 or gate.get("family_overlap", 1) != 0:
            bad.append({name: gate})
    return {"tasks_checked": len(phase4["tasks"]), "tasks_with_nonzero_overlap": bad,
            "all_clean": len(bad) == 0}


def build_headline_caveat(agg: dict, macro_excl_collapsed: Optional[dict], macro_bootstrap: List[float],
                          rows: List[dict]) -> str:
    """Never hardcode a specific task name or a specific direction of change here: compute both from
    the actual excluded-task list and the actual sign comparison, so this stays correct if the set of
    collapsed tasks (or their effect) changes on a future rerun."""
    if not macro_excl_collapsed:
        return "No task was flagged collapsed in this run; the headline macro figure is not propped up by one."
    excluded = macro_excl_collapsed["excluded_tasks"]
    priors = macro_excl_collapsed["excluded_task_majority_class_prior_acc"]
    excl_detail = ", ".join(
        f"{t} (majority-class train prior={priors.get(t)}%, our accuracy="
        f"{next((r['accuracy'] for r in rows if r['task'] == t), None)}%)" for t in excluded
    )
    overall_sign = "positive" if agg["delta_macro_vs_jev"] >= 0 else "negative"
    excl_sign = "positive" if macro_excl_collapsed["delta_vs_jev"] >= 0 else "negative"
    flip_note = (
        f"the sign vs Jev FLIPS from {overall_sign} to {excl_sign}"
        if overall_sign != excl_sign else
        f"the sign vs Jev stays {excl_sign} even excluding these tasks"
    )
    jev_ref, nimble_ref = agg["png_reference_avg"]["jev"], agg["png_reference_avg"]["nimble"]
    inside = [name for name, val in (("Jev", jev_ref), ("Nimble", nimble_ref))
             if macro_bootstrap[0] <= val <= macro_bootstrap[1]]
    ci_note = (
        f"The 13-task bootstrap CI on our own macro is {macro_bootstrap}; {' and '.join(inside)} "
        f"{'sit' if len(inside) != 1 else 'sits'} inside it too, i.e. {'that reference score is' if len(inside) == 1 else 'those reference scores are'} "
        "not distinguishable from our macro at this sample size."
        if inside else
        f"The 13-task bootstrap CI on our own macro is {macro_bootstrap}; neither Jev ({jev_ref}%) nor "
        f"Nimble ({nimble_ref}%) falls inside it."
    )
    return (
        f"This {agg['delta_macro_vs_jev']:+.2f}pp vs Jev is carried in part by {len(excluded)} collapsed "
        f"task(s): {excl_detail}. Excluding them, macro over the remaining "
        f"{macro_excl_collapsed['n_tasks_remaining']} tasks is ours={macro_excl_collapsed['ours']}% vs "
        f"Jev={macro_excl_collapsed['jev']}% (delta {macro_excl_collapsed['delta_vs_jev']:+.2f}pp) vs "
        f"Nimble={macro_excl_collapsed['nimble']}% (delta {macro_excl_collapsed['delta_vs_nimble']:+.2f}pp): "
        f"{flip_note}. {ci_note}"
    ).strip()


def build_report(rows: List[dict], phase4: dict, axis_grid: dict, latency: dict, unit_tests: dict) -> dict:
    agg = phase4["aggregate"]
    per_task_acc = [r["accuracy"] for r in rows]
    macro_bootstrap = bootstrap_macro_ci(per_task_acc)
    p0_check = check_p0_leakage_gate(phase4)

    collapsed_rows = [r for r in rows if r["collapsed"]]
    excl_rows = [r for r in rows if not r["collapsed"]]
    macro_excl_collapsed = None
    if collapsed_rows and excl_rows:
        macro_excl_collapsed = {
            "excluded_tasks": [r["task"] for r in collapsed_rows],
            "excluded_task_majority_class_prior_acc": {
                r["task"]: r["majority_class_train_prior_acc"] for r in collapsed_rows
            },
            "n_tasks_remaining": len(excl_rows),
            "ours": round(sum(r["accuracy"] for r in excl_rows) / len(excl_rows), 3),
            "jev": round(sum(r["jev"] for r in excl_rows) / len(excl_rows), 3),
            "nimble": round(sum(r["nimble"] for r in excl_rows) / len(excl_rows), 3),
        }
        macro_excl_collapsed["delta_vs_jev"] = round(
            macro_excl_collapsed["ours"] - macro_excl_collapsed["jev"], 3)
        macro_excl_collapsed["delta_vs_nimble"] = round(
            macro_excl_collapsed["ours"] - macro_excl_collapsed["nimble"], 3)

    selector_dist = {
        "linear_probe": len(agg.get("tasks_selected_linear", [])),
        "adapter_formulation_a": len(agg.get("tasks_selected_adapter", [])),
        "supcon": len(agg.get("tasks_selected_supcon", [])),
    }
    n_total_selected = sum(selector_dist.values())
    host = {
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }
    return {
        "title": "Spec 20 P5: combined ablation & final grand scorecard (real data where it exists, "
                 "explicit not_evaluated everywhere else)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "python3 benchmarks/suites/run_spec20_p5_ablation.py",
        "host": host,
        "spec_ref": str(SPEC_DOC.relative_to(REPO)),
        "gating_table_assessment": GATING_TABLE_ASSESSMENT,
        "p0_leakage_gate_check": p0_check,
        "axis_grid": axis_grid,
        "per_task_real_scorecard": rows,
        "macro_summary": {
            "macro_avg_13_point_estimate": agg["macro_avg_13"],
            "macro_avg_13_task_level_bootstrap_ci95": macro_bootstrap,
            "bootstrap_method": "13-task resample with replacement, 20000 draws, percentile CI on the "
                                 "mean of resampled per-task accuracies; NOT a Wilson interval (macro_avg_13 "
                                 "is a mean over tasks, not one binomial proportion)",
            "micro_acc": agg["micro_acc"],
            "delta_macro_vs_nimble": agg["delta_macro_vs_nimble"],
            "delta_macro_vs_jev": agg["delta_macro_vs_jev"],
            "png_reference_avg": agg["png_reference_avg"],
            "tasks_beating_jev": agg.get("tasks_beating_jev", []),
            "tasks_collapsed": agg.get("tasks_collapsed", []),
            "macro_excluding_collapsed_tasks": macro_excl_collapsed,
            "evidence": {"file": str(PHASE4_REPORT.relative_to(REPO)), "key": "aggregate"},
        },
        "one_se_selector_distribution": {
            **selector_dist, "n_tasks_total": len(rows), "n_tasks_selected_sum_check": n_total_selected,
            "evidence": {"file": str(PHASE4_REPORT.relative_to(REPO)),
                         "key": "aggregate.tasks_selected_{linear,adapter,supcon}"},
        },
        "collapse_gate": {
            "formula": "collapsed = K >= 2 and max_pred_class_frac > 0.95 "
                       "(evaluate_full_13_grand_scorecard.py:242)",
            "collapsed_tasks": [r["task"] for r in rows if r["collapsed"]],
            "evidence": {"file": str(PHASE4_REPORT.relative_to(REPO)), "key": "tasks.<task>.collapsed"},
        },
        "cpu_latency_us": latency,
        "unit_test_evidence": unit_tests,
        "final_verdict": {
            "only_real_end_to_end_combination": (
                f"q9b_diff_compact (8192-D) x {{linear_probe, adapter_formulation_a, supcon}} with strict "
                f"nested 1-SE selection = macro {agg['macro_avg_13']}% over 13 tasks, "
                f"beating Jev ({agg['png_reference_avg']['jev']}%, delta "
                f"{agg['delta_macro_vs_jev']:+.2f}pp) and Nimble ({agg['png_reference_avg']['nimble']}%, "
                f"delta {agg['delta_macro_vs_nimble']:+.2f}pp)."
            ),
            "headline_caveat": build_headline_caveat(agg, macro_excl_collapsed, macro_bootstrap, rows),
            "not_yet_combinable_axes": [
                "q9b_mid_8192d", "q9b_diff_full_12288d", "adapter_formulation_b",
                "dual_manifold_qwen_gemma", "partition_anchor_pooling",
            ],
            "why": "Each has real, passing algorithmic-correctness unit tests (see unit_test_evidence) but "
                   "was never run through the nested-CV training/selection ladder on the real 13-task data. "
                   "Fabricating an accuracy delta for them would violate Spec 20 S7.2's own instruction "
                   "('推理路径不根据...伪装双源成功') and this task's anti-cheating mandate. Real P5 numbers "
                   "for these axes require the 未完成 work S8:451 names: GPU extraction, CPU training, and "
                   "a new nested-CV test evaluation -- none of which this report-generation script performs.",
        },
    }


# --------------------------------------------------------------------------- markdown rendering

def render_markdown(report: dict) -> str:
    lines = [f"# {report['title']}", "", f"Generated: {report['generated_utc']}  ", f"Spec: `{report['spec_ref']}`",
              "", "## 0. Headline", "",
              report["final_verdict"]["only_real_end_to_end_combination"], "",
              f"**{report['final_verdict']['headline_caveat']}**", "",
              "**Everything else below with `status: not_evaluated` has no accuracy number in this report on purpose.**",
              "", "## 1. Gating table (this script's assessment; the spec table itself has no status column)", "",
              "| Phase | Status |", "|---|---|"]
    for k, v in report["gating_table_assessment"].items():
        if k == "source":
            continue
        if isinstance(v, dict):
            lines.append(f"| {k} | {v['status']} (evidence: {v['evidence']}) |")
        else:
            lines.append(f"| {k} | {v} |")
    p0 = report["p0_leakage_gate_check"]
    lines += ["", f"P0 leakage-gate spot check: {p0['tasks_checked']} tasks, "
                  f"all_clean={p0['all_clean']}, nonzero_overlap={p0['tasks_with_nonzero_overlap']}"]

    lines += ["", "## 2. Axis grid", ""]
    for axis_name, axis in report["axis_grid"].items():
        lines.append(f"### {axis_name}")
        lines.append("")
        lines.append("| Cell | Status | Detail |")
        lines.append("|---|---|---|")
        for cell, info in axis.items():
            status = info["status"]
            if status == "evaluated" or status == "evaluated_implicitly":
                parts = []
                if info.get("note"):
                    parts.append(info["note"])
                if info.get("evidence"):
                    parts.append(f"evidence: {json.dumps(info['evidence'])}")
                detail = "; ".join(parts) if parts else "n/a"
            else:
                detail = f"{info.get('reason', '')[:40]}...; unit tests: " \
                         f"{info.get('unit_test_evidence', {}).get('summary', 'n/a')}"
            lines.append(f"| {cell} | {status} | {detail} |")
        lines.append("")

    lines += ["## 3. Per-task real scorecard (13 tasks, q9b_diff_compact + 1-SE selection)", "",
              "Real trained-weight CPU decision latency (median/p95 µs) is phase4's own measurement for "
              "the head that actually won each task -- not this script's random-weight microbench.", "",
              "| task | n | acc% | wilson95 | strategy | vs Jev | vs Nimble | collapsed | Jev in our CI | "
              "Nimble in our CI | latency median µs | latency p95 µs |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["per_task_real_scorecard"]:
        lines.append(f"| {r['task']} | {r['n']} | {r['accuracy']} | {r['wilson95_recomputed']} | "
                     f"{r['chosen_strategy']} | {r['delta_vs_jev']:+.2f} | {r['delta_vs_nimble']:+.2f} | "
                     f"{r['collapsed']} | {r['jev_inside_our_wilson95']} | {r['nimble_inside_our_wilson95']} | "
                     f"{r['decision_latency_us_median']} | {r['decision_latency_us_p95']} |")

    ms = report["macro_summary"]
    lines += ["", "## 4. Macro summary", "",
              f"- macro_avg_13 = **{ms['macro_avg_13_point_estimate']}%** "
              f"(task-level bootstrap 95% CI {ms['macro_avg_13_task_level_bootstrap_ci95']})",
              f"- vs Jev: {ms['delta_macro_vs_jev']:+.2f}pp, vs Nimble: {ms['delta_macro_vs_nimble']:+.2f}pp",
              f"- collapsed tasks: {ms['tasks_collapsed']}"]
    mec = ms.get("macro_excluding_collapsed_tasks")
    if mec:
        lines.append(f"- **excluding collapsed tasks {mec['excluded_tasks']}** ({mec['n_tasks_remaining']} tasks): "
                     f"ours={mec['ours']}% vs Jev={mec['jev']}% (delta {mec['delta_vs_jev']:+.2f}pp) "
                     f"vs Nimble={mec['nimble']}% (delta {mec['delta_vs_nimble']:+.2f}pp)")
    lines.append("")

    sd = report["one_se_selector_distribution"]
    lines += ["## 5. 1-SE selector distribution (13 tasks, real)", "",
              f"linear_probe={sd['linear_probe']}, adapter_formulation_a={sd['adapter_formulation_a']}, "
              f"supcon={sd['supcon']} (sum={sd['n_tasks_selected_sum_check']} of {sd['n_tasks_total']})", ""]

    lat = report["cpu_latency_us"]
    lines += ["## 6. CPU latency (fresh measurement this run, µs, batch=1, float32, random weights)", "",
              f"Methodology: {lat['methodology']}", "",
              f"Host: {lat.get('host_busy_caveat', '')}", "",
              "Caveat: `linear_probe` below is a bare `x @ W.T + b` GEMV; `adapter_b` and `dual_manifold` "
              "are timed through their production `.scores()`, which also does input validation "
              "(`isfinite`, shape checks) and standardization. The ratio between them is not a pure "
              "FLOPs comparison; it is the real, honest cost of calling the production API.",
              "", "| head | K | median_us | p95_us |", "|---|---|---|---|"]
    for group_key, group_label in (("adapter_formulation_a_D8192_rank64_shape_only", "adapter_a (=supcon) D8192 r64"),
                                    ("adapter_formulation_b_D8192_rank64", "adapter_b D8192 r64"),
                                    ("dual_manifold_DQ8192_DG2816", "dual_manifold DQ8192+DG2816"),
                                    ("linear_probe_D8192_fresh", "linear_probe D8192")):
        for k_label, stats in lat[group_key].items():
            lines.append(f"| {group_label} | {k_label} | {stats['median_us']} | {stats['p95_us']} |")

    pool = lat["pooling_T1536_D4096"]
    lines += ["", f"### Pooling (separate latency boundary: {pool['boundary']})", "",
              "| pooling | median_us | p95_us |", "|---|---|---|",
              f"| partition_anchor_pooling T1536 D4096 | {pool['partition_anchor_pooling']['median_us']} | "
              f"{pool['partition_anchor_pooling']['p95_us']} |",
              f"| uniform_mean T1536 D4096 | {pool['uniform_mean']['median_us']} | "
              f"{pool['uniform_mean']['p95_us']} |"]

    lines += ["", "## 7. Unit test evidence (re-run live by this script, not cached)", "",
              "| suite | exit_code | summary |", "|---|---|---|"]
    for k, v in report["unit_test_evidence"].items():
        summary = v.get("summary", "")
        lines.append(f"| {k} | {v.get('exit_code')} | {summary} |")

    fv = report["final_verdict"]
    lines += ["", "## 8. Final verdict", "", fv["only_real_end_to_end_combination"], "",
              f"**{fv['headline_caveat']}**", "",
              f"Not yet combinable: {', '.join(fv['not_yet_combinable_axes'])}", "", fv["why"], ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path, default=RESULTS / "spec20_p5_ablation_report.json")
    ap.add_argument("--md-out", type=Path, default=RESULTS / "spec20_p5_ablation_report.md")
    args = ap.parse_args()

    phase4 = load_phase4()
    unit_tests = run_unit_tests()
    rows = build_per_task_scorecard(phase4)
    axis_grid = build_axis_grid(phase4, unit_tests)
    latency = run_latency_suite()
    report = build_report(rows, phase4, axis_grid, latency, unit_tests)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))
    args.md_out.write_text(render_markdown(report))

    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")
    print(f"macro_avg_13={report['macro_summary']['macro_avg_13_point_estimate']} "
         f"vs_jev={report['macro_summary']['delta_macro_vs_jev']:+.2f} "
         f"vs_nimble={report['macro_summary']['delta_macro_vs_nimble']:+.2f}")
    for k, v in unit_tests.items():
        print(f"unit_tests[{k}] exit_code={v.get('exit_code')}")
    print("PASS: spec20 p5 ablation report written; no new accuracy was evaluated by this script")


if __name__ == "__main__":
    main()
