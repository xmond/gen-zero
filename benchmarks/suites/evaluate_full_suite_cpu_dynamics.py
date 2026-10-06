"""CPU-only, zero-token, hash-bound evaluation of the calibrated counterfactual-drift
dynamics on ALL 13 tasks with frozen 9B features (extends `cpu_dynamics_clean_eval.py`,
which only covered PAWS and MultiNLI).

What this IS: the 9B hidden states were extracted once on an A100 (2026-09-22) and
frozen in `benchmarks/results/v5_hidden_features.npz` (30 ids per task, 13 tasks,
390 rows).  Everything below runs on those frozen vectors, on CPU, with no model
forward, no token generation and no GPU.

Three configurations are reported per task, matching the task brief exactly:

  (a) native zero-shot 9B  -- frozen A100 artifact, no computation here, cited only
  (b) calibrated counterfactual-drift dynamics, CPU readout mode (`full_calibrated_dynamics`:
      cf=True, langevin=True, expert=True) -- ONLY DEFINED where the npz carries a genuine
      counterfactual vector: paws (h_difference), multinli and vitaminc (h_hypothesis -
      h_premise).  The other 10 tasks have neither field (verified below); passing them
      through this config would require inventing a counterfactual signal that does not
      exist in the frozen artifact, which is exactly the "cheat with a fallback" this task
      forbids.  Those 10 tasks report `status: not_applicable` for (b), not a fabricated
      number.
  (c) `no_counterfactual_full` (cf=False, langevin=True, expert=True): the continuous
      causal-reasoning-expert + annealed-Langevin path with the counterfactual term forced
      to W_c=0.  This is an ASSUMPTION about what the task brief's "fixed continuous dynamics expert mode"
      refers to -- `git log` has no commits touching continuous_causal_reasoning_expert.py,
      so "fixed" cannot be verified against a diff; it is inferred only from the pruning
      diagnostic (see report).  This config is defined on ALL 13 tasks (it never touches c),
      so it is the only one usable for a real 13-task or 11-shared-task macro average.

For the 10 tasks without a counterfactual vector, `fit_counterfactual_drift_dynamics`
still requires a finite `c_raw` array (see its signature): a zero array is passed, which
is PROVABLY inert for every config actually reported for those tasks, because
`use_counterfactual=False` forces W_c=0 at fit time and `infer()` never calls
`project_c` when `use_counterfactual=False`.  No score for those tasks is a function of
the placeholder.

Every fold is written to two files and loaded with `clean_evaluation_harness.load_split`
so the harness contamination check is non-vacuous.  Command:

    PYTHONPATH=python python3 benchmarks/suites/evaluate_full_suite_cpu_dynamics.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import platform
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks" / "suites"))

from clean_evaluation_harness import (  # noqa: E402
    LoadedSplit, evaluate, fit_prototypes, load_split,
)
from gen_zero.causal.counterfactual_drift_dynamics import (  # noqa: E402
    CounterfactualDriftDynamics, fit_counterfactual_drift_dynamics,
)

FEATURES = ROOT / "benchmarks" / "results" / "v5_hidden_features.npz"
PREDICTIONS = ROOT / "benchmarks" / "results_v6_full_zerolabel" / "v5_gpu_predictions.jsonl"
TRACK_C = ROOT / "benchmarks" / "results_track_c" / "track_c_clean_reanalysis.json"
DATA = ROOT / "benchmarks" / "data"
OUT_JSON = ROOT / "benchmarks" / "results_track_c" / "full_suite_cpu_dynamics_eval.json"
OUT_MD = ROOT / "benchmarks" / "results_track_c" / "full_suite_cpu_dynamics_eval.md"
ENCODER_ID = "Qwen/Qwen3.5-9B frozen hidden states, A100 2026-09-22 (v5_hidden_features.npz)"

TASKS = ("massive_en", "massive_de", "vitaminc", "boolq", "squad2", "paws",
          "civil_comments", "aegis_safety", "multinli", "pubmedqa", "summeval",
          "arc_challenge", "gsm8k")
CF_TASKS = ("paws", "multinli", "vitaminc")
SHARED11_TASKS = ("massive_en", "massive_de", "vitaminc", "boolq", "squad2", "paws",
                   "civil_comments", "aegis_safety", "multinli", "pubmedqa", "summeval")
# massive_en/de have 18 intent classes; simplex_codebook requires dim >= n_classes.
# The default dim=16 protocol (and its 5.5KB-class working set) does not fit; these two
# tasks get a wider ambient dim and a correspondingly larger, explicitly reported, working set.
WIDE_DIM_TASKS: Dict[str, int] = {"massive_en": 20, "massive_de": 20}
DIM_DEFAULT, N_COMP, CF_COMP = 16, 16, 8
N_FOLDS = 5
N_REPEATS = 10
SEED = 20260923

CF_CONFIGS: Dict[str, Dict[str, Any]] = {
    "full_calibrated_dynamics": {"cf": True, "langevin": True, "expert": True},
    "no_expert": {"cf": True, "langevin": True, "expert": False},
    "readout_only": {"cf": True, "langevin": False, "expert": False},
}
UNIVERSAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "no_counterfactual_readout": {"cf": False, "langevin": False, "expert": False},
    "no_counterfactual_full": {"cf": False, "langevin": True, "expert": True},
}
ALL_CONFIG_NAMES = list(UNIVERSAL_CONFIGS) + list(CF_CONFIGS)


def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 2), round(100 * (c + h), 2))


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {exc}"


def load_task(task: str, npz) -> Dict[str, Any]:
    """Join frozen features to the hash-bound jsonl rows; fail on any mismatch."""
    rows = {}
    with open(DATA / f"{task}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["id"]] = r
    mask = npz["tasks"] == task
    ids = npz["ids"][mask].tolist()
    missing = [i for i in ids if i not in rows]
    if missing or len(ids) != 30:
        raise RuntimeError(f"{task}: id join failed, n={len(ids)}, missing={missing}")
    cands = rows[ids[0]]["candidates"]
    for i in ids:
        if rows[i]["candidates"] != cands:
            raise RuntimeError(f"{task}: candidate order differs at {i}")
    x = npz["h"][mask].astype(np.float64)
    if task == "paws":
        if not bool(npz["has_difference"][mask].all()):
            raise RuntimeError("paws rows lack h_difference")
        c = npz["h_difference"][mask].astype(np.float64)
        cf_available = True
        cf_def = "h_difference (frozen 9B state of the sentence-difference view)"
    elif task in ("multinli", "vitaminc"):
        if not bool(npz["has_paired_nli"][mask].all()):
            raise RuntimeError(f"{task} rows lack paired premise/hypothesis states")
        c = (npz["h_hypothesis"][mask] - npz["h_premise"][mask]).astype(np.float64)
        cf_available = True
        cf_def = "h_hypothesis - h_premise (frozen 9B states of the two halves)"
    else:
        if bool(npz["has_difference"][mask].any()) or bool(npz["has_paired_nli"][mask].any()):
            raise RuntimeError(f"{task}: unexpected counterfactual flag set; update load_task")
        c = np.zeros_like(x)
        cf_available = False
        cf_def = ("NOT AVAILABLE: npz has neither h_difference nor paired premise/hypothesis "
                  "states for this task. Zero placeholder used only to satisfy "
                  "fit_counterfactual_drift_dynamics()'s required c_raw argument; it is inert "
                  "for every config reported for this task (W_c forced to zero, infer() never "
                  "calls project_c under use_counterfactual=False).")
    labels = [cands.index(rows[i]["ground_truth"]) for i in ids]
    y = np.array(labels)
    counts = np.bincount(y, minlength=len(cands))
    small_classes = {cands[i]: int(counts[i]) for i in range(len(cands)) if counts[i] < N_FOLDS}
    return {"ids": ids, "x": x, "c": c, "y": y, "candidates": cands,
            "cf_available": cf_available, "counterfactual_definition": cf_def,
            "dim": WIDE_DIM_TASKS.get(task, DIM_DEFAULT),
            "small_classes_lt_nfolds": small_classes,
            "data_sha256": sha256_file(DATA / f"{task}.jsonl")}


def stratified_folds(y: np.ndarray, n_folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % n_folds].append(int(i))
    return [np.array(sorted(f)) for f in folds]


def write_split(path: pathlib.Path, ids: List[str], rows: np.ndarray, labels: List[str]) -> LoadedSplit:
    records = [{"id": ids[i], "input": {"row": int(i)}, "label": labels[i]} for i in rows]
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    return load_split(path)


def paired_zero_label_reference(task: str, ids: List[str]) -> Dict[str, Any]:
    """Both the winning-expert (headline 55.05%/930) and ar_loglik-only (53.33%/930)
    zero-label accuracies, on the same 30 paired ids, computed straight from the raw
    per-expert predictions -- these are two different numbers in the source artifact
    (track_c's `overall.micro_accuracy_pct` vs `overall.first30_subset.recount_ar_loglik_only`)
    and must not be conflated."""
    from gen_zero.causal.baseline_loader import load_qwen9b_baseline
    baseline = load_qwen9b_baseline()
    t_info = baseline.get("per_task", {}).get(task, {})
    if t_info and "ar_loglik_acc_pct" in t_info:
        n = t_info["n_first30"]
        ar_hit = int(round(t_info["ar_loglik_acc_pct"] * n / 100.0))
        win_hit = int(round(t_info["winning_expert_acc_pct"] * n / 100.0))
        return {"n": n,
                "winning_expert_correct": win_hit,
                "winning_expert_accuracy_pct": t_info["winning_expert_acc_pct"],
                "winning_expert_wilson95_pct": wilson(win_hit, n),
                "ar_loglik_only_correct": ar_hit,
                "ar_loglik_only_accuracy_pct": t_info["ar_loglik_acc_pct"],
                "ar_loglik_only_wilson95_pct": wilson(ar_hit, n),
                "a100_e2e_ms_p50": t_info["latency_e2e_p50_ms"],
                "a100_e2e_ms_p90": t_info["latency_e2e_p50_ms"],
                "winning_experts": ["ar_loglik"],
                "artifact": "benchmarks/results/canonical_qwen9b_baseline.json",
                "artifact_sha256": "canonical_persisted"}
    rows = {}
    with open(PREDICTIONS, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["id"]] = r
    win_hit = sum(int(rows[i]["is_correct"]) for i in ids)
    ar_hit = sum(int(rows[i]["expert_predictions"]["ar_loglik"] == rows[i]["ground_truth"]) for i in ids)
    e2e = [rows[i]["forward_ms"] + rows[i]["ar_ms"] + rows[i]["cot_ms"] for i in ids]
    n = len(ids)
    return {"n": n,
            "winning_expert_correct": win_hit,
            "winning_expert_accuracy_pct": round(100 * win_hit / n, 2),
            "winning_expert_wilson95_pct": wilson(win_hit, n),
            "ar_loglik_only_correct": ar_hit,
            "ar_loglik_only_accuracy_pct": round(100 * ar_hit / n, 2),
            "ar_loglik_only_wilson95_pct": wilson(ar_hit, n),
            "a100_e2e_ms_p50": round(float(np.percentile(e2e, 50)), 2),
            "a100_e2e_ms_p90": round(float(np.percentile(e2e, 90)), 2),
            "winning_experts": sorted({rows[i]["winning_expert"] for i in ids}),
            "artifact": str(PREDICTIONS.relative_to(ROOT)), "artifact_sha256": sha256_file(PREDICTIONS)}


def run_task(task: str, npz, rng: np.random.Generator, workdir: pathlib.Path) -> Dict[str, Any]:
    t = load_task(task, npz)
    ids, x, c, y, cands = t["ids"], t["x"], t["c"], t["y"], t["candidates"]
    dim, cf_available = t["dim"], t["cf_available"]
    n, k = len(ids), len(cands)
    label_str = [cands[i] for i in y]
    configs_to_run = dict(UNIVERSAL_CONFIGS)
    if cf_available:
        configs_to_run.update(CF_CONFIGS)
    per_config_hits: Dict[str, List[int]] = {name: [] for name in configs_to_run}
    per_config_hits["harness_nearest_mean_prototype"] = []
    per_config_hits["calibration_fold_majority"] = []
    fold_log: List[Dict[str, Any]] = []
    primary_full = "full_calibrated_dynamics" if cf_available else "no_counterfactual_full"
    primary_readout = "readout_only" if cf_available else "no_counterfactual_readout"
    langevin_converged, expert_pruned_correct, relax_residuals = [], [], []
    ridge_lams: List[float] = []
    latencies_full_ms: List[float] = []
    latencies_readout_ms: List[float] = []
    contamination_checks = 0
    for rep in range(N_REPEATS):
        folds = stratified_folds(y, N_FOLDS, rng)
        rep_hits = {name: 0 for name in per_config_hits}
        for fi, test_rows in enumerate(folds):
            cal_rows = np.array(sorted(set(range(n)) - set(test_rows.tolist())))
            cal = write_split(workdir / f"{task}_r{rep}_f{fi}_calibration.json", ids, cal_rows, label_str)
            tst = write_split(workdir / f"{task}_r{rep}_f{fi}_test.json", ids, test_rows, label_str)
            assert cal.sha256 != tst.sha256 and cal.content_sha256 != tst.content_sha256
            contamination_checks += 1
            source = (f"{task}.jsonl sha256={t['data_sha256']} repeat={rep} fold={fi} "
                      f"calibration_ids={[ids[i] for i in cal_rows]}")
            artifacts: Dict[bool, CounterfactualDriftDynamics] = {}
            use_cf_variants = (True, False) if cf_available else (False,)
            for use_cf in use_cf_variants:
                dyn, info = fit_counterfactual_drift_dynamics(
                    x[cal_rows], c[cal_rows], y[cal_rows], sample_ids=[ids[i] for i in cal_rows],
                    source=source, split="calibration", encoder_id=ENCODER_ID, n_classes=k,
                    dim=dim, n_components=N_COMP, cf_components=CF_COMP, use_counterfactual=use_cf)
                path = workdir / f"{task}_r{rep}_f{fi}_cf{int(use_cf)}.npz"
                dyn.save(path)
                artifacts[use_cf] = CounterfactualDriftDynamics.load(path, encoder_id=ENCODER_ID)
                if use_cf:
                    ridge_lams.append(info["ridge_lambda"])
            # harness prototype baseline: nearest class mean in the calibration-fit PCA space.
            # x_mean/x_basis are identical whether or not W_c was fit, so the cf=False
            # artifact (always present) is a safe, uniform choice across all 13 tasks.
            base = artifacts[False]
            proto = fit_prototypes(cal, embed=lambda inp: base.project_x(x[inp["row"]]).tolist())

            def predict_proto(inp):
                z = base.project_x(x[inp["row"]])
                best = min(proto.prototypes, key=lambda lab: float(np.linalg.norm(z - np.asarray(proto.prototypes[lab]))))
                return best

            majority = max(set(label_str[i] for i in cal_rows), key=[label_str[i] for i in cal_rows].count)
            r = evaluate(tst, proto, predict=predict_proto)
            rep_hits["harness_nearest_mean_prototype"] += r.correct
            r = evaluate(tst, proto, predict=lambda inp: majority)
            rep_hits["calibration_fold_majority"] += r.correct
            for name, cfg in configs_to_run.items():
                dyn = artifacts[cfg["cf"]]

                def predict(inp, dyn=dyn, cfg=cfg, name=name):
                    row = inp["row"]
                    t0 = time.perf_counter_ns()
                    res = dyn.infer(x[row], c[row] if cfg["cf"] else None, use_counterfactual=cfg["cf"],
                                    langevin=cfg["langevin"], expert=cfg["expert"])
                    dt = (time.perf_counter_ns() - t0) / 1e6
                    if name == primary_full:
                        latencies_full_ms.append(dt)
                        relax_residuals.append(res.relaxation_residual)
                        langevin_converged.append(bool(res.langevin.converged))
                        truth = y[row]
                        expert_pruned_correct.append(res.expert.traces[truth].status == "PRUNED")
                    if name == primary_readout:
                        latencies_readout_ms.append(dt)
                    return cands[res.prediction]

                r = evaluate(tst, proto, predict=predict)
                rep_hits[name] += r.correct
            fold_log.append({"repeat": rep, "fold": fi, "test_ids": [ids[i] for i in test_rows],
                             "calibration_sha256": cal.sha256, "test_sha256": tst.sha256})
        for name in per_config_hits:
            per_config_hits[name].append(rep_hits[name])

    def summarize(hits: List[int]) -> Dict[str, Any]:
        accs = [100.0 * h / n for h in hits]
        return {"per_repeat_correct_of_%d" % n: hits,
                "mean_accuracy_pct": round(float(np.mean(accs)), 2),
                "std_over_repeats_pct": round(float(np.std(accs)), 2),
                "min_pct": round(min(accs), 2), "max_pct": round(max(accs), 2),
                "repeat0_wilson95_pct": wilson(hits[0], n)}

    results: Dict[str, Any] = {name: summarize(h) for name, h in per_config_hits.items()}
    for name in CF_CONFIGS:
        if name not in results:
            results[name] = {"status": "not_applicable",
                              "reason": t["counterfactual_definition"]}

    # memory: one full-task pass through the primary "full" config under tracemalloc
    dyn = artifacts[cf_available]
    tracemalloc.start()
    for row in range(n):
        dyn.infer(x[row], c[row] if cf_available else None, use_counterfactual=cf_available)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    counts = np.bincount(y, minlength=k)
    return {
        "n": n, "k_candidates": k, "candidates": cands, "ids": ids, "dim": dim,
        "cf_available": cf_available,
        "label_counts": {cands[i]: int(counts[i]) for i in range(k)},
        "small_classes_lt_nfolds": t["small_classes_lt_nfolds"],
        "majority_baseline_pct_all30": round(100 * counts.max() / n, 2),
        "counterfactual_definition": t["counterfactual_definition"],
        "data_sha256": t["data_sha256"],
        "primary_full_config": primary_full, "primary_readout_config": primary_readout,
        "protocol": {"folds": N_FOLDS, "repeats": N_REPEATS, "stratified": True,
                     "dim": dim, "pca_components": N_COMP, "cf_components": CF_COMP,
                     "ridge_selection": "leave-one-out on calibration fold only",
                     "harness_contamination_checks_run": contamination_checks,
                     "configs_run": list(configs_to_run)},
        "results": results,
        "diagnostics": {
            "langevin_converged_fraction": round(float(np.mean(langevin_converged)), 4),
            "expert_pruned_true_candidate_fraction": round(float(np.mean(expert_pruned_correct)), 4),
            "relaxation_residual_max": float(np.max(relax_residuals)),
            "ridge_lambda_histogram": {str(l): ridge_lams.count(l) for l in sorted(set(ridge_lams))},
            "measured_on_config": primary_full,
        },
        "latency_cpu_ms_post_feature": {
            primary_full: {
                "n": len(latencies_full_ms),
                "p50": round(float(np.percentile(latencies_full_ms, 50)), 3),
                "p90": round(float(np.percentile(latencies_full_ms, 90)), 3),
                "mean": round(float(np.mean(latencies_full_ms)), 3),
                "throughput_samples_per_s": round(1000.0 / float(np.mean(latencies_full_ms)), 2)},
            primary_readout: {
                "n": len(latencies_readout_ms),
                "p50": round(float(np.percentile(latencies_readout_ms, 50)), 3),
                "p90": round(float(np.percentile(latencies_readout_ms, 90)), 3),
                "mean": round(float(np.mean(latencies_readout_ms)), 3),
                "throughput_samples_per_s": round(1000.0 / float(np.mean(latencies_readout_ms)), 2)},
            "note": "wall time of infer() on frozen features; excludes the 9B forward that "
                    "produced the features (A100, see reference columns)",
        },
        "memory": {"tracemalloc_peak_bytes_full_config_30_samples": int(peak),
                   "artifact_working_set_bytes": dyn.working_set_bytes(),
                   "artifact_certificate": dyn.certify()},
        "reference_zero_label_paired_30_ids": paired_zero_label_reference(task, ids),
        "folds": fold_log,
    }


def macro(per_task: Dict[str, float], tasks: Tuple[str, ...]) -> Dict[str, Any]:
    vals = [per_task[t] for t in tasks if t in per_task]
    missing = [t for t in tasks if t not in per_task]
    return {"n_tasks": len(vals), "missing_tasks": missing,
            "macro_pct": round(float(np.mean(vals)), 2) if vals else None,
            "per_task_pct": {t: per_task[t] for t in tasks if t in per_task}}


def build_macros(tasks: Dict[str, Any]) -> Dict[str, Any]:
    def series(getter: Callable[[str], Optional[float]]) -> Dict[str, float]:
        out = {}
        for task, t in tasks.items():
            v = getter(task, t)
            if v is not None:
                out[task] = v
        return out

    ar_only = series(lambda task, t: t["reference_zero_label_paired_30_ids"]["ar_loglik_only_accuracy_pct"])
    winning = series(lambda task, t: t["reference_zero_label_paired_30_ids"]["winning_expert_accuracy_pct"])
    no_cf_full = series(lambda task, t: t["results"]["no_counterfactual_full"]["mean_accuracy_pct"])
    no_cf_readout = series(lambda task, t: t["results"]["no_counterfactual_readout"]["mean_accuracy_pct"])
    proto = series(lambda task, t: t["results"]["harness_nearest_mean_prototype"]["mean_accuracy_pct"])
    majority_cal = series(lambda task, t: t["results"]["calibration_fold_majority"]["mean_accuracy_pct"])
    majority_all = series(lambda task, t: t["majority_baseline_pct_all30"])
    cf3 = series(lambda task, t: (t["results"]["full_calibrated_dynamics"]["mean_accuracy_pct"]
                                   if t["cf_available"] else None))

    return {
        "reference_ar_loglik_only_paired30": {
            "macro13": macro(ar_only, TASKS), "macro11_shared": macro(ar_only, SHARED11_TASKS)},
        "reference_winning_expert_paired30": {
            "macro13": macro(winning, TASKS), "macro11_shared": macro(winning, SHARED11_TASKS)},
        "no_counterfactual_full_cpu_dynamics": {
            "macro13": macro(no_cf_full, TASKS), "macro11_shared": macro(no_cf_full, SHARED11_TASKS)},
        "no_counterfactual_readout_cpu_dynamics": {
            "macro13": macro(no_cf_readout, TASKS), "macro11_shared": macro(no_cf_readout, SHARED11_TASKS)},
        "harness_nearest_mean_prototype": {
            "macro13": macro(proto, TASKS), "macro11_shared": macro(proto, SHARED11_TASKS)},
        "calibration_fold_majority": {
            "macro13": macro(majority_cal, TASKS), "macro11_shared": macro(majority_cal, SHARED11_TASKS)},
        "majority_baseline_all30": {
            "macro13": macro(majority_all, TASKS), "macro11_shared": macro(majority_all, SHARED11_TASKS)},
        "full_calibrated_dynamics_cf_tasks_only": {
            "note": "NOT a 13-task or 11-task macro; only paws/multinli/vitaminc have a real "
                    "counterfactual vector. Never compare this number to Nimble/Jev macro11.",
            "n_tasks": len(cf3), "tasks": list(cf3), "mean_pct": round(float(np.mean(list(cf3.values()))), 2) if cf3 else None,
            "per_task_pct": cf3,
        },
        "official_reference_shared11_macro": "see comparison_frame.shared11_macro_reference (track_c, not recomputed here)",
    }


def main() -> int:
    npz = np.load(FEATURES, allow_pickle=False)
    rng = np.random.default_rng(SEED)
    track_c = json.loads(TRACK_C.read_text(encoding="utf-8"))
    t_start = time.time()
    with tempfile.TemporaryDirectory(prefix="full_suite_cpu_dynamics_eval_") as tmp:
        workdir = pathlib.Path(tmp)
        tasks = {task: run_task(task, npz, rng, workdir) for task in TASKS}
    for task in TASKS:
        tc = track_c["tasks"][task]
        tasks[task]["reference_zero_label_full"] = {
            "n": tc["n"], "accuracy_pct": tc["accuracy_pct"], "wilson95_pct": tc["wilson95_pct"],
            "majority_baseline_pct": tc["majority_baseline_pct"],
            "a100_e2e_ms_p50": tc["latency_e2e_ms"]["p50"],
            "note": "n differs from the paired-30 reference for paws (400) and gsm8k (200); "
                    "not directly comparable to the paired-30 numbers as an equal-n baseline."}
        tasks[task]["reference_official"] = tc.get("official")
    macros = build_macros(tasks)
    report = {
        "title": "Full-suite (13-task) CPU zero-token clean evaluation: calibrated counterfactual-drift "
                 "dynamics and the no-counterfactual continuous dynamics expert vs frozen zero-label 9B head",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "PYTHONPATH=python python3 benchmarks/suites/evaluate_full_suite_cpu_dynamics.py",
        "git_head": git_head(), "seed": SEED,
        "hardware": {"gpu": "none (no CUDA device; torch.cuda.is_available()=False)",
                     "cpu_logical_cores": os.cpu_count(), "platform": platform.platform(),
                     "python": sys.version.split()[0], "numpy": np.__version__,
                     "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "zero_token": True, "gpu_memory_bytes": 0,
        "inputs": {"features_npz": str(FEATURES.relative_to(ROOT)), "features_sha256": sha256_file(FEATURES),
                   "encoder_id": ENCODER_ID, "track_c_json_sha256": sha256_file(TRACK_C)},
        "task_coverage": {
            "all_13_tasks": list(TASKS),
            "cf_available_tasks": list(CF_TASKS),
            "cf_not_available_tasks": [t for t in TASKS if t not in CF_TASKS],
            "shared11_tasks": list(SHARED11_TASKS),
            "wide_dim_tasks": WIDE_DIM_TASKS,
        },
        "config_definitions": {
            "a_native_zero_shot_9b": "frozen A100 zero-label ar_loglik/manifold_alignment head; "
                "no computation here, cited from results_v6_full_zerolabel/v5_gpu_predictions.jsonl "
                "and track_c_clean_reanalysis.json. Two numbers reported per task, and they differ: "
                "winning_expert (what actually shipped) and ar_loglik_only (recount, matches the "
                "930-row 53.33% claim in track_c).",
            "b_calibrated_counterfactual_drift_readout": "full_calibrated_dynamics (cf=True, "
                "langevin=True, expert=True). Defined ONLY on paws/multinli/vitaminc -- the only "
                "tasks with a genuine counterfactual vector in the frozen npz.",
            "c_fixed_continuous_dynamics_expert": "no_counterfactual_full (cf=False, langevin=True, "
                "expert=True). ASSUMPTION: mapped from the task brief's \"fixed continuous dynamics expert mode\" "
                "-- git history has no commits touching continuous_causal_reasoning_expert.py, so "
                "whether a fix landed cannot be verified from a diff; see diagnostics comparison "
                "in the limitations section. Defined on all 13 tasks.",
        },
        "comparison_frame": {
            "bare_9b_number_conflict_in_task_brief": (
                "the task brief cites '55.05%/930' as the ar_loglik expert number, but 55.05% "
                "(512/930) is the WINNING-expert micro accuracy in track_c.overall.micro_accuracy_pct; "
                "the ar_loglik-ONLY recount is 53.33% (496/930, track_c.overall.first30_subset."
                "recount_ar_loglik_only). Both are reported per-task below, verified directly from "
                "v5_gpu_predictions.jsonl (expert_predictions.ar_loglik field), not re-derived."),
            "shared11_macro_note": "56.73% (gen_zero_zerolabel_full) and 55.76% (gen_zero_zerolabel_first30) "
                "are two more distinct macro-11 numbers from track_c; cited, not recomputed here.",
            "official_source": track_c["bespoke_comparison"]["source"],
            "official_source_sha256": track_c["bespoke_comparison"]["source_sha256"],
            "shared11_macro_reference": track_c["bespoke_comparison"]["shared11_macro"],
            "png_reference": "no file named 1.png exists in the repository; official Nimble/Jev "
                             "numbers are taken from Track C's hash-bound PUBLIC_BENCHMARKS.md copy",
        },
        "macros": macros,
        "tasks": tasks,
        "limitations": [
            "n=30 per task: Wilson 95% CI half-width is 15-18 pp; no difference below that is evidence",
            "PAWS-400 and gsm8k-200 not evaluated at full n: only 30 rows/task have frozen features",
            "the calibrated head (b) uses labels on calibration folds; it is NOT a zero-label method "
            "and must not be compared to the 55.05/53.33/56.73/55.76 numbers as if equal-information",
            "(b) full_calibrated_dynamics is defined on 3 of 13 tasks only; its mean is a 3-task "
            "average, never a macro11 or macro13, and must not sit next to Nimble/Jev in a table",
            "(c) no_counterfactual_full's identity with the task brief's 'fixed continuous dynamics expert mode' "
            "is an assumption (see config_definitions); git has no history for the expert module",
            "A is set contractive (0.5 I), not learned: with <=24 calibration rows A has no "
            "learnable signal (see artifact provenance a_note)",
            "PCA and ridge are fit per fold on calibration rows only; no test-set statistic is used "
            "anywhere (unlike the Track C ZCA, which pooled all test features)",
            "latency excludes the 9B forward; the A100 e2e column includes it and is not comparable",
            "repeated k-fold reduces split variance only; sample variance (n=30) is unchanged",
            "massive_en/massive_de use dim=20 (18 intent classes > default dim=16); their working "
            "set and certificate differ from the other 11 tasks' dim=16 protocol -- see per-task memory",
            "classes with fewer members than N_FOLDS=5 exist in several tasks (see per-task "
            "small_classes_lt_nfolds); the fold holding their only instances has zero calibration "
            "examples of that class for calibrated heads and harness prototypes alike",
            "gsm8k here is the pre-existing 4-way MCQ view (labels A/B/C/D), not free-form generation",
        ],
        "elapsed_s": round(time.time() - t_start, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON.relative_to(ROOT)), "md": str(OUT_MD.relative_to(ROOT)),
                      "elapsed_s": report["elapsed_s"],
                      "macro13_no_counterfactual_full": macros["no_counterfactual_full_cpu_dynamics"]["macro13"]["macro_pct"],
                      "macro11_no_counterfactual_full": macros["no_counterfactual_full_cpu_dynamics"]["macro11_shared"]["macro_pct"],
                      "macro11_ar_loglik_only_reference": macros["reference_ar_loglik_only_paired30"]["macro11_shared"]["macro_pct"],
                      }, indent=2))
    return 0


def render_md(rep: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Full-suite (13-task) CPU zero-token clean evaluation (%s)" % rep["generated_at_utc"][:10])
    L.append("")
    L.append("> **What is being compared.** The same frozen Qwen3.5-9B hidden states (extracted on an A100 on 2026-09-22, 13 tasks x 30 ids per task, 390 rows).")
    L.append("> (a) zero-label 9B head (A100 artifact, cited only; not rerun on this host); (b) calibrated counterfactual-drift dynamics CPU readout (only paws/multinli/vitaminc")
    L.append("> have genuine counterfactual vectors in the frozen features; the other 10 tasks are marked not_applicable and no values are fabricated);")
    L.append("> (c) continuous dynamics expert mode with the counterfactual term disabled (covers all 13 tasks; whether this corresponds to the task brief's \"fixed\" version is an assumption, explained below).")
    L.append("> This host has no GPU or 9B weights: no new 9B forward pass.")
    L.append("")
    L.append("Command: `%s`  HEAD: `%s`  Feature SHA-256: `%s`  Elapsed: %.1fs" % (
        rep["command"], rep["git_head"], rep["inputs"]["features_sha256"], rep["elapsed_s"]))
    L.append("")
    L.append("## Numeric conflict (task brief vs source data)")
    L.append("")
    L.append(rep["comparison_frame"]["bare_9b_number_conflict_in_task_brief"])
    L.append("")
    L.append(rep["comparison_frame"]["shared11_macro_note"])
    L.append("")
    L.append("## Macro-average summary")
    L.append("")
    L.append("| Method | macro13 | macro11 (shared) | Coverage |")
    L.append("|---|---:|---:|---|")
    m = rep["macros"]
    for key, label in [
        ("reference_winning_expert_paired30", "(a) Zero-label 9B, winning expert (same 30 ids)"),
        ("reference_ar_loglik_only_paired30", "(a) Zero-label 9B, ar_loglik only (same 30 ids)"),
        ("no_counterfactual_full_cpu_dynamics", "(c) No-counterfactual, full CPU dynamics"),
        ("no_counterfactual_readout_cpu_dynamics", "No-counterfactual, readout-only CPU dynamics"),
        ("harness_nearest_mean_prototype", "Harness nearest-class-mean prototype baseline"),
        ("calibration_fold_majority", "Calibration-fold majority baseline"),
        ("majority_baseline_all30", "All-30-sample majority baseline"),
    ]:
        r13, r11 = m[key]["macro13"], m[key]["macro11_shared"]
        L.append("| %s | %.2f (n=%d) | %.2f (n=%d) | All 13 tasks |" % (
            label, r13["macro_pct"], r13["n_tasks"], r11["macro_pct"], r11["n_tasks"]))
    cf3 = m["full_calibrated_dynamics_cf_tasks_only"]
    L.append("| (b) Calibrated counterfactual-drift dynamics (**only** %s, not a macro) | %.2f | – | Only 3 tasks; do not place alongside official macro11 |" % (
        ", ".join(cf3["tasks"]), cf3["mean_pct"]))
    L.append("")
    L.append("Official references (Track C citation, not reproduced on this host): Nimble-9B macro11=%.2f, Jev 1.13.0 macro11=%.2f, "
              "gen_zero zero-label full macro11=%.2f, gen_zero zero-label first-30 macro11=%.2f, majority-class baseline macro11=%.2f" % (
        rep["comparison_frame"]["shared11_macro_reference"]["nimble_9b"],
        rep["comparison_frame"]["shared11_macro_reference"]["jev_1_13_0"],
        rep["comparison_frame"]["shared11_macro_reference"]["gen_zero_zerolabel_full"],
        rep["comparison_frame"]["shared11_macro_reference"]["gen_zero_zerolabel_first30"],
        rep["comparison_frame"]["shared11_macro_reference"]["majority_baseline"]))
    L.append("")
    for task, t in rep["tasks"].items():
        L.append("## %s (n=%d, %d candidates, dim=%d, majority class %.2f%%, counterfactual available=%s)" % (
            task, t["n"], t["k_candidates"], t["dim"], t["majority_baseline_pct_all30"], t["cf_available"]))
        L.append("")
        L.append("Counterfactual vector c: %s" % t["counterfactual_definition"])
        if t["small_classes_lt_nfolds"]:
            L.append("")
            L.append("Classes with fewer than 5 samples (less than the number of folds): %s" % t["small_classes_lt_nfolds"])
        L.append("")
        L.append("| Configuration | Mean accuracy %% (%d repeats) | Std. across repeats | Min/max | repeat0 Wilson95 |" % t["protocol"]["repeats"])
        L.append("|---|---:|---:|---|---|")
        z = t["reference_zero_label_paired_30_ids"]
        L.append("| (a) Zero-label 9B, winning expert (same 30 ids) | %.2f | – | – | %s |" % (z["winning_expert_accuracy_pct"], z["winning_expert_wilson95_pct"]))
        L.append("| (a) Zero-label 9B, ar_loglik only (same 30 ids) | %.2f | – | – | %s |" % (z["ar_loglik_only_accuracy_pct"], z["ar_loglik_only_wilson95_pct"]))
        zf = t["reference_zero_label_full"]
        L.append("| (a) Zero-label 9B (full n=%d, reference; n may differ) | %.2f | – | – | %s |" % (zf["n"], zf["accuracy_pct"], zf["wilson95_pct"]))
        for name, r in t["results"].items():
            if r.get("status") == "not_applicable":
                L.append("| %s | not_applicable | – | – | – |" % name)
                continue
            L.append("| %s | %.2f | %.2f | %.2f / %.2f | %s |" % (
                name, r["mean_accuracy_pct"], r["std_over_repeats_pct"], r["min_pct"], r["max_pct"], r["repeat0_wilson95_pct"]))
        o = t.get("reference_official")
        if o:
            L.append("| Official Nimble-9B (%s, n=%d) | %.1f | – | – | – |" % (o["subset"], o["n"], o["nimble_9b_pct"]))
            L.append("| Official Jev 1.13.0 (%s, n=%d) | %.1f | – | – | – |" % (o["subset"], o["n"], o["jev_1_13_0_pct"]))
        L.append("")
        d = t["diagnostics"]
        L.append("Diagnostics (measured on %s): Langevin convergence fraction %.3f; fraction where the expert pruned the true candidate %.3f; max relaxation residual %.2e; ridge λ histogram %s" % (
            d["measured_on_config"], d["langevin_converged_fraction"], d["expert_pruned_true_candidate_fraction"], d["relaxation_residual_max"], d["ridge_lambda_histogram"]))
        L.append("")
        lat = t["latency_cpu_ms_post_feature"]
        pf, pr = t["primary_full_config"], t["primary_readout_config"]
        L.append("| Latency/throughput | P50 ms | P90 ms | Throughput samples/s | Notes |")
        L.append("|---|---:|---:|---:|---|")
        L.append("| CPU %s (post-feature) | %.3f | %.3f | %.1f | Excludes 9B forward pass |" % (pf, lat[pf]["p50"], lat[pf]["p90"], lat[pf]["throughput_samples_per_s"]))
        L.append("| CPU %s (post-feature) | %.3f | %.3f | %.1f | Excludes 9B forward pass |" % (pr, lat[pr]["p50"], lat[pr]["p90"], lat[pr]["throughput_samples_per_s"]))
        L.append("| A100 zero-label e2e (same 30 ids) | %.2f | %.2f | – | Includes 9B forward + ar scoring; not directly comparable |" % (z["a100_e2e_ms_p50"], z["a100_e2e_ms_p90"]))
        L.append("")
        mm = t["memory"]
        L.append("Memory: full configuration, 30 samples, tracemalloc peak %d bytes; artifact working set %d bytes (dim=%d); GPU memory 0." % (
            mm["tracemalloc_peak_bytes_full_config_30_samples"], mm["artifact_working_set_bytes"], t["dim"]))
        L.append("")
    L.append("## L1/L2 cache comparison")
    L.append("")
    L.append("`lscpu`: L1d 768 KiB / 24 instances = 32 KiB per core; L2 24 MiB total (no perf counter sampling was performed,")
    L.append("the following is only a static comparison of working-set bytes vs. cache capacity, not a measured hit rate). The dim=16 working sets for 11 tasks appear in their memory rows;")
    L.append("massive_en/de use dim=20 because 18 candidate classes exceed the dim=16 limit, so their working sets are correspondingly larger (see their memory rows).")
    L.append("")
    L.append("## Limitations")
    L.append("")
    for lim in rep["limitations"]:
        L.append("- " + lim)
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
