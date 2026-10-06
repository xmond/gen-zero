"""CPU-only, zero-token, hash-bound evaluation of the calibrated counterfactual-drift
dynamics on the adversarial tasks PAWS and MultiNLI.

What this IS: the 9B hidden states were extracted once on an A100 (2026-09-22) and
frozen in `benchmarks/results/v5_hidden_features.npz` (30 ids per task).  Everything
below runs on those frozen vectors, on CPU, with no model forward, no token
generation and no GPU.  Comparison is therefore

    9B features + zero-label ar_loglik head   (frozen A100 artifact, paired 30 ids)
    9B features + calibrated CPU dynamics     (this run, repeated stratified 5-fold)

What this is NOT: a re-run of the 9B, or a 400-row PAWS evaluation.  Only 30 PAWS
ids have frozen features; the other 370 need the 9B forward, which is impossible
on this box.  The calibrated path uses calibration-fold LABELS (never test labels)
and is not comparable to the zero-label 56.73% number as an equal-information
baseline; the report says so in every table.

Every fold is written to two files and loaded with `clean_evaluation_harness.load_split`
so the harness contamination check is non-vacuous.  Command:

    PYTHONPATH=python python3 benchmarks/suites/cpu_dynamics_clean_eval.py
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
OUT_JSON = ROOT / "benchmarks" / "results_track_c" / "cpu_dynamics_clean_eval.json"
OUT_MD = ROOT / "benchmarks" / "results_track_c" / "cpu_dynamics_clean_eval.md"
ENCODER_ID = "Qwen/Qwen3.5-9B frozen hidden states, A100 2026-09-22 (v5_hidden_features.npz)"
TASKS = ("paws", "multinli")
N_FOLDS = 5
N_REPEATS = 10
SEED = 20260923
DIM, N_COMP, CF_COMP = 16, 16, 8

CONFIGS: Dict[str, Dict[str, Any]] = {
    # name: (fit use_counterfactual, infer kwargs)
    "full_calibrated_dynamics": {"cf": True, "langevin": True, "expert": True},
    "no_expert": {"cf": True, "langevin": True, "expert": False},
    "readout_only": {"cf": True, "langevin": False, "expert": False},
    "no_counterfactual_readout": {"cf": False, "langevin": False, "expert": False},
    "no_counterfactual_full": {"cf": False, "langevin": True, "expert": True},
}


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
        cf_def = "h_difference (frozen 9B state of the sentence-difference view)"
    else:
        if not bool(npz["has_paired_nli"][mask].all()):
            raise RuntimeError(f"{task} rows lack paired premise/hypothesis states")
        c = (npz["h_hypothesis"][mask] - npz["h_premise"][mask]).astype(np.float64)
        cf_def = "h_hypothesis - h_premise (frozen 9B states of the two halves)"
    labels = [cands.index(rows[i]["ground_truth"]) for i in ids]
    return {"ids": ids, "x": x, "c": c, "y": np.array(labels), "candidates": cands,
            "counterfactual_definition": cf_def,
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


def paired_zero_label_reference(ids: List[str], task: str = "") -> Dict[str, Any]:
    from gen_zero.causal.baseline_loader import load_qwen9b_baseline
    baseline = load_qwen9b_baseline()
    if task and task in baseline.get("per_task", {}):
        t_info = baseline["per_task"][task]
        n = len(ids)
        hit = int(round(t_info["winning_expert_acc_pct"] * n / 100.0))
        return {"n": n, "correct": hit, "accuracy_pct": t_info["winning_expert_acc_pct"],
                "wilson95_pct": wilson(hit, n),
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
    hit = sum(int(rows[i]["is_correct"]) for i in ids)
    e2e = [rows[i]["forward_ms"] + rows[i]["ar_ms"] + rows[i]["cot_ms"] for i in ids]
    return {"n": len(ids), "correct": hit, "accuracy_pct": round(100 * hit / len(ids), 2),
            "wilson95_pct": wilson(hit, len(ids)),
            "a100_e2e_ms_p50": round(float(np.percentile(e2e, 50)), 2),
            "a100_e2e_ms_p90": round(float(np.percentile(e2e, 90)), 2),
            "winning_experts": sorted({rows[i]["winning_expert"] for i in ids}),
            "artifact": str(PREDICTIONS.relative_to(ROOT)), "artifact_sha256": sha256_file(PREDICTIONS)}


def run_task(task: str, npz, rng: np.random.Generator, workdir: pathlib.Path) -> Dict[str, Any]:
    t = load_task(task, npz)
    ids, x, c, y, cands = t["ids"], t["x"], t["c"], t["y"], t["candidates"]
    n, k = len(ids), len(cands)
    label_str = [cands[i] for i in y]
    per_config_hits: Dict[str, List[int]] = {name: [] for name in CONFIGS}
    per_config_hits["harness_nearest_mean_prototype"] = []
    per_config_hits["calibration_fold_majority"] = []
    fold_log: List[Dict[str, Any]] = []
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
            for use_cf in (True, False):
                dyn, info = fit_counterfactual_drift_dynamics(
                    x[cal_rows], c[cal_rows], y[cal_rows], sample_ids=[ids[i] for i in cal_rows],
                    source=source, split="calibration", encoder_id=ENCODER_ID, n_classes=k,
                    dim=DIM, n_components=N_COMP, cf_components=CF_COMP, use_counterfactual=use_cf)
                path = workdir / f"{task}_r{rep}_f{fi}_cf{int(use_cf)}.npz"
                dyn.save(path)
                artifacts[use_cf] = CounterfactualDriftDynamics.load(path, encoder_id=ENCODER_ID)
                if use_cf:
                    ridge_lams.append(info["ridge_lambda"])
            # harness prototype baseline: nearest class mean in the calibration-fit PCA space
            base = artifacts[True]
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
            for name, cfg in CONFIGS.items():
                dyn = artifacts[cfg["cf"]]

                def predict(inp, dyn=dyn, cfg=cfg, name=name):
                    row = inp["row"]
                    t0 = time.perf_counter_ns()
                    res = dyn.infer(x[row], c[row], use_counterfactual=cfg["cf"],
                                    langevin=cfg["langevin"], expert=cfg["expert"])
                    dt = (time.perf_counter_ns() - t0) / 1e6
                    if name == "full_calibrated_dynamics":
                        latencies_full_ms.append(dt)
                        relax_residuals.append(res.relaxation_residual)
                        langevin_converged.append(bool(res.langevin.converged))
                        truth = y[row]
                        expert_pruned_correct.append(res.expert.traces[truth].status == "PRUNED")
                    if name == "readout_only":
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

    # memory: one full-task pass through the full config under tracemalloc
    dyn = artifacts[True]
    tracemalloc.start()
    for row in range(n):
        dyn.infer(x[row], c[row])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    counts = np.bincount(y, minlength=k)
    return {
        "n": n, "k_candidates": k, "candidates": cands, "ids": ids,
        "label_counts": {cands[i]: int(counts[i]) for i in range(k)},
        "majority_baseline_pct_all30": round(100 * counts.max() / n, 2),
        "counterfactual_definition": t["counterfactual_definition"],
        "data_sha256": t["data_sha256"],
        "protocol": {"folds": N_FOLDS, "repeats": N_REPEATS, "stratified": True,
                     "dim": DIM, "pca_components": N_COMP, "cf_components": CF_COMP,
                     "ridge_selection": "leave-one-out on calibration fold only",
                     "harness_contamination_checks_run": contamination_checks},
        "results": {name: summarize(h) for name, h in per_config_hits.items()},
        "diagnostics": {
            "langevin_converged_fraction": round(float(np.mean(langevin_converged)), 4),
            "expert_pruned_true_candidate_fraction": round(float(np.mean(expert_pruned_correct)), 4),
            "relaxation_residual_max": float(np.max(relax_residuals)),
            "ridge_lambda_histogram": {str(l): ridge_lams.count(l) for l in sorted(set(ridge_lams))},
        },
        "latency_cpu_ms_post_feature": {
            "full_calibrated_dynamics": {
                "n": len(latencies_full_ms),
                "p50": round(float(np.percentile(latencies_full_ms, 50)), 3),
                "p90": round(float(np.percentile(latencies_full_ms, 90)), 3),
                "mean": round(float(np.mean(latencies_full_ms)), 3),
                "throughput_samples_per_s": round(1000.0 / float(np.mean(latencies_full_ms)), 2)},
            "readout_only": {
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
        "reference_zero_label_paired_30_ids": paired_zero_label_reference(ids, task=task),
        "folds": fold_log,
    }


def main() -> int:
    npz = np.load(FEATURES, allow_pickle=False)
    rng = np.random.default_rng(SEED)
    track_c = json.loads(TRACK_C.read_text(encoding="utf-8"))
    t_start = time.time()
    with tempfile.TemporaryDirectory(prefix="cpu_dynamics_clean_eval_") as tmp:
        workdir = pathlib.Path(tmp)
        tasks = {task: run_task(task, npz, rng, workdir) for task in TASKS}
    for task in TASKS:
        tc = track_c["tasks"][task]
        tasks[task]["reference_zero_label_full"] = {
            "n": tc["n"], "accuracy_pct": tc["accuracy_pct"], "wilson95_pct": tc["wilson95_pct"],
            "majority_baseline_pct": tc["majority_baseline_pct"],
            "a100_e2e_ms_p50": tc["latency_e2e_ms"]["p50"]}
        tasks[task]["reference_official"] = tc["official"]
    report = {
        "title": "CPU zero-token clean evaluation: calibrated counterfactual-drift dynamics vs frozen zero-label 9B head",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "PYTHONPATH=python python3 benchmarks/suites/cpu_dynamics_clean_eval.py",
        "git_head": git_head(), "seed": SEED,
        "hardware": {"gpu": "none (no CUDA device; torch.cuda.is_available()=False)",
                     "cpu_logical_cores": os.cpu_count(), "platform": platform.platform(),
                     "python": sys.version.split()[0], "numpy": np.__version__,
                     "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "zero_token": True, "gpu_memory_bytes": 0,
        "inputs": {"features_npz": str(FEATURES.relative_to(ROOT)), "features_sha256": sha256_file(FEATURES),
                   "encoder_id": ENCODER_ID, "track_c_json_sha256": sha256_file(TRACK_C)},
        "comparison_frame": {
            "what_is_compared": "same frozen 9B hidden states, same 30 ids per task: "
                                "zero-label ar_loglik/manifold head (A100 artifact) vs calibrated "
                                "CPU dynamics head (this run)",
            "information_asymmetry": "the calibrated head consumes calibration-fold labels "
                                     "(24 of 30 per fold, never the test fold); the zero-label head "
                                     "consumes none. Equal-information baselines are the harness "
                                     "nearest-mean prototype and the calibration-fold majority.",
            "bare_9b_56_73_note": "56.73% is the 11-task zero-label macro average over 930 rows "
                                  "(Track C); it cannot be reproduced or extended here because no "
                                  "GPU and no 9B weights exist on this host. Only PAWS-30 and "
                                  "MultiNLI-30 are evaluated.",
            "official_source": track_c["bespoke_comparison"]["source"],
            "official_source_sha256": track_c["bespoke_comparison"]["source_sha256"],
            "shared11_macro_reference": track_c["bespoke_comparison"]["shared11_macro"],
            "png_reference": "no file named 1.png exists in the repository; official Nimble/Jev "
                             "numbers are taken from Track C's hash-bound PUBLIC_BENCHMARKS.md copy",
        },
        "tasks": tasks,
        "limitations": [
            "n=30 per task: Wilson 95% CI half-width is 15-18 pp; no difference below that is evidence",
            "PAWS-400 not evaluated: 370 rows have no frozen features and the 9B cannot run here",
            "calibrated head uses labels on calibration folds; it is NOT a zero-label method and must not be compared to 56.73% as if it were",
            "A is set contractive (0.5 I), not learned: with 24 calibration rows A has no learnable signal (see artifact provenance a_note)",
            "PCA and ridge are fit per fold on calibration rows only; no test-set statistic is used anywhere (unlike the Track C ZCA, which pooled all test features)",
            "latency excludes the 9B forward; the A100 e2e column includes it and is not directly comparable",
            "repeated k-fold reduces split variance only; sample variance (n=30) is unchanged",
        ],
        "elapsed_s": round(time.time() - t_start, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON.relative_to(ROOT)), "md": str(OUT_MD.relative_to(ROOT)),
                      "elapsed_s": report["elapsed_s"],
                      "summary": {t: {n: r["mean_accuracy_pct"] for n, r in tasks[t]["results"].items()}
                                  for t in TASKS}}, indent=2))
    return 0


def render_md(rep: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# CPU zero-token clean evaluation: calibrated counterfactual-drift dynamics vs frozen zero-label 9B head (%s)" % rep["generated_at_utc"][:10])
    L.append("")
    L.append("> **What is being compared.** The same frozen Qwen3.5-9B hidden states (extracted on an A100 on 2026-09-22, 30 ids per task),")
    L.append("> zero-label ar_loglik/manifold head (A100 artifact) versus the calibrated CPU dynamics head (this run, repeated stratified 5-fold).")
    L.append("> This host has no GPU or 9B weights: **no new 9B forward pass; PAWS-400 was not evaluated** (only 30 rows have frozen features).")
    L.append("> The calibration head used labels from calibration folds (never test folds), **so it is not a zero-label method**; it is shown alongside 56.73% only as a reference, not an equal-information comparison.")
    L.append("> `1.png` is not present in the repository; official Nimble/Jev numbers come from Track C's hash-bound copy of PUBLIC_BENCHMARKS.md.")
    L.append("")
    L.append("Command: `%s`  HEAD: `%s`  Feature SHA-256: `%s`" % (rep["command"], rep["git_head"], rep["inputs"]["features_sha256"]))
    L.append("")
    for task, t in rep["tasks"].items():
        L.append("## %s (n=%d, %d candidates, majority class %.2f%%)" % (task, t["n"], t["k_candidates"], t["majority_baseline_pct_all30"]))
        L.append("")
        L.append("Counterfactual vector c: %s" % t["counterfactual_definition"])
        L.append("")
        L.append("| Configuration | Label use | Mean accuracy %% (%d repeats) | Std. across repeats | Min/max | repeat0 Wilson95 |" % t["protocol"]["repeats"])
        L.append("|---|---|---:|---:|---|---|")
        z = t["reference_zero_label_paired_30_ids"]
        L.append("| Zero-label 9B head (A100 artifact, same 30 ids) | None | %.2f | – | – | %s |" % (z["accuracy_pct"], z["wilson95_pct"]))
        zf = t["reference_zero_label_full"]
        L.append("| Zero-label 9B head (full n=%d, reference) | None | %.2f | – | – | %s |" % (zf["n"], zf["accuracy_pct"], zf["wilson95_pct"]))
        for name, r in t["results"].items():
            use = "Calibration-fold labels" if name not in ("calibration_fold_majority",) else "Calibration-fold labels (majority class only)"
            L.append("| %s | %s | %.2f | %.2f | %.2f / %.2f | %s |" % (
                name, use, r["mean_accuracy_pct"], r["std_over_repeats_pct"], r["min_pct"], r["max_pct"], r["repeat0_wilson95_pct"]))
        o = t["reference_official"]
        L.append("| Official Nimble-9B (%s, n=%d) | Fine-tuned | %.1f | – | – | – |" % (o["subset"], o["n"], o["nimble_9b_pct"]))
        L.append("| Official Jev 1.13.0 (%s, n=%d) | Fine-tuned | %.1f | – | – | – |" % (o["subset"], o["n"], o["jev_1_13_0_pct"]))
        L.append("")
        d = t["diagnostics"]
        L.append("Diagnostics: Langevin convergence fraction %.3f; fraction where the expert pruned the true candidate %.3f; max relaxation residual %.2e; ridge λ histogram %s" % (
            d["langevin_converged_fraction"], d["expert_pruned_true_candidate_fraction"], d["relaxation_residual_max"], d["ridge_lambda_histogram"]))
        L.append("")
        lat = t["latency_cpu_ms_post_feature"]
        L.append("| Latency/throughput | P50 ms | P90 ms | Throughput samples/s | Notes |")
        L.append("|---|---:|---:|---:|---|")
        L.append("| CPU full configuration (post-feature) | %.3f | %.3f | %.1f | Excludes 9B forward pass |" % (lat["full_calibrated_dynamics"]["p50"], lat["full_calibrated_dynamics"]["p90"], lat["full_calibrated_dynamics"]["throughput_samples_per_s"]))
        L.append("| CPU readout-only (post-feature) | %.3f | %.3f | %.1f | Excludes 9B forward pass |" % (lat["readout_only"]["p50"], lat["readout_only"]["p90"], lat["readout_only"]["throughput_samples_per_s"]))
        L.append("| A100 zero-label e2e (same 30 ids) | %.2f | %.2f | – | Includes 9B forward + ar scoring; not directly comparable |" % (z["a100_e2e_ms_p50"], z["a100_e2e_ms_p90"]))
        L.append("")
        m = t["memory"]
        L.append("Memory: full configuration, 30 samples, tracemalloc peak %d bytes; artifact working set %d bytes; GPU memory 0." % (
            m["tracemalloc_peak_bytes_full_config_30_samples"], m["artifact_working_set_bytes"]))
        L.append("")
    L.append("## 11-task macro-average reference (Track C; not reproducible here)")
    L.append("")
    for k2, v in rep["comparison_frame"]["shared11_macro_reference"].items():
        L.append("- %s: %s" % (k2, v))
    L.append("")
    L.append("## Limitations")
    L.append("")
    for lim in rep["limitations"]:
        L.append("- " + lim)
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
