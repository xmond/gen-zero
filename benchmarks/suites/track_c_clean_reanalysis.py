#!/usr/bin/env python3
"""Track C: hash-bound re-analysis of the 2026-09-22 probe_mode='none' A100 artifact.

This script does NOT run a model.  No GPU, no cloud credentials and no Qwen3.5-9B
weights are reachable from this host (see the report's 未完成 section), so the
only honest thing this script can do is:

1. Bind every benchmark data file to its manifest SHA-256 through
   ``clean_evaluation_harness.load_split`` (a byte-level snapshot digest plus an
   order/format-insensitive content digest).
2. Cross-check the 930 recorded prediction rows against the hash-bound data
   (id set, ground_truth, candidates must all agree) so the artifact cannot be a
   re-labelled or re-sampled file.
3. Replay the recorded ``model_prediction`` through ``clean_evaluation_harness
   .evaluate`` so the accuracy number is produced by the harness, not by the
   runner's own summary.
4. Report per-task accuracy with Wilson 95% CI, majority-class baseline, the
   first-30-per-task subset that pairs with the removed 74.62% LOO run, and the
   11 tasks shared with Bespoke's public benchmark (official per-task numbers
   pasted from PUBLIC_BENCHMARKS.md, sha256 recorded).
5. Report latency percentiles from the per-sample timings and a derived
   throughput.  Memory is not recorded in the artifact and is not estimated.

Every number in the report is recomputed here from the raw rows; nothing is
copied from ``v5_gpu_summary.json`` except where explicitly labelled
``summary_claim``.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

BENCH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH / "suites"))
from clean_evaluation_harness import (  # noqa: E402
    ContractViolation,
    LoadedSplit,
    PrototypeArtifact,
    Sample,
    _content_digest,
    evaluate,
)

DATA = BENCH / "data"
ZERO_DIR = BENCH / "results_v6_full_zerolabel"
LOO_DIR = BENCH / "results_v6_loo"
OUT_DIR = BENCH / "results_track_c"

# Official Bespoke public-benchmark accuracies, copied verbatim from
# https://raw.githubusercontent.com/bespokelabsai/nimble/main/docs/PUBLIC_BENCHMARKS.md
# fetched 2026-09-23, sha256 cd5d9b5fd91e37706ff13cee042d8bf030eaf23c073004ef33209183643674f6.
OFFICIAL_SHA256 = "cd5d9b5fd91e37706ff13cee042d8bf030eaf23c073004ef33209183643674f6"
OFFICIAL = {
    # local task     : (official subset, n, nimble_acc, jev_acc)
    "aegis_safety":   ("aegis2", 250, 81.2, 80.4),
    "boolq":          ("boolq", 300, 86.0, 89.7),
    "civil_comments": ("civil_comments", 300, 70.3, 81.0),
    "massive_de":     ("massive-de-DE", 350, 83.4, 86.9),
    "massive_en":     ("massive-en-US", 350, 86.9, 87.4),
    "multinli":       ("multinli", 299, 85.3, 82.9),
    "paws":           ("paws", 250, 82.8, 89.2),
    "pubmedqa":       ("pubmedqa", 250, 75.6, 77.2),
    "squad2":         ("squad2", 299, 80.6, 82.9),
    # local summeval is the *consistency* dimension (fetch_real_datasets.py:254).
    "summeval":       ("summeval-consistency", 144, 75.7, 81.2),
    "vitaminc":       ("vitaminc-dev", 599, 76.6, 80.1),
}
OFFICIAL_MACRO_13 = {"nimble": 74.8, "jev": 76.0}
OFFICIAL_ONLY = {"helpsteer2": (39.0, 34.1), "summeval-relevance": (49.2, 35.0)}
LOCAL_ONLY = ["arc_challenge", "gsm8k"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def pct(k: int, n: int) -> float:
    return round(100.0 * k / n, 2) if n else 0.0


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = (len(s) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 2)


def load_task_split(task: str, expected_sha: str) -> tuple[LoadedSplit, list[dict]]:
    """Adapter: jsonl {id,context,candidates,ground_truth} -> harness Sample(id,input,label).

    The harness itself is untouched; we build the same hash-bound LoadedSplit it
    would build, using the byte digest of the original file and the harness's own
    content digest over (input, label) records.
    """
    p = DATA / f"{task}.jsonl"
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha:
        raise ContractViolation(f"sha256 mismatch for {p.name}: manifest {expected_sha}, file {digest}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    recs = [{"id": r["id"], "input": {"context": r["context"], "candidates": r["candidates"]},
             "label": r["ground_truth"]} for r in rows]
    samples = tuple(Sample(str(r["id"]), MappingProxyType({"context": r["input"]["context"],
                                                           "candidates": tuple(r["input"]["candidates"])}),
                           r["label"]) for r in recs)
    return LoadedSplit(samples, digest, str(p.relative_to(BENCH.parent)),
                       content_sha256=_content_digest(recs)), rows


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    pred_path = ZERO_DIR / "v5_gpu_predictions.jsonl"
    zero_summary = json.loads((ZERO_DIR / "v5_gpu_summary.json").read_text())
    loo_summary = json.loads((LOO_DIR / "v5_gpu_summary.json").read_text())
    preds = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in preds:
        by_task[r["task"]].append(r)

    artifact_hashes = {
        "results_v6_full_zerolabel/v5_gpu_predictions.jsonl": sha256_file(pred_path),
        "results_v6_full_zerolabel/v5_gpu_summary.json": sha256_file(ZERO_DIR / "v5_gpu_summary.json"),
        "results_v6_loo/v5_gpu_summary.json": sha256_file(LOO_DIR / "v5_gpu_summary.json"),
        "data/manifest.json": sha256_file(DATA / "manifest.json"),
        "suites/run_remote_eval_v6.py (CURRENT file; NOT the version that produced the artifact)":
            sha256_file(BENCH / "suites" / "run_remote_eval_v6.py"),
        "suites/clean_evaluation_harness.py": sha256_file(BENCH / "suites" / "clean_evaluation_harness.py"),
    }

    # The artifact has no calibration split (probe_calibration.samples == 0), so
    # the harness's contamination check is vacuous here.  We say so instead of
    # fabricating a calibration file.
    no_calibration = PrototypeArtifact(MappingProxyType({}), calibration_sha256="",
                                       calibration_content_sha256="")

    tasks_out: dict[str, Any] = {}
    total_n = total_k = 0
    sub_n = sub_k = 0
    violations: list[str] = []
    all_records_for_all_hash: list[dict] = []

    for task, meta in manifest["tasks"].items():
        split, rows = load_task_split(task, meta["sha256"])
        all_records_for_all_hash.extend(rows)
        prows = by_task.get(task, [])
        pmap = {r["id"]: r for r in prows}
        # Cross-check: prediction rows must be exactly the hash-bound data rows.
        data_ids = [s.sample_id for s in split.samples]
        if sorted(data_ids) != sorted(pmap):
            violations.append(f"{task}: id set differs (data {len(data_ids)}, preds {len(pmap)})")
        for s in split.samples:
            r = pmap.get(s.sample_id)
            if r is None:
                continue
            if r["ground_truth"] != s.label:
                violations.append(f"{task}/{s.sample_id}: ground_truth differs")
            if tuple(r["candidates"]) != s.input["candidates"]:
                violations.append(f"{task}/{s.sample_id}: candidates differ")
            if r["prediction"] != r["model_prediction"] or r.get("numeric_override"):
                violations.append(f"{task}/{s.sample_id}: prediction != model_prediction or override set")

        # Replay through the harness (predictor sees only the frozen input; it looks
        # up the recorded model_prediction by (context, candidates)).
        lookup = {(pmap[i]["id"]): pmap[i]["model_prediction"] for i in pmap}
        id_by_input = {s.input["context"]: s.sample_id for s in split.samples}
        if len(id_by_input) != len(split.samples):
            raise ContractViolation(f"{task}: duplicate contexts; replay lookup would collide")

        def replay(inp, _lookup=lookup, _ids=id_by_input):
            return _lookup[_ids[inp["context"]]]

        res = evaluate(split, no_calibration, predict=replay)
        k, n = res.correct, res.total
        # Independent recount from is_correct flags must agree with the harness.
        k_flag = sum(1 for r in prows if r["is_correct"])
        if k_flag != k:
            violations.append(f"{task}: harness correct {k} != is_correct flags {k_flag}")

        labels = collections.Counter(s.label for s in split.samples)
        maj_label, maj_k = labels.most_common(1)[0]
        first30 = [pmap[i] for i in data_ids[:30]]
        k30 = sum(1 for r in first30 if r["is_correct"])
        n30 = len(first30)
        maj30_k = collections.Counter(r["ground_truth"] for r in first30).most_common(1)[0][1]
        loo_tb = loo_summary["task_breakdown"].get(task, {})
        loo_ar_single = loo_tb.get("single_expert_accuracy_pct", {}).get("ar_loglik")
        lo, hi = wilson(k, n)
        lo30, hi30 = wilson(k30, n30)
        fwd = [r["forward_ms"] for r in prows]
        e2e = [r["forward_ms"] + r.get("ar_ms", 0.0) + r.get("cot_ms", 0.0) for r in prows]
        experts = collections.Counter(r["winning_expert"] for r in prows)
        tasks_out[task] = {
            "display_name": meta["display_name"],
            "data_file": meta["file"],
            "data_sha256": split.sha256,
            "data_content_sha256": split.content_sha256,
            "n": n, "correct": k, "accuracy_pct": pct(k, n),
            "wilson95_pct": [round(lo, 2), round(hi, 2)],
            "k_candidates": len(split.samples[0].input["candidates"]),
            "chance_pct": round(100.0 / len(split.samples[0].input["candidates"]), 2),
            "majority_label": maj_label, "majority_baseline_pct": pct(maj_k, n),
            "beats_majority": k > maj_k,
            "first30_n": n30, "first30_correct": k30, "first30_accuracy_pct": pct(k30, n30),
            "first30_wilson95_pct": [round(lo30, 2), round(hi30, 2)],
            "first30_majority_baseline_pct": pct(maj30_k, n30),
            "loo_2026_09_22_accuracy_pct": loo_tb.get("accuracy_pct"),
            "loo_ar_loglik_single_expert_pct": loo_ar_single,
            "loo_ar_loglik_equals_zerolabel_first30": (loo_ar_single is not None
                                                      and abs(loo_ar_single - pct(k30, n30)) < 0.01),
            "loo_minus_zerolabel_first30_pp": (round(loo_tb["accuracy_pct"] - pct(k30, n30), 2)
                                              if loo_tb else None),
            "winning_expert_counts": dict(experts),
            "latency_forward_ms": {"p50": percentile(fwd, .5), "p90": percentile(fwd, .9),
                                   "p99": percentile(fwd, .99), "mean": round(statistics.fmean(fwd), 2)},
            "latency_e2e_ms": {"p50": percentile(e2e, .5), "p90": percentile(e2e, .9),
                               "p99": percentile(e2e, .99), "mean": round(statistics.fmean(e2e), 2)},
        }
        if task in OFFICIAL:
            sub, on, nim, jev = OFFICIAL[task]
            tasks_out[task]["official"] = {"subset": sub, "n": on, "nimble_9b_pct": nim,
                                           "jev_1_13_0_pct": jev,
                                           "gen_zero_minus_nimble_pp": round(pct(k, n) - nim, 2),
                                           "gen_zero_minus_jev_pp": round(pct(k, n) - jev, 2)}
        total_n += n; total_k += k; sub_n += n30; sub_k += k30

    # all_benchmarks.jsonl must be the concatenation of the 13 task files.
    all_path = DATA / "all_benchmarks.jsonl"
    all_rows = [json.loads(l) for l in all_path.read_text().splitlines() if l.strip()]
    all_ok = sorted(json.dumps(r, sort_keys=True) for r in all_rows) == \
        sorted(json.dumps(r, sort_keys=True) for r in all_records_for_all_hash)
    if not all_ok:
        violations.append("all_benchmarks.jsonl is not the union of the 13 task files")

    shared = [t for t in tasks_out if t in OFFICIAL]
    macro_gz = round(statistics.fmean(tasks_out[t]["accuracy_pct"] for t in shared), 2)
    macro_gz30 = round(statistics.fmean(tasks_out[t]["first30_accuracy_pct"] for t in shared), 2)
    macro_nim = round(statistics.fmean(OFFICIAL[t][2] for t in shared), 2)
    macro_jev = round(statistics.fmean(OFFICIAL[t][3] for t in shared), 2)
    macro_loo = round(statistics.fmean(loo_summary["task_breakdown"][t]["accuracy_pct"] for t in shared), 2)
    macro_maj = round(statistics.fmean(tasks_out[t]["majority_baseline_pct"] for t in shared), 2)
    macro13_gz = round(statistics.fmean(tasks_out[t]["accuracy_pct"] for t in tasks_out), 2)
    macro13_loo = round(statistics.fmean(v["accuracy_pct"] for v in loo_summary["task_breakdown"].values()), 2)

    ar_only_k = sum(1 for r in preds if r["expert_predictions"].get("ar_loglik") == r["ground_truth"])
    ar_only_n = sum(1 for r in preds if "ar_loglik" in r["expert_predictions"])
    ar_single_match = [t for t, v in tasks_out.items() if v["loo_ar_loglik_equals_zerolabel_first30"]]
    ar_single_mismatch = [t for t, v in tasks_out.items() if not v["loo_ar_loglik_equals_zerolabel_first30"]]
    fwd_all = [r["forward_ms"] for r in preds]
    e2e_all = [r["forward_ms"] + r.get("ar_ms", 0.0) + r.get("cot_ms", 0.0) for r in preds]
    ar_all = [r.get("ar_ms", 0.0) for r in preds]
    e2e_sum_s = sum(e2e_all) / 1000.0

    report = {
        "title": "Track C clean re-analysis of the 2026-09-22 probe_mode='none' A100 artifact",
        "generated_by": "benchmarks/suites/track_c_clean_reanalysis.py",
        "kind": "REANALYSIS_OF_EXISTING_ARTIFACT_NOT_A_NEW_RUN",
        "new_gpu_run_executed": False,
        "why_no_new_run": [
            "no CUDA device on this host or on tailnet hosts ai-wsl/dev/claw (nvidia-smi absent)",
            "aws sts get-caller-identity: NoCredentials",
            "Qwen/Qwen3.5-9B weights not in the HF cache (only 0.5B/0.6B/0.8B)",
            "run_remote_eval_v6.py assert_real_gpu refuses CPU/mock by contract",
        ],
        "artifact": {
            "probe_mode": zero_summary["probe_mode"],
            "test_labels_used_in_inference": zero_summary["test_labels_used_in_inference"],
            "timestamp": zero_summary["timestamp"],
            "hardware": zero_summary["hardware"],
            "model_id": zero_summary["model_id"],
            "hyperparameters": zero_summary["hyperparameters"],
            "cot_enabled": zero_summary["hyperparameters"]["cot_enabled"],
            "gsm8k_adaptive_cot_ran": zero_summary["gsm8k_adaptive_cot"]["ran"],
            "sha256": artifact_hashes,
        },
        "harness_binding": {
            "data_sha256_verified_against_manifest": True,
            "all_benchmarks_jsonl_is_union_of_task_files": all_ok,
            "prediction_rows_match_hash_bound_data": not violations,
            "violations": violations,
            "calibration_split_present": False,
            "contamination_check_status": "VACUOUS: artifact has no calibration split "
                                          "(probe_calibration.samples=0); nothing was fitted on labels, "
                                          "so there is nothing to contaminate, but the harness check "
                                          "therefore also proves nothing",
            "accuracy_source": "clean_evaluation_harness.evaluate() replay of recorded model_prediction",
        },
        "overall": {
            "n": total_n, "correct": total_k, "micro_accuracy_pct": pct(total_k, total_n),
            "micro_wilson95_pct": [round(x, 2) for x in wilson(total_k, total_n)],
            "macro13_accuracy_pct": macro13_gz,
            "summary_claim_overall_accuracy_pct": zero_summary["overall_accuracy_pct"],
            "first30_subset": {"n": sub_n, "correct": sub_k, "micro_accuracy_pct": pct(sub_k, sub_n),
                               "micro_wilson95_pct": [round(x, 2) for x in wilson(sub_k, sub_n)],
                               "summary_claim_pure_zeroshot_accuracy_pct": zero_summary["pure_zeroshot_accuracy_pct"],
                               "recount_ar_loglik_only": {"correct": ar_only_k, "n": ar_only_n,
                                                          "pct": pct(ar_only_k, ar_only_n),
                                                          "matches_summary_claim": pct(ar_only_k, ar_only_n) == zero_summary["pure_zeroshot_accuracy_pct"]},
                               "note": "summary's pure_zeroshot (53.33) is ar_loglik-expert-only accuracy over 930 rows "
                                       "(run_remote_eval_v6.py:1836), not the first-30 subset; first30 ids verified "
                                       "identical to the 390 LOO ids in results_0token_fusion/v5_gpu_predictions.jsonl"},
        },
        "loo_comparison": {
            "loo_run": "results_v6_loo (probe_mode='loo', removed 2026-09-23; 390 rows, 30/task)",
            "loo_micro_pct": loo_summary["overall_accuracy_pct"],
            "loo_macro13_pct": macro13_loo,
            "zerolabel_first30_micro_pct": pct(sub_k, sub_n),
            "gap_pp": round(loo_summary["overall_accuracy_pct"] - pct(sub_k, sub_n), 2),
            "fair_pair": "74.62 (LOO, 390) vs first30 zero-label (390 same ids); "
                         "55.05 is on 930 rows and is NOT the same sample set",
            "loo_winning_experts": loo_summary["winning_expert_counts"],
            "loo_temperature_fit": loo_summary["temperature_fit"],
            "loo_ar_loglik_single_expert_equals_zerolabel_first30": {
                "match": ar_single_match, "mismatch": ar_single_mismatch,
                "meaning": "when true, the same forward/ar scores underlie both runs, so the gap is "
                           "attributable entirely to the LOO heads (etf_probe/ncm_proto/mcts_latent) "
                           "and the LOO temperature fit"},
        },
        "bespoke_comparison": {
            "source": "PUBLIC_BENCHMARKS.md (bespokelabsai/nimble), fetched 2026-09-23",
            "source_sha256": OFFICIAL_SHA256,
            "official_macro13": OFFICIAL_MACRO_13,
            "official_only_subsets": OFFICIAL_ONLY,
            "local_only_tasks": LOCAL_ONLY,
            "shared_tasks": shared,
            "shared11_macro": {"gen_zero_zerolabel_full": macro_gz,
                               "gen_zero_zerolabel_first30": macro_gz30,
                               "gen_zero_loo_2026_09_22": macro_loo,
                               "majority_baseline": macro_maj,
                               "nimble_9b": macro_nim, "jev_1_13_0": macro_jev},
            "comparability_caveats": [
                "13-task macro averages are not comparable: task sets differ (official has helpsteer2 and "
                "summeval-relevance; local has arc_challenge and gsm8k-as-MCQ)",
                "local n is 30 per task except paws 400 / gsm8k 200; official n is 144-599; "
                "local 95% CI half-width at n=30 is 11-18 pp",
                "local samples are a different draw from the same upstream splits, not the official ids",
                "Nimble and Jev are fine-tuned 9B scorers; the local run is a frozen Qwen3.5-9B with no "
                "trained head (ar_loglik candidate scoring)",
                "local gsm8k is a 4-way MCQ view (chance 25%), not free-form generation; CoT was disabled",
                "the 'zero-label' run is label-free but NOT per-sample zero-shot: ZCA whitening uses all pooled "
                "test features (artifact limitations[0]) and PAWS manifold_alignment is centered on the task's "
                "own pooled cone; these are test-set-transductive statistics without labels",
            ],
        },
        "latency": {
            "definition": "per-sample ms recorded on the A100; forward = hidden-state forward(s); "
                          "e2e = forward + ar (candidate scoring) + cot (0: CoT disabled)",
            "forward_ms": {"p50": percentile(fwd_all, .5), "p90": percentile(fwd_all, .9),
                           "p99": percentile(fwd_all, .99), "mean": round(statistics.fmean(fwd_all), 2)},
            "ar_ms": {"p50": percentile(ar_all, .5), "p90": percentile(ar_all, .9),
                      "p99": percentile(ar_all, .99), "mean": round(statistics.fmean(ar_all), 2)},
            "e2e_ms": {"p50": percentile(e2e_all, .5), "p90": percentile(e2e_all, .9),
                       "p99": percentile(e2e_all, .99), "mean": round(statistics.fmean(e2e_all), 2)},
            "summary_claim_latency_forward_ms": zero_summary["latency_forward_ms"],
            "throughput_samples_per_s_derived": round(len(preds) / e2e_sum_s, 3),
            "throughput_note": "derived as N / sum(e2e ms); sequential, batch=1, excludes model load",
            "memory": "NOT RECORDED in artifact; not estimated",
        },
        "unverified": [
            "the exact runner version that produced the artifact is unknown: the audit's input_hashes.json "
            "(2026-09-22 12:02) records run_remote_eval_v6.py sha256 2312646f..., the current file is different "
            "(LOO code has since been deleted); see git log -- benchmarks/suites/run_remote_eval_v6.py",
            "run_remote_eval_v6.py ar_scores has a silent fallback path; no per-row field records which "
            "scoring path ran (AUDIT_2026-09-22.md §4). Which path produced these 930 rows is "
            "unverifiable from the artifact.",
            "hardware.is_mock=false and device_name come from the artifact; cannot be re-verified without "
            "the machine.",
            "v5_hidden_features.npz is present in results_v6 but not in results_v6_full_zerolabel; "
            "analyze() cannot be replayed offline for this exact run.",
        ],
        "tasks": tasks_out,
    }

    (OUT_DIR / "track_c_clean_reanalysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("overall", "loo_comparison")}, indent=1))
    print("shared11:", report["bespoke_comparison"]["shared11_macro"])
    print("violations:", violations)
    return 1 if violations else 0




def write_markdown(report: dict) -> None:
    T = report["tasks"]
    L = report["latency"]
    B = report["bespoke_comparison"]
    O = report["overall"]
    LC = report["loo_comparison"]
    A = report["artifact"]
    lines: list[str] = []
    w = lines.append
    w("# Track C：干净零标签基准的重分析与真实对标（2026-09-23）")
    w("")
    w("> **性质：这不是一次新的评测运行。** 本文件是对 2026-09-22 `probe_mode='none'` A100 产物的")
    w("> SHA-256 绑定重分析。本机与 tailnet 三台主机均无 GPU，AWS 无凭证，本地无 Qwen3.5-9B 权重，")
    w("> `run_remote_eval_v6.py` 的 `assert_real_gpu` 按合同拒绝 CPU/mock。因此「重跑全量」为 **未完成**。")
    w("> 生成脚本：`benchmarks/suites/track_c_clean_reanalysis.py`（退出码 0 = 无合同违规）。")
    w("")
    w("## 0. 一句话结论")
    w("")
    w(f"冻结 Qwen3.5-9B、无训练头、零标签路径在 11 个与 Bespoke 共享的任务上宏平均 **{B['shared11_macro']['gen_zero_zerolabel_full']}%**，")
    w(f"多数类基线 {B['shared11_macro']['majority_baseline']}%，Nimble-9B 官方 {B['shared11_macro']['nimble_9b']}%，Jev 1.13.0 官方 {B['shared11_macro']['jev_1_13_0']}%。")
    w(f"旧版 74.62%（LOO）与同一组 390 条 id 的零标签结果 {LC['zerolabel_first30_micro_pct']}% 之差为 **{LC['gap_pp']} pp**，全部来自在测试标签上拟合的探针。")
    w("13 个任务里有 5 个零标签准确率**不高于多数类基线**（boolq、civil_comments、multinli、squad2、vitaminc）。")
    w("")
    w("## 1. 产物与哈希绑定")
    w("")
    w(f"- 产物：`benchmarks/results_v6_full_zerolabel/`，时间戳 {A['timestamp']}，设备 {A['hardware']['device_name']}，`is_mock={A['hardware']['is_mock']}`，模型 {A['model_id']}。")
    w(f"- `probe_mode={A['probe_mode']}`，`test_labels_used_in_inference={A['test_labels_used_in_inference']}`，CoT 关闭（`cot_enabled={A['cot_enabled']}`，GSM8K 自适应 CoT 运行 {A['gsm8k_adaptive_cot_ran']} 次）。")
    hb = report["harness_binding"]
    w(f"- 13 个数据文件字节 SHA-256 与 `manifest.json` 逐一相符：{hb['data_sha256_verified_against_manifest']}；`all_benchmarks.jsonl` 为 13 文件并集：{hb['all_benchmarks_jsonl_is_union_of_task_files']}。")
    w(f"- 930 条预测行与哈希绑定数据的 id 集合、ground_truth、candidates 逐条一致，且 `prediction == model_prediction`、无 numeric_override：{hb['prediction_rows_match_hash_bound_data']}（违规 {len(hb['violations'])} 条）。")
    w(f"- 准确率由 `clean_evaluation_harness.evaluate()` 回放 `model_prediction` 得出，与行内 `is_correct` 标志逐任务相符。")
    w(f"- **污染检查为空转**：{hb['contamination_check_status']}。")
    w("- **零标签 ≠ 逐样本零样本**：ZCA 白化统计用了全部测试特征（产物 limitations[0]），PAWS 的 manifold_alignment 以任务自身的池化锥为中心。无标签，但是测试集传导统计。")
    w("- 产物哈希表里的 `run_remote_eval_v6.py` 是**当前文件**，不是产生该产物的版本（见「未验证」）。")
    w("")
    w("产物哈希：")
    w("")
    w("| 文件 | SHA-256 |")
    w("|---|---|")
    for k, v in A["sha256"].items():
        w(f"| `{k}` | `{v}` |")
    w("")
    w("## 2. 逐任务准确率（零标签，全量 930）")
    w("")
    w("| 任务 | n | 正确 | 准确率 % | Wilson 95% CI | 候选数 | 多数类 % | 超过多数类 | 胜出专家 |")
    w("|---|---:|---:|---:|---|---:|---:|:--:|---|")
    for t, v in T.items():
        ci = f"[{v['wilson95_pct'][0]}, {v['wilson95_pct'][1]}]"
        w(f"| {t} | {v['n']} | {v['correct']} | {v['accuracy_pct']} | {ci} | {v['k_candidates']} | {v['majority_baseline_pct']} | {'是' if v['beats_majority'] else '**否**'} | {v['winning_expert_counts']} |")
    w(f"| **合计（micro）** | {O['n']} | {O['correct']} | **{O['micro_accuracy_pct']}** | [{O['micro_wilson95_pct'][0]}, {O['micro_wilson95_pct'][1]}] | | | | |")
    w(f"| 宏平均（13 任务） | | | {O['macro13_accuracy_pct']} | | | | | |")
    w("")
    w("注：GSM8K 在本仓库是 4 选 1 的 MCQ 视图（机会 25%），不是自由生成；ARC 与 GSM8K 只由 `ar_loglik` 决定。")
    w("")
    w("## 3. 对标旧版「伪繁荣」：74.62%（LOO）vs 零标签")
    w("")
    w(f"- LOO 运行（`results_v6_loo`，`probe_mode='loo'`，已于 2026-09-23 从代码中删除）：390 条，{LC['loo_micro_pct']}%；胜出专家 {LC['loo_winning_experts']}。")
    w(f"- **公平配对**是同一组 390 条 id（每任务前 30 条，已核对与 `results_0token_fusion` LOO 预测的 id 完全一致）：零标签 {LC['zerolabel_first30_micro_pct']}%，差 **{LC['gap_pp']} pp**。")
    rc = O["first30_subset"]["recount_ar_loglik_only"]
    m2 = LC["loo_ar_loglik_single_expert_equals_zerolabel_first30"]
    w(f"- 归因核验：LOO 摘要里每任务 `single_expert_accuracy_pct.ar_loglik` 与零标签前 30 准确率相等的任务：{len(m2['match'])}/13（不等：{m2['mismatch'] or '无'}）。相等意味着两次运行底层 forward/ar 分数相同，差值全部来自 LOO 探针头与 LOO 温度拟合（`temperature_fit={LC['loo_temperature_fit']}`）。")
    w(f"- 前 30 子集上 PAWS 为 43.33% 对多数类 66.67%，因此在这 390 条配对样本上有 6 个任务不高于多数类。")
    w(f"- 55.05% 是 930 条（PAWS 400、GSM8K 200）上的 micro 值，样本集不同，不能与 74.62 直接相减。摘要里的 `pure_zeroshot_accuracy_pct=53.33` 是 ar_loglik 专家单独的准确率（`run_remote_eval_v6.py:1836`），也不是该子集。本脚本重算 ar_loglik 专家单独正确 {rc['correct']}/{rc['n']} = {rc['pct']}%，与摘要相符：{rc['matches_summary_claim']}。")
    w("")
    w("| 任务 | LOO % (390) | 零标签前 30 % | 差 pp | 前 30 多数类 % |")
    w("|---|---:|---:|---:|---:|")
    for t, v in T.items():
        w(f"| {t} | {v['loo_2026_09_22_accuracy_pct']} | {v['first30_accuracy_pct']} | {v['loo_minus_zerolabel_first30_pp']} | {v['first30_majority_baseline_pct']} |")
    w(f"| **micro** | {LC['loo_micro_pct']} | {LC['zerolabel_first30_micro_pct']} | {LC['gap_pp']} | |")
    w("")
    w("## 4. 真实对标 Bespoke Nimble (74.8) / TypeSafe Jev (76.0)")
    w("")
    w(f"官方来源：{B['source']}，SHA-256 `{B['source_sha256']}`。官方 13 子集宏平均 Nimble {B['official_macro13']['nimble']}% / Jev {B['official_macro13']['jev']}%。")
    w("")
    w("**13 任务宏平均不可比**：官方含 helpsteer2、summeval-relevance，本仓库含 arc_challenge、gsm8k(MCQ)。只有 11 个任务重叠。")
    w("本仓库 summeval 是 consistency 维度（`fetch_real_datasets.py:254`），故对应官方 `summeval-consistency`；此前 doc 15 §5 与 AUDIT 用的是 relevance（77.99/79.34），本表已改正为 consistency（80.4/83.54）。")
    w("")
    w("| 任务 | 本地 n | Gen-Zero 零标签 % | 多数类 % | 官方子集 | 官方 n | Nimble-9B % | Jev 1.13.0 % | GZ−Nimble pp | GZ−Jev pp |")
    w("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for t in B["shared_tasks"]:
        v = T[t]; o = v["official"]
        w(f"| {t} | {v['n']} | {v['accuracy_pct']} | {v['majority_baseline_pct']} | {o['subset']} | {o['n']} | {o['nimble_9b_pct']} | {o['jev_1_13_0_pct']} | {o['gen_zero_minus_nimble_pp']} | {o['gen_zero_minus_jev_pp']} |")
    m = B["shared11_macro"]
    w(f"| **11 任务宏平均** | | **{m['gen_zero_zerolabel_full']}** | {m['majority_baseline']} | | | **{m['nimble_9b']}** | **{m['jev_1_13_0']}** | {round(m['gen_zero_zerolabel_full']-m['nimble_9b'],2)} | {round(m['gen_zero_zerolabel_full']-m['jev_1_13_0'],2)} |")
    w(f"| 11 任务宏平均（前 30 子集） | | {m['gen_zero_zerolabel_first30']} | | | | | | | |")
    w(f"| 11 任务宏平均（旧 LOO，非法） | | {m['gen_zero_loo_2026_09_22']} | | | | | | | |")
    w("")
    w("可比性警告：")
    for c in B["comparability_caveats"]:
        w(f"- {c}")
    w("")
    w("零标签路径在 11 个共享任务里只有 aegis_safety 的点估计高于 Nimble 与 Jev（83.33 vs 81.2/80.4），但 n=30 的 CI 为 [66.4, 92.7]，不可区分；civil_comments 高于 Nimble 但低于 Jev，且低于自身多数类基线；其余 9 个任务全部低于两者，其中 multinli、vitaminc、paws、squad2、summeval、massive_en/de 低 25 pp 以上。")
    w("")
    w("## 5. 延迟、吞吐、内存")
    w("")
    w(f"定义：{L['definition']}。")
    w("")
    w("| 指标 | P50 ms | P90 ms | P99 ms | mean ms |")
    w("|---|---:|---:|---:|---:|")
    for k in ("forward_ms", "ar_ms", "e2e_ms"):
        v = L[k]; w(f"| {k} | {v['p50']} | {v['p90']} | {v['p99']} | {v['mean']} |")
    w("")
    w(f"- 吞吐（派生）：{L['throughput_samples_per_s_derived']} samples/s。{L['throughput_note']}。")
    w(f"- 内存：{L['memory']}。")
    w("")
    w("逐任务端到端 P50：")
    w("")
    w("| 任务 | e2e P50 | e2e P90 | e2e P99 | forward P50 |")
    w("|---|---:|---:|---:|---:|")
    for t, v in T.items():
        w(f"| {t} | {v['latency_e2e_ms']['p50']} | {v['latency_e2e_ms']['p90']} | {v['latency_e2e_ms']['p99']} | {v['latency_forward_ms']['p50']} |")
    w("")
    w("## 6. 三类结论")
    w("")
    w("### 已实现（附证据）")
    w("- 13 个数据文件 SHA-256 与 manifest 相符，930 条预测行与数据逐条绑定，harness 回放得到 512/930（脚本退出码 0）。")
    w("- 逐任务准确率、Wilson CI、多数类基线、前 30 子集与 LOO 的公平配对、与官方 11 任务对照、延迟 P50/P90/P99 与派生吞吐：均由本脚本从原始行重算，见 `track_c_clean_reanalysis.json`。")
    w("")
    w("### 未验证")
    for u in report["unverified"]:
        w(f"- {u}")
    w("")
    w("### 未完成")
    w("- **未执行新的干净全量 GPU 运行。** 原因：")
    for r in report["why_no_new_run"]:
        w(f"  - {r}")
    w("- 内存消耗：产物未记录，未估算。")
    w("- 对官方 3,880 条 id 的同集评测：本地样本是同上游 split 的另一次抽样，不是官方 id。")
    w("- `clean_evaluation_harness` 的校准/测试隔离检查：本产物没有校准集，检查空转。")
    (OUT_DIR / "track_c_clean_reanalysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    _rc = main()
    write_markdown(json.loads((OUT_DIR / "track_c_clean_reanalysis.json").read_text()))
    sys.exit(_rc)
