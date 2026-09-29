#!/usr/bin/env python3
"""CPU-only benchmark for the late causal consensus ensemble over independent adapters.

For each `--model NAME:adapter.npz:features.npz`, the features file must
carry the same layout as artifacts/qwen35_9b/parity_val200.npz:
    q       (N, D)       query hidden state per record
    cands   (M, D)       all candidate hidden states, concatenated
    offsets (N+1,)       cands[offsets[i]:offsets[i+1]] are record i's candidates
    labels  (N,)         index of the correct candidate within its slice
    tasks   (N,)         task name per record (for macro accuracy)

Every model scores its own (q, cands) in its own space; nothing here
concatenates or shares features across models. Reports each model's solo
Macro/Micro accuracy plus, when >=2 models are given, the ensemble's
Macro/Micro accuracy under all three fusion strategies and the agreement
rate (fraction of records where every model's argmax matches), along with
accuracy conditioned on agreement vs disagreement -- the actually
measurable "consensus lift".

Passing a single --model still runs for real and reports solo accuracy; it
prints an explicit warning instead of fabricating ensemble numbers when
fewer than two models are supplied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.ensemble_causal_engine import score_ensemble, score_single  # noqa: E402
from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime  # noqa: E402

RESULTS = REPO / "benchmarks" / "results"
_STRATEGIES = ("logits_sum", "log_prob_sum", "consensus_veto")


def load_features(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        required = ("q", "cands", "offsets", "labels", "tasks")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise SystemExit(f"{path}: missing keys {missing}; expected the parity_val200.npz layout")
        return {k: np.asarray(z[k]) for k in required}


def parse_model_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--model must be NAME:adapter.npz:features.npz, got {spec!r}")
    name, adapter_path, features_path = parts
    return name, Path(adapter_path), Path(features_path)


def macro_micro(correct_by_task: dict, total_by_task: dict) -> tuple[float, float]:
    per_task_acc = [correct_by_task[t] / total_by_task[t] for t in total_by_task]
    macro = float(np.mean(per_task_acc)) if per_task_acc else 0.0
    total_correct, total_n = sum(correct_by_task.values()), sum(total_by_task.values())
    micro = total_correct / total_n if total_n else 0.0
    return macro, micro


def run(models: dict) -> dict:
    """models: name -> (RNNSetAdapterRuntime, features dict). Returns the full report."""
    names = list(models)
    n_records = len(next(iter(models.values()))[1]["q"])
    for name, (_, feats) in models.items():
        if len(feats["q"]) != n_records:
            raise SystemExit(f"model {name!r} has {len(feats['q'])} records, "
                              f"expected {n_records} (all --model files must cover the same records)")

    per_model_report: dict = {}
    per_model_preds: dict = {}
    latencies_ms: dict = defaultdict(list)

    for name, (adapter, feats) in models.items():
        q, cands, offsets, labels, tasks = (feats["q"], feats["cands"], feats["offsets"],
                                             feats["labels"], feats["tasks"])
        correct_by_task, total_by_task = defaultdict(int), defaultdict(int)
        preds = np.empty(n_records, dtype=np.int64)
        for i in range(n_records):
            C = cands[offsets[i]:offsets[i + 1]]
            t0 = time.perf_counter()
            result = score_single(adapter, q[i], C)
            latencies_ms[name].append((time.perf_counter() - t0) * 1e3)
            preds[i] = result["pred"]
            task = str(tasks[i])
            total_by_task[task] += 1
            correct_by_task[task] += int(result["pred"] == int(labels[i]))
        macro, micro = macro_micro(correct_by_task, total_by_task)
        per_model_preds[name] = preds
        per_model_report[name] = {
            "n_records": n_records,
            "macro_accuracy": macro,
            "micro_accuracy": micro,
            "per_task_accuracy": {t: correct_by_task[t] / total_by_task[t] for t in total_by_task},
            "latency_ms": {"median": float(np.median(latencies_ms[name])),
                           "p90": float(np.percentile(latencies_ms[name], 90)),
                           "max": float(np.max(latencies_ms[name]))},
        }

    report: dict = {"n_records": n_records, "models": names, "per_model": per_model_report}

    if len(names) < 2:
        report["ensemble"] = None
        report["warning"] = (f"only {len(names)} model(s) supplied; ensemble fusion needs >=2 "
                              f"independent adapters, skipping ensemble metrics")
        return report

    labels_by_name = {name: models[name][1]["labels"] for name in names}
    tasks_ref = models[names[0]][1]["tasks"]
    labels_ref = labels_by_name[names[0]]
    for name in names[1:]:
        if not np.array_equal(labels_by_name[name], labels_ref):
            raise SystemExit(f"model {name!r} has different labels than {names[0]!r}; "
                              f"--model files must be aligned to the same records/labels")

    ensemble_report: dict = {}
    for strategy in _STRATEGIES:
        correct_by_task, total_by_task = defaultdict(int), defaultdict(int)
        agree_correct = agree_total = disagree_correct = disagree_total = 0
        for i in range(n_records):
            inputs = {}
            for name in names:
                feats = models[name][1]
                C = feats["cands"][feats["offsets"][i]:feats["offsets"][i + 1]]
                inputs[name] = {"query": feats["q"][i], "candidates": C}
            out = score_ensemble({name: models[name][0] for name in names}, inputs, strategy)
            task = str(tasks_ref[i])
            total_by_task[task] += 1
            is_correct = int(out["pred"] == int(labels_ref[i]))
            correct_by_task[task] += is_correct
            if out["agreement"]:
                agree_total += 1
                agree_correct += is_correct
            else:
                disagree_total += 1
                disagree_correct += is_correct
        macro, micro = macro_micro(correct_by_task, total_by_task)
        ensemble_report[strategy] = {
            "macro_accuracy": macro,
            "micro_accuracy": micro,
            "per_task_accuracy": {t: correct_by_task[t] / total_by_task[t] for t in total_by_task},
            "agreement_rate": agree_total / n_records,
            "accuracy_when_agree": (agree_correct / agree_total) if agree_total else None,
            "accuracy_when_disagree": (disagree_correct / disagree_total) if disagree_total else None,
        }
    report["ensemble"] = ensemble_report
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, dest="models",
                        metavar="NAME:ADAPTER.npz:FEATURES.npz",
                        help="repeatable; one independent model per flag")
    parser.add_argument("--out", type=Path, default=RESULTS / "ensemble_cpu_benchmark.json")
    args = parser.parse_args(argv)

    models: dict = {}
    for spec in args.models:
        name, adapter_path, features_path = parse_model_spec(spec)
        if name in models:
            raise SystemExit(f"duplicate model name {name!r}")
        if not adapter_path.is_file():
            raise SystemExit(f"model {name!r}: adapter file not found: {adapter_path}")
        if not features_path.is_file():
            raise SystemExit(f"model {name!r}: features file not found: {features_path}")
        adapter = RNNSetAdapterRuntime.from_npz(adapter_path)
        feats = load_features(features_path)
        if feats["cands"].shape[1] != adapter.cfg["in_dim"]:
            raise SystemExit(f"model {name!r}: features dim {feats['cands'].shape[1]} != "
                              f"adapter in_dim {adapter.cfg['in_dim']}")
        models[name] = (adapter, feats)
        print(f"[load] {name}: in_dim={adapter.cfg['in_dim']} d={adapter.cfg['d']} "
              f"n_records={len(feats['q'])} <- {adapter_path.name}, {features_path.name}", flush=True)

    report = run(models)

    print(json.dumps({k: v for k, v in report.items() if k != "per_model"}, indent=2, default=str))
    for name, m in report["per_model"].items():
        print(f"[solo] {name}: macro={m['macro_accuracy']:.4f} micro={m['micro_accuracy']:.4f} "
              f"median_latency_ms={m['latency_ms']['median']:.3f}", flush=True)
    if report.get("warning"):
        print(f"[warn] {report['warning']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[write] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
