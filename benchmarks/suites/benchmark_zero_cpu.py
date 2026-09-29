#!/usr/bin/env python3
"""Multithreaded CPU end-to-end benchmark for Zero (text -> tokens -> Zero -> 64-D -> decision).

Two commands, both real, both single process:

  fit-manifold   Extract label-free Zero states for the locked calibration split
                 (sha256-pinned, zero overlap with the test manifest) and fit the
                 64-D ZCA manifold. Writes the manifold artifact and a report.

  run            Load Zero + manifold, decide every record of the frozen test set
                 with nothing but ``context`` and ``candidates`` as input, and
                 report per-task accuracy next to chance and majority baselines,
                 a candidate-order control, latency breakdown and
                 resident/peak memory.

Nothing here reads a label before the decision is made. Labels are used only
by the scorer after predictions are frozen. No task name is passed to the
runtime. There is no fallback model and no synthetic data path: missing
weights, a missing manifold or a hash mismatch abort the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.zero_runtime import (  # noqa: E402
    MANIFOLD_DIM,
    ZeroStandaloneRuntime,
    build_int8_artifact,
    find_local_snapshot,
)

DATA = REPO / "benchmarks" / "data"
ARTIFACTS = REPO / "benchmarks" / "artifacts" / "zero"
RESULTS = REPO / "benchmarks" / "results"
CALIBRATION = DATA / "calibration_clean_16.jsonl"
CALIBRATION_SHA256 = "bd45f4df430ee7c74440afca44b7880ac033e24013a34345ce4e54f1583efa4a"
TEST_SET = DATA / "all_benchmarks.jsonl"
INT8_ARTIFACT = ARTIFACTS / "zero_int8_v2.safetensors"
MANIFOLD = ARTIFACTS / "zero_manifold_v1.npz"
TASK_HEAD = ARTIFACTS / "zero_task_head_v1.npz"
GIB = 1024 ** 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def peak_rss_bytes() -> int:
    if os.name == "nt":
        return int(psutil.Process().memory_info().peak_wset)
    with open("/proc/self/status", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM not available")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.fmean(values), "p50": percentile(values, .5),
            "p90": percentile(values, .9), "max": max(values)}


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (center - half, center + half)


def environment() -> dict:
    return {
        "hostname": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
        "transformers_imported": "transformers" in sys.modules,
    }


def load_runtime(args, *, with_manifold: bool) -> ZeroStandaloneRuntime:
    if args.precision == "int8" and not INT8_ARTIFACT.exists():
        print(f"[zero] building INT8 artifact once -> {INT8_ARTIFACT}", flush=True)
        build_int8_artifact(find_local_snapshot(), INT8_ARTIFACT)
    manifold_file = getattr(args, "manifold", None) or MANIFOLD
    task_head_file = getattr(args, "task_head_path", None) or TASK_HEAD
    task_head_path = None
    if with_manifold and args.task_head:
        if not task_head_file.exists():
            raise SystemExit(f"no task head at {task_head_file}; run gen_zero.train.train_zero_task_head "
                             f"first, or pass --no-task-head for the unsupervised expert path")
        task_head_path = task_head_file
    runtime = ZeroStandaloneRuntime(
        precision=args.precision,
        int8_artifact=INT8_ARTIFACT if args.precision == "int8" else None,
        manifold_path=manifold_file if with_manifold else None,
        task_head_path=task_head_path,
        max_length=args.max_length,
        num_threads=args.num_threads,
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this is not a CPU-only run")
    return runtime


# --------------------------------------------------------------------------
# fit-manifold
# --------------------------------------------------------------------------

def mean_pairwise_cosine(x: np.ndarray) -> float:
    unit = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    total = unit.sum(axis=0)
    n = x.shape[0]
    return float((total @ total - n) / (n * (n - 1)))


def cmd_fit_manifold(args) -> int:
    observed = sha256_file(CALIBRATION)
    if observed != CALIBRATION_SHA256:
        raise SystemExit(f"calibration split hash mismatch: {observed} != {CALIBRATION_SHA256}")
    records = read_jsonl(CALIBRATION)
    if args.limit:
        records = records[: args.limit]
    runtime = load_runtime(args, with_manifold=False)
    process = psutil.Process()
    states, forward_ms, started = [], [], time.perf_counter()
    for i, record in enumerate(records):
        q0, cands, info = runtime.encode_prompt_with_candidates(record["context"], record["candidates"])
        states.append(q0)
        states.extend(cands)
        forward_ms.append(info["forward_ms"])
        if (i + 1) % 25 == 0:
            print(f"[fit-manifold] {i + 1}/{len(records)} records, {time.perf_counter() - started:.0f}s, "
                  f"rss {process.memory_info().rss / 2**20:.0f} MiB", flush=True)
    matrix = np.stack(states)
    manifold = runtime.fit_manifold(matrix, source=f"{CALIBRATION.name}:{CALIBRATION_SHA256[:12]}",
                                    split="calibration", dim=MANIFOLD_DIM)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifold.save(MANIFOLD)
    projected = manifold.project(matrix)
    report = {
        "artifact": str(MANIFOLD),
        "manifold_sha256": sha256_file(MANIFOLD),
        "encoder_id": runtime.encoder_id,
        "calibration_file": str(CALIBRATION),
        "calibration_sha256": observed,
        "records": len(records),
        "states": int(matrix.shape[0]),
        "hidden": int(matrix.shape[1]),
        "manifold_dim": manifold.dim,
        "energy_kept": manifold.provenance["energy_kept"],
        "mean_pairwise_cosine_raw": mean_pairwise_cosine(matrix.astype(np.float64)),
        "mean_pairwise_cosine_manifold": mean_pairwise_cosine(projected),
        "forward_ms": summary(forward_ms),
        "wall_seconds": time.perf_counter() - started,
        "rss_after_bytes": process.memory_info().rss,
        "peak_rss_bytes": peak_rss_bytes(),
        "environment": environment(),
    }
    out = RESULTS / "zero_manifold_fit_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def assert_disjoint(test: list[dict], calibration: list[dict]) -> dict:
    test_ids = {r["id"] for r in test}
    cal_ids = {r["id"] for r in calibration}
    test_contexts = {hashlib.sha256(r["context"].encode("utf-8")).hexdigest() for r in test}
    cal_contexts = {hashlib.sha256(r["context"].encode("utf-8")).hexdigest() for r in calibration}
    id_overlap, ctx_overlap = test_ids & cal_ids, test_contexts & cal_contexts
    if id_overlap or ctx_overlap:
        raise SystemExit(f"calibration/test overlap: ids={len(id_overlap)} contexts={len(ctx_overlap)}")
    return {"test_records": len(test), "calibration_records": len(calibration),
            "id_overlap": 0, "context_overlap": 0}


def prompt_only_latency(runtime: ZeroStandaloneRuntime, text: str, iterations: int) -> dict:
    """Pure text-in -> 64-D state path with no candidates (the deployment hot path)."""
    tok, fwd, man = [], [], []
    for _ in range(iterations):
        states, info = runtime.encode([text])
        t0 = time.perf_counter()
        runtime.manifold.project(states[0])
        man.append((time.perf_counter() - t0) * 1000.0)
        tok.append(info["tokenize_ms"])
        fwd.append(info["forward_ms"])
    total = [a + b + c for a, b, c in zip(tok, fwd, man)]
    return {"tokens": info["tokens"], "tokenize_ms": summary(tok), "forward_ms": summary(fwd),
            "manifold_ms": summary(man), "total_ms": summary(total)}


def cmd_run(args) -> int:
    manifold_file = getattr(args, "manifold", None) or MANIFOLD
    task_head_file = getattr(args, "task_head_path", None) or TASK_HEAD
    if not manifold_file.exists():
        raise SystemExit(f"no manifold at {manifold_file}; run fit-manifold first")
    train_file = getattr(args, "train_dataset", None) or CALIBRATION
    if train_file == CALIBRATION:
        observed = sha256_file(CALIBRATION)
        if observed != CALIBRATION_SHA256:
            raise SystemExit("calibration split hash mismatch")
    else:
        observed = sha256_file(train_file)
    test = read_jsonl(TEST_SET)
    calibration = read_jsonl(train_file)
    isolation = assert_disjoint(test, calibration)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        test = [r for r in test if r["task"] in wanted]
    if args.limit:
        test = test[: args.limit]
    if not test:
        raise SystemExit("no test records selected")

    process = psutil.Process()
    rss_before = process.memory_info().rss
    t_load = time.perf_counter()
    runtime = load_runtime(args, with_manifold=True)
    load_seconds = time.perf_counter() - t_load
    rss_loaded = process.memory_info().rss
    tensor_bytes = runtime.tensor_bytes()

    rng = random.Random(args.seed)
    predictions, per_task = [], defaultdict(list)
    timings = defaultdict(list)
    order_control = {"checked": 0, "agree": 0}
    started = time.perf_counter()
    for i, record in enumerate(test):
        candidates = list(record["candidates"])
        decision = runtime.decide(record["context"], candidates, rank=args.rank, seed=args.seed)
        predicted = candidates[decision.index]
        entry = {
            "id": record["id"], "task": record["task"], "prediction": predicted,
            "ground_truth": record["ground_truth"], "correct": predicted == record["ground_truth"],
            "k": len(candidates), "scores": [float(s) for s in decision.scores],
            "prompt_tokens": decision.prompt_tokens, "candidate_tokens": decision.candidate_tokens,
            "tokenize_ms": decision.tokenize_ms, "forward_ms": decision.forward_ms,
            "manifold_ms": decision.manifold_ms, "dynamics_ms": decision.dynamics_ms,
            "total_ms": decision.total_ms, "task_head_used": decision.task_head_used,
        }
        if args.order_control_every and i % args.order_control_every == 0:
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            alt = runtime.decide(record["context"], shuffled, rank=args.rank, seed=args.seed)
            order_control["checked"] += 1
            order_control["agree"] += int(shuffled[alt.index] == predicted)
            entry["order_control_prediction"] = shuffled[alt.index]
        predictions.append(entry)
        per_task[record["task"]].append(entry)
        for key in ("tokenize_ms", "forward_ms", "manifold_ms", "dynamics_ms", "total_ms"):
            timings[key].append(entry[key])
        if (i + 1) % 25 == 0 or i + 1 == len(test):
            done = sum(e["correct"] for e in predictions)
            print(f"[run] {i + 1}/{len(test)} acc-so-far {done / len(predictions):.3f} "
                  f"{time.perf_counter() - started:.0f}s rss {process.memory_info().rss / 2**20:.0f} MiB",
                  flush=True)
    wall = time.perf_counter() - started
    rss_after = process.memory_info().rss
    peak = peak_rss_bytes()

    # Scoring happens only now, after every prediction is frozen.
    tasks = {}
    for task, entries in sorted(per_task.items()):
        n = len(entries)
        correct = sum(e["correct"] for e in entries)
        labels = Counter(e["ground_truth"] for e in entries)
        majority = labels.most_common(1)[0][1] / n
        chance = statistics.fmean(1.0 / e["k"] for e in entries)
        low, high = wilson(correct, n)
        tasks[task] = {"n": n, "correct": correct, "accuracy": correct / n, "ci95": [low, high],
                       "chance": chance, "majority": majority,
                       "beats_chance_ci": low > chance, "beats_majority_ci": low > majority,
                       "mean_total_ms": statistics.fmean(e["total_ms"] for e in entries)}
    micro = sum(e["correct"] for e in predictions) / len(predictions)
    macro = statistics.fmean(t["accuracy"] for t in tasks.values())
    macro_chance = statistics.fmean(t["chance"] for t in tasks.values())
    macro_majority = statistics.fmean(t["majority"] for t in tasks.values())

    probe_text = test[0]["context"]
    hot_path = prompt_only_latency(runtime, probe_text, args.latency_iterations)

    result = {
        "model": "Zero",
        "backbone": runtime.weight_metadata,
        "encoder_id": runtime.encoder_id,
        "manifold": {"path": str(manifold_file), "sha256": sha256_file(manifold_file),
                     "provenance": runtime.manifold.provenance},
        "task_head": ({"path": str(task_head_file), "sha256": sha256_file(task_head_file),
                      "provenance": runtime.task_head.provenance}
                     if runtime.task_head is not None else None),
        "precision": args.precision,
        "device": "cpu",
        "cpu_threads": {"torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads()},
        "isolation": {**isolation, "calibration_sha256": observed,
                      "test_sha256": sha256_file(TEST_SET)},
        "memory": {
            "tensor_bytes": tensor_bytes,
            "rss_before_load_bytes": rss_before,
            "rss_after_load_bytes": rss_loaded,
            "rss_after_run_bytes": rss_after,
            "peak_rss_bytes": peak,
            "gpu_bytes": 0,
        },
        "load_seconds": load_seconds,
        "latency_text_to_state": hot_path,
        "latency_decision": {key: summary(values) for key, values in timings.items()},
        "accuracy": {
            "records": len(predictions),
            "micro": micro,
            "macro": macro,
            "macro_chance": macro_chance,
            "macro_majority": macro_majority,
            "per_task": tasks,
            "candidate_order_control": {
                **order_control,
                "agreement": (order_control["agree"] / order_control["checked"]) if order_control["checked"] else None,
            },
        },
        "wall_seconds": wall,
        "environment": environment(),
        "predictions_file": str(args.predictions or RESULTS / f"zero_cpu_{args.precision}_predictions.jsonl"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(result["predictions_file"], "w", encoding="utf-8") as stream:
        for entry in predictions:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    out = args.output or RESULTS / f"zero_cpu_{args.precision}_summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("memory", "latency_text_to_state", "accuracy")},
                     ensure_ascii=False, indent=2))
    print(f"[run] summary -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fit-manifold", "run"))
    parser.add_argument("--precision", choices=("int8", "bf16", "fp32"), default="int8")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-threads", type=int, default=None,
                        help="PyTorch CPU threads (default: up to 8 logical cores)")
    parser.add_argument("--limit", type=int, default=0, help="records to process (0 = all)")
    parser.add_argument("--tasks", default="", help="comma-separated task filter for run")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order-control-every", type=int, default=5,
                        help="re-decide every Nth record with shuffled candidates (0 = off)")
    parser.add_argument("--latency-iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions", type=Path,
                        help="predictions jsonl path (default: results/zero_cpu_<precision>_predictions.jsonl)")
    parser.add_argument("--manifold", type=Path, default=MANIFOLD, help="manifold artifact path")
    parser.add_argument("--task-head-path", dest="task_head_path", type=Path, default=TASK_HEAD,
                        help="task head artifact path")
    parser.add_argument("--train-dataset", dest="train_dataset", type=Path, default=None,
                        help="training dataset path to verify isolation against")
    parser.add_argument("--task-head", dest="task_head", action="store_true", default=True,
                        help="score with the calibrated task head when present (default)")
    parser.add_argument("--no-task-head", dest="task_head", action="store_false",
                        help="force the unsupervised continuous_causal_reasoning_expert path")
    args = parser.parse_args()
    if args.command == "fit-manifold":
        return cmd_fit_manifold(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
