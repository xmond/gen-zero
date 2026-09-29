#!/usr/bin/env python3
"""Multithreaded CPU multi-dimensional benchmark for Zero's pre-trained manifold+head pairs.

Evaluates the four pre-trained (64-D, 128-D, 256-D, 896-D) manifold+task-head
pairs against the frozen 930-question test set, entirely on CPU,
using the real INT8 Zero backbone. The backbone forward pass runs exactly
once per record (896-D hidden states, prompt + candidates); each dimension
then applies its own cheap numpy projection and bilinear score to that same
cached state. No task name or ground truth is read anywhere in the prediction
loop -- only after every prediction is frozen does a separate scoring pass
join predictions against ground truth.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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

from gen_zero.causal.zero_runtime import ZeroStandaloneRuntime, build_int8_artifact, find_local_snapshot  # noqa: E402

_ZERO_CPU_SPEC = importlib.util.spec_from_file_location(
    "benchmark_zero_cpu", Path(__file__).resolve().parent / "benchmark_zero_cpu.py")
_zero_cpu = importlib.util.module_from_spec(_ZERO_CPU_SPEC)
sys.modules[_ZERO_CPU_SPEC.name] = _zero_cpu
_ZERO_CPU_SPEC.loader.exec_module(_zero_cpu)

sha256_file = _zero_cpu.sha256_file
read_jsonl = _zero_cpu.read_jsonl
peak_rss_bytes = _zero_cpu.peak_rss_bytes
percentile = _zero_cpu.percentile
summary = _zero_cpu.summary
wilson = _zero_cpu.wilson
environment = _zero_cpu.environment

DATA = Path(os.environ.get("ZERO_DATA_DIR", str(REPO / "benchmarks" / "data")))
ARTIFACTS = Path(os.environ.get("ZERO_ARTIFACTS_DIR", str(REPO / "benchmarks" / "artifacts" / "zero")))
RESULTS = Path(os.environ.get("ZERO_RESULTS_DIR", str(REPO / "benchmarks" / "results")))
TEST_SET = DATA / "all_benchmarks.jsonl"
INT8_ARTIFACT = ARTIFACTS / "zero_int8_v2.safetensors"
DIMS = (64, 128, 256, 896)
GPU_REPORT = RESULTS / "zero_gpu_natural_multidim_report.json"
DEFAULT_OUTPUT = RESULTS / "zero_cpu_natural_multidim_eval_summary.json"


def manifold_path(dim: int) -> Path:
    return ARTIFACTS / f"zero_manifold_natural_gpu_{dim}d.npz"


def task_head_path(dim: int) -> Path:
    return ARTIFACTS / f"zero_task_head_natural_gpu_{dim}d.npz"


class MultiDimManifold:
    """A read-only view of one GPU-fitted ZCA manifold npz, independent of ZeroManifold's schema."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.mean = np.asarray(data["mean"], dtype=np.float32)
            self.projection = np.asarray(data["projection"], dtype=np.float32)
            self.dim = int(data["dim"])
            self.energy_kept = float(data["energy_kept"])
            self.encoder_id = str(data["encoder_id"])
        if self.projection.shape != (self.mean.shape[0], self.dim):
            raise ValueError(f"manifold {path} has inconsistent shapes")


class MultiDimTaskHead:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.W = np.asarray(data["W"], dtype=np.float32)
            self.dim = int(data["dim"])
            self.base_accuracy = float(data["base_accuracy"])
            self.train_accuracy = float(data["train_accuracy"])
        if self.W.shape != (self.dim, self.dim):
            raise ValueError(f"task head {path} has inconsistent shapes")


def project_to_sphere(x: np.ndarray, mu: np.ndarray, zca_mat: np.ndarray) -> np.ndarray:
    centered = x - mu
    proj = centered @ zca_mat
    norm = np.linalg.norm(proj, axis=-1, keepdims=True)
    return proj / np.maximum(norm, 1e-12)


def score_candidates(q0_sphere: np.ndarray, c_states_sphere: np.ndarray, W: np.ndarray) -> np.ndarray:
    q_trans = q0_sphere @ W
    return q_trans @ c_states_sphere.T


def residual_energy_fraction(states: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Fraction of centered test-state energy outside the fitted manifold subspace."""
    centered = np.asarray(states, dtype=np.float64) - mean
    projected = centered @ basis
    total = np.sum(centered * centered, axis=-1)
    retained = np.sum(projected * projected, axis=-1)
    return np.clip((total - retained) / np.maximum(total, 1e-24), 0.0, 1.0)


def load_dim_artifacts(dims: tuple[int, ...]) -> dict[int, dict]:
    loaded = {}
    for dim in dims:
        m_path, h_path = manifold_path(dim), task_head_path(dim)
        if not m_path.exists():
            raise SystemExit(f"no manifold artifact at {m_path}")
        if not h_path.exists():
            raise SystemExit(f"no task head artifact at {h_path}")
        loaded[dim] = {
            "manifold": MultiDimManifold(m_path),
            "head": MultiDimTaskHead(h_path),
            "manifold_path": m_path,
            "head_path": h_path,
            "manifold_sha256": sha256_file(m_path),
            "head_sha256": sha256_file(h_path),
            "manifold_bytes": m_path.stat().st_size,
            "head_bytes": h_path.stat().st_size,
        }
        loaded[dim]["basis"] = np.linalg.qr(
            np.asarray(loaded[dim]["manifold"].projection, dtype=np.float64), mode="reduced")[0]
    return loaded


def score_adapter_candidates(q0_adapter: np.ndarray, c_states_adapter: np.ndarray) -> np.ndarray:
    """Identity-W scoring for the adapter column: plain cosine (both sides are unit-norm)."""
    return c_states_adapter @ q0_adapter


def predict_record(runtime: ZeroStandaloneRuntime, artifacts: dict[int, dict], record: dict,
                   dims: tuple[int, ...], adapter=None) -> dict:
    """Build one record's prediction across every dimension. Never reads ground_truth.

    ``adapter`` (a ``DeepProjectionAdapter``, optional) adds one extra labeled
    "adapter" column to ``per_dim``, scored with identity W (cosine) per the
    training report -- it is a 64-D output space of its own, not the same
    space as the GPU-fitted ``dims`` manifolds, so it is never merged into or
    averaged with any integer-keyed dim entry.
    """
    candidates = list(record["candidates"])
    q0, c_states, info = runtime.encode_prompt_with_candidates(record["context"], candidates)
    per_dim = {}
    for dim in dims:
        art = artifacts[dim]
        t0 = time.perf_counter()
        q0_sphere = project_to_sphere(q0, art["manifold"].mean, art["manifold"].projection)
        c_sphere = project_to_sphere(c_states, art["manifold"].mean, art["manifold"].projection)
        t1 = time.perf_counter()
        scores = score_candidates(q0_sphere, c_sphere, art["head"].W)
        residual = residual_energy_fraction(np.concatenate((q0[None, :], c_states), axis=0),
                                            art["manifold"].mean, art["basis"])
        t2 = time.perf_counter()
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"dim={dim} produced a non-finite score vector")
        index = int(np.argmax(scores))
        per_dim[dim] = {
            "prediction": candidates[index],
            "scores": [float(s) for s in scores],
            "projection_ms": (t1 - t0) * 1000.0,
            "head_ms": (t2 - t1) * 1000.0,
            "residual_energy_prompt": float(residual[0]),
            "residual_energy_candidates_mean": float(np.mean(residual[1:])),
        }
    if adapter is not None:
        t0 = time.perf_counter()
        with torch.inference_mode():
            q0_adapter = adapter(torch.from_numpy(q0).float()).detach().numpy().astype(np.float64)
            c_adapter = adapter(torch.from_numpy(c_states).float()).detach().numpy().astype(np.float64)
        t1 = time.perf_counter()
        scores = score_adapter_candidates(q0_adapter, c_adapter)
        t2 = time.perf_counter()
        if not np.isfinite(scores).all():
            raise FloatingPointError("adapter produced a non-finite score vector")
        index = int(np.argmax(scores))
        per_dim["adapter"] = {
            "prediction": candidates[index],
            "scores": [float(s) for s in scores],
            "projection_ms": (t1 - t0) * 1000.0,
            "head_ms": (t2 - t1) * 1000.0,
        }
    return {
        "id": record["id"], "task": record["task"], "k": len(candidates),
        "tokenize_ms": info["tokenize_ms"], "forward_ms": info["forward_ms"],
        "per_dim": per_dim,
    }


def order_control_record(runtime: ZeroStandaloneRuntime, artifacts: dict[int, dict], record: dict,
                         dims: tuple[int, ...], rng: random.Random, adapter=None) -> dict:
    shuffled = list(record["candidates"])
    rng.shuffle(shuffled)
    prediction = predict_record(runtime, artifacts, {**record, "candidates": shuffled}, dims, adapter=adapter)
    keys = list(dims) + (["adapter"] if adapter is not None else [])
    return {key: shuffled[int(np.argmax(prediction["per_dim"][key]["scores"]))] for key in keys}


def _score_column(predictions: list[dict], ground_truth: dict[str, str], key,
                  order_control_raw: list[dict]) -> dict:
    """Scores one column (an integer dim, or the labeled "adapter" column) against ground truth."""
    by_id = {e["id"]: e for e in predictions}
    per_task = defaultdict(list)
    for entry in predictions:
        gt = ground_truth[entry["id"]]
        correct = entry["per_dim"][key]["prediction"] == gt
        per_task[entry["task"]].append({"correct": correct, "k": entry["k"], "ground_truth": gt,
                                         "total_ms": entry["tokenize_ms"] + entry["forward_ms"]
                                         + entry["per_dim"][key]["projection_ms"]
                                         + entry["per_dim"][key]["head_ms"]})
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
    micro = sum(sum(e["correct"] for e in v) for v in per_task.values()) / len(predictions)
    macro = statistics.fmean(t["accuracy"] for t in tasks.values())
    macro_chance = statistics.fmean(t["chance"] for t in tasks.values())
    macro_majority = statistics.fmean(t["majority"] for t in tasks.values())
    checked = sum(1 for oc in order_control_raw if key in oc)
    agree = sum(1 for oc in order_control_raw
               if key in oc and oc[key] == by_id[oc["_id"]]["per_dim"][key]["prediction"])
    return {
        "records": len(predictions), "micro": micro, "macro": macro,
        "macro_chance": macro_chance, "macro_majority": macro_majority, "per_task": tasks,
        "candidate_order_control": {"checked": checked, "agree": agree,
                                    "agreement": (agree / checked) if checked else None},
    }


def score_predictions(predictions: list[dict], ground_truth: dict[str, str], dims: tuple[int, ...],
                      order_control_raw: list[dict]) -> dict:
    """Joins frozen predictions against ground truth. Called only after the prediction loop ends."""
    return {dim: _score_column(predictions, ground_truth, dim, order_control_raw) for dim in dims}


def cmd_run(args) -> int:
    dims = DIMS
    dataset_sha256 = sha256_file(TEST_SET)
    test = read_jsonl(TEST_SET)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        test = [r for r in test if r["task"] in wanted]
    if args.limit:
        test = test[: args.limit]
    if not test:
        raise SystemExit("no test records selected")

    if not INT8_ARTIFACT.exists():
        print(f"[zero] building INT8 artifact once -> {INT8_ARTIFACT}", flush=True)
        build_int8_artifact(find_local_snapshot(), INT8_ARTIFACT)

    artifacts = load_dim_artifacts(dims)

    process = psutil.Process()
    rss_before = process.memory_info().rss
    t_load = time.perf_counter()
    runtime = ZeroStandaloneRuntime(int8_artifact=INT8_ARTIFACT, manifold_path=None,
                                    task_head_path=None, max_length=args.max_length,
                                    num_threads=args.num_threads)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this is not a CPU-only run")
    load_seconds = time.perf_counter() - t_load
    rss_loaded = process.memory_info().rss

    adapter = None
    if args.adapter is not None:
        from gen_zero.causal.deep_projection_adapter import DeepProjectionAdapter  # noqa: WPS433 (lazy, mirrors zero_runtime)
        adapter = DeepProjectionAdapter(encoder_id=runtime.encoder_id)
        adapter.load_state_dict(torch.load(args.adapter, map_location="cpu", weights_only=True))
        adapter.eval()
        print(f"[adapter] mounted {args.adapter} (encoder_id={runtime.encoder_id}); "
              "scored with identity W (cosine), reported as a separate 'adapter' column", flush=True)

    rng = random.Random(args.seed)
    predictions, order_control_raw = [], []
    started = time.perf_counter()
    for i, record in enumerate(test):
        entry = predict_record(runtime, artifacts, record, dims, adapter=adapter)
        predictions.append(entry)
        if args.order_control_every and i % args.order_control_every == 0:
            top = order_control_record(runtime, artifacts, record, dims, rng, adapter=adapter)
            order_control_raw.append({"_id": record["id"], **top})
        if (i + 1) % 25 == 0 or i + 1 == len(test):
            print(f"[run] {i + 1}/{len(test)} {time.perf_counter() - started:.0f}s "
                  f"rss {process.memory_info().rss / 2**20:.0f} MiB", flush=True)
    wall = time.perf_counter() - started
    rss_after = process.memory_info().rss
    peak = peak_rss_bytes()

    # Ground truth is read only now, after every prediction across every dimension is frozen.
    ground_truth = {r["id"]: r["ground_truth"] for r in test}
    per_dim_accuracy = score_predictions(predictions, ground_truth, dims, order_control_raw)
    if adapter is not None:
        # A separate, labeled column -- the adapter's 64-D space is not the same
        # space as dims' GPU-fitted 64-D manifold, so it must never be merged
        # into or averaged with per_dim_accuracy[64].
        per_dim_accuracy["adapter"] = _score_column(predictions, ground_truth, "adapter", order_control_raw)

    tokenize_ms = [e["tokenize_ms"] for e in predictions]
    forward_ms = [e["forward_ms"] for e in predictions]
    scoring_summary = {}
    residual_summary = {}
    for dim in dims:
        proj = [e["per_dim"][dim]["projection_ms"] for e in predictions]
        head = [e["per_dim"][dim]["head_ms"] for e in predictions]
        scoring_summary[dim] = {"projection_ms": summary(proj), "head_ms": summary(head)}
        residual_summary[dim] = {
            "prompt": summary([e["per_dim"][dim]["residual_energy_prompt"] for e in predictions]),
            "candidates_mean_per_record": summary([
                e["per_dim"][dim]["residual_energy_candidates_mean"] for e in predictions]),
        }
    if adapter is not None:
        proj = [e["per_dim"]["adapter"]["projection_ms"] for e in predictions]
        head = [e["per_dim"]["adapter"]["head_ms"] for e in predictions]
        scoring_summary["adapter"] = {"projection_ms": summary(proj), "head_ms": summary(head)}

    shared_mean = statistics.fmean(tokenize_ms) + statistics.fmean(forward_ms)
    gpu_report = json.loads(GPU_REPORT.read_text(encoding="utf-8")) if GPU_REPORT.exists() else {}

    RESULTS.mkdir(parents=True, exist_ok=True)
    predictions_paths = {}
    for dim in dims:
        path = args.predictions_dir / f"zero_cpu_natural_multidim_{dim}d_predictions.jsonl"
        predictions_paths[dim] = str(path)
        with open(path, "w", encoding="utf-8") as stream:
            for entry in predictions:
                gt = ground_truth[entry["id"]]
                prediction = entry["per_dim"][dim]["prediction"]
                stream.write(json.dumps({
                    "id": entry["id"], "task": entry["task"], "prediction": prediction,
                    "ground_truth": gt, "correct": prediction == gt,
                    "scores": entry["per_dim"][dim]["scores"], "k": entry["k"],
                }, ensure_ascii=False) + "\n")
    if adapter is not None:
        path = args.predictions_dir / "zero_cpu_natural_multidim_adapter_predictions.jsonl"
        predictions_paths["adapter"] = str(path)
        with open(path, "w", encoding="utf-8") as stream:
            for entry in predictions:
                gt = ground_truth[entry["id"]]
                prediction = entry["per_dim"]["adapter"]["prediction"]
                stream.write(json.dumps({
                    "id": entry["id"], "task": entry["task"], "prediction": prediction,
                    "ground_truth": gt, "correct": prediction == gt,
                    "scores": entry["per_dim"]["adapter"]["scores"], "k": entry["k"],
                }, ensure_ascii=False) + "\n")

    dimensions = {}
    for dim in dims:
        art = artifacts[dim]
        dimensions[str(dim)] = {
            "manifold": {"path": str(art["manifold_path"]), "sha256": art["manifold_sha256"],
                        "bytes": art["manifold_bytes"], "energy_kept": art["manifold"].energy_kept,
                        "gpu_encoder_id": art["manifold"].encoder_id},
            "task_head": {"path": str(art["head_path"]), "sha256": art["head_sha256"],
                         "bytes": art["head_bytes"], "base_accuracy": art["head"].base_accuracy,
                         "train_accuracy": art["head"].train_accuracy},
            "scoring_ms": scoring_summary[dim],
            "test_residual_energy_fraction": residual_summary[dim],
            "total_ms_shared_only": shared_mean,
            "total_ms_effective": (shared_mean + scoring_summary[dim]["projection_ms"]["mean"]
                                   + scoring_summary[dim]["head_ms"]["mean"]),
            "accuracy": per_dim_accuracy[dim],
        }

    adapter_result = None
    if adapter is not None:
        adapter_result = {
            "path": str(args.adapter),
            "sha256": sha256_file(args.adapter),
            "bytes": args.adapter.stat().st_size,
            "encoder_id": runtime.encoder_id,
            "note": "DeepProjectionAdapter's 64-D output space is NOT the same space as "
                    "dims[64]'s GPU-fitted manifold; scored with identity W (cosine) per the "
                    "training report, kept as its own column, never merged into dimensions['64']",
            "scoring_ms": scoring_summary["adapter"],
            "total_ms_shared_only": shared_mean,
            "total_ms_effective": (shared_mean + scoring_summary["adapter"]["projection_ms"]["mean"]
                                   + scoring_summary["adapter"]["head_ms"]["mean"]),
            "accuracy": per_dim_accuracy["adapter"],
        }

    result = {
        "model": "Zero",
        "backbone": runtime.weight_metadata,
        "cpu_encoder_id": runtime.encoder_id,
        "precision": "int8",
        "device": "cpu",
        "cpu_threads": {"torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads()},
        "isolation": {
            "test_records": len(test),
            "dataset_sha256": dataset_sha256,
            "calibration_pool_records": gpu_report.get("total_records"),
            "calibration_pool_states": gpu_report.get("total_states"),
            "calibration_pool_source": gpu_report.get("dataset"),
            "note": "manifold/head artifacts are pre-trained and read-only; ground_truth is never "
                    "read inside the per-record prediction loop, only after every prediction is frozen",
        },
        "memory": {
            "rss_before_load_bytes": rss_before, "rss_after_load_bytes": rss_loaded,
            "rss_after_run_bytes": rss_after, "peak_rss_bytes": peak,
            "note": "resident/peak RSS is for the whole run (all 4 manifolds/heads loaded in one "
                    "process), not per-dimension",
        },
        "load_seconds": load_seconds,
        "shared_latency": {"tokenize_ms": summary(tokenize_ms), "forward_ms": summary(forward_ms)},
        "dimensions": dimensions,
        "adapter": adapter_result,
        "wall_seconds": wall,
        "environment": environment(),
        "predictions_files": predictions_paths,
    }
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    compact = {dim: {"micro": per_dim_accuracy[dim]["micro"], "macro": per_dim_accuracy[dim]["macro"]}
              for dim in dims}
    if adapter is not None:
        compact["adapter"] = {"micro": per_dim_accuracy["adapter"]["micro"],
                              "macro": per_dim_accuracy["adapter"]["macro"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"[run] summary -> {out}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-threads", type=int, default=None,
                        help="PyTorch CPU threads (default: up to 8 logical cores)")
    parser.add_argument("--limit", type=int, default=0, help="records to process (0 = all)")
    parser.add_argument("--tasks", default="", help="comma-separated task filter")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order-control-every", type=int, default=5,
                        help="re-encode every Nth record with shuffled candidates (0 = off)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions-dir", type=Path, default=RESULTS)
    parser.add_argument("--adapter", type=Path, default=None,
                        help="path to a trained DeepProjectionAdapter state_dict (.pt); when given, "
                             "scores it with identity W (cosine) and reports the result as a "
                             "separate 'adapter' column, never merged into the dim=64 numbers "
                             "(default: disabled)")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
