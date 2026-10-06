#!/usr/bin/env python3
"""Measure the real CPU cost of Qwen2.5-0.5B and its 64-D bridge.

No synthetic model or fallback is used. Missing local weights are a hard error.
The emitted JSON distinguishes file/parameter bytes from process RSS; RSS is the
physical process measurement and is expected to exceed the advertised weight size.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import psutil
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.student_manifold_projector import (  # noqa: E402
    StudentCausalDecisionPipeline,
    StudentManifoldProjector,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(seconds: list[float], tokens: int) -> dict:
    total = sum(seconds)
    return {
        "iterations": len(seconds),
        "p50_ms": percentile(seconds, 0.50) * 1000,
        "p90_ms": percentile(seconds, 0.90) * 1000,
        "mean_ms": statistics.fmean(seconds) * 1000,
        "samples_per_second": len(seconds) / total,
        "input_tokens_per_second": tokens * len(seconds) / total,
    }


class PeakRss:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.005):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def local_snapshot(repo_cache: Path) -> Path:
    snapshots = repo_cache / "snapshots"
    candidates = sorted(p for p in snapshots.iterdir() if p.is_dir()) if snapshots.is_dir() else []
    complete = [p for p in candidates if (p / "config.json").exists() and
                ((p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists())]
    if not complete:
        raise FileNotFoundError(
            f"no complete local model snapshot under {snapshots}; model weights are required "
            "and this benchmark will not download or substitute them"
        )
    return complete[-1]


def storage_bytes(snapshot: Path) -> int:
    total, seen = 0, set()
    for item in snapshot.iterdir():
        if item.is_file():
            target = item.resolve()
            if target not in seen:
                seen.add(target)
                total += target.stat().st_size
    return total


def timed(call: Callable[[], object], iterations: int) -> list[float]:
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return durations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", type=Path, default=Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B")
    parser.add_argument("--precision", choices=("bf16", "fp32", "int8", "int4"), default="bf16")
    parser.add_argument("--prompt", default="What evidence should be checked before making a robust decision?")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dynamics-steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 2 or args.warmup < 0 or args.threads < 1:
        parser.error("iterations >= 2, warmup >= 0, and threads >= 1 are required")
    if args.precision == "int4":
        parser.error("INT4 has no supported CPU implementation in this environment; refusing fake quantization")

    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    snapshot = local_snapshot(args.model_cache)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    with PeakRss() as loading_peak:
        model = AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=dtype,
                                          low_cpu_mem_usage=True).eval().to("cpu")
        if args.precision == "int8":
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    rss_loaded = process.memory_info().rss
    # Quantized packed weights are not all reported by ``parameters()``.  Keep
    # this metric honestly named; RSS below is the authoritative runtime total.
    registered_parameter_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

    pipeline = StudentCausalDecisionPipeline(
        StudentManifoldProjector(model.config.hidden_size, 64, rank=args.rank),
        num_decisions=4,
        dynamics_steps=args.dynamics_steps,
        calibrated=False,
    ).eval().to(device="cpu", dtype=dtype)
    encoded = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
    token_count = int(encoded["attention_mask"].sum())

    def encoder_forward():
        with torch.inference_mode():
            return model(**encoded, use_cache=False, return_dict=True).last_hidden_state

    def full_forward():
        batch = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        with torch.inference_mode():
            hidden = model(**batch, use_cache=False, return_dict=True).last_hidden_state
            return pipeline(hidden, batch["attention_mask"])

    with torch.inference_mode():
        cached_hidden = encoder_forward()

    def bridge_forward():
        with torch.inference_mode():
            return pipeline(cached_hidden, encoded["attention_mask"])

    for _ in range(args.warmup):
        encoder_forward()
        full_forward()
    with PeakRss() as inference_peak:
        encoder_times = timed(encoder_forward, args.iterations)
        bridge_times = timed(bridge_forward, args.iterations)
        full_times = timed(full_forward, args.iterations)
    rss_final = process.memory_info().rss
    diagnostic = full_forward()
    result = {
        "model": str(snapshot),
        "precision": args.precision,
        "device": "cpu",
        "cpu": {"logical_cores": psutil.cpu_count(), "threads_used": args.threads},
        "input_tokens": token_count,
        "hidden_size": model.config.hidden_size,
        "manifold_size": 64,
        "memory": {
            "snapshot_storage_bytes": storage_bytes(snapshot),
            "registered_parameter_bytes": registered_parameter_bytes,
            "rss_before_load_bytes": rss_before,
            "rss_after_load_bytes": rss_loaded,
            "rss_load_peak_bytes": loading_peak.peak,
            "rss_inference_peak_bytes": inference_peak.peak,
            "rss_final_bytes": rss_final,
            "rss_model_load_delta_bytes": rss_loaded - rss_before,
            "gpu_peak_bytes": 0,
        },
        "encoder_forward": timing_summary(encoder_times, token_count),
        "cached_hidden_to_uncalibrated_logits": timing_summary(bridge_times, token_count),
        "text_to_uncalibrated_logits": timing_summary(full_times, token_count),
        "bridge": {
            "rank": args.rank,
            "dynamics_steps": args.dynamics_steps,
            "finite": bool(torch.isfinite(diagnostic.logits).all()),
            "semantic_decision_emitted": diagnostic.decision is not None,
            "calibrated": pipeline.calibrated,
        },
        "accuracy": {
            "measured": False,
            "reason": "no trained projector/dynamics/decision artifact or labelled evaluation set was supplied",
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
