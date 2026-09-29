#!/usr/bin/env python3
"""Fail-closed CPU benchmark for a converted Gen-Zero Qwen artifact."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.model.qwen_1gb_converter import load_native_artifact  # noqa: E402


class PeakRSS:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.005):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "iterations": len(samples),
        "mean_ms": statistics.fmean(samples) * 1000,
        "p50_ms": statistics.median(samples) * 1000,
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))] * 1000,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO / "benchmarks/results/gen_zero_qwen_1gb.pt")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt", default="在证据不足时，应该如何做出稳健决策？")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"converted artifact does not exist: {args.model}")
    if args.iterations < 2 or args.warmup < 0 or args.threads < 1:
        parser.error("iterations >= 2, warmup >= 0, threads >= 1 are required")
    tokenizer_path = args.tokenizer or args.model.with_suffix("").with_name(args.model.stem + "_tokenizer")
    if not tokenizer_path.is_dir():
        parser.error(f"local tokenizer directory does not exist: {tokenizer_path}")

    from transformers import AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    with PeakRSS() as load_peak:
        model, bridge, provenance = load_native_artifact(args.model)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True,
                                                   trust_remote_code=False)
    rss_loaded = process.memory_info().rss
    if any(parameter.device.type != "cpu" for parameter in (*model.parameters(), *bridge.parameters())):
        raise RuntimeError("non-CPU parameter detected")
    parameter_count = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in bridge.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        p.numel() * p.element_size() for p in bridge.parameters()
    )

    def full_chain() -> dict[str, object]:
        encoded = tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
            hidden = output.hidden_states[-1]
            causal_state = bridge(hidden[:, -1, :])
            # This is the native LM next-token decision.  The 64-D state has no
            # calibrated task decision head, and the report says so explicitly.
            token_id = output.logits[:, -1, :].argmax(-1)
        return {"hidden": hidden, "causal_state": causal_state, "token_id": token_id,
                "tokens": int(encoded["attention_mask"].sum())}

    for _ in range(args.warmup):
        full_chain()
    times: list[float] = []
    with PeakRSS() as inference_peak:
        for _ in range(args.iterations):
            started = time.perf_counter()
            diagnostic = full_chain()
            times.append(time.perf_counter() - started)
    rss_final = process.memory_info().rss
    token_id = int(diagnostic["token_id"].item())
    result = {
        "artifact": str(args.model.resolve()),
        "device": "cpu",
        "cuda_used": False,
        "threads": args.threads,
        "input_tokens": diagnostic["tokens"],
        "parameters": parameter_count,
        "artifact_bytes": args.model.stat().st_size,
        "registered_parameter_bytes": parameter_bytes,
        "memory": {
            "rss_before_load_bytes": rss_before,
            "rss_after_load_bytes": rss_loaded,
            "rss_model_load_delta_bytes": rss_loaded - rss_before,
            "rss_load_peak_bytes": load_peak.peak,
            "rss_inference_peak_bytes": inference_peak.peak,
            "rss_final_bytes": rss_final,
            "gpu_peak_bytes": 0,
        },
        "text_to_hidden_to_manifold_to_token_decision": summary(times),
        "output": {
            "hidden_shape": list(diagnostic["hidden"].shape),
            "causal_state_shape": list(diagnostic["causal_state"].shape),
            "finite": bool(torch.isfinite(diagnostic["causal_state"]).all()),
            "next_token_id": token_id,
            "next_token_text": tokenizer.decode([token_id]),
            "semantic_64d_decision_calibrated": False,
        },
        "provenance": provenance,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
