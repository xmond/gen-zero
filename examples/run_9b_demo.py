#!/usr/bin/env python3
"""Local smoke test for the Qwen3.5-9B feature-adapter: numeric sanity + Lyapunov
stability check, not a benchmark. Requires only numpy.

This is a lightweight local verification, not an accuracy benchmark: it checks that
the adapter loads, produces finite scores, and holds Lyapunov spectral stability
(sigma_max_A < 1). It does not measure task accuracy against a held-out set.

Loads artifacts/qwen35_9b/zero_rnn_set_adapter_qwen35_9b.npz and scores a few
real teacher-validation records when artifacts/qwen35_9b/parity_val200.npz is
present locally (that file is not included in this repository). Without it,
this script falls back to a fixed-seed (seed=0) synthetic query, which is a
smoke test of the code path, not a measurement of model quality. Imports
rnn_set_adapter.py by file path so importing gen_zero's package __init__
(which pulls in torch) is never triggered: this script's only dependency is
numpy.

Latency reported by this script has two distinct meanings, do not conflate them:
  - `latency_ms.load`: one-time cold-start weight loading from the .npz file.
  - `latency_ms.per_record` / `mean_score_call`: warm, per-sample scoring time,
    measured after a warm-up call outside the timed loop.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts" / "qwen35_9b" / "zero_rnn_set_adapter_qwen35_9b.npz"
PARITY_SAMPLE = REPO / "artifacts" / "qwen35_9b" / "parity_val200.npz"
N_RECORDS = 3


def _load_runtime_class():
    spec = importlib.util.spec_from_file_location(
        "rnn_set_adapter", REPO / "python" / "gen_zero" / "causal" / "rnn_set_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RNNSetAdapterRuntime


def _sample_records(in_dim: int):
    """Return (source, [(q, C, label_or_None), ...]) for N_RECORDS records."""
    if PARITY_SAMPLE.exists():
        with np.load(PARITY_SAMPLE) as z:
            q_all, cands, offsets, labels = z["q"], z["cands"], z["offsets"], z["labels"]
        records = [
            (q_all[i], cands[offsets[i]:offsets[i + 1]], int(labels[i]))
            for i in range(min(N_RECORDS, len(q_all)))
        ]
        return "parity_val200.npz (real teacher-validation records)", records
    rng = np.random.default_rng(0)
    records = [(rng.normal(size=in_dim).astype(np.float32),
                rng.normal(size=(5, in_dim)).astype(np.float32), None)
               for _ in range(N_RECORDS)]
    return "seeded synthetic (parity_val200.npz not found)", records


def main():
    if not ARTIFACT.exists():
        print(f"error: artifact not found at {ARTIFACT}", file=sys.stderr)
        return 1

    Runtime = _load_runtime_class()
    t0 = time.perf_counter()
    rt = Runtime.from_npz(ARTIFACT)
    load_ms = (time.perf_counter() - t0) * 1e3

    source, records = _sample_records(rt.cfg["in_dim"])

    rt.score(*records[0][:2])  # warm-up: pay BLAS thread startup outside the timed loop

    results = []
    latencies_ms = []
    for q, C, label in records:
        t0 = time.perf_counter()
        scores = rt.score(q, C)
        latencies_ms.append((time.perf_counter() - t0) * 1e3)
        chosen = int(np.argmax(scores))
        entry = {"n_candidates": int(C.shape[0]), "chosen": chosen,
                 # 3 decimals: OpenBLAS reorders float32 sums differently per thread
                 # count, so scores agree to ~1e-4 but not bit-exactly past that.
                 "scores": [round(float(s), 3) for s in scores]}
        if label is not None:
            entry["label"] = label
            entry["correct"] = chosen == label
        results.append(entry)

    out = {
        "artifact": str(ARTIFACT.relative_to(REPO)),
        "sample_source": source,
        "model": {
            "in_dim": rt.cfg["in_dim"], "d": rt.cfg["d"], "rank": rt.cfg["rank"],
            "think_steps": rt.cfg["think_steps"], "n_heads": rt.cfg["n_heads"],
            "n_layers": rt.cfg["n_layers"], "ffn_dim": rt.cfg["ffn_dim"],
        },
        "stability": {
            "a_scale": round(float(rt.a_scale), 6),
            "rho_max": rt.rho_max,
            "sigma_max_A": round(rt.sigma_max_A, 6),
        },
        "meta": rt.meta,
        "results": results,
        "latency_ms": {
            "load": round(load_ms, 3),
            "per_record": [round(x, 3) for x in latencies_ms],
            "mean_score_call": round(sum(latencies_ms) / len(latencies_ms), 3),
        },
    }
    print(json.dumps(out, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
