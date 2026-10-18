#!/usr/bin/env python3
"""End-to-end CAD demo on the public Qwen2.5-1.5B-Instruct Q4_K_M GGUF.

    pip install -e ./python llama-cpp-python
    python -m gen_zero.scripts.setup_qwen15b_models          # one-time, ~1.1 GB, sha256-verified
    python examples/quickstart_qwen15b_cad.py [--gguf PATH] [--maybe-bias B]

For each (question, context) pair the engine runs two forward passes, with and without the
context, and prints the context-conditioned verbalizer argmax next to the CAD argmax
(delta = cond - alpha * prior). The last two cases have inconclusive contexts: they show
how the "maybe" class behaves.

Honest scope: no calibration head ships with this repository. The 12 hand-written
cases are an exploratory paired benchmark, not an independent holdout set; their labels
are the author's reading of each context and do not establish population accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_GGUF = "models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"

CASES = (
    ("Does the drug reduce blood pressure?",
     "In a randomized trial of 200 adults, systolic pressure fell by 14 mmHg in the treated group "
     "and by 2 mmHg in the placebo group (p < 0.001).", "yes"),
    ("Is the bridge open to traffic?",
     "The city announced on Monday that the bridge was closed indefinitely after inspectors found "
     "cracks in two support beams. No reopening date has been set.", "no"),
    ("Did the new fertilizer increase wheat yield?",
     "Yields rose 9% on the treated plots and fell 7% on a second set of treated plots; the authors "
     "say the difference between the two sets is unexplained.", "maybe"),
    ("Will the company be profitable next year?",
     "The company reported a small loss this year. Analysts disagree about whether new contracts "
     "will cover rising costs.", "maybe"),
    ("Is the museum open on Sundays?",
     "The museum welcomes visitors from Tuesday to Sunday, 10:00 to 17:00. It is closed on Mondays.", "yes"),
    ("Can the vaccine be stored at room temperature for a month?",
     "Stability tests showed the vaccine loses potency after two days at 25 C and must be kept "
     "between 2 and 8 C.", "no"),
    ("Does the drug reduce blood pressure?", "The placebo group improved, but pressure rose in the drug group.", "no"),
    ("Is the bridge open to traffic?", "The report gives the bridge's length but no opening hours.", "maybe"),
    ("Did the new fertilizer increase wheat yield?", "Treated plots yielded 20% more wheat than controls.", "yes"),
    ("Will the company be profitable next year?", "The forecast does not include projected revenue or costs.", "maybe"),
    ("Is the museum open on Sundays?", "The museum is closed on weekends, including Sunday.", "no"),
    ("Can the vaccine be stored at room temperature for a month?", "No stability measurements were performed.", "maybe"),
)


def benchmark(engine, gguf_path, load_seconds):
    """Paired, sequential full inference timings; fail if any pair changes raw logits."""
    import numpy as np

    records = {}
    for name, adaptive, cached, dynamic in (("baseline", False, False, False),
                                            ("optimized", True, True, True)):
        engine.adaptive = adaptive
        engine.cache_priors = cached
        engine.extractor.dynamic_reuse = dynamic
        engine.extractor.reused_tokens = 0
        engine.prior_cache_hits = 0
        engine._prior_cache.clear()
        started = time.perf_counter()
        results = [engine.classify(q, c) for q, c, _ in CASES]
        elapsed = time.perf_counter() - started
        if len(results) == 0:
            raise RuntimeError("empty CAD benchmark")
        accuracy = sum(r.label == label for r, (_, _, label) in zip(results, CASES)) / len(results)
        brier = sum(sum((r.probabilities[k] - float(k == label)) ** 2 for k in ("yes", "no", "maybe"))
                    for r, (_, _, label) in zip(results, CASES)) / len(results)
        records[name] = {"seconds": elapsed, "end_to_end_seconds": load_seconds + elapsed,
                         "accuracy": accuracy, "brier": brier,
                         "reused_tokens": engine.extractor.reused_tokens,
                         "prior_cache_hits": engine.prior_cache_hits, "results": results}
    for a, b in zip(records["baseline"]["results"], records["optimized"]["results"]):
        np.testing.assert_allclose(a.raw_class_logits, b.raw_class_logits, rtol=0, atol=1e-4)
        np.testing.assert_allclose(a.unconditional_logits, b.unconditional_logits, rtol=0, atol=1e-4)
    pairs = [{"expected": label, "baseline": a.label, "optimized": b.label,
              "baseline_probs": a.probabilities, "optimized_probs": b.probabilities,
              "cond": list(a.raw_class_logits), "prior": list(a.unconditional_logits)}
             for (_, _, label), a, b in zip(CASES, records["baseline"]["results"], records["optimized"]["results"])]
    print(json.dumps({"model": str(gguf_path), "n": len(CASES), "model_load_seconds": load_seconds,
                      "pairs": pairs,
                      **{name: {k: v for k, v in row.items() if k != "results"} for name, row in records.items()}},
                     indent=2))
    baseline, optimized = records["baseline"], records["optimized"]
    if not (optimized["end_to_end_seconds"] < baseline["end_to_end_seconds"]
            and optimized["accuracy"] > baseline["accuracy"]
            and optimized["brier"] < baseline["brier"]
            and optimized["reused_tokens"] > baseline["reused_tokens"]
            and optimized["prior_cache_hits"] > 0):
        raise RuntimeError("CAD paired benchmark did not improve speed, accuracy, and Brier together")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--head", help="optional user-supplied calibration head JSON")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--maybe-bias", type=float, default=0.0, help="added to the maybe score (uncalibrated mode)")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="paired 12-case baseline and optimized benchmark")
    args = parser.parse_args(argv)

    if not Path(args.gguf).is_file():
        print(f"error: GGUF not found at {args.gguf}; run "
              "`python -m gen_zero.scripts.setup_qwen15b_models` first", file=sys.stderr)
        return 1
    try:
        from gen_zero.causal.cad_engine import CADEngine, LABELS
    except ImportError as exc:
        print(f"error: cannot import gen_zero ({exc}); run `pip install -e ./python`", file=sys.stderr)
        return 1

    load_started = time.perf_counter()
    engine = CADEngine.from_gguf(args.gguf, head_path=args.head, n_ctx=1024, n_threads=args.threads,
                                 alpha=args.alpha, maybe_bias=args.maybe_bias, adaptive=args.adaptive)
    load_seconds = time.perf_counter() - load_started
    if args.benchmark:
        benchmark(engine, args.gguf, load_seconds)
        return 0
    print(f"{'expected':<9}{'cond-only':<11}{'CAD':<7}{'p(yes)':>8}{'p(no)':>8}{'p(maybe)':>10}  question")
    for question, context, expected in CASES:
        r = engine.classify(question, context)
        cond_only = LABELS[max(range(len(LABELS)), key=r.raw_class_logits.__getitem__)]
        p = r.probabilities
        print(f"{expected:<9}{cond_only:<11}{r.label:<7}{p['yes']:>8.3f}{p['no']:>8.3f}{p['maybe']:>10.3f}  {question}")
    mode = "calibrated head" if engine.head else "UNCALIBRATED (no head): scores, not confidences"
    print(f"\nmode: {mode}; alpha={engine.alpha}; prefix tokens reused across prompts: "
          f"{engine.extractor.reused_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
