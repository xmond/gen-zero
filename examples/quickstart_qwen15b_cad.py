#!/usr/bin/env python3
"""End-to-end CAD demo on the public Qwen2.5-1.5B-Instruct Q4_K_M GGUF.

    pip install -e ./python llama-cpp-python
    python -m gen_zero.scripts.setup_qwen15b_models          # one-time, ~1.1 GB, sha256-verified
    python examples/quickstart_qwen15b_cad.py [--gguf PATH] [--maybe-bias B]

For each (question, context) pair the engine runs two forward passes, with and without the
context, and prints the context-conditioned verbalizer argmax next to the CAD argmax
(delta = cond - alpha * prior). The last two cases have inconclusive contexts: they show
how the "maybe" class behaves.

Honest scope: no calibration head ships with this repository, so the output is
UNCALIBRATED (alpha, temperature and maybe_bias are untuned defaults). Six hand-written
cases are an illustration, not a benchmark; the "expected" column is the author's reading
of each context and proves nothing about accuracy.
"""
from __future__ import annotations

import argparse
import sys
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
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--head", help="optional user-supplied calibration head JSON")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--maybe-bias", type=float, default=0.0, help="added to the maybe score (uncalibrated mode)")
    parser.add_argument("--threads", type=int, default=None)
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

    engine = CADEngine.from_gguf(args.gguf, head_path=args.head, n_ctx=1024, n_threads=args.threads,
                                 alpha=args.alpha, maybe_bias=args.maybe_bias)
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
