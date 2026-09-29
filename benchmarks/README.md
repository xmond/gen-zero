# Gen-Zero benchmarks

This catalog separates reproducible CPU checks from archived model evaluations. Run commands from the repository root. Each report describes its own inputs and measurement boundary; archived scores are references, not results of the CPU commands below.

## Benchmark philosophy

- **Subsecond latency:** measure wall-clock distributions and report the timed boundary. CLI reflex latency includes process startup and I/O; projection and GEMV probes time different operations. A subsecond goal is not a universal guarantee.
- **Zero-token cognitive decisions:** evaluate action selection without generated reasoning tokens. Refusals and failed calls are errors, not fast decisions.
- **Lyapunov spectral stability:** inspect contraction and spectral behavior of dynamics separately from task accuracy. A finite batch cannot prove global stability.
- **Permutation equivariance:** reorder candidates and count changed choices. An observed zero-flip rate is evidence for the sampled contexts, not an algebraic proof.

## Core benchmark suites

| Area | Entry point | Measurement and scope |
|---|---|---|
| Latency and microbenchmarks | [run_benchmark.py](run_benchmark.py), [latency_suite.py](suites/latency_suite.py) | Rust CLI reflex wall time and sequential throughput on generated contexts. |
| Latency and microbenchmarks | [profile_nanocore_latency.py](suites/profile_nanocore_latency.py) | Projection and MCP payload assembly on supplied fitted artifact and real feature rows; no transport timing. Requires private inputs supplied by the operator. |
| Latency and microbenchmarks | [profile_fused_int8_gemv_cpu.py](suites/profile_fused_int8_gemv_cpu.py) | Single CPU fused INT8 GEMV probe with synthetic weights and real trunk shapes; no model accuracy claim. |
| Equivariance and algebraic guarantees | [equivariance_suite.py](suites/equivariance_suite.py) | Seeded candidate-order flip count through the Rust CLI; failed samples cause a nonzero runner exit. |
| Equivariance and algebraic guarantees | [deadlock_torus_env.py](suites/deadlock_torus_env.py) | Directed torus fixture with deadlock branches for planner ablations. |
| Cognitive runtime and planners | [benchmark_world_model_mcts_ablation.py](suites/benchmark_world_model_mcts_ablation.py) | Paired MCTS lookahead versus a greedy baseline using privileged exact graph dynamics. |
| Cognitive runtime and planners | [run_6_orthogonal_planners_benchmark.py](../python/gen_zero/scripts/run_6_orthogonal_planners_benchmark.py) | CPU fixtures for A* search, MCTS, MPC-CEM, and other planning engines. These are algorithm checks, not learned-model results. |

## Reproduce on a standard CPU

Install this repository's Python dependencies and build the Rust release CLI where required. The CLI runner also needs the model and authentication prerequisites in the [Rust CLI quickstart](../README.md). Run from the repository root:

```bash
# Rust CLI latency and candidate-order sensitivity; writes benchmarks/results/latest_report.{json,md}
cargo build --release --bin gen-zero
python3 benchmarks/run_benchmark.py --binary target/release/gen-zero --latency-runs 50 --equivariance-trials 30

# CPU-only synthetic GEMV probe; JSON on stdout
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/suites/profile_fused_int8_gemv_cpu.py

# Paired torus MCTS ablation; writes JSON and Markdown at --output
python3 benchmarks/suites/benchmark_world_model_mcts_ablation.py --episodes 100 --seed 0 --simulations 32 --max-steps 20 --output benchmarks/results/world_model_mcts_ablation_report.json

# CPU planner fixtures (A*, MCTS, MPC-CEM, others); writes a planner report
PYTHONPATH=python python3 python/gen_zero/scripts/run_6_orthogonal_planners_benchmark.py
```

`latency_suite.py` and `equivariance_suite.py` are invoked by `run_benchmark.py`; `deadlock_torus_env.py` is a fixture invoked by the torus ablation. The runner writes only the two CLI dimensions. The planner and torus scripts have their own outputs.

The nanocore profiler needs a fitted version 2 artifact, extracted feature store, and matching target core manifest. These inputs are outside the public repository. Supply paths explicitly when available:

```bash
python3 benchmarks/suites/profile_nanocore_latency.py \
  --artifact /path/to/distilled_128d_llama70b_boolq.npz \
  --features /path/to/boolq_features.npz \
  --core-manifest /path/to/core_manifest.json \
  --iters 100 --results-dir /path/to/output
```

Use `python3 benchmarks/run_benchmark.py --help` for runner options. Optional error analysis requires an explicitly supplied real predictions JSONL file; raw predictions are not part of this catalog.

## Official result references

- [Canonical Qwen3.5-9B baseline](results/canonical_qwen9b_baseline.md): frozen 13-task model reference with hardware and protocol notes; requires external GPU/model inputs to reproduce.
- [13-Task SOTA Macro 81.52% Dual-70B Manifold Reproduction Guide](#13-task-sota-macro-8152-dual-70b-manifold-reproduction-guide) (below): reproducible paired ablation over frozen Qwen2.5-72B + LLaMA-3.1-70B features; needs only the locally cached feature `.npz` files, no 123B/405B assets.

Compare figures only when workload, hardware, sample set, and timed boundary match. Reference reports are historical evidence, not fresh measurements from `run_benchmark.py`.

A prior "master manifold, 13 tasks" archived report and its random-projection
generator script depended on Mistral-123B features that were never actually
extracted (the feature directory was a broken symlink) and could not be
reproduced. Both were removed on 2026-09-29. The guide below replaces it with
a path that only needs already-extracted, locally verifiable Qwen2.5-72B and
LLaMA-3.1-70B features.

## 13-Task SOTA Macro 81.52% Dual-70B Manifold Reproduction Guide

> **Read this box before the numbers below.** "SOTA" here is this project's internal
> track name for the highest-scoring selection rule in its own search, not a claim of
> beating any published external baseline. The no-search fixed control `concat+bbp`
> (81.60%) actually scores *higher* than the searched Peak SOTA track (81.52%); the
> paired-bootstrap delta `peak_minus_ctl_concat_bbp` is **-0.08pp, 95% CI [-0.85, +0.70]**
> — the two are statistically indistinguishable, and the search does not demonstrably
> beat picking one fixed config. The `Peak macro accuracy >= 82.5` target was **not
> met** (81.52). Balanced accuracy sits around 71 versus ~81.5 raw accuracy on the Peak
> track, which is evidence of class imbalance, not a modeling defect this guide fixes.

This is a 13-task, CPU-only, closed-form (ridge / LDA / BBP probes) evaluation over
**frozen** Qwen2.5-72B and LLaMA-3.1-70B hidden-state features. It needs no GPU, no
LLM inference, and no Mistral-123B or Llama-3.1-405B assets — only the two already
extracted 8192-d feature caches.

### Data prep

Verify the 26 pinned feature files (13 tasks x {Qwen, Llama}) before running anything:

```bash
# If downloading from GitHub Release assets:
# gh release download v0.1.0 --pattern "*.tar.gz" --dir /path/to/extracted_features
# tar -xzf llama70b_13tasks_features.tar.gz -C /path/to/extracted_features/llama70b
# tar -xzf qwen72b_13tasks_features.tar.gz -C /path/to/extracted_features/qwen72b

python3 scripts/download_benchmark_features.py
```

This prints MISSING / SIZE / sha256 OK / MISMATCH per file and exits 0 only when all
26 verify against the pinned manifest. The official pre-extracted packages are published
as release assets on [GitHub Releases v0.1.0](https://github.com/xmond/gen-zero/releases/tag/v0.1.0).

### Commands

```bash
export MASTER_QWEN_DIR=/ebs/data/extracted_features/qwen72b/features
export MASTER_LLAMA_DIR=/ebs/data/extracted_features/llama70b

# Full 13-task Pareto/CALA/conformal search (produces the Peak/Robust/fixed-control table below)
python3 benchmarks/suites/evaluate_manifold_pareto_ensemble.py --workers 3

# Dual ridge-probe ensemble (a separate, deliberately simpler probe; see its own numbers below)
python3 benchmarks/suites/evaluate_dual_70b_72b_ensemble.py

# One-task quick check (fast smoke test; expected Peak track test accuracy 89.71 for massive_en)
python3 benchmarks/suites/evaluate_manifold_pareto_ensemble.py --tasks massive_en --workers 1
```

### Runtime note

The original run used **24 CPU cores** and `--workers 3`. Two tasks are heavy relative
to the rest and dominate wall time: `massive_de` (11,247 training rows) and `boolq`
(9,264 rows); the other 11 tasks have at most ~1,000 rows each. Expect the full
13-task run to take on the order of minutes, not seconds, mostly spent on those two
tasks' 5-fold OOF fits.

### Metrics (from `benchmarks/results/spec21_manifold_pareto_ensemble_report.md`, test split, macro over 13 tasks, %)

| Track | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Peak SOTA (selected on OOF accuracy) | 81.52 | 71.17 | 71.18 |
| Fixed control `concat+bbp` (no search) | 81.60 | 71.50 | 71.49 |
| Certified Robust (selected on OOF balanced acc + F1) | 78.80 | 74.59 | 71.57 |

`peak_minus_ctl_concat_bbp` = -0.08pp, 95% CI [-0.85, +0.70] (paired bootstrap on test
accuracy, rows resampled inside tasks). `peak_minus_peak_single` = +0.37pp, 95% CI
[-0.27, +1.01]. Full per-task breakdown, the CALA ablation, and every other control in
the table (`geo035+bbp`, `llama+lda`, etc.) are in
[`results/spec21_manifold_pareto_ensemble_report.md`](results/spec21_manifold_pareto_ensemble_report.md)'s
per-task table and headline table.

The separate, simpler dual ridge-probe ensemble
([`evaluate_dual_70b_72b_ensemble.py`](suites/evaluate_dual_70b_72b_ensemble.py)) reports
its own macro numbers in
[`results/spec21_dual_70b_72b_ensemble_report.md`](results/spec21_dual_70b_72b_ensemble_report.md):

| Method | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| qwen_probe | 79.91 | 70.21 | 68.89 |
| llama_probe | 79.74 | 68.85 | 68.87 |
| equal_fusion | 80.86 | 70.91 | 70.29 |
| oof_weight_fusion | 80.82 | 70.89 | 70.32 |

This dual-probe script is a deliberately bounded, separate baseline from the larger
Pareto/CALA search above; its equal-fusion macro accuracy (80.86%) does not beat
either published single-model macro reference (Qwen 81.07% / Llama 81.09%), and its
own report notes several tasks have much lower balanced accuracy and macro F1 than
accuracy, again pointing at class imbalance.
