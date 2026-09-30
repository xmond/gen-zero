# Frozen Qwen3.5-9B native baseline archive (permanent static reference)

> **Note**: this file records the native Qwen3.5-9B zero-shot / candidate-scoring baseline on 13 tasks (930 samples).
> Computed once and frozen as the standard baseline; any later module, dynamics evaluation, or experiment
> **cites this archive directly and does not recompute it**.

## 1. Hardware and resource baseline
- **Model**: Qwen/Qwen3.5-9B (BF16)
- **VRAM usage**: 18.0 GB VRAM (requires an A100/H100 to run)
- **End-to-end inference latency (P50)**: 170.49 ms / sample
- **Inference throughput**: ~5.0 samples / second

## 2. Macro-average baseline comparison

| Metric | Bare 9B (ar_loglik) | Bare 9B (winning zero-label) | Majority-class random guess | Official Nimble-9B (fine-tuned) | Official Jevons (fine-tuned) |
|---|---:|---:|---:|---:|---:|
| **Macro avg, 11 shared tasks** | **54.85%** | **56.73%** | 51.21% | **80.40%** | **83.54%** |
| **Macro avg, all 13 tasks** | **56.92%** | **58.91%** | 50.26% | — | — |
| **Micro accuracy, all 930 items** | **53.33%** | **55.05%** | — | — | — |

## 3. Per-task native baseline table

| Task | Candidates $K$ | Sample size $n$ | Bare 9B accuracy % | Majority-class baseline % | Random-guess chance % | Beats majority class |
|---|---:|---:|---:|---:|---:|:---:|
| massive_en | 18 | 30 | 50.00% | 23.33% | 5.56% | ✅ Yes |
| massive_de | 18 | 30 | 46.67% | 23.33% | 5.56% | ✅ Yes |
| vitaminc | 3 | 30 | 36.67% | 36.67% | 33.33% | ❌ No |
| boolq | 2 | 30 | 80.00% | 83.33% | 50.00% | ❌ No |
| squad2 | 2 | 30 | 53.33% | 53.33% | 50.00% | ❌ No |
| paws | 2 | 400 | 54.00% | 50.00% | 50.00% | ✅ Yes |
| civil_comments | 2 | 30 | 80.00% | 86.67% | 50.00% | ❌ No |
| aegis_safety | 2 | 30 | 83.33% | 63.33% | 50.00% | ✅ Yes |
| multinli | 3 | 30 | 23.33% | 40.00% | 33.33% | ❌ No |
| pubmedqa | 3 | 30 | 73.33% | 70.00% | 33.33% | ✅ Yes |
| summeval | 5 | 30 | 43.33% | 33.33% | 20.00% | ✅ Yes |
| arc_challenge | 4 | 30 | 93.33% | 36.67% | 25.00% | ✅ Yes |
| gsm8k | 4 | 200 | 48.50% | 25.00% | 25.00% | ✅ Yes |
