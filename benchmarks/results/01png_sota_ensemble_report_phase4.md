# 01.PNG 13-task SOTA: Spec 19 Folded Deep Residual Adapter + Linear Probe (1-SE Rule)

Generated: 2026-09-24T08:15:40Z

Command: `D:\genz\benchmarks\suites\evaluate_full_13_grand_scorecard.py --features-dir D:\genz\features_uncap_v1\q9b_diff_compact\features --ranks 32,64,128 --device cuda --results-dir D:\genz\benchmarks\results_phase4_compact`

## Headline

- Macro avg (evaluated tasks): **76.06%** (Micro: 77.19%)
- Δ vs Nimble (74.80%): +1.26%
- Δ vs Jev (76.00%): +0.06%
- Δ vs Laya A100 (55.48%): +20.58%
- No Phase 1 report found at the output path; no Phase 1 comparison available.
- Tasks selecting linear probe: 6 (aegis_safety, boolq, massive_de, massive_en, multinli, vitaminc)
- Tasks selecting adapter: 5 (civil_comments, paws, pubmedqa, squad2, summeval_relevance)
- Tasks selecting SupCon: 2 (helpsteer2, summeval_consistency)

## Per-task results

| Dataset | n | Chosen | CV acc | Test acc | Wilson95 | Bal.acc | Macro F1 | Win | Nimble | Jev | ΔJev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MASSIVE en-US | 350 | linear_probe | 83.30% | 82.57% | [78.25, 86.19] | 83.19% | 82.23% | - | 86.90% | 87.40% | -4.83% |
| MASSIVE de-DE | 350 | linear_probe | 87.02% | 88.00% | [84.18, 91.00] | 88.38% | 88.31% | WIN | 83.40% | 86.90% | +1.10% |
| MultiNLI | 299 | linear_probe | 83.70% | 83.28% | [78.63, 87.08] | 83.23% | 83.20% | - | 85.30% | 82.90% | +0.38% |
| PubMedQA | 250 | adapter_r64 | 71.33% | 68.40% | [62.40, 73.85] | 52.97% | 51.10% | - | 75.60% | 77.20% | -8.80% |
| VitaminC | 599 | linear_probe | 79.10% | 80.13% | [76.75, 83.13] | 71.67% | 71.09% | WIN | 76.60% | 80.10% | +0.03% |
| BoolQ | 300 | linear_probe | 87.47% | 82.00% | [77.26, 85.93] | 81.42% | 81.49% | - | 86.00% | 89.70% | -7.70% |
| SQuAD 2.0 | 299 | adapter_r128 | 88.10% | 86.96% | [82.67, 90.31] | 86.97% | 86.94% | WIN | 80.60% | 82.90% | +4.06% |
| PAWS | 250 | adapter_r64 | 88.00% | 88.80% | [84.29, 92.14] | 88.66% | 88.74% | - | 82.80% | 89.20% | -0.40% |
| Civil Comments | 300 | adapter_r128 | 88.90% | 88.67% | [84.58, 91.78] | 56.51% | 58.31% | COLLAPSED | 70.30% | 81.00% | +7.67% |
| Aegis 2.0 | 250 | linear_probe | 81.00% | 73.60% | [67.81, 78.68] | 72.66% | 72.83% | - | 81.20% | 80.40% | -6.80% |
| HelpSteer2 | 249 | supcon_r128 | 36.50% | 38.96% | [33.11, 45.14] | 33.33% | 32.42% | - | 39.00% | 34.10% | +4.86% |
| SummEval relevance | 240 | adapter_r128 | 48.50% | 41.25% | [35.21, 47.57] | 31.50% | 24.64% | - | 49.20% | 35.00% | +6.25% |
| SummEval consistency | 144 | supcon_r32 | 85.90% | 86.11% | [79.52, 90.83] | 40.51% | 40.27% | WIN | 75.70% | 81.20% | +4.91% |

## 1-SE selection ladder

| Task | Linear CV | Adapter CV (rank) | Adapter cleared/collapse-ok | SupCon CV (rank) | SupCon cleared/collapse-ok | Winner |
|---|---|---|---|---|---|---|
| massive_en | 83.30%±0.700 | 83.00%±0.880 (r32) | False/True | 83.60%±0.914 (r64) | False/True | linear_probe |
| massive_de | 87.02%±0.378 | 87.27%±0.624 (r64) | False/True | 86.49%±0.372 (r128) | False/True | linear_probe |
| multinli | 83.70%±1.079 | 82.50%±1.204 (r64) | False/True | 81.70%±1.411 (r128) | False/True | linear_probe |
| pubmedqa | 65.07%±0.777 | 71.33%±0.816 (r64) | True/True | 70.53%±0.854 (r32) | False/True | adapter_r64 |
| vitaminc | 79.10%±0.992 | 80.00%±0.962 (r128) | False/True | 79.50%±1.061 (r32) | False/True | linear_probe |
| boolq | 87.47%±0.377 | 87.42%±0.352 (r64) | False/True | 87.63%±0.383 (r32) | False/True | linear_probe |
| squad2 | 84.60%±0.843 | 88.10%±0.534 (r128) | True/True | 87.00%±1.294 (r32) | False/True | adapter_r128 |
| paws | 87.10%±0.886 | 88.00%±1.323 (r64) | True/True | 87.00%±1.204 (r128) | False/True | adapter_r64 |
| civil_comments | 52.20%±1.814 | 88.90%±1.005 (r128) | True/True | 89.10%±0.600 (r128) | False/False | adapter_r128 |
| aegis_safety | 81.00%±1.475 | 81.10%±0.967 (r128) | False/True | 82.40%±0.187 (r64) | False/True | linear_probe |
| helpsteer2 | 30.00%±1.837 | 34.50%±1.541 (r64) | True/True | 36.50%±1.061 (r128) | True/True | supcon_r128 |
| summeval_relevance | 40.60%±2.261 | 48.50%±1.651 (r128) | True/True | 50.10%±1.065 (r128) | False/True | adapter_r128 |
| summeval_consistency | 70.00%±1.557 | 84.90%±0.748 (r128) | True/True | 85.90%±1.364 (r32) | True/True | supcon_r32 |

## Latency

| Task | Median ms | p95 ms | Median µs | p95 µs |
|---|---|---|---|---|
| massive_en | 0.0128 | 0.0150 | 12.8 | 15.0 |
| massive_de | 0.0095 | 0.0115 | 9.5 | 11.5 |
| multinli | 0.0078 | 0.0082 | 7.8 | 8.2 |
| pubmedqa | 0.0524 | 0.0576 | 52.4 | 57.6 |
| vitaminc | 0.0078 | 0.0082 | 7.8 | 8.2 |
| boolq | 0.0062 | 0.0066 | 6.2 | 6.6 |
| squad2 | 0.0457 | 0.0525 | 45.7 | 52.5 |
| paws | 0.0373 | 0.0429 | 37.3 | 42.9 |
| civil_comments | 0.0460 | 0.0708 | 46.0 | 70.8 |
| aegis_safety | 0.0063 | 0.0067 | 6.3 | 6.7 |
| helpsteer2 | 0.0739 | 0.0876 | 73.9 | 87.6 |
| summeval_relevance | 0.0496 | 0.0552 | 49.6 | 55.2 |
| summeval_consistency | 0.0454 | 0.0557 | 45.4 | 55.7 |

## Verdict

- Macro 76.06% is above Jev (76.00%).
- SupCon cleared the 1-SE bar over its champion on 2/13 task(s): helpsteer2, summeval_consistency.

## Status

What this run verified:
- Test-set row counts match `n_expected_01png`: all tasks.
- Leakage gate fields (`leakage_gate.<source variant>`) present for every task.
- Inference on the official test set is pure-NumPy, CPU-only (`score_folded_numpy` / plain matmul), timed per sample.
- Every adapter/SupCon fit (per CV fold and the full-data refit) passed the folded-vs-unfolded numeric self-check in `train_adapter_gpu` before being used for scoring.

What this run did NOT do:
- No ensemble fusion (single head per task, chosen by the 1-SE ladder — not a fused score).
- No McNemar or other significance test between strategies.
- Single seed per fold (fold seeds are `FOLD_SEED + k`, not repeated/averaged across seeds).
- Nested-CV accuracy is optimistic relative to the held-out test accuracy reported per task.
- This script reuses no prior Baseline/Deep-Wide heads; `vs_prior_heads` is empty by construction.
