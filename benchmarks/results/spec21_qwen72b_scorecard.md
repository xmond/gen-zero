# Spec 21 grand scorecard: LW-LDA, BBP probe, logit adjustment vs linear / adapter / SupCon (1-SE)

Generated: 2026-09-25T04:12:08Z

Command: `D:\genz\benchmarks\suites\evaluate_spec21_scorecard.py --features-dir D:\genz\features_uncap_v1\q72b_last\features --results-dir D:\genz\benchmarks\results --out-stem spec21_qwen72b_scorecard --device cuda`

## Headline

- Macro accuracy, chosen expert, all 13 tasks: **81.07%** (bootstrap 95% CI [79.86, 82.31], row-level within-task noise only). Micro 82.16%, macro balanced acc 70.42%, macro F1 69.69%.
- Admitted tasks only (12/13, chosen expert also passed the test-split prior gate and had a gate-passing CV pool): macro 80.33%.
- Baseline (Spec 19 Phase 4 report, macro over the same 13 tasks): 75.30% (CI [73.99, 76.59]). Delta: **+5.77 pp**, 13 task(s) changed winner. Baseline feature source matches this run: False (baseline `None`).
- the baseline report stores only correct/n per task, so its predictions cannot be paired with ours; the two macro CIs are independent bootstraps and the delta has no paired test. The paired test is full vs legacy3_tau0, re-run in this script.
- Not admitted: `civil_comments` (no gate-passing CV candidate, test acc below train prior collapsed)

## Per-task results (chosen expert, test split)

| Task | n | Chosen | tau | CV acc | Test acc | Wilson 95% | Bal.acc | Macro F1 | Train prior | Admitted |
|---|---|---|---|---|---|---|---|---|---|---|
| massive_en | 350 | bbp_probe | 0 | 90.70% | 89.71% | [86.09, 92.48] | 91.00% | 90.07% | 12.40% | True |
| massive_de | 350 | bbp_probe | 0 | 91.98% | 91.14% | [87.70, 93.69] | 91.41% | 91.19% | 14.70% | True |
| multinli | 299 | bbp_probe | 0 | 88.40% | 88.29% | [84.16, 91.46] | 88.35% | 88.27% | 34.00% | True |
| pubmedqa | 250 | bbp_probe | 0 | 80.93% | 73.20% | [67.39, 78.31] | 56.33% | 52.98% | 55.87% | True |
| vitaminc | 599 | bbp_probe | 0 | 84.90% | 84.64% | [81.53, 87.31] | 74.10% | 75.86% | 48.70% | True |
| boolq | 300 | bbp_probe | 0 | 91.25% | 88.00% | [83.83, 91.20] | 88.12% | 87.78% | 62.48% | True |
| squad2 | 299 | bbp_probe | 0 | 89.70% | 89.97% | [86.04, 92.88] | 89.99% | 89.92% | 67.60% | True |
| paws | 250 | bbp_probe | 0 | 90.10% | 91.60% | [87.50, 94.44] | 91.53% | 91.58% | 57.90% | True |
| civil_comments | 300 | bbp_probe | 0 | 91.70% | 90.00% | [86.08, 92.91] | 55.88% | 57.86% | 92.10% | False |
| aegis_safety | 250 | bbp_probe+LA0.5 | 0.5 | 85.70% | 82.00% | [76.76, 86.27] | 81.05% | 81.40% | 57.60% | True |
| helpsteer2 | 249 | bbp_probe | 0 | 41.40% | 42.57% | [36.59, 48.78] | 33.82% | 35.01% | 40.30% | True |
| summeval_relevance | 240 | bbp_probe | 0 | 53.30% | 54.58% | [48.26, 60.76] | 40.86% | 33.30% | 52.20% | True |
| summeval_consistency | 144 | bbp_probe | 0 | 87.50% | 88.19% | [81.91, 92.50] | 33.00% | 30.70% | 86.10% | True |

## Winner change vs the 76.06% baseline

| Task | Baseline winner | New winner | Baseline acc [Wilson] | New acc [Wilson] | Delta pp |
|---|---|---|---|---|---|
| massive_en | linear_probe | bbp_probe (changed) | 86.00% [81.97, 89.25] | 89.71% [86.09, 92.48] | +3.71 |
| massive_de | linear_probe | bbp_probe (changed) | 87.14% [83.23, 90.25] | 91.14% [87.70, 93.69] | +4.00 |
| multinli | linear_probe | bbp_probe (changed) | 85.28% [80.82, 88.85] | 88.29% [84.16, 91.46] | +3.01 |
| pubmedqa | supcon_r32 | bbp_probe (changed) | 67.20% [61.16, 72.72] | 73.20% [67.39, 78.31] | +6.00 |
| vitaminc | linear_probe | bbp_probe (changed) | 79.13% [75.70, 82.20] | 84.64% [81.53, 87.31] | +5.51 |
| boolq | adapter_r64 | bbp_probe (changed) | 83.00% [78.34, 86.83] | 88.00% [83.83, 91.20] | +5.00 |
| squad2 | adapter_r128 | bbp_probe (changed) | 86.96% [82.67, 90.31] | 89.97% [86.04, 92.88] | +3.01 |
| paws | linear_probe | bbp_probe (changed) | 88.80% [84.29, 92.14] | 91.60% [87.50, 94.44] | +2.80 |
| civil_comments | adapter_r128 | bbp_probe (changed) | 92.00% [88.37, 94.57] | 90.00% [86.08, 92.91] | -2.00 |
| aegis_safety | linear_probe | bbp_probe+LA0.5 (changed) | 76.80% [71.19, 81.60] | 82.00% [76.76, 86.27] | +5.20 |
| helpsteer2 | supcon_r128 | bbp_probe (changed) | 34.94% [29.29, 41.05] | 42.57% [36.59, 48.78] | +7.63 |
| summeval_relevance | supcon_r128 | bbp_probe (changed) | 34.58% [28.85, 40.80] | 54.58% [48.26, 60.76] | +20.00 |
| summeval_consistency | adapter_r64 | bbp_probe (changed) | 77.08% [69.57, 83.19] | 88.19% [81.91, 92.50] | +11.11 |

## Ablation arms (same OOF pool, same test predictions, different candidate sets)

| Arm | Macro acc | Bootstrap 95% CI |
|---|---|---|
| legacy3_tau0 | 78.38% | [77.13, 79.59] |
| plus_new_heads_tau0 | 80.64% | [79.42, 81.85] |
| legacy3_plus_tau | 78.38% | [77.13, 79.59] |
| full | 81.07% | [79.87, 82.26] |

Paired bootstrap, full minus legacy3_tau0 (same test rows): +2.69 pp, 95% CI [+1.76, +3.61].

Per-task exact McNemar, full vs legacy3_tau0 (a = full only correct, b = legacy only correct):

| Task | Full pick | Legacy pick | only full | only legacy | p (two-sided) |
|---|---|---|---|---|---|
| massive_en | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 5 | 3 | 0.727 |
| massive_de | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 11 | 5 | 0.210 |
| multinli | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 11 | 8 | 0.648 |
| pubmedqa | bbp_probe|tau=0.0 | supcon_r128|tau=0.0 | 8 | 8 | 1.000 |
| vitaminc | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 29 | 7 | 0.000 |
| boolq | bbp_probe|tau=0.0 | supcon_r32|tau=0.0 | 4 | 3 | 1.000 |
| squad2 | bbp_probe|tau=0.0 | supcon_r32|tau=0.0 | 10 | 3 | 0.092 |
| paws | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 4 | 3 | 1.000 |
| civil_comments | bbp_probe|tau=0.0 | supcon_r128|tau=0.0 | 4 | 2 | 0.688 |
| aegis_safety | bbp_probe|tau=0.5 | adapter_r64|tau=0.0 | 20 | 6 | 0.009 |
| helpsteer2 | bbp_probe|tau=0.0 | adapter_r64|tau=0.0 | 30 | 25 | 0.590 |
| summeval_relevance | bbp_probe|tau=0.0 | supcon_r128|tau=0.0 | 47 | 12 | 0.000 |
| summeval_consistency | bbp_probe|tau=0.0 | supcon_r64|tau=0.0 | 5 | 2 | 0.453 |

## Logit adjustment on the linear probe (test split, DIAGNOSTIC, never used for selection)

| Task | Train prior | tau=0 acc / bal / F1 | tau=0.5 acc / bal / F1 | tau=1 acc / bal / F1 |
|---|---|---|---|---|
| civil_comments | 92.1% | 80.3 / 61.5 / 58.8 | 78.3 / 61.7 / 57.9 | 76.3 / 63.4 / 57.7 |
| summeval_consistency | 86.1% | 79.9 / 50.9 / 34.3 | 69.4 / 49.9 / 31.4 | 54.9 / 44.9 / 28.1 |
| squad2 | 67.6% | 90.3 / 90.3 / 90.3 | 90.0 / 90.0 / 90.0 | 90.0 / 90.0 / 90.0 |
| boolq | 62.5% | 88.0 / 88.0 / 87.8 | 87.7 / 87.7 / 87.4 | 87.7 / 87.9 / 87.5 |
| paws | 57.9% | 91.2 / 91.1 / 91.2 | 91.2 / 91.1 / 91.2 | 91.6 / 91.5 / 91.6 |
| aegis_safety | 57.6% | 76.4 / 75.7 / 75.8 | 76.4 / 75.7 / 75.8 | 76.4 / 75.7 / 75.8 |
| pubmedqa | 55.9% | 74.0 / 61.7 / 61.3 | 72.4 / 61.4 / 61.0 | 70.0 / 61.1 / 60.5 |
| summeval_relevance | 52.2% | 40.4 / 35.6 / 28.9 | 37.9 / 34.5 / 28.3 | 35.8 / 33.3 / 28.3 |
| vitaminc | 48.7% | 81.0 / 71.5 / 71.6 | 79.6 / 70.9 / 70.5 | 79.3 / 71.0 / 70.3 |
| helpsteer2 | 40.3% | 37.8 / 35.5 / 33.7 | 37.3 / 35.8 / 33.5 | 34.9 / 35.2 / 32.4 |
| multinli | 34.0% | 87.3 / 87.4 / 87.3 | 87.3 / 87.4 / 87.3 | 87.3 / 87.4 / 87.3 |
| massive_de | 14.7% | 89.4 / 89.4 / 89.3 | 89.1 / 89.3 / 89.1 | 89.1 / 89.4 / 88.8 |
| massive_en | 12.4% | 89.1 / 91.3 / 90.1 | 89.1 / 91.5 / 89.6 | 88.9 / 91.5 / 89.2 |

## Selection ladder (OOF pool, best rank per family and tau)

| Task | Best CV cand. | Threshold | Admissible | Chosen | Chosen params |
|---|---|---|---|---|---|
| massive_en | supcon_r128|tau=0.5 | 90.47% | 15/15 | bbp_probe|tau=0.0 | 5,130 |
| massive_de | bbp_probe|tau=0.0 | 91.80% | 15/15 | bbp_probe|tau=0.0 | 35,802 |
| multinli | bbp_probe|tau=1.0 | 87.26% | 15/15 | bbp_probe|tau=0.0 | 756 |
| pubmedqa | bbp_probe|tau=0.0 | 80.25% | 15/15 | bbp_probe|tau=0.0 | 495 |
| vitaminc | bbp_probe|tau=0.0 | 83.50% | 15/15 | bbp_probe|tau=0.0 | 720 |
| boolq | bbp_probe|tau=0.0 | 90.82% | 15/15 | bbp_probe|tau=0.0 | 3,924 |
| squad2 | bbp_probe|tau=0.5 | 89.26% | 15/15 | bbp_probe|tau=0.0 | 476 |
| paws | bbp_probe|tau=0.0 | 89.27% | 15/15 | bbp_probe|tau=0.0 | 486 |
| civil_comments | bbp_probe|tau=0.0 | 90.83% | 0/15 (FALLBACK: none passed) | bbp_probe|tau=0.0 | 484 |
| aegis_safety | adapter_r64|tau=0.0 | 85.10% | 15/15 | bbp_probe|tau=0.5 | 527 |
| helpsteer2 | bbp_probe|tau=0.0 | 40.41% | 2/15 | bbp_probe|tau=0.0 | 1,340 |
| summeval_relevance | bbp_probe|tau=0.0 | 52.58% | 1/15 | bbp_probe|tau=0.0 | 1,440 |
| summeval_consistency | lw_lda|tau=0.0 | 86.77% | 6/15 | bbp_probe|tau=0.0 | 1,430 |

## Latency (chosen head, CPU NumPy, one row per call, includes Python overhead)

| Task | Median us | p95 us | rows timed |
|---|---|---|---|
| massive_en | 33.7 | 36.1 | 350 |
| massive_de | 33.2 | 36.2 | 350 |
| multinli | 31.9 | 33.4 | 299 |
| pubmedqa | 28.8 | 30.5 | 250 |
| vitaminc | 32.0 | 33.2 | 599 |
| boolq | 26.7 | 28.3 | 300 |
| squad2 | 28.8 | 30.2 | 299 |
| paws | 26.5 | 27.9 | 250 |
| civil_comments | 26.7 | 28.1 | 300 |
| aegis_safety | 29.1 | 30.6 | 250 |
| helpsteer2 | 34.1 | 35.8 | 249 |
| summeval_relevance | 34.5 | 36.8 | 240 |
| summeval_consistency | 36.0 | 59.9 | 144 |

## Status

Verified in this run:
- Every candidate was fitted per fold and on the full train split; OOF and test predictions come from those fits (see `cv_candidates`, `posthoc_test_all_candidates` in the JSON).
- The OOF prior gate, not test accuracy, decides which candidates may be selected; the test gate only decides admission of the already-chosen expert.
- Permuting the test labels leaves the chosen candidate unchanged (unit test `test_selection_does_not_depend_on_test_labels`).

NOT verified / limits:
- CV is 5-fold OOF on one seed and NOT nested; 15 candidates per task share it, so CV accuracy is optimistic.
- No leakage audit was re-run here: train/test overlap checks are inherited from feature extraction and the run does not re-derive them.
- Baseline predictions are not available, so there is no paired test against the 76.06% report itself; the paired test is against the re-run `legacy3_tau0` arm, which uses the canonical rule, not the old ladder.
- Macro CIs resample test rows inside each task; they do not include between-task or training-set variance.
- Latency figures come from whatever box ran this (loadavg None).

NOT done:
- No conformal / abstain layer (Design E), no ETF head, no class-balanced SupCon sampler, no GD-2017 singular-value shrinkage (see `spec21_advanced_heads.py` scope notes).
