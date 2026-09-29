# Spec 21 grand scorecard: LW-LDA, BBP probe, logit adjustment vs linear / adapter / SupCon (1-SE)

Generated: 2026-09-25T02:59:02Z

Command: `D:\genz\benchmarks\suites\evaluate_spec21_scorecard.py --features-dir D:\genz\features_uncap_v1\llama70b_last\features --results-dir D:\genz\benchmarks\results --out-stem spec21_llama70b_scorecard --device cuda`

## Headline

- Macro accuracy, chosen expert, all 13 tasks: **81.09%** (bootstrap 95% CI [79.87, 82.29], row-level within-task noise only). Micro 82.11%, macro balanced acc 70.83%, macro F1 70.33%.
- Admitted tasks only (12/13, chosen expert also passed the test-split prior gate and had a gate-passing CV pool): macro 80.32%.
- Baseline (Spec 19 Phase 4 report, macro over the same 13 tasks): 75.30% (CI [73.99, 76.59]). Delta: **+5.79 pp**, 13 task(s) changed winner. Baseline feature source matches this run: False (baseline `None`).
- the baseline report stores only correct/n per task, so its predictions cannot be paired with ours; the two macro CIs are independent bootstraps and the delta has no paired test. The paired test is full vs legacy3_tau0, re-run in this script.
- Not admitted: `civil_comments` (no gate-passing CV candidate, test acc below train prior collapsed)

## Per-task results (chosen expert, test split)

| Task | n | Chosen | tau | CV acc | Test acc | Wilson 95% | Bal.acc | Macro F1 | Train prior | Admitted |
|---|---|---|---|---|---|---|---|---|---|---|
| massive_en | 350 | lw_lda | 0 | 91.60% | 89.14% | [85.45, 91.99] | 89.20% | 88.91% | 12.40% | True |
| massive_de | 350 | bbp_probe | 0 | 91.93% | 91.43% | [88.03, 93.93] | 91.87% | 91.88% | 14.70% | True |
| multinli | 299 | bbp_probe | 0 | 84.40% | 86.29% | [81.93, 89.73] | 86.36% | 86.34% | 34.00% | True |
| pubmedqa | 250 | bbp_probe | 0 | 79.73% | 76.40% | [70.76, 81.24] | 59.96% | 57.50% | 55.87% | True |
| vitaminc | 599 | bbp_probe | 0 | 82.40% | 83.47% | [80.29, 86.23] | 74.61% | 76.57% | 48.70% | True |
| boolq | 300 | bbp_probe | 0 | 90.98% | 88.33% | [84.21, 91.49] | 87.97% | 88.01% | 62.48% | True |
| squad2 | 299 | bbp_probe | 0 | 90.80% | 91.64% | [87.95, 94.27] | 91.65% | 91.63% | 67.60% | True |
| paws | 250 | bbp_probe+LA1 | 1 | 89.10% | 92.80% | [88.91, 95.40] | 92.79% | 92.79% | 57.90% | True |
| civil_comments | 300 | bbp_probe | 0 | 91.80% | 90.33% | [86.46, 93.19] | 56.06% | 58.24% | 92.10% | False |
| aegis_safety | 250 | bbp_probe | 0 | 85.70% | 81.60% | [76.33, 85.91] | 80.70% | 81.01% | 57.60% | True |
| helpsteer2 | 249 | bbp_probe | 0 | 40.60% | 42.17% | [36.20, 48.38] | 37.66% | 37.58% | 40.30% | True |
| summeval_relevance | 240 | bbp_probe | 0 | 55.20% | 53.75% | [47.43, 59.95] | 40.81% | 33.66% | 52.20% | True |
| summeval_consistency | 144 | bbp_probe | 0 | 87.70% | 86.81% | [80.31, 91.39] | 31.17% | 30.23% | 86.10% | True |

## Winner change vs the 76.06% baseline

| Task | Baseline winner | New winner | Baseline acc [Wilson] | New acc [Wilson] | Delta pp |
|---|---|---|---|---|---|
| massive_en | linear_probe | lw_lda (changed) | 86.00% [81.97, 89.25] | 89.14% [85.45, 91.99] | +3.14 |
| massive_de | linear_probe | bbp_probe (changed) | 87.14% [83.23, 90.25] | 91.43% [88.03, 93.93] | +4.29 |
| multinli | linear_probe | bbp_probe (changed) | 85.28% [80.82, 88.85] | 86.29% [81.93, 89.73] | +1.01 |
| pubmedqa | supcon_r32 | bbp_probe (changed) | 67.20% [61.16, 72.72] | 76.40% [70.76, 81.24] | +9.20 |
| vitaminc | linear_probe | bbp_probe (changed) | 79.13% [75.70, 82.20] | 83.47% [80.29, 86.23] | +4.34 |
| boolq | adapter_r64 | bbp_probe (changed) | 83.00% [78.34, 86.83] | 88.33% [84.21, 91.49] | +5.33 |
| squad2 | adapter_r128 | bbp_probe (changed) | 86.96% [82.67, 90.31] | 91.64% [87.95, 94.27] | +4.68 |
| paws | linear_probe | bbp_probe+LA1 (changed) | 88.80% [84.29, 92.14] | 92.80% [88.91, 95.40] | +4.00 |
| civil_comments | adapter_r128 | bbp_probe (changed) | 92.00% [88.37, 94.57] | 90.33% [86.46, 93.19] | -1.67 |
| aegis_safety | linear_probe | bbp_probe (changed) | 76.80% [71.19, 81.60] | 81.60% [76.33, 85.91] | +4.80 |
| helpsteer2 | supcon_r128 | bbp_probe (changed) | 34.94% [29.29, 41.05] | 42.17% [36.20, 48.38] | +7.23 |
| summeval_relevance | supcon_r128 | bbp_probe (changed) | 34.58% [28.85, 40.80] | 53.75% [47.43, 59.95] | +19.17 |
| summeval_consistency | adapter_r64 | bbp_probe (changed) | 77.08% [69.57, 83.19] | 86.81% [80.31, 91.39] | +9.73 |

## Ablation arms (same OOF pool, same test predictions, different candidate sets)

| Arm | Macro acc | Bootstrap 95% CI |
|---|---|---|
| legacy3_tau0 | 79.76% | [78.52, 80.95] |
| plus_new_heads_tau0 | 80.90% | [79.69, 82.10] |
| legacy3_plus_tau | 79.55% | [78.29, 80.76] |
| full | 81.09% | [79.88, 82.29] |

Paired bootstrap, full minus legacy3_tau0 (same test rows): +1.33 pp, 95% CI [+0.39, +2.26].

Per-task exact McNemar, full vs legacy3_tau0 (a = full only correct, b = legacy only correct):

| Task | Full pick | Legacy pick | only full | only legacy | p (two-sided) |
|---|---|---|---|---|---|
| massive_en | lw_lda|tau=0.0 | linear_probe|tau=0.0 | 3 | 5 | 0.727 |
| massive_de | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 10 | 5 | 0.302 |
| multinli | bbp_probe|tau=0.0 | adapter_r32|tau=0.0 | 12 | 9 | 0.664 |
| pubmedqa | bbp_probe|tau=0.0 | supcon_r32|tau=0.0 | 14 | 13 | 1.000 |
| vitaminc | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 18 | 10 | 0.185 |
| boolq | bbp_probe|tau=0.0 | supcon_r128|tau=0.0 | 9 | 6 | 0.607 |
| squad2 | bbp_probe|tau=0.0 | adapter_r32|tau=0.0 | 15 | 9 | 0.307 |
| paws | bbp_probe|tau=1.0 | adapter_r64|tau=0.0 | 7 | 1 | 0.070 |
| civil_comments | bbp_probe|tau=0.0 | supcon_r128|tau=0.0 | 4 | 2 | 0.688 |
| aegis_safety | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 11 | 9 | 0.824 |
| helpsteer2 | bbp_probe|tau=0.0 | supcon_r32|tau=0.0 | 33 | 24 | 0.289 |
| summeval_relevance | bbp_probe|tau=0.0 | adapter_r128|tau=0.0 | 28 | 17 | 0.135 |
| summeval_consistency | bbp_probe|tau=0.0 | supcon_r128|tau=0.0 | 0 | 2 | 0.500 |

## Logit adjustment on the linear probe (test split, DIAGNOSTIC, never used for selection)

| Task | Train prior | tau=0 acc / bal / F1 | tau=0.5 acc / bal / F1 | tau=1 acc / bal / F1 |
|---|---|---|---|---|
| civil_comments | 92.1% | 85.7 / 64.5 / 63.9 | 83.3 / 64.5 / 62.3 | 80.7 / 65.8 / 61.4 |
| summeval_consistency | 86.1% | 79.9 / 37.2 / 34.3 | 71.5 / 35.2 / 31.4 | 62.5 / 33.1 / 28.9 |
| squad2 | 67.6% | 90.3 / 90.3 / 90.3 | 90.3 / 90.3 / 90.3 | 90.0 / 90.0 / 90.0 |
| boolq | 62.5% | 89.0 / 88.7 / 88.7 | 89.3 / 89.3 / 89.1 | 89.3 / 89.4 / 89.1 |
| paws | 57.9% | 91.2 / 91.1 / 91.2 | 91.2 / 91.1 / 91.2 | 91.2 / 91.1 / 91.2 |
| aegis_safety | 57.6% | 80.8 / 80.1 / 80.3 | 81.2 / 80.6 / 80.7 | 81.2 / 80.6 / 80.7 |
| pubmedqa | 55.9% | 76.0 / 65.2 / 65.7 | 73.6 / 63.4 / 63.4 | 70.8 / 61.5 / 61.0 |
| summeval_relevance | 52.2% | 49.6 / 43.4 / 36.1 | 45.8 / 39.6 / 33.5 | 35.8 / 30.8 / 27.6 |
| vitaminc | 48.7% | 82.1 / 74.7 / 74.7 | 82.0 / 75.7 / 75.0 | 81.1 / 75.5 / 74.1 |
| helpsteer2 | 40.3% | 37.8 / 35.4 / 33.7 | 35.3 / 34.3 / 31.9 | 34.9 / 34.1 / 31.6 |
| multinli | 34.0% | 82.9 / 82.9 / 82.9 | 82.9 / 82.9 / 82.9 | 82.9 / 82.9 / 82.9 |
| massive_de | 14.7% | 90.0 / 90.4 / 90.2 | 89.7 / 90.3 / 90.0 | 89.1 / 90.1 / 89.6 |
| massive_en | 12.4% | 89.7 / 91.2 / 90.2 | 89.4 / 91.1 / 89.9 | 89.1 / 90.9 / 89.5 |

## Selection ladder (OOF pool, best rank per family and tau)

| Task | Best CV cand. | Threshold | Admissible | Chosen | Chosen params |
|---|---|---|---|---|---|
| massive_en | lw_lda|tau=0.0 | 90.92% | 15/15 | lw_lda|tau=0.0 | 147,474 |
| massive_de | bbp_probe|tau=0.0 | 91.77% | 15/15 | bbp_probe|tau=0.0 | 38,394 |
| multinli | bbp_probe|tau=0.0 | 83.28% | 15/15 | bbp_probe|tau=0.0 | 708 |
| pubmedqa | bbp_probe|tau=0.0 | 78.19% | 15/15 | bbp_probe|tau=0.0 | 519 |
| vitaminc | bbp_probe|tau=0.5 | 80.91% | 15/15 | bbp_probe|tau=0.0 | 690 |
| boolq | bbp_probe|tau=0.0 | 90.65% | 15/15 | bbp_probe|tau=0.0 | 4,154 |
| squad2 | bbp_probe|tau=0.5 | 90.31% | 15/15 | bbp_probe|tau=0.0 | 458 |
| paws | adapter_r64|tau=0.0 | 88.56% | 15/15 | bbp_probe|tau=1.0 | 481 |
| civil_comments | bbp_probe|tau=0.0 | 91.29% | 0/15 (FALLBACK: none passed) | bbp_probe|tau=0.0 | 486 |
| aegis_safety | bbp_probe|tau=0.0 | 84.93% | 15/15 | bbp_probe|tau=0.0 | 506 |
| helpsteer2 | bbp_probe|tau=0.0 | 39.55% | 1/15 | bbp_probe|tau=0.0 | 1,200 |
| summeval_relevance | bbp_probe|tau=0.0 | 53.67% | 2/15 | bbp_probe|tau=0.0 | 1,450 |
| summeval_consistency | bbp_probe|tau=0.0 | 86.74% | 4/15 | bbp_probe|tau=0.0 | 1,400 |

## Latency (chosen head, CPU NumPy, one row per call, includes Python overhead)

| Task | Median us | p95 us | rows timed |
|---|---|---|---|
| massive_en | 33.5 | 35.8 | 350 |
| massive_de | 32.5 | 34.7 | 350 |
| multinli | 32.1 | 33.5 | 299 |
| pubmedqa | 32.0 | 33.6 | 250 |
| vitaminc | 31.9 | 33.4 | 599 |
| boolq | 26.7 | 28.0 | 300 |
| squad2 | 29.0 | 30.2 | 299 |
| paws | 26.2 | 27.4 | 250 |
| civil_comments | 29.1 | 30.6 | 300 |
| aegis_safety | 29.1 | 30.4 | 250 |
| helpsteer2 | 32.6 | 33.6 | 249 |
| summeval_relevance | 34.3 | 37.7 | 240 |
| summeval_consistency | 32.6 | 34.0 | 144 |

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
