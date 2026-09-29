# Spec 21 grand scorecard: LW-LDA, BBP probe, logit adjustment vs linear / adapter / SupCon (1-SE)

Generated: 2026-09-24T12:12:57Z

Command: `D:\genz\benchmarks\suites\evaluate_spec21_scorecard.py --features-dir D:\genz\features_uncap_v1\q9b_diff_compact\features --baseline D:\genz\spec21_baseline.json --results-dir D:\genz\spec21_results --device cuda`

## Headline

- Macro accuracy, chosen expert, all 13 tasks: **79.03%** (bootstrap 95% CI [77.77, 80.28], row-level within-task noise only). Micro 80.26%, macro balanced acc 69.44%, macro F1 68.13%.
- Admitted tasks only (9/13, chosen expert also passed the test-split prior gate and had a gate-passing CV pool): macro 84.59%.
- Baseline (Spec 19 Phase 4 report, macro over the same 13 tasks): 76.06% (CI [74.74, 77.34]). Delta: **+2.97 pp**, 13 task(s) changed winner. Baseline feature source matches this run: True (baseline `q9b_diff_16_24_compact`).
- the baseline report stores only correct/n per task, so its predictions cannot be paired with ours; the two macro CIs are independent bootstraps and the delta has no paired test. The paired test is full vs legacy3_tau0, re-run in this script.
- Not admitted: `civil_comments` (no gate-passing CV candidate, test acc below train prior collapsed); `helpsteer2` (no gate-passing CV candidate); `summeval_relevance` (test acc below train prior); `summeval_consistency` (test acc below train prior)

## Per-task results (chosen expert, test split)

| Task | n | Chosen | tau | CV acc | Test acc | Wilson 95% | Bal.acc | Macro F1 | Train prior | Admitted |
|---|---|---|---|---|---|---|---|---|---|---|
| massive_en | 350 | lw_lda | 0 | 87.50% | 85.71% | [81.66, 88.99] | 84.63% | 85.19% | 12.40% | True |
| massive_de | 350 | lw_lda | 0 | 90.11% | 91.14% | [87.70, 93.69] | 90.61% | 91.57% | 14.70% | True |
| multinli | 299 | bbp_probe | 0 | 85.60% | 86.62% | [82.30, 90.02] | 86.48% | 86.49% | 34.00% | True |
| pubmedqa | 250 | bbp_probe | 0 | 75.33% | 72.40% | [66.55, 77.57] | 55.81% | 52.85% | 55.87% | True |
| vitaminc | 599 | bbp_probe | 0 | 82.70% | 83.31% | [80.11, 86.08] | 71.38% | 73.26% | 48.70% | True |
| boolq | 300 | bbp_probe | 0 | 89.05% | 83.67% | [79.06, 87.42] | 83.07% | 83.18% | 62.48% | True |
| squad2 | 299 | bbp_probe | 0 | 88.40% | 89.30% | [85.28, 92.32] | 89.32% | 89.26% | 67.60% | True |
| paws | 250 | bbp_probe | 0 | 88.50% | 87.60% | [82.94, 91.13] | 87.47% | 87.54% | 57.90% | True |
| civil_comments | 300 | bbp_probe | 0 | 92.00% | 90.33% | [86.46, 93.19] | 56.06% | 58.24% | 92.10% | False |
| aegis_safety | 250 | bbp_probe | 0 | 84.50% | 81.60% | [76.33, 85.91] | 80.59% | 80.96% | 57.60% | True |
| helpsteer2 | 249 | bbp_probe | 0 | 38.80% | 40.56% | [34.65, 46.76] | 30.66% | 31.30% | 40.30% | False |
| summeval_relevance | 240 | bbp_probe | 0 | 54.70% | 50.42% | [44.13, 56.69] | 39.19% | 33.32% | 52.20% | False |
| summeval_consistency | 144 | bbp_probe | 0 | 86.90% | 84.72% | [77.95, 89.69] | 47.51% | 32.48% | 86.10% | False |

## Winner change vs the 76.06% baseline

| Task | Baseline winner | New winner | Baseline acc [Wilson] | New acc [Wilson] | Delta pp |
|---|---|---|---|---|---|
| massive_en | linear_probe | lw_lda (changed) | 82.57% [78.25, 86.19] | 85.71% [81.66, 88.99] | +3.14 |
| massive_de | linear_probe | lw_lda (changed) | 88.00% [84.18, 91.00] | 91.14% [87.70, 93.69] | +3.14 |
| multinli | linear_probe | bbp_probe (changed) | 83.28% [78.63, 87.08] | 86.62% [82.30, 90.02] | +3.34 |
| pubmedqa | adapter_r64 | bbp_probe (changed) | 68.40% [62.40, 73.85] | 72.40% [66.55, 77.57] | +4.00 |
| vitaminc | linear_probe | bbp_probe (changed) | 80.13% [76.75, 83.13] | 83.31% [80.11, 86.08] | +3.18 |
| boolq | linear_probe | bbp_probe (changed) | 82.00% [77.26, 85.93] | 83.67% [79.06, 87.42] | +1.67 |
| squad2 | adapter_r128 | bbp_probe (changed) | 86.96% [82.67, 90.31] | 89.30% [85.28, 92.32] | +2.34 |
| paws | adapter_r64 | bbp_probe (changed) | 88.80% [84.29, 92.14] | 87.60% [82.94, 91.13] | -1.20 |
| civil_comments | adapter_r128 | bbp_probe (changed) | 88.67% [84.58, 91.78] | 90.33% [86.46, 93.19] | +1.66 |
| aegis_safety | linear_probe | bbp_probe (changed) | 73.60% [67.81, 78.68] | 81.60% [76.33, 85.91] | +8.00 |
| helpsteer2 | supcon_r128 | bbp_probe (changed) | 38.96% [33.11, 45.14] | 40.56% [34.65, 46.76] | +1.60 |
| summeval_relevance | adapter_r128 | bbp_probe (changed) | 41.25% [35.21, 47.57] | 50.42% [44.13, 56.69] | +9.17 |
| summeval_consistency | supcon_r32 | bbp_probe (changed) | 86.11% [79.52, 90.83] | 84.72% [77.95, 89.69] | -1.39 |

## Ablation arms (same OOF pool, same test predictions, different candidate sets)

| Arm | Macro acc | Bootstrap 95% CI |
|---|---|---|
| legacy3_tau0 | 75.99% | [74.66, 77.30] |
| plus_new_heads_tau0 | 79.03% | [77.77, 80.27] |
| legacy3_plus_tau | 75.99% | [74.66, 77.30] |
| full | 79.03% | [77.77, 80.27] |

Paired bootstrap, full minus legacy3_tau0 (same test rows): +3.04 pp, 95% CI [+1.96, +4.10].

Per-task exact McNemar, full vs legacy3_tau0 (a = full only correct, b = legacy only correct):

| Task | Full pick | Legacy pick | only full | only legacy | p (two-sided) |
|---|---|---|---|---|---|
| massive_en | lw_lda|tau=0.0 | linear_probe|tau=0.0 | 24 | 13 | 0.099 |
| massive_de | lw_lda|tau=0.0 | linear_probe|tau=0.0 | 15 | 4 | 0.019 |
| multinli | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 17 | 7 | 0.064 |
| pubmedqa | bbp_probe|tau=0.0 | supcon_r32|tau=0.0 | 20 | 8 | 0.036 |
| vitaminc | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 39 | 20 | 0.018 |
| boolq | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 10 | 5 | 0.302 |
| squad2 | bbp_probe|tau=0.0 | adapter_r128|tau=0.0 | 14 | 7 | 0.189 |
| paws | bbp_probe|tau=0.0 | linear_probe|tau=0.0 | 4 | 8 | 0.388 |
| civil_comments | bbp_probe|tau=0.0 | adapter_r128|tau=0.0 | 9 | 4 | 0.267 |
| aegis_safety | bbp_probe|tau=0.0 | supcon_r64|tau=0.0 | 26 | 6 | 0.001 |
| helpsteer2 | bbp_probe|tau=0.0 | supcon_r128|tau=0.0 | 30 | 26 | 0.689 |
| summeval_relevance | bbp_probe|tau=0.0 | supcon_r128|tau=0.0 | 49 | 26 | 0.011 |
| summeval_consistency | bbp_probe|tau=0.0 | supcon_r32|tau=0.0 | 4 | 6 | 0.754 |

## Logit adjustment on the linear probe (test split, DIAGNOSTIC, never used for selection)

| Task | Train prior | tau=0 acc / bal / F1 | tau=0.5 acc / bal / F1 | tau=1 acc / bal / F1 |
|---|---|---|---|---|
| civil_comments | 92.1% | 86.7 / 63.6 / 64.0 | 85.0 / 66.8 / 64.9 | 83.0 / 69.8 / 65.0 |
| summeval_consistency | 86.1% | 74.3 / 51.4 / 34.8 | 66.7 / 49.5 / 32.4 | 53.5 / 43.6 / 24.1 |
| squad2 | 67.6% | 87.3 / 87.3 / 87.3 | 87.6 / 87.6 / 87.6 | 87.3 / 87.3 / 87.3 |
| boolq | 62.5% | 82.0 / 81.4 / 81.5 | 82.3 / 82.0 / 81.9 | 82.3 / 82.2 / 82.0 |
| paws | 57.9% | 89.2 / 89.1 / 89.2 | 88.8 / 88.7 / 88.8 | 88.8 / 88.7 / 88.8 |
| aegis_safety | 57.6% | 73.6 / 72.7 / 72.8 | 73.6 / 72.7 / 72.8 | 73.6 / 72.7 / 72.8 |
| pubmedqa | 55.9% | 67.2 / 53.5 / 51.9 | 65.2 / 52.3 / 51.2 | 65.2 / 53.6 / 53.0 |
| summeval_relevance | 52.2% | 40.8 / 40.3 / 31.1 | 37.9 / 37.6 / 29.3 | 35.4 / 35.7 / 27.9 |
| vitaminc | 48.7% | 80.1 / 71.7 / 71.1 | 80.3 / 72.6 / 71.7 | 79.8 / 72.6 / 71.4 |
| helpsteer2 | 40.3% | 34.1 / 30.2 / 29.1 | 34.9 / 32.5 / 30.6 | 33.3 / 31.9 / 29.1 |
| multinli | 34.0% | 83.3 / 83.2 / 83.2 | 83.3 / 83.2 / 83.2 | 83.6 / 83.5 / 83.5 |
| massive_de | 14.7% | 88.0 / 88.4 / 88.3 | 88.6 / 90.1 / 89.2 | 88.3 / 90.5 / 89.0 |
| massive_en | 12.4% | 82.6 / 83.2 / 82.2 | 82.9 / 84.8 / 83.2 | 81.7 / 84.3 / 81.4 |

## Selection ladder (OOF pool, best rank per family and tau)

| Task | Best CV cand. | Threshold | Admissible | Chosen | Chosen params |
|---|---|---|---|---|---|
| massive_en | lw_lda|tau=0.0 | 86.62% | 15/15 | lw_lda|tau=0.0 | 147,474 |
| massive_de | lw_lda|tau=1.0 | 90.00% | 15/15 | lw_lda|tau=0.0 | 147,474 |
| multinli | bbp_probe|tau=0.0 | 84.77% | 15/15 | bbp_probe|tau=0.0 | 819 |
| pubmedqa | bbp_probe|tau=0.0 | 74.57% | 15/15 | bbp_probe|tau=0.0 | 606 |
| vitaminc | bbp_probe|tau=0.5 | 82.48% | 15/15 | bbp_probe|tau=0.0 | 813 |
| boolq | bbp_probe|tau=0.0 | 88.73% | 15/15 | bbp_probe|tau=0.0 | 4,818 |
| squad2 | bbp_probe|tau=0.0 | 87.18% | 15/15 | bbp_probe|tau=0.0 | 550 |
| paws | bbp_probe|tau=1.0 | 88.09% | 15/15 | bbp_probe|tau=0.0 | 556 |
| civil_comments | bbp_probe|tau=0.0 | 91.13% | 0/15 (FALLBACK: none passed) | bbp_probe|tau=0.0 | 588 |
| aegis_safety | bbp_probe|tau=0.0 | 84.05% | 15/15 | bbp_probe|tau=0.0 | 596 |
| helpsteer2 | bbp_probe|tau=0.0 | 38.02% | 0/15 (FALLBACK: none passed) | bbp_probe|tau=0.0 | 1,390 |
| summeval_relevance | bbp_probe|tau=0.0 | 53.04% | 1/15 | bbp_probe|tau=0.0 | 1,555 |
| summeval_consistency | bbp_probe|tau=0.0 | 85.91% | 4/15 | bbp_probe|tau=0.0 | 1,545 |

## Latency (chosen head, CPU NumPy, one row per call, includes Python overhead)

| Task | Median us | p95 us | rows timed |
|---|---|---|---|
| massive_en | 34.5 | 49.1 | 350 |
| massive_de | 52.8 | 57.0 | 350 |
| multinli | 29.0 | 30.5 | 299 |
| pubmedqa | 28.9 | 30.5 | 250 |
| vitaminc | 29.0 | 30.3 | 599 |
| boolq | 29.0 | 30.8 | 300 |
| squad2 | 28.4 | 30.2 | 299 |
| paws | 28.5 | 29.9 | 250 |
| civil_comments | 28.6 | 30.4 | 300 |
| aegis_safety | 28.4 | 29.6 | 250 |
| helpsteer2 | 61.0 | 71.8 | 249 |
| summeval_relevance | 33.7 | 35.1 | 240 |
| summeval_consistency | 31.4 | 32.4 | 144 |

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
