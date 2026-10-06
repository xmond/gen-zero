# Manifold-Pareto dual 70B/72B ensemble: geometric fusion + CALA gate + conformal control

Generated 2026-09-25T11:41:33.139390+00:00 on `DESKTOP-B0ALJME` (cores 24, loadavg at start [1.38, 7.24, 5.73]).

Command: `suites/evaluate_manifold_pareto_ensemble.py --qwen-dir <FEATURES_ROOT>/q --llama-dir <FEATURES_ROOT>/l --out <FEATURES_ROOT>/out --workers 3 --skip-references`

## Headline (test split, macro over 13 tasks, %)

| Arm | Accuracy | Balanced acc | Macro F1 |
|---|---:|---:|---:|
| Peak SOTA track (selected on OOF accuracy) | 81.52 | 71.17 | 71.18 |
|   matched control: same search, single models only | 81.15 | 70.86 | 70.76 |
| Certified Robust track (selected on OOF BA+F1) | 78.80 | 74.59 | 71.57 |
|   matched control: same search, single models only | 78.36 | 74.41 | 71.53 |
| fixed control `concat+bbp` (no search) | 81.60 | 71.50 | 71.49 |
| fixed control `concat+lda` (no search) | 79.38 | 70.48 | 70.73 |
| fixed control `geo035+bbp` (no search) | 80.99 | 70.28 | 69.69 |
| fixed control `geo035+lda` (no search) | 80.98 | 73.59 | 72.93 |
| fixed control `geo100+bbp` (no search) | 81.31 | 70.83 | 69.89 |
| fixed control `geo100+lda` (no search) | 81.53 | 74.15 | 73.32 |
| fixed control `llama+bbp` (no search) | 81.04 | 70.89 | 70.97 |
| fixed control `llama+lda` (no search) | 78.61 | 69.28 | 69.50 |
| fixed control `qwen+bbp` (no search) | 80.99 | 70.32 | 69.59 |
| fixed control `qwen+lda` (no search) | 78.10 | 68.92 | 69.05 |

External references (different protocol, not matched): published Qwen-72B n/a, Llama-70B n/a, earlier 256-d dual ensemble n/a.

## Target check

- Peak macro accuracy >= 82.5: **False** (81.52)
- Peak beats published Qwen-72B scorecard: **None**
- Robust macro balanced accuracy >= 76: **False** (74.59)
- Robust macro F1 >= 76: **False** (71.57)
- Robust tasks with a collapse or a zero-recall class on test: ['summeval_consistency']
- Robust tasks that no candidate could admit through the OOF gate: none
- Peak tasks that no candidate could admit through the OOF gate: ['civil_comments']

## Paired bootstrap on test accuracy (macro, pp, rows resampled inside tasks)

| Comparison | Delta pp | 95% CI |
|---|---:|---|
| peak_minus_peak_single | +0.37 | [-0.27, +1.01] |
| peak_minus_ctl_qwen_bbp | +0.53 | [-0.30, +1.35] |
| peak_minus_ctl_llama_bbp | +0.48 | [-0.34, +1.28] |
| robust_minus_robust_single | +0.44 | [-0.24, +1.12] |
| peak_minus_ctl_concat_bbp | -0.08 | [-0.85, +0.70] |
| ctl_geo100_bbp_minus_ctl_concat_bbp | -0.29 | [-1.05, +0.48] |

## CALA ablation (fixed base, no search; macro test %, delta vs the raw head in pp)

| Base head | Prior shift | Accuracy | d Acc | Balanced acc | d BA | Macro F1 | d F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| geo100+bbp | raw | 81.31 | +0.00 | 70.83 | +0.00 | 69.89 | +0.00 |
| geo100+bbp | static:t1 | 63.61 | -17.70 | 69.30 | -1.53 | 63.68 | -6.21 |
| geo100+bbp | entropy:g2:t1 | 73.90 | -7.42 | 73.35 | +2.52 | 69.39 | -0.50 |
| geo100+bbp | margin:g2:t1 | 76.10 | -5.21 | 73.54 | +2.71 | 70.54 | +0.65 |
| geo100+bbp | centroid:g2:t1 | 67.26 | -14.06 | 71.16 | +0.32 | 66.20 | -3.69 |
| geo100+lda | raw | 81.53 | +0.00 | 74.15 | +0.00 | 73.32 | +0.00 |
| geo100+lda | static:t1 | 58.11 | -23.42 | 66.93 | -7.22 | 59.71 | -13.61 |
| geo100+lda | entropy:g2:t1 | 65.50 | -16.03 | 69.97 | -4.18 | 64.80 | -8.52 |
| geo100+lda | margin:g2:t1 | 65.98 | -15.55 | 69.01 | -5.14 | 64.79 | -8.53 |
| geo100+lda | centroid:g2:t1 | 61.48 | -20.05 | 68.62 | -5.53 | 62.33 | -10.99 |

## Per task

| Task | n_train | Peak config | Peak OOF acc | Peak test acc | Single-model control acc | Robust config | Robust OOF obj | Robust test BA | Robust test F1 | min recall | Robust admitted |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| massive_en | 1000 | `llama+bbp|raw` | 90.50 | 89.71 | 89.71 | `llama+lda|raw` | 91.04 | 89.20 | 88.91 | 54.5 | True |
| massive_de | 11247 | `fuse0.5+bbp|raw` | 92.35 | 92.00 | 91.43 | `fuse0.5+bbp|raw` | 92.02 | 91.42 | 92.16 | 63.6 | True |
| multinli | 1000 | `qwen+bbp|raw` | 88.70 | 88.29 | 88.29 | `qwen+bbp|raw` | 88.67 | 88.35 | 88.27 | 87.0 | True |
| pubmedqa | 750 | `geo100+lda|raw` | 80.93 | 76.80 | 73.20 | `geo035+lda|raw` | 62.34 | 59.95 | 58.47 | 8.1 | True |
| vitaminc | 1000 | `qwen+bbp|raw` | 84.60 | 84.64 | 84.64 | `qwen+bbp|raw` | 79.60 | 74.10 | 75.86 | 42.4 | True |
| boolq | 9264 | `concat+bbp|raw` | 91.35 | 89.33 | 88.33 | `geo035+bbp|raw` | 91.15 | 87.51 | 87.38 | 86.5 | True |
| squad2 | 1000 | `geo035+bbp|raw` | 91.40 | 91.30 | 91.64 | `fuse0.25+bbp|margin:g1:t0.5` | 91.25 | 92.65 | 92.64 | 91.3 | True |
| paws | 1000 | `concat+bbp|raw` | 90.30 | 92.40 | 91.60 | `concat+bbp|raw` | 90.20 | 92.35 | 92.39 | 90.9 | True |
| civil_comments | 1000 | `llama+bbp|raw` | 92.30 | 90.33 | 90.33 | `qwen+bbp|static:t0.5` | 69.08 | 79.29 | 68.80 | 75.0 | True |
| aegis_safety | 1000 | `llama+bbp|raw` | 86.60 | 81.60 | 81.60 | `llama+bbp|raw` | 86.16 | 80.70 | 81.01 | 74.1 | True |
| helpsteer2 | 1000 | `geo035+bbp|raw` | 44.20 | 41.37 | 42.17 | `concat+bbp|static:t0.5` | 36.14 | 36.07 | 33.84 | 19.4 | True |
| summeval_relevance | 1000 | `llama+bbp|raw` | 57.50 | 53.75 | 53.75 | `llama+bbp|centroid:g2:t0.5` | 43.27 | 47.12 | 35.57 | 35.5 | True |
| summeval_consistency | 1000 | `qwen+bbp|raw` | 87.70 | 88.19 | 88.19 | `qwen+bbp|entropy:g2:t0.5` | 42.85 | 51.02 | 35.07 | 0.0 | True |

## Per-task fixed controls, test accuracy (no search)

| Task | concat+bbp | concat+lda | geo035+bbp | geo035+lda | geo100+bbp | geo100+lda | llama+bbp | llama+lda | qwen+bbp | qwen+lda |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| massive_en | 90.29 | 90.00 | 90.29 | 89.43 | 90.57 | 88.86 | 89.71 | 89.14 | 89.71 | 89.14 |
| massive_de | 91.43 | 91.71 | 91.71 | 92.00 | 92.29 | 92.00 | 91.43 | 92.00 | 90.86 | 91.43 |
| multinli | 89.30 | 86.62 | 88.63 | 87.29 | 89.30 | 86.96 | 86.29 | 82.61 | 88.29 | 86.96 |
| pubmedqa | 76.40 | 75.20 | 77.60 | 75.60 | 78.00 | 76.80 | 76.40 | 71.60 | 73.20 | 72.80 |
| vitaminc | 84.14 | 83.64 | 83.64 | 84.81 | 83.47 | 84.64 | 83.47 | 81.30 | 84.64 | 80.97 |
| boolq | 89.33 | 84.33 | 87.67 | 87.00 | 88.67 | 87.67 | 88.33 | 88.00 | 88.00 | 85.00 |
| squad2 | 92.64 | 91.64 | 91.30 | 90.64 | 91.97 | 91.64 | 91.64 | 91.30 | 89.97 | 87.96 |
| paws | 92.40 | 90.40 | 92.00 | 92.80 | 91.60 | 94.00 | 91.60 | 85.20 | 91.60 | 90.00 |
| civil_comments | 90.33 | 89.00 | 89.67 | 90.00 | 90.33 | 90.33 | 90.33 | 90.00 | 90.00 | 87.33 |
| aegis_safety | 81.60 | 82.80 | 78.80 | 80.80 | 78.80 | 81.20 | 81.60 | 80.40 | 81.20 | 76.40 |
| helpsteer2 | 39.76 | 39.76 | 41.37 | 42.57 | 40.96 | 42.17 | 42.17 | 41.37 | 42.57 | 41.37 |
| summeval_relevance | 55.00 | 40.00 | 53.33 | 51.67 | 52.92 | 55.42 | 53.75 | 45.00 | 54.58 | 41.25 |
| summeval_consistency | 88.19 | 86.81 | 86.81 | 88.19 | 88.19 | 88.19 | 86.81 | 84.03 | 88.19 | 84.72 |

## Conformal (alpha 0.1; model refit on 80% of train, calibrated on 20%)

| Task | Track | Marginal coverage | Min class coverage | Mean set size | Singleton rate | Singleton acc | Trivial classes |
|---|---|---:|---:|---:|---:|---:|---|
| massive_en | peak | 93.1 | 58.3 | 10.63 | 0.0 | n/a | [0, 1, 3, 4, 9, 10, 13, 14, 15] |
| massive_en | robust | 91.1 | 50.0 | 10.61 | 0.0 | n/a | [0, 1, 3, 4, 9, 10, 13, 14, 15] |
| massive_de | peak | 89.4 | 55.6 | 1.13 | 82.3 | 91.0 | - |
| massive_de | robust | 89.4 | 55.6 | 1.13 | 82.3 | 91.0 | - |
| multinli | peak | 90.0 | 87.0 | 1.05 | 94.0 | 90.0 | - |
| multinli | robust | 90.0 | 87.0 | 1.05 | 94.0 | 90.0 | - |
| pubmedqa | peak | 89.2 | 81.1 | 1.73 | 29.6 | 83.8 | - |
| pubmedqa | robust | 90.0 | 83.8 | 1.82 | 22.4 | 82.1 | - |
| vitaminc | peak | 86.8 | 65.2 | 1.17 | 82.6 | 86.7 | - |
| vitaminc | robust | 86.8 | 65.2 | 1.17 | 82.6 | 86.7 | - |
| boolq | peak | 87.7 | 85.6 | 0.98 | 97.7 | 89.8 | - |
| boolq | robust | 85.7 | 84.9 | 0.94 | 94.3 | 90.8 | - |
| squad2 | peak | 95.7 | 94.0 | 1.16 | 84.3 | 94.8 | - |
| squad2 | robust | 93.6 | 91.3 | 1.05 | 95.3 | 93.3 | - |
| paws | peak | 91.6 | 89.3 | 1.00 | 100.0 | 91.6 | - |
| paws | robust | 91.6 | 89.3 | 1.00 | 100.0 | 91.6 | - |
| civil_comments | peak | 92.7 | 87.5 | 1.28 | 72.3 | 89.9 | - |
| civil_comments | robust | 93.7 | 90.6 | 1.49 | 51.0 | 87.6 | - |
| aegis_safety | peak | 85.2 | 83.3 | 1.10 | 90.0 | 83.6 | - |
| aegis_safety | robust | 85.2 | 83.3 | 1.10 | 90.0 | 83.6 | - |
| helpsteer2 | peak | 85.9 | 80.6 | 3.18 | 0.4 | 100.0 | - |
| helpsteer2 | robust | 91.2 | 84.7 | 3.52 | 1.2 | 33.3 | - |
| summeval_relevance | peak | 88.3 | 81.6 | 3.12 | 0.0 | n/a | [0] |
| summeval_relevance | robust | 88.8 | 84.2 | 3.29 | 0.0 | n/a | [0] |
| summeval_consistency | peak | 92.4 | 90.9 | 4.15 | 0.0 | n/a | [0, 2, 3] |
| summeval_consistency | robust | 91.7 | 90.1 | 4.15 | 0.0 | n/a | [0, 2, 3] |

## Pareto fronts (OOF accuracy vs OOF BA+F1; test columns are post-hoc, never used to choose)

- **massive_en**: `llama+lda|raw` OOF 91.3/91.0 test 89.1/89.1; `fuse0.5+lda|entropy:g2:t0.5` OOF 90.4/91.3 test 84.9/87.4
- **massive_de**: `fuse0.5+bbp|raw` OOF 92.4/92.0 test 92.0/91.8; `fuse0.5+bbp|entropy:g2:t0.5` OOF 92.1/92.1 test 90.6/91.7
- **multinli**: `geo100+lda|static:t2` OOF 88.8/88.8 test 87.3/87.5
- **pubmedqa**: `fuse0.5+bbp|margin:g2:t0.5` OOF 81.3/64.7 test 76.4/61.7; `geo035+lda|margin:g2:t0.5` OOF 77.6/65.1 test 74.0/60.2
- **vitaminc**: `fuse0.75+bbp|raw` OOF 85.9/80.2 test 85.8/77.4; `fuse0.5+bbp|margin:g2:t0.5` OOF 84.8/80.8 test 85.1/78.4
- **boolq**: `geo035+bbp|centroid:g2:t0.5` OOF 91.7/91.3 test 87.7/87.6; `geo035+bbp|static:t0.5` OOF 91.5/91.3 test 89.0/89.2; `geo035+bbp|margin:g2:t1` OOF 91.5/91.4 test 89.3/89.6
- **squad2**: `fuse0.25+bbp|margin:g2:t0.5` OOF 92.1/91.4 test 92.6/92.6
- **paws**: `fuse0.5+bbp|raw` OOF 90.7/90.6 test 92.0/91.9
- **civil_comments**: `qwen+lda|raw` OOF 91.7/64.4 test 87.3/57.8; `fuse0.5+bbp|margin:g2:t0.5` OOF 90.6/68.6 test 90.7/74.5; `fuse0.75+bbp|margin:g1:t0.5` OOF 87.9/69.1 test 89.3/75.1; `geo100+bbp|margin:g1:t0.5` OOF 86.9/69.5 test 90.0/76.8; `fuse0.5+bbp|margin:g2:t1` OOF 86.7/69.8 test 90.3/79.1; `fuse0.75+bbp|entropy:g2:t0.5` OOF 86.4/69.9 test 88.7/76.5
- **aegis_safety**: `concat+bbp|margin:g1:t0.5` OOF 86.9/86.6 test 80.8/80.0
- **helpsteer2**: `geo035+bbp|raw` OOF 44.2/35.0 test 41.4/35.2; `geo035+bbp|entropy:g2:t0.5` OOF 40.4/35.4 test 39.8/37.3; `geo035+bbp|margin:g2:t0.5` OOF 40.2/35.7 test 39.0/36.9; `geo100+bbp|entropy:g2:t0.5` OOF 39.8/35.9 test 37.3/35.7; `geo035+bbp|entropy:g2:t1` OOF 37.5/36.1 test 35.3/35.0; `concat+bbp|centroid:g2:t0.5` OOF 36.9/36.2 test 35.7/35.3
- **summeval_relevance**: `fuse0.25+bbp|raw` OOF 57.9/37.6 test 54.2/39.0; `fuse0.5+bbp|raw` OOF 57.6/40.0 test 55.0/39.3; `llama+bbp|raw` OOF 57.5/40.1 test 53.8/41.4; `geo100+bbp|raw` OOF 55.7/41.3 test 52.9/37.0; `geo035+lda|raw` OOF 54.2/43.8 test 51.7/42.9; `fuse0.25+bbp|entropy:g2:t0.5` OOF 47.6/45.2 test 52.1/44.8
- **summeval_consistency**: `fuse0.75+bbp|raw` OOF 88.1/34.5 test 88.9/34.3; `llama+lda|raw` OOF 87.9/34.8 test 84.0/31.3; `concat+bbp|raw` OOF 87.8/36.4 test 88.2/33.1; `geo035+lda|raw` OOF 87.6/37.5 test 88.2/48.1; `geo100+lda|raw` OOF 87.5/37.6 test 88.2/47.2; `fuse0.5+bbp|margin:g2:t0.5` OOF 83.2/39.6 test 84.0/41.1

## Honesty notes

- CALA sign: the term is subtracted (`f - tau*Phi*log pi`). The brief's `+` would favour the majority class.
- Selection scans a few hundred candidates per task on OOF. Read the OOF-minus-test gap in the JSON as the selection optimism; the fixed controls need no selection.
- Conformal guarantees hold for the 80%-train refit model under exchangeability; the reported point metrics come from the full-train model, which the theorem does not cover.
- Abstention (singleton-only prediction) is reported separately and never feeds the accuracy numbers.
- The earlier 256-d dual ensemble files (`evaluate_dual_70b_72b_*.py`) are untracked files of another task and were left in place; this suite does not import them.
