# Manifold-Pareto dual 70B/72B ensemble: geometric fusion + CALA gate + conformal control

Generated 2026-09-26T01:35:41.917811+00:00 on `luy-open-box` (cores 24, loadavg at start [85.68, 73.25, 38.67]).

Command: `benchmarks/suites/evaluate_manifold_pareto_ensemble.py --qwen-dir /ebs/data/extracted_features/qwen72b/features --llama-dir /ebs/data/extracted_features/llama70b --tasks pubmedqa aegis_safety --out benchmarks/results/unified_manifold_pareto_eval --workers 2`

## Headline (test split, macro over 2 tasks, %)

| Arm | Accuracy | Balanced acc | Macro F1 |
|---|---:|---:|---:|
| Peak SOTA track (selected on OOF accuracy) | 80.00 | 70.81 | 68.62 |
|   matched control: same search, single models only | 79.40 | 70.26 | 68.17 |
| Certified Robust track (selected on OOF BA+F1) | 78.60 | 70.32 | 69.74 |
|   matched control: same search, single models only | 77.80 | 74.05 | 73.79 |
| fixed control `concat+bbp` (no search) | 79.60 | 70.69 | 69.03 |
| fixed control `concat+lda` (no search) | 79.00 | 71.65 | 71.38 |
| fixed control `geo035+bbp` (no search) | 78.20 | 69.26 | 67.60 |
| fixed control `geo035+lda` (no search) | 78.20 | 69.86 | 69.30 |
| fixed control `geo100+bbp` (no search) | 78.40 | 69.06 | 66.85 |
| fixed control `geo100+lda` (no search) | 79.00 | 70.69 | 69.98 |
| fixed control `llama+bbp` (no search) | 79.60 | 70.71 | 69.03 |
| fixed control `llama+lda` (no search) | 76.00 | 68.64 | 68.18 |
| fixed control `qwen+bbp` (no search) | 79.20 | 69.92 | 67.89 |
| fixed control `qwen+lda` (no search) | 74.60 | 67.01 | 66.62 |

External references (different protocol, not matched): published Qwen-72B 81.07, Llama-70B 81.09, earlier 256-d dual ensemble 77.21.

## Target check

- Peak macro accuracy >= 82.5: **False** (80.00)
- Peak beats published Qwen-72B scorecard: **False**
- Robust macro balanced accuracy >= 76: **False** (70.32)
- Robust macro F1 >= 76: **False** (69.74)
- Robust tasks with a collapse or a zero-recall class on test: none
- Robust tasks that no candidate could admit through the OOF gate: none
- Peak tasks that no candidate could admit through the OOF gate: none

## Paired bootstrap on test accuracy (macro, pp, rows resampled inside tasks)

| Comparison | Delta pp | 95% CI |
|---|---:|---|
| peak_minus_peak_single | +0.60 | [-0.20, +1.60] |
| peak_minus_ctl_qwen_bbp | +0.80 | [-1.00, +2.60] |
| peak_minus_ctl_llama_bbp | +0.40 | [-1.00, +1.80] |
| robust_minus_robust_single | +0.80 | [-1.60, +3.20] |
| peak_minus_ctl_concat_bbp | +0.40 | [-1.00, +2.00] |
| ctl_geo100_bbp_minus_ctl_concat_bbp | -1.20 | [-2.80, +0.20] |

## CALA ablation (fixed base, no search; macro test %, delta vs the raw head in pp)

| Base head | Prior shift | Accuracy | d Acc | Balanced acc | d BA | Macro F1 | d F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| geo100+bbp | raw | 78.40 | +0.00 | 69.06 | +0.00 | 66.85 | +0.00 |
| geo100+bbp | static:t1 | 69.40 | -9.00 | 69.60 | +0.54 | 68.30 | +1.45 |
| geo100+bbp | entropy:g2:t1 | 76.20 | -2.20 | 71.35 | +2.29 | 71.44 | +4.59 |
| geo100+bbp | margin:g2:t1 | 77.60 | -0.80 | 71.80 | +2.74 | 72.06 | +5.21 |
| geo100+bbp | centroid:g2:t1 | 71.40 | -7.00 | 71.02 | +1.96 | 69.78 | +2.93 |
| geo100+lda | raw | 79.00 | +0.00 | 70.69 | +0.00 | 69.98 | +0.00 |
| geo100+lda | static:t1 | 67.80 | -11.20 | 69.97 | -0.72 | 67.68 | -2.30 |
| geo100+lda | entropy:g2:t1 | 74.80 | -4.20 | 71.73 | +1.04 | 71.48 | +1.50 |
| geo100+lda | margin:g2:t1 | 76.60 | -2.40 | 72.24 | +1.55 | 72.13 | +2.15 |
| geo100+lda | centroid:g2:t1 | 69.40 | -9.60 | 70.95 | +0.26 | 68.88 | -1.10 |

## Per task

| Task | n_train | Peak config | Peak OOF acc | Peak test acc | Single-model control acc | Robust config | Robust OOF obj | Robust test BA | Robust test F1 | min recall | Robust admitted |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| pubmedqa | 750 | `fuse0.75+bbp|raw` | 81.47 | 78.40 | 77.20 | `geo035+lda|raw` | 62.34 | 59.95 | 58.47 | 8.1 | True |
| aegis_safety | 1000 | `llama+bbp|raw` | 86.60 | 81.60 | 81.60 | `llama+bbp|raw` | 86.16 | 80.70 | 81.01 | 74.1 | True |

## Per-task fixed controls, test accuracy (no search)

| Task | concat+bbp | concat+lda | geo035+bbp | geo035+lda | geo100+bbp | geo100+lda | llama+bbp | llama+lda | qwen+bbp | qwen+lda |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pubmedqa | 77.60 | 75.20 | 77.60 | 75.60 | 78.00 | 76.80 | 77.60 | 71.60 | 77.20 | 72.80 |
| aegis_safety | 81.60 | 82.80 | 78.80 | 80.80 | 78.80 | 81.20 | 81.60 | 80.40 | 81.20 | 76.40 |

## Conformal (alpha 0.1; model refit on 80% of train, calibrated on 20%)

| Task | Track | Marginal coverage | Min class coverage | Mean set size | Singleton rate | Singleton acc | Trivial classes |
|---|---|---:|---:|---:|---:|---:|---|
| pubmedqa | peak | 92.4 | 88.8 | 1.88 | 13.2 | 93.9 | - |
| pubmedqa | robust | 90.0 | 83.8 | 1.82 | 22.4 | 82.1 | - |
| aegis_safety | peak | 85.2 | 83.3 | 1.10 | 90.0 | 83.6 | - |
| aegis_safety | robust | 85.2 | 83.3 | 1.10 | 90.0 | 83.6 | - |

## Pareto fronts (OOF accuracy vs OOF BA+F1; test columns are post-hoc, never used to choose)

- **pubmedqa**: `fuse0.5+bbp|static:t0.5` OOF 81.7/60.5 test 76.8/58.3; `concat+bbp|margin:g2:t0.5` OOF 81.3/61.8 test 78.4/63.3; `concat+bbp|margin:g1:t0.5` OOF 81.2/62.2 test 78.8/64.5; `geo100+lda|raw` OOF 80.9/63.1 test 76.8/60.2; `fuse0.25+bbp|margin:g1:t1` OOF 79.9/63.5 test 77.6/64.0; `geo035+lda|margin:g2:t0.5` OOF 77.6/65.1 test 74.0/60.2
- **aegis_safety**: `concat+bbp|margin:g1:t0.5` OOF 86.9/86.6 test 80.8/80.0

## Honesty notes

- CALA sign: the term is subtracted (`f - tau*Phi*log pi`). The brief's `+` would favour the majority class.
- Selection scans a few hundred candidates per task on OOF. Read the OOF-minus-test gap in the JSON as the selection optimism; the fixed controls need no selection.
- Conformal guarantees hold for the 80%-train refit model under exchangeability; the reported point metrics come from the full-train model, which the theorem does not cover.
- Abstention (singleton-only prediction) is reported separately and never feeds the accuracy numbers.
- The earlier 256-d dual ensemble files (`evaluate_dual_70b_72b_*.py`) are untracked files of another task and were left in place; this suite does not import them.
