# Advanced dual 70B/72B expert evaluation

Five stratified training folds select the branch, expert and fusion weight. Test labels are loaded only after selection. Scores are percentages.

Published 81.07% / 81.09% single-model references used a different selection protocol, so the matched single-model columns are the direct controls.

| Method | Macro accuracy | Macro balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| qwen_best | 76.14 | 72.98 | 68.73 |
| llama_best | 76.23 | 73.20 | 69.78 |
| selected | 77.21 | 74.30 | 70.65 |

| Task | Qwen best head | Qwen acc | Llama best head | Llama acc | Selected expert | Selected acc | Selected balanced acc | Selected F1 |
|---|---|---:|---|---:|---|---:|---:|---:|
| massive_en | weighted_ridge+la0.5 | 86.57 | weighted_ridge+la0 | 88.29 | lw_lda+la0 / weighted_ridge+la1 @ 0.75 | 88.86 | 91.34 | 89.47 |
| massive_de | weighted_ridge+la0 | 86.00 | weighted_ridge+la0 | 86.86 | weighted_ridge+la0 / weighted_logistic+la0.5 @ 0.50 | 86.57 | 90.51 | 86.65 |
| multinli | ridge+la0 | 87.63 | bbp+la0 | 82.27 | ridge+la0 / ridge+la0 @ 0.75 | 88.63 | 88.52 | 88.50 |
| pubmedqa | ridge+la0.5 | 73.60 | lw_lda+la0.5 | 65.20 | ridge+la1 / lw_lda+la0 @ 0.50 | 73.60 | 62.59 | 62.37 |
| vitaminc | weighted_ridge+la0 | 79.80 | bbp+la0.5 | 77.30 | weighted_ridge+la0 / bbp+la0 @ 0.50 | 84.14 | 76.76 | 76.96 |
| boolq | ridge+la0 | 87.33 | ridge+la0 | 89.33 | ridge+la0 / logistic+la0 @ 0.50 | 87.67 | 87.62 | 87.41 |
| squad2 | ridge+la0 | 88.96 | ridge+la0 | 89.97 | lw_lda+la0 / bbp+la0 @ 0.50 | 89.30 | 89.32 | 89.25 |
| paws | lw_lda+la0.5 | 90.80 | ridge+la0 | 91.20 | lw_lda+la0 / ridge+la0 @ 0.50 | 92.80 | 92.72 | 92.78 |
| civil_comments | logistic+la0.5 | 72.00 | ridge+la0.5 | 86.00 | ridge+la0.5 / bbp+la1 @ 0.75 | 82.67 | 73.79 | 66.62 |
| aegis_safety | weighted_ridge+la1 | 80.40 | ridge+la0.5 | 79.60 | bbp+la0 / weighted_ridge+la0 @ 0.50 | 80.40 | 79.09 | 79.55 |
| helpsteer2 | weighted_ridge+la0 | 31.33 | weighted_ridge+la0 | 34.14 | weighted_ridge+la0 / lw_lda+la0.5 @ 0.50 | 32.13 | 37.22 | 32.01 |
| summeval_relevance | weighted_ridge+la0 | 42.08 | weighted_ridge+la0 | 43.75 | lw_lda+la0 / weighted_ridge+la1 @ 0.75 | 39.17 | 44.25 | 31.63 |
| summeval_consistency | ridge+la0.5 | 83.33 | ridge+la0.5 | 77.08 | ridge+la1 / bbp+la0 @ 0.50 | 77.78 | 52.19 | 35.30 |

Selected macro accuracy 77.21% versus published Qwen 81.07% and Llama 81.09%: the published threshold was not reached.

Macro lift versus the stronger matched single model (percentage points): accuracy +0.98, balanced_accuracy +1.10, macro_f1 +0.87

## Selected head type distribution

- lw_lda: 6
- weighted_ridge: 6
- weighted_logistic: 1
- ridge: 7
- bbp: 5
- logistic: 1

## Selected expert combinations

- lw_lda+la0 / weighted_ridge+la1: 2
- weighted_ridge+la0 / weighted_logistic+la0.5: 1
- ridge+la0 / ridge+la0: 1
- ridge+la1 / lw_lda+la0: 1
- weighted_ridge+la0 / bbp+la0: 1
- ridge+la0 / logistic+la0: 1
- lw_lda+la0 / bbp+la0: 1
- lw_lda+la0 / ridge+la0: 1
- ridge+la0.5 / bbp+la1: 1
- bbp+la0 / weighted_ridge+la0: 1
- weighted_ridge+la0 / lw_lda+la0.5: 1
- ridge+la1 / bbp+la0: 1

Full hashes, OOF selections, metrics and fit failures are in the JSON report.
