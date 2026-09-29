# Dual 70B/72B ensemble: 13-task evaluation

Cross-fitted ridge probes on aligned frozen features; the Qwen fusion weight is selected on training OOF accuracy. Test labels are read after selection.

The published Qwen 81.07% / Llama 81.09% scorecards use different selected expert families. Their numbers are external references, not matched probe controls.

The equal fusion does not beat either published macro reference. Some tasks have high accuracy but much lower balanced accuracy and macro F1, indicating class imbalance or class collapse.

## Macro metrics (%)

| Method | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| qwen_probe | 79.91 | 70.21 | 68.89 |
| llama_probe | 79.74 | 68.85 | 68.87 |
| equal_fusion | 80.86 | 70.91 | 70.29 |
| oof_weight_fusion | 80.82 | 70.89 | 70.32 |

## Per-task accuracy (%)

| Task | Qwen probe | Llama probe | Equal fusion | OOF-weight fusion | Matched winner | Qwen weight | Published Qwen | Published Llama | Equal minus best published |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| massive_en | 88.86 | 89.71 | 89.14 | 89.14 | llama_probe | 0.50 | 89.71 | 89.14 | -0.57 |
| massive_de | 89.14 | 88.29 | 89.43 | 89.43 | equal_fusion, oof_weight_fusion | 0.50 | 91.14 | 91.43 | -2.00 |
| multinli | 87.63 | 83.28 | 86.96 | 88.63 | oof_weight_fusion | 0.75 | 88.29 | 86.29 | -1.34 |
| pubmedqa | 73.60 | 73.60 | 76.00 | 75.20 | equal_fusion | 0.75 | 73.20 | 76.40 | -0.40 |
| vitaminc | 83.31 | 81.47 | 84.81 | 83.81 | equal_fusion | 0.75 | 84.64 | 83.47 | +0.17 |
| boolq | 87.33 | 89.33 | 87.67 | 87.67 | llama_probe | 0.50 | 88.00 | 88.33 | -0.67 |
| squad2 | 88.96 | 89.97 | 91.30 | 91.30 | equal_fusion, oof_weight_fusion | 0.50 | 89.97 | 91.64 | -0.33 |
| paws | 91.20 | 91.20 | 94.00 | 94.00 | equal_fusion, oof_weight_fusion | 0.50 | 91.60 | 92.80 | +1.20 |
| civil_comments | 90.00 | 90.33 | 89.67 | 89.67 | llama_probe | 0.50 | 90.00 | 90.33 | -0.67 |
| aegis_safety | 80.00 | 78.80 | 79.20 | 79.20 | qwen_probe | 0.50 | 82.00 | 81.60 | -2.80 |
| helpsteer2 | 41.37 | 43.37 | 42.57 | 42.57 | llama_probe | 0.50 | 42.57 | 42.17 | +0.00 |
| summeval_relevance | 47.92 | 50.42 | 50.83 | 50.42 | equal_fusion | 0.25 | 54.58 | 53.75 | -3.75 |
| summeval_consistency | 89.58 | 86.81 | 89.58 | 89.58 | qwen_probe, equal_fusion, oof_weight_fusion | 0.50 | 88.19 | 86.81 | +1.39 |

All full metrics, OOF weights, feature hashes, and test-file hashes are in the JSON report.
