# Qwen-72B vs other models: multinli + pubmedqa (plus massive_en, boolq, paws)

Date: 2026-09-25. No other model was run. Every non-Qwen-72B number below is copied from an existing report (source column).

## 1. What was run (Qwen-72B only)

```
python3 benchmarks/suites/run_qwen72b_downstream_eval.py --task multinli --probe-c 0.01 --out benchmarks/results/qwen72b_linear_probe_multinli_report.json   # exit 0
python3 benchmarks/suites/run_qwen72b_downstream_eval.py --task pubmedqa --probe-c 0.01 --out benchmarks/results/qwen72b_linear_probe_pubmedqa_report.json   # exit 0
```

Input: stored features `data/extracted_features/qwen72b/features/<task>.npz` (Qwen2.5-72B-Instruct GGUF Q4_K_M, last-token pooling, 8192-D). Probe: sklearn `LogisticRegression(C=0.01, lbfgs, max_iter=2000)`, raw features, one fixed C. sklearn 1.9.1.

| Task | K | n_train | n_test | Train acc | **Test acc** | Wilson 95% | Majority baseline | Gain over majority |
|---|---|---|---|---|---|---|---|---|
| multinli | 3 | 1000 | 299 | 99.60% | **90.30%** (270/299) | [86.42, 93.16] | 36.12% | +54.2 pp |
| pubmedqa | 3 | 750 | 250 | 100.00% | **74.80%** (187/250) | [69.07, 79.78] | 53.20% | +21.6 pp |

Confusion matrices (rows = true, columns = predicted; class order = candidate order in the data):
- multinli: `[[99,9,0],[5,81,5],[0,10,90]]`. Recall 91.7 / 89.0 / 90.0%. Balanced and healthy.
- pubmedqa (yes/no/maybe): `[[119,11,3],[13,64,3],[21,12,4]]`. Recall 89.5 / 80.0 / **10.8%**. Balanced accuracy 60.1%.

The Qwen-72B numbers for the three earlier tasks come from the existing reports `qwen72b_linear_probe_{massive_en,boolq,paws}_report.json` (same script, same C).

## 2. Cross-model table (test accuracy %, same test rows: n = 350 / 299 / 250 / 300 / 250)

| Model / method | massive_en | multinli | pubmedqa | boolq | paws | Source |
|---|---|---|---|---|---|---|
| **Qwen-72B, LR probe, raw, C=0.01 (this run)** | 89.14 | **90.30** | 74.80 | 88.00 | 91.60 | `qwen72b_linear_probe_*_report.json` |
| Llama-3.1-70B, Spec 21 chosen expert | 89.14 | 86.29 | 76.40 | 88.33 | **92.80** | `spec21_llama70b_scorecard.md` per-task table |
| Llama-3.1-70B, its own linear probe (tau=0) | 89.71 | 82.94 | 76.00 | 89.00 | 91.20 | same file, "Logit adjustment" table |
| Qwen3.5-9B, Spec 21 chosen expert | 85.71 | 86.62 | 72.40 | 83.67 | 87.60 | `spec21_grand_scorecard_report.md` per-task table |
| Qwen3.5-9B, its own linear probe (tau=0) | 82.57 | 83.28 | 67.20 | 82.00 | 89.20 | same file, "Logit adjustment" table |
| Nimble (official, fine-tuned) | 86.9 | 85.3 | 75.6 | 86.0 | 82.8 | `nimble` field in the Spec 21 JSON, per task |
| Jevons (official, fine-tuned) | 87.4 | 82.9 | 77.2 | 89.7 | 89.2 | `jev` field in the Spec 21 JSON, per task |
| Qwen3.5-9B bare zero-shot (ar_loglik) | 50.00 | 23.33 | 73.33 | 80.00 | 54.00 | `canonical_qwen9b_baseline.md`; n = 30/30/30/30/400 |
| Majority-class baseline (n_test) | 16.29 | 36.12 | 53.20 | 58.00 | 51.60 | this run's reports |

Nimble and Jevons per-task figures were read from `spec21_llama70b_scorecard.json` (`tasks.<task>.nimble` / `.jev`). The `.md` files only give their 11-task macro (Nimble-9B 80.40, Jevons 83.54, `canonical_qwen9b_baseline.md`).

## 3. Reading it honestly

**Where Qwen-72B leads.**
- multinli: 90.30 is the top score in the table. It is +4.0 pp over Llama-70B's chosen expert and +7.4 pp over Llama-70B's linear probe. It is also +3.3 pp over the best 9B figure.
- Gap to the Llama chosen expert: the Wilson intervals overlap ([86.42, 93.16] vs [81.93, 89.73]). We have no paired predictions, so no McNemar test. Treat +4.0 pp as suggestive, not proven.
- The gap to Llama's own plain linear probe is larger (+7.4 pp) but has the same caveat.

**Where it does not lead.**
- pubmedqa: 74.80 is below Llama-70B (76.40 / 76.00), Nimble (75.6) and Jevons (77.2). The gap to Llama is 1.6 pp, well inside the interval [69.07, 79.78]. This is a tie, not a win or loss.
- paws: Llama-70B's chosen expert (92.80) is 1.2 pp ahead. Also a tie inside noise (n = 250).
- boolq and massive_en: Qwen-72B is level with Llama-70B (88.00 vs 88.33 and 89.14 vs 89.14 chosen expert; 89.00 and 89.71 for the plain probe).

**pubmedqa: the 74.8% hides a failed class.** The probe recalls only 4 of 37 "maybe" rows (10.8%). Balanced accuracy is 60.1%, and 74.8% is only 21.6 pp over majority. Llama-70B is in the same place (balanced accuracy 59.96%, macro F1 57.50%), and the 9B model is worse (55.81%). All three models solve "yes vs no" and fail on "maybe". The task is limited by the uncertain class, not by model size. Extra parameters bought nothing measurable here.

**multinli: a real, balanced NLI result.** Recall is 90 ± 1.5% on all three classes. Train accuracy 99.6% on n = 1000 shows a large gap to test, as expected for 8192-D features and 1000 rows, but the test set is held out.

**Geometry vs Llama-70B: what can and cannot be said.**
- Measured: no Qwen-72B vs Llama-70B alignment (CKA / Procrustes) exists in the repo. The only 72B geometry report is `manifold_alignment_qwen72b_vs_gte7b_massive_en.json` (linear CKA 0.668 train / 0.678 test, Procrustes residual 0.580 / 0.520, against GTE-7B). Running a 72B vs Llama-70B alignment would need Llama features, which were not touched here (they live on the AI server).
- Not measured: any claim such as "Qwen's manifold is more separable than Llama's" is not supported by a metric. Only accuracy is. The accuracy pattern is: Qwen-72B is stronger on NLI (multinli), Llama-70B is equal or slightly ahead on pair/paraphrase and medical, and both saturate near 88-92 on the easy tasks.
- Possible causes (hypotheses, not tested): Qwen2.5-72B-Instruct has heavier instruction/NLI-style training data; Llama-70B's Spec 21 head selection (BBP probe, CV over 15 candidates) is stronger than a fixed-C probe.

**Versus the 9B model.** Qwen-72B beats the Qwen3.5-9B chosen expert (`spec21_grand_scorecard_report.md`) by +3.43 (massive_en), +3.68 (multinli), +2.40 (pubmedqa), +4.33 (boolq) and +4.00 (paws) pp. Each single gap is inside one Wilson interval, but all five point the same way. Against the fine-tuned official models: ahead of Nimble on four of five tasks (behind by 0.8 pp on pubmedqa); ahead of Jevons on three (behind by 2.4 pp on pubmedqa and 1.7 pp on boolq).

## 4. Caveats that weaken the comparison

1. **Protocols differ.** Qwen-72B here = raw features, one fixed C = 0.01, no selection. The Spec 21 rows (Llama, 9B) standardize features, run inner-CV over C (`LogisticRegressionCV`, `benchmarks/suites/sota_ensemble_experts.py:257-263`) and, for the "chosen expert" rows, pick the best of 15 heads by CV. So the 72B number is a weaker recipe than the Llama number, and the plain-probe rows are still not identical recipes. Direction of bias: probably against Qwen-72B, but not measured.
2. **Quantization.** Both 70B-class feature sets came from Q4_K_M GGUF via llama-server, so they are matched on that axis. The 9B features are not matched on that axis.
3. **The 9B zero-shot row is a different task type** (candidate log-likelihood, n = 30 on four of five tasks). It is a floor, not a competitor. Its multinli 23.33% is below chance (33%) on 30 rows.
4. **No paired tests.** The Spec 21 JSONs carry `correct_mask` for the Llama and 9B chosen experts, but the Qwen-72B report stores only a confusion matrix, not per-row predictions. So no McNemar test was run.
5. **Single seed, single split.** n_test = 250-350; one test-set flip is 0.3-0.4 pp.
6. **Label reconstruction.** Qwen-72B test labels are rebuilt as `candidates.index(ground_truth)` because the npz has no test labels. For `massive_en` the script gives 89.14 where an earlier unsourced report said 89.71 (2/350 rows; see the docstring at `run_qwen72b_downstream_eval.py:8-21`). The 89.14 in this table is the script's number.

## 5. Status

Verified (command, exit code 0, report file on disk):
- multinli: 90.30%, train 99.60%, majority 36.12%.
- pubmedqa: 74.80%, train 100%, majority 53.20%.

From existing reports, copied, not re-run: all other models' figures in section 2.

Not verified: any geometric (CKA / Procrustes) claim on Qwen-72B vs Llama-70B; any statistical significance of cross-model gaps; the causes listed as hypotheses.

Not done: a Qwen-72B run with the Spec 21 recipe (standardize + CV over C + head selection), which is the fair comparison to Llama's chosen expert. It needs a script change and was outside the stated command.
