# 01.PNG grand challenge: 13 public benchmarks, 3,880 test records

Generated 2026-09-23T16:17:41Z by `python3 benchmarks/suites/benchmark_01png_grand_challenge.py all`. Raw data: `benchmarks/results/01png_grand_challenge_report.json`.

## Headline

- **baseline**: macro avg 63.82% over 13 tasks (micro 63.89%); Nimble 74.8%, Jev 76.0% (delta vs Jev -12.18). Beats the 01.PNG best on 1 task(s): civil_comments; of those, also above the majority-class floor: none.
- **deep_wide**: macro avg 58.89% over 13 tasks (micro 58.74%); Nimble 74.8%, Jev 76.0% (delta vs Jev -17.11). Beats the 01.PNG best on 1 task(s): civil_comments; of those, also above the majority-class floor: none.
- Majority-class (train prior) macro: 49.77%. Laya (A100 report, not re-run) macro: 55.48%.

## Per task (accuracy %, 95% Wilson CI)

| Task | n | Baseline | Deep-Wide | McNemar p | Majority | Nimble | Jev | Best | Delta DW vs best | Laya |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MASSIVE en-US | 350 | 79.43 [74.89, 83.33] | 75.14 [70.36, 79.38] | 0.0581 | 16.29 | 86.9 | 87.4 | 87.4 | -12.3 | 47.43 |
| MASSIVE de-DE | 350 | 68.29 [63.23, 72.94] | 72.0 [67.08, 76.45] | 0.171 | 16.29 | 83.4 | 86.9 | 86.9 | -14.9 | 33.43 |
| MultiNLI (pair) | 299 | 67.22 [61.71, 72.3] | 46.49 [40.92, 52.15] | 8.11e-08 | 33.44 | 85.3 | 82.9 | 85.3 | -38.8 | 88.96 |
| PubMedQA | 250 | 59.6 [53.42, 65.49] | 59.6 [53.42, 65.49] | 1 | 53.2 | 75.6 | 77.2 | 77.2 | -17.6 | 50.4 |
| VitaminC (pair) | 599 | 58.93 [54.95, 62.8] | 49.92 [45.93, 53.91] | 0.000732 | 50.25 | 76.6 | 80.1 | 80.1 | -30.2 | 73.12 |
| BoolQ | 300 | 56.67 [51.01, 62.15] | 57.33 [51.68, 62.8] | 0.892 | 58.0 | 86.0 | 89.7 | 89.7 | -32.4 | 82.0 |
| SQuAD 2.0 | 299 | 58.19 [52.53, 63.65] | 58.19 [52.53, 63.65] | 1 | 49.83 | 80.6 | 82.9 | 82.9 | -24.7 | 66.22 |
| PAWS (pair) | 250 | 78.4 [72.89, 83.05] | 50.4 [44.24, 56.54] | 1.85e-10 | 51.6 | 82.8 | 89.2 | 89.2 | -38.8 | 88.4 |
| Civil Comments | 300 | 89.0 [84.95, 92.06] | 86.67 [82.35, 90.05] | 0.265 | 89.33 | 70.3 | 81.0 | 81.0 | +5.7 | 95.33 |
| Aegis 2.0 | 250 | 74.0 [68.23, 79.05] | 70.4 [64.47, 75.72] | 0.188 | 56.8 | 81.2 | 80.4 | 81.2 | -10.8 | 40.8 |
| HelpSteer2 | 249 | 26.51 [21.41, 32.32] | 25.7 [20.67, 31.47] | 0.88 | 42.17 | 39.0 | 34.1 | 39.0 | -13.3 | 35.34 |
| SummEval relevance | 240 | 39.17 [33.21, 45.47] | 42.92 [36.81, 49.24] | 0.272 | 45.83 | 49.2 | 35.0 | 49.2 | -6.3 | 12.92 |
| SummEval consistency | 144 | 74.31 [66.6, 80.75] | 70.83 [62.95, 77.64] | 0.332 | 84.03 | 75.7 | 81.2 | 81.2 | -10.4 | 6.94 |

## CPU latency (ms per record, median)

| Task | encoder batch-1, whole context | encoder batch-1, pair fields | baseline head | deep-wide head |
|---|---:|---:|---:|---:|
| MASSIVE en-US | 145 | - | 1.06 | 7.86 |
| MASSIVE de-DE | 151 | - | 0.91 | 8.45 |
| MultiNLI | 176 | 247 | 0.74 | 3.88 |
| PubMedQA | 434 | - | 0.57 | 3.47 |
| VitaminC | 199 | 250 | 0.56 | 3.79 |
| BoolQ | 237 | - | 0.55 | 3.42 |
| SQuAD 2.0 | 404 | - | 0.56 | 3.20 |
| PAWS | 290 | 340 | 0.55 | 3.68 |
| Civil Comments | 293 | - | 0.69 | 3.20 |
| Aegis 2.0 | 244 | - | 0.55 | 3.32 |
| HelpSteer2 | 430 | - | 0.67 | 5.26 |
| SummEval relevance | 338 | - | 0.67 | 5.37 |
| SummEval consistency | 326 | - | 0.96 | 5.55 |

Host loadavg at eval start/end: [0.0, 0.0, 0.0] / [0.06, 0.02, 0.01] on 24 CPUs (shared box).

## Protocol

```json
{
 "encoder": "D:\\models\\Qwen2.5-0.5B",
 "encoder_precision": "fp32",
 "device": "cpu",
 "pooling": [
  "mean@layer12",
  "last@layer18"
 ],
 "max_tokens": 256,
 "truncation": "first 64 + last 192 tokens",
 "train_records_per_task_max": 1000,
 "early_stop_fraction_of_train": 0.15,
 "optimizer": {
  "name": "AdamW",
  "lr": 0.001,
  "weight_decay": 0.01,
  "batch": 32,
  "max_epochs": 80,
  "patience": 10,
  "seed": 0
 },
 "configs": {
  "baseline": "ParallelRNNSetAdapter d=256 rank=16 T=6, 1 set layer x 4 heads; query = pooled whole context",
  "deep_wide": "DeepWideRNNSetAdapter d=512 rank=16 T=6, 3 residual RNN layers rho_max 0.95/0.85/0.75, 2 set layers x 8 heads SwiGLU; query = cross-difference of separately encoded fields on multinli/paws/vitaminc, pooled whole context elsewhere"
 },
 "scoring": "argmax over the task's candidate list vs ground_truth, exact match",
 "comparability_caveat": "Nimble and Jev in 01.PNG are zero-shot judges; both configs here are per-task heads supervised on public train splits (leakage-gated). Laya numbers are copied from its A100 report, not re-run."
}
```

## Verdict

Deep-Wide macro average is **58.89%**, below Jev (76.0%) by -17.11 points and -15.91 vs Nimble (74.8%). Baseline macro is 63.82%. Baseline vs Deep-Wide differs at p < 0.05 (exact McNemar) on: multinli, paws, vitaminc.

## Status (three classes)

### Implemented, with evidence

- Test set = 01.PNG record counts on every task: True (sha256 per file in the JSON `test_file`).
- Leakage gate id/family/normalized-text overlap all zero: True (JSON `leakage_gate`).
- NumPy CPU runtime reproduces the trained torch head: min argmax agreement 1.0000 (JSON `torch_numpy_argmax_agreement`).
- Per-task accuracy, Wilson 95% CI, deltas vs Nimble/Jev/best, majority-class floor, exact McNemar baseline vs Deep-Wide: all computed by `stage_eval` from runtime predictions.

### Not verified

- Latency: shared box (see loadavg); single run; numbers are indicative only.
- One training seed per head; run-to-run variance is not measured.
- Pooling (mean@12 + last@18) was picked on a TRAIN-internal pilot of 3 tasks, not tuned per task.
- No per-item comparison with Nimble/Jev: 01.PNG publishes aggregates only, so no McNemar vs them.

### Not done

- `causal_moe_engine.py` consensus arbitration and `causal_mcts_rnn.py` search are NOT part of this run: the two compared configs are single heads.
- Laya was not re-run (owner instruction); its column is copied from `laya_full_13_report.json`.
- Training capped at 1000 records per task and the encoder is a 0.5B model on CPU; larger encoders / more data were not tried.
- Cross-difference path is used only on MultiNLI, PAWS, VitaminC (as specified), not on other two-field tasks (BoolQ, SQuAD 2.0, HelpSteer2, SummEval).

### Comparability

- Nimble and Jev are zero-shot judges; these heads are supervised on each task's public train split. A higher number here is not the same claim as theirs.
- A task where accuracy beats 01.PNG but not the majority-class floor (e.g. a skewed label prior) is not evidence of skill; see `tasks_beating_best_01png_AND_majority`.
