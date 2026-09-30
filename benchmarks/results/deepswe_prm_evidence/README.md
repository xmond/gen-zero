# DeepSWE offline PRM reproduction evidence

Implemented: full read of 44,409 rows, 113 tasks, 449 trajectories; validated required columns, non-empty metadata, binary and within-trajectory-consistent reward, unique step_idx, finite and correctly-shaped embeddings, finite weights and projections; strict loading of the scoring head. The scoring head's SHA256 matches the local release README. supplied best_head.pt is re-run fresh at every step; no existing scored.pkl was read and no parquet prm_score stood in for inference.

| Metric | All 113 tasks | Held-out 38 tasks |
|---|---:|---:|
| Offline random single-trajectory Pass@1 (task macro-average) | 73.00885% | 73.68421% |
| Strict BoN=2 expected solve rate | 79.05605% | 75.43860% |
| BoN=min(4, available) | 85.84071% (97/113) | 81.57895% (31/38) |
| Strict BoN=4, full-task solve rate | not computable (3 tasks short) | not computable (1 task short) |
| Strict BoN=8 | not computable | not computable |
| Trajectory ROC AUC | 0.794104 | 0.519595 |
| Trajectory Spearman rho | 0.453211 | 0.029954 |
| Step-level ROC AUC | 0.683208 | 0.491701 |
| Step-level Spearman rho | 0.276906 | -0.012513 |

The N=8 capped figure equals N=4 and must not be treated as real BoN=8. 110 tasks have 4 trajectories each, 3 tasks have 3 each. BoN uses the exact expectation over uniform sampling without replacement, with ties for top score handled uniformly. Trajectory score is the mean cosine score over the last 12 available steps after sorting by step_idx. Step labels repeat the trajectory's final-outcome label, so step-level metrics are biased toward long trajectories; the usual correlation p-values do not account for within-task correlation, and 0.0 is numeric underflow, not a true probability of zero.

The local release notes at `/tmp/clmrepro/heads/README.md:23` report a 31/38 reproduction success, -0.02105 percentage points off 81.6%, which is rounding difference. That release protocol also keeps tasks with fewer than 4 trajectories. The full-set 85.84% is 4.24071 percentage points above 81.6%, but it includes 75 training-partition tasks, so it is not a like-for-like generalization gain. Held-out trajectory AUC is close to 0.5 and Spearman close to 0, while the training-partition AUC is 0.934564: a mixed-set metric must not be used to mask the held-out set's weak correlation.

Not verified: independent upstream checksums/authenticity of the parquet, re-execution of reward in its own environment, whether the raw trajectories are complete and untruncated. The existing local SHA256 only guarantees this run's input is locatable. The result is offline best-selection of the CLM head over already-stored Opus 5 trajectories, not newly generated B2 trajectories or a full environment benchmark.

Not done (input does not support it): true greedy-decoding accuracy, strict N=4 across all 113 tasks, strict N=8. These require explicit greedy trajectories and at least 8 real candidates per task; no trajectories were fabricated, no repeated sampling was passed off as new candidates, and there was no silent degradation.

## Commands and raw exit codes

Working directory: `/ebs/pj/gen-zero-worktree/eval-b2-deepswe`

```bash
python benchmarks/eval_deepswe_prm.py > benchmarks/results/deepswe_prm_evidence/run.log 2>&1
rc=$?
printf '%s\n' "$rc" > benchmarks/results/deepswe_prm_evidence/exit_code.txt
exit "$rc"
```

Raw exit code: `0`, recorded in `exit_code.txt:1`. Full actual inference log: `run.log`; completion of 44,409 rows at `run.log:46`; explicit unavailable metrics at `run.log:47`; output at `run.log:425`.

```bash
python benchmarks/results/deepswe_prm_evidence/verify.py > benchmarks/results/deepswe_prm_evidence/verification.log 2>&1
rc=$?
printf '%s\n' "$rc" > benchmarks/results/deepswe_prm_evidence/verification_exit_code.txt
exit "$rc"
```

Raw exit code: `0`. `verification.log:1`: six sklearn ROC AUC comparisons and 1,024 exhaustive subset/tie cases passed. Every real candidate group's subset calculation also checked by independent enumeration during evaluation. `git -c core.fsmonitor=false diff --check` exited 0.

Implementation: `benchmarks/eval_deepswe_prm.py:60` validation, `:83` fresh inference, `:111` task aggregation and exact selection. Results: `benchmarks/results/deepswe_prm_eval_results.json`. Recomputed raw scores: `step_scores.parquet`; trajectory audit: `trajectory_scores.csv`. Input, reference code, evaluator and score artifact hashes are in results JSON. Reproduction requires the explicitly recorded local reference repository and Python dependencies.

No old module was replaced; no unrelated tracked file was modified. No subagents/reviewers were launched.
