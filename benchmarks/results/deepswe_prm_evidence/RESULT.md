This run's goal was not met: the new head scored 29/38 on the single final held-out evaluation, below the baseline's
31/38, and further below the required 32/38. This experiment must not be described as beating or dominating CLM-8B.

| Held-out metric | Actual baseline best_head.pt | Newly trained head |
|---|---:|---:|
| BoN=1 | 73.6842% | 73.6842% |
| BoN=2 | 75.4386% | 72.8070% |
| BoN=4 | 81.5789% (31/38) | 76.3158% (29/38) |
| Trajectory-level AUC | 0.519595 | 0.413964 |
| Trajectory-level Spearman | 0.029954 | -0.131521 |

**Implemented**: feature extraction and ranking-head training on the training partition (75 tasks, 30,372 steps,
298 trajectories); task-grouped cross-validation for model selection; final weights frozen; independent-process
evaluation on 38 tasks / 151 trajectories; a full report of candidate scores, misranked tasks, and
weight/data/source-file SHA-256 hashes.
6 focused tests passed, and independent verification against the original CLM `best_of_n` passed.

**Not verified**: causal identifiability, effect on new datasets or in production, or a statistically significant
generalization gain. These features are observed geometric associations and do not establish causation. The
30,372 steps also cannot be treated as that many independent supervised samples. Global AUC and within-task BoN
measure different things; their difference alone cannot pin down any single overfitting mechanism.

**Not done**: the 32/38 acceptance target, the AUC/Spearman improvement, or the gain the geometric branch was meant
to bring. The final configuration chosen by training cross-validation was semantic kernel, alpha=0.001,
outcome_weight=0.1; the geometric branch did not win. Neither the 8B backbone nor the original baseline MLP was
fine-tuned.

Per-task comparison against the baseline: 2 tasks fixed, 4 regressed, net -2 tasks. The paired-bootstrap 95% interval
for the BoN=4 difference is [-18.42, +7.89] percentage points.
This is a small-sample evaluation; it cannot support any claim of statistical dominance.

5 avoidable selection errors made by the new head (a successful candidate existed in each case):

- anko-typed-variable-bindings
- claude-code-by-agents-recursive-delegation
- prometheus-transactional-reload-status
- tengo-callable-instance-isolation
- textual-richlog-follow-state

4 further tasks where every candidate failed, which no re-ranking could fix:

- bandit-structured-nosec-directives
- kcp-go-multiplexed-kcp-streams
- meriyah-explicit-resource-declarations
- obsidian-linter-link-format-conversion

Of the baseline's 7 failures, only 3 were avoidable selection errors; calling all 7 of them scoring-head selection
errors would be inaccurate. The complete baseline/new-head failed-candidate lists, scores, and rewards are in
`../deepswe_prm_enhanced_results.json`, under `wrong_tasks` and `per_task`.

Evidence entry points:

- Implementation: `benchmarks/eval_deepswe_enhanced_prm.py:59` features, `:139` ranking loss,
  `:228` training-set selection, `:296` real baseline load, `:322` frozen-model evaluation.
- Final weights: `benchmarks/artifacts/deepswe_prm/enhanced_rank_head.pt`.
  SHA-256: `8c039e3fe48cc671797af3cfc0b8c75a6482a4c38b88760ce6c0e26ed4d51a17`.
- Raw commands and exit codes: `handoff.json`; full logs: the matching `.txt` file in the same directory.
- Training, evaluation, the 6 tests, and independent verification all had raw exit code **0**.
- The target acceptance gate's raw exit code was **1**, see `acceptance.json` and `acceptance.txt`.
- Reproduction command and boundaries: `REPRODUCE.md`; run environment: `environment.json`.
- Local weights/features are excluded by the repo's existing ignore rules and are not carried along by an
  ordinary git commit.

No further model selection or hyperparameter tuning happened after the held-out result was revealed. Continuing to
iterate against these failed tasks toward 32/38 would contaminate the held-out set and cannot count as compliant
success evidence for this task's requirement.
