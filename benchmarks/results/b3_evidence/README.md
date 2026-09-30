# B3 run evidence

Formal summary: `../league_and_constraints_eval_results.json`. Overall verdict: NOT_ACCEPTED.

## Raw commands and exit codes

- `python python/gen_zero/scripts/benchmark_issue_76_constraint_compiler.py`: 0; see constraints.log. The original environment lacks OR-Tools, so this cannot count as a real CP-SAT measurement.
- `python python/gen_zero/run_league_benchmark.py`: 1; see league.log. Historical average win rate 52%, minimum 30%, the assertion is not satisfied.
- `python python/gen_zero/evaluate/web_agent_benchmark.py`: 0; see web_agent.log. Only 3 scenarios, 30 action/parameter matches, 20 successes; this is not a real web task.

## Supplementary run

Dependency install: `uv venv --system-site-packages /tmp/b3-eval-venv`, `uv pip install --python /tmp/b3-eval-venv/bin/python ortools`. Install exit code 0, version recorded in environment.json. The first run failed for missing torch; the failure log was kept. It was retried after explicitly loading the original environment's `/home/luy/.hermes-venv/lib/python3.11/site-packages` via a `b3_base_dependencies.pth` file in the isolated venv.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python /tmp/b3-eval-venv/bin/python benchmarks/b3_measure.py
```

Formal supplementary run exit code **1** (degradation and League acceptance failed); see supplemental_final_command.json / supplemental_final.log. supplemental.json retains all 1000 solves, 20 compiles, and the 3240 training matches with policy trajectories across 3 seeds. The collector only observes real call returns and does not modify production methods. No profiler runs during the timed interval; the League's profiler is only used to identify actually-active objects, to keep archive ID reuse from polluting per-role win rates.

- Compile throughput: 34.2633 rules/s.
- Nominal solve-rule throughput: 1714.5873 rules/s (8 rules x 1000 iterations / total call seconds, includes degraded cases, not pure CP-SAT throughput).
- Solve wall-clock min/max: 0.033798 / 16.054933 ms; mean 4.665846 ms; P99 9.117924 ms.
- 242/242 violating proposals rejected; 1000 samples with no unsafe pass-through, but this is only limited-sample evidence.
- 844 CP_SAT_OPTIMAL labels, 53 unique feasible candidates returned directly, 103 UNKNOWN degradations. Labels come from production code and do not guarantee the raw OR-Tools state was OPTIMAL in every case.
- Real web-task success rate, interaction step count, and backtrack rate were not measured; the summary uses null for these and candidate-action counts must not stand in for step counts.

argv, environment, and raw subprocess exit codes for every command are in `*_commands.json` / `*_command.json`. The driver script itself returning 0 does not mean the subcommands passed. The first supplementary-run environment collection also failed for missing pytest metadata; that run produced no environment.json, so only the later successful collection is the formal environment record.

## Verification and limits

focused_tests.log: 19 passed, exit 0; web_tests.log: 12 passed, exit 0. Pre-existing synthetic/stub cases in the unit tests must not be cited as evidence of real agent capability.

Fixed authentication boilerplate and unmeasured weight drift found in legacy entrypoint output are kept as raw evidence and are not part of the acceptance verdict. supplemental_prior.json is an early diagnostic run whose per-ID role win rates are affected by archive name reuse; it has been superseded by the formal file. Do not cite its per-role win rates.

This task was a re-run and record of evidence. It did not fix pre-existing defects in the production model/benchmark, did not replace any production module, made no commit or push, and launched no subagents or reviewers. Detailed issues and path:line references are in the formal summary findings.
