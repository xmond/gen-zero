# R4 evidence

This directory contains actual subprocess logs, return codes and per-trial reports.
`run_recorded.py LABEL COMMAND ...` writes LABEL.log and LABEL.json and propagates
COMMAND's original return code. A nonzero benchmark result is retained as failure.

Runtime used: `/tmp/r4-planners-venv/bin/python`, OR-Tools 9.15.6755.
The temporary venv was created with `uv venv --system-site-packages
/tmp/r4-planners-venv`, followed by `uv pip install --python
/tmp/r4-planners-venv/bin/python ortools`. To reuse the installed project dependencies,
its `r4_shared_dependencies.pth` points to
`~/.hermes-venv/lib/python3.11/site-packages`. The repository's default
`python` initially lacked OR-Tools; it has not been modified.

Reproduction from the repository root:

```sh
PYTHONPATH=python /tmp/r4-planners-venv/bin/python benchmarks/results/r4_evidence/run_recorded.py constraints_after /tmp/r4-planners-venv/bin/python python/gen_zero/scripts/benchmark_issue_76_constraint_compiler.py
PYTHONPATH=python /tmp/r4-planners-venv/bin/python benchmarks/results/r4_evidence/run_recorded.py planners_after /tmp/r4-planners-venv/bin/python python/gen_zero/scripts/run_12_planners_benchmark.py
PYTHONPATH=python /tmp/r4-planners-venv/bin/python benchmarks/results/r4_evidence/run_recorded.py league_after /tmp/r4-planners-venv/bin/python python/gen_zero/run_league_benchmark.py
```

`compare_original.py` reads the original compiler verbatim from Git HEAD and runs
it through the updated audit scenarios. It does not substitute a fake solver.
Its report includes the original source SHA-256 and HEAD. It writes the common
constraint report too, so run the final compiler benchmark AFTER this comparison.
The initial `*_before` runs used the original scripts with the default interpreter;
`constraints_original_audited` instead uses real OR-Tools and the updated audit.
Do not confuse these different baselines.

`tune_cpsat.py` compares workers, presolve and variable elimination on 1000 real
CP-SAT problems per configuration. `tune_compact.py` measures direct protobuf
construction and probing/symmetry settings. Their exit 0 means the experiment
completed, NOT that every trial met its deadline. All trials are retained.

Interpretation limits:

- External wall timing includes the entire decision call. No warm-up trials are
  discarded. Late results, missing solvers and unavailable proofs fail closed.
- A finite sample cannot guarantee hard real-time completion on shared Linux.
- Rejected decisions are counted separately from accepted unsafe decisions;
  rejection is not counted as successful solving.
- The 12 tests are algorithm fixtures, not evidence of trained model quality.
  Reflex must have a real loaded dual-head checkpoint; random initialization does
  not pass. The existing Qwen INT8 trunk artifact is a different architecture.
- League convergence means late Elo stability in this simulated game. It does not
  mean Elo improvement, Nash convergence, monotonic improvement or poker skill.
  The confidence interval is descriptive and not an IID statistical certificate.
- Functional tests may use a generous solver budget, explicitly noted in their
  source. Only the constraint benchmark establishes observations at 2.0ms.
