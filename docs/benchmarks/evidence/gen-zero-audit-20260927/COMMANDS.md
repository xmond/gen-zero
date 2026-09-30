# Investigation Commands and Verification

Baseline: `/ebs/pj/gen-zero`, HEAD `6dfd6099591f68a8bba8bbb8ae3b2a6c302f7e2c`.

All logs come from actual executions in this investigation. The `.txt` files are complete copies of stdout and stderr, including failures. The JSON and exit files record the original exit codes. Paths under `/tmp/gen-zero-audit-20260927` in older JSON files refer to the initial artifact location. No real external model generation or redeployment to dev was performed.

| Command | Result | Log |
|---|---|---|
| `cargo test -p gen-zero-gate --lib --locked` | Exit 0; 43 passed | `gate-rust.txt` |
| `cargo test -p gen-zero-planner --locked` | Exit 0; 69 passed; 0 doc tests | `planner-rust.txt` |
| `PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q python/gen_zero/tests/test_policy_evidence_gate.py python/gen_zero/tests/test_issue_22_semantic_review_gate.py` | Exit 0; 43 passed | `gate-python.txt` |
| `PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q benchmarks/tests/test_gen_zero_tb_adapter.py benchmarks/tests/test_deepswe_adapter.py` | Exit 2; the adapter module was absent from the search path | `adapters-python.txt` |
| `PYTHONPATH=python:benchmarks PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q benchmarks/tests/test_gen_zero_tb_adapter.py benchmarks/tests/test_deepswe_adapter.py` | Exit 2; Harbor was unavailable in this Python environment | `adapters-python-corrected.txt` |
| `PYTHONPATH=python:benchmarks PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q benchmarks/tests/test_deepswe_adapter.py` | Exit 0; 9 passed | `deepswe-only.txt` |
| `python3 benchmarks/tests/verify_deepswe_prm_report.py --report benchmarks/results/deepswe_prm_enhanced_results.json --reference-repo /tmp/clmrepro/repo` | Exit 1; enhanced head weights were missing; the hash check was not skipped | `prm-report-verify.txt` |
| `PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 /tmp/gen-zero-audit-20260927/gate_probe.py` | Exit 0; printed three real gate bypass counterexamples | `gate-counterexample.txt` |

`gate_probe.py` is an investigation script. It neither loads or fabricates a model nor executes the evaluated actions. The script remains in the same directory; substitute its current absolute path when reproducing the run. The first interactive probe mistakenly called the nonexistent `.evaluate` method. The saved script and log use the actual `.evaluate_policy` method. These counterexamples are not a production task success or failure rate.

Read-only inspection commands (output is in the session tool record):

```bash
pwd
git status --short
git rev-parse HEAD
rg --files -g '*clm*' -g '*adapter*' -g '*choice*' -g '*nanocore*' -g '*policy*' -g '*worldmodel*' -g '*mcts*'
rg -n 'PolicyGate|LinearConstraint|evaluate_with|Tier3HardStop' crates/gen-zero-service/src python/gen_zero/service benchmarks/gen_zero*
rg -n 'add_constraint\(|register_confirm_action\(|PolicyGate::default' crates/gen-zero-service/src --glob '*.rs'
nl -ba crates/gen-zero-gate/src/constraint.rs
nl -ba crates/gen-zero-gate/src/policy.rs
nl -ba python/gen_zero/gate/policy_gate.py
nl -ba python/gen_zero/runtime/loop_state_machine.py
nl -ba benchmarks/results/deepswe_prm_evidence/RESULT.md
```

External source commands:

```bash
firecrawl --status
firecrawl search 'CLM-8B DeepSWE 81.6 Terminal Bench 87.6' --limit 3 --scrape -o .firecrawl/gen-zero-audit-clm.json --json
```

The analysis cites only the official model card, whose complete copy is `clm-official-model-card.md`. No project benefit was inferred from secondary articles or other leaderboards returned by the search.

Additional commands (earlier logs were retained):

```bash
cargo test -p gen-zero-service --lib imagine::tests --locked
PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q python/gen_zero/tests/test_world_model_simulation_endpoints.py -k 'simulate_neural_vector_runs_full_horizon or simulate_neural_vector_truncates_at_trap or what_if_neural_flags_trap_and_picks_safe or audit_neural_safe_approved_and_trap_rejected'
python3 /tmp/gen-zero-audit-20260927/parameter_probe.py
git diff 6dfd609 701d800 -- crates/gen-zero-planner/src/engine.rs crates/gen-zero-gate/src/constraint.rs crates/gen-zero-gate/src/dual_track.rs crates/gen-zero-gate/src/sheaf_gate.rs crates/gen-zero-planner/src/pipeline.rs python/gen_zero/client.py
cargo test -p gen-zero-gate -p gen-zero-planner --locked
```

The corresponding logs are `semantic-mcts-rust.txt` (7 passed), `worldmodel-python.txt` (4 passed, 50 deselected), `parameters.txt`, `concurrent-changes.diff`, and `current-head-rust.txt` (gate: 51 passed; planner: 78 passed). All commands exited 0. The last command ran after the concurrent changes were merged; a subsequent HEAD check still returned `701d800a5d3c55db6dcdb002b7bda5608a6bba11`. `parameter_probe.py` remains in the same directory.
