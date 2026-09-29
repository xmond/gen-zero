# DeepSWE Gen-Zero integration evidence

Workspace: `/ebs/pj/gen-zero-worktree/wire-deepswe`.
Base HEAD: `fd1b8bafbbbb23183c60c9a7050cecd23826e82d`.

Commands run against the final implementation:

```bash
pytest benchmarks/tests/test_deepswe_adapter_genzero_integration.py
# 29 passed, exit 0; integration.log and integration.exit
pytest benchmarks/tests/test_deepswe_adapter.py python/gen_zero/tests/test_latent_mcts.py python/gen_zero/tests/test_vectorized_latent_mcts.py python/gen_zero/tests/test_policy_evidence_gate.py
# 65 passed, exit 0; regression.log and regression.exit
python -m py_compile benchmarks/gen_zero_deepswe_adapter.py benchmarks/deepswe_genzero_gate.py python/gen_zero/world_model/imagination_planner.py
bash -n benchmarks/run_deepswe_smoke.sh
git -c core.fsmonitor=false diff --check
```

Logs were redirected to files without pipelines, and each pytest process's
original exit status was saved and returned. `first-run.log` preserves the
initial failure: `shutil.rmtree('tests')` escaped the default lexical gate.
The adapter now extends the real policy profile with destructive Python APIs;
that adversarial case passes. This demonstrates a repaired case, not a proof
that arbitrary malicious Python can always be detected.

Evidence chain:

- `benchmarks/gen_zero_deepswe_adapter.py:286`: every sandbox command gated before execution.
- `benchmarks/gen_zero_deepswe_adapter.py:298`: edits gated and ranked before upload.
- `benchmarks/deepswe_genzero_gate.py:99`: real DecisionPolicyGate, structural checks and semantic assessment.
- `benchmarks/deepswe_genzero_gate.py:151`: real multi-step MCTS backed by validated service transitions.
- `python/gen_zero/world_model/imagination_planner.py`: explicit strict evaluation forbids heuristic fallback.
- `benchmarks/tests/test_deepswe_adapter_genzero_integration.py`: actual policy/MCTS and loopback HTTP contract tests; Pier is a dispatch spy.

Limits: no live Pier/Docker/DeepSWE run or deployed learned cognitive service was
available or claimed. A compliant loopback cognitive service is now a required
runtime dependency; missing/invalid evidence stops execution. Service predictions
are not formal correctness proofs. Non-Python candidate edits fail closed.
Independent Luna read-only review identified deployment, planner fallback,
provenance and latency concerns. Required dependency is documented/preflighted,
source evidence includes the bridge, strict mode and transition budgets were
added. The general policy whitelist bypass is not used (profile whitelist is
empty). No unrelated changes or forbidden Git operations were performed.
