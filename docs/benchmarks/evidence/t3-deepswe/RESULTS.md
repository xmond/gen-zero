# T3 Results: Repair Goal Incomplete; Real Single-Task Evaluation Failed

## Implemented and verified

- Added `benchmarks/gen_zero_deepswe_adapter.py` and synchronized it to `/home/user/benchmarks/gen_zero_deepswe_adapter.py` on dev.
- This is a **fail-closed capability probe**, not a complete production repair adapter. It connects to the Pier plugin, reads code in the sandbox, makes real requests to four MCP verbs, and records raw logs and exit codes.
- Local `python3 -m py_compile benchmarks/gen_zero_deepswe_adapter.py` exited 0. Pier's Python environment on dev imported the module successfully and rejected all three invalid configurations. No mocks were used.
- Ran the official task `tomlkit-toml-table-converters` once. Job: `t3-deepswe-smoke-20260926T142503Z`; trial: `tomlkit-toml-table-converters__H9TAzpk`.
- The agent and verifier ran in different Docker containers: agent `1df8b337d3b6d1989c684b0b3684a76d9012cff718b41148d19a56a8b4858e5c`; verifier `6995e28ee18f411ce8079283e8e3c56fdcae31dfab4c41d261361309756d472f`. `raw/docker-events.jsonl` contains the raw Docker events. `raw/docker-execs.json` pairs commands and exit codes by execID, allowing verification that the separate verifier actually ran.

| Item | Actual result |
|---|---|
| Original Pier command exit code | 0 |
| Original Gen-Zero process exit code | 0 |
| Original agent probe process exit code | 78 |
| Original verifier `/tests/test.sh` exit code | 0 |
| Reward | **0; failed** |
| New requirements, F2P | **0 / 60** |
| Existing tests, P2P | 964 / 964 |
| `model.patch` exported by official collect | **0 bytes** |
| Exit code of this report's acceptance script | **1; rejected** |

`partial=0.94140625` reflects that the empty patch still passed the existing tests. **It must not be presented as a 94.1% repair success rate.** Successful execution of the verifier script is distinct from earning reward 1 for the task.

The actual invocation is in [pier-command.txt](raw/pier-command.txt); the complete runner is in [run-smoke.sh](raw/run-smoke.sh). Recheck the saved result with:

```bash
python3 docs/benchmarks/evidence/t3-deepswe/check_result.py
# Expected exit code: 1. The saved trial fails the repair acceptance criteria.
```

See [acceptance.json](acceptance.json) for the result. The raw trial directory is `raw/jobs/t3-deepswe-smoke-20260926T142503Z/tomlkit-toml-table-converters__H9TAzpk/`. It contains `result.json`, `agent/engine.stdout`, `agent/mcp.requests.jsonl`, `agent/adapter.exit`, `artifacts/model.patch`, `verifier/reward.json`, `verifier/test-stdout.txt`, and `verifier/ctrf.json`.

## Most serious findings

1. **The engine on dev contains hard-coded claims of capability.** It actually returned `expected_reward=1.45` and `formal_checked=true`; `/home/user/gen-zero/crates/gen-zero-service/src/zero.rs:330` and `:331` on dev hard-code those values. The `matches=1` at `:372` is also hard-coded. [The dev source excerpt](raw/dev-zero-excerpt.txt) preserves line numbers and the file SHA256. The adapter did not treat these success fields as evidence of code reasoning or correctness.
2. **The current worktree corrected the capability claims but still lacks the required capability.** `crates/gen-zero-service/src/worldsim.rs:4` states that the dynamics are untrained; `:13` states that there is no text encoder. A code hash or zero-padded array cannot serve as a latent state capable of predicting patch outcomes.
3. **`compact` returns size statistics only.** `crates/gen-zero-service/src/zero.rs:3895` returns no recoverable content. The actual response likewise reported only a size change of 47303→10918 bytes; it does not establish a working context compression and consumption path.
4. **There is no code generation backend.** The system cannot yet generate a minimal reproduction, candidate patches, or regression commands. The core solving path remains unimplemented; this is more than an outstanding test.

## Unverified

- A successful repair path, recoverable context compression, reliable code-state encoding, and a counterfactual model have not been verified.
- The timeout and cancellation cleanup paths have not been tested with fault injection.
- The old source directory on dev has no `.git`; `git rev-parse` exited 128, so its commit was not verified. Binary and source file hashes were saved, but they are not evidence of this worktree's HEAD.

## Incomplete

The production repair adapter, generative reproduction, best-patch selection, post-repair regression, and nonempty submitted patch remain incomplete. Completion requires a real code generation backend and a validated code-state and transition model. Alternatively, the user could explicitly change the approach to select patches using a generative model and real execution tests, with the Gen-Zero counterfactual capability claim withdrawn.

## Runtime and worktree boundaries

- No reference solution was read; no oracle was used; official tasks and verifiers were not modified; no subagents or reviewers were assigned.
- No existing module was replaced. The target file did not previously exist, so there was no old adapter symbol to migrate or remove.
- No stash, checkout, reset, clean, or force-push was performed; this worktree was not committed.
- The initial task-set clone exited 128 because concurrent infrastructure work had already created the same path. The existing task-set HEAD was checked against the official source before use. `dataset-clone.log` and its exit code were retained.
- During image preparation, the old engine was found to use the `action` protocol. The request was changed to use `action`, which both the current and old engines accept, and the probe's own source hash was added. Pier's parent process loaded the initial module, while the actual MCP subprocess executed the final file: `agent/adapter-source.sha256` matches `raw/adapter-final.sha256`. The parent process's newly added cancellation cleanup branch was not exercised in that run. `raw/sha256.txt` retains the old adapter hash from startup; this discrepancy was not overwritten.
- The full raw evidence is also retained on dev at `/home/user/benchmarks/t3-evidence/`.
