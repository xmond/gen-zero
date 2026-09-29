# T3 DeepSWE integration evidence

Status: **incomplete**. The adapter is an explicit capability probe, not a
production repair agent. Do not describe a failed trial, an empty collected
patch, or Pier's job exit code as a solved task.

## Established protocol

- Official DeepSWE source: https://github.com/datacurve-ai/deep-swe,
  revision `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`.
- Pier installed on dev: `datacurve-pier==0.3.1`.
- Custom host-side agent: `--agent-import-path module:Class`, implementing
  `BaseAgent.setup(environment)` and `run(instruction, environment, context)`.
- Official task: `tomlkit-toml-table-converters`. The task instruction is passed
  by Pier; repository inspection uses `environment.exec` in the agent container.
- The task's collect hook exports `git diff --binary BASE HEAD`, not the working
  tree diff. A successful solver must commit inside its disposable task repo.
  No commit was made in the shared development worktree.
- The task sets `verifier.environment_mode = "separate"`. The verifier receives
  `/logs/artifacts/model.patch` and grades in a pristine container. The adapter
  does not read held-out tests or the reference solution.

## Why repair is blocked

1. Gen-Zero's Rust `compact` compresses the text but returns only byte counts:
   `crates/gen-zero-service/src/zero.rs:3895`. There is no recoverable payload
   available to an adapter. Counting compressed bytes is not usable LLM context
   compression.
2. `crates/gen-zero-service/src/worldsim.rs:4` explicitly describes untrained,
   uncalibrated dynamics; lines 13–15 require a numeric latent and state that no
   text encoder exists. Hashing code or padding zeros would not be a validated
   state representation. Python `GenZero.what_if` uses heuristic transitions
   (`python/gen_zero/client.py:1459`), not execution of candidate code patches.
3. No code-generation backend has been configured or implemented by this
   adapter. It cannot generate a reproduction or a repair. No fixed solution,
   oracle patch, synthetic reproduction, or test-bypass path was substituted.
4. The existing dev binary is older than this worktree: its `tools/list` has
   only `zero`, uses `action`, and does not advertise the current `verb` or
   `lines` schema. Its raw outputs must not be attributed to this source HEAD.

## Implemented

The adapter implements the Pier plugin interface, sandbox source acquisition,
real MCP process calls for `grep`, `compact`, `what_if`, and `simulate`, raw
stdout/stderr and process exit recording, bounded execution, and explicit
nonzero failure. Text world-model requests are diagnostic rejection probes,
not successful counterfactual simulations. `NonZeroAgentExitCodeError` lets
Pier continue to its official collect/verifier stages while preserving the
agent failure. There is no success override.

## Not implemented / not verified

Generated minimal reproduction, actual candidate patches, grounded code
counterfactual ranking, regression validation, and a successful committed
repair are **not implemented**. No repair pass rate or SOTA claim follows.

Runtime details and final validation results are recorded in `RESULTS.md`.
Full dev evidence: `/home/user/benchmarks/t3-evidence/`.
