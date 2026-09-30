# F01–F09 Regression Evidence

Worktree: `/ebs/pj/gen-zero-worktree/rm-p10`; baseline HEAD `6dfd6099591f68a8bba8bbb8ae3b2a6c302f7e2c`.
Audit basis: `docs/architecture/planner_architecture_optimality_audit.md`, Section 5.3.
Toolchain: `rustc 1.96.0 (ac68faa20 2026-05-25)`, `cargo 1.96.0 (30a34c682 2026-05-25)`.

These tests use a synthetic fault model that explicitly returns `Ok`, to avoid a real model's own validation masking planner defects. No claim is made that the six engines have full algorithmic capability, real-task safety, or optimality.

## Code Evidence

The line numbers below correspond to the files after this change; paths are relative to the repository root.

| Defect | Regression test: `crates/gen-zero-planner/tests/fault_regression_tests.rs` | Implementation and coverage boundary |
| --- | --- | --- |
| F01 | :124 | `crates/gen-zero-gate/src/constraint.rs:78`, `:92`, `:103`: single-action, bundle, and context all use i128 multiply-add; the test exercises repeated MAX coefficients and concurrent duplicate occupancy, and does not rely on a panic to count as success. |
| F02 | :149 | `crates/gen-zero-planner/src/engine.rs:211`: MCTS validates the full transition; direct calls into all six engines reject NaN/±Inf reward, with a valid model used as the positive control. |
| F03 | :181 | `crates/gen-zero-planner/src/engine.rs:697`: CFR validates the successor; MCTS/CFR reject NaN/±Inf in the trailing coordinate. |
| F04 | :204 | `crates/gen-zero-planner/src/pipeline.rs:588`: `decide_with_context` uses the occupancy snapshot for both the initial filter and the final gate; the test checks HardStop, the rule number, and zero model calls — all seven modes reject. |
| F05 | :244 | After registering a real SheafProblem, an action lacking a certificate is rejected by both the gate and all seven decide modes, with zero model calls; actions with no certificate requirement remain selectable. No planning entry point with a live certificate has been implemented. |
| F06 | :284, :439, :454 | `crates/gen-zero-planner/src/pipeline.rs:633`, `:38`, `:705`: candidate first-step filtering, mid-search rejection, and final first-step re-verification. All seven modes exclude high-reward done actions; rejection occurs when everything is hazardous; also tests a transient hazard that appears during search and a final prediction that turns bad. |
| F07 | :309 | `crates/gen-zero-planner/src/pipeline.rs:138`, `:145`: `policy_allowed()` and `hazard_free()` are independent; `is_safe()` is their conjunction. The test covers all four true/false combinations, preserving explicit counterfactual simulation. |
| F08 | :332 | `crates/gen-zero-planner/src/lib.rs:52`, `:91`: the context-free legacy Reflex now explicitly rejects; the explicit gate API filters out forbidden first actions and rejects when all are forbidden; ProductionPipeline Reflex is also covered. |
| F09 | :387 | `crates/gen-zero-planner/src/router.rs:104`: both dispatch entry points reject invalid entropy, with zero model calls; valid entropy covers all three routing bands. The router preserves the gate's filtering result for the requested entropy. |

The three gate/router/Reflex files were modified on Luna's side branch; the main thread verified the actual diff and HEAD on the shared worktree, and is responsible for the rest of the implementation, all regression testing, and integration acceptance. No commit, push, or worktree-restore operation was performed.

## Raw Verification Commands and Logs

All commands were run in the working directory above. Each command redirected its full stdout/stderr directly, and `$?` was captured before the log was read; no piping was used that would truncate the test output and swallow the exit code.

```bash
cargo test -p gen-zero-planner --test fault_regression_tests
cargo check -p gen-zero-planner -p gen-zero-gate
cargo test -p gen-zero-planner -p gen-zero-gate
cargo test --release -p gen-zero-planner --test fault_regression_tests
```

Debug targeted regression: 11 passed / 0 failed, exit code 0; compile check exit code 0; full tests for both packages: 126 passed / 0 failed, exit code 0; release targeted regression: 11 passed / 0 failed, exit code 0.

Final results are in the same directory: `regression.log`/`.exit`, `check.log`/`.exit`, `all-tests.log`/`.exit`, `release.log`/`.exit`.
`initial-compile.log` retains the first compile error in the test code (a unit struct was mistakenly written as `::default()`, exit code 101); after the fix, `intermediate-tests.log` records an intermediate version passing 10/10, exit code 0. Neither is the final acceptance log.

## Unimplemented Capabilities and Compatibility Impact

- F05 only locks in the rejection of actions lacking a certificate; **it does not mean a usable planning entry point with a certificate already exists**.
- The default `decide` explicitly uses an empty concurrency context; callers with occupancy must use `decide_with_context`. The snapshot is not an atomic cross-request resource reservation; there is as yet no TOCTOU guarantee.
- `done=true` is treated as hazardous inside decide, including the conservative rejection that a normal-termination model may produce; distinguishing normal termination from hazard would require a separate change to the model contract.
- Only the model outputs observed in this run were verified. Safety with a random model, or in a real environment, cannot be inferred from a finite number of model calls; no new full-trajectory reachability proof has been added. A hazard observed during search causes an immediate error, with no fallback to allow it through.
- The original `DecisionEngine::evaluate_reflex` had no policy parameter and now always returns an explicit error; callers must migrate to the explicit gate entry point. That entry point performs only basic policy filtering and does not predict dynamic hazards.
- `simulate` can still generate counterfactual trajectories that policy forbids; `is_safe` now requires both policy permission and the absence of a model hazard to hold simultaneously.
- decide adds a candidate prediction and a final prediction, increasing model-call cost; latency and online service integration have not been verified, and no claim of meeting 2ms is made.
