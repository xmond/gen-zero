# P1–P10 integration evidence

Workspace: `/ebs/pj/gen-zero-worktree/merge-road`
Branch: `feat/unified-merge-road`; starting commit: `fd1b8ba`.

## Scope and limits

The source branches are independent changes, not a tested cumulative series.
P9 was written against old single-step engines. Its deadline API must be ported
onto P5/P6/P7/P8 instead of replacing the new algorithms.

- MCTS is sequential finite-horizon PUCT with discounted ancestor backups and
  deterministic-model successor caching. The new search uses its own nodes;
  the separately tested atomic TypedArena does not prove concurrent MCTS.
- A* uses exact state keys and h=0, a valid Dijkstra special case. It requires
  an explicit goal; missing goals fail closed. No learned heuristic is claimed.
- CEM fits independent categorical distributions by time position. Gaussian
  parameter validation is present; continuous dynamics execution is unsupported.
- CFR implements two-player normal-form regret matching with a zero-sum gap.
  The production single-agent adapter has one opponent column, not a game tree.
- GFlowNet is an exact depth-one star flow sampler, not learned multi-step
  trajectory-balance training or general backward trajectory sampling.
- CP-SAT is finite candidate enumeration with explicit solve status, not an
  external CP-SAT solver or multi-step constraint encoding.
- Budgeted decisions bound the caller's wait using a worker and checkpoints.
  A synchronous model call cannot be killed. General-purpose OS scheduling
  does not provide a strict 2.000 ms worst-case guarantee.

These limitations are real capability gaps relative to the broad task wording;
passing tests cannot turn them into implemented capabilities.

## Conflict decisions

- Preserve P1 fail-closed i32 intermediate-domain checks using checked i128
  multiplication/accumulation. Negative underflow is rejected, not allowed by
  cancellation or a later comparison.
- Preserve NaN/Inf rejection across all six engines. P7's original divergent
  trajectory skipping conflicts with P2 and is replaced with error propagation.
- Preserve duplicate-action rejection and graph/context checks.
- Preserve request active_context when moving work into the deadline worker,
  and apply it when publishing anytime candidates.
- Preserve explicit goal requirements and expose serialized A* targets and
  request budgets through the service rather than silently discarding them.

## Verification

Only exit-code files survive in this directory. The exact commands, stdout and
full logs were not preserved, so the table below cannot be re-checked against
output. Earlier branch evidence is historical and does not certify this
integrated tree.

| File | Exit | Reading |
| --- | ---: | --- |
| `gate-first.exit` | 0 | passed |
| `planner-lib.exit` | 0 | passed |
| `planner-all.exit` | 0 | passed |
| `cli-tests.exit` | 0 | passed |
| `workspace-first.exit` | 101 | **failed** (cargo build or test error) |
| `packages-first.exit` | 101 | **failed** |
| `packages-second.exit` | 101 | **failed** |
| `packages-third.exit` | 101 | **failed** |

No passing workspace-wide or package-wide run is recorded here. Do not cite this
directory as proof that the integrated tree was green.

`commits.txt` lists the eight source commits; all eight still resolve in this
repository (checked 2026-09-29 with `git cat-file -e`).

## Test migrations (not relaxed safety assertions)

- The old MCTS f32 accumulator-overflow expectation no longer describes P5:
  return sums are now f64. Tests require a valid finite result for repeated
  finite f32::MAX rewards; NaN/Inf rejection across every latent coordinate
  remains exhaustive and unchanged.
- A* fixtures provide explicit reachable goals, including real model successors;
  they do not infer success from hazard/termination or use an always-true goal.
- Deadline fixtures account for root screening and final revalidation. A
  certified incumbent requires screening plus complete engine evaluation.
  The slow-model <3ms assertion and blocked-worker release test are retained.
  Search itself may consume the 2ms budget before the next slow call begins;
  tests do not require a forbidden post-expiration model call.
- P10's proposal to simulate blocked actions conflicts with the existing
  fail-closed simulation contract. Blocked simulation still returns an error
  before model execution; policy_allowed/hazard_free remain separate diagnostics
  for permitted rollouts.
