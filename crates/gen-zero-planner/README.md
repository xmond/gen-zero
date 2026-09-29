# gen-zero-planner

Gen-Zero's lookahead planning subsystem: 6 orthogonal planning engines, a
64-byte cache-line aligned compact tree node arena, and a dynamic K-MoE router
that picks how much deliberation a state gets.

## Architecture

- `config`: caller-tunable hyperparameters for `ProductionPipeline`'s search
  engines; only knobs with a measured effect on search or scoring are exposed.
- `engine`: the 6 orthogonal planning engines: `MctsEngine` (finite-horizon PUCT tree +
  model rollout + discounted ancestor backups), `AStarEngine` (uncertainty-penalized A* + backtrace
  pool), `MpcCemEngine` (rolling horizon + Cross-Entropy Method), plus
  `CfrNashEngine`, `CpSatFormalEngine`, and `ManifoldGFlowNetEngine`, all
  behind the shared `PlanningEngine` trait.
- `error`: crate error type (`PlannerError`).
- `pipeline`: the production pipeline (`simulate`, `what_if`, `audit_action`,
  and multi-mode `decide`) built on one rollout loop over one world model and
  one `PolicyGate`.
- `router`: dynamic K-MoE router. K=1 (reflex) for low-entropy states
  (H <= 0.2), K=2 (CP-SAT filter then A*/MCTS) for moderate uncertainty
  (0.2 < H <= 0.7), K=3 (multi-engine committee vote) for high-entropy or
  adversarial states (H > 0.7).

The gated entry point is `ProductionPipeline::decide`; `DecideMode::Reflex`
ranks candidates through the gated CP-SAT wrapper. The former `GenZeroPlanner`
(first Tier-0 candidate reflex), `PlannerContext`, and the unused legacy
tree/arena search nodes were removed: no production path called them.

## Key exports

- `PlannerConfig`: engine hyperparameters.
- `AStarEngine`, `CfrNashEngine`, `CpSatFormalEngine`, `ManifoldGFlowNetEngine`,
  `MctsEngine`, `MpcCemEngine`, `PlanningEngine`: the planning engines and
  their shared trait.
- `PlannerError`: crate error type.
- `AuditReport`, `AuditVerdict`, `CandidateOutcome`, `DecideMode`,
  `DecideRequest`, `Decision`, `ProductionPipeline`, `PrunedAction`, `Rollout`,
  `SimStep`, `WhatIfReport`, `DEFAULT_WARN_RISK`, `MAX_DECIDE_CANDIDATES`,
  `MAX_HORIZON`, `MAX_WHAT_IF_CANDIDATES`: the production pipeline.
- `DynamicKMoERouter`, `RoutingTier`: the K-MoE router.

## Dependencies

- `gen-zero-core`: base types and the `WorldModelDynamics` trait.
- `gen-zero-gate`: `PolicyGate` used by the pipeline and router.
- `gen-zero-lod`: graph/geometry types used by the planning engines.
- (dev-only) `gen-zero-worldmodel`: used in tests.

MCTS defaults to horizon 4 and discount 0.99, configurable through
`PlannerConfig::mcts_horizon` / `mcts_discount`. It caches deterministic model
successors, reuses the supplied gate-feasible action set at every depth, and
stops on `done`. Its rollout policy rotates actions; it has no learned leaf
value or finite-budget optimality guarantee. The persistent node budget fails
closed on exhaustion, as do model errors and non-finite transitions. Search is
sequential.

## Game solving, flow sampling, and formal status boundaries

The historic engine names are retained for compatibility; they do not certify
capabilities beyond the following implemented algorithms:

- `CfrNashEngine::solve_game` accepts two rectangular payoff matrices with
  `[row action][column action]` indexing. Both players perform simultaneous
  cumulative regret matching for the requested iteration budget. The result
  contains average mixed strategies and seeded-RNG sampling methods. In a
  zero-sum game, `zero_sum_gap` measures the best-response duality gap in original
  payoff units. Inspect that gap before claiming convergence; general-sum
  marginal strategies carry no Nash guarantee. The `PlanningEngine::plan`
  adapter has no opponent input: it explicitly solves a one-column game for
  4096 iterations and selects the modal action. It cannot model an adversary.
- `ManifoldGFlowNetEngine` samples an exact depth-one star graph. For supplied
  nonnegative target flows, `flow_distribution` returns `P_F(a)=R(a)/sum R` over
  gate-allowed leaves, and `sample` draws from that distribution. `plan` and
  `plan_with_rng` use world-model rewards as **log target flows**, i.e.
  `R(a)=exp(reward(a))`, with stable maximum shifting. There is no distance
  penalty, learned manifold, or multi-step trajectory-balance training. A
  candidate is a planning leaf even when the environment is not terminal.
  Invalid/zero total flows and numerical underflow fail explicitly; blocked
  branches are excluded before normalization and world-model evaluation.
- `CpSatFormalEngine::solve` enumerates the supplied finite candidates under
  the gate and returns `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, or `TIMEOUT`
  (`SolveStatus` serializes to these names). Only exhaustive evaluation yields
  `OPTIMAL`, scoped to these candidates and their one-step model rewards.
  Intentional first-feasible stopping yields `FEASIBLE`. Evaluation-budget or
  deadline exhaustion yields `TIMEOUT`, optionally with an incumbent; this is
  never an infeasibility or optimality proof. Errors abort the solve. Deadlines
  are cooperative and cannot interrupt a synchronous world-model call. This
  implementation has no general CP-SAT encoding or branch-and-bound backend.

Seeded distribution and convergence regressions are in
`tests/game_theory_and_sampling_tests.rs`.
## Decision deadlines and anytime results

Set `DecideRequest::budget_ms = Some(2.0)` or an absolute
`deadline: Option<std::time::Instant>`. `PlannerConfig` supplies the same limits
as defaults. The earliest request/config limit wins; a relative budget starts
at entry to `decide`, including validation, queue submission, gates, search and an
optional trajectory. Omitted limits retain the existing unbounded API. Negative,
non-finite and unrepresentable budgets are rejected. Zero/expired limits return
`PlannerError::TimeoutExceeded`; process-local `Instant` values cannot be supplied
through serialized configuration.

MCTS checks before each simulation, A* before each candidate expansion, and CEM
before each iteration, sample and horizon step. All engines check around model
calls. A cutoff returns only a candidate whose finite, nonterminal evaluation
completed and whose full PolicyGate check (including graph and request entropy)
finished before the deadline. CEM publishes only completed trajectories. Without
such a candidate the result is `TimeoutExceeded`, never a first/random action.
An anytime `Decision` carries `timed_out: true`, unknown distribution entropy
(`1.0`), and `trajectory: None`; confirm/escalate gate tiers remain explicit.
The gate verdict certifies policy admissibility at evaluation time, not calibrated
physical safety or optimality, and a partial Auto result is not committee consensus.
Model/numerical errors received before the cutoff remain errors.

**This is not a hard real-time guarantee.** `WorldModelDynamics::step` is synchronous
and cannot safely be killed. Budgeted decisions use a background worker and a
caller-side deadline wait; a running model call may finish after the caller
returns. No further model step is started after a checkpoint observes expiration.
One outstanding budgeted worker is allowed per pipeline, shared by its clones;
while it remains stuck, subsequent budgeted requests return `PlannerBusy`.
A ready worker is created during pipeline construction; no model runs during
initialization, and worker creation failure makes budgeted calls fail closed.
The final 200us use active polling to avoid sleeping past the target, consuming CPU
during that interval. OS scheduling and allocation can still exceed 2.000ms. Direct
`PlanningEngine::plan_until` calls provide cooperative checks only and can wait
for the current model call. `simulate`, `what_if`, and `audit_action`
do not acquire a deadline implicitly.

`tests/deadline_anytime_tests.rs` uses a real 3ms sleeping model with a 2ms budget
and asserts response in less than 3ms. It also proves return before a separately
blocked model is released, prevents worker accumulation, rejects partial CEM
trajectories, checks graph revocation and numeric faults, and covers optional
trajectory/Auto deadlines. Timing overruns fail the tests; passing on a general
purpose OS is regression evidence, not a worst-case scheduling proof.

```sh
cargo check -p gen-zero-planner
cargo test -p gen-zero-planner
cargo test -p gen-zero-planner --test deadline_anytime_tests
```
