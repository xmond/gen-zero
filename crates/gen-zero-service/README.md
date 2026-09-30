# gen-zero-service

The Gen-Zero MCP server: a dual-transport server (stdio and SSE/HTTP REST), a
single polymorphic `zero` tool router with 11 cognitive verbs, a high-performance
simd-json protocol loop, the semantic bridge to the Python scorer, and the
Spec 25 cognitive runtime (mount snapshots, tangent SSM, geometry gate).

## Architecture

- `bridge`: semantic bridge client, Rust `zero` tool to the Python semantic
  scorer (`python/gen_zero/service/app.py`, port 8995 by default), exposing
  `/v1/semantic_ask`, `/v1/semantic_route` and `/v1/semantic_risk`.
- `cognitive`: the one cognitive runtime behind every entry (Spec 25 §1.2 /
  §5.1 / §5.5): tangent map, parallel SSM scan, geometry gate, and the action
  verifier, driven from a caller-captured `MountSnapshot`.
- `error`: crate error type (`ServiceError`).
- `imagine`: multi-step semantic lookahead for the `imagine` verb; PUCT Monte
  Carlo tree search over action sequences, with priors from a `PriorOracle`
  (the semantic bridge in production).
- `mount`: Spec 25 §5.2-§5.4 immutable mount snapshots and the CAS atomic
  swap. `MountSnapshot` is immutable and SHA-256 sealed; content changes go
  through `MountSnapshot::derive`, which bumps `Version`.
- `pipeline_verb`: the `pipeline` verb, exposing the planner's
  `ProductionPipeline` (`simulate`, `what_if`, `audit_action`, `decide`) over
  `zero`, MCP, HTTP (`POST /v1/pipeline/{op}`) and the CLI.
- `reflex_registry`: `ReflexRegistry`, lock-free (`ArcSwap`) serving of
  reflex plugins by task, with verified live patch hot-swap: readers keep the
  plugin they loaded while a patched copy is published.
- `reflex_adapter`: `ReflexOnlineAdapter`, one head-local softmax
  cross-entropy step over unconsumed SQLite feedback, rejected unless the
  batch loss does not regress, then hot-swapped and marked consumed.
  Neither reflex module is mounted by `server` yet; only the CLI's
  `reflex-*` commands construct them.
- `server`: the dual-transport MCP server (Doc 07): a simd-json stdio parsing
  loop, and Axum 0.7 HTTP/SSE routing (`/sse`, `/message`, `/v1/decisions`,
  `/v1/decisions/stream`, ...).
- `snapshot`: in-process golden snapshots of mounted cognitive assets, called
  from a trusted admin layer. Bounded to 16 entries per engine, do not
  survive a restart; rollback publishes a new generation.
- `tangent_ssm`: parallel tangent-space SSM (Spec 25, ch. 4 / §5.4). A step
  is an affine map `F_t = (M_t, q_t)`; steps compose associatively.
- `worldsim`: world-model rollouts behind `simulate`, `what_if`, `audit`,
  and the latent planner modes of `decide`. Dynamics chosen via `dynamics`:
  `residual` (default), `symplectic`, or `contact` (conformal symplectic flow on
  the contact manifold; `damping` sets the rate `gamma >= 0`, `gamma = 0` is the
  symplectic flow, `gamma > 0` contracts each `(q_i, p_i)` pair by
  `exp(-2 gamma dt)` per step).
- `zero`: the single polymorphic `zero` tool router, exposing 12 cognitive
  verbs (`ask`/`decide`, `route`, `imagine`, `stream`, `grep`, `compact`,
  `entail`, `causal_fold`, `pipeline`, `simulate`, `what_if`, `audit`). Every request
  captures one immutable mount snapshot; `ask`/`route`/`imagine` run request
  text through the semantic risk classifier before scoring.

## Key exports

- `BridgeConfig`, `BridgeError`, `SemanticBridgeClient`: the semantic bridge
  client.
- `CognitiveRuntime`, `Rejection`: the cognitive runtime and its typed
  refusals.
- `ServiceError`: crate error type.
- `AtomicMountRegistry`, `Budget`, `MountKey`, `MountRegistry`, `MountSnapshot`,
  `Proposal`, `Reject`, `RequestBinding`, `Snapshot`, `Version`: mount
  snapshots and the CAS registry.
- `McpServer`: the dual-transport MCP server.
- `ReflexRegistry`, `ReflexError`, `ReflexOnlineAdapter`, `AdaptationReport`:
  reflex plugin serving and head-local adaptation.
- `PolymorphicZeroEngine`, `ZeroContentBlock`, `ZeroToolOutcome`, `ZeroVerb`:
  the `zero` tool router and its outcome types.

## Dependencies

- `gen-zero-core`, `gen-zero-storage`, `gen-zero-model`, `gen-zero-nanocore`,
  `gen-zero-planner`, `gen-zero-worldmodel`, `gen-zero-lod`, `gen-zero-gate`,
  `gen-zero-provenance`: this crate integrates all other Gen-Zero crates into
  the runtime and MCP surface.

For `pipeline.decide`, `budget_ms` is an optional finite nonnegative request
budget; `planner_config.budget_ms` also applies and the earliest limit wins.
Timeouts never fabricate a candidate. The caller-side wait is bounded with
cooperative search checkpoints, not a hard real-time scheduling guarantee.

A* (including Auto tiers that invoke A*) requires an explicit `astar_goal`:
`{"state": [/* exactly 1024 finite numbers */], "tolerance": 0.001}`.
The target uses Euclidean latent distance. Missing goals fail closed; no goal
is inferred from `done`, reward, or the first candidate. `active_context` is
preserved in both ordinary and budgeted decisions.
