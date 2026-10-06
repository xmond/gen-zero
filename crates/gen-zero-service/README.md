# gen-zero-service

The Gen-Zero MCP server: a dual-transport server (stdio and SSE/HTTP REST), a
single polymorphic `zero` tool router with 24 cognitive verbs (core decision,
world-model pipeline, exact causal-DAG planning, LodGraph memory verbs,
Text-to-Graph induction, and the two-stage QA gate), a high-performance
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
- `zero`: the single polymorphic `zero` tool router, exposing 24 cognitive
  verbs (`ask`/`decide`, `route`, `imagine`, `stream`, `grep`, `compact`,
  `entail`, `causal_fold`, `causal_plan`, `pipeline`, `simulate`, `what_if`,
  `audit`, and 10 `graph_*` verbs: 8 LodGraph memory operations,
  `graph_induce`, and `graph_execute_operator`, and `qa_gate`). Every request
  captures one immutable mount snapshot; `ask`/`route`/`imagine` run request
  text through the semantic risk classifier before scoring. An unknown or
  non-string `action`, or a request with no `action` and no verb-selecting
  field, is refused with HTTP 400 `InvalidParams`; it is never answered as
  `ask`. `causal_plan` is also served at `POST /v1/causal_plan` and by
  `gen-zero causal-plan --input <file|->`.
- `text_to_graph`: `graph_induce` (RFC-20261002 Phase 1). A deterministic
  lexical rule parser, not a neural model: an answerability gate (fixed
  injection blocklist, heuristic confidence, refused below the threshold),
  then a causal action DAG in the planner's `CausalDagSpec` form, optionally
  deposited into the LodGraph as `depends_on` edges.
- `qa_gate` (alias `qa_verify`): the two-stage answerability gate
  (`gen_zero_gate::TwoStageDualTrackGateway`) over caller-supplied reader
  scores (`context`, `question`, `candidate`, `best_span_score`,
  `null_score`). Outside the ambiguity band stage 1 decides alone. Inside it,
  the tri-teacher verifier (`gen_zero_model::TriTeacherPairDecider`, Qwen2.5
  with the LoRA merged in) must be configured with
  `GENZERO_TRI_TEACHER_ADAPTER` plus `GENZERO_QWEN_MODEL_PATH`; without it the
  request is refused (HTTP 503, `GateError`), never answered from stage 1.
  The trained adapter is not shipped in this repository. Also served by
  `gen-zero qa-gate`.

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


## Operator closed loop

`zero` exposes these actions through MCP `tools/call` and HTTP `POST /message`.
First induce and deposit a DAG:

```json
{"action":"graph_induce","graph":{"text":"fetch data then clean it and save to db","auto_deposit":true}}
```

Each action carries the full `builtin:dcm_executor` signature (version `1`,
`operator_kind: hard_dcm`, `pure: false`, `embedder_space: null`). Use a returned
`deposited_node_ids` value, **not** its entity/action id, to execute:

```json
{"action":"graph_execute_operator","graph":{"node_id":0,"nonce":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","input":{"reason":"record this action"}}}
```

HardDcm writes the action and input into the graph runtime's bounded execution
audit and verifies that actual record against input, output and the node state.
It does **not** fetch, clean, save, run a shell, or interpret the action label as
a command. The returned `output.state_delta` identifies `effect: audit_append`
and `durability: process_local`; the graph does not apply arbitrary state deltas.
Nonces are exactly 32 bytes encoded as 64 hex characters; reuse is rejected,
including a different hex casing. Once the nonce gate accepts a nonce, later precondition or transit failures
still consume it. Target/signature checks happen before that gate. **Audit and nonce history do not survive restart**, including when node
storage is persistent; this is not durable exactly-once execution.

To use SoftPcm, `graph_deposit` a node with:

```json
{"name":"builtin:pcm_evaluator","operator_kind":"soft_pcm","embedder_space":null,"version":"1","pure":true}
```

as its `operator`, then call `graph_execute_operator` with its node id and input,
omitting nonce. It returns a deterministic node-state summary and audit count,
without writing state or audit records. A supplied nonce is rejected. This is a
state evaluator, not a trained cognitive model.

`graph_prune` revocations and falsified nodes block execution with typed errors.
The graph rechecks after preconditions and holds a read lock through transit, so
revocation cannot commit between that check and the side effect. Implementations
must not call graph mutations from transit. Postconditions receive output,
input and the node at completion. Dependencies remain graph evidence; this verb
executes one selected node and does not automatically schedule an entire DAG or
enforce completion of its ancestors.

The induction engine remains `lexical_rule_parser_v1`: tests demonstrate grammar
handling and production wiring, not trained intent extraction or causal reasoning.
