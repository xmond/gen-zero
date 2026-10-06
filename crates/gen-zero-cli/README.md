# gen-zero-cli

Command-line entry point for the Gen-Zero decision engine. Builds the
`gen-zero` binary, a thin `clap` front end over `gen-zero-service`.

## Architecture

Single binary crate. `src/main.rs` parses arguments with `clap` and wires them
into `gen_zero_service` (`McpServer`, `MountRegistry`, `PolymorphicZeroEngine`);
`src/reflex_cmd.rs` implements the `reflex-*` plugin commands on top of
`ReflexRegistry`, `ReflexOnlineAdapter` and `SqliteFeedbackStore`. Every
command prints JSON to stdout. A refused or gated outcome is a failed command
(exit 1).

## Binary and subcommands

Binary name: `gen-zero`.

- `audit-ledger`: execute a pipeline decision and print its in-process audit
  commitment; takes engine hyperparameter overrides (MCTS, MPC/CEM, A*,
  K-MoE router thresholds).
- `serve`: launch the polymorphic MCP service in stdio or SSE mode, with an
  optional auth token and cognitive assets file to mount before serving.
- `mcp`: alias for the MCP server command. Serves SSE on `--host` (default
  `127.0.0.1`) and `--port`, or stdio with `--stdio`.
- `keygen`: generate a cryptographically secure Gen-Zero connection token.
- `reflex`: evaluate a single-step reflex decision, optionally against the
  numeric cognitive runtime.
- `decide`: decide among candidates (verb `ask`/`decide`). `--mode` selects
  semantic ask (`auto`/`reflex`), semantic PUCT lookahead or planner MCTS
  (`mcts`), or the latent planner (`mpc_cem`, `astar`).
- `simulate`: roll a fixed action plan forward on the latent world model.
- `what-if`: compare candidate first actions on the latent world model.
- `audit`: shadow risk review of one planned action; never returns an
  approval.
- `entail`: Busemann entailment (`passage ⊃ question`) on the mounted preset
  geometry, with an optional tangent-event scheme.
- `fold`: causal relation trace fold on the discrete relation semiring.
- `qwen`: run the native Qwen2.5 scorer directly (no server, no Python): score
  candidates after a prompt, or assess text safety risk.

Reflex plugin commands (plugin archives come from `gen-zero-research` or a
prior `reflex-adapt`; see `examples/reflex_bench_fixture.rs` for an untrained
latency fixture):

- `reflex-bench`: predict latency percentiles against an in-process
  `ReflexRegistry`, plus live hot-swap latency under concurrent readers when
  `--patch` is given.
- `reflex-patch-create` / `reflex-patch-apply` / `reflex-patch-inspect`:
  build, apply and inspect a hash-verified differential patch. Create refuses
  to write a patch that does not reproduce its target bit for bit.
- `reflex-feedback-status` / `reflex-feedback-record` / `reflex-feedback-prune`:
  count, label and prune traces in the SQLite feedback store.
- `reflex-adapt`: one head-local gradient step on unconsumed labeled feedback,
  gated on the batch loss not regressing, hot-swapped into an in-process
  registry and written as a new checkpoint.

Not wired yet: `serve` does not load reflex plugins or record reflex traces,
so the feedback store is only filled by library callers of
`SqliteFeedbackStore::insert_trace`.

## Dependencies

- `gen-zero-service`: the MCP server, mount registry and `zero` tool engine
  this CLI drives, plus the reflex registry and online adapter.
- `gen-zero-model`: reflex plugin and patch archives.
- `gen-zero-storage`: the SQLite reflex feedback store.
- (dev-only) `gen-zero-core`, `gen-zero-nanocore`, `gen-zero-worldmodel`: used in tests.
