# gen-zero-cli

Command-line entry point for the Gen-Zero decision engine. Builds the
`gen-zero` binary, a thin `clap` front end over `gen-zero-service`.

## Architecture

Single binary crate (`src/main.rs`), no library modules. It parses arguments
with `clap`, wires them into `gen_zero_service` (`McpServer`, `MountRegistry`,
`PolymorphicZeroEngine`), and prints JSON outcomes to stdout. A refused or
gated outcome is a failed command (exit 1).

## Binary and subcommands

Binary name: `gen-zero`.

- `audit-ledger`: execute a pipeline decision and print its in-process audit
  commitment; takes engine hyperparameter overrides (MCTS, MPC/CEM, A*,
  K-MoE router thresholds).
- `serve`: launch the polymorphic MCP service in stdio or SSE mode, with an
  optional auth token and cognitive assets file to mount before serving.
- `mcp`: alias for the MCP server command, with explicit `--stdio`/`--sse`
  flags.
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

## Dependencies

- `gen-zero-service`: the MCP server, mount registry and `zero` tool engine
  this CLI drives.
- (dev-only) `gen-zero-core`, `gen-zero-nanocore`, `gen-zero-worldmodel`: used in tests.
