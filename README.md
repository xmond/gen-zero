# Gen-Zero

[![Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Rust 1.88+](https://img.shields.io/badge/Rust-1.88%2B-orange.svg)](Cargo.toml) [![MCP](https://img.shields.io/badge/MCP-stdio%20%7C%20SSE-purple.svg)](crates/gen-zero-service/README.md) [![Latency](https://img.shields.io/badge/latency-subsecond%20target-green.svg)](benchmarks/README.md)

**A pure-Rust cognitive decision engine and runtime SDK for agents that need to choose, look ahead, and know when to stop.** Gen-Zero combines reflex plugins, latent world-model simulation, search, and a safety gate in a Rust runtime exposed through a CLI and Model Context Protocol (MCP) server. The production server has no Python runtime dependency. Some semantic scoring paths use an optional Python bridge and fail closed when required evidence is unavailable.

- **Zero token:** a reflex decision takes a feature vector and returns a probability distribution over named candidates. No tokens are generated.
- **Sub-millisecond reflex, measured:** predict p50 164 µs, p99 238 µs on one thread at input dim 1024 (see [Reflex runtime](#reflex-runtime)). The 100 µs target is not met at that shape on the measured machine.
- **Hot-patchable:** hash-verified differential patches swap a plugin's weights under concurrent readers; the delta add uses AVX2 when the CPU has it.

## Product boundary

| This repository (`gen-zero`) | `gen-zero-research` and the tuning API (`tuning.gen-zero.ai`) |
| :--- | :--- |
| Rust runtime engine: reflex plugin inference, patch apply and hot-swap, feedback capture store, head-local online adaptation, planning, world-model simulation, safety gate, audit ledger, MCP/HTTP/CLI | Offline continuous learning: teacher training, curriculum, distillation, replay, self-play, large-model feature extraction |
| Python client SDK (`python/gen_zero`: `client.py`, `protocol/`, `mcp/`, inference-side modules) | Compiling trained checkpoints into reflex plugins and patches |
| Public benchmark data used for evaluation | Private training pools and datasets |

A reflex plugin or patch is produced upstream and consumed here. This repository ships no training loop for reflex operators; `reflex-adapt` only takes one bounded, loss-gated gradient step on a single linear head.

## Why Gen-Zero?

- **Look ahead:** simulate candidate actions in latent state and use MCTS, MPC/CEM, or A* planning where the input contract supports them.
- **Decide quickly:** native reflex and search paths target low latency. Timing depends on mode, assets, hardware, and any external scorer; see the [benchmarks](benchmarks/README.md) for measured scope.
- **Fail closed:** missing models, invalid assets, and unsafe or uncertain semantic requests can produce an explicit refusal or confirmation request instead of a fabricated decision.
- **MCP native:** one Rust binary offers CLI commands and MCP over stdio or SSE/HTTP, with bearer-token authorization for network service.

## Quickstart

Requires Rust 1.88 or newer. From the repository root:

```bash
git clone https://github.com/xmond/gen-zero.git
cd gen-zero
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo build --release
./target/release/gen-zero --help
```

Run MCP locally via stdio, or start a token-protected SSE endpoint:

```bash
./target/release/gen-zero serve --mode stdio
GENZERO_API_KEY=replace-with-a-secret ./target/release/gen-zero serve --mode sse --port 8999
```

Run a local numeric smoke test for the Qwen3.5-9B feature adapter (requires only `numpy`):

```bash
python3 examples/run_9b_demo.py
```

This is a lightweight local smoke test, not a benchmark or a 9B model run: it loads the packaged
adapter (`artifacts/qwen35_9b/zero_rnn_set_adapter_qwen35_9b.npz`, ~6.4MB) and checks numeric
soundness plus Lyapunov spectral stability (`sigma_max_A < 1`). The full Qwen3.5-9B base model never
runs on CPU here; the only dependency is `numpy`, and the input is a pre-extracted feature vector, not
raw text. On a fresh clone, `artifacts/qwen35_9b/parity_val200.npz` is not shipped (it is gitignored),
so the script safely falls back to a fixed-seed (seed=0) synthetic vector — a code-path check, not a
measurement of model quality. Latency has two parts measured on this machine: a one-time cold-start
weight load (~40ms) and warm per-sample scoring after that (~2-5ms, timed after an explicit warm-up
call); both numbers vary with hardware and candidate count. See [QUICKSTART.md](QUICKSTART.md) for
client configuration and deployment details.

### Instant Decision Benchmark (CPU-Only Local Reproduction)

Reproduce the trap-avoidance ablation below on your own machine, CPU only, no checkpoint or feature
file downloads — just this repo's own Python package (`numpy` and `torch`, both already needed
elsewhere in this repo; see [Quickstart: Python](#quickstart-python) below). It generates its own
seeded torus mazes, so the success, trap, and step-count columns you get should match the numbers
here; latency is wall-clock and varies by machine and run:

```bash
python3 benchmarks/suites/benchmark_world_model_mcts_ablation.py --episodes 10
```

Takes about 4s on CPU: importing this repo's Python package (numpy, torch) is a one-time ~2.5s cost,
and the episode loop itself is fast — about 30ms for 10 episodes, about 290ms for the default
`--episodes 100`. Real output from this repository, this machine:

```text
====================================================================================
Gen-Zero MCTS (exact-graph oracle) vs Greedy Baseline -- Deadlock Torus (10 episodes, seed 0-9)
====================================================================================
Method                             | Success % | Trapped % | Mean Steps | Latency ms
------------------------------------------------------------------------------------
Greedy baseline (no model)         |      0.00 |    100.00 |       1.00 |      0.010
Gen-Zero MCTS (exact-graph oracle) |    100.00 |      0.00 |       4.60 |      0.544
====================================================================================
McNemar exact (paired): baseline-only=0, world_model-only=10, p=0.00195
CAVEAT: "world model" here is env.step()'s privileged exact graph dynamics, not the
trained neural network. This is a planning diagnostic, not evidence of neural world-model
or production-planner performance. See README.md "What is measured" before quoting it.
====================================================================================
```

**Read the caveat in the box above before you quote this table.** The greedy baseline walks straight
into an absorbing trap on its first move every time (0% success, not a "deadlock loop"); the MCTS
column uses the environment's own exact transition function as its "world model," which is a
privileged oracle, not the trained neural network described in [What is measured](#what-is-measured)
below. The search loop itself also lives only in this benchmark script, not in the Rust production
planner. This benchmark shows the planner uses a correct model correctly — nothing about neural
dynamics or a production planner.

## Architecture

```text
CLI or MCP client
       │
       ▼
gen-zero binary ── gen-zero-service (stdio / SSE / HTTP)
       │                    │
       └──── cognitive runtime, mount registry, safety gate
                            │
              Rust world model + planner + nanocore + provenance
                            │
                   optional semantic bridge
```

The [Rust workspace](crates/) contains the decision engine, world model, planners, gate, storage, provenance, service, and CLI. The [MCP service](crates/gen-zero-service/README.md) routes `zero` and `pipeline` tools through the cognitive runtime. Mounted assets are versioned and checked before use.

## What is measured

### Neural world model, validation split

Source: `benchmarks/results/world_model_training_report.json` (produced at git `2d27b8a`, CPU, torch 2.14).
Data: 8,146 transitions from 1,620 episodes, split by episode (6,454 train / 1,692 val), state dim 64,
16 actions. Data sha256 `a7bd2f77…`, checkpoint sha256 `0e748c33…`.

| Metric | Value | Baseline in the same report |
| :--- | ---: | :--- |
| Val MSE, per state dimension | **0.015217** | identity transition: 0.19541 |
| Val AUC, safe-or-goal head | **0.99993** | constant reward BCE: 0.3855 (val BCE 0.01059) |
| Val reward accuracy | 0.99704 | |

What this shows: on held-out episodes from the same generator, the model predicts the next latent state
about 13x better than "nothing changes", and separates safe from dead states almost perfectly.
What it does not show: anything about other environments or about planning quality.

### Trap avoidance, MCTS vs. greedy, paired

Source: `benchmarks/results/world_model_mcts_ablation_report.{json,md}`.
100 seeded torus episodes, 32 simulations per decision, paired per seed, exact McNemar test.

| Policy | Success | Trapped | Mean steps | Mean decision latency |
| :--- | ---: | ---: | ---: | ---: |
| Greedy one-step (no model) | **0 %** | 100 % | 1.0 | 0.011 ms |
| MCTS with world model | **100 %** | 0 % | 4.71 | 0.682 ms |

McNemar: baseline-only wins 0, model-only wins 100, p = 1.6e-30. Latency is machine-dependent; this row
is from this repository's own regenerated report, not a historical figure.

**Read the caveat before quoting this.** The world model in this ablation is the *privileged exact graph
dynamics* of the environment, not the trained neural network. The report's own first line says it cannot
establish neural-world-model gains or production planner gains. It shows that the planner uses a model
correctly when the model is right. The neural checkpoint has not yet been run through this ablation.

### Semantic risk gate, held-out set

Source: `python/gen_zero/service/risk_data/README.md` (Qwen2.5-0.5B, fp32 CPU, 36 held-out requests in
five languages plus bare shell commands, AUC 0.944).

| Held-out class | Hard stop | Escalate | Proceed |
| :--- | ---: | ---: | ---: |
| Dangerous (18) | 12 | 6 | **0** |
| Safe (18) | **0** | 9 | 9 |

No dangerous request proceeded and no safe one was hard-stopped, but half of the safe requests were
escalated to a human. Known miss, documented there with a strict xfail test: bare `chmod -R 777 /` scores
the same as `git status` and proceeds. The claim "every bare destructive command is gated" is false.

### Not measured

Reflex predict latency is now measured (see [Reflex runtime](#reflex-runtime)); it does not meet
100 µs at input dim 1024. Earlier versions of this README also quoted a 0.85 ms planning SLA, a 0.00 % versus
14.2 % permutation flip rate against named LLMs, a 450 MB memory ceiling, and node-allocation and ledger
throughput figures. No artifact in this repository measures them end to end. The planning latencies were
hardcoded constants in `benchmarks/suites/latency_suite.py`. Treat latency and memory budgets
as engineering targets, not measured guarantees. There is no evidence here for
"100.0% zero breach", "<0.1ms reflexes", "infinite lookahead", or elimination of catastrophic
forgetting. Planning uses finite horizons and compute budgets.

---

## Reflex runtime

A reflex plugin (`gen-zero-model::ReflexPlugin`) is a contractive low-rank recurrent operator
over an input vector plus one or more named linear heads. It is stored as a hash-addressed
archive and served from `gen-zero-service::ReflexRegistry`, which swaps plugins with `ArcSwap`
so readers never block.

### Commands

```bash
# Latency of predict, and of a live patch hot-swap under concurrent readers.
./target/release/gen-zero reflex-bench --plugin base.gzr --threads 4 --iterations 20000 [--patch p.patch]

# Differential patches: create refuses to write a patch that does not reproduce its target.
./target/release/gen-zero reflex-patch-create --base base.gzr --target target.gzr --out p.patch
./target/release/gen-zero reflex-patch-inspect --patch p.patch
./target/release/gen-zero reflex-patch-apply --base base.gzr --patch p.patch --out applied.gzr

# SQLite feedback store (WAL): counts, labels, pruning.
./target/release/gen-zero reflex-feedback-status --db fb.sqlite3 --input-dim 1024
./target/release/gen-zero reflex-feedback-record --db fb.sqlite3 --input-dim 1024 --trace-id <id> --label <candidate>
./target/release/gen-zero reflex-feedback-prune --db fb.sqlite3 --input-dim 1024 --before-unix-ms <ms>

# One head-local adaptation step on unconsumed labeled feedback, loss-gated, hot-swapped, checkpointed.
./target/release/gen-zero reflex-adapt --plugin base.gzr --out adapted.gzr --db fb.sqlite3 --task <task> --head <head>

# MCP / HTTP service (does not load reflex plugins yet, see below).
./target/release/gen-zero serve --mode stdio
```

Build with `--release`. A debug `cargo run` of the same benchmark measured p50 6.7 ms, about 40 times slower.

### Measured latency

Plugin: input dim 1024, LoRA rank 16, 8 recurrence steps, one head with 4 candidates, written by
`cargo run --release -p gen-zero-cli --example reflex_bench_fixture -- --out base.gzr --seed 1`.
Its weights are seeded random, not trained: latency depends on the shapes, the decisions do not
mean anything. Release build, 24-core x86_64 host shared with other jobs (loadavg 4.8 to 7.7 during
the run), 20,000 predictions per thread.

| Run | p50 | p90 | p99 | max |
| :--- | ---: | ---: | ---: | ---: |
| predict, 1 thread | 164 µs | 194 µs | 238 µs | 3.1 ms |
| predict, 4 threads | 173 µs | 204 µs | 246 µs | 8.9 ms |
| predict, 4 threads, with one live hot-swap | 172 µs | 197 µs | 225 µs | 4.2 ms |

The live hot-swap itself (`apply_patch_live`: clone, add, fix-ups, two SHA-256 checks, publish)
took 3.8 ms for a patch between two independently seeded plugins; readers kept serving during it.
Latency is timed with microsecond resolution per call and includes the registry lookup.

### Patch exactness and SIMD scope

`base + (target - base)` is not always bit-identical to `target` in f32 (sign changes, very
different magnitudes, overflow). A patch therefore carries the f32 delta plus a sparse list of
exact fix-up values for every element where the add does not round to the target; the example
patch above needed 5,183 of 54,308 elements. Apply adds the delta with AVX2 when the CPU supports
it (runtime detection, scalar loop otherwise), writes the fix-ups, and rejects the result unless
it hashes to the patch's declared target. SIMD is used only in patch diff and apply, not in predict.

### Not wired yet

`serve` does not load reflex plugins, expose a reflex predict route, or record reflex traces.
`ReflexRegistry` is constructed only by the CLI commands above, and nothing in the serving path
calls `SqliteFeedbackStore::insert_trace`, so `reflex-feedback-record` has no trace ids to label
unless a library caller inserts traces. Mounting the registry and trace capture into `serve` is
open work.

---

## Two cores: Rust and Python

| | Rust (`crates/`) | Python (`python/gen_zero/`) |
| :--- | :--- | :--- |
| Role | Runtime engine and serving gateway | Client SDK, world model, simulation |
| Entry points | `gen-zero serve` (MCP stdio / SSE), `reflex`, `entail`, `fold`, `keygen` | `GenZero()` client, `python -m gen_zero.cli semantic` HTTP service |
| Reflex decision | `reflex` / `ask` verb, with semantic scoring through the Python bridge | `decide(mode="reflex")` |
| Planning engines | six `PlanningEngine` impls (one-step-payoff approximations for three of them: GFlowNet, CFR, CP-SAT; MCTS is a genuine finite-horizon search, see above) | six experimental engines, Dynamic-K router |
| World model | Wired deterministic 1024-d latent dynamics; not the trained Python neural model | Neural dynamics used by MCTS and latent simulation; MPC-CEM caveat above |
| `simulate` / `what_if` / `audit_action` | Rust CLI, MCP and HTTP paths; distinct state/action contract | `GenZero` methods and `/v1/simulate`, `/v1/what_if`, `/v1/audit_action` |
| Training and trajectory extraction | no | world-model dynamics only (`scripts/extract_trajectories.py`, `scripts/train_world_model_dynamics.py`); every other training pipeline lives in `gen-zero-research` |
| Safety gate | four-tier `PolicyGate`, semantic risk via bridge | same tiers, plus CP-SAT pre-filter |
| Audit ledger | Appended for ask and pipeline decisions; optional durable checkpoints | no |

The Rust service exposes MCP, authenticated HTTP, decisions, simulation and ledger operations.
Its default dynamics are deterministic and illustrative. Python hosts the trained residual
neural dynamics model; porting that checkpoint into Rust remains separate work.

---

## Quickstart: Python

Prerequisites: Python 3.10+, CPU is enough.

```bash
git clone https://github.com/xmond/gen-zero.git
cd gen-zero/python
pip install -e .
```

Dict states use the grid-world schema `{"size": N, "body": [[x, y], ...]}` with actions `north`, `south`,
`east`, `west`. The head of the body is the agent. Grid simulation uses this schema;
`is_visual` dicts use the text heuristic, and unsupported dicts raise `ValueError`.

```python
from gen_zero import GenZero

# No dual-head checkpoint is shipped; inspect degraded metadata before using decisions.
gz = GenZero()
state = {"size": 5, "body": [[2, 2], [2, 1]]}
candidates = ["north", "east", "south"]

d = gz.decide(state, candidates)
print(d["action"], d["confidence"], d["experts_activated"])

s = gz.simulate(state, ["north", "east"])
print(s["survival_horizon"], s["first_hazard_step"], s["provenance"])

w = gz.what_if(state, candidates, horizon=3)
print(w["best_candidate"], w["traps_detected"], w["provenance"])

a = gz.audit_action(state, "south", horizon=3)
print(a["verdict"], round(a["risk_score"], 3), a["provenance"])
```

Output (dict states always go to the symbolic grid simulator, checkpoint or not):

```text
north 0.7697 ['world_model']
2 None ['symbolic_grid_world']
north [] ['symbolic_grid_world']
REJECT_LETHAL 1.0 ['symbolic_grid_world']
```

Two things worth knowing from that run. `task_hint` must be `None` or an exact paradigm name from
`UniversalParadigmRouter.all_paradigms` (`reflex`, `mcts`, `astar`, `bidirectional`, `world_model`,
`mpc_cem`, `gflownet`, `cfr`, `cp_sat`); `task_hint="maze"` is not one of these and `decide` raises
`ValueError` for it. Without a `task_hint`, the default `auto` mode falls back to a fixed default
paradigm-prior order (`world_model` first) plus structural boosts that do not fire here — this dict
state has no `position`/`goal` keys (needed for the A* boost) and 3 candidates skip the low-complexity
reflex boost (needs 2 or fewer). So it routes to `world_model`, and picks `north` rather than
`south`, matching the `['world_model']` provenance above. That expert does not use the dual-head
network weights. It rolls each candidate forward through `GenZeroTextWorldModel.adaptive_rollout`
on the symbolic grid (transitions from `virtual_step`, lookahead actions from
`predict_consequences`, all hand-set heuristic constants, see
`python/gen_zero/world_model/text_world_model.py`) and returns the softmax of the discounted
rollout returns: here `north 4.408`, `east 2.508`, `south 2.508` at horizon 2, which gives the
`0.7697`. Because this expert never reads the dual-head weights, the result reports what really ran:
`scorer: heuristic_grid_rollout`, `scorer_expert: world_model`,
`confidence_kind: uncalibrated_normalized_rollout_score`, `degraded: False`, and, as a separate fact,
`weights_loaded_from_checkpoint: False`. The top-level `scorer` comes from the primary expert with or
without a checkpoint. Only a neural fast-thinking path (e.g. `mode="reflex"`, or `mcts` whose leaves are
scored by the reflex head) run without a checkpoint is relabelled `scorer: untrained_weights_fallback`
with `degraded: True`. If such an untrained neural expert is fused as a secondary behind a non-neural
primary, the primary keeps its label but the result still comes back `degraded: True` with
`degraded_reason` naming `untrained_weights_fallback:<expert>`. The `cp_sat` expert reports
`scorer: python_predicate_filter`: it checks the hard rules as Python predicates and does not call
OR-Tools. A dict state the grid simulator cannot parse makes `world_model` raise
`UnsupportedStateError` instead of returning uniform scores. `audit_action("south")` still correctly flags `south` as
`REJECT_LETHAL` regardless of which head `decide` used — that is the gap the audit path exists to
close. Separately, `decide(mode="mcts")` on this dict state resolves through an `eval_fn` backed by
the reflex head's scoring, not a random rollout; the `MctsEngine.plan: no eval_fn, reward_fn or
dynamics_model; leaf values are uniform random` warning only fires when `MctsEngine` is driven
directly, without any of `eval_fn`, `reward_fn`, or a mounted `dynamics_model`.

With a trained checkpoint mounted via `GENZERO_NEURAL_DYNAMICS_CHECKPOINT` (see below), the
neural path works for `simulate`, `what_if` and `audit_action` on the 64-dim latent states and 16-dim
action vectors from the trajectory file, and reports `provenance: ['neural_residual_dynamics']`.
`decide()` does **not** yet accept action vectors as candidates: it raises
`TypeError: unhashable type: 'numpy.ndarray'` (`python/gen_zero/client.py`). The MCTS engine itself
is unit-tested with the neural model (`python/gen_zero/tests/test_neural_dynamics.py`), but there is no
end-to-end `decide` call that reaches it today. There is no implicit or relative-path auto-mount: set the
absolute-path environment variable
`GENZERO_NEURAL_DYNAMICS_CHECKPOINT=/abs/path/to/world_model_dynamics_v1.pt` (or pass
`GenZeroConfig(neural_dynamics_checkpoint=...)` explicitly with an absolute path) before constructing
`GenZero`. A relative path raises `ValueError` at config load time (`python/gen_zero/config.py`), and
with the variable unset the checkpoint is simply `None` — no directory is probed.

HTTP operations have separate model requirements. `/v1/decisions` and
`/v1/decide_step` return 503 until `GENZERO_DUAL_HEAD_CHECKPOINT` names a valid
trained dual-head checkpoint. The Qwen semantic backbone and neural dynamics
checkpoint do not satisfy that requirement. Symbolic grid simulation does not
require those weights. Both endpoints' `usage.prompt_tokens` / `usage.total_tokens` (including `/v1/decisions` batch mode)
are computed from character-length heuristics (`len(text) // 4`,
`python/gen_zero/service/app.py:867,1045,1109`), not a real tokenizer count; the
response marks this with `usage.estimated: true` and an `estimate_method`
string. Do not bill or rate-limit against it.

From the repository root:

```bash
cd python
GENZERO_API_KEY=gz_test python3 -m gen_zero.cli semantic --port 8995 &
sleep 20
curl -s -X POST http://127.0.0.1:8995/v1/what_if \
  -H 'Authorization: Bearer gz_test' -H 'content-type: application/json' \
  -d '{"state":{"size":5,"body":[[2,2],[2,1]]},"candidates":["north","east","south"],"horizon":3}'
```

Endpoints: `POST /v1/decisions`, `/v1/decide_step`, `/v1/simulate`, `/v1/what_if`, `/v1/audit_action`,
`GET /health`. The same process serves the semantic bridge used by the Rust CLI, and for that it needs a
complete local Hugging Face snapshot of `Qwen/Qwen2.5-0.5B` under
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B` (override with `ZERO_MODEL_CACHE`). The runtime
never downloads (`python/gen_zero/causal/zero_runtime.py`); fetch it once with
`huggingface-cli download Qwen/Qwen2.5-0.5B` before starting the service.

---

## Quickstart: Rust CLI and MCP server

Prerequisites: [Rust 1.88+](https://rustup.rs/). CI checks and tests the workspace
on 1.88.0, and release builds use that toolchain. See the
[dependency resolution policy](docs/README.md#rust-compatibility).

```bash
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo build --release -p gen-zero-cli
./target/release/gen-zero --help
```

Subcommands include `serve`, `mcp`, `keygen`, `reflex`, `decide`, `simulate`,
`what-if`, `audit`, `audit-ledger`, `entail`, `fold`, and the reflex plugin
commands `reflex-bench`, `reflex-patch-create`, `reflex-patch-apply`,
`reflex-patch-inspect`, `reflex-feedback-status`, `reflex-feedback-record`,
`reflex-feedback-prune`, and `reflex-adapt` (see [Reflex runtime](#reflex-runtime)).
Run `--help` on each subcommand for its input contract.

**Name collision with the Python CLI.** `python/pyproject.toml`'s `[project.scripts]` registers a
console script also named `gen-zero` (`python/setup.py` is a thin legacy `setup()` shim that reads the
same `pyproject.toml` metadata; it does not declare a separate entry point), with a completely different
subcommand set (`ask`, `route`, `imagine`, `stream`, `grep`, `compact`, `mcp`, `status`; see
[python/README.md](python/README.md)). Its `gen-zero mcp --transport stdio` is not the same command as
this Rust binary's `gen-zero serve --mode stdio`. If both are installed on the same `PATH`, whichever
resolves first silently shadows the other; neither warns you. Invoke the Rust binary by its built path
(`./target/release/gen-zero` or an explicit install path) rather than a bare `gen-zero` on a machine that
also has the Python package installed, or keep them in separate virtualenv/PATH scopes. `gen-zero-py` is
registered as an alias for the Python CLI but is not yet the default in its own docs, so the collision is
live today, not hypothetical. Resolving this (renaming one of the two, or removing the ambiguous alias) is
a real, un-scheduled follow-up; it is out of scope for this documentation pass and needs its own PR that
also updates `python/README.md`'s command table.

```bash
./target/release/gen-zero reflex \
  --context "Core temperature 98C in sector 4" \
  --candidates "vent_coolant,shutdown_reactor,ignore"
```

Without the Python bridge running, the answer is fail-closed:

```text
"is_error": true,
"text": "ConfirmationRequired: the safety of this request could not be assessed (semantic bridge transport error ...)"
"engine": "local_fast_reflex_fallback"   (candidates get uniform 1/N probability)
```

With the bridge up (`GENZERO_API_KEY=gz_test python3 -m gen_zero.cli semantic` on port 8995), the same
command scored the candidates (`vent_coolant`, confidence 0.869, `engine: semantic_bridge`) and then the
risk classifier rated the request p=0.70 dangerous, above the 0.45 escalate threshold, so the CLI still
returned `ConfirmationRequired` with exit code 1. That is the gate working, not a bug.

Serve as an MCP server:

```bash
./target/release/gen-zero keygen --prefix gz_live_
./target/release/gen-zero serve --mode stdio
GENZERO_API_KEY=gz_live_your_token ./target/release/gen-zero serve --mode sse --port 8999
```

Bridge settings: `GENZERO_PYTHON_ENDPOINT` (default `http://127.0.0.1:8995`, `off` disables),
`GENZERO_BRIDGE_REQUIRED=1` refuses to start without it. Do not run the scorer on the MCP port.
`ask`, `route`, `imagine` and the request risk check need a semantic backend. There are two;
the server picks exactly one at startup and never switches:

- **Native (no Python).** `--qwen-model-path <path>` or `GENZERO_QWEN_MODEL_PATH` runs Qwen2.5-0.5B in the
  Rust process on candle. The path is a GGUF file (e.g. `Qwen2.5-0.5B.Q8_0.gguf`, with
  `tokenizer.json` beside it or `--qwen-tokenizer`) or a Hugging Face directory with `config.json`,
  `model.safetensors` and `tokenizer.json`. A model that fails to load stops the server; it does not
  fall back to the Python bridge. GGUF matrices are dequantized to f32 at load, so RAM is about
  2.5 GB either way. `RAYON_NUM_THREADS` caps the compute threads (on a shared 24-core host, 4 was
  faster than 24). Try it without a server:
  `gen-zero qwen --model <path> risk --text "run rm -rf /"`.
- **Python bridge.** Without a Qwen path, the CLI calls the Python scorer at
  `GENZERO_PYTHON_ENDPOINT` (default `http://127.0.0.1:8995`). Start it with
  `GENZERO_API_KEY=<token> python3 -m gen_zero.cli semantic`.

Both backends implement the same prompts, demonstrations and thresholds. Against the Python fp32
scorer, the native safetensors path differs by at most 0.00004 in `p_dangerous` over the 91
calibration and held-out risk rows, and the Q8_0 GGUF path by at most 0.015 (one safe held-out row
crosses into HardStop). Commands and numbers: `crates/gen-zero-model/tests/qwen_native_parity.rs`.

Without a backend, decisions fail closed with `ConfirmationRequired`. `GENZERO_BRIDGE_REQUIRED=1`
makes the server refuse to start without one. A working backend does not remove every 428: the risk
classifier escalates about half of ordinary requests (`python/gen_zero/service/risk_data/README.md`),
and the gate escalates any decision whose normalized entropy is 0.65 or more.

The Rust HTTP server exposes these operational endpoints:

- `/health`: immediate 200 liveness response, independent of dependencies.
- `/ready` and `/healthz`: readiness, returning 200 only when the semantic bridge is reachable
  and the default tenant/workspace has valid cognitive assets; otherwise 503 with component
  status. Publish assets or start with `--mount-assets` before expecting readiness. Bridge
  results are cached for 10 seconds; an uncached network probe has a 3-second timeout.
- `/metrics`: Prometheus text exposition, using the configured API token when authentication
  is enabled. `genzero_requests_total{verb,status}` counts completed engine calls (including
  stdio), `genzero_gate_tier_total{tier}` counts reported gate verdicts, and
  `genzero_request_duration_seconds` is an engine-call latency histogram. HTTP response
  counts, including auth failures, are in `genzero_http_requests_total{method,status}`.
  Metrics are process-local and reset on restart. Probe endpoints require no token.

SIGINT/Ctrl-C and Unix SIGTERM stop accepting HTTP connections and drain in-flight requests.

Set `GENZERO_MMR_PERSIST_PATH=/var/lib/gen-zero/audit.json` to checkpoint the audit ledger
on each append and restore it on startup. Create the parent directory first and use durable
storage. Library hosts can set `ZeroEngineConfig::mmr_persist_path` and call
`PolymorphicZeroEngine::try_from_config`. Without a path, the ledger remains in memory.
Checkpoints preserve the hash key, full-history root/count, and the most recent 4096 leaves
and their proof material; older pruned leaves are not an archival log. A successful durable
append requires a synced atomic snapshot replacement. Corrupt checkpoints and I/O errors
are surfaced instead of resetting history. Only one engine may own a checkpoint path at a
time. The snapshot contains the audit hash key and is created with mode 0600 on Unix.

### Integrated Rust subsystems

The `pipeline` tool and `POST /v1/pipeline/decide` accept
`mode: "manifold_gflownet"` or `mode: "cfr_nash"` in addition to the existing
planner modes. Both use the same feasibility filtering and world-model error
handling as the other engines; these remain the one-step approximations described
above.

Load multiple operator Nanocores with a platform-separated path list, for example
`GENZERO_NANOCORE_PATHS=/models/vision.json:/models/dynamics.json` on Linux.
This takes precedence over the backward-compatible `GENZERO_NANOCORE_PATH`.
Each file must have a distinct domain ID and a non-empty `action_vocab` of unique
candidate names (with `out_dim >= len(action_vocab) - 1`); files without it fail to
load. Each output channel is bound to one vocabulary name, so a candidate outside the
vocabulary refuses the request, and candidate order or pruning never changes a
surviving candidate's score. On a text `ask`, select the mixture with
`"nanocore_domains": [1, 2]` and a `nanocore_state` containing 128 finite numbers.
The original `nanocore_domain` selects one core; do not supply both selectors.
Missing cores or invalid selectors refuse the request instead of dropping experts.

Rust hosts can call `PolymorphicZeroEngine::capture_mount_snapshot(id, &key)`
and `rollback_mount_snapshot(id, &key, observed_version)` from their trusted
administration layer. The storage crate compresses and authenticates saved
cognitive assets; rollback validates and publishes them as a new mount generation
using compare-and-swap. The engine retains at most 16 snapshots in memory.
Snapshots do not survive restart or rewind the audit ledger, graph/atlas, request
state, or Nanocore registrations. These methods do not add an HTTP admin endpoint.

### MCP client configuration

#### Hosted public cloud preview (one-click connect without building locally)

```bash
# Codex CLI
codex mcp add gen-zero --url "https://api.gen-zero.ai/sse?token=gz_public_free"

# Google Antigravity CLI
agy mcp add gen-zero "https://api.gen-zero.ai/sse?token=gz_public_free"
```

#### Local build configuration

Desktop or IDE MCP host (e.g., Cursor `mcp.json`), stdio mode:

```json
{
  "mcpServers": {
    "gen-zero": {
      "command": "/usr/local/bin/gen-zero",
      "args": ["serve", "--mode", "stdio"]
    }
  }
}
```

SSE mode with a token:

```json
{
  "mcpServers": {
    "gen-zero": {
      "url": "http://127.0.0.1:8999/sse",
      "headers": { "Authorization": "Bearer gz_live_your_token_here" }
    }
  }
}
```

The server registers `zero` and `pipeline` tools. `zero` accepts `ask`, `route`,
`imagine`, `stream`, `grep`, `compact`, `entail`, `causal_fold`, `pipeline`,
`simulate`, `what_if`, and `audit`. `ask`, `route` and `imagine` need
the bridge; without it `ask` escalates, `route` returns `isError` with `degraded: true`, and `imagine`
returns no plan.

---



## Open core

| Capability | Community (this repository) | Enterprise / commercial offering |
| --- | --- | --- |
| Rust engine, CLI, stdio and SSE MCP | Apache 2.0 implementation | Included |
| Planning, simulation, gate, provenance | Public implementation | Deployment and integration options may vary |
| Local 9B feature-adapter smoke test | Included (`examples/run_9b_demo.py`, 6.4MB adapter) | Included |
| Dense multi-model manifold clusters (405B / 180B / 123B / 72B) | Private research assets and extraction pipelines not shipped | Contact maintainers for availability |
| Reflex plugin runtime: inference, patch apply, hot-swap, feedback store, head-local adapt | Included | Included |
| Reflex plugin training and patch compilation | Not included | `gen-zero-research` / tuning API |

The private `gen-zero-research` repository contains extraction, offline datasets, and training loops. This public repository does not include those pipelines or model weights. The table describes repository boundaries, not a guarantee that a commercial edition or particular cluster is currently available.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [Apache 2.0 license](LICENSE). Issues and pull requests belong at [xmond/gen-zero](https://github.com/xmond/gen-zero).


## Two decision pathways

Gen-Zero scores a decision through one of two unrelated pathways, chosen by which
field a request sets. They share a safety gate; text requests also create a deterministic manifold observation when cognitive coordinates are absent. External reviews that describe "gen-zero" as a 0.5B language model
picking actions by lexical co-occurrence are describing pathway 1 only; that
description is accurate for pathway 1 and does not apply to pathway 2.

### 1. Semantic prior pathway: text-level candidate scoring

`zero ask` (`decide` is an alias of `ask`, `crates/gen-zero-service/src/zero.rs:137`),
`route`, and `imagine` rank discrete textual candidates with the Qwen2.5-0.5B
foundation model described above, run either natively via candle or through the
Python bridge; the two backends agree within the parity bounds given earlier
(`crates/gen-zero-model/tests/qwen_native_parity.rs`). `imagine` also uses this
model's output as the prior for a Monte Carlo tree search over action sequences
(`crates/gen-zero-service/README.md:17-19`), so it is a hybrid of this pathway
and planning, not purely either one.

The scoring rule is domain-conditional PMI (`crates/gen-zero-model/src/semantic_qwen.rs:10`,
citing Holtzman et al. 2021): for each candidate it computes the log-odds the
model assigns to that continuation against an empty-context baseline. That is a
measure of text continuity, not of outcome quality. Qwen2.5-0.5B here is an
unaligned base foundation model: nothing in this pipeline fine-tunes it on
decisions or applies RLHF. Raw PMI rewards the candidate that reads as natural
language after the given context, regardless of whether the action it names is
sensible. A fluent but unwise candidate (an external review's example: drinking
expired milk) can out-score a safe one, because lexical co-occurrence has no
notion of consequence. That is a property of the method, not a bug in its
implementation; removing it needs either decision-specific fine-tuning or RLHF
of the backbone (neither exists in this repository today) or moving the
decision to pathway 2, which does not score text at all.

Because this pathway cannot recognize danger on its own, every `ask`/`route`/`imagine`
call first runs the request through a risk classifier that fails closed:

- **Escalate** (HTTP 428, human confirmation required) at
  `p_dangerous >= 0.4494` (`RISK_ESCALATE_THRESHOLD`,
  `crates/gen-zero-model/src/semantic_qwen.rs:44`).
- **Hard stop** (refuse outright, no confirmation offered) at
  `p_dangerous >= 0.7620` (`RISK_HARD_STOP_THRESHOLD`, same file, line 45).
- Escalate also fires whenever the policy gate's normalized decision entropy is
  `>= 0.65` (`crates/gen-zero-gate/src/policy.rs`).

These thresholds are deliberately strict, and the cost is visible:
`python/gen_zero/service/risk_data/README.md` records 9 of 18 ordinary
held-out requests escalating under them, a measured false-escalation rate
near one in two. The same file calls this out directly ("False escalations
are frequent") rather than hiding it. That rate is the intended trade-off,
not a defect: at this calibration, `rm -rf /` scores only 0.02 above the
escalate line, so a threshold loose enough to stop annoying every routine
query would also let destructive commands through. A fail-closed
gate that is this cautious will ask a human to confirm routine work; a gate
tuned to stop asking would also sometimes stop refusing. The project chose
the side that fails safe. Loosening the threshold is a one-line change
(`RISK_ESCALATE_THRESHOLD`) that any operator can make; it is not the
default because the held-out evidence for a looser value is not in this
repository.

### 2. Continuous cognitive manifold and symplectic world-model pathway

The `cognitive` field on `zero ask`/`zero stream`, and the separate `pipeline`
verb (`pipeline.simulate`/`what_if`/`audit_action`/`decide`,
`crates/gen-zero-service/src/pipeline_verb.rs`), take an explicit numeric state
vector, not natural-language text. `pipeline`'s A* goal, for example, requires
`{"state": [/* exactly 1024 finite numbers */], "tolerance": ...}`
(`crates/gen-zero-service/README.md:72-76`). The numeric pathway does not compute PMI. Text requests without explicit cognitive coordinates also project into a deterministic 128-dimensional manifold observation; this projection is a hash-based mapping, not a learned semantic representation.

What runs on that state vector:

- **Geometry and a certified action verifier** (`crates/gen-zero-service/src/cognitive.rs`,
  Spec 25 §1.2/§5.1/§5.5): the state is checked against a Poincare ball,
  mapped to the tangent space at the origin, advanced through a parallel
  tangent-space SSM scan, and relaxed by a sheaf-cohomology gate kernel. The
  gate accepts a candidate only if its geodesic energy to the goal strictly
  falls (`upper(after) < lower(before)`), and only then emits a
  `CertifiedAction` with interval bounds — a real, checkable certificate on
  the chosen geometry, not a probability estimate. The file is explicit that
  this geometry is fixed, not learned: "No trained atlas or projection"
  (`crates/gen-zero-service/src/cognitive.rs:46`). The certificate says the
  candidate strictly reduces distance to the goal under this fixed metric; it
  does not say the metric represents anything about the real world unless the
  caller's state vector does.
- **World-model dynamics** (`gen-zero-worldmodel`): Contact Hamiltonian
  integrators with Strang splitting, Stormer-Verlet symplectic integrators,
  and Koopman spectral jump operators for O(1) lookahead
  (`crates/gen-zero-worldmodel/README.md:3-5`). These give real, provable
  properties of the integrator itself: the symplectic integrator preserves
  phase volume exactly; the contact integrator contracts phase volume at a
  known rate `exp(-2 gamma dt)` and reduces to the symplectic case at
  `gamma = 0`; the Koopman jump operator keeps spectral radius <= 1.0 by
  construction. Those guarantees hold for whatever dynamics function is
  plugged in. The dynamics shipped as the service default,
  `LatentDynamicsWorldModel`, is a fixed illustrative residual update: each
  dimension decays toward zero at `residual_scale = 0.95` per step and is
  nudged by a sine-of-action perturbation scaled by `action_scale = 0.05`
  (`crates/gen-zero-worldmodel/src/dynamics.rs`), not a model trained on any real environment (`docs/zero/README.md`'s own
  audit section says the same). The integrator's math is sound; whether the
  trajectory means anything physically depends on supplying a dynamics model
  or state encoding that actually represents your domain, which this
  repository does not ship.
- **Planners** (`gen-zero-planner`): MCTS (PUCT), uncertainty-penalized A*,
  and MPC/Cross-Entropy-Method search over the state vector and the chosen
  dynamics, routed by entropy through a K-MoE router. These engines carry
  their own documented limits: MCTS "has no learned leaf value or
  finite-budget optimality guarantee"; the gate verdict "certifies policy
  admissibility at evaluation time, not calibrated physical safety or
  optimality" (`crates/gen-zero-planner/README.md:53-56,112-113`). Read those
  caveats before treating a planner's output as a correctness proof.

### Getting real-world decision performance out of either pathway

Neither pathway above is "finished" without the caller doing one of the
following. Each changes what the engine is actually scoring, not just a
setting:

**a. Supply real structured state.** The geometric pathway only reasons about
the state vector it is given. A caller that encodes real sensor, business, or
environment state into the `cognitive` or `pipeline` state vectors gets
geometric guarantees over that real state. A caller that passes the service
default illustrative dynamics gets a certified result about a toy dynamics
model, not about their domain.

**b. Use the closed-loop tuning pipeline to adapt reflex weights.** The
reflex tuning daemon (`services/tuning/tuning_service.py`) retrains the fast
reflex path from operator feedback: a CUDA online learner updates LoRA
weights under a contraction-loss penalty, and a Golden gate only publishes a
patch if accuracy does not regress and every measured contraction ratio stays
below 1. Real business accuracy and long-running production behavior remain unverified. The Golden gate's loss is an empirical bound on the measured set, not a global Lipschitz certificate. Treat it as an available feedback loop to tune with your own data, not
as a source of out-of-the-box accuracy.
