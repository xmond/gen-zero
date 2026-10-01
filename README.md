# Gen-Zero

[![Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Rust 1.88+](https://img.shields.io/badge/Rust-1.88%2B-orange.svg)](Cargo.toml) [![MCP](https://img.shields.io/badge/MCP-stdio%20%7C%20SSE-purple.svg)](crates/gen-zero-service/README.md) [![Latency](https://img.shields.io/badge/latency-subsecond%20target-green.svg)](benchmarks/README.md) [![Live Demo](https://img.shields.io/badge/demo-gen--zero.ai-informational.svg)](https://gen-zero.ai) [![Research Papers](https://img.shields.io/badge/research-5%20papers-lightgrey.svg)](https://gen-zero.ai)

**Gen-Zero is a co-pilot and pre-execution decision gate for LLM agents.** Given a set of candidate actions, it scores them, simulates a few steps ahead, and runs the result through a four-tier safety gate — all outside your agent's own reasoning loop. It mounts over MCP next to Claude Code, Cursor, Codex, LangChain, or any other MCP host as a tool, not a replacement for them. The production server binary has zero Python dependency when run with its native Qwen backend; an external Python scorer is an opt-in alternative, not a requirement.

> **[Try it in your browser, no install →](https://gen-zero.ai)** Run the interactive simulation and browse the five research papers behind it.

## Key Highlights

- **Zero-token candidate scoring.** Decisions come back as a probability distribution over named candidates, not generated prose — zero output tokens. The semantic pathway still runs one Qwen2.5-0.5B forward pass to produce that distribution (see [Dual Pathways](#two-decision-pathways)); it is the generation step that is eliminated, not the model compute.
- **Pre-execution fail-closed gate.** Four tiers — Proceed, Confirm, Escalate, HardStop. A missing backend, invalid assets, high decision entropy (≥ 0.65), or a high risk score return an explicit error or confirmation request, never a guessed answer. The default policy ships with empty rules; you configure the constraints for your domain.
- **Latent lookahead world-model simulation.** Candidate actions roll forward on a 1024-dim latent state through symplectic (Störmer–Verlet) and contact-Hamiltonian integrators, reporting cumulative return and first hazard step. The integrators' conservation and contraction properties are proven; the dynamics function shipped as the service default is an illustrative sine-perturbed decay, not a trained world model. Supply your own dynamics or state encoding for results that mean something about your domain.
- **Drop-in MCP native.** One ~24 MB binary, dynamically linked only against `libc`/`libm`/`libgcc_s`, serves MCP over stdio or SSE/HTTP and registers three tools: `zero`, `causal_fold`, `pipeline`. The reflex primitive in the benchmarks below runs today only through CLI commands (`reflex-bench`, `reflex-adapt`); it is not yet wired into `serve`.

## Quickstart

### Option A: Connect via MCP

<a id="quickstart-rust-cli-and-mcp-server"></a>
Hosted public preview, no local build:

```bash
codex mcp add gen-zero --url "https://api.gen-zero.ai/sse?token=gz_public_free"
agy mcp add gen-zero "https://api.gen-zero.ai/sse?token=gz_public_free"
```

Local build, stdio (Claude Code, Cursor, any desktop MCP host):

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

SSE with a bearer token:

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

### Option B: Run locally via Binary or Docker

```bash
git clone https://github.com/xmond/gen-zero.git && cd gen-zero
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo build --release -p gen-zero-cli
./target/release/gen-zero keygen --prefix gz_live_
./target/release/gen-zero serve --mode stdio
GENZERO_API_KEY=gz_live_your_token ./target/release/gen-zero serve --mode sse --port 8999
```

```bash
docker build -t gen-zero .
docker run -p 8999:8999 -e GENZERO_API_KEY=gz_live_your_token gen-zero
```

Without `--qwen-model-path`/`GENZERO_QWEN_MODEL_PATH`, `ask`/`route`/`imagine` call an external Python scorer at `GENZERO_PYTHON_ENDPOINT` instead (`GENZERO_BRIDGE_REQUIRED=1` refuses to start without either). Either way, a missing or failed backend returns `ConfirmationRequired`, not a guess:

```bash
./target/release/gen-zero reflex \
  --context "Core temperature 98C in sector 4" \
  --candidates "vent_coolant,shutdown_reactor,ignore"
```

```text
"is_error": true,
"text": "ConfirmationRequired: the safety of this request could not be assessed (semantic bridge transport error ...)"
"engine": "local_fast_reflex_fallback"   (candidates get uniform 1/N probability)
```

With a backend up, the same command can still return `ConfirmationRequired` if the risk classifier rates the request dangerous — that is the gate working, not a bug.

### Option C: Python Client SDK

<a id="quickstart-python"></a>

```bash
pip install -e ./python
```

```python
from gen_zero import GenZero

gz = GenZero()
print(gz.decide({"size": 5, "body": [[2, 2], [2, 1]]}, ["north", "east", "south"]))
```

No dual-head checkpoint ships with this package; check the response's `degraded`/`scorer` fields before trusting a decision. See [python/README.md](python/README.md) for the full client contract, HTTP endpoints, and routing rules.

## Architecture & Dual Pathways

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

<a id="two-decision-pathways"></a>
Gen-Zero scores a decision through one of two unrelated pathways, chosen by which field a request sets. They share the same safety gate.

**1. Semantic prior pathway — text, ~1.0–2.5 s.** `ask` (alias `decide`), `route`, and `imagine` rank textual candidates with Qwen2.5-0.5B, run natively via candle or through the Python bridge (the two backends agree within 0.00004 on safetensors, 0.015 on Q8_0 GGUF). The scoring rule is domain-conditional PMI — a measure of text continuity, not outcome quality, so a fluent-but-unwise candidate can outscore a safe one. Every call therefore first runs a risk classifier that escalates at `p_dangerous ≥ 0.4494` and hard-stops at `≥ 0.7620`; on a 36-request held-out set this let zero dangerous requests through but escalated 9 of 18 safe ones to a human — a deliberate fail-safe trade-off, not a defect.

**2. Continuous cognitive manifold & symplectic world-model pathway — numeric state, 164 µs reflex primitive to ~5.8 ms MCTS lookahead.** The `cognitive` field and the `pipeline` verb take an explicit numeric state vector, not text. A certified action verifier checks candidates against a Poincaré-ball geometry and only accepts one whose geodesic energy to the goal strictly falls. Six planning engines (MCTS, A*, MPC/CEM, and three one-step approximations) search over this state using the integrators described above. The geometry is explicitly fixed, not learned ("no trained atlas or projection"); its certificate says a candidate reduces distance under this metric, not that the metric represents anything about the real world unless your state vector does.

<a id="integrated-rust-subsystems"></a>
The `pipeline` tool and `POST /v1/pipeline/decide` also accept `manifold_gflownet` and `cfr_nash` modes, and multiple Nanocore operators can be mounted with `GENZERO_NANOCORE_PATHS`. See [crates/gen-zero-service](crates/gen-zero-service/README.md) for the full request contract, readiness/metrics endpoints, and audit-ledger persistence.

> **Integration gotcha:** a fail-closed response still fills `meta.best_action`, even when every candidate tied (e.g., three equal 1/3 probabilities). Always check `is_error` before reading `meta` — a populated `best_action` on an error response is not a decision.

### Where this fits

Gen-Zero does not generate candidates and does not manage your agent's control flow — it only scores, simulates, and gates. The recommended shape is "host proposes, Gen-Zero reviews, executor acts":

```text
Host agent (LangGraph / Claude Code / CrewAI / ...)
   │  1. LLM generates 3-5 candidate actions
   ▼
Gen-Zero (MCP: zero.ask / zero.what_if / pipeline.decide)
   │  2. score + lookahead + four-tier gate
   │     Proceed → continue; Confirm/Escalate → is_error + ConfirmationRequired; HardStop → refuse
   ▼
Host executor (sandbox / human confirmation)
```

It is a layer, not a framework: orchestration tools like LangChain or CrewAI still own the loop, memory, and tool calls. Gen-Zero only owns the answer to "should this one step proceed."

## Research Foundations

The ideas above are documented in a five-paper series on the project site. These are research write-ups, not a description of what ships in this repository today — read the sections above and [`benchmarks/README.md`](benchmarks/README.md) for what is actually implemented and measured.

| # | Paper | Topic |
| :-: | --- | --- |
| 1 | [Zero-Token Decision](https://gen-zero.ai/papers/Paper1_Zero_Token_Decision_Interactive.html) | Scoring candidates as a distribution instead of generating text |
| 2 | [Latent World Model Simulation](https://gen-zero.ai/papers/Paper2_Latent_World_Model_Interactive.html) | Symplectic and contact-Hamiltonian lookahead on latent state |
| 3 | [Cross-Model Manifold Alignment](https://gen-zero.ai/papers/Paper3_Cross_Model_Alignment_Interactive.html) | Aligning representations across different backbone models |
| 4 | [Equivariant Choice Head](https://gen-zero.ai/papers/Paper4_Equivariant_Choice_Head_Interactive.html) | Candidate-order invariance in the scoring head |
| 5 | [Semantic Risk Gating](https://gen-zero.ai/papers/Paper5_Semantic_Risk_Gating_Interactive.html) | The fail-closed risk classifier behind the gate |

## Verified Benchmarks & Performance

| Mechanism | Latency | Source and caveat |
| --- | ---: | --- |
| Reflex memory primitive, predict | p50 164 µs / p99 238 µs | CLI `reflex-bench`, untrained random-weight plugin (shape only, not a learned decision). **Not wired into `serve`/MCP.** |
| Pipeline MCTS lookahead, stdio, horizon 8 | p50 5.8 ms / p99 11.0 ms | Launch evidence, 2026-10-01, `pipeline decide[mode=mcts]` over stdio, n=200. Distinct from the privileged-oracle ablation below — don't conflate the two. |
| Semantic text scoring, Qwen2.5-0.5B native | ~1.1–2.4 s per call | Launch evidence, 2026-10-01, hosted `native_qwen` bridge timing on two sample requests; not a formal benchmark sweep. |

**Neural world model, held-out validation** (`benchmarks/results/world_model_training_report.json`, 8,146 transitions from 1,620 episodes, split by episode, state dim 64, 16 actions):

| Metric | Value | Baseline in the same report |
| :--- | ---: | :--- |
| Val MSE, per state dimension | **0.015217** | identity transition: 0.19541 |
| Val AUC, safe-or-goal head | **0.99993** | constant-reward BCE: 0.3855 |
| Val reward accuracy | 0.99704 | — |

This shows the trained model predicts the next latent state about 13x better than "nothing changes" on held-out episodes from the same generator. It shows nothing about other environments or about planning quality, and this checkpoint has not yet been run through the MCTS ablation below.

<a id="instant-decision-benchmark-cpu-only-local-reproduction"></a>
**Semantic risk gate, 36-request held-out set** (`python/gen_zero/service/risk_data/README.md`, AUC 0.944): 0 of 18 dangerous requests proceeded, 0 of 18 safe requests hard-stopped, but 9 of 18 safe requests escalated to a human. A known miss, covered by a strict xfail test: bare `chmod -R 777 /` scores the same as `git status` and proceeds.

**Trap avoidance, MCTS vs. greedy, paired** (`benchmarks/results/world_model_mcts_ablation_report.{json,md}`, 100 seeded torus episodes): greedy baseline 0% success / 100% trapped; MCTS with the environment's own exact transition function 100% success / 0% trapped (McNemar p ≈ 1.6e-30). The "world model" here is a privileged oracle, not the trained neural network — this is a planning diagnostic, not evidence of neural world-model or production-planner performance.

Reproduce the torus ablation yourself, CPU-only, no checkpoint downloads, from the repository root:

```bash
python3 benchmarks/suites/benchmark_world_model_mcts_ablation.py --episodes 10
```

```text
Method                             | Success % | Trapped % | Mean Steps | Latency ms
------------------------------------------------------------------------------------
Greedy baseline (no model)         |      0.00 |    100.00 |       1.00 |      0.010
Gen-Zero MCTS (exact-graph oracle) |    100.00 |      0.00 |       4.60 |      0.544
McNemar exact (paired): baseline-only=0, world_model-only=10, p=0.00195
CAVEAT: "world model" here is env.step()'s privileged exact graph dynamics, not the
trained neural network. This is a planning diagnostic, not evidence of neural
world-model or production-planner performance.
```

**Not measured, not claimed.** Earlier drafts of this README quoted a 0.85 ms planning SLA, a 450 MB memory ceiling, node-allocation/ledger throughput figures, and "100.0% zero breach" / "infinite lookahead" language. No artifact in this repository measures those end to end; they have been retracted, not merely reworded. Treat latency and memory figures here as measured on the stated machine, not as portable guarantees.

Full methodology, additional ablations, and the neural world-model validation numbers: [`benchmarks/README.md`](benchmarks/README.md).

## Ecosystem, Documentation & Contributing

| | This repository (`gen-zero`) | `gen-zero-research` and the tuning API (`tuning.gen-zero.ai`) |
| :--- | :--- | :--- |
| Scope | Rust runtime: reflex inference, patch hot-swap, planning, world-model simulation, safety gate, audit ledger, MCP/HTTP/CLI | Offline continuous learning: teacher training, curriculum, distillation, self-play |
| Client | Python SDK (`python/gen_zero`) | Compiling trained checkpoints into reflex plugins and patches |

A reflex plugin or patch is produced upstream and consumed here; this repository ships no training loop for reflex operators.

**Name collision:** the Python package also registers a console script named `gen-zero`, with a completely different subcommand set. If both are installed on the same `PATH`, whichever resolves first silently shadows the other. Invoke the Rust binary by its built path, or keep the two in separate environments. Details: [python/README.md](python/README.md).

### Open core

| Capability | This repository | Enterprise / commercial offering |
| --- | --- | --- |
| Rust engine, CLI, stdio and SSE MCP | Apache 2.0, full implementation | Included |
| Planning, simulation, gate, provenance | Public implementation | Deployment and integration options may vary |
| Local 9B feature-adapter smoke test | Included (`examples/run_9b_demo.py`) | Included |
| Reflex plugin runtime: inference, patch apply, hot-swap, feedback store | Included | Included |
| Reflex plugin training and patch compilation, dense multi-model manifold clusters | Not included — private research assets and pipelines | `gen-zero-research` / tuning API |

This table describes repository boundaries, not a guarantee that a commercial edition is currently available for every row.

### Documentation map

- [`docs/architecture/`](docs/architecture/) — subsystem audits and design notes, including known gaps.
- [`docs/research/`](docs/research/) — experiment reports behind the research papers above.
- [`docs/manuals/`](docs/manuals/) — operational playbooks (e.g., mounting a production Nanocore).
- [`crates/*/README.md`](crates/) — per-crate contracts for the service, planner, world model, and gate.
- [`examples/run_9b_demo.py`](examples/run_9b_demo.py) — numeric soundness and Lyapunov-stability check for a packaged feature adapter; requires only `numpy`, not a GPU or the full model.
- [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [LICENSE](LICENSE) (Apache 2.0).

Issues and pull requests belong at [xmond/gen-zero](https://github.com/xmond/gen-zero).
