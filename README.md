# Gen-Zero

[![Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![Rust 1.88+](https://img.shields.io/badge/Rust-1.88%2B-orange.svg)](Cargo.toml) [![MCP](https://img.shields.io/badge/MCP-stdio%20%7C%20SSE%20%7C%20UDS-purple.svg)](crates/gen-zero-service/README.md) [![Pure Inference](https://img.shields.io/badge/inference-zero--token%20CAD-green.svg)](python/gen_zero/causal/README.md) [![Live Demo](https://img.shields.io/badge/demo-gen--zero.ai-informational.svg)](https://gen-zero.ai) [![Research Papers](https://img.shields.io/badge/research-5%20papers-lightgrey.svg)](https://gen-zero.ai)

**Gen-Zero is a high-performance, deterministic cognitive decision runtime and causal pure-inference engine for AI agents.** Given a set of candidate actions, it scores them via Contrastive Decoding (CAD), simulates forward transitions through geometric symplectic lookahead, and executes a four-tier fail-closed safety gate — all completely outside your agent's autoregressive generation loop.

It mounts natively over the **Model Context Protocol (MCP)** alongside Claude Code, Cursor, Codex, LangChain, or any custom agent framework as a tool, not a replacement.

> **[Try it in your browser, no install →](https://gen-zero.ai)** Run interactive simulations and read the five research papers behind the architecture.

---

## 🏛️ Open Source Architecture & Repository Boundaries

Gen-Zero adheres to a strict physical separation between open-source runtime infrastructure and internal research assets:

* **This Repository (`https://github.com/xmond/gen-zero`) — Open Source Community Edition**:
  * **Pure-Inference Hybrid Engine**: High-throughput Contrastive Decoding (CAD), zero-copy Verbalizer logit extraction, and NUMA-aware multi-process GGUF pools based on open backbones (such as `Qwen2.5-1.5B-Instruct`).
  * **Rust Microservice Core**: Zero-heap preallocated MCTS arena, SIMD AVX-512 vectorization, and Unix Domain Socket (UDS) / SSE / stdio MCP servers.
  * **Developer Tooling**: Native Rust binary (`gen-zero`), Python CLI (`python -m gen_zero.cli`), and lightweight client SDKs.
  * **Zero Private Weights & Zero Training Loops**: Does not include internal training code, backward passes, loss optimizers, or proprietary model checkpoints.

* **Internal Research (`gen-zero-research`) — Proprietary Foundation**:
  * Home to the native 1B continuous S-DEQ foundation model, multi-teacher distillation pipelines (LLaMA-405B / Qwen-72B), curriculum optimizers, and cloud services powering the commercial endpoint `api.gen-zero.ai`.

---

## 🚀 Key Highlights

* **Zero-Token Candidate Scoring**: Decisions return as exact probability distributions over candidate actions rather than generated text tokens — eliminating autoregressive generation latency while running pure-inference forward passes.
* **Causal Contrastive Decoding (CAD)**: Employs counterfactual prior conditioning ($\delta = \text{logits}_{\text{cond}} - \alpha \cdot \text{logits}_{\text{prior}}$) to strip conversational bias and surface true causal grounding.
* **Four-Tier Fail-Closed Safety Gate**: Explicit escalation tiers (`Proceed`, `Confirm`, `Escalate`, `HardStop`). Missing models, high decision entropy, or ambiguous causal bounds fail closed safely, never hallucinating a guessed action.
* **Zero-Allocation MCTS Lookahead**: Preallocated scratch arenas in Rust eliminate runtime heap jitter, delivering sub-millisecond planning latency.
* **Multi-Transport MCP Native**: Serves MCP across `stdio`, `SSE / HTTP`, and Unix Domain Sockets (`UDS`).

---

## 🛠️ CLI Reference (Rust & Python)

Gen-Zero provides dual command-line interfaces: a compiled Rust binary for microsecond runtime services and a Python CLI for high-level CAD workflows.

### 1. Rust CLI (`gen-zero`)

```bash
# Build the native binary (Rust 1.88+ required)
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo build --release
./target/release/gen-zero --help
```

| Command | Usage | Description |
| :--- | :--- | :--- |
| `gen-zero mcp --stdio` | Desktop MCP client connection | Runs MCP server over standard input/output (for Claude Code, Cursor, Codex). |
| `gen-zero mcp --host 127.0.0.1 --port 8999` | Remote / Network MCP server | Runs HTTP / Server-Sent Events (SSE) server. Authenticate with `--token` or `GENZERO_API_KEY`. |
| `gen-zero mcp --uds-path /tmp/gen-zero.sock` | Local High-Speed IPC | Serves length-prefixed MCP JSON-RPC over a Unix Domain Socket with microsecond latency. |
| `gen-zero keygen --prefix gz_live_` | Security Key Generation | Generates a cryptographically secure token with standard prefix. |
| `gen-zero run [FLAGS]` | Direct Cognitive Execution | Directly executes a JSON scenario or candidate evaluation without an MCP host. |
| `gen-zero qa-gate [FLAGS]` | Two-Stage QA Gate | Runs confidence margin check and counterfactual verification. |

### 2. Python CLI (`python -m gen_zero.cli`)

```bash
# Install the Python package in editable mode
pip install -e ./python
# Optional: install llama-cpp-python (CPU prebuilt wheel, no C++ compilation needed)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 1. Download official open-source Qwen2.5-1.5B GGUF weights (~1.1GB, sha256 verified)
python -m gen_zero.scripts.setup_qwen15b_models # or add --source modelscope for fast mirror

# 2. Immediately run the verified end-to-end CAD demo
python examples/quickstart_qwen15b_cad.py

# 3. Or run custom Contrastive Decoding (CAD) on single queries
python -m gen_zero.cli cad \
  --gguf ./models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --question "Does drug X cause symptom Y?" \
  --context "Clinical trial observed symptom Y in placebo and drug X equally." \
  --alpha 0.5

# 4. High-throughput batch evaluation with NUMA node pinning
python -m gen_zero.cli cad \
  --gguf ./models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --input test_queries.jsonl \
  --workers 4 \
  --numa-pin \
  --json
```

---

## 🔌 Model Context Protocol (MCP) Server

Gen-Zero embeds an enterprise-grade MCP server supporting three standard transport protocols:

### Transport Modes

1. **`stdio` Mode (Recommended for Local Agents)**:
   Spawned directly as a child process by desktop agents. Cleanly terminates when the parent session exits.
2. **`SSE / HTTP` Mode (Recommended for Shared Servers & Containers)**:
   Runs as a persistent daemon. Requires `Authorization: Bearer <token>` when `GENZERO_API_KEY` is configured.
3. **`UDS` Mode (Recommended for High-Performance Local Workloads)**:
   Uses Unix Domain Sockets (`--uds-path /tmp/gen-zero.sock`) to eliminate TCP/IP overhead and network port collisions.

### Host Integration Examples

#### Claude Code / Cursor / Codex Configuration (`mcpServers`)

Add the following to your MCP settings file (e.g. `~/.config/claude-code/config.json` or `cursor.json`):

```json
{
  "mcpServers": {
    "gen-zero-local": {
      "command": "/usr/local/bin/gen-zero",
      "args": ["mcp", "--stdio"]
    },
    "gen-zero-remote": {
      "url": "http://127.0.0.1:8999/sse",
      "headers": {
        "Authorization": "Bearer gz_live_your_token_here"
      }
    }
  }
}
```

#### Codex / Antigravity CLI Quick Add

```bash
# Connect to a hosted or local SSE instance
codex mcp add gen-zero --url "http://127.0.0.1:8999/sse"
agy mcp add gen-zero "http://127.0.0.1:8999/sse"
```

### Registered Tools

| Tool Name | Action / Verbs | Description |
| :--- | :--- | :--- |
| `zero` | `ask`, `imagine`, `what_if`, `verify` | Performs typed zero-token candidate scoring, counterfactual tree lookahead, and fail-closed safety assessment. |
| `causal_fold` | `arbitrate`, `contrast` | Evaluates causal DAG constraints, eliminating spurious correlations via contrastive logit boundaries. |
| `pipeline` | `decide`, `mcts`, `symplectic` | Runs preallocated zero-heap MCTS and contact Hamiltonian symplectic forward simulations. |
| `qa_gate` | `evaluate`, `fast_pass` | Two-stage candidate release gate with ambiguity margin analysis. |

---

## 🌐 API Reference (Python SDK & REST Microservice)

Gen-Zero provides flexible interfaces across three distinct usage tiers:

### 1. Local Pure-Inference Python API (`CADEngine`)

Directly load GGUF weights locally for contrastive causal validation with zero external server dependencies:

```python
from gen_zero.causal.cad_engine import CADEngine

# 1. Load the official Qwen2.5-1.5B GGUF weights
engine = CADEngine.from_gguf(
    "models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    alpha=0.5,         # Contrastive decoding context-prior reduction (0.0 ~ 1.0)
    temperature=1.0,   # Temperature scaling
    n_ctx=2048,
)

# 2. Run causal contrastive classification
result = engine.classify(
    question="Does drug X cause symptom Y?",
    context="In a clinical trial, symptom Y occurred in 5% of placebo and 5% of drug X recipients."
)

print(f"Decision:      {result.label}")            # 'yes' / 'no' / 'maybe'
print(f"Probabilities: {result.probabilities}")    # {'yes': 0.08, 'no': 0.58, 'maybe': 0.34}
print(f"Logit Delta:   {result.delta}")
```

### 2. High-Throughput Parallel Pool with NUMA Affinity

For high-concurrency batch evaluation pipelines across physical CPU cores:

```python
from gen_zero.causal.gguf_parallel_pool import GGUFParallelPool

# Initialize worker pool pinned to dedicated physical CPU cores
pool = GGUFParallelPool(
    model_path="models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_workers=4,
    n_threads_per_worker=4,
    numa_pin=True,
)

prompts = [
    ("Context A...", "Question A..."),
    ("Context B...", "Question B..."),
]
results = pool.batch_score(prompts)
for res in results:
    print(res.best_candidate, res.margin)
```

### 3. HTTP / REST Microservice API (`POST /v1/decisions`)

Run Gen-Zero as a microservice container for cross-language (Node.js, Go, Rust, Web) integration:

```bash
# Start the local decision microservice
export GENZERO_API_KEY="gz_live_your_token_here"
uvicorn gen_zero.service.app:app --host 0.0.0.0 --port 8999
```

Invoke the decision gateway via standard JSON HTTP requests:

```bash
curl -X POST "http://localhost:8999/v1/decisions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer gz_live_your_token_here" \
  -d '{
    "model": "typesafe/zero-1.13",
    "state": {
      "task": "database_migration",
      "risk_level": "medium",
      "target_table": "users"
    },
    "questions": {
      "safety_check": {
        "type": "noul",
        "instructions": "Is it safe to drop table users directly in production?",
        "criteria": ["yes", "no"]
      },
      "recommended_action": {
        "type": "choice",
        "instructions": "Select safest mitigation action",
        "criteria": {
          "shadow_table": "Create shadow table and dual-write",
          "maintenance_window": "Schedule downtime maintenance window",
          "abort": "Abort migration and notify DBA"
        }
      }
    }
  }'
```

Response JSON:

```json
{
  "id": "dec_8f29ac01",
  "model": "typesafe/zero-1.13",
  "decisions": {
    "safety_check": {
      "type": "noul",
      "value": false,
      "probability": 0.0012,
      "abstain": false
    },
    "recommended_action": {
      "type": "choice",
      "best_criterion": "shadow_table",
      "probabilities": {
        "shadow_table": 0.84,
        "maintenance_window": 0.14,
        "abort": 0.02
      }
    }
  },
  "latency_ms": 14.8
}
```

### 4. Cloud Gateway Client (`GenZeroClient`)

Connect client agents to cloud-hosted endpoints (`api.gen-zero.ai`):

```python
from gen_zero import GenZeroClient

client = GenZeroClient(
    endpoint="https://api.gen-zero.ai",
    api_key="gz_live_your_token_here"
)

decision = client.decide(
    context={"task": "financial_audit", "amount": 1000000},
    candidates=["approve", "manual_review", "block"]
)

print(f"Approved action: {decision.best_action}")
```

---

## 🔬 Research Foundations

The algorithms and formal proofs behind Gen-Zero are documented in a series of five papers:

| # | Research Paper | Core Topic |
| :-: | :--- | :--- |
| 1 | [Zero-Token Decision](https://gen-zero.ai/papers/Paper1_Zero_Token_Decision_Interactive.html) | Direct candidate logit probability distribution without token generation. |
| 2 | [Latent World Model Simulation](https://gen-zero.ai/papers/Paper2_Latent_World_Model_Interactive.html) | Symplectic (Störmer–Verlet) and contact-Hamiltonian lookahead on latent manifolds. |
| 3 | [Cross-Model Manifold Alignment](https://gen-zero.ai/papers/Paper3_Cross_Model_Alignment_Interactive.html) | Geodesic projection and representation alignment across distinct backbones. |
| 4 | [Equivariant Choice Head](https://gen-zero.ai/papers/Paper4_Equivariant_Choice_Head_Interactive.html) | Permutation-invariant candidate heads that eliminate positional ordering bias. |
| 5 | [Semantic Risk Gating](https://gen-zero.ai/papers/Paper5_Semantic_Risk_Gating_Interactive.html) | Fail-closed decision boundaries and entropy-calibrated risk escalation. |

---

## 🔒 Security & Anti-Leakage Gate

To prevent accidental inclusion of proprietary model binaries, private keys, or training scripts into the public repository, Gen-Zero includes an automated static gatekeeper:

```bash
# Run the anti-leakage static security linter
python3 scripts/anti_leakage_lint.py --respect-gitignore

# Install the pre-push safety hook locally
python3 scripts/anti_leakage_lint.py --install-hook
```

This gate runs automatically on every pull request via GitHub Actions ([`.github/workflows/anti_leakage_gate.yml`](.github/workflows/anti_leakage_gate.yml)), enforcing zero leakage of non-public intellectual property.

---

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
