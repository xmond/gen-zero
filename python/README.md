# Gen-Zero Python SDK and research implementation

Gen-Zero provides candidate scoring, experimental planning engines and world-model
simulation. Capability and safety depend on the selected backend, model weights,
input schema and constraints. These interfaces do not establish universal optimality
or a general safety guarantee.

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e './python[dev]'
python examples/01_quickstart_decision.py
python examples/02_world_model_simulation.py
```

A default `GenZero()` has no trained dual-head checkpoint. Direct SDK decisions
report degraded metadata; HTTP decision endpoints refuse requests with 503 until
valid weights are configured. Grid simulation is symbolic. Neural latent simulation
requires its own trained dynamics checkpoint. See the [root quickstart](../README.md#quickstart-python)
and [examples](../examples/README.md) for input contracts and provenance.

Optional extras: `.[train]` (scikit-learn, pyarrow) for the training scripts, `.[vision]` (torch, torchvision) for the vision engine.

Or install using `requirements.txt` from the repository root:

```bash
pip install -r python/requirements.txt
```

---

## ⚡ Quickstart

### 1. System 1 Non-Autoregressive Decision (Fast Path)

Score a finite candidate set in one forward pass without generating output tokens.
This does not establish correctness, calibration or a latency guarantee:

```python
from gen_zero import GenZero, GenZeroConfig

# In-process engine. Point GENZERO_DUAL_HEAD_CHECKPOINT (or dual_head_checkpoint=)
# at a trained state_dict; without one the model is random-init and the result
# is marked degraded=True.
client = GenZero(GenZeroConfig())

# Evaluate candidate actions given environmental context
state = "Database memory pressure at 94%. Active connections 2,400/2,500. Read replication lag 450ms."
candidates = [
    "REJECT_NEW_SESSIONS",
    "SCALE_READ_REPLICAS",
    "KILL_IDLE_CONNECTIONS",
    "FLUSH_QUERY_CACHE",
]

result = client.decide(state=state, candidates=candidates, mode="fast")

print(f"Selected Action: {result['action']}")
print(f"Confidence     : {result['confidence']:.4f}")
print(f"Degraded       : {result['degraded']}")
print("Policy Dist    :", result["probs"])
```

### 2. System 2 Graph Planning (Uncertainty A*)

Plan a path through a graph with uncertainty-weighted edges. `get_neighbors` returns `(next_state, action_name, safety_prob)` tuples:

```python
from gen_zero import GenZero

client = GenZero()

goal = (3, 3)
walls = {(1, 1), (1, 2), (2, 1)}

def get_neighbors(pos):
    r, c = pos
    out = []
    for dr, dc, act in [(-1, 0, "north"), (1, 0, "south"), (0, 1, "east"), (0, -1, "west")]:
        nxt = (r + dr, c + dc)
        if 0 <= nxt[0] <= 4 and 0 <= nxt[1] <= 4 and nxt not in walls:
            out.append((nxt, act, 0.99))
    return out

plan = client.plan_path(
    start_state=(0, 0),
    is_goal_fn=lambda pos: pos == goal,
    get_neighbors_fn=get_neighbors,
    heuristic_fn=lambda pos: abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]),
)

print(f"Plan Success      : {plan['success']}")
print(f"Action Trajectory : {plan['path']}")
```

### 3. Counterfactual "What If" Simulation

Roll each candidate forward as the first move and rank the branches by the
simulator's safety estimate. String states, as below, use uncalibrated keyword
heuristics; they do not model a real database. Inspect the returned provenance:

```python
from gen_zero import GenZero

client = GenZero()

report = client.what_if(
    state="Database memory pressure at 94%.",
    candidates=["REJECT_NEW_SESSIONS", "SCALE_READ_REPLICAS"],
    horizon=3,
)

print("Best candidate :", report["best_candidate"])
print("Traps detected :", report["traps_detected"])
print("Provenance     :", report["provenance"])
```

### 4. Adaptive Tool Catalog Pruning

Prune a large OpenAPI or MCP tool catalog down to the top-K relevant tools:

```python
from gen_zero.router import ToolSchemaPruningGate

full_tool_catalog = [
    {"name": "deploy_container", "description": "Deploy a container to a Kubernetes cluster"},
    {"name": "scale_deployment", "description": "Scale a Kubernetes deployment"},
    {"name": "send_email", "description": "Send an email message"},
    {"name": "read_file", "description": "Read a file from disk"},
]

pruned_tools = ToolSchemaPruningGate().prune_tools(
    task_goal="Deploy microservice container to staging Kubernetes cluster",
    tools=full_tool_catalog,
    top_k=2,
)
print([t["name"] for t in pruned_tools])
```

---

## 🔌 Integration with Rust MCP Server

Gen-Zero delivers a hybrid architecture:
- **Python Suite**: Rapid algorithmic research, neural architecture design, SCM exploration, and policy training.
- **Rust MCP Server (`gen-zero-service`)**: Serving gateway with serialization, storage and Model Context Protocol (MCP) endpoints. End-to-end latency depends on the workload and semantic bridge; no microsecond service guarantee is established.

### 1. Launch the Rust MCP Server
```bash
# Build the native daemon, then serve MCP over SSE on port 8999
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo build --release -p gen-zero-cli
GENZERO_API_KEY=gz_live_your_token ./target/release/gen-zero serve --mode sse --port 8999
```

### 2. Connect an MCP Client
Any MCP client can attach to the SSE endpoint `http://127.0.0.1:8999/sse`.
The Python package ships the same transport, so you can serve it locally too (see below).

### 3. Launch Python Native MCP Server via CLI
```bash
gen-zero mcp --transport stdio
# or over SSE (GET /sse, POST /messages):
gen-zero mcp --transport sse --port 8999
```

---

## 💻 CLI Commands

The installed `gen-zero` command provides immediate access to all subsystems:

| Command | Description | Example |
|---|---|---|
| `gen-zero ask` | Fast single-forward decision evaluation | `gen-zero ask "Memory pressure 92%" "Safe to migrate?"` |
| `gen-zero route` | Tool catalog pruning router | `gen-zero route "Deploy k8s pod" --tools tools.json --top-k 3` |
| `gen-zero imagine` | Counterfactual search with optional CP-SAT safety verification | `gen-zero imagine "High error rate" --actions A B C` |
| `gen-zero stream` | Bounded rolling-window attention-sink stream | `gen-zero stream "frame_001" --actions ACT_A HOLD` |
| `gen-zero grep` | Propositional semantic search & Boolean filter | `gen-zero grep --expr '("err" AND NOT "warn")' app.log` |
| `gen-zero compact` | Sliding-window / attention sink compaction | `gen-zero compact history.json --head 5 --tail 5` |
| `gen-zero mcp` | Launch Model Context Protocol server | `gen-zero mcp --transport stdio` |
| `gen-zero status` | Check engine status, connectivity & latency | `gen-zero status` |

---

## 🧪 Testing & Verification

Run the Python verification suite:
```bash
python -m pytest python/gen_zero/tests/
```

For the Rust server, from the repository root:
```bash
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo build --release -p gen-zero-cli
./target/release/gen-zero serve --help
```

Follow the [Rust service setup](../README.md#quickstart-rust-cli-and-mcp-server)
for bridge, model, authentication and readiness prerequisites.

Licensed under [Apache License 2.0](../LICENSE).
