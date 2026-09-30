# Gen-Zero Python client

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)

`gen_zero` is the lightweight Python client and runtime interface for the Gen-Zero decision engine. It exposes candidate scoring, experimental planning, world-model simulation, and runtime gates. Installation does not supply trained weights, calibrated confidence, universal optimality, or a general safety guarantee.

## Architecture and scope

| Component | Runtime role | Boundary |
| --- | --- | --- |
| Set-Attention dual head | Scores candidate sets through policy and value heads, with an optional abstain slot. | The model starts with random weights unless a trained checkpoint is loaded. Set equivariance alone does not prove end-to-end decision invariance. |
| Planning mixture of experts (MoE) | Routes among reflex scoring, PUCT MCTS, uncertainty-weighted A*, bidirectional search, text world model, MPC/CEM, GFlowNet, CFR, and constraint filtering. | Routing uses declared hints and structural preconditions. Planners require suitable states, transitions, dependencies, or artifacts. |
| Runtime safety gates | Applies available alignment, action-constraint, perturbation, and policy checks at their respective entry points. | Coverage depends on the invoked path and supplied constraints. A heuristic or unavailable solver is not a formal proof. |

The public package is an inference and runtime interface. Offline training, dataset generation, LoRA BPTT, patch compilation, replay, and self-play pipelines reside in the separate `gen-zero-research` project and are accessed through the tuning service at `tuning.gen-zero.ai`. This repository does not ship those pipelines or their trained artifacts. This architectural boundary does not establish availability of any particular hosted service or model.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e './python'
```

Core dependencies are declared in [`python/pyproject.toml`](../pyproject.toml). Optional extras include `all` (OR-Tools, Transformers, and Hugging Face Hub), `vision`, and `dev`; for example, `python -m pip install -e './python[dev]'`. PyTorch is a core dependency and can run on CPU. GPU hardware is optional and does not supply trained weights.

## Quickstart

### Candidate decision

```python
from gen_zero import GenZero, GenZeroConfig

# Replace this placeholder with an absolute path to a trained state_dict.
client = GenZero(GenZeroConfig(dual_head_checkpoint="/absolute/path/to/dual_head.pt"))
result = client.decide(
    state="Database memory pressure is high",
    candidates=["scale_read_replicas", "reject_new_sessions"],
    mode="auto",
)

if result.get("degraded"):
    raise RuntimeError(f"Decision degraded: {result.get('degraded_reason')}")
if result.get("action") is None:
    raise RuntimeError("The engine did not select an action")
print(result["action"], result["probs"])
print(result["experts_activated"], result["latency_ms"])
```

The checkpoint path is a placeholder, not a bundled artifact. A default `GenZero()` can be constructed without a checkpoint, but its neural decision output is marked degraded and must not be treated as a trained recommendation. HTTP decision endpoints reject requests with status 503 until a valid dual-head checkpoint is configured. For `task_hint`, supply an exact paradigm name such as `"astar"`, or omit it; free-text descriptions are not routing hints. Reported latency is per invocation and has no published service-level guarantee.

### Uncertainty-weighted A* path

This symbolic graph example does not require a trained dual-head checkpoint. The neighbor probability is supplied by the caller; Gen-Zero does not estimate or calibrate it.

```python
from gen_zero import GenZero

goal = (2, 2)
walls = {(1, 1)}

def neighbors(position):
    row, col = position
    for dr, dc, action in [(-1, 0, "north"), (1, 0, "south"),
                           (0, 1, "east"), (0, -1, "west")]:
        nxt = (row + dr, col + dc)
        if 0 <= nxt[0] <= 2 and 0 <= nxt[1] <= 2 and nxt not in walls:
            yield (nxt, action, 0.99)

client = GenZero()
plan = client.plan_path(
    start_state=(0, 0),
    is_goal_fn=lambda position: position == goal,
    get_neighbors_fn=lambda position: list(neighbors(position)),
    heuristic_fn=lambda position: abs(position[0] - goal[0]) + abs(position[1] - goal[1]),
)
print(plan["success"], plan["path"])
```

## `GenZeroConfig`

[`config.py`](config.py) defines the following runtime settings. Defaults reflect the current source.

| Field | Default | Purpose |
| --- | --- | --- |
| `hidden_dim` / `embed_dim` | `4096` / `128` | Dual-head model dimensions. |
| `num_attention_layers` / `num_attention_heads` | `2` / `4` | Set-Attention depth and head count. |
| `use_value_head` / `enable_abstain` | `True` / `True` | Enable the value head and abstain slot. |
| `backbone_name` | `"Qwen/Qwen3.5-9B"` | Configured backbone identifier; it does not load weights by itself. |
| `dual_head_checkpoint` | `None` | Absolute path to trained dual-head weights; also read from `GENZERO_DUAL_HEAD_CHECKPOINT`. |
| `astar_lambda` | `1.0` | Weight of the A* uncertainty penalty `-log(p)` on an edge. |
| `mcts_simulations` / `mcts_depth` / `mcts_cpuct` | `64` / `6` / `1.4` | PUCT search budget, depth, and exploration coefficient. |
| `neural_dynamics_checkpoint` | `None` | Absolute path to trained dynamics weights; also read from `GENZERO_NEURAL_DYNAMICS_CHECKPOINT`. |
| `enable_adaptive_gating` / `adaptive_gating_artifact` | `False` / `None` | Artifact-backed gating; enabling it without the artifact fails closed. |
| `arbiter_endpoint` / `arbiter_timeout_s` | `"http://localhost:8090/arbitrate"` / `2.0` | Optional remote arbitration endpoint and timeout. |
| `hard_sample_history_steps` | `5` | Runtime hard-sample history window. |

Checkpoint and adaptive-gating artifact paths must be absolute. Neural dynamics weights are separate from dual-head weights: a dual-head checkpoint does not enable neural latent transitions. Consult the source for the remaining thresholds and storage settings.

## Evidence and limitations

The former benchmark table contained historical figures without source data or reproducible artifacts in this repository, so it is omitted. Run the relevant evaluations in your own environment and report checkpoint, inputs, hardware, and dependency versions before making performance claims. A successful symbolic plan or an available gate is not evidence of calibrated neural predictions or universal safety.

## License

This project is released under the [Apache License 2.0](../../LICENSE).
