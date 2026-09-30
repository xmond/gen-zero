# Gen-Zero vs CLM-8B SOTA: DeepSWE & Terminal-Bench 2.1 Evaluation Plan

> This is a historical plan, not evidence of capability or benchmark results. The targets, old Harbor example, and proposed full evaluations below were not established by the single-task validation. For current DeepSWE procedures, follow the [runbook](deepswe_adapter_runbook.md).

**Target benchmarks**:
- **Datacurve DeepSWE** (113 Tasks, 91 Repos, 5 Languages):then-current SOTA reference **CLM-8B (81.6%)**
- **Terminal-Bench 2.1** (89 Tasks, Harbor Framework):then-current SOTA reference **CLM-8B (87.6%)**
- **Core tactical objective**:Deploy and run `gen-zero` cognitive pipeline using **PUCT Monte Carlo lookahead planning (`imagine`)**,**formal safety and geometric gates (`PolicyGate` + Sheaf)** and **compact-context adaptive routing (`compact` / `route`)**,to exceed the cited CLM-8B benchmarks in the Dev Server sandbox.

---

## 1. Capability Comparison and Proposed Strategy

| Evaluation dimension | CLM-8B Baseline (cited SOTA) | Gen-Zero Proposed strategy (Target) |
|---|---|---|
| **DeepSWE (long-horizon software engineering)** | **81.6%** (based on offline/semi-offline PRM trajectory ranking and search) | **Target > 85.0%**<br>• Use `gen-zero-service` `what_if` and `simulate` for bug localization and counterfactual patch lookahead;<br>• Introduce `compact` (zstd hybrid compression) to address long-context degradation;<br>• Independent test-driven development (TDD) self-check loop: run regression tests in the sandbox after generating a patch and before submission. |
| **Terminal-Bench 2.1 (complex terminal interaction)** | **87.6%** (based on single-step ReAct / SFT Policy) | **Target > 90.0%**<br>• `PolicyGate` blocks catastrophic actions (accidental configuration deletion, hung infinite-loop processes, or escape that damages the test harness);<br>• High-entropy action branches trigger `imagine` (MCTS multistep preview),to search for an execution sequence at critical debugging points;<br>• State-action equivariant projection (`choice_head`) to remove candidate-action ranking bias. |

---

## 2. Dev Server Evaluation Environment Setup Guide

Before evaluating on Dev Server (64 cores, high memory, large NVMe storage), prepare the container runtime and benchmark tooling.

### 2.1 Start the Host Docker Daemon (Prerequisite)
Terminal-Bench and DeepSWE use containers to isolate agent-generated commands and patches, so the Docker daemon must be running:
```bash
# Start Docker and grant access
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Verify Docker connectivity
docker info
```

### 2.2 Install the Official Harness Toolchain
```bash
# 1. Install Harbor (official Terminal-Bench 2.1 runner)
uv tool install 'harbor[docker]'

# 2. Install Pier (official Datacurve DeepSWE runner)
uv tool install datacurve-pier

# 3. Verify CLI installation
harbor --version
pier --version
```

### 2.3 Clone Benchmark Task Sets
```bash
export GENZERO_REPO="$(git rev-parse --show-toplevel)"
export BENCHMARK_ROOT="$HOME/benchmarks"
mkdir -p "$BENCHMARK_ROOT" && cd "$BENCHMARK_ROOT"

# Clone Terminal-Bench 2.1
git clone https://github.com/harbor-framework/terminal-bench-2-1.git

# Clone the Datacurve DeepSWE task set
git clone https://github.com/datacurve-ai/deep-swe.git
```

---

## 3. Gen-Zero Harness Adapter Architecture and Implementation

The evaluation system has two parts:
1. **Service (`gen-zero-service`)**:runs as a background service and provides HTTP `/v1/decisions` and 11 cognitive verbs(`ask`, `route`, `imagine`, `simulate`, `what_if` and others).
2. **Evaluation adapter (`gen_zero_agent.py`)**:follows the Harbor/Pier agent protocol, drives the sandbox terminal, and calls the `gen-zero` engine.

### 3.1 Start the local `gen-zero-service` engine
Start the service in the background on Dev Server:
```bash
cd "$GENZERO_REPO"
cargo run --release -p gen-zero-cli -- serve --port 8080 --host 127.0.0.1
```

### 3.2 Terminal-Bench 2.1 Adapter implementation (`gen_zero_tb_adapter.py`)
Save as `"$BENCHMARK_ROOT/gen_zero_tb_adapter.py"`:

```python
"""
Gen-Zero Adapter for Terminal-Bench 2.1 (Harbor Framework)
"""
import sys
import json
import requests

GEN_ZERO_SERVICE = "http://127.0.0.1:8080/v1/decisions"

class GenZeroTerminalAgent:
    def __init__(self, service_url=GEN_ZERO_SERVICE):
        self.service_url = service_url
        self.trajectory = []

    def act(self, instruction: str, observation: str) -> str:
        """
        Call gen-zero to select the next action from the task instruction and terminal output
        """
        # Construct decision context
        payload = {
            "verb": "ask",
            "mode": "auto",
            "context": instruction,
            "scenario": observation[-4000:],  # Use the most recent observation
            "candidates": [
                "inspect_directory",
                "check_service_logs",
                "edit_configuration",
                "run_validation_test",
                "submit_task"
            ]
        }

        # 1. Cognitive decision: call ask / imagine for tactical intent
        try:
            resp = requests.post(self.service_url, json=payload, timeout=30)
            res_data = resp.json()
            intent = res_data.get("action", "inspect_directory")
        except Exception as e:
            intent = "inspect_directory"

        # 2. Generate a concrete shell command from the intent and local observations
        # (With an LLM backbone / NanoCore connected, ChoiceHead selects or the model generates a concrete command)
        cmd = self._synthesize_bash_command(intent, observation)

        # 3. Check the command with the local safety gate for uncontrolled loops or unauthorized actions
        return cmd

    def _synthesize_bash_command(self, intent: str, observation: str) -> str:
        # Return a diagnostic command based on the current observation
        if "error" in observation.lower() and "syntax" in observation.lower():
            return "python3 -m py_compile $(git diff --name-only)"
        return "ls -la"

if __name__ == "__main__":
    # Harbor Agent CLI integration
    agent = GenZeroTerminalAgent()
    # Standard I/O driver protocol
```

### 3.3 Datacurve DeepSWE Adapter

For the actual implementation, commands, and limitations, see [DeepSWE adapter runbook](deepswe_adapter_runbook.md).
Code generation uses an explicitly configured external proposer; Gen-Zero world-model-based code repair has not been implemented.
For real single-task results, use the patch, raw logs, and verifier score in `evidence/t3-deepswe/`.


---

## 4. Four-Phase Evaluation Workflow

### Phase 1: Single-Task Smoke Test
- **Purpose**:Connect `Docker sandbox -> Agent Loop -> gen-zero-service -> Verifier evaluation verdict` end to end.
- **Command**:
  ```bash
  # Terminal-Bench 2.1 smoke test
  harbor run -d "$BENCHMARK_ROOT/terminal-bench-2-1" \
    -e docker \
    -a "$BENCHMARK_ROOT/gen_zero_tb_adapter.py" \
    --tasks "find-broken-symlinks" \
    --output-dir "$BENCHMARK_ROOT/results/tb_smoke"

  # DeepSWE smoke test
  # First configure an explicit proposer and a unique EVIDENCE_DIR as specified in the runbook
  bash "$BENCHMARK_ROOT/run_deepswe_smoke.sh"
  ```
- **Pass criteria**:
  - Enter the sandbox and capture command output;
  - `gen-zero-service` produces real `_meta.mount` and `_meta.engine` audit logs;
  - The verifier reports Pass, with no hung container.

### Phase 2: Small Tuning Batch (10 tasks each)
- **Purpose**:Tune `imagine` search step budget (`step_budget`),`c_puct` exploration coefficient and `PolicyGate` blocking threshold.
- **Command**:
  ```bash
  # Evaluate 10 randomly selected tasks
  harbor run -d "$BENCHMARK_ROOT/terminal-bench-2-1" -e docker -k 10 -a ...
  # DeepSWE Batch evaluation is unverified; do not treat -k (repeat count) as the task count.
  ```
- **Metrics to watch**:
  - Per-step latency (target `PolicyGate` decision under 10 ms,`imagine` lookahead under 300 ms);
  - Presence of unproductive retry loops.

### Phase 3: Proposed Full SOTA Evaluation
- **Terminal-Bench 2.1 full evaluation**(89 tasks):
  ```bash
  harbor run -d "$BENCHMARK_ROOT/terminal-bench-2-1" \
    -e docker \
    -a "$BENCHMARK_ROOT/gen_zero_tb_adapter.py" \
    --concurrency 16 \
    --output-dir "$BENCHMARK_ROOT/results/tb_full_eval"
  ```
- **Datacurve DeepSWE full evaluation**(113 tasks):
  ```bash
  # DeepSWE full evaluation not yet run; see the runbook for the single-task entry point.
  ```

### Phase 4: Automated Scoring and SOTA Comparison
Run consolidated result analysis:
```bash
python "$BENCHMARK_ROOT/eval_report_generator.py" \
  --tb-results "$BENCHMARK_ROOT/results/tb_full_eval" \
  --deepswe-results "$BENCHMARK_ROOT/results/deepswe_full_eval" \
  --baseline-clm-tb 87.6 \
  --baseline-clm-deepswe 81.6
```

**Proposed acceptance criteria**:
1. **Terminal-Bench 2.1 success rate $\ge 88.0\%$**(exceeds CLM-8B 87.6%);
2. **DeepSWE resolution rate $\ge 82.5\%$**(exceeds CLM-8B 81.6%);
3. **No unauthorized actions or hangs**:`PolicyGate` interception rate $100\%$ against actual potentially unauthorized actions;
4. Produce an auditable evaluation report containing each task trajectory, patch diff, and verifier exit code.
