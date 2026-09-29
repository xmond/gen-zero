# Gen-Zero vs CLM-8B SOTA: DeepSWE & Terminal-Bench 2.1 专项评测实施方案

> 本文是历史规划，不是能力或成绩证明。下文目标、旧版 Harbor 示例和全量评测设想均未由本次单题验证确立；DeepSWE 当前操作以 [runbook](deepswe_adapter_runbook.md) 为准。

**目标基准**：
- **Datacurve DeepSWE** (113 Tasks, 91 Repos, 5 Languages)：当前 SOTA 标杆 **CLM-8B (81.6%)**
- **Terminal-Bench 2.1** (89 Tasks, Harbor Framework)：当前 SOTA 标杆 **CLM-8B (87.6%)**
- **核心战术目标**：部署并运行 `gen-zero` 认知流水线，借助 **PUCT 蒙特卡洛前瞻规划 (`imagine`)**、**形式化安全与几何门控 (`PolicyGate` + Sheaf)** 以及 **紧凑上下文自适应路由 (`compact` / `route`)**，在 Dev Server 物理沙箱环境中全面超越 CLM-8B SOTA 指标。

---

## 1. 核心能力对比与超越策略

| 评测维度 | CLM-8B Baseline (现有 SOTA) | Gen-Zero 破局与超越策略 (Target) |
|---|---|---|
| **DeepSWE (长程软件工程)** | **81.6%** (基于 PRM 离线/半离线轨迹排序与搜索) | **目标 > 85.0%**<br>• 使用 `gen-zero-service` 的 `what_if` 与 `simulate` 进行 Bug 定位与反事实补丁前瞻；<br>• 引入 `compact` (zstd 混合压缩) 克服长上下文退化；<br>• 独立测试驱动开发 (TDD) 自检闭环，补丁生成后在沙箱内完成单元测试回归再提交。 |
| **Terminal-Bench 2.1 (复杂终端交互)** | **87.6%** (基于单步 ReAct / SFT Policy) | **目标 > 90.0%**<br>• `PolicyGate` 阻断灾难性操作（误删配置、死循环进程挂起、逃逸破坏测试 harness）；<br>• 高熵动作分支自动触发 `imagine` (MCTS 多步预演)，在关键排错转折点上搜索最优执行链；<br>• 状态-动作等变投影 (`choice_head`) 消除候选动作排序偏置。 |

---

## 2. Dev Server 评测环境基础设施就绪指南

在 Dev Server（64 核、高内存、大容量 NVMe）上执行评测前，需打通底层容器与评测工具链。

### 2.1 启动宿主机 Docker 守护进程（必要条件）
由于 Terminal-Bench 和 DeepSWE 均依赖容器环境隔离执行 Agent 生成的命令与补丁，必须保证 Docker Daemon 处于运行状态：
```bash
# 启动 Docker 并赋予权限
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# 验证 Docker 连接
docker info
```

### 2.2 安装官方 Harness 工具链
```bash
# 1. 安装 Harbor (Terminal-Bench 2.1 官方运行框架)
uv tool install 'harbor[docker]'

# 2. 安装 Pier (Datacurve DeepSWE 官方运行工具)
uv tool install datacurve-pier

# 3. 验证 CLI 安装
harbor --version
pier --version
```

### 2.3 克隆基准测试任务集
```bash
export GENZERO_REPO="$(git rev-parse --show-toplevel)"
export BENCHMARK_ROOT="$HOME/benchmarks"
mkdir -p "$BENCHMARK_ROOT" && cd "$BENCHMARK_ROOT"

# 克隆 Terminal-Bench 2.1
git clone https://github.com/harbor-framework/terminal-bench-2-1.git

# 克隆 Datacurve DeepSWE 任务集
git clone https://github.com/datacurve-ai/deep-swe.git
```

---

## 3. Gen-Zero Harness Adapter 架构与实现

评测系统由两部分构成：
1. **服务端 (`gen-zero-service`)**：在后台常驻运行，提供 HTTP `/v1/decisions` 及 11 个认知动词（`ask`, `route`, `imagine`, `simulate`, `what_if` 等）。
2. **评测适配器 (`gen_zero_agent.py`)**：遵循 Harbor / Pier 的 Agent 交互规范，驱动沙箱终端并调用 `gen-zero` 大脑。

### 3.1 启动本地 `gen-zero-service` 引擎
在 Dev Server 后台拉起生产级服务：
```bash
cd "$GENZERO_REPO"
cargo run --release -p gen-zero-cli -- serve --port 8080 --host 127.0.0.1
```

### 3.2 Terminal-Bench 2.1 适配器实现 (`gen_zero_tb_adapter.py`)
保存在 `"$BENCHMARK_ROOT/gen_zero_tb_adapter.py"`：

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
        根据终端输出与任务指令，调用 gen-zero 做出下一步决策
        """
        # 构造决策上下文
        payload = {
            "verb": "ask",
            "mode": "auto",
            "context": instruction,
            "scenario": observation[-4000:],  # 截取近端观察
            "candidates": [
                "inspect_directory", 
                "check_service_logs", 
                "edit_configuration", 
                "run_validation_test", 
                "submit_task"
            ]
        }
        
        # 1. 认知决策：调用 ask / imagine 获取战术意图
        try:
            resp = requests.post(self.service_url, json=payload, timeout=30)
            res_data = resp.json()
            intent = res_data.get("action", "inspect_directory")
        except Exception as e:
            intent = "inspect_directory"

        # 2. 结合意图与局部反思生成具体 Shell 命令
        # （在接入 LLM backbone / NanoCore 时，由 ChoiceHead 选择或由模型生成具象化命令）
        cmd = self._synthesize_bash_command(intent, observation)
        
        # 3. 经过本地安全门控校验，确保无未受控的死循环或越权
        return cmd

    def _synthesize_bash_command(self, intent: str, observation: str) -> str:
        # 具体根据当前观测状态输出相应排查命令
        if "error" in observation.lower() and "syntax" in observation.lower():
            return "python3 -m py_compile $(git diff --name-only)"
        return "ls -la"

if __name__ == "__main__":
    # Harbor Agent CLI 接口对接
    agent = GenZeroTerminalAgent()
    # 标准 I/O 驱动协议
```

### 3.3 Datacurve DeepSWE 适配器

实际实现、运行命令与限制见 [DeepSWE adapter runbook](deepswe_adapter_runbook.md)。
代码生成使用明确配置的外部 Proposer；没有实现 Gen-Zero 世界模型代码修复能力。
真实单题结果以 `evidence/t3-deepswe/` 的 patch、原始日志及判分为准。


---

## 4. 四阶段实测运行与推进流水线

### 阶段一：单题冒烟验证 (Phase 1: Single-Task Smoke Test)
- **目的**：打通 `Docker 沙箱 -> Agent Loop -> gen-zero-service -> Verifier 评测判定` 的完整链路。
- **执行命令**：
  ```bash
  # Terminal-Bench 2.1 冒烟
  harbor run -d "$BENCHMARK_ROOT/terminal-bench-2-1" \
    -e docker \
    -a "$BENCHMARK_ROOT/gen_zero_tb_adapter.py" \
    --tasks "find-broken-symlinks" \
    --output-dir "$BENCHMARK_ROOT/results/tb_smoke"

  # DeepSWE 冒烟
  # 先按 runbook 配置明确的 Proposer 与唯一 EVIDENCE_DIR
  bash "$BENCHMARK_ROOT/run_deepswe_smoke.sh"
  ```
- **通过标准**：
  - 正常进入沙箱并捕获命令回显；
  - `gen-zero-service` 产生真实 `_meta.mount` 与 `_meta.engine` 审计日志；
  - 评测判定结果为通过（Pass），无容器卡死。

### 阶段二：小批量调优 (Phase 2: Dev Tuning Batch, 各 10 题)
- **目的**：调优 `imagine` 搜索步长预算 (`step_budget`)、`c_puct` 探索系数与 `PolicyGate` 阻断阈值。
- **执行命令**：
  ```bash
  # 随机抽取 10 题评测
  harbor run -d "$BENCHMARK_ROOT/terminal-bench-2-1" -e docker -k 10 -a ...
  # DeepSWE 批量评测未验证；不要将 -k（重复次数）当作任务数量。
  ```
- **关注指标**：
  - 单步耗时（确保 `PolicyGate` 判定保持在 10ms 以内，`imagine` 前瞻在 300ms 以内）；
  - 是否存在无效重试死循环。

### 阶段三：全量 SOTA 冲刺评测 (Phase 3: Full Benchmark SOTA Run)
- **Terminal-Bench 2.1 全量评测**（共 89 个任务）：
  ```bash
  harbor run -d "$BENCHMARK_ROOT/terminal-bench-2-1" \
    -e docker \
    -a "$BENCHMARK_ROOT/gen_zero_tb_adapter.py" \
    --concurrency 16 \
    --output-dir "$BENCHMARK_ROOT/results/tb_full_eval"
  ```
- **Datacurve DeepSWE 全量评测**（共 113 个任务）：
  ```bash
  # DeepSWE 全量评测尚未实施；单题入口见 runbook。
  ```

### 阶段四：自动化判分与 SOTA 胜出判定 (Phase 4: SOTA Evaluation)
运行统一结果汇总分析：
```bash
python "$BENCHMARK_ROOT/eval_report_generator.py" \
  --tb-results "$BENCHMARK_ROOT/results/tb_full_eval" \
  --deepswe-results "$BENCHMARK_ROOT/results/deepswe_full_eval" \
  --baseline-clm-tb 87.6 \
  --baseline-clm-deepswe 81.6
```

**胜出验收准则**：
1. **Terminal-Bench 2.1 成功率 $\ge 88.0\%$**（击败 CLM-8B 87.6%）；
2. **DeepSWE 解决率 $\ge 82.5\%$**（击败 CLM-8B 81.6%）；
3. **零越权与零挂死**：`PolicyGate` 拦截率 $100\%$ 命中真实潜在越权；
4. 产出包含每道任务执行轨迹日志、Patch Diff 与 Verifier 退出码的权威评测报告。
