# B1 实测与有效性审计

基线 HEAD：`478699c079bf4a053b953bb83b7e7f310eae4922`；时间：`2026-09-26T14:22:55Z`。

## 已实现

执行两个指定入口，并逐项调用原有 12 个测试方法（每项请求 50 次），保留完整 stdout/stderr、原始退出码、异常堆栈和逐次耗时。没有修改生产代码、修补接口或替换旧模块。诊断在单项异常后显式记录并继续，最终退出 1。

### 命令与退出码

| 命令 | 退出码 | 日志 |
|---|---:|---|
| `python -m pytest python/gen_zero/evaluate/decision_foundation_benchmark.py` | 5 | [01.log](b1_logs/01.log) |
| `python python/gen_zero/evaluate/decision_foundation_benchmark.py` | 1 | [02.log](b1_logs/02.log) |
| `python python/gen_zero/scripts/run_12_planners_benchmark.py` | 1 | [03.log](b1_logs/03.log) |
| `python benchmarks/results/b1_logs/audit_runner.py` | 1 | [04.log](b1_logs/04.log) |
| `PYTHONPATH=python python python/gen_zero/evaluate/decision_foundation_benchmark.py` | 0 | [05.log](b1_logs/05.log) |

pytest：0 个测试，退出 5，测试通过率未定义。Foundation 直接运行导入失败；显式设置 PYTHONPATH 后退出 0，但仅导入类，没有运行评测，不能算通过。原始 12 Planners 在第 3 项异常退出，未输出汇总。

### 12 Planners 诊断实测

10/12 项完成（83.33%），2/12 项异常；这是诊断完成率，不是解决率。完成项各有 50 次原始耗时，存于 JSON 的 legacy_metrics；成功率仅沿用原脚本谓词。所有延迟是整次调用耗时，不是单个搜索步骤。

| 项目 | 原脚本成功率 | Mean ms | P95 ms | P99 ms | 状态 |
|---|---:|---:|---:|---:|---|
| 1. Reflex Fast Prior | 100.0% | 7.792 | 8.724 | 94.854 | 仅诊断完成 |
| 2. Uncertainty A* Graph | 100.0% | 0.053 | 0.082 | 0.155 | 仅诊断完成 |
| _test_bidirectional | 未取得 | — | — | — | `TypeError: 'NoneType' object is not callable` |
| 4. Dual-Head PUCT MCTS | 100.0% | 0.008 | 0.011 | 0.037 | 仅诊断完成 |
| 5. Text World Model | 100.0% | 0.025 | 0.06 | 0.081 | 仅诊断完成 |
| _test_mpc_cem | 未取得 | — | — | — | `TypeError: MpcCemEngine.plan() missing 1 required positional argument: 'state'` |
| 7. GFlowNet Sampler | 100.0% | 31.811 | 35.105 | 40.589 | 仅诊断完成 |
| 8. Causal CFR Regret | 100.0% | 0.159 | 0.236 | 0.3 | 仅诊断完成 |
| 9. CP-SAT Formal Solver | 100.0% | 0.145 | 0.196 | 0.296 | 仅诊断完成 |
| 10. Continuous Latent MPC | 100.0% | 4.789 | 5.412 | 8.967 | 仅诊断完成 |
| 11. PRM Safety Barrier | 100.0% | 0.001 | 0.002 | 0.009 | 仅诊断完成 |
| 12. D-SCM Bluff Detector | 100.0% | 0.045 | 0.092 | 0.162 | 仅诊断完成 |

### 致命有效性问题

Foundation 生成 495 条、11 域模板样本；真实模型评测 0 条。配置名称 Qwen/Qwen3.5-9B 不代表加载了模型。为避免将模式匹配冒充推理，没有执行模拟打分并将其包装成模型指标。Accuracy、Latency、Entropy distribution、Tier 分布均为 null。

默认 Planner 客户端未加载 checkpoint；日志记录随机初始化权重降级及 localhost:8090 仲裁连接拒绝。因此 Reflex 的 100% 动作成员检查不能证明模型正确性。MCTS/CEM fixture 的动作奖励恒为 1，缺少区分正确决策的质量标签。

- `python/gen_zero/evaluate/decision_foundation_benchmark.py:61`：Calls simulated regex/hash scoring; not actual model inference. No pytest tests or executable main.

- `python/gen_zero/train/qwen_post_trainer.py:140`：Hard-coded regex/action associations plus hash logits, not learned LoRA weights.

- `python/gen_zero/evaluate/decision_foundation_benchmark.py:176`：ECE report PASS is unconditional; report conclusion at line 194 claims delivery and hardware SLA without runtime evidence.

- `python/gen_zero/planner/engines/mcts_engine.py:392`：search performs fixed-action discounted rollouts; eval_fn and simulations are unused; timing is not evidence of PUCT tree search.

- `python/gen_zero/scripts/run_12_planners_benchmark.py:113`：Reflex success only tests action membership; most other cases test returned action/output presence. Not solved-task accuracy.

- `python/gen_zero/planner/engines/astar_engine.py:193`：Bidirectional invocation dispatches to forward search with absent goal predicate; runtime TypeError.

- `python/gen_zero/scripts/run_12_planners_benchmark.py:188`：CEM passes initial_state but current plan requires state; runtime TypeError.

## 未验证

真实模型推理准确率、概率熵和 Tier 分布；生产单步延迟与硬件 SLA；规划得分、解决率和搜索步数；跨随机种子稳定性。原入口未提供这些可接受的测量，未用占位数字填充。

## 未完成

真实 Decision Foundation 多模型基准、12 项全部成功执行及目标指标验收仍未完成。阻塞项是模拟打分入口、缺少已加载模型、双向搜索接口错配、CEM 参数错配和指标定义/采集缺失。本任务保留失败证据，没有将修复后的新问题或替代算法混为原基准。

## 复现

在仓库根目录执行 `python benchmarks/results/b1_logs/audit_runner.py`（预期退出 1）；随后执行 `python benchmarks/results/b1_logs/finalize_report.py` 更新汇总。命令 1–3 的历史原始日志不被重写。单次默认随机状态、无预热，耗时仅描述本机本次运行。

结构化结果：[decision_foundation_eval_results.json](decision_foundation_eval_results.json)。其中包含源码/日志 SHA-256、495 条输入的路径与 SHA-256、各项完整异常和逐调用延迟。
