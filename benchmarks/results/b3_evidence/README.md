# B3 运行证据

正式汇总：`../league_and_constraints_eval_results.json`。总体验收：NOT_ACCEPTED。

## 原始命令与退出码

- `python python/gen_zero/scripts/benchmark_issue_76_constraint_compiler.py`：0；见 constraints.log。原环境缺少 OR-Tools，不能视作 CP-SAT 实测。
- `python python/gen_zero/run_league_benchmark.py`：1；见 league.log。历史平均胜率 52%，最低 30%，不满足断言。
- `python python/gen_zero/evaluate/web_agent_benchmark.py`：0；见 web_agent.log。只是 3 场景、30 次动作/参数匹配，20 次成功；并非真实 Web 任务。

## 补测

依赖安装：`uv venv --system-site-packages /tmp/b3-eval-venv`，`uv pip install --python /tmp/b3-eval-venv/bin/python ortools`。安装退出码 0，版本见 environment.json。初次运行缺 torch，保留失败日志；通过独立 venv 的 b3_base_dependencies.pth 显式加载原环境 `/home/luy/.hermes-venv/lib/python3.11/site-packages` 后重试。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python /tmp/b3-eval-venv/bin/python benchmarks/b3_measure.py
```

正式补测退出码 **1**（降级与 League 验收失败），见 supplemental_final_command.json / supplemental_final.log。supplemental.json 保留全部 1000 次求解、20 次编译、3 个种子的 3240 场训练比赛及策略轨迹。采集器只观察真实调用返回，不修改生产方法。计时区间不启用 profiler；League 的 profiler 仅用于识别实际活动对象，避免 archive 重用 ID 污染角色胜率。

- 编译吞吐：34.2633 rules/s。
- 名义求解规则吞吐：1714.5873 rules/s（8 规则 × 1000 次 / 总调用秒数，包含降级，不是纯 CP-SAT 吞吐）。
- 求解墙钟最小/最大：0.033798 / 16.054933 ms；均值 4.665846 ms；P99 9.117924 ms。
- 242/242 违规提案被拒绝；1000 样本无不安全放行，仅是有限样本证据。
- 844 个 CP_SAT_OPTIMAL 标签、53 次唯一可行候选直接返回、103 次 UNKNOWN 降级。标签来自生产代码，不保证 OR-Tools 原始状态全部 OPTIMAL。
- Web 真实任务成功率、交互步数、回溯率未测，汇总用 null，不能用候选动作数冒充步数。

所有命令的 argv、环境、原始子进程退出码在 *_commands.json / *_command.json 中。调度脚本本身返回 0 不代表子命令通过。初次补测环境收集也因缺 pytest metadata 失败；该次没有生成 environment.json，后续成功采集的环境才是正式环境。

## 验证与限制

focused_tests.log：19 passed，退出 0；web_tests.log：12 passed，退出 0。单元测试中的既有合成/替身用例不用于宣称真实智能体能力。

旧入口输出中的固定认证文案、未测量的权重漂移等保留为原始证据，不作为验收。supplemental_prior.json 为早期诊断，其按 ID 汇总的角色胜率受存档同名影响，已经被正式文件替代；请勿引用其角色胜率。

本次任务是重测和记录，未修复生产模型/基准的既有缺陷，未替换生产模块，未提交或推送，未派子代理或 reviewer。详细问题与 path:line 见正式汇总 findings。
