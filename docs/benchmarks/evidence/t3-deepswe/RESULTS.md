# T3 结果：未完成修复目标，真实单题判分失败

## 已实现 / 已验证

- 新增 `benchmarks/gen_zero_deepswe_adapter.py`，同步至 dev 的
  `/home/user/benchmarks/gen_zero_deepswe_adapter.py`。
- 这是 **Fail-Closed 能力探针**，不是完整生产修复适配器。已接通 Pier
  插件、沙箱代码读取、四个 MCP 动词的真实请求、原始日志和退出码。
- 本地 `python3 -m py_compile benchmarks/gen_zero_deepswe_adapter.py` 退出 0；
  dev 的 Pier Python 实际导入成功，三项非法配置均被拒绝，无 mock。
- 官方标准任务 `tomlkit-toml-table-converters` 运行一次。
  Job: `t3-deepswe-smoke-20260926T142503Z`；
  trial: `tomlkit-toml-table-converters__H9TAzpk`。
- 代理与 verifier 使用不同 Docker 容器：
  agent `1df8b337d3b6d1989c684b0b3684a76d9012cff718b41148d19a56a8b4858e5c`；
  verifier `6995e28ee18f411ce8079283e8e3c56fdcae31dfab4c41d261361309756d472f`。
  `raw/docker-events.jsonl` 为原始 Docker 事件；`raw/docker-execs.json` 按
  execID 配对命令与 exitCode，可核查独立 verifier 的真实执行。

| 项目 | 实际结果 |
| --- | --- |
| Pier 命令原始退出码 | 0 |
| Gen-Zero 进程原始退出码 | 0 |
| 代理探针进程原始退出码 | 78 |
| verifier `/tests/test.sh` 原始退出码 | 0 |
| reward | **0，失败** |
| 新增需求 F2P | **0 / 60** |
| 原有测试 P2P | 964 / 964 |
| 官方 collect 导出的 model.patch | **0 字节** |
| 本报告验收脚本退出码 | **1，不接受** |

`partial=0.94140625` 来自空补丁仍通过原有测试，**不能宣传成 94.1% 修复成功**。
verifier 脚本成功完成判分和任务得到 reward=1 是不同结果。

真实运行命令见 [pier-command.txt](raw/pier-command.txt)；完整 runner 见
[run-smoke.sh](raw/run-smoke.sh)。可重复检查已保存结果：

```bash
python3 docs/benchmarks/evidence/t3-deepswe/check_result.py
# 预期退出 1：已保存的试验不满足修复验收条件。
```

结果见 [acceptance.json](acceptance.json)；原始 trial 目录：
`raw/jobs/t3-deepswe-smoke-20260926T142503Z/tomlkit-toml-table-converters__H9TAzpk/`，
包含 `result.json`、`agent/engine.stdout`、`agent/mcp.requests.jsonl`、
`agent/adapter.exit`、`artifacts/model.patch`、`verifier/reward.json`、
`verifier/test-stdout.txt` 和 `verifier/ctrf.json`。

## 最致命的问题

1. **dev 引擎确有硬编码伪能力。** 实际返回 `expected_reward=1.45` 和
   `formal_checked=true`；dev 的 `/home/user/gen-zero/crates/gen-zero-service/src/zero.rs:330`
   和 `:331` 将二者写死；`:372` 的 `matches=1` 同样写死。
   [dev 源码摘录](raw/dev-zero-excerpt.txt) 保存了行号和文件 SHA256。
   适配器没有将这些成功字段接受为代码推理或正确性证明。
2. **当前工作树修正了能力声明，但仍缺少所需能力。**
   `crates/gen-zero-service/src/worldsim.rs:4` 说明动力学未训练；`:13` 明确没有
   文本编码器。不能把代码哈希或补零数组伪装成可预测补丁结果的 latent。
3. **compact 只返回大小统计。**
   `crates/gen-zero-service/src/zero.rs:3895` 不返回可恢复内容；真实返回也只有
   47303→10918 字节统计，不能当作已完成上下文压缩消费链路。
4. **没有接入代码生成后端。** 尚不能生成最小复现、候选补丁和回归命令。
   这不是“功能已实现、只差一次测试”，而是核心求解链路未实现。

## 未验证

- 成功修复路径、恢复型上下文压缩、可靠代码状态编码和反事实模型均未验证。
- 超时/取消进程清理分支未做故障注入验证。
- dev 旧源码目录没有 `.git`，`git rev-parse` 退出 128，不能声称已验证其
  commit。保存了二进制和源码文件哈希，未将它冒充本工作树 HEAD。

## 未完成

生产修复适配器、生成式复现、最佳 patch 选择、修复后回归和非空提交补丁均未完成。
需要真实代码生成后端，以及经过验证的代码状态/转移能力；或者由用户明确改变
方案，采用生成模型加实际执行测试来选择补丁，并如实撤下 Gen-Zero 反事实能力宣称。

## 运行与工作树边界

- 未读取参考 solution，未使用 oracle，未修改官方 task/verifier，未派子代理或 reviewer。
- 未替换已有模块：目标文件原本不存在，没有旧适配器符号需要迁移或删除。
- 未执行 stash / checkout / reset / clean / force-push；未提交本工作树。
- 任务集 clone 最初退出 128：并行基础设施工作已创建同一路径；核对既有
  任务集 HEAD 与官方一致后使用，保留 `dataset-clone.log` 与退出码。
- 镜像准备期间发现旧引擎使用 `action` 协议，将请求改为兼容当前与旧引擎的
  `action`，并增加探针自身源码哈希。Pier 父进程加载初始模块，但实际 MCP
  子进程执行的是最终文件：`agent/adapter-source.sha256` 与
  `raw/adapter-final.sha256` 一致。父进程新增的取消清理分支未在该次运行中执行。
  `raw/sha256.txt` 保留运行启动时的旧适配器哈希，没有覆盖这项差异。
- 全量原始证据也保存在 dev `/home/user/benchmarks/t3-evidence/`。
