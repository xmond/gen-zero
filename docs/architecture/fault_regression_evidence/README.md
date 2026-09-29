# F01–F09 回归证据

工作树 `/ebs/pj/gen-zero-worktree/rm-p10`；基线 HEAD `6dfd6099591f68a8bba8bbb8ae3b2a6c302f7e2c`。
审计依据：`docs/architecture/planner_architecture_optimality_audit.md` 第 5.3 节。
工具链：`rustc 1.96.0 (ac68faa20 2026-05-25)`，`cargo 1.96.0 (30a34c682 2026-05-25)`。

这些测试使用明确返回 `Ok` 的合成故障模型，避免由真实模型自身校验掩盖 planner 漏洞。未声称六个引擎具备完整算法能力、真实任务安全或最优性。

## 代码证据

以下行号对应本次修改后的文件，路径相对仓库根目录。

| 缺陷 | 回归测试：`crates/gen-zero-planner/tests/fault_regression_tests.rs` | 实现与覆盖边界 |
| --- | --- | --- |
| F01 | :124 | `crates/gen-zero-gate/src/constraint.rs:78`、`:92`、`:103`：单动作、bundle、context 均采用 i128 乘加；测试重复 MAX 系数及并发重复占用，不依赖 panic 视为成功。 |
| F02 | :149 | `crates/gen-zero-planner/src/engine.rs:211`：MCTS 校验完整 transition；六引擎直接调用均拒绝 NaN/±Inf reward，并用有效模型作正对照。 |
| F03 | :181 | `crates/gen-zero-planner/src/engine.rs:697`：CFR 校验 successor；MCTS/CFR 拒绝末尾坐标 NaN/±Inf。 |
| F04 | :204 | `crates/gen-zero-planner/src/pipeline.rs:588`：`decide_with_context` 将占用快照用于初筛与最终门禁；测试 HardStop、规则编号与零模型调用，七种模式全拒绝。 |
| F05 | :244 | 注册真实 SheafProblem 后，缺证动作在 gate 与七种 decide 模式均拒绝，模型调用为零；无证书要求的动作仍可选。没有实现带热证书的规划入口。 |
| F06 | :284、:439、:454 | `crates/gen-zero-planner/src/pipeline.rs:633`、`:38`、`:705`：候选首步筛选、搜索期拒绝、最终首步复验。七种模式排除高奖励 done 动作；全部危险时拒绝；另测搜索瞬时 hazard 和最终预测变坏。 |
| F07 | :309 | `crates/gen-zero-planner/src/pipeline.rs:138`、`:145`：`policy_allowed()` 与 `hazard_free()` 独立；`is_safe()` 为两者合取。测试四种真假组合，保留明确的反事实模拟。 |
| F08 | :332 | `crates/gen-zero-planner/src/lib.rs:52`、`:91`：无上下文 legacy Reflex 明确拒绝；显式 gate API 过滤禁止首动作，全禁拒绝；同时覆盖 ProductionPipeline Reflex。 |
| F09 | :387 | `crates/gen-zero-planner/src/router.rs:104`：两个 dispatch 入口拒绝非法 entropy，模型调用为零；有效 entropy 覆盖三个路由区间。Router 保留请求 entropy 的门禁筛选结果。 |

门禁／路由／Reflex 三文件由 Luna 侧线修改，主线程检查实际共享工作树差异与 HEAD，负责其余实现、全部回归与集成验收。没有 commit、push 或工作树恢复操作。

## 原始验证命令与日志

所有命令在上述工作目录运行。每次命令直接重定向完整 stdout/stderr，保存 `$?` 后才读取日志；没有用管道截断测试并吞掉退出码。

```bash
cargo test -p gen-zero-planner --test fault_regression_tests
cargo check -p gen-zero-planner -p gen-zero-gate
cargo test -p gen-zero-planner -p gen-zero-gate
cargo test --release -p gen-zero-planner --test fault_regression_tests
```

Debug 指定回归：11 passed / 0 failed，退出码 0；编译检查退出码 0；两包完整测试：126 passed / 0 failed，退出码 0；Release 指定回归：11 passed / 0 failed，退出码 0。

最终结果见同目录 `regression.log`/`.exit`、`check.log`/`.exit`、`all-tests.log`/`.exit`、`release.log`/`.exit`。
`initial-compile.log` 保留首次测试代码编译错误（把单元结构体误写为 `::default()`，退出码 101）；修复后 `intermediate-tests.log` 为中间版本 10/10 通过，退出码 0。两者不是最终验收日志。

## 未实现能力与兼容性影响

- F05 只固化缺证拒绝，**不代表已有可用的带证书规划入口**。
- 默认 `decide` 明确采用空并发上下文；有占用的调用者必须使用 `decide_with_context`。快照不是跨请求原子资源预留，尚无 TOCTOU 保证。
- `done=true` 在 decide 中按危险处理，包括正常终止模型可能产生的保守拒绝；若需要区分正常终止与危险，需另行改变模型契约。
- 只验证本次观察到的模型输出。随机模型与真实环境的安全不能从有限模型调用推出；没有新增全轨迹可达性证明。搜索期间观察到 hazard 直接报错，不回退放行。
- 原 `DecisionEngine::evaluate_reflex` 没有政策参数，现始终返回明确错误；调用者需迁移到显式 gate 入口。该入口仅做基础政策过滤，不预测动态 hazard。
- `simulate` 仍可生成政策禁止的反事实轨迹；`is_safe` 现在要求政策许可与无模型 hazard 同时成立。
- decide 新增候选预测及最终预测，增加模型调用成本；未验证延迟或 service 在线接入，未声称满足 2ms。
