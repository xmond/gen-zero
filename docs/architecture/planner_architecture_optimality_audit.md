# Rust Planning Architecture 最优性与真实性审计

审计日期：2026-09-27。工作树：`/workspace/pj/gen-zero-worktree/b0927-opt-planner-audit`。代码基线：`025360f5c1699e13312b885874d66282a0ad89d2`。审计前 `git status --short` 为空。本次只增加报告及审计证据，不修补生产实现，不提交或推送。

## 1. 终审裁定

**当前实现应归类为【严重次优／存在结构性缺陷】。** 这不是声称已证明另一个架构在所有任务上最优，而是当前系统连“六个真正的规划范式 + 全链路形式化门禁”的必要条件都没有满足。保留独立安全裁决层、收敛重复实现，是合理工程方向；固定为六个名字既无数学最优性证明，也不能替代能力验收。

最致命的问题不是六个引擎少了第七个，而是：

1. **算法身份不实。** MCTS 是 depth-one PUCT bandit；A* 是单步排序；GFlowNet 没有 flow/TB 训练与采样；CFR 没有博弈树、信息集、累计反事实遗憾或平均策略；CP-SAT 没有 SAT、约束传播或 branch-and-bound。只有 CEM 确实进行有限 horizon 的候选轨迹评估，但它是离散 categorical 变体，而非连续高斯 MPC。
2. **形式化边界没有贯通。** Planner 不传并发上下文、agent id 或 heat certificate；PolicyGate 本身不接收预测 successor，也没有 Nanocore invariant 调用。局部规则通过不等于动态轨迹安全。
3. **存在已复现的 Fail-Closed 破口。** release 整数溢出可让不满足的线性约束通过；MCTS 吞下模型以 `Ok` 返回的 NaN reward；MCTS/CFR 接受 NaN successor。
4. **“高并发”和“2ms”不能由结构名推得。** 委员会内部串行，MCTS simulations 串行；TypedArena 分配加互斥锁；规划调用没有 deadline 参数或实际超时检查。实测默认 CEM 中位数已超过 2ms。
5. **验证环境本身很弱。** 默认“真实 Rust world model”是固定正弦扰动的线性收缩系统，不是经过任务数据验证的学习动力学。测试通过证明软件路径可执行，不证明真实决策能力。

### 1.1 已实现

- 六个 `PlanningEngine` 实现及统一 `ProductionPipeline`、七个可选 decide modes；基础数值校验、模型 `Err` 传播、全部动作被 gate 拒绝时返回 `NoFeasibleAction` 等有真实代码与测试。
- 确实存在 PUCT 访问计数、有限 horizon categorical CEM、即时收益／距离评分、一次 regret matching、线性不等式检查。
- 图撤销检查、确认／升级 tier、已注册 heat requirement 的缺证拒绝，以及独立的终态证书重新计算验证。
- 64-byte／64-byte alignment 的节点、预分配 arena、发布前初始化、原子读写与串行分配；不是完整的并行 MCTS。
- 本次实际编译测试：**112 tests passed，0 failed**；独立 release 探针编译运行成功。详见第 8 节。

### 1.2 未验证

- 真实任务上的规划成功率、长 horizon 性能、随机模型下校准、对抗 exploitability、从 Python 到 Rust 的能力保持率。
- 生产负载端到端 P99/P999、WCET、持续高并发、NUMA/allocator/缓存争用、执行器确认和安全动作落地。
- 任何“理论最优”“SOTA”“所有关键状态机覆盖”“99.5% 能力保持”断言。本文没有全行业 leaderboard 实验，也不把补充文献当作最新 SOTA 排名。

### 1.3 未完成／致命缺陷

- P0：门禁整数溢出；模型数值污染可穿过部分引擎；安全批准没有 successor／并发上下文的闭环语义。
- P1：多步 MCTS/A*、真正 CFR/GFlowNet/CP-SAT、连续动作优化、deadline 与可验证安全 incumbent 均未完成。
- P1：热证书支持停留在独立 gate API，pipeline 无证书入口；未接 Nanocore 的不变量不能以“已形式化”对外承诺。
- P2：未校准 entropy 被跨引擎复用；委员会的固定票权与顺序偏置；每次构造 arena 与成功路径字符串分配。

## 2. 证据边界与方法

主线程逐段检查 `engine.rs` 全部六引擎实现、`pipeline.rs` 决策与 rollout、`policy.rs` 判定链、`constraint.rs`，并追踪 `WorldModelDynamics`、默认动力学和 service pipeline 接线。两个只读 Luna 侧线分别检查 Python 能力映射及 router/tree/历史基准；最终裁定由主线程整合。没有把搜索命中当作实现证明，也没有用旧 Python benchmark 代替 Rust 实测。

证据目录：`planner_audit_evidence/`（历史证据，当前提交未包含）。包含原始 test log、release probe log、探针源代码、依赖 lock、环境信息和复现脚本。探针使用未修改的仓库 crate；故障模型明确标为合成输入。**探针正常退出仅表示观测成功，不表示被审计行为安全。**

报告中的 `path:line` 对应上述 HEAD；区间描述的行号是可定位的入口。对“未实现”的判断严格限于已读调用链：其他 crate 存在同名数学工具，不意味着六引擎已调用它。

## 3. 六引擎的原始实现与真实能力

| 名称 | 实际计算 | 缺失的算法核心 | 结论及代码定位 |
|---|---|---|---|
| MctsEngine | 根 + 每个合法动作一个 child；128 次默认 simulation，每次从同一个根调用 `step`，只备份即时 reward；按访问数选动作 | successor expansion、rollout/value bootstrap、终态安全语义、树复用、并行 workers | 有效的单步 bandit 原型，非完整 MCTS。`crates/gen-zero-planner/src/engine.rs:85`、`:144`、`:168`、`:206`、`:221` |
| AStarEngine | 每个动作一次 `step`；最小化 `-r + 0.05 λ ||s'-s|| + 0.1`；BinaryHeap 只 pop 一次 | 目标谓词、累计 g、可采纳 h、closed/reopen、路径回溯、多层 frontier | 堆排序的一步 cost ranker，非 A* 图搜索。`crates/gen-zero-planner/src/engine.rs:286`、`:338`、`:354`、`:366` |
| MpcCemEngine | 默认 32 samples × 3 iterations × horizon 4；分类分布采样，elite 更新，返回最好轨迹首动作 | 连续控制变量、高斯均值/协方差、每个时间步的独立分布、warm start、递归可行性 | 真正有多步计算，但只是受限离散 CEM。`crates/gen-zero-planner/src/engine.rs:403`、`:460`、`:471`、`:497`、`:544` |
| ManifoldGFlowNetEngine | `reward - 0.1 * L2 distance` 的 argmax；softmax 仅用于输出 entropy | P_F/P_B、Z、trajectory balance loss、学习、reward-proportional 随机采样、混合曲率运算 | 命名与算法不符。`crates/gen-zero-planner/src/engine.rs:598`、`:625`、`:644` |
| CfrNashEngine | 对每个动作单步 reward；减平均收益后取正，归一化，返回最大正 regret 动作 | 多玩家 payoff、信息集、reach probabilities、反事实价值、迭代累计 regret、平均策略、exploitability | 单轮 regret-shaped greedy，不是 CFR/Nash solver。`crates/gen-zero-planner/src/engine.rs:661`、`:684`、`:700` |
| CpSatFormalEngine | gate 过滤每个候选，逐个模型步进，返回即时 reward 最大者，entropy 恒 0 | 约束模型变量与搜索空间、SAT/CP propagation、分支定界、最优性界、solver statuses | gate wrapper + greedy；不是 CP-SAT/ILP 求解器。`crates/gen-zero-planner/src/engine.rs:745`、`:801`、`:819` |

**六者都不能据此签收为其名称所指的完整生产级算法。** 这不等于六者全是返回常数的 mock：它们执行真实算术、调用可替换模型，也会改变动作。问题是能力等级、命名和证明义务不匹配。代码已有诚实修正的文档，例如 `engine.rs:85`、`:286` 和 `pipeline.rs:164`；但文件顶部和若干 section 仍保留超出实现的名称／承诺。

### 3.1 可以直接推导的能力重叠

记合法候选集合为 F，固定根状态的一步收益为 r(a)，d(a)=||s'(a)-s||。

- A* 实际为 `argmax_F [r(a) - 0.05 λ d(a)]`；GFlowNet 实际为 `argmax_F [r(a) - 0.1 d(a)]`。**把 A* 的 λ 设为 2，其动作目标与 GFlowNet 完全相同**（忽略浮点实现／平局顺序差异；entropy 温度仍不同）。默认 λ=0.5 也只是同族目标的权重不同。
- CFR 的 `R(a)=max(r(a)-mean(r),0)` 不改变最高收益动作的位置。除平局处理外，它与 CP-SAT wrapper 的即时 greedy 选同一最优收益动作。这是代码公式推论，不需要借用 CFR 收敛定理。
- MCTS 在当前确定性单步模型中重复估计同一动作的相同即时收益；增加模拟量不会形成多步能力。随机模型中重复采样可以估计即时均值，但没有风险敏感或 belief-state 语义。

因此，“严格正交”不只是缺证明，**现有实现可构造明确等价关系反驳它**。

### 3.2 CEM 的真实局限

CEM 仅存储一个长度 K 的 `probs`，各 horizon 步共用；每条 elite 只保留 `(first_act_idx, total_reward)`，没有整条动作序列。更新只统计首动作，却把更新后的分布再用于所有后续时间步（`engine.rs:526`、`:544`）。对于必须“第一步 A、第二步 B”的任务，这种参数化无法独立表达时间条件。

另外，`total_reward += sub_r * 0.9`（`engine.rs:519`）对所有后续步使用相同 0.9，而不是常见的 `γ^t` 折扣。可以把它定义成特定目标，但必须明说，不能以标准 discounted MPC 验收。最优首动作保留跨 iteration 的最高单次轨迹分数，对随机模型是 best-of-samples，而非可靠期望最优估计。

`done` 会停止 rollout，这是已实现的正确边界；**停止不等于认为该动作不安全并拒绝**。默认模型的 done 表示 divergence，而 CEM 仍可以选高 reward 的 done 动作。

## 4. 正交性、覆盖性与合并的数学边界

### 4.1 “六种方法”不是决策问题空间的一组正交基

严格正交至少需要说明对象空间、内积／独立性定义及覆盖映射。这里的方法混合了：搜索策略（MCTS/A*）、滚动控制结构（MPC）、分布学习目标（GFlowNet）、博弈求解（CFR）、可行性建模／求解（CP-SAT）。这些层级天然可组合：MPC 可调用 CP-SAT；MCTS 可用 learned proposal；GFlowNet 可为树搜索生成候选。不能像线性代数基底一样宣称两两正交、数量六即完备。

算法数量也无法证明最优。应先给任务分布 D、损失 L、计算／内存／安全预算 B，然后比较 `E_D[L]` 与约束满足率的 Pareto 前沿。当前没有这种目标和实验矩阵。“减少维护模块”是工程收益；“不损失能力”是独立、尚未满足的验收要求。

### 4.2 覆盖矩阵：表示空间不等于求解能力

| 维度 | 当前可见能力 | 未覆盖或不足 |
|---|---|---|
| 离散动作 | 最多 16 个 `ActionId` 的 local frame；有限 horizon CEM | 大动作空间、组合动作、动态 successor 合法动作生成 |
| 连续状态 | 1024 维 `FullLatent` | 没有自动获得连续控制优化能力；连续 action vector 未进入契约 |
| 连续控制 | 无上述六引擎可接收的连续动作参数 | 高斯 CEM、iLQR/DDP、MPPI、混合控制变量 |
| 确定性问题 | 单步排序及固定模型 rollout | goal-conditioned shortest path、可采纳界、最优终止证明 |
| 随机问题 | `step` 可由调用者实现随机采样，MCTS/CEM 可多次调用 | 显式转移概率、belief update、chance constraint、CVaR、样本置信区间 |
| 多智能体／博弈 | 单个 action id、单个标量 reward | 玩家／联合动作、对手模型、信息集、均衡定义与 exploitability |
| 硬约束 | 已注册线性规则、graph revocation、独立 heat certificate checker | 轨迹约束、运行中的资源上下文、组合最优化、时间逻辑／可达性 |
| 部分可观测／非平稳 | 一个外部 entropy 标量 | belief state、变化检测、online model adaptation、dynamic regret |
| 因果／语言任务 | 可以人为编码到数值输入，但没有能力证明 | 因果干预、自然语言 transition、step verifier 的正式接口 |

定位：`crates/gen-zero-core/src/traits.rs:40` 的模型契约仅有 `step(state, ActionId)->(state,reward,done)`；`crates/gen-zero-planner/src/engine.rs:72` 的计划输出仅 `(ActionId, NormalizedEntropy)`；`crates/gen-zero-planner/src/pipeline.rs:207` 没有 goal、玩家、控制向量、证书或 deadline。

### 4.3 扩散、目标可达性、非平稳博弈是否盲区

- **Diffusion Planner：是尚未具备的 proposal/trajectory prior 能力，但不是必须新建“第七引擎”的数学理由。** 轨迹扩散通过学习轨迹分布和条件／引导生成候选，与当前 categorical CEM 不等价。可作为候选生成插件进入优化器，再由独立门禁验证。必须用任务数据、可行率和延迟证明值得接入；生成结果本身不构成安全证明。[Planning with Diffusion for Flexible Behavior Synthesis](https://proceedings.mlr.press/v162/janner22a.html)
- **Goal-conditioned Reachability：是更根本的语义缺口。** 当前没有目标谓词和 backward reachable/viability set，无法回答“某个安全首动作是否必然进入未来无路可走的状态”。图上的 backward search、控制系统 HJ 可达性及 learned goal-conditioned value 是不同保证等级，不能互换。学习的 reachable tube 仍需验证误差／概率保证，不能因用了 PDE 名称就视为证书。[Verification of neural reachable tubes](https://proceedings.mlr.press/v242/lin24a.html)
- **非平稳自适应博弈：未实现。** 本地无持久 regret、对手状态、时间窗或变化检测。应按真实任务选择 discounted/sliding-window regret、对手模型或在线规划，并明确相对于移动比较器的指标；标准静态 Nash 收敛不是自动可继承的承诺。

## 5. PolicyGate：实际契约、数学风险与安全断层

### 5.1 当前调用链

```text
ProductionPipeline.decide
  validate request
  prune(candidates) -> gate.evaluate(action, entropy=0, graph, agent=None)
  selected engine / router
    -> gate.evaluate_basic(action, entropy=0) repeatedly
    -> world_model.step(...)
  check returned action belongs to feasible frame
  gate.evaluate(action, REQUEST entropy, graph, agent=None)
  optionally run an independent greedy rollout for display
  return Decision { action, gate_tier, requires_confirmation, trajectory }
```

证据：`crates/gen-zero-planner/src/pipeline.rs:521`、`:534`、`:549`、`:565`、`:572`、`:580`、`:605`。`PolicyGate::evaluate` 始终转发 `active_context=[]`、`certificate=None`（`crates/gen-zero-gate/src/policy.rs:258`）。

**这不是前后门禁与算法“硬解耦”的实现。** 每个引擎都依赖具体 `PolicyGate` 并重复调用它；同时完整安全上下文又没有穿透到内部。当前兼具重复开销与契约不完整两种代价。

### 5.2 0–1 ILP、Sheaf、Nanocore 分别实现到哪一步

**0–1 ILP：** `policy.rs:145` 遍历规则，`constraint.rs:100` 计算给定 candidate 加 active context 的左端值并与 rhs 比较。检查一个已给定赋值是可行性检查，不是求解整数规划。没有优化变量搜索、bound 或 infeasibility proof。CP-SAT 引擎也没有调用外部求解器。真正的 CP-SAT 至少区分 OPTIMAL、FEASIBLE、INFEASIBLE、UNKNOWN；超时未找到解不能伪装成证明不可行。[OR-Tools CP-SAT status contract](https://developers.google.com/optimization/cp/cp_solver)

**Sheaf：** 确有实质性数值实现，不应误报成纯占位。`crates/gen-zero-gate/src/sheaf_gate.rs:398` 重新计算 residual、energy、gradient，并检查诊断值／步长；它自己明确只证明给定 problem 下的 terminal compliance，不证明 relaxation history。`policy.rs:179` 对注册的 problem 只读验证 certificate，缺失则 hard stop。这是有价值的检查，但没有证据把这个 terminal state 绑定到 planner 的 `world_model.step` 预测、实际执行动作、当前物理观测和完整轨迹。

**Nanocore：** 在本次审计的 `engine/pipeline/policy` 链上没有 Nanocore invariant 调用或证据参数。service 的 pipeline 分支也只组装 world model、PolicyGate、LodGraph（`crates/gen-zero-service/src/pipeline_verb.rs:35`、`:51`）。其他服务路径存在 Nanocore 不能填补此处契约空洞。service 的 audit metadata 还明确写 `formal_certificate: "unavailable"`（`crates/gen-zero-service/src/zero.rs:1560`）。

### 5.3 已复现的边界问题

| 编号／等级 | 证据与真实观察 | 判断 |
|---|---|---|
| F01 / P0 | `constraint.rs:105`–`:114` 使用 i32 乘加。release 探针构造同一 action 的两个 `i32::MAX` 项、rhs=0：数学左端 4294967294 > 0，实际 gate 返回 Tier0Proceed | **已复现的拒绝失效**。配置来源是否可被远端控制另当别论；合法公开结构能表达该输入。debug 溢出 panic 也不是合格的 typed fail-closed |
| F02 / P0 | `engine.rs:208` reward 未校验直接 `add_value`；探针 NaN reward 返回 `Ok(ActionId(1), H≈0.0659)` | 不仅污染决策，还给出低 entropy，不能把“模型返回 Ok”理解为模型输出可信 |
| F03 / P0 | MCTS 忽略 successor；CFR 只取 `.1` reward（`engine.rs:692`）；NaN successor 两者都返回 Ok | pipeline 输入校验不能覆盖模型输出；现有真实模型主动拒绝 NaN，会掩盖可替换模型的这个缺陷 |
| F04 / P0 契约缺口 | `evaluate_basic` 下互斥规则通过；带另一个 active action 的 `evaluate_with_context` 则 HardStop。planner 一直用空上下文 | 不能保障并发动作互斥／运行中配额。不是 checker 不会检查，而是入口没有传入事实 |
| F05 / P1 可用性 | 注册 heat requirement 后 pipeline 无证书字段，全部相关动作被剪掉（已有 pipeline test） | 此处是安全拒绝而非 bypass；但“支持带证书规划”未完成 |
| F06 / P0 动态安全语义 | 合成模型给 action 1 reward=10 且 done=true，action 2 reward=0 且 done=false；六引擎经 pipeline 全部选择 action 1 | `decide` 的 Tier0 是规则允许，不是模型轨迹安全。`pipeline.rs:7` 自称 done 视为 hazard，但决定动作时没强制这一条件 |
| F07 / P1 诊断语义 | 禁止动作的 `simulate` 返回 gate=HardStop、steps=1、`is_safe=true`。`pipeline.rs:710` 在 `:743` gate_check 前调用模型；is_safe 只看 hazard（`:90`） | 证明“所有方法先剪枝”的广义宣称不成立。simulate 是想象，不证明外部动作已被执行；若允许违反政策的反事实模拟，必须显式区分 policy_allowed 与 hazard_free |
| F08 / P1 旁路 API | `GenZeroPlanner::evaluate_reflex` 直接返回 `action_slice[0]`，没有 gate（`crates/gen-zero-planner/src/lib.rs:51`、`:61`） | legacy 公共接口存在无门禁建议路径；不要错误声称 ProductionPipeline 的 Reflex 使用此实现，它用的是 gated CpSat wrapper |
| F09 / P0 直接路由入口 | `router.rs:120` 因非法 entropy 得 HardStop，却在 `:77` 选择 K2；后续引擎重新用 entropy=0 检查。探针对 NaN 和 1.5 的 entropy 均返回 Ok | **已复现的上下文丢失导致拒绝失效**。ProductionPipeline 已检查范围，不受同一路径影响；direct router 和 legacy lookahead 仍暴露此契约问题 |

独立探针源：`planner_audit_evidence/probe.rs`（历史证据，当前提交未包含），原始观察：`planner_audit_evidence/probe-release.txt`（历史证据，当前提交未包含）。F01/F02/F03 不是“可能存在”的猜测。F09 补充原始观察：`planner_audit_evidence/faults-extra.txt`（历史证据，当前提交未包含）。

补充边界：`audit_action` 缺 safety estimate 时会拒绝，这是实质性保护；但 uncalibrated estimate 只增加 reasons（`pipeline.rs:493`），不自动阻止 Approved。默认 margin estimate 的 `calibrated=false`（`crates/gen-zero-worldmodel/src/dynamics.rs:121`）。应把“模型内 margin 通过”和“实测风险经校准批准”拆成不同类型。

### 5.4 用户提出的三类收敛风险：必须分清现实与假设

**a) MCTS 的 PUCT 价值失真／死胡同惩罚。** 当前 gate 在根部过滤，uniform prior 在可行动作上重归一（`engine.rs:130`、`:148`）；没有深层树，也就没有已实现的“扩展后 gate 剪枝导致深层死胡同探索惩罚”机制。现在的问题是只有即时 reward，不能看见未来死胡同。对于未来真正的 constrained MCTS，若合法集是固定、正确的 `A_safe(s)`，在约束后的 MDP 上搜索不自动构成价值偏差；偏差来自把 gate error 当普通低奖励、对 rejected samples 忽略分母、状态变化仍沿用旧 mask、把 infeasible 与 unknown 混为一谈。应明确定义 illegal terminal/成本、backup 语义、剩余预算与 viability，并用相同约束问题作基准，不能拿 unconstrained optimum 指责安全裁剪“损失最优”。

**b) CFR 的非凸约束／Nash 收敛。** 当前没有 CFR 迭代，故不能谈“剪枝破坏已有 Nash 收敛”。一般地，删去固定非法纯动作后，剩余纯动作的 mixed-strategy simplex 仍是凸集；“原始动作域非凸”本身不推出 CFR 不收敛。真正危险的是约束依赖隐藏状态、在同一信息集提供不同合法集合导致信息泄露／抽象失真，或双方共享资源产生 coupled feasible sets，此时问题可能是 constrained/generalized Nash 而非标准二人零和博弈。标准 CFR 的平均策略保证有博弈假设，不是单次 argmax 的保证。[原始 CFR](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)；[Last-iterate Convergence in Extensive-Form Games](https://arxiv.org/abs/2106.14326)

**c) CEM 的粒子拒绝／协方差退化。** 本实现没有高斯粒子或协方差，不能诊断“已经发生协方差退化”。它会拒绝包含禁用动作的整条轨迹（`engine.rs:504`），零存活即 NoFeasibleAction（`:529`）。在 pipeline 先去掉静态禁止动作后，内部同一静态 gate 通常不会再因这些动作拒绝；直接调用 engine 才更容易浪费采样。未来状态依赖的可行性若每步接受率近似 p，独立近似下整条 H 步轨迹存活率为 p^H；这是说明 sample starvation 的模型假设，**不是本次测得的概率**。真正 Gaussian CEM 在 elite 数 m 小于维度 d+1 时，样本协方差 rank≤m−1，这是线性代数条件，不是本仓库功能。应采取 feasibility-aware proposal、分阶段约束、平滑／协方差下界与 minimum feasible elites；修复／投影之后必须重新认证，不能用软惩罚替代最后硬检查。

### 5.5 什么边界设计才合理

保留独立、安全优先、不可被优化器 override 的**最终裁决**；同时为搜索提供纯函数式／快照绑定的 feasibility oracle，才是合理折中。只在入口和出口检查 action id 无法保障轨迹；把所有安全修复混进不可审计的优化器也不可取。

建议 contract：`PlanningProblem{state, goal, action_space, model_version, policy_snapshot, active_context, agent_id, budget}`；`PlanCandidate{trajectory, predicted_states, objective, uncertainty, status, evidence}`；`GateResult=Allowed(certificate)|Rejected(reason)|Unknown(reason)`。Unknown 一律不能授权执行，不能暗中变成 first action 或 zero reward。修复器负责提出修复后的 candidate；validator 只验证，不偷偷改状态。执行前绑定 observation/model/policy epoch、action digest、资源 reservation，再做最终验证，解决并发状态与 TOCTOU 问题。此设计是建议，尚未实现，也未被证明全局最优。

## 6. Router：规则调度器，而非学习到的动力学 MoE

`crates/gen-zero-planner/src/router.rs:75` 直接按外部 entropy、worst gate tier 与固定阈值 0.2/0.7 做 if-else；K1 调 A*（`:141`），K2 gate filter 后 MCTS（`:145`），K3 顺序执行 MCTS、CEM、A*（`:175`）。没有模型 Jacobian、可控性、reward landscape、分支因子预测、在线性能反馈或 learned gating 网络。

它确实随输入 entropy 改变 route，因此不能说“完全不动态”；准确描述是**固定阈值的输入条件调度**。Auto 只有四种内部组件，没有路由到 CFR 或 GFlowNet（`:27`）。因此也不是“六专家自适应竞争”。

K3 权重 2:1:1，三个成员先后调用，不并行；当 MCTS 支持 A、另两者支持 B 时 2:2 平局依请求顺序打破（`:185`）。三个算法共享模型、相似的即时目标，错误强相关；“委员会”不自动增加证据独立性。成员 Err 用 `?` 使整体失败，这部分确实 fail-closed（`:173`）。

entropy 也不是统一的风险尺度：MCTS 是访问频率熵，A* 是任意温度 Boltzmann 熵，GFlowNet 是另一个温度的 softmax，CFR 是正 regret 熵，CP wrapper 恒 0，K3 原样返回输入 entropy（`:195`）。`pipeline.rs:221` 所谓 engine own entropy 对 K3 不成立；最终 gate 又用 request entropy 而非 engine entropy（`:574`）。不能用同一阈值赋予这些量相同的“置信度”含义。建议分字段报告 observation uncertainty、model uncertainty、search uncertainty、disagreement，并对任务数据校准。

## 7. 2.0ms 与高并发内存：承诺核验

### 7.1 没有实际 2ms deadline

在 `crates/gen-zero-planner/src` 的 deadline/timeout/Instant 搜索中，只有 `error.rs:13` 的 `TimeoutExceeded` 错误定义；未见生成该错误的计时路径。`PlanningEngine::plan` 没 budget/deadline；pipeline 同样没有。`PlannerConfig` 还允许 50,000 次 simulation、10,000 CEM samples、100 horizon（`config.rs:69`、`:81`、`:87`）。这是 work-count 上界，不是 wall-clock 上界。

本次让两次 `step` 各 sleep 3ms，A* 仍在 **6.295ms 后返回 Ok**。sleep 故障是 deadline 契约探针，不是正常性能样本。由此足以否定入口有通用 2ms 硬超时。

默认 K3 在无 done 时约需 `128 + 32×3×4 + K = 528` 次模型 step（K=16），未计 gate 与分配。若总预算 2ms，平均每步及其摊销开销只能约 3.79µs。这是调用计数推导，不是 WCET。当前 CEM 本机测量约 5.8ms 已超预算。

2ms 不是对所有树搜索不可逾越的数学禁令：小问题、预计算、warm start、批量模型调用、可验证 incumbent 都可提供 deadline 下的有限质量结果。**“任意问题 + 多步全局最优 + 统一2ms”不成立。** 对硬实时，应分离固定预算安全响应层与 anytime 规划层，逐步传 deadline/cancellation，保留已认证 incumbent，timeout 返回明确状态。同步 `step` 若能阻塞，仅在循环外检查时间也不能给硬时限；必须限制模型执行、资源调度及 preemption 机制。不得在 timeout 时退回未经验证的首动作。

### 7.2 64-byte 为真，“整条零分配／无锁”为假

`crates/gen-zero-planner/src/tree.rs:15` 的 repr(C, align(64)) 与 `:52` 静态断言、本次 `size=64 align=64` 一致。字段中的 64-byte 是节点 metadata，不包含 1024×f32 的 latent state。一个 cache line 的布局不能推出完整搜索的缓存命中率或吞吐率。

TypedArena::new 用 `alloc_zeroed`（`:173`），每次 MCTS plan 都新建默认 1024-capacity arena（`engine.rs:144`），即约 64KiB 节点存储；16 个候选当前只用 17 个节点。arena.alloc 不逐节点 heap allocate，但每次持有 Mutex（`tree.rs:198`）；发布后的 get 用 Acquire（`:220`）。这应称“预分配、串行分配、无锁已发布读取”，不是 lock-free allocator。节点 value 用 CAS 不等于整个搜索并行。

MCTS 没连 first_child/sibling：`engine.rs:160` 的分支为空，后续靠 index+1 寻址；state_hash/hot/cold state 字段并未形成实际 state arena／转置表。`act.0 as u16`（`:153`）可截断 u32 ActionId；当前返回动作取自独立 valid_actions，所以不能虚报为已观察到错选，但未来若靠 node id 回溯会存在信息损失。

## 8. 本次执行证据与真实性限制

### 8.1 实际测试

命令：`cargo test -p gen-zero-planner -p gen-zero-gate`，原始退出码 **0**。gate unit 43；planner unit 23；numeric integration 7；latent model integration 5；config integration 6；pipeline integration 28；合计 **112**。两个 doc-test 集各 0。原始日志：`planner_audit_evidence/cargo-test.txt`（历史证据，当前提交未包含）。

这不是 112 个算法性能验收；例如 MCTS 的“真实模型 NaN 拒绝”能由模型自身拒绝实现，而不是引擎完成统一输出检查。探针以 `Ok(NaN)` 明确暴露了这个差异。没有运行整个 workspace，也没有把 service live acceptance 当作已完成。

### 8.2 Release 单引擎延迟／分配

环境：x86_64 Linux VM，24 个可见 vCPU，Microsoft hypervisor；rustc 1.96.0；release optimized；16 candidates、1024维零初态、默认引擎参数、默认 `LatentDynamicsWorldModel`、空 PolicyGate。每引擎 warmup 20 次，测量 200 次；nearest-rank P50/P95/P99。使用 `black_box` 保留结果，所有 plan 必须成功，否则程序 panic。自定义 global allocator 统计 alloc/alloc_zeroed/realloc 调用；**不是存活对象数或字节数**。

| 引擎 | P50 µs | P95 µs | P99 µs | max µs | alloc/realloc calls / plan |
|---|---:|---:|---:|---:|---:|
| MCTS | 1774.930 | 2167.115 | 2679.594 | 2892.086 | 17 |
| A* | 246.390 | 329.587 | 409.684 | 473.682 | 51 |
| CEM | 5822.570 | 7283.112 | 8337.970 | 9431.527 | 1440 |
| GFlowNet named ranker | 257.390 | 318.888 | 368.086 | 548.578 | 48 |
| CFR named ranker | 216.692 | 264.389 | 275.989 | 291.889 | 16 |
| CP-SAT named ranker | 233.090 | 301.588 | 349.587 | 423.283 | 48 |

MCTS 的 17 次可由 16 个 gate verdict reason String + arena 解释；A* 还有成功路径 `format!` 与 heap 扩容；CEM 在 384 次 transition 中反复构造 label String、successor label 等。这与“SmallVec/stack array”宣传不矛盾：局部容器不分配不代表被调用链不分配（`engine.rs:45`、`:491`；`policy.rs:247`）。

### 8.3 请求并发和 shared arena 压力探针

同一个 Arc<ProductionPipeline>、Auto K3、每 worker 50 次请求、barrier 同时启动；统计 wall time，包含 join，不含线程创建。没有 HTTP／序列化／真实模型服务成本。

| 请求 workers | 总请求 | elapsed s | requests/s |
|---|---:|---:|---:|
| 1 | 50 | 0.408776 | 122.32 |
| 4 | 200 | 0.406330 | 492.21 |
| 8 | 400 | 0.413471 | 967.42 |

**外层并发有实际吞吐扩展**，值得保留，但不等于一棵树内有并行 rollout，也不能声称 2ms 响应。

shared TypedArena 预分配 100,000 节点，1/4/8 workers 竞争 alloc；只计填充阶段和 join，排除 arena 创建及释放：

| workers | nodes | elapsed ms | nodes/s |
|---|---:|---:|---:|
| 1 | 100000 | 2.663 | 37551718 |
| 4 | 100000 | 12.129 | 8244689 |
| 8 | 100000 | 18.251 | 5478858 |

该负载下锁竞争显著，增加线程反而下降；这支持“不能由原子 cursor 推导高并发无锁分配”。它没有测共享节点 backup、NUMA，也不是当前串行 MCTS 的真实 tree contention。

**限制：** 单次短测、未绑核、无独占机器、无长期 soak，global allocator 的原子计数也引入开销，尤其并发时可能产生共享计数器竞争。200 个样本的 P99 只是本轮经验分位数，不能作 SLA/WCET。默认动力学虽然是仓库真实实现，其业务内容仍是合成公式：`next_i=.95*s_i+.05*sin(.05*i+.17*a)`，reward 是 sum(next_i)*.001 截断（`crates/gen-zero-worldmodel/src/dynamics.rs:70`）。不能把这些数字推广到真实语言世界模型、GPU inference 或环境执行。

历史复现命令（脚本及 lock 未随当前提交发布，不能在此 checkout 直接复跑）：`python3 docs/architecture/planner_audit_evidence/reproduce.py`。脚本在新临时目录构建，保留完整日志及退出码，固定本次依赖 lock；生产源不变。`reproduce.py --faults-only` 已在新临时目录用 `--locked --release` 编译运行，退出码 0。性能会随机器／调度变化，不应要求与表中逐位相同。

## 9. Python 11/12 项到 Rust 六引擎：哪些能力真丢了

“12 planners”本来就混合了引擎、模型、校验器与对手分析器；不能按数量作等价合并。当前 Python client 已把 `bidirectional_planner` alias 到 AStarEngine、`continuous_mpc_planner` alias 到 MpcCemEngine，同时保留 text world model、PRM、D-SCM 和 bluff detector（`python/gen_zero/client.py:465`、`:574`）。这说明“收敛模块”可以做到保留子能力；Rust 是否保留必须逐条验证。

| Python 模块 | 原代码真实能力与局限 | Rust 迁移判断 |
|---|---|---|
| Bidirectional Search | 两个 heap/frontier、交汇检测、反向动作映射和路径拼接，确实进行图搜索；正确性仍依赖反向转移／代价语义。`python/gen_zero/planner/engines/astar_engine.py:327`；client 转发 `client.py:1604` | **真实多步路径搜索能力未保留**。Rust A* 无 goal/frontier expansion，不能称功能等价。可归入未来 graph-search backend 的选项，不必独立部署一个 engine |
| Text World Model | 明确是未训练、未校准的规则模型；string 分支只拼接 `" -> action"` 并给固定 reward、单步结束（`python/gen_zero/world_model/text_world_model.py:1`、`:84`） | string/dict 环境适配语义丢失，但不能夸大为“丢失已训练语言世界模型”。应恢复 typed adapter 与受测语义，而不是恢复虚假模型质量承诺 |
| Continuous Latent MPC | 确有 N×H×D Gaussian 轨迹、bounds clipping／simplex projection、分时间步 mean/std 更新；但模拟 `curr_z += .1*act`，并未使用传入 latent model（`python/gen_zero/planner/engines/mpc_cem_engine.py:107`、`:125`、`:154`、`:166`） | **连续动作参数化、bounds/simplex 与时间条件分布确实缺失**。Rust 反而有真正逐步调用 `WorldModelDynamics`，不是所有维度都倒退。下一步应把连续 action 接入真实模型，不是照搬 Python 假动力学 |
| PRM | 此仓库指 **ProcessRewardModel**，不是机器人 Probabilistic Roadmap。是 fatal reward、flood-fill pocket、金融止损和文本关键字规则（`python/gen_zero/model/prm.py:65`）；可由调用方注入 MCTS invariant lambda（`client.py:1002`） | 特定 domain 的 transition verifier 能力没有等价迁移；它属于 safety/model scoring plugin，不应按“不是规划器”就把语义扔掉，也不能称其为 learned PRM |
| D-SCM / Bluff detector | 手写结构方程、noise abduction、锁定 noise 的反事实模拟（`python/gen_zero/multiagent/decentralized_scm.py:68`、`:168`）；意图推断是 logistic/threshold 规则（`:207`） | Rust CFR 单次 payoff 比较无法替代 causal intervention、opponent state 与 intent features。**接口能力丢失，真实性／校准原本也有限**。适合作为独立 opponent/causal model 服务，不属于均衡求解器内部的同义替换 |

这里的“不可替代”是相对于当前 Rust 六个公开契约而言：不能不添加信息就重建这些能力；不是说这些 Python 实现是唯一算法或值得逐行移植。

### 9.1 历史基准为何不能证明 12→6 无损

- `run_12_planners_benchmark.py:156` 的 Text World Model 用 grid 字典，不测试任意自然语言；`:168` 的 MPC-CEM Trajectory 用 BUY/HOLD/SELL，走离散分支。
- 连续 MPC 成功条件只是输出 action 长度为 4（`:248`），不是目标收益、约束保持或动态模型正确性。
- PRM 测的是 generic pos 字典（`:258`），不触发主要领域规则；D-SCM 提供恰好模型一致的 next state 及 revealed_strength=0.20（`:272`、`:282`），成功判定是预设 bluff 类型／阈值（`:286`）。
- `benchmarks/results/r4_evidence/planners_after_report.json:2` 明示 `algorithm fixtures; does not establish learned-model quality`。其中 50 次 fixture 的 Bidirectional 100%/mean 0.059ms、Text 100%/0.028ms、离散 MPC 100%/1.852ms、连续 MPC 100%/3.808ms、PRM 100%/0.001ms、D-SCM 100%/0.036ms，是**历史记录，非本次复跑**（对应行 129、251、312、556、617、678）。这些口径完全不同于第 8 节 Rust engine timing。
- `benchmarks/results/planners_and_constraints_fixed_results.json:59` 明确 strict run 仅 **10/12，exit=1**；CP-SAT 50 次成功率 96%（`:418`），并非 12/12；`:60` 还记录历史 2.0ms 测试 **127/1000 fallback 和 wall deadline miss**。此数据不可移植为当前 Rust failure rate，但必须保留，不能只摘录成功项。
- 六引擎 Python benchmark 脚本指定输出 `python/results/gen_zero/issue_93_6_planners_benchmark_report.json`（`run_6_orthogonal_planners_benchmark.py:232`）；本 HEAD 未跟踪该结果文件。脚本存在不等于已完成实验。
- Rust arena 历史吞吐并没有可用的同口径 benchmark；`README.md:121` 已撤回旧 node-allocation／planning SLA 数字，`benchmarks/suites/latency_suite.py:108` 将 in-engine MCTS 标为 `not_measured`。本次新增探针填补局部观测，仍没有填补生产验收。

## 10. 数学承诺的最小反例与证明义务

### 10.1 单步接口不能推出全局路径最优

构造根状态两个动作：A 即时收益 1，之后所有路径总收益 −100；B 即时收益 0，下一步可得 100。若两条首步 state displacement 一样，当前 A*/GFlowNet/CFR/CP wrapper 都偏向 A；MCTS 只比较根的即时 reward，也无法由更多 simulation 得知第二步收益。此为根据读取到的目标函数给出的**解析反例**，不是声称已执行的新实验；第 8 节执行的是独立的 immediate terminal-hazard 反例。CEM 可在有限 horizon 看到一些后续收益，但 horizon 截断、采样与共享分类分布依然不提供全局最优保证。

### 10.2 一个纯动作不能代表通用 Nash 解

在零和石头剪刀布中，均衡是均匀混合；固定选最大 regret 的纯动作可被对手利用。当前 CFR 输出一个 ActionId 和 entropy，没有输出并执行其 mixed strategy，也没有对手 payoff matrix 或 extensive game。哪怕三项 payoff 相同，它也只是选择第一个合法动作并返回高 entropy，**高 entropy 元数据不等于行为真的随机化**。

标准反事实遗憾要累计信息集的 action advantage，并按对应 reach probabilities 定义；有限 regret bound 推导的通常是平均策略 exploitability，不是一次向量减均值。在约束改变时，应证明求解的是哪个受约束博弈，不能要求恢复被规则排除的 unconstrained Nash。

### 10.3 Softmax 分数不是 trajectory balance

Trajectory balance 需要正 terminal reward、前向／反向路径概率与归一化量，典型目标：

`L_TB(τ) = [log Z + Σ log P_F(s_t|s_{t−1}) − log R(x) − Σ log P_B(s_{t−1}|s_t)]²`。

reward-proportional sampling 的结论依赖目标／支持集等条件，而不是“计算 softmax”本身。当前 Rust 没有这些量，也没有学习或采样动作：它直接 argmax。混合曲率 geometry 工具存在于其他模块，同样不证明这里实现了 manifold GFlowNet。[Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

### 10.4 证书必须对准所宣称的命题

`||Ds-b|| <= ε` 可以证明给定矩阵／边界下的残差小；没有额外建模与误差界，不能推出动作在真实环境无伤害、长期约束可保持、目标必可达。一个整数不等式 checker 也只能证明所提供赋值满足所注册约束，而且首先必须保证算术不溢出。审计不接受“有数学公式→整个系统形式化”的推理跳跃。

## 11. 下一步架构：按能力与证据演进，而非凑足引擎数

以下是建议路线，**全部属于未完成工作**；本次只交付审计，不把它们写成已实现。

### P0：先封住真实 Fail-Closed 破口

1. 约束编译／注册校验 coefficient、重复项和上下文容量；乘加用 checked arithmetic 或足够宽且证明有界的整数，溢出必须 typed reject。对 debug/release 两种 profile 验证相同拒绝结果。
2. 所有 engine 统一检查 input、successor、reward、累计值和内部评分。`Ok` 不能免除校验。给 MCTS/CFR 加入 NaN/Inf/overflow 模型输出案例；模型 Err、非法数据不得变成低 reward、跳过失败成员或首动作。
3. 统一入口验证，封闭 legacy reflex 和 direct router 绕过完整 gate 的公开路径；必须明确“建议”与“授权”的类型差异。
4. 区分 `Terminal::GoalReached`、`Hazard`、`Truncated` 和 `Unknown`。决定动作必须调用 state/transition safety contract；若只提供 policy eligibility，则字段和 API 文档不得叫 certified safe。
5. 把 active resource context、agent identity、model/policy epoch 与候选证书贯通；缺少必须事实时 Unknown/Reject。高并发配额采用原子 reservation/commit，不能多个请求各自检查空上下文后同时通过。
6. `simulate` 若保留违规反事实能力，显式返回 `policy_allowed=false`，避免 `is_safe=true` 的混淆；不能把它输出为可执行授权。审核 `audit_action` 的 Approved 是否允许未校准安全读数，并以契约区分保证级别。

**验收门槛：** 本文 F01–F09 的独立反例在应拒绝的公共入口必须稳定拒绝；非法输入不能产生无标记 Ok，不能靠默认模型自行报错掩盖引擎缺陷。

### P1：建立可表达真实问题的核心契约

把 world model、proposal、search、verification、execution 五项职责分清。不是拆成五个网络服务，而是明确可替换边界与证据流。

| 责任 | 应补契约 | 验证指标 |
|---|---|---|
| Dynamics | discrete/continuous/hybrid action、batch step、终态原因、模型版本和校准误差 | held-out transition error、校准、模型失配下拒绝率 |
| Problem | goal、合法动作生成、成本、horizon、玩家/信息集或 belief | 小规模精确 oracle 对比、语义一致性 |
| Search | anytime budget、persistent workspace、candidate trajectory、objective bounds/status | quality-vs-budget、最优性 gap、deadline miss |
| Safety | stateful feasible oracle、certificate request/validation、Unknown | 反例覆盖、资源互斥、证书绑定／过期拒绝 |
| Execution | final recheck、reservation、confirmation、safe fallback | TOCTOU／撤销竞态、真实执行 trace |

### P1：补算法或诚实删除算法身份

- 把当前 shallow MCTS/A*/GFlowNet/CFR/CP wrapper 改为描述实际行为的 baseline 名称，或对外标记 approximation。可以共享 `OneStepScorer`，避免维护多个数学上等价的实现。
- 真 MCTS：successor tree、terminal/value backup、合法 mask、转置与状态存储、可复用搜索、batch rollout；并发 correctness 先于吞吐。没有价值模型时明确 rollout 策略和误差。
- 真 graph search：goal predicate、g+h、admissibility 声明、重复状态/reopen、path reconstruction；双向只是 backend variant，需要正确逆转移。
- 连续 MPC：连续动作 space、每时间步 mean/covariance（或对角 std）和真正模型 dynamics、bounds/manifold adapter、warm start、递归可行性；保留 categorical backend，别用连续 latent 冒充连续 control。
- 真 CFR 只在确有 imperfect-information game 时建设；补信息集／玩家、counterfactual reach、累计策略、平均策略与 exploitability benchmark。否则删除 Nash 承诺，保留 greedy baseline。
- 真 GFlowNet 只在需要多样性／reward-proportional proposal 且有训练数据时建设；给出 TB/flow loss、采样一致性和 diversity 指标。无需让它独立作安全裁决。
- 真 CP-SAT 如需组合约束最优解，接有状态／界的 solver backend；区分 feasible、optimal、infeasible、unknown 和 timeout。简单 action eligibility 放回 shared verifier，避免把验证器列为“第六类独立规划范式”。

### P2：按预算调度和真实性基准优化

- Router 先成为可审计的 budget/capability scheduler：基于 problem 类型、action cardinality、model latency、horizon 和安全要求路由；有数据后再评估 learned gating 是否优于规则。不要先升级 MoE 名称。
- 小步动作响应与后台 anytime search 分离。每个 deadline 下只返回已认证 candidate；没有可认证动作就明确拒绝／请求接管。所谓安全 fallback 必须有自己的可行性证据。
- workspace/arena 池化、预编译静态 mask、success-path lazy formatting、`step_batch` 接入；再做 1/4/8/16 workers 的稳定负载对比。当前锁保护发布正确性不可为了吞吐直接移除。
- 性能验收同时报告质量与拒绝率，禁止通过减 horizon、关 gate、吞 error、偷偷用默认动作来满足延迟表。

### P3：有需求与证据后接入缺失能力

Diffusion proposal、goal-conditioned value／可达性 oracle、PRM transition verifier、causal/opponent model 都作为有版本、有输入语义的插件；SOTA 不是算法清单。只有明确任务对照表明增加它改善成功率／安全／预算 Pareto 前沿，才进入默认路由。

### 11.1 建议验收矩阵

| 工作负载 | 必需 ground truth／对照 | 不能作弊的验收条件 |
|---|---|---|
| 延迟奖励与陷阱图 | 小图 exhaustive DP/Dijkstra、固定 successor 模型 | 报最优路径 gap，禁止只检查 action 属于候选 |
| 随机／POMDP | 已知概率 toy model、belief oracle、多 seeds | 校准／置信区间、风险预算、模型调用数 |
| 连续控制 | LQR 可解析解、受约束小系统、非凸障碍 | trajectory cost、constraint violation、feasible elite ratio |
| 二人零和不完全信息 | Kuhn/Leduc 等可算 exploitability 的固定规则 | 平均策略 exploitability，禁止“返回合法动作”作为 Nash 验证 |
| 硬约束与资源并发 | 穷举小 ILP、两请求竞争同一资源、故障注入 | 每次授权可复核；溢出／超时／缺证一律不授权 |
| 生成式 proposal | 固定 reward target 与 sampling baseline | 多样性、分布误差、约束成功率、训练/推理成本分列 |
| 真实任务迁移 | Python/Rust 相同模型、相同 gate、相同预算与 seeds | paired 成功率／质量，禁止跨机器历史 latency 直接比值 |
| 性能 | 长时混合负载、模型慢请求、取消、OOM/容量限制 | P50/P99/P999、wall deadline miss、吞吐、分配字节与质量一起报告 |

## 12. 交付状态与剩余范围

**已实现（本次交付）：** 报告、源码定位、六引擎公式对照、Python 能力丢失映射、112 项现有测试实际运行、release 数值／门禁故障探针、六引擎 latency/alloc 观测及请求／arena 并发压力探针。只读侧线已完成且报告 workspace/HEAD 一致，主线程核实关键证据并负责裁定。

**未验证：** 真实生产模型和环境、长期负载、安全执行器、全工作区集成、当前行业最优名次。本文引用论文用于界定算法和保证条件，不用于宣称本仓库已具备该能力。

**未完成／致命缺陷：** 本文列出的生产代码问题没有在审计任务中修复；它们不会因报告写完或测试全绿而消失。上线／安全签收不得依赖“六正交引擎”“形式化完备”“2ms硬实时”“全路径零分配”的未兑现表述。
