# Gen-Zero 能力边界、8B 归属与提升路径调查

调查日期：2026-09-27。源码基线：`6dfd6099591f68a8bba8bbb8ae3b2a6c302f7e2c`，工作目录 `/ebs/pj/gen-zero`。开始时工作树干净。正文源码行号以该基线为准，可用 `git show 6dfd609:<path> | nl -ba` 核对；调查中其他工作合并到 `701d800a5d3c55db6dcdb002b7bda5608a6bba11`，增量说明见末尾，不能把本文旧行号直接当新版本行号。本次只新增调查报告与证据，不修改生产实现，不提交，不执行 stash/checkout/reset/clean/force-push。

## 结论先行

1. 用户判断正确：原文的“8B”指外部 CLM 的冻结 Qwen3-8B 编码器，不是 Gen-Zero 自研的 8B 打分头；严格说，CLM 自己也不是“一个 8B 参数的投影头”，而是 8B 编码器加两个较小的投影头。
2. Gen-Zero 的合理定位是：将候选生成、状态建模、受约束搜索、风险决策与执行验证连接起来的决策控制系统。当前代码拥有相关算法构件，但不能据此宣称已经具备通用终端世界模型、任意 shell 安全证明、或 90%+ 解决率。
3. 原文遗漏了最重要的失败证据：已有增强 PRM 实验，BoN=4 从基线 31/38 降到 29/38，修正 2 题、退步 4 题，几何分支未胜出。继续在这 38 题上调到满意不能再算独立留出验证。
4. 最值得投入的方向是“真实状态—候选—受控执行—观测—再规划”的闭环，以及可重放、可拒绝、不可绕过的执行门禁。堆更多规划算法或把几何特征加进外部评分器，均不能替代这条链。
5. “同基座增加 10%–15%”“冲刺 90%+”只能列为待检验目标，不能作为已知收益。现有证据连本系统同协议的 80% 真机起点也未建立。

## 1. 原始说法逐项判定

| 原说法 | 调查判定 |
|---|---|
| 8B 打分头是 Gen-Zero 的 | 错。外部 CLM 的 encoder 为 8B，head 是另一个规模；Gen-Zero 有自己的小型评分/决策模块，也可使用外部大模型表征。 |
| CLM 只是“弱打分头” | 不严谨。它是非生成式状态—动作评分系统；0.5196 是特定本地留出轨迹的 pooled AUC，不足以概括模型所有任务的能力。 |
| 多纠正 2 题就 86.8%，即可压制 81.6% | 33/38 的算术正确，工程和统计结论不成立；必须扣除新增退步，控制候选和预算，并保持未见测试集。 |
| 几何特征平滑方差可提升 | 待检验假设，且已有一次反例：增强头 29/38。不能把几何平滑等同于正确排序。 |
| Harbor/Pier、容器、adapter 100% 就绪 | 只能按环境、版本、具体任务与证据分层陈述；不能由 smoke 或边界测试推出完整生产链路及解决率。 |
| ILP 防误删、防死循环 | 只有在动作与资源语义被正确编码、规则注册、执行器不可绕过时才可能覆盖指定风险。线性约束本身不理解 shell，也不证明任意程序终止。 |
| 真正多步 MCTS 已能改善编码成功率 | 算法与实际调用链需分别验收；树深度不等于预测环境状态正确，语义价值不等于真实 reward。 |

CLM 官方模型卡明示 frozen Qwen3-8B + state/action heads，并说明没有生成能力、verifier 指标来自微调头而不是公开通用 checkpoint 的零样本性能：[官方模型卡](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)。本次抓取副本见 `docs/benchmarks/evidence/gen-zero-audit-20260927/clm-official-model-card.md`。这些数字作为对方声明保留，不升级为我们独立复现的全链路结果。

## 2. 38 题实验暴露的真实瓶颈

历史结果见 `benchmarks/results/deepswe_prm_evidence/RESULT.md:1`，机器可读结果见 `benchmarks/results/deepswe_prm_enhanced_results.json`，失败验收命令及退出码见 `benchmarks/results/deepswe_prm_evidence/acceptance.json`。

| 指标 | 原 CLM 基线 | 增强头 |
|---|---:|---:|
| BoN=4 | 31/38 = 81.5789% | 29/38 = 76.3158% |
| 轨迹级 AUC | 0.519595 | 0.413964 |
| 轨迹级 Spearman | 0.029954 | -0.131521 |
| 相对基线逐题变化 | — | 修正 2，退步 4，净负 2 |

这说明“增加两个正确选择”并非净提升。AUC 在全体轨迹上比较，BoN 在同一任务的候选间选择；两者对象不同。低 pooled AUC 可以与较高任务内 BoN 共存，不能由此推导“随便加点特征就能超过”。

`RESULT.md:38` 记录 4 个任务所有候选均失败：按当前每题可用候选（released/up-to-4 协议，评测代码使用 min(4,候选数)，并非严格每题都有4条）计算，纯重排上限是 34/38 = 89.47%。因此在这个候选池里，连完美评分器都达不到 90%。要越过上限必须改善候选生成、执行反馈和修复能力。

样本量也不足以支持“压制”：31/38 与 33/38 的 Wilson 95% 区间约为 [66.58%,90.78%]、[72.67%,94.25%]。即使同题配对只有 2 胜 0 负，双侧精确符号/McNemar 检验 p=0.5。这里是解释小样本局限的计算，不是新模型实测。

训练证据描述 75 个训练任务、298 条轨迹、30,372 步；步数不等于独立监督样本。历史交叉验证赢家是 semantic kernel，几何分支没胜出，也没微调 8B 主干或原 MLP（`RESULT.md:12,23`；同目录 `REPRODUCE.md`）。

复现边界：本次重新运行 `python3 benchmarks/tests/verify_deepswe_prm_report.py --report benchmarks/results/deepswe_prm_enhanced_results.json --reference-repo /tmp/clmrepro/repo`，退出 1，因为当前工作树缺少 `benchmarks/artifacts/deepswe_prm/enhanced_rank_head.pt`。因此上表属于已归档报告及 JSON 一致记录，不是本次重新训练/推理得到的结果；完整权重 provenance 校验未完成。没有跳过缺失校验来宣称复现成功。

## 3. PolicyGate 的真实能力与必须先修的问题

Rust 的 `LinearConstraint` 表示 `sum(c_i*x_i)<=rhs`，通过已知 ActionId 及 active_context 求值；见 `crates/gen-zero-gate/src/constraint.rs:13,99`。这是约束检查，不是自动把任意 shell 程序转换成可证明安全的 ILP。`PolicyGate::default()` 的规则、确认动作和 heat requirement 都为空（`policy.rs:43`）；服务默认构造也使用它（`crates/gen-zero-service/src/zero.rs:756`）。不能仅凭返回 `formal_checked` 就认定规则覆盖了真实文件系统风险。

已有可保留的严谨部分：非有限 entropy/阈值拒绝（`policy.rs:130`）；已注册 heat requirement 缺证书就拒绝（`:179`）；语义风险缺失或非法升级而非放行（`risk.rs:29`）；服务只让 Tier0 输出正常成功，确认/升级均为错误结果（`zero.rs:1797`）。Sheaf terminal 验证会重新计算残差与能量，而不是盲信诊断字段（`sheaf_gate.rs:395`）。但它证明的是给定问题和快照下的数值终态合规，不是候选程序语义正确，更不是一般软件安全证明。

Python `DecisionPolicyGate` 是另一套规则/置信度门禁，不能与上述 Rust 实现混称为同一套保证。本次直接调用真实函数，得到以下反例：

- `{}` 缺动作、风险、置信度时，默认 confidence=0.5、risk=0，最终 PROCEED（`python/gen_zero/gate/policy_gate.py:139,266`）。
- 对显式加入 whitelist 的目标，action=`delete`、confidence=0.99、risk=1.0，仍 PROCEED（`:194,219`）。
- risk=NaN 先被改为 1.0，但同样被 whitelist 分支放行（`:152,194`）。

这是纯函数边界反例，没有执行删除，也没有证明 TB adapter 走了这一条 Python 路径；不能将这个局部缺陷扩大成“所有 Rust/TB 门禁均失效”。不过它明确与全系统 Fail-Closed 的承诺冲突。复现脚本与输出在本报告证据目录 `gate_probe.py`、`gate-counterexample.txt`。

另一个成功语义漏洞：`python/gen_zero/runtime/loop_state_machine.py:260` 在模型声称 finish/done 时，仅在 `verify_fn is not None` 才验收；没有 verifier 仍返回 SUCCESS 和“verified”（`:283`）。正确边界应区分 agent_finished、unverified、verified_success。TB adapter 已采用 verifier_pending（`benchmarks/gen_zero_tb_adapter.py:330`），值得统一到其他运行入口。

整改优先级：输入 schema/必填字段验证 → 缺策略或不识别动作拒绝 → 白名单不得覆盖未知/非法风险和硬禁令 → 统一 gate verdict 与执行授权契约 → 验证结论与完成声明分离。程序超时/进程树终止/资源限额属于执行器强制约束，不能只靠模型预测“不会死循环”。

## 4. 如何从 80% 走向 90%：先建立可测的改进闭环

以下全部是建议与验收设计，不是已实现能力或收益保证。

**先固定比较对象。** 同一个 Proposer checkpoint/API 版本、prompt、工具权限、候选数、token、墙钟/成本预算、数据集版本和运行种子。基线应是同基座的标准执行循环；另设等预算 best-of-N/多次重试基线，避免把额外采样带来的收益记到 Gen-Zero 搜索上。不同 DeepSWE/Terminal-Bench 版本、不同轨迹池与不同生成模型的百分比不能直接相减。

**建立最小真实闭环。** Proposer 仅生成候选（命令或补丁及预期效果），不授予安全权限、不决定评测成功；Gen-Zero 管理观测、候选去重与分支多样性、可执行性检查、搜索预算、选支和重规划。执行器在隔离环境中完成动作并回传文件 diff、退出码、测试结果、进程/资源状态。最终成功由隔离的 benchmark verifier 裁定。隐藏验收测试不能作为候选评分 oracle 泄漏给 agent。

建议的状态至少包含环境/仓库快照标识、文件变更、运行进程、已见测试结果、工具权限、剩余预算。所有转移缓存都以快照+动作+环境配置为键；没有可信转移的节点明确标注 unknown，不用伪造成功 reward 补齐树。沙箱试运行只在授权隔离副本中进行，不通过重置共享工作树实现回滚。

**首先衡量候选覆盖，而不是首先训练价值网络。** 对固定候选集合统计 oracle success@K、实际选择成功率、两者差值（选择损失）；再统计可修复失败的回收率、新增回退率、无效工具调用、死循环/超时、拒绝率及错误拒绝率。外部 PRM 不是必要条件：编译、公开测试、结构化任务检查、文件与资源约束可提供可验证信号；Gen-Zero 自有小型 outcome/value 模型只补充这些信号，不能覆盖硬约束。

**从浅层、真实反馈搜索开始。** 第一阶段只做少量不同补丁/命令分支的浅层试执行与修复，强制单步证据闭环。只有它在等预算对照下获益，再增加多步 PUCT、状态合并、transposition cache 和 learned rollout。优先在失败风险高且信息价值高的节点搜索，避免每一步都重复 route/ask/imagine 或无差别展开。所有缓存和降成本路径都必须保留同一门禁和证据要求。

**世界模型从可观测子问题训练。** 先预测编译/测试状态、动作失败类型、资源代价、关键依赖变化与预测不确定度，而不是直接宣称能模拟任意代码。对 task/repository 做隔离划分，验收一阶及多步预测误差、校准、分布外拒绝率，以及最终决策收益。发现 OOD 或不确定度超预算时停止模型内深搜，改用受控观测/实际试执行，不能静默换随机权重继续。

**几何约束保持正确角色。** 将几何残差、跨流不一致性、能量变化用作可解释的异常检测或候选排序辅助；每一项都要证明相对无该特征的增量收益。能量下降只说明所定义的能量下降，不说明补丁正确。不要强迫离散、不可逆的软件操作满足保守物理动力学：Koopman/辛结构宜用于有数据支持的局部动力学或潜在子系统，而非所有终端状态的普适假设。

**80→90 的收益账本。** 假设基线真机成功率已经严格测得 80%，要达到 90%，在不引入回退时必须救回原失败任务的一半；有回退时要求更高。按任务计数写成 `新成功数 = 基线成功数 + 修复数 - 回退数`，而不是给每种算法随意分配几个百分点。若固定候选上限低于 90%，必须通过新候选或执行修复扩大覆盖，重排再准也不够。

| 阶段 | 交付物 | 进入下一阶段的条件 |
|---|---|---|
| P0 可信性修复 | 输入/策略缺失拒绝；统一完成状态；不可绕过的执行授权；固定测试与权重 manifest | 所有执行入口的负向测试、缺证据/超时/非法值拒绝测试通过；不允许用 stub 冒充真实服务 |
| P1 最小真机闭环 | 固定 Proposer + Gen-Zero 选支 + 隔离执行 + 外部 verifier | 逐动作有完整证据；真实命令数>0；任务 verifier 真正运行；成功与失败均保留 |
| P2 等预算增益 | 无搜索/浅层搜索/PUCT/几何或 worldmodel 消融 | 冻结配置后在新任务集配对比较；收益与成本、回退和拒绝率一起报告 |
| P3 泛化与 90% 目标 | 多仓库、不同失败类型、重复种子与正式 benchmark | 预先确定样本、统计口径与退出条件；达到目标才宣称，未达到保留负结果 |

在全部比较组保留共同的执行安全底线；门禁增益可用离线负向动作集或有硬隔离的受控实验分析，不能为了消融就向真实环境关闭安全措施。A* 适合可离散化的依赖/子目标图；CEM 适合连续或参数化优化；GFlowNet 可研究候选覆盖；CFR 仅对明确博弈建模的问题有价值；CP-SAT 处理显式离散约束。它们不是都必须塞进每一个编码任务的流水线。

## 5. 真机接线与“100% 就绪”核查

**Terminal-Bench：有静态接线，没有成功闭环证据。** 当前 adapter 的 `route → ask → imagine → environment.exec` 在 `benchmarks/gen_zero_tb_adapter.py:289,339`。它对缺 semantic/gate/MCTS 证据拒绝执行（`:46`），并明示 formal scope 不覆盖任意 shell（`:250`）。这些是应保留的边界。

历史 Harbor 运行的原始命令在 `benchmarks/reports/t2-harbor/command-3.txt:1`。Docker 启动和隔离检查完成，但因缺 proposer URL/model 拒绝；没有任务执行、没有 verifier reward，独立 `acceptance.exit` 为 1。`REPORT.md:75` 明确记载未完成；Harbor CLI exit 0 不能覆盖 trial error。当前 adapter SHA 与报告清单一致，但历史 traceback 行号/文案与当前源码不同，不能把旧 dev trial 冒充当前 HEAD 已运行。

**DeepSWE：外部生成器的受限修复适配器，不是 Gen-Zero 规划闭环。** 当前 `benchmarks/gen_zero_deepswe_adapter.py:290` 记录 `code_generator=external-proposer`、`gen_zero_world_model_used=false`；其源码没有 MCTS/PolicyGate/Gen-Zero decision RPC 调用。它提供受限文件操作、Git patch、回归检查与 Pier 接口，并非已经接通符号世界模型。范围限制还包括语言/文件类型、禁止测试配置修改、指定回归命令等（`docs/benchmarks/deepswe_adapter_runbook.md:87`）；不能将受限 smoke 当作全任务能力。

`docs/benchmarks/evidence/t3-deepswe/attempt-02/acceptance.json:1`：accepted=false、patch_bytes=0、reward=0、F2P=0/60、P2P=964/964，Pier/verifier 进程均可退出 0。已有测试不退步不代表需求完成，partial=0.9414 更不能写成 94.14% 任务成功。attempt-01 也失败。attempt-02 日志记录 action rejection 后外部读取超时；不能隐去这些失败只报“Docker/Pier 就绪”。

历史 raw MCP probe 还暴露旧 dev 引擎硬编码 `expected_reward=1.45`/`formal_checked=true`，证据在 `docs/benchmarks/evidence/t3-deepswe/RESULTS.md:51` 和 `raw/dev-zero-excerpt.txt:15`。这是真实应追责的历史伪能力，但不应误归因当前 HEAD：当前 `crates/gen-zero-service/src/zero.rs:3146` 明确将 MCTS 值定义为动作序列 likelihood，并标记 `value_is_environment_reward=false`。两次 DeepSWE trial 的部署 adapter SHA 与当前文件也不同。

因此准确结论是“基础设施部分实测可运行，完整 Gen-Zero 真机闭环与全量解决率未验收”。本次未连接 dev 重跑、未部署模型、未运行全量 TB/DeepSWE。下一次验收首先固定 adapter/Rust binary/模型/数据集/容器镜像的版本与 hash；单题跑通后才扩大规模。

## 6. 规划引擎和世界模型：同名模块必须分开看

最容易误报的是把不同路径都统称“Gen-Zero MCTS”。当前至少存在：

- **Rust service 语义 PUCT**：`crates/gen-zero-service/src/imagine.rs:1,175,245,263` 确有多层动作历史树、扩展、访问计数与 PUCT；由 `zero.rs:3082,3101` 调用。节点是动作历史；bridge 仍收到原始 state 与新增 history（`imagine.rs:51`），不是执行后的文件/进程快照。价值是逐步概率的几何平均，不是环境 reward（`:8`）。它是真实语义序列搜索，但不是终端环境多步预演。
- **Rust planner MctsEngine**：`crates/gen-zero-planner/src/engine.rs:206` 明写 depth-1，只备份根状态单次 step reward，丢弃 successor/done。它不能与上述 service 实现混用能力描述。
- **Python MctsEngine**：神经路径可有多步 greedy imagined rollout（`python/gen_zero/planner/engines/mcts_engine.py:372`），但主树的 successor 扩展有限；缺模型/奖励路径存在随机 leaf value（`:265` 附近）。另有 `LatentMctsPlanner` 与两步分支单测，不能自动当作默认生产接线。

| 引擎 | 当前源码事实 | 应如何发挥价值 |
|---|---|---|
| A* | Rust `engine.rs:338` 是单步候选评分/heap；Python `astar_engine.py:204` 有真正 goal/neighbors 图搜索，但 client 依状态分流（`client.py:910`） | 给代码任务构造明确子目标/依赖图；证明启发式与状态语义，而不是只保留 A* 名称 |
| CEM | Rust `engine.rs:467` 有有限多步 rollout；Python 离散 `mpc_cem_engine.py:94` 调 transition_fn；连续分支仍有合成转移（`:188`） | 优化参数化动作/预算；必须用真实可校准转移及时间位置相关的序列分布 |
| GFlowNet | Rust `engine.rs:598` 是单步 reward-distance 评分；Python `continuous_gflownet.py:218,252,362` 可采样轨迹，但 plan 不执行完整 TB 训练 | 作为多样候选覆盖的研究分支，测 oracle coverage 增益；未有训练证据不要称已学到流分布 |
| CFR | Rust `engine.rs:661` 一次正 regret matching；Python `cfr_nash_engine.py:157` 缺博弈时走 utility/启发式 | 只有明确玩家、信息集、收益与重复迭代的任务才引入；不将一般编码选支称作纳什求解 |
| CP-SAT | Rust `engine.rs:737` 是 gate过滤+单步最大reward；Python `cpsat_formal_engine.py:23` 明示 predicates，并非 OR-Tools solver | 显式建模资源/依赖/互斥，保存 solver status 和可行性证据；真实 OR-Tools 调用另见 `gate/action_constraints.py:294` |

Rust `ProductionPipeline` 的 simulate/what_if/trajectory 路径确实有多步推进（`crates/gen-zero-planner/src/pipeline.rs:313,392,645`），但后续固定/greedy rollout不等于所有 engine 都变成完整树搜索。规划接口 `crates/gen-zero-core/src/traits.rs:39` 与 `gen-zero-planner/src/engine.rs:68` 还需要丰富 goal、状态相关合法动作、预算、终止/风险和证书语义，才能可靠地接真实代码状态。

**世界模型并不是现成的终端模拟器。** Rust `crates/gen-zero-worldmodel/src/dynamics.rs:54` 使用固定残差先验 `next≈0.95*state+0.05*sin(action phase)`；Symplectic 确有 Stormer-Verlet/Hamiltonian 数值步骤，但 action centre、reward、done 是手工先验（`symplectic_dynamics.rs:1,123,176`）。service 对两者显式标记未训练/未校准，输入仅为规定维数 numeric latent；没有文本/代码到该 latent 的编码器（`crates/gen-zero-service/src/worldsim.rs:1`）。因此不能把稳定积分器写成“理解代码执行后果”。

Koopman 在 `crates/gen-zero-worldmodel/src/koopman.rs:100`、`koopman_spectral.rs:155` 有数学实现；本次搜索未见 Rust planner/service 调用。Python MCTS 的 `use_koopman_jumps` 仅保存字段（`mcts_engine.py:156,170`），不能据此宣称已接入跳步规划。

Python `NeuralDynamicsWorldModel` 是可训练 residual MLP，但已归档训练来自 deadlock torus 合成环境（`scripts/extract_trajectories.py:16,202`），不是终端补丁轨迹。训练脚本自己说明同源 generator 留出不能证明真实任务能力（`scripts/train_world_model_dynamics.py:8`）。本地 checkpoint 的 64-D state/16-D action 配置与 provenance 也不能冒充当前 HEAD 的真实编码世界模型。

建议把“算法正确性、数据拟合、预测校准、规划收益、真机闭环”列为五道独立验收。数值不变量适合证明指定数学对象的性质，不能跨越中间三层直接承诺任务成功。

## 7. 模型规模与权重归属清单

| 模块 | 核查到的架构与规模 | 权重/能力边界 |
|---|---|---|
| 外部 CLM DeepSWE head | Qwen3-8B 的 4096-D embedding；`best_head.pt` cfg width=1536、depth=3、projection=512；state/action 两个 MLP 各 9,443,840 参数，合计 **18,887,680** | 8B 是冻结编码器，不是head。实读的是专用 best_head，不是通用 CLM checkpoint；`/tmp/clmrepro/repo/src/clm/heads.py:1,21` |
| Rust ChoiceHead | dimension/temperature 配置 + 在线 ETF 几何打分，**无学习权重矩阵** | `crates/gen-zero-model/src/choice_head.rs:10,88`；4096维输入不等于8B参数 |
| Python NanoCore ChoiceHead（model/choice_head.py:20复用该实现） | 默认固定seed 128×128投影，共16,384值；Torch wrapper对应16,384参数Linear | `python/gen_zero/nanocore/choice_head.py:500,629`；该模块本身不证明已载入训练checkpoint |
| Rust NanoCore | prototype128 + projection(out_dim×128) + value128，即 `256+128*out_dim` 个f32 | `crates/gen-zero-nanocore/src/core_type.rs:22,43`；默认sin/cos初始化，不是大型预训练模型 |
| Python runtime NanoCore | 默认browser 368,832、vision 123,072、specialist 180,480个权重值 | `runtime/nano_core_browser.py:21`、`nano_core_vision.py:21`、`specialist_nano_core.py:21`（均位于 `python/gen_zero/`）；默认seeded NumPy，checkpoint加载为可选，不能将默认值当训练成果 |
| SemanticScorer | 本地 Qwen2.5-0.5B，实读safetensors共290 tensors、**494,032,768参数**；使用 tied embed_tokens 作为lm_head，做候选续写likelihood/PMI校准 | `python/gen_zero/service/semantic_scorer.py:133,237`；没有另外的8B评分head；缺本地权重会报错（`causal/zero_runtime.py:109`） |
| Rust worldmodel | 固定残差/手工Hamiltonian先验，1024维latent | 没有8B训练权重；几何数值模型与语言编码器不是同一对象 |
| PolicyGate | 线性约束、风险分层、证书验证程序 | 不是神经网络，没有所谓8B参数；语义风险概率由上游模型提供 |

Qwen本地配置为 hidden=896、intermediate=4864、24层、14 attention heads、2 KV heads、vocab=151936、tie_word_embeddings=true。这个外部预训练骨干应计入系统依赖和真实成本，不能一边称“无外部PRM”一边暗示“完全不使用外部模型”。不借助外部PRM可实现的含义是：不把外部专用奖励模型作为决策的必要依赖；仍可使用Proposer和编码表征，并用真实执行反馈训练自有小型价值模块。

参数数值为模块/默认配置或已检查本地checkpoint的范围，不是全仓库唯一总参数量。本次没有全量遍历所有远程机器的权重，不能证明某台未检查主机绝无8B文件；但已足以明确原文8B/AUC指标所指是外部CLM实验。

## 8. 本次验证与共享工作树增量说明

本次实际执行而非仅引用历史报告：

| 验证 | 原始结果 | 能证明什么 |
|---|---|---|
| 原基线 Rust gate | 43通过，退出0 | 门禁/几何单测覆盖范围内成立 |
| 原基线 Rust planner | 69通过，退出0 | 算法/接口/fixture按当前测试工作，不代表真机解决率 |
| service semantic MCTS | 7通过，退出0 | 动作序列搜索的局部算法性质；不是环境模拟 |
| Python gate/evidence | 43通过，退出0 | 对应证据边界测试；没有消除本文额外反例 |
| Python neural worldmodel targeted tests | 4通过、50 deselected，退出0 | 临时小型合成训练模型的rollout/trap/what-if/audit，不代表代码世界模型 |
| DeepSWE adapter | 9通过，退出0 | helper与临时仓库边界，非外部模型或真机任务通过 |
| TB adapter | 两次收集失败，退出2 | 首次缺搜索路径，修正后缺Harbor；未伪造依赖让测试变绿 |
| 增强PRM独立校验 | 退出1 | 前段候选/指标断言完成；checkpoint缺失使完整provenance校验失败 |
| Python真实门禁反例 | 退出0，打印3个PROCEED | `{}`、白名单高风险、白名单NaN风险的边界缺陷 |
| 权重张量计数 | 退出0 | 实读CLM专用head和本地Qwen张量规模，不证明模型准确率 |

完整原始命令、日志、退出码和探针脚本在 `docs/benchmarks/evidence/gen-zero-audit-20260927/COMMANDS.md`。测试不是全量仓库验收；没有本次真实Proposer/容器任务成功日志，因此不能声称live-accepted。

调查过程中，其他工作将共享分支从 `6dfd609` 合并至 `701d800a5d3c55db6dcdb002b7bda5608a6bba11`。本次未执行这些合并。新增改动中：

- Rust约束求和改为i128，补上溢出测试。
- Rust MCTS新增successor/reward数值验证、重复动作/零预算拒绝；仍然是depth-1。
- Rust CEM已改成每个时间位置单独的类别分布，elite完整序列更新，并按深度折扣、terminal截断。本文将其列为原基线提升建议，但这部分现已由并行工作实现，不应重复报为待开发。
- GFlowNet/CFR/CP-SAT的注释与命名描述更诚实地标明legacy局限；没有因此变成完整流学习、博弈求解器或CP-SAT求解器。
- Python client部分吞异常路径改为明确报错；本文Python PolicyGate白名单/缺字段反例所在文件未变。

为避免沿用旧测试冒充新验收，本次在新HEAD追加执行 `cargo test -p gen-zero-gate -p gen-zero-planner --locked`：gate 51通过，planner 78通过，共129通过，退出0。该复跑和独立service/Python测试不应混为同一时点的完整仓库快照验收。新HEAD没有补上terminal-state编码器、真实终端模拟reward或DeepSWE的Gen-Zero调用；核心结论不变。

只读调查由主线程及三个Luna sidecar分别核查模型、规划、评测链路；主线程复核关键代码/原始JSON并运行上述测试。未把worker结论当作部署授权，未调用外部团队消息接口，未更改生产实现。
