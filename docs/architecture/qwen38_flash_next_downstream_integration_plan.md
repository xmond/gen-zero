# Qwen3.8-Flash-Next 表征的下游接入、隐空间对齐与端到端评测分层方案

日期:2026-09-27 · HEAD `fa6cddb` · 性质:**设计方案,不是成果报告**

本文所有"接入""增益"均为待实现设计。凡陈述现状处,均附 `path:line` 或命令。

---

## 0. 先说最致命的事实(前提校正)

任务描述中有三条前提,在本仓库和本机上**不成立或未验证**。方案必须建立在校正后的前提上。

| # | 任务中的前提 | 现场事实 | 证据 |
|---|---|---|---|
| F1 | "提取出的 Qwen3.8-Flash-Next 核心流形" | **不存在任何 Flash-Next 提取特征。** 仓库里只有可行性报告(结论是现在不建议换)、合成随机权重的流式原型、只读 safetensors 头的分片分析脚本。 | `artifacts/qwen_flash_next_feasibility_report_2026-09-24.md:10`;`scripts/prototype_layer_streaming.py:20`("we do not have Qwen3.8-Flash-Next weights");`scripts/analyze_flash_next_shard_layout.py:4`("No weight is downloaded");`ls /ebs/data/extracted_features/` 只有 `qwen72b llama70b gte7b_cpu manifold_alignment` |
| F2 | "64-D / 128-D / 256-D / 896-D 核心流形" | 磁盘上的 64/128/256/896-D 流形来自 **Qwen2.5-0.5B**(hidden=896),不是 Flash-Next。Flash-Next `hidden_size=2560`,对应的阶梯应是 64/128/256/**2560**。 | `benchmarks/results/zero_cpu_natural_multidim_eval_summary.json` → `backbone.family = zero-qwen2.5-0.5b-trunk`;`zero_manifold_natural_gpu_896d.npz` 的 `mean.shape = (896,)`;config.json 实取 `hidden_size 2560` |
| F3 | "6B 活跃参数"意味着轻量 | 活跃参数只降算力,不降显存。bf16 全量 360 GB;单卡 A100-80G 只能放 UD-Q2_K_XL(78.9 GB),无 KV 余量。 | `artifacts/qwen_flash_next_feasibility_report_2026-09-24.md:17,79-92` |

已核实为真的前提:

- **262K 原生上下文属实**:`max_position_embeddings = 262144`(`curl -sL https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`,HTTP 200,2026-09-27 实取)。
- **GDN 结构属实**:48 层,`[linear_attention ×3, full_attention ×1] ×12`;线性注意力 value 头 48、key 头 16、`dk = dv = 128`;`hc_count = 4`(同一 config.json)。config 只写 `linear_attention`;"Gated DeltaNet"这个名字来自官方 README,见可行性报告 §1.2 表格(`artifacts/qwen_flash_next_feasibility_report_2026-09-24.md:59`)。
- **注意**:以上 config 取自 `main` 分支,不是固定 revision。P0 必须固定 revision 并重新核对(与 `docs/zero/qwen38-flash-next-extraction-system-design.md` 第 1 节一致)。

**结论**:下文四层中,每一层的"输入数据"目前都是 0。第一阶段必须先产出真实特征(§4 阶段 P0),否则后面所有验收都无从谈起。

---

## 1. 接入分层

### 1.0 总体约束(沿用生产主干已有的 fail-closed 合约)

- 规划器:世界模型 `Err` 直接中止规划,不得转成跳过、零奖励或默认动作(`crates/gen-zero-planner/src/engine.rs:10-13`)。新打分器沿用此合约:打分器 `Err` 即中止。
- 服务:语义桥不可达时 `_meta.engine = "local_fast_reflex_fallback"`,且声明未打分(`crates/gen-zero-service/src/zero.rs:20-24,848`)。新编码器通路必须有同等显式标注,并提供 `required` 开关,开启后不可达即拒绝(对齐已有 `GENZERO_BRIDGE_REQUIRED`)。
- 挂载:认知资产通过不可变 mount 快照封存,无资产即 `BackendUnavailable`,不给默认模型(`crates/gen-zero-service/src/cognitive.rs:8-10`)。Flash-Next 投影矩阵必须走同一 mount 封存流程,摘要进 `_meta.mount`。

### 1.1 Layer 1:Rust ChoiceHead ETF 几何打分层

#### 现状(致命问题)

`ActionETFChoiceHead::evaluate` 的计算是:

1. 对 `gather_rep[..D]` 整体做 L2 归一化(`crates/gen-zero-model/src/choice_head.rs:72-86`);
2. 按 `ActionId` 升序给候选分配单纯形顶点(`choice_head.rs:91-93`),而 `ActionId = blake3(候选名)` 前 4 字节(`crates/gen-zero-service/src/zero.rs:916-922`);
3. `SimplexEtfFrame::project_logits` **只读前 K−1 个坐标**(`crates/gen-zero-core/src/etf.rs:132-143`)。

由此得到三条后果:

- **候选内容从未进入这个头。** 哪个候选拿哪个顶点,只由名字哈希决定,与候选文本的表征无关。
- **logit 幅度被压扁。** 顶点是单位向量,输入是单位向量,所以 `|logit_i| ≤ ‖x[..K−1]‖`。对各向同性的 D 维单位向量,`‖x[..K−1]‖ ≈ sqrt((K−1)/D)`。D=2560、K=4 时约为 0.034,T=1 下 softmax 几乎均匀(最大概率比 ≤ e^{0.068} ≈ 1.07)。即使是 PCA 坐标(方差集中在前几维),也只是把"状态本身的前 K−1 个主成分"映射到"按名字哈希排的顶点",语义上仍无意义。
- **熵有下限,K≥3 时必然触发升级。** 归一化之后 `|logit| ≤ 1`,与输入多"尖锐"无关。最好情况是 `rep` 正对某个顶点,logit 为 `(1, −1/(K−1), …)`。在调用点固定的 T=1 下(`zero.rs:2464` 传 `1.0`):K=2 时 p_max=0.881、归一化熵 0.527;K=3 时 0.691、0.757;K=4 时 0.558、0.845。这个熵进入 `PolicyGate::evaluate`(`zero.rs:2499-2502`),而默认升级阈值是 0.65(`crates/gen-zero-gate/src/policy.rs:47,233`)。所以**只要候选数 ≥ 3,ETF 头的输出一律被判为 Tier2Escalate**,与输入无关。
- **生产调用点只有一个**,且输入是调用方传入的数字(`etf_rep`)或未训练的 nanocore 输出(`zero.rs:2446-2469`)。nanocore 的投影权重是 `sin()` 固定图案,未经训练(`crates/gen-zero-nanocore/src/core_type.rs:44-49`)。服务头注释明说"没有文本编码器进入流形"(`zero.rs:37-41`)。

所以"把 Flash-Next 流形灌进 ETF"**不能**理解为"把状态向量喂给 `gather_rep`"。那样做等于掷骰子。

#### 设计:ETF 是坐标系,不是打分器

正确做法是先在对齐流形里对**每个候选**打分,再把分数合成到 ETF 顶点基上:

```
输入:状态文本 x,候选 c_1..c_K(文本)
1. 编码   h_x  = E(x),     h_c = E(x ⊕ c)        E = Flash-Next 第 ℓ* 层 last-token;4 路残差分支的聚合方式(均值 / 拼接 / 取门控读出)是显式配置项,写入 mount 摘要,默认值由 P0 消融决定
2. 降维   z    = P (h − μ)                        P ∈ R^{d×2560},d ∈ {64,128,256};只在训练集拟合
3. 候选分 s_c  = ⟨W z_x, z_c⟩ / τ                 W 为任务头(双线性),训练集拟合
4. 合成   rep  = Σ_c s_c · v_{π(c)}               v 为 Helmert 顶点,π 为 ActionId 排序
5. ETF    ⟨v_{π(c)}, rep⟩ = s_c − (1/(K−1))·Σ_{c'≠c} s_{c'} = (K/(K−1))·s_c − S/(K−1),  S = Σ_c s_c
```

第 5 步说明:因为 `⟨v_i, v_j⟩ = −1/(K−1)`,合成后的 logit 是 `s_c` 的正斜率仿射变换,**序不变**。

但上文的熵下限意味着:只要 `evaluate` 还做 L2 归一化且 T 固定为 1,再好的 `s_c` 也只能输出被压平的概率。**采用方案 (b)**:

- **概率**直接取 `softmax(s_c / τ)`,τ 在训练集上标定(温度缩放,最小化验证集 NLL),作为 mount 资产封存。
- **ETF** 只负责置换等变的 argmax 与按 `ActionId` 的确定性平局裁决,其概率输出不再进入门禁熵。
- 不选方案 (a)(去掉归一化、把 τ 传进 `ActionETFChoiceHead::new`):它会改变 `evaluate` 对所有现有调用方的语义,影响面更大。

ETF 在这里只提供置换等变性和与现有 `decide` 协议的兼容,真正的判别力全在第 3 步。

需要改动的生产代码(设计,未实现):

| 位置 | 改动 |
|---|---|
| `crates/gen-zero-service/src/zero.rs` 的 `head == "etf"` 分支(`:2446`) | 新增 `etf_source = "encoder"`:由服务调用编码器 sidecar 得到 `s_c`,在 Rust 内合成 `rep`。旧的"整段状态向量直接当 rep"路径改为拒绝(K>1 且未给候选分数时返回 `InvalidParams`),避免继续用哈希决策。 |
| `crates/gen-zero-service/src/bridge.rs` | 增加 `/v1/encode_candidates` 客户端调用,沿用熔断与 `required` 语义。 |
| `crates/gen-zero-service/src/cognitive.rs` 的 `TopologyPreset` 白名单(注释 `:22-24`,仅 64/128/256) | 不加 2560。2560 仅作编码器内部宽度,流形宽度仍从白名单选,保持 mount 摘要封存。 |
| `zero.rs:2464`(`ActionETFChoiceHead::new(rep.len(), 1.0)`)与 `:2471-2499`(概率合成与熵) | `etf_source=encoder` 时,概率改用 `softmax(s_c/τ)`,τ 从 mount 读取;熵由这个概率计算后再交给 `PolicyGate::evaluate`。 |
| `crates/gen-zero-service/src/server.rs:1661,1670`(MCP schema)、`crates/gen-zero-cli/src/main.rs:119,554`(CLI `--etf-rep`) | `etf_rep` 的两个非测试入口。改为只接受"每候选一个分数"(长度 = K),文档与 schema 同步改;旧的"任意长度状态向量"语义删除。`zero.rs` 内 `:4281,4309,4337` 的测试随之改写。 |
| mount 资产 | 封存 `P, μ, W, τ, ℓ*, 分支聚合方式, 模型 revision, 量化档位` 的 sha256。 |

必须同时删除或改写:`zero.rs:2447-2448` 把 `state.as_slice()` 直接当 `rep` 的路径(规则 5:被替代的旧逻辑物理删除)。

#### 为什么不直接用 896-D / 2560-D

- 896-D 属于 Qwen2.5-0.5B,与 Flash-Next 不在同一坐标系,不能混用(F2)。
- 2560-D 的全宽 ZCA 在已有 0.5B 实验里没有带来可证明的收益:64-D 任务头 `train_accuracy 0.504`、`base_accuracy 0.189`(`zero_cpu_natural_multidim_eval_summary.json` 的 `dimensions.64.task_head`)。宽度不是瓶颈,判别信号才是。先用 128-D 作默认,64/256 作消融。

### 1.2 Layer 2:辛几何世界模型动力学对齐层

#### 现状

- `SymplecticWorldModelDynamics` 是**手设先验,未训练、未标定**(`crates/gen-zero-worldmodel/src/symplectic_dynamics.rs:19-21`)。`H_a = ½|p|² + k/2 |q − c_a|²`,`c_a` 来自动作 id 的 `sin` 编码(`:88-96`)。
- 已有"10 步能量守恒"测试:`|drift|/H0 < 1e-4`,但测的是这口手设的势阱(`symplectic_dynamics.rs:254-256`),不是对任何数据的拟合。
- 潜空间宽度固定:`FullLatent` 1024 = q 512 + p 512(`crates/gen-zero-worldmodel/src/contact.rs:254`,`crates/gen-zero-service/src/worldsim.rs:32`)。
- 耗散动力学 `contact.rs`(Strang 分裂,相体积收缩)**在生产路径 0 引用**:`worldsim.rs:137-140` 只构造 `Residual | Symplectic`;`ContactState` 只在 `contact.rs` 自身和 `tests/compression.rs` 出现。Koopman(`koopman.rs`、`koopman_spectral.rs`)在 `crates/gen-zero-service/src` 中 0 引用。

#### 数学矛盾:守恒和收缩不能同时要

Störmer-Verlet 是辛积分器,保相体积(`symplectic_dynamics.rs:12-13`)。保体积的映射不可能是压缩映射。任务要求"10 步能量守恒**与**收缩性",二者在同一个流上互斥。

更重要的一点:**GDN 的状态递推本身就是耗散的。**Gated DeltaNet 的更新是

```
S_t = α_t · S_{t−1} · (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,   α_t ∈ (0,1),β_t ∈ (0,1),‖k_t‖ = 1
```

`(I − β k kᵀ)` 的谱范数 ≤ 1,再乘 `α_t < 1`,所以齐次部分是严格压缩。GDN 给出的是"带输入驱动的衰减记忆",天然对应接触/共形辛结构,不是守恒的哈密顿流。拿它去拟合纯辛 `H(q,p)`,是在错误的结构上找参数。

#### 设计:共形辛(= 线性阻尼的接触哈密顿)分解

```
H_a(q, p) = ½ pᵀ M⁻¹ p + ½ (q − c_a)ᵀ K (q − c_a)
dq/dt = M⁻¹ p
dp/dt = −K (q − c_a) − γ p                 γ ≥ 0
```

- γ = 0 部分:沿用现有 Störmer-Verlet,守恒验收只对这一部分做。
- γ > 0 部分:精确解 `p ← e^{−γ Δt} p` 与辛步做 Strang 分裂。于是每步 `H_a` 单调不增,10 步满足 `H_a(t+10) ≤ H_a(t)`,且可给出显式上界。收缩验收只对这一部分做。
- 这正是 `contact.rs` 已有的 Strang 分裂结构。**推荐方案:把 `contact.rs` 接进 `worldsim::WorldDynamics` 作为第三种 `DynamicsKind::Contact`**;若 P3 阶段结束仍未接入,则按规则 5 删除 `contact.rs`,不保留孤岛。

#### 从 Flash-Next 表征到 (q, p) 的映射(设计选择,未测试)

GDN 状态 `S_t` 是每层 48 个 128×128 矩阵(786,432 个数),不是 (q, p) 对。我们不直接用 `S_t`,而是用可观测的残差流:

1. **选层 ℓ\***:在训练集上逐层计算相邻层线性 CKA,取 CKA 曲线的最大负曲率点作为"相变层"。4 路残差分支的聚合方式沿用 Layer 1 的显式配置项,不做静默 flatten/mean(与 `docs/zero/qwen38-flash-next-extraction-system-design.md` 第 1 节的禁止事项一致;`scripts/test_intermediate_layer_probe.py:100-106` 的取均值只是原型,自述未验证)。
2. **q**:`q_t = P_q (h_t^{ℓ*} − μ)`,`P_q ∈ R^{512×2560}` 由训练集 PCA 得到。t 是**智能体步**(一次 read/edit/command 之后的状态),不是 token 步。
3. **p**:`p_t = (q_{t+1} − q_{t−1}) / (2Δt)`,中心差分。需要轨迹数据,单条样本无法得到 p。
4. **拟合**:动作按操作类型分组(`read/search/edit/command/finalize`,与 `benchmarks/deepswe_genzero_gate.py:118` 的白名单一致)。对每组学 `c_a`,全局学对角 `K` 与标量 `γ`。辛欧拉残差 `p_{t+1} − e^{−γΔt} p_t = −Δt K (q_t − c_a)` 对 `(K, c_a)` 是线性最小二乘,闭式解。
5. **稳定性**:拟合后逐维检查 `0 < K_ii` 且 `K_ii Δt² < 4`,`γ ≥ 0`,违反则拒绝加载(与 `symplectic_dynamics.rs:66-78` 同一组条件,不做钳位)。最小二乘可能给出负刚度,这一条不能省。

需要改动的生产代码(设计,未实现):

| 位置 | 现状 | 改动 |
|---|---|---|
| `crates/gen-zero-worldmodel/src/symplectic_dynamics.rs:53-57` | 只有标量 `stiffness`、`action_scale` | 新增 `FittedPhaseParams { k_diag: [f32; 512], centres: Vec<[f32; 512]>, gamma: f32 }`,由 `SymplecticWorldModelDynamics::from_fitted` 构造;校验同上。 |
| 同文件 `:88-96` `action_centre` | `c_a` 由 `sin(action.0)` 生成 | 有拟合参数时按动作类别查表;类别未知则返回 `Err`,不回退到 `sin` 编码。 |
| `ActionId → 动作类别` | `ActionId = blake3(name)`(`zero.rs:916-922`),类别无法从 id 反推 | 在 mount 资产中封存 `{候选名模式 → 类别}` 显式表;服务在构造 `ActionId` 时同时查出类别,一并传给世界模型。 |
| `crates/gen-zero-service/src/worldsim.rs:137-160` | `WorldDynamics` 只有 `Residual`、`Symplectic`,均用 `default()` | 新增 `Contact`(γ>0)分支;`Symplectic`/`Contact` 在 mount 有拟合参数时从 mount 加载,无则保持现有"未训练先验"标注。 |
| mount 资产 | 无世界模型参数 | 封存 `K, c_a, γ, P_q, μ, Δt, 类别表` 的 sha256。 |

#### 诚实边界

- 真实代码智能体的轨迹不是哈密顿系统。这里的"拟合 H"只是一个带结构约束的线性模型。能量守恒是积分器的性质,**不是**模型对真实世界的正确性证明。
- 唯一有意义的验收是预测误差:在按仓库隔离的留出集上,k 步预测误差必须显著低于"持续性基线"`q_{t+k} = q_t`。
- 轨迹数据集目前不存在。现有 Python `NeuralDynamicsWorldModel` 训练于合成 torus 环境(`docs/architecture/gen_zero_capability_audit_20260927.md:131`),不能复用为证据。

### 1.3 Layer 3:System 2 规划器层(MCTS / A\* / CEM)

#### 现状

- `MctsEngine`:无先验、无价值函数钩子,回报只来自 `world_model.step`,叶子续值为 0(`crates/gen-zero-planner/src/engine.rs:218-232,386-400`)。
- `AStarEngine`:故意用 h = 0(Dijkstra),因为潜空间距离与回报之间没有已证明的界(`engine.rs:508-513`)。边代价 `1 + max(−r,0) + w·0.05·distance`(`engine.rs:651`)。
- `MpcCemEngine`:采样动作序列,按世界模型折扣回报排序(`engine.rs:872-911`)。
- **唯一现成的先验钩子**在服务层:`imagine` 动词的 `PriorOracle` trait(`crates/gen-zero-service/src/imagine.rs:39-40`),实现为 `BridgeOracle`(`imagine.rs:44-52`),给 PUCT 提供语义先验。

#### 设计

| 引擎 | 流形距离的用法 | 数学代价 |
|---|---|---|
| **service `imagine`(推荐首选)** | 新增 `EncoderOracle: PriorOracle`,先验 `π(c | history) = softmax(s_c)`,`s_c` 来自 §1.1 第 3 步。 | 无需改 planner crate 的 trait 签名。PUCT 对先验没有可采纳性要求。 |
| MCTS(planner crate) | 需在 `PlanningEngine` 增加可选 `&dyn PriorProvider` 参数。 | 改公共 trait 签名,影响 6 个引擎。**放到 P4,且以 imagine 路径有正增益为前提。** |
| A\* | 两个选项:(a) 仅把流形距离放进现有 `uncertainty_penalty_weight · distance` 边代价,保持 h = 0 与最优性;(b) 做加权 A\*,`h = λ·d_M(s, goal)`,**明确放弃最优性声明**,在结果中标注 `admissible: false`。 | 推荐 (a)。(b) 只有在证明 `d_M ≤ 真实剩余代价` 后才可声明可采纳,目前无此证明。 |
| CEM | 用 `s_c` 作为初始分布的 logits,代替均匀初始化。 | 只改初始化,不改目标函数,风险最低。 |

所有新打分器沿用 `engine.rs:10-13` 的合约:打分失败即中止,不得回退为均匀先验而不报错。若需要均匀先验作为消融对照,必须作为显式参数传入,并写进 `_meta`。

### 1.4 Layer 4:多模型双流流形对齐层

#### 现状:已有的双流结果是**零增益**

`benchmarks/results/spec21_manifold_pareto_ensemble_report.md` 的配对 bootstrap(13 任务宏平均,测试集):

| 比较 | Δ pp | 95% CI |
|---|---:|---|
| Peak 轨道 − 单模型对照 | +0.37 | [−0.27, +1.01] |
| Peak − `qwen+bbp` | +0.53 | [−0.30, +1.35] |
| Peak − `concat+bbp` | −0.08 | [−0.85, +0.70] |
| `geo100+bbp` − `concat+bbp` | −0.29 | [−1.05, +0.48] |

所有 CI 都跨 0。预设目标全部为 False(同文件"Target check"段)。跨模型 CKA 测试集均值 0.516,Procrustes 残差 0.679(`benchmarks/results/manifold_alignment_qwen72b_vs_llama70b_13tasks.md` test_full 均值行)。

结论:**"双流 Pareto 前沿"目前没有统计上可辩护的增益。**加第三路 Flash-Next 之前,必须先预注册停止规则。

#### 设计

1. **宽度对齐**:Flash-Next 2560 vs 72B/70B 8192。正交 Procrustes 要求同宽。方案:三方各自在训练集上 PCA 到公共宽度 d = 256,再做 Procrustes。d 的选择在读取任何测试标签之前冻结,写入报告头。
2. **联合对齐**:以 Qwen-72B 为锚,`R_F = argmin_{RᵀR=I} ‖Z_F R − Z_Q‖_F`(SVD 闭式解)。CKA 用于诊断,不用于选择。报告 null CKA 和 null Procrustes(置换对照),沿用现有报告格式。
3. **融合**:沿用 `benchmarks/suites/evaluate_manifold_pareto_ensemble.py` 的 5 折 OOF + 1-SE 选择,新增表征 `flash`、`concat3`、`geo3`。测试标签只在选择冻结后读取(该脚本 `select_task` 无测试标签参数,见文件头 `:17-19`)。
4. **预注册停止规则**:三流选中配置 vs 最佳双流配置,逐行 bits 做配对 bootstrap(脚本已有 `paired_macro_bootstrap`,`:549`)。**95% CI 下界 ≤ 0,则 Layer 4 的 Flash-Next 分支删除,不进生产。**

---

## 2. 真机评测落地(Terminal-Bench 2.1 与 DeepSWE)

### 2.1 现状中的两个死依赖

1. **DeepSWE 门禁依赖的认知服务不存在。**`DeepSWEGate` 调用 `operation: "assess"`(`benchmarks/deepswe_genzero_gate.py:145`)和 `operation: "transition"`(`:180`)。全仓只有测试 mock 响应这两个操作(`benchmarks/tests/test_deepswe_adapter_genzero_integration.py:36,48`)。Rust 服务没有 `/evaluate` 路由(`crates/gen-zero-service/src/server.rs:876-894`)。所以真机运行时 `gen_zero_mcts_used` 只能是 false,或者门禁 fail-closed(`deepswe_genzero_gate.py:143-144` 返回 `COGNITIVE_SERVICE_UNAVAILABLE`)。另外,该路径驱动的是 Python `ImaginationMCTSPlanner`(`:20,202`),不是 Rust `MctsEngine`。
2. **TB 适配器的上下文门禁是 250,000 字符。**`len(encode(state)) > 250_000` 即拒绝(HEAD `fa6cddb` 的 `benchmarks/gen_zero_tb_adapter.py:325-326`;工作树中另一会话的未提交改动已把它移到 `:338-339`,阈值未变)。约合 6-8 万 token。在它被有意识地调整之前,262K 上下文毫无用处。

### 2.2 Flash-Next 作为 Code Proposer + PolicyGate

分工原则:**Flash-Next 只提议,Gen-Zero 只裁决。**提议者永远不拥有执行权。

```
TB/DeepSWE 任务
  └─ Flash-Next(vLLM,Qwen4ExpForCausalLM,262K)生成 N 个 JSON 动作候选
       └─ 结构门禁 DecisionPolicyGate(路径、白名单、ast.parse、受保护目录)   deepswe_genzero_gate.py:101-139
            └─ 语义门禁 /evaluate assess(新建,Rust)                          deepswe_genzero_gate.py:143-157
                 └─ 前瞻 /evaluate transition + imagine(EncoderOracle 先验)  deepswe_genzero_gate.py:166-216
                      └─ 选中动作 → Docker 沙箱执行 → 真实观测回灌
```

262K 上下文的正确用法:

- **把仓库检索结果和已执行观测放进上下文**,而不是把整个仓库塞进去。长上下文减少的是"读错文件"类失败,不能替代执行验证。
- 上调 TB 上限时同时加两条门禁:(a) 上限以 token 计(用 Flash-Next 的 tokenizer 实测),不以字符计;(b) 超限时 fail-closed 拒绝,不做静默截断。沿用现有的拒绝语义。
- 已知风险:llama.cpp 有"长上下文 decode 线性变慢"的开放 issue #28734(可行性报告 `:115`)。部署走 vLLM CUDA 路径(报告 `:127-145`),llama.cpp 路径只作备用,且必须显式标注。

### 2.3 如何克服单打分头在留出任务上的"瞎猜"

基线事实(不是推测):

- 外部 CLM(冻结 Qwen3-8B 编码器 + 两个约 944 万参数的 MLP 头)在本地留出轨迹上 pooled AUC **0.5196**(`docs/architecture/gen_zero_capability_audit_20260927.md:18,139`),接近随机。
- 已有增强 PRM 在 38 题上 BoN=4 从 31/38 **降到** 29/38(同文件 `:9`)。这 38 题已被反复调参,不再能作为独立留出集。

机制设计(每一条都要由实验证明,现在都是**未验证**):

1. **符号门禁做"排除",不做"排序"。**结构门禁能确定性地排除非法编辑(受保护路径、非 `.py`、语法错误,`deepswe_genzero_gate.py:124-139`)。它提升的是下限,不是胜率本身。必须单独统计"门禁拦截数"和"拦截后的最终通过率",两者不能合并报告。
2. **前瞻预演用真实执行,不用想象。**在沙箱里对 top-k 候选各跑一次仓库现有测试(不是隐藏测试),以真实退出码作为 transition 的回报。世界模型只负责剪枝排序,最终判据是执行结果。这是对 CLM 单头的根本区别:**把"打分"换成"观测"**。
3. **打分头只在编码器与流形对齐都通过 P1/P2 验收后启用**,并与"无打分头(均匀先验)"做配对对照。

胜率声明的前置条件:

- 数据:按**仓库**隔离的新留出集,不含那 38 题,冻结后不再调参。
- 统计:逐任务配对(同一任务、同一种子、同一预算),McNemar 精确检验或配对 bootstrap。没有逐样本配对统计,不声称任何提升。
- 对照臂:(A) Flash-Next 裸跑;(B) A + 结构门禁;(C) B + 语义门禁;(D) C + 前瞻预演。增益归因只能来自相邻臂之间的配对差。

---

## 3. 接入链路图

```
                         ┌───────────────────────────── 离线(P0/P1)──────────────────────────────┐
  HF Qwen/Qwen3.8-Flash-Next (revision 固定)                                                        │
    └─ vLLM Qwen4ExpForCausalLM,FP8,多卡 / 或 CPU offload 截断 K 层                              │
         └─ 逐层 hidden(分支聚合=显式配置)→ CKA 选层 ℓ*                                            │
              └─ npz + SHA256SUMS + manifest(模型 rev、量化档位、ℓ*、数据集哈希)                   │
                   ├─ PCA/ZCA P, μ(64/128/256) ── 任务头 W ──┐                                    │
                   ├─ Procrustes R_F → Qwen72B 锚 ─────────────┤ (Layer 4,过停止规则才保留)      │
                   └─ 轨迹 (q,p) → 拟合 K, c_a, γ ─────────────┤ (Layer 2)                       │
                                                               ▼                                   │
                                                 mount 资产发布  POST /v1/mounts  (server.rs:894)   │
└──────────────────────────────────────────────────────────────┬──────────────────────────────────┘
                                                               │ 摘要封存,请求期不可变
┌──────────────────────────── 在线(Rust gen-zero-service)─────▼──────────────────────────────────┐
│ /v1/decisions  ask/decide head=etf, etf_source=encoder                                            │
│   └─ bridge.rs → Python 编码器 sidecar /v1/encode_candidates → s_c                               │
│        └─ rep = Σ s_c v_π(c) → ActionETFChoiceHead::evaluate (choice_head.rs)      [Layer 1]      │
│ imagine  → EncoderOracle: PriorOracle (imagine.rs:39) → PUCT                         [Layer 3]      │
│ simulate/what_if → WorldDynamics::{Symplectic, Contact(新)} (worldsim.rs:137)        [Layer 2]      │
│ /evaluate assess|transition(新建)→ PolicyGate + WorldDynamics                     [DeepSWE 桥]   │
└──────────────────────────────────────────────────────────────┬──────────────────────────────────┘
                                                               │
┌──────────────────────────── 评测(Harbor / Pier)──────────────▼──────────────────────────────────┐
│ gen_zero_tb_adapter.py  → /v1/decisions route+ask   (上下文门禁改 token 计数,fail-closed)          │
│ gen_zero_deepswe_adapter.py → DeepSWEGate → /evaluate  (Proposer = Flash-Next)                   │
│ 结果:逐任务 bits + telemetry(gen_zero_gate_used / mcts_used / world_model_backend)               │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 阶段验收标准

每阶段三类证据缺一不可:(1) 一条 grep,证明新符号在生产路径有 ≥1 个非测试引用;(2) 一条命令及其期望退出码;(3) 具名测试行 `... ok`。任一阶段失败,停在该阶段,不跳级。

| 阶段 | 交付 | 生产引用 grep(期望 ≥1 非测试命中) | 命令与判据 | 失败处置 |
|---|---|---|---|---|
| **P0 特征** | 真实 Flash-Next 特征:13 任务 × 逐层 hidden,SHA256SUMS,manifest。提取工程按 `docs/zero/qwen38-flash-next-extraction-system-design.md` 执行,本方案只消费其产物 | 不适用(离线资产) | `sha256sum -c SHA256SUMS` 退出码 0;manifest 含模型 revision 与量化档位;与 27B/72B 同任务线性探针对照,记录在报告里 | 无特征则 P1-P5 全部冻结 |
| **P1 Layer 1** | 编码器 sidecar + `etf_source=encoder` + 删除"状态向量直接当 rep"路径 | `grep -n "etf_source" crates/gen-zero-service/src/zero.rs`;`grep -n "encode_candidates" crates/gen-zero-service/src/bridge.rs` | `cargo test -p gen-zero-service etf` 具名测试:① 同一状态、交换候选名,选择随内容而非名字变化;② sidecar 不可达且 required=1 时拒绝;③ 旧路径输入返回 `InvalidParams`;④ K=4、`s_c` 足够尖锐时返回的最大概率 > 0.9,且熵低于门禁阈值 0.65(防止熵下限回归) | 候选内容置换测试不过,则不得宣称"几何打分" |
| **P2 Layer 4** | 三流对齐 + 预注册停止规则 | `grep -n "flash" benchmarks/suites/evaluate_manifold_pareto_ensemble.py` | 报告中"三流 − 最佳双流"配对 bootstrap CI 下界 > 0 | CI 下界 ≤ 0:删除 Flash-Next 融合分支 |
| **P3 Layer 2** | `DynamicsKind::Contact` 接入 `worldsim.rs`;拟合脚本;轨迹数据集 | `grep -n "Contact" crates/gen-zero-service/src/worldsim.rs` | 具名测试:γ=0 时 10 步 `|ΔH|/H0 < 1e-4`;γ>0 时 10 步 `H` 单调不增;加载 `K_ii Δt² ≥ 4` 的资产被拒。留出集 k=1,3 步预测误差低于持续性基线,配对 bootstrap CI 下界 > 0 | 预测误差不优于基线:不上线拟合参数;P3 结束仍未接入则删除 `contact.rs` |
| **P4 Layer 3** | `EncoderOracle` 接入 `imagine`;CEM 初始化 | `grep -n "EncoderOracle" crates/gen-zero-service/src/zero.rs` | 具名测试:oracle `Err` 时 imagine 返回错误而非均匀先验;同一任务集 EncoderOracle vs 均匀先验,配对对照 | 无正增益:不改 planner crate trait |
| **P5 真机** | `/evaluate assess|transition` 在 Rust 服务落地;TB 上下文门禁改 token 计数 | `grep -n "\"/evaluate\"" crates/gen-zero-service/src/server.rs` | 仓库隔离留出集,A/B/C/D 四臂逐任务配对,McNemar 精确检验;telemetry 中 `gen_zero_mcts_used=true` 的比例 = 实际调用比例 | 未过统计:报告"无可证明增益",不写胜率 |

**0 引用清除清单**(到期未接入即删除,规则 5):

| 符号 | 当前生产引用 | 截止阶段 |
|---|---|---|
| `crates/gen-zero-worldmodel/src/contact.rs` 的 `ContactState` 与积分器 | 0(仅测试) | P3 |
| `crates/gen-zero-worldmodel/src/koopman.rs`、`koopman_spectral.rs` | 服务层 0 | P3(本方案不用它;若无其他 owner 认领,应删除) |
| `zero.rs:2447-2448` 状态向量直接当 ETF 输入;`etf_rep` 的"任意长度向量"语义(`server.rs:1670`、`gen-zero-cli/src/main.rs:119,554`) | 有,但语义错误 | P1 |

---

## 5. 分类汇报

### 已实现(本次交付,附证据)

- 本设计文档。所有现状陈述已逐条读源码或实跑命令核对,引用见各节 `path:line`。
- 实取核对:Flash-Next `max_position_embeddings=262144`、`hidden_size=2560`、48 层 GDN/全注意力 3:1 交错(`curl` config.json,HTTP 200)。
- 计算核对:ETF 头在 T=1 下的熵下限(K=2/3/4 → 0.527/0.757/0.845),对照门禁默认阈值 0.65(`policy.rs:47`)。用 `python3` 按 `logit = (1, −1/(K−1), …)` 直接算出,见 §1.1。
- 实取核对:896-D 流形的骨干是 Qwen2.5-0.5B(`zero_cpu_natural_multidim_eval_summary.json` 的 `backbone.family`)。

### 未验证

- GDN 状态与 CKA 相变层能给出可拟合的 (q, p) 结构:纯设计,无数据。
- `Qwen4ExpForCausalLM` 能否直接加载多模态 checkpoint 跳过视觉塔(可行性报告已列为未验证)。
- 4 路残差流取均值是否是合理的聚合方式(`scripts/test_intermediate_layer_probe.py:103-104` 自述未验证)。
- 262K 上下文在 A100 上的实际吞吐与显存:未测。

### 未完成

- P0-P5 全部。原因:Flash-Next 权重未下载,无任何提取特征;无代码智能体轨迹数据集;`/evaluate` 服务不存在。
- 任何增益、胜率声明:没有逐样本配对统计,一概不作。

---

## 附:现场发现的 fail-open 风险(非本方案引入,未改动)

工作树中未提交的 `benchmarks/gen_zero_deepswe_adapter.py` 改动(非本次作者)含两处问题:

1. `self.key = os.environ.get(key_env) or "local-gpu-key"`:凭据缺失时静默使用硬编码值,替换了原先"未设置即报错"的行为。违反规则 2。
2. 局域网判定用子串匹配 `any(h in url for h in ("100.", "192.168.", ...))`:`http://evil.example/100.x` 之类的 URL 也会通过,且放行明文 HTTP。应改为解析 hostname 后做 IP 网段判定。

建议该改动的作者在提交前修复。本方案未修改该文件。
