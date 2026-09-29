# Spec 30：Qwen3.8-Flash-Next 特征与因果流形分层提取设计案

- 日期：2026-09-27
- 源码基线：`fa6cddb656f49d9ae01bf417e476514962e03a34`，工作目录 `/ebs/pj/gen-zero`
- 目标模型：`Qwen/Qwen3.8-Flash-Next`，HF revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`（2026-08-27），`config.json` sha256 `889658f2…2e74b`
- 参考实现：`transformers 5.17.0` 的 `models/qwen4_exp/modeling_qwen4_exp.py`（下文简写 `M:`，行号以 `~/.hermes-venv/lib/python3.11/site-packages/transformers/` 下该文件为准）；GDN 核心 kernel 在 `models/qwen3_5/modeling_qwen3_5.py`（简写 `Q35:`）
- 证据目录：`docs/zero/evidence/qwen38-extraction-design/`（每条命令、退出码、日志尾部见 `commands.json`）

## 0. 结论先行，三类分开

**已实现（有证据）**

1. Flash-Next 与本地 `qwen4_exp` 实现是同一架构。真实 `config.json` 的 `architectures` 为 `Qwen4ExpForConditionalGeneration`，`model_type=qwen4_exp_text`，48 层，`layer_types` 36 个 `linear_attention` + 12 个 `full_attention`（`configuration_qwen4_exp.py` 的 `__post_init__` 把 `full_attention` 改写为 `qwen_sparse_attention`），`ple_layer_ids=[2]`，`hc_count=4`，512 专家 top-10，indexer budget 2048 / compress 4。证据：`flash-next-config.json`、`flash-next-revision.json`。
2. §8 总表中的**每一项 Hook 点**都在同一份 `qwen4_exp` 代码上被实际挂载并验证了张量形状、取值范围和代码路径，用的是 CPU 上的微型随机权重模型。命令 `/home/luy/.hermes-venv/bin/python scripts/test_qwen4exp_extraction_hooks_tiny.py`，退出码 0，日志 `hooks-tiny.log`。关键结果：GDN 逐 token 递归路径得到的最终状态与分块 kernel 的最终状态最大绝对误差 6.4e-10；`output_router_logits=True` 返回的 logits 与 hook 抓到的 logits 逐元素相等；读门输出宽度等于 `hidden`，层间残差宽度等于 `hc_count*hidden`。
3. 现有 `StreamingCovarianceAccumulator` 存在灾难性抵消缺陷（§6），已复现并归档：`covariance-cancellation.log`，退出码 0。

**未验证（做了，但没有证据证明它对）**

- §2–§5 中所有"GDN 状态天然契合 Lyapunov 相空间"、"路由熵映射认知不确定性"、"N-gram 局部流形可对齐"的表述，都是**数学论证与设计假设**，没有一次真实 Flash-Next 前向。本机无 GPU（`nvidia-smi` 不存在），权重 360 GB 不在本地。
- 两个规范已写、脚本未验证的读数：§2.3 的 `q_readout`（`core_attn_out`，归一化前 6144 维）与 §4.3 的 PLE 门值（需 hook `norm_key`/`norm_query` 后按 `M:1246-1247` 复算）。

**未完成（没做，说明原因）**

- 真实模型上的提取、任何 CKA 数值、任何路由熵与风险的配对统计：没有算力与权重。
- 把路由熵接进 `PolicyGate`：这需要一条新输入线（§3.4），本设计只给规范，不改生产代码。
- 视觉编码器与 MTP 层：明确排除在提取范围外。
- `StreamingCovarianceAccumulator` 的修复：属于生产代码改动，超出设计案范围，列为**阻断性前置条件**（§6），由 owner 裁定。

## 0.1 相关文档与分工（同日、同一 HEAD 的三份并行设计）

| 文档 | 负责的问题 | 与本文的关系 |
|---|---|---|
| 本文 `docs/zero/30-…` | 架构机理、数学定义、Hook 点、数据格式（问题一至三） | 定义"取什么、在哪取、怎么算" |
| `docs/zero/qwen38-flash-next-extraction-system-design.md` | 加载器、显存/内存账本、Transformers 与 vLLM/SGLang 引擎选择、offload、远端闭环 | 定义"在什么机器上、用什么引擎把本文的 Hook 跑起来"；其"不得静默 flatten/mean/select branch"与本文 §5.2 主口径一致 |
| `docs/architecture/qwen38_flash_next_downstream_integration_plan.md` | 提取产物的下游接入、隐空间对齐、Terminal-Bench/DeepSWE 评测分层 | 消费本文 §7 的产物格式；其 F1 判定"当前不存在任何 Flash-Next 提取特征"与本文 §0 一致 |
| `artifacts/qwen_flash_next_feasibility_report_2026-09-24.md` | 早前可行性结论（bf16 全量约 360 GB，单卡 A100-80G 无 KV 余量） | 本文 §9 的算力边界以它为准 |

三份文档均未改动生产代码。本文与下游接入案在一点上互补：那份文档取的 `config.json` 来自 `main`，本文固定到 revision `de4b8e4d…` 并归档了 sha256。

## 1. 与稠密 Transformer（Qwen2.5-72B）相比，表征提取的本质差异

现有提取器的隐含前提，都来自稠密模型：一条残差流、每层一个 `hidden_states`、宽度恒等于 `hidden_size`、注意力是无状态的全量检索。`run_universal_extraction_a100.py:115-120` 直接取 `output_hidden_states` 的中层与末层最后 token；`gpu_extract_qwen72b_13tasks.py` 取 llama-server 的 8192 维 pooled 向量。Flash-Next 把这四条前提全部打破：

| 维度 | Qwen2.5-72B（稠密） | Qwen3.8-Flash-Next | 对提取的后果 |
|---|---|---|---|
| 残差流 | 1 条，宽 8192 | 4 条并行流，层间宽 `4×2560=10240`（`M:1480` `repeat(1,1,hc_count)`；`M:1303,1309` 写回 `hyper_input + injection`） | 层输出不是"那个表征"，而是 4 条流；每个子块读到的 2560 维 `mixed_input` 由数据依赖的读门决定（`M:1023-1025`） |
| 序列混合 | 全量 softmax 注意力，无状态 | 36 层 GDN：每头一个 128×128 矩阵状态，线性时变递推；12 层 QSA：索引器按 4-token 微块 top-512 选 2048 个 token 再做注意力（`M:755-757`） | GDN 层有真正的**连续动力学状态**可截取；QSA 层多出一个离散选择信号（被选块集合） |
| 前馈 | 稠密 MLP | 512 专家 top-10 + 1 共享专家，路由 softmax 全 512（`M:971-975`） | 每层每 token 多出一个 512 维路由分布：这是稠密模型完全没有的显式不确定性信号 |
| 词法输入 | 仅 token embedding | 第 2 层（0 索引 layer 1）额外注入 2-gram/3-gram 哈希嵌入，51B 参数，经 key/value 投影与门控写入 4 条流（`M:1246-1248`） | 存在一条与上下文无关、由最近 3 个 token id 决定的**离散局部特征**路径 |
| 层间可比性 | 各层同质 | 周期 4：`(GDN→MoE)×3 → (QSA→MoE)` | 相邻层 CKA 会被架构周期调制，`argmin` 会误定位（§5） |
| 最终归一化 | final RMSNorm | `hyper_connection_mixer`（`M:1493`）把 4 流混成 2560 | "最后一层隐状态"要指明是混合前 10240 维还是混合后 2560 维 |

结论：对稠密模型，"取 hidden_states[l][:, -1]" 就是提取；对 Flash-Next，提取必须先回答四个问题：**取哪条流（或读门混合后）、取哪类层（GDN/QSA）、要不要路由分布、要不要 N-gram 路径**。下面的分层就是把这四个问题各占一层。

## 2. Layer 1：GDN 递归状态 vs QSA 注意力隐状态

### 2.1 GDN 状态的真实数学形式（来自代码，不是假设）

`Q35:478-490` 的逐 token 更新，对每个 value 头 $h$（48 个，状态 $S_t^{(h)} \in \mathbb{R}^{128\times128}$）：

$$
S_t = e^{g_t}\,S_{t-1} + \beta_t\, k_t\,\bigl(v_t - e^{g_t} S_{t-1}^{\top}k_t\bigr)^{\top}
    = e^{g_t}\bigl(I - \beta_t k_t k_t^{\top}\bigr) S_{t-1} + \beta_t k_t v_t^{\top}
$$

（先衰减再算 `kv_mem`，`Q35:483-488`。）其中（`M:579,581,596`）：$\beta_t=\sigma(b_t)\in(0,1)$；$g_t=-e^{A_{\log}}\cdot\mathrm{softplus}(a_t+\mathrm{dt\_bias})\le 0$，所以 $e^{g_t}\in(0,1]$；$q,k$ 在 kernel 内做了 L2 归一化（`use_qk_l2norm_in_kernel=True`），故 $\|k_t\|=1$，$(I-\beta_t k_t k_t^\top)$ 是特征值 $\{1-\beta_t, 1,\dots\}$ 的收缩投影。

于是 GDN 状态是一个**线性时变、逐步非扩张**的系统：$S_t = A_t S_{t-1} + B_t u_t$，$\|A_t\|_2 \le e^{g_t} \le 1$。

### 2.2 是否"天然契合" $\dot z = Az + Bu$

**部分契合，且有明确的差距，不能直接宣称契合。**

- 契合处：状态线性递推、输入线性注入、按构造收缩。这与 `LyapunovPhaseSpaceReconstructor` 要求的 $\rho(A)<1$ 同向（`universal_manifold_extractor.py:349-361`）。
- 差距一：Gen-Zero 拟合**常系数** $A$；GDN 的 $A_t=e^{g_t}(I-\beta_t k_tk_t^\top)$ 是**输入依赖**的。常系数拟合是对 $\mathbb{E}_t[A_t]$ 的近似，残差必须成为验收指标（§2.4）。
- B11 修复：连续生成元按 Hurwitz 判据 `max(Re(lambda)) < -epsilon` 检查，离散转移矩阵按 Schur 判据检查；不稳定拟合直接报错，不再缩放成所谓稳定证书。A100 入口必须显式提供有序轨迹和 dt，独立 benchmark 样本行不能冒充轨迹。
- 差距三（最关键，来自代码）：**HF 前向不暴露逐 token 状态。** `torch_chunk_gated_delta_rule` 只在 `output_final_state=cache_params is not None` 时返回**最终**状态（`M:595,608`），`use_cache=False` 时什么都不返回。要得到轨迹 $z_1,\dots,z_T$，只有两条路：(a) 逐 token 递归解码（`seq_len==1` 路径），每步读 `cache.layers[i].recurrent_states[0]`；(b) hook 参考 kernel 内部。本设计采用 (a)，并已在微型模型上验证 (a) 与分块 kernel 结果一致（误差 6.4e-10）。
- 明确否定一个既有做法：`run_universal_extraction_a100.py:190` 把 64 条**互不相关的 prompt** 的末层向量按顺序拼成 `Z_manifold` 当"轨迹"去拟合 $A$。那不是动力学轨迹，是样本序。Flash-Next 有真实的时间轴状态，Layer 1 必须用它，不得沿用旧做法。

### 2.3 状态截取规范

| 项 | 规范 |
|---|---|
| Hook 位置 | 逐 token 前向后读取 `past_key_values.layers[i].recurrent_states[0]`，`i ∈ GDN 层`（36 层）；形状 `[B, 48, 128, 128]`（`Q35:323`），bf16 前向后转 float32 |
| 观测向量 $z_t$ | 默认 $z_t^{(i)} = \mathrm{vec}(S_t^{(i)}) \in \mathbb{R}^{786432}$ 太大。降维两级：(1) 每头做 $S^{(h)}\in\mathbb{R}^{128\times128}$ 的前 $r$ 个奇异值 + 左奇异向量投影系数；(2) 或用该层自己的 query 读出 $y_t = S_t^\top q_t$（即 `core_attn_out`，`Q35:490`），维 `48×128=6144`，这是模型真正"读到"的状态。**两者都存，`state_readout` 标记 `svd_r` 或 `q_readout`** |
| 时间轴 | token 位置 $t$；`dt=1`；连续化时记 $A_{\text{cont}}=(A_{\text{disc}}-I)$ |
| 填充 | 左填充（GDN 对右填充会持续衰减状态，见 `scripts/test_intermediate_layer_probe.py:11`）；只对 `attention_mask==1` 的位置记录 |
| 控制输入 $u_t$ | 该层 GDN 的输入 `mixed_input`（读门输出，2560 维，`M:1025`），作为 $Bu$ 项的 $u$ |
| 分层覆盖 | 不是每层都存全轨迹。默认存 3 个 GDN 层：周期内位置固定（如 layer 0/1/2 的第 1 个）、Layer 4 判定的 mid 层、最后一个 GDN 层（layer 46）。其余层只存最终状态 $S_T$ |

### 2.4 QSA 层的隐状态

QSA 层没有递归状态，可取的是三样：

1. `self_attn` 输出（2560 维，`M:1302` 前）与该层 `mixed_input`；
2. 索引器输出的 token 选择掩码，形状 `[B, 1, T, kv_len]` bool（`M:773`，`unsqueeze(1)` 在 head 维；已验证），密度 $\rho_{\text{sel}}=\#\text{选中}/kv\_len$；
3. 若用 eager 注意力可拿 `attn_weights`（`M:` `Qwen4ExpTextAttention` 返回值），sdpa 下为 `None`，不作为必需项。

选择掩码本身是一个离散信号，不进入连续相空间；它作为 Layer 2 之外的第二个"认知信号"保留（§3.5），用于解释 QSA 层 CKA 的跳变。

**验收判据（Layer 1）**：对每条存全轨迹的层，`LyapunovPhaseSpaceReconstructor.fit` 得到 $A$ 后，一步预测相对残差 $\|z_{t+1}-(I+A)z_t-Bu_t\|/\|z_{t+1}\|$ 的中位数必须报告；没有这个数字，不得写"契合"。

## 3. Layer 2：MoE 路由器特征

### 3.1 可提取量（全部已验证 hook 路径）

每层（48 层全部有 MoE）、每 token：

- 路由 logits $\ell \in \mathbb{R}^{512}$：hook `layers[i].mlp.gate` 输出第 0 项，或 `output_router_logits=True` 后的 `out.router_logits[i]`，形状 `[B*T, 512]`（`M:971`；`M:1381` OutputRecorder）。两条路已验证逐元素相等。
- 全量分布 $p=\mathrm{softmax}(\ell)$（`M:972`）。
- top-10 归一化权重 $\tilde p$ 与专家索引（`M:975`，`norm_topk_prob=True`）。
- 共享专家门 $s=\sigma(w_s^\top x)\in(0,1)$（`M:996`），形状 `[B*T,1]`。
- 路由器输入 $x$ = `mlp_hyper_connection` 的 `mixed_input`（`M:1306` 后进入 `self.mlp`）。

### 3.2 数学定义

- 全量路由熵：$H_{512}(p) = -\sum_{e=1}^{512} p_e\ln p_e \,/\, \ln 512 \in[0,1]$
- 激活集熵：$H_{10}(\tilde p) = -\sum_{e\in\text{top10}} \tilde p_e\ln\tilde p_e \,/\, \ln 10 \in[0,1]$
- 路由稀疏度（质量集中度）：$m_{10} = \sum_{e\in\text{top10}} p_e \in(0,1]$，即 top-10 拿走了全量分布多少质量。$1-m_{10}$ 是"被丢弃的路由质量"，直接量化 6B/125B 激活带来的信息截断。
- 路由边际：$\Delta_{10} = p_{(10)} - p_{(11)}$，第 10 与第 11 名的差；接近 0 表示专家选择处在决策边界。
- 层聚合：三种都存，不预设哪种有用：均值 $\bar H$、按深度分段（前/中/后 16 层）均值、最后 token 的逐层向量 $(H^{(0)},\dots,H^{(47)})\in\mathbb{R}^{48}$。
- 序列聚合：最后 token 值、掩码内均值、掩码内最大值。

归一化与 `NormalizedEntropy::from_probabilities` 一致（`crates/gen-zero-core/src/types.rs:194`：$H/\ln\max(K,2)$），这保证数值上能直接装进 `NormalizedEntropy(f32)`。

### 3.3 与 PolicyGate 的映射：写成假设，不写成能力

现状（`path:line` 核实）：

- Rust `PolicyGate::evaluate` 的唯一连续输入是 `NormalizedEntropy`，生产端在 `crates/gen-zero-service/src/zero.rs:2053` 用**候选动作概率**算这个熵，然后 `:2072` 送入 gate；阈值默认 0.65（`crates/gen-zero-gate/src/policy.rs:47`），超过则 Tier2 升级。`PolicyGate::default()` 无任何约束、无 confirm 动作、无 heat requirement（`policy.rs:43-50`；`docs/architecture/gen_zero_capability_audit_20260927.md` §3 已指出）。
- Python `DecisionPolicyGate` 用 `confidence` 三档（0.30/0.50）与 `risk_prob` 阈值 0.20（`python/gen_zero/gate/policy_gate.py:44-46`）。

**语义错配必须说明**：`zero.rs` 的熵是"在 K 个动作候选上犹豫多少"，路由熵是"在 512 个专家上分散多少"。两者是不同随机变量，不能直接把 $H_{512}$ 塞进 `evaluate(action, entropy)` 冒充动作熵。正确的接法是**新增一路输入**：

```text
GateVerdict evaluate_with_context(action, active_context, certificate,
                                  entropy: NormalizedEntropy,          // 现有：动作熵
                                  routing: Option<RoutingSignal>,      // 新增：{h512, h10, m10, margin, layer_band}
                                  fact_provider, agent_id)
```

映射假设（待配对统计检验，不是结论）：

| 路由信号 | 假设的认知含义 | 假设的门控动作 |
|---|---|---|
| $\bar H_{512}$ 高且 $m_{10}$ 低 | 输入落在专家分工的边界外（分布外） | 提升到 Tier2Escalate，或在 Python 侧把 `confidence` 下压 |
| $\Delta_{10}\approx 0$ 在深层持续 | 专家选择不稳定，等价于"多个假设并存" | Tier1Confirm 类动作要求二次确认 |
| $\bar H_{512}$ 低且 $s$（共享门）高 | 路由坍缩到少数专家，共享专家承担主干 | 不改变 tier；作为"过度自信"监控指标记录 |

**验收前置**：上表任何一行要变成"能力"，需要在同一批样本上同时拿到路由信号与真实结果（答对/答错、风险标签），做逐样本配对统计（如 AUC 与其 bootstrap 区间），并且候选动作熵作为对照基线同场比较。`docs/zero/29` 已说明本仓库此前没有一次合格的逐样本配对区间，这里不重复那个错误。

### 3.4 Hook 规范（Layer 2）

| 项 | 规范 |
|---|---|
| 位置 | `model.model.language_model.layers[i].mlp.gate`（forward hook，输出三元组）；`…mlp.shared_expert_gate`（输出 `[B*T,1]`）；`…mlp_hyper_connection`（输出三元组的第 0 项为路由器输入） |
| 精度 | logits 以 float32 落盘；`softmax` 用 float32（与 `M:972` 一致） |
| 落盘 | 每层每 token 的 512 维 logits 全存代价大（48×512×T）。默认存：最后 token 全 512 维 logits；所有 token 的 top-10 索引（int16）与权重（float16）、$H_{512}$、$H_{10}$、$m_{10}$、$\Delta_{10}$、$s$（float32） |

## 4. Layer 3：51B N-gram 查找表与连续流形对齐

### 4.1 它到底是什么（代码事实）

- 只有 layer 1（一索引 2）有 PLE（真实 config `ple_layer_ids=[2]`）。
- 输入不是隐状态，是 token id：把最近 3 个 id 分别乘每层随机奇数乘子后 XOR 混合，对每个头的素数词表取模（`M:1165-1171`），2-gram 8 头 + 3-gram 8 头共 16 头，每头查 `2560/16=160` 维，拼成 2560 维。**这是最近 3 个 token id 的确定性哈希函数**，与上下文无关，与前面层的表征也无关。
- 对齐是模型自己做的（`M:1246-1248`）：`key_proj` 把 2560 维投到 4 条流各 2560 维作 key，与当前流的 RMSNorm 后状态做点积得每流一个标量门 $g_c$，再做符号平方根压缩与 sigmoid，乘上 `value_proj` 的 2560 维 value，最后加一个膨胀深度卷积的局部上下文项，写回 10240 维残差（`M:1284`）。
- 表约 90 GiB bf16，只在一层触碰；截断加载时可以单独不载入（`scripts/analyze_flash_next_shard_layout.py` 已按 `ngram_table` 分类字节）。

### 4.2 局部流形表示与对齐方式

N-gram 嵌入张成的是一个**离散点集**：$E_{ng}=\{\phi(w_{t-2},w_{t-1},w_t)\}\subset\mathbb{R}^{2560}$，哈希碰撞使它不是单射。它不是连续流形，不能对它拟合动力学。能做且值得做的三件事：

1. **写入强度**：记每流门值 $\sigma(g_c)\in(0,1)$，$c=1..4$（`M:1248` 前的 `gate`）。这是"模型此刻多依赖词法先验"的直接读数；与 Layer 2 的 $H_{512}$ 同样是一个可门控信号。
2. **投影到主流形**：Layer 4/Pillar 4 得到 $U_k$（2560 维读门空间上的主基）后，把 PLE 的 value 输出 $v=\mathrm{value\_proj}(\phi)$ 投影为 $U_k^\top(v-\mu)$，得到 N-gram 先验在连续因果流形上的坐标。定义"词法先验偏移量" $\delta_{ng} = \|U_k^\top v\|/\|v\|$：接近 1 表示 N-gram 写入落在主流形内，接近 0 表示它写入了主流形之外的方向。
3. **反事实用途**：Pillar 2 的 $(x, do(x'))$ 若只改一个 token，N-gram 路径的变化是可精确计算的（哈希确定），可以把 $\Delta v$ 作为已知扰动源，从末层位移 $\Delta z$ 中减掉它对应的线性像，得到"去词法"的因果位移。这是一个**可做但未做**的分析。

### 4.3 Hook 规范（Layer 3）

| 项 | 规范 |
|---|---|
| 位置 | `layers[1].ple.ple_embedding`（输出 `[B,T,2560]` 原始拼接嵌入）；`layers[1].ple`（输出 `[B,T,10240]` 门控+卷积后写入量）；门值需在 `Qwen4ExpTextPLELayer.forward` 内部，hook 拿不到，用 `register_forward_hook` 于 `ple.norm_key`/`ple.norm_query` 分别取 key/query 后按 `M:1246-1247` 复算 |
| 落盘 | 最后 token 的 2560 维 $\phi$ 与 4 个门值；全 token 只存门值 |
| 说明 | 已验证两处宽度（微型模型 `ngram_embed_width=64`，`ple_output_width=256`，对应真实 2560 / 10240） |

## 5. Layer 4：混合结构中的跨层 CKA 相变探测

### 5.1 现有工具在这里会错

- `PhaseTransitionLayerExtractor.detect_phase_transitions`（`universal_manifold_extractor.py:173-182`）取相邻层 CKA 的裸 `argmin`。Flash-Next 每 4 层一个 QSA 层，QSA 层的输出统计与 GDN 层系统性不同，相邻 CKA 会在每个 QSA 层周期性下探。裸 `argmin` 会选中某个 QSA 层，那是**架构周期**，不是概念相变。
- `extract_concept_and_causal_manifolds` 用固定 `mid_fraction=0.5`（`:184-207`），与数据无关，对 48 层就是 layer 24（恰好是一个 QSA 层的下一层）。这不是探测，是常数。

### 5.2 定位规则

在哪个空间算 CKA，先定死：**默认在读门混合后的 2560 维**（每层两个读门，取 `mlp_hyper_connection` 的 `mixed_input`，它是 MoE 看到的输入，`M:1306`），另存 10240 维原始 4 流作第二产物。`collapse_streams` 的四流平均（`scripts/test_intermediate_layer_probe.py:99-105`）自己注明"未经验证"，本设计不用它作主口径。

定位分三步：

1. **同类比较**：分别在 GDN 子序列（36 层）与 QSA 子序列（12 层）内算相邻 CKA；再算跨周期 CKA，即 layer $l$ 与 $l+4$（同相位）。周期为 4 的相位混淆在同相位序列里消失。
2. **周期性对照**：把 48 层相邻 CKA 序列减去按相位（$l \bmod 4$）分组的均值，得到去周期残差；只在残差上找极小。若残差极小值的深度不比随机相位打乱后的极小值更极端（置换检验），就报告"未检出相变"，不得硬选一层。
3. **两个层位的定义**：
   - 概念相变层 $l_c$：去周期残差在 GDN 子序列上的最深显著极小，且要求 $l_c$ 之后到 $l_c+8$ 的同相位 CKA 稳定高于 $l_c$ 之前的水平（表征在此后"成形"）。
   - 因果坍缩层 $l_k$：从末层向前，同相位 CKA 首次跌破阈值（对末层混合后表征的 CKA < 0.9，阈值写进 manifest）的位置；坍缩流形取 `hyper_connection_mixer` 输出（`M:1493`）而非任一单层。
4. **QSA 层的额外读数**：每个 QSA 层的选择掩码密度 $\rho_{\text{sel}}$ 与 CKA 跳变一起记录，用于区分"检索模式切换"与"表征重组"。

### 5.3 与 Pillar 4（流式协方差）的关系

主基 $U_k$ 在 2560 维读门空间上累计。每层各累计一份的代价是 48 个 2560² 的 float64 矩阵（约 2.5 GB），可以接受；但 10240 维原始流的 48 份约 40 GB，不做，只对 $l_c$、$l_k$、末层混合输出三处累计。

## 6. 阻断性前置缺陷：协方差累计的灾难性抵消

`StreamingCovarianceAccumulator.covariance` 用 $C/N - \mu\mu^\top$（`universal_manifold_extractor.py:97-102`）。当特征均值远大于标准差时这是教科书式的抵消。复现（`covariance-cancellation.log`，退出码 0，合成高斯，非 Qwen 推理）：

| 均值偏移 | 直接法最大绝对误差 | 先减首批均值再累计 |
|---|---|---|
| 0 | 6.7e-16 | 1.1e-15 |
| 1e3 | 1.3e-9 | 6.7e-16 |
| 1e5 | 1.4e-5 | 6.7e-16 |
| 1e8 | **25.0**（真值量级 1） | 2.8e-13 |

为什么在 Flash-Next 上是阻断项：Qwen 系中间层有少数"巨量激活"通道，量级远超其余维度（`scripts/test_intermediate_layer_probe.py:13` 已记录，并因此对每层做 z-score）。GDN 状态与 10240 维原始流未经归一化，直接喂给累计器会把误差集中在这些通道，`compute_principal_basis` 的前几个主方向会被污染。

要求（供 owner 裁定，本设计不改生产代码）：

1. 累计器加 `shift` 参数（首批均值或外部给定），内部对 `x - shift` 累计，或改为 Welford 分批合并。
2. `covariance()` 加自检：若 $\max_i |\mu_i|^2 / \mathrm{Var}_i > 10^{6}$ 则**抛错**而不是返回结果（fail-closed）。
3. 单测覆盖偏移 1e8 的情形，判据 `max_abs_err < 1e-9`。

修复落地前，Flash-Next 的 Pillar 4 累计不得启动。

## 7. 数据保存格式规范

沿用 `export_codebook`/`load_codebook` 的 sha256 校验 npz 约定（`universal_manifold_extractor.py:469-500`），不另造二进制。它不是 `GZCBK001`（`knowledge_compiler.py` 头部布局），后者的 A/B/基向量由本产物喂入编译器生成。

**文件**：`<out>/flash_next_manifold_v1.npz`，`metadata` 键为 JSON 字符串，`load` 时校验 sha256、每个数组的形状与 `np.isfinite`，任一不符**抛错**，没有降级路径。

**manifest（`metadata`）必填字段**

```json
{
 "format": "flash-next-layered-manifold", "version": 1,
 "model": {"repo": "Qwen/Qwen3.8-Flash-Next", "revision": "de4b8e4d…", "config_sha256": "889658f2…",
           "transformers": "5.17.0", "torch": "…", "dtype": "bfloat16", "attn_implementation": "sdpa"},
 "hooks": {"read_gate": "layers[i].mlp_hyper_connection[0]", "residual": "layers[i]",
           "router": "layers[i].mlp.gate", "gdn_state": "cache.layers[i].recurrent_states[0]",
           "qsa_mask": "layers[i].self_attn.indexer", "ple": "layers[1].ple", "ngram": "layers[1].ple.ple_embedding"},
 "tokenization": {"padding_side": "left", "truncation_side": "left", "max_tok": 0, "head_tok": 0},
 "token_selection": "last_real_token | masked_mean",
 "layers": {"layer_types": ["linear_attention", "…"], "gdn": [0,1,2,4,…], "qsa": [3,7,…,47], "ple": [1]},
 "trajectory_layers": [0, "<l_c>", 46], "state_readout": "svd_r=8 | q_readout",
 "phase_transition": {"space": "read_gate_2560", "l_c": null, "l_k": null, "detrended": true, "permutation_p": null},
 "covariance": {"shift": "first_batch_mean", "dim": 2560, "k": 64, "n_samples": 0},
 "sample_ids": ["…"], "sha256": "…"
}
```

**数组命名**（`L{层号}` 两位十进制，`S` 为流索引 0–3）

| 键 | 形状 | dtype | 含义 |
|---|---|---|---|
| `rg_last/L{ll}` | `(N, 2560)` | f32 | 读门混合后表征，最后真实 token |
| `res_last/L{ll}/S{s}` | `(N, 2560)` | f16 | 原始 4 流之一（仅 $l_c$、$l_k$、47） |
| `mixer_last` | `(N, 2560)` | f32 | `hyper_connection_mixer` 输出 |
| `gdn_traj/L{ll}` | `(N, T_max, d_z)` + `gdn_traj_len/L{ll}` `(N,)` | f32 / i32 | Layer 1 轨迹与有效长度；$d_z$ 由 `state_readout` 决定 |
| `gdn_final/L{ll}` | `(N, 48, r, r)` 或 `(N, 6144)` | f16 | 其余 GDN 层最终状态（SVD 截断或 q 读出） |
| `router_logits_last/L{ll}` | `(N, 512)` | f32 | 最后 token 全量 logits |
| `router_topk_idx/L{ll}`, `router_topk_w/L{ll}` | `(N, T_max, 10)` | i16 / f16 | 全 token top-10 |
| `router_stats/L{ll}` | `(N, T_max, 5)` | f32 | $[H_{512}, H_{10}, m_{10}, \Delta_{10}, s]$ |
| `qsa_density/L{ll}` | `(N, T_max)` | f32 | 选择掩码密度 |
| `ngram_last` | `(N, 2560)` | f32 | 原始 N-gram 拼接嵌入 |
| `ple_gate` | `(N, T_max, 4)` | f32 | 每流写入门 $\sigma(g_c)$ |
| `cka/consecutive`, `cka/same_phase`, `cka/to_mixer` | `(47,)`, `(44,)`, `(48,)` | f64 | Layer 4 曲线 |
| `U_k`, `mean`, `eigenvalues` | `(2560,64)`, `(2560,)`, `(64,)` | f32 | Pillar 4（在 `rg_last/L{l_k}` 上） |
| `A`, `B`, `rho` | `(d_z,d_z)`, `(d_z,2560)`, 标量 | f32 | Pillar 3，逐 `trajectory_layers` 一组，键加 `/L{ll}` |
| `fit_residual/L{ll}` | `(N,)` | f32 | §2.4 一步预测相对残差，缺失则文件无效 |

**行对齐**：`sample_ids` 决定所有 `N` 维的顺序；跨模型对齐沿用 `benchmarks/suites/cross_model_manifold_alignment.py` 的"先验对齐再算几何"规则。

## 8. Hook 点总表（已在 `qwen4_exp` 微型模型上逐项验证）

| 层 | 模块路径（`model.model.language_model.` 前缀省略） | 输出形状（真实模型） | 验证证据 |
|---|---|---|---|
| L4 | `layers[i]`（DecoderLayer） | `[B,T,10240]` | `hooks-tiny.log` `decoder_layer_width` |
| L4 | `layers[i].attn_hyper_connection` / `mlp_hyper_connection` → `(mixed, streams, inject)` | `[B,T,2560]`, `[B,T,10240]`, `[B,T,4]`，`inject∈(0,2)` | 同上 `read_gate_mixed_width` |
| L4 | `hyper_connection_mixer` | `[B,T,2560]` | 同上 |
| L2 | `layers[i].mlp.gate` → `(logits, topk_w, topk_idx)` | `[B*T,512]`, `[B*T,10]`, `[B*T,10]` | 与 `out.router_logits[i]` 逐元素相等 |
| L2 | `layers[i].mlp.shared_expert_gate` | `[B*T,1]` | 同上 |
| L1 | `cache.layers[i].recurrent_states[0]`（GDN 层，`use_cache=True`） | `[B,48,128,128]` | `gdn_state_shape`；逐 token 与分块一致 6.4e-10 |
| L1 | `layers[i].linear_attn` 输出 | `[B,T,2560]` | 同上 |
| L1 | `layers[i].self_attn.indexer`（QSA 层） | `[B,1,T,kv_len]` bool | `qsa_mask_dtype`, `qsa_mask_density` |
| L3 | `layers[1].ple.ple_embedding` / `layers[1].ple` | `[B,T,2560]` / `[B,T,10240]` | `ngram_embed_width`, `ple_output_width` |

排除项：`model.visual`、`mtp.*`（`Qwen4ExpForCausalLM._keys_to_ignore_on_load_unexpected` 已忽略）。

## 9. 算力与执行边界

- 真实前向需要约 250 GB bf16 主干 + 90 GB N-gram 表；前 K 层截断加载（`scripts/test_intermediate_layer_probe.py:365 load_truncated`）已在微型模型上证明字节级只读所需分片。Layer 4 需要全部 48 层，Layer 1/2/3 可以在 K=2（L3）或任意 K 上分别做。
- Layer 1 轨迹的代价要写明：路径 (a) 是每个样本做 $T$ 次单 token 全模型前向（QSA 索引器还在 Python 里逐 query 循环，`M:` `Qwen4ExpTextQSAIndexer.forward`），不是一次预填充。该模式下 36 个 GDN 层的状态每一步都在缓存里，§2.3 "只存 3 层"是存储选择，不是算力选择。
- 本机与 dev/stg/ai-wsl 均无 GPU 信息可核验，本设计不对哪台机器能跑作任何承诺。

## 10. 证据清单

| 文件（`docs/zero/evidence/qwen38-extraction-design/`） | 内容 |
|---|---|
| `identity.log` | HEAD `fa6cddb6…` |
| `flash-next-config.json`, `flash-next-revision.json` | 真实 config 与 HF revision，HTTP 200 |
| `hooks-tiny.log`, `hooks-tiny.exit` | 钩子验证完整 JSON 报告，`EXIT=0` |
| `covariance-cancellation.log`, `.exit` | §6 复现，`EXIT=0` |
| `numerics.log`, `callers.log` | 早前会话留下的累计器诊断与调用方清单 |
| `commands.json` | 全部命令、退出码、日志尾部、来源文件 sha256 |
| `scripts/test_qwen4exp_extraction_hooks_tiny.py` | 可重跑的验证脚本，任何断言失败即非零退出 |
