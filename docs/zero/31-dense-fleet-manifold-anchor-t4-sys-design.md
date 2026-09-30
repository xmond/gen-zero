# 31 · 超大 Dense 模型（123B/180B/405B）单卡突破与流形锚点蒸馏系统设计

任务代号：b0927c-t4-sys。日期：2026-09-27。

**性质：设计交付，不是已实现、已加载或已上线声明。** 本文档不改动任何代码文件，只给出可执行的工程路线图、可核验的假设清单、以及每一步的验收判据。凡引用 `path:line` 的地方，均为当前工作树的真实现状，已用 `Read`/`grep` 逐条核实（HEAD `acb2c0c`）；凡是设计中的新组件，一律标注 **未验证** 或 **未完成** 并说明原因。

## 状态对照矩阵

本节是本次校订（HEAD `edb3d78`）新增的内容。本节核实截至本次校订，本文档正文里各结论是否仍然成立。正文本身不改动（保留原始设计推理过程），只在这里追加最新状态；正文与本节冲突处以本节为准。**注意：本文档 §0-§3 引用的 `path:line` 全部按 HEAD `acb2c0c` 核实，此后 `crates/gen-zero-service/src/zero.rs` 又经过十几个提交（ETF choice head 重写、contact manifold 接入等），行号已整体漂移约 +100 行**（例如 `validate_nanocore` 从 `zero.rs:346` 移到 `zero.rs:446`，`nanocore_ask` 从 `zero.rs:2146` 移到 `zero.rs:2246`）。函数名和行为本身未变，仅行号漂移；引用本文档 `path:line` 前请先用 `grep -n` 重新定位。

| 正文条目 | 原状态 | 当前状态 | 依据 |
| :--- | :--- | :--- | :--- |
| §0：GGUF 截断构建脚本 | 未完成（§3.1 表格标"新建...[未完成]"） | **已实现（构建工具本体），未接入抽取管线**。`scripts/slice_gguf_layers.py`（提交 `2e1ef0e` 新增，`655d533` 修复原子写入与校验，与本文档同日提交但晚于本文档、未在本文档写作时纳入）实现了 §1.2 路线 A 描述的"保留 `blk.0..K-1` + `token_embd`/`output_norm`，去掉 `output.weight`"字节级切片。**未完成的部分依旧未完成**：没有任何脚本调用它去驱动 `base.ServerEncoder` 单槽 `llama-server` 管线，§1.2 末尾要求的"GGUF 截断路径与 safetensors 截断路径数值一致性验证"仍未做（`grep -rln slice_gguf_layers .` 只命中脚本自身和它的单测） | `scripts/slice_gguf_layers.py:1-19`，`scripts/tests/test_slice_gguf_layers.py` |
| §2.2 步骤 5："client.py 目前不调用 Rust 服务... 这段连线是全新代码" | 未完成 | **部分已实现，部分仍未完成**。提交 `6561f1a..6fb3c83`（"wire bridge into client/CLI"等）确实新增了 `GenZero.load_manifold_anchor_artifact` / `project_hidden_to_nanocore_state` / `generate_nanocore_ask_payload`（`python/gen_zero/client.py:3190-3211`）和 `python -m gen_zero.cli anchor` 子命令（`python/gen_zero/cli.py:408-460`）。但 `generate_nanocore_ask_payload` 只**构造** MCP 请求 JSON，不发送；CLI 的 `--execute` 走的是 `GenZero.decide_nanocore`（`client.py:3230-3274`），这是一条**纯 Python 进程内**决策路径（用 Python 自己的 `ActionETFChoiceHead`），**根本不调用 Rust 服务**。也就是说："Python 生成 128 维向量" 已实现，"Python 把它发给正在运行的 Rust `nanocore_ask`" 仍然没有任何生产代码在做——今天唯一把这两端真正接起来的是 Rust 侧测试 `test_nanocore_live.rs`（通过子进程调用 Python 生成 payload，与 Rust 计算结果进行容差比较而非逐字节一致），目前尚无跨进程的生产包装器调用它 | `python/gen_zero/client.py:3190-3274`，`python/gen_zero/cli.py:408-460`，`crates/gen-zero-service/tests/test_nanocore_live.rs:1-25` |
| §2.4 里程碑 0：`StreamingCovarianceAccumulator` 数值修复、`detect_phase_transitions` 去 argmin、"online SVD" 改名 | 阻塞前置条件，未完成 | **仍未完成，原样阻塞**。`git log -S"StreamingCovarianceAccumulator" -- python/gen_zero/causal/universal_manifold_extractor.py` 与 `git log -S"detect_phase_transitions"` 均只命中最初引入该代码的提交（`35c00ae`），此后没有任何提交修改过这两处逻辑；"online SVD" 命名同样只在最初的设计文档提交（`605cf9d`）里出现过，未被改名。§1-§3 依赖里程碑 0 的所有后续步骤因此仍然全部未开始 | `git log --oneline -S"StreamingCovarianceAccumulator" -- python/gen_zero/causal/universal_manifold_extractor.py` |
| §2 隐含假设："GCCA 多视角干涉"是 128 维锚点基的输入 | 本文档未直接断言，但标题任务描述隐含此链路 | **今天唯一跑通的锚点基与 GCCA 无关**。`grep -rn "ManifoldAnchorDistiller(" .` 显示 `.fit()` 只被两处调用：它自己的 CLI（`manifold_anchor_distiller.py:325`）和 `profile_nanocore_latency.py:132` 的**合成随机数据自检**（该函数自己的 docstring 写明"Throwaway artifact fit from random data; checks the harness, not real latency"）。唯一真实产物 `benchmarks/results/manifold/distilled_128d_llama70b_boolq.npz` 是直接在 **LLaMA-70B 原始隐藏特征**（`/ebs/data/extracted_features/llama70b/boolq.npz`）上拟合的，中间没有经过 `python/gen_zero/manifold/gcca_fusion.py` 的多视角融合。GCCA 融合组件（`python/gen_zero/manifold/gcca_fusion.py`，有 24 项 GCCA 单测，manifold 包合计 60 项覆盖）独立存在并由 `cli.py manifold-fuse` 调用，但并未被那个已于 2026-09-29 删除的三模型高斯随机投影基准脚本（其 123B 特征依赖从未真实存在，见 `docs/manuals/closed_loop_dense_pipeline.md` 文首移除说明）或锚点基管线使用，和这条锚点基管线目前是两个互不调用的独立系统，不是同一条链路的两段 | `python/gen_zero/causal/manifold_anchor_distiller.py:325`，`benchmarks/suites/profile_nanocore_latency.py:117-132`，`crates/gen-zero-service/tests/fixtures/nanocore_anchor_state_boolq_row0.json` |
| §1：教师模型是 123B/180B/405B | 设计目标 | **今天唯一有真实产物的教师是 LLaMA-70B，不在 123B/180B/405B 名单里**，且用的也不是 §1 设计的层截断路线，而是完整模型的末层特征（`extracted_features/llama70b/boolq.npz` 是常规抽取，不是 GGUF 截断产物）。123B/180B/405B 三个模型上，本文档 §1-§3 描述的每一步仍然是设计阶段，无新证据 | 同上 fixture provenance 字段 |
| §3.1："当前仓库内没有任何脚本设置 `GENZERO_NANOCORE_PATHS`" | 未完成 | **仍然成立**，本次校订用 `grep -rn GENZERO_NANOCORE_PATHS --include=*.sh --include=*.bat --include=Makefile -r .` 复核，零命中 | 同上 grep |

**本节结论**：doc 31 正文对"405B 单卡截断 + GCCA 多视角融合 + 128 维锚点基 + NanoCore 挂载"这条完整链路的判断——大部分环节未完成——依然成立，且比正文写作时更精确：GGUF 截断工具已经单独造出来了，但没接进抽取管线；Python→Rust 的载荷构造已经写了，但没有一条生产代码路径真正发送它；唯一走通的最小闭环（LLaMA-70B 单模型特征 → 128 维正交投影 → `nanocore_ask` → 门控 → 决策）刻意绕开了 GCCA，教师也不是 405B。这个最小闭环的完整实操步骤见 [`docs/manuals/closed_loop_dense_pipeline.md`](../manuals/closed_loop_dense_pipeline.md)。

---

## 0. 先把地基说清楚：任务描述的系统现在不存在

任务描述假定"GGUF 流形抽取 → Rust 决策引擎"是既有生产链路。核实结果不支持这个假设，必须先纠正，否则后面的方案会建在空中。

**Rust 工作区对 GGUF / GPU 一无所知。**
- 11 个 crate（`Cargo.toml:3-14`）里，`crates/gen-zero-model` 的全部内容是注意力掩码、ETF 选择头、prompt 消毒（`crates/gen-zero-model/README.md:1-5`，文件只有 `mask.rs/choice_head.rs/sanitize.rs/error.rs`）——没有模型加载、没有层、没有量化类型的概念。
- `crates/gen-zero-lod/src/manifold.rs` 里的"manifold"是混合曲率黎曼几何（双曲/球面积空间，用于 entailment gate 的数学），和 LLM 隐藏层完全无关。
- 全仓 `grep -rniE "cuda|cublas|\bgpu\b|ngl|n_gpu_layers" crates/*/Cargo.toml crates/*/src/*.rs Cargo.toml` 零命中。Rust 侧是纯 CPU 运行时。

**现有的 GGUF/llama-server 抽取是"单一 pooled 向量"，不是逐层流形。**
`benchmarks/suites/gpu_extract_*_13tasks.py` 系列（`base.ServerEncoder`，`gpu_extract_gemma26b_13tasks.py:94-174`）对每条文本只取 `--pooling last` 的**末层 post-norm 最后一个 token** 的单一向量，明确写着"the final post-norm state of the last token at its native scale"（`gpu_extract_qwen72b_13tasks.py:17-20`）。它已经做对的部分：单槽强制校验（`total_slots != 1` 拒绝启动，`gpu_extract_qwen72b_13tasks.py:81-89`）、模型文件哈希/路径核对（`:100-111`）、`--resume` 时校验旧产物完整性而非静默跳过（`:153-179`）、原子写入（`.tmp.npz` + rename，`:146-150`）。这些是可以直接复用的 fail-closed 基础设施，但它抽的是**1 个向量**，不是分层流形。

**"四支柱"流形抽取代码是真实存在的，但从未在真实大模型上跑过，且已知有数值缺陷。**
`python/gen_zero/causal/universal_manifold_extractor.py` 确有 `StreamingCovarianceAccumulator`（流式协方差/PCA）、`PhaseTransitionLayerExtractor`（相变层探测，CKA）、`CounterfactualGridSampler`、`LyapunovPhaseSpaceReconstructor` 四个类。但：
- `PhaseTransitionLayerExtractor.detect_phase_transitions` 对 CKA 序列直接取 `np.argmin`（`universal_manifold_extractor.py:213,225`），`docs/zero/30-...md:189` 已指出：对周期性架构（如 Flash-Next 的 QSA 层），这会选中"架构周期"而非"概念相变"。
- `StreamingCovarianceAccumulator` 有已复现的灾难性抵消缺陷（`docs/zero/30-...md:15,209-228`），且把全量 `eigh` 叫作"online SVD"是术语误用（`docs/zero/qwen38-flash-next-extraction-system-design.md:31`）。
- 项目自己的设计文档写得非常直白："not one real Flash-Next forward pass has been run. This machine has no GPU... the 360GB of weights are not local"（`docs/zero/30-...md:19`），"日期：2026-09-27。性质：设计交付，不是已实现、已加载或已上线声明"（`docs/zero/qwen38-flash-next-extraction-system-design.md:3-4`）。

**"NanoCore"确实是能承接 128 维决策向量的生产组件，但它不产生这个向量，只消费它——这是本方案唯一站得住脚的挂载点，下面会用到。**
`crates/gen-zero-nanocore/src/core_type.rs:19-31`：`NanoCoreInstance` 携带 `projection_weights: Vec<f32>`，注释明写"shape (out_dim, 128)"，即固定吃 128 维输入（`CompressedLatent = LatentState<128>`，`crates/gen-zero-core/src/types.rs:151`），投影到 `out_dim`（≤4096）。生产侧已经有完整的 fail-closed 校验（`validate_nanocore`，`crates/gen-zero-service/src/zero.rs:346-373`：`out_dim` 范围检查、`projection_weights.len() == out_dim*128` 精确匹配、逐元素有限性检查）和真实调用链（MCP `zero` 工具 `engine=nanocore` 分支 → `nanocore_ask`，`zero.rs:2146-2500`：校验 `nanocore_state` 必须恰好 128 个有限数，`zero.rs:2161-2176`）。**这条链路今天就能跑，只是没人喂给它跟三个大模型有关的 128 维向量。**

结论：这份报告要设计的，不是"接入一个已有的抽取管线"，而是**从零建一条离线抽取→蒸馏管线，把它的产物塞进已经存在、已经生产挂载、但目前空转的 NanoCore 128 维入口**。下面 §1-§3 就按这个真实拓扑展开。

**`crates/gen-zero-model` 在这条路径上零参与，写明理由避免误导**：该 crate 唯一的决策相关组件 `ActionETFChoiceHead` 走的是 `head=etf` 分支，而 `engine=nanocore` 请求里出现 `etf_rep` 会被显式拒绝（`crates/gen-zero-service/src/zero.rs:252`："head != "etf" && has("etf_rep") || engine == "nanocore" && has("etf_rep")" 触发 Rejection）。两条决策后端互斥，`gen-zero-model` 不在 nanocore 挂载路径上，本方案不涉及它。

---

## 1. 405B/180B 单卡突破：层截断是唯一可行路径，不是优化项

### 1.1 先算账，不猜

三个模型的层数/隐藏维度（`.bat` 注释里能直接核实的部分，标注来源）：

| 模型 | 层数 | 隐藏维度 | 量化 | 权重体积 | 来源 |
|---|---|---|---|---|---|
| Mistral Large 2 (123B) | 88 | 12288（外部规格，仓库内未见数字，需部署前用 `scripts/inspect_gguf_layer_bytes.py` 对真实文件核实） | Q3_K_M | ~60GB（估计，`run_mistral123b_extract.bat:13,21` 明写"ESTIMATE, not measured"） | `benchmarks/suites/run_mistral123b_extract.bat:13` |
| Falcon-180B | 80 | 14848 | Q2_K | ~74GB | `benchmarks/suites/run_falcon180b_extract.bat:17-19` |
| Llama 3.1 405B | 126 | 16384（外部规格） | Q2_K | ~141GB | `benchmarks/suites/run_llama405b_extract.bat:2`，任务描述 |

405B 是唯一物理上装不进 80GB 显存的（141GB > 80GB）。可用显存预算：80GB 减去 KV cache（`-c 2048` 量级下几百 MB 到几 GB）和 compute buffer（估计 2-5GB，参照 Falcon 180B 注释口径），按 ~75GB 可用算：

```
每层平均权重 ≈ 141GB / 126 ≈ 1.12 GB/层
K_max ≈ 75GB / 1.12GB ≈ 67 层
```

即：**截断到约 64-67 层（126 层的一半左右）之后，剩余的模型可以整个常驻显存，不需要任何 CPU-GPU 换页或 NVMe 流式加载。** 这不是"提速的可选项"，是唯一让 405B 能被单卡处理的路径——`run_llama405b_extract.bat:25` 里 `-ngl 50` 本身就被标注"a starting configuration, NOT a measured fit"，说明连现有脚本自己都没有真正验证过这个配置能跑通。

对 180B（Q2_K ~74GB，`run_falcon180b_extract.bat:18`）：不截断也能塞满卡（"at the edge of 80GB"），截断在这里的收益是把显存余量让给更大的 batch/更长 context，不是生死问题。
对 123B（Q3_K_M ~60GB）：整模型就能跑，是唯一能拿到"全层真实激活"作为基准（ground truth）的模型——这决定了下面 §1.3 的实验顺序。

### 1.2 层截断的两条技术路线，选一条，不是都做

**路线 A：物理截断 GGUF 文件 + 复用现有单槽 llama-server 管线（推荐）**

GGUF 格式的张量表带绝对偏移量，`scripts/inspect_gguf_layer_bytes.py:6-8`（已验证的真实代码，纯 stdlib 解析头部，不读权重）已经证明"a first-K-layers slice is a set of byte ranges, exactly like safetensors"。做法：
1. 用 `inspect_gguf_layer_bytes.py` 解析出每个 `blk.N.*` 张量的偏移和大小。
2. 写一个新的 GGUF 文件：保留 `blk.0..K-1` 的全部张量 + token embedding，修改元数据 `n_layer` = K，追加原模型的 `output_norm`/`output`（或者去掉 `output`，只留 `output_norm`，视是否需要 logits 而定；本方案只要隐藏状态，`output` 张量可以裁掉，省下一大块——Mistral/Llama 类模型的 `lm_head` 通常是 `hidden × vocab`，vocab 3-15 万，裁掉能再省几百 MB 到几 GB）。
3. 用现有的、已经过 fail-closed 校验的单槽 `llama-server --embedding --pooling last` 管线跑这个"K 层伪模型"，直接复用 `base.ServerEncoder`（`gpu_extract_gemma26b_13tasks.py:94-174`）和 `Qwen72bEncoder` 的槽位/模型哈希/上下文长度三重校验（`gpu_extract_qwen72b_13tasks.py:81-111`）。

**必须写明的失真**：这样拿到的是 `output_norm(h_K)`，不是原模型第 K 层的原始隐藏状态 `h_K`。Mistral Large 2 / Llama 3.1 是 RMSNorm，失真是"除以均方根、乘缩放系数"；**Falcon-180B 用的是带 bias 的 LayerNorm（先减均值、再除以标准差、再乘缩放加偏置），不是 RMSNorm**——三个模型里不能套用同一个"RMSNorm(h_K)"的说法，Falcon 必须单独处理归一化逆变换。这是一个每个模型固定的可预测偏差，不是随机噪声，但下游如果需要"原始 `h_K`"必须显式做归一化逆变换或接受这个偏差——**这一步目前没有代码验证过两者的差异有多大，标记为未验证**。

**路线 B：llama.cpp eval-callback 精确取 h_K（不推荐，工作量大）**
用 `llama.cpp` 的 `cb_eval` 或 `llama-cpp-python` 的回调机制，在第 K 层算完时直接截获原始隐藏状态，不用改文件。精确，但需要写全新的 C++/Python 桥接代码，且完全没有现有的 fail-closed 校验可以复用——`gpu_extract_*` 系列的槽位/哈希/续跑校验全部要重写一遍。

选路线 A：能复用已验证的生产级单槽管线（`total_slots` 检查、模型哈希核对、原子续跑），代价是接受一个已知、可测量、可文档化的 `output_norm` 失真。路线 B 精度更高但等于重新造一遍轮子，且目前没有资源去验证新桥接代码的正确性。

**验收判据（路线 A 落地前必须拿到，缺一不可）**：
- 用 `scripts/test_intermediate_layer_probe.py` 里已经验证过的 `rel_err` 方法（`max_abs`/`rel_l2`/`mean_cos`，`test_intermediate_layer_probe.py:422-425`）——但注意它是 safetensors + HF `transformers` 路径，跟 GGUF + llama-server 路径是两套完全不同的代码，**目前没有任何代码把两者的输出做过交叉验证**。落地前必须先跑一次：小模型（如 Qwen3.5-9B，已有权重）分别用 safetensors 截断加载和 GGUF 截断加载取同一层，比较 `rel_l2`/`mean_cos`，确认两条路径数值一致，再推广到 123B/180B/405B。这是一个新的、当前未完成的验证步骤，理由：目前两条路径分别独立验证过，从未互相对照过。

### 1.3 相变层 K 怎么定，不是猜一个数

`docs/zero/30-...md:189` 已经指出 argmin CKA 方法在周期性架构上会选错层。本方案不猜测任何具体 K 值，而是规定实验顺序：

1. **先在 123B 上做全层扫描**（它是三个模型里唯一能整模型装进显存、能拿到全部 88 层真实激活的）。用现有 `linear_cka`/`compute_cka_matrix`（`universal_manifold_extractor.py:154-179`，这两个函数本身没有 argmin 问题，问题只在 `detect_phase_transitions` 的判定规则）算出全层 CKA 矩阵，人工核实相变候选层，不采信自动 argmin 的原始输出。
2. **同时修正 `detect_phase_transitions`**：不能再用原始 `argmin`（`universal_manifold_extractor.py:213,225`——这是要被替换掉的旧逻辑，不是扩展），改成周期感知的去趋势 CKA（`docs/zero/30-...md` §5 已给出方法，本方案采纳）。
3. 123B 的 K 校准结果（一个"相对深度比例"，例如"K/88 ≈ 0.6"这种可迁移的相对指标，而非绝对层号）作为先验，应用到 180B/405B，但**必须在 180B/405B 各自的小规模抽样上复核**，不能直接照搬绝对层号（不同架构的深度语义不可比）。

在拿到 123B 真实扫描结果之前，任何"第几层是相变层"的具体数字都是未验证的猜测，本报告不给出。

**量化对流形保真度的影响，与显存账目分开算，任务明确要求但目前缺失这一半**：`run_falcon180b_extract.bat:17-23` 和 `run_llama405b_extract.bat` 用的是 Q2_K，是 llama.cpp 量化方案里最激进的档位之一。显存账目（§1.1）只回答"装不装得下"，回答不了"Q2_K 量化误差会不会把 `h_K` 的流形结构本身搞坏，导致蒸馏出来的教师目标是在教一个失真的流形"。这是**未验证**项，落地前必须做的测量：在 123B 上（它有 Q3_K_M 现成配置）分别跑 Q3_K_M 和一个更低档位（如 Q2_K 或 Q4_K_M 对照），取同一批输入的第 K 层输出，逐样本算余弦相似度分布（不是均值一个数），确认量化误差在流形保真度上是否可接受。如果 405B/180B 只有 Q2_K 权重可用、又测出严重失真，整个蒸馏管线的教师质量就有上限，这个上限必须在里程碑0之后、真实训练开始之前测出来，不能假设"反正装得下就能用"。

### 1.4 CPU-GPU 动态置换 / NVMe 流式加载

`scripts/prototype_layer_streaming.py` 已经验证了机制（不是模型）：合成权重（`HIDDEN=5120`，40 头，与任何真实模型无关，`prototype_layer_streaming.py:38-44`）对比三种策略——`full_preload`（全常驻）、`naive_offload`（单缓冲同步换页）、`double_buffer`（双缓冲 + 后台线程预取 + 独立 CUDA stream，H2D 拷贝与计算重叠）。**这证明了双缓冲流水线这个机制本身有效，但从未在真实模型权重上测过**，标记未验证。

在 §1.1 的账算清楚之后，180B/405B 截断到 K 层后模型整个装进显存，**双缓冲换页在本方案里其实用不上**——这是 §1.1 算账带来的直接推论：如果目标只是拿到第 K 层的隐藏状态（不需要跑完整个模型），层截断已经让问题变成"一个能装进显存的小模型"，NVMe 流式加载这类"边算边加载后半段权重"的机制不需要，因为后半段权重根本不加载。双缓冲/NVMe 流式加载只在"需要探索多个候选 K 值、且每次都要重新加载不同截断点"这种迭代场景下有价值（避免每次都从磁盘重新读全部截断层），属于 §1.3 实验加速的工程优化，不是主链路的必需项。

---

## 2. 离线流形锚点蒸馏 + NanoCore 瞬时投影：谁生产 128 维向量

### 2.1 真正的缺口是"生产者"，不是"投影本身"

`NanoCoreInstance` 的 `projection_weights` 是 `(out_dim, 128)`——输入端已经定死是 128 维（`crates/gen-zero-nanocore/src/core_type.rs:19-31`），且这 128 维在运行时必须由调用方在请求里直接给出（`nanocore_state`，`crates/gen-zero-service/src/zero.rs:2161-2176`，恰好 128 个有限数，维度不对直接拒绝，不做静默填充或截断）。

问题是：CPU-only 的生产服务器，请求到来的那一刻，谁来算出这 128 维？它不可能是"对 16384 维教师隐藏状态做一次投影"——**生产环境从来没有、也不会有一次真实的 405B 前向计算**。任务描述里"16384 维到决策空间的连续动力学积分"这句话，如果理解成"每次请求都拿到一个真实的 405B 第 K 层隐藏状态再投影"，本身就是一个不成立的生产者假设，必须先戳破，再往下设计。

### 2.2 正确的角色划分：投影矩阵是训练时工具，不是运行时权重

参照 `docs/zero/README.md:3-5` 自己的定位（"9B 教师模型...通过结构化参数切片抽取与反向传播连续因果流形蒸馏，将知识与反事实动力学压铸入 Zero 权重中"），本方案采用同样的角色分工，扩展到 123B/180B/405B 三个新教师：

**离线（一次性，在有 GPU 的机器上跑）**：
1. §1 的截断+抽取管线，对每个模型的相变层 K，跑一批覆盖真实任务分布的输入，拿到 `(input, h_K)` 对。
2. 用修正后的 `StreamingCovarianceAccumulator`（先修数值缺陷，见 §2.4）算出 h_K 的主成分/薄 SVD 字典，或者训练一个保角投影（conformal projection），把 h_K 映射到一个 128 维目标空间——这一步的输出是**一个投影矩阵，作为训练目标生成器，不作为运行时权重加载**。
3. 用这个投影矩阵，把整批 `(input, h_K)` 转换成 `(input, target_128d)` 监督对。
4. **离线训练一个 CPU 友好的小编码器**（复用现有 distiller 基础设施：the distiller in gen-zero-research (moved out of this repo)），让它直接从原始输入（不经过教师模型）预测 `target_128d`。这个小编码器就是 128 维向量真正的**生产者**——它在运行时跑在 CPU 上，不依赖教师模型，也不依赖 GPU。

**这里必须拆成两件不同的产物，不能混为一谈**——`NanoCoreInstance.projection_weights` 的形状是 `(out_dim, 128)`（`core_type.rs:19-31`），即它吃 128 维、吐 `out_dim` 维，跟步骤 4 的"原始输入 → 128 维"编码器方向正好相反，物理上不可能把步骤 4 的编码器直接打包成 `NanoCoreInstance`。真正需要落地的是两个独立产物：

- **产物 (a)：客户端 128 维编码器**（步骤 4 的产物）。它是原始输入 → 128 维 `CompressedLatent` 的映射，跑在调用方一侧（`python/gen_zero/client.py` 或其他上游服务），**不装进 `NanoCoreInstance`，也不在 Rust 侧加载**——它只是把 128 维数字算出来，作为请求参数的一部分发给 Rust 服务。
- **产物 (b)：服务端 `NanoCoreInstance`**（128 维 → `out_dim` 决策分数）。它的 `projection_weights`/`value_weights` 不能是随机初始化或占位值——必须用 `(target_128d, decision_label)` 监督对（decision_label 来自具体业务场景的历史决策结果或人工标注，而非教师隐藏状态本身）拟合出来，否则 128 维向量送进去乘的是一个跟流形无关的矩阵，整条链路在决策语义上是空的，只是形状对得上。**这一步的训练数据（决策标签）目前完全没有，是本方案当前最大的未完成项**，比编码器训练更靠后，因为它依赖具体业务场景定义"正确决策"是什么。
5. 把产物 (a) 部署到调用方能执行的位置（Python 进程内或独立微服务），把产物 (b) 按模型/领域分文件导出为 `NanoCoreInstance` 期望的 JSON 格式（`domain_id/name/prototype/projection_weights/value_weights/out_dim/base_confidence`，序列化格式见 `load_nanocores_from_paths` 用 `serde_json::from_slice` 反序列化，`crates/gen-zero-service/src/zero.rs:401-403`），放到 `GENZERO_NANOCORE_PATHS`（常量名 `NANOCORE_PATHS_ENV = "GENZERO_NANOCORE_PATHS"`，`zero.rs:324`；文档记录的操作员配置项，`crates/gen-zero-service/README.md:330`；**当前仓库内没有任何部署脚本设置这个变量**，即：生产代码路径已存在，但没有部署配置去触发它加载任何真核）指向的路径。

**在线（生产请求路径，CPU only）**：
1. 请求进来 → 产物 (a) 编码器（跑在 CPU，不是教师模型）算出 128 维 `CompressedLatent`，耗时见 §2.3。
2. 这 128 维作为 `nanocore_state` 随 MCP `tools/call zero` 请求（`engine=nanocore`）发出。**这一步在今天的调用链路里是新代码，不是"接入现有链路"**：`python/gen_zero/client.py` 目前不调用 Rust 服务，实际连线是反过来的（Rust `gen-zero-service` 的 `bridge.rs` → HTTP → Python `app.py`）。要让产物 (a) 的输出真正走到 Rust，需要新写一个"调用方 → MCP `zero` 工具"的客户端代码路径，`client.py` 本身没有这段代码，必须新建。
3. 请求到达后，`nanocore_ask`（`zero.rs:2146-2500`）走**现有的、今天就能跑的**校验+推理链路：`validate_nanocore` 维度/有限性检查（`zero.rs:346-373`）→ `NanoCoreFleetScheduler`（`crates/gen-zero-nanocore/src/scheduler.rs`，有界 RAM LRU）+ `MoVFusionEngine`（`mov.rs`，多领域向量融合）完成决策，返回。

这样"16384 维教师隐藏状态"和"CPU 上的亚毫秒决策"之间没有任何运行时依赖关系——教师只在离线训练时出现过。这是唯一能同时满足"CPU-only 生产"和"亚毫秒延迟"的架构，但**只有产物 (b) 真正用决策标签训练过，这条链路才携带决策信号；否则它是一条形状正确、语义为空的管道，同样构成孤岛（有调用，无信号）**。

### 2.3 体积和延迟，按能对上号的方向算

`NanoCoreInstance::projection_weights` 是 `(out_dim, 128)`，f32 下 `out_dim × 128 × 4` 字节。如果小编码器本身也是一个线性/浅层网络（输入维度取决于原始特征，不是 16384——它从原始输入算起，不需要教师隐藏状态），体积由编码器架构决定，不是"教师投影矩阵"决定，这里不能简单套用 `16384×128` 的算法去宣称"数十 MB"。诚实的说法是：

- 教师端投影字典（离线工具，一次性）：单个模型 `16384×128 f32 = 8.4MB`（Llama 405B），`14848×128 f32 ≈ 7.6MB`（Falcon 180B），`12288×128 f32 ≈ 6.3MB`（Mistral 123B），三个模型合计约 **22MB f32 / 6MB int8 量化**——这个数量级符合"数十 MB"的说法，但它是离线训练工具，不装入生产服务器。
- 生产端小编码器：体积取决于编码器架构选型（本报告不预设，需要在 §1.3 拿到真实教师目标之后，用验证集做架构搜索），这是**未完成**项，原因是目前没有真实的 `(input, target_128d)` 监督数据可供训练。

单次 `128×out_dim` 的线性投影（`MoVFusionEngine` 内部）本身确实是亚毫秒级：`out_dim` 取上限 4096 时 MACs ≈ 128×4096 ≈ 52 万次，单核 AVX-512 按 ~50 GFLOPS 估算 ≈ 0.02ms。**但这只是 NanoCore 内部那一次投影，不是"16384 维教师隐藏状态到决策空间"的端到端延迟**——端到端延迟的大头在小编码器的前向计算，其耗时由 §2.2 步骤 4 尚未确定的架构决定，此刻无法承诺一个具体数字。任务描述里"亚毫秒级完成 16384 维到决策空间的连续动力学积分"这个表述，只有把"16384 维"理解为"教师侧离线训练用到的维度"而非"运行时输入维度"才成立；作为运行时端到端承诺，本报告不背书，标记未完成，原因：小编码器架构未定、无真实训练数据。

### 2.4 阻塞前置条件（里程碑 0，必须先做，否则后面全部作废）

`StreamingCovarianceAccumulator` 的灾难性抵消缺陷（`docs/zero/30-...md:15,209-228`）必须先修复。任何在它之上算出来的 SVD 字典/投影矩阵都会继承这个数值错误，蒸馏出来的小编码器目标本身就是错的——修好之前，§2.2 的整条离线链路不能开始跑真实数据，只能在合成数据上验证机制。这是本方案的里程碑 0，没有条件绕过。

同时要处理的命名问题：把全量 `eigh` 叫"online SVD"是术语误用（`docs/zero/qwen38-flash-next-extraction-system-design.md:31`），要么真的实现增量 SVD（如 Brand 算法），要么把函数/文档里的"online SVD"字样改成准确名称——这两个是本方案要求物理替换掉的旧符号，不是保留兼容层。

---

## 3. Rust 主干挂载拓扑 + 端到端流程 + 降级/告警 + 验收指标

### 3.1 Caller → Callee 拓扑（区分"已存在，今天能跑"和"需要新建"）

```
【离线，一次性，GPU 机器，不进生产服务器】
benchmarks/suites/run_{mistral123b,falcon180b,llama405b}_extract.bat  [已存在，需按§1.2改造成截断GGUF版本]
  → 新建：GGUF 截断构建脚本（读 inspect_gguf_layer_bytes.py 的偏移表，写新 GGUF）  [未完成]
  → 复用：base.ServerEncoder 单槽 llama-server 管线                    [已存在，gpu_extract_gemma26b_13tasks.py:94-174]
  → 修复：StreamingCovarianceAccumulator（先修数值缺陷）                [里程碑0，未完成]
  → 修复：PhaseTransitionLayerExtractor.detect_phase_transitions（换掉 argmin）[未完成]
  → 新建：教师投影字典（SVD/conformal，128维目标空间）                 [未完成，依赖里程碑0]
  → 复用：gen-zero-research distiller (moved out of this repo)（训练产物(a) CPU 小编码器）[已存在框架，需接入新数据源]
  → 新建：产物(b) NanoCoreInstance 拟合器（128维目标+决策标签→projection_weights/value_weights）[未完成，缺决策标签数据]

【在线，生产请求路径，CPU only】
调用方（新建代码：python/gen_zero/client.py 目前不调用 Rust 服务，需新写 MCP 客户端路径）
  → 产物(a) 编码器前向（CPU，新建产物）→ 128维 CompressedLatent    [依赖上面的离线产物]
  → MCP tools/call "zero"，engine=nanocore，nanocore_state=<128维>   [服务端已存在，crates/gen-zero-service/src/zero.rs:2146；调用方是新代码]
  → validate_nanocore 维度/有限性校验                                 [已存在，zero.rs:346-373]
  → NanoCoreFleetScheduler + MoVFusionEngine 融合产物(b)的决策         [已存在，crates/gen-zero-nanocore/src/{scheduler,mov}.rs；需先加载产物(b)]
  → ZeroToolOutcome → MCP/HTTP 响应                                   [已存在，server.rs:1253-1316]
```

**这张图刻意分两段，是为了不让"投影矩阵"变成一个 0 调用孤岛**：教师投影字典在离线段被产物(a)编码器的训练过程实际消费；产物(b)（真正携带决策信号的部分）在在线段被 `nanocore_ask` 实际调用——但这个调用今天没有任何调用方代码去触发（`python/gen_zero/client.py` 现状是反向连线：Rust `bridge.rs` → HTTP → Python `app.py`，客户端到 MCP `zero` 工具这一段是全新代码，不是"复用现有链路"）。

### 3.2 降级与告警：不允许静默 bypass

- `nanocore_state` 维度不对、含非有限数：**已经是 fail-closed**，`zero.rs:2161-2176` 直接拒绝，不需要新代码。
- 投影产物（`NanoCoreInstance` 文件）缺失或加载失败：`load_nanocores_from_paths` 对文件读取失败、JSON 反序列化失败（`serde_json::from_slice`）、`validate_nanocore` 校验失败、重复 `domain_id` 四种情况全部 `unwrap_or_else(|error| panic!(...))`（`zero.rs:399-415`）——**已核实为 fail-closed：任何一个配置的核加载失败，整个服务启动直接 panic 退出，不是跳过单个文件继续跑**。这个行为本身是对的（不会带着一个坏核悄悄上线），落地时不需要改。
- 小编码器输出如果是全零/NaN（离线训练失败或加载了错误模型的编码器）：不能静默传给 NanoCore 当正常请求处理——上游客户端（新建的 MCP 客户端代码）在发送 `nanocore_state` 前必须做有限性自检并显式报错，这是新代码，**未完成**，需要在离线产物导出时同时导出预期输入分布的统计量（均值/方差/范数范围），供在线侧做异常检测，而不是等 Rust 侧兜底。
- 严禁的反模式（对照任务的防腐要求逐条排除）：不允许"NanoCore 里没有对应领域的核，就退回暴力全模型推理"这种静默降级——今天的代码已经是反过来的：`operator_domains.is_some() && matches!(specialized, Ok(true))` 时显式拒绝"nanocore_domain(s) cannot combine with engine/head decision backends"（`zero.rs:1491-1498`，代码逻辑本身，不是靠测试名推断），落地时要保持这个设计，不能为了"总要有个结果"而加兜底路径。
- **"Speculative Latent Cache"（任务标题提到的概念）在本方案里的定位**：只能是"精确输入命中缓存"这一层，加在产物(a)编码器前面——请求的原始输入如果和历史见过的输入完全一致（或在严格定义的近邻半径内），直接返回缓存的 128 维向量，省掉一次编码器前向。它解决不了"生产者"问题（不认识的新输入仍然要靠编码器），只是重复请求的加速层。**缓存未命中必须落到编码器计算，绝不能返回零向量或缓存里最近的一条充数**——这是需要在实现时写清楚的 fail-closed 规则，不是本方案默认行为。

### 3.3 性能基准与验收指标（分层，不合并成一个数字）

| 阶段 | 指标 | 判据 |
|---|---|---|
| 里程碑0：数值修复 | `StreamingCovarianceAccumulator` 在已知构造的病态输入上不再抵消 | 单元测试：构造一组已知会触发灾难性抵消的输入，修复前后对比误差，误差量级需回到浮点精度范围 |
| §1.3：相变层扫描 | 123B 全层 CKA 矩阵 + 人工核实的候选相变层 | 产出 CKA 矩阵 + 至少 3 个独立视角（架构先验、CKA 极值、下游任务探测线性可分性）交叉确认的候选层区间，不是单一 argmin 输出 |
| §1.2：截断保真度 | GGUF 截断路径与 safetensors 截断路径（`test_intermediate_layer_probe.py`）在同一小模型上的 `rel_l2`/`mean_cos` | `rel_l2` 需在预先定义的容差内（依 §2.4 未修复前无法定数，留待里程碑0之后用真实数据定），且必须逐样本配对统计，不是均值一个数 |
| §2.2：小编码器蒸馏 | 编码器输出的 128 维目标 vs 教师投影目标的逐样本余弦相似度 | 均值 + 分布（不能只报均值），且要在留出的验证任务上做下游决策质量对比（有编码器 vs 无编码器，用真实业务指标），无此对比不得声称"突破" |
| §3.1 在线路径 | 端到端请求延迟（小编码器前向 + NanoCore 投影 + 融合） | 命令、退出码、真实压测日志（如 `wrk`/`hey` 或项目自带压测工具的原始输出尾部），不接受"预计"或"理论上" |

### 3.4 旧逻辑/旧符号清点（防腐第4项）

需要物理替换、不保留兼容层的：
- `PhaseTransitionLayerExtractor.detect_phase_transitions` 的裸 `argmin` 判定（`universal_manifold_extractor.py:213,225`）——替换为周期感知去趋势方法。
- `StreamingCovarianceAccumulator` 现有算法——替换为数值稳定版本。
- 文档/代码里"online SVD"这个不准确的命名——按 §2.4 处理。

Rust 侧没有需要删除的旧代码，因为目前没有任何 GGUF/大模型相关的 Rust 代码存在（§0 已核实），这不是遗漏，是当前状态本身如此。

---

## 4. 三类结论分类小结（按任务要求的框架收尾）

**已实现（附证据）**：
- 单槽 llama-server 抽取管线的 fail-closed 校验（槽位数、模型哈希、上下文长度、续跑完整性、原子写入）——`gpu_extract_qwen72b_13tasks.py:81-179`。
- GGUF 头部字节级解析（不读权重）——`scripts/inspect_gguf_layer_bytes.py`，可直接复用于 §1.2 截断构建。
- 安全的截断加载 + 保真度验证方法论（`rel_err`）——`scripts/test_intermediate_layer_probe.py:365-425`，但仅在 safetensors 路径、且"已在微型模型上证明"（未在真实大模型上验证）。
- 双缓冲层流式加载机制的可行性——`scripts/prototype_layer_streaming.py`，但仅在合成权重上验证。
- NanoCore 128 维投影入口 + fail-closed 校验 + 生产 MCP 挂载——`crates/gen-zero-nanocore/src/core_type.rs:19-31`、`crates/gen-zero-service/src/zero.rs:346-373,2146-2500`。这条链路今天就能跑，只是没有产物喂给它。

**未验证（做了但没有证据证明它对）**：
- GGUF 截断路径（路线 A）与 safetensors 截断路径的数值一致性——两套代码从未交叉验证过。
- 双缓冲流式加载在真实模型权重（而非合成权重）上的实测收益。
- Q2_K/Q3_K_M 等激进量化档位对 `h_K` 流形保真度的影响——目前只测过显存是否装得下，没测过量化误差是否已经把流形结构本身搞坏（§1.3 末尾）。

**已核实（可以确定，不是假设）**：
- `load_nanocores_from_paths` 对读取失败、反序列化失败、`validate_nanocore` 校验失败、重复 `domain_id` 四种情形一律 `panic!` 使服务启动失败，不是跳过单个坏文件继续跑（`zero.rs:399-415`）。
- `gen-zero-model` 的 `ActionETFChoiceHead` 与 `engine=nanocore` 互斥，不在本方案挂载路径上（`zero.rs:252`）。
- `GENZERO_NANOCORE_PATHS`（`zero.rs:324`）是真实存在、文档记录（`crates/gen-zero-service/README.md:330`）的操作员配置项，但当前仓库内没有任何脚本设置它——生产代码路径存在，当前无部署配置触发它加载任何真核。

**未完成（没有做，附原因）**：
- 123B/180B/405B 各自的真实相变层 K 值——需要先跑 §1.3 的全层扫描，本报告不预设具体层号。
- 教师投影字典与产物(a) CPU 小编码器——依赖里程碑0（数值缺陷修复）完成之后才能用真实数据训练，目前只有框架（`distiller.py`）没有数据。
- 产物(b) `NanoCoreInstance`（`projection_weights`/`value_weights` 拟合出真实决策语义）——缺决策标签数据，这是比编码器训练更靠后、更难补的一环，因为它依赖具体业务场景定义"正确决策"是什么；没有它，在线路径只是形状对得上的空管道。
- 调用方到 MCP `zero` 工具的客户端代码——`python/gen_zero/client.py` 今天不调用 Rust 服务，这段连线是全新代码，不是复用。
- 端到端在线延迟的具体承诺数字——编码器架构未定，之前无法测量，不接受"理论估算"当验收结果。
- GGUF 截断构建脚本、`NanoCoreInstance` 格式导出器——均为全新代码，尚未编写。
