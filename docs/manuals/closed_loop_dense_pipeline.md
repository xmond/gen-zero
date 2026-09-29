# 端到端操作手册：稠密模型特征 → 流形融合 → NanoCore 生产挂载

任务代号：b0928-t6-docs。日期：2026-09-28。HEAD `edb3d78`。

> **2026-09-29 移除说明**：本文档 §1、§4b 与文末〈命令与证据〉里描述并引用输出的那个三模型（Qwen-72B +
> Llama-70B + Mistral-123B）高斯随机投影主脚本，以及它产出的 13 任务结果文件，已在 b0929u-t2 任务中删除——
> 该脚本依赖的 123B 特征目录 `/ebs/data/extracted_features/mistral123b` 是一个断链的软链接（目标从未存在），
> 结果从未被真实复现过。下文相关段落保留为**历史记录**（当时真实执行过、当时的证据链本身没有错），但其中
> 的命令今天已经无法重跑；请勿依赖它们验证任何当前结论。当前唯一可复现、无 123B/405B 依赖的路径是
> Qwen2.5-72B + LLaMA-3.1-70B 双模型融合，见 [`benchmarks/README.md`](../../benchmarks/README.md#13-task-sota-macro-8152-dual-70b-manifold-reproduction-guide)
> 的复现指南和 [`scripts/download_benchmark_features.py`](../../scripts/download_benchmark_features.py)。

**这份手册只记录今天能真实跑通的部分，并把每一步的真实状态写在它自己头上。** 任务标题设想的链路是
"405B 相变截断 → GCCA 多视角干涉 → 128 维锚点基生成 → NanoCore 生产挂载 → 拒答与置换等变验收"。核实结果是：
这条链路里只有**后两段**（128 维锚点基 → NanoCore 挂载、拒答验收）今天有真实、可重跑、全部通过的测试；
前两段（405B 截断、GCCA 多视角融合接入这条锚点管线）是设计中或与这条管线并不相连的独立组件；置换等变
在本管线实际使用的 `nanocore_ask` 独立入口**没有**测试覆盖，但另一条不同的入口（`decide` + `engine: nanocore`
内联核路径）已有测试覆盖，两者不能互相借用，§4b 分开说明。下表是完整判据，详细证据在各节正文和文末〈命令与证据〉。

状态图例与 [docs/zero/README.md](../zero/README.md#实现状态图例) 一致：**已实现**（代码存在、可运行、有测试或实测覆盖）、
**实验中**（可运行但只验证过合成/小规模数据，或已知有数值缺陷）、**目标愿景**（design-only，代码不存在或从未被真实数据路径调用）。

## 阶段状态矩阵

| 阶段 | 任务标题里的设想 | 今天的真实状态 | 证据 |
| :--- | :--- | :--- | :--- |
| 0. 特征来源 | 405B/180B/123B 模型相变层截断抽取 | **目标愿景**。GGUF 字节级切片工具 `scripts/slice_gguf_layers.py` 已实现且有单测,但没有任何脚本调用它去驱动抽取管线；123B/180B/405B 三个模型至今零真实抽取；今天唯一有真实产物的教师是 LLaMA-70B（**不在**任务标题的模型名单里）,且用的是常规末层抽取,不经过这个截断工具 | [`docs/zero/31-...md` 状态对照矩阵](../zero/31-dense-fleet-manifold-anchor-t4-sys-design.md#状态对照矩阵) |
| 1. GCCA 多视角融合 | 对多个模型的视角做流形干涉 | **已实现，有独立 CLI 入口（`cli.py manifold-fuse`），但未接评测/锚点管线**。`python/gen_zero/manifold/gcca_fusion.py`（`GCCAMidFusion`）是真实、有单测覆盖的正则化 MAX-VAR GCCA，由 `cli.py:610-627` 的 `manifold-fuse` CLI 子命令调用；但**没有任何评测脚本、流形锚点管线或 Rust 生产路径调用它**。此前一个（已于 2026-09-29 删除的）三模型高斯随机投影主脚本产出过一份 13 项分类任务的结果文件，该脚本用 `sklearn.random_projection.GaussianRandomProjection` 做三模型联合基线，**不导入 `gen_zero.manifold` 也不调用 `gcca_fusion`**，不能把那份数字归因为"GCCA 验证"；它连同其 123B 依赖已被移除，见文首移除说明 | §1 本文 |
| 2. 128 维锚点基生成 | GCCA 融合结果 → 128 维锚点 | **已实现（单模型，非 GCCA 输入）**。`ManifoldAnchorDistiller`（`python/gen_zero/causal/manifold_anchor_distiller.py`）对**原始 LLaMA-70B 隐藏特征**做正交投影,产出真实、sha256 校验的 `.npz` 产物,已用一条真实 BoolQ 记录端到端验证 | §2 本文 + `crates/gen-zero-service/tests/test_nanocore_live.rs` |
| 3. NanoCore 生产挂载 | 128 维向量喂给生产决策组件 | **已实现，但驱动决策的风险分类器是 test-stub**。`nanocore_ask`（`crates/gen-zero-service/src/zero.rs:2246`）+ fail-closed 校验（`validate_nanocore`,`zero.rs:446`）+ 真实端到端测试,全部本次校订重跑通过；准确表述是"特征投影与 Rust 引擎集成测试,风险分类器使用 stub"（固定 `p_dangerous=0.01`,见 §3） | §3 本文，命令见文末 |
| 4a. 拒答验收 | 非法输入必须被拒绝,不能静默降级 | **已实现**。NaN、f32 溢出、127/129 维四种非法输入全部被拒绝,断言信息里带原因 | §4 本文 |
| 4b. 置换等变验收 | 候选顺序不影响决策 | **部分验证,两条入口不能互相借用**。`decide` + `engine: nanocore`（内联核,走 `specialized_ask`）路径在 `zero.rs:4471` 已有置换等变性测试（`specialized_scores_are_exactly_invariant_to_candidate_order`）；而生产/本手册 §3 实际用的 `nanocore_state` 走 `nanocore_ask`（`zero.rs:2246`）独立入口尚无覆盖。已删除的三模型随机投影主脚本（见文首移除说明）里存在过一个不同含义的 `shuffled_candidates` 阶段（见 §1 的重要限定),也不能借用来证明这两条入口任何一条的置换等变 | §1、§4 本文 |

---

## 0. 405B 相变截断：目标愿景，今天不要在生产计划里假设它存在

不要跳过这一节直接看 §1-§3——如果读者只记住"128 维锚点基 → NanoCore 能跑",很容易误以为整条链路（含 405B
前端）都已经打通。事实是：

- `scripts/slice_gguf_layers.py` 能把一个 GGUF 文件按层截断，保留 `blk.0..K-1`、丢弃 `output.weight`,单测覆盖原子写入和头部解析正确性。这是构建工具，**不是**已接入的抽取管线：全仓 `grep -rln slice_gguf_layers .` 只命中脚本自身和它的测试,没有任何 `.bat`/`.sh`/编排脚本调用它。
- `docs/zero/31-dense-fleet-manifold-anchor-t4-sys-design.md` 的〈里程碑 0〉（`StreamingCovarianceAccumulator` 数值修复、
  `detect_phase_transitions` 去 argmin、"online SVD" 改名）**仍未完成**——用 `git log -S"StreamingCovarianceAccumulator" -- python/gen_zero/causal/universal_manifold_extractor.py` 核实,自最初引入以来没有任何提交碰过这段逻辑。该文档设计里，123B/180B/405B 三个模型的相变层扫描、量化保真度测量，全部要等这个里程碑之后才能开始，目前都还没有开始。
- 结论：如果你的目标是"接入 405B",先去看 `docs/zero/31-....md` 的原始设计和它的状态对照矩阵，不要假设本手册后面几节的任何产物来自 405B——它们全部来自 LLaMA-70B。

## 1. GCCA 多视角融合与已删除的 13 项任务评测数字：两个不相关的组件，本文上一版把它们错误地划了等号

**历史纠正（该评测脚本已于 2026-09-29 删除，见文首移除说明，以下按已删除时的状态记述）：那个三模型
随机投影主脚本不是 GCCA 的端到端评测脚本，它完全不调用 GCCA。** 它导入的是
`sklearn.random_projection.GaussianRandomProjection`，用**高斯随机投影**
把 Qwen-72B/Llama-70B/Mistral-123B 三个模型各自的隐藏特征投影到同一个 256 维空间做拼接，再叠加候选先验
（`candidate_prior`）、图拉普拉斯正则（`graph_laplacian`）、log-linear 池化（`log_linear_pool`）做闭式求解。
它曾产出过一份 13 项分类任务（massive/multinli/pubmedqa/boolq/paws/squad2/arc_challenge/vitaminc/
civil_comments/aegis_safety/helpsteer2/summeval_relevance/summeval_consistency）的结果文件，那些数字
全部来自这个**高斯随机投影多视角联合基线**，不是 GCCA 验证。这是**候选打分融合**任务：融合的是三个模型对
同一批候选答案的打分/特征，不是把一个模型的隐藏状态在不同层/不同截断点之间做融合。该脚本与其结果文件已
随 123B 依赖一起删除，不可再重跑；下面几段仅作历史记录。

`python/gen_zero/manifold/gcca_fusion.py` 里的 `GCCAMidFusion` 才是真实实现的正则化 MAX-VAR GCCA，有单测
覆盖（`gen_zero/tests/test_gcca_fusion.py` 等 60 项相关单测），并由 `python/gen_zero/cli.py:610-627` 的
`manifold-fuse` CLI 子命令调用；但**没有任何评测脚本、流形锚点管线或 Rust 生产路径调用它**——这一结论在
上述脚本删除后依然成立，因为它本来就与该脚本无关。其导出位于 `python/gen_zero/__init__.py` 与
`python/gen_zero/manifold/__init__.py`。

重跑 `gcca_fusion` 自己的单测（离线，与已删除的 13 项评测数字无关，今天仍可运行）：

```bash
cd python
python3 -m pytest gen_zero/tests/test_gcca_fusion.py gen_zero/tests/test_candidate_prior.py \
  gen_zero/tests/test_manifold_master_objective.py -q
```

结果见文末〈命令与证据〉，60 项全部通过——**这 60 项测的是 `gen_zero.manifold` 包
（`GCCAMidFusion`/`CandidateSemanticPrior`/`MasterClosedFormSolver`），不是已删除脚本自己内联的
`fit`/`pool`/`graph_gram` 函数，两者代码完全独立**，不要把这 60 项通过当成对该已删除脚本的单测覆盖。

已删除脚本原本可以用 `--smoke` 跑第一个任务做快速管线健康检查，完整 13 项跑法去掉 `--smoke`；这条命令
今天已不存在，不要再尝试运行它。

**"shuffled_candidates" 不是"置换等变证明"，读之前先看清它当时测的是什么（历史记录）。** 该已删除脚本
对每个视角的候选顺序做一次固定种子的打乱，然后用 `(True, True, True)` 配置（candidate_prior + graph_laplacian +
log_linear_pool）重新打分。这个配置**不是** `master_solver` 列使用的
求解器；两列的数字不能直接相减当成"置换前后的差异"——已发表报告里 `master_solver` 和 `shuffled_candidates`
两列本来就来自不同算法，例如 paws 一列是 93.20，另一列是 91.60，这个差不是置换造成的退化，是两个不同求解器
的正常差异。这一阶段真正能说的是：`log_linear_pool` 配置在候选顺序打乱后的表现（`shuffled_candidates` 列自
己），不能推广到 `master_solver`，更不能推广到下面第 3 节的 `nanocore_ask`——那是完全不同的代码路径。

`ManifoldAnchorDistiller` 的文档字符串把自己描述成"GCCA 特征的离线保角压缩"（`manifold_anchor_distiller.py:1`），
但这只是它设计时设想的输入类型之一。用 `grep -rn "ManifoldAnchorDistiller(" .` 核实：`.fit()` 在全仓库只被
两处调用——它自己的 CLI（下面 §2 会用到）和 `profile_nanocore_latency.py:132` 的**合成随机数据自检**（该函数
自己的注释写明"Throwaway artifact fit from random data; checks the harness, not real latency"）。**没有任何脚本
把 GCCA 的输出接到 `ManifoldAnchorDistiller.fit()` 的输入上。** 下一节的真实产物完全绕开了 GCCA。

## 2. 128 维锚点基生成：真实产物，但输入是原始 LLaMA-70B 特征,不是 GCCA 融合结果

真实、已交付、已用哈希校验的产物是 `benchmarks/results/manifold/distilled_128d_llama70b_boolq.npz`
（`git ls-files` 确认已跟踪）。它是对 `/ebs/data/extracted_features/llama70b/boolq.npz`（一个机器本地路径，
不在仓库内,由 `test_nanocore_live.rs` 在测试时重新校验其存在与内容哈希）里的原始 LLaMA-70B 隐藏特征做的
正交投影,**不经过 GCCA**。

从零重新拟合一份锚点基（用你自己的特征文件替换 `--features`）：

```bash
cd python
python3 -m gen_zero.causal.manifold_anchor_distiller \
  --features /path/to/your_features.npz --block train_full \
  --output-dim 128 --out /tmp/my_anchor.npz
```

`--features` 指向的 `.npz` 必须含有 `--block`（默认 `train_full`）指定键下的一个二维、全有限值的矩阵；
`ManifoldAnchorDistiller.fit` 用薄 SVD 求正交投影 `P`（`P @ P.T == I`），只做一次性拟合，不做在线更新——
上游特征分布变化后必须重新拟合，本模块不会自己检测漂移（`manifold_anchor_distiller.py:9-16`）。

重跑单测（离线拟合 + 投影桥接的正确性）：

```bash
cd python
python3 -m pytest gen_zero/causal/tests/test_nanocore_bridge.py \
  gen_zero/causal/tests/test_manifold_anchor_distiller.py -q
```

结果见文末，全部通过。

## 3. NanoCore 生产挂载：真实端到端闭环,今天就能重跑

这是本手册唯一一段"从原始特征到生产决策，全链路真实运行过、有测试断言"的闭环：

```
真实 LLaMA-70B BoolQ 隐藏特征（磁盘 .npz）
  → ManifoldAnchorDistiller.project()（128 维正交投影，sha256 校验）
  → NanocoreAnchorBridge.generate_mcp_ask_payload()（组装 MCP ask 请求体）
  → nanocore_ask（Rust，zero.rs:2246）
  → validate_nanocore 校验 + NanoCoreFleetScheduler + MoVFusionEngine
  → 决策 + confidence + chosen_action
```

`crates/gen-zero-service/tests/test_nanocore_live.rs` 用两种方式证明这条链路是真的：一是加载一份**已经生成
好的**固定 fixture（`tests/fixtures/nanocore_anchor_state_boolq_row0.json`）直接喂给 `nanocore_ask`；二是（更
关键的一条）用 `std::process::Command` 实际 `spawn` 一个 `python3` 子进程，现场重新跑一遍
`NanocoreAnchorBridge.project_to_nanocore_state()`，断言现场算出的 128 维向量和固定 fixture 逐字节一致——
这样固定 fixture 就不会在代码改动后悄悄失真而没人发现。

**准确表述：这是"特征投影与 Rust 引擎集成测试，风险分类器使用 stub"，不是风险判别能力的验证。**
`real_projected_state_decides_between_two_candidates`/`real_projected_128d_state_drives_a_real_nanocore_decision`
这两条成功路径要经过共享风险门才能拿到 `is_error=false`；测试用一个本地 HTTP stub（`stub_scorer`，
`test_nanocore_live.rs:71-88`）顶替真实的语义风险打分后端，固定返回 `p_dangerous: 0.01`、
`classifier.name: "test-stub"`（`test_nanocore_live.rs:76,78`），只是为了让请求稳定落在 `Tier0Proceed`、
不被 gate 升级拦下。没有配置真实风险后端时，gate 按 `crates/gen-zero-gate/src/risk.rs` 的 fail-closed 规则
本会把结果升级为 `Tier2Escalate`——这条测试验证的是"128 维向量能不能真的驱动 `nanocore_ask` 产出决策"这条
接线是否打通，不是任何真实风险判别模型的准确度。

**生产环境挂载方式**（复用 [README.md「Integrated Rust subsystems」](../../README.md#integrated-rust-subsystems)
已经记录的机制，这里只补上锚点基产物如何落地成 `NanoCoreInstance` 文件）：把拟合好的锚点基和一组
`(target_128d, decision_label)` 监督对训练出的 `projection_weights`/`value_weights` 导出成
`NanoCoreInstance` 期望的 JSON（`domain_id/name/prototype/projection_weights/value_weights/out_dim/
base_confidence`），放到 `GENZERO_NANOCORE_PATHS` 指向的路径。**这一步——用真实决策标签训练
`value_weights`——今天没有任何脚本做**；`test_nanocore_live.rs` 里的 `value_weights` 直接复用了
`prototype`（`prototype.clone()`，见 fixture 加载代码），是测试夹具的简化，不是一份真实拟合过的决策头。
把这份手册的闭环部署到真实业务场景之前，`value_weights` 必须先用该场景的历史决策数据重新拟合，
否则线上决策在语义上是空的，只是形状对得上（`docs/zero/31-...md` §2.2 步骤 4 已经指出这一点，本手册
的实测重新确认它依然成立）。

**重要边界：Python `decide_nanocore` 与 Rust `nanocore_ask` 是两条独立的决策实现路径**：
- `cli.py anchor --execute` 调用的是 `GenZero.decide_nanocore()`（`python/gen_zero/client.py:3230`），这是一条**纯 Python 进程内**的决策路径（使用 Python 侧的 `ActionETFChoiceHead` 和注册的 core），**完全不调用 Rust 服务**。
- 真正调用 Rust 决策引擎的是 `nanocore_ask`（`crates/gen-zero-service/src/zero.rs:2246`），它接收 JSON-RPC 请求，通过 `validate_nanocore` 门禁后在 Rust 内部完成评分。今天唯一把 Python 特征投影和 Rust `nanocore_ask` 连通的是 Rust 集成测试 `test_nanocore_live.rs`（通过子进程计算特征后作为请求 payload 传入），目前尚无跨进程的生产包装器调用它。

同时要确认：**`GENZERO_NANOCORE_PATHS` 今天没有任何部署脚本设置它**（`grep -rn GENZERO_NANOCORE_PATHS
--include=*.sh --include=*.bat --include=Makefile -r .` 零命中）。挂载这条链路到一个真实运行的
`gen-zero serve` 进程，需要你自己写部署配置去设置这个环境变量，这不是"复用现成脚本"，是本手册明确
要求新增的运维步骤。

## 4. 验收：拒答与置换等变

### 4a. 拒答（已实现，测试覆盖四种非法输入）

`non_finite_or_wrong_length_state_is_refused_fail_closed`（`test_nanocore_live.rs:399`）对同一个真实 fixture
分别注入 NaN、f32 窄化溢出（`1e39` 在 f64 有限但转 f32 变 `+inf`）、127 维、129 维四种非法状态，断言全部
被 `is_error=true` 拒绝，且错误文本包含 `"nanocore_state must contain 128 finite numbers"`——不是静默截断
或补零。`unregistered_domain_is_refused_fail_closed` 另外覆盖了"域未注册"这一种拒答路径。

重跑：

```bash
cargo test -p gen-zero-service \
  --test test_nanocore_live --test provenance_nanocore_integration --no-fail-fast -- --nocapture
```

结果见文末，8 项全部通过（5 + 3）。

### 4b. 置换等变（部分验证——两条入口分开看，不要把一条的覆盖套到另一条头上）

`PolymorphicZeroEngine` 里有两条不同的代码路径都能触发 NanoCore 打分逻辑，覆盖状态不一样：

- **`decide` + `engine: nanocore`（内联核，走 `specialized_ask`，`zero.rs:2510`）：已验证。**
  `specialized_scores_are_exactly_invariant_to_candidate_order`（`zero.rs:4471`）对 `nanocore`/`generic`
  两种 backend、`etf`/`linear` 两种 head、2/3 候选两种规模都做了"候选顺序反转，断言每个候选自己的打分
  不变"的断言。这条路径请求体里带 `nanocore_core`（内联 `NanoCoreInstance`）和 `decision_state`，不经过
  `nano_fleet` 注册表。
- **`nanocore_state` 走的 `nanocore_ask` 独立入口（`zero.rs:2246`，依赖 `nano_fleet.get_core` 注册核，本
  手册 §3 的真实端到端闭环走的正是这条）：未验证。** `grep -rn "permut\|reorder\|shuffle" crates/
  gen-zero-service/tests/*.rs crates/gen-zero-service/src/zero.rs` 在这条入口上零命中——没有任何测试对
  `nanocore_ask` 做候选顺序打乱断言。`MoVFusionEngine`（`crates/gen-zero-nanocore/src/mov.rs`）对每个候选
  是否独立打分、打分是否与候选在列表中的位置无关，本次校订**没有**做静态分析或补写测试去证实。这条独立
  入口今天没有置换等变测试覆盖，不能借用 `specialized_ask` 那条测试的结论——两者是不同函数、不同参数形状
  （`nanocore_core`+`decision_state` vs `nanocore_domain(s)`+`nanocore_state`）。补一条测试（构造同一组
  候选的两种排列，断言 `nanocore_ask` 返回的每个候选各自分数不变）不属于本次文档任务范围，留给后续实现
  工作。

曾经真实存在、但同样**不能**替代上面这条缺失测试的是 §1 提到的已删除脚本里的
`shuffled_candidates` 阶段——它测的是候选顺序打乱后 `log_linear_pool` 配置在 13 项分类任务上的准确率
变化，衡量对象、代码路径、任务性质三者都和上面两条 NanoCore 入口不同，不构成任何一条的"置换等变"证据。
把两者混为一谈就是本手册开头警告过的"偷换概念"，这里明确切割开；该脚本已删除，这一段仅作历史说明。

---

## 命令与证据

以下命令在本次校订（HEAD `edb3d78`，工作树 `/tmp/fleet-wt/b0928-t6-docs`）中实际执行，退出码与输出尾部照录。

**Rust：NanoCore 端到端闭环 + 拒答测试**

```
$ cargo test -p gen-zero-service --test test_nanocore_live --test provenance_nanocore_integration --no-fail-fast -- --nocapture
...
     Running tests/provenance_nanocore_integration.rs
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
     Running tests/test_nanocore_live.rs
test real_projected_state_decides_between_two_candidates ... ok
test unregistered_domain_is_refused_fail_closed ... ok
test real_projected_128d_state_drives_a_real_nanocore_decision ... ok
test non_finite_or_wrong_length_state_is_refused_fail_closed ... ok
test fixture_state_matches_a_fresh_run_of_the_real_python_bridge ... ok
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.28s
EXIT:0
```

**Python：GCCA 融合 + master-objective 单测（60 项）**

```
$ cd python && python3 -m pytest gen_zero/tests/test_gcca_fusion.py gen_zero/tests/test_candidate_prior.py gen_zero/tests/test_manifold_master_objective.py -q
............................................................             [100%]
60 passed in 6.13s
EXIT:0
```

**Python：锚点蒸馏器 + bridge 单测**

```
$ cd python && python3 -m pytest gen_zero/causal/tests/test_nanocore_bridge.py gen_zero/causal/tests/test_manifold_anchor_distiller.py -q
79 passed, 9 warnings in 68.25s (0:01:08)
EXIT:0
```

九条警告全部是 `test_*_rejects_*overflow*`/`test_*_rejects_*energy_overflow*` 这几个故意构造病态输入
（极端值、精心构造的投影矩阵）来验证拒绝路径的测试触发的 `RuntimeWarning`（`invalid value encountered in
scalar divide`/`overflow encountered in matmul`/`overflow encountered in subtract`），是测试断言"必须拒绝
溢出输入"时途中产生的 NumPy 警告，不是失败，也不是生产路径会遇到的静默降级——这些测试本身就是在检查
`ManifoldAnchorDistiller` 在这些输入下确实抛出异常而不是返回一个悄悄错误的数字。

**Python：高斯随机投影多视角联合基线 13 项任务评测，非 GCCA（历史记录，脚本与结果文件已于 2026-09-29 删除，命令不可再重跑）**

```
$ python3 <已删除脚本> --smoke --output /tmp/smoke_manifold
Task                        baseline candidate_pr graph_laplac log_linear_p master_solve shuffled_can
massive_en                     88.86        89.14        89.14        89.43        89.14        89.43  10.97s
MACRO                          88.86        89.14        89.14        89.43        89.14        89.43
WROTE /tmp/smoke_manifold.json /tmp/smoke_manifold.md
EXIT:0
```

当时现场重跑的 `massive_en` 单任务数字与仓库里彼时提交的 13 项结果文件一致（该结果文件的完整 13 项 macro
结果已随脚本一起删除；本条记录只是历史证据，不代表今天可重跑）。当前可复现、无 123B 依赖的复现路径见
[`benchmarks/README.md`](../../benchmarks/README.md) 的 13-task SOTA Macro 81.52% 双模型复现指南。
