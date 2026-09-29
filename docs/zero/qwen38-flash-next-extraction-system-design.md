# Qwen3.8-Flash-Next 物理提取与系统工程实现案

日期：2026-09-27。性质：设计交付，不是已实现、已加载或已上线声明。
代码基线：`fa6cddb656f49d9ae01bf417e476514962e03a34`。未改动现有生产代码。

## 1. 决策与事实边界

建议采用两条明确隔离的路径：**Transformers/PyTorch 的模型内提取路径作为表征基准；vLLM 或 SGLang 作为后续吞吐优化路径，须增加 worker 内提取适配器并通过逐样本一致性验收**。HTTP 文本生成成功不能证明中间表征提取成功。现有 llama-server 最终 pooled embedding 不能替代指定中间层。

官方模型卡确认：125B 主模型、每 token 激活 6B，另有 51B N-gram embedding 和 4B MTP；隐藏宽度 2560、48 层、4 个 gated-residual 分支、512 专家、10 routed + 1 shared；N-gram 在 layer 2，表规模约 20,000,000；注意力为 GDN/QSA 混合。不能用旧 Qwen3-Next、Qwen3.5 或普通 decoder 的模块名/张量布局猜测加载器。

来源：[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)。网页抓取存于 `.firecrawl/qwen38-official.md`、`qwen38-github.md`；文件摘要在本报告 evidence 的 `commands.json`。模型配置与权重索引尚未下载核验，实际字节数、模块路径、dtype 和 loader 版本仍须以固定 revision 的文件为准。

最重要的禁止事项：

- **6B active 不是 6B resident**。路由未选中的专家也必须有完整 backing storage；预填充 batch 的专家并集可能覆盖大量专家。
- **90GB RAM 放不下 51B BF16 表**。不能以 mmap 成功或 swap 尚未报错宣布 host offload 可用。
- **2560 hidden size 不代表每个 hook 都返回 `[B,T,2560]`**。4 分支 residual 必须区分 gated read、完整 branch state、final norm；不得静默 flatten/mean/select branch。
- “可运行生成”与“可提取指定层”是两个验收项；网页支持声明也不是本机实测。
- 几何表征、PCA、CKA、稳定矩阵都不能自动被称为因果能力、相变或推理突破。

## 2. 现有代码最致命的问题

| 位置（仓库相对 path:line） | 已核查的行为 | 新方案要求 |
|---|---|---|
| `benchmarks/suites/run_universal_extraction_a100.py:27` | Windows 路径、固定旧模型 | 参数化路径、固定模型 revision/hash，不能静默替代模型 |
| 同文件 `:34`, `:44`, `:60` | 任意 JSONL 扫描，猜字段，异常直接 continue | 复用 13-task schema，文件/行号报错，缺任务失败 |
| 同文件 `:97`, `:115` | 固定 4096 维；开启全部 hidden states | 从已验证适配器得到维度；只捕获目标位置 |
| 同文件 `:99`, `:124`, `:147` | 保留样本列表和差分列表 | 固定队列/分块落盘；不得以 accumulator 的 O(d²) 代表全管线 |
| 同文件 `:131` | 通过 sentence 字段/正则生成所谓 counterfactual | 格式解析不是反事实干预；无预先定义配对实验时禁用此能力声明 |
| 同文件 `:158`, `:170`, `:175`, `:202` | 全量 eigh 被叫 online SVD；CKA 即宣布相变；不相关样本次序被当轨迹 | 改正术语；不生成未定义的因果/动力学产物 |
| `python/gen_zero/causal/universal_manifold_extractor.py:83`, `:102` | 用原始二阶矩相减计算协方差 | 改成中心矩合并，并测试大偏置、小方差 |
| 同文件 `:115`, `:121` | eigh 产生额外工作区；memory_bytes 只统计持久状态 | 内存预算覆盖矩阵副本、临时项、BLAS 工作区、队列 |
| 同文件 `:481` | sha256 只覆盖拼接数组字节，不覆盖名称/shape/dtype/metadata；直接写目标 | 完整文件摘要、严格 schema、原子发布及 manifest 链 |
| `benchmarks/suites/gpu_extract_qwen72b_13tasks.py:100` | 以 basename 核查模型；缺路径时返回未验证 | 新路径要求 revision 与分片哈希，路径未知即拒绝 |
| 同文件 `:123` | token budget 可采用带来源标记的 fallback | 严格运行 manifest 要求完整预算；不可仅沿用默认数值 |
| `benchmarks/suites/cpu_extract_gte7b_13tasks.py:83` | 模型专属末尾 token 处理 | 复用数据契约，不能复制 GTE token 特例到新模型 |

数值问题已经用真实类复现，但输入是数学诊断样本，不是模型激活：`1e8 + arange(32).reshape(16,2)` 分四批，现实现得到 `[[86,86],[86,84]]`，先中心化的参考得到 `[[85,85],[85,85]]`，最大绝对误差 1.0。命令原始退出码为 0，代表诊断执行成功，不代表数值正确。见 `evidence/qwen38-extraction-design/numerics.log` 和 `commands.json`。

调用搜索发现 accumulator 被 `gepa_daemon.py:39,66,82` 调用，不能宣称整个模块零引用；但本次搜索没有发现 Rust 调用 `load_codebook` 的证据。搜索范围/命令见 evidence；它不是对所有动态加载路径的证明。必须新增真实生产入口验收，不能把 benchmark 单独跑通当上线。

## 3. 容量账本与硬件部署

以下为十进制 GB 裸参数下界，不含量化 scales、padding、视觉模块、加载峰值、缓存和工作区；GB 与 GiB 不混用。

| 精度假设 | 125B 主模型 | 51B 表 | 4B MTP | 合计下界 |
|---|---:|---:|---:|---:|
| BF16，2 bytes/param | 250 | 102 | 8 | 360 |
| 全部 8-bit，1 byte/param | 125 | 51 | 4 | 180 |
| 全部 4-bit，0.5 byte/param | 62.5 | 25.5 | 2 | 90 |

“全部 4-bit”只是算术下界，绝不表示存在对应内核或该 checkpoint。实际混合量化可能远大于此值。抓取时 vLLM recipe 面板列出 BF16 423GB、FP8 250GB、NVFP4 130GB；SGLang 文档的某 NVFP4 变体为约 126GiB，其中 FP8 表约 47.7GiB。这些不是统一 checkpoint 的可互换精确大小，进一步证明必须读取目标仓库文件清单，不能只用参数乘字节预留磁盘。

来源：[vLLM 官方 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)、[SGLang 官方 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)。其中 vLLM 概述对“125B 是否含表”的文字与官方卡有歧义，本报告使用官方模型卡的加法口径。

| 节点方案 | 设计判断与限制 |
|---|---|
| dev：64 CPU / 90GB RAM / 316GB 空闲盘 | 适合调度、manifest、协方差和产物校验。不接纳整套 BF16 下载/转换；不接纳 BF16 全表 host 常驻。CPU-only 的 4-bit 90GB 裸下界已无运行余量，不能作为保底方案。 |
| 1×A100/H100 80GB + 90GB host | BF16 主干无解。特定已验证混合量化 + FP8/INT8 表 host 驻留可能有容量机会，但须实测内核支持及峰值；不作为首次完整提取承诺。 |
| 2×80GB + 90GB host | 对某些量化主干 + host 表可能够；不能按激活参数保证。模型并行布局、工作区、GPU 间带宽需单独验收。 |
| 4×80GB + ≥192GB host | BF16 主干约 250GB 放 GPU、102GB 表放 host 的优先候选；192GB 是否足够需含加载峰值核算，256GB 更宽裕。建议另配 ≥1TB 空闲高速存储保存固定 checkpoint 与转换临时数据。 |
| 4×80GB + 90GB host | 如固定变体与内核支持，FP8 表或表按 CPU/GPU 分片可探索；不采用 BF16 全表 host。H100 的 FP8 路径和 A100 的兼容内核要分开验证。 |
| 8×80GB | BF16 全驻 GPU 是正确性基准候选，仍要检查逐卡可用空间、分片约束和加载磁盘，不等于已跑通。 |

A100 不应被当作具备 H100 的原生 FP8 路径；A100/H100 均不能照搬 B200/Blackwell 的 NVFP4 recipe。要用该 GPU、该 checkpoint、该 kernel 的组合矩阵验收，不能自动转 dtype、换量化格式、退回 CPU。

加载前输出逐设备 memory plan：tensor 所有者、storage dtype、compute dtype、bytes、CPU/GPU 位置、复制数、预填充工作区、GDN state、QSA/indexer/cache、hook staging、临时加载空间。以 **可用** RAM/VRAM 而非总量准入。总体预算为 `weights + states + activations + workspace + staging + safety_margin`；从 batch=1 和任务最长输入验证，再确定 token budget。

本次未 SSH 探测 dev/stg/ai-wsl，也未发现用户提供的实际 GPU 数量/互联信息；以上是条件化选型。执行期先采集 `uptime`、`/proc/loadavg`、`free -b`、`df -B1`、`nproc`、`nvidia-smi`、`nvidia-smi topo -m`。1m/15m load 按核数归一化，准入阈值写入 manifest；≥8GB RAM 和 ≥10GB 磁盘只是用户要求的基础门槛，不能替代模型预算。任一节点不满足即记录拒绝原因，不启动任务。

## 4. 引擎选择与 offloading 实现

| 引擎 | 用途 | 必须满足的条件 |
|---|---|---|
| Transformers + PyTorch | 首个可审计提取基准，直接 hook | 固定支持此架构的 commit，验证 AutoClass、4 分支语义、官方算子、精度和 checkpoint。不能沿用旧脚本的 AutoModelForCausalLM 假定。 |
| vLLM | 后续吞吐型 worker | 官方已有模型 recipe，抓取页面标注 0.29.0+；固定版本和容器 digest。提取逻辑必须进入实际 model worker，处理 packed tokens、TP 和 CUDA graph。生成 API 不提供所需证据。 |
| SGLang | 优先评估现成 PLE offload 工程实现 | cookbook 有 PLE offload 和分支特定 file backend；固定真实支持版本。仍需 worker 内中间表征适配，不能靠 API 猜张量。 |
| llama.cpp / GGUF | 经验证后的 CPU/GPU 混合部署或最终 embedding 对照 | 本次未核实具体支持 commit/转换器；Qwen2.5 的现有脚本不证明新架构支持。指定层需要 C++ graph 提取实现，不支持 PyTorch hook。 |
| Ollama | 用户交互封装候选 | 不作为本项目指定层物理提取首选；现有 API 输出不能代替中间层。具体后端和硬件支持未在本次实测。 |

### 4.1 N-gram 表

优先顺序是全 GPU 基准 → 官方/经验证的 host row-gather → 显式 file-backed 实验路径，顺序不是允许自动 fallback。

1. 从固定模型实现获得真实 n-gram ID 算法、边界 token 处理和索引范围；保留 bigram/trigram 历史。禁止自己按文本模式猜 ID，禁止改变 hash、collision 或 padding 语义。
2. 全 prompt token 已知时，可提前生成查表请求；按实际 ID 去重 gather，再按 inverse index 恢复原顺序。prefetch 只依赖已知 token；若实现索引还依赖 hidden state，则不得提前虚构计算。
3. host 保留真实表或经批准的量化存储；CPU gather 后只传命中行。不要把 102GB 表全部 pin；使用有上限的 pinned 双缓冲，量化按行反量化，记录 scale/zero-point 和误差。
4. 拷贝 stream 记录 CUDA event；layer 2 使用前显式 wait，不能读尚未完成的数据。缓存 miss 应正常读取真实 backing table，不是算法降级；backend 不可用、I/O 错误、校验失败必须报错，绝不能以零向量或普通 embedding 代替。
5. TP 下按真实表布局选择共享只读 host 存储或分片；禁止每 rank 悄悄复制整表到 RAM。共享页也要测进程 PSS 和节点 MemAvailable，不能简单求 RSS。
6. mmap/file backend 必须独立配置并记录 page fault、page-cache/RSS、I/O 延迟与队列压力；不把 mmap 当作没有内存成本。普通 PCIe 节点不得套用统一内存设备的性能结论。

一般 Accelerate CPU/disk offload 会在执行时移动模块权重，不能假定它天然实现 N-gram **按行** gather。必须检查并禁用会将整个 embedding 模块搬入 GPU 的通用 hook，为表实现经过验证的专用适配器。[Accelerate 文档](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)

### 4.2 MoE 专家

第一阶段让全体专家在多 GPU 上常驻，按层 device-map 或经过验证的 TP/EP 分布；dispatch 由真实 router 选择专家。容量不足时才能选择独立实验配置的 expert paging/CPU 计算。

专家选择依赖当前层 hidden state，不能像已知 token 的 N-gram 表一样把所有未来访问准确预取。若实现 paging：完整 backing weights + 按 layer/expert 键管理的缓存 + CUDA event + 有界请求队列；缺专家必须等待或失败，禁止 top-k 裁剪、热门专家替代、零输出。统计 per-layer 命中率、搬运字节和等待时间；冷启动与热缓存分别测量。prefill 往往扩大专家并集，不能拿 decode 的“6B active”推算提取吞吐。

## 5. 四层端到端管线

### Layer 1：数据、prompt 调度与批处理

复用 `grand_challenge_data.py:49` 的 13 tasks：massive_en、massive_de、multinli、pubmedqa、vitaminc、boolq、squad2、paws、civil_comments、aegis_safety、helpsteer2、summeval_relevance、summeval_consistency。

复用 `gd.load_test` / `gd.build_train` 的 leakage gate 与 `gpu_extract_qwen72b_13tasks.py:184` 的对齐契约：train/test IDs、顺序、标签、candidate 顺序逐项验证。原始 prompt 为 `context + '\n\n' + instruction`；若为新模型使用 chat template，必须建立不同 representation ID，不能伪装与 raw prompt 同协议。

manifest 固定 dataset 文件 SHA256、ID 顺序、seed、split、候选列表、tokenizer revision、special tokens、template、每任务 max/head token budget、截断策略。不同 tokenizer 下相同 token budget 不保证保留相同文本；记录 raw/kept token 长度及实际 token-ID hash，比较结果须披露此差别。

用磁盘 manifest 固定 row_index，长度分桶只改变执行顺序；输出按 row_index 恢复。队列按**字节与 token 数**设上限，不能只限制条目数。初始 batch=1，小规模逐级提升，允许的 batch 缩小只在显式配置的重试协议内执行并记录 attempt；不得自动截短输入、换模型、跳样本。

PCA/basis 只用 train split。test 与 candidates 只做变换，不参与均值/协方差、层选择、量化参数选择。多任务统计须声明按样本加权或固定 task 权重；默认按样本加权，不能偷偷均衡类别。标签不进入 model prompt 或特征提取决策。

已有数据构建器可能物化整个 task；大任务时先一次性构建、校验并落盘 manifest，再由 worker 流读。不能宣称原构建器本身已实现 O(1) 流式内存。

### Layer 2：目标 hook 与张量寿命

`Qwen38Adapter` 的职责：匹配 architecture/revision；定位文本 backbone、最终 norm、目标层的 gated read 或完整 residual；声明张量 layout 和维度；关闭 MTP、视觉输入、KV cache（在实现允许的范围内）及全量 hidden states；避开不必要的 `[B,T,vocab]` logits 计算。不能为了省内存删除参与文本 forward 的必要模块。

建议第一版输出 final post-norm 的 2560-D 表征；中间层明确指定 index 与读出位置，未确认 layout 前不写死 L24 路径。若保留四分支，则显式声明 `branches=4`、维度/轴顺序，必要时展平为 10240-D 的**新协议**；不得把它当 2560-D，也不得平均后假称原始表征。

PyTorch eager + `eval()` + `inference_mode()` 起步，禁用 graph capture/compile；用真实短输入把 hook 输出与模型官方对应输出逐样本核对后再开启优化。forward hook 只能避免额外保留全层张量，不能消除当前层 forward 自身的激活峰值。

- 在 GPU 上先选目标 token，再 detach/copy 到有界 CPU buffer；不得先 `.cpu()` 整个 `[B,T,d]`。
- 最后有效 token 位置采用 `max(where(attention_mask != 0, positions, -1))`，对左右 padding 均正确；空序列失败。`sum(mask)-1` 只适用于右 padding，`-1` 只在已验证布局下成立。
- 捕获 batch nonce、sample IDs、layer name、命中次数、tensor shape；每个目标应命中预期次数。漏 hook、多 hook、维度漂移、非有限值均终止。
- callback 返回 None，不修改 forward 输出；不保留原 output、计算图或跨 batch closure 引用。句柄在 finally 中移除；处理完即释放当前 batch。
- 异步 D2H 要等待 event 后才读 CPU tensor，缓冲用完才能复用；首版同步 copy 更易审计。
- TP 输出若按 hidden 维分片，在正确维度 gather；若复制则只写唯一 owner；若 PP 则由该层 owner 发送带 ID 的特征。不得把 rank 数误计成样本数。
- vLLM/SGLang 的 packed/chunked prefill 必须以 request ID 和 position 映射最后 token，状态跨 chunk 保持一致；prefix cache 可能跳过 hook，提取基准阶段应明确关闭，否则缓存必须同时保存可验证的目标表征。

### Layer 3：稳定的 StreamingCovarianceAccumulator

保留现有 public class/API，原地替换二阶矩算法，避免出现新 accumulator 无调用。持久状态使用 FP64 的 `(n, mean, M2)`；对 batch 的 `(m, mean_b, M2_b)`：

```text
delta = mean_b - mean
n_new = n + m
mean_new = mean + delta * m/n_new
M2_new = M2 + M2_b + outer(delta, delta) * n*m/n_new
covariance = M2/n             # population, ddof=0
```

空 batch、维度不符、非有限输入、累积溢出报错；先形成且检查候选状态，成功后提交，失败不得留半更新状态。第一批独立初始化，n=0/1 无 covariance；求 rank-k 必须满足 `k <= min(d,n-1)`，并检查数值秩，不能只有 `n>=k`。禁止 NaN 转零或失败后 identity basis。

状态约 `8d²+8d` bytes，d=2560 约 50MiB，d=10240 约 800MiB；每个目标层、每个 task 同时累积都会倍增。默认 task 顺序处理，仅保持必要层数；全局统计可按固定次序 merge task 统计。

整个峰值应预算多份 d² 临时矩阵、`B*d` 的 FP64 batch、eigh 工作区、读取/写出缓存；`memory_bytes()` 不能作为进程峰值证明。用 RSS/PSS 和 GPU peak allocated/reserved 实测随 N 增长是否平台化。固定 batch、层数、队列后为 O(d²+Bd)，不是对任意模型/序列长度的绝对常量。

求解前对称化并记录修正量；检查 PSD 误差尺度，显著负特征值失败。只允许对阈值内舍入级负值显式裁零并记录数量/幅度。完整 eigh 为 O(d³)，不是 online SVD；若选迭代 top-k，需固定容差、检查残差和收敛，不收敛即失败。比较子空间投影矩阵，不能要求退化特征空间的基向量逐元素相同。

basis 在训练结束后才能确定：第一遍将固定大小 raw-feature shards 落盘，同时更新统计；第二遍流读投影为 Z。若不要 raw features，则需重新 forward，代价明确记录。禁止保留全部 X 等待 basis。跨模型低维几何相似度只是观测量，突破需固定 test IDs 的逐样本配对统计。

### Layer 4：产物与校验链

建议目录：`run/<run_id>/{manifest.json,events.jsonl,features/,statistics/,basis/,checkpoints/}`。大数据采用分块 NPZ；NPZ 不是可靠的直接 mmap 容器，不允许用 `np.load(...,mmap_mode=...)` 宣称其 zip member 零拷贝。

| 文件 | 必需字段 |
|---|---|
| raw feature shard | `X` FP32 `[n,d]`、`row_index` int64、`sample_ids` 无 object dtype、`token_count`、`info_json`；每 shard 固定最大字节 |
| statistics checkpoint | `n` int64、`mean` FP64 `[d]`、`M2` FP64 `[d,d]`、已提交 shard 前缀与输入 cursor、配置摘要 |
| basis.npz | `U_k`、`mean`、`eigenvalues`、`n_samples`、`info_json`；保留 FP64 审计版，部署 FP32 另产物/另 hash |
| benchmark 兼容导出 | `train_full,test_full,cands,train_label,train_ids,test_ids,info_json`，与当前 consumer 契约一致；representation ID 明确维度和 pooling |

大规模 consumer 应直接迭代 shards。若必须生成单文件 task NPZ，使用有界写出/磁盘中间数组，并确认读取端不会把所有 task 同时加载；超过事先预算就拒绝兼容导出，不能破坏内存约束。

metadata 至少包含：schema/version、run_id、代码 HEAD 与工作树差异摘要、model/tokenizer revision、所有权重分片 sha256、engine/kernel/container 版本、dtype/quant 配置、设备拓扑与 offload 策略、hook 完整模块路径/语义、任务及数据摘要、split、token 策略、训练统计范围、shape/dtype、样本数量、资源峰值、失败/重试事件摘要。没有完成的验证写 `not_verified`，不得填 true。

按同目录临时文件写入 → flush/fsync → `allow_pickle=False` 重读验 shape/dtype/有限值/IDs → 对最终文件字节流式 SHA256 → 原子 rename。最后写 manifest，再发布 COMMITTED 标记；读者必须以已提交 manifest 为根。文件哈希放外部 manifest，避免自引用；manifest 自身 hash 放独立 commit marker，外部报告固定该 hash。数组级语义 hash 要包含字段名、shape、dtype、规范字节序及数据，不能只拼裸 bytes。SHA256 证明完整性，不证明真实性或模型正确。

checkpoint 的统计和 shard cursor 必须属于同一个提交代次；恢复验证全部输入/模型/配置哈希和 shard 前缀。未提交的 orphan 文件隔离或显式清理；不以“文件存在”跳过。写满盘、hash 不符、重复 ID、缺行均非零退出，绝不能发布成功 manifest。

## 6. Python 脚本框架与生产接线

以下是**接口设计伪代码，不是可执行实现**，不制造一个带 TODO/NotImplemented 却被称为已完成的脚本。

```python
def run(config):
    # 全部依赖通过显式 config 构造，无默认替代模型/引擎。
    manifest = validate_and_freeze_inputs(config)
    plan = preflight_resources_and_checkpoint(manifest)
    adapter = load_verified_qwen38_adapter(plan)
    writer = TransactionalShardWriter(manifest)
    stats = StreamingCovarianceAccumulator(adapter.feature_dim)

    try:
        adapter.verify_real_probe_and_hook_contract()
        with TargetTokenCapture(adapter, bounded_buffers=config.buffers) as capture:
            for batch in scheduler.iter_batches(manifest):
                capture.begin(batch.ids, batch.mask, batch.nonce)
                with torch.inference_mode():
                    adapter.forward_text_backbone(batch, use_cache=False,
                                                  output_hidden_states=False)
                x = capture.take_exactly_once()  # event、维度、数量、有限值校验
                writer.stage_features(batch, x)
                if batch.split == "train":
                    stats.update(x)
                writer.commit_batch_with_statistics(stats, batch.cursor)
        basis = checked_eigensolve(stats, config.k)
        writer.project_shards_and_validate(basis)
        return writer.publish_committed_manifest()
    except BaseException as exc:
        writer.record_failure_without_publishing_success(exc)
        raise
    finally:
        adapter.close()
```

实际实现必须保证构造期失败也被 CLI 顶层记录；日志故障输出 stderr 并保留原异常，清理异常不能覆盖首要失败。进程被 SIGKILL 时由父进程记录 signal/exit status，并依赖事务恢复，不能依赖 finally 必然执行。磁盘/统计事务应有独立一致性实现，以上方法名不是已经存在的函数。

建议职责落点：

- 改造 `benchmarks/suites/run_universal_extraction_a100.py` 为薄入口，调用公共 orchestrator，移除硬编码路径/维度、全层保留、猜字段、假反事实/相变/轨迹逻辑；不保留新旧执行分支供自动回退。
- `python/gen_zero/causal/universal_manifold_extractor.py` 原地更新 accumulator 和严格产物 API；审查 `GepaEvolutionDaemon` 等全部现有调用，迁移旧状态/checkpoint 时显式版本拒绝或离线转换。
- 公共 `qwen38_extraction` 模块分为 `manifest`, `adapter`, `capture`, `scheduler`, `artifact`, `runner`；只有被 runner 与实际入口调用才计已实现。
- 为 `crates/gen-zero-cli/src/main.rs` 的 Commands 增加明确提取子命令（拟议名称 `extract-manifold`），以 argv 方式启动固定 Python worker，保留原始退出码/信号、转发中断、限定产物目录、验证完成 manifest。不得用 shell 拼接配置，不在 Rust 内复制数值算法。
- 若目标包括在线消费，再实现独立 mount 适配：schema/hash/dim/representation 校验通过后，真实请求进入同一 encoder 与投影路径。只注册文件或打印“loaded”不算生效。当前 `serve --mount-assets` 接受的是 cognitive-assets JSON，不能直接把 NPZ 塞进去假装兼容。

最小实际调用链必须是 `Rust CLI → Python runner → adapter.forward → hook → accumulator → committed artifact`。在线能力另需 `HTTP/MCP request → validated mounted projection → observable result`。二者各自独立验收，前者成功不能冒充后者。

旧逻辑清除验收应针对被替换实现/符号列清单，`rg` 检查 executable source 0 残留，并更新旧测试与文档；本设计文档中的历史证据不是可执行 fallback。本次是方案任务，未删除旧模块，也未声称已完成接线。

## 7. 验收顺序、失败策略与远端闭环

1. 固定官方 model/config/tokenizer/weight-index 的 revision 和 hash，核查 architecture、tensor bytes 与模块图；来源或支持组合不明，状态 BLOCKED，不下载完整巨型权重试运气。
2. 健康检查 + 逐卡/host/磁盘峰值预算通过，远端无 .git 的 sandbox 仅同步源代码与配置；保留文件清单/hash，实际编辑只在本地。禁止自动卸载到另一未验环境。
3. 真模型最小 prefill：hook 对照官方读出；同一真实样本的单条/batch、左右 padding、长短序列、不同 shard owner 逐样本误差与 cosine；对量化与 offload 另外做 matched-pair 报告。阈值先写 manifest，不能看结果后调阈值宣布通过。
4. CPU 数值验证：分批/merge 与中心化参考、巨大偏置/低方差、溢出拒绝、n/k 约束、PSD/特征残差。测试使用人工数值样本是数值单测，不是模型能力证据。
5. 有界内存验证：固定 d、batch/token cap、层数，对不同 N 跑相同链路，记录 RSS/PSS、GPU peak、queue bytes 与磁盘增长。至少测短输入与任务最长输入；不拿几条短 prompt 宣称不会 OOM。
6. 故障注入：真实 hook 漏触发、数据坏行、缺权重分片、磁盘耗尽、摘要篡改、worker 杀死、恢复重放。预期非零退出，无成功 marker、无跳样本、无 fallback，恢复后 ID 集合精确一次。
7. 完整 13 tasks：训练/测试泄漏检查、原始/候选 ID 对齐、产物 schema 校验，先报告提取完整性。下游性能需要与基线共享测试 IDs 的逐样本输出、配对 bootstrap 或适当配对检验、seed/置信区间；没有这些不称突破。
8. Rust CLI 的真实验收包含有效配置成功产物、无效配置非零退出、日志能追到 hook 样本数；若增加 mount，须执行一次真实在线请求和破坏 artifact 的拒绝用例。记录线上 acceptance 单独状态。

每个 run 保存 argv、环境版本、原始退出码/信号、stdout/stderr 完整日志、尾部摘要、输入/输出 hash、资源曲线。父任务失败不得被 `tee/head/tail` 的成功码覆盖。重型编译按节点健康状况使用 `CARGO_BUILD_JOBS=$(nproc)`；提取/BLAS 不能也无条件占满所有线程而挤掉加载与 I/O。跨节点优先分配独立 task/split，前提是不重复加载超出容量；同模型 TP 优先同机高速互联，禁止“全力齐发”变成三个不足容量的重复失败任务。

远端只保留可复用且授权的缓存；临时调试文件在验收后按 run_id 清理。原始验证日志先归档并拉回核验 hash，不能为“零残留”删掉证据。源码不从远端覆盖本地；生成文件按 manifest provenance 拉回。此方案任务不提交、不推送；后续实现的 commit/push 需任务明确授权。

## 8. 本次交付分类与复核入口

**已实现（本次交付）**：本设计文档、只读代码审计、官方资料核对、现有 accumulator 数值诊断及本地 evidence。验证命令与原始退出码见 `evidence/qwen38-extraction-design/commands.json`；数值诊断输出见 `numerics.log`。这里“已实现”只指交付物，不指提取系统已经实现。

**未验证**：dev 实时资源与 GPU 拓扑；checkpoint 实际分片/磁盘占用；精确 HF 模块路径及 hook 张量布局；各 engine/量化/offload 组合在目标机器的运行正确性、吞吐、峰值；任何模型能力提升。

**未完成**：可执行提取脚本改造、稳定 accumulator 修复、Rust/HTTP/MCP 新接线、旧逻辑删除、远端编译与真模型 13-task 提取、训练、生产上线验收。原因：本任务要求系统工程方案设计；本次没有加载模型或执行实现部署。不得以本文的拟议调用链宣称生产已经有引用。

用户规则逐项复核：1 已列致命问题；2 明确失败策略且本次无运行 fallback；3 未作能力声明；4 三类状态及证据已区分；5 生产接线作为交付门槛，尚未实施；6 未执行禁止 git 命令；7 已将实施拆成可验收阶段；8 本次无重型任务/远端任务，故健康检查与清理未执行；9 未派 reviewer 或子代理。
