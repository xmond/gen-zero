# b0927c-t2-wm：Dense 教师引导的零 Token 连续世界模型

研究与工程方案；2026-09-27；工作树 `/ebs/pj/gen-zero`；核查基准 HEAD `acb2c0ccf3f30a708cd9a4f638248973c4709188`。

本次只编写研究报告与证据记录，没有修改代码、训练模型、运行重型编译、部署或派遣子代理。以下“建议”“目标”“验收门槛”均非已实现能力；代码阅读证据与运行证据分别标明。已有 checkpoint 和历史 JSON 的存在不构成本次复现实验。

## 1. 决策结论：能做什么，不能承诺什么

可执行的主路线是：**把 Dense 大模型当作昂贵的表征教师，以真实动作转移训练小型、动作条件、带风险与终止语义的潜空间模型，再由 Rust 规划器在该模型上执行无语言解码的前瞻。** 亚毫秒的候选对象是这个小模型的局部转移核，不是 405B 全模型前向，不是完整搜索，也不是冷启动的端到端决策。

不应以“Dense 天然存在光滑语义势能面，因此辛积分即可推理”作为项目立项依据。它把四件尚未成立的事连在了一起：隐藏状态具有 Markov 充分性、隐藏状态具备已知辛结构、任务转移可由保守动力学刻画、保能运动会提高任务正确率。任何一项不成立，都可能得到一个极快、稳定、能量漂亮，但预测错误的系统。

建议先建立等数据、等预算的残差 MLP 基线，再比较 Neural ODE、保守 Hamiltonian、受控耗散 Hamiltonian 和混合事件模型。**Hamiltonian 结构是待检验的归纳偏置，不是预设获胜者。** 目标是可靠的预测和决策收益，而非一张低能量漂移图。

“零 Token”必须分账：

| 范围 | 允许的计算 | 必须报告 |
|---|---|---|
| 离线教师提取 | 输入分词、大模型前向、可选有明确来源的训练轨迹 | 提取 FLOPs/时间、模型及量化版本、训练数据来源 |
| 在线根状态编码 | 一次教师或已训练学生编码；读取真实观测 | 是否调用大模型、prefill 时间、历史/KV 成本 |
| 想象内循环 | 小型潜空间转移、风险/价值头、合法动作选择 | `generated_tokens=0`、`teacher_forward_calls=0`、真实转移调用数 |
| 环境执行/重新观测 | 真实工具动作和状态更新 | 工具时间、失败、必要的重新编码成本 |

只用缓存特征测得的速度称为“冻结特征后的计算延迟”，不能称为“新任务端到端零 Token 推理延迟”。零语言解码也不意味着零候选动作、零离散事件或零计算。

## 2. 仓库现实：已经存在的能力与危险断层

下表引用当前工作树的源码位置；静态确认不等于上线验证。完整命令、原始退出码与输出存于本目录 `01`—`06` 日志及 `evidence.json`。

| 事实 | 证据位置（相对仓库根） | 工程影响 |
|---|---|---|
| Python 神经世界模型是残差 MLP，加 sigmoid outcome/safety 头 | `python/gen_zero/world_model/neural_dynamics.py:119` | 不需要从零建立训练模型，但其语义须升级 |
| `step` 要求已加载 checkpoint；`forward` 本身不检查加载状态 | 同文件 `:153`、`:119`、`:197` | 生产必须进入受检接口；checkpoint 结构正确仍不证明真实训练或泛化 |
| 奖励是 safety/outcome 概率；`done = r_hat < threshold`，数据无独立 done 标签 | 同文件 `:17`、`:169` | 无法区分目标、碰撞、陷阱、超时；安全概率不能替代任务奖励 |
| Python 客户端可加载神经 checkpoint 并挂载 MCTS/MPC | `python/gen_zero/client.py:432`、`:467` | 已有 Python 路径，不应误报为完全孤岛 |
| 同时配置 Hamiltonian 和神经模型时，客户端给 MCTS 传入的神经模型是 `None` | 同文件 `:471` | 必须拒绝含混配置，不能让“已加载”被理解为“正在生效” |
| Python Hamiltonian 势能/动作矩阵随机初始化；字符串动作变成 hash 驱动随机向量 | `python/gen_zero/world_model/hamiltonian_dynamics.py:113`、`:186`、`:247` | 不是学到的动作语义；字符串 hash 还不保证跨进程稳定 |
| 该模型动作维度错误时静默补零/截断 | 同文件 `:220` | 明确违反本任务 fail-closed 要求；后续实现必须删除此行为 |
| Rust 已有辛积分、接触动力学，以及对应世界模型 | `crates/gen-zero-worldmodel/src/lib.rs:7` | 不应另起重复数值积分孤岛 |
| Rust 辛世界模型使用手设谐振势阱和 ActionId 正弦编码，明确标注未训练、未校准 | `crates/gen-zero-worldmodel/src/symplectic_dynamics.rs:20`、`:34`、`:132`、`:233` | 只能证明数值结构，不能证明语言或环境预测能力 |
| 辛模型确有服务构造和 planner trait 入口 | `crates/gen-zero-service/src/worldsim.rs:240`、`:275` | 不能说其生产引用为零；但未证明 Dense 训练模型已接入 |
| 通用服务默认世界模型仍构造为 `LatentDynamicsWorldModel::default()` | `crates/gen-zero-service/src/zero.rs:818` | `worldsim` 可选辛模型与通用 `pipeline` 默认装配是不同链路 |
| `pipeline` 是 CLI/MCP/HTTP 共用主链入口 | `crates/gen-zero-service/src/pipeline_verb.rs:1`、`:35`；`zero.rs:3439` | 新模型必须注入此链路，不能只新增模拟端点 |
| Rust 状态固定为 1024 维，已有 `step_batch` trait | `crates/gen-zero-core/src/types.rs:152`；`traits.rs:40` | 64 维 Python checkpoint 不能直接塞入；需要显式表征协议和导出 |
| 当前 Rust MCTS 顺序展开、缓存确定性后继，各深度复用同一动作集 | `crates/gen-zero-planner/src/engine.rs:213`、`:247`、`:355`、`:397` | 有 batch 接口不等于搜索已批处理；有 arena 不等于 GPU 大规模搜索 |
| pipeline 将所有 done 当危险，搜索包装器遇到 terminal 直接报错 | `crates/gen-zero-planner/src/pipeline.rs:4`、`:49` | 新模型若对 goal 返回 done，会错误拒绝成功路线；这是接入前的阻断问题 |
| Dense 特征 schema 是静态 train/test/candidate 矩阵与 ID | `benchmarks/suites/gpu_extract_llama405b_13tasks.py:38` | 不包含充分的 `(s,a,s')` 因果轨迹；有提取器不代表五种模型特征都已成功提取 |

用户背景中的 `python/gen_zero/model/set_choice_head.py` 在本次文件清单中未找到；现有目录包含 `choice_head.py`、`dual_head.py` 等。报告不将过时路径当作已核实接口。System 1 在新方案中的职责定义为候选动作先验与排序；具体导出契约须以实际部署实现为准。

### 2.1 不能再用来“证明突破”的三类已有实验

1. `python/gen_zero/scripts/benchmark_issue_86_hamiltonian_world_model.py:40` 使用随机 MLP；`:101` 使用随机 Hamiltonian 模型；`:112` 无动作 rollout。两者不同动力学、不同能量定义，没有配对真实后继。它比较的是构造系统的数值行为，不能证明学到的 Hamiltonian 优于已训练 `NeuralDynamicsWorldModel`。
2. `benchmarks/suites/evaluate_cpu_zero_token_dynamics.py:68` 测随机矩阵递推；`:99` 的收敛过程直接持有目标向量。它是小核算术实验，不能证明未知答案推理或 Dense 世界建模。
3. `benchmarks/suites/benchmark_world_model_mcts_ablation.py:21` 的模型直接访问环境地图与陷阱；`:133` 已诚实标注 privileged exact dynamics。可作为规划诊断参照，不能作为神经世界模型或 Rust 生产 planner 的性能证据。

此外，`scripts/extract_trajectories.py:16` 的状态含安全后继密度、前方阻塞/目标可达类特征。它们是否属于部署可见观测必须逐项判定；若只能靠完整地图/未来模拟得到，就必须放在 privileged 轨道，不能与受限观测模型比较后宣称泛化。这个问题比更换积分器更紧迫。

本次轻量行为复现得到：无 checkpoint 可构造 Python Hamiltonian 模型；两维动作与手动补零到四维动作得到完全相同的力；未训练模型接受字符串动作并返回有限状态。原始退出码为 0。这只验证上述风险行为，没有验证预测能力或延迟，见 `13-behavior.log`。

## 3. 文献给出的支持边界

以下文献用于方法依据，不转移其基准成绩到 Gen-Zero。Firecrawl Research 已执行多方向检索、参考文献扩展和关键正文读取；关键正文/元数据留存在 `07`—`12` 日志。检索不是穷尽式最新排行榜审计。

| 方法族 | 可借鉴内容 | 不支持的推断 |
|---|---|---|
| [Neural ODE，Chen 等](https://arxiv.org/abs/1806.07366) | 用学习向量场定义连续深度，由求解器推进 | 连续形式不保证低 NFE、低延迟或安全 |
| [Hamiltonian Neural Networks，Greydanus 等](https://arxiv.org/abs/1906.01563) | 学标量 Hamiltonian，再取偏导构造动力学 | LLM 隐状态天然是正则相空间；能量守恒等于推理正确 |
| [Hamiltonian Generative Networks](https://arxiv.org/abs/1909.13789)、[Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630) | 从观测学习动力学表征；比较不同力学结构与坐标假设 | 随意把向量前后两半命名 q/p 就识别了物理结构 |
| [基于辛积分器的 Deep Hamiltonian Networks](https://arxiv.org/abs/2004.13830) | 训练时纳入离散积分格式，研究离散模型与修正方程 | 任意控制力、量化和投影仍严格辛 |
| [Port-Hamiltonian Neural Networks](https://arxiv.org/abs/2107.08024)、[Dissipative HNN](https://arxiv.org/abs/2201.10085) | 显式建模输入功和耗散，避免对开放系统强求保能 | 仅有耗散就能识别逻辑不可逆事件 |
| [MuZero](https://arxiv.org/abs/1911.08265)、[TD-MPC2](https://arxiv.org/abs/2310.16828)、[UniZero](https://arxiv.org/abs/2406.10667) | 学习对预测奖励、价值、策略有用的潜动力学并规划 | 无真实环境训练即可把冻结 LLM 表征变为正确的世界模型 |
| [Coconut](https://arxiv.org/abs/2412.06769) | 训练连续 thought；隐藏向量作为后续输入 | 绕开离散 token 就绕开大模型前向，或天然达到亚毫秒 |
| [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) | top-k 路由带来分段计算结构 | 所有 MoE 必然不连续、所有 Dense 必然更平滑 |

Coconut 与本方案的成本边界尤其不同：连续 thought 仍送入语言模型处理。本方案要让搜索内部不再遍历教师网络，因此需要真实训练过的小型代理动力学；不能把前者的实验结果当作后者的证明。

## 4. Dense、光滑性与“势能面”：正确的数学表述

### 4.1 Dense 的合理优势是局部计算图规则，而非天然保守性

固定序列长度、位置与注意力 mask，把输入 embedding 当连续变量。若网络组成算子光滑，归一化分母有正 epsilon，则 Dense 映射在该连续输入域上可微；使用 ReLU 等则通常只分段光滑。文字到 token 的映射并不连续，量化、截断、缓存及离散事件也不继承上述性质。

硬 top-k MoE 可以写成

\[
F(x)=\sum_{i\in S_k(x)}g_i(x)E_i(x).
\]

在专家集合不变的区域可光滑；集合切换处可能出现 Jacobian 跳变，若切换专家输出不匹配，还可能出现函数跳变。软路由、专家匹配或其他约束可缓和这些现象。Dense 只是没有这一项路由切换源，不能据此排序两种模型的实际 Lipschitz 常数、曲率或任务可预测性。

普通向量场 \(f(q)\) 存在局部标量势 \(f=-\nabla V\) 需要相应可积条件；在简单连通域中，对足够光滑场要求 Jacobian 对称。LLM 的残差映射一般没有此约束。**“表征可微”不推出“有势能”，“有势能”不推出“该势能代表真值或安全”。** Dense 隐状态通常也不是从某个已识别能量函数采样而来。

### 4.2 要实测而不是宣称的几何性质

在同一输入、同一语义扰动与相近计算预算下，对 Dense 和 MoE 教师测量：局部 Jacobian 范数、有限差分斜率变化、routing flip 条件统计、后继预测误差，以及低维投影后的任务信息损失。用 JVP/VJP 或随机方向估计，避免构造巨大全 Jacobian。

在可导实现中，比较 \(\|J_F(x+\epsilon v)-J_F(x)\|/\epsilon\)；若只有黑盒提取 API，只能报告有限差分代理，不能称为完整曲率。embedding 扰动可能离开自然语言分布，必须同时用真实文本改写、真实状态扰动与动作干预评估。匹配量化精度、层/池化方式、样本 ID、上下文截断与尺度；不能把不同量化误差归因于 Dense/MoE。

HNN 本身的势能光滑性由小模型参数化控制，不必等教师先证明有物理势能。教师可能提供更可预测的表征，这才是需要消融验证的因果路径。

## 5. 状态构造：先解决充分性，再谈 q/p

设真实环境状态 \(s_t\)，可见观测 \(o_t\)，结构化动作 \(a_t\)，任务/目标条件 \(c\)。Dense 教师 \(E_m\) 给出 \(h_t^m=E_m(o_{\le t},a_{<t})\)。要学习的是

\[
x_t=\Phi_\phi(h_t^m,\text{observable history}),\qquad
p_\theta(x_{t+1},r_t,e_t\mid x_t,a_t,c).
\]

单个末 token pooled hidden state 未必是充分统计量。若两个历史有相近 \(x_t\)，但同一动作产生冲突后继，就有状态混叠；应增加历史编码/记忆或概率 belief，而不是靠能量正则把冲突平均成“稳定预测”。

建议首轮使用 \(x=(q,p)\in\mathbb R^{2d}\)，\(d\in\{64,128,256\}\)，独立保存 context、event 和 uncertainty。该维度是实验网格，不是性能承诺。可先选择 1024 维、512+512 的 Rust 契约完成最小贯通，再决定是否迁移成受版本约束的更小 latent 类型。不同维度必须显式 schema，禁止补零、截断或依靠 shape 恰好能装下。

\(q_t\) 由观测编码；\(p_t\) 由观测历史与前一动作推断。只有观测确有可识别速度时才考虑 \(p=M\Delta q/\Delta t\)。文本任务的伪时间与物理时间不能混用；不能取 \(q_{t+1}\) 来构造在线 \(p_t\)，否则泄漏未来。把原始 h 对半切分只是数组布局，不是学习了正则坐标。

若存在已知原相空间及辛形式，才可要求编码 \(\Phi\) 满足相应辛映射条件。普通 LLM 隐状态没有已知原辛形式；此时只能说“在学习到的坐标中施加 canonical inductive bias”。局部 canonical chart 不保证全局单一坐标；Torus 等拓扑尤其需要周期表示或多 chart 及显式切换。

多教师使用各自 \(\Phi_m\)，在相同观测/动作/episode 上对齐。现有 `cross_model_manifold_alignment.py:128` 的 ID/label 一致性检查可复用，但 CKA 或 Procrustes 的静态相似不能证明动力学共轭。后者需要验证

\[
A_{m\to n}T_m(x,a)\approx T_n(A_{m\to n}x,a)
\]

在未见轨迹与动作上成立。对齐、归一化和降维仅在训练集拟合；验证集用于选择，测试集保持封存。

## 6. 连续动力学、辛积分与不可逆性

### 6.1 先建立一般受控 Neural ODE

\[
\dot x=f_\theta(x,u,c),\quad u=\psi(a,\text{parameters},c).
\]

使用固定步长 RK2/RK4 与同参数量离散残差模型作基线；记录 NFE。部署优先固定小 NFE，以获得可预测延迟。自适应 solver 有误差控制价值，但可能因刚性导致 NFE 和 p99 激增；达到预算就返回显式拒绝，不得悄悄少积分几步。

ODE 的层深度、想象时间、实际动作持续时间是三个不同量。只有离散一步训练数据时，中间连续轨迹通常不可辨识；多个不同向量场可能得到同一步映射。因此不能未经中间观测验证，就称积分子步为真实环境中间状态。

### 6.2 可分离 Hamiltonian 与 Störmer–Verlet

在保守分支中，选常正定质量矩阵 M（初期对角）与动作条件势：

\[
\mathcal H_\theta(q,p;u,c)=\tfrac12p^TM^{-1}p+V_\theta(q;u,c),
\quad
\dot q=M^{-1}p,\quad\dot p=-\nabla_qV_\theta.
\]

每个子步冻结 u,c，令步长 \(\delta\)：

\[
p_{k+1/2}=p_k-\tfrac\delta2\nabla V(q_k;u,c),
\]
\[
q_{k+1}=q_k+\delta M^{-1}p_{k+1/2},
\]
\[
p_{k+1}=p_{k+1/2}-\tfrac\delta2\nabla V(q_{k+1};u,c).
\]

满足相应光滑性与数值条件时，该格式对冻结控制下的可分离系统是二阶、辛且可逆的；并非精确保能。长期能量误差有界的常见结论还有步长、轨迹有界、光滑性等前提。对谐振子，稳定区间需 \(\delta\omega<2\)；高曲率训练势可能迫使更小步长。若 \(M=M(q)\)，则 H 不再按此式可分离，不能沿用这三行并继续宣称辛。

单步动作变化改变 Hamiltonian，本来就可能做功。应分别记录积分残差、控制功、耗散和事件跳跃；不能把真实控制能量变化一律记作积分器失效。

变分路线可选离散 Lagrangian \(L_d(q_k,q_{k+1};u_k)\)，通过受迫离散 Euler–Lagrange 方程

\[
D_2L_d(q_{k-1},q_k)+D_1L_d(q_k,q_{k+1})+F_d^++F_d^-=0
\]

求解。它适用于更一般机械结构，但隐式求解开销、收敛失败与导出复杂性真实存在；暂不作为亚毫秒 MVP 默认路线。迭代不收敛必须报错，不能把最后一次迭代当成合法解。

### 6.3 势能低成本参数化，不重复遍历 Dense 教师

一个可导出、可学习的候选参数化为

\[
V(q;u,c)=\tfrac12q^TKq+b(u,c)^Tq+
\sum_{j=1}^{w}\alpha_j\,\operatorname{softplus}(w_j^Tq+\beta_j(u,c)),
\]

\[
\nabla V=Kq+b+W^T[\alpha\odot\sigma(Wq+\beta)].
\]

K 初期对角；保持适当谱界。该解析梯度与所训练势能严格对应，避免 Rust 内嵌 Python autograd。允许带符号的 \(\alpha\) 才能表达非凸结构；若全部强制凸，不能又宣称模型学到了多个势阱。正定二次项可约束远场，但它不证明训练区域内的预测正确。

禁止独立训练一个任意“力网络”再声称它必然等于此势能梯度。若采用独立力网络，应明确归入一般 Neural ODE，并测 curl/可积性残差。

### 6.4 首选可表达开放系统的受控耗散模型

对实际工具和逻辑任务，更合理的候选族是

\[
\dot x=(J-R_\theta(x))\nabla\mathcal H_\theta(x)+G_\theta(x)u,
\quad J^T=-J,\quad R_\theta\succeq0.
\]

自治 H 下：

\[
\dot{\mathcal H}=-\nabla\mathcal H^TR\nabla\mathcal H+
\nabla\mathcal H^TGu.
\]

显式时间依赖再加 \(\partial_t\mathcal H\)。这是能量收支关系，不是安全证明。初版可限制为 \(\dot p=-\nabla V-\Gamma p+B(q)u\)，使用耗散/保守/耗散 Strang 分裂；常对角阻尼允许精确指数子步。

只有特定均匀阻尼等条件下才能声称 conformally symplectic；一般状态相关 R/G 或事件系统不能继承这一称谓。复用现有 Rust `contact` 数值基础时，要核对其实际方程与目标模型，而不是因为名称相近就直接挂接。

### 6.5 不可逆事件必须显式建模

唯一解的光滑 ODE 在有限时间内通常定义可逆流；纯 Hamiltonian 还保持相体积，不适合把多个状态直接压成同一个吸收失败态。甚至阻尼 ODE 在有限时间内也不自动成为离散意义上的多对一 reset。

加入模式 e 与事件 guard/reset：

\[
\dot x=f_{\theta,e}(x,u),\qquad
g_j(x,u)=0\Rightarrow(x^+,e^+)=\mathcal R_j(x^-,u,e^-).
\]

目标、碰撞、工具失败、资源耗尽、不可逆承诺具有不同事件类型。只有模型预测的事件概率，不应伪装成已发生事实；只有真实环境执行产生观察到的事件标签。guard 定位和 reset 均需监督或明确机制来源。

“相变”在本报告中指行为模式/可达性发生突变；不自动具有统计物理热力学相变含义。有限维神经网络输出变化不构成热力学证明。

## 7. 动作几何：pullback、冲量与真正的语义

动作使用稳定 `ActionId + typed parameters + schema/version + duration`；名称可展示，但不能以 hash 随机向量代替语义。有限固定动作可使用经训练的 embedding 或明确坐标编码；这只是表示，不是“硬编码推理”。未知动作、非法参数和缺失 embedding 一律拒绝。工具文本描述的教师编码可在根节点/缓存阶段完成，不在每个搜索子步调用 405B。

令潜空间到教师表征的可学习浸入/局部解码为 \(h=D_\psi(q)\)，Jacobian \(J_D\)。教师空间的作用力是协向量 \(\alpha_h\)，它的自然 pullback 为

\[
\alpha_q=D^*\alpha_h=J_D(q)^T\alpha_h.
\]

向量场没有任意映射下无条件成立的同类 pullback。若欲把教师空间目标速度 v 投影到潜切空间，可在指定度量下解

\[
v_q=(J_D^TJ_D+\lambda I)^{-1}J_D^Tv_h,
\]

这是正则化最小二乘投影；不是无损几何同构。秩不足、离流形投影残差大必须暴露。完整 Jacobian 太贵时离线蒸馏为低秩 B(q)，并测动作响应误差。

若动作产生标量势改变量 \(U_a(q)\)，则 \(-dU_a\) 是力的一形式；由辛形式建立它与 Hamiltonian 向量场的关系。外微分 d 是数学算子，不是能从动作字符串中自动挖出因果效应的模块。

冲量建模为 \(q^+=q^-\)、\(p^+=p^-+I_\theta(q,u)\)。该映射只有在冲量对应适当闭一形式（局部为标量梯度）等条件下才辛；任意神经 B(q)u 不满足自动保证。非保守工具动作应使用受控力/事件分支，明确放弃不适用的守恒声明。

势阱可被动作改变，例如 \(V(q;u)=V_0(q)+\sum_i u_iU_i(q)\)。它有助于表示吸引/排斥和决策边界，但势阱深度必须通过后继和任务监督学习，不能人为设置“正确答案低能量”再在测试时输入答案。

MCTS 始终从合法动作集合搜索；连续向量仅是动作编码。离散工具不能对 embedding 做 CEM 后随意 nearest-neighbor 映射为动作并宣称连续控制。现有 Rust MPC 若采用离散候选，保留 categorical 分布；真实连续参数才允许在合法范围内优化，参数耦合约束需显式处理。

## 8. 安全、目标与不可逆陷阱：拆开监督与规划语义

模型返回至少六类独立信息：后继分布、任务奖励、即刻 hazard、goal、terminal reason、认知不确定性。`safe_or_goal` 合并标签丢失了关键区分，必须新增数据，无法从当前一个二元 r 凭空还原。

安全集合 \(\mathcal S\) 与目标集合 \(\mathcal G\) 分别定义。可学习 barrier margin \(b_\theta(x)\)，但只有经过相应验证/形式证明才是安全证书。碰撞可发生在两个安全端点之间；有中间观测时训练路径风险，没有中间观测时只能报告一步事件概率及盲区，不得把隐空间插值称为真实扫掠碰撞检测。

陷阱是可达性问题：对于有限 horizon H，可定义到达目标且不中途失效的最优概率

\[
V_0(s)=1_{\mathcal G}(s),\qquad
V_{h+1}(s)=1_{\mathcal G}(s)+1_{\mathcal S\setminus\mathcal G}(s)
\max_{a\in\mathcal A(s)}\mathbb E[V_h(s')].
\]

真实不可逆失败的定义依赖动作空间、可见信息和时域。H=10 到不了目标，不等于永远无法到达。必须区分 `hazard_now`、`unreachable_within_horizon`、`irreversible_failure` 与 `unknown`；最后一类不能自动算安全。

风险约束规划可优化

\[
\max_\pi\;\mathbb E[\sum_{t=0}^{H-1}\gamma^tr_t+\gamma^HV(x_H)]
\quad\text{s.t.}\quad P(\exists t\le H:\text{hazard}_t)\le\epsilon.
\]

只有正确条件化的生存概率才可递乘；一般校准分数不能假设独立。若每步有有效概率上界，可用 union bound 作保守预算，但在分布外不保证继续有效。记录 Brier、NLL、可靠性图、风险—覆盖率和置信区间；不能拿 normalized action entropy 当作模型认知不确定性。

模型错误（NaN、未知 schema、checkpoint 错配、预算不足、缺风险估计）是 `Err`，默认使整个 plan 拒绝；预测到的合法 hazard 是模型结果，可按明确策略剪枝并记录原因。goal 是成功 terminal，不应触发危险拒绝。当前 `HazardCheckedDynamics` 的语义必须与此一起迁移，不能靠把 goal 的 done 改成 false 来绕过。

## 9. 性能：亚毫秒来自计算规模缩减，不来自数学名词

### 9.1 大模型重复前向的计算账

用理想 roofline 估计而非实测承诺：一次 batch=1 Dense token/latent pass 的参数主导计算量约为 \(2P\) FLOPs，未计注意力上下文等成本；若权重必须从高带宽存储读取，还需至少相应权重流量。粗略下界：

\[
t\gtrsim\max(2P/F_{\rm eff},\;Pb/B_{\rm eff})+t_{\rm communication}.
\]

以 **假设** 聚合有效带宽 10 TB/s、BF16 2 bytes/参数且每次权重流过该层存储为例：405B 权重约 810 GB，对应 81 ms 的带宽项；理想 4-bit 原始权重约 202.5 GB，对应 20.25 ms，尚未计量化元数据、反量化和通信。这个例子不是对现有机器的测量，也不是跨所有硬件的绝对下界；它说明“省去采样/softmax”不足以推出 <1 ms。

Dense 教师不进入每个 \(\nabla V\) 计算。如果每次梯度都回传 405B，Verlet 两次梯度可能比普通一次前向更贵。

### 9.2 小核的可测目标

上述 shallow potential 每次梯度主要是两次 d×w 矩阵向量运算，约 4dw FLOPs；Verlet 两次梯度约 8dw，另加动作、风险、价值、耗散和内存成本。d=128、w=256 时，仅此梯度部分约 262,144 FLOPs。多层势能、ensemble、子步数均会扩大成本；这只是设计估算，**不是实测微秒数**。

候选实现：常驻 Rust 权重、预分配 scratch、SIMD/GEMV；批量较大时使用 GEMM/GPU，测量拐点后决定。不要按节点调用 Python/HTTP；不要为了 GPU 名称牺牲 B=1 的调度延迟。量化须配对验证轨迹和事件边界，不以单次输出相似替代。

总账：

\[
T_{decision}=T_{encode}+T_{legal/actions}+T_{search}+T_{gate/audit}+T_{serialization}.
\]

H=10 的单条轨迹仍有时间依赖，不能把 B 条轨迹的吞吐均摊时间当作单条延迟。顺序 MCTS 的模型调用数由缓存与 rollout 决定，数量级可达 simulations×H，必须实际计数。即使每次 0.2 ms，1000 次也有约 200 ms 纯串行核耗时。

验收分别报告：B=1 的完整 transition p50/p95/p99；B=8/32/128 的批次延迟与吞吐；H=1/5/10/20 的 rollout；固定决策预算的完整 planner；包含/不包含教师编码的端到端结果。GPU 计时要同步，冷启动、预热、并发负载、CPU affinity、硬件/频率、精度、NFE 与拒绝率必须随报告发布。初始门槛可设 `B=1 transition p99<1ms`，但在测量前只称目标。

## 10. 生产接入方案：从导出到 Rust arena 的完整链

目标链路：真实观测 → 已版本化 encoder → learned latent → 合法动作/PolicyGate → `ProductionPipeline` → MCTS/MPC → 加载训练权重的 Rust world model → 转移/风险/事件 → 备份价值 → 真实动作执行 → 新观测校正。System 1 给合法候选的先验，不替代转移模型。

### 10.1 模型包和状态协议

模型包必须绑定：format version、模型族、checkpoint hash、训练运行 ID、训练/验证数据 hash、encoder/model/tokenizer/量化 hash、层与 pooling、normalization、latent schema、action schema、事件标签定义、积分器/dt/substeps、校准集 hash、导出精度及数值容差。`trained=true` 自报字段不能充当训练证据；需要可追溯训练日志及验收产物。

加载阶段验证 shape、finite、质量矩阵正性、积分参数、动作注册表与校准身份。缺文件、不兼容或未校准的安全决策路径显式失败。禁止自动改用谐振器、旧残差默认模型、零向量或字符串 hash。

建议版本化的逻辑接口（设计草案，尚未实现）：

```text
ModelIdentity = weights_hash + latent_schema + action_schema + calibration_hash
LatentState = values + schema_id + encoder_id + observation_version
Action = stable_id + typed_parameters + duration + schema_id
Transition = next_state + task_reward + event_distribution
             + terminal_reason + hazard_probability + uncertainty
             + numerical_diagnostics + model_identity
TerminalReason = None | Goal | Hazard | EnvironmentFailure | TimeLimit
WorldModel.transition_batch(requests, output_buffer) -> Result<(), ModelError>
ActionProvider.legal_actions(state, context) -> Result<ActionSet, ActionError>
```

不能把 source/action-dependent hazard 偷塞入当前只收到 `(next_state,reward,done)` 的 `safety_estimate` 再用全局“上次预测缓存”取回：并发下会串线。应让安全估计随同一次 Transition 原子返回。批调用失败时调用方禁止消费部分已写输出；明确 whole-batch 原子发布或逐项显式状态，不能默认缺失项安全。

### 10.2 迁移顺序与文件责任

| 阶段 | 主要现有位置 | 必须实际完成的行为 |
|---|---|---|
| Python 参考模型与训练 | `python/gen_zero/world_model/neural_dynamics.py`；`scripts/train_world_model_dynamics.py` | 独立 reward/safety/goal/event，真实多步数据训练；保持已训练 residual 基线 |
| 统一 trait/事件契约 | `crates/gen-zero-core/src/traits.rs`；`types.rs` | Transition v2、schema 与动作参数、错误语义；更新所有实现与调用方 |
| Rust 模型核与导出 | `crates/gen-zero-worldmodel/src/` 现有积分模块 | 导入训练权重，Python/Rust 按样本对齐；解析梯度和标量势一致 |
| planner 语义 | `crates/gen-zero-planner/src/engine.rs`、`pipeline.rs` | goal 正常终止，hazard 可审计剪枝，模型错误拒绝，动态合法动作与风险预算 |
| 服务装配 | `crates/gen-zero-service/src/zero.rs:818`；`pipeline_verb.rs`；`worldsim.rs` | 同一模型实例注入 pipeline 与相关入口；去掉生产默认手设模型的隐式选择 |
| 外部验收 | CLI、HTTP `/v1/pipeline/{op}`、MCP | 同一真实 fixture 的请求、响应、调用计数和模型 hash 一致；不只测试模拟函数 |

单个 PR 不必同时完成全部研究变体，但一个生产发布必须完成对应模型的端到端契约迁移。不能发布“trait 已更新，入口仍旧模型”的半成品。这里列的是未来修改范围，本次未修改这些文件。

### 10.3 MCTS arena 与 MPC 批量化

第一阶段先在现有顺序 MCTS 上验证真实模型生效，避免把算法变化和模型变化混为一谈。第二阶段再改 arena：节点元数据和连续状态 buffer 分开；记录 model/schema/version、父节点/动作、terminal、风险和 uncertainty；生命周期绑定一次不可变模型快照。

当前 1024 f32 状态每节点仅数据就需 4096 bytes；一百万节点约 4.096 GB，尚未计 children、价值、索引与 allocator。所谓 64-byte 节点描述符不能代表完整状态内存。

批 MCTS 使用明确的叶节点选择、in-flight 标记/virtual loss、批转移和备份协议；保证同一 simulation 不重复入队、失败释放预约、超时不消费残缺结果。多根独立请求更容易批处理。MPC 则按 horizon 顺序、同一层不同候选批处理。现有 `step_batch` 循环实现必须替换为真实批核后再声称 SIMD/GPU 加速。

若动力学是随机分布，当前每边缓存单一确定后继不再正确。需固定粒子/ensemble 语义、chance nodes 或明确风险估计；不能第一次随机采样后永远当作真后继。若实验先采用确定性均值模型，必须承认多峰丢失风险。

缓存 key 包含状态、模型版本、动作参数与时间，不只 ActionId。近邻合并/量化碰撞必须单独消融；不得为速度静默把不同语义状态合并。

### 10.4 Fail-closed 与可审计降级

未知动作、错误维度、非有限值、encoder 错配、checkpoint 缺失、风险头缺失、不支持的积分模式、事件过密、隐式求解失败、NFE/时间预算耗尽均为 typed error。明确区分“数值失败”“认知不足”“不合法动作”“预测到危险”。

如业务允许返回此前已验证 incumbent，响应须包含 `status=partial_budget_exhausted`、实际 horizon、完成 simulations、模型身份、风险检查覆盖和原因；不得和完整规划成功共用不可区分的结果。高风险路径仍可规定超时一律拒绝。当前 `engine.rs:188` 的超时返回 incumbent 行为必须纳入此状态契约，不能只给 entropy=1 就算完成解释。

复用/替代边界要在实现任务中先列符号清单：废弃的随机 Hamiltonian 模型、hash 动作、静默补齐及旧生产默认选择若被替代，应从代码、导出、配置、调用方、测试与文档一起删除；对该清单运行全仓 `rg`，预期无匹配（退出码 1）。底层正确积分器可直接复用，不应为了“全删”而重写。保留的 residual 基线是显式实验对照，不得兼任失败 fallback；本次没有替代代码，因此不声称旧符号已清零。

## 11. 训练数据与目标函数：最大的工作量在这里

静态分类数据只给 `(observation, candidates, label)`；跨 Transformer 层隐藏状态是网络计算深度，不是环境时间。把层差当作 `(s_t,s_{t+1})` 可研究计算流，但必须独立命名，不能冒充动作条件世界模型。

每条真实转移建议记录 episode/environment/layout/family ID、observed history、合法动作集合、实际动作及参数、执行时间、下一观测、奖励、终止原因、hazard/goal/viability 标签及标签来源、teacher identity、feature hash。teacher 用已发生的下一观测编码，不能仅靠自己生成的预测当 ground truth。探索/干预数据须含失败、罕见事件及同状态不同动作；只有单策略日志且缺动作支持时，不得声称识别了全动作因果效应。

数据分离按 episode、布局、文档/问题 family 和来源，而非随机行；设独立校准集。预先冻结测试 manifest。动作参数、措辞和模型量化分布变化也进入 OOD 测试。teacher 分词及预算必须与 schema 绑定。

建议多步训练目标：

\[
\mathcal L=\sum_{k=1}^{K}w_k\{\lambda_z\ell_z(\hat x_{t+k},\operatorname{sg}(\bar\Phi(o_{\le t+k})))
+\lambda_r\ell_r(\hat r,r)+\lambda_e\operatorname{CE}(\hat e,e)
+\lambda_s\operatorname{BCE}(\hat p_{hazard},y_{hazard})
+\lambda_v\ell_v(\hat V,V^{target})\}+\lambda_{reg}\mathcal R.
\]

潜变量目标使用冻结或 EMA target encoder，另外保留任务预测/方差约束，防止编码全零坍缩。状态多峰时用分布损失而非仅 MSE；均值可能落在没有物理意义的位置。target value 需明示来自真实回报、可验证 simulator 或 bootstrap，各自误差分开报告。

\(\mathcal R\) 可含 Jacobian/曲率控制、合法能量收支残差、表示稳定性；不能在有控制功/耗散/事件的数据上强制 \(\Delta H=0\)。训练时使用与部署相同的积分器，先一步，再逐渐增加真实多步 unroll；同时报告 teacher forcing 和自由 rollout，不能只展示前者。

先冻结教师，训练 projection + residual + heads；再在相同数据/参数/调参预算下换动力学；最后才评估多教师融合和学生编码。不能同时换数据规模、教师、模型容量、planner 预算后把提升归给辛积分。

## 12. 基于现有 suites 的可复现实验矩阵

### 12.1 先统一基准身份

当前有两套不同“13 tasks”：`grand_challenge_data.py:49` 是 MASSIVE en/de、MultiNLI、PubMedQA、VitaminC、BoolQ、SQuAD2、PAWS、Civil Comments、Aegis、HelpSteer2、SummEval relevance/consistency；`evaluate_full_suite_cpu_dynamics.py:78` 使用包含 ARC/GSM8K 等的旧集合，其说明是每任务 30 个冻结 9B 样本。

不得把它们混合成同一“13-task 成绩”。运行身份需含 suite 文件 hash、task 清单、split、行数、ID/family hash。3880 条等注释信息需在正式运行时从数据重新计数。本次未审计全部五类教师产物是否实际存在、完整且对应同一 manifest。

### 12.2 四条实验轨道

| 轨道 | 复用入口/数据 | 能证明的范围与必要修改 |
|---|---|---|
| A：静态表征与零解码读出 | `grand_challenge_data.py`、Dense 提取器、`cross_model_manifold_alignment.py`、`equivariance_suite.py` | 测候选准确率/排序、教师贡献、排列等变性；不能证明环境 rollout |
| B：真实受控转移 | `deadlock_torus_env.py`、`scripts/extract_trajectories.py`、`scripts/evaluate_world_model_dataset.py` | 区分文件中 `TorusWorld` 与 `DeadlockTorusEnv` 两个 API；声明观测可见性，补齐 goal/hazard/trap/duration 标签 |
| C：生产规划收益 | Rust MCTS/MPC + CLI/HTTP/MCP，Torus 未见布局 | 替换 privileged exact model 为实际导出训练模型；相同 planner/预算配对，记录真实环境结果 |
| D：数值与系统成本 | 现有 latency/profile suites + 真实模型产物 | 数值保真、batch 收益、端到端延迟/内存；随机小核只作独立 microbenchmark |

静态 NLP 轨道若想研究多步“逻辑动作”，必须新建有真实执行语义的环境，例如受验证的证据检索动作、约束更新或工具查询；动作不应是“猜下一步文字”。原始静态样本本身不提供这些转移监督，缺数据即记未完成。

### 12.3 必做消融

| 问题 | 同时保持不变 | 对照 |
|---|---|---|
| 有没有必要规划？ | encoder、训练数据、测试 episode | System 1、H=1、H=5、H=10；匹配墙钟预算 |
| 连续形式有用吗？ | 参数量/训练步数与数据、事件头 | residual MLP、固定 NFE Neural ODE |
| 辛结构有用吗？ | 相同坐标、势能容量、数据 | 同一学习系统的非辛/辛积分；另比较训练好的 unrestricted MLP |
| 守恒假设是否有害？ | 样本/动作与容量预算 | conservative、damped/port、hybrid events |
| 动作是否真的影响预测？ | state encoder | 真实动作、去动作、训练期动作 shuffle 负对照；绝不以负对照作为生产实现 |
| 教师越大是否越好？ | 配对样本、下游训练预算、精度/提取元数据 | 各 Dense 教师、无教师/小 encoder；另控预算对比 MoE |
| q/p 是否提供信息？ | 训练数据 | learned history momentum、单状态、已知物理速度（只在可观测轨道） |
| 改进来自耗时吗？ | 墙钟和调用数分别匹配 | 更多积分子步、更多 simulations、同总算力 MLP |
| 是否只是终点分类？ | 预测头/数据 | 直接 readout、一步模型、多步自由 rollout |
| 量化/批处理是否破坏语义？ | 同一权重与逐样本输入 | Python FP32、Rust FP32、后续量化；B=1 对 B>1 |

额外测候选排列。策略分数应按候选置换等变；并列最优时比较最优集合或以稳定动作 ID 明确 tie-break，不能把 slot 顺序造成的差异隐去。

### 12.4 配对统计与“可宣称”门槛

为每个 sample/episode 保存 baseline 与 variant 的预测/动作、真实结果、hazard/goal、风险、延迟、模型调用数、种子、模型/data hash、错误状态。分类配对用 McNemar；回报/成功率差用 episode/family cluster bootstrap；多 seed 是训练随机性，不能把同一样本多次推理当作独立样本扩充 n。多消融预先指定主要终点并校正多重比较。

准确率差 \(\Delta=\frac1N\sum_i(I_i^{new}-I_i^{base})\) 必须附配对置信区间和原始 b/c 分歧数；只有均值变化没有逐样本配对，不声称突破。安全漏报需单侧上置信界；例如 n 个独立风险样本零漏报时，95% 上界近似 3/n，而非零风险。采样相关时不能套独立样本结论。

建议预注册验收：正确率/成功率提升的配对区间下界 >0；安全漏报上界不劣于预设容忍值；B=1 transition p99<1ms；固定墙钟下规划收益仍存在；新入口调用计数非零；错误注入全部 fail-closed。具体风险容忍值由部署任务确定，在确定前不能发布“安全通过”。

模型数值稳定、风险校准、任务有效与生产接通是四个独立 gate，不能互相替代。若 Hamiltonian 无收益而 residual 达标，应明确报告结构假设未获支持，而非改指标救结论。

## 13. 技术路线图与停止条件

| 阶段 | 可交付物 | 出口条件 |
|---|---|---|
| P0 事实与契约 | 数据/模型 manifest；终止、安全、动作与 schema 定义；旧符号替代清单 | 没有混用静态/轨迹数据，没有把 goal 视作 hazard |
| P1 可信基线 | 用可部署观测训练 residual；保留独立校准与测试；导出模型包 | 未见 episode 有配对预测统计；训练与产物可追溯 |
| P2 最小生产贯通 | Python/Rust parity；CLI/HTTP/MCP 同模型；现有顺序 MCTS 调用真实 checkpoint | 真实请求 trace 中模型核调用>0，模型缺失请求失败；goal/hazard 分离正确 |
| P3 动力学结构实验 | ODE、Hamiltonian、受控耗散、hybrid 受控消融 | 数据支持才能选择结构；不支持就保留训练基线并报告负结果 |
| P4 性能 | 解析梯度、预分配、MPC batch、再到批 MCTS；必要量化 | 完整 transition p99 与固定预算决策均达标，失败/拒绝计入结果 |
| P5 Dense 扩展 | 五教师逐样本比较、跨模型动力学一致性、学生根编码 | 收益超过成本且 OOD 不恶化；不以参数量替代证据 |
| P6 生产迁移与清理 | 统一装配、错误注入、旧符号零残留、可回滚版本产物 | 两类数值/语义检查和各入口真实验收均通过；外部 reviewer 独立评审 |

阶段不是日期承诺。最可能阻断的是缺失真实转移数据、表征状态混叠及校准不足，未必是 Rust 算力。只有静态题目数据时先完成 A 轨道，明确 B/C 未完成；不能制造“虚拟轨迹”假装闭环。

后续确有重型训练/编译时，按用户指定远端规范执行：先核验每台目标机 1m/15m 负载、CPU 数、可用内存≥8GB、磁盘≥10GB，并结合任务实际峰值留余量；只向无 `.git` 沙箱传内容校验后的源码快照；隔离可复用缓存，避免多个全核任务互抢；`CARGO_BUILD_JOBS=$(nproc)` 只在该节点负载与内存允许时使用。保存命令、环境、完整日志、原始退出码与产物 hash，回传时校验来源。最终报告/验收日志保留在本地，临时远端日志和沙箱清理。提交推送需后续明确任务授权；本次未做任何远端重任务、提交或推送。

## 14. 证据、复查命令与三类成果

### 已实现 / 本次已完成的交付

完成源码核查、文献核对、数学设计、接口/迁移路线、消融与统计计划，以及本报告。新算法实现数量为零，符合“不要修改代码”。

源码证据命令是 `evidence.json` 中保存的 argv，可原样从仓库根执行。例如：

```bash
rg -n 'pub type FullLatent|fn step_batch|fn safety_estimate|world_model.step|struct MctsNode' crates/gen-zero-core/src crates/gen-zero-planner/src/engine.rs
rg -n 'world_model: Arc::new|execute_pipeline|SymplecticWorldModelDynamics::default|as_dyn|neural_dynamics_model|dynamics_model=None' crates/gen-zero-service/src/zero.rs crates/gen-zero-service/src/worldsim.rs crates/gen-zero-service/src/pipeline_verb.rs python/gen_zero/client.py
rg -n 'rng.uniform|rng.randn|Pad or truncate|hash\(action\)|r_hat <|weights_loaded|load_checkpoint' python/gen_zero/world_model/hamiltonian_dynamics.py python/gen_zero/world_model/neural_dynamics.py
```

上述正式核查日志 `01`—`06` 原始退出码均为 0。`05` 输出尾部包含 `hamiltonian_dynamics.py:221: # Pad or truncate to match action_dim` 与 `:249: h = abs(hash(action)) % (2**31)`。行为复现退出码 0，输出见 `13-behavior.log`。这些是风险证据，不是模型验收通过证据。

探索中曾对不存在的 `gen-zero-api/gen-zero-mcp/gen-zero-server/src` 路径做 rg，工具返回退出码 2；随后依据真实 `gen-zero-service` 路径完成核查。未把那些失败检索当作“全仓零引用”证据。初始清单搜索也使用过截取输出，只用于定位，不作为完整性或测试成功证明。正式证据命令无构建管道截断。

### 未验证

五教师特征完整性与模型权重实际可用性；本工作树现有 checkpoint 的训练质量；新模型的 Python/Rust 数值一致性；Dense 相比 MoE 的可预测性优势；Hamiltonian 相比 trained residual 的收益；亚毫秒完整转移、端到端延迟与安全校准。这些均需要未来真实执行。

### 未完成

真实转移数据补齐、新模型训练/导出、trait 与 terminal 迁移、Rust 主链装配、批 MCTS、旧模块删除、生产部署以及独立 reviewer 验收。原因是本次任务明确限定研究与方案设计、禁止修改代码；不能把设计文本写成已上线成果。

审查四项的结论：当前发现部分模型有真实入口，不能泛称孤岛；发现 Python 静默动作形状修正和非语义随机动作；现有数值基准不足以支持世界建模突破；拟议新模型尚未实现、上线或替代旧逻辑。报告提供的是能够逐阶段证伪的实施方案。
