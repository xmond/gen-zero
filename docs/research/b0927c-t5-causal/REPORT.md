# b0927c-t5-causal：不可逆动作审计与保形安全屏障方案

**结论先行：目前不能证明 Gen-Zero 对不可逆动作零漏报。** Dense 模型的隐空间平滑性、切空间方向或曲率，都不能单独赋予一个向量 Pearl 意义上的因果语义。保形预测能在明确的交换性条件下给出有限样本**边际覆盖**，不能推出任意危险动作的零假阴性，更不能保证分布漂移或对抗输入下的安全性。方案应把模型用于发现风险，把执行权限交给强制经过的门禁、显式确认和可核验的审计记录。

## 一、现状核查：最容易产生虚假安全感的地方

| 发现 | 证据与含义 |
|---|---|
| Rust `PolicyGate` 默认约束表和不可逆动作确认表均为空；未命中约束、确认表且熵低时返回 `Tier0Proceed`。 | [policy.rs](../../../crates/gen-zero-gate/src/policy.rs#L43)、[policy.rs](../../../crates/gen-zero-gate/src/policy.rs#L219)。**默认门禁不是“识别所有不可逆动作”的证明。** |
| Rust `audit_action` 会拒绝缺少安全估计的轨迹，但在估计未校准时仍可能返回 `Approved`，同时附加“未校准”的文字说明。 | [pipeline.rs](../../../crates/gen-zero-planner/src/pipeline.rs#L571)、[pipeline.rs](../../../crates/gen-zero-planner/src/pipeline.rs#L610)。说明文字不能代替放行条件。 |
| Rust 对外接口说明把默认 residual 动力学明确称为**未训练先验**；另有独立的 `worldsim` 动词路径，以及 `ProductionPipeline` 路径。只接入其中之一会留下旁路。 | [server.rs](../../../crates/gen-zero-service/src/server.rs#L1631)、[zero.rs](../../../crates/gen-zero-service/src/zero.rs#L1545)、[pipeline_verb.rs](../../../crates/gen-zero-service/src/pipeline_verb.rs#L105)。 |
| Python `audit_action` 对文本启发式拒发 `APPROVED`，这是有价值的防腐处理；但其神经模型分数被直接当作 `safe_prob`，现有代码没有在此链路核验该分数的保形校准证书。`what_if` 返回排序和 `best_candidate`，不是执行许可。 | [client.py](../../../python/gen_zero/client.py#L1417)、[client.py](../../../python/gen_zero/client.py#L1491)、[client.py](../../../python/gen_zero/client.py#L1513)。HTTP 入口见 [app.py](../../../python/gen_zero/service/app.py#L925)。 |
| 现有 `ConformalMarginGate` 使用固定 `theta` 比较 margin；所查代码未见由独立校准样本计算保形分位数的过程。名称中的 “Conformal” 不应被当作覆盖证明。其评测还区分了可部署视图和使用真实类别的 *oracle* 诊断视图。 | [conformal_margin_gate.py](../../../benchmarks/suites/conformal_margin_gate.py#L68)、[aegis_dual_track.py](../../../benchmarks/suites/aegis_dual_track.py#L190)。 |
| Rust MMR 有可选持久化；未配置时是进程内账本。所查主流程中 `ask` 成功后调用 `record_decision`，因此不能据此宣称每次 `what_if`、`audit_action` 及工具执行都已持久留痕。MMR 证明记录未篡改，不证明判断正确。 | [zero.rs](../../../crates/gen-zero-service/src/zero.rs#L791)、[zero.rs](../../../crates/gen-zero-service/src/zero.rs#L1579)、[zero.rs](../../../crates/gen-zero-service/src/zero.rs#L2274)。 |
| 13-task 中直接标为 safety 的是 `aegis_safety` 和 `civil_comments`；前者标注的是提示词内容安全，后者是文本毒性，不是工具执行后的损害结果。已有文档所列冻结集各 30 条样本表现，不能充当本方案的动作安全实验结果。 | [grand_challenge_data.py](../../../benchmarks/suites/grand_challenge_data.py#L49)、[grand_challenge_data.py](../../../benchmarks/suites/grand_challenge_data.py#L366)、[README.md](../../../docs/zero/README.md#L136)。 |

## 二、因果流形：把可操作的干预与几何诊断分开

定义上下文 \(C\)、权限与资源状态 \(E\)、候选动作 \(A\)、外生扰动 \(U\)、实际后果 \(Y\in\{\text{safe},\text{irreversible harm}\}\)。目标量是
\[
P\!\left(Y=\text{harm}\mid do(A=a),C=c,E=e\right).
\]
`do(A=a)` 必须在一个**明确的结构因果模型或隔离的可复现实验环境**中替换动作机制，并保持可比的 \(C,E,U\)。仅计算
\(\phi_\theta(c,a)-\phi_\theta(c,a')\) 是表征差分，**不是** do-operator。因果表征的可识别性通常还需要干预环境等额外条件；不能从高维观测表征自动获得。[因果表征可识别性研究](https://arxiv.org/abs/2306.00542)。

拟议的几何层只承担两个任务：

1. **发现对扰动敏感的候选。** 在固定模型版本与状态下，估计危险评分 \(r(\phi)\) 沿有物理或语义定义的动作扰动 \(v\) 的方向导数 \(D_vr\)，以及局部二阶变化 \(v^\top H_rv\)。用成对的权限、目标路径、事务范围、回滚条件变更来定义 \(v\)；不能把任意隐空间方向命名为“删除因子”。
2. **检测所谓不变核是否真的不变。** 由事先规定的无害改写配对样本估计噪声子空间 \(N\)，可研究度量 \(G\) 下的投影
   \[
   P_{\perp}=I-N(N^\top GN+\lambda I)^{-1}N^\top G.
   \]
   但投影可能删掉关键风险线索。因此原始输入、投影表征和显式环境变量须同时进入审计；任一路径显示危险或彼此冲突就升级，**投影结果不得单独批准动作**。

反事实训练数据应包含同一初态下的动作配对、明确的权限与资源快照、观察到的后果及未观察到的标记。`rm -rf`、脏写和越权测试只能在隔离靶场与可恢复快照中执行；没有真实后果标签时只能报告“模拟推断”，不能报告因果效果。405B 的规模既不改变识别条件，也不自动提高安全覆盖。

## 三、保形屏障：能够证明什么，不能证明什么

冻结模型、评分函数及其几何特征后，另取**未参与训练和选阈值**的校准集 \((X_i,Y_i)_{i=1}^n\)。设非符合分数为 \(s(X,y)\)，例如针对两个标签分别计算 \(1-\hat p(y\mid X)\)；曲率、能量、反事实差异可以进入预先冻结的 \(s\)。令
\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil,\qquad
q_\alpha=\begin{cases}
s_{(k)},&k\le n,\\
+\infty,&k>n,
\end{cases}
\quad
\Gamma_\alpha(x)=\{y:s(x,y)\le q_\alpha\}.
\]
在校准样本与下一样本的分数**可交换**、评分规则冻结且标签定义一致时，分数的秩论证给出
\[
P\{Y_{n+1}\in\Gamma_\alpha(X_{n+1})\}\ge1-\alpha .
\]
这是一项边际结论。若仅在 \(\Gamma_\alpha(x)=\{\text{safe}\}\) 时允许进一步进入执行门禁，则
\[
P\{Y=\text{harm}\ \land\ \text{模型屏障放行}\}\le\alpha .
\]
它**不**保证 \(P(\text{放行}\mid Y=\text{harm})\le\alpha\)：危险动作很稀少时，全部错误都可能集中在危险类。无条件、逐输入的分布无关条件覆盖一般也不可得。[条件覆盖限制的原始论文](https://arxiv.org/abs/1903.04684)。

可为预先定义的危险类别分别校准，使交换性成立时获得**类别条件**的覆盖结论；类别必须有足够且独立的真实标签。若 \(\alpha=0.01\)，少于 99 个相应校准样本时，上述非随机分位数规则连有限的 \(q_\alpha\) 都取不到。即使独立测试中 30 个危险样本零漏报，其危险类漏报率的单侧 95% 二项上界仍约为 **9.5%**，不能写成“零风险”。分布漂移、选择性上线、人工改写及自适应攻击会破坏校准条件；检测到这些情况必须撤销证书并升级，而不能沿用旧阈值。

**可以给出确定性证明的是软件互锁性质，而不是模型永不漏报：**在所有执行入口确实强制经过同一门禁、输入与证书均验证、并且执行对象没有检查后替换的前提下，`harm ∈ Γ`、空集、缺证书、超时、审计写入失败或依赖失效均不会走到自动执行。该性质需要入口覆盖与故障注入测试证明；目前尚无这份证明。

## 四、生产接入契约

拟议统一返回 `SafetyEvidence`，字段至少包含：规范化动作及资源标识、上下文与权限快照哈希、模型及特征版本、训练与校准数据清单哈希、标签定义、适用域、校准样本数及 \(\alpha\)、预测集、几何诊断、模拟器来源、失效原因、最终 tier、MMR 叶与持久化状态。缺字段或版本不匹配即拒绝自动放行，并记录原因。

```text
audit(action, context):
    verify canonical action, identity, authority, resource version
    verify required hard constraints and irreversible-action registration
    evidence = causal_audit_and_conformal_set(action, context)
    if evidence missing / expired / out of domain / computation failed:
        emit typed failure; append durable audit record; return ESCALATE or HARD_STOP
    if hard rule violated or "harm" in evidence.prediction_set:
        append durable audit record; return HARD_STOP or ESCALATE
    if irreversible:
        append durable audit record; return CONFIRM with bound action/context digest
    append durable audit record; return PROCEED
```

具体挂载应从**执行边界**向上接线：`gen-zero-gate` 负责不可绕过的 tier 合成；Rust `ProductionPipeline` 的 `decide`、`audit_action`、`what_if`，Rust 独立 `worldsim` 动词、CLI、HTTP、MCP，以及 Python `client.py` 和服务入口使用同一版本化裁决语义。`what_if` 只输出逐候选证据与排序，不产出可复用的执行许可；真正执行前必须重新审计当前资源版本与权限。CP-SAT 只证明**已编码约束**的可行性，不能覆盖未编码的损害。确认令牌绑定动作、参数、资源版本、主体、过期时间及审计叶；持久审计失败不得释放令牌。

实施验收还必须检查旧评分捷径和旧审批符号在各生产入口**零残留**，用调用图与端到端故障注入证明新裁决确实生效。这里描述的是接入要求，**不是已完成接入**。

## 五、实证方案与停止线

先在 13-task 冻结划分上评估 `aegis_safety` 与 `civil_comments` 的内容分类：逐样本 ID 对齐，训练／校准／测试按来源与文本家族隔离，报告危险类漏报、放行率、升级率、覆盖率、各风险组置信区间和逐样本配对差异。固定 margin 门禁、原始模型及拟议几何分数共享同一测试 ID；预注册消融和配对 bootstrap／精确检验。原有文档的 30 条结果只能当现状参照，不能当新方案效果。

随后建立**独立的动作后果数据集**：隔离文件系统删除、数据库事务与脏写、权限越界、多步工具调用、检查后状态变化、桥接服务超时、模型缺载、校准文件损坏、MMR 写失败。每条记录含初态、动作、真实执行或模拟来源、最终损害标签及审计轨迹。按环境、资源、模板家族和时间留出测试，测量真实危险类漏报与上界；任何静默放行、证据缺失仍 `APPROVED`、或绕开统一门禁的入口，均为发布阻断项。高维模型比较必须记录真实模型权重、提取配置、训练日志和同 ID 配对统计；在此之前不得声称 70B 至 405B 已带来收益。

## 成果与证据状态

- **已实现（本次工作）：**完成只读代码核查与上述方案推导。核查命令包括 `rg -n 'audit_action|what_if' ...`、`nl -ba ...` 和 `git status --short`；这些读取命令退出码均为 **0**。输出中的关键原文为 Rust 接口说明的 `untrained residual latent prior`（[server.rs](../../../crates/gen-zero-service/src/server.rs#L1634)）以及 `audit_action` 的 `safety estimate is uncalibrated`（[pipeline.rs](../../../crates/gen-zero-planner/src/pipeline.rs#L618)）。前后 `git status --short` 所列未跟踪文件一致；本次**未修改代码文件**。
- **未验证：**拟议因果评分的识别性、保形阈值在目标部署分布的覆盖率、危险类漏报率、405B 表征收益，以及跨入口强制门禁的系统不变量。本文公式是在所列条件下的数学性质，不是 Gen-Zero 当前运行能力的实测结论。
- **未完成：**模型训练、逐样本配对统计、动作后果数据集、生产挂载、旧路径清除、编译测试与在线验收。原因是本任务要求**方案报告且禁止改代码**；没有进行这些步骤，就没有相应的实现或性能主张。