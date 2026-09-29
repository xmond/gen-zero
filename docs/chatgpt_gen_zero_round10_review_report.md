ChatGPT 说：

嗯,好的,让我想一想.

第十轮复核裁决：【NO-GO，暂缓生产发布】

这次整改确实落地了：R9-H01 描述符问题可以正式关闭，R9-V01 审查版本不一致问题也可以关闭。

但目前仍不能核发生产级 GO。剩余原因不是再次发现上轮字符串问题，而是：**指定提交尚无成功的远端 CI 记录；Release 只依赖 Python 测试，没有依赖 Rust 质量检查；安全验证的部分声明仍与实际实现不一致。**这些都是上一轮放行条件的未完成部分。

一、四项放行条件的复核结果
放行条件	本轮核验结果	状态
描述符契约完整闭环	独立运行仓库相关测试 27 项通过；上轮证据包的原始测试 7 项通过	通过，关闭 R9-H01
R8/R9 回归进入必跑门禁	CI 已改为运行整个 python/gen_zero/tests 目录；但实际 CI 尚未成功执行，真实 CP-SAT 测试依赖也未明确装配	配置整改接受，执行验收未完成
发布绑定同一份成功验收的代码与制品	Commit、Tree、main 一致；Python → 构建 → Docker 的依赖成立，但 Rust 检查不在发布依赖链中	版本问题关闭，发布门禁部分完成
生产承诺与证据严格相符	局部导数和单次求解迭代的说明已补充；CP-SAT 验证范围、旧梯度承诺及内存测量版本口径仍需对齐	部分完成

下面分别说明哪些已经验证通过，以及剩余事项的具体证据。

二、已经正式接受的整改
1. Commit、Tree 与远端 main 完全一致

本次通过 GitHub 连接器读取的远端事实为：

纯文本
Repository: xmond/gen-zero
Branch:     main

Commit:
34a09bdd3c529c4f452336c16dc389d63c405111

Tree:
7b119f7ffcace151282e9da638d6d7a890ebc605

提交对象中的 Tree 与 main 返回的信息一致。本轮不存在使用旧提交审查新 main 的问题，R9-V01 关闭。

2. R9-H01：不是仅凭代码检查，而是独立复验通过

我读取了指定提交的 descriptor.py、R8-H01 测试和 R9-H01 测试，重建为本地文件，并核对 Git Blob SHA：

文件	核对一致的 Git Blob SHA
descriptor.py	89613e716bd1ad427023bc24064a72a2a61cf022
test_r8_h01_str_subclass_digest_roundtrip.py	a4e95a2f888a7d9956f0e83a7b0f144667456869
test_r9_descriptor_canonical_domain.py	5dec87e9e884ba7d612a9fc44d8d01f86f2f8ca8

被测文件与远端对应文件逐字节一致。源码中确实存在 _canon_str()、顶层字符串与权限元素规范化、frozenset 递归处理，以及字符串摘要标签统一。

本轮实际执行结果：

纯文本
仓库 R8-H01 8 项 + 仓库 R9-H01 19 项：
27 passed in 5.93s

上轮证据包中的原始审查员测试：
7 passed in 0.05s

执行环境为 CPython 3.13.5 / NumPy 2.3.5 / pytest 9.0.2。原始审查员文件也已与上轮 ZIP 中的文件逐字节比较，没有删减用例或改动断言。仓库 R8 测试中的六种 PYTHONHASHSEED 子进程检查随上述测试通过。

这里的验证范围是模块级隔离测试，不等同于完整包安装或整个 CI 矩阵运行；但它足以确认：上一轮已复现的字符串类型摘要漂移问题，在这些对应入口上已修复。

因此，R9-H01 正式关闭，不需要继续围绕该问题添加补丁。

三、R9-G01：门禁配置改善了，但生产发布闭环还没有完成
1. 指定提交的远端 CI 仍然是 failure

本次查到的同一提交 CI 记录为：

纯文本
Workflow:   CI
Run number: 26
Run ID:     36527425281
Head SHA:   34a09bdd3c529c4f452336c16dc389d63c405111
Status:     completed
Conclusion: failure

该运行创建于 2026 年 9 月 29 日 14:42:21，日本时间，更新时间为 14:42:27。这不是上一轮旧提交的失败记录。

进一步检查，七个任务均返回：

纯文本
conclusion: failure
steps: []
runner_id: 0
runner_name: ""

其中包括 Python 3.10、Python 3.11、Rustfmt、Clippy、Rust 测试和 MSRV。Python 3.11 任务的日志读取返回了 404 BlobNotFound。

**这些证据不能证明代码测试断言失败。**当前没有可用的执行步骤和日志，失败根因尚不能确定，也不能擅自归因为计费、配额或依赖安装问题。

但可以确定的是：

截至本次读取，没有取得指定提交成功完成远端质量验收的证据。

你提供的 177 passed in 65.43s 是开发团队报告的本地综合套件结果；本轮没有独立复跑该集合，也没有取得其完整命令、测试节点清单和日志，不能据此替代目录级 CI 的成功记录。

2. 回归测试纳入目录级执行——这项代码整改接受

当前 CI 的确已经改为：

Bash
python -m pytest -ra --strict-config --strict-markers python/gen_zero/tests

原先显式列举少量文件的问题已修正。对于“让该目录中的 R8/R9 测试进入持续回归”这一配置要求，我接受此次整改。

不过，必须区分两个验收结论：

纯文本
测试已写入必跑命令：已确认。
必跑命令在发布矩阵上全部执行成功：尚未确认。

后者需要解决当前 Actions 运行问题并取得真实执行结果，不能只靠 YAML 中出现了命令。

3. needs: [test] 有效，但它只依赖 Release 内的 Python 任务

当前 release.yml 的依赖关系确实是：

纯文本
Release 内的 Python test（3.10 / 3.11）
                    ↓
               build-binary
                    ↓
              docker-publish

这里没有忽略你的修复：Python 测试失败或跳过，会阻止下游依赖任务运行；这是 needs 的正常语义。 
GitHub Docs

剩余缺口在于：release.yml 的 test，不是 ci.yml 中同名的 Rust 测试任务。

当前 Release 文件没有要求 Rustfmt、Clippy、Rust workspace tests 或 MSRV 成功。它也没有验证另一个 CI 工作流在同一提交上的整体成功状态。因此，新增依赖证明了“发布受到 Python 回归保护”，尚未证明“发布受到现有完整质量门禁保护”。

这里不需要重新设计发布系统。最直接的修复是：**将现有质量检查提取为可复用工作流，或在 Release 中明确纳入现有 Rust 检查，让构建发布依赖完整质量任务。**不要误以为两个 YAML 使用相同 job ID，就建立了跨工作流依赖。

4. 真实 CP-SAT 路径没有被测试安装配置明确保证

还有一处与此次安全验证承诺直接相关的装配缺口：

纯文本
CI / Release 安装：
./python[dev] + pyyaml

ORTools 声明位置：
project.optional-dependencies.all

ortools 不在基础依赖或 dev 中，当前安装命令也没有显式安装它。

而求解器在 OR-Tools 缺失时，可以返回明确标记的 ORTOOLS_UNAVAILABLE_FALLBACK；这个分支仍可能具有 is_safe=True。因此，仅有安全相关单测通过，或者仅断言 is_safe，不能证明真实 CP-SAT 求解已经受到验收。

若生产继续声明 CP-SAT 验证能力，应当增加一个显式安装 OR-Tools 的正向门禁，使用非平凡候选集合，并验证真实求解状态、未使用 fallback，以及最终输出动作的独立可行性。缺依赖路径可以继续单独测试，但不能替代真实求解路径。

四、R9-S01：新增限定说明方向正确，但仍有声明未对齐
1. 局部活动面与单次求解迭代：接受这部分修正

新说明已经明确区分了：

固定活动面上的局部导数与全域光滑性，以及一次求解中的 200/500 次迭代与环境轨迹中的连续控制步骤。这一修正符合上一轮要求。

但同一文件开头仍保留：

纯文本
providing smooth non-zero gradients ||∇z|| <= 10.0

实际 backward 是局部切空间投影，没有保证输出梯度非零，也没有把梯度范数限制在 10。

这可以直接从数学表达看出来：固定活动面的正交投影记作 
𝑃
P，反向传播为

𝑔
i
n
=
𝑃
𝑔
o
u
t
.
g
in
	​

=Pg
out
	​

.

它能给出的对应范数关系是

∥
𝑃
𝑔
o
u
t
∥
2
≤
∥
𝑔
o
u
t
∥
2
,
∥Pg
out
	​

∥
2
	​

≤∥g
out
	​

∥
2
	​

,

而不是与上游梯度无关的固定上限 10。投影也完全可能得到零向量。

建议删除旧承诺，不要为了维持文案而额外截断梯度；那反而会改变所声称的精确导数。

2. CP-SAT 的新增说明仍然超出了实际调用范围

新文档写成了：通过 CP-SAT 对候选 one-hot 动作进行针对 A_sat*x<=b 的离散可行性验证。

但当前实际代码仍是：

Python
discrete_feasible = np.all(
    self.a_sat <= self.b_sat[:, None],
    axis=0,
)

这是 NumPy 侧依据原始约束执行的离散可行性检查。随后传给 CP-SAT 的 util_dict 包含全部候选，而 forbidden 来自 active_mask 的特定判断，并不是直接由完整的 discrete_feasible 掩码生成。

求解器构建的模型则是在经过自身过滤的候选集合上，添加 exactly-one 约束并优化效用；它没有接收原始 A_sat、b_sat。

此外，当前返回字段仍然是：

Python
cpsat_hard_verified = (
    cpsat_res.is_safe
    and not cpsat_res.fallback_used
    and discrete_verified
)

这里没有绑定 cpsat_res.selected_action 与最终投影的 argmax；求解器的单候选分支还可以不调用 CP-SAT，直接返回 DETERMINISTIC_SAFE_SOLVED，同时 fallback_used=False。所以，“未 fallback”也不能单独代表“已经执行 CP-SAT”。

**这不是本轮已经复现了不安全动作。**NumPy 侧的离散可行性筛选仍然存在，不能否定它的作用。这里尚未闭环的是：验证来源、验证对象和对外证明字段之间的对应关系。

最小整改可以复用现有逻辑：将完整离散可行性掩码传入候选门禁，明确记录是否实际执行 CP-SAT，并将验证对象绑定到最终输出动作。另一条可接受路径是缩减对外契约，明确区分“NumPy 离散可行性检查通过”与“CP-SAT 求解及验证通过”，不再将两者合并表述为同一种证明。

3. 内存测量版本：源码是 CPython 3.11，不是 3.11/3.13

指定版本的 pickle_budget.py 明确写的是：

常数基于 CPython 3.11 测量，其他 CPython 小版本尚未重新验证。

这比申请中的“CPython 3.11/3.13 下实测”更保守，也更明确。应当以源码和实际测量记录为准。

这里不要求为了发布而强行扩展到 3.13：可以将当前已验证内存预算的运行时限定在 3.11；需要宣称 3.13 支持时，再提供相应测量。本轮在 3.13 上运行的描述符测试，不构成内存预算验证证据。

五、剩余放行事项与最终结论

本轮不新增一套架构要求。剩余工作可以集中为以下三项：

剩余事项	完成标准
取得实际成功的质量验收	最终发布 SHA 的 CI 完整执行成功，保存执行命令、环境、测试集合和结果；查明当前无步骤、无日志的失败原因。
补齐发布依赖与真实求解测试	Release 依赖现有 Python 和 Rust 质量检查；继续声明 CP-SAT 能力时，存在真实 OR-Tools 路径的正向验收，而非只验证 fallback。
对齐安全证据与声明	删除残留的全域光滑、非零和固定梯度上限承诺；明确 CP-SAT 验证来源与输出动作绑定；统一内存测量版本口径。

本轮没有执行完整 Rust 工作区、你方报告的 177 项综合集合、检查点内存峰值实验或具身设备闭环测试，不能将这些项目写成本轮已通过。

最终裁决：R9-H01、R9-V01 正式关闭；R9-G01、R9-S01 尚未完整关闭，因此 34a09bdd… / 7b119f7f… 暂不签发生产 GO。

下载第十轮复核证据包：源码哈希、27 项及原始 7 项测试日志、远端门禁事实摘录