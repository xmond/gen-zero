本次目标未完成：新头在唯一一次最终留出评测中得到 29/38，低于基线的
31/38，更未达到要求的 32/38。不得将本实验描述为超过或压制 CLM-8B。

| 留出指标 | 实际基线 best_head.pt | 新训练头 |
|---|---:|---:|
| BoN=1 | 73.6842% | 73.6842% |
| BoN=2 | 75.4386% | 72.8070% |
| BoN=4 | 81.5789% (31/38) | 76.3158% (29/38) |
| 轨迹级 AUC | 0.519595 | 0.413964 |
| 轨迹级 Spearman | 0.029954 | -0.131521 |

**已实现**：训练分区 75 个任务、30,372 步、298 条轨迹的特征提取与排序头
训练；任务分组交叉验证选型；最终权重冻结；38 个任务、151 条轨迹的独立
进程评测；完整候选分数、选错任务和权重/数据/源文件 SHA-256 报告。
6 项聚焦测试通过，原版 CLM `best_of_n` 独立核验通过。

**未验证**：因果识别能力、新数据集及线上效果、统计显著的泛化优势。
这些特征是观测几何关联，不能证明因果。30,372 步也不能当作同等数量的
独立监督样本。全局 AUC 与任务内 BoN 衡量的对象不同；仅凭它们的差异不能
证明某一个具体过拟合机制。

**未完成**：32/38 的验收目标、AUC/Spearman 改善，以及几何分支带来的
提升。训练交叉验证选中的最终配置为 semantic kernel、alpha=0.001、
outcome_weight=0.1；几何分支没有胜出。没有微调 8B 主干或原基线 MLP。

与基线逐任务对比：修正 2 个、退步 4 个，净下降 2 个任务。任务配对
bootstrap 的 BoN=4 差值 95% 区间为 [-18.42, +7.89] 个百分点。
这是小样本评测，不能据此宣称任何统计上的压制。

新头的 5 个可避免选择错误（都存在成功候选）：

- anko-typed-variable-bindings
- claude-code-by-agents-recursive-delegation
- prometheus-transactional-reload-status
- tengo-callable-instance-isolation
- textual-richlog-follow-state

另有 4 个任务所有候选均失败，任何纯重排都不能解决：

- bandit-structured-nosec-directives
- kcp-go-multiplexed-kcp-streams
- meriyah-explicit-resource-declarations
- obsidian-linter-link-format-conversion

基线的 7 个失败中只有 3 个是可避免选择错误；把全部 7 个都称为评分头
选错不准确。完整基线/新头失败候选、分数与 reward 在
`../deepswe_prm_enhanced_results.json` 的 `wrong_tasks` 和 `per_task` 中。

证据入口：

- 实现：`benchmarks/eval_deepswe_enhanced_prm.py:59` 特征、`:139` 排序损失、
  `:228` 训练集选型、`:296` 真实基线加载、`:322` 冻结模型评测。
- 最终权重：`benchmarks/artifacts/deepswe_prm/enhanced_rank_head.pt`。
  SHA-256：`8c039e3fe48cc671797af3cfc0b8c75a6482a4c38b88760ce6c0e26ed4d51a17`。
- 原始命令和退出码：`handoff.json`；完整日志：同目录对应 `.txt` 文件。
- 训练、评测、6 项测试、独立核验原始退出码均为 **0**。
- 目标准入验收原始退出码为 **1**，见 `acceptance.json` 和 `acceptance.txt`。
- 复现命令及边界：`REPRODUCE.md`；运行环境：`environment.json`。
- 本地权重/特征受仓库原有 ignore 规则排除，不会随普通 git 提交自动携带。

留出结果揭晓后没有再选模型或调整超参数。继续依据这批失败任务试到
32/38 会污染留出集，不能作为本次要求的合规成功证据。
