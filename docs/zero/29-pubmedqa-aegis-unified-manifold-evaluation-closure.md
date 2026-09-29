# Spec 29：PubMedQA 与 Aegis 2.0 评测结论勘误

- 日期：2026-09-26
- 原始评测：[统一评测报告](../../benchmarks/results/unified_manifold_pareto_eval.md)与[机器可读结果](../../benchmarks/results/unified_manifold_pareto_eval.json)
- 范围：独立 Python 评测套件；报告所写 `evaluate_manifold_pareto_ensemble.py` 在当前工作树缺失、未被 git 追踪，故本次无法据此复跑。`bb38d1e` 不能作为该套件已合入主干或已部署的证明。

## 已观察结果与统计边界

| 任务与口径 | 选择结果 | 同口径对照 | 可得结论 |
| --- | ---: | ---: | --- |
| PubMedQA，测试 250 题 | 融合 `fuse0.75+bbp|raw` 78.40%（196/250） | Qwen 单模 77.20%（193/250）；LLaMA 单模 77.60%（194/250） | 比 Qwen 多答对 3 题，比 LLaMA 多答对 2 题；均为描述性差值 |
| Aegis Track A，全量 250 题 | OOF 选出的 `llama+bbp|raw` 81.60%（204/250） | Qwen 单模 81.20%；外部 Jev 80.40%、Nimble 81.20% | 仅 Track A 的分母允许与全量外部数字并列；协议是否完全一致仍需核验 |
| Aegis Track B，剔除 25 条 `Needs Caution` 后的 225 题 | 历史报告称 84.44%（对应 190/225；本仓库未找到逐行或结果文件佐证） | 无同子集外部基线 | **绝不可与 Jev/Nimble 的全量 250 题结果做跨口径比较** |

PubMedQA 的 78.40% 是本次 OOF 选型后的测试观察值，不是已确立的 SOTA。原始报告的峰值宏平均准确率目标检查（≥82.5）为 **False**；这不是 PubMedQA 单任务显著性检验。报告给出的配对 bootstrap 95% CI `[-0.20, +1.60]` pp 对应**两个任务的宏平均** `peak_minus_peak_single`（点估计 +0.60 pp，按任务内测试行重采样 5,000 次），**不是 PubMedQA 单任务区间**。该区间包含 0；本仓库尚无经核验的 PubMedQA 单任务逐题配对区间，因此不能声称其严格统计显著超越单模或外部系统。外部 Jev/Nimble 缺少本次相同逐题预测，不能计算配对显著性。

Aegis Track A 的 `concat+lda` 固定对照在同一测试集得到 82.80%。这是**后验对照观察值**，不是 OOF 选出的正式结果，不应列为“全量测试最高点”或 SOTA。历史报告所称 Track B 84.44% 是条件子集准确率，当前缺少原始结果佐证；撤回“超越 Jev +4.04 pp”等跨分母断言，也不将剔除的样本定性为已证实的噪声。

## 方法名与适用范围

- 原称“Certified Robust”的轨道实际按 **OOF 平衡准确率与 F1、1-SE 经验筛选**。这不是形式化鲁棒性证明；其两任务宏平均测试平衡准确率为 70.32%，F1 为 69.74%，对应目标检查均为 False。
- Aegis Margin Gate 使用经验固定阈值 `θ=1.0`，在 `|m|<θ` 时拒答或升级。它**没有** split-conformal 校准集分位数保证，不得称为“Conformal guarantee”。与之分开的预测集合实验对 PubMedQA 报告边际覆盖率 92.4%、最低类别覆盖率 88.8%；这些覆盖率不能转移为 Margin Gate 的保证。
- 原始 Aegis 门控统计若报告高危样本 0 漏放，只限于该批测试行和相应门控视图。`oracle_high_risk` 使用金标类别，仅是诊断视图；不能冒充可部署检测器或线上零漏放。
- `aegis_dual_track.py` 被独立评测入口调用；这不证明 Rust CLI、HTTP、MCP 或生产调度入口已调用它或 Margin Gate。生产接入与上线状态未在此报告验证。

## 可复核依据

原始运行命令记录在[统一评测报告](../../benchmarks/results/unified_manifold_pareto_eval.md)首部；该历史报告没有可核验的原始退出码日志，本勘误不补造 `EXIT=0`。PubMedQA 与 Track A 数值见该报告的 `Per task`、`Per-task fixed controls`、`Paired bootstrap`、`Target check` 与 `Conformal` 节；Track B 定义及分母与基线可比标志见 [`aegis_dual_track.py`](../../benchmarks/suites/aegis_dual_track.py)。脚本、数据和环境的完整重跑不属于本次文档更正的验证范围。
