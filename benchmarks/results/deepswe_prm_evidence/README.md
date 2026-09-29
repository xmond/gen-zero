# DeepSWE offline PRM reproduction evidence

已实现：完整读取 44,409 行、113 任务、449 轨迹；校验必需列、非空元数据、二元且轨迹内一致的 reward、唯一 step_idx、有限且维度正确的 embedding、有限权重和投影；strict 加载评分头。评分头 SHA256 与本地发布 README 一致。每一步均重新运行 supplied best_head.pt，未读取已有 scored.pkl 或用 parquet prm_score 代替推理。

| 指标 | 全部 113 任务 | 留出 38 任务 |
|---|---:|---:|
| 离线随机单轨迹 Pass@1（任务宏平均） | 73.00885% | 73.68421% |
| 严格 BoN=2 期望解决率 | 79.05605% | 75.43860% |
| BoN=min(4, available) | 85.84071% (97/113) | 81.57895% (31/38) |
| 严格 BoN=4 全任务解决率 | 不可计算（3 任务不足） | 不可计算（1 任务不足） |
| 严格 BoN=8 | 不可计算 | 不可计算 |
| 轨迹 ROC AUC | 0.794104 | 0.519595 |
| 轨迹 Spearman rho | 0.453211 | 0.029954 |
| 步级 ROC AUC | 0.683208 | 0.491701 |
| 步级 Spearman rho | 0.276906 | -0.012513 |

N=8 的 capped 数字等于 N=4，不能作为真实 BoN=8。110 任务各有 4 轨迹，3 任务各有 3 轨迹。BoN 采用不放回均匀子集的精确期望，并列最高分均匀处理。轨迹评分为按 step_idx 排序后最后 12 个可用步骤的余弦分数均值。步骤标签为轨迹终局标签的重复，步级指标偏重长轨迹；常规相关性 p 值不处理任务内相关，0.0 是数值下溢，不代表真实概率为零。

本地发布说明 `/tmp/clmrepro/heads/README.md:23` 的 31/38 复现成功，相对 81.6% 差 -0.02105 个百分点，属于舍入差异。此发布协议也保留不足 4 条的任务。全量 85.84% 比 81.6% 高 4.24071 个百分点，但包含 75 个训练分区任务，不是同口径泛化提升。留出轨迹 AUC 接近 0.5、Spearman 接近 0，而训练分区 AUC 为 0.934564：不能用混合集指标掩盖留出集的弱相关性。

未验证：parquet 独立上游校验和/真实性、reward 的环境重执行、原始轨迹是否完整无截断。已有本地 SHA256 仅保证本次输入可定位。结果是已存储 Opus 5 轨迹上的 CLM head 离线选优，不是 B2 新生成轨迹或 full environment benchmark。

未完成（输入不支持）：真正 greedy 解码准确率、严格 N=4 覆盖 113 任务、严格 N=8。需要明确 greedy 轨迹以及每任务至少 8 条真实候选才能完成；没有补造轨迹、重复采样冒充新候选或静默降级。

## Commands and raw exit codes

Working directory: `/ebs/pj/gen-zero-worktree/eval-b2-deepswe`

```bash
python benchmarks/eval_deepswe_prm.py > benchmarks/results/deepswe_prm_evidence/run.log 2>&1
rc=$?
printf '%s\n' "$rc" > benchmarks/results/deepswe_prm_evidence/exit_code.txt
exit "$rc"
```

Raw exit code: `0`, recorded in `exit_code.txt:1`. Full actual inference log: `run.log`; completion of 44,409 rows at `run.log:46`; explicit unavailable metrics at `run.log:47`; output at `run.log:425`.

```bash
python benchmarks/results/deepswe_prm_evidence/verify.py > benchmarks/results/deepswe_prm_evidence/verification.log 2>&1
rc=$?
printf '%s\n' "$rc" > benchmarks/results/deepswe_prm_evidence/verification_exit_code.txt
exit "$rc"
```

Raw exit code: `0`. `verification.log:1`: six sklearn ROC AUC comparisons and 1,024 exhaustive subset/tie cases passed. Every real candidate group's subset calculation also checked by independent enumeration during evaluation. `git -c core.fsmonitor=false diff --check` exited 0.

Implementation: `benchmarks/eval_deepswe_prm.py:60` validation, `:83` fresh inference, `:111` task aggregation and exact selection. Results: `benchmarks/results/deepswe_prm_eval_results.json`. Recomputed raw scores: `step_scores.parquet`; trajectory audit: `trajectory_scores.csv`. Input, reference code, evaluator and score artifact hashes are in results JSON. Reproduction requires the explicitly recorded local reference repository and Python dependencies.

No old module was replaced; no unrelated tracked file was modified. No subagents/reviewers were launched.
