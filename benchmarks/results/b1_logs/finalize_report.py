import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = root / 'benchmarks/results'
p = out / 'decision_foundation_eval_results.json'
d = json.loads(p.read_text())
d['commands'] = json.loads((out/'b1_logs/commands.json').read_text()) + [json.loads((out/'b1_logs/audit_command.json').read_text()), json.loads((out/'b1_logs/import_command.json').read_text())]
d['findings'] = [
 {'source':'python/gen_zero/evaluate/decision_foundation_benchmark.py:61','finding':'Calls simulated regex/hash scoring; not actual model inference. No pytest tests or executable main.'},
 {'source':'python/gen_zero/train/qwen_post_trainer.py:140','finding':'Hard-coded regex/action associations plus hash logits, not learned LoRA weights.'},
 {'source':'python/gen_zero/evaluate/decision_foundation_benchmark.py:176','finding':'ECE report PASS is unconditional; report conclusion at line 194 claims delivery and hardware SLA without runtime evidence.'},
 {'source':'python/gen_zero/planner/engines/mcts_engine.py:392','finding':'search performs fixed-action discounted rollouts; eval_fn and simulations are unused; timing is not evidence of PUCT tree search.'},
 {'source':'python/gen_zero/scripts/run_12_planners_benchmark.py:113','finding':'Reflex success only tests action membership; most other cases test returned action/output presence. Not solved-task accuracy.'},
 {'source':'python/gen_zero/planner/engines/astar_engine.py:193','finding':'Bidirectional invocation dispatches to forward search with absent goal predicate; runtime TypeError.'},
 {'source':'python/gen_zero/scripts/run_12_planners_benchmark.py:188','finding':'CEM passes initial_state but current plan requires state; runtime TypeError.'},
]
sources = [root / x for x in sorted(set(f['source'].rsplit(':',1)[0] for f in d['findings']))]
d['source_sha256'] = {str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
d['evidence_sha256'] = {str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((out/'b1_logs').glob('*.log'))}
d['limitations'] = ['Single local run, default settings, no warmup or controlled RNG seed; timing cannot establish production SLA.', '50 repeated fixtures per completed planner, not 50 independent benchmark problems.', 'No new model checkpoint or inference backend provisioned. No production code replaced or removed.', 'Scores, actual solution rates, search-step counts and per-step latency are not recorded by supplied scripts; left null, never inferred from action existence.']
p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
lines = ['# B1 实测与有效性审计', '', f"基线 HEAD：`{d['revision']}`；时间：`{d['utc_timestamp']}`。", '', '## 已实现', '', '执行两个指定入口，并逐项调用原有 12 个测试方法（每项请求 50 次），保留完整 stdout/stderr、原始退出码、异常堆栈和逐次耗时。没有修改生产代码、修补接口或替换旧模块。诊断在单项异常后显式记录并继续，最终退出 1。', '', '### 命令与退出码', '', '| 命令 | 退出码 | 日志 |', '|---|---:|---|']
for c in d['commands']:
 cmd = ('PYTHONPATH=python ' if c.get('environment_override') else '')+' '.join(c['command'])
 lines.append(f"| `{cmd}` | {c['exit_code']} | [{Path(c['log']).name}](b1_logs/{Path(c['log']).name}) |")
lines += ['', 'pytest：0 个测试，退出 5，测试通过率未定义。Foundation 直接运行导入失败；显式设置 PYTHONPATH 后退出 0，但仅导入类，没有运行评测，不能算通过。原始 12 Planners 在第 3 项异常退出，未输出汇总。', '', '### 12 Planners 诊断实测', '', '10/12 项完成（83.33%），2/12 项异常；这是诊断完成率，不是解决率。完成项各有 50 次原始耗时，存于 JSON 的 legacy_metrics；成功率仅沿用原脚本谓词。所有延迟是整次调用耗时，不是单个搜索步骤。', '', '| 项目 | 原脚本成功率 | Mean ms | P95 ms | P99 ms | 状态 |', '|---|---:|---:|---:|---:|---|']
for r in d['planners']:
 if r['status']=='error':
  lines.append(f"| {r['method']} | 未取得 | — | — | — | `{r['error_type']}: {r['error']}` |")
 else:
  for name,m in r['legacy_metrics'].items():
   lines.append(f"| {name} | {m['success_rate']*100:.1f}% | {m['mean_ms']} | {m['p95_ms']} | {m['p99_ms']} | 仅诊断完成 |")
lines += ['', '### 致命有效性问题', '', 'Foundation 生成 495 条、11 域模板样本；真实模型评测 0 条。配置名称 Qwen/Qwen3.5-9B 不代表加载了模型。为避免将模式匹配冒充推理，没有执行模拟打分并将其包装成模型指标。Accuracy、Latency、Entropy distribution、Tier 分布均为 null。', '', '默认 Planner 客户端未加载 checkpoint；日志记录随机初始化权重降级及 localhost:8090 仲裁连接拒绝。因此 Reflex 的 100% 动作成员检查不能证明模型正确性。MCTS/CEM fixture 的动作奖励恒为 1，缺少区分正确决策的质量标签。']
for f in d['findings']:
 lines += ['', f"- `{f['source']}`：{f['finding']}"]
lines += ['', '## 未验证', '', '真实模型推理准确率、概率熵和 Tier 分布；生产单步延迟与硬件 SLA；规划得分、解决率和搜索步数；跨随机种子稳定性。原入口未提供这些可接受的测量，未用占位数字填充。', '', '## 未完成', '', '真实 Decision Foundation 多模型基准、12 项全部成功执行及目标指标验收仍未完成。阻塞项是模拟打分入口、缺少已加载模型、双向搜索接口错配、CEM 参数错配和指标定义/采集缺失。本任务保留失败证据，没有将修复后的新问题或替代算法混为原基准。', '', '## 复现', '', '在仓库根目录执行 `python benchmarks/results/b1_logs/audit_runner.py`（预期退出 1）；随后执行 `python benchmarks/results/b1_logs/finalize_report.py` 更新汇总。命令 1–3 的历史原始日志不被重写。单次默认随机状态、无预热，耗时仅描述本机本次运行。', '', '结构化结果：[decision_foundation_eval_results.json](decision_foundation_eval_results.json)。其中包含源码/日志 SHA-256、495 条输入的路径与 SHA-256、各项完整异常和逐调用延迟。']
(out/'decision_foundation_eval_report.md').write_text('\n'.join(lines)+'\n')
