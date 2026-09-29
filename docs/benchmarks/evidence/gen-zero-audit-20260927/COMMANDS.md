# 本次调查命令与验证

基线：`/ebs/pj/gen-zero`，HEAD `6dfd6099591f68a8bba8bbb8ae3b2a6c302f7e2c`。

所有日志来自本次真实执行。`.txt` 是 stdout/stderr 原文副本，不截断失败。JSON/exit 文件记录原始退出码；旧 JSON 中 `/tmp/gen-zero-audit-20260927` 是首次生成位置。未运行真实外部模型生成或重新部署 dev。

| 命令 | 结果 | 日志 |
|---|---|---|
| `cargo test -p gen-zero-gate --lib --locked` | 退出0，43通过 | `gate-rust.txt` |
| `cargo test -p gen-zero-planner --locked` | 退出0，69通过，0 doc tests | `planner-rust.txt` |
| `PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q python/gen_zero/tests/test_policy_evidence_gate.py python/gen_zero/tests/test_issue_22_semantic_review_gate.py` | 退出0，43通过 | `gate-python.txt` |
| `PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q benchmarks/tests/test_gen_zero_tb_adapter.py benchmarks/tests/test_deepswe_adapter.py` | 退出2，缺adapter模块搜索路径 | `adapters-python.txt` |
| `PYTHONPATH=python:benchmarks PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q benchmarks/tests/test_gen_zero_tb_adapter.py benchmarks/tests/test_deepswe_adapter.py` | 退出2，当前Python缺Harbor | `adapters-python-corrected.txt` |
| `PYTHONPATH=python:benchmarks PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q benchmarks/tests/test_deepswe_adapter.py` | 退出0，9通过 | `deepswe-only.txt` |
| `python3 benchmarks/tests/verify_deepswe_prm_report.py --report benchmarks/results/deepswe_prm_enhanced_results.json --reference-repo /tmp/clmrepro/repo` | 退出1，缺增强head权重；未跳过hash检查 | `prm-report-verify.txt` |
| `PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 /tmp/gen-zero-audit-20260927/gate_probe.py` | 退出0，打印3个真实门禁放行反例 | `gate-counterexample.txt` |

gate_probe.py 为本次边界调查脚本，不加载/伪造模型、不执行被评分动作。脚本同目录保留，复现时可替换为当前绝对路径。第一次交互探针误用不存在的 `.evaluate` 方法后，已改成真实 `.evaluate_policy`，本文件的脚本及日志为正确调用。反例不是生产任务成功/失败率。

只读定位命令（输出在会话工具记录）：

```bash
pwd
git status --short
git rev-parse HEAD
rg --files -g '*clm*' -g '*adapter*' -g '*choice*' -g '*nanocore*' -g '*policy*' -g '*worldmodel*' -g '*mcts*'
rg -n 'PolicyGate|LinearConstraint|evaluate_with|Tier3HardStop' crates/gen-zero-service/src python/gen_zero/service benchmarks/gen_zero*
rg -n 'add_constraint\(|register_confirm_action\(|PolicyGate::default' crates/gen-zero-service/src --glob '*.rs'
nl -ba crates/gen-zero-gate/src/constraint.rs
nl -ba crates/gen-zero-gate/src/policy.rs
nl -ba python/gen_zero/gate/policy_gate.py
nl -ba python/gen_zero/runtime/loop_state_machine.py
nl -ba benchmarks/results/deepswe_prm_evidence/RESULT.md
```

外部来源命令：

```bash
firecrawl --status
firecrawl search 'CLM-8B DeepSWE 81.6 Terminal Bench 87.6' --limit 3 --scrape -o .firecrawl/gen-zero-audit-clm.json --json
```

分析只引用官方模型卡；其全文副本为 `clm-official-model-card.md`。未据搜索中的二手文章或其他排行榜推断本项目收益。

追加命令（未覆盖先前日志）：

```bash
cargo test -p gen-zero-service --lib imagine::tests --locked
PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q python/gen_zero/tests/test_world_model_simulation_endpoints.py -k 'simulate_neural_vector_runs_full_horizon or simulate_neural_vector_truncates_at_trap or what_if_neural_flags_trap_and_picks_safe or audit_neural_safe_approved_and_trap_rejected'
python3 /tmp/gen-zero-audit-20260927/parameter_probe.py
git diff 6dfd609 701d800 -- crates/gen-zero-planner/src/engine.rs crates/gen-zero-gate/src/constraint.rs crates/gen-zero-gate/src/dual_track.rs crates/gen-zero-gate/src/sheaf_gate.rs crates/gen-zero-planner/src/pipeline.rs python/gen_zero/client.py
cargo test -p gen-zero-gate -p gen-zero-planner --locked
```

日志依次为 `semantic-mcts-rust.txt`（7通过）、`worldmodel-python.txt`（4通过，50 deselected）、`parameters.txt`、`concurrent-changes.diff`、`current-head-rust.txt`（gate51+planner78通过）。退出码均为0。末条在并行合并后执行；完成后HEAD核对仍为 `701d800a5d3c55db6dcdb002b7bda5608a6bba11`。parameter_probe.py同目录保存。
