# B11 / B12 / B13 / B18 修复与证据

工作树 `/tmp/fleet-wt/b0928-t4-manifold`。未提交、未推送；未调用子代理或 reviewer。

## 已实现与验证

| 项目 | 实际行为与主调用链 | 可核查位置 |
|---|---|---|
| B11 | 连续系统要求最大特征值实部严格小于负 epsilon；离散系统要求谱半径小于 1。拟合结果不做稳定性“修饰”，失败抛错。导出 float32 后再次检查，加载重新检查。 | `python/gen_zero/causal/universal_manifold_extractor.py:395`、`:512`、`:529` |
| B11 生产脚本 | 删除把独立题目行当作时序轨迹、宣称已证明收缩的流程；现在必须提供 `--trajectory` 与 `--dt`。 | `benchmarks/suites/run_universal_extraction_a100.py:71`、`:185` |
| B12 | GCCA CLI 默认 rank 128，显式请求 128 而可用秩不足时拒绝。保存每个 view 的 MAXVAR 岭回归 W、训练均值、尺度；锚点允许 128→128，拒绝 64→128。 | `benchmarks/suites/generalized_cca_manifold_interference.py:155`、`:166`、`:315`、`:472` |
| B12 验证隔离 | 保存源数据摘要与拟合/验证索引，核对导出时的拟合行确实来自本次 fit；锚点 CLI 复用 GCCA 划分，不再随机重切分或在验证集拟合。GCCA 文件、索引、manifest 具有完整性摘要。 | `python/gen_zero/causal/manifold_anchor_distiller.py:84`、`:349` |
| B13 | 锚点 manifest 绑定 source_model、layer、norm、GCCA_map、anchor_basis、core、domain_id。bridge 必须接收独立传入的目标 core manifest；源样本必须携带 values/space；缺失身份、同维异基、错误 domain 均拒绝。 | `python/gen_zero/causal/manifold_anchor_distiller.py:116`、`:126`；`python/gen_zero/causal/nanocore_bridge.py:44` |
| B13 主入口 | `gen_zero.cli anchor --core-manifest` 与 client 加载入口传递契约，client 决策还核对注册 core 的 space_manifest。延迟脚本同步新契约，删除假产物自测入口。 | `python/gen_zero/cli.py:417`；`python/gen_zero/client.py:3190`、`:3268`；`benchmarks/suites/profile_nanocore_latency.py:73` |
| B18 | launcher 必须显式指定路径和 MODEL_PROFILE。full: 100 GB 下限、126 层、输出头；slice: 显式 SLICE_LAYERS/NGL、截断层数、无输出头，按结构校验，不套完整版体积门槛。 | `benchmarks/suites/run_llama405b_extract.bat:12`；`benchmarks/suites/verify_gguf_model.py:45`；`scripts/slice_gguf_layers.py:193` |

运行命令（从工作树根目录执行）：

```bash
bash /tmp/b0928-t4-evidence/run-tests.sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:benchmarks/suites python /tmp/b0928-t4-evidence/real_transform_check.py
```

> **Pruned 2026-09-29.** `run-tests.sh` was removed. It named two test files that no longer exist (`benchmarks/tests/test_run_llama405b_launcher.py`, `benchmarks/tests/test_llama405b_extraction.py`), so it cannot be re-run as shipped. The recorded result (`final-tests.exit`) is historical. `real_transform_check.py` is kept: its imports still resolve, but it reads features from a host-local path.

- 第一条真实退出码 **0**，输出尾部：`187 passed, 11 warnings in 24.52s`。完整命令在 `run-tests.sh`，原始日志在 `final-tests.log`，原始退出码在 `final-tests.exit`。警告来自极端值拒绝测试的数值溢出；没有忽略测试失败或静默回退。
- 第二条真实退出码 **0**，见 `real.log`、`real.exit`。LLaMA-70B/Qwen-72B BoolQ 特征原始配对 ID/标签一致性校验通过；500 行训练、200 行验证。保存/重载后逐样本坐标最大误差均为 **0.0**，逐样本 ID、两份实际输出、误差、fit/eval 索引保存在 `/tmp/b0928-t4-evidence/real-paired-transform-evidence.npz`。`artifacts.json` 给出产物大小、路径、SHA256。
- 真实锚点为 128→128 正交变换，所以能量保留率/余弦相关系数为 1 不是任务能力突破。此次没有计算分类正确率，也不宣称泛化提升。
- `git diff --check`、指定文件 `python -m compileall -q ...`、Rust 测试文件 `rustfmt --check` 退出码均为 **0**；完整 argv/stdout/stderr 在 `checks.json`。compileall 只证明 Python 编译，rustfmt 只证明 Rust 语法格式，不代表 cargo 编译。
- 开发中初轮 10 项失败、扩展轮 4 项失败均已记录在 `/tmp/b0928-t4-evidence/initial-tests.log` 与 `tests.log`，真实退出码均为 1；修复后才得到最终绿色结果。

测试变更说明：原测试要求“发散拟合经缩放必定稳定”、默认机器路径、v1 无来源产物可直通 bridge，这些预期与整改要求冲突。已替换为 Hurwitz/Schur 正反例、GCCA 原始岭回归数值对照、共享划分拒绝、manifest 错配拒绝、CLI 实际入口测试、启动 profile 结构校验。候选数、domain、重复候选、非有限输入、f32 溢出等检查改为参数化。旧 v1 真实产物保留，测试要求明确拒绝它；没有给它补造 GCCA/core 来源。原 Rust replay 测试改为校验该旧产物被 Python 拒绝，没有把历史快照再当作新链路成功证据。

## 未验证

- Windows `.bat` 在 Windows 上启动完整 405B / 小切片的实际 llama-server 加载与显存占用；已验证参数/结构门禁，未执行大模型启动。
- A100 有序轨迹上的真实动态系统辨识；只运行了数学稳定/发散反例与生产代码编译检查。
- 更新后的 Rust 测试没有运行 cargo 编译/执行。此次未进行重型编译，没有远程部署。
- 本次边界是产物、Python bridge、CLI/client。Rust `nanocore_ask` 的直接裸向量调用仍是原有 shape 检查（`crates/gen-zero-service/src/zero.rs:2261`），不会自动获得 Python 身份校验；不能把本次结果表述为“所有 Rust/MCP 裸调用均强制空间身份”。

## 未完成

- 与新 GCCA/anchor 空间匹配的真实 core 训练、core manifest 发布与 live Rust/MCP 接受验证。仓内现有 core/fixture 未提供这种绑定；真实验证产物刻意保持 **unbound**，bridge 实测拒绝信息为 `artifact lacks complete source -> GCCA -> anchor -> core identity`。未用随意 core 摘要给真实产物伪造可用性。
- 没有任务准确率提升结论，也没有用模型假设代替训练证据。

## 对要求 1–10 的核查

1. 明确披露独立样本伪轨迹、错用判稳、真实 core 缺失、Rust 直接调用边界。
2. 不稳定、低秩、旧版本、错配身份、坏划分、坏启动 profile 均报错；无新增静默 fallback。
3. 数学单测、真实拟合与能力评估分开；逐样本 transform 数据可复核，不声称能力突破。
4. 上述三类状态与原始退出码、完整日志、代码位置对应。
5. 新变换已用于锚点 CLI、bridge、client；数值 CLI 入口测试通过。被替代稳定缩放符号和旧生产调用已移除。Rust 裸入口边界明确列出，未冒称覆盖。
6. 未执行 stash/checkout/reset/clean/push --force。
7. 修复后继续排查生产调用、隔离划分、修改受影响测试并运行全套聚焦检查。
8. 本次是单线程 NumPy 与 Python 测试，无大体积编译；未制造远端任务或残留服务。
9. 不用同维空间伪装兼容，不用数学构造数据宣称训练能力，不把旧 fixture 直通称为新契约接受。
10. 不保留旧稳定投影或无来源 bridge 兼容分支；v1 产物明确拒绝，完整模型与切片显式分离。
