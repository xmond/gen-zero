# B1001F T3 — graph persistence evidence

Workspace: `/ebs/pj/gen-zero-worktree/b1001f-t3-persist`  
Branch: `feat/b1001f-t3-persist`  
Base: `0a2a378421cc4bf0e6bd36f097c393bfa76f207c`

## 已实现

- 二进制版本 1 快照；128 节点一块，CSR 与元数据独立成块。节点、payload/digest、coord、HDC、band、confidence/status、CSR、pending edges/tickets、两类撤销、权限和 validated dependencies 全部保存。检查点身份、generation/discarded 是进程局部状态，不跨启动恢复。
- SHA-256 内容寻址；未变块不重写。写临时文件、fsync、rename、目录 fsync，最后原子替换带 SHA-256 的 `CURRENT.sha256`。首建目录逐级同步父目录；清单提交持久后清理未引用块。是增量磁盘写出，不是 O(delta) WAL：序列化/哈希仍扫描全图，CSR 改变时整个 CSR 块重写。
- `save_to_dir` / `load_from_dir` 与独占目录挂载。校验文件、版本、几何、节点、CSR 和边票号；坏状态拒绝恢复。CSR 反序列化本身也验证不变量，防止直接反序列化绕过校验。
- ZeroEngine 配置/环境变量 `GENZERO_GRAPH_PERSIST_DIR`；graph 事务与 `reflect_failure`（含失败时的 manual quarantine）提交接入磁盘。刷盘失败返回错误并 poison 引擎；后续请求不能继续决策。第二写者拒绝。
- 首个 seed 在持目录锁时加载，成功后才发布首快照；seed 失败不会留下可被下次启动静默接受的空快照。恢复不重复播种，有明确日志。
- HTTP 自动反思 → 落盘 → 新引擎 → `pipeline.decide` 仍阻断，及 prune/retract 跨重启效果均有生产入口集成测试。API 的 persisted 元数据改为实际挂载状态。

## 代码与测试定位（相对仓库根目录，path:line）

| 证据 | 定位 |
|---|---|
| 保存与原子发布 | `crates/gen-zero-lod/src/graph/persistence.rs:117`, `:129` |
| 恢复、校验、持锁挂载 | `crates/gen-zero-lod/src/graph/persistence.rs:173`, `:178`, `:258` |
| CSR 构造不变量 | `crates/gen-zero-lod/src/graph.rs:93` |
| 事务真实提交 | `crates/gen-zero-lod/src/graph.rs:2097`, `:2105` |
| 反思成功/失败路径均提交 | `crates/gen-zero-lod/src/graph.rs:2180`, `:2277`, `:2284` |
| 生产配置/恢复/失败阻断 | `crates/gen-zero-service/src/zero.rs:800`, `:834`, `:1042`, `:1866` |
| HTTP/MCP 构造同一引擎 | `crates/gen-zero-service/src/server.rs:382` |
| 节点/CSR 逐位对比、增量块复用 | `crates/gen-zero-lod/src/graph/persistence.rs:358` |
| 畸形 CSR、checksum/version/geometry/missing block | `crates/gen-zero-lod/src/graph/persistence.rs:424`, `:452` |
| 真实磁盘失败、独占写者 | `crates/gen-zero-lod/src/graph/persistence.rs:493` |
| 反思和失败 quarantine 重启 | `crates/gen-zero-lod/src/graph/persistence.rs:511` |
| 未提交文件不覆盖有效快照 | `crates/gen-zero-lod/src/graph/persistence.rs:527` |
| 引擎恢复/prune/evolve、写失败、seed 失败 | `crates/gen-zero-service/tests/graph_verbs_tests.rs:1285`, `:1330`, `:1345` |
| HTTP auto_reflect 恢复阻断 | `crates/gen-zero-service/tests/pipeline_service_tests.rs:878` |

## 可复现验证

远端 `dev` 的无 `.git` 沙箱 `/home/luy/b1001f-t3-persist`，代码只在本地编辑，rsync 后运行：

```sh
export CARGO_BUILD_JOBS=$(nproc)
export CARGO_TARGET_DIR=/home/luy/b1001f-t3-target
cargo test -p gen-zero-lod -p gen-zero-service > /home/luy/b1001f-t3-test.log 2>&1
rc=$?
echo "$rc" > /home/luy/b1001f-t3-test.exit
exit "$rc"
```

原始日志 `b1001f-t3-test.log`；原始退出码 `b1001f-t3-test.exit`。没有用 tail 管道代替 cargo 的退出码。最终统计见 `RESULTS.md`。

性能样本命令：

```sh
cargo test -p gen-zero-lod exact_snapshot_roundtrip_and_incremental_blocks -- --nocapture
cargo test -p gen-zero-service --test graph_verbs_tests durable_graph_restart_preserves_prune_and_evolve_gate_effects -- --nocapture
# 同一已构建 lod 单测二进制再连续运行 20 次，逐次记录原始退出码：
/home/luy/b1001f-t3-target/debug/deps/gen_zero_lod-189fe852edb3a79d \
  graph::persistence::tests::exact_snapshot_roundtrip_and_incremental_blocks --exact --nocapture
```

`restore-samples.log` 保留全部样本。载入计时包含打开目录锁、读清单/块、SHA 校验、反序列化、节点/几何/CSR 验证与内存重建，不含测试图生成/写盘。是 debug 构建、已热文件缓存、1024 节点、1 条 CSR 边、1 条 pending 边；不能外推到大图或冷盘。

`source.sha256` 包含 138 个 Rust 源文件/构建配置；远端 `sha256sum --quiet -c` 校验结果在 `b1001f-t3-source-check.exit`。`Cargo.lock.txt` 保存实际远端依赖解析；`environment.log` 保存工具链/健康信息。本地 `cargo fmt --all -- --check` 与 `git diff --check` 退出码均为 0。

首次编译退出 101（子模块方法可见性错误），修正后重跑；失败原文仍保留在 `b1001f-t3-test-1.log` / `.exit`。其余中间验证日志亦保留，不替换成成功日志。最初 `/ebs/tmp` 写入被权限拒绝，未在那里启动测试；随后采用有权限且空间充裕的 home 沙箱。首次 dev 健康检查：64 核，1m/15m 负载 0.10/0.36，可用内存 66724 MiB；home 可用磁盘约 279 GiB。

## 未验证

- 真正 Firecracker 重启/宿主断电/块设备失效验收。当前验证为实际文件系统 I/O、引擎销毁后重新创建、损坏/缺失文件、未提交文件与真实写失败。没有使用 mock 持久化器。
- 任意图规模、冷缓存、低速数据盘的 `<50ms` SLA；没有声明突破或无规模限制的恢复能力。
- 全量命令中原有的 7 项 ignored 测试：1 项需要 Qwen 权重，6 项需要在线 Python 语义服务。这些并未启用，也没有改动其忽略标记。
- 并发请求可见性延续原图 API 语义；本变更未增加跨整个请求的可串行化隔离保证。

## 未完成

- 实际 MicroVM 数据盘挂载与服务部署：本仓库没有对应 VM 挂载配置/可供本任务验收的运行实例。需要宿主确保 `/var/lib/gen3` 是独立持久盘并在缺失挂载时拒绝启动，再设置 `GENZERO_GRAPH_PERSIST_DIR=/var/lib/gen3/lodgraph`。若把该路径放在每次重置的根盘上，代码无法使它跨 VM 生命周期保存；README 已明确此边界。
- 未推送或部署；本任务交付为指定分支本地 commit。

## 对要求 1–10 的核查范围

1. 正面记录增量写放大、数据盘前提、ignored 测试与性能样本边界。
2. 损坏拒绝、写失败 poison、第二写者拒绝、seed 失败不发布空图；没有 fallback。
3. 所有数值仅来自原始运行日志；本次不涉及训练或推理能力升级。
4. 本报告区分已实现、未验证、未完成；失败日志也保留。
5. `transact` 与 `reflect_failure` 直接调用持久化；HTTP/MCP 共用引擎。此前没有旧持久化模块可下线；已移除相关接口“永不持久化”的硬编码描述/字段。
6. 未执行 stash/checkout/reset/clean/force push；只提交本任务改动。
7. 修复实际发现的可见性、seed 初始化和 CSR 构造校验问题后重新验证。
8. dev 健康检查、64 核构建、无 git 沙箱、原始退出码、源 SHA 校验、日志归档；远端临时目录清理见 `cleanup.log`。
9. 无 mock、空壳能力声明、TODO 占位、删断言或降低校验；增量块复用由实际内容地址与测试证明。
10. 无兼容 shim 或旧格式降级；未知版本直接拒绝。真实生产上线验收没有冒充已完成。
