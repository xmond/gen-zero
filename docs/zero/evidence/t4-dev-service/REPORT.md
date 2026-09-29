# T4 dev 决策服务部署验收（2026-09-26）

## 已实现

- 源工作区 `/workspace/pj/gen-zero-worktree/d4-dev-service`，基线 HEAD `478699c079bf4a053b953bb83b7e7f310eae4922`。首次 `git status --short` 无输出；本轮代码改动仅 CLI host/mode 参数。
- `rsync -az --exclude target --exclude .git ./ dev:/home/user/gen-zero-dev-eval/` 完成，退出码 0。未使用删除同步；目标初始不存在。后续 checksum dry-run 无文件内容差异，仅目录时间与证据目录差异，见 `source-check.log`。远端无 `.git`，见 `no-git.exit`。
- dev 为 `luy-dev`，`nproc=64`。构建设置 64 个 Cargo 作业（不等于证明每时每刻 64 核满载）。Cargo 1.98.1 / rustc 1.98.1。
- release 构建退出码 0，Cargo 报告 **54.48 秒**，秒级墙钟计时 54 秒；二进制 **10,779,448 bytes**，见 `build.log`、`build.exit`、`build.seconds`、`binary-size.txt`、`binary.sha256`。
- 原命令的 `gen_zero` 实际不存在；Cargo 定义名为 `gen-zero`（`crates/gen-zero-cli/Cargo.toml:14`）。原 CLI 无 `--host` 且默认 stdio。本轮在 `crates/gen-zero-cli/src/main.rs:48` 增加 host IP 校验和 mode 枚举，在 `:398` 使用指定地址。非法 mode/host 均退出 2；`cargo fmt --all -- --check` 退出 0。
- Rust 与 Python scorer 均以 nohup 运行，监听回环地址 8080 / 8995。PID 与进程命令见 `process.txt`，8080 监听证据见 `listener.txt`。这是后台常驻进程，不具备 systemd 重启/开机保障。
- 为满足实际语义调用，在远端 `.venv-t4` 安装依赖，以 `.pth` 显式复用 `/home/user/genz-eval-venv/lib/python3.14/site-packages` 的 PyTorch 2.14.0+cpu。完整版本见 `uv-python-freeze.txt`；初始失败和修复记录均保留。
- 复制本机既有 Qwen2.5-0.5B snapshot `060db6499f32faf8b98477b0a26969ef7d8b9987`，权重 SHA-256 两端均为 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`。scorer 使用真实 fp32 前向计算，8 线程。
- scorer 的鉴权 token 只存在 dev `.scorer-token`（0600）；无密钥写入本证据目录。Rust 用 `GENZERO_PYTHON_API_KEY` 访问 scorer，并开启 `GENZERO_BRIDGE_REQUIRED=1`。8080 保持用户要求的回环开放访问。

## 真实 HTTP 结果

| 请求 | HTTP | curl 原始退出码 | JSON/语义断言退出码 |
|---|---:|---:|---:|
| health | 200 | 0 | 0 |
| ready | 503 | 22 | 1 |
| ask | 200 | 0 | 0 |
| route | 200 | 0 | 0 |
| what_if | 200 | 0 | 0 |
| invalid_what_if | 400 | 22 | 0 |

`connected.exit=1`，原因是 readiness 503，未将整套测试伪报为通过。每次 curl 使用 `--fail-with-body`，因此 4xx/5xx 的原始退出码是 22；负例的 JSON 断言成功不改变原始 curl 退出码。

- `ask` 选择 `read README.md`；`route` 选择 `read_file`；两者 `engine=semantic_bridge`、`semantic_scoring=true`、risk assessed=true。候选分数、模型标识与完整 JSON 见 `connected/*.body.json`。
- `what_if` 完成 1024 维输入、两个候选、3 步 rollout，但明确 `trained=false`、`calibrated=false`、`advisory_only=true`。只验收接口与算法执行，不能声称真实世界预测正确。
- `invalid_what_if` 只给 1 维 state，返回 400 / isError=true。
- 首轮 scorer 未接通时 ask/route=428、ready=503，完整保留 `baseline/`、`baseline.log`、`service-baseline.log`。没有通过空 ask 固定返回 proceed 来冒充成功。

## 可复现命令与原始日志

```bash
ssh dev 'cd /home/user/gen-zero-dev-eval; export PATH=/home/user/.cargo/bin:$PATH; CARGO_BUILD_JOBS=64 cargo build --release -p gen-zero-cli'
# 启动程序的完整环境、鉴权读取、PID 管理见 start-services.sh；实际 Rust 命令：
nohup target/release/gen-zero serve --mode sse --port 8080 --host 127.0.0.1 > service.log 2>&1 < /dev/null &
ssh dev 'cd /home/user/gen-zero-dev-eval; bash t4-evidence/http-check.sh connected'
```

`http-check.sh` 包含全部 curl 与 JSON 断言；payload 在 `ask.json`、`route.json`、`what_if.json`、`invalid_what_if.json`。每个请求的完全展开命令、响应头、正文、stderr、HTTP 状态、原始退出码、断言退出码分别保存在 `connected/<name>.*`。`connected.log` 保留整轮输出。后续 ready 查询命令：

```bash
curl --silent --show-error --fail-with-body -D t4-evidence/ready-final.headers -o t4-evidence/ready-final.json -w '%{http_code}\n' http://127.0.0.1:8080/ready
```

最终原文见 `ready-final.json`：backbone_loaded=true，但 cognitive_assets unavailable。

## 未验证

- 当前工作区是否包含所有上游声称的“最新 S1~S5”提交：本轮部署以给定工作区为准，没有擅自 merge/pull。工作区包含 `SheafOperator`，但存在不等于本次请求已执行。
- 统一 Sheaf 认知路径端到端效果。ask/route 明确 `cognitive_runtime=not_engaged: request carries no numeric manifold coordinates`，mount has_cognitive_assets=false。
- 模型准确率、安全保证、性能基准、长期稳定性及重启恢复。本轮没有跑全仓测试。

## 未完成 / 实际阻塞

- **完整 readiness 尚未通过**：`/ready=503`，缺少真实 `gen-zero/cognitive-assets/v1` 资产。判定代码在 `crates/gen-zero-service/src/server.rs:992`、`:1001`，详细响应见 `ready-final.json:1`。仅有测试夹具，未拿它们冒充生产资产。已向用户索取真实资产路径。
- Python 初始化还报告 dual-head 无 checkpoint、vision 无真实权重；这些端点不属于本轮成功调用的语义 scorer，未被声称可用，原始告警见 `scorer.log`。
- 未替换或新增核心模块，无本轮被替代的旧模块需要移除；没有进行全仓旧符号清除验收。未提交、推送或执行 stash/checkout/reset/clean。
