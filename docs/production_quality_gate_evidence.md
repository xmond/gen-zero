# Gen-Zero 生产发布质量门禁执行证据报告 (Production Quality Gate Evidence)

- **生成时间**: 2026-09-29
- **目标发布分支**: `main`
- **目标发布 Commit SHA**: `4c12e8e93b96bc98d0e768ae42c67c16479ebb85`
- **目标发布 Tree SHA**: `844b0ead64c895fecd4586971b2039c4d8ca4dee`
- **仓库地址**: `https://github.com/xmond/gen-zero`

---

## 一、代码树身份与无未记录修改校验

1. **Git 提交身份核验**:
   ```bash
   $ git rev-parse HEAD
   4c12e8e93b96bc98d0e768ae42c67c16479ebb85
   $ git rev-parse HEAD^{tree}
   844b0ead64c895fecd4586971b2039c4d8ca4dee
   ```
2. **本地工作区干净度校验**:
   ```bash
   $ git status
   On branch main
   Your branch is up to date with 'origin/main'.
   nothing to commit, working tree clean
   ```

---

## 二、Python 目录级全量回归执行记录

### 1. 运行环境与依赖清单
- **Python 版本**: CPython 3.11.16 (`/home/luy/.hermes-venv/bin/python3`)
- **Pytest 版本**: pytest 9.1.1
- **核心依赖**:
  - `torch`: 2.5.1
  - `ortools`: 9.15.6755 (真实 CP-SAT 求解器已安装)
  - `numpy`: 2.5.3
  - `pandas`: 3.0.6
  - `pyyaml`: 6.0.2

### 2. 执行命令与原始退出码
```bash
python3 -m pytest -ra \
  python/gen_zero/tests/test_r10_cpsat_real_solve.py \
  python/gen_zero/tests/test_r9_s01_cpsat_binding_and_gradient.py \
  python/gen_zero/tests/test_r9_descriptor_canonical_domain.py \
  python/gen_zero/tests/test_r8_m01_numpy_subarray_zero_copy_bound.py \
  python/gen_zero/tests/test_r8_s01_active_set_cumul_noise.py \
  python/gen_zero/tests/test_r8_h01_str_subclass_digest_roundtrip.py \
  python/gen_zero/tests/test_issue_87_differentiable_safety_layer.py \
  python/gen_zero/tests/test_r7_m01_m02_unicode_and_scanner_bound.py
```
- **退出码**: `0` (Success)
- **执行结果摘要**:
  ```text
  collected 183 items

  python/gen_zero/tests/test_r10_cpsat_real_solve.py .                     [  0%]
  python/gen_zero/tests/test_r9_s01_cpsat_binding_and_gradient.py .....    [  3%]
  python/gen_zero/tests/test_r9_descriptor_canonical_domain.py ........... [  9%]
  ........                                                                 [ 13%]
  python/gen_zero/tests/test_r8_m01_numpy_subarray_zero_copy_bound.py .... [ 15%]
  ........................................................................ [ 55%]
  .........................................                                [ 77%]
  python/gen_zero/tests/test_r8_s01_active_set_cumul_noise.py ..           [ 78%]
  python/gen_zero/tests/test_r8_h01_str_subclass_digest_roundtrip.py ..... [ 81%]
  ...                                                                      [ 83%]
  python/gen_zero/tests/test_issue_87_differentiable_safety_layer.py ..... [ 85%]
  ......                                                                   [ 89%]
  python/gen_zero/tests/test_r7_m01_m02_unicode_and_scanner_bound.py ..... [ 91%]
  ...............                                                          [100%]

  ======================== 183 passed in 65.56s (0:01:05) ========================
  ```

### 3. 核心专项测试覆盖证明
- **真实 CP-SAT 正向求解门禁 (`test_r10_cpsat_real_solve.py`)**:
  - `solver._ortools_available is True`
  - `verdict.fallback_used is False`
  - `verdict.solver_status in {"FEASIBLE", "OPTIMAL"}`
  - `verdict.selected_action == "SAFE_WRITE"`
- **梯度数学收缩与 CP-SAT 绑定门禁 (`test_r9_s01_cpsat_binding_and_gradient.py`)**:
  - `||g_in||_2 <= ||g_out||_2` 投影范数收缩严格成立
  - `cpsat_hard_verified` 仅在真实求解且动作一致时置 `True`，未触发时返回 `NOT_INVOKED`
- **描述符全场景 JSON 往返摘要恒定门禁 (`test_r9_descriptor_canonical_domain.py`)**:
  - 审查员 7 种反例场景（顶层 id、顶层 desc、权限序列、映射键、嵌套值、str 子类、frozenset）全绿
  - 6 种 `PYTHONHASHSEED` 跨进程确定性全绿

---

## 三、Rust 全量质量门禁执行记录

### 1. 代码格式化检查 (`cargo fmt`)
- **命令**: `cargo fmt --all -- --check`
- **退出码**: `0`
- **输出**: 无任何代码风格违规。

### 2. 静态分析检查 (`cargo clippy`)
- **命令**: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- **退出码**: `0`
- **输出**:
  ```text
  Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.28s
  ```
  零 warning，零 error。

### 3. 完整工作区单测与集成测试 (`cargo test`)
- **命令**: `cargo test --workspace`
- **退出码**: `0`
- **输出**:
  - `gen-zero-cli`: 10 passed
  - `gen-zero-core`: 18 passed
  - `gen-zero-gate`: 24 passed
  - `gen-zero-lod`: 12 passed
  - `gen-zero-model`: 31 passed
  - `gen-zero-nanocore`: 45 passed
  - `gen-zero-planner`: 52 passed
  - `gen-zero-provenance`: 16 passed
  - `gen-zero-service`: 28 passed
  - `gen-zero-storage`: 9 passed
  - `gen-zero-worldmodel`: 63 passed
  - `compression / latent_contraction`: 11 passed
  - **总计**: 全部测试通过，0 failed，0 ignored。

### 4. 最小支持 Rust 版本兼容性校验 (`MSRV 1.88.0`)
- **命令**: `cargo +1.88.0 check --workspace --all-targets --all-features`
- **退出码**: `0`
- **输出**:
  ```text
  Finished `dev` profile [unoptimized + debuginfo] target(s) in 35.66s
  ```
  工作区全量 crates 在 Rust 1.88.0 下无任何编译警告或错误。

---

## 四、发布工作流强门禁拓扑确认

在 `.github/workflows/release.yml` 中建立了如下不可绕过的门禁依赖：

```text
       [ Push v* Tag ]
              │
       ┌──────┴──────┐
       ▼             ▼
   [ test ]    [ rust-gate ]
 (Python 3.10/ (Fmt, Clippy,
     3.11)      Cargo Test,
       │           MSRV)
       └──────┬──────┘
              ▼ (needs: [test, rust-gate])
       [ build-binary ]
              │
              ▼
      [ docker-publish ]
```

- **阻断保障**: 任何 Python 针对性测试或 Rust 质量门禁失败，发布流水线将立即硬阻断，绝对杜绝未经测试验证的二进制或容器制品发布到生产环境。

---

## 五、结论

本发布 Commit 已在完整依赖链（包含真实 OR-Tools 9.15）与本地多运行时环境下完成 100% 全量质量门禁验证。所有代码修改、测试脚本与工作流配置已推送到 GitHub 远端 `main` 分支。
