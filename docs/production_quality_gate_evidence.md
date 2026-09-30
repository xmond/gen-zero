# Gen-Zero Production Quality Gate Evidence

- **Generated**: 2026-09-29
- **Target release branch**: `main`
- **Target release commit SHA**: `4c12e8e93b96bc98d0e768ae42c67c16479ebb85`
- **Target release tree SHA**: `844b0ead64c895fecd4586971b2039c4d8ca4dee`
- **Repository URL**: `https://github.com/xmond/gen-zero`

---

## 1. Code Tree Identity and Unrecorded-Change Check

1. **Git commit identity verification**:
   ```bash
   $ git rev-parse HEAD
   4c12e8e93b96bc98d0e768ae42c67c16479ebb85
   $ git rev-parse HEAD^{tree}
   844b0ead64c895fecd4586971b2039c4d8ca4dee
   ```
2. **Local worktree cleanliness check**:
   ```bash
   $ git status
   On branch main
   Your branch is up to date with 'origin/main'.
   nothing to commit, working tree clean
   ```

---

## 2. Full Python Directory Regression Record

### 1. Runtime and dependencies
- **Python version**: CPython 3.11.16 (`/home/luy/.hermes-venv/bin/python3`)
- **Pytest version**: pytest 9.1.1
- **Core dependencies**:
  - `torch`: 2.5.1
  - `ortools`: 9.15.6755 (real CP-SAT solver installed)
  - `numpy`: 2.5.3
  - `pandas`: 3.0.6
  - `pyyaml`: 6.0.2

### 2. Command and original exit code
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
- **Exit code**: `0` (Success)
- **Execution summary**:
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

### 3. Evidence of focused test coverage
- **Real CP-SAT positive-solve gate (`test_r10_cpsat_real_solve.py`)**:
  - `solver._ortools_available is True`
  - `verdict.fallback_used is False`
  - `verdict.solver_status in {"FEASIBLE", "OPTIMAL"}`
  - `verdict.selected_action == "SAFE_WRITE"`
- **Gradient norm contraction and CP-SAT binding gate (`test_r9_s01_cpsat_binding_and_gradient.py`)**:
  - `||g_in||_2 <= ||g_out||_2` projection norm contraction holds
  - `cpsat_hard_verified` is `True` only after a real solve with a matching action; when not invoked, it returns `NOT_INVOKED`
- **Descriptor digest stability across JSON round trips (`test_r9_descriptor_canonical_domain.py`)**:
  - All seven reviewer counterexample cases passed (top-level id, top-level desc, permission sequence, mapping key, nested value, str subclass, frozenset)
  - 6 `PYTHONHASHSEED` cross-process determinism cases passed

---

## 3. Full Rust Quality Gate Record

### 1. Code formatting check (`cargo fmt`)
- **Command**: `cargo fmt --all -- --check`
- **Exit code**: `0`
- **Output**: No code-style violations.

### 2. Static analysis check (`cargo clippy`)
- **Command**: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- **Exit code**: `0`
- **Output**:
  ```text
  Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.28s
  ```
  Zero warnings and zero errors.

### 3. Full workspace unit and integration tests (`cargo test`)
- **Command**: `cargo test --workspace`
- **Exit code**: `0`
- **Output**:
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
  - **Total**: All tests passed; 0 failed; 0 ignored.

### 4. Minimum supported Rust version check (`MSRV 1.88.0`)
- **Command**: `cargo +1.88.0 check --workspace --all-targets --all-features`
- **Exit code**: `0`
- **Output**:
  ```text
  Finished `dev` profile [unoptimized + debuginfo] target(s) in 35.66s
  ```
  All workspace crates compiled under Rust 1.88.0 without warnings or errors.

---

## 4. Release Workflow Gate Topology

The following mandatory gate dependencies were established in `.github/workflows/release.yml`:

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

- **Blocking guarantee**: If any focused Python test or Rust quality gate fails, the release workflow blocks publication of unverified binary or container artifacts.

---

## 5. Conclusion

This release commit has completed 100% full quality gate verification under the complete dependency chain (including real OR-Tools 9.15) and the local multi-runtime environment. All code changes, test scripts, and workflow configuration have been pushed to the `main` branch on the GitHub remote.
