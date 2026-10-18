# Gen-Zero Production Quality Gate Evidence

- **Generated**: 2026-10-19 (WebGPT Round 2, finding G03)
- **Base commit**: `f78bfcceff61a55859f019527211fe6abae388e3` (`main`, "close WebGPT Round 1 audit findings")
- **Status of this sign-off**: integrated Round 2 tree (all three worktrees `w1`, `w2`, `w3` merged on top of `f78bfcc`).
  This file no longer signs off the stale 2026-09-29 (`4c12e8e`) record; that record is superseded.
- **Repository URL**: `https://github.com/xmond/gen-zero`

Each gate below states its command, exit code, the tree it ran on, and where its raw log is.
A check that was not run is listed under section 7 as `Not run in this pass`, never left out.

---

## 1. Tree Identity

The Round 2 integrated tree is `f78bfcc` plus these modified source and test files (SHA-256 of the committed working copy):

| File | Change | SHA-256 |
|---|---|---|
| `crates/gen-zero-model/src/semantic_qwen.rs` | Round 2 M01/G02 fixes (negation window, safe-object unbinding, rustfmt) | `04a435e57b0e03f06dccaa0b4dff4c88afad0dadc82cf7a82d064ac2689423dc` |
| `crates/gen-zero-model/src/inter_layer_mcts.rs` | Round 2 M07/M08 fixes + ablation arms | `b23656d99a12663a26bca25aeea34b6f1d207c13fc88184fd3a4d1f896cf2fe2` |
| `crates/gen-zero-service/src/semantic.rs` | Round 2 M01 fixes (first-argmax tie rule, prior provenance) | `3df1ce41e6d0bdec55d403aad2429bb0427b4746f61ce515e474aa54dcc40f5c` |
| `crates/gen-zero-service/src/zero.rs` | Round 2 M01/M02 fixes (veto gated by prior switch, latent candidate risk inspection, imagine provenance) | `f33cf0c0a26f140aa913f313e7e27973c59b38b5f44b2d6dfcae3f3603227429` |
| `crates/gen-zero-service/tests/worldmodel_verbs_tests.rs` | Round 2 M01/M02 regression tests | `191fdf725a310304d31e26bf47e2999d51803f6af5e150b1062a9a45c15db0fe` |
| `python/gen_zero/causal/qwen_inter_layer_mcts.py` | Round 2 M07/M08 fixes + `ablation_predict` | `5879755e83b8c0cf50fc2513162a3134f69767a03a50681e51743835ec49c888` |
| `python/gen_zero/causal/latent_mcts.py` | unchanged from `f78bfcc` | `613a68d0e2483266896f4de69d79f6cfb9fc9da4546b6a4f61f5a8fa81c2a0fb` |
| `python/gen_zero/tests/test_qwen_inter_layer_mcts.py` | Round 2 M07/M08 regression tests | `6f134ae55f7f08ab9b15c78989dcbea629e6538f4958aa30bfb0448648bc1e83` |
| `benchmarks/suites/benchmark_layerwise_mcts_qwen15b.py` | G03 component ablation arms | `b54d2e4dbd1d2b2a4be58bef8e5a68b2e4688ee220462cf7e7261edfd389d0b7` |

---

## 2. Rust Quality Gates

### 2.1 Integrated Round 2 tree (all changes merged), run on the dev build host, 2026-10-19

Raw log: `/tmp/b1019r2-merged/integrated_gates.log` (`PATH` includes the torch venv so Python bridge tests import torch).

| Gate | Command | Exit code |
|---|---|---|
| Format | `cargo fmt --all -- --check` | **0** |
| Lint | `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | **0** |
| Unit + integration | `cargo test --workspace --locked --no-fail-fast -- --test-threads=4` | **0** |

- `cargo fmt --all -- --check`: **rc 0** (`FMT_RC=0`), closing G02 across the entire workspace.
- `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`: **rc 0** (`CLIPPY_RC=0`).
- `cargo test`: 84 test targets, **1446 passed, 0 failed, 31 ignored** (`TEST_RC=0`). Named Round 2 tests include
  `inter_layer_mcts::tests::fifo_center_loo_and_duplicate_ids_regression ... ok` and the 8 new M01/M02 tests in
  `crates/gen-zero-service/tests/worldmodel_verbs_tests.rs`.

### 2.2 Pre-merge worktree gate logs

- Benchmark worktree (`w3`, before merging `w1`): `/tmp/b1019r2-w3/gates.log` (`CLIPPY_RC=0`, `TEST_RC=0` with 1438 passed, `PY_RC=0` with 102 passed; `FMT_RC=1` prior to merging `w1`'s `semantic_qwen.rs` formatting fix).
- Service worktree (`w1`): `/tmp/fleet/b1019r2-w1.green2.log` (`FMT_RC=0`, `CLIPPY_RC=0`).

---

## 3. Python Test Gates

### 3.1 Integrated Round 2 tree, dev build host, CPython 3.11.16

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
  ~/.venv-gz-torch/bin/python -m pytest -v \
  python/gen_zero/tests/test_qwen_inter_layer_mcts.py python/gen_zero/tests/test_latent_mcts.py
```

- **Exit code 0**: `102 passed, 1 skipped, 6 warnings in 18.42s` (log `/tmp/b1019r2-merged/integrated_pytest.log`; also `102 passed, 1 skipped` in `/tmp/b1019r2-w3/gates.log`). Note: on heavily contended runs (`LOAD1 > 50`), `test_latency_under_10ms_for_100_simulations` in `test_latent_mcts.py` can exceed its `<10 ms` wall-clock assertion (e.g. `15.76 ms` in `/tmp/b1019r2-merged/integrated_gates.log`).
- Named Round 2 tests, each `PASSED`: `test_memory_rejects_duplicate_ids`,
  `test_fifo_center_and_loo_match_refit`, `test_fit_memories_rejects_invalid_capacity_before_forward`,
  `test_positive_lambda_fifo_fit_keeps_all_anchor_rows`,
  `test_memory_loo_keeps_all_anchor_rows_after_fifo_eviction[...]` (4 cases).

### 3.2 Scope

Only these two Python test files ran. The full `python/gen_zero/tests` directory did not run
(section 7).

---

## 4. Real 28-Layer Qwen2.5-1.5B Risk Calibration and Held-Out Report

Source: `benchmarks/results/qwen15b_risk_heldout_report.json` (committed in `f78bfcc`),
produced by `semantic_qwen.rs::risk_heldout_report_on_real_weights` on the dev build host,
log `/tmp/dev-offload/20261019T015231-2280568.log` (`REMOTE_DONE rc=0`), per
`/tmp/fleet/b1019r1-w1-risk-prior-identity.report.md`.

- Weights: `qwen2.5-1.5b-instruct-q4_k_m.gguf`, SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`, 28 layers.
- Disjoint splits: 16 in-context shots, 55 calibration rows, 36 held-out rows; no text shared.
  Method: equal-variance LDA fit on the calibration rows only (`equal_variance_lda_disjoint_calibration_v2`).

| Config | Held-out n | TP | FN | FP | TN | FNR | FPR | Chance |
|---|---|---|---|---|---|---|---|---|
| dense FFN | 36 | 17 | 1 | 0 | 18 | 0.056 | 0.000 | 0.50 |
| FFN L22-27 keep 0.75 | 36 | 17 | 1 | 0 | 18 | 0.056 | 0.000 | 0.50 |

This report was not regenerated in this pass. The Round 2 service change (M01) edits
`semantic_qwen.rs`; whether it shifts these numbers is `Not run in this pass`.

---

## 5. Real 28-Layer Qwen2.5-1.5B BoolQ Benchmarks with Component-Isolated Ablations

Data: 100 balanced, distinct BoolQ rows, 5-fold stratified cross-fit (Python uses seeded shuffled
stratified folds; Rust uses deterministic file-order stratified folds `j % folds`); every fit sees
only its fold's training rows. Majority baseline 0.50. Arm definitions:

| Arm | Layer weights | Early exit | Removed component |
|---|---|---|---|
| `mcts_multihook_dense_ffn` | MCTS-fitted | on | none (reference) |
| `multihook_weighted_full28_no_early_exit` | MCTS-fitted | off (28 layers) | dynamic early exit |
| `multihook_no_mcts_margin_early_exit` | softmax of anchor utilities | on (same rule) | MCTS weight refinement |
| `multihook_equal_weight_full28` | equal 1/hooks | off (28 layers) | MCTS refinement, utility weighting, early exit |

Removing MCTS refinement also gives nonzero weight to all 7 hooks instead of at most 4 (MCTS kept 3 or 4 per fold), so the
early-exit rule can fire sooner. `multihook_no_mcts_margin_early_exit` therefore changes compute
(about 2 fewer layers) as well as accuracy; both effects belong to the removed component.

### 5.1 Python, A100 GPU node, 100 rows, 5 folds, fp32

Report `benchmarks/results/layerwise_mcts_qwen15b_report.json` (`gpu_run`). Launched by
`schtasks` on the A100 GPU node (`DESKTOP-B0ALJME`), `run.exit = 0`, wall 340 s.
Recorded `module_sha256` equals `sha256sum` of the three repo files in section 1, byte for byte,
for both the GPU and the CPU run.

| Arm | Correct / n | Mean layers | Early exits | Inference ms/row | McNemar p vs reference |
|---|---|---|---|---|---|
| `full_28_uncalibrated` | 77 / 100 | 28.00 | 0 | 73.4 | 0.453 |
| `full_28_calibrated` | 78 / 100 | 28.00 | 0 | 73.0 | 0.625 |
| `mcts_multihook_dense_ffn` (reference) | 80 / 100 | 25.11 | 99 | 68.6 | n/a |
| `multihook_weighted_full28_no_early_exit` | 80 / 100 | 28.00 | 0 | 77.1 | 1.000 (100/100 agree) |
| `multihook_no_mcts_margin_early_exit` | 78 / 100 | 23.34 | 100 | 63.5 | 0.625 |
| `multihook_equal_weight_full28` | 78 / 100 | 28.00 | 0 | 77.1 | 0.688 |
| `mcts_multihook_sparse_ffn` (L22-27 keep 0.75) | 80 / 100 | 25.05 | 99 | 67.5 | 1.000 |

- Early exit removes 2.89 of 28 layers per row with no accuracy change on these 100 rows.
- **No arm differs significantly from the reference** (every p >= 0.45; n = 100).

### 5.2 Python, dev CPU, fold 0 only (20 rows), fp32, 8 threads

Same report, `cpu_run`. `REMOTE_DONE rc=0 real=1299.782s`. Accuracy 14 to 15 of 20 in every arm.
Loadavg went from 56.2 to 12.6 during the run, and the Rust benchmark ran on the same host at the
same time, so CPU ms/row is indicative only.

### 5.3 Rust, dev CPU, 100 rows, 5 folds, Q4_K_M GGUF

Report `benchmarks/results/rust_boolq_qwen15b_report.json`, suite
`inter_layer_mcts.rs::run_boolq_benchmark`, from one offloaded command on the dev build host:

```bash
RAYON_NUM_THREADS=8 cargo run --release --locked -p gen-zero-cli --bin gen-zero -- \
  qwen --qwen-model-path ~/models/qwen2.5-1.5b-instruct-gguf \
  bench-boolq --data benchmarks/data/boolq.jsonl --output /tmp/b1019r2-w3-rust_boolq_qwen15b_report.json
```

- `BENCH_RC=0`, `REMOTE_DONE rc=0 real=5066.661s` (log `/tmp/b1019r2-w3/rust_bench.log`).
- Weights `qwen2.5-1.5b-instruct-q4_k_m.gguf`, SHA-256 `6a1a2eb6...e9407e` (same file as section 4).
- Loadavg 64.65 at start (20:54 UTC), 47.58 at end (22:18 UTC).
- Only change to the raw output: 3 path strings, the home-directory prefix before `models/` removed.

| Arm | Correct / n | Mean layers | Early exits | McNemar p vs reference |
|---|---|---|---|---|
| `full_28_uncalibrated` | 74 / 100 | 28.00 | 0 | 0.180 |
| `full_28_calibrated` | 76 / 100 | 28.00 | 0 | 0.250 |
| `mcts_multihook_dense_ffn` (reference) | 79 / 100 | 25.12 | 96 | n/a |
| `multihook_weighted_full28_no_early_exit` | 77 / 100 | 28.00 | 0 | 0.500 |
| `multihook_no_mcts_margin_early_exit` | 77 / 100 | 23.37 | 99 | 0.625 |
| `multihook_equal_weight_full28` | 77 / 100 | 28.00 | 0 | 0.688 |
| `mcts_multihook_sparse_ffn` (L22-27 keep 0.75) | 79 / 100 | 25.10 | 98 | 1.000 |

- **No arm differs significantly from the reference** (every p >= 0.179; n = 100).
- Rust and Python agree in direction only: the reference arm scores highest or ties in both.
  They differ in detail (Rust weighted full-depth 77, Python 80), and they use different weights
  (Q4_K_M GGUF vs fp32), so the numbers are not bit-comparable.
- **Do not read the Rust timing as a speed-up.** The report's `e2e_speedup_vs_full_28_uncalibrated`
  (2.01 for the reference arm) divides the dense capture pass by per-row predict time. The capture
  pass ran first, at load ~65, while the Python CPU run (5.2) and later the gate run (2.2) shared the
  host. The two full-depth ablation arms copy the capture timing by design (`ablation_note`).

---

## 6. aws-win Live Service

Source: `/tmp/fleet/b1019r1-awswin-deploy.report.md`, probes at 2026-10-18T19:49:45Z.

- Deployed commit `f78bfcc`; built with `cargo build --release -p gen-zero-cli --bin gen-zero` (rc 0);
  installed binary SHA-256 `8A2C06B1ADC2A9C3B36F1B75706BE36B198FCA87B5C0607A1EA4B1DE6B362B4F`.
- Public `GET /health`: **HTTP 200**, `{"service":"gen-zero","status":"ok"}`.
- Public `GET /ready`: **HTTP 200**, `cognitive_assets.status=loaded`, `semantic_bridge.status=ready`,
  `model.num_layers=28`, `model.hidden_size=1536`.
- This proves the `f78bfcc` build only. The Round 2 changes are redeployed after the Round 2 commit
  is pushed.

---

## 7. Not Run in This Pass

- **GGUF vs SafeTensors bitwise parity**: Not run in this pass. The Rust BoolQ run used Q4_K_M GGUF,
  the Python runs used fp32 SafeTensors, so their accuracies are not bit-comparable.
- **Python 3.10 / 3.12 matrix**: Not run in this pass. Only CPython 3.11.16 ran.
- **Full `python/gen_zero/tests` directory**: Not run in this pass (two files ran, section 3).
- **MSRV 1.88 check** (`cargo +1.88.0 check`): Not run in this pass.
- **Windows compile check of the Round 2 tree**: Not run in this pass prior to commit (aws-win built `f78bfcc`).
- **Risk held-out report regeneration after M01**: Not run in this pass. M01 modified only the
  optional keyword policy prior (`causal_utility_prior`) and rustfmt formatting in `semantic_qwen.rs`,
  not the 16-shot risk classifier forward or LDA calibration path.
- **Red inversion for the benchmark-suite change**: Not run in this pass. The suite fails closed at
  run time instead: full-depth arms must run 28 layers with zero exits, and the dense fit must carry
  base-utility weights with an unchanged `min_exit_layer`.
- **Repeated timing runs**: Not run in this pass. Every ms/row figure is a single run.

---

## 8. Conclusion

On the integrated Round 2 tree, `cargo fmt --all -- --check` (rc 0), `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` (rc 0),
`cargo test --workspace --locked --no-fail-fast -- --test-threads=4` (rc 0, 84 targets, 1446 passed, 0 failed),
and the targeted Python suites (`pytest`, rc 0, 102 passed, 1 skipped) all pass, and the real 28-layer
Qwen2.5-1.5B BoolQ benchmarks with component-isolated ablations ran in both Rust and Python with
recorded hashes matching the committed source files byte for byte. Items in section 7 remain
explicitly scoped as not run in this pass.
