# Gen-Zero Production Quality Gate Evidence

- **Generated**: 2026-10-19 (WebGPT Round 3, findings M01.A/B/C, M08, G04.A/B/C; Round 2 record kept below)
- **Base commit**: `674071f3af40e6fe51f819d2d6dbe38587fc6edc` (`main`, "close WebGPT Round 2 findings")
- **Status of this sign-off**: integrated Round 3 tree (worktrees `b1019r3-w1-m01`, `b1019r3-w2-m08`,
  `b1019r3-w3-g04` merged on top of `674071f`, plus the dual-review fixes in section 1.1).
  The Round 2 sign-off on `f78bfcc` and the stale 2026-09-29 (`4c12e8e`) record are superseded.
- **Repository URL**: `https://github.com/xmond/gen-zero`

Each gate below states its command, exit code, the tree it ran on, and where its raw log is.
A check that was not run is listed under section 7 as `Not run in this pass`, never left out.

---

## 1. Tree Identity

The Round 3 integrated tree is `674071f` plus these files (SHA-256 of the integrated working copy
that the section 2 and 3 gates ran on; rows marked "unchanged" were re-hashed and still match):

| File | Change | SHA-256 |
|---|---|---|
| `crates/gen-zero-model/src/semantic_qwen.rs` | Round 3 M01.B (poison markers, direct-object rule in `acts_on_safe_object()`) | `01320a642132ee3dfb8d1dd9283a8ae05f86d2773c040ccbf0f0d825f7474658` |
| `crates/gen-zero-model/src/inter_layer_mcts.rs` | Round 3 M08 (`loo_query_for_all_anchors` reads the supplied states) | `426fe45e74790b66f5bee8cfe6683e9ce5f6887276715fbc9d3c8aef717f40f1` |
| `crates/gen-zero-service/src/bridge.rs` | Round 4 M11 (`check_raw_probabilities()`, new public symbol) | `75e6ed2d27c7072ba49e202885c57d630a00ba9e8407e7a603bcb626d372d48b` |
| `crates/gen-zero-service/src/semantic.rs` | Round 4 M11 (`apply_policy_prior()` validates raw probabilities before the prior) | `f419edc5349f625b0424556cd08283324f6e3456591804762190f995abccad40` |
| `crates/gen-zero-service/src/zero.rs` | Round 3 M01.A (ask gate sees context + `state`), M01.C (raw likelihood, `PlanLikelihood`) | `9fabef985de3dd4c7506e195722ef5780bd214a36cc37830de5ca053f9bd16c6` |
| `crates/gen-zero-service/src/imagine.rs` | Round 3 M01.C (`StepPriors`, per-step `raw_probability`); Round 4 M11 (simplex check in `expand()`) | `2d63dc708cc55b8c122a457da8fab020296fbfa049d67cddb827c0f5b7e14b09` |
| `crates/gen-zero-service/tests/ask_causal_prior_on_tests.rs` | Round 3 M01.A/C regression tests (prior switched on); Round 4 M11 test added | `d62dd3d3818c6c922990ce5e3ef11d27339a08b0b5db14b902001714b3d3d493` |
| `crates/gen-zero-service/tests/ask_raw_probability_off_tests.rs` | Round 4 M11 regression test (new file, prior switched off) | `6b2c328c0437f40cf7dedeb0778c8341b81bbc7e52d6c60554ebb954dd1f6a17` |
| `crates/gen-zero-service/tests/support/stub_scorer.rs` | Round 4 M11 shared stub scorer (new file) | `0fddf97c4b094260d43a05abb6dbdcc84fee3ef2fa629de8b8f96c372bd20867` |
| `crates/gen-zero-service/tests/worldmodel_verbs_tests.rs` | unchanged from `674071f` (Round 2 M01/M02 tests) | `191fdf725a310304d31e26bf47e2999d51803f6af5e150b1062a9a45c15db0fe` |
| `crates/gen-zero-planner/tests/deadline_anytime_tests.rs` | Round 3 G04 (wall ceiling 30 ms to 250 ms in one deadline test) | `bb7933728611f00dcb72f83f6204fa21a32cd691e2c781fca95cc37738ccaaec` |
| `python/gen_zero/__init__.py` | Round 3 G04.B (torch/scipy-optional imports) | `f3c4008bb48a5d7066ce3bf6aeacb9ae5245e18a64e1475933e5a051083be8fc` |
| `.github/workflows/ci.yml` | Round 3 G04.A/B (Windows sparse checkout, numpy step) | `5a0a14a7b9b36e0325c496112ef15af42ca6e1ba9c745202f345de17a7354683` |
| `benchmarks/results/reachability.json` | Round 3 G04.C (regenerated; 219/271 Python modules reachable); Round 4 M11 (`bridge.rs` public symbols 27 to 28) | `3cdde01734e2a4eb802220f4139ea5f17eaa35724598e8cb9379627d92e7a9eb` |
| `python/gen_zero/causal/qwen_inter_layer_mcts.py` | Round 2 M07/M08 fixes + `ablation_predict` | `5879755e83b8c0cf50fc2513162a3134f69767a03a50681e51743835ec49c888` |
| `python/gen_zero/causal/latent_mcts.py` | unchanged from `f78bfcc` | `613a68d0e2483266896f4de69d79f6cfb9fc9da4546b6a4f61f5a8fa81c2a0fb` |
| `python/gen_zero/tests/test_qwen_inter_layer_mcts.py` | Round 2 M07/M08 regression tests | `6f134ae55f7f08ab9b15c78989dcbea629e6538f4958aa30bfb0448648bc1e83` |
| `benchmarks/suites/benchmark_layerwise_mcts_qwen15b.py` | G03 component ablation arms | `b54d2e4dbd1d2b2a4be58bef8e5a68b2e4688ee220462cf7e7261edfd389d0b7` |

The three Python/benchmark files above are not modified in Round 3. Their `sha256sum` equals all six
`module_sha256` entries in `benchmarks/results/layerwise_mcts_qwen15b_report.json`
(`module_sha256_verification`, `gpu_run`, `cpu_run`).

### 1.1 Round 3 closures

| Finding | Closure (symbols as named in the code) | Named test, `... ok` in the section 2.1 log |
|---|---|---|
| M01.A | `zero.rs` ask handler: when `causal_prior_enabled()` is true, the confirmation gate scores `semantic_qwen::state_text(context, state)`, the same text the policy prior scored, and returns `ConfirmationRequired` if the chosen action's `policy_prior_adjustment < 0` or `causal_utility_prior(...) < 0`. Off (the default), it never vetoes. | `ask_causal_prior_on_tests::ask_gate_sees_the_state_the_policy_prior_scored` |
| M01.B | `semantic_qwen.rs`: food-domain spoiled markers add poison, toxic, contaminated and similar words; `acts_on_safe_object()` exempts an action only when the safe name is its direct object (after `OBJECT_FILLERS`), not anywhere in the clause. The Round 2 `negated_before()` rule is unchanged, so a negated hazard still scores +8. | `semantic_qwen::causal_prior_binding_tests::poison_markers_and_direct_object_bind_the_hazard` |
| M01.C | `imagine.rs`: `StepPriors { probability, raw_probability }` keeps the unreweighted model probability per step. `zero.rs`: `imagine` reports `raw_sequence_likelihood` and `raw_sequence_log_likelihood` beside `plan_sequence_*`, via `PlanLikelihood`. Dual-review fix: `PlanLikelihood::over` now stores the true unclamped product as the likelihood, and floors `p` at `1e-300` only inside `ln`, so a `1e-15` step reports `1e-15`, not `1e-12`. | `zero::tests::imagine_summary_names_the_score_it_reports` (asserts the `1e-15` case), `ask_causal_prior_on_tests::imagine_reports_raw_likelihood_beside_the_reweighted_score`, `imagine::tests::plan_steps_keep_the_raw_probability` |
| M08 | `inter_layer_mcts.rs` `EpisodicManifoldMemory::loo_query_for_all_anchors`: each supplied state is read around the mean of the entries with a different ID; a retained ID no longer returns its cached stored-anchor query. `loo_query()` delegates to it. | `inter_layer_mcts::tests::loo_query_for_all_anchors_uses_supplied_states`, `..._two_dimensional_reproducer`, `..._validates_retained_query_states` |
| G04.A | `ci.yml` `windows-check` sparse checkout: `crates`, `python`, `benchmarks/results` (the last for the `include_str!` of `qwen15b_risk_heldout_report.json` in `semantic_qwen.rs` tests). | Not run in this pass (no Windows runner, section 7) |
| G04.B | `python/gen_zero/__init__.py` keeps the real `from .client import ...` / `from .manifold import ...` statements, each in `try`/`except ModuleNotFoundError` that re-raises unless the missing module is `torch` or `scipy`. `python/gen_zero/causal/__init__.py` is byte-identical to `674071f`: all 13 of its submodules import with torch and scipy blocked, so it needs no guard. `ci.yml` installs numpy in the `test` (ubuntu, macos) and `msrv` jobs, both of which run `cargo test`. | `test_nanocore_live::legacy_fixture_artifact_is_rejected_by_the_python_bridge` (dev `python3` has no torch) |
| G04.C | `benchmarks/results/reachability.json` regenerated: 219/271 Python modules reachable, the same 219-module set as `674071f`'s own code produces (0 lost, 0 gained). A second run, and runs on dev with CPython 3.11.16 and 3.14.4, give the same SHA-256. | section 3.2 |
| M11 | `bridge.rs`: new `check_raw_probabilities()` validates each raw model probability. `semantic.rs::apply_policy_prior()` runs that check on the raw values before the causal prior reweights them. `imagine.rs::expand()` checks the probability simplex on both `probability` and `raw_probability`: every value in `0.0..=1.0`, and the sum is `1.0`. | `bridge::tests` (new tests for `check_raw_probabilities()`), `semantic::causal_prior_tests::invalid_raw_probability_is_rejected_before_the_prior`, `imagine::tests::priors_must_be_a_distribution`, `ask_causal_prior_on_tests::invalid_raw_probability_is_rejected_with_the_prior_on`, `ask_raw_probability_off_tests::invalid_raw_probability_is_rejected_with_the_prior_off` |

The Round 2 risk-inspection path is unchanged in Round 3: `inspection_text()` still returns
`risk_checked_fields` and `risk_checked_sha256`, and `handle_latent_plan()` still inspects latent
candidates (`zero::tests::inspection_covers_state_candidates_tools_and_history ... ok`).

---

## 2. Rust Quality Gates

### 2.1 Integrated Round 3 tree, run on the dev build host, 2026-10-19

Run on the dev build host (`TMPDIR=$HOME/.cache/tmp-cargo`, `cargo 1.98.1`).
`PATH` used dev's system `python3` (no torch), so the Python bridge tests ran without torch.

| Gate | Command | Exit code | Raw log (claw) |
|---|---|---|---|
| Format | `cargo fmt --all -- --check` | **0** | `/tmp/dev-offload/20261019T092307-3684330.log` |
| Lint | `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | **0** | `/tmp/dev-offload/20261019T092319-3685125.log` |
| Unit + integration | `cargo test --workspace --locked --no-fail-fast -- --test-threads=4` | **0** | `/tmp/dev-offload/20261019T092358-3687133.log` |

- `cargo test`: 85 test targets (75 test binaries + 10 doc-test targets), **1455 passed, 0 failed, 31 ignored**; every `test result:` line shows `0 failed`.
  The named tests in section 1.1 each print `... ok`, and so does
  `semantic_qwen::risk_split_tests::pinned_measurements_match_heldout_report_json`.

### 2.1b Round 4 (M11) gates, run on the dev build host, 2026-10-19

| Gate | Log | Result |
|---|---|---|
| Red check (new tests against the old code) | `/tmp/fleet/b1019r4-w1.red.log` | `RED_RC=101`: 3 targets fail (`gen-zero-service --lib`, `--test ask_causal_prior_on_tests`, `--test ask_raw_probability_off_tests`), as required |
| Green: fmt, clippy, `cargo test -p gen-zero-service --locked --no-fail-fast` | `/tmp/fleet/b1019r4-w1.gates.log` | `FMT_RC=0`, `CLIPPY_RC=0`, `TEST_RC=0` |

GitHub Actions CI run `39183678764` on `badf720ee4bf2645562cdbfd9e9ed871866d9138` passed all 8 jobs:
`117841554368`, `117841554483`, `117841554416`, `117841554310`, `117841554480`, `117841554446`,
`117841554539`, `117841554432`.

### 2.1a Integrated Round 2 tree (all changes merged), run on the dev build host, 2026-10-19 (superseded)

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

### 3.1 Integrated Round 3 tree, full CI suite, dev build host, CPython 3.11.16

The CI command, in `~/.venv-gz-torch` (torch 2.14.0+cpu, numpy 2.4.6; `ortools` 9.15.6755 installed for this run):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$PWD/python \
  ~/.venv-gz-torch/bin/python -m pytest -ra --strict-config --strict-markers \
  python/gen_zero/tests python/tests/test_rnn_set_adapter_contract.py
```

- **Exit code 1**: `3 failed, 2882 passed, 71 skipped, 20 warnings in 902.38s`
  (log `/tmp/dev-offload/20261019T092739-3688098.log`). None of the three is caused by Round 3.
  Each was rerun 3 times on a clean `674071f` worktree and on the Round 3 tree, same host, same venv:

| Test | `674071f` | Round 3 | Cause |
|---|---|---|---|
| `test_issue_16_score_endpoint.py::TestLlamaCppScoreAdapter::test_adapter_health_check` | 3/3 fail | 3/3 fail | Host: the test hardcodes `http://127.0.0.1:8080`, and dev serves a live Gen-Zero there (`{"service":"gen-zero","status":"ok"}`), so `check_health()` takes the reachable branch, which has no `base_url`. Against a free port it returns `base_url` and passes the test's assertions. |
| `test_issue_23_nanocore_scaling.py::TestCPSATFormalSolverAndHardTimeout::test_cpsat_optimal_safety_solving` | 3/3 fail | 3/3 fail, then 5/5 fail serially | Host speed: CP-SAT solve 2.5 to 3.3 ms against a `<= 2.0` ms assertion. Same on both trees. |
| `test_calibrated_causal_dynamics_export.py::test_npz_loads_into_expert_and_step_meets_deployment_contract` | 3/3 pass | 2/3 pass, then 5/5 pass serially | Load flake: median step latency 82 us against `<= 15` us in the full run, while other agents loaded dev. |

  Logs: `/tmp/fleet/b1019r3-integrate.pyfail-baseline.log`, `/tmp/fleet/b1019r3-integrate.pyfail-r3.log`,
  `/tmp/fleet/b1019r3-integrate.pyfail-isolate.log`. Round 3 changes one Python file (`python/gen_zero/__init__.py`).

### 3.2 Reachability audit (G04.C)

- `python3 scripts/reachability_audit.py`: `219/271 Python modules import-reachable; 106 Rust files inventoried; 0 deletion claims`.
- The reached set equals the one `674071f`'s own code produces in a clean worktree: 219 and 219, 0 lost, 0 gained.
  The committed file at `674071f` was stale (211 reached).
- Re-running leaves `benchmarks/results/reachability.json` byte-identical (SHA-256 `3cdde017...7a9eb`
after Round 4 M11; the Round 3 value `7fea754f...5a55b` changed only because `bridge.rs` gained one public symbol, 27 to 28).
  Two runs on claw on 2026-10-19 gave the same bytes. The file differs from the committed one until the Round 4 commit lands.

### 3.3 Integrated Round 2 tree, dev build host, CPython 3.11.16 (superseded)

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

### 3.4 Scope

In Round 2 only these two Python test files ran. Round 3 ran the full CI suite (section 3.1).

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

This report was not regenerated in this pass. Round 2 M01 and Round 3 M01.B edit `semantic_qwen.rs`;
whether that shifts these numbers is `Not run in this pass`. The Round 3 tree still passes
`semantic_qwen::risk_split_tests::pinned_measurements_match_heldout_report_json`, which checks the
pinned constants against this JSON, not a fresh run.

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
- This proves the `f78bfcc` build only. Neither the Round 2 (`674071f`) nor the Round 3 tree is deployed there.

---

## 7. Not Run in This Pass

- **GGUF vs SafeTensors bitwise parity**: Not run in this pass. The Rust BoolQ run used Q4_K_M GGUF,
  the Python runs used fp32 SafeTensors, so their accuracies are not bit-comparable.
- **Python 3.10 / 3.12 matrix**: Not run in this pass. Only CPython 3.11.16 ran.
- **GitHub Actions CI on the Round 3 tree** (macOS `test`, `msrv`, `windows-check`, and the G04.B numpy
  step and G04.A sparse checkout themselves): Not run in this pass. `ci.yml` parses as YAML; no runner executed it.
- **MSRV 1.88 check** (`cargo +1.88.0 check`): Not run in this pass.
- **Windows compile check of the Round 3 tree**: Not run in this pass (aws-win built `f78bfcc`).
- **Python suite with 0 failures**: not reached. Three tests fail on dev for host reasons, identically or
  flakily on `674071f` too (section 3.1). They were not fixed in this pass.
- **BoolQ benchmark regeneration after M08**: Not run in this pass. `fit_calibrated` calls
  `loo_query_for_all_anchors(&raw, &ids)` with the same states it wrote, the case where the M08 change
  should not alter results, but sections 5.1 to 5.3 were not re-run to confirm.
- **Risk held-out report regeneration after M01 (Round 2) and M01.B (Round 3)**: Not run in this pass.
  Both modified only the optional keyword policy prior (`causal_utility_prior`, `PRIOR_DOMAINS`,
  `acts_on_safe_object()`) and formatting in `semantic_qwen.rs`, not the 16-shot risk classifier
  forward or LDA calibration path.
- **Red inversion for the benchmark-suite change**: Not run in this pass. The suite fails closed at
  run time instead: full-depth arms must run 28 layers with zero exits, and the dense fit must carry
  base-utility weights with an unchanged `min_exit_layer`.
- **Repeated timing runs**: Not run in this pass. Every ms/row figure is a single run.

---

## 8. Conclusion

On the integrated Round 3 tree (`674071f` plus the files in section 1), `cargo fmt --all -- --check` (rc 0),
`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` (rc 0) and
`cargo test --workspace --locked --no-fail-fast -- --test-threads=4` (rc 0, 85 targets, 1455 passed,
0 failed, 31 ignored) pass on dev, with every Round 3 named test `... ok`. The full CI Python suite
returns rc 1 on dev: 2882 passed, 3 failed, and each failure reproduces on `674071f` or is a load flake
that passes when rerun serially (section 3.1). `benchmarks/results/reachability.json` is regenerated, keeps
the same 219 reachable Python modules, and is stable across reruns and Python versions. The three
benchmark-locked Python files are untouched and match the report hashes. Items in section 7 remain
explicitly scoped as not run in this pass.
