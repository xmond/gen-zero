# End-to-End Operations Manual: Dense Model Features → Manifold Fusion → NanoCore Production Mount

Task ID: b0928-t6-docs. Date: 2026-09-28. HEAD `edb3d78`.

> **Removal notice, 2026-09-29:** The three-model (Qwen-72B + Llama-70B + Mistral-123B) Gaussian random-projection main script described and cited in §1, §4b, and “Commands and Evidence,” along with its 13-task result file, was removed in b0929u-t2. Its required 123B feature directory, `/ebs/data/extracted_features/mistral123b`, was a dangling symlink whose target never existed; the results were never independently reproduced. The related passages remain as a **historical record** of execution and the evidence available at the time, but their commands cannot be rerun today and must not support any current claim. The currently reproducible path without a 123B or 405B dependency is Qwen2.5-72B + LLaMA-3.1-70B dual-model fusion. See the [reproduction guide](../../benchmarks/README.md#13-task-sota-macro-8152-dual-70b-manifold-reproduction-guide) and [`scripts/download_benchmark_features.py`](../../scripts/download_benchmark_features.py).

**This manual records what can actually run and labels each stage with its observed status.** The proposed pipeline in the task title is “405B phase-transition truncation → GCCA multiview interference → 128-dimensional anchor basis → NanoCore production mount → refusal and permutation-equivariance acceptance.” Verification found repeatable, passing tests only for the **last two stages** (128-dimensional anchor to NanoCore mount, and refusal). The first two stages (405B truncation and GCCA multiview fusion wired into this anchor pipeline) are still designs or independent components. Permutation equivariance has **no test coverage** on the `nanocore_ask` entry point used by this pipeline; a different path, `decide` with `engine: nanocore` and an inline core, does have coverage. Results from one path cannot be credited to the other. §4b explains the distinction. The matrix below gives the criteria; each section and “Commands and Evidence” provide details.

The status legend follows [docs/zero/README.md](../zero/README.md#implementation-status-legend): **implemented** (code exists, runs, and has test or measured coverage); **experimental** (runs but was tested only on synthetic or small-scale data, or has a known numerical defect); **proposed** (design only, with no code or no invocation from a real-data path).

## Stage Status Matrix

| Stage | Proposed task-title behavior | Actual status | Evidence |
|:---|:---|:---|:---|
| 0. Feature source | Extract phase-transition layers from 405B/180B/123B models | **Proposed.** The GGUF byte-level slicing tool `scripts/slice_gguf_layers.py` exists and has unit tests, but no script calls it to drive extraction. No real extraction from any of the 123B, 180B, or 405B models has occurred. The only teacher with a real artifact here is LLaMA-70B, which is **not** in the task title's model list; it uses ordinary final-layer extraction without the slicing tool. | [Status matrix in `docs/zero/31-...md`](../zero/31-dense-fleet-manifold-anchor-t4-sys-design.md#status-comparison-matrix) |
| 1. GCCA multiview fusion | Interfere views from multiple models on a manifold | **Implemented as a standalone CLI (`cli.py manifold-fuse`), but disconnected from evaluation and the anchor pipeline.** `GCCAMidFusion` in `python/gen_zero/manifold/gcca_fusion.py` is a real, unit-tested regularized MAX-VAR GCCA implementation called by the `manifold-fuse` CLI subcommand at `cli.py:610-627`. No evaluation script, manifold-anchor pipeline, or Rust production path calls it. A former three-model Gaussian random-projection script produced a 13-task classification result file using `sklearn.random_projection.GaussianRandomProjection`; it **neither imported `gen_zero.manifold` nor called `gcca_fusion`**. Its numbers cannot be called “GCCA validation.” The script and its 123B dependency were removed, as described above. | §1 |
| 2. 128-dimensional anchor basis | GCCA fusion output → 128-dimensional anchor | **Implemented for one model, without GCCA input.** `ManifoldAnchorDistiller` in `python/gen_zero/causal/manifold_anchor_distiller.py` applies an orthogonal projection to **raw LLaMA-70B hidden features**. It produced a real SHA256-checked `.npz` artifact and was verified end to end on one real BoolQ record. | §2 and `crates/gen-zero-service/tests/test_nanocore_live.rs` |
| 3. NanoCore production mount | Feed a 128-dimensional vector to the production decision component | **Integrated, but the risk classifier driving the tested decision is a test stub.** `nanocore_ask` (`crates/gen-zero-service/src/zero.rs:2246`), fail-closed `validate_nanocore` (`zero.rs:446`), and real end-to-end tests all passed when rerun for this revision. The precise claim is “feature projection and Rust engine integration were tested using a stub risk classifier” with fixed `p_dangerous=0.01` (§3). | §3; command below |
| 4a. Refusal acceptance | Invalid input must be rejected without silent degradation | **Implemented.** NaN, float32 overflow, and 127- and 129-dimensional inputs were all rejected with an explanatory assertion. | §4 |
| 4b. Permutation equivariance | Candidate order must not affect decisions | **Partially verified across distinct entry points.** The `decide` + `engine: nanocore` inline-core path through `specialized_ask` has a permutation-equivariance test at `zero.rs:4471` (`specialized_scores_are_exactly_invariant_to_candidate_order`). The `nanocore_state` path used in §3 goes through a separate `nanocore_ask` entry point (`zero.rs:2246`) and has no such coverage. The removed three-model projection script also had a `shuffled_candidates` stage with a different meaning (§1); it cannot prove equivariance for either entry point. | §§1, 4 |

---

## 0. 405B Phase-Transition Truncation: Proposed, Not a Production Dependency Today

Do not skip directly to §§1–3. Seeing “128-dimensional anchor → NanoCore runs” could suggest that the entire pipeline, including its 405B front end, is connected. It is not:

- `scripts/slice_gguf_layers.py` can truncate a GGUF file by layer, retaining `blk.0..K-1` and removing `output.weight`. Unit tests cover atomic writes and header parsing. It is a build tool, **not an integrated extraction pipeline**: repository-wide `grep -rln slice_gguf_layers .` found only the script and its tests, with no `.bat`, `.sh`, or orchestration script invoking it.
- “Milestone 0” in `docs/zero/31-dense-fleet-manifold-anchor-t4-sys-design.md` (numerical repair of `StreamingCovarianceAccumulator`, removal of `argmin` from `detect_phase_transitions`, and renaming “online SVD”) **remains incomplete**. `git log -S"StreamingCovarianceAccumulator" -- python/gen_zero/causal/universal_manifold_extractor.py` showed no commit changing that logic since its introduction. Scanning phase-transition layers and measuring quantization fidelity for 123B/180B/405B models are downstream of that milestone and have not started.
- **Conclusion:** For a 405B integration goal, consult the original `docs/zero/31-....md` design and its status matrix. None of the artifacts below came from 405B; they came from LLaMA-70B.

## 1. GCCA Multiview Fusion and the Removed 13-Task Numbers Are Unrelated Components

**Historical correction (the evaluation script was removed on 2026-09-29; this describes its state before removal): the three-model random-projection main script was not an end-to-end GCCA evaluation and never called GCCA.** It imported `sklearn.random_projection.GaussianRandomProjection`, projected hidden features from Qwen-72B, Llama-70B, and Mistral-123B into the same 256-dimensional space, concatenated them, and added a candidate prior (`candidate_prior`), graph-Laplacian regularization (`graph_laplacian`), and log-linear pooling (`log_linear_pool`) for a closed-form solve. It once produced a result file for 13 classification tasks (massive/multinli/pubmedqa/boolq/paws/squad2/arc_challenge/vitaminc/civil_comments/aegis_safety/helpsteer2/summeval_relevance/summeval_consistency). Every number came from that **Gaussian random-projection multiview baseline**, not GCCA. The task fused three models' scores and features for the same candidate answers; it did not fuse one model's hidden states across layers or truncation points. The script and results were removed with the 123B dependency and cannot be rerun. The following details are historical only.

`GCCAMidFusion` in `python/gen_zero/manifold/gcca_fusion.py` is the actual regularized MAX-VAR GCCA implementation. It has unit-test coverage (including 60 related tests in `gen_zero/tests/test_gcca_fusion.py`) and is called by the `manifold-fuse` CLI subcommand at `python/gen_zero/cli.py:610-627`. **No evaluation script, manifold-anchor pipeline, or Rust production path calls it.** This remains true after the former script's removal because it never depended on GCCA. Exports are in `python/gen_zero/__init__.py` and `python/gen_zero/manifold/__init__.py`.

Rerun `gcca_fusion`'s own unit tests, offline and independently of the removed 13-task numbers:

```bash
cd python
python3 -m pytest gen_zero/tests/test_gcca_fusion.py gen_zero/tests/test_candidate_prior.py \
  gen_zero/tests/test_manifold_master_objective.py -q
```

As recorded below, all 60 passed. **They test the `gen_zero.manifold` package (`GCCAMidFusion`, `CandidateSemanticPrior`, `MasterClosedFormSolver`), not the removed script's inline `fit`, `pool`, or `graph_gram` functions.** The two codebases are independent; the 60 passes do not establish unit-test coverage for the removed script.

The removed script once supported `--smoke` for a quick check on the first task; omitting it ran all 13. The command no longer exists and must not be attempted today.

**`shuffled_candidates` did not prove permutation equivariance.** Historically, the removed script shuffled candidate order once with a fixed seed for each view, then rescored using `(True, True, True)` (`candidate_prior` + `graph_laplacian` + `log_linear_pool`). That configuration was **not** the solver used for the `master_solver` column. Subtracting the columns does not measure an order effect: for example, the published `paws` values were 93.20 and 91.60, respectively, because the columns used different solvers. The only valid observation was the `log_linear_pool` configuration's performance after shuffling, represented by the `shuffled_candidates` column. It does not generalize to `master_solver`, much less to §3's separate `nanocore_ask` path.

The `ManifoldAnchorDistiller` docstring calls it “offline conformal compression of GCCA features” (`manifold_anchor_distiller.py:1`), but that is merely one intended input type. Repository-wide `grep -rn "ManifoldAnchorDistiller(" .` found only two calls to `.fit()`: its own CLI (§2) and a **synthetic random-data self-check** at `profile_nanocore_latency.py:132`. The latter function explicitly says “Throwaway artifact fit from random data; checks the harness, not real latency.” **No script feeds GCCA output into `ManifoldAnchorDistiller.fit()`.** The real artifact in the next section bypasses GCCA entirely.

## 2. 128-Dimensional Anchor Basis: Real Artifact from Raw LLaMA-70B Features, Not GCCA

The delivered, hash-checked artifact is `benchmarks/results/manifold/distilled_128d_llama70b_boolq.npz` (`git ls-files` confirmed it is tracked). It orthogonally projects raw LLaMA-70B hidden features from `/ebs/data/extracted_features/llama70b/boolq.npz`. That feature file is local to one machine and outside the repository; `test_nanocore_live.rs` checks its existence and content hash when run. **GCCA is not involved.**

Fit an anchor basis from scratch, replacing `--features` with your own feature file:

```bash
cd python
python3 -m gen_zero.causal.manifold_anchor_distiller \
  --features /path/to/your_features.npz --block train_full \
  --output-dim 128 --out /tmp/my_anchor.npz
```

The `.npz` given to `--features` must contain a two-dimensional matrix of finite values under the key selected by `--block` (default `train_full`). `ManifoldAnchorDistiller.fit` uses thin SVD to obtain the orthogonal projection `P` (`P @ P.T == I`). It fits once and does not update online. Refit it after the upstream feature distribution changes: this module does not detect drift (`manifold_anchor_distiller.py:9-16`).

Rerun offline fitting and projection-bridge correctness tests:

```bash
cd python
python3 -m pytest gen_zero/causal/tests/test_nanocore_bridge.py \
  gen_zero/causal/tests/test_manifold_anchor_distiller.py -q
```

All passed in the recorded run below.

## 3. NanoCore Production Mount: Repeatable End-to-End Integration

This is the only stage in the manual with a full recorded run and assertions from raw features to a production decision entry point:

```text
Real LLaMA-70B BoolQ hidden features (on-disk .npz)
  → ManifoldAnchorDistiller.project() (128-dimensional orthogonal projection, SHA256 checked)
  → NanocoreAnchorBridge.generate_mcp_ask_payload() (constructs an MCP ask payload)
  → nanocore_ask (Rust, zero.rs:2246)
  → validate_nanocore + NanoCoreFleetScheduler + MoVFusionEngine
  → decision + confidence + chosen_action
```

`crates/gen-zero-service/tests/test_nanocore_live.rs` establishes the integration in two ways. First, it loads a previously generated fixture, `tests/fixtures/nanocore_anchor_state_boolq_row0.json`, and passes it directly to `nanocore_ask`. More importantly, it spawns a real `python3` subprocess with `std::process::Command`, reruns `NanocoreAnchorBridge.project_to_nanocore_state()`, and asserts that its 128-dimensional vector matches the fixture byte for byte. This catches a stale fixture after code changes.

**Precise claim: feature projection and Rust engine integration are tested with a stub risk classifier. This does not validate risk-classification capability.** The two success paths, `real_projected_state_decides_between_two_candidates` and `real_projected_128d_state_drives_a_real_nanocore_decision`, must pass the shared risk gate to obtain `is_error=false`. The test substitutes a local HTTP `stub_scorer` (`test_nanocore_live.rs:71-88`) for the semantic risk-scoring backend and returns fixed `p_dangerous: 0.01` and `classifier.name: "test-stub"` (`test_nanocore_live.rs:76,78`). This keeps the request at `Tier0Proceed` rather than escalating at the gate. Without a configured real risk backend, the fail-closed rule in `crates/gen-zero-gate/src/risk.rs` would escalate to `Tier2Escalate`. The test establishes that a 128-dimensional vector reaches `nanocore_ask` and drives a decision; it measures no real risk model's accuracy.

**Production mount procedure:** The [“Integrated Rust subsystems” README section](../../README.md#integrated-rust-subsystems) describes the existing mechanism. To turn the anchor artifact into a `NanoCoreInstance` file, export the fitted basis and `projection_weights`/`value_weights` trained on supervised `(target_128d, decision_label)` pairs as JSON fields `domain_id/name/prototype/projection_weights/value_weights/out_dim/base_confidence`, then place the file at a path in `GENZERO_NANOCORE_PATHS`. **No script currently trains `value_weights` on real decision labels.** In `test_nanocore_live.rs`, `value_weights` simply reuses `prototype` via `prototype.clone()` in fixture loading. This is a test fixture simplification, not a fitted decision head. Before using this integration in a real business setting, refit `value_weights` on historical decisions from that setting. Otherwise, an online decision is only dimensionally valid and has no semantic grounding. Step 4 of §2.2 in `docs/zero/31-...md` had already identified this gap; the present run confirms it remains.

**Python `decide_nanocore` and Rust `nanocore_ask` are separate decision implementations:**

- `cli.py anchor --execute` calls `GenZero.decide_nanocore()` (`python/gen_zero/client.py:3230`), a **Python in-process** decision path using the Python `ActionETFChoiceHead` and registered core. It never calls the Rust service.
- The Rust decision engine is invoked through `nanocore_ask` (`crates/gen-zero-service/src/zero.rs:2246`). It accepts a JSON-RPC request, passes `validate_nanocore`, and scores inside Rust. The only current connection between Python feature projection and Rust `nanocore_ask` is the Rust integration test `test_nanocore_live.rs`, which computes features in a subprocess and puts them in the request payload. No cross-process production wrapper invokes it yet.

Also, **no deployment script currently sets `GENZERO_NANOCORE_PATHS`**: `grep -rn GENZERO_NANOCORE_PATHS --include=*.sh --include=*.bat --include=Makefile -r .` had no matches. Mounting this pipeline in a running `gen-zero serve` process requires new deployment configuration for that variable. It is not handled by an existing script.

## 4. Acceptance: Refusal and Permutation Equivariance

### 4a. Refusal (Implemented; Four Invalid Inputs Tested)

`non_finite_or_wrong_length_state_is_refused_fail_closed` (`test_nanocore_live.rs:399`) injects NaN, float32 narrowing overflow (`1e39` is finite in f64 but becomes `+inf` in f32), and lengths 127 and 129 into the same real fixture. All four yield `is_error=true`, with error text containing `"nanocore_state must contain 128 finite numbers"`. No silent truncation or zero padding occurs. `unregistered_domain_is_refused_fail_closed` separately tests refusal of an unregistered domain.

Rerun:

```bash
cargo test -p gen-zero-service \
  --test test_nanocore_live --test provenance_nanocore_integration --no-fail-fast -- --nocapture
```

The recorded run below passed all eight tests (5 + 3).

### 4b. Permutation Equivariance (Partial Verification; Distinct Entry Points)

`PolymorphicZeroEngine` has two different NanoCore scoring paths with different coverage:

- **`decide` + `engine: nanocore` (inline core through `specialized_ask`, `zero.rs:2510`): verified.** `specialized_scores_are_exactly_invariant_to_candidate_order` (`zero.rs:4471`) reverses candidate order and asserts that each candidate retains its own score across the `nanocore`/`generic` backends, `etf`/`linear` heads, and 2/3-candidate cases. Its request carries `nanocore_core` (an inline `NanoCoreInstance`) and `decision_state`; it does not use the `nano_fleet` registry.
- **The separate `nanocore_ask` entry point used by `nanocore_state` (`zero.rs:2246`, with a registered core from `nano_fleet.get_core`): unverified.** This is the path used by §3. `grep -rn "permut\|reorder\|shuffle" crates/gen-zero-service/tests/*.rs crates/gen-zero-service/src/zero.rs` found no candidate-order-shuffling assertion on this entry point. This revision neither statically analyzed nor added tests to establish whether `MoVFusionEngine` (`crates/gen-zero-nanocore/src/mov.rs`) scores each candidate independently of list position. The `specialized_ask` test cannot cover a different function with different request shapes (`nanocore_core` + `decision_state` versus `nanocore_domain(s)` + `nanocore_state`). A test that submits two permutations of the same candidates and checks their individual scores from `nanocore_ask` remains future implementation work outside this documentation task.

The removed script's `shuffled_candidates` stage (§1) likewise **cannot** fill this gap. It measured classification accuracy for a shuffled `log_linear_pool` configuration on 13 tasks. Its measured property, code path, and task differ from both NanoCore entry points. Treating it as evidence of permutation equivariance here would conflate distinct claims. The script has been removed; this paragraph is historical context only.

---

## Commands and Evidence

The commands below actually ran during this revision (HEAD `edb3d78`, worktree `/tmp/fleet-wt/b0928-t6-docs`). Original exit codes and output tails are reproduced.

**Rust: NanoCore end-to-end integration and refusal tests**

```text
$ cargo test -p gen-zero-service --test test_nanocore_live --test provenance_nanocore_integration --no-fail-fast -- --nocapture
...
     Running tests/provenance_nanocore_integration.rs
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
     Running tests/test_nanocore_live.rs
test real_projected_state_decides_between_two_candidates ... ok
test unregistered_domain_is_refused_fail_closed ... ok
test real_projected_128d_state_drives_a_real_nanocore_decision ... ok
test non_finite_or_wrong_length_state_is_refused_fail_closed ... ok
test fixture_state_matches_a_fresh_run_of_the_real_python_bridge ... ok
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.28s
EXIT:0
```

**Python: GCCA fusion and master-objective unit tests (60 cases)**

```text
$ cd python && python3 -m pytest gen_zero/tests/test_gcca_fusion.py gen_zero/tests/test_candidate_prior.py gen_zero/tests/test_manifold_master_objective.py -q
............................................................             [100%]
60 passed in 6.13s
EXIT:0
```

**Python: anchor distiller and bridge unit tests**

```text
$ cd python && python3 -m pytest gen_zero/causal/tests/test_nanocore_bridge.py gen_zero/causal/tests/test_manifold_anchor_distiller.py -q
79 passed, 9 warnings in 68.25s (0:01:08)
EXIT:0
```

All nine warnings came from tests named `test_*_rejects_*overflow*` or `test_*_rejects_*energy_overflow*`. They deliberately construct pathological input (extreme values and crafted projection matrices) to exercise refusal paths, producing NumPy `RuntimeWarning` messages such as `invalid value encountered in scalar divide`, `overflow encountered in matmul`, and `overflow encountered in subtract`. They were not test failures or evidence of silent production fallback. The tests assert that `ManifoldAnchorDistiller` raises instead of returning an incorrect number on these inputs.

**Python: Gaussian random-projection multiview baseline on 13 tasks, not GCCA (historical; script and results removed on 2026-09-29, so the command cannot be rerun)**

```text
$ python3 <removed script> --smoke --output /tmp/smoke_manifold
Task                        baseline candidate_pr graph_laplac log_linear_p master_solve shuffled_can
massive_en                     88.86        89.14        89.14        89.43        89.14        89.43  10.97s
MACRO                          88.86        89.14        89.14        89.43        89.14        89.43
WROTE /tmp/smoke_manifold.json /tmp/smoke_manifold.md
EXIT:0
```

At the time, a fresh run's `massive_en` single-task numbers matched the then-committed 13-task result file. That result file and its complete 13-task macro results have since been removed with the script. This is historical evidence, not a reproducible command today. For a reproducible path without a 123B dependency, see the dual-model 13-task SOTA Macro 81.52% [reproduction guide](../../benchmarks/README.md).
