# P2 delivery evidence

Implemented on `feat/b1001d-p2-reflex`. This is graph-based candidate exclusion and observation-driven epistemic evolution, not a newly trained model or a measured planning-quality breakthrough.

## Implemented and verified

- `crates/gen-zero-lod/src/graph.rs:2088`: `planning_prior` checks revocation, confidence < 0.3, falsified causal successors and coarse/parent hierarchy. It reads CSR and pending edges under one read lock. Cycles terminate using a visited set; returned facts have deterministic ordering. The repository's actual macro/global bands are `Lod2Milestone` and `Lod3Systemic`.
- `crates/gen-zero-planner/src/pipeline.rs:1143`: prior checks enter production candidate pruning, rollout gate checks and final decision certification. Original PolicyGate revocation rule IDs are preserved. Timeout incumbents are rechecked against the live graph before return.
- `crates/gen-zero-planner/src/pipeline.rs:1052`: graph context includes high-confidence hierarchical facts, band and confidence; `used_in_gate` distinguishes these from advisory PPR facts. PPR scores are not planning reward estimates.
- `crates/gen-zero-service/src/pipeline_verb.rs:85`: strict boolean `auto_reflect`, default false, for pipeline `simulate`, `what_if`, `audit_action`; false is explicitly reported. The shared service entry is used by HTTP, MCP and CLI. No alternative backend or adapter was added.
- `crates/gen-zero-service/src/pipeline_verb.rs:261`: actual hazardous transition (including continuation action), step, before/after states, reward and safety-source/calibration metadata become evidence; policy hard stops are also deposited. No innocent first action is blamed for its continuation's failure.
- `crates/gen-zero-lod/src/graph.rs:2147`: one write lock covers staged evidence, a directed Falsifies edge and the existing SCC solver with beta=0.85, gamma=1, tolerance=1e-6, thresholds 0.3/0.6. Timestamp is stored in the evidence node. Action+payload hashes deduplicate observations; collisions/missing edges error. Solver failure or axiomatic conflict returns an error and manually quarantines the target; no partial evidence is published or convergence invented.
- `crates/gen-zero-service/tests/pipeline_service_tests.rs:631`: real default LatentDynamicsWorldModel through HTTP; valid action -> terminal simulation -> evidence/edge/convergence/revocation -> subsequent decision excludes it and simulation rejects it. Further tests cover what_if, audit_action, pending causal edges, macro context, deterministic repeated decisions, opt-out/type validation and explicit conflict errors.
- `crates/gen-zero-lod/tests/reflection_tests.rs:16`: concurrent duplicate deposition, noncontractive evolution and axiomatic conflicts. No new mocked dynamics or hard-coded simulated responses.

## Commands and raw results

Remote source-only sandbox: `worker-node-1:/tmp/b1001d-p2-reflex-src`, no `.git`. Source SHA-256 manifest compared byte-for-byte with the local workspace (cmp exit 0). Dependency resolution is saved as `dependency-lock.txt`; the repository itself ignores Cargo.lock.

Health checks used `hostname; uptime; nproc; free -m; df -Pm /tmp; command -v cargo` across available compute nodes. Node worker-node-1 had ~98GB available RAM and ~948GB disk and was selected. Cargo required its explicit `~/.cargo/bin` PATH. A later environment capture is in `remote-environment.txt`.

```sh
cd /tmp/b1001d-p2-reflex-src
PATH="$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS=$(nproc) \
  CARGO_TARGET_DIR=/tmp/b1001d-p2-reflex-target \
  cargo test -p gen-zero-lod -p gen-zero-planner -p gen-zero-service
```

Final original exit code: **0** (`cargo-test.exit`). Full stdout/stderr: `cargo-test.txt`. Sum across 31 test-result records: **625 passed, 0 failed, 6 ignored**. Tail:

```text
   Doc-tests gen_zero_service
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo fmt --all -- --check`: exit **0**, raw empty stdout/stderr in `p2-fmt-check.txt`, code in `p2-fmt-check.exit`. `git diff --check` before staging evidence: exit **0**. After adding byte-exact raw logs, `git diff --cached --check` exits **2** solely for Cargo's trailing blank line in `cargo-test.txt:825` and `b1001d-p2-deadline.txt:41`; raw logs are intentionally preserved without normalization.

Failures retained rather than hidden:

1. Initial test exit **101** (`first-failure.*`): graph prior returned before the original revocation rule ID was recorded. Implementation corrected; existing rule-ID assertion unchanged.
2. A subsequent full run exit **101** (`timing-failure.*`): deadline assertion measured 31.959119ms against 30ms. Isolated unchanged test passed with ~2.0–2.2ms across modes (`b1001d-p2-deadline.*`, exit **0**). Final unchanged full command passed. No timing assertion was weakened or test disabled.
3. Initial rsync included the absent/ignored local Cargo.lock and returned 23. Source files transferred; later sync used only existing Cargo.toml/crates and source manifests verified exact equality. The remote-generated lock is preserved for reproducibility.

The prior PPR display test changed a falsified neighbor's relation from CausalTransition to Semantic because causal successors must now reject the action. A separate real HTTP test explicitly verifies the stronger causal rejection and low-confidence exclusion; no security assertion was removed.

## Not verified / limits

- Six pre-existing ignored semantic-bridge tests require a live Python semantic scorer; this task did not provide/run that dependency. They remain unchanged (`crates/gen-zero-service/tests/semantic_bridge_e2e.rs`).
- No live deployment, production load/latency benchmark, new model training or paired planning-quality evaluation. The timing failure prevents claiming hard real-time guarantees.
- Reflection is opt-in on the pipeline verbs above. Standalone worldsim verbs have no new reflection option; direct ProductionPipeline simulation without the service wrapper does not deposit automatically. Revocations reside in the current LodGraph; restart durability is not added or claimed.
- Quarantining a model-observed hazardous action is a conservative global policy choice. Its evidence says model diagnostic, not proof of real-world causal invalidity. Recovery of manual quarantine requires an explicit graph-management decision.

## Audit scope

Production call sites and HTTP tests prove non-isolation. All new failures return typed errors or explicit disabled metadata; no fallback search/backend was added. No prior algorithm/module was replaced, so there are no obsolete module names or compatibility shims to retain. The SCC implementation was extracted into one shared locked helper, not duplicated. No reviewer/subagent, stash, checkout, reset, clean or force push was used. Only this task's code, tests and evidence are committed; deployment and push are not performed by this delivery.

Remote source sandbox, build directory and task temporary logs were removed after artifact/hash verification (cleanup exit 0).
