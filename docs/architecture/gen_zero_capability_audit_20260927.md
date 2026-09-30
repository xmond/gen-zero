# Gen-Zero Capability Boundaries, 8B Attribution, and Improvement Path Investigation

Investigation date: 2026-09-27. Source baseline: `6dfd6099591f68a8bba8bbb8ae3b2a6c302f7e2c`, working directory `/ebs/pj/gen-zero`. The working tree was clean at the start. Line numbers in the body refer to this baseline and can be verified with `git show 6dfd609:<path> | nl -ba`; other work merged into `701d800a5d3c55db6dcdb002b7bda5608a6bba11` during the investigation — see the delta note at the end — so old line numbers in this document must not be read against the new version. This investigation only adds a report and evidence; it does not modify production implementation, and does not commit or run stash/checkout/reset/clean/force-push.

## Conclusions Up Front

1. The user's judgment is correct: the “8B” in the original claim refers to the external CLM's frozen Qwen3-8B encoder, not a Gen-Zero in-house 8B scoring head. Strictly speaking, CLM itself is not “an 8B-parameter projection head” either, but an 8B encoder plus two smaller projection heads.
2. Gen-Zero's reasonable positioning is a decision-control system that connects candidate generation, state modeling, constrained search, risk decisions, and execution verification. The current code has the relevant algorithmic building blocks, but this does not support claiming a general-purpose terminal world model, a safety proof for arbitrary shell commands, or a 90%+ solve rate.
3. The original claim omits the most important negative evidence: an existing enhanced-PRM experiment shows BoN=4 dropping from a 31/38 baseline to 29/38, fixing 2 problems while regressing on 4, with the geometric branch not winning. Continuing to tune on these same 38 problems until satisfied can no longer count as an independent held-out validation.
4. The most worthwhile direction to invest in is a closed loop of “real state -> candidates -> controlled execution -> observation -> replanning,” together with a replayable, rejectable, non-bypassable execution gate. Stacking more planning algorithms or adding geometric features to an external scorer cannot substitute for this chain.
5. “A 10%-15% gain on the same base model” and “pushing for 90%+” can only be listed as targets yet to be verified, not as known gains. The existing evidence has not even established an 80% live baseline for this system under the same protocol.

## 1. Item-by-Item Verdict on the Original Claims

| Original claim | Investigation verdict |
|---|---|
| The 8B scoring head belongs to Gen-Zero | False. The external CLM's encoder is 8B; the head is a different, much smaller size. Gen-Zero has its own small scoring/decision modules, and can also consume representations from an external large model. |
| CLM is just a “weak scoring head” | Imprecise. It is a non-generative state-action scoring system; 0.5196 is the pooled AUC on a specific local held-out trajectory set, and is not enough to summarize the model's capability across all tasks. |
| Fixing 2 more problems gives 86.8%, which beats 81.6% | The arithmetic 33/38 is correct, but the engineering and statistical conclusion does not hold; newly introduced regressions must be subtracted, candidates and budget must be controlled, and an unseen test set must be maintained. |
| Geometric features that smooth variance improve results | An untested hypothesis, and there is already one counterexample: the enhanced head scores 29/38. Geometric smoothing must not be equated with correct ranking. |
| Harbor/Pier, containers, and adapters are 100% ready | This can only be stated in layers, by environment, version, specific task, and evidence; a full production pipeline and solve rate cannot be inferred from smoke or boundary tests. |
| ILP prevents accidental deletion and infinite loops | This can cover the specified risks only when action and resource semantics are correctly encoded, rules are registered, and the executor cannot be bypassed. Linear constraints themselves do not understand shell semantics, and do not prove termination of an arbitrary program. |
| Genuine multi-step MCTS already improves coding success rate | The algorithm and the actual call chain must be verified separately; tree depth does not equal correct prediction of environment state, and semantic value does not equal a true reward. |

The official CLM model card explicitly states frozen Qwen3-8B + state/action heads, and notes it has no generation capability; the verifier metrics come from the fine-tuned head, not from the zero-shot performance of the public general-purpose checkpoint: [official model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B). A snapshot captured for this investigation is at `docs/benchmarks/evidence/gen-zero-audit-20260927/clm-official-model-card.md`. These figures are retained as the other party's stated claim and are not upgraded into an independently reproduced end-to-end result of ours.

## 2. Real Bottlenecks Exposed by the 38-Problem Experiment

Historical results are at `benchmarks/results/deepswe_prm_evidence/RESULT.md:1`, machine-readable results are at `benchmarks/results/deepswe_prm_enhanced_results.json`, and the failing acceptance command plus exit code are at `benchmarks/results/deepswe_prm_evidence/acceptance.json`.

| Metric | Original CLM baseline | Enhanced head |
|---|---:|---:|
| BoN=4 | 31/38 = 81.5789% | 29/38 = 76.3158% |
| Trajectory-level AUC | 0.519595 | 0.413964 |
| Trajectory-level Spearman | 0.029954 | -0.131521 |
| Per-problem change vs. baseline | — | Fixed 2, regressed 4, net -2 |

This shows that “getting two more correct choices” is not a net improvement. AUC compares across all trajectories, while BoN selects among candidates within the same task; the two measure different things. A low pooled AUC can coexist with a higher within-task BoN, so one cannot infer from this that “adding a few features casually will beat the baseline.”

`RESULT.md:38` records that for 4 tasks, all candidates fail: based on the candidates currently available per problem (the released/up-to-4 protocol, where the evaluation code uses min(4, number of candidates), not strictly 4 per problem), the pure-reranking ceiling is 34/38 = 89.47%. So within this candidate pool, even a perfect scorer cannot reach 90%. Exceeding this ceiling requires improving candidate generation, execution feedback, and repair capability.

The sample size is also too small to support a claim of “beating”: the Wilson 95% intervals for 31/38 and 33/38 are approximately [66.58%, 90.78%] and [72.67%, 94.25%]. Even with a paired comparison of 2 wins to 0 losses on the same problems, a two-sided exact sign test / McNemar test gives p=0.5. This is a calculation illustrating the limits of a small sample, not a measurement of a new model.

The training evidence describes 75 training tasks, 298 trajectories, and 30,372 steps; step count is not the same as an independent supervised sample count. The historical cross-validation winner was the semantic kernel; the geometric branch did not win, and neither the 8B backbone nor the original MLP was fine-tuned (`RESULT.md:12,23`; also `REPRODUCE.md` in the same directory).

Reproduction boundary: this investigation re-ran `python3 benchmarks/tests/verify_deepswe_prm_report.py --report benchmarks/results/deepswe_prm_enhanced_results.json --reference-repo /tmp/clmrepro/repo`, which exited 1 because the current working tree is missing `benchmarks/artifacts/deepswe_prm/enhanced_rank_head.pt`. So the table above reflects the archived report and its JSON record consistently, not results freshly retrained/re-inferred in this investigation; full weight provenance verification was not completed. No missing check was skipped in order to claim a successful reproduction.

## 3. PolicyGate's Real Capabilities and Issues That Must Be Fixed First

Rust's `LinearConstraint` represents `sum(c_i*x_i)<=rhs`, evaluated over known ActionIds and an active_context; see `crates/gen-zero-gate/src/constraint.rs:13,99`. This is constraint checking, not an automatic translation of an arbitrary shell program into a provably safe ILP. `PolicyGate::default()` has empty rules, confirmation actions, and heat requirements (`policy.rs:43`); the service's default construction also uses it (`crates/gen-zero-service/src/zero.rs:756`). Returning `formal_checked` alone is not sufficient grounds to conclude that the rules cover real filesystem risk.

There are rigorous parts worth keeping: rejection on non-finite entropy/threshold (`policy.rs:130`); rejection when a registered heat requirement lacks a certificate (`:179`); semantic risk that is missing or illegally escalated is rejected rather than passed (`risk.rs:29`); the service only lets Tier0 produce a normal success, treating confirmation/escalation as error results (`zero.rs:1797`). Sheaf terminal verification recomputes the residual and energy rather than blindly trusting a diagnostic field (`sheaf_gate.rs:395`). But what this proves is numerical terminal-state compliance for the given problem and snapshot, not semantic correctness of the candidate program, let alone a general software safety proof.

The Python `DecisionPolicyGate` is a separate rule/confidence gate, and must not be conflated with the same guarantees as the Rust implementation above. This investigation called the real functions directly and obtained the following counterexamples:

- When `{}` is missing action, risk, and confidence, it defaults to confidence=0.5, risk=0, and ultimately PROCEEDs (`python/gen_zero/gate/policy_gate.py:139,266`).
- For a target explicitly added to the whitelist, with action=`delete`, confidence=0.99, risk=1.0, it still PROCEEDs (`:194,219`).
- risk=NaN is first coerced to 1.0, but is likewise let through by the whitelist branch (`:152,194`).

These are pure-function boundary counterexamples; no deletion was executed, and it was not shown that the TB adapter goes through this particular Python path. This local defect must not be inflated into “all Rust/TB gates are broken.” Still, it clearly conflicts with the system-wide Fail-Closed commitment. The reproduction script and its output are in this report's evidence directory as `gate_probe.py` and `gate-counterexample.txt`.

Another success-semantics gap: `python/gen_zero/runtime/loop_state_machine.py:260` only performs acceptance verification when `verify_fn is not None`, upon the model's claim of finish/done; without a verifier it still returns SUCCESS and “verified” (`:283`). The correct boundary should distinguish agent_finished, unverified, and verified_success. The TB adapter already uses verifier_pending (`benchmarks/gen_zero_tb_adapter.py:330`), and this is worth unifying across other execution entry points.

Remediation priority: input schema/required-field validation -> reject when policy is missing or the action is unrecognized -> whitelist must not override unknown/illegal risk and hard prohibitions -> unify the gate verdict and execution-authorization contract -> separate the verification conclusion from the completion declaration. Program timeout / process-tree termination / resource limits are the executor's mandatory constraints, and must not rely solely on the model predicting “it won't infinite-loop.”

## 4. How to Go from 80% to 90%: First Build a Testable Improvement Loop

Everything below is a proposal and acceptance design, not an implemented capability or a guaranteed gain.

**Fix the comparison target first.** The same Proposer checkpoint/API version, prompt, tool permissions, candidate count, token budget, wall-clock/cost budget, dataset version, and run seed. The baseline should be the standard execution loop on the same base model; also set up an equal-budget best-of-N/multi-retry baseline, to avoid crediting gains from extra sampling to Gen-Zero's search. Percentages from different DeepSWE/Terminal-Bench versions, different trajectory pools, and different generator models cannot be directly subtracted from one another.

**Build a minimal real closed loop.** The Proposer only generates candidates (commands or patches, plus their expected effect); it grants no safety privileges and does not decide evaluation success. Gen-Zero manages observation, candidate deduplication and branch diversity, feasibility checks, search budget, branch selection, and replanning. The executor performs actions in an isolated environment and reports back the file diff, exit code, test results, and process/resource state. Final success is adjudicated by an isolated benchmark verifier. Hidden acceptance tests must not leak to the agent as a candidate-scoring oracle.

The proposed state should at minimum include environment/repository snapshot identifier, file changes, running processes, observed test results, tool permissions, and remaining budget. All transition caches should be keyed by snapshot + action + environment configuration; nodes without a trustworthy transition should be explicitly marked unknown, rather than backfilling the tree with fabricated success rewards. Sandbox trial runs should only happen in an authorized, isolated copy, not achieve rollback by resetting the shared working tree.

**Measure candidate coverage first, before training a value network.** For a fixed candidate set, measure oracle success@K, actual selection success rate, and the gap between the two (the selection loss); then measure the recovery rate of fixable failures, the newly introduced regression rate, invalid tool calls, infinite loops/timeouts, the rejection rate, and the false-rejection rate. An external PRM is not a necessary condition: compilation, public tests, structured task checks, and file/resource constraints can provide verifiable signals; Gen-Zero's own small outcome/value model only supplements these signals and cannot override hard constraints.

**Start with shallow search on real feedback.** In the first phase, only do shallow trial execution and repair on a small number of distinct patch/command branches, enforcing a single-step evidence loop. Only after this shows a gain under an equal-budget control should multi-step PUCT, state merging, transposition caches, and learned rollout be added. Prioritize search at nodes with both high failure risk and high information value, avoiding repeating route/ask/imagine at every step or expanding indiscriminately. All caching and cost-reduction paths must preserve the same gate and evidence requirements.

**Train the world model on observable sub-problems.** First predict compile/test status, action failure type, resource cost, key dependency changes, and prediction uncertainty, rather than directly claiming the ability to simulate arbitrary code. Split by task/repository for isolation, and verify first-order and multi-step prediction error, calibration, out-of-distribution rejection rate, and the eventual decision gain. When OOD or uncertainty exceeds budget, stop deep search inside the model and switch to controlled observation/actual trial execution, rather than silently continuing with random weights.

**Keep geometric constraints in their proper role.** Use geometric residuals, cross-stream inconsistency, and energy change as interpretable anomaly detection or candidate-ranking aids; each must demonstrate an incremental gain relative to not having that feature. A drop in energy only means the defined energy dropped, not that the patch is correct. Do not force discrete, irreversible software operations to satisfy conservative physical dynamics: Koopman/symplectic structure should be applied to data-supported local dynamics or latent subsystems, not treated as a universal assumption about all terminal states.

**The 80-to-90 gains ledger.** Assuming the baseline live success rate has been rigorously measured at 80%, reaching 90% requires recovering half of the originally failed tasks if no regressions are introduced; the requirement is higher if there are regressions. State this as a task count: `new successes = baseline successes + fixes - regressions`, rather than arbitrarily assigning a few percentage points to each algorithm. If the fixed-candidate ceiling is below 90%, coverage must be expanded through new candidates or execution-time repair; no amount of reranking accuracy is enough on its own.

| Stage | Deliverable | Condition to enter the next stage |
|---|---|---|
| P0 trustworthiness fixes | Reject on missing input/policy; unified completion status; non-bypassable execution authorization; fixed test and weight manifest | Negative tests at all execution entry points, and rejection tests for missing evidence/timeout/illegal values, all pass; stubs are not allowed to impersonate real services |
| P1 minimal live loop | Fixed Proposer + Gen-Zero branch selection + isolated execution + external verifier | Complete evidence per action; count of real commands > 0; the task verifier actually runs; both successes and failures are retained |
| P2 equal-budget gains | Ablations of no-search / shallow search / PUCT / geometric or world-model variants | Paired comparison on a new task set after freezing the configuration; gains reported together with cost, regressions, and rejection rate |
| P3 generalization and the 90% target | Multiple repositories, different failure types, repeated seeds, and a formal benchmark | Sample size, statistical methodology, and exit conditions determined in advance; only claim the target once reached, and retain negative results if it is not |

Keep the same execution safety floor across all comparison groups; gate-related gains can be analyzed with an offline negative-action set or a hard-isolation controlled experiment, but safety measures must not be turned off in the real environment merely for the sake of an ablation. A* is suited to discretizable dependency/sub-goal graphs; CEM is suited to continuous or parameterized optimization; GFlowNet can be used to study candidate coverage; CFR only has value for problems explicitly modeled as a game; CP-SAT handles explicit discrete constraints. None of these need to be crammed into the pipeline for every coding task.

## 5. Live Wiring and “100% Ready” Verification

**Terminal-Bench: static wiring exists, but there is no evidence of a successful closed loop.** The current adapter's `route -> ask -> imagine -> environment.exec` is at `benchmarks/gen_zero_tb_adapter.py:289,339`. It refuses execution when semantic/gate/MCTS evidence is missing (`:46`), and explicitly states that the formal scope does not cover arbitrary shell commands (`:250`). These boundaries should be preserved.

The raw command for a historical Harbor run is at `benchmarks/reports/t2-harbor/command-3.txt:1`. Docker startup and isolation checks completed, but it was rejected for missing the proposer URL/model; there was no task execution and no verifier reward, and the standalone `acceptance.exit` is 1. `REPORT.md:75` explicitly records this as incomplete; a Harbor CLI exit code of 0 does not override the trial error. The current adapter SHA matches the report manifest, but the historical traceback line numbers/text differ from the current source, so this old dev trial must not be passed off as having run against the current HEAD.

**DeepSWE: a constrained repair adapter for an external generator, not the Gen-Zero planning loop.** The current `benchmarks/gen_zero_deepswe_adapter.py:290` records `code_generator=external-proposer` and `gen_zero_world_model_used=false`; its source contains no MCTS/PolicyGate/Gen-Zero decision RPC calls. It provides constrained file operations, Git patches, regression checks, and a Pier interface, and has not actually wired up a symbolic world model. The scope restrictions also include language/file type limits, a prohibition on modifying test configuration, specified regression commands, and so on (`docs/benchmarks/deepswe_adapter_runbook.md:87`); a constrained smoke test must not be treated as full-task capability.

`docs/benchmarks/evidence/t3-deepswe/attempt-02/acceptance.json:1`: accepted=false, patch_bytes=0, reward=0, F2P=0/60, P2P=964/964; both the Pier and verifier processes can exit 0. Existing tests not regressing does not mean the requirement is fulfilled, and partial=0.9414 certainly must not be written up as 94.14% task success. attempt-01 also failed. The attempt-02 log records an action rejection followed by an external read timeout; these failures must not be hidden while only reporting “Docker/Pier ready.”

A historical raw MCP probe also exposed the old dev engine hardcoding `expected_reward=1.45`/`formal_checked=true`, with evidence at `docs/benchmarks/evidence/t3-deepswe/RESULTS.md:51` and `raw/dev-zero-excerpt.txt:15`. This is a genuine historical false-capability issue that should be held accountable, but it should not be misattributed to the current HEAD: the current `crates/gen-zero-service/src/zero.rs:3146` explicitly defines the MCTS value as an action-sequence likelihood, and marks `value_is_environment_reward=false`. The deployed adapter SHAs for both DeepSWE trials also differ from the current file.

So the accurate conclusion is: “the infrastructure has been shown to run in part, but a complete Gen-Zero live closed loop and the full solve rate have not been verified.” This investigation did not connect to dev for a rerun, did not deploy a model, and did not run the full TB/DeepSWE suite. The next round of verification should first fix the versions and hashes of the adapter/Rust binary/model/dataset/container image, then scale up only after a single problem runs successfully.

## 6. Planning Engines and World Models: Same-Named Modules Must Be Considered Separately

The easiest mistake is lumping different code paths together under the single name “Gen-Zero MCTS.” At present, at least the following exist:

- **Rust service semantic PUCT**: `crates/gen-zero-service/src/imagine.rs:1,175,245,263` genuinely has a multi-level action-history tree with expansion, visit counts, and PUCT, called from `zero.rs:3082,3101`. Nodes are action histories; the bridge still receives the original state plus appended history (`imagine.rs:51`), not a post-execution file/process snapshot. The value is the geometric mean of step-wise probabilities, not an environment reward (`:8`). This is a genuine semantic sequence search, but not a multi-step rehearsal of the terminal environment.
- **Rust planner MctsEngine**: `crates/gen-zero-planner/src/engine.rs:206` explicitly implements depth-1, backing up only a single step reward at the root state and discarding successor/done. It must not share a capability description with the service implementation above.
- **Python MctsEngine**: the neural path can perform a multi-step greedy imagined rollout (`python/gen_zero/planner/engines/mcts_engine.py:372`), but successor expansion of the main tree is limited; when the model/reward path is missing, there is a random leaf value (around `:265`). There is also a `LatentMctsPlanner` and a two-step-branch unit test, but these must not be automatically treated as the default production wiring.

| Engine | Current source-code facts | How it should be used to add value |
|---|---|---|
| A* | Rust `engine.rs:338` is single-step candidate scoring/a heap; Python `astar_engine.py:204` has a genuine goal/neighbors graph search, but the client routes by state (`client.py:910`) | Construct an explicit sub-goal/dependency graph for coding tasks; demonstrate the heuristic and state semantics, rather than keeping only the name “A*” |
| CEM | Rust `engine.rs:467` has a limited multi-step rollout; the Python discrete `mpc_cem_engine.py:94` calls transition_fn; the continuous branch still has a synthetic transition (`:188`) | Optimize parameterized actions/budget; must use a real, calibratable transition and a time-position-dependent sequence distribution |
| GFlowNet | Rust `engine.rs:598` is single-step reward-distance scoring; Python `continuous_gflownet.py:218,252,362` can sample trajectories, but the plan does not run full TB training | Use as a research branch for diverse candidate coverage, and measure the oracle-coverage gain; without training evidence, do not claim the flow distribution has been learned |
| CFR | Rust `engine.rs:661` performs one pass of regret matching; Python `cfr_nash_engine.py:157` falls back to utility/heuristics when there is no game | Introduce only for tasks with explicit players, information sets, payoffs, and repeated iteration; do not call ordinary coding branch selection a Nash-equilibrium solve |
| CP-SAT | Rust `engine.rs:737` is gate filtering plus single-step max reward; Python `cpsat_formal_engine.py:23` explicitly states it uses predicates, not the OR-Tools solver | Explicitly model resources/dependencies/mutual exclusion, and save solver status and feasibility evidence; the real OR-Tools call is instead at `gate/action_constraints.py:294` |

The Rust `ProductionPipeline`'s simulate/what_if/trajectory paths genuinely do multi-step advancement (`crates/gen-zero-planner/src/pipeline.rs:313,392,645`), but a subsequent fixed/greedy rollout does not mean every engine has become a full tree search. The planning interfaces `crates/gen-zero-core/src/traits.rs:39` and `gen-zero-planner/src/engine.rs:68` still need to enrich the goal, state-dependent legal actions, budget, and termination/risk/certificate semantics before they can reliably interface with real code state.

**The world model is not a ready-made terminal simulator.** The Rust `crates/gen-zero-worldmodel/src/dynamics.rs:54` uses a fixed residual prior `next≈0.95*state+0.05*sin(action phase)`; the Symplectic path genuinely has Stormer-Verlet/Hamiltonian numerical steps, but the action centre, reward, and done are hand-crafted priors (`symplectic_dynamics.rs:1,123,176`). The service explicitly marks both as untrained/uncalibrated, and the input is only a numeric latent of a fixed dimension; there is no encoder from text/code into this latent (`crates/gen-zero-service/src/worldsim.rs:1`). So a stable integrator must not be written up as “understanding the consequences of code execution.”

Koopman has a mathematical implementation at `crates/gen-zero-worldmodel/src/koopman.rs:100` and `koopman_spectral.rs:155`; this investigation's search found no call to it from the Rust planner/service. The Python MCTS's `use_koopman_jumps` merely stores a field (`mcts_engine.py:156,170`), so it must not be claimed that jump-planning has been wired in on this basis.

The Python `NeuralDynamicsWorldModel` is a trainable residual MLP, but its archived training data comes from a deadlock-torus synthetic environment (`scripts/extract_trajectories.py:16,202`), not terminal patch trajectories. The training script itself states that a same-generator held-out set cannot prove real-task capability (`scripts/train_world_model_dynamics.py:8`). The local checkpoint's 64-D state/16-D action configuration and provenance also cannot be passed off as the current HEAD's real coding world model.

The recommendation is to treat “algorithmic correctness, data fitting, prediction calibration, planning gain, and the live closed loop” as five separate acceptance gates. Numerical invariants are suited to proving properties of a specified mathematical object; they cannot skip the three intermediate layers and directly guarantee task success.

## 7. Model Scale and Weight Attribution Inventory

| Module | Verified architecture and scale | Weight/capability boundary |
|---|---|---|
| External CLM DeepSWE head | Qwen3-8B's 4096-D embedding; `best_head.pt` cfg width=1536, depth=3, projection=512; the state/action MLPs each have 9,443,840 parameters, totaling **18,887,680** | The 8B is the frozen encoder, not the head. What was actually read is the dedicated best_head, not the general-purpose CLM checkpoint; `/tmp/clmrepro/repo/src/clm/heads.py:1,21` |
| Rust ChoiceHead | dimension/temperature configuration + online ETF geometric scoring, **no learned weight matrix** | `crates/gen-zero-model/src/choice_head.rs:10,88`; a 4096-D input does not equal 8B parameters |
| Python NanoCore ChoiceHead (reused by model/choice_head.py:20) | Default fixed-seed 128x128 projection, 16,384 values total; the Torch wrapper corresponds to a 16,384-parameter Linear layer | `python/gen_zero/nanocore/choice_head.py:500,629`; this module by itself does not prove a trained checkpoint has been loaded |
| Rust NanoCore | prototype128 + projection(out_dim x 128) + value128, i.e. `256+128*out_dim` f32 values | `crates/gen-zero-nanocore/src/core_type.rs:22,43`; default sin/cos initialization, not a large pretrained model |
| Python runtime NanoCore | Defaults: browser 368,832, vision 123,072, specialist 180,480 weight values | `runtime/nano_core_browser.py:21`, `nano_core_vision.py:21`, `specialist_nano_core.py:21` (all under `python/gen_zero/`); default seeded NumPy, with checkpoint loading optional; the default values must not be taken as a trained result |
| SemanticScorer | Local Qwen2.5-0.5B; actually reads safetensors with 290 tensors, **494,032,768 parameters**; uses tied embed_tokens as the lm_head, for candidate-continuation likelihood/PMI calibration | `python/gen_zero/service/semantic_scorer.py:133,237`; there is no separate 8B scoring head; missing local weights raises an error (`causal/zero_runtime.py:109`) |
| Rust worldmodel | Fixed residual/hand-crafted Hamiltonian prior, 1024-D latent | No 8B trained weights; the geometric numerical model and the language encoder are not the same object |
| PolicyGate | Linear constraints, risk tiering, certificate verification procedures | Not a neural network, and has no so-called 8B parameters; semantic risk probability is supplied by an upstream model |

The local Qwen configuration is hidden=896, intermediate=4864, 24 layers, 14 attention heads, 2 KV heads, vocab=151936, tie_word_embeddings=true. This external pretrained backbone should be counted as a system dependency and a real cost; one must not claim “no external PRM” while implying “no external model is used at all.” What “achievable without relying on an external PRM” means is: not making an external, dedicated reward model a necessary dependency for the decision; it is still fine to use the Proposer and encoding representations, and to train one's own small value module with real execution feedback.

The parameter figures are for the module/default configuration or the scope of the local checkpoints inspected, not a single total parameter count for the whole repository. This investigation did not exhaustively scan the weights on every remote machine, so it cannot prove that some unchecked host definitely has no 8B file; but it is sufficient to establish that the 8B/AUC figures in the original claim refer to the external CLM experiment.

## 8. Verifications Performed This Time and Shared-Working-Tree Delta Notes

The following were actually executed this time, rather than merely cited from historical reports:

| Verification | Raw result | What it proves |
|---|---|---|
| Original-baseline Rust gate | 43 passed, exit 0 | Holds within the scope of the gate/geometry unit tests |
| Original-baseline Rust planner | 69 passed, exit 0 | The algorithm/interface/fixtures work per the current tests; does not represent a live solve rate |
| service semantic MCTS | 7 passed, exit 0 | A local algorithmic property of action-sequence search; not an environment simulation |
| Python gate/evidence | 43 passed, exit 0 | Corresponding evidence-boundary tests; does not eliminate the additional counterexamples in this document |
| Python neural worldmodel targeted tests | 4 passed, 50 deselected, exit 0 | rollout/trap/what-if/audit of a small, temporary, synthetically-trained model; does not represent a code world model |
| DeepSWE adapter | 9 passed, exit 0 | Boundary of the helper and temporary repository; not passage of an external model or a live task |
| TB adapter | Two collection failures, exit 2 | First missing a search path, then after correction missing Harbor; no dependency was faked to turn the tests green |
| Enhanced PRM independent verification | exit 1 | The earlier candidate/metric assertions completed; a missing checkpoint made the full provenance verification fail |
| Python real-gate counterexamples | exit 0, printed 3 PROCEED results | Boundary defects for `{}`, whitelisted high risk, and whitelisted NaN risk |
| Weight tensor count | exit 0 | Actually reads the CLM-dedicated head and local Qwen tensor sizes; does not prove model accuracy |

The complete raw commands, logs, exit codes, and probe scripts are at `docs/benchmarks/evidence/gen-zero-audit-20260927/COMMANDS.md`. These tests are not a full-repository acceptance run; there is no successful log from a real Proposer/container task in this investigation, so live-accepted cannot be claimed.

During the investigation, other work merged the shared branch from `6dfd609` to `701d800a5d3c55db6dcdb002b7bda5608a6bba11`. This investigation did not perform those merges. Among the new changes:

- The Rust constraint sum was changed to i128, with overflow tests added.
- Rust MCTS added successor/reward numeric validation and rejection of repeated actions/zero budget; it is still depth-1.
- Rust CEM has been changed to a separate categorical distribution per time position, with elite updates over the full sequence, depth discounting, and terminal truncation. This document lists it as an improvement suggestion for the original baseline, but this part has now been implemented by parallel work and should not be re-reported as pending.
- GFlowNet/CFR/CP-SAT comments and naming now more honestly indicate their legacy limitations; this does not turn them into full flow learning, a game solver, or a CP-SAT solver.
- Some exception-swallowing paths in the Python client were changed to raise explicit errors; the file containing this document's Python PolicyGate whitelist/missing-field counterexamples is unchanged.

To avoid reusing old tests to pass off as a new acceptance run, this investigation additionally ran `cargo test -p gen-zero-gate -p gen-zero-planner --locked` on the new HEAD: gate 51 passed, planner 78 passed, 129 passed in total, exit 0. This rerun and the separate service/Python tests should not be conflated into a single, complete repository-snapshot acceptance at one point in time. The new HEAD has not added a terminal-state encoder, a real terminal-simulation reward, or a Gen-Zero call from DeepSWE; the core conclusions are unchanged.

The read-only investigation was carried out by the main thread and three Luna sidecars, each checking the model, planning, and evaluation-chain paths respectively; the main thread reviewed the key code/raw JSON and ran the tests listed above. Worker conclusions were not treated as deployment authorization, no external team messaging interface was called, and no production implementation was changed.
