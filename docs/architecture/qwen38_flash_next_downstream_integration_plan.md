# Downstream Integration, Latent-Space Alignment, and End-to-End Evaluation Tiering Plan for Qwen3.8-Flash-Next Representations

Date: 2026-09-27 · HEAD `fa6cddb` · Nature: **a design plan, not a results report**

Every "integration" and "gain" mentioned in this document is an unimplemented design. Every statement of current state is accompanied by a `path:line` reference or a command.

---

## 0. The most critical facts first (premise correction)

The task description contains three premises that **do not hold, or are unverified**, in this repository and on this machine. The plan must be built on the corrected premises.

| # | Premise in the task | On-the-ground fact | Evidence |
|---|---|---|---|
| F1 | "the extracted Qwen3.8-Flash-Next core manifold" | **No Flash-Next extracted features exist at all.** The repository contains only a feasibility report (whose conclusion is that switching is not currently recommended), a streaming prototype built on synthetic random weights, and a shard-layout analysis script that only reads safetensors headers. | `artifacts/qwen_flash_next_feasibility_report_2026-09-24.md:10`; `scripts/prototype_layer_streaming.py:20` ("we do not have Qwen3.8-Flash-Next weights"); `scripts/analyze_flash_next_shard_layout.py:4` ("No weight is downloaded"); `ls /ebs/data/extracted_features/` shows only `qwen72b llama70b gte7b_cpu manifold_alignment` |
| F2 | "the 64-D / 128-D / 256-D / 896-D core manifold" | The 64/128/256/896-D manifolds on disk come from **Qwen2.5-0.5B** (hidden=896), not Flash-Next. Flash-Next has `hidden_size=2560`, so the corresponding ladder should be 64/128/256/**2560**. | `benchmarks/results/zero_cpu_natural_multidim_eval_summary.json` → `backbone.family = zero-qwen2.5-0.5b-trunk`; `mean.shape = (896,)` in `zero_manifold_natural_gpu_896d.npz`; config.json shows an actual `hidden_size` of 2560 |
| F3 | "6B active parameters" implies lightweight | Active-parameter count only reduces compute, not memory. Full bf16 is 360 GB; a single A100-80G can only fit UD-Q2_K_XL (78.9 GB), leaving no headroom for KV cache. | `artifacts/qwen_flash_next_feasibility_report_2026-09-24.md:17,79-92` |

Premises verified as true:

- **The 262K native context length is real**: `max_position_embeddings = 262144` (fetched via `curl -sL https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`, HTTP 200, fetched 2026-09-27).
- **The GDN structure is real**: 48 layers, `[linear_attention ×3, full_attention ×1] ×12`; linear attention has 48 value heads, 16 key heads, `dk = dv = 128`; `hc_count = 4` (same config.json). The config only states `linear_attention`; the name "Gated DeltaNet" comes from the official README — see the table in feasibility report §1.2 (`artifacts/qwen_flash_next_feasibility_report_2026-09-24.md:59`).
- **Note**: the config above was fetched from the `main` branch, not a pinned revision. P0 must pin a revision and re-verify (consistent with §1 of `docs/zero/qwen38-flash-next-extraction-system-design.md`).

**Conclusion**: for every one of the four layers below, the current "input data" is zero. The first stage must produce real features before anything else (§4, stage P0); otherwise none of the downstream acceptance criteria can be evaluated at all.

---

## 1. Integration tiers

### 1.0 Overall constraints (carrying forward the existing fail-closed contract on the production trunk)

- Planner: when the world model returns `Err`, planning aborts immediately; it must never be converted into a skip, a zero reward, or a default action (`crates/gen-zero-planner/src/engine.rs:10-13`). The new scorer follows the same contract: an `Err` from the scorer aborts.
- Service: when the semantic bridge is unreachable, `_meta.engine = "local_fast_reflex_fallback"` is set and the result is explicitly marked as unscored (`crates/gen-zero-service/src/zero.rs:20-24,848`). The new encoder path must carry the same explicit marking, and must provide a `required` switch that, once enabled, rejects the request outright when unreachable (aligned with the existing `GENZERO_BRIDGE_REQUIRED`).
- Mounting: cognitive assets are sealed via immutable mount snapshots; with no asset present the result is `BackendUnavailable`, never a default model (`crates/gen-zero-service/src/cognitive.rs:8-10`). The Flash-Next projection matrices must go through the same mount-sealing process, with a summary recorded under `_meta.mount`.

### 1.1 Layer 1: the Rust ChoiceHead ETF geometric scoring tier

#### Current state (a critical problem)

`ActionETFChoiceHead::evaluate` computes as follows:

1. L2-normalizes the whole of `gather_rep[..D]` (`crates/gen-zero-model/src/choice_head.rs:72-86`);
2. Assigns simplex vertices to candidates in ascending `ActionId` order (`choice_head.rs:91-93`), where `ActionId` is the first 4 bytes of `blake3(candidate name)` (`crates/gen-zero-service/src/zero.rs:916-922`);
3. `SimplexEtfFrame::project_logits` **reads only the first K−1 coordinates** (`crates/gen-zero-core/src/etf.rs:132-143`).

This produces three consequences:

- **Candidate content never reaches this head.** Which candidate gets which vertex is decided purely by the hash of its name, with no dependence on the candidate text's representation.
- **Logit magnitude is squashed.** Both the vertex and the input are unit vectors, so `|logit_i| ≤ ‖x[..K−1]‖`. For an isotropic D-dimensional unit vector, `‖x[..K−1]‖ ≈ sqrt((K−1)/D)`. At D=2560, K=4 this is about 0.034; at T=1 the softmax is nearly uniform (max probability ratio ≤ e^{0.068} ≈ 1.07). Even with PCA coordinates (where variance concentrates in the leading dimensions), this only maps "the state's own leading K−1 principal components" onto "vertices ordered by name hash" — still semantically meaningless.
- **Entropy has a floor, and any K≥3 necessarily triggers escalation.** After normalization, `|logit| ≤ 1` regardless of how "sharp" the input is. The best case is `rep` pointing exactly at one vertex, giving logits `(1, −1/(K−1), …)`. At the fixed T=1 used at the call site (`zero.rs:2464` passes `1.0`): for K=2, p_max=0.881 with normalized entropy 0.527; for K=3, 0.691 and 0.757; for K=4, 0.558 and 0.845. This entropy feeds into `PolicyGate::evaluate` (`zero.rs:2499-2502`), whose default escalation threshold is 0.65 (`crates/gen-zero-gate/src/policy.rs:47,233`). So **whenever the candidate count is ≥ 3, the ETF head's output is always classified as Tier2Escalate**, regardless of the input.
- **There is only a single production call site**, and its input is either a caller-supplied number (`etf_rep`) or the output of the untrained nanocore (`zero.rs:2446-2469`). The nanocore's projection weights follow a fixed `sin()` pattern and are untrained (`crates/gen-zero-nanocore/src/core_type.rs:44-49`). The service head's own comment states plainly that "no text encoder feeds into the manifold" (`zero.rs:37-41`).

So "feeding the Flash-Next manifold into the ETF" **cannot** be understood as "feed the state vector into `gather_rep`." Doing that would be equivalent to rolling dice.

#### Design: ETF is a coordinate system, not a scorer

The correct approach is to score **each candidate** in the aligned manifold first, then compose the scores onto the ETF vertex basis:

```
Input: state text x, candidates c_1..c_K (text)
1. Encode   h_x  = E(x),     h_c = E(x ⊕ c)        E = last-token hidden state at Flash-Next layer ℓ*; the aggregation method for the 4-way residual branches (mean / concat / gated readout) is an explicit config item, recorded in the mount summary, with the default decided by P0 ablations
2. Reduce   z    = P (h − μ)                        P ∈ R^{d×2560}, d ∈ {64,128,256}; fit only on the training set
3. Candidate score s_c  = ⟨W z_x, z_c⟩ / τ          W is a bilinear task head, fit on the training set
4. Compose  rep  = Σ_c s_c · v_{π(c)}               v is the Helmert vertex, π is the ordering by ActionId
5. ETF      ⟨v_{π(c)}, rep⟩ = s_c − (1/(K−1))·Σ_{c'≠c} s_{c'} = (K/(K−1))·s_c − S/(K−1),  S = Σ_c s_c
```

Note on step 5: because `⟨v_i, v_j⟩ = −1/(K−1)`, the composed logit is a positive-slope affine transform of `s_c`, so **order is preserved**.

But the entropy floor discussed above means that as long as `evaluate` still performs L2 normalization with T fixed at 1, no matter how good `s_c` is, the output probability will still be flattened. **Adopt option (b):**

- **Probability** is taken directly as `softmax(s_c / τ)`, where τ is calibrated on the training set (temperature scaling, minimizing validation-set NLL) and sealed as a mount asset.
- **ETF** is responsible only for the permutation-equivariant argmax and the deterministic tie-break by `ActionId`; its probability output no longer feeds the gate's entropy computation.
- Option (a) is rejected (removing normalization and passing τ into `ActionETFChoiceHead::new`): it would change `evaluate`'s semantics for every existing caller, a much larger blast radius.

Here, the ETF only provides permutation-equivariance and compatibility with the existing `decide` protocol; all the actual discriminative power lives in step 3.

Production code changes required (design, not implemented):

| Location | Change |
|---|---|
| The `head == "etf"` branch (`:2446`) in `crates/gen-zero-service/src/zero.rs` | Add `etf_source = "encoder"`: the service calls the encoder sidecar to obtain `s_c`, and composes `rep` in Rust. The old path of "treat the raw state vector directly as rep" is changed to reject the request (returning `InvalidParams` when K>1 and no candidate scores are supplied), so hash-based decisions can no longer occur. |
| `crates/gen-zero-service/src/bridge.rs` | Add a `/v1/encode_candidates` client call, reusing the existing circuit-breaker and `required` semantics. |
| The `TopologyPreset` allowlist in `crates/gen-zero-service/src/cognitive.rs` (comment `:22-24`, currently only 64/128/256) | Do not add 2560. 2560 is only the encoder's internal width; the manifold width is still chosen from the allowlist, keeping the mount-summary sealing intact. |
| `zero.rs:2464` (`ActionETFChoiceHead::new(rep.len(), 1.0)`) and `:2471-2499` (probability composition and entropy) | When `etf_source=encoder`, the probability is computed as `softmax(s_c/τ)`, with τ read from the mount; entropy is computed from this probability before being passed to `PolicyGate::evaluate`. |
| `crates/gen-zero-service/src/server.rs:1661,1670` (MCP schema), `crates/gen-zero-cli/src/main.rs:119,554` (CLI `--etf-rep`) | The two non-test entry points for `etf_rep`. Change them to accept only "one score per candidate" (length = K), updating documentation and schema accordingly; remove the old "arbitrary-length state vector" semantics. The tests at `zero.rs:4281,4309,4337` are rewritten accordingly. |
| Mount asset | Seal the sha256 of `P, μ, W, τ, ℓ*, branch aggregation method, model revision, quantization tier`. |

Must also be deleted or rewritten: the path at `zero.rs:2447-2448` that treats `state.as_slice()` directly as `rep` (Rule 5: superseded old logic is physically removed).

#### Why not use 896-D / 2560-D directly

- 896-D belongs to Qwen2.5-0.5B, which is not in the same coordinate system as Flash-Next, so the two cannot be mixed (F2).
- Full-width 2560-D ZCA has shown no demonstrable gain in existing 0.5B experiments: the 64-D task head gets `train_accuracy 0.504`, `base_accuracy 0.189` (`dimensions.64.task_head` in `zero_cpu_natural_multidim_eval_summary.json`). Width is not the bottleneck; the discriminative signal is. Use 128-D as the default first, with 64/256 as ablations.

### 1.2 Layer 2: the symplectic world-model dynamics alignment tier

#### Current state

- `SymplecticWorldModelDynamics` is a **hand-designed prior, untrained and uncalibrated** (`crates/gen-zero-worldmodel/src/symplectic_dynamics.rs:19-21`). `H_a = ½|p|² + k/2 |q − c_a|²`, with `c_a` derived from a `sin` encoding of the action id (`:88-96`).
- An existing "10-step energy conservation" test checks `|drift|/H0 < 1e-4`, but it tests this hand-designed potential well (`symplectic_dynamics.rs:254-256`), not a fit to any data.
- The latent-space width is fixed: `FullLatent` 1024 = q 512 + p 512 (`crates/gen-zero-worldmodel/src/contact.rs:254`, `crates/gen-zero-service/src/worldsim.rs:32`).
- The dissipative dynamics in `contact.rs` (Strang splitting, phase-volume contraction) has **zero references on the production path**: `worldsim.rs:137-140` constructs only `Residual | Symplectic`; `ContactState` appears only in `contact.rs` itself and in `tests/compression.rs`. Koopman (`koopman.rs`, `koopman_spectral.rs`) has zero references in `crates/gen-zero-service/src`.

#### A mathematical contradiction: conservation and contraction cannot both be required

Störmer-Verlet is a symplectic integrator, so it preserves phase volume (`symplectic_dynamics.rs:12-13`). A volume-preserving map cannot be a contraction map. The task requires both "10-step energy conservation" **and** contractivity, and the two are mutually exclusive on the same flow.

More importantly: **the GDN state recurrence is itself dissipative.** The Gated DeltaNet update is

```
S_t = α_t · S_{t−1} · (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,   α_t ∈ (0,1), β_t ∈ (0,1), ‖k_t‖ = 1
```

The spectral norm of `(I − β k kᵀ)` is ≤ 1, and multiplying by `α_t < 1` on top of that makes the homogeneous part a strict contraction. What GDN gives us is "input-driven decaying memory," which naturally corresponds to a contact/conformal-symplectic structure, not a conservative Hamiltonian flow. Fitting it to a pure symplectic `H(q,p)` is searching for parameters under the wrong structure.

#### Design: a conformal-symplectic (= contact Hamiltonian with linear damping) decomposition

```
H_a(q, p) = ½ pᵀ M⁻¹ p + ½ (q − c_a)ᵀ K (q − c_a)
dq/dt = M⁻¹ p
dp/dt = −K (q − c_a) − γ p                 γ ≥ 0
```

- γ = 0 part: keep the existing Störmer-Verlet; conservation acceptance applies only to this part.
- γ > 0 part: the exact solution `p ← e^{−γ Δt} p` is combined with the symplectic step via Strang splitting. `H_a` is then non-increasing at every step, satisfying `H_a(t+10) ≤ H_a(t)` after 10 steps, with an explicit upper bound available. Contraction acceptance applies only to this part.
- This is exactly the Strang-splitting structure `contact.rs` already has. **Recommendation: wire `contact.rs` into `worldsim::WorldDynamics` as a third `DynamicsKind::Contact` variant**; if it is still not integrated by the end of stage P3, delete `contact.rs` per Rule 5 rather than leave it as an island.

#### Mapping Flash-Next representations to (q, p) (design choice, untested)

The GDN state `S_t` is 48 128×128 matrices per layer (786,432 numbers), not a (q, p) pair. We do not use `S_t` directly; instead we use the observable residual stream:

1. **Layer selection ℓ\***: compute layer-wise linear CKA between adjacent layers on the training set, and take the point of maximum negative curvature on the CKA curve as the "phase-transition layer." The aggregation method for the 4-way residual branches follows the same explicit config item as Layer 1, with no silent flatten/mean (consistent with the prohibitions in §1 of `docs/zero/qwen38-flash-next-extraction-system-design.md`; the mean used in `scripts/test_intermediate_layer_probe.py:100-106` is only a prototype and is self-described as unverified).
2. **q**: `q_t = P_q (h_t^{ℓ*} − μ)`, with `P_q ∈ R^{512×2560}` obtained via PCA on the training set. Here t is an **agent step** (the state after one read/edit/command), not a token step.
3. **p**: `p_t = (q_{t+1} − q_{t−1}) / (2Δt)`, central difference. This requires trajectory data; a single sample cannot yield p.
4. **Fitting**: actions are grouped by operation type (`read/search/edit/command/finalize`, matching the allowlist at `benchmarks/deepswe_genzero_gate.py:118`). `c_a` is learned per group, while a diagonal `K` and a scalar `γ` are learned globally. The symplectic-Euler residual `p_{t+1} − e^{−γΔt} p_t = −Δt K (q_t − c_a)` is linear least squares in `(K, c_a)`, with a closed-form solution.
5. **Stability**: after fitting, check per-dimension that `0 < K_ii` and `K_ii Δt² < 4`, and `γ ≥ 0`; reject loading if violated (the same condition set as `symplectic_dynamics.rs:66-78`, with no clamping). Least squares can produce negative stiffness, so this check cannot be skipped.

Production code changes required (design, not implemented):

| Location | Current state | Change |
|---|---|---|
| `crates/gen-zero-worldmodel/src/symplectic_dynamics.rs:53-57` | Only scalar `stiffness`, `action_scale` | Add `FittedPhaseParams { k_diag: [f32; 512], centres: Vec<[f32; 512]>, gamma: f32 }`, constructed via `SymplecticWorldModelDynamics::from_fitted`; validated as above. |
| `action_centre` in the same file `:88-96` | `c_a` generated by `sin(action.0)` | Look up by action category when fitted parameters are available; if the category is unknown, return `Err` rather than falling back to the `sin` encoding. |
| `ActionId → action category` | `ActionId = blake3(name)` (`zero.rs:916-922`); the category cannot be recovered from the id | Seal an explicit `{candidate-name pattern → category}` table as a mount asset; when the service constructs the `ActionId`, it also looks up the category and passes it to the world model. |
| `crates/gen-zero-service/src/worldsim.rs:137-160` | `WorldDynamics` has only `Residual`, `Symplectic`, both using `default()` | Add a `Contact` (γ>0) variant; `Symplectic`/`Contact` load from the mount when fitted parameters are present, otherwise keep the existing "untrained prior" marking. |
| Mount asset | No world-model parameters | Seal the sha256 of `K, c_a, γ, P_q, μ, Δt, category table`. |

#### Honest boundaries

- A real coding agent's trajectory is not a Hamiltonian system. The "fitted H" here is merely a structurally constrained linear model. Energy conservation is a property of the integrator, **not** proof that the model correctly represents the real world.
- The only meaningful acceptance criterion is prediction error: on a repository-isolated holdout set, the k-step prediction error must be significantly lower than the "persistence baseline" `q_{t+k} = q_t`.
- No trajectory dataset currently exists. The existing Python `NeuralDynamicsWorldModel` is trained on a synthetic torus environment (`docs/architecture/gen_zero_capability_audit_20260927.md:131`) and cannot be reused as evidence.

### 1.3 Layer 3: the System 2 planner tier (MCTS / A\* / CEM)

#### Current state

- `MctsEngine`: no prior, no value-function hook; return comes only from `world_model.step`, and leaf continuation value is 0 (`crates/gen-zero-planner/src/engine.rs:218-232,386-400`).
- `AStarEngine`: deliberately uses h = 0 (Dijkstra), because there is no proven bound between latent-space distance and reward (`engine.rs:508-513`). The edge cost is `1 + max(−r,0) + w·0.05·distance` (`engine.rs:651`).
- `MpcCemEngine`: samples action sequences and ranks them by the world model's discounted return (`engine.rs:872-911`).
- **The only existing prior hook** is at the service layer: the `PriorOracle` trait for the `imagine` verb (`crates/gen-zero-service/src/imagine.rs:39-40`), implemented as `BridgeOracle` (`imagine.rs:44-52`), which supplies a semantic prior to PUCT.

#### Design

| Engine | Use of manifold distance | Mathematical cost |
|---|---|---|
| **service `imagine` (recommended first choice)** | Add `EncoderOracle: PriorOracle`, with prior `π(c | history) = softmax(s_c)`, where `s_c` comes from step 3 of §1.1. | No need to change the planner crate's trait signature. PUCT has no admissibility requirement for priors. |
| MCTS (planner crate) | Requires adding an optional `&dyn PriorProvider` parameter to `PlanningEngine`. | Changes a public trait signature, affecting 6 engines. **Defer to P4, contingent on the imagine path showing a positive gain.** |
| A\* | Two options: (a) fold the manifold distance only into the existing `uncertainty_penalty_weight · distance` edge cost, preserving h = 0 and optimality; (b) weighted A\*, `h = λ·d_M(s, goal)`, **explicitly giving up the optimality claim**, marking `admissible: false` in the result. | Option (a) is recommended. Option (b) can only claim admissibility once `d_M ≤ true remaining cost` is proven, which is not currently proven. |
| CEM | Use `s_c` as the logits of the initial distribution, in place of uniform initialization. | Only initialization changes, not the objective function; lowest risk. |

All new scorers follow the same contract as `engine.rs:10-13`: scoring failure aborts; it must never silently fall back to a uniform prior without reporting an error. If a uniform prior is needed as an ablation control, it must be passed in as an explicit parameter and recorded in `_meta`.

### 1.4 Layer 4: the multi-model dual-stream manifold alignment tier

#### Current state: the existing dual-stream result is **zero gain**

Paired bootstrap from `benchmarks/results/spec21_manifold_pareto_ensemble_report.md` (13-task macro average, test set):

| Comparison | Δ pp | 95% CI |
|---|---:|---|
| Peak trajectory − single-model control | +0.37 | [−0.27, +1.01] |
| Peak − `qwen+bbp` | +0.53 | [−0.30, +1.35] |
| Peak − `concat+bbp` | −0.08 | [−0.85, +0.70] |
| `geo100+bbp` − `concat+bbp` | −0.29 | [−1.05, +0.48] |

Every CI crosses 0. All preset targets are False (the "Target check" section of the same file). Cross-model CKA averages 0.516 on the test set, with Procrustes residual 0.679 (the test_full mean row in `benchmarks/results/manifold_alignment_qwen72b_vs_llama70b_13tasks.md`).

Conclusion: **the "dual-stream Pareto frontier" currently has no statistically defensible gain.** A pre-registered stopping rule must be defined before adding a third Flash-Next stream.

#### Design

1. **Width alignment**: Flash-Next is 2560 vs. 8192 for 72B/70B. Orthogonal Procrustes requires equal width. Approach: each of the three parties is independently PCA-reduced to a common width d = 256 on the training set, then Procrustes is applied. The choice of d is frozen before any test label is read, and recorded in the report header.
2. **Joint alignment**: anchored on Qwen-72B, `R_F = argmin_{RᵀR=I} ‖Z_F R − Z_Q‖_F` (closed-form SVD solution). CKA is used for diagnostics only, not for selection. Report null CKA and null Procrustes (permutation controls), following the existing report format.
3. **Fusion**: follow the existing 5-fold OOF + 1-SE selection in `benchmarks/suites/evaluate_manifold_pareto_ensemble.py`, adding representations `flash`, `concat3`, `geo3`. Test labels are read only after selection is frozen (the script's `select_task` takes no test-label parameter; see the file header `:17-19`).
4. **Pre-registered stopping rule**: run a row-wise paired bootstrap on bits comparing the selected three-stream configuration against the best two-stream configuration (the script already has `paired_macro_bootstrap`, `:549`). **If the 95% CI lower bound ≤ 0, the Flash-Next branch of Layer 4 is deleted and does not ship to production.**

---

## 2. Real-machine evaluation deployment (Terminal-Bench 2.1 and DeepSWE)

### 2.1 Two dead dependencies in the current state

1. **The cognitive service that the DeepSWE gate depends on does not exist.** `DeepSWEGate` calls `operation: "assess"` (`benchmarks/deepswe_genzero_gate.py:145`) and `operation: "transition"` (`:180`). Across the whole repository, only test mocks respond to these two operations (`benchmarks/tests/test_deepswe_adapter_genzero_integration.py:36,48`). The Rust service has no `/evaluate` route (`crates/gen-zero-service/src/server.rs:876-894`). So on a real run, `gen_zero_mcts_used` can only be false, or the gate fails closed (`deepswe_genzero_gate.py:143-144` returns `COGNITIVE_SERVICE_UNAVAILABLE`). Separately, this path drives the Python `ImaginationMCTSPlanner` (`:20,202`), not the Rust `MctsEngine`.
2. **The TB adapter's context gate is 250,000 characters.** `len(encode(state)) > 250_000` triggers rejection (`benchmarks/gen_zero_tb_adapter.py:325-326` at HEAD `fa6cddb`; uncommitted changes from another session in the working tree have moved this to `:338-339`, but the threshold is unchanged). That is roughly 60-80K tokens. Until this is deliberately raised, the 262K context is of no use at all.

### 2.2 Flash-Next as Code Proposer + PolicyGate

Division of labor principle: **Flash-Next only proposes, Gen-Zero only decides.** The proposer never holds execution authority.

```
TB/DeepSWE task
  └─ Flash-Next (vLLM, Qwen4ExpForCausalLM, 262K) generates N JSON action candidates
       └─ Structural gate DecisionPolicyGate (paths, allowlist, ast.parse, protected directories)   deepswe_genzero_gate.py:101-139
            └─ Semantic gate /evaluate assess (new, Rust)                                            deepswe_genzero_gate.py:143-157
                 └─ Lookahead /evaluate transition + imagine (EncoderOracle prior)                   deepswe_genzero_gate.py:166-216
                      └─ Selected action → Docker sandbox execution → real observation fed back
```

Correct use of the 262K context:

- **Put repository retrieval results and already-executed observations into the context**, rather than stuffing the entire repository in. A long context reduces "read the wrong file" style failures; it cannot substitute for execution verification.
- When raising the TB limit, add two gates simultaneously: (a) the limit is measured in tokens (using Flash-Next's tokenizer directly), not characters; (b) exceeding the limit fails closed and is rejected, with no silent truncation. This follows the existing rejection semantics.
- Known risk: llama.cpp has an open issue, #28734, on "long-context decode slows down linearly" (feasibility report `:115`). Deployment goes through the vLLM CUDA path (report `:127-145`); the llama.cpp path is a fallback only, and must be explicitly marked as such.

### 2.3 How to overcome the single scoring head's "blind guessing" on holdout tasks

Baseline facts (not speculation):

- The external CLM (a frozen Qwen3-8B encoder plus two MLP heads of about 9.44 million parameters each) achieves a pooled AUC of **0.5196** on local holdout trajectories (`docs/architecture/gen_zero_capability_audit_20260927.md:18,139`), close to random.
- The existing enhanced PRM **dropped** BoN=4 from 31/38 to 29/38 on the 38-question set (same file, `:9`). These 38 questions have been repeatedly tuned against and can no longer serve as an independent holdout set.

Mechanism design (each item must be proven experimentally; all are currently **unverified**):

1. **Symbolic gating does "exclusion," not "ranking."** The structural gate can deterministically exclude illegal edits (protected paths, non-`.py`, syntax errors — `deepswe_genzero_gate.py:124-139`). It raises the floor, not the win rate itself. "Number of gate rejections" and "final pass rate after rejection" must be tracked separately and never merged into a single reported figure.
2. **Lookahead rehearsal uses real execution, not imagination.** In the sandbox, run each of the top-k candidates against the repository's existing tests once (not hidden tests), using the real exit code as the transition reward. The world model is responsible only for pruning and ranking; the final criterion is the execution result. This is the fundamental distinction from the single CLM head: **replace "scoring" with "observation."**
3. **The scoring head is enabled only after both the encoder and the manifold alignment pass P1/P2 acceptance**, and is run against a paired control of "no scoring head (uniform prior)."

Preconditions for any win-rate claim:

- Data: a new holdout set isolated by **repository**, excluding those 38 questions, frozen with no further tuning.
- Statistics: per-task pairing (same task, same seed, same budget), using McNemar's exact test or paired bootstrap. No improvement is claimed without per-sample paired statistics.
- Control arms: (A) Flash-Next bare run; (B) A + structural gate; (C) B + semantic gate; (D) C + lookahead rehearsal. Gain attribution can only come from paired differences between adjacent arms.

---

## 3. Integration pipeline diagram

```
                         ┌───────────────────────────── Offline (P0/P1) ─────────────────────────┐
  HF Qwen/Qwen3.8-Flash-Next (revision pinned)                                                      │
    └─ vLLM Qwen4ExpForCausalLM, FP8, multi-GPU / or CPU offload truncated to K layers            │
         └─ Per-layer hidden states (branch aggregation = explicit config) → CKA selects layer ℓ*  │
              └─ npz + SHA256SUMS + manifest (model rev, quantization tier, ℓ*, dataset hash)      │
                   ├─ PCA/ZCA P, μ (64/128/256) ── task head W ──┐                                 │
                   ├─ Procrustes R_F → Qwen72B anchor ─────────────┤ (Layer 4, kept only if it passes the stopping rule) │
                   └─ Trajectories (q,p) → fit K, c_a, γ ─────────────┤ (Layer 2)                  │
                                                               ▼                                   │
                                                 Mount asset published  POST /v1/mounts  (server.rs:894) │
└──────────────────────────────────────────────────────────────┬──────────────────────────────────┘
                                                               │ Sealed summary, immutable during requests
┌──────────────────────────── Online (Rust gen-zero-service) ──▼──────────────────────────────────┐
│ /v1/decisions  ask/decide head=etf, etf_source=encoder                                            │
│   └─ bridge.rs → Python encoder sidecar /v1/encode_candidates → s_c                               │
│        └─ rep = Σ s_c v_π(c) → ActionETFChoiceHead::evaluate (choice_head.rs)      [Layer 1]      │
│ imagine  → EncoderOracle: PriorOracle (imagine.rs:39) → PUCT                         [Layer 3]      │
│ simulate/what_if → WorldDynamics::{Symplectic, Contact (new)} (worldsim.rs:137)      [Layer 2]      │
│ /evaluate assess|transition (new) → PolicyGate + WorldDynamics                    [DeepSWE bridge] │
└──────────────────────────────────────────────────────────────┬──────────────────────────────────┘
                                                               │
┌──────────────────────────── Evaluation (Harbor / Pier) ──────▼──────────────────────────────────┐
│ gen_zero_tb_adapter.py  → /v1/decisions route+ask   (context gate switched to token counting, fail-closed) │
│ gen_zero_deepswe_adapter.py → DeepSWEGate → /evaluate  (Proposer = Flash-Next)                   │
│ Results: per-task bits + telemetry (gen_zero_gate_used / mcts_used / world_model_backend)         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage acceptance criteria

Every stage requires all three kinds of evidence: (1) a grep proving the new symbol has ≥1 non-test reference on the production path; (2) a command plus its expected exit code; (3) a named test line `... ok`. If any stage fails, work stops at that stage; no skipping ahead.

| Stage | Deliverable | Production-reference grep (expect ≥1 non-test hit) | Command and criterion | On failure |
|---|---|---|---|---|
| **P0 Features** | Real Flash-Next features: 13 tasks × per-layer hidden states, SHA256SUMS, manifest. Extraction engineering follows `docs/zero/qwen38-flash-next-extraction-system-design.md`; this plan only consumes its output | N/A (offline asset) | `sha256sum -c SHA256SUMS` exits 0; manifest contains model revision and quantization tier; compared against linear probes on the same tasks for 27B/72B, recorded in the report | No features means P1-P5 are all frozen |
| **P1 Layer 1** | Encoder sidecar + `etf_source=encoder` + removal of the "state vector directly as rep" path | `grep -n "etf_source" crates/gen-zero-service/src/zero.rs`; `grep -n "encode_candidates" crates/gen-zero-service/src/bridge.rs` | `cargo test -p gen-zero-service etf` named tests: (1) same state, swap candidate names, selection changes with content rather than name; (2) rejected when the sidecar is unreachable and required=1; (3) old-path input returns `InvalidParams`; (4) at K=4 with a sufficiently sharp `s_c`, the returned max probability is > 0.9 and entropy is below the gate threshold of 0.65 (guarding against the entropy-floor regression) | If the candidate-content-permutation test fails, "geometric scoring" cannot be claimed |
| **P2 Layer 4** | Three-stream alignment + pre-registered stopping rule | `grep -n "flash" benchmarks/suites/evaluate_manifold_pareto_ensemble.py` | Report shows "three-stream − best two-stream" paired bootstrap CI lower bound > 0 | CI lower bound ≤ 0: delete the Flash-Next fusion branch |
| **P3 Layer 2** | `DynamicsKind::Contact` wired into `worldsim.rs`; fitting script; trajectory dataset | `grep -n "Contact" crates/gen-zero-service/src/worldsim.rs` | Named tests: at γ=0, `|ΔH|/H0 < 1e-4` over 10 steps; at γ>0, `H` is non-increasing over 10 steps; an asset with `K_ii Δt² ≥ 4` is rejected on load. On the holdout set, k=1,3-step prediction error is below the persistence baseline, with paired bootstrap CI lower bound > 0 | If prediction error is not better than baseline: fitted parameters are not shipped; if still not integrated by the end of P3, delete `contact.rs` |
| **P4 Layer 3** | `EncoderOracle` wired into `imagine`; CEM initialization | `grep -n "EncoderOracle" crates/gen-zero-service/src/zero.rs` | Named tests: imagine returns an error (not a uniform prior) when the oracle returns `Err`; paired control of EncoderOracle vs. uniform prior on the same task set | No positive gain: do not change the planner crate trait |
| **P5 Real machine** | `/evaluate assess|transition` shipped in the Rust service; TB context gate switched to token counting | `grep -n "\"/evaluate\"" crates/gen-zero-service/src/server.rs` | Repository-isolated holdout set, per-task paired A/B/C/D four-arm comparison, McNemar's exact test; in telemetry, the proportion of `gen_zero_mcts_used=true` matches the actual call rate | Statistics not passed: report "no provable gain," do not report a win rate |

**Zero-reference removal list** (delete on expiry if not integrated, Rule 5):

| Symbol | Current production references | Deadline stage |
|---|---|---|
| `ContactState` and its integrator in `crates/gen-zero-worldmodel/src/contact.rs` | 0 (tests only) | P3 |
| `crates/gen-zero-worldmodel/src/koopman.rs`, `koopman_spectral.rs` | 0 at the service layer | P3 (this plan does not use it; delete if no other owner claims it) |
| `zero.rs:2447-2448` treating the state vector directly as ETF input; the "arbitrary-length vector" semantics of `etf_rep` (`server.rs:1670`, `gen-zero-cli/src/main.rs:119,554`) | Present, but semantically wrong | P1 |

---

## 5. Classified reporting

### Done (delivered this round, with evidence)

- This design document. Every statement of current state has been checked line by line against source code or a real command run; references appear as `path:line` in each section.
- Verified by direct fetch: Flash-Next's `max_position_embeddings=262144`, `hidden_size=2560`, 48 layers with a 3:1 GDN/full-attention interleave (`curl` on config.json, HTTP 200).
- Verified by computation: the ETF head's entropy floor at T=1 (K=2/3/4 → 0.527/0.757/0.845), compared against the gate's default threshold of 0.65 (`policy.rs:47`). Computed directly via `python3` from `logit = (1, −1/(K−1), …)`; see §1.1.
- Verified by direct fetch: the backbone of the 896-D manifold is Qwen2.5-0.5B (`backbone.family` in `zero_cpu_natural_multidim_eval_summary.json`).

### Unverified

- Whether the GDN state and the CKA phase-transition layer can yield a fittable (q, p) structure: pure design, no data.
- Whether `Qwen4ExpForCausalLM` can load a multimodal checkpoint directly while skipping the vision tower (already listed as unverified in the feasibility report).
- Whether averaging the 4-way residual streams is a reasonable aggregation method (`scripts/test_intermediate_layer_probe.py:103-104` self-describes this as unverified).
- Actual throughput and memory usage of the 262K context on an A100: untested.

### Not done

- All of P0-P5. Reason: the Flash-Next weights have not been downloaded, so no extracted features exist; there is no coding-agent trajectory dataset; the `/evaluate` service does not exist.
- Any gain or win-rate claim: none is made, since there are no per-sample paired statistics.

---

## Appendix: a fail-open risk found on-site (not introduced by this plan, not modified)

An uncommitted change to `benchmarks/gen_zero_deepswe_adapter.py` in the working tree (not by this document's author) contains two problems:

1. `self.key = os.environ.get(key_env) or "local-gpu-key"`: silently falls back to a hardcoded value when the credential is missing, replacing the original "error if unset" behavior. This violates Rule 2.
2. The LAN check uses substring matching, `any(h in url for h in ("100.", "192.168.", ...))`: a URL like `http://evil.example/100.x` would also pass, and plain HTTP is allowed through. This should be changed to parse the hostname and then check the IP against the subnet.

The author of that change is advised to fix it before committing. This plan has not modified that file.
