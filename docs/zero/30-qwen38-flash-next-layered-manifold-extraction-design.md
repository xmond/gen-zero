# Spec 30: Qwen3.8-Flash-Next Feature and Causal-Manifold Layered Extraction Design

- Date: 2026-09-27
- Source baseline: `fa6cddb656f49d9ae01bf417e476514962e03a34`, working directory `/ebs/pj/gen-zero`
- Target model: `Qwen/Qwen3.8-Flash-Next`, HF revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` (2026-08-27), `config.json` sha256 `889658f2...2e74b`
- Reference implementation: `transformers 5.17.0`'s `models/qwen4_exp/modeling_qwen4_exp.py` (abbreviated `M:` below, line numbers against that file under `~/.hermes-venv/lib/python3.11/site-packages/transformers/`); the GDN core kernel is in `models/qwen3_5/modeling_qwen3_5.py` (abbreviated `Q35:`)
- Evidence directory: `docs/zero/evidence/qwen38-extraction-design/` (every command, exit code, and log tail is in `commands.json`)

## 0. Conclusions up front, split into three categories

**Implemented (with evidence)**

1. Flash-Next and the local `qwen4_exp` implementation are the same architecture. The real `config.json`'s `architectures` field is `Qwen4ExpForConditionalGeneration`, `model_type=qwen4_exp_text`, 48 layers, `layer_types` of 36 `linear_attention` + 12 `full_attention` (`configuration_qwen4_exp.py`'s `__post_init__` rewrites `full_attention` to `qwen_sparse_attention`), `ple_layer_ids=[2]`, `hc_count=4`, 512 experts with top-10, indexer budget 2048 / compress 4. Evidence: `flash-next-config.json`, `flash-next-revision.json`.
2. **Every hook point** in the §8 summary table has been actually mounted on the same `qwen4_exp` code and had its tensor shape, value range, and code path verified, using a tiny randomly weighted model on CPU. Command: `/home/luy/.hermes-venv/bin/python scripts/test_qwen4exp_extraction_hooks_tiny.py`, exit code 0, log `hooks-tiny.log`. Key results: the maximum absolute error between the final state from the per-token recurrent path and the final state from the chunked kernel is 6.4e-10; the logits returned by `output_router_logits=True` are elementwise equal to the logits captured by the hook; the read-gate output width equals `hidden`, and the inter-layer residual width equals `hc_count*hidden`.
3. The existing `StreamingCovarianceAccumulator` has a catastrophic-cancellation defect (§6), reproduced and archived: `covariance-cancellation.log`, exit code 0.

**Unverified (done, but with no evidence that it is correct)**

- Every statement in §2-§5 of the form "the GDN state naturally fits the Lyapunov phase space," "routing entropy maps to cognitive uncertainty," or "the N-gram path can be aligned to the local manifold" is a **mathematical argument and design hypothesis**, with not a single real Flash-Next forward pass behind it. This machine has no GPU (`nvidia-smi` does not exist), and the 360 GB of weights are not stored locally.
- Two readouts whose specification is written but whose scripts are unverified: §2.3's `q_readout` (`core_attn_out`, 6144 dimensions before normalization) and §4.3's PLE gate value (requires hooking `norm_key`/`norm_query` and recomputing per `M:1246-1247`).

**Not done (not attempted, with the reason)**

- Extraction on the real model, any CKA numbers, any paired statistics between routing entropy and risk: no compute and no weights available.
- Wiring routing entropy into `PolicyGate`: this needs a new input channel (§3.4); this design only specifies it, and does not modify production code.
- The vision encoder and the MTP layer: explicitly excluded from the extraction scope.
- Fixing `StreamingCovarianceAccumulator`: a production code change, out of scope for this design document, listed as a **blocking precondition** (§6), pending an owner decision.

## 0.1 Related documents and division of labor (three parallel designs from the same day, same HEAD)

| Document | Problem it owns | Relationship to this document |
|---|---|---|
| This document, `docs/zero/30-...` | Architectural mechanism, mathematical definitions, hook points, data format (questions 1-3) | Defines "what to take, where to take it, how to compute it" |
| `docs/zero/qwen38-flash-next-extraction-system-design.md` | Loader, VRAM/memory budget, choice of Transformers vs. vLLM/SGLang engine, offload, remote closed loop | Defines "on which machine, with which engine, to run this document's hooks"; its rule against silently flattening/averaging/selecting a branch is consistent with this document's §5.2 primary convention |
| `docs/architecture/qwen38_flash_next_downstream_integration_plan.md` | Downstream consumption of extraction outputs, latent-space alignment, Terminal-Bench/DeepSWE evaluation tiers | Consumes the product format from this document's §7; its F1 finding that "no Flash-Next extraction feature currently exists" is consistent with this document's §0 |
| `artifacts/qwen_flash_next_feasibility_report_2026-09-24.md` | Earlier feasibility conclusion (full bf16 is about 360 GB, a single A100-80G has no headroom for KV cache) | This document's §9 compute boundary follows it |

None of the three documents modify production code. This document and the downstream integration plan are complementary in one respect: that document's `config.json` was fetched from `main`, whereas this document pins revision `de4b8e4d...` and archives its sha256.

## 1. Essential differences in representation extraction compared with a dense Transformer (Qwen2.5-72B)

The implicit assumptions of existing extractors all come from dense models: a single residual stream, one `hidden_states` per layer, width always equal to `hidden_size`, and stateless full-attention retrieval. `run_universal_extraction_a100.py:115-120` directly takes the middle and last layers' last token from `output_hidden_states`; `gpu_extract_qwen72b_13tasks.py` takes the 8192-dimensional pooled vector from llama-server. Flash-Next breaks all four assumptions:

| Dimension | Qwen2.5-72B (dense) | Qwen3.8-Flash-Next | Consequence for extraction |
|---|---|---|---|
| Residual stream | 1 stream, width 8192 | 4 parallel streams, inter-layer width `4x2560=10240` (`M:1480` `repeat(1,1,hc_count)`; `M:1303,1309` write back `hyper_input + injection`) | The layer output is not "the representation" but 4 streams; the 2560-dimensional `mixed_input` read by each sub-block is determined by a data-dependent read gate (`M:1023-1025`) |
| Sequence mixing | Full softmax attention, stateless | 36 GDN layers: a 128x128 matrix state per head, linear time-varying recurrence; 12 QSA layers: the indexer selects 2048 tokens via top-512 over 4-token micro-blocks before attention (`M:755-757`) | GDN layers carry a genuine **continuous dynamical state** that can be extracted; QSA layers additionally expose a discrete selection signal (the set of selected blocks) |
| Feed-forward | Dense MLP | 512 experts with top-10 plus 1 shared expert, routing softmax over all 512 (`M:971-975`) | Every layer, every token, yields an extra 512-dimensional routing distribution: an explicit uncertainty signal that dense models simply do not have |
| Lexical input | Token embedding only | Layer 2 (0-indexed layer 1) additionally injects 2-gram/3-gram hash embeddings, 51B parameters, written into the 4 streams via key/value projection and gating (`M:1246-1248`) | There exists a **discrete, local feature** path that is context-independent and determined solely by the last 3 token ids |
| Cross-layer comparability | Homogeneous across layers | Period 4: `(GDN->MoE)x3 -> (QSA->MoE)` | Adjacent-layer CKA is modulated by the architectural period, so a naive `argmin` will mislocate the transition (§5) |
| Final normalization | final RMSNorm | `hyper_connection_mixer` (`M:1493`) mixes the 4 streams down to 2560 | "The last layer's hidden state" must specify whether it means the pre-mix 10240 dimensions or the post-mix 2560 dimensions |

Conclusion: for a dense model, "take `hidden_states[l][:, -1]`" is the extraction; for Flash-Next, extraction must first answer four questions: **which stream to take (or the post-read-gate mix), which layer type to take (GDN/QSA), whether to take the routing distribution, and whether to take the N-gram path**. The layering below assigns each of these four questions to its own layer.

## 2. Layer 1: GDN recurrent state vs. QSA attention hidden state

### 2.1 The real mathematical form of the GDN state (from the code, not an assumption)

The per-token update in `Q35:478-490`, for each value head $h$ (48 heads, state $S_t^{(h)} \in \mathbb{R}^{128\times128}$):

$$
S_t = e^{g_t}\,S_{t-1} + \beta_t\, k_t\,\bigl(v_t - e^{g_t} S_{t-1}^{\top}k_t\bigr)^{\top}
    = e^{g_t}\bigl(I - \beta_t k_t k_t^{\top}\bigr) S_{t-1} + \beta_t k_t v_t^{\top}
$$

(decay is applied before computing `kv_mem`, `Q35:483-488`.) Where (`M:579,581,596`): $\beta_t=\sigma(b_t)\in(0,1)$; $g_t=-e^{A_{\log}}\cdot\mathrm{softplus}(a_t+\mathrm{dt\_bias})\le 0$, so $e^{g_t}\in(0,1]$; $q,k$ are L2-normalized inside the kernel (`use_qk_l2norm_in_kernel=True`), so $\|k_t\|=1$, and $(I-\beta_t k_t k_t^\top)$ is a contraction projection with eigenvalues $\{1-\beta_t, 1,\dots\}$.

Hence the GDN state is a **linear, time-varying, step-wise non-expansive** system: $S_t = A_t S_{t-1} + B_t u_t$, $\|A_t\|_2 \le e^{g_t} \le 1$.

### 2.2 Does it "naturally fit" $\dot z = Az + Bu$?

**Partially fits, with clear gaps; a direct claim of fit cannot be made.**

- Where it fits: linear state recurrence, linear input injection, contraction by construction. This aligns with what `LyapunovPhaseSpaceReconstructor` requires, $\rho(A)<1$ (`universal_manifold_extractor.py:349-361`).
- Gap 1: Gen-Zero fits a **constant-coefficient** $A$; GDN's $A_t=e^{g_t}(I-\beta_t k_tk_t^\top)$ is **input-dependent**. A constant-coefficient fit is an approximation of $\mathbb{E}_t[A_t]$, and the residual must become an acceptance metric (§2.4).
- B11 fix: the continuous generator is checked against the Hurwitz criterion `max(Re(lambda)) < -epsilon`, and the discrete transition matrix against the Schur criterion; an unstable fit now raises an error directly instead of being rescaled into a so-called stability certificate. The A100 entry point must be given an explicitly ordered trajectory and dt; independent benchmark sample rows must not be passed off as a trajectory.
- Gap 3 (the most critical, from the code): **the HF forward pass does not expose the per-token state.** `torch_chunk_gated_delta_rule` returns the **final** state only when `output_final_state=cache_params is not None` (`M:595,608`); with `use_cache=False` it returns nothing at all. To obtain the trajectory $z_1,\dots,z_T$, there are only two routes: (a) per-token recurrent decoding (the `seq_len==1` path), reading `cache.layers[i].recurrent_states[0]` at each step; (b) hooking the kernel internals directly. This design adopts (a), and has already verified on a tiny model that (a) agrees with the chunked-kernel result (error 6.4e-10).
- One existing practice is explicitly rejected: `run_universal_extraction_a100.py:190` concatenates the last-layer vectors of 64 **mutually unrelated prompts** in sequence into `Z_manifold` and fits $A$ to that as a "trajectory." That is not a dynamical trajectory, it is a sample sequence. Flash-Next has a genuine state trajectory along the time axis; Layer 1 must use it, and must not carry over the old practice.

### 2.3 State-extraction specification

| Item | Specification |
|---|---|
| Hook location | Read `past_key_values.layers[i].recurrent_states[0]` after each per-token forward pass, `i in GDN layers` (36 layers); shape `[B, 48, 128, 128]` (`Q35:323`), cast to float32 after the bf16 forward pass |
| Observation vector $z_t$ | The default $z_t^{(i)} = \mathrm{vec}(S_t^{(i)}) \in \mathbb{R}^{786432}$ is too large. Two levels of dimensionality reduction: (1) for each head, take the top $r$ singular values of $S^{(h)}\in\mathbb{R}^{128\times128}$ plus the left-singular-vector projection coefficients; (2) or use that layer's own query readout $y_t = S_t^\top q_t$ (i.e. `core_attn_out`, `Q35:490`), dimension `48x128=6144`, which is the state the model actually "reads." **Both are stored; `state_readout` is tagged as `svd_r` or `q_readout`** |
| Time axis | token position $t$; `dt=1`; when continuized, denote $A_{\text{cont}}=(A_{\text{disc}}-I)$ |
| Padding | left padding (right padding causes the GDN state to keep decaying, see `scripts/test_intermediate_layer_probe.py:11`); only positions with `attention_mask==1` are recorded |
| Control input $u_t$ | that GDN layer's input `mixed_input` (the read-gate output, 2560 dimensions, `M:1025`), used as the $u$ in the $Bu$ term |
| Layer coverage | not every layer stores the full trajectory. By default, 3 GDN layers store it: a fixed within-period position (e.g. the first of layers 0/1/2), the mid layer as determined by Layer 4, and the last GDN layer (layer 46). All other layers store only the final state $S_T$ |

### 2.4 QSA layer hidden state

QSA layers have no recurrent state; three things can be taken:

1. the `self_attn` output (2560 dimensions, just before `M:1302`) and that layer's `mixed_input`;
2. the indexer's token-selection mask, shape `[B, 1, T, kv_len]` bool (`M:773`, `unsqueeze(1)` on the head dimension; verified), with density $\rho_{\text{sel}}=\#\text{selected}/kv\_len$;
3. if eager attention is used, `attn_weights` can be obtained (the return value of `Qwen4ExpTextAttention` at `M:`); under sdpa it is `None` and is not treated as a required item.

The selection mask is itself a discrete signal and does not enter the continuous phase space; it is retained as a second "cognitive signal" alongside Layer 2 (§3.5), used to explain jumps in QSA-layer CKA.

**Acceptance criterion (Layer 1)**: for every layer whose full trajectory is stored, after `LyapunovPhaseSpaceReconstructor.fit` produces $A$, the median one-step prediction relative residual $\|z_{t+1}-(I+A)z_t-Bu_t\|/\|z_{t+1}\|$ must be reported; without this number, "fits" must not be written.

## 3. Layer 2: MoE router features

### 3.1 Extractable quantities (all hook paths verified)

Per layer (all 48 layers have MoE), per token:

- routing logits $\ell \in \mathbb{R}^{512}$: hooked from `layers[i].mlp.gate`'s output item 0, or from `out.router_logits[i]` after `output_router_logits=True`, shape `[B*T, 512]` (`M:971`; `M:1381` OutputRecorder). Both routes have been verified to be elementwise equal.
- full distribution $p=\mathrm{softmax}(\ell)$ (`M:972`).
- top-10 normalized weights $\tilde p$ and expert indices (`M:975`, `norm_topk_prob=True`).
- shared-expert gate $s=\sigma(w_s^\top x)\in(0,1)$ (`M:996`), shape `[B*T,1]`.
- router input $x$ = `mlp_hyper_connection`'s `mixed_input` (`M:1306`, feeding into `self.mlp` afterward).

### 3.2 Mathematical definitions

- full routing entropy: $H_{512}(p) = -\sum_{e=1}^{512} p_e\ln p_e \,/\, \ln 512 \in[0,1]$
- active-set entropy: $H_{10}(\tilde p) = -\sum_{e\in\text{top10}} \tilde p_e\ln \tilde p_e \,/\, \ln 10 \in[0,1]$
- routing sparsity (mass concentration): $m_{10} = \sum_{e\in\text{top10}} p_e \in(0,1]$, i.e. how much of the full distribution's mass the top-10 captures. $1-m_{10}$ is the "discarded routing mass," a direct quantification of the information truncation introduced by the 6B/125B active-parameter split.
- routing margin: $\Delta_{10} = p_{(10)} - p_{(11)}$, the gap between the 10th- and 11th-ranked experts; near 0 indicates the expert choice sits on a decision boundary.
- layer aggregation: all three are stored, with no presumption of which is useful: the mean $\bar H$, means grouped by depth segment (front/middle/back 16 layers), and the per-layer vector at the last token $(H^{(0)},\dots,H^{(47)})\in\mathbb{R}^{48}$.
- sequence aggregation: the last-token value, the mean within the mask, and the maximum within the mask.

Normalization is consistent with `NormalizedEntropy::from_probabilities` (`crates/gen-zero-core/src/types.rs:194`: $H/\ln\max(K,2)$), which guarantees the values can be loaded directly into `NormalizedEntropy(f32)`.

### 3.3 Mapping to PolicyGate: written as a hypothesis, not as a capability

Current state (verified with `path:line`):

- The only continuous input to Rust's `PolicyGate::evaluate` is `NormalizedEntropy`; production code computes this entropy from **candidate-action probabilities** at `crates/gen-zero-service/src/zero.rs:2053`, then feeds it into the gate at `:2072`; the default threshold is 0.65 (`crates/gen-zero-gate/src/policy.rs:47`), above which it escalates to Tier2. `PolicyGate::default()` has no constraints, no confirm action, and no heat requirement (`policy.rs:43-50`; already pointed out in `docs/architecture/gen_zero_capability_audit_20260927.md` §3).
- Python's `DecisionPolicyGate` uses three `confidence` tiers (0.30/0.50) and a `risk_prob` threshold of 0.20 (`python/gen_zero/gate/policy_gate.py:44-46`).

**The semantic mismatch must be stated explicitly**: `zero.rs`'s entropy measures "how much hesitation exists over K action candidates," whereas routing entropy measures "how spread out the choice is over 512 experts." These are different random variables, and $H_{512}$ must not be stuffed directly into `evaluate(action, entropy)` as a stand-in for action entropy. The correct approach is to **add a new input channel**:

```text
GateVerdict evaluate_with_context(action, active_context, certificate,
                                  entropy: NormalizedEntropy,          // existing: action entropy
                                  routing: Option<RoutingSignal>,      // new: {h512, h10, m10, margin, layer_band}
                                  fact_provider, agent_id)
```

Mapping hypotheses (pending paired statistical testing, not conclusions):

| Routing signal | Hypothesized cognitive meaning | Hypothesized gating action |
|---|---|---|
| $\bar H_{512}$ high and $m_{10}$ low | Input falls outside the boundary of expert specialization (out-of-distribution) | Escalate to Tier2Escalate, or push `confidence` down on the Python side |
| $\Delta_{10}\approx 0$ persistently in deep layers | Expert selection is unstable, equivalent to "multiple hypotheses coexisting" | Tier1Confirm-class actions require a second confirmation |
| $\bar H_{512}$ low and $s$ (shared gate) high | Routing has collapsed onto a small number of experts, with the shared expert carrying the load | Does not change the tier; recorded as an "overconfidence" monitoring metric |

**Acceptance precondition**: for any row of the table above to become a "capability," routing signals and ground-truth outcomes (correct/incorrect, risk label) must be obtained on the same batch of samples simultaneously, with per-sample paired statistics (e.g. AUC and its bootstrap interval), and the candidate-action entropy must be compared side by side as a control baseline. `docs/zero/29` has already established that this repository has never produced a valid per-sample paired interval; this design does not repeat that mistake.

### 3.4 Hook specification (Layer 2)

| Item | Specification |
|---|---|
| Location | `model.model.language_model.layers[i].mlp.gate` (forward hook, three-tuple output); `...mlp.shared_expert_gate` (output `[B*T,1]`); `...mlp_hyper_connection` (item 0 of the three-tuple output is the router input) |
| Precision | logits are stored as float32; `softmax` is computed in float32 (consistent with `M:972`) |
| Persistence | storing the full 512-dimensional logits for every layer and every token is costly (48x512xT). Default persistence: full 512-dimensional logits for the last token only; top-10 indices (int16) and weights (float16), $H_{512}$, $H_{10}$, $m_{10}$, $\Delta_{10}$, $s$ (float32) for all tokens |

## 4. Layer 3: the 51B-parameter N-gram lookup table and its alignment to the continuous manifold

### 4.1 What it actually is (code facts)

- Only layer 1 (index 2 using 1-based counting) has a PLE (real config `ple_layer_ids=[2]`).
- The input is not a hidden state but token ids: the last 3 ids are each multiplied by a per-layer random odd multiplier and XOR-mixed, then reduced modulo a prime vocabulary size for each head (`M:1165-1171`); 8 heads for 2-gram and 8 heads for 3-gram, 16 heads total, each looking up `2560/16=160` dimensions, concatenated to 2560 dimensions. **This is a deterministic hash function of the last 3 token ids**, independent of context and independent of the representations from earlier layers.
- The alignment is performed by the model itself (`M:1246-1248`): `key_proj` projects the 2560 dimensions into each of the 4 streams' 2560-dimensional key space, taking a dot product with that stream's post-RMSNorm state to produce one scalar gate $g_c$ per stream, which is then passed through a signed-square-root compression and sigmoid, multiplied by the 2560-dimensional value from `value_proj`, and finally a dilated depthwise-convolution local-context term is added before writing back into the 10240-dimensional residual (`M:1284`).
- The table is about 90 GiB in bf16 and is touched only at one layer; truncated loading can skip it entirely (`scripts/analyze_flash_next_shard_layout.py` already classifies bytes by `ngram_table`).

### 4.2 Local manifold representation and alignment method

The span of the N-gram embeddings is a **discrete point set**: $E_{ng}=\{\phi(w_{t-2},w_{t-1},w_t)\}\subset\mathbb{R}^{2560}$, and hash collisions make it non-injective. It is not a continuous manifold, and no dynamics should be fitted to it. Three things can and should be done:

1. **Write-in strength**: record each stream's gate value $\sigma(g_c)\in(0,1)$, $c=1..4$ (the `gate` just before `M:1248`). This is a direct readout of "how much the model relies on the lexical prior right now"; like Layer 2's $H_{512}$, it is a gateable signal.
2. **Projection onto the primary manifold**: once Layer 4/Pillar 4 produces $U_k$ (the primary basis in the 2560-dimensional read-gate space), project the PLE's value output $v=\mathrm{value\_proj}(\phi)$ as $U_k^\top(v-\mu)$, giving the coordinates of the N-gram prior on the continuous causal manifold. Define the "lexical-prior offset" $\delta_{ng} = \|U_k^\top v\|/\|v\|$: near 1 means the N-gram write falls within the primary manifold, near 0 means it writes into a direction outside the primary manifold.
3. **Counterfactual use**: for Pillar 2's $(x, do(x'))$, if only one token is changed, the change in the N-gram path is exactly computable (the hash is deterministic); $\Delta v$ can be treated as a known perturbation source and its corresponding linear image subtracted from the final-layer displacement $\Delta z$, yielding a "lexically deconfounded" causal displacement. This is an analysis that **can be done but has not been done**.

### 4.3 Hook specification (Layer 3)

| Item | Specification |
|---|---|
| Location | `layers[1].ple.ple_embedding` (output `[B,T,2560]`, the raw concatenated embedding); `layers[1].ple` (output `[B,T,10240]`, the value written in after gating and convolution); the gate value is computed inside `Qwen4ExpTextPLELayer.forward` and is not accessible via a hook, so it is recomputed via `register_forward_hook` on `ple.norm_key`/`ple.norm_query` to obtain key/query respectively, following `M:1246-1247` |
| Persistence | the 2560-dimensional $\phi$ and 4 gate values for the last token; only gate values for all tokens |
| Note | both widths have been verified (tiny model `ngram_embed_width=64`, `ple_output_width=256`, corresponding to real values 2560 / 10240) |

## 5. Layer 4: cross-layer CKA phase-transition detection in the hybrid architecture

### 5.1 Where existing tools go wrong here

- `PhaseTransitionLayerExtractor.detect_phase_transitions` (`universal_manifold_extractor.py:173-182`) takes the bare `argmin` of adjacent-layer CKA. Flash-Next has one QSA layer every 4 layers, and QSA-layer output statistics are systematically different from GDN-layer statistics, causing adjacent CKA to periodically dip at every QSA layer. A bare `argmin` will pick out some QSA layer, which is an **architectural period**, not a conceptual phase transition.
- `extract_concept_and_causal_manifolds` uses a fixed `mid_fraction=0.5` (`:184-207`), independent of the data; for 48 layers that is layer 24 (which happens to be the layer right after a QSA layer). This is not detection, it is a constant.

### 5.2 Localization rule

The space in which CKA is computed is fixed up front: **by default, the 2560-dimensional space after read-gate mixing** (each layer has two read gates; take `mlp_hyper_connection`'s `mixed_input`, which is what MoE sees as input, `M:1306`), with the raw 10240-dimensional 4-stream version stored separately as a secondary product. `collapse_streams`'s four-stream average (`scripts/test_intermediate_layer_probe.py:99-105`) is itself annotated as "unverified," and this design does not use it as the primary convention.

Localization proceeds in three steps:

1. **Like-for-like comparison**: compute adjacent CKA separately within the GDN subsequence (36 layers) and within the QSA subsequence (12 layers); also compute cross-period CKA, i.e. layer $l$ against $l+4$ (same phase). The period-4 phase confound disappears within a same-phase sequence.
2. **Periodicity control**: subtract, from the 48-layer adjacent-CKA sequence, the mean grouped by phase ($l \bmod 4$), yielding a detrended residual; search for a minimum only in the residual. If the depth of the residual's minimum is no more extreme than the minima obtained after a phase-permutation test, report "no phase transition detected" and do not force-select a layer.
3. **Definition of the two layer positions**:
   - Concept-transition layer $l_c$: the deepest significant minimum of the detrended residual within the GDN subsequence, additionally requiring that the same-phase CKA from $l_c$ through $l_c+8$ stays stably above the level before $l_c$ (the representation "settles" after this point).
   - Causal-collapse layer $l_k$: scanning backward from the final layer, the first position where same-phase CKA (against the post-mix representation of the final layer) drops below a threshold (< 0.9, the threshold recorded in the manifest); the collapsed manifold is taken from the `hyper_connection_mixer` output (`M:1493`), not from any single layer.
4. **Additional QSA-layer readout**: each QSA layer's selection-mask density $\rho_{\text{sel}}$ is recorded alongside CKA jumps, used to distinguish "retrieval-mode switching" from "representation reorganization."

### 5.3 Relationship to Pillar 4 (streaming covariance)

The primary basis $U_k$ is accumulated in the 2560-dimensional read-gate space. Accumulating one basis per layer costs 48 float64 matrices of 2560^2 each (about 2.5 GB), which is acceptable; but 48 copies of the raw 10240-dimensional stream would cost about 40 GB and is not done -- accumulation is performed only at $l_c$, $l_k$, and the final post-mix output.

## 6. Blocking pre-existing defect: catastrophic cancellation in covariance accumulation

`StreamingCovarianceAccumulator.covariance` uses $C/N - \mu\mu^\top$ (`universal_manifold_extractor.py:97-102`). When the feature mean is much larger than the standard deviation, this is textbook cancellation. Reproduced (`covariance-cancellation.log`, exit code 0, synthetic Gaussian data, not Qwen inference):

| Mean shift | Direct-method max absolute error | Subtract first-batch mean before accumulating |
|---|---|---|
| 0 | 6.7e-16 | 1.1e-15 |
| 1e3 | 1.3e-9 | 6.7e-16 |
| 1e5 | 1.4e-5 | 6.7e-16 |
| 1e8 | **25.0** (true value is order 1) | 2.8e-13 |

Why this is a blocker for Flash-Next: middle layers of the Qwen family have a small number of "massive activation" channels whose magnitude far exceeds the other dimensions (already recorded in `scripts/test_intermediate_layer_probe.py:13`, which is why a per-layer z-score is applied there). The GDN state and the raw 10240-dimensional stream are unnormalized, and feeding them directly into the accumulator concentrates the error in these channels, contaminating the leading principal directions produced by `compute_principal_basis`.

Requirements (for the owner to decide; this design does not modify production code):

1. Add a `shift` parameter to the accumulator (the first-batch mean, or an externally supplied value); accumulate on `x - shift` internally, or switch to a Welford-style batched merge.
2. Add a self-check to `covariance()`: if $\max_i |\mu_i|^2 / \mathrm{Var}_i > 10^{6}$, **raise an error** instead of returning a result (fail-closed).
3. Unit-test coverage of the shift = 1e8 case, with acceptance criterion `max_abs_err < 1e-9`.

Flash-Next's Pillar 4 accumulation must not be started until this fix lands.

## 7. Data-persistence format specification

This reuses the sha256-verified npz convention of `export_codebook`/`load_codebook` (`universal_manifold_extractor.py:469-500`); no separate binary format is invented. It is not `GZCBK001` (the header layout of `knowledge_compiler.py`); that format's A/B and basis vectors are generated by feeding this document's output into the compiler.

**File**: `<out>/flash_next_manifold_v1.npz`, with the `metadata` key holding a JSON string. `load` verifies the sha256, the shape of every array, and `np.isfinite`; any mismatch **raises an error**, with no degraded fallback path.

**Manifest (`metadata`) required fields**

```json
{
 "format": "flash-next-layered-manifold", "version": 1,
 "model": {"repo": "Qwen/Qwen3.8-Flash-Next", "revision": "de4b8e4d...", "config_sha256": "889658f2...",
           "transformers": "5.17.0", "torch": "...", "dtype": "bfloat16", "attn_implementation": "sdpa"},
 "hooks": {"read_gate": "layers[i].mlp_hyper_connection[0]", "residual": "layers[i]",
           "router": "layers[i].mlp.gate", "gdn_state": "cache.layers[i].recurrent_states[0]",
           "qsa_mask": "layers[i].self_attn.indexer", "ple": "layers[1].ple", "ngram": "layers[1].ple.ple_embedding"},
 "tokenization": {"padding_side": "left", "truncation_side": "left", "max_tok": 0, "head_tok": 0},
 "token_selection": "last_real_token | masked_mean",
 "layers": {"layer_types": ["linear_attention", "..."], "gdn": [0,1,2,4,...], "qsa": [3,7,...,47], "ple": [1]},
 "trajectory_layers": [0, "<l_c>", 46], "state_readout": "svd_r=8 | q_readout",
 "phase_transition": {"space": "read_gate_2560", "l_c": null, "l_k": null, "detrended": true, "permutation_p": null},
 "covariance": {"shift": "first_batch_mean", "dim": 2560, "k": 64, "n_samples": 0},
 "sample_ids": ["..."], "sha256": "..."
}
```

**Array naming** (`L{layer number}` two decimal digits, `S` is the stream index 0-3)

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `rg_last/L{ll}` | `(N, 2560)` | f32 | post-read-gate-mix representation, last real token |
| `res_last/L{ll}/S{s}` | `(N, 2560)` | f16 | one of the raw 4 streams (only at $l_c$, $l_k$, 47) |
| `mixer_last` | `(N, 2560)` | f32 | `hyper_connection_mixer` output |
| `gdn_traj/L{ll}` | `(N, T_max, d_z)` + `gdn_traj_len/L{ll}` `(N,)` | f32 / i32 | Layer 1 trajectory and effective length; $d_z$ determined by `state_readout` |
| `gdn_final/L{ll}` | `(N, 48, r, r)` or `(N, 6144)` | f16 | final state of the remaining GDN layers (SVD-truncated or q-readout) |
| `router_logits_last/L{ll}` | `(N, 512)` | f32 | full logits for the last token |
| `router_topk_idx/L{ll}`, `router_topk_w/L{ll}` | `(N, T_max, 10)` | i16 / f16 | top-10 for all tokens |
| `router_stats/L{ll}` | `(N, T_max, 5)` | f32 | $[H_{512}, H_{10}, m_{10}, \Delta_{10}, s]$ |
| `qsa_density/L{ll}` | `(N, T_max)` | f32 | selection-mask density |
| `ngram_last` | `(N, 2560)` | f32 | raw concatenated N-gram embedding |
| `ple_gate` | `(N, T_max, 4)` | f32 | per-stream write-in gate $\sigma(g_c)$ |
| `cka/consecutive`, `cka/same_phase`, `cka/to_mixer` | `(47,)`, `(44,)`, `(48,)` | f64 | Layer 4 curves |
| `U_k`, `mean`, `eigenvalues` | `(2560,64)`, `(2560,)`, `(64,)` | f32 | Pillar 4 (computed on `rg_last/L{l_k}`) |
| `A`, `B`, `rho` | `(d_z,d_z)`, `(d_z,2560)`, scalar | f32 | Pillar 3, one set per `trajectory_layers` entry, key suffixed `/L{ll}` |
| `fit_residual/L{ll}` | `(N,)` | f32 | §2.4 one-step prediction relative residual; a file without this is invalid |

**Row alignment**: `sample_ids` determines the order of all `N`-sized dimensions; cross-model alignment follows the "align priors first, then compute geometry" rule from `benchmarks/suites/cross_model_manifold_alignment.py`.

## 8. Full hook-point table (each item verified on the `qwen4_exp` tiny model)

| Layer | Module path (prefix `model.model.language_model.` omitted) | Output shape (real model) | Verification evidence |
|---|---|---|---|
| L4 | `layers[i]` (DecoderLayer) | `[B,T,10240]` | `hooks-tiny.log` `decoder_layer_width` |
| L4 | `layers[i].attn_hyper_connection` / `mlp_hyper_connection` -> `(mixed, streams, inject)` | `[B,T,2560]`, `[B,T,10240]`, `[B,T,4]`, `inject in (0,2)` | same as above, `read_gate_mixed_width` |
| L4 | `hyper_connection_mixer` | `[B,T,2560]` | same as above |
| L2 | `layers[i].mlp.gate` -> `(logits, topk_w, topk_idx)` | `[B*T,512]`, `[B*T,10]`, `[B*T,10]` | elementwise equal to `out.router_logits[i]` |
| L2 | `layers[i].mlp.shared_expert_gate` | `[B*T,1]` | same as above |
| L1 | `cache.layers[i].recurrent_states[0]` (GDN layer, `use_cache=True`) | `[B,48,128,128]` | `gdn_state_shape`; per-token vs. chunked agreement 6.4e-10 |
| L1 | `layers[i].linear_attn` output | `[B,T,2560]` | same as above |
| L1 | `layers[i].self_attn.indexer` (QSA layer) | `[B,1,T,kv_len]` bool | `qsa_mask_dtype`, `qsa_mask_density` |
| L3 | `layers[1].ple.ple_embedding` / `layers[1].ple` | `[B,T,2560]` / `[B,T,10240]` | `ngram_embed_width`, `ple_output_width` |

Excluded: `model.visual`, `mtp.*` (already ignored by `Qwen4ExpForCausalLM._keys_to_ignore_on_load_unexpected`).

## 9. Compute and execution boundaries

- A real forward pass requires about 250 GB of bf16 backbone plus 90 GB of N-gram table; truncated loading of the first K layers (`scripts/test_intermediate_layer_probe.py:365 load_truncated`) has already been proven on a tiny model to require only the byte-level read-only shards. Layer 4 needs all 48 layers; Layer 1/2/3 can each be done separately at K=2 (for L3) or any K.
- The cost of the Layer 1 trajectory must be stated plainly: path (a) performs $T$ single-token full-model forward passes per sample (the QSA indexer still loops per query in Python, `Qwen4ExpTextQSAIndexer.forward` at `M:`), not a single prefill pass. In this mode, the state of all 36 GDN layers is in the cache at every step, so §2.3's "store only 3 layers" is a storage choice, not a compute-driven choice.
- Neither this machine nor dev/stg/ai-wsl has any verifiable GPU information; this design makes no commitment about which machine can run this.

## 10. Evidence manifest

| File (`docs/zero/evidence/qwen38-extraction-design/`) | Content |
|---|---|
| `identity.log` | HEAD `fa6cddb6...` |
| `flash-next-config.json`, `flash-next-revision.json` | real config and HF revision, HTTP 200 |
| `hooks-tiny.log`, `hooks-tiny.exit` | full JSON report of hook verification, `EXIT=0` |
| `covariance-cancellation.log`, `.exit` | §6 reproduction, `EXIT=0` |
| `numerics.log`, `callers.log` | accumulator diagnostics and caller list left from an earlier session |
| `commands.json` | every command, exit code, log tail, and source-file sha256 |
| `scripts/test_qwen4exp_extraction_hooks_tiny.py` | re-runnable verification script; any assertion failure produces a non-zero exit |
