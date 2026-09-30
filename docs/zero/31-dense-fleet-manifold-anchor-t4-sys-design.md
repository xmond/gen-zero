# 31 - Single-GPU Breakthrough and Manifold-Anchor Distillation System Design for Extra-Large Dense Models (123B/180B/405B)

Task code: b0927c-t4-sys. Date: 2026-09-27.

**Nature: a design deliverable, not a claim of implementation, deployment, or production status.** This document does not modify any code file; it only provides an executable engineering roadmap, a verifiable list of assumptions, and acceptance criteria for every step. Every place that cites `path:line` reflects the real state of the current working tree, verified item by item with `Read`/`grep` (HEAD `acb2c0c`); every new component in the design is explicitly marked **unverified** or **not done**, with the reason stated.

## Status comparison matrix

This section was added in this revision (HEAD `edb3d78`). It verifies, as of this revision, whether each conclusion in the body of this document still holds. The body itself is left unchanged (preserving the original design reasoning); the latest status is appended here only, and where the body and this section conflict, this section takes precedence. **Note: the `path:line` references cited in §0-§3 of this document were all verified against HEAD `acb2c0c`. Since then, `crates/gen-zero-service/src/zero.rs` has gone through more than a dozen additional commits (the ETF choice-head rewrite, the contact-manifold integration, etc.), and line numbers have drifted overall by about +100 lines** (for example, `validate_nanocore` moved from `zero.rs:346` to `zero.rs:446`, and `nanocore_ask` moved from `zero.rs:2146` to `zero.rs:2246`). Function names and behavior are unchanged, only the line numbers have drifted; re-locate with `grep -n` before using any `path:line` reference from this document.

| Body item | Original status | Current status | Basis |
| :--- | :--- | :--- | :--- |
| §0: the GGUF truncation build script | Not done (§3.1's table marks it "new... [not done]") | **Implemented (the build tool itself), not wired into the extraction pipeline**. `scripts/slice_gguf_layers.py` (added in commit `2e1ef0e`, atomic-write and validation fixed in `655d533`; committed the same day as this document but after it, and not incorporated when this document was written) implements the byte-level slicing described in §1.2 route A: "keep `blk.0..K-1` plus `token_embd`/`output_norm`, drop `output.weight`." **The part that was not done is still not done**: no script calls it to drive the single-slot `llama-server` pipeline in `base.ServerEncoder`, and the "numerical-consistency check between the GGUF-truncation path and the safetensors-truncation path" required at the end of §1.2 is still not done (`grep -rln slice_gguf_layers .` matches only the script itself and its unit test) | `scripts/slice_gguf_layers.py:1-19`, `scripts/tests/test_slice_gguf_layers.py` |
| §2.2 step 5: "client.py currently does not call the Rust service... this wiring is entirely new code" | Not done | **Partially implemented, partially still not done**. Commits `6561f1a..6fb3c83` ("wire bridge into client/CLI," etc.) did add `GenZero.load_manifold_anchor_artifact` / `project_hidden_to_nanocore_state` / `generate_nanocore_ask_payload` (`python/gen_zero/client.py:3190-3211`) and a `python -m gen_zero.cli anchor` subcommand (`python/gen_zero/cli.py:408-460`). But `generate_nanocore_ask_payload` only **constructs** the MCP request JSON, it does not send it; the CLI's `--execute` path goes through `GenZero.decide_nanocore` (`client.py:3230-3274`), which is a **pure in-process Python** decision path (using Python's own `ActionETFChoiceHead`) that **never calls the Rust service at all**. In other words: "Python generates the 128-dimensional vector" is implemented, but "Python sends it to the running Rust `nanocore_ask`" still has no production code doing it -- the only thing that actually connects the two ends today is the Rust-side test `test_nanocore_live.rs` (which calls Python via a subprocess to generate the payload and compares it against the Rust computation with a tolerance, not byte-for-byte), and there is currently no production wrapper across processes calling it | `python/gen_zero/client.py:3190-3274`, `python/gen_zero/cli.py:408-460`, `crates/gen-zero-service/tests/test_nanocore_live.rs:1-25` |
| §2.4 Milestone 0: `StreamingCovarianceAccumulator` numerical fix, removing `argmin` from `detect_phase_transitions`, renaming "online SVD" | Blocking precondition, not done | **Still not done, blocking as before**. Both `git log -S"StreamingCovarianceAccumulator" -- python/gen_zero/causal/universal_manifold_extractor.py` and `git log -S"detect_phase_transitions"` match only the commit that originally introduced this code (`35c00ae`); no subsequent commit has modified either piece of logic. The "online SVD" name likewise appears only in the original design-document commit (`605cf9d`) and has not been renamed. All subsequent steps in §1-§3 that depend on Milestone 0 therefore remain entirely unstarted | `git log --oneline -S"StreamingCovarianceAccumulator" -- python/gen_zero/causal/universal_manifold_extractor.py` |
| §2 implicit assumption: "GCCA multi-view interference" feeds the 128-dimensional anchor basis | This document does not assert this directly, but the task title's description implies this chain | **The only anchor basis that actually runs today has nothing to do with GCCA.** `grep -rn "ManifoldAnchorDistiller(" .` shows `.fit()` is called in only two places: its own CLI (`manifold_anchor_distiller.py:325`) and the **synthetic-random-data self-check** in `profile_nanocore_latency.py:132` (that function's own docstring states "Throwaway artifact fit from random data; checks the harness, not real latency"). The one real artifact, `benchmarks/results/manifold/distilled_128d_llama70b_boolq.npz`, was fit directly on **raw LLaMA-70B hidden features** (`/ebs/data/extracted_features/llama70b/boolq.npz`), with no multi-view fusion from `python/gen_zero/manifold/gcca_fusion.py` in between. The GCCA fusion component (`python/gen_zero/manifold/gcca_fusion.py`, with 24 GCCA unit tests, 60 tests total across the manifold package) exists independently and is invoked by `cli.py manifold-fuse`, but it is used neither by the three-model Gaussian-random-projection benchmark script that was removed on 2026-09-29 (whose 123B feature dependency never actually existed, see the removal note at the top of `docs/manuals/closed_loop_dense_pipeline.md`) nor by the anchor-basis pipeline; it is currently a system that never calls, and is never called by, the anchor-basis pipeline -- two independent systems, not two stages of the same chain | `python/gen_zero/causal/manifold_anchor_distiller.py:325`, `benchmarks/suites/profile_nanocore_latency.py:117-132`, `crates/gen-zero-service/tests/fixtures/nanocore_anchor_state_boolq_row0.json` |
| §1: the teacher models are 123B/180B/405B | Design target | **The only teacher with a real artifact today is LLaMA-70B, which is not on the 123B/180B/405B list**, and it does not even use the layer-truncation route designed in §1 -- it uses the full model's final-layer features (`extracted_features/llama70b/boolq.npz` is a conventional extraction, not a GGUF-truncation artifact). For the 123B/180B/405B models, every step described in §1-§3 of this document is still at the design stage, with no new evidence | same fixture provenance field as above |
| §3.1: "there is currently no script in this repository that sets `GENZERO_NANOCORE_PATHS`" | Not done | **Still holds**; this revision re-checked with `grep -rn GENZERO_NANOCORE_PATHS --include=*.sh --include=*.bat --include=Makefile -r .`, zero matches | same grep as above |

**Conclusion of this section**: the body of doc 31's judgment on the full chain "405B single-GPU truncation + GCCA multi-view fusion + 128-dimensional anchor basis + NanoCore mounting" -- that most steps are not done -- still holds, and is now more precise than when the body was written: the GGUF-truncation tool has been built on its own, but it is not wired into the extraction pipeline; the Python-to-Rust payload construction has been written, but no production code path actually sends it; the one minimal closed loop that does work end to end (LLaMA-70B single-model features -> 128-dimensional orthogonal projection -> `nanocore_ask` -> gating -> decision) deliberately bypasses GCCA, and its teacher is not 405B either. The full practical steps for this minimal closed loop are in [`docs/manuals/closed_loop_dense_pipeline.md`](../manuals/closed_loop_dense_pipeline.md).

---

## 0. Laying the groundwork first: the system described in the task does not exist today

The task description assumes that "GGUF manifold extraction -> Rust decision engine" is an existing production chain. Verification does not support this assumption, and it must be corrected first, or everything designed afterward is building on air.

**The Rust workspace knows nothing about GGUF / GPU.**
- Of the 11 crates (`Cargo.toml:3-14`), the entire content of `crates/gen-zero-model` is an attention mask, an ETF choice head, and prompt sanitization (`crates/gen-zero-model/README.md:1-5`; the crate contains only `mask.rs/choice_head.rs/sanitize.rs/error.rs`) -- there is no concept of model loading, layers, or quantization types.
- The "manifold" in `crates/gen-zero-lod/src/manifold.rs` is mixed-curvature Riemannian geometry (hyperbolic/spherical product space, used for the entailment gate's math), and has nothing to do with LLM hidden layers.
- A repo-wide `grep -rniE "cuda|cublas|\bgpu\b|ngl|n_gpu_layers" crates/*/Cargo.toml crates/*/src/*.rs Cargo.toml` returns zero matches. The Rust side is a pure-CPU runtime.

**The existing GGUF/llama-server extraction produces "a single pooled vector," not a layered manifold.**
The `benchmarks/suites/gpu_extract_*_13tasks.py` family (`base.ServerEncoder`, `gpu_extract_gemma26b_13tasks.py:94-174`) takes, for each piece of text, only a single vector: the `--pooling last` **final post-norm last token**, explicitly documented as "the final post-norm state of the last token at its native scale" (`gpu_extract_qwen72b_13tasks.py:17-20`). What it already gets right: enforced single-slot verification (rejects startup if `total_slots != 1`, `gpu_extract_qwen72b_13tasks.py:81-89`), model-file hash/path verification (`:100-111`), verifying the integrity of prior artifacts on `--resume` rather than silently skipping (`:153-179`), and atomic writes (`.tmp.npz` + rename, `:146-150`). This is fail-closed infrastructure that can be reused directly, but what it extracts is **1 vector**, not a layered manifold.

**The "four-pillar" manifold-extraction code genuinely exists, but has never been run on a real large model and has a known numerical defect.**
`python/gen_zero/causal/universal_manifold_extractor.py` does contain four classes: `StreamingCovarianceAccumulator` (streaming covariance/PCA), `PhaseTransitionLayerExtractor` (phase-transition-layer detection via CKA), `CounterfactualGridSampler`, and `LyapunovPhaseSpaceReconstructor`. But:
- `PhaseTransitionLayerExtractor.detect_phase_transitions` takes a bare `np.argmin` over the CKA sequence (`universal_manifold_extractor.py:213,225`); `docs/zero/30-...md:189` has already pointed out that for a periodic architecture (such as Flash-Next's QSA layers), this selects the "architectural period" instead of the "conceptual phase transition."
- `StreamingCovarianceAccumulator` has a reproduced catastrophic-cancellation defect (`docs/zero/30-...md:15,209-228`), and calling a full `eigh` an "online SVD" is a terminology misuse (`docs/zero/qwen38-flash-next-extraction-system-design.md:31`).
- The project's own design document states this very plainly: "not one real Flash-Next forward pass has been run. This machine has no GPU... the 360GB of weights are not local" (`docs/zero/30-...md:19`), "Date: 2026-09-27. Nature: a design deliverable, not a claim of implementation, deployment, or production status" (`docs/zero/qwen38-flash-next-extraction-system-design.md:3-4`).

**"NanoCore" genuinely is a production component capable of consuming a 128-dimensional decision vector, but it does not produce that vector, only consumes it -- this is the only mounting point in this plan that actually holds up, and it is used below.**
`crates/gen-zero-nanocore/src/core_type.rs:19-31`: `NanoCoreInstance` carries `projection_weights: Vec<f32>`, with a comment explicitly stating "shape (out_dim, 128)," i.e. it always consumes a 128-dimensional input (`CompressedLatent = LatentState<128>`, `crates/gen-zero-core/src/types.rs:151`), projecting it to `out_dim` (<= 4096). The production side already has complete fail-closed validation (`validate_nanocore`, `crates/gen-zero-service/src/zero.rs:346-373`: `out_dim` range check, exact match of `projection_weights.len() == out_dim*128`, elementwise finiteness check) and a real call chain (the MCP `zero` tool's `engine=nanocore` branch -> `nanocore_ask`, `zero.rs:2146-2500`: verifies that `nanocore_state` is exactly 128 finite numbers, `zero.rs:2161-2176`). **This chain runs today; it's just that nobody feeds it a 128-dimensional vector related to any of the three large models.**

Conclusion: what this report needs to design is not "wiring into an existing extraction pipeline," but **building an offline extraction-to-distillation pipeline from scratch, and feeding its output into the already-existing, already production-mounted, but currently idle NanoCore 128-dimensional entry point**. §1-§3 below proceed on this real topology.

**`crates/gen-zero-model` has zero participation in this path, and the reason is stated to avoid confusion**: the only decision-related component in that crate, `ActionETFChoiceHead`, goes through the `head=etf` branch, and a request with `engine=nanocore` that also contains `etf_rep` is explicitly rejected (`crates/gen-zero-service/src/zero.rs:252`: the condition `head != "etf" && has("etf_rep") || engine == "nanocore" && has("etf_rep")` triggers a Rejection). The two decision backends are mutually exclusive; `gen-zero-model` is not on the nanocore mounting path, and this plan does not involve it.

---

## 1. Single-GPU breakthrough for 405B/180B: layer truncation is the only viable path, not an optimization

### 1.1 Do the math first, don't guess

Layer count / hidden dimension for the three models (parts directly verifiable from `.bat` comments, with sources noted):

| Model | Layers | Hidden dim | Quantization | Weight size | Source |
|---|---|---|---|---|---|
| Mistral Large 2 (123B) | 88 | 12288 (external spec, not found as a number in the repo; must be verified against the real file with `scripts/inspect_gguf_layer_bytes.py` before deployment) | Q3_K_M | ~60GB (an estimate; `run_mistral123b_extract.bat:13,21` explicitly says "ESTIMATE, not measured") | `benchmarks/suites/run_mistral123b_extract.bat:13` |
| Falcon-180B | 80 | 14848 | Q2_K | ~74GB | `benchmarks/suites/run_falcon180b_extract.bat:17-19` |
| Llama 3.1 405B | 126 | 16384 (external spec) | Q2_K | ~141GB | `benchmarks/suites/run_llama405b_extract.bat:2`, task description |

405B is the only one that physically does not fit in 80GB of VRAM (141GB > 80GB). Usable VRAM budget: 80GB minus KV cache (a few hundred MB to a few GB at `-c 2048` scale) and a compute buffer (estimated 2-5GB, per the Falcon 180B comment convention), figuring on ~75GB usable:

```
Average weight per layer ~= 141GB / 126 ~= 1.12 GB/layer
K_max ~= 75GB / 1.12GB ~= 67 layers
```

That is: **once truncated to roughly 64-67 layers (about half of the 126 layers), the remaining model can be fully resident in VRAM, with no CPU-GPU paging or NVMe streaming needed at all.** This is not an "optional speedup," it is the only path that makes 405B viable on a single GPU -- `run_llama405b_extract.bat:25`'s `-ngl 50` is itself annotated as "a starting configuration, NOT a measured fit," which shows that even the existing script's own author never actually verified this configuration works.

For 180B (Q2_K ~74GB, `run_falcon180b_extract.bat:18`): it can fit on the card without truncation ("at the edge of 80GB"); truncation's benefit here is freeing VRAM headroom for a larger batch/longer context, not survival.
For 123B (Q3_K_M ~60GB): the full model can run as-is, and it is the only model that can provide "real full-layer activations" as a ground truth -- this determines the experiment order in §1.3 below.

### 1.2 Two technical routes for layer truncation: pick one, don't do both

**Route A: physically truncate the GGUF file + reuse the existing single-slot llama-server pipeline (recommended)**

The GGUF format's tensor table carries absolute offsets; `scripts/inspect_gguf_layer_bytes.py:6-8` (verified real code, pure-stdlib header parsing, does not read the weights) has already proven that "a first-K-layers slice is a set of byte ranges, exactly like safetensors." Method:
1. Use `inspect_gguf_layer_bytes.py` to parse out the offset and size of every `blk.N.*` tensor.
2. Write a new GGUF file: keep all tensors of `blk.0..K-1` plus the token embedding, change the `n_layer` metadata to K, and append the original model's `output_norm`/`output` (or drop `output` and keep only `output_norm`, depending on whether logits are needed; this plan only needs hidden states, so the `output` tensor can be dropped -- for Mistral/Llama-class models, `lm_head` is typically `hidden x vocab` with a 30k-150k vocabulary, so dropping it saves several hundred MB to a few GB more).
3. Run this "K-layer pseudo-model" through the existing, already fail-closed-verified single-slot `llama-server --embedding --pooling last` pipeline, reusing `base.ServerEncoder`'s (`gpu_extract_gemma26b_13tasks.py:94-174`) and `Qwen72bEncoder`'s triple check of slot count / model hash / context length (`gpu_extract_qwen72b_13tasks.py:81-111`) directly.

**The distortion introduced must be stated explicitly**: what this obtains is `output_norm(h_K)`, not the original model's raw hidden state `h_K` at layer K. For Mistral Large 2 / Llama 3.1, which use RMSNorm, the distortion is "divide by the root-mean-square, multiply by a scale factor"; **Falcon-180B uses LayerNorm with bias (subtract the mean, divide by the standard deviation, then multiply by a scale and add a bias) -- not RMSNorm**. The same "RMSNorm(h_K)" story cannot be applied to all three models; Falcon requires a separately handled inverse normalization. This is a fixed, predictable, per-model bias, not random noise, but if downstream consumers need the "raw `h_K`," an explicit inverse-normalization transform must be applied, or this bias accepted -- **no code has yet measured how large the difference between the two actually is; this is marked unverified.**

**Route B: use llama.cpp's eval-callback to capture h_K exactly (not recommended, high effort)**
Use `llama.cpp`'s `cb_eval` or `llama-cpp-python`'s callback mechanism to intercept the raw hidden state directly right after layer K finishes computing, without modifying the file. Exact, but requires writing entirely new C++/Python bridging code, and none of the existing fail-closed verification can be reused -- the slot/hash/resume checks in the `gpu_extract_*` family would all have to be rewritten from scratch.

Choosing route A: it lets us reuse the already-verified, production-grade single-slot pipeline (`total_slots` check, model-hash verification, atomic resume), at the cost of accepting a known, measurable, documentable `output_norm` distortion. Route B is more accurate but amounts to reinventing the wheel, and there are currently no resources to verify the correctness of new bridging code.

**Acceptance criteria (must be obtained before route A ships, none optional)**:
- Use the already-verified `rel_err` method from `scripts/test_intermediate_layer_probe.py` (`max_abs`/`rel_l2`/`mean_cos`, `test_intermediate_layer_probe.py:422-425`) -- but note this is a safetensors + HF `transformers` code path, a completely different code path from GGUF + llama-server, and **no code today has cross-validated the two.** Before shipping, a run must first be done: for a small model (e.g. Qwen3.5-9B, weights already on hand), load the same layer via both safetensors truncation and GGUF truncation, compare `rel_l2`/`mean_cos`, confirm the two paths agree numerically, and only then generalize to 123B/180B/405B. This is a new, currently-not-done verification step; the reason is that the two paths have each been independently verified but never cross-checked against each other.

### 1.3 How the phase-transition layer K is determined: not a guessed number

`docs/zero/30-...md:189` has already pointed out that the argmin-CKA method picks the wrong layer on a periodic architecture. This plan does not guess any specific K value; instead it prescribes an experiment order:

1. **Do a full-layer scan on 123B first** (it is the only one of the three models that fits entirely in VRAM and can produce real activations from all 88 layers). Use the existing `linear_cka`/`compute_cka_matrix` (`universal_manifold_extractor.py:154-179` -- these two functions themselves have no argmin problem; the problem is only in `detect_phase_transitions`'s decision rule) to compute the full-layer CKA matrix, and manually verify the candidate transition layers rather than trusting the raw output of the automatic argmin.
2. **Simultaneously fix `detect_phase_transitions`**: it can no longer use the raw `argmin` (`universal_manifold_extractor.py:213,225` -- this is old logic to be replaced, not extended); switch to the period-aware detrended CKA method (`docs/zero/30-...md` §5 already provides the method, and this plan adopts it).
3. Use 123B's calibrated K result (a "relative depth fraction," e.g. a transferable relative metric like "K/88 ~= 0.6," rather than an absolute layer number) as a prior applied to 180B/405B, but **it must be re-checked against small-scale sampling on each of 180B/405B individually**; the absolute layer number cannot simply be carried over (depth semantics are not comparable across different architectures).

Until the real scan results for 123B are obtained, any specific number claiming "layer N is the phase-transition layer" is an unverified guess, and this report does not provide one.

**Quantization's effect on manifold fidelity, computed separately from the VRAM budget -- the task explicitly requires this, but it is currently missing**: `run_falcon180b_extract.bat:17-23` and `run_llama405b_extract.bat` use Q2_K, one of the most aggressive tiers in llama.cpp's quantization scheme. The VRAM budget (§1.1) only answers "does it fit"; it cannot answer "does the Q2_K quantization error itself damage the manifold structure of `h_K`, such that the distilled teacher target is teaching a distorted manifold." This is an **unverified** item, and the measurement that must be done before shipping is: on 123B (which already has a Q3_K_M configuration on hand), run both Q3_K_M and a lower tier (e.g. Q2_K, or Q4_K_M as a control), take layer-K outputs for the same batch of inputs, and compute the per-sample cosine-similarity distribution (not a single mean number), to confirm whether the quantization error is acceptable in terms of manifold fidelity. If only Q2_K weights are available for 405B/180B and severe distortion is measured, that puts a ceiling on the whole distillation pipeline's teacher quality, and that ceiling must be measured after Milestone 0 and before real training begins -- it cannot be assumed that "if it fits, it's usable."

### 1.4 CPU-GPU dynamic swapping / NVMe streaming

`scripts/prototype_layer_streaming.py` has already verified the mechanism (not the model): using synthetic weights (`HIDDEN=5120`, 40 heads, unrelated to any real model, `prototype_layer_streaming.py:38-44`), it compares three strategies -- `full_preload` (fully resident), `naive_offload` (single-buffer synchronous paging), and `double_buffer` (double buffering + background-thread prefetch + a dedicated CUDA stream, with H2D copy overlapping computation). **This proves the double-buffered pipelining mechanism itself works, but it has never been tested against real model weights**; marked unverified.

Once §1.1's arithmetic is settled, after truncating 180B/405B to K layers the whole model fits in VRAM, so **double-buffered paging is in fact not needed in this plan** -- this is a direct corollary of §1.1's arithmetic: if the goal is only to obtain layer K's hidden state (not to run the entire model), layer truncation has already turned the problem into "a small model that fits in VRAM," and mechanisms like NVMe streaming that "load the back half of the weights while computing" are not needed, because the back half of the weights is never loaded at all. Double-buffering/NVMe streaming only has value in an iterative scenario, such as "needing to explore multiple candidate K values, each requiring a reload at a different truncation point" (avoiding re-reading all truncated layers from disk every time); it is an engineering optimization for accelerating the §1.3 experiments, not a requirement of the main chain.

---

## 2. Offline manifold-anchor distillation + NanoCore instantaneous projection: who produces the 128-dimensional vector

### 2.1 The real gap is the "producer," not "the projection itself"

`NanoCoreInstance`'s `projection_weights` is `(out_dim, 128)` -- the input side is already fixed at 128 dimensions (`crates/gen-zero-nanocore/src/core_type.rs:19-31`), and at runtime these 128 dimensions must be supplied directly by the caller in the request (`nanocore_state`, `crates/gen-zero-service/src/zero.rs:2161-2176`, exactly 128 finite numbers; a dimension mismatch is rejected outright, with no silent padding or truncation).

The question is: on a CPU-only production server, at the moment a request arrives, who computes these 128 dimensions? It cannot be "run a projection of a 16384-dimensional teacher hidden state" -- **production has never had, and will never have, a real 405B forward pass.** If the task description's phrase "continuous dynamical integration from 16384 dimensions into decision space" is read as "every request obtains a real 405B layer-K hidden state and then projects it," that is itself an untenable producer assumption, and it must be debunked before designing further.

### 2.2 The correct division of roles: the projection matrix is a training-time tool, not a runtime weight

Following `docs/zero/README.md:3-5`'s own positioning ("the 9B teacher model... distills continuous causal-manifold structure and counterfactual dynamics into Zero's weights via structured parameter slicing extraction and backpropagation"), this plan adopts the same division of roles, extended to the three new teachers 123B/180B/405B:

**Offline (one-time, run on a GPU machine)**:
1. §1's truncation + extraction pipeline: for each model's phase-transition layer K, run a batch of inputs covering the real task distribution, obtaining `(input, h_K)` pairs.
2. Use a fixed `StreamingCovarianceAccumulator` (fix the numerical defect first, see §2.4) to compute a principal-component/thin-SVD dictionary for h_K, or train a conformal projection mapping h_K into a 128-dimensional target space -- the output of this step is **a projection matrix used as a training-target generator, not loaded as a runtime weight.**
3. Use this projection matrix to convert the whole batch of `(input, h_K)` pairs into `(input, target_128d)` supervision pairs.
4. **Offline-train a CPU-friendly small encoder** (reusing existing distiller infrastructure: the distiller in gen-zero-research, moved out of this repo), so it predicts `target_128d` directly from the raw input (without going through the teacher model). This small encoder is the true **producer** of the 128-dimensional vector -- it runs on CPU at inference time, with no dependency on the teacher model or a GPU.

**Two distinct artifacts must be kept separate here, not conflated** -- `NanoCoreInstance.projection_weights` has shape `(out_dim, 128)` (`core_type.rs:19-31`), i.e. it consumes 128 dimensions and emits `out_dim` dimensions, which is the exact opposite direction of step 4's "raw input -> 128 dimensions" encoder; it is physically impossible to package step 4's encoder directly as a `NanoCoreInstance`. What actually needs to ship is two independent artifacts:

- **Artifact (a): the client-side 128-dimensional encoder** (the output of step 4). It maps raw input to a 128-dimensional `CompressedLatent`, running on the caller's side (`python/gen_zero/client.py` or another upstream service); **it is not loaded into `NanoCoreInstance`, nor loaded on the Rust side** -- it only computes the 128-dimensional numbers and sends them as part of the request parameters to the Rust service.
- **Artifact (b): the server-side `NanoCoreInstance`** (128 dimensions -> `out_dim` decision score). Its `projection_weights`/`value_weights` cannot be randomly initialized or a placeholder -- they must be fit from `(target_128d, decision_label)` supervision pairs (`decision_label` comes from historical decision outcomes or human annotation for the specific business scenario, not from the teacher hidden state itself); otherwise, the 128-dimensional vector fed in is multiplied by a matrix with no relation to the manifold, and the whole chain is semantically empty even though its shapes line up. **The training data for this step (decision labels) does not exist at all right now, and is the single largest not-done item in this plan**, coming even later than encoder training because it depends on the specific business scenario's definition of a "correct decision."
5. Deploy artifact (a) wherever the caller can execute it (in-process in Python, or a standalone microservice); export artifact (b), per model/domain, to the JSON format `NanoCoreInstance` expects (`domain_id/name/prototype/projection_weights/value_weights/out_dim/base_confidence`; the serialization format is deserialized with `serde_json::from_slice` in `load_nanocores_from_paths`, `crates/gen-zero-service/src/zero.rs:401-403`), and place it at the path(s) pointed to by `GENZERO_NANOCORE_PATHS` (constant name `NANOCORE_PATHS_ENV = "GENZERO_NANOCORE_PATHS"`, `zero.rs:324`; a documented operator configuration item, `crates/gen-zero-service/README.md:330`; **there is currently no deployment script in this repository that sets this variable**, meaning: the production code path already exists, but there is no deployment configuration to trigger it to load any real core).

**Online (the production request path, CPU only)**:
1. A request arrives -> artifact (a)'s encoder (running on CPU, not the teacher model) computes a 128-dimensional `CompressedLatent`; timing is discussed in §2.3.
2. These 128 dimensions are sent as `nanocore_state` in an MCP `tools/call zero` request (`engine=nanocore`). **This step is new code in today's call chain, not "wiring into an existing chain"**: `python/gen_zero/client.py` currently does not call the Rust service at all; the actual wiring runs the other way (Rust `gen-zero-service`'s `bridge.rs` -> HTTP -> Python `app.py`). To get artifact (a)'s output actually flowing to Rust, a new "caller -> MCP `zero` tool" client code path must be written; `client.py` has no such code today, and it must be built from scratch.
3. Once the request arrives, `nanocore_ask` (`zero.rs:2146-2500`) goes through the **existing, already-runnable-today** validation-plus-inference chain: `validate_nanocore`'s dimension/finiteness check (`zero.rs:346-373`) -> `NanoCoreFleetScheduler` (`crates/gen-zero-nanocore/src/scheduler.rs`, a bounded-RAM LRU) + `MoVFusionEngine` (`mov.rs`, multi-domain vector fusion) completes the decision and returns.

In this design there is no runtime dependency at all between "the 16384-dimensional teacher hidden state" and "sub-millisecond decisions on CPU" -- the teacher appears only during offline training. This is the only architecture that can satisfy both "CPU-only production" and "sub-millisecond latency" simultaneously, but **only once artifact (b) has actually been trained with decision labels does this chain carry a decision signal; otherwise it is a pipe with the right shape and empty semantics, which is equally an island (it is called, but carries no signal).**

### 2.3 Size and latency, computed in a direction that actually matches up

`NanoCoreInstance::projection_weights` is `(out_dim, 128)`, i.e. `out_dim x 128 x 4` bytes in f32. If the small encoder itself is also a linear/shallow network (its input dimension depends on the raw features, not 16384 -- it starts from the raw input and does not need the teacher hidden state), its size is determined by the encoder architecture, not by the "teacher projection matrix"; the `16384x128` arithmetic cannot simply be reused here to claim "tens of MB." The honest statement is:

- Teacher-side projection dictionary (an offline tool, one-time): a single model is `16384x128 f32 = 8.4MB` (Llama 405B), `14848x128 f32 ~= 7.6MB` (Falcon 180B), `12288x128 f32 ~= 6.3MB` (Mistral 123B); the three models together total about **22MB in f32 / 6MB with int8 quantization** -- this order of magnitude matches the "tens of MB" claim, but it is an offline training tool and is not loaded into the production server.
- Production-side small encoder: size depends on the encoder architecture chosen (this report does not presume one; it needs an architecture search on a validation set once §1.3 has produced real teacher targets); this is a **not-done** item, because there is currently no real `(input, target_128d)` supervision data to train on.

A single `128 x out_dim` linear projection (inside `MoVFusionEngine`) is indeed sub-millisecond by itself: with `out_dim` at its upper bound of 4096, MACs ~= 128x4096 ~= 520,000, and at an estimated single-core AVX-512 throughput of ~50 GFLOPS, that's ~0.02ms. **But this is only that one projection inside NanoCore, not the end-to-end latency from "the 16384-dimensional teacher hidden state" to "decision space."** The bulk of the end-to-end latency is in the small encoder's forward pass, whose cost is determined by an architecture not yet decided in §2.2 step 4, and no specific number can be committed at this time. The task description's phrase "sub-millisecond completion of continuous dynamical integration from 16384 dimensions into decision space" only holds if "16384 dimensions" is understood as "the dimension used during the teacher's offline training," not "the runtime input dimension"; as a runtime end-to-end commitment, this report does not endorse it, marking it not done, for the reason that the small-encoder architecture is undecided and there is no real training data.

### 2.4 Blocking precondition (Milestone 0, must be done first, or everything after it is void)

The catastrophic-cancellation defect in `StreamingCovarianceAccumulator` (`docs/zero/30-...md:15,209-228`) must be fixed first. Any SVD dictionary/projection matrix computed on top of it will inherit this numerical error, and the distilled small-encoder target would itself be wrong -- until this is fixed, the entire offline chain in §2.2 cannot begin running on real data, and can only verify the mechanism on synthetic data. This is Milestone 0 of this plan, with no conditional bypass.

A naming issue must also be addressed at the same time: calling a full `eigh` an "online SVD" is a terminology misuse (`docs/zero/qwen38-flash-next-extraction-system-design.md:31`); either a genuine incremental SVD (e.g. Brand's algorithm) must be implemented, or the "online SVD" wording in the function/documentation must be changed to an accurate name -- these are old symbols this plan requires to be physically replaced, not kept behind a compatibility layer.

---

## 3. Rust trunk mounting topology + end-to-end flow + degradation/alerting + acceptance metrics

### 3.1 Caller -> Callee topology (distinguishing "already exists, runs today" from "needs to be built")

```
[Offline, one-time, GPU machine, never enters the production server]
benchmarks/suites/run_{mistral123b,falcon180b,llama405b}_extract.bat  [exists, needs conversion to a truncated-GGUF version per §1.2]
  -> New: GGUF truncation build script (reads the offset table from inspect_gguf_layer_bytes.py, writes a new GGUF)  [not done]
  -> Reuse: base.ServerEncoder single-slot llama-server pipeline                    [exists, gpu_extract_gemma26b_13tasks.py:94-174]
  -> Fix: StreamingCovarianceAccumulator (fix the numerical defect first)           [Milestone 0, not done]
  -> Fix: PhaseTransitionLayerExtractor.detect_phase_transitions (replace argmin)   [not done]
  -> New: teacher projection dictionary (SVD/conformal, 128-dim target space)       [not done, depends on Milestone 0]
  -> Reuse: gen-zero-research distiller (moved out of this repo) (trains artifact (a), the CPU small encoder)  [framework exists, needs wiring to new data source]
  -> New: artifact (b) NanoCoreInstance fitter (128-dim target + decision label -> projection_weights/value_weights)  [not done, missing decision-label data]

[Online, production request path, CPU only]
Caller (new code: python/gen_zero/client.py currently does not call the Rust service, a new MCP client path must be written)
  -> Artifact (a) encoder forward pass (CPU, new artifact) -> 128-dim CompressedLatent    [depends on the offline artifacts above]
  -> MCP tools/call "zero", engine=nanocore, nanocore_state=<128-dim>   [server side exists, crates/gen-zero-service/src/zero.rs:2146; the caller side is new code]
  -> validate_nanocore dimension/finiteness check                                 [exists, zero.rs:346-373]
  -> NanoCoreFleetScheduler + MoVFusionEngine fuse artifact (b)'s decision          [exists, crates/gen-zero-nanocore/src/{scheduler,mov}.rs; needs artifact (b) loaded first]
  -> ZeroToolOutcome -> MCP/HTTP response                                   [exists, server.rs:1253-1316]
```

**This diagram is deliberately split into two stages so that the "projection matrix" does not become a zero-call island**: the teacher projection dictionary is actually consumed, in the offline stage, by artifact (a)'s encoder training; artifact (b) (the part that actually carries the decision signal) is actually invoked, in the online stage, by `nanocore_ask` -- but today there is no caller code at all to trigger that call (`python/gen_zero/client.py`'s current state is the wiring running the other way: Rust `bridge.rs` -> HTTP -> Python `app.py`; the client-to-MCP-`zero`-tool segment is entirely new code, not "reusing an existing chain").

### 3.2 Degradation and alerting: no silent bypass allowed

- `nanocore_state` with the wrong dimension or containing a non-finite number: **already fail-closed**, `zero.rs:2161-2176` rejects it outright, no new code needed.
- Missing or failed load of a projection artifact (a `NanoCoreInstance` file): `load_nanocores_from_paths` calls `unwrap_or_else(|error| panic!(...))` (`zero.rs:399-415`) for all four cases -- file-read failure, JSON deserialization failure (`serde_json::from_slice`), `validate_nanocore` check failure, and duplicate `domain_id` -- **already verified as fail-closed: if any one configured core fails to load, the entire service exits with a panic at startup, rather than skipping the single file and continuing.** This behavior is correct as-is (it prevents quietly going live with a bad core) and needs no change when shipping.
- If the small encoder's output is all-zero/NaN (offline training failure, or the wrong model's encoder was loaded): it must not be silently passed to NanoCore as a normal request -- the upstream client (the new MCP client code) must perform a finiteness self-check before sending `nanocore_state` and raise an explicit error; this is new code, **not done**, and requires exporting the expected input distribution's statistics (mean/variance/norm range) alongside the offline artifact export, so the online side can detect anomalies rather than relying on a Rust-side fallback.
- Forbidden anti-pattern (excluding it explicitly against the task's anti-rot requirements): silently degrading to "fall back to brute-force full-model inference if NanoCore has no core for the matching domain" is not allowed -- today's code already does the opposite: when `operator_domains.is_some() && matches!(specialized, Ok(true))`, it explicitly rejects with "nanocore_domain(s) cannot combine with engine/head decision backends" (`zero.rs:1491-1498`, this is the code's own logic, not something inferred from a test name). This design must be preserved when shipping; no fallback path should be added just to "always produce some result."
- **The role of "Speculative Latent Cache" (a concept mentioned in the task title) in this plan**: it can only be an "exact input-match cache" layer, placed in front of artifact (a)'s encoder -- if the request's raw input exactly matches a previously seen input (or falls within a strictly defined neighborhood radius), the cached 128-dimensional vector is returned directly, skipping one encoder forward pass. It does not solve the "producer" problem (an unrecognized new input still needs the encoder); it is only a speedup layer for repeated requests. **A cache miss must fall through to the encoder computation, and must never return a zero vector or the nearest cached entry as a substitute** -- this is a fail-closed rule that must be written explicitly at implementation time; it is not this plan's default behavior.

### 3.3 Performance benchmarks and acceptance metrics (layered, not merged into one number)

| Stage | Metric | Criterion |
|---|---|---|
| Milestone 0: numerical fix | `StreamingCovarianceAccumulator` no longer cancels on a known ill-conditioned constructed input | Unit test: construct a set of inputs known to trigger catastrophic cancellation, compare the error before and after the fix; the error magnitude must return to floating-point precision range |
| §1.3: phase-transition-layer scan | 123B full-layer CKA matrix + manually verified candidate transition layers | Produce the CKA matrix plus a candidate-layer interval cross-confirmed by at least 3 independent perspectives (architectural prior, CKA extremum, downstream-task probe linear separability), not a single argmin output |
| §1.2: truncation fidelity | `rel_l2`/`mean_cos` between the GGUF-truncation path and the safetensors-truncation path (`test_intermediate_layer_probe.py`) on the same small model | `rel_l2` must be within a predefined tolerance (cannot be fixed until after §2.4 is fixed, to be set from real data after Milestone 0), and must be computed as paired per-sample statistics, not a single mean number |
| §2.2: small-encoder distillation | Per-sample cosine similarity between the encoder's 128-dimensional output target and the teacher's projection target | Mean + distribution (mean alone is insufficient), plus a downstream decision-quality comparison on a held-out validation task (with encoder vs. without, using real business metrics); no "breakthrough" claim without this comparison |
| §3.1 online path | End-to-end request latency (small-encoder forward pass + NanoCore projection + fusion) | Command, exit code, and raw load-test log tail (e.g. from `wrk`/`hey` or the project's own load-test tool); "estimated" or "in theory" is not accepted |

### 3.4 Inventory of old logic/old symbols (anti-rot item 4)

Items that must be physically replaced, with no compatibility layer kept:
- The bare `argmin` decision in `PhaseTransitionLayerExtractor.detect_phase_transitions` (`universal_manifold_extractor.py:213,225`) -- replace with the period-aware detrended method.
- `StreamingCovarianceAccumulator`'s current algorithm -- replace with a numerically stable version.
- The inaccurate "online SVD" naming in documentation/code -- handle per §2.4.

There is no old Rust code to delete, because there is currently no GGUF/large-model-related Rust code in existence at all (verified in §0); this is not an omission, it is simply the current state of affairs.

---

## 4. Summary in three conclusion categories (closing per the task's required framework)

**Implemented (with evidence)**:
- Fail-closed validation in the single-slot llama-server extraction pipeline (slot count, model hash, context length, resume integrity, atomic write) -- `gpu_extract_qwen72b_13tasks.py:81-179`.
- Byte-level parsing of the GGUF header (does not read the weights) -- `scripts/inspect_gguf_layer_bytes.py`, directly reusable for the §1.2 truncation build.
- A safe truncated-load-plus-fidelity-verification methodology (`rel_err`) -- `scripts/test_intermediate_layer_probe.py:365-425`, but only on the safetensors path, and only "proven on a tiny model" (not verified on a real large model).
- Feasibility of the double-buffered layer-streaming mechanism -- `scripts/prototype_layer_streaming.py`, but only verified on synthetic weights.
- The NanoCore 128-dimensional projection entry point + fail-closed validation + production MCP mounting -- `crates/gen-zero-nanocore/src/core_type.rs:19-31`, `crates/gen-zero-service/src/zero.rs:346-373,2146-2500`. This chain runs today; it just has no artifact feeding it.

**Unverified (done, but with no evidence that it is correct)**:
- Numerical consistency between the GGUF-truncation path (route A) and the safetensors-truncation path -- the two code paths have never been cross-validated.
- The measured benefit of double-buffered streaming on real model weights (as opposed to synthetic weights).
- The effect of aggressive quantization tiers such as Q2_K/Q3_K_M on `h_K`'s manifold fidelity -- so far only whether it fits in VRAM has been tested, not whether the quantization error has already damaged the manifold structure itself (end of §1.3).

**Verified (established as fact, not an assumption)**:
- `load_nanocores_from_paths` uses `panic!` to fail the service at startup for all four cases -- read failure, deserialization failure, `validate_nanocore` check failure, and duplicate `domain_id` -- rather than skipping a single bad file and continuing (`zero.rs:399-415`).
- `gen-zero-model`'s `ActionETFChoiceHead` is mutually exclusive with `engine=nanocore` and is not on this plan's mounting path (`zero.rs:252`).
- `GENZERO_NANOCORE_PATHS` (`zero.rs:324`) is a real, documented (`crates/gen-zero-service/README.md:330`) operator configuration item, but there is currently no script in this repository that sets it -- the production code path exists, but there is currently no deployment configuration to trigger it to load any real core.

**Not done (not attempted, with the reason)**:
- The real phase-transition layer K for each of 123B/180B/405B -- requires running the §1.3 full-layer scan first; this report does not presume a specific layer number.
- The teacher projection dictionary and artifact (a)'s CPU small encoder -- depends on Milestone 0 (the numerical-defect fix) being completed before training on real data can begin; currently only the framework (`distiller.py`) exists, with no data.
- Artifact (b), the `NanoCoreInstance` (`projection_weights`/`value_weights` fit to carry real decision semantics) -- missing decision-label data; this is a step further out and harder to fill than encoder training, because it depends on the specific business scenario's definition of a "correct decision"; without it, the online path is only an empty pipe whose shapes happen to line up.
- The client code from the caller to the MCP `zero` tool -- `python/gen_zero/client.py` does not call the Rust service today; this wiring is entirely new code, not a reuse.
- A specific committed number for end-to-end online latency -- the encoder architecture is undecided, so it could not previously be measured; "a theoretical estimate" is not accepted as an acceptance result.
- The GGUF-truncation build script and the `NanoCoreInstance` format exporter -- both are entirely new code, not yet written.
