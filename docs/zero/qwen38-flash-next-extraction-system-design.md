# Qwen3.8-Flash-Next Physical Extraction and Systems-Engineering Implementation Plan

Date: 2026-09-27. Nature: a design deliverable, not a claim of implementation, deployment, or production status.
Code baseline: `fa6cddb656f49d9ae01bf417e476514962e03a34`. No existing production code was modified.

## 1. Decisions and the boundary of what is established fact

Recommendation: adopt two clearly isolated paths: **an in-model extraction path via Transformers/PyTorch as the representation baseline; vLLM or SGLang as a subsequent throughput-optimization path, which requires adding an in-worker extraction adapter and passing a per-sample consistency check.** Successful HTTP text generation does not prove that intermediate-representation extraction has succeeded. The existing llama-server's final pooled embedding cannot substitute for a specified intermediate layer.

Confirmed by the official model card: a 125B main model with 6B active per token, plus a separate 51B N-gram embedding and 4B MTP; hidden width 2560, 48 layers, 4 gated-residual branches, 512 experts (10 routed + 1 shared); N-gram at layer 2, table size roughly 20,000,000; attention is a GDN/QSA hybrid. The loader must not be guessed from the module names or tensor layout of the old Qwen3-Next, Qwen3.5, or a generic decoder.

Sources: the [official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), the [official repository](https://github.com/QwenLM/Qwen3.8-Flash-Next). Web scrapes are saved at `.firecrawl/qwen38-official.md`, `qwen38-github.md`; the file summary is in this report's evidence, `commands.json`. The model config and weight index have not yet been downloaded and verified; the actual byte counts, module paths, dtype, and loader version must still be confirmed against files at a fixed revision.

The most important prohibitions:

- **6B active does not mean 6B resident.** Experts not selected by the router must still have full backing storage; the union of experts across a prefill batch can cover a large number of experts.
- **90GB of RAM cannot hold the 51B BF16 table.** A successful mmap or the absence of a swap error so far must not be used to declare host offload viable.
- **A 2560 hidden size does not mean every hook returns `[B,T,2560]`.** The 4-branch residual must distinguish the gated read, the full branch state, and the final norm; silently flattening/averaging/selecting a branch is not allowed.
- "Can run generation" and "can extract a specified layer" are two separate acceptance items; a webpage's claim of support is not a local measurement either.
- Geometric representations, PCA, CKA, and stability matrices must never be automatically labeled a causal capability, a phase transition, or a reasoning breakthrough.

## 2. The most critical problems in the existing code

| Location (repo-relative path:line) | Verified behavior | New-plan requirement |
|---|---|---|
| `benchmarks/suites/run_universal_extraction_a100.py:27` | Windows path, hardcoded to an old model | Parameterized path, pinned model revision/hash; the model must never be silently substituted |
| same file `:34`, `:44`, `:60` | Scans arbitrary JSONL, guesses fields, `continue`s straight through exceptions | Reuse the 13-task schema; raise with file/line number; fail on a missing task |
| same file `:97`, `:115` | Fixed 4096-dim; enables all hidden states | Derive the dimension from a verified adapter; capture only the target position |
| same file `:99`, `:124`, `:147` | Keeps sample lists and diff lists | Fixed queue/chunked persistence; an accumulator's O(d^2) must not stand in for the whole pipeline |
| same file `:131` | Generates so-called counterfactuals via a sentence field / regex | Format parsing is not a counterfactual intervention; this capability must not be claimed without a predefined paired experiment |
| same file `:158`, `:170`, `:175`, `:202` | Calls a full eigh an "online SVD"; declares a phase transition from CKA alone; treats unrelated sample order as a trajectory | Correct the terminology; do not produce undefined causal/dynamical artifacts |
| `python/gen_zero/causal/universal_manifold_extractor.py:83`, `:102` | Computes covariance by subtracting raw second moments | Switch to merged central moments, and test with large bias / small variance |
| same file `:115`, `:121` | eigh produces extra workspace; `memory_bytes` only counts persistent state | The memory budget must cover matrix copies, temporaries, BLAS workspace, and the queue |
| same file `:481` | sha256 covers only the concatenated array bytes, not names/shape/dtype/metadata; writes to the destination directly | A full file digest, strict schema, atomic publish, and a manifest chain |
| `benchmarks/suites/gpu_extract_qwen72b_13tasks.py:100` | Verifies the model by basename; returns unverified when the path is missing | The new path must require revision and shard hashes; reject if the path is unknown |
| same file `:123` | Token budget can fall back to a value with a provenance flag | A strict run manifest must require the full budget; a default number must not simply be carried over |
| `benchmarks/suites/cpu_extract_gte7b_13tasks.py:83` | Model-specific trailing-token handling | Reuse the data contract; do not copy GTE's token special-case onto the new model |

The numerical problem has already been reproduced with the real class, but the input is a mathematical diagnostic sample, not model activations: `1e8 + arange(32).reshape(16,2)` split into four batches; the current implementation gets `[[86,86],[86,84]]`, while the centered-first reference gets `[[85,85],[85,85]]`, a maximum absolute error of 1.0. The command's raw exit code of 0 indicates the diagnostic ran successfully, not that the numerics are correct. See `evidence/qwen38-extraction-design/numerics.log` and `commands.json`.

A call-site search found the accumulator is invoked by `gepa_daemon.py:39,66,82`, so the module as a whole cannot be claimed to have zero references; but this search found no evidence of Rust calling `load_codebook`. The search scope/command is in the evidence; it is not proof covering every dynamic-loading path. A real production entry point must be added and verified; getting a benchmark to run standalone must not be treated as going live.

## 3. Capacity ledger and hardware deployment

The following are decimal-GB bare-parameter lower bounds, excluding quantization scales, padding, the vision module, loading peaks, caches, and workspace; GB and GiB are not mixed.

| Precision assumption | 125B main model | 51B table | 4B MTP | Total lower bound |
|---|---:|---:|---:|---:|
| BF16, 2 bytes/param | 250 | 102 | 8 | 360 |
| All 8-bit, 1 byte/param | 125 | 51 | 4 | 180 |
| All 4-bit, 0.5 byte/param | 62.5 | 25.5 | 2 | 90 |

"All 4-bit" is only an arithmetic lower bound; it does not imply a corresponding kernel or checkpoint actually exists. Real mixed quantization may be substantially larger than this figure. At scrape time, the vLLM recipe panel listed BF16 423GB, FP8 250GB, NVFP4 130GB; one NVFP4 variant in the SGLang documentation was about 126GiB, of which the FP8 table was about 47.7GiB. These are not interchangeable exact sizes for one unified checkpoint, which further shows that the target repository's file manifest must be read directly; disk cannot be reserved just by multiplying parameter count by bytes.

Sources: the [official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next), the [official SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next). The vLLM overview's wording on "whether the 125B figure includes the table" is ambiguous relative to the official card; this report uses the official model card's additive accounting.

| Node plan | Design judgment and constraints |
|---|---|
| dev: 64 CPU / 90GB RAM / 316GB free disk | Suitable for scheduling, manifest handling, covariance, and artifact validation. Cannot accept a full BF16 download/conversion; cannot keep the full BF16 table resident on host. The CPU-only 4-bit 90GB bare lower bound already leaves no runtime headroom, and cannot serve as a fallback plan. |
| 1xA100/H100 80GB + 90GB host | No solution for a BF16 backbone. Specific verified mixed quantization + an FP8/INT8 table resident on host may have a capacity opportunity, but kernel support and peaks must be measured; not committed as the first full-extraction attempt. |
| 2x80GB + 90GB host | May be enough for certain quantized backbones + a host table; cannot be guaranteed from active-parameter count alone. Model-parallel layout, workspace, and inter-GPU bandwidth need separate verification. |
| 4x80GB + >=192GB host | The leading candidate for a ~250GB BF16 backbone on GPU with a 102GB table on host; whether 192GB is enough needs accounting that includes the loading peak, and 256GB gives more headroom. Recommend also provisioning >=1TB of free fast storage for the pinned checkpoint and conversion temporaries. |
| 4x80GB + 90GB host | Worth exploring an FP8 table, or a table sharded across CPU/GPU, if a fixed variant and kernel support exist; a full BF16 table on host is not adopted. H100's FP8 path and A100's compatible kernels must be verified separately. |
| 8x80GB | A fully GPU-resident BF16 candidate for a correctness baseline; per-GPU available space, sharding constraints, and loading disk still need checking, and this does not mean it has already been run. |

A100 must not be treated as having H100's native FP8 path; neither A100 nor H100 can simply reuse B200/Blackwell's NVFP4 recipe. Acceptance must use the actual matrix of GPU, checkpoint, and kernel combination in use; dtype must not be auto-converted, the quantization format must not be swapped, and there must be no silent fallback to CPU.

Before loading, output a per-device memory plan: tensor owner, storage dtype, compute dtype, bytes, CPU/GPU location, copy count, prefill workspace, GDN state, QSA/indexer/cache, hook staging, and temporary loading space. Admission is based on **available** RAM/VRAM, not total. The overall budget is `weights + states + activations + workspace + staging + safety_margin`; verify at batch=1 and the task's longest input before deciding the token budget.

No SSH probing of dev/stg/ai-wsl was done this time, and no user-provided actual GPU count/interconnect information was found; the above is conditional sizing. At execution time, first collect `uptime`, `/proc/loadavg`, `free -b`, `df -B1`, `nproc`, `nvidia-smi`, `nvidia-smi topo -m`. Normalize 1m/15m load by core count, and write the admission threshold into the manifest; >=8GB RAM and >=10GB disk are only the user's baseline requirement and cannot substitute for the model's own budget. If any node fails to meet the bar, record the rejection reason and do not start the task.

## 4. Engine choice and offloading implementation

| Engine | Purpose | Required conditions |
|---|---|---|
| Transformers + PyTorch | First auditable extraction baseline, direct hooking | Pin a commit that supports this architecture; verify the AutoClass, the 4-branch semantics, the official operators, precision, and checkpoint. Must not carry over an old script's `AutoModelForCausalLM` assumption. |
| vLLM | Subsequent high-throughput worker | Has an official model recipe; the scraped page notes 0.29.0+; pin the version and container digest. The extraction logic must enter the actual model worker, handling packed tokens, TP, and CUDA graphs. The generation API alone does not provide the required evidence. |
| SGLang | First priority: evaluate the existing PLE-offload engineering implementation | The cookbook has PLE offload and a branch-specific file backend; pin the version actually verified to support it. Still needs an in-worker intermediate-representation adapter; tensors cannot be guessed from the API. |
| llama.cpp / GGUF | A verified CPU/GPU hybrid deployment, or a final-embedding control | The specific supported commit/converter has not been verified this time; the existing Qwen2.5 script does not prove support for the new architecture. A specified layer requires a C++ graph-extraction implementation; PyTorch hooks are not supported. |
| Ollama | A candidate wrapper for user interaction | Not the preferred choice for this project's specified-layer physical extraction; the existing API output cannot substitute for an intermediate layer. The specific backend and hardware support have not been measured here. |

### 4.1 The N-gram table

Priority order: full-GPU baseline -> official/verified host row-gather -> an explicit file-backed experimental path. This order is not permission for an automatic fallback.

1. Obtain the real n-gram ID algorithm, boundary-token handling, and index range from the pinned model implementation; preserve bigram/trigram history. Guessing IDs from text patterns is forbidden, and changing the hash, collision, or padding semantics is forbidden.
2. When all prompt tokens are known, lookup requests can be generated ahead of time; gather by deduplicated real IDs, then restore the original order via an inverse index. Prefetch depends only on known tokens; if the real implementation's index also depends on a hidden state, computation must not be fabricated ahead of time.
3. Host keeps the real table or an approved quantized storage; after CPU gather, transfer only the hit rows. Do not pin the entire 102GB table; use a bounded pinned double buffer, dequantize per row for quantized storage, and record the scale/zero-point and error.
4. The copy stream records a CUDA event; layer 2 must explicitly wait before use, and must not read data that is not yet complete. A cache miss should read the real backing table normally, not degrade the algorithm; a backend being unavailable, an I/O error, or a checksum failure must raise, and must never be replaced with a zero vector or an ordinary embedding.
5. Under TP, choose shared read-only host storage or sharding according to the real table layout; silently copying the whole table into RAM per rank is forbidden. Shared pages must also be measured via process PSS and node MemAvailable, not simply via RSS.
6. An mmap/file backend must be configured independently and must record page faults, page-cache/RSS, I/O latency, and queue pressure; mmap must not be treated as having no memory cost. Performance conclusions from a unified-memory device must not be applied to an ordinary PCIe node.

Generic Accelerate CPU/disk offload moves module weights at execution time, and it cannot be assumed to naturally implement **row-wise** N-gram gather. The generic hook that would move the entire embedding module to GPU must be checked and disabled, and a verified dedicated adapter must be implemented for the table. [Accelerate documentation](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference)

### 4.2 MoE experts

In the first phase, keep all experts resident across multiple GPUs, via a per-layer device map or a verified TP/EP distribution; dispatch is decided by the real router's expert selection. Expert paging/CPU computation in an independently designed experiment should only be chosen when capacity is insufficient.

Expert selection depends on the current layer's hidden state, so, unlike the N-gram table with known tokens, all future accesses cannot be accurately prefetched. If paging is implemented: full backing weights + a cache keyed by layer/expert + CUDA events + a bounded request queue; a missing expert must wait or fail -- top-k truncation, substituting a popular expert, or zero output are all forbidden. Measure per-layer hit rate, bytes moved, and wait time; measure cold start and warm cache separately. Prefill typically expands the union of experts touched, so the decode-time "6B active" figure must not be used to estimate extraction throughput.

## 5. The four-layer end-to-end pipeline

### Layer 1: data, prompt scheduling, and batching

Reuse the 13 tasks from `grand_challenge_data.py:49`: massive_en, massive_de, multinli, pubmedqa, vitaminc, boolq, squad2, paws, civil_comments, aegis_safety, helpsteer2, summeval_relevance, summeval_consistency.

Reuse the leakage gate from `gd.load_test` / `gd.build_train` and the alignment contract from `gpu_extract_qwen72b_13tasks.py:184`: verify train/test IDs, order, labels, and candidate order item by item. The raw prompt is `context + '\n\n' + instruction`; if a chat template is used for the new model, a distinct representation ID must be established, and it must not be disguised as using the same protocol as the raw prompt.

The manifest pins the dataset file SHA256, ID order, seed, split, candidate list, tokenizer revision, special tokens, template, per-task max/head token budget, and truncation strategy. The same token budget does not guarantee the same retained text under a different tokenizer; record both the raw/kept token length and the actual token-ID hash, and disclose this difference wherever results are compared.

Use an on-disk manifest to pin `row_index`; length-bucketing only changes execution order, and output is restored by `row_index`. The queue caps by **both bytes and token count**, not by entry count alone. Start with batch=1 and scale up incrementally; any allowed batch shrink is only performed within an explicitly configured retry protocol, with the attempt recorded; automatically truncating input, switching models, or skipping samples is forbidden.

PCA/basis fitting uses the train split only. Test data and candidates are only transformed, and never participate in the mean/covariance, layer selection, or quantization-parameter selection. Multi-task statistics must state whether they are sample-weighted or use fixed task weights; the default is sample-weighted, and class balancing must never be done silently. Labels never enter the model prompt or the feature-extraction decision.

The existing data builder may materialize an entire task in memory. For large tasks, build once, validate, and persist to a manifest first, then have workers stream-read it. The original builder itself must not be claimed to already implement O(1) streaming memory.

### Layer 2: target hooks and tensor lifetime

`Qwen38Adapter`'s responsibilities: match architecture/revision; locate the text backbone, the final norm, and the target layer's gated read or full residual; declare the tensor layout and dimensions; disable MTP, vision input, KV cache (to the extent the implementation allows), and full hidden-state output; avoid unnecessary `[B,T,vocab]` logits computation. Modules necessary for the text forward pass must not be removed just to save memory.

Recommendation: have the first version output the final post-norm 2560-D representation; for intermediate layers, explicitly specify the index and readout position, and do not hardcode an L24 path before the layout is confirmed. If the four branches are kept, explicitly declare `branches=4`, the dimension/axis order, and, if flattened, treat the resulting 10240-D as a **new protocol**; it must not be treated as 2560-D, nor averaged and then passed off as the original representation.

Start with PyTorch eager + `eval()` + `inference_mode()`, with graph capture/compile disabled; use a real short input to cross-check the hook output against the model's official corresponding output sample by sample before enabling optimizations. A forward hook can only avoid retaining unnecessary full-layer tensors; it cannot eliminate the current layer's own forward-pass activation peak.

- On GPU, select the target token first, then detach/copy into a bounded CPU buffer; never `.cpu()` the entire `[B,T,d]` first.
- Take the last valid token position as `max(where(attention_mask != 0, positions, -1))`, which is correct for both left and right padding; fail on an empty sequence. `sum(mask)-1` only works for right padding, and `-1` only holds under a verified layout.
- Capture the batch nonce, sample IDs, layer name, and hit count; every target should be hit exactly the expected number of times. A missed hook, a duplicate hook, dimension drift, or a non-finite value must all terminate the run.
- The callback returns `None` and does not modify the forward output; it does not retain the original output, the computation graph, or a cross-batch closure reference. The handle is removed in a `finally` block, and each batch's resources are released as soon as it is processed.
- For an asynchronous D2H copy, wait for the event before reading the CPU tensor, and only reuse the buffer once it is free; a synchronous copy in the first version is easier to audit.
- If the TP output is sharded along the hidden dimension, gather on the correct dimension; if it is replicated, write from only the single owner; under PP, the layer's owner sends the feature with an ID attached. The rank count must not be mistaken for the sample count.
- For vLLM/SGLang's packed/chunked prefill, the last token must be mapped by request ID and position, with state kept consistent across chunks; a prefix cache may skip the hook and should be explicitly disabled during the extraction-baseline phase, or the cache must also store a verifiable target representation alongside it.

### Layer 3: a numerically stable StreamingCovarianceAccumulator

Keep the existing public class/API, replacing the second-moment algorithm in place, to avoid ending up with a new, uncalled accumulator. Persistent state uses FP64 `(n, mean, M2)`; for a batch's `(m, mean_b, M2_b)`:

```text
delta = mean_b - mean
n_new = n + m
mean_new = mean + delta * m/n_new
M2_new = M2 + M2_b + outer(delta, delta) * n*m/n_new
covariance = M2/n             # population, ddof=0
```

Raise on an empty batch, a dimension mismatch, a non-finite input, or accumulation overflow; form and check a candidate state first, commit only on success, and never leave a half-updated state on failure. The first batch is initialized independently, with no covariance at n=0/1; computing rank-k requires `k <= min(d,n-1)` and checking the numerical rank, not merely `n>=k`. Converting NaN to zero, or falling back to an identity basis on failure, is forbidden.

State is about `8d^2+8d` bytes: roughly 50MiB at d=2560, roughly 800MiB at d=10240; accumulating simultaneously for every target layer and every task multiplies this. By default, process tasks sequentially, keeping only the necessary number of layers resident; global statistics can be merged across tasks in a fixed order.

The overall peak must budget for multiple d^2 temporary matrices, an FP64 batch of `B*d`, eigh workspace, and read/write buffers; `memory_bytes()` cannot serve as proof of the process peak. Measure RSS/PSS and GPU peak allocated/reserved empirically to see whether they plateau as N grows. With batch size, layer count, and queue fixed, this is O(d^2+Bd), not an absolute constant for arbitrary model/sequence length.

Symmetrize before solving and record the correction magnitude; check the scale of PSD violation, and fail on significantly negative eigenvalues. Only rounding-scale negative values within a threshold may be explicitly clipped to zero, with the count/magnitude recorded. A full eigh is O(d^3), not an online SVD; if an iterative top-k method is chosen, fix the tolerance and check the residual and convergence, failing if it does not converge. Compare subspace projection matrices; do not require basis vectors of a degenerate eigenspace to match elementwise.

The basis can only be finalized after training completes: on the first pass, persist fixed-size raw-feature shards to disk while updating the statistics; on the second pass, stream-read and project into Z. If raw features are not kept, a re-forward pass is needed, and its cost must be explicitly recorded. Retaining all of X while waiting for the basis is forbidden. Cross-model low-dimensional geometric similarity is only an observation; a breakthrough claim requires per-sample paired statistics on fixed test IDs.

### Layer 4: artifacts and the validation chain

Recommended directory layout: `run/<run_id>/{manifest.json,events.jsonl,features/,statistics/,basis/,checkpoints/}`. Large data uses chunked NPZ; NPZ is not a reliable direct-mmap container, and `np.load(...,mmap_mode=...)` must not be used to claim zero-copy access to its zip members.

| File | Required fields |
|---|---|
| raw feature shard | `X` FP32 `[n,d]`, `row_index` int64, `sample_ids` with no object dtype, `token_count`, `info_json`; each shard has a fixed maximum byte size |
| statistics checkpoint | `n` int64, `mean` FP64 `[d]`, `M2` FP64 `[d,d]`, the prefix of committed shards and the input cursor, a configuration digest |
| basis.npz | `U_k`, `mean`, `eigenvalues`, `n_samples`, `info_json`; keep an FP64 audit version, with a separate FP32 deployment artifact and hash |
| benchmark-compatible export | `train_full,test_full,cands,train_label,train_ids,test_ids,info_json`, consistent with the current consumer contract; the representation ID explicitly states dimension and pooling |

Large-scale consumers should iterate shards directly. If a single-file per-task NPZ must be generated, use bounded write-out / on-disk intermediate arrays, and confirm the reader does not load every task simultaneously; refuse the compatible export once it exceeds the pre-set budget rather than breaking the memory constraint.

Metadata must include at least: schema/version, run_id, the code HEAD and a summary of the working-tree diff, model/tokenizer revision, sha256 of every weight shard, engine/kernel/container versions, dtype/quantization configuration, device topology and offload strategy, the hook's full module path/semantics, task and data summary, split, token strategy, the range of the training statistics, shape/dtype, sample count, resource peaks, and a summary of failure/retry events. Any check that was not completed is written as `not_verified`, and must not be filled in as `true`.

Write to a temp file in the same directory -> flush/fsync -> reopen with `allow_pickle=False` to verify shape/dtype/finiteness/IDs -> stream SHA256 over the final file's bytes -> atomic rename. Write the manifest last, then publish a COMMITTED marker; readers must treat the committed manifest as the root of trust. File hashes live in an external manifest to avoid self-reference; the manifest's own hash lives in a separate commit marker, and external reports pin that hash. An array-level semantic hash must include the field name, shape, dtype, canonical byte order, and the data, not just concatenated raw bytes. SHA256 proves integrity, not authenticity or model correctness.

A checkpoint's statistics and its shard cursor must belong to the same commit generation; recovery verifies all input/model/configuration hashes and the shard prefix. Orphaned, uncommitted files are isolated or explicitly cleaned up, never skipped just because "the file exists." Disk full, a hash mismatch, a duplicate ID, or a missing row must all produce a non-zero exit, and a successful manifest must never be published in those cases.

## 6. Python script framework and production wiring

The following is **interface-design pseudocode, not an executable implementation**; it must not be turned into a script with TODOs/NotImplemented that is then called complete.

```python
def run(config):
    # Every dependency is constructed from an explicit config; no default fallback model/engine.
    manifest = validate_and_freeze_inputs(config)
    plan = preflight_resources_and_checkpoint(manifest)
    adapter = load_verified_qwen38_adapter(plan)
    writer = TransactionalShardWriter(manifest)
    stats = StreamingCovarianceAccumulator(adapter.feature_dim)

    try:
        adapter.verify_real_probe_and_hook_contract()
        with TargetTokenCapture(adapter, bounded_buffers=config.buffers) as capture:
            for batch in scheduler.iter_batches(manifest):
                capture.begin(batch.ids, batch.mask, batch.nonce)
                with torch.inference_mode():
                    adapter.forward_text_backbone(batch, use_cache=False,
                                                  output_hidden_states=False)
                x = capture.take_exactly_once()  # event, dimension, count, finiteness checks
                writer.stage_features(batch, x)
                if batch.split == "train":
                    stats.update(x)
                writer.commit_batch_with_statistics(stats, batch.cursor)
        basis = checked_eigensolve(stats, config.k)
        writer.project_shards_and_validate(basis)
        return writer.publish_committed_manifest()
    except BaseException as exc:
        writer.record_failure_without_publishing_success(exc)
        raise
    finally:
        adapter.close()
```

The actual implementation must ensure that even a construction-time failure is recorded by the top-level CLI; a logging failure must be written to stderr while preserving the original exception, and a cleanup-time exception must not overwrite the primary failure. If the process is SIGKILLed, the parent process must record the signal/exit status, and recovery must rely on the transaction, not on the `finally` block necessarily running. Disk/statistics transactions should have an independent consistency implementation; the method names above are not existing functions.

Recommended ownership breakdown:

- Refactor `benchmarks/suites/run_universal_extraction_a100.py` into a thin entry point that calls a shared orchestrator, removing the hardcoded path/dimensions, retention of all layers, field guessing, and fake counterfactual/phase-transition/trajectory logic; do not keep an old-and-new execution branch for automatic fallback.
- Update `python/gen_zero/causal/universal_manifold_extractor.py`'s accumulator and strict-artifact API in place; audit every existing call site including `GepaEvolutionDaemon`, and use an explicit version rejection or an offline conversion when migrating old state/checkpoints.
- Split the shared `qwen38_extraction` module into `manifest`, `adapter`, `capture`, `scheduler`, `artifact`, `runner`; count it implemented only once it is actually called by the runner and a real entry point.
- Add an explicit extraction subcommand (proposed name `extract-manifold`) to `crates/gen-zero-cli/src/main.rs`'s Commands, launching a pinned Python worker via argv, preserving the raw exit code/signal, forwarding interrupts, restricting the artifact directory, and verifying the completed manifest. Do not assemble configuration via shell string concatenation, and do not duplicate the numerical algorithm inside Rust.
- If online consumption is in scope, implement a separate mount adapter: once schema/hash/dim/representation validation passes, real requests enter the same encoder and projection path. Merely registering a file or printing "loaded" does not count as effective. The `serve --mount-assets` command currently accepts a cognitive-assets JSON; an NPZ must not simply be stuffed in and passed off as compatible.

The minimum real call chain must be `Rust CLI -> Python runner -> adapter.forward -> hook -> accumulator -> committed artifact`. Online capability separately requires `HTTP/MCP request -> validated mounted projection -> observable result`. The two are accepted independently; success on the former must not be passed off as success on the latter.

Old-logic removal acceptance should list the replaced implementations/symbols, use `rg` to check for 0 residue in executable source, and update the old tests and documentation; the historical evidence in this design document is not an executable fallback. This round is a planning task: no old module was removed, and no wiring is claimed to be complete.

## 7. Acceptance order, failure strategy, and the remote closed loop

1. Pin the revision and hash of the official model/config/tokenizer/weight-index, and check the architecture, tensor bytes, and module graph; if the source or the supported combination is unclear, the status is BLOCKED, and the full giant weights are not downloaded on a gamble.
2. Pass the health check plus per-GPU/host/disk peak budget; a remote sandbox with no `.git` syncs only source code and configuration, keeping a file manifest/hash, with actual edits only ever made locally. Automatic offload to another unverified environment is forbidden.
3. A minimal real-model prefill: cross-check the hook against the official readout; for the same real sample, compare single/batch, left/right padding, long/short sequences, and different shard owners with per-sample error and cosine similarity; produce a separate matched-pair report for quantization and offload. The threshold is written into the manifest beforehand; it must not be tuned after seeing the results and then declared passing.
4. CPU numerical verification: batched/merged vs. a centered reference, extreme bias / low variance, overflow rejection, the n/k constraint, PSD/eigen-residual. A test using an artificial numerical sample is a numerical unit test, not evidence of model capability.
5. Bounded-memory verification: with d, batch/token cap, and layer count fixed, run the same chain for different N, recording RSS/PSS, GPU peak, queue bytes, and disk growth. Test at least the shortest input and the task's longest input; a lack of OOM on a few short prompts is not proof of no OOM in general.
6. Fault injection: a real hook missing its trigger, a bad data row, a missing weight shard, disk exhaustion, a tampered digest, a killed worker, and recovery replay. Expect a non-zero exit, with no success marker, no skipped sample, and no fallback; after recovery, the ID set is covered exactly once.
7. The full 13 tasks: a train/test leakage check, raw/candidate ID alignment, and artifact schema validation, reporting extraction completeness first. Downstream performance requires per-sample output sharing the baseline's test IDs, a paired bootstrap or an appropriate paired test, and a seed/confidence interval; without these, it is not called a breakthrough.
8. Real acceptance of the Rust CLI includes: a successful artifact from a valid configuration, a non-zero exit from an invalid configuration, and logs that can be traced back to the hook's sample count; if a mount is added, a real online request and a rejection test case with a corrupted artifact must both be run. Record online acceptance as a separately tracked status.

Every run saves: argv, environment versions, the raw exit code/signal, the full stdout/stderr log, a tail summary, input/output hashes, and resource curves. A parent task's failure must not be masked by the success code of `tee/head/tail`. Heavy compilation uses `CARGO_BUILD_JOBS=$(nproc)` according to node health; extraction/BLAS must not unconditionally saturate every thread and crowd out loading and I/O either. Across nodes, prefer allocating independent tasks/splits, provided this does not duplicate loading beyond capacity; for the same model, TP prefers a single machine with fast interconnect -- "go all out on every node" must not turn into three redundant, under-capacity failing tasks.

The remote side keeps only reusable, authorized caches; temporary debug files are cleaned up by run_id after acceptance. Raw verification logs are archived first and their hashes checked on pull-back; evidence must never be deleted just to achieve "zero residue." Source code is never overwritten locally from the remote; generated files are pulled back according to manifest provenance. This planning task does not commit or push; any commit/push from a later implementation needs explicit task authorization.

## 8. Classification of this delivery and the review entry point

**Implemented (this delivery)**: this design document, a read-only code audit, cross-checking the official materials, numerical diagnostics of the existing accumulator, and local evidence. Verification commands and raw exit codes are in `evidence/qwen38-extraction-design/commands.json`; the numerical-diagnostic output is in `numerics.log`. Here, "implemented" refers only to the deliverable itself, not to the extraction system having been implemented.

**Unverified**: dev's live resources and GPU topology; the checkpoint's actual sharding/disk footprint; the exact HF module paths and hook tensor layout; the correctness, throughput, and peaks of each engine/quantization/offload combination on the target machine; any model capability improvement.

**Not done**: refactoring the executable extraction script, fixing the accumulator to be numerically stable, new Rust/HTTP/MCP wiring, removing old logic, remote compilation and the real-model 13-task extraction, training, and production go-live acceptance. Reason: this task called for a systems-engineering plan design; no model was loaded and no implementation was deployed this round. The proposed call chain in this document must not be used to claim production already references it.

Review against the user rules, item by item: 1, the critical problems are listed; 2, an explicit failure strategy is given and this round has no runtime fallback; 3, no capability claim is made; 4, the three status categories and their evidence are distinguished; 5, production wiring is set as an acceptance gate, not yet implemented; 6, no forbidden git command was run; 7, implementation has been broken into acceptable stages; 8, no heavy task/remote task was run this round, so the health check and cleanup were not executed; 9, no reviewer or subagent was dispatched.
