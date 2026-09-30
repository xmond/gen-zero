# Provenance of pre-built assets in this repository

This repo ships a small number of binary/pre-built assets alongside source
code. This file lists each one's origin, license, and a SHA-256 checksum so a
downstream consumer can verify what they received without re-deriving it.

## 1. Qwen3.5-9B feature adapter weights

**File:** `artifacts/qwen35_9b/zero_rnn_set_adapter_qwen35_9b.npz`

| Field | Value |
|---|---|
| SHA-256 | `3fca76bcdf72ec16824e6f24fff4bae96e066bfaf04a249cc4b741afdc7c12b7` |
| Size | ~6.4 MB |
| Contents | NumPy arrays only (weight matrices, biases, and int/float config scalars for a custom RNN + set-attention scorer). See the array list and shapes in `python/gen_zero/causal/rnn_set_adapter.py` (`_ARRAYS`, `_INTS`). |
| Architecture | Original, defined in this repository at `python/gen_zero/causal/rnn_set_adapter.py`. Not a copy or fine-tune of any third-party model architecture. |
| Training input | Hidden-state feature vectors (`in_dim=4096`) extracted from Qwen3.5-9B by running the teacher model locally; feature extraction is not part of this repository (see [`data/README.md`](../data/README.md)). |
| What's redistributed | Only the trained adapter's own numeric parameters (matrices sized `256`/`16`-dimensional per `python/gen_zero/causal/rnn_set_adapter.py`). **No Qwen3.5-9B model weights are included, copied, or redistributed in this file or anywhere in this repository.** |
| License | Apache-2.0, same as the rest of this repository (see [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE)). Because no third-party model weights are embedded, this artifact carries no separate upstream model license obligation; the code that produced and that reads it (`rnn_set_adapter.py`) is itself Apache-2.0 licensed. Qwen3.5-9B itself is a separate third-party model with its own license — obtaining and running that model to extract features, if you choose to do so, is governed by its publisher's license, not this repository's. |
| Verification | Loading enforces internal self-consistency at read time: `RNNSetAdapterRuntime.from_npz` re-derives the spectral clamp from the raw factors and rejects the file if the stored `a_scale` disagrees (rtol 1e-4) or if `sigma_max(A) >= 1`. This checks numerical integrity, not benchmark accuracy. |

## 2. Auxiliary Qwen3.5-9B evaluation artifacts (results, not shipped inputs)

These are pre-computed outputs from a local benchmark run, kept for
reproducibility of the numbers reported in project docs. They are derived
data (projections/statistics computed from Qwen3.5-9B feature vectors), not
redistributed model weights, and carry the same Apache-2.0 / self-authored
status as item 1.

| File | SHA-256 |
|---|---|
| `benchmarks/results/qwen35_9b_universal_manifold.npz` | `578e3444951eb14f817a8616f81835836e4a291cf1505751fb3ad412409a1c0d` |
| `benchmarks/results/ensemble_cpu_benchmark_qwen35_9b_solo.json` | `e68a20869da6cf9cfee2772ad1a6e348ea45eaa6bff0ff933257c58cfe471b8f` |

## 3. World model dynamics checkpoint and calibration manifold

**File:** `benchmarks/artifacts/zero/world_model_dynamics_v1.pt`

| Field | Value |
|---|---|
| SHA-256 | `0e748c33e4cbe79a5538cf56a32ba1fd00dcdfd8bfeafdcd1f0eec365359bd18` |
| Size | ~648 KB |
| Contents | A plain `torch.save` dict (`format`, `config`, `state_dict`, 36 tensors) holding the trained weights of `NeuralDynamicsWorldModel`, this repository's own transition-network architecture (`python/gen_zero/world_model/neural_dynamics.py`). Loadable with `torch.load(..., weights_only=True)`; no arbitrary pickled objects beyond that dict. |
| Architecture | Original, defined in this repository. Not a copy or fine-tune of any third-party model. |
| Training input | `(s, a, s', r)` transitions from `benchmarks/artifacts/zero/trajectories_v1.npz`, split by episode; see `scripts/train_world_model_dynamics.py`. |
| What's redistributed | Only this repository's own trained transition-network weights. **No third-party model weights are included, copied, or redistributed in this file.** |
| License | Apache-2.0, same as the rest of this repository (see [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE)). |

**File:** `benchmarks/artifacts/zero/zero_manifold_v1.npz`

| Field | Value |
|---|---|
| SHA-256 | `670539b71f384e17db05e1f1e94c1b8796569d66d83b2c783ecc04289ca52de3` |
| Size | ~230 KB |
| Contents | NumPy arrays only (`mean` (896,), `basis` (896, 64), `scale` (64,), plus a `metadata` string): a fitted ZCA whitening/projection from an 896-dim hidden-state space to a 64-dim manifold. Per the embedded metadata it was fit from `zero-qwen2.5-0.5b-trunk:int8:last-token` hidden states over `calibration_clean_16.jsonl` (1,944 samples). The extraction script lives in `gen-zero-research`. |
| What's redistributed | Only derived summary statistics of a linear projection (mean/basis/scale), not any Qwen2.5-0.5B model weights. **No third-party model weights are included, copied, or redistributed in this file.** |
| License | Apache-2.0, same as the rest of this repository. Because no third-party model weights are embedded, this artifact carries no separate upstream model license obligation; running Qwen2.5-0.5B yourself to reproduce the source features is governed by its own publisher's license, not this repository's. |

## 4. Benchmark manifests

**File:** `benchmarks/data/manifest.json` (SHA-256: `4e3dfe0c08d0b36ded411673fa8ff9bf59133f6dcbeea0f9b91ea645d200eb46`)

This manifest declares `"license": "Apache-2.0"` for the 45 `.jsonl` task
shards under `benchmarks/data/` and records a per-file SHA-256 for each shard
under `tasks.<name>.sha256` — see the manifest itself for the authoritative,
per-file list rather than duplicating 45 hashes here.

Each task's `hf_repo` field in the manifest names the upstream Hugging Face
dataset the shard was sampled from (for example
`SetFit/amazon_massive_scenario_en-US` for `massive_en`). The manifest's
top-level `"license": "Apache-2.0"` covers this repository's own packaging and
shard format; it does not relicense the underlying third-party dataset
content. Those upstream datasets carry their own licenses set by their
original publishers, separate from this repository's Apache-2.0 license, and
some require attribution on redistribution. For example, `aegis_safety.jsonl`
(100 samples, `tasks.aegis_safety` in the manifest) is sampled from
`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`, which NVIDIA publishes under
CC-BY-4.0: redistributing or building on that shard requires attribution to
NVIDIA under CC-BY-4.0, on top of and separate from this repository's own
Apache-2.0 terms. Consult each `hf_repo` before redistributing a shard outside
benchmarking use.

## How to re-verify

```bash
sha256sum artifacts/qwen35_9b/zero_rnn_set_adapter_qwen35_9b.npz
sha256sum benchmarks/results/qwen35_9b_universal_manifold.npz
sha256sum benchmarks/results/ensemble_cpu_benchmark_qwen35_9b_solo.json
sha256sum benchmarks/artifacts/zero/world_model_dynamics_v1.pt
sha256sum benchmarks/artifacts/zero/zero_manifold_v1.npz
sha256sum benchmarks/data/manifest.json
```

If a hash here disagrees with what you downloaded, treat the file as
untrusted and re-fetch it rather than reporting results computed from it.
