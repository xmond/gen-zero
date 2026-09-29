# Local data layout

`data/` holds downloaded or generated inputs used by local experiments. Keep datasets, extracted tensors, and model weights out of git. Small, reviewed fixtures and benchmark manifests belong under `benchmarks/data/` instead.

| Path | Expected contents | How to populate it |
| --- | --- | --- |
| `data/deepswe/` | DeepSWE CLM training embeddings: Parquet shards under `data/`, as supplied by `Contrastive-LM/deepswe-clm-train-embeddings-8k`. | Download the dataset into this directory while preserving its `data/*.parquet` layout. |
| `data/extracted_features/` | Model-specific feature arrays, usually `.npz` files grouped by model and task; local symlinks to external storage are also valid. | Run the relevant feature extraction workflow and point it at local storage. If your features live on a shared volume or NAS, symlink each model's entry (e.g. `llama70b`, `qwen72b`) into this directory rather than copying, and set the mount path via your own environment configuration. |

Both directories are ignored by git. Do not commit downloaded shards, feature tensors, checkpoints, or model binaries. Record reproducible metadata and evaluation summaries in the appropriate tracked documentation or benchmark result files.
