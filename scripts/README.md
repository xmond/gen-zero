# Scripts

Run these utilities from the repository root with `python3 scripts/<name>.py`. Use `--help` where supported for options.

| Script | Purpose and example |
| --- | --- |
| `reachability_audit.py` | Inventory Python and Rust source reachability; `python3 scripts/reachability_audit.py`. Writes `benchmarks/results/reachability.json`. |
| `extract_trajectories.py` | Generate deadlock-torus transition data; `python3 scripts/extract_trajectories.py --help`. |
| `train_world_model_dynamics.py` | Train a dynamics model from transitions; `python3 scripts/train_world_model_dynamics.py --data benchmarks/artifacts/zero/trajectories_v1.npz`. |
| `evaluate_world_model_dataset.py` | Evaluate the trained model against recorded trajectories; `python3 scripts/evaluate_world_model_dataset.py --help`. |
| `analyze_flash_next_shard_layout.py` | Estimate model shard and layer sizes from remote safetensors headers; `python3 scripts/analyze_flash_next_shard_layout.py --help`. |
| `inspect_gguf_layer_bytes.py` | Inspect GGUF tensor and layer byte sizes; `python3 scripts/inspect_gguf_layer_bytes.py model.gguf`. |
| `prototype_layer_streaming.py` | Benchmark CUDA layer streaming strategies with synthetic weights; `python3 scripts/prototype_layer_streaming.py --help`. |
| `rebuild_open_training_pool_natural_text.py` | Rebuild the open training pool with answer text; `python3 scripts/rebuild_open_training_pool_natural_text.py --help`. |
| `slice_gguf_layers.py` | Write a GGUF slice containing the first K transformer blocks; `python3 scripts/slice_gguf_layers.py --model-path model.gguf --max-layers 16 --output-path sliced.gguf`. |
| `download_benchmark_features.py` | Locate and verify the 26 pinned Qwen2.5-72B + LLaMA-3.1-70B feature files (13 tasks x 2 models, ~2.2 GB) for the dual-70B manifold reproduction; there is no public mirror, it only locates and hash-verifies local files. `python3 scripts/download_benchmark_features.py --help`. |

Tests for the GGUF slicer are in `scripts/tests/test_slice_gguf_layers.py`.
Tests for the feature verifier are in `scripts/tests/test_download_benchmark_features.py`.
