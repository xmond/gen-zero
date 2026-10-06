# Quickstart

This guide covers the public Rust binary, its reflex plugin runtime, and the MCP service. Run commands from the repository root unless stated otherwise.

`gen-zero` is the production runtime engine and client SDK. Training reflex plugins, continuous offline learning, and compiling checkpoints into patches happen in `gen-zero-research` and the tuning API (`tuning.gen-zero.ai`); this repository consumes their plugin and patch archives.

## What you can run today

Gen-Zero provides a deterministic Rust runtime engine, an MCP service with 24 cognitive verbs, and the **Two-Stage Dual-Track QA Gate** (`qa-gate` CLI command & `qa_gate` MCP verb):

1. **Stage 1 Fast Pass**: Evaluates the best span margin `diff = null_score - best_span_score`. When confident (`diff < -1.5` or `diff > 0.5`), it immediately releases or denies the candidate without invoking neural teachers (<1 ms latency).
2. **Stage 2 Tri-Teacher Verifier**: When inputs fall inside the ambiguity band `[-1.5, 0.5]`, it evaluates the candidate across the joint latent manifold of three independent teachers (LLaMA-3.1-405B, Qwen2.5-72B, LLaMA-3.1-70B) via a distilled LoRA adapter on Qwen2.5-0.5B. If the counterfactual alignment score `tri_sim` falls below the threshold (default 0.91), the decision flips and refuses to hallucinate.

Gen-Zero operates in pure-inference mode on open-source community weights (such as `Qwen2.5-1.5B-Instruct`), requiring zero internal training pipelines. You can pull official open-source weights using our automated setup script:

```bash
pip install -e ./python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m gen_zero.scripts.setup_qwen15b_models
```

## 1. Build the native binary

Install Rust 1.88 or newer, then:

```bash
git clone https://github.com/xmond/gen-zero.git
cd gen-zero
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo build --release
./target/release/gen-zero --help
```

Cargo builds the workspace and places the CLI at `target/release/gen-zero`.

## 2. Fast Decision & Causal Validation

### Example A: Fast Margin Evaluation (<1ms)

When confidence margin is clear, the gate releases candidates with sub-millisecond latency:

```bash
./target/release/gen-zero qa-gate \
  --context "The Eiffel Tower is in Paris, France." \
  --question "Where is the Eiffel Tower?" \
  --candidate "Paris" \
  --best-span-score 3.0 \
  --null-score 0.0
```

Output:

```json
{
  "decision_flipped": false,
  "fast_pass": true,
  "final_answer": "Paris",
  "is_answerable": true,
  "score_diff": -3.0,
  "stage2_triggered": false
}
```

### Example B: Contrastive Decoding (CAD) Pure Inference via Python CLI

Evaluate ambiguous or contentious queries with counterfactual prior conditioning:

```bash
python -m gen_zero.cli cad \
  --gguf ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --question "Does medication A cause drowsiness?" \
  --context "Medication A shows equal incidence of drowsiness compared to control groups." \
  --alpha 0.5
```

Output:

```json
{
  "best_candidate": "No",
  "calibrated": false,
  "clipped": false,
  "scores": {
    "No": 1.482,
    "Yes": -1.215,
    "Maybe": -0.892
  }
}
```

### Example C: Fail-Closed Decision Without Model Weights

When semantic backends are explicitly disabled, the policy gate safely fails closed with an explicit refusal:

```bash
GENZERO_PYTHON_ENDPOINT=off ./target/release/gen-zero decide \
  --context "Should I run an unreviewed production database migration?" \
  --candidates "run_now,request_review"
```

## 3. Start MCP locally

For a client that launches Gen-Zero as a child process:

```bash
./target/release/gen-zero serve --mode stdio
```

For a local network endpoint, generate a secret and start SSE:

```bash
./target/release/gen-zero keygen
GENZERO_API_KEY=your-generated-token ./target/release/gen-zero serve --mode sse --host 127.0.0.1 --port 8999
```

The SSE URL is `http://127.0.0.1:8999/sse`. Keep the token private; use a secret store for deployment.

## 4. Connect an MCP client

For Claude Code, add the local stdio server with `claude mcp add gen-zero -- /absolute/path/to/gen-zero/target/release/gen-zero serve --mode stdio` (check your installed CLI syntax). For Cursor, put the following entry in its `mcp.json`. Replace the path with the absolute path to your built binary:

```json
{
  "mcpServers": {
    "gen-zero": {
      "command": "/absolute/path/to/gen-zero/target/release/gen-zero",
      "args": ["serve", "--mode", "stdio"]
    }
  }
}
```

For SSE clients that support request headers:

```json
{
  "mcpServers": {
    "gen-zero": {
      "url": "http://127.0.0.1:8999/sse",
      "headers": { "Authorization": "Bearer your-generated-token" }
    }
  }
}
```

For Antigravity CLI, use its MCP server configuration with the same stdio command or SSE URL and bearer header. Client syntax and supported transports vary by version; verify them with the installed client's help.

## 5. Reflex runtime: bench, patch, feedback, adapt

Build release binaries and write two seeded, untrained plugins of the same shape (input dim 1024, rank 16, 8 steps, 4 candidates). They stand in for two trained checkpoints; latency depends on shape, not on weight values, but their decisions mean nothing.

```bash
cargo build --release -p gen-zero-cli --bin gen-zero --example reflex_bench_fixture
./target/release/examples/reflex_bench_fixture --out /tmp/base.gzr --seed 1
./target/release/examples/reflex_bench_fixture --out /tmp/target.gzr --seed 2
```

Create, inspect and apply a differential patch. `reflex-patch-create` verifies that the patch reproduces the target's SHA-256 before it writes anything; `exact_fixups` counts the elements where the f32 delta alone would not have been bit-exact.

```bash
./target/release/gen-zero reflex-patch-create --base /tmp/base.gzr --target /tmp/target.gzr --out /tmp/p.patch
./target/release/gen-zero reflex-patch-inspect --patch /tmp/p.patch
./target/release/gen-zero reflex-patch-apply --base /tmp/base.gzr --patch /tmp/p.patch --out /tmp/applied.gzr
cmp /tmp/applied.gzr /tmp/target.gzr && echo identical
```

Measure predict latency, then predict latency while the patch is hot-swapped in under 4 reader threads:

```bash
./target/release/gen-zero reflex-bench --plugin /tmp/base.gzr --threads 1 --iterations 20000
./target/release/gen-zero reflex-bench --plugin /tmp/base.gzr --threads 4 --iterations 20000 --patch /tmp/p.patch
```

One real run on a shared 24-core x86_64 host (loadavg about 5 to 8), release build:

```json
{"threads": 1, "total_predictions": 20000, "predict_latency_us": {"p50": 164, "p90": 194, "p99": 238, "max": 3112}, "hot_swap_latency_us": null}
{"threads": 4, "total_predictions": 80000, "predict_latency_us": {"p50": 172, "p90": 197, "p99": 225, "max": 4249}, "hot_swap_latency_us": 3815}
```

Numbers vary with hardware, load and plugin shape. Use `--release`: a debug `cargo run -p gen-zero-cli -- reflex-bench --plugin /tmp/base.gzr` measured p50 6.7 ms. `reflex-bench` without `--plugin` exits 2 with a usage error; no plugin ships in this repository.

Feedback and adaptation run against a SQLite store:

```bash
./target/release/gen-zero reflex-feedback-status --db /tmp/fb.sqlite3 --input-dim 1024
./target/release/gen-zero reflex-feedback-record --db /tmp/fb.sqlite3 --input-dim 1024 --trace-id <trace-id> --label action_0
./target/release/gen-zero reflex-feedback-prune --db /tmp/fb.sqlite3 --input-dim 1024 --before-unix-ms 0
./target/release/gen-zero reflex-adapt --plugin /tmp/base.gzr --out /tmp/adapted.gzr --db /tmp/fb.sqlite3 --task bench --head main
```

On a fresh store `reflex-adapt` prints `"adapted": false, "reason": "no unconsumed feedback"` and writes no checkpoint, and `reflex-feedback-record` for an unknown trace id exits 1. Traces are inserted by library callers of `SqliteFeedbackStore::insert_trace`; `serve` does not load reflex plugins or record traces yet.

## 6. Reproduce the trap-avoidance decision benchmark

No Rust build, checkpoint, or downloaded artifact required: needs `numpy` and `torch` installed in
your Python environment (this repo's `python/` package declares both; see [Quickstart:
Python](README.md#quickstart-python) to install it, or `pip install numpy torch` directly). It
generates its own seeded torus mazes on CPU and prints a pass/fail table for greedy navigation versus
MCTS planning with a privileged exact-graph model.

```bash
python3 benchmarks/suites/benchmark_world_model_mcts_ablation.py --episodes 10
```

Takes about 4s (a one-time ~2.5s import of this repo's Python package; the episode loop itself is
about 30ms for 10 episodes, about 290ms for the default `--episodes 100`). The console prints a
comparison table with success rate, trap rate, and decision latency for both policies, plus an exact
McNemar significance test, and writes the full paired-episode JSON/Markdown report to
`benchmarks/results/`. Read the caveat printed with the table before quoting the numbers: the "world
model" here is the environment's own exact transition function, not a trained neural network, and the
search loop is local to this benchmark script, not the Rust production planner. See the [README
benchmark section](README.md#instant-decision-benchmark-cpu-only-local-reproduction) for a full example
run and [benchmarks/README.md](benchmarks/README.md) for the rest of the CPU benchmark suite.

## 7. Run the 9B adapter smoke test

A standalone pre-extracted feature adapter for Qwen3.5-9B is included in `artifacts/qwen35_9b/zero_rnn_set_adapter_qwen35_9b.npz` (~6.4MB). This is a **local numeric smoke test, not an accuracy benchmark**: it verifies the adapter loads, scores candidates, and holds Lyapunov spectral stability (`sigma_max_A < 1`) on standard CPU without installing PyTorch or GPU drivers (requires only `numpy`). The repository does not ship the real teacher-validation parity set (`parity_val200.npz`); without it, the script falls back to fixed-seed (seed=0) synthetic vectors, so the `results`/`scores` values below are not a measurement of task accuracy:

```bash
python3 examples/run_9b_demo.py
```

Expected output (exact `latency_ms` values are machine-dependent; this is one real local run):
```json
{
  "artifact": "artifacts/qwen35_9b/zero_rnn_set_adapter_qwen35_9b.npz",
  "sample_source": "seeded synthetic (parity_val200.npz not found)",
  "stability": {
    "sigma_max_A": 0.949998
  },
  "results": [
    {"chosen": 4, "n_candidates": 5, "scores": [-97.021, -94.517, -98.409, -100.092, -94.393]}
  ],
  "latency_ms": {
    "load": 36.621,
    "mean_score_call": 4.056,
    "per_record": [5.349, 3.184, 3.634]
  }
}
```
`latency_ms.load` is the one-time cold-start weight-loading time (~37ms here). `latency_ms.mean_score_call` / `per_record` are the warm, post-warm-up per-sample scoring time (~3-5ms here) — a separate measurement, not interchangeable with load time. `sample_source` says whether the scores came from the real held-out parity set or the synthetic fallback; without `parity_val200.npz` locally (not shipped in this repository), `results`/`scores` are not a measurement of task accuracy. Two separate mechanisms give the stability and order properties. Lyapunov spectral stability of the recurrent think loop comes from a spectral-norm clamp on its state-transition matrix A: A is rescaled by `a_scale` so that its largest singular value `sigma_max_A` is below 1 (which also bounds its spectral radius below 1), and `RNNSetAdapterRuntime` re-derives this by SVD at load and refuses a checkpoint with `sigma_max_A >= 1` (`python/gen_zero/causal/rnn_set_adapter.py`). Permutation equivariance over candidate actions comes from the network structure: the Set-Attention block has no positional encoding, so reordering the candidates reorders the scores the same way.

## 8. Authorization and deployment

- Set `GENZERO_API_KEY` (or `serve --token`) for any SSE service reachable beyond a trusted local machine. Without a token, SSE starts in open mode.
- Send `Authorization: Bearer <token>` to `/sse` and other protected routes. Avoid query-string tokens because URLs can enter logs.
- Bind to `127.0.0.1` for local development. For remote access, set `--host` deliberately and terminate TLS at a trusted proxy. Preserve the Authorization header and SSE streaming behavior through Cloudflare or another WAF; test the deployed endpoint with the actual client.
- `/health` is liveness; `/ready` also checks the semantic bridge and mounted assets and can return 503 until those dependencies are present. See the [service documentation](crates/gen-zero-service/README.md).
