# Official examples

Three independent ways to try Gen-Zero, from least to most setup. The 9B
demo and each Python quickstart print one JSON object; the CLI walkthrough
prints JSON for `what-if`, `simulate`, and the MCP `initialize` handshake,
and a plain-text token banner for `keygen`. Real output you can inspect
instead of trusting a README.

## 1. The 9B model demo

`run_9b_demo.py` needs only `numpy` (no `pip install -e ./python`, no torch):
it loads the runtime module by file path so it never imports `gen_zero`'s
package `__init__`.

```sh
python3 examples/run_9b_demo.py
```

It requires `artifacts/qwen35_9b/zero_rnn_set_adapter_qwen35_9b.npz` (the
adapter weights). If `artifacts/qwen35_9b/parity_val200.npz` is also present,
it scores 3 real teacher-validation records from that file; otherwise it
falls back to a seeded synthetic query. Output is one JSON object: model
config, the Lyapunov stability check (`sigma_max_A < 1`), candidate scores
and chosen index per record, and load/score latency in milliseconds.
The lightweight 6.4 MB adapter weights (`zero_rnn_set_adapter_qwen35_9b.npz`)
are tracked and shipped directly in this repository, so the demo runs
immediately out of the box. If the artifact is missing, the script exits 1
and prints a plain `error: artifact not found at ...` message to stderr; it
never dumps a stack trace.

## 2. The pure Rust CLI walkthrough

`01_rust_cli_native.sh` needs only a built `gen-zero` binary: no Python, no
running server, no checkpoint. It demonstrates four native capabilities in
one pass:

- `what-if`: compare candidate first actions on the latent world model.
- `simulate`: roll a fixed action plan forward on the latent world model.
- `keygen`: generate a cryptographically secure connection token.
- `serve --mode stdio`: an MCP `initialize` handshake over a pipe.

```sh
cargo build --release -p gen-zero-cli
bash examples/01_rust_cli_native.sh
```

Set `GEN_ZERO_BIN` to point at a different binary (e.g. a debug build). Every
score in the walkthrough comes from the untrained latent prior (no checkpoint
is mounted): it demonstrates plumbing, not a learned model's judgment. The
script fails loudly (nonzero exit, message on stderr) if the binary is
missing or any subcommand is refused.

## 3. The Python SDK quickstart

`01_quickstart_decision.py`, `02_world_model_simulation.py`, and
`03_mpc_cem_continuous.py` run directly, with no `pip install -e ./python`:
each inserts the repository's `python/` directory onto `sys.path` before
importing `gen_zero`.

```sh
python3 examples/01_quickstart_decision.py
python3 examples/02_world_model_simulation.py
python3 examples/03_mpc_cem_continuous.py
```

Each program prints one JSON object. The quickstart calls the public
`GenZero()` constructor, `decide()` for discrete actions, and
`decide_continuous()` for continuous actions. The simulation example reports
the model's predicted rollout, branches, and audit; the `provenance` and risk
verdicts describe model outputs, not measured real-world safety. The MPC
example uses the SDK's built-in illustrative latent update and reward.
`GenZero()` does not train a model; CEM optimization alone is not proof of a
learned world model or useful real-world control. No checkpoint is mounted
in this state, so `GenZero()` logs degraded-mode warnings to stderr; stdout
stays a single valid JSON object.

## 4. Qwen2.5-1.5B contrastive decoding (CAD) on CPU

`quickstart_qwen15b_cad.py` runs the public Qwen2.5-1.5B-Instruct Q4_K_M GGUF
through the pure-inference CAD engine (`delta = cond - alpha * prior`, yes / no /
maybe verbalizer logits, last-position logits only). No weights ship in this repo.

```sh
pip install -e "./python[gguf]"
python3 -m gen_zero.scripts.setup_qwen15b_models      # ~1.1 GB, size + sha256 verified
python3 examples/quickstart_qwen15b_cad.py
gen-zero cad --gguf models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --question "Is the bridge open?" --context "The bridge was closed after inspectors found cracks."
```

With no calibration head the output is UNCALIBRATED: `alpha`, `temperature` and
`maybe_bias` are untuned defaults and the six demo cases are an illustration,
not a benchmark. On those six cases the CAD argmax equals the context-only
argmax, so the demo shows no CAD gain. `--workers N [--numa-pin]` fans a JSONL
batch out over a pinned multi-process GGUF pool (Linux).

## Running the tests

```sh
python3 -m unittest discover -s examples -p 'test_*.py'
```
