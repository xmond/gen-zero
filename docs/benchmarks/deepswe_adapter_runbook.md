# DeepSWE repair adapter

`benchmarks/gen_zero_deepswe_adapter.py` is a Pier host adapter with an external
Proposer API. It does **not** implement a Gen-Zero code-state encoder, learned
world model, or zero-token code synthesis. Provider response logs retain the
actual returned model identity; a routing alias is not a fixed model guarantee.

The host sends only the task instruction and sandbox repository observations to
the explicitly configured Proposer. Credentials stay in a host environment
variable. No Claude/Codex worker or reviewer is launched. The agent container
retains the official no-network configuration. The model cannot issue shell
commands, inspect the host task directory, or access the hidden verifier.

The adapter stages Pier's supplied `instruction.md` contents into the sandbox
and checks the readback. The Proposer extracts a problem statement and target
identifiers, then receives literal `git grep -n` locations. It can request
tracked source/test files with line numbers and content hashes. Full-file source
edits require the current hash; new files require a null old hash. Path escape,
symlinks, hidden paths, test modifications, unsupported file types, empty edits,
and invalid Python syntax are rejected explicitly. Rejections enter the log and
are fed back for correction within a finite step budget. There is no alternate
code generator or success override.

A configured real regression command runs before generation and after each edit.
The model sees actual regression results and can repair failures. A finalized
submission requires nonempty source edits and regression exit 0. It is committed
**only in the disposable task container**, because the official collection hook
compares the base commit to HEAD. New source files are staged so they are included.
The adapter writes `/logs/artifacts/model.patch`; Pier then collects that same
standard binary-capable Git diff for its separate verifier container.

## Running on dev

Set `BENCHMARK_ROOT` to your benchmark directory (for example,
`export BENCHMARK_ROOT="$HOME/benchmarks"`) and deploy these files there:

- `benchmarks/gen_zero_deepswe_adapter.py`
- `benchmarks/deepswe_audited_docker.py`
- `benchmarks/run_deepswe_smoke.sh`

Source an existing provider credential environment file without printing its
contents. Configure these nonsecret variables and run the smoke script:

```bash
export PROPOSER_URL='https://your-provider.example/v1/messages'
export PROPOSER_MODEL='your-model-id'
export PROPOSER_PROTOCOL=anthropic
export PROPOSER_KEY_ENV=ANTHROPIC_AUTH_TOKEN
export EVIDENCE_DIR="$BENCHMARK_ROOT/a-new-unique-evidence-directory"
bash "$BENCHMARK_ROOT/run_deepswe_smoke.sh"
```

For an OpenAI-compatible chat endpoint, use protocol `openai` and the full
`/v1/chat/completions` URL. This transport is implemented but requires its own
live verification; the run evidence identifies which protocol was actually used.
No missing credentials, transport errors, truncated generation, context overflow,
or step-budget exhaustion are silently replaced by a fallback.

The stock Pier verifier discards its `exec()` result. The audited Docker subclass
records each real command's result and returns it unchanged. Both agent and
separate verifier use this subclass. Docker events prove distinct container IDs.
Observation failures are fatal; grading scripts and expected test sets are not
modified. Logs may expose upstream ancillary command failures; they are not erased.

Copy the evidence directory back, then run:

```bash
python3 benchmarks/verify_deepswe_evidence.py /path/to/evidence
```

The validator exits nonzero for missing evidence, empty/mismatched patches,
agent exceptions, failed/missing independent verification, missing distinct
containers, or incomplete f2p/p2p success. **Pier CLI exit 0 and verifier script
exit 0 are not success scores.** The reward and test logs remain authoritative.
The validator's result and exit code must be retained separately from raw Pier
and verifier results.

## Verification and limitations

```bash
python3 -m unittest discover -s benchmarks/tests -p test_deepswe_adapter.py -v
```

These tests use real helper subprocesses and temporary Git repositories. They do
not mock a model or pretend to solve a benchmark. End-to-end generation evidence
belongs in `docs/benchmarks/evidence/t3-deepswe/`.

Current deliberate scope limits: Linux Docker tasks; UTF-8 source files no larger
than 1 MB; listed source-language extensions; no test/configuration changes;
full-file edits rather than arbitrary shell execution; explicit regression
command supplied by the operator; target identifiers must appear in the task
instruction. Only Python receives compile validation before writing; other
languages depend on the configured build/test command. Regression success does
not establish new-feature correctness. This is not evidence of general DeepSWE
performance or any SOTA claim.

## Gen-Zero gate and planning dependency (adapter 0.3)

Deploy `benchmarks/deepswe_genzero_gate.py` beside the adapter and install the
repository's Python `gen_zero` package in the Pier host environment (or add its
`python` directory to `PYTHONPATH`). Set `GEN_ZERO_SERVICE_URL` for the smoke
script, or pass `gen_zero_service_url` to the adapter. It must be an explicit
loopback HTTP(S) endpoint. Redirects are rejected. There is **no bundled trained
code world model or compatible running service claimed by this change**. Without
that dependency the adapter rejects execution; old smoke invocations now fail.

The endpoint accepts JSON POST requests with `operation`, `instruction`,
`candidate`, observed repository context for assessments, and a SHA-256 `request_id` over the sorted JSON payload before adding
that ID. Responses must echo the ID. `assess` returns finite [0,1] `confidence`,
`risk`, `relevance`, and boolean `syntax_valid`. Commands also use this contract;
`syntax_valid` then describes command syntax. `transition` additionally receives
`state`, initially `{"remote": {"observations": {...}}, "value": 0.0}`; subsequent states contain the
previous returned remote state and validated value. It returns a JSON-object
`state`, finite [-1,1] `reward` and `value`, and [0,1] `risk` and `confidence`.
The server must implement meaningful code-state transitions, including repeated
candidate application. This bridge does not train or validate that server.

Every sandbox command is evaluated before dispatch. Edits pass the real
`DecisionPolicyGate`, protected-path checks, Python AST parsing, and remote
semantic assessment before depth-three, 16-simulation `ImaginationMCTSPlanner`
selection. Up to eight alternative edits may be proposed in a `candidates`
action. Non-Python edits are rejected until a syntax validator is implemented.
Any unsafe imagined continuation conservatively rejects the entire selection.
A service outage, invalid response, unknown score, NaN/Infinity, or required
confirmation cannot become permission. Python syntax and risk predictions are
not formal proofs; lexical rules are deliberately conservative and cannot prove
absence of arbitrary malicious behavior. Existing sandbox hash/symlink checks
and real regression execution remain necessary.

`gate-*.json` records actual policy verdicts, assessments and transitions, and
successful planning records include visits and the imagined trajectory.
`gen-zero-telemetry.json` is the current cumulative record; the identity file is
only a setup snapshot. `gen_zero_world_model_used` becomes true only after a
successful remote-transition MCTS call, with backend explicitly labeled
`external-cognitive-service`; it never denotes Gen-Zero's trained numeric model.
Blocked count counts failed verdicts (including commands and planning failures),
not unique patches. Completion and failure metadata include the same counters.

Validation command:

```bash
pytest benchmarks/tests/test_deepswe_adapter_genzero_integration.py
```

The integration tests use the real Gen-Zero policy and MCTS, a deterministic
loopback HTTP test server, and Pier dispatch spies. They verify the bridge and
failure boundaries, not semantic prediction quality or a live DeepSWE score.

The bridge enables the planner's `strict_evaluation` mode: missing, exceptional,
out-of-range, or non-finite value evaluations raise instead of taking the legacy
heuristic fallback. Each selection also caps transitions at 128 calls and checks
a 60-second deadline before and after each transition. This is a between-call
budget, not a hard process deadline; an in-flight HTTP read may run until its
15-second socket timeout (which itself is not a wall-clock stream deadline).
