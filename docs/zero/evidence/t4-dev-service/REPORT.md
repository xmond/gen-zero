# T4 dev Decision Service Deployment Acceptance (2026-09-26)

## Implemented

- Source workspace `/workspace/pj/gen-zero-worktree/d4-dev-service`, baseline HEAD `478699c079bf4a053b953bb83b7e7f310eae4922`. The first `git status --short` produced no output; this round's code changes are limited to the CLI host/mode arguments.
- `rsync -az --exclude target --exclude .git ./ dev:/home/user/gen-zero-dev-eval/` completed with exit code 0. No deleting sync was used; the target did not exist initially. A subsequent checksum dry-run showed no file-content differences, only directory-timestamp and evidence-directory differences; see `source-check.log`. The remote host has no `.git`; see `no-git.exit`.
- dev is `luy-dev`, `nproc=64`. The build was configured for 64 Cargo jobs (this does not prove all 64 cores were saturated at every moment). Cargo 1.98.1 / rustc 1.98.1.
- The release build exited with code 0; Cargo reported **54.48 seconds**, and wall-clock timing measured 54 seconds; the binary is **10,779,448 bytes**; see `build.log`, `build.exit`, `build.seconds`, `binary-size.txt`, `binary.sha256`.
- The original command's `gen_zero` does not actually exist; the Cargo package is named `gen-zero` (`crates/gen-zero-cli/Cargo.toml:14`). The original CLI had no `--host` flag and defaulted to stdio. This round added host-IP validation and a mode enum at `crates/gen-zero-cli/src/main.rs:48`, using the specified address at `:398`. An invalid mode/host now exits with code 2; `cargo fmt --all -- --check` exits with code 0.
- Both the Rust service and the Python scorer run under nohup, listening on loopback addresses 8080 / 8995. PIDs and process commands are in `process.txt`; evidence of the 8080 listener is in `listener.txt`. These are background daemon processes with no systemd restart or boot-persistence guarantee.
- To support real semantic-call behavior, dependencies were installed in the remote `.venv-t4`, with a `.pth` file explicitly reusing PyTorch 2.14.0+cpu from `/home/user/genz-eval-venv/lib/python3.14/site-packages`. The full version list is in `uv-python-freeze.txt`; both the initial failure and the fix are recorded.
- The local machine's existing Qwen2.5-0.5B snapshot `060db6499f32faf8b98477b0a26969ef7d8b9987` was copied over; the weight SHA-256 matches on both ends as `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`. The scorer performs a real fp32 forward pass, using 8 threads.
- The scorer's auth token exists only as `dev`'s `.scorer-token` (mode 0600); no secret was written into this evidence directory. Rust accesses the scorer via `GENZERO_PYTHON_API_KEY` with `GENZERO_BRIDGE_REQUIRED=1` enabled. Port 8080 remains open on loopback per the user's requirement.

## Real HTTP results

| Request | HTTP | Raw curl exit code | JSON/semantic assertion exit code |
|---|---:|---:|---:|
| health | 200 | 0 | 0 |
| ready | 503 | 22 | 1 |
| ask | 200 | 0 | 0 |
| route | 200 | 0 | 0 |
| what_if | 200 | 0 | 0 |
| invalid_what_if | 400 | 22 | 0 |

`connected.exit=1`, because of the readiness 503; the full test suite was not falsely reported as passing. Every curl call uses `--fail-with-body`, so the raw exit code for any 4xx/5xx is 22; a successful JSON assertion on a negative case does not change the raw curl exit code.

- `ask` selected `read README.md`; `route` selected `read_file`; both show `engine=semantic_bridge`, `semantic_scoring=true`, risk assessed=true. Candidate scores, model identifiers, and the full JSON are in `connected/*.body.json`.
- `what_if` completed a 1024-dimensional input, two candidates, and a 3-step rollout, but explicitly reports `trained=false`, `calibrated=false`, `advisory_only=true`. Only the interface and algorithm execution are accepted here; no claim of real-world prediction correctness is made.
- `invalid_what_if` supplied only a 1-dimensional state and returned 400 / isError=true.
- Before the scorer was first connected, ask/route returned 428 and ready returned 503; `baseline/`, `baseline.log`, and `service-baseline.log` are fully preserved. Success was not faked by hardcoding a fixed "proceed" response for an empty ask.

## Reproducible commands and raw logs

```bash
ssh worker-node 'cd /home/user/gen-zero-dev-eval; export PATH=/home/user/.cargo/bin:$PATH; CARGO_BUILD_JOBS=64 cargo build --release -p gen-zero-cli'
# Full environment, auth-token reading, and PID management for starting the service are in start-services.sh; the actual Rust command:
nohup target/release/gen-zero serve --mode sse --port 8080 --host 127.0.0.1 > service.log 2>&1 < /dev/null &
ssh worker-node 'cd /home/user/gen-zero-dev-eval; bash t4-evidence/http-check.sh connected'
```

`http-check.sh` contains all curl calls and JSON assertions; payloads are in `ask.json`, `route.json`, `what_if.json`, `invalid_what_if.json`. The fully expanded command, response headers, body, stderr, HTTP status, raw exit code, and assertion exit code for each request are saved separately under `connected/<name>.*`. `connected.log` preserves the full round's output. The subsequent readiness query command:

```bash
curl --silent --show-error --fail-with-body -D t4-evidence/ready-final.headers -o t4-evidence/ready-final.json -w '%{http_code}\n' http://127.0.0.1:8080/ready
```

Final raw output is in `ready-final.json`: backbone_loaded=true, but cognitive_assets unavailable.

## Unverified

- Whether the current workspace includes all the upstream-claimed "latest S1-S5" commits: this deployment round used the given workspace as-is, with no unauthorized merge/pull. The workspace contains `SheafOperator`, but its presence does not mean this request actually exercised it.
- End-to-end effectiveness of the unified Sheaf cognitive path. ask/route explicitly report `cognitive_runtime=not_engaged: request carries no numeric manifold coordinates`, mount has_cognitive_assets=false.
- Model accuracy, safety guarantees, performance benchmarks, long-term stability, and restart recovery. No full-repo test run was performed this round.

## Not done / actual blockers

- **Full readiness has not yet been achieved**: `/ready=503`, missing the real `gen-zero/cognitive-assets/v1` asset. The determining code is at `crates/gen-zero-service/src/server.rs:992` and `:1001`; the detailed response is in `ready-final.json:1`. Only test fixtures exist; they were not passed off as production assets. The real asset path has been requested from the user.
- Python initialization also reports that the dual-head has no checkpoint and vision has no real weights; these endpoints are not part of the semantic scorer successfully exercised this round, and no usability claim is made for them; the raw warnings are in `scorer.log`.
- No core module was replaced or added this round, so there is no superseded old module to remove; a full-repo stale-symbol cleanup acceptance was not performed. Nothing was committed, pushed, or run through stash/checkout/reset/clean.
