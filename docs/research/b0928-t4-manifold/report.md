# B11 / B12 / B13 / B18 Fixes and Evidence

Worktree: `/tmp/fleet-wt/b0928-t4-manifold`. No commit or push was made; no subagent or reviewer was called.

## Implemented and verified

| Item | Actual behavior and main call path | Verifiable location |
|---|---|---|
| B11 | Continuous systems require the largest real part of any eigenvalue to be strictly below negative epsilon; discrete systems require a spectral radius below 1. A fitted result is never altered to appear stable: failure raises an error. Stability is checked again after float32 export and on load. | `python/gen_zero/causal/universal_manifold_extractor.py:395`, `:512`, `:529` |
| B11 production script | Removed the path that treated independent question rows as a time series and claimed to prove contraction. `--trajectory` and `--dt` are now required. | `benchmarks/suites/run_universal_extraction_a100.py:71`, `:185` |
| B12 | The GCCA CLI defaults to rank 128 and rejects an explicit rank-128 request when the available rank is lower. It saves each view's MAXVAR ridge regression matrix W, training mean, and scale. The anchor accepts 128→128 and rejects 64→128. | `benchmarks/suites/generalized_cca_manifold_interference.py:155`, `:166`, `:315`, `:472` |
| B12 validation isolation | Saves source-data hashes and fit/validation indices, and checks that exported fit rows came from this fit. The anchor CLI reuses the GCCA split instead of creating another random split or fitting on validation data. GCCA files, indices, and manifests have integrity hashes. | `python/gen_zero/causal/manifold_anchor_distiller.py:84`, `:349` |
| B13 | The anchor manifest binds `source_model`, `layer`, `norm`, `GCCA_map`, `anchor_basis`, `core`, and `domain_id`. The bridge requires a separately supplied target core manifest; source samples must carry `values` and `space`. Missing identity, equal dimensions with different bases, and the wrong domain are rejected. | `python/gen_zero/causal/manifold_anchor_distiller.py:116`, `:126`; `python/gen_zero/causal/nanocore_bridge.py:44` |
| B13 main entry points | `gen_zero.cli anchor --core-manifest` and the client loading entry point pass through the contract; client decisions also check the registered core's `space_manifest`. The latency script follows the new contract, and the entry point that self-tested against a fabricated artifact was removed. | `python/gen_zero/cli.py:417`; `python/gen_zero/client.py:3190`, `:3268`; `benchmarks/suites/profile_nanocore_latency.py:73` |
| B18 | The launcher requires an explicit path and `MODEL_PROFILE`. `full` requires at least 100 GB, 126 layers, and an output head. `slice` requires explicit `SLICE_LAYERS/NGL`, a truncated layer count, and no output head. Structural checks apply to each profile; the full-model size threshold is not imposed on slices. | `benchmarks/suites/run_llama405b_extract.bat:12`; `benchmarks/suites/verify_gguf_model.py:45`; `scripts/slice_gguf_layers.py:193` |

Commands run from the worktree root:

```bash
bash /tmp/b0928-t4-evidence/run-tests.sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:benchmarks/suites python /tmp/b0928-t4-evidence/real_transform_check.py
```

> **Pruned 2026-09-29.** `run-tests.sh` was removed. It named two test files that no longer exist (`benchmarks/tests/test_run_llama405b_launcher.py`, `benchmarks/tests/test_llama405b_extraction.py`), so it cannot be re-run as shipped. The recorded result (`final-tests.exit`) is historical. `real_transform_check.py` is kept: its imports still resolve, but it reads features from a host-local path.

- The first command had an actual exit code of **0**; its output ended with `187 passed, 11 warnings in 24.52s`. The complete command was in `run-tests.sh`, the raw log is `final-tests.log`, and the original exit code is in `final-tests.exit`. The warnings arose from numerical overflow in extreme-value rejection tests. No test failure was ignored and no silent fallback was added.
- The second command had an actual exit code of **0**; see `real.log` and `real.exit`. Paired BoolQ features from LLaMA-70B and Qwen-72B passed raw ID and label consistency checks: 500 training rows and 200 validation rows. After saving and reloading, the maximum coordinate error per sample was **0.0**. Per-sample IDs, both actual outputs, errors, and fit/evaluation indices are in `/tmp/b0928-t4-evidence/real-paired-transform-evidence.npz`. `artifacts.json` records artifact sizes, paths, and SHA256 hashes.
- The real anchor is a 128→128 orthogonal transform, so energy retention and cosine correlation of 1 are **not evidence of improved task capability**. Classification accuracy was not computed, and no generalization gain is claimed.
- `git diff --check`, `python -m compileall -q ...` on the specified files, and `rustfmt --check` on the Rust test file all exited **0**. Full argv, stdout, and stderr are in `checks.json`. `compileall` establishes only Python compilation; `rustfmt` checks Rust syntax and formatting, not cargo compilation.
- An initial set of 10 failures and an expanded set of 4 failures during development are recorded in `/tmp/b0928-t4-evidence/initial-tests.log` and `tests.log`; both original exit codes were 1. The final passing result was obtained only after fixes.

Test changes: Previous tests assumed that scaling a divergent fit necessarily made it stable, relied on a default machine path, and allowed an unsourced v1 artifact through the bridge. Those expectations conflicted with the fixes. They were replaced with positive and negative Hurwitz/Schur cases, a numerical comparison against the original GCCA ridge regression, rejection of inconsistent splits and mismatched manifests, tests of the actual CLI entry point, and launcher profile structure checks. Checks for candidate count, domain, duplicate candidates, nonfinite input, and float32 overflow were parameterized. The real old v1 artifact remains, but tests require its explicit rejection; no GCCA/core provenance was fabricated for it. The old Rust replay test was changed to verify Python's rejection of that artifact, rather than treating a historical snapshot as evidence that the new path succeeded.

## Unverified

- Actual Windows `.bat` startup of a full 405B model or a small slice in llama-server, including VRAM use. Argument and structure gates were verified; the large model was not started.
- Dynamic system identification from a real ordered trajectory on A100. Only mathematical stable/divergent counterexamples and a production-code compilation check were run.
- The updated Rust tests were not compiled or executed with cargo. No large build or remote deployment was performed.
- This work covers artifacts, the Python bridge, CLI, and client. Direct raw-vector calls to Rust `nanocore_ask` still have the preexisting shape check (`crates/gen-zero-service/src/zero.rs:2261`); they do not automatically receive Python's identity check. This result does not establish mandatory space identity for all raw Rust/MCP calls.

## Incomplete

- Real core training matched to the new GCCA/anchor space, publication of a core manifest, and live Rust/MCP acceptance verification. Existing repository cores and fixtures lack this binding. The real validation artifact was deliberately left **unbound**; the measured bridge rejection said `artifact lacks complete source -> GCCA -> anchor -> core identity`. An arbitrary core hash was not attached to make the artifact look usable.
- No task accuracy improvement is claimed, and model assumptions were not substituted for training evidence.

## Review against requirements 1–10

1. Discloses the false trajectory made from independent samples, incorrect stability checking, missing real core, and the boundary of direct Rust calls.
2. Unstable fits, insufficient rank, old versions, mismatched identity, invalid splits, and invalid launch profiles raise errors; no new silent fallback was added.
3. Mathematical unit tests, fitting on real features, and capability evaluation are kept distinct. Per-sample transform data is available for review; no capability breakthrough is claimed.
4. The three evidence states above are tied to original exit codes, complete logs, and code locations.
5. The new transform is used by the anchor CLI, bridge, and client; numerical CLI entry-point tests passed. Superseded stability-scaling symbols and the old production call were removed. The raw Rust entry-point boundary is disclosed rather than claimed as covered.
6. No stash, checkout, reset, clean, or force-push was performed.
7. After the initial fixes, the production calls and split isolation were checked, affected tests were updated, and all focused checks were run.
8. The work involved single-threaded NumPy and Python tests, with no large build. No remote task or residual service was created.
9. Equal-dimensional spaces were not presented as compatible; mathematically constructed data was not presented as trained capability; and an old fixture passing through was not called acceptance of the new contract.
10. No old stability projection or unsourced bridge compatibility branch remains. V1 artifacts are explicitly rejected, and full models and slices have separate explicit profiles.
