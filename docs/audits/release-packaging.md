# Slot 10: packaging, docs, MSRV and path hygiene

Audit date: 2026-09-27. Baseline: `6dfd6099591f68a8bba8bbb8ae3b2a6c302f7e2c`.
Branch: `feat/audit10-10`. Changes are restricted to the requested release and
documentation scope. A read-only Luna review checked README claims against code;
the main agent integrated corrections and ran verification.

## Findings and changes

| Area | Result and evidence |
| --- | --- |
| MSRV | `Cargo.toml:23` declares 1.88; all eleven member manifests inherit it. `.github/workflows/ci.yml:100` installs 1.88.0 and checks all targets/features, then runs workspace tests. |
| Missing lock | The original MSRV command failed with exit 101 because Cargo.lock is not shipped. `.github/workflows/ci.yml:103` now resolves dependencies with MSRV fallback before locked verification. Resolution is per build, not reproducible across dates. |
| Docker | `Dockerfile:2` retains `rust:1.88-slim`; lines 5–6 resolve compatible dependencies and build with the generated lock. |
| Release | `.github/workflows/release.yml:60` installs 1.88.0 for every target. Cross is version-pinned; checkout credentials are not persisted; jobs have timeouts. The retired macOS 13 runner is replaced by macOS 15 Intel. |
| MCTS and committee | `README.md:42` documents fixed 2:1:1 weights; `README.md:47` describes depth-one Rust MCTS. Code evidence: `crates/gen-zero-planner/src/engine.rs:154` expands only root children; `crates/gen-zero-planner/src/router.rs:173` applies fixed votes. |
| Entropy | `README.md:33` distinguishes Shannon entropy from Python ordinal dispersion and committee pass-through. Code: `crates/gen-zero-core/src/types.rs:195`, `crates/gen-zero-planner/src/engine.rs:239`, `python/gen_zero/model/choice_head.py:169`. |
| Other claims | Corrected MPC-CEM neural-rollout, universal-gating and checkpoint-shipping claims; removed unsupported Python latency claims and labeled text simulation as heuristic. `README.md:57`, `README.md:67`, `python/README.md:100`. |
| Paths | Sanitized host roots in 43 documentation/evidence files. Runnable scripts use configurable roots. `docs/README.md:15` explicitly identifies redacted logs and explains why historical hashes do not authenticate sanitized bytes. |
| Links | Replaced two public CI URLs returning 404 with a local workflow link (`README.md:5`). Five links to absent planner-audit artifacts are marked as unavailable historical evidence (`docs/architecture/planner_architecture_optimality_audit.md:42`). |

Runner selection was checked against the [official GitHub-hosted runners reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

## Verification

Full local command output and original exit codes are retained in
`/tmp/audit10-10-evidence/` on the audit host. This directory is not a shipped
artifact and is not a cross-machine dependency.

- `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo +1.88.0 generate-lockfile`:
  exit 0; 201 packages resolved for Rust 1.88.
- `cargo +1.88.0 check --workspace --all-targets --all-features --locked`:
  exit 0; completed in 2m 28s (`msrv-check-fixed.log`).
- `cargo +1.88.0 test --workspace --all-features --locked`: exit 0;
  596 passed, 0 failed, 6 ignored across 44 suite summaries (`msrv-test.log`).
- Actionlint 1.7.12: both workflows pass, exit 0.
- Across 15 Markdown files, 47 local targets/heading anchors pass; all 14 external
  linked URLs return HTTP 200 (reachability only, not validation of source claims).
- Scoped path scan, 200 JSON/JSONL parses, all documentation shell syntax checks,
  and five Python README snippet syntax checks pass. Python examples were not
  executed.
- Archived data diff review: 34 non-Markdown/non-shell evidence files differ only
  by path redaction and trailing command whitespace. Recorded result values were
  preserved. `git diff --check` passes.

## Limits

The Docker daemon is unavailable, so no container build or runtime health test
was performed. GitHub-hosted jobs, cross-compilation, macOS/Windows binaries and
publishing were not exercised. No live release acceptance is claimed. Cargo.lock
remains untracked under the existing repository policy; future dependency
resolution can change. This audit does not establish benchmark performance or
model quality.
