# Gen-Zero In-Tree Documentation

This index covers every document shipped in `docs/`. Most files are dated
engineering records: each one states its own baseline commit and its own
evidence status. A record describes the tree at that commit, not the current
tree. Where a record and the code disagree, the code wins.

Historical research catalogs mention further specifications and artifacts that
are not included here. Those entries are not runnable instructions or evidence
of current capabilities.

## How to read status labels

- **Implemented**: code exists in this tree and a test or recorded run covers it.
- **Design**: a plan or specification. The components it describes may not exist.
- **Historical evidence**: logs, exit codes and reports from a past run. They
  are kept for provenance and are not re-verified against the current tree.

## 1. Architecture & technical specifications

`docs/architecture/`

| Document | Kind |
| --- | --- |
| [Planner architecture optimality audit](architecture/planner_architecture_optimality_audit.md) | Audit, baseline `025360f` |
| [Capability boundary and 8B attribution survey](architecture/gen_zero_capability_audit_20260927.md) | Survey, baseline `6dfd609` |
| [Qwen3.8-Flash-Next downstream integration plan](architecture/qwen38_flash_next_downstream_integration_plan.md) | Design |
| [F01–F09 planner fault regression evidence](architecture/fault_regression_evidence/README.md) | Historical evidence |

`docs/zero/` (the Zero model line)

| Document | Kind |
| --- | --- |
| [Zero technical overview and status legend](zero/README.md) | Overview; start here |
| [Spec 29: PubMedQA / Aegis unified manifold evaluation closure](zero/29-pubmedqa-aegis-unified-manifold-evaluation-closure.md) | Evaluation corrections |
| [Spec 30: Qwen3.8-Flash-Next layered manifold extraction](zero/30-qwen38-flash-next-layered-manifold-extraction-design.md) | Design |
| [Qwen3.8-Flash-Next extraction system design](zero/qwen38-flash-next-extraction-system-design.md) | Design |
| [Spec 31: Dense fleet manifold anchor (T4)](zero/31-dense-fleet-manifold-anchor-t4-sys-design.md) | Design |
| [Spec 31: Multiscale dense resonance, ETF and dual process](zero/31-multiscale-dense-resonance-etf-dual-process-plan.md) | Design |

Evidence for these specs lives in `zero/evidence/`: `b0927c-t1-geom/`,
`b0927c-t3-multi/`, `qwen38-extraction-design/` and `t4-dev-service/`
(see its [`REPORT.md`](zero/evidence/t4-dev-service/REPORT.md)).

Manifold fusion:

- The closed-form ridge / GCCA / Mahalanobis-weighted fusion library
  (`python/gen_zero/manifold/`) is described in
  [`manuals/closed_loop_dense_pipeline.md`](manuals/closed_loop_dense_pipeline.md) §1.
  A prior evaluation report for this library tied a 13-task ablation to it, but that
  ablation actually ran through an unrelated random-projection script depending on
  never-extracted Mistral-123B features; both the report and the script were removed
  on 2026-09-29 (see the removal note at the top of `closed_loop_dense_pipeline.md`).
  For a reproducible 13-task manifold ablation, see the Qwen2.5-72B + LLaMA-3.1-70B
  dual-model guide in [`../benchmarks/README.md`](../benchmarks/README.md).

## 2. Operations & mount manuals

`docs/manuals/`

- [Closed-loop dense pipeline: features → manifold fusion → NanoCore mount](manuals/closed_loop_dense_pipeline.md).
  Each step is labelled with its real status. Only the anchor-to-NanoCore mount
  and refusal checks have runnable, passing tests.

Benchmark runbooks (`docs/benchmarks/`):

- [DeepSWE adapter runbook](benchmarks/deepswe_adapter_runbook.md)
- [DeepSWE / Terminal-Bench evaluation plan](benchmarks/deepswe_terminal_bench_eval_plan.md)

## 3. Formal audits & safety proofs

`docs/audits/`

| Document | Scope |
| --- | --- |
| [CI validation, 2026-09-26](audits/ci-validation.md) | CI workflow hardening; Rust and Python test runs |
| [Release packaging, MSRV and path hygiene, 2026-09-27](audits/release-packaging.md) | MSRV 1.88, release workflow, path redaction, link checks |

`docs/road-integration-evidence/`

- [P1–P10 planner integration evidence](road-integration-evidence/README.md):
  conflict decisions and capability limits for the merged planner engines.
  **Four of eight recorded runs exited 101**; this directory does not show a
  green integrated tree.

These documents are a provenance trail. They record what was checked, at which
commit, and what was left unverified. They are not safety certificates for the
current tree.

## 4. Research & benchmark evidence

`docs/research/` (research reports; code citations refer to each report's own baseline)

| Report | Topic |
| --- | --- |
| [b0927c-t2: dense-teacher zero-token continuous world model](research/b0927c-t2-wm/REPORT.md) | World model plan |
| [b0927c-t5: irreversible-action audit and conformal safety barrier](research/b0927c-t5-causal/REPORT.md) | Safety gate |
| [b0928-t4: manifold anchor fixes B11/B12/B13/B18](research/b0928-t4-manifold/report.md) | Manifold anchors |

`docs/benchmarks/evidence/`

- [T3 DeepSWE integration evidence](benchmarks/evidence/t3-deepswe/README.md) (status: incomplete)
- [DeepSWE Gen-Zero integration evidence](benchmarks/evidence/deepswe-genzero-integration/README.md)
- [Gen-Zero audit 2026-09-27 commands](benchmarks/evidence/gen-zero-audit-20260927/COMMANDS.md)
- [Gate fix 2026-09-27 commands](benchmarks/evidence/fix-gate-20260927/COMMANDS.md)

## Published evidence paths

Archived logs and reports use `/home/user` and `/workspace` as redacted host
roots. They are placeholders, not paths required on a reader's machine. Host
path redaction changes log bytes; historical hashes still describe the original
artifacts, not the sanitized copies. Model/patch hashes and recorded outcomes
have not been recomputed or presented as new benchmark runs. Configure the roots
shown in runnable recipes before use. The service evidence scripts require
`GENZERO_EVAL_ROOT` to point to an existing evaluation checkout.

## Rust compatibility

The workspace MSRV is Rust 1.88. All eleven crates inherit it. CI checks and tests
all features on 1.88.0; release binaries and the Docker builder also use 1.88.
Stable CI remains a separate forward-compatibility check. Historical reports
elsewhere in `docs/` describe their original environments, not current release requirements.

This repository does not ship Cargo.lock. CI's MSRV job, release builds and Docker
explicitly resolve dependencies with Cargo's MSRV-compatible fallback before
using `--locked`. The resulting lock is local to each build; this does not promise
identical dependency versions across future builds.
