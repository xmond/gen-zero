# CI validation — 2026-09-26

Scope: `.github/workflows/ci.yml` and the Python dev dependency required by its
real training-script tests. Baseline: `fd6b67137db3942e77b384cfab789c4808d458e5`.
No application algorithms, test bodies, ignored-test annotations or release
behavior were changed. No agents/reviewers were dispatched.

> Historical report: references to Rust 1.80 below describe the original audit.
> The current MSRV is 1.88; see [current packaging policy](../README.md#rust-compatibility).
> Release configuration has since changed; outstanding findings below are historical.

## Implemented

- `ci.yml:9`: read-only repository permission; no private secrets or privileged
  PR trigger; checkout credentials are not persisted; all Actions pinned to SHA.
- `ci.yml:13`: superseded runs cancelled; every job has a finite timeout.
- `ci.yml:59`: existing Linux/macOS Rust tests now also run `cargo check --workspace`.
  Stable Rust satisfies >=1.80; this is **not** evidence of 1.80 MSRV compatibility.
- `ci.yml:89`: Python 3.10/3.11 install the package with dev dependencies, check
  dependencies and directly execute all four requested test modules. CPU Torch is
  explicitly installed from its CPU index, not selected through a fallback.
- `python/pyproject.toml:63`: dev dependencies include scikit-learn because
  `scripts/train_world_model_dynamics.py:43` imports its ROC-AUC metric.
  A clean 3.10 run exposed this missing dependency (exit 1: 4 failed, 100 passed);
  the tests were not weakened to hide it.

## Reproducible verification

Raw logs and original exit-code files: `/tmp/t8-ci-evidence/` on the task host.
The Python 3.10 environment was created with `uv venv --python 3.10`, then
`uv pip install --python <venv>/bin/python 'torch>=2.0.0' --index-url
https://download.pytorch.org/whl/cpu` and `uv pip install --python <venv>/bin/python
-e './python[dev]'`. This uses uv locally; CI's pip installer itself has not run
on GitHub here. Python 3.11 used the existing local environment.

Both test commands set `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` and execute:

```sh
python -m pytest -ra --strict-config --strict-markers \
  python/gen_zero/tests/test_neural_dynamics.py \
  python/gen_zero/tests/test_world_model_simulation_endpoints.py \
  python/gen_zero/tests/test_gen_zero.py \
  python/gen_zero/tests/test_issue_93_orthogonal_6_planner_engines.py
```

Rust ran on dev with `CARGO_BUILD_JOBS=$(nproc) RUSTFLAGS="-D warnings"`,
rustc 1.98.1, in an isolated directory without `.git`. Preflight: 64 CPUs,
1m/15m loads 1.52/0.63, 86,951 MiB available memory, 378,991,320 KiB free on
source/build filesystem. Only local `Cargo.toml` and `crates/` were transferred;
source archive SHA-256 matched on both ends:
`04beaad401bc0f09a0ca52eb6901433ac0213bf308517cec66631f014fa93c18`.

| Command | Original exit | Evidence / output tail |
| --- | --- | --- |
| `actionlint .github/workflows/ci.yml` (v1.7.12) | 0 | Empty output (`actionlint.log`) |
| `cargo check --workspace` | 0 | `Finished dev profile ... in 27.11s` (`check.log`) |
| `cargo test --workspace` | 0 | 359 passed, 0 failed, **6 existing ignored** across test binaries/doc suites (`test.log`); final doc suite: `0 passed; 0 failed` |
| Python 3.10.19 command above after dependency fix | 0 | `104 passed in 26.57s` (`pytest310-fixed.log`) |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | 101 | `could not compile gen-zero-gate (lib test)`; `field_reassign_with_default` at `crates/gen-zero-gate/src/sheaf_gate.rs:708` (`clippy.log`) |
| Python 3.11.16 command above | 0 | `104 passed in 27.57s` (`pytest311.log`) |
| `cargo fmt --all -- --check` | 1 | Existing formatting differences, including `crates/gen-zero-worldmodel/src/symplectic.rs:314` (`fmt.log`) |
| `actionlint .github/workflows/release.yml` | 1 | `release.yml:33:17: label "macos-13" is unknown` (`release-actionlint.log`) |

## Unverified

GitHub-hosted execution, macOS compilation, exact Rust 1.80 compatibility, and
release publishing have not been exercised. No push, deployment or online green
status is claimed. Dependencies remain version ranges, and Cargo.lock is absent;
these runs do not prove future dependency resolution is reproducible.

## Outstanding audit findings

`release.yml` was reviewed but is outside this CI implementation change. It still
has global `contents: write` / `packages: write` (`release.yml:9`), no job timeout,
mutable Action references, persisted checkout credentials, an unpinned cross Git
install (`release.yml:62`), and the invalid runner label (`release.yml:33`). Its
only explicit secret references are GitHub's generated `GITHUB_TOKEN`; this does
not establish a repository-wide absence of secret leaks.

Existing Rust formatting and Clippy failures remain enforced by CI. No `continue-on-error`,
`|| true`, mock replacement or test-selection fallback was introduced. There is
no new runtime module to wire or replaced application symbol to remove: the
actual push/PR workflow calls the real package tests and Rust workspace commands.

## Raw evidence integrity

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  actionlint.log
1811914cb58bfe1987b2b2797ef35a5285d03f5bef5333f9308067473b095f49  check.log
a9ac74fe25dee2985f8c136ce88aa600cb487024cbe343b975434a6206e06387  test.log
0bcf0df4c7767cd5f5391ac2232e77465b395445ff1cf88aac702e7950b259f1  fmt.log
62a5ced043b1684851999a032ea388409a24e722350a9d5ba199fbe52e7d70c7  clippy.log
bef7f7a4af0cea9e1a901c498d5bf3a8d0518a4a14ca2e1cbb706e92e60467ee  pytest311.log
ad710ecc1d56b93a0b395e62e42524aa19ffaa3b4fb6f993d115b570eb0a6075  pytest310.log
4a8fc0005d11ee28f8a27808bbc8ca9f4136d91231315b34b564c0fa000542db  pytest310-fixed.log
7de199ea3342c5d99f2a62792ebd2286ca16629eb22d7b273130f1d7dc6d1b16  release-actionlint.log
```
