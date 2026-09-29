# Contributing to Gen-Zero

Thank you for contributing. For nontrivial changes, open an issue first to discuss scope and acceptance criteria. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md), and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development environment

- Rust 1.88 or newer, with Cargo, `rustfmt`, and `clippy`.
- Python 3.10 or newer only for contributions to the optional Python package and examples. Use a virtual environment for Python dependencies.

```bash
rustup component add rustfmt clippy
```

For Python work, additionally run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './python[dev]'
```

The Python package declares optional `all` dependencies in `python/pyproject.toml`; install them when the affected feature needs them. Check the relevant package and test prerequisites before running its tests.

## Style and verification

Run checks for the languages you changed:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
ruff check python
python -m pytest python/gen_zero/tests
```

Use `cargo fmt --all` and `ruff format python` to format edited code. In the pull request, record the exact commands, exit codes, and any checks you could not run. A passing command proves only the behavior it exercised; do not describe untested safety, performance, or model behavior as verified.

## Test contract and fail-closed behavior

- Add tests for changed behavior, including failure paths, boundary cases, and the production entry point that invokes it. For safety or provenance changes, include a test showing that invalid or unavailable inputs cannot produce a successful decision.
- Preserve the real data flow in tests. Do not use mocks, hard-coded answers, or input-format matching to claim that an algorithm or model works. Document fixture scope and limitations.
- When a required dependency, mount, model, solver, or evidence source is unavailable, surface an explicit error or clearly labeled degraded result with diagnostic logging. Do not silently substitute a scan, heuristic, cached answer, or other path that changes the contract.
- Verify integration through the affected CLI, HTTP, MCP, or scheduler path where applicable. A unit test alone does not establish that production calls the new code.
- Keep claims proportional to evidence. Mathematical arguments do not replace paired sample statistics, and model capability claims require real training and evaluation evidence.

## Pull requests

Keep changes focused, explain the user-visible contract, and link relevant issues. Include test commands and results, known limitations, and any migrations or disclosure concerns. Preserve third-party notices and license headers. Contributions are submitted under the repository's [Apache License 2.0](LICENSE) unless explicitly stated otherwise, as described in section 5 of that license.
