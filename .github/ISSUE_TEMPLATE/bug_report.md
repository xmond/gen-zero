---
name: Bug Report
about: Create a report to help us improve Gen-Zero
title: '[BUG] '
labels: ['bug', 'triage']
assignees: ''
---

## Description
A clear and concise description of what the bug is.

## Subsystem / Crate
Which component of Gen-Zero is affected?
- [ ] `gen-zero-cli` (Command line interface)
- [ ] `gen-zero-service` (MCP server / SSE / Stdio transport)
- [ ] `gen-zero-planner` (MCTS / A* / GFlowNet / CFR / MPC)
- [ ] `gen-zero-model` (Set-Attention / Simplex ETF / Block causal mask)
- [ ] `gen-zero-nanocore` (Micro-kernel scheduler / Compression / MoV)
- [ ] `gen-zero-gate` (0-1 ILP CP-SAT / PolicyGate / Dual-track safety)
- [ ] `gen-zero-lod` (LodGraph / PPR / Two-stage recall)
- [ ] `gen-zero-worldmodel` (Contact dynamics / Koopman operator)
- [ ] `gen-zero-provenance` (MMR / BLAKE3 decision ledger)
- [ ] `gen-zero-storage` (Causal replay / Golden snapshot / Fenwick)
- [ ] `gen-zero-core` (SIMD / Latent states / Types / Interneer)

## Environment Information
- **OS**: [e.g. Linux Ubuntu 22.04, macOS Sonoma (M3), Windows 11]
- **Rust Version (`rustc --version`)**: [e.g. rustc 1.80.0]
- **Cargo Version (`cargo --version`)**:
- **Hardware Architecture**: [e.g. x86_64 AVX-512, aarch64 Apple Silicon Neon]

## Steps to Reproduce
Steps to reproduce the behavior:
1. Run command '...'
2. Pass context / candidate payload '...'
3. See error / panic

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Output / Backtrace
```
Paste relevant cargo test / runtime logs or RUST_BACKTRACE=1 output here.
```

## Additional Context
Add any other context about the problem here (e.g. MCP host environment: Claude Desktop, Cursor, Custom Agent).
