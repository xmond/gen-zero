# Gen-Zero Comprehensive Benchmark & Competitive Analysis Report

> **Execution Timestamp**: `2026-09-26T16:38:34.529072`  
> **Binary**: `/ebs/pj/gen-zero/target/release/gen-zero` sha256 `8bae5609a8527b0d` (built `2026-09-21T11:06:15.278102`), repo HEAD `59d4215`, loadavg `(19.568359375, 21.22119140625, 16.7626953125)`  
> **Evaluation Profile**: Dual-Engine (Rust Bare-Metal Engine + Python Algorithmic Suite)  
> **License**: Apache-2.0

---

## ⚡ 1. Latency & Throughput Benchmark

Every number below comes from timed `gen-zero reflex` CLI subprocess calls (50 iterations, sequential, single caller).

- **CLI reflex call (process spawn + IO included)**:
  - Mean Latency: `9287.3 µs` (~9.29 ms)
  - P50 Latency: `8247.85 µs`
  - P90 Latency: `11925.48 µs`
  - P95 Latency: `15170.21 µs`
  - P99 Latency: `22425.85 µs`
  - Sequential CLI throughput: `107.63 decisions/s`
- **In-engine reflex latency**: `not_measured`
- **In-engine MCTS latency**: `not_measured`

S1Bench rows in the JSON report are third-party figures, not reproduced here (`provenance` field).

---

## ⚖️ 2. Candidate-Order Flip Rate (empirical)

Counts how often the chosen action changes when the candidate list is shuffled.

- **Valid Comparisons**: `280`
- **Observed Decision Flips**: `0`
- **Candidate Order Flip Rate**: `0.0%` (over valid comparisons only)
- **Error Samples**: `2` (status: `partial_errors`)
- **Scope**: 5 fixed candidate pools of 4, one context per trial; empirical count only, no equivariance proof is claimed

This is an observed count, not a proof, and no LLM baseline was measured here.
