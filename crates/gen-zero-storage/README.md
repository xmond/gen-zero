# gen-zero-storage

Storage primitives for Gen-Zero: a columnar causal replay buffer with
prioritized sampling, chunked zstd streaming, and golden snapshot rollback.
The production service uses `GoldenSnapshotManager` to capture and restore
mounted cognitive assets; the replay buffer remains available to training
consumers.

## Architecture

- `error`: crate error type (`StorageError`).
- `fenwick`: Fenwick tree (binary indexed tree) for O(log N) prioritized
  causal sampling (Causal PER). Maintains prefix sums of transition
  priorities with O(log N) point updates and prefix-sum/quantile lookups.
- `replay`: columnar causal replay buffer with Causal PER and SIMD vector
  batches. Contiguous columnar arrays for states, actions, rewards, next
  states and dones, backed by the Fenwick tree for prioritized sampling.
- `snapshot`: golden snapshot manager with zstd chunked streaming and
  copy-on-write rollback. Sub-16ms atomic snapshot capture and rollback,
  with 85%+ memory footprint reduction from chunked zstd compression.

## Key exports

- `StorageError`: crate error type.
- `FenwickTree`: prioritized sampling tree.
- `CausalSampleBatch`, `ColumnarCausalReplayBuffer`, `TrajectoryStep`: the
  columnar causal replay buffer.
- `GoldenSnapshot`, `GoldenSnapshotManager`: snapshot capture and rollback.

## Dependencies

- `gen-zero-core`: base types (`ActionId`, `FullLatent`) used by the replay
  buffer.
