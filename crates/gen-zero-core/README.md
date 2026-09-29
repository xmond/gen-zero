# gen-zero-core

Mathematical foundations for the Gen-Zero Rust decision engine: SIMD kernels, the
Helmert Simplex Equiangular Tight Frame (ETF), action interning, and the trait
contracts every other crate in the workspace builds on.

## Architecture

- `error`: core error types (`CoreError`, `EtfError`, `InternerError`).
- `etf`: Helmert regular simplex ETF used to assign geometric vertices to actions.
- `interner`: thread-safe interner that maps action symbols to `ActionId`s.
- `simd`: SIMD-accelerated math kernels (cosine/L2/Hamming distance, dot product)
  and POPCNT bit operations.
- `traits`: fundamental trait contracts shared across subsystems (decision
  engines, world model dynamics, encoders, safety estimates).
- `types`: fundamental mathematical and action types (latents, entropy, action
  frames).

## Key exports

- `CoreError`, `EtfError`, `InternerError`: crate error types.
- `SimplexEtfFrame`: Helmert simplex ETF frame.
- `ActionInterner`: thread-safe action symbol interner (deprecated API).
- `cosine_distance_f32`, `dot_product_f32`, `hamming_distance_u64`, `l2_distance_f32`:
  SIMD math kernels.
- `GraphFactProvider`, `LatentContraction`,
  `LosslessInvertibleEncoder`, `SafetyEstimate`, `WorldModelDynamics`: trait
  contracts implemented by other crates.
- `ActionId`, `CompressedLatent`, `FoundationLatent`, `FullLatent`, `LatentState`,
  `LocalActionFrame`, `MicroLatent`, `NormalizedEntropy`: core value types.

## Dependencies

None. `gen-zero-core` has no internal gen-zero dependencies; every other crate
in the workspace depends on it.
