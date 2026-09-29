# gen-zero-nanocore

Quantized micro-kernel fleet for Gen-Zero: Mixture of Vectors (MoV) fusion, a
fallback watchdog, and a bounded-RAM LRU fleet scheduler.

## Architecture

- `core_type`: domain specifications and micro-kernel instances (`NanoCoreInstance`,
  domain identifiers for browser, code, general, safety, SQL, trading and vision).
  Each core carries a required `action_vocab`: ETF vertex `i` always scores
  `action_vocab[i]`, and `forward` takes candidate names, so reordering or
  pruning candidates never changes a surviving action's score. Unknown or
  repeated candidates are errors. Build production cores with `from_parts`
  (validated); the synthetic sin/cos weights live in `core_type::fixtures`,
  compiled only for tests or with the dev-only `test-fixtures` feature.
- `error`: crate error type (`NanoCoreError`).
- `mov`: Mixture of Vectors (MoV) fusion engine and fallback watchdog.
- `scheduler`: fleet scheduler with a bounded RAM budget and LRU eviction.

## Key exports

- `DomainId`, `NanoCoreInstance`, `DOMAIN_BROWSER`, `DOMAIN_CODE`,
  `DOMAIN_GENERAL`, `DOMAIN_SAFETY`, `DOMAIN_SQL`, `DOMAIN_TRADING`,
  `DOMAIN_VISION`: domain specifications and instances.
- `NanoCoreError`: crate error type.
- `FusedDecision`, `MoVFusionEngine`, `WatchdogConfig`: MoV fusion and
  fallback watchdog.
- `NanoCoreFleetScheduler`, `DEFAULT_RAM_BUDGET_BYTES`: bounded-RAM LRU fleet
  scheduler.

## Dependencies

- `gen-zero-core`: base types and traits.
- `gen-zero-model`: shared model primitives used by the micro-kernel fleet.
