# gen-zero-provenance

Audit and access-control primitives for Gen-Zero: a 192-byte aligned decision
audit entry, a keyed BLAKE3 Merkle Mountain Range (MMR) ledger with O(log N)
inclusion proofs, and a capability-level token arbiter.

## Architecture

- `arbiter`: capability-level permission arbiter and agent sandbox registry.
  Granular capability-based access control (Execute, Query, Rollback, Admin)
  with per-agent capability tokens.
- `entry`: `DecisionAuditEntry` layout (Doc 05): exact 192-byte size, exact
  64-byte alignment (3 cache lines).
- `error`: crate error type (`ProvenanceError`).
- `mmr`: Merkle Mountain Range ledger with keyed BLAKE3 cryptographic
  guarantees. Monotonically append-only; amortized O(1) append with an
  incrementally updated, O(1)-read root.

## Key exports

- `CapabilityArbiter`, `CapabilityFlags`, `CapabilityToken`: capability-based
  access control.
- `DecisionAuditEntry`: the 192-byte audit entry.
- `ProvenanceError`: crate error type.
- `MmrInclusionProof`, `MmrLedger`, `MAX_AUDIT_LEAVES`: the MMR ledger and its
  inclusion proofs.

## Dependencies

- `gen-zero-core`: base types (`ActionId` used in audit entries).
