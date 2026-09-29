//! gen-zero-provenance
//!
//! 192-byte Aligned DecisionAuditEntry, Keyed BLAKE3 Merkle Mountain Range (MMR) Ledger,
//! O(log N) Cryptographic Inclusion Proofs, and Granular Capability-Level Token Arbiter.

#![allow(clippy::manual_is_multiple_of)]

pub mod arbiter;
pub mod entry;
pub mod error;
pub mod mmr;

pub use arbiter::{CapabilityArbiter, CapabilityFlags, CapabilityToken};
pub use entry::DecisionAuditEntry;
pub use error::ProvenanceError;
pub use mmr::{MmrInclusionProof, MmrLedger, MAX_AUDIT_LEAVES};

#[cfg(test)]
mod tests {
    use super::*;
    use gen_zero_core::ActionId;

    #[test]
    fn test_provenance_full_pipeline() {
        let key = [0x5au8; 32];
        let mut ledger = MmrLedger::new(key);
        let arbiter = CapabilityArbiter::new(key);

        arbiter.register_agent("worker_alpha", CapabilityFlags::ALL);
        let token = arbiter.issue_token("worker_alpha", 1).unwrap();
        assert!(arbiter
            .authorize(&token, CapabilityFlags::AUDIT_ADMIN)
            .is_ok());

        let entry = DecisionAuditEntry::new(
            0,
            [0; 32],
            [1; 32],
            [2; 32],
            [3; 32],
            ActionId(7),
            0,
            12345678,
        );

        let leaf_idx = ledger.append(entry);
        assert_eq!(leaf_idx, 0);

        let proof = ledger.generate_proof(0).unwrap();
        assert!(MmrLedger::verify_against_root(
            &key,
            &proof,
            &ledger.get_root(),
            &ledger.get_entry(0).unwrap().hash_entry(&key)
        )
        .unwrap());
    }
}
