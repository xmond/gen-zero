//! Decision Audit Entry layout adhering to Doc 05.
//!
//! Strict layout invariant:
//! - Exact size: 192 bytes
//! - Exact alignment: 64 bytes (occupies exactly 3 cache lines)

use gen_zero_core::ActionId;
use serde::{Deserialize, Serialize};

/// Cryptographic Decision Audit Entry: strictly 192 bytes aligned to 64 bytes.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionAuditEntry {
    pub leaf_index: u64, // offset   0..8   (8 bytes: monotonic leaf index)
    pub previous_mmr_root: [u8; 32], // offset   8..40  (32 bytes: prior MMR peak/root hash)
    pub state_digest: [u8; 32], // offset  40..72  (32 bytes: state context hash)
    pub candidate_actions_hash: [u8; 32], // offset  72..104 (32 bytes: candidate actions hash)
    pub solved_ilp_proof: [u8; 32], // offset 104..136 (32 bytes: CP-SAT formal cert)
    pub chosen_action_id: ActionId, // offset 136..140 (4 bytes: ActionId u32)
    pub policy_tier: u8, // offset 140..141 (1 byte: PolicyDecisionTier scalar)
    pub _pad: [u8; 3],   // offset 141..144 (3 bytes: natural alignment padding)
    pub timestamp_nanos: u64, // offset 144..152 (8 bytes: timestamp)
    pub _reserved_1: [u8; 32], // offset 152..184 (32 bytes: reserved padding)
    pub _reserved_2: [u8; 8], // offset 184..192 (8 bytes: reserved padding)
}

// Static compile-time layout assertions
const _: () = assert!(std::mem::size_of::<DecisionAuditEntry>() == 192);
const _: () = assert!(std::mem::align_of::<DecisionAuditEntry>() == 64);

impl DecisionAuditEntry {
    /// Create a new audit entry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        leaf_index: u64,
        previous_mmr_root: [u8; 32],
        state_digest: [u8; 32],
        candidate_actions_hash: [u8; 32],
        solved_ilp_proof: [u8; 32],
        chosen_action_id: ActionId,
        policy_tier: u8,
        timestamp_nanos: u64,
    ) -> Self {
        Self {
            leaf_index,
            previous_mmr_root,
            state_digest,
            candidate_actions_hash,
            solved_ilp_proof,
            chosen_action_id,
            policy_tier,
            _pad: [0; 3],
            timestamp_nanos,
            _reserved_1: [0; 32],
            _reserved_2: [0; 8],
        }
    }

    /// Compute Keyed BLAKE3 hash of this audit entry.
    pub fn hash_entry(&self, key: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new_keyed(key);
        // Canonical little-endian encoding preserves the 192-byte layout without
        // depending on host endianness or reading a Rust object's memory.
        hasher.update(&self.leaf_index.to_le_bytes());
        hasher.update(&self.previous_mmr_root);
        hasher.update(&self.state_digest);
        hasher.update(&self.candidate_actions_hash);
        hasher.update(&self.solved_ilp_proof);
        hasher.update(&self.chosen_action_id.0.to_le_bytes());
        hasher.update(&[self.policy_tier]);
        hasher.update(&self._pad);
        hasher.update(&self.timestamp_nanos.to_le_bytes());
        hasher.update(&self._reserved_1);
        hasher.update(&self._reserved_2);
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_canonical_192_byte_layout() {
        let entry = DecisionAuditEntry::new(
            0x0102030405060708,
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            ActionId(0x11223344),
            5,
            0x0807060504030201,
        );
        let mut bytes = [0; 192];
        bytes[0..8].copy_from_slice(&entry.leaf_index.to_le_bytes());
        bytes[8..40].fill(1);
        bytes[40..72].fill(2);
        bytes[72..104].fill(3);
        bytes[104..136].fill(4);
        bytes[136..140].copy_from_slice(&[0x44, 0x33, 0x22, 0x11]);
        bytes[140] = 5;
        bytes[144..152].copy_from_slice(&entry.timestamp_nanos.to_le_bytes());
        assert_eq!(
            entry.hash_entry(&[7; 32]),
            *blake3::keyed_hash(&[7; 32], &bytes).as_bytes()
        );
        assert_eq!(
            std::mem::offset_of!(DecisionAuditEntry, timestamp_nanos),
            144
        );
        assert_eq!(std::mem::offset_of!(DecisionAuditEntry, _reserved_2), 184);
    }

    #[test]
    fn test_decision_audit_entry_size_and_alignment() {
        assert_eq!(std::mem::size_of::<DecisionAuditEntry>(), 192);
        assert_eq!(std::mem::align_of::<DecisionAuditEntry>(), 64);

        let entry = DecisionAuditEntry::new(
            0,
            [1; 32],
            [2; 32],
            [3; 32],
            [4; 32],
            ActionId(42),
            0,
            123456789,
        );

        let key = [7u8; 32];
        let h1 = entry.hash_entry(&key);
        let h2 = entry.hash_entry(&key);
        assert_eq!(h1, h2);
    }
}
