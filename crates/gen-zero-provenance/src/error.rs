//! Error definitions for gen-zero-provenance.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProvenanceError {
    #[error("MMR error: {0}")]
    MmrError(String),

    #[error("MMR persistence error: {0}")]
    PersistenceError(String),

    #[error("Invalid inclusion proof: leaf {leaf_index} cannot be verified")]
    InvalidInclusionProof { leaf_index: u64 },

    #[error("Leaf {leaf_index} left the proof window; oldest provable leaf is {oldest_retained}")]
    LeafPruned {
        leaf_index: u64,
        oldest_retained: u64,
    },

    #[error("Permission denied: agent {agent_id} lacks capability {capability:?}")]
    PermissionDenied {
        agent_id: String,
        capability: String,
    },

    #[error("Audit log verification failed: expected {expected}, actual {actual}")]
    VerificationFailed { expected: String, actual: String },
}
