//! Error types for gen-zero-storage.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Capacity exceeded: max entries is {max}")]
    CapacityExceeded { max: usize },

    #[error("Replay buffer empty: cannot sample")]
    BufferEmpty,

    #[error("Invalid replay sampling parameters: {0}")]
    InvalidSampleParameters(&'static str),

    #[error("Invalid replay priority: {0}")]
    InvalidPriority(&'static str),

    #[error("Snapshot not found: ID {0}")]
    SnapshotNotFound(u64),

    #[error("Zstd compression error: {0}")]
    Compression(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Integrity error: checksum mismatch (expected {expected}, got {actual})")]
    IntegrityError { expected: String, actual: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Reflex feedback store schema drift: expected columns {expected:?}, got {actual:?}")]
    SchemaMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },

    #[error("Failed to set PRAGMA {pragma}: expected {expected:?}, got {actual:?}")]
    PragmaFailed {
        pragma: &'static str,
        expected: String,
        actual: String,
    },

    #[error("Reflex trace input dimension mismatch: expected {expected}, got {actual}")]
    InputDimMismatch { expected: usize, actual: usize },

    #[error("Reflex trace not found: {0}")]
    TraceNotFound(String),

    #[error("Reflex trace {0} already has recorded feedback; feedback is write-once")]
    FeedbackAlreadyRecorded(String),

    #[error("Reflex trace blob corrupt: {0}")]
    CorruptBlob(String),

    #[error("Durable refusal store schema drift on table {table}: expected columns {expected:?}, got {actual:?}")]
    DurableSchemaMismatch {
        table: &'static str,
        expected: Vec<String>,
        actual: Vec<String>,
    },

    #[error("Durable refusal trace not found: {0}")]
    DurableTraceNotFound(String),

    #[error(
        "Durable refusal trace {trace_id} is not in the expected arbitration state: expected {expected}, got {actual}"
    )]
    InvalidArbitrationState {
        trace_id: String,
        expected: &'static str,
        actual: String,
    },

    #[error("Durable refusal trace {0} is already consumed")]
    TraceAlreadyConsumed(String),

    #[error("Durable patch not found: {0}")]
    DurablePatchNotFound(String),

    #[error("Durable patch {0} has already been rolled back")]
    PatchAlreadyRolledBack(String),

    #[error("Durable patch {patch_id} is not ACTIVE (status {status})")]
    PatchNotActive { patch_id: String, status: String },

    #[error("Invalid canary metric request: {0}")]
    InvalidCanaryMetric(String),

    #[error("Invalid patch commit: {0}")]
    InvalidPatchCommit(String),

    #[error("Unrecognized arbitration status string: {0:?}")]
    InvalidArbitrationStatusString(String),

    #[error("Durable refusal store mutex poisoned")]
    LockPoisoned,
}
