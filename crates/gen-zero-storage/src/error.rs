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
}
