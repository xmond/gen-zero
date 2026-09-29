//! gen-zero-nanocore error types.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum NanoCoreError {
    #[error("Core with DomainId({0}) not found in registry")]
    CoreNotFound(u32),
    #[error("RAM budget exceeded: {current_bytes} > {limit_bytes}")]
    RamBudgetExceeded {
        current_bytes: usize,
        limit_bytes: usize,
    },
    #[error("Confidence watchdog tripped: composite confidence {confidence:.4} < threshold {threshold:.4}")]
    ConfidenceWatchdogTripped { confidence: f32, threshold: f32 },
    #[error("Value watchdog tripped: composite value {value:.4} < threshold {threshold:.4}")]
    ValueWatchdogTripped { value: f32, threshold: f32 },
    #[error("Zstd decompression failure: {0}")]
    CompressionError(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid micro-core: {0}")]
    InvalidCore(String),
    #[error("Candidate {0:?} is not in the micro-core action vocabulary")]
    UnknownAction(String),
    #[error("Candidate {0:?} appears more than once")]
    DuplicateAction(String),
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
    #[error("ETF error: {0}")]
    Etf(#[from] gen_zero_core::EtfError),
    #[error("Model error: {0}")]
    Model(#[from] gen_zero_model::ModelError),
}
