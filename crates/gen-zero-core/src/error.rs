//! gen-zero-core error types.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum EtfError {
    #[error("Candidate count must be at least 1, got 0")]
    ZeroCandidates,
    #[error("Embedding dimension {dimension} is strictly less than required degrees of freedom {required} (K - 1)")]
    DimensionTooLow { dimension: usize, required: usize },
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum InternerError {
    #[error("Action interner address space exhausted (max 2^32 entries)")]
    AddressSpaceExhausted,
    #[error("Symbol not found for ActionId({0})")]
    SymbolNotFound(u32),
}

#[derive(Error, Debug, Clone, PartialEq)]
pub enum CoreError {
    #[error("ETF error: {0}")]
    Etf(#[from] EtfError),
    #[error("Interner error: {0}")]
    Interner(#[from] InternerError),
    #[error("Numerical instability error: {0}")]
    NumericalInstability(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// World-model failure with no closer core variant (bad horizon, bad action).
    /// Carried as text because gen-zero-core cannot depend on gen-zero-worldmodel.
    #[error("World model error: {0}")]
    WorldModel(String),
}
