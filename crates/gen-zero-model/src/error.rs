//! gen-zero-model error types.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ModelError {
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
    #[error("Options list cannot be empty")]
    EmptyOptions,
    #[error("Option at index {option_index} has zero length")]
    EmptyOptionLength { option_index: usize },
    #[error("Temperature must be strictly positive, got {0}")]
    InvalidTemperature(String),
    #[error("Candidate count mismatch: expected {expected}, got {actual}")]
    CandidateMismatch { expected: usize, actual: usize },
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
}
