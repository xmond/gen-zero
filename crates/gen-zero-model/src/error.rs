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
    #[error("Reflex plugin config invalid: {0}")]
    ReflexConfig(String),
    #[error("Reflex plugin archive malformed: {0}")]
    ReflexArtifact(String),
    #[error("Reflex plugin input invalid: {0}")]
    ReflexInput(String),
    #[error("Reflex plugin has no head named {0:?}")]
    ReflexUnknownHead(String),
    #[error("Reflex patch invalid: {0}")]
    ReflexPatch(String),
    #[error("Qwen model load failed: {0}")]
    QwenLoad(String),
    #[error("Qwen input rejected: {0}")]
    QwenInput(String),
    #[error("Qwen inference failed: {0}")]
    Inference(String),
}
