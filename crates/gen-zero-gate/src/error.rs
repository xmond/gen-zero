//! gen-zero-gate error types.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum GateError {
    #[error(
        "Formal hard interlock triggered: action violates rule {rule_name} (RuleId={rule_id})"
    )]
    ConstraintViolation {
        rule_id: u32,
        rule_name: &'static str,
    },
    #[error("Action prohibited due to revoked status or lacking privileges")]
    PermissionDenied,
    #[error("Constraint compilation error: {0}")]
    CompilationError(String),
    #[error("Solving budget exceeded ({0} ms), fail-closed triggered")]
    Timeout(f32),
    #[error("Two-stage gateway config invalid: {0}")]
    InvalidConfig(String),
    #[error("Two-stage gateway input rejected: {0}")]
    InvalidInput(String),
    #[error("Stage 2 causal verifier failed, fail-closed triggered: {0}")]
    VerifierFailure(String),
    #[error("Stage 2 numerical fault, fail-closed triggered: {0}")]
    NumericalFault(String),
    #[error("Refusal trace sink failed to record event: {0}")]
    RefusalSinkFailure(String),
    #[error("Canary guard failed to record or act on a decision: {0}")]
    CanaryFailure(String),
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
}
