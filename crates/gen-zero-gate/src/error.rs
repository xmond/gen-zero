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
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
}
