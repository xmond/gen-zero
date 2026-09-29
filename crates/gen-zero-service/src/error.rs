//! Error definitions for gen-zero-service.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error("CPU capacity exhausted; retry later")]
    Overloaded,

    #[error("MCP JSON-RPC parsing error: {0}")]
    JsonRpc(String),

    #[error("Method not found: {0}")]
    MethodNotFound(String),

    #[error("Unknown verb or invalid intent for 'zero' tool: {0}")]
    InvalidVerb(String),

    #[error("Safety interlock rejected: {0}")]
    SafetyRejected(String),

    #[error("Confirmation required for destructive action: {0}")]
    ConfirmationRequired(String),

    #[error("Core error: {0}")]
    Core(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
