//! gen-zero-service
//!
//! Dual-Transport MCP Server (Stdio & SSE / HTTP REST Gateway),
//! Single Polymorphic `zero` Tool Router with 11 Cognitive Verbs,
//! High-Performance simd-json Protocol Loop, and the semantic bridge to the
//! Python scorer used by the `ask`, `route` and `imagine` verbs, and the
//! Spec 25 cognitive runtime (mount snapshot, tangent SSM, geometry gate).

pub mod bridge;
pub mod cognitive;
pub mod error;
pub mod imagine;
pub mod mount;
pub mod pipeline_verb;
pub mod server;
pub mod snapshot;
pub mod tangent_ssm;
pub mod worldsim;
pub mod zero;

pub use bridge::{BridgeConfig, BridgeError, SemanticBridgeClient};
pub use cognitive::{CognitiveRuntime, Rejection};
pub use error::ServiceError;
pub use mount::{
    AtomicMountRegistry, Budget, MountKey, MountRegistry, MountSnapshot, Proposal, Reject,
    RequestBinding, Snapshot, Version,
};
pub use server::McpServer;
pub use zero::{PolymorphicZeroEngine, ZeroContentBlock, ZeroToolOutcome, ZeroVerb};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_initialization() {
        let server = McpServer::new();
        assert!(server.auth_token.is_none());
    }
}
