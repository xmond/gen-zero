//! gen-zero-service
//!
//! Dual-Transport MCP Server (Stdio & SSE / HTTP REST Gateway),
//! Single Polymorphic `zero` Tool Router with 24 Cognitive Verbs,
//! High-Performance simd-json Protocol Loop, the semantic backend of the
//! `ask`, `route` and `imagine` verbs (native Qwen in process, or the HTTP
//! bridge to the Python scorer), and the Spec 25 cognitive runtime (mount
//! snapshot, tangent SSM, geometry gate).

#![allow(clippy::result_large_err)]

pub mod arbitrator;
pub mod bridge;
pub mod closed_loop;
pub mod cognitive;
pub mod error;
pub mod graph_verb;
pub mod hot_reload;
pub mod imagine;
pub mod mount;
pub mod patch_builder;
pub mod pipeline_verb;
pub mod reflex_adapter;
pub mod reflex_registry;
pub mod semantic;
pub mod server;
pub mod snapshot;
pub mod tangent_ssm;
pub mod text_to_graph;
pub mod uds;
pub mod worldsim;
pub mod zero;

pub use arbitrator::{
    run_once, spawn_arbitrator_daemon, ArbitrationBatchReport, ArbitratorError, LlmArbitratorConfig,
};
pub use bridge::{BridgeConfig, BridgeError, SemanticBridgeClient};
pub use closed_loop::{
    spawn_feedback_syncer, spawn_patch_poller, ClosedLoopConfig, FeedbackBuffer, FeedbackRecord,
};
pub use cognitive::{CognitiveRuntime, Rejection};
pub use error::ServiceError;
pub use hot_reload::{
    CanaryConfig, CanaryDecision, CanaryGuard, HotReloadError, HotReloadManager, MemoryRollback,
    RollbackReport,
};
pub use mount::{
    AtomicMountRegistry, Budget, MountKey, MountRegistry, MountSnapshot, Proposal, Reject,
    RequestBinding, Snapshot, Version,
};
pub use patch_builder::{
    spawn_patch_builder_daemon, validate_patch, CompiledPatch, PatchBuildConfig, PatchBuilder,
    PatchBuilderError, TrainingSamplePair,
};
pub use reflex_adapter::{AdaptationReport, ReflexOnlineAdapter};
pub use reflex_registry::{ReflexError, ReflexRegistry};
pub use semantic::{NativeConfig, NativeQwen, SemanticBackend};
pub use server::McpServer;
pub use text_to_graph::{
    AnswerabilityReport, InduceOutcome, InduceRequest, InductedAction, TextToGraphInducer,
};
pub use zero::{
    PolymorphicZeroEngine, ZeroContentBlock, ZeroEngineConfig, ZeroToolOutcome, ZeroVerb,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_initialization() {
        let server = McpServer::new();
        assert!(server.auth_token.is_none());
    }
}
