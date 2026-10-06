//! # gen-zero-gate
//!
//! Formal safety interlocks, 0-1 ILP linear constraint compiler,
//! 4-tier PolicyGate state machine, and the two-stage answerability gateway.

pub mod constraint;
pub mod error;
pub mod policy;
pub mod risk;
pub mod sheaf_gate;
pub mod two_stage;

pub use constraint::{LinearConstraint, RuleId};
pub use error::GateError;
pub use policy::{GateVerdict, PolicyGate, PolicyTier};
pub use risk::SemanticRisk;
pub use sheaf_gate::{
    AcceptedState, Budget, CandidateState, CertifiedCandidate, DynamicsStep, EnergyRoseKind,
    ExplicitMatrixProblem, GeometryGate, LaplacianHeatFlowGate, ManifoldGuard, Pin, Reject,
    RelaxationStatus, SheafOperator, SheafProblem, WindowDynamicsProblem,
};
pub use two_stage::{
    CausalVerifier, RefusalTraceEvent, RefusalTraceSink, TeacherProjectionPair, TeacherWeights,
    TriTeacherProjections, TwoStageConfig, TwoStageDualTrackGateway, TwoStageGateEvidence,
    VerifierType, MIN_PROJECTION_NORM,
};
