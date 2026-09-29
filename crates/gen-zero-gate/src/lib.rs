//! # gen-zero-gate
//!
//! Formal safety interlocks, 0-1 ILP linear constraint compiler,
//! 4-tier PolicyGate state machine, and dual-track promotion gates.

pub mod constraint;
pub mod dual_track;
pub mod error;
pub mod policy;
pub mod risk;
pub mod sheaf_gate;

pub use constraint::{LinearConstraint, RuleId};
pub use dual_track::{DualTrackVerifier, SafetyAuditReport};
pub use error::GateError;
pub use policy::{GateVerdict, PolicyGate, PolicyTier};
pub use risk::SemanticRisk;
pub use sheaf_gate::{
    AcceptedState, Budget, CandidateState, CertifiedCandidate, DynamicsStep, EnergyRoseKind,
    ExplicitMatrixProblem, GeometryGate, LaplacianHeatFlowGate, ManifoldGuard, Pin, Reject,
    RelaxationStatus, SheafOperator, SheafProblem, WindowDynamicsProblem,
};
