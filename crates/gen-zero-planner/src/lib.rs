//! gen-zero-planner
//!
//! 6 Orthogonal Planning Engines, the gated `ProductionPipeline`, and the
//! Dynamic K-MoE Router.

pub mod config;
pub mod engine;
pub mod error;
mod game;
pub mod pipeline;
pub mod router;

pub use config::PlannerConfig;
pub use engine::{
    AStarEngine, AStarGoal, AStarPath, CemDistribution, CemPlan, CfrNashEngine, CpSatFormalEngine,
    FlowDistribution, FormalSolveOptions, FormalSolveResult, GameSolution, ManifoldGFlowNetEngine,
    MctsEngine, MpcCemEngine, NormalFormGame, PlanningEngine, SolveStatus,
};
pub use error::PlannerError;
pub use pipeline::{
    AuditReport, AuditVerdict, CandidateOutcome, DecideMode, DecideRequest, Decision,
    ProductionPipeline, PrunedAction, Rollout, SimStep, WhatIfReport, DEFAULT_WARN_RISK,
    MAX_DECIDE_CANDIDATES, MAX_HORIZON, MAX_WHAT_IF_CANDIDATES,
};
pub use router::{DynamicKMoERouter, RoutingTier};
