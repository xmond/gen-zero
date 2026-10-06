//! gen-zero-planner
//!
//! 6 Orthogonal Planning Engines, the gated `ProductionPipeline`, and the
//! Dynamic K-MoE Router, plus the exact AND/OR causal-DAG planner behind the
//! `causal_plan` verb and the sampled causal triad behind the `causal_triad`
//! and `tournament_triad` decide modes.

pub mod causal_dag;
pub mod config;
pub mod engine;
pub mod error;
mod game;
pub mod pipeline;
pub mod router;
pub mod triad;

pub use causal_dag::{
    exact_causal_plan, CausalDagError, CausalDagPlan, CausalDagRequest, CausalNode, MAX_CONE_NODES,
    MAX_DAG_NODES, MAX_EXPANDED_STATES, PLAN_DEADLINE,
};
pub use config::PlannerConfig;
pub use engine::{
    AStarEngine, AStarGoal, AStarPath, CemDistribution, CemPlan, CfrNashEngine, CpSatFormalEngine,
    FlowDistribution, FormalSolveOptions, FormalSolveResult, GameSolution, ManifoldGFlowNetEngine,
    MctsEngine, MpcCemEngine, NormalFormGame, PlanReport, PlanningEngine, SolveStatus,
};
pub use error::PlannerError;
pub use pipeline::{
    AuditReport, AuditVerdict, CandidateOutcome, DecideMode, DecideRequest, Decision, GraphContext,
    GraphFact, ProductionPipeline, PruneSource, PrunedAction, Rollout, SimStep, WhatIfReport,
    DEFAULT_WARN_RISK, MAX_DECIDE_CANDIDATES, MAX_HORIZON, MAX_WHAT_IF_CANDIDATES,
};
pub use router::{DynamicKMoERouter, RoutingTier};
pub use triad::{CausalTriadRequest, TournamentTriadPipeline, TriadReport};

pub mod masked_dynamics;
pub use masked_dynamics::{FiniteStateActionMask, MaskedDynamics, StateActionMask};
