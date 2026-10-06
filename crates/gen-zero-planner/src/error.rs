//! gen-zero-planner Error Types.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum PlannerError {
    #[error("Search encountered a masked dead end; use plan_report or a dead-end observer")]
    DeadEndRequiresReport,
    #[error("A* requires an explicit search goal")]
    MissingSearchGoal,
    #[error("A* goal is unreachable in the candidate-action graph")]
    SearchGoalUnreachable,
    #[error("A* expansion budget exhausted after {expanded} expansions")]
    SearchBudgetExceeded { expanded: usize },
    #[error("A* input already satisfies the goal; no first action exists")]
    SearchAlreadyAtGoal,
    #[error("A previous deadline worker is still running")]
    PlannerBusy,
    #[error("Arena memory capacity exceeded (max {capacity} nodes)")]
    ArenaCapacityExceeded { capacity: usize },
    #[error("Tree node index out of bounds: {0}")]
    NodeIndexOutOfBounds(u32),
    #[error("No feasible action found satisfying formal constraints")]
    NoFeasibleAction,
    #[error("Planning timeout exceeded ({0:.2} ms)")]
    TimeoutExceeded(f64),
    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Operator table must have at least one action's forward/backward operators")]
    EmptyOperatorTable,
    #[error("Horizon {horizon} outside 1..={max}")]
    InvalidHorizon { horizon: usize, max: usize },
    #[error("Causal gate passed 0 of {sampled} sampled plans (reasons {reasons})")]
    CausalGateEmpty { sampled: usize, reasons: String },
    #[error("Causal plan infeasible before sampling: {0}")]
    CausalInfeasible(String),
    #[error("Divergent state refused: {0}")]
    DivergentState(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error(
        "Unknown decide mode {0:?}; expected auto, mcts, mpc_cem, astar, manifold_gflownet, cfr_nash, reflex, causal_triad or tournament_triad"
    )]
    UnknownMode(String),
    #[error(
        "World model gave a safety estimate on {covered} of {steps} steps; risk cannot be scored"
    )]
    MissingSafetyEstimate { covered: usize, steps: usize },
    #[error("World model safety estimate {0} is outside [0, 1]")]
    InvalidSafetyEstimate(f32),
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
}
