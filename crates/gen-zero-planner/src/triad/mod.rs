//! Causal triad: geodesic-flow plan sampling, a deterministic causal gate, and
//! a multi-core sharded tournament.
//!
//! Rust port of the gen-zero-research repository's `python/gen_zero/planner/triad/`. Reached from production
//! through `DecideMode::CausalTriad` (one shard, inline) and
//! `DecideMode::TournamentTriad` (sharded, parallel) of
//! [`crate::ProductionPipeline::decide`], with a [`CausalTriadRequest`] on the
//! request. Nothing here calls a world model or any external system: the triad
//! plans against the DAG's nominal model only, and `decide` keeps its own
//! PolicyGate pruning and world-model screening around it.
//!
//! The sampler is energy-steered by default: a causal potential
//! `Phi = -(g + h)` biases every draw by `alpha * dPhi` (`alpha` = 2.0, set per
//! request by [`TriadRunOptions::energy_alpha`]). The potential and the
//! four-band layout of the DAG live in `gen_zero_lod::causal_lod`; see
//! [`sampler`] for what the ablation behind the steering did and did not show.
//!
//! Optional robust arbitration ([`robust_gate`]): `TriadRunOptions::robust`
//! carries a fitted disturbance model; both tiers then rank gate-passing plans
//! by `P(finish <= budget)` under it, try causal reorders, and can commit the
//! cheapest legal probe action first. See [`robust_gate`] for the deviations
//! from the Python port.

pub mod dag;
pub mod gate;
pub mod pruner;
pub mod robust_gate;
pub mod sampler;
pub mod tournament;

pub use dag::{CausalDag, CausalDagSpec, CausalEdge, CausalNode, MAX_TRIAD_BUDGET};
pub use gate::{arbiter_cmp, reason_histogram, CausalGate, GateContext, GateReason, PathVerdict};
pub use pruner::{CausalPruner, EmptyActionSet, PruneResult};
pub use robust_gate::{
    choose_probe_action, score_path_into, shortest_first_order, DisturbanceModel,
    DisturbanceModelSpec, RobustChoice, RobustObjective, RobustScore, RobustSlackSelector,
    RobustSpec, ScoreScratch, Scored, StepObservation, MAX_EXTRA_SUPPORT, MAX_ROBUST_WINDOW,
    REORDER_MAX_PASSES, REORDER_TOP,
};
pub use sampler::{
    GeodesicFlowConfig, GeodesicFlowSampler, SampleSet, SampledPath, SamplerStart,
    DEFAULT_ENERGY_ALPHA, MAX_ENERGY_ALPHA,
};
pub use tournament::{
    Arbitration, RobustReport, ShardOutcome, ShardSpec, TournamentTriadPipeline, TriadLodReport,
    TriadProblem, TriadReport, MAX_TRIAD_SAMPLES, MAX_TRIAD_SHARDS,
};

use serde::Deserialize;

/// Default sample budget of `DecideMode::CausalTriad` (Python `TriadPipeline`).
pub const DEFAULT_TRIAD_SAMPLES: usize = 64;
/// Defaults of `DecideMode::TournamentTriad` (Python `TournamentTriadPipeline`).
pub const DEFAULT_TOURNAMENT_SAMPLES: usize = 256;
pub const DEFAULT_TOURNAMENT_SHARDS: usize = 4;
pub const DEFAULT_TOURNAMENT_TOP_P: usize = 2;

/// Run options of a triad decide. Unknown fields are refused.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TriadRunOptions {
    /// Actions already completed.
    #[serde(default)]
    pub done: Vec<u32>,
    /// Nominal time already spent.
    #[serde(default)]
    pub time_used: u32,
    pub n_samples: Option<usize>,
    /// Tournament mode only.
    pub shards: Option<usize>,
    /// Tournament mode only.
    pub top_p: Option<usize>,
    /// `None`: every shard draws OS entropy.
    pub seed: Option<u64>,
    /// Causal-potential coupling in `[0, MAX_ENERGY_ALPHA]`. `None`: the
    /// default [`DEFAULT_ENERGY_ALPHA`]; `Some(0.0)`: the native flow.
    pub energy_alpha: Option<f64>,
    /// Distributionally robust arbitration. `None`: nominal ranking.
    pub robust: Option<RobustSpec>,
}

/// The triad input of a `DecideRequest`. Owned so a budgeted decide can move it
/// to its deadline worker.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalTriadRequest {
    pub dag: CausalDagSpec,
    pub options: TriadRunOptions,
}
