//! gen-zero-lod
//!
//! Multi-scale Level of Detail (Lod) Graph Fusion, Epistemic Lifecycle State Machine,
//! Spec 25 mixed-curvature product geometry (H^{d_h} x R^{d_e} x S_R^{d_s}, 128/256 stored
//! coordinates) plus the legacy 16-coordinate Lod coordinate (H^4 x S^3 x R^8),
//! and Pearl Causal Cascading Pruning with Virtual Loss Rollback.

#![allow(clippy::manual_is_multiple_of)]

pub mod error;
pub mod graph;
pub mod manifold;
pub mod node;
pub mod ppr;
pub mod semiring;
pub mod weighted;

pub use error::LodError;
pub use graph::{BufferedEdge, CsrGraph, EdgeType, LodGraph};
pub use manifold::{
    ContainmentCriteria, ContainmentScore, Digest, Epochs, FiberId, GeometryParams, Layout,
    MixedCurvatureCoord, Point, ProductGeometry, ProductManifold, Reject, Result as GeometryResult,
    Tangent, TopologyPreset, Version, MAX_PRESET_DIM,
};
pub use node::{hdc_hamming_distance_256, EpistemicStatus, LodBand, LodNode};
pub use ppr::compute_ppr_csr;
pub use semiring::{
    AssociativityReport, AssociativityViolation, FoldOutcome, Gender, RelId, RelationKey,
    RelationSemiring, ResultSet, BUDGET_EXCEEDED_REASON, DEFAULT_CHART_STEP_BUDGET,
};
pub use weighted::{
    AxiomWeightError, AxiomWeights, LogProbSemiring, TropicalSemiring, WeightedCandidate,
    WeightedFoldOutcome, WeightedSemiring,
};
