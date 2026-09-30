//! gen-zero-lod
//!
//! Multi-scale Level of Detail (Lod) Graph Fusion, Epistemic Lifecycle State Machine,
//! Spec 25 mixed-curvature product geometry (H^{d_h} x R^{d_e} x S_R^{d_s}, 128/256 stored
//! coordinates) and its 16-coordinate Lod graph chart (H^4 x R^8 x S^3) under the
//! graph's `GeometryParams`, Banach fixed-point confidence evolution over the
//! dependency edges, coarse-graining across Lod bands, atomic graph checkpoints
//! with rollback, and discrete and soft relation semirings.

#![allow(clippy::manual_is_multiple_of)]

pub mod error;
pub mod graph;
pub mod manifold;
pub mod node;
pub mod ppr;
pub mod semiring;
pub mod weighted;

pub use error::LodError;
pub use graph::{
    BufferedEdge, CsrGraph, EdgeType, FixedPointReport, FlushReport, GraphCheckpoint, LodGraph,
    PprRanking, StatusTransition, MAX_FIXED_POINT_STEPS,
};
pub use manifold::{
    ContainmentCriteria, ContainmentScore, Digest, Epochs, FiberId, GeometryParams, Layout,
    MixedCurvatureCoord, Point, ProductGeometry, ProductManifold, Reject, Result as GeometryResult,
    Tangent, TopologyPreset, Version, MAX_PRESET_DIM,
};
pub use node::{
    band_from_scale, band_scale_width, hdc_hamming_distance_256, max_chart_depth,
    normalized_depth, scale_from_depth, EpistemicStatus, LodBand, LodNode, ZoomDirection,
};
pub use ppr::{compute_ppr_csr, PprScores};
pub use semiring::{
    AssociativityReport, AssociativityViolation, FoldOutcome, Gender, RelId, RelationKey,
    RelationSemiring, ResultSet, SoftResultSet, SoftSetError, BUDGET_EXCEEDED_REASON,
    DEFAULT_CHART_STEP_BUDGET, SOFT_MASS_TOLERANCE, SOFT_TIE_TOLERANCE,
};
pub use weighted::{
    AxiomWeightError, AxiomWeights, LogProbSemiring, TropicalSemiring, WeightedCandidate,
    WeightedFoldOutcome, WeightedSemiring,
};
