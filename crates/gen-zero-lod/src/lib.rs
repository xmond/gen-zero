//! gen-zero-lod
//!
//! Multi-scale Level of Detail (Lod) Graph Fusion, Epistemic Lifecycle State Machine,
//! Spec 25 mixed-curvature product geometry (H^{d_h} x R^{d_e} x S_R^{d_s}, 128/256 stored
//! coordinates) and its 16-coordinate Lod graph chart (H^4 x R^8 x S^3) under the
//! graph's `GeometryParams`, Banach fixed-point confidence evolution over the
//! dependency edges, coarse-graining across Lod bands, atomic graph checkpoints
//! with rollback, discrete and soft relation semirings, and retrieval: a
//! lexical text-to-chart projector, a dense-vector-to-chart projector for
//! embeddings an external model made, payload-carrying nodes with aliases, and
//! a three-stage hybrid search (HDC prefilter, geodesic rerank, PPR diffusion),
//! and the causal AND/OR DAG laid out on the four bands with the closure
//! potential the planner's causal triad steers by ([`causal_lod`]), and the
//! causal operator contract binding nodes to hard tools and soft operators
//! behind a precondition, nonce and postcondition gate ([`operator`]).

#![allow(clippy::manual_is_multiple_of)]

pub mod builtin_operator;
pub mod causal_lod;
pub mod error;
pub mod graph;
pub mod manifold;
pub mod node;
pub mod operator;
pub mod ppr;
pub mod projection;
pub mod semiring;
pub mod weighted;

pub use causal_lod::{
    CausalCluster, CausalLod, CausalLodContext, CausalLodError, CausalLodSummary, CausalMaskSource,
    CausalMilestone, CausalPotential, ClusterKind, MilestoneKind, MAX_CAUSAL_ATOMS,
    SYNTHETIC_ENTITY_BASE,
};
pub use error::LodError;
pub use graph::{
    AdaptedBlock, AnchorMatch, BufferedEdge, ConflictDirection, ConflictEdge, CsrGraph,
    DiffusionQuality, EdgeType, FixedPointReport, FlushReport, GraphCheckpoint, GraphState,
    HybridRagResult, LodGraph, PprRanking, RagDiffusion, RagHit, ReflectionReport,
    ReflectionRevocation, StatusTransition, ADMISSION_BETA, ADMISSION_GAMMA, ALIAS_LINK_WEIGHT,
    DEFAULT_FALSIFICATION_GAIN, DENSE_RERANK_POOL, HYBRID_PPR_TOLERANCE, MAX_ALIAS_HOLDERS,
    MAX_FIXED_POINT_STEPS, MAX_GRAPH_NODES,
};
pub use manifold::{
    ContainmentCriteria, ContainmentScore, Digest, Epochs, FiberId, GeometryParams, Layout,
    MixedCurvatureCoord, Point, ProductGeometry, ProductManifold, Reject, Result as GeometryResult,
    Tangent, TopologyPreset, Version, MAX_PRESET_DIM,
};
pub use node::{
    band_from_scale, band_scale_width, hdc_hamming_distance_256, max_chart_depth, normalized_depth,
    payload_digest, scale_from_depth, ChartAnchor, EpistemicStatus, LodBand, LodNode, Placement,
    ZoomDirection, MAX_ALIASES, MAX_ALIAS_BYTES, MAX_EMBEDDING_DIM, MAX_PAYLOAD_BYTES,
    MAX_SOURCE_URI_BYTES, MIN_EMBEDDING_DIM,
};
pub use operator::{
    CausalOperator, OperatorExecution, OperatorInput, OperatorKind, OperatorOutput,
    OperatorSignature, PostconditionReport, MAX_OPERATOR_NAME_BYTES, MAX_OPERATOR_SPACE_BYTES,
    MAX_OPERATOR_VERSION_BYTES, MAX_REGISTERED_OPERATORS, MAX_SPENT_NONCES,
};
pub use ppr::{compute_ppr_csr, PprScores};
pub use projection::{
    TextEmbeddingProjector, DENSE_PROJECTOR_VERSION, HDC_BITS, PROJECTOR_VERSION,
};
pub use semiring::{
    AssociativityReport, AssociativityViolation, FoldOutcome, Gender, RelId, RelationKey,
    RelationSemiring, ResultSet, SoftResultSet, SoftSetError, BUDGET_EXCEEDED_REASON,
    DEFAULT_CHART_STEP_BUDGET, SOFT_MASS_TOLERANCE, SOFT_TIE_TOLERANCE,
};
pub use weighted::{
    AxiomWeightError, AxiomWeights, LogProbSemiring, TropicalSemiring, WeightedCandidate,
    WeightedFoldOutcome, WeightedSemiring,
};
