//! gen-zero-lod error types.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum LodError {
    #[error("Graph node capacity exceeded: {current} existing + {additional} requested > {max}")]
    GraphCapacityExceeded {
        current: usize,
        additional: usize,
        max: usize,
    },
    #[error("Graph persistence refused: {0}")]
    Persistence(String),
    #[error("Node ID {0} not found in LodGraph")]
    NodeNotFound(u32),
    #[error("Hyperbolic boundary violation: norm squared {norm_sq:.6} >= boundary floor")]
    HyperbolicBoundaryViolation { norm_sq: f32 },
    #[error(
        "Spine breathe operation out of bounds: cannot breathe {direction} from level {level:?}"
    )]
    SpineBreatheOutOfBounds { direction: &'static str, level: u8 },
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),
    #[error("Busemann test needs a defined radial direction: {0} has hyperbolic norm ~0")]
    DegenerateRadialDirection(&'static str),
    #[error("Geometry rejected: {0}")]
    Geometry(#[from] crate::manifold::Reject),
    #[error("Invalid node: {0}")]
    InvalidNode(String),
    #[error("Entity {0} already has a node in this LodGraph")]
    DuplicateEntity(u64),
    #[error("Entity {0} has no node in this LodGraph")]
    EntityNotFound(u64),
    #[error("Invalid edge: {0}")]
    InvalidEdge(String),
    #[error("Invalid graph query: {0}")]
    InvalidQuery(String),
    #[error("CSR snapshot failed validation, old snapshot kept: {0}")]
    CsrInvariant(String),
    #[error("A rollback ran while this flush was building; nothing was committed")]
    FlushConflict,
    #[error("Checkpoint rejected: {0}")]
    CheckpointRejected(String),
    #[error(
        "Confidence fixed point not reached: residual {residual:e} >= tolerance {tolerance:e} \
         after {iterations} step(s), bound {k_max}; nothing was committed"
    )]
    FixedPointDiverged {
        iterations: usize,
        k_max: usize,
        residual: f64,
        tolerance: f64,
    },
    #[error(
        "Confidence map is not a contraction on a cycle of {block_size} node(s): Lipschitz \
         bound {contraction} >= 1 at beta {beta}, gamma {gamma} (beta < 1 / (1 + gamma) \
         guarantees one); nothing was committed"
    )]
    FixedPointNotContractive {
        block_size: usize,
        contraction: f64,
        beta: f64,
        gamma: f64,
    },
    #[error(
        "Confidence map on a cycle of {block_size} node(s) contracts too slowly: Lipschitz \
         bound {contraction} may need {k_max} step(s) to reach the tolerance, over the \
         budget of {max_steps}; refused"
    )]
    FixedPointTooSlow {
        block_size: usize,
        contraction: f64,
        k_max: usize,
        max_steps: usize,
    },
    #[error("Empty input: {0}")]
    EmptyInput(String),
    #[error("Payload is {len} bytes; one chunk holds at most {max} bytes")]
    PayloadTooLarge { len: usize, max: usize },
    #[error("Invalid payload: {0}")]
    InvalidPayload(String),
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
}
