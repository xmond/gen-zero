//! gen-zero-lod error types.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum LodError {
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
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
}
