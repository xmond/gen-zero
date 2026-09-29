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
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
}
