//! gen-zero-worldmodel error types.

use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum WorldModelError {
    #[error("Simulation diverged or reached non-finite values (NaN / Inf)")]
    NumericalDivergence,
    #[error("Horizon must be at least 1, got {0}")]
    InvalidHorizon(usize),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Action out of bounds: {0}")]
    InvalidAction(u32),
    #[error("Compression error: {0}")]
    Compression(String),
    #[error("Core error: {0}")]
    Core(#[from] gen_zero_core::CoreError),
}

/// Lets a world model satisfy `WorldModelDynamics<Error = CoreError>`, the bound the
/// planner requires. The orphan rule allows this impl here because `WorldModelError`
/// is local; it cannot live in gen-zero-core, which gen-zero-worldmodel depends on.
impl From<WorldModelError> for gen_zero_core::CoreError {
    fn from(e: WorldModelError) -> Self {
        use gen_zero_core::CoreError;
        match e {
            WorldModelError::Core(inner) => inner,
            WorldModelError::DimensionMismatch { expected, actual } => {
                CoreError::DimensionMismatch { expected, actual }
            }
            WorldModelError::NumericalDivergence => {
                CoreError::NumericalInstability(WorldModelError::NumericalDivergence.to_string())
            }
            other @ (WorldModelError::InvalidHorizon(_)
            | WorldModelError::InvalidAction(_)
            | WorldModelError::Compression(_)) => CoreError::WorldModel(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_zero_core::CoreError;

    #[test]
    fn world_model_error_maps_to_core_error_without_losing_the_kind() {
        assert!(matches!(
            CoreError::from(WorldModelError::NumericalDivergence),
            CoreError::NumericalInstability(_)
        ));
        assert_eq!(
            CoreError::from(WorldModelError::DimensionMismatch {
                expected: 3,
                actual: 2
            }),
            CoreError::DimensionMismatch {
                expected: 3,
                actual: 2
            }
        );
        assert_eq!(
            CoreError::from(WorldModelError::InvalidAction(7)),
            CoreError::WorldModel("Action out of bounds: 7".to_string())
        );
        assert_eq!(
            CoreError::from(WorldModelError::InvalidHorizon(0)),
            CoreError::WorldModel("Horizon must be at least 1, got 0".to_string())
        );
        assert!(matches!(
            CoreError::from(WorldModelError::Compression("bad frame".into())),
            CoreError::WorldModel(message) if message.contains("bad frame")
        ));
        let inner = CoreError::NumericalInstability("x".to_string());
        assert_eq!(CoreError::from(WorldModelError::Core(inner.clone())), inner);
    }
}
