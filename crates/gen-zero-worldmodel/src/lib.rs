//! # gen-zero-worldmodel
//!
//! Latent dynamics, Contact Hamiltonian integrators with Strang splitting, and
//! the conformal symplectic (contact) world model built on them.

mod compression;
pub mod conformal_dynamics;
pub mod contact;
pub mod dynamics;
pub mod error;
pub mod symplectic;
pub mod symplectic_dynamics;

pub use conformal_dynamics::{ConformalWorldModelDynamics, ContactTransition};
pub use contact::{
    compress_trajectory_zstd as compress_contact_trajectory_zstd,
    decompress_trajectory_zstd as decompress_contact_trajectory_zstd, ContactIntegrator,
    ContactState,
};
pub use dynamics::{LatentDynamicsWorldModel, DONE_NORM, SAFETY_SOURCE_NORM_MARGIN};
pub use error::WorldModelError;
pub use symplectic::{
    compress_trajectory_zstd as compress_phase_trajectory_zstd,
    decompress_trajectory_zstd as decompress_phase_trajectory_zstd, PhaseState,
    SymplecticIntegrator,
};
pub use symplectic_dynamics::{PhaseTransition, SymplecticWorldModelDynamics};
