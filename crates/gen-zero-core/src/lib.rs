//! # gen-zero-core
//!
//! Mathematical foundations, SIMD kernels, Helmert Simplex ETF, Action Frames,
//! and fundamental trait contracts for the Gen-Zero Rust Decision Engine.

pub mod error;
pub mod etf;
pub mod interner;
pub mod simd;
pub mod traits;
pub mod types;

// Re-export core items for clean ergonomics
pub use error::{CoreError, EtfError, InternerError};
pub use etf::SimplexEtfFrame;
#[allow(deprecated)]
pub use interner::ActionInterner;
pub use simd::{cosine_distance_f32, dot_product_f32, hamming_distance_u64, l2_distance_f32};
pub use traits::{
    GraphFactProvider, LatentContraction, LosslessInvertibleEncoder, SafetyEstimate,
    WorldModelDynamics,
};
pub use types::{
    ActionId, CompressedLatent, FoundationLatent, FullLatent, LatentState, LocalActionFrame,
    MicroLatent, NormalizedEntropy,
};
