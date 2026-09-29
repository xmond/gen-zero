//! # gen-zero-model
//!
//! Permutation-equivariant set-attention masking, shared position IDs,
//! 0-token prefill choice head, and prompt sanitization.

pub mod choice_head;
pub mod error;
pub mod mask;
pub mod sanitize;

pub use choice_head::{
    ActionETFChoiceHead, ChoiceScores, MetricKind, DEFAULT_UNCALIBRATED_TEMPERATURE,
    METRIC_NORM_FLOOR,
};
pub use error::ModelError;
pub use mask::{generate_shared_position_ids, BlockCausalMask, PrefixMode};
pub use sanitize::{contains_raw_control_marker, sanitize_control_tokens};
