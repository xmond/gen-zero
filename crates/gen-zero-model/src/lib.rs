//! # gen-zero-model
//!
//! Permutation-equivariant set-attention masking, shared position IDs,
//! 0-token prefill choice head, prompt sanitization, and the native Qwen2.5
//! semantic scorer (`qwen`, `semantic_qwen`) that runs in process on candle.

pub mod choice_head;
pub mod error;
pub mod mask;
pub mod patch;
pub mod qwen;
pub mod reflex;
pub mod sanitize;
pub mod semantic_qwen;

pub use choice_head::{
    ActionETFChoiceHead, ChoiceScores, MetricKind, DEFAULT_UNCALIBRATED_TEMPERATURE,
    METRIC_NORM_FLOOR,
};
pub use error::ModelError;
pub use mask::{generate_shared_position_ids, BlockCausalMask, PrefixMode};
pub use patch::{ExactFixup, ReflexHeadDelta, ReflexPatch, PATCH_FORMAT};
pub use qwen::{QwenConfig, QwenModel, WeightFormat};
pub use reflex::{
    ReflexDecision, ReflexHead, ReflexOperator, ReflexOperatorConfig, ReflexPlugin,
    ReflexTelemetry, CONTRACTION_EPS, LAYER_NORM_EPS,
};
pub use sanitize::{contains_raw_control_marker, sanitize_control_tokens};
pub use semantic_qwen::{
    QwenModelInfo, QwenSemanticScorer, RiskAssessment, ScoreResult, RISK_ESCALATE_THRESHOLD,
    RISK_HARD_STOP_THRESHOLD,
};
