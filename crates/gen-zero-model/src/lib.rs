//! # gen-zero-model
//!
//! 0-token prefill choice head, prompt sanitization, the native Qwen2.5
//! semantic scorer (`qwen`, `semantic_qwen`) that runs in process on candle,
//! and the tri-teacher stage 2 verifier of the two-stage gateway (`tri_teacher`).

pub mod choice_head;
pub mod error;
pub mod patch;
pub mod qwen;
pub mod reflex;
pub mod sanitize;
pub mod semantic_qwen;
pub mod tri_teacher;
mod tri_teacher_gate;

pub use choice_head::{
    ActionETFChoiceHead, ChoiceScores, MetricKind, DEFAULT_UNCALIBRATED_TEMPERATURE,
    METRIC_NORM_FLOOR,
};
pub use error::ModelError;
pub use patch::{ExactFixup, ReflexHeadDelta, ReflexPatch, PATCH_FORMAT};
pub use qwen::{LoraTarget, QwenConfig, QwenModel, WeightFormat};
pub use reflex::{
    ReflexDecision, ReflexHead, ReflexOperator, ReflexOperatorConfig, ReflexPlugin,
    ReflexTelemetry, CONTRACTION_EPS, LAYER_NORM_EPS,
};
pub use sanitize::{contains_raw_control_marker, sanitize_control_tokens};
pub use semantic_qwen::{
    PoolingMode, QwenModelInfo, QwenSemanticScorer, RiskAssessment, ScoreResult,
    RISK_ESCALATE_THRESHOLD, RISK_HARD_STOP_THRESHOLD,
};
pub use tri_teacher::{
    calibrated_probability, decide_projections, ProjectionHead, TeacherSlot, TeacherWeights,
    TriTeacherAdapterConfig, TriTeacherDecision, TriTeacherDeciderInfo, TriTeacherLoRAAdapter,
    TriTeacherPairDecider, TriTeacherProjector, DEFAULT_TRI_TEACHER_THRESHOLD,
    TRI_TEACHER_ADAPTER_FORMAT, TRI_TEACHER_SIGMOID_SLOPE,
};
