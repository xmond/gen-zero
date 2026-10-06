//! gen-zero-gate two-stage dual-track answerability gateway.
//!
//! Rust port of the reference two-stage gateway and of the decision math of
//! the tri-teacher pair decider it was trained with (training is not part of
//! this repository; the decider side lives in `gen_zero_model::tri_teacher`).
//!
//! Stage 1 is an extractive reader that has already scored the context. Its
//! `score_diff = null_score - best_span_score` decides alone outside the
//! ambiguity band `[ambiguity_low, ambiguity_high]` (both ends inclusive, as
//! in Python). Stage 1's own verdict is "no answer" iff `score_diff > 0`, the
//! fixed threshold of `NeuralDistilledReader`.
//!
//! Inside the band, stage 2 asks a [`CausalVerifier`] for the three teacher
//! projections of `sentence_1 = context` and
//! `sentence_2 = question + " " + candidate_span`. The gateway owns the math:
//! L2-normalize, cosine per teacher, weighted sum `tri_sim`, and
//! `p_same_meaning = sigmoid(slope * (tri_sim - threshold))`. The candidate is
//! kept iff `tri_sim >= threshold` (equivalently `p_same_meaning >= 0.5`),
//! exactly the rule the trained reference decider applies.
//!
//! Every fault on the stage 2 path (verifier error, non-finite value, zero
//! norm, dimension mismatch) is a [`GateError`]. Nothing falls back to the
//! stage 1 verdict.

use crate::error::GateError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Below this norm a projection is treated as zero: its direction, and so
/// its cosine, is undefined.
pub const MIN_PROJECTION_NORM: f32 = 1e-12;

/// Which stage 2 verifier the gateway expects. Only the tri-teacher verifier
/// has a Rust implementation; the Python `single_teacher` path is not ported
/// and is rejected rather than silently mapped to something else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierType {
    TriTeacher,
}

impl VerifierType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TriTeacher => "tri_teacher",
        }
    }
}

impl FromStr for VerifierType {
    type Err = GateError;

    fn from_str(s: &str) -> Result<Self, GateError> {
        match s {
            "tri_teacher" => Ok(Self::TriTeacher),
            other => Err(GateError::InvalidConfig(format!(
                "verifier_type {other:?} is not supported; only \"tri_teacher\" has a Rust verifier"
            ))),
        }
    }
}

/// Mixing weights of the three teacher cosines. Defaults are the trained
/// adapter's `weights` metadata (`adapter_paws_tri.safetensors`:
/// 405b 0.5, q72b 0.3, llama70b 0.2), also hardcoded in Python `decide`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TeacherWeights {
    pub w_405b: f32,
    pub w_q72b: f32,
    pub w_llama70b: f32,
}

impl Default for TeacherWeights {
    fn default() -> Self {
        Self {
            w_405b: 0.5,
            w_q72b: 0.3,
            w_llama70b: 0.2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoStageConfig {
    pub ambiguity_low: f32,
    pub ambiguity_high: f32,
    /// Decision threshold on `tri_sim`, and the centre of the logistic that
    /// maps `tri_sim` to `p_same_meaning`.
    pub tri_teacher_threshold: f32,
    pub verifier_type: VerifierType,
    pub teacher_weights: TeacherWeights,
    /// Logistic slope of `p_same_meaning` (Python `decide` uses 15.0).
    pub logistic_slope: f32,
}

impl Default for TwoStageConfig {
    fn default() -> Self {
        Self {
            ambiguity_low: -1.5,
            ambiguity_high: 0.5,
            tri_teacher_threshold: 0.91,
            verifier_type: VerifierType::TriTeacher,
            teacher_weights: TeacherWeights::default(),
            logistic_slope: 15.0,
        }
    }
}

impl TwoStageConfig {
    pub fn validate(&self) -> Result<(), GateError> {
        let bad = |msg: String| Err(GateError::InvalidConfig(msg));
        if !self.ambiguity_low.is_finite() || !self.ambiguity_high.is_finite() {
            return bad(format!(
                "ambiguity band [{}, {}] must be finite",
                self.ambiguity_low, self.ambiguity_high
            ));
        }
        if self.ambiguity_low > self.ambiguity_high {
            return bad(format!(
                "ambiguity_low ({}) must be <= ambiguity_high ({})",
                self.ambiguity_low, self.ambiguity_high
            ));
        }
        // A weighted sum of cosines lies in [-1, 1]; a threshold outside it
        // would make stage 2 always accept or always reject.
        if !(self.tri_teacher_threshold > -1.0 && self.tri_teacher_threshold < 1.0) {
            return bad(format!(
                "tri_teacher_threshold ({}) must lie in (-1, 1), the range of tri_sim",
                self.tri_teacher_threshold
            ));
        }
        if !(self.logistic_slope.is_finite() && self.logistic_slope > 0.0) {
            return bad(format!(
                "logistic_slope ({}) must be finite and > 0",
                self.logistic_slope
            ));
        }
        let w = self.teacher_weights;
        let ws = [w.w_405b, w.w_q72b, w.w_llama70b];
        if ws.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return bad(format!("teacher weights {ws:?} must be finite and >= 0"));
        }
        let sum: f32 = ws.iter().sum();
        if (sum - 1.0).abs() > 1e-4 {
            return bad(format!("teacher weights {ws:?} must sum to 1, got {sum}"));
        }
        Ok(())
    }
}

/// One teacher head's projections of `sentence_1` and `sentence_2`.
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherProjectionPair {
    pub z1: Vec<f32>,
    pub z2: Vec<f32>,
}

/// Raw (un-normalized) projections from the three teacher heads.
#[derive(Clone, Debug, PartialEq)]
pub struct TriTeacherProjections {
    pub proj_405b: TeacherProjectionPair,
    pub proj_q72b: TeacherProjectionPair,
    pub proj_llama70b: TeacherProjectionPair,
}

/// Stage 2 encoder: maps a sentence pair to three teacher projections.
/// The real implementation is `gen_zero_model::TriTeacherPairDecider`
/// (Qwen2.5-0.5B + merged LoRA + three LayerNorm/Linear heads).
pub trait CausalVerifier {
    /// Stable identifier recorded in the evidence (backbone, adapter hash).
    fn verifier_id(&self) -> String;

    fn project_pair(
        &self,
        sentence_1: &str,
        sentence_2: &str,
    ) -> Result<TriTeacherProjections, GateError>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoStageGateEvidence {
    pub fast_pass: bool,
    pub stage2_triggered: bool,
    /// Final verdict differs from stage 1's (`score_diff > 0` => no answer).
    pub decision_flipped: bool,
    pub score_diff: f32,
    pub tri_sim: Option<f32>,
    pub sim_405b: Option<f32>,
    pub sim_q72b: Option<f32>,
    pub sim_llama70b: Option<f32>,
    pub p_same_meaning: Option<f32>,
    pub final_answer: Option<String>,
    pub is_answerable: bool,
    pub verifier_type: VerifierType,
    pub verifier_id: Option<String>,
    pub sentence_1: Option<String>,
    pub sentence_2: Option<String>,
}

/// Durable sink for refusal/ambiguous-band events. The gateway calls this
/// every time stage 2 triggers (ambiguity band) or the final verdict flips
/// stage 1's, so a later arbitration pass can review the decision.
///
/// An `Err` here means the event was NOT durably recorded; the gateway
/// surfaces this as a hard error rather than silently losing training data
/// (fail-closed).
pub trait RefusalTraceSink {
    fn record_refusal(&self, event: RefusalTraceEvent) -> Result<(), String>;
}

/// One refusal/ambiguous-band event, handed to a [`RefusalTraceSink`] for
/// durable storage and later arbitration.
#[derive(Clone, Debug, PartialEq)]
pub struct RefusalTraceEvent {
    pub context: String,
    pub question: String,
    pub candidate: String,
    pub best_span_score: f32,
    pub null_score: f32,
    pub score_diff: f32,
    pub stage2_triggered: bool,
    pub decision_flipped: bool,
    pub is_answerable: bool,
    pub tri_sim: Option<f32>,
    pub p_same_meaning: Option<f32>,
    pub verifier_id: Option<String>,
    pub verifier_type: VerifierType,
}

#[derive(Clone)]
pub struct TwoStageDualTrackGateway {
    config: TwoStageConfig,
    refusal_sink: Option<std::sync::Arc<dyn RefusalTraceSink + Send + Sync>>,
}

impl std::fmt::Debug for TwoStageDualTrackGateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoStageDualTrackGateway")
            .field("config", &self.config)
            .field("has_refusal_sink", &self.refusal_sink.is_some())
            .finish()
    }
}

impl TwoStageDualTrackGateway {
    pub fn new(config: TwoStageConfig) -> Result<Self, GateError> {
        config.validate()?;
        Ok(Self {
            config,
            refusal_sink: None,
        })
    }

    /// Attach a durable sink for refusal/ambiguous-band events. Consuming
    /// builder so existing `new(config)` callers are unaffected.
    pub fn with_refusal_sink(
        mut self,
        sink: std::sync::Arc<dyn RefusalTraceSink + Send + Sync>,
    ) -> Self {
        self.refusal_sink = Some(sink);
        self
    }

    pub fn config(&self) -> &TwoStageConfig {
        &self.config
    }

    pub fn in_ambiguity_band(&self, score_diff: f32) -> bool {
        self.config.ambiguity_low <= score_diff && score_diff <= self.config.ambiguity_high
    }

    /// Decide answerability from stage 1's scores, calling `verifier` only
    /// inside the ambiguity band.
    ///
    /// An empty `candidate_span` is legal only when stage 1 confidently says
    /// "no answer" (fast pass with `score_diff > ambiguity_high`). Anywhere
    /// else there is nothing to answer with or to verify, so it is an error.
    pub fn decide_with_scores(
        &self,
        context: &str,
        question: &str,
        candidate_span: &str,
        best_span_score: f32,
        null_score: f32,
        verifier: &dyn CausalVerifier,
    ) -> Result<TwoStageGateEvidence, GateError> {
        self.decide_with_optional_verifier(
            context,
            question,
            candidate_span,
            best_span_score,
            null_score,
            Some(verifier),
        )
    }

    /// Run the same gate when stage 2 may be unavailable. An ambiguous score
    /// without a configured verifier is an explicit fail-closed error.
    pub fn decide_with_optional_verifier(
        &self,
        context: &str,
        question: &str,
        candidate_span: &str,
        best_span_score: f32,
        null_score: f32,
        verifier: Option<&dyn CausalVerifier>,
    ) -> Result<TwoStageGateEvidence, GateError> {
        if !best_span_score.is_finite() || !null_score.is_finite() {
            return Err(GateError::InvalidInput(format!(
                "stage 1 scores must be finite (best_span_score={best_span_score}, null_score={null_score})"
            )));
        }
        let score_diff = null_score - best_span_score;
        if !score_diff.is_finite() {
            return Err(GateError::InvalidInput(format!(
                "score_diff overflowed ({null_score} - {best_span_score})"
            )));
        }
        let stage1_answerable = score_diff <= 0.0;
        let candidate = candidate_span.trim();

        if !self.in_ambiguity_band(score_diff) {
            if stage1_answerable && candidate.is_empty() {
                return Err(GateError::InvalidInput(format!(
                    "stage 1 says answerable (score_diff={score_diff}) but candidate_span is empty"
                )));
            }
            return Ok(TwoStageGateEvidence {
                fast_pass: true,
                stage2_triggered: false,
                decision_flipped: false,
                score_diff,
                tri_sim: None,
                sim_405b: None,
                sim_q72b: None,
                sim_llama70b: None,
                p_same_meaning: None,
                final_answer: stage1_answerable.then(|| candidate.to_string()),
                is_answerable: stage1_answerable,
                verifier_type: self.config.verifier_type,
                verifier_id: None,
                sentence_1: None,
                sentence_2: None,
            });
        }

        if candidate.is_empty() {
            return Err(GateError::InvalidInput(format!(
                "score_diff={score_diff} is in the ambiguity band but candidate_span is empty; \
                 stage 2 has nothing to verify"
            )));
        }
        if context.trim().is_empty() || question.trim().is_empty() {
            return Err(GateError::InvalidInput(
                "stage 2 needs a non-empty context and question".into(),
            ));
        }

        let verifier = verifier.ok_or_else(|| {
            GateError::VerifierFailure(
                "tri-teacher adapter and native Qwen model are not configured".into(),
            )
        })?;
        let sentence_1 = context.to_string();
        let sentence_2 = format!("{} {}", question.trim(), candidate);
        let proj = verifier.project_pair(&sentence_1, &sentence_2)?;
        let sim_405b = cosine("405b", &proj.proj_405b)?;
        let sim_q72b = cosine("q72b", &proj.proj_q72b)?;
        let sim_llama70b = cosine("llama70b", &proj.proj_llama70b)?;
        let w = self.config.teacher_weights;
        let tri_sim = w.w_405b * sim_405b + w.w_q72b * sim_q72b + w.w_llama70b * sim_llama70b;
        let threshold = self.config.tri_teacher_threshold;
        let p_same_meaning =
            1.0 / (1.0 + (-self.config.logistic_slope * (tri_sim - threshold)).exp());
        if !tri_sim.is_finite() || !p_same_meaning.is_finite() {
            return Err(GateError::NumericalFault(format!(
                "tri_sim={tri_sim}, p_same_meaning={p_same_meaning}"
            )));
        }

        let is_answerable = tri_sim >= threshold;
        let evidence = TwoStageGateEvidence {
            fast_pass: false,
            stage2_triggered: true,
            decision_flipped: is_answerable != stage1_answerable,
            score_diff,
            tri_sim: Some(tri_sim),
            sim_405b: Some(sim_405b),
            sim_q72b: Some(sim_q72b),
            sim_llama70b: Some(sim_llama70b),
            p_same_meaning: Some(p_same_meaning),
            final_answer: is_answerable.then(|| candidate.to_string()),
            is_answerable,
            verifier_type: self.config.verifier_type,
            verifier_id: Some(verifier.verifier_id()),
            sentence_1: Some(sentence_1),
            sentence_2: Some(sentence_2),
        };

        if evidence.stage2_triggered || evidence.decision_flipped {
            if let Some(sink) = &self.refusal_sink {
                let event = RefusalTraceEvent {
                    context: context.to_string(),
                    question: question.to_string(),
                    candidate: candidate.to_string(),
                    best_span_score,
                    null_score,
                    score_diff: evidence.score_diff,
                    stage2_triggered: evidence.stage2_triggered,
                    decision_flipped: evidence.decision_flipped,
                    is_answerable: evidence.is_answerable,
                    tri_sim: evidence.tri_sim,
                    p_same_meaning: evidence.p_same_meaning,
                    verifier_id: evidence.verifier_id.clone(),
                    verifier_type: evidence.verifier_type,
                };
                if let Err(msg) = sink.record_refusal(event) {
                    return Err(GateError::RefusalSinkFailure(msg));
                }
            }
        }

        Ok(evidence)
    }
}

/// Cosine of one teacher pair, computed in f64. Fails closed on empty or
/// mismatched vectors, non-finite components and zero norms.
fn cosine(teacher: &str, pair: &TeacherProjectionPair) -> Result<f32, GateError> {
    let (a, b) = (&pair.z1, &pair.z2);
    if a.is_empty() || a.len() != b.len() {
        return Err(GateError::NumericalFault(format!(
            "{teacher} projections have dims {} and {}",
            a.len(),
            b.len()
        )));
    }
    if a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err(GateError::NumericalFault(format!(
            "{teacher} projection has a non-finite component"
        )));
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let (na, nb) = (na.sqrt(), nb.sqrt());
    if na < f64::from(MIN_PROJECTION_NORM) || nb < f64::from(MIN_PROJECTION_NORM) {
        return Err(GateError::NumericalFault(format!(
            "{teacher} projection has zero norm (|z1|={na}, |z2|={nb}); cosine undefined"
        )));
    }
    Ok((dot / (na * nb)).clamp(-1.0, 1.0) as f32)
}
