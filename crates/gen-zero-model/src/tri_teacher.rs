//! Tri-teacher LoRA adapter and pair decider: the Rust inference side of the
//! Python `TriTeacherCausalLoRAQwen` / `TriTeacherPairDecider` reference.
//! Training is not part of this repository; only the adapter file format below
//! crosses the boundary.
//!
//! The adapter (`gen_zero.tri_teacher_lora.v1` safetensors) holds:
//! * `backbone.layers.<i>.self_attn.<target>.lora_{A,B}`: attention LoRA,
//!   `A: [rank, in]`, `B: [out, rank]`, scaled by `alpha / rank`;
//! * `proj_<slot>.0.{weight,bias}` (LayerNorm over the student width) and
//!   `proj_<slot>.1.{weight,bias}` (Linear to the teacher width), one head per
//!   teacher slot `405b`, `q72b`, `llama70b`;
//! * `task_head.*` and `teacher_{mean,std}_<slot>`: training-only tensors.
//!   They are shape- and finiteness-checked so a corrupt file is refused, then
//!   dropped: the decider never reads them, exactly like the Python decider.
//!
//! Rank, alpha, widths, targets and teacher weights come from the file's
//! `adapter_config` metadata, never from constants here.
//!
//! The decision for a pair with pooled student states `u`, `v`:
//!
//! ```text
//! sim_t    = <normalize(head_t(u)), normalize(head_t(v))>      per teacher t
//! tri_sim  = sum_t w_t * sim_t                                 (weights as given)
//! p        = 1 / (1 + exp(-15 * (tri_sim - threshold)))
//! same     = tri_sim >= threshold
//! ```
//!
//! The student state is the final-norm hidden state of the last token of the
//! sentence, from Qwen2.5-0.5B with the LoRA deltas merged into its weights.

use crate::error::ModelError;
use crate::qwen::{candle_err, LoraTarget, QwenModel, WeightFormat};
use candle_core::{DType, Device, Tensor};
use gen_zero_gate::{TeacherProjectionPair, TriTeacherProjections};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// `format` metadata value of a tri-teacher adapter.
pub const TRI_TEACHER_ADAPTER_FORMAT: &str = "gen_zero.tri_teacher_lora.v1";
/// Default decision threshold on `tri_sim` (the Python decider's default).
pub const DEFAULT_TRI_TEACHER_THRESHOLD: f64 = 0.91;
/// Slope of the calibration sigmoid around the threshold.
pub const TRI_TEACHER_SIGMOID_SLOPE: f64 = 15.0;
/// `torch.nn.LayerNorm` default epsilon, used by every projection head.
pub const PROJECTION_LAYER_NORM_EPS: f32 = 1e-5;
/// Below this L2 norm a projection has no direction and its cosine is
/// undefined. `F.normalize` would silently return zeros (cosine 0); this
/// decider refuses instead.
pub const MIN_PROJECTION_NORM: f64 = 1e-12;

/// Largest safetensors JSON header accepted (the real one is about 20 KB).
const MAX_HEADER_BYTES: u64 = 16 << 20;

/// One of the three teachers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum TeacherSlot {
    /// Llama-3.1-405B, 16384-d.
    #[serde(rename = "405b")]
    Llama405b,
    /// Qwen2.5-72B, 8192-d.
    #[serde(rename = "q72b")]
    Qwen72b,
    /// Llama-3.1-70B, 8192-d.
    #[serde(rename = "llama70b")]
    Llama70b,
}

impl TeacherSlot {
    pub const ALL: [TeacherSlot; 3] = [Self::Llama405b, Self::Qwen72b, Self::Llama70b];

    /// The slot name used in the adapter's keys and config.
    pub fn key(self) -> &'static str {
        match self {
            Self::Llama405b => "405b",
            Self::Qwen72b => "q72b",
            Self::Llama70b => "llama70b",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

fn artifact(msg: impl Into<String>) -> ModelError {
    ModelError::TriTeacherArtifact(msg.into())
}

fn input(msg: impl Into<String>) -> ModelError {
    ModelError::TriTeacherInput(msg.into())
}

/// Loss weights of the three teachers, used as given (never renormalized).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TeacherWeights {
    #[serde(rename = "405b")]
    pub w_405b: f64,
    #[serde(rename = "q72b")]
    pub w_q72b: f64,
    #[serde(rename = "llama70b")]
    pub w_llama70b: f64,
}

impl TeacherWeights {
    pub fn new(w_405b: f64, w_q72b: f64, w_llama70b: f64) -> Result<Self, ModelError> {
        let w = Self {
            w_405b,
            w_q72b,
            w_llama70b,
        };
        let all = w.as_array();
        if all.iter().any(|x| !x.is_finite() || *x < 0.0) || all.iter().sum::<f64>() <= 0.0 {
            return Err(artifact(format!(
                "teacher weights must be finite, >= 0 and not all zero: {all:?}"
            )));
        }
        Ok(w)
    }

    /// Weights in [`TeacherSlot::ALL`] order.
    pub fn as_array(&self) -> [f64; 3] {
        [self.w_405b, self.w_q72b, self.w_llama70b]
    }
}

/// The `adapter_config` metadata the Python trainer writes
/// (`TriTeacherAdapterConfig`). Unknown fields are refused: a format change
/// must be handled here, not ignored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriTeacherAdapterConfig {
    pub model_id_or_path: String,
    pub rank: usize,
    pub alpha: f64,
    pub student_dim: usize,
    pub teacher_dims: BTreeMap<String, usize>,
    pub weights: BTreeMap<String, f64>,
    pub lam: f64,
    pub num_classes: usize,
    pub tau: f64,
    pub target_modules: Vec<String>,
    #[serde(default)]
    pub variants: BTreeMap<String, String>,
}

impl TriTeacherAdapterConfig {
    fn slot_map<T: Copy>(map: &BTreeMap<String, T>, what: &str) -> Result<[T; 3], ModelError> {
        let want: BTreeSet<&str> = TeacherSlot::ALL.iter().map(|s| s.key()).collect();
        let got: BTreeSet<&str> = map.keys().map(String::as_str).collect();
        if want != got {
            return Err(artifact(format!(
                "{what} need exactly the slots {want:?}, got {got:?}"
            )));
        }
        Ok(TeacherSlot::ALL.map(|s| map[s.key()]))
    }

    /// Teacher widths in [`TeacherSlot::ALL`] order.
    pub fn teacher_dims(&self) -> Result<[usize; 3], ModelError> {
        let dims = Self::slot_map(&self.teacher_dims, "teacher_dims")?;
        if dims.contains(&0) {
            return Err(artifact(format!("teacher_dims must be positive: {dims:?}")));
        }
        Ok(dims)
    }

    pub fn teacher_weights(&self) -> Result<TeacherWeights, ModelError> {
        let [a, b, c] = Self::slot_map(&self.weights, "weights")?;
        TeacherWeights::new(a, b, c)
    }

    /// `alpha / rank`, the factor applied to `B @ A`.
    pub fn lora_scaling(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    /// The attention projections the adapter adapts. Only those the native
    /// Qwen forward can merge into are accepted.
    pub fn lora_targets(&self) -> Result<Vec<LoraTarget>, ModelError> {
        if self.target_modules.is_empty() {
            return Err(artifact("target_modules is empty"));
        }
        let mut out = Vec::new();
        for name in &self.target_modules {
            let t = lora_target(name)?;
            if out.contains(&t) {
                return Err(artifact(format!("target_modules repeats {name:?}")));
            }
            out.push(t);
        }
        Ok(out)
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.rank == 0 || self.student_dim == 0 {
            return Err(artifact(format!(
                "rank ({}) and student_dim ({}) must be positive",
                self.rank, self.student_dim
            )));
        }
        if !(self.alpha.is_finite() && self.alpha > 0.0) {
            return Err(artifact(format!(
                "alpha must be finite and > 0, got {}",
                self.alpha
            )));
        }
        self.teacher_dims()?;
        self.teacher_weights()?;
        self.lora_targets()?;
        Ok(())
    }
}

fn lora_target(name: &str) -> Result<LoraTarget, ModelError> {
    match name {
        "q_proj" => Ok(LoraTarget::QProj),
        "v_proj" => Ok(LoraTarget::VProj),
        other => Err(artifact(format!(
            "LoRA target {other:?} is not supported (native Qwen merges q_proj and v_proj)"
        ))),
    }
}

fn target_name(t: LoraTarget) -> &'static str {
    match t {
        LoraTarget::QProj => "q_proj",
        LoraTarget::VProj => "v_proj",
    }
}

fn finite_values(t: &Tensor, name: &str) -> Result<Vec<f32>, ModelError> {
    let v = t
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| artifact(format!("{name}: {e}")))?;
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(artifact(format!("{name} holds a non-finite value at {i}")));
    }
    Ok(v)
}

/// `LayerNorm(in) -> Linear(in, out)` with bias: one teacher head.
#[derive(Clone, Debug)]
pub struct ProjectionHead {
    ln_weight: Tensor,
    ln_bias: Tensor,
    /// `[out, in]`, as stored by `nn.Linear`.
    weight: Tensor,
    bias: Tensor,
    in_dim: usize,
    out_dim: usize,
}

impl ProjectionHead {
    /// Build a head from row-major values. `weight` is `[out_dim, in_dim]`.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        ln_weight: Vec<f32>,
        ln_bias: Vec<f32>,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, ModelError> {
        let dev = Device::Cpu;
        let t = |v: Vec<f32>, shape: &[usize], name: &str| {
            let n: usize = shape.iter().product();
            if v.len() != n {
                return Err(artifact(format!(
                    "{name}: {} values for shape {shape:?}",
                    v.len()
                )));
            }
            Tensor::from_vec(v, shape, &dev).map_err(|e| artifact(format!("{name}: {e}")))
        };
        Self::from_tensors(
            t(ln_weight, &[in_dim], "ln_weight")?,
            t(ln_bias, &[in_dim], "ln_bias")?,
            t(weight, &[out_dim, in_dim], "weight")?,
            t(bias, &[out_dim], "bias")?,
            "projection head",
        )
    }

    fn from_tensors(
        ln_weight: Tensor,
        ln_bias: Tensor,
        weight: Tensor,
        bias: Tensor,
        name: &str,
    ) -> Result<Self, ModelError> {
        let (out_dim, in_dim) = weight
            .dims2()
            .map_err(|e| artifact(format!("{name} linear weight: {e}")))?;
        if in_dim == 0 || out_dim == 0 {
            return Err(artifact(format!(
                "{name}: empty linear weight [{out_dim}, {in_dim}]"
            )));
        }
        for (t, n, what) in [
            (&ln_weight, in_dim, "LayerNorm weight"),
            (&ln_bias, in_dim, "LayerNorm bias"),
            (&bias, out_dim, "linear bias"),
        ] {
            if t.dims() != [n] {
                return Err(artifact(format!(
                    "{name} {what} has shape {:?}, expected [{n}]",
                    t.dims()
                )));
            }
        }
        for (t, what) in [
            (&ln_weight, "LayerNorm weight"),
            (&ln_bias, "LayerNorm bias"),
            (&weight, "linear weight"),
            (&bias, "linear bias"),
        ] {
            if t.dtype() != DType::F32 {
                return Err(artifact(format!(
                    "{name} {what} is {:?}, expected F32",
                    t.dtype()
                )));
            }
            finite_values(t, &format!("{name} {what}"))?;
        }
        Ok(Self {
            ln_weight,
            ln_bias,
            weight,
            bias,
            in_dim,
            out_dim,
        })
    }

    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Project rows `[n, in_dim]` to `[n, out_dim]`.
    fn forward(&self, x: &Tensor) -> Result<Vec<Vec<f32>>, ModelError> {
        let err = candle_err("tri-teacher projection");
        candle_nn::ops::layer_norm(x, &self.ln_weight, &self.ln_bias, PROJECTION_LAYER_NORM_EPS)
            .and_then(|h| h.matmul(&self.weight.t()?))
            .and_then(|h| h.broadcast_add(&self.bias))
            .and_then(|h| h.to_vec2::<f32>())
            .map_err(err)
    }
}

/// One teacher's raw projections inside the gate's projection record.
pub fn slot_projections(p: &TriTeacherProjections, slot: TeacherSlot) -> &TeacherProjectionPair {
    match slot {
        TeacherSlot::Llama405b => &p.proj_405b,
        TeacherSlot::Qwen72b => &p.proj_q72b,
        TeacherSlot::Llama70b => &p.proj_llama70b,
    }
}

/// The three heads plus their weights: everything needed to decide from
/// student states, with no base model.
#[derive(Clone, Debug)]
pub struct TriTeacherProjector {
    student_dim: usize,
    heads: [ProjectionHead; 3],
    weights: TeacherWeights,
}

impl TriTeacherProjector {
    /// `heads` in [`TeacherSlot::ALL`] order; every head must read `student_dim`.
    pub fn new(
        student_dim: usize,
        heads: [ProjectionHead; 3],
        weights: TeacherWeights,
    ) -> Result<Self, ModelError> {
        if student_dim == 0 {
            return Err(artifact("student_dim must be positive"));
        }
        for (slot, head) in TeacherSlot::ALL.iter().zip(&heads) {
            if head.in_dim != student_dim {
                return Err(artifact(format!(
                    "head {} reads {} dims, student_dim is {student_dim}",
                    slot.key(),
                    head.in_dim
                )));
            }
        }
        // Re-validate: the fields are public and may have been built by hand.
        let weights = TeacherWeights::new(weights.w_405b, weights.w_q72b, weights.w_llama70b)?;
        Ok(Self {
            student_dim,
            heads,
            weights,
        })
    }

    pub fn student_dim(&self) -> usize {
        self.student_dim
    }

    pub fn weights(&self) -> TeacherWeights {
        self.weights
    }

    pub fn head(&self, slot: TeacherSlot) -> &ProjectionHead {
        &self.heads[slot.index()]
    }

    fn check_state(&self, x: &[f32], name: &str) -> Result<(), ModelError> {
        if x.len() != self.student_dim {
            return Err(input(format!(
                "{name} has {} dims, the student width is {}",
                x.len(),
                self.student_dim
            )));
        }
        if let Some(i) = x.iter().position(|v| !v.is_finite()) {
            return Err(input(format!("{name}[{i}] = {} is not finite", x[i])));
        }
        Ok(())
    }

    /// Run `u` and `v` through all three heads (one batched matmul per head).
    pub fn project(&self, u: &[f32], v: &[f32]) -> Result<TriTeacherProjections, ModelError> {
        self.check_state(u, "u")?;
        self.check_state(v, "v")?;
        let x = Tensor::from_iter(u.iter().chain(v).copied(), &Device::Cpu)
            .and_then(|t| t.reshape((2, self.student_dim)))
            .map_err(candle_err("tri-teacher input"))?;
        let mut out = Vec::with_capacity(3);
        for (slot, head) in TeacherSlot::ALL.iter().zip(&self.heads) {
            let mut rows = head.forward(&x)?.into_iter();
            let (z1, z2) = match (rows.next(), rows.next()) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    return Err(ModelError::Inference(format!(
                        "head {} returned fewer than 2 rows",
                        slot.key()
                    )))
                }
            };
            out.push(TeacherProjectionPair { z1, z2 });
        }
        let mut it = out.into_iter();
        let (Some(proj_405b), Some(proj_q72b), Some(proj_llama70b)) =
            (it.next(), it.next(), it.next())
        else {
            return Err(ModelError::Inference("missing teacher projection".into()));
        };
        Ok(TriTeacherProjections {
            proj_405b,
            proj_q72b,
            proj_llama70b,
        })
    }
}

/// Cosine of two projections. Refuses non-finite values and directionless
/// (near-zero) vectors instead of reporting a meaningless 0.
pub fn projection_cosine(a: &[f32], b: &[f32], name: &str) -> Result<f64, ModelError> {
    if a.len() != b.len() || a.is_empty() {
        return Err(input(format!(
            "{name}: projections of {} and {} dims",
            a.len(),
            b.len()
        )));
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let (na, nb) = (na.sqrt(), nb.sqrt());
    if !(dot.is_finite() && na.is_finite() && nb.is_finite()) {
        return Err(ModelError::NumericalInstability(format!(
            "{name}: non-finite projection"
        )));
    }
    if na < MIN_PROJECTION_NORM || nb < MIN_PROJECTION_NORM {
        return Err(ModelError::NumericalInstability(format!(
            "{name}: projection norm {na:e} / {nb:e} below {MIN_PROJECTION_NORM:e}, cosine undefined"
        )));
    }
    Ok(dot / (na * nb))
}

/// `1 / (1 + exp(-15 (tri_sim - threshold)))`.
pub fn calibrated_probability(tri_sim: f64, threshold: f64) -> f64 {
    1.0 / (1.0 + (-TRI_TEACHER_SIGMOID_SLOPE * (tri_sim - threshold)).exp())
}

fn check_threshold(threshold: f64) -> Result<f64, ModelError> {
    if threshold.is_finite() && (-1.0..=1.0).contains(&threshold) {
        Ok(threshold)
    } else {
        Err(input(format!(
            "threshold {threshold} must be finite and in [-1, 1], the range of tri_sim"
        )))
    }
}

/// The decider's verdict on one pair. Field names match the Python dict.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TriTeacherDecision {
    pub same_meaning: bool,
    pub p_same_meaning: f64,
    pub tri_sim: f64,
    pub sim_405b: f64,
    pub sim_q72b: f64,
    pub sim_llama70b: f64,
    pub threshold: f64,
}

/// Decide from raw projections: cosine per teacher, weighted sum, sigmoid.
///
/// `TwoStageDualTrackGateway` (gen-zero-gate) does the same math in f32
/// inside its own decision path; the gate cannot call this crate (it is a
/// dependency of it), so the formula lives in both places.
pub fn decide_projections(
    projections: &TriTeacherProjections,
    weights: TeacherWeights,
    threshold: f64,
) -> Result<TriTeacherDecision, ModelError> {
    let threshold = check_threshold(threshold)?;
    let mut sims = [0f64; 3];
    for slot in TeacherSlot::ALL {
        let p = slot_projections(projections, slot);
        sims[slot.index()] = projection_cosine(&p.z1, &p.z2, &format!("proj_{}", slot.key()))?;
    }
    let tri_sim: f64 = weights
        .as_array()
        .iter()
        .zip(&sims)
        .map(|(w, s)| w * s)
        .sum();
    let p = calibrated_probability(tri_sim, threshold);
    if !(tri_sim.is_finite() && p.is_finite()) {
        return Err(ModelError::NumericalInstability(format!(
            "tri_sim {tri_sim} / p {p} not finite"
        )));
    }
    Ok(TriTeacherDecision {
        same_meaning: tri_sim >= threshold,
        p_same_meaning: p,
        tri_sim,
        sim_405b: sims[0],
        sim_q72b: sims[1],
        sim_llama70b: sims[2],
        threshold,
    })
}

/// One attention LoRA pair of one layer.
#[derive(Clone, Debug)]
pub struct AttentionLora {
    pub layer: usize,
    pub target: LoraTarget,
    /// `[rank, in]`.
    pub a: Tensor,
    /// `[out, rank]`.
    pub b: Tensor,
}

/// Where an adapter came from.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AdapterProvenance {
    pub path: PathBuf,
    pub sha256: String,
    pub format: &'static str,
    pub num_layers: usize,
    pub lora_pairs: usize,
    /// Training-only tensors that were validated and dropped.
    pub dropped_training_tensors: Vec<String>,
}

/// A parsed, validated `gen_zero.tri_teacher_lora.v1` adapter.
#[derive(Clone, Debug)]
pub struct TriTeacherLoRAAdapter {
    config: TriTeacherAdapterConfig,
    projector: TriTeacherProjector,
    lora: Vec<AttentionLora>,
    num_layers: usize,
    provenance: AdapterProvenance,
}

#[derive(Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
}

/// `__metadata__` map and tensor index of a safetensors header.
type Header = (HashMap<String, String>, HashMap<String, HeaderEntry>);

/// Split a safetensors file into its `__metadata__` map and tensor index.
fn read_header(bytes: &[u8], path: &Path) -> Result<Header, ModelError> {
    let n = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or_else(|| {
            artifact(format!(
                "{} is shorter than a safetensors header",
                path.display()
            ))
        })?;
    if n > MAX_HEADER_BYTES || 8 + n > bytes.len() as u64 {
        return Err(artifact(format!(
            "{}: header length {n} is implausible for a {}-byte file",
            path.display(),
            bytes.len()
        )));
    }
    let mut raw: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + n as usize])
            .map_err(|e| artifact(format!("{}: header JSON: {e}", path.display())))?;
    let meta = match raw.remove("__metadata__") {
        Some(m) => serde_json::from_value(m)
            .map_err(|e| artifact(format!("{}: __metadata__: {e}", path.display())))?,
        None => HashMap::new(),
    };
    let index = raw
        .into_iter()
        .map(|(k, v)| {
            serde_json::from_value::<HeaderEntry>(v)
                .map(|e| (k.clone(), e))
                .map_err(|e| artifact(format!("{}: tensor {k}: {e}", path.display())))
        })
        .collect::<Result<_, _>>()?;
    Ok((meta, index))
}

impl TriTeacherLoRAAdapter {
    /// Read and validate an adapter file. Every tensor must be F32, finite,
    /// expected by the config and of the expected shape; any missing or
    /// unexpected tensor is an error.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        use sha2::{Digest, Sha256};
        let bytes =
            std::fs::read(path).map_err(|e| artifact(format!("read {}: {e}", path.display())))?;
        let sha256: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let (meta, index) = read_header(&bytes, path)?;
        match meta.get("format").map(String::as_str) {
            Some(TRI_TEACHER_ADAPTER_FORMAT) => {}
            other => {
                return Err(artifact(format!(
                    "{} has format {other:?}, expected {TRI_TEACHER_ADAPTER_FORMAT:?}",
                    path.display()
                )))
            }
        }
        let config: TriTeacherAdapterConfig = serde_json::from_str(
            meta.get("adapter_config")
                .ok_or_else(|| artifact(format!("{} lacks adapter_config", path.display())))?,
        )
        .map_err(|e| artifact(format!("{}: adapter_config: {e}", path.display())))?;
        config.validate()?;
        if let Some((name, e)) = index.iter().find(|(_, e)| e.dtype != "F32") {
            return Err(artifact(format!("{name} is {}, expected F32", e.dtype)));
        }

        let d = config.student_dim;
        let dims = config.teacher_dims()?;
        let targets = config.lora_targets()?;
        let mut expected: HashMap<String, Vec<usize>> = HashMap::new();
        let mut optional_expected: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (slot, &t) in TeacherSlot::ALL.iter().zip(&dims) {
            let k = slot.key();
            expected.insert(format!("proj_{k}.0.weight"), vec![d]);
            expected.insert(format!("proj_{k}.0.bias"), vec![d]);
            expected.insert(format!("proj_{k}.1.weight"), vec![t, d]);
            expected.insert(format!("proj_{k}.1.bias"), vec![t]);
            optional_expected.insert(format!("teacher_mean_{k}"), vec![t]);
            optional_expected.insert(format!("teacher_std_{k}"), vec![t]);
        }
        if config.num_classes > 0 {
            let c = config.num_classes;
            optional_expected.insert("task_head.0.weight".into(), vec![d]);
            optional_expected.insert("task_head.0.bias".into(), vec![d]);
            optional_expected.insert("task_head.1.weight".into(), vec![c, d]);
            optional_expected.insert("task_head.1.bias".into(), vec![c]);
        }
        for (k, v) in optional_expected {
            if index.contains_key(&k) {
                expected.insert(k, v);
            }
        }
        // The layer count is not in the config: read it off the LoRA keys,
        // then require every layer to carry every target.
        let mut layers = BTreeSet::new();
        for name in index.keys() {
            if let Some(rest) = name.strip_prefix("backbone.layers.") {
                let layer = rest
                    .split('.')
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or_else(|| artifact(format!("unparseable LoRA key {name}")))?;
                layers.insert(layer);
            }
        }
        let num_layers = layers.len();
        if num_layers == 0 || layers.last() != Some(&(num_layers - 1)) {
            return Err(artifact(format!(
                "LoRA layers must be 0..N with none missing, got {layers:?}"
            )));
        }
        for layer in 0..num_layers {
            for &t in &targets {
                let p = format!("backbone.layers.{layer}.self_attn.{}", target_name(t));
                expected.insert(format!("{p}.lora_A"), vec![config.rank, d]);
                // B's output width depends on the base (q: hidden, v: kv width);
                // its rank side is checked here, the rest at merge time.
                let b = index.get(&format!("{p}.lora_B"));
                let out = b.and_then(|e| e.shape.first().copied()).unwrap_or(0);
                expected.insert(format!("{p}.lora_B"), vec![out, config.rank]);
            }
        }
        let have: BTreeSet<&String> = index.keys().collect();
        let want: BTreeSet<&String> = expected.keys().collect();
        if have != want {
            let missing: Vec<_> = want.difference(&have).take(5).collect();
            let extra: Vec<_> = have.difference(&want).take(5).collect();
            return Err(artifact(format!(
                "{}: tensor set mismatch, missing {missing:?}, unexpected {extra:?}",
                path.display()
            )));
        }
        for (name, shape) in &expected {
            let got = &index[name].shape;
            if got != shape || shape.contains(&0) {
                return Err(artifact(format!(
                    "{name}: shape {got:?}, expected {shape:?}"
                )));
            }
        }

        let mut tensors = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)
            .map_err(|e| artifact(format!("{}: {e}", path.display())))?;
        let mut take = |name: &str| {
            tensors
                .remove(name)
                .ok_or_else(|| artifact(format!("{name} vanished between index and load")))
        };
        let mut heads = Vec::with_capacity(3);
        for slot in TeacherSlot::ALL {
            let k = slot.key();
            heads.push(ProjectionHead::from_tensors(
                take(&format!("proj_{k}.0.weight"))?,
                take(&format!("proj_{k}.0.bias"))?,
                take(&format!("proj_{k}.1.weight"))?,
                take(&format!("proj_{k}.1.bias"))?,
                &format!("proj_{k}"),
            )?);
        }
        let heads: [ProjectionHead; 3] = heads
            .try_into()
            .map_err(|_| artifact("expected three projection heads"))?;
        let mut lora = Vec::with_capacity(num_layers * targets.len());
        for layer in 0..num_layers {
            for &target in &targets {
                let p = format!("backbone.layers.{layer}.self_attn.{}", target_name(target));
                let a = take(&format!("{p}.lora_A"))?;
                let b = take(&format!("{p}.lora_B"))?;
                finite_values(&a, &format!("{p}.lora_A"))?;
                finite_values(&b, &format!("{p}.lora_B"))?;
                lora.push(AttentionLora {
                    layer,
                    target,
                    a,
                    b,
                });
            }
        }
        // What is left is training-only: check it is sane, then drop it.
        let mut dropped: Vec<String> = tensors.keys().cloned().collect();
        dropped.sort();
        for name in &dropped {
            let values = finite_values(&tensors[name], name)?;
            if name.starts_with("teacher_std_") && values.iter().any(|&s| s <= 0.0) {
                return Err(artifact(format!("{name} must be > 0 everywhere")));
            }
        }
        let projector = TriTeacherProjector::new(d, heads, config.teacher_weights()?)?;
        let provenance = AdapterProvenance {
            path: path.to_path_buf(),
            sha256,
            format: TRI_TEACHER_ADAPTER_FORMAT,
            num_layers,
            lora_pairs: lora.len(),
            dropped_training_tensors: dropped,
        };
        Ok(Self {
            config,
            projector,
            lora,
            num_layers,
            provenance,
        })
    }

    pub fn config(&self) -> &TriTeacherAdapterConfig {
        &self.config
    }

    pub fn projector(&self) -> &TriTeacherProjector {
        &self.projector
    }

    pub fn lora(&self) -> &[AttentionLora] {
        &self.lora
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn provenance(&self) -> &AdapterProvenance {
        &self.provenance
    }

    /// Merge every LoRA pair into `model`: `W += (alpha / rank) * B @ A`.
    /// The model must match the adapter's width and layer count.
    pub fn merge_into(&self, model: &mut QwenModel) -> Result<(), ModelError> {
        let cfg = model.config();
        if cfg.hidden_size != self.config.student_dim || cfg.num_layers != self.num_layers {
            return Err(artifact(format!(
                "adapter is for {} layers of width {}, base model has {} of width {}",
                self.num_layers, self.config.student_dim, cfg.num_layers, cfg.hidden_size
            )));
        }
        // Check every pair before touching the model, so a bad adapter never
        // leaves it half merged.
        let kv = cfg.num_kv_heads * cfg.head_dim;
        for l in &self.lora {
            let out = match l.target {
                LoraTarget::QProj => cfg.hidden_size,
                LoraTarget::VProj => kv,
            };
            if l.b.dims()[0] != out {
                return Err(artifact(format!(
                    "layer {} {}: lora_B has {} output rows, the base projection has {out}",
                    l.layer,
                    target_name(l.target),
                    l.b.dims()[0]
                )));
            }
        }
        let scaling = self.config.lora_scaling();
        for l in &self.lora {
            let delta =
                l.b.matmul(&l.a)
                    .and_then(|d| d * scaling)
                    .map_err(candle_err("lora delta"))?;
            model.merge_linear_delta(l.layer, l.target, &delta)?;
        }
        Ok(())
    }
}

/// Where the student encoder came from.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EncoderProvenance {
    pub base_model: PathBuf,
    pub base_sha256: String,
    pub tokenizer: PathBuf,
    pub format: WeightFormat,
    pub pooling: &'static str,
}

/// Qwen2.5 with the adapter's LoRA merged, plus its tokenizer.
struct StudentEncoder {
    model: QwenModel,
    tokenizer: Tokenizer,
    provenance: EncoderProvenance,
}

impl StudentEncoder {
    fn load(
        adapter: &TriTeacherLoRAAdapter,
        base_model: &Path,
        tokenizer: Option<&Path>,
    ) -> Result<Self, ModelError> {
        // The adapter was trained on the full-precision HF checkpoint. A GGUF
        // base would add an unmeasured quantization error to every decision,
        // so it is refused rather than accepted with a quiet accuracy loss.
        if !base_model.is_dir() {
            return Err(artifact(format!(
                "{} must be a Hugging Face directory (config.json + model.safetensors); \
                 GGUF bases are refused for the tri-teacher adapter",
                base_model.display()
            )));
        }
        let tokenizer_path = tokenizer
            .map(Path::to_path_buf)
            .unwrap_or_else(|| base_model.join("tokenizer.json"));
        let tok = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            ModelError::QwenLoad(format!("tokenizer {}: {e}", tokenizer_path.display()))
        })?;
        let mut model = QwenModel::from_safetensors_dir(base_model)?;
        if tok.get_vocab_size(true) > model.config().vocab_size {
            return Err(ModelError::QwenLoad(format!(
                "tokenizer {} has {} tokens, model vocabulary is {}",
                tokenizer_path.display(),
                tok.get_vocab_size(true),
                model.config().vocab_size
            )));
        }
        adapter.merge_into(&mut model)?;
        let base_sha256 = crate::qwen::sha256_file(&base_model.join("model.safetensors"))?;
        Ok(Self {
            provenance: EncoderProvenance {
                base_model: base_model.to_path_buf(),
                base_sha256,
                tokenizer: tokenizer_path,
                format: model.format().clone(),
                pooling: "last_token_final_norm",
            },
            model,
            tokenizer: tok,
        })
    }

    /// Final-norm hidden state of the last token. Special tokens are added the
    /// way the Python tokenizer call does (`add_special_tokens=True`); Qwen2.5
    /// adds none.
    fn encode(&self, text: &str) -> Result<Vec<f32>, ModelError> {
        let ids = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| input(format!("tokenize: {e}")))?
            .get_ids()
            .to_vec();
        if ids.is_empty() {
            return Err(input(format!("{text:?} tokenizes to nothing")));
        }
        let t = ids.len();
        let (hidden, _) = self.model.forward(&[ids], None)?;
        let pooled = hidden
            .narrow(1, t - 1, 1)
            .and_then(|h| h.flatten_all())
            .and_then(|h| h.to_vec1::<f32>())
            .map_err(candle_err("tri-teacher pooling"))?;
        if pooled.iter().any(|v| !v.is_finite()) {
            return Err(ModelError::NumericalInstability(format!(
                "student state of {text:?} is not finite"
            )));
        }
        Ok(pooled)
    }
}

/// Provenance and settings of a decider, for logs and CLI output.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TriTeacherDeciderInfo {
    pub threshold: f64,
    pub sigmoid_slope: f64,
    pub weights: TeacherWeights,
    pub student_dim: usize,
    pub teacher_dims: [usize; 3],
    pub adapter: Option<AdapterProvenance>,
    pub encoder: Option<EncoderProvenance>,
}

/// Three-teacher (405B + Q72B + 70B) pair decider.
///
/// Two modes: with a student encoder ([`Self::load`]) it decides on text;
/// without one ([`Self::new`], [`Self::from_adapter`]) it decides on student
/// states given by the caller, and [`Self::decide`] fails with
/// [`ModelError::TriTeacherEncoderMissing`].
pub struct TriTeacherPairDecider {
    projector: TriTeacherProjector,
    encoder: Option<StudentEncoder>,
    threshold: f64,
    adapter: Option<AdapterProvenance>,
}

impl TriTeacherPairDecider {
    /// Projection-only decider from explicit heads.
    pub fn new(projector: TriTeacherProjector, threshold: f64) -> Result<Self, ModelError> {
        Ok(Self {
            projector,
            encoder: None,
            threshold: check_threshold(threshold)?,
            adapter: None,
        })
    }

    /// Projection-only decider with the adapter's real heads. The LoRA
    /// pairs need a base model and are not used in this mode.
    pub fn from_adapter(
        adapter: TriTeacherLoRAAdapter,
        threshold: f64,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            threshold: check_threshold(threshold)?,
            projector: adapter.projector,
            encoder: None,
            adapter: Some(adapter.provenance),
        })
    }

    /// Full decider: adapter + Qwen2.5-0.5B HF directory (+ tokenizer,
    /// default `tokenizer.json` in that directory).
    pub fn load(
        adapter_path: &Path,
        base_model: &Path,
        tokenizer: Option<&Path>,
        threshold: f64,
    ) -> Result<Self, ModelError> {
        let threshold = check_threshold(threshold)?;
        let adapter = TriTeacherLoRAAdapter::load(adapter_path)?;
        let encoder = StudentEncoder::load(&adapter, base_model, tokenizer)?;
        Ok(Self {
            threshold,
            projector: adapter.projector,
            encoder: Some(encoder),
            adapter: Some(adapter.provenance),
        })
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    pub fn projector(&self) -> &TriTeacherProjector {
        &self.projector
    }

    pub fn info(&self) -> TriTeacherDeciderInfo {
        TriTeacherDeciderInfo {
            threshold: self.threshold,
            sigmoid_slope: TRI_TEACHER_SIGMOID_SLOPE,
            weights: self.projector.weights,
            student_dim: self.projector.student_dim,
            teacher_dims: TeacherSlot::ALL.map(|s| self.projector.head(s).out_dim),
            adapter: self.adapter.clone(),
            encoder: self.encoder.as_ref().map(|e| e.provenance.clone()),
        }
    }

    fn encoder(&self) -> Result<&StudentEncoder, ModelError> {
        self.encoder.as_ref().ok_or_else(|| {
            ModelError::TriTeacherEncoderMissing(
                "built without a base model; load it with TriTeacherPairDecider::load \
                 or call decide_embeddings"
                    .into(),
            )
        })
    }

    /// Pooled student state of one sentence (needs the encoder).
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, ModelError> {
        self.encoder()?.encode(text)
    }

    pub fn project_embeddings(
        &self,
        u: &[f32],
        v: &[f32],
    ) -> Result<TriTeacherProjections, ModelError> {
        self.projector.project(u, v)
    }

    /// Raw projections of two sentences (needs the encoder).
    pub fn project_texts(
        &self,
        sentence_1: &str,
        sentence_2: &str,
    ) -> Result<TriTeacherProjections, ModelError> {
        let enc = self.encoder()?;
        self.projector
            .project(&enc.encode(sentence_1)?, &enc.encode(sentence_2)?)
    }

    pub fn decide_embeddings(
        &self,
        u: &[f32],
        v: &[f32],
    ) -> Result<TriTeacherDecision, ModelError> {
        decide_projections(
            &self.projector.project(u, v)?,
            self.projector.weights,
            self.threshold,
        )
    }

    pub fn decide(
        &self,
        sentence_1: &str,
        sentence_2: &str,
    ) -> Result<TriTeacherDecision, ModelError> {
        decide_projections(
            &self.project_texts(sentence_1, sentence_2)?,
            self.projector.weights,
            self.threshold,
        )
    }
}
