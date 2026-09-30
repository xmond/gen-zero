//! Reflex plugins: a trained low-rank LoRA residual latent updater plus one or
//! more linear readout heads, served as a zero-token (G=0), multi-step (R>0)
//! System-1 reflex.
//!
//! Recurrence (non-affine LayerNorm, no dense bottleneck: the state lives in
//! the full `input_dim` space):
//!
//! ```text
//! z_0     = LayerNorm(x)
//! h_t     = z_t . W_down                       (R^r)
//! c       = x . W_ctx                          (R^r, computed once)
//! g_t     = sigmoid([h_t; c] . W_gate + b_gate)
//! c_t     = tanh([h_t; c] . W_cand + b_cand)
//! Delta_t = (g_t * c_t) . W_up                  (R^input_dim)
//! z_{t+1} = LayerNorm(z_t + alpha * Delta_t)
//! ```
//!
//! Readout inverts the non-affine LayerNorm with x's own row statistics
//! (`input_row_stats`), so a linear head trained on raw `x` can be applied to
//! the reflex state unchanged:
//!
//! ```text
//! z~     = z_T * std(x) + mean(x)
//! logits = W_head . z~ + b_head
//! ```
//!
//! Telemetry reports the step residual `||z_{t+1} - z_t||` (used for the
//! early-stop convergence check) and the step-residual contraction ratio
//! `gamma_t = ||z_{t+1} - z_t|| / ||z_t - z_{t-1}||`. This ratio is defined on
//! the LayerNorm'd state trajectory, which differs from the
//! `python/gen_zero/model/latent_updater.py` telemetry (which ratios the
//! pre-alpha, pre-LayerNorm update norms `||Delta_{t+1}|| / ||Delta_t||`); the
//! two are related but not numerically identical, so do not treat a mismatch
//! between them as a bug when cross-checking against the Python reference.
//!
//! This module implements only the low-rank residual mode: it is the only
//! mode the shipped reflex plugins (`python/gen_zero/model/reflex_plugin.py`)
//! use, and the dense bottleneck mode has no place in a sub-millisecond
//! serving path at `input_dim = 8192`.

use crate::error::ModelError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// LayerNorm epsilon, matching `LAYER_NORM_EPS` in the Python reference.
pub const LAYER_NORM_EPS: f32 = 1e-5;

/// Denominator floor for the contraction ratio, matching `CONTRACTION_EPS` in
/// the Python reference. Without it, two consecutive exactly-zero residuals
/// (already converged) would divide 0 / 0 into NaN instead of reporting a
/// flat ratio.
pub const CONTRACTION_EPS: f32 = 1e-12;

const ARCHIVE_MAGIC: &[u8; 8] = b"GZRFLX1\0";
const ARCHIVE_FORMAT: &str = "gen_zero.reflex_plugin.rust.v1";

// ---------------------------------------------------------------------------
// Numerical primitives
// ---------------------------------------------------------------------------

/// `1 / (1 + exp(-v))`. Never produces NaN for finite `v`: the largest
/// magnitude `exp` can take is `+inf`, and `1 / (1 + inf) = 0.0` exactly.
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// `y = x . w` for `x: [n]` and row-major `w: [n, m]`, i.e. `y[j] = sum_i x[i] * w[i*m+j]`.
fn matvec(x: &[f32], w: &[f32], n: usize, m: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(w.len(), n * m);
    let mut y = vec![0.0f32; m];
    for i in 0..n {
        let xi = x[i];
        if xi == 0.0 {
            continue;
        }
        let row = &w[i * m..(i + 1) * m];
        for j in 0..m {
            y[j] += xi * row[j];
        }
    }
    y
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Non-affine LayerNorm: `(v - mean(v)) / sqrt(var(v) + eps)`. Two-pass
/// (mean, then sum of squared deviations): the one-pass `E[x^2] - E[x]^2`
/// form catastrophically cancels once `x` is not already zero-centered.
fn layer_norm(v: &[f32]) -> Vec<f32> {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&e| (e - mean) * (e - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + LAYER_NORM_EPS).sqrt();
    v.iter().map(|&e| (e - mean) * inv_std).collect()
}

/// Per-row `(mean, sqrt(var + eps))`, the statistics `layer_norm` removes.
/// `z * std + mean` inverts the LayerNorm exactly (up to `eps`), so a linear
/// head fit on raw `x` can be applied to any reflex state `z_t`.
fn input_row_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&e| (e - mean) * (e - mean)).sum::<f32>() / n;
    (mean, (var + LAYER_NORM_EPS).sqrt())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

/// Telemetry from one reflex recurrence unroll.
///
/// `residual_history[t] = ||z_{t+1} - z_t||`, in step order, one entry per
/// step actually taken (`len == reflex_depth`). `gamma_history[t] =
/// residual_history[t+1] / (residual_history[t] + CONTRACTION_EPS)`, one
/// entry per consecutive pair (`len == reflex_depth - 1`, empty when
/// `reflex_depth <= 1`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexTelemetry {
    /// Number of recurrence steps actually taken ("R" in the G=0, R>0 reflex contract).
    pub reflex_depth: usize,
    /// True iff the loop stopped early because `residual < epsilon`.
    pub converged: bool,
    /// `||z_{t+1} - z_t||` at the last step taken.
    pub final_residual: f32,
    pub residual_history: Vec<f32>,
    pub gamma_history: Vec<f32>,
    /// `max(gamma_history)`, or `None` when fewer than two steps were taken.
    pub max_gamma: Option<f32>,
}

fn run_recurrence(op: &ReflexOperator, z0: Vec<f32>, ctx: &[f32]) -> (Vec<f32>, ReflexTelemetry) {
    let mut z = z0;
    let mut residual_history = Vec::with_capacity(op.config.steps);
    let mut converged = false;
    for _ in 0..op.config.steps {
        let z_next = op.step(&z, ctx);
        let residual = l2_distance(&z_next, &z);
        residual_history.push(residual);
        z = z_next;
        if residual < op.config.epsilon {
            converged = true;
            break;
        }
    }
    let gamma_history: Vec<f32> = residual_history
        .windows(2)
        .map(|w| w[1] / (w[0] + CONTRACTION_EPS))
        .collect();
    let max_gamma = gamma_history
        .iter()
        .cloned()
        .fold(None, |acc, g| Some(acc.map_or(g, |m: f32| m.max(g))));
    let telemetry = ReflexTelemetry {
        reflex_depth: residual_history.len(),
        converged,
        final_residual: *residual_history
            .last()
            .expect("op.config.steps >= 1, so at least one step always runs"),
        residual_history,
        gamma_history,
        max_gamma,
    };
    (z, telemetry)
}

// ---------------------------------------------------------------------------
// Operator
// ---------------------------------------------------------------------------

/// Validated configuration for the low-rank recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReflexOperatorConfig {
    pub input_dim: usize,
    pub lora_rank: usize,
    pub alpha: f32,
    pub steps: usize,
    pub epsilon: f32,
}

impl ReflexOperatorConfig {
    pub fn validate(self) -> Result<Self, ModelError> {
        if self.input_dim == 0 {
            return Err(ModelError::ReflexConfig(
                "input_dim must be positive".into(),
            ));
        }
        if self.lora_rank == 0 || self.lora_rank >= self.input_dim {
            return Err(ModelError::ReflexConfig(format!(
                "lora_rank must be in [1, input_dim={}), got {}",
                self.input_dim, self.lora_rank
            )));
        }
        if !(self.alpha.is_finite() && self.alpha > 0.0 && self.alpha <= 1.0) {
            return Err(ModelError::ReflexConfig(format!(
                "alpha must be finite and in (0, 1], got {}",
                self.alpha
            )));
        }
        if self.steps == 0 {
            return Err(ModelError::ReflexConfig("steps (T) must be >= 1".into()));
        }
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(ModelError::ReflexConfig(format!(
                "epsilon must be finite and positive, got {}",
                self.epsilon
            )));
        }
        Ok(self)
    }
}

/// The trained low-rank LoRA residual updater: `W_down`, `W_ctx`, the gate and
/// candidate projections, and `W_up`. Every weight is stored row-major, input
/// dimension first (`z . W`, not `W . z`), matching the NumPy reference layout.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexOperator {
    config: ReflexOperatorConfig,
    /// `(input_dim, r)`: `z_t . w_down -> h_t in R^r`.
    w_down: Vec<f32>,
    /// `(input_dim, r)`: `x . w_ctx -> c in R^r`.
    w_ctx: Vec<f32>,
    /// `(2r, r)`.
    w_gate: Vec<f32>,
    /// `(r,)`.
    b_gate: Vec<f32>,
    /// `(2r, r)`.
    w_cand: Vec<f32>,
    /// `(r,)`.
    b_cand: Vec<f32>,
    /// `(r, input_dim)`: `(g_t * c_t) . w_up -> Delta_t in R^input_dim`.
    w_up: Vec<f32>,
}

impl ReflexOperator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: ReflexOperatorConfig,
        w_down: Vec<f32>,
        w_ctx: Vec<f32>,
        w_gate: Vec<f32>,
        b_gate: Vec<f32>,
        w_cand: Vec<f32>,
        b_cand: Vec<f32>,
        w_up: Vec<f32>,
    ) -> Result<Self, ModelError> {
        let config = config.validate()?;
        let d = config.input_dim;
        let r = config.lora_rank;
        let expected: [(&str, usize, usize); 7] = [
            ("w_down", w_down.len(), d * r),
            ("w_ctx", w_ctx.len(), d * r),
            ("w_gate", w_gate.len(), 2 * r * r),
            ("b_gate", b_gate.len(), r),
            ("w_cand", w_cand.len(), 2 * r * r),
            ("b_cand", b_cand.len(), r),
            ("w_up", w_up.len(), r * d),
        ];
        for (name, got, want) in expected {
            if got != want {
                return Err(ModelError::ReflexArtifact(format!(
                    "operator weight {name} expected {want} elements, got {got}"
                )));
            }
        }
        for (name, arr) in [
            ("w_down", &w_down),
            ("w_ctx", &w_ctx),
            ("w_gate", &w_gate),
            ("b_gate", &b_gate),
            ("w_cand", &w_cand),
            ("b_cand", &b_cand),
            ("w_up", &w_up),
        ] {
            if !all_finite(arr) {
                return Err(ModelError::ReflexArtifact(format!(
                    "operator weight {name} contains NaN or Inf"
                )));
            }
        }
        Ok(Self {
            config,
            w_down,
            w_ctx,
            w_gate,
            b_gate,
            w_cand,
            b_cand,
            w_up,
        })
    }

    pub fn config(&self) -> ReflexOperatorConfig {
        self.config
    }

    /// Every weight array, in the exact fixed order [`ReflexPlugin::to_bytes`]
    /// writes them and [`ReflexOperator::new`] takes them. Public: an
    /// external online-adaptation loop (`gen-zero-service`'s
    /// `ReflexOnlineAdapter`) needs read access to the same arrays the patch
    /// engine (`patch.rs`) diffs, to build a same-shape all-zero delta for
    /// the arrays it does not touch.
    pub fn weight_arrays(&self) -> [&[f32]; 7] {
        [
            &self.w_down,
            &self.w_ctx,
            &self.w_gate,
            &self.b_gate,
            &self.w_cand,
            &self.b_cand,
            &self.w_up,
        ]
    }

    /// Mutable counterpart of [`Self::weight_arrays`], for the zero-allocation
    /// in-place patch path, which adds deltas directly onto these buffers
    /// instead of cloning them first.
    pub(crate) fn weight_arrays_mut(&mut self) -> [&mut [f32]; 7] {
        [
            &mut self.w_down,
            &mut self.w_ctx,
            &mut self.w_gate,
            &mut self.b_gate,
            &mut self.w_cand,
            &mut self.b_cand,
            &mut self.w_up,
        ]
    }

    fn context(&self, x: &[f32]) -> Vec<f32> {
        matvec(x, &self.w_ctx, self.config.input_dim, self.config.lora_rank)
    }

    /// One recurrence step: `z_{t+1} = LayerNorm(z_t + alpha * Delta_t)`.
    fn step(&self, z: &[f32], ctx: &[f32]) -> Vec<f32> {
        let r = self.config.lora_rank;
        let d = self.config.input_dim;
        let h = matvec(z, &self.w_down, d, r);
        let mut hc = Vec::with_capacity(2 * r);
        hc.extend_from_slice(&h);
        hc.extend_from_slice(ctx);
        let g_pre = matvec(&hc, &self.w_gate, 2 * r, r);
        let c_pre = matvec(&hc, &self.w_cand, 2 * r, r);
        let gated: Vec<f32> = (0..r)
            .map(|k| sigmoid(g_pre[k] + self.b_gate[k]) * (c_pre[k] + self.b_cand[k]).tanh())
            .collect();
        let delta = matvec(&gated, &self.w_up, r, d);
        let raw: Vec<f32> = z
            .iter()
            .zip(delta.iter())
            .map(|(&zi, &di)| zi + self.config.alpha * di)
            .collect();
        layer_norm(&raw)
    }

    /// `z_0 = LayerNorm(x)`, then run the recurrence with early stop.
    ///
    /// Returns `(z_final, telemetry)`. `z_final` is `z_T` where `T =
    /// telemetry.reflex_depth` (the recurrence stops as soon as it converges,
    /// so `T` may be less than `config.steps`).
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, ReflexTelemetry), ModelError> {
        self.validate_input(x)?;
        let ctx = self.context(x);
        let z0 = layer_norm(x);
        Ok(run_recurrence(self, z0, &ctx))
    }

    fn validate_input(&self, x: &[f32]) -> Result<(), ModelError> {
        if x.len() != self.config.input_dim {
            return Err(ModelError::ReflexInput(format!(
                "expected input of length {}, got {}",
                self.config.input_dim,
                x.len()
            )));
        }
        if !all_finite(x) {
            return Err(ModelError::ReflexInput("input contains NaN or Inf".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Readout head
// ---------------------------------------------------------------------------

/// One linear readout: `logits = weight . restored_x + bias`, labels in
/// `candidates` order.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexHead {
    pub name: String,
    /// Row-major `(K, input_dim)`.
    pub weight: Vec<f32>,
    /// `(K,)`.
    pub bias: Vec<f32>,
    pub candidates: Vec<String>,
}

impl ReflexHead {
    pub fn new(
        name: impl Into<String>,
        input_dim: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
        candidates: Vec<String>,
    ) -> Result<Self, ModelError> {
        let name = name.into();
        if candidates.is_empty() {
            return Err(ModelError::ReflexArtifact(format!(
                "head {name:?} has no candidates"
            )));
        }
        let k = candidates.len();
        if weight.len() != k * input_dim {
            return Err(ModelError::ReflexArtifact(format!(
                "head {name:?} weight expected {} elements ({k} x {input_dim}), got {}",
                k * input_dim,
                weight.len()
            )));
        }
        if bias.len() != k {
            return Err(ModelError::ReflexArtifact(format!(
                "head {name:?} bias expected {k} elements, got {}",
                bias.len()
            )));
        }
        if !all_finite(&weight) || !all_finite(&bias) {
            return Err(ModelError::ReflexArtifact(format!(
                "head {name:?} weight or bias contains NaN or Inf"
            )));
        }
        Ok(Self {
            name,
            weight,
            bias,
            candidates,
        })
    }

    fn input_dim(&self) -> usize {
        self.weight.len() / self.candidates.len().max(1)
    }

    fn logits(&self, restored: &[f32]) -> Vec<f32> {
        let k = self.candidates.len();
        let d = restored.len();
        (0..k)
            .map(|row| {
                let w = &self.weight[row * d..(row + 1) * d];
                let dot: f32 = w.iter().zip(restored.iter()).map(|(&a, &b)| a * b).sum();
                dot + self.bias[row]
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// The result of one `predict` call.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexDecision {
    pub head: String,
    pub class_index: usize,
    pub label: String,
    pub probabilities: Vec<f32>,
    pub confidence: f32,
    pub telemetry: ReflexTelemetry,
    pub latency_ms: f64,
}

/// A trained low-rank updater plus one or more readout heads, served as a reflex.
///
/// `default_head` is the head `forward`/`predict` use when called with
/// `head: None`. `None` means every call must name a head explicitly (a
/// multi-head plugin with no sensible default, e.g. the 13-head `universal`
/// plugin).
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexPlugin {
    task: String,
    operator: ReflexOperator,
    heads: BTreeMap<String, ReflexHead>,
    default_head: Option<String>,
}

impl ReflexPlugin {
    pub fn new(
        task: impl Into<String>,
        operator: ReflexOperator,
        heads: Vec<ReflexHead>,
        default_head: Option<String>,
    ) -> Result<Self, ModelError> {
        let task = task.into();
        if heads.is_empty() {
            return Err(ModelError::ReflexConfig(format!(
                "reflex plugin {task:?} needs at least one head"
            )));
        }
        let mut map = BTreeMap::new();
        for head in heads {
            if head.input_dim() != operator.config.input_dim {
                return Err(ModelError::ReflexConfig(format!(
                    "head {:?} input dim {} != operator input dim {}",
                    head.name,
                    head.input_dim(),
                    operator.config.input_dim
                )));
            }
            if map.insert(head.name.clone(), head).is_some() {
                return Err(ModelError::ReflexConfig(
                    "duplicate head name in reflex plugin".into(),
                ));
            }
        }
        if let Some(ref d) = default_head {
            if !map.contains_key(d) {
                return Err(ModelError::ReflexConfig(format!(
                    "default_head {d:?} not among the plugin's heads"
                )));
            }
        }
        Ok(Self {
            task,
            operator,
            heads: map,
            default_head,
        })
    }

    pub fn task(&self) -> &str {
        &self.task
    }

    pub fn input_dim(&self) -> usize {
        self.operator.config.input_dim
    }

    pub fn operator(&self) -> &ReflexOperator {
        &self.operator
    }

    pub fn heads(&self) -> impl Iterator<Item = &ReflexHead> {
        self.heads.values()
    }

    pub fn default_head(&self) -> Option<&str> {
        self.default_head.as_deref()
    }

    /// SHA-256 of [`Self::to_bytes`]'s output: a content address for the
    /// plugin's exact weights, config, and head set. Used by the patch engine
    /// (`patch.rs`) to fail closed when a patch is applied to the wrong base
    /// or its own resulting weights don't match what the patch promised.
    pub fn sha256(&self) -> Result<[u8; 32], ModelError> {
        use sha2::{Digest, Sha256};
        let bytes = self.to_bytes()?;
        Ok(Sha256::digest(&bytes).into())
    }

    /// Mutable access for the zero-allocation in-place patch path
    /// (`patch.rs`), which mutates the operator's weight buffers and the
    /// matched heads' `weight`/`bias` buffers directly instead of cloning.
    pub(crate) fn operator_mut(&mut self) -> &mut ReflexOperator {
        &mut self.operator
    }

    pub(crate) fn head_mut(&mut self, name: &str) -> Option<&mut ReflexHead> {
        self.heads.get_mut(name)
    }

    fn resolve_head(&self, head: Option<&str>) -> Result<&ReflexHead, ModelError> {
        let key = match head {
            Some(h) => h,
            None => self.default_head.as_deref().ok_or_else(|| {
                ModelError::ReflexUnknownHead(format!(
                    "{}: no default head; name one explicitly",
                    self.task
                ))
            })?,
        };
        self.heads
            .get(key)
            .ok_or_else(|| ModelError::ReflexUnknownHead(key.to_string()))
    }

    /// Run the recurrence and the named (or default) head's readout.
    /// Returns `(logits, telemetry)`.
    pub fn forward(
        &self,
        x: &[f32],
        head: Option<&str>,
    ) -> Result<(Vec<f32>, ReflexTelemetry), ModelError> {
        let h = self.resolve_head(head)?;
        let (restored, telemetry) = self.restored_features(x)?;
        let logits = h.logits(&restored);
        if !all_finite(&logits) {
            return Err(ModelError::NumericalInstability(format!(
                "{}/{}: non-finite logits from finite input",
                self.task, h.name
            )));
        }
        Ok((logits, telemetry))
    }

    /// Run the recurrence and return the exact feature vector a head's linear
    /// readout is applied to (`restored` in [`Self::forward`]'s body),
    /// alongside the reflex telemetry. This crate has no autodiff, so the
    /// head-local adapter (`gen-zero-service`'s `ReflexOnlineAdapter`) that
    /// computes a gradient for a head's `weight`/`bias` needs this exact input
    /// vector, not just the finished logits `forward` returns.
    pub fn restored_features(&self, x: &[f32]) -> Result<(Vec<f32>, ReflexTelemetry), ModelError> {
        let (z_final, telemetry) = self.operator.forward(x)?;
        let (mean, std) = input_row_stats(x);
        let restored: Vec<f32> = z_final.iter().map(|&zi| zi * std + mean).collect();
        Ok((restored, telemetry))
    }

    /// Classify one sample: softmax the head logits and report the reflex
    /// telemetry alongside the decision. `latency_ms` covers validation
    /// through softmax.
    pub fn predict(&self, x: &[f32], head: Option<&str>) -> Result<ReflexDecision, ModelError> {
        let t0 = std::time::Instant::now();
        let h = self.resolve_head(head)?;
        let head_name = h.name.clone();
        let candidates = h.candidates.clone();
        let (logits, telemetry) = self.forward(x, Some(&head_name))?;
        let probs = softmax(&logits);
        let (idx, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("softmax output is never NaN"))
            .expect("head has at least one candidate");
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(ReflexDecision {
            head: head_name,
            class_index: idx,
            label: candidates[idx].clone(),
            probabilities: probs,
            confidence,
            telemetry,
            latency_ms,
        })
    }

    // -- binary loader ---------------------------------------------------

    /// Serialize to the zero-copy-loadable binary format: an 8-byte magic, an
    /// 8-byte little-endian JSON header length, the JSON header (names,
    /// shapes, config), zero-padding out to a 4-byte boundary, then every
    /// weight tensor as tightly packed little-endian `f32`, in a fixed order
    /// (operator weights, then each head's weight and bias in header order).
    pub fn to_bytes(&self) -> Result<Vec<u8>, ModelError> {
        let header = ReflexArchiveHeader {
            format: ARCHIVE_FORMAT.to_string(),
            task: self.task.clone(),
            config: self.operator.config,
            default_head: self.default_head.clone(),
            heads: self
                .heads
                .values()
                .map(|h| ReflexHeadHeader {
                    name: h.name.clone(),
                    candidates: h.candidates.clone(),
                })
                .collect(),
        };
        let header_bytes = serde_json::to_vec(&header)
            .map_err(|e| ModelError::ReflexArtifact(format!("encoding header: {e}")))?;

        let mut out = Vec::new();
        out.extend_from_slice(ARCHIVE_MAGIC);
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        let pad = (4 - out.len() % 4) % 4;
        out.resize(out.len() + pad, 0);

        for arr in [
            &self.operator.w_down,
            &self.operator.w_ctx,
            &self.operator.w_gate,
            &self.operator.b_gate,
            &self.operator.w_cand,
            &self.operator.b_cand,
            &self.operator.w_up,
        ] {
            out.extend_from_slice(bytemuck::cast_slice(arr));
        }
        for h in self.heads.values() {
            out.extend_from_slice(bytemuck::cast_slice(&h.weight));
            out.extend_from_slice(bytemuck::cast_slice(&h.bias));
        }
        Ok(out)
    }

    /// Parse the binary format written by [`Self::to_bytes`].
    ///
    /// The JSON header is a small, ordinary decode. Every weight tensor is
    /// read with a zero-copy `bytemuck` cast directly over `bytes` (no
    /// per-element parsing, no intermediate serialization tree); the plugin
    /// then copies those slices into its own owned storage so the caller's
    /// buffer can be dropped or reused. Every array is validated for shape
    /// and finiteness before use: a corrupt or truncated archive fails
    /// closed with a `ReflexArtifact` error, never a panic.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() < ARCHIVE_MAGIC.len() + 8 || &bytes[..ARCHIVE_MAGIC.len()] != ARCHIVE_MAGIC {
            return Err(ModelError::ReflexArtifact(
                "bad magic: not a gen-zero reflex plugin archive".into(),
            ));
        }
        let mut cursor = ARCHIVE_MAGIC.len();
        let header_len =
            u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().expect("8-byte slice"))
                as usize;
        cursor += 8;
        let header_end = cursor.checked_add(header_len).ok_or_else(|| {
            ModelError::ReflexArtifact("header length overflows archive size".into())
        })?;
        if header_end > bytes.len() {
            return Err(ModelError::ReflexArtifact(
                "archive truncated: header length exceeds buffer".into(),
            ));
        }
        let header: ReflexArchiveHeader = serde_json::from_slice(&bytes[cursor..header_end])
            .map_err(|e| ModelError::ReflexArtifact(format!("decoding header: {e}")))?;
        if header.format != ARCHIVE_FORMAT {
            return Err(ModelError::ReflexArtifact(format!(
                "archive format {:?} != {ARCHIVE_FORMAT:?}",
                header.format
            )));
        }
        cursor = header_end + (4 - header_end % 4) % 4;

        let config = header.config;
        let d = config.validate()?.input_dim;
        let r = config.lora_rank;

        // `d` and `r` come straight from an untrusted archive header (only
        // range-checked by `validate`, not bounded in absolute size), so
        // every shape product below must fail closed on overflow instead of
        // panicking (debug builds) or wrapping to a too-small allocation
        // (release builds).
        let checked_mul = |a: usize, b: usize| -> Result<usize, ModelError> {
            a.checked_mul(b)
                .ok_or_else(|| ModelError::ReflexArtifact("archive shape overflows usize".into()))
        };

        let mut take = |n: usize| -> Result<Vec<f32>, ModelError> {
            let nbytes = n
                .checked_mul(4)
                .ok_or_else(|| ModelError::ReflexArtifact("array length overflow".into()))?;
            let end = cursor.checked_add(nbytes).ok_or_else(|| {
                ModelError::ReflexArtifact("array extends past archive size".into())
            })?;
            if end > bytes.len() {
                return Err(ModelError::ReflexArtifact(
                    "archive truncated: payload shorter than declared shapes".into(),
                ));
            }
            let slice: &[f32] = bytemuck::try_cast_slice(&bytes[cursor..end]).map_err(|e| {
                ModelError::ReflexArtifact(format!("misaligned array in archive: {e}"))
            })?;
            cursor = end;
            Ok(slice.to_vec())
        };

        let dr = checked_mul(d, r)?;
        let rr2 = checked_mul(checked_mul(2, r)?, r)?;
        let w_down = take(dr)?;
        let w_ctx = take(dr)?;
        let w_gate = take(rr2)?;
        let b_gate = take(r)?;
        let w_cand = take(rr2)?;
        let b_cand = take(r)?;
        let w_up = take(dr)?;
        let operator =
            ReflexOperator::new(config, w_down, w_ctx, w_gate, b_gate, w_cand, b_cand, w_up)?;

        let mut heads = Vec::with_capacity(header.heads.len());
        for hh in &header.heads {
            let k = hh.candidates.len();
            let weight = take(checked_mul(k, d)?)?;
            let bias = take(k)?;
            heads.push(ReflexHead::new(
                hh.name.clone(),
                d,
                weight,
                bias,
                hh.candidates.clone(),
            )?);
        }

        if cursor != bytes.len() {
            return Err(ModelError::ReflexArtifact(format!(
                "archive has {} unexpected trailing bytes",
                bytes.len() - cursor
            )));
        }

        Self::new(header.task, operator, heads, header.default_head)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReflexArchiveHeader {
    format: String,
    task: String,
    config: ReflexOperatorConfig,
    default_head: Option<String>,
    heads: Vec<ReflexHeadHeader>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReflexHeadHeader {
    name: String,
    candidates: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 generator (xorshift32), so tests never
    /// depend on an external `rand` dependency for reflex.rs and are
    /// reproducible across runs.
    struct Xorshift32(u32);
    impl Xorshift32 {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            // Map to roughly [-1, 1].
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    fn zero_up_operator(input_dim: usize, rank: usize, steps: usize) -> ReflexOperator {
        let mut rng = Xorshift32(0xC0FFEE);
        let config = ReflexOperatorConfig {
            input_dim,
            lora_rank: rank,
            alpha: 0.5,
            steps,
            epsilon: 1e-4,
        };
        ReflexOperator::new(
            config,
            rng.vec(input_dim * rank),
            rng.vec(input_dim * rank),
            rng.vec(2 * rank * rank),
            rng.vec(rank),
            rng.vec(2 * rank * rank),
            rng.vec(rank),
            vec![0.0; rank * input_dim], // W_up = 0: identity operator at init.
        )
        .unwrap()
    }

    fn random_operator(input_dim: usize, rank: usize, steps: usize, seed: u32) -> ReflexOperator {
        let mut rng = Xorshift32(seed);
        let config = ReflexOperatorConfig {
            input_dim,
            lora_rank: rank,
            alpha: 0.5,
            steps,
            epsilon: 1e-6,
        };
        ReflexOperator::new(
            config,
            rng.vec(input_dim * rank),
            rng.vec(input_dim * rank),
            rng.vec(2 * rank * rank),
            rng.vec(rank),
            rng.vec(2 * rank * rank),
            rng.vec(rank),
            rng.vec(rank * input_dim),
        )
        .unwrap()
    }

    // -- W_up = 0 identity property (matches the Python module docstring's
    //    "every z_t equals LayerNorm(x) up to the LayerNorm epsilon" claim) --

    #[test]
    fn zero_up_operator_is_identity_on_layer_normed_input() {
        let op = zero_up_operator(16, 4, 4);
        let mut rng = Xorshift32(42);
        let x = rng.vec(16);
        let (z_final, telemetry) = op.forward(&x).unwrap();
        let expected = layer_norm(&x);
        // Delta_t = 0 exactly (W_up = 0), so z_1 = LayerNorm(LayerNorm(x)).
        // Re-normalizing an already-normalized vector is NOT an exact no-op
        // once epsilon > 0 (var(LayerNorm(x)) is epsilon-close to 1, not
        // exactly 1), so this only holds "up to the LayerNorm epsilon", as
        // the Python module docstring states; 1e-3 comfortably covers that
        // epsilon-scale drift while still catching a real implementation bug.
        for (a, b) in z_final.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        // The recurrence still takes exactly one step and converges: the
        // epsilon-scale re-normalization drift is far below `epsilon`.
        assert_eq!(telemetry.reflex_depth, 1);
        assert!(telemetry.converged);
        assert!(telemetry.final_residual < op.config().epsilon);
        assert!(telemetry.gamma_history.is_empty());
        assert_eq!(telemetry.max_gamma, None);
    }

    #[test]
    fn zero_up_plugin_readout_recovers_raw_input_linear_head() {
        let op = zero_up_operator(8, 2, 3);
        let mut rng = Xorshift32(7);
        let weight = rng.vec(3 * 8);
        let bias = rng.vec(3);
        let head = ReflexHead::new(
            "task",
            8,
            weight.clone(),
            bias.clone(),
            vec!["a".into(), "b".into(), "c".into()],
        )
        .unwrap();
        let plugin = ReflexPlugin::new("task", op, vec![head], Some("task".into())).unwrap();

        let mut xr = Xorshift32(99);
        let x = xr.vec(8);
        let (logits, _telemetry) = plugin.forward(&x, None).unwrap();

        // With W_up = 0, z_T == LayerNorm(x) exactly, and the readout undoes
        // the LayerNorm with x's own row stats: restored == x (up to the
        // LayerNorm epsilon), so logits == W_head . x + b_head.
        let mut expected = [0.0f32; 3];
        for k in 0..3 {
            let mut dot = 0.0;
            for i in 0..8 {
                dot += weight[k * 8 + i] * x[i];
            }
            expected[k] = dot + bias[k];
        }
        for (a, b) in logits.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn restored_features_matches_forward_logits_via_manual_dot_product() {
        let op = random_operator(16, 4, 3, 4242);
        let mut rng = Xorshift32(17);
        let weight = rng.vec(2 * 16);
        let bias = rng.vec(2);
        let head = ReflexHead::new(
            "h",
            16,
            weight.clone(),
            bias.clone(),
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        let plugin = ReflexPlugin::new("t", op, vec![head], Some("h".into())).unwrap();
        let mut xr = Xorshift32(303);
        let x = xr.vec(16);

        let (restored, telemetry_a) = plugin.restored_features(&x).unwrap();
        let (logits, telemetry_b) = plugin.forward(&x, None).unwrap();
        assert_eq!(telemetry_a, telemetry_b);

        let mut expected = vec![0.0f32; 2];
        for k in 0..2 {
            let mut dot = 0.0;
            for i in 0..16 {
                dot += weight[k * 16 + i] * restored[i];
            }
            expected[k] = dot + bias[k];
        }
        assert_eq!(logits, expected);
    }

    // -- hand-checked tiny case (D=4, r=2), one step against an f64 reference --

    #[test]
    fn one_step_matches_hand_computed_f64_reference() {
        let config = ReflexOperatorConfig {
            input_dim: 4,
            lora_rank: 2,
            alpha: 0.5,
            steps: 1,
            epsilon: 1e-9, // small enough that the single step never "converges" early
        };
        let w_down = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // (4,2)
        let w_ctx = vec![0.05, -0.1, 0.15, -0.2, 0.25, -0.3, 0.35, -0.4]; // (4,2)
        let w_gate = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]; // (4,2)
        let b_gate = vec![0.01, -0.02];
        let w_cand = vec![-0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4]; // (4,2)
        let b_cand = vec![0.02, -0.01];
        let w_up = vec![0.3, -0.3, 0.1, 0.2, 0.6, -0.6, -0.1, 0.05]; // (2,4)
        let op = ReflexOperator::new(
            config,
            w_down.clone(),
            w_ctx.clone(),
            w_gate.clone(),
            b_gate.clone(),
            w_cand.clone(),
            b_cand.clone(),
            w_up.clone(),
        )
        .unwrap();

        let x = [1.0f32, 2.0, 3.0, 4.0];
        let (z_final, telemetry) = op.forward(&x).unwrap();

        // f64 reference computation, independent code path.
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let mean = xf.iter().sum::<f64>() / 4.0;
        let var = xf.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 4.0;
        let z0: Vec<f64> = xf
            .iter()
            .map(|&v| (v - mean) / (var + 1e-5f64).sqrt())
            .collect();

        let matvec64 = |v: &[f64], w: &[f32], n: usize, m: usize| -> Vec<f64> {
            let mut out = vec![0.0f64; m];
            for i in 0..n {
                for j in 0..m {
                    out[j] += v[i] * w[i * m + j] as f64;
                }
            }
            out
        };
        let h = matvec64(&z0, &w_down, 4, 2);
        let c = matvec64(&xf, &w_ctx, 4, 2);
        let hc = [h[0], h[1], c[0], c[1]];
        let g_pre = matvec64(&hc, &w_gate, 4, 2);
        let c_pre = matvec64(&hc, &w_cand, 4, 2);
        let gated: Vec<f64> = (0..2)
            .map(|k| {
                let g = 1.0 / (1.0 + (-(g_pre[k] + b_gate[k] as f64)).exp());
                let c = (c_pre[k] + b_cand[k] as f64).tanh();
                g * c
            })
            .collect();
        let delta = matvec64(&gated, &w_up, 2, 4);
        let raw: Vec<f64> = z0
            .iter()
            .zip(delta.iter())
            .map(|(&z, &d)| z + 0.5 * d)
            .collect();
        let rmean = raw.iter().sum::<f64>() / 4.0;
        let rvar = raw.iter().map(|&v| (v - rmean).powi(2)).sum::<f64>() / 4.0;
        let expected: Vec<f64> = raw
            .iter()
            .map(|&v| (v - rmean) / (rvar + 1e-5f64).sqrt())
            .collect();

        for (a, b) in z_final.iter().zip(expected.iter()) {
            assert!((*a as f64 - b).abs() < 1e-4, "{a} vs {b}");
        }
        assert_eq!(telemetry.reflex_depth, 1);
        assert!(!telemetry.converged); // epsilon is tiny; one step never satisfies it here
    }

    // -- multi-step convergence / gamma telemetry shape --

    #[test]
    fn multi_step_telemetry_has_expected_shape() {
        let op = random_operator(32, 8, 6, 123);
        let mut rng = Xorshift32(555);
        let x = rng.vec(32);
        let (_z, telemetry) = op.forward(&x).unwrap();
        assert!(telemetry.reflex_depth >= 1 && telemetry.reflex_depth <= 6);
        assert_eq!(telemetry.residual_history.len(), telemetry.reflex_depth);
        assert_eq!(
            telemetry.gamma_history.len(),
            telemetry.reflex_depth.saturating_sub(1)
        );
        if telemetry.reflex_depth >= 2 {
            let max = telemetry
                .gamma_history
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(telemetry.max_gamma, Some(max));
        } else {
            assert_eq!(telemetry.max_gamma, None);
        }
        if telemetry.converged {
            assert!(telemetry.final_residual < op.config().epsilon);
        } else {
            assert_eq!(telemetry.reflex_depth, op.config().steps);
        }
    }

    // -- fail-closed validation --

    #[test]
    fn config_rejects_out_of_range_rank() {
        let bad = ReflexOperatorConfig {
            input_dim: 8,
            lora_rank: 8, // must be < input_dim
            alpha: 0.5,
            steps: 4,
            epsilon: 1e-4,
        };
        assert!(matches!(bad.validate(), Err(ModelError::ReflexConfig(_))));
    }

    #[test]
    fn config_rejects_alpha_out_of_range() {
        let bad = ReflexOperatorConfig {
            input_dim: 8,
            lora_rank: 2,
            alpha: 1.5,
            steps: 4,
            epsilon: 1e-4,
        };
        assert!(matches!(bad.validate(), Err(ModelError::ReflexConfig(_))));
    }

    #[test]
    fn operator_rejects_wrong_shape() {
        let config = ReflexOperatorConfig {
            input_dim: 4,
            lora_rank: 2,
            alpha: 0.5,
            steps: 1,
            epsilon: 1e-4,
        };
        let err = ReflexOperator::new(
            config,
            vec![0.0; 4 * 2],
            vec![0.0; 4 * 2],
            vec![0.0; 2 * 2 * 2],
            vec![0.0; 2],
            vec![0.0; 2 * 2 * 2],
            vec![0.0; 2],
            vec![0.0; 999], // wrong: should be r*d = 8
        )
        .unwrap_err();
        assert!(matches!(err, ModelError::ReflexArtifact(_)));
    }

    #[test]
    fn operator_rejects_non_finite_weight() {
        let config = ReflexOperatorConfig {
            input_dim: 4,
            lora_rank: 2,
            alpha: 0.5,
            steps: 1,
            epsilon: 1e-4,
        };
        let mut w_down = vec![0.0; 4 * 2];
        w_down[3] = f32::NAN;
        let err = ReflexOperator::new(
            config,
            w_down,
            vec![0.0; 4 * 2],
            vec![0.0; 2 * 2 * 2],
            vec![0.0; 2],
            vec![0.0; 2 * 2 * 2],
            vec![0.0; 2],
            vec![0.0; 2 * 4],
        )
        .unwrap_err();
        assert!(matches!(err, ModelError::ReflexArtifact(_)));
    }

    #[test]
    fn forward_rejects_wrong_length_input() {
        let op = zero_up_operator(8, 2, 4);
        let err = op.forward(&[0.0; 4]).unwrap_err();
        assert!(matches!(err, ModelError::ReflexInput(_)));
    }

    #[test]
    fn forward_rejects_non_finite_input() {
        let op = zero_up_operator(8, 2, 4);
        let mut x = vec![0.0f32; 8];
        x[0] = f32::INFINITY;
        let err = op.forward(&x).unwrap_err();
        assert!(matches!(err, ModelError::ReflexInput(_)));
    }

    #[test]
    fn plugin_predict_unknown_head_fails_closed() {
        let op = zero_up_operator(8, 2, 4);
        let head = ReflexHead::new("h", 8, vec![0.0; 8], vec![0.0], vec!["x".into()]).unwrap();
        let plugin = ReflexPlugin::new("t", op, vec![head], Some("h".into())).unwrap();
        let x = vec![0.1f32; 8];
        let err = plugin.predict(&x, Some("missing")).unwrap_err();
        assert!(matches!(err, ModelError::ReflexUnknownHead(_)));
    }

    #[test]
    fn plugin_with_no_default_head_requires_explicit_head() {
        let op = zero_up_operator(8, 2, 4);
        let head = ReflexHead::new("h", 8, vec![0.0; 8], vec![0.0], vec!["x".into()]).unwrap();
        let plugin = ReflexPlugin::new("t", op, vec![head], None).unwrap();
        let x = vec![0.1f32; 8];
        assert!(matches!(
            plugin.predict(&x, None).unwrap_err(),
            ModelError::ReflexUnknownHead(_)
        ));
        assert!(plugin.predict(&x, Some("h")).is_ok());
    }

    #[test]
    fn plugin_rejects_default_head_not_in_heads() {
        let op = zero_up_operator(8, 2, 4);
        let head = ReflexHead::new("h", 8, vec![0.0; 8], vec![0.0], vec!["x".into()]).unwrap();
        let err = ReflexPlugin::new("t", op, vec![head], Some("nope".into())).unwrap_err();
        assert!(matches!(err, ModelError::ReflexConfig(_)));
    }

    #[test]
    fn plugin_rejects_head_dimension_mismatch() {
        let op = zero_up_operator(8, 2, 4);
        // Head built for input_dim=4 while the operator is input_dim=8.
        let head = ReflexHead::new("h", 4, vec![0.0; 4], vec![0.0], vec!["x".into()]).unwrap();
        let err = ReflexPlugin::new("t", op, vec![head], None).unwrap_err();
        assert!(matches!(err, ModelError::ReflexConfig(_)));
    }

    // -- binary round trip and corruption handling --

    fn sample_plugin() -> ReflexPlugin {
        let op = random_operator(32, 6, 5, 321);
        let mut rng = Xorshift32(654);
        let head_a = ReflexHead::new(
            "a",
            32,
            rng.vec(3 * 32),
            rng.vec(3),
            vec!["x".into(), "y".into(), "z".into()],
        )
        .unwrap();
        let head_b = ReflexHead::new(
            "b",
            32,
            rng.vec(2 * 32),
            rng.vec(2),
            vec!["p".into(), "q".into()],
        )
        .unwrap();
        ReflexPlugin::new("multi", op, vec![head_a, head_b], Some("a".into())).unwrap()
    }

    #[test]
    fn binary_round_trip_preserves_predictions() {
        let plugin = sample_plugin();
        let bytes = plugin.to_bytes().unwrap();
        let loaded = ReflexPlugin::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.task(), plugin.task());
        assert_eq!(loaded.input_dim(), plugin.input_dim());

        let mut rng = Xorshift32(2024);
        let x = rng.vec(32);
        let (logits_orig, _) = plugin.forward(&x, Some("b")).unwrap();
        let (logits_loaded, _) = loaded.forward(&x, Some("b")).unwrap();
        for (a, b) in logits_orig.iter().zip(logits_loaded.iter()) {
            assert_eq!(a, b, "byte-for-byte round trip must be exact");
        }
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let plugin = sample_plugin();
        let mut bytes = plugin.to_bytes().unwrap();
        bytes[0] ^= 0xFF;
        let err = ReflexPlugin::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ModelError::ReflexArtifact(_)));
    }

    #[test]
    fn from_bytes_rejects_truncated_payload() {
        let plugin = sample_plugin();
        let bytes = plugin.to_bytes().unwrap();
        let truncated = &bytes[..bytes.len() - 4];
        let err = ReflexPlugin::from_bytes(truncated).unwrap_err();
        assert!(matches!(err, ModelError::ReflexArtifact(_)));
    }

    #[test]
    fn from_bytes_rejects_trailing_garbage() {
        let plugin = sample_plugin();
        let mut bytes = plugin.to_bytes().unwrap();
        bytes.extend_from_slice(&[0u8; 4]);
        let err = ReflexPlugin::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ModelError::ReflexArtifact(_)));
    }

    #[test]
    fn from_bytes_rejects_nan_weight() {
        let plugin = sample_plugin();
        let mut bytes = plugin.to_bytes().unwrap();
        // Corrupt the first weight float (w_down[0]) in the payload region.
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header_end = 16 + header_len;
        let payload_start = header_end + (4 - header_end % 4) % 4;
        bytes[payload_start..payload_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let err = ReflexPlugin::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ModelError::ReflexArtifact(_)));
    }

    #[test]
    fn from_bytes_rejects_unknown_format_string() {
        let plugin = sample_plugin();
        let bytes = plugin.to_bytes().unwrap();
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header_end = 16 + header_len;
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[16..header_end]).unwrap();
        header["format"] = serde_json::Value::String("bogus".into());
        let new_header_bytes = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(ARCHIVE_MAGIC);
        out.extend_from_slice(&(new_header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&new_header_bytes);
        let pad = (4 - out.len() % 4) % 4;
        out.resize(out.len() + pad, 0);
        out.extend_from_slice(&bytes[header_end + (4 - header_end % 4) % 4..]);
        let err = ReflexPlugin::from_bytes(&out).unwrap_err();
        assert!(matches!(err, ModelError::ReflexArtifact(_)));
    }

    // -- production scale: D=8192, a supported rank --

    #[test]
    fn production_scale_8192_dim_runs_and_converges_or_exhausts_budget() {
        let op = random_operator(8192, 64, 4, 8192);
        let mut rng = Xorshift32(31415);
        let x = rng.vec(8192);
        let start = std::time::Instant::now();
        let (z_final, telemetry) = op.forward(&x).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(z_final.len(), 8192);
        assert!(telemetry.reflex_depth >= 1 && telemetry.reflex_depth <= 4);
        // Debug builds are not representative of production latency; only
        // bound wall clock in release mode, and keep the bound generous
        // (this asserts "fast enough to be a reflex", not a tight SLA).
        if !cfg!(debug_assertions) {
            assert!(
                elapsed.as_millis() < 50,
                "8192-d forward took {elapsed:?}, expected well under a reflex budget"
            );
        }
    }
}
