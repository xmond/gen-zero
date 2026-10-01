//! Spec 25 §1.2 / §5.1 / §5.5: the one cognitive runtime behind every entry.
//!
//! Flow of one geometric request (stage S1 of §5.6, fixed fiber):
//!
//! 1. The caller already captured one [`MountSnapshot`] (see
//!    [`crate::mount::RequestBinding`]). Everything below reads that snapshot
//!    only; nothing is re-loaded mid-request.
//! 2. [`CognitiveAssets::from_snapshot`] decodes the geometry, the SSM
//!    generator and the gate policy that the snapshot seals. A snapshot with
//!    no assets is `BackendUnavailable`, never a default model.
//! 3. [`PoincareBall`] checks the state is inside the ball and maps it to the
//!    tangent space at the origin (`log_0`).
//! 4. [`crate::tangent_ssm::ZohTangentSsm`] runs the parallel prefix scan.
//! 5. the shared gate kernel relaxes the dynamics sheaf energy
//!    `L(s) = 1/2 sum_t |s_t - M_t s_{t-1} - q_t|^2 + w/2 sum_pins |s_t - y_t|^2`
//!    by a bounded gradient flow and certifies it with interval bounds.
//! 6. [`ActionVerifier`] accepts a candidate only if its geodesic energy to the
//!    goal strictly falls (`upper(after) < lower(before)`), and emits a
//!    [`CertifiedAction`].
//!
//! Entailment (scheme 1): a snapshot whose assets carry an `entailment` block
//! also seals one product geometry from the closed [`TopologyPreset`]
//! whitelist (`compact_64d`, `balanced_128d`, `boolq_128d`, `extended_256d`).
//! The preset is frozen into the mount digest at publish time; there is no
//! run-time width change. [`CognitiveRuntime::evaluate_entailment`] refuses
//! inputs whose width is not the sealed `dim()` (`FiberMismatch`), then runs
//! the asymmetric Busemann containment test of `gen-zero-lod`.
//!
//! Entailment (scheme 2, fiber cross-difference SSM): when the request also
//! carries `passage_events` and `question_events` and the mount seals
//! `entailment.dynamics`, [`CognitiveRuntime::evaluate_entailment`] first
//! 1. builds the gauge `Gamma_{p->q}` as the Levi-Civita parallel transport
//!    of the product geometry along the geodesic `p -> q` (composed with the
//!    tangent projection at `p`), materialized as a `dim x dim` matrix;
//! 2. forms `v_delta_t = v_q,t - Gamma v_p,t` with
//!    [`crate::tangent_ssm::FiberCrossDiff`] and scans it as ZOH input on the
//!    question fiber from `h_0 = 0`;
//! 3. moves the question point to `q' = exp_q(h_T)` and runs the Busemann
//!    test on `(p, q')`. The unfiltered verdict on `(p, q)` is reported next
//!    to it for comparison, never instead of it.
//!
//! Events without sealed dynamics are refused (`BackendUnavailable`); they
//! are never dropped to run the scheme 1 test.
//!
//! What this module is NOT (say so, do not imply it):
//! - No trained atlas or projection. The SSM path uses one Poincare factor,
//!   tangent space at the origin, identity transport. The generator comes from
//!   whatever assets an operator mounted; nothing here is learned.
//! - No trained entailment model. The Busemann test is closed-form geometry on
//!   coordinates the caller supplies; its thresholds are presets and its
//!   `confidence` is a margin, not a calibrated probability.
//! - No trained text encoder. The `zero` router can project request text
//!   into a 128-dimensional point by feature hashing and scan it here when
//!   the mount is 128-dimensional (`_meta.cognitive_runtime`). That hash is
//!   deterministic and untrained. Entailment takes numeric points only.
//! - No trained fiber dynamics. `entailment.dynamics` is an operator-stated
//!   generator; the gauge is derived from the sealed geometry, not learned.
//!   Nothing shows yet that the cross-difference improves any benchmark.

use crate::mount::{digest_hex, Digest, MountSnapshot, Reject};
use crate::tangent_ssm::{
    Backend, Event, FiberCrossDiff, FiberId, FrozenContext, Matrix, ParallelTangentSsm,
    PreparedWindow, ScanBudget, ScanEvidence, ScanOutput, Tangent, ZohTangentSsm,
};
use gen_zero_lod::{
    ContainmentScore, Epochs as LodEpochs, GeometryParams, Layout, LodError, Point as LodPoint,
    ProductGeometry, ProductManifold, Reject as LodReject, TopologyPreset, Version as LodVersion,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest as Sha2Digest, Sha256};
use std::str::FromStr;

pub const ASSETS_SCHEMA: &str = "gen-zero/cognitive-assets/v1";
pub const ENGINE_COGNITIVE: &str = "cognitive_runtime";
/// Largest state width (`dim`) and input width (`generator.b` columns) of
/// the gated cognitive SSM. The entailment geometry is not bound by it: its
/// width is the sealed topology preset (64, 128 or 256).
pub const MAX_DIM: usize = 128;
pub const MAX_WINDOW: usize = 4096;
/// Window limit of the fiber path. Each step there exponentiates a dense
/// 256 x 256 generator, so it gets a smaller budget than `MAX_WINDOW`.
pub const MAX_FIBER_WINDOW: usize = 256;
pub const MAX_GATE_STEPS: usize = 100_000;
pub const MAX_THREADS: usize = 64;

const DOMAIN_CERT: &[u8] = b"gen-zero/certified-action/v1\0";
const DOMAIN_FIBER: &[u8] = b"gen-zero/fiber/origin-identity/v1\0";
const DOMAIN_CROSS: &[u8] = b"gen-zero/fiber/cross-diff/v1\0";
/// Relative tolerance (to `1 + |v|_inf`) for "this vector is tangent": the
/// distance between a vector and its tangent projection.
const TANGENT_TOL: f64 = 1e-9;

fn sha256(parts: &[&[u8]]) -> Digest {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// Typed refusal of one request. `code` is a §5.1 [`Reject`] name, or
/// `InvalidParams` when the request does not describe a valid problem.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Rejection {
    pub code: String,
    pub stage: String,
    pub detail: String,
    pub http_status: u16,
}

impl Rejection {
    pub fn reject(code: Reject, stage: &str, detail: impl Into<String>) -> Self {
        Self {
            code: code.code().to_string(),
            stage: stage.to_string(),
            detail: detail.into(),
            http_status: code.http_status(),
        }
    }

    pub fn invalid(stage: &str, detail: impl Into<String>) -> Self {
        Self {
            code: "InvalidParams".to_string(),
            stage: stage.to_string(),
            detail: detail.into(),
            http_status: 400,
        }
    }
}

type Outcome<T> = std::result::Result<T, Rejection>;

fn at(stage: &'static str) -> impl Fn(Reject) -> Rejection {
    move |r| Rejection::reject(r, stage, r.to_string())
}

// ---------------------------------------------------------------- assets

/// Gate contract sealed in the mount (§5.1 "versioned step-size contract").
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GatePolicy {
    pub max_steps: usize,
    pub residual_limit: f64,
    /// Half-width of the energy interval, relative: `h = numeric_error * (1 + L)`.
    pub numeric_error: f64,
    /// Euler step as a fraction of `2 / lambda_max` bound, so `(0, 2)` is the
    /// provably non-increasing range. Checked when the asset is mounted.
    pub step_fraction: f64,
    pub pin_weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ScanPolicy {
    max_growth: f64,
    max_state_norm: f64,
    /// 0 means "all available cores".
    threads: usize,
}

/// Sealed preset, parameters and cone cap of the entailment geometry. Nothing
/// here is fitted; an operator states the numbers and they are versioned with
/// the mount.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct EntailmentWire {
    topology: String,
    curvature: f64,
    radius: f64,
    alpha_h: f64,
    alpha_e: f64,
    alpha_s: f64,
    cone_half_angle: f64,
    /// Optional fiber dynamics `dh/dt = A h + B v_delta` on the question
    /// fiber (scheme 2), `dim x dim` for the sealed preset. Absent: requests with event streams are
    /// refused. Serialized only when present, so older assets keep their
    /// geometry digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dynamics: Option<Generator>,
}

impl EntailmentWire {
    fn params(&self) -> GeometryParams {
        GeometryParams {
            curvature: self.curvature,
            radius: self.radius,
            alpha_h: self.alpha_h,
            alpha_e: self.alpha_e,
            alpha_s: self.alpha_s,
        }
    }

    /// The whitelisted preset `topology` names. Anything else is refused.
    fn preset(&self) -> Outcome<TopologyPreset> {
        TopologyPreset::from_str(&self.topology).map_err(|_| {
            Rejection::invalid(
                "assets",
                format!(
                    "unsupported topology preset {:?}; expected one of compact_64d, \
                     balanced_128d, boolq_128d, extended_256d",
                    self.topology
                ),
            )
        })
    }

    fn validate(&self) -> Outcome<()> {
        let bad = |msg: String| Err(Rejection::invalid("assets", msg));
        self.preset()?;
        self.params().validate().map_err(|_| {
            Rejection::invalid(
                "assets",
                "entailment curvature, radius and alpha weights must be finite and > 0, \
                 and alpha_h + alpha_e + alpha_s must be finite",
            )
        })?;
        let a = self.cone_half_angle;
        if !(a.is_finite() && a > 0.0 && a <= std::f64::consts::PI) {
            return bad("entailment.cone_half_angle must be finite and in (0, pi]".into());
        }
        Ok(())
    }

    /// `entailment.dynamics` as matrices: `A` and `B` both `dim x dim` of the
    /// sealed preset (64, 128 or 256).
    fn fiber_dynamics(&self) -> Outcome<Option<(Matrix, Matrix)>> {
        let Some(g) = &self.dynamics else {
            return Ok(None);
        };
        let preset = self.preset()?;
        let n = preset.dim();
        let bad = |msg: String| Rejection::invalid("assets", msg);
        let a = Matrix::from_rows(&g.a).map_err(|r| bad(format!("entailment.dynamics.a: {r}")))?;
        let b = Matrix::from_rows(&g.b).map_err(|r| bad(format!("entailment.dynamics.b: {r}")))?;
        if a.rows() != n || a.cols() != n {
            return Err(bad(format!(
                "entailment.dynamics.a must be {n} x {n} for topology {preset}, got {} x {}",
                a.rows(),
                a.cols()
            )));
        }
        if b.rows() != n || b.cols() != n {
            return Err(bad(format!(
                "entailment.dynamics.b must be {n} x {n} for topology {preset}, got {} x {}: \
                 its input is the cross-difference tangent",
                b.rows(),
                b.cols()
            )));
        }
        Ok(Some((a, b)))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Generator {
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct AssetsWire {
    schema: String,
    /// Who produced these numbers and how. Free text, bound into the digest.
    provenance: String,
    dim: usize,
    curvature: f64,
    generator: Generator,
    gate: GatePolicy,
    scan: ScanPolicy,
    decision_temperature: f64,
    /// Optional entailment geometry. Absent means "not mounted": entailment
    /// requests are refused, never served from a default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    entailment: Option<EntailmentWire>,
}

/// Decoded, validated assets of one snapshot.
#[derive(Clone, Debug)]
pub struct CognitiveAssets {
    wire: AssetsWire,
    a: Matrix,
    b: Matrix,
    ball: PoincareBall,
    /// `entailment.dynamics`, decoded.
    fiber: Option<(Matrix, Matrix)>,
}

/// Content digests of the three asset families, for the mount epochs.
pub struct AssetFamilies {
    pub model: Digest,
    pub geometry: Digest,
    pub policy: Digest,
}

fn finite(x: f64) -> bool {
    x.is_finite()
}

impl CognitiveAssets {
    /// Parse and validate a JSON asset document.
    pub fn from_json(value: &Value) -> Outcome<Self> {
        let wire: AssetsWire = serde_json::from_value(value.clone())
            .map_err(|e| Rejection::invalid("assets", format!("asset schema: {e}")))?;
        Self::validate(wire)
    }

    /// Decode the assets a snapshot seals. Empty assets mean no geometry was
    /// ever mounted for this key: the runtime has nothing to run.
    pub fn from_snapshot(snapshot: &MountSnapshot) -> Outcome<Self> {
        if snapshot.assets().is_empty() {
            return Err(Rejection::reject(
                Reject::BackendUnavailable,
                "assets",
                format!(
                    "mount version {} carries no cognitive assets; publish them with \
                     POST /v1/mounts or start with --mount-assets",
                    snapshot.version().0
                ),
            ));
        }
        let wire: AssetsWire = serde_json::from_slice(snapshot.assets()).map_err(|e| {
            Rejection::reject(
                Reject::InvalidCertificate,
                "assets",
                format!("sealed assets do not decode: {e}"),
            )
        })?;
        Self::validate(wire)
    }

    fn validate(mut wire: AssetsWire) -> Outcome<Self> {
        let bad = |msg: String| Err(Rejection::invalid("assets", msg));
        if wire.schema != ASSETS_SCHEMA {
            return bad(format!("schema must be {ASSETS_SCHEMA}"));
        }
        if wire.provenance.trim().is_empty() {
            return bad("provenance must say where the numbers come from".into());
        }
        if wire.dim == 0 || wire.dim > MAX_DIM {
            return bad(format!("dim must be in 1..={MAX_DIM}"));
        }
        if !(finite(wire.curvature) && wire.curvature > 0.0) {
            return bad("curvature must be finite and > 0 (ball of radius 1/sqrt(c))".into());
        }
        let g = &wire.gate;
        if g.max_steps == 0 || g.max_steps > MAX_GATE_STEPS {
            return bad(format!("gate.max_steps must be in 1..={MAX_GATE_STEPS}"));
        }
        if !(finite(g.residual_limit) && g.residual_limit >= 0.0) {
            return bad("gate.residual_limit must be finite and >= 0".into());
        }
        if !(finite(g.numeric_error) && g.numeric_error > 0.0 && g.numeric_error < 1e-3) {
            return bad("gate.numeric_error must be in (0, 1e-3)".into());
        }
        if !(finite(g.step_fraction) && g.step_fraction > 0.0 && g.step_fraction < 2.0) {
            return bad("gate.step_fraction must be in (0, 2), the stable Euler range".into());
        }
        if !(finite(g.pin_weight) && g.pin_weight > 0.0) {
            return bad("gate.pin_weight must be finite and > 0".into());
        }
        let s = &wire.scan;
        if !(finite(s.max_growth) && s.max_growth >= 1.0) {
            return bad("scan.max_growth must be finite and >= 1".into());
        }
        if !(finite(s.max_state_norm) && s.max_state_norm > 0.0) {
            return bad("scan.max_state_norm must be finite and > 0".into());
        }
        if s.threads > MAX_THREADS {
            return bad(format!("scan.threads must be <= {MAX_THREADS}"));
        }
        if !(finite(wire.decision_temperature) && wire.decision_temperature > 0.0) {
            return bad("decision_temperature must be finite and > 0".into());
        }
        let fiber = match &mut wire.entailment {
            Some(e) => {
                e.validate()?;
                // Seal the canonical name: `deep_128d` and `boolq_128d` are one
                // geometry and must give one digest.
                e.topology = e.preset()?.as_str().to_string();
                e.fiber_dynamics()?
            }
            None => None,
        };
        let a = Matrix::from_rows(&wire.generator.a)
            .map_err(|r| Rejection::invalid("assets", format!("generator.a: {r}")))?;
        let b = Matrix::from_rows(&wire.generator.b)
            .map_err(|r| Rejection::invalid("assets", format!("generator.b: {r}")))?;
        if a.rows() != wire.dim || a.cols() != wire.dim {
            return bad("generator.a must be dim x dim".into());
        }
        if b.rows() != wire.dim || b.cols() == 0 || b.cols() > MAX_DIM {
            return bad(format!(
                "generator.b must be dim x m with m in 1..={MAX_DIM}"
            ));
        }
        if a.as_slice()
            .iter()
            .chain(b.as_slice())
            .any(|x| !x.is_finite())
        {
            return bad("generator entries must be finite".into());
        }
        let ball = PoincareBall::new(wire.curvature, wire.dim);
        Ok(Self {
            wire,
            a,
            b,
            ball,
            fiber,
        })
    }

    /// Canonical bytes: the validated document re-serialized, so the digest
    /// does not depend on the caller's whitespace or key order.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.wire).expect("asset wire types always serialize")
    }

    pub fn families(&self) -> AssetFamilies {
        let enc = |v: Value| serde_json::to_vec(&v).expect("json values always serialize");
        let w = &self.wire;
        AssetFamilies {
            model: sha256(&[
                b"gen-zero/assets/model\0",
                &enc(json!({"generator": w.generator, "provenance": w.provenance})),
            ]),
            geometry: {
                let mut geo = json!({"dim": w.dim, "curvature": w.curvature});
                // Only when mounted, so assets without it keep their digest.
                if let Some(e) = &w.entailment {
                    geo["entailment"] = json!(e);
                }
                sha256(&[b"gen-zero/assets/geometry\0", &enc(geo)])
            },
            policy: sha256(&[
                b"gen-zero/assets/policy\0",
                &enc(json!({
                    "gate": w.gate, "scan": w.scan,
                    "decision_temperature": w.decision_temperature,
                })),
            ]),
        }
    }

    pub fn gate(&self) -> &GatePolicy {
        &self.wire.gate
    }

    pub fn dim(&self) -> usize {
        self.wire.dim
    }

    pub fn input_dim(&self) -> usize {
        self.b.cols()
    }

    pub fn summary(&self) -> Value {
        json!({
            "schema": self.wire.schema,
            "provenance": self.wire.provenance,
            "dim": self.wire.dim,
            "input_dim": self.b.cols(),
            "curvature": self.wire.curvature,
            "entailment": self.wire.entailment.as_ref().map(|e| e.topology.as_str()),
            "entailment_dim": self
                .wire
                .entailment
                .as_ref()
                .and_then(|e| e.preset().ok())
                .map(|p| p.dim()),
            "entailment_dynamics": self.fiber.is_some(),
            "trained": false,
        })
    }
}

// ---------------------------------------------------------------- geometry

/// Poincare ball of curvature `-c`, radius `1/sqrt(c)`. Charts at the origin.
#[derive(Clone, Copy, Debug)]
pub struct PoincareBall {
    sqrt_c: f64,
    dim: usize,
}

/// Euclidean norm, scaled by the largest magnitude coordinate first so that
/// `x*x` never overflows for a finite input: naive `sum(x*x).sqrt()`
/// overflows to `inf` once any `|x| > ~1.34e154`, and `inf` silently folds
/// downstream computations (e.g. `exp0`'s `tanh(inf)/inf = 0`) to the
/// origin instead of being rejected.
fn norm(v: &[f64]) -> f64 {
    let max_abs = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    if max_abs == 0.0 {
        return 0.0;
    }
    let scaled_sum_sq: f64 = v
        .iter()
        .map(|x| {
            let s = x / max_abs;
            s * s
        })
        .sum();
    max_abs * scaled_sum_sq.sqrt()
}

/// Radius `sqrt(c)*|x|` at which the critical boundary zone starts. A state
/// at or past it needs a human decision, so the geometry gate rejects it.
pub const SAFETY_RADIUS: f64 = 0.85;

impl PoincareBall {
    /// Geometric safety radius: reject a ball point in the critical boundary
    /// zone, `sqrt(c)*|x| >= SAFETY_RADIUS`.
    pub fn check_safety_radius(&self, x: &[f64], what: &str) -> Outcome<()> {
        let radius = self.sqrt_c * norm(x);
        // A non-finite radius must not pass the comparison.
        if radius < SAFETY_RADIUS {
            return Ok(());
        }
        Err(Rejection::reject(
            Reject::DomainViolation,
            "geometry_gate",
            format!(
                "{what} is in the critical boundary zone: sqrt(c)*|x| = {radius} >= \
                 {SAFETY_RADIUS}; human intervention is required"
            ),
        ))
    }

    pub fn new(curvature: f64, dim: usize) -> Self {
        Self {
            sqrt_c: curvature.sqrt(),
            dim,
        }
    }

    /// A point must have the layout width, be finite, and lie strictly inside
    /// the ball. `sqrt(c) |x| >= 1` is on or past the ideal boundary.
    pub fn check_point(&self, x: &[f64], what: &str) -> Outcome<()> {
        if x.len() != self.dim {
            return Err(Rejection::reject(
                Reject::FiberMismatch,
                "domain",
                format!(
                    "{what} has width {}, the mounted layout is {}",
                    x.len(),
                    self.dim
                ),
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(Rejection::reject(
                Reject::NonFiniteState,
                "domain",
                format!("{what} has a non-finite coordinate"),
            ));
        }
        let scaled = self.sqrt_c * norm(x);
        if scaled >= 1.0 {
            return Err(Rejection::reject(
                Reject::DomainViolation,
                "domain",
                format!("{what} is outside the Poincare ball: sqrt(c)*|x| = {scaled} >= 1"),
            ));
        }
        Ok(())
    }

    /// `log_0(x) = artanh(sqrt(c)|x|) x / (sqrt(c)|x|)`.
    pub fn log0(&self, x: &[f64], what: &str) -> Outcome<Vec<f64>> {
        self.check_point(x, what)?;
        let n = norm(x);
        if n == 0.0 {
            return Ok(x.to_vec());
        }
        let k = (self.sqrt_c * n).atanh() / (self.sqrt_c * n);
        let v: Vec<f64> = x.iter().map(|xi| xi * k).collect();
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Rejection::reject(
                Reject::NonFiniteState,
                "log0",
                what.to_string(),
            ));
        }
        Ok(v)
    }

    /// `exp_0(v) = tanh(sqrt(c)|v|) v / (sqrt(c)|v|)`. When `tanh` rounds to 1
    /// the point is on the boundary in f64: that is a domain violation, not a
    /// point we can return. A tangent vector is unbounded input (unlike a
    /// ball point), so every intermediate value here is checked for finite
    /// and sub-boundary *before* it is used to scale `v`: a huge `|v|` must
    /// be rejected outright, never silently folded to the origin by a
    /// division that has gone non-finite.
    pub fn exp0(&self, v: &[f64], what: &str) -> Outcome<Vec<f64>> {
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Rejection::reject(
                Reject::NonFiniteState,
                "exp0",
                what.to_string(),
            ));
        }
        let n = norm(v);
        if !n.is_finite() {
            return Err(Rejection::reject(
                Reject::NonFiniteState,
                "exp0",
                format!("{what}: |v| overflowed to a non-finite value"),
            ));
        }
        if n == 0.0 {
            return Ok(v.to_vec());
        }
        let scaled_n = self.sqrt_c * n;
        if !scaled_n.is_finite() {
            return Err(Rejection::reject(
                Reject::NonFiniteState,
                "exp0",
                format!("{what}: sqrt(c)*|v| overflowed to a non-finite value"),
            ));
        }
        let t = scaled_n.tanh();
        if t >= 1.0 {
            return Err(Rejection::reject(
                Reject::DomainViolation,
                "exp0",
                format!(
                    "{what}: tangent vector maps to or past the Poincare ball boundary: \
                     tanh(sqrt(c)*|v|) = {t} >= 1"
                ),
            ));
        }
        let k = t / scaled_n;
        let x: Vec<f64> = v.iter().map(|vi| vi * k).collect();
        self.check_point(&x, what)?;
        Ok(x)
    }

    /// Geodesic distance
    /// `arcosh(1 + 2c|x-y|^2 / ((1-c|x|^2)(1-c|y|^2))) / sqrt(c)`.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> Outcome<f64> {
        self.check_point(x, "distance lhs")?;
        self.check_point(y, "distance rhs")?;
        let c = self.sqrt_c * self.sqrt_c;
        let diff: f64 = x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum();
        let dx = 1.0 - c * x.iter().map(|a| a * a).sum::<f64>();
        let dy = 1.0 - c * y.iter().map(|a| a * a).sum::<f64>();
        if dx <= 0.0 || dy <= 0.0 {
            return Err(Rejection::reject(
                Reject::DomainViolation,
                "distance",
                "conformal factor is not positive",
            ));
        }
        let d = (1.0 + 2.0 * c * diff / (dx * dy)).acosh() / self.sqrt_c;
        if !d.is_finite() {
            return Err(Rejection::reject(
                Reject::NonFiniteState,
                "distance",
                "non-finite",
            ));
        }
        Ok(d)
    }
}

// ---------------------------------------------------------------- gate

/// Interval `[lo, hi]` around a computed energy.
fn interval(l: f64, rel: f64) -> [f64; 2] {
    let h = rel * (1.0 + l.abs());
    [l - h, l + h]
}

/// Report of one gate run, for the request trace.
#[derive(Clone, Debug, Serialize)]
pub struct GateReport {
    pub energy_before: [f64; 2],
    pub energy_after: [f64; 2],
    pub residual: f64,
    pub residual_limit: f64,
    pub steps: usize,
    pub eta: f64,
    pub lambda_bound: f64,
    /// `within_budget` or `strictly_decreased` for accepted candidates;
    /// `stalled` or `step_budget_exhausted` for non-converged diagnostics.
    pub status: &'static str,
}

/// Borrowed service input converted into the gate crate's validated operator.
struct WindowGateInput<'a> {
    steps: &'a PreparedWindow,
    s0: Vec<f64>,
    pins: Vec<(usize, Vec<f64>)>,
    pin_weight: f64,
}

pub type GateResult =
    std::result::Result<(Vec<Vec<f64>>, GateReport), (Reject, String, Option<GateReport>)>;

fn certify_window(
    problem: &WindowGateInput<'_>,
    candidate: Vec<Vec<f64>>,
    policy: &GatePolicy,
) -> GateResult {
    use gen_zero_gate::sheaf_gate::{
        Budget, CandidateState, DynamicsStep, LaplacianHeatFlowGate, RelaxationStatus,
        SheafOperator, WindowDynamicsProblem,
    };
    let map_error = |e: gen_zero_gate::Reject| {
        let code = match &e {
            gen_zero_gate::Reject::EnergyRose { .. } => Reject::EnergyRose,
            gen_zero_gate::Reject::UncertainEnergy { .. } => Reject::UncertainEnergy,
            gen_zero_gate::Reject::PinnedMoved { .. } => Reject::PinnedMoved,
            gen_zero_gate::Reject::NonFiniteState { .. } => Reject::NonFiniteState,
            gen_zero_gate::Reject::FiberMismatch { .. } => Reject::FiberMismatch,
            _ => Reject::DomainViolation,
        };
        (code, e.to_string(), None)
    };
    let steps = problem
        .steps
        .steps()
        .iter()
        .map(|step| {
            let m = step.linear();
            DynamicsStep {
                matrix: (0..m.rows())
                    .map(|i| (0..m.cols()).map(|j| m.get(i, j)).collect())
                    .collect(),
                bias: step.bias().to_vec(),
            }
        })
        .collect();
    let operator = WindowDynamicsProblem::new(
        problem.s0.clone(),
        steps,
        problem.pins.clone(),
        problem.pin_weight,
        0,
    )
    .map_err(map_error)?;
    if candidate.len() != problem.steps.steps().len()
        || candidate.iter().any(|s| s.len() != problem.s0.len())
    {
        return Err((
            Reject::FiberMismatch,
            "candidate window shape mismatch".into(),
            None,
        ));
    }
    let max_steps = u32::try_from(policy.max_steps).map_err(|_| {
        (
            Reject::BudgetExceeded,
            "gate step budget exceeds u32".into(),
            None,
        )
    })?;
    if !policy.residual_limit.is_finite() || policy.residual_limit < 0.0 {
        return Err((
            Reject::DomainViolation,
            "energy residual limit must be finite and nonnegative".into(),
            None,
        ));
    }
    let lambda_bound = 2.0 / operator.safe_eta_bound();
    let budget = Budget {
        eta: Some(policy.step_fraction / lambda_bound),
        max_steps,
        // Service policy is an energy threshold, core policy is a residual norm.
        residual_tol: policy.residual_limit.sqrt() * std::f64::consts::SQRT_2,
        energy_uncertainty_tol: policy.numeric_error,
        ..Budget::default()
    };
    let c = LaplacianHeatFlowGate
        .certify_and_relax(
            &operator,
            CandidateState {
                s: candidate.into_iter().flatten().collect(),
                epoch: 0,
            },
            &budget,
        )
        .map_err(map_error)?;
    let mut report = GateReport {
        energy_before: interval(c.initial_energy, policy.numeric_error),
        energy_after: interval(c.final_energy, policy.numeric_error),
        residual: c.final_energy,
        residual_limit: policy.residual_limit,
        steps: c.steps_taken as usize,
        eta: c.eta_used,
        lambda_bound,
        status: match c.status {
            RelaxationStatus::Stalled => "stalled",
            RelaxationStatus::StepBudgetExhausted => "step_budget_exhausted",
            RelaxationStatus::Converged if c.steps_taken == 0 => "within_budget",
            RelaxationStatus::Converged => "pending_interval_check",
        },
    };
    if !report
        .energy_before
        .iter()
        .chain(&report.energy_after)
        .all(|x| x.is_finite())
    {
        return Err((
            Reject::NonFiniteState,
            "non-finite energy interval".into(),
            Some(report),
        ));
    }
    let rejection = match c.status {
        RelaxationStatus::Stalled if c.steps_taken == 0 => Some(Reject::Stalled),
        RelaxationStatus::Stalled => Some(Reject::ResidualExceeded),
        RelaxationStatus::StepBudgetExhausted => Some(Reject::NotConverged),
        RelaxationStatus::Converged => None,
    };
    if let Some(code) = rejection {
        return Err((
            code,
            format!(
                "window relaxation {:?}: energy={}, residual_norm={}, gradient_norm={}",
                c.status, c.final_energy, c.residual_norm, c.grad_norm
            ),
            Some(report),
        ));
    }
    if c.final_energy > policy.residual_limit {
        report.status = "residual_exceeded";
        return Err((
            Reject::ResidualExceeded,
            "energy exceeds service threshold".into(),
            Some(report),
        ));
    }
    if c.steps_taken > 0 && report.energy_after[1] >= report.energy_before[0] {
        report.status = "uncertain_energy";
        return Err((
            Reject::UncertainEnergy,
            "decrease is inside the error interval".into(),
            Some(report),
        ));
    }
    if c.steps_taken > 0 {
        report.status = "strictly_decreased";
    }
    Ok((c.states, report))
}

// ---------------------------------------------------------------- runtime

/// Owned, parsed geometric request. Built on the async side, run on a
/// blocking thread.
#[derive(Clone, Debug)]
pub struct TrajectoryRequest {
    state: Vec<f64>,
    window_start_ns: u64,
    pins: Vec<(usize, Vec<f64>)>,
    backend: Option<Backend>,
    digest: Digest,
}

/// Parsed `cognitive` block of an `ask`: one control window per candidate.
#[derive(Clone, Debug)]
pub struct DecisionRequest {
    base: TrajectoryRequest,
    goal: Vec<f64>,
    controls: Vec<(String, Vec<Event>)>,
}

fn f64_vec(v: Option<&Value>, field: &str) -> Outcome<Vec<f64>> {
    let arr = v.and_then(Value::as_array).ok_or_else(|| {
        Rejection::invalid("request", format!("{field} must be an array of numbers"))
    })?;
    arr.iter()
        .map(|x| {
            x.as_f64().ok_or_else(|| {
                Rejection::invalid("request", format!("{field} must hold only numbers"))
            })
        })
        .collect()
}

fn parse_events(v: Option<&Value>, field: &str) -> Outcome<Vec<Event>> {
    let arr = v.and_then(Value::as_array).ok_or_else(|| {
        Rejection::invalid(
            "request",
            format!("{field} must be an array of {{time_ns, input}}"),
        )
    })?;
    if arr.is_empty() {
        return Err(Rejection::invalid("request", format!("{field} is empty")));
    }
    if arr.len() > MAX_WINDOW {
        return Err(Rejection::reject(
            Reject::BudgetExceeded,
            "request",
            format!("{field} has {} events > {MAX_WINDOW}", arr.len()),
        ));
    }
    arr.iter()
        .enumerate()
        .map(|(i, ev)| {
            let time_ns = ev.get("time_ns").and_then(Value::as_u64).ok_or_else(|| {
                Rejection::invalid("request", format!("{field}[{i}].time_ns must be a u64"))
            })?;
            let input = f64_vec(ev.get("input"), &format!("{field}[{i}].input"))?;
            let mut hasher = blake3::Hasher::new();
            hasher.update(&(i as u64).to_le_bytes());
            hasher.update(&time_ns.to_le_bytes());
            for x in &input {
                hasher.update(&x.to_le_bytes());
            }
            Ok(Event {
                id: *hasher.finalize().as_bytes(),
                time_ns,
                input: input.into_boxed_slice(),
            })
        })
        .collect()
}

fn parse_backend(v: Option<&Value>) -> Outcome<Option<Backend>> {
    match v.map(Value::as_str) {
        None => Ok(None),
        Some(Some("cpu_parallel")) => Ok(None),
        Some(Some("cpu_serial")) => Ok(Some(Backend::CpuSerial)),
        Some(Some("gpu")) => Ok(Some(Backend::Gpu { device: 0 })),
        _ => Err(Rejection::invalid(
            "request",
            "backend must be one of cpu_parallel, cpu_serial, gpu",
        )),
    }
}

fn parse_base(block: &Value) -> Outcome<TrajectoryRequest> {
    if !block.is_object() {
        return Err(Rejection::invalid("request", "cognitive must be an object"));
    }
    let state = f64_vec(block.get("state"), "cognitive.state")?;
    let window_start_ns = block
        .get("window_start_ns")
        .and_then(Value::as_u64)
        .ok_or_else(|| Rejection::invalid("request", "cognitive.window_start_ns must be a u64"))?;
    let mut pins = Vec::new();
    if let Some(p) = block.get("pins") {
        let arr = p
            .as_array()
            .ok_or_else(|| Rejection::invalid("request", "cognitive.pins must be an array"))?;
        for (i, pin) in arr.iter().enumerate() {
            let step = pin.get("step").and_then(Value::as_u64).ok_or_else(|| {
                Rejection::invalid("request", format!("cognitive.pins[{i}].step must be >= 1"))
            })? as usize;
            let point = f64_vec(pin.get("point"), &format!("cognitive.pins[{i}].point"))?;
            pins.push((step, point));
        }
    }
    let backend = parse_backend(block.get("backend"))?;
    let canonical = serde_json::to_vec(block).expect("json values always serialize");
    Ok(TrajectoryRequest {
        state,
        window_start_ns,
        pins,
        backend,
        digest: sha256(&[b"gen-zero/cognitive-request/v1\0", &canonical]),
    })
}

/// `stream`: `{"cognitive": {"state", "window_start_ns", "events", "pins"?}}`.
pub fn parse_stream(block: &Value) -> Outcome<(TrajectoryRequest, Vec<Event>)> {
    let base = parse_base(block)?;
    let events = parse_events(block.get("events"), "cognitive.events")?;
    Ok((base, events))
}

/// `ask`: `{"cognitive": {"state", "goal", "window_start_ns",
/// "controls": {"<candidate>": [events]}, "pins"?}}`. Every feasible
/// candidate needs a control window: a candidate the runtime cannot simulate
/// is an input error, never skipped.
pub fn parse_decision(block: &Value, candidates: &[String]) -> Outcome<DecisionRequest> {
    let base = parse_base(block)?;
    let goal = f64_vec(block.get("goal"), "cognitive.goal")?;
    let controls_obj = block
        .get("controls")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            Rejection::invalid(
                "request",
                "cognitive.controls must map each candidate to events",
            )
        })?;
    let mut controls = Vec::with_capacity(candidates.len());
    for name in candidates {
        let events = parse_events(
            controls_obj.get(name),
            &format!("cognitive.controls.{name}"),
        )?;
        controls.push((name.clone(), events));
    }
    Ok(DecisionRequest {
        base,
        goal,
        controls,
    })
}

/// Parsed `entailment` block of an `entail` request: two stored points
/// `[H | R | S]` of the sealed preset geometry, `dim()` numbers each (64, 128
/// or 256). The width is checked against the mount, not here. There is no
/// text encoder: the caller supplies the coordinates.
#[derive(Clone, Debug)]
pub struct EntailmentRequest {
    passage: Vec<f64>,
    question: Vec<f64>,
    fiber: Option<FiberStreams>,
    digest: Digest,
}

/// The two tangent event streams of a scheme 2 request: `passage_events[t]`
/// is a tangent vector at the passage point, `question_events[t]` one at the
/// question point, both at the same `time_ns`.
#[derive(Clone, Debug)]
pub struct FiberStreams {
    window_start_ns: u64,
    passage_events: Vec<Event>,
    question_events: Vec<Event>,
    backend: Option<Backend>,
}

/// `entail`: `{"entailment": {"passage": [dim numbers], "question": [dim numbers]}}`,
/// optionally with the scheme 2 streams `"passage_events"`, `"question_events"`
/// (`[{time_ns, input: [dim numbers]}]`), `"window_start_ns"` and `"backend"`.
/// `dim` is the width of the preset the mount seals.
/// The two streams come together or not at all; stream options without
/// streams are refused. Unknown keys are refused, not ignored.
pub fn parse_entailment(block: &Value) -> Outcome<EntailmentRequest> {
    const FIELDS: &str =
        "passage, question, passage_events, question_events, window_start_ns, backend";
    let obj = block
        .as_object()
        .ok_or_else(|| Rejection::invalid("request", "entailment must be an object"))?;
    if let Some(extra) = obj.keys().find(|k| {
        !matches!(
            k.as_str(),
            "passage"
                | "question"
                | "passage_events"
                | "question_events"
                | "window_start_ns"
                | "backend"
        )
    }) {
        return Err(Rejection::invalid(
            "request",
            format!("entailment.{extra} is not a known field ({FIELDS})"),
        ));
    }
    let passage = f64_vec(obj.get("passage"), "entailment.passage")?;
    let question = f64_vec(obj.get("question"), "entailment.question")?;
    let fiber = match (obj.get("passage_events"), obj.get("question_events")) {
        (None, None) => {
            if let Some(k) = ["window_start_ns", "backend"]
                .into_iter()
                .find(|k| obj.contains_key(*k))
            {
                return Err(Rejection::invalid(
                    "request",
                    format!("entailment.{k} only applies with passage_events and question_events"),
                ));
            }
            None
        }
        (Some(p), Some(q)) => Some(FiberStreams {
            window_start_ns: obj
                .get("window_start_ns")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    Rejection::invalid(
                        "request",
                        "entailment.window_start_ns must be a u64 when event streams are given",
                    )
                })?,
            passage_events: parse_events(Some(p), "entailment.passage_events")?,
            question_events: parse_events(Some(q), "entailment.question_events")?,
            backend: parse_backend(obj.get("backend"))?,
        }),
        _ => {
            return Err(Rejection::invalid(
                "request",
                "entailment.passage_events and entailment.question_events come together: \
                 a cross-difference needs both channels",
            ))
        }
    };
    let canonical = serde_json::to_vec(block).expect("json values always serialize");
    Ok(EntailmentRequest {
        passage,
        question,
        fiber,
        digest: sha256(&[b"gen-zero/entailment-request/v1\0", &canonical]),
    })
}

/// One Busemann entailment verdict, bound to the mount it was computed on.
#[derive(Clone, Debug)]
pub struct EntailmentVerdict {
    pub score: ContainmentScore,
    /// Scheme 2 trace; `None` for a direct (scheme 1) verdict.
    fiber: Option<Value>,
    cone_half_angle: f64,
    preset: TopologyPreset,
    layout: Layout,
    geometry_frame: Digest,
    mount_version: u64,
    request_digest: Digest,
}

impl EntailmentVerdict {
    pub fn is_entailed(&self) -> bool {
        self.score.is_entailed
    }

    pub fn mode(&self) -> &'static str {
        if self.fiber.is_some() {
            "fiber_cross_diff_ssm"
        } else {
            "direct"
        }
    }

    pub fn trace(&self) -> Value {
        let s = &self.score;
        let l = &self.layout;
        let mut out = json!({
            "topology": self.preset.as_str(),
            "dim": l.store_dim(),
            "hyperbolic_dim": l.h(),
            "euclidean_dim": l.e(),
            "spherical_dim": l.s_ambient(),
            "mode": self.mode(),
            "is_entailed": s.is_entailed,
            "confidence": s.confidence,
            "confidence_kind": "alpha_weighted_margin_not_a_probability",
            "calibrated": false,
            // Continuous evidence for a refusal; `is_entailed` alone is the gate.
            "violation_energy": s.violation_energy,
            "soft_confidence": s.soft_confidence,
            "soft_confidence_kind": "exp_minus_half_violation_energy_not_a_probability",
            "hyperbolic_distance": s.hyperbolic_distance,
            "cone_angle": s.cone_angle,
            "aperture": s.aperture,
            "busemann_depth_gain": s.busemann_depth_gain,
            "sphere_angle": s.sphere_angle,
            "topic_shift": s.topic_shift,
            "factors": {
                "in_cone": s.in_cone,
                "deeper": s.deeper,
                "sphere_absorbed": s.sphere_absorbed,
                "topic_aligned": s.topic_aligned,
            },
            "geometry": {
                "topology": self.preset.as_str(),
                "cone_half_angle": self.cone_half_angle,
                "frame": digest_hex(&self.geometry_frame),
            },
            "mount_version": self.mount_version,
            "request_digest": digest_hex(&self.request_digest),
        });
        if let Some(f) = &self.fiber {
            out["fiber_ssm"] = f.clone();
        }
        out
    }
}

fn lod_reject(r: LodReject) -> Reject {
    match r {
        LodReject::EnergyRose => Reject::EnergyRose,
        LodReject::UncertainEnergy => Reject::UncertainEnergy,
        LodReject::Stalled => Reject::Stalled,
        LodReject::ResidualExceeded => Reject::ResidualExceeded,
        LodReject::PinnedMoved => Reject::PinnedMoved,
        LodReject::NonFiniteState => Reject::NonFiniteState,
        LodReject::DomainViolation => Reject::DomainViolation,
        LodReject::CutLocus => Reject::CutLocus,
        LodReject::FiberMismatch => Reject::FiberMismatch,
        LodReject::EpochMismatch => Reject::EpochMismatch,
        LodReject::CocycleViolation => Reject::CocycleViolation,
        LodReject::Obstruction => Reject::Obstruction,
        LodReject::NotConverged => Reject::NotConverged,
        LodReject::BudgetExceeded => Reject::BudgetExceeded,
        LodReject::NoFeasibleExpert => Reject::NoFeasibleExpert,
        LodReject::AmbiguousAction => Reject::AmbiguousAction,
        LodReject::InvalidCertificate => Reject::InvalidCertificate,
        LodReject::CoverageLost => Reject::CoverageLost,
        LodReject::DepositConflict => Reject::DepositConflict,
        LodReject::CasConflict => Reject::CasConflict,
        LodReject::BackendUnavailable => Reject::BackendUnavailable,
        LodReject::UnsupportedOperatorFamily => Reject::UnsupportedOperatorFamily,
    }
}

fn lod_rejection(stage: &'static str) -> impl Fn(LodError) -> Rejection {
    move |e| match e {
        LodError::Geometry(r) => Rejection::reject(lod_reject(r), stage, r.to_string()),
        LodError::DegenerateRadialDirection(_) => {
            Rejection::reject(Reject::DomainViolation, stage, e.to_string())
        }
        // The entailment call path cannot produce graph or core errors. If it
        // ever does, refuse loudly instead of guessing a verdict.
        other => Rejection::reject(
            Reject::BackendUnavailable,
            stage,
            format!("unexpected geometry error: {other}"),
        ),
    }
}

/// One simulated and gate-certified trajectory.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub final_point: Vec<f64>,
    pub final_tangent: Vec<f64>,
    pub prefix_digest: Digest,
    pub scan: Value,
    pub gate: GateReport,
}

impl Trajectory {
    pub fn trace(&self) -> Value {
        json!({
            "final_point": self.final_point,
            "final_tangent": self.final_tangent,
            "scan": self.scan,
            "gate": self.gate,
        })
    }
}

/// The geometric certificate of one chosen action.
#[derive(Clone, Debug, Serialize)]
pub struct CertifiedAction {
    pub action: String,
    pub certificate: String,
    pub mount_version: u64,
    pub mount_digest: String,
    pub energy_before: [f64; 2],
    pub energy_after: [f64; 2],
}

/// Per-candidate result of a decision.
#[derive(Clone, Debug)]
pub struct CandidateResult {
    pub name: String,
    pub outcome: std::result::Result<(Trajectory, [f64; 2]), Rejection>,
}

pub struct Decision {
    pub energy_before: [f64; 2],
    pub candidates: Vec<CandidateResult>,
    pub chosen: Outcome<(CertifiedAction, Vec<f64>)>,
}

impl Decision {
    pub fn trace(&self) -> Value {
        let cands: Vec<Value> = self
            .candidates
            .iter()
            .map(|c| match &c.outcome {
                Ok((traj, v)) => json!({
                    "action": c.name, "certified": true,
                    "energy_after": v, "trajectory": traj.trace(),
                }),
                Err(r) => json!({"action": c.name, "certified": false, "reject": r}),
            })
            .collect();
        json!({"energy_before": self.energy_before, "candidates": cands})
    }
}

/// `scan.threads` of the policy, `0` meaning all cores.
fn default_backend(scan: &ScanPolicy) -> Backend {
    let threads = match scan.threads {
        0 => std::thread::available_parallelism().map_or(1, usize::from),
        n => n,
    };
    Backend::CpuParallel { threads }
}

fn scan_trace(ev: &ScanEvidence) -> Value {
    json!({
        "backend": ev.backend.label(),
        "steps": ev.steps,
        "scan_depth": ev.depth,
        "compositions": ev.compositions,
        "input_digest": blake3::Hash::from(ev.input_digest).to_hex().to_string(),
        "prefix_digest": blake3::Hash::from(ev.prefix_digest).to_hex().to_string(),
        "growth_bound_inf": ev.growth_bound,
        "elapsed_ns": ev.elapsed_ns,
    })
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// §5.5 `CognitiveRuntime`: the single execution path of geometric requests.
#[derive(Default)]
pub struct CognitiveRuntime {
    ssm: ZohTangentSsm,
}

impl CognitiveRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    fn fiber(snapshot: &MountSnapshot, assets: &CognitiveAssets) -> FiberId {
        let bytes = assets.canonical_bytes();
        FiberId {
            patch: 0,
            base: sha256(&[
                DOMAIN_FIBER,
                b"origin",
                &(assets.dim() as u64).to_le_bytes(),
            ]),
            frame: sha256(&[DOMAIN_FIBER, b"frame", &bytes]),
            path: sha256(&[DOMAIN_FIBER, b"identity-transport"]),
            epochs: snapshot.epochs().clone(),
        }
    }

    /// Request -> tangent map -> parallel SSM scan -> geometry gate.
    pub fn simulate(
        &self,
        snapshot: &MountSnapshot,
        assets: &CognitiveAssets,
        req: &TrajectoryRequest,
        events: &[Event],
    ) -> Outcome<Trajectory> {
        let ball = assets.ball;
        let v0 = ball.log0(&req.state, "cognitive.state")?;
        ball.check_safety_radius(&req.state, "cognitive.state")?;
        let t_max = events.len();
        let mut pins = Vec::with_capacity(req.pins.len());
        for (i, (step, point)) in req.pins.iter().enumerate() {
            if *step == 0 || *step > t_max {
                return Err(Rejection::invalid(
                    "request",
                    format!("cognitive.pins[{i}].step must be in 1..={t_max}"),
                ));
            }
            pins.push((
                *step,
                ball.log0(point, &format!("cognitive.pins[{i}].point"))?,
            ));
        }

        let fiber = Self::fiber(snapshot, assets);
        let wire = &assets.wire;
        let budget = ScanBudget {
            max_steps: MAX_WINDOW,
            max_growth: wire.scan.max_growth,
            max_state_norm: wire.scan.max_state_norm,
        };
        let ctx = FrozenContext::new(
            fiber.clone(),
            assets.a.clone(),
            assets.b.clone(),
            req.window_start_ns,
            budget,
        )
        .map_err(at("scan"))?;
        let window = self.ssm.prepare(events, &ctx).map_err(at("scan"))?;
        let backend = req.backend.unwrap_or_else(|| default_backend(&wire.scan));
        let h0 = Tangent::new(fiber, v0.clone()).map_err(at("scan"))?;
        let out: ScanOutput = self.ssm.scan(&window, &h0, backend).map_err(at("scan"))?;
        let candidate: Vec<Vec<f64>> = out.states.iter().map(|t| t.coords().to_vec()).collect();

        let problem = WindowGateInput {
            steps: &window,
            s0: v0,
            pins,
            pin_weight: wire.gate.pin_weight,
        };
        let (accepted, gate) =
            certify_window(&problem, candidate, &wire.gate).map_err(|(code, detail, report)| {
                let mut r = Rejection::reject(code, "geometry_gate", detail);
                if let Some(rep) = report {
                    r.detail = format!("{}; gate={}", r.detail, json!(rep));
                }
                r
            })?;
        // No certified state may sit in the critical boundary zone.
        for (i, tangent) in accepted.iter().enumerate() {
            let point = ball.exp0(tangent, "window state")?;
            ball.check_safety_radius(&point, &format!("window state {}", i + 1))?;
        }
        let final_tangent = accepted
            .last()
            .cloned()
            .ok_or_else(|| Rejection::reject(Reject::NonFiniteState, "readout", "empty window"))?;
        let final_point = ball.exp0(&final_tangent, "final state")?;
        let ev = &out.evidence;
        let scan = scan_trace(ev);
        Ok(Trajectory {
            final_point,
            final_tangent,
            prefix_digest: ev.prefix_digest,
            scan,
            gate,
        })
    }

    /// Simulate every candidate's control window and certify the one whose
    /// geodesic energy to the goal falls the most. Ties within the interval
    /// are `AmbiguousAction`: the runtime does not pick by list order.
    pub fn decide(
        &self,
        snapshot: &MountSnapshot,
        assets: &CognitiveAssets,
        req: &DecisionRequest,
    ) -> Outcome<Decision> {
        let ball = assets.ball;
        ball.check_point(&req.base.state, "cognitive.state")?;
        ball.check_point(&req.goal, "cognitive.goal")?;
        let rel = assets.gate().numeric_error;
        let d0 = ball.distance(&req.base.state, &req.goal)?;
        let before = interval(d0 * d0, rel);

        let candidates: Vec<CandidateResult> = req
            .controls
            .iter()
            .map(|(name, events)| {
                let outcome = self
                    .simulate(snapshot, assets, &req.base, events)
                    .and_then(|traj| {
                        let d = ball.distance(&traj.final_point, &req.goal)?;
                        let after = interval(d * d, rel);
                        ActionVerifier.verify(before, after)?;
                        Ok((traj, after))
                    });
                CandidateResult {
                    name: name.clone(),
                    outcome,
                }
            })
            .collect();

        let chosen = self.choose(snapshot, assets, req, before, &candidates);
        Ok(Decision {
            energy_before: before,
            candidates,
            chosen,
        })
    }

    fn choose(
        &self,
        snapshot: &MountSnapshot,
        assets: &CognitiveAssets,
        req: &DecisionRequest,
        before: [f64; 2],
        candidates: &[CandidateResult],
    ) -> Outcome<(CertifiedAction, Vec<f64>)> {
        let mut ok: Vec<(usize, &Trajectory, [f64; 2])> = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.outcome.as_ref().ok().map(|(t, v)| (i, t, *v)))
            .collect();
        if ok.is_empty() {
            let rejects: Vec<&Rejection> = candidates
                .iter()
                .filter_map(|c| c.outcome.as_ref().err())
                .collect();
            let first = rejects[0];
            // One shared cause is reported as itself; mixed causes are
            // "no candidate survived", with every cause in the detail.
            if rejects.iter().all(|r| r.code == first.code) {
                let mut r = first.clone();
                r.detail = format!("every candidate refused: {}", r.detail);
                return Err(r);
            }
            let causes: Vec<String> = candidates
                .iter()
                .filter_map(|c| {
                    c.outcome
                        .as_ref()
                        .err()
                        .map(|r| format!("{}={}", c.name, r.code))
                })
                .collect();
            return Err(Rejection::reject(
                Reject::NoFeasibleExpert,
                "action_verifier",
                format!("no candidate certified: {}", causes.join(", ")),
            ));
        }
        ok.sort_by(|a, b| a.2[0].total_cmp(&b.2[0]));
        if ok.len() > 1 && ok[1].2[0] <= ok[0].2[1] {
            return Err(Rejection::reject(
                Reject::AmbiguousAction,
                "action_verifier",
                format!(
                    "'{}' and '{}' reach the goal within the energy interval",
                    candidates[ok[0].0].name, candidates[ok[1].0].name
                ),
            ));
        }
        // Softmin over certified energies; refused candidates get 0.
        let tau = assets.wire.decision_temperature;
        let e_min = ok[0].2[0];
        let mut probs = vec![0.0; candidates.len()];
        for (i, _, v) in &ok {
            probs[*i] = (-(v[0] - e_min) / tau).exp();
        }
        let z: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= z);

        let (idx, traj, after) = ok[0];
        let name = &candidates[idx].name;
        let cert = sha256(&[
            DOMAIN_CERT,
            snapshot.digest(),
            &req.base.digest,
            &(name.len() as u64).to_le_bytes(),
            name.as_bytes(),
            &traj.prefix_digest,
            &before[0].to_le_bytes(),
            &after[1].to_le_bytes(),
        ]);
        Ok((
            CertifiedAction {
                action: name.clone(),
                certificate: digest_hex(&cert),
                mount_version: snapshot.version().0,
                mount_digest: digest_hex(snapshot.digest()),
                energy_before: before,
                energy_after: after,
            },
            probs,
        ))
    }

    /// The preset product geometry a snapshot seals. The lod epochs copy the
    /// mount's version and digests, except `geometry`, which lod derives from
    /// the layout and parameters (blake3) so a point cannot cross geometries.
    /// No `entailment` block means nothing was mounted: refused.
    pub fn sealed_geometry(
        &self,
        snapshot: &MountSnapshot,
        assets: &CognitiveAssets,
    ) -> Outcome<ProductManifold> {
        let cfg = assets.wire.entailment.as_ref().ok_or_else(|| {
            Rejection::reject(
                Reject::BackendUnavailable,
                "entailment",
                format!(
                    "mount version {} seals no entailment geometry; publish assets with an \
                     `entailment` block",
                    snapshot.version().0
                ),
            )
        })?;
        let e = snapshot.epochs();
        let base = LodEpochs {
            version: LodVersion(e.version.0),
            model: e.model,
            geometry: e.geometry,
            atlas: e.atlas,
            graph: e.graph,
            policy: e.policy,
        };
        // `preset()` already passed at mount time; a sealed name that no
        // longer parses (a removed preset) is refused here, not remapped.
        ProductManifold::from_preset_with(cfg.preset()?, cfg.params(), base)
            .map_err(|r| Rejection::reject(lod_reject(r), "entailment", r.to_string()))
    }

    /// Asymmetric Busemann test `passage ⊃ question` on the sealed preset
    /// geometry. A point or event whose width is not the sealed `dim()` is
    /// `FiberMismatch`; off-domain, non-finite or degenerate input is a typed
    /// refusal. Never a `false` verdict in place of a refusal.
    pub fn evaluate_entailment(
        &self,
        snapshot: &MountSnapshot,
        assets: &CognitiveAssets,
        req: &EntailmentRequest,
    ) -> Outcome<EntailmentVerdict> {
        let geometry = self.sealed_geometry(snapshot, assets)?;
        let cfg = assets.wire.entailment.as_ref().ok_or_else(|| {
            Rejection::reject(
                Reject::BackendUnavailable,
                "entailment",
                "no entailment block",
            )
        })?;
        let (cone_half_angle, preset) = (cfg.cone_half_angle, cfg.preset()?);
        check_widths(&geometry, req)?;
        let (question, fiber) = match &req.fiber {
            None => (req.question.clone(), None),
            Some(streams) => {
                let (q_eff, mut trace) =
                    self.fiber_cross_diff(snapshot, assets, &geometry, req, streams)?;
                // The unfiltered verdict, for comparison only. Its refusal
                // does not block the fiber verdict; it is reported as such.
                trace["direct"] = match geometry.busemann_containment(
                    &req.passage,
                    &req.question,
                    cone_half_angle,
                ) {
                    Ok(s) => json!({"is_entailed": s.is_entailed, "confidence": s.confidence}),
                    Err(e) => json!({"refused": lod_rejection("entailment")(e).code}),
                };
                (q_eff, Some(trace))
            }
        };
        let score = geometry
            .busemann_containment(&req.passage, &question, cone_half_angle)
            .map_err(lod_rejection("entailment"))?;
        // A NaN confidence serializes as JSON `null` under HTTP 200: refuse it.
        if !score.confidence.is_finite() {
            return Err(Rejection::reject(
                Reject::NonFiniteState,
                "entailment",
                "non-finite confidence computed from geometry",
            ));
        }
        Ok(EntailmentVerdict {
            score,
            fiber,
            cone_half_angle,
            preset,
            layout: *geometry.layout(),
            geometry_frame: geometry.frame(),
            mount_version: snapshot.version().0,
            request_digest: req.digest,
        })
    }
}

impl CognitiveRuntime {
    /// Scheme 2: gauge `Gamma_{p->q}` from the sealed geometry, cross-difference
    /// of the two streams, parallel scan on the question fiber, and the
    /// evidence-moved question point `q' = exp_q(h_T)`. Returns `q'` and the
    /// trace. Every failure is a typed refusal; nothing falls back to `q`.
    fn fiber_cross_diff(
        &self,
        snapshot: &MountSnapshot,
        assets: &CognitiveAssets,
        geometry: &ProductManifold,
        req: &EntailmentRequest,
        streams: &FiberStreams,
    ) -> Outcome<(Vec<f64>, Value)> {
        const STAGE: &str = "fiber_cross_diff";
        let (a, b) = assets.fiber.as_ref().ok_or_else(|| {
            Rejection::reject(
                Reject::BackendUnavailable,
                STAGE,
                format!(
                    "mount version {} seals no entailment.dynamics; a request with \
                     passage_events/question_events cannot run on it",
                    snapshot.version().0
                ),
            )
        })?;
        let g = |r: LodReject| Rejection::reject(lod_reject(r), STAGE, r.to_string());
        let p = geometry.point(&req.passage).map_err(g)?;
        let q = geometry.point(&req.question).map_err(g)?;
        check_tangent_stream(
            geometry,
            &p,
            &streams.passage_events,
            "entailment.passage_events",
        )?;
        check_tangent_stream(
            geometry,
            &q,
            &streams.question_events,
            "entailment.question_events",
        )?;

        // Levi-Civita transport along the geodesic p -> q. For p == q the path
        // is the single point: transport along it is the identity.
        let (pd, qd) = (p.digest(), q.digest());
        let trivial = pd == qd;
        let path: Vec<LodPoint> = if trivial {
            vec![p.clone()]
        } else {
            vec![p.clone(), q.clone()]
        };
        let arc_length = geometry.distance(&p, &q).map_err(g)?;
        let n = geometry.dim();
        let mut gauge = vec![0.0; n * n];
        let mut unit = vec![0.0; n];
        for i in 0..n {
            unit[i] = 1.0;
            let t = geometry.project_tangent(&p, &unit).map_err(g)?;
            let moved = geometry.transport(&path, &t).map_err(g)?;
            for (r, x) in moved.coords().iter().enumerate() {
                gauge[r * n + i] = *x;
            }
            unit[i] = 0.0;
        }
        let gauge = Matrix::new(n, n, gauge).map_err(at(STAGE))?;

        let fiber_at = |base: Digest, path: Digest| FiberId {
            patch: 0,
            base,
            frame: geometry.frame(),
            path,
            epochs: snapshot.epochs().clone(),
        };
        let source = fiber_at(pd, sha256(&[DOMAIN_CROSS, b"rest", &pd]));
        let target = fiber_at(qd, sha256(&[DOMAIN_CROSS, b"levi-civita", &pd, &qd]));
        let op = FiberCrossDiff::new(source, target.clone(), gauge).map_err(at(STAGE))?;

        let scan = &assets.wire.scan;
        let budget = ScanBudget {
            max_steps: MAX_FIBER_WINDOW,
            max_growth: scan.max_growth,
            max_state_norm: scan.max_state_norm,
        };
        let ctx = FrozenContext::new(
            target,
            a.clone(),
            b.clone(),
            streams.window_start_ns,
            budget,
        )
        .map_err(at("fiber_scan"))?;
        let backend = streams.backend.unwrap_or_else(|| default_backend(scan));
        let run = op
            .scan(
                &self.ssm,
                &ctx,
                &streams.passage_events,
                &streams.question_events,
                backend,
            )
            .map_err(at("fiber_scan"))?;
        // Every prefix state h_t must already be tangent at q, not only h_T:
        // A or B may push a state out of the sphere's tangent plane and a
        // later step may cancel it, and projecting would hide it either way.
        let mut lifted = None;
        for (t, state) in run.output.states.iter().enumerate() {
            let h = state.coords();
            let proj = geometry.project_tangent(&q, h).map_err(g)?;
            let defect = proj
                .coords()
                .iter()
                .zip(h)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            if defect > TANGENT_TOL * (1.0 + inf_norm(h)) {
                return Err(Rejection::reject(
                    Reject::DomainViolation,
                    "fiber_readout",
                    format!(
                        "SSM state h_{} is not tangent at the question point (defect {defect:e})",
                        t + 1
                    ),
                ));
            }
            lifted = Some(proj);
        }
        let lifted = lifted
            .ok_or_else(|| Rejection::reject(Reject::NonFiniteState, "fiber_scan", "empty"))?;
        let h = run.output.states.last().map_or(&[][..], |t| t.coords());
        let q_eff = geometry.exp(&q, &lifted).map_err(g)?;
        let shift = geometry.distance(&q, &q_eff).map_err(g)?;
        let diff_norms: Vec<f64> = run.diffs.iter().map(|d| inf_norm(d.coords())).collect();
        let trace = json!({
            "connection": "levi_civita_product_transport",
            "trivial_path": trivial,
            "arc_length": arc_length,
            "diff_norms_inf": diff_norms,
            "scan": scan_trace(&run.output.evidence),
            "final_state_norm_inf": inf_norm(h),
            "question_shift": shift,
            "evidence_question": q_eff.coords(),
        });
        Ok((q_eff.coords().to_vec(), trace))
    }
}

/// Width contract of the sealed preset: `passage`, `question` and every event
/// input must have exactly `geometry.dim()` coordinates. Runs before any
/// geometry call, so a width error is `FiberMismatch`, not the lod
/// `DomainViolation` a wrong-length point would otherwise produce.
fn check_widths(geometry: &ProductManifold, req: &EntailmentRequest) -> Outcome<()> {
    let dim = geometry.dim();
    for got in [req.passage.len(), req.question.len()] {
        if got != dim {
            return Err(Rejection::reject(
                Reject::FiberMismatch,
                "entailment",
                format!("passage/question dimension mismatch: expected {dim}, got {got}"),
            ));
        }
    }
    if let Some(streams) = &req.fiber {
        for (field, events) in [
            ("entailment.passage_events", &streams.passage_events),
            ("entailment.question_events", &streams.question_events),
        ] {
            for (i, ev) in events.iter().enumerate() {
                if ev.input.len() != dim {
                    return Err(Rejection::reject(
                        Reject::FiberMismatch,
                        "entailment",
                        format!(
                            "{field}[{i}].input dimension mismatch: expected {dim}, got {}",
                            ev.input.len()
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Every input of a stream must be a tangent vector at `base`. Widths were
/// already checked by [`check_widths`].
fn check_tangent_stream(
    geometry: &ProductManifold,
    base: &LodPoint,
    events: &[Event],
    field: &str,
) -> Outcome<()> {
    for (i, ev) in events.iter().enumerate() {
        let t = geometry
            .project_tangent(base, &ev.input)
            .map_err(|r| Rejection::reject(lod_reject(r), "fiber_cross_diff", r.to_string()))?;
        let defect = t
            .coords()
            .iter()
            .zip(ev.input.iter())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        if defect > TANGENT_TOL * (1.0 + inf_norm(&ev.input)) {
            return Err(Rejection::reject(
                Reject::DomainViolation,
                "fiber_cross_diff",
                format!("{field}[{i}].input is not tangent at its base point (defect {defect:e})"),
            ));
        }
    }
    Ok(())
}

/// §3.2 Lyapunov check of a proposed transition: the goal energy
/// `V = d(x, goal)^2` must provably fall.
pub struct ActionVerifier;

impl ActionVerifier {
    pub fn verify(&self, before: [f64; 2], after: [f64; 2]) -> Outcome<()> {
        if after[1] < before[0] {
            return Ok(());
        }
        let (code, what) = if after[0] > before[1] {
            (Reject::EnergyRose, "rose")
        } else {
            (Reject::UncertainEnergy, "did not provably fall")
        };
        Err(Rejection::reject(
            code,
            "action_verifier",
            format!(
                "goal energy {what}: before [{:.6e}, {:.6e}], after [{:.6e}, {:.6e}]",
                before[0], before[1], after[0], after[1]
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tangent_ssm::{serial_prefixes, tree_prefixes, AffineStep};

    fn assets(step_fraction: f64) -> CognitiveAssets {
        CognitiveAssets::validate(AssetsWire {
            schema: ASSETS_SCHEMA.into(),
            provenance: "unit test".into(),
            dim: 2,
            curvature: 1.0,
            generator: Generator {
                a: vec![vec![-1.0, 0.0], vec![0.0, -1.0]],
                b: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            },
            gate: GatePolicy {
                max_steps: 500,
                residual_limit: 1e-10,
                numeric_error: 1e-12,
                step_fraction,
                pin_weight: 1.0,
            },
            scan: ScanPolicy {
                max_growth: 1e6,
                max_state_norm: 1e6,
                threads: 4,
            },
            decision_temperature: 0.1,
            entailment: None,
        })
        .unwrap()
    }

    #[test]
    fn log_exp_round_trip_and_boundary_is_rejected() {
        let ball = PoincareBall::new(1.0, 2);
        let x = [0.3, -0.4];
        let v = ball.log0(&x, "x").unwrap();
        let y = ball.exp0(&v, "y").unwrap();
        assert!((x[0] - y[0]).abs() < 1e-14 && (x[1] - y[1]).abs() < 1e-14);
        assert_eq!(
            ball.log0(&[0.6, 0.8], "x").unwrap_err().code,
            "DomainViolation"
        );
        assert_eq!(
            ball.log0(&[1.2, 0.0], "x").unwrap_err().code,
            "DomainViolation"
        );
        assert_eq!(
            ball.log0(&[f64::NAN, 0.0], "x").unwrap_err().code,
            "NonFiniteState"
        );
        assert_eq!(ball.log0(&[0.1], "x").unwrap_err().code, "FiberMismatch");
        // tanh saturates to 1 in f64: a boundary point, not a result.
        assert_eq!(
            ball.exp0(&[40.0, 0.0], "y").unwrap_err().code,
            "DomainViolation"
        );
    }

    #[test]
    fn exp0_rejects_an_overflowing_tangent_instead_of_folding_it_to_the_origin() {
        let ball = PoincareBall::new(1.0, 2);
        // sum(x*x) for 1e155 overflows f64 to `inf`; the naive
        // `tanh(sqrt_c*inf)/(sqrt_c*inf) = 1.0/inf = 0.0` used to fold this
        // huge, nonzero tangent vector to the ball's origin, and the origin
        // passes `check_point`. That must never happen: this has to be a
        // domain violation, and the result must never be `[0.0, 0.0]`.
        let err = ball.exp0(&[1e155, 0.0], "y").unwrap_err();
        assert_eq!(err.code, "DomainViolation");

        // Same failure mode on the other overflow edge: norm() itself must
        // not silently become non-finite for finite input.
        assert!(
            (1e155f64 * 1e155).is_infinite(),
            "test assumes x*x overflows"
        );
        assert!(norm(&[1e155, 0.0]).is_finite());
        assert_eq!(norm(&[1e155, 0.0]), 1e155);

        // A genuinely non-finite tangent (e.g. from an upstream overflow)
        // is also rejected, not folded.
        let err = ball.exp0(&[f64::INFINITY, 0.0], "y").unwrap_err();
        assert_eq!(err.code, "NonFiniteState");
    }

    #[test]
    fn distance_matches_the_origin_closed_form() {
        let ball = PoincareBall::new(1.0, 2);
        // d(0, x) = 2 artanh(|x|)
        let d = ball.distance(&[0.0, 0.0], &[0.5, 0.0]).unwrap();
        assert!((d - 2.0 * 0.5f64.atanh()).abs() < 1e-14);
    }

    #[test]
    fn gate_relaxation_decreases_and_unstable_step_is_energy_rose() {
        let a = assets(1.0);
        let fiber = CognitiveRuntime::fiber(
            &crate::mount::MountSnapshot::genesis(
                crate::mount::MountKey::new("t", "w"),
                crate::mount::AssetDigests {
                    model: [1; 32],
                    geometry: [2; 32],
                    atlas: [3; 32],
                    graph: [4; 32],
                    policy: [5; 32],
                },
                0,
                std::sync::Arc::from(a.canonical_bytes()),
            )
            .unwrap(),
            &a,
        );
        let m = Matrix::from_rows(&[vec![0.5, 0.0], vec![0.0, 0.5]]).unwrap();
        let steps: Vec<AffineStep> = (0..6)
            .map(|_| {
                AffineStep::new(fiber.clone(), fiber.clone(), m.clone(), vec![0.1, 0.0]).unwrap()
            })
            .collect();
        let window = PreparedWindow::from_steps(
            fiber,
            steps,
            ScanBudget {
                max_steps: 16,
                max_growth: 10.0,
                max_state_norm: 10.0,
            },
        )
        .unwrap();
        // Pins that the dynamics can only partly explain.
        let problem = WindowGateInput {
            steps: &window,
            s0: vec![0.0, 0.0],
            pins: vec![(3, vec![1.0, 1.0]), (6, vec![-1.0, 0.5])],
            pin_weight: 1.0,
        };
        let mut s = vec![0.0f64; 2];
        let cand: Vec<Vec<f64>> = (0..6)
            .map(|_| {
                s = vec![0.5 * s[0] + 0.1, 0.5 * s[1]];
                s.clone()
            })
            .collect();
        let mut policy = a.gate().clone();
        policy.residual_limit = 1e-10; // unreachable: least-squares floor > 0
        policy.max_steps = 50_000;
        let (code, _, rep) = certify_window(&problem, cand.clone(), &policy).unwrap_err();
        assert_eq!(code, Reject::ResidualExceeded);
        let rep = rep.unwrap();
        assert_eq!(rep.status, "stalled");
        let (l0, floor) = (rep.energy_before[0], rep.residual);
        assert!(floor > 0.1 && floor < l0, "floor {floor}, start {l0}");

        // A budget between the floor and the start: reached by strict decrease.
        policy.residual_limit = 0.5 * (floor + l0);
        let (_, rep) = certify_window(&problem, cand.clone(), &policy).unwrap();
        assert_eq!(rep.status, "strictly_decreased");
        assert!(rep.energy_after[1] < rep.energy_before[0]);
        assert!(rep.residual <= policy.residual_limit);

        // Step beyond 2/lambda_max: the gate observes the rise, never clamps.
        policy.step_fraction = 6.0;
        policy.max_steps = 20;
        let (code, _, _) = certify_window(&problem, cand, &policy).unwrap_err();
        assert_eq!(code, Reject::EnergyRose);
    }

    #[test]
    fn parallel_prefix_scan_matches_serial_on_every_prefix() {
        let a = assets(1.0);
        let snap = crate::mount::MountSnapshot::genesis(
            crate::mount::MountKey::new("t", "w"),
            crate::mount::AssetDigests {
                model: [1; 32],
                geometry: [2; 32],
                atlas: [3; 32],
                graph: [4; 32],
                policy: [5; 32],
            },
            0,
            std::sync::Arc::from(a.canonical_bytes()),
        )
        .unwrap();
        let fiber = CognitiveRuntime::fiber(&snap, &a);
        let steps: Vec<AffineStep> = (0..257)
            .map(|i| {
                let t = i as f64;
                let m = Matrix::from_rows(&[vec![0.9 * (t * 0.1).cos(), 0.05], vec![-0.05, 0.8]])
                    .unwrap();
                AffineStep::new(
                    fiber.clone(),
                    fiber.clone(),
                    m,
                    vec![(t * 0.3).sin(), 0.01 * t],
                )
                .unwrap()
            })
            .collect();
        let (serial, _, _) = serial_prefixes(&steps).unwrap();
        let (tree, _, _) = tree_prefixes(&steps, 8).unwrap();
        let mut worst = 0.0f64;
        for (s, t) in serial.iter().zip(&tree) {
            for (x, y) in s.bias().iter().zip(t.bias()) {
                worst = worst.max((x - y).abs() / (1.0 + x.abs()));
            }
        }
        assert!(worst <= 1e-10, "worst relative prefix error {worst}");
    }
}
