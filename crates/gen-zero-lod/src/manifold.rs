//! gen-zero-lod mixed-curvature product manifolds (Spec 25, chapter 2 and section 5.4).
//!
//! `M = H_{-c}^{d_h} x R^{d_e} x S_R^{d_s}` with product metric
//! `g = alpha_h g_H + alpha_e g_E + alpha_s g_S`.
//!
//! - `H` uses `d_h` Poincare ball coordinates (`c ||x||^2 < 1`).
//! - `R` uses `d_e` flat coordinates.
//! - `S_R^{d_s}` uses `d_s + 1` embedding coordinates (`||s|| = R`).
//!
//! A point is stored as `[H | E | S]`, so `H^80 x R^24 x S^23` has intrinsic
//! dimension 127 and 128 stored coordinates, and `H^160 x R^48 x S^47` has
//! intrinsic dimension 255 and 256 stored coordinates.
//!
//! `TopologyPreset` is the closed whitelist of layouts a mount may seal:
//! `compact_64d`, `balanced_128d`, `boolq_128d` (alias `deep_128d`) and
//! `extended_256d`, all at most `MAX_PRESET_DIM = 256` stored coordinates.
//!
//! Every operator fails closed: a point outside the ball, off the sphere, or
//! with a non-finite value returns `Reject::DomainViolation` or
//! `Reject::NonFiniteState`; a sphere pair at (or numerically at) the antipode
//! returns `Reject::CutLocus`. No operator substitutes a default value.
//!
//! The 16-coordinate `MixedCurvatureCoord` (`H^4 x R^8 x S^3`, f32) is the Lod
//! graph's node chart. It has no parameters of its own: its distance takes the
//! graph's `GeometryParams` and equals `ProductGeometry::distance` on the
//! points `MixedCurvatureCoord::to_point` gives.

/// Smallest fatigue fraction accepted, and the step of the decimal grid
/// [`ClockPhase::onset_tick`] computes in exact integers. With budget at least
/// 1 it keeps `fraction * budget` far above that function's machine-precision
/// band, so the onset is never tick 0.
pub const FATIGUE_FRAC_QUANTUM: f64 = 1e-9;

/// Parts per unit of the decimal grid; `1 / FATIGUE_FRAC_QUANTUM`.
const FATIGUE_FRAC_PARTS: u64 = 1_000_000_000;
const _: () = assert!(FATIGUE_FRAC_QUANTUM * FATIGUE_FRAC_PARTS as f64 == 1.0);

/// The one validity rule for a fatigue fraction: finite and in
/// `[FATIGUE_FRAC_QUANTUM, 1]`. Shared by [`ClockPhase::new`] and the planner's
/// disturbance model so the two never disagree.
pub fn valid_fatigue_frac(fraction: f64) -> bool {
    fraction.is_finite() && (FATIGUE_FRAC_QUANTUM..=1.0).contains(&fraction)
}

/// Execution clock drawn on a great circle of S^3, for the robust report only.
///
/// No planner decision reads `theta_clock`, `spherical` or `theta_fatigue`:
/// the robust scorer and the `fatigued` flag both use the integer
/// [`ClockPhase::onset_tick`], never an angle. The angles are telemetry the
/// service echoes as `clock_phase`. `theta_clock` is kept unwrapped (it grows
/// past `2 pi` once `time_used > budget`) because the embedding alone maps an
/// empty and an exhausted budget to the same point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ClockPhase {
    pub theta_clock: f64,
    pub spherical: [f64; 4],
    pub theta_fatigue: Option<f64>,
    pub fatigued: bool,
}

impl ClockPhase {
    pub fn new(
        time_used: u64,
        budget: u32,
        fatigue_frac: Option<f64>,
    ) -> std::result::Result<Self, LodError> {
        if budget == 0 || fatigue_frac.is_some_and(|f| !valid_fatigue_frac(f)) {
            return Err(LodError::Geometry(Reject::DomainViolation));
        }
        let theta_clock = std::f64::consts::TAU * (time_used as f64 / f64::from(budget));
        let theta_fatigue = fatigue_frac.map(|f| std::f64::consts::TAU * f);
        // Decided on the integer onset tick, not by comparing the two angles.
        let fatigued = match fatigue_frac {
            Some(f) => time_used >= Self::onset_tick(budget, f)?,
            None => false,
        };
        let (sin, cos) = theta_clock.sin_cos();
        Ok(Self {
            theta_clock,
            spherical: [cos, sin, 0.0, 0.0],
            theta_fatigue,
            fatigued,
        })
    }

    /// First clock tick at or after `fraction * budget`, for a fraction that
    /// passes [`valid_fatigue_frac`].
    ///
    /// A raw `f64` ceil is wrong at integers: `0.07 * 100.0` is
    /// `7.000000000000001`, and its `ceil` is 8. Two paths avoid that:
    ///
    /// - Decimal grid. If `fraction` is exactly the `f64` nearest to `q / 1e9`
    ///   for an integer `q` (true of every literal with at most 9 decimals),
    ///   the onset is `ceil(q * budget / 1e9)` in `u64`, with no rounding at
    ///   any budget. `q * budget <= 1e9 * u32::MAX` cannot overflow.
    /// - Anything else (`2/3`, `0.0700000001`). A product within
    ///   `4 * EPSILON * prod` above an integer `k` is taken as `k`, which
    ///   absorbs the rounding of the stored fraction and of the multiply, so
    ///   `2/3 * 3` is 2 and `1/6 * 6` is 1. There is no absolute floor, so
    ///   `0.070000000000005 * 100` is 8. Gotcha: a genuine excess below
    ///   `4 * EPSILON * prod` is still read as `k`; that is under one part in
    ///   `1e15` of the budget.
    ///
    /// An invalid fraction or a zero budget returns a domain violation.
    pub fn onset_tick(budget: u32, fraction: f64) -> std::result::Result<u64, LodError> {
        if budget == 0 || !valid_fatigue_frac(fraction) {
            return Err(LodError::Geometry(Reject::DomainViolation));
        }
        let q = (fraction * FATIGUE_FRAC_PARTS as f64).round() as u64;
        if q as f64 / FATIGUE_FRAC_PARTS as f64 == fraction {
            return Ok((q * u64::from(budget)).div_ceil(FATIGUE_FRAC_PARTS));
        }
        let prod = fraction * f64::from(budget);
        let floor_k = prod.floor();
        if prod - floor_k <= prod * f64::EPSILON * 4.0 {
            Ok(floor_k as u64)
        } else {
            Ok(prod.ceil() as u64)
        }
    }
}

/// Dedicated statistical chart; does not overwrite a graph node's existing R^8
/// semantic coordinates. None is an explicitly infinite Dirichlet strength.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DisturbanceMoments {
    pub mean: f64,
    pub variance: f64,
    pub kappa: Option<f64>,
}

use crate::error::LodError;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt;
use std::ops::Range;

// ---------------------------------------------------------------------------
// Spec 25 section 5.4 core types
// ---------------------------------------------------------------------------

pub type Digest = [u8; 32];
pub type Result<T> = std::result::Result<T, Reject>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Version(pub u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Epochs {
    pub version: Version,
    pub model: Digest,
    pub geometry: Digest,
    pub atlas: Digest,
    pub graph: Digest,
    pub policy: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiberId {
    /// Atlas patch. This module implements one global chart, so it is always
    /// `GLOBAL_PATCH`; chart refinement (Spec 25 section 5.2) is not built here.
    pub patch: u64,
    /// Digest of the base point the tangent lives at.
    pub base: Digest,
    /// Digest of the geometry parameters (layout, c, R, alphas).
    pub frame: Digest,
    /// Digest chain of the points the tangent was transported through.
    pub path: Digest,
    pub epochs: Epochs,
}

pub const GLOBAL_PATCH: u64 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reject {
    EnergyRose,
    UncertainEnergy,
    Stalled,
    ResidualExceeded,
    PinnedMoved,
    NonFiniteState,
    DomainViolation,
    CutLocus,
    FiberMismatch,
    EpochMismatch,
    CocycleViolation,
    Obstruction,
    NotConverged,
    BudgetExceeded,
    NoFeasibleExpert,
    AmbiguousAction,
    InvalidCertificate,
    CoverageLost,
    DepositConflict,
    CasConflict,
    BackendUnavailable,
    UnsupportedOperatorFamily,
}

impl fmt::Display for Reject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Reject {}

/// Factor dimensions. `s_intrinsic` is the sphere dimension; the sphere
/// stores `s_intrinsic + 1` embedding coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Layout {
    h: usize,
    e: usize,
    s_intrinsic: usize,
}

impl Layout {
    /// `H^32 x R^16 x S^15`: 63 intrinsic dimensions, 64 stored coordinates.
    pub const STORE_64: Layout = Layout {
        h: 32,
        e: 16,
        s_intrinsic: 15,
    };
    /// `H^64 x R^32 x S^31`: 127 intrinsic dimensions, 128 stored coordinates.
    pub const STORE_128_BALANCED: Layout = Layout {
        h: 64,
        e: 32,
        s_intrinsic: 31,
    };
    /// `H^80 x R^24 x S^23`: 127 intrinsic dimensions, 128 stored coordinates.
    /// The BoolQ entailment layout: the hyperbolic factor gets the most room.
    pub const STORE_128_DEEP: Layout = Layout {
        h: 80,
        e: 24,
        s_intrinsic: 23,
    };
    /// `H^160 x R^48 x S^47`: 255 intrinsic dimensions, 256 stored coordinates.
    pub const STORE_256: Layout = Layout {
        h: 160,
        e: 48,
        s_intrinsic: 47,
    };

    /// A zero-dimensional hyperbolic or spherical factor is not a manifold
    /// factor this module supports; the Euclidean factor may be empty.
    pub fn new(h: usize, e: usize, s_intrinsic: usize) -> Result<Self> {
        if h == 0 || s_intrinsic == 0 {
            return Err(Reject::DomainViolation);
        }
        if h.checked_add(e)
            .and_then(|dim| dim.checked_add(s_intrinsic))
            .and_then(|dim| dim.checked_add(1))
            .is_none()
        {
            return Err(Reject::DomainViolation);
        }
        Ok(Self { h, e, s_intrinsic })
    }

    pub fn h(&self) -> usize {
        self.h
    }
    pub fn e(&self) -> usize {
        self.e
    }
    pub fn s_intrinsic(&self) -> usize {
        self.s_intrinsic
    }
    pub const fn s_ambient(&self) -> usize {
        self.s_intrinsic + 1
    }
    pub fn intrinsic_dim(&self) -> usize {
        self.h + self.e + self.s_intrinsic
    }
    pub const fn store_dim(&self) -> usize {
        self.h + self.e + self.s_ambient()
    }
    pub fn h_range(&self) -> Range<usize> {
        0..self.h
    }
    pub fn e_range(&self) -> Range<usize> {
        self.h..self.h + self.e
    }
    pub fn s_range(&self) -> Range<usize> {
        self.h + self.e..self.store_dim()
    }

    fn hash_into(&self, hasher: &mut blake3::Hasher) {
        for dim in [self.h, self.e, self.s_intrinsic] {
            hasher.update(&(dim as u64).to_le_bytes());
        }
    }
}

/// Largest stored width any topology preset may have.
pub const MAX_PRESET_DIM: usize = 256;

/// The closed whitelist of product topologies a mount may seal. The width is
/// chosen once, at publish time, and frozen into the mount digest; there is no
/// way to ask for an arbitrary layout at run time.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TopologyPreset {
    /// `H^32 x R^16 x S^15`, 64 stored coordinates.
    Compact64d,
    /// `H^64 x R^32 x S^31`, 128 stored coordinates.
    Balanced128d,
    /// `H^80 x R^24 x S^23`, 128 stored coordinates (alias `deep_128d`).
    Boolq128d,
    /// `H^160 x R^48 x S^47`, 256 stored coordinates.
    Extended256d,
}

impl TopologyPreset {
    pub const ALL: [TopologyPreset; 4] = [
        TopologyPreset::Compact64d,
        TopologyPreset::Balanced128d,
        TopologyPreset::Boolq128d,
        TopologyPreset::Extended256d,
    ];

    /// Canonical name. `deep_128d` parses but is never produced.
    pub const fn as_str(&self) -> &'static str {
        match self {
            TopologyPreset::Compact64d => "compact_64d",
            TopologyPreset::Balanced128d => "balanced_128d",
            TopologyPreset::Boolq128d => "boolq_128d",
            TopologyPreset::Extended256d => "extended_256d",
        }
    }

    pub const fn layout(&self) -> Layout {
        match self {
            TopologyPreset::Compact64d => Layout::STORE_64,
            TopologyPreset::Balanced128d => Layout::STORE_128_BALANCED,
            TopologyPreset::Boolq128d => Layout::STORE_128_DEEP,
            TopologyPreset::Extended256d => Layout::STORE_256,
        }
    }

    /// Stored width (`h + e + s_intrinsic + 1`).
    pub const fn dim(&self) -> usize {
        self.layout().store_dim()
    }
}

impl std::str::FromStr for TopologyPreset {
    type Err = Reject;

    /// Exact, case-sensitive match against the whitelist. Anything else,
    /// including a well-formed name of an unlisted width, is `DomainViolation`.
    fn from_str(name: &str) -> Result<Self> {
        match name {
            "compact_64d" => Ok(TopologyPreset::Compact64d),
            "balanced_128d" => Ok(TopologyPreset::Balanced128d),
            "boolq_128d" | "deep_128d" => Ok(TopologyPreset::Boolq128d),
            "extended_256d" => Ok(TopologyPreset::Extended256d),
            _ => Err(Reject::DomainViolation),
        }
    }
}

impl fmt::Display for TopologyPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// Every preset stays under the hard cap and fills whole 64-byte cache lines.
const _: () = {
    let mut i = 0;
    while i < TopologyPreset::ALL.len() {
        let d = TopologyPreset::ALL[i].dim();
        assert!(d <= MAX_PRESET_DIM);
        assert!(d * std::mem::size_of::<f64>() % 64 == 0);
        i += 1;
    }
};

/// A validated point. Built only by `ProductManifold::point` or by the
/// geometry operators, so fields stay private.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    layout: Layout,
    epochs: Epochs,
    coords: Box<[f64]>,
}

impl Point {
    pub fn layout(&self) -> Layout {
        self.layout
    }
    pub fn epochs(&self) -> &Epochs {
        &self.epochs
    }
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }
    /// Content digest over layout and exact coordinate bits.
    pub fn digest(&self) -> Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"gen-zero/manifold/point/v1");
        self.layout.hash_into(&mut hasher);
        for x in self.coords.iter() {
            hasher.update(&x.to_bits().to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

/// A tangent vector bound to its fiber (base point, frame, path, epochs).
#[derive(Clone, Debug, PartialEq)]
pub struct Tangent {
    fiber: FiberId,
    coords: Box<[f64]>,
}

impl Tangent {
    pub fn fiber(&self) -> &FiberId {
        &self.fiber
    }
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }
}

pub trait ProductGeometry: Send + Sync {
    fn log(&self, base: &Point, point: &Point) -> Result<Tangent>;
    fn exp(&self, base: &Point, tangent: &Tangent) -> Result<Point>;
    fn distance(&self, lhs: &Point, rhs: &Point) -> Result<f64>;
    fn project_tangent(&self, base: &Point, ambient: &[f64]) -> Result<Tangent>;
    fn transport(&self, path: &[Point], tangent: &Tangent) -> Result<Tangent>;
    /// Asymmetric Busemann entailment test `passage ⊃ question` on raw stored
    /// coordinates (`[H | E | S]`). `cone_half_angle` caps the entailment cone
    /// at the passage; the other thresholds come from
    /// [`ContainmentCriteria::from_cone_half_angle`]. Fails closed: a wrong
    /// length, non-finite value, off-domain point or degenerate radial
    /// direction is an error, never a `false` verdict.
    fn busemann_containment(
        &self,
        passage: &[f64],
        question: &[f64],
        cone_half_angle: f64,
    ) -> std::result::Result<ContainmentScore, LodError>;
}

// ---------------------------------------------------------------------------
// Factor kernels (slice based, f64). Shared by `ProductManifold` and
// `MixedCurvatureCoord`.
// ---------------------------------------------------------------------------

/// Points with `1 - c ||x||^2 <= BALL_MARGIN` count as on the ball boundary.
pub const BALL_MARGIN: f64 = 1e-12;
/// Relative tolerance on `||s||^2 / R^2 - 1` for f64 sphere points.
///
/// A 256-coordinate norm has worst-case summation error on the order of
/// `256 * f64::EPSILON` (about `6e-14`); `1e-9` leaves several orders of
/// magnitude for coordinate and normalization roundoff while remaining much
/// tighter than a meaningful off-sphere input.
pub const SPHERE_NORM_TOL: f64 = 1e-9;
/// Angular distance from the antipode (radians) below which sphere log and
/// transport refuse: the minimizing geodesic is not unique there.
pub const CUT_LOCUS_TOL: f64 = 1e-6;
/// Tolerance on `|<s, v>| / R` relative to `1 + ||v||` for sphere tangents.
const SPHERE_TANGENT_TOL: f64 = 1e-8;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    // `sqrt(sum(x*x))` overflows as soon as an individual coordinate is
    // larger than `sqrt(MAX)`, even when the norm itself is representable.
    // `hypot` scales each accumulation and preserves the largest finite
    // result until the true norm exceeds the f64 range.
    a.iter().fold(0.0, |acc, x| acc.hypot(*x))
}

fn norm_scaled(scale: f64, a: &[f64]) -> f64 {
    a.iter().fold(0.0, |acc, x| acc.hypot(scale * *x))
}

/// Dot a finite vector against a vector whose norm is at most one. Scaling the
/// second vector avoids overflowing an individual product when the first
/// vector is large, while retaining a finite result when the true dot product
/// is finite.
fn dot_unit_scaled(unit: &[f64], values: &[f64]) -> f64 {
    let scale = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let scaled = unit
        .iter()
        .zip(values)
        .map(|(u, value)| *u * (*value / scale))
        .sum::<f64>();
    scaled * scale
}

fn check_same_lengths(out: &[f64], a: &[f64], b: &[f64]) -> Result<()> {
    if out.len() == a.len() && a.len() == b.len() && !a.is_empty() {
        Ok(())
    } else {
        Err(Reject::DomainViolation)
    }
}

fn check_same_lengths4(out: &[f64], a: &[f64], b: &[f64], c: &[f64]) -> Result<()> {
    if out.len() == a.len() && a.len() == b.len() && b.len() == c.len() && !a.is_empty() {
        Ok(())
    } else {
        Err(Reject::DomainViolation)
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn ensure_finite(a: &[f64]) -> Result<()> {
    if a.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Reject::NonFiniteState)
    }
}

pub mod kernel {
    //! Closed-form factor operators. Each function validates its own inputs.

    use super::*;

    /// Returns the conformal denominator `1 - c ||x||^2`, which must be > 0.
    pub fn ball_conformal(c: f64, x: &[f64]) -> Result<f64> {
        if !positive_finite(c) || x.is_empty() {
            return Err(Reject::DomainViolation);
        }
        ensure_finite(x)?;
        let sc = c.sqrt();
        let radius = norm_scaled(sc, x);
        let radius_sq = radius * radius;
        let conf = 1.0 - radius_sq;
        // Strict `c ||x||^2 < 1`, with a margin: rounding in `c ||x||^2` is about
        // `d_h * eps` (~4e-14 for d_h = 160), so below `BALL_MARGIN` the conformal
        // factor has too few correct digits to be meaningful. A NaN conf also fails.
        if conf.is_finite() && conf > BALL_MARGIN {
            Ok(conf)
        } else {
            Err(Reject::DomainViolation)
        }
    }

    /// Mobius addition `x (+)_c y`.
    pub fn mobius_add(c: f64, x: &[f64], y: &[f64], out: &mut [f64]) -> Result<()> {
        check_same_lengths(out, x, y)?;
        let conf_x = ball_conformal(c, x)?;
        let conf_y = ball_conformal(c, y)?;
        let sc = c.sqrt();
        let sx: Vec<f64> = x.iter().map(|value| sc * *value).collect();
        let sy: Vec<f64> = y.iter().map(|value| sc * *value).collect();
        let sum_norm = sx.iter().zip(&sy).fold(0.0_f64, |n, (a, b)| n.hypot(a + b));
        let sum_sq = sum_norm * sum_norm;
        // Equivalent to 1 + 2<x,y> + |x|²|y|², without cancellation
        // when two near-boundary points are almost negatives of one another.
        let den = conf_x * conf_y + sum_sq;
        if !(den > 0.0 && den.is_finite()) {
            return Err(Reject::DomainViolation);
        }
        // Keep coordinates in physical units: dimensionless round trips can
        // erase small orthogonal components at tiny curvature.
        for ((o, xi), yi) in out.iter_mut().zip(x).zip(y) {
            *o = (conf_x * (*xi + *yi) + sum_sq * *xi) / den;
        }
        ensure_finite(out)?;
        ball_conformal(c, out).map(|_| ())
    }

    /// `d_H(x, y) = (2/sqrt(c)) arsinh(sqrt(c) ||x - y|| / (sqrt(1 - c||x||^2) sqrt(1 - c||y||^2)))`.
    /// Equal to the arcosh form of Spec 25 (2.3) since `arcosh(1 + 2u^2) = 2 arsinh(u)`;
    /// arsinh keeps full precision near zero distance.
    pub fn hyperbolic_distance(c: f64, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Err(Reject::DomainViolation);
        }
        let conf_x = ball_conformal(c, x)?;
        let conf_y = ball_conformal(c, y)?;
        let sc = c.sqrt();
        let diff = norm_diff(x, y);
        if !diff.is_finite() {
            return Err(Reject::NonFiniteState);
        }
        let denominator = conf_x.sqrt() * conf_y.sqrt();
        let diff_over_den = diff / denominator;
        let arg = sc * diff_over_den;
        if !arg.is_finite() {
            return Err(Reject::NonFiniteState);
        }
        let numerator = 2.0 * arg.asinh();
        // Divide after evaluating the dimensionless distance. This avoids the
        // `infinity * zero` NaN at coincident points when c is subnormal.
        let d = if diff == 0.0 {
            0.0
        } else if arg == 0.0 {
            // `sc * diff` underflowed, so use the first-order limit
            // `2 ||x-y|| / sqrt(conf_x conf_y)` directly.
            2.0 * diff_over_den
        } else {
            numerator / sc
        };
        if d.is_finite() {
            Ok(d)
        } else {
            Err(Reject::NonFiniteState)
        }
    }

    /// `log_x(y) = (2 / (sqrt(c) lambda_x)) artanh(sqrt(c) ||u||) u / ||u||`, `u = (-x) (+) y`.
    pub fn hyperbolic_log(c: f64, x: &[f64], y: &[f64], out: &mut [f64]) -> Result<()> {
        check_same_lengths(out, x, y)?;
        let conf_x = ball_conformal(c, x)?;
        ball_conformal(c, y)?;
        let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
        let mut u = vec![0.0; x.len()];
        mobius_add(c, &neg_x, y, &mut u)?;
        let sc = c.sqrt();
        let z = norm_scaled(sc, &u);
        if z == 0.0 {
            // `atanh(z) / z -> 1` as z -> 0. This is the Euclidean limit for
            // subnormal curvature/coordinates where the scaled norm is zero.
            let scale = conf_x;
            for (o, ui) in out.iter_mut().zip(&u) {
                *o = scale * *ui;
            }
            ensure_finite(out)?;
            return Ok(());
        }
        if !z.is_finite() || z >= 1.0 {
            return Err(Reject::DomainViolation);
        }
        // `u` is already in physical coordinates, so the scale simplifies to
        // `conf_x * atanh(z) / z` and never forms a large `1/sqrt(c)` factor.
        let scale = conf_x * (z.atanh() / z);
        for (o, ui) in out.iter_mut().zip(&u) {
            *o = scale * ui;
        }
        ensure_finite(out)
    }

    /// `exp_x(v) = x (+) (tanh(sqrt(c) lambda_x ||v|| / 2) v / (sqrt(c) ||v||))`.
    /// A result pushed onto the ball boundary by `tanh` saturation is rejected.
    pub fn hyperbolic_exp(c: f64, x: &[f64], v: &[f64], out: &mut [f64]) -> Result<()> {
        check_same_lengths(out, x, v)?;
        let conf_x = ball_conformal(c, x)?;
        ensure_finite(v)?;
        let vn = norm(v);
        if !vn.is_finite() {
            return Err(Reject::NonFiniteState);
        }
        if vn == 0.0 {
            out.copy_from_slice(x);
            return Ok(());
        }
        let sc = c.sqrt();
        let scaled_speed = sc * vn / conf_x;
        let w: Vec<f64> = if scaled_speed < 1e-8 {
            // tanh(z)/z differs from one by less than one ulp here. Work in
            // physical units to preserve even subnormal scaled displacements.
            v.iter().map(|value| *value / conf_x).collect()
        } else {
            let length = scaled_speed.tanh() / sc;
            v.iter().map(|value| length * (*value / vn)).collect()
        };
        mobius_add(c, x, &w, out)?;
        ball_conformal(c, out)?;
        Ok(())
    }

    /// Gyration `gyr[a, b] w` in closed form (linear in `w`, valid for any `w`).
    pub fn gyration(c: f64, a: &[f64], b: &[f64], w: &[f64], out: &mut [f64]) -> Result<()> {
        check_same_lengths4(out, a, b, w)?;
        let conf_a = ball_conformal(c, a)?;
        let conf_b = ball_conformal(c, b)?;
        ensure_finite(w)?;
        let sc = c.sqrt();
        let sa: Vec<f64> = a.iter().map(|value| sc * *value).collect();
        let sb: Vec<f64> = b.iter().map(|value| sc * *value).collect();
        let ab = dot(&sa, &sb);
        let aw = dot_unit_scaled(&sa, w);
        let bw = dot_unit_scaled(&sb, w);
        let a2 = dot(&sa, &sa);
        let b2 = dot(&sb, &sb);
        // The expressions below are the dimensionless closed form rewritten
        // so `c`, `c²`, and coordinates near `1/sqrt(c)` never overflow.
        let big_a = -aw * b2 + bw + 2.0 * ab * bw;
        let big_b = -bw * a2 - aw;
        let sum_norm = sa.iter().zip(&sb).fold(0.0_f64, |n, (a, b)| n.hypot(a + b));
        let den = conf_a * conf_b + sum_norm * sum_norm;
        if !(den > 0.0 && den.is_finite()) {
            return Err(Reject::DomainViolation);
        }
        for (((o, wi), ai), bi) in out.iter_mut().zip(w).zip(a).zip(b) {
            *o = wi + 2.0 * (big_a * (sc * *ai) + big_b * (sc * *bi)) / den;
        }
        ensure_finite(out)
    }

    /// Parallel transport along the geodesic `x -> y`:
    /// `P(v) = (lambda_x / lambda_y) gyr[y, -x] v`.
    pub fn hyperbolic_transport(
        c: f64,
        x: &[f64],
        y: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<()> {
        check_same_lengths4(out, x, y, v)?;
        let conf_x = ball_conformal(c, x)?;
        let conf_y = ball_conformal(c, y)?;
        ensure_finite(v)?;
        let neg_x: Vec<f64> = x.iter().map(|t| -t).collect();
        gyration(c, y, &neg_x, v, out)?;
        // lambda_x / lambda_y = conf_y / conf_x
        let ratio = conf_y / conf_x;
        for o in out.iter_mut() {
            *o *= ratio;
        }
        ensure_finite(out)
    }

    /// Checks `| ||s||^2 / R^2 - 1 | <= tol`.
    pub fn sphere_check(radius: f64, s: &[f64], tol: f64) -> Result<()> {
        if !positive_finite(radius) || !tol.is_finite() || !(0.0..1.0).contains(&tol) {
            return Err(Reject::DomainViolation);
        }
        if s.is_empty() {
            return Err(Reject::DomainViolation);
        }
        ensure_finite(s)?;
        // Compare the dimensionless norm before squaring. This remains valid
        // for radii near both `f64::MAX` and the smallest subnormal, where
        // `radius * radius` would overflow or underflow.
        let ratio = s.iter().fold(0.0_f64, |n, x| n.hypot(*x / radius));
        if !ratio.is_finite() {
            return Err(Reject::DomainViolation);
        }
        let rel = ratio.mul_add(ratio, -1.0);
        if rel.is_finite() && rel.abs() <= tol {
            Ok(())
        } else {
            Err(Reject::DomainViolation)
        }
    }

    /// Geodesic angle between two validated sphere points.
    ///
    /// Spec 25 (2.3) writes `theta = arccos(<s, t> / R^2)`. The argument is
    /// checked against the band that the norm tolerance can produce
    /// (`|<s,t>|/R^2 <= 1 + 2 tol` plus rounding); outside that band the input
    /// is rejected. No clamp is applied: the angle is evaluated as
    /// `2 atan2(||s - t||, ||s + t||)`, which equals the arccos form on the
    /// sphere, never leaves its domain, and keeps full precision near 0 and pi
    /// where `acos` loses about half the significant digits.
    pub fn sphere_angle(radius: f64, s: &[f64], t: &[f64], tol: f64) -> Result<f64> {
        if s.len() != t.len() || s.is_empty() {
            return Err(Reject::DomainViolation);
        }
        sphere_check(radius, s, tol)?;
        sphere_check(radius, t, tol)?;
        let cos_arg = s
            .iter()
            .zip(t)
            .map(|(a, b)| (*a / radius) * (*b / radius))
            .sum::<f64>();
        let band = 2.0 * tol + 8.0 * f64::EPSILON * s.len() as f64;
        if !cos_arg.is_finite() || cos_arg.abs() > 1.0 + band {
            return Err(Reject::DomainViolation);
        }
        // Normalize each operand before subtraction/addition. Even if both
        // raw coordinates are near `f64::MAX`, their dimensionless values are
        // bounded by the sphere norm and cannot overflow.
        let diff: f64 = s
            .iter()
            .zip(t)
            .fold(0.0, |acc, (a, b)| acc.hypot(*a / radius - *b / radius));
        let sum: f64 = s
            .iter()
            .zip(t)
            .fold(0.0, |acc, (a, b)| acc.hypot(*a / radius + *b / radius));
        if !diff.is_finite() || !sum.is_finite() {
            return Err(Reject::DomainViolation);
        }
        let angle = 2.0 * diff.atan2(sum);
        if angle == 0.0 && s != t {
            // Normalizing by a huge radius can erase a physical separation.
            return Err(Reject::NonFiniteState);
        }
        if angle.is_finite() {
            Ok(angle)
        } else {
            Err(Reject::NonFiniteState)
        }
    }

    /// `d_S(s, t) = R * theta`.
    pub fn spherical_distance(radius: f64, s: &[f64], t: &[f64], tol: f64) -> Result<f64> {
        let angle = sphere_angle(radius, s, t, tol)?;
        let distance = radius * angle;
        if distance.is_finite() {
            Ok(distance)
        } else {
            Err(Reject::NonFiniteState)
        }
    }

    /// Orthogonal tangent projection `z - s <s, z> / R^2`.
    pub fn sphere_project(radius: f64, s: &[f64], z: &[f64], out: &mut [f64]) -> Result<()> {
        check_same_lengths(out, s, z)?;
        sphere_check(radius, s, SPHERE_NORM_TOL)?;
        ensure_finite(z)?;
        let unit_s: Vec<f64> = s.iter().map(|value| *value / radius).collect();
        let component = dot_unit_scaled(&unit_s, z);
        for ((o, zi), si) in out.iter_mut().zip(z).zip(&unit_s) {
            *o = *zi - component * *si;
        }
        ensure_finite(out)
    }

    /// `log_s(t)`: tangent at `s` of length `R theta` towards `t`.
    /// Rejects `theta >= pi - CUT_LOCUS_TOL` with `CutLocus`.
    pub fn spherical_log(
        radius: f64,
        s: &[f64],
        t: &[f64],
        tol: f64,
        out: &mut [f64],
    ) -> Result<()> {
        check_same_lengths(out, s, t)?;
        let theta = sphere_angle(radius, s, t, tol)?;
        if theta >= PI - CUT_LOCUS_TOL {
            return Err(Reject::CutLocus);
        }
        let unit_s: Vec<f64> = s.iter().map(|value| *value / radius).collect();
        let unit_t: Vec<f64> = t.iter().map(|value| *value / radius).collect();
        // u = t/R - cos(theta) s/R, the dimensionless tangent direction.
        let k = dot(&unit_s, &unit_t);
        for ((o, ti), si) in out.iter_mut().zip(&unit_t).zip(&unit_s) {
            *o = *ti - k * *si;
        }
        let un = norm(out);
        if theta == 0.0 || un == 0.0 {
            out.fill(0.0);
            return Ok(());
        }
        let length = radius * theta;
        if !length.is_finite() {
            return Err(Reject::NonFiniteState);
        }
        for o in out.iter_mut() {
            *o = length * (*o / un);
        }
        ensure_finite(out)
    }

    /// `exp_s(v) = cos(||v||/R) s + R sin(||v||/R) v / ||v||`.
    /// `v` must be tangent at `s`; `||v|| >= pi R` leaves the injectivity
    /// radius and is rejected with `CutLocus`.
    pub fn spherical_exp(
        radius: f64,
        s: &[f64],
        v: &[f64],
        tol: f64,
        out: &mut [f64],
    ) -> Result<()> {
        check_same_lengths(out, s, v)?;
        sphere_check(radius, s, tol)?;
        ensure_finite(v)?;
        let vn = norm(v);
        let unit_s: Vec<f64> = s.iter().map(|value| *value / radius).collect();
        let projection = dot_unit_scaled(&unit_s, v).abs();
        let (orth_ratio, allowed_ratio) = if vn >= 1.0 {
            (projection / vn, SPHERE_TANGENT_TOL * (1.0 + 1.0 / vn))
        } else {
            (projection / (1.0 + vn), SPHERE_TANGENT_TOL)
        };
        if !orth_ratio.is_finite() || !allowed_ratio.is_finite() || orth_ratio > allowed_ratio {
            return Err(Reject::DomainViolation);
        }
        if vn == 0.0 {
            out.copy_from_slice(s);
            return Ok(());
        }
        let theta = vn / radius;
        if theta >= PI - CUT_LOCUS_TOL {
            return Err(Reject::CutLocus);
        }
        let (sin_t, cos_t) = theta.sin_cos();
        for ((o, si), vi) in out.iter_mut().zip(&unit_s).zip(v) {
            let unit_value = cos_t * *si + sin_t * (*vi / vn);
            *o = radius * unit_value;
        }
        ensure_finite(out)?;
        sphere_check(radius, out, tol)
    }

    /// Parallel transport along the minimizing geodesic `s -> t`:
    /// `P(v) = v - (<t, v> / (R^2 + <s, t>)) (s + t)`.
    pub fn spherical_transport(
        radius: f64,
        s: &[f64],
        t: &[f64],
        v: &[f64],
        tol: f64,
        out: &mut [f64],
    ) -> Result<()> {
        check_same_lengths4(out, s, t, v)?;
        let theta = sphere_angle(radius, s, t, tol)?;
        if theta >= PI - CUT_LOCUS_TOL {
            return Err(Reject::CutLocus);
        }
        ensure_finite(v)?;
        let unit_s: Vec<f64> = s.iter().map(|value| *value / radius).collect();
        let unit_t: Vec<f64> = t.iter().map(|value| *value / radius).collect();
        let den = 1.0 + dot(&unit_s, &unit_t);
        if !(den > 0.0 && den.is_finite()) {
            return Err(Reject::CutLocus);
        }
        let k = dot_unit_scaled(&unit_t, v) / den;
        for (((o, vi), si), ti) in out.iter_mut().zip(v).zip(&unit_s).zip(&unit_t) {
            *o = *vi - k * (*si + *ti);
        }
        ensure_finite(out)
    }
}

// ---------------------------------------------------------------------------
// ProductManifold: the ProductGeometry implementation
// ---------------------------------------------------------------------------

/// Snapshot-fixed geometry parameters. All must be finite and > 0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryParams {
    pub curvature: f64,
    pub radius: f64,
    pub alpha_h: f64,
    pub alpha_e: f64,
    pub alpha_s: f64,
}

impl GeometryParams {
    /// `c = 1`, `R = 1`, unit weights.
    pub const UNIT: Self = Self {
        curvature: 1.0,
        radius: 1.0,
        alpha_h: 1.0,
        alpha_e: 1.0,
        alpha_s: 1.0,
    };

    pub fn validate(&self) -> Result<()> {
        let all = [
            self.curvature,
            self.radius,
            self.alpha_h,
            self.alpha_e,
            self.alpha_s,
        ];
        if !all.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(Reject::DomainViolation);
        }
        // Each weight finite is not enough: `1e308 x 3` overflows to `+inf`,
        // and the confidence average would then be `inf / inf = NaN`.
        let alpha_sum = self.alpha_h + self.alpha_e + self.alpha_s;
        if alpha_sum.is_finite() && alpha_sum > 0.0 {
            Ok(())
        } else {
            Err(Reject::DomainViolation)
        }
    }

    /// Geometry digest bound into `Epochs::geometry` and `FiberId::frame`.
    pub fn digest(&self, layout: Layout) -> Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"gen-zero/manifold/product-geometry/v1");
        layout.hash_into(&mut hasher);
        for v in [
            self.curvature,
            self.radius,
            self.alpha_h,
            self.alpha_e,
            self.alpha_s,
        ] {
            hasher.update(&v.to_bits().to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

/// `H_{-c}^{d_h} x R^{d_e} x S_R^{d_s}` with metric weights `alpha`.
#[derive(Clone, Debug)]
pub struct ProductManifold {
    layout: Layout,
    params: GeometryParams,
    epochs: Epochs,
    frame: Digest,
}

impl ProductManifold {
    /// `epochs.geometry` must equal `params.digest(layout)`: a snapshot cannot
    /// carry points from one geometry into another.
    pub fn new(layout: Layout, params: GeometryParams, epochs: Epochs) -> Result<Self> {
        Layout::new(layout.h, layout.e, layout.s_intrinsic)?;
        params.validate()?;
        let frame = params.digest(layout);
        if epochs.geometry != frame {
            return Err(Reject::EpochMismatch);
        }
        Ok(Self {
            layout,
            params,
            epochs,
            frame,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    /// Stored width of a point of this manifold.
    pub fn dim(&self) -> usize {
        self.layout.store_dim()
    }
    pub fn params(&self) -> GeometryParams {
        self.params
    }
    pub fn epochs(&self) -> &Epochs {
        &self.epochs
    }
    pub fn frame(&self) -> Digest {
        self.frame
    }

    /// Validates raw stored coordinates and stamps them with this snapshot.
    pub fn point(&self, coords: &[f64]) -> Result<Point> {
        self.validate_coords(coords)?;
        Ok(Point {
            layout: self.layout,
            epochs: self.epochs.clone(),
            coords: coords.into(),
        })
    }

    /// Riemannian norm `sqrt(alpha_h lambda_x^2 |v_H|^2 + alpha_e |v_E|^2 + alpha_s |v_S|^2)`.
    pub fn tangent_norm(&self, base: &Point, tangent: &Tangent) -> Result<f64> {
        self.check_point(base)?;
        self.check_tangent(base, tangent)?;
        let l = &self.layout;
        let conf = kernel::ball_conformal(self.params.curvature, &base.coords[l.h_range()])?;
        let lambda = 2.0 / conf;
        let v = &tangent.coords;
        let h = self.params.alpha_h.sqrt() * lambda * norm(&v[l.h_range()]);
        let e = self.params.alpha_e.sqrt() * norm(&v[l.e_range()]);
        let s = self.params.alpha_s.sqrt() * norm(&v[l.s_range()]);
        let result = h.hypot(e).hypot(s);
        if result.is_finite() {
            Ok(result)
        } else {
            Err(Reject::NonFiniteState)
        }
    }

    fn validate_coords(&self, coords: &[f64]) -> Result<()> {
        let l = &self.layout;
        if coords.len() != l.store_dim() {
            return Err(Reject::DomainViolation);
        }
        ensure_finite(coords)?;
        kernel::ball_conformal(self.params.curvature, &coords[l.h_range()])?;
        kernel::sphere_check(self.params.radius, &coords[l.s_range()], SPHERE_NORM_TOL)
    }

    fn check_point(&self, p: &Point) -> Result<()> {
        if p.layout != self.layout {
            return Err(Reject::DomainViolation);
        }
        if p.epochs != self.epochs {
            return Err(Reject::EpochMismatch);
        }
        // Coordinates are validated at construction; re-check so that no
        // operator ever runs on an out-of-domain point.
        self.validate_coords(&p.coords)
    }

    fn check_tangent(&self, base: &Point, t: &Tangent) -> Result<()> {
        if t.fiber.epochs != self.epochs {
            return Err(Reject::EpochMismatch);
        }
        if t.fiber.patch != GLOBAL_PATCH || t.fiber.frame != self.frame {
            return Err(Reject::FiberMismatch);
        }
        if t.fiber.base != base.digest() {
            return Err(Reject::FiberMismatch);
        }
        if t.coords.len() != self.layout.store_dim() {
            return Err(Reject::DomainViolation);
        }
        ensure_finite(&t.coords)
    }

    fn fresh_path(base: &Digest) -> Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"gen-zero/manifold/path/v1");
        hasher.update(base);
        *hasher.finalize().as_bytes()
    }

    fn tangent_at(&self, base: &Point, coords: Vec<f64>, path: Option<Digest>) -> Tangent {
        let base_digest = base.digest();
        let path = path.unwrap_or_else(|| Self::fresh_path(&base_digest));
        Tangent {
            fiber: FiberId {
                patch: GLOBAL_PATCH,
                base: base_digest,
                frame: self.frame,
                path,
                epochs: self.epochs.clone(),
            },
            coords: coords.into_boxed_slice(),
        }
    }
}

impl ProductGeometry for ProductManifold {
    fn log(&self, base: &Point, point: &Point) -> Result<Tangent> {
        self.check_point(base)?;
        self.check_point(point)?;
        let l = &self.layout;
        let (x, y) = (&base.coords, &point.coords);
        let mut out = vec![0.0; l.store_dim()];
        kernel::hyperbolic_log(
            self.params.curvature,
            &x[l.h_range()],
            &y[l.h_range()],
            &mut out[l.h_range()],
        )?;
        for i in l.e_range() {
            out[i] = y[i] - x[i];
        }
        kernel::spherical_log(
            self.params.radius,
            &x[l.s_range()],
            &y[l.s_range()],
            SPHERE_NORM_TOL,
            &mut out[l.s_range()],
        )?;
        Ok(self.tangent_at(base, out, None))
    }

    fn exp(&self, base: &Point, tangent: &Tangent) -> Result<Point> {
        self.check_point(base)?;
        self.check_tangent(base, tangent)?;
        let l = &self.layout;
        let (x, v) = (&base.coords, &tangent.coords);
        let mut out = vec![0.0; l.store_dim()];
        kernel::hyperbolic_exp(
            self.params.curvature,
            &x[l.h_range()],
            &v[l.h_range()],
            &mut out[l.h_range()],
        )?;
        for i in l.e_range() {
            out[i] = x[i] + v[i];
        }
        kernel::spherical_exp(
            self.params.radius,
            &x[l.s_range()],
            &v[l.s_range()],
            SPHERE_NORM_TOL,
            &mut out[l.s_range()],
        )?;
        self.point(&out)
    }

    fn distance(&self, lhs: &Point, rhs: &Point) -> Result<f64> {
        self.check_point(lhs)?;
        self.check_point(rhs)?;
        let l = &self.layout;
        let (x, y) = (&lhs.coords, &rhs.coords);
        let dh =
            kernel::hyperbolic_distance(self.params.curvature, &x[l.h_range()], &y[l.h_range()])?;
        let de = norm_diff(&x[l.e_range()], &y[l.e_range()]);
        let ds = kernel::spherical_distance(
            self.params.radius,
            &x[l.s_range()],
            &y[l.s_range()],
            SPHERE_NORM_TOL,
        )?;
        let d = (self.params.alpha_h.sqrt() * dh)
            .hypot(self.params.alpha_e.sqrt() * de)
            .hypot(self.params.alpha_s.sqrt() * ds);
        if d.is_finite() {
            Ok(d)
        } else {
            Err(Reject::NonFiniteState)
        }
    }

    fn project_tangent(&self, base: &Point, ambient: &[f64]) -> Result<Tangent> {
        self.check_point(base)?;
        let l = &self.layout;
        if ambient.len() != l.store_dim() {
            return Err(Reject::DomainViolation);
        }
        ensure_finite(ambient)?;
        // The Poincare ball is open in R^{d_h} and the Euclidean factor is flat,
        // so both tangent spaces are the full coordinate space.
        let mut out = ambient.to_vec();
        kernel::sphere_project(
            self.params.radius,
            &base.coords[l.s_range()],
            &ambient[l.s_range()],
            &mut out[l.s_range()],
        )?;
        Ok(self.tangent_at(base, out, None))
    }

    /// Transports along the piecewise geodesic `path[0] -> path[1] -> ...`.
    /// `path[0]` must be the tangent's base; an empty path is a fiber mismatch.
    fn transport(&self, path: &[Point], tangent: &Tangent) -> Result<Tangent> {
        let first = path.first().ok_or(Reject::FiberMismatch)?;
        for p in path {
            self.check_point(p)?;
        }
        self.check_tangent(first, tangent)?;
        let l = &self.layout;
        let mut v = tangent.coords.to_vec();
        let mut next = vec![0.0; l.store_dim()];
        let mut path_hasher = blake3::Hasher::new();
        path_hasher.update(b"gen-zero/manifold/path/v1");
        path_hasher.update(&tangent.fiber.path);
        for pair in path.windows(2) {
            let (x, y) = (&pair[0].coords, &pair[1].coords);
            kernel::hyperbolic_transport(
                self.params.curvature,
                &x[l.h_range()],
                &y[l.h_range()],
                &v[l.h_range()],
                &mut next[l.h_range()],
            )?;
            next[l.e_range()].copy_from_slice(&v[l.e_range()]);
            kernel::spherical_transport(
                self.params.radius,
                &x[l.s_range()],
                &y[l.s_range()],
                &v[l.s_range()],
                SPHERE_NORM_TOL,
                &mut next[l.s_range()],
            )?;
            std::mem::swap(&mut v, &mut next);
            path_hasher.update(&pair[1].digest());
        }
        let last = path.last().ok_or(Reject::FiberMismatch)?;
        let path_digest = if path.len() == 1 {
            tangent.fiber.path
        } else {
            *path_hasher.finalize().as_bytes()
        };
        Ok(self.tangent_at(last, v, Some(path_digest)))
    }

    fn busemann_containment(
        &self,
        passage: &[f64],
        question: &[f64],
        cone_half_angle: f64,
    ) -> std::result::Result<ContainmentScore, LodError> {
        self.busemann_containment_with(
            passage,
            question,
            &ContainmentCriteria::from_cone_half_angle(cone_half_angle),
        )
    }
}

// ---------------------------------------------------------------------------
// Preset topologies and Busemann entailment (scheme 1)
// ---------------------------------------------------------------------------

/// Hyperbolic radial directions shorter than this (in unit-ball scale) are
/// undefined: the angle test would divide by ~0.
pub const RADIAL_DIRECTION_EPS: f64 = 1e-9;
/// Isotropy constant `K` of the entailment-cone aperture (Ganea et al. 2018
/// use 0.1). A preset, not a fitted value.
pub const DEFAULT_APERTURE_K: f64 = 0.1;
/// Largest Euclidean topic shift (norm of the `R^24` difference) that still
/// counts as the same topic. Uncalibrated preset.
pub const DEFAULT_TOPIC_SHIFT_TOL: f64 = 1.0;

/// Thresholds of one Busemann entailment test. All must be finite and > 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContainmentCriteria {
    /// Upper cap on the cone half-angle at the passage, in `(0, pi]`.
    pub cone_half_angle: f64,
    /// `K` in the aperture `psi(p) = asin(K (1 - r^2) / r)`, `r = sqrt(c) |p_H|`.
    pub aperture_k: f64,
    /// Largest sphere angle (radians, `<= pi`) absorbed as a rhetorical variant.
    pub sphere_absorb_angle: f64,
    /// Largest Euclidean topic shift accepted.
    pub topic_shift_tol: f64,
}

impl ContainmentCriteria {
    /// The sphere factor gets the same angular budget as the cone; the aperture
    /// constant and topic tolerance are the documented presets.
    pub fn from_cone_half_angle(cone_half_angle: f64) -> Self {
        Self {
            cone_half_angle,
            aperture_k: DEFAULT_APERTURE_K,
            sphere_absorb_angle: cone_half_angle,
            topic_shift_tol: DEFAULT_TOPIC_SHIFT_TOL,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let pos = |v: f64| v.is_finite() && v > 0.0;
        if pos(self.cone_half_angle)
            && self.cone_half_angle <= PI
            && pos(self.aperture_k)
            && pos(self.sphere_absorb_angle)
            && self.sphere_absorb_angle <= PI
            && pos(self.topic_shift_tol)
        {
            Ok(())
        } else {
            Err(Reject::DomainViolation)
        }
    }
}

/// Verdict and evidence of one `passage ⊃ question` test.
///
/// `confidence` is the `alpha_h/e/s`-weighted average of the three factor
/// margins (`H` takes the weaker of its cone and depth sub-margins), 0 when
/// not entailed. It is a margin score, not a calibrated probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContainmentScore {
    pub is_entailed: bool,
    pub confidence: f64,
    /// Geodesic distance `d_H(p, q)` in the hyperbolic factor.
    pub hyperbolic_distance: f64,
    /// Angle between the radial geodesics (`log_0` directions) of `p` and `q`.
    pub cone_angle: f64,
    /// Effective cone half-angle at `p`: `min(cap, psi(p))`.
    pub aperture: f64,
    /// `B_xi(p) - B_xi(q)` with `xi = p / |p|`; `> 0` means `q` is strictly
    /// deeper towards the ideal point than `p`.
    pub busemann_depth_gain: f64,
    /// Spherical centre angle between the sphere parts.
    pub sphere_angle: f64,
    /// Euclidean distance between the Euclidean parts.
    pub topic_shift: f64,
    pub in_cone: bool,
    pub deeper: bool,
    pub sphere_absorbed: bool,
    pub topic_aligned: bool,
    /// Gaussian violation energy: the `alpha_h/e/s`-weighted mean of the squared
    /// normalised distances by which each factor test is missed. `H` adds its
    /// cone excess `(angle - aperture) / aperture` and its depth deficit
    /// `sqrt(c) * max(0, -depth_gain)`; `S` and `R` use the excess over their
    /// threshold, divided by the threshold. 0 when no test is missed by a
    /// positive distance. Evidence only: `is_entailed` is the gate.
    pub violation_energy: f64,
    /// `exp(-violation_energy / 2)` in `[0, 1]`: 1 at zero violation, falling
    /// continuously with it. Uncalibrated; it never turns a refusal into a pass.
    pub soft_confidence: f64,
}

/// Busemann function of the Poincare ball of curvature `-c` at the ideal
/// point `xi` (a unit vector), normalised so `B_xi(0) = 0`:
/// `B(x) = ln(|xi - sqrt(c) x|^2 / (1 - c |x|^2)) / sqrt(c)`.
/// Along the ray to `xi` it equals minus the distance from the origin.
fn busemann(c: f64, xi: &[f64], x: &[f64]) -> Result<f64> {
    let conf = kernel::ball_conformal(c, x)?;
    let sc = c.sqrt();
    if xi.len() != x.len() || xi.is_empty() {
        return Err(Reject::DomainViolation);
    }
    ensure_finite(xi)?;
    let num_norm: f64 = xi
        .iter()
        .zip(x)
        .fold(0.0, |acc, (a, b)| acc.hypot(*a - sc * *b));
    let num = num_norm * num_norm;
    let ratio = num / conf;
    if !(ratio > 0.0 && ratio.is_finite()) {
        return Err(Reject::NonFiniteState);
    }
    let b = ratio.ln() / sc;
    if b.is_finite() {
        Ok(b)
    } else {
        Err(Reject::NonFiniteState)
    }
}

fn unit(v: &[f64], n: f64) -> Vec<f64> {
    v.iter().map(|x| x / n).collect()
}

fn norm_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |acc, (x, y)| acc.hypot(*x - *y))
}

impl ProductManifold {
    /// A whitelisted preset layout with explicit parameters. `base.geometry`
    /// is replaced by the product-geometry digest, which `new` requires; the
    /// other five epoch fields are kept as given.
    pub fn from_preset_with(
        preset: TopologyPreset,
        params: GeometryParams,
        base: Epochs,
    ) -> Result<Self> {
        let layout = preset.layout();
        let epochs = Epochs {
            geometry: params.digest(layout),
            ..base
        };
        Self::new(layout, params, epochs)
    }

    /// Busemann entailment `passage ⊃ question` with explicit thresholds.
    ///
    /// Four factor tests, all must hold:
    /// 1. `H`: the radial angle between `p` and `q` is at most the aperture
    ///    `min(cap, asin(K (1 - r^2) / r))` at `p` (general concepts sit near
    ///    the origin and get wide cones; a ratio `>= 1` saturates at `pi/2`);
    /// 2. `H`: `q` is strictly deeper than `p` along `xi = p/|p|`, i.e.
    ///    `B_xi(q) < B_xi(p)`. Because `B_xi(x) >= -d(0, x)`, this forces
    ///    `d(0, q) > d(0, p)`, so the reverse test can never also pass;
    /// 3. `S`: the centre angle is at most `sphere_absorb_angle`;
    /// 4. `R`: the topic shift is at most `topic_shift_tol`.
    ///
    /// Not yet validated on any data: the thresholds are presets.
    pub fn busemann_containment_with(
        &self,
        passage: &[f64],
        question: &[f64],
        criteria: &ContainmentCriteria,
    ) -> std::result::Result<ContainmentScore, LodError> {
        criteria.validate()?;
        self.validate_coords(passage)?;
        self.validate_coords(question)?;
        let l = &self.layout;
        let c = self.params.curvature;
        let sc = c.sqrt();
        let (hp, hq) = (&passage[l.h_range()], &question[l.h_range()]);
        let (np, nq) = (norm(hp), norm(hq));
        if sc * np < RADIAL_DIRECTION_EPS {
            return Err(LodError::DegenerateRadialDirection("passage"));
        }
        if sc * nq < RADIAL_DIRECTION_EPS {
            return Err(LodError::DegenerateRadialDirection("question"));
        }
        let (up, uq) = (unit(hp, np), unit(hq, nq));

        // Angle between the log_0 directions (log_0 is a radial rescaling).
        // atan2 form: no clamp, no acos precision loss near 0 and pi.
        let diff = norm_diff(&up, &uq);
        let sum: f64 = up
            .iter()
            .zip(&uq)
            .fold(0.0, |acc, (a, b)| acc.hypot(*a + *b));
        let cone_angle = 2.0 * diff.atan2(sum);

        let r = sc * np;
        let conf_p = kernel::ball_conformal(c, hp)?;
        if !(r.is_finite() && r > 0.0 && r < 1.0) {
            return Err(Reject::DomainViolation.into());
        }
        let aperture_argument = if criteria.aperture_k >= r / conf_p {
            1.0
        } else {
            criteria.aperture_k * conf_p / r
        };
        let psi = aperture_argument.min(1.0).asin();
        let aperture = criteria.cone_half_angle.min(psi);
        if !(aperture > 0.0 && aperture.is_finite()) {
            return Err(Reject::DomainViolation.into());
        }

        let hyperbolic_distance = kernel::hyperbolic_distance(c, hp, hq)?;
        let depth_gain = busemann(c, &up, hp)? - busemann(c, &up, hq)?;
        let sphere_angle = kernel::sphere_angle(
            self.params.radius,
            &passage[l.s_range()],
            &question[l.s_range()],
            SPHERE_NORM_TOL,
        )?;
        let topic_shift = norm_diff(&passage[l.e_range()], &question[l.e_range()]);
        for v in [cone_angle, aperture, depth_gain, topic_shift] {
            if !v.is_finite() {
                return Err(Reject::NonFiniteState.into());
            }
        }

        let in_cone = cone_angle <= aperture;
        let deeper = depth_gain > 0.0;
        let sphere_absorbed = sphere_angle <= criteria.sphere_absorb_angle;
        let topic_aligned = topic_shift <= criteria.topic_shift_tol;
        let is_entailed = in_cone && deeper && sphere_absorbed && topic_aligned;
        let confidence = if is_entailed {
            // `H` covers two tests (cone, depth); its margin is the weaker of
            // the two. Combine with the same `alpha_h/e/s` weights the metric
            // uses (module doc, `riemannian_norm`), not an unweighted min.
            let margin_h = (1.0 - cone_angle / aperture).min(-(-depth_gain * sc).exp_m1());
            let margin_e = 1.0 - topic_shift / criteria.topic_shift_tol;
            let margin_s = 1.0 - sphere_angle / criteria.sphere_absorb_angle;
            // Scale by the largest weight first: each ratio is in (0, 1], so
            // neither the sum nor the numerator can overflow.
            let p = &self.params;
            let max_alpha = p.alpha_h.max(p.alpha_e).max(p.alpha_s);
            if !max_alpha.is_finite() || max_alpha <= 0.0 {
                return Err(Reject::DomainViolation.into());
            }
            let (ah, ae, as_) = (
                p.alpha_h / max_alpha,
                p.alpha_e / max_alpha,
                p.alpha_s / max_alpha,
            );
            let weight_sum = ah + ae + as_;
            let weighted = (ah * margin_h + ae * margin_e + as_ * margin_s) / weight_sum;
            // `clamp` passes NaN through; refuse it rather than report it.
            if !weighted.is_finite() {
                return Err(Reject::NonFiniteState.into());
            }
            weighted.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if !confidence.is_finite() {
            return Err(Reject::NonFiniteState.into());
        }
        let violation_energy = {
            let excess = |value: f64, limit: f64| (value - limit).max(0.0) / limit;
            let v_cone = excess(cone_angle, aperture);
            let v_depth = (-depth_gain).max(0.0) * sc;
            let v_e = excess(topic_shift, criteria.topic_shift_tol);
            let v_s = excess(sphere_angle, criteria.sphere_absorb_angle);
            let p = &self.params;
            let max_alpha = p.alpha_h.max(p.alpha_e).max(p.alpha_s);
            let (ah, ae, as_) = (
                p.alpha_h / max_alpha,
                p.alpha_e / max_alpha,
                p.alpha_s / max_alpha,
            );
            (ah * (v_cone * v_cone + v_depth * v_depth) + ae * v_e * v_e + as_ * v_s * v_s)
                / (ah + ae + as_)
        };
        // An overflowed energy is refused, not reported as "certainly not entailed".
        if !(violation_energy.is_finite() && violation_energy >= 0.0) {
            return Err(Reject::NonFiniteState.into());
        }
        let soft_confidence = (-0.5 * violation_energy).exp();
        Ok(ContainmentScore {
            is_entailed,
            confidence,
            hyperbolic_distance,
            cone_angle,
            aperture,
            busemann_depth_gain: depth_gain,
            sphere_angle,
            topic_shift,
            in_cone,
            deeper,
            sphere_absorbed,
            topic_aligned,
            violation_energy,
            soft_confidence,
        })
    }
}

// ---------------------------------------------------------------------------
// 16-coordinate Lod graph coordinate
// ---------------------------------------------------------------------------

/// Norm tolerance for f32 sphere coordinates (f32 rounding of a unit vector).
const COORD_SPHERE_TOL: f64 = 4e-6;
/// Smallest `1 - c ||x_H||^2` a constructor accepts.
pub(crate) const COORD_BOUNDARY_FLOOR: f32 = 1e-4;

/// 16-coordinate chart of `H_{-c}^4 x R^8 x S_R^3` for Lod graph nodes.
///
/// The coordinate carries no geometry parameters of its own. Curvature `c`,
/// sphere radius `R` and the metric weights come from the [`GeometryParams`] of
/// the graph that holds the node:
/// - `hyperbolic` is a Poincare ball point and must satisfy `c ||x||^2 < 1`;
/// - `spherical` is a unit direction; the point on `S_R^3` is `R * spherical`;
/// - `euclidean` is flat.
///
/// Guaranteed to occupy exactly 64 bytes (1 CPU cache line).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixedCurvatureCoord {
    /// Poincare ball coordinates (H^4).
    pub hyperbolic: [f32; 4],
    /// Unit direction of the sphere point (S^3 in R^4).
    pub spherical: [f32; 4],
    /// Flat Euclidean coordinates (R^8).
    pub euclidean: [f32; 8],
}

const _: () = assert!(std::mem::size_of::<MixedCurvatureCoord>() == 64);
const _: () = assert!(std::mem::align_of::<MixedCurvatureCoord>() == 64);

impl Default for MixedCurvatureCoord {
    fn default() -> Self {
        Self::origin()
    }
}

fn widen<const N: usize>(a: &[f32; N]) -> [f64; N] {
    a.map(f64::from)
}

impl MixedCurvatureCoord {
    /// Layout of this coordinate in Spec 25 terms: `H^4 x R^8 x S^3`.
    pub const LAYOUT: Layout = Layout {
        h: 4,
        e: 8,
        s_intrinsic: 3,
    };

    /// Standard origin coordinate
    pub fn origin() -> Self {
        let mut spherical = [0.0_f32; 4];
        spherical[0] = 1.0; // North pole on S^3
        Self {
            hyperbolic: [0.0; 4],
            spherical,
            euclidean: [0.0; 8],
        }
    }

    /// [`Self::with_curvature`] on the unit ball (`c = 1`).
    pub fn new(
        hyperbolic: [f32; 4],
        spherical: [f32; 4],
        euclidean: [f32; 8],
    ) -> std::result::Result<Self, LodError> {
        Self::with_curvature(hyperbolic, spherical, euclidean, 1.0)
    }

    /// Construct for a ball of curvature `-curvature`, with a boundary floor on
    /// the hyperbolic block (`1 - c ||x_H||^2 >= 1e-4`). A non-zero spherical
    /// block is normalized to a unit direction; a zero or non-finite block is
    /// rejected.
    pub fn with_curvature(
        hyperbolic: [f32; 4],
        spherical: [f32; 4],
        euclidean: [f32; 8],
        curvature: f32,
    ) -> std::result::Result<Self, LodError> {
        if !(curvature.is_finite() && curvature > 0.0) {
            return Err(Reject::DomainViolation.into());
        }
        let h_norm_sq = gen_zero_core::dot_product_f32(&hyperbolic, &hyperbolic);
        if !h_norm_sq.is_finite() || (1.0 - curvature * h_norm_sq < COORD_BOUNDARY_FLOOR) {
            return Err(LodError::HyperbolicBoundaryViolation { norm_sq: h_norm_sq });
        }
        if !spherical
            .iter()
            .chain(euclidean.iter())
            .all(|v| v.is_finite())
        {
            return Err(Reject::NonFiniteState.into());
        }
        let s_norm = gen_zero_core::dot_product_f32(&spherical, &spherical).sqrt();
        // An overflowed (infinite) norm would silently normalize to zero.
        if !s_norm.is_finite() || s_norm <= 1e-6 {
            return Err(Reject::DomainViolation.into());
        }
        Ok(Self {
            hyperbolic,
            spherical: spherical.map(|v| v / s_norm),
            euclidean,
        })
    }

    /// The validated [`Point`] of `manifold` this coordinate names: stored as
    /// `[H | E | S]` with the unit direction scaled onto the sphere of radius
    /// `R`. Refused when the manifold is not `H^4 x R^8 x S^3`, when the
    /// hyperbolic block is outside the ball of the manifold's curvature, or when
    /// the spherical block is not a unit direction within f32 rounding.
    pub fn to_point(&self, manifold: &ProductManifold) -> std::result::Result<Point, LodError> {
        if *manifold.layout() != Self::LAYOUT {
            return Err(Reject::DomainViolation.into());
        }
        let direction = widen(&self.spherical);
        kernel::sphere_check(1.0, &direction, COORD_SPHERE_TOL)?;
        // Remove the f32 rounding of the direction, then scale to radius R.
        let scale = manifold.params().radius / norm(&direction);
        let mut coords = [0.0; 16];
        coords[0..4].copy_from_slice(&widen(&self.hyperbolic));
        coords[4..12].copy_from_slice(&widen(&self.euclidean));
        for (out, d) in coords[12..16].iter_mut().zip(&direction) {
            *out = d * scale;
        }
        Ok(manifold.point(&coords)?)
    }

    /// Product geodesic distance under explicit geometry parameters:
    /// `sqrt(alpha_h d_H^2 + alpha_e d_E^2 + alpha_s d_S^2)` with `d_H` on the
    /// ball of curvature `-c`, `d_S = r * angle` on the sphere of radius `r`, and
    /// `alphas = [alpha_h, alpha_e, alpha_s]`. The same value as
    /// [`ProductGeometry::distance`] on the points [`Self::to_point`] gives.
    ///
    /// Fails closed: a parameter that is not finite and positive, a hyperbolic
    /// block outside the ball of curvature `-c`, a spherical block that is not a
    /// unit direction, or a non-finite result is an error.
    pub fn product_distance_with_params(
        &self,
        other: &Self,
        alphas: [f32; 3],
        c: f32,
        r: f32,
    ) -> std::result::Result<f32, LodError> {
        let [alpha_h, alpha_e, alpha_s] = alphas.map(f64::from);
        let params = GeometryParams {
            curvature: f64::from(c),
            radius: f64::from(r),
            alpha_h,
            alpha_e,
            alpha_s,
        };
        params.validate()?;
        let dh = kernel::hyperbolic_distance(
            params.curvature,
            &widen(&self.hyperbolic),
            &widen(&other.hyperbolic),
        )?;
        let angle = kernel::sphere_angle(
            1.0,
            &widen(&self.spherical),
            &widen(&other.spherical),
            COORD_SPHERE_TOL,
        )?;
        let ds = params.radius * angle;
        let de = norm_diff(&widen(&self.euclidean), &widen(&other.euclidean));
        let distance = (alpha_h.sqrt() * dh)
            .hypot(alpha_e.sqrt() * de)
            .hypot(alpha_s.sqrt() * ds) as f32;
        if distance.is_finite() {
            Ok(distance)
        } else {
            Err(Reject::NonFiniteState.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    /// Deterministic xorshift64* so the tests need no extra dependency.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-300);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
        fn direction(&mut self, n: usize) -> Vec<f64> {
            let v: Vec<f64> = (0..n).map(|_| self.normal()).collect();
            let vn = norm(&v);
            v.iter().map(|x| x / vn).collect()
        }
    }

    fn epochs_for(layout: Layout, params: GeometryParams) -> Epochs {
        Epochs {
            version: Version(1),
            model: [1; 32],
            geometry: params.digest(layout),
            atlas: [2; 32],
            graph: [3; 32],
            policy: [4; 32],
        }
    }

    fn params() -> GeometryParams {
        GeometryParams {
            curvature: 0.7,
            radius: 2.5,
            alpha_h: 1.3,
            alpha_e: 0.6,
            alpha_s: 0.9,
        }
    }

    fn manifold(layout: Layout) -> ProductManifold {
        let p = params();
        ProductManifold::new(layout, p, epochs_for(layout, p)).unwrap()
    }

    /// Random valid point: hyperbolic radius up to `h_frac / sqrt(c)`.
    fn random_point(m: &ProductManifold, rng: &mut Rng, h_frac: f64) -> Point {
        let l = m.layout();
        let p = m.params();
        let mut coords = vec![0.0; l.store_dim()];
        let dir = rng.direction(l.h());
        let r = h_frac * rng.uniform() / p.curvature.sqrt();
        for (o, d) in coords[l.h_range()].iter_mut().zip(&dir) {
            *o = r * d;
        }
        for o in coords[l.e_range()].iter_mut() {
            *o = 6.0 * rng.uniform() - 3.0;
        }
        let s = rng.direction(l.s_ambient());
        for (o, d) in coords[l.s_range()].iter_mut().zip(&s) {
            *o = p.radius * d;
        }
        m.point(&coords).unwrap()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn topology_presets_are_a_closed_whitelist() {
        use std::str::FromStr;
        let table = [
            ("compact_64d", (32, 16, 15), 64),
            ("balanced_128d", (64, 32, 31), 128),
            ("boolq_128d", (80, 24, 23), 128),
            ("extended_256d", (160, 48, 47), 256),
        ];
        for (preset, (name, (h, e, s), dim)) in TopologyPreset::ALL.iter().zip(table) {
            assert_eq!(preset.as_str(), name);
            assert_eq!(TopologyPreset::from_str(name), Ok(*preset));
            let l = preset.layout();
            assert_eq!((l.h(), l.e(), l.s_intrinsic()), (h, e, s));
            assert_eq!(preset.dim(), dim);
            assert_eq!(l.store_dim(), dim);
            assert!(dim <= MAX_PRESET_DIM);
        }
        assert_eq!(
            TopologyPreset::from_str("deep_128d"),
            Ok(TopologyPreset::Boolq128d)
        );
        assert_eq!(TopologyPreset::Boolq128d.as_str(), "boolq_128d");
        for bad in [
            "",
            "dynamic_100d",
            "random_512d",
            "extended_512d",
            "BOOLQ_128D",
            " boolq_128d",
            "boolq_128d ",
            "compact_64",
            "128",
        ] {
            assert_eq!(
                TopologyPreset::from_str(bad),
                Err(Reject::DomainViolation),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn from_preset_with_seals_the_preset_layout_and_a_distinct_frame() {
        let p = params();
        let base = epochs_for(Layout::STORE_64, p);
        let mut frames = std::collections::HashSet::new();
        for preset in TopologyPreset::ALL {
            let m = ProductManifold::from_preset_with(preset, p, base.clone()).unwrap();
            assert_eq!(*m.layout(), preset.layout());
            assert_eq!(m.dim(), preset.dim());
            assert_eq!(m.epochs().geometry, p.digest(preset.layout()));
            assert_eq!(m.epochs().model, base.model);
            // A point of one width is refused by every other preset.
            let mut x = vec![0.0; m.dim()];
            x[0] = 0.1;
            x[m.layout().s_range().start] = p.radius;
            assert!(m.point(&x).is_ok());
            for other in TopologyPreset::ALL
                .iter()
                .filter(|o| o.dim() != preset.dim())
            {
                let o = ProductManifold::from_preset_with(*other, p, base.clone()).unwrap();
                assert_eq!(o.point(&x).err(), Some(Reject::DomainViolation));
            }
            assert!(frames.insert(m.frame()), "{preset} frame collides");
        }
    }

    /// An entailed `(general, specific)` pair on `m`: same axis-0 direction
    /// (0.02 rad apart), deeper question, sphere at the pole, no topic shift.
    fn entailed_pair(m: &ProductManifold) -> (Vec<f64>, Vec<f64>) {
        let sc = m.params.curvature.sqrt();
        let s0 = m.layout().s_range().start;
        let mut general = vec![0.0; m.dim()];
        general[0] = 0.3 / sc;
        general[s0] = m.params.radius;
        let mut specific = general.clone();
        specific[0] = 0.8 / sc * 0.02f64.cos();
        specific[1] = 0.8 / sc * 0.02f64.sin();
        (general, specific)
    }

    #[test]
    fn overflowing_alpha_sum_is_refused_by_validate() {
        let base = params();
        let huge = GeometryParams {
            alpha_h: 1e308,
            alpha_e: 1e308,
            alpha_s: 1e308,
            ..base
        };
        // Each weight alone passes the old per-field test; the sum is `+inf`.
        assert!([huge.alpha_h, huge.alpha_e, huge.alpha_s]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        assert!((huge.alpha_h + huge.alpha_e + huge.alpha_s).is_infinite());
        assert_eq!(huge.validate(), Err(Reject::DomainViolation));
        for preset in TopologyPreset::ALL {
            let epochs = epochs_for(preset.layout(), huge);
            assert_eq!(
                ProductManifold::new(preset.layout(), huge, epochs).err(),
                Some(Reject::DomainViolation),
                "{preset}"
            );
            assert_eq!(
                ProductManifold::from_preset_with(preset, huge, epochs_for(preset.layout(), base))
                    .err(),
                Some(Reject::DomainViolation),
                "{preset}"
            );
        }
        // Two huge weights overflow too; `f64::MAX` plus a tiny weight does not.
        let two = GeometryParams {
            alpha_h: f64::MAX,
            alpha_e: f64::MAX,
            alpha_s: 1.0,
            ..base
        };
        assert_eq!(two.validate(), Err(Reject::DomainViolation));
        let one = GeometryParams {
            alpha_h: f64::MAX,
            alpha_e: 1e-300,
            alpha_s: 1.0,
            ..base
        };
        assert_eq!(one.validate(), Ok(()));
        for bad in [f64::INFINITY, f64::NAN, 0.0, -1.0, -f64::MIN_POSITIVE] {
            let p = GeometryParams {
                alpha_s: bad,
                ..base
            };
            assert_eq!(p.validate(), Err(Reject::DomainViolation), "{bad}");
        }
    }

    #[test]
    fn extreme_alpha_weights_give_a_finite_confidence_or_a_refusal() {
        let base = params();
        let criteria = ContainmentCriteria::from_cone_half_angle(1.0);
        let unit = GeometryParams {
            alpha_h: 1.0,
            alpha_e: 1.0,
            alpha_s: 1.0,
            ..base
        };
        let m = ProductManifold::from_preset_with(
            TopologyPreset::Compact64d,
            unit,
            epochs_for(TopologyPreset::Compact64d.layout(), unit),
        )
        .unwrap();
        let (g, q) = entailed_pair(&m);
        let reference = m.busemann_containment_with(&g, &q, &criteria).unwrap();
        assert!(reference.is_entailed);
        assert!(reference.confidence.is_finite() && reference.confidence > 0.0);

        // Admissible extremes: finite weights whose sum is finite. The
        // normalised average is finite and in [0, 1].
        for (ah, ae, as_) in [
            (f64::MAX, 1e-300, 1.0),
            (1e-300, 1e-300, 1e-300),
            (f64::MIN_POSITIVE, f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            (1e300, 1e300, 1e300),
        ] {
            let p = GeometryParams {
                alpha_h: ah,
                alpha_e: ae,
                alpha_s: as_,
                ..unit
            };
            let m = ProductManifold::from_preset_with(
                TopologyPreset::Compact64d,
                p,
                epochs_for(TopologyPreset::Compact64d.layout(), p),
            )
            .unwrap();
            let s = m.busemann_containment_with(&g, &q, &criteria).unwrap();
            assert!(s.is_entailed);
            assert!(
                s.confidence.is_finite() && (0.0..=1.0).contains(&s.confidence),
                "({ah}, {ae}, {as_}) -> {}",
                s.confidence
            );
        }
        // Equal weights of any scale give the same average as unit weights.
        let tiny = GeometryParams {
            alpha_h: 1e-300,
            alpha_e: 1e-300,
            alpha_s: 1e-300,
            ..unit
        };
        let m_tiny = ProductManifold::from_preset_with(
            TopologyPreset::Compact64d,
            tiny,
            epochs_for(TopologyPreset::Compact64d.layout(), tiny),
        )
        .unwrap();
        let s = m_tiny.busemann_containment_with(&g, &q, &criteria).unwrap();
        assert!((s.confidence - reference.confidence).abs() < 1e-12);

        // Bypass `validate` (the fields are only reachable inside this
        // module) to prove the scorer itself never emits NaN. `1e308 x 3`
        // is the exact reject-review input: normalised, it equals unit weights.
        let mut forced = m.clone();
        forced.params.alpha_h = 1e308;
        forced.params.alpha_e = 1e308;
        forced.params.alpha_s = 1e308;
        let s = forced.busemann_containment_with(&g, &q, &criteria).unwrap();
        assert!((s.confidence - reference.confidence).abs() < 1e-12);
        for bad in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
            let mut forced = m.clone();
            forced.params.alpha_h = bad;
            forced.params.alpha_e = bad;
            forced.params.alpha_s = bad;
            let err = forced
                .busemann_containment_with(&g, &q, &criteria)
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    LodError::Geometry(Reject::DomainViolation | Reject::NonFiniteState)
                ),
                "{bad}: {err:?}"
            );
        }
    }

    #[test]
    fn tiny_curvature_preserves_small_hyperbolic_displacements() {
        let c = f64::from_bits(1);
        let x = [0.0_f64];
        let y = [1e-200_f64];
        let distance = kernel::hyperbolic_distance(c, &x, &y).unwrap();
        assert!((distance - 2e-200).abs() < 1e-215, "distance={distance:e}");

        let mut logged = [0.0];
        kernel::hyperbolic_log(c, &x, &y, &mut logged).unwrap();
        assert!((logged[0] - 1e-200).abs() < 1e-215, "log={:e}", logged[0]);

        let mut expd = [0.0];
        kernel::hyperbolic_exp(c, &x, &y, &mut expd).unwrap();
        assert!((expd[0] - 1e-200).abs() < 1e-215, "exp={:e}", expd[0]);
    }

    #[test]
    fn tiny_tangents_are_not_erased_and_overflow_does_not_become_identity() {
        let mut out = [0.0; 2];
        kernel::hyperbolic_exp(1.0, &[0.0; 2], &[0.0, 1e-200], &mut out).unwrap();
        assert_eq!(out, [0.0, 1e-200]);
        kernel::spherical_exp(1.0, &[1.0, 0.0], &[0.0, 1e-200], SPHERE_NORM_TOL, &mut out).unwrap();
        assert_eq!(out, [1.0, 1e-200]);
        assert_eq!(
            kernel::hyperbolic_exp(1.0, &[0.0; 2], &[f64::MAX; 2], &mut out),
            Err(Reject::NonFiniteState)
        );
        assert_eq!(
            kernel::sphere_angle(1e200, &[1e200, 0.0], &[1e200, 1e-200], SPHERE_NORM_TOL),
            Err(Reject::NonFiniteState)
        );

        // Retain a tiny orthogonal component even when another dimension has
        // non-negligible curvature. Scaling the entire result back loses it.
        let c = f64::from_bits(1);
        let base = [0.5 / c.sqrt(), 0.0];
        kernel::hyperbolic_exp(c, &base, &[0.0, 1e-200], &mut out).unwrap();
        assert!((out[1] / 1e-200 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn subnormal_sphere_validation_and_near_boundary_cancellation() {
        let tiny = f64::from_bits(1);
        assert_eq!(
            kernel::sphere_check(tiny, &[tiny, tiny], SPHERE_NORM_TOL),
            Err(Reject::DomainViolation)
        );
        assert!(kernel::sphere_check(tiny, &[tiny, 0.0], SPHERE_NORM_TOL).is_ok());
        let x = [1.0 - 1e-8];
        let mut out = [1.0];
        kernel::mobius_add(1.0, &x, &[-x[0]], &mut out).unwrap();
        assert_eq!(out, [0.0]);
        kernel::hyperbolic_log(1.0, &x, &x, &mut out).unwrap();
        assert_eq!(out, [0.0]);
    }

    #[test]
    fn extreme_scale_log_exp_and_transport_round_trip() {
        for radius in [1e-200, 1e200] {
            let s = [radius, 0.0];
            let t = [radius * 0.4_f64.cos(), radius * 0.4_f64.sin()];
            let mut logged = [0.0; 2];
            kernel::spherical_log(radius, &s, &t, SPHERE_NORM_TOL, &mut logged).unwrap();
            assert!((norm(&logged) / radius - 0.4).abs() < 1e-14);
            let mut recovered = [0.0; 2];
            kernel::spherical_exp(radius, &s, &logged, SPHERE_NORM_TOL, &mut recovered).unwrap();
            assert!((recovered[0] / radius - t[0] / radius).abs() < 1e-14);
            assert!((recovered[1] / radius - t[1] / radius).abs() < 1e-14);
            let mut transported = [0.0; 2];
            kernel::spherical_transport(radius, &s, &t, &logged, SPHERE_NORM_TOL, &mut transported)
                .unwrap();
            assert!((norm(&transported) / radius - 0.4).abs() < 1e-14);
        }
        for c in [1e-300_f64, 1e300_f64] {
            let origin = [0.0; 2];
            let y = [0.2 / c.sqrt(), 0.0];
            let mut logged = [0.0; 2];
            kernel::hyperbolic_log(c, &origin, &y, &mut logged).unwrap();
            assert!((logged[0] * c.sqrt() - 0.2_f64.atanh()).abs() < 1e-14);
            let mut recovered = [0.0; 2];
            kernel::hyperbolic_exp(c, &origin, &logged, &mut recovered).unwrap();
            assert!((recovered[0] * c.sqrt() - 0.2).abs() < 1e-14);
        }
    }

    #[test]
    fn public_kernels_reject_invalid_scalars_and_mismatched_shapes() {
        let mut out = [0.0; 2];
        for c in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(kernel::ball_conformal(c, &[0.0; 2]).is_err());
            assert!(kernel::hyperbolic_exp(c, &[0.0; 2], &[0.0; 2], &mut out).is_err());
        }
        for tol in [-1.0, 1.0, f64::NAN, f64::INFINITY] {
            assert!(kernel::sphere_check(1.0, &[1.0, 0.0], tol).is_err());
        }
        for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(kernel::sphere_check(radius, &[1.0, 0.0], SPHERE_NORM_TOL).is_err());
        }
        assert!(kernel::hyperbolic_log(1.0, &[0.0], &[0.0], &mut out).is_err());
        assert!(kernel::hyperbolic_transport(1.0, &[0.0], &[0.0], &[0.0], &mut out).is_err());
        assert!(kernel::sphere_project(1.0, &[1.0], &[0.0], &mut out).is_err());
        assert!(kernel::spherical_exp(1.0, &[1.0], &[0.0], SPHERE_NORM_TOL, &mut out).is_err());
        assert!(kernel::spherical_transport(
            1.0,
            &[1.0],
            &[1.0],
            &[0.0],
            SPHERE_NORM_TOL,
            &mut out
        )
        .is_err());
    }

    #[test]
    fn extreme_sphere_radius_stays_scale_free() {
        let radius = 1e200_f64;
        let angle = 1.2_f64;
        let s = [radius, 0.0];
        let t = [radius * angle.cos(), radius * angle.sin()];
        assert!(
            (kernel::sphere_angle(radius, &s, &t, SPHERE_NORM_TOL).unwrap() - angle).abs() < 1e-12
        );

        let mut projected = [0.0; 2];
        kernel::sphere_project(radius, &s, &[1.0, 2.0], &mut projected).unwrap();
        assert_eq!(projected, [0.0, 2.0]);
        assert_eq!(
            kernel::sphere_angle(radius, &s, &[radius, 0.0, 0.0], SPHERE_NORM_TOL),
            Err(Reject::DomainViolation)
        );
        assert_eq!(
            kernel::sphere_check(radius, &s, 1.0),
            Err(Reject::DomainViolation)
        );
    }

    #[test]
    fn large_coordinate_norms_and_layout_additions_fail_closed() {
        let m = manifold(Layout::STORE_64);
        let base = {
            let mut coords = vec![0.0; m.dim()];
            coords[m.layout().s_range().start] = m.params().radius;
            m.point(&coords).unwrap()
        };
        let mut ambient = vec![0.0; m.dim()];
        ambient[0] = f64::MAX / 4.0;
        let tangent = m.project_tangent(&base, &ambient).unwrap();
        assert!(m.tangent_norm(&base, &tangent).unwrap().is_finite());

        assert_eq!(Layout::new(usize::MAX, 1, 1), Err(Reject::DomainViolation));
        assert_eq!(
            Layout::new(usize::MAX - 2, 1, 1),
            Err(Reject::DomainViolation)
        );
        let mut out = [0.0; 1];
        assert_eq!(
            kernel::mobius_add(1.0, &[0.0], &[0.0, 0.0], &mut out),
            Err(Reject::DomainViolation)
        );
    }

    #[test]
    fn layouts_store_128_and_256_coordinates() {
        assert_eq!(Layout::STORE_128_DEEP.store_dim(), 128);
        assert_eq!(Layout::STORE_128_DEEP.intrinsic_dim(), 127);
        assert_eq!(
            (
                Layout::STORE_128_DEEP.h_range().len(),
                Layout::STORE_128_DEEP.e_range().len(),
                Layout::STORE_128_DEEP.s_range().len()
            ),
            (80, 24, 24)
        );
        assert_eq!(Layout::STORE_256.store_dim(), 256);
        assert_eq!(Layout::STORE_256.intrinsic_dim(), 255);
        assert_eq!(
            (
                Layout::STORE_256.h_range().len(),
                Layout::STORE_256.e_range().len(),
                Layout::STORE_256.s_range().len()
            ),
            (160, 48, 48)
        );
        assert_eq!(Layout::new(0, 1, 1), Err(Reject::DomainViolation));
        assert_eq!(Layout::new(1, 1, 0), Err(Reject::DomainViolation));
        assert_eq!(MixedCurvatureCoord::LAYOUT.store_dim(), 16);
    }

    #[test]
    fn geometry_is_object_safe_and_shareable() {
        fn takes(_: std::sync::Arc<dyn ProductGeometry>) {}
        takes(std::sync::Arc::new(manifold(Layout::STORE_128_DEEP)));
    }

    #[test]
    fn exp_of_log_recovers_point() {
        for layout in [Layout::STORE_128_DEEP, Layout::STORE_256] {
            let m = manifold(layout);
            let mut rng = Rng(250925);
            for _ in 0..200 {
                let base = random_point(&m, &mut rng, 0.9);
                let p = random_point(&m, &mut rng, 0.9);
                let v = m.log(&base, &p).unwrap();
                let q = m.exp(&base, &v).unwrap();
                let err = max_abs_diff(q.coords(), p.coords());
                assert!(err < TOL, "{layout:?}: exp(log) error {err:e}");
            }
        }
    }

    #[test]
    fn log_of_exp_recovers_tangent_inside_injectivity_radius() {
        let m = manifold(Layout::STORE_128_DEEP);
        let mut rng = Rng(7);
        for _ in 0..100 {
            let base = random_point(&m, &mut rng, 0.8);
            let raw: Vec<f64> = (0..128).map(|_| rng.normal() * 0.3).collect();
            let v = m.project_tangent(&base, &raw).unwrap();
            let p = m.exp(&base, &v).unwrap();
            let back = m.log(&base, &p).unwrap();
            let err = max_abs_diff(back.coords(), v.coords());
            assert!(err < 1e-9, "log(exp) error {err:e}");
        }
    }

    #[test]
    fn log_norm_equals_distance() {
        let m = manifold(Layout::STORE_256);
        let mut rng = Rng(11);
        for _ in 0..100 {
            let a = random_point(&m, &mut rng, 0.9);
            let b = random_point(&m, &mut rng, 0.9);
            let d = m.distance(&a, &b).unwrap();
            let n = m.tangent_norm(&a, &m.log(&a, &b).unwrap()).unwrap();
            assert!((d - n).abs() <= TOL * d.max(1.0), "|log| {n} vs d {d}");
        }
    }

    #[test]
    fn distance_is_symmetric_and_satisfies_triangle_inequality() {
        for layout in [Layout::STORE_128_DEEP, Layout::STORE_256] {
            let m = manifold(layout);
            let mut rng = Rng(42);
            for _ in 0..300 {
                let a = random_point(&m, &mut rng, 0.95);
                let b = random_point(&m, &mut rng, 0.95);
                let c = random_point(&m, &mut rng, 0.95);
                let ab = m.distance(&a, &b).unwrap();
                let ba = m.distance(&b, &a).unwrap();
                let bc = m.distance(&b, &c).unwrap();
                let ac = m.distance(&a, &c).unwrap();
                assert!((ab - ba).abs() <= 1e-12 * ab.max(1.0));
                assert!(ac <= ab + bc + 1e-10, "triangle: {ac} > {ab} + {bc}");
                assert_eq!(m.distance(&a, &a).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn arsinh_distance_matches_arcosh_and_mobius_forms() {
        let c: f64 = 0.7;
        let mut rng = Rng(3);
        for _ in 0..200 {
            let x: Vec<f64> = rng
                .direction(80)
                .iter()
                .map(|v| v * 0.9 * rng.uniform() / c.sqrt())
                .collect();
            let y: Vec<f64> = rng
                .direction(80)
                .iter()
                .map(|v| v * 0.9 * rng.uniform() / c.sqrt())
                .collect();
            let d = kernel::hyperbolic_distance(c, &x, &y).unwrap();
            let diff2: f64 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum();
            let arcosh = (1.0
                + 2.0 * c * diff2 / ((1.0 - c * dot(&x, &x)) * (1.0 - c * dot(&y, &y))))
            .acosh()
                / c.sqrt();
            let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
            let mut u = vec![0.0; 80];
            kernel::mobius_add(c, &neg_x, &y, &mut u).unwrap();
            let mobius = 2.0 / c.sqrt() * (c.sqrt() * norm(&u)).atanh();
            assert!(
                (d - arcosh).abs() < 1e-9 * d.max(1.0),
                "{d} vs arcosh {arcosh}"
            );
            assert!(
                (d - mobius).abs() < 1e-12 * d.max(1.0),
                "{d} vs mobius {mobius}"
            );
        }
    }

    #[test]
    fn distance_composes_alpha_weighted_factors() {
        let layout = Layout::STORE_128_DEEP;
        let m = manifold(layout);
        let p = m.params();
        let mut rng = Rng(61);
        let a = random_point(&m, &mut rng, 0.9);
        let b = random_point(&m, &mut rng, 0.9);
        let (x, y) = (a.coords(), b.coords());
        let dh =
            kernel::hyperbolic_distance(p.curvature, &x[layout.h_range()], &y[layout.h_range()])
                .unwrap();
        let de2: f64 = x[layout.e_range()]
            .iter()
            .zip(&y[layout.e_range()])
            .map(|(u, v)| (u - v) * (u - v))
            .sum();
        let ds = kernel::spherical_distance(
            p.radius,
            &x[layout.s_range()],
            &y[layout.s_range()],
            SPHERE_NORM_TOL,
        )
        .unwrap();
        let expected = (p.alpha_h * dh * dh + p.alpha_e * de2 + p.alpha_s * ds * ds).sqrt();
        let d = m.distance(&a, &b).unwrap();
        assert!((d - expected).abs() < 1e-12 * expected);

        // Scaling every alpha by 4 doubles the distance; scaling one factor
        // changes it by exactly that factor's share.
        let scaled = GeometryParams {
            alpha_h: 4.0 * p.alpha_h,
            alpha_e: 4.0 * p.alpha_e,
            alpha_s: 4.0 * p.alpha_s,
            ..p
        };
        let m4 = ProductManifold::new(layout, scaled, epochs_for(layout, scaled)).unwrap();
        let (a4, b4) = (m4.point(x).unwrap(), m4.point(y).unwrap());
        assert!((m4.distance(&a4, &b4).unwrap() - 2.0 * d).abs() < 1e-12 * d);
        let only_s = GeometryParams {
            alpha_s: 4.0 * p.alpha_s,
            ..p
        };
        let ms = ProductManifold::new(layout, only_s, epochs_for(layout, only_s)).unwrap();
        let got = ms
            .distance(&ms.point(x).unwrap(), &ms.point(y).unwrap())
            .unwrap();
        let want = (d * d + 3.0 * p.alpha_s * ds * ds).sqrt();
        assert!((got - want).abs() < 1e-12 * want);
    }

    #[test]
    fn sphere_angle_matches_arccos_form() {
        let r = 2.5;
        let mut rng = Rng(5);
        for _ in 0..200 {
            let s: Vec<f64> = rng.direction(24).iter().map(|v| v * r).collect();
            let t: Vec<f64> = rng.direction(24).iter().map(|v| v * r).collect();
            let d = kernel::spherical_distance(r, &s, &t, SPHERE_NORM_TOL).unwrap();
            let arccos = r * (dot(&s, &t) / (r * r)).clamp(-1.0, 1.0).acos();
            assert!((d - arccos).abs() < 1e-12, "{d} vs {arccos}");
        }
    }

    #[test]
    fn hyperbolic_domain_violation_fails_closed() {
        let m = manifold(Layout::STORE_128_DEEP);
        let l = m.layout();
        let mut rng = Rng(9);
        let good = random_point(&m, &mut rng, 0.5);
        let c = m.params().curvature;
        // Exactly on the boundary: c ||x||^2 == 1.
        let mut on_boundary = good.coords().to_vec();
        on_boundary[l.h_range()].fill(0.0);
        on_boundary[0] = 1.0 / c.sqrt();
        assert_eq!(m.point(&on_boundary), Err(Reject::DomainViolation));
        // Inside by 1e-6 in conformal factor: valid. Inside by 1e-14: boundary.
        let mut near = on_boundary.clone();
        near[0] = ((1.0 - 1e-6) / c).sqrt();
        assert!(m.point(&near).is_ok());
        near[0] = ((1.0 - 1e-14) / c).sqrt();
        assert_eq!(m.point(&near), Err(Reject::DomainViolation));
        let mut outside = on_boundary.clone();
        outside[0] = 2.0 / c.sqrt();
        assert_eq!(m.point(&outside), Err(Reject::DomainViolation));
        assert_eq!(
            kernel::hyperbolic_distance(c, &outside[l.h_range()], &good.coords()[l.h_range()]),
            Err(Reject::DomainViolation)
        );
        // exp that saturates tanh onto the boundary is rejected, not clamped.
        let mut huge = vec![0.0; l.store_dim()];
        huge[0] = 1e6;
        let v = m.project_tangent(&good, &huge).unwrap();
        assert_eq!(m.exp(&good, &v), Err(Reject::DomainViolation));
    }

    #[test]
    fn non_finite_and_malformed_input_fails_closed() {
        let m = manifold(Layout::STORE_128_DEEP);
        let mut rng = Rng(10);
        let good = random_point(&m, &mut rng, 0.5);
        let mut nan = good.coords().to_vec();
        nan[90] = f64::NAN;
        assert_eq!(m.point(&nan), Err(Reject::NonFiniteState));
        assert_eq!(m.point(&good.coords()[..127]), Err(Reject::DomainViolation));
        let mut off_sphere = good.coords().to_vec();
        off_sphere[127] += 1e-3;
        assert_eq!(m.point(&off_sphere), Err(Reject::DomainViolation));
        let mut amb = vec![0.0; 128];
        amb[3] = f64::INFINITY;
        assert_eq!(m.project_tangent(&good, &amb), Err(Reject::NonFiniteState));
        assert_eq!(
            m.project_tangent(&good, &[0.0; 12]),
            Err(Reject::DomainViolation)
        );
        let bad = GeometryParams {
            curvature: 0.0,
            ..params()
        };
        assert_eq!(
            ProductManifold::new(
                Layout::STORE_128_DEEP,
                bad,
                epochs_for(Layout::STORE_128_DEEP, bad)
            )
            .err(),
            Some(Reject::DomainViolation)
        );
    }

    #[test]
    fn sphere_antipode_is_cut_locus() {
        let m = manifold(Layout::STORE_128_DEEP);
        let l = m.layout();
        let mut rng = Rng(12);
        let base = random_point(&m, &mut rng, 0.5);
        let mut anti = base.coords().to_vec();
        for i in l.s_range() {
            anti[i] = -anti[i];
        }
        let anti = m.point(&anti).unwrap();
        assert_eq!(m.log(&base, &anti), Err(Reject::CutLocus));
        // The distance itself is well defined at the antipode: pi R.
        let d_s = kernel::spherical_distance(
            m.params().radius,
            &base.coords()[l.s_range()],
            &anti.coords()[l.s_range()],
            SPHERE_NORM_TOL,
        )
        .unwrap();
        assert!((d_s - PI * m.params().radius).abs() < 1e-12);
        // Transport across the antipode is refused.
        let v = m.project_tangent(&base, &vec![0.1; 128]).unwrap();
        assert_eq!(
            m.transport(&[base.clone(), anti], &v),
            Err(Reject::CutLocus)
        );
        // exp beyond the injectivity radius (||v_S|| >= pi R) is refused.
        let dir = m.log(&base, &random_point(&m, &mut rng, 0.5)).unwrap();
        let s_norm = norm(&dir.coords()[l.s_range()]);
        let mut long = vec![0.0; 128];
        for i in l.s_range() {
            long[i] = dir.coords()[i] / s_norm * PI * m.params().radius * 1.01;
        }
        let long = m.project_tangent(&base, &long).unwrap();
        assert_eq!(m.exp(&base, &long), Err(Reject::CutLocus));
    }

    #[test]
    fn tangent_bound_to_other_fiber_or_epoch_is_rejected() {
        let m = manifold(Layout::STORE_128_DEEP);
        let mut rng = Rng(13);
        let a = random_point(&m, &mut rng, 0.5);
        let b = random_point(&m, &mut rng, 0.5);
        let v_at_a = m.log(&a, &b).unwrap();
        assert_eq!(m.exp(&b, &v_at_a), Err(Reject::FiberMismatch));
        assert_eq!(m.transport(&[], &v_at_a), Err(Reject::FiberMismatch));
        assert_eq!(
            m.transport(&[b.clone(), a.clone()], &v_at_a),
            Err(Reject::FiberMismatch)
        );

        // Same parameters, different snapshot version.
        let p = params();
        let mut other_epochs = epochs_for(Layout::STORE_128_DEEP, p);
        other_epochs.version = Version(2);
        let m2 = ProductManifold::new(Layout::STORE_128_DEEP, p, other_epochs).unwrap();
        assert_eq!(m2.distance(&a, &b), Err(Reject::EpochMismatch));
        assert_eq!(m2.exp(&a, &v_at_a), Err(Reject::EpochMismatch));

        // epochs.geometry must bind the parameters.
        let wrong = GeometryParams {
            curvature: 1.1,
            ..p
        };
        assert_eq!(
            ProductManifold::new(
                Layout::STORE_128_DEEP,
                wrong,
                epochs_for(Layout::STORE_128_DEEP, p)
            )
            .err(),
            Some(Reject::EpochMismatch)
        );
        // Layout mismatch.
        let m256 = manifold(Layout::STORE_256);
        assert_eq!(m256.distance(&a, &b), Err(Reject::DomainViolation));
    }

    #[test]
    fn project_tangent_128_and_256() {
        for layout in [Layout::STORE_128_DEEP, Layout::STORE_256] {
            let m = manifold(layout);
            let r2 = m.params().radius * m.params().radius;
            let mut rng = Rng(21);
            for _ in 0..50 {
                let base = random_point(&m, &mut rng, 0.9);
                let z: Vec<f64> = (0..layout.store_dim())
                    .map(|_| rng.normal() * 2.0)
                    .collect();
                let t = m.project_tangent(&base, &z).unwrap();
                let v = t.coords();
                assert_eq!(v.len(), layout.store_dim());
                // H and E blocks unchanged.
                assert_eq!(&v[layout.h_range()], &z[layout.h_range()]);
                assert_eq!(&v[layout.e_range()], &z[layout.e_range()]);
                // S block orthogonal to the base point and equal to z - s<s,z>/R^2.
                let s = &base.coords()[layout.s_range()];
                let vs = &v[layout.s_range()];
                assert!(dot(s, vs).abs() < 1e-12 * r2.max(norm(&z)));
                let k = dot(s, &z[layout.s_range()]) / r2;
                for ((vi, zi), si) in vs.iter().zip(&z[layout.s_range()]).zip(s) {
                    assert!((vi - (zi - k * si)).abs() < 1e-12);
                }
                // Idempotent.
                let t2 = m.project_tangent(&base, v).unwrap();
                assert!(max_abs_diff(t2.coords(), v) < 1e-12);
                assert_eq!(t.fiber().base, base.digest());
            }
        }
    }

    #[test]
    fn transport_is_isometric_and_maps_log_to_negated_log() {
        for layout in [Layout::STORE_128_DEEP, Layout::STORE_256] {
            let m = manifold(layout);
            let mut rng = Rng(31);
            for _ in 0..100 {
                let x = random_point(&m, &mut rng, 0.9);
                let y = random_point(&m, &mut rng, 0.9);
                // Short vectors, so that exp at y stays inside the ball and the
                // sphere injectivity radius.
                let z: Vec<f64> = (0..layout.store_dim())
                    .map(|_| 0.02 * rng.normal())
                    .collect();
                let v = m.project_tangent(&x, &z).unwrap();
                let pv = m.transport(&[x.clone(), y.clone()], &v).unwrap();
                assert_eq!(pv.fiber().base, y.digest());
                let (n0, n1) = (
                    m.tangent_norm(&x, &v).unwrap(),
                    m.tangent_norm(&y, &pv).unwrap(),
                );
                assert!(
                    (n0 - n1).abs() < 1e-10 * n0.max(1.0),
                    "isometry {n0} vs {n1}"
                );
                let s = &y.coords()[layout.s_range()];
                assert!(dot(s, &pv.coords()[layout.s_range()]).abs() < 1e-10);
                // The result is a valid tangent at y: exp accepts it.
                m.exp(&y, &pv).unwrap();

                let log_xy = m.log(&x, &y).unwrap();
                let moved = m.transport(&[x.clone(), y.clone()], &log_xy).unwrap();
                let log_yx = m.log(&y, &x).unwrap();
                let neg: Vec<f64> = log_yx.coords().iter().map(|v| -v).collect();
                assert!(max_abs_diff(moved.coords(), &neg) < 1e-9);
            }
        }
    }

    #[test]
    fn transport_along_multi_point_path_and_back_preserves_norm() {
        let m = manifold(Layout::STORE_128_DEEP);
        let mut rng = Rng(41);
        let a = random_point(&m, &mut rng, 0.7);
        let b = random_point(&m, &mut rng, 0.7);
        let c = random_point(&m, &mut rng, 0.7);
        let v = m
            .project_tangent(&a, &(0..128).map(|_| rng.normal()).collect::<Vec<_>>())
            .unwrap();
        let moved = m.transport(&[a.clone(), b.clone(), c.clone()], &v).unwrap();
        assert_eq!(moved.fiber().base, c.digest());
        assert_ne!(moved.fiber().path, v.fiber().path);
        let n0 = m.tangent_norm(&a, &v).unwrap();
        assert!((n0 - m.tangent_norm(&c, &moved).unwrap()).abs() < 1e-10 * n0.max(1.0));
        // Round trip along a single geodesic returns the same vector.
        let there = m.transport(&[a.clone(), b.clone()], &v).unwrap();
        let back = m.transport(&[b, a.clone()], &there).unwrap();
        assert!(max_abs_diff(back.coords(), v.coords()) < 1e-10);
        // A one-point path is the identity.
        assert_eq!(m.transport(std::slice::from_ref(&a), &v).unwrap(), v);
    }

    #[test]
    fn gyration_closed_form_matches_mobius_definition() {
        // gyr[a,b]w = -(a (+) b) (+) (a (+) (b (+) w)) for w inside the ball.
        let c: f64 = 0.7;
        let mut rng = Rng(51);
        for _ in 0..100 {
            let pt = |rng: &mut Rng| -> Vec<f64> {
                rng.direction(6)
                    .iter()
                    .map(|v| v * 0.8 * rng.uniform() / c.sqrt())
                    .collect()
            };
            let (a, b, w) = (pt(&mut rng), pt(&mut rng), pt(&mut rng));
            let mut closed = vec![0.0; 6];
            kernel::gyration(c, &a, &b, &w, &mut closed).unwrap();
            let mut bw = vec![0.0; 6];
            kernel::mobius_add(c, &b, &w, &mut bw).unwrap();
            let mut abw = vec![0.0; 6];
            kernel::mobius_add(c, &a, &bw, &mut abw).unwrap();
            let mut ab = vec![0.0; 6];
            kernel::mobius_add(c, &a, &b, &mut ab).unwrap();
            let neg_ab: Vec<f64> = ab.iter().map(|v| -v).collect();
            let mut def = vec![0.0; 6];
            kernel::mobius_add(c, &neg_ab, &abw, &mut def).unwrap();
            assert!(max_abs_diff(&closed, &def) < 1e-12);
        }
    }

    const UNIT_ALPHAS: [f32; 3] = [1.0; 3];

    fn coord_manifold(p: GeometryParams) -> ProductManifold {
        let layout = MixedCurvatureCoord::LAYOUT;
        ProductManifold::new(layout, p, epochs_for(layout, p)).unwrap()
    }

    #[test]
    fn coord_distance_is_zero_at_identity_and_grows() {
        let p1 = MixedCurvatureCoord::origin();
        let p2 = MixedCurvatureCoord::origin();
        let d = |a: &MixedCurvatureCoord, b| {
            a.product_distance_with_params(b, UNIT_ALPHAS, 1.0, 1.0)
                .unwrap()
        };
        assert_eq!(d(&p1, &p2), 0.0);

        let mut p3 = MixedCurvatureCoord::origin();
        p3.hyperbolic[0] = 0.5;
        p3.euclidean[0] = 1.0;
        assert!(d(&p1, &p3) > 1.0);
    }

    /// One geometry, two entry points: the coordinate distance under explicit
    /// parameters equals `ProductManifold::distance` on the converted points,
    /// for non-unit curvature, radius and weights.
    #[test]
    fn coord_distance_equals_product_manifold_distance() {
        // All exactly representable in f32, so both sides see the same numbers.
        let cases = [
            GeometryParams::UNIT,
            GeometryParams {
                curvature: 0.5,
                radius: 2.5,
                alpha_h: 0.25,
                alpha_e: 2.0,
                alpha_s: 4.0,
            },
            GeometryParams {
                curvature: 2.0,
                radius: 0.125,
                alpha_h: 8.0,
                alpha_e: 0.5,
                alpha_s: 0.0625,
            },
        ];
        let mut rng = Rng(0x5EED_C0DE);
        for p in cases {
            let m = coord_manifold(p);
            let c = p.curvature as f32;
            for _ in 0..50 {
                let mut gen = || {
                    let dir = rng.direction(4);
                    let reach = 0.6 * rng.uniform() / p.curvature.sqrt();
                    let h: [f32; 4] = std::array::from_fn(|i| (dir[i] * reach) as f32);
                    let s: [f32; 4] = std::array::from_fn(|_| rng.normal() as f32);
                    let e: [f32; 8] = std::array::from_fn(|_| rng.normal() as f32);
                    MixedCurvatureCoord::with_curvature(h, s, e, c).unwrap()
                };
                let (a, b) = (gen(), gen());
                let alphas = [p.alpha_h as f32, p.alpha_e as f32, p.alpha_s as f32];
                let got = a
                    .product_distance_with_params(&b, alphas, c, p.radius as f32)
                    .unwrap();
                let want = m
                    .distance(&a.to_point(&m).unwrap(), &b.to_point(&m).unwrap())
                    .unwrap();
                assert!(
                    (f64::from(got) - want).abs() <= 1e-5 * want.max(1.0),
                    "{p:?}: coord {got} vs manifold {want}"
                );
            }
        }
    }

    /// Each parameter moves the distance the way the metric says it must.
    #[test]
    fn coord_distance_responds_to_every_parameter() {
        let a = MixedCurvatureCoord::origin();
        let only = |h: f32, s: [f32; 4], e: f32| {
            let mut c = MixedCurvatureCoord::origin();
            c.hyperbolic[0] = h;
            c.spherical = s;
            c.euclidean[0] = e;
            c
        };
        let north = [1.0, 0.0, 0.0, 0.0];
        let (bh, be, bs) = (
            only(0.3, north, 0.0),
            only(0.0, north, 2.0),
            only(0.0, [0.0, 1.0, 0.0, 0.0], 0.0),
        );
        let d = |b: &MixedCurvatureCoord, alphas, c, r| {
            f64::from(a.product_distance_with_params(b, alphas, c, r).unwrap())
        };
        let base = d(&bh, UNIT_ALPHAS, 1.0, 1.0);
        // alpha_h = 4 doubles a purely hyperbolic distance and leaves the others alone.
        assert!((d(&bh, [4.0, 1.0, 1.0], 1.0, 1.0) - 2.0 * base).abs() < 1e-6);
        assert!((d(&be, [4.0, 1.0, 1.0], 1.0, 1.0) - 2.0).abs() < 1e-6);
        assert!((d(&be, [1.0, 9.0, 1.0], 1.0, 1.0) - 6.0).abs() < 1e-6);
        assert!((d(&bs, [1.0, 1.0, 4.0], 1.0, 1.0) - PI).abs() < 1e-6);
        // Radius scales the sphere distance: a quarter turn on S_R is R pi / 2.
        assert!((d(&bs, UNIT_ALPHAS, 1.0, 3.0) - 1.5 * PI).abs() < 1e-6);
        // Curvature: d_H(0, x) = (2 / sqrt(c)) artanh(sqrt(c) |x|).
        let c = 4.0_f64;
        let want = 2.0 / c.sqrt() * (c.sqrt() * f64::from(0.3_f32)).atanh();
        assert!((d(&bh, UNIT_ALPHAS, 4.0, 1.0) - want).abs() < 1e-6);
        assert!((want - base).abs() > 0.05);
    }

    #[test]
    fn coord_rejects_invalid_state_and_parameters() {
        assert!(MixedCurvatureCoord::new([0.0; 4], [0.0; 4], [0.0; 8]).is_err());
        assert!(MixedCurvatureCoord::new([0.0; 4], [1.0, 0.0, 0.0, 0.0], [f32::NAN; 8]).is_err());
        assert!(MixedCurvatureCoord::new([0.0; 4], [1e30, 1e30, 0.0, 0.0], [0.0; 8]).is_err());
        let origin = MixedCurvatureCoord::origin();
        let north = [1.0, 0.0, 0.0, 0.0];
        // The boundary floor follows the curvature: |x| = 0.8 is inside the unit
        // ball and outside the ball of curvature 4 (radius 0.5); |x| = 1.2 is the
        // other way round for curvature 0.25 (radius 2).
        let inner = [0.8, 0.0, 0.0, 0.0];
        let outer = [1.2, 0.0, 0.0, 0.0];
        assert!(MixedCurvatureCoord::with_curvature(inner, north, [0.0; 8], 1.0).is_ok());
        assert!(matches!(
            MixedCurvatureCoord::with_curvature(inner, north, [0.0; 8], 4.0),
            Err(LodError::HyperbolicBoundaryViolation { .. })
        ));
        assert!(MixedCurvatureCoord::new(outer, north, [0.0; 8]).is_err());
        let wide = MixedCurvatureCoord::with_curvature(outer, north, [0.0; 8], 0.25).unwrap();
        assert!(wide
            .product_distance_with_params(&origin, UNIT_ALPHAS, 0.25, 1.0)
            .is_ok());
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(MixedCurvatureCoord::with_curvature(inner, north, [0.0; 8], bad).is_err());
            for (alphas, c, r) in [
                ([bad, 1.0, 1.0], 1.0, 1.0),
                ([1.0, bad, 1.0], 1.0, 1.0),
                ([1.0, 1.0, bad], 1.0, 1.0),
                (UNIT_ALPHAS, bad, 1.0),
                (UNIT_ALPHAS, 1.0, bad),
            ] {
                assert_eq!(
                    origin.product_distance_with_params(&origin, alphas, c, r),
                    Err(LodError::Geometry(Reject::DomainViolation))
                );
            }
        }
        // Public fields can bypass the constructors; distance and conversion
        // still fail closed, against the curvature in force.
        let mut bad = MixedCurvatureCoord::origin();
        bad.hyperbolic = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            bad.product_distance_with_params(&origin, UNIT_ALPHAS, 1.0, 1.0),
            Err(LodError::Geometry(Reject::DomainViolation))
        );
        assert_eq!(
            wide.product_distance_with_params(&origin, UNIT_ALPHAS, 1.0, 1.0),
            Err(LodError::Geometry(Reject::DomainViolation))
        );
        assert!(wide
            .to_point(&coord_manifold(GeometryParams::UNIT))
            .is_err());
        let mut off = MixedCurvatureCoord::origin();
        off.spherical = [2.0, 0.0, 0.0, 0.0];
        assert_eq!(
            off.product_distance_with_params(&origin, UNIT_ALPHAS, 1.0, 1.0),
            Err(LodError::Geometry(Reject::DomainViolation))
        );
        assert!(off.to_point(&coord_manifold(GeometryParams::UNIT)).is_err());
        // Only the 16-coordinate layout converts.
        let p = GeometryParams::UNIT;
        let layout = TopologyPreset::Compact64d.layout();
        let other = ProductManifold::new(layout, p, epochs_for(layout, p)).unwrap();
        assert!(origin.to_point(&other).is_err());
    }
}

#[cfg(test)]
mod fatigue_phase_tests {
    use super::{ClockPhase, FATIGUE_FRAC_PARTS, FATIGUE_FRAC_QUANTUM};
    #[test]
    fn clock_unwraps_and_crosses_fractional_tick_without_resetting_at_budget() {
        let before = ClockPhase::new(3, 10, Some(0.35)).unwrap();
        let after = ClockPhase::new(4, 10, Some(0.35)).unwrap();
        assert!(!before.fatigued);
        assert!(after.fatigued);
        for tick in [0, 4, 10, 20] {
            let phase = ClockPhase::new(tick, 10, Some(0.35)).unwrap();
            assert!((phase.theta_clock - std::f64::consts::TAU * tick as f64 / 10.0).abs() < 1e-14);
            assert!((phase.spherical.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-14);
            assert_eq!(phase.fatigued, tick >= 4);
        }
        assert!(!ClockPhase::new(20, 10, None).unwrap().fatigued);
    }
    #[test]
    fn invalid_clock_parameters_are_refused() {
        assert!(ClockPhase::new(0, 0, None).is_err());
        for f in [0.0, -0.1, 1.1, 1e-10, f64::NAN, f64::INFINITY] {
            assert!(ClockPhase::new(1, 10, Some(f)).is_err());
        }
        assert!(ClockPhase::new(10, 10, Some(1.0)).unwrap().fatigued);
    }
    #[test]
    fn onset_tick_rejects_invalid_inputs_fail_closed() {
        assert!(ClockPhase::onset_tick(0, 0.07).is_err());
        assert!(ClockPhase::onset_tick(100, 0.0).is_err());
        assert!(ClockPhase::onset_tick(100, -0.07).is_err());
        assert!(ClockPhase::onset_tick(100, 1.01).is_err());
        assert!(ClockPhase::onset_tick(100, f64::NAN).is_err());
        assert!(ClockPhase::onset_tick(100, f64::INFINITY).is_err());
    }
    #[test]
    fn onset_tick_is_exact_for_decimal_fractions() {
        // 0.07 * 100.0 == 7.000000000000001 in f64; a raw ceil() gives 8.
        assert_eq!(ClockPhase::onset_tick(100, 0.07).unwrap(), 7);
        // A real excess past the 9th decimal is not rounded away.
        assert_eq!(ClockPhase::onset_tick(100, 0.0700000001).unwrap(), 8);
        for k in 1..=99_u64 {
            assert_eq!(ClockPhase::onset_tick(100, k as f64 / 100.0).unwrap(), k, "k = {k}");
        }
        assert_eq!(ClockPhase::onset_tick(10, 0.35).unwrap(), 4);
        assert_eq!(ClockPhase::onset_tick(7, 0.42).unwrap(), 3);
        assert_eq!(ClockPhase::onset_tick(u32::MAX, 1.0).unwrap(), u64::from(u32::MAX));
        assert_eq!(ClockPhase::onset_tick(1, FATIGUE_FRAC_QUANTUM).unwrap(), 1);
        assert!(!ClockPhase::new(6, 100, Some(0.07)).unwrap().fatigued);
        assert!(ClockPhase::new(7, 100, Some(0.07)).unwrap().fatigued);
    }
    #[test]
    fn onset_tick_is_exact_for_decimal_fractions_at_large_budgets() {
        // A relative rounding band widens with the budget; past 600_000 it
        // swallowed a whole real excess of 1e-9 * b and fired one tick early.
        assert_eq!(ClockPhase::onset_tick(600_001, 0.999400001).unwrap(), 599_642);
        assert_eq!(ClockPhase::onset_tick(1_437_833, 0.515309497).unwrap(), 740_930);
    }
    #[test]
    fn onset_tick_keeps_tiny_real_excess() {
        // Off the 1e-9 grid: an absolute floor on the band would read these as 7.
        assert_eq!(ClockPhase::onset_tick(100, 0.0700000001).unwrap(), 8);
        assert_eq!(ClockPhase::onset_tick(100, 0.070000000000005).unwrap(), 8);
        assert_eq!(ClockPhase::onset_tick(100, 0.07 + 5e-16).unwrap(), 8);
    }
    #[test]
    fn onset_tick_matches_exact_integers_across_budgets() {
        // Budgets up to the planner's MAX_TRIAD_BUDGET (1 << 24) and past it.
        let budgets = [1_u32, 7, 599_999, 600_001, 1_437_833, 1 << 24, u32::MAX];
        let mut q = 1_u64;
        while q <= FATIGUE_FRAC_PARTS {
            let f = q as f64 / FATIGUE_FRAC_PARTS as f64;
            for b in budgets {
                let exact = (q * u64::from(b)).div_ceil(FATIGUE_FRAC_PARTS);
                assert_eq!(ClockPhase::onset_tick(b, f).unwrap(), exact, "q = {q}, b = {b}");
            }
            q += 999_983;
        }
        // Off the grid: k / b * b is k at large budgets too.
        for b in [7_u32, 999_983, (1 << 24) - 3, u32::MAX] {
            for k in [u64::from(b) / 3, 2 * u64::from(b) / 7, u64::from(b) - 1] {
                let f = k as f64 / f64::from(b);
                assert_eq!(ClockPhase::onset_tick(b, f).unwrap(), k, "{k}/{b}");
            }
        }
    }
    #[test]
    fn onset_tick_is_exact_for_non_decimal_fractions() {
        // Off the 1e-9 grid, so these take the machine-precision path.
        assert_eq!(ClockPhase::onset_tick(3, 2.0 / 3.0).unwrap(), 2);
        assert_eq!(ClockPhase::onset_tick(6, 1.0 / 6.0).unwrap(), 1);
        assert_eq!(ClockPhase::onset_tick(3, 1.0 / 3.0).unwrap(), 1);
        // k / b * b is exactly k, so the onset is k; any (k + 0.5) / b lies
        // strictly between k and k + 1, so the onset is k + 1.
        for b in [3_u32, 6, 7, 10, 12, 100] {
            for k in 1..=u64::from(b) {
                let f = k as f64 / f64::from(b);
                assert_eq!(ClockPhase::onset_tick(b, f).unwrap(), k, "{k}/{b}");
                if k < u64::from(b) {
                    let half = (k as f64 + 0.5) / f64::from(b);
                    assert_eq!(ClockPhase::onset_tick(b, half).unwrap(), k + 1, "{k}.5/{b}");
                }
            }
        }
    }
}
