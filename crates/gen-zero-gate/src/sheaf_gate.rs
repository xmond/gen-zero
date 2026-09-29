//! gen-zero-gate Spec 25 geometric + sheaf-cohomology deterministic fast gate.
//!
//! A single operator-based quadratic relaxation loop supports explicit weighted
//! coboundaries and affine dynamics windows with soft observations. Windows use
//! O(T*d^2 + P*d) work per evaluation and never form a global dense matrix.
//!
//! Every step checks finite diagnostics, non-increasing energy, the exact
//! quadratic Taylor identity, fixed coordinates, and manifold guards.
//! `certify_and_relax` returns terminal diagnostics including non-convergence;
//! only `Converged` satisfies the requested residual tolerance. `certify` retains
//! the strict accept/reject interface for explicit matrix callers.

use gen_zero_core::dot_product_f32;
use gen_zero_lod::manifold::MixedCurvatureCoord;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Local result alias for this module. Distinct from any crate-root `Result`.
pub type Result<T> = core::result::Result<T, Reject>;

/// Which check produced [`Reject::EnergyRose`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyRoseKind {
    /// Requested (or auto-selected) `eta` violates the theoretical Lyapunov
    /// safety bound `eta < 2/lambda_max(D^T*W*D)` before any step was taken.
    StepBoundViolated,
    /// The bound held but the empirical post-step energy increased
    /// (guards against ill-conditioning / floating point breakdown of the
    /// theoretical bound).
    EmpiricalIncrease,
}

/// Geometry rejection codes, including invalid caller budgets. Every failure
/// returns an explicit rejection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Error)]
pub enum Reject {
    #[error(
        "energy rose ({kind:?}): E(s)={e_before:.6e} -> E(s')={e_after:.6e}, eta={eta:.6e}, safe_bound={safe_eta_bound:.6e}"
    )]
    EnergyRose {
        kind: EnergyRoseKind,
        e_before: f64,
        e_after: f64,
        eta: f64,
        safe_eta_bound: f64,
    },

    #[error(
        "energy is numerically uncertain: direct-path delta={direct_delta:.6e}, Taylor-identity delta={predicted_delta:.6e}, |diff|={diff:.3e} > tol={tol:.3e}"
    )]
    UncertainEnergy {
        direct_delta: f64,
        predicted_delta: f64,
        diff: f64,
        tol: f64,
    },

    #[error(
        "gradient stalled: ||grad||={grad_norm:.3e} < grad_tol={grad_tol:.3e} while residual={residual:.3e} > residual_tol={residual_tol:.3e} (stationary or numerically unresolved residual above target)"
    )]
    Stalled {
        grad_norm: f64,
        grad_tol: f64,
        residual: f64,
        residual_tol: f64,
    },

    #[error("residual {residual:.6e} exceeds tolerance {residual_tol:.6e} after exhausting the step budget")]
    ResidualExceeded { residual: f64, residual_tol: f64 },

    #[error(
        "pinned node {index} moved from anchor {anchor:.6e} to {actual:.6e} (|delta|={delta:.3e} > pin_tol={pin_tol:.3e})"
    )]
    PinnedMoved {
        index: usize,
        anchor: f64,
        actual: f64,
        delta: f64,
        pin_tol: f64,
    },

    #[error("state has a non-finite value at index {index}: {value}")]
    NonFiniteState { index: usize, value: f64 },

    #[error("domain violation at node block starting {offset}: {detail}")]
    DomainViolation { offset: usize, detail: String },

    #[error("cut locus singularity at node block starting {offset}: {detail}")]
    CutLocus { offset: usize, detail: String },

    #[error("fiber/tangent space mismatch: problem expects dimension {expected}, candidate has {actual}")]
    FiberMismatch { expected: usize, actual: usize },

    #[error(
        "epoch mismatch: problem snapshot epoch={problem_epoch}, candidate epoch={candidate_epoch}"
    )]
    EpochMismatch {
        problem_epoch: u64,
        candidate_epoch: u64,
    },
    #[error("invalid certification budget: {field}={value} is outside its permitted finite range")]
    InvalidBudget { field: String, value: f64 },
}

/// A node block whose 16 contiguous state components are interpreted as a
/// [`MixedCurvatureCoord`] (H^4 x S^4 x R^8), guarding [`Reject::DomainViolation`]
/// (hyperbolic boundary floor, mirroring `gen_zero_lod::manifold` exactly) and
/// [`Reject::CutLocus`] (spherical antipodal singularity relative to the
/// manifold's north-pole convention).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManifoldGuard {
    /// Index of the first of 16 contiguous components in the state vector.
    pub offset: usize,
}

/// Distance (in normalized dot-product terms) from the spherical block to its
/// antipode at which the geodesic cut locus is declared. `acos` is finite
/// everywhere on `[-1, 1]`, but the geodesic stops being unique at (and loses
/// differentiability approaching) the antipode of the reference pole.
const CUT_LOCUS_EPS: f32 = 1e-6;
const SPHERE_NORM_TOL: f64 = 1e-5;

/// Euclidean norm without intermediate squaring underflow or overflow.
fn stable_norm(v: &DVector<f64>) -> f64 {
    v.iter().fold(0.0_f64, |n, &x| n.hypot(x))
}

impl ManifoldGuard {
    fn check(&self, s: &[f64]) -> Result<()> {
        let end = self
            .offset
            .checked_add(16)
            .ok_or_else(|| Reject::DomainViolation {
                offset: self.offset,
                detail: "manifold block offset overflow".into(),
            })?;
        if end > s.len() {
            return Err(Reject::FiberMismatch {
                expected: end,
                actual: s.len(),
            });
        }
        let sphere_norm = s[self.offset + 4..self.offset + 8]
            .iter()
            .fold(0.0_f64, |n, &x| n.hypot(x));
        if !sphere_norm.is_finite() || (sphere_norm - 1.0).abs() > SPHERE_NORM_TOL {
            return Err(Reject::DomainViolation {
                offset: self.offset,
                detail: "spherical block must have unit norm".into(),
            });
        }
        let mut hyperbolic = [0.0f32; 4];
        let mut spherical = [0.0f32; 4];
        let mut euclidean = [0.0f32; 8];
        for i in 0..4 {
            hyperbolic[i] = s[self.offset + i] as f32;
            spherical[i] = s[self.offset + 4 + i] as f32;
        }
        for i in 0..8 {
            euclidean[i] = s[self.offset + 8 + i] as f32;
        }

        let coord = MixedCurvatureCoord::new(hyperbolic, spherical, euclidean).map_err(|e| {
            Reject::DomainViolation {
                offset: self.offset,
                detail: e.to_string(),
            }
        })?;

        // North pole [1,0,0,0] is the manifold's origin convention
        // (`MixedCurvatureCoord::origin`); its antipode is the cut locus.
        let mut pole = [0.0f32; 4];
        pole[0] = 1.0;
        let cos_val = dot_product_f32(&coord.spherical, &pole);
        if cos_val <= -1.0 + CUT_LOCUS_EPS {
            return Err(Reject::CutLocus {
                offset: self.offset,
                detail: format!(
                    "spherical block at antipode of reference pole (cos={cos_val:.6e}); geodesic non-unique"
                ),
            });
        }
        Ok(())
    }
}

/// A Dirichlet boundary anchor: node index `index` must remain at `value`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pin {
    pub index: usize,
    pub value: f64,
}

/// A weighted cellular sheaf harmonic-relaxation problem instance.
///
/// `D` (n_edges x n_nodes) is the coboundary operator, `w` (n_edges) is the
/// diagonal edge-weight metric, `b` (n_edges) is the target/potential vector.
/// The sheaf Laplacian is `L = D^T * diag(w) * D`.
#[derive(Clone, Debug)]
pub struct ExplicitMatrixProblem {
    d: DMatrix<f64>,
    w: DVector<f64>,
    b: DVector<f64>,
    pins: Vec<Pin>,
    manifold_guards: Vec<ManifoldGuard>,
    epoch: u64,
    /// Cached `D^T * diag(w) * D`, built once at construction.
    l: DMatrix<f64>,
    /// Cached `2.0 / lambda_max(l)`.
    safe_eta_bound: f64,
}

impl ExplicitMatrixProblem {
    /// Construct a new sheaf problem. Validates shapes, finiteness, strictly
    /// positive edge weights, and a well-posed (non-degenerate) Laplacian —
    /// all fail closed at construction rather than surfacing as a confusing
    /// `Stalled`/divide-by-zero later.
    pub fn new(
        d: DMatrix<f64>,
        w: DVector<f64>,
        b: DVector<f64>,
        pins: Vec<Pin>,
        manifold_guards: Vec<ManifoldGuard>,
        epoch: u64,
    ) -> Result<Self> {
        let n_edges = d.nrows();
        let n_nodes = d.ncols();
        if w.len() != n_edges || b.len() != n_edges {
            return Err(Reject::FiberMismatch {
                expected: n_edges,
                actual: w.len().min(b.len()),
            });
        }
        for (i, &v) in d.iter().enumerate() {
            if !v.is_finite() {
                return Err(Reject::NonFiniteState { index: i, value: v });
            }
        }
        for (i, &v) in w.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(Reject::DomainViolation {
                    offset: i,
                    detail: format!("edge weight must be finite and strictly positive, got {v}"),
                });
            }
        }
        for (i, &v) in b.iter().enumerate() {
            if !v.is_finite() {
                return Err(Reject::NonFiniteState { index: i, value: v });
            }
        }
        for pin in &pins {
            if !pin.value.is_finite() {
                return Err(Reject::NonFiniteState {
                    index: pin.index,
                    value: pin.value,
                });
            }
            if pin.index >= n_nodes {
                return Err(Reject::FiberMismatch {
                    expected: n_nodes,
                    actual: pin.index.saturating_add(1),
                });
            }
        }

        let l = laplacian(&d, &w);
        finite(l.iter().copied())?;
        let lambda_max = if n_nodes == 0 {
            0.0
        } else {
            let eigen =
                SymmetricEigen::try_new(l.clone(), f64::EPSILON, 1000).ok_or_else(|| {
                    Reject::DomainViolation {
                        offset: 0,
                        detail: "sheaf Laplacian eigensolve did not converge".into(),
                    }
                })?;
            // f64::max ignores individual NaNs, so validate the entire spectrum.
            finite(eigen.eigenvalues.iter().copied())?;
            eigen.eigenvalues.iter().copied().fold(f64::MIN, f64::max)
        };
        if !lambda_max.is_finite() || lambda_max <= 0.0 {
            return Err(Reject::DomainViolation {
                offset: 0,
                detail: format!(
                    "degenerate sheaf Laplacian: lambda_max={lambda_max} (D or W is structurally deficient)"
                ),
            });
        }

        let safe_eta_bound = 2.0 / lambda_max;
        finite([safe_eta_bound])?;
        if safe_eta_bound <= 0.0 {
            return Err(Reject::InvalidBudget {
                field: "safe_eta_bound".into(),
                value: safe_eta_bound,
            });
        }
        Ok(Self {
            d,
            w,
            b,
            pins,
            manifold_guards,
            epoch,
            l,
            safe_eta_bound,
        })
    }

    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.d.ncols()
    }

    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// `eta` must satisfy `0 < eta < safe_eta_bound` for the Lyapunov
    /// monotonicity guarantee `E(s') < E(s)` to hold theoretically.
    #[inline]
    pub fn safe_eta_bound(&self) -> f64 {
        self.safe_eta_bound
    }

    // Shared input validation; never takes a relaxation step.
    fn validate_candidate(&self, candidate: &CandidateState, budget: &Budget) -> Result<()> {
        let problem = self;
        for (field, value) in [
            ("residual_tol", budget.residual_tol),
            ("grad_tol", budget.grad_tol),
            ("energy_uncertainty_tol", budget.energy_uncertainty_tol),
            ("pin_tol", budget.pin_tol),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Reject::InvalidBudget {
                    field: field.into(),
                    value,
                });
            }
        }
        if let Some(value) = budget.eta {
            if !value.is_finite() || value <= 0.0 {
                return Err(Reject::InvalidBudget {
                    field: "eta".into(),
                    value,
                });
            }
        }
        if candidate.epoch != problem.epoch {
            return Err(Reject::EpochMismatch {
                problem_epoch: problem.epoch,
                candidate_epoch: candidate.epoch,
            });
        }

        let expected = problem.n_nodes();
        if candidate.s.len() != expected {
            return Err(Reject::FiberMismatch {
                expected,
                actual: candidate.s.len(),
            });
        }

        for (i, &v) in candidate.s.iter().enumerate() {
            if !v.is_finite() {
                return Err(Reject::NonFiniteState { index: i, value: v });
            }
        }

        for pin in &problem.pins {
            let actual = candidate.s[pin.index];
            let delta = (actual - pin.value).abs();
            if delta > budget.pin_tol {
                return Err(Reject::PinnedMoved {
                    index: pin.index,
                    anchor: pin.value,
                    actual,
                    delta,
                    pin_tol: budget.pin_tol,
                });
            }
        }

        for guard in &problem.manifold_guards {
            guard.check(&candidate.s)?;
        }

        Ok(())
    }

    /// Verify a supplied terminal certificate without modifying or repairing its state.
    /// Public diagnostic fields are untrusted: recompute them against this problem.
    /// This proves terminal compliance, not the history of relaxation steps.
    pub fn verify_terminal(&self, accepted: &AcceptedState, budget: &Budget) -> Result<()> {
        self.validate_candidate(&accepted.state, budget)?;
        let s = DVector::from_column_slice(&accepted.state.s);
        let residual = stable_norm(&residual_vec(self, &s));
        if !residual.is_finite() || residual > budget.residual_tol {
            return Err(Reject::ResidualExceeded {
                residual,
                residual_tol: budget.residual_tol,
            });
        }
        let actual_energy = energy(self, &s);
        let actual_grad = stable_norm(&masked_gradient(self, &s));
        if !actual_energy.is_finite()
            || !actual_grad.is_finite()
            || accepted.residual != residual
            || accepted.energy != actual_energy
            || accepted.grad_norm != actual_grad
            || !accepted.eta_used.is_finite()
            || accepted.eta_used <= 0.0
            || accepted.eta_used >= self.safe_eta_bound
        {
            return Err(Reject::DomainViolation {
                offset: 0,
                detail: "invalid terminal certificate diagnostics".into(),
            });
        }
        Ok(())
    }
}

/// Canonical alias maintaining backwards compatibility across the repository.
pub type SheafProblem = ExplicitMatrixProblem;

fn laplacian(d: &DMatrix<f64>, w: &DVector<f64>) -> DMatrix<f64> {
    // Use the same sqrt(W) scaling as energy and gradient, and form a
    // symmetric Gram matrix without materializing a dense diagonal W.
    let mut wd = d.clone();
    for (mut row, &wi) in wd.row_iter_mut().zip(w.iter()) {
        row *= wi.sqrt();
    }
    wd.transpose() * &wd
}

/// A candidate sheaf state to certify: a stalk assignment `s` (length
/// `n_nodes`) tagged with the snapshot epoch it was computed against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateState {
    pub s: Vec<f64>,
    pub epoch: u64,
}

/// The step budget and acceptance tolerances for one [`GeometryGate::certify`]
/// call.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// Step size. `None` auto-selects `0.9 * safe_eta_bound`. Anything at or
    /// above the safe bound is rejected before a single step is taken.
    pub eta: Option<f64>,
    /// Maximum relaxation steps before giving up with `ResidualExceeded`.
    pub max_steps: u32,
    /// Convergence target for the operator residual norm. Zero requests exact
    /// satisfaction in `certify_and_relax`; legacy `certify` requires > 0.
    pub residual_tol: f64,
    /// Below this gradient norm, further steps cannot reduce the residual;
    /// used to detect a harmonic obstruction (`Stalled`) instead of burning
    /// the remaining step budget.
    pub grad_tol: f64,
    /// Relative tolerance for the direct-vs-Taylor-identity energy delta
    /// cross-check (see [`check_uncertain_energy`]).
    pub energy_uncertainty_tol: f64,
    /// Absolute tolerance for pinned-node displacement.
    pub pin_tol: f64,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            eta: None,
            max_steps: 256,
            residual_tol: 1e-6,
            grad_tol: 1e-9,
            energy_uncertainty_tol: 1e-6,
            pin_tol: 1e-9,
        }
    }
}

/// The accepted terminal state and its certified diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcceptedState {
    pub state: CandidateState,
    pub energy: f64,
    pub residual: f64,
    pub grad_norm: f64,
    pub steps_taken: u32,
    pub eta_used: f64,
}

/// Spec 25 geometric + sheaf-cohomology deterministic fast gate.
pub trait GeometryGate: Send + Sync {
    /// Certify an explicit candidate by running the shared relaxation kernel.
    /// Non-convergence and numerical failures return an explicit rejection.
    fn certify(
        &self,
        problem: &ExplicitMatrixProblem,
        candidate: CandidateState,
        budget: &Budget,
    ) -> Result<AcceptedState>;
}

/// Common quadratic operator contract. Implementations are sealed so certification
/// cannot accidentally trust an unvalidated operator or spectral bound.
/// Numerical evaluations fail closed on empty or mismatched inputs: scalars
/// return NaN and vectors contain NaNs with the expected output dimension.
pub trait SheafOperator: sealed::Sealed {
    fn n_nodes(&self) -> usize;
    fn epoch(&self) -> u64;
    fn safe_eta_bound(&self) -> f64;
    fn pins(&self) -> &[Pin];
    fn guards(&self) -> &[ManifoldGuard];
    fn residual_vec(&self, s: &DVector<f64>) -> DVector<f64>;
    fn energy(&self, s: &DVector<f64>) -> f64;
    fn gradient(&self, s: &DVector<f64>) -> DVector<f64>;
    fn curvature(&self, g: &DVector<f64>) -> f64;
    fn states(&self, s: &[f64]) -> Vec<Vec<f64>>;
}
mod sealed {
    pub trait Sealed {}
}
impl sealed::Sealed for ExplicitMatrixProblem {}
impl SheafOperator for ExplicitMatrixProblem {
    fn n_nodes(&self) -> usize {
        self.n_nodes()
    }
    fn epoch(&self) -> u64 {
        self.epoch
    }
    fn safe_eta_bound(&self) -> f64 {
        self.safe_eta_bound
    }
    fn pins(&self) -> &[Pin] {
        &self.pins
    }
    fn guards(&self) -> &[ManifoldGuard] {
        &self.manifold_guards
    }
    fn residual_vec(&self, s: &DVector<f64>) -> DVector<f64> {
        if s.is_empty() || s.len() != self.n_nodes() {
            return DVector::from_element(self.d.nrows(), f64::NAN);
        }
        &self.d * s - &self.b
    }
    fn energy(&self, s: &DVector<f64>) -> f64 {
        if s.is_empty() || s.len() != self.n_nodes() {
            return f64::NAN;
        }
        let norm = self
            .residual_vec(s)
            .iter()
            .zip(self.w.iter())
            .fold(0.0_f64, |norm, (r, w)| norm.hypot(r * w.sqrt()));
        // Apply 1/2 before squaring: the energy may fit even when norm^2 does not.
        (0.5 * norm) * norm
    }
    fn gradient(&self, s: &DVector<f64>) -> DVector<f64> {
        if s.is_empty() || s.len() != self.n_nodes() {
            return DVector::from_element(self.n_nodes(), f64::NAN);
        }
        let mut r = self.residual_vec(s);
        let mut wd = self.d.clone();
        for ((mut row, &w), r) in wd.row_iter_mut().zip(self.w.iter()).zip(r.iter_mut()) {
            let root = w.sqrt();
            row *= root;
            *r *= root;
        }
        let mut g = wd.transpose() * r;
        for pin in &self.pins {
            g[pin.index] = 0.0;
        }
        g
    }
    fn curvature(&self, g: &DVector<f64>) -> f64 {
        if g.is_empty() || g.len() != self.n_nodes() {
            return f64::NAN;
        }
        g.dot(&(&self.l * g))
    }
    fn states(&self, s: &[f64]) -> Vec<Vec<f64>> {
        vec![s.to_vec()]
    }
}

/// One affine transition. Matrix entries are row-major; no global D or L is built.
#[derive(Clone, Debug)]
pub struct DynamicsStep {
    pub matrix: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

/// Window states are s_1..s_T; s_0 is immutable. Observation times are 1-based.
/// Residual norm includes sqrt(pin_weight)-weighted observation residuals.
#[derive(Clone, Debug)]
pub struct WindowDynamicsProblem {
    s0: Vec<f64>,
    steps: Vec<DynamicsStep>,
    observations: Vec<(usize, Vec<f64>)>,
    pin_weight: f64,
    epoch: u64,
    safe_eta_bound: f64,
}
fn finite(values: impl IntoIterator<Item = f64>) -> Result<()> {
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(Reject::NonFiniteState { index, value });
        }
    }
    Ok(())
}
// Scale before subtracting only when subtraction overflows. The ordinary
// path preserves cancellation precision for nearby observations.
fn scaled_difference(a: f64, b: f64, scale: f64) -> f64 {
    if scale == 0.0 {
        return 0.0;
    }
    let delta = a - b;
    if delta.is_finite() {
        scale * delta
    } else {
        scale * a - scale * b
    }
}

impl WindowDynamicsProblem {
    pub fn new(
        s0: Vec<f64>,
        steps: Vec<DynamicsStep>,
        observations: Vec<(usize, Vec<f64>)>,
        pin_weight: f64,
        epoch: u64,
    ) -> Result<Self> {
        let d = s0.len();
        if d == 0 || steps.is_empty() {
            return Err(Reject::DomainViolation {
                offset: 0,
                detail: "empty dynamics window or fiber".into(),
            });
        }
        finite(s0.iter().copied())?;
        if !pin_weight.is_finite() || pin_weight < 0.0 {
            return Err(Reject::InvalidBudget {
                field: "pin_weight".into(),
                value: pin_weight,
            });
        }
        let mut max_norm: f64 = 0.0;
        for step in &steps {
            for actual in [step.matrix.len(), step.bias.len()]
                .into_iter()
                .chain(step.matrix.iter().map(Vec::len))
            {
                if actual != d {
                    return Err(Reject::FiberMismatch {
                        expected: d,
                        actual,
                    });
                }
            }
            finite(step.matrix.iter().flatten().chain(&step.bias).copied())?;
            let inf = step
                .matrix
                .iter()
                .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
                .fold(0.0, f64::max);
            let one = (0..d)
                .map(|j| step.matrix.iter().map(|r| r[j].abs()).sum::<f64>())
                .fold(0.0, f64::max);
            let bound = one.sqrt() * inf.sqrt();
            finite([bound])?;
            max_norm = max_norm.max(bound);
        }
        let mut counts = vec![0usize; steps.len()];
        for (t, y) in &observations {
            if *t == 0 || *t > steps.len() {
                return Err(Reject::DomainViolation {
                    offset: *t,
                    detail: "observation time outside 1..=T".into(),
                });
            }
            if y.len() != d {
                return Err(Reject::FiberMismatch {
                    expected: d,
                    actual: y.len(),
                });
            }
            finite(y.iter().copied())?;
            counts[t - 1] += 1;
        }
        // Duplicate observations add their Hessians; count them in the bound.
        let lambda =
            (1.0 + max_norm).powi(2) + pin_weight * counts.into_iter().max().unwrap_or(0) as f64;
        let safe_eta_bound = 2.0 / lambda;
        finite([lambda, safe_eta_bound])?;
        if safe_eta_bound <= 0.0 {
            return Err(Reject::InvalidBudget {
                field: "safe_eta_bound".into(),
                value: safe_eta_bound,
            });
        }
        Ok(Self {
            s0,
            steps,
            observations,
            pin_weight,
            epoch,
            safe_eta_bound,
        })
    }
    fn edge_residuals(&self, s: &DVector<f64>, homogeneous: bool) -> DVector<f64> {
        let d = self.s0.len();
        DVector::from_iterator(
            self.n_nodes(),
            (0..self.steps.len()).flat_map(|t| {
                (0..d).map(move |i| {
                    let step = &self.steps[t];
                    let mp: f64 = (0..d)
                        .map(|j| {
                            step.matrix[i][j]
                                * if t == 0 {
                                    if homogeneous {
                                        0.0
                                    } else {
                                        self.s0[j]
                                    }
                                } else {
                                    s[(t - 1) * d + j]
                                }
                        })
                        .sum();
                    s[t * d + i] - mp - if homogeneous { 0.0 } else { step.bias[i] }
                })
            }),
        )
    }
}
impl sealed::Sealed for WindowDynamicsProblem {}
impl SheafOperator for WindowDynamicsProblem {
    fn n_nodes(&self) -> usize {
        self.s0.len() * self.steps.len()
    }
    fn epoch(&self) -> u64 {
        self.epoch
    }
    fn safe_eta_bound(&self) -> f64 {
        self.safe_eta_bound
    }
    fn pins(&self) -> &[Pin] {
        &[]
    }
    fn guards(&self) -> &[ManifoldGuard] {
        &[]
    }
    fn residual_vec(&self, s: &DVector<f64>) -> DVector<f64> {
        if s.is_empty() || s.len() != self.n_nodes() {
            return DVector::from_element(
                self.n_nodes() + self.observations.len() * self.s0.len(),
                f64::NAN,
            );
        }
        let mut r = self.edge_residuals(s, false).as_slice().to_vec();
        let d = self.s0.len();
        for (t, y) in &self.observations {
            r.extend(
                (0..d).map(|i| scaled_difference(s[(t - 1) * d + i], y[i], self.pin_weight.sqrt())),
            );
        }
        DVector::from_vec(r)
    }
    fn energy(&self, s: &DVector<f64>) -> f64 {
        if s.is_empty() || s.len() != self.n_nodes() {
            return f64::NAN;
        }
        let norm = stable_norm(&self.residual_vec(s));
        (0.5 * norm) * norm
    }
    fn gradient(&self, s: &DVector<f64>) -> DVector<f64> {
        if s.is_empty() || s.len() != self.n_nodes() {
            return DVector::from_element(self.n_nodes(), f64::NAN);
        }
        let d = self.s0.len();
        let r = self.edge_residuals(s, false);
        let mut g = r.clone();
        for t in 1..self.steps.len() {
            for j in 0..d {
                g[(t - 1) * d + j] -= (0..d)
                    .map(|i| self.steps[t].matrix[i][j] * r[t * d + i])
                    .sum::<f64>();
            }
        }
        for (t, y) in &self.observations {
            if self.pin_weight == 0.0 {
                continue;
            }
            for i in 0..d {
                g[(t - 1) * d + i] += scaled_difference(s[(t - 1) * d + i], y[i], self.pin_weight);
            }
        }
        g
    }
    fn curvature(&self, g: &DVector<f64>) -> f64 {
        if g.is_empty() || g.len() != self.n_nodes() {
            return f64::NAN;
        }
        let d = self.s0.len();
        self.edge_residuals(g, true).norm_squared()
            + self
                .observations
                .iter()
                .map(|(t, _)| {
                    if self.pin_weight == 0.0 {
                        0.0
                    } else {
                        let norm = (0..d).fold(0.0_f64, |norm, i| {
                            norm.hypot(self.pin_weight.sqrt() * g[(t - 1) * d + i])
                        });
                        norm * norm
                    }
                })
                .sum::<f64>()
    }
    fn states(&self, s: &[f64]) -> Vec<Vec<f64>> {
        s.chunks_exact(self.s0.len()).map(|s| s.to_vec()).collect()
    }
}
fn residual_vec(problem: &impl SheafOperator, s: &DVector<f64>) -> DVector<f64> {
    problem.residual_vec(s)
}
fn energy(problem: &impl SheafOperator, s: &DVector<f64>) -> f64 {
    problem.energy(s)
}
fn masked_gradient(problem: &impl SheafOperator, s: &DVector<f64>) -> DVector<f64> {
    problem.gradient(s)
}

/// One Lyapunov-monitored relaxation step: `s' = s - eta * masked_grad(s)`,
/// verified against energy monotonicity, numerical determinism, and pinned
/// invariance. Returns `(s', energy(s'), grad_norm(s'))` on success.
pub(crate) fn relax_step(
    problem: &impl SheafOperator,
    s: &DVector<f64>,
    eta: f64,
    energy_uncertainty_tol: f64,
    pin_tol: f64,
) -> Result<(DVector<f64>, f64, DVector<f64>)> {
    let e_before = energy(problem, s);
    let g = masked_gradient(problem, s);
    let s_next = s - eta * &g;

    for (i, &v) in s_next.iter().enumerate() {
        if !v.is_finite() {
            return Err(Reject::NonFiniteState { index: i, value: v });
        }
    }

    for pin in problem.pins() {
        let actual = s_next[pin.index];
        let delta = (actual - pin.value).abs();
        if delta > pin_tol {
            return Err(Reject::PinnedMoved {
                index: pin.index,
                anchor: pin.value,
                actual,
                delta,
                pin_tol,
            });
        }
    }

    let e_after = energy(problem, &s_next);
    finite([e_before, e_after, stable_norm(&g)])?;
    check_uncertain_energy(problem, &g, eta, e_before, e_after, energy_uncertainty_tol)?;

    if e_after > e_before {
        return Err(Reject::EnergyRose {
            kind: EnergyRoseKind::EmpiricalIncrease,
            e_before,
            e_after,
            eta,
            safe_eta_bound: problem.safe_eta_bound(),
        });
    }

    for guard in problem.guards() {
        guard.check(s_next.as_slice())?;
    }
    let grad_next = masked_gradient(problem, &s_next);
    finite([stable_norm(&grad_next)])?;
    Ok((s_next, e_after, grad_next))
}

/// Cross-checks the observed energy delta against the exact quadratic Taylor
/// identity `E(s - eta*g) - E(s) = -eta*||g||^2 + 0.5*eta^2*g^T*L*g` (exact
/// because `E` is an exact quadratic form; `L = D^T*W*D`). `g` is the masked
/// gradient actually applied, so the identity holds even though `g` differs
/// from the true gradient of `E` at pinned indices: at every unmasked index
/// `L*g`'s dual pairing with `g` reduces to `||g||^2` in the linear term.
///
/// A mismatch beyond `tol` means one of the two independent code paths
/// (direct re-evaluation of `E` vs. this closed-form identity) lost precision
/// or disagrees outright — a genuine numerical-determinism failure, not a
/// cosmetic check.
pub(crate) fn check_uncertain_energy(
    problem: &impl SheafOperator,
    g: &DVector<f64>,
    eta: f64,
    e_before: f64,
    e_after: f64,
    tol: f64,
) -> Result<()> {
    let direct_delta = e_after - e_before;
    let predicted_delta = taylor_delta(problem, g, eta);

    finite([e_before, e_after, direct_delta, predicted_delta])?;
    let diff = (direct_delta - predicted_delta).abs();
    let scale = 1.0_f64.max(e_before.abs());
    let allowed = tol * scale;
    finite([diff, allowed])?;
    if diff > allowed {
        return Err(Reject::UncertainEnergy {
            direct_delta,
            predicted_delta,
            diff,
            tol,
        });
    }
    Ok(())
}

/// Exact quadratic Taylor identity: `E(s - eta*g) - E(s) = -eta*||g||^2 +
/// 0.5*eta^2*g^T*L*g` (see [`check_uncertain_energy`] for why this is exact,
/// not an approximation, and why it holds even for the pin-masked `g`).
pub(crate) fn taylor_delta(problem: &impl SheafOperator, g: &DVector<f64>, eta: f64) -> f64 {
    // Evaluate with the actual displacement to avoid eta^2 overflowing or
    // underflowing before it multiplies the compensating curvature scale.
    let displacement = eta * g;
    -g.dot(&displacement) + 0.5 * problem.curvature(&displacement)
}

/// Concrete [`GeometryGate`]: discrete Lyapunov energy descent on a weighted
/// cellular sheaf, via masked (Dirichlet-pinned) gradient relaxation.
#[derive(Debug, Clone, Copy, Default)]
pub struct LaplacianHeatFlowGate;

impl LaplacianHeatFlowGate {
    /// Relax either validated operator with identical per-step checks. Input is
    /// flat: explicit coordinates, or time-major s_1..s_T for windows. The fixed
    /// s_0 is owned by the window and cannot move. Numerical failures return Err;
    /// budget exhaustion and stagnation return diagnostics with a non-Converged
    /// status, which callers MUST reject when an acceptance certificate is needed.
    pub fn certify_and_relax(
        &self,
        problem: &impl SheafOperator,
        candidate: CandidateState,
        budget: &Budget,
    ) -> Result<CertifiedCandidate> {
        if !budget.residual_tol.is_finite() || budget.residual_tol < 0.0 {
            return Err(Reject::InvalidBudget {
                field: "residual_tol".into(),
                value: budget.residual_tol,
            });
        }
        for (field, value) in [
            ("grad_tol", budget.grad_tol),
            ("energy_uncertainty_tol", budget.energy_uncertainty_tol),
            ("pin_tol", budget.pin_tol),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Reject::InvalidBudget {
                    field: field.into(),
                    value,
                });
            }
        }
        if let Some(value) = budget.eta {
            if !value.is_finite() || value <= 0.0 {
                return Err(Reject::InvalidBudget {
                    field: "eta".into(),
                    value,
                });
            }
        }
        if candidate.epoch != problem.epoch() {
            return Err(Reject::EpochMismatch {
                problem_epoch: problem.epoch(),
                candidate_epoch: candidate.epoch,
            });
        }

        let expected = problem.n_nodes();
        if candidate.s.len() != expected {
            return Err(Reject::FiberMismatch {
                expected,
                actual: candidate.s.len(),
            });
        }

        for (i, &v) in candidate.s.iter().enumerate() {
            if !v.is_finite() {
                return Err(Reject::NonFiniteState { index: i, value: v });
            }
        }

        for pin in problem.pins() {
            let actual = candidate.s[pin.index];
            let delta = (actual - pin.value).abs();
            if delta > budget.pin_tol {
                return Err(Reject::PinnedMoved {
                    index: pin.index,
                    anchor: pin.value,
                    actual,
                    delta,
                    pin_tol: budget.pin_tol,
                });
            }
        }

        for guard in problem.guards() {
            guard.check(&candidate.s)?;
        }

        let eta = budget.eta.unwrap_or(0.9 * problem.safe_eta_bound());
        if !(eta > 0.0 && eta < problem.safe_eta_bound()) {
            let e0 = energy(problem, &DVector::from_vec(candidate.s.clone()));
            return Err(Reject::EnergyRose {
                kind: EnergyRoseKind::StepBoundViolated,
                e_before: e0,
                e_after: e0,
                eta,
                safe_eta_bound: problem.safe_eta_bound(),
            });
        }

        let mut s = DVector::from_vec(candidate.s);
        let mut grad_norm = stable_norm(&masked_gradient(problem, &s));
        let mut residual = stable_norm(&residual_vec(problem, &s));
        let mut e_current = energy(problem, &s);

        finite([e_current, residual, grad_norm])?;
        let initial_energy = e_current;
        let mut steps_taken = 0;
        let status = loop {
            if residual <= budget.residual_tol {
                break RelaxationStatus::Converged;
            }
            let g_current = masked_gradient(problem, &s);
            grad_norm = stable_norm(&g_current);
            let predicted_decrease = -taylor_delta(problem, &g_current, eta);
            finite([grad_norm, predicted_decrease])?;
            let ulp_floor = 16.0 * f64::EPSILON * e_current.abs().max(1.0);
            if grad_norm < budget.grad_tol || predicted_decrease < ulp_floor {
                break RelaxationStatus::Stalled;
            }
            if steps_taken == budget.max_steps {
                break RelaxationStatus::StepBudgetExhausted;
            }
            let (next, e, g) = relax_step(
                problem,
                &s,
                eta,
                budget.energy_uncertainty_tol,
                budget.pin_tol,
            )?;
            s = next;
            e_current = e;
            residual = stable_norm(&residual_vec(problem, &s));
            grad_norm = stable_norm(&g);
            finite([e_current, residual, grad_norm])?;
            steps_taken += 1;
        };
        Ok(CertifiedCandidate {
            states: problem.states(s.as_slice()),
            state: CandidateState {
                s: s.as_slice().to_vec(),
                epoch: candidate.epoch,
            },
            initial_energy,
            final_energy: e_current,
            residual_norm: residual,
            grad_norm,
            steps_taken,
            eta_used: eta,
            status,
        })
    }
}

/// Non-converged candidates are diagnostics, never acceptance certificates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelaxationStatus {
    Converged,
    /// Stationary or below floating-point resolution with residual above target.
    /// This diagnoses an obstruction; it does not prove topological H^1 != 0.
    Stalled,
    StepBudgetExhausted,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertifiedCandidate {
    pub states: Vec<Vec<f64>>,
    pub state: CandidateState,
    pub initial_energy: f64,
    pub final_energy: f64,
    pub residual_norm: f64,
    pub grad_norm: f64,
    pub steps_taken: u32,
    pub eta_used: f64,
    pub status: RelaxationStatus,
}
impl GeometryGate for LaplacianHeatFlowGate {
    fn certify(
        &self,
        problem: &ExplicitMatrixProblem,
        candidate: CandidateState,
        budget: &Budget,
    ) -> Result<AcceptedState> {
        if budget.residual_tol == 0.0 {
            return Err(Reject::InvalidBudget {
                field: "residual_tol".into(),
                value: budget.residual_tol,
            });
        }
        let c = self.certify_and_relax(problem, candidate, budget)?;
        match c.status {
            RelaxationStatus::Converged => Ok(AcceptedState {
                state: c.state,
                energy: c.final_energy,
                residual: c.residual_norm,
                grad_norm: c.grad_norm,
                steps_taken: c.steps_taken,
                eta_used: c.eta_used,
            }),
            RelaxationStatus::Stalled => Err(Reject::Stalled {
                grad_norm: c.grad_norm,
                grad_tol: budget.grad_tol,
                residual: c.residual_norm,
                residual_tol: budget.residual_tol,
            }),
            RelaxationStatus::StepBudgetExhausted => Err(Reject::ResidualExceeded {
                residual: c.residual_norm,
                residual_tol: budget.residual_tol,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity_problem(n: usize) -> ExplicitMatrixProblem {
        ExplicitMatrixProblem::new(
            DMatrix::identity(n, n),
            DVector::from_element(n, 1.0),
            DVector::zeros(n),
            vec![],
            vec![],
            9,
        )
        .unwrap()
    }

    #[test]
    fn weighted_quadratics_preserve_representable_extreme_scales() {
        for (d, w, state, expected_energy) in [
            (1e-100, 1e300, 1e-100, 5e-101),
            (1e100, 1e-300, 1e100, 5e99),
        ] {
            let p = ExplicitMatrixProblem::new(
                DMatrix::from_element(1, 1, d),
                DVector::from_element(1, w),
                DVector::zeros(1),
                vec![],
                vec![],
                0,
            )
            .unwrap();
            let s = DVector::from_element(1, state);
            assert!((p.energy(&s) / expected_energy - 1.0).abs() < 1e-14);
            assert!((p.gradient(&s)[0] - 1.0).abs() < 1e-14);
        }
        // A genuinely unrepresentable Laplacian must still fail closed.
        let p = ExplicitMatrixProblem::new(
            DMatrix::from_element(1, 1, 1e-100),
            DVector::from_element(1, 1e-220),
            DVector::zeros(1),
            vec![],
            vec![],
            0,
        );
        assert!(p.is_err()); // The final Laplacian itself is unrepresentable.
        let p = ExplicitMatrixProblem::new(
            DMatrix::from_element(1, 1, 1e200),
            DVector::from_element(1, 1e-300),
            DVector::zeros(1),
            vec![],
            vec![],
            0,
        )
        .unwrap();
        assert!((p.safe_eta_bound() / 2e-100 - 1.0).abs() < 1e-14);
        // W*r underflows, but D^T W*r is representable.
        let g = p.gradient(&DVector::from_element(1, 1e-300));
        assert!((g[0] / 1e-200 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn ill_conditioned_spectrum_does_not_hide_unresolved_residual() {
        // Nullspaces are valid for sheaves; an infinite full condition number
        // must not be mistaken for convergence in the nonzero modes.
        let p = ExplicitMatrixProblem::new(
            DMatrix::from_diagonal(&DVector::from_vec(vec![0.0, 1e-150, 1e150])),
            DVector::from_element(3, 1.0),
            DVector::zeros(3),
            vec![],
            vec![],
            0,
        )
        .unwrap();
        assert!((p.safe_eta_bound() / 2e-300 - 1.0).abs() < 1e-14);
        let result = LaplacianHeatFlowGate
            .certify_and_relax(
                &p,
                CandidateState {
                    s: vec![0.0, 1.0, 0.0],
                    epoch: 0,
                },
                &Budget {
                    residual_tol: 1e-160,
                    ..Budget::default()
                },
            )
            .unwrap();
        assert_eq!(result.status, RelaxationStatus::Stalled);
        assert_eq!(result.residual_norm, 1e-150);
    }

    #[test]
    fn taylor_identity_avoids_step_size_square_overflow() {
        let p = ExplicitMatrixProblem::new(
            DMatrix::from_element(1, 1, 1e-100),
            DVector::from_element(1, 1.0),
            DVector::zeros(1),
            vec![],
            vec![],
            0,
        )
        .unwrap();
        let s = DVector::from_element(1, 1e100);
        let (next, energy, _) = relax_step(&p, &s, 1e200, 1e-12, 1e-9).unwrap();
        assert!(energy < 1e-28);
        assert!(next[0].abs() < 1e85);
    }

    #[test]
    fn zero_weight_observations_are_inert_even_when_difference_overflows() {
        let p = WindowDynamicsProblem::new(
            vec![f64::MAX],
            vec![DynamicsStep {
                matrix: vec![vec![1.0]],
                bias: vec![0.0],
            }],
            vec![(1, vec![-f64::MAX])],
            0.0,
            0,
        )
        .unwrap();
        let result = LaplacianHeatFlowGate
            .certify_and_relax(
                &p,
                CandidateState {
                    s: vec![f64::MAX],
                    epoch: 0,
                },
                &Budget::default(),
            )
            .unwrap();
        assert_eq!(result.status, RelaxationStatus::Converged);
        assert_eq!(result.final_energy, 0.0);
        assert_eq!(result.grad_norm, 0.0);
    }

    #[test]
    fn small_observation_weight_keeps_scaled_difference_finite() {
        let p = WindowDynamicsProblem::new(
            vec![1e308],
            vec![DynamicsStep {
                matrix: vec![vec![1.0]],
                bias: vec![0.0],
            }],
            vec![(1, vec![-1e308])],
            1e-310,
            0,
        )
        .unwrap();
        let s = DVector::from_element(1, 1e308);
        assert!((p.residual_vec(&s)[1] / 2e153 - 1.0).abs() < 1e-13);
        assert!((p.energy(&s) / 2e306 - 1.0).abs() < 1e-13);
        assert!((p.gradient(&s)[0] / 0.02 - 1.0).abs() < 1e-13);
    }

    #[test]
    fn half_squared_norm_avoids_unnecessary_energy_overflow() {
        let p = identity_problem(1);
        let energy = p.energy(&DVector::from_element(1, 1.5e154));
        assert!(energy.is_finite());
        assert!((energy / 1.125e308 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn stable_norm_preserves_extreme_magnitudes() {
        for scale in [1e-200, 1e200] {
            let norm = stable_norm(&DVector::from_vec(vec![3.0 * scale, 4.0 * scale]));
            assert!((norm / scale - 5.0).abs() < 1e-14);
        }
    }

    #[test]
    fn tiny_residual_cannot_converge_or_verify_above_tolerance() {
        let explicit = identity_problem(1);
        let window = WindowDynamicsProblem::new(
            vec![0.0],
            vec![DynamicsStep {
                matrix: vec![vec![1.0]],
                bias: vec![0.0],
            }],
            vec![],
            0.0,
            9,
        )
        .unwrap();
        let candidate = CandidateState {
            s: vec![1e-200],
            epoch: 9,
        };
        let budget = Budget {
            residual_tol: 1e-210,
            ..Budget::default()
        };
        let gate = LaplacianHeatFlowGate;
        let results = [
            gate.certify_and_relax(&explicit, candidate.clone(), &budget)
                .unwrap(),
            gate.certify_and_relax(&window, candidate.clone(), &budget)
                .unwrap(),
        ];
        for result in results {
            assert_eq!(result.status, RelaxationStatus::Stalled);
            assert_eq!(result.residual_norm, 1e-200);
            assert_eq!(result.grad_norm, 1e-200);
        }
        assert!(matches!(
            gate.certify(&explicit, candidate.clone(), &budget),
            Err(Reject::Stalled { .. })
        ));
        let forged = AcceptedState {
            state: candidate,
            energy: 0.0,
            residual: 0.0,
            grad_norm: 0.0,
            steps_taken: 0,
            eta_used: 1.0,
        };
        assert!(matches!(
            explicit.verify_terminal(&forged, &budget),
            Err(Reject::ResidualExceeded { .. })
        ));
    }

    #[test]
    fn public_operators_fail_closed_on_wrong_dimensions() {
        fn check(p: &impl SheafOperator) {
            for len in [0, p.n_nodes() - 1, p.n_nodes() + 1] {
                let s = DVector::zeros(len);
                let residual = p.residual_vec(&s);
                assert!(!residual.is_empty());
                assert!(residual.iter().all(|x| x.is_nan()));
                let gradient = p.gradient(&s);
                assert_eq!(gradient.len(), p.n_nodes());
                assert!(gradient.iter().all(|x| x.is_nan()));
                assert!(p.energy(&s).is_nan());
                assert!(p.curvature(&s).is_nan());
            }
        }
        check(&identity_problem(4));
        check(&window_problem());
    }

    #[test]
    fn manifold_requires_unit_sphere_in_f64() {
        let guard = ManifoldGuard { offset: 2 };
        let mut s = vec![0.0; 18];
        for radius in [
            0.0,
            0.5,
            2.0,
            1e200,
            f64::INFINITY,
            f64::NAN,
            1.0 + 1.001e-5,
        ] {
            s[6] = radius;
            assert!(matches!(
                guard.check(&s),
                Err(Reject::DomainViolation { .. })
            ));
        }
        for radius in [1.0, 1.0 + 0.999e-5] {
            s[6] = radius;
            assert!(guard.check(&s).is_ok());
        }
        s[6] = -1.0;
        assert!(matches!(guard.check(&s), Err(Reject::CutLocus { .. })));
    }

    #[test]
    fn maximum_pin_index_returns_shape_error() {
        let result = ExplicitMatrixProblem::new(
            DMatrix::identity(1, 1),
            DVector::from_element(1, 1.0),
            DVector::zeros(1),
            vec![Pin {
                index: usize::MAX,
                value: 0.0,
            }],
            vec![],
            9,
        );
        assert!(matches!(
            result,
            Err(Reject::FiberMismatch {
                expected: 1,
                actual: usize::MAX
            })
        ));
    }

    fn window_problem() -> WindowDynamicsProblem {
        WindowDynamicsProblem::new(
            vec![0.4, -0.2],
            vec![
                DynamicsStep {
                    matrix: vec![vec![0.5, 0.2], vec![-0.1, 0.8]],
                    bias: vec![0.1, -0.3],
                },
                DynamicsStep {
                    matrix: vec![vec![0.7, -0.4], vec![0.3, 0.6]],
                    bias: vec![-0.2, 0.5],
                },
            ],
            vec![
                (1, vec![0.3, 0.1]),
                (2, vec![-0.4, 0.2]),
                (2, vec![0.2, -0.1]),
            ],
            1.7,
            9,
        )
        .unwrap()
    }

    #[test]
    fn window_operator_matches_independent_dense_quadratic_and_finite_differences() {
        let p = window_problem();
        // Dense construction is a test oracle only; production stays block-local.
        let d = DMatrix::from_row_slice(
            10,
            4,
            &[
                1., 0., 0., 0., 0., 1., 0., 0., -0.7, 0.4, 1., 0., -0.3, -0.6, 0., 1., 1., 0., 0.,
                0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
        );
        let b = DVector::from_vec(vec![0.26, -0.5, -0.2, 0.5, 0.3, 0.1, -0.4, 0.2, 0.2, -0.1]);
        let dense = ExplicitMatrixProblem::new(
            d,
            DVector::from_vec(vec![1., 1., 1., 1., 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]),
            b,
            vec![],
            vec![],
            9,
        )
        .unwrap();
        let s = DVector::from_vec(vec![0.8, -0.7, 0.1, 0.9]);
        assert!((p.energy(&s) - dense.energy(&s)).abs() < 1e-12);
        assert!((p.gradient(&s) - dense.gradient(&s)).norm() < 1e-12);
        assert!((p.curvature(&s) - dense.curvature(&s)).abs() < 1e-12);
        assert!(p.safe_eta_bound() <= dense.safe_eta_bound());
        for i in 0..s.len() {
            let mut plus = s.clone();
            let mut minus = s.clone();
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            let derivative = (p.energy(&plus) - p.energy(&minus)) / 2e-6;
            assert!((derivative - p.gradient(&s)[i]).abs() < 1e-8);
        }
        let budget = Budget {
            eta: Some(0.1),
            max_steps: 1000,
            ..Budget::default()
        };
        let candidate = CandidateState {
            s: s.as_slice().to_vec(),
            epoch: 9,
        };
        let a = LaplacianHeatFlowGate
            .certify_and_relax(&p, candidate.clone(), &budget)
            .unwrap();
        let b = LaplacianHeatFlowGate
            .certify_and_relax(&dense, candidate, &budget)
            .unwrap();
        assert_eq!(a.status, RelaxationStatus::Stalled);
        assert_eq!(a.status, b.status);
        assert!((a.final_energy - b.final_energy).abs() < 1e-12);
        assert!((DVector::from_vec(a.state.s) - DVector::from_vec(b.state.s)).norm() < 1e-10);
    }

    #[test]
    fn window_relaxes_to_affine_trajectory_and_checks_each_step() {
        let p = WindowDynamicsProblem::new(
            vec![2.],
            vec![
                DynamicsStep {
                    matrix: vec![vec![0.5]],
                    bias: vec![1.],
                },
                DynamicsStep {
                    matrix: vec![vec![-0.25]],
                    bias: vec![0.3],
                },
            ],
            vec![(2, vec![-0.2])],
            2.,
            4,
        )
        .unwrap();
        let candidate = CandidateState {
            s: vec![-3., 4.],
            epoch: 4,
        };
        assert!(matches!(
            relax_step(
                &p,
                &DVector::from_vec(candidate.s.clone()),
                100.0 * p.safe_eta_bound(),
                1e-6,
                1e-9
            ),
            Err(Reject::EnergyRose {
                kind: EnergyRoseKind::EmpiricalIncrease,
                ..
            })
        ));

        let budget = Budget {
            max_steps: 1000,
            ..Budget::default()
        };
        let mut s = DVector::from_vec(candidate.s.clone());
        let mut e = p.energy(&s);
        for _ in 0..20 {
            let (next, after, _) =
                relax_step(&p, &s, 0.5 * p.safe_eta_bound(), 1e-6, 1e-9).unwrap();
            assert!(after.is_finite() && after <= e);
            s = next;
            e = after;
        }
        let c = LaplacianHeatFlowGate
            .certify_and_relax(&p, candidate, &budget)
            .unwrap();
        assert_eq!(c.status, RelaxationStatus::Converged);
        assert_eq!(c.states.len(), 2);
        assert!((c.states[0][0] - 2.).abs() < 1e-6);
        assert!((c.states[1][0] + 0.2).abs() < 1e-6);
        assert!(c.final_energy < c.initial_energy);
        assert!(c.residual_norm <= budget.residual_tol);
        assert!(c.steps_taken > 0);
        assert_eq!(p.s0, vec![2.]);
        let accepted = LaplacianHeatFlowGate
            .certify_and_relax(&p, c.state, &budget)
            .unwrap();
        assert_eq!(accepted.steps_taken, 0);
    }

    #[test]
    fn window_budget_stall_and_invalid_inputs_are_explicit() {
        let p = window_problem();
        let candidate = CandidateState {
            s: vec![0.; 4],
            epoch: 9,
        };
        let c = LaplacianHeatFlowGate
            .certify_and_relax(
                &p,
                candidate.clone(),
                &Budget {
                    max_steps: 0,
                    ..Budget::default()
                },
            )
            .unwrap();
        assert_eq!(c.status, RelaxationStatus::StepBudgetExhausted);
        assert_eq!(c.initial_energy, c.final_energy);
        assert!(matches!(
            LaplacianHeatFlowGate.certify_and_relax(
                &p,
                candidate.clone(),
                &Budget {
                    eta: Some(p.safe_eta_bound()),
                    ..Budget::default()
                }
            ),
            Err(Reject::EnergyRose {
                kind: EnergyRoseKind::StepBoundViolated,
                ..
            })
        ));
        for s in [vec![0.; 3], vec![f64::NAN; 4], vec![f64::MAX; 4]] {
            assert!(LaplacianHeatFlowGate
                .certify_and_relax(&p, CandidateState { s, epoch: 9 }, &Budget::default())
                .is_err());
        }
        assert!(matches!(
            LaplacianHeatFlowGate.certify_and_relax(
                &p,
                CandidateState {
                    epoch: 0,
                    ..candidate
                },
                &Budget::default()
            ),
            Err(Reject::EpochMismatch { .. })
        ));
        for t in [0, 3] {
            assert!(WindowDynamicsProblem::new(
                p.s0.clone(),
                p.steps.clone(),
                vec![(t, vec![0.; 2])],
                1.,
                0
            )
            .is_err());
        }
        for weight in [-1., f64::NAN, f64::INFINITY] {
            assert!(
                WindowDynamicsProblem::new(p.s0.clone(), p.steps.clone(), vec![], weight, 0)
                    .is_err()
            );
        }
        for step in [
            DynamicsStep {
                matrix: vec![vec![1.]],
                bias: vec![0.; 2],
            },
            DynamicsStep {
                matrix: vec![vec![f64::MAX; 2]; 2],
                bias: vec![0.; 2],
            },
            DynamicsStep {
                matrix: vec![vec![0.; 2]; 2],
                bias: vec![f64::NAN; 2],
            },
        ] {
            assert!(WindowDynamicsProblem::new(p.s0.clone(), vec![step], vec![], 0., 0).is_err());
        }
        assert!(WindowDynamicsProblem::new(vec![], vec![], vec![], 0., 0).is_err());
    }

    #[test]
    fn long_window_uses_local_operators_and_zero_weight_observations() {
        let p = WindowDynamicsProblem::new(
            vec![1., 2.],
            vec![
                DynamicsStep {
                    matrix: vec![vec![1., 0.], vec![0., 1.]],
                    bias: vec![0., 0.],
                };
                10_000
            ],
            vec![(1, vec![100., 100.])],
            0.,
            0,
        )
        .unwrap();
        let c = LaplacianHeatFlowGate
            .certify_and_relax(
                &p,
                CandidateState {
                    s: [1., 2.].repeat(10_000),
                    epoch: 0,
                },
                &Budget {
                    residual_tol: 0.0,
                    ..Budget::default()
                },
            )
            .unwrap();
        assert_eq!(c.status, RelaxationStatus::Converged);
        assert_eq!(c.steps_taken, 0);
        assert_eq!(c.final_energy, 0.);
        assert_eq!(c.states.len(), 10_000);
    }

    #[test]
    fn explicit_overflow_and_nan_crosscheck_fail_closed() {
        assert!(matches!(
            ExplicitMatrixProblem::new(
                DMatrix::from_element(1, 1, f64::MAX),
                DVector::from_element(1, 1.),
                DVector::zeros(1),
                vec![],
                vec![],
                0
            ),
            Err(Reject::NonFiniteState { .. })
        ));
        let p = path_problem(1);
        assert!(matches!(
            check_uncertain_energy(&p, &DVector::zeros(3), 0.1, 1., f64::NAN, 1e-6),
            Err(Reject::NonFiniteState { .. })
        ));
        assert!(matches!(
            LaplacianHeatFlowGate.certify(
                &p,
                CandidateState {
                    s: vec![f64::MAX, -f64::MAX, 0.],
                    epoch: 1
                },
                &Budget::default()
            ),
            Err(Reject::NonFiniteState { .. })
        ));
    }

    #[test]
    fn invalid_tolerances_fail_before_certification() {
        let problem = path_problem(1);
        let candidate = CandidateState {
            s: vec![2.0, 1.0, 0.0],
            epoch: 1,
        };
        for invalid in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            for field in 0..4 {
                let mut budget = Budget::default();
                match field {
                    0 => budget.residual_tol = invalid,
                    1 => budget.grad_tol = invalid,
                    2 => budget.energy_uncertainty_tol = invalid,
                    _ => budget.pin_tol = invalid,
                }
                assert!(matches!(
                    LaplacianHeatFlowGate.certify(&problem, candidate.clone(), &budget),
                    Err(Reject::InvalidBudget { .. })
                ));
            }
        }
    }

    /// Path graph 0-1-2 with a single consistent target: D is the incidence
    /// map of edges (0,1) and (1,2), b is chosen so `b = D*s*` for some `s*`,
    /// i.e. `b` is exactly a coboundary and the flow converges to zero
    /// residual.
    fn path_problem(epoch: u64) -> ExplicitMatrixProblem {
        let d = DMatrix::from_row_slice(2, 3, &[1.0, -1.0, 0.0, 0.0, 1.0, -1.0]);
        let w = DVector::from_row_slice(&[1.0, 1.0]);
        // s* = [2.0, 1.0, 0.0] => D*s* = [1.0, 1.0] = b: exactly reachable.
        let b = DVector::from_row_slice(&[1.0, 1.0]);
        ExplicitMatrixProblem::new(d, w, b, vec![], vec![], epoch).unwrap()
    }

    /// Directed 3-cycle 0->1->2->0. `D` is the coboundary of a cycle graph:
    /// its image is exactly the zero-sum hyperplane (rank 2, kernel =
    /// constants). Any `b` with `sum(b) != 0` is *not* in `range(D)`: a
    /// genuine nonzero-H^1 harmonic obstruction (this is the classic
    /// cycle-consistency example from cellular sheaf theory).
    fn cycle_problem_with_obstruction(epoch: u64) -> ExplicitMatrixProblem {
        let d = DMatrix::from_row_slice(
            3,
            3,
            &[
                1.0, -1.0, 0.0, //
                0.0, 1.0, -1.0, //
                -1.0, 0.0, 1.0, //
            ],
        );
        let w = DVector::from_row_slice(&[1.0, 1.0, 1.0]);
        let b = DVector::from_row_slice(&[1.0, 0.0, 0.0]); // sum = 1 != 0
        ExplicitMatrixProblem::new(d, w, b, vec![], vec![], epoch).unwrap()
    }

    #[test]
    fn laplacian_relaxation_strictly_decreases_energy() {
        let problem = path_problem(1);
        let s0 = DVector::from_row_slice(&[0.0, 0.0, 0.0]);
        let eta = 0.9 * problem.safe_eta_bound();
        let e0 = energy(&problem, &s0);
        let (s1, e1, _grad) = relax_step(&problem, &s0, eta, 1e-6, 1e-9).unwrap();
        assert!(e1 < e0, "energy must strictly decrease: {e0} -> {e1}");

        // A second step decreases it further still (monotone, not one-shot).
        let (_s2, e2, _grad2) = relax_step(&problem, &s1, eta, 1e-6, 1e-9).unwrap();
        assert!(e2 < e1, "energy must keep decreasing: {e1} -> {e2}");
    }

    #[test]
    fn certify_converges_to_zero_residual_on_reachable_target() {
        let problem = path_problem(7);
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![0.0, 0.0, 0.0],
            epoch: 7,
        };
        let budget = Budget::default();
        let accepted = gate.certify(&problem, candidate, &budget).unwrap();
        assert!(accepted.residual <= budget.residual_tol);
        assert!(accepted.steps_taken > 0);
        assert!(accepted.energy < 0.5); // started well above zero, converged down
    }

    #[test]
    fn certify_rejects_eta_at_or_above_safe_bound() {
        let problem = path_problem(1);
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![0.0, 0.0, 0.0],
            epoch: 1,
        };
        let budget = Budget {
            eta: Some(problem.safe_eta_bound()), // not strictly less than bound
            ..Budget::default()
        };
        let err = gate.certify(&problem, candidate, &budget).unwrap_err();
        match err {
            Reject::EnergyRose {
                kind: EnergyRoseKind::StepBoundViolated,
                ..
            } => {}
            other => panic!("expected StepBoundViolated EnergyRose, got {other:?}"),
        }
    }

    #[test]
    fn relax_step_with_unsafe_eta_actually_raises_energy() {
        // Directly exercise the empirical monotonicity check (not just the
        // cheap pre-check) by stepping with eta far above the safe bound
        // along the dominant eigenvector direction.
        let problem = path_problem(1);
        let unsafe_eta = 4.0 * problem.safe_eta_bound(); // > 2/lambda_max
        let s0 = DVector::from_row_slice(&[0.0, 0.0, 0.0]);
        let err = relax_step(&problem, &s0, unsafe_eta, 1e-6, 1e-9).unwrap_err();
        match err {
            Reject::EnergyRose {
                kind: EnergyRoseKind::EmpiricalIncrease,
                e_before,
                e_after,
                ..
            } => assert!(e_after > e_before, "energy should have genuinely risen"),
            other => panic!("expected EmpiricalIncrease EnergyRose, got {other:?}"),
        }
    }

    #[test]
    fn certify_rejects_residual_exceeded_when_budget_exhausted() {
        let problem = path_problem(3);
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![0.0, 0.0, 0.0],
            epoch: 3,
        };
        let budget = Budget {
            eta: None,
            max_steps: 1, // too few steps to reach residual_tol
            residual_tol: 1e-12,
            grad_tol: 1e-15,
            energy_uncertainty_tol: 1e-6,
            pin_tol: 1e-9,
        };
        let err = gate.certify(&problem, candidate, &budget).unwrap_err();
        match err {
            Reject::ResidualExceeded { residual, .. } => assert!(residual > 1e-12),
            other => panic!("expected ResidualExceeded, got {other:?}"),
        }
    }

    #[test]
    fn certify_detects_stalled_harmonic_obstruction() {
        // b is not in range(D): the flow's gradient collapses toward zero
        // while the residual plateaus above tolerance -- H^1 != 0.
        let problem = cycle_problem_with_obstruction(1);
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![0.0, 0.0, 0.0],
            epoch: 1,
        };
        let budget = Budget {
            eta: None,
            max_steps: 5000,
            residual_tol: 1e-9, // unreachable: obstruction floor is > 0
            grad_tol: 1e-6,
            energy_uncertainty_tol: 1e-6,
            pin_tol: 1e-9,
        };
        let err = gate.certify(&problem, candidate, &budget).unwrap_err();
        match err {
            Reject::Stalled {
                grad_norm,
                residual,
                ..
            } => {
                assert!(grad_norm < budget.grad_tol);
                assert!(residual > budget.residual_tol);
            }
            other => panic!("expected Stalled, got {other:?}"),
        }
    }

    #[test]
    fn certify_with_default_budget_classifies_obstruction_as_stalled() {
        // Regression: with a default `grad_tol` (1e-9) the harmonic
        // obstruction floor is reached via a step whose predicted energy
        // decrease is sub-ulp, so a naive post-step `e_after >= e_before`
        // check misclassifies it as EnergyRose instead of Stalled.
        let problem = cycle_problem_with_obstruction(1);
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![0.0, 0.0, 0.0],
            epoch: 1,
        };
        let budget = Budget::default();
        let err = gate.certify(&problem, candidate, &budget).unwrap_err();
        assert!(
            matches!(err, Reject::Stalled { .. }),
            "expected Stalled under default budget, got {err:?}"
        );
    }

    #[test]
    fn certify_rejects_pinned_node_that_already_moved() {
        let problem = ExplicitMatrixProblem::new(
            DMatrix::from_row_slice(2, 3, &[1.0, -1.0, 0.0, 0.0, 1.0, -1.0]),
            DVector::from_row_slice(&[1.0, 1.0]),
            DVector::from_row_slice(&[1.0, 1.0]),
            vec![Pin {
                index: 0,
                value: 5.0,
            }],
            vec![],
            1,
        )
        .unwrap();
        let gate = LaplacianHeatFlowGate;
        // Node 0 is pinned at 5.0 but the candidate submits 5.5: already
        // displaced before the gate ever ran a step.
        let candidate = CandidateState {
            s: vec![5.5, 0.0, 0.0],
            epoch: 1,
        };
        let budget = Budget::default();
        let err = gate.certify(&problem, candidate, &budget).unwrap_err();
        match err {
            Reject::PinnedMoved { index, .. } => assert_eq!(index, 0),
            other => panic!("expected PinnedMoved, got {other:?}"),
        }
    }

    #[test]
    fn certify_keeps_pinned_node_fixed_across_relaxation() {
        // Pin node 0 at its starting value; verify the accepted terminal
        // state left it untouched even after many relaxation steps.
        let problem = ExplicitMatrixProblem::new(
            DMatrix::from_row_slice(2, 3, &[1.0, -1.0, 0.0, 0.0, 1.0, -1.0]),
            DVector::from_row_slice(&[1.0, 1.0]),
            DVector::from_row_slice(&[1.0, 1.0]),
            vec![Pin {
                index: 0,
                value: 0.0,
            }],
            vec![],
            1,
        )
        .unwrap();
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![0.0, 0.0, 0.0],
            epoch: 1,
        };
        let accepted = gate
            .certify(&problem, candidate, &Budget::default())
            .unwrap();
        assert_eq!(accepted.state.s[0], 0.0);
    }

    #[test]
    fn certify_rejects_non_finite_candidate_state() {
        let problem = path_problem(1);
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![f64::NAN, 0.0, 0.0],
            epoch: 1,
        };
        let err = gate
            .certify(&problem, candidate, &Budget::default())
            .unwrap_err();
        match err {
            Reject::NonFiniteState { index, .. } => assert_eq!(index, 0),
            other => panic!("expected NonFiniteState, got {other:?}"),
        }

        let candidate_inf = CandidateState {
            s: vec![0.0, f64::INFINITY, 0.0],
            epoch: 1,
        };
        let err_inf = gate
            .certify(&problem, candidate_inf, &Budget::default())
            .unwrap_err();
        match err_inf {
            Reject::NonFiniteState { index, .. } => assert_eq!(index, 1),
            other => panic!("expected NonFiniteState, got {other:?}"),
        }
    }

    #[test]
    fn certify_rejects_epoch_mismatch() {
        let problem = path_problem(42);
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![0.0, 0.0, 0.0],
            epoch: 41,
        };
        let err = gate
            .certify(&problem, candidate, &Budget::default())
            .unwrap_err();
        match err {
            Reject::EpochMismatch {
                problem_epoch,
                candidate_epoch,
            } => {
                assert_eq!(problem_epoch, 42);
                assert_eq!(candidate_epoch, 41);
            }
            other => panic!("expected EpochMismatch, got {other:?}"),
        }
    }

    #[test]
    fn certify_rejects_fiber_dimension_mismatch() {
        let problem = path_problem(1);
        let gate = LaplacianHeatFlowGate;
        let candidate = CandidateState {
            s: vec![0.0, 0.0], // problem has 3 nodes, not 2
            epoch: 1,
        };
        let err = gate
            .certify(&problem, candidate, &Budget::default())
            .unwrap_err();
        match err {
            Reject::FiberMismatch { expected, actual } => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("expected FiberMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifold_guard_rejects_hyperbolic_domain_violation() {
        // 16-node-block problem: single isolated node (D has 1 zero-weight
        // row, w must stay positive so give it a self-loop-like edge to
        // itself does not apply; instead attach a trivial 2-node edge and
        // only guard node 0's 16-wide block).
        let mut row = vec![0.0; 32];
        row[0] = 1.0;
        row[16] = -1.0;
        let d = DMatrix::from_row_slice(1, 32, &row);
        let w = DVector::from_row_slice(&[1.0]);
        let b = DVector::from_row_slice(&[0.0]);
        let problem =
            ExplicitMatrixProblem::new(d, w, b, vec![], vec![ManifoldGuard { offset: 0 }], 1)
                .unwrap();
        let gate = LaplacianHeatFlowGate;
        let mut s = vec![0.0; 32];
        // Hyperbolic component with norm^2 >= 1 violates the boundary floor
        // `1 - ||x_H||^2 >= 1e-4` unconditionally.
        s[0] = 1.5;
        let candidate = CandidateState { s, epoch: 1 };
        let err = gate
            .certify(&problem, candidate, &Budget::default())
            .unwrap_err();
        match err {
            Reject::DomainViolation { offset, .. } => assert_eq!(offset, 0),
            other => panic!("expected DomainViolation, got {other:?}"),
        }
    }

    #[test]
    fn manifold_guard_rejects_spherical_cut_locus() {
        let mut row = vec![0.0; 32];
        row[0] = 1.0;
        row[16] = -1.0;
        let d = DMatrix::from_row_slice(1, 32, &row);
        let w = DVector::from_row_slice(&[1.0]);
        let b = DVector::from_row_slice(&[0.0]);
        let problem =
            ExplicitMatrixProblem::new(d, w, b, vec![], vec![ManifoldGuard { offset: 0 }], 1)
                .unwrap();
        let gate = LaplacianHeatFlowGate;
        let mut s = vec![0.0; 32];
        // Spherical block at the antipode of the north pole [1,0,0,0]: [-1,0,0,0].
        s[4] = -1.0;
        let candidate = CandidateState { s, epoch: 1 };
        let err = gate
            .certify(&problem, candidate, &Budget::default())
            .unwrap_err();
        match err {
            Reject::CutLocus { offset, .. } => assert_eq!(offset, 0),
            other => panic!("expected CutLocus, got {other:?}"),
        }
    }

    #[test]
    fn check_uncertain_energy_detects_a_deliberately_wrong_prediction() {
        // White-box test of the determinism cross-check itself: feed it a
        // predicted delta that does not match the direct one by more than
        // tolerance, and confirm it fails closed rather than accepting.
        let problem = path_problem(1);
        let s0 = DVector::from_row_slice(&[0.0, 0.0, 0.0]);
        let g = masked_gradient(&problem, &s0);
        let eta = 0.1;
        let e_before = energy(&problem, &s0);
        let s1 = &s0 - eta * &g;
        let e_after = energy(&problem, &s1);

        // Correct call passes.
        assert!(check_uncertain_energy(&problem, &g, eta, e_before, e_after, 1e-6).is_ok());

        // Corrupt e_after far beyond the exact Taylor identity's prediction.
        let corrupted_e_after = e_after + 10.0;
        let err = check_uncertain_energy(&problem, &g, eta, e_before, corrupted_e_after, 1e-6)
            .unwrap_err();
        assert!(matches!(err, Reject::UncertainEnergy { .. }));
    }

    #[test]
    fn safe_eta_bound_matches_two_over_lambda_max() {
        let problem = path_problem(1);
        // lambda_max(D^T W D) for this 3-node path graph is known: L =
        // [[1,-1,0],[-1,2,-1],[0,-1,1]], eigenvalues {0, 1, 3}. So
        // safe_eta_bound = 2/3.
        assert!((problem.safe_eta_bound() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn degenerate_laplacian_rejected_at_construction() {
        // All-zero D gives lambda_max = 0, an unusable step bound.
        let d = DMatrix::from_row_slice(1, 2, &[0.0, 0.0]);
        let w = DVector::from_row_slice(&[1.0]);
        let b = DVector::from_row_slice(&[0.0]);
        let err = ExplicitMatrixProblem::new(d, w, b, vec![], vec![], 1).unwrap_err();
        assert!(matches!(err, Reject::DomainViolation { .. }));
    }

    #[test]
    fn non_finite_pin_anchor_rejected_as_non_finite_not_fiber_mismatch() {
        let d = DMatrix::from_row_slice(2, 3, &[1.0, -1.0, 0.0, 0.0, 1.0, -1.0]);
        let w = DVector::from_row_slice(&[1.0, 1.0]);
        let b = DVector::from_row_slice(&[1.0, 1.0]);
        let err = ExplicitMatrixProblem::new(
            d,
            w,
            b,
            vec![Pin {
                index: 0,
                value: f64::NAN,
            }],
            vec![],
            1,
        )
        .unwrap_err();
        match err {
            Reject::NonFiniteState { index, .. } => assert_eq!(index, 0),
            other => panic!("expected NonFiniteState for a NaN pin anchor, got {other:?}"),
        }
    }
}
