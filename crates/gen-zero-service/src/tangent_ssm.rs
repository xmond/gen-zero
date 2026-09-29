//! Parallel tangent-space SSM (Spec 25, chapter 4 and section 5.4).
//!
//! A step is an affine map `F_t = (M_t, q_t)` on a tangent fiber:
//! `h_t = M_t h_{t-1} + q_t`. Steps compose as
//! `(M_2, q_2) o (M_1, q_1) = (M_2 M_1, M_2 q_1 + q_2)`, which is strictly
//! associative, so all prefixes `F_{1:t} = (P_t, b_t)` come out of a balanced
//! prefix-scan tree with O(T) compositions and O(log T) composition depth.
//!
//! Scope of this implementation (stage S2 of Spec 25 section 5.6):
//! - One fixed fiber per window. Transport between steps is the identity,
//!   so `M_t = exp(dt_t A)`. There is no curved geometry, atlas or
//!   `ProductGeometry` here; a window whose steps do not share the declared
//!   fiber is rejected, never silently re-based.
//! - ZOH (eq. 4.1) through the exponential of the augmented generator
//!   `dt [[A, B], [0, 0]]`. No `A^{-1}` anywhere. `A^2 = 0` takes an exact
//!   polynomial branch.
//! - Floating point is not associative: serial and tree results differ by
//!   rounding. The evidence records which backend actually ran.
//! - [`FiberCrossDiff`] (scheme 2) is the one place two fibers meet: it
//!   moves a passage vector into the question fiber by a caller-supplied
//!   gauge matrix and scans the cross-difference on the target fiber. The
//!   gauge is data here; its geometry lives with the caller.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

// One reject vocabulary for the whole crate: the mount module owns it.
pub use crate::mount::{Digest, Epochs, Reject, Version};
pub type Result<T> = std::result::Result<T, Reject>;

/// Identity of one tangent fiber: base point, frame, transport path and
/// versions. Two steps compose only when the target of the first equals the
/// source of the second.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiberId {
    pub patch: u64,
    pub base: Digest,
    pub frame: Digest,
    pub path: Digest,
    pub epochs: Epochs,
}

/// Dense row-major matrix with validated shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Box<[f64]>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(data.len()) {
            return Err(Reject::FiberMismatch);
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err(Reject::NonFiniteState);
        }
        Ok(Self {
            rows,
            cols,
            data: data.into_boxed_slice(),
        })
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self> {
        let r = rows.len();
        let c = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != c) {
            return Err(Reject::FiberMismatch);
        }
        Self::new(r, c, rows.concat())
    }

    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self {
            rows: n,
            cols: n,
            data: data.into_boxed_slice(),
        }
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols].into_boxed_slice(),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    fn is_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }

    fn is_exact_zero(&self) -> bool {
        self.data.iter().all(|&x| x == 0.0)
    }

    /// `self * rhs`; shape mismatch is a fiber mismatch.
    pub fn matmul(&self, rhs: &Matrix) -> Result<Matrix> {
        if self.cols != rhs.rows {
            return Err(Reject::FiberMismatch);
        }
        let mut out = Matrix::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k];
                if a == 0.0 {
                    continue;
                }
                let rrow = &rhs.data[k * rhs.cols..(k + 1) * rhs.cols];
                let orow = &mut out.data[i * rhs.cols..(i + 1) * rhs.cols];
                for (o, &b) in orow.iter_mut().zip(rrow) {
                    *o += a * b;
                }
            }
        }
        Ok(out)
    }

    pub fn matvec(&self, v: &[f64]) -> Result<Vec<f64>> {
        if self.cols != v.len() {
            return Err(Reject::FiberMismatch);
        }
        Ok((0..self.rows)
            .map(|i| {
                self.data[i * self.cols..(i + 1) * self.cols]
                    .iter()
                    .zip(v)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect())
    }

    fn scaled(&self, s: f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    /// Induced infinity norm (max absolute row sum). Submultiplicative, so
    /// `||M_t ... M_1|| <= prod ||M_s||` is a rigorous bound.
    pub fn norm_inf(&self) -> f64 {
        (0..self.rows)
            .map(|i| {
                self.data[i * self.cols..(i + 1) * self.cols]
                    .iter()
                    .map(|x| x.abs())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    }

    /// Induced 1-norm (max absolute column sum).
    fn norm_1(&self) -> f64 {
        (0..self.cols)
            .map(|j| (0..self.rows).map(|i| self.get(i, j).abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    /// Logarithmic norm for the infinity norm:
    /// `mu(A) = max_i (a_ii + sum_{j != i} |a_ij|)`, and
    /// `||exp(t A)||_inf <= exp(t mu(A))` for `t >= 0`. A positive value
    /// means the continuous generator can expand states in this norm.
    pub fn log_norm_inf(&self) -> Result<f64> {
        if self.rows != self.cols {
            return Err(Reject::FiberMismatch);
        }
        Ok((0..self.rows)
            .map(|i| {
                let off: f64 = (0..self.cols)
                    .filter(|&j| j != i)
                    .map(|j| self.get(i, j).abs())
                    .sum();
                self.get(i, i) + off
            })
            .fold(f64::NEG_INFINITY, f64::max))
    }
}

/// A tangent vector with the fiber it lives in.
#[derive(Clone, Debug, PartialEq)]
pub struct Tangent {
    fiber: FiberId,
    coords: Box<[f64]>,
}

impl Tangent {
    pub fn new(fiber: FiberId, coords: Vec<f64>) -> Result<Self> {
        if coords.is_empty() {
            return Err(Reject::FiberMismatch);
        }
        if coords.iter().any(|x| !x.is_finite()) {
            return Err(Reject::NonFiniteState);
        }
        Ok(Self {
            fiber,
            coords: coords.into_boxed_slice(),
        })
    }

    pub fn fiber(&self) -> &FiberId {
        &self.fiber
    }

    pub fn coords(&self) -> &[f64] {
        &self.coords
    }
}

/// Typed affine arrow `source -> target`: `h -> linear * h + bias`.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineStep {
    source: FiberId,
    target: FiberId,
    linear: Matrix,
    bias: Box<[f64]>,
}

impl AffineStep {
    pub fn new(source: FiberId, target: FiberId, linear: Matrix, bias: Vec<f64>) -> Result<Self> {
        if bias.len() != linear.rows {
            return Err(Reject::FiberMismatch);
        }
        if !linear.is_finite() || bias.iter().any(|x| !x.is_finite()) {
            return Err(Reject::NonFiniteState);
        }
        Ok(Self {
            source,
            target,
            linear,
            bias: bias.into_boxed_slice(),
        })
    }

    pub fn source(&self) -> &FiberId {
        &self.source
    }

    pub fn target(&self) -> &FiberId {
        &self.target
    }

    pub fn linear(&self) -> &Matrix {
        &self.linear
    }

    pub fn bias(&self) -> &[f64] {
        &self.bias
    }
}

/// `later o earlier = (A_2 A_1, A_2 b_1 + b_2)`.
///
/// `A_2 A_1` needs `cols(A_2) == rows(A_1)`; with typed fibers the target of
/// `earlier` must also be the source of `later`. Either failure is a
/// `FiberMismatch`. A non-finite product is a `NonFiniteState`.
pub fn compose_affine(earlier: &AffineStep, later: &AffineStep) -> Result<AffineStep> {
    if earlier.target != later.source || later.linear.cols != earlier.linear.rows {
        return Err(Reject::FiberMismatch);
    }
    let linear = later.linear.matmul(&earlier.linear)?;
    let mut bias = later.linear.matvec(&earlier.bias)?;
    for (b, q) in bias.iter_mut().zip(later.bias.iter()) {
        *b += q;
    }
    if !linear.is_finite() || bias.iter().any(|x| !x.is_finite()) {
        return Err(Reject::NonFiniteState);
    }
    Ok(AffineStep {
        source: earlier.source.clone(),
        target: later.target.clone(),
        linear,
        bias: bias.into_boxed_slice(),
    })
}

/// `h' = linear * h + bias`, with fiber and finiteness checks.
pub fn apply_affine(step: &AffineStep, initial: &Tangent) -> Result<Tangent> {
    if initial.fiber != step.source || initial.coords.len() != step.linear.cols {
        return Err(Reject::FiberMismatch);
    }
    let mut coords = step.linear.matvec(&initial.coords)?;
    for (h, q) in coords.iter_mut().zip(step.bias.iter()) {
        *h += q;
    }
    if coords.iter().any(|x| !x.is_finite()) {
        return Err(Reject::NonFiniteState);
    }
    Ok(Tangent {
        fiber: step.target.clone(),
        coords: coords.into_boxed_slice(),
    })
}

/// Which ZOH formula produced a step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZohBranch {
    /// `A^2 = 0` exactly: `Abar = I + dt A`, `Bbar = dt B + dt^2/2 A B`.
    /// The series terminates, so this is exact up to one rounding per entry.
    NilpotentPolynomial,
    /// Taylor series of the augmented generator with scaling and squaring.
    ScalingSquaring { squarings: u32 },
}

/// Keep only a small number of discretisations.  The cache is deliberately
/// bounded because a caller can supply arbitrary event intervals and model
/// matrices over the lifetime of a process.
const ZOH_CACHE_CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ZohCacheKey {
    model: [u8; 32],
    dt_bits: u64,
}

#[derive(Clone, Debug)]
struct CachedZoh {
    abar: Matrix,
    bbar: Matrix,
    branch: ZohBranch,
}

#[derive(Debug, Default)]
struct ZohCache {
    entries: VecDeque<(ZohCacheKey, Arc<CachedZoh>)>,
}

impl ZohCache {
    /// Look up an entry and promote it to the back of the bounded LRU queue.
    fn get(&mut self, key: ZohCacheKey) -> Option<Arc<CachedZoh>> {
        let position = self
            .entries
            .iter()
            .position(|(entry_key, _)| *entry_key == key)?;
        let (entry_key, value) = self.entries.remove(position)?;
        self.entries.push_back((entry_key, value.clone()));
        Some(value)
    }

    fn insert(&mut self, key: ZohCacheKey, value: Arc<CachedZoh>) {
        if let Some(position) = self
            .entries
            .iter()
            .position(|(entry_key, _)| *entry_key == key)
        {
            self.entries.remove(position);
        }
        while self.entries.len() >= ZOH_CACHE_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back((key, value));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn global_zoh_cache() -> &'static Mutex<ZohCache> {
    static CACHE: OnceLock<Mutex<ZohCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(ZohCache::default()))
}

fn with_global_zoh_cache<T>(f: impl FnOnce(&mut ZohCache) -> T) -> T {
    let mut cache = global_zoh_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&mut cache)
}

/// The ZOH result depends on the complete generator and input matrices.  A
/// content key lets the process-wide cache safely serve separate contexts,
/// including contexts whose matrices were cloned or later changed.
fn zoh_model_key(a: &Matrix, b: &Matrix) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"gen-zero/zoh-cache/model/v1\0");
    hasher.update(&(a.rows as u64).to_le_bytes());
    hasher.update(&(a.cols as u64).to_le_bytes());
    hash_f64s(&mut hasher, &a.data);
    hasher.update(&(b.rows as u64).to_le_bytes());
    hasher.update(&(b.cols as u64).to_le_bytes());
    hash_f64s(&mut hasher, &b.data);
    *hasher.finalize().as_bytes()
}

fn cached_zoh_discretize(
    a: &Matrix,
    b: &Matrix,
    model_key: [u8; 32],
    dt: f64,
) -> Result<(Matrix, Matrix, ZohBranch)> {
    // Preserve the uncached validation path for invalid values.  In
    // particular, NaN and negative values must never become cache entries.
    if !dt.is_finite() || dt < 0.0 {
        return zoh_discretize(a, b, dt);
    }

    let key = ZohCacheKey {
        model: model_key,
        dt_bits: dt.to_bits(),
    };
    if let Some(value) = with_global_zoh_cache(|cache| cache.get(key)) {
        return Ok((value.abar.clone(), value.bbar.clone(), value.branch));
    }

    // Do not hold the global lock while Taylor expansion and squaring run.
    // A concurrent miss may do the same work, but the second insertion is
    // deduplicated and all callers still receive the same validated result.
    let computed = zoh_discretize(a, b, dt)?;
    let value = Arc::new(CachedZoh {
        abar: computed.0.clone(),
        bbar: computed.1.clone(),
        branch: computed.2,
    });
    if let Some(value) = with_global_zoh_cache(|cache| {
        if let Some(existing) = cache.get(key) {
            Some(existing)
        } else {
            cache.insert(key, value);
            None
        }
    }) {
        return Ok((value.abar.clone(), value.bbar.clone(), value.branch));
    }
    Ok(computed)
}

/// Zero-order-hold discretisation (Spec 25 eq. 4.1):
/// `exp(dt [[A, B], [0, 0]]) = [[Abar, Bbar], [0, I]]`.
pub fn zoh_discretize(a: &Matrix, b: &Matrix, dt: f64) -> Result<(Matrix, Matrix, ZohBranch)> {
    let n = a.rows;
    if a.cols != n || b.rows != n {
        return Err(Reject::FiberMismatch);
    }
    if !dt.is_finite() || dt < 0.0 {
        return Err(Reject::DomainViolation);
    }
    let m = b.cols;

    if a.matmul(a)?.is_exact_zero() {
        let mut abar = Matrix::identity(n);
        for (x, y) in abar.data.iter_mut().zip(a.data.iter()) {
            *x += dt * y;
        }
        let ab = a.matmul(b)?;
        let bbar_data = b
            .data
            .iter()
            .zip(ab.data.iter())
            .map(|(bij, abij)| dt * bij + 0.5 * dt * dt * abij)
            .collect();
        let bbar = Matrix {
            rows: n,
            cols: m,
            data: bbar_data,
        };
        if !abar.is_finite() || !bbar.is_finite() {
            return Err(Reject::NonFiniteState);
        }
        return Ok((abar, bbar, ZohBranch::NilpotentPolynomial));
    }

    let d = n + m;
    let mut gen = Matrix::zeros(d, d);
    for i in 0..n {
        for j in 0..n {
            gen.data[i * d + j] = dt * a.get(i, j);
        }
        for j in 0..m {
            gen.data[i * d + n + j] = dt * b.get(i, j);
        }
    }
    let (e, squarings) = expm(&gen)?;
    let mut abar = Matrix::zeros(n, n);
    let mut bbar = Matrix::zeros(n, m);
    for i in 0..n {
        for j in 0..n {
            abar.data[i * n + j] = e.get(i, j);
        }
        for j in 0..m {
            bbar.data[i * m + j] = e.get(i, n + j);
        }
    }
    Ok((abar, bbar, ZohBranch::ScalingSquaring { squarings }))
}

/// Largest generator 1-norm the exponential accepts. Beyond it the result
/// overflows or loses all accuracy, so it is a domain violation, not a
/// number to pass on.
const EXPM_MAX_NORM: f64 = 700.0;

/// Matrix exponential: scale to norm <= 1/2, Taylor to machine precision,
/// square back. Terms that become exactly zero end the series early.
fn expm(x: &Matrix) -> Result<(Matrix, u32)> {
    let norm = x.norm_1();
    if !norm.is_finite() {
        return Err(Reject::NonFiniteState);
    }
    if norm > EXPM_MAX_NORM {
        return Err(Reject::DomainViolation);
    }
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32
    } else {
        0
    };
    let scaled = x.scaled(0.5f64.powi(squarings as i32));
    let n = x.rows;
    let mut sum = Matrix::identity(n);
    let mut term = Matrix::identity(n);
    for k in 1..=30u32 {
        term = term.matmul(&scaled)?.scaled(1.0 / f64::from(k));
        if term.is_exact_zero() {
            break;
        }
        for (s, t) in sum.data.iter_mut().zip(term.data.iter()) {
            *s += t;
        }
        if term.norm_1() <= f64::EPSILON * 1e-3 * sum.norm_1() {
            break;
        }
    }
    for _ in 0..squarings {
        sum = sum.matmul(&sum)?;
    }
    if !sum.is_finite() {
        return Err(Reject::NonFiniteState);
    }
    Ok((sum, squarings))
}

/// One observed event of the window.
#[derive(Clone, Debug)]
pub struct Event {
    pub id: Digest,
    pub time_ns: u64,
    pub input: Box<[f64]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// Reference left-to-right recurrence. A separate label: never counted
    /// as a parallel result.
    CpuSerial,
    /// Balanced prefix-scan tree on scoped OS threads.
    CpuParallel { threads: usize },
    /// Not built into this binary; always `BackendUnavailable`.
    Gpu { device: u32 },
}

impl Backend {
    pub fn label(self) -> String {
        match self {
            Self::CpuSerial => "cpu_serial".to_string(),
            Self::CpuParallel { threads } => format!("cpu_parallel(threads={threads})"),
            Self::Gpu { device } => format!("gpu(device={device})"),
        }
    }
}

/// Limits a window must satisfy before and after the scan.
#[derive(Clone, Debug)]
pub struct ScanBudget {
    /// Window length limit (`BudgetExceeded` above it).
    pub max_steps: usize,
    /// Bound on the certified growth `max_t prod_{s<=t} ||M_s||_inf`.
    /// A window whose operators can amplify states beyond it is rejected
    /// as divergent (`DomainViolation`).
    pub max_growth: f64,
    /// Bound on every state `||h_t||_inf` (`DomainViolation` above it).
    pub max_state_norm: f64,
}

/// Continuous-time model of one fixed fiber: `dh/dt = A h + B u`.
#[derive(Clone, Debug)]
pub struct FrozenContext {
    fiber: FiberId,
    a: Matrix,
    b: Matrix,
    window_start_ns: u64,
    budget: ScanBudget,
}

impl FrozenContext {
    pub fn new(
        fiber: FiberId,
        a: Matrix,
        b: Matrix,
        window_start_ns: u64,
        budget: ScanBudget,
    ) -> Result<Self> {
        if a.rows != a.cols || b.rows != a.rows {
            return Err(Reject::FiberMismatch);
        }
        if budget.max_steps == 0
            || !(budget.max_growth.is_finite() && budget.max_growth >= 1.0)
            || !(budget.max_state_norm.is_finite() && budget.max_state_norm > 0.0)
        {
            return Err(Reject::DomainViolation);
        }
        Ok(Self {
            fiber,
            a,
            b,
            window_start_ns,
            budget,
        })
    }

    pub fn fiber(&self) -> &FiberId {
        &self.fiber
    }
}

/// A window of steps ready to scan, with its stability certificate.
#[derive(Clone, Debug)]
pub struct PreparedWindow {
    steps: Box<[AffineStep]>,
    input_digest: Digest,
    fiber: FiberId,
    budget: ScanBudget,
    log_norm_inf: f64,
    growth_bound: f64,
    nilpotent_steps: usize,
    max_squarings: u32,
}

impl PreparedWindow {
    /// A window assembled from explicit steps (no ZOH). Steps must all be
    /// endomorphisms of `fiber`; the stability certificate is recomputed.
    pub fn from_steps(fiber: FiberId, steps: Vec<AffineStep>, budget: ScanBudget) -> Result<Self> {
        if steps.is_empty() {
            return Err(Reject::DomainViolation);
        }
        if steps.len() > budget.max_steps {
            return Err(Reject::BudgetExceeded);
        }
        if steps
            .iter()
            .any(|s| s.source != fiber || s.target != fiber || s.linear.rows != s.linear.cols)
        {
            return Err(Reject::FiberMismatch);
        }
        let growth_bound = certify_growth(&steps, &budget)?;
        let mut hasher = blake3::Hasher::new();
        for s in &steps {
            hash_step(&mut hasher, s);
        }
        Ok(Self {
            steps: steps.into_boxed_slice(),
            input_digest: *hasher.finalize().as_bytes(),
            fiber,
            budget,
            log_norm_inf: f64::NAN,
            growth_bound,
            nilpotent_steps: 0,
            max_squarings: 0,
        })
    }

    pub fn steps(&self) -> &[AffineStep] {
        &self.steps
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn input_digest(&self) -> &Digest {
        &self.input_digest
    }
}

/// `max_t prod_{s<=t} ||M_s||_inf`; rejects a window that may diverge.
fn certify_growth(steps: &[AffineStep], budget: &ScanBudget) -> Result<f64> {
    let mut running = 1.0f64;
    let mut worst = 1.0f64;
    for s in steps {
        running *= s.linear.norm_inf();
        if !running.is_finite() {
            return Err(Reject::NonFiniteState);
        }
        worst = worst.max(running);
        if worst > budget.max_growth {
            return Err(Reject::DomainViolation);
        }
    }
    Ok(worst)
}

fn hash_f64s(hasher: &mut blake3::Hasher, xs: &[f64]) {
    hasher.update(&(xs.len() as u64).to_le_bytes());
    for x in xs {
        hasher.update(&x.to_le_bytes());
    }
}

fn hash_step(hasher: &mut blake3::Hasher, s: &AffineStep) {
    hasher.update(&(s.linear.rows as u64).to_le_bytes());
    hash_f64s(hasher, &s.linear.data);
    hash_f64s(hasher, &s.bias);
}

/// What the scan did, for the request trace.
#[derive(Clone, Debug)]
pub struct ScanEvidence {
    pub input_digest: Digest,
    /// blake3 over every prefix `(P_t, b_t)` in time order.
    pub prefix_digest: Digest,
    pub steps: usize,
    pub elapsed_ns: u64,
    pub backend: Backend,
    /// Composition depth of the scan (T for serial, tree height for parallel).
    pub depth: usize,
    pub compositions: usize,
    pub growth_bound: f64,
    /// `mu_inf(A)` of the generator; NaN when the window came from explicit
    /// steps rather than a generator.
    pub log_norm_inf: f64,
    pub nilpotent_steps: usize,
    pub max_squarings: u32,
}

#[derive(Clone, Debug)]
pub struct ScanOutput {
    /// `h_1 ... h_T`.
    pub states: Vec<Tangent>,
    /// `F_{1:1} ... F_{1:T}`.
    pub prefixes: Vec<AffineStep>,
    pub evidence: ScanEvidence,
}

/// Spec 25 section 5.4 contract.
pub trait ParallelTangentSsm: Send + Sync {
    fn prepare(&self, events: &[Event], ctx: &FrozenContext) -> Result<PreparedWindow>;
    fn compose(&self, earlier: &AffineStep, later: &AffineStep) -> Result<AffineStep>;
    fn step(&self, step: &AffineStep, initial: &Tangent) -> Result<Tangent>;
    fn scan(
        &self,
        window: &PreparedWindow,
        initial: &Tangent,
        backend: Backend,
    ) -> Result<ScanOutput>;
}

/// Fixed-fiber ZOH engine: identity transport, exact affine prefix scan.
#[derive(Clone, Copy, Debug, Default)]
pub struct ZohTangentSsm;

impl ZohTangentSsm {
    pub fn new() -> Self {
        Self
    }
}

impl ParallelTangentSsm for ZohTangentSsm {
    /// `dt_t = (time_t - time_{t-1}) * 1e-9 s` with `time_0 = window_start_ns`.
    /// Times must not go backwards.
    fn prepare(&self, events: &[Event], ctx: &FrozenContext) -> Result<PreparedWindow> {
        if events.is_empty() {
            return Err(Reject::DomainViolation);
        }
        if events.len() > ctx.budget.max_steps {
            return Err(Reject::BudgetExceeded);
        }
        let log_norm_inf = ctx.a.log_norm_inf()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(&ctx.fiber.patch.to_le_bytes());
        hasher.update(&ctx.fiber.frame);
        hash_f64s(&mut hasher, &ctx.a.data);
        hash_f64s(&mut hasher, &ctx.b.data);
        hasher.update(&ctx.window_start_ns.to_le_bytes());
        let model_key = zoh_model_key(&ctx.a, &ctx.b);

        let mut steps = Vec::with_capacity(events.len());
        let mut prev = ctx.window_start_ns;
        let mut nilpotent_steps = 0;
        let mut max_squarings = 0;
        for ev in events {
            if ev.time_ns < prev {
                return Err(Reject::DomainViolation);
            }
            if ev.input.len() != ctx.b.cols {
                return Err(Reject::FiberMismatch);
            }
            if ev.input.iter().any(|x| !x.is_finite()) {
                return Err(Reject::NonFiniteState);
            }
            hasher.update(&ev.id);
            hasher.update(&ev.time_ns.to_le_bytes());
            hash_f64s(&mut hasher, &ev.input);

            let dt = (ev.time_ns - prev) as f64 * 1e-9;
            prev = ev.time_ns;
            let (abar, bbar, branch) = cached_zoh_discretize(&ctx.a, &ctx.b, model_key, dt)?;
            match branch {
                ZohBranch::NilpotentPolynomial => nilpotent_steps += 1,
                ZohBranch::ScalingSquaring { squarings } => {
                    max_squarings = max_squarings.max(squarings)
                }
            }
            let q = bbar.matvec(&ev.input)?;
            steps.push(AffineStep::new(
                ctx.fiber.clone(),
                ctx.fiber.clone(),
                abar,
                q,
            )?);
        }
        let growth_bound = certify_growth(&steps, &ctx.budget)?;
        Ok(PreparedWindow {
            steps: steps.into_boxed_slice(),
            input_digest: *hasher.finalize().as_bytes(),
            fiber: ctx.fiber.clone(),
            budget: ctx.budget.clone(),
            log_norm_inf,
            growth_bound,
            nilpotent_steps,
            max_squarings,
        })
    }

    fn compose(&self, earlier: &AffineStep, later: &AffineStep) -> Result<AffineStep> {
        compose_affine(earlier, later)
    }

    fn step(&self, step: &AffineStep, initial: &Tangent) -> Result<Tangent> {
        apply_affine(step, initial)
    }

    fn scan(
        &self,
        window: &PreparedWindow,
        initial: &Tangent,
        backend: Backend,
    ) -> Result<ScanOutput> {
        if window.steps.is_empty() {
            return Err(Reject::DomainViolation);
        }
        if initial.fiber != window.fiber || initial.coords.len() != window.steps[0].linear.cols {
            return Err(Reject::FiberMismatch);
        }
        let started = Instant::now();
        let (prefixes, depth, compositions) = match backend {
            Backend::CpuSerial => serial_prefixes(&window.steps)?,
            Backend::CpuParallel { threads } => {
                if threads == 0 {
                    return Err(Reject::BackendUnavailable);
                }
                tree_prefixes(&window.steps, threads)?
            }
            Backend::Gpu { .. } => return Err(Reject::BackendUnavailable),
        };

        let mut states = Vec::with_capacity(prefixes.len());
        let mut hasher = blake3::Hasher::new();
        for p in &prefixes {
            hash_step(&mut hasher, p);
            let h = apply_affine(p, initial)?;
            let norm = h.coords.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            if norm > window.budget.max_state_norm {
                return Err(Reject::DomainViolation);
            }
            states.push(h);
        }
        Ok(ScanOutput {
            states,
            evidence: ScanEvidence {
                input_digest: window.input_digest,
                prefix_digest: *hasher.finalize().as_bytes(),
                steps: prefixes.len(),
                elapsed_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                backend,
                depth,
                compositions,
                growth_bound: window.growth_bound,
                log_norm_inf: window.log_norm_inf,
                nilpotent_steps: window.nilpotent_steps,
                max_squarings: window.max_squarings,
            },
            prefixes,
        })
    }
}

/// Two-channel fiber cross-difference (scheme 2).
///
/// A passage channel lives in fiber `source` (the tangent space at `p`), a
/// question channel in fiber `target` (the tangent space at `q`). The gauge
/// `Gamma: source -> target` is the connection matrix that parallel-moves a
/// passage vector into the question fiber. The cross-difference
/// `v_delta = v_q - Gamma v_p` is what the question adds on top of the
/// passage once both are expressed in one frame: a component shared by both
/// channels (the background) cancels.
///
/// This type is pure typed algebra: it does not know where the gauge came
/// from. The caller derives it from a geometry (see
/// `CognitiveRuntime::evaluate_entailment`); nothing here is trained.
#[derive(Clone, Debug)]
pub struct FiberCrossDiff {
    source: FiberId,
    target: FiberId,
    gauge: Matrix,
}

/// The cross-difference inputs and the scan they drove.
#[derive(Clone, Debug)]
pub struct CrossDiffScan {
    /// `v_delta_t` for every paired event, in the target fiber.
    pub diffs: Vec<Tangent>,
    /// Scan of `h_t = Abar_t h_{t-1} + Bbar_t v_delta_t` from `h_0 = 0`.
    pub output: ScanOutput,
}

impl FiberCrossDiff {
    /// `gauge` maps `source` coordinates (its columns) to `target`
    /// coordinates (its rows). A non-finite gauge is not a connection:
    /// `DomainViolation`. (`Matrix::new` already refuses non-finite data as
    /// `NonFiniteState`; this re-check holds for any `Matrix` passed in.)
    /// Parallel transport is an isomorphism between fibers of one width, so
    /// the gauge must be square and non-empty: anything else is
    /// `FiberMismatch`. (`Matrix::identity(0)` bypasses `Matrix::new`.)
    pub fn new(source: FiberId, target: FiberId, gauge: Matrix) -> Result<Self> {
        if !gauge.is_finite() {
            return Err(Reject::DomainViolation);
        }
        if gauge.rows == 0 || gauge.rows != gauge.cols {
            return Err(Reject::FiberMismatch);
        }
        Ok(Self {
            source,
            target,
            gauge,
        })
    }

    /// Build from raw rows. Finiteness is checked first (`DomainViolation`),
    /// then shape (`FiberMismatch` for empty, ragged or non-square rows).
    pub fn from_rows(source: FiberId, target: FiberId, rows: &[Vec<f64>]) -> Result<Self> {
        if rows.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Reject::DomainViolation);
        }
        Self::new(source, target, Matrix::from_rows(rows)?)
    }

    /// `Gamma v_p`: the passage vector moved into the target fiber.
    pub fn transport(&self, v_p: &Tangent) -> Result<Tangent> {
        if v_p.fiber != self.source || v_p.coords.len() != self.gauge.cols {
            return Err(Reject::FiberMismatch);
        }
        let coords = self.gauge.matvec(&v_p.coords)?;
        if coords.iter().any(|x| !x.is_finite()) {
            return Err(Reject::NonFiniteState);
        }
        Ok(Tangent {
            fiber: self.target.clone(),
            coords: coords.into_boxed_slice(),
        })
    }

    /// `v_delta = v_q - Gamma v_p`, in the target fiber.
    pub fn cross_diff(&self, v_p: &Tangent, v_q: &Tangent) -> Result<Tangent> {
        if v_q.fiber != self.target || v_q.coords.len() != self.gauge.rows {
            return Err(Reject::FiberMismatch);
        }
        let moved = self.transport(v_p)?;
        let coords: Vec<f64> = v_q
            .coords
            .iter()
            .zip(moved.coords.iter())
            .map(|(q, g)| q - g)
            .collect();
        if coords.iter().any(|x| !x.is_finite()) {
            return Err(Reject::NonFiniteState);
        }
        Ok(Tangent {
            fiber: self.target.clone(),
            coords: coords.into_boxed_slice(),
        })
    }

    /// Pair the two event streams index by index and replace each pair by one
    /// event whose input is `v_delta_t`. The streams must have the same length
    /// (`FiberMismatch`) and the same timestamp at every index
    /// (`DomainViolation`): nothing is interpolated or re-aligned.
    pub fn diff_events(
        &self,
        passage: &[Event],
        question: &[Event],
    ) -> Result<(Vec<Event>, Vec<Tangent>)> {
        if passage.is_empty() {
            return Err(Reject::DomainViolation);
        }
        if passage.len() != question.len() {
            return Err(Reject::FiberMismatch);
        }
        let mut events = Vec::with_capacity(passage.len());
        let mut diffs = Vec::with_capacity(passage.len());
        for (ep, eq) in passage.iter().zip(question) {
            if ep.time_ns != eq.time_ns {
                return Err(Reject::DomainViolation);
            }
            let v_p = Tangent::new(self.source.clone(), ep.input.to_vec())?;
            let v_q = Tangent::new(self.target.clone(), eq.input.to_vec())?;
            let d = self.cross_diff(&v_p, &v_q)?;
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"gen-zero/fiber-cross-diff/event/v1\0");
            hasher.update(&ep.id);
            hasher.update(&eq.id);
            events.push(Event {
                id: *hasher.finalize().as_bytes(),
                time_ns: ep.time_ns,
                input: d.coords.clone(),
            });
            diffs.push(d);
        }
        Ok((events, diffs))
    }

    /// Turn the paired streams into ZOH steps on the target fiber and run the
    /// prefix scan from the ground state `h_0 = 0`. `ctx` must model the
    /// target fiber, and its input width must be the target width.
    pub fn scan(
        &self,
        ssm: &dyn ParallelTangentSsm,
        ctx: &FrozenContext,
        passage: &[Event],
        question: &[Event],
        backend: Backend,
    ) -> Result<CrossDiffScan> {
        if ctx.fiber != self.target || ctx.b.cols != self.gauge.rows {
            return Err(Reject::FiberMismatch);
        }
        let (events, diffs) = self.diff_events(passage, question)?;
        let window = ssm.prepare(&events, ctx)?;
        let h0 = Tangent::new(self.target.clone(), vec![0.0; ctx.a.rows])?;
        let output = ssm.scan(&window, &h0, backend)?;
        Ok(CrossDiffScan { diffs, output })
    }
}

type Prefixes = (Vec<AffineStep>, usize, usize);

/// Reference: `F_{1:t} = F_t o F_{1:t-1}`, left to right.
pub fn serial_prefixes(steps: &[AffineStep]) -> Result<Prefixes> {
    let mut out: Vec<AffineStep> = Vec::with_capacity(steps.len());
    for s in steps {
        let next = match out.last() {
            Some(prev) => compose_affine(prev, s)?,
            None => s.clone(),
        };
        out.push(next);
    }
    let comps = steps.len().saturating_sub(1);
    Ok((out, comps, comps))
}

/// Up-sweep node: the composed map of a contiguous segment.
struct Node {
    total: AffineStep,
    children: Option<Box<(Node, Node)>>,
}

impl Node {
    fn height(&self) -> usize {
        self.children
            .as_ref()
            .map_or(0, |c| 1 + c.0.height().max(c.1.height()))
    }
}

/// Segments shorter than this are not worth a thread.
const MIN_PARALLEL_SEGMENT: usize = 8;

/// Balanced prefix scan (Blelloch up-sweep / down-sweep over a tree split
/// at `len / 2`, so any length works without padding).
///
/// Work: `T-1` compositions up, at most `T-1` down plus at most `T` at the
/// leaves, so O(T). Depth: two passes over a tree of height `ceil(log2 T)`.
/// Segment order is never swapped. `threads` caps the scoped threads used
/// at the top of the tree.
pub fn tree_prefixes(steps: &[AffineStep], threads: usize) -> Result<Prefixes> {
    if steps.is_empty() {
        return Err(Reject::DomainViolation);
    }
    let spawn_depth = usize::BITS - threads.max(1).leading_zeros() - 1;
    let spawn_depth = spawn_depth as usize;
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let root = up_sweep(steps, spawn_depth, &counter)?;
    let mut out: Vec<Option<AffineStep>> = vec![None; steps.len()];
    down_sweep(&root, steps, None, &mut out, spawn_depth, &counter)?;
    let height = root.height();
    let prefixes = out
        .into_iter()
        .map(|p| p.ok_or(Reject::NonFiniteState))
        .collect::<Result<Vec<_>>>()?;
    let comps = counter.load(std::sync::atomic::Ordering::Relaxed);
    // Up-sweep and down-sweep each walk the tree once, plus the leaf layer.
    Ok((prefixes, 2 * height + 1, comps))
}

fn counted_compose(
    earlier: &AffineStep,
    later: &AffineStep,
    counter: &std::sync::atomic::AtomicUsize,
) -> Result<AffineStep> {
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    compose_affine(earlier, later)
}

fn up_sweep(
    steps: &[AffineStep],
    spawn_depth: usize,
    counter: &std::sync::atomic::AtomicUsize,
) -> Result<Node> {
    if steps.len() == 1 {
        return Ok(Node {
            total: steps[0].clone(),
            children: None,
        });
    }
    let (l, r) = steps.split_at(steps.len() / 2);
    let (left, right) = if spawn_depth > 0 && steps.len() >= MIN_PARALLEL_SEGMENT {
        std::thread::scope(|s| {
            let handle = s.spawn(|| up_sweep(r, spawn_depth - 1, counter));
            let left = up_sweep(l, spawn_depth - 1, counter);
            let right = handle.join().map_err(|_| Reject::BackendUnavailable)?;
            Ok::<_, Reject>((left?, right?))
        })?
    } else {
        (up_sweep(l, 0, counter)?, up_sweep(r, 0, counter)?)
    };
    let total = counted_compose(&left.total, &right.total, counter)?;
    Ok(Node {
        total,
        children: Some(Box::new((left, right))),
    })
}

/// `prefix` is the composed map of everything before this segment (`None`
/// at the left edge, which stands for the identity of the correct fiber).
fn down_sweep(
    node: &Node,
    steps: &[AffineStep],
    prefix: Option<&AffineStep>,
    out: &mut [Option<AffineStep>],
    spawn_depth: usize,
    counter: &std::sync::atomic::AtomicUsize,
) -> Result<()> {
    let Some(children) = node.children.as_ref() else {
        out[0] = Some(match prefix {
            Some(p) => counted_compose(p, &steps[0], counter)?,
            None => steps[0].clone(),
        });
        return Ok(());
    };
    let (left, right) = (&children.0, &children.1);
    let mid = steps.len() / 2;
    let right_prefix = match prefix {
        Some(p) => counted_compose(p, &left.total, counter)?,
        None => left.total.clone(),
    };
    let (ls, rs) = steps.split_at(mid);
    let (lo, ro) = out.split_at_mut(mid);
    if spawn_depth > 0 && steps.len() >= MIN_PARALLEL_SEGMENT {
        std::thread::scope(|s| {
            let handle = s
                .spawn(|| down_sweep(right, rs, Some(&right_prefix), ro, spawn_depth - 1, counter));
            let left_res = down_sweep(left, ls, prefix, lo, spawn_depth - 1, counter);
            let right_res = handle.join().map_err(|_| Reject::BackendUnavailable)?;
            left_res.and(right_res)
        })
    } else {
        down_sweep(left, ls, prefix, lo, 0, counter)?;
        down_sweep(right, rs, Some(&right_prefix), ro, 0, counter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epochs() -> Epochs {
        Epochs {
            version: Version(1),
            model: [1; 32],
            geometry: [2; 32],
            atlas: [3; 32],
            graph: [4; 32],
            policy: [5; 32],
        }
    }

    fn fiber(patch: u64) -> FiberId {
        FiberId {
            patch,
            base: [0; 32],
            frame: [0; 32],
            path: [0; 32],
            epochs: epochs(),
        }
    }

    fn budget() -> ScanBudget {
        ScanBudget {
            max_steps: 4096,
            max_growth: 1e6,
            max_state_norm: 1e9,
        }
    }

    /// Deterministic generator (SplitMix64) so no test depends on a seed
    /// crate or wall clock.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    /// Random step on `fiber(0)` with `||M||_inf <= 0.95` (well-conditioned,
    /// bounded family).
    fn random_step(rng: &mut Rng, n: usize) -> AffineStep {
        let mut data: Vec<f64> = (0..n * n).map(|_| rng.next()).collect();
        let m = Matrix::new(n, n, data.clone()).unwrap();
        let scale = 0.95 / m.norm_inf();
        data.iter_mut().for_each(|x| *x *= scale);
        let bias = (0..n).map(|_| rng.next()).collect();
        AffineStep::new(fiber(0), fiber(0), Matrix::new(n, n, data).unwrap(), bias).unwrap()
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    fn step_diff(a: &AffineStep, b: &AffineStep) -> f64 {
        max_diff(a.linear.as_slice(), b.linear.as_slice()).max(max_diff(&a.bias, &b.bias))
    }

    #[test]
    fn compose_matches_closed_form() {
        let a1 = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let a2 = Matrix::from_rows(&[vec![0.0, 1.0], vec![-1.0, 0.5]]).unwrap();
        let f1 = AffineStep::new(fiber(0), fiber(0), a1, vec![1.0, -1.0]).unwrap();
        let f2 = AffineStep::new(fiber(0), fiber(0), a2, vec![0.5, 2.0]).unwrap();
        let c = compose_affine(&f1, &f2).unwrap();
        // A2 A1 = [[3, 4], [0.5, 0]], A2 b1 + b2 = [-1 + 0.5, -1.5 + 2].
        assert_eq!(c.linear.as_slice(), &[3.0, 4.0, 0.5, 0.0]);
        assert_eq!(c.bias(), &[-0.5, 0.5]);
    }

    #[test]
    fn affine_monoid_is_associative() {
        let mut rng = Rng(250925);
        for n in [1usize, 2, 3, 5, 8] {
            let f1 = random_step(&mut rng, n);
            let f2 = random_step(&mut rng, n);
            let f3 = random_step(&mut rng, n);
            // (f3 o f2) o f1 versus f3 o (f2 o f1).
            let left = compose_affine(&f1, &compose_affine(&f2, &f3).unwrap()).unwrap();
            let right = compose_affine(&compose_affine(&f1, &f2).unwrap(), &f3).unwrap();
            let d = step_diff(&left, &right);
            assert!(d <= 1e-12, "n={n} associativity error {d:e}");
        }
    }

    #[test]
    fn associativity_holds_across_rectangular_fibers() {
        // R^2 -> R^3 -> R^4 -> R^1 with distinct fiber identities.
        let mut rng = Rng(7);
        let mut mk = |src: u64, dst: u64, rows: usize, cols: usize| {
            let data = (0..rows * cols).map(|_| rng.next()).collect();
            let bias = (0..rows).map(|_| rng.next()).collect();
            AffineStep::new(
                fiber(src),
                fiber(dst),
                Matrix::new(rows, cols, data).unwrap(),
                bias,
            )
            .unwrap()
        };
        let f1 = mk(0, 1, 3, 2);
        let f2 = mk(1, 2, 4, 3);
        let f3 = mk(2, 3, 1, 4);
        let left = compose_affine(&f1, &compose_affine(&f2, &f3).unwrap()).unwrap();
        let right = compose_affine(&compose_affine(&f1, &f2).unwrap(), &f3).unwrap();
        assert_eq!(left.linear.rows(), 1);
        assert_eq!(left.linear.cols(), 2);
        assert_eq!(left.source(), &fiber(0));
        assert_eq!(left.target(), &fiber(3));
        assert!(step_diff(&left, &right) <= 1e-12);
    }

    #[test]
    fn compose_rejects_dimension_mismatch_fail_closed() {
        // earlier: R^2 -> R^3 (3x2). later expects R^2 input (4x2): cols(A2)=2
        // != rows(A1)=3, even though both fibers match by identity.
        let earlier = AffineStep::new(
            fiber(0),
            fiber(1),
            Matrix::new(3, 2, vec![1.0; 6]).unwrap(),
            vec![0.0; 3],
        )
        .unwrap();
        let later = AffineStep::new(
            fiber(1),
            fiber(2),
            Matrix::new(4, 2, vec![1.0; 8]).unwrap(),
            vec![0.0; 4],
        )
        .unwrap();
        assert_eq!(compose_affine(&earlier, &later), Err(Reject::FiberMismatch));
    }

    #[test]
    fn compose_rejects_wrong_fiber_even_with_matching_shapes() {
        let f1 = AffineStep::new(fiber(0), fiber(1), Matrix::identity(2), vec![0.0; 2]).unwrap();
        let f2 = AffineStep::new(fiber(9), fiber(2), Matrix::identity(2), vec![0.0; 2]).unwrap();
        assert_eq!(compose_affine(&f1, &f2), Err(Reject::FiberMismatch));
        let h = Tangent::new(fiber(5), vec![1.0, 2.0]).unwrap();
        assert_eq!(apply_affine(&f1, &h), Err(Reject::FiberMismatch));
    }

    #[test]
    fn compose_overflow_is_non_finite_state() {
        let big = AffineStep::new(
            fiber(0),
            fiber(0),
            Matrix::new(1, 1, vec![1e200]).unwrap(),
            vec![0.0],
        )
        .unwrap();
        assert_eq!(compose_affine(&big, &big), Err(Reject::NonFiniteState));
    }

    #[test]
    fn serial_and_parallel_scan_agree_for_many_lengths() {
        let ssm = ZohTangentSsm::new();
        let n = 4;
        for &t in &[1usize, 2, 3, 17, 64, 129] {
            let mut rng = Rng(1000 + t as u64);
            let steps: Vec<AffineStep> = (0..t).map(|_| random_step(&mut rng, n)).collect();
            let window = PreparedWindow::from_steps(fiber(0), steps.clone(), budget()).unwrap();
            let h0 = Tangent::new(fiber(0), (0..n).map(|_| rng.next()).collect()).unwrap();

            let serial = ssm.scan(&window, &h0, Backend::CpuSerial).unwrap();
            // The recurrence one step at a time, independent of any prefix.
            let mut h = h0.clone();
            for (i, s) in steps.iter().enumerate() {
                h = ssm.step(s, &h).unwrap();
                assert!(max_diff(h.coords(), serial.states[i].coords()) <= 1e-12);
            }
            for threads in [1usize, 2, 4, 8] {
                let par = ssm
                    .scan(&window, &h0, Backend::CpuParallel { threads })
                    .unwrap();
                assert_eq!(par.states.len(), t);
                for i in 0..t {
                    let dp = step_diff(&serial.prefixes[i], &par.prefixes[i]);
                    let ds = max_diff(serial.states[i].coords(), par.states[i].coords());
                    assert!(
                        dp <= 1e-10 && ds <= 1e-10,
                        "T={t} threads={threads} i={i}: prefix {dp:e} state {ds:e}"
                    );
                }
                // O(T) work: at most 3T compositions, never T log T.
                assert!(par.evidence.compositions <= 3 * t, "T={t}");
                // O(log T) depth.
                let log2 = usize::BITS as usize - t.leading_zeros() as usize;
                assert!(par.evidence.depth <= 2 * log2 + 1, "T={t}");
                assert_eq!(par.evidence.backend, Backend::CpuParallel { threads });
                assert_eq!(par.evidence.input_digest, serial.evidence.input_digest);
            }
            assert_eq!(serial.evidence.backend, Backend::CpuSerial);
        }
    }

    #[test]
    fn gpu_and_zero_thread_backends_are_rejected_not_rerouted() {
        let ssm = ZohTangentSsm::new();
        let mut rng = Rng(3);
        let window =
            PreparedWindow::from_steps(fiber(0), vec![random_step(&mut rng, 2)], budget()).unwrap();
        let h0 = Tangent::new(fiber(0), vec![0.0, 1.0]).unwrap();
        assert_eq!(
            ssm.scan(&window, &h0, Backend::Gpu { device: 0 })
                .unwrap_err(),
            Reject::BackendUnavailable
        );
        assert_eq!(
            ssm.scan(&window, &h0, Backend::CpuParallel { threads: 0 })
                .unwrap_err(),
            Reject::BackendUnavailable
        );
        let wrong = Tangent::new(fiber(1), vec![0.0, 1.0]).unwrap();
        assert_eq!(
            ssm.scan(&window, &wrong, Backend::CpuSerial).unwrap_err(),
            Reject::FiberMismatch
        );
    }

    #[test]
    fn zoh_nilpotent_generator_is_exact() {
        // Double integrator: A = [[0, 1], [0, 0]], A^2 = 0.
        let a = Matrix::from_rows(&[vec![0.0, 1.0], vec![0.0, 0.0]]).unwrap();
        let b = Matrix::from_rows(&[vec![0.0], vec![1.0]]).unwrap();
        let dt = 0.25;
        let (abar, bbar, branch) = zoh_discretize(&a, &b, dt).unwrap();
        assert_eq!(branch, ZohBranch::NilpotentPolynomial);
        assert_eq!(abar.as_slice(), &[1.0, dt, 0.0, 1.0]);
        assert_eq!(bbar.as_slice(), &[dt * dt / 2.0, dt]);

        // A = 0: Abar = I, Bbar = dt B, no inverse involved.
        let zero = Matrix::new(2, 2, vec![0.0; 4]).unwrap();
        let (abar, bbar, branch) = zoh_discretize(&zero, &b, dt).unwrap();
        assert_eq!(branch, ZohBranch::NilpotentPolynomial);
        assert_eq!(abar, Matrix::identity(2));
        assert_eq!(bbar.as_slice(), &[0.0, dt]);
    }

    #[test]
    fn zoh_cache_reuses_dt_and_tracks_model_contents() {
        let a = Matrix::from_rows(&[vec![-0.75, 0.25], vec![0.0, -0.5]]).unwrap();
        let b = Matrix::from_rows(&[vec![1.0], vec![0.5]]).unwrap();
        let model_key = zoh_model_key(&a, &b);
        let dt = 0.125;

        let first = cached_zoh_discretize(&a, &b, model_key, dt).unwrap();
        let second = cached_zoh_discretize(&a, &b, model_key, dt).unwrap();
        assert_eq!(first, second);

        // The exact bit key distinguishes intervals, even when they are very
        // close numerically. A changed model gets a different cache namespace
        // and therefore cannot reuse the old Abar/Bbar pair.
        let mut changed_a = a.clone();
        changed_a.data[0] -= 0.125;
        let changed_key = zoh_model_key(&changed_a, &b);
        assert_ne!(model_key, changed_key);
        let changed = cached_zoh_discretize(&changed_a, &b, changed_key, dt).unwrap();
        let expected = zoh_discretize(&changed_a, &b, dt).unwrap();
        assert_eq!(changed, expected);

        let nearby_dt = f64::from_bits(dt.to_bits() + 1);
        assert_ne!(dt.to_bits(), nearby_dt.to_bits());
        let nearby = cached_zoh_discretize(&a, &b, model_key, nearby_dt).unwrap();
        let nearby_expected = zoh_discretize(&a, &b, nearby_dt).unwrap();
        assert_eq!(nearby, nearby_expected);
    }

    #[test]
    fn zoh_cache_is_bounded_and_evicts_old_entries() {
        let model = [7; 32];
        let mut cache = ZohCache::default();
        let value = Arc::new(CachedZoh {
            abar: Matrix::identity(1),
            bbar: Matrix::identity(1),
            branch: ZohBranch::NilpotentPolynomial,
        });
        let first_key = ZohCacheKey {
            model,
            dt_bits: 0.0f64.to_bits(),
        };
        cache.insert(first_key, value.clone());
        assert_eq!(cache.get(first_key).unwrap().abar, value.abar);

        for i in 1..=ZOH_CACHE_CAPACITY {
            cache.insert(
                ZohCacheKey {
                    model,
                    dt_bits: (i as f64).to_bits(),
                },
                value.clone(),
            );
        }
        assert_eq!(cache.len(), ZOH_CACHE_CAPACITY);
        assert!(cache.get(first_key).is_none());
        assert!(cache
            .get(ZohCacheKey {
                model,
                dt_bits: (ZOH_CACHE_CAPACITY as f64).to_bits(),
            })
            .is_some());
    }

    #[test]
    fn zoh_continuous_limit_approaches_the_nilpotent_branch() {
        // A_eps = [[-eps, 1], [0, -eps]] has closed form
        // exp(dt A) = e^{-eps dt} [[1, dt], [0, 1]] and
        // Bbar = int_0^dt exp(s A) B ds for B = [0, 1]^T:
        //   Bbar_0 = (1 - e^{-eps dt}(1 + eps dt)) / eps^2
        //   Bbar_1 = (1 - e^{-eps dt}) / eps
        // As eps -> 0 these tend to the exact nilpotent result.
        let b = Matrix::from_rows(&[vec![0.0], vec![1.0]]).unwrap();
        let dt = 0.5;
        let nil = Matrix::from_rows(&[vec![0.0, 1.0], vec![0.0, 0.0]]).unwrap();
        let (abar0, bbar0, _) = zoh_discretize(&nil, &b, dt).unwrap();
        let mut prev_gap = f64::INFINITY;
        for &eps in &[1e-1, 1e-2, 1e-3, 1e-4, 1e-6] {
            let a = Matrix::from_rows(&[vec![-eps, 1.0], vec![0.0, -eps]]).unwrap();
            let (abar, bbar, branch) = zoh_discretize(&a, &b, dt).unwrap();
            assert!(matches!(branch, ZohBranch::ScalingSquaring { .. }));
            let e = (-eps * dt).exp();
            let exact_a = [e, e * dt, 0.0, e];
            // Power series of the two integrals (eps * dt <= 0.05), which
            // avoids the cancellation of the closed forms at small eps:
            //   b0 = sum_k (-eps)^k dt^{k+2} / (k! (k+2))
            //   b1 = sum_k (-eps)^k dt^{k+1} / (k+1)!
            let (mut exact_b0, mut exact_b1) = (0.0f64, 0.0f64);
            let mut c = 1.0f64; // (-eps)^k dt^k / k!
            for k in 0..40 {
                exact_b0 += c * dt * dt / (k as f64 + 2.0);
                exact_b1 += c * dt / (k as f64 + 1.0);
                c *= -eps * dt / (k as f64 + 1.0);
            }
            assert!(max_diff(abar.as_slice(), &exact_a) <= 1e-14, "eps={eps}");
            assert!(
                max_diff(bbar.as_slice(), &[exact_b0, exact_b1]) <= 1e-10,
                "eps={eps}: {:?} vs {:?}",
                bbar.as_slice(),
                [exact_b0, exact_b1]
            );
            let gap = max_diff(abar.as_slice(), abar0.as_slice())
                .max(max_diff(bbar.as_slice(), bbar0.as_slice()));
            assert!(gap < prev_gap, "eps={eps}: gap {gap:e} did not shrink");
            prev_gap = gap;
        }
        assert!(prev_gap <= 1e-6, "limit gap {prev_gap:e}");
    }

    #[test]
    fn zoh_matches_scalar_closed_form() {
        let dt = 0.3;
        for &a in &[-2.0, -0.5, 0.7] {
            let am = Matrix::new(1, 1, vec![a]).unwrap();
            let bm = Matrix::new(1, 1, vec![1.5]).unwrap();
            let (abar, bbar, _) = zoh_discretize(&am, &bm, dt).unwrap();
            assert!((abar.get(0, 0) - (a * dt).exp()).abs() <= 1e-14);
            assert!((bbar.get(0, 0) - 1.5 * (a * dt).exp_m1() / a).abs() <= 1e-14);
        }
    }

    fn ctx_with(a: Matrix, b: Matrix, budget: ScanBudget) -> FrozenContext {
        FrozenContext::new(fiber(0), a, b, 0, budget).unwrap()
    }

    fn event(t: u64, u: f64) -> Event {
        Event {
            id: [0; 32],
            time_ns: t,
            input: vec![u].into_boxed_slice(),
        }
    }

    #[test]
    fn prepare_then_scan_follows_the_continuous_system() {
        // Stable 2x2 system; compare the scanned state with the exact
        // solution of dh/dt = A h + B u for piecewise-constant u.
        let a = Matrix::from_rows(&[vec![-1.0, 0.5], vec![-0.5, -1.0]]).unwrap();
        let b = Matrix::from_rows(&[vec![1.0], vec![0.0]]).unwrap();
        let ctx = ctx_with(a.clone(), b.clone(), budget());
        let events: Vec<Event> = (1..=64)
            .map(|k| event(k * 10_000_000, (k as f64 * 0.37).sin()))
            .collect();
        let ssm = ZohTangentSsm::new();
        let window = ssm.prepare(&events, &ctx).unwrap();
        assert!(window.log_norm_inf < 0.0);
        let h0 = Tangent::new(fiber(0), vec![1.0, -1.0]).unwrap();
        let par = ssm
            .scan(&window, &h0, Backend::CpuParallel { threads: 4 })
            .unwrap();
        let ser = ssm.scan(&window, &h0, Backend::CpuSerial).unwrap();
        // Independent reference: fine forward integration (RK4) of the ODE.
        let mut h = [1.0f64, -1.0];
        let f = |h: &[f64; 2], u: f64| {
            [
                a.get(0, 0) * h[0] + a.get(0, 1) * h[1] + b.get(0, 0) * u,
                a.get(1, 0) * h[0] + a.get(1, 1) * h[1] + b.get(1, 0) * u,
            ]
        };
        for (k, ev) in events.iter().enumerate() {
            let u = ev.input[0];
            let sub = 1000;
            let hstep = 0.01 / sub as f64;
            for _ in 0..sub {
                let k1 = f(&h, u);
                let k2 = f(&[h[0] + 0.5 * hstep * k1[0], h[1] + 0.5 * hstep * k1[1]], u);
                let k3 = f(&[h[0] + 0.5 * hstep * k2[0], h[1] + 0.5 * hstep * k2[1]], u);
                let k4 = f(&[h[0] + hstep * k3[0], h[1] + hstep * k3[1]], u);
                for i in 0..2 {
                    h[i] += hstep / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
                }
            }
            assert!(max_diff(par.states[k].coords(), &h) <= 1e-10, "k={k}");
            assert!(max_diff(par.states[k].coords(), ser.states[k].coords()) <= 1e-10);
        }
    }

    #[test]
    fn prepare_rejects_bad_windows() {
        let a = Matrix::from_rows(&[vec![-1.0]]).unwrap();
        let b = Matrix::from_rows(&[vec![1.0]]).unwrap();
        let ssm = ZohTangentSsm::new();
        let ctx = ctx_with(a.clone(), b.clone(), budget());
        assert_eq!(ssm.prepare(&[], &ctx).unwrap_err(), Reject::DomainViolation);
        // Time goes backwards.
        let back = [event(20, 1.0), event(10, 1.0)];
        assert_eq!(
            ssm.prepare(&back, &ctx).unwrap_err(),
            Reject::DomainViolation
        );
        // Wrong input width.
        let wide = [Event {
            id: [0; 32],
            time_ns: 1,
            input: vec![1.0, 2.0].into_boxed_slice(),
        }];
        assert_eq!(ssm.prepare(&wide, &ctx).unwrap_err(), Reject::FiberMismatch);
        // Non-finite input.
        assert_eq!(
            ssm.prepare(&[event(1, f64::NAN)], &ctx).unwrap_err(),
            Reject::NonFiniteState
        );
        // Too many steps.
        let small = ScanBudget {
            max_steps: 2,
            ..budget()
        };
        let ctx2 = ctx_with(a, b, small);
        let three = [event(1, 0.0), event(2, 0.0), event(3, 0.0)];
        assert_eq!(
            ssm.prepare(&three, &ctx2).unwrap_err(),
            Reject::BudgetExceeded
        );
    }

    #[test]
    fn divergent_generator_is_rejected_as_domain_violation() {
        // mu_inf(A) = +2: states can grow like e^{2t}. Over 64 steps of
        // 0.1 s the certified growth e^{12.8} ~ 3.6e5 passes a 1e6 budget,
        // but 128 steps (e^{25.6} ~ 1.3e11) must not.
        let a = Matrix::from_rows(&[vec![2.0]]).unwrap();
        let b = Matrix::from_rows(&[vec![1.0]]).unwrap();
        let ctx = ctx_with(a, b, budget());
        let ssm = ZohTangentSsm::new();
        let ok: Vec<Event> = (1..=64).map(|k| event(k * 100_000_000, 0.0)).collect();
        let w = ssm.prepare(&ok, &ctx).unwrap();
        assert!(w.log_norm_inf > 0.0);
        let bad: Vec<Event> = (1..=128).map(|k| event(k * 100_000_000, 0.0)).collect();
        assert_eq!(
            ssm.prepare(&bad, &ctx).unwrap_err(),
            Reject::DomainViolation
        );

        // Spec 25 section 4.3: per-step spectral radius 0.5 is not enough.
        // Alternating [[.5,2],[0,.5]] and [[.5,0],[2,.5]] grows without bound.
        let up = Matrix::from_rows(&[vec![0.5, 2.0], vec![0.0, 0.5]]).unwrap();
        let lo = Matrix::from_rows(&[vec![0.5, 0.0], vec![2.0, 0.5]]).unwrap();
        let steps: Vec<AffineStep> = (0..40)
            .map(|i| {
                let m = if i % 2 == 0 { up.clone() } else { lo.clone() };
                AffineStep::new(fiber(0), fiber(0), m, vec![0.0, 0.0]).unwrap()
            })
            .collect();
        assert_eq!(
            PreparedWindow::from_steps(fiber(0), steps, budget()).unwrap_err(),
            Reject::DomainViolation
        );
    }

    #[test]
    fn huge_generator_step_is_rejected_before_overflow() {
        let a = Matrix::from_rows(&[vec![1e6]]).unwrap();
        let b = Matrix::from_rows(&[vec![1.0]]).unwrap();
        let ctx = ctx_with(a, b, budget());
        let ssm = ZohTangentSsm::new();
        assert_eq!(
            ssm.prepare(&[event(1_000_000_000, 1.0)], &ctx).unwrap_err(),
            Reject::DomainViolation
        );
    }

    #[test]
    fn state_bound_is_enforced() {
        let a = Matrix::from_rows(&[vec![0.0]]).unwrap();
        let b = Matrix::from_rows(&[vec![1.0]]).unwrap();
        let tight = ScanBudget {
            max_state_norm: 1.5,
            ..budget()
        };
        let ctx = ctx_with(a, b, tight);
        let ssm = ZohTangentSsm::new();
        // Integrator: h_t = h_0 + sum dt u. Reaches 2.0 > 1.5.
        let events: Vec<Event> = (1..=4).map(|k| event(k * 500_000_000, 1.0)).collect();
        let w = ssm.prepare(&events, &ctx).unwrap();
        let h0 = Tangent::new(fiber(0), vec![0.0]).unwrap();
        assert_eq!(
            ssm.scan(&w, &h0, Backend::CpuSerial).unwrap_err(),
            Reject::DomainViolation
        );
    }
}
