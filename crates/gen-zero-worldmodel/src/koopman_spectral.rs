//! Continuous-time Koopman generator with fast matrix exponentials.
//!
//! The Koopman semigroup satisfies `K_t = exp(L t)`, where `L` is the generator.
//! This module computes `exp(A)` two ways, both by scaling and squaring:
//!
//! * [`expm_pade`]: Higham (2005) degree-13 Pade approximant. Default. Accurate to
//!   near machine precision for well-conditioned exponentials; extreme scaling
//!   and non-normality can amplify rounding error.
//! * [`expm_taylor`]: truncated Taylor series on a matrix scaled to `||A||_1 <= 0.5`.
//!   Cheaper per step, and a useful cross-check for the Pade path.
//!
//! Internals run in `f64`. The `f32` observable API in [`crate::koopman`] loses too
//! much precision inside repeated squaring.

use crate::error::WorldModelError;
use gen_zero_core::traits::LatentContraction;
use gen_zero_core::FullLatent;
use nalgebra::DMatrix;

/// Higham's bound on `||A||_1` for which the degree-13 Pade approximant meets
/// double-precision accuracy without scaling.
const PADE13_THETA: f64 = 5.371_920_351_148_152;

/// Scale target for the Taylor path.
const TAYLOR_THETA: f64 = 0.5;

/// Taylor terms. At `||A||_1 <= 0.5` the tail `0.5^20 / 20!` is far below `f64` epsilon.
const TAYLOR_TERMS: usize = 20;

/// Pade-13 numerator/denominator coefficients `b_0..b_13`.
const PADE13_B: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

/// Algorithm used to compute `exp(L t)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpmMethod {
    /// Scaling and squaring with a degree-13 Pade approximant.
    #[default]
    Pade,
    /// Scaling and squaring with a truncated Taylor series.
    Taylor,
}

/// Logarithmic induced 1-norm, avoiding overflow in finite column sums.
fn log2_norm1(a: &DMatrix<f64>) -> f64 {
    let largest = a.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    if largest == 0.0 {
        return f64::NEG_INFINITY;
    }
    let normalized = a
        .column_iter()
        .map(|c| c.iter().map(|x| x.abs() / largest).sum::<f64>())
        .fold(0.0, f64::max);
    largest.log2() + normalized.log2()
}

fn check_square_finite(a: &DMatrix<f64>) -> Result<(), WorldModelError> {
    if a.nrows() != a.ncols() {
        return Err(WorldModelError::DimensionMismatch {
            expected: a.nrows(),
            actual: a.ncols(),
        });
    }
    if a.iter().any(|x| !x.is_finite()) {
        return Err(WorldModelError::NumericalDivergence);
    }
    Ok(())
}

/// Number of halvings so that `||A / 2^s||_1 <= theta`.
fn scaling_power(a: &DMatrix<f64>, theta: f64) -> i32 {
    (log2_norm1(a) - theta.log2()).ceil().max(0.0) as i32
}

/// Apply powers in normal-sized chunks: powi(-s) may underflow internally
/// for s > 1023 even when the desired matrix entries are representable.
fn scale_down(a: &DMatrix<f64>, mut s: i32) -> DMatrix<f64> {
    let mut scaled = a.clone();
    while s > 0 {
        let chunk = s.min(1022);
        scaled *= 2f64.powi(-chunk);
        s -= chunk;
    }
    scaled
}

fn square_in_place(mut r: DMatrix<f64>, times: i32) -> Result<DMatrix<f64>, WorldModelError> {
    for _ in 0..times {
        r = ensure_finite(&r * &r)?;
    }
    ensure_finite(r)
}

fn ensure_finite(r: DMatrix<f64>) -> Result<DMatrix<f64>, WorldModelError> {
    if r.iter().all(|x| x.is_finite()) {
        Ok(r)
    } else {
        Err(WorldModelError::NumericalDivergence)
    }
}

/// Matrix exponential by scaling and squaring with a degree-13 Pade approximant.
pub fn expm_pade(a: &DMatrix<f64>) -> Result<DMatrix<f64>, WorldModelError> {
    check_square_finite(a)?;
    let n = a.nrows();
    if n == 0 {
        return Ok(DMatrix::zeros(0, 0));
    }
    // Avoid overscaling slow diagonal modes when decay rates span many orders
    // of magnitude. Scalar exponentials also handle representable underflow.
    if a.column_iter()
        .enumerate()
        .all(|(j, c)| c.iter().enumerate().all(|(i, &x)| i == j || x == 0.0))
    {
        return ensure_finite(DMatrix::from_diagonal(&a.diagonal().map(f64::exp)));
    }

    let s = scaling_power(a, PADE13_THETA);
    let a = scale_down(a, s);
    let id = DMatrix::<f64>::identity(n, n);
    let b = &PADE13_B;

    let a2 = &a * &a;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;

    let u_inner = &a6 * (&a6 * b[13] + &a4 * b[11] + &a2 * b[9])
        + &a6 * b[7]
        + &a4 * b[5]
        + &a2 * b[3]
        + &id * b[1];
    let u = &a * u_inner;
    let v = &a6 * (&a6 * b[12] + &a4 * b[10] + &a2 * b[8])
        + &a6 * b[6]
        + &a4 * b[4]
        + &a2 * b[2]
        + &id * b[0];

    // exp(A) ~= (V - U)^{-1} (V + U)
    let r = (&v - &u)
        .lu()
        .solve(&(&v + &u))
        .ok_or(WorldModelError::NumericalDivergence)?;
    square_in_place(r, s)
}

/// Matrix exponential by scaling and squaring with a truncated Taylor series.
pub fn expm_taylor(a: &DMatrix<f64>) -> Result<DMatrix<f64>, WorldModelError> {
    check_square_finite(a)?;
    let n = a.nrows();
    if n == 0 {
        return Ok(DMatrix::zeros(0, 0));
    }
    // Avoid overscaling slow diagonal modes when decay rates span many orders
    // of magnitude. Scalar exponentials also handle representable underflow.
    if a.column_iter()
        .enumerate()
        .all(|(j, c)| c.iter().enumerate().all(|(i, &x)| i == j || x == 0.0))
    {
        return ensure_finite(DMatrix::from_diagonal(&a.diagonal().map(f64::exp)));
    }

    let s = scaling_power(a, TAYLOR_THETA);
    let a = scale_down(a, s);

    let mut sum = DMatrix::<f64>::identity(n, n);
    let mut term = sum.clone();
    for k in 1..=TAYLOR_TERMS {
        term = (&term * &a) / k as f64;
        sum += &term;
    }
    square_in_place(sum, s)
}

/// Generator `L` of a continuous-time Koopman semigroup, `K_t = exp(L t)`.
#[derive(Debug, Clone, PartialEq)]
pub struct KoopmanGenerator {
    matrix: DMatrix<f64>,
    /// Jump time used by [`LatentContraction`]. Defaults to 1.0.
    dt: f64,
}

impl KoopmanGenerator {
    /// Wrap a square, finite generator matrix.
    pub fn new(matrix: DMatrix<f64>) -> Result<Self, WorldModelError> {
        check_square_finite(&matrix)?;
        Ok(Self { matrix, dt: 1.0 })
    }

    /// Build from a row-major slice of length `dim * dim`.
    pub fn from_row_slice(dim: usize, data: &[f64]) -> Result<Self, WorldModelError> {
        let expected = dim
            .checked_mul(dim)
            .ok_or(WorldModelError::NumericalDivergence)?;
        if data.len() != expected {
            return Err(WorldModelError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Self::new(DMatrix::from_row_slice(dim, dim, data))
    }

    /// Set the jump time `dt` used by [`LatentContraction::step_in_place`].
    pub fn with_dt(mut self, dt: f64) -> Result<Self, WorldModelError> {
        if !dt.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }
        self.dt = dt;
        Ok(self)
    }

    #[inline]
    pub fn dt(&self) -> f64 {
        self.dt
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    #[inline]
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Operator `exp(L t)` with the default (Pade) method.
    pub fn exp(&self, t: f64) -> Result<DMatrix<f64>, WorldModelError> {
        self.exp_with(t, ExpmMethod::Pade)
    }

    /// Operator `exp(L t)` with an explicit method. `t` may be negative (backward jump).
    pub fn exp_with(&self, t: f64, method: ExpmMethod) -> Result<DMatrix<f64>, WorldModelError> {
        if !t.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }
        let scaled = &self.matrix * t;
        match method {
            ExpmMethod::Pade => expm_pade(&scaled),
            ExpmMethod::Taylor => expm_taylor(&scaled),
        }
    }

    /// Advance an observable vector by time `t`: `psi_t = exp(L t) psi_0`.
    pub fn propagate(&self, observable: &[f32], t: f64) -> Result<Vec<f32>, WorldModelError> {
        if observable.len() != self.dim() {
            return Err(WorldModelError::DimensionMismatch {
                expected: self.dim(),
                actual: observable.len(),
            });
        }
        let k = self.exp(t)?;
        let x = nalgebra::DVector::from_iterator(
            observable.len(),
            observable.iter().map(|&v| f64::from(v)),
        );
        let y = k * x;
        if y.iter().any(|&v| !(v as f32).is_finite()) {
            return Err(WorldModelError::NumericalDivergence);
        }
        Ok(y.iter().map(|&v| v as f32).collect())
    }
}

/// Jump `z <- exp(dt A) z` on the leading `dim` coordinates of the latent.
/// Coordinates beyond `dim` are left untouched. The context is ignored (autonomous flow).
impl LatentContraction for KoopmanGenerator {
    type Error = WorldModelError;

    fn step_in_place(&self, z: &mut FullLatent, _context: &FullLatent) -> Result<(), Self::Error> {
        let d = self.dim();
        let slice = z.as_mut_slice();
        if d > slice.len() {
            return Err(WorldModelError::DimensionMismatch {
                expected: slice.len(),
                actual: d,
            });
        }
        let y = self.propagate(&slice[..d], self.dt)?;
        slice[..d].copy_from_slice(&y);
        Ok(())
    }

    /// Spectral radius of `exp(dt A)`: `exp(max(dt * Re(lambda)))`. This is exact for the
    /// asymptotic rate for either sign of `dt`; a non-normal generator can still grow
    /// transiently.
    /// The empty generator is the identity on zero coordinates: rate 1. A spectrum that
    /// cannot be computed finitely is undecidable: NaN, never a guessed value.
    fn contraction_rate(&self) -> f64 {
        if self.matrix.nrows() == 0 {
            return 1.0;
        }
        match self.max_scaled_real_eigenvalue() {
            Some(max_scaled_re) => max_scaled_re.exp(),
            None => f64::NAN,
        }
    }

    /// Dissipative only when the finite contraction radius is strictly below one.
    fn is_dissipative(&self) -> bool {
        self.contraction_rate() < 1.0
    }
}

impl KoopmanGenerator {
    /// Largest scaled real part `max(dt * Re(lambda))` over the generator's eigenvalues.
    /// `None` for the empty matrix or a non-finite spectrum.
    fn max_scaled_real_eigenvalue(&self) -> Option<f64> {
        if self.matrix.nrows() == 0 {
            return None;
        }
        if self.dt == 0.0 {
            return Some(0.0);
        }
        // Bound QR iteration: a failed/ill-conditioned spectrum must not hang a rollout.
        let schur = nalgebra::linalg::Schur::try_new(self.matrix.clone(), f64::EPSILON, 10_000)?;
        let spectrum = schur.complex_eigenvalues();
        if spectrum
            .iter()
            .any(|l| !l.re.is_finite() || !l.im.is_finite())
        {
            return None;
        }
        let max_re = spectrum
            .iter()
            .map(|l| self.dt * l.re)
            .fold(f64::NEG_INFINITY, f64::max);
        max_re.is_finite().then_some(max_re)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::koopman::{JordanBlock, KoopmanSpectralJumper};

    fn max_abs_diff(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        (a - b).iter().fold(0.0, |m, x| m.max(x.abs()))
    }

    #[test]
    fn extreme_finite_norm_does_not_overflow_scaling() {
        // The column sum exceeds f64::MAX; exp of this triangular dissipative
        // matrix still tends to zero. Previously this requested i32::MAX squarings.
        let a = DMatrix::from_row_slice(2, 2, &[-1e308, 1e308, 0.0, -1e308]);
        for expm in [expm_pade, expm_taylor] {
            assert_eq!(expm(&a).unwrap(), DMatrix::zeros(2, 2));
            let scalar = DMatrix::from_element(1, 1, -1e308);
            assert_eq!(expm(&scalar).unwrap()[(0, 0)], 0.0);
        }
    }

    #[test]
    fn diagonal_stiffness_preserves_slow_modes() {
        let a = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[-1e308, -1.0, 0.0]));
        let expected = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
            0.0,
            (-1.0_f64).exp(),
            1.0,
        ]));
        for expm in [expm_pade, expm_taylor] {
            assert_eq!(expm(&a).unwrap(), expected);
            assert_eq!(
                expm(&DMatrix::from_element(1, 1, 1e308)),
                Err(WorldModelError::NumericalDivergence)
            );
        }
    }

    #[test]
    fn propagation_rejects_f32_overflow_without_mutation() {
        let g = KoopmanGenerator::from_row_slice(1, &[100.0]).unwrap();
        assert_eq!(
            g.propagate(&[1.0], 1.0),
            Err(WorldModelError::NumericalDivergence)
        );
        let mut z = FullLatent::zeros();
        z.values[0] = 1.0;
        let original = z.clone();
        assert!(g.step_in_place(&mut z, &FullLatent::zeros()).is_err());
        assert_eq!(z.values, original.values);
        assert!(KoopmanGenerator::from_row_slice(usize::MAX, &[]).is_err());
    }

    #[test]
    fn nonnormal_generator_matches_analytic_transient_and_reverse() {
        let g = KoopmanGenerator::from_row_slice(2, &[-1.0, 1000.0, 0.0, -1.0]).unwrap();
        for method in [ExpmMethod::Pade, ExpmMethod::Taylor] {
            for t in [-0.5_f64, 0.5, 2.0] {
                let e = (-t).exp();
                let expected = DMatrix::from_row_slice(2, 2, &[e, 1000.0 * t * e, 0.0, e]);
                assert!(max_abs_diff(&g.exp_with(t, method).unwrap(), &expected) < 1e-9);
            }
            let forward = g.exp_with(0.5, method).unwrap();
            let backward = g.exp_with(-0.5, method).unwrap();
            assert!(max_abs_diff(&(forward * backward), &DMatrix::identity(2, 2)) < 1e-9);
        }
        assert!(g.contraction_rate() < 1.0);
        assert!(g.propagate(&[0.0, 1.0], 0.5).unwrap()[0] > 100.0);
    }

    #[test]
    fn zero_generator_gives_identity() {
        let z = DMatrix::<f64>::zeros(3, 3);
        let id = DMatrix::<f64>::identity(3, 3);
        assert!(max_abs_diff(&expm_pade(&z).unwrap(), &id) < 1e-15);
        assert!(max_abs_diff(&expm_taylor(&z).unwrap(), &id) < 1e-15);
    }

    #[test]
    fn diagonal_matches_scalar_exp() {
        let d = [-2.0_f64, 0.5, 3.0];
        let a = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&d));
        let expected = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            3,
            d.iter().map(|x| x.exp()),
        ));
        for m in [expm_pade(&a).unwrap(), expm_taylor(&a).unwrap()] {
            let rel = (&m - &expected)
                .iter()
                .zip(expected.iter())
                .fold(0.0_f64, |acc, (e, x)| {
                    acc.max(if *x == 0.0 { e.abs() } else { (e / x).abs() })
                });
            assert!(rel < 1e-13, "relative error {rel}");
        }
    }

    #[test]
    fn rotation_generator_matches_closed_form() {
        // exp([[0, -w], [w, 0]] t) = [[cos wt, -sin wt], [sin wt, cos wt]]
        let w = 1.7_f64;
        let t = 3.3;
        let a = DMatrix::from_row_slice(2, 2, &[0.0, -w, w, 0.0]) * t;
        let (s, c) = (w * t).sin_cos();
        let expected = DMatrix::from_row_slice(2, 2, &[c, -s, s, c]);
        assert!(max_abs_diff(&expm_pade(&a).unwrap(), &expected) < 1e-13);
        assert!(max_abs_diff(&expm_taylor(&a).unwrap(), &expected) < 1e-13);
    }

    #[test]
    fn nilpotent_series_is_exact() {
        // N^3 = 0, so exp(N) = I + N + N^2 / 2.
        let n = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let expected =
            DMatrix::from_row_slice(3, 3, &[1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        assert!(max_abs_diff(&expm_pade(&n).unwrap(), &expected) < 1e-14);
        assert!(max_abs_diff(&expm_taylor(&n).unwrap(), &expected) < 1e-14);
    }

    #[test]
    fn large_norm_pade_and_taylor_agree() {
        // ||A||_1 = 60 forces several squarings on both paths.
        let a = DMatrix::from_row_slice(2, 2, &[-30.0, 20.0, -20.0, -30.0]);
        let p = expm_pade(&a).unwrap();
        let t = expm_taylor(&a).unwrap();
        // Closed form: e^{-30} * rotation(20) in the [[c, s], [-s, c]] convention.
        let (s, c) = 20.0_f64.sin_cos();
        let e = (-30.0_f64).exp();
        let expected = DMatrix::from_row_slice(2, 2, &[e * c, e * s, -e * s, e * c]);
        assert!(max_abs_diff(&p, &expected) < 1e-13);
        assert!(max_abs_diff(&t, &expected) < 1e-13);
    }

    #[test]
    fn semigroup_and_inverse_properties() {
        let g =
            KoopmanGenerator::from_row_slice(3, &[-0.3, 0.8, 0.0, -0.8, -0.3, 0.1, 0.0, 0.0, -0.5])
                .unwrap();
        let ks = g.exp(0.7).unwrap();
        let kt = g.exp(1.9).unwrap();
        let kst = g.exp(2.6).unwrap();
        assert!(max_abs_diff(&(&ks * &kt), &kst) < 1e-12);

        let back = g.exp(-0.7).unwrap();
        let id = DMatrix::<f64>::identity(3, 3);
        assert!(max_abs_diff(&(&ks * &back), &id) < 1e-12);
    }

    #[test]
    fn matches_jordan_block_jump() {
        // Complex pair r e^{+/- i theta} as a generator: [[ln r, w], [-w, ln r]] per step.
        let (sigma, omega) = (0.9_f32, 0.3_f32);
        let r = f64::from(sigma.hypot(omega));
        let theta = f64::from(omega.atan2(sigma));
        let g = KoopmanGenerator::from_row_slice(2, &[r.ln(), theta, -theta, r.ln()]).unwrap();

        let jumper = KoopmanSpectralJumper::new(vec![JordanBlock::complex_pair(sigma, omega)]);
        let obs = [1.0_f32, -0.5];
        let horizon = 25;
        let expected = jumper.forward_jump(&obs, horizon).unwrap();
        let got = g.propagate(&obs, horizon as f64).unwrap();
        for (e, a) in expected.iter().zip(&got) {
            assert!((e - a).abs() < 1e-5, "expected {e}, got {a}");
        }
    }

    #[test]
    fn method_selection_is_consistent() {
        let g = KoopmanGenerator::from_row_slice(2, &[-0.1, 1.2, -1.2, -0.1]).unwrap();
        let p = g.exp_with(4.0, ExpmMethod::Pade).unwrap();
        let t = g.exp_with(4.0, ExpmMethod::Taylor).unwrap();
        assert!(max_abs_diff(&p, &t) < 1e-13);
        assert_eq!(ExpmMethod::default(), ExpmMethod::Pade);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(
            KoopmanGenerator::new(DMatrix::zeros(2, 3)),
            Err(WorldModelError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            KoopmanGenerator::from_row_slice(2, &[0.0; 3]),
            Err(WorldModelError::DimensionMismatch { .. })
        ));
        assert_eq!(
            KoopmanGenerator::from_row_slice(1, &[f64::NAN]),
            Err(WorldModelError::NumericalDivergence)
        );

        let g = KoopmanGenerator::from_row_slice(2, &[0.0, 1.0, -1.0, 0.0]).unwrap();
        assert!(matches!(
            g.propagate(&[1.0], 1.0),
            Err(WorldModelError::DimensionMismatch { .. })
        ));
        assert_eq!(
            g.exp(f64::INFINITY),
            Err(WorldModelError::NumericalDivergence)
        );
    }

    #[test]
    fn overflow_is_reported_not_returned() {
        let g = KoopmanGenerator::from_row_slice(1, &[1.0]).unwrap();
        assert_eq!(g.exp(1.0e4), Err(WorldModelError::NumericalDivergence));
    }

    #[test]
    fn contraction_rate_matches_spectrum() {
        let g = KoopmanGenerator::from_row_slice(2, &[-1.0, 0.0, 0.0, -2.0])
            .unwrap()
            .with_dt(0.5)
            .unwrap();
        assert!((g.contraction_rate() - (-0.5_f64).exp()).abs() < 1e-12);
        assert!(g.is_dissipative());
        // Negative dt runs the stable flow backward: expanding.
        let back = g.clone().with_dt(-0.5).unwrap();
        assert!(back.contraction_rate() > 1.0);
        assert!(!back.is_dissipative());
        let rot = KoopmanGenerator::from_row_slice(2, &[0.0, -1.0, 1.0, 0.0]).unwrap();
        assert!((rot.contraction_rate() - 1.0).abs() < 1e-12);
        // Weak but real damping is still detected above the rounding margin.
        let weak = KoopmanGenerator::from_row_slice(2, &[-1e-6, -1.0, 1.0, -1e-6]).unwrap();
        assert!(weak.is_dissipative());
    }

    #[test]
    fn signed_dt_uses_maximum_scaled_real_eigenvalue() {
        // Backward time selects the most negative generator eigenvalue. The old
        // max(Re(lambda)) selection incorrectly reported this mixed spectrum as
        // contracting.
        let backward = KoopmanGenerator::from_row_slice(2, &[-2.0, 0.0, 0.0, 1.0])
            .unwrap()
            .with_dt(-0.5)
            .unwrap();

        assert!((backward.contraction_rate() - 1.0_f64.exp()).abs() < 1e-12);
        assert!(!backward.is_dissipative());
    }

    #[test]
    fn negative_dt_selects_slowest_positive_mode() {
        let backward = KoopmanGenerator::from_row_slice(2, &[1.0, 0.0, 0.0, 2.0])
            .unwrap()
            .with_dt(-0.5)
            .unwrap();

        assert!((backward.contraction_rate() - (-0.5_f64).exp()).abs() < 1e-12);
        assert!(backward.is_dissipative());
    }

    #[test]
    fn empty_generator_is_identity_not_dissipative() {
        let g = KoopmanGenerator::new(DMatrix::zeros(0, 0)).unwrap();
        assert_eq!(g.contraction_rate(), 1.0);
        assert!(!g.is_dissipative());
    }
}
