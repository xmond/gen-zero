//! gen-zero-worldmodel Koopman spectral jump operator for O(1) constant-time lookahead.
//!
//! Uses Real Jordan canonical form (2x2 rotation-scaling blocks for complex conjugate pairs),
//! non-NaN spectral radius contraction (<= 1.0), and optional eigenvector basis transforms
//! Psi_H = V * Lambda^H * V^{-1} * Psi_0.

use crate::error::WorldModelError;
use nalgebra::{DMatrix, DVector};

/// Raise a finite `f32` base to a `usize` exponent without narrowing the exponent.
#[inline]
fn pow_usize(mut base: f32, mut exponent: usize) -> f32 {
    let mut result = 1.0_f32;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result *= base;
        }
        exponent >>= 1;
        if exponent != 0 {
            base *= base;
        }
    }
    result
}

/// Real Jordan Block representation of Koopman eigenvalues.
#[derive(Debug, Clone, PartialEq)]
pub enum JordanBlock {
    /// 1D real eigenvalue: lambda \in [-1.0, 1.0]
    Real { lambda: f32 },
    /// 2D complex conjugate pair: lambda = sigma +/- i * omega
    /// Polar form: r * exp(+/- i * theta)
    ComplexPair {
        radius: f32, // Contracted to <= 1.0
        theta: f32,  // Phase angle in radians
    },
}

impl JordanBlock {
    /// Real eigenvalue with robust non-NaN spectral radius contraction <= 1.0
    pub fn real(lambda: f32) -> Self {
        let contracted = if lambda.is_finite() {
            lambda.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        Self::Real { lambda: contracted }
    }

    /// Complex conjugate pair with robust non-NaN spectral radius contraction <= 1.0
    pub fn complex_pair(sigma: f32, omega: f32) -> Self {
        let r = f64::from(sigma).hypot(f64::from(omega));
        let contracted_r = if r.is_finite() { r.min(1.0) } else { 0.0 };
        let theta = if sigma.is_finite() && omega.is_finite() {
            omega.atan2(sigma)
        } else {
            0.0
        };
        Self::ComplexPair {
            radius: contracted_r as f32,
            theta,
        }
    }

    /// Dimension spanned by this block (1 for Real, 2 for ComplexPair)
    #[inline]
    pub fn dim(&self) -> usize {
        match self {
            JordanBlock::Real { .. } => 1,
            JordanBlock::ComplexPair { .. } => 2,
        }
    }

    /// Forward jump by horizon H steps in O(1) closed-form!
    pub fn jump_h(&self, input: &[f32], horizon: usize, output: &mut [f32]) {
        if horizon == 0 {
            output[..self.dim()].copy_from_slice(&input[..self.dim()]);
            return;
        }
        match self {
            JordanBlock::Real { lambda } => {
                let factor = pow_usize(*lambda, horizon);
                output[0] = input[0] * factor;
            }
            JordanBlock::ComplexPair { radius, theta } => {
                let r_h = pow_usize(*radius, horizon);
                // A f32 phase has at most 24 significant bits. Multiply it by
                // 24-bit integer chunks so every partial angle is exact in f64,
                // then compose rotations. Reducing theta modulo an approximate
                // 2*pi before multiplication would amplify phase error for huge H.
                const CHUNK_BITS: u32 = 24;
                const CHUNK_BASE: usize = 1 << CHUNK_BITS;
                let mut h = horizon;
                let mut theta_scale = f64::from(*theta);
                let (mut sin_h, mut cos_h) = (0.0, 1.0);
                while h != 0 {
                    let angle = theta_scale * (h & (CHUNK_BASE - 1)) as f64;
                    let (s, c) = angle.sin_cos();
                    (sin_h, cos_h) = (sin_h * c + cos_h * s, cos_h * c - sin_h * s);
                    h >>= CHUNK_BITS;
                    theta_scale *= CHUNK_BASE as f64;
                }

                // [ cos(H theta)   sin(H theta) ] [ x0 ]
                // [-sin(H theta)   cos(H theta) ] [ x1 ]
                let x0 = f64::from(input[0]);
                let x1 = f64::from(input[1]);

                output[0] = (f64::from(r_h) * (cos_h * x0 + sin_h * x1)) as f32;
                output[1] = (f64::from(r_h) * (-sin_h * x0 + cos_h * x1)) as f32;
            }
        }
    }
}

/// Koopman Spectral Jumper holding Real Jordan blocks and optional eigenvector basis transforms.
#[derive(Debug, Clone)]
pub struct KoopmanSpectralJumper {
    blocks: Vec<JordanBlock>,
    total_dim: usize,
    basis_v: Option<DMatrix<f32>>,
    basis_inv_v: Option<DMatrix<f32>>,
}

impl KoopmanSpectralJumper {
    /// Create jumper operating directly on decoupled modal observables.
    pub fn new(blocks: Vec<JordanBlock>) -> Self {
        let total_dim = blocks.iter().map(|b| b.dim()).sum();
        Self {
            blocks,
            total_dim,
            basis_v: None,
            basis_inv_v: None,
        }
    }

    /// Create jumper with full eigenvector similarity transform matrices V and V^{-1}.
    /// Reject non-finite matrices, inverse residuals above 1e-3 per entry, or
    /// `cond_1(V) * f32::EPSILON > 0.01` (excessive roundoff amplification).
    pub fn with_basis(
        blocks: Vec<JordanBlock>,
        basis_v: DMatrix<f32>,
        basis_inv_v: DMatrix<f32>,
    ) -> Result<Self, WorldModelError> {
        let total_dim: usize = blocks.iter().map(|b| b.dim()).sum();
        if basis_v.nrows() != total_dim
            || basis_v.ncols() != total_dim
            || basis_inv_v.nrows() != total_dim
            || basis_inv_v.ncols() != total_dim
        {
            return Err(WorldModelError::DimensionMismatch {
                expected: total_dim,
                actual: basis_v.nrows(),
            });
        }
        // A supplied inverse must actually invert V. Reject bases whose f32
        // roundoff amplification consumes more than roughly 1% relative accuracy.
        let v = basis_v.map(f64::from);
        let inv = basis_inv_v.map(f64::from);
        let norm = |m: &DMatrix<f64>| {
            m.column_iter()
                .map(|c| c.iter().map(|x| x.abs()).sum::<f64>())
                .fold(0.0, f64::max)
        };
        let condition = norm(&v) * norm(&inv);
        let residual = &v * &inv - DMatrix::identity(total_dim, total_dim);
        if basis_v
            .iter()
            .chain(basis_inv_v.iter())
            .any(|x| !x.is_finite())
            || condition * f64::from(f32::EPSILON) > 0.01
            || residual.iter().any(|x| !x.is_finite() || x.abs() > 1e-3)
        {
            return Err(WorldModelError::NumericalDivergence);
        }
        Ok(Self {
            blocks,
            total_dim,
            basis_v: Some(basis_v),
            basis_inv_v: Some(basis_inv_v),
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.total_dim
    }

    /// Project observable vector forward by H steps in O(1) time complexity.
    /// Computes Psi_H = V * Lambda^H * V^{-1} * Psi_0 (or pure Lambda^H if modal).
    pub fn forward_jump(
        &self,
        observable: &[f32],
        horizon: usize,
    ) -> Result<Vec<f32>, WorldModelError> {
        if observable.len() != self.total_dim {
            return Err(WorldModelError::DimensionMismatch {
                expected: self.total_dim,
                actual: observable.len(),
            });
        }
        if observable.iter().any(|x| !x.is_finite()) {
            return Err(WorldModelError::NumericalDivergence);
        }
        if horizon == 0 {
            return Ok(observable.to_vec());
        }

        // Project physical observables to modal coordinates if basis is provided
        let modal_input = if let Some(ref inv_v) = self.basis_inv_v {
            let obs_vec = DVector::from_row_slice(observable);
            let modal_vec = inv_v * obs_vec;
            modal_vec.as_slice().to_vec()
        } else {
            observable.to_vec()
        };

        // Modal forward jump across Jordan blocks
        let mut modal_jumped = vec![0.0_f32; self.total_dim];
        let mut offset = 0;

        for block in &self.blocks {
            let d = block.dim();
            block.jump_h(
                &modal_input[offset..offset + d],
                horizon,
                &mut modal_jumped[offset..offset + d],
            );
            offset += d;
        }

        // Project modal coordinates back to physical space if basis is provided
        let result = if let Some(ref v) = self.basis_v {
            let jumped_vec = DVector::from_row_slice(&modal_jumped);
            let physical_vec = v * jumped_vec;
            physical_vec.as_slice().to_vec()
        } else {
            modal_jumped
        };
        if result.iter().any(|x| !x.is_finite()) {
            return Err(WorldModelError::NumericalDivergence);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_jump_retains_large_horizon_phase() {
        let block = JordanBlock::ComplexPair {
            radius: 1.0,
            theta: 1.0,
        };
        let h = (1_usize << 24) + 1;
        let mut out = [0.0; 2];
        block.jump_h(&[1.0, 0.0], h, &mut out);
        assert!((out[0] - (h as f64).cos() as f32).abs() < 1e-6);
        assert!((out[1] + (h as f64).sin() as f32).abs() < 1e-6);
        // Keep the low bit even when H cannot be represented exactly as f64.
        let h = usize::MAX;
        let (s, c) = (2.0_f64.powi(usize::BITS as i32)).sin_cos();
        block.jump_h(&[1.0, 0.0], h, &mut out);
        let expected_cos = c * 1.0_f64.cos() + s * 1.0_f64.sin();
        let expected_sin = s * 1.0_f64.cos() - c * 1.0_f64.sin();
        assert!((f64::from(out[0]) - expected_cos).abs() < 1e-6);
        assert!((f64::from(out[1]) + expected_sin).abs() < 1e-6);
        let huge = JordanBlock::ComplexPair {
            radius: 1.0,
            theta: f32::MAX,
        };
        huge.jump_h(&[1.0, 0.0], 1, &mut out);
        assert!((out[0] - f64::from(f32::MAX).cos() as f32).abs() < 1e-6);
        huge.jump_h(&[1.0, 0.0], usize::MAX, &mut out);
        assert!(out.iter().all(|x| x.is_finite()));
        assert!((out[0].hypot(out[1]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn finite_large_pair_is_contracted_not_erased() {
        let JordanBlock::ComplexPair { radius, .. } = JordanBlock::complex_pair(f32::MAX, f32::MAX)
        else {
            unreachable!()
        };
        assert_eq!(radius, 1.0);
    }

    #[test]
    fn rejects_invalid_or_ill_conditioned_bases() {
        for (v, inv) in [
            (DMatrix::identity(2, 2), DMatrix::zeros(2, 2)),
            (
                DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1e-8])),
                DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1e8])),
            ),
            (
                DMatrix::from_element(2, 2, f32::NAN),
                DMatrix::identity(2, 2),
            ),
        ] {
            assert!(
                KoopmanSpectralJumper::with_basis(vec![JordanBlock::real(1.0); 2], v, inv).is_err()
            );
        }
        let jumper = KoopmanSpectralJumper::new(vec![JordanBlock::real(1.0)]);
        assert!(jumper.forward_jump(&[f32::INFINITY], 1).is_err());
    }

    #[test]
    fn test_koopman_spectral_jump_stability() {
        let blocks = vec![
            JordanBlock::real(0.95),                     // Exponential decay
            JordanBlock::complex_pair(0.9, 0.435_889_9), // Unit circle rotation: 0.9^2 + 0.4358^2 = 1.0
            JordanBlock::real(1.5),                      // Over-unity: must be contracted to 1.0
        ];
        let jumper = KoopmanSpectralJumper::new(blocks);
        assert_eq!(jumper.dim(), 4);

        let initial_obs = vec![1.0, 1.0, 0.0, 2.0];

        // Jump H=100 steps
        let jumped = jumper.forward_jump(&initial_obs, 100).unwrap();
        assert_eq!(jumped.len(), 4);

        // First component decays to near 0: 0.95^100 ~= 0.0059
        assert!(jumped[0] < 0.01);

        // Rotation pair maintains norm <= initial
        let rot_norm = (jumped[1] * jumped[1] + jumped[2] * jumped[2]).sqrt();
        assert!((rot_norm - 1.0).abs() < 1e-4);

        // Contracted over-unity component remains bounded at 2.0 * (1.0)^100 = 2.0
        assert!((jumped[3] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_koopman_with_basis_transform() {
        let blocks = vec![JordanBlock::real(0.5)];
        // Identity basis matrices 1x1
        let v = DMatrix::from_element(1, 1, 2.0);
        let inv_v = DMatrix::from_element(1, 1, 0.5);

        let jumper = KoopmanSpectralJumper::with_basis(blocks, v, inv_v).unwrap();
        let jumped = jumper.forward_jump(&[4.0], 1).unwrap();
        // modal: 0.5 * 4.0 = 2.0
        // jump: 2.0 * 0.5 = 1.0
        // physical: 2.0 * 1.0 = 2.0
        assert_eq!(jumped, vec![2.0]);
    }

    #[test]
    fn real_jump_supports_horizon_beyond_i32() {
        let horizon = i32::MAX as usize + 1;
        let block = JordanBlock::real(0.5);
        let mut output = [f32::NAN];

        block.jump_h(&[1.0], horizon, &mut output);

        assert_eq!(output, [0.0]);
    }

    #[test]
    fn real_jump_preserves_parity_at_max_usize_horizon() {
        let block = JordanBlock::real(-1.0);
        let mut output = [f32::NAN];

        block.jump_h(&[1.0], usize::MAX, &mut output);

        assert_eq!(output, [-1.0]);
    }
}
