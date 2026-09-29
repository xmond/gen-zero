//! gen-zero-core Helmert regular simplex Equiangular Tight Frame (ETF).

use crate::error::EtfError;

/// Simplex ETF Frame for candidate action embedding.
/// Guarantees exact equiangular geometry:
/// <e_i, e_j> = -1 / (K - 1) for all i != j, and ||e_i|| = 1.0 for all i.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplexEtfFrame {
    dimension: usize,
    n_candidates: usize,
    /// Row-major matrix of shape (n_candidates, dimension)
    frame_matrix: Vec<f32>,
}

impl SimplexEtfFrame {
    /// Construct a new Simplex ETF frame for K candidates in dimension D.
    ///
    /// Mathematical Invariants & Guarantees:
    /// - K == 0: Returns Err(EtfError::ZeroCandidates)
    /// - K == 1: Degenerates into a single unit vector [1.0, 0.0, ...]
    /// - D < K - 1: Returns Err(EtfError::DimensionTooLow)
    /// - K >= 2: Generates an exact Helmert orthogonal simplex embedded in R^D
    pub fn new(n_candidates: usize, dimension: usize) -> Result<Self, EtfError> {
        if n_candidates == 0 {
            return Err(EtfError::ZeroCandidates);
        }

        if n_candidates == 1 {
            if dimension == 0 {
                return Err(EtfError::DimensionTooLow {
                    dimension: 0,
                    required: 1,
                });
            }
            let mut matrix = vec![0.0_f32; dimension];
            matrix[0] = 1.0_f32;
            return Ok(Self {
                dimension,
                n_candidates: 1,
                frame_matrix: matrix,
            });
        }

        let required_dim = n_candidates - 1;
        if dimension < required_dim {
            return Err(EtfError::DimensionTooLow {
                dimension,
                required: required_dim,
            });
        }

        let k = n_candidates;
        let mut matrix = vec![0.0_f32; k * dimension];

        // Construct orthonormal basis of the hyperplane sum(x_i) = 0 using Helmert matrix
        // Helmert rows h_j (for j = 1..k-1):
        // h_j = [1, 1, ..., 1, -j, 0, ..., 0] / sqrt(j * (j + 1))
        // where there are j ones, followed by -j at index j, and zeros elsewhere.
        //
        // Vertex v_i in R^k is:
        // v_i = sqrt(k / (k - 1)) * (e_i - (1/k) * 1)
        //
        // Then the j-th coordinate in R^(k-1) is w_i[j] = <h_{j+1}, v_i>.
        // Since <h_r, 1> = 0, <h_r, v_i> = sqrt(k / (k - 1)) * (h_r)_i.
        let scale = ((k as f64) / ((k - 1) as f64)).sqrt();

        for i in 0..k {
            let row_offset = i * dimension;
            for r in 1..k {
                // r is 1-indexed, corresponding to row j = r - 1 in R^(k-1)
                let j = r - 1;
                let rf = r as f64;
                let norm_const = 1.0 / ((rf * (rf + 1.0)).sqrt());

                let h_r_i = if i < r {
                    1.0 * norm_const
                } else if i == r {
                    -(r as f64) * norm_const
                } else {
                    0.0
                };

                let coord = (scale * h_r_i) as f32;
                matrix[row_offset + j] = coord;
            }
        }

        Ok(Self {
            dimension,
            n_candidates,
            frame_matrix: matrix,
        })
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline]
    pub fn n_candidates(&self) -> usize {
        self.n_candidates
    }

    /// Get reference to the embedding vector of candidate `idx`.
    #[inline]
    pub fn candidate_vector(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.n_candidates {
            return None;
        }
        let start = idx * self.dimension;
        Some(&self.frame_matrix[start..start + self.dimension])
    }

    /// Project latent representation vector onto all candidate ETF vertices.
    /// Returns logits slice of length `n_candidates`.
    pub fn project_logits(&self, latent: &[f32], logits: &mut [f32]) {
        assert!(
            latent.len() >= self.dimension,
            "Latent dimension {} must be >= ETF dimension {}",
            latent.len(),
            self.dimension
        );
        assert!(
            logits.len() >= self.n_candidates,
            "Logits buffer len {} must be >= n_candidates {}",
            logits.len(),
            self.n_candidates
        );

        // In Helmert regular simplex ETF, coordinates beyond (n_candidates - 1) are strictly zero.
        // Truncate dot product to effective dimension to eliminate needless multiplies.
        let eff_dim = if self.n_candidates <= 1 {
            1
        } else {
            (self.n_candidates - 1).min(self.dimension)
        };

        for (i, logit) in logits[..self.n_candidates].iter_mut().enumerate() {
            let vec_i = self.candidate_vector(i).unwrap();
            *logit = crate::simd::dot_product_f32(&vec_i[..eff_dim], &latent[..eff_dim]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_etf_properties() {
        for k in 2..=8 {
            let dim = 16;
            let etf = SimplexEtfFrame::new(k, dim).expect("ETF creation should succeed");

            let theoretical_cos = -1.0_f32 / ((k - 1) as f32);

            // Verify each vector is unit norm
            for i in 0..k {
                let vi = etf.candidate_vector(i).unwrap();
                let norm = crate::simd::dot_product_f32(vi, vi).sqrt();
                assert!((norm - 1.0).abs() < 1e-5, "k={}, i={}, norm={}", k, i, norm);
            }

            // Verify pairwise inner products
            for i in 0..k {
                for j in 0..k {
                    let vi = etf.candidate_vector(i).unwrap();
                    let vj = etf.candidate_vector(j).unwrap();
                    let dot = crate::simd::dot_product_f32(vi, vj);
                    if i == j {
                        assert!((dot - 1.0).abs() < 1e-5);
                    } else {
                        assert!(
                            (dot - theoretical_cos).abs() < 1e-5,
                            "k={}, i={}, j={}, dot={}, theoretical={}",
                            k,
                            i,
                            j,
                            dot,
                            theoretical_cos
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_degenerate_and_error_cases() {
        assert_eq!(
            SimplexEtfFrame::new(0, 10).unwrap_err(),
            EtfError::ZeroCandidates
        );
        assert_eq!(
            SimplexEtfFrame::new(5, 3).unwrap_err(),
            EtfError::DimensionTooLow {
                dimension: 3,
                required: 4
            }
        );

        // k = 1 scalar case
        let etf1 = SimplexEtfFrame::new(1, 4).unwrap();
        assert_eq!(etf1.n_candidates(), 1);
        let v0 = etf1.candidate_vector(0).unwrap();
        assert_eq!(v0, &[1.0, 0.0, 0.0, 0.0]);
    }
}
