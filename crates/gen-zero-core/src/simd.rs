//! gen-zero-core SIMD-accelerated math kernels and POPCNT bit operations.

/// Compute inner product between two f32 slices.
/// Uses 8-way unrolling to utilize FMA pipeline on modern x86_64 / aarch64 cores.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");

    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut sum4 = 0.0_f32;
    let mut sum5 = 0.0_f32;
    let mut sum6 = 0.0_f32;
    let mut sum7 = 0.0_f32;

    for i in 0..chunks {
        let base = i * 8;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
        sum2 += a[base + 2] * b[base + 2];
        sum3 += a[base + 3] * b[base + 3];
        sum4 += a[base + 4] * b[base + 4];
        sum5 += a[base + 5] * b[base + 5];
        sum6 += a[base + 6] * b[base + 6];
        sum7 += a[base + 7] * b[base + 7];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);

    let rem_base = chunks * 8;
    for j in 0..remainder {
        total += a[rem_base + j] * b[rem_base + j];
    }

    total
}

/// Compute Euclidean distance ||a - b||_2.
#[inline]
pub fn l2_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");
    let mut sum_sq = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Compute cosine distance 1.0 - cos_sim(a, b).
#[inline]
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_f32(a, b);
    let n1 = dot_product_f32(a, a).sqrt().max(1e-12);
    let n2 = dot_product_f32(b, b).sqrt().max(1e-12);
    1.0 - (dot / (n1 * n2)).clamp(-1.0, 1.0)
}

/// Compute Hamming distance between two 64-bit word bit-vectors using hardware POPCNT.
/// Used for Stage 1 kappa-HDC ultra-fast binary candidate recall (POPCNT filtering).
#[inline]
pub fn hamming_distance_u64(a: &[u64], b: &[u64]) -> u32 {
    assert_eq!(a.len(), b.len(), "Word slices must have equal length");
    let mut dist: u32 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dist += (x ^ y).count_ones();
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let dot = dot_product_f32(&a, &b);
        let expected: f32 = a.iter().map(|&x| x * 2.0).sum();
        assert!((dot - expected).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_distance() {
        let a = [0b10101010_u64, 0b11110000_u64];
        let b = [0b00101010_u64, 0b00001111_u64];
        // Word 0 diff: 1 bit; Word 1 diff: 8 bits => total 9 bits
        assert_eq!(hamming_distance_u64(&a, &b), 9);
    }
}
