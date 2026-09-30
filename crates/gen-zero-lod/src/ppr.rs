//! Personalized PageRank (PPR) Sparse Flow Engine.
//!
//! Implements lock-free / fast CSR-based sparse power iteration for context diffusion:
//! p_{t+1} = (1 - alpha) * W_semantic * p_t + alpha * e_seed.

use crate::error::LodError;

/// Scores of one Personalized PageRank run.
#[derive(Clone, Debug, PartialEq)]
pub struct PprScores {
    /// Score of every node, indexed by node id.
    pub scores: Vec<f32>,
    /// Power iterations actually run.
    pub iterations: usize,
    /// L1 change of the last iteration.
    pub residual: f32,
    /// `residual < tolerance` before `max_iters` ran out.
    pub converged: bool,
}

fn invalid(detail: impl Into<String>) -> LodError {
    LodError::InvalidQuery(detail.into())
}

/// Compute Personalized PageRank over a CSR graph.
///
/// Every input is checked and a bad one is an error: nothing is clamped,
/// skipped or replaced. `edge_weights` may be empty (unit weights) or must be
/// parallel to `col_indices` with finite, nonnegative values.
#[allow(clippy::too_many_arguments)]
pub fn compute_ppr_csr(
    num_nodes: usize,
    row_offsets: &[usize],
    col_indices: &[u32],
    edge_weights: &[f32],
    seeds: &[(u32, f32)],
    alpha: f32,
    max_iters: usize,
    tolerance: f32,
) -> Result<PprScores, LodError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(invalid(format!("alpha must lie in (0, 1), got {alpha}")));
    }
    if max_iters == 0 {
        return Err(invalid("max_iters must be at least 1"));
    }
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(invalid(format!(
            "tolerance must be finite and nonnegative, got {tolerance}"
        )));
    }
    if seeds.is_empty() {
        return Err(invalid("PPR needs at least one seed"));
    }
    if row_offsets.len() != num_nodes + 1
        || row_offsets[0] != 0
        || row_offsets.windows(2).any(|w| w[0] > w[1])
        || row_offsets[num_nodes] != col_indices.len()
    {
        return Err(invalid(
            "row_offsets do not describe a CSR of num_nodes rows",
        ));
    }
    let has_weights = !edge_weights.is_empty();
    if has_weights && edge_weights.len() != col_indices.len() {
        return Err(invalid(
            "edge_weights must be empty or parallel to col_indices",
        ));
    }
    if let Some(v) = col_indices.iter().find(|&&v| v as usize >= num_nodes) {
        return Err(invalid(format!(
            "edge target {v} is outside {num_nodes} nodes"
        )));
    }
    if let Some(w) = edge_weights.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
        return Err(invalid(format!(
            "edge weight {w} is not finite and nonnegative"
        )));
    }

    let mut seed_vec = vec![0.0_f32; num_nodes];
    let mut total_seed_weight = 0.0_f32;
    for &(node, w) in seeds {
        let idx = node as usize;
        if idx >= num_nodes {
            return Err(invalid(format!(
                "seed node {node} is outside {num_nodes} nodes"
            )));
        }
        if !(w.is_finite() && w > 0.0) {
            return Err(invalid(format!(
                "seed weight {w} for node {node} must be finite and positive"
            )));
        }
        seed_vec[idx] += w;
        total_seed_weight += w;
    }
    if !total_seed_weight.is_finite() {
        return Err(invalid("seed weights overflow"));
    }
    for s in &mut seed_vec {
        *s /= total_seed_weight;
    }

    let weight = |idx: usize| if has_weights { edge_weights[idx] } else { 1.0 };
    let mut out_sums = vec![0.0_f32; num_nodes];
    for (u, out_sum) in out_sums.iter_mut().enumerate() {
        *out_sum = (row_offsets[u]..row_offsets[u + 1]).map(weight).sum();
    }

    let mut p = seed_vec.clone();
    let mut p_next = vec![0.0_f32; num_nodes];
    let mut iterations = 0;
    let mut residual = f32::INFINITY;
    let scale = 1.0 - alpha;

    while iterations < max_iters {
        iterations += 1;
        // Base seed injection: alpha * e_seed
        for i in 0..num_nodes {
            p_next[i] = alpha * seed_vec[i];
        }

        // Dangling nodes (no outgoing weight) return their mass to the seeds.
        let dangling_mass: f32 = (0..num_nodes)
            .filter(|&u| out_sums[u] <= 1e-12)
            .map(|u| p[u])
            .sum();
        let dangling_reinject = scale * dangling_mass;
        if dangling_reinject > 0.0 {
            for i in 0..num_nodes {
                p_next[i] += dangling_reinject * seed_vec[i];
            }
        }

        // Push probability mass along outgoing edges: (1 - alpha) * p(u) * (w / out_sums[u])
        for u in 0..num_nodes {
            let out_sum = out_sums[u];
            if out_sum > 1e-12 && p[u] > 0.0 {
                let p_u_scaled = scale * (p[u] / out_sum);
                for edge_idx in row_offsets[u]..row_offsets[u + 1] {
                    p_next[col_indices[edge_idx] as usize] += p_u_scaled * weight(edge_idx);
                }
            }
        }

        residual = (0..num_nodes).map(|i| (p_next[i] - p[i]).abs()).sum();
        std::mem::swap(&mut p, &mut p_next);
        if residual < tolerance {
            break;
        }
    }

    Ok(PprScores {
        scores: p,
        iterations,
        residual,
        converged: residual < tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppr_triangle() {
        // Graph: 0 -> 1 -> 2 -> 0
        let num_nodes = 3;
        let row_offsets = vec![0, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let weights = vec![1.0, 1.0, 1.0];
        let seeds = vec![(0, 1.0)];

        let scores = compute_ppr_csr(
            num_nodes,
            &row_offsets,
            &col_indices,
            &weights,
            &seeds,
            0.15,
            30,
            1e-5,
        )
        .unwrap()
        .scores;

        assert_eq!(scores.len(), 3);
        // Node 0 should have highest score due to direct seed teleport
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
        let total: f32 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_ppr_refuses_bad_inputs_instead_of_clamping() {
        let ok = |alpha, seeds: &[(u32, f32)], w: &[f32]| {
            compute_ppr_csr(2, &[0, 1, 1], &[1], w, seeds, alpha, 10, 1e-5)
        };
        assert!(ok(0.15, &[(0, 1.0)], &[1.0]).is_ok());
        assert!(ok(0.0, &[(0, 1.0)], &[1.0]).is_err());
        assert!(ok(1.5, &[(0, 1.0)], &[1.0]).is_err());
        assert!(ok(0.15, &[], &[1.0]).is_err());
        assert!(ok(0.15, &[(5, 1.0)], &[1.0]).is_err());
        assert!(ok(0.15, &[(0, -1.0)], &[1.0]).is_err());
        assert!(ok(0.15, &[(0, 1.0)], &[f32::NAN]).is_err());
        assert!(compute_ppr_csr(2, &[0, 1, 1], &[7], &[], &[(0, 1.0)], 0.15, 10, 1e-5).is_err());
    }

    #[test]
    fn test_ppr_reports_nonconvergence() {
        let out =
            compute_ppr_csr(3, &[0, 1, 2, 3], &[1, 2, 0], &[], &[(0, 1.0)], 0.15, 1, 0.0).unwrap();
        assert_eq!(out.iterations, 1);
        assert!(!out.converged);
    }
}
