//! Personalized PageRank (PPR) Sparse Flow Engine.
//!
//! Implements lock-free / fast CSR-based sparse power iteration for context diffusion:
//! p_{t+1} = (1 - alpha) * W_semantic * p_t + alpha * e_seed.

/// Compute Personalized PageRank over a CSR graph.
///
/// Returns PageRank scores for all nodes.
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
) -> Vec<f32> {
    if num_nodes == 0 || seeds.is_empty() || row_offsets.len() < num_nodes + 1 {
        return vec![0.0; num_nodes];
    }

    let alpha = alpha.clamp(0.01, 0.99);

    // Build normalized seed vector
    let mut seed_vec = vec![0.0_f32; num_nodes];
    let mut total_seed_weight = 0.0_f32;
    for &(node, w) in seeds {
        let idx = node as usize;
        if idx < num_nodes && w > 0.0 {
            seed_vec[idx] += w;
            total_seed_weight += w;
        }
    }

    if total_seed_weight <= 0.0 {
        return vec![0.0; num_nodes];
    }

    for s in &mut seed_vec {
        *s /= total_seed_weight;
    }

    // Precompute out-degree sum for each node (excluding any out-of-bound target vertices)
    let mut out_sums = vec![0.0_f32; num_nodes];
    let has_weights = !edge_weights.is_empty() && edge_weights.len() == col_indices.len();

    for u in 0..num_nodes {
        let start = row_offsets[u].min(col_indices.len());
        let end = row_offsets[u + 1].min(col_indices.len());
        if start >= end {
            continue;
        }
        if has_weights {
            let mut sum = 0.0;
            for idx in start..end {
                let v = col_indices[idx] as usize;
                if v < num_nodes {
                    sum += edge_weights[idx].max(0.0);
                }
            }
            out_sums[u] = sum;
        } else {
            let mut valid_count = 0;
            for &col in &col_indices[start..end] {
                if (col as usize) < num_nodes {
                    valid_count += 1;
                }
            }
            out_sums[u] = valid_count as f32;
        }
    }

    let mut p = seed_vec.clone();
    let mut p_next = vec![0.0_f32; num_nodes];

    for _ in 0..max_iters {
        // Base seed injection: alpha * e_seed
        for i in 0..num_nodes {
            p_next[i] = alpha * seed_vec[i];
        }

        // Calculate mass from dangling nodes (out_sums == 0)
        let mut dangling_mass = 0.0_f32;
        for u in 0..num_nodes {
            if out_sums[u] <= 1e-12 {
                dangling_mass += p[u];
            }
        }

        // Distribute dangling mass proportionally to seeds
        let dangling_reinject = (1.0 - alpha) * dangling_mass;
        if dangling_reinject > 0.0 {
            for i in 0..num_nodes {
                p_next[i] += dangling_reinject * seed_vec[i];
            }
        }

        // Push probability mass along outgoing edges: (1 - alpha) * p(u) * (w / out_sums[u])
        let scale = 1.0 - alpha;
        for u in 0..num_nodes {
            let out_sum = out_sums[u];
            if out_sum > 1e-12 && p[u] > 1e-12 {
                let p_u_scaled = scale * (p[u] / out_sum);
                let start = row_offsets[u];
                let end = row_offsets[u + 1];
                for edge_idx in start..end {
                    let v = col_indices[edge_idx] as usize;
                    if v < num_nodes {
                        let w = if has_weights {
                            edge_weights[edge_idx].max(0.0)
                        } else {
                            1.0
                        };
                        p_next[v] += p_u_scaled * w;
                    }
                }
            }
        }

        // Check L1 convergence
        let mut diff = 0.0_f32;
        for i in 0..num_nodes {
            diff += (p_next[i] - p[i]).abs();
        }

        std::mem::swap(&mut p, &mut p_next);

        if diff < tolerance {
            break;
        }
    }

    p
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
        );

        assert_eq!(scores.len(), 3);
        // Node 0 should have highest score due to direct seed teleport
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
        let total: f32 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-3);
    }
}
