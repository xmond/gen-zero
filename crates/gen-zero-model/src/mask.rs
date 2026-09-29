//! gen-zero-model block-causal attention masking semantics.
//!
//! Provides strict mathematical isolation across candidate options:
//! - Prefix: self-attention (causal or bidirectional)
//! - Option k: attends to Prefix + Option k (strictly isolated from other options)
//! - Gather: attends to All tokens (all-to-all aggregation)

use crate::error::ModelError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixMode {
    /// Causal autoregressive masking within prefix: j <= i
    Causal,
    /// Full bidirectional self-attention within prefix
    Bidirectional,
}

/// Block-causal attention mask generator.
#[derive(Debug, Clone)]
pub struct BlockCausalMask {
    total_len: usize,
    prefix_len: usize,
    option_lens: Vec<usize>,
    has_gather: bool,
    prefix_mode: PrefixMode,
    /// Flat row-major boolean mask of shape (total_len, total_len)
    /// true indicates attention is ALLOWED (value 0.0 in additive attention mask),
    /// false indicates attention is MASKED (value -inf in additive attention mask).
    mask: Vec<bool>,
}

impl BlockCausalMask {
    /// Build a new block-causal attention mask.
    pub fn new(
        prefix_len: usize,
        option_lens: &[usize],
        has_gather: bool,
        prefix_mode: PrefixMode,
    ) -> Result<Self, ModelError> {
        if option_lens.is_empty() {
            return Err(ModelError::EmptyOptions);
        }
        for (idx, &len) in option_lens.iter().enumerate() {
            if len == 0 {
                return Err(ModelError::EmptyOptionLength { option_index: idx });
            }
        }

        let sum_options: usize = option_lens.iter().sum();
        let gather_count = if has_gather { 1 } else { 0 };
        let total_len = prefix_len + sum_options + gather_count;

        let mut mask = vec![false; total_len * total_len];

        // Token category and segment helper
        // Returns (segment_type, local_idx)
        // segment_type: 0 for prefix, k+1 for option k, usize::MAX for gather
        let classify = |idx: usize| -> (usize, usize) {
            if idx < prefix_len {
                (0, idx)
            } else if has_gather && idx == total_len - 1 {
                (usize::MAX, 0)
            } else {
                let mut offset = prefix_len;
                for (opt_idx, &opt_len) in option_lens.iter().enumerate() {
                    if idx < offset + opt_len {
                        return (opt_idx + 1, idx - offset);
                    }
                    offset += opt_len;
                }
                unreachable!("Token index out of bounds in classify");
            }
        };

        for i in 0..total_len {
            let (seg_i, local_i) = classify(i);
            let row_offset = i * total_len;

            for j in 0..total_len {
                let (seg_j, local_j) = classify(j);

                let allow = match seg_i {
                    // Token i is in Prefix
                    0 => match seg_j {
                        0 => match prefix_mode {
                            PrefixMode::Causal => local_j <= local_i,
                            PrefixMode::Bidirectional => true,
                        },
                        _ => false, // Prefix never attends to options or gather
                    },
                    // Token i is in Option k (seg_i = k + 1)
                    k if k <= option_lens.len() => match seg_j {
                        0 => true,           // Can attend to entire prefix
                        s if s == k => true, // Can attend to same option
                        _ => false,          // Strictly isolated from other options and gather
                    },
                    // Token i is Gather Token
                    usize::MAX => true, // Gather token attends to ALL tokens
                    _ => false,
                };

                mask[row_offset + j] = allow;
            }
        }

        Ok(Self {
            total_len,
            prefix_len,
            option_lens: option_lens.to_vec(),
            has_gather,
            prefix_mode,
            mask,
        })
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        self.total_len
    }

    #[inline]
    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    #[inline]
    pub fn option_lens(&self) -> &[usize] {
        &self.option_lens
    }

    #[inline]
    pub fn has_gather(&self) -> bool {
        self.has_gather
    }

    #[inline]
    pub fn prefix_mode(&self) -> PrefixMode {
        self.prefix_mode
    }

    #[inline]
    pub fn is_allowed(&self, query_idx: usize, key_idx: usize) -> bool {
        assert!(query_idx < self.total_len && key_idx < self.total_len);
        self.mask[query_idx * self.total_len + key_idx]
    }

    /// Convert mask to additive attention bias (-inf for masked, 0.0 for allowed)
    pub fn to_additive_bias(&self) -> Vec<f32> {
        let neg_inf = -1e9_f32;
        self.mask
            .iter()
            .map(|&allowed| if allowed { 0.0_f32 } else { neg_inf })
            .collect()
    }
}

/// Generate shared position IDs across options to guarantee absolute zero position bias.
///
/// Invariant:
/// - Prefix: [0, 1, ..., prefix_len - 1]
/// - Option k: [prefix_len, prefix_len + 1, ..., prefix_len + len_k - 1]
/// - All options start at identical offset `prefix_len`.
/// - Gather token: prefix_len + max_len(options).
pub fn generate_shared_position_ids(
    prefix_len: usize,
    option_lens: &[usize],
    has_gather: bool,
) -> Result<Vec<usize>, ModelError> {
    if option_lens.is_empty() {
        return Err(ModelError::EmptyOptions);
    }

    let sum_options: usize = option_lens.iter().sum();
    let total_len = prefix_len + sum_options + if has_gather { 1 } else { 0 };
    let mut pos_ids = Vec::with_capacity(total_len);

    // Prefix positions
    for p in 0..prefix_len {
        pos_ids.push(p);
    }

    // Option positions: all start at prefix_len
    let mut max_opt_len = 0;
    for (opt_idx, &opt_len) in option_lens.iter().enumerate() {
        if opt_len == 0 {
            return Err(ModelError::EmptyOptionLength {
                option_index: opt_idx,
            });
        }
        if opt_len > max_opt_len {
            max_opt_len = opt_len;
        }
        for local_p in 0..opt_len {
            pos_ids.push(prefix_len + local_p);
        }
    }

    // Gather token position
    if has_gather {
        pos_ids.push(prefix_len + max_opt_len);
    }

    Ok(pos_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_causal_mask_isolation() {
        // Prefix: 3 tokens, Option 0: 2 tokens, Option 1: 2 tokens, Gather: 1 token
        let mask = BlockCausalMask::new(3, &[2, 2], true, PrefixMode::Causal).unwrap();
        assert_eq!(mask.total_len(), 3 + 2 + 2 + 1); // 8 tokens

        // Token 0, 1, 2 = Prefix
        // Token 3, 4 = Option 0
        // Token 5, 6 = Option 1
        // Token 7 = Gather

        // Prefix causal check
        assert!(mask.is_allowed(1, 0));
        assert!(mask.is_allowed(1, 1));
        assert!(!mask.is_allowed(1, 2)); // future in causal

        // Option 0 attends to prefix + Option 0
        assert!(mask.is_allowed(3, 0)); // prefix
        assert!(mask.is_allowed(3, 1)); // prefix
        assert!(mask.is_allowed(3, 2)); // prefix
        assert!(mask.is_allowed(4, 3)); // Option 0 self
        assert!(!mask.is_allowed(3, 5)); // Option 0 to Option 1: MUST BE FALSE (isolated)
        assert!(!mask.is_allowed(4, 6)); // Option 0 to Option 1: MUST BE FALSE

        // Option 1 attends to prefix + Option 1, isolated from Option 0
        assert!(mask.is_allowed(5, 0)); // prefix
        assert!(!mask.is_allowed(5, 3)); // Option 1 to Option 0: MUST BE FALSE
        assert!(!mask.is_allowed(6, 4)); // Option 1 to Option 0: MUST BE FALSE
        assert!(mask.is_allowed(6, 5)); // Option 1 self

        // Gather attends to ALL
        for j in 0..8 {
            assert!(mask.is_allowed(7, j), "Gather must attend to token {}", j);
        }
    }

    #[test]
    fn test_shared_position_ids() {
        let pos = generate_shared_position_ids(3, &[2, 3], true).unwrap();
        // Prefix (3 tokens): [0, 1, 2]
        // Option 0 (2 tokens): [3, 4]
        // Option 1 (3 tokens): [3, 4, 5]
        // Gather: 3 + max(2, 3) = 6
        assert_eq!(pos, vec![0, 1, 2, 3, 4, 3, 4, 5, 6]);
    }
}
