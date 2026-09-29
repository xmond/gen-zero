//! Fenwick Tree (Binary Indexed Tree) for O(log N) prioritized causal sampling.
//!
//! Maintains prefix sums of transition priorities with point updates in O(log N)
//! and prefix sum / quantile lookups in O(log N).

use crate::error::StorageError;

pub struct FenwickTree {
    tree: Vec<f64>,
    capacity: usize,
}

impl FenwickTree {
    /// Create a new Fenwick tree for up to `capacity` elements (1-indexed internally).
    pub fn new(capacity: usize) -> Self {
        Self {
            tree: vec![0.0; capacity + 1],
            capacity,
        }
    }

    /// Point update: add `delta` to element at 0-indexed position `index`.
    #[inline]
    pub fn update(&mut self, index: usize, delta: f64) {
        if index >= self.capacity {
            return;
        }
        let mut idx = index + 1; // Convert to 1-indexed
        while idx <= self.capacity {
            self.tree[idx] += delta;
            idx += idx & (!idx + 1); // idx += idx & -idx
        }
    }

    /// Compute prefix sum of priorities from index 0 to `index` (inclusive).
    #[inline]
    pub fn prefix_sum(&self, index: usize) -> f64 {
        if self.capacity == 0 || index >= self.capacity {
            return self.total_sum();
        }
        let mut sum = 0.0;
        let mut idx = index + 1;
        while idx > 0 {
            sum += self.tree[idx];
            idx -= idx & (!idx + 1); // idx -= idx & -idx
        }
        sum
    }

    /// Total sum of all priorities.
    #[inline]
    pub fn total_sum(&self) -> f64 {
        let mut sum = 0.0;
        let mut idx = self.capacity;
        while idx > 0 {
            sum += self.tree[idx];
            idx -= idx & (!idx + 1);
        }
        sum
    }

    /// Binary search for largest index such that prefix_sum(index) <= target.
    /// Used for O(log N) priority sampling given uniform target in [0, total_sum].
    pub fn find_prefix_quantile(&self, target: f64) -> Result<usize, StorageError> {
        let total = self.total_sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(StorageError::BufferEmpty);
        }
        if !target.is_finite() {
            return Err(StorageError::Serialization(
                "Quantile target must be finite".into(),
            ));
        }

        let clamped_target = target.max(0.0).min(total);
        let mut idx = 0;
        let mut bit_mask = 1;
        while (bit_mask << 1) <= self.capacity {
            bit_mask <<= 1;
        }

        let mut current_sum = 0.0;
        while bit_mask != 0 {
            let next_idx = idx + bit_mask;
            if next_idx <= self.capacity && current_sum + self.tree[next_idx] < clamped_target {
                idx = next_idx;
                current_sum += self.tree[next_idx];
            }
            bit_mask >>= 1;
        }

        // Return 0-indexed position, clamped within valid capacity
        let result = idx.min(self.capacity - 1);
        Ok(result)
    }

    /// Clear all priorities.
    pub fn clear(&mut self) {
        self.tree.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fenwick_tree_prefix_and_point_update() {
        let mut bit = FenwickTree::new(8);
        bit.update(0, 1.0);
        bit.update(1, 2.0);
        bit.update(2, 3.0);
        bit.update(3, 4.0);

        assert!((bit.prefix_sum(0) - 1.0).abs() < 1e-6);
        assert!((bit.prefix_sum(1) - 3.0).abs() < 1e-6);
        assert!((bit.prefix_sum(2) - 6.0).abs() < 1e-6);
        assert!((bit.prefix_sum(3) - 10.0).abs() < 1e-6);
        assert!((bit.total_sum() - 10.0).abs() < 1e-6);

        // Update index 1 with +5.0 (new value 7.0)
        bit.update(1, 5.0);
        assert!((bit.prefix_sum(1) - 8.0).abs() < 1e-6);
        assert!((bit.total_sum() - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_fenwick_tree_quantile_sampling() {
        let mut bit = FenwickTree::new(4);
        bit.update(0, 10.0);
        bit.update(1, 20.0);
        bit.update(2, 30.0);
        bit.update(3, 40.0);
        // Total = 100.0. Cumulative: [10, 30, 60, 100]

        assert_eq!(bit.find_prefix_quantile(5.0).unwrap(), 0);
        assert_eq!(bit.find_prefix_quantile(15.0).unwrap(), 1);
        assert_eq!(bit.find_prefix_quantile(35.0).unwrap(), 2);
        assert_eq!(bit.find_prefix_quantile(75.0).unwrap(), 3);
    }
}
