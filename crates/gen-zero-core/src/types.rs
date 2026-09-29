//! gen-zero-core fundamental mathematical and action types.

use crate::error::CoreError;
use serde::{Deserialize, Serialize};

/// Fixed-dimension latent state vector, guaranteed to be aligned to 64 bytes (CPU Cache Line).
#[repr(C, align(64))]
#[derive(Clone, Debug, PartialEq)]
pub struct LatentState<const D: usize> {
    pub values: [f32; D],
}

impl<const D: usize> Serialize for LatentState<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(D))?;
        for elem in &self.values {
            seq.serialize_element(elem)?;
        }
        seq.end()
    }
}

impl<'de, const D: usize> Deserialize<'de> for LatentState<D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        struct LatentVisitor<const N: usize>;
        impl<'de, const N: usize> serde::de::Visitor<'de> for LatentVisitor<N> {
            type Value = LatentState<N>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "a sequence of {} floats", N)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut values = [0.0_f32; N];
                for (i, val) in values.iter_mut().enumerate() {
                    *val = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                }
                Ok(LatentState { values })
            }
        }
        deserializer.deserialize_seq(LatentVisitor::<D>)
    }
}

impl<const D: usize> Default for LatentState<D> {
    #[inline]
    fn default() -> Self {
        Self { values: [0.0; D] }
    }
}

impl<const D: usize> LatentState<D> {
    #[inline]
    pub const fn zeros() -> Self {
        Self { values: [0.0; D] }
    }

    #[inline]
    pub fn from_slice(slice: &[f32]) -> Result<Self, CoreError> {
        if slice.len() != D {
            return Err(CoreError::DimensionMismatch {
                expected: D,
                actual: slice.len(),
            });
        }
        let mut values = [0.0; D];
        values.copy_from_slice(slice);
        Ok(Self { values })
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    #[inline]
    pub const fn dim(&self) -> usize {
        D
    }

    /// Compute L2 norm with epsilon floor to prevent division by zero.
    #[inline]
    pub fn l2_norm(&self) -> f32 {
        crate::simd::dot_product_f32(&self.values, &self.values).sqrt()
    }

    /// In-place normalize with epsilon floor 1e-12.
    #[inline]
    pub fn normalize_in_place(&mut self) {
        let norm = self.l2_norm().max(1e-12);
        let inv_norm = 1.0 / norm;
        for x in self.values.iter_mut() {
            *x *= inv_norm;
        }
    }

    /// Return a normalized copy.
    #[inline]
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize_in_place();
        copy
    }

    /// Compute dot product with another latent vector.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        crate::simd::dot_product_f32(&self.values, &other.values)
    }

    /// Compute cosine similarity in [-1.0, 1.0].
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = self.dot(other);
        let n1 = self.l2_norm().max(1e-12);
        let n2 = other.l2_norm().max(1e-12);
        (dot / (n1 * n2)).clamp(-1.0, 1.0)
    }
}

// Ensure 64-byte alignment and exact cache-line multiple sizing statically
const _: () = assert!(std::mem::align_of::<LatentState<64>>() == 64);
const _: () = assert!(std::mem::align_of::<LatentState<128>>() == 64);
const _: () = assert!(std::mem::align_of::<LatentState<1024>>() == 64);
const _: () = assert!(std::mem::align_of::<LatentState<4096>>() == 64);

const _: () = assert!(std::mem::size_of::<LatentState<64>>() == 256);
const _: () = assert!(std::mem::size_of::<LatentState<128>>() == 512);
const _: () = assert!(std::mem::size_of::<LatentState<1024>>() == 4096);
const _: () = assert!(std::mem::size_of::<LatentState<4096>>() == 16384);

/// Explicit type aliases as mandated by architecture Doc 02 & Milestone Foundation Tier
pub type MicroLatent = LatentState<64>;
pub type CompressedLatent = LatentState<128>;
pub type FullLatent = LatentState<1024>;

/// Foundation-tier latent state for raw Qwen3.5-9B hidden representations (D = 4096).
/// Size: exactly 16,384 bytes (256 cache lines). Guaranteed 64-byte alignment.
/// Prefer passing by reference (`&FoundationLatent`) or boxed to avoid 16 KiB stack frame bloat.
pub type FoundationLatent = LatentState<4096>;

/// Strongly typed Action identifier (u32)
#[repr(transparent)]
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ActionId(pub u32);

impl From<u32> for ActionId {
    #[inline]
    fn from(val: u32) -> Self {
        ActionId(val)
    }
}

impl From<ActionId> for u32 {
    #[inline]
    fn from(val: ActionId) -> Self {
        val.0
    }
}

/// Normalized Shannon Entropy scalar H \in [0.0, 1.0]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NormalizedEntropy(pub f32);

impl NormalizedEntropy {
    pub const ZERO: Self = NormalizedEntropy(0.0);
    pub const ONE: Self = NormalizedEntropy(1.0);

    #[inline]
    pub const fn value(&self) -> f32 {
        self.0
    }

    /// Compute normalized entropy from discrete probabilities:
    /// H = -\sum p_i \ln(p_i + 1e-15) / \ln(\max(K, 2))
    pub fn from_probabilities(probs: &[f32]) -> Self {
        let k = probs.len();
        if k <= 1 {
            return NormalizedEntropy(0.0);
        }
        let eps = 1e-15_f32;
        let mut h = 0.0_f32;
        for &p in probs {
            if p > 0.0 {
                h -= p * (p + eps).ln();
            }
        }
        let max_h = (k as f32).max(2.0).ln();
        let normalized = (h / max_h).clamp(0.0, 1.0);
        NormalizedEntropy(normalized)
    }
}

/// Local action frame for a single decision session.
/// Guarantees stack allocation for up to 16 candidate actions without heap alloc.
#[derive(Debug, Clone, Copy)]
pub struct LocalActionFrame<'a> {
    pub candidate_actions: [&'a str; 16],
    pub local_to_global: [ActionId; 16],
    // Private: `new` guarantees `n_candidates <= 16`, which `actions()` and
    // friends rely on when slicing the fixed arrays.
    n_candidates: usize,
}

impl<'a> LocalActionFrame<'a> {
    /// Create a new local action frame.
    /// Returns None if candidate actions count exceeds stack capacity (16).
    pub fn new(candidate_actions: &'a [&'a str], global_ids: &[ActionId]) -> Option<Self> {
        let n = candidate_actions.len();
        if n > 16 || n != global_ids.len() {
            return None;
        }
        let mut names = [""; 16];
        let mut local_to_global = [ActionId(0); 16];
        for (i, &name) in candidate_actions.iter().enumerate() {
            names[i] = name;
        }
        for (i, &gid) in global_ids.iter().enumerate() {
            local_to_global[i] = gid;
        }
        Some(Self {
            candidate_actions: names,
            local_to_global,
            n_candidates: n,
        })
    }

    /// Number of live candidate actions (always `<= 16`).
    #[inline]
    pub fn n_candidates(&self) -> usize {
        self.n_candidates
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n_candidates
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_candidates == 0
    }

    #[inline]
    pub fn actions(&self) -> &[ActionId] {
        let safe_len = self.n_candidates.min(16);
        &self.local_to_global[..safe_len]
    }

    #[inline]
    pub fn candidate_names(&self) -> &[&'a str] {
        let safe_len = self.n_candidates.min(16);
        &self.candidate_actions[..safe_len]
    }

    #[inline]
    pub fn to_global(&self, local_idx: usize) -> Option<ActionId> {
        let safe_len = self.n_candidates.min(16);
        if local_idx < safe_len {
            Some(self.local_to_global[local_idx])
        } else {
            None
        }
    }

    #[inline]
    pub fn to_local(&self, global_id: ActionId) -> Option<usize> {
        let safe_len = self.n_candidates.min(16);
        self.local_to_global[..safe_len]
            .iter()
            .position(|&id| id == global_id)
    }

    /// Filter frame down to a subset of feasible ActionIds while maintaining alignment.
    pub fn filter_by_actions(&self, feasible: &[ActionId]) -> Option<Self> {
        let mut new_names = [""; 16];
        let mut new_ids = [ActionId(0); 16];
        let mut count = 0;

        for &act in feasible {
            if let Some(loc) = self.to_local(act) {
                if count < 16 {
                    new_names[count] = self.candidate_actions[loc];
                    new_ids[count] = act;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return None;
        }

        Some(Self {
            candidate_actions: new_names,
            local_to_global: new_ids,
            n_candidates: count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_state_alignment_and_operations() {
        let s1 = LatentState::<64>::default();
        assert_eq!(std::mem::align_of::<LatentState<64>>(), 64);
        assert_eq!(std::mem::size_of::<LatentState<64>>(), 256);
        assert_eq!(s1.l2_norm(), 0.0);

        let mut v = [0.0_f32; 64];
        v[0] = 3.0;
        v[1] = 4.0;
        let s2 = LatentState::<64>::from_slice(&v).unwrap();
        assert_eq!(s2.l2_norm(), 5.0);

        let s2_norm = s2.normalized();
        assert!((s2_norm.l2_norm() - 1.0).abs() < 1e-6);
        assert!((s2.cosine_similarity(&s2_norm) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_foundation_latent_properties() {
        assert_eq!(std::mem::align_of::<FoundationLatent>(), 64);
        assert_eq!(std::mem::size_of::<FoundationLatent>(), 16384);

        let mut fl = Box::new(FoundationLatent::default());
        assert_eq!(fl.dim(), 4096);
        assert_eq!(fl.l2_norm(), 0.0);

        fl.values[0] = 3.0;
        fl.values[4095] = 4.0;
        assert_eq!(fl.l2_norm(), 5.0);

        fl.normalize_in_place();
        assert!((fl.l2_norm() - 1.0).abs() < 1e-6);
        assert!((fl.values[0] - 0.6).abs() < 1e-6);
        assert!((fl.values[4095] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_entropy_properties() {
        assert_eq!(NormalizedEntropy::ZERO.value(), 0.0);
        assert_eq!(NormalizedEntropy::ONE.value(), 1.0);

        let h_degenerate = NormalizedEntropy::from_probabilities(&[1.0]);
        assert_eq!(h_degenerate.value(), 0.0);

        let h_one_hot = NormalizedEntropy::from_probabilities(&[1.0, 0.0, 0.0]);
        assert_eq!(h_one_hot.value(), 0.0);

        let h_uniform = NormalizedEntropy::from_probabilities(&[0.5, 0.5]);
        assert!((h_uniform.value() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_local_action_frame_filter() {
        let names = ["ask", "route", "compact"];
        let ids = [ActionId(10), ActionId(20), ActionId(30)];
        let frame = LocalActionFrame::new(&names, &ids).unwrap();
        assert_eq!(frame.len(), 3);
        assert_eq!(frame.candidate_names(), &["ask", "route", "compact"]);

        let filtered = frame
            .filter_by_actions(&[ActionId(30), ActionId(10)])
            .unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered.candidate_names(), &["compact", "ask"]);
        assert_eq!(filtered.actions(), &[ActionId(30), ActionId(10)]);
    }

    #[test]
    fn local_action_frame_n_candidates_accessor() {
        let names = ["a", "b", "c"];
        let ids = [ActionId(7), ActionId(8), ActionId(9)];
        let frame = LocalActionFrame::new(&names, &ids).unwrap();
        assert_eq!(frame.n_candidates(), 3);
        assert_eq!(frame.n_candidates(), frame.len());
        let too_many = ["x"; 17];
        assert!(LocalActionFrame::new(&too_many, &[ActionId(0); 17]).is_none());
    }
}
