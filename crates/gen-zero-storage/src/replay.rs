//! Columnar Causal Replay Buffer with Causal PER and SIMD vector batches.
//!
//! Features:
//! - Contiguous columnar arrays for states, actions, rewards, next states, and dones
//! - Fenwick tree for O(log N) prioritized causal sampling
//! - Causal priority computation: P(i) \propto (|TD| + \gamma * ||\Delta z||_causal)^\alpha
//! - Batch sampling with importance sampling weights

use crate::error::StorageError;
use crate::fenwick::FenwickTree;
use gen_zero_core::{ActionId, FullLatent};

/// A single step of causal trajectory transition.
#[derive(Clone, Debug)]
pub struct TrajectoryStep {
    pub step_id: u64,
    pub state: FullLatent,
    pub action: ActionId,
    pub reward: f32,
    pub next_state: FullLatent,
    pub done: bool,
    pub td_error: f32,
    pub causal_shock: f32, // ||\Delta z|| causal intervention magnitude
}

/// Sampled batch from the causal replay buffer.
pub struct CausalSampleBatch {
    pub indices: Vec<usize>,
    pub step_ids: Vec<u64>,
    pub states: Vec<FullLatent>,
    pub actions: Vec<ActionId>,
    pub rewards: Vec<f32>,
    pub next_states: Vec<FullLatent>,
    pub dones: Vec<bool>,
    pub weights: Vec<f32>, // Normalized importance sampling weights
}

/// Columnar Causal Replay Buffer.
pub struct ColumnarCausalReplayBuffer {
    capacity: usize,
    size: usize,
    cursor: usize,

    // Columnar storage
    step_ids: Vec<u64>,
    states: Vec<FullLatent>,
    actions: Vec<ActionId>,
    rewards: Vec<f32>,
    next_states: Vec<FullLatent>,
    dones: Vec<bool>,
    priorities: Vec<f32>,

    // Prioritized sampling
    fenwick: FenwickTree,
    alpha: f32,        // Priority exponent: P(i) \propto p_i^\alpha
    causal_gamma: f32, // Causal shock multiplier: p_i = |TD| + gamma * causal_shock + eps
    max_priority: f32,
}

impl ColumnarCausalReplayBuffer {
    /// Create a new columnar causal replay buffer with fixed capacity.
    pub fn new(capacity: usize, alpha: f32, causal_gamma: f32) -> Self {
        assert!(capacity > 0, "Capacity must be > 0");
        Self {
            capacity,
            size: 0,
            cursor: 0,
            step_ids: vec![0; capacity],
            states: vec![FullLatent::default(); capacity],
            actions: vec![ActionId(0); capacity],
            rewards: vec![0.0; capacity],
            next_states: vec![FullLatent::default(); capacity],
            dones: vec![false; capacity],
            priorities: vec![0.0; capacity],
            fenwick: FenwickTree::new(capacity),
            alpha,
            causal_gamma,
            max_priority: 1.0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Compute composite causal priority with strict NaN/Inf guards:
    /// (|td_error| + gamma * causal_shock + 1e-5)^alpha
    #[inline]
    fn compute_priority(&self, td_error: f32, causal_shock: f32) -> Result<f32, StorageError> {
        if !td_error.is_finite() || !causal_shock.is_finite() {
            return Err(StorageError::InvalidPriority(
                "td_error and causal_shock must be finite",
            ));
        }
        if !self.alpha.is_finite()
            || self.alpha < 0.0
            || !self.causal_gamma.is_finite()
            || self.causal_gamma < 0.0
        {
            return Err(StorageError::InvalidPriority(
                "alpha and causal_gamma must be finite and nonnegative",
            ));
        }
        let base = td_error.abs() + self.causal_gamma * causal_shock.abs() + 1e-5;
        let p = base.powf(self.alpha);
        if !base.is_finite() || !p.is_finite() || p <= 0.0 {
            return Err(StorageError::InvalidPriority(
                "computed priority must be finite and positive",
            ));
        }
        Ok(p.clamp(1e-5, 1e5))
    }

    /// Push a new trajectory step into the buffer.
    pub fn push(&mut self, step: TrajectoryStep) -> Result<(), StorageError> {
        let idx = self.cursor;

        let priority = self.compute_priority(step.td_error, step.causal_shock)?;
        let priority_f64 = priority as f64;

        let old_priority_f64 = self.priorities[idx] as f64;
        let delta = priority_f64 - old_priority_f64;

        // Write columnar data
        self.step_ids[idx] = step.step_id;
        self.states[idx] = step.state;
        self.actions[idx] = step.action;
        self.rewards[idx] = step.reward;
        self.next_states[idx] = step.next_state;
        self.dones[idx] = step.done;
        self.priorities[idx] = priority;

        if priority > self.max_priority {
            self.max_priority = priority;
        }

        // Update Fenwick tree
        self.fenwick.update(idx, delta);

        self.cursor = (self.cursor + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
        Ok(())
    }

    /// Update transition priorities after a gradient step with step_id validation against ring-buffer dirty writes.
    pub fn update_priorities(
        &mut self,
        indices: &[usize],
        expected_step_ids: &[u64],
        td_errors: &[f32],
        causal_shocks: &[f32],
    ) -> Result<(), StorageError> {
        if indices.len() != expected_step_ids.len()
            || indices.len() != td_errors.len()
            || indices.len() != causal_shocks.len()
        {
            return Err(StorageError::InvalidPriority(
                "priority update lengths differ",
            ));
        }
        // Validate the entire batch before mutating any slot, including stale updates.
        let priorities = td_errors
            .iter()
            .zip(causal_shocks)
            .map(|(&td, &shock)| self.compute_priority(td, shock))
            .collect::<Result<Vec<_>, _>>()?;

        for (&idx, &step_id) in indices.iter().zip(expected_step_ids) {
            if idx >= self.size || self.step_ids[idx] != step_id {
                return Err(StorageError::InvalidPriority(
                    "priority update targets an absent or overwritten step",
                ));
            }
        }
        for (&idx, &p) in indices.iter().zip(&priorities) {
            let delta = p as f64 - self.priorities[idx] as f64;
            self.priorities[idx] = p;
            self.fenwick.update(idx, delta);
            self.max_priority = self.max_priority.max(p);
        }
        Ok(())
    }

    /// Sample a prioritized batch using Fenwick tree quantile binary search.
    /// Importance sampling weights: w_i = (N * P(i))^{-beta} / max(w)
    pub fn sample(&self, batch_size: usize, beta: f32) -> Result<CausalSampleBatch, StorageError> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if batch_size == 0 || !beta.is_finite() || !(0.0..=1.0).contains(&beta) {
            return Err(StorageError::InvalidSampleParameters(
                "batch_size must be positive and beta finite in [0, 1]",
            ));
        }

        if self.size == 0 {
            return Err(StorageError::BufferEmpty);
        }

        let total_p = self.fenwick.total_sum();
        if !total_p.is_finite() || total_p <= 0.0 {
            return Err(StorageError::InvalidSampleParameters(
                "priority sum must be finite and positive",
            ));
        }

        let segment = total_p / (batch_size as f64);
        let mut indices = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        let mut max_weight = 0.0_f32;
        let n_f64 = self.size as f64;

        for i in 0..batch_size {
            // True stochastic stratified sampling within segment: u ~ U(a, b)
            let a = segment * (i as f64);
            let b = segment * ((i + 1) as f64);
            let u: f64 = rng.gen_range(0.0..1.0);
            let target = a + u * (b - a);

            let idx = self
                .fenwick
                .find_prefix_quantile(target)?
                .min(self.size - 1);
            indices.push(idx);

            // Compute importance sampling weight
            let p_i = (self.priorities[idx] as f64) / total_p;
            if !p_i.is_finite() || p_i <= 0.0 {
                return Err(StorageError::InvalidSampleParameters(
                    "sampled probability must be finite and positive",
                ));
            }
            let w = ((n_f64 * p_i).powf(-beta as f64)) as f32;
            if !w.is_finite() || w <= 0.0 {
                return Err(StorageError::InvalidSampleParameters(
                    "sampled weight must be finite and positive",
                ));
            }
            weights.push(w);
            if w > max_weight {
                max_weight = w;
            }
        }

        // Normalize weights by max_weight
        if max_weight > 0.0 {
            for w in weights.iter_mut() {
                *w /= max_weight;
            }
        }

        // Gather columnar data into batch
        let mut step_ids = Vec::with_capacity(batch_size);
        let mut states = Vec::with_capacity(batch_size);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);

        for &idx in &indices {
            step_ids.push(self.step_ids[idx]);
            states.push(self.states[idx].clone());
            actions.push(self.actions[idx]);
            rewards.push(self.rewards[idx]);
            next_states.push(self.next_states[idx].clone());
            dones.push(self.dones[idx]);
        }

        Ok(CausalSampleBatch {
            indices,
            step_ids,
            states,
            actions,
            rewards,
            next_states,
            dones,
            weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(td_error: f32, causal_shock: f32) -> TrajectoryStep {
        TrajectoryStep {
            step_id: 7,
            state: FullLatent::default(),
            action: ActionId(0),
            reward: 0.0,
            next_state: FullLatent::default(),
            done: false,
            td_error,
            causal_shock,
        }
    }

    #[test]
    fn invalid_priorities_leave_ring_and_batch_unchanged() {
        let mut buffer = ColumnarCausalReplayBuffer::new(2, 0.6, 0.5);
        buffer.push(step(1.0, 0.2)).unwrap();
        buffer.push(step(2.0, 0.3)).unwrap();
        let before = buffer.priorities.clone();
        let total = buffer.fenwick.total_sum();
        let cursor = buffer.cursor;
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for (td, shock) in [(invalid, 0.2), (1.0, invalid)] {
                assert!(matches!(
                    buffer.push(step(td, shock)),
                    Err(StorageError::InvalidPriority(_))
                ));
                assert!(matches!(
                    buffer.update_priorities(&[0, 1], &[7, 7], &[9.0, td], &[0.1, shock]),
                    Err(StorageError::InvalidPriority(_))
                ));
                assert_eq!(buffer.priorities, before);
                assert_eq!(buffer.fenwick.total_sum(), total);
                assert_eq!(buffer.cursor, cursor);
                assert_eq!(buffer.len(), 2);
            }
        }
        assert!(matches!(
            buffer.push(step(f32::MAX, f32::MAX)),
            Err(StorageError::InvalidPriority(_))
        ));
        for (indices, ids, td, shock) in [
            (vec![0, 1], vec![7, 99], vec![9.0, 1.0], vec![0.1, 0.2]),
            (vec![0, 2], vec![7, 7], vec![9.0, 1.0], vec![0.1, 0.2]),
            (vec![0], vec![], vec![1.0], vec![0.2]),
        ] {
            assert!(matches!(
                buffer.update_priorities(&indices, &ids, &td, &shock),
                Err(StorageError::InvalidPriority(_))
            ));
            assert_eq!(buffer.priorities, before);
            assert_eq!(buffer.fenwick.total_sum(), total);
        }
        buffer
            .update_priorities(&[0], &[7], &[3.0], &[0.4])
            .unwrap();
        assert_ne!(buffer.priorities, before);
        assert!(buffer.sample(2, 0.5).is_ok());
    }

    #[test]
    fn test_causal_replay_buffer_push_and_sample() {
        let mut buffer = ColumnarCausalReplayBuffer::new(100, 0.6, 0.5);

        for i in 0..50 {
            let mut s = FullLatent::default();
            s.as_mut_slice()[0] = i as f32;
            let mut ns = s.clone();
            ns.as_mut_slice()[0] += 1.0;

            buffer
                .push(TrajectoryStep {
                    step_id: i as u64,
                    state: s,
                    action: ActionId(i as u32),
                    reward: 1.0,
                    next_state: ns,
                    done: false,
                    td_error: (i as f32) * 0.1,
                    causal_shock: 0.2,
                })
                .unwrap();
        }

        assert_eq!(buffer.len(), 50);
        for beta in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(buffer.sample(8, beta).is_err());
        }
        assert!(buffer.sample(0, 0.4).is_err());

        let batch = buffer.sample(8, 0.4).unwrap();
        assert_eq!(batch.indices.len(), 8);
        assert_eq!(batch.states.len(), 8);
        assert_eq!(batch.actions.len(), 8);
        assert_eq!(batch.weights.len(), 8);

        for &w in &batch.weights {
            assert!((0.0..=1.0001).contains(&w));
        }
    }
}
