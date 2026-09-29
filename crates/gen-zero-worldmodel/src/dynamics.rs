//! gen-zero-worldmodel Latent Dynamics engine implementing WorldModelDynamics trait.
//!
//! The error type is `CoreError` so the model plugs straight into the planner's
//! `Arc<dyn WorldModelDynamics<Error = CoreError>>`. Failures are still raised as
//! `WorldModelError` and converted by `From<WorldModelError> for CoreError`.

use crate::error::WorldModelError;
use gen_zero_core::{ActionId, CoreError, FullLatent, SafetyEstimate, WorldModelDynamics};

/// L2 norm past which a transition ends the episode (`done`). This is the model's
/// only terminal condition: it marks divergence, so every `done` is a hazard.
pub const DONE_NORM: f32 = 100.0;

/// Source tag of [`LatentDynamicsWorldModel`]'s safety estimate.
pub const SAFETY_SOURCE_NORM_MARGIN: &str = "latent_norm_boundary_margin";

/// Latent Dynamics World Model.
/// Simulates environment state transitions in continuous latent space R^1024.
#[derive(Debug, Clone)]
pub struct LatentDynamicsWorldModel {
    /// Hidden dimension (default 1024)
    dim: usize,
    /// Residual scale factor
    residual_scale: f32,
    /// Action modulation weight
    action_scale: f32,
}

impl Default for LatentDynamicsWorldModel {
    fn default() -> Self {
        Self {
            dim: 1024,
            residual_scale: 0.95,
            action_scale: 0.05,
        }
    }
}

impl LatentDynamicsWorldModel {
    pub fn new(residual_scale: f32, action_scale: f32) -> Self {
        Self {
            dim: 1024,
            residual_scale: residual_scale.clamp(0.1, 1.0),
            action_scale,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl WorldModelDynamics for LatentDynamicsWorldModel {
    type Error = CoreError;

    /// Single-step forward transition in continuous latent space R^1024.
    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), Self::Error> {
        let mut next_values = [0.0_f32; 1024];
        let state_slice = state.as_slice();

        // Deterministic action perturbation encoding based on ActionId bits
        let a_seed = action.0 as f32;
        let mut reward_acc = 0.0_f32;

        for i in 0..1024 {
            let phase = (i as f32) * 0.05 + a_seed * 0.17;
            let action_delta = phase.sin() * self.action_scale;

            // Residual transition: next = residual_scale * curr + action_delta
            let next_val = state_slice[i] * self.residual_scale + action_delta;
            next_values[i] = next_val;
            reward_acc += next_val * 0.001;
        }

        let next_state = FullLatent {
            values: next_values,
        };
        let reward = reward_acc.clamp(-1.0, 1.0);
        if !reward.is_finite() || !next_state.values.iter().all(|x| x.is_finite()) {
            return Err(WorldModelError::NumericalDivergence.into());
        }
        // Episode terminates if norm drifts beyond threshold
        let done = next_state.l2_norm() > DONE_NORM;

        Ok((next_state, reward, done))
    }

    /// Batch transition kernel for parallel search rollouts (zero heap allocation).
    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next_states: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), Self::Error> {
        let n = states.len();
        if actions.len() != n || next_states.len() != n || rewards.len() != n || dones.len() != n {
            return Err(WorldModelError::DimensionMismatch {
                expected: n,
                actual: actions.len(),
            }
            .into());
        }

        for i in 0..n {
            let (next_s, r, d) = self.step(&states[i], actions[i])?;
            next_states[i] = next_s;
            rewards[i] = r;
            dones[i] = d;
        }

        Ok(())
    }

    /// Margin to the termination boundary: `1 - ||s'|| / DONE_NORM`, clamped to
    /// `[0, 1]`, and 0 on `done`. It is exact with respect to this model's own
    /// terminal rule but it is not fit to observed outcomes, so `calibrated` is false.
    fn safety_estimate(
        &self,
        next_state: &FullLatent,
        _reward: f32,
        done: bool,
    ) -> Option<SafetyEstimate> {
        let norm = next_state.l2_norm();
        let safe_prob = if done || !norm.is_finite() {
            0.0
        } else {
            (1.0 - norm / DONE_NORM).clamp(0.0, 1.0)
        };
        Some(SafetyEstimate {
            safe_prob,
            calibrated: false,
            source: SAFETY_SOURCE_NORM_MARGIN,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_matches_scalar_transitions_and_terminal_boundary() {
        let model = LatentDynamicsWorldModel::new(1.0, 0.0);
        let mut boundary = FullLatent::zeros();
        boundary.values[0] = DONE_NORM;
        let mut outside = boundary.clone();
        outside.values[0] += 1.0;
        let states = [boundary, outside];
        let actions = [ActionId(0), ActionId(u32::MAX)];
        let mut next = [FullLatent::zeros(), FullLatent::zeros()];
        let mut rewards = [0.0; 2];
        let mut dones = [false; 2];
        model
            .step_batch(&states, &actions, &mut next, &mut rewards, &mut dones)
            .unwrap();
        for i in 0..2 {
            let (expected, reward, done) = model.step(&states[i], actions[i]).unwrap();
            assert_eq!(next[i].values, expected.values);
            assert_eq!(rewards[i], reward);
            assert_eq!(dones[i], done);
        }
        assert_eq!(dones, [false, true]);
    }

    #[test]
    fn repeated_unforced_dynamics_contracts() {
        let model = LatentDynamicsWorldModel::new(0.95, 0.0);
        let mut state = FullLatent {
            values: [1.0; 1024],
        };
        for _ in 0..1000 {
            let previous = state.l2_norm();
            let (next, reward, done) = model.step(&state, ActionId(0)).unwrap();
            assert!(next.l2_norm() <= previous);
            assert!(reward.is_finite());
            assert!(!done);
            state = next;
        }
        assert!(state.values.iter().all(|x| x.abs() < 1e-20));
    }

    #[test]
    fn test_latent_dynamics_transition() {
        let model = LatentDynamicsWorldModel::default();
        let s0 = FullLatent::zeros();
        let a0 = ActionId(42);

        let (s1, reward, done) = model.step(&s0, a0).unwrap();
        assert!(!done);
        assert!(s1.l2_norm() > 0.0);
        assert!(reward.is_finite());

        // Batch step test
        let mut next_states = vec![FullLatent::zeros(); 2];
        let mut rewards = vec![0.0_f32; 2];
        let mut dones = vec![false; 2];

        model
            .step_batch(
                &[s0.clone(), s1],
                &[ActionId(1), ActionId(2)],
                &mut next_states,
                &mut rewards,
                &mut dones,
            )
            .unwrap();

        assert_eq!(next_states.len(), 2);
    }

    #[test]
    fn safety_estimate_is_the_uncalibrated_norm_margin() {
        let model = LatentDynamicsWorldModel::default();
        let (s1, r, d) = model.step(&FullLatent::zeros(), ActionId(1)).unwrap();
        let est = model.safety_estimate(&s1, r, d).unwrap();
        assert!(!est.calibrated);
        assert_eq!(est.source, SAFETY_SOURCE_NORM_MARGIN);
        assert!((est.safe_prob - (1.0 - s1.l2_norm() / DONE_NORM)).abs() < 1e-6);
        assert_eq!(model.safety_estimate(&s1, r, true).unwrap().safe_prob, 0.0);
    }

    #[test]
    fn test_non_finite_state_is_an_error_not_a_state() {
        let model = LatentDynamicsWorldModel::default();
        let bad = FullLatent {
            values: [f32::NAN; 1024],
        };
        assert!(matches!(
            model.step(&bad, ActionId(1)),
            Err(CoreError::NumericalInstability(_))
        ));
    }
}
