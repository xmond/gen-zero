//! Symplectic world model: [`WorldModelDynamics`] on a Hamiltonian phase space.
//!
//! The latent `z` is read as a phase point `(q, p)` with `q = z[..512]` and
//! `p = z[512..]`. An action `a` selects the Hamiltonian of its step:
//!
//! ```text
//! H_a(q, p) = 1/2 |p|^2 + k/2 |q - c_a|^2
//! ```
//!
//! `c_a` is an equilibrium shift derived from the action id, so the action acts
//! as a constant external force `k c_a` on the unit-stiffness well. One step is
//! one kick-drift-kick Stormer-Verlet step of [`SymplecticIntegrator`] under `H_a`:
//! volume-preserving and time-reversible.
//!
//! What is conserved: `H_a` of one action, up to the integrator's bounded
//! `O(dt^2)` error. When the action changes between steps the Hamiltonian
//! changes too, so energy is only comparable across steps that share an action.
//! Every transition reports `H_a` before and after the step for that reason.
//!
//! This model is **not trained and not calibrated**. The action shift `c_a`,
//! the reward and the `done` rule are hand-set priors, not fits to any data.

use crate::dynamics::{DONE_NORM, SAFETY_SOURCE_NORM_MARGIN};
use crate::error::WorldModelError;
use crate::symplectic::{PhaseState, SymplecticIntegrator};
use gen_zero_core::{ActionId, CoreError, FullLatent, SafetyEstimate, WorldModelDynamics};

const HALF: usize = LATENT_HALF;
pub(crate) use crate::contact::LATENT_HALF;

/// Equilibrium shift `c_a` of an action: a deterministic phase encoding shared by
/// the symplectic and the conformal (contact) models, so both read one action id
/// the same way.
pub(crate) fn action_centre(action: ActionId, action_scale: f32) -> [f32; HALF] {
    let seed = action.0 as f32;
    let mut c = [0.0_f32; HALF];
    for (i, ci) in c.iter_mut().enumerate() {
        *ci = ((i as f32) * 0.05 + seed * 0.17).sin() * action_scale;
    }
    c
}

/// One transition of [`SymplecticWorldModelDynamics`] with its energy ledger.
#[derive(Clone, Debug)]
pub struct PhaseTransition {
    pub next: FullLatent,
    pub reward: f32,
    pub done: bool,
    /// `H_a` of the step's action, at the start state.
    pub energy_before: f64,
    /// `H_a` of the step's action, at the next state.
    pub energy_after: f64,
}

impl PhaseTransition {
    /// Integrator error of this step: `H_a(next) - H_a(start)`.
    #[inline]
    pub fn energy_drift(&self) -> f64 {
        self.energy_after - self.energy_before
    }
}

/// Hamiltonian world model stepped by Stormer-Verlet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymplecticWorldModelDynamics {
    integrator: SymplecticIntegrator,
    stiffness: f32,
    action_scale: f32,
}

impl SymplecticWorldModelDynamics {
    pub const DEFAULT_DT: f32 = 0.01;
    pub const DEFAULT_STIFFNESS: f32 = 1.0;
    pub const DEFAULT_ACTION_SCALE: f32 = 0.05;

    /// Refuses a step that Stormer-Verlet cannot keep stable: it is stable on a
    /// well of stiffness `k` only for `k dt^2 < 4`.
    pub fn new(dt: f32, stiffness: f32, action_scale: f32) -> Result<Self, WorldModelError> {
        let finite = dt.is_finite() && stiffness.is_finite() && action_scale.is_finite();
        if !finite || dt <= 0.0 || stiffness <= 0.0 || action_scale < 0.0 {
            return Err(WorldModelError::NumericalDivergence);
        }
        if f64::from(stiffness) * f64::from(dt) * f64::from(dt) >= 4.0 {
            return Err(WorldModelError::NumericalDivergence);
        }
        Ok(Self {
            integrator: SymplecticIntegrator::new(dt),
            stiffness,
            action_scale,
        })
    }

    #[inline]
    pub fn dt(&self) -> f32 {
        self.integrator.dt
    }

    /// Equilibrium shift `c_a` of an action. Same deterministic phase encoding
    /// as the residual prior, so both models read one action id the same way.
    pub fn action_centre(&self, action: ActionId) -> [f32; HALF] {
        action_centre(action, self.action_scale)
    }

    /// `H_a(q, p) = 1/2 |p|^2 + k/2 |q - c_a|^2`, summed in f64 so the ledger
    /// is not dominated by f32 rounding of a 512-term sum.
    pub fn hamiltonian(&self, state: &PhaseState<HALF>, centre: &[f32; HALF]) -> f64 {
        let kinetic: f64 = state.p.iter().map(|&p| f64::from(p).powi(2)).sum();
        let potential: f64 = state
            .q
            .iter()
            .zip(centre)
            .map(|(&q, &c)| (f64::from(q) - f64::from(c)).powi(2))
            .sum();
        0.5 * kinetic + 0.5 * f64::from(self.stiffness) * potential
    }

    /// Unforced energy `1/2 |p|^2 + k/2 |q|^2`: the energy stored in the well.
    fn stored_energy(&self, state: &PhaseState<HALF>) -> f64 {
        self.hamiltonian(state, &[0.0; HALF])
    }

    pub fn phase_of(latent: &FullLatent) -> PhaseState<HALF> {
        let z = latent.as_slice();
        let mut state = PhaseState::<HALF>::zeros();
        state.q.copy_from_slice(&z[..HALF]);
        state.p.copy_from_slice(&z[HALF..2 * HALF]);
        state
    }

    /// One step under `H_a` with the energy ledger.
    ///
    /// Reward is the stored energy the step removes from the well,
    /// `E(z) - E(z')`, clamped to `[-1, 1]`: a hand-set objective that prefers
    /// actions that calm the oscillator. It is not learned.
    pub fn transition(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<PhaseTransition, WorldModelError> {
        if !state.as_slice().iter().all(|x| x.is_finite()) {
            return Err(WorldModelError::NumericalDivergence);
        }
        let centre = self.action_centre(action);
        let mut phase = Self::phase_of(state);
        let energy_before = self.hamiltonian(&phase, &centre);
        let stored_before = self.stored_energy(&phase);
        let k = self.stiffness;
        self.integrator.step(&mut phase, |q, g| {
            for ((gi, &qi), &ci) in g.iter_mut().zip(q).zip(&centre) {
                *gi = k * (qi - ci);
            }
        })?;
        let energy_after = self.hamiltonian(&phase, &centre);
        let reward = (stored_before - self.stored_energy(&phase)).clamp(-1.0, 1.0) as f32;
        if !energy_after.is_finite() || !reward.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }
        let mut next = FullLatent::zeros();
        next.values[..HALF].copy_from_slice(&phase.q);
        next.values[HALF..2 * HALF].copy_from_slice(&phase.p);
        let done = next.l2_norm() > DONE_NORM;
        Ok(PhaseTransition {
            next,
            reward,
            done,
            energy_before,
            energy_after,
        })
    }
}

impl Default for SymplecticWorldModelDynamics {
    fn default() -> Self {
        Self::new(
            Self::DEFAULT_DT,
            Self::DEFAULT_STIFFNESS,
            Self::DEFAULT_ACTION_SCALE,
        )
        .expect("default parameters satisfy k dt^2 < 4")
    }
}

impl WorldModelDynamics for SymplecticWorldModelDynamics {
    type Error = CoreError;

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), Self::Error> {
        let t = self.transition(state, action)?;
        Ok((t.next, t.reward, t.done))
    }

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
            let (next, r, d) = self.step(&states[i], actions[i])?;
            next_states[i] = next;
            rewards[i] = r;
            dones[i] = d;
        }
        Ok(())
    }

    /// Same uncalibrated margin to the `DONE_NORM` ball as the residual prior.
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

    fn start() -> FullLatent {
        let mut z = FullLatent::zeros();
        for (i, v) in z.values.iter_mut().enumerate() {
            *v = 0.1 * ((i as f32) * 0.013).cos();
        }
        z
    }

    #[test]
    fn fixed_action_conserves_its_hamiltonian_over_ten_steps() {
        let model = SymplecticWorldModelDynamics::default();
        let a = ActionId(7);
        let centre = model.action_centre(a);
        let z0 = start();
        let h0 = model.hamiltonian(&SymplecticWorldModelDynamics::phase_of(&z0), &centre);
        let mut z = z0.clone();
        for _ in 0..10 {
            let t = model.transition(&z, a).unwrap();
            assert!(t.energy_drift().abs() / h0 < 1e-4, "{}", t.energy_drift());
            z = t.next;
        }
        let h10 = model.hamiltonian(&SymplecticWorldModelDynamics::phase_of(&z), &centre);
        assert!(((h10 - h0) / h0).abs() < 1e-4, "h0 {h0} h10 {h10}");
        // Not a frozen state: the flow moved q and p.
        let moved: f32 = z0
            .values
            .iter()
            .zip(z.values.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(moved > 1e-3, "moved {moved}");
    }

    /// The 1e-4 bound discriminates: explicit Euler on the same forces and
    /// the same `dt` gains energy by a factor `1 + k dt^2` per step and
    /// breaks the bound within ten steps.
    #[test]
    fn explicit_euler_would_fail_the_same_energy_bound() {
        let model = SymplecticWorldModelDynamics::default();
        let a = ActionId(7);
        let centre = model.action_centre(a);
        let mut state = SymplecticWorldModelDynamics::phase_of(&start());
        let h0 = model.hamiltonian(&state, &centre);
        let dt = model.dt();
        for _ in 0..10 {
            let (q, p) = (state.q, state.p);
            for i in 0..HALF {
                state.q[i] = q[i] + dt * p[i];
                state.p[i] = p[i] - dt * (q[i] - centre[i]);
            }
        }
        let rel = ((model.hamiltonian(&state, &centre) - h0) / h0).abs();
        assert!(rel > 1e-4, "euler drift {rel}");
    }

    #[test]
    fn different_actions_reach_different_states() {
        let model = SymplecticWorldModelDynamics::default();
        let (a, _, _) = model.step(&start(), ActionId(1)).unwrap();
        let (b, _, _) = model.step(&start(), ActionId(2)).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn step_is_time_reversible() {
        let model = SymplecticWorldModelDynamics::default();
        let a = ActionId(3);
        let z0 = start();
        let (mut z, _, _) = model.step(&z0, a).unwrap();
        for v in &mut z.values[HALF..] {
            *v = -*v;
        }
        let (mut back, _, _) = model.step(&z, a).unwrap();
        for v in &mut back.values[HALF..] {
            *v = -*v;
        }
        for (x, y) in back.values.iter().zip(z0.values.iter()) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn unstable_or_non_finite_parameters_are_refused() {
        assert!(SymplecticWorldModelDynamics::new(2.0, 1.0, 0.05).is_err());
        assert!(SymplecticWorldModelDynamics::new(0.5, 16.0, 0.05).is_err());
        assert!(SymplecticWorldModelDynamics::new(0.0, 1.0, 0.05).is_err());
        assert!(SymplecticWorldModelDynamics::new(f32::NAN, 1.0, 0.05).is_err());
        assert!(SymplecticWorldModelDynamics::new(0.01, -1.0, 0.05).is_err());
        assert!(SymplecticWorldModelDynamics::new(0.01, 1.0, f32::INFINITY).is_err());
        assert!(SymplecticWorldModelDynamics::new(0.01, 1.0, 0.05).is_ok());
    }

    #[test]
    fn non_finite_state_is_an_error_not_a_state() {
        let model = SymplecticWorldModelDynamics::default();
        let bad = FullLatent {
            values: [f32::NAN; 1024],
        };
        assert!(matches!(
            model.step(&bad, ActionId(1)),
            Err(CoreError::NumericalInstability(_))
        ));
    }

    #[test]
    fn plugs_into_the_planner_trait_object() {
        let model: std::sync::Arc<dyn WorldModelDynamics<Error = CoreError>> =
            std::sync::Arc::new(SymplecticWorldModelDynamics::default());
        let (s1, r, d) = model.step(&start(), ActionId(5)).unwrap();
        assert!(!d && r.is_finite());
        assert!(!model.safety_estimate(&s1, r, d).unwrap().calibrated);
    }
}
