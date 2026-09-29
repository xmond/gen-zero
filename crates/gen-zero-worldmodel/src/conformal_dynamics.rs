//! Conformal symplectic world model: [`WorldModelDynamics`] on the contact manifold.
//!
//! Liouville's theorem makes a symplectic flow preserve phase volume exactly, so
//! the Hamiltonian model in [`crate::symplectic_dynamics`] can never contract.
//! A recurrent cognitive state is dissipative, so this model splits one step in
//! two parts:
//!
//! * the **conservative** part is the same Hamiltonian well as the symplectic
//!   model, `H_a(q, p) = 1/2 |p|^2 + k/2 |q - c_a|^2`, kept by the symmetric
//!   Strang split of [`ContactIntegrator`], which reduces to kick-drift-kick
//!   Stormer-Verlet when the damping vanishes;
//! * the **dissipative** part is the contact flow with a linear friction
//!   `dp/dt = -grad V - 2 gamma p`, `gamma >= 0`, applied as two half-step
//!   scalings of `p` inside the split.
//!
//! Damping convention. The request-level `gamma` is the conformal rate: every
//! `(q_i, p_i)` pair contracts its area by exactly `exp(-2 gamma dt)` per step,
//! and the whole `2N`-dimensional projection by `exp(-2 gamma N dt)`.
//! [`ContactIntegrator`] parameterises the same friction as `gamma_contact * p`,
//! so this model hands it `gamma_contact = 2 gamma`. Both numbers are reported
//! in every transition so the mapping is never implicit.
//!
//! The latent `z` holds only `(q, p)`: `q = z[..512]`, `p = z[512..]`. The
//! contact action `s` has no slot in the latent. It starts at zero on every
//! step and its value after the step is reported in the transition ledger.
//!
//! What is conserved: for `gamma = 0` this is the symplectic model, so `H_a` of
//! one action is conserved up to bounded `O(dt^2)` error. For `gamma > 0`
//! nothing is conserved; `H_a` decays and the ledger shows the decay.
//!
//! This model is **not trained and not calibrated**. Stiffness, action shift,
//! damping, reward and the `done` rule are hand-set priors, not fits to data.

use crate::contact::{spectral_radius_2x2, well_step_matrix, ContactIntegrator, ContactState};
use crate::dynamics::{DONE_NORM, SAFETY_SOURCE_NORM_MARGIN};
use crate::error::WorldModelError;
use crate::symplectic_dynamics::{action_centre, LATENT_HALF};
use gen_zero_core::{ActionId, CoreError, FullLatent, SafetyEstimate, WorldModelDynamics};

const HALF: usize = LATENT_HALF;

/// One transition of [`ConformalWorldModelDynamics`] with its energy and volume ledger.
#[derive(Clone, Debug)]
pub struct ContactTransition {
    pub next: FullLatent,
    pub reward: f32,
    pub done: bool,
    /// `H_a` of the step's action, at the start state.
    pub energy_before: f64,
    /// `H_a` of the step's action, at the next state.
    pub energy_after: f64,
    /// Contact action coordinate `s` after the step, starting from `s = 0`.
    pub contact_action: f32,
    /// Exact area factor of every `(q_i, p_i)` pair: `exp(-2 gamma dt)`.
    pub phase_volume_factor_per_pair: f64,
}

impl ContactTransition {
    /// `H_a(next) - H_a(start)`: integrator error for `gamma = 0`, dissipation otherwise.
    #[inline]
    pub fn energy_drift(&self) -> f64 {
        self.energy_after - self.energy_before
    }
}

/// Conformal symplectic (contact) world model stepped by the Strang split.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformalWorldModelDynamics {
    integrator: ContactIntegrator,
    stiffness: f32,
    action_scale: f32,
    /// Conformal damping rate `gamma` (friction `2 gamma p`).
    gamma: f32,
}

impl ConformalWorldModelDynamics {
    pub const DEFAULT_DT: f32 = 0.01;
    pub const DEFAULT_STIFFNESS: f32 = 1.0;
    pub const DEFAULT_ACTION_SCALE: f32 = 0.05;
    /// Default conformal rate: each pair loses `1 - exp(-0.01)` of its area per step.
    pub const DEFAULT_DAMPING: f32 = 0.5;

    /// Refuses, at load time, every parameter the split cannot run safely on:
    /// non-finite values, `dt <= 0`, `stiffness <= 0`, `action_scale < 0`,
    /// `gamma < 0`, and a step outside the strict stability band of the damped
    /// well, `k dt^2 d < 2 (1 + d^2)` with `d = exp(-gamma dt)`. For `gamma = 0`
    /// that band is Stormer-Verlet's `k dt^2 < 4`.
    pub fn new(
        dt: f32,
        stiffness: f32,
        action_scale: f32,
        gamma: f32,
    ) -> Result<Self, WorldModelError> {
        let finite = dt.is_finite()
            && stiffness.is_finite()
            && action_scale.is_finite()
            && gamma.is_finite();
        if !finite || dt <= 0.0 || stiffness <= 0.0 || action_scale < 0.0 || gamma < 0.0 {
            return Err(WorldModelError::NumericalDivergence);
        }
        let model = Self {
            integrator: ContactIntegrator::new(dt, 2.0 * gamma),
            stiffness,
            action_scale,
            gamma,
        };
        if !model.integrator.gamma.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }
        let d = model.half_damping();
        let scaled = model.scaled_step();
        let split_stable = scaled * scaled * d < 2.0 * (1.0 + d * d);
        let rate = model.contraction_rate();
        if !d.is_finite() || !split_stable || !rate.is_finite() || rate > 1.0 {
            return Err(WorldModelError::NumericalDivergence);
        }
        Ok(model)
    }

    #[inline]
    pub fn dt(&self) -> f32 {
        self.integrator.dt
    }

    #[inline]
    pub fn stiffness(&self) -> f32 {
        self.stiffness
    }

    /// Conformal damping rate `gamma` of the request.
    #[inline]
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// The friction coefficient handed to [`ContactIntegrator`]: `2 gamma`.
    #[inline]
    pub fn integrator_gamma(&self) -> f32 {
        self.integrator.gamma
    }

    /// Scaling of `p` in each dissipation half-step: `exp(-gamma dt)`.
    #[inline]
    fn half_damping(&self) -> f64 {
        (-f64::from(self.integrator.gamma) * f64::from(self.integrator.dt) * 0.5).exp()
    }

    /// The well of stiffness `k` stepped by `dt` is the unit well stepped by
    /// `dt sqrt(k)` (rescale `q` by `sqrt(k)`; the damping factor is unchanged).
    #[inline]
    fn scaled_step(&self) -> f64 {
        f64::from(self.integrator.dt) * f64::from(self.stiffness).sqrt()
    }

    /// Exact area factor of one `(q_i, p_i)` pair per step, `exp(-2 gamma dt)`.
    #[inline]
    pub fn phase_volume_factor_per_pair(&self) -> f64 {
        (-2.0 * f64::from(self.gamma) * f64::from(self.integrator.dt)).exp()
    }

    /// Exact volume factor of the whole `2N` projection per step, `exp(-2 gamma N dt)`.
    #[inline]
    pub fn phase_volume_factor(&self) -> f64 {
        (-2.0 * f64::from(self.gamma) * (HALF as f64) * f64::from(self.integrator.dt)).exp()
    }

    /// Spectral radius of the per-pair step map: `1` for `gamma = 0`, below `1` for a
    /// stable damped step.
    pub fn contraction_rate(&self) -> f64 {
        let d = self.half_damping();
        let m = well_step_matrix(self.scaled_step(), d);
        spectral_radius_2x2(m[0][0] + m[1][1], d * d)
    }

    /// Equilibrium shift `c_a`: the same encoding as the symplectic model.
    pub fn action_centre(&self, action: ActionId) -> [f32; HALF] {
        action_centre(action, self.action_scale)
    }

    /// `H_a(q, p) = 1/2 |p|^2 + k/2 |q - c_a|^2` in f64.
    pub fn hamiltonian(&self, state: &ContactState<HALF>, centre: &[f32; HALF]) -> f64 {
        let kinetic: f64 = state.p.iter().map(|&p| f64::from(p).powi(2)).sum();
        let potential: f64 = state
            .q
            .iter()
            .zip(centre)
            .map(|(&q, &c)| (f64::from(q) - f64::from(c)).powi(2))
            .sum();
        0.5 * kinetic + 0.5 * f64::from(self.stiffness) * potential
    }

    fn stored_energy(&self, state: &ContactState<HALF>) -> f64 {
        self.hamiltonian(state, &[0.0; HALF])
    }

    /// Read the latent as `(q, p)` with `s = 0`.
    pub fn contact_of(latent: &FullLatent) -> ContactState<HALF> {
        let z = latent.as_slice();
        let mut state = ContactState::<HALF>::zeros();
        state.q.copy_from_slice(&z[..HALF]);
        state.p.copy_from_slice(&z[HALF..2 * HALF]);
        state
    }

    /// One Strang step under `H_a` with friction `2 gamma p`, with the ledger.
    ///
    /// Reward is the stored energy the step removes from the well, clamped to
    /// `[-1, 1]`, the same hand-set objective as the symplectic model.
    pub fn transition(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<ContactTransition, WorldModelError> {
        if !state.as_slice().iter().all(|x| x.is_finite()) {
            return Err(WorldModelError::NumericalDivergence);
        }
        let centre = self.action_centre(action);
        let mut contact = Self::contact_of(state);
        let energy_before = self.hamiltonian(&contact, &centre);
        let stored_before = self.stored_energy(&contact);
        let k = self.stiffness;
        let potential = |q: &[f32; HALF]| -> f32 {
            let sum: f32 = q
                .iter()
                .zip(&centre)
                .map(|(&qi, &ci)| (qi - ci) * (qi - ci))
                .sum();
            0.5 * k * sum
        };
        self.integrator.step(&mut contact, potential, |q, g| {
            for ((gi, &qi), &ci) in g.iter_mut().zip(q).zip(&centre) {
                *gi = k * (qi - ci);
            }
        })?;
        let energy_after = self.hamiltonian(&contact, &centre);
        let reward = (stored_before - self.stored_energy(&contact)).clamp(-1.0, 1.0) as f32;
        if !energy_after.is_finite() || !reward.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }
        let mut next = FullLatent::zeros();
        next.values[..HALF].copy_from_slice(&contact.q);
        next.values[HALF..2 * HALF].copy_from_slice(&contact.p);
        let done = next.l2_norm() > DONE_NORM;
        Ok(ContactTransition {
            next,
            reward,
            done,
            energy_before,
            energy_after,
            contact_action: contact.s,
            phase_volume_factor_per_pair: self.phase_volume_factor_per_pair(),
        })
    }
}

impl Default for ConformalWorldModelDynamics {
    fn default() -> Self {
        Self::new(
            Self::DEFAULT_DT,
            Self::DEFAULT_STIFFNESS,
            Self::DEFAULT_ACTION_SCALE,
            Self::DEFAULT_DAMPING,
        )
        .expect("default parameters are inside the stability band")
    }
}

impl WorldModelDynamics for ConformalWorldModelDynamics {
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

    /// Same uncalibrated margin to the `DONE_NORM` ball as the other priors.
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
    use crate::symplectic_dynamics::SymplecticWorldModelDynamics;

    fn start() -> FullLatent {
        let mut z = FullLatent::zeros();
        for (i, v) in z.values.iter_mut().enumerate() {
            *v = 0.1 * ((i as f32) * 0.013).cos();
        }
        z
    }

    fn model(gamma: f32) -> ConformalWorldModelDynamics {
        ConformalWorldModelDynamics::new(0.01, 1.0, 0.05, gamma).unwrap()
    }

    #[test]
    fn zero_damping_conserves_the_hamiltonian_over_ten_steps() {
        let m = model(0.0);
        let a = ActionId(7);
        let centre = m.action_centre(a);
        let z0 = start();
        let h0 = m.hamiltonian(&ConformalWorldModelDynamics::contact_of(&z0), &centre);
        let mut z = z0.clone();
        for _ in 0..10 {
            let t = m.transition(&z, a).unwrap();
            assert!(t.energy_drift().abs() / h0 < 1e-4, "{}", t.energy_drift());
            assert_eq!(t.phase_volume_factor_per_pair, 1.0);
            z = t.next;
        }
        let h10 = m.hamiltonian(&ConformalWorldModelDynamics::contact_of(&z), &centre);
        assert!(((h10 - h0) / h0).abs() < 1e-4, "h0 {h0} h10 {h10}");
        assert_eq!(m.contraction_rate(), 1.0);
        let moved: f32 = z0
            .values
            .iter()
            .zip(z.values.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(moved > 1e-3, "moved {moved}");
    }

    /// With `gamma = 0` the split `V/2 D T D V/2` has `D = 1`, so it is exactly
    /// kick-drift-kick Stormer-Verlet: the two models must walk the same path.
    #[test]
    fn zero_damping_degenerates_to_the_symplectic_model() {
        let contact = model(0.0);
        let symplectic = SymplecticWorldModelDynamics::new(0.01, 1.0, 0.05).unwrap();
        let a = ActionId(3);
        let mut zc = start();
        let mut zs = start();
        for _ in 0..10 {
            zc = contact.transition(&zc, a).unwrap().next;
            zs = symplectic.transition(&zs, a).unwrap().next;
        }
        // Same operations in the same order, and the damping factor exp(0) is exactly
        // one, so the two paths agree bit for bit.
        assert_eq!(zc.values, zs.values);
    }

    /// The well of stiffness `k` is the unit well with step `dt sqrt(k)` after
    /// rescaling `q` by `sqrt(k)`: the measured Jacobian of a `k = 4` step must
    /// match `well_step_matrix(dt sqrt(k), d)` in those coordinates.
    #[test]
    fn stiffness_rescaling_matches_the_spectral_matrix() {
        let (dt, k, gamma) = (0.05_f32, 4.0_f32, 0.75_f32);
        let m = ConformalWorldModelDynamics::new(dt, k, 0.05, gamma).unwrap();
        let d = (-f64::from(gamma) * f64::from(dt)).exp();
        let expected = well_step_matrix(f64::from(dt) * f64::from(k).sqrt(), d);
        let a = ActionId(2);
        let base = start();
        let out0 = m.transition(&base, a).unwrap().next;
        let sk = f64::from(k).sqrt();
        for i in [3_usize, 100, 500] {
            let eps = 0.5_f32;
            let mut dq = base.clone();
            dq.values[i] += eps;
            let mut dp = base.clone();
            dp.values[HALF + i] += eps;
            let oq = m.transition(&dq, a).unwrap().next;
            let op = m.transition(&dp, a).unwrap().next;
            let col = |o: &FullLatent| {
                [
                    f64::from(o.values[i] - out0.values[i]) / f64::from(eps),
                    f64::from(o.values[HALF + i] - out0.values[HALF + i]) / f64::from(eps),
                ]
            };
            let (cq, cp) = (col(&oq), col(&op));
            // Jacobian in (q, p); in (sqrt(k) q, p) coordinates it is S J S^-1 with
            // S = diag(sqrt(k), 1).
            let scaled = [[cq[0], cp[0] * sk], [cq[1] / sk, cp[1]]];
            for r in 0..2 {
                for c in 0..2 {
                    assert!(
                        (scaled[r][c] - expected[r][c]).abs() < 1e-5,
                        "pair {i} entry ({r},{c}): {} vs {}",
                        scaled[r][c],
                        expected[r][c]
                    );
                }
            }
        }
        let det = expected[0][0] * expected[1][1] - expected[0][1] * expected[1][0];
        assert!((det - m.phase_volume_factor_per_pair()).abs() < 1e-12);
        assert!(m.contraction_rate() < 1.0);
    }

    /// The step is affine in `(q, p)`, so finite differences of basis vectors give
    /// the exact Jacobian. Its per-pair determinant must be `exp(-2 gamma dt)`.
    #[test]
    fn positive_damping_contracts_each_pair_by_exp_minus_two_gamma_dt() {
        for gamma in [0.5_f32, 2.0, 10.0] {
            let m = model(gamma);
            let a = ActionId(5);
            let base = start();
            let out0 = m.transition(&base, a).unwrap().next;
            let expected = (-2.0 * f64::from(gamma) * f64::from(m.dt())).exp();
            assert!(expected < 1.0);
            for i in [0_usize, 17, 255, 511] {
                let eps = 0.5_f32;
                let mut dq = base.clone();
                dq.values[i] += eps;
                let mut dp = base.clone();
                dp.values[HALF + i] += eps;
                let oq = m.transition(&dq, a).unwrap().next;
                let op = m.transition(&dp, a).unwrap().next;
                let col = |o: &FullLatent| {
                    [
                        f64::from(o.values[i] - out0.values[i]) / f64::from(eps),
                        f64::from(o.values[HALF + i] - out0.values[HALF + i]) / f64::from(eps),
                    ]
                };
                let (cq, cp) = (col(&oq), col(&op));
                let det = cq[0] * cp[1] - cq[1] * cp[0];
                assert!(
                    (det - expected).abs() < 1e-5,
                    "gamma {gamma} pair {i}: det {det} vs {expected}"
                );
            }
            assert!((m.phase_volume_factor_per_pair() - expected).abs() < 1e-12);
            assert!(m.contraction_rate() < 1.0);
        }
    }

    #[test]
    fn positive_damping_dissipates_energy_over_a_run() {
        let m = model(0.5);
        let a = ActionId(7);
        let centre = m.action_centre(a);
        let mut z = start();
        let h0 = m.hamiltonian(&ConformalWorldModelDynamics::contact_of(&z), &centre);
        for _ in 0..100 {
            let t = m.transition(&z, a).unwrap();
            assert!(t.contact_action.is_finite());
            z = t.next;
        }
        let h = m.hamiltonian(&ConformalWorldModelDynamics::contact_of(&z), &centre);
        // 100 steps of exp(-0.01) damping on p: the energy must have fallen clearly.
        assert!(h < 0.9 * h0, "h0 {h0} h {h}");
    }

    #[test]
    fn invalid_parameters_are_refused_at_load_time() {
        let new = ConformalWorldModelDynamics::new;
        assert!(new(0.01, 1.0, 0.05, -0.5).is_err());
        assert!(new(0.01, 1.0, 0.05, f32::NAN).is_err());
        assert!(new(0.01, 1.0, 0.05, f32::INFINITY).is_err());
        assert!(new(0.01, -1.0, 0.05, 0.5).is_err());
        assert!(new(0.01, 0.0, 0.05, 0.5).is_err());
        assert!(new(0.0, 1.0, 0.05, 0.5).is_err());
        assert!(new(-0.01, 1.0, 0.05, 0.5).is_err());
        assert!(new(f32::NAN, 1.0, 0.05, 0.5).is_err());
        assert!(new(0.01, 1.0, -0.05, 0.5).is_err());
        // Outside the stability band: k dt^2 = 4 at gamma = 0, and a damped step
        // whose spectral radius exceeds one.
        assert!(new(2.0, 1.0, 0.05, 0.0).is_err());
        assert!(new(0.5, 16.0, 0.05, 0.0).is_err());
        assert!(new(3.0, 1.0, 0.05, 0.005).is_err());
        // Inside the band.
        assert!(new(0.01, 1.0, 0.05, 0.0).is_ok());
        assert!(new(0.01, 1.0, 0.05, 0.5).is_ok());
        let m = new(0.1, 4.0, 0.05, 0.25).unwrap();
        assert_eq!(m.integrator_gamma(), 0.5);
        assert_eq!(m.gamma(), 0.25);
    }

    #[test]
    fn non_finite_state_is_an_error_not_a_state() {
        let m = model(0.5);
        let bad = FullLatent {
            values: [f32::NAN; 1024],
        };
        assert!(matches!(
            m.step(&bad, ActionId(1)),
            Err(CoreError::NumericalInstability(_))
        ));
    }

    #[test]
    fn plugs_into_the_planner_trait_object() {
        let m: std::sync::Arc<dyn WorldModelDynamics<Error = CoreError>> =
            std::sync::Arc::new(ConformalWorldModelDynamics::default());
        let (s1, r, d) = m.step(&start(), ActionId(5)).unwrap();
        assert!(!d && r.is_finite());
        assert!(!m.safety_estimate(&s1, r, d).unwrap().calibrated);
    }
}
