//! gen-zero-worldmodel Contact Hamiltonian Dynamics and Strang splitting integrators.
//!
//! Models dissipative open systems on contact manifolds (2N + 1 dimensions: [q, p, s]^T).
//! Implements symmetric Strang operator splitting with phase volume contraction
//! and gauge clamping to limit floating-point drift.

use crate::error::WorldModelError;

/// Contact Hamiltonian state on (2N + 1)-dimensional contact manifold.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct ContactState<const N: usize> {
    /// Generalized coordinates in R^N
    pub q: [f32; N],
    /// Generalized conjugate momenta in R^N
    pub p: [f32; N],
    /// Dissipated action/energy scalar coordinate s in R
    pub s: f32,
}

impl<const N: usize> ContactState<N> {
    pub fn new(q: [f32; N], p: [f32; N], s: f32) -> Self {
        Self { q, p, s }
    }

    /// Encode one finite state as a versioned zstd stream of little-endian f32 bits.
    pub fn compress_zstd(&self, level: i32) -> Result<Vec<u8>, WorldModelError> {
        compress_trajectory_zstd(std::slice::from_ref(self), level)
    }

    /// Decode exactly one state; trajectories and mismatched dimensions are rejected.
    pub fn decompress_zstd(compressed: &[u8]) -> Result<Self, WorldModelError> {
        let mut states = decompress_trajectory_zstd(compressed)?;
        if states.len() != 1 {
            return Err(WorldModelError::Compression(
                "expected exactly one contact state".into(),
            ));
        }
        Ok(states.remove(0))
    }

    pub fn zeros() -> Self {
        Self {
            q: [0.0; N],
            p: [0.0; N],
            s: 0.0,
        }
    }

    /// Total Contact Hamiltonian energy: K(q, p, s) = 1/2 ||p||^2 + V(q) + gamma * s
    pub fn hamiltonian(&self, potential_fn: impl Fn(&[f32; N]) -> f32, gamma: f32) -> f32 {
        let kinetic = 0.5 * gen_zero_core::dot_product_f32(&self.p, &self.p);
        let potential = potential_fn(&self.q);
        kinetic + potential + gamma * self.s
    }

    /// Clamp a finite gauge coordinate `s` to prevent it from growing without bound.
    ///
    /// Non-finite values and invalid bounds are left untouched so clamping cannot turn a
    /// numerical failure into an apparently valid state. The integrator validates both before
    /// committing a step and reports such a state as `NumericalDivergence`.
    #[inline]
    pub fn clamp_gauge(&mut self, bound: f32) {
        if self.s.is_finite() && bound.is_finite() && bound >= 0.0 {
            self.s = self.s.clamp(-bound, bound);
        }
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.q.iter().all(|x| x.is_finite())
            && self.p.iter().all(|x| x.is_finite())
            && self.s.is_finite()
    }
}

/// Compress a contact trajectory into one checked zstd stream.
pub fn compress_trajectory_zstd<const N: usize>(
    trajectory: &[ContactState<N>],
    level: i32,
) -> Result<Vec<u8>, WorldModelError> {
    let fields = N
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| WorldModelError::Compression("field count overflow".into()))?;
    let _ = trajectory
        .len()
        .checked_mul(fields)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| WorldModelError::Compression("trajectory length overflow".into()))?;
    crate::compression::encode(
        1,
        N,
        trajectory.len(),
        trajectory.iter().flat_map(|state| {
            state
                .q
                .iter()
                .chain(state.p.iter())
                .chain(std::iter::once(&state.s))
                .copied()
        }),
        level,
    )
}

/// Decode a contact trajectory with exact dimensions, length and finite values.
pub fn decompress_trajectory_zstd<const N: usize>(
    compressed: &[u8],
) -> Result<Vec<ContactState<N>>, WorldModelError> {
    let fields = N
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| WorldModelError::Compression("field count overflow".into()))?;
    let values = crate::compression::decode(1, N, fields, compressed)?;
    let mut states = Vec::with_capacity(values.len() / fields);
    for row in values.chunks_exact(fields) {
        let mut q = [0.0; N];
        let mut p = [0.0; N];
        q.copy_from_slice(&row[..N]);
        p.copy_from_slice(&row[N..N * 2]);
        states.push(ContactState::new(q, p, row[N * 2]));
    }
    Ok(states)
}

/// Symmetric Strang Operator Splitting Contact Integrator.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactIntegrator {
    pub dt: f32,
    pub gamma: f32,
    pub gauge_clamp_bound: f32,
}

impl ContactIntegrator {
    /// Stores the parameters as given. A negative or non-finite `gamma` is kept, not clamped,
    /// so `step` and `step_in_place` refuse it with `NumericalDivergence`: silently turning a
    /// negative damping into zero would make an invalid parameter look conservative.
    pub fn new(dt: f32, gamma: f32) -> Self {
        Self {
            dt,
            gamma,
            gauge_clamp_bound: 100.0,
        }
    }

    #[inline]
    fn parameters_are_valid(&self) -> bool {
        self.dt.is_finite()
            && self.gamma.is_finite()
            && self.gamma >= 0.0
            && self.gauge_clamp_bound.is_finite()
            && self.gauge_clamp_bound >= 0.0
    }

    /// Perform a single symmetric Strang integration step (second-order accurate):
    /// Phi_{dt/2}^V o Phi_{dt/2}^D o Phi_{dt}^T o Phi_{dt/2}^D o Phi_{dt/2}^V
    ///
    /// The split flow is symmetric before the final gauge clamp. An active clamp can therefore
    /// break exact time reversibility when `s` reaches its bound. Numerical failures leave the
    /// input state unchanged.
    pub fn step<const N: usize>(
        &self,
        state: &mut ContactState<N>,
        potential_fn: impl Fn(&[f32; N]) -> f32,
        grad_potential: impl Fn(&[f32; N], &mut [f32; N]),
    ) -> Result<(), WorldModelError> {
        if !self.parameters_are_valid() || !state.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }
        if self.dt == 0.0 {
            return Ok(());
        }

        // Compute on a candidate and publish only after every stage and the gauge clamp remain
        // finite. This keeps failures atomic even if a caller's gradient or potential diverges.
        let mut next = state.clone();
        let half_dt = self.dt * 0.5;
        let half_dissipation = (-self.gamma * half_dt).exp();

        // Stage 1: Potential half-step (V_{dt/2})
        let mut grad_v = [0.0_f32; N];
        grad_potential(&next.q, &mut grad_v);
        for (pi, &gvi) in next.p.iter_mut().zip(grad_v.iter()) {
            *pi -= half_dt * gvi;
        }
        next.s -= half_dt * potential_fn(&next.q);

        // Stage 2: Dissipation half-step (D_{dt/2})
        for pi in next.p.iter_mut() {
            *pi *= half_dissipation;
        }
        next.s *= half_dissipation;

        // Stage 3: Kinetic full-step (T_{dt})
        for (qi, &pi) in next.q.iter_mut().zip(next.p.iter()) {
            *qi += self.dt * pi;
        }
        let kinetic = 0.5 * gen_zero_core::dot_product_f32(&next.p, &next.p);
        next.s += self.dt * kinetic;

        // Stage 4: Dissipation half-step (D_{dt/2})
        for pi in next.p.iter_mut() {
            *pi *= half_dissipation;
        }
        next.s *= half_dissipation;

        // Stage 5: Potential half-step (V_{dt/2})
        grad_v.fill(0.0);
        grad_potential(&next.q, &mut grad_v);
        for (pi, &gvi) in next.p.iter_mut().zip(grad_v.iter()) {
            *pi -= half_dt * gvi;
        }
        next.s -= half_dt * potential_fn(&next.q);

        // Check before clamping so an infinite action cannot be hidden by `clamp`.
        if !next.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }

        // Gauge clamping to prevent numerical drift.
        next.clamp_gauge(self.gauge_clamp_bound);

        // Sanity check for numerical finiteness
        if !next.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }

        *state = next;
        Ok(())
    }

    /// Theoretical phase-space volume contraction on the symplectic projection:
    /// det(J_{2N}) = exp(-gamma * N * dt). Gauge clamping acts only on the omitted `s`
    /// coordinate, so it does not change this projected determinant.
    #[inline]
    pub fn phase_volume_contraction<const N: usize>(&self) -> f32 {
        (-self.gamma * (N as f32) * self.dt).exp()
    }

    /// Theoretical full contact-manifold contraction before gauge clamping:
    /// det(J_{2N+1}) = exp(-gamma * (N + 1) * dt). Gauge clipping can change the local volume
    /// at the bound.
    #[inline]
    pub fn phase_volume_contraction_contact<const N: usize>(&self) -> f32 {
        (-self.gamma * ((N + 1) as f32) * self.dt).exp()
    }
}

/// Half of the latent read as positions `q`; the other half is momenta `p`.
pub(crate) const LATENT_HALF: usize = 512;

/// One Strang step on the unit-stiffness well `V(x) = x^2 / 2`, for a single `(x, p)`
/// pair, as the exact 2x2 matrix of the splitting
/// `V_{dt/2} o D_{dt/2} o T_{dt} o D_{dt/2} o V_{dt/2}`, where `D` scales `p` by
/// `half_damp`. `half_damp = 1` gives the Stormer-Verlet map. Built in f64 so the
/// spectral analysis is not limited by the f32 state.
pub(crate) fn well_step_matrix(dt: f64, half_damp: f64) -> [[f64; 2]; 2] {
    let h = 0.5 * dt;
    let kick = [[1.0, 0.0], [-h, 1.0]];
    let damp = [[1.0, 0.0], [0.0, half_damp]];
    let drift = [[1.0, dt], [0.0, 1.0]];
    // Rightmost factor acts first.
    [kick, damp, drift, damp, kick]
        .iter()
        .fold([[1.0, 0.0], [0.0, 1.0]], |acc, f| mat2_mul(f, &acc))
}

fn mat2_mul(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut c = [[0.0; 2]; 2];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, cij) in row.iter_mut().enumerate() {
            *cij = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    c
}

/// Spectral radius of a 2x2 map with the given trace and determinant.
/// The determinant is passed in exactly (product of factor determinants) rather than
/// recomputed from entries, so a volume-preserving map reports modulus 1 without
/// cancellation noise when its eigenvalues are complex.
pub(crate) fn spectral_radius_2x2(trace: f64, det: f64) -> f64 {
    let disc = trace * trace - 4.0 * det;
    if disc < 0.0 {
        det.sqrt()
    } else {
        let r = disc.sqrt();
        (0.5 * (trace + r)).abs().max((0.5 * (trace - r)).abs())
    }
}

impl ContactIntegrator {
    /// Damping factor applied to `p` in each dissipation half-step.
    #[inline]
    fn half_damping(&self) -> f64 {
        (-f64::from(self.gamma) * f64::from(self.dt) * 0.5).exp()
    }
}

/// Damped harmonic flow on the latent, read as `z = (q, p)` with `q = z[..512]`,
/// `p = z[512..]`. Potential `V(q) = |q - c_q|^2 / 2` is centred on the context's `q`
/// half, so for `gamma > 0` every trajectory contracts to the attractor `(c_q, 0)`.
/// The contact action `s` does not feed back into `(q, p)` under constant damping, so
/// it has no slot in the latent and is dropped after each step.
impl gen_zero_core::traits::LatentContraction for ContactIntegrator {
    type Error = WorldModelError;

    fn step_in_place(
        &self,
        z: &mut gen_zero_core::FullLatent,
        context: &gen_zero_core::FullLatent,
    ) -> Result<(), Self::Error> {
        const HALF: usize = LATENT_HALF;
        // Reject non-finite parameters, unstable spectrum, and the marginal boundary. A
        // conservative map is allowed inside its |dt| < 2 band; a damped map must satisfy the
        // strict split stability inequality and cannot have an expanding spectral estimate,
        // while dt = 0 is the exact identity.
        let rate = self.contraction_rate();
        let d = self.half_damping();
        let dt = f64::from(self.dt);
        let split_stable = dt * dt * d < 2.0 * (1.0 + d * d);
        let stable = self.parameters_are_valid()
            && rate.is_finite()
            && if self.dt == 0.0 {
                true
            } else if self.gamma == 0.0 {
                self.dt.abs() < 2.0
            } else {
                self.dt > 0.0 && split_stable && rate <= 1.0
            };
        if !stable
            || !z.as_slice().iter().all(|x| x.is_finite())
            || !context.as_slice()[..HALF].iter().all(|x| x.is_finite())
        {
            return Err(WorldModelError::NumericalDivergence);
        }
        let cq = &context.as_slice()[..HALF];
        let zs = z.as_mut_slice();
        let mut state = ContactState::<HALF>::zeros();
        state.q.copy_from_slice(&zs[..HALF]);
        state.p.copy_from_slice(&zs[HALF..2 * HALF]);
        let deviation_sq =
            |q: &[f32; HALF]| -> f32 { q.iter().zip(cq).map(|(a, c)| (a - c) * (a - c)).sum() };
        self.step(
            &mut state,
            |q| 0.5 * deviation_sq(q),
            |q, g| {
                for ((gi, a), c) in g.iter_mut().zip(q).zip(cq) {
                    *gi = a - c;
                }
            },
        )?;
        zs[..HALF].copy_from_slice(&state.q);
        zs[HALF..2 * HALF].copy_from_slice(&state.p);
        Ok(())
    }

    /// Every `(q_i, p_i)` pair evolves under the same 2x2 map, so its spectral radius is
    /// the radius of the whole step. Determinant is exactly `exp(-gamma * dt)`.
    fn contraction_rate(&self) -> f64 {
        let dt = f64::from(self.dt);
        if !dt.is_finite() || !self.gamma.is_finite() || self.gamma < 0.0 {
            return f64::NAN;
        }
        let d = self.half_damping();
        if !d.is_finite() {
            return f64::NAN;
        }
        let m = well_step_matrix(dt, d);
        spectral_radius_2x2(m[0][0] + m[1][1], d * d)
    }

    /// Needs real damping (`gamma * dt > 0`) and a stable step (rate strictly below 1).
    /// With `gamma = 0` the map is symplectic, modulus 1, and this answers false.
    fn is_dissipative(&self) -> bool {
        self.parameters_are_valid()
            && self.gamma > 0.0
            && self.dt > 0.0
            && self.contraction_rate() < 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contact_dynamics_phase_contraction() {
        let integrator = ContactIntegrator::new(0.01, 0.5);
        let contraction = integrator.phase_volume_contraction::<4>();
        // det(J) = exp(-0.5 * 4 * 0.01) = exp(-0.02) < 1.0
        assert!(contraction < 1.0);
        assert!((contraction - (-0.02_f32).exp()).abs() < 1e-6);

        // Harmonic oscillator potential: V(q) = 1/2 ||q||^2, grad V = q
        let mut state = ContactState::<2>::new([1.0, 0.5], [0.0, 0.0], 0.0);

        for _ in 0..100 {
            integrator
                .step(
                    &mut state,
                    |q| 0.5 * gen_zero_core::dot_product_f32(q, q),
                    |q, grad| {
                        grad.copy_from_slice(q);
                    },
                )
                .unwrap();
        }

        // Dissipation should cause coordinates and momenta to contract towards origin
        let q_norm = gen_zero_core::dot_product_f32(&state.q, &state.q).sqrt();
        let p_norm = gen_zero_core::dot_product_f32(&state.p, &state.p).sqrt();
        assert!(q_norm < 1.0);
        assert!(p_norm < 1.0);
        assert!(state.s.abs() <= 100.0);
    }

    #[test]
    fn test_step_matrix_matches_real_step() {
        // Push the basis vectors through the f32 integrator; columns must match the
        // analytic matrix used for the spectral analysis.
        let c = ContactIntegrator::new(0.3, 0.7);
        let m = well_step_matrix(0.3, c.half_damping());
        for (col, (q0, p0)) in [(1.0_f32, 0.0_f32), (0.0, 1.0)].into_iter().enumerate() {
            let mut st = ContactState::<1>::new([q0], [p0], 0.0);
            c.step(&mut st, |q| 0.5 * q[0] * q[0], |q, g| g[0] = q[0])
                .unwrap();
            assert!((f64::from(st.q[0]) - m[0][col]).abs() < 1e-6);
            assert!((f64::from(st.p[0]) - m[1][col]).abs() < 1e-6);
        }
        // Determinant identity: det = half_damp^2 = exp(-gamma dt).
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        assert!((det - (-f64::from(0.7_f32) * f64::from(0.3_f32)).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_contraction_rate_bounds() {
        use gen_zero_core::traits::LatentContraction;
        let damped = ContactIntegrator::new(0.1, 0.5);
        let r = damped.contraction_rate();
        // Underdamped: complex pair, |lambda| = sqrt(det) = exp(-gamma dt / 2).
        assert!((r - (-0.25_f64 * f64::from(0.1_f32)).exp()).abs() < 1e-12);
        assert!(damped.lyapunov_exponent() < 0.0);

        let conservative = ContactIntegrator::new(0.1, 0.0);
        assert_eq!(conservative.contraction_rate(), 1.0);
        assert!(!conservative.is_dissipative());

        // Past the stability band the damped step still expands: not dissipative.
        let unstable = ContactIntegrator::new(3.0, 0.01);
        assert!(unstable.contraction_rate() > 1.0);
        assert!(!unstable.is_dissipative());
        let mut z = gen_zero_core::FullLatent::zeros();
        assert_eq!(
            unstable.step_in_place(&mut z, &gen_zero_core::FullLatent::zeros()),
            Err(WorldModelError::NumericalDivergence)
        );

        assert!(ContactIntegrator::new(f32::NAN, 0.5)
            .contraction_rate()
            .is_nan());
    }

    #[test]
    fn test_latent_step_contracts_distance_to_attractor() {
        use gen_zero_core::traits::LatentContraction;
        let c = ContactIntegrator::new(0.1, 0.5);
        let rate = c.contraction_rate();
        let ctx = gen_zero_core::FullLatent::zeros();
        let mut z = gen_zero_core::FullLatent::zeros();
        z.values[0] = 1.0;
        z.values[LATENT_HALF + 3] = -2.0;
        let e0 = z.l2_norm();
        for _ in 0..50 {
            c.step_in_place(&mut z, &ctx).unwrap();
        }
        // Asymptotic rate bounds the decay up to the eigenvector conditioning constant.
        let bound = f64::from(e0) * rate.powi(50) * 4.0;
        assert!(f64::from(z.l2_norm()) < bound, "{} vs {bound}", z.l2_norm());

        z.values[7] = f32::INFINITY;
        assert_eq!(
            c.step_in_place(&mut z, &ctx),
            Err(WorldModelError::NumericalDivergence)
        );
    }

    #[test]
    fn test_failure_is_atomic_and_gauge_clamp_does_not_hide_infinity() {
        let c = ContactIntegrator::new(0.1, 0.5);
        let before = ContactState::<1>::new([1.0], [0.25], 0.5);
        let mut state = before.clone();
        assert_eq!(
            c.step(&mut state, |_q| f32::INFINITY, |_q, g| g[0] = 0.0),
            Err(WorldModelError::NumericalDivergence)
        );
        assert_eq!(state, before);

        let mut nonfinite = ContactState::<1>::new([0.0], [0.0], f32::INFINITY);
        nonfinite.clamp_gauge(100.0);
        assert!(nonfinite.s.is_infinite());

        let mut invalid_bound = ContactState::<1>::new([0.0], [0.0], 200.0);
        invalid_bound.clamp_gauge(f32::NAN);
        assert_eq!(invalid_bound.s, 200.0);
    }

    #[test]
    fn test_nonfinite_parameters_are_rejected_and_zero_dt_is_identity() {
        let nan_gamma = ContactIntegrator::new(0.1, f32::NAN);
        assert!(nan_gamma.gamma.is_nan());
        // Negative damping is kept as given and refused by `step`, never clamped to zero.
        let negative = ContactIntegrator::new(0.1, -0.5);
        assert_eq!(negative.gamma, -0.5);
        let mut state = ContactState::<1>::new([1.0], [0.25], 0.5);
        let before = state.clone();
        assert_eq!(
            negative.step(&mut state, |q| 0.5 * q[0] * q[0], |q, g| g[0] = q[0]),
            Err(WorldModelError::NumericalDivergence)
        );
        assert_eq!(state, before);
        let mut state = ContactState::<1>::new([1.0], [0.25], 0.5);
        let before = state.clone();
        assert_eq!(
            nan_gamma.step(&mut state, |q| 0.5 * q[0] * q[0], |q, g| g[0] = q[0]),
            Err(WorldModelError::NumericalDivergence)
        );
        assert_eq!(state, before);

        let mut zero_dt = ContactIntegrator::new(0.0, 0.5);
        zero_dt.gauge_clamp_bound = 1.0;
        let mut state = ContactState::<1>::new([1.0], [0.25], 200.0);
        let before = state.clone();
        zero_dt
            .step(&mut state, |_q| f32::NAN, |_q, g| g[0] = f32::NAN)
            .unwrap();
        assert_eq!(state, before);

        for (dt, bound) in [(f32::NAN, 100.0), (f32::INFINITY, 100.0), (0.1, -1.0)] {
            let mut invalid = ContactIntegrator::new(dt, 0.5);
            invalid.gauge_clamp_bound = bound;
            let mut state = ContactState::<1>::new([1.0], [0.25], 0.5);
            let before = state.clone();
            assert_eq!(
                invalid.step(&mut state, |q| 0.5 * q[0] * q[0], |q, g| g[0] = q[0]),
                Err(WorldModelError::NumericalDivergence)
            );
            assert_eq!(state, before);
        }
    }

    #[test]
    fn test_signed_dt_reversibility_and_latent_stability_boundary() {
        use gen_zero_core::traits::LatentContraction;

        for gamma in [0.0_f32, 0.5] {
            let forward = ContactIntegrator::new(0.02, gamma);
            let backward = ContactIntegrator::new(-0.02, gamma);
            let start = ContactState::<1>::new([0.8], [0.3], 0.1);
            let mut state = start.clone();
            for _ in 0..100 {
                forward
                    .step(&mut state, |q| 0.5 * q[0] * q[0], |q, g| g[0] = q[0])
                    .unwrap();
            }
            for _ in 0..100 {
                backward
                    .step(&mut state, |q| 0.5 * q[0] * q[0], |q, g| g[0] = q[0])
                    .unwrap();
            }
            assert!((state.q[0] - start.q[0]).abs() < 1e-5);
            assert!((state.p[0] - start.p[0]).abs() < 1e-5);
            assert!((state.s - start.s).abs() < 1e-5);
        }

        let ctx = gen_zero_core::FullLatent::zeros();
        for (dt, gamma) in [(2.0_f32, 0.0_f32), (-2.0, 0.0), (-0.1, 0.5)] {
            let mut z = gen_zero_core::FullLatent::zeros();
            let before = z.clone();
            assert_eq!(
                ContactIntegrator::new(dt, gamma).step_in_place(&mut z, &ctx),
                Err(WorldModelError::NumericalDivergence)
            );
            assert_eq!(z, before);
        }

        // Positive damping can extend the strict stability interval beyond |dt| = 2.
        let mut stable = gen_zero_core::FullLatent::zeros();
        ContactIntegrator::new(2.0, 0.5)
            .step_in_place(&mut stable, &ctx)
            .unwrap();

        // The strict analytic test remains usable when a tiny positive damping factor rounds to
        // one in the spectral estimate.
        let mut tiny_damping = gen_zero_core::FullLatent::zeros();
        ContactIntegrator::new(1.9, 1e-30)
            .step_in_place(&mut tiny_damping, &ctx)
            .unwrap();
    }
}
