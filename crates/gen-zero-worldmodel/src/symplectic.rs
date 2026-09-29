//! Symplectic (Stormer-Verlet) integrator for separable Hamiltonians H(q, p) = 1/2 ||p||^2 + V(q).
//!
//! Conservative counterpart of the contact integrator: no dissipation, phase volume is
//! preserved exactly and, for suitably resolved trajectories, the energy error stays bounded and
//! oscillatory instead of drifting monotonically.
//! All state lives in fixed-size stack arrays, so a step never touches the heap.

use crate::error::WorldModelError;

/// Phase-space state (q, p) on a 2N-dimensional symplectic manifold.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseState<const N: usize> {
    /// Generalized coordinates in R^N
    pub q: [f32; N],
    /// Generalized conjugate momenta in R^N
    pub p: [f32; N],
}

impl<const N: usize> PhaseState<N> {
    pub fn new(q: [f32; N], p: [f32; N]) -> Self {
        Self { q, p }
    }

    /// Encode one finite phase state as a versioned zstd stream.
    pub fn compress_zstd(&self, level: i32) -> Result<Vec<u8>, WorldModelError> {
        compress_trajectory_zstd(std::slice::from_ref(self), level)
    }

    /// Decode exactly one phase state.
    pub fn decompress_zstd(compressed: &[u8]) -> Result<Self, WorldModelError> {
        let mut states = decompress_trajectory_zstd(compressed)?;
        if states.len() != 1 {
            return Err(WorldModelError::Compression(
                "expected exactly one phase state".into(),
            ));
        }
        Ok(states.remove(0))
    }

    pub fn zeros() -> Self {
        Self {
            q: [0.0; N],
            p: [0.0; N],
        }
    }

    /// Total energy H(q, p) = 1/2 ||p||^2 + V(q).
    pub fn hamiltonian(&self, potential_fn: impl Fn(&[f32; N]) -> f32) -> f32 {
        0.5 * gen_zero_core::dot_product_f32(&self.p, &self.p) + potential_fn(&self.q)
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.q.iter().all(|x| x.is_finite()) && self.p.iter().all(|x| x.is_finite())
    }
}

/// Compress a phase trajectory into one checked zstd stream.
pub fn compress_trajectory_zstd<const N: usize>(
    trajectory: &[PhaseState<N>],
    level: i32,
) -> Result<Vec<u8>, WorldModelError> {
    if N == 0 {
        return Err(WorldModelError::Compression(
            "zero-dimensional phase state".into(),
        ));
    }
    let fields = N
        .checked_mul(2)
        .ok_or_else(|| WorldModelError::Compression("field count overflow".into()))?;
    let _ = trajectory
        .len()
        .checked_mul(fields)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| WorldModelError::Compression("trajectory length overflow".into()))?;
    crate::compression::encode(
        2,
        N,
        trajectory.len(),
        trajectory
            .iter()
            .flat_map(|state| state.q.iter().chain(state.p.iter()).copied()),
        level,
    )
}

/// Decode a phase trajectory with exact dimensions, length and finite values.
pub fn decompress_trajectory_zstd<const N: usize>(
    compressed: &[u8],
) -> Result<Vec<PhaseState<N>>, WorldModelError> {
    if N == 0 {
        return Err(WorldModelError::Compression(
            "zero-dimensional phase state".into(),
        ));
    }
    let fields = N
        .checked_mul(2)
        .ok_or_else(|| WorldModelError::Compression("field count overflow".into()))?;
    let values = crate::compression::decode(2, N, fields, compressed)?;
    let mut states = Vec::with_capacity(values.len() / fields);
    for row in values.chunks_exact(fields) {
        let mut q = [0.0; N];
        let mut p = [0.0; N];
        q.copy_from_slice(&row[..N]);
        p.copy_from_slice(&row[N..]);
        states.push(PhaseState::new(q, p));
    }
    Ok(states)
}

/// Second-order symplectic integrator (kick-drift-kick Stormer-Verlet).
///
/// Phi_{dt/2}^V o Phi_{dt}^T o Phi_{dt/2}^V. Time-reversible and volume-preserving.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymplecticIntegrator {
    pub dt: f32,
}

impl SymplecticIntegrator {
    pub fn new(dt: f32) -> Self {
        Self { dt }
    }

    /// Advance `state` by one step of size `dt`. `grad_potential` writes grad V(q) into its
    /// second argument. On `NumericalDivergence` the input state is left unchanged.
    pub fn step<const N: usize>(
        &self,
        state: &mut PhaseState<N>,
        grad_potential: impl Fn(&[f32; N], &mut [f32; N]),
    ) -> Result<(), WorldModelError> {
        if !self.dt.is_finite() || !state.is_finite() {
            return Err(WorldModelError::NumericalDivergence);
        }
        if self.dt == 0.0 {
            return Ok(());
        }

        // Keep the public operation atomic: a non-finite gradient or arithmetic result must not
        // leave a caller with a partially advanced phase point.
        let mut next = state.clone();
        let half_dt = self.dt * 0.5;
        let mut grad_v = [0.0_f32; N];

        // Kick (half)
        grad_potential(&next.q, &mut grad_v);
        for (pi, &gi) in next.p.iter_mut().zip(grad_v.iter()) {
            *pi -= half_dt * gi;
        }

        // Drift (full)
        for (qi, &pi) in next.q.iter_mut().zip(next.p.iter()) {
            *qi += self.dt * pi;
        }

        // Kick (half)
        grad_v.fill(0.0);
        grad_potential(&next.q, &mut grad_v);
        for (pi, &gi) in next.p.iter_mut().zip(grad_v.iter()) {
            *pi -= half_dt * gi;
        }

        if next.is_finite() {
            *state = next;
            Ok(())
        } else {
            Err(WorldModelError::NumericalDivergence)
        }
    }

    /// Advance `state` by `steps` steps. Stops at the first divergence.
    pub fn integrate<const N: usize>(
        &self,
        state: &mut PhaseState<N>,
        steps: usize,
        grad_potential: impl Fn(&[f32; N], &mut [f32; N]),
    ) -> Result<(), WorldModelError> {
        for _ in 0..steps {
            self.step(state, &grad_potential)?;
        }
        Ok(())
    }

    /// Phase-space volume factor per step. Exactly 1 for a symplectic map.
    #[inline]
    pub fn phase_volume_contraction(&self) -> f32 {
        1.0
    }
}

/// One Stormer-Verlet step on the latent, read as `z = (q, p)` with `q = z[..512]`,
/// `p = z[512..]`. Potential is `V(q) = |q - c_q|^2 / 2` centred on the context's `q` half.
/// Volume-preserving, so never dissipative: only budget termination applies.
/// Stormer-Verlet is stable on this unit-stiffness well only for |dt| < 2, so any other `dt`
/// is rejected up front instead of blowing up over repeated composition.
impl gen_zero_core::traits::LatentContraction for SymplecticIntegrator {
    type Error = WorldModelError;

    fn step_in_place(
        &self,
        z: &mut gen_zero_core::FullLatent,
        context: &gen_zero_core::FullLatent,
    ) -> Result<(), Self::Error> {
        const HALF: usize = crate::contact::LATENT_HALF;
        if !self.dt.is_finite() || self.dt.abs() >= 2.0 {
            return Err(WorldModelError::NumericalDivergence);
        }
        if !z.as_slice().iter().all(|x| x.is_finite())
            || !context.as_slice()[..HALF].iter().all(|x| x.is_finite())
        {
            return Err(WorldModelError::NumericalDivergence);
        }
        let zs = z.as_mut_slice();
        let cq = &context.as_slice()[..HALF];
        let mut state = PhaseState::<HALF>::zeros();
        state.q.copy_from_slice(&zs[..HALF]);
        state.p.copy_from_slice(&zs[HALF..2 * HALF]);
        self.step(&mut state, |q, g| {
            for ((gi, a), c) in g.iter_mut().zip(q).zip(cq) {
                *gi = a - c;
            }
        })?;
        zs[..HALF].copy_from_slice(&state.q);
        zs[HALF..2 * HALF].copy_from_slice(&state.p);
        Ok(())
    }

    /// Stormer-Verlet is a product of shears, so the determinant is exactly 1. The
    /// radius is 1 inside the stability band `|dt| < 2` and above 1 outside it.
    fn contraction_rate(&self) -> f64 {
        let dt = f64::from(self.dt);
        if !dt.is_finite() {
            return f64::NAN;
        }
        let m = crate::contact::well_step_matrix(dt, 1.0);
        crate::contact::spectral_radius_2x2(m[0][0] + m[1][1], 1.0)
    }

    fn is_dissipative(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    struct CountingAlloc;

    thread_local! {
        static ALLOCS: Cell<usize> = const { Cell::new(0) };
    }

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
            System.alloc(layout)
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout)
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
            System.realloc(ptr, layout, new_size)
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAlloc = CountingAlloc;

    fn harmonic_potential(q: &[f32; 2]) -> f32 {
        0.5 * gen_zero_core::dot_product_f32(q, q)
    }

    fn harmonic_grad(q: &[f32; 2], grad: &mut [f32; 2]) {
        grad.copy_from_slice(q);
    }

    fn quartic_potential(q: &[f32; 1]) -> f32 {
        0.25 * q[0].powi(4)
    }

    fn quartic_grad(q: &[f32; 1], grad: &mut [f32; 1]) {
        grad[0] = q[0].powi(3);
    }

    #[test]
    fn test_energy_bounded_over_long_run() {
        let integrator = SymplecticIntegrator::new(0.01);
        let mut state = PhaseState::<2>::new([1.0, 0.5], [0.0, 0.25]);
        let h0 = state.hamiltonian(harmonic_potential);

        for _ in 0..10_000 {
            integrator.step(&mut state, harmonic_grad).unwrap();
            let h = state.hamiltonian(harmonic_potential);
            assert!((h - h0).abs() < 1e-3 * h0, "energy drift: {h} vs {h0}");
        }
    }

    #[test]
    fn test_nonlinear_energy_bounded_over_long_run() {
        let integrator = SymplecticIntegrator::new(0.01);
        let mut state = PhaseState::<1>::new([0.8], [0.3]);
        let h0 = state.hamiltonian(quartic_potential);
        let mut max_error = 0.0_f32;

        for _ in 0..20_000 {
            integrator.step(&mut state, quartic_grad).unwrap();
            let h = state.hamiltonian(quartic_potential);
            max_error = max_error.max((h - h0).abs());
        }

        assert!(max_error < 2e-4, "nonlinear energy error: {max_error}");
    }

    #[test]
    fn test_second_order_accuracy() {
        // Exact solution for q'' = -q, q(0)=1, p(0)=0 is q = cos(t).
        let err = |dt: f32, steps: usize| {
            let mut s = PhaseState::<1>::new([1.0], [0.0]);
            SymplecticIntegrator::new(dt)
                .integrate(&mut s, steps, |q, g| g[0] = q[0])
                .unwrap();
            (s.q[0] - (dt * steps as f32).cos()).abs()
        };
        let coarse = err(0.1, 10);
        let fine = err(0.05, 20);
        assert!(coarse / fine > 3.0, "ratio {}", coarse / fine);
    }

    #[test]
    fn test_time_reversibility() {
        let fwd = SymplecticIntegrator::new(0.02);
        let bwd = SymplecticIntegrator::new(-0.02);
        let start = PhaseState::<2>::new([1.0, -0.3], [0.2, 0.7]);
        let mut state = start.clone();

        fwd.integrate(&mut state, 200, harmonic_grad).unwrap();
        bwd.integrate(&mut state, 200, harmonic_grad).unwrap();

        for i in 0..2 {
            assert!((state.q[i] - start.q[i]).abs() < 1e-4);
            assert!((state.p[i] - start.p[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_divergence_detected() {
        let integrator = SymplecticIntegrator::new(1.0);
        let mut state = PhaseState::<1>::new([1.0], [0.0]);
        let before = state.clone();
        let result = integrator.integrate(&mut state, 10, |_q, g| g[0] = f32::NAN);
        assert_eq!(result, Err(WorldModelError::NumericalDivergence));
        assert_eq!(state, before);
    }

    #[test]
    fn test_step_rejects_nonfinite_dt_and_input_atomically() {
        let before = PhaseState::<1>::new([1.0], [0.25]);
        let mut state = before.clone();
        assert_eq!(
            SymplecticIntegrator::new(f32::NAN).step(&mut state, |_q, g| g[0] = 0.0),
            Err(WorldModelError::NumericalDivergence)
        );
        assert_eq!(state, before);

        let mut state = before.clone();
        assert_eq!(
            SymplecticIntegrator::new(0.1).step(&mut state, |_q, g| g[0] = f32::INFINITY),
            Err(WorldModelError::NumericalDivergence)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn test_step_is_zero_heap_allocation() {
        let integrator = SymplecticIntegrator::new(0.01);
        let mut state = PhaseState::<8>::new([0.1; 8], [0.0; 8]);
        let grad = |q: &[f32; 8], g: &mut [f32; 8]| g.copy_from_slice(q);

        // Guard against a vacuous test: the counter must see a real allocation.
        let probe = ALLOCS.with(|c| c.get());
        drop(std::hint::black_box(Vec::<u8>::with_capacity(16)));
        assert!(
            ALLOCS.with(|c| c.get()) > probe,
            "allocation counter is dead"
        );

        let before = ALLOCS.with(|c| c.get());
        integrator.integrate(&mut state, 1_000, grad).unwrap();
        let after = ALLOCS.with(|c| c.get());

        assert_eq!(after - before, 0);
    }

    #[test]
    fn test_latent_ten_step_rollout_energy_bounded() {
        use gen_zero_core::traits::LatentContraction;
        let integrator = SymplecticIntegrator::new(0.1);
        let ctx = gen_zero_core::FullLatent::zeros();
        let mut vals = [0.0_f32; 1024];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = ((i % 9) as f32 - 4.0) * 0.25;
        }
        let mut z = gen_zero_core::FullLatent { values: vals };
        let e0 = z.l2_norm();
        for _ in 0..12 {
            integrator.step_in_place(&mut z, &ctx).unwrap();
            assert!(z.values.iter().all(|x| x.is_finite()));
            assert!(
                (z.l2_norm() - e0).abs() / e0 < 0.05,
                "norm {} vs {e0}",
                z.l2_norm()
            );
        }
    }

    #[test]
    fn test_latent_step_rejects_unstable_dt_and_non_finite_input() {
        use gen_zero_core::traits::LatentContraction;
        let ctx = gen_zero_core::FullLatent::zeros();
        for dt in [2.0_f32, -2.0, 5.0, f32::NAN, f32::INFINITY] {
            let mut z = gen_zero_core::FullLatent::zeros();
            assert!(SymplecticIntegrator::new(dt)
                .step_in_place(&mut z, &ctx)
                .is_err());
        }
        let mut z = gen_zero_core::FullLatent::zeros();
        z.values[3] = f32::NAN;
        assert!(SymplecticIntegrator::new(0.1)
            .step_in_place(&mut z, &ctx)
            .is_err());

        let mut bad_context = gen_zero_core::FullLatent::zeros();
        bad_context.values[0] = f32::NAN;
        let mut z = gen_zero_core::FullLatent::zeros();
        assert!(SymplecticIntegrator::new(0.1)
            .step_in_place(&mut z, &bad_context)
            .is_err());
    }

    #[test]
    fn test_contraction_rate_is_one_inside_stability_band() {
        use gen_zero_core::traits::LatentContraction;
        for dt in [0.01_f32, 0.5, 1.9, -1.0] {
            assert_eq!(SymplecticIntegrator::new(dt).contraction_rate(), 1.0);
        }
        assert!(SymplecticIntegrator::new(2.5).contraction_rate() > 1.0);
        assert!(SymplecticIntegrator::new(f32::NAN)
            .contraction_rate()
            .is_nan());
    }
}
