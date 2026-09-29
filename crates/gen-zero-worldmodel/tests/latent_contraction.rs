//! `LatentContraction` on the three flows: Koopman, contact, symplectic.

use gen_zero_core::traits::LatentContraction;
use gen_zero_core::FullLatent;
use gen_zero_worldmodel::{ContactIntegrator, KoopmanGenerator, SymplecticIntegrator};

fn latent(f: impl Fn(usize) -> f32) -> FullLatent {
    let v: Vec<f32> = (0..1024).map(f).collect();
    FullLatent::from_slice(&v).unwrap()
}

fn norm(z: &FullLatent) -> f32 {
    z.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn koopman_step_matches_exp_and_is_dissipative() {
    let g = KoopmanGenerator::from_row_slice(2, &[-1.0, 0.0, 0.0, -2.0])
        .unwrap()
        .with_dt(0.5)
        .unwrap();
    assert!(g.is_dissipative());
    let mut z = latent(|i| if i < 2 { 1.0 } else { 7.0 });
    g.step_in_place(&mut z, &latent(|_| 0.0)).unwrap();
    let s = z.as_slice();
    assert!((s[0] - (-0.5_f32).exp()).abs() < 1e-6);
    assert!((s[1] - (-1.0_f32).exp()).abs() < 1e-6);
    assert_eq!(s[2], 7.0, "coordinates beyond dim are untouched");
}

#[test]
fn koopman_growing_or_oversized_is_not_ok() {
    let grow = KoopmanGenerator::from_row_slice(2, &[0.1, 0.0, 0.0, -2.0]).unwrap();
    assert!(!grow.is_dissipative());
    // Rotation: eigenvalues purely imaginary, radius exactly 1, not dissipative.
    let rot = KoopmanGenerator::from_row_slice(2, &[0.0, -1.0, 1.0, 0.0]).unwrap();
    assert!(!rot.is_dissipative());
    let big = KoopmanGenerator::new(nalgebra::DMatrix::zeros(1025, 1025)).unwrap();
    let mut z = latent(|_| 1.0);
    assert!(big.step_in_place(&mut z, &latent(|_| 0.0)).is_err());
}

#[test]
fn contact_contracts_to_context_attractor() {
    let c = ContactIntegrator::new(0.1, 0.5);
    assert!(c.is_dissipative());
    assert!(!ContactIntegrator::new(0.1, 0.0).is_dissipative());
    let ctx = latent(|i| if i < 512 { 0.25 } else { 0.0 });
    let mut z = latent(|i| ((i % 7) as f32) - 3.0);
    for _ in 0..400 {
        c.step_in_place(&mut z, &ctx).unwrap();
    }
    let s = z.as_slice();
    assert!(s[..512].iter().all(|q| (q - 0.25).abs() < 1e-2));
    assert!(s[512..].iter().all(|p| p.abs() < 1e-2));
}

#[test]
fn symplectic_conserves_energy_and_is_not_dissipative() {
    let s = SymplecticIntegrator::new(0.05);
    assert!(!s.is_dissipative());
    let ctx = latent(|_| 0.0);
    let mut z = latent(|i| ((i % 5) as f32) * 0.2);
    let e0 = norm(&z);
    for _ in 0..500 {
        s.step_in_place(&mut z, &ctx).unwrap();
    }
    // H = (|q|^2 + |p|^2)/2 for the harmonic well, so |z| is near-conserved.
    assert!((norm(&z) - e0).abs() / e0 < 1e-2);
}

#[test]
fn divergence_is_an_error() {
    let s = SymplecticIntegrator::new(1.0e30);
    let mut z = latent(|_| 1.0);
    assert!(s.step_in_place(&mut z, &latent(|_| 0.0)).is_err());
}
