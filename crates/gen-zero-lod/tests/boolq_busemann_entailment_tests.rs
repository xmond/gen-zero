//! Busemann entailment on the BoolQ topology `H^80 x R^24 x S^23` (128 f64).
//!
//! Every expected value below is derived independently of the code under test:
//! closed-form Poincare-ball formulas, or the pre-existing `kernel` distances.
//! No trained data is involved; these tests check the geometry, not accuracy.

use gen_zero_lod::manifold::{kernel, ProductGeometry, RADIAL_DIRECTION_EPS};
use gen_zero_lod::{
    ContainmentCriteria, Epochs, GeometryParams, Layout, LodError, ProductManifold, Reject,
    TopologyPreset, Version,
};
use std::f64::consts::PI;

const H: usize = 80;
const E: usize = 24;
const S: usize = 24; // S^23 stored with 24 embedding coordinates
const DIM: usize = H + E + S;
const CAP: f64 = 1.0;

fn manifold(c: f64) -> ProductManifold {
    let params = GeometryParams {
        curvature: c,
        radius: 1.0,
        alpha_h: 1.0,
        alpha_e: 1.0,
        alpha_s: 1.0,
    };
    let base = Epochs {
        version: Version(0),
        model: [0; 32],
        geometry: [0; 32],
        atlas: [0; 32],
        graph: [0; 32],
        policy: [0; 32],
    };
    ProductManifold::from_preset_with(TopologyPreset::Boolq128d, params, base)
        .expect("valid boolq geometry")
}

/// Stored point: hyperbolic part `h`, Euclidean part `e0` on axis 0, sphere
/// point at angle `phi` from the sphere base point.
fn point(h: &[f64], e0: f64, phi: f64) -> Vec<f64> {
    assert_eq!(h.len(), H);
    let mut v = vec![0.0; DIM];
    v[..H].copy_from_slice(h);
    v[H] = e0;
    v[H + E] = phi.cos();
    v[H + E + 1] = phi.sin();
    v
}

/// Hyperbolic part with unit-ball radius `r` (coordinate norm `r / sqrt(c)`)
/// along the unit vector `dir`.
fn radial(c: f64, r: f64, dir: &[f64]) -> Vec<f64> {
    let n = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    dir.iter().map(|x| x / n * r / c.sqrt()).collect()
}

/// Unit-ish direction at angle `theta` from axis 0, rotating into axis 1.
fn dir_at(theta: f64) -> Vec<f64> {
    let mut d = vec![0.0; H];
    d[0] = theta.cos();
    d[1] = theta.sin();
    d
}

fn axis0() -> Vec<f64> {
    dir_at(0.0)
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------- geometry

#[test]
fn boolq_topology_is_h80_r24_s23_in_16_cache_lines() {
    let m = manifold(1.0);
    let l = *m.layout();
    assert_eq!(l, Layout::STORE_128_DEEP);
    assert_eq!((l.h(), l.e(), l.s_intrinsic()), (80, 24, 23));
    assert_eq!(l.store_dim(), 128);
    assert_eq!(l.store_dim() * std::mem::size_of::<f64>() % 64, 0);
    assert_eq!(m.epochs().geometry, m.params().digest(l));
}

// ---------------------------------------------------------------- asymmetry

#[test]
fn ancestor_entails_descendant_and_never_the_reverse() {
    let m = manifold(1.0);
    let general = point(&radial(1.0, 0.30, &axis0()), 0.0, 0.0);
    let specific = point(&radial(1.0, 0.80, &dir_at(0.02)), 0.0, 0.01);

    let fwd = m.busemann_containment(&general, &specific, CAP).unwrap();
    assert!(fwd.is_entailed, "{fwd:?}");
    assert!(fwd.in_cone && fwd.deeper && fwd.sphere_absorbed && fwd.topic_aligned);
    assert!(fwd.confidence > 0.0 && fwd.confidence <= 1.0);
    assert!(fwd.busemann_depth_gain > 0.0);

    let rev = m.busemann_containment(&specific, &general, CAP).unwrap();
    assert!(!rev.is_entailed, "{rev:?}");
    assert!(!rev.deeper, "reverse must fail on depth: {rev:?}");
    assert!(rev.busemann_depth_gain < 0.0);
    assert_eq!(rev.confidence, 0.0);
}

#[test]
fn a_concept_does_not_strictly_contain_itself() {
    let m = manifold(1.0);
    let p = point(&radial(1.0, 0.5, &axis0()), 0.0, 0.0);
    let s = m.busemann_containment(&p, &p, CAP).unwrap();
    assert!(!s.is_entailed);
    assert_eq!(s.busemann_depth_gain, 0.0);
    assert_eq!(s.cone_angle, 0.0);
}

/// On the ray to the ideal point the Busemann function is minus the distance
/// from the origin, so the depth gain must equal `d(0,q) - d(0,p)`. The
/// reference side uses `kernel::hyperbolic_distance`, not the Busemann code.
#[test]
fn collinear_depth_gain_equals_difference_of_origin_distances() {
    for c in [0.25, 1.0, 3.0] {
        let m = manifold(c);
        let (hp, hq) = (radial(c, 0.35, &axis0()), radial(c, 0.85, &axis0()));
        let origin = vec![0.0; H];
        let dp = kernel::hyperbolic_distance(c, &origin, &hp).unwrap();
        let dq = kernel::hyperbolic_distance(c, &origin, &hq).unwrap();
        let s = m
            .busemann_containment(&point(&hp, 0.0, 0.0), &point(&hq, 0.0, 0.0), CAP)
            .unwrap();
        assert!(
            (s.busemann_depth_gain - (dq - dp)).abs() < 1e-9,
            "c={c}: gain {} vs {}",
            s.busemann_depth_gain,
            dq - dp
        );
        let dpq = kernel::hyperbolic_distance(c, &hp, &hq).unwrap();
        assert!((s.hyperbolic_distance - dpq).abs() < 1e-12);
    }
}

/// Random clustered pairs: the reverse of an entailed pair is never entailed,
/// and the sweep really contains entailed pairs (not vacuously true).
#[test]
fn asymmetry_holds_on_a_random_sweep() {
    let m = manifold(1.0);
    let mut rng = Lcg(0x5eed_b001);
    let mut pts = Vec::new();
    for _ in 0..60 {
        let r = 0.05 + 0.9 * rng.next();
        let mut d = axis0();
        for x in d.iter_mut() {
            *x += 0.04 * (rng.next() - 0.5);
        }
        pts.push(point(
            &radial(1.0, r, &d),
            0.2 * rng.next(),
            0.05 * rng.next(),
        ));
    }
    let (mut entailed, mut checked) = (0usize, 0usize);
    for (i, a) in pts.iter().enumerate() {
        for (j, b) in pts.iter().enumerate() {
            if i == j {
                continue;
            }
            let ab = m.busemann_containment(a, b, CAP).unwrap();
            let ba = m.busemann_containment(b, a, CAP).unwrap();
            assert!(
                !(ab.is_entailed && ba.is_entailed),
                "pair {i},{j} entails both ways"
            );
            entailed += usize::from(ab.is_entailed);
            checked += 1;
        }
    }
    assert!(
        checked > 3000 && entailed > 100,
        "vacuous sweep: {entailed}/{checked}"
    );
}

// ---------------------------------------------------------------- factors

#[test]
fn sphere_factor_absorbs_small_rotations_and_rejects_large_ones() {
    let m = manifold(1.0);
    let hp = radial(1.0, 0.3, &axis0());
    let hq = radial(1.0, 0.8, &axis0());
    let p = point(&hp, 0.0, 0.0);
    let absorbed = m
        .busemann_containment(&p, &point(&hq, 0.0, 0.4), CAP)
        .unwrap();
    assert!((absorbed.sphere_angle - 0.4).abs() < 1e-12);
    assert!(absorbed.is_entailed);
    let far = m
        .busemann_containment(&p, &point(&hq, 0.0, 1.5), CAP)
        .unwrap();
    assert!((far.sphere_angle - 1.5).abs() < 1e-12);
    assert!(!far.sphere_absorbed && !far.is_entailed);
    // The antipode is a verdict (angle pi), not an error.
    let anti = m
        .busemann_containment(&p, &point(&hq, 0.0, PI), CAP)
        .unwrap();
    assert!((anti.sphere_angle - PI).abs() < 1e-9 && !anti.is_entailed);
}

#[test]
fn euclidean_topic_shift_is_measured_and_gated() {
    let m = manifold(1.0);
    let p = point(&radial(1.0, 0.3, &axis0()), 0.0, 0.0);
    let q_near = point(&radial(1.0, 0.8, &axis0()), 0.5, 0.0);
    let q_far = point(&radial(1.0, 0.8, &axis0()), 3.0, 0.0);
    let near = m.busemann_containment(&p, &q_near, CAP).unwrap();
    assert!((near.topic_shift - 0.5).abs() < 1e-12 && near.is_entailed);
    let far = m.busemann_containment(&p, &q_far, CAP).unwrap();
    assert!((far.topic_shift - 3.0).abs() < 1e-12);
    assert!(!far.topic_aligned && !far.is_entailed);
}

#[test]
fn radial_angle_matches_the_constructed_angle() {
    let m = manifold(1.0);
    let p = point(&radial(1.0, 0.3, &axis0()), 0.0, 0.0);
    for theta in [0.001, 0.05, 0.4, 1.2, 3.0] {
        let q = point(&radial(1.0, 0.8, &dir_at(theta)), 0.0, 0.0);
        let s = m.busemann_containment(&p, &q, PI).unwrap();
        assert!(
            (s.cone_angle - theta).abs() < 1e-12,
            "theta {theta}: {}",
            s.cone_angle
        );
    }
}

// ---------------------------------------------------------------- curvature

/// Aperture `psi = asin(K (1 - c |x|^2) / (sqrt(c) |x|))` for a FIXED
/// coordinate `x` falls strictly with curvature (numerator down, denominator
/// up) and reaches 0 as `c |x|^2 -> 1`; it saturates at `pi/2` for small `c`.
#[test]
fn cone_aperture_shrinks_monotonically_with_curvature() {
    let norm_p = 0.4;
    let hp: Vec<f64> = axis0().iter().map(|x| x * norm_p).collect();
    let hq: Vec<f64> = axis0().iter().map(|x| x * 0.3).collect(); // in the ball for every c below
    let p = point(&hp, 0.0, 0.0);
    let q = point(&hq, 0.0, 0.0);
    // 1/norm_p^2 = 6.25 is the boundary curvature for p.
    let curvatures = [0.01, 0.0625, 0.1, 0.3, 0.8, 1.5, 3.0, 5.0, 6.0, 6.2, 6.24];
    let mut prev = f64::INFINITY;
    let mut first = 0.0;
    let mut last = 0.0;
    for (i, &c) in curvatures.iter().enumerate() {
        let m = manifold(c);
        let s = m.busemann_containment(&p, &q, PI).unwrap();
        let r = c.sqrt() * norm_p;
        let expect = (0.1 * (1.0 - r * r) / r).min(1.0).asin();
        assert!(
            (s.aperture - expect).abs() < 1e-12,
            "c={c}: {} vs {expect}",
            s.aperture
        );
        if i == 0 {
            first = s.aperture;
            assert!((s.aperture - PI / 2.0).abs() < 1e-12, "saturates at pi/2");
        } else {
            assert!(
                s.aperture <= prev,
                "aperture rose at c={c}: {} > {prev}",
                s.aperture
            );
        }
        prev = s.aperture;
        last = s.aperture;
    }
    assert!(
        last < 1e-3 && last < first,
        "aperture must converge to 0, got {last}"
    );
    // Strictly decreasing once off the saturation plateau.
    let unsat: Vec<f64> = curvatures[2..]
        .iter()
        .map(|&c| {
            manifold(c)
                .busemann_containment(&p, &q, PI)
                .unwrap()
                .aperture
        })
        .collect();
    assert!(unsat.windows(2).all(|w| w[1] < w[0]), "{unsat:?}");
}

/// A fixed question at a fixed angle is inside the cone for small curvature
/// and drops out as curvature grows; once out it never comes back.
#[test]
fn entailment_verdict_flips_once_as_curvature_grows() {
    let (norm_p, norm_q, theta) = (0.6, 0.9, 0.15);
    let hp: Vec<f64> = axis0().iter().map(|x| x * norm_p).collect();
    let hq: Vec<f64> = dir_at(theta).iter().map(|x| x * norm_q).collect();
    let (p, q) = (point(&hp, 0.0, 0.0), point(&hq, 0.0, 0.0));
    let verdicts: Vec<bool> = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.2]
        .iter()
        .map(|&c| {
            manifold(c)
                .busemann_containment(&p, &q, PI)
                .unwrap()
                .is_entailed
        })
        .collect();
    assert!(verdicts[0] && !verdicts[verdicts.len() - 1], "{verdicts:?}");
    let flips = verdicts.windows(2).filter(|w| w[0] != w[1]).count();
    assert_eq!(
        flips, 1,
        "cone membership must be monotone in c: {verdicts:?}"
    );
}

#[test]
fn cap_bounds_the_aperture_from_above() {
    let m = manifold(1.0);
    let p = point(&radial(1.0, 0.05, &axis0()), 0.0, 0.0); // psi saturates near pi/2
    let q = point(&radial(1.0, 0.9, &dir_at(0.5)), 0.0, 0.0);
    let wide = m.busemann_containment(&p, &q, PI).unwrap();
    assert!((wide.aperture - PI / 2.0).abs() < 1e-12 && wide.in_cone);
    let tight = m.busemann_containment(&p, &q, 0.1).unwrap();
    assert!((tight.aperture - 0.1).abs() < 1e-15 && !tight.in_cone);
}

// ---------------------------------------------------------------- fail closed

fn good_pair() -> (Vec<f64>, Vec<f64>) {
    (
        point(&radial(1.0, 0.3, &axis0()), 0.0, 0.0),
        point(&radial(1.0, 0.8, &dir_at(0.02)), 0.0, 0.0),
    )
}

fn geometry_err(r: Result<impl std::fmt::Debug, LodError>) -> Reject {
    match r {
        Err(LodError::Geometry(rej)) => rej,
        other => panic!("expected LodError::Geometry, got {other:?}"),
    }
}

#[test]
fn control_pair_passes() {
    let m = manifold(1.0);
    let (p, q) = good_pair();
    assert!(m.busemann_containment(&p, &q, CAP).unwrap().is_entailed);
}

#[test]
fn non_finite_coordinates_are_rejected_in_every_factor() {
    let m = manifold(1.0);
    let (p, q) = good_pair();
    for idx in [0, H - 1, H, H + E - 1, H + E, DIM - 1] {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut broken = q.clone();
            broken[idx] = bad;
            assert_eq!(
                geometry_err(m.busemann_containment(&p, &broken, CAP)),
                Reject::NonFiniteState,
                "question idx {idx} = {bad}"
            );
            assert_eq!(
                geometry_err(m.busemann_containment(&broken, &q, CAP)),
                Reject::NonFiniteState,
                "passage idx {idx} = {bad}"
            );
        }
    }
}

#[test]
fn ball_boundary_and_overflow_are_rejected() {
    let m = manifold(1.0);
    let (p, good_q) = good_pair();
    let boundary = |norm: f64| {
        let mut q = good_q.clone();
        q[..H].copy_from_slice(&axis0().iter().map(|x| x * norm).collect::<Vec<_>>());
        q
    };
    // Exactly on the boundary, inside the 1e-12 margin, and outside.
    for norm in [1.0, 1.0 - 2e-13, 1.0000001, 1.5, 10.0] {
        assert_eq!(
            geometry_err(m.busemann_containment(&p, &boundary(norm), CAP)),
            Reject::DomainViolation,
            "norm {norm}"
        );
    }
    // Squared norm overflows to +inf: must not wrap around to the origin.
    for norm in [1e155, 1e200, 1e300, f64::MAX] {
        assert_eq!(
            geometry_err(m.busemann_containment(&boundary(norm), &good_q, CAP)),
            Reject::DomainViolation,
            "overflow norm {norm}"
        );
    }
    // Sphere part off the sphere.
    let mut off = good_q.clone();
    off[H + E] *= 1.1;
    assert_eq!(
        geometry_err(m.busemann_containment(&p, &off, CAP)),
        Reject::DomainViolation
    );
}

#[test]
fn wrong_lengths_are_rejected() {
    let m = manifold(1.0);
    let (p, q) = good_pair();
    for len in [0, 1, DIM - 1, DIM + 1, 2 * DIM] {
        let mut bad = q.clone();
        bad.resize(len, 0.0);
        assert_eq!(
            geometry_err(m.busemann_containment(&p, &bad, CAP)),
            Reject::DomainViolation,
            "len {len}"
        );
        assert_eq!(
            geometry_err(m.busemann_containment(&bad, &q, CAP)),
            Reject::DomainViolation,
            "len {len}"
        );
    }
}

#[test]
fn invalid_cone_angles_are_rejected() {
    let m = manifold(1.0);
    let (p, q) = good_pair();
    for bad in [f64::NAN, f64::INFINITY, 0.0, -0.1, PI + 1e-9, 10.0] {
        assert_eq!(
            geometry_err(m.busemann_containment(&p, &q, bad)),
            Reject::DomainViolation,
            "cone angle {bad}"
        );
    }
    let mut criteria = ContainmentCriteria::from_cone_half_angle(CAP);
    criteria.aperture_k = f64::NAN;
    assert_eq!(
        geometry_err(m.busemann_containment_with(&p, &q, &criteria)),
        Reject::DomainViolation
    );
}

#[test]
fn undefined_radial_direction_is_an_error_not_a_verdict() {
    let m = manifold(1.0);
    let (p, q) = good_pair();
    let root = point(&vec![0.0; H], 0.0, 0.0);
    assert_eq!(
        m.busemann_containment(&root, &q, CAP),
        Err(LodError::DegenerateRadialDirection("passage"))
    );
    assert_eq!(
        m.busemann_containment(&p, &root, CAP),
        Err(LodError::DegenerateRadialDirection("question"))
    );
    let tiny = point(
        &radial(1.0, RADIAL_DIRECTION_EPS / 10.0, &axis0()),
        0.0,
        0.0,
    );
    assert!(matches!(
        m.busemann_containment(&tiny, &q, CAP),
        Err(LodError::DegenerateRadialDirection("passage"))
    ));
}

// ---------------------------------------------------------- alpha weights

/// `confidence` must match the `alpha_h/e/s`-weighted average of the three
/// factor margins exactly, and must actually move when the weights change
/// (a dead knob would leave it fixed regardless of alpha).
#[test]
fn alpha_weights_are_live_in_confidence_not_dead_knobs() {
    let c = 1.0;
    let p = point(&radial(c, 0.3, &axis0()), 0.0, 0.0);
    let q = point(&radial(c, 0.8, &dir_at(0.02)), 0.3, 0.15);
    let criteria = ContainmentCriteria::from_cone_half_angle(CAP);

    let score_at = |alpha_h: f64, alpha_e: f64, alpha_s: f64| {
        let params = GeometryParams {
            curvature: c,
            radius: 1.0,
            alpha_h,
            alpha_e,
            alpha_s,
        };
        let base = Epochs {
            version: Version(0),
            model: [0; 32],
            geometry: [0; 32],
            atlas: [0; 32],
            graph: [0; 32],
            policy: [0; 32],
        };
        ProductManifold::from_preset_with(TopologyPreset::Boolq128d, params, base)
            .expect("valid boolq geometry")
            .busemann_containment(&p, &q, CAP)
            .unwrap()
    };

    let uniform = score_at(1.0, 1.0, 1.0);
    assert!(uniform.is_entailed, "{uniform:?}");
    let sc = c.sqrt();
    let margin_h = (1.0 - uniform.cone_angle / uniform.aperture)
        .min(-(-uniform.busemann_depth_gain * sc).exp_m1());
    let margin_e = 1.0 - uniform.topic_shift / criteria.topic_shift_tol;
    let margin_s = 1.0 - uniform.sphere_angle / criteria.sphere_absorb_angle;
    assert!(
        (margin_h - margin_e).abs() > 1e-3 && (margin_e - margin_s).abs() > 1e-3,
        "fixture must give distinct margins: h={margin_h} e={margin_e} s={margin_s}"
    );

    for (ah, ae, as_) in [
        (1.0, 1.0, 1.0),
        (5.0, 1.0, 1.0),
        (1.0, 5.0, 1.0),
        (1.0, 1.0, 5.0),
    ] {
        let want = (ah * margin_h + ae * margin_e + as_ * margin_s) / (ah + ae + as_);
        let got = score_at(ah, ae, as_).confidence;
        assert!(
            (got - want).abs() < 1e-9,
            "alpha=({ah},{ae},{as_}): want {want}, got {got}"
        );
    }

    let confidences: Vec<f64> = [(5.0, 1.0, 1.0), (1.0, 5.0, 1.0), (1.0, 1.0, 5.0)]
        .into_iter()
        .map(|(ah, ae, as_)| score_at(ah, ae, as_).confidence)
        .collect();
    assert!(
        confidences[0] != confidences[1] || confidences[1] != confidences[2],
        "confidence is invariant to alpha: {confidences:?}"
    );
}

// ------------------------------------------------------- sphere overflow

/// At `R = 1e154` the old code overflowed `(a + b)^2` to `inf` and folded
/// a real 1.2 rad separation to `atan2(finite, inf) == 0.0`.
#[test]
fn sphere_angle_does_not_overflow_to_zero_at_extreme_radius() {
    let radius = 1e154_f64;
    let true_angle = 1.2_f64;
    let s = [radius, 0.0];
    let t = [radius * true_angle.cos(), radius * true_angle.sin()];
    let angle = kernel::sphere_angle(radius, &s, &t, 1e-9).unwrap();
    assert!(
        (angle - true_angle).abs() < 1e-9,
        "expected {true_angle}, got {angle} -- the old code folded this to 0.0 on overflow"
    );
}

fn manifold_radius(c: f64, radius: f64) -> ProductManifold {
    let params = GeometryParams {
        curvature: c,
        radius,
        alpha_h: 1.0,
        alpha_e: 1.0,
        alpha_s: 1.0,
    };
    let base = Epochs {
        version: Version(0),
        model: [0; 32],
        geometry: [0; 32],
        atlas: [0; 32],
        graph: [0; 32],
        policy: [0; 32],
    };
    ProductManifold::from_preset_with(TopologyPreset::Boolq128d, params, base)
        .expect("valid boolq geometry")
}

/// Same as `point`, but the sphere part is scaled to sit on radius `radius`
/// instead of the unit sphere.
fn point_at_radius(h: &[f64], e0: f64, phi: f64, radius: f64) -> Vec<f64> {
    assert_eq!(h.len(), H);
    let mut v = vec![0.0; DIM];
    v[..H].copy_from_slice(h);
    v[H] = e0;
    v[H + E] = radius * phi.cos();
    v[H + E + 1] = radius * phi.sin();
    v
}

/// Same overflow, at the `busemann_containment` level: every other factor
/// passes, so only a correctly measured sphere angle blocks entailment.
#[test]
fn extreme_radius_sphere_separation_correctly_blocks_entailment() {
    let radius = 1e154_f64;
    let m = manifold_radius(1.0, radius);
    let general = point_at_radius(&radial(1.0, 0.30, &axis0()), 0.0, 0.0, radius);
    let specific = point_at_radius(&radial(1.0, 0.80, &dir_at(0.02)), 0.0, 1.2, radius);

    let result = m.busemann_containment(&general, &specific, CAP).unwrap();
    assert!(
        (result.sphere_angle - 1.2).abs() < 1e-9,
        "sphere_angle should measure the true 1.2 rad separation: {result:?}"
    );
    assert!(
        result.in_cone && result.deeper && result.topic_aligned,
        "{result:?}"
    );
    assert!(
        !result.sphere_absorbed && !result.is_entailed,
        "a 1.2 rad sphere separation must not be absorbed by a 1.0 rad threshold: {result:?}"
    );
}

// ---------------------------------------------------------------- violation energy

/// The violation energy is the alpha-weighted mean of the squared normalised
/// misses, in closed form per factor. It is zero when every test passes, grows
/// continuously with the miss, and never stands in for the gate: a miss of
/// 1e-9 leaves `soft_confidence` at 1 to twelve digits and is still refused.
#[test]
fn violation_energy_is_continuous_evidence_and_never_the_gate() {
    let m = manifold(1.0);
    let hp = radial(1.0, 0.3, &axis0());
    let hq = radial(1.0, 0.8, &axis0());
    let p = point(&hp, 0.0, 0.0);
    let score = |q: &[f64]| m.busemann_containment(&p, q, CAP).unwrap();

    let pass = score(&point(&hq, 0.5, 0.4));
    assert!(pass.is_entailed);
    assert_eq!((pass.violation_energy, pass.soft_confidence), (0.0, 1.0));

    // Topic shift 3 against tolerance 1: v_e = 2, unit weights, E = 4 / 3.
    let topic = score(&point(&hq, 3.0, 0.0));
    assert!(!topic.is_entailed);
    assert!((topic.violation_energy - 4.0 / 3.0).abs() < 1e-12);
    assert!((topic.soft_confidence - (-2.0_f64 / 3.0).exp()).abs() < 1e-12);
    // Sphere angle 1.5 against 1: v_s = 0.5, E = 0.25 / 3. Misses add.
    let sphere = score(&point(&hq, 0.0, 1.5));
    assert!((sphere.violation_energy - 0.25 / 3.0).abs() < 1e-12);
    let both = score(&point(&hq, 3.0, 1.5));
    assert!((both.violation_energy - 4.25 / 3.0).abs() < 1e-12);

    // Depth: a shallower collinear question misses by d(0, p) - d(0, q), in cone.
    let shallow = m
        .busemann_containment(&point(&hq, 0.0, 0.0), &point(&hp, 0.0, 0.0), CAP)
        .unwrap();
    let deficit = 2.0 * (0.8_f64.atanh() - 0.3_f64.atanh());
    assert!(shallow.in_cone && !shallow.deeper && !shallow.is_entailed);
    assert!((shallow.violation_energy - deficit * deficit / 3.0).abs() < 1e-9);
    // Cone: 0.32 rad off axis against the aperture asin(0.1 * 0.91 / 0.3) = 0.308
    // at p, still deeper.
    let off_axis = score(&point(&radial(1.0, 0.6, &dir_at(0.32)), 0.0, 0.0));
    assert!(!off_axis.in_cone && off_axis.deeper);
    let v_cone = (off_axis.cone_angle - off_axis.aperture) / off_axis.aperture;
    assert!((off_axis.violation_energy - v_cone * v_cone / 3.0).abs() < 1e-12);

    // Continuous at the threshold and nondecreasing beyond it.
    let edge = score(&point(&hq, 1.0 + 1e-9, 0.0));
    assert!(!edge.is_entailed && !edge.topic_aligned);
    assert!(edge.violation_energy > 0.0 && edge.violation_energy < 1e-17);
    assert!(edge.soft_confidence > 1.0 - 1e-12);
    let mut last = 0.0;
    for step in 0..=40 {
        let s = score(&point(&hq, 1.0 + 0.1 * f64::from(step), 0.0));
        assert!(s.violation_energy >= last && s.soft_confidence <= 1.0);
        assert!((s.soft_confidence - (-0.5 * s.violation_energy).exp()).abs() < 1e-15);
        last = s.violation_energy;
    }
    assert!(last > 5.0);

    // The metric weights weigh the factors: alpha_e = 4 gives 4 * 4 / 6.
    let weighted = ProductManifold::from_preset_with(
        TopologyPreset::Boolq128d,
        GeometryParams {
            curvature: 1.0,
            radius: 1.0,
            alpha_h: 1.0,
            alpha_e: 4.0,
            alpha_s: 1.0,
        },
        m.epochs().clone(),
    )
    .unwrap();
    let heavy = weighted
        .busemann_containment(&p, &point(&hq, 3.0, 0.0), CAP)
        .unwrap();
    assert!((heavy.violation_energy - 16.0 / 6.0).abs() < 1e-12);

    // An energy that overflows is refused, not reported as zero confidence.
    let tiny_tol = ContainmentCriteria {
        topic_shift_tol: 1e-200,
        ..ContainmentCriteria::from_cone_half_angle(CAP)
    };
    assert_eq!(
        m.busemann_containment_with(&p, &point(&hq, 3.0, 0.0), &tiny_tol),
        Err(LodError::Geometry(Reject::NonFiniteState))
    );
}
