//! gen-zero-lod Node Representation & Epistemic State Machine.
//!
//! Provides 4-tier Lod bands (Lod 0..3), the map from a continuous scale and
//! from a hyperbolic coordinate to a band, 256-bit HDC binary fingerprints,
//! and strict epistemic verification status machine.

use crate::error::LodError;
use crate::manifold::{GeometryParams, MixedCurvatureCoord, COORD_BOUNDARY_FLOOR};
use serde::{Deserialize, Serialize};

/// Cognitive multi-scale Level of Detail (Lod) bands.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LodBand {
    /// Lod 0: Atomic Frame (raw sensor / single-step action / token frame)
    Lod0Atomic = 0,
    /// Lod 1: Local State Cluster (sub-goals / path segments)
    Lod1Cluster = 1,
    /// Lod 2: Macro Milestone (composite subtasks / phases)
    Lod2Milestone = 2,
    /// Lod 3: Systemic Objective (global policy horizon / root intent)
    Lod3Systemic = 3,
}

impl LodBand {
    /// Coarsen (zoom out) to higher abstraction band.
    #[inline]
    pub fn zoom_out(self) -> Result<Self, LodError> {
        match self {
            Self::Lod0Atomic => Ok(Self::Lod1Cluster),
            Self::Lod1Cluster => Ok(Self::Lod2Milestone),
            Self::Lod2Milestone => Ok(Self::Lod3Systemic),
            Self::Lod3Systemic => Err(LodError::SpineBreatheOutOfBounds {
                direction: "zoom_out",
                level: 3,
            }),
        }
    }

    /// Refine (zoom in) to lower concrete band.
    #[inline]
    pub fn zoom_in(self) -> Result<Self, LodError> {
        match self {
            Self::Lod3Systemic => Ok(Self::Lod2Milestone),
            Self::Lod2Milestone => Ok(Self::Lod1Cluster),
            Self::Lod1Cluster => Ok(Self::Lod0Atomic),
            Self::Lod0Atomic => Err(LodError::SpineBreatheOutOfBounds {
                direction: "zoom_in",
                level: 0,
            }),
        }
    }
}

/// Direction of one [`LodBand`] step.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZoomDirection {
    /// Refine: one band down ([`LodBand::zoom_in`]).
    In,
    /// Coarsen: one band up ([`LodBand::zoom_out`]).
    Out,
}

/// Deepest normalized hyperbolic depth a node chart can hold:
/// `rho_max = 2 artanh(sqrt(1 - floor))`, where `floor` is the boundary floor
/// `1 - c ||x_H||^2 >= 1e-4` that [`MixedCurvatureCoord::with_curvature`]
/// enforces. About 10.5966, the same for every curvature.
pub fn max_chart_depth() -> f64 {
    2.0 * (1.0 - f64::from(COORD_BOUNDARY_FLOOR)).sqrt().atanh()
}

/// Width of one band on the scale axis: a quarter of [`max_chart_depth`], so
/// the four bands split the representable depth range evenly.
pub fn band_scale_width() -> f64 {
    max_chart_depth() / 4.0
}

/// Normalized hyperbolic depth of a Poincare-ball point under curvature `-c`:
/// `rho = sqrt(c) d_H(0, x) = 2 artanh(sqrt(c) ||x||)`. Dimensionless, so the
/// band map does not change when the curvature rescales the ball. Refuses a
/// curvature that is not finite and positive and a point on or outside the
/// ball.
pub fn normalized_depth(hyperbolic: &[f32; 4], curvature: f64) -> Result<f64, LodError> {
    if !(curvature.is_finite() && curvature > 0.0) {
        return Err(crate::manifold::Reject::DomainViolation.into());
    }
    let norm_sq: f64 = hyperbolic.iter().map(|&v| f64::from(v).powi(2)).sum();
    let r = curvature.sqrt() * norm_sq.sqrt();
    if !(r.is_finite() && r < 1.0) {
        return Err(LodError::HyperbolicBoundaryViolation {
            norm_sq: norm_sq as f32,
        });
    }
    Ok(2.0 * r.atanh())
}

/// Coarse-graining scale of a Poincare-ball point: `t = max(0, rho_max - rho)`.
///
/// The convention is the usual one for hierarchies in hyperbolic space: general
/// concepts sit near the origin, specific ones near the boundary. So the origin
/// has the largest scale (`rho_max`) and a point at the chart's boundary floor
/// has scale 0. A point deeper than the floor (possible for a coordinate built
/// without [`MixedCurvatureCoord::with_curvature`]) is finer than every band
/// boundary and gets scale 0.
pub fn scale_from_depth(rho: f64) -> f64 {
    (max_chart_depth() - rho).max(0.0)
}

/// Band of a continuous coarse-graining scale `t >= 0`. The bands are the four
/// intervals of width [`band_scale_width`]: `[0, w)` is `Lod0Atomic`,
/// `[w, 2w)` `Lod1Cluster`, `[2w, 3w)` `Lod2Milestone`, `[3w, inf)`
/// `Lod3Systemic`. A negative or non-finite scale is refused, never clamped.
pub fn band_from_scale(scale: f32) -> Result<LodBand, LodError> {
    if !(scale.is_finite() && scale >= 0.0) {
        return Err(LodError::InvalidQuery(format!(
            "scale must be finite and nonnegative, got {scale}"
        )));
    }
    let steps = f64::from(scale) / band_scale_width();
    Ok(if steps < 1.0 {
        LodBand::Lod0Atomic
    } else if steps < 2.0 {
        LodBand::Lod1Cluster
    } else if steps < 3.0 {
        LodBand::Lod2Milestone
    } else {
        LodBand::Lod3Systemic
    })
}

/// Epistemic Lifecycle State Machine.
///
/// `Axiomatic` is frozen. The other three are moved by
/// `LodGraph::evolve_epistemic_fixed_point` with hysteresis on the node's
/// fixed-point confidence `c`: `c < theta_lo` gives `Falsified`, `c > theta_hi`
/// gives `Validated`, and in between the status is kept. A node refuted by
/// direct evidence (`LodNode::refuted`) is `Falsified` until that evidence is
/// retracted.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicStatus {
    /// Conjectured hypothesis pending real intervention verification.
    #[serde(alias = "conjectured")]
    Hypothesized = 0,
    /// Empirically verified through environment intervention or formal proof.
    Validated = 1,
    /// Falsified by direct evidence, or because its confidence fell below `theta_lo`.
    Falsified = 2,
    /// Axiomatically frozen truth; immutable root premises.
    #[serde(alias = "frozen")]
    Axiomatic = 3,
}

impl EpistemicStatus {
    #[inline]
    pub fn is_active_truth(self) -> bool {
        matches!(self, Self::Validated | Self::Axiomatic)
    }

    #[inline]
    pub fn is_falsified(self) -> bool {
        matches!(self, Self::Falsified)
    }
}

/// 256-bit Hyperdimensional Computing (HDC) binary fingerprint.
/// Enables POPCNT-accelerated sub-microsecond coarse similarity search.
#[inline(always)]
pub fn hdc_hamming_distance_256(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    (a[0] ^ b[0]).count_ones()
        + (a[1] ^ b[1]).count_ones()
        + (a[2] ^ b[2]).count_ones()
        + (a[3] ^ b[3]).count_ones()
}

/// A node in the LodGraph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LodNode {
    /// Unique node identifier within the LodGraph.
    pub id: u32,
    /// Hierarchical Lod band.
    pub band: LodBand,
    /// Current epistemic lifecycle status.
    pub status: EpistemicStatus,
    /// 16-coordinate chart (H^4 x R^8 x S^3) under the owning graph's geometry.
    pub coord: MixedCurvatureCoord,
    /// 256-bit HDC fingerprint for Stage-1 sub-microsecond filtering.
    pub hdc_fingerprint: [u64; 4],
    /// Human-readable label or entity name.
    pub label: String,
    /// Datalog/Formal entity identifier (used in GraphFactProvider).
    pub entity_id: u64,
    /// Evidence prior `pi` in [0.0, 1.0]: the confidence this node has on its own
    /// evidence, before its dependencies are counted. Fixed at insert.
    pub prior: f32,
    /// Posterior confidence in [0.0, 1.0]: the prior until the first
    /// `LodGraph::evolve_epistemic_fixed_point`, then the fixed-point value.
    pub confidence: f32,
    /// Refuted by direct evidence: pinned to confidence 0 and `Falsified` until
    /// `LodGraph::retract_falsification`. The graph sets it; an inserted
    /// `Falsified` node is refuted.
    pub refuted: bool,
    /// Optional parent node in the hierarchy.
    pub parent_id: Option<u32>,
}

impl LodNode {
    pub fn new(
        id: u32,
        band: LodBand,
        coord: MixedCurvatureCoord,
        label: impl Into<String>,
        entity_id: u64,
    ) -> Self {
        Self {
            id,
            band,
            status: EpistemicStatus::Hypothesized,
            coord,
            hdc_fingerprint: [0; 4],
            label: label.into(),
            entity_id,
            prior: 0.5,
            confidence: 0.5,
            refuted: false,
            parent_id: None,
        }
    }

    /// Set the evidence prior. The posterior starts equal to it.
    pub fn with_prior(mut self, prior: f32) -> Self {
        self.prior = prior;
        self.confidence = prior;
        self
    }

    /// Set HDC fingerprint.
    pub fn with_hdc_fingerprint(mut self, fp: [u64; 4]) -> Self {
        self.hdc_fingerprint = fp;
        self
    }

    /// Set epistemic status.
    pub fn with_status(mut self, status: EpistemicStatus) -> Self {
        self.status = status;
        self
    }

    /// The band this node's coordinate implies under `geometry`: its hyperbolic
    /// block's normalized depth, turned into a scale by [`scale_from_depth`]
    /// and into a band by [`band_from_scale`]. Refuses an invalid geometry and a
    /// hyperbolic block on or outside the ball of `geometry.curvature`.
    pub fn derive_band_from_coord(&self, geometry: &GeometryParams) -> Result<LodBand, LodError> {
        geometry.validate()?;
        let rho = normalized_depth(&self.coord.hyperbolic, geometry.curvature)?;
        band_from_scale(scale_from_depth(rho) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A coordinate at normalized depth `rho` along the first axis of the ball
    /// of curvature `c`.
    fn at_depth(rho: f64, c: f64) -> MixedCurvatureCoord {
        let r = ((rho / 2.0).tanh() / c.sqrt()) as f32;
        MixedCurvatureCoord::with_curvature(
            [r, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0; 8],
            c as f32,
        )
        .unwrap()
    }

    fn node(coord: MixedCurvatureCoord) -> LodNode {
        LodNode::new(0, LodBand::Lod0Atomic, coord, "n", 1)
    }

    #[test]
    fn band_from_scale_splits_the_scale_axis_into_four_intervals() {
        let w = band_scale_width() as f32;
        assert!((max_chart_depth() - 10.596_585).abs() < 1e-5);
        assert_eq!(band_from_scale(0.0).unwrap(), LodBand::Lod0Atomic);
        assert_eq!(band_from_scale(0.99 * w).unwrap(), LodBand::Lod0Atomic);
        assert_eq!(band_from_scale(1.01 * w).unwrap(), LodBand::Lod1Cluster);
        assert_eq!(band_from_scale(2.5 * w).unwrap(), LodBand::Lod2Milestone);
        assert_eq!(band_from_scale(3.01 * w).unwrap(), LodBand::Lod3Systemic);
        assert_eq!(band_from_scale(1e9).unwrap(), LodBand::Lod3Systemic);
        // Monotone: a larger scale is never a finer band.
        let mut last = LodBand::Lod0Atomic;
        for i in 0..400 {
            let band = band_from_scale(i as f32 * 0.05).unwrap();
            assert!(band >= last);
            last = band;
        }
        for bad in [-1e-6, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(band_from_scale(bad).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn derive_band_puts_the_origin_on_top_and_the_boundary_at_the_bottom() {
        let unit = GeometryParams::UNIT;
        let rho_max = max_chart_depth();
        let origin = node(MixedCurvatureCoord::origin());
        assert_eq!(
            origin.derive_band_from_coord(&unit).unwrap(),
            LodBand::Lod3Systemic
        );
        // Band centers, from the top band down.
        let w = band_scale_width();
        for (k, band) in [
            LodBand::Lod3Systemic,
            LodBand::Lod2Milestone,
            LodBand::Lod1Cluster,
            LodBand::Lod0Atomic,
        ]
        .into_iter()
        .enumerate()
        {
            let rho = (k as f64 + 0.5) * w;
            assert!(rho < rho_max);
            assert_eq!(
                node(at_depth(rho, 1.0))
                    .derive_band_from_coord(&unit)
                    .unwrap(),
                band,
                "depth {rho}"
            );
        }
    }

    #[test]
    fn derive_band_is_invariant_under_curvature_rescaling() {
        for c in [0.25, 1.0, 4.0] {
            let geometry = GeometryParams {
                curvature: c,
                ..GeometryParams::UNIT
            };
            for (rho, band) in [(1.0, LodBand::Lod3Systemic), (9.5, LodBand::Lod0Atomic)] {
                assert_eq!(
                    node(at_depth(rho, c))
                        .derive_band_from_coord(&geometry)
                        .unwrap(),
                    band,
                    "c {c} depth {rho}"
                );
            }
        }
    }

    #[test]
    fn derive_band_refuses_a_point_outside_the_ball_and_a_bad_geometry() {
        // Inside the unit ball, outside the ball of curvature 4.
        let coord = at_depth(3.0, 1.0);
        let steep = GeometryParams {
            curvature: 4.0,
            ..GeometryParams::UNIT
        };
        assert!(matches!(
            node(coord).derive_band_from_coord(&steep),
            Err(LodError::HyperbolicBoundaryViolation { .. })
        ));
        let broken = GeometryParams {
            curvature: f64::NAN,
            ..GeometryParams::UNIT
        };
        assert!(node(coord).derive_band_from_coord(&broken).is_err());
    }

    #[test]
    fn zoom_steps_are_inverse_and_bounded() {
        for band in [
            LodBand::Lod0Atomic,
            LodBand::Lod1Cluster,
            LodBand::Lod2Milestone,
        ] {
            assert_eq!(band.zoom_out().unwrap().zoom_in().unwrap(), band);
        }
        assert!(LodBand::Lod3Systemic.zoom_out().is_err());
        assert!(LodBand::Lod0Atomic.zoom_in().is_err());
    }
}
