//! gen-zero-lod Node Representation & Epistemic State Machine.
//!
//! Provides 4-tier Lod bands (Lod 0..3), 256-bit HDC binary fingerprints,
//! and strict epistemic verification status machine.

use crate::error::LodError;
use crate::manifold::MixedCurvatureCoord;
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

/// Epistemic Lifecycle State Machine.
///
/// Follows Pearl's causality ladder:
/// Hypothesized --[Validated]--> Validated
/// Hypothesized --[Falsified]--> Falsified
/// Axiomatic (Frozen system invariants)
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicStatus {
    /// Conjectured hypothesis pending real intervention verification.
    #[serde(alias = "conjectured")]
    Hypothesized = 0,
    /// Empirically verified through environment intervention or formal proof.
    Validated = 1,
    /// Falsified by counterexample or contradiction; targeted for cascading prune.
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
    /// 16-coordinate mixed-curvature coordinate (H^4 x S^3 x R^8).
    pub coord: MixedCurvatureCoord,
    /// 256-bit HDC fingerprint for Stage-1 sub-microsecond filtering.
    pub hdc_fingerprint: [u64; 4],
    /// Human-readable label or entity name.
    pub label: String,
    /// Datalog/Formal entity identifier (used in GraphFactProvider).
    pub entity_id: u64,
    /// Epistemic confidence level [0.0, 1.0].
    pub confidence: f32,
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
            confidence: 0.5,
            parent_id: None,
        }
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
}
