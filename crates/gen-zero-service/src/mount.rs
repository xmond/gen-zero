//! Spec 25 §5.2 to §5.4: immutable mount snapshots and the CAS atomic swap.
//!
//! What is here:
//! - [`MountSnapshot`]: an immutable, SHA-256 sealed generation. Every content
//!   change goes through [`MountSnapshot::derive`], which bumps [`Version`] by
//!   exactly one. Nothing can mutate a snapshot that a reader already holds.
//! - [`AtomicMountRegistry`]: the [`MountRegistry`] implementation. One
//!   `ArcSwap` cell holds `(snapshot, data watermark)`, so a single pointer
//!   compare-and-swap covers both the version check and the watermark check.
//! - [`RequestBinding`]: a request captures one snapshot and keeps it. Any
//!   artefact from another generation is refused with `Reject::EpochMismatch`.
//! - Production callers: `PolymorphicZeroEngine::execute` captures a binding
//!   per request; `PolymorphicZeroEngine::publish_assets` (HTTP
//!   `POST /v1/mounts`, CLI `--mount-assets`) runs validate + CAS.
//!
//! What is NOT here (Spec 25 marks all of it as later stages):
//! - The slow loop (tension -> refinement proposal) and the deposit writer.
//!   No production source of either exists yet, so neither is shipped.
//! - Durable prepare/commit and restart recovery. The swap is in memory only.
//!   A memory pointer swap is not a durable publish (§5.3 step 4).
//! - The sub-chart geometry itself (projection re-estimation, restriction
//!   maps). A [`Proposal`] is a record of *why* and *from which generation*.
//! - `ProductGeometry`. It lives outside this crate, so the snapshot binds the
//!   geometry by `Epochs::geometry` (a digest) instead of holding an object.
//!
//! Nothing here falls back silently. Every failure is a [`Reject`].

use arc_swap::{ArcSwap, Guard};
use sha2::{Digest as Sha2Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

/// SHA-256 output.
pub type Digest = [u8; 32];
pub type Result<T> = std::result::Result<T, Reject>;

/// Monotonic generation number. Never reused, never moved backwards.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Version(pub u64);

impl Version {
    /// The next generation. Overflow is refused, never wrapped.
    pub fn next(self) -> Result<Version> {
        self.0
            .checked_add(1)
            .map(Version)
            .ok_or(Reject::BudgetExceeded)
    }
}

/// Generation number plus one content digest per asset family.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Epochs {
    pub version: Version,
    pub model: Digest,
    pub geometry: Digest,
    pub atlas: Digest,
    pub graph: Digest,
    pub policy: Digest,
}

impl Epochs {
    fn digests(&self) -> [&Digest; 5] {
        [
            &self.model,
            &self.geometry,
            &self.atlas,
            &self.graph,
            &self.policy,
        ]
    }
}

/// `(tenant, workspace)`: the unit that owns one mount.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MountKey {
    pub tenant: String,
    pub workspace: String,
}

impl MountKey {
    pub fn new(tenant: impl Into<String>, workspace: impl Into<String>) -> Self {
        Self {
            tenant: tenant.into(),
            workspace: workspace.into(),
        }
    }
}

/// Spec 25 §5.1 rejection codes. Every code keeps its exact §5.1 meaning.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum Reject {
    #[error("energy rose")]
    EnergyRose,
    #[error("energy decrease not provable")]
    UncertainEnergy,
    #[error("no progress")]
    Stalled,
    #[error("residual over budget")]
    ResidualExceeded,
    #[error("pinned value moved")]
    PinnedMoved,
    #[error("non-finite value")]
    NonFiniteState,
    #[error("geometry domain violation")]
    DomainViolation,
    #[error("cut locus")]
    CutLocus,
    #[error("fiber mismatch")]
    FiberMismatch,
    #[error("epoch mismatch: artefact belongs to another generation")]
    EpochMismatch,
    #[error("cocycle violation")]
    CocycleViolation,
    #[error("obstruction")]
    Obstruction,
    #[error("not converged")]
    NotConverged,
    #[error("budget exceeded")]
    BudgetExceeded,
    #[error("no feasible expert")]
    NoFeasibleExpert,
    #[error("ambiguous action")]
    AmbiguousAction,
    #[error("invalid certificate")]
    InvalidCertificate,
    #[error("coverage lost: no mount for this key")]
    CoverageLost,
    #[error("deposit conflict: the data watermark moved")]
    DepositConflict,
    #[error("cas conflict: the mounted generation is not the sealed base")]
    CasConflict,
    #[error("backend unavailable")]
    BackendUnavailable,
    #[error("unsupported operator family")]
    UnsupportedOperatorFamily,
}

impl Reject {
    /// The exact §5.1 name, used as the typed error code on every entry.
    pub fn code(self) -> &'static str {
        match self {
            Self::EnergyRose => "EnergyRose",
            Self::UncertainEnergy => "UncertainEnergy",
            Self::Stalled => "Stalled",
            Self::ResidualExceeded => "ResidualExceeded",
            Self::PinnedMoved => "PinnedMoved",
            Self::NonFiniteState => "NonFiniteState",
            Self::DomainViolation => "DomainViolation",
            Self::CutLocus => "CutLocus",
            Self::FiberMismatch => "FiberMismatch",
            Self::EpochMismatch => "EpochMismatch",
            Self::CocycleViolation => "CocycleViolation",
            Self::Obstruction => "Obstruction",
            Self::NotConverged => "NotConverged",
            Self::BudgetExceeded => "BudgetExceeded",
            Self::NoFeasibleExpert => "NoFeasibleExpert",
            Self::AmbiguousAction => "AmbiguousAction",
            Self::InvalidCertificate => "InvalidCertificate",
            Self::CoverageLost => "CoverageLost",
            Self::DepositConflict => "DepositConflict",
            Self::CasConflict => "CasConflict",
            Self::BackendUnavailable => "BackendUnavailable",
            Self::UnsupportedOperatorFamily => "UnsupportedOperatorFamily",
        }
    }

    /// HTTP status of an entry that refuses with this code. Malformed or
    /// out-of-domain input is 400, a generation conflict is 409, a well-formed
    /// request the gate refuses to certify is 422, a key with no mount is 404,
    /// a missing backend is 503.
    pub fn http_status(self) -> u16 {
        match self {
            Self::DomainViolation
            | Self::CutLocus
            | Self::NonFiniteState
            | Self::FiberMismatch
            | Self::UnsupportedOperatorFamily => 400,
            Self::EpochMismatch | Self::CasConflict | Self::DepositConflict => 409,
            Self::CoverageLost => 404,
            Self::BackendUnavailable => 503,
            Self::EnergyRose
            | Self::UncertainEnergy
            | Self::Stalled
            | Self::ResidualExceeded
            | Self::PinnedMoved
            | Self::CocycleViolation
            | Self::Obstruction
            | Self::NotConverged
            | Self::BudgetExceeded
            | Self::NoFeasibleExpert
            | Self::AmbiguousAction
            | Self::InvalidCertificate => 422,
        }
    }
}

/// Resource and policy contract for one validation.
#[derive(Clone, Debug)]
pub struct Budget {
    /// Consumed by the geometry gate, not by mount validation.
    pub max_steps: usize,
    /// Validation wall time limit in nanoseconds.
    pub max_time_ns: u64,
    /// Upper bound on `assets` bytes of the candidate.
    pub max_bytes: usize,
    pub residual_limit: f64,
    pub numeric_error: f64,
    /// Digest of the validation policy. It is bound into the seal.
    pub policy: Digest,
}

const DOMAIN_SNAPSHOT: &[u8] = b"gen-zero/mount-snapshot/v1\0";
const DOMAIN_PROPOSAL: &[u8] = b"gen-zero/mount-proposal/v1\0";
const DOMAIN_SEAL: &[u8] = b"gen-zero/mount-seal/v1\0";

fn sha256(parts: &[&[u8]]) -> Digest {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// Lower-case hex of a digest, for logs and `_meta`.
pub fn digest_hex(digest: &Digest) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn is_zero(digest: &Digest) -> bool {
    digest.iter().all(|b| *b == 0)
}

/// Content digests of one generation, before a version is assigned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetDigests {
    pub model: Digest,
    pub geometry: Digest,
    pub atlas: Digest,
    pub graph: Digest,
    pub policy: Digest,
}

/// A requested content change. `None` keeps the base value.
#[derive(Clone, Debug, Default)]
pub struct SnapshotChange {
    pub model: Option<Digest>,
    pub geometry: Option<Digest>,
    pub atlas: Option<Digest>,
    pub graph: Option<Digest>,
    pub policy: Option<Digest>,
    pub assets: Option<Arc<[u8]>>,
    pub watermark: Option<u64>,
}

/// One immutable generation. It has no `&mut` API and its digest covers every
/// field, including the full `assets` bytes.
#[derive(Debug)]
pub struct MountSnapshot {
    key: MountKey,
    epochs: Epochs,
    watermark: u64,
    assets: Arc<[u8]>,
    digest: Digest,
}

/// Spec 25 §5.4 name for [`MountSnapshot`].
pub type Snapshot = MountSnapshot;

impl MountSnapshot {
    /// First generation of a key. It gets `Version(1)`.
    pub fn genesis(
        key: MountKey,
        digests: AssetDigests,
        watermark: u64,
        assets: Arc<[u8]>,
    ) -> Result<Self> {
        let epochs = Epochs {
            version: Version(1),
            model: digests.model,
            geometry: digests.geometry,
            atlas: digests.atlas,
            graph: digests.graph,
            policy: digests.policy,
        };
        Self::seal(key, epochs, watermark, assets)
    }

    fn seal(key: MountKey, epochs: Epochs, watermark: u64, assets: Arc<[u8]>) -> Result<Self> {
        if key.tenant.is_empty() || key.workspace.is_empty() {
            return Err(Reject::InvalidCertificate);
        }
        if epochs.digests().iter().any(|d| is_zero(d)) {
            return Err(Reject::InvalidCertificate);
        }
        let digest = Self::compute_digest(&key, &epochs, watermark, &assets);
        Ok(Self {
            key,
            epochs,
            watermark,
            assets,
            digest,
        })
    }

    /// Length-prefixed canonical encoding, so no two field layouts collide.
    fn compute_digest(key: &MountKey, epochs: &Epochs, watermark: u64, assets: &[u8]) -> Digest {
        let tenant_len = (key.tenant.len() as u64).to_le_bytes();
        let workspace_len = (key.workspace.len() as u64).to_le_bytes();
        let version = epochs.version.0.to_le_bytes();
        let watermark = watermark.to_le_bytes();
        let assets_len = (assets.len() as u64).to_le_bytes();
        sha256(&[
            DOMAIN_SNAPSHOT,
            &tenant_len,
            key.tenant.as_bytes(),
            &workspace_len,
            key.workspace.as_bytes(),
            &version,
            &epochs.model,
            &epochs.geometry,
            &epochs.atlas,
            &epochs.graph,
            &epochs.policy,
            &watermark,
            &assets_len,
            assets,
        ])
    }

    /// Recompute the digest from the contents and compare with the stored one.
    pub fn verify_digest(&self) -> bool {
        Self::compute_digest(&self.key, &self.epochs, self.watermark, &self.assets) == self.digest
    }

    /// Build the next generation. The version rises by exactly one. A change
    /// that alters nothing is `Stalled`: it would only burn a version.
    pub fn derive(&self, change: &SnapshotChange) -> Result<MountSnapshot> {
        let epochs = Epochs {
            version: self.epochs.version.next()?,
            model: change.model.unwrap_or(self.epochs.model),
            geometry: change.geometry.unwrap_or(self.epochs.geometry),
            atlas: change.atlas.unwrap_or(self.epochs.atlas),
            graph: change.graph.unwrap_or(self.epochs.graph),
            policy: change.policy.unwrap_or(self.epochs.policy),
        };
        let watermark = change.watermark.unwrap_or(self.watermark);
        if watermark < self.watermark {
            // The data watermark never goes back.
            return Err(Reject::InvalidCertificate);
        }
        let assets = change
            .assets
            .clone()
            .unwrap_or_else(|| Arc::clone(&self.assets));
        let unchanged = epochs.digests() == self.epochs.digests()
            && watermark == self.watermark
            && *assets == *self.assets;
        if unchanged {
            return Err(Reject::Stalled);
        }
        Self::seal(self.key.clone(), epochs, watermark, assets)
    }

    pub fn key(&self) -> &MountKey {
        &self.key
    }
    pub fn epochs(&self) -> &Epochs {
        &self.epochs
    }
    pub fn version(&self) -> Version {
        self.epochs.version
    }
    pub fn watermark(&self) -> u64 {
        self.watermark
    }
    pub fn digest(&self) -> &Digest {
        &self.digest
    }
    pub fn assets(&self) -> &[u8] {
        &self.assets
    }
}

/// A reason to change a mount, bound to the exact generation it was seen on.
#[derive(Clone, Debug)]
pub struct Proposal {
    base: Arc<MountSnapshot>,
    plan: Arc<[u8]>,
    digest: Digest,
}

impl Proposal {
    fn digest_of(base: &MountSnapshot, plan: &[u8]) -> Digest {
        sha256(&[
            DOMAIN_PROPOSAL,
            &base.digest,
            &(plan.len() as u64).to_le_bytes(),
            plan,
        ])
    }

    pub fn new(base: Arc<MountSnapshot>, plan: Arc<[u8]>) -> Result<Self> {
        if plan.is_empty() {
            return Err(Reject::InvalidCertificate);
        }
        let digest = Self::digest_of(&base, &plan);
        Ok(Self { base, plan, digest })
    }

    pub fn base(&self) -> &Arc<MountSnapshot> {
        &self.base
    }
    pub fn plan(&self) -> &[u8] {
        &self.plan
    }
    pub fn digest(&self) -> &Digest {
        &self.digest
    }
}

/// The next generation, built off-line and not yet mounted.
#[derive(Clone, Debug)]
pub struct CandidateMount {
    next: Arc<MountSnapshot>,
    proposal: Digest,
}

impl CandidateMount {
    pub fn new(next: Arc<MountSnapshot>, proposal: &Proposal) -> Self {
        Self {
            next,
            proposal: proposal.digest,
        }
    }

    pub fn next(&self) -> &Arc<MountSnapshot> {
        &self.next
    }
}

/// A candidate that passed [`MountRegistry::validate`]. Only `validate` can
/// build one, and it is consumed by `compare_and_mount`: a seal is used once.
#[derive(Debug)]
pub struct ValidatedMount {
    candidate: CandidateMount,
    base_version: Version,
    base_digest: Digest,
    watermark: u64,
    validation_digest: Digest,
}

impl ValidatedMount {
    pub fn base_version(&self) -> Version {
        self.base_version
    }
    pub fn watermark(&self) -> u64 {
        self.watermark
    }
    pub fn validation_digest(&self) -> &Digest {
        &self.validation_digest
    }
    pub fn next(&self) -> &Arc<MountSnapshot> {
        &self.candidate.next
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Published {
    pub from: Version,
    pub to: Version,
    pub digest: Digest,
}

pub trait MountRegistry: Send + Sync {
    fn load(&self, key: &MountKey) -> Result<Arc<Snapshot>>;
    fn validate(
        &self,
        proposal: Proposal,
        candidate: CandidateMount,
        budget: &Budget,
    ) -> Result<ValidatedMount>;
    fn compare_and_mount(&self, key: &MountKey, sealed: ValidatedMount) -> Result<Published>;
}

/// The value behind the one atomic pointer of a key. It is immutable: a
/// deposit or a mount builds a new cell and swaps the pointer.
#[derive(Debug)]
struct MountCell {
    snapshot: Arc<MountSnapshot>,
    /// Live data watermark: how many deposits the workspace has accepted.
    watermark: u64,
}

type SharedCell = Arc<ArcSwap<MountCell>>;

/// Lock-free registry of mounts (reads and swaps use `arc_swap`, no mutex).
#[derive(Default)]
pub struct AtomicMountRegistry {
    cells: ArcSwap<HashMap<MountKey, SharedCell>>,
}

/// Swap `new` in only if the cell still holds exactly `expected`.
fn swap_if_current<T>(cell: &ArcSwap<T>, expected: &Arc<T>, new: Arc<T>) -> bool {
    let previous: Arc<T> = Guard::into_inner(cell.compare_and_swap(expected, new));
    Arc::ptr_eq(&previous, expected)
}

impl AtomicMountRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mount the first generation of a key. A second registration of the same
    /// key is refused: only `compare_and_mount` may replace a mount.
    pub fn register(&self, genesis: MountSnapshot) -> Result<Arc<MountSnapshot>> {
        let snapshot = Arc::new(genesis);
        let cell: SharedCell = Arc::new(ArcSwap::from_pointee(MountCell {
            snapshot: Arc::clone(&snapshot),
            watermark: snapshot.watermark,
        }));
        loop {
            let current = self.cells.load_full();
            if current.contains_key(&snapshot.key) {
                return Err(Reject::CasConflict);
            }
            let mut table = HashMap::clone(&current);
            table.insert(snapshot.key.clone(), Arc::clone(&cell));
            if swap_if_current(&self.cells, &current, Arc::new(table)) {
                return Ok(snapshot);
            }
        }
    }

    fn cell(&self, key: &MountKey) -> Result<SharedCell> {
        self.cells
            .load()
            .get(key)
            .cloned()
            .ok_or(Reject::CoverageLost)
    }

    /// Which reject applies when `cell` is not what the seal expects.
    fn check_seal(cell: &MountCell, sealed: &ValidatedMount) -> Result<()> {
        if cell.snapshot.version() != sealed.base_version
            || cell.snapshot.digest != sealed.base_digest
        {
            return Err(Reject::CasConflict);
        }
        if cell.watermark != sealed.watermark {
            return Err(Reject::DepositConflict);
        }
        Ok(())
    }
}

fn check_budget(budget: &Budget) -> Result<()> {
    let finite_non_negative = |v: f64| v.is_finite() && v >= 0.0;
    if !finite_non_negative(budget.residual_limit) || !finite_non_negative(budget.numeric_error) {
        return Err(Reject::NonFiniteState);
    }
    if is_zero(&budget.policy) {
        return Err(Reject::InvalidCertificate);
    }
    Ok(())
}

impl MountRegistry for AtomicMountRegistry {
    fn load(&self, key: &MountKey) -> Result<Arc<Snapshot>> {
        Ok(Arc::clone(&self.cell(key)?.load().snapshot))
    }

    fn validate(
        &self,
        proposal: Proposal,
        candidate: CandidateMount,
        budget: &Budget,
    ) -> Result<ValidatedMount> {
        let started = Instant::now();
        check_budget(budget)?;
        let base = &proposal.base;
        let next = &candidate.next;

        if candidate.proposal != proposal.digest
            || Proposal::digest_of(base, &proposal.plan) != proposal.digest
        {
            return Err(Reject::InvalidCertificate);
        }
        if next.key != base.key {
            return Err(Reject::EpochMismatch);
        }
        if next.epochs.version != base.epochs.version.next()? {
            return Err(Reject::EpochMismatch);
        }
        if !base.verify_digest()
            || !next.verify_digest()
            || next.epochs.digests().iter().any(|d| is_zero(d))
        {
            return Err(Reject::InvalidCertificate);
        }
        if next.assets.len() > budget.max_bytes {
            return Err(Reject::BudgetExceeded);
        }

        // Fail fast on the live state. The final word is the CAS in
        // `compare_and_mount`; this only avoids sealing a doomed candidate.
        let live = self.cell(&base.key)?.load_full();
        if live.snapshot.version() != base.version() || live.snapshot.digest != base.digest {
            return Err(Reject::CasConflict);
        }
        match next.watermark.cmp(&live.watermark) {
            std::cmp::Ordering::Less => return Err(Reject::DepositConflict),
            // A candidate cannot claim data that was never deposited.
            std::cmp::Ordering::Greater => return Err(Reject::InvalidCertificate),
            std::cmp::Ordering::Equal => {}
        }
        if u128::from(budget.max_time_ns) < started.elapsed().as_nanos() {
            return Err(Reject::BudgetExceeded);
        }

        let validation_digest = sha256(&[
            DOMAIN_SEAL,
            &base.epochs.version.0.to_le_bytes(),
            &base.digest,
            &next.watermark.to_le_bytes(),
            &next.digest,
            &budget.policy,
            &(budget.max_bytes as u64).to_le_bytes(),
        ]);
        Ok(ValidatedMount {
            base_version: base.epochs.version,
            base_digest: base.digest,
            watermark: next.watermark,
            validation_digest,
            candidate,
        })
    }

    fn compare_and_mount(&self, key: &MountKey, sealed: ValidatedMount) -> Result<Published> {
        if sealed.candidate.next.key != *key {
            return Err(Reject::EpochMismatch);
        }
        let cell = self.cell(key)?;
        let expected = cell.load_full();
        Self::check_seal(&expected, &sealed)?;

        let next = Arc::new(MountCell {
            snapshot: Arc::clone(&sealed.candidate.next),
            watermark: expected.watermark,
        });
        if swap_if_current(&cell, &expected, next) {
            return Ok(Published {
                from: sealed.base_version,
                to: sealed.candidate.next.version(),
                digest: sealed.candidate.next.digest,
            });
        }
        // Someone moved the cell between our read and our swap. Say which
        // invariant broke. The seal is spent: the caller must capture again.
        Self::check_seal(&cell.load(), &sealed)?;
        Err(Reject::CasConflict)
    }
}

/// A request's view of the mount: one snapshot, captured once, kept to the end.
#[derive(Clone, Debug)]
pub struct RequestBinding {
    snapshot: Arc<MountSnapshot>,
}

impl RequestBinding {
    pub fn capture(registry: &dyn MountRegistry, key: &MountKey) -> Result<Self> {
        Ok(Self {
            snapshot: registry.load(key)?,
        })
    }

    pub fn snapshot(&self) -> &Arc<MountSnapshot> {
        &self.snapshot
    }

    /// Accept an artefact only if it carries exactly the bound generation.
    /// No migration and no re-basing happens here.
    pub fn admit(&self, epochs: &Epochs) -> Result<()> {
        if *epochs == self.snapshot.epochs {
            Ok(())
        } else {
            Err(Reject::EpochMismatch)
        }
    }

    /// Commit-point check for work that must not act on a retired generation.
    pub fn ensure_current(&self, registry: &dyn MountRegistry) -> Result<()> {
        let live = registry.load(&self.snapshot.key)?;
        self.admit(&live.epochs)
    }
}

#[cfg(test)]
mod tests;
