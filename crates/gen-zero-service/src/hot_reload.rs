//! Phase 4 of the closed-loop pipeline: hot swap of the deployed patch,
//! canary guard, and atomic rollback.
//!
//! Scope, stated plainly: a [`CompiledPatch`] is a bundle of training
//! samples (see `patch_builder`). Nothing on the inference path reads it
//! yet, and no model weights change when one is applied. What this module
//! does is real but narrow:
//!
//! - [`HotReloadManager`] verifies a patch file against its
//!   `durable_patches` row, records the deployment, and swaps the in-memory
//!   active pointer with one atomic store. Readers call
//!   [`HotReloadManager::active`] without a lock.
//! - [`CanaryGuard`] appends one `canary_metrics` row per serving decision
//!   and trips when the recent refusal or hard-stop rate exceeds its limit.
//!   Because the patch does not change the gateway yet, the metrics describe
//!   the gateway while the patch is deployed, not the patch's own effect.
//! - [`HotReloadManager::rollback`] marks the patch `ROLLED_BACK`, revokes
//!   its traces (one SQLite transaction, `DurableRefusalStore::rollback_patch`),
//!   then swaps memory back to the snapshot.
//!
//! Storage is always written first. If the storage step fails, memory is
//! left exactly as it was.

use crate::patch_builder::{validate_patch, CompiledPatch, PatchBuilderError};
use arc_swap::ArcSwapOption;
use gen_zero_storage::{
    CanaryMetricInput, CanaryStats, DurablePatchRow, DurableRefusalStore, StorageError,
};
use parking_lot::Mutex;
use serde::Serialize;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const DEFAULT_CANARY_MIN_SAMPLES: usize = 20;
pub const DEFAULT_CANARY_MAX_REFUSAL_RATE: f64 = 0.35;
pub const DEFAULT_CANARY_MAX_HARD_STOP_RATE: f64 = 0.05;
pub const DEFAULT_CANARY_WINDOW: usize = 200;

#[derive(Debug, thiserror::Error)]
pub enum HotReloadError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("patch {0} not found")]
    PatchNotFound(String),
    #[error("patch {patch_id} is not ACTIVE (status {status})")]
    PatchNotActive { patch_id: String, status: String },
    #[error("patch {patch_id} did not pass validation (validation_status {status})")]
    PatchNotValidated { patch_id: String, status: String },
    #[error("patch {0} is already the active patch")]
    AlreadyActive(String),
    #[error("patch file for {patch_id} failed verification: {source}")]
    PatchFile {
        patch_id: String,
        source: PatchBuilderError,
    },
    #[error(
        "patch {patch_id} file disagrees with its database row on {field}: db {db}, file {file}"
    )]
    IntegrityMismatch {
        patch_id: String,
        field: &'static str,
        db: String,
        file: String,
    },
    #[error("canary config invalid: {0}")]
    Config(String),
    #[error("system clock is before the UNIX epoch: {0}")]
    Clock(String),
}

/// What a rollback did to the in-memory pointers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryRollback {
    /// The rolled-back patch was active; the snapshot is active now.
    RestoredSnapshot { patch_id: String },
    /// The rolled-back patch was active and no usable snapshot existed, so
    /// no patch is active now. `snapshot_rejected` says why a snapshot that
    /// did exist was not restored.
    Cleared { snapshot_rejected: Option<String> },
    /// The rolled-back patch was the snapshot; the snapshot is now empty.
    SnapshotDropped,
    /// The rolled-back patch was neither active nor the snapshot in this
    /// process. Only storage changed.
    Untouched,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RollbackReport {
    pub patch_id: String,
    pub memory: MemoryRollback,
    /// Wall time of the storage transaction plus the pointer swap.
    pub elapsed_us: u64,
}

/// Holds the deployed patch behind an atomic pointer, plus the patch it
/// replaced (the snapshot) so a rollback can return to it.
///
/// Reads ([`Self::active`], [`Self::snapshot`]) are lock-free. Transitions
/// (apply, rollback) are serialized by `transition` so two of them can never
/// interleave their storage write and their pointer swap.
pub struct HotReloadManager {
    store: Arc<DurableRefusalStore>,
    active: ArcSwapOption<CompiledPatch>,
    snapshot: ArcSwapOption<CompiledPatch>,
    transition: Mutex<()>,
}

impl HotReloadManager {
    /// A manager with no active patch and no snapshot. Use this when the
    /// caller only needs the verified storage transitions (the CLI); a
    /// serving process uses [`Self::restore`].
    pub fn new(store: Arc<DurableRefusalStore>) -> Self {
        Self {
            store,
            active: ArcSwapOption::empty(),
            snapshot: ArcSwapOption::empty(),
            transition: Mutex::new(()),
        }
    }

    /// Rebuild the in-memory state from storage: the most recently deployed
    /// `ACTIVE` patch becomes active, the one before it the snapshot. Both
    /// are re-verified against their files.
    ///
    /// The two failures are treated differently on purpose. An active patch
    /// that fails verification is an error: a server must not start serving
    /// a deployed patch it cannot prove intact. A snapshot that fails is not
    /// served, so it is dropped with an error log and startup continues; a
    /// later rollback of the active patch then clears memory instead of
    /// restoring it. Deleting an old patch file must not stop the server.
    pub fn restore(store: Arc<DurableRefusalStore>) -> Result<Self, HotReloadError> {
        let manager = Self::new(store);
        let rows = manager.store.fetch_deployed_active_patches(2)?;
        let mut rows = rows.iter();
        if let Some(row) = rows.next() {
            let active = Arc::new(verify_row(row)?);
            tracing::info!(patch_id = %active.patch_id, "restored deployed patch");
            manager.active.store(Some(active));
        }
        if let Some(row) = rows.next() {
            match verify_row(row) {
                Ok(snapshot) => manager.snapshot.store(Some(Arc::new(snapshot))),
                Err(error) => tracing::error!(
                    snapshot = %row.patch_id,
                    %error,
                    "snapshot patch failed verification; starting without a snapshot"
                ),
            }
        }
        Ok(manager)
    }

    pub fn store(&self) -> &Arc<DurableRefusalStore> {
        &self.store
    }

    pub fn active(&self) -> Option<Arc<CompiledPatch>> {
        self.active.load_full()
    }

    pub fn snapshot(&self) -> Option<Arc<CompiledPatch>> {
        self.snapshot.load_full()
    }

    /// Verify `patch_id` and make it the active patch.
    ///
    /// Checks, in order: the row exists, is `ACTIVE` and `PASSED`, is not
    /// already active here, its file re-validates (body SHA-256, id, path,
    /// sample count) and agrees with the row on `sha256`, `patch_id`,
    /// `samples_count` and `base_model_hash`. Then `deployed_at_ms` is
    /// written, and only then is the pointer swapped, with the previous
    /// active patch kept as the snapshot. Any failure leaves memory
    /// unchanged.
    pub fn apply_patch(&self, patch_id: &str) -> Result<Arc<CompiledPatch>, HotReloadError> {
        let _guard = self.transition.lock();
        let row = self
            .store
            .get_patch(patch_id)?
            .ok_or_else(|| HotReloadError::PatchNotFound(patch_id.to_string()))?;
        if self
            .active
            .load()
            .as_ref()
            .is_some_and(|p| p.patch_id == patch_id)
        {
            return Err(HotReloadError::AlreadyActive(patch_id.to_string()));
        }
        let patch = Arc::new(verify_row(&row)?);
        self.store.mark_patch_deployed(patch_id, now_ms()?)?;
        let previous = self.active.swap(Some(Arc::clone(&patch)));
        self.snapshot.store(previous);
        tracing::info!(
            patch_id = %patch.patch_id,
            snapshot = ?self.snapshot.load().as_ref().map(|p| p.patch_id.clone()),
            "patch applied"
        );
        Ok(patch)
    }

    /// Roll `patch_id` back in storage and in memory.
    ///
    /// Storage first: `rollback_patch` marks the patch `ROLLED_BACK` and
    /// revokes every trace it consumed, in one transaction. Any storage
    /// error (unknown patch, `PatchAlreadyRolledBack`) returns before memory
    /// is touched. Then memory follows the patch's identity, see
    /// [`MemoryRollback`].
    pub fn rollback(&self, patch_id: &str, reason: &str) -> Result<RollbackReport, HotReloadError> {
        let _guard = self.transition.lock();
        let started = Instant::now();
        self.store.rollback_patch(patch_id, reason)?;
        let memory = self.retire_in_memory(patch_id);
        let elapsed_us = started.elapsed().as_micros() as u64;
        tracing::warn!(patch_id, reason, ?memory, elapsed_us, "patch rolled back");
        Ok(RollbackReport {
            patch_id: patch_id.to_string(),
            memory,
            elapsed_us,
        })
    }

    /// Drop `patch_id` from memory after another process rolled it back in
    /// storage. A serving process learns this when a canary write fails with
    /// `PatchNotActive`.
    pub fn reconcile_external_rollback(&self, patch_id: &str) -> MemoryRollback {
        let _guard = self.transition.lock();
        let memory = self.retire_in_memory(patch_id);
        tracing::warn!(
            patch_id,
            ?memory,
            "patch was rolled back by another process; memory reconciled"
        );
        memory
    }

    /// Caller holds `transition`.
    fn retire_in_memory(&self, patch_id: &str) -> MemoryRollback {
        let is = |slot: &ArcSwapOption<CompiledPatch>| {
            slot.load().as_ref().is_some_and(|p| p.patch_id == patch_id)
        };
        if is(&self.active) {
            let snapshot = self.snapshot.swap(None);
            let (restored, rejected) = match snapshot {
                None => (None, None),
                Some(snap) => match self.snapshot_still_active(&snap) {
                    Ok(()) => (Some(snap), None),
                    Err(reason) => {
                        tracing::error!(snapshot = %snap.patch_id, %reason, "snapshot not restored");
                        (None, Some(reason))
                    }
                },
            };
            let memory = match &restored {
                Some(snap) => MemoryRollback::RestoredSnapshot {
                    patch_id: snap.patch_id.clone(),
                },
                None => MemoryRollback::Cleared {
                    snapshot_rejected: rejected,
                },
            };
            self.active.store(restored);
            memory
        } else if is(&self.snapshot) {
            self.snapshot.store(None);
            MemoryRollback::SnapshotDropped
        } else {
            MemoryRollback::Untouched
        }
    }

    /// The snapshot may have been rolled back in storage since it was
    /// active (by another process). Restoring it then would serve a revoked
    /// patch, so its row is re-read first.
    fn snapshot_still_active(&self, snap: &CompiledPatch) -> Result<(), String> {
        match self.store.get_patch(&snap.patch_id) {
            Ok(Some(row)) if row.status == "ACTIVE" => Ok(()),
            Ok(Some(row)) => Err(format!("snapshot status is {}", row.status)),
            Ok(None) => Err("snapshot row no longer exists".to_string()),
            Err(e) => Err(format!("snapshot status unreadable: {e}")),
        }
    }
}

/// Re-verify a patch row against its file. Shared by apply and restore.
fn verify_row(row: &DurablePatchRow) -> Result<CompiledPatch, HotReloadError> {
    if row.status != "ACTIVE" {
        return Err(HotReloadError::PatchNotActive {
            patch_id: row.patch_id.clone(),
            status: row.status.clone(),
        });
    }
    if row.validation_status != "PASSED" {
        return Err(HotReloadError::PatchNotValidated {
            patch_id: row.patch_id.clone(),
            status: row.validation_status.clone(),
        });
    }
    let patch = validate_patch(Path::new(&row.patch_file_path)).map_err(|source| {
        HotReloadError::PatchFile {
            patch_id: row.patch_id.clone(),
            source,
        }
    })?;
    let mismatch =
        |field: &'static str, db: String, file: String| HotReloadError::IntegrityMismatch {
            patch_id: row.patch_id.clone(),
            field,
            db,
            file,
        };
    if patch.sha256 != row.sha256 {
        return Err(mismatch("sha256", row.sha256.clone(), patch.sha256));
    }
    if patch.patch_id != row.patch_id {
        return Err(mismatch("patch_id", row.patch_id.clone(), patch.patch_id));
    }
    if patch.samples_count as i64 != row.samples_count {
        return Err(mismatch(
            "samples_count",
            row.samples_count.to_string(),
            patch.samples_count.to_string(),
        ));
    }
    if patch.base_model_hash != row.base_model_hash {
        return Err(mismatch(
            "base_model_hash",
            row.base_model_hash.clone(),
            patch.base_model_hash,
        ));
    }
    Ok(patch)
}

fn now_ms() -> Result<i64, HotReloadError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .map_err(|e| HotReloadError::Clock(e.to_string()))
}

/// Trip thresholds for [`CanaryGuard`]. Rates are compared with `>`, over
/// the latest `window` samples, and only once `min_samples` exist.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanaryConfig {
    pub min_samples: usize,
    pub max_refusal_rate: f64,
    pub max_hard_stop_rate: f64,
    pub window: usize,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            min_samples: DEFAULT_CANARY_MIN_SAMPLES,
            max_refusal_rate: DEFAULT_CANARY_MAX_REFUSAL_RATE,
            max_hard_stop_rate: DEFAULT_CANARY_MAX_HARD_STOP_RATE,
            window: DEFAULT_CANARY_WINDOW,
        }
    }
}

impl CanaryConfig {
    pub fn validate(&self) -> Result<(), HotReloadError> {
        let fail = |m: String| Err(HotReloadError::Config(m));
        if self.min_samples == 0 {
            return fail("min_samples must be > 0".to_string());
        }
        if self.window < self.min_samples {
            return fail(format!(
                "window {} must be >= min_samples {}",
                self.window, self.min_samples
            ));
        }
        for (name, rate) in [
            ("max_refusal_rate", self.max_refusal_rate),
            ("max_hard_stop_rate", self.max_hard_stop_rate),
        ] {
            if !(0.0..=1.0).contains(&rate) {
                return fail(format!("{name} must be in [0, 1], got {rate}"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum CanaryDecision {
    Pass,
    Trip { reason: String },
}

pub struct CanaryGuard {
    store: Arc<DurableRefusalStore>,
    config: CanaryConfig,
}

impl CanaryGuard {
    pub fn new(
        store: Arc<DurableRefusalStore>,
        config: CanaryConfig,
    ) -> Result<Self, HotReloadError> {
        config.validate()?;
        Ok(Self { store, config })
    }

    pub fn config(&self) -> &CanaryConfig {
        &self.config
    }

    pub fn stats(&self, patch_id: &str) -> Result<CanaryStats, HotReloadError> {
        Ok(self
            .store
            .compute_canary_stats(patch_id, self.config.window)?)
    }

    /// Pure threshold check on already-computed stats.
    pub fn evaluate(&self, stats: &CanaryStats) -> CanaryDecision {
        if stats.total_samples < self.config.min_samples {
            return CanaryDecision::Pass;
        }
        let mut reasons = Vec::new();
        if stats.refusal_rate > self.config.max_refusal_rate {
            reasons.push(format!(
                "refusal_rate {:.4} > {:.4}",
                stats.refusal_rate, self.config.max_refusal_rate
            ));
        }
        if stats.hard_stop_rate > self.config.max_hard_stop_rate {
            reasons.push(format!(
                "hard_stop_rate {:.4} > {:.4}",
                stats.hard_stop_rate, self.config.max_hard_stop_rate
            ));
        }
        if reasons.is_empty() {
            CanaryDecision::Pass
        } else {
            CanaryDecision::Trip {
                reason: format!(
                    "canary trip on {} over {} samples: {}",
                    stats.patch_id,
                    stats.total_samples,
                    reasons.join("; ")
                ),
            }
        }
    }

    /// Append `metric`, then evaluate the latest window of its patch.
    pub fn record_and_evaluate(
        &self,
        metric: CanaryMetricInput,
    ) -> Result<CanaryDecision, HotReloadError> {
        self.store.record_canary_metric(&metric)?;
        Ok(self.evaluate(&self.stats(&metric.patch_id)?))
    }

    /// Re-read the window of `patch_id` and, if it trips, roll the patch
    /// back through `manager`. `Ok(true)` means this call rolled it back.
    pub fn check_and_auto_rollback(
        &self,
        patch_id: &str,
        manager: &HotReloadManager,
    ) -> Result<bool, HotReloadError> {
        match self.evaluate(&self.stats(patch_id)?) {
            CanaryDecision::Pass => Ok(false),
            CanaryDecision::Trip { reason } => {
                manager.rollback(patch_id, &reason)?;
                Ok(true)
            }
        }
    }
}
