//! In-process golden snapshots of mounted cognitive assets.
//!
//! Hosts call these engine operations from their trusted administration layer.
//! Snapshots are bounded to 16 entries per engine and do not survive a restart.
//! Rollback publishes saved assets as a new generation; audit history, request
//! state, graph/atlas contents and operator Nanocore registrations are not rewound.

use crate::cognitive::{CognitiveAssets, Rejection};
use crate::mount::{MountKey, MountRegistry, Reject, Version};
use crate::PolymorphicZeroEngine;
use gen_zero_storage::{GoldenSnapshot, StorageError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Serialize, Deserialize)]
struct SavedAssets {
    tenant: String,
    workspace: String,
    assets: Value,
}

fn storage_error(error: StorageError) -> Rejection {
    match error {
        StorageError::SnapshotNotFound(_) => Rejection::invalid("snapshot", error.to_string()),
        _ => Rejection::reject(Reject::InvalidCertificate, "snapshot", error.to_string()),
    }
}

impl PolymorphicZeroEngine {
    /// Capture one mounted generation, sealed by the engine's random MAC key.
    /// IDs are engine-wide; capturing an existing ID replaces that snapshot.
    /// Empty/unconfigured mounts are refused. The step is the captured version.
    pub fn capture_mount_snapshot(
        &self,
        snapshot_id: u64,
        key: &MountKey,
    ) -> Result<GoldenSnapshot, Rejection> {
        let mounted = self
            .mounts()
            .load(key)
            .map_err(|e| Rejection::reject(e, "snapshot", e.to_string()))?;
        let assets = CognitiveAssets::from_snapshot(&mounted)?;
        let saved = SavedAssets {
            tenant: key.tenant.clone(),
            workspace: key.workspace.clone(),
            assets: serde_json::from_slice(&assets.canonical_bytes())
                .map_err(|e| Rejection::invalid("snapshot", e.to_string()))?,
        };
        let bytes = serde_json::to_vec(&saved)
            .map_err(|e| Rejection::invalid("snapshot", e.to_string()))?;
        self.golden_snapshots
            .lock()
            .capture_snapshot(snapshot_id, mounted.version().0, &bytes)
            .cloned()
            .map_err(storage_error)
    }

    /// Restore a snapshot to its original tenant/workspace through validated CAS.
    /// `base_version` must be the caller's observed current mount version, not
    /// the saved version. A failed restore leaves the live mount untouched.
    /// Restoring already-mounted assets is refused as `Stalled`.
    pub fn rollback_mount_snapshot(
        &self,
        snapshot_id: u64,
        key: &MountKey,
        base_version: Version,
    ) -> Result<Value, Rejection> {
        let bytes = self
            .golden_snapshots
            .lock()
            .rollback_snapshot(snapshot_id)
            .map_err(storage_error)?;
        let saved: SavedAssets = serde_json::from_slice(&bytes)
            .map_err(|e| Rejection::invalid("snapshot", e.to_string()))?;
        if saved.tenant != key.tenant || saved.workspace != key.workspace {
            return Err(Rejection::invalid(
                "snapshot",
                "snapshot belongs to a different mount",
            ));
        }
        self.publish_assets(&json!({
            "tenant": key.tenant,
            "workspace": key.workspace,
            "base_version": base_version.0,
            "reason": format!("rollback golden snapshot {snapshot_id}"),
            "assets": saved.assets,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured() -> (PolymorphicZeroEngine, MountKey, Value) {
        let engine = PolymorphicZeroEngine::new().with_bridge(None);
        let key = MountKey::new("default", "default");
        let assets: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/cognitive_assets_linear2d.json"
        ))
        .unwrap();
        engine
            .publish_assets(&json!({"base_version": 1, "reason": "initial", "assets": assets}))
            .unwrap();
        (engine, key, assets)
    }

    #[test]
    fn golden_rollback_restores_assets_as_new_generation() {
        let (engine, key, mut assets) = configured();
        let before = engine.mounts().load(&key).unwrap();
        let snap = engine.capture_mount_snapshot(42, &key).unwrap();
        assert_eq!(snap.created_at_step, 2);
        assets["decision_temperature"] = json!(0.5);
        engine
            .publish_assets(&json!({"base_version": 2, "reason": "update", "assets": assets}))
            .unwrap();
        assert_ne!(
            engine.mounts().load(&key).unwrap().assets(),
            before.assets()
        );
        engine
            .rollback_mount_snapshot(42, &key, Version(3))
            .unwrap();
        let after = engine.mounts().load(&key).unwrap();
        assert_eq!(after.version(), Version(4));
        assert_eq!(after.assets(), before.assets());
        assert_eq!(after.epochs().model, before.epochs().model);
        assert_eq!(after.epochs().geometry, before.epochs().geometry);
        assert_eq!(after.epochs().policy, before.epochs().policy);
        assert!(after.verify_digest());
    }

    #[test]
    fn rollback_refuses_stale_version_wrong_mount_and_missing_id() {
        let (engine, key, _) = configured();
        engine.capture_mount_snapshot(1, &key).unwrap();
        let before = engine.mounts().load(&key).unwrap();
        assert_eq!(
            engine
                .rollback_mount_snapshot(1, &key, Version(0))
                .unwrap_err()
                .code,
            "CasConflict"
        );
        assert!(engine
            .rollback_mount_snapshot(1, &MountKey::new("other", "default"), Version(1))
            .is_err());
        assert!(engine
            .rollback_mount_snapshot(999, &key, Version(1))
            .is_err());
        assert_eq!(
            engine.mounts().load(&key).unwrap().digest(),
            before.digest()
        );
    }

    #[test]
    fn snapshots_are_bounded_and_unconfigured_mounts_are_refused() {
        let (engine, key, _) = configured();
        assert!(PolymorphicZeroEngine::new()
            .with_bridge(None)
            .capture_mount_snapshot(1, &key)
            .is_err());
        for id in 0..17 {
            engine.capture_mount_snapshot(id, &key).unwrap();
        }
        assert_eq!(engine.golden_snapshots.lock().len(), 16);
        assert_eq!(
            engine
                .rollback_mount_snapshot(16, &key, Version(2))
                .unwrap_err()
                .code,
            "Stalled"
        );
    }
}
