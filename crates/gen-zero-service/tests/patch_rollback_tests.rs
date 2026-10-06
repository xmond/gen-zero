//! Integration tests for Phase 4 (`gen_zero_service::hot_reload`): verified
//! hot swap of the deployed patch, canary statistics and trips, and the
//! storage-plus-memory rollback, including the qa_gate wiring in the engine.
//!
//! Patches are real: arbitrated traces are compiled by `PatchBuilder`, so
//! every apply below re-verifies a real file against its real row.

use gen_zero_service::patch_builder::{CompiledPatch, PatchBuildConfig, PatchBuilder};
use gen_zero_service::{
    CanaryConfig, CanaryDecision, CanaryGuard, HotReloadError, HotReloadManager, MemoryRollback,
    PolymorphicZeroEngine, ZeroEngineConfig,
};
use gen_zero_storage::{
    ArbitrationResult, ArbitrationStatus, CanaryMetricInput, DurableRefusalStore,
    RefusalTraceInput, StorageError,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::TempDir;

const CONTEXT: &str =
    "Mount Everest is the tallest mountain on Earth, rising 8,849 meters above sea level.";

struct Fixture {
    _dir: TempDir,
    db_path: PathBuf,
    output_dir: PathBuf,
    store: Arc<DurableRefusalStore>,
}

fn fixture() -> Fixture {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("durable_refusal.sqlite3");
    let output_dir = dir.path().join("patches");
    let store = Arc::new(DurableRefusalStore::new(&db_path).unwrap());
    Fixture {
        _dir: dir,
        db_path,
        output_dir,
        store,
    }
}

fn record(store: &DurableRefusalStore, trace_id: &str, created_at_ms: i64) {
    store
        .record_refusal(&RefusalTraceInput {
            trace_id: trace_id.to_string(),
            created_at_ms,
            context: CONTEXT.to_string(),
            question: format!("How tall is Mount Everest? ({trace_id})"),
            candidate: "I cannot answer that from the given context.".to_string(),
            best_span_score: 0.12,
            null_score: 0.95,
            score_diff: -0.83,
            verifier_output: None,
        })
        .unwrap();
}

fn arbitrate(store: &DurableRefusalStore, trace_id: &str, at_ms: i64) {
    store
        .complete_arbitration(
            trace_id,
            &ArbitrationResult {
                is_answerable: true,
                gold_answer: Some("8,849 meters".to_string()),
                evidence_span: Some("rising 8,849 meters above sea level".to_string()),
                contradiction_fact: None,
                arbitrator_model: "test-arbiter".to_string(),
                arbitration_raw: "{}".to_string(),
                arbitrated_at_ms: at_ms,
            },
        )
        .unwrap();
}

/// Arbitrate `n` fresh traces tagged `tag` and compile them into one real
/// patch file plus its `durable_patches` row.
fn build_patch(f: &Fixture, tag: &str, n: usize) -> CompiledPatch {
    for i in 0..n {
        let id = format!("{tag}-{i}");
        record(&f.store, &id, 1_000 + i as i64);
        arbitrate(&f.store, &id, 2_000 + i as i64);
    }
    let config = PatchBuildConfig::new(&f.db_path, &f.output_dir);
    let patch = PatchBuilder::new(config, Arc::clone(&f.store))
        .run_once()
        .unwrap()
        .expect("arbitrated traces must compile into a patch");
    assert_eq!(patch.samples_count, n);
    patch
}

fn active_id(m: &HotReloadManager) -> Option<String> {
    m.active().map(|p| p.patch_id.clone())
}

fn snapshot_id(m: &HotReloadManager) -> Option<String> {
    m.snapshot().map(|p| p.patch_id.clone())
}

fn metric(patch_id: &str, refused: bool, hard_stop: bool, latency_us: i64) -> CanaryMetricInput {
    CanaryMetricInput {
        patch_id: patch_id.to_string(),
        recorded_at_ms: 10_000,
        is_fast_pass: !refused,
        is_refused: refused,
        latency_us,
        is_hard_stop: hard_stop,
    }
}

#[test]
fn apply_patch_verifies_sha256_and_swaps_the_active_pointer() {
    let f = fixture();
    let a = build_patch(&f, "a", 2);
    let b = build_patch(&f, "b", 3);
    let manager = HotReloadManager::new(Arc::clone(&f.store));
    assert!(manager.active().is_none());

    let applied = manager.apply_patch(&a.patch_id).unwrap();
    assert_eq!(*applied, a, "the loaded patch must equal the compiled one");
    assert_eq!(active_id(&manager), Some(a.patch_id.clone()));
    assert_eq!(snapshot_id(&manager), None);
    let row_a = f.store.get_patch(&a.patch_id).unwrap().unwrap();
    assert!(row_a.deployed_at_ms.is_some(), "{row_a:?}");
    assert_eq!(row_a.status, "ACTIVE");

    manager.apply_patch(&b.patch_id).unwrap();
    assert_eq!(active_id(&manager), Some(b.patch_id.clone()));
    assert_eq!(
        snapshot_id(&manager),
        Some(a.patch_id.clone()),
        "the replaced patch becomes the golden snapshot"
    );

    let again = manager.apply_patch(&b.patch_id).unwrap_err();
    assert!(
        matches!(again, HotReloadError::AlreadyActive(ref id) if *id == b.patch_id),
        "{again:?}"
    );
    assert_eq!(active_id(&manager), Some(b.patch_id.clone()));
    assert_eq!(snapshot_id(&manager), Some(a.patch_id.clone()));

    let missing = manager.apply_patch("patch_does_not_exist").unwrap_err();
    assert!(
        matches!(missing, HotReloadError::PatchNotFound(_)),
        "{missing:?}"
    );
}

#[test]
fn tampered_file_is_rejected_and_memory_is_unchanged() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    let b = build_patch(&f, "b", 2);
    let manager = HotReloadManager::new(Arc::clone(&f.store));
    manager.apply_patch(&a.patch_id).unwrap();

    // Change one training sample but keep the header sha256: the recomputed
    // body hash no longer matches.
    let path = Path::new(&b.patch_file_path);
    let original = std::fs::read(path).unwrap();
    let mut doc: serde_json::Value = serde_json::from_slice(&original).unwrap();
    doc["samples"][0]["gold_answer"] = json!("9,000 meters");
    std::fs::write(path, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();

    let err = manager.apply_patch(&b.patch_id).unwrap_err();
    assert!(
        matches!(err, HotReloadError::PatchFile { ref patch_id, .. } if *patch_id == b.patch_id),
        "{err:?}"
    );
    assert!(err.to_string().contains("sha256 mismatch"), "{err}");
    assert_eq!(active_id(&manager), Some(a.patch_id.clone()));
    assert_eq!(snapshot_id(&manager), None);
    assert_eq!(
        f.store
            .get_patch(&b.patch_id)
            .unwrap()
            .unwrap()
            .deployed_at_ms,
        None,
        "a rejected patch must not be marked deployed"
    );

    // A missing file is rejected the same way.
    std::fs::remove_file(path).unwrap();
    let err = manager.apply_patch(&b.patch_id).unwrap_err();
    assert!(matches!(err, HotReloadError::PatchFile { .. }), "{err:?}");
    assert_eq!(active_id(&manager), Some(a.patch_id.clone()));
}

#[test]
fn database_hash_disagreeing_with_an_intact_file_is_rejected() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    let conn = rusqlite::Connection::open(&f.db_path).unwrap();
    conn.execute(
        "UPDATE durable_patches SET sha256 = ?1 WHERE patch_id = ?2",
        rusqlite::params!["0".repeat(64), a.patch_id],
    )
    .unwrap();
    drop(conn);

    let manager = HotReloadManager::new(Arc::clone(&f.store));
    let err = manager.apply_patch(&a.patch_id).unwrap_err();
    assert!(
        matches!(
            err,
            HotReloadError::IntegrityMismatch {
                field: "sha256",
                ..
            }
        ),
        "{err:?}"
    );
    assert!(manager.active().is_none());
    assert_eq!(
        f.store
            .get_patch(&a.patch_id)
            .unwrap()
            .unwrap()
            .deployed_at_ms,
        None
    );
}

#[test]
fn rolled_back_patch_cannot_be_applied() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    f.store.rollback_patch(&a.patch_id, "bad canary").unwrap();
    let manager = HotReloadManager::new(Arc::clone(&f.store));
    let err = manager.apply_patch(&a.patch_id).unwrap_err();
    assert!(
        matches!(err, HotReloadError::PatchNotActive { ref status, .. } if status == "ROLLED_BACK"),
        "{err:?}"
    );
    assert!(manager.active().is_none());
}

#[test]
fn canary_stats_are_computed_over_the_latest_window() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    // Oldest first: 2 clean, then 3 refused (one of them a hard stop).
    let rows = [
        (false, false, 100),
        (false, false, 200),
        (true, false, 300),
        (true, true, 400),
        (true, false, 500),
    ];
    let mut ids = Vec::new();
    for (refused, hard, lat) in rows {
        ids.push(
            f.store
                .record_canary_metric(&metric(&a.patch_id, refused, hard, lat))
                .unwrap(),
        );
    }
    assert!(ids.windows(2).all(|w| w[1] > w[0]), "{ids:?}");

    let all = f.store.compute_canary_stats(&a.patch_id, 100).unwrap();
    assert_eq!(all.total_samples, 5);
    assert_eq!(all.refused_count, 3);
    assert_eq!(all.fast_pass_count, 2);
    assert_eq!(all.hard_stop_count, 1);
    assert!((all.refusal_rate - 0.6).abs() < 1e-12, "{all:?}");
    assert!((all.hard_stop_rate - 0.2).abs() < 1e-12, "{all:?}");
    assert!((all.fast_pass_rate - 0.4).abs() < 1e-12, "{all:?}");
    assert!((all.avg_latency_us - 300.0).abs() < 1e-9, "{all:?}");

    // Window of 3 = the three newest rows only.
    let recent = f.store.compute_canary_stats(&a.patch_id, 3).unwrap();
    assert_eq!(recent.total_samples, 3);
    assert_eq!(recent.refused_count, 3);
    assert!((recent.avg_latency_us - 400.0).abs() < 1e-9, "{recent:?}");

    let fetched = f.store.fetch_canary_metrics(&a.patch_id, 2).unwrap();
    assert_eq!(fetched.len(), 2);
    assert_eq!(fetched[0].sample_id, ids[4], "newest first");
    assert_eq!(fetched[0].latency_us, 500);
    assert_eq!(fetched[1].latency_us, 400);
    assert!(fetched[1].is_hard_stop);

    let empty = f
        .store
        .compute_canary_stats("patch_without_rows", 10)
        .unwrap();
    assert_eq!(empty.total_samples, 0);
    assert_eq!(empty.refusal_rate, 0.0);
}

#[test]
fn canary_guard_trips_on_refusal_surge_but_not_below_min_samples() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    let guard = CanaryGuard::new(Arc::clone(&f.store), CanaryConfig::default()).unwrap();

    // 19 samples, all refused: below min_samples (20), so no verdict yet.
    for _ in 0..19 {
        let d = guard
            .record_and_evaluate(metric(&a.patch_id, true, false, 50))
            .unwrap();
        assert_eq!(d, CanaryDecision::Pass);
    }
    // The 20th sample reaches min_samples; refusal_rate 1.0 > 0.35.
    let d = guard
        .record_and_evaluate(metric(&a.patch_id, true, false, 50))
        .unwrap();
    match d {
        CanaryDecision::Trip { reason } => {
            assert!(reason.contains("refusal_rate"), "{reason}");
            assert!(!reason.contains("hard_stop_rate"), "{reason}");
        }
        other => panic!("expected trip, got {other:?}"),
    }

    // Exactly at the limit does not trip: 7/20 = 0.35 is not > 0.35.
    let b = build_patch(&f, "b", 1);
    for i in 0..20 {
        let d = guard
            .record_and_evaluate(metric(&b.patch_id, i < 7, false, 50))
            .unwrap();
        assert_eq!(d, CanaryDecision::Pass, "sample {i}");
    }
    // One more refusal pushes it to 8/21 = 0.381.
    let d = guard
        .record_and_evaluate(metric(&b.patch_id, true, false, 50))
        .unwrap();
    assert!(matches!(d, CanaryDecision::Trip { .. }), "{d:?}");
}

#[test]
fn canary_guard_trips_on_hard_stops() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    let guard = CanaryGuard::new(Arc::clone(&f.store), CanaryConfig::default()).unwrap();
    // 1 hard stop in 20 = 0.05: at the limit, passes. Hard stops count as
    // refusals too, but 1/20 is far below 0.35.
    for i in 0..20 {
        let hard = i == 0;
        let d = guard
            .record_and_evaluate(metric(&a.patch_id, hard, hard, 50))
            .unwrap();
        assert_eq!(d, CanaryDecision::Pass, "sample {i}");
    }
    let d = guard
        .record_and_evaluate(metric(&a.patch_id, true, true, 50))
        .unwrap();
    match d {
        CanaryDecision::Trip { reason } => {
            assert!(reason.contains("hard_stop_rate"), "{reason}");
            assert!(!reason.contains("refusal_rate"), "{reason}");
        }
        other => panic!("expected trip, got {other:?}"),
    }
}

#[test]
fn canary_config_and_metric_inputs_fail_closed() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    for bad in [
        CanaryConfig {
            min_samples: 0,
            ..Default::default()
        },
        CanaryConfig {
            window: 5,
            ..Default::default()
        },
        CanaryConfig {
            max_refusal_rate: f64::NAN,
            ..Default::default()
        },
        CanaryConfig {
            max_hard_stop_rate: 1.5,
            ..Default::default()
        },
    ] {
        let err = CanaryGuard::new(Arc::clone(&f.store), bad.clone()).err();
        assert!(
            matches!(err, Some(HotReloadError::Config(_))),
            "{bad:?} -> {err:?}"
        );
    }
    let unknown = f
        .store
        .record_canary_metric(&metric("patch_unknown", false, false, 1))
        .unwrap_err();
    assert!(
        matches!(unknown, StorageError::DurablePatchNotFound(_)),
        "{unknown:?}"
    );
    let negative = f
        .store
        .record_canary_metric(&metric(&a.patch_id, false, false, -1))
        .unwrap_err();
    assert!(
        matches!(negative, StorageError::InvalidCanaryMetric(_)),
        "{negative:?}"
    );
    assert!(matches!(
        f.store.compute_canary_stats(&a.patch_id, 0),
        Err(StorageError::InvalidCanaryMetric(_))
    ));
}

#[test]
fn rollback_restores_snapshot_and_permanently_isolates_revoked_traces() {
    let f = fixture();
    let a = build_patch(&f, "a", 2);
    let b = build_patch(&f, "b", 3);
    // Unrelated work still in flight: one pending, one arbitrated-unconsumed.
    record(&f.store, "free-pending", 5_000);
    record(&f.store, "free-arbitrated", 5_001);
    arbitrate(&f.store, "free-arbitrated", 6_000);

    let manager = HotReloadManager::new(Arc::clone(&f.store));
    manager.apply_patch(&a.patch_id).unwrap();
    manager.apply_patch(&b.patch_id).unwrap();
    assert_eq!(active_id(&manager), Some(b.patch_id.clone()));

    let report = manager
        .rollback(&b.patch_id, "canary: refusal surge")
        .unwrap();
    println!(
        "rollback of {} took {} us (storage tx + pointer swap)",
        b.patch_id, report.elapsed_us
    );
    assert_eq!(
        report.memory,
        MemoryRollback::RestoredSnapshot {
            patch_id: a.patch_id.clone()
        }
    );
    // Generous bound for a shared box; the measured number is printed above.
    assert!(report.elapsed_us < 1_000_000, "{report:?}");

    // Memory: back to the snapshot, snapshot slot emptied.
    assert_eq!(active_id(&manager), Some(a.patch_id.clone()));
    assert_eq!(snapshot_id(&manager), None);

    // Storage: patch row.
    let row = f.store.get_patch(&b.patch_id).unwrap().unwrap();
    assert_eq!(row.status, "ROLLED_BACK");
    assert_eq!(
        row.rollback_reason.as_deref(),
        Some("canary: refusal surge")
    );
    assert!(row.rolled_back_at_ms.is_some());
    assert_eq!(
        f.store.get_patch(&a.patch_id).unwrap().unwrap().status,
        "ACTIVE"
    );

    // Storage: every trace of B revoked, A's untouched.
    for i in 0..3 {
        let t = f.store.get_trace(&format!("b-{i}")).unwrap().unwrap();
        assert_eq!(t.arbitration_status, ArbitrationStatus::Revoked, "{t:?}");
        assert_eq!(t.applied_patch_id.as_deref(), Some(b.patch_id.as_str()));
    }
    for i in 0..2 {
        let t = f.store.get_trace(&format!("a-{i}")).unwrap().unwrap();
        assert_eq!(t.arbitration_status, ArbitrationStatus::Arbitrated);
    }

    // Isolation: neither extraction path can ever return a revoked trace.
    let unconsumed = f.store.fetch_unconsumed_arbitrated(1_000).unwrap();
    let pending = f.store.fetch_pending_arbitration(1_000).unwrap();
    let ids = |v: &[gen_zero_storage::DurableRefusalTrace]| {
        v.iter().map(|t| t.trace_id.clone()).collect::<Vec<_>>()
    };
    assert_eq!(
        ids(unconsumed.as_slice()),
        vec!["free-arbitrated".to_string()]
    );
    assert_eq!(ids(pending.as_slice()), vec!["free-pending".to_string()]);
    // And a revoked trace cannot be pushed back into either queue.
    assert!(matches!(
        f.store.fail_arbitration("b-0", "retry"),
        Err(StorageError::InvalidArbitrationState { .. })
    ));
    assert!(matches!(
        f.store.mark_consumed(&["b-0"], "patch_other"),
        Err(StorageError::InvalidArbitrationState { .. })
    ));
    // The next patch build only sees the free trace.
    let next = PatchBuilder::new(
        PatchBuildConfig::new(&f.db_path, &f.output_dir),
        Arc::clone(&f.store),
    )
    .run_once()
    .unwrap()
    .unwrap();
    let next_ids: Vec<_> = next.samples.iter().map(|s| s.trace_id.as_str()).collect();
    assert_eq!(next_ids, vec!["free-arbitrated"]);

    // No canary rows can be written for a rolled-back patch.
    assert!(matches!(
        f.store
            .record_canary_metric(&metric(&b.patch_id, false, false, 1)),
        Err(StorageError::PatchNotActive { .. })
    ));

    // A second rollback is refused and changes nothing.
    let err = manager.rollback(&b.patch_id, "again").unwrap_err();
    assert!(
        matches!(err, HotReloadError::Storage(StorageError::PatchAlreadyRolledBack(ref id)) if *id == b.patch_id),
        "{err:?}"
    );
    assert_eq!(active_id(&manager), Some(a.patch_id.clone()));
    assert_eq!(
        f.store
            .get_patch(&b.patch_id)
            .unwrap()
            .unwrap()
            .rollback_reason
            .as_deref(),
        Some("canary: refusal surge"),
        "the first reason must survive a refused second rollback"
    );
}

#[test]
fn rollback_memory_follows_patch_identity() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    let b = build_patch(&f, "b", 1);
    let c = build_patch(&f, "c", 1);
    let manager = HotReloadManager::new(Arc::clone(&f.store));
    manager.apply_patch(&a.patch_id).unwrap();
    manager.apply_patch(&b.patch_id).unwrap();

    // Rolling back a patch that is neither active nor snapshot: storage only.
    let r = manager.rollback(&c.patch_id, "never deployed").unwrap();
    assert_eq!(r.memory, MemoryRollback::Untouched);
    assert_eq!(active_id(&manager), Some(b.patch_id.clone()));
    assert_eq!(snapshot_id(&manager), Some(a.patch_id.clone()));

    // Rolling back the snapshot drops it, active stays.
    let r = manager.rollback(&a.patch_id, "old patch bad").unwrap();
    assert_eq!(r.memory, MemoryRollback::SnapshotDropped);
    assert_eq!(active_id(&manager), Some(b.patch_id.clone()));
    assert_eq!(snapshot_id(&manager), None);

    // Rolling back the active with no snapshot clears it.
    let r = manager.rollback(&b.patch_id, "bad").unwrap();
    assert_eq!(
        r.memory,
        MemoryRollback::Cleared {
            snapshot_rejected: None
        }
    );
    assert!(manager.active().is_none());
}

#[test]
fn snapshot_rolled_back_elsewhere_is_not_restored() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    let b = build_patch(&f, "b", 1);
    let manager = HotReloadManager::new(Arc::clone(&f.store));
    manager.apply_patch(&a.patch_id).unwrap();
    manager.apply_patch(&b.patch_id).unwrap();
    // Another process rolls back A directly in storage.
    f.store.rollback_patch(&a.patch_id, "external").unwrap();

    let r = manager.rollback(&b.patch_id, "bad").unwrap();
    match r.memory {
        MemoryRollback::Cleared {
            snapshot_rejected: Some(reason),
        } => assert!(reason.contains("ROLLED_BACK"), "{reason}"),
        other => panic!("expected cleared with rejected snapshot, got {other:?}"),
    }
    assert!(
        manager.active().is_none(),
        "a revoked snapshot must never be served"
    );
}

#[test]
fn check_and_auto_rollback_rolls_back_only_on_trip() {
    let f = fixture();
    let a = build_patch(&f, "a", 2);
    let manager = HotReloadManager::new(Arc::clone(&f.store));
    manager.apply_patch(&a.patch_id).unwrap();
    let guard = CanaryGuard::new(Arc::clone(&f.store), CanaryConfig::default()).unwrap();

    for _ in 0..20 {
        guard
            .record_and_evaluate(metric(&a.patch_id, false, false, 10))
            .unwrap();
    }
    assert!(!guard
        .check_and_auto_rollback(&a.patch_id, &manager)
        .unwrap());
    assert_eq!(active_id(&manager), Some(a.patch_id.clone()));

    // 20 refusals: the latest 40 now hold 20/40 = 0.5 > 0.35.
    for _ in 0..20 {
        f.store
            .record_canary_metric(&metric(&a.patch_id, true, false, 10))
            .unwrap();
    }
    assert!(guard
        .check_and_auto_rollback(&a.patch_id, &manager)
        .unwrap());
    assert!(manager.active().is_none());
    let row = f.store.get_patch(&a.patch_id).unwrap().unwrap();
    assert_eq!(row.status, "ROLLED_BACK");
    let reason = row.rollback_reason.unwrap();
    assert!(reason.contains("refusal_rate 0.5000 > 0.3500"), "{reason}");
    for i in 0..2 {
        let t = f.store.get_trace(&format!("a-{i}")).unwrap().unwrap();
        assert_eq!(t.arbitration_status, ArbitrationStatus::Revoked);
    }
}

#[test]
fn restore_rebuilds_active_and_snapshot_and_fails_closed_on_tamper() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    let b = build_patch(&f, "b", 1);
    {
        let manager = HotReloadManager::new(Arc::clone(&f.store));
        manager.apply_patch(&a.patch_id).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        manager.apply_patch(&b.patch_id).unwrap();
    }
    let restored = HotReloadManager::restore(Arc::clone(&f.store)).unwrap();
    assert_eq!(active_id(&restored), Some(b.patch_id.clone()));
    assert_eq!(snapshot_id(&restored), Some(a.patch_id.clone()));

    // A snapshot that no longer verifies is dropped, startup continues.
    let a_bytes = std::fs::read(&a.patch_file_path).unwrap();
    std::fs::remove_file(&a.patch_file_path).unwrap();
    let no_snapshot = HotReloadManager::restore(Arc::clone(&f.store)).unwrap();
    assert_eq!(active_id(&no_snapshot), Some(b.patch_id.clone()));
    assert_eq!(snapshot_id(&no_snapshot), None);
    std::fs::write(&a.patch_file_path, a_bytes).unwrap();

    // An active patch that no longer verifies stops the restore.
    std::fs::write(&b.patch_file_path, b"{}").unwrap();
    let err = HotReloadManager::restore(Arc::clone(&f.store)).err();
    assert!(
        matches!(err, Some(HotReloadError::PatchFile { ref patch_id, .. }) if *patch_id == b.patch_id),
        "{err:?}"
    );
}

/// The engine's qa_gate path: a deployed patch is restored at startup,
/// every decision writes a canary row, and a refusal surge rolls the patch
/// back from inside the request path.
#[tokio::test]
async fn engine_qa_gate_records_canary_and_auto_rolls_back() {
    let f = fixture();
    let a = build_patch(&f, "a", 2);
    HotReloadManager::new(Arc::clone(&f.store))
        .apply_patch(&a.patch_id)
        .unwrap();

    let canary = CanaryConfig {
        min_samples: 5,
        window: 10,
        ..Default::default()
    };
    let engine = PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default()
            .with_refusal_db_path(&f.db_path)
            .with_canary_config(canary),
    )
    .unwrap();
    let hot = engine
        .hot_reload()
        .expect("hot reload wired with refusal db");
    assert_eq!(
        active_id(&hot),
        Some(a.patch_id.clone()),
        "restored at startup"
    );

    let answerable = json!({"action": "qa_gate", "context": "Paris is in France.",
        "question": "Where is Paris?", "candidate": "France",
        "best_span_score": 3.0, "null_score": 0.0});
    let out = engine.execute(&answerable).await.unwrap();
    assert!(!out.is_error, "{out:?}");
    assert_eq!(out.meta["canary"]["patch_id"], a.patch_id.as_str());
    assert_eq!(out.meta["canary"]["recorded"], true);
    assert_eq!(out.meta["canary"]["auto_rolled_back"], false);

    // A caller error (400) is not a canary sample.
    let bad = json!({"action": "qa_gate", "context": "c", "question": "q"});
    assert!(engine.execute(&bad).await.unwrap().is_error);
    assert_eq!(
        f.store
            .compute_canary_stats(&a.patch_id, 100)
            .unwrap()
            .total_samples,
        1
    );

    // Refusals until the window holds 5 samples, 4 refused: 0.8 > 0.35.
    let refuse = json!({"action": "qa_gate", "context": "Paris is in France.",
        "question": "How tall is K2?", "candidate": "K2",
        "best_span_score": 0.0, "null_score": 3.0});
    for i in 0..3 {
        let out = engine.execute(&refuse).await.unwrap();
        assert!(!out.is_error, "{out:?}");
        assert_eq!(out.meta["evidence"]["is_answerable"], false);
        assert_eq!(out.meta["canary"]["auto_rolled_back"], false, "sample {i}");
    }
    let out = engine.execute(&refuse).await.unwrap();
    assert_eq!(out.meta["canary"]["auto_rolled_back"], true, "{out:?}");

    assert!(
        hot.active().is_none(),
        "memory rolled back in the request path"
    );
    let row = f.store.get_patch(&a.patch_id).unwrap().unwrap();
    assert_eq!(row.status, "ROLLED_BACK");
    assert!(row.rollback_reason.unwrap().contains("refusal_rate"));
    let stats = f.store.compute_canary_stats(&a.patch_id, 100).unwrap();
    assert_eq!(stats.total_samples, 5);
    assert_eq!(stats.refused_count, 4);
    assert!(stats.avg_latency_us > 0.0, "{stats:?}");

    // With no active patch, decisions are served and record nothing.
    let out = engine.execute(&answerable).await.unwrap();
    assert!(!out.is_error);
    assert!(out.meta.get("canary").is_none(), "{out:?}");
}

#[tokio::test]
async fn engine_records_gate_failures_as_hard_stops() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    HotReloadManager::new(Arc::clone(&f.store))
        .apply_patch(&a.patch_id)
        .unwrap();
    let engine = PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default().with_refusal_db_path(&f.db_path),
    )
    .unwrap();
    // In the ambiguity band with no verifier configured, the gateway fails
    // closed (503): that is a hard stop.
    let ambiguous = json!({"action": "qa_gate", "context": "Paris is in France",
        "question": "Where?", "candidate": "Paris",
        "best_span_score": 1.0, "null_score": 1.0});
    let out = engine.execute(&ambiguous).await.unwrap();
    assert!(out.is_error);
    let rows = f.store.fetch_canary_metrics(&a.patch_id, 10).unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(rows[0].is_hard_stop && rows[0].is_refused, "{rows:?}");
}

#[tokio::test]
async fn engine_drops_a_patch_rolled_back_by_another_process() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    HotReloadManager::new(Arc::clone(&f.store))
        .apply_patch(&a.patch_id)
        .unwrap();
    let engine = PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default().with_refusal_db_path(&f.db_path),
    )
    .unwrap();
    let hot = engine.hot_reload().unwrap();
    assert_eq!(active_id(&hot), Some(a.patch_id.clone()));

    // The operator rolls back from a separate store handle (as the CLI does).
    let other = Arc::new(DurableRefusalStore::new(&f.db_path).unwrap());
    HotReloadManager::new(other)
        .rollback(&a.patch_id, "operator")
        .unwrap();
    assert_eq!(
        active_id(&hot),
        Some(a.patch_id.clone()),
        "no watcher: memory still holds it until the next canary write"
    );

    let request = json!({"action": "qa_gate", "context": "Paris is in France.",
        "question": "Where is Paris?", "candidate": "France",
        "best_span_score": 3.0, "null_score": 0.0});
    let out = engine.execute(&request).await.unwrap();
    assert!(!out.is_error, "{out:?}");
    assert_eq!(
        out.meta["canary"]["patch_no_longer_active"], true,
        "{out:?}"
    );
    assert!(hot.active().is_none());
}

#[test]
fn engine_startup_fails_closed_on_a_tampered_deployed_patch() {
    let f = fixture();
    let a = build_patch(&f, "a", 1);
    HotReloadManager::new(Arc::clone(&f.store))
        .apply_patch(&a.patch_id)
        .unwrap();
    std::fs::remove_file(&a.patch_file_path).unwrap();
    let err = PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default().with_refusal_db_path(&f.db_path),
    )
    .err()
    .expect("a deployed patch whose file is gone must stop startup");
    assert!(err.to_string().contains("restore deployed patch"), "{err}");
}
