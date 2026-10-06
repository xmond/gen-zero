//! Integration tests for `DurableRefusalStore`, exercised only through its
//! public API.

use gen_zero_storage::{
    ArbitrationResult, ArbitrationStatus, CanaryMetricInput, CanaryMetricRow, CanaryStats,
    DurablePatchRecord, DurableRefusalStore, RefusalTraceInput, StorageError,
};
use tempfile::TempDir;

fn open_temp_store() -> (TempDir, DurableRefusalStore) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("durable_refusal.sqlite3");
    let store = DurableRefusalStore::new(&path).unwrap();
    (dir, store)
}

fn sample_trace(trace_id: &str, created_at_ms: i64) -> RefusalTraceInput {
    RefusalTraceInput {
        trace_id: trace_id.to_string(),
        created_at_ms,
        context: "Mount Everest is the tallest mountain on Earth.".to_string(),
        question: "How tall is K2?".to_string(),
        candidate: "I cannot answer that from the given context.".to_string(),
        best_span_score: 0.12,
        null_score: 0.95,
        score_diff: -0.83,
        verifier_output: Some("tri_teacher: low confidence".to_string()),
    }
}

fn sample_arbitration(arbitrated_at_ms: i64) -> ArbitrationResult {
    ArbitrationResult {
        is_answerable: false,
        gold_answer: None,
        evidence_span: None,
        contradiction_fact: Some("context never mentions K2".to_string()),
        arbitrator_model: "gpt-arbiter-1".to_string(),
        arbitration_raw: "{\"verdict\": \"unanswerable\"}".to_string(),
        arbitrated_at_ms,
    }
}

fn sample_patch(patch_id: &str, created_at_ms: i64) -> DurablePatchRecord {
    DurablePatchRecord {
        patch_id: patch_id.to_string(),
        created_at_ms,
        base_model_hash: "sha256:basehash".to_string(),
        patch_file_path: "/patches/p1.safetensors".to_string(),
        sha256: "sha256:patchhash".to_string(),
        samples_count: 1,
        validation_status: "PASSED".to_string(),
        status: "CANDIDATE".to_string(),
        deployed_at_ms: None,
    }
}

#[test]
fn record_and_fetch_pending_round_trips_all_fields() {
    let (_dir, store) = open_temp_store();
    let input = sample_trace("t1", 1_000);
    store.record_refusal(&input).unwrap();

    let pending = store.fetch_pending_arbitration(10).unwrap();
    assert_eq!(pending.len(), 1);
    let row = &pending[0];
    assert_eq!(row.trace_id, "t1");
    assert_eq!(row.created_at_ms, 1_000);
    assert_eq!(row.context, input.context);
    assert_eq!(row.question, input.question);
    assert_eq!(row.candidate, input.candidate);
    assert!((row.best_span_score - input.best_span_score).abs() < 1e-6);
    assert!((row.null_score - input.null_score).abs() < 1e-6);
    assert!((row.score_diff - input.score_diff).abs() < 1e-6);
    assert_eq!(row.verifier_output, input.verifier_output);
    assert_eq!(row.arbitration_status, ArbitrationStatus::Pending);
    assert_eq!(row.is_answerable, None);
    assert_eq!(row.gold_answer, None);
    assert_eq!(row.retry_count, 0);
    assert_eq!(row.last_error, None);
    assert!(!row.is_consumed);
    assert_eq!(row.applied_patch_id, None);
}

#[test]
fn complete_arbitration_moves_trace_to_unconsumed_arbitrated() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let result = sample_arbitration(2_000);
    store.complete_arbitration("t1", &result).unwrap();

    assert_eq!(store.fetch_pending_arbitration(10).unwrap().len(), 0);

    let unconsumed = store.fetch_unconsumed_arbitrated(10).unwrap();
    assert_eq!(unconsumed.len(), 1);
    let row = &unconsumed[0];
    assert_eq!(row.arbitration_status, ArbitrationStatus::Arbitrated);
    assert_eq!(row.is_answerable, Some(false));
    assert_eq!(row.contradiction_fact, result.contradiction_fact);
    assert_eq!(row.arbitrator_model, Some(result.arbitrator_model.clone()));
    assert_eq!(row.arbitration_raw, Some(result.arbitration_raw.clone()));
    assert_eq!(row.arbitrated_at_ms, Some(2_000));
    assert!(!row.is_consumed);
}

#[test]
fn fail_arbitration_increments_retry_and_returns_to_pending() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    store.fail_arbitration("t1", "arbiter timed out").unwrap();

    let pending = store.fetch_pending_arbitration(10).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].retry_count, 1);
    assert_eq!(pending[0].last_error, Some("arbiter timed out".to_string()));
    assert_eq!(pending[0].arbitration_status, ArbitrationStatus::Pending);

    // A second failure keeps incrementing and the row is still pending.
    store
        .fail_arbitration("t1", "arbiter errored again")
        .unwrap();
    let pending = store.fetch_pending_arbitration(10).unwrap();
    assert_eq!(pending[0].retry_count, 2);
    assert_eq!(
        pending[0].last_error,
        Some("arbiter errored again".to_string())
    );
}

#[test]
fn complete_and_fail_arbitration_reject_unknown_trace_id() {
    let (_dir, store) = open_temp_store();

    let err = store
        .complete_arbitration("missing", &sample_arbitration(2_000))
        .unwrap_err();
    assert!(matches!(err, StorageError::DurableTraceNotFound(_)));

    let err = store.fail_arbitration("missing", "nope").unwrap_err();
    assert!(matches!(err, StorageError::DurableTraceNotFound(_)));
}

#[test]
fn mark_consumed_hides_trace_and_sets_applied_patch_id() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();
    store
        .complete_arbitration("t1", &sample_arbitration(2_000))
        .unwrap();
    store.record_patch(&sample_patch("p1", 3_000)).unwrap();

    store.mark_consumed(&["t1"], "p1").unwrap();

    assert_eq!(store.fetch_unconsumed_arbitrated(10).unwrap().len(), 0);

    let row = store.get_trace("t1").unwrap().unwrap();
    assert!(row.is_consumed);
    assert_eq!(row.applied_patch_id, Some("p1".to_string()));
}

#[test]
fn mark_consumed_rejects_mixed_batch_and_does_not_partially_apply() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("good", 1_000)).unwrap();
    store.record_refusal(&sample_trace("bad", 1_500)).unwrap();
    store
        .complete_arbitration("good", &sample_arbitration(2_000))
        .unwrap();
    // "bad" is left PENDING (not arbitrated).
    store.record_patch(&sample_patch("p1", 3_000)).unwrap();

    let err = store.mark_consumed(&["good", "bad"], "p1").unwrap_err();
    assert!(matches!(err, StorageError::InvalidArbitrationState { .. }));

    // Neither id should have been consumed: the good one must still show
    // up as unconsumed-arbitrated.
    let unconsumed = store.fetch_unconsumed_arbitrated(10).unwrap();
    assert_eq!(unconsumed.len(), 1);
    assert_eq!(unconsumed[0].trace_id, "good");
    assert!(!unconsumed[0].is_consumed);
    assert_eq!(unconsumed[0].applied_patch_id, None);
}

#[test]
fn commit_patch_inserts_patch_and_consumes_traces_together() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();
    store
        .complete_arbitration("t1", &sample_arbitration(2_000))
        .unwrap();

    store
        .commit_patch(&sample_patch("p1", 3_000), &["t1"])
        .unwrap();

    assert_eq!(store.get_patch("p1").unwrap().unwrap().samples_count, 1);
    let row = store.get_trace("t1").unwrap().unwrap();
    assert!(row.is_consumed);
    assert_eq!(row.applied_patch_id, Some("p1".to_string()));
}

#[test]
fn commit_patch_rolls_back_patch_row_when_a_trace_cannot_be_consumed() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("good", 1_000)).unwrap();
    store.record_refusal(&sample_trace("bad", 1_500)).unwrap();
    store
        .complete_arbitration("good", &sample_arbitration(2_000))
        .unwrap();
    // "bad" is left PENDING, so consuming it must fail.
    let mut patch = sample_patch("p1", 3_000);
    patch.samples_count = 2;

    let err = store.commit_patch(&patch, &["good", "bad"]).unwrap_err();
    assert!(matches!(err, StorageError::InvalidArbitrationState { .. }));

    // The patch insert ran first in the same transaction; it must be gone.
    assert!(store.get_patch("p1").unwrap().is_none());
    let good = store.get_trace("good").unwrap().unwrap();
    assert!(!good.is_consumed);
    assert_eq!(good.applied_patch_id, None);
}

#[test]
fn commit_patch_rejects_empty_or_miscounted_trace_list() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();
    store
        .complete_arbitration("t1", &sample_arbitration(2_000))
        .unwrap();

    let err = store
        .commit_patch(&sample_patch("p1", 3_000), &[])
        .unwrap_err();
    assert!(matches!(err, StorageError::InvalidPatchCommit(_)));

    let mut patch = sample_patch("p1", 3_000);
    patch.samples_count = 2;
    let err = store.commit_patch(&patch, &["t1"]).unwrap_err();
    assert!(matches!(err, StorageError::InvalidPatchCommit(_)));

    assert!(store.get_patch("p1").unwrap().is_none());
    assert!(!store.get_trace("t1").unwrap().unwrap().is_consumed);
}

#[test]
fn rollback_isolation_revokes_consumed_traces_and_blocks_resurrection() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();
    store
        .complete_arbitration("t1", &sample_arbitration(2_000))
        .unwrap();
    store.record_patch(&sample_patch("p1", 3_000)).unwrap();
    store.mark_consumed(&["t1"], "p1").unwrap();

    store.rollback_patch("p1", "canary regression").unwrap();

    // (a) patch reads back rolled back with the reason recorded.
    let patch = store.get_patch("p1").unwrap().unwrap();
    assert_eq!(patch.status, "ROLLED_BACK");
    assert_eq!(patch.rollback_reason, Some("canary regression".to_string()));
    assert!(patch.rolled_back_at_ms.is_some());

    // (b) the trace's arbitration_status is now Revoked.
    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Revoked);

    // (c) + (d) neither fetch path returns it.
    assert_eq!(store.fetch_pending_arbitration(10).unwrap().len(), 0);
    assert_eq!(store.fetch_unconsumed_arbitrated(10).unwrap().len(), 0);

    // (e) complete/fail arbitration on it now fail instead of resurrecting it.
    let err = store
        .complete_arbitration("t1", &sample_arbitration(9_999))
        .unwrap_err();
    match err {
        StorageError::InvalidArbitrationState { actual, .. } => assert_eq!(actual, "REVOKED"),
        other => panic!("expected InvalidArbitrationState, got {other:?}"),
    }
    let err = store.fail_arbitration("t1", "late failure").unwrap_err();
    match err {
        StorageError::InvalidArbitrationState { actual, .. } => assert_eq!(actual, "REVOKED"),
        other => panic!("expected InvalidArbitrationState, got {other:?}"),
    }

    // Re-rolling back the same patch is rejected, not a silent no-op.
    let err = store.rollback_patch("p1", "again").unwrap_err();
    assert!(matches!(err, StorageError::PatchAlreadyRolledBack(_)));
}

#[test]
fn fetch_pending_arbitration_under_retry_excludes_rows_at_the_retry_cap() {
    let (_dir, store) = open_temp_store();
    let max_retries: u32 = 3;
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();
    store.record_refusal(&sample_trace("t2", 2_000)).unwrap();
    store.record_refusal(&sample_trace("t3", 3_000)).unwrap();

    for _ in 0..max_retries {
        store.fail_arbitration("t2", "arbiter unavailable").unwrap();
    }

    // The unfiltered fetch still returns all three, t2 included.
    let all_pending = store.fetch_pending_arbitration(10).unwrap();
    assert_eq!(all_pending.len(), 3);

    // The retry-filtered fetch excludes t2, which is now at the cap.
    let under_retry = store
        .fetch_pending_arbitration_under_retry(10, max_retries)
        .unwrap();
    let ids: Vec<&str> = under_retry.iter().map(|t| t.trace_id.as_str()).collect();
    assert_eq!(ids, vec!["t1", "t3"]);
}

#[test]
fn durability_across_reopen_preserves_arbitrated_trace() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("durable_refusal.sqlite3");
    {
        let store = DurableRefusalStore::new(&path).unwrap();
        store.record_refusal(&sample_trace("t1", 1_000)).unwrap();
        store
            .complete_arbitration("t1", &sample_arbitration(2_000))
            .unwrap();
    }

    let store = DurableRefusalStore::new(&path).unwrap();
    let unconsumed = store.fetch_unconsumed_arbitrated(10).unwrap();
    assert_eq!(unconsumed.len(), 1);
    let row = &unconsumed[0];
    assert_eq!(row.trace_id, "t1");
    assert_eq!(row.arbitration_status, ArbitrationStatus::Arbitrated);
    assert_eq!(row.is_answerable, Some(false));
    assert_eq!(row.arbitrated_at_ms, Some(2_000));
    assert!(!row.is_consumed);
}

#[test]
fn store_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DurableRefusalStore>();
}

fn canary(patch_id: &str, refused: bool, hard_stop: bool, latency_us: i64) -> CanaryMetricInput {
    CanaryMetricInput {
        patch_id: patch_id.to_string(),
        recorded_at_ms: 9_000,
        is_fast_pass: !refused,
        is_refused: refused,
        latency_us,
        is_hard_stop: hard_stop,
    }
}

fn active_patch(patch_id: &str) -> DurablePatchRecord {
    DurablePatchRecord {
        status: "ACTIVE".to_string(),
        ..sample_patch(patch_id, 1_000)
    }
}

#[test]
fn canary_metrics_round_trip_and_aggregate_over_latest_window() {
    let (_dir, store) = open_temp_store();
    store.record_patch(&active_patch("p1")).unwrap();
    store.record_patch(&active_patch("p2")).unwrap();
    let id1 = store
        .record_canary_metric(&canary("p1", false, false, 10))
        .unwrap();
    let id2 = store
        .record_canary_metric(&canary("p1", true, true, 30))
        .unwrap();
    store
        .record_canary_metric(&canary("p2", true, false, 999))
        .unwrap();
    assert!(id2 > id1);

    let rows = store.fetch_canary_metrics("p1", 10).unwrap();
    assert_eq!(rows.len(), 2, "other patches' rows are excluded");
    assert_eq!(
        rows[0],
        CanaryMetricRow {
            sample_id: id2,
            patch_id: "p1".to_string(),
            recorded_at_ms: 9_000,
            is_fast_pass: false,
            is_refused: true,
            latency_us: 30,
            is_hard_stop: true,
        }
    );
    assert_eq!(rows[1].sample_id, id1);

    let stats = store.compute_canary_stats("p1", 10).unwrap();
    assert_eq!(
        stats,
        CanaryStats {
            patch_id: "p1".to_string(),
            window: 10,
            total_samples: 2,
            refused_count: 1,
            fast_pass_count: 1,
            hard_stop_count: 1,
            refusal_rate: 0.5,
            fast_pass_rate: 0.5,
            hard_stop_rate: 0.5,
            avg_latency_us: 20.0,
        }
    );
    let latest = store.compute_canary_stats("p1", 1).unwrap();
    assert_eq!(latest.total_samples, 1);
    assert_eq!(latest.avg_latency_us, 30.0);
}

#[test]
fn canary_metric_requires_an_active_patch() {
    let (_dir, store) = open_temp_store();
    assert!(matches!(
        store.record_canary_metric(&canary("missing", false, false, 1)),
        Err(StorageError::DurablePatchNotFound(_))
    ));
    store
        .record_patch(&sample_patch("candidate", 1_000))
        .unwrap();
    assert!(matches!(
        store.record_canary_metric(&canary("candidate", false, false, 1)),
        Err(StorageError::PatchNotActive { ref status, .. }) if status == "CANDIDATE"
    ));
    store.record_patch(&active_patch("p1")).unwrap();
    store.rollback_patch("p1", "bad").unwrap();
    assert!(matches!(
        store.record_canary_metric(&canary("p1", false, false, 1)),
        Err(StorageError::PatchNotActive { ref status, .. }) if status == "ROLLED_BACK"
    ));
    assert!(store.fetch_canary_metrics("p1", 10).unwrap().is_empty());
    assert!(matches!(
        store.fetch_canary_metrics("p1", 0),
        Err(StorageError::InvalidCanaryMetric(_))
    ));
}

#[test]
fn mark_patch_deployed_requires_active_and_orders_deployments() {
    let (_dir, store) = open_temp_store();
    assert!(matches!(
        store.mark_patch_deployed("missing", 1),
        Err(StorageError::DurablePatchNotFound(_))
    ));
    store.record_patch(&active_patch("p1")).unwrap();
    store.record_patch(&active_patch("p2")).unwrap();
    store.record_patch(&active_patch("p3")).unwrap();
    assert!(store.fetch_deployed_active_patches(5).unwrap().is_empty());

    store.mark_patch_deployed("p1", 100).unwrap();
    store.mark_patch_deployed("p2", 200).unwrap();
    store.mark_patch_deployed("p3", 300).unwrap();
    assert_eq!(
        store.get_patch("p2").unwrap().unwrap().deployed_at_ms,
        Some(200)
    );
    store.rollback_patch("p3", "bad").unwrap();
    assert!(matches!(
        store.mark_patch_deployed("p3", 400),
        Err(StorageError::PatchNotActive { .. })
    ));
    let order: Vec<_> = store
        .fetch_deployed_active_patches(5)
        .unwrap()
        .into_iter()
        .map(|p| p.patch_id)
        .collect();
    assert_eq!(order, vec!["p2", "p1"], "rolled-back p3 is excluded");
}
