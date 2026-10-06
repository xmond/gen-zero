//! Integration tests for the Phase 3 patch builder
//! (`gen_zero_service::patch_builder`): compiling ARBITRATED durable-refusal
//! traces into a sha256-addressed patch file, and recording the patch and
//! consumption atomically. Arbitration is done directly through
//! `DurableRefusalStore::complete_arbitration`, so no LLM mock is needed.

use gen_zero_service::patch_builder::{
    body_sha256, spawn_patch_builder_daemon, validate_patch, PatchBuildConfig, PatchBuilder,
    PatchBuilderError, DEFAULT_BASE_MODEL_HASH,
};
use gen_zero_storage::{ArbitrationResult, DurableRefusalStore, RefusalTraceInput};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
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

fn builder(f: &Fixture, output_dir: &Path) -> PatchBuilder {
    let mut config = PatchBuildConfig::new(&f.db_path, output_dir);
    config.batch_size = 50;
    PatchBuilder::new(config, Arc::clone(&f.store))
}

fn record(store: &DurableRefusalStore, trace_id: &str, created_at_ms: i64, question: &str) {
    store
        .record_refusal(&RefusalTraceInput {
            trace_id: trace_id.to_string(),
            created_at_ms,
            context: CONTEXT.to_string(),
            question: question.to_string(),
            candidate: "I cannot answer that from the given context.".to_string(),
            best_span_score: 0.12,
            null_score: 0.95,
            score_diff: -0.83,
            verifier_output: None,
        })
        .unwrap();
}

/// One answerable (positive) and one unanswerable (negative) trace, both
/// arbitrated.
fn seed_positive_and_negative(store: &DurableRefusalStore) {
    record(store, "pos", 1_000, "How tall is Mount Everest?");
    record(store, "neg", 1_100, "How tall is K2?");
    store
        .complete_arbitration(
            "pos",
            &ArbitrationResult {
                is_answerable: true,
                gold_answer: Some("8,849 meters".to_string()),
                evidence_span: Some("rising 8,849 meters above sea level".to_string()),
                contradiction_fact: None,
                arbitrator_model: "test-arbiter".to_string(),
                arbitration_raw: "{}".to_string(),
                arbitrated_at_ms: 2_000,
            },
        )
        .unwrap();
    store
        .complete_arbitration(
            "neg",
            &ArbitrationResult {
                is_answerable: false,
                gold_answer: None,
                evidence_span: None,
                contradiction_fact: Some("the context never mentions K2".to_string()),
                arbitrator_model: "test-arbiter".to_string(),
                arbitration_raw: "{}".to_string(),
                arbitrated_at_ms: 2_100,
            },
        )
        .unwrap();
}

fn assert_nothing_consumed(store: &DurableRefusalStore, ids: &[&str]) {
    for id in ids {
        let row = store.get_trace(id).unwrap().unwrap();
        assert!(!row.is_consumed, "{id} must not be consumed");
        assert_eq!(
            row.applied_patch_id, None,
            "{id} must not be linked to a patch"
        );
    }
    assert_eq!(
        store.fetch_unconsumed_arbitrated(10).unwrap().len(),
        ids.len()
    );
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(entries) => entries.map(|e| e.unwrap().path()).collect(),
        Err(_) => Vec::new(),
    }
}

#[test]
fn empty_database_returns_none_and_writes_nothing() {
    let f = fixture();
    let out = f.output_dir.clone();
    assert!(builder(&f, &out).run_once().unwrap().is_none());
    assert!(json_files(&out).is_empty());
}

#[test]
fn pending_only_traces_are_not_compiled() {
    let f = fixture();
    record(&f.store, "pending", 1_000, "How tall is Mount Everest?");
    let out = f.output_dir.clone();
    assert!(builder(&f, &out).run_once().unwrap().is_none());
    assert!(!f.store.get_trace("pending").unwrap().unwrap().is_consumed);
}

#[test]
fn run_once_compiles_writes_records_and_consumes_then_does_not_repeat() {
    let f = fixture();
    seed_positive_and_negative(&f.store);
    let out = f.output_dir.clone();
    let b = builder(&f, &out);

    let patch = b
        .run_once()
        .unwrap()
        .expect("two arbitrated traces must compile");

    // Header.
    assert_eq!(patch.samples_count, 2);
    assert_eq!(patch.base_model_hash, DEFAULT_BASE_MODEL_HASH);
    assert_eq!(patch.sha256.len(), 64);
    assert!(patch.sha256.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(patch.patch_id, format!("patch_{}", &patch.sha256[..16]));

    // Samples: oldest arbitration first, positive and negative preserved.
    let ids: Vec<&str> = patch.samples.iter().map(|s| s.trace_id.as_str()).collect();
    assert_eq!(ids, ["pos", "neg"]);
    let pos = &patch.samples[0];
    assert!(pos.is_answerable);
    assert_eq!(pos.gold_answer.as_deref(), Some("8,849 meters"));
    assert_eq!(
        pos.evidence_span.as_deref(),
        Some("rising 8,849 meters above sea level")
    );
    assert_eq!(pos.context, CONTEXT);
    let neg = &patch.samples[1];
    assert!(!neg.is_answerable);
    assert_eq!(neg.gold_answer, None);
    assert_eq!(
        neg.contradiction_fact.as_deref(),
        Some("the context never mentions K2")
    );

    // File: exactly one, at the recorded path, hash independently recomputed.
    let file = out.join(format!("{}.json", patch.patch_id));
    assert_eq!(json_files(&out), vec![file.clone()]);
    assert_eq!(patch.patch_file_path, file.to_str().unwrap());
    let on_disk = validate_patch(&file).unwrap();
    assert_eq!(on_disk, patch);
    // The documented hash contract, restated here so the test does not
    // trust the crate's own body struct. Not `json!`: it sorts keys.
    #[derive(serde::Serialize)]
    struct Body<'a> {
        base_model_hash: &'a str,
        created_at_ms: i64,
        samples_count: usize,
        samples: &'a [gen_zero_service::TrainingSamplePair],
    }
    let body = Body {
        base_model_hash: &on_disk.base_model_hash,
        created_at_ms: on_disk.created_at_ms,
        samples_count: on_disk.samples.len(),
        samples: &on_disk.samples,
    };
    let independent = format!("{:x}", Sha256::digest(serde_json::to_vec(&body).unwrap()));
    assert_eq!(independent, patch.sha256);
    assert_eq!(
        body_sha256(
            &on_disk.base_model_hash,
            on_disk.created_at_ms,
            &on_disk.samples
        )
        .unwrap(),
        patch.sha256
    );

    // durable_patches row.
    let row = f.store.get_patch(&patch.patch_id).unwrap().unwrap();
    assert_eq!(row.status, "ACTIVE");
    assert_eq!(row.validation_status, "PASSED");
    assert_eq!(row.sha256, patch.sha256);
    assert_eq!(row.samples_count, 2);
    assert_eq!(row.base_model_hash, DEFAULT_BASE_MODEL_HASH);
    assert_eq!(row.patch_file_path, patch.patch_file_path);
    assert_eq!(row.created_at_ms, patch.created_at_ms);
    assert_eq!(row.deployed_at_ms, None);
    assert_eq!(row.rollback_reason, None);

    // durable_refusal_traces consumption.
    for id in ["pos", "neg"] {
        let t = f.store.get_trace(id).unwrap().unwrap();
        assert!(t.is_consumed, "{id} must be consumed");
        assert_eq!(t.applied_patch_id.as_deref(), Some(patch.patch_id.as_str()));
    }

    // Second run: nothing left, no new file.
    assert!(b.run_once().unwrap().is_none());
    assert_eq!(json_files(&out).len(), 1);
}

#[test]
fn batch_size_limits_each_patch() {
    let f = fixture();
    seed_positive_and_negative(&f.store);
    let out = f.output_dir.clone();
    let mut config = PatchBuildConfig::new(&f.db_path, &out);
    config.batch_size = 1;
    let b = PatchBuilder::new(config, Arc::clone(&f.store));

    let first = b.run_once().unwrap().unwrap();
    let second = b.run_once().unwrap().unwrap();
    assert!(b.run_once().unwrap().is_none());
    assert_eq!(first.samples[0].trace_id, "pos");
    assert_eq!(second.samples[0].trace_id, "neg");
    assert_ne!(first.patch_id, second.patch_id);
    assert_eq!(json_files(&out).len(), 2);
}

#[test]
fn unwritable_output_dir_fails_and_consumes_nothing() {
    let f = fixture();
    seed_positive_and_negative(&f.store);
    // A regular file where the output directory should be.
    let blocker = f.db_path.with_file_name("not_a_dir");
    std::fs::write(&blocker, b"x").unwrap();

    let err = builder(&f, &blocker).run_once().unwrap_err();
    assert!(matches!(err, PatchBuilderError::Io { .. }), "{err}");
    assert_nothing_consumed(&f.store, &["pos", "neg"]);
}

/// Install a trigger that aborts a statement, as a real database-side
/// commit failure. `DurableRefusalStore` checks only column layout, so a
/// trigger does not trip its schema guard.
fn install_failing_trigger(db_path: &Path, sql: &str) {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.execute_batch(sql).unwrap();
}

fn patch_row_count(db_path: &Path) -> i64 {
    let conn = rusqlite::Connection::open(db_path).unwrap();
    conn.query_row("SELECT COUNT(*) FROM durable_patches", [], |r| r.get(0))
        .unwrap()
}

#[test]
fn patch_insert_failure_deletes_file_and_consumes_nothing() {
    let f = fixture();
    seed_positive_and_negative(&f.store);
    install_failing_trigger(
        &f.db_path,
        "CREATE TRIGGER fail_patch_insert BEFORE INSERT ON durable_patches
         BEGIN SELECT RAISE(ABORT, 'injected patch insert failure'); END;",
    );
    let out = f.output_dir.clone();

    let err = builder(&f, &out).run_once().unwrap_err();
    assert!(matches!(err, PatchBuilderError::Storage(_)), "{err}");
    assert!(
        err.to_string().contains("injected patch insert failure"),
        "{err}"
    );
    assert!(
        json_files(&out).is_empty(),
        "uncommitted patch file must be deleted"
    );
    assert_eq!(patch_row_count(&f.db_path), 0);
    assert_nothing_consumed(&f.store, &["pos", "neg"]);
}

#[test]
fn consume_failure_rolls_back_patch_row_and_deletes_file() {
    let f = fixture();
    seed_positive_and_negative(&f.store);
    // Fails on the second trace, after the patch row insert and the first
    // trace update have already run inside the same transaction.
    install_failing_trigger(
        &f.db_path,
        "CREATE TRIGGER fail_consume BEFORE UPDATE OF is_consumed ON durable_refusal_traces
         WHEN NEW.trace_id = 'neg'
         BEGIN SELECT RAISE(ABORT, 'injected consume failure'); END;",
    );
    let out = f.output_dir.clone();

    let err = builder(&f, &out).run_once().unwrap_err();
    assert!(
        err.to_string().contains("injected consume failure"),
        "{err}"
    );
    assert!(
        json_files(&out).is_empty(),
        "uncommitted patch file must be deleted"
    );
    assert_eq!(
        patch_row_count(&f.db_path),
        0,
        "patch insert must roll back"
    );
    assert_nothing_consumed(&f.store, &["pos", "neg"]);
}

#[test]
fn arbitrated_row_missing_its_justification_fails_closed() {
    let f = fixture();
    record(&f.store, "bad", 1_000, "How tall is K2?");
    // The store accepts this; the arbitrator never writes it. An
    // unanswerable verdict with no contradiction_fact is not a usable sample.
    f.store
        .complete_arbitration(
            "bad",
            &ArbitrationResult {
                is_answerable: false,
                gold_answer: None,
                evidence_span: None,
                contradiction_fact: None,
                arbitrator_model: "test-arbiter".to_string(),
                arbitration_raw: "{}".to_string(),
                arbitrated_at_ms: 2_000,
            },
        )
        .unwrap();
    let out = f.output_dir.clone();

    let err = builder(&f, &out).run_once().unwrap_err();
    match err {
        PatchBuilderError::CorruptTrace { trace_id, .. } => assert_eq!(trace_id, "bad"),
        other => panic!("expected CorruptTrace, got {other}"),
    }
    assert!(json_files(&out).is_empty());
    assert_eq!(patch_row_count(&f.db_path), 0);
    assert_nothing_consumed(&f.store, &["bad"]);
}

#[test]
fn zero_batch_size_is_rejected() {
    let f = fixture();
    seed_positive_and_negative(&f.store);
    let mut config = PatchBuildConfig::new(&f.db_path, &f.output_dir);
    config.batch_size = 0;
    let err = PatchBuilder::new(config, Arc::clone(&f.store))
        .run_once()
        .unwrap_err();
    assert!(matches!(err, PatchBuilderError::Config(_)), "{err}");
    assert_nothing_consumed(&f.store, &["pos", "neg"]);
}

/// Build one real patch, then return its path and parsed JSON for tamper
/// tests.
fn built_patch_json(f: &Fixture) -> (PathBuf, serde_json::Value) {
    seed_positive_and_negative(&f.store);
    let out = f.output_dir.clone();
    let patch = builder(f, &out).run_once().unwrap().unwrap();
    let path = PathBuf::from(&patch.patch_file_path);
    let json = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    (path, json)
}

fn assert_tamper_rejected(path: &Path, json: &serde_json::Value, needle: &str) {
    std::fs::write(path, serde_json::to_vec(json).unwrap()).unwrap();
    let err = validate_patch(path).unwrap_err();
    assert!(matches!(err, PatchBuilderError::Validation { .. }), "{err}");
    assert!(err.to_string().contains(needle), "{err}");
}

#[test]
fn validate_patch_rejects_tampered_sample() {
    let f = fixture();
    let (path, mut json) = built_patch_json(&f);
    json["samples"][0]["gold_answer"] = serde_json::json!("9,000 meters");
    assert_tamper_rejected(&path, &json, "sha256 mismatch");
}

#[test]
fn validate_patch_rejects_wrong_samples_count() {
    let f = fixture();
    let (path, mut json) = built_patch_json(&f);
    json["samples_count"] = serde_json::json!(3);
    assert_tamper_rejected(&path, &json, "samples_count");
}

#[test]
fn validate_patch_rejects_empty_samples() {
    let f = fixture();
    let (path, mut json) = built_patch_json(&f);
    json["samples"] = serde_json::json!([]);
    json["samples_count"] = serde_json::json!(0);
    assert_tamper_rejected(&path, &json, "no samples");
}

#[test]
fn validate_patch_rejects_patch_id_not_derived_from_hash() {
    let f = fixture();
    let (path, mut json) = built_patch_json(&f);
    json["patch_id"] = serde_json::json!("patch_0000000000000000");
    assert_tamper_rejected(&path, &json, "patch_id");
}

#[test]
fn validate_patch_rejects_non_json_file() {
    let f = fixture();
    let (path, _) = built_patch_json(&f);
    std::fs::write(&path, b"not json").unwrap();
    let err = validate_patch(&path).unwrap_err();
    assert!(err.to_string().contains("not a patch file"), "{err}");
}

#[tokio::test]
async fn daemon_drains_backlog_and_stops_on_shutdown() {
    let f = fixture();
    seed_positive_and_negative(&f.store);
    let mut config = PatchBuildConfig::new(&f.db_path, &f.output_dir);
    config.batch_size = 1;
    config.poll_interval_secs = 1;
    let b = PatchBuilder::new(config, Arc::clone(&f.store));
    let (tx, rx) = tokio::sync::watch::channel(false);
    let handle = spawn_patch_builder_daemon(b, rx);

    // The first tick fires at once and drains both single-trace batches.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !f.store.fetch_unconsumed_arbitrated(10).unwrap().is_empty() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "daemon did not drain"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(json_files(&f.output_dir).len(), 2);
    assert_eq!(patch_row_count(&f.db_path), 2);

    tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("daemon must stop on shutdown")
        .unwrap();
}
