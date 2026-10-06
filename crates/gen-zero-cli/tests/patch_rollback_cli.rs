//! `gen-zero patch-apply`, `canary-status` and `rollback` end to end: the
//! real binary on a real durable refusal database and a real patch file
//! built by `gen-zero patch-build --once`.

use gen_zero_storage::{
    ArbitrationResult, ArbitrationStatus, CanaryMetricInput, DurableRefusalStore, RefusalTraceInput,
};
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

fn gen_zero(args: &[&str], db: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gen-zero"))
        .args(args)
        .arg("--db")
        .arg(db)
        .output()
        .unwrap()
}

fn ok_json(out: Output) -> serde_json::Value {
    assert!(
        out.status.success(),
        "exit {:?}, stderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

/// Seed two arbitrated traces and build them into one patch with the real
/// `patch-build --once`. Returns the patch id.
fn build_patch(dir: &Path, db: &Path) -> String {
    {
        let store = DurableRefusalStore::new(db).unwrap();
        for (i, id) in ["t1", "t2"].iter().enumerate() {
            store
                .record_refusal(&RefusalTraceInput {
                    trace_id: id.to_string(),
                    created_at_ms: 1_000 + i as i64,
                    context: "Mount Everest rises 8,849 meters above sea level.".to_string(),
                    question: format!("How tall is Mount Everest? ({id})"),
                    candidate: "I cannot answer.".to_string(),
                    best_span_score: 0.1,
                    null_score: 0.9,
                    score_diff: -0.8,
                    verifier_output: None,
                })
                .unwrap();
            store
                .complete_arbitration(
                    id,
                    &ArbitrationResult {
                        is_answerable: true,
                        gold_answer: Some("8,849 meters".to_string()),
                        evidence_span: Some("8,849 meters above sea level".to_string()),
                        contradiction_fact: None,
                        arbitrator_model: "test-arbiter".to_string(),
                        arbitration_raw: "{}".to_string(),
                        arbitrated_at_ms: 2_000 + i as i64,
                    },
                )
                .unwrap();
        }
    }
    let out = dir.join("patches");
    let built = ok_json(
        Command::new(env!("CARGO_BIN_EXE_gen-zero"))
            .args(["patch-build", "--once", "--db"])
            .arg(db)
            .arg("--output-dir")
            .arg(&out)
            .output()
            .unwrap(),
    );
    assert_eq!(built["built"], true, "{built}");
    built["patch_id"].as_str().unwrap().to_string()
}

#[test]
fn apply_canary_status_and_rollback_end_to_end() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("durable_refusal.sqlite3");
    let patch_id = build_patch(dir.path(), &db);

    let applied = ok_json(gen_zero(&["patch-apply", "--patch-id", &patch_id], &db));
    assert_eq!(applied["applied"], true);
    assert_eq!(applied["patch_id"], patch_id.as_str());
    assert_eq!(applied["samples_count"], 2);
    assert!(applied["deployed_at_ms"].as_i64().is_some(), "{applied}");

    {
        let store = DurableRefusalStore::new(&db).unwrap();
        for i in 0..20 {
            store
                .record_canary_metric(&CanaryMetricInput {
                    patch_id: patch_id.clone(),
                    recorded_at_ms: 3_000 + i,
                    is_fast_pass: i % 2 == 0,
                    is_refused: i % 2 == 1,
                    latency_us: 100,
                    is_hard_stop: false,
                })
                .unwrap();
        }
    }
    let status = ok_json(gen_zero(&["canary-status", "--patch-id", &patch_id], &db));
    assert_eq!(status["status"], "ACTIVE");
    assert_eq!(status["stats"]["total_samples"], 20);
    assert_eq!(status["stats"]["refused_count"], 10);
    assert_eq!(status["stats"]["refusal_rate"], 0.5);
    assert_eq!(status["stats"]["avg_latency_us"], 100.0);
    assert_eq!(status["decision"]["decision"], "trip", "{status}");
    assert!(status["decision"]["reason"]
        .as_str()
        .unwrap()
        .contains("refusal_rate 0.5000 > 0.3500"));

    let windowed = ok_json(gen_zero(
        &[
            "canary-status",
            "--patch-id",
            &patch_id,
            "--limit",
            "4",
            "--min-samples",
            "4",
        ],
        &db,
    ));
    assert_eq!(windowed["stats"]["total_samples"], 4);
    assert_eq!(windowed["stats"]["window"], 4);
    // A window smaller than min_samples is refused, not silently clamped.
    let bad = gen_zero(
        &["canary-status", "--patch-id", &patch_id, "--limit", "4"],
        &db,
    );
    assert!(!bad.status.success());
    let stderr = String::from_utf8_lossy(&bad.stderr);
    assert!(stderr.contains("must be >= min_samples"), "{stderr}");

    let rolled = ok_json(gen_zero(
        &[
            "rollback",
            "--patch-id",
            &patch_id,
            "--reason",
            "canary refusal surge",
        ],
        &db,
    ));
    assert_eq!(rolled["rolled_back"], true);
    assert_eq!(rolled["status"], "ROLLED_BACK");
    assert_eq!(rolled["rollback_reason"], "canary refusal surge");
    println!("cli rollback elapsed_us = {}", rolled["elapsed_us"]);

    let store = DurableRefusalStore::new(&db).unwrap();
    for id in ["t1", "t2"] {
        let t = store.get_trace(id).unwrap().unwrap();
        assert_eq!(t.arbitration_status, ArbitrationStatus::Revoked);
    }
    assert!(store.fetch_unconsumed_arbitrated(10).unwrap().is_empty());
    assert!(store.fetch_pending_arbitration(10).unwrap().is_empty());
    drop(store);

    let again = gen_zero(
        &["rollback", "--patch-id", &patch_id, "--reason", "twice"],
        &db,
    );
    assert!(!again.status.success());
    let stderr = String::from_utf8_lossy(&again.stderr);
    assert!(stderr.contains("already been rolled back"), "{stderr}");

    let reapply = gen_zero(&["patch-apply", "--patch-id", &patch_id], &db);
    assert!(!reapply.status.success());
    let stderr = String::from_utf8_lossy(&reapply.stderr);
    assert!(stderr.contains("not ACTIVE"), "{stderr}");

    let after = ok_json(gen_zero(&["canary-status", "--patch-id", &patch_id], &db));
    assert_eq!(after["status"], "ROLLED_BACK");
    assert_eq!(after["rollback_reason"], "canary refusal surge");
}

#[test]
fn commands_refuse_a_missing_database_instead_of_creating_one() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("typo.sqlite3");
    for args in [
        vec!["rollback", "--patch-id", "p", "--reason", "r"],
        vec!["patch-apply", "--patch-id", "p"],
        vec!["canary-status", "--patch-id", "p"],
    ] {
        let out = gen_zero(&args, &db);
        assert!(!out.status.success(), "{args:?}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("does not exist"), "{args:?}: {stderr}");
    }
    assert!(!db.exists());
}

#[test]
fn patch_apply_rejects_a_tampered_patch_file() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("durable_refusal.sqlite3");
    let patch_id = build_patch(dir.path(), &db);
    let path = {
        let store = DurableRefusalStore::new(&db).unwrap();
        store.get_patch(&patch_id).unwrap().unwrap().patch_file_path
    };
    let mut doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    doc["samples"][0]["question"] = serde_json::json!("tampered");
    std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();

    let out = gen_zero(&["patch-apply", "--patch-id", &patch_id], &db);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("sha256 mismatch"), "{stderr}");
    let store = DurableRefusalStore::new(&db).unwrap();
    assert_eq!(
        store.get_patch(&patch_id).unwrap().unwrap().deployed_at_ms,
        None
    );
}
