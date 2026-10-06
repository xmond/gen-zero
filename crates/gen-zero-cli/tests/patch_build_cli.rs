//! `gen-zero patch-build --once` end to end: the real binary, a real
//! durable refusal database, and the patch file it writes.

use gen_zero_storage::{ArbitrationResult, DurableRefusalStore, RefusalTraceInput};
use std::process::Command;
use tempfile::TempDir;

fn run_once(db: &std::path::Path, out: &std::path::Path) -> serde_json::Value {
    run_once_in(db, out, None)
}

fn run_once_in(
    db: &std::path::Path,
    out: &std::path::Path,
    cwd: Option<&std::path::Path>,
) -> serde_json::Value {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_gen-zero"));
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    let output = cmd
        .args(["patch-build", "--once", "--batch-size", "10"])
        .args(["--base-model-hash", "sha256:cli-test-base"])
        .arg("--db")
        .arg(db)
        .arg("--output-dir")
        .arg(out)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "exit {:?}, stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn patch_build_once_compiles_consumes_and_then_reports_nothing_to_build() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("durable_refusal.sqlite3");
    let out = dir.path().join("patches");
    {
        let store = DurableRefusalStore::new(&db).unwrap();
        store
            .record_refusal(&RefusalTraceInput {
                trace_id: "t1".to_string(),
                created_at_ms: 1_000,
                context: "Mount Everest rises 8,849 meters above sea level.".to_string(),
                question: "How tall is Mount Everest?".to_string(),
                candidate: "I cannot answer.".to_string(),
                best_span_score: 0.1,
                null_score: 0.9,
                score_diff: -0.8,
                verifier_output: None,
            })
            .unwrap();
        store
            .complete_arbitration(
                "t1",
                &ArbitrationResult {
                    is_answerable: true,
                    gold_answer: Some("8,849 meters".to_string()),
                    evidence_span: Some("8,849 meters above sea level".to_string()),
                    contradiction_fact: None,
                    arbitrator_model: "test-arbiter".to_string(),
                    arbitration_raw: "{}".to_string(),
                    arbitrated_at_ms: 2_000,
                },
            )
            .unwrap();
    }

    let first = run_once(&db, &out);
    assert_eq!(first["built"], true);
    assert_eq!(first["samples_count"], 1);
    assert_eq!(first["base_model_hash"], "sha256:cli-test-base");
    let patch_id = first["patch_id"].as_str().unwrap().to_string();
    let file = std::path::PathBuf::from(first["patch_file_path"].as_str().unwrap());
    assert!(file.is_file());
    assert_eq!(file, out.join(format!("{patch_id}.json")));

    let store = DurableRefusalStore::new(&db).unwrap();
    let row = store.get_patch(&patch_id).unwrap().unwrap();
    assert_eq!(row.status, "ACTIVE");
    assert_eq!(row.sha256, first["sha256"].as_str().unwrap());
    let trace = store.get_trace("t1").unwrap().unwrap();
    assert!(trace.is_consumed);
    assert_eq!(trace.applied_patch_id.as_deref(), Some(patch_id.as_str()));
    drop(store);

    let second = run_once(&db, &out);
    assert_eq!(second, serde_json::json!({ "built": false }));
}

#[test]
fn relative_output_dir_is_recorded_as_an_absolute_path() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("durable_refusal.sqlite3");
    {
        let store = DurableRefusalStore::new(&db).unwrap();
        store
            .record_refusal(&RefusalTraceInput {
                trace_id: "t1".to_string(),
                created_at_ms: 1_000,
                context: "K2 is in the Karakoram range.".to_string(),
                question: "How tall is K2?".to_string(),
                candidate: "I cannot answer.".to_string(),
                best_span_score: 0.1,
                null_score: 0.9,
                score_diff: -0.8,
                verifier_output: None,
            })
            .unwrap();
        store
            .complete_arbitration(
                "t1",
                &ArbitrationResult {
                    is_answerable: false,
                    gold_answer: None,
                    evidence_span: None,
                    contradiction_fact: Some("the context gives no height".to_string()),
                    arbitrator_model: "test-arbiter".to_string(),
                    arbitration_raw: "{}".to_string(),
                    arbitrated_at_ms: 2_000,
                },
            )
            .unwrap();
    }

    let out = run_once_in(&db, std::path::Path::new("rel_patches"), Some(dir.path()));
    assert_eq!(out["built"], true);
    let recorded = std::path::PathBuf::from(out["patch_file_path"].as_str().unwrap());
    assert!(recorded.is_absolute(), "{}", recorded.display());
    assert!(recorded.is_file());
    let expected_dir = dir.path().canonicalize().unwrap().join("rel_patches");
    assert_eq!(
        recorded.parent().unwrap().canonicalize().unwrap(),
        expected_dir
    );

    let store = DurableRefusalStore::new(&db).unwrap();
    let patch_id = out["patch_id"].as_str().unwrap();
    assert_eq!(
        store.get_patch(patch_id).unwrap().unwrap().patch_file_path,
        recorded.to_str().unwrap()
    );
}
