//! Real SQLite/file/HTTP integration; only the external LLM is scripted.
//! These tests validate lifecycle plumbing, not training or model improvement.
use axum::{extract::State, routing::post, Json, Router};
use gen_zero_service::{
    arbitrator::{run_once, LlmArbitratorConfig},
    hot_reload::{CanaryConfig, CanaryDecision, CanaryGuard, HotReloadError, HotReloadManager},
    patch_builder::{validate_patch, CompiledPatch, PatchBuildConfig, PatchBuilder},
};
use gen_zero_storage::{
    ArbitrationStatus, CanaryMetricInput, DurableRefusalStore, RefusalTraceInput, StorageError,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, VecDeque},
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tempfile::TempDir;

const CONTEXT: &str = "The valve remained closed. No water reached the turbine.";
const EVIDENCE: &str = "The valve remained closed.";
const CONTRADICTION: &str =
    "Water could reach the turbine only if the valve opened; it remained closed.";

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

fn seed(store: &DurableRefusalStore, prefix: &str) -> Vec<String> {
    (0..2)
        .map(|i| {
            let id = format!("{prefix}-{i}");
            store
                .record_refusal(&RefusalTraceInput {
                    trace_id: id.clone(),
                    created_at_ms: i,
                    context: CONTEXT.into(),
                    question: if i == 0 {
                        "What was the valve state?"
                    } else {
                        "Did water drive the turbine?"
                    }
                    .into(),
                    candidate: if i == 0 { "closed" } else { "yes" }.into(),
                    best_span_score: 0.1,
                    null_score: 0.9,
                    score_diff: -0.8,
                    verifier_output: Some("refused".into()),
                })
                .unwrap();
            id
        })
        .collect()
}

fn judgement(positive: bool) -> Value {
    if positive {
        json!({"is_answerable":true,"gold_answer":"closed","evidence_span":EVIDENCE,"contradiction_fact":"","rationale":"Explicit state in context"})
    } else {
        json!({"is_answerable":false,"gold_answer":"","evidence_span":"","contradiction_fact":CONTRADICTION,"rationale":"Candidate contradicts the causal precondition"})
    }
}

// A response script is an explicit external-service fixture, not an inference implementation.
struct Reply {
    positive: bool,
    timeout: bool,
}
struct MockServer {
    config: LlmArbitratorConfig,
    remaining: Arc<Mutex<VecDeque<Reply>>>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for MockServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl MockServer {
    async fn start(replies: Vec<Reply>) -> Self {
        async fn handler(
            State(queue): State<Arc<Mutex<VecDeque<Reply>>>>,
            Json(request): Json<Value>,
        ) -> Json<Value> {
            assert_eq!(request["response_format"]["type"], "json_object");
            let reply = queue
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected extra HTTP arbitration");
            let question = if reply.positive {
                "What was the valve state?"
            } else {
                "Did water drive the turbine?"
            };
            assert!(request["messages"][1]["content"]
                .as_str()
                .unwrap()
                .contains(question));
            if reply.timeout {
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
            Json(json!({"choices":[{"message":{"content":judgement(reply.positive).to_string()}}]}))
        }
        let remaining = Arc::new(Mutex::new(VecDeque::from(replies)));
        let app = Router::new()
            .route("/chat", post(handler))
            .with_state(remaining.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            remaining,
            task,
            config: LlmArbitratorConfig {
                api_endpoint: format!("http://{addr}/chat"),
                api_key: Some("fixture-only".into()),
                model: "scripted-boundary-fixture".into(),
                batch_size: 10,
                request_timeout_secs: 1,
                ..Default::default()
            },
        }
    }
    fn exhausted(&self) {
        assert!(self.remaining.lock().unwrap().is_empty());
    }
}

fn open(path: &Path) -> Arc<DurableRefusalStore> {
    let store = Arc::new(DurableRefusalStore::new(path).unwrap());
    let conn = rusqlite::Connection::open(path).unwrap();
    let mode: String = conn
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .unwrap();
    assert_eq!(mode, "wal");
    store
}

async fn arbitrate_pair(store: &DurableRefusalStore) {
    let server = MockServer::start(vec![
        Reply {
            positive: true,
            timeout: false,
        },
        Reply {
            positive: false,
            timeout: false,
        },
    ])
    .await;
    let report = run_once(&server.config, store).await.unwrap();
    assert_eq!(
        (
            report.fetched,
            report.arbitrated,
            report.transport_failures,
            report.parse_failures,
            report.hallucinations_rejected
        ),
        (2, 2, 0, 0, 0)
    );
    server.exhausted();
}

fn assert_arbitrated(store: &DurableRefusalStore, ids: &[String], retries: &[i64]) {
    for (i, id) in ids.iter().enumerate() {
        let trace = store.get_trace(id).unwrap().unwrap();
        assert_eq!(trace.arbitration_status, ArbitrationStatus::Arbitrated);
        assert_eq!(trace.retry_count, retries[i]);
        assert!(trace.last_error.is_none());
        assert_eq!(trace.is_answerable, Some(i == 0));
        if i == 0 {
            assert_eq!(trace.evidence_span.as_deref(), Some(EVIDENCE));
            assert!(trace
                .context
                .contains(trace.evidence_span.as_ref().unwrap()));
        } else {
            assert_eq!(trace.contradiction_fact.as_deref(), Some(CONTRADICTION));
        }
    }
}

fn build(dir: &Path, store: &Arc<DurableRefusalStore>, ids: &[String]) -> CompiledPatch {
    let builder = PatchBuilder::new(
        PatchBuildConfig::new(dir.join("refusal.db"), dir.join("patches")),
        store.clone(),
    );
    let patch = builder
        .run_once()
        .unwrap()
        .expect("unconsumed samples must yield a patch");
    assert_eq!(
        validate_patch(Path::new(&patch.patch_file_path)).unwrap(),
        patch
    );
    assert_eq!(
        patch
            .samples
            .iter()
            .map(|s| s.trace_id.clone())
            .collect::<BTreeSet<_>>(),
        ids.iter().cloned().collect()
    );
    assert_eq!(patch.samples_count, ids.len());
    let row = store.get_patch(&patch.patch_id).unwrap().unwrap();
    assert_eq!(row.status, "ACTIVE");
    assert_eq!(row.validation_status, "PASSED");
    assert_eq!(row.sha256, patch.sha256);
    assert_eq!(row.samples_count, ids.len() as i64);
    assert!(row.deployed_at_ms.is_none());
    for id in ids {
        let trace = store.get_trace(id).unwrap().unwrap();
        assert!(trace.is_consumed);
        assert_eq!(
            trace.applied_patch_id.as_deref(),
            Some(patch.patch_id.as_str())
        );
    }
    assert!(store.fetch_unconsumed_arbitrated(100).unwrap().is_empty());
    assert!(
        builder.run_once().unwrap().is_none(),
        "no duplicate consumption"
    );
    patch
}

fn metric(id: &str, refused: bool, hard_stop: bool) -> CanaryMetricInput {
    CanaryMetricInput {
        patch_id: id.into(),
        recorded_at_ms: now_ms(),
        is_fast_pass: !refused && !hard_stop,
        is_refused: refused,
        latency_us: 100,
        is_hard_stop: hard_stop,
    }
}

#[tokio::test]
async fn self_healing_full_loop() {
    let dir = TempDir::new().unwrap();
    let store = open(&dir.path().join("refusal.db"));
    let ids = seed(&store, "normal");
    arbitrate_pair(&store).await;
    assert_arbitrated(&store, &ids, &[0, 0]);
    let patch = build(dir.path(), &store, &ids);
    let manager = HotReloadManager::new(store.clone());
    // Corrupt the actual artifact: the deployment gate must reject it without
    // changing either the pointer or the durable deployment timestamp.
    let original = std::fs::read(&patch.patch_file_path).unwrap();
    let mut corrupted: Value = serde_json::from_slice(&original).unwrap();
    corrupted["samples"][0]["context"] = json!("tampered context");
    std::fs::write(
        &patch.patch_file_path,
        serde_json::to_vec(&corrupted).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        manager.apply_patch(&patch.patch_id),
        Err(HotReloadError::PatchFile { .. })
    ));
    assert!(manager.active().is_none());
    assert!(store
        .get_patch(&patch.patch_id)
        .unwrap()
        .unwrap()
        .deployed_at_ms
        .is_none());
    std::fs::write(&patch.patch_file_path, original).unwrap();
    let before = now_ms();
    let deployed = manager.apply_patch(&patch.patch_id).unwrap();
    assert!(Arc::ptr_eq(&deployed, &manager.active().unwrap()));
    let deployed_at = store
        .get_patch(&patch.patch_id)
        .unwrap()
        .unwrap()
        .deployed_at_ms
        .unwrap();
    assert!((before..=now_ms()).contains(&deployed_at));
    let guard = CanaryGuard::new(store.clone(), CanaryConfig::default()).unwrap();
    for i in 0..40 {
        assert!(matches!(
            guard
                .record_and_evaluate(metric(&patch.patch_id, i % 10 == 0, false))
                .unwrap(),
            CanaryDecision::Pass
        ));
    }
    let stats = guard.stats(&patch.patch_id).unwrap();
    assert_eq!(
        (
            stats.total_samples,
            stats.refused_count,
            stats.hard_stop_count
        ),
        (40, 4, 0)
    );
    assert!(!guard
        .check_and_auto_rollback(&patch.patch_id, &manager)
        .unwrap());
    assert!(Arc::ptr_eq(&deployed, &manager.active().unwrap()));
}

// Invoked only by the recovery test. process::exit deliberately bypasses Drop,
// leaving live SQLite and Tokio state behind. This is process-crash testing,
// not a claim about power loss, disk failure or uncommitted transactions.
#[tokio::test]
#[ignore = "subprocess fixture invoked by crash_recovery_and_resume"]
async fn crash_writer() {
    let path = std::env::var_os("GEN_ZERO_E2E_CRASH_DB").expect("child database path required");
    let store = open(Path::new(&path));
    seed(&store, "crash");
    let server = MockServer::start(vec![
        Reply {
            positive: true,
            timeout: false,
        },
        Reply {
            positive: false,
            timeout: true,
        },
    ])
    .await;
    let report = run_once(&server.config, &store).await.unwrap();
    assert_eq!(
        (report.fetched, report.arbitrated, report.transport_failures),
        (2, 1, 1)
    );
    server.exhausted();
    println!("CRASH_READY completed=1 pending=1 transport_failures=1 exit=73");
    std::process::exit(73);
}

#[tokio::test]
async fn crash_recovery_and_resume() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("refusal.db");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_writer", "--ignored", "--nocapture"])
        .env("GEN_ZERO_E2E_CRASH_DB", &path)
        .output()
        .unwrap();
    println!(
        "child status={} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(73));
    assert!(
        std::fs::metadata(path.with_extension("db-wal"))
            .unwrap()
            .len()
            > 0
    );
    let store = open(&path);
    let ids = vec!["crash-0".to_string(), "crash-1".to_string()];
    let completed = store.get_trace(&ids[0]).unwrap().unwrap();
    assert_eq!(completed.arbitration_status, ArbitrationStatus::Arbitrated);
    assert_eq!(completed.retry_count, 0);
    let pending = store.fetch_pending_arbitration(100).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].trace_id, ids[1]);
    assert_eq!(pending[0].retry_count, 1);
    assert!(!pending[0].last_error.as_ref().unwrap().is_empty());
    let server = MockServer::start(vec![Reply {
        positive: false,
        timeout: false,
    }])
    .await;
    let report = run_once(&server.config, &store).await.unwrap();
    assert_eq!(
        (report.fetched, report.arbitrated, report.transport_failures),
        (1, 1, 0)
    );
    server.exhausted();
    assert_arbitrated(&store, &ids, &[0, 1]);
    let unchanged = store.get_trace(&ids[0]).unwrap().unwrap();
    assert_eq!(unchanged.arbitration_raw, completed.arbitration_raw);
    assert_eq!(unchanged.arbitrated_at_ms, completed.arbitrated_at_ms);
    assert!(store.fetch_pending_arbitration(100).unwrap().is_empty());
    build(dir.path(), &store, &ids);
    let conn = rusqlite::Connection::open(&path).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM durable_patches", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
}

async fn chaos(hard_stop: bool) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("refusal.db");
    let store = open(&path);
    let golden_ids = seed(&store, "golden");
    arbitrate_pair(&store).await;
    let golden = build(dir.path(), &store, &golden_ids);
    let manager = HotReloadManager::new(store.clone());
    let golden_ptr = manager.apply_patch(&golden.patch_id).unwrap();
    let poisoned_ids = seed(&store, "poisoned");
    arbitrate_pair(&store).await;
    let poisoned = build(dir.path(), &store, &poisoned_ids);
    let held_reader = manager.apply_patch(&poisoned.patch_id).unwrap();
    assert!(Arc::ptr_eq(&golden_ptr, &manager.snapshot().unwrap()));
    let guard = CanaryGuard::new(store.clone(), CanaryConfig::default()).unwrap();
    for i in 0..20 {
        let decision = guard
            .record_and_evaluate(metric(&poisoned.patch_id, !hard_stop, hard_stop))
            .unwrap();
        assert_eq!(matches!(decision, CanaryDecision::Trip { .. }), i == 19);
    }
    let stats = guard.stats(&poisoned.patch_id).unwrap();
    assert_eq!(stats.total_samples, 20);
    assert_eq!(stats.refused_count, if hard_stop { 0 } else { 20 });
    assert_eq!(stats.hard_stop_count, if hard_stop { 20 } else { 0 });
    let before = now_ms();
    let start = Instant::now();
    assert!(guard
        .check_and_auto_rollback(&poisoned.patch_id, &manager)
        .unwrap());
    println!(
        "CHAOS hard_stop={hard_stop} auto_rollback_elapsed_us={}",
        start.elapsed().as_micros()
    );
    // Includes SQLite I/O and scheduling: measurement, not a fabricated microsecond SLA.
    assert!(Arc::ptr_eq(&golden_ptr, &manager.active().unwrap()));
    assert_eq!(held_reader.patch_id, poisoned.patch_id); // old readers retain a complete immutable generation
    assert!(manager.snapshot().is_none());
    let row = store.get_patch(&poisoned.patch_id).unwrap().unwrap();
    assert_eq!(row.status, "ROLLED_BACK");
    assert!((before..=now_ms()).contains(&row.rolled_back_at_ms.unwrap()));
    assert!(row.rollback_reason.unwrap().contains(if hard_stop {
        "hard_stop_rate"
    } else {
        "refusal_rate"
    }));
    for id in &poisoned_ids {
        let trace = store.get_trace(id).unwrap().unwrap();
        assert_eq!(trace.arbitration_status, ArbitrationStatus::Revoked);
        assert!(trace.is_consumed);
        assert_eq!(
            trace.applied_patch_id.as_deref(),
            Some(poisoned.patch_id.as_str())
        );
        assert!(matches!(
            store.fail_arbitration(id, "late retry"),
            Err(StorageError::InvalidArbitrationState { .. })
        ));
    }
    for id in &golden_ids {
        assert_eq!(
            store.get_trace(id).unwrap().unwrap().arbitration_status,
            ArbitrationStatus::Arbitrated
        );
    }
    assert!(store.fetch_unconsumed_arbitrated(100).unwrap().is_empty());
    assert!(store.fetch_pending_arbitration(100).unwrap().is_empty());
    assert!(
        matches!(manager.rollback(&poisoned.patch_id, "repeat"), Err(HotReloadError::Storage(StorageError::PatchAlreadyRolledBack(id))) if id == poisoned.patch_id)
    );
    assert!(Arc::ptr_eq(&golden_ptr, &manager.active().unwrap()));
    drop(guard);
    drop(manager);
    drop(store);
    let reopened = open(&path);
    for id in &poisoned_ids {
        assert_eq!(
            reopened.get_trace(id).unwrap().unwrap().arbitration_status,
            ArbitrationStatus::Revoked
        );
    }
    assert!(reopened
        .fetch_unconsumed_arbitrated(100)
        .unwrap()
        .is_empty());
    assert!(reopened.fetch_pending_arbitration(100).unwrap().is_empty());
    let restored = HotReloadManager::restore(reopened.clone()).unwrap();
    assert_eq!(restored.active().unwrap().patch_id, golden.patch_id);
    let builder = PatchBuilder::new(
        PatchBuildConfig::new(&path, dir.path().join("patches")),
        reopened,
    );
    assert!(builder.run_once().unwrap().is_none());
}

#[tokio::test]
async fn refusal_spike_trips_and_quarantines() {
    chaos(false).await;
}
#[tokio::test]
async fn hard_stop_trips_and_quarantines() {
    chaos(true).await;
}
