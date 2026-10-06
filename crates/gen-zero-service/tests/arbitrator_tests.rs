//! End-to-end tests for the Phase 2 LLM arbitrator
//! (`gen_zero_service::arbitrator`): draining `PENDING` durable-refusal
//! traces, judging them against a mock OpenAI-compatible chat-completions
//! server, and writing the verdict back with fail-closed anti-hallucination
//! validation. The arbitrator LLM is stood in by a local Axum mock, as
//! `test_closed_loop.rs` does for the tuning server, so these tests never
//! depend on network access.

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use gen_zero_service::arbitrator::{run_once, spawn_arbitrator_daemon, LlmArbitratorConfig};
use gen_zero_storage::{ArbitrationStatus, DurableRefusalStore, RefusalTraceInput};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::TempDir;

fn open_temp_store() -> (TempDir, DurableRefusalStore) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("durable_refusal.sqlite3");
    let store = DurableRefusalStore::new(&path).unwrap();
    (dir, store)
}

const SAMPLE_CONTEXT: &str =
    "Mount Everest is the tallest mountain on Earth, rising 8,849 meters above sea level.";

fn sample_trace(trace_id: &str, created_at_ms: i64) -> RefusalTraceInput {
    RefusalTraceInput {
        trace_id: trace_id.to_string(),
        created_at_ms,
        context: SAMPLE_CONTEXT.to_string(),
        question: "How tall is Mount Everest?".to_string(),
        candidate: "I cannot answer that from the given context.".to_string(),
        best_span_score: 0.12,
        null_score: 0.95,
        score_diff: -0.83,
        verifier_output: Some("tri_teacher: low confidence".to_string()),
    }
}

/// Shared mock state for the arbitrator's chat-completions endpoint.
/// Configurable per test: force a server error, sleep before responding
/// (to exercise the client timeout), or return a fixed judgement content
/// string.
#[derive(Clone)]
struct ArbiterMockState {
    hits: Arc<AtomicUsize>,
    should_fail: Arc<AtomicBool>,
    sleep_ms: Arc<AtomicU64>,
    content: Arc<Mutex<String>>,
}

impl ArbiterMockState {
    fn new(content: Value) -> Self {
        Self {
            hits: Arc::new(AtomicUsize::new(0)),
            should_fail: Arc::new(AtomicBool::new(false)),
            sleep_ms: Arc::new(AtomicU64::new(0)),
            content: Arc::new(Mutex::new(content.to_string())),
        }
    }
}

async fn arbiter_mock_handler(State(state): State<ArbiterMockState>) -> (StatusCode, Json<Value>) {
    state.hits.fetch_add(1, Ordering::SeqCst);
    let sleep_ms = state.sleep_ms.load(Ordering::SeqCst);
    if sleep_ms > 0 {
        tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
    }
    if state.should_fail.load(Ordering::SeqCst) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "mock arbitrator failure"})),
        );
    }
    let content = state.content.lock().unwrap().clone();
    (
        StatusCode::OK,
        Json(json!({
            "choices": [
                {"message": {"role": "assistant", "content": content}}
            ]
        })),
    )
}

async fn start_arbiter_mock(content: Value) -> (String, ArbiterMockState) {
    let state = ArbiterMockState::new(content);
    let app = Router::new()
        .route("/v1/chat/completions", post(arbiter_mock_handler))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}/v1/chat/completions"), state)
}

fn test_config(endpoint: String) -> LlmArbitratorConfig {
    LlmArbitratorConfig {
        api_endpoint: endpoint,
        api_key: Some("test-arbitrator-key".to_string()),
        model: "gpt-4o-test".to_string(),
        batch_size: 10,
        poll_interval_secs: 30,
        max_retries: 5,
        request_timeout_secs: 2,
    }
}

// ---------------------------------------------------------------------
// (a) positive sample success
// ---------------------------------------------------------------------

#[tokio::test]
async fn positive_sample_is_arbitrated_and_written_back() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let evidence_span = "rising 8,849 meters above sea level";
    assert!(SAMPLE_CONTEXT.contains(evidence_span));
    let (endpoint, _mock) = start_arbiter_mock(json!({
        "is_answerable": true,
        "gold_answer": "8,849 meters",
        "evidence_span": evidence_span,
        "contradiction_fact": "",
        "rationale": "the span states the height directly",
    }))
    .await;
    let config = test_config(endpoint);

    let report = run_once(&config, &store).await.unwrap();
    assert_eq!(report.fetched, 1);
    assert_eq!(report.arbitrated, 1);
    assert_eq!(report.hallucinations_rejected, 0);
    assert_eq!(report.transport_failures, 0);
    assert_eq!(report.parse_failures, 0);

    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Arbitrated);
    assert_eq!(trace.is_answerable, Some(true));
    assert_eq!(trace.evidence_span, Some(evidence_span.to_string()));
    assert_eq!(trace.gold_answer, Some("8,849 meters".to_string()));
    assert_eq!(trace.arbitrator_model, Some(config.model.clone()));
    assert_eq!(trace.retry_count, 0);
}

// ---------------------------------------------------------------------
// (b) negative/contradiction sample success
// ---------------------------------------------------------------------

#[tokio::test]
async fn negative_sample_is_arbitrated_and_written_back() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let (endpoint, _mock) = start_arbiter_mock(json!({
        "is_answerable": false,
        "gold_answer": "",
        "evidence_span": "",
        "contradiction_fact": "the context never mentions K2's height",
        "rationale": "the question asks about a different mountain",
    }))
    .await;
    let config = test_config(endpoint);

    let report = run_once(&config, &store).await.unwrap();
    assert_eq!(report.arbitrated, 1);
    assert_eq!(report.hallucinations_rejected, 0);

    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Arbitrated);
    assert_eq!(trace.is_answerable, Some(false));
    assert_eq!(
        trace.contradiction_fact,
        Some("the context never mentions K2's height".to_string())
    );
    assert_eq!(trace.evidence_span, None);
    assert_eq!(trace.gold_answer, None);
}

// ---------------------------------------------------------------------
// (c) hallucination rejected
// ---------------------------------------------------------------------

#[tokio::test]
async fn fabricated_evidence_span_is_rejected_as_hallucination() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let fabricated_span = "K2 is 8,611 meters tall according to this context";
    assert!(!SAMPLE_CONTEXT.contains(fabricated_span));
    let (endpoint, mock) = start_arbiter_mock(json!({
        "is_answerable": true,
        "gold_answer": "8,611 meters",
        "evidence_span": fabricated_span,
        "contradiction_fact": "",
        "rationale": "fabricated",
    }))
    .await;
    let config = test_config(endpoint);

    let report = run_once(&config, &store).await.unwrap();
    assert_eq!(report.fetched, 1);
    assert_eq!(report.arbitrated, 0);
    assert_eq!(report.hallucinations_rejected, 1);
    assert_eq!(mock.hits.load(Ordering::SeqCst), 1);

    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Pending);
    assert_eq!(trace.retry_count, 1);
    let last_error = trace.last_error.expect("last_error must be recorded");
    assert!(
        last_error.to_lowercase().contains("hallucinat")
            || last_error.to_lowercase().contains("evidence_span"),
        "last_error should describe the hallucination: {last_error}"
    );
    assert_eq!(trace.is_answerable, None);
    assert_eq!(trace.evidence_span, None);
}

#[tokio::test]
async fn empty_evidence_span_with_is_answerable_true_is_rejected_as_hallucination() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let (endpoint, _mock) = start_arbiter_mock(json!({
        "is_answerable": true,
        "gold_answer": "8,849 meters",
        "evidence_span": "",
        "contradiction_fact": "",
        "rationale": "empty span",
    }))
    .await;
    let config = test_config(endpoint);

    let report = run_once(&config, &store).await.unwrap();
    assert_eq!(report.arbitrated, 0);
    assert_eq!(report.hallucinations_rejected, 1);

    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Pending);
    assert_eq!(trace.retry_count, 1);
    let last_error = trace.last_error.expect("last_error must be recorded");
    assert!(
        last_error.to_lowercase().contains("hallucinat")
            || last_error.to_lowercase().contains("evidence_span"),
        "last_error should describe the hallucination: {last_error}"
    );
    assert_eq!(trace.is_answerable, None);
    assert_eq!(trace.evidence_span, None);
}

// ---------------------------------------------------------------------
// (d) API fault + timeout retry/resume
// ---------------------------------------------------------------------

#[tokio::test]
async fn transport_failure_is_retried_and_later_succeeds() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let evidence_span = "rising 8,849 meters above sea level";
    let (endpoint, mock) = start_arbiter_mock(json!({
        "is_answerable": true,
        "gold_answer": "8,849 meters",
        "evidence_span": evidence_span,
        "contradiction_fact": "",
        "rationale": "ok",
    }))
    .await;
    mock.should_fail.store(true, Ordering::SeqCst);
    let config = test_config(endpoint);

    // First call: the mock returns HTTP 500.
    let report = run_once(&config, &store).await.unwrap();
    assert_eq!(report.transport_failures, 1);
    assert_eq!(report.arbitrated, 0);

    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Pending);
    assert_eq!(trace.retry_count, 1);

    // Flip the switch: the same store/trace now succeeds.
    mock.should_fail.store(false, Ordering::SeqCst);
    let report = run_once(&config, &store).await.unwrap();
    assert_eq!(report.arbitrated, 1);
    assert_eq!(report.transport_failures, 0);

    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Arbitrated);
    assert_eq!(trace.is_answerable, Some(true));
}

#[tokio::test]
async fn a_slow_response_past_the_timeout_counts_as_a_transport_failure() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let (endpoint, mock) = start_arbiter_mock(json!({
        "is_answerable": false,
        "gold_answer": "",
        "evidence_span": "",
        "contradiction_fact": "irrelevant, never reached before timeout",
        "rationale": "slow",
    }))
    .await;
    // request_timeout_secs is 2 in test_config; sleep well past that.
    mock.sleep_ms.store(4_000, Ordering::SeqCst);
    let config = test_config(endpoint);

    let report = tokio::time::timeout(Duration::from_secs(10), run_once(&config, &store))
        .await
        .expect("run_once must not hang past the configured request timeout")
        .unwrap();
    assert_eq!(report.transport_failures, 1);
    assert_eq!(report.arbitrated, 0);

    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Pending);
    assert_eq!(trace.retry_count, 1);
}

// ---------------------------------------------------------------------
// (e) retry cap stops delivery
// ---------------------------------------------------------------------

#[tokio::test]
async fn traces_at_the_retry_cap_are_never_sent_to_the_arbitrator() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let max_retries: u32 = 3;
    for _ in 0..max_retries {
        store
            .fail_arbitration("t1", "simulated prior failure")
            .unwrap();
    }
    assert_eq!(
        store.get_trace("t1").unwrap().unwrap().retry_count as u32,
        max_retries
    );

    // A mock that would succeed if it were ever called.
    let evidence_span = "rising 8,849 meters above sea level";
    let (endpoint, mock) = start_arbiter_mock(json!({
        "is_answerable": true,
        "gold_answer": "8,849 meters",
        "evidence_span": evidence_span,
        "contradiction_fact": "",
        "rationale": "would succeed",
    }))
    .await;
    let mut config = test_config(endpoint);
    config.max_retries = max_retries;

    let report = run_once(&config, &store).await.unwrap();
    assert_eq!(
        report.fetched, 0,
        "the retry-capped trace must not be fetched"
    );
    assert_eq!(report.arbitrated, 0);
    assert_eq!(
        mock.hits.load(Ordering::SeqCst),
        0,
        "fetch_pending_arbitration_under_retry must be wired into run_once"
    );

    // Still PENDING, untouched.
    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Pending);
    assert_eq!(trace.retry_count as u32, max_retries);
}

// ---------------------------------------------------------------------
// Misc: missing API key fails closed without touching the store.
// ---------------------------------------------------------------------

#[tokio::test]
async fn missing_api_key_errors_without_fetching() {
    let (_dir, store) = open_temp_store();
    store.record_refusal(&sample_trace("t1", 1_000)).unwrap();

    let mut config = test_config("http://127.0.0.1:1/v1/chat/completions".to_string());
    config.api_key = None;

    let err = run_once(&config, &store).await.unwrap_err();
    assert!(matches!(
        err,
        gen_zero_service::arbitrator::ArbitratorError::MissingApiKey
    ));

    // The trace must be untouched: still pending, no retry recorded.
    let trace = store.get_trace("t1").unwrap().unwrap();
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Pending);
    assert_eq!(trace.retry_count, 0);
}

// ---------------------------------------------------------------------
// spawn_arbitrator_daemon
// ---------------------------------------------------------------------

#[tokio::test]
async fn daemon_stops_on_shutdown_signal() {
    let (_dir, store) = open_temp_store();
    let (endpoint, _mock) = start_arbiter_mock(json!({
        "is_answerable": false,
        "gold_answer": "",
        "evidence_span": "",
        "contradiction_fact": "unused",
        "rationale": "unused",
    }))
    .await;
    let mut config = test_config(endpoint);
    config.poll_interval_secs = 3600; // never ticks during the test

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let handle = spawn_arbitrator_daemon(config, Arc::new(store), shutdown_rx);

    shutdown_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("arbitrator daemon did not stop within 2s of the shutdown signal")
        .unwrap();
}
