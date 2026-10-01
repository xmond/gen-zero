//! End-to-end tests for the native Rust closed-loop subsystem
//! (`gen_zero_service::closed_loop`): the `/v1/feedback` REST ingestion
//! route, the feedback syncer background task, and the tuning-patch poller
//! background task. The tuning server is stood in by a local Axum mock so
//! these tests never depend on network access.

use arc_swap::ArcSwap;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::get;
use axum::{routing::post, Json, Router};
use gen_zero_service::{
    spawn_feedback_syncer, spawn_patch_poller, ClosedLoopConfig, FeedbackBuffer, FeedbackRecord,
    McpServer, PolymorphicZeroEngine,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::watch;
use tower::util::ServiceExt;

const TUNING_TOKEN: &str = "tuning-secret-token";

fn feedback_record(id: &str) -> FeedbackRecord {
    FeedbackRecord {
        trace_id: id.to_string(),
        task: "synthetic".to_string(),
        input_features: vec![0.1, -0.2, 0.3],
        label: 0,
        timestamp: 1727700000.0,
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The `Authorization` header value seen by the last request into a given
/// mock endpoint, captured for assertion.
#[derive(Default, Clone)]
struct CapturedAuth(Arc<Mutex<Option<String>>>);

impl CapturedAuth {
    fn capture(&self, headers: &HeaderMap) {
        let value = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        *self.0.lock().unwrap() = value;
    }

    fn get(&self) -> Option<String> {
        self.0.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------------
// POST /v1/feedback (REST ingestion into FeedbackBuffer)
// ---------------------------------------------------------------------

#[tokio::test]
async fn feedback_endpoint_ingests_records_into_the_buffer() {
    let engine = Arc::new(PolymorphicZeroEngine::new().with_semantic(None));
    let app = McpServer::build_router(engine, None);

    let records = json!([
        {"trace_id": "t1", "task": "synthetic", "input_features": [0.1, -0.2], "label": 0, "timestamp": 1727700000.0},
        {"trace_id": "t2", "task": "synthetic", "input_features": [0.4], "label": 1, "timestamp": 1727700001.0},
    ]);
    let resp = app
        .oneshot(
            axum::http::Request::post("/v1/feedback")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(records.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), 1 << 16)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["ingested"], 2, "{body}");
    assert_eq!(body["buffered"], 2, "{body}");
}

#[tokio::test]
async fn feedback_endpoint_rejects_a_non_array_body() {
    let engine = Arc::new(PolymorphicZeroEngine::new().with_semantic(None));
    let app = McpServer::build_router(engine, None);

    let resp = app
        .oneshot(
            axum::http::Request::post("/v1/feedback")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    json!({"not": "an array"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------
// spawn_feedback_syncer
// ---------------------------------------------------------------------

#[derive(Clone, Default)]
struct FeedbackMockState {
    hits: Arc<AtomicUsize>,
    received: Arc<Mutex<Vec<FeedbackRecord>>>,
    auth: CapturedAuth,
    status: Arc<Mutex<StatusCode>>,
}

async fn feedback_mock_handler(
    State(state): State<FeedbackMockState>,
    headers: HeaderMap,
    Json(records): Json<Vec<FeedbackRecord>>,
) -> (StatusCode, Json<Value>) {
    state.hits.fetch_add(1, Ordering::SeqCst);
    state.auth.capture(&headers);
    let status = *state.status.lock().unwrap();
    if status == StatusCode::OK {
        state.received.lock().unwrap().extend(records.clone());
    }
    (status, Json(json!({"ingested": records.len()})))
}

async fn start_feedback_mock(status: StatusCode) -> (String, FeedbackMockState) {
    let state = FeedbackMockState {
        status: Arc::new(Mutex::new(status)),
        ..Default::default()
    };
    let app = Router::new()
        .route("/api/v1/feedback", post(feedback_mock_handler))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), state)
}

fn test_config(endpoint: String) -> ClosedLoopConfig {
    ClosedLoopConfig {
        tuning_endpoint: Some(endpoint),
        tuning_token: Some(TUNING_TOKEN.to_string()),
        task: "synthetic".to_string(),
        sync_interval_secs: 1,
        poll_interval_secs: 1,
        models_dir: std::env::temp_dir().join(format!("gen-zero-closed-loop-test-{}", uuid_like())),
    }
}

/// Cheap unique suffix without pulling in a uuid dependency.
fn uuid_like() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        ^ (std::process::id() as u128) << 64
}

#[tokio::test]
async fn feedback_syncer_drains_buffer_and_posts_with_bearer_auth() {
    let (endpoint, mock) = start_feedback_mock(StatusCode::OK).await;
    let config = test_config(endpoint);
    let buffer = Arc::new(FeedbackBuffer::new());
    buffer.record(feedback_record("a"));
    buffer.record(feedback_record("b"));

    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = spawn_feedback_syncer(config, buffer.clone(), shutdown_rx);

    tokio::time::timeout(Duration::from_secs(5), async {
        while mock.hits.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("feedback syncer never called the mock tuning server");

    assert_eq!(buffer.len(), 0, "delivered records must be drained");
    let received = mock.received.lock().unwrap();
    assert_eq!(received.len(), 2);
    assert_eq!(received[0].trace_id, "a");
    assert_eq!(received[1].trace_id, "b");
    assert_eq!(
        mock.auth.get().as_deref(),
        Some(format!("Bearer {TUNING_TOKEN}").as_str()),
        "tuning server must see the configured bearer token"
    );

    handle.abort();
}

#[tokio::test]
async fn feedback_syncer_requeues_records_on_server_error() {
    let (endpoint, mock) = start_feedback_mock(StatusCode::INTERNAL_SERVER_ERROR).await;
    let config = test_config(endpoint);
    let buffer = Arc::new(FeedbackBuffer::new());
    buffer.record(feedback_record("x"));

    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = spawn_feedback_syncer(config, buffer.clone(), shutdown_rx);

    tokio::time::timeout(Duration::from_secs(5), async {
        while mock.hits.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("feedback syncer never attempted delivery");

    // Give the requeue a moment to land (it happens right after the failed response).
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        buffer.len(),
        1,
        "a failed POST must requeue the batch, not drop it"
    );
    assert!(
        mock.received.lock().unwrap().is_empty(),
        "the mock only records a batch on success"
    );

    handle.abort();
}

#[tokio::test]
async fn feedback_syncer_stops_on_shutdown_signal() {
    let (endpoint, _mock) = start_feedback_mock(StatusCode::OK).await;
    let mut config = test_config(endpoint);
    config.sync_interval_secs = 3600; // never ticks during the test
    let buffer = Arc::new(FeedbackBuffer::new());

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = spawn_feedback_syncer(config, buffer, shutdown_rx);

    shutdown_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("feedback syncer did not stop within 2s of the shutdown signal")
        .unwrap();
}

// ---------------------------------------------------------------------
// spawn_patch_poller
// ---------------------------------------------------------------------

#[derive(Clone)]
struct PatchMockState {
    metadata: Arc<Mutex<Value>>,
    payload: Arc<Vec<u8>>,
    payload_sha_header: Arc<Mutex<Option<String>>>,
    metadata_hits: Arc<AtomicUsize>,
    download_hits: Arc<AtomicUsize>,
    auth: CapturedAuth,
}

async fn patch_metadata_handler(
    State(state): State<PatchMockState>,
    headers: HeaderMap,
    Query(_params): Query<HashMap<String, String>>,
) -> Json<Value> {
    state.metadata_hits.fetch_add(1, Ordering::SeqCst);
    state.auth.capture(&headers);
    Json(state.metadata.lock().unwrap().clone())
}

async fn patch_download_handler(State(state): State<PatchMockState>) -> (HeaderMap, Bytes) {
    state.download_hits.fetch_add(1, Ordering::SeqCst);
    let mut headers = HeaderMap::new();
    if let Some(sha) = state.payload_sha_header.lock().unwrap().clone() {
        headers.insert("X-SHA256", sha.parse().unwrap());
    }
    (headers, Bytes::from((*state.payload).clone()))
}

async fn start_patch_mock(
    payload: Vec<u8>,
    metadata: Value,
    correct_header: bool,
) -> (String, PatchMockState) {
    let actual_sha = hex_sha256(&payload);
    let state = PatchMockState {
        metadata: Arc::new(Mutex::new(metadata)),
        payload: Arc::new(payload),
        payload_sha_header: Arc::new(Mutex::new(Some(if correct_header {
            actual_sha
        } else {
            "0".repeat(64)
        }))),
        metadata_hits: Arc::new(AtomicUsize::new(0)),
        download_hits: Arc::new(AtomicUsize::new(0)),
        auth: CapturedAuth::default(),
    };
    let app = Router::new()
        .route("/api/v1/patch/latest", get(patch_metadata_handler))
        .route("/api/v1/patch/latest/download", get(patch_download_handler))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), state)
}

#[tokio::test]
async fn patch_poller_downloads_verifies_and_writes_a_new_patch() {
    let payload = b"reflex patch npz envelope bytes".to_vec();
    let target_sha256 = hex_sha256(&payload);
    let metadata = json!({
        "task": "synthetic",
        "version": "v1",
        "sha256": target_sha256,
        "target_sha256": target_sha256,
        "size_bytes": payload.len(),
    });
    let (endpoint, mock) = start_patch_mock(payload.clone(), metadata, true).await;
    let config = test_config(endpoint);
    let models_dir = config.models_dir.clone();
    let current_version = Arc::new(ArcSwap::from_pointee(String::new()));

    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = spawn_patch_poller(config, current_version.clone(), shutdown_rx);

    tokio::time::timeout(Duration::from_secs(5), async {
        while mock.download_hits.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("patch poller never downloaded the patch");
    // Give the async file write and ArcSwap store a moment to land.
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(current_version.load().as_str(), target_sha256);
    let written = std::fs::read(models_dir.join(format!("{target_sha256}.npz")))
        .expect("patch file must be written to models_dir");
    assert_eq!(written, payload);
    assert_eq!(
        mock.auth.get().as_deref(),
        Some(format!("Bearer {TUNING_TOKEN}").as_str())
    );

    handle.abort();
    let _ = std::fs::remove_dir_all(&models_dir);
}

#[tokio::test]
async fn patch_poller_refuses_a_checksum_mismatch_and_writes_nothing() {
    let payload = b"tampered-in-transit".to_vec();
    let claimed_sha256 = "0".repeat(64); // does not match the real payload hash
    let metadata = json!({
        "task": "synthetic",
        "target_sha256": claimed_sha256,
    });
    let (endpoint, mock) = start_patch_mock(payload, metadata, false).await;
    let config = test_config(endpoint);
    let models_dir = config.models_dir.clone();
    let current_version = Arc::new(ArcSwap::from_pointee(String::new()));

    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = spawn_patch_poller(config, current_version.clone(), shutdown_rx);

    tokio::time::timeout(Duration::from_secs(5), async {
        while mock.download_hits.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("patch poller never attempted the download");
    tokio::time::sleep(Duration::from_millis(150)).await;

    assert_eq!(
        current_version.load().as_str(),
        "",
        "a checksum mismatch must never be published as the current version"
    );
    assert!(
        !models_dir.exists() || std::fs::read_dir(&models_dir).unwrap().next().is_none(),
        "a checksum mismatch must never leave a patch file on disk"
    );

    handle.abort();
    let _ = std::fs::remove_dir_all(&models_dir);
}

#[tokio::test]
async fn patch_poller_skips_download_when_already_current() {
    let payload = b"already-applied-patch".to_vec();
    let target_sha256 = hex_sha256(&payload);
    let metadata = json!({
        "task": "synthetic",
        "target_sha256": target_sha256,
    });
    let (endpoint, mock) = start_patch_mock(payload, metadata, true).await;
    let config = test_config(endpoint);
    let models_dir = config.models_dir.clone();
    let current_version = Arc::new(ArcSwap::from_pointee(target_sha256.clone()));

    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = spawn_patch_poller(config, current_version.clone(), shutdown_rx);

    tokio::time::timeout(Duration::from_secs(5), async {
        while mock.metadata_hits.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("patch poller never checked metadata");
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert_eq!(
        mock.download_hits.load(Ordering::SeqCst),
        0,
        "a patch already applied must never be re-downloaded"
    );

    handle.abort();
    let _ = std::fs::remove_dir_all(&models_dir);
}

#[tokio::test]
async fn patch_poller_stops_on_shutdown_signal() {
    let (endpoint, _mock) =
        start_patch_mock(b"x".to_vec(), json!({"target_sha256": "abc"}), true).await;
    let mut config = test_config(endpoint);
    config.poll_interval_secs = 3600; // never ticks during the test
    let current_version = Arc::new(ArcSwap::from_pointee(String::new()));

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let handle = spawn_patch_poller(config, current_version, shutdown_rx);

    shutdown_tx.send(true).unwrap();
    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("patch poller did not stop within 2s of the shutdown signal")
        .unwrap();
}
