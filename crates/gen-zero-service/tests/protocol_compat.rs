use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    routing::post as axum_post,
    Json, Router,
};
use futures_util::StreamExt;
use gen_zero_service::{
    server::McpServer, zero::PolymorphicZeroEngine, BridgeConfig, SemanticBackend,
    SemanticBridgeClient,
};
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use tower::ServiceExt;

fn app() -> (Router, Arc<PolymorphicZeroEngine>) {
    let engine = Arc::new(PolymorphicZeroEngine::new().with_semantic(None));
    (McpServer::build_router(engine.clone(), None), engine)
}

/// Canned low-risk scorer, same shape as `test_nanocore_live::stub_scorer` --
/// with `bridge: None` every ask escalates fail-closed (`gen-zero-gate/src/risk.rs`),
/// so a genuine `Tier0Proceed` / 200 needs a real (stubbed) request-risk bridge.
async fn stub_low_risk_bridge() -> String {
    let stub = Router::new().route(
        "/v1/semantic_risk",
        axum_post(|Json(_): Json<Value>| async {
            Json(json!({
                "p_dangerous": 0.01, "log_odds": -4.6, "windows": 1,
                "thresholds": {"escalate": 0.5, "hard_stop": 0.9},
                "classifier": {"name": "test-stub"}, "forward_ms": 0.0,
            }))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub scorer");
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, stub).await.unwrap() });
    format!("http://{addr}")
}

async fn app_with_low_risk_bridge() -> (Router, Arc<PolymorphicZeroEngine>) {
    let client = SemanticBridgeClient::new(BridgeConfig::new(stub_low_risk_bridge().await))
        .expect("stub bridge client");
    let engine = Arc::new(
        PolymorphicZeroEngine::new().with_semantic(Some(Arc::new(SemanticBackend::Remote(client)))),
    );
    (McpServer::build_router(engine.clone(), None), engine)
}

async fn post(app: &Router, uri: &str, body: Value) -> axum::response::Response {
    app.clone()
        .oneshot(
            Request::post(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn json_body(response: axum::response::Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap()
}

#[tokio::test]
async fn http_routes_mcp_frames_and_preserves_request_ids() {
    let (app, _) = app();
    let response = post(
        &app,
        "/message",
        json!({"jsonrpc":"2.0", "id":"call-7", "method":"tools/call",
        "params":{"name":"zero", "arguments":{"action":"compact", "text":"hello"}}}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["id"], "call-7");
    assert_eq!(body["jsonrpc"], "2.0");
    assert_eq!(body["result"]["isError"], false, "{body}");
    let response = post(
        &app,
        "/message",
        json!({"jsonrpc":"2.0", "id":8, "method":"tools/call",
        "params":{"name":"missing", "arguments":{}}}),
    )
    .await;
    let body = json_body(response).await;
    assert_eq!(body["id"], 8);
    assert_eq!(body["error"]["code"], -32601);
    let response = post(
        &app,
        "/message",
        json!({"jsonrpc":"2.0", "method":"notifications/initialized"}),
    )
    .await;
    assert!(response.status().is_success());
    assert!(to_bytes(response.into_body(), 1024)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn malformed_jsonrpc_returns_protocol_errors() {
    let (app, _) = app();
    let response = app
        .clone()
        .oneshot(
            Request::post("/message")
                .header("content-type", "application/json")
                .body(Body::from("{broken"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(json_body(response).await["error"]["code"], -32700);
    for frame in [
        json!({"jsonrpc":"1.0", "method":"tools/list"}),
        json!({"jsonrpc":"2.0", "method":9}),
        json!({"jsonrpc":"2.0", "method":"tools/list", "id":{}}),
    ] {
        let response = post(&app, "/message", frame).await;
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], -32600, "{body}");
        assert!(body["id"].is_null());
    }
    let response = post(
        &app,
        "/message",
        json!({"jsonrpc":"2.0", "id":9, "method":"tools/call",
        "params":{"name":"zero", "arguments":[]}}),
    )
    .await;
    let body = json_body(response).await;
    assert_eq!(body["id"], 9);
    assert_eq!(body["error"]["code"], -32602);
}

#[tokio::test]
async fn decision_http_adapter_fails_closed_on_empty_and_forbidden_candidates() {
    let (app, _) = app();
    // B21: no candidates means nothing to choose from. The server must
    // refuse with 400, never invent a "proceed" out of an empty request.
    let response = post(&app, "/v1/decisions", json!({"action":"ask"})).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["isError"], true);
    assert_eq!(body["error"]["code"], "InvalidParams");
    assert_eq!(body["backend_reachable"], true);
    assert_ne!(body["status"], "ok");
    // A real candidate that is also forbidden leaves nothing feasible: the
    // gate hard-stops rather than picking an action outside the request.
    let response = post(
        &app,
        "/v1/decisions",
        json!({"action":"ask", "candidates":["proceed"], "forbidden_actions":["proceed"]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let body = json_body(response).await;
    assert_ne!(body["status"], "ok");
    assert_eq!(body["isError"], true);
}

#[tokio::test]
async fn decision_http_adapter_preserves_the_success_shape() {
    let (app, _) = app_with_low_risk_bridge().await;
    let response = post(
        &app,
        "/v1/decisions",
        json!({"action":"ask", "candidates":["proceed"]}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["chosen_action"], "proceed");
    assert_eq!(body["best_action"], "proceed");
    assert_eq!(body["probs"], json!({"proceed":1.0}));
    assert_eq!(body["confidence"], 1.0);
    assert_eq!(body["backend_reachable"], true);
    assert_eq!(body["status"], "ok");
    assert!(body["candidates"].is_array());
}

#[tokio::test]
async fn sse_sessions_deliver_only_to_the_owner_and_expire_on_disconnect() {
    let (app, engine) = app();
    let response = app
        .clone()
        .oneshot(Request::get("/sse").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut first = response.into_body().into_data_stream();
    let chunk = first.next().await.unwrap().unwrap();
    let event = std::str::from_utf8(&chunk).unwrap();
    let endpoint = event
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap()
        .to_owned();
    assert!(endpoint.starts_with("/message?session_id="));
    let response = app
        .clone()
        .oneshot(Request::get("/sse").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let mut second = response.into_body().into_data_stream();
    let chunk = second.next().await.unwrap().unwrap();
    let event = std::str::from_utf8(&chunk).unwrap();
    let other_endpoint = event
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    assert_ne!(endpoint, other_endpoint);
    let response = post(
        &app,
        &endpoint,
        json!({"jsonrpc":"2.0", "id":42, "method":"tools/list"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let chunk = tokio::time::timeout(Duration::from_secs(1), first.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let event = std::str::from_utf8(&chunk).unwrap();
    assert!(event.contains("event: message"), "{event}");
    let body: Value = serde_json::from_str(
        event
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["id"], 42);
    assert!(body["result"]["tools"].is_array());
    assert!(
        tokio::time::timeout(Duration::from_millis(20), second.next())
            .await
            .is_err()
    );
    drop(first);
    let before = engine.audit_ledger().0;
    let response = post(
        &app,
        &endpoint,
        json!({"jsonrpc":"2.0", "id":43, "method":"tools/call",
        "params":{"name":"zero", "arguments":{"action":"ask"}}}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        engine.audit_ledger().0,
        before,
        "expired session must not execute the request"
    );
}
