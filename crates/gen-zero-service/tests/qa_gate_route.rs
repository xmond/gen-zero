use gen_zero_service::{McpServer, PolymorphicZeroEngine, ZeroEngineConfig, ZeroVerb};
use gen_zero_storage::ArbitrationStatus;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[tokio::test]
async fn fast_pass_and_implicit_route() {
    let engine = PolymorphicZeroEngine::new();
    let out = engine
        .execute(&json!({
            "context": "The capital is Paris.", "question": "What is the capital?",
            "candidate": "Paris", "best_span_score": 3.0, "null_score": 0.0
        }))
        .await
        .unwrap();
    assert_eq!(out.verb, ZeroVerb::QaGate);
    assert!(!out.is_error, "{out:?}");
    assert_eq!(out.meta["evidence"]["fast_pass"], true);
    assert_eq!(out.meta["evidence"]["stage2_triggered"], false);
    assert_eq!(out.meta["evidence"]["tri_sim"], Value::Null);
    assert_eq!(out.meta["evidence"]["final_answer"], "Paris");
}

#[tokio::test]
async fn ambiguity_without_adapter_refuses_and_boundaries_validate() {
    let engine = PolymorphicZeroEngine::new();
    let base = json!({"action":"qa_verify", "context":"Paris is in France", "question":"Where?",
        "candidate":"Paris", "best_span_score": 1.0, "null_score": 1.0});
    let out = engine.execute(&base).await.unwrap();
    assert!(out.is_error);
    assert_eq!(out.rejection.unwrap().code, "GateError");
    for bad in [
        json!({"action":"qa_gate","context":"c","question":"q","candidate":"x","best_span_score":1.0}),
        json!({"action":"qa_gate","context":"c","question":"q","candidate":"x","best_span_score":1.0,"null_score":1.0,"ambiguity_low":2.0,"ambiguity_high":1.0}),
        json!({"action":"qa_gate","context":"c","question":"q","candidate":"","best_span_score":2.0,"null_score":0.0}),
    ] {
        assert!(engine.execute(&bad).await.unwrap().is_error, "{bad}");
    }
}

#[tokio::test]
async fn mcp_route_exposes_gate_evidence() {
    let server = McpServer::try_from_config(ZeroEngineConfig::default()).unwrap();
    let mut frame = br#"{"jsonrpc":"2.0","method":"tools/call","params":{"name":"zero","arguments":{"action":"qa_gate","context":"Paris is in France","question":"Where?","candidate":"Paris","best_span_score":3,"null_score":0}},"id":1}"#.to_vec();
    let reply: Value =
        serde_json::from_str(&server.handle_jsonrpc_frame(&mut frame).await).unwrap();
    assert_eq!(reply["result"]["isError"], false, "{reply}");
    assert_eq!(
        reply["result"]["_meta"]["evidence"]["fast_pass"], true,
        "{reply}"
    );
}

#[tokio::test]
async fn http_route_exposes_gate_and_refusal_status() {
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let router = McpServer::build_router(Arc::new(PolymorphicZeroEngine::new()), None);
    let request = |null_score: f32| {
        Request::builder()
            .method("POST")
            .uri("/v1/decisions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"action":"qa_gate", "context":"Paris is in France",
            "question":"Where?", "candidate":"Paris", "best_span_score":3.0,
            "null_score": null_score})
                .to_string(),
            ))
            .unwrap()
    };
    let response = router.clone().oneshot(request(0.0)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["evidence"]["fast_pass"], true, "{body}");
    let response = router.oneshot(request(3.0)).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["isError"], true, "{body}");
}

#[test]
fn adapter_without_qwen_is_a_startup_error() {
    let config = ZeroEngineConfig {
        tri_teacher_adapter: Some("/missing/adapter.safetensors".into()),
        ..ZeroEngineConfig::default()
    };
    let err = PolymorphicZeroEngine::try_from_config(config)
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("requires GENZERO_QWEN_MODEL_PATH"),
        "{err}"
    );
}

/// Stage 2 trigger is durably recorded for later arbitration.
#[tokio::test]
async fn durable_refusal_store_records_stage2_trigger() {
    let base_model = match std::env::var("GENZERO_QWEN_MODEL_PATH") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_default();
            let default_path = home.join(".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987");
            if default_path.exists() {
                default_path
            } else {
                eprintln!("skipping durable_refusal_store_records_stage2_trigger: no Qwen2.5-0.5B found in cache");
                return;
            }
        }
    };
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let demo_adapter = repo_root.join("examples/weights/tri_teacher_demo.safetensors");
    assert!(
        demo_adapter.exists(),
        "demo adapter must exist at {}",
        demo_adapter.display()
    );

    let db_dir = tempfile::tempdir().unwrap();
    let db_path = db_dir.path().join("refusals.sqlite");

    let config = ZeroEngineConfig::default()
        .with_qwen_model(PathBuf::from(base_model), None)
        .with_refusal_db_path(db_path);
    let config = ZeroEngineConfig {
        tri_teacher_adapter: Some(demo_adapter),
        ..config
    };
    let engine =
        PolymorphicZeroEngine::try_from_config(config).expect("engine with tri-teacher adapter");

    const CONTEXT: &str = "Marie Curie won the Nobel Prize in Physics in 1903.";
    const QUESTION: &str = "When did Marie Curie win the Nobel Prize in Physics?";
    const CANDIDATE: &str = "1903";
    let request = json!({
        "action": "qa_gate",
        "context": CONTEXT,
        "question": QUESTION,
        "candidate": CANDIDATE,
        "best_span_score": 5.0,
        "null_score": 4.5
    });
    let out = engine.execute(&request).await.unwrap();
    assert!(!out.is_error, "{out:?}");
    assert_eq!(out.meta["evidence"]["stage2_triggered"], true, "{out:?}");

    let store = engine
        .durable_refusal_store()
        .expect("refusal store must be configured");
    let pending = store.fetch_pending_arbitration(10).unwrap();
    assert_eq!(pending.len(), 1, "{pending:?}");
    let trace = &pending[0];
    assert_eq!(trace.arbitration_status, ArbitrationStatus::Pending);
    assert_eq!(trace.context, CONTEXT);
    assert_eq!(trace.question, QUESTION);
    assert_eq!(trace.candidate, CANDIDATE);
    assert_eq!(trace.best_span_score, 5.0);
    assert_eq!(trace.null_score, 4.5);
    assert!((trace.score_diff - (-0.5)).abs() < 1e-6, "{trace:?}");
    let verifier_output: Value =
        serde_json::from_str(trace.verifier_output.as_ref().expect("verifier_output set"))
            .unwrap();
    assert_eq!(verifier_output["stage2_triggered"], true);
    assert!(
        verifier_output["tri_sim"].as_f64().is_some(),
        "{verifier_output}"
    );
}
