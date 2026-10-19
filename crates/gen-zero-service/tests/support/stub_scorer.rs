//! A local stub scorer for the `ask` / `imagine` plumbing tests. It checks
//! scorer -> policy prior -> gate, not a model.

// Each test binary uses its own subset of these helpers.
#![allow(dead_code)]

use axum::routing::post;
use axum::{Json, Router};
use gen_zero_service::{
    BridgeConfig, PolymorphicZeroEngine, SemanticBackend, SemanticBridgeClient,
};
use serde_json::{json, Value};
use std::sync::Arc;

/// `semantic_ask` answers `probs` in candidate order, plus `raw` as each
/// candidate's `raw_probability` when given (omitted otherwise, as the
/// Python bridge does). `semantic_risk` always answers p_dangerous = 0.001.
pub async fn stub_scorer(probs: Vec<f64>, raw: Option<Vec<f64>>) -> String {
    let app = Router::new()
        .route(
            "/v1/semantic_risk",
            post(|| async {
                Json(json!({
                    "p_dangerous": 0.001, "log_odds": 0.0, "windows": 1,
                    "thresholds": {"escalate": 0.4, "hard_stop": 0.8},
                    "classifier": {"name": "test-stub"}, "forward_ms": 0.0,
                }))
            }),
        )
        .route(
            "/v1/semantic_ask",
            post(move |Json(req): Json<Value>| {
                let probs = probs.clone();
                let raw = raw.clone();
                async move {
                    let names: Vec<String> = req["candidates"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|c| c.as_str().unwrap().to_string())
                        .collect();
                    let scores: Vec<Value> = names
                        .iter()
                        .zip(&probs)
                        .enumerate()
                        .map(|(index, (name, p))| {
                            let mut score = json!({
                                "name": name, "log_likelihood": p.ln(),
                                "baseline_log_likelihood": 0.0, "pmi": p.ln(), "probability": p,
                            });
                            if let Some(raw) = &raw {
                                score["raw_probability"] = json!(raw[index]);
                            }
                            score
                        })
                        .collect();
                    let entropy =
                        -probs.iter().map(|p| p * p.ln()).sum::<f64>() / (names.len() as f64).ln();
                    Json(json!({
                        "chosen": names[0], "chosen_index": 0, "candidates": scores,
                        "entropy": entropy, "scorer": {"name": "test-stub"},
                        "embedding_dim": 0, "timing_ms": 0.0,
                    }))
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

/// An engine on a fresh stub client (its own circuit breaker).
pub async fn engine_with(probs: Vec<f64>, raw: Option<Vec<f64>>) -> PolymorphicZeroEngine {
    let client =
        SemanticBridgeClient::new(BridgeConfig::new(stub_scorer(probs, raw).await)).unwrap();
    PolymorphicZeroEngine::new().with_semantic(Some(Arc::new(SemanticBackend::Remote(client))))
}

/// Explicit `raw_probability` vectors that must be rejected next to a valid
/// `probability: [0.99, 0.01]`: out of range, not summing to 1, or (with no
/// prior applied yet) disagreeing with `probability`.
pub const BAD_RAW: [[f64; 2]; 6] = [
    [2.0, 2.0],
    [0.8, 0.8],
    [-0.1, -0.1],
    [-0.1, 1.1],
    [1e100, 2.0],
    [0.01, 0.99],
];

/// The expired-milk `ask` request.
pub fn milk_ask() -> Value {
    json!({
        "action": "ask", "context": "The milk is expired",
        "candidates": ["drink it", "throw it away"],
    })
}

/// The expired-milk `imagine` request.
pub fn milk_imagine() -> Value {
    json!({
        "action": "imagine", "scenario": "The milk is expired",
        "candidate_actions": ["drink it", "throw it away"],
        "horizon": 2, "simulations": 8,
    })
}

/// `out` is a `BridgeProtocolError` rejection that names `raw_probability`.
pub fn assert_raw_rejected(out: &gen_zero_service::ZeroToolOutcome, what: &str) {
    assert!(out.is_error, "{what}: {}", out.meta);
    let reject = &out.meta["reject"];
    assert_eq!(reject["code"], "InvalidParams", "{what}: {}", out.meta);
    assert_eq!(
        reject["stage"], "BridgeProtocolError",
        "{what}: {}",
        out.meta
    );
    let detail = reject["detail"].as_str().unwrap_or_default();
    assert!(detail.contains("raw_probabilit"), "{what}: {detail}");
}
