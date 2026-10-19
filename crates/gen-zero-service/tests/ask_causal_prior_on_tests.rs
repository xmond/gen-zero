//! `ask` with the policy prior ON (`GENZERO_CAUSAL_PRIOR=on`).
//!
//! Its own test binary on purpose: the variable is process-global, and
//! `worldmodel_verbs_tests` relies on nobody in its process turning it on.
//!
//! A local stub scorer answers with canned probabilities. This checks the
//! plumbing (scorer -> policy prior -> confirmation gate), not a model.

use axum::routing::post;
use axum::{Json, Router};
use gen_zero_service::{
    BridgeConfig, PolymorphicZeroEngine, SemanticBackend, SemanticBridgeClient, ServiceError,
};
use serde_json::{json, Value};
use std::sync::Arc;

/// `semantic_ask` answers `probs` in candidate order; `semantic_risk`
/// always answers p_dangerous = 0.001.
async fn stub_scorer(probs: Vec<f64>) -> String {
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
                        .map(|(name, p)| {
                            json!({
                                "name": name, "log_likelihood": p.ln(),
                                "baseline_log_likelihood": 0.0, "pmi": p.ln(), "probability": p,
                            })
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

/// WebGPT R3 M01.A: the hazard is only in `state`. The scorer's prior sees
/// context + state and penalises both candidates by -16, so the ranking
/// does not move and "drink it" stays chosen. The gate must see the same
/// text and stop the ask; it used to check `context` alone and say ok.
#[tokio::test]
async fn ask_gate_sees_the_state_the_policy_prior_scored() {
    std::env::set_var("GENZERO_CAUSAL_PRIOR", "on");
    let client =
        SemanticBridgeClient::new(BridgeConfig::new(stub_scorer(vec![0.99, 0.01]).await)).unwrap();
    let engine =
        PolymorphicZeroEngine::new().with_semantic(Some(Arc::new(SemanticBackend::Remote(client))));

    let hazard = json!({
        "action": "ask", "context": "select storage",
        "state": {"food": "The milk is expired"},
        "candidates": ["drink it", "eat it"],
    });
    match engine.execute(&hazard).await {
        Err(ServiceError::ConfirmationRequired(msg)) => {
            assert!(msg.contains("causal utility prior"), "{msg}");
            assert!(msg.contains("drink it"), "{msg}");
        }
        Ok(out) => panic!("expected ConfirmationRequired, got ok: {}", out.meta),
        Err(e) => panic!("expected ConfirmationRequired, got {e:?}"),
    }

    // Control: the same request without the hazardous state proceeds.
    let safe = json!({
        "action": "ask", "context": "select storage",
        "candidates": ["drink it", "eat it"],
    });
    let out = engine.execute(&safe).await.expect("no hazard, no veto");
    assert!(!out.is_error, "{}", out.meta);
    assert_eq!(out.meta["chosen_action"], "drink it", "{}", out.meta);
}

/// WebGPT R3 M01.C: with the prior on, `imagine` reports the model's raw
/// step probabilities and raw plan likelihood next to the reweighted ones,
/// and its summary does not call the reweighted score a model likelihood.
#[tokio::test]
async fn imagine_reports_raw_likelihood_beside_the_reweighted_score() {
    std::env::set_var("GENZERO_CAUSAL_PRIOR", "on");
    let client =
        SemanticBridgeClient::new(BridgeConfig::new(stub_scorer(vec![0.99, 0.01]).await)).unwrap();
    let engine =
        PolymorphicZeroEngine::new().with_semantic(Some(Arc::new(SemanticBackend::Remote(client))));
    let out = engine
        .execute(&json!({
            "action": "imagine", "scenario": "The milk is expired",
            "candidate_actions": ["drink it", "throw it away"],
            "horizon": 2, "simulations": 8,
        }))
        .await
        .expect("imagine runs");
    let meta = &out.meta;
    assert_eq!(meta["policy_prior_enabled"], true, "{meta}");
    assert_eq!(meta["likelihood_source"], "policy_prior_reweighted");
    assert_eq!(meta["best_action"], "throw it away", "{meta}");

    let plan = meta["plan"].as_array().expect("plan");
    assert!(!plan.is_empty(), "{meta}");
    let (mut raw_log, mut log) = (0.0, 0.0);
    for step in plan {
        let raw = step["raw_probability"].as_f64().expect("raw_probability");
        let p = step["probability"].as_f64().unwrap();
        // The stub's raw score for "throw it away" is 0.01; the prior lifts it.
        assert!((raw - 0.01).abs() < 1e-9, "{step}");
        assert!(p > 0.99, "{step}");
        raw_log += raw.max(1e-12).ln();
        log += p.max(1e-12).ln();
    }
    let num = |k: &str| meta[k].as_f64().unwrap_or_else(|| panic!("{k}: {meta}"));
    assert!((num("raw_sequence_log_likelihood") - raw_log).abs() < 1e-9);
    assert!((num("raw_sequence_likelihood") - raw_log.exp()).abs() < 1e-12);
    assert!((num("plan_sequence_log_likelihood") - log).abs() < 1e-9);

    let text: String = out.content.iter().map(|c| c.text.as_str()).collect();
    assert!(
        text.contains("policy-prior-reweighted sequence score"),
        "{text}"
    );
    assert!(
        text.contains("raw language-model likelihood of the plan"),
        "{text}"
    );
    assert!(!text.contains("a language-model likelihood"), "{text}");
}
