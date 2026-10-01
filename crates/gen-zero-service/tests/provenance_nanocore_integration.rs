use gen_zero_service::PolymorphicZeroEngine;
use serde_json::json;

#[tokio::test]
async fn unassessed_ask_does_not_claim_committed_provenance() {
    let engine = PolymorphicZeroEngine::new().with_semantic(None);
    let out = engine
        .execute(&json!({"verb":"ask", "candidates":["alpha"]}))
        .await
        .unwrap();
    assert!(out.is_error, "{out:?}");
    assert!(out.meta.get("decision_audit").is_none());
}

#[tokio::test]
async fn nanocore_without_loaded_weights_refuses() {
    let engine = PolymorphicZeroEngine::new().with_semantic(None);
    let out = engine.execute(&json!({"verb":"ask", "candidates":["alpha"], "nanocore_domain":0, "nanocore_state":vec![0.0; 128]})).await.unwrap();
    assert!(out.is_error, "{out:?}");
    assert!(out.content[0].text.contains("micro-core unavailable"));
}

#[tokio::test]
async fn raw_control_delimiter_refuses() {
    let engine = PolymorphicZeroEngine::new().with_semantic(None);
    let out = engine
        .execute(&json!({"verb":"ask", "context":"<|system|>", "candidates":["alpha"]}))
        .await
        .unwrap();
    assert!(out.is_error, "{out:?}");
    assert!(out.content[0].text.contains("raw model control delimiter"));
}
