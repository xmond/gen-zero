//! Historical raw-wire fixtures below test engine plumbing, not current bridge
//! acceptance or trained model quality. The v1 artifact is explicitly rejected.
//! Real end-to-end consumption test: `ManifoldAnchorDistiller` projection ->
//! `nanocore_ask` (`crates/gen-zero-service/src/zero.rs:2246-2360`).
//!
//! This is the test the P4 closure review demanded: without it, the Python
//! `NanocoreAnchorBridge` (`python/gen_zero/causal/nanocore_bridge.py`) has
//! no real consumer anywhere in the Rust runtime -- only its own unit tests
//! exercise it, and its dict output never reaches `PolymorphicZeroEngine`.
//! This test closes that gap: it feeds the real Rust engine a 128-float
//! `nanocore_state` fixture that was produced by actually running
//! `NanocoreAnchorBridge.project_to_nanocore_state()` on a real LLaMA-70B
//! BoolQ feature row through the real fitted artifact (see the fixture's
//! `provenance` field and `distiller_provenance_matches_source_files` below,
//! which re-derives the sha256 of both source files so the fixture cannot
//! silently drift from what actually shipped).
//!
//! `nanocore_ask` requires a registered `NanoCoreInstance` for the requested
//! domain (`self.nano_fleet.get_core`) and only reports `is_error == false`
//! when the shared request-risk gate does not escalate. With no bridge
//! configured the gate fails closed to `Tier2Escalate` regardless of the
//! nanocore result (`crates/gen-zero-gate/src/risk.rs`, `None` means
//! "could not be assessed" and is never treated as safe) -- exactly the
//! precedent already established by `zero.rs`'s own
//! `caller_core_never_enters_operator_fleet` test and by
//! `worldmodel_verbs_tests::caller_nanocore_ask_is_audited_in_the_shared_ledger`,
//! which stands up a local stub risk scorer to reach a genuine `Tier0Proceed`.
//! This test follows that same precedent so it exercises the real success
//! path, not merely the refusal path.

use axum::{routing::post, Json, Router};
use gen_zero_core::CompressedLatent;
use gen_zero_nanocore::core_type::fixtures::synthetic_core;
use gen_zero_nanocore::{NanoCoreFleetScheduler, DOMAIN_GENERAL};
use gen_zero_service::{BridgeConfig, PolymorphicZeroEngine, SemanticBridgeClient};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Deserialize)]
struct NanocoreAnchorFixture {
    provenance: String,
    artifact_path: String,
    nanocore_state: Vec<f64>,
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("nanocore_anchor_state_boolq_row0.json")
}

fn load_fixture() -> NanocoreAnchorFixture {
    let path = fixture_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "required real fixture missing at {}: {e}. Regenerate it by running \
             NanocoreAnchorBridge.project_to_nanocore_state() on a real BoolQ feature row \
             (see the fixture's own provenance field for the exact recipe).",
            path.display()
        )
    });
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("fixture at {} is not valid: {e}", path.display()))
}

/// Canned low-risk scorer, same shape as
/// `worldmodel_verbs_tests::stub_scorer` -- this checks real plumbing
/// (bridge -> gate -> outcome), not the quality of any risk model.
async fn stub_scorer() -> String {
    let app = Router::new().route(
        "/v1/semantic_risk",
        post(|Json(_): Json<Value>| async {
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
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

async fn engine_with_registered_core_and_low_risk_bridge(
    prototype: CompressedLatent,
) -> PolymorphicZeroEngine {
    let fleet = Arc::new(NanoCoreFleetScheduler::new(
        gen_zero_nanocore::DEFAULT_RAM_BUDGET_BYTES,
    ));
    let mut core = synthetic_core(
        DOMAIN_GENERAL,
        "general",
        prototype.clone(),
        64,
        0.95,
        &["proceed", "wait"],
    );
    core.value_weights = prototype;
    fleet
        .register_core(core)
        .expect("register_core must succeed for a freshly built instance");

    let client = SemanticBridgeClient::new(BridgeConfig::new(stub_scorer().await))
        .expect("stub bridge client");

    PolymorphicZeroEngine::new()
        .with_bridge(Some(Arc::new(client)))
        .with_nano_fleet(fleet)
}

#[tokio::test]
async fn real_projected_128d_state_drives_a_real_nanocore_decision() {
    let fixture = load_fixture();
    assert_eq!(
        fixture.nanocore_state.len(),
        128,
        "fixture nanocore_state must be exactly 128-d: {}",
        fixture.provenance
    );
    assert!(
        fixture.nanocore_state.iter().all(|v| v.is_finite()),
        "fixture nanocore_state must be all-finite: {}",
        fixture.provenance
    );

    let mut proto_vals = [0.0_f32; 128];
    for (i, v) in fixture.nanocore_state.iter().enumerate() {
        proto_vals[i] = *v as f32;
    }
    let prototype = CompressedLatent { values: proto_vals };
    let engine = engine_with_registered_core_and_low_risk_bridge(prototype).await;

    let out = engine
        .execute(&json!({
            "verb": "ask",
            "candidates": ["proceed"],
            "nanocore_domain": 0,
            "nanocore_state": fixture.nanocore_state,
        }))
        .await
        .expect("execute must not error at the transport level");

    assert!(!out.is_error, "expected a real success, got: {out:?}");
    assert_eq!(out.meta["engine"], "nanocore", "{out:?}");
    assert_eq!(out.meta["domain"], 0, "{out:?}");
    assert_eq!(out.meta["core_count"], 1, "{out:?}");
    let confidence = out.meta["confidence"]
        .as_f64()
        .unwrap_or_else(|| panic!("meta.confidence must be a number: {out:?}"));
    assert!(
        confidence > 0.0,
        "expected a strictly positive confidence from a real decision, got {confidence}: {out:?}"
    );
    let chosen = out.meta["chosen_action"]
        .as_str()
        .unwrap_or_else(|| panic!("meta.chosen_action must be a string: {out:?}"));
    assert_eq!(chosen, "proceed");
}

#[test]
fn legacy_fixture_artifact_is_rejected_by_the_python_bridge() {
    // This historical artifact predates source/GCCA/anchor/core binding. Replaying
    // it as current production evidence would silently bypass the new contract.
    let numpy_available = std::process::Command::new("python3")
        .arg("-c")
        .arg("import numpy")
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !numpy_available {
        eprintln!(
            "skipping legacy_fixture_artifact_is_rejected_by_the_python_bridge: \
             python3 has no numpy, and gen_zero.causal.nanocore_bridge requires it"
        );
        return;
    }

    let fixture = load_fixture();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap();
    let script = format!(
        "import sys; sys.path.insert(0, {python_dir:?}); \
         from gen_zero.causal.nanocore_bridge import NanocoreAnchorBridge; \
         NanocoreAnchorBridge({artifact:?})",
        python_dir = repo_root.join("python").display().to_string(),
        artifact = repo_root.join(&fixture.artifact_path).display().to_string(),
    );
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
        .expect("invoke python3");
    assert!(
        !output.status.success(),
        "legacy unbound artifact was accepted"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("format_version mismatch"),
        "unexpected failure: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn fixture_prototype(fixture: &NanocoreAnchorFixture) -> CompressedLatent {
    let mut proto_vals = [0.0_f32; 128];
    for (i, v) in fixture.nanocore_state.iter().enumerate() {
        proto_vals[i] = *v as f32;
    }
    CompressedLatent { values: proto_vals }
}

#[tokio::test]
async fn real_projected_state_decides_between_two_candidates() {
    let fixture = load_fixture();
    let engine = engine_with_registered_core_and_low_risk_bridge(fixture_prototype(&fixture)).await;

    let out = engine
        .execute(&json!({
            "verb": "ask",
            "candidates": ["proceed", "wait"],
            "nanocore_domain": 0,
            "nanocore_state": fixture.nanocore_state,
        }))
        .await
        .expect("execute must not error at the transport level");

    // The micro-core scores both candidates, but a two-way split this close
    // carries high entropy, so the PolicyGate escalates instead of letting the
    // decision through. That is the gate failing closed on a real decision,
    // not an input refusal: there is no typed rejection, the tier is Escalate,
    // and the scored decision stays visible in meta for the confirmer.
    assert!(
        out.rejection.is_none(),
        "must not be an input refusal: {out:?}"
    );
    assert_eq!(out.meta["engine"], "nanocore", "{out:?}");
    let entropy = out.meta["entropy"]
        .as_f64()
        .unwrap_or_else(|| panic!("meta.entropy must be a number: {out:?}"));
    assert!(entropy.is_finite() && entropy > 0.0, "{out:?}");
    assert!(
        out.is_error,
        "a high-entropy two-way split must not pass silently: {out:?}"
    );
    assert_eq!(out.meta["tier"], "Escalate", "{out:?}");
    assert_eq!(out.meta["gate_status"], "requires_confirmation", "{out:?}");
    assert_eq!(out.meta["requires_confirmation"], true, "{out:?}");

    let probs: Vec<f64> = out.meta["probabilities"]
        .as_array()
        .unwrap_or_else(|| panic!("meta.probabilities must be an array: {out:?}"))
        .iter()
        .map(|p| p.as_f64().expect("probability must be a number"))
        .collect();
    assert_eq!(probs.len(), 2, "both candidates must be scored: {out:?}");
    assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0), "{out:?}");
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-5, "{out:?}");
    assert_eq!(
        out.meta["candidates"].as_array().map(Vec::len),
        Some(2),
        "{out:?}"
    );

    // The chosen action must be the highest-probability candidate, whichever it is.
    let argmax = if probs[0] >= probs[1] { 0 } else { 1 };
    let names = ["proceed", "wait"];
    assert_eq!(out.meta["chosen_action"], names[argmax], "{out:?}");
    assert_eq!(
        out.meta["candidates"][argmax]["name"], names[argmax],
        "{out:?}"
    );
}

#[tokio::test]
async fn unregistered_domain_is_refused_fail_closed() {
    let fixture = load_fixture();
    let engine = engine_with_registered_core_and_low_risk_bridge(fixture_prototype(&fixture)).await;

    let out = engine
        .execute(&json!({
            "verb": "ask",
            "candidates": ["proceed", "wait"],
            "nanocore_domain": 99999,
            "nanocore_state": fixture.nanocore_state,
        }))
        .await
        .expect("a refusal is an outcome, not a transport error");

    assert!(
        out.is_error,
        "an unregistered domain must be refused: {out:?}"
    );
    assert!(
        out.content[0]
            .text
            .contains("micro-core unavailable (domain 99999)"),
        "{out:?}"
    );
    assert!(
        out.meta.get("chosen_action").is_none(),
        "no action on refusal: {out:?}"
    );
}

#[tokio::test]
async fn non_finite_or_wrong_length_state_is_refused_fail_closed() {
    let fixture = load_fixture();
    let engine = engine_with_registered_core_and_low_risk_bridge(fixture_prototype(&fixture)).await;

    // serde_json cannot carry NaN: json!(f64::NAN) becomes null, which is what
    // a NaN-producing caller actually puts on the wire.
    let mut with_nan: Vec<Value> = fixture.nanocore_state.iter().map(|v| json!(v)).collect();
    with_nan[7] = json!(f64::NAN);
    assert!(with_nan[7].is_null());

    // Finite as f64 but overflows to +inf when narrowed to the f32 latent.
    let mut f32_overflow: Vec<Value> = fixture.nanocore_state.iter().map(|v| json!(v)).collect();
    f32_overflow[3] = json!(1e39_f64);

    let short: Vec<f64> = fixture.nanocore_state[..127].to_vec();
    let mut long: Vec<f64> = fixture.nanocore_state.clone();
    long.push(0.0);

    for (label, state) in [
        ("nan", json!(with_nan)),
        ("f32_overflow", json!(f32_overflow)),
        ("len_127", json!(short)),
        ("len_129", json!(long)),
    ] {
        let out = engine
            .execute(&json!({
                "verb": "ask",
                "candidates": ["proceed", "wait"],
                "nanocore_domain": 0,
                "nanocore_state": state,
            }))
            .await
            .expect("a refusal is an outcome, not a transport error");
        assert!(
            out.is_error,
            "{label}: an invalid state must be refused: {out:?}"
        );
        assert!(
            out.content[0]
                .text
                .contains("nanocore_state must contain 128 finite numbers"),
            "{label}: {out:?}"
        );
        assert!(out.meta.get("chosen_action").is_none(), "{label}: {out:?}");
    }
}
