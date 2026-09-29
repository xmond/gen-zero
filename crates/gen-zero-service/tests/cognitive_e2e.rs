//! End-to-end: real HTTP routes and the MCP frame handler into the Spec 25
//! cognitive runtime (mount snapshot -> tangent map -> parallel SSM scan ->
//! geometry gate -> action verifier). No trait is called directly here: every
//! request enters through an exposed entry, as §5.5 requires.
//!
//! The mounted assets are a hand-written linear generator
//! (`tests/fixtures/cognitive_assets_linear2d.json`, dh/dt = -h + u). They
//! exercise the pipeline; they are not a trained model.

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use gen_zero_service::zero::{DEFAULT_TENANT, DEFAULT_WORKSPACE};
use gen_zero_service::{McpServer, MountKey, MountRegistry, PolymorphicZeroEngine, Proposal};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

const TOKEN: &str = "gz_test_cognitive_e2e";

fn assets() -> Value {
    serde_json::from_str(include_str!("fixtures/cognitive_assets_linear2d.json")).unwrap()
}

/// Bridge off: nothing here may depend on the Python scorer.
fn engine() -> Arc<PolymorphicZeroEngine> {
    Arc::new(PolymorphicZeroEngine::new().with_bridge(None))
}

async fn send(
    engine: &Arc<PolymorphicZeroEngine>,
    path: &str,
    body: Value,
    token: Option<&str>,
) -> (StatusCode, Value) {
    let app = McpServer::build_router(Arc::clone(engine), Some(TOKEN.to_string()));
    let mut req = Request::post(path).header(header::CONTENT_TYPE, "application/json");
    if let Some(t) = token {
        req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let resp = app
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 22)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    eprintln!("{path} -> {status}: {}", serde_json::to_string(&v).unwrap());
    (status, v)
}

/// Publish the fixture as the next generation of the default mount.
async fn mounted_engine() -> Arc<PolymorphicZeroEngine> {
    let engine = engine();
    let (status, body) = send(
        &engine,
        "/v1/mounts",
        json!({"base_version": 1, "assets": assets(), "reason": "e2e fixture"}),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["published"]["from"], 1);
    assert_eq!(body["published"]["to"], 2);
    assert_eq!(body["assets"]["trained"], false);
    engine
}

/// `n` events of constant input `u`, 0.1 s apart.
fn window(u: [f64; 2], n: u64) -> Value {
    json!((1..=n)
        .map(|i| json!({"time_ns": i * 100_000_000, "input": u}))
        .collect::<Vec<_>>())
}

fn stream_req(state: [f64; 2]) -> Value {
    json!({"action": "stream", "cognitive": {
        "state": state, "window_start_ns": 0, "events": window([1.0, 0.0], 10),
    }})
}

fn ask_req(state: [f64; 2], candidates: &[&str]) -> Value {
    let mut controls = serde_json::Map::new();
    for c in candidates {
        let u = if *c == "advance" {
            [1.0, 0.0]
        } else {
            [-1.0, 0.0]
        };
        controls.insert(c.to_string(), window(u, 10));
    }
    json!({"action": "ask", "candidates": candidates, "cognitive": {
        "state": state, "goal": [0.5, 0.0], "window_start_ns": 0, "controls": controls,
    }})
}

#[tokio::test]
async fn stream_request_penetrates_the_whole_pipeline_over_http() {
    let engine = mounted_engine().await;
    let (status, body) = send(&engine, "/message", stream_req([0.1, 0.0]), Some(TOKEN)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let meta = &body["result"]["meta"];
    assert_eq!(body["result"]["is_error"], false);
    assert_eq!(meta["engine"], "cognitive_runtime");
    assert_eq!(meta["mount"]["version"], 2);
    assert_eq!(meta["mount"]["has_cognitive_assets"], true);
    let traj = &meta["trajectory"];
    assert_eq!(traj["scan"]["backend"], "cpu_parallel(threads=4)");
    assert_eq!(traj["scan"]["steps"], 10);
    assert_eq!(traj["scan"]["prefix_digest"].as_str().unwrap().len(), 64);
    assert_eq!(traj["gate"]["status"], "within_budget");

    // Closed form of dh/dt = -h + u, u = 1 on axis 0, over 1 s from
    // h0 = log_0(0.1) = artanh(0.1): h(1) = 1 + (h0 - 1) e^-1.
    let h0 = 0.1f64.atanh();
    let h1 = 1.0 + (h0 - 1.0) * (-1.0f64).exp();
    let got = traj["final_tangent"][0].as_f64().unwrap();
    assert!((got - h1).abs() < 1e-9, "tangent {got} vs closed form {h1}");
    let point = traj["final_point"][0].as_f64().unwrap();
    assert!((point - h1.tanh()).abs() < 1e-9, "exp_0 readout {point}");
}

#[tokio::test]
async fn certified_ask_picks_the_descending_candidate_and_refuses_the_rising_one() {
    let engine = mounted_engine().await;
    let (status, body) = send(
        &engine,
        "/v1/decisions",
        ask_req([0.1, 0.0], &["advance", "retreat"]),
        Some(TOKEN),
    )
    .await;
    // The geometry certified "advance". The request text has no reachable
    // risk classifier (bridge off), so the gate escalates: not committed,
    // and the HTTP status says so (428), never 200.
    assert_eq!(status, StatusCode::PRECONDITION_REQUIRED, "{body}");
    assert_eq!(body["certified_action"]["action"], "advance");
    assert_eq!(body["certified_action"]["mount_version"], 2);
    assert_eq!(
        body["certified_action"]["certificate"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(body["committed"], false);
    assert_eq!(body["risk"]["fail_closed"], true);
    let cands = body["cognitive"]["candidates"].as_array().unwrap();
    let retreat = cands.iter().find(|c| c["action"] == "retreat").unwrap();
    assert_eq!(retreat["certified"], false);
    assert_eq!(retreat["reject"]["code"], "EnergyRose");
    let e = &body["certified_action"];
    assert!(e["energy_after"][1].as_f64().unwrap() < e["energy_before"][0].as_f64().unwrap());
}

#[tokio::test]
async fn out_of_ball_state_is_domain_violation_and_http_fails() {
    let engine = mounted_engine().await;
    for state in [[0.6, 0.8], [1.2, 0.0]] {
        let (status, body) = send(&engine, "/message", stream_req(state), Some(TOKEN)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "DomainViolation");
        assert_eq!(body["result"]["rejection"]["code"], "DomainViolation");
        assert!(body["result"]["meta"].get("trajectory").is_none());

        let (status, body) = send(
            &engine,
            "/v1/decisions",
            ask_req(state, &["advance", "retreat"]),
            Some(TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "DomainViolation");
        assert!(body["certified_action"].is_null());
        assert_eq!(body["committed"], false);
    }
}

/// Reproduces the Reviewer 2 finding on `baf2be3`: a tangent vector with a
/// component large enough to overflow a naive `sum(x*x).sqrt()` norm
/// (`|input| = 1e200 > sqrt(f64::MAX) ~= 1.34e154`) must never silently fold
/// to the Poincare ball's origin and pass as a valid state. It has to be
/// rejected as a domain violation, with the request failing closed over
/// HTTP, never a 200.
#[tokio::test]
async fn overflowing_tangent_input_is_domain_violation_not_folded_to_origin() {
    let engine = engine();
    // `scan.max_state_norm` raised so the tangent-space scan budget (which
    // already uses a safe max-abs norm) does not itself intercept the huge
    // input before it reaches `exp0`: this isolates the `exp0` overflow path.
    let mut big_assets = assets();
    big_assets["scan"]["max_state_norm"] = json!(1e250);
    let (status, body) = send(
        &engine,
        "/v1/mounts",
        json!({"base_version": 1, "assets": big_assets, "reason": "e2e overflow fixture"}),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let req = json!({"action": "stream", "cognitive": {
        "state": [0.1, 0.0], "window_start_ns": 0,
        "events": [{"time_ns": 100_000_000u64, "input": [1e200, 0.0]}],
    }});
    let (status, body) = send(&engine, "/message", req, Some(TOKEN)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["result"]["is_error"], true, "{body}");
    assert_eq!(body["error"]["code"], "DomainViolation", "{body}");
    assert_eq!(
        body["result"]["rejection"]["code"], "DomainViolation",
        "{body}"
    );
    // The bug returned a fabricated trajectory rooted at the origin; a
    // rejected request must carry no trajectory at all.
    assert!(body["result"]["meta"].get("trajectory").is_none(), "{body}");
}

#[tokio::test]
async fn rising_energy_is_energy_rose_and_commits_no_action() {
    let engine = mounted_engine().await;
    let (status, body) = send(
        &engine,
        "/v1/decisions",
        ask_req([0.1, 0.0], &["retreat"]),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "EnergyRose");
    assert_eq!(body["error"]["stage"], "action_verifier");
    assert!(body["certified_action"].is_null());
    assert_eq!(body["committed"], false);
    assert!(body.get("chosen_action").is_none());
}

#[tokio::test]
async fn contradictory_pins_are_residual_exceeded() {
    let engine = mounted_engine().await;
    let mut req = stream_req([0.1, 0.0]);
    req["cognitive"]["pins"] = json!([
        {"step": 5, "point": [-0.9, 0.0]},
        {"step": 6, "point": [0.9, 0.0]},
    ]);
    let (status, body) = send(&engine, "/message", req, Some(TOKEN)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "ResidualExceeded");
    assert_eq!(body["error"]["stage"], "geometry_gate");
}

#[tokio::test]
async fn stale_publish_is_cas_conflict_and_stale_request_is_epoch_mismatch() {
    let engine = mounted_engine().await;

    // Publishing against a replaced base is refused, the mount is unchanged.
    let mut next = assets();
    next["decision_temperature"] = json!(0.2);
    let (status, body) = send(
        &engine,
        "/v1/mounts",
        json!({"base_version": 1, "assets": next, "reason": "stale writer"}),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "CasConflict");

    // A request pinned to a retired generation is refused, not re-based.
    let mut req = stream_req([0.1, 0.0]);
    req["mount_version"] = json!(1);
    let (status, body) = send(&engine, "/message", req, Some(TOKEN)).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "EpochMismatch");

    // Registry race: a seal validated on v2, then v3 lands first.
    let key = MountKey::new(DEFAULT_TENANT, DEFAULT_WORKSPACE);
    let v2 = engine.mounts().load(&key).unwrap();
    assert_eq!(v2.version().0, 2);
    let change = gen_zero_service::mount::SnapshotChange {
        atlas: Some([7u8; 32]),
        ..Default::default()
    };
    let proposal = Proposal::new(Arc::clone(&v2), Arc::from(&b"race"[..])).unwrap();
    let candidate = gen_zero_service::mount::CandidateMount::new(
        Arc::new(v2.derive(&change).unwrap()),
        &proposal,
    );
    let budget = gen_zero_service::Budget {
        max_steps: 1,
        max_time_ns: 1_000_000_000,
        max_bytes: 1 << 20,
        residual_limit: 0.0,
        numeric_error: 0.0,
        policy: [9u8; 32],
    };
    let seal = engine
        .mounts()
        .validate(proposal, candidate, &budget)
        .unwrap();
    let (status, _) = send(
        &engine,
        "/v1/mounts",
        json!({"base_version": 2, "assets": next, "reason": "winner"}),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let err = engine.mounts().compare_and_mount(&key, seal).unwrap_err();
    assert_eq!(err.code(), "CasConflict");
    assert_eq!(engine.mounts().load(&key).unwrap().version().0, 3);
}

#[tokio::test]
async fn unmounted_assets_unknown_tenant_and_open_publish_are_refused() {
    let engine = engine();
    // Genesis carries no cognitive assets: no default model is invented.
    let (status, body) = send(&engine, "/message", stream_req([0.1, 0.0]), Some(TOKEN)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "BackendUnavailable");

    let mut req = stream_req([0.1, 0.0]);
    req["tenant"] = json!("ghost");
    let (status, body) = send(&engine, "/message", req, Some(TOKEN)).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "CoverageLost");

    // Invalid assets never become a generation.
    let mut bad = assets();
    bad["gate"]["step_fraction"] = json!(2.5);
    let (status, body) = send(
        &engine,
        "/v1/mounts",
        json!({"base_version": 1, "assets": bad, "reason": "unstable step"}),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "InvalidParams");

    // An open server (no token) does not accept publishes at all.
    let open = McpServer::build_router(Arc::clone(&engine), None);
    let resp = open
        .oneshot(
            Request::post("/v1/mounts")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"base_version": 1, "assets": assets(), "reason": "x"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let key = MountKey::new(DEFAULT_TENANT, DEFAULT_WORKSPACE);
    assert_eq!(engine.mounts().load(&key).unwrap().version().0, 1);
}

#[tokio::test]
async fn degraded_semantic_outcomes_are_not_http_200_either() {
    let engine = engine();
    // Route without a scorer returns input order: an error, so not 200.
    let (status, body) = send(
        &engine,
        "/v1/decisions",
        json!({"intent": "send mail", "tools": ["a", "b"]}),
        Some(TOKEN),
    )
    .await;
    assert_ne!(status, StatusCode::OK, "{body}");
    assert_eq!(body["isError"], true);
}

#[tokio::test]
async fn mcp_frames_reach_the_same_runtime() {
    let engine = mounted_engine().await;
    let server = McpServer {
        engine: Arc::clone(&engine),
        auth_token: None,
        bridge_required: false,
    };
    let call = |args: Value| {
        let mut buf = json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "zero", "arguments": args},
        })
        .to_string()
        .into_bytes();
        buf.resize(buf.len() + simd_json::SIMDJSON_PADDING, 0);
        buf
    };
    let mut ok = call(stream_req([0.1, 0.0]));
    let v: Value = serde_json::from_str(&server.handle_jsonrpc_frame(&mut ok).await).unwrap();
    assert_eq!(v["result"]["isError"], false, "{v}");
    assert_eq!(v["result"]["_meta"]["trajectory"]["scan"]["steps"], 10);
    assert_eq!(v["result"]["_meta"]["mount"]["version"], 2);

    let mut bad = call(stream_req([0.6, 0.8]));
    let v: Value = serde_json::from_str(&server.handle_jsonrpc_frame(&mut bad).await).unwrap();
    assert_eq!(v["result"]["isError"], true, "{v}");
    assert_eq!(v["result"]["_meta"]["reject"]["code"], "DomainViolation");
}

// ---------------------------------------------------------------- entailment
// Scheme 1: the `entail` verb runs the asymmetric Busemann test of
// `gen-zero-lod` on the BoolQ geometry the mount seals. Hand-built points, no
// encoder, no trained model.

fn entail_assets() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/cognitive_assets_boolq_entailment.json"
    ))
    .unwrap()
}

async fn entail_engine() -> Arc<PolymorphicZeroEngine> {
    let engine = engine();
    let (status, body) = send(
        &engine,
        "/v1/mounts",
        json!({"base_version": 1, "assets": entail_assets(), "reason": "entail fixture"}),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["assets"]["entailment"], "boolq_128d");
    engine
}

/// `[H^80 | R^24 | S^23 (24 values)]`: hyperbolic radius `r` at angle `theta`
/// from axis 0, sphere point at angle `phi`.
fn boolq_point(r: f64, theta: f64, phi: f64) -> Vec<f64> {
    let mut v = vec![0.0; 128];
    v[0] = r * theta.cos();
    v[1] = r * theta.sin();
    v[104] = phi.cos();
    v[105] = phi.sin();
    v
}

fn entail_req(passage: &[f64], question: &[f64]) -> Value {
    json!({"action": "entail", "entailment": {"passage": passage, "question": question}})
}

#[tokio::test]
async fn entail_over_http_is_asymmetric_and_bound_to_the_mount() {
    let engine = entail_engine().await;
    let general = boolq_point(0.3, 0.0, 0.0);
    let specific = boolq_point(0.8, 0.02, 0.01);

    let (status, body) = send(
        &engine,
        "/message",
        entail_req(&general, &specific),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let meta = &body["result"]["meta"];
    assert_eq!(body["result"]["is_error"], false);
    assert_eq!(meta["engine"], "cognitive_runtime");
    assert_eq!(meta["mount"]["version"], 2);
    let e = &meta["entailment"];
    assert_eq!(e["is_entailed"], true, "{e}");
    assert_eq!(e["calibrated"], false);
    assert_eq!(e["mount_version"], 2);
    assert_eq!(e["geometry"]["topology"], "boolq_128d");
    assert_eq!(e["geometry"]["frame"].as_str().unwrap().len(), 64);
    assert!(e["confidence"].as_f64().unwrap() > 0.0);
    assert!((e["cone_angle"].as_f64().unwrap() - 0.02).abs() < 1e-12);
    assert_eq!(e["factors"]["deeper"], true);

    // Reverse: a valid answer (false), not an error.
    let (status, body) = send(
        &engine,
        "/message",
        entail_req(&specific, &general),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let e = &body["result"]["meta"]["entailment"];
    assert_eq!(e["is_entailed"], false, "{e}");
    assert_eq!(e["factors"]["deeper"], false);
    assert_eq!(e["confidence"], 0.0);
}

#[tokio::test]
async fn entail_without_a_sealed_geometry_is_refused_not_defaulted() {
    let engine = mounted_engine().await; // linear2d assets: no entailment block
    let (status, body) = send(
        &engine,
        "/message",
        entail_req(&boolq_point(0.3, 0.0, 0.0), &boolq_point(0.8, 0.0, 0.0)),
        Some(TOKEN),
    )
    .await;
    assert_ne!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["result"]["rejection"]["code"], "BackendUnavailable",
        "{body}"
    );
    assert!(body["result"]["meta"].get("entailment").is_none(), "{body}");
}

#[tokio::test]
async fn entail_fails_closed_on_bad_input() {
    let engine = entail_engine().await;
    let good = boolq_point(0.3, 0.0, 0.0);
    let expect = |code: &'static str| {
        move |status: StatusCode, body: &Value| {
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(body["result"]["rejection"]["code"], code, "{body}");
            assert!(body["result"]["meta"].get("entailment").is_none(), "{body}");
        }
    };

    // On the ball boundary and past it; and a norm whose square overflows.
    for r in [1.0, 1.5, 1e200] {
        let (status, body) = send(
            &engine,
            "/message",
            entail_req(&good, &boolq_point(r, 0.0, 0.0)),
            Some(TOKEN),
        )
        .await;
        expect("DomainViolation")(status, &body);
    }
    // Wrong length: the sealed width contract, checked before any geometry.
    let (status, body) = send(
        &engine,
        "/message",
        entail_req(&good, &good[..127]),
        Some(TOKEN),
    )
    .await;
    expect("FiberMismatch")(status, &body);
    // Passage at the origin: no radial direction.
    let (status, body) = send(
        &engine,
        "/message",
        entail_req(&boolq_point(0.0, 0.0, 0.0), &good),
        Some(TOKEN),
    )
    .await;
    expect("DomainViolation")(status, &body);
    // Non-numeric coordinates, an unknown field, and the block on another verb.
    let mut text = json!(good);
    text[3] = json!("nan");
    let (status, body) = send(
        &engine,
        "/message",
        json!({"action": "entail", "entailment": {"passage": text, "question": good}}),
        Some(TOKEN),
    )
    .await;
    expect("InvalidParams")(status, &body);
    let (status, body) = send(
        &engine,
        "/message",
        json!({"action": "entail", "entailment": {"passage": good, "question": good, "cone": 3.0}}),
        Some(TOKEN),
    )
    .await;
    expect("InvalidParams")(status, &body);
    let (status, body) = send(&engine, "/message", json!({"action": "ask", "candidates": ["a"], "entailment": {"passage": good, "question": good}}), Some(TOKEN)).await;
    expect("InvalidParams")(status, &body);
}

/// A displaced pin forces actual relaxation: an unmodified scan has energy
/// 0.005, above 0.003; the least-squares minimum is 0.0025. Neither a zero-step
/// gate nor merely attaching a certificate can satisfy these assertions.
#[tokio::test]
async fn http_and_mcp_ask_require_measured_sheaf_relaxation() {
    let engine = engine();
    let mut fixture = assets();
    fixture["gate"]["residual_limit"] = json!(0.003);
    let (status, body) = send(
        &engine,
        "/v1/mounts",
        json!({"base_version": 1, "assets": fixture, "reason": "forced relaxation"}),
        Some(TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let scan_end = 1.0 + (0.1f64.atanh() - 1.0) * (-0.1f64).exp();
    let req = json!({"action": "ask", "candidates": ["advance"], "cognitive": {
        "state": [0.1, 0.0], "goal": [0.5, 0.0], "window_start_ns": 0,
        "controls": {"advance": window([1.0, 0.0], 1)},
        "pins": [{"step": 1, "point": [(scan_end + 0.1).tanh(), 0.0]}]
    }});
    let (status, http) = send(&engine, "/v1/decisions", req.clone(), Some(TOKEN)).await;
    assert_eq!(status, StatusCode::PRECONDITION_REQUIRED, "{http}");
    let server = McpServer {
        engine,
        auth_token: None,
        bridge_required: false,
    };
    let mut frame = json!({"jsonrpc": "2.0", "id": 42, "method": "tools/call",
        "params": {"name": "zero", "arguments": req}})
    .to_string()
    .into_bytes();
    frame.resize(frame.len() + simd_json::SIMDJSON_PADDING, 0);
    let mcp: Value = serde_json::from_str(&server.handle_jsonrpc_frame(&mut frame).await).unwrap();
    assert_eq!(mcp["id"], 42);
    assert_eq!(
        mcp["result"]["isError"], true,
        "risk unavailable must hold the action: {mcp}"
    );
    for meta in [&http, &mcp["result"]["_meta"]] {
        assert_eq!(meta["certified_action"]["action"], "advance", "{meta}");
        assert_eq!(meta["certified_action"]["mount_version"], 2);
        assert_eq!(
            meta["certified_action"]["certificate"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(meta["committed"], false);
        assert_eq!(meta["risk"]["fail_closed"], true);
        let candidate = &meta["cognitive"]["candidates"][0];
        assert_eq!(candidate["certified"], true);
        let trajectory = &candidate["trajectory"];
        assert_eq!(trajectory["scan"]["steps"], 1);
        let gate = &trajectory["gate"];
        assert_eq!(gate["status"], "strictly_decreased", "{gate}");
        assert!(gate["steps"].as_u64().unwrap() > 0);
        assert!(
            gate["energy_after"][1].as_f64().unwrap() < gate["energy_before"][0].as_f64().unwrap()
        );
        assert!(gate["residual"].as_f64().unwrap() <= 0.003);
        assert!(trajectory["final_tangent"][0].as_f64().unwrap() > scan_end);
        let cert = &meta["certified_action"];
        assert!(
            cert["energy_after"][1].as_f64().unwrap() < cert["energy_before"][0].as_f64().unwrap()
        );
    }
}

#[tokio::test]
async fn contradictory_decision_pins_reject_over_http_and_mcp() {
    let engine = mounted_engine().await;
    let mut req = ask_req([0.1, 0.0], &["advance"]);
    req["cognitive"]["pins"] = json!([
        {"step": 5, "point": [-0.9, 0.0]}, {"step": 6, "point": [0.9, 0.0]}
    ]);
    let (status, http) = send(&engine, "/v1/decisions", req.clone(), Some(TOKEN)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{http}");
    let server = McpServer {
        engine,
        auth_token: None,
        bridge_required: false,
    };
    let mut frame = json!({"jsonrpc": "2.0", "id": 43, "method": "tools/call",
        "params": {"name": "zero", "arguments": req}})
    .to_string()
    .into_bytes();
    frame.resize(frame.len() + simd_json::SIMDJSON_PADDING, 0);
    let mcp: Value = serde_json::from_str(&server.handle_jsonrpc_frame(&mut frame).await).unwrap();
    assert_eq!(mcp["id"], 43);
    assert_eq!(mcp["result"]["isError"], true, "{mcp}");
    for meta in [&http, &mcp["result"]["_meta"]] {
        assert!(meta["certified_action"].is_null(), "{meta}");
        assert_eq!(meta["committed"], false);
        assert!(meta.get("chosen_action").is_none());
        let candidate = &meta["cognitive"]["candidates"][0];
        assert_eq!(candidate["certified"], false);
        assert_eq!(candidate["reject"]["code"], "ResidualExceeded");
        assert_eq!(candidate["reject"]["stage"], "geometry_gate");
        assert!(candidate.get("trajectory").is_none());
    }
}
