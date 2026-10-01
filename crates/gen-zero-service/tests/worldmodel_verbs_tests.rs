//! `simulate`, `what_if`, `audit` and the `decide` modes through the real
//! entries: `zero` (`PolymorphicZeroEngine::execute`), HTTP
//! (`/v1/simulate`, `/v1/what_if`, `/v1/audit_action`, `/message`) and MCP
//! (`tools/list`, `tools/call`).
//!
//! The dynamics under test are the untrained latent prior. These tests prove
//! routing, validation, fail-closed behaviour and the shape of the answers.
//! They say nothing about how good the prior is.
//!
//! Where a test needs a request-risk classifier it starts a local stub
//! scorer that answers with canned numbers. That checks the plumbing (bridge
//! -> gate -> outcome), not the quality of any scorer.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use gen_zero_service::{
    BridgeConfig, McpServer, PolymorphicZeroEngine, SemanticBackend, SemanticBridgeClient, ZeroVerb,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

const DIM: usize = 1024;

fn latent(fill: f64) -> Value {
    json!(vec![fill; DIM])
}

/// Bridge off: the request text of `audit` and `decide` stays unassessed.
fn engine_without_bridge() -> Arc<PolymorphicZeroEngine> {
    Arc::new(PolymorphicZeroEngine::new().with_semantic(None))
}

/// Canned scorer: every text is low risk, the first candidate wins with 0.9
/// (confident enough to stay under the gate's entropy escalation).
async fn stub_scorer() -> String {
    let app = Router::new()
        .route(
            "/v1/semantic_risk",
            post(|Json(_): Json<Value>| async {
                Json(json!({
                    "p_dangerous": 0.01, "log_odds": -4.6, "windows": 1,
                    "thresholds": {"escalate": 0.5, "hard_stop": 0.9},
                    "classifier": {"name": "test-stub"}, "forward_ms": 0.0,
                }))
            }),
        )
        .route(
            "/v1/semantic_ask",
            post(|Json(req): Json<Value>| async move {
                let names: Vec<String> = req["candidates"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_str().unwrap().to_string())
                    .collect();
                let n = names.len();
                let prob = |i: usize| {
                    if n == 1 {
                        1.0
                    } else if i == 0 {
                        0.9
                    } else {
                        0.1 / (n - 1) as f64
                    }
                };
                let scores: Vec<Value> = names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        json!({
                            "name": name, "log_likelihood": -1.0, "baseline_log_likelihood": -1.0,
                            "pmi": 0.0, "probability": prob(i),
                        })
                    })
                    .collect();
                Json(json!({
                    "chosen": names[0], "chosen_index": 0, "candidates": scores,
                    "entropy": 0.5, "scorer": {"name": "test-stub"},
                    "embedding_dim": 0, "timing_ms": 0.0,
                }))
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

async fn engine_with_stub() -> Arc<PolymorphicZeroEngine> {
    let client = SemanticBridgeClient::new(BridgeConfig::new(stub_scorer().await)).unwrap();
    Arc::new(
        PolymorphicZeroEngine::new().with_semantic(Some(Arc::new(SemanticBackend::Remote(client)))),
    )
}

async fn post_json(
    engine: &Arc<PolymorphicZeroEngine>,
    path: &str,
    body: Value,
) -> (StatusCode, Value) {
    let app = McpServer::build_router(Arc::clone(engine), None);
    let resp = app
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 22)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

async fn run(engine: &PolymorphicZeroEngine, args: Value) -> gen_zero_service::ZeroToolOutcome {
    engine.execute(&args).await.expect("engine call")
}

// ------------------------------------------------------------ verb dispatch

#[test]
fn simulate_is_its_own_verb_not_an_imagine_alias() {
    let verb = |v: Value| ZeroVerb::infer_from_input(&v).unwrap();
    assert_eq!(verb(json!({"action": "simulate"})), ZeroVerb::Simulate);
    assert_eq!(verb(json!({"verb": "what_if"})), ZeroVerb::WhatIf);
    assert_eq!(verb(json!({"action": "what-if"})), ZeroVerb::WhatIf);
    assert_eq!(verb(json!({"action": "audit"})), ZeroVerb::Audit);
    assert_eq!(verb(json!({"action": "audit_action"})), ZeroVerb::Audit);
    assert_eq!(verb(json!({"actions": ["a"]})), ZeroVerb::Simulate);
    assert_eq!(verb(json!({"target_action": "a"})), ZeroVerb::Audit);
    // Imagine keeps its own name and its own heuristics.
    assert_eq!(verb(json!({"action": "imagine"})), ZeroVerb::Imagine);
    assert_eq!(verb(json!({"scenario": "s"})), ZeroVerb::Imagine);
}

#[tokio::test]
async fn a_simulate_request_shaped_for_imagine_is_refused_not_rerouted() {
    let engine = engine_without_bridge();
    let out = run(
        &engine,
        json!({"action": "simulate", "scenario": "x", "candidate_actions": ["a", "b"]}),
    )
    .await;
    assert_eq!(out.verb, ZeroVerb::Simulate);
    assert!(out.is_error);
    assert_eq!(out.rejection.as_ref().unwrap().code, "InvalidParams");
}

// ------------------------------------------------------------------- zero

#[tokio::test]
async fn simulate_runs_the_plan_and_says_the_dynamics_are_untrained() {
    let engine = engine_without_bridge();
    let out = run(
        &engine,
        json!({"action": "simulate", "state": latent(0.0), "actions": ["a", "b", "c"]}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.verb, ZeroVerb::Simulate);
    assert_eq!(out.meta["provenance"], "latent_residual_dynamics_untrained");
    assert_eq!(out.meta["trained"], false);
    assert_eq!(out.meta["calibrated"], false);
    assert_eq!(out.meta["simulation"]["steps_simulated"], 3);
    assert_eq!(out.meta["risk"]["assessed"], false);
    assert!(out.meta["mount"]["version"].is_number());
    // The "no manifold coordinates" note belongs to the text verbs.
    assert!(out.meta.get("cognitive_runtime").is_none());
}

#[tokio::test]
async fn simulate_is_deterministic() {
    let engine = engine_without_bridge();
    let req = json!({"action": "simulate", "state": latent(0.1), "actions": ["a", "b"]});
    let a = run(&engine, req.clone()).await;
    let b = run(&engine, req).await;
    assert_eq!(a.meta["simulation"], b.meta["simulation"]);
}

#[tokio::test]
async fn different_actions_give_different_trajectories() {
    let engine = engine_without_bridge();
    let of = |name: &str| {
        let engine = Arc::clone(&engine);
        let req = json!({"action": "simulate", "state": latent(0.0), "actions": [name]});
        async move { run(&engine, req).await.meta["simulation"]["final_state"].clone() }
    };
    assert_ne!(of("a").await, of("b").await);
}

#[tokio::test]
async fn a_hazardous_state_stops_the_rollout_at_the_first_hazard() {
    let engine = engine_without_bridge();
    let out = run(
        &engine,
        json!({"action": "simulate", "state": latent(10.0), "actions": ["a", "b", "c"]}),
    )
    .await;
    assert!(!out.is_error);
    let sim = &out.meta["simulation"];
    assert_eq!(sim["is_safe"], false);
    assert_eq!(sim["first_hazard_step"], 1);
    assert_eq!(sim["steps_simulated"], 1);
}

#[tokio::test]
async fn what_if_ranks_every_candidate_and_stays_advisory() {
    let engine = engine_without_bridge();
    let out = run(
        &engine,
        json!({"action": "what_if", "state": latent(0.0), "candidates": ["a", "b", "c"], "horizon": 3}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.verb, ZeroVerb::WhatIf);
    assert_eq!(out.meta["ranking"].as_array().unwrap().len(), 3);
    assert_eq!(out.meta["outcomes"].as_array().unwrap().len(), 3);
    assert_eq!(out.meta["advisory_only"], true);
    assert_eq!(out.meta["provenance"], "latent_residual_dynamics_untrained");
}

#[tokio::test]
async fn audit_without_a_scorer_stays_unassessed_and_asks_for_confirmation() {
    let engine = engine_without_bridge();
    let out = run(
        &engine,
        json!({"action": "audit", "state": latent(0.0), "target_action": "a"}),
    )
    .await;
    assert!(!out.is_error, "the audit ran: {:?}", out.meta);
    assert_eq!(out.meta["verdict"], "REQUIRES_CONFIRMATION");
    assert_eq!(out.meta["risk"]["assessed"], false);
    assert_eq!(out.meta["risk"]["fail_closed"], true);
}

#[tokio::test]
async fn audit_never_approves_even_with_a_clean_risk_check() {
    let engine = engine_with_stub().await;
    let out = run(
        &engine,
        json!({"action": "audit", "state": latent(0.0), "target_action": "a"}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.meta["risk"]["assessed"], true);
    assert_eq!(out.meta["verdict"], "UNVERIFIED_UNTRAINED_DYNAMICS");
    assert_ne!(out.meta["verdict"], "APPROVED");
}

#[tokio::test]
async fn audit_reports_a_lethal_rollout() {
    let engine = engine_with_stub().await;
    let out = run(
        &engine,
        json!({"action": "audit", "state": latent(10.0), "target_action": "a"}),
    )
    .await;
    assert_eq!(out.meta["verdict"], "REJECT_LETHAL");
    assert_eq!(out.meta["hazard_detected"], true);
}

#[tokio::test]
async fn world_model_fields_are_refused_on_other_verbs() {
    let engine = engine_without_bridge();
    for (verb, field) in [
        ("ask", "actions"),
        ("route", "target_action"),
        ("imagine", "continuation_actions"),
    ] {
        let out = run(&engine, json!({"action": verb, field: ["a"]})).await;
        assert!(out.is_error, "{verb}/{field}");
        let rejection = out.rejection.unwrap();
        assert_eq!(rejection.code, "InvalidParams", "{verb}/{field}");
        assert!(rejection.detail.contains(field), "{}", rejection.detail);
    }
    let out = run(
        &engine,
        json!({"action": "simulate", "state": latent(0.0), "actions": ["a"], "cognitive": {}}),
    )
    .await;
    assert!(out.is_error, "cognitive is not a simulate field");
}

#[tokio::test]
async fn bad_world_model_input_is_refused_and_never_repaired() {
    let engine = engine_without_bridge();
    let cases = [
        json!({"action": "simulate", "state": [0.0, 1.0], "actions": ["a"]}),
        json!({"action": "simulate", "state": "text state", "actions": ["a"]}),
        json!({"action": "simulate", "state": latent(0.0)}),
        json!({"action": "simulate", "state": latent(0.0), "actions": ["a", 3]}),
        json!({"action": "simulate", "state": latent(0.0), "actions": ["a"], "horizon": 0}),
        json!({"action": "simulate", "state": latent(0.0), "actions": ["a"], "horizon": true}),
        json!({"action": "simulate", "state": latent(0.0), "actions": ["a"], "horizon": 2}),
        json!({"action": "simulate", "state": latent(0.0), "actions": ["a"], "horizon": 100000}),
        json!({"action": "what_if", "state": latent(0.0), "candidates": ["a", "a"]}),
        json!({"action": "what_if", "state": latent(0.0), "candidates": (0..17).map(|i| format!("c{i}")).collect::<Vec<_>>()}),
        json!({"action": "audit", "state": latent(0.0)}),
        json!({"action": "audit", "state": latent(0.0), "target_action": "  "}),
    ];
    for case in cases {
        let out = run(&engine, case.clone()).await;
        assert!(out.is_error, "{case}");
        let rejection = out.rejection.as_ref().expect("typed rejection");
        assert_eq!(rejection.http_status, 400, "{case}");
    }
}

// ------------------------------------------------------------------- HTTP

#[tokio::test]
async fn http_simulate_what_if_and_audit_return_structured_json() {
    let engine = engine_with_stub().await;

    let (status, body) = post_json(
        &engine,
        "/v1/simulate",
        json!({"state": latent(0.0), "actions": ["a", "b"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "simulate");
    assert_eq!(body["result"]["meta"]["simulation"]["steps_simulated"], 2);
    assert!(body.get("error").is_none());

    let (status, body) = post_json(
        &engine,
        "/v1/what_if",
        json!({"state": latent(0.0), "candidates": ["a", "b"], "horizon": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "what_if");
    assert_eq!(
        body["result"]["meta"]["ranking"].as_array().unwrap().len(),
        2
    );

    // The REST field is `action`, as on the Python service.
    let (status, body) = post_json(
        &engine,
        "/v1/audit_action",
        json!({"state": latent(0.0), "action": "a", "horizon": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "audit");
    assert_eq!(
        body["result"]["meta"]["verdict"],
        "UNVERIFIED_UNTRAINED_DYNAMICS"
    );
}

#[tokio::test]
async fn http_refuses_bad_bodies_with_a_typed_400() {
    let engine = engine_without_bridge();
    let (status, body) = post_json(
        &engine,
        "/v1/simulate",
        json!({"state": [1.0], "actions": ["a"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "InvalidParams");
    assert_eq!(body["result"]["is_error"], true);

    // A body must not pick its own verb: the route does.
    for body in [
        json!({"state": latent(0.0), "actions": ["a"], "verb": "ask"}),
        json!({"state": latent(0.0), "actions": ["a"], "action": "ask"}),
    ] {
        let (status, resp) = post_json(&engine, "/v1/simulate", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
    }

    let (status, _) = post_json(
        &engine,
        "/v1/audit_action",
        json!({"state": latent(0.0), "action": "a", "target_action": "b"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = post_json(&engine, "/v1/what_if", json!([1, 2])).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn message_endpoint_reaches_the_same_verbs() {
    let engine = engine_without_bridge();
    let (status, body) = post_json(
        &engine,
        "/message",
        json!({"verb": "simulate", "state": latent(0.0), "actions": ["a"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "simulate");
}

// -------------------------------------------------------------------- MCP

async fn mcp(server: &McpServer, frame: Value) -> Value {
    let mut buf = frame.to_string().into_bytes();
    buf.resize(buf.len() + simd_json::SIMDJSON_PADDING, 0);
    serde_json::from_str(&server.handle_jsonrpc_frame(&mut buf).await).unwrap()
}

#[tokio::test]
async fn mcp_advertises_and_runs_the_new_verbs() {
    let server = McpServer::new();
    let listed = mcp(
        &server,
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    )
    .await;
    let tools = listed["result"]["tools"].as_array().unwrap();
    let zero = tools.iter().find(|t| t["name"] == "zero").unwrap();
    let verbs = zero["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap();
    for verb in ["simulate", "what_if", "audit"] {
        assert!(verbs.iter().any(|v| v == verb), "{verb} not advertised");
    }
    let props = zero["inputSchema"]["properties"].as_object().unwrap();
    assert_eq!(
        zero["inputSchema"]["properties"]["dynamics"]["enum"],
        json!(["residual", "symplectic", "contact"])
    );
    assert_eq!(
        zero["inputSchema"]["properties"]["damping"]["type"],
        json!("number")
    );
    for field in [
        "state",
        "mode",
        "latent",
        "return_trajectory",
        "actions",
        "target_action",
        "continuation_actions",
        "dynamics",
    ] {
        assert!(props.contains_key(field), "{field} not advertised");
    }
    assert!(
        tools.iter().any(|t| t["name"] == "causal_fold"),
        "the other tool stays listed"
    );

    let called = mcp(
        &server,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "zero",
            "arguments": {"action": "simulate", "state": latent(0.0), "actions": ["a"]},
        }}),
    )
    .await;
    assert_eq!(called["result"]["isError"], false, "{called}");
    assert_eq!(
        called["result"]["_meta"]["provenance"],
        "latent_residual_dynamics_untrained"
    );
}

// ---------------------------------------------------------- decide modes

/// What each planner mode returns for one request on the untrained prior.
async fn latent_decide(
    engine: &PolymorphicZeroEngine,
    mode: &str,
) -> gen_zero_service::ZeroToolOutcome {
    run(
        engine,
        json!({
            "action": "decide", "context": "pick a move", "mode": mode,
            "candidates": ["left", "right", "wait"], "latent": latent(0.0),
            "return_trajectory": true, "horizon": 3,
        }),
    )
    .await
}

#[tokio::test]
async fn every_latent_mode_chooses_a_candidate_and_passes_the_gate() {
    let engine = engine_with_stub().await;
    for mode in ["mcts", "mpc_cem", "astar"] {
        let out = latent_decide(&engine, mode).await;
        eprintln!(
            "{mode}: tier={} entropy={}",
            out.meta["tier"], out.meta["entropy"]
        );
        assert_eq!(out.verb, ZeroVerb::Ask);
        assert_eq!(out.meta["engine"], "latent_planner", "{mode}");
        assert_eq!(out.meta["planner_mode"], mode);
        assert_eq!(
            out.meta["mode"]["resolved"],
            format!("latent_planner:{mode}")
        );
        let chosen = out.meta["chosen_action"].as_str().unwrap();
        assert!(["left", "right", "wait"].contains(&chosen), "{chosen}");
        assert_eq!(out.meta["provenance"], "latent_residual_dynamics_untrained");
        // The pick went through the PolicyGate, and the outcome follows its
        // tier: only a Proceed is a success.
        assert!(out.meta["tier"].is_string(), "{mode}");
        assert_eq!(out.is_error, out.meta["tier"] != "Proceed", "{mode}");
        let trajectory = &out.meta["trajectory"];
        assert_eq!(trajectory["trajectory"][0]["action"], chosen, "{mode}");
        assert_eq!(trajectory["steps_simulated"], 3);
    }
}

/// A planner that is unsure escalates, it is not overridden. The MCTS visit
/// distribution over this untrained prior is close to uniform, and A* costs
/// are near-tied, so its Boltzmann entropy is high too: the gate holds both back.
#[tokio::test]
async fn the_gate_holds_back_a_planner_that_is_unsure() {
    let engine = engine_with_stub().await;
    for mode in ["astar", "mcts"] {
        let out = latent_decide(&engine, mode).await;
        assert!(out.is_error, "{mode}: {:?}", out.meta["tier"]);
        assert!(out.meta["entropy"].as_f64().unwrap() > 0.65, "{mode}");
        assert_eq!(out.meta["gate_status"], "requires_confirmation", "{mode}");
        assert_eq!(out.meta["requires_confirmation"], true, "{mode}");
        // The pick stays visible for whoever confirms it.
        assert!(out.meta["chosen_action"].is_string(), "{mode}");
    }
}

#[tokio::test]
async fn latent_modes_still_fail_closed_without_a_risk_verdict() {
    let engine = engine_without_bridge();
    let out = run(
        &engine,
        json!({
            "action": "decide", "context": "pick a move", "mode": "astar",
            "candidates": ["left", "right"], "latent": latent(0.0),
        }),
    )
    .await;
    assert!(out.is_error);
    assert_eq!(out.meta["gate_status"], "requires_confirmation");
}

#[tokio::test]
async fn mcts_on_text_is_the_semantic_lookahead_and_says_so() {
    let engine = engine_with_stub().await;
    let out = run(
        &engine,
        json!({
            "action": "decide", "context": "back up before deleting", "mode": "mcts",
            "candidates": ["delete", "backup", "wait"], "horizon": 2,
        }),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.verb, ZeroVerb::Imagine);
    assert_eq!(out.meta["planner"], "puct_mcts");
    assert_eq!(out.meta["mode"]["resolved"], "semantic_puct_lookahead");
}

#[tokio::test]
async fn auto_and_reflex_are_the_semantic_ask() {
    let engine = engine_with_stub().await;
    for mode in ["auto", "reflex"] {
        let out = run(
            &engine,
            json!({
                "action": "decide", "context": "pick a move", "mode": mode,
                "candidates": ["left", "right"], "return_trajectory": true,
            }),
        )
        .await;
        assert!(!out.is_error, "{mode}: {:?}", out.meta);
        assert_eq!(out.verb, ZeroVerb::Ask);
        assert_eq!(out.meta["engine"], "semantic_bridge");
        assert_eq!(out.meta["mode"]["resolved"], "semantic_ask");
        // No latent state, so no trajectory, and the answer says why.
        assert!(out.meta["trajectory_status"]
            .as_str()
            .unwrap()
            .starts_with("unsupported_without_latent_state"));
        assert!(out.meta.get("trajectory").is_none());
    }
}

#[tokio::test]
async fn decide_refuses_mode_and_latent_combinations_it_cannot_honour() {
    let engine = engine_with_stub().await;
    let base = |extra: Value| {
        let mut req = json!({"action": "decide", "context": "c", "candidates": ["a", "b"]});
        for (k, v) in extra.as_object().unwrap() {
            req[k] = v.clone();
        }
        req
    };
    let cases = [
        // No text encoder into the latent space: these modes need a latent.
        base(json!({"mode": "mpc_cem"})),
        base(json!({"mode": "astar"})),
        // A latent is read only by the planner modes.
        base(json!({"mode": "auto", "latent": latent(0.0)})),
        base(json!({"latent": latent(0.0)})),
        // Unknown mode, wrong types, wrong width, too many candidates.
        base(json!({"mode": "nonsense"})),
        base(json!({"mode": 3})),
        base(json!({"return_trajectory": "yes"})),
        base(json!({"mode": "astar", "latent": [1.0, 2.0]})),
        base(json!({"mode": "astar", "latent": latent(0.0), "horizon": 3})),
        // A field the chosen engine never reads is refused, not dropped.
        base(json!({"mode": "astar", "latent": latent(0.0), "cognitive": {"state": [0.0]}})),
        base(json!({"mode": "auto", "horizon": 5})),
        base(json!({"mode": "reflex", "horizon": 5})),
        base(json!({
            "mode": "astar", "latent": latent(0.0),
            "candidates": (0..17).map(|i| format!("c{i}")).collect::<Vec<_>>(),
        })),
    ];
    for case in cases {
        let out = run(&engine, case.clone()).await;
        assert!(
            out.is_error,
            "{}",
            case.to_string().chars().take(120).collect::<String>()
        );
        assert_eq!(
            out.rejection.as_ref().map(|r| r.http_status),
            Some(400),
            "{:?}",
            out.meta
        );
    }
    // Mode fields are read only by ask/decide.
    let out = run(
        &engine,
        json!({"action": "route", "intent": "x", "tools": ["t"], "mode": "astar"}),
    )
    .await;
    assert!(out.is_error);
    assert_eq!(out.rejection.unwrap().code, "InvalidParams");
}

// ------------------------------------------------------ symplectic dynamics

/// A non-uniform phase point, so q and p differ per coordinate.
fn phase_latent() -> Value {
    let v: Vec<f64> = (0..DIM).map(|i| 0.1 * ((i as f64) * 0.013).cos()).collect();
    json!(v)
}

fn f64_at(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// End-to-end through `zero`: ten Stormer-Verlet steps under one action keep
/// that action's Hamiltonian within 1e-4 (relative), the flow really moves
/// the state, and the full (q, p) trajectory comes back.
#[tokio::test]
async fn symplectic_simulate_conserves_energy_over_ten_steps() {
    let engine = engine_without_bridge();
    let plan = vec!["push"; 10];
    let out = run(
        &engine,
        json!({"action": "simulate", "dynamics": "symplectic",
               "state": phase_latent(), "actions": plan}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.meta["dynamics"], "symplectic");
    assert_eq!(
        out.meta["provenance"],
        "symplectic_hamiltonian_dynamics_untrained"
    );
    assert_eq!(out.meta["trained"], false);
    let sim = &out.meta["simulation"];
    assert_eq!(sim["steps_simulated"], 10);

    let ledger = &sim["energy_ledger"];
    assert_eq!(ledger["single_action"], true);
    let h0 = f64_at(&ledger["run"]["initial"]);
    let h10 = f64_at(&ledger["run"]["final"]);
    let rel = f64_at(&ledger["run"]["relative_drift"]);
    eprintln!(
        "H0={h0:.9} H10={h10:.9} rel_drift={rel:.3e} max_step_rel={:.3e}",
        f64_at(&ledger["max_relative_step_drift"])
    );
    assert!(h0 > 0.0);
    assert!(rel < 1e-4, "relative drift {rel}");
    assert!(((h10 - h0) / h0).abs() < 1e-4);
    assert!(f64_at(&ledger["max_relative_step_drift"]) < 1e-4);

    let steps = sim["trajectory"].as_array().unwrap();
    for (i, step) in steps.iter().enumerate() {
        let drift = f64_at(&step["energy_drift"]);
        let before = f64_at(&step["hamiltonian_before"]);
        assert!((drift / before).abs() < 1e-4, "step {i}: {drift}");
        assert!(step["q_norm"].is_number() && step["p_norm"].is_number());
    }
    // Consecutive steps chain: H after step n is H before step n+1.
    for pair in steps.windows(2) {
        assert_eq!(pair[0]["hamiltonian_after"], pair[1]["hamiltonian_before"]);
    }

    let phase = sim["phase_trajectory"].as_array().unwrap();
    assert_eq!(phase.len(), 10);
    for (i, point) in phase.iter().enumerate() {
        assert_eq!(point["step"], i + 1);
        assert_eq!(point["q"].as_array().unwrap().len(), DIM / 2);
        assert_eq!(point["p"].as_array().unwrap().len(), DIM / 2);
    }
    // The last phase point is the final state, and the flow moved it.
    let last = &phase[9];
    let final_state = sim["final_state"].as_array().unwrap();
    assert_eq!(
        &final_state[..DIM / 2],
        last["q"].as_array().unwrap().as_slice()
    );
    assert_eq!(
        &final_state[DIM / 2..],
        last["p"].as_array().unwrap().as_slice()
    );
    let start = phase_latent();
    let moved: f64 = start
        .as_array()
        .unwrap()
        .iter()
        .zip(final_state)
        .map(|(a, b)| (f64_at(a) - f64_at(b)).abs())
        .sum();
    assert!(moved > 1e-3, "state did not move: {moved}");
}

#[tokio::test]
async fn symplectic_and_residual_are_different_models() {
    let engine = engine_without_bridge();
    let sim = |dynamics: Option<&str>| {
        let engine = Arc::clone(&engine);
        let mut req = json!({"action": "simulate", "state": phase_latent(), "actions": ["a", "b"]});
        if let Some(d) = dynamics {
            req["dynamics"] = json!(d);
        }
        async move { run(&engine, req).await.meta }
    };
    let default = sim(None).await;
    let residual = sim(Some("residual")).await;
    let symplectic = sim(Some("symplectic")).await;
    assert_eq!(default["dynamics"], "residual");
    assert_eq!(default["simulation"], residual["simulation"]);
    assert!(residual["simulation"].get("energy_ledger").is_none());
    assert!(residual["simulation"].get("phase_trajectory").is_none());
    assert_ne!(
        residual["simulation"]["final_state"],
        symplectic["simulation"]["final_state"]
    );
    // Two actions, two Hamiltonians: no whole-run drift is claimed.
    let ledger = &symplectic["simulation"]["energy_ledger"];
    assert_eq!(ledger["single_action"], false);
    assert!(ledger["run"].is_null());
    assert!(f64_at(&ledger["max_relative_step_drift"]) < 1e-4);
}

#[tokio::test]
async fn symplectic_what_if_and_audit_report_their_dynamics() {
    let engine = engine_with_stub().await;
    let out = run(
        &engine,
        json!({"action": "what_if", "dynamics": "symplectic", "state": phase_latent(),
               "candidates": ["a", "b", "c"], "horizon": 4}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.meta["dynamics"], "symplectic");
    let outcomes = out.meta["outcomes"].as_array().unwrap();
    assert_eq!(outcomes.len(), 3);
    for o in outcomes {
        assert!(f64_at(&o["energy_ledger"]["max_relative_step_drift"]) < 1e-4);
        assert!(o.get("phase_trajectory").is_none());
    }
    let out = run(
        &engine,
        json!({"action": "audit", "dynamics": "symplectic", "state": phase_latent(),
               "target_action": "a", "horizon": 3}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.meta["dynamics"], "symplectic");
    assert_eq!(out.meta["verdict"], "UNVERIFIED_UNTRAINED_DYNAMICS");
    assert!(out.meta["rollout"]["energy_ledger"].is_object());
}

#[tokio::test]
async fn latent_decide_plans_and_rolls_on_symplectic_dynamics() {
    let engine = engine_with_stub().await;
    let out = run(
        &engine,
        json!({
            "action": "decide", "context": "pick a move", "mode": "astar",
            "candidates": ["left", "right", "wait"], "latent": phase_latent(),
            "return_trajectory": true, "horizon": 3, "dynamics": "symplectic",
        }),
    )
    .await;
    assert_eq!(out.meta["dynamics"], "symplectic", "{:?}", out.meta);
    assert_eq!(
        out.meta["provenance"],
        "symplectic_hamiltonian_dynamics_untrained"
    );
    assert_eq!(out.meta["trajectory"]["dynamics"], "symplectic");
    assert!(out.meta["trajectory"]["energy_ledger"].is_object());
}

#[tokio::test]
async fn http_routes_carry_dynamics_through() {
    let engine = engine_without_bridge();
    let (status, body) = post_json(
        &engine,
        "/v1/simulate",
        json!({"state": phase_latent(), "actions": vec!["push"; 10], "dynamics": "symplectic"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let meta = &body["result"]["meta"];
    assert_eq!(meta["dynamics"], "symplectic", "{body}");
    assert!(f64_at(&meta["simulation"]["energy_ledger"]["run"]["relative_drift"]) < 1e-4);
    assert_eq!(
        meta["simulation"]["phase_trajectory"]
            .as_array()
            .unwrap()
            .len(),
        10
    );
    let (status, body) = post_json(
        &engine,
        "/v1/what_if",
        json!({"state": phase_latent(), "candidates": ["a", "b"], "dynamics": "symplectic"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["meta"]["dynamics"], "symplectic");
    let (status, body) = post_json(
        &engine,
        "/v1/simulate",
        json!({"state": phase_latent(), "actions": ["a"], "dynamics": "conformal_x"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

// --------------------------------------------------------- contact dynamics

/// `contact` with `damping: 0` must be the symplectic flow: ten steps conserve
/// `H_a`, the per-pair volume factor is exactly one, and the final state matches
/// the symplectic run of the same plan.
#[tokio::test]
async fn contact_with_zero_damping_conserves_energy_and_matches_symplectic() {
    let engine = engine_with_stub().await;
    let plan = vec!["push"; 10];
    let contact = run(
        &engine,
        json!({"action": "simulate", "dynamics": "contact", "damping": 0.0,
               "state": phase_latent(), "actions": plan}),
    )
    .await;
    assert!(!contact.is_error, "{:?}", contact.rejection);
    assert_eq!(contact.meta["dynamics"], "contact");
    assert_eq!(contact.meta["damping"], 0.0);
    assert_eq!(
        contact.meta["provenance"],
        "conformal_symplectic_contact_dynamics_untrained"
    );
    let ledger = &contact.meta["simulation"]["energy_ledger"];
    assert_eq!(ledger["conservative"], true);
    assert!(ledger["conserved_quantity"].is_string(), "{ledger}");
    assert_eq!(ledger["phase_volume_factor_per_pair"], 1.0);
    assert_eq!(ledger["contraction_rate"], 1.0);
    assert_eq!(ledger["damping_gamma"], 0.0);
    assert_eq!(ledger["integrator_gamma"], 0.0);
    assert!(f64_at(&ledger["run"]["relative_drift"]) < 1e-4, "{ledger}");
    assert_eq!(contact.meta["simulation"]["steps_simulated"], 10);

    let symplectic = run(
        &engine,
        json!({"action": "simulate", "dynamics": "symplectic",
               "state": phase_latent(), "actions": plan}),
    )
    .await;
    assert!(!symplectic.is_error);
    let a = contact.meta["simulation"]["final_state"]
        .as_array()
        .unwrap();
    let b = symplectic.meta["simulation"]["final_state"]
        .as_array()
        .unwrap();
    assert_eq!(a.len(), DIM);
    for (x, y) in a.iter().zip(b) {
        assert!((f64_at(x) - f64_at(y)).abs() < 1e-6, "{x} vs {y}");
    }
}

/// `contact` with `damping > 0` dissipates: the reported per-pair volume factor
/// is `exp(-2 gamma dt)`, the run energy falls, and each step carries the
/// contact action and the factor.
#[tokio::test]
async fn contact_with_positive_damping_contracts_and_dissipates() {
    let engine = engine_with_stub().await;
    let gamma = 2.0_f64;
    let out = run(
        &engine,
        json!({"action": "simulate", "dynamics": "contact", "damping": gamma,
               "state": phase_latent(), "actions": vec!["push"; 20]}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.rejection);
    let sim = &out.meta["simulation"];
    let ledger = &sim["energy_ledger"];
    assert_eq!(ledger["conservative"], false);
    assert!(ledger["conserved_quantity"].is_null(), "{ledger}");
    let expected = (-2.0 * gamma * 0.01_f64).exp();
    let factor = f64_at(&ledger["phase_volume_factor_per_pair"]);
    assert!((factor - expected).abs() < 1e-6, "{factor} vs {expected}");
    let projection = f64_at(&ledger["phase_volume_factor_projection"]);
    assert!(
        (projection - (-2.0 * gamma * 512.0 * 0.01_f64).exp()).abs() < 1e-6,
        "{projection}"
    );
    assert!(f64_at(&ledger["contraction_rate"]) < 1.0, "{ledger}");
    assert_eq!(ledger["damping_gamma"], gamma);
    assert_eq!(ledger["integrator_gamma"], 2.0 * gamma);
    assert!(f64_at(&ledger["dissipated_energy"]) > 0.0, "{ledger}");
    let run_entry = &ledger["run"];
    assert!(
        f64_at(&run_entry["final"]) < f64_at(&run_entry["initial"]),
        "{run_entry}"
    );
    let step = &sim["trajectory"][0];
    assert!(step["contact_action"].is_number(), "{step}");
    assert!((f64_at(&step["phase_volume_factor_per_pair"]) - expected).abs() < 1e-6);
    assert!(step["hamiltonian_before"].is_number() && step["hamiltonian_after"].is_number());
}

#[tokio::test]
async fn contact_what_if_audit_and_latent_decide_run_on_the_contact_prior() {
    let engine = engine_with_stub().await;
    let out = run(
        &engine,
        json!({"action": "what_if", "dynamics": "contact", "state": phase_latent(),
               "candidates": ["a", "b"], "horizon": 3}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.rejection);
    assert_eq!(out.meta["dynamics"], "contact");
    assert_eq!(out.meta["damping"], 0.5);
    assert_eq!(
        out.meta["outcomes"][0]["energy_ledger"]["damping_gamma"],
        0.5
    );

    let out = run(
        &engine,
        json!({"action": "audit", "dynamics": "contact", "damping": 1.0, "state": phase_latent(),
               "target_action": "a", "horizon": 3}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.rejection);
    assert_eq!(out.meta["dynamics"], "contact");
    assert_eq!(out.meta["damping"], 1.0);
    assert_eq!(out.meta["rollout"]["energy_ledger"]["damping_gamma"], 1.0);
    assert_ne!(out.meta["verdict"], "APPROVED");

    for mode in ["mcts", "mpc_cem", "astar"] {
        let out = run(
            &engine,
            json!({"action": "decide", "context": "c", "candidates": ["a", "b"], "mode": mode,
                   "latent": phase_latent(), "return_trajectory": true, "horizon": 3,
                   "dynamics": "contact", "damping": 0.25}),
        )
        .await;
        // The policy gate may hold the pick for confirmation; that is a gated
        // outcome, not a rejection. A refused request would carry a rejection.
        assert!(out.rejection.is_none(), "{mode}: {:?}", out.rejection);
        assert!(out.meta["planner"].is_string(), "{mode}: {:?}", out.meta);
        assert_eq!(out.meta["dynamics"], "contact", "{mode}");
        assert_eq!(out.meta["damping"], 0.25, "{mode}");
        assert_eq!(
            out.meta["provenance"],
            "conformal_symplectic_contact_dynamics_untrained"
        );
        assert_eq!(out.meta["trajectory"]["dynamics"], "contact", "{mode}");
        assert_eq!(out.meta["trajectory"]["damping"], 0.25, "{mode}");
        assert_eq!(
            out.meta["trajectory"]["energy_ledger"]["damping_gamma"], 0.25,
            "{mode}"
        );
    }
}

/// `damping` is fail-closed: negative, non-finite or non-numeric values are
/// refused, and the field is refused with any dynamics other than `contact`.
#[tokio::test]
async fn damping_is_refused_when_invalid_or_without_contact() {
    let engine = engine_with_stub().await;
    let cases = [
        json!({"dynamics": "contact", "damping": -0.5}),
        json!({"dynamics": "contact", "damping": "0.5"}),
        json!({"dynamics": "contact", "damping": null}),
        json!({"dynamics": "contact", "damping": 1e300}),
        json!({"dynamics": "contact", "damping": 1e-50}),
        json!({"dynamics": "symplectic", "damping": 0.5}),
        json!({"dynamics": "residual", "damping": 0.0}),
        json!({"damping": 0.5}),
    ];
    for case in cases {
        let mut req = json!({"action": "simulate", "state": phase_latent(), "actions": ["a"]});
        for (k, v) in case.as_object().unwrap() {
            req[k] = v.clone();
        }
        let out = run(&engine, req).await;
        assert!(out.is_error, "{case} was accepted");
        let rejection = out.rejection.as_ref().unwrap();
        assert_eq!(rejection.code, "InvalidParams", "{case}");
        assert!(
            rejection.detail.contains("damping"),
            "{case}: {}",
            rejection.detail
        );
    }
    // The HTTP route refuses too.
    let (status, body) = post_json(
        &engine,
        "/v1/simulate",
        json!({"state": phase_latent(), "actions": ["a"], "dynamics": "contact", "damping": -1}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn unknown_or_unread_dynamics_is_refused_not_defaulted() {
    let engine = engine_with_stub().await;
    for bad in [
        json!("conformal_x"),
        json!("Symplectic"),
        json!(1),
        json!(null),
    ] {
        let out = run(
            &engine,
            json!({"action": "simulate", "state": phase_latent(), "actions": ["a"], "dynamics": bad}),
        )
        .await;
        assert!(out.is_error, "{bad} was accepted");
        let rejection = out.rejection.as_ref().unwrap();
        assert_eq!(rejection.code, "InvalidParams");
        assert!(
            rejection.detail.contains("dynamics"),
            "{}",
            rejection.detail
        );
        assert!(out.meta["provenance"].is_null(), "{:?}", out.meta);
    }
    // Verbs and modes that have no latent dynamics refuse the field, and refuse
    // `damping` the same way: neither is ever dropped silently.
    let cases = [
        json!({"action": "route", "intent": "x", "tools": ["t"], "dynamics": "symplectic"}),
        json!({"action": "route", "intent": "x", "tools": ["t"], "damping": 0.5}),
        json!({"action": "imagine", "state": "s", "candidates": ["a", "b"], "damping": 0.5}),
        json!({"action": "decide", "context": "c", "candidates": ["a", "b"],
               "dynamics": "symplectic"}),
        json!({"action": "decide", "context": "c", "candidates": ["a", "b"], "damping": 0.5}),
        json!({"action": "decide", "context": "c", "candidates": ["a", "b"], "mode": "mcts",
               "dynamics": "symplectic"}),
        json!({"action": "decide", "context": "c", "candidates": ["a", "b"], "mode": "mcts",
               "damping": 0.5}),
    ];
    for case in cases {
        let out = run(&engine, case.clone()).await;
        assert!(out.is_error, "{case}");
        assert_eq!(out.rejection.unwrap().code, "InvalidParams", "{case}");
    }
}

/// A proceeding `ask` on a caller-supplied micro-core lands in the one audit ledger that
/// `/audit/ledger` serves, and its inclusion proof verifies against the reported root.
#[tokio::test]
async fn caller_nanocore_ask_is_audited_in_the_shared_ledger() {
    let e = engine_with_stub().await;
    let core = gen_zero_nanocore::core_type::fixtures::synthetic_core(
        gen_zero_nanocore::DOMAIN_GENERAL,
        "caller core",
        gen_zero_core::CompressedLatent { values: [1.0; 128] },
        4,
        0.5,
        &["alpha"],
    );
    for (index, verb) in ["ask", "decide"].into_iter().enumerate() {
        let out = e
            .execute(
                &json!({"verb": verb, "context": "pick", "candidates": ["alpha"],
            "engine": "nanocore", "nanocore_core": core, "decision_state": vec![1.0_f32; 128]}),
            )
            .await
            .unwrap();
        assert!(!out.is_error, "{out:?}");
        assert_eq!(out.meta["tier"], "Proceed");
        let (count, root) = e.audit_ledger();
        assert_eq!(count, index + 1);
        assert_eq!(out.meta["decision_audit"]["leaf_index"], index);
        let hex: String = root.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(out.meta["decision_audit"]["root"], hex);
        let proof = e.audit_proof(index as u64).unwrap();
        assert!(e.verify_audit_proof(&proof, &root));
        let mut forged = proof.clone();
        forged.leaf_count = 1;
        forged.leaf_index = 0;
        forged.leaf_hash = root;
        forged.siblings.clear();
        if count > 1 {
            assert!(!e.verify_audit_proof(&forged, &root));
        }
    }
}
