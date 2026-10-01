//! End-to-end: the `pipeline` verb over HTTP (`POST /v1/pipeline/{op}`, `zero` via
//! `/message`) and MCP (`tools/list`, `tools/call` for `zero` and `pipeline`).
//! The engine's default world model is the real `LatentDynamicsWorldModel`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gen_zero_gate::{LinearConstraint, PolicyGate, RuleId};
use gen_zero_lod::{EpistemicStatus, LodBand, LodGraph, LodNode, MixedCurvatureCoord};
use gen_zero_service::{McpServer, PolymorphicZeroEngine};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

fn engine() -> Arc<PolymorphicZeroEngine> {
    Arc::new(PolymorphicZeroEngine::new().with_semantic(None))
}

#[tokio::test]
async fn live_graph_revocation_blocks_http_pipeline_action() {
    let graph = Arc::new(LodGraph::new());
    graph
        .add_node(
            LodNode::new(
                0,
                LodBand::Lod0Atomic,
                MixedCurvatureCoord::origin(),
                "revoked action",
                2,
            )
            .with_status(EpistemicStatus::Falsified),
        )
        .unwrap();
    let engine = Arc::new(
        PolymorphicZeroEngine::new()
            .with_semantic(None)
            .with_lod_graph(graph),
    );
    let (status, body) = post(
        &engine,
        "/v1/pipeline/decide",
        json!({"state": zeros(), "candidates": [2], "mode": "reflex", "entropy": 0.0}),
    )
    .await;
    assert_ne!(status, StatusCode::OK, "{body}");
    assert!(
        body.to_string().contains("NoFeasibleAction") || body.to_string().contains("no feasible"),
        "{body}"
    );
}

fn engine_prohibiting(ids: &[u32]) -> Arc<PolymorphicZeroEngine> {
    let mut gate = PolicyGate::default();
    for &id in ids {
        gate.add_constraint(LinearConstraint::prohibit(
            RuleId(700 + id),
            "test_prohibit",
            gen_zero_core::ActionId(id),
        ));
    }
    Arc::new(
        PolymorphicZeroEngine::new()
            .with_semantic(None)
            .with_gate(gate),
    )
}

async fn post(engine: &Arc<PolymorphicZeroEngine>, path: &str, body: Value) -> (StatusCode, Value) {
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
    let bytes = axum::body::to_bytes(resp.into_body(), 64 << 20)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

fn astar_goal() -> Value {
    use gen_zero_core::{ActionId, FullLatent, WorldModelDynamics};
    let (target, _, _) = gen_zero_worldmodel::LatentDynamicsWorldModel::default()
        .step(&FullLatent::zeros(), ActionId(1))
        .unwrap();
    json!({"state": target.as_slice(), "tolerance": 0.00001})
}

fn zeros() -> Value {
    json!(vec![0.0_f32; 1024])
}

/// Same geometry as the planner's trap test: action 0 crosses the termination
/// norm, action 18 stays inside.
fn trap_state() -> Value {
    let u: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin()).collect();
    let norm = u.iter().map(|x| x * x).sum::<f32>().sqrt();
    let c = (101.0 / norm - 0.05) / 0.95;
    json!(u.iter().map(|x| x * c).collect::<Vec<f32>>())
}

fn meta_pipeline(body: &Value) -> &Value {
    &body["result"]["meta"]["pipeline"]
}

// -------------------------------------------------------------- HTTP /v1/pipeline/{op}

#[tokio::test]
async fn http_simulate_returns_the_full_trajectory() {
    let (status, body) = post(
        &engine(),
        "/v1/pipeline/simulate",
        json!({"state": zeros(), "actions": [1, 2, 3]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "pipeline");
    let sim = &meta_pipeline(&body)["simulation"];
    assert_eq!(sim["steps_simulated"], 3);
    assert_eq!(sim["survival_horizon"], 3);
    assert_eq!(sim["terminated_early"], false);
    assert_eq!(sim["safety_calibrated"], false);
    assert_eq!(
        sim["safety_sources"],
        json!(["latent_norm_boundary_margin"])
    );
    let steps = sim["trajectory"].as_array().unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0]["state"].as_array().unwrap().len(), 1024);
    assert!(steps[0]["reward"].is_number() && steps[0]["safe_prob"].is_number());
    assert_eq!(steps[2]["action"], 3);
}

#[tokio::test]
async fn http_what_if_names_the_trap_and_the_escape() {
    let (status, body) = post(
        &engine(),
        "/v1/pipeline/what_if",
        json!({"state": trap_state(), "candidates": [0, 18], "horizon": 5}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let w = &meta_pipeline(&body)["what_if"];
    assert_eq!(w["best_candidate"], 18);
    assert_eq!(w["traps_detected"], json!([0]));
    assert_eq!(w["candidate_outcomes"][0]["survival_horizon"], 0);
    assert_eq!(w["candidate_outcomes"][1]["survival_horizon"], 5);
}

#[tokio::test]
async fn http_audit_gives_all_three_verdicts() {
    let e = engine();
    let (_, approved) = post(
        &e,
        "/v1/pipeline/audit_action",
        json!({"state": zeros(), "action": 1}),
    )
    .await;
    assert_eq!(
        meta_pipeline(&approved)["audit"]["verdict"],
        "Approved",
        "{approved}"
    );
    // Approved on an uncalibrated margin must say so next to the verdict.
    assert_eq!(
        meta_pipeline(&approved)["audit"]["safety_calibrated"],
        false
    );

    let (_, warn) = post(
        &e,
        "/v1/pipeline/audit_action",
        json!({"state": trap_state(), "action": 18}),
    )
    .await;
    assert_eq!(
        meta_pipeline(&warn)["audit"]["verdict"],
        "WarnHazard",
        "{warn}"
    );

    let (status, lethal) = post(
        &e,
        "/v1/pipeline/audit_action",
        json!({"state": trap_state(), "action": 0, "horizon": 3}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let audit = &meta_pipeline(&lethal)["audit"];
    assert_eq!(audit["verdict"], "RejectLethal");
    assert_eq!(audit["first_hazard_step"], 1);
    assert_eq!(audit["risk_score"], 1.0);
}

#[tokio::test]
async fn http_decide_runs_each_mode_and_returns_a_trajectory() {
    let e = engine();
    for (mode, engine_name) in [
        ("auto", "DynamicKMoERouter"),
        ("mcts", "MctsEngine"),
        ("mpc_cem", "MpcCemEngine"),
        ("astar", "AStarEngine"),
        ("manifold_gflownet", "ManifoldGFlowNetEngine"),
        ("cfr_nash", "CfrNashEngine"),
        ("reflex", "CpSatFormalEngine"),
    ] {
        let (status, body) = post(
            &e,
            "/v1/pipeline/decide",
            json!({"state": zeros(), "candidates": [1, 2, 3], "mode": mode, "entropy": 0.5, "astar_goal": astar_goal(),
                   "return_trajectory": true, "horizon": 4}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{mode}: {body}");
        let d = &meta_pipeline(&body)["decision"];
        assert_eq!(d["engine"], engine_name);
        assert_eq!(d["mode"], mode);
        assert_eq!(d["trajectory"]["steps_simulated"], 4);
        assert_eq!(d["trajectory"]["trajectory"][0]["action"], d["action"]);
    }
}

/// `planner_config.router_entropy_threshold_high` reaches the router the same
/// request would otherwise use with its shipped default (0.70). At
/// `entropy: 0.5` the default keeps `auto` in `K2Pipeline`; lowering the high
/// threshold to 0.3 pushes that same entropy into `K3Committee` instead. This
/// is threaded through the real HTTP body, so it also proves `planner_config`
/// is not merely deserialized and dropped.
#[tokio::test]
async fn http_decide_planner_config_changes_auto_routing_tier() {
    let e = engine();
    let base = json!({"state": zeros(), "candidates": [1, 2, 3], "mode": "auto", "entropy": 0.5, "astar_goal": astar_goal()});

    let (status, body) = post(&e, "/v1/pipeline/decide", base.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        meta_pipeline(&body)["decision"]["routing_tier"],
        "K2Pipeline"
    );

    let mut with_config = base;
    with_config["planner_config"] = json!({"router_entropy_threshold_high": 0.3});
    let (status, body) = post(&e, "/v1/pipeline/decide", with_config).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        meta_pipeline(&body)["decision"]["routing_tier"],
        "K3Committee"
    );
}

/// A `planner_config` value outside its valid range is refused before any
/// engine runs, not clamped or silently ignored.
#[tokio::test]
async fn http_decide_refuses_an_invalid_planner_config() {
    let (status, body) = post(
        &engine(),
        "/v1/pipeline/decide",
        json!({"state": zeros(), "candidates": [1, 2], "mode": "mcts", "entropy": 0.5,
               "planner_config": {"mcts_c_puct": -1.0}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "InvalidParams");
}

/// `simulate`, `what_if` and `audit_action` never call an engine, so
/// `planner_config` has nothing to change there. Rather than accept and
/// silently drop it, those ops refuse it as an unknown field, same as any
/// other op/field mismatch this endpoint already enforces.
#[tokio::test]
async fn http_simulate_refuses_a_planner_config_it_cannot_use() {
    let (status, body) = post(
        &engine(),
        "/v1/pipeline/simulate",
        json!({"state": zeros(), "actions": [1], "planner_config": {"mcts_c_puct": 2.0}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "InvalidParams");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("planner_config"));
}

#[tokio::test]
async fn decisions_append_to_shared_audit_ledger_and_expose_proof() {
    let e = engine();
    let (status, body) = post(
        &e,
        "/v1/pipeline/decide",
        json!({"state": zeros(), "candidates": [1,2], "mode": "reflex", "entropy": 0.5}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (count, root) = e.audit_ledger();
    assert_eq!(count, 1);
    assert_eq!(body["result"]["meta"]["audit_ledger"]["leaf_index"], 0);
    assert_eq!(
        body["result"]["meta"]["audit_ledger"]["root"],
        root.iter().map(|b| format!("{b:02x}")).collect::<String>()
    );
    let (status, ledger) = post_get(&e, "/audit/ledger").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ledger["leaf_count"], 1);
    let (status, proof) = post_get(&e, "/audit/ledger/0").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(proof["root"], ledger["root"]);
    let inclusion = e.audit_proof(0).unwrap();
    assert_eq!(inclusion.mmr_root, root);
    assert!(e.verify_audit_proof(&inclusion, &root));
    assert!(!e.verify_audit_proof(&inclusion, &[0; 32]));
}

#[tokio::test]
async fn escalated_pipeline_decision_fails_closed_and_audits_tier_two() {
    let e = engine();
    let (status, body) = post(
        &e,
        "/v1/pipeline/decide",
        json!({"state": zeros(), "candidates": [1], "mode": "reflex", "entropy": 0.9}),
    )
    .await;
    assert_ne!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["is_error"], true);
    let meta = &body["result"]["meta"];
    assert_eq!(meta["requires_confirmation"], true);
    assert_eq!(meta["tier"], "Escalate");
    assert_eq!(meta["gate_status"], "requires_confirmation");
    assert_eq!(meta["pipeline"]["decision"]["gate_tier"], "Tier2Escalate");
    assert_eq!(meta["audit_ledger"]["tier"], "Escalate");
    assert_eq!(meta["audit_ledger"]["gate_status"], "requires_confirmation");
    assert_eq!(e.audit_ledger().0, 1);
}

async fn post_get(engine: &Arc<PolymorphicZeroEngine>, path: &str) -> (StatusCode, Value) {
    let response = McpServer::build_router(Arc::clone(engine), None)
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn http_decide_prunes_gated_actions_and_refuses_when_none_remain() {
    let (status, body) = post(
        &engine_prohibiting(&[3]),
        "/v1/pipeline/decide",
        json!({"state": zeros(), "candidates": [1, 2, 3], "mode": "mcts", "entropy": 0.5}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let d = &meta_pipeline(&body)["decision"];
    assert_ne!(d["action"], 3);
    assert_eq!(d["pruned"][0]["action"], 3);
    assert_eq!(d["pruned"][0]["violated_rules"], json!([703]));
    assert_eq!(d["pruned"][0]["tier"], "Tier3HardStop");

    let (status, body) = post(
        &engine_prohibiting(&[1, 2, 3]),
        "/v1/pipeline/decide",
        json!({"state": zeros(), "candidates": [1, 2, 3], "mode": "auto", "entropy": 0.5}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "NoFeasibleAction");
}

#[tokio::test]
async fn http_refuses_malformed_or_divergent_requests() {
    let e = engine();
    let cases = [
        (
            "/v1/pipeline/simulate",
            json!({"state": vec![0.0; 1023], "actions": [1]}),
            400,
            "InvalidParams",
        ),
        (
            "/v1/pipeline/simulate",
            json!({"state": zeros(), "actions": [1], "horizn": 1}),
            400,
            "InvalidParams",
        ),
        (
            "/v1/pipeline/simulate",
            json!({"state": zeros(), "actions": [1], "horizon": 2}),
            400,
            "InvalidParams",
        ),
        (
            "/v1/pipeline/simulate",
            json!({"state": vec![1e30; 1024], "actions": [1]}),
            422,
            "DivergentState",
        ),
        (
            "/v1/pipeline/decide",
            json!({"state": zeros(), "candidates": [1], "mode": "auto"}),
            400,
            "InvalidParams",
        ),
        (
            "/v1/pipeline/decide",
            json!({"state": zeros(), "candidates": [1], "mode": "greedy", "entropy": 0.5}),
            400,
            "InvalidParams",
        ),
        (
            "/v1/pipeline/explode",
            json!({"state": zeros()}),
            400,
            "InvalidParams",
        ),
    ];
    for (path, req, want_status, want_code) in cases {
        let (status, body) = post(&e, path, req.clone()).await;
        assert_eq!(status.as_u16(), want_status, "{path} {req}: {body}");
        assert_eq!(body["error"]["code"], want_code, "{path}: {body}");
        assert_eq!(body["result"]["is_error"], true);
    }

    // Body op must agree with the path.
    let (status, body) = post(
        &e,
        "/v1/pipeline/simulate",
        json!({"op": "decide", "state": zeros(), "actions": [1]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

// -------------------------------------------------------------- zero via /message

#[tokio::test]
async fn zero_routes_a_pipeline_block_to_the_pipeline_verb() {
    let (status, body) = post(
        &engine(),
        "/message",
        json!({"pipeline": {"op": "simulate", "state": zeros(), "actions": [1]}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "pipeline");
    assert_eq!(meta_pipeline(&body)["simulation"]["steps_simulated"], 1);
}

#[tokio::test]
async fn zero_refuses_a_pipeline_block_under_another_verb() {
    let (status, body) = post(
        &engine(),
        "/message",
        json!({"action": "ask", "pipeline": {"op": "simulate"}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("only accepted by pipeline"));
}

// -------------------------------------------------------------- MCP

async fn frame(server: &McpServer, req: Value) -> Value {
    let mut buf = req.to_string().into_bytes();
    buf.resize(buf.len() + simd_json::SIMDJSON_PADDING, 0);
    serde_json::from_str(&server.handle_jsonrpc_frame(&mut buf).await).unwrap()
}

#[tokio::test]
async fn mcp_lists_and_calls_the_pipeline_tool() {
    let server = McpServer::new();
    let list = frame(
        &server,
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    )
    .await;
    let tools = list["result"]["tools"].as_array().unwrap();
    let pipeline = tools.iter().find(|t| t["name"] == "pipeline").unwrap();
    let modes = pipeline["inputSchema"]["properties"]["mode"]["enum"]
        .as_array()
        .unwrap();
    for mode in ["manifold_gflownet", "cfr_nash"] {
        assert!(modes.contains(&json!(mode)));
    }
    let zero = tools.iter().find(|t| t["name"] == "zero").unwrap();
    assert!(zero["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap()
        .contains(&json!("pipeline")));

    let call = frame(
        &server,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "pipeline",
            "arguments": {"op": "audit_action", "state": zeros(), "action": 1, "horizon": 2},
        }}),
    )
    .await;
    assert_eq!(call["result"]["isError"], false, "{call}");
    assert_eq!(
        call["result"]["_meta"]["pipeline"]["audit"]["verdict"],
        "Approved"
    );

    let via_zero = frame(
        &server,
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "zero",
            "arguments": {"action": "pipeline", "pipeline":
                {"op": "what_if", "state": zeros(), "candidates": [1, 2], "horizon": 2}},
        }}),
    )
    .await;
    assert_eq!(via_zero["result"]["isError"], false, "{via_zero}");
    assert!(via_zero["result"]["_meta"]["pipeline"]["what_if"]["best_candidate"].is_u64());
}

#[tokio::test]
async fn http_decide_refuses_unknown_planner_config_fields() {
    for field in ["mcts_c_puct_typo", "virtual_loss", "arena_capacity"] {
        let (status, body) = post(
            &engine(),
            "/v1/pipeline/decide",
            json!({"state": zeros(), "candidates": [1, 2], "mode": "mcts", "entropy": 0.5,
                   "planner_config": {"mcts_c_puct": 2.0, (field): -1}}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "InvalidParams", "{body}");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("unknown field") && message.contains(field),
            "{body}"
        );
    }
}

#[tokio::test]
async fn http_decide_preserves_active_context_and_rejects_malformed_context() {
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::mutex(
        RuleId(801),
        "mutex",
        gen_zero_core::ActionId(1),
        gen_zero_core::ActionId(2),
    ));
    let engine = Arc::new(
        PolymorphicZeroEngine::new()
            .with_semantic(None)
            .with_gate(gate),
    );
    let (status, body) = post(
        &engine,
        "/v1/pipeline/decide",
        json!({
            "state": zeros(), "candidates": [2, 3], "active_context": [1],
            "mode": "reflex", "entropy": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let d = &meta_pipeline(&body)["decision"];
    assert_eq!(d["action"], 3);
    assert_eq!(d["pruned"][0]["violated_rules"], json!([801]));
    assert_eq!(d["hazard_detected"], false);
    for context in [Value::Null, json!("1"), json!([-1])] {
        let (status, body) = post(
            &engine,
            "/v1/pipeline/decide",
            json!({
                "state": zeros(), "candidates": [2, 3], "active_context": context,
                "mode": "reflex", "entropy": 0.0
            }),
        )
        .await;
        assert_ne!(status, StatusCode::OK, "{body}");
    }
}

#[tokio::test]
async fn http_decide_requires_explicit_astar_goal_and_forwards_budget() {
    let e = engine();
    let (status, body) = post(
        &e,
        "/v1/pipeline/decide",
        json!({
            "state": zeros(), "candidates": [1, 2], "mode": "astar", "entropy": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.to_string().contains("explicit search goal"), "{body}");
    let (status, body) = post(
        &e,
        "/v1/pipeline/decide",
        json!({
            "state": zeros(), "candidates": [1, 2], "mode": "reflex", "entropy": 0.0,
            "budget_ms": 0.0
        }),
    )
    .await;
    assert_ne!(status, StatusCode::OK, "{body}");
    assert!(
        body.to_string().contains("Timeout") || body.to_string().contains("timeout"),
        "{body}"
    );
    let (status, body) = post(
        &e,
        "/v1/pipeline/decide",
        json!({
            "state": zeros(), "candidates": [1, 2], "mode": "reflex", "entropy": 0.0,
            "budget_ms": "2"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn reflection_closes_http_simulation_to_policy_gate_loop() {
    use gen_zero_core::GraphFactProvider;
    use gen_zero_lod::EdgeType;
    let graph = Arc::new(LodGraph::new());
    let action = graph
        .add_node(
            LodNode::new(
                0,
                LodBand::Lod0Atomic,
                MixedCurvatureCoord::origin(),
                "action zero",
                0,
            )
            .with_prior(0.9),
        )
        .unwrap();
    let engine = Arc::new(
        PolymorphicZeroEngine::new()
            .with_bridge(None)
            .with_lod_graph(graph.clone()),
    );
    assert!(!graph.is_revoked(0));
    let (status, before) = post(
        &engine,
        "/v1/pipeline/decide",
        json!({"state": zeros(), "candidates": [0], "mode": "reflex", "entropy": 0.0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{before}");
    let (status, body) = post(
        &engine,
        "/v1/pipeline/simulate",
        json!({"state": trap_state(), "actions": [0], "auto_reflect": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let result = meta_pipeline(&body);
    assert_eq!(result["simulation"]["terminated_early"], true);
    let reflection = &result["graph_reflection"]["observations"][0];
    assert_eq!(reflection["evolution"]["converged"], true);
    assert_eq!(reflection["revoked_entities"], json!([0]));
    let evidence = reflection["evidence_node_id"].as_u64().unwrap() as u32;
    let node = graph.get_node(evidence).unwrap();
    assert!(node.timestamp_ns > 0);
    assert!(node.payload.unwrap().contains("model diagnostic"));
    graph.flush_edges_to_csr().unwrap();
    assert!(graph
        .csr_snapshot()
        .neighbors(evidence)
        .any(|(to, ty, _)| to == action && ty == EdgeType::Falsifies));
    assert!(graph.get_node(action).unwrap().confidence < 0.3);
    assert!(graph.is_revoked(0));
    let (status, after) = post(
        &engine,
        "/v1/pipeline/decide",
        json!({"state": zeros(), "candidates": [0, 18], "mode": "reflex", "entropy": 0.0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{after}");
    let decision = &meta_pipeline(&after)["decision"];
    assert_eq!(decision["action"], 18);
    assert_eq!(decision["pruned"][0]["action"], 0);
    let (status, rejected) = post(
        &engine,
        "/v1/pipeline/simulate",
        json!({"state": zeros(), "actions": [0]}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{rejected}");
}

#[tokio::test]
async fn reflection_what_if_and_audit_use_real_dynamics() {
    use gen_zero_core::GraphFactProvider;
    for (op, fields) in [
        ("what_if", json!({"candidates": [0, 18]})),
        ("audit_action", json!({"action": 0})),
    ] {
        let graph = Arc::new(LodGraph::new());
        let engine = Arc::new(
            PolymorphicZeroEngine::new()
                .with_bridge(None)
                .with_lod_graph(graph.clone()),
        );
        let mut request = fields;
        request["state"] = trap_state();
        request["horizon"] = json!(1);
        request["auto_reflect"] = json!(true);
        let (status, body) = post(&engine, &format!("/v1/pipeline/{op}"), request).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(graph.is_revoked(0), "{body}");
        assert!(
            !graph.is_revoked(18),
            "safe candidate was incorrectly revoked: {body}"
        );
        assert_eq!(
            meta_pipeline(&body)["graph_reflection"]["observations"][0]["evolution"]["converged"],
            true
        );
    }
}

#[tokio::test]
async fn hierarchical_prior_blocks_pending_falsified_successors_and_injects_macro() {
    use gen_zero_lod::EdgeType;
    let graph = Arc::new(LodGraph::new());
    let add = |entity, band, prior, status| {
        graph
            .add_node(
                LodNode::new(
                    0,
                    band,
                    MixedCurvatureCoord::origin(),
                    format!("node {entity}"),
                    entity,
                )
                .with_prior(prior)
                .with_status(status),
            )
            .unwrap()
    };
    let bad = add(0, LodBand::Lod0Atomic, 0.9, EpistemicStatus::Validated);
    let effect = add(101, LodBand::Lod0Atomic, 0.5, EpistemicStatus::Falsified);
    let good = add(18, LodBand::Lod0Atomic, 0.9, EpistemicStatus::Validated);
    let macro_node = add(102, LodBand::Lod2Milestone, 0.9, EpistemicStatus::Validated);
    add(19, LodBand::Lod0Atomic, 0.2, EpistemicStatus::Hypothesized);
    graph
        .add_edge(bad, effect, EdgeType::CausalTransition, 1.0)
        .unwrap();
    graph
        .add_edge(good, macro_node, EdgeType::CoarseGrain, 1.0)
        .unwrap();
    let engine = Arc::new(
        PolymorphicZeroEngine::new()
            .with_bridge(None)
            .with_lod_graph(graph),
    );
    let request =
        json!({"state": zeros(), "candidates": [0, 18, 19], "mode": "reflex", "entropy": 0.0});
    let (status, body) = post(&engine, "/v1/pipeline/decide", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let decision = &meta_pipeline(&body)["decision"];
    assert_eq!(decision["action"], 18);
    assert_eq!(decision["pruned"].as_array().unwrap().len(), 2);
    assert_eq!(decision["graph_context"]["facts"][0]["entity_id"], 102);
    assert_eq!(
        decision["graph_context"]["facts"][0]["band"],
        "Lod2Milestone"
    );
    let (status, repeated) = post(&engine, "/v1/pipeline/decide", request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(decision, &meta_pipeline(&repeated)["decision"]);
}

#[tokio::test]
async fn reflection_opt_in_is_strict_and_default_has_no_mutation() {
    let graph = Arc::new(LodGraph::new());
    let engine = Arc::new(
        PolymorphicZeroEngine::new()
            .with_bridge(None)
            .with_lod_graph(graph.clone()),
    );
    let (status, body) = post(
        &engine,
        "/v1/pipeline/simulate",
        json!({"state": trap_state(), "actions": [0]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(meta_pipeline(&body)["graph_reflection"]["enabled"], false);
    assert_eq!(graph.node_count(), 0);
    let (status, body) = post(
        &engine,
        "/v1/pipeline/simulate",
        json!({"state": trap_state(), "actions": [0], "auto_reflect": "true"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(graph.node_count(), 0);
}

#[tokio::test]
async fn policy_audit_reflection_is_idempotent_and_evolution_conflict_is_explicit() {
    use gen_zero_core::{ActionId, GraphFactProvider};
    let graph = Arc::new(LodGraph::new());
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::prohibit(
        RuleId(777),
        "prohibited",
        ActionId(7),
    ));
    let engine = Arc::new(
        PolymorphicZeroEngine::new()
            .with_bridge(None)
            .with_gate(gate)
            .with_lod_graph(graph.clone()),
    );
    let request = json!({"state": zeros(), "action": 7, "auto_reflect": true});
    let (status, body) = post(&engine, "/v1/pipeline/audit_action", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(graph.is_revoked(7));
    assert_eq!(meta_pipeline(&body)["audit"]["verdict"], "RejectLethal");
    // Once graph revocation contributes a second policy reason, that distinct
    // diagnostic may add evidence. Subsequent identical audits must deduplicate.
    let (status, second) = post(&engine, "/v1/pipeline/audit_action", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let count = graph.node_count();
    let (status, third) = post(&engine, "/v1/pipeline/audit_action", request).await;
    assert_eq!(status, StatusCode::OK, "{third}");
    assert_eq!(graph.node_count(), count);
    assert_eq!(
        meta_pipeline(&second)["graph_reflection"]["observations"][0]["evidence_node_id"],
        meta_pipeline(&third)["graph_reflection"]["observations"][0]["evidence_node_id"]
    );

    let graph = Arc::new(LodGraph::new());
    graph
        .add_node(
            LodNode::new(
                0,
                LodBand::Lod0Atomic,
                MixedCurvatureCoord::origin(),
                "axiomatic action",
                0,
            )
            .with_status(EpistemicStatus::Axiomatic),
        )
        .unwrap();
    let engine = Arc::new(
        PolymorphicZeroEngine::new()
            .with_bridge(None)
            .with_lod_graph(graph.clone()),
    );
    let (status, body) = post(
        &engine,
        "/v1/pipeline/simulate",
        json!({"state": trap_state(), "actions": [0], "auto_reflect": true}),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert!(body.to_string().contains("GraphReflectionFailed"));
    assert!(body.to_string().contains("quarantined"));
    assert!(graph.is_revoked(0));
    assert_eq!(graph.node_count(), 1);
}
