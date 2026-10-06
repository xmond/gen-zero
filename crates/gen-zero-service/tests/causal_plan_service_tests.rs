//! End-to-end: `causal_plan` over HTTP (`/v1/causal_plan`, `zero` via
//! `/message`) and MCP (`tools/list`, `tools/call zero`), plus the refusal of
//! requests that name no verb (they used to be answered as `ask`).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gen_zero_service::{McpServer, PolymorphicZeroEngine, ZeroVerb};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

/// Bridge off: nothing here may depend on the Python scorer.
fn engine() -> Arc<PolymorphicZeroEngine> {
    Arc::new(PolymorphicZeroEngine::new().with_semantic(None))
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
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    eprintln!("{path} -> {status}: {}", serde_json::to_string(&v).unwrap());
    (status, v)
}

async fn frame(server: &McpServer, req: Value) -> Value {
    let mut buf = req.to_string().into_bytes();
    buf.resize(buf.len() + simd_json::SIMDJSON_PADDING, 0);
    serde_json::from_str(&server.handle_jsonrpc_frame(&mut buf).await).unwrap()
}

fn mcp_server() -> McpServer {
    McpServer {
        engine: engine(),
        auth_token: None,
        bridge_required: false,
        closed_loop: None,
    }
}

/// target 10 needs 4 AND one of {5, 6}; 5 needs 1 and 2, 6 needs the
/// expensive 3. Nodes 20..=22 are unrelated. Optimum: 1, 2, 4, 5, 10 in some
/// valid order, cost 6.
fn and_or_body(budget: f64) -> Value {
    json!({
        "nodes": [
            {"id": 1, "cost": 1.0},
            {"id": 2, "cost": 1.0},
            {"id": 3, "cost": 10.0},
            {"id": 4, "cost": 2.0},
            {"id": 5, "cost": 1.0, "and_parents": [1, 2]},
            {"id": 6, "cost": 1.0, "and_parents": [3]},
            {"id": 10, "cost": 1.0, "and_parents": [4], "or_parents": [[5, 6]]},
            {"id": 20, "cost": 0.0},
            {"id": 21, "cost": 0.0, "and_parents": [20]},
            {"id": 22, "cost": 0.0, "and_parents": [10, 21]}
        ],
        "target": 10,
        "budget": budget
    })
}

fn assert_and_or_plan(plan: &Value) {
    assert_eq!(plan["total_cost"], 6.0, "{plan}");
    assert_eq!(plan["cone_nodes"], 7, "{plan}");
    let path: Vec<u64> = plan["path"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let mut sorted = path.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![1, 2, 4, 5, 10], "{plan}");
    assert_eq!(*path.last().unwrap(), 10, "{plan}");
}

// -------------------------------------------------------------- HTTP /v1/causal_plan

#[tokio::test]
async fn http_and_or_plan_takes_the_cheaper_branch_and_prunes_unrelated_nodes() {
    let (status, body) = post(&engine(), "/v1/causal_plan", and_or_body(6.0)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "causal_plan", "{body}");
    assert_eq!(
        body["result"]["meta"]["engine"], "causal_dag_exact",
        "{body}"
    );
    assert_and_or_plan(&body["result"]["meta"]["causal_plan"]);
    assert!(body.get("error").is_none(), "{body}");
}

#[tokio::test]
async fn http_over_budget_is_422_budget_exceeded() {
    let (status, body) = post(&engine(), "/v1/causal_plan", and_or_body(5.0)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "BudgetExceeded", "{body}");
    assert!(
        body["result"]["meta"].get("causal_plan").is_none(),
        "{body}"
    );
}

#[tokio::test]
async fn http_deadlocked_target_is_422_unreachable_goal() {
    let body = json!({
        "nodes": [
            {"id": 1, "cost": 1.0, "or_parents": [[2]]},
            {"id": 2, "cost": 1.0, "and_parents": [1]},
            {"id": 3, "cost": 1.0, "and_parents": [2]}
        ],
        "target": 3,
        "budget": 100.0
    });
    let (status, body) = post(&engine(), "/v1/causal_plan", body).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "UnreachableGoal", "{body}");
}

#[tokio::test]
async fn http_cone_above_cap_is_422_target_too_complex() {
    let mut nodes: Vec<Value> = (0..33).map(|i| json!({"id": i, "cost": 1.0})).collect();
    nodes.push(json!({"id": 99, "cost": 1.0, "and_parents": (0..33).collect::<Vec<u32>>()}));
    let body = json!({"nodes": nodes, "target": 99, "budget": 1e6});
    let (status, body) = post(&engine(), "/v1/causal_plan", body).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "TargetTooComplex", "{body}");
}

#[tokio::test]
async fn http_malformed_requests_are_400_invalid_params() {
    let cases = [
        json!({"nodes": [{"id": 1, "cost": 1.0}], "target": 1, "budget": 1.0, "approx": true}),
        json!({"nodes": [{"id": 1, "cost": 1.0, "weight": 2}], "target": 1, "budget": 1.0}),
        json!({"nodes": [{"id": 1, "cost": -1.0}], "target": 1, "budget": 1.0}),
        json!({"nodes": [{"id": 1, "cost": 1.0, "and_parents": [7]}], "target": 1, "budget": 1.0}),
        json!({"nodes": [{"id": 1, "cost": 1.0, "or_parents": [[]]}], "target": 1, "budget": 1.0}),
        json!({"nodes": [{"id": 1, "cost": 1.0}], "target": 2, "budget": 1.0}),
        json!({"nodes": [{"id": 1, "cost": 1.0}], "target": 1}),
        json!({"nodes": [{"id": -1, "cost": 1.0}], "target": 1, "budget": 1.0}),
        json!({"nodes": [], "target": 1, "budget": 1.0}),
    ];
    for case in cases {
        let (status, body) = post(&engine(), "/v1/causal_plan", case.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{case} -> {body}");
        assert_eq!(body["error"]["code"], "InvalidParams", "{case} -> {body}");
    }
}

#[tokio::test]
async fn message_zero_action_causal_plan_succeeds() {
    let request = json!({"action": "causal_plan", "causal_plan": and_or_body(10.0)});
    let (status, body) = post(&engine(), "/message", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_and_or_plan(&body["result"]["meta"]["causal_plan"]);
}

#[tokio::test]
async fn causal_plan_block_is_inferred_and_refused_under_another_verb() {
    let engine = engine();
    let inferred = engine
        .execute(&json!({"causal_plan": and_or_body(10.0)}))
        .await
        .unwrap();
    assert_eq!(inferred.verb, ZeroVerb::CausalPlan);
    assert!(!inferred.is_error);

    let out = engine
        .execute(&json!({"action": "grep", "lines": ["a"], "causal_plan": and_or_body(10.0)}))
        .await
        .unwrap();
    assert!(out.is_error);
    assert_eq!(out.rejection.unwrap().code, "InvalidParams");

    let missing = engine
        .execute(&json!({"action": "causal_plan"}))
        .await
        .unwrap();
    assert_eq!(missing.verb, ZeroVerb::CausalPlan);
    assert_eq!(missing.rejection.unwrap().code, "InvalidParams");
}

// -------------------------------------------------------------- MCP

#[tokio::test]
async fn mcp_tools_list_advertises_causal_plan_in_zero() {
    let v = frame(
        &mcp_server(),
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    )
    .await;
    let zero = v["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "zero")
        .unwrap()
        .clone();
    let actions = zero["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap();
    assert!(actions.contains(&json!("causal_plan")), "{zero}");
    let schema = &zero["inputSchema"]["properties"]["causal_plan"];
    assert_eq!(
        schema["required"],
        json!(["nodes", "target", "budget"]),
        "{schema}"
    );
    assert_eq!(schema["additionalProperties"], false, "{schema}");
    assert!(zero["description"]
        .as_str()
        .unwrap()
        .contains("24 cognitive verbs"));
}

#[tokio::test]
async fn mcp_tools_call_zero_causal_plan_succeeds_and_refuses() {
    let server = mcp_server();
    let ok = frame(
        &server,
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "zero",
            "arguments": {"action": "causal_plan", "causal_plan": and_or_body(6.0)}
        }}),
    )
    .await;
    assert_eq!(ok["result"]["isError"], false, "{ok}");
    assert_and_or_plan(&ok["result"]["_meta"]["causal_plan"]);

    let refused = frame(
        &server,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "zero",
            "arguments": {"action": "causal_plan", "causal_plan": and_or_body(5.0)}
        }}),
    )
    .await;
    assert_eq!(refused["result"]["isError"], true, "{refused}");
    assert_eq!(
        refused["result"]["_meta"]["reject"]["code"], "BudgetExceeded",
        "{refused}"
    );
}

// -------------------------------------------------------------- no silent ask

#[tokio::test]
async fn unknown_or_missing_verb_is_refused_never_answered_as_ask() {
    let engine = engine();
    let cases = [
        // Unknown name next to fields the old heuristics would route to ask.
        json!({"action": "bogus", "context": "x", "candidates": ["a", "b"]}),
        json!({"verb": "plan_it", "context": "x", "candidates": ["a", "b"]}),
        // Non-string action used to be skipped silently.
        json!({"action": 5, "context": "x", "candidates": ["a", "b"]}),
        json!({"action": null, "context": "x", "candidates": ["a", "b"]}),
        // No action and no field any verb reads used to default to ask.
        json!({}),
        json!({"foo": 1}),
        json!({"nodes": [], "target": 1}),
        json!("ask"),
    ];
    for case in cases {
        let err = engine.execute(&case).await.expect_err(&case.to_string());
        let message = err.to_string();
        assert!(
            message.contains("unknown action") || message.contains("JSON object"),
            "{case}: {message}"
        );
    }
}

#[tokio::test]
async fn http_unknown_action_is_400_invalid_params() {
    for path in ["/message", "/v1/decisions"] {
        let (status, body) = post(
            &engine(),
            path,
            json!({"action": "bogus", "context": "x", "candidates": ["a", "b"]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
        assert_eq!(body["error"]["code"], "InvalidParams", "{path}: {body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("unknown action"),
            "{path}: {body}"
        );
    }
}

#[tokio::test]
async fn mcp_unknown_action_is_an_error_not_an_ask() {
    let v = frame(
        &mcp_server(),
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "zero",
            "arguments": {"action": "bogus", "context": "x", "candidates": ["a", "b"]}
        }}),
    )
    .await;
    assert_eq!(v["result"]["isError"], true, "{v}");
    assert!(
        v["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("unknown action"),
        "{v}"
    );
}
