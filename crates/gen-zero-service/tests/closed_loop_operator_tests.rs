//! Real production dispatch: lexical induction -> deposit -> builtin audit
//! execution. This tests wiring and safety contracts, not learned inference.
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use gen_zero_lod::{
    builtin_operator::{dcm_signature, pcm_signature},
    LodGraph,
};
use gen_zero_service::{McpServer, PolymorphicZeroEngine, ZeroToolOutcome};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

async fn call(engine: &PolymorphicZeroEngine, action: &str, graph: Value) -> ZeroToolOutcome {
    engine
        .execute(&json!({"action": action, "graph": graph}))
        .await
        .unwrap()
}
fn success(out: &ZeroToolOutcome) -> &Value {
    assert!(!out.is_error, "{:?}", out.rejection);
    &out.meta["graph_op"]
}
fn rejected(out: &ZeroToolOutcome, expected: &str) {
    assert!(out.is_error, "{:?}", out.meta);
    assert_eq!(out.rejection.as_ref().unwrap().code, expected);
}
fn engine() -> (PolymorphicZeroEngine, Arc<LodGraph>) {
    let graph = Arc::new(LodGraph::new());
    (
        PolymorphicZeroEngine::new()
            .with_semantic(None)
            .with_lod_graph(graph.clone()),
        graph,
    )
}
async fn induce(engine: &PolymorphicZeroEngine) -> Value {
    let out = call(
        engine,
        "graph_induce",
        json!({"text": "fetch data then clean it and save to db", "auto_deposit": true}),
    )
    .await;
    success(&out).clone()
}

#[tokio::test]
async fn hard_dcm_induction_execution_replay_and_causal_pruning() {
    let (engine, graph) = engine();
    let induced = induce(&engine).await;
    assert_eq!(induced["deposited_node_ids"], json!([0, 1, 2]));
    assert_eq!(
        induced["deposit"]["edge_tickets"].as_array().unwrap().len(),
        2
    );
    for id in 0..3 {
        assert_eq!(graph.get_node(id).unwrap().operator, Some(dcm_signature()));
    }
    let req =
        json!({"node_id": 0, "nonce": "ab".repeat(32), "input": {"reason": "integration audit"}});
    let out = call(&engine, "graph_execute_operator", req.clone()).await;
    let result = success(&out);
    assert_eq!(result["signature"]["operator_kind"], "hard_dcm");
    assert_eq!(result["output"]["state_delta"]["effect"], "audit_append");
    assert_eq!(result["output"]["state_delta"]["audit_records"], 1);
    assert_eq!(
        result["output"]["state_delta"]["record"]["input"],
        req["input"]
    );
    assert_eq!(result["report"]["valid"], true);
    rejected(
        &call(&engine, "graph_execute_operator", req.clone()).await,
        "OperatorNonceRejected",
    );
    // Hex case cannot mint a second nonce.
    let mut upper = req.clone();
    upper["nonce"] = json!("AB".repeat(32));
    rejected(
        &call(&engine, "graph_execute_operator", upper).await,
        "OperatorNonceRejected",
    );
    success(&call(&engine, "graph_prune", json!({"action": "fetch data"})).await);
    let fresh = json!({"node_id": 0, "nonce": "cd".repeat(32), "input": {}});
    rejected(
        &call(&engine, "graph_execute_operator", fresh).await,
        "OperatorNodeRevoked",
    );
    // Causal propagation revokes the direct dependent too.
    rejected(
        &call(
            &engine,
            "graph_execute_operator",
            json!({"node_id": 1, "nonce": "ef".repeat(32), "input": {}}),
        )
        .await,
        "OperatorNodeRevoked",
    );
}

#[tokio::test]
async fn soft_pcm_requires_no_nonce_and_changes_neither_graph_nor_audit() {
    let (engine, graph) = engine();
    induce(&engine).await;
    success(
        &call(
            &engine,
            "graph_execute_operator",
            json!({"node_id": 0, "nonce": "01".repeat(32), "input": {}}),
        )
        .await,
    );
    let deposit = call(
        &engine,
        "graph_deposit",
        json!({"nodes": [{"entity_id": 9001,
        "label": "state evaluation", "band": 1, "status": "validated", "confidence": 0.9,
        "payload": "state evaluation", "operator": pcm_signature()}]}),
    )
    .await;
    success(&deposit);
    let id = graph.node_for_entity(9001).unwrap();
    let before = serde_json::to_value(graph.get_node(id).unwrap()).unwrap();
    let req = json!({"node_id": id, "input": {}});
    let first = call(&engine, "graph_execute_operator", req.clone()).await;
    let second = call(&engine, "graph_execute_operator", req.clone()).await;
    let a = success(&first);
    let b = success(&second);
    assert_eq!(a["signature"]["operator_kind"], "soft_pcm");
    assert_eq!(a["nonce"], Value::Null);
    assert_eq!(a["output"]["state_delta"]["effect"], "none");
    assert_eq!(a["output"]["state_delta"]["audit_records"], 1);
    assert_eq!(a["output"]["state_delta"], b["output"]["state_delta"]);
    assert_eq!(graph.node_count(), 4);
    assert_eq!(
        before,
        serde_json::to_value(graph.get_node(id).unwrap()).unwrap()
    );
    let mut bad = req;
    bad["nonce"] = json!("02".repeat(32));
    rejected(
        &call(&engine, "graph_execute_operator", bad).await,
        "OperatorNonceRejected",
    );
}

#[tokio::test]
async fn operator_wire_contract_fails_closed() {
    let (engine, _) = engine();
    induce(&engine).await;
    rejected(
        &call(
            &engine,
            "graph_execute_operator",
            json!({"node_id": 0, "input": {}}),
        )
        .await,
        "OperatorNonceRejected",
    );
    for nonce in [
        "00".repeat(31),
        "00".repeat(33),
        "gg".repeat(32),
        "é".repeat(32),
    ] {
        rejected(
            &call(
                &engine,
                "graph_execute_operator",
                json!({"node_id": 0, "nonce": nonce, "input": {}}),
            )
            .await,
            "InvalidParams",
        );
    }
    rejected(
        &call(
            &engine,
            "graph_execute_operator",
            json!({"node_id": u64::MAX, "input": {}}),
        )
        .await,
        "InvalidParams",
    );
    rejected(
        &call(
            &engine,
            "graph_execute_operator",
            json!({"node_id": 99, "input": {}}),
        )
        .await,
        "EntityNotFound",
    );
}

#[tokio::test]
async fn http_and_mcp_expose_and_execute_operator() {
    let (engine, _) = engine();
    induce(&engine).await;
    let engine = Arc::new(engine);
    let response = McpServer::build_router(engine.clone(), None)
        .oneshot(
            Request::post("/message")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"action": "graph_execute_operator", "graph": {
                        "node_id": 0, "nonce": "03".repeat(32), "input": {"transport": "http"}
                    }})
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        body["result"]["meta"]["graph_op"]["output"]["state_delta"]["effect"], "audit_append",
        "{body}"
    );
    let server = McpServer {
        engine,
        auth_token: None,
        bridge_required: false,
        closed_loop: None,
    };
    let mut frame = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        .to_string()
        .into_bytes();
    frame.resize(frame.len() + simd_json::SIMDJSON_PADDING, 0);
    let list: Value = serde_json::from_str(&server.handle_jsonrpc_frame(&mut frame).await).unwrap();
    let zero = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "zero")
        .unwrap();
    let verbs = zero["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap();
    assert_eq!(verbs.len(), 24);
    assert!(verbs.contains(&json!("qa_gate")));
    assert!(verbs.contains(&json!("graph_execute_operator")));
    let mut frame = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
    "name": "zero", "arguments": {"action": "graph_execute_operator", "graph": {
        "node_id": 1, "nonce": "04".repeat(32), "input": {"transport": "mcp"}
    }}}})
    .to_string()
    .into_bytes();
    frame.resize(frame.len() + simd_json::SIMDJSON_PADDING, 0);
    let result: Value =
        serde_json::from_str(&server.handle_jsonrpc_frame(&mut frame).await).unwrap();
    assert_eq!(result["result"]["isError"], false, "{result}");
    assert_eq!(
        result["result"]["_meta"]["graph_op"]["output"]["state_delta"]["audit_records"], 2,
        "{result}"
    );
}

#[tokio::test]
async fn postconditions_detect_corrupt_output_input_and_target() {
    use gen_zero_lod::{OperatorExecution, OperatorInput};
    let (engine, graph) = engine();
    induce(&engine).await;
    let out = call(
        &engine,
        "graph_execute_operator",
        json!({
            "node_id": 0, "nonce": "55".repeat(32), "input": {"value": 7}
        }),
    )
    .await;
    let run: OperatorExecution = serde_json::from_value(success(&out).clone()).unwrap();
    let input = OperatorInput {
        node_id: 0,
        parameters: json!({"value": 7}),
        context_digest: run.context_digest,
        nonce: run.nonce,
    };
    let mut wrong_target = input.clone();
    wrong_target.node_id = 1;
    wrong_target.nonce = Some([0x91; 32]);
    assert!(matches!(
        graph.execute_operator(0, &wrong_target),
        Err(gen_zero_lod::LodError::OperatorPreconditionFailed { .. })
    ));
    let op = graph.operator(&dcm_signature()).unwrap();
    let node = graph.get_node(0).unwrap();
    assert!(
        op.verify_postconditions(&run.output, &input, &node)
            .unwrap()
            .valid
    );
    let mut corrupt = run.output.clone();
    corrupt.state_delta["record"]["input"]["value"] = json!(8);
    assert!(
        !op.verify_postconditions(&corrupt, &input, &node)
            .unwrap()
            .valid
    );
    let mut changed = input.clone();
    changed.parameters["value"] = json!(8);
    assert!(
        !op.verify_postconditions(&run.output, &changed, &node)
            .unwrap()
            .valid
    );
    assert!(
        !op.verify_postconditions(&run.output, &input, &graph.get_node(1).unwrap())
            .unwrap()
            .valid
    );
    // Every byte is part of the nonce identity, not just a u64 prefix.
    success(
        &call(
            &engine,
            "graph_execute_operator",
            json!({
                "node_id": 0, "nonce": format!("{}56", "55".repeat(31)), "input": {"value": 7}
            }),
        )
        .await,
    );
}

#[tokio::test]
async fn concurrent_replay_has_exactly_one_audit_effect() {
    let (engine, _) = engine();
    induce(&engine).await;
    let req = json!({"node_id": 0, "nonce": "31".repeat(32), "input": {}});
    let (a, b) = tokio::join!(
        call(&engine, "graph_execute_operator", req.clone()),
        call(&engine, "graph_execute_operator", req),
    );
    assert_ne!(a.is_error, b.is_error);
    let (ok, err) = if a.is_error { (&b, &a) } else { (&a, &b) };
    assert_eq!(success(ok)["output"]["state_delta"]["audit_records"], 1);
    rejected(err, "OperatorNonceRejected");
}
