//! End-to-end: `causal_fold` over HTTP (`/v1/causal_fold`, `zero` via
//! `/message`), and MCP (`tools/list`, `tools/call` for both `zero` and
//! `causal_fold`). Tables mirror the fixtures in `gen-zero-lod/src/semiring.rs`
//! `#[cfg(test)] mod tests`; no trained model or mounted asset is needed
//! since `causal_fold` never touches the cognitive runtime.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gen_zero_service::{McpServer, PolymorphicZeroEngine};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

/// Bridge off: nothing here may depend on the Python scorer.
fn engine() -> Arc<PolymorphicZeroEngine> {
    Arc::new(PolymorphicZeroEngine::new().with_bridge(None))
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

fn two_plus_two_axioms() -> Value {
    json!([
        {"r1": 10, "r2": 11, "gender": "Female", "result": [14]},
        {"r1": 12, "r2": 13, "gender": "Unknown", "result": [15]},
        {"r1": 14, "r2": 15, "gender": "Unknown", "result": [16]},
    ])
}

fn two_plus_two_body(strategy: &str) -> Value {
    json!({
        "edges": [10, 11, 12, 13],
        "genders": ["Male", "Male", "Female", "Male", "Unknown"],
        "axioms": two_plus_two_axioms(),
        "strategy": strategy,
    })
}

// -------------------------------------------------------------- HTTP /v1/causal_fold

#[tokio::test]
async fn http_chart_success_returns_proof_path() {
    let (status, body) = post(&engine(), "/v1/causal_fold", two_plus_two_body("chart")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["is_error"], false, "{body}");
    let cf = &body["result"]["meta"]["causal_fold"];
    assert_eq!(cf["predicted"], 16, "{body}");
    assert_eq!(cf["steps"], 3, "{body}");
    assert_eq!(
        cf["proof_path"],
        json!([10, 11, 14, 12, 13, 15, 16]),
        "{body}"
    );
    assert_eq!(cf["strategy"], "chart", "{body}");
    assert!(body.get("error").is_none(), "{body}");
}

#[tokio::test]
async fn http_left_refusal_is_422_with_causal_fold_refused() {
    let (status, body) = post(&engine(), "/v1/causal_fold", two_plus_two_body("left")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["result"]["is_error"], true, "{body}");
    assert_eq!(body["error"]["code"], "CausalFoldRefused", "{body}");
    assert_eq!(body["result"]["rejection"]["http_status"], 422, "{body}");
    // `engine` must survive onto a refusal too, not just a success.
    assert_eq!(
        body["result"]["meta"]["engine"], "relation_semiring_fold",
        "{body}"
    );
    assert_eq!(
        body["result"]["meta"]["causal_fold"]["strategy"], "left",
        "{body}"
    );
}

#[tokio::test]
async fn http_invalid_strategy_and_length_mismatch_are_400_invalid_params() {
    let mut bad_strategy = two_plus_two_body("chart");
    bad_strategy["strategy"] = json!("sideways");
    let (status, body) = post(&engine(), "/v1/causal_fold", bad_strategy).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "InvalidParams", "{body}");

    let mut null_strategy = two_plus_two_body("chart");
    null_strategy["strategy"] = Value::Null;
    let (status, body) = post(&engine(), "/v1/causal_fold", null_strategy).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "InvalidParams", "{body}");

    let mismatched = json!({"edges": [10, 11], "genders": ["Male", "Male"]});
    let (status, body) = post(&engine(), "/v1/causal_fold", mismatched).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "InvalidParams", "{body}");
}

// -------------------------------------------------------------- HTTP /message (zero)

#[tokio::test]
async fn message_zero_action_causal_fold_succeeds() {
    let request = json!({"action": "causal_fold", "causal_fold": two_plus_two_body("chart")});
    let (status, body) = post(&engine(), "/message", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["is_error"], false, "{body}");
    assert_eq!(
        body["result"]["meta"]["causal_fold"]["predicted"], 16,
        "{body}"
    );
}

// -------------------------------------------------------------- MCP

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
    }
}

#[tokio::test]
async fn mcp_tools_list_advertises_causal_fold() {
    let server = mcp_server();
    let v = frame(
        &server,
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
    )
    .await;
    let tools = v["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"causal_fold"), "{names:?}");
    let cf = tools.iter().find(|t| t["name"] == "causal_fold").unwrap();
    let schema = &cf["inputSchema"];
    assert_eq!(
        schema["properties"]["weights"]["items"]["required"],
        json!(["r1", "r2", "gender", "relation", "count"])
    );
    assert_eq!(schema["properties"]["margin_threshold"]["minimum"], 0);
    assert_eq!(
        schema["properties"]["strategy"]["enum"],
        json!([
            "chart",
            "tiered",
            "left",
            "weighted_tropical",
            "weighted_logprob"
        ])
    );
    assert_eq!(
        schema["properties"]["sets"]["items"]["items"]["type"],
        "integer"
    );
    assert_eq!(
        schema["properties"]["gender"]["enum"],
        json!(["Male", "Female", "Unknown"])
    );
    assert_eq!(schema["anyOf"][1]["required"], json!(["sets", "gender"]));
    assert!(schema.get("required").is_none());
    let zero = tools.iter().find(|t| t["name"] == "zero").unwrap();
    let zero_actions: Vec<&str> = zero["inputSchema"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|a| a.as_str())
        .collect();
    assert!(zero_actions.contains(&"causal_fold"), "{zero_actions:?}");
}

#[tokio::test]
async fn mcp_tools_call_causal_fold_tool_succeeds_and_refuses() {
    let server = mcp_server();
    let call = |name: &str, args: Value| {
        json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": args},
        })
    };

    let ok = frame(&server, call("causal_fold", two_plus_two_body("chart"))).await;
    assert_eq!(ok["result"]["isError"], false, "{ok}");
    assert_eq!(
        ok["result"]["_meta"]["causal_fold"]["predicted"], 16,
        "{ok}"
    );

    let refused = frame(&server, call("causal_fold", two_plus_two_body("left"))).await;
    assert_eq!(refused["result"]["isError"], true, "{refused}");
    assert_eq!(
        refused["result"]["_meta"]["reject"]["code"], "CausalFoldRefused",
        "{refused}"
    );
}

#[tokio::test]
async fn mcp_tools_call_zero_action_causal_fold_succeeds() {
    let server = mcp_server();
    let args = json!({"action": "causal_fold", "causal_fold": two_plus_two_body("chart")});
    let v = frame(
        &server,
        json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "zero", "arguments": args},
        }),
    )
    .await;
    assert_eq!(v["result"]["isError"], false, "{v}");
    assert_eq!(v["result"]["_meta"]["causal_fold"]["predicted"], 16, "{v}");
}

// -------------------------------------------------------------- limits and table checks

async fn expect_invalid(body: Value, needle: &str) {
    let (status, out) = post(&engine(), "/v1/causal_fold", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{out}");
    assert_eq!(out["error"]["code"], "InvalidParams", "{out}");
    let message = out["error"]["message"].as_str().unwrap_or("");
    assert!(message.contains(needle), "{needle:?} not in {message:?}");
}

#[tokio::test]
async fn caps_and_malformed_tables_are_400_invalid_params() {
    let male = |n: usize| vec!["Male"; n];

    let edges: Vec<u16> = vec![0; 65];
    expect_invalid(
        json!({"edges": edges, "genders": male(66)}),
        "more than the cap of 64",
    )
    .await;

    expect_invalid(
        json!({"edges": [], "genders": male(1)}),
        "must not be empty",
    )
    .await;

    // One axiom whose result names 65,536 ids would make a 4-edge chart do
    // billions of lookups; 65 distinct ids is already over the cap.
    let wide: Vec<u16> = (0..65).collect();
    expect_invalid(
        json!({"edges": [0, 0, 0, 0], "genders": male(5),
               "axioms": [{"r1": 0, "r2": 0, "gender": "Male", "result": wide}]}),
        "more than 64 distinct relation",
    )
    .await;

    let many: Vec<Value> = (0..4097u32)
        .map(|i| json!({"r1": i % 65536, "r2": 0, "gender": "Male", "result": [1]}))
        .collect();
    expect_invalid(
        json!({"edges": [0], "genders": male(2), "axioms": many}),
        "cap of 4096",
    )
    .await;

    expect_invalid(
        json!({"edges": [10, 11], "genders": male(3),
               "axioms": [{"r1": 10, "r2": 11, "gender": "Male", "result": [14, 14]}]}),
        "repeats a relation id",
    )
    .await;

    expect_invalid(
        json!({"edges": [10, 11], "genders": male(3),
               "axioms": [{"r1": 10, "r2": 11, "gender": "Male", "result": []}]}),
        "must not be empty",
    )
    .await;

    expect_invalid(
        json!({"edges": [10, 11], "genders": male(3), "axioms": [
            {"r1": 10, "r2": 11, "gender": "Male", "result": [14]},
            {"r1": 10, "r2": 11, "gender": "Male", "result": [15]},
        ]}),
        "duplicates key",
    )
    .await;

    expect_invalid(
        json!({"edges": [10], "genders": male(2), "extra": 1}),
        "unknown field",
    )
    .await;
}

/// Pins the documented chart semantics: a conflict key is carried forward
/// with every candidate, and the chart concludes only because exactly one
/// candidate survives at the root. `left` refuses the same input, and the
/// conflict key is counted in meta rather than hidden.
#[tokio::test]
async fn chart_carries_a_conflict_key_forward_and_reports_it() {
    let body = |strategy: &str| {
        json!({
            "edges": [10, 11, 12],
            "genders": ["Male", "Male", "Male", "Male"],
            "axioms": [
                {"r1": 10, "r2": 11, "gender": "Male", "result": [14, 15]},
                {"r1": 14, "r2": 12, "gender": "Male", "result": [16]},
            ],
            "strategy": strategy,
        })
    };
    let (status, out) = post(&engine(), "/v1/causal_fold", body("left")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{out}");
    assert_eq!(out["error"]["code"], "CausalFoldRefused", "{out}");
    assert_eq!(
        out["result"]["meta"]["causal_fold"]["conflict_keys"], 1,
        "{out}"
    );

    let (status, out) = post(&engine(), "/v1/causal_fold", body("chart")).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    let fold = &out["result"]["meta"]["causal_fold"];
    assert_eq!(fold["predicted"], 16, "{out}");
    assert_eq!(fold["proof_path"], json!([10, 11, 14, 12, 16]), "{out}");
    assert_eq!(fold["conflict_keys"], 1, "{out}");
}

fn sets_body() -> Value {
    json!({"sets": [[1], [2, 3]], "gender": "Male", "axioms": [
        {"r1": 1, "r2": 3, "gender": "Male", "result": [9]}
    ]})
}

#[tokio::test]
async fn http_chart_sets_success_returns_concluded() {
    let (status, body) = post(&engine(), "/v1/causal_fold", sets_body()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let cf = &body["result"]["meta"]["causal_fold"];
    assert_eq!(body["result"]["is_error"], false);
    assert_eq!(cf["predicted"], 9);
    assert_eq!(cf["proof_path"], json!([1, 3, 9]));
    assert_eq!(cf["edges"], 2);
    assert_eq!(cf["steps"], 1);
    assert_eq!(cf["strategy"], "chart");
}

#[tokio::test]
async fn mcp_tools_call_causal_fold_with_sets_succeeds() {
    let out = frame(
        &mcp_server(),
        json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "causal_fold", "arguments": sets_body()}
        }),
    )
    .await;
    assert_eq!(out["result"]["isError"], false, "{out}");
    assert_eq!(out["result"]["_meta"]["causal_fold"]["predicted"], 9);
    assert_eq!(
        out["result"]["_meta"]["causal_fold"]["proof_path"],
        json!([1, 3, 9])
    );
}

#[tokio::test]
async fn sets_invalid_inputs_fail_closed() {
    expect_invalid(
        json!({"sets": null, "edges": [1], "genders": ["Male", "Male"]}),
        "sets must not be null",
    )
    .await;
    expect_invalid(
        json!({"sets": [[1]]}),
        "causal_fold.gender is required when sets is provided",
    )
    .await;
    expect_invalid(json!({"sets": [], "gender": "Male"}), "1 to 64").await;
    expect_invalid(
        json!({"sets": vec![vec![1]; 65], "gender": "Male"}),
        "1 to 64",
    )
    .await;
    expect_invalid(json!({"sets": [[]], "gender": "Male"}), "must not be empty").await;
    expect_invalid(
        json!({"sets": [[1, 1]], "gender": "Male"}),
        "repeats a relation id",
    )
    .await;
    expect_invalid(
        json!({"sets": [[1]], "gender": "Male", "edges": [1]}),
        "cannot provide both",
    )
    .await;
    expect_invalid(
        json!({"sets": [[1]], "gender": "Male", "strategy": "left"}),
        "requires chart",
    )
    .await;
    expect_invalid(
        json!({"sets": [[1]], "gender": "Male", "genders": ["Male"]}),
        "use gender",
    )
    .await;
    let (status, body) = post(
        &engine(),
        "/v1/causal_fold",
        json!({"sets": [[1], [2, 3]], "gender": "Male"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "CausalFoldRefused");
    assert_eq!(body["result"]["meta"]["causal_fold"]["edges"], 2);
}

fn weighted_body(strategy: &str) -> Value {
    json!({
        "strategy": strategy, "edges": [10, 11], "genders": ["Male", "Male", "Male"],
        "axioms": [{"r1":10,"r2":11,"gender":"Male","result":[14,15]}],
        "weights": [
            {"r1":10,"r2":11,"gender":"Male","relation":14,"count":9},
            {"r1":10,"r2":11,"gender":"Male","relation":15,"count":1}
        ]
    })
}

#[tokio::test]
async fn http_weighted_tropical_success_returns_energy_and_margin() {
    let (status, body) = post(
        &engine(),
        "/v1/causal_fold",
        weighted_body("weighted_tropical"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let cf = &body["result"]["meta"]["causal_fold"];
    assert_eq!(cf["predicted"], 14);
    assert!((cf["energy"].as_f64().unwrap() + 0.9_f64.ln()).abs() < 1e-12);
    assert!((cf["margin"].as_f64().unwrap() - 9_f64.ln()).abs() < 1e-12);
    assert!((cf["confidence"].as_f64().unwrap() - 0.9).abs() < 1e-12);
    assert_eq!(cf["proof_path"], json!([10, 11, 14]));
    assert_eq!(cf["candidates"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn http_weighted_logprob_success_returns_marginal_energy() {
    // Both bracketings derive 1: tropical energy 0, marginal energy -ln(2).
    let mut request = json!({
        "strategy":"weighted_logprob", "edges":[1,1,1],
        "genders":["Male","Male","Male","Male"],
        "axioms":[{"r1":1,"r2":1,"gender":"Male","result":[1]}],
        "weights":[{"r1":1,"r2":1,"gender":"Male","relation":1,"count":18446744073709551615u64}]
    });
    let eng = engine();
    for (strategy, expected) in [
        ("weighted_logprob", -2_f64.ln()),
        ("weighted_tropical", 0.0),
    ] {
        request["strategy"] = json!(strategy);
        let (status, body) = post(&eng, "/v1/causal_fold", request.clone()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let cf = &body["result"]["meta"]["causal_fold"];
        assert!((cf["energy"].as_f64().unwrap() - expected).abs() < 1e-12);
        assert_eq!(cf["margin"], "infinity");
        assert_eq!(cf["proof_path"].as_array().unwrap().len(), 5);
    }
}

#[tokio::test]
async fn http_weighted_margin_refusal_when_ambiguous() {
    for strategy in ["weighted_tropical", "weighted_logprob"] {
        let mut request = weighted_body(strategy);
        request["margin_threshold"] = json!(3.0);
        let (status, body) = post(&engine(), "/v1/causal_fold", request.clone()).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(body["error"]["code"], "CausalFoldRefused");
        let cf = &body["result"]["meta"]["causal_fold"];
        assert!(cf["reason"].as_str().unwrap().contains("margin"));
        assert_eq!(cf["candidates"].as_array().unwrap().len(), 2);
        request["margin_threshold"] = json!(0);
        request["weights"][0]["count"] = json!(1);
        assert_eq!(
            post(&engine(), "/v1/causal_fold", request).await.0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }
}

#[tokio::test]
async fn mcp_tools_call_weighted_causal_fold_succeeds() {
    let server = mcp_server();
    for strategy in ["weighted_tropical", "weighted_logprob", "tiered"] {
        let response = frame(
            &server,
            json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
            "params":{"name":"causal_fold","arguments":weighted_body(strategy)}}),
        )
        .await;
        assert_eq!(response["result"]["isError"], false, "{response}");
        assert_eq!(response["result"]["_meta"]["causal_fold"]["predicted"], 14);
        assert!(response["result"]["_meta"]["causal_fold"]["energy"].is_number());
    }
}

#[tokio::test]
async fn sets_payload_size_caps_fail_closed_immediately() {
    let eng = engine();
    for sets in [
        json!([(0..10000).collect::<Vec<_>>()]),
        json!(vec![vec![1; 64]; 5]),
    ] {
        let request = json!({"action":"causal_fold","causal_fold":{"sets":sets,"gender":"Male"}});
        let start = std::time::Instant::now();
        let outcome = eng.execute(&request).await.unwrap();
        let elapsed = start.elapsed();
        eprintln!("borrowed-payload cap gate elapsed={elapsed:?}");
        assert_eq!(outcome.rejection.unwrap().code, "InvalidParams");
        assert!(elapsed < std::time::Duration::from_millis(1), "{elapsed:?}");
        let start = std::time::Instant::now();
        let (status, body) = post(&eng, "/v1/causal_fold", request["causal_fold"].clone()).await;
        eprintln!(
            "HTTP including serialization/parsing elapsed={:?}",
            start.elapsed()
        );
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "InvalidParams");
    }
    // Inclusive boundaries must still reach the algorithm.
    let sets = vec![(0..64).collect::<Vec<_>>(); 4];
    let (status, _) = post(
        &eng,
        "/v1/causal_fold",
        json!({"sets":sets,"gender":"Male"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn weighted_invalid_inputs_fail_closed() {
    let eng = engine();
    let mut cases = Vec::new();
    for field in ["weights", "edges", "genders"] {
        let mut b = weighted_body("weighted_tropical");
        b.as_object_mut().unwrap().remove(field);
        cases.push(b);
    }
    let mut b = weighted_body("weighted_tropical");
    b["weights"] = json!(vec![b["weights"][0].clone(); 4097]);
    cases.push(b);
    for weights in [Value::Null, json!([])] {
        let mut b = weighted_body("weighted_tropical");
        b["weights"] = weights;
        cases.push(b);
    }
    for threshold in [json!(-1), Value::Null, json!("NaN")] {
        let mut b = weighted_body("weighted_tropical");
        b["margin_threshold"] = threshold;
        cases.push(b);
    }
    for count in [json!(0), json!(-1)] {
        let mut b = weighted_body("weighted_tropical");
        b["weights"][0]["count"] = count;
        cases.push(b);
    }
    let mut b = weighted_body("weighted_tropical");
    b["weights"][1] = b["weights"][0].clone();
    cases.push(b);
    let mut b = weighted_body("chart");
    b["weights"] = json!([]);
    cases.push(b);
    for b in cases {
        let (status, body) = post(&eng, "/v1/causal_fold", b).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    let mut b = weighted_body("weighted_tropical");
    b["weights"].as_array_mut().unwrap().pop();
    let (status, body) = post(&eng, "/v1/causal_fold", b).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
}

#[tokio::test]
async fn http_bidirectional_strategy_is_refused_as_retired() {
    expect_invalid(two_plus_two_body("bidirectional"), "deprecated and retired").await;
}

#[tokio::test]
async fn http_tiered_dispatch_resolves_via_band0_when_unambiguous() {
    let (status, body) = post(&engine(), "/v1/causal_fold", two_plus_two_body("tiered")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let cf = &body["result"]["meta"]["causal_fold"];
    assert_eq!(cf["dispatched_band"], 0);
    assert_eq!(cf["predicted"], 16);
    assert!(cf.get("energy").is_none());
}

#[tokio::test]
async fn http_tiered_dispatch_escalates_to_band1_on_ambiguity() {
    for semiring in ["logprob", "tropical"] {
        let mut request = weighted_body("tiered");
        request["semiring"] = json!(semiring);
        let (status, body) = post(&engine(), "/v1/causal_fold", request).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let cf = &body["result"]["meta"]["causal_fold"];
        assert_eq!(cf["dispatched_band"], 1);
        assert_eq!(cf["predicted"], 14);
        assert_eq!(cf["semiring"], semiring);
        assert!(cf["band0_refusal"]["reason"].is_string());
        assert!((cf["confidence"].as_f64().unwrap() - 0.9).abs() < 1e-12);
    }
}

#[tokio::test]
async fn tiered_defaults_and_refusals_fail_closed() {
    let eng = engine();
    let mut request = weighted_body("tiered");
    request.as_object_mut().unwrap().remove("strategy");
    let (status, body) = post(&eng, "/v1/causal_fold", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["meta"]["causal_fold"]["strategy"], "tiered");
    request["strategy"] = json!("tiered");
    request["margin_threshold"] = json!(3);
    let (status, body) = post(&eng, "/v1/causal_fold", request.clone()).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    let cf = &body["result"]["meta"]["causal_fold"];
    assert_eq!(cf["dispatched_band"], 1);
    assert_eq!(cf["candidates"].as_array().unwrap().len(), 2);
    request.as_object_mut().unwrap().remove("weights");
    let (status, body) = post(&eng, "/v1/causal_fold", request).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["result"]["meta"]["causal_fold"]["dispatched_band"], 0);
    assert!(body["result"]["rejection"]["detail"]
        .as_str()
        .unwrap()
        .contains("provide weights"));
    let mut request = weighted_body("tiered");
    request["weights"][0]["count"] = json!(0);
    request["axioms"][0]["result"] = json!([14]);
    expect_invalid(request, "zero count").await;
}

#[tokio::test]
async fn tiered_band0_bypasses_scoring_but_not_invalid_input() {
    let mut request = weighted_body("tiered");
    request["axioms"][0]["result"] = json!([14]);
    request["margin_threshold"] = json!(100);
    let (status, body) = post(&engine(), "/v1/causal_fold", request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let cf = &body["result"]["meta"]["causal_fold"];
    assert_eq!(cf["dispatched_band"], 0);
    assert!(cf.get("energy").is_none());
    for broken in ["tie", "missing_support", "missing_weight"] {
        let mut request = weighted_body("tiered");
        match broken {
            "tie" => request["weights"][0]["count"] = json!(1),
            "missing_support" => request["axioms"] = json!([]),
            _ => {
                request["weights"].as_array_mut().unwrap().pop();
            }
        }
        let (status, body) = post(&engine(), "/v1/causal_fold", request).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(body["error"]["code"], "CausalFoldRefused");
        assert_eq!(body["result"]["meta"]["causal_fold"]["dispatched_band"], 1);
        assert!(body["result"]["meta"]["causal_fold"]["candidates"].is_array());
    }
}
