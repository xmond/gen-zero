//! The graph verbs through the real `zero` entry (`PolymorphicZeroEngine::execute`),
//! and their effect on the other verbs that share the engine's live LodGraph:
//! a `graph_prune` must hard-stop the pruned actions in `pipeline decide`,
//! `simulate` and `audit`, a `graph_evolve` that retracts the evidence must
//! lift that stop, `pipeline decide` must carry the PPR context of the chosen
//! action, the engine's graph geometry must decide `graph_recall`, and the
//! signed evolution's `gamma` must reach the gate over `POST /message`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gen_zero_lod::GeometryParams;
use gen_zero_service::zero::ZeroEngineConfig;
use gen_zero_service::{McpServer, PolymorphicZeroEngine, ZeroToolOutcome, ZeroVerb};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

const DIM: usize = 1024;
/// `gen_zero_gate::REVOCATION_RULE_ID`.
const REVOCATION_RULE: u32 = u32::MAX;

fn engine() -> PolymorphicZeroEngine {
    PolymorphicZeroEngine::new().with_semantic(None)
}

fn origin() -> Value {
    json!({"hyperbolic": [0, 0, 0, 0], "spherical": [1, 0, 0, 0], "euclidean": [0, 0, 0, 0, 0, 0, 0, 0]})
}

fn node(key: Value, label: &str, status: &str, hdc: u64) -> Value {
    let mut n = json!({
        "label": label, "band": 0, "status": status, "coord": origin(),
        "hdc": [hdc, 0, 0, 0], "confidence": 0.5,
    });
    for (k, v) in key.as_object().unwrap() {
        n[k] = v.clone();
    }
    n
}

async fn run(engine: &PolymorphicZeroEngine, req: Value) -> ZeroToolOutcome {
    engine.execute(&req).await.expect("engine call")
}

fn code(out: &ZeroToolOutcome) -> &str {
    out.rejection.as_ref().map_or("", |r| r.code.as_str())
}

/// Action entity 7 (a pipeline action id), the named action "vent" that depends
/// on it, and a sensor fact next to it.
async fn deposit_world(engine: &PolymorphicZeroEngine) -> ZeroToolOutcome {
    run(
        engine,
        json!({"action": "graph_deposit", "graph": {
            "nodes": [
                node(json!({"entity_id": 7}), "open main valve", "validated", 0b1111),
                node(json!({"action": "vent"}), "vent", "hypothesized", 0b0111),
                node(json!({"entity_id": 900}), "pressure sensor", "validated", 0),
            ],
            "edges": [
                {"source": {"entity_id": 7}, "target": {"action": "vent"}, "type": "depends_on", "weight": 1.0},
                {"source": {"entity_id": 7}, "target": {"entity_id": 900}, "type": "semantic", "weight": 1.0},
            ],
        }}),
    )
    .await
}

fn decide(candidates: &[u32]) -> Value {
    json!({"action": "pipeline", "pipeline": {
        "op": "decide", "state": vec![0.0; DIM], "candidates": candidates,
        "mode": "reflex", "entropy": 0.0,
    }})
}

fn simulate_vent() -> Value {
    json!({"action": "simulate", "state": vec![0.0; DIM], "actions": ["vent"]})
}

#[tokio::test]
async fn deposit_flushes_into_the_csr_and_feeds_recall_and_ppr() {
    let engine = engine();
    let out = deposit_world(&engine).await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.verb, ZeroVerb::GraphDeposit);
    let op = &out.meta["graph_op"];
    assert_eq!(op["flush"]["merged_edges"], 2);
    assert_eq!(op["graph"]["csr_edges"], 2);
    assert_eq!(op["graph"]["pending_edges"], 0);
    assert_eq!(op["graph"]["nodes"], 3);
    assert_eq!(op["graph"]["persisted"], false);

    let out = run(
        &engine,
        json!({"action": "graph_ppr", "graph": {"seeds": [{"entity_id": 7, "weight": 1.0}], "top_k": 3}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let results = out.meta["graph_op"]["results"].as_array().unwrap();
    assert_eq!(results[0]["entity_id"], 7);
    let labels: Vec<&str> = results
        .iter()
        .map(|r| r["label"].as_str().unwrap())
        .collect();
    assert!(
        labels.contains(&"vent") && labels.contains(&"pressure sensor"),
        "{labels:?}"
    );
    assert_eq!(out.meta["graph_op"]["converged"], true);

    let out = run(
        &engine,
        json!({"action": "graph_recall", "graph": {
            "coord": origin(), "hdc": [0b1111, 0, 0, 0], "top_k": 1, "crag_margin": 0.0,
        }}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.meta["graph_op"]["results"][0]["entity_id"], 7);
}

#[tokio::test]
async fn pipeline_decide_carries_advisory_graph_context() {
    let engine = engine();
    deposit_world(&engine).await;
    let out = run(&engine, decide(&[7])).await;
    assert!(!out.is_error, "{:?}", out.meta);
    let ctx = &out.meta["pipeline"]["decision"]["graph_context"];
    assert_eq!(ctx["status"], "diffused");
    assert_eq!(ctx["used_in_choice"], false);
    assert_eq!(ctx["seed_entities"], json!([7]));
    assert_eq!(ctx["facts"].as_array().unwrap().len(), 2, "{ctx}");

    let out = run(&engine, decide(&[8])).await;
    let ctx = &out.meta["pipeline"]["decision"]["graph_context"];
    assert_eq!(ctx["status"], "unavailable");
    assert_eq!(ctx["reason"], "action 8 has no node in the live graph");
}

/// The end-to-end proof that the graph is not an island: pruning entity 7
/// cascades to "vent", and both are hard-stopped by the other verbs.
#[tokio::test]
async fn prune_revokes_actions_across_pipeline_simulate_and_audit() {
    let engine = engine();
    deposit_world(&engine).await;
    let before = run(&engine, simulate_vent()).await;
    assert_eq!(
        before.meta["simulation"]["trajectory"][0]["tier"],
        "Proceed"
    );

    // A dry run reports the cascade and changes nothing.
    let dry = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"entity_id": 7, "dry_run": true}}),
    )
    .await;
    assert!(!dry.is_error, "{:?}", dry.meta);
    assert_eq!(dry.meta["graph_op"]["applied"], false);
    assert_eq!(dry.meta["graph_op"]["pruned"].as_array().unwrap().len(), 2);
    let still = run(&engine, simulate_vent()).await;
    assert_eq!(still.meta["simulation"]["trajectory"][0]["tier"], "Proceed");
    assert!(!run(&engine, decide(&[7])).await.is_error);

    let pruned = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"entity_id": 7}}),
    )
    .await;
    assert!(!pruned.is_error, "{:?}", pruned.meta);
    let op = &pruned.meta["graph_op"];
    assert_eq!(op["applied"], true);
    let labels: Vec<&str> = op["pruned"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, vec!["open main valve", "vent"]);
    assert_eq!(op["retracted_dependencies"], 0);

    let only_revoked = run(&engine, decide(&[7])).await;
    assert!(only_revoked.is_error);
    assert_eq!(code(&only_revoked), "NoFeasibleAction");

    let mixed = run(&engine, decide(&[7, 8])).await;
    assert!(!mixed.is_error, "{:?}", mixed.meta);
    let decision = &mixed.meta["pipeline"]["decision"];
    assert_eq!(decision["action"], 8);
    assert_eq!(decision["pruned"][0]["action"], 7);
    assert_eq!(
        decision["pruned"][0]["violated_rules"],
        json!([REVOCATION_RULE])
    );

    let after = run(&engine, simulate_vent()).await;
    assert_eq!(
        after.meta["simulation"]["trajectory"][0]["tier"],
        "HardStop"
    );
    let audit = run(
        &engine,
        json!({"action": "audit", "state": vec![0.0; DIM], "target_action": "vent", "horizon": 2}),
    )
    .await;
    assert_eq!(audit.meta["verdict"], "REJECT_POLICY", "{:?}", audit.meta);
}

#[tokio::test]
async fn failed_deposit_rolls_back_every_part() {
    let engine = engine();
    let out = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {
            "nodes": [node(json!({"entity_id": 1}), "kept only if the edge lands", "hypothesized", 0)],
            "edges": [{"source": {"entity_id": 1}, "target": {"entity_id": 404}, "type": "semantic", "weight": 1.0}],
        }}),
    )
    .await;
    assert!(out.is_error);
    assert_eq!(code(&out), "EntityNotFound");
    // Node 1 was added inside the transaction and must be gone.
    let ppr = run(
        &engine,
        json!({"action": "graph_ppr", "graph": {"seeds": [{"entity_id": 1, "weight": 1.0}], "top_k": 1}}),
    )
    .await;
    assert_eq!(code(&ppr), "EntityNotFound");
}

#[tokio::test]
async fn graph_requests_fail_closed() {
    let engine = engine();
    let axiom = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {
            "nodes": [node(json!({"entity_id": 1}), "axiom", "axiomatic", 0)],
        }}),
    )
    .await;
    assert_eq!(code(&axiom), "InvalidParams");
    assert!(axiom.rejection.unwrap().detail.contains("seed file"));

    let unknown = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"entity_id": 1, "force": true}}),
    )
    .await;
    assert_eq!(code(&unknown), "InvalidParams");

    let both = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"entity_id": 1, "action": "x"}}),
    )
    .await;
    assert_eq!(code(&both), "InvalidParams");

    let misplaced = run(&engine, json!({"action": "simulate", "graph": {}})).await;
    assert!(misplaced.is_error);
    assert!(misplaced
        .rejection
        .unwrap()
        .detail
        .contains("graph_deposit"));

    let bad_coord = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {"nodes": [{
            "entity_id": 2, "label": "x", "band": 0, "status": "hypothesized",
            "coord": {"hyperbolic": [1, 0, 0, 0], "spherical": [1, 0, 0, 0], "euclidean": [0, 0, 0, 0, 0, 0, 0, 0]},
            "hdc": [0, 0, 0, 0], "confidence": 0.5,
        }]}}),
    )
    .await;
    assert_eq!(code(&bad_coord), "InvalidParams");

    run(&engine, json!({"action": "graph_deposit", "graph": {"nodes": [node(json!({"entity_id": 3}), "a", "hypothesized", 0)]}})).await;
    let dup = run(&engine, json!({"action": "graph_deposit", "graph": {"nodes": [node(json!({"entity_id": 3}), "b", "hypothesized", 0)]}})).await;
    assert_eq!(code(&dup), "DuplicateEntity");
}

#[tokio::test]
async fn seed_file_loads_at_startup_and_a_bad_seed_fails_startup() {
    let dir = tempfile::tempdir().unwrap();
    let seed = dir.path().join("seed.json");
    std::fs::write(
        &seed,
        json!({
            "nodes": [
                node(json!({"entity_id": 11}), "safety invariant", "axiomatic", 0),
                node(json!({"entity_id": 12}), "derived rule", "validated", 0),
            ],
            "edges": [{"source": {"entity_id": 11}, "target": {"entity_id": 12}, "type": "depends_on", "weight": 1.0}],
        })
        .to_string(),
    )
    .unwrap();
    let engine = PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default().with_graph_seed_path(&seed),
    )
    .unwrap()
    .with_semantic(None);
    let ppr = run(
        &engine,
        json!({"action": "graph_ppr", "graph": {"seeds": [{"entity_id": 11, "weight": 1.0}], "top_k": 2}}),
    )
    .await;
    assert!(!ppr.is_error, "{:?}", ppr.meta);
    assert_eq!(ppr.meta["graph_op"]["graph"]["csr_edges"], 1);
    let axiom = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"entity_id": 11}}),
    )
    .await;
    assert_eq!(code(&axiom), "InvalidParams");
    assert!(axiom.rejection.unwrap().detail.contains("axiomatic"));

    let bad = dir.path().join("bad.json");
    std::fs::write(
        &bad,
        json!({"edges": [{"source": {"entity_id": 1}, "target": {"entity_id": 2}, "type": "semantic", "weight": 1.0}]}).to_string(),
    )
    .unwrap();
    assert!(PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default().with_graph_seed_path(&bad)
    )
    .is_err());
}

/// Seeded axiom 11 feeding the named action "vent", and a feedback loop
/// vent -> 30 -> 31 -> vent.
async fn engine_with_cycle() -> (PolymorphicZeroEngine, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let seed = dir.path().join("seed.json");
    std::fs::write(
        &seed,
        json!({"nodes": [node(json!({"entity_id": 11}), "safety invariant", "axiomatic", 0)]})
            .to_string(),
    )
    .unwrap();
    let engine = PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default().with_graph_seed_path(&seed),
    )
    .unwrap()
    .with_semantic(None);
    let edge = |source: Value, target: Value| json!({"source": source, "target": target, "type": "depends_on", "weight": 1.0});
    let out = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {
            "nodes": [
                node(json!({"action": "vent"}), "vent", "hypothesized", 0),
                node(json!({"entity_id": 30}), "relief line", "hypothesized", 0),
                node(json!({"entity_id": 31}), "relief valve", "hypothesized", 0),
            ],
            "edges": [
                edge(json!({"entity_id": 11}), json!({"action": "vent"})),
                edge(json!({"action": "vent"}), json!({"entity_id": 30})),
                edge(json!({"entity_id": 30}), json!({"entity_id": 31})),
                edge(json!({"entity_id": 31}), json!({"action": "vent"})),
            ],
        }}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    (engine, dir)
}

fn evolve_request(extra: Value) -> Value {
    let mut graph = json!({"theta_lo": 0.2, "theta_hi": 0.6});
    for (k, v) in extra.as_object().unwrap() {
        graph[k] = v.clone();
    }
    json!({"action": "graph_evolve", "graph": graph})
}

/// Status by label, read back through `graph_ppr` from the axiom.
async fn statuses(engine: &PolymorphicZeroEngine) -> Vec<(String, String)> {
    let out = run(
        engine,
        json!({"action": "graph_ppr", "graph": {"seeds": [
            {"entity_id": 11, "weight": 1.0}, {"action": "vent", "weight": 1.0},
            {"entity_id": 30, "weight": 1.0}, {"entity_id": 31, "weight": 1.0},
        ], "top_k": 8}}),
    )
    .await;
    let mut seen: Vec<(String, String)> = out.meta["graph_op"]["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["label"].as_str().unwrap().to_string(),
                r["status"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    seen.sort();
    seen
}

/// `graph_evolve` through the real `zero` entry, on a graph with a feedback
/// loop: it converges inside its reported bound, a dry run changes nothing, a
/// prune falsifies the loop and hard-stops the action at the gate, and
/// retracting the evidence restores confidences, statuses and the gate.
#[tokio::test]
async fn evolve_converges_on_a_cycle_and_reverses_a_prune_at_the_gate() {
    let (engine, _dir) = engine_with_cycle().await;
    let hypothesized = statuses(&engine).await;
    assert!(hypothesized
        .iter()
        .all(|(label, status)| label == "safety invariant" || status == "hypothesized"));

    let dry = run(&engine, evolve_request(json!({"dry_run": true}))).await;
    assert!(!dry.is_error, "{:?}", dry.meta);
    assert_eq!(dry.verb, ZeroVerb::GraphEvolve);
    assert_eq!(dry.meta["graph_op"]["op"], "graph_evolve");
    assert_eq!(dry.meta["graph_op"]["applied"], false);
    assert_eq!(dry.meta["graph_op"]["fixed_point"]["transitions_total"], 3);
    assert_eq!(statuses(&engine).await, hypothesized);

    let up = run(&engine, evolve_request(json!({}))).await;
    assert!(!up.is_error, "{:?}", up.meta);
    let fp = &up.meta["graph_op"]["fixed_point"];
    assert_eq!(fp["converged"], true);
    assert_eq!((&fp["nodes"], &fp["pinned"]), (&json!(4), &json!(1)));
    assert_eq!(fp["dependency_edges"], 4);
    let (iterations, k_max) = (
        fp["iterations"].as_u64().unwrap(),
        fp["k_max"].as_u64().unwrap(),
    );
    assert!(
        iterations >= 1 && iterations <= k_max,
        "{iterations} > {k_max}"
    );
    assert!(fp["residual"].as_f64().unwrap() < 1e-6);
    // Echoed defaults and the request's thresholds.
    assert!((fp["beta"].as_f64().unwrap() - 0.85).abs() < 1e-6);
    assert!((fp["theta_hi"].as_f64().unwrap() - 0.6).abs() < 1e-6);
    assert_eq!(fp["max_steps"], 10_000);
    // vent = k/2 + beta (1/2 + c31/2) with the loop closed: 0.80666.
    let confidence = |fp: &Value, label: &str| {
        fp["transitions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["label"] == label)
            .unwrap_or_else(|| panic!("no transition for {label}: {fp}"))["confidence"]
            .as_f64()
            .unwrap()
    };
    let beta = f64::from(0.85_f32);
    let k = 1.0 - beta;
    let vent = (0.5 * k + 0.5 * beta + 0.5 * beta * (0.5 * k + beta * 0.5 * k))
        / (1.0 - 0.5 * beta.powi(3));
    assert!((confidence(fp, "vent") - vent).abs() < 1e-5);
    assert!((confidence(fp, "relief line") - (0.5 * k + beta * vent)).abs() < 1e-5);
    let validated = statuses(&engine).await;
    assert!(validated
        .iter()
        .all(|(label, status)| label == "safety invariant" || status == "validated"));
    let before = run(&engine, simulate_vent()).await;
    assert_eq!(
        before.meta["simulation"]["trajectory"][0]["tier"],
        "Proceed"
    );

    // Evidence against "vent": the loop it feeds falls below theta_lo with it.
    let pruned = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"action": "vent", "theta_lo": 0.2, "theta_hi": 0.6}}),
    )
    .await;
    assert!(!pruned.is_error, "{:?}", pruned.meta);
    let op = &pruned.meta["graph_op"];
    assert_eq!(op["pruned"].as_array().unwrap().len(), 3, "{op}");
    assert_eq!(op["revoked_entities"].as_array().unwrap().len(), 3);
    // (11, vent), (vent, 30), (31, vent) by the evidence, (30, 31) by the evolution.
    assert_eq!(op["retracted_dependencies"], 4);
    assert_eq!(op["fixed_point"]["pinned"], 2);
    let stopped = run(&engine, simulate_vent()).await;
    assert_eq!(
        stopped.meta["simulation"]["trajectory"][0]["tier"],
        "HardStop"
    );

    // Retract the evidence: same confidences as before the prune, gate open again.
    let back = run(
        &engine,
        evolve_request(json!({"retract": [{"action": "vent"}]})),
    )
    .await;
    assert!(!back.is_error, "{:?}", back.meta);
    let fp_back = &back.meta["graph_op"]["fixed_point"];
    assert_eq!(fp_back["transitions_total"], 3);
    assert_eq!(fp_back["reinstated_entities"], json!([30, 31]));
    assert_eq!(fp_back["added_dependencies"], 4);
    for label in ["vent", "relief line", "relief valve"] {
        assert_eq!(confidence(fp_back, label), confidence(fp, label), "{label}");
    }
    assert_eq!(statuses(&engine).await, validated);
    let reopened = run(&engine, simulate_vent()).await;
    assert_eq!(
        reopened.meta["simulation"]["trajectory"][0]["tier"],
        "Proceed"
    );
}

#[tokio::test]
async fn evolve_fails_closed_on_budget_and_bad_requests() {
    let (engine, _dir) = engine_with_cycle().await;
    let before = statuses(&engine).await;

    // One step cannot reach 1e-6 on the loop: refused, nothing committed.
    let short = run(&engine, evolve_request(json!({"max_steps": 1}))).await;
    assert!(short.is_error);
    assert_eq!(code(&short), "FixedPointDiverged");
    let rejection = short.rejection.as_ref().unwrap();
    assert_eq!(rejection.http_status, 422);
    assert!(
        rejection.detail.contains("nothing was committed"),
        "{}",
        rejection.detail
    );
    assert_eq!(statuses(&engine).await, before);
    // The same refusal inside a prune leaves the root unrevoked too. The root
    // must sit outside the loop: pruning a loop node pins it and cuts the
    // loop into a chain, which is solved in one evaluation per node.
    let out = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {"nodes": [
            node(json!({"entity_id": 50}), "gauge", "hypothesized", 0),
        ]}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let before = statuses(&engine).await;
    assert!(before.contains(&("gauge".into(), "hypothesized".into())));
    let prune = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"entity_id": 50, "max_steps": 1}}),
    )
    .await;
    assert_eq!(code(&prune), "FixedPointDiverged");
    assert_eq!(statuses(&engine).await, before);
    let cut = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"action": "vent", "max_steps": 1, "dry_run": true}}),
    )
    .await;
    assert!(!cut.is_error, "{:?}", cut.meta);
    let fp = &cut.meta["graph_op"]["fixed_point"];
    assert_eq!(
        (&fp["cyclic_scc_count"], &fp["iterations"]),
        (&json!(0), &json!(0))
    );
    assert_eq!(fp["node_updates"], fp["nodes"]);
    assert_eq!(statuses(&engine).await, before);
    let sim = run(&engine, simulate_vent()).await;
    assert_eq!(sim.meta["simulation"]["trajectory"][0]["tier"], "Proceed");

    for bad in [
        json!({"beta": 1.0}),
        json!({"beta": 0.0}),
        json!({"tolerance": 0.0}),
        json!({"theta_lo": 0.7}),
        json!({"max_steps": 0}),
        json!({"max_steps": 10_001}),
        json!({"force": true}),
        json!({"retract": [{"entity_id": 30}]}),
        json!({"retract": [{"entity_id": 11}]}),
        json!({"retract": [{"entity_id": 30, "action": "x"}]}),
    ] {
        let out = run(&engine, evolve_request(bad.clone())).await;
        assert_eq!(code(&out), "InvalidParams", "{bad}");
    }
    let missing = run(
        &engine,
        evolve_request(json!({"retract": [{"entity_id": 404}]})),
    )
    .await;
    assert_eq!(code(&missing), "EntityNotFound");
    let no_block = run(&engine, json!({"action": "graph_evolve"})).await;
    assert!(no_block.is_error);
    assert_eq!(statuses(&engine).await, before);
}

fn geometry(curvature: f64, radius: f64, alphas: [f64; 3]) -> GeometryParams {
    GeometryParams {
        curvature,
        radius,
        alpha_h: alphas[0],
        alpha_e: alphas[1],
        alpha_s: alphas[2],
    }
}

fn coord_node(entity: u64, label: &str, hyperbolic: f64, euclidean: f64) -> Value {
    json!({
        "entity_id": entity, "label": label, "band": 0, "status": "hypothesized",
        "coord": {
            "hyperbolic": [hyperbolic, 0, 0, 0], "spherical": [1, 0, 0, 0],
            "euclidean": [euclidean, 0, 0, 0, 0, 0, 0, 0],
        },
        "hdc": [0, 0, 0, 0], "confidence": 0.5,
    })
}

/// Recall order of a hyperbolic-offset node and a Euclidean-offset node under
/// the engine's configured graph geometry, through the `zero` entry.
async fn recall_under(params: GeometryParams) -> (Vec<String>, Value) {
    let engine = PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default().with_graph_geometry(params),
    )
    .unwrap()
    .with_semantic(None);
    let out = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {"nodes": [
            coord_node(1, "H", 0.4, 0.0), coord_node(2, "E", 0.0, 1.0),
        ]}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let out = run(
        &engine,
        json!({"action": "graph_recall", "graph": {
            "coord": origin(), "hdc": [0, 0, 0, 0], "top_k": 2, "crag_margin": 0.0,
        }}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let op = &out.meta["graph_op"];
    let labels = op["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["label"].as_str().unwrap().to_string())
        .collect();
    (labels, op.clone())
}

/// The geometry the engine is configured with reaches the graph nodes: it is
/// echoed, it reorders `graph_recall`, and its curvature decides which
/// coordinates a deposit accepts.
#[tokio::test]
async fn configured_graph_geometry_controls_recall_and_coordinate_domain() {
    // d_H = 2 artanh(0.4) = 0.847, d_E = 1.
    let (unit, op) = recall_under(GeometryParams::UNIT).await;
    assert_eq!(unit, ["H", "E"]);
    assert_eq!(op["graph"]["geometry"], json!(GeometryParams::UNIT));
    assert!((op["results"][0]["distance"].as_f64().unwrap() - 2.0 * 0.4_f64.atanh()).abs() < 1e-5);

    let (by_alpha_h, op) = recall_under(geometry(1.0, 1.0, [4.0, 1.0, 1.0])).await;
    assert_eq!(by_alpha_h, ["E", "H"]);
    assert_eq!(op["graph"]["geometry"]["alpha_h"], 4.0);
    assert!((op["results"][1]["distance"].as_f64().unwrap() - 4.0 * 0.4_f64.atanh()).abs() < 1e-5);

    let (by_alpha_e, _) = recall_under(geometry(1.0, 1.0, [1.0, 4.0, 1.0])).await;
    assert_eq!(by_alpha_e, ["H", "E"]);
    let (by_alpha_e, op) = recall_under(geometry(1.0, 1.0, [1.0, 0.25, 1.0])).await;
    assert_eq!(by_alpha_e, ["E", "H"]);
    assert!((op["results"][0]["distance"].as_f64().unwrap() - 0.5).abs() < 1e-5);

    // Curvature 4: d_H = artanh(0.8) = 1.0986 > 1.
    let (by_curvature, op) = recall_under(geometry(4.0, 1.0, [1.0; 3])).await;
    assert_eq!(by_curvature, ["E", "H"]);
    assert!((op["results"][1]["distance"].as_f64().unwrap() - 0.8_f64.atanh()).abs() < 1e-5);

    // Curvature decides the domain: |x| = 0.8 is a node at c = 1, refused at
    // c = 4; |x| = 1.5 is a node at c = 0.25, refused at c = 1.
    let deposit_at = |params: GeometryParams, norm: f64| async move {
        let engine = PolymorphicZeroEngine::try_from_config(
            ZeroEngineConfig::default().with_graph_geometry(params),
        )
        .unwrap()
        .with_semantic(None);
        let out = run(
            &engine,
            json!({"action": "graph_deposit", "graph": {"nodes": [coord_node(1, "x", norm, 0.0)]}}),
        )
        .await;
        code(&out).to_string()
    };
    assert_eq!(deposit_at(GeometryParams::UNIT, 0.8).await, "");
    assert_eq!(
        deposit_at(geometry(4.0, 1.0, [1.0; 3]), 0.8).await,
        "InvalidParams"
    );
    assert_eq!(deposit_at(geometry(0.25, 1.0, [1.0; 3]), 1.5).await, "");
    assert_eq!(deposit_at(GeometryParams::UNIT, 1.5).await, "InvalidParams");
}

#[test]
fn graph_geometry_configuration_fails_closed() {
    let ok = ZeroEngineConfig::parse_graph_geometry(
        r#"{"curvature": 0.5, "radius": 2, "alpha_h": 1, "alpha_e": 3, "alpha_s": 0.25}"#,
    )
    .unwrap();
    assert_eq!(ok, geometry(0.5, 2.0, [1.0, 3.0, 0.25]));
    for bad in [
        "",
        "unit",
        r#"{"curvature": 1, "radius": 1, "alpha_h": 1, "alpha_e": 1}"#,
        r#"{"curvature": 1, "radius": 1, "alpha_h": 1, "alpha_e": 1, "alpha_s": 1, "extra": 1}"#,
        r#"{"curvature": 0, "radius": 1, "alpha_h": 1, "alpha_e": 1, "alpha_s": 1}"#,
        r#"{"curvature": 1, "radius": -1, "alpha_h": 1, "alpha_e": 1, "alpha_s": 1}"#,
    ] {
        assert!(
            ZeroEngineConfig::parse_graph_geometry(bad).is_err(),
            "{bad:?}"
        );
    }
    // A geometry the f32 node chart cannot hold fails startup.
    assert!(PolymorphicZeroEngine::try_from_config(
        ZeroEngineConfig::default().with_graph_geometry(geometry(1e-60, 1.0, [1.0; 3]))
    )
    .is_err());
}

/// A unit-ball coordinate at Euclidean radius `r` along hyperbolic axis `axis`.
fn radial(r: f64, axis: usize) -> Value {
    let mut h = [0.0; 4];
    h[axis] = r;
    json!({"hyperbolic": h, "spherical": [1, 0, 0, 0], "euclidean": [0, 0, 0, 0, 0, 0, 0, 0]})
}

/// Normalized depth 2 artanh(0.9998) ~ 9.21: band 0. 2 artanh(0.995) ~ 5.99: band 1.
const DEEP: f64 = 0.9998;
const MID: f64 = 0.995;

async fn deposit_leaves(engine: &PolymorphicZeroEngine) -> ZeroToolOutcome {
    let leaf = |entity: u64, axis: usize| {
        json!({"entity_id": entity, "label": format!("leaf {entity}"), "status": "validated",
               "coord": radial(DEEP, axis), "hdc": [entity, 0, 0, 0], "confidence": 0.9})
    };
    run(
        engine,
        json!({"action": "graph_deposit", "graph": {"nodes": [leaf(1, 0), leaf(2, 1), leaf(3, 2)]}}),
    )
    .await
}

fn coarse_grain_request(dry_run: bool) -> Value {
    json!({"action": "graph_coarse_grain", "graph": {
        "members": [{"entity_id": 1}, {"entity_id": 2}, {"entity_id": 3}],
        "entity_id": 500, "coord": radial(MID, 0), "hdc": [0, 0, 0, 0], "dry_run": dry_run,
    }})
}

#[tokio::test]
async fn coarse_grain_and_zoom_run_through_the_zero_entry() {
    let engine = engine();
    let deposited = deposit_leaves(&engine).await;
    assert!(!deposited.is_error, "{:?}", deposited.meta);
    // No `band` given: every leaf gets the band its coordinate implies.
    for n in deposited.meta["graph_op"]["nodes"].as_array().unwrap() {
        assert_eq!(n["band"], 0, "{n}");
    }

    let dry = run(&engine, coarse_grain_request(true)).await;
    assert!(!dry.is_error, "{:?}", dry.meta);
    assert_eq!(dry.verb, ZeroVerb::GraphCoarseGrain);
    assert_eq!(dry.meta["graph_op"]["applied"], false);
    assert_eq!(dry.meta["graph_op"]["graph"]["nodes"], 3);
    assert_eq!(dry.meta["graph_op"]["graph"]["csr_edges"], 0);

    let out = run(&engine, coarse_grain_request(false)).await;
    assert!(!out.is_error, "{:?}", out.meta);
    let op = &out.meta["graph_op"];
    assert_eq!(op["summary"]["entity_id"], 500);
    assert_eq!(op["summary"]["band"], 1);
    assert_eq!(op["edge_type"], "coarse_grain");
    assert_eq!(op["graph"]["nodes"], 4);
    assert_eq!(op["graph"]["csr_edges"], 3);
    assert_eq!(op["graph"]["pending_edges"], 0);

    // The coarse-grain edges are in the CSR: diffusion from a leaf reaches the summary.
    let ppr = run(
        &engine,
        json!({"action": "graph_ppr", "graph": {"seeds": [{"entity_id": 1, "weight": 1.0}], "top_k": 4}}),
    )
    .await;
    let results = ppr.meta["graph_op"]["results"].as_array().unwrap();
    let summary = results.iter().find(|r| r["entity_id"] == 500).unwrap();
    assert!(summary["score"].as_f64().unwrap() > 0.0, "{results:?}");

    // The summary cannot sink to its members' band; it can rise.
    let sink = run(
        &engine,
        json!({"action": "graph_zoom", "graph": {"entity_id": 500, "direction": "in"}}),
    )
    .await;
    assert!(sink.is_error);
    assert_eq!(code(&sink), "InvalidParams");
    let rise_dry = run(
        &engine,
        json!({"action": "graph_zoom", "graph": {"entity_id": 500, "direction": "out", "dry_run": true}}),
    )
    .await;
    assert_eq!(rise_dry.verb, ZeroVerb::GraphZoom);
    assert_eq!(rise_dry.meta["graph_op"]["to_band"], 2);
    assert_eq!(rise_dry.meta["graph_op"]["applied"], false);
    let rise = run(
        &engine,
        json!({"action": "graph_zoom", "graph": {"entity_id": 500, "direction": "out"}}),
    )
    .await;
    assert!(!rise.is_error, "{:?}", rise.meta);
    assert_eq!(rise.meta["graph_op"]["from_band"], 1);
    assert_eq!(rise.meta["graph_op"]["node"]["band"], 2);
    // Back to the band its coordinate implies.
    let back = run(
        &engine,
        json!({"action": "graph_zoom", "graph": {"entity_id": 500, "direction": "to_coord"}}),
    )
    .await;
    assert!(!back.is_error, "{:?}", back.meta);
    assert_eq!(back.meta["graph_op"]["to_band"], 1);

    // Past the top band is a 409; a member already in a cluster is refused.
    let top = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {"nodes": [
            {"entity_id": 9, "label": "root", "status": "validated", "coord": origin(),
             "hdc": [0, 0, 0, 0], "confidence": 0.5}]}}),
    )
    .await;
    assert_eq!(top.meta["graph_op"]["nodes"][0]["band"], 3);
    let past = run(
        &engine,
        json!({"action": "graph_zoom", "graph": {"entity_id": 9, "direction": "out"}}),
    )
    .await;
    assert_eq!(code(&past), "BandOutOfRange");
    let again = run(&engine, coarse_grain_request(false)).await;
    assert!(again.is_error);
    assert_eq!(code(&again), "InvalidParams");
}

#[tokio::test]
async fn coarse_grain_edges_and_requests_fail_closed() {
    let engine = engine();
    deposit_leaves(&engine).await;
    // A deposited coarse_grain edge must go from a finer band to a coarser one.
    let flat = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {"edges": [
            {"source": {"entity_id": 1}, "target": {"entity_id": 2}, "type": "coarse_grain", "weight": 1.0}]}}),
    )
    .await;
    assert!(flat.is_error);
    assert_eq!(code(&flat), "InvalidParams");

    // A summary as deep as its members is refused and changes nothing.
    let mut deep = coarse_grain_request(false);
    deep["graph"]["coord"] = radial(DEEP, 3);
    let refused = run(&engine, deep).await;
    assert!(refused.is_error);
    for bad in [
        json!({"action": "graph_coarse_grain", "graph": {"members": [], "entity_id": 500,
               "coord": radial(MID, 0), "hdc": [0, 0, 0, 0]}}),
        json!({"action": "graph_coarse_grain", "graph": {"members": [{"entity_id": 404}],
               "entity_id": 500, "coord": radial(MID, 0), "hdc": [0, 0, 0, 0]}}),
        json!({"action": "graph_zoom", "graph": {"entity_id": 1, "direction": "sideways"}}),
        json!({"action": "graph_zoom", "graph": {"entity_id": 1, "direction": "in", "extra": 1}}),
    ] {
        assert!(run(&engine, bad.clone()).await.is_error, "{bad}");
    }
    let ppr = run(
        &engine,
        json!({"action": "graph_ppr", "graph": {"seeds": [{"entity_id": 1, "weight": 1.0}], "top_k": 1}}),
    )
    .await;
    assert_eq!(ppr.meta["graph_op"]["graph"]["nodes"], 3);
    assert_eq!(ppr.meta["graph_op"]["graph"]["csr_edges"], 0);
}

const PUMP: &str = "the reactor coolant pump failed during the night shift";
const LOG: &str = "maintenance ticket 4411 replaced seal kit on unit two";
const BUDGET: &str = "quarterly marketing budget for the new espresso brand";

/// A node placed by projecting its payload: no `coord`, no `hdc`.
fn text_node(entity: u64, text: &str, status: &str) -> Value {
    json!({
        "entity_id": entity, "label": format!("fact {entity}"), "band": 0,
        "status": status, "confidence": 0.9,
        "payload": text, "source_uri": format!("doc://kb/{entity}.md"),
    })
}

async fn deposit_texts(engine: &PolymorphicZeroEngine) {
    let mut pump = text_node(1, PUMP, "validated");
    pump["timestamp_ns"] = json!(1_700_000_000_000_000_000_u64);
    let out = run(
        engine,
        json!({"action": "graph_deposit", "graph": {
            "nodes": [
                pump,
                text_node(2, LOG, "validated"),
                text_node(3, BUDGET, "validated"),
                // The query's own words, but refuted: must never be recalled.
                text_node(4, "coolant pump failed during night shift", "falsified"),
            ],
            "edges": [
                {"source": {"entity_id": 1}, "target": {"entity_id": 2}, "type": "causal_transition", "weight": 1.0},
                {"source": {"entity_id": 4}, "target": {"entity_id": 3}, "type": "semantic", "weight": 1.0},
            ],
        }}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let nodes = out.meta["graph_op"]["nodes"].as_array().unwrap();
    assert_eq!(nodes[0]["payload_bytes"], PUMP.len());
    assert_eq!(
        nodes[0]["payload_digest"],
        blake3::hash(PUMP.as_bytes()).to_hex().to_string()
    );
    assert_eq!(nodes[0]["timestamp_ns"], 1_700_000_000_000_000_000_u64);
    // No timestamp given: the deposit's wall-clock time, echoed.
    assert!(nodes[1]["timestamp_ns"].as_u64().unwrap() > 1_600_000_000_000_000_000);
    assert_eq!(nodes[1]["source_uri"], "doc://kb/2.md");
}

#[tokio::test]
async fn graph_rag_answers_a_text_query_with_payload_and_graph_evidence() {
    let engine = engine();
    deposit_texts(&engine).await;
    let out = run(
        &engine,
        json!({"action": "graph_rag", "graph": {
            "query_text": "Coolant pump failed during the night shift?", "top_k": 1,
        }}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert_eq!(out.verb, ZeroVerb::GraphRag);
    let op = &out.meta["graph_op"];
    assert_eq!(op["op"], "graph_rag");
    assert_eq!(op["query"]["kind"], "text");
    assert_eq!(op["query"]["projector"], gen_zero_lod::PROJECTOR_VERSION);
    assert_eq!(op["crag_margin"], 0.0);
    assert_eq!(op["stage1_candidates"], 3, "{op}");
    assert_eq!(op["anchors"][0]["entity_id"], 1);
    assert_eq!(op["diffusion"]["converged"], true);

    let hits = op["hits"].as_array().unwrap();
    let entities: Vec<u64> = hits
        .iter()
        .map(|h| h["entity_id"].as_u64().unwrap())
        .collect();
    // Anchor first, its causal successor by diffusion; the budget fact is
    // reached by no anchor and the refuted fact is never a candidate.
    assert_eq!(entities, vec![1, 2], "{op}");
    let top = &hits[0];
    assert_eq!(top["via"], "anchor");
    assert_eq!(top["payload"], PUMP);
    assert_eq!(top["source_uri"], "doc://kb/1.md");
    assert_eq!(top["timestamp_ns"], 1_700_000_000_000_000_000_u64);
    assert_eq!(
        top["payload_digest"],
        blake3::hash(PUMP.as_bytes()).to_hex().to_string()
    );
    assert!(top["anchor_distance"].as_f64().unwrap() >= 0.0);
    let next = &hits[1];
    assert_eq!(next["via"], "diffusion");
    assert!(next["anchor_distance"].is_null());
    assert_eq!(next["payload"], LOG);
    assert_eq!(next["source_uri"], "doc://kb/2.md");
    assert!(next["ppr_score"].as_f64().unwrap() > 0.0);
    assert!(next["ppr_score"].as_f64().unwrap() < top["ppr_score"].as_f64().unwrap());
}

#[tokio::test]
async fn graph_rag_takes_a_coord_query_and_mixes_bare_and_text_nodes() {
    let engine = engine();
    let out = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {"nodes": [
            node(json!({"entity_id": 10}), "bare fact", "validated", 0b1111),
            text_node(11, PUMP, "hypothesized"),
        ], "edges": [
            {"source": {"entity_id": 10}, "target": {"entity_id": 11}, "type": "semantic", "weight": 1.0},
        ]}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    assert!(out.meta["graph_op"]["nodes"][0]["payload_digest"].is_null());
    let out = run(
        &engine,
        json!({"action": "graph_rag", "graph": {
            "coord": origin(), "hdc": [0b1111, 0, 0, 0], "top_k": 1, "alpha": 0.3, "max_iters": 500,
        }}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let op = &out.meta["graph_op"];
    assert_eq!(op["query"]["kind"], "coord");
    assert_eq!(op["diffusion"]["alpha"], json!(0.3_f32));
    let hits = op["hits"].as_array().unwrap();
    assert_eq!(hits[0]["entity_id"], 10);
    assert!(hits[0]["payload"].is_null());
    assert_eq!(hits[1]["entity_id"], 11);
    assert_eq!(hits[1]["payload"], PUMP);
}

#[tokio::test]
async fn graph_rag_and_text_deposits_fail_closed() {
    let engine = engine();
    let empty = run(
        &engine,
        json!({"action": "graph_rag", "graph": {"query_text": "anything", "top_k": 2}}),
    )
    .await;
    assert!(!empty.is_error, "{:?}", empty.meta);
    assert_eq!(empty.meta["graph_op"]["hits"], json!([]));
    assert!(empty.meta["graph_op"]["diffusion"].is_null());

    deposit_texts(&engine).await;
    for (req, want) in [
        (json!({"query_text": "   ", "top_k": 1}), "EmptyInput"),
        (json!({"query_text": "?!", "top_k": 1}), "EmptyInput"),
        (json!({"query_text": "pump", "top_k": 0}), "InvalidParams"),
        (json!({"query_text": "pump", "top_k": 33}), "InvalidParams"),
        (
            json!({"query_text": "pump", "top_k": 1, "alpha": 1.0}),
            "InvalidParams",
        ),
        (
            json!({"query_text": "pump", "top_k": 1, "max_iters": 0}),
            "InvalidParams",
        ),
        (
            json!({"query_text": "pump", "top_k": 1, "coord": origin(), "hdc": [0, 0, 0, 0]}),
            "InvalidParams",
        ),
        (json!({"coord": origin(), "top_k": 1}), "InvalidParams"),
        (json!({"top_k": 1}), "InvalidParams"),
        (
            json!({"query_text": "pump", "top_k": 1, "extra": 1}),
            "InvalidParams",
        ),
    ] {
        let out = run(
            &engine,
            json!({"action": "graph_rag", "graph": req.clone()}),
        )
        .await;
        assert!(out.is_error, "{req}");
        assert_eq!(code(&out), want, "{req}");
    }

    let mut no_band = text_node(20, "a fact", "validated");
    no_band.as_object_mut().unwrap().remove("band");
    let mut coord_only = text_node(21, "a fact", "validated");
    coord_only["coord"] = origin();
    let mut uri_only = node(json!({"entity_id": 22}), "bare", "validated", 1);
    uri_only["source_uri"] = json!("doc://x");
    let mut stamp_only = node(json!({"entity_id": 23}), "bare", "validated", 1);
    stamp_only["timestamp_ns"] = json!(5);
    let mut neither = text_node(24, "a fact", "validated");
    neither.as_object_mut().unwrap().remove("payload");
    neither.as_object_mut().unwrap().remove("source_uri");
    let blank = text_node(25, " \n ", "validated");
    for (bad, want) in [
        (no_band, "InvalidParams"),
        (coord_only, "InvalidParams"),
        (uri_only, "InvalidParams"),
        (stamp_only, "InvalidParams"),
        (neither, "InvalidParams"),
        (blank, "EmptyInput"),
        (
            text_node(26, &"x".repeat(64 * 1024 + 1), "validated"),
            "PayloadTooLarge",
        ),
    ] {
        // A good node rides along: the whole deposit must be refused.
        let out = run(
            &engine,
            json!({"action": "graph_deposit", "graph": {"nodes": [text_node(30, LOG, "validated"), bad.clone()]}}),
        )
        .await;
        assert!(out.is_error, "{bad}");
        assert_eq!(code(&out), want, "{bad}");
    }
    let out = run(
        &engine,
        json!({"action": "graph_rag", "graph": {"query_text": PUMP, "top_k": 1}}),
    )
    .await;
    assert_eq!(out.meta["graph_op"]["graph"]["nodes"], 4, "{:?}", out.meta);
}

const CLOSE_MAIN_VALVE: &str = "关闭主阀";
const HANDWHEEL: &str = "turn the handwheel clockwise until the stem stops";

/// Thirteen facts that are not about closing a valve. The first two share a
/// word with "valve closure". With `top_k` 1 the Stage 1 pool holds 4 nodes, so
/// most of these and anything lexically unrelated fall outside it.
fn distractors() -> Vec<Value> {
    [
        "the relief valve was replaced last week",
        "closure of the quarterly accounts is due friday",
        PUMP,
        LOG,
        BUDGET,
        "the night shift supervisor signed the handover log",
        "spare bearings are stored in warehouse three",
        "the canteen menu changes every monday",
        "calibration of the flow meter is overdue",
        "the forklift battery needs charging",
        "fire drill scheduled for the second floor",
        "new safety boots arrive next month",
        "the turbine hall lighting was upgraded",
    ]
    .iter()
    .enumerate()
    .map(|(i, text)| text_node(100 + i as u64, text, "validated"))
    .collect()
}

fn hit_entities(op: &Value) -> Vec<u64> {
    op["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["entity_id"].as_u64().unwrap())
        .collect()
}

/// The valve facts in two languages on top of the distractors, with or
/// without the aliases that name them the same thing.
async fn deposit_valve_facts(engine: &PolymorphicZeroEngine, aliases: bool) -> Value {
    let mut zh = text_node(1, CLOSE_MAIN_VALVE, "validated");
    let mut en = text_node(2, HANDWHEEL, "validated");
    if aliases {
        zh["aliases"] = json!(["valve closure", "主阀关断"]);
        en["aliases"] = json!(["Valve  CLOSURE"]);
    }
    let mut nodes = distractors();
    nodes.extend([zh, en]);
    let out = run(
        engine,
        json!({"action": "graph_deposit", "graph": {"nodes": nodes}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    out.meta["graph_op"].clone()
}

async fn rag_text(engine: &PolymorphicZeroEngine, text: &str, top_k: usize) -> Value {
    let out = run(
        engine,
        json!({"action": "graph_rag", "graph": {"query_text": text, "top_k": top_k}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    out.meta["graph_op"].clone()
}

/// A query in one language reaches the fact written in the other only through
/// the alias. The control engine holds the same facts without aliases and
/// misses them.
#[tokio::test]
async fn graph_rag_recalls_a_translation_through_aliases_and_the_control_misses_it() {
    let control = engine();
    let deposited = deposit_valve_facts(&control, false).await;
    assert_eq!(deposited["alias_link_edges"], 0);
    let op = rag_text(&control, "valve closure", 1).await;
    assert_eq!(op["searchable_nodes"], 15);
    assert_eq!(op["stage1_candidates"], 4);
    let missed = hit_entities(&op);
    assert!(!missed.contains(&1) && !missed.contains(&2), "{op}");
    assert!([100, 101].contains(&missed[0]), "{op}");
    let op = rag_text(&control, CLOSE_MAIN_VALVE, 1).await;
    assert_eq!(hit_entities(&op), vec![1], "{op}");

    let engine = engine();
    let deposited = deposit_valve_facts(&engine, true).await;
    // One pair of nodes shares the alias: one semantic edge each way.
    assert_eq!(deposited["alias_link_edges"], 2);
    assert_eq!(deposited["flush"]["merged_edges"], 2);
    let nodes = deposited["nodes"].as_array().unwrap();
    assert_eq!(nodes[13]["aliases"], json!(["valve closure", "主阀关断"]));
    assert_eq!(nodes[13]["placement"], "chart");

    // The English name finds both facts, each by its alias.
    let op = rag_text(&engine, "valve closure", 2).await;
    let mut anchors = hit_entities(&op);
    anchors.truncate(2);
    anchors.sort_unstable();
    assert_eq!(anchors, vec![1, 2], "{op}");
    for hit in &op["hits"].as_array().unwrap()[..2] {
        assert_eq!(hit["via"], "anchor");
        assert_eq!(hit["matched"], "alias");
        assert!(hit["anchor_distance"].as_f64().unwrap() < 1e-6);
    }
    let zh = op["hits"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["entity_id"] == 1)
        .unwrap();
    assert_eq!(zh["matched_alias"], "valve closure");
    assert_eq!(zh["payload"], CLOSE_MAIN_VALVE);

    // The Chinese text anchors its own fact; the alias link carries the
    // diffusion to the English one, which shares no n-gram with the query.
    let op = rag_text(&engine, CLOSE_MAIN_VALVE, 1).await;
    assert_eq!(op["anchors"].as_array().unwrap().len(), 1);
    assert_eq!(hit_entities(&op), vec![1, 2], "{op}");
    let hits = op["hits"].as_array().unwrap();
    assert_eq!(hits[0]["matched"], "primary");
    assert_eq!(hits[1]["via"], "diffusion");
    assert!(hits[1]["matched"].is_null());
    assert_eq!(hits[1]["payload"], HANDWHEEL);
    assert!(hits[1]["ppr_score"].as_f64().unwrap() > 0.0);
}

/// `dim` values in [-1, 1) fixed by `seed` (SplitMix64).
fn random_vector(seed: u64, dim: usize) -> Vec<f32> {
    (0..dim as u64)
        .map(|i| {
            let mut z = seed
                .wrapping_mul(0xD6E8_FEB8_6659_FD93)
                .wrapping_add((i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

/// `base` with `noise` of the given size mixed in.
fn near(base: &[f32], noise: f32, seed: u64) -> Vec<f32> {
    base.iter()
        .zip(random_vector(seed, base.len()))
        .map(|(b, n)| b + noise * n)
        .collect()
}

const VECTOR_DIM: usize = 128;

/// Thirteen embedding-only distractors, a leak fact near `topic` (entity 1,
/// embedding only), a text fact with its own embedding (entity 3) and a
/// text-only procedure (entity 2) the leak fact points to.
async fn deposit_embedded(engine: &PolymorphicZeroEngine, topic: &[f32], link: bool) -> Value {
    let embedded = |entity: u64, label: &str, embedding: Vec<f32>| {
        json!({
            "entity_id": entity, "label": label, "band": 0, "status": "validated",
            "confidence": 0.9, "embedding": embedding,
        })
    };
    let mut nodes: Vec<Value> = (0..13)
        .map(|i| embedded(100 + i, "distractor", random_vector(900 + i, VECTOR_DIM)))
        .collect();
    nodes.push(embedded(1, "冷却液泄漏", near(topic, 0.3, 7)));
    nodes.push(text_node(2, HANDWHEEL, "validated"));
    let mut both = text_node(3, PUMP, "validated");
    both["embedding"] = json!(random_vector(55, VECTOR_DIM));
    nodes.push(both);
    let edges = if link {
        json!([{"source": {"entity_id": 1}, "target": {"entity_id": 2}, "type": "semantic", "weight": 1.0}])
    } else {
        json!([])
    };
    let out = run(
        engine,
        json!({"action": "graph_deposit", "graph": {"nodes": nodes, "edges": edges}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    out.meta["graph_op"].clone()
}

/// `query_vector` over HTTP `/message`: the dense track anchors the node whose
/// embedding is closest, PPR carries on to a text-only node, and the response
/// says which track and which anchor answered.
#[tokio::test]
async fn http_graph_rag_takes_a_query_vector_and_diffuses_from_the_embedding_anchor() {
    let topic = random_vector(42, VECTOR_DIM);
    let engine = Arc::new(engine());
    let deposited = deposit_embedded(&engine, &topic, true).await;
    let nodes = deposited["nodes"].as_array().unwrap();
    assert_eq!(nodes[13]["placement"], "embedding");
    assert_eq!(nodes[13]["embedding_dim"], VECTOR_DIM);
    assert_eq!(nodes[14]["placement"], "chart");
    assert!(nodes[14]["embedding_dim"].is_null());
    // Payload and embedding together: placed by the text, searchable by both.
    assert_eq!(nodes[15]["placement"], "chart");
    assert_eq!(nodes[15]["embedding_dim"], VECTOR_DIM);

    let (status, body) = http_zero(
        &engine,
        json!({"action": "graph_rag", "graph": {"query_vector": topic, "top_k": 1}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "graph_rag");
    let op = &body["result"]["meta"]["graph_op"];
    assert_eq!(op["query"]["kind"], "vector");
    assert_eq!(
        op["query"]["dense_projector"],
        gen_zero_lod::DENSE_PROJECTOR_VERSION
    );
    assert!(op["query"]["projector"].is_null());
    assert_eq!(op["query"]["vector_dim"], VECTOR_DIM);
    // 15 of the 16 nodes carry an embedding; the pool is 4 of them.
    assert_eq!(op["searchable_nodes"], 15);
    assert_eq!(op["stage1_candidates"], 4);
    assert_eq!(op["anchors"][0]["entity_id"], 1);
    assert_eq!(hit_entities(op), vec![1, 2], "{op}");
    let hits = op["hits"].as_array().unwrap();
    assert_eq!(hits[0]["matched"], "embedding");
    assert_eq!(hits[0]["label"], "冷却液泄漏");
    assert_eq!(hits[1]["via"], "diffusion");
    assert_eq!(hits[1]["payload"], HANDWHEEL);
    assert_eq!(op["diffusion"]["converged"], true);

    // Text and vector in one request: one anchor from each track.
    let (status, body) = http_zero(
        &engine,
        json!({"action": "graph_rag", "graph": {
            "query_text": "handwheel clockwise", "query_vector": topic, "top_k": 1,
        }}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let op = &body["result"]["meta"]["graph_op"];
    assert_eq!(op["query"]["kind"], "text+vector");
    assert_eq!(op["query"]["projector"], gen_zero_lod::PROJECTOR_VERSION);
    assert_eq!(op["searchable_nodes"], 16);
    let matched: Vec<(u64, &str)> = op["hits"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|h| h["via"] == "anchor")
        .map(|h| {
            (
                h["entity_id"].as_u64().unwrap(),
                h["matched"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(matched.len(), 2, "{op}");
    assert!(matched.contains(&(1, "embedding")), "{op}");
    assert!(matched.contains(&(2, "primary")), "{op}");

    // A text query alone never lands on an embedding-only node.
    let op = rag_text(&engine, "handwheel clockwise", 1).await;
    assert_eq!(op["searchable_nodes"], 2);
    assert_eq!(op["hits"][0]["entity_id"], 2);

    // Control: the same facts without the edge do not recall the procedure.
    let control = Arc::new(self::engine());
    deposit_embedded(&control, &topic, false).await;
    let (_, body) = http_zero(
        &control,
        json!({"action": "graph_rag", "graph": {"query_vector": topic, "top_k": 1}}),
    )
    .await;
    assert_eq!(
        hit_entities(&body["result"]["meta"]["graph_op"]),
        vec![1],
        "{body}"
    );
}

#[tokio::test]
async fn aliases_embeddings_and_query_vectors_fail_closed() {
    let engine = engine();
    let vector = random_vector(1, VECTOR_DIM);
    // No node carries an embedding yet: a vector has nothing to be compared with.
    let out = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {"nodes": [text_node(60, BUDGET, "validated")]}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let out = run(
        &engine,
        json!({"action": "graph_rag", "graph": {"query_vector": vector, "top_k": 1}}),
    )
    .await;
    assert!(out.is_error);
    assert_eq!(code(&out), "InvalidParams");
    assert!(
        out.rejection.as_ref().unwrap().detail.contains("no node"),
        "{:?}",
        out.rejection
    );

    deposit_embedded(&engine, &vector, false).await;
    let nodes_before = 1 + 16;
    for (req, want) in [
        (
            json!({"query_vector": random_vector(2, 256), "top_k": 1}),
            "InvalidParams",
        ),
        (json!({"query_vector": [], "top_k": 1}), "InvalidParams"),
        (
            json!({"query_vector": vec![0.0; VECTOR_DIM], "top_k": 1}),
            "InvalidParams",
        ),
        (
            json!({"query_vector": vector, "top_k": 1, "coord": origin(), "hdc": [0, 0, 0, 0]}),
            "InvalidParams",
        ),
        (
            json!({"query_vector": vector, "query_text": " ", "top_k": 1}),
            "EmptyInput",
        ),
        (
            json!({"query_vector": "not a vector", "top_k": 1}),
            "InvalidParams",
        ),
    ] {
        let out = run(
            &engine,
            json!({"action": "graph_rag", "graph": req.clone()}),
        )
        .await;
        assert!(out.is_error, "{req}");
        assert_eq!(code(&out), want, "{req}");
    }

    let with = |entity: u64, key: &str, value: Value| {
        let mut n = text_node(entity, "a fact", "validated");
        n[key] = value;
        n
    };
    let mut embedding_no_band = json!({
        "entity_id": 44, "label": "v", "status": "validated", "confidence": 0.5,
        "embedding": vector,
    });
    let many: Vec<String> = (0..17).map(|i| format!("name {i}")).collect();
    for (bad, want) in [
        (
            with(40, "embedding", json!(random_vector(3, 256))),
            "InvalidParams",
        ),
        (
            with(41, "embedding", json!(vec![0.0; VECTOR_DIM])),
            "InvalidParams",
        ),
        (with(42, "aliases", json!(many)), "InvalidParams"),
        (with(43, "aliases", json!(["ok", "  "])), "EmptyInput"),
        (
            with(45, "aliases", json!(["same", "SAME"])),
            "InvalidParams",
        ),
        (with(46, "aliases", json!("not a list")), "InvalidParams"),
        (embedding_no_band.take(), "InvalidParams"),
    ] {
        // A good node rides along: the whole deposit must be refused.
        let out = run(
            &engine,
            json!({"action": "graph_deposit", "graph": {"nodes": [text_node(50, LOG, "validated"), bad.clone()]}}),
        )
        .await;
        assert!(out.is_error, "{bad}");
        assert_eq!(code(&out), want, "{bad}");
    }
    let op = rag_text(&engine, PUMP, 1).await;
    assert_eq!(op["graph"]["nodes"], nodes_before, "{op}");
}

/// `zero` over HTTP: `POST /message` with the verb's request as the body.
async fn http_zero(engine: &Arc<PolymorphicZeroEngine>, req: Value) -> (StatusCode, Value) {
    let resp = McpServer::build_router(Arc::clone(engine), None)
        .oneshot(
            Request::post("/message")
                .header("content-type", "application/json")
                .body(Body::from(req.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 22)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    eprintln!("/message -> {status}: {body}");
    (status, body)
}

/// The cycle world plus a fully confident "leak report" (entity 40) that
/// falsifies "vent". Components: {11}, {40}, {vent, 30, 31}.
async fn engine_with_falsifier() -> (Arc<PolymorphicZeroEngine>, tempfile::TempDir) {
    let (engine, dir) = engine_with_cycle().await;
    let mut leak = node(json!({"entity_id": 40}), "leak report", "validated", 0);
    leak["confidence"] = json!(1.0);
    let out = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {
            "nodes": [leak],
            "edges": [{"source": {"entity_id": 40}, "target": {"action": "vent"},
                       "type": "falsifies", "weight": 1.0}],
        }}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    (Arc::new(engine), dir)
}

/// `graph_evolve` with `gamma` through `POST /message`: the falsifier drives the
/// loop below `theta_lo` and hard-stops "vent" at the gate; `gamma: 0` restores
/// it; the SCC block counts and `gamma` come back in the report; a bad or
/// non-contractive `gamma` is refused and commits nothing.
#[tokio::test]
async fn http_evolve_takes_gamma_and_reports_scc_blocks() {
    let (engine, _dir) = engine_with_falsifier().await;
    let fixed_point = |body: &Value| body["result"]["meta"]["graph_op"]["fixed_point"].clone();
    let blocks = |fp: &Value| {
        [
            "scc_count",
            "trivial_scc_count",
            "cyclic_scc_count",
            "max_scc_size",
        ]
        .map(|k| {
            fp[k]
                .as_u64()
                .unwrap_or_else(|| panic!("{k} missing: {fp}"))
        })
    };

    // gamma 0: the falsifies edge is left out; the loop validates as before.
    let (status, body) = http_zero(&engine, evolve_request(json!({"gamma": 0.0}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["result"]["verb"], "graph_evolve");
    let fp = fixed_point(&body);
    assert_eq!(fp["gamma"], 0.0);
    assert_eq!(
        (&fp["dependency_edges"], &fp["falsification_edges"]),
        (&json!(4), &json!(0))
    );
    assert_eq!(blocks(&fp), [3, 2, 1, 3]);
    assert_eq!(fp["contraction"], f64::from(0.85_f32));
    let iterations = fp["iterations"].as_u64().unwrap();
    assert!(iterations >= 1 && iterations <= fp["k_max"].as_u64().unwrap());
    assert_eq!(fp["node_updates"], 2 + 3 * (iterations + 1));
    let validated = statuses(&engine).await;
    assert!(validated
        .iter()
        .all(|(label, status)| label == "safety invariant" || status == "validated"));

    // Default gamma (1), dry run: reported, rolled back.
    let (status, body) = http_zero(&engine, evolve_request(json!({"dry_run": true}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let fp = fixed_point(&body);
    assert_eq!(fp["gamma"], 1.0);
    assert_eq!(fp["falsification_edges"], 1);
    assert_eq!(fp["transitions_total"], 3);
    assert_eq!(statuses(&engine).await, validated);

    // gamma 1 applied. vent = (1 - beta) 0.5 + beta max(0, (1 + c31) / 2 - 1)
    // = (1 - beta) 0.5, since c31 <= 1: the loop falls and "vent" is revoked.
    let (status, body) = http_zero(&engine, evolve_request(json!({"gamma": 1.0}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let fp = fixed_point(&body);
    assert_eq!(blocks(&fp), [3, 2, 1, 3]);
    let vent = fp["transitions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["label"] == "vent")
        .unwrap_or_else(|| panic!("no transition for vent: {fp}"));
    assert_eq!(vent["to"], "falsified");
    let k = 1.0 - f64::from(0.85_f32);
    assert!((vent["confidence"].as_f64().unwrap() - 0.5 * k).abs() < 1e-6);
    assert_eq!(fp["revoked_entities"].as_array().unwrap().len(), 3);
    assert!(statuses(&engine).await.iter().all(|(label, status)| [
        "safety invariant",
        "leak report"
    ]
    .contains(&label.as_str())
        || status == "falsified"));
    let stopped = run(&engine, simulate_vent()).await;
    assert_eq!(
        stopped.meta["simulation"]["trajectory"][0]["tier"],
        "HardStop"
    );

    // Refusals commit nothing: a negative, an overflowing and a non-contractive gamma.
    let falsified = statuses(&engine).await;
    for bad in [json!(-1.0), json!(1e39)] {
        let (status, body) = http_zero(&engine, evolve_request(json!({"gamma": bad}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
    }
    // 31 also falsifies vent: row vent weighs 1/2 (31 in P+) + gamma 1/2 (31 in
    // P-) inside the loop, so gamma 2 gives q = 1.5 beta > 1.
    let out = run(
        &engine,
        json!({"action": "graph_deposit", "graph": {"edges": [
            {"source": {"entity_id": 31}, "target": {"action": "vent"}, "type": "falsifies", "weight": 1.0},
        ]}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let (status, body) = http_zero(&engine, evolve_request(json!({"gamma": 2.0}))).await;
    assert_eq!(status.as_u16(), 422, "{body}");
    assert!(
        body.to_string().contains("FixedPointNotContractive"),
        "{body}"
    );
    let direct = run(&engine, evolve_request(json!({"gamma": 2.0}))).await;
    assert_eq!(code(&direct), "FixedPointNotContractive");
    assert_eq!(statuses(&engine).await, falsified);

    // gamma 0 lifts the falsifiers again and reopens the gate.
    let (status, body) = http_zero(&engine, evolve_request(json!({"gamma": 0.0}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        fixed_point(&body)["reinstated_entities"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(statuses(&engine).await, validated);
    let reopened = run(&engine, simulate_vent()).await;
    assert_eq!(
        reopened.meta["simulation"]["trajectory"][0]["tier"],
        "Proceed"
    );
}

/// `graph_prune` runs the same signed evolution and takes the same `gamma`.
#[tokio::test]
async fn prune_takes_gamma_and_reports_it() {
    let (engine, _dir) = engine_with_falsifier().await;
    let out = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"entity_id": 30, "gamma": 0.25, "dry_run": true}}),
    )
    .await;
    assert!(!out.is_error, "{:?}", out.meta);
    let fp = &out.meta["graph_op"]["fixed_point"];
    assert_eq!(fp["gamma"], 0.25);
    assert_eq!(fp["falsification_edges"], 1);
    assert_eq!(fp["pinned"], 2);
    let bad = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"entity_id": 30, "gamma": -0.5}}),
    )
    .await;
    assert_eq!(code(&bad), "InvalidParams");
}

fn persistent_config(dir: &std::path::Path) -> ZeroEngineConfig {
    ZeroEngineConfig {
        graph_persist_dir: Some(dir.to_path_buf()),
        ..Default::default()
    }
}

#[tokio::test]
async fn durable_graph_restart_preserves_prune_and_evolve_gate_effects() {
    let dir = tempfile::tempdir().unwrap();
    let config = persistent_config(dir.path());
    let engine = PolymorphicZeroEngine::try_from_config(config.clone())
        .unwrap()
        .with_semantic(None);
    let out = deposit_world(&engine).await;
    assert!(!out.is_error, "{out:?}");
    assert_eq!(out.meta["graph_op"]["graph"]["persisted"], true);
    assert!(PolymorphicZeroEngine::try_from_config(config.clone()).is_err());
    let out = run(
        &engine,
        json!({"action":"graph_prune","graph":{"entity_id":7}}),
    )
    .await;
    assert!(!out.is_error, "{out:?}");
    drop(engine);
    let started = std::time::Instant::now();
    let engine = PolymorphicZeroEngine::try_from_config(config.clone())
        .unwrap()
        .with_semantic(None);
    eprintln!(
        "engine restart sample: nodes=3 elapsed_us={}",
        started.elapsed().as_micros()
    );
    let out = run(&engine, decide(&[7])).await;
    assert!(out.is_error);
    assert_eq!(code(&out), "NoFeasibleAction");
    let out = run(
        &engine,
        json!({"action":"graph_evolve","graph":{"retract":[{"entity_id":7}],"theta_hi":0.4}}),
    )
    .await;
    assert!(!out.is_error, "{out:?}");
    drop(engine);
    let engine = PolymorphicZeroEngine::try_from_config(config)
        .unwrap()
        .with_semantic(None);
    assert!(!run(&engine, decide(&[7])).await.is_error);
    drop(engine);
    std::fs::write(dir.path().join("CURRENT.sha256"), b"damaged snapshot").unwrap();
    assert!(PolymorphicZeroEngine::try_from_config(persistent_config(dir.path())).is_err());
}

#[tokio::test]
async fn failed_graph_disk_commit_blocks_subsequent_service_decisions() {
    let dir = tempfile::tempdir().unwrap();
    let engine = PolymorphicZeroEngine::try_from_config(persistent_config(dir.path()))
        .unwrap()
        .with_semantic(None);
    assert!(!deposit_world(&engine).await.is_error);
    std::fs::create_dir(dir.path().join("WRITE.tmp")).unwrap();
    let failure = engine
        .execute(&json!({"action":"graph_prune","graph":{"entity_id":7}}))
        .await;
    assert!(failure.is_err(), "{failure:?}");
    assert!(engine.execute(&decide(&[7])).await.is_err());
}

#[test]
fn failed_startup_seed_never_commits_an_empty_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let seed = dir.path().join("seed.json");
    let store = dir.path().join("graph");
    std::fs::write(&seed, "not json").unwrap();
    let config = ZeroEngineConfig {
        graph_seed_path: Some(seed.clone()),
        ..persistent_config(&store)
    };
    assert!(PolymorphicZeroEngine::try_from_config(config.clone()).is_err());
    assert!(!store.join("CURRENT.sha256").exists());
    assert!(PolymorphicZeroEngine::try_from_config(config.clone()).is_err());
    std::fs::write(
        &seed,
        json!({"nodes":[node(json!({"entity_id":7}),"seed fact","validated",1)]}).to_string(),
    )
    .unwrap();
    let engine = PolymorphicZeroEngine::try_from_config(config.clone()).unwrap();
    drop(engine);
    // Recovery uses the committed graph even if the original seed is gone.
    std::fs::remove_file(seed).unwrap();
    assert!(PolymorphicZeroEngine::try_from_config(config).is_ok());
}
