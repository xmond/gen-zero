//! The graph verbs through the real `zero` entry (`PolymorphicZeroEngine::execute`),
//! and their effect on the other verbs that share the engine's live LodGraph:
//! a `graph_prune` must hard-stop the pruned actions in `pipeline decide`,
//! `simulate` and `audit`, a `graph_evolve` that retracts the evidence must
//! lift that stop, `pipeline decide` must carry the PPR context of the chosen
//! action, and the engine's graph geometry must decide `graph_recall`.

use gen_zero_lod::GeometryParams;
use gen_zero_service::zero::ZeroEngineConfig;
use gen_zero_service::{PolymorphicZeroEngine, ZeroToolOutcome, ZeroVerb};
use serde_json::{json, Value};

const DIM: usize = 1024;
/// `gen_zero_gate::REVOCATION_RULE_ID`.
const REVOCATION_RULE: u32 = u32::MAX;

fn engine() -> PolymorphicZeroEngine {
    PolymorphicZeroEngine::new().with_bridge(None)
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
    .with_bridge(None);
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
    .with_bridge(None);
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
    // The same refusal inside a prune leaves the root unrevoked too.
    let prune = run(
        &engine,
        json!({"action": "graph_prune", "graph": {"action": "vent", "max_steps": 1}}),
    )
    .await;
    assert_eq!(code(&prune), "FixedPointDiverged");
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
    .with_bridge(None);
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
        .with_bridge(None);
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
