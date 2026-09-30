//! The graph verbs through the real `zero` entry (`PolymorphicZeroEngine::execute`),
//! and their effect on the other verbs that share the engine's live LodGraph:
//! a `graph_prune` must hard-stop the pruned actions in `pipeline decide`,
//! `simulate` and `audit`, and `pipeline decide` must carry the PPR context of
//! the chosen action.

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
