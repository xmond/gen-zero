//! LOD-Graph + Qwen dense track through the real `zero` entry
//! (`PolymorphicZeroEngine::execute`).
//!
//! The tests without weights run by default: the 896-wide dense projection,
//! and the fail-closed reports of an engine with no backend or with the
//! Python bridge. The weight-backed tests are ignored by default and run with:
//!
//! ```text
//! GENZERO_QWEN_MODEL_PATH=/path/Qwen2.5-0.5B.Q8_0.gguf \
//! GENZERO_QWEN_TOKENIZER_PATH=/path/tokenizer.json \
//! GENZERO_PYTHON_ENDPOINT=off \
//! cargo test --release -p gen-zero-service --test lod_qwen_embedding_tests -- --ignored --nocapture
//! ```
//!
//! A missing variable is a test failure, never a silent pass.

use gen_zero_lod::{LodGraph, DENSE_PROJECTOR_VERSION};
use gen_zero_service::zero::ZeroEngineConfig;
use gen_zero_service::{
    BridgeConfig, PolymorphicZeroEngine, SemanticBackend, SemanticBridgeClient, ZeroToolOutcome,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

const QWEN_DIM: usize = 896;

/// Deterministic pseudo-random unit vector (splitmix64).
fn unit_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed;
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn hamming(a: &[u64; 4], b: &[u64; 4]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

async fn run(engine: &PolymorphicZeroEngine, req: Value) -> ZeroToolOutcome {
    engine.execute(&req).await.expect("engine call")
}

fn graph_op(out: &ZeroToolOutcome) -> &Value {
    assert!(out.rejection.is_none(), "refused: {:?}", out.rejection);
    &out.meta["graph_op"]
}

fn payload_node(entity_id: u64, label: &str, payload: &str) -> Value {
    json!({
        "entity_id": entity_id, "label": label, "band": 0, "status": "validated",
        "confidence": 0.9, "payload": payload,
    })
}

async fn deposit(engine: &PolymorphicZeroEngine, nodes: Vec<Value>) -> Value {
    let out = run(
        engine,
        json!({"action": "graph_deposit", "graph": {"nodes": nodes}}),
    )
    .await;
    graph_op(&out).clone()
}

async fn rag(engine: &PolymorphicZeroEngine, graph: Value) -> Value {
    let out = run(engine, json!({"action": "graph_rag", "graph": graph})).await;
    graph_op(&out).clone()
}

fn hit_entities(op: &Value) -> Vec<u64> {
    op["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["entity_id"].as_u64().unwrap())
        .collect()
}

/// An 896-wide Qwen-sized vector lands in the graph's chart: deterministic,
/// a near copy keeps most fingerprint bits and an unrelated vector does not,
/// and a graph holding 896-wide embeddings refuses a query of another width.
#[test]
fn project_dense_maps_896_wide_vectors_into_the_chart() {
    let graph = LodGraph::new();
    let v = unit_vector(7, QWEN_DIM);
    let (coord, hdc) = graph.project_dense(&v).unwrap();
    let (coord_again, hdc_again) = graph.project_dense(&v).unwrap();
    assert_eq!(hdc, hdc_again);
    assert_eq!(format!("{coord:?}"), format!("{coord_again:?}"));

    let near: Vec<f32> = v
        .iter()
        .zip(unit_vector(8, QWEN_DIM))
        .map(|(a, b)| a + 0.05 * b)
        .collect();
    let (_, near_hdc) = graph.project_dense(&near).unwrap();
    let (_, far_hdc) = graph.project_dense(&unit_vector(9, QWEN_DIM)).unwrap();
    let (d_near, d_far) = (hamming(&hdc, &near_hdc), hamming(&hdc, &far_hdc));
    assert!(d_near < 40, "near copy flipped {d_near} of 256 bits");
    assert!(
        d_far > 90,
        "unrelated vector flipped only {d_far} of 256 bits"
    );

    assert!(graph.project_dense(&vec![0.0; QWEN_DIM]).is_err());
    let mut bad = v.clone();
    bad[3] = f32::NAN;
    assert!(graph.project_dense(&bad).is_err());

    assert_eq!(graph.embedding_dim(), None);
    let node = gen_zero_lod::LodNode::new(0, gen_zero_lod::LodBand::Lod0Atomic, coord, "n", 11)
        .with_hdc_fingerprint(hdc)
        .with_embedding(v.clone());
    graph.add_node(node).unwrap();
    assert_eq!(graph.embedding_dim(), Some(QWEN_DIM));
    let err = graph
        .hybrid_rag_search_query(None, Some(&unit_vector(1, 384)), None, 1, 0.0, 0.15, 100)
        .unwrap_err();
    assert!(err.to_string().contains("896"), "{err}");
    let hit = graph
        .hybrid_rag_search_query(None, Some(&v), None, 1, 0.0, 0.15, 100)
        .unwrap();
    assert_eq!(hit.anchors.len(), 1);
}

/// No backend configured: deposits get no Qwen vector, a text query runs the
/// lexical track only, and both responses say so with the reason.
#[tokio::test]
async fn no_backend_is_lexical_only_and_says_so() {
    let engine = PolymorphicZeroEngine::new().with_semantic(None);
    let op = deposit(
        &engine,
        vec![payload_node(
            1,
            "pump",
            "Coolant pump seal replaced after leak",
        )],
    )
    .await;
    assert_eq!(op["qwen_embedded_nodes"], 0);
    assert!(op["qwen_embedder"].is_null());
    assert!(op["qwen_vector_dim"].is_null());
    let reason = op["qwen_skip_reason"].as_str().unwrap();
    assert!(reason.contains("GENZERO_QWEN_MODEL_PATH"), "{reason}");
    assert!(op["nodes"][0]["embedding_dim"].is_null());

    let op = rag(
        &engine,
        json!({"query_text": "coolant pump leak", "top_k": 1}),
    )
    .await;
    assert_eq!(op["qwen_embedded"], false);
    assert!(op["qwen_skip_reason"]
        .as_str()
        .unwrap()
        .contains("GENZERO_QWEN_MODEL_PATH"));
    assert_eq!(op["query"]["kind"], "text");
    assert!(op["query"]["vector_dim"].is_null());
    assert!(op["query"]["vector_source"].is_null());
    assert_eq!(hit_entities(&op), vec![1]);
    assert_eq!(op["hits"][0]["matched"], "primary");
}

/// A caller vector is searched as given: Qwen is not asked, no skip reason.
#[tokio::test]
async fn a_caller_query_vector_is_not_replaced() {
    let engine = PolymorphicZeroEngine::new().with_semantic(None);
    let v = unit_vector(3, QWEN_DIM);
    let mut node = payload_node(5, "valve", "Main valve opened");
    node["embedding"] = json!(v);
    let op = deposit(&engine, vec![node]).await;
    assert_eq!(op["qwen_embedded_nodes"], 0);
    // Every payload node carried its own embedding: nothing was there to embed.
    assert!(op["qwen_skip_reason"].is_null());
    assert_eq!(op["nodes"][0]["embedding_dim"], QWEN_DIM);

    let op = rag(
        &engine,
        json!({"query_text": "valve", "query_vector": v, "top_k": 1}),
    )
    .await;
    assert_eq!(op["qwen_embedded"], false);
    assert!(op["qwen_skip_reason"].is_null());
    assert_eq!(op["query"]["kind"], "text+vector");
    assert_eq!(op["query"]["vector_source"], "caller");
    assert_eq!(op["query"]["vector_dim"], QWEN_DIM);
}

/// The Python bridge has no text embedder: lexical only, reason named, and
/// the bridge is never called (its endpoint does not exist).
#[tokio::test]
async fn the_python_bridge_backend_is_reported_not_faked() {
    let client = SemanticBridgeClient::new(BridgeConfig::new("http://127.0.0.1:9")).unwrap();
    let backend = SemanticBackend::Remote(client);
    let err = backend.embed("anything").await.unwrap_err();
    assert!(err.to_string().contains("no text embedder"), "{err}");
    let engine = PolymorphicZeroEngine::new().with_semantic(Some(Arc::new(backend)));

    let op = deposit(&engine, vec![payload_node(1, "pump", "Coolant pump seal")]).await;
    assert_eq!(op["qwen_embedded_nodes"], 0);
    let reason = op["qwen_skip_reason"].as_str().unwrap();
    assert!(
        reason.contains("semantic_bridge") && reason.contains("no text embedder"),
        "{reason}"
    );

    let op = rag(&engine, json!({"query_text": "coolant pump", "top_k": 1})).await;
    assert_eq!(op["qwen_embedded"], false);
    assert!(op["qwen_skip_reason"]
        .as_str()
        .unwrap()
        .contains("no text embedder"));
    assert_eq!(op["query"]["kind"], "text");
}

fn native_engine() -> PolymorphicZeroEngine {
    let model = std::env::var("GENZERO_QWEN_MODEL_PATH")
        .expect("set GENZERO_QWEN_MODEL_PATH (see file header)");
    let tokenizer = std::env::var("GENZERO_QWEN_TOKENIZER_PATH")
        .ok()
        .map(PathBuf::from);
    let config = ZeroEngineConfig::from_env()
        .unwrap()
        .with_qwen_model(model, tokenizer);
    let engine = PolymorphicZeroEngine::try_from_config(config).expect("native engine");
    assert!(matches!(
        engine.semantic(),
        Some(SemanticBackend::Native(_))
    ));
    engine
}

/// The doctor/antibiotics fact, a lexical decoy that shares the query's
/// words but not its meaning, and two unrelated facts.
const TARGET: &str = "The physician prescribed antibiotics for the bacterial infection.";
const DECOY: &str = "The patient gave the doctor a birthday gift at the party.";
const MARKET: &str = "Shares dropped when the central bank raised borrowing costs.";
const GARDEN: &str = "She planted tomatoes in the garden this spring.";
/// Shares `doctor`, `gave`, `patient` with the decoy and no content word with the target.
const PARAPHRASE: &str = "A doctor gave the patient medicine to kill germs.";

fn corpus() -> Vec<Value> {
    vec![
        payload_node(1, "antibiotics", TARGET),
        payload_node(2, "gift", DECOY),
        payload_node(3, "market", MARKET),
        payload_node(4, "garden", GARDEN),
    ]
}

/// Deposits embed every payload with Qwen (896 wide) and a text query runs
/// both tracks. Paired with a lexical-only engine on the same corpus.
#[tokio::test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
async fn qwen_dense_track_finds_a_paraphrase_the_lexical_track_misses() {
    let native = native_engine();
    let lexical = PolymorphicZeroEngine::new().with_semantic(None);

    let started = std::time::Instant::now();
    let op = deposit(&native, corpus()).await;
    println!("deposit of 4 payloads with Qwen: {:?}", started.elapsed());
    assert_eq!(op["qwen_embedded_nodes"], 4);
    assert_eq!(op["qwen_vector_dim"], QWEN_DIM);
    assert!(op["qwen_skip_reason"].is_null());
    let embedder = op["qwen_embedder"].as_str().unwrap();
    assert!(embedder.ends_with("/final-norm-mean-pool-l2"), "{embedder}");
    assert_eq!(op["qwen_pooling"], "mean");
    for n in op["nodes"].as_array().unwrap() {
        assert_eq!(n["embedding_dim"], QWEN_DIM);
        assert_eq!(n["placement"], "chart");
    }
    deposit(&lexical, corpus()).await;

    // No shared content word with the target.
    let started = std::time::Instant::now();
    let q = rag(&native, json!({"query_text": PARAPHRASE, "top_k": 1})).await;
    println!("rag with Qwen query embedding: {:?}", started.elapsed());
    let control = rag(&lexical, json!({"query_text": PARAPHRASE, "top_k": 1})).await;
    println!("qwen anchors: {}", q["anchors"]);
    println!("lexical anchors: {}", control["anchors"]);
    assert_eq!(q["qwen_embedded"], true);
    assert!(q["qwen_skip_reason"].is_null());
    assert_eq!(q["qwen_embedder"], embedder);
    assert_eq!(q["qwen_pooling"], "mean");
    assert_eq!(q["query"]["kind"], "text+vector");
    assert_eq!(q["query"]["vector_source"], "qwen");
    assert_eq!(q["query"]["vector_dim"], QWEN_DIM);
    assert_eq!(q["query"]["dense_projector"], DENSE_PROJECTOR_VERSION);
    let embedding_anchor: Vec<&Value> = q["hits"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|h| h["matched"] == "embedding")
        .collect();
    assert_eq!(embedding_anchor.len(), 1, "{}", q["hits"]);
    assert_eq!(embedding_anchor[0]["entity_id"], 1, "{}", q["hits"]);
    assert_eq!(control["qwen_embedded"], false);
    assert!(
        !hit_entities(&control).contains(&1),
        "lexical control already finds the target: {}",
        control["hits"]
    );

    // Shared words with the target: both tracks agree on it.
    let q = rag(
        &native,
        json!({"query_text": "antibiotics for a bacterial infection", "top_k": 1}),
    )
    .await;
    let control = rag(
        &lexical,
        json!({"query_text": "antibiotics for a bacterial infection", "top_k": 1}),
    )
    .await;
    assert_eq!(q["qwen_embedded"], true);
    assert_eq!(hit_entities(&control), vec![1]);
    let anchors: Vec<(u64, &str)> = q["hits"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|h| h["via"] != "diffusion")
        .map(|h| {
            (
                h["entity_id"].as_u64().unwrap(),
                h["matched"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        anchors,
        vec![(1, "primary")],
        "one node found by both tracks counts once"
    );
}

/// Payloads over the per-request token budget are refused with 413 before any
/// forward pass, and the graph is unchanged: never cut, never lexical-only.
#[tokio::test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
async fn qwen_refuses_a_deposit_over_the_embedding_budget() {
    let native = native_engine();
    let long = "Coolant pump seal inspection log entry. ".repeat(600);
    let started = std::time::Instant::now();
    let out = run(
        &native,
        json!({"action": "graph_deposit", "graph": {"nodes": [payload_node(1, "log", &long)]}}),
    )
    .await;
    let rej = out.rejection.expect("over-budget deposit must be refused");
    assert_eq!(rej.code, "EmbedBudgetExceeded");
    assert_eq!(rej.http_status, 413);
    assert!(rej.detail.contains("4096"), "{}", rej.detail);
    // Refused on the token count alone: far less than one forward pass per window.
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
    let op = rag(&native, json!({"query_text": "coolant pump", "top_k": 1})).await;
    assert_eq!(op["searchable_nodes"], 0);
}

/// A graph whose embeddings are not 896 wide: Qwen is loaded and the
/// dimensions disagree, so the request is refused outright (fail-closed),
/// never silently downgraded to a lexical-only success.
#[tokio::test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
async fn qwen_rejects_a_graph_of_another_embedding_width() {
    let native = native_engine();
    let mut node = payload_node(1, "pump", "Coolant pump seal");
    node["embedding"] = json!(unit_vector(5, 384));
    deposit(&native, vec![node]).await;

    let out = run(
        &native,
        json!({
            "action": "graph_deposit",
            "graph": {"nodes": [payload_node(2, "valve", "Main valve opened")]},
        }),
    )
    .await;
    let rej = out
        .rejection
        .expect("a Qwen/graph dimension conflict must be rejected, not skipped to lexical-only");
    assert!(
        rej.detail.contains("384") && rej.detail.contains("896"),
        "{}",
        rej.detail
    );

    let out = run(
        &native,
        json!({"action": "graph_rag", "graph": {"query_text": "coolant pump", "top_k": 1}}),
    )
    .await;
    let rej = out
        .rejection
        .expect("a Qwen/graph dimension conflict must be rejected, not skipped to lexical-only");
    assert!(
        rej.detail.contains("384") && rej.detail.contains("896"),
        "{}",
        rej.detail
    );
}

/// Paired per query: the order of raw Qwen cosines and the order of the LOD
/// vector track over the same four payloads must agree, so the graph keeps
/// the signal the model gives. Prints both, with the Hamming distances of the
/// projected fingerprints that serve only the prefilter.
#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn lod_vector_track_keeps_the_qwen_cosine_order() {
    let model = std::env::var("GENZERO_QWEN_MODEL_PATH")
        .expect("set GENZERO_QWEN_MODEL_PATH (see file header)");
    let tokenizer = std::env::var("GENZERO_QWEN_TOKENIZER_PATH")
        .ok()
        .map(PathBuf::from);
    let scorer =
        gen_zero_model::QwenSemanticScorer::load(model.as_ref(), tokenizer.as_deref()).unwrap();
    let docs = [TARGET, DECOY, MARKET, GARDEN];
    let graph = LodGraph::new();
    let doc_vectors: Vec<Vec<f32>> = docs.iter().map(|d| scorer.embed(d).unwrap()).collect();
    for (i, v) in doc_vectors.iter().enumerate() {
        let (coord, hdc) = graph.project_dense(v).unwrap();
        let node =
            gen_zero_lod::LodNode::new(0, gen_zero_lod::LodBand::Lod0Atomic, coord, "d", i as u64)
                .with_hdc_fingerprint(hdc)
                .with_embedding(v.clone())
                .placed_by_embedding();
        graph.add_node(node).unwrap();
    }
    let queries = [
        PARAPHRASE,
        "medicine to kill germs",
        "How do doctors treat a bacterial illness?",
        "Which drug cures an infection caused by germs?",
    ];
    for q in queries {
        let qv = scorer.embed(q).unwrap();
        let cos: Vec<f32> = doc_vectors
            .iter()
            .map(|d| d.iter().zip(&qv).map(|(a, b)| a * b).sum())
            .collect();
        let mut by_cosine: Vec<u64> = (0..docs.len() as u64).collect();
        by_cosine.sort_by(|&a, &b| cos[b as usize].total_cmp(&cos[a as usize]));
        let (_, qh) = graph.project_dense(&qv).unwrap();
        let ham: Vec<u32> = doc_vectors
            .iter()
            .map(|d| hamming(&graph.project_dense(d).unwrap().1, &qh))
            .collect();
        let r = graph
            .hybrid_rag_search_query(None, Some(&qv), None, docs.len(), 0.0, 0.15, 100)
            .unwrap();
        let by_lod: Vec<u64> = r
            .anchors
            .iter()
            .map(|&(id, _)| graph.get_node(id).unwrap().entity_id)
            .collect();
        println!("{q:?}\n  cosine {cos:.3?} -> {by_cosine:?}\n  hamming {ham:?}\n  lod {by_lod:?}");
        assert_eq!(by_lod, by_cosine, "{q}");
    }
}

/// [`gen_zero_model::PoolingMode::LastToken`] is a real, working pooling, not
/// an unused enum variant: it returns a unit-norm vector distinct from
/// [`gen_zero_model::PoolingMode::Mean`]'s for the same text, and its id
/// string names it.
#[test]
#[ignore = "needs Qwen2.5-0.5B GGUF weights"]
fn last_token_pooling_is_wired_and_differs_from_mean() {
    use gen_zero_model::PoolingMode;

    let model = std::env::var("GENZERO_QWEN_MODEL_PATH")
        .expect("set GENZERO_QWEN_MODEL_PATH (see file header)");
    let tokenizer = std::env::var("GENZERO_QWEN_TOKENIZER_PATH")
        .ok()
        .map(PathBuf::from);
    let scorer =
        gen_zero_model::QwenSemanticScorer::load(model.as_ref(), tokenizer.as_deref()).unwrap();

    let mean = scorer
        .embed_with_pooling(TARGET, PoolingMode::Mean)
        .unwrap();
    let last = scorer
        .embed_with_pooling(TARGET, PoolingMode::LastToken)
        .unwrap();
    assert_eq!(mean.len(), QWEN_DIM);
    assert_eq!(last.len(), QWEN_DIM);
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm(&mean) - 1.0).abs() < 1e-4,
        "mean norm {}",
        norm(&mean)
    );
    assert!(
        (norm(&last) - 1.0).abs() < 1e-4,
        "last-token norm {}",
        norm(&last)
    );
    let cosine: f32 = mean.iter().zip(&last).map(|(a, b)| a * b).sum();
    assert!(
        cosine < 0.999,
        "mean and last-token pooling gave near-identical vectors: {cosine}"
    );

    assert_eq!(
        scorer.embedder_id_for(PoolingMode::Mean),
        scorer.embedder_id()
    );
    assert!(scorer
        .embedder_id_for(PoolingMode::Mean)
        .ends_with("/final-norm-mean-pool-l2"));
    assert!(scorer
        .embedder_id_for(PoolingMode::LastToken)
        .ends_with("/final-norm-last_token-pool-l2"));
}
