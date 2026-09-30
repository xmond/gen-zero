//! Graph fact verbs over the engine's live [`LodGraph`], the same graph the
//! PolicyGate reads for revocations in `ask`, `pipeline` and the world-model verbs:
//!
//! - `graph_deposit`: append nodes and edges in one transaction, then flush them
//!   into the CSR snapshot. Any failure rolls the whole deposit back.
//! - `graph_recall`: two-stage HDC + manifold recall.
//! - `graph_ppr`: Personalized PageRank diffusion from seed entities.
//! - `graph_prune`: causal cascade prune; `dry_run` reports the prune and rolls
//!   it back.
//!
//! Request block: `{"graph": {...}}`. Unknown keys are refused. Nodes are named by
//! `entity_id`, or by `action`, whose entity id is `action_id(action)`: the key
//! the gate checks, so a pruned action is hard-stopped everywhere.
//!
//! The graph lives in process memory. Deposits and prunes are not persisted; a
//! restart keeps only what the seed file (`GENZERO_GRAPH_SEED`) loads.

use crate::cognitive::Rejection;
use crate::zero::action_id;
use gen_zero_lod::{
    EdgeType, EpistemicStatus, LodBand, LodError, LodGraph, LodNode, MixedCurvatureCoord,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

const STAGE: &str = "graph";

/// Per-request and total caps. The verbs mutate gate-relevant state, so one
/// caller must not be able to exhaust memory.
pub const MAX_DEPOSIT_NODES: usize = 1024;
pub const MAX_DEPOSIT_EDGES: usize = 4096;
pub const MAX_GRAPH_NODES: usize = 1 << 20;
pub const MAX_TOP_K: usize = 256;
pub const MAX_PPR_SEEDS: usize = 64;
pub const MAX_PPR_ITERS: usize = 1000;
const MAX_LABEL_BYTES: usize = 256;

/// PPR parameters used when a request names none. Echoed in every response.
pub const DEFAULT_PPR_ALPHA: f32 = 0.15;
pub const DEFAULT_PPR_MAX_ITERS: usize = 100;
pub const DEFAULT_PPR_TOLERANCE: f32 = 1e-6;

/// The four graph operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphOp {
    Deposit,
    Recall,
    Ppr,
    Prune,
}

impl GraphOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Deposit => "graph_deposit",
            Self::Recall => "graph_recall",
            Self::Ppr => "graph_ppr",
            Self::Prune => "graph_prune",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EntityRef {
    entity_id: Option<u64>,
    action: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoordSpec {
    hyperbolic: [f32; 4],
    spherical: [f32; 4],
    euclidean: [f32; 8],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeSpec {
    entity_id: Option<u64>,
    action: Option<String>,
    label: String,
    band: u8,
    status: String,
    coord: CoordSpec,
    hdc: [u64; 4],
    confidence: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgeSpec {
    source: EntityRef,
    target: EntityRef,
    #[serde(rename = "type")]
    edge_type: String,
    weight: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DepositSpec {
    #[serde(default)]
    nodes: Vec<NodeSpec>,
    #[serde(default)]
    edges: Vec<EdgeSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecallSpec {
    coord: CoordSpec,
    hdc: [u64; 4],
    top_k: usize,
    crag_margin: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedSpec {
    entity_id: Option<u64>,
    action: Option<String>,
    weight: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PprSpec {
    seeds: Vec<SeedSpec>,
    top_k: usize,
    alpha: Option<f32>,
    max_iters: Option<usize>,
    tolerance: Option<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PruneSpec {
    entity_id: Option<u64>,
    action: Option<String>,
    dry_run: Option<bool>,
}

fn invalid(detail: impl Into<String>) -> Rejection {
    Rejection::invalid(STAGE, detail)
}

/// Input faults are 400, a duplicate entity 409, a missing entity 404. A CSR or
/// checkpoint failure is an engine fault, 500.
fn graph_rejection(e: LodError) -> Rejection {
    let (code, status) = match &e {
        LodError::DuplicateEntity(_) => ("DuplicateEntity", 409),
        LodError::EntityNotFound(_) | LodError::NodeNotFound(_) => ("EntityNotFound", 404),
        LodError::CsrInvariant(_) | LodError::FlushConflict | LodError::CheckpointRejected(_) => {
            ("GraphError", 500)
        }
        _ => ("InvalidParams", 400),
    };
    Rejection {
        code: code.to_string(),
        stage: STAGE.to_string(),
        detail: e.to_string(),
        http_status: status,
    }
}

/// Exactly one of `entity_id` / `action` names an entity.
fn entity(entity_id: Option<u64>, action: Option<&str>, what: &str) -> Result<u64, Rejection> {
    match (entity_id, action.map(str::trim)) {
        (Some(id), None) => Ok(id),
        (None, Some(name)) if !name.is_empty() => Ok(u64::from(action_id(name).0)),
        _ => Err(invalid(format!(
            "{what} needs exactly one of `entity_id` or a non-empty `action`"
        ))),
    }
}

fn coord(spec: &CoordSpec) -> Result<MixedCurvatureCoord, Rejection> {
    MixedCurvatureCoord::new(spec.hyperbolic, spec.spherical, spec.euclidean)
        .map_err(graph_rejection)
}

fn parse_status(name: &str, allow_axiomatic: bool) -> Result<EpistemicStatus, Rejection> {
    match name {
        "hypothesized" => Ok(EpistemicStatus::Hypothesized),
        "validated" => Ok(EpistemicStatus::Validated),
        "falsified" => Ok(EpistemicStatus::Falsified),
        "axiomatic" if allow_axiomatic => Ok(EpistemicStatus::Axiomatic),
        "axiomatic" => Err(invalid(
            "status `axiomatic` is accepted only from the operator seed file: an axiom \
             can never be pruned",
        )),
        other => Err(invalid(format!(
            "unknown status `{other}`; expected hypothesized | validated | falsified"
        ))),
    }
}

fn parse_band(band: u8) -> Result<LodBand, Rejection> {
    match band {
        0 => Ok(LodBand::Lod0Atomic),
        1 => Ok(LodBand::Lod1Cluster),
        2 => Ok(LodBand::Lod2Milestone),
        3 => Ok(LodBand::Lod3Systemic),
        _ => Err(invalid(format!("band must be 0..=3, got {band}"))),
    }
}

fn parse_edge_type(name: &str) -> Result<EdgeType, Rejection> {
    Ok(match name {
        "validates" => EdgeType::Validates,
        "falsifies" => EdgeType::Falsifies,
        "causal_transition" => EdgeType::CausalTransition,
        "semantic" => EdgeType::Semantic,
        "coarse_grain" => EdgeType::CoarseGrain,
        "depends_on" => EdgeType::DependsOn,
        other => {
            return Err(invalid(format!(
                "unknown edge type `{other}`; expected validates | falsifies | \
                 causal_transition | semantic | coarse_grain | depends_on"
            )))
        }
    })
}

fn status_name(status: EpistemicStatus) -> &'static str {
    match status {
        EpistemicStatus::Hypothesized => "hypothesized",
        EpistemicStatus::Validated => "validated",
        EpistemicStatus::Falsified => "falsified",
        EpistemicStatus::Axiomatic => "axiomatic",
    }
}

fn parse<T: for<'de> Deserialize<'de>>(block: &Value, op: GraphOp) -> Result<T, Rejection> {
    T::deserialize(block).map_err(|e| invalid(format!("invalid {} request: {e}", op.name())))
}

/// Size of the live graph, attached to every response.
fn graph_meta(graph: &LodGraph) -> Value {
    json!({
        "nodes": graph.node_count(),
        "csr_edges": graph.csr_snapshot().num_edges(),
        "pending_edges": graph.pending_edge_count(),
        "persisted": false,
    })
}

fn node_json(graph: &LodGraph, id: u32) -> Value {
    match graph.get_node(id) {
        Some(n) => json!({
            "node": id,
            "entity_id": n.entity_id,
            "label": n.label,
            "status": status_name(n.status),
        }),
        None => json!({"node": id, "missing": true}),
    }
}

/// Run one graph verb. `Ok` holds a one-line summary and the result object.
pub fn execute_graph(
    graph: &LodGraph,
    op: GraphOp,
    block: &Value,
) -> Result<(String, Value), Rejection> {
    let (summary, mut result) = match op {
        GraphOp::Deposit => deposit(graph, parse(block, op)?, false)?,
        GraphOp::Recall => recall(graph, parse(block, op)?)?,
        GraphOp::Ppr => ppr(graph, parse(block, op)?)?,
        GraphOp::Prune => prune(graph, parse(block, op)?)?,
    };
    result["op"] = json!(op.name());
    result["graph"] = graph_meta(graph);
    Ok((summary, result))
}

/// Load the operator seed file into `graph` as one transaction. The file has the
/// `graph_deposit` shape and may also carry `axiomatic` nodes. Any bad node or
/// edge fails the whole load and leaves the graph unchanged. The report carries
/// the file's blake3 `digest`.
pub fn load_seed(graph: &LodGraph, path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read graph seed {}: {e}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| format!("graph seed {} is not JSON: {e}", path.display()))?;
    let spec: DepositSpec = DepositSpec::deserialize(&value)
        .map_err(|e| format!("graph seed {}: {e}", path.display()))?;
    let (_, mut report) = deposit(graph, spec, true)
        .map_err(|r| format!("graph seed {}: {}", path.display(), r.detail))?;
    report["digest"] = json!(blake3::hash(text.as_bytes()).to_hex().to_string());
    Ok(report)
}

fn deposit(
    graph: &LodGraph,
    spec: DepositSpec,
    allow_axiomatic: bool,
) -> Result<(String, Value), Rejection> {
    if spec.nodes.is_empty() && spec.edges.is_empty() {
        return Err(invalid("graph_deposit needs at least one node or edge"));
    }
    // The seed file is operator-controlled; only API deposits are capped per request.
    if !allow_axiomatic
        && (spec.nodes.len() > MAX_DEPOSIT_NODES || spec.edges.len() > MAX_DEPOSIT_EDGES)
    {
        return Err(invalid(format!(
            "one deposit takes at most {MAX_DEPOSIT_NODES} nodes and {MAX_DEPOSIT_EDGES} edges"
        )));
    }
    // Parse everything before the transaction, so a malformed field never
    // reaches the graph.
    let mut nodes = Vec::with_capacity(spec.nodes.len());
    for (i, n) in spec.nodes.iter().enumerate() {
        let entity_id = entity(n.entity_id, n.action.as_deref(), &format!("nodes[{i}]"))?;
        let label = n.label.trim();
        if label.is_empty() || label.len() > MAX_LABEL_BYTES {
            return Err(invalid(format!(
                "nodes[{i}].label must be 1..={MAX_LABEL_BYTES} bytes"
            )));
        }
        let mut node = LodNode::new(0, parse_band(n.band)?, coord(&n.coord)?, label, entity_id)
            .with_status(parse_status(&n.status, allow_axiomatic)?)
            .with_hdc_fingerprint(n.hdc);
        node.confidence = n.confidence;
        nodes.push(node);
    }
    let mut edges = Vec::with_capacity(spec.edges.len());
    for (i, e) in spec.edges.iter().enumerate() {
        let source = entity(
            e.source.entity_id,
            e.source.action.as_deref(),
            &format!("edges[{i}].source"),
        )?;
        let target = entity(
            e.target.entity_id,
            e.target.action.as_deref(),
            &format!("edges[{i}].target"),
        )?;
        edges.push((source, target, parse_edge_type(&e.edge_type)?, e.weight));
    }

    let (node_ids, tickets, flush) = graph
        .transact(|g| {
            if g.node_count() + nodes.len() > MAX_GRAPH_NODES {
                return Err(LodError::InvalidNode(format!(
                    "the live graph is capped at {MAX_GRAPH_NODES} nodes"
                )));
            }
            let mut ids = Vec::with_capacity(nodes.len());
            for node in nodes {
                ids.push(g.add_node(node)?);
            }
            let mut tickets = Vec::with_capacity(edges.len());
            for &(source, target, edge_type, weight) in &edges {
                let s = g
                    .node_for_entity(source)
                    .ok_or(LodError::EntityNotFound(source))?;
                let t = g
                    .node_for_entity(target)
                    .ok_or(LodError::EntityNotFound(target))?;
                tickets.push(g.add_edge(s, t, edge_type, weight)?);
            }
            let flush = g.flush_edges_to_csr()?;
            Ok((ids, tickets, flush))
        })
        .map_err(graph_rejection)?;

    let summary = format!(
        "graph_deposit: {} node(s), {} edge(s) committed; CSR now {} edge(s)",
        node_ids.len(),
        tickets.len(),
        flush.csr_edges
    );
    let deposited: Vec<Value> = node_ids.iter().map(|&id| node_json(graph, id)).collect();
    Ok((
        summary,
        json!({
            "nodes": deposited,
            "edge_tickets": tickets,
            "flush": {
                "merged_edges": flush.merged_edges,
                "csr_nodes": flush.csr_nodes,
                "csr_edges": flush.csr_edges,
                "pending_edges": flush.pending_edges,
            },
        }),
    ))
}

fn recall(graph: &LodGraph, spec: RecallSpec) -> Result<(String, Value), Rejection> {
    if spec.top_k == 0 || spec.top_k > MAX_TOP_K {
        return Err(invalid(format!("top_k must be 1..={MAX_TOP_K}")));
    }
    let query = coord(&spec.coord)?;
    let hits = graph
        .two_stage_recall(&query, &spec.hdc, spec.top_k, spec.crag_margin)
        .map_err(graph_rejection)?;
    let results: Vec<Value> = hits
        .iter()
        .map(|&(id, distance)| {
            let mut v = node_json(graph, id);
            v["distance"] = json!(distance);
            v
        })
        .collect();
    let summary = format!("graph_recall: {} live node(s) recalled", results.len());
    Ok((
        summary,
        json!({"results": results, "top_k": spec.top_k, "crag_margin": spec.crag_margin}),
    ))
}

fn ppr(graph: &LodGraph, spec: PprSpec) -> Result<(String, Value), Rejection> {
    if spec.seeds.is_empty() || spec.seeds.len() > MAX_PPR_SEEDS {
        return Err(invalid(format!(
            "seeds must hold 1..={MAX_PPR_SEEDS} entries"
        )));
    }
    if spec.top_k == 0 || spec.top_k > MAX_TOP_K {
        return Err(invalid(format!("top_k must be 1..={MAX_TOP_K}")));
    }
    let alpha = spec.alpha.unwrap_or(DEFAULT_PPR_ALPHA);
    let max_iters = spec.max_iters.unwrap_or(DEFAULT_PPR_MAX_ITERS);
    let tolerance = spec.tolerance.unwrap_or(DEFAULT_PPR_TOLERANCE);
    if max_iters > MAX_PPR_ITERS {
        return Err(invalid(format!(
            "max_iters must be at most {MAX_PPR_ITERS}"
        )));
    }
    let mut seeds = Vec::with_capacity(spec.seeds.len());
    for (i, s) in spec.seeds.iter().enumerate() {
        let e = entity(s.entity_id, s.action.as_deref(), &format!("seeds[{i}]"))?;
        let node = graph
            .node_for_entity(e)
            .ok_or_else(|| graph_rejection(LodError::EntityNotFound(e)))?;
        seeds.push((node, s.weight));
    }
    let ranking = graph
        .query_ppr(&seeds, alpha, max_iters, tolerance)
        .map_err(graph_rejection)?;
    let results: Vec<Value> = ranking
        .ranked
        .iter()
        .take(spec.top_k)
        .map(|&(id, score)| {
            let mut v = node_json(graph, id);
            v["score"] = json!(score);
            v
        })
        .collect();
    let summary = format!(
        "graph_ppr: {} node(s) ranked after {} iteration(s), converged {}",
        results.len(),
        ranking.iterations,
        ranking.converged
    );
    Ok((
        summary,
        json!({
            "results": results,
            "alpha": alpha,
            "max_iters": max_iters,
            "tolerance": tolerance,
            "iterations": ranking.iterations,
            "residual": ranking.residual,
            "converged": ranking.converged,
        }),
    ))
}

fn prune(graph: &LodGraph, spec: PruneSpec) -> Result<(String, Value), Rejection> {
    let e = entity(spec.entity_id, spec.action.as_deref(), "graph_prune")?;
    let dry_run = spec.dry_run.unwrap_or(false);
    let node = graph
        .node_for_entity(e)
        .ok_or_else(|| graph_rejection(LodError::EntityNotFound(e)))?;
    let (outcome, pruned) = graph
        .transact(|g| {
            let outcome = g.cascade_prune_and_rollback(node)?;
            // Read the labels while the prune is still applied.
            let pruned: Vec<Value> = outcome.pruned.iter().map(|&id| node_json(g, id)).collect();
            if dry_run {
                g.rollback_checkpoint(&outcome.checkpoint)?;
            }
            Ok((outcome, pruned))
        })
        .map_err(graph_rejection)?;
    let summary = format!(
        "graph_prune: {} node(s) {} from entity {e}",
        outcome.pruned.len(),
        if dry_run {
            "would be pruned (dry run, rolled back)"
        } else {
            "pruned and revoked"
        }
    );
    Ok((
        summary,
        json!({
            "root_entity": e,
            "pruned": pruned,
            "revoked_entities": outcome.revoked_entities,
            "retracted_dependencies": outcome.retracted_dependencies,
            "dry_run": dry_run,
            "applied": !dry_run,
        }),
    ))
}
