//! Graph fact verbs over the engine's live [`LodGraph`], the same graph the
//! PolicyGate reads for revocations in `ask`, `pipeline` and the world-model verbs:
//!
//! - `graph_deposit`: append nodes and edges in one transaction, then flush them
//!   into the CSR snapshot. Any failure rolls the whole deposit back.
//! - `graph_recall`: two-stage HDC + manifold recall under the graph's geometry.
//! - `graph_rag`: three-stage retrieval (`LodGraph::hybrid_rag_search`): HDC
//!   prefilter, geodesic rerank to anchors, PPR diffusion from the anchors.
//!   The query is text (`query_text`, lexical projection), a dense vector
//!   from an external embedding model (`query_vector`, dense projection), both
//!   at once, or a `coord` + `hdc` pair. Text and coordinates are compared
//!   with each node's own coordinate and its aliases; a vector only with node
//!   embeddings. Every hit carries its payload, source, digest and aliases,
//!   and names the anchor that matched.
//! - `graph_ppr`: Personalized PageRank diffusion from seed entities.
//! - `graph_prune`: record evidence against one entity, then evolve every
//!   confidence to the fixed point; dependents that fall below `theta_lo` are
//!   falsified and revoked. `dry_run` reports and rolls back.
//! - `graph_evolve`: optionally retract such evidence, then evolve every
//!   confidence to the fixed point. `dry_run` reports and rolls back.
//!
//!   Both evolutions take `gamma` (default 1), the gain of `falsifies` edges in
//!   `c = (1 - beta) pi + beta max(0, P+ c - gamma P- c)`, and solve it by
//!   strongly connected components in topological order. The response echoes
//!   `gamma` and the block counts (`scc_count`, `trivial_scc_count`,
//!   `cyclic_scc_count`, `max_scc_size`).
//! - `graph_coarse_grain`: insert a summary node for a cluster of member nodes
//!   on the band its coordinate implies (strictly coarser than every member),
//!   link each member to it with a `CoarseGrain` edge and flush. `dry_run`
//!   reports and rolls back.
//! - `graph_zoom`: move one node one band `in` or `out`, or `to_coord`: to the
//!   band its coordinate implies. The coarse-grain order is kept. `dry_run`
//!   reports and rolls back.
//!
//! A deposited node without `band` gets the band its coordinate implies
//! (`LodNode::derive_band_from_coord`).
//!
//! A deposited node may carry a `payload` (knowledge text, at most 64 KiB) with
//! an optional `source_uri` and `timestamp_ns` (default: the deposit's wall-clock
//! time, echoed). A node without `coord` and `hdc` is placed by projecting its
//! payload, or, with no payload, its `embedding`; it must then name its `band`,
//! because the depth of a hashed projection carries no hierarchy.
//!
//! A deposited node may carry `aliases` (other names: synonyms, translations)
//! and an `embedding`. Each alias is one more anchor for text queries, and
//! nodes that share an alias are linked by `semantic` edges both ways
//! (`alias_link_edges` in the response). The engine holds no embedding model:
//! the caller makes the vectors, all of one dimension per graph.
//!
//! The graph's geometry (curvature, sphere radius, metric weights) is fixed when
//! the engine starts (`GENZERO_GRAPH_GEOMETRY`) and echoed in every response.
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
    AnchorMatch, EdgeType, EpistemicStatus, FixedPointReport, HybridRagResult, LodBand, LodError,
    LodGraph, LodNode, MixedCurvatureCoord, Placement, ZoomDirection, DEFAULT_FALSIFICATION_GAIN,
    DENSE_PROJECTOR_VERSION, PROJECTOR_VERSION,
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
/// Most anchors one `graph_rag` asks for per track. A response holds at most
/// `3 * MAX_RAG_TOP_K` hits (text and vector together) of at most 64 KiB
/// payload each: 6 MiB.
pub const MAX_RAG_TOP_K: usize = 32;
/// CRAG margin of `graph_rag` when the request names none: 0 turns the
/// neighbor expansion off. Echoed in every response.
pub const DEFAULT_RAG_CRAG_MARGIN: f32 = 0.0;

/// PPR parameters used when a request names none. Echoed in every response.
pub const DEFAULT_PPR_ALPHA: f32 = 0.15;
pub const DEFAULT_PPR_MAX_ITERS: usize = 100;
pub const DEFAULT_PPR_TOLERANCE: f32 = 1e-6;

/// Confidence evolution parameters used when a request names none. Echoed in
/// every response. Uncalibrated presets.
pub const DEFAULT_EVOLVE_BETA: f32 = 0.85;
/// Falsification gain: how strongly a `falsifies` edge presses its target.
pub const DEFAULT_EVOLVE_GAMMA: f32 = DEFAULT_FALSIFICATION_GAIN;
pub const DEFAULT_EVOLVE_TOLERANCE: f32 = 1e-6;
pub const DEFAULT_EVOLVE_THETA_LO: f32 = 0.2;
pub const DEFAULT_EVOLVE_THETA_HI: f32 = 0.8;
/// Largest step budget one request may ask for, and the default.
pub const MAX_EVOLVE_STEPS: usize = 10_000;
/// Most evidence retractions in one `graph_evolve`.
pub const MAX_EVOLVE_RETRACTIONS: usize = 256;
/// Most status transitions listed in one response; the total is always reported.
pub const MAX_LISTED_TRANSITIONS: usize = 256;

/// The eight graph operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphOp {
    Deposit,
    Recall,
    Rag,
    Ppr,
    Prune,
    Evolve,
    CoarseGrain,
    Zoom,
}

impl GraphOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Deposit => "graph_deposit",
            Self::Recall => "graph_recall",
            Self::Rag => "graph_rag",
            Self::Ppr => "graph_ppr",
            Self::Prune => "graph_prune",
            Self::Evolve => "graph_evolve",
            Self::CoarseGrain => "graph_coarse_grain",
            Self::Zoom => "graph_zoom",
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
    /// Absent: the band the coordinate implies.
    band: Option<u8>,
    status: String,
    /// With `hdc`, or neither: then the payload is projected, or with no
    /// payload the embedding.
    coord: Option<CoordSpec>,
    hdc: Option<[u64; 4]>,
    confidence: f32,
    payload: Option<String>,
    source_uri: Option<String>,
    timestamp_ns: Option<u64>,
    #[serde(default)]
    aliases: Vec<String>,
    embedding: Option<Vec<f32>>,
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
struct RagSpec {
    /// `query_text`, `query_vector` or both; or `coord` with `hdc` alone.
    query_text: Option<String>,
    query_vector: Option<Vec<f32>>,
    coord: Option<CoordSpec>,
    hdc: Option<[u64; 4]>,
    top_k: usize,
    crag_margin: Option<f32>,
    alpha: Option<f32>,
    max_iters: Option<usize>,
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
    beta: Option<f32>,
    gamma: Option<f32>,
    tolerance: Option<f32>,
    theta_lo: Option<f32>,
    theta_hi: Option<f32>,
    max_steps: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvolveSpec {
    #[serde(default)]
    retract: Vec<EntityRef>,
    dry_run: Option<bool>,
    beta: Option<f32>,
    gamma: Option<f32>,
    tolerance: Option<f32>,
    theta_lo: Option<f32>,
    theta_hi: Option<f32>,
    max_steps: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CoarseGrainSpec {
    members: Vec<EntityRef>,
    entity_id: Option<u64>,
    action: Option<String>,
    coord: CoordSpec,
    hdc: [u64; 4],
    dry_run: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ZoomSpec {
    entity_id: Option<u64>,
    action: Option<String>,
    direction: String,
    dry_run: Option<bool>,
}

/// Resolved parameters of one confidence evolution.
#[derive(Clone, Copy)]
struct EvolveParams {
    beta: f32,
    gamma: f32,
    tolerance: f32,
    theta_lo: f32,
    theta_hi: f32,
    max_steps: usize,
}

impl EvolveParams {
    fn resolve(
        beta: Option<f32>,
        gamma: Option<f32>,
        tolerance: Option<f32>,
        theta_lo: Option<f32>,
        theta_hi: Option<f32>,
        max_steps: Option<usize>,
    ) -> Result<Self, Rejection> {
        let max_steps = max_steps.unwrap_or(MAX_EVOLVE_STEPS);
        if max_steps == 0 || max_steps > MAX_EVOLVE_STEPS {
            return Err(invalid(format!("max_steps must be 1..={MAX_EVOLVE_STEPS}")));
        }
        Ok(Self {
            beta: beta.unwrap_or(DEFAULT_EVOLVE_BETA),
            gamma: gamma.unwrap_or(DEFAULT_EVOLVE_GAMMA),
            tolerance: tolerance.unwrap_or(DEFAULT_EVOLVE_TOLERANCE),
            theta_lo: theta_lo.unwrap_or(DEFAULT_EVOLVE_THETA_LO),
            theta_hi: theta_hi.unwrap_or(DEFAULT_EVOLVE_THETA_HI),
            max_steps,
        })
    }

    fn run(self, graph: &LodGraph) -> Result<FixedPointReport, LodError> {
        graph.evolve_signed_epistemic_fixed_point_within(
            self.beta,
            self.gamma,
            self.tolerance,
            self.theta_lo,
            self.theta_hi,
            self.max_steps,
        )
    }
}

fn invalid(detail: impl Into<String>) -> Rejection {
    Rejection::invalid(STAGE, detail)
}

/// Input faults are 400 (blank text `EmptyInput`), an oversized payload 413, a
/// duplicate entity 409, a missing entity 404, a
/// confidence evolution that did not converge inside its step bound, or whose
/// map is not a contraction on some cycle, 422. A CSR
/// or checkpoint failure is an engine fault, 500.
fn graph_rejection(e: LodError) -> Rejection {
    let (code, status) = match &e {
        LodError::FixedPointDiverged { .. } => ("FixedPointDiverged", 422),
        LodError::FixedPointNotContractive { .. } => ("FixedPointNotContractive", 422),
        LodError::SpineBreatheOutOfBounds { .. } => ("BandOutOfRange", 409),
        LodError::DuplicateEntity(_) => ("DuplicateEntity", 409),
        LodError::EntityNotFound(_) | LodError::NodeNotFound(_) => ("EntityNotFound", 404),
        LodError::EmptyInput(_) => ("EmptyInput", 400),
        LodError::PayloadTooLarge { .. } => ("PayloadTooLarge", 413),
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

/// A request coordinate, checked against the ball of the graph's curvature.
fn coord(graph: &LodGraph, spec: &CoordSpec) -> Result<MixedCurvatureCoord, Rejection> {
    MixedCurvatureCoord::with_curvature(
        spec.hyperbolic,
        spec.spherical,
        spec.euclidean,
        graph.geometry().curvature as f32,
    )
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

fn band_level(band: LodBand) -> u8 {
    band as u8
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

fn placement_name(placement: Placement) -> &'static str {
    match placement {
        Placement::Chart => "chart",
        Placement::Embedding => "embedding",
    }
}

fn parse<T: for<'de> Deserialize<'de>>(block: &Value, op: GraphOp) -> Result<T, Rejection> {
    T::deserialize(block).map_err(|e| invalid(format!("invalid {} request: {e}", op.name())))
}

/// Size and geometry of the live graph, attached to every response.
fn graph_meta(graph: &LodGraph) -> Value {
    json!({
        "nodes": graph.node_count(),
        "csr_edges": graph.csr_snapshot().num_edges(),
        "pending_edges": graph.pending_edge_count(),
        "geometry": graph.geometry(),
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
            "band": band_level(n.band),
            "prior": n.prior,
            "confidence": n.confidence,
            "payload_bytes": n.payload.as_ref().map(String::len),
            "payload_digest": n.payload.as_ref().map(|_| digest_hex(&n.payload_digest)),
            "source_uri": n.source_uri,
            "timestamp_ns": n.payload.as_ref().map(|_| n.timestamp_ns),
            "aliases": n.aliases,
            "embedding_dim": n.embedding.as_ref().map(Vec::len),
            "placement": placement_name(n.placement),
        }),
        None => json!({"node": id, "missing": true}),
    }
}

fn digest_hex(digest: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*digest).to_hex().to_string()
}

/// Nanoseconds since the Unix epoch, the default version time of a payload.
/// A clock before the epoch or past u64 nanoseconds is an engine fault, 500.
fn now_ns() -> Result<u64, Rejection> {
    let clock_fault = |detail: String| Rejection {
        code: "GraphError".to_string(),
        stage: STAGE.to_string(),
        detail,
        http_status: 500,
    };
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| clock_fault(format!("system clock is before the Unix epoch: {e}")))?;
    u64::try_from(elapsed.as_nanos())
        .map_err(|_| clock_fault("system clock overflows u64 nanoseconds".into()))
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
        GraphOp::Rag => rag(graph, parse(block, op)?)?,
        GraphOp::Ppr => ppr(graph, parse(block, op)?)?,
        GraphOp::Prune => prune(graph, parse(block, op)?)?,
        GraphOp::Evolve => evolve(graph, parse(block, op)?)?,
        GraphOp::CoarseGrain => coarse_grain(graph, parse(block, op)?)?,
        GraphOp::Zoom => zoom(graph, parse(block, op)?)?,
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
        let projected = n.coord.is_none() && n.hdc.is_none();
        if projected && n.band.is_none() && (n.payload.is_some() || n.embedding.is_some()) {
            return Err(invalid(format!(
                "nodes[{i}] is placed by projecting its payload or embedding and must name \
                 its `band`: a hashed projection's depth carries no hierarchy"
            )));
        }
        let (place, hdc, placement) = match (&n.coord, n.hdc, &n.payload, &n.embedding) {
            (Some(c), Some(hdc), _, _) => (coord(graph, c)?, hdc, Placement::Chart),
            (None, None, Some(text), _) => {
                let (place, hdc) = graph.project_text(text).map_err(graph_rejection)?;
                (place, hdc, Placement::Chart)
            }
            (None, None, None, Some(embedding)) => {
                let (place, hdc) = graph.project_dense(embedding).map_err(graph_rejection)?;
                (place, hdc, Placement::Embedding)
            }
            (None, None, None, None) => {
                return Err(invalid(format!(
                    "nodes[{i}] needs `coord` and `hdc`, or a `payload` or an `embedding` to \
                     project"
                )))
            }
            _ => {
                return Err(invalid(format!(
                    "nodes[{i}] needs both `coord` and `hdc`, or neither"
                )))
            }
        };
        let mut node = LodNode::new(0, LodBand::Lod0Atomic, place, label, entity_id)
            .with_status(parse_status(&n.status, allow_axiomatic)?)
            .with_hdc_fingerprint(hdc)
            .with_prior(n.confidence)
            .with_aliases(n.aliases.iter().map(String::as_str));
        if let Some(embedding) = &n.embedding {
            node = node.with_embedding(embedding.clone());
        }
        if placement == Placement::Embedding {
            node = node.placed_by_embedding();
        }
        match &n.payload {
            Some(text) => {
                let timestamp_ns = match n.timestamp_ns {
                    Some(t) => t,
                    None => now_ns()?,
                };
                node = node
                    .with_payload(text.as_str(), n.source_uri.clone(), timestamp_ns)
                    .map_err(graph_rejection)?;
            }
            None if n.source_uri.is_some() || n.timestamp_ns.is_some() => {
                return Err(invalid(format!(
                    "nodes[{i}]: `source_uri` and `timestamp_ns` need a `payload`"
                )))
            }
            None => {}
        }
        node.band = match n.band {
            Some(band) => parse_band(band)?,
            None => node
                .derive_band_from_coord(&graph.geometry())
                .map_err(graph_rejection)?,
        };
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

    let (node_ids, alias_link_edges, tickets, flush) = graph
        .transact(|g| {
            if g.node_count() + nodes.len() > MAX_GRAPH_NODES {
                return Err(LodError::InvalidNode(format!(
                    "the live graph is capped at {MAX_GRAPH_NODES} nodes"
                )));
            }
            let mut ids = Vec::with_capacity(nodes.len());
            let pending_before = g.pending_edge_count();
            for node in nodes {
                ids.push(g.add_node(node)?);
            }
            // Inserts append edges only between nodes that share an alias.
            let alias_link_edges = g.pending_edge_count() - pending_before;
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
            Ok((ids, alias_link_edges, tickets, flush))
        })
        .map_err(graph_rejection)?;

    let summary = format!(
        "graph_deposit: {} node(s), {} edge(s) and {} alias link edge(s) committed; CSR now \
         {} edge(s)",
        node_ids.len(),
        tickets.len(),
        alias_link_edges,
        flush.csr_edges
    );
    let deposited: Vec<Value> = node_ids.iter().map(|&id| node_json(graph, id)).collect();
    Ok((
        summary,
        json!({
            "nodes": deposited,
            "edge_tickets": tickets,
            "alias_link_edges": alias_link_edges,
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
    let query = coord(graph, &spec.coord)?;
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

fn rag(graph: &LodGraph, spec: RagSpec) -> Result<(String, Value), Rejection> {
    if spec.top_k == 0 || spec.top_k > MAX_RAG_TOP_K {
        return Err(invalid(format!("top_k must be 1..={MAX_RAG_TOP_K}")));
    }
    let crag_margin = spec.crag_margin.unwrap_or(DEFAULT_RAG_CRAG_MARGIN);
    let alpha = spec.alpha.unwrap_or(DEFAULT_PPR_ALPHA);
    let max_iters = spec.max_iters.unwrap_or(DEFAULT_PPR_MAX_ITERS);
    if max_iters > MAX_PPR_ITERS {
        return Err(invalid(format!(
            "max_iters must be at most {MAX_PPR_ITERS}"
        )));
    }
    let (text, vector) = (spec.query_text.as_deref(), spec.query_vector.as_deref());
    let (result, query) = match (text.is_some() || vector.is_some(), &spec.coord, spec.hdc) {
        (true, None, None) => (
            graph
                .hybrid_rag_search_query(text, vector, spec.top_k, crag_margin, alpha, max_iters)
                .map_err(graph_rejection)?,
            json!({
                "kind": match (text, vector) {
                    (Some(_), Some(_)) => "text+vector",
                    (Some(_), None) => "text",
                    _ => "vector",
                },
                "projector": text.map(|_| PROJECTOR_VERSION),
                "dense_projector": vector.map(|_| DENSE_PROJECTOR_VERSION),
                "vector_dim": vector.map(<[f32]>::len),
            }),
        ),
        (false, Some(c), Some(hdc)) => (
            graph
                .hybrid_rag_search(
                    &coord(graph, c)?,
                    &hdc,
                    spec.top_k,
                    crag_margin,
                    alpha,
                    max_iters,
                )
                .map_err(graph_rejection)?,
            json!({"kind": "coord"}),
        ),
        _ => {
            return Err(invalid(
                "graph_rag needs `query_text`, `query_vector` or both, or else `coord` with \
                 `hdc` alone",
            ))
        }
    };
    let summary = rag_summary(&result);
    let HybridRagResult {
        hits,
        anchors,
        stage1_candidates,
        searchable_nodes,
        diffusion,
    } = result;
    let hits: Vec<Value> = hits
        .into_iter()
        .map(|h| {
            json!({
                "node": h.node_id,
                "entity_id": h.entity_id,
                "label": h.label,
                "status": status_name(h.status),
                "band": band_level(h.band),
                "confidence": h.confidence,
                "ppr_score": h.ppr_score,
                "anchor_distance": h.anchor_distance,
                "via": if h.anchor_distance.is_some() { "anchor" } else { "diffusion" },
                "matched": h.anchor_match.map(|m| match m {
                    AnchorMatch::Primary => "primary",
                    AnchorMatch::Alias(_) => "alias",
                    AnchorMatch::Embedding => "embedding",
                }),
                "matched_alias": match h.anchor_match {
                    Some(AnchorMatch::Alias(i)) => h.aliases.get(i).cloned(),
                    _ => None,
                },
                "aliases": h.aliases,
                "payload_digest": h.payload.as_ref().map(|_| digest_hex(&h.payload_digest)),
                "timestamp_ns": h.payload.as_ref().map(|_| h.timestamp_ns),
                "payload": h.payload,
                "source_uri": h.source_uri,
            })
        })
        .collect();
    let anchors: Vec<Value> = anchors
        .iter()
        .map(|&(id, distance)| {
            let mut v = node_json(graph, id);
            v["distance"] = json!(distance);
            v
        })
        .collect();
    let diffusion = diffusion.map(|d| {
        json!({
            "alpha": d.alpha,
            "max_iters": d.max_iters,
            "tolerance": d.tolerance,
            "iterations": d.iterations,
            "residual": d.residual,
            "converged": d.converged,
        })
    });
    Ok((
        summary,
        json!({
            "query": query,
            "hits": hits,
            "anchors": anchors,
            "stage1_candidates": stage1_candidates,
            "searchable_nodes": searchable_nodes,
            "diffusion": diffusion,
            "top_k": spec.top_k,
            "crag_margin": crag_margin,
        }),
    ))
}

fn rag_summary(result: &HybridRagResult) -> String {
    match &result.diffusion {
        None => "graph_rag: no live node to recall".to_string(),
        Some(d) => format!(
            "graph_rag: {} hit(s) from {} anchor(s); PPR {} after {} iteration(s)",
            result.hits.len(),
            result.anchors.len(),
            if d.converged {
                "converged"
            } else {
                "NOT converged"
            },
            d.iterations
        ),
    }
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

/// The report of one evolution as JSON. Labels and statuses are read from
/// `graph`, so call it while the evolution is still applied.
fn fixed_point_json(graph: &LodGraph, params: EvolveParams, report: &FixedPointReport) -> Value {
    let transitions: Vec<Value> = report
        .transitions
        .iter()
        .take(MAX_LISTED_TRANSITIONS)
        .map(|t| {
            let mut v = node_json(graph, t.node);
            v["from"] = json!(status_name(t.from));
            v["to"] = json!(status_name(t.to));
            v
        })
        .collect();
    json!({
        "beta": report.beta,
        "gamma": report.gamma,
        "tolerance": report.tolerance,
        "theta_lo": report.theta_lo,
        "theta_hi": report.theta_hi,
        "max_steps": params.max_steps,
        "nodes": report.nodes,
        "pinned": report.pinned,
        "dependency_edges": report.dependency_edges,
        "falsification_edges": report.falsification_edges,
        "scc_count": report.scc_count,
        "trivial_scc_count": report.trivial_scc_count,
        "cyclic_scc_count": report.cyclic_scc_count,
        "max_scc_size": report.max_scc_size,
        "contraction": report.contraction,
        "node_updates": report.node_updates,
        "iterations": report.iterations,
        "k_max": report.k_max,
        "initial_delta": report.initial_delta,
        "residual": report.residual,
        "error_bound": report.error_bound,
        "converged": true,
        "transitions": transitions,
        "transitions_total": report.transitions.len(),
        "transitions_truncated": report.transitions.len() > MAX_LISTED_TRANSITIONS,
        "revoked_entities": report.revoked_entities,
        "reinstated_entities": report.reinstated_entities,
        "retracted_dependencies": report.retracted_dependencies,
        "added_dependencies": report.added_dependencies,
    })
}

fn prune(graph: &LodGraph, spec: PruneSpec) -> Result<(String, Value), Rejection> {
    let e = entity(spec.entity_id, spec.action.as_deref(), "graph_prune")?;
    let dry_run = spec.dry_run.unwrap_or(false);
    let params = EvolveParams::resolve(
        spec.beta,
        spec.gamma,
        spec.tolerance,
        spec.theta_lo,
        spec.theta_hi,
        spec.max_steps,
    )?;
    let node = graph
        .node_for_entity(e)
        .ok_or_else(|| graph_rejection(LodError::EntityNotFound(e)))?;
    let (pruned, revoked, retracted, fixed_point) = graph
        .transact(|g| {
            let before = g.create_checkpoint();
            let retracted_by_evidence = g.falsify_node(node)?;
            let report = params.run(g)?;
            let retracted = retracted_by_evidence + report.retracted_dependencies;
            // The root, then every node the evolution falsified. Read the labels
            // while the prune is still applied.
            let falsified = report
                .transitions
                .iter()
                .filter(|t| t.to == EpistemicStatus::Falsified);
            let pruned: Vec<Value> = std::iter::once(node)
                .chain(falsified.map(|t| t.node))
                .take(MAX_LISTED_TRANSITIONS + 1)
                .map(|id| node_json(g, id))
                .collect();
            let revoked: Vec<u64> = std::iter::once(e)
                .chain(report.revoked_entities.iter().copied())
                .collect();
            let fixed_point = fixed_point_json(g, params, &report);
            if dry_run {
                g.rollback_checkpoint(&before)?;
            }
            Ok((pruned, revoked, retracted, fixed_point))
        })
        .map_err(graph_rejection)?;
    let summary = format!(
        "graph_prune: {} node(s) {} from entity {e}",
        revoked.len(),
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
            "revoked_entities": revoked,
            "retracted_dependencies": retracted,
            "fixed_point": fixed_point,
            "dry_run": dry_run,
            "applied": !dry_run,
        }),
    ))
}

fn evolve(graph: &LodGraph, spec: EvolveSpec) -> Result<(String, Value), Rejection> {
    let dry_run = spec.dry_run.unwrap_or(false);
    let params = EvolveParams::resolve(
        spec.beta,
        spec.gamma,
        spec.tolerance,
        spec.theta_lo,
        spec.theta_hi,
        spec.max_steps,
    )?;
    if spec.retract.len() > MAX_EVOLVE_RETRACTIONS {
        return Err(invalid(format!(
            "retract takes at most {MAX_EVOLVE_RETRACTIONS} entities"
        )));
    }
    let mut retract = Vec::with_capacity(spec.retract.len());
    for (i, r) in spec.retract.iter().enumerate() {
        let e = entity(r.entity_id, r.action.as_deref(), &format!("retract[{i}]"))?;
        let node = graph
            .node_for_entity(e)
            .ok_or_else(|| graph_rejection(LodError::EntityNotFound(e)))?;
        retract.push((e, node));
    }
    let fixed_point = graph
        .transact(|g| {
            let before = g.create_checkpoint();
            for &(_, node) in &retract {
                g.retract_falsification(node)?;
            }
            let report = params.run(g)?;
            let fixed_point = fixed_point_json(g, params, &report);
            if dry_run {
                g.rollback_checkpoint(&before)?;
            }
            Ok(fixed_point)
        })
        .map_err(graph_rejection)?;
    let summary = format!(
        "graph_evolve: fixed point in {} step(s) (bound {}), {} status change(s){}",
        fixed_point["iterations"],
        fixed_point["k_max"],
        fixed_point["transitions_total"],
        if dry_run {
            " (dry run, rolled back)"
        } else {
            ""
        }
    );
    let retracted: Vec<u64> = retract.iter().map(|&(e, _)| e).collect();
    Ok((
        summary,
        json!({
            "retracted_evidence": retracted,
            "fixed_point": fixed_point,
            "dry_run": dry_run,
            "applied": !dry_run,
        }),
    ))
}

fn coarse_grain(graph: &LodGraph, spec: CoarseGrainSpec) -> Result<(String, Value), Rejection> {
    if spec.members.is_empty() || spec.members.len() > MAX_DEPOSIT_NODES {
        return Err(invalid(format!(
            "members must hold 1..={MAX_DEPOSIT_NODES} entities"
        )));
    }
    let summary_entity = entity(spec.entity_id, spec.action.as_deref(), "graph_coarse_grain")?;
    let summary_coord = coord(graph, &spec.coord)?;
    let dry_run = spec.dry_run.unwrap_or(false);
    let mut members = Vec::with_capacity(spec.members.len());
    for (i, m) in spec.members.iter().enumerate() {
        let e = entity(m.entity_id, m.action.as_deref(), &format!("members[{i}]"))?;
        let node = graph
            .node_for_entity(e)
            .ok_or_else(|| graph_rejection(LodError::EntityNotFound(e)))?;
        members.push(node);
    }
    let (summary, member_nodes, csr_edges) = graph
        .transact(|g| {
            if g.node_count() + 1 > MAX_GRAPH_NODES {
                return Err(LodError::InvalidNode(format!(
                    "the live graph is capped at {MAX_GRAPH_NODES} nodes"
                )));
            }
            let before = g.create_checkpoint();
            let id = g.coarse_grain_cluster(&members, summary_entity, summary_coord, spec.hdc)?;
            let summary = node_json(g, id);
            let member_nodes: Vec<Value> = members.iter().map(|&m| node_json(g, m)).collect();
            let csr_edges = g.csr_snapshot().num_edges();
            if dry_run {
                g.rollback_checkpoint(&before)?;
            }
            Ok((summary, member_nodes, csr_edges))
        })
        .map_err(graph_rejection)?;
    let summary_line = format!(
        "graph_coarse_grain: {} member(s) {} under entity {summary_entity} at band {}",
        members.len(),
        if dry_run {
            "would be coarse-grained (dry run, rolled back)"
        } else {
            "coarse-grained"
        },
        summary["band"]
    );
    Ok((
        summary_line,
        json!({
            "summary": summary,
            "members": member_nodes,
            "edge_type": "coarse_grain",
            "csr_edges_after": csr_edges,
            "dry_run": dry_run,
            "applied": !dry_run,
        }),
    ))
}

fn zoom(graph: &LodGraph, spec: ZoomSpec) -> Result<(String, Value), Rejection> {
    let e = entity(spec.entity_id, spec.action.as_deref(), "graph_zoom")?;
    let direction = match spec.direction.as_str() {
        "in" => Some(ZoomDirection::In),
        "out" => Some(ZoomDirection::Out),
        "to_coord" => None,
        other => {
            return Err(invalid(format!(
                "unknown direction `{other}`; expected in | out | to_coord"
            )))
        }
    };
    let dry_run = spec.dry_run.unwrap_or(false);
    let node = graph
        .node_for_entity(e)
        .ok_or_else(|| graph_rejection(LodError::EntityNotFound(e)))?;
    let (from, to, node_after) = graph
        .transact(|g| {
            let before = g.create_checkpoint();
            let (from, to) = match direction {
                Some(d) => {
                    let from = g.get_node(node).ok_or(LodError::NodeNotFound(node))?.band;
                    (from, g.zoom_node(node, d)?)
                }
                None => g.migrate_band_to_coord(node)?,
            };
            let node_after = node_json(g, node);
            if dry_run {
                g.rollback_checkpoint(&before)?;
            }
            Ok((from, to, node_after))
        })
        .map_err(graph_rejection)?;
    let summary = format!(
        "graph_zoom: entity {e} band {} -> {}{}",
        band_level(from),
        band_level(to),
        if dry_run {
            " (dry run, rolled back)"
        } else {
            ""
        }
    );
    Ok((
        summary,
        json!({
            "entity_id": e,
            "direction": spec.direction,
            "from_band": band_level(from),
            "to_band": band_level(to),
            "node": node_after,
            "dry_run": dry_run,
            "applied": !dry_run,
        }),
    ))
}
