//! Graph fact verbs over the engine's live [`LodGraph`], the same graph the
//! PolicyGate reads for revocations in `ask`, `pipeline` and the world-model verbs:
//!
//! - `graph_deposit`: append nodes and edges in one transaction, then flush them
//!   into the CSR snapshot. Any failure rolls the whole deposit back.
//! - `graph_recall`: two-stage HDC + manifold recall under the graph's geometry.
//! - `graph_rag`: three-stage retrieval (`LodGraph::hybrid_rag_search`): HDC
//!   prefilter, geodesic rerank to anchors (exact vector angle on the vector
//!   track), PPR diffusion from the anchors.
//!   The query is text (`query_text`, lexical projection), a dense vector
//!   from an external embedding model (`query_vector`, dense projection), both
//!   at once, or a `coord` + `hdc` pair. Text and coordinates are compared
//!   with each node's own coordinate and its aliases; a vector only with node
//!   embeddings. Every hit carries its payload, source, digest and aliases,
//!   and names the anchor that matched.
//! - `graph_ppr`: Personalized PageRank diffusion from seed entities.
//! - `graph_prune`: record evidence against one entity, then evolve every
//!   confidence to the fixed point; dependents that fall below `theta_lo` are
//!   falsified and revoked. `dry_run` reports and changes nothing.
//! - `graph_evolve`: optionally retract such evidence, then evolve every
//!   confidence to the fixed point. `dry_run` reports and changes nothing.
//!
//!   Both evolutions take `gamma` (default 1), the gain of `falsifies` edges in
//!   `c = (1 - beta) pi + beta max(0, P+ c - gamma P- c)`, and solve it by
//!   strongly connected components in topological order. The response echoes
//!   `gamma` and the block counts (`scc_count`, `trivial_scc_count`,
//!   `cyclic_scc_count`, `max_scc_size`).
//! - `graph_coarse_grain`: insert a summary node for a cluster of member nodes
//!   on the band its coordinate implies (strictly coarser than every member),
//!   link each member to it with a `CoarseGrain` edge and flush. `dry_run`
//!   reports and changes nothing.
//! - `graph_zoom`: move one node one band `in` or `out`, or `to_coord`: to the
//!   band its coordinate implies. The coarse-grain order is kept. `dry_run`
//!   reports and changes nothing.
//! - `graph_induce`: Text-to-Graph ([`crate::text_to_graph`]). A task text
//!   passes the answerability gate and becomes a causal action DAG (the
//!   planner's `CausalDagSpec`). With `auto_deposit: true` an answerable
//!   result is deposited exactly like a `graph_deposit` (same transaction,
//!   same Qwen embedding of payloads): one band-1 `hypothesized` node per
//!   action, named by `action`, and one `depends_on` edge (weight 1) from each
//!   parent to its child. Only `depends_on`: a parallel `causal_transition`
//!   edge would count the same prerequisite twice in the row-normalized
//!   support of `graph_prune` and `graph_evolve`. A refused text deposits
//!   nothing; an action already in the graph fails the whole deposit (409).
//!
//! A deposited node may carry an `operator` signature (`name`,
//! `operator_kind`: `hard_dcm` or `soft_pcm`, `embedder_space`, `version`,
//! `pure`), the causal operator it stands for. The graph refuses an invalid
//! one at insert; it is persisted and echoed in every node report
//! (`node_json`), but not in `graph_rag` hits. `graph_execute_operator` runs
//! registered operators; graph construction registers the audit DCM and state-summary PCM.
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
//! (`alias_link_edges` in the response). Vectors come from the caller or from
//! the native Qwen backend (below), all of one dimension per graph; a Qwen
//! vector also carries the native backend's identity
//! ([`gen_zero_lod::LodNode::embedder_space`]), which the graph locks the same
//! way once the first one arrives.
//!
//! The graph's geometry (curvature, sphere radius, metric weights) is fixed when
//! the engine starts (`GENZERO_GRAPH_GEOMETRY`) and echoed in every response.
//!
//! Request block: `{"graph": {...}}`. Unknown keys are refused. Nodes are named by
//! `entity_id`, or by `action`, whose entity id is `action_id(action)`: the key
//! the gate checks, so a pruned action is hard-stopped everywhere.
//!
//! Qwen dense track: when the native Qwen backend is loaded
//! (`GENZERO_QWEN_MODEL_PATH`), [`embed_graph_request`] runs before the verb
//! and the graph stays model-free; it only receives vectors and, when it
//! declares one, [`gen_zero_lod::LodNode::embedder_space`], a mean-pooled
//! vector's identity string ([`crate::semantic::NativeQwen::embedder_id`]).
//! - `graph_deposit`: each node with a `payload` and no caller `embedding` gets
//!   the Qwen embedding of its payload (896 wide for Qwen2.5-0.5B, mean-pooled:
//!   `qwen_pooling: "mean"`), so text queries have a dense track to meet.
//! - `graph_rag`: a `query_text` without `query_vector` is also embedded and
//!   searched as `text+vector`; the response says `qwen_embedded: true` and
//!   `qwen_pooling: "mean"`.
//!
//! When no vector is made although text was there because no backend is
//! configured, or the configured backend is the Python bridge (no in-process
//! embedder), the verb runs lexical-only, says `qwen_embedded: false` and
//! names `qwen_skip_reason`, and a warning is logged: there is no Qwen claim
//! to protect, only an absent one. But once a native Qwen backend is loaded,
//! a graph whose embeddings have another width, or are locked to a different
//! embedder identity ([`gen_zero_lod::LodGraph::embedder_space`]), is a
//! caller or graph-selection mistake: the request is refused outright
//! (fail-closed), never silently downgraded to lexical-only. A configured
//! backend that fails to embed also fails the request, as does text over the
//! per-request token budget ([`crate::semantic::MAX_EMBED_TOKENS_PER_CALL`],
//! 413); none of these ever fall back to lexical.
//!
//! GENZERO_GRAPH_PERSIST_DIR mounts durable transaction and reflection commits.
//! Without it, a restart keeps only the startup seed (GENZERO_GRAPH_SEED).

use crate::bridge::BridgeError;
use crate::cognitive::Rejection;
use crate::semantic::{NativeQwen, SemanticBackend};
use crate::text_to_graph::{InduceOutcome, InduceRequest, TextToGraphInducer};
use crate::zero::action_id;
use gen_zero_lod::{
    AnchorMatch, DiffusionQuality, EdgeType, EpistemicStatus, FixedPointReport, HybridRagResult,
    LodBand, LodError, LodGraph, LodNode, MixedCurvatureCoord, OperatorSignature, Placement,
    RagHit, ZoomDirection, ADMISSION_BETA, ADMISSION_GAMMA, DENSE_PROJECTOR_VERSION,
    PROJECTOR_VERSION,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;

const STAGE: &str = "graph";

/// Per-request and total caps. The verbs mutate gate-relevant state, so one
/// caller must not be able to exhaust memory.
pub const MAX_DEPOSIT_NODES: usize = 1024;
pub const MAX_DEPOSIT_EDGES: usize = 4096;
pub use gen_zero_lod::MAX_GRAPH_NODES;
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
/// every response. Uncalibrated presets. `beta` and `gamma` are the ones every
/// edge is admitted against, so a default evolution never meets a cycle it
/// must refuse.
pub const DEFAULT_EVOLVE_BETA: f32 = ADMISSION_BETA;
/// Falsification gain: how strongly a `falsifies` edge presses its target.
pub const DEFAULT_EVOLVE_GAMMA: f32 = ADMISSION_GAMMA;
pub const DEFAULT_EVOLVE_TOLERANCE: f32 = 1e-6;
pub const DEFAULT_EVOLVE_THETA_LO: f32 = 0.2;
pub const DEFAULT_EVOLVE_THETA_HI: f32 = 0.8;
/// Largest step budget one request may ask for, and the default.
pub const MAX_EVOLVE_STEPS: usize = 10_000;
/// Most evidence retractions in one `graph_evolve`.
pub const MAX_EVOLVE_RETRACTIONS: usize = 256;
/// Most status transitions listed in one response; the total is always reported.
pub const MAX_LISTED_TRANSITIONS: usize = 256;

/// The ten graph operations.
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
    Induce,
    ExecuteOperator,
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
            Self::Induce => "graph_induce",
            Self::ExecuteOperator => "graph_execute_operator",
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
    /// The causal operator the node stands for; validated by the graph at insert.
    operator: Option<OperatorSignature>,
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
    /// Caps the LOD payload variant chosen per hit: detail when it fits,
    /// else a `CoarseGrain` summary ancestor's payload, else the hit is
    /// dropped. `None`: every hit keeps its own detailed payload, as before.
    token_budget: Option<usize>,
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
/// map is not a contraction on some cycle, 422; so is a deposited edge that
/// admission refuses because it closes such a cycle, or one too slow to
/// converge inside the step budget. A CSR
/// or checkpoint failure is an engine fault, 500.
pub(crate) fn graph_rejection(e: LodError) -> Rejection {
    let (code, status) = match &e {
        LodError::OperatorNodeRevoked(_) => ("OperatorNodeRevoked", 409),
        LodError::OperatorNodeFalsified(_) => ("OperatorNodeFalsified", 409),
        LodError::OperatorNonceRejected { .. } => ("OperatorNonceRejected", 409),
        LodError::OperatorPreconditionFailed { .. } => ("OperatorPreconditionFailed", 409),
        LodError::OperatorTransitFailed { .. } => ("OperatorTransitFailed", 422),
        LodError::OperatorPostconditionFailed { .. } => ("OperatorPostconditionFailed", 422),
        LodError::OperatorNotFound(_) => ("OperatorNotFound", 404),
        LodError::FixedPointDiverged { .. } => ("FixedPointDiverged", 422),
        LodError::FixedPointNotContractive { .. } => ("FixedPointNotContractive", 422),
        LodError::FixedPointTooSlow { .. } => ("FixedPointTooSlow", 422),
        LodError::SpineBreatheOutOfBounds { .. } => ("BandOutOfRange", 409),
        LodError::GraphCapacityExceeded { .. } => ("GraphCapacityExceeded", 409),
        LodError::DuplicateEntity(_) => ("DuplicateEntity", 409),
        LodError::EntityNotFound(_) | LodError::NodeNotFound(_) => ("EntityNotFound", 404),
        LodError::EmptyInput(_) => ("EmptyInput", 400),
        LodError::PayloadTooLarge { .. } => ("PayloadTooLarge", 413),
        LodError::CsrInvariant(_) | LodError::FlushConflict | LodError::CheckpointRejected(_) => {
            ("GraphError", 500)
        }
        // The disk refused a commit; the request itself was valid.
        LodError::Persistence(_) => ("GraphPersistence", 503),
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
        "persisted": graph.is_persistent(),
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
            "operator": n.operator,
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

/// Qwen vectors made for one graph request before it runs
/// ([`embed_graph_request`]), or why none were made for text that was there.
#[derive(Debug, Default)]
pub struct QwenVectors {
    /// Set when at least one vector was made.
    embedder: Option<String>,
    /// The pooling [`Self::embedder`] used, e.g. `"mean"`. Set together with
    /// `embedder`.
    pooling: Option<String>,
    /// `graph_rag`: the embedding of `query_text`.
    query: Option<Vec<f32>>,
    /// `graph_deposit`: the payload embedding of `nodes[i]`, by `i`.
    nodes: HashMap<usize, Vec<f32>>,
    skip_reason: Option<String>,
}

impl QwenVectors {
    fn skipped(op: GraphOp, reason: String, configured: bool) -> Self {
        if configured {
            tracing::warn!(op = op.name(), reason = %reason, "graph: Qwen dense track skipped; lexical only");
        } else {
            tracing::info!(op = op.name(), reason = %reason, "graph: Qwen dense track off; lexical only");
        }
        Self {
            skip_reason: Some(reason),
            ..Self::default()
        }
    }

    fn report(&self, out: &mut Value, embedded_key: &str, embedded: Value) {
        out[embedded_key] = embedded;
        out["qwen_embedder"] = json!(self.embedder);
        out["qwen_pooling"] = json!(self.pooling);
        out["qwen_skip_reason"] = json!(self.skip_reason);
    }
}

/// A configured backend that fails to embed fails the request: text over the
/// token budget is 413, a full queue 503, every other failure 500.
fn embed_rejection(e: BridgeError) -> Rejection {
    let (code, status) = match &e {
        BridgeError::EmbedBudget { .. } => ("EmbedBudgetExceeded", 413),
        BridgeError::Overloaded { .. } => ("EmbedderOverloaded", 503),
        _ => ("EmbedderError", 500),
    };
    Rejection {
        code: code.to_string(),
        stage: STAGE.to_string(),
        detail: e.to_string(),
        http_status: status,
    }
}

/// Texts of a deposit that get a Qwen embedding: the non-blank payload of
/// every node that carries no caller `embedding`, by node index.
fn deposit_texts(spec: &DepositSpec) -> Vec<(usize, String)> {
    spec.nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.embedding.is_none())
        .filter_map(|(i, n)| n.payload.as_ref().map(|p| (i, p)))
        .filter(|(_, p)| !p.trim().is_empty())
        .map(|(i, p)| (i, p.clone()))
        .collect()
}

/// Why this deposit's Qwen vectors (width `dim`, identity `embedder`) cannot
/// join `graph`, fail-closed: the graph or a caller embedding in the same
/// deposit has another width, or the graph is locked to a different embedder
/// identity ([`LodGraph::embedder_space`]; two models can share a width while
/// embedding different semantic spaces). Never silently skipped to
/// lexical-only: a configured, dimension-matched Qwen backend that disagrees
/// with an already-populated graph is a caller or graph-selection mistake,
/// not a normal degraded mode.
fn deposit_qwen_conflict(
    graph: &LodGraph,
    spec: &DepositSpec,
    dim: usize,
    embedder: &str,
) -> Option<String> {
    if let Some(have) = graph.embedding_dim().filter(|&have| have != dim) {
        return Some(format!(
            "graph embedding dimension {have} does not match configured native Qwen dimension {dim}"
        ));
    }
    if let Some(locked) = graph.embedder_space().filter(|locked| locked != embedder) {
        return Some(format!(
            "graph embedder `{locked}` does not match configured native Qwen embedder `{embedder}`"
        ));
    }
    spec.nodes
        .iter()
        .enumerate()
        .find_map(|(i, n)| n.embedding.as_ref().map(|e| (i, e.len())))
        .filter(|&(_, len)| len != dim)
        .map(|(i, len)| {
            format!("nodes[{i}] carries a {len}-dimensional caller embedding; the Qwen embedder gives {dim}")
        })
}

/// The backend and its in-process embedder, or the skip report when there is
/// none: no backend at all, or the Python bridge. Unlike
/// [`deposit_qwen_conflict`], these are not fail-closed: no Qwen backend is
/// configured at all, so there is no conflicting claim to protect, only an
/// absent one.
fn qwen_embedder(
    op: GraphOp,
    semantic: Option<&SemanticBackend>,
) -> Result<(&SemanticBackend, &NativeQwen), QwenVectors> {
    let Some(backend) = semantic else {
        let reason = "no semantic backend is configured (GENZERO_QWEN_MODEL_PATH unset)";
        return Err(QwenVectors::skipped(op, reason.into(), false));
    };
    match backend.embedder() {
        Some(native) => Ok((backend, native)),
        None => Err(QwenVectors::skipped(
            op,
            format!(
                "the {} semantic backend has no text embedder",
                backend.engine_name()
            ),
            true,
        )),
    }
}

/// Why [`deposit_embedder`] made no Qwen vectors: a lexical-only skip (no
/// backend, or the Python bridge), or a width/identity conflict the caller
/// must fix, which [`embed_graph_request`] and [`load_seed`] surface as a
/// hard failure rather than papering over it as [`QwenVectors::skipped`].
enum QwenEmbedFault {
    Skip(QwenVectors),
    Reject(Rejection),
}

/// [`qwen_embedder`] for a deposit, also refused (fail-closed, never skipped)
/// when [`deposit_qwen_conflict`] finds one.
fn deposit_embedder<'a>(
    graph: &LodGraph,
    op: GraphOp,
    spec: &DepositSpec,
    semantic: Option<&'a SemanticBackend>,
) -> Result<&'a NativeQwen, QwenEmbedFault> {
    let (_, native) = qwen_embedder(op, semantic).map_err(QwenEmbedFault::Skip)?;
    let embedder_id = native.embedder_id();
    match deposit_qwen_conflict(graph, spec, native.embedding_dim(), &embedder_id) {
        Some(reason) => Err(QwenEmbedFault::Reject(invalid(reason))),
        None => Ok(native),
    }
}

/// Make the Qwen vectors `op` uses, before the verb runs. Only `graph_rag`
/// with `query_text` and no `query_vector`, and `graph_deposit` nodes with
/// a payload and no `embedding`, get any. A malformed block is refused here
/// exactly as [`execute_graph`] would refuse it.
pub async fn embed_graph_request(
    graph: &LodGraph,
    op: GraphOp,
    block: &Value,
    semantic: Option<&SemanticBackend>,
) -> Result<QwenVectors, Rejection> {
    match op {
        GraphOp::Rag => {
            let spec: RagSpec = parse(block, op)?;
            let text = match (&spec.query_text, &spec.query_vector, &spec.coord, spec.hdc) {
                (Some(t), None, None, None) if !t.trim().is_empty() => t.clone(),
                _ => return Ok(QwenVectors::default()),
            };
            let (backend, native) = match qwen_embedder(op, semantic) {
                Ok(found) => found,
                Err(skipped) => return Ok(skipped),
            };
            let dim = native.embedding_dim();
            let embedder_id = native.embedder_id();
            match graph.embedding_dim() {
                Some(have) if have == dim => {}
                Some(have) => {
                    // Fail-closed: a configured Qwen backend whose dimension
                    // disagrees with an already-populated graph is refused,
                    // never silently downgraded to a lexical-only success.
                    return Err(invalid(format!(
                        "graph embedding dimension {have} does not match configured native \
                         Qwen dimension {dim}"
                    )));
                }
                None => {
                    let reason = "no node of this graph carries an embedding to compare a Qwen \
                                  query vector with"
                        .to_string();
                    return Ok(QwenVectors::skipped(op, reason, true));
                }
            }
            if let Some(locked) = graph
                .embedder_space()
                .filter(|locked| locked != &embedder_id)
            {
                // Fail-closed: same width, different model. A bare dimension
                // match cannot tell the semantic spaces apart.
                return Err(invalid(format!(
                    "graph embedder `{locked}` does not match configured native Qwen embedder \
                     `{embedder_id}`"
                )));
            }
            let vector = backend.embed(&text).await.map_err(embed_rejection)?;
            Ok(QwenVectors {
                embedder: Some(embedder_id),
                pooling: Some(native.pooling().as_str().to_string()),
                query: Some(vector),
                ..QwenVectors::default()
            })
        }
        GraphOp::Deposit => embed_deposit(graph, op, &parse(block, op)?, semantic).await,
        GraphOp::Induce => {
            // The induction is deterministic, so the deposit `execute_graph`
            // builds from the same block has these node indices.
            let req: InduceRequest = parse(block, op)?;
            if req.auto_deposit != Some(true) {
                return Ok(QwenVectors::default());
            }
            let outcome = TextToGraphInducer::new().induce(&req)?;
            match induced_deposit_spec(&req, &outcome) {
                Some(spec) => embed_deposit(graph, op, &spec, semantic).await,
                None => Ok(QwenVectors::default()),
            }
        }
        _ => Ok(QwenVectors::default()),
    }
}

/// The Qwen payload vectors of one deposit (`graph_deposit`, or the deposit
/// of a `graph_induce`).
async fn embed_deposit(
    graph: &LodGraph,
    op: GraphOp,
    spec: &DepositSpec,
    semantic: Option<&SemanticBackend>,
) -> Result<QwenVectors, Rejection> {
    let texts = deposit_texts(spec);
    // Over the cap the verb refuses the deposit; embed nothing first.
    if texts.is_empty() || spec.nodes.len() > MAX_DEPOSIT_NODES {
        return Ok(QwenVectors::default());
    }
    let native = match deposit_embedder(graph, op, spec, semantic) {
        Ok(native) => native,
        Err(QwenEmbedFault::Skip(skipped)) => return Ok(skipped),
        Err(QwenEmbedFault::Reject(rejection)) => return Err(rejection),
    };
    let (index, texts): (Vec<usize>, Vec<String>) = texts.into_iter().unzip();
    let vectors = native.embed_texts(texts).await.map_err(embed_rejection)?;
    Ok(QwenVectors {
        embedder: Some(native.embedder_id()),
        pooling: Some(native.pooling().as_str().to_string()),
        nodes: index.into_iter().zip(vectors).collect(),
        ..QwenVectors::default()
    })
}

/// Band of every induced action node: Lod1, the operator-binding layer of
/// RFC-20261002 section 4.4.
const INDUCED_ACTION_BAND: u8 = 1;

/// The `graph_deposit` of an answerable induction: one node per action and
/// one `depends_on` edge per parent. `None` when the text was refused.
fn induced_deposit_spec(req: &InduceRequest, outcome: &InduceOutcome) -> Option<DepositSpec> {
    if !outcome.answerable || outcome.actions.is_empty() {
        return None;
    }
    let source_uri = format!(
        "graph_induce:{}:{}",
        outcome.engine,
        blake3::hash(req.text.as_bytes()).to_hex()
    );
    let names: HashMap<u32, &str> = outcome
        .actions
        .iter()
        .map(|a| (a.action_id, a.name.as_str()))
        .collect();
    let nodes = outcome
        .actions
        .iter()
        .map(|a| NodeSpec {
            entity_id: None,
            action: Some(a.name.clone()),
            label: a.name.clone(),
            band: Some(INDUCED_ACTION_BAND),
            status: "hypothesized".into(),
            coord: None,
            hdc: None,
            confidence: outcome.confidence,
            payload: Some(match &req.context {
                Some(context) => format!("{}\ncontext: {context}", a.name),
                None => a.name.clone(),
            }),
            source_uri: Some(source_uri.clone()),
            timestamp_ns: None,
            aliases: Vec::new(),
            embedding: None,
            operator: Some(a.operator.clone()),
        })
        .collect();
    let edges = outcome
        .actions
        .iter()
        .flat_map(|child| {
            let names = &names;
            child.parents.iter().map(move |p| EdgeSpec {
                source: EntityRef {
                    entity_id: None,
                    action: Some(names[p].to_string()),
                },
                target: EntityRef {
                    entity_id: None,
                    action: Some(child.name.clone()),
                },
                edge_type: "depends_on".into(),
                weight: 1.0,
            })
        })
        .collect();
    Some(DepositSpec { nodes, edges })
}

fn induce(
    graph: &LodGraph,
    req: InduceRequest,
    qwen: &QwenVectors,
) -> Result<(String, Value), Rejection> {
    let mut outcome = TextToGraphInducer::new().induce(&req)?;
    let deposit_report = match (
        req.auto_deposit == Some(true),
        induced_deposit_spec(&req, &outcome),
    ) {
        (true, Some(spec)) => {
            let (_, report) = deposit(graph, spec, false, qwen)?;
            let ids = report["nodes"]
                .as_array()
                .map(|nodes| nodes.iter().filter_map(|n| n["node"].as_u64()).collect())
                .unwrap_or_default();
            outcome.deposited_node_ids = Some(ids);
            Some(report)
        }
        _ => None,
    };
    let summary = match &outcome.dag_spec {
        Some(spec) => format!(
            "graph_induce: {} action(s), target {}, confidence {:.3}{}",
            outcome.actions.len(),
            spec.target,
            outcome.confidence,
            if deposit_report.is_some() {
                ", deposited"
            } else {
                ""
            }
        ),
        None => format!(
            "graph_induce: not answerable ({})",
            outcome.refusal_reason.as_deref().unwrap_or("refused")
        ),
    };
    let mut out = serde_json::to_value(&outcome)
        .map_err(|e| invalid(format!("internal: graph_induce outcome: {e}")))?;
    out["auto_deposit"] = json!(req.auto_deposit == Some(true));
    out["deposit"] = match deposit_report {
        Some(report) => report,
        None if req.auto_deposit == Some(true) => {
            json!({"skipped": "the text is not answerable; nothing was deposited"})
        }
        None => Value::Null,
    };
    Ok((summary, out))
}

/// Run one graph verb with the vectors [`embed_graph_request`] made for it.
/// `Ok` holds a one-line summary and the result object.
pub fn execute_graph(
    graph: &LodGraph,
    op: GraphOp,
    block: &Value,
    qwen: &QwenVectors,
) -> Result<(String, Value), Rejection> {
    let (summary, mut result) = match op {
        GraphOp::Deposit => deposit(graph, parse(block, op)?, false, qwen)?,
        GraphOp::Recall => recall(graph, parse(block, op)?)?,
        GraphOp::Rag => rag(graph, parse(block, op)?, qwen)?,
        GraphOp::Ppr => ppr(graph, parse(block, op)?)?,
        GraphOp::Prune => prune(graph, parse(block, op)?)?,
        GraphOp::Evolve => evolve(graph, parse(block, op)?)?,
        GraphOp::CoarseGrain => coarse_grain(graph, parse(block, op)?)?,
        GraphOp::Zoom => zoom(graph, parse(block, op)?)?,
        GraphOp::Induce => induce(graph, parse(block, op)?, qwen)?,
        GraphOp::ExecuteOperator => execute_operator(graph, parse(block, op)?)?,
    };
    result["op"] = json!(op.name());
    result["graph"] = graph_meta(graph);
    Ok((summary, result))
}

/// Load the operator seed file into `graph` as one transaction. The file has the
/// `graph_deposit` shape and may also carry `axiomatic` nodes. Any bad node or
/// edge fails the whole load and leaves the graph unchanged. The report carries
/// the file's blake3 `digest`. With the native Qwen backend, seed payloads
/// get Qwen embeddings like a `graph_deposit`; an embedding failure fails
/// the load.
pub fn load_seed(
    graph: &LodGraph,
    path: &Path,
    semantic: Option<&SemanticBackend>,
) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read graph seed {}: {e}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| format!("graph seed {} is not JSON: {e}", path.display()))?;
    let spec: DepositSpec = DepositSpec::deserialize(&value)
        .map_err(|e| format!("graph seed {}: {e}", path.display()))?;
    let texts = deposit_texts(&spec);
    let qwen = if texts.is_empty() {
        QwenVectors::default()
    } else {
        match deposit_embedder(graph, GraphOp::Deposit, &spec, semantic) {
            Err(QwenEmbedFault::Skip(skipped)) => skipped,
            Err(QwenEmbedFault::Reject(rejection)) => {
                return Err(format!(
                    "graph seed {}: {}",
                    path.display(),
                    rejection.detail
                ))
            }
            Ok(native) => {
                let (index, texts): (Vec<usize>, Vec<String>) = texts.into_iter().unzip();
                let vectors = native.embed_texts_blocking(&texts).map_err(|e| {
                    format!("graph seed {}: Qwen embedding failed: {e}", path.display())
                })?;
                QwenVectors {
                    embedder: Some(native.embedder_id()),
                    pooling: Some(native.pooling().as_str().to_string()),
                    nodes: index.into_iter().zip(vectors).collect(),
                    ..QwenVectors::default()
                }
            }
        }
    };
    let (_, mut report) = deposit(graph, spec, true, &qwen)
        .map_err(|r| format!("graph seed {}: {}", path.display(), r.detail))?;
    report["digest"] = json!(blake3::hash(text.as_bytes()).to_hex().to_string());
    Ok(report)
}

fn deposit(
    graph: &LodGraph,
    spec: DepositSpec,
    allow_axiomatic: bool,
    qwen: &QwenVectors,
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
        match (n.embedding.as_ref(), qwen.nodes.get(&i)) {
            (Some(embedding), _) => {
                // A caller-supplied embedding declares no model identity: its
                // provenance is the caller's own business.
                node = node.with_embedding(embedding.clone());
            }
            (None, Some(embedding)) => {
                node = node.with_embedding(embedding.clone());
                if let Some(embedder) = &qwen.embedder {
                    node = node.with_embedder_space(embedder.clone());
                }
            }
            (None, None) => {}
        }
        if placement == Placement::Embedding {
            node = node.placed_by_embedding();
        }
        if let Some(signature) = &n.operator {
            node = node.with_operator(signature.clone());
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
            let mut resolved = Vec::with_capacity(edges.len());
            for &(source, target, edge_type, weight) in &edges {
                let s = g
                    .node_for_entity(source)
                    .ok_or(LodError::EntityNotFound(source))?;
                let t = g
                    .node_for_entity(target)
                    .ok_or(LodError::EntityNotFound(target))?;
                resolved.push((s, t, edge_type, weight));
            }
            // One admission pass for the whole deposit.
            let tickets = g.add_edges(&resolved)?;
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
    let mut out = json!({
        "nodes": deposited,
        "edge_tickets": tickets,
        "alias_link_edges": alias_link_edges,
        "flush": {
            "merged_edges": flush.merged_edges,
            "csr_nodes": flush.csr_nodes,
            "csr_edges": flush.csr_edges,
            "pending_edges": flush.pending_edges,
        },
    });
    qwen.report(&mut out, "qwen_embedded_nodes", json!(qwen.nodes.len()));
    if let Some(dim) = qwen.nodes.values().next().map(Vec::len) {
        out["qwen_vector_dim"] = json!(dim);
    }
    Ok((summary, out))
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

fn rag(graph: &LodGraph, spec: RagSpec, qwen: &QwenVectors) -> Result<(String, Value), Rejection> {
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
    let text = spec.query_text.as_deref();
    let (vector, vector_source) = match (spec.query_vector.as_deref(), qwen.query.as_deref()) {
        (Some(v), _) => (Some(v), Some("caller")),
        (None, Some(v)) => (Some(v), Some("qwen")),
        (None, None) => (None, None),
    };
    // A caller-supplied `query_vector` declares no model identity; only the
    // Qwen-embedded query does (`qwen.embedder`).
    let query_embedder = match vector_source {
        Some("qwen") => qwen.embedder.as_deref(),
        _ => None,
    };
    let (result, query) = match (text.is_some() || vector.is_some(), &spec.coord, spec.hdc) {
        (true, None, None) => (
            graph
                .hybrid_rag_search_query(
                    text,
                    vector,
                    query_embedder,
                    spec.top_k,
                    crag_margin,
                    alpha,
                    max_iters,
                )
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
                "vector_source": vector_source,
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

    // P0-2: two hits that refute each other must never both reach the caller
    // as confirmed premises.
    let (hits, conflict_resolved, unresolved_conflict, tied_conflicts, excluded) =
        resolve_falsifies_conflicts(hits);
    // P1-2: fold each surviving hit's payload to what `token_budget` allows.
    let (hits, tokens_used, token_budget_dropped_hits) =
        apply_token_budget(graph, hits, spec.token_budget, &excluded);

    // `anchors` is the unfiltered set from `hybrid_rag_search`; every anchor's
    // node_id is also in `hits` before filtering (see `HybridRagResult::hits`'s
    // doc comment), so the same Falsifies-conflict exclusion set applies here.
    let anchors: Vec<Value> = anchors
        .iter()
        .filter(|&&(id, _)| !excluded.contains(&id))
        .map(|&(id, distance)| {
            let mut v = node_json(graph, id);
            v["distance"] = json!(distance);
            v
        })
        .collect();
    let diffusion_json = diffusion.as_ref().map(|d| {
        json!({
            "alpha": d.alpha,
            "max_iters": d.max_iters,
            "tolerance": d.tolerance,
            "iterations": d.iterations,
            "residual": d.residual,
            "converged": d.converged,
            "quality": d.quality.as_str(),
        })
    });
    let diffusion_quality = diffusion.as_ref().map(|d| d.quality);
    let mut out = json!({
        "query": query,
        "hits": hits,
        "anchors": anchors,
        "stage1_candidates": stage1_candidates,
        "searchable_nodes": searchable_nodes,
        "distance_normalization": "per_track_max",
        "diffusion": diffusion_json,
        "top_k": spec.top_k,
        "crag_margin": crag_margin,
        "conflict_resolved": conflict_resolved,
        "unresolved_conflict": unresolved_conflict,
        "tied_conflicts": tied_conflicts,
        // `null` when no anchor was found, so no diffusion ran.
        "diffusion_quality": diffusion_quality.map(DiffusionQuality::as_str),
    });
    qwen.report(&mut out, "qwen_embedded", json!(qwen.query.is_some()));
    // A diffusion that did not converge is never handed back looking normal.
    if let Some(d) = diffusion.filter(|d| d.quality == DiffusionQuality::Degraded) {
        tracing::warn!(
            iterations = d.iterations,
            max_iters = d.max_iters,
            residual = d.residual,
            tolerance = d.tolerance,
            "graph_rag: PPR diffusion degraded; hits ranked by an unconverged iterate"
        );
        out["non_converged_warning"] = json!("PPR residual exceeded tolerance");
    }
    if let Some(budget) = spec.token_budget {
        out["token_budget"] = json!(budget);
        out["estimated_tokens_used"] = json!(tokens_used);
        out["selection_strategy"] = json!("lod_budget_fit");
        out["token_budget_dropped_hits"] = json!(token_budget_dropped_hits);
    }
    Ok((summary, out))
}

/// Drop the losing side of every `Falsifies` conflict among `hits`, so the
/// response never carries two nodes that refute each other as if both were
/// confirmed. Equal confidence cannot be resolved by this rule: both sides
/// are dropped and `unresolved_conflict` is raised instead of guessing which
/// one a downstream reasoner should trust; the dropped pair is still recorded
/// in `tied_conflicts` so a caller can see which entities were lost and why.
///
/// Each `Falsifies` edge is judged on its own two endpoints' confidence, not
/// on the whole conflict graph at once; a node already excluded by one
/// conflict can still cause another node to be excluded by a separate one.
///
/// The returned `HashSet<u32>` of excluded node_ids is the same exclusion set
/// applied to `hits`; callers must apply it to any other view derived from
/// the same unfiltered node set (e.g. `anchors`) so a losing/falsified node
/// never leaks out through a different field.
fn resolve_falsifies_conflicts(
    hits: Vec<RagHit>,
) -> (Vec<RagHit>, Vec<Value>, bool, Vec<Value>, HashSet<u32>) {
    // Unordered pairs, once each, from the conflict edges the search read
    // under its own lock.
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let conflicts: Vec<(u32, u32)> = hits
        .iter()
        .flat_map(|h| {
            h.conflict_edges.iter().map(move |e| {
                let other = e.counterpart_node_id;
                (h.node_id.min(other), h.node_id.max(other))
            })
        })
        .filter(|pair| seen.insert(*pair))
        .collect();
    if conflicts.is_empty() {
        return (hits, Vec::new(), false, Vec::new(), HashSet::new());
    }
    let by_id: HashMap<u32, &RagHit> = hits.iter().map(|h| (h.node_id, h)).collect();
    const CONFIDENCE_EPS: f32 = 1e-6;
    let mut excluded: HashSet<u32> = HashSet::new();
    let mut unresolved_conflict = false;
    let mut conflict_resolved = Vec::new();
    let mut tied_conflicts = Vec::new();
    for (a, b) in conflicts {
        let ha = by_id[&a];
        let hb = by_id[&b];
        if (ha.confidence - hb.confidence).abs() <= CONFIDENCE_EPS {
            unresolved_conflict = true;
            excluded.insert(a);
            excluded.insert(b);
            tied_conflicts.push(json!({
                "a": ha.entity_id,
                "b": hb.entity_id,
                "edge": "Falsifies",
            }));
        } else if ha.confidence > hb.confidence {
            excluded.insert(b);
            conflict_resolved.push(json!({
                "winner": ha.entity_id,
                "loser": hb.entity_id,
                "edge": "Falsifies",
            }));
        } else {
            excluded.insert(a);
            conflict_resolved.push(json!({
                "winner": hb.entity_id,
                "loser": ha.entity_id,
                "edge": "Falsifies",
            }));
        }
    }
    let kept = hits
        .into_iter()
        .filter(|h| !excluded.contains(&h.node_id))
        .collect();
    (
        kept,
        conflict_resolved,
        unresolved_conflict,
        tied_conflicts,
        excluded,
    )
}

/// Estimated LLM tokens in `text`: `ceil(chars / 4)`. This is a budgeting
/// approximation, not a model-specific tokenizer; `estimated_tokens_used` in
/// the response is always computed with this same estimate, so it is internally
/// consistent even though it is not the exact count any particular model
/// would charge.
fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

/// One `graph_rag` hit as JSON. With `summary`, the payload, its source and
/// timestamp come from that `CoarseGrain` ancestor instead of `h`'s own node,
/// and the hit is tagged `payload_band: "summary"`; so are its identity and
/// trust: `entity_id`, `confidence` (the lesser of `h`'s and the summary's,
/// never overstating trust in the substituted text) and `status` all come
/// from the summary node itself, not from `h`. `node` stays `h.node_id`
/// unconditionally: it names the graph-internal node actually recalled,
/// regardless of which payload variant was chosen. PPR score, anchor
/// distance/match, aliases and `band` (the hit's own LOD level, 0..=3;
/// a separate key from `payload_band` — reusing `band` here would silently
/// overwrite that field with a string) still describe `h` itself.
fn hit_json(h: &RagHit, summary: Option<&LodNode>, excluded: &HashSet<u32>) -> Value {
    let (payload, source_uri, timestamp_ns, payload_digest) = match summary {
        Some(s) => (
            s.payload.as_deref(),
            s.source_uri.as_deref(),
            s.timestamp_ns,
            &s.payload_digest,
        ),
        None => (
            h.payload.as_deref(),
            h.source_uri.as_deref(),
            h.timestamp_ns,
            &h.payload_digest,
        ),
    };
    let (entity_id, confidence, status) = match summary {
        Some(s) => (s.entity_id, h.confidence.min(s.confidence), s.status),
        None => (h.entity_id, h.confidence, h.status),
    };
    let mut v = json!({
        "node": h.node_id,
        "entity_id": entity_id,
        "label": h.label,
        "status": status_name(status),
        "band": band_level(h.band),
        "confidence": confidence,
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
        "payload_digest": payload.map(|_| digest_hex(payload_digest)),
        "timestamp_ns": payload.map(|_| timestamp_ns),
        "payload": payload,
        "source_uri": source_uri,
        "conflict_edges": h.conflict_edges.iter().map(|e| json!({
            "counterpart_entity_id": e.counterpart_entity_id,
            "direction": e.direction.as_str(),
            "weight": e.weight,
            "edge": "Falsifies",
            // false: the counterpart lost the conflict and is not in `hits`.
            "counterpart_kept": !excluded.contains(&e.counterpart_node_id),
        })).collect::<Vec<Value>>(),
    });
    if summary.is_some() {
        v["payload_band"] = json!("summary");
    }
    v
}

/// Fold `hits` (already ranked by relevance, most relevant first) to fit
/// `token_budget`: a greedy budget fold, most-relevant-first, two variants
/// per hit. Each hit takes its own detailed payload if the remaining budget
/// covers it; short of that, its [`LodGraph::coarse_grain_summary_of`]
/// ancestor's payload if one exists, actually carries a payload, and fits;
/// failing both, the hit is dropped rather than silently truncated, and its
/// entity is reported in `token_budget_dropped_hits` so a caller can see
/// exactly what it lost rather than wondering why a result is thinner than
/// expected.
///
/// A summary node with no payload of its own (today, every summary a
/// `graph_coarse_grain` call creates: it names a cluster but carries no text)
/// is never substituted in: that would silently blank evidence out instead of
/// condensing it. Nor is a `Falsified` summary ever substituted in, even if it
/// carries a payload: that would surface a refuted node's text next to the
/// original hit's own confidence as if it were trustworthy. Either case is
/// treated the same as "no usable summary" and falls through to `dropped`.
///
/// `token_budget: None` keeps every hit's detailed payload, unchanged from
/// before this field existed.
fn apply_token_budget(
    graph: &LodGraph,
    hits: Vec<RagHit>,
    token_budget: Option<usize>,
    excluded: &HashSet<u32>,
) -> (Vec<Value>, usize, Vec<u64>) {
    let Some(budget) = token_budget else {
        return (
            hits.iter().map(|h| hit_json(h, None, excluded)).collect(),
            0,
            Vec::new(),
        );
    };
    let mut remaining = budget;
    let mut tokens_used = 0usize;
    let mut out = Vec::with_capacity(hits.len());
    let mut dropped = Vec::new();
    for hit in &hits {
        let detail_tokens = hit.payload.as_deref().map(estimate_tokens).unwrap_or(0);
        if detail_tokens <= remaining {
            remaining -= detail_tokens;
            tokens_used += detail_tokens;
            out.push(hit_json(hit, None, excluded));
            continue;
        }
        let summary = graph
            .coarse_grain_summary_of(hit.node_id)
            .filter(|id| !excluded.contains(id))
            .and_then(|id| graph.get_node(id))
            .filter(|s| s.status != EpistemicStatus::Falsified);
        match summary.as_ref().and_then(|s| s.payload.as_deref()) {
            Some(text) if estimate_tokens(text) <= remaining => {
                let tokens = estimate_tokens(text);
                remaining -= tokens;
                tokens_used += tokens;
                out.push(hit_json(hit, summary.as_ref(), excluded));
            }
            _ => dropped.push(hit.entity_id),
        }
    }
    (out, tokens_used, dropped)
}

fn rag_summary(result: &HybridRagResult) -> String {
    match &result.diffusion {
        None => "graph_rag: no live node to recall".to_string(),
        Some(d) => format!(
            "graph_rag: {} hit(s) from {} anchor(s); PPR {} after {} iteration(s)",
            result.hits.len(),
            result.anchors.len(),
            d.quality.as_str(),
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
    let quality = DiffusionQuality::assess(ranking.converged, ranking.residual, tolerance);
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
    let mut out = json!({
        "results": results,
        "alpha": alpha,
        "max_iters": max_iters,
        "tolerance": tolerance,
        "iterations": ranking.iterations,
        "residual": ranking.residual,
        "converged": ranking.converged,
        "diffusion_quality": quality.as_str(),
    });
    // A diffusion that did not converge is never handed back looking normal.
    if quality == DiffusionQuality::Degraded {
        tracing::warn!(
            iterations = ranking.iterations,
            max_iters,
            residual = ranking.residual,
            tolerance,
            "graph_ppr: PPR diffusion degraded; scores are an unconverged iterate"
        );
        out["non_converged_warning"] = json!("PPR residual exceeded tolerance");
    }
    Ok((summary, out))
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

/// Run a verb as one transaction, or with `dry_run` on a private copy that is
/// never published: no reader sees a trial state, and nothing is persisted.
fn apply<T>(
    graph: &LodGraph,
    dry_run: bool,
    f: impl FnOnce(&LodGraph) -> Result<T, LodError>,
) -> Result<T, LodError> {
    if dry_run {
        graph.dry_run(f)
    } else {
        graph.transact(f)
    }
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
    let (pruned, revoked, retracted, fixed_point) = apply(graph, dry_run, |g| {
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
        Ok((pruned, revoked, retracted, fixed_point))
    })
    .map_err(graph_rejection)?;
    let summary = format!(
        "graph_prune: {} node(s) {} from entity {e}",
        revoked.len(),
        if dry_run {
            "would be pruned (dry run, not applied)"
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
    let fixed_point = apply(graph, dry_run, |g| {
        for &(_, node) in &retract {
            g.retract_falsification(node)?;
        }
        let report = params.run(g)?;
        let fixed_point = fixed_point_json(g, params, &report);
        Ok(fixed_point)
    })
    .map_err(graph_rejection)?;
    let summary = format!(
        "graph_evolve: fixed point in {} step(s) (bound {}), {} status change(s){}",
        fixed_point["iterations"],
        fixed_point["k_max"],
        fixed_point["transitions_total"],
        if dry_run {
            " (dry run, not applied)"
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
    let (summary, member_nodes, csr_edges) = apply(graph, dry_run, |g| {
        if g.node_count() + 1 > MAX_GRAPH_NODES {
            return Err(LodError::InvalidNode(format!(
                "the live graph is capped at {MAX_GRAPH_NODES} nodes"
            )));
        }
        let id = g.coarse_grain_cluster(&members, summary_entity, summary_coord, spec.hdc)?;
        let summary = node_json(g, id);
        let member_nodes: Vec<Value> = members.iter().map(|&m| node_json(g, m)).collect();
        let csr_edges = g.csr_snapshot().num_edges();
        Ok((summary, member_nodes, csr_edges))
    })
    .map_err(graph_rejection)?;
    let summary_line = format!(
        "graph_coarse_grain: {} member(s) {} under entity {summary_entity} at band {}",
        members.len(),
        if dry_run {
            "would be coarse-grained (dry run, not applied)"
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
    let (from, to, node_after) = apply(graph, dry_run, |g| {
        let (from, to) = match direction {
            Some(d) => {
                let from = g.get_node(node).ok_or(LodError::NodeNotFound(node))?.band;
                (from, g.zoom_node(node, d)?)
            }
            None => g.migrate_band_to_coord(node)?,
        };
        let node_after = node_json(g, node);
        Ok((from, to, node_after))
    })
    .map_err(graph_rejection)?;
    let summary = format!(
        "graph_zoom: entity {e} band {} -> {}{}",
        band_level(from),
        band_level(to),
        if dry_run {
            " (dry run, not applied)"
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecuteOperatorSpec {
    node_id: u64,
    nonce: Option<String>,
    input: Value,
}
fn execute_operator(
    graph: &LodGraph,
    spec: ExecuteOperatorSpec,
) -> Result<(String, Value), Rejection> {
    let node_id = u32::try_from(spec.node_id).map_err(|_| invalid("node_id exceeds u32"))?;
    let nonce = spec
        .nonce
        .as_deref()
        .map(|text| {
            if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(invalid(
                    "nonce must be exactly 64 hexadecimal characters (32 bytes)",
                ));
            }
            let mut bytes = [0u8; 32];
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)
                    .map_err(|_| invalid("invalid nonce hex"))?;
            }
            Ok(bytes)
        })
        .transpose()?;
    let encoded = serde_json::to_vec(&json!({"node_id": node_id, "input": spec.input}))
        .map_err(|e| invalid(e.to_string()))?;
    let input = gen_zero_lod::OperatorInput {
        node_id,
        parameters: spec.input,
        nonce,
        context_digest: *blake3::hash(&encoded).as_bytes(),
    };
    let execution = graph
        .execute_operator(node_id, &input)
        .map_err(graph_rejection)?;
    if execution.signature.operator_kind == gen_zero_lod::OperatorKind::HardDcm {
        tracing::warn!(
            node_id,
            persistent_graph = graph.is_persistent(),
            "operator replay protection is process-local; nonce history does not survive restart"
        );
    }
    let result = serde_json::to_value(execution).map_err(|e| invalid(e.to_string()))?;
    Ok((
        format!("graph_execute_operator: node {node_id} executed"),
        result,
    ))
}
