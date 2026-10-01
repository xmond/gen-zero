//! gen-zero-lod dynamic graph topology, graph geometry and epistemic lifecycle.
//!
//! Readers traverse an immutable CSR snapshot (`ArcSwap<CsrGraph>`) without locks.
//! Writers append ticketed edges to a pending buffer. `flush_edges_to_csr` merges
//! only the pending edges into a new snapshot and drains them from the buffer, so
//! the buffer never holds committed history.
//!
//! Every distance the graph evaluates uses the [`GeometryParams`] the graph was
//! built with (`LodGraph::with_geometry`): curvature, sphere radius and the three
//! metric weights. There is no other metric in this module.
//!
//! Confidence is the fixed point of
//! `c = (1 - beta) pi + beta max(0, P+ c - gamma P- c)`: `P+` over the
//! `DependsOn` / `CausalTransition` / `CoarseGrain` edges, `P-` over the
//! `Falsifies` edges (`evolve_signed_epistemic_fixed_point_within`). The
//! dependency graph is cut into strongly connected components (Tarjan) and
//! solved sources first: a node on no cycle in one evaluation, a cycle by its
//! own contraction, refused when it has none. Admission keeps such cycles out:
//! an edge that would close one at [`ADMISSION_BETA`], [`ADMISSION_GAMMA`], or
//! one too slow to finish in [`MAX_FIXED_POINT_STEPS`], is refused when it is
//! added. Evidence enters through
//! `falsify_node` and leaves through `retract_falsification`; the next
//! evolution moves every dependent accordingly, in either direction.
//!
//! `hybrid_rag_search` chains the three retrieval stages under one read lock:
//! HDC Hamming prefilter, product-geodesic rerank, then PPR diffusion from the
//! reranked anchors, and returns each hit with its payload and source.
//! `hybrid_rag_search_query` first projects a query text, a query vector or
//! both with the graph's own [`TextEmbeddingProjector`], the one inserts use.
//!
//! A node has up to three kinds of retrieval anchor. Its own coordinate and
//! the lexical projection of each alias are chart anchors: text and coordinate
//! queries take the closest of them. The dense projection of its embedding is
//! the anchor vector queries search. The two kinds are never compared with
//! each other. Nodes that share an alias are linked by `Semantic` edges at
//! insert, so diffusion from one reaches the other.
//!
//! `create_checkpoint` / `rollback_checkpoint` restore the whole mutable graph state
//! atomically. Nothing here touches a search tree: the planner's MCTS is sequential
//! and has no virtual loss (see `gen-zero-planner/src/config.rs`).

mod persistence;

use crate::error::LodError;
use crate::manifold::{Epochs, GeometryParams, MixedCurvatureCoord, ProductManifold, Version};
use crate::node::{
    hdc_hamming_distance_256, ChartAnchor, EpistemicStatus, LodBand, LodNode, Placement,
    ZoomDirection,
};
use crate::ppr::compute_ppr_csr;
use crate::projection::{normalized, TextEmbeddingProjector};
use arc_swap::ArcSwap;
use gen_zero_core::GraphFactProvider;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Type of directional relational edge in the cognitive LodGraph.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Forward confirmation edge verifying a hypothesis.
    Validates = 0,
    /// Counterexample edge falsifying a hypothesis.
    Falsifies = 1,
    /// Forward temporal/causal action state transition: s --[a]--> s'.
    CausalTransition = 2,
    /// Semantic association in latent concept space.
    Semantic = 3,
    /// Coarse-graining link between hierarchical Lod bands.
    CoarseGrain = 4,
    /// Prerequisite dependency: Target depends on Source.
    DependsOn = 5,
}

/// A buffered edge pending compaction into CSR snapshot.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BufferedEdge {
    pub source: u32,
    pub target: u32,
    pub edge_type: EdgeType,
    pub weight: f32,
    pub ticket: u64,
}

/// Immutable Compressed Sparse Row (CSR) Graph Snapshot, traversed lock-free
/// by PPR and the CRAG expansion of two-stage recall.
///
/// Fields are private so the CSR invariants hold for every value:
/// `row_offsets.len() == num_nodes + 1`, `row_offsets` is non-decreasing and
/// ends at `col_indices.len()`, the three edge arrays share one length, every
/// target is a node, and every weight is finite and nonnegative.
#[derive(Clone, Debug, Serialize)]
pub struct CsrGraph {
    num_nodes: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<u32>,
    edge_weights: Vec<f32>,
    edge_types: Vec<EdgeType>,
}

// Deserialization is a constructor too: never expose an invalid CSR value to
// callers, even when they deserialize outside the snapshot loader.
impl<'de> Deserialize<'de> for CsrGraph {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Fields {
            num_nodes: usize,
            row_offsets: Vec<usize>,
            col_indices: Vec<u32>,
            edge_weights: Vec<f32>,
            edge_types: Vec<EdgeType>,
        }
        let fields = Fields::deserialize(deserializer)?;
        let graph = Self {
            num_nodes: fields.num_nodes,
            row_offsets: fields.row_offsets,
            col_indices: fields.col_indices,
            edge_weights: fields.edge_weights,
            edge_types: fields.edge_types,
        };
        graph.validate().map_err(serde::de::Error::custom)?;
        Ok(graph)
    }
}

impl Default for CsrGraph {
    // A derived default would leave `row_offsets` empty and break the invariant.
    fn default() -> Self {
        Self::empty()
    }
}

impl CsrGraph {
    /// Create an empty CSR graph.
    pub fn empty() -> Self {
        Self {
            num_nodes: 0,
            row_offsets: vec![0],
            col_indices: Vec::new(),
            edge_weights: Vec::new(),
            edge_types: Vec::new(),
        }
    }

    /// Construct a CSR snapshot from buffered edges. An edge whose endpoint is
    /// outside `num_nodes` is an error, never dropped.
    pub fn from_edges(num_nodes: usize, edges: &[BufferedEdge]) -> Result<Self, LodError> {
        Self::empty().merged(num_nodes, edges)
    }

    /// New snapshot holding this snapshot's edges plus `delta`, over `num_nodes`
    /// nodes (never fewer than this snapshot has). Within a row, existing edges keep
    /// their order and delta edges follow in slice order.
    ///
    /// Cost is O(num_nodes + existing edges + delta): an immutable CSR array is
    /// rebuilt, not patched. What this avoids is re-reading the whole edge history.
    pub fn merged(&self, num_nodes: usize, delta: &[BufferedEdge]) -> Result<Self, LodError> {
        if num_nodes < self.num_nodes {
            return Err(LodError::CsrInvariant(format!(
                "cannot shrink a snapshot from {} to {num_nodes} nodes",
                self.num_nodes
            )));
        }
        for edge in delta {
            check_edge(num_nodes, edge.source, edge.target, edge.weight)?;
        }
        if num_nodes == 0 {
            return Ok(Self::empty());
        }

        let mut degrees = vec![0usize; num_nodes];
        for (u, degree) in degrees.iter_mut().enumerate().take(self.num_nodes) {
            *degree = self.row_offsets[u + 1] - self.row_offsets[u];
        }
        for edge in delta {
            degrees[edge.source as usize] += 1;
        }

        let mut row_offsets = Vec::with_capacity(num_nodes + 1);
        row_offsets.push(0);
        let mut current_offset = 0;
        for &d in &degrees {
            current_offset += d;
            row_offsets.push(current_offset);
        }

        let total_edges = current_offset;
        let mut col_indices = vec![0u32; total_edges];
        let mut edge_weights = vec![0.0_f32; total_edges];
        let mut edge_types = vec![EdgeType::Semantic; total_edges];

        let mut insert_cursor = row_offsets[..num_nodes].to_vec();
        for (cursor, row) in insert_cursor.iter_mut().zip(self.row_offsets.windows(2)) {
            let (start, end) = (row[0], row[1]);
            let (at, len) = (*cursor, end - start);
            col_indices[at..at + len].copy_from_slice(&self.col_indices[start..end]);
            edge_weights[at..at + len].copy_from_slice(&self.edge_weights[start..end]);
            edge_types[at..at + len].copy_from_slice(&self.edge_types[start..end]);
            *cursor += len;
        }
        for edge in delta {
            let u = edge.source as usize;
            let idx = insert_cursor[u];
            col_indices[idx] = edge.target;
            edge_weights[idx] = edge.weight;
            edge_types[idx] = edge.edge_type;
            insert_cursor[u] += 1;
        }

        let csr = Self {
            num_nodes,
            row_offsets,
            col_indices,
            edge_weights,
            edge_types,
        };
        csr.validate()?;
        Ok(csr)
    }

    /// Check every CSR invariant. A snapshot that fails is never published.
    pub fn validate(&self) -> Result<(), LodError> {
        let fail = |detail: String| Err(LodError::CsrInvariant(detail));
        let edges = self.col_indices.len();
        if self.num_nodes.checked_add(1) != Some(self.row_offsets.len()) {
            return fail(format!(
                "{} row offsets for {} nodes",
                self.row_offsets.len(),
                self.num_nodes
            ));
        }
        if self.row_offsets[0] != 0 || self.row_offsets[self.num_nodes] != edges {
            return fail("row offsets must start at 0 and end at the edge count".into());
        }
        if self.row_offsets.windows(2).any(|w| w[0] > w[1]) {
            return fail("row offsets decrease".into());
        }
        if self.edge_weights.len() != edges || self.edge_types.len() != edges {
            return fail("edge arrays differ in length".into());
        }
        if let Some(v) = self
            .col_indices
            .iter()
            .find(|&&v| v as usize >= self.num_nodes)
        {
            return fail(format!("edge target {v} outside {} nodes", self.num_nodes));
        }
        if let Some(w) = self
            .edge_weights
            .iter()
            .find(|w| !(w.is_finite() && **w >= 0.0))
        {
            return fail(format!("edge weight {w} is not finite and nonnegative"));
        }
        Ok(())
    }

    /// Number of edges in this snapshot.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.col_indices.len()
    }

    /// Number of nodes covered by this snapshot.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Row offsets (a.k.a. row pointers), length `num_nodes + 1`.
    #[inline]
    pub fn row_ptrs(&self) -> &[usize] {
        &self.row_offsets
    }

    /// Target node of each edge, grouped by source row.
    #[inline]
    pub fn col_indices(&self) -> &[u32] {
        &self.col_indices
    }

    /// Weight of each edge, parallel to [`Self::col_indices`].
    #[inline]
    pub fn edge_weights(&self) -> &[f32] {
        &self.edge_weights
    }

    /// Type of each edge, parallel to [`Self::col_indices`].
    #[inline]
    pub fn edge_types(&self) -> &[EdgeType] {
        &self.edge_types
    }

    /// Outgoing neighbor iterator for node u: (target_node, edge_type, weight).
    pub fn neighbors(&self, u: u32) -> impl Iterator<Item = (u32, EdgeType, f32)> + '_ {
        let u_idx = u as usize;
        let (start, end) = if u_idx + 1 < self.row_offsets.len() {
            let s = self.row_offsets[u_idx].min(self.col_indices.len());
            let e = self.row_offsets[u_idx + 1].min(self.col_indices.len());
            if s <= e {
                (s, e)
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };

        (start..end).map(move |i| {
            (
                self.col_indices[i],
                self.edge_types[i],
                self.edge_weights[i],
            )
        })
    }
}

/// An edge endpoint must be an existing node and its weight finite and nonnegative.
fn check_edge(num_nodes: usize, source: u32, target: u32, weight: f32) -> Result<(), LodError> {
    if source as usize >= num_nodes || target as usize >= num_nodes {
        return Err(LodError::InvalidEdge(format!(
            "edge {source} -> {target} names a node outside the {num_nodes} in the graph"
        )));
    }
    if !(weight.is_finite() && weight >= 0.0) {
        return Err(LodError::InvalidEdge(format!(
            "edge {source} -> {target} weight {weight} must be finite and nonnegative"
        )));
    }
    Ok(())
}

/// Nodes per copy-on-write chunk, and per persisted node block.
const NODE_CHUNK: usize = 128;

/// Nodes in copy-on-write chunks of [`NODE_CHUNK`]. A clone shares every
/// chunk and a write copies only the chunk it touches, so the nodes of a
/// transaction candidate cost `O(nodes / NODE_CHUNK)` to share, and
/// persistence finds the chunks a commit changed by pointer identity alone.
#[derive(Clone, Default)]
struct NodeChunks {
    chunks: Vec<Arc<Vec<LodNode>>>,
    len: usize,
}

impl NodeChunks {
    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, id: usize) -> Option<&LodNode> {
        (id < self.len).then(|| &self.chunks[id / NODE_CHUNK][id % NODE_CHUNK])
    }

    /// Copies the node's chunk first when another state shares it.
    fn get_mut(&mut self, id: usize) -> Option<&mut LodNode> {
        (id < self.len)
            .then(|| &mut Arc::make_mut(&mut self.chunks[id / NODE_CHUNK])[id % NODE_CHUNK])
    }

    fn push(&mut self, node: LodNode) {
        if self.len % NODE_CHUNK == 0 {
            self.chunks.push(Arc::new(Vec::with_capacity(NODE_CHUNK)));
        }
        let last = self.chunks.last_mut().expect("a chunk was just ensured");
        Arc::make_mut(last).push(node);
        self.len += 1;
    }

    fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        self.chunks.truncate(len.div_ceil(NODE_CHUNK));
        if len % NODE_CHUNK != 0 {
            let last = self.chunks.last_mut().expect("len > 0 keeps a chunk");
            Arc::make_mut(last).truncate(len % NODE_CHUNK);
        }
        self.len = len;
    }

    fn iter(&self) -> impl Iterator<Item = &LodNode> + '_ {
        self.chunks.iter().flat_map(|chunk| chunk.iter())
    }

    fn chunks(&self) -> &[Arc<Vec<LodNode>>] {
        &self.chunks
    }

    /// Reassemble restored chunks. Every chunk but the last must be full.
    fn from_chunks(chunks: Vec<Arc<Vec<LodNode>>>) -> Result<Self, LodError> {
        let mut len = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            if chunk.is_empty() || chunk.len() > NODE_CHUNK || (!last && chunk.len() != NODE_CHUNK)
            {
                return Err(LodError::Persistence(format!(
                    "node block {i} holds {} nodes, not a {NODE_CHUNK}-node chunk",
                    chunk.len()
                )));
            }
            len += chunk.len();
        }
        Ok(Self { chunks, len })
    }
}

impl std::ops::Index<usize> for NodeChunks {
    type Output = LodNode;

    fn index(&self, id: usize) -> &LodNode {
        self.get(id).expect("node id out of range")
    }
}

impl std::ops::IndexMut<usize> for NodeChunks {
    fn index_mut(&mut self, id: usize) -> &mut LodNode {
        self.get_mut(id).expect("node id out of range")
    }
}

/// Add the checkpoint range `(after, upto]` to sorted, disjoint `discarded`.
fn discard_range(discarded: &mut Vec<(u64, u64)>, after: u64, upto: u64) {
    discarded.push((after, upto));
    discarded.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(discarded.len());
    for &(a, b) in discarded.iter() {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    *discarded = merged;
}

/// Mutable graph state. One lock guards all of it, so a checkpoint or a rollback
/// reads or restores a single consistent state.
///
/// Writers never change the live state in place: [`LodGraph::transact`] runs
/// on a private candidate cloned from it and publishes the candidate whole,
/// after a durable commit when the graph is persistent. Readers see only
/// published states. Cost of a candidate: nodes and both indexes are shared
/// copy-on-write, but pending edges, revocations, privileges and validated
/// dependencies are cloned, and the first insert copies the entity index, a
/// first alias the alias index: `O(nodes)` memory work, though no disk work.
///
/// Lock order: `txn_lock` -> `flush_lock` -> `state`. The CSR snapshot is stored
/// only while `state` is write-locked; methods that must see it consistent with
/// the nodes load it while holding `state`.
#[derive(Clone, Default)]
struct GraphState {
    nodes: NodeChunks,
    /// Shared copy-on-write: the first insert of a transaction copies the map.
    entity_index: Arc<HashMap<u64, u32>>,
    /// Edges not yet merged into the CSR snapshot, in ticket order.
    edge_buffer: Vec<BufferedEdge>,
    revocations: HashSet<u64>,
    /// Entities revoked through [`LodGraph::revoke_entity`]. A subset of
    /// `revocations` that no confidence evolution ever lifts.
    manual_revocations: HashSet<u64>,
    privileges: HashMap<u64, Vec<u32>>,
    validated_deps: HashSet<(u64, u64)>,
    /// Nodes holding each alias, keyed by the alias in normalized form.
    alias_index: Arc<HashMap<String, Vec<u32>>>,
    /// Dimension of every node embedding in this graph, fixed by the first one.
    embedding_dim: Option<usize>,
    /// Bumped by every CSR store and every rollback. A flush built against an
    /// older generation is discarded.
    generation: u64,
    /// Sorted, disjoint checkpoint sequence ranges `(after, upto]` whose state a
    /// rollback discarded. Checkpoints in them can no longer be restored.
    discarded: Vec<(u64, u64)>,
}

/// The node fields a graph method may change after insert. Label, coordinate,
/// fingerprint, aliases, embedding, entity and prior are fixed at insert.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NodeMutable {
    status: EpistemicStatus,
    confidence: f32,
    refuted: bool,
    band: LodBand,
    parent_id: Option<u32>,
}

impl NodeMutable {
    fn of(node: &LodNode) -> Self {
        Self {
            status: node.status,
            confidence: node.confidence,
            refuted: node.refuted,
            band: node.band,
            parent_id: node.parent_id,
        }
    }

    fn restore(self, node: &mut LodNode) {
        node.status = self.status;
        node.confidence = self.confidence;
        node.refuted = self.refuted;
        node.band = self.band;
        node.parent_id = self.parent_id;
    }
}

/// Everything a rollback restores: the node count, each node's mutable fields
/// ([`NodeMutable`]: status, confidence, refutation mark, band and parent), the
/// CSR snapshot reference, pending edges, revocations, privileges, validated
/// dependencies and the embedding dimension. The alias index follows the nodes.
/// The edge ticket counter is never rewound, so tickets stay unique.
///
/// Memory is O(nodes + pending edges + revocations + dependencies) per checkpoint.
#[derive(Clone, Debug)]
pub struct GraphCheckpoint {
    graph_id: u64,
    seq: u64,
    node_states: Vec<NodeMutable>,
    csr: Arc<CsrGraph>,
    edge_buffer: Vec<BufferedEdge>,
    revocations: HashSet<u64>,
    manual_revocations: HashSet<u64>,
    privileges: HashMap<u64, Vec<u32>>,
    validated_deps: HashSet<(u64, u64)>,
    embedding_dim: Option<usize>,
}

impl GraphCheckpoint {
    /// Number of nodes the graph had when this checkpoint was taken.
    pub fn node_count(&self) -> usize {
        self.node_states.len()
    }
}

/// Result of one [`LodGraph::flush_edges_to_csr`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlushReport {
    /// Pending edges merged into the new snapshot and drained from the buffer.
    pub merged_edges: usize,
    /// Nodes covered by the published snapshot.
    pub csr_nodes: usize,
    /// Edges in the published snapshot.
    pub csr_edges: usize,
    /// Edges appended while the flush was building; they wait for the next flush.
    pub pending_edges: usize,
}

/// Most steps one cyclic block of a confidence evolution may take. A block whose
/// bound `k_max` is larger fails with [`LodError::FixedPointDiverged`] once it has
/// used this many.
pub const MAX_FIXED_POINT_STEPS: usize = 100_000;

/// Falsification gain `gamma` of [`LodGraph::evolve_epistemic_fixed_point`]: a
/// fully confident falsifier cancels a fully confident support.
pub const DEFAULT_FALSIFICATION_GAIN: f32 = 1.0;

/// Damping `beta` every edge is admitted against and every reflection evolves
/// with. See [`LodGraph::add_edge`].
pub const ADMISSION_BETA: f32 = 0.85;

/// Falsification gain every edge is admitted against and every reflection
/// evolves with.
pub const ADMISSION_GAMMA: f32 = DEFAULT_FALSIFICATION_GAIN;

/// Tolerance and hysteresis band of [`LodGraph::reflect_failure`].
const REFLECTION_TOLERANCE: f32 = 1e-6;
const REFLECTION_THETA_LO: f32 = 0.3;
const REFLECTION_THETA_HI: f32 = 0.6;

/// What a cyclic block whose Lipschitz bound is `>= 1` does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OnNonContractive {
    /// Refuse the whole evolution with [`LodError::FixedPointNotContractive`].
    Refuse,
    /// Lower the gain of the falsifier edges inside the block until it
    /// contracts, and report it in [`FixedPointReport::adapted_blocks`].
    AdaptGain,
}

/// A cyclic block whose internal falsifier gain an evolution lowered, because at
/// the requested gain its map was not a contraction, or not one that surely
/// converges inside the step budget.
///
/// Only falsifier edges with both ends inside the block run at `applied_gamma`.
/// Support edges, and falsifiers from outside the block, keep their weight. The
/// fixed point of such a block is the one of the weaker map, not of the
/// requested one: the report is how a caller sees that.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptedBlock {
    /// Node ids of the block.
    pub nodes: Vec<u32>,
    pub requested_gamma: f64,
    pub applied_gamma: f64,
    /// Lipschitz bound at `requested_gamma`: `>= 1`, or below 1 but so close
    /// that the block may not converge inside its step budget.
    pub requested_contraction: f64,
    /// Lipschitz bound at `applied_gamma` (`< 1`).
    pub contraction: f64,
}

/// How [`LodGraph::reflect_failure`] revoked the action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectionRevocation {
    /// The evolution left the action's node `Falsified`: its confidence fell
    /// below the threshold, or it was already refuted by evidence.
    Evolution,
    /// The evolution left the action at or above the threshold (other support
    /// outweighs the observation). The action is revoked by hand, as
    /// [`LodGraph::revoke_entity`] does, and stays revoked.
    Quarantine,
}

/// Result of one [`LodGraph::reflect_failure`].
#[derive(Clone, Debug)]
pub struct ReflectionReport {
    /// Node of the failure observation.
    pub evidence: u32,
    /// Node of the action.
    pub target: u32,
    pub revocation: ReflectionRevocation,
    /// Confidence of the action at the fixed point.
    pub target_confidence: f32,
    pub evolution: FixedPointReport,
}

/// One status change made by a confidence evolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatusTransition {
    pub node: u32,
    pub entity_id: u64,
    pub from: EpistemicStatus,
    pub to: EpistemicStatus,
    /// The node's confidence at the fixed point.
    pub confidence: f32,
}

/// Result of one [`LodGraph::evolve_signed_epistemic_fixed_point_within`].
///
/// `iterations`, `k_max`, `initial_delta` and `residual` describe one cyclic
/// block: the one with the largest `k_max` (the first such in evaluation order).
/// With no cyclic block they are 0.
#[derive(Clone, Debug)]
pub struct FixedPointReport {
    pub beta: f32,
    /// Falsification gain: how strongly `Falsifies` edges press their targets.
    pub gamma: f32,
    pub tolerance: f32,
    pub theta_lo: f32,
    pub theta_hi: f32,
    /// Nodes in the iteration.
    pub nodes: usize,
    /// Nodes held at their prior by evidence: axioms at 1, refuted nodes at 0.
    pub pinned: usize,
    /// Support edges (`DependsOn` / `CausalTransition` / `CoarseGrain`, positive
    /// weight, into an unpinned node) `P+` was built from.
    pub dependency_edges: usize,
    /// `Falsifies` edges (positive weight, into an unpinned node) `P-` was built
    /// from. 0 when `gamma` is 0: the edges are then left out of the graph.
    pub falsification_edges: usize,
    /// Strongly connected components of the dependency graph (Tarjan), each
    /// solved once in topological order, sources first.
    pub scc_count: usize,
    /// Components of one node without a self-loop: evaluated in one step.
    pub trivial_scc_count: usize,
    /// Components with a cycle: iterated to their own fixed point.
    pub cyclic_scc_count: usize,
    pub max_scc_size: usize,
    /// Largest Lipschitz bound `q` of the map on a cyclic block (0 without one).
    /// Every block was checked to have `q < 1`.
    pub contraction: f64,
    /// Node evaluations made: one per trivial node, `|S| (k + 1)` per cyclic
    /// block `S` that stopped at step `k`.
    pub node_updates: usize,
    /// The `k` of the first iterate with `||c^{k+1} - c^k||_inf < tolerance` on
    /// the reported block. Never above its `k_max`.
    pub iterations: usize,
    /// `ceil(ln(tolerance (1 - q) / ||c^1 - c^0||_inf) / ln q)` on the reported
    /// block, or 0 when its first step is already below the tolerance.
    pub k_max: usize,
    /// `||c^1 - c^0||_inf` on the reported block.
    pub initial_delta: f64,
    /// `||c^{k+1} - c^k||_inf` at the stop of the reported block.
    pub residual: f64,
    /// A-posteriori bound on the max-norm distance from the committed
    /// confidences to the exact fixed point. Each block adds its own stop error
    /// `q r / (1 - q)` to the error it inherits from the blocks it reads.
    pub error_bound: f64,
    /// Status changes, in node order.
    pub transitions: Vec<StatusTransition>,
    /// Entities revoked because their node became `Falsified`.
    pub revoked_entities: Vec<u64>,
    /// Entities no longer revoked because their node left `Falsified`.
    pub reinstated_entities: Vec<u64>,
    /// Validated dependencies dropped and gained by the status changes.
    pub retracted_dependencies: usize,
    pub added_dependencies: usize,
    /// Cyclic blocks solved at a lowered internal falsifier gain. Always empty
    /// for [`LodGraph::evolve_signed_epistemic_fixed_point_within`], which
    /// refuses such a block; only [`LodGraph::reflect_failure`] adapts.
    pub adapted_blocks: Vec<AdaptedBlock>,
    /// State just before the evolution, taken under the same lock. Pass it to
    /// [`LodGraph::rollback_checkpoint`] to undo the evolution.
    pub checkpoint: GraphCheckpoint,
}

/// `P+` and `P-` of the confidence iteration, stored by dependent (row). Each
/// entry is a support edge or a falsifier edge into the row's node; support
/// probabilities sum to 1 over the row's support entries, falsifier
/// probabilities to 1 over its falsifier entries.
struct SignedRows {
    offsets: Vec<usize>,
    sources: Vec<u32>,
    probabilities: Vec<f64>,
    falsifies: Vec<bool>,
    /// The row has a support entry. A row without one is supported by the
    /// node's own prior.
    supported: Vec<bool>,
}

impl SignedRows {
    fn row(&self, v: usize) -> std::ops::Range<usize> {
        self.offsets[v]..self.offsets[v + 1]
    }

    /// Weight of entry `i` in the Lipschitz bound of its row: its probability,
    /// times `gamma` for a falsifier.
    fn lipschitz_weight(&self, i: usize, gamma: f64) -> f64 {
        if self.falsifies[i] {
            gamma * self.probabilities[i]
        } else {
            self.probabilities[i]
        }
    }
}

/// Strongly connected components of a graph in CSR form (`targets[offsets[v]..
/// offsets[v + 1]]` are the arcs out of `v`), by Tarjan's algorithm.
///
/// Iterative, so a long chain cannot overflow the call stack. Tarjan emits a
/// component only after every component its arcs reach, so with arcs from a
/// dependent to its sources the output lists sources first: the evaluation order.
struct Condensation {
    /// Nodes of component `k` are `members[starts[k]..starts[k + 1]]`.
    starts: Vec<usize>,
    members: Vec<u32>,
}

impl Condensation {
    fn component_count(&self) -> usize {
        self.starts.len() - 1
    }

    fn component(&self, k: usize) -> &[u32] {
        &self.members[self.starts[k]..self.starts[k + 1]]
    }
}

fn tarjan_scc(offsets: &[usize], targets: &[u32]) -> Condensation {
    const UNVISITED: usize = usize::MAX;
    let n = offsets.len() - 1;
    let mut index = vec![UNVISITED; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    // Simulated recursion: (node, position of its next unexplored arc).
    let mut calls: Vec<(usize, usize)> = Vec::new();
    let mut next_index = 0;
    let mut members = Vec::with_capacity(n);
    let mut starts = vec![0];

    for root in 0..n {
        if index[root] != UNVISITED {
            continue;
        }
        // Number `v`, put it on the component stack and enter it.
        macro_rules! visit {
            ($v:expr) => {{
                let v: usize = $v;
                index[v] = next_index;
                low[v] = next_index;
                next_index += 1;
                stack.push(v as u32);
                on_stack[v] = true;
                calls.push((v, offsets[v]));
            }};
        }
        visit!(root);
        while let Some(&(v, arc)) = calls.last() {
            if arc < offsets[v + 1] {
                calls.last_mut().expect("frame just read").1 += 1;
                let w = targets[arc] as usize;
                if index[w] == UNVISITED {
                    visit!(w);
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            calls.pop();
            if let Some(&(parent, _)) = calls.last() {
                low[parent] = low[parent].min(low[v]);
            }
            if low[v] == index[v] {
                loop {
                    let w = stack.pop().expect("v is on the stack");
                    on_stack[w as usize] = false;
                    members.push(w);
                    if w as usize == v {
                        break;
                    }
                }
                starts.push(members.len());
            }
        }
    }
    Condensation { starts, members }
}

/// Gain of the falsifier entries whose source lies in the same block as their
/// row, when an [`OnNonContractive::AdaptGain`] evolution lowered it: the
/// block of every node, and the lowered gain.
type InnerGain<'a> = Option<(&'a [usize], f64)>;

/// `T_v(c) = (1 - beta) pi_v + beta max(0, P+_v c - gamma P-_v c)`, clamped to
/// [0, 1]. A row without support entries uses `pi_v` for `P+_v c`, written so
/// that a node with no falsifier either reproduces `pi_v` exactly. With `inner`,
/// falsifiers from `v`'s own block weigh its gain instead of `gamma`.
fn evaluate_node(
    rows: &SignedRows,
    prior: &[f64],
    beta: f64,
    gamma: f64,
    inner: InnerGain<'_>,
    c: &[f64],
    v: usize,
) -> f64 {
    let (mut support, mut outer, mut within) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in rows.row(v) {
        let u = rows.sources[i] as usize;
        let term = rows.probabilities[i] * c[u];
        if !rows.falsifies[i] {
            support += term;
        } else if inner.is_some_and(|(block_of, _)| block_of[u] == block_of[v]) {
            within += term;
        } else {
            outer += term;
        }
    }
    // Without `inner`, `within` is 0 and the penalty is `gamma outer` exactly.
    let penalty = match inner {
        Some((_, gain)) => gamma * outer + gain * within,
        None => gamma * outer,
    };
    let value = if rows.supported[v] {
        (1.0 - beta) * prior[v] + beta * (support - penalty).max(0.0)
    } else {
        // (1 - beta) pi + beta max(0, pi - penalty), rearranged.
        prior[v] - beta * prior[v].min(penalty)
    };
    value.clamp(0.0, 1.0)
}

/// How one cyclic block reached its fixed point.
#[derive(Clone, Copy, Debug)]
struct BlockRun {
    iterations: usize,
    k_max: usize,
    initial_delta: f64,
    residual: f64,
}

/// Iterate `T` on the nodes of one cyclic block `members`, every other node
/// held at its value in `c`, from the block's values in `c`, until
/// `||x^{k+1} - x^k||_inf < tolerance`. On success the block's entries of `c`
/// hold the last iterate.
///
/// `q < 1` is the Lipschitz bound of `T` on the block in the max norm, so
/// `||x^{k+1} - x^k|| <= q^k ||x^1 - x^0||` and the stop is reached by
/// `k_max = ceil(ln(tolerance (1 - q) / ||x^1 - x^0||) / ln q)`. A run that has
/// not stopped at `min(k_max, max_steps)`, or meets a non-finite value, is
/// [`LodError::FixedPointDiverged`].
#[allow(clippy::too_many_arguments)]
fn iterate_block(
    rows: &SignedRows,
    prior: &[f64],
    members: &[u32],
    q: f64,
    beta: f64,
    gamma: f64,
    inner: InnerGain<'_>,
    tolerance: f64,
    max_steps: usize,
    c: &mut [f64],
) -> Result<BlockRun, LodError> {
    let diverged = |iterations, k_max, residual| LodError::FixedPointDiverged {
        iterations,
        k_max,
        residual,
        tolerance,
    };
    let mut next = vec![0.0_f64; members.len()];
    // One Jacobi step into `next`; `||next - x||_inf`, or `None` on a non-finite value.
    let step = |c: &[f64], next: &mut [f64]| -> Option<f64> {
        let mut delta = 0.0_f64;
        for (out, &v) in next.iter_mut().zip(members) {
            let v = v as usize;
            *out = evaluate_node(rows, prior, beta, gamma, inner, c, v);
            let moved = (*out - c[v]).abs();
            if !(out.is_finite() && moved.is_finite()) {
                return None;
            }
            delta = delta.max(moved);
        }
        Some(delta)
    };
    let initial_delta = step(c, &mut next).ok_or_else(|| diverged(0, 0, f64::NAN))?;
    let k_max = if initial_delta < tolerance {
        0
    } else if q == 0.0 {
        1
    } else {
        // Both logarithms are negative, so the ratio is positive. The cast
        // saturates; a bound that large can only fail against `max_steps`.
        ((tolerance * (1.0 - q) / initial_delta).ln() / q.ln()).ceil() as usize
    };
    let limit = k_max.min(max_steps);
    let (mut iterations, mut residual) = (0, initial_delta);
    while residual >= tolerance {
        if iterations == limit {
            return Err(diverged(iterations, k_max, residual));
        }
        for (&v, &x) in members.iter().zip(&next) {
            c[v as usize] = x;
        }
        iterations += 1;
        residual = step(c, &mut next).ok_or_else(|| diverged(iterations, k_max, f64::NAN))?;
    }
    for (&v, &x) in members.iter().zip(&next) {
        c[v as usize] = x;
    }
    Ok(BlockRun {
        iterations,
        k_max,
        initial_delta,
        residual,
    })
}

/// The fixed point of `T` over the whole graph, solved block by block.
struct BlockSolve {
    confidences: Vec<f64>,
    scc_count: usize,
    trivial_scc_count: usize,
    cyclic_scc_count: usize,
    max_scc_size: usize,
    contraction: f64,
    node_updates: usize,
    /// The cyclic block with the largest `k_max`, if any.
    reported: Option<BlockRun>,
    error_bound: f64,
    adapted: Vec<AdaptedBlock>,
}

/// The strongly connected components of the dependency graph, and the
/// component of every node.
struct Blocks {
    condensation: Condensation,
    block_of: Vec<usize>,
}

impl Blocks {
    fn of(rows: &SignedRows) -> Self {
        let condensation = tarjan_scc(&rows.offsets, &rows.sources);
        let mut block_of = vec![0usize; rows.offsets.len() - 1];
        for k in 0..condensation.component_count() {
            for &v in condensation.component(k) {
                block_of[v as usize] = k;
            }
        }
        Self {
            condensation,
            block_of,
        }
    }

    /// The block has a cycle: more than one node, or one node with a self-loop.
    fn is_cyclic(&self, rows: &SignedRows, k: usize) -> bool {
        let members = self.condensation.component(k);
        members.len() > 1 || rows.sources[rows.row(members[0] as usize)].contains(&members[0])
    }

    /// Lipschitz data of block `k`: its bound `q = beta max_v sum_{u in S}
    /// |P_vu|` at `gamma`, the largest propagated upstream error
    /// `beta max_v sum_{u outside} |P_vu| e_u` (0 without `error`), and per
    /// member the support and the falsifier weight from inside the block, the
    /// latter before any gain.
    fn weights(
        &self,
        rows: &SignedRows,
        k: usize,
        beta: f64,
        gamma: f64,
        error: Option<&[f64]>,
    ) -> (f64, f64, Vec<(f64, f64)>) {
        let members = self.condensation.component(k);
        let mut q = 0.0_f64;
        let mut inherited = 0.0_f64;
        let mut inside = Vec::with_capacity(members.len());
        for &v in members {
            let (mut internal, mut external) = (0.0_f64, 0.0_f64);
            let (mut support, mut against) = (0.0_f64, 0.0_f64);
            for i in rows.row(v as usize) {
                let u = rows.sources[i] as usize;
                let weight = rows.lipschitz_weight(i, gamma);
                if self.block_of[u] == k {
                    internal += weight;
                    if rows.falsifies[i] {
                        against += rows.probabilities[i];
                    } else {
                        support += rows.probabilities[i];
                    }
                } else if let Some(error) = error {
                    external += weight * error[u];
                }
            }
            q = q.max(beta * internal);
            inherited = inherited.max(beta * external);
            inside.push((support, against));
        }
        (q, inherited, inside)
    }
}

/// The gain of a block's internal falsifier edges that makes it a contraction
/// with margin, and the Lipschitz bound at that gain.
///
/// Row `v` weighs `a_v + g b_v` inside the block (`a_v` support, `b_v`
/// falsifiers before gain). Support alone gives `beta a_max <= beta < 1`. The
/// target bound is `q* = (1 + beta a_max) / 2`, halfway to 1, reached by
/// `g = min(gamma, min_{b_v > 0} (q* / beta - a_v) / b_v)`; each row is
/// bounded by its own `a_v` and `b_v`, which is tighter than pairing the
/// largest `a` with the largest `b` when they lie on different rows.
///
/// Why not the limit `(1 / beta - a) / (b + eps)`: its bound sits within about
/// `eps` of 1, and `k_max` grows like `|ln tolerance| / eps`. For a small `eps`
/// that passes [`MAX_FIXED_POINT_STEPS`], and the refusal becomes a divergence.
///
/// A bound that is still not below 1 (non-finite weights) is `None`.
fn adapted_gain(beta: f64, gamma: f64, inside: &[(f64, f64)]) -> Option<(f64, f64)> {
    let support_max = inside.iter().map(|&(a, _)| a).fold(0.0, f64::max);
    let target = 0.5 * (1.0 + beta * support_max);
    let gain = inside
        .iter()
        .filter(|&&(_, b)| b > 0.0)
        .map(|&(a, b)| (target / beta - a) / b)
        .fold(gamma, f64::min)
        .max(0.0);
    let q = inside
        .iter()
        .map(|&(a, b)| beta * (a + gain * b))
        .fold(0.0, f64::max);
    (gain.is_finite() && q.is_finite() && q < 1.0).then_some((gain, q))
}

/// Steps a block with Lipschitz bound `q < 1` may need to reach `tolerance`
/// from any start: `k_max` of [`iterate_block`] at `||x^1 - x^0||_inf = 1`, the
/// largest it can be, since every confidence lies in [0, 1].
fn worst_case_steps(q: f64, tolerance: f64) -> usize {
    if q == 0.0 {
        1
    } else {
        // The cast saturates.
        ((tolerance * (1.0 - q)).ln() / q.ln()).ceil() as usize
    }
}

/// The first cyclic block admission refuses, as its error: one whose map is
/// not a contraction at [`ADMISSION_BETA`], [`ADMISSION_GAMMA`]
/// ([`LodError::FixedPointNotContractive`]), or contracts so slowly that an
/// evolution at the reflection tolerance may run out of
/// [`MAX_FIXED_POINT_STEPS`] ([`LodError::FixedPointTooSlow`]; about
/// `q > 0.9998`).
fn inadmissible_block(rows: &SignedRows) -> Option<LodError> {
    let (beta, gamma) = (f64::from(ADMISSION_BETA), f64::from(ADMISSION_GAMMA));
    let blocks = Blocks::of(rows);
    (0..blocks.condensation.component_count()).find_map(|k| {
        if !blocks.is_cyclic(rows, k) {
            return None;
        }
        let (q, _, _) = blocks.weights(rows, k, beta, gamma, None);
        let block_size = blocks.condensation.component(k).len();
        if q.is_nan() || q >= 1.0 {
            return Some(LodError::FixedPointNotContractive {
                block_size,
                contraction: q,
                beta,
                gamma,
            });
        }
        let k_max = worst_case_steps(q, f64::from(REFLECTION_TOLERANCE));
        (k_max > MAX_FIXED_POINT_STEPS).then_some(LodError::FixedPointTooSlow {
            block_size,
            contraction: q,
            k_max,
            max_steps: MAX_FIXED_POINT_STEPS,
        })
    })
}

/// Decompose the dependency graph into strongly connected components and solve
/// them in topological order, sources first.
///
/// - A component of one node without a self-loop reads only nodes already
///   solved: one evaluation of `T` gives its value, no iteration.
/// - A cyclic component `S` is iterated alone, its inputs from outside held.
///   Its Lipschitz bound is `q_S = beta max_{v in S} sum_{u in S} |P_vu|`, with
///   `P- ` entries weighted by `gamma`. `q_S >= 1` is
///   [`LodError::FixedPointNotContractive`] under [`OnNonContractive::Refuse`]:
///   without a contraction the iteration has no bound and the fixed point need
///   not be unique. Under [`OnNonContractive::AdaptGain`] the falsifier entries
///   inside `S` take the gain of [`adapted_gain`] instead, and the block is
///   listed in [`BlockSolve::adapted`].
///
/// Error propagation: a block reads upstream values that are off by at most
/// their own bounds `e_u`, so its bound is
/// `(q r + beta max_v sum_{u outside} |P_vu| e_u) / (1 - q)`, `r` its stop residual.
#[allow(clippy::too_many_arguments)]
fn solve_by_blocks(
    rows: &SignedRows,
    prior: &[f64],
    beta: f64,
    gamma: f64,
    tolerance: f64,
    max_steps: usize,
    on_non_contractive: OnNonContractive,
) -> Result<BlockSolve, LodError> {
    let n = prior.len();
    let blocks = Blocks::of(rows);
    let condensation = &blocks.condensation;
    let mut c = prior.to_vec();
    let mut error = vec![0.0_f64; n];
    let mut solve = BlockSolve {
        confidences: Vec::new(),
        scc_count: condensation.component_count(),
        trivial_scc_count: 0,
        cyclic_scc_count: 0,
        max_scc_size: 0,
        contraction: 0.0,
        node_updates: 0,
        reported: None,
        error_bound: 0.0,
        adapted: Vec::new(),
    };
    for k in 0..condensation.component_count() {
        let members = condensation.component(k);
        solve.max_scc_size = solve.max_scc_size.max(members.len());
        let (requested, inherited, inside) = blocks.weights(rows, k, beta, gamma, Some(&error));
        if !blocks.is_cyclic(rows, k) {
            let v = members[0] as usize;
            let value = evaluate_node(rows, prior, beta, gamma, None, &c, v);
            if !value.is_finite() {
                return Err(LodError::FixedPointDiverged {
                    iterations: 0,
                    k_max: 0,
                    residual: f64::NAN,
                    tolerance,
                });
            }
            c[v] = value;
            error[v] = inherited;
            solve.trivial_scc_count += 1;
            solve.node_updates += 1;
            continue;
        }
        let contracts = !requested.is_nan() && requested < 1.0;
        let steps = if contracts {
            worst_case_steps(requested, tolerance)
        } else {
            usize::MAX
        };
        // Refuse: only a missing contraction is refused here; a slow one runs
        // and fails against its step budget. AdaptGain also adapts a block that
        // may not finish inside the budget.
        let adapt = match on_non_contractive {
            OnNonContractive::Refuse => false,
            OnNonContractive::AdaptGain => steps > max_steps,
        };
        let refusal = || {
            if contracts {
                LodError::FixedPointTooSlow {
                    block_size: members.len(),
                    contraction: requested,
                    k_max: steps,
                    max_steps,
                }
            } else {
                LodError::FixedPointNotContractive {
                    block_size: members.len(),
                    contraction: requested,
                    beta,
                    gamma,
                }
            }
        };
        let (q, inner) = if adapt {
            let (gain, q) = adapted_gain(beta, gamma, &inside).ok_or_else(refusal)?;
            solve.adapted.push(AdaptedBlock {
                nodes: members.to_vec(),
                requested_gamma: gamma,
                applied_gamma: gain,
                requested_contraction: requested,
                contraction: q,
            });
            (q, Some((blocks.block_of.as_slice(), gain)))
        } else if contracts {
            (requested, None)
        } else {
            return Err(refusal());
        };
        let run = iterate_block(
            rows, prior, members, q, beta, gamma, inner, tolerance, max_steps, &mut c,
        )?;
        let bound = (q * run.residual + inherited) / (1.0 - q);
        for &v in members {
            error[v as usize] = bound;
        }
        solve.cyclic_scc_count += 1;
        solve.contraction = solve.contraction.max(q);
        solve.node_updates += members.len() * (run.iterations + 1);
        if solve.reported.is_none_or(|r| run.k_max > r.k_max) {
            solve.reported = Some(run);
        }
    }
    solve.error_bound = error.iter().copied().fold(0.0, f64::max);
    solve.confidences = c;
    Ok(solve)
}

/// The edge enters the confidence iteration and its target reaches its source
/// along edges that do: adding it closes a cycle. An edge that closes none
/// cannot raise any block's Lipschitz bound, it only dilutes its target's row.
fn closes_confidence_cycle(
    nodes: &NodeChunks,
    snapshot: &CsrGraph,
    pending: &[BufferedEdge],
    edge: &BufferedEdge,
) -> bool {
    let with_falsifiers = ADMISSION_GAMMA > 0.0;
    let enters = |target: u32, edge_type: EdgeType, weight: f32| {
        (carries_confidence(edge_type) || (with_falsifiers && edge_type == EdgeType::Falsifies))
            && weight > 0.0
            && !is_pinned(&nodes[target as usize])
    };
    if !enters(edge.target, edge.edge_type, edge.weight) {
        return false;
    }
    let mut buffered: HashMap<u32, Vec<(u32, EdgeType, f32)>> = HashMap::new();
    for e in pending {
        buffered
            .entry(e.source)
            .or_default()
            .push((e.target, e.edge_type, e.weight));
    }
    let mut seen = HashSet::from([edge.target]);
    let mut stack = vec![edge.target];
    while let Some(u) = stack.pop() {
        if u == edge.source {
            return true;
        }
        let later = buffered.get(&u).into_iter().flatten().copied();
        for (v, edge_type, weight) in snapshot.neighbors(u).chain(later) {
            if enters(v, edge_type, weight) && seen.insert(v) {
                stack.push(v);
            }
        }
    }
    false
}

/// The refusal admission gives the graph made of `nodes` and these edges: its
/// first cycle that is not a contraction at [`ADMISSION_BETA`],
/// [`ADMISSION_GAMMA`].
fn admission_refusal(
    nodes: &NodeChunks,
    snapshot: &CsrGraph,
    pending: &[BufferedEdge],
) -> Option<LodError> {
    let rows = signed_rows(nodes, snapshot, pending, ADMISSION_GAMMA > 0.0);
    inadmissible_block(&rows)
}

/// Edges along which confidence is inherited: the target depends on the source.
/// A coarse-grained summary depends on its members, so refuting members lowers
/// the summary at the next evolution.
fn carries_confidence(edge_type: EdgeType) -> bool {
    matches!(
        edge_type,
        EdgeType::DependsOn | EdgeType::CausalTransition | EdgeType::CoarseGrain
    )
}

/// Edges that make a validated dependency when both ends are active truths.
fn states_dependency(edge_type: EdgeType) -> bool {
    matches!(edge_type, EdgeType::DependsOn | EdgeType::Validates)
}

/// Every edge of the graph: the committed snapshot, then the pending buffer.
fn all_edges<'a>(
    snapshot: &'a CsrGraph,
    pending: &'a [BufferedEdge],
) -> impl Iterator<Item = (u32, u32, EdgeType, f32)> + 'a {
    let committed = (0..snapshot.num_nodes() as u32).flat_map(move |u| {
        snapshot
            .neighbors(u)
            .map(move |(v, edge_type, weight)| (u, v, edge_type, weight))
    });
    let buffered = pending
        .iter()
        .map(|e| (e.source, e.target, e.edge_type, e.weight));
    committed.chain(buffered)
}

/// Held at its prior by evidence: an axiom (1) or a refuted node (0).
fn is_pinned(node: &LodNode) -> bool {
    node.status == EpistemicStatus::Axiomatic || node.refuted
}

/// Row-normalise, into each unpinned node, the positive-weight support edges and,
/// when `with_falsifiers`, the positive-weight `Falsifies` edges, each kind over
/// its own total.
fn signed_rows(
    nodes: &NodeChunks,
    snapshot: &CsrGraph,
    pending: &[BufferedEdge],
    with_falsifiers: bool,
) -> SignedRows {
    let n = nodes.len();
    let enters = |&(_, target, edge_type, weight): &(u32, u32, EdgeType, f32)| {
        (carries_confidence(edge_type) || (with_falsifiers && edge_type == EdgeType::Falsifies))
            && weight > 0.0
            && !is_pinned(&nodes[target as usize])
    };
    let mut offsets = vec![0usize; n + 1];
    // Per node: total support weight, total falsifier weight.
    let mut totals = vec![(0.0_f64, 0.0_f64); n];
    for (_, target, edge_type, weight) in all_edges(snapshot, pending).filter(enters) {
        offsets[target as usize + 1] += 1;
        let total = &mut totals[target as usize];
        if edge_type == EdgeType::Falsifies {
            total.1 += f64::from(weight);
        } else {
            total.0 += f64::from(weight);
        }
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets[..n].to_vec();
    let mut sources = vec![0u32; offsets[n]];
    let mut probabilities = vec![0.0_f64; offsets[n]];
    let mut falsifies = vec![false; offsets[n]];
    for (source, target, edge_type, weight) in all_edges(snapshot, pending).filter(enters) {
        let (support, against) = totals[target as usize];
        let is_falsifier = edge_type == EdgeType::Falsifies;
        let at = &mut cursor[target as usize];
        sources[*at] = source;
        probabilities[*at] = f64::from(weight) / if is_falsifier { against } else { support };
        falsifies[*at] = is_falsifier;
        *at += 1;
    }
    SignedRows {
        offsets,
        sources,
        probabilities,
        falsifies,
        supported: totals.iter().map(|&(support, _)| support > 0.0).collect(),
    }
}

/// The validated dependencies the current statuses and edges imply.
fn validated_dependencies(
    nodes: &NodeChunks,
    snapshot: &CsrGraph,
    pending: &[BufferedEdge],
) -> HashSet<(u64, u64)> {
    all_edges(snapshot, pending)
        .filter(|(_, _, edge_type, _)| states_dependency(*edge_type))
        .filter_map(|(u, v, _, _)| {
            let (u, v) = (&nodes[u as usize], &nodes[v as usize]);
            (u.status.is_active_truth() && v.status.is_active_truth())
                .then_some((u.entity_id, v.entity_id))
        })
        .collect()
}

/// Ranked output of [`LodGraph::query_ppr`].
#[derive(Clone, Debug, PartialEq)]
pub struct PprRanking {
    /// `(node id, score)` for live nodes, highest score first.
    pub ranked: Vec<(u32, f32)>,
    pub iterations: usize,
    pub residual: f32,
    pub converged: bool,
}

/// PPR convergence tolerance of [`LodGraph::hybrid_rag_search`].
pub const HYBRID_PPR_TOLERANCE: f32 = 1e-6;

/// Hard node cap shared by every insertion path, including reflection.
pub const MAX_GRAPH_NODES: usize = 1 << 20;

fn check_node_capacity(current: usize, additional: usize) -> Result<(), LodError> {
    if additional > MAX_GRAPH_NODES.saturating_sub(current) || current > MAX_GRAPH_NODES {
        return Err(LodError::GraphCapacityExceeded {
            current,
            additional,
            max: MAX_GRAPH_NODES,
        });
    }
    Ok(())
}

/// Normalize each track independently before comparing or weighting anchors.
/// An all-zero track consists entirely of exact matches and stays zero.
fn normalize_anchor_distances(anchors: &mut [(u32, f32, AnchorMatch)]) -> Result<(), LodError> {
    let mut max_distance = 0.0_f32;
    for &(_, distance, _) in anchors.iter() {
        if !distance.is_finite() || distance < 0.0 {
            return Err(LodError::InvalidQuery(
                "invalid retrieval anchor distance".into(),
            ));
        }
        max_distance = max_distance.max(distance);
    }
    if max_distance > 0.0 {
        for (_, distance, _) in anchors {
            *distance /= max_distance;
        }
    }
    Ok(())
}

/// Most nodes that can hold one alias. Each new holder is linked to every
/// earlier one, so this bounds the edges one insert adds.
pub const MAX_ALIAS_HOLDERS: usize = 64;

/// Weight of the `Semantic` edges between two nodes that share an alias.
pub const ALIAS_LINK_WEIGHT: f32 = 1.0;

/// Which anchor of a node was closest to the query.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AnchorMatch {
    /// The node's own coordinate.
    Primary,
    /// The alias at this index of `LodNode::aliases`.
    Alias(usize),
    /// The dense projection of the node's embedding.
    Embedding,
}

/// One query coordinate with its fingerprint, and the kind of anchor it may be
/// compared with.
#[derive(Clone, Copy)]
struct Probe {
    coord: MixedCurvatureCoord,
    hdc: [u64; 4],
    space: Placement,
}

/// Anchors of [`LodGraph::recall_in`] with the Stage 1 counts.
struct Recall {
    /// `(node id, distance, closest anchor)`, closest first.
    anchors: Vec<(u32, f32, AnchorMatch)>,
    stage1_candidates: usize,
    searchable_nodes: usize,
}

/// The anchors of `node` a probe of `space` is compared with.
fn anchors_in(
    node: &LodNode,
    space: Placement,
) -> impl Iterator<Item = (AnchorMatch, &MixedCurvatureCoord, &[u64; 4])> {
    let chart = space == Placement::Chart;
    let primary = (chart && node.placement == Placement::Chart).then_some((
        AnchorMatch::Primary,
        &node.coord,
        &node.hdc_fingerprint,
    ));
    let aliases = node
        .alias_anchors
        .iter()
        .enumerate()
        .filter(move |_| chart)
        .map(|(i, a)| (AnchorMatch::Alias(i), &a.coord, &a.hdc_fingerprint));
    let dense = node
        .embedding_anchor
        .as_ref()
        .filter(|_| !chart)
        .map(|a| (AnchorMatch::Embedding, &a.coord, &a.hdc_fingerprint));
    primary.into_iter().chain(aliases).chain(dense)
}

/// One hit of [`LodGraph::hybrid_rag_search`], with its evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct RagHit {
    pub node_id: u32,
    pub entity_id: u64,
    pub label: String,
    pub band: LodBand,
    pub status: EpistemicStatus,
    /// The node's posterior confidence.
    pub confidence: f32,
    /// PPR relevance from the anchors; the ranking key.
    pub ppr_score: f32,
    /// Dimensionless distance: raw geodesic distance divided by the maximum
    /// of its track's recalled anchors (all-zero tracks remain zero). Minimum
    /// across tracks for duplicates; `None` when diffusion alone reached it.
    pub anchor_distance: Option<f32>,
    /// Which of the node's anchors that was; `None` with `anchor_distance`.
    pub anchor_match: Option<AnchorMatch>,
    /// The node's aliases.
    pub aliases: Vec<String>,
    pub payload: Option<String>,
    pub source_uri: Option<String>,
    pub timestamp_ns: u64,
    pub payload_digest: [u8; 32],
}

/// The PPR run of one [`LodGraph::hybrid_rag_search`].
#[derive(Clone, Debug, PartialEq)]
pub struct RagDiffusion {
    pub alpha: f32,
    pub max_iters: usize,
    pub tolerance: f32,
    pub iterations: usize,
    pub residual: f32,
    pub converged: bool,
}

/// Output of [`LodGraph::hybrid_rag_search`].
#[derive(Clone, Debug, PartialEq)]
pub struct HybridRagResult {
    /// Every anchor plus at most `top_k` diffusion-reached nodes, highest PPR
    /// score first.
    pub hits: Vec<RagHit>,
    /// Stage 2 anchors `(node id, normalized distance)`, closest first.
    /// Distances are dimensionless, normalized independently within each track.
    pub anchors: Vec<(u32, f32)>,
    /// Live nodes kept by the Stage 1 Hamming prefilter.
    pub stage1_candidates: usize,
    /// Live nodes with an anchor the query can be compared with. A vector
    /// query sees only nodes that carry an embedding; a text or coordinate
    /// query does not see a node placed by its embedding that has no alias.
    /// The rest can only be reached by diffusion.
    pub searchable_nodes: usize,
    /// `None` when there was no anchor, so no diffusion ran.
    pub diffusion: Option<RagDiffusion>,
}

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// Dynamic LodGraph integrating multi-scale nodes, ticketed edge buffer,
/// lock-free CSR snapshots, and epistemic governance.
pub struct LodGraph {
    /// Identity checked by `rollback_checkpoint`, so a checkpoint cannot be
    /// restored into another graph.
    graph_id: u64,
    /// `H^4 x R^8 x S^3` under this graph's parameters. Node and query
    /// coordinates are validated as points of it.
    manifold: ProductManifold,
    /// The manifold's parameters in the f32 precision of the node chart:
    /// `[alpha_h, alpha_e, alpha_s]`, curvature, radius.
    metric: ([f32; 3], f32, f32),
    /// Text projector onto this graph's ball. Node texts and query texts must
    /// both go through it, so they share one curvature.
    projector: TextEmbeddingProjector,
    state: RwLock<GraphState>,
    csr_snapshot: ArcSwap<CsrGraph>,
    /// Shared with every candidate of this graph, so a ticket or checkpoint
    /// number is issued once, whichever copy issues it.
    ticket_counter: Arc<AtomicU64>,
    checkpoint_seq: Arc<AtomicU64>,
    /// Serializes flush builds against each other and against rollbacks.
    flush_lock: Mutex<()>,
    /// Serializes writers: every transaction and every direct write.
    txn_lock: Mutex<()>,
    persistence: Option<persistence::Persistence>,
    /// A private transaction candidate ([`LodGraph::transact`]). Its writes
    /// change it in place; nothing else can see it until it is published.
    candidate: bool,
    /// Checkpoint numbers this candidate issued. If it is dropped unpublished,
    /// they are discarded in the graph it came from.
    issued: Mutex<Vec<u64>>,
}

impl Default for LodGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl LodGraph {
    /// An empty graph with [`GeometryParams::UNIT`].
    pub fn new() -> Self {
        Self::with_geometry(GeometryParams::UNIT).expect("the unit geometry is valid")
    }

    /// An empty graph whose distances use `params`: curvature and radius fix the
    /// domain of every node coordinate, the weights fix the recall metric.
    /// The parameters are rounded to f32, the precision of the node chart, and
    /// must stay finite and positive after rounding.
    pub fn with_geometry(params: GeometryParams) -> Result<Self, LodError> {
        params.validate()?;
        let alphas = [
            params.alpha_h as f32,
            params.alpha_e as f32,
            params.alpha_s as f32,
        ];
        let (c, r) = (params.curvature as f32, params.radius as f32);
        let rounded = GeometryParams {
            curvature: f64::from(c),
            radius: f64::from(r),
            alpha_h: f64::from(alphas[0]),
            alpha_e: f64::from(alphas[1]),
            alpha_s: f64::from(alphas[2]),
        };
        let layout = MixedCurvatureCoord::LAYOUT;
        // The frame is local to this graph: only the geometry digest is bound.
        let epochs = Epochs {
            version: Version(0),
            model: [0; 32],
            geometry: rounded.digest(layout),
            atlas: [0; 32],
            graph: [0; 32],
            policy: [0; 32],
        };
        let manifold = ProductManifold::new(layout, rounded, epochs)?;
        Ok(Self {
            graph_id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            manifold,
            metric: (alphas, c, r),
            projector: TextEmbeddingProjector::new(c)?,
            state: RwLock::new(GraphState::default()),
            csr_snapshot: ArcSwap::from_pointee(CsrGraph::empty()),
            ticket_counter: Arc::new(AtomicU64::new(1)),
            checkpoint_seq: Arc::new(AtomicU64::new(0)),
            flush_lock: Mutex::new(()),
            txn_lock: Mutex::new(()),
            persistence: None,
            candidate: false,
            issued: Mutex::new(Vec::new()),
        })
    }

    /// A private candidate holding this graph's current state. It shares the
    /// graph id, so a checkpoint of this graph can be restored into it, and
    /// shares its ticket and checkpoint counters.
    fn fork(&self) -> LodGraph {
        let st = self.state.read();
        LodGraph {
            graph_id: self.graph_id,
            manifold: self.manifold.clone(),
            metric: self.metric,
            projector: self.projector.clone(),
            state: RwLock::new(st.clone()),
            csr_snapshot: ArcSwap::new(self.csr_snapshot.load_full()),
            ticket_counter: Arc::clone(&self.ticket_counter),
            checkpoint_seq: Arc::clone(&self.checkpoint_seq),
            flush_lock: Mutex::new(()),
            txn_lock: Mutex::new(()),
            persistence: None,
            candidate: true,
            issued: Mutex::new(Vec::new()),
        }
    }

    /// Commit `candidate` durably when this graph is persistent, then make it
    /// the live state in one step. On a commit error the live state is untouched.
    /// The caller holds `txn_lock`.
    fn publish(&self, candidate: LodGraph) -> Result<(), LodError> {
        let csr = candidate.csr_snapshot.load_full();
        let mut staged = candidate.state.into_inner();
        if let Some(p) = &self.persistence {
            let next_ticket = self.ticket_counter.load(Ordering::Relaxed);
            p.commit(&staged, &csr, next_ticket, self.geometry())?;
        }
        let _flush = self.flush_lock.lock();
        let mut live = self.state.write();
        // A flush built against the replaced state must not commit.
        staged.generation = staged.generation.max(live.generation) + 1;
        // Dry runs dropped while this candidate ran discarded into live.
        for &(after, upto) in &live.discarded {
            discard_range(&mut staged.discarded, after, upto);
        }
        *live = staged;
        self.csr_snapshot.store(csr);
        Ok(())
    }

    /// A candidate dropped unpublished: its checkpoints describe states that
    /// never existed here, so no rollback may restore them.
    fn discard_unpublished(&self, candidate: LodGraph) {
        let seqs = candidate.issued.into_inner();
        if seqs.is_empty() {
            return;
        }
        {
            let _flush = self.flush_lock.lock();
            let mut st = self.state.write();
            for &seq in &seqs {
                discard_range(&mut st.discarded, seq - 1, seq);
            }
        }
        if self.candidate {
            self.issued.lock().extend(seqs);
        }
    }

    /// Run one direct write. A candidate changes in place. A persistent live
    /// graph runs it as a [`Self::transact`], so it is durable before anyone
    /// sees it. An in-memory live graph runs it in place under `txn_lock`, so
    /// it cannot interleave with a transaction that would publish over it.
    fn write<T>(&self, op: impl FnOnce(&LodGraph) -> Result<T, LodError>) -> Result<T, LodError> {
        if self.candidate {
            return op(self);
        }
        if self.persistence.is_some() {
            return self.transact(op);
        }
        let _txn = self.txn_lock.lock();
        op(self)
    }

    /// The geometry every distance of this graph uses.
    pub fn geometry(&self) -> GeometryParams {
        self.manifold.params()
    }

    /// Project `text` to a coordinate of this graph's chart and a 256-bit HDC
    /// fingerprint with the graph's [`TextEmbeddingProjector`]. Blank text is
    /// [`LodError::EmptyInput`].
    pub fn project_text(&self, text: &str) -> Result<(MixedCurvatureCoord, [u64; 4]), LodError> {
        self.projector.project_text(text)
    }

    /// Project a dense embedding to a coordinate of this graph's chart and a
    /// 256-bit HDC fingerprint ([`TextEmbeddingProjector::project_dense`]).
    pub fn project_dense(
        &self,
        embedding: &[f32],
    ) -> Result<(MixedCurvatureCoord, [u64; 4]), LodError> {
        self.projector.project_dense(embedding)
    }

    /// Distance between two coordinates under this graph's geometry.
    fn distance(&self, a: &MixedCurvatureCoord, b: &MixedCurvatureCoord) -> Result<f32, LodError> {
        let (alphas, c, r) = self.metric;
        a.product_distance_with_params(b, alphas, c, r)
    }

    /// Insert a node and return its id. Refuses a coordinate outside this
    /// graph's geometry, a prior outside [0, 1], a posterior that differs from
    /// the prior, a refutation mark on a node that is not `Falsified`, an unknown
    /// parent, a payload that breaks [`LodNode::validate_payload`], and an entity
    /// id that already has a node: the gate addresses nodes by entity.
    ///
    /// An `Axiomatic` node gets confidence 1. A `Falsified` node is refuted by
    /// evidence: confidence 0, entity revoked.
    ///
    /// Each alias is projected with the graph's text projector and becomes an
    /// anchor of the node. For every earlier node that holds the same alias
    /// (same tokens, any case and spacing) two `Semantic` edges of weight
    /// [`ALIAS_LINK_WEIGHT`] are appended, one each way; like every edge they
    /// reach diffusion at the next flush. An embedding is projected with
    /// [`Self::project_dense`]; a node with [`Placement::Embedding`] takes that
    /// projection as its coordinate and fingerprint. Also refused: an alias
    /// that breaks [`LodNode::validate_aliases`], has no alphanumeric
    /// character, repeats another alias of the node or already has
    /// [`MAX_ALIAS_HOLDERS`] holders; an embedding [`Self::project_dense`]
    /// refuses or whose dimension differs from the graph's earlier embeddings;
    /// and [`Placement::Embedding`] without an embedding.
    pub fn add_node(&self, node: LodNode) -> Result<u32, LodError> {
        self.write(|g| g.add_node_now(node))
    }

    fn add_node_now(&self, node: LodNode) -> Result<u32, LodError> {
        let mut st = self.state.write();
        self.insert_node(&mut st, node)
    }

    /// [`Self::add_node`] under a held write lock.
    fn insert_node(&self, st: &mut GraphState, mut node: LodNode) -> Result<u32, LodError> {
        check_node_capacity(st.nodes.len(), 1)?;
        node.validate_aliases()?;
        let mut alias_keys: Vec<String> = Vec::with_capacity(node.aliases.len());
        node.alias_anchors.clear();
        for alias in &node.aliases {
            let (coord, hdc_fingerprint) = self.projector.project_text(alias)?;
            let key = normalized(alias);
            if alias_keys.contains(&key) {
                return Err(LodError::InvalidNode(format!(
                    "alias `{alias}` repeats another alias of the node"
                )));
            }
            let holders = st.alias_index.get(&key).map_or(0, Vec::len);
            if holders >= MAX_ALIAS_HOLDERS {
                return Err(LodError::InvalidNode(format!(
                    "alias `{alias}` already has {holders} holders, the most one alias can link"
                )));
            }
            alias_keys.push(key);
            node.alias_anchors.push(ChartAnchor {
                coord,
                hdc_fingerprint,
            });
        }
        node.embedding_anchor = match &node.embedding {
            Some(embedding) => {
                let (coord, hdc_fingerprint) = self.projector.project_dense(embedding)?;
                if st.embedding_dim.is_some_and(|dim| dim != embedding.len()) {
                    return Err(LodError::InvalidNode(format!(
                        "embedding has {} dimensions but this graph's embeddings have {}",
                        embedding.len(),
                        st.embedding_dim.unwrap_or_default()
                    )));
                }
                Some(ChartAnchor {
                    coord,
                    hdc_fingerprint,
                })
            }
            None => None,
        };
        if node.placement == Placement::Embedding {
            let anchor = node.embedding_anchor.ok_or_else(|| {
                LodError::InvalidNode("a node placed by its embedding needs an embedding".into())
            })?;
            node.coord = anchor.coord;
            node.hdc_fingerprint = anchor.hdc_fingerprint;
        }
        node.coord.to_point(&self.manifold)?;
        node.validate_payload()?;
        if !(node.prior.is_finite() && (0.0..=1.0).contains(&node.prior)) {
            return Err(LodError::InvalidNode(format!(
                "prior {} must lie in [0, 1]",
                node.prior
            )));
        }
        if node.confidence != node.prior {
            return Err(LodError::InvalidNode(format!(
                "confidence {} differs from prior {}; set both with `with_prior`",
                node.confidence, node.prior
            )));
        }
        if node.refuted && !node.status.is_falsified() {
            return Err(LodError::InvalidNode(
                "only a Falsified node can be inserted as refuted".into(),
            ));
        }
        match node.status {
            EpistemicStatus::Axiomatic => node.confidence = 1.0,
            EpistemicStatus::Falsified => {
                node.refuted = true;
                node.confidence = 0.0;
            }
            EpistemicStatus::Hypothesized | EpistemicStatus::Validated => {}
        }
        let id = u32::try_from(st.nodes.len())
            .map_err(|_| LodError::InvalidNode("graph holds u32::MAX nodes".into()))?;
        if let Some(parent) = node.parent_id {
            if parent >= id {
                return Err(LodError::InvalidNode(format!(
                    "parent {parent} is not an existing node"
                )));
            }
        }
        if st.entity_index.contains_key(&node.entity_id) {
            return Err(LodError::DuplicateEntity(node.entity_id));
        }
        node.id = id;
        if node.status.is_falsified() {
            st.revocations.insert(node.entity_id);
        }
        Arc::make_mut(&mut st.entity_index).insert(node.entity_id, id);
        if st.embedding_dim.is_none() {
            st.embedding_dim = node.embedding.as_ref().map(Vec::len);
        }
        st.nodes.push(node);
        // Nothing below can fail: both ends exist and the weight is valid.
        let mut linked = HashSet::new();
        for key in alias_keys {
            let holders = Arc::make_mut(&mut st.alias_index).entry(key).or_default();
            let earlier = holders.clone();
            holders.push(id);
            for holder in earlier {
                if linked.insert(holder) {
                    self.push_edge(st, id, holder, EdgeType::Semantic, ALIAS_LINK_WEIGHT)?;
                    self.push_edge(st, holder, id, EdgeType::Semantic, ALIAS_LINK_WEIGHT)?;
                }
            }
        }
        Ok(id)
    }

    /// Retrieve a cloned copy of a node by ID.
    pub fn get_node(&self, id: u32) -> Option<LodNode> {
        self.state.read().nodes.get(id as usize).cloned()
    }

    /// Node id holding `entity_id`, if any.
    pub fn node_for_entity(&self, entity_id: u64) -> Option<u32> {
        self.state.read().entity_index.get(&entity_id).copied()
    }

    /// Number of nodes currently in graph.
    pub fn node_count(&self) -> usize {
        self.state.read().nodes.len()
    }

    /// Edges appended but not yet merged into the CSR snapshot.
    pub fn pending_edge_count(&self) -> usize {
        self.state.read().edge_buffer.len()
    }

    /// The currently published CSR snapshot.
    pub fn csr_snapshot(&self) -> Arc<CsrGraph> {
        self.csr_snapshot.load_full()
    }

    /// Append a directional edge to the pending buffer and return its ticket.
    /// Both endpoints must be existing nodes; the weight must be finite and
    /// nonnegative. A `CoarseGrain` edge must run from a finer band to a strictly
    /// coarser one.
    ///
    /// Admission: an edge that closes a cycle of confidence-carrying or
    /// `Falsifies` edges is refused with [`LodError::FixedPointNotContractive`]
    /// when some cycle of the resulting graph is not a contraction at
    /// [`ADMISSION_BETA`], [`ADMISSION_GAMMA`], and with
    /// [`LodError::FixedPointTooSlow`] when it contracts so slowly that the
    /// reflection's evolution may exhaust [`MAX_FIXED_POINT_STEPS`]. The graph
    /// then never holds a cycle an evolution at those parameters, the
    /// reflection's among them, must refuse or can fail to finish. An edge that closes no cycle only dilutes its target's row
    /// and is admitted after a reachability walk from the target; a cycle-closing
    /// edge costs one `O(V + E)` check.
    pub fn add_edge(
        &self,
        source: u32,
        target: u32,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<u64, LodError> {
        self.write(|g| g.add_edge_now(source, target, edge_type, weight))
    }

    fn add_edge_now(
        &self,
        source: u32,
        target: u32,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<u64, LodError> {
        let mut st = self.state.write();
        self.push_edge(&mut st, source, target, edge_type, weight)
    }

    /// Append several edges as one batch and return their tickets, in order.
    /// Each edge is checked as [`Self::add_edge`] checks it, and admission runs
    /// once over the graph with the whole batch (one `O(V + E)` pass instead of
    /// one per edge, so a large deposit stays linear). Any refusal leaves the
    /// buffer as it was: no edge of the batch is added.
    pub fn add_edges(&self, edges: &[(u32, u32, EdgeType, f32)]) -> Result<Vec<u64>, LodError> {
        self.write(|g| g.add_edges_now(edges))
    }

    fn add_edges_now(&self, edges: &[(u32, u32, EdgeType, f32)]) -> Result<Vec<u64>, LodError> {
        let mut st = self.state.write();
        let mut batch = Vec::with_capacity(edges.len());
        for &(source, target, edge_type, weight) in edges {
            batch.push(self.checked_edge(&st, source, target, edge_type, weight)?);
        }
        let snapshot = self.csr_snapshot.load_full();
        let kept = st.edge_buffer.len();
        st.edge_buffer.extend_from_slice(&batch);
        let refusal = admission_refusal(&st.nodes, &snapshot, &st.edge_buffer);
        st.edge_buffer.truncate(kept);
        if let Some(refusal) = refusal {
            return Err(refusal);
        }
        Ok(batch
            .into_iter()
            .map(|edge| self.commit_edge(&mut st, edge))
            .collect())
    }

    /// [`Self::add_edge`] under a held write lock.
    fn push_edge(
        &self,
        st: &mut GraphState,
        source: u32,
        target: u32,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<u64, LodError> {
        let edge = self.checked_edge(st, source, target, edge_type, weight)?;
        let snapshot = self.csr_snapshot.load_full();
        if closes_confidence_cycle(&st.nodes, &snapshot, &st.edge_buffer, &edge) {
            st.edge_buffer.push(edge);
            let refusal = admission_refusal(&st.nodes, &snapshot, &st.edge_buffer);
            st.edge_buffer.pop();
            if let Some(refusal) = refusal {
                return Err(refusal);
            }
        }
        Ok(self.commit_edge(st, edge))
    }

    /// The edge, once its endpoints, weight and band order are valid. Its
    /// ticket is set by [`Self::commit_edge`].
    fn checked_edge(
        &self,
        st: &GraphState,
        source: u32,
        target: u32,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<BufferedEdge, LodError> {
        check_edge(st.nodes.len(), source, target, weight)?;
        if edge_type == EdgeType::CoarseGrain {
            let (fine, coarse) = (
                st.nodes[source as usize].band,
                st.nodes[target as usize].band,
            );
            if fine >= coarse {
                return Err(LodError::InvalidEdge(format!(
                    "coarse-grain edge {source} -> {target} must go from a finer band to a \
                     coarser one, got {fine:?} -> {coarse:?}"
                )));
            }
        }
        Ok(BufferedEdge {
            source,
            target,
            edge_type,
            weight,
            ticket: 0,
        })
    }

    /// Append an admitted edge to the buffer and return its ticket.
    fn commit_edge(&self, st: &mut GraphState, mut edge: BufferedEdge) -> u64 {
        // Taken under the lock so buffer order is ticket order.
        edge.ticket = self.ticket_counter.fetch_add(1, Ordering::Relaxed);
        st.edge_buffer.push(edge);
        if states_dependency(edge.edge_type) {
            let (u, v) = (
                &st.nodes[edge.source as usize],
                &st.nodes[edge.target as usize],
            );
            if u.status.is_active_truth() && v.status.is_active_truth() {
                let dep = (u.entity_id, v.entity_id);
                st.validated_deps.insert(dep);
            }
        }
        edge.ticket
    }

    /// Merge the pending edges into a new CSR snapshot and drain them from the
    /// buffer. The new snapshot is the current snapshot plus the pending delta,
    /// grown to the current node count, and is validated before it is published.
    ///
    /// On any error the old snapshot and the whole buffer are kept. Like every
    /// direct write it is serialized with the other writers (a transaction on a
    /// persistent graph), so the build blocks writers, never readers.
    pub fn flush_edges_to_csr(&self) -> Result<FlushReport, LodError> {
        self.write(|g| g.flush_edges_to_csr_now())
    }

    fn flush_edges_to_csr_now(&self) -> Result<FlushReport, LodError> {
        let _flush = self.flush_lock.lock();
        let (base, pending, num_nodes, generation) = {
            let st = self.state.read();
            (
                self.csr_snapshot.load_full(),
                st.edge_buffer.clone(),
                st.nodes.len(),
                st.generation,
            )
        };
        let merged = base.merged(num_nodes, &pending)?;

        let mut st = self.state.write();
        let prefix_matches = st.edge_buffer.len() >= pending.len()
            && st
                .edge_buffer
                .iter()
                .zip(&pending)
                .all(|(a, b)| a.ticket == b.ticket);
        if st.generation != generation || !prefix_matches {
            return Err(LodError::FlushConflict);
        }
        let report = FlushReport {
            merged_edges: pending.len(),
            csr_nodes: merged.num_nodes(),
            csr_edges: merged.num_edges(),
            pending_edges: st.edge_buffer.len() - pending.len(),
        };
        self.csr_snapshot.store(Arc::new(merged));
        st.edge_buffer.drain(..pending.len());
        st.generation += 1;
        Ok(report)
    }

    /// Move one node one band up or down with [`LodBand::zoom_out`] /
    /// [`LodBand::zoom_in`] and return the new band. Refused past `Lod3Systemic`
    /// or below `Lod0Atomic`, and when the move would break the coarse-grain
    /// order: every member of the node (source of a `CoarseGrain` edge into it)
    /// must stay strictly finer and every summary it belongs to strictly
    /// coarser. The coordinate is not moved.
    pub fn zoom_node(&self, node_id: u32, direction: ZoomDirection) -> Result<LodBand, LodError> {
        self.write(|g| g.zoom_node_now(node_id, direction))
    }

    fn zoom_node_now(&self, node_id: u32, direction: ZoomDirection) -> Result<LodBand, LodError> {
        let mut st = self.state.write();
        let band = st
            .nodes
            .get(node_id as usize)
            .ok_or(LodError::NodeNotFound(node_id))?
            .band;
        let to = match direction {
            ZoomDirection::In => band.zoom_in()?,
            ZoomDirection::Out => band.zoom_out()?,
        };
        self.set_band(&mut st, node_id, to)?;
        Ok(to)
    }

    /// Move a node's band, one [`Self::zoom_node`] step at a time, to the band its
    /// coordinate implies under this graph's geometry
    /// ([`LodNode::derive_band_from_coord`]). Returns `(from, to)`; `from == to`
    /// changes nothing. The whole move is refused, and nothing changes, when the
    /// target band breaks the coarse-grain order.
    pub fn migrate_band_to_coord(&self, node_id: u32) -> Result<(LodBand, LodBand), LodError> {
        self.write(|g| g.migrate_band_to_coord_now(node_id))
    }

    fn migrate_band_to_coord_now(&self, node_id: u32) -> Result<(LodBand, LodBand), LodError> {
        let mut st = self.state.write();
        let node = st
            .nodes
            .get(node_id as usize)
            .ok_or(LodError::NodeNotFound(node_id))?;
        let from = node.band;
        let target = node.derive_band_from_coord(&self.geometry())?;
        let mut to = from;
        while to < target {
            to = to.zoom_out()?;
        }
        while to > target {
            to = to.zoom_in()?;
        }
        if to != from {
            self.set_band(&mut st, node_id, to)?;
        }
        Ok((from, to))
    }

    /// Set a node's band after checking the coarse-grain order against every
    /// `CoarseGrain` edge touching it, committed and pending. O(edges).
    fn set_band(&self, st: &mut GraphState, node_id: u32, to: LodBand) -> Result<(), LodError> {
        let snapshot = self.csr_snapshot.load_full();
        for (source, target, edge_type, _) in all_edges(&snapshot, &st.edge_buffer) {
            if edge_type != EdgeType::CoarseGrain {
                continue;
            }
            let broken = if target == node_id {
                let member = st.nodes[source as usize].band;
                (member >= to).then(|| format!("member {source} is at {member:?}"))
            } else if source == node_id {
                let summary = st.nodes[target as usize].band;
                (summary <= to).then(|| format!("summary {target} is at {summary:?}"))
            } else {
                None
            };
            if let Some(why) = broken {
                return Err(LodError::InvalidStateTransition(format!(
                    "node {node_id} cannot move to {to:?}: {why}, and a coarse-grain edge \
                     must go from a finer band to a coarser one"
                )));
            }
        }
        st.nodes[node_id as usize].band = to;
        Ok(())
    }

    /// Coarse-grain a cluster: insert one summary node for `cluster_node_ids`,
    /// link every member to it with a `CoarseGrain` edge of weight 1, make it the
    /// members' parent, and flush, so the link is in the CSR snapshot on return.
    /// Returns the summary's node id.
    ///
    /// The summary's band is the one `summary_coord` implies
    /// ([`LodNode::derive_band_from_coord`]), and it must be strictly coarser than
    /// every member: a summary sits nearer the origin than what it summarizes.
    /// Its prior is the [`LodNode::new`] default 0.5; its confidence follows the
    /// members at the next [`Self::evolve_epistemic_fixed_point`], since
    /// `CoarseGrain` edges carry confidence.
    ///
    /// Refused, with nothing changed: an empty cluster, a repeated or unknown
    /// member, a falsified or revoked member, a member that already has a parent,
    /// a summary band not above every member, a coordinate outside this graph's
    /// geometry and an entity that already has a node. On a persistent graph
    /// the call is one transaction. On an in-memory graph a flush failure is
    /// returned after the summary and its pending edges are inserted, visible to
    /// readers; run the call inside [`Self::transact`] to make it atomic.
    pub fn coarse_grain_cluster(
        &self,
        cluster_node_ids: &[u32],
        summary_entity_id: u64,
        summary_coord: MixedCurvatureCoord,
        hdc: [u64; 4],
    ) -> Result<u32, LodError> {
        self.write(|g| {
            g.coarse_grain_cluster_now(cluster_node_ids, summary_entity_id, summary_coord, hdc)
        })
    }

    fn coarse_grain_cluster_now(
        &self,
        cluster_node_ids: &[u32],
        summary_entity_id: u64,
        summary_coord: MixedCurvatureCoord,
        hdc: [u64; 4],
    ) -> Result<u32, LodError> {
        if cluster_node_ids.is_empty() {
            return Err(LodError::InvalidQuery(
                "a coarse-grain cluster needs at least one member".into(),
            ));
        }
        let summary_id = {
            let mut st = self.state.write();
            let mut seen = HashSet::with_capacity(cluster_node_ids.len());
            let mut finest_ceiling = LodBand::Lod0Atomic;
            for &id in cluster_node_ids {
                if !seen.insert(id) {
                    return Err(LodError::InvalidQuery(format!(
                        "node {id} is listed twice in the cluster"
                    )));
                }
                let member = st
                    .nodes
                    .get(id as usize)
                    .ok_or(LodError::NodeNotFound(id))?;
                if member.status.is_falsified() || st.revocations.contains(&member.entity_id) {
                    return Err(LodError::InvalidNode(format!(
                        "member {id} is falsified or revoked and cannot be coarse-grained"
                    )));
                }
                if let Some(parent) = member.parent_id {
                    return Err(LodError::InvalidNode(format!(
                        "member {id} already has parent {parent}"
                    )));
                }
                finest_ceiling = finest_ceiling.max(member.band);
            }
            let label = format!("coarse_grain({} members)", cluster_node_ids.len());
            let mut summary = LodNode::new(
                0,
                LodBand::Lod0Atomic,
                summary_coord,
                label,
                summary_entity_id,
            )
            .with_hdc_fingerprint(hdc);
            summary.band = summary.derive_band_from_coord(&self.geometry())?;
            if summary.band <= finest_ceiling {
                return Err(LodError::InvalidNode(format!(
                    "summary coordinate implies band {:?}, not above the coarsest member \
                     band {finest_ceiling:?}; place the summary nearer the origin",
                    summary.band
                )));
            }
            // Every check that can fail on the members is done; `insert_node`
            // checks the coordinate and entity before it changes anything, and
            // `push_edge` cannot fail on these endpoints and this band order.
            let summary_id = self.insert_node(&mut st, summary)?;
            for &id in cluster_node_ids {
                self.push_edge(&mut st, id, summary_id, EdgeType::CoarseGrain, 1.0)?;
                st.nodes[id as usize].parent_id = Some(summary_id);
            }
            summary_id
        };
        self.flush_edges_to_csr_now()?;
        Ok(summary_id)
    }

    /// Run Personalized PageRank over the committed CSR snapshot. Nodes added
    /// since the last flush take part with no edges. Every seed must be a node;
    /// bad parameters are errors. Falsified and revoked nodes are left out of the
    /// ranking.
    pub fn query_ppr(
        &self,
        seeds: &[(u32, f32)],
        alpha: f32,
        max_iters: usize,
        tolerance: f32,
    ) -> Result<PprRanking, LodError> {
        let st = self.state.read();
        self.ppr_in(&st, seeds, alpha, max_iters, tolerance)
    }

    /// [`Self::query_ppr`] under a held read lock.
    fn ppr_in(
        &self,
        st: &GraphState,
        seeds: &[(u32, f32)],
        alpha: f32,
        max_iters: usize,
        tolerance: f32,
    ) -> Result<PprRanking, LodError> {
        let snapshot = self.csr_snapshot.load();
        let num_nodes = st.nodes.len();
        if snapshot.num_nodes() > num_nodes {
            return Err(LodError::CsrInvariant(format!(
                "snapshot covers {} nodes but the graph has {num_nodes}",
                snapshot.num_nodes()
            )));
        }
        let padded;
        let row_offsets = if snapshot.num_nodes() < num_nodes {
            let mut rows = snapshot.row_ptrs().to_vec();
            rows.resize(num_nodes + 1, snapshot.num_edges());
            padded = rows;
            &padded[..]
        } else {
            snapshot.row_ptrs()
        };
        let ppr = compute_ppr_csr(
            num_nodes,
            row_offsets,
            snapshot.col_indices(),
            snapshot.edge_weights(),
            seeds,
            alpha,
            max_iters,
            tolerance,
        )?;

        let mut ranked: Vec<(u32, f32)> = ppr
            .scores
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| {
                let node = &st.nodes[*idx];
                !node.status.is_falsified() && !st.revocations.contains(&node.entity_id)
            })
            .map(|(idx, s)| (idx as u32, s))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(PprRanking {
            ranked,
            iterations: ppr.iterations,
            residual: ppr.residual,
            converged: ppr.converged,
        })
    }

    /// Two-Stage Memory Recall:
    /// Stage 1: HDC Hamming distance over every live node (a linear POPCNT scan),
    /// keeping the `4 * top_k` closest. A node's distance is the smallest over
    /// its chart anchors: its own fingerprint and one per alias.
    /// Stage 2: product geodesic rerank of those candidates under this graph's
    /// geometry (curvature, radius and the three metric weights), again by each
    /// node's closest chart anchor, with a Corrective RAG (CRAG) margin: when
    /// the top two are closer than `crag_margin`, the 1-hop CSR neighbors of the
    /// top one join the rerank.
    ///
    /// Excludes falsified and revoked nodes, and nodes placed by their embedding
    /// that have no alias. `top_k` must be at least 1 and `crag_margin` finite
    /// and nonnegative; a coordinate outside this graph's geometry is an error.
    pub fn two_stage_recall(
        &self,
        query_coord: &MixedCurvatureCoord,
        query_hdc: &[u64; 4],
        top_k: usize,
        crag_margin: f32,
    ) -> Result<Vec<(u32, f32)>, LodError> {
        let st = self.state.read();
        let probe = Probe {
            coord: *query_coord,
            hdc: *query_hdc,
            space: Placement::Chart,
        };
        let recall = self.recall_in(&st, &probe, top_k, crag_margin)?;
        Ok(recall.anchors.iter().map(|&(id, d, _)| (id, d)).collect())
    }

    /// Stages 1 and 2 under a held read lock, for one probe. A node is measured
    /// by the closest of its anchors of the probe's kind ([`anchors_in`]); a
    /// node with no such anchor is not a candidate.
    fn recall_in(
        &self,
        st: &GraphState,
        probe: &Probe,
        top_k: usize,
        crag_margin: f32,
    ) -> Result<Recall, LodError> {
        if top_k == 0 {
            return Err(LodError::InvalidQuery("top_k must be at least 1".into()));
        }
        if !(crag_margin.is_finite() && crag_margin >= 0.0) {
            return Err(LodError::InvalidQuery(format!(
                "crag_margin must be finite and nonnegative, got {crag_margin}"
            )));
        }
        probe.coord.to_point(&self.manifold)?;
        let (nodes, revs) = (&st.nodes, &st.revocations);
        let live = |n: &LodNode| {
            !n.is_internal_evidence() && !n.status.is_falsified() && !revs.contains(&n.entity_id)
        };

        let mut candidates: Vec<(u32, u32)> = nodes
            .iter()
            .filter(|n| live(n))
            .filter_map(|n| {
                anchors_in(n, probe.space)
                    .map(|(_, _, fp)| hdc_hamming_distance_256(fp, &probe.hdc))
                    .min()
                    .map(|hamming| (n.id, hamming))
            })
            .collect();
        let searchable_nodes = candidates.len();

        let candidate_pool_size = top_k.saturating_mul(4).min(candidates.len());
        if candidate_pool_size < candidates.len() {
            // O(N) linear selection instead of O(N log N) full sort
            candidates.select_nth_unstable_by_key(candidate_pool_size, |c| c.1);
            candidates.truncate(candidate_pool_size);
        }

        // An out-of-domain coordinate aborts the recall instead of being skipped.
        let closest = |node: &LodNode| -> Result<Option<(f32, AnchorMatch)>, LodError> {
            let mut best: Option<(f32, AnchorMatch)> = None;
            for (matched, coord, _) in anchors_in(node, probe.space) {
                let distance = self.distance(coord, &probe.coord)?;
                if best.is_none_or(|(d, _)| distance < d) {
                    best = Some((distance, matched));
                }
            }
            Ok(best)
        };
        let mut reranked: Vec<(u32, f32, AnchorMatch)> = Vec::with_capacity(candidates.len());
        for &(id, _) in &candidates {
            if let Some((distance, matched)) = closest(&nodes[id as usize])? {
                reranked.push((id, distance, matched));
            }
        }
        reranked.sort_by(|a, b| a.1.total_cmp(&b.1));

        if reranked.len() >= 2 && (reranked[1].1 - reranked[0].1).abs() < crag_margin {
            let snapshot = self.csr_snapshot.load();
            let top1_id = reranked[0].0;
            for (nbr, _, _) in snapshot.neighbors(top1_id) {
                if reranked.iter().any(|r| r.0 == nbr) {
                    continue;
                }
                let nbr_node = &nodes[nbr as usize];
                if !live(nbr_node) {
                    continue;
                }
                // A neighbor the query cannot be compared with stays out of the
                // rerank; diffusion still reaches it.
                if let Some((distance, matched)) = closest(nbr_node)? {
                    reranked.push((nbr, distance, matched));
                }
            }
            reranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        }

        reranked.truncate(top_k);
        Ok(Recall {
            anchors: reranked,
            stage1_candidates: candidate_pool_size,
            searchable_nodes,
        })
    }

    /// Three-stage hybrid retrieval over one consistent state (one read lock):
    ///
    /// 1. HDC prefilter: Hamming distance to every live node, keep the `4 * top_k`
    ///    closest (see [`Self::two_stage_recall`]).
    /// 2. Product-geodesic rerank of those under this graph's geometry, with the
    ///    CRAG neighbor expansion when the top two are within `crag_margin`; the
    ///    `top_k` closest are the anchors.
    /// 3. Personalized PageRank over the committed CSR snapshot, seeded with
    ///    each anchor at weight `1 / (1 + normalized_distance)` (normalized by PPR), teleport
    ///    probability `ppr_alpha`, at most `ppr_iters` iterations, tolerance
    ///    [`HYBRID_PPR_TOLERANCE`]. PPR follows every edge type, in the edge's
    ///    direction only: an anchor reaches the targets of its `Semantic`,
    ///    `Validates` and other out-edges, not their sources.
    ///
    /// Before fusion, divide each track's anchor distances by its maximum; an
    /// all-zero track stays zero. This removes multiplicative scale differences,
    /// but does not calibrate relevance across tracks or across queries.
    /// Internal reflection evidence is excluded from recall and result hits.
    ///
    /// In stages 1 and 2 a node counts by its closest chart anchor, so a query
    /// that matches an alias makes the node an anchor ([`RagHit::anchor_match`]).
    ///
    /// Hits are every anchor plus the `top_k` best other live nodes with a
    /// positive PPR score (at most `2 * top_k`), all ordered by PPR score,
    /// highest first. A node no anchor reaches has score 0 and is left out.
    /// Each hit carries its confidence, PPR score, anchor distance (none for a
    /// node reached only by diffusion) and payload evidence.
    ///
    /// Falsified and revoked nodes never appear. A graph with no live node gives
    /// no anchors, no hits and `diffusion: None`. Refused: `top_k` 0, a bad
    /// `crag_margin`, `ppr_alpha` outside (0, 1), `ppr_iters` 0, and a query
    /// coordinate outside this graph's geometry. PPR that stops at `ppr_iters`
    /// before reaching the tolerance is reported in `diffusion.converged`.
    pub fn hybrid_rag_search(
        &self,
        query_coord: &MixedCurvatureCoord,
        query_hdc: &[u64; 4],
        top_k: usize,
        crag_margin: f32,
        ppr_alpha: f32,
        ppr_iters: usize,
    ) -> Result<HybridRagResult, LodError> {
        let probe = Probe {
            coord: *query_coord,
            hdc: *query_hdc,
            space: Placement::Chart,
        };
        let st = self.state.read();
        self.rag_in(&st, &[probe], top_k, crag_margin, ppr_alpha, ppr_iters)
    }

    /// [`Self::hybrid_rag_search`] for a query given as text, as a dense
    /// vector, or as both.
    ///
    /// The text is projected with [`Self::project_text`] and compared with the
    /// chart anchors of each node (its own coordinate and its aliases). The
    /// vector is projected with [`Self::project_dense`] and compared with the
    /// embedding anchors only. With both, each track runs stages 1 and 2 on its
    /// own and gives up to `top_k` anchors; a node both tracks found counts
    /// once, by the smaller independently normalized distance. So there are at most `2 * top_k` anchors
    /// and `3 * top_k` hits, `stage1_candidates` is the sum over the tracks,
    /// and neither track can crowd the other out.
    ///
    /// Refused: neither given, blank text ([`LodError::EmptyInput`]), a vector
    /// [`Self::project_dense`] refuses, and a vector whose dimension is not the
    /// one this graph's node embeddings have. A graph with no node embedding
    /// refuses every vector: it has nothing to compare one with.
    pub fn hybrid_rag_search_query(
        &self,
        query_text: Option<&str>,
        query_vector: Option<&[f32]>,
        top_k: usize,
        crag_margin: f32,
        ppr_alpha: f32,
        ppr_iters: usize,
    ) -> Result<HybridRagResult, LodError> {
        let mut probes = Vec::with_capacity(2);
        if let Some(text) = query_text {
            let (coord, hdc) = self.project_text(text)?;
            probes.push(Probe {
                coord,
                hdc,
                space: Placement::Chart,
            });
        }
        if let Some(vector) = query_vector {
            let (coord, hdc) = self.project_dense(vector)?;
            probes.push(Probe {
                coord,
                hdc,
                space: Placement::Embedding,
            });
        }
        if probes.is_empty() {
            return Err(LodError::InvalidQuery(
                "a query needs a text, a vector or both".into(),
            ));
        }
        let st = self.state.read();
        if let Some(vector) = query_vector {
            match st.embedding_dim {
                Some(dim) if dim == vector.len() => {}
                Some(dim) => {
                    return Err(LodError::InvalidQuery(format!(
                        "query vector has {} dimensions but this graph's embeddings have {dim}",
                        vector.len()
                    )))
                }
                None => {
                    return Err(LodError::InvalidQuery(
                        "no node of this graph carries an embedding, so a query vector has \
                         nothing to be compared with"
                            .into(),
                    ))
                }
            }
        }
        self.rag_in(&st, &probes, top_k, crag_margin, ppr_alpha, ppr_iters)
    }

    /// The three stages under a held read lock.
    fn rag_in(
        &self,
        st: &GraphState,
        probes: &[Probe],
        top_k: usize,
        crag_margin: f32,
        ppr_alpha: f32,
        ppr_iters: usize,
    ) -> Result<HybridRagResult, LodError> {
        if !(ppr_alpha > 0.0 && ppr_alpha < 1.0) {
            return Err(LodError::InvalidQuery(format!(
                "ppr_alpha must lie in (0, 1), got {ppr_alpha}"
            )));
        }
        if ppr_iters == 0 {
            return Err(LodError::InvalidQuery(
                "ppr_iters must be at least 1".into(),
            ));
        }
        let mut matched: Vec<(u32, f32, AnchorMatch)> = Vec::new();
        let (mut stage1_candidates, mut searchable_nodes) = (0, 0);
        for probe in probes {
            let mut recall = self.recall_in(st, probe, top_k, crag_margin)?;
            normalize_anchor_distances(&mut recall.anchors)?;
            stage1_candidates += recall.stage1_candidates;
            searchable_nodes = recall.searchable_nodes;
            for anchor in recall.anchors {
                match matched.iter_mut().find(|m| m.0 == anchor.0) {
                    Some(seen) if anchor.1 < seen.1 => *seen = anchor,
                    Some(_) => {}
                    None => matched.push(anchor),
                }
            }
        }
        if probes.len() > 1 {
            matched.sort_by(|a, b| a.1.total_cmp(&b.1));
            searchable_nodes = st
                .nodes
                .iter()
                .filter(|n| {
                    !n.is_internal_evidence()
                        && !n.status.is_falsified()
                        && !st.revocations.contains(&n.entity_id)
                })
                .filter(|n| {
                    probes
                        .iter()
                        .any(|p| anchors_in(n, p.space).next().is_some())
                })
                .count();
        }
        let anchors: Vec<(u32, f32)> = matched.iter().map(|&(id, d, _)| (id, d)).collect();
        if anchors.is_empty() {
            return Ok(HybridRagResult {
                hits: Vec::new(),
                anchors,
                stage1_candidates,
                searchable_nodes,
                diffusion: None,
            });
        }
        let seeds: Vec<(u32, f32)> = anchors
            .iter()
            .map(|&(id, dist)| (id, 1.0 / (1.0 + dist)))
            .collect();
        let ranking = self.ppr_in(st, &seeds, ppr_alpha, ppr_iters, HYBRID_PPR_TOLERANCE)?;
        let mut expanded = 0;
        let hits = ranking
            .ranked
            .iter()
            .filter(|&&(id, score)| {
                if st.nodes[id as usize].is_internal_evidence() {
                    return false;
                }
                if anchors.iter().any(|a| a.0 == id) {
                    return true;
                }
                let keep = score > 0.0 && expanded < top_k;
                expanded += usize::from(keep);
                keep
            })
            .map(|&(id, ppr_score)| {
                let node = &st.nodes[id as usize];
                let anchor = matched.iter().find(|a| a.0 == id);
                RagHit {
                    node_id: id,
                    entity_id: node.entity_id,
                    label: node.label.clone(),
                    band: node.band,
                    status: node.status,
                    confidence: node.confidence,
                    ppr_score,
                    anchor_distance: anchor.map(|a| a.1),
                    anchor_match: anchor.map(|a| a.2),
                    aliases: node.aliases.clone(),
                    payload: node.payload.clone(),
                    source_uri: node.source_uri.clone(),
                    timestamp_ns: node.timestamp_ns,
                    payload_digest: node.payload_digest,
                }
            })
            .collect();
        Ok(HybridRagResult {
            hits,
            anchors,
            stage1_candidates,
            searchable_nodes,
            diffusion: Some(RagDiffusion {
                alpha: ppr_alpha,
                max_iters: ppr_iters,
                tolerance: HYBRID_PPR_TOLERANCE,
                iterations: ranking.iterations,
                residual: ranking.residual,
                converged: ranking.converged,
            }),
        })
    }

    /// Record direct evidence against a node: it becomes `Falsified` and refuted,
    /// its confidence is pinned to 0, its entity is revoked and every validated
    /// dependency touching it is retracted. Dependents move at the next
    /// [`Self::evolve_epistemic_fixed_point`]. Repeating the call changes nothing.
    /// Returns the number of validated dependencies retracted. An unknown node or
    /// an axiom is an error.
    pub fn falsify_node(&self, node_id: u32) -> Result<usize, LodError> {
        self.write(|g| g.falsify_node_now(node_id))
    }

    fn falsify_node_now(&self, node_id: u32) -> Result<usize, LodError> {
        let mut guard = self.state.write();
        let st = &mut *guard;
        let node = st
            .nodes
            .get_mut(node_id as usize)
            .ok_or(LodError::NodeNotFound(node_id))?;
        if node.status == EpistemicStatus::Axiomatic {
            return Err(LodError::InvalidStateTransition(format!(
                "node {node_id} is axiomatic and cannot be falsified"
            )));
        }
        node.status = EpistemicStatus::Falsified;
        node.refuted = true;
        node.confidence = 0.0;
        let entity = node.entity_id;
        st.revocations.insert(entity);
        let before = st.validated_deps.len();
        st.validated_deps
            .retain(|(u, v)| *u != entity && *v != entity);
        Ok(before - st.validated_deps.len())
    }

    /// Withdraw the evidence [`Self::falsify_node`] recorded (or an inserted
    /// `Falsified` status): the node returns to `Hypothesized` at its prior and
    /// its entity is no longer revoked, unless [`Self::revoke_entity`] revoked it.
    /// Dependents recover at the next [`Self::evolve_epistemic_fixed_point`].
    /// A node that is not refuted is an error. Refused, with nothing changed,
    /// when the node's edges would close a cycle that admission
    /// ([`Self::add_edge`]) refuses.
    pub fn retract_falsification(&self, node_id: u32) -> Result<(), LodError> {
        self.write(|g| g.retract_falsification_now(node_id))
    }

    fn retract_falsification_now(&self, node_id: u32) -> Result<(), LodError> {
        let mut guard = self.state.write();
        let st = &mut *guard;
        let node = st
            .nodes
            .get_mut(node_id as usize)
            .ok_or(LodError::NodeNotFound(node_id))?;
        if !node.refuted {
            return Err(LodError::InvalidStateTransition(format!(
                "node {node_id} is not refuted by evidence"
            )));
        }
        let before = NodeMutable::of(node);
        node.refuted = false;
        node.status = EpistemicStatus::Hypothesized;
        node.confidence = node.prior;
        let entity = node.entity_id;
        // Unpinning gives the node its incoming edges back, which may close a
        // cycle admission never saw.
        let snapshot = self.csr_snapshot.load_full();
        if let Some(refusal) = admission_refusal(&st.nodes, &snapshot, &st.edge_buffer) {
            before.restore(&mut st.nodes[node_id as usize]);
            return Err(refusal);
        }
        if !st.manual_revocations.contains(&entity) {
            st.revocations.remove(&entity);
        }
        Ok(())
    }

    /// [`Self::evolve_signed_epistemic_fixed_point_within`] with the default
    /// falsification gain [`DEFAULT_FALSIFICATION_GAIN`] and the step budget
    /// [`MAX_FIXED_POINT_STEPS`].
    pub fn evolve_epistemic_fixed_point(
        &self,
        beta: f32,
        tolerance: f32,
        theta_lo: f32,
        theta_hi: f32,
    ) -> Result<FixedPointReport, LodError> {
        self.evolve_signed_epistemic_fixed_point_within(
            beta,
            DEFAULT_FALSIFICATION_GAIN,
            tolerance,
            theta_lo,
            theta_hi,
            MAX_FIXED_POINT_STEPS,
        )
    }

    /// Evolve every node's confidence to the fixed point of
    ///
    /// `c_v = (1 - beta) pi_v + beta max(0, P+_v c - gamma P-_v c)`, clamped to
    /// [0, 1], `0 < beta < 1`, `gamma >= 0`,
    ///
    /// and move statuses by hysteresis on the result.
    ///
    /// - `P+[v][u]` is the weight of the `DependsOn` / `CausalTransition` /
    ///   `CoarseGrain` edges `u -> v` (`v` depends on `u`) divided by the total
    ///   such weight into `v`; `P-[v][u]` the same over the `Falsifies` edges
    ///   (`u` is evidence against `v`). Both read the CSR snapshot and the
    ///   pending buffer. A node with no support edge coming in is supported by
    ///   its own prior: `P+_v c` is `pi_v`. An axiom and a refuted node take no
    ///   edge and hold their prior. With `gamma = 0` the `Falsifies` edges are
    ///   left out entirely and the map is the unsigned `(1 - beta) pi + beta P+ c`.
    /// - `pi` is 1 for an axiom, 0 for a refuted node and the node's prior
    ///   otherwise.
    /// - The dependency graph is split into strongly connected components
    ///   (Tarjan) and solved in topological order, sources first. A component
    ///   of one node without a self-loop takes one evaluation. A cyclic
    ///   component is iterated alone from `pi`, its inputs held, until
    ///   `||x^{k+1} - x^k||_inf < tolerance`, which its Lipschitz bound `q < 1`
    ///   reaches by `k_max = ceil(ln(tolerance (1 - q) / ||x^1 - x^0||_inf) / ln q)`.
    ///   `q` is `beta` times the largest row sum of `P+ + gamma P-` inside the
    ///   component; `q >= 1` is [`LodError::FixedPointNotContractive`]. A block
    ///   that has not stopped at `min(k_max, max_steps, MAX_FIXED_POINT_STEPS)`
    ///   steps, or any non-finite value, is [`LodError::FixedPointDiverged`].
    ///   On either error the graph is unchanged.
    /// - The result depends only on priors, evidence, edges and the arguments:
    ///   undoing a change of evidence and evolving again reproduces the earlier
    ///   confidences bit for bit.
    /// - Hysteresis: confidence below `theta_lo` makes a node `Falsified` and
    ///   revokes its entity; above `theta_hi` makes it `Validated` and lifts that
    ///   revocation (never one made by [`Self::revoke_entity`]); in between the
    ///   status is kept. Validated dependencies are rebuilt from the new statuses.
    ///
    /// Arguments outside `0 < beta < 1`, finite `gamma >= 0`, `tolerance > 0`,
    /// `0 < theta_lo < theta_hi < 1` are `InvalidQuery`.
    pub fn evolve_signed_epistemic_fixed_point_within(
        &self,
        beta: f32,
        gamma: f32,
        tolerance: f32,
        theta_lo: f32,
        theta_hi: f32,
        max_steps: usize,
    ) -> Result<FixedPointReport, LodError> {
        self.write(|g| {
            g.evolve_signed_epistemic_fixed_point_within_now(
                beta, gamma, tolerance, theta_lo, theta_hi, max_steps,
            )
        })
    }

    fn evolve_signed_epistemic_fixed_point_within_now(
        &self,
        beta: f32,
        gamma: f32,
        tolerance: f32,
        theta_lo: f32,
        theta_hi: f32,
        max_steps: usize,
    ) -> Result<FixedPointReport, LodError> {
        let bad = |detail: String| Err(LodError::InvalidQuery(detail));
        if !(beta.is_finite() && beta > 0.0 && beta < 1.0) {
            return bad(format!("beta must satisfy 0 < beta < 1, got {beta}"));
        }
        if !(gamma.is_finite() && gamma >= 0.0) {
            return bad(format!("gamma must be finite and nonnegative, got {gamma}"));
        }
        if !(tolerance.is_finite() && tolerance > 0.0) {
            return bad(format!(
                "tolerance must be finite and positive, got {tolerance}"
            ));
        }
        if !(theta_lo.is_finite()
            && theta_hi.is_finite()
            && 0.0 < theta_lo
            && theta_lo < theta_hi
            && theta_hi < 1.0)
        {
            return bad(format!(
                "thresholds must satisfy 0 < theta_lo < theta_hi < 1, got {theta_lo} and {theta_hi}"
            ));
        }

        let mut guard = self.state.write();
        self.evolve_locked(
            &mut guard,
            beta,
            gamma,
            tolerance,
            theta_lo,
            theta_hi,
            max_steps,
            OnNonContractive::Refuse,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn evolve_locked(
        &self,
        st: &mut GraphState,
        beta: f32,
        gamma: f32,
        tolerance: f32,
        theta_lo: f32,
        theta_hi: f32,
        max_steps: usize,
        on_non_contractive: OnNonContractive,
    ) -> Result<FixedPointReport, LodError> {
        let checkpoint = self.capture(st);
        let snapshot = self.csr_snapshot.load_full();
        let rows = signed_rows(&st.nodes, &snapshot, &st.edge_buffer, gamma > 0.0);
        let prior: Vec<f64> = st
            .nodes
            .iter()
            .map(|n| match (n.status, n.refuted) {
                (EpistemicStatus::Axiomatic, _) => 1.0,
                (_, true) => 0.0,
                _ => f64::from(n.prior),
            })
            .collect();
        let run = solve_by_blocks(
            &rows,
            &prior,
            f64::from(beta),
            f64::from(gamma),
            f64::from(tolerance),
            max_steps.min(MAX_FIXED_POINT_STEPS),
            on_non_contractive,
        )?;

        // Converged: commit. Nothing above this line changed the graph.
        let mut transitions = Vec::new();
        let mut revoked_entities = Vec::new();
        let mut reinstated_entities = Vec::new();
        let mut pinned = 0;
        for (id, &c) in run.confidences.iter().enumerate() {
            let confidence = c.clamp(0.0, 1.0) as f32;
            let node = &st.nodes[id];
            let from = node.status;
            let to = if is_pinned(node) {
                pinned += 1;
                from
            } else if confidence < theta_lo {
                EpistemicStatus::Falsified
            } else if confidence > theta_hi {
                EpistemicStatus::Validated
            } else {
                from
            };
            // Write only a node that changes: a write copies its chunk, and a
            // copied chunk is a block the next durable commit must write.
            if node.confidence.to_bits() != confidence.to_bits() || to != from {
                let node = &mut st.nodes[id];
                node.confidence = confidence;
                node.status = to;
            }
            if to == from {
                continue;
            }
            let node = &st.nodes[id];
            transitions.push(StatusTransition {
                node: node.id,
                entity_id: node.entity_id,
                from,
                to,
                confidence,
            });
            if to.is_falsified() {
                st.revocations.insert(node.entity_id);
                revoked_entities.push(node.entity_id);
            } else if from.is_falsified() && !st.manual_revocations.contains(&node.entity_id) {
                st.revocations.remove(&node.entity_id);
                reinstated_entities.push(node.entity_id);
            }
        }
        let dependencies = validated_dependencies(&st.nodes, &snapshot, &st.edge_buffer);
        let retracted_dependencies = st.validated_deps.difference(&dependencies).count();
        let added_dependencies = dependencies.difference(&st.validated_deps).count();
        st.validated_deps = dependencies;

        let falsification_edges = rows.falsifies.iter().filter(|&&f| f).count();
        let block = run.reported.unwrap_or(BlockRun {
            iterations: 0,
            k_max: 0,
            initial_delta: 0.0,
            residual: 0.0,
        });
        Ok(FixedPointReport {
            beta,
            gamma,
            tolerance,
            theta_lo,
            theta_hi,
            nodes: st.nodes.len(),
            pinned,
            dependency_edges: rows.sources.len() - falsification_edges,
            falsification_edges,
            scc_count: run.scc_count,
            trivial_scc_count: run.trivial_scc_count,
            cyclic_scc_count: run.cyclic_scc_count,
            max_scc_size: run.max_scc_size,
            contraction: run.contraction,
            node_updates: run.node_updates,
            iterations: block.iterations,
            k_max: block.k_max,
            initial_delta: block.initial_delta,
            residual: block.residual,
            error_bound: run.error_bound,
            transitions,
            revoked_entities,
            reinstated_entities,
            retracted_dependencies,
            added_dependencies,
            adapted_blocks: run.adapted,
            checkpoint,
        })
    }

    /// Capture the whole mutable state atomically. See [`GraphCheckpoint`].
    pub fn create_checkpoint(&self) -> GraphCheckpoint {
        let st = self.state.read();
        self.capture(&st)
    }

    fn capture(&self, st: &GraphState) -> GraphCheckpoint {
        GraphCheckpoint {
            graph_id: self.graph_id,
            seq: {
                let seq = self.checkpoint_seq.fetch_add(1, Ordering::Relaxed) + 1;
                if self.candidate {
                    self.issued.lock().push(seq);
                }
                seq
            },
            node_states: st.nodes.iter().map(NodeMutable::of).collect(),
            csr: self.csr_snapshot.load_full(),
            edge_buffer: st.edge_buffer.clone(),
            revocations: st.revocations.clone(),
            manual_revocations: st.manual_revocations.clone(),
            privileges: st.privileges.clone(),
            validated_deps: st.validated_deps.clone(),
            embedding_dim: st.embedding_dim,
        }
    }

    /// Restore `checkpoint` atomically: nodes added after it are removed (their
    /// ids become free again, with their aliases and the edges those linked),
    /// statuses, confidences, refutation marks, bands, parents, CSR snapshot,
    /// pending edges, revocations, privileges, validated dependencies and the
    /// embedding dimension return to its values.
    ///
    /// Every write since the checkpoint is discarded, including writes by other
    /// threads; use [`Self::transact`] to keep writers serialized. Refused: a
    /// checkpoint of another graph, and one taken after an earlier checkpoint that
    /// has since been restored (its state no longer exists).
    pub fn rollback_checkpoint(&self, checkpoint: &GraphCheckpoint) -> Result<(), LodError> {
        self.write(|g| g.rollback_checkpoint_now(checkpoint))
    }

    fn rollback_checkpoint_now(&self, checkpoint: &GraphCheckpoint) -> Result<(), LodError> {
        let _flush = self.flush_lock.lock();
        let mut st = self.state.write();
        if checkpoint.graph_id != self.graph_id {
            return Err(LodError::CheckpointRejected(
                "checkpoint belongs to another graph".into(),
            ));
        }
        let seq = checkpoint.seq;
        if st.discarded.iter().any(|&(a, b)| seq > a && seq <= b) {
            return Err(LodError::CheckpointRejected(format!(
                "checkpoint {seq} was discarded by an earlier rollback"
            )));
        }
        let keep = checkpoint.node_states.len();
        if keep > st.nodes.len() {
            return Err(LodError::CheckpointRejected(format!(
                "checkpoint has {keep} nodes but the graph has {}",
                st.nodes.len()
            )));
        }

        // Copy-on-write: touch only what differs, so a rollback dirties
        // only the chunks and indexes its discarded writes changed.
        if keep < st.nodes.len() {
            st.nodes.truncate(keep);
            Arc::make_mut(&mut st.entity_index).retain(|_, id| (*id as usize) < keep);
            if st
                .alias_index
                .values()
                .flatten()
                .any(|id| *id as usize >= keep)
            {
                Arc::make_mut(&mut st.alias_index).retain(|_, holders| {
                    holders.retain(|id| (*id as usize) < keep);
                    !holders.is_empty()
                });
            }
        }
        st.embedding_dim = checkpoint.embedding_dim;
        for (id, state) in checkpoint.node_states.iter().enumerate() {
            if NodeMutable::of(&st.nodes[id]) != *state {
                state.restore(&mut st.nodes[id]);
            }
        }
        st.edge_buffer = checkpoint.edge_buffer.clone();
        st.revocations = checkpoint.revocations.clone();
        st.manual_revocations = checkpoint.manual_revocations.clone();
        st.privileges = checkpoint.privileges.clone();
        st.validated_deps = checkpoint.validated_deps.clone();
        self.csr_snapshot.store(Arc::clone(&checkpoint.csr));
        st.generation += 1;

        // Checkpoints taken after this one describe states that no longer exist.
        let upto = self.checkpoint_seq.load(Ordering::Relaxed);
        if upto > seq {
            while st.discarded.last().is_some_and(|&(a, _)| a >= seq) {
                st.discarded.pop();
            }
            match st.discarded.last_mut() {
                Some(last) if last.1 >= seq => last.1 = upto,
                _ => st.discarded.push((seq, upto)),
            }
        }
        Ok(())
    }

    /// Run `f` as one write transaction on a private candidate of the graph.
    /// Readers keep seeing the state before `f` until the candidate is
    /// published whole: after its durable commit when the graph is persistent.
    /// An error from `f` or from the commit drops the candidate; the live state
    /// is never touched. Writers are serialized. A `transact` inside `f` runs
    /// on a candidate of the candidate and publishes into it. `f` must write
    /// through its argument: a direct write to this graph from inside `f`
    /// waits for the transaction and deadlocks.
    pub fn transact<T>(
        &self,
        f: impl FnOnce(&LodGraph) -> Result<T, LodError>,
    ) -> Result<T, LodError> {
        let _txn = self.txn_lock.lock();
        self.check_persistence()?;
        let candidate = self.fork();
        let value = match f(&candidate) {
            Ok(value) => value,
            Err(error) => {
                self.discard_unpublished(candidate);
                return Err(error);
            }
        };
        self.publish(candidate)?;
        Ok(value)
    }

    /// Run `f` on a private candidate and drop it: the report of a write that
    /// is never applied. Nothing is published or persisted, so no reader can
    /// see the trial state, not even for a moment.
    pub fn dry_run<T>(
        &self,
        f: impl FnOnce(&LodGraph) -> Result<T, LodError>,
    ) -> Result<T, LodError> {
        let candidate = self.fork();
        let result = f(&candidate);
        self.discard_unpublished(candidate);
        result
    }

    /// Fail-closed prior check over the action, its coarse ancestors and causal
    /// successors. Reads pending edges as well as CSR under one graph read lock.
    /// Missing action knowledge is explicitly represented by an empty prior.
    pub fn planning_prior(&self, entity: u64) -> Result<Vec<LodNode>, LodError> {
        let st = self.state.read();
        let fail = |reason: String| LodError::InvalidQuery(reason);
        if st.revocations.contains(&entity) {
            return Err(fail(format!("entity {entity} is revoked")));
        }
        let Some(&root) = st.entity_index.get(&entity) else {
            return Ok(Vec::new());
        };
        let csr = self.csr_snapshot.load_full();
        let mut pending = vec![root];
        let mut visited = HashSet::new();
        let mut facts = Vec::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            let node = &st.nodes[id as usize];
            if st.revocations.contains(&node.entity_id)
                || node.status.is_falsified()
                || !node.confidence.is_finite()
                || node.confidence < 0.3
            {
                return Err(fail(format!(
                    "graph prior blocks action {entity}: node {} has status {:?}, confidence {}",
                    node.entity_id, node.status, node.confidence
                )));
            }
            if id != root
                && node.confidence >= 0.6
                && (node.status.is_active_truth()
                    || matches!(node.band, LodBand::Lod2Milestone | LodBand::Lod3Systemic))
            {
                facts.push(node.clone());
            }
            if let Some(parent) = node.parent_id {
                pending.push(parent);
            }
            for (target, ty, weight) in csr.neighbors(id).chain(
                st.edge_buffer
                    .iter()
                    .filter(|e| e.source == id)
                    .map(|e| (e.target, e.edge_type, e.weight)),
            ) {
                if weight > 0.0 && matches!(ty, EdgeType::CausalTransition | EdgeType::CoarseGrain)
                {
                    pending.push(target);
                }
            }
        }
        facts.sort_by_key(|n| (std::cmp::Reverse(n.band), n.entity_id));
        Ok(facts)
    }

    /// Record a model/policy observation and evolve, as one transaction.
    /// Observation confidence expresses certainty that the diagnostic occurred,
    /// not calibrated real-world causality. Identical payloads reuse evidence.
    ///
    /// The evolution runs at [`ADMISSION_BETA`], [`ADMISSION_GAMMA`], so on a
    /// graph built through [`Self::add_edge`] every cycle contracts inside the
    /// step budget. A cycle that does not (an invariant broken some other way)
    /// does not fail the reflection: its internal falsifier gain is lowered
    /// until it does and the block is listed in
    /// [`FixedPointReport::adapted_blocks`]. The
    /// observation's own `Falsifies` edge comes from a node with no incoming
    /// edge, so it is never inside a cycle and always presses at full gain.
    ///
    /// The action always ends revoked: by the evolution when its confidence
    /// falls below the threshold, otherwise by an explicit quarantine
    /// ([`ReflectionRevocation::Quarantine`]).
    ///
    /// On failure no staged evidence is published and the caller receives the
    /// error. A refused reflection quarantines the target durably, by a second
    /// transaction. A reflection or quarantine the disk refuses revokes the
    /// target in memory only and poisons the graph: never fail open. Call it on
    /// the live graph: inside a transaction the quarantine is part of the
    /// outer candidate and is dropped with it if that transaction fails. An axiom cannot be silently rewritten.
    pub fn reflect_failure(
        &self,
        action: u32,
        payload: &str,
        timestamp_ns: u64,
    ) -> Result<ReflectionReport, LodError> {
        let result = self.transact(|g| {
            let entity = u64::from(action);
            let digest = crate::node::payload_digest(&format!("action:{action}\n{payload}"));
            let evidence_entity =
                u64::from_le_bytes(digest[..8].try_into().unwrap()) | (1_u64 << 63);
            let mut guard = g.state.write();
            let staged = &mut *guard;
            let additional = usize::from(!staged.entity_index.contains_key(&entity))
                + usize::from(!staged.entity_index.contains_key(&evidence_entity));
            check_node_capacity(staged.nodes.len(), additional)?;
            let target = match staged.entity_index.get(&entity) {
                Some(&id) => id,
                None => g.insert_node(
                    staged,
                    LodNode::new(
                        0,
                        LodBand::Lod0Atomic,
                        MixedCurvatureCoord::origin(),
                        format!("action {action}"),
                        entity,
                    ),
                )?,
            };
            if staged.nodes[target as usize].status == EpistemicStatus::Axiomatic {
                return Err(LodError::InvalidQuery(format!(
                    "cannot evolve axiomatic action {action}; quarantined"
                )));
            }
            let evidence = match staged.entity_index.get(&evidence_entity) {
                Some(&id) => {
                    if !staged.nodes[id as usize].is_internal_evidence()
                        || staged.nodes[id as usize].payload.as_deref() != Some(payload)
                    {
                        return Err(LodError::InvalidQuery("reflection entity collision".into()));
                    }
                    let csr = g.csr_snapshot.load_full();
                    let has_edge = csr
                        .neighbors(id)
                        .chain(
                            staged
                                .edge_buffer
                                .iter()
                                .filter(|e| e.source == id)
                                .map(|e| (e.target, e.edge_type, e.weight)),
                        )
                        .any(|(to, ty, weight)| {
                            to == target && ty == EdgeType::Falsifies && weight > 0.0
                        });
                    if !has_edge {
                        return Err(LodError::InvalidQuery(
                            "existing reflection lacks its falsification edge".into(),
                        ));
                    }
                    id
                }
                None => {
                    let node = LodNode::new(
                        0,
                        LodBand::Lod0Atomic,
                        MixedCurvatureCoord::origin(),
                        format!("failure observation for action {action}"),
                        evidence_entity,
                    )
                    .with_prior(1.0)
                    .with_payload(
                        payload,
                        Some(crate::node::INTERNAL_EVIDENCE_SOURCE.into()),
                        timestamp_ns,
                    )?;
                    let id = g.insert_node(staged, node)?;
                    g.push_edge(staged, id, target, EdgeType::Falsifies, 1.0)?;
                    id
                }
            };
            let evolution = g.evolve_locked(
                staged,
                ADMISSION_BETA,
                ADMISSION_GAMMA,
                REFLECTION_TOLERANCE,
                REFLECTION_THETA_LO,
                REFLECTION_THETA_HI,
                MAX_FIXED_POINT_STEPS,
                OnNonContractive::AdaptGain,
            )?;
            // Judged by the action's own status, not by the revocation set: an
            // earlier quarantine or manual revocation is already in that set.
            let revocation = if staged.nodes[target as usize].status.is_falsified() {
                ReflectionRevocation::Evolution
            } else {
                staged.revocations.insert(entity);
                staged.manual_revocations.insert(entity);
                ReflectionRevocation::Quarantine
            };
            Ok(ReflectionReport {
                evidence,
                target,
                revocation,
                target_confidence: staged.nodes[target as usize].confidence,
                evolution,
            })
        });
        result.map_err(|error| {
            let entity = u64::from(action);
            // The disk refused the reflection itself: no durable quarantine for
            // a disk fault, but never fail open either. Revoke in memory and
            // poison, so a host that honors `check_persistence` stops serving
            // until a restart reloads the last committed state.
            if matches!(error, LodError::Persistence(_)) {
                tracing::error!(action, error = %error, "lodgraph: reflection not committed; revoked in memory and poisoned");
                self.revoke_uncommitted(entity);
                return error;
            }
            // The reflection was refused on its merits: quarantine the action
            // durably, in its own transaction.
            if let Err(quarantine) = self.transact(|g| {
                g.revoke_entity_now(entity);
                Ok(())
            }) {
                tracing::error!(
                    action,
                    reflection = %error,
                    quarantine = %quarantine,
                    "lodgraph: quarantine not committed; revoked in memory and poisoned"
                );
                self.revoke_uncommitted(entity);
            }
            error
        })
    }

    /// Register a privilege bitflag for an agent.
    pub fn add_privilege(&self, agent_id: u64, privilege: u32) -> Result<(), LodError> {
        self.write(|g| {
            g.add_privilege_now(agent_id, privilege);
            Ok(())
        })
    }

    fn add_privilege_now(&self, agent_id: u64, privilege: u32) {
        let mut st = self.state.write();
        let list = st.privileges.entry(agent_id).or_default();
        if !list.contains(&privilege) {
            list.push(privilege);
        }
    }

    /// Explicitly revoke an entity ID in the cognitive graph. No confidence
    /// evolution and no evidence retraction lifts this revocation.
    pub fn revoke_entity(&self, entity_id: u64) -> Result<(), LodError> {
        self.write(|g| {
            g.revoke_entity_now(entity_id);
            Ok(())
        })
    }

    fn revoke_entity_now(&self, entity_id: u64) {
        let mut st = self.state.write();
        st.revocations.insert(entity_id);
        st.manual_revocations.insert(entity_id);
    }
}

impl GraphFactProvider for LodGraph {
    type DepIter<'a> = std::vec::IntoIter<(u64, u64)>;

    fn active_validated_dependencies<'a>(&'a self) -> Self::DepIter<'a> {
        let deps: Vec<(u64, u64)> = self.state.read().validated_deps.iter().copied().collect();
        deps.into_iter()
    }

    fn is_revoked(&self, entity_id: u64) -> bool {
        self.state.read().revocations.contains(&entity_id)
    }

    fn has_privilege(&self, agent_id: u64, privilege: u32) -> bool {
        self.state
            .read()
            .privileges
            .get(&agent_id)
            .is_some_and(|list| list.contains(&privilege))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::LodBand;

    #[test]
    fn normalization_handles_empty_exact_and_invalid_tracks() {
        normalize_anchor_distances(&mut []).unwrap();
        let mut exact = [
            (0, 0.0, AnchorMatch::Primary),
            (1, 0.0, AnchorMatch::Alias(0)),
        ];
        normalize_anchor_distances(&mut exact).unwrap();
        assert!(exact.iter().all(|a| a.1 == 0.0));
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            assert!(normalize_anchor_distances(&mut [(0, bad, AnchorMatch::Embedding)]).is_err());
        }
        assert!(check_node_capacity(MAX_GRAPH_NODES, 0).is_ok());
        assert!(check_node_capacity(MAX_GRAPH_NODES, 1).is_err());
    }

    fn node(label: &str, entity: u64) -> LodNode {
        LodNode::new(
            0,
            LodBand::Lod0Atomic,
            MixedCurvatureCoord::origin(),
            label,
            entity,
        )
    }

    fn node_with(label: &str, entity: u64, coord: MixedCurvatureCoord) -> LodNode {
        LodNode::new(0, LodBand::Lod0Atomic, coord, label, entity)
    }

    const BETA: f32 = 0.85;
    const TOL: f32 = 1e-6;

    fn evolve(graph: &LodGraph, theta_lo: f32, theta_hi: f32) -> FixedPointReport {
        graph
            .evolve_epistemic_fixed_point(BETA, TOL, theta_lo, theta_hi)
            .unwrap()
    }

    fn status(graph: &LodGraph, id: u32) -> EpistemicStatus {
        graph.get_node(id).unwrap().status
    }

    fn confidences(graph: &LodGraph) -> Vec<f32> {
        (0..graph.node_count() as u32)
            .map(|id| graph.get_node(id).unwrap().confidence)
            .collect()
    }

    fn sorted_deps(graph: &LodGraph) -> Vec<(u64, u64)> {
        let mut deps: Vec<_> = graph.active_validated_dependencies().collect();
        deps.sort_unstable();
        deps
    }

    #[test]
    fn axiom_survives_direct_falsification_and_evolution() {
        let graph = LodGraph::new();
        let source = graph.add_node(node("source", 10)).unwrap();
        let axiom = graph
            .add_node(node("axiom", 11).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        graph
            .add_edge(source, axiom, EdgeType::DependsOn, 1.0)
            .unwrap();
        graph.flush_edges_to_csr().unwrap();
        assert!(matches!(
            graph.falsify_node(axiom),
            Err(LodError::InvalidStateTransition(_))
        ));
        graph.falsify_node(source).unwrap();
        let report = evolve(&graph, 0.2, 0.8);
        assert!(report.transitions.is_empty());
        assert_eq!(report.pinned, 2);
        // The axiom's only dependency is refuted; it is pinned all the same.
        assert_eq!(status(&graph, axiom), EpistemicStatus::Axiomatic);
        assert_eq!(confidences(&graph), vec![0.0, 1.0]);
        assert!(graph.is_revoked(10) && !graph.is_revoked(11));
        assert_eq!(
            graph.falsify_node(99).unwrap_err(),
            LodError::NodeNotFound(99)
        );
        assert_eq!(
            graph.retract_falsification(99).unwrap_err(),
            LodError::NodeNotFound(99)
        );
        assert!(matches!(
            graph.retract_falsification(axiom),
            Err(LodError::InvalidStateTransition(_))
        ));
    }

    #[test]
    fn ppr_omits_falsified_and_revoked_nodes() {
        let graph = LodGraph::new();
        let live = graph.add_node(node("live", 20)).unwrap();
        let false_node = graph
            .add_node(node("false", 21).with_status(EpistemicStatus::Falsified))
            .unwrap();
        let revoked = graph.add_node(node("revoked", 22)).unwrap();
        graph.revoke_entity(22).unwrap();
        let result = graph
            .query_ppr(
                &[(live, 1.0), (false_node, 1.0), (revoked, 1.0)],
                0.15,
                10,
                1e-4,
            )
            .unwrap();
        assert_eq!(
            result.ranked.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![live]
        );
    }

    #[test]
    fn ppr_refuses_unknown_seed_and_bad_alpha() {
        let graph = LodGraph::new();
        let a = graph.add_node(node("a", 1)).unwrap();
        assert!(matches!(
            graph.query_ppr(&[(a + 1, 1.0)], 0.15, 10, 1e-4),
            Err(LodError::InvalidQuery(_))
        ));
        assert!(matches!(
            graph.query_ppr(&[(a, 1.0)], 1.0, 10, 1e-4),
            Err(LodError::InvalidQuery(_))
        ));
    }

    #[test]
    fn test_lod_graph_workflow() {
        let graph = LodGraph::new();
        let coord0 = MixedCurvatureCoord::origin();
        let id0 = graph
            .add_node(
                LodNode::new(0, LodBand::Lod0Atomic, coord0, "sensor_read", 1001)
                    .with_status(EpistemicStatus::Validated),
            )
            .unwrap();
        let id1 = graph
            .add_node(LodNode::new(
                1,
                LodBand::Lod1Cluster,
                coord0,
                "sub_goal",
                1002,
            ))
            .unwrap();
        let id2 = graph
            .add_node(LodNode::new(
                2,
                LodBand::Lod2Milestone,
                coord0,
                "milestone",
                1003,
            ))
            .unwrap();

        // Add dependencies: 0 -> 1 -> 2
        graph.add_edge(id0, id1, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(id1, id2, EdgeType::DependsOn, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();

        assert!(!graph.is_revoked(1001));
        assert!(!graph.is_revoked(1002));

        // Refuting node 1 takes node 2, whose only dependency it is, below theta_lo:
        // c_2 = (1 - beta) 0.5 + beta 0 = 0.075.
        graph.falsify_node(id1).unwrap();
        assert!(graph.is_revoked(1002) && !graph.is_revoked(1003));
        let report = evolve(&graph, 0.2, 0.8);
        let [moved] = report.transitions[..] else {
            panic!("expected one transition, got {:?}", report.transitions);
        };
        assert_eq!((moved.node, moved.entity_id), (id2, 1003));
        assert_eq!(
            (moved.from, moved.to),
            (EpistemicStatus::Hypothesized, EpistemicStatus::Falsified)
        );
        assert!((moved.confidence - 0.075).abs() < 1e-6);
        assert_eq!(report.revoked_entities, vec![1003]);
        // Node 0 has no dependency: it keeps its prior and, inside the band, its status.
        let got = confidences(&graph);
        assert_eq!((got[0], got[1], got[2]), (0.5, 0.0, moved.confidence));
        assert_eq!(status(&graph, id0), EpistemicStatus::Validated);

        assert!(graph.is_revoked(1002));
        assert!(graph.is_revoked(1003));
        assert!(!graph.is_revoked(1001));
    }

    #[test]
    fn test_two_stage_recall_filters_falsified() {
        let graph = LodGraph::new();
        let coord0 = MixedCurvatureCoord::origin();
        let mut coord1 = MixedCurvatureCoord::origin();
        coord1.euclidean[0] = 5.0;

        let id0 = graph
            .add_node(
                LodNode::new(0, LodBand::Lod0Atomic, coord0, "target_node", 2001)
                    .with_hdc_fingerprint([0b1111, 0, 0, 0]),
            )
            .unwrap();
        let id1 = graph
            .add_node(
                LodNode::new(1, LodBand::Lod0Atomic, coord1, "distant_node", 2002)
                    .with_hdc_fingerprint([0b0000, 0, 0, 0]),
            )
            .unwrap();

        let query_fp = [0b1111, 0, 0, 0];
        let recalled = graph.two_stage_recall(&coord0, &query_fp, 2, 0.1).unwrap();
        assert_eq!(recalled.len(), 2);
        assert_eq!(recalled[0].0, id0);

        // A falsified node is never recalled again.
        graph.falsify_node(id0).unwrap();
        let recalled_after = graph.two_stage_recall(&coord0, &query_fp, 2, 0.1).unwrap();
        assert_eq!(recalled_after.len(), 1);
        assert_eq!(recalled_after[0].0, id1);
        assert!(graph.two_stage_recall(&coord0, &query_fp, 0, 0.1).is_err());
        assert!(graph
            .two_stage_recall(&coord0, &query_fp, 1, f32::NAN)
            .is_err());
    }

    #[test]
    fn out_of_domain_coord_is_refused_at_insert_and_query() {
        let graph = LodGraph::new();
        let mut bad = MixedCurvatureCoord::origin();
        // Public fields bypass `new`; the graph must still refuse it.
        bad.hyperbolic = [1.0, 0.0, 0.0, 0.0];
        let err = graph
            .add_node(LodNode::new(0, LodBand::Lod0Atomic, bad, "bad", 4001))
            .unwrap_err();
        assert_eq!(
            err,
            LodError::Geometry(crate::manifold::Reject::DomainViolation)
        );
        assert_eq!(graph.node_count(), 0);
        graph.add_node(node("good", 4002)).unwrap();
        assert!(graph.two_stage_recall(&bad, &[0; 4], 1, 0.1).is_err());
    }

    #[test]
    fn add_node_refuses_duplicate_entity_and_bad_parent() {
        let graph = LodGraph::new();
        let a = graph.add_node(node("a", 7)).unwrap();
        assert_eq!(graph.node_for_entity(7), Some(a));
        assert_eq!(
            graph.add_node(node("again", 7)).unwrap_err(),
            LodError::DuplicateEntity(7)
        );
        let mut orphan = node("orphan", 8);
        orphan.parent_id = Some(5);
        assert!(matches!(
            graph.add_node(orphan),
            Err(LodError::InvalidNode(_))
        ));
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn add_edge_refuses_unknown_endpoint_and_bad_weight() {
        let graph = LodGraph::new();
        let a = graph.add_node(node("a", 1)).unwrap();
        assert!(matches!(
            graph.add_edge(a, 3, EdgeType::Semantic, 1.0),
            Err(LodError::InvalidEdge(_))
        ));
        assert!(matches!(
            graph.add_edge(a, a, EdgeType::Semantic, f32::NAN),
            Err(LodError::InvalidEdge(_))
        ));
        assert!(matches!(
            graph.add_edge(a, a, EdgeType::Semantic, -1.0),
            Err(LodError::InvalidEdge(_))
        ));
        assert_eq!(graph.pending_edge_count(), 0);
    }

    #[test]
    fn evolution_validates_supported_chain_and_refutation_falsifies_it() {
        let graph = LodGraph::new();
        let axiom = graph
            .add_node(node("axiom", 3000).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        let id0 = graph.add_node(node("hypo_0", 3001)).unwrap();
        let id1 = graph.add_node(node("hypo_1", 3002)).unwrap();
        let id2 = graph.add_node(node("hypo_2", 3003)).unwrap();
        graph
            .add_edge(axiom, id0, EdgeType::DependsOn, 1.0)
            .unwrap();
        graph.add_edge(id0, id1, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(id1, id2, EdgeType::DependsOn, 1.0).unwrap();
        assert!(sorted_deps(&graph).is_empty());

        // c = 0.075 + 0.85 * (dependency): 0.925, 0.86125, 0.8070625, all above 0.8.
        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!(report.transitions.len(), 3);
        assert_eq!(report.added_dependencies, 3);
        for (id, want) in [(id0, 0.925), (id1, 0.86125), (id2, 0.807_062_5)] {
            let n = graph.get_node(id).unwrap();
            assert_eq!(n.status, EpistemicStatus::Validated);
            assert!((n.confidence - want).abs() < 1e-5, "{id}: {}", n.confidence);
            assert_eq!(n.prior, 0.5);
        }
        assert_eq!(
            sorted_deps(&graph),
            vec![(3000, 3001), (3001, 3002), (3002, 3003)]
        );

        // Refuting node 0: c_1 = 0.075, c_2 = 0.075 + 0.85 * 0.075 = 0.13875.
        assert_eq!(graph.falsify_node(id0).unwrap(), 2);
        assert_eq!(graph.falsify_node(id0).unwrap(), 0);
        assert_eq!(sorted_deps(&graph), vec![(3002, 3003)]);
        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!(
            report
                .transitions
                .iter()
                .map(|t| (t.node, t.to))
                .collect::<Vec<_>>(),
            vec![
                (id1, EpistemicStatus::Falsified),
                (id2, EpistemicStatus::Falsified)
            ]
        );
        assert_eq!(report.revoked_entities, vec![3002, 3003]);
        assert_eq!(report.retracted_dependencies, 1);
        assert!(graph.is_revoked(3001) && graph.is_revoked(3002) && graph.is_revoked(3003));
        assert!(!graph.is_revoked(3000));
        assert!(sorted_deps(&graph).is_empty());
    }

    /// The dependency rebuild reads flushed (CSR) edges as well as pending ones.
    #[test]
    fn evolution_rebuilds_dependencies_from_flushed_edges() {
        let graph = LodGraph::new();
        let src = graph
            .add_node(node("src", 5001).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        let dst = graph.add_node(node("dst", 5002)).unwrap();
        graph.add_edge(src, dst, EdgeType::DependsOn, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();
        assert_eq!(graph.pending_edge_count(), 0);
        assert!(graph.active_validated_dependencies().next().is_none());

        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!(report.dependency_edges, 1);
        assert_eq!(status(&graph, dst), EpistemicStatus::Validated);
        assert_eq!(sorted_deps(&graph), vec![(5001, 5002)]);
    }

    /// A weighted cycle `A -> B -> C -> A` fed by an axiom `X -> A`: the iteration
    /// stops within the logarithmic bound at the solution of `(I - beta P) c =
    /// (1 - beta) pi`, solved here by hand.
    #[test]
    fn cycle_converges_to_the_unique_fixed_point_within_k_max() {
        let graph = LodGraph::new();
        let x = graph
            .add_node(node("x", 1).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        let a = graph.add_node(node("a", 2).with_prior(0.5)).unwrap();
        let b = graph.add_node(node("b", 3).with_prior(0.25)).unwrap();
        let c = graph.add_node(node("c", 4).with_prior(0.75)).unwrap();
        graph.add_edge(x, a, EdgeType::DependsOn, 3.0).unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();
        // The rest stays pending: the iteration must read both stores.
        graph
            .add_edge(b, c, EdgeType::CausalTransition, 2.0)
            .unwrap();
        graph.add_edge(c, a, EdgeType::DependsOn, 1.0).unwrap();
        // Edges of other types carry no confidence. (`Falsifies` does, with a
        // negative sign: see the signed tests below.)
        graph.add_edge(b, a, EdgeType::Semantic, 50.0).unwrap();
        graph.add_edge(c, b, EdgeType::Validates, 50.0).unwrap();

        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!((report.nodes, report.pinned), (4, 1));
        assert_eq!(report.dependency_edges, 4);
        assert_eq!(report.falsification_edges, 0);
        // The axiom alone, then the loop {a, b, c}. Inside the loop the largest
        // row sum is 1 (b from a, c from b), so q = beta.
        assert_eq!(
            (
                report.scc_count,
                report.trivial_scc_count,
                report.cyclic_scc_count,
                report.max_scc_size
            ),
            (2, 1, 1, 3)
        );
        assert_eq!(report.contraction, f64::from(BETA));
        assert_eq!(report.node_updates, 1 + 3 * (report.iterations + 1));

        // a = k pa + beta (3/4 + c/4), b = k pb + beta a, c = k pc + beta b, k = 1 - beta.
        let beta = f64::from(BETA);
        let k = 1.0 - beta;
        let (pa, pb, pc) = (0.5, 0.25, 0.75);
        let exact_a = (k * pa + 0.75 * beta + 0.25 * beta * (k * pc + beta * k * pb))
            / (1.0 - 0.25 * beta.powi(3));
        let exact_b = k * pb + beta * exact_a;
        let exact_c = k * pc + beta * exact_b;
        let got = confidences(&graph);
        assert_eq!(got[x as usize], 1.0);
        for (id, exact) in [(a, exact_a), (b, exact_b), (c, exact_c)] {
            let err = (f64::from(got[id as usize]) - exact).abs();
            assert!(
                err <= report.error_bound + 1e-7,
                "node {id}: {} vs {exact}, bound {}",
                got[id as usize],
                report.error_bound
            );
        }

        // The explicit bound, recomputed from the report, and the stop inside it.
        let tol = f64::from(TOL);
        assert!(report.initial_delta > tol);
        let k_max = ((tol * k / report.initial_delta).ln() / beta.ln()).ceil() as usize;
        assert_eq!(report.k_max, k_max);
        assert!(report.iterations >= 1 && report.iterations <= report.k_max);
        assert!(report.residual < tol);
        assert!(report.error_bound < tol * beta / k);

        // An unchanged graph evolves to the same bits and changes no status.
        let again = evolve(&graph, 0.2, 0.8);
        assert_eq!(confidences(&graph), got);
        assert!(again.transitions.is_empty());
        assert_eq!(again.iterations, report.iterations);
    }

    /// Uniqueness: a cycle with no anchor at all, started from four different
    /// vectors, reaches one fixed point, and that point satisfies the equation.
    #[test]
    fn every_start_reaches_the_same_fixed_point_on_a_pure_cycle() {
        let graph = LodGraph::new();
        let priors = [0.2_f32, 0.5, 0.9];
        let ids: Vec<u32> = priors
            .iter()
            .enumerate()
            .map(|(i, &p)| graph.add_node(node("n", i as u64).with_prior(p)).unwrap())
            .collect();
        for i in 0..3 {
            graph
                .add_edge(ids[i], ids[(i + 1) % 3], EdgeType::DependsOn, 1.0)
                .unwrap();
        }
        let (beta, tol) = (0.9_f64, 1e-9_f64);
        let prior: Vec<f64> = priors.iter().map(|&p| f64::from(p)).collect();
        let runs: Vec<(BlockRun, Vec<f64>)> = {
            let st = graph.state.read();
            let rows = signed_rows(&st.nodes, &graph.csr_snapshot(), &st.edge_buffer, true);
            let members: Vec<u32> = ids.clone();
            [
                prior.clone(),
                vec![0.0; 3],
                vec![1.0; 3],
                vec![1.0, 0.0, 0.37],
            ]
            .into_iter()
            .map(|mut c| {
                let run = iterate_block(
                    &rows,
                    &prior,
                    &members,
                    beta,
                    beta,
                    1.0,
                    None,
                    tol,
                    usize::MAX,
                    &mut c,
                )
                .unwrap();
                (run, c)
            })
            .collect()
        };
        let bound = 2.0 * tol * beta / (1.0 - beta);
        for (run, c) in &runs {
            assert!(run.iterations <= run.k_max);
            for i in 0..3 {
                assert!((c[i] - runs[0].1[i]).abs() <= bound);
                // Node i depends on node i - 1.
                let rhs = (1.0 - beta) * prior[i] + beta * c[(i + 2) % 3];
                assert!((c[i] - rhs).abs() < tol);
            }
        }
    }

    /// A step budget below the bound is a refusal, and the graph is untouched.
    #[test]
    fn exhausted_step_budget_fails_closed_and_commits_nothing() {
        let graph = LodGraph::new();
        let priors = [0.2_f32, 0.5, 0.9];
        let ids: Vec<u32> = priors
            .iter()
            .enumerate()
            .map(|(i, &p)| graph.add_node(node("n", i as u64).with_prior(p)).unwrap())
            .collect();
        for i in 0..3 {
            graph
                .add_edge(ids[i], ids[(i + 1) % 3], EdgeType::DependsOn, 1.0)
                .unwrap();
        }
        let before = (confidences(&graph), sorted_deps(&graph));

        let err = graph
            .evolve_signed_epistemic_fixed_point_within(BETA, 1.0, TOL, 0.3, 0.6, 3)
            .unwrap_err();
        match err {
            LodError::FixedPointDiverged {
                iterations,
                k_max,
                residual,
                tolerance,
            } => {
                assert_eq!(iterations, 3);
                assert!(k_max > 3, "k_max {k_max}");
                assert!(residual >= tolerance);
            }
            other => panic!("expected FixedPointDiverged, got {other}"),
        }

        // beta = 0.9999 on a cycle needs about 134 000 steps for 1e-6: above the
        // hard cap, so the public entry point refuses too.
        let err = graph
            .evolve_epistemic_fixed_point(0.9999, TOL, 0.3, 0.6)
            .unwrap_err();
        match err {
            LodError::FixedPointDiverged {
                iterations, k_max, ..
            } => {
                assert_eq!(iterations, MAX_FIXED_POINT_STEPS);
                assert!(k_max > MAX_FIXED_POINT_STEPS);
            }
            other => panic!("expected FixedPointDiverged, got {other}"),
        }

        assert_eq!((confidences(&graph), sorted_deps(&graph)), before);
        for (&id, &p) in ids.iter().zip(&priors) {
            let n = graph.get_node(id).unwrap();
            assert_eq!((n.status, n.confidence), (EpistemicStatus::Hypothesized, p));
            assert!(!graph.is_revoked(n.entity_id));
        }
        // The same graph converges once the budget covers the bound.
        let report = evolve(&graph, 0.3, 0.6);
        assert!(report.iterations > 3);
    }

    #[test]
    fn evolution_refuses_out_of_range_arguments() {
        let graph = LodGraph::new();
        graph.add_node(node("a", 1)).unwrap();
        let bad = [
            (0.0, TOL, 0.2, 0.8),
            (1.0, TOL, 0.2, 0.8),
            (-0.5, TOL, 0.2, 0.8),
            (f32::NAN, TOL, 0.2, 0.8),
            (BETA, 0.0, 0.2, 0.8),
            (BETA, -1e-6, 0.2, 0.8),
            (BETA, f32::INFINITY, 0.2, 0.8),
            (BETA, TOL, 0.8, 0.2),
            (BETA, TOL, 0.5, 0.5),
            (BETA, TOL, 0.0, 0.8),
            (BETA, TOL, 0.2, 1.0),
            (BETA, TOL, f32::NAN, 0.8),
        ];
        for gamma in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.5,
            -f32::MIN_POSITIVE,
        ] {
            assert!(
                matches!(
                    graph
                        .evolve_signed_epistemic_fixed_point_within(BETA, gamma, TOL, 0.2, 0.8, 10),
                    Err(LodError::InvalidQuery(_))
                ),
                "gamma {gamma} was accepted"
            );
        }
        for (beta, tol, lo, hi) in bad {
            assert!(
                matches!(
                    graph.evolve_epistemic_fixed_point(beta, tol, lo, hi),
                    Err(LodError::InvalidQuery(_))
                ),
                "({beta}, {tol}, {lo}, {hi}) was accepted"
            );
        }
        // An empty graph is a fixed point already.
        let empty = LodGraph::new();
        let report = evolve(&empty, 0.2, 0.8);
        assert_eq!((report.nodes, report.iterations, report.k_max), (0, 0, 0));
    }

    /// Refute, evolve, retract, evolve: confidences return bit for bit, statuses,
    /// revocations and dependencies with them. A manual revocation stays.
    #[test]
    fn retracting_evidence_reverses_the_evolution_exactly() {
        let graph = LodGraph::new();
        let x = graph
            .add_node(node("x", 1).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        let a = graph.add_node(node("a", 2)).unwrap();
        let b = graph.add_node(node("b", 3)).unwrap();
        let c = graph.add_node(node("c", 4)).unwrap();
        let d = graph.add_node(node("d", 5)).unwrap();
        graph.add_edge(x, a, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        // Feedback loop b <-> c, and d hanging off c.
        graph.add_edge(b, c, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(c, b, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(c, d, EdgeType::DependsOn, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();
        // Refuted a gives b = 0.167, c = 0.217, d = 0.260: all below 0.3.
        let (lo, hi) = (0.3, 0.6);

        evolve(&graph, lo, hi);
        let supported = (confidences(&graph), sorted_deps(&graph));
        for id in [a, b, c, d] {
            assert_eq!(status(&graph, id), EpistemicStatus::Validated, "node {id}");
        }
        assert_eq!(supported.1.len(), 5);

        graph.falsify_node(a).unwrap();
        graph.revoke_entity(5).unwrap();
        let down = evolve(&graph, lo, hi);
        assert_eq!(down.transitions.len(), 3);
        for id in [a, b, c, d] {
            assert_eq!(status(&graph, id), EpistemicStatus::Falsified, "node {id}");
            assert!(graph.is_revoked(u64::from(id) + 1));
        }
        assert!(confidences(&graph)[1..].iter().all(|&v| v < lo));
        assert!(sorted_deps(&graph).is_empty());

        graph.retract_falsification(a).unwrap();
        assert_eq!(status(&graph, a), EpistemicStatus::Hypothesized);
        let up = evolve(&graph, lo, hi);
        assert_eq!(up.transitions.len(), 4);
        assert_eq!(up.reinstated_entities, vec![3, 4]);
        assert_eq!((confidences(&graph), sorted_deps(&graph)), supported);
        for id in [a, b, c, d] {
            assert_eq!(status(&graph, id), EpistemicStatus::Validated, "node {id}");
        }
        assert!(!graph.is_revoked(2) && !graph.is_revoked(3) && !graph.is_revoked(4));
        // `revoke_entity` was not evidence about confidence; it is not lifted.
        assert!(graph.is_revoked(5));
    }

    /// Between the thresholds a status is kept, in both directions.
    #[test]
    fn hysteresis_band_keeps_the_status() {
        let graph = LodGraph::new();
        let root = graph.add_node(node("root", 1).with_prior(0.9)).unwrap();
        let leaf = graph.add_node(node("leaf", 2).with_prior(0.5)).unwrap();
        graph
            .add_edge(root, leaf, EdgeType::DependsOn, 1.0)
            .unwrap();
        // leaf = 0.075 + 0.85 * 0.9 = 0.84: validated. root = 0.9: validated.
        evolve(&graph, 0.2, 0.8);
        assert_eq!(status(&graph, leaf), EpistemicStatus::Validated);

        graph.falsify_node(root).unwrap();
        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!(report.revoked_entities, vec![2]);
        assert_eq!(status(&graph, leaf), EpistemicStatus::Falsified);

        // Root back at its prior: leaf returns to 0.84 and is validated again
        // under theta_hi = 0.8, but stays falsified and revoked under 0.9.
        graph.retract_falsification(root).unwrap();
        let strict = evolve(&graph, 0.2, 0.9);
        assert_eq!(strict.transitions.len(), 0);
        assert_eq!(status(&graph, root), EpistemicStatus::Hypothesized);
        assert_eq!(status(&graph, leaf), EpistemicStatus::Falsified);
        assert!(graph.is_revoked(2));
        assert!((graph.get_node(leaf).unwrap().confidence - 0.84).abs() < 1e-5);
        let loose = evolve(&graph, 0.2, 0.8);
        assert_eq!(loose.reinstated_entities, vec![2]);
        assert_eq!(status(&graph, leaf), EpistemicStatus::Validated);
        assert!(!graph.is_revoked(2));
    }

    /// Edge weights decide: losing the heavy dependency falsifies the node,
    /// losing the light one does not. An unweighted traversal cannot tell them apart.
    #[test]
    fn edge_weights_decide_which_refutation_propagates() {
        let build = || {
            let graph = LodGraph::new();
            let heavy = graph.add_node(node("heavy", 1).with_prior(0.9)).unwrap();
            let light = graph.add_node(node("light", 2).with_prior(0.9)).unwrap();
            let leaf = graph.add_node(node("leaf", 3)).unwrap();
            graph
                .add_edge(heavy, leaf, EdgeType::DependsOn, 9.0)
                .unwrap();
            graph
                .add_edge(light, leaf, EdgeType::DependsOn, 1.0)
                .unwrap();
            evolve(&graph, 0.2, 0.8);
            assert_eq!(status(&graph, leaf), EpistemicStatus::Validated);
            (graph, heavy, light, leaf)
        };
        // leaf = 0.075 + 0.85 * (0.9 * 0 + 0.1 * 0.9) = 0.1515.
        let (graph, heavy, _, leaf) = build();
        graph.falsify_node(heavy).unwrap();
        evolve(&graph, 0.2, 0.8);
        assert_eq!(status(&graph, leaf), EpistemicStatus::Falsified);
        assert!((graph.get_node(leaf).unwrap().confidence - 0.1515).abs() < 1e-5);
        // leaf = 0.075 + 0.85 * (0.9 * 0.9 + 0.1 * 0) = 0.7635.
        let (graph, _, light, leaf) = build();
        graph.falsify_node(light).unwrap();
        evolve(&graph, 0.2, 0.8);
        assert_eq!(status(&graph, leaf), EpistemicStatus::Validated);
        assert!((graph.get_node(leaf).unwrap().confidence - 0.7635).abs() < 1e-5);
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

    /// Three nodes, each one factor away from the query, with one fingerprint so
    /// stage 1 cannot rank them. Returns the recall order by label and distance.
    fn recall_order(params: GeometryParams) -> Vec<(String, f32)> {
        let graph = LodGraph::with_geometry(params).unwrap();
        assert_eq!(graph.geometry(), params);
        let mut hyperbolic = MixedCurvatureCoord::origin();
        hyperbolic.hyperbolic[0] = 0.4;
        let mut euclidean = MixedCurvatureCoord::origin();
        euclidean.euclidean[0] = 1.0;
        let mut spherical = MixedCurvatureCoord::origin();
        // 0.6 rad from the north pole.
        spherical.spherical = [0.6_f32.cos(), 0.6_f32.sin(), 0.0, 0.0];
        for (i, (label, coord)) in [("H", hyperbolic), ("E", euclidean), ("S", spherical)]
            .into_iter()
            .enumerate()
        {
            graph.add_node(node_with(label, i as u64, coord)).unwrap();
        }
        graph
            .two_stage_recall(&MixedCurvatureCoord::origin(), &[0; 4], 3, 0.0)
            .unwrap()
            .into_iter()
            .map(|(id, d)| (graph.get_node(id).unwrap().label, d))
            .collect()
    }

    fn labels(order: &[(String, f32)]) -> String {
        order.iter().map(|(l, _)| l.as_str()).collect()
    }

    /// The knobs are live: each of alpha_h, alpha_e, alpha_s, c and R reorders
    /// the recall, and the distances are the ones the metric defines.
    #[test]
    fn geometry_parameters_reorder_two_stage_recall() {
        // Unit geometry: d_S = 0.6, d_H = 2 artanh(0.4) = 0.8473, d_E = 1.
        let unit = recall_order(GeometryParams::UNIT);
        assert_eq!(labels(&unit), "SHE");
        let d_h = 2.0 * 0.4_f32.atanh();
        for ((_, got), want) in unit.iter().zip([0.6, d_h, 1.0]) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }

        // Each weight alone moves its factor from first or second place to last.
        let by_alpha_h = recall_order(geometry(1.0, 1.0, [16.0, 1.0, 1.0]));
        assert_eq!(labels(&by_alpha_h), "SEH");
        assert!((by_alpha_h[2].1 - 4.0 * d_h).abs() < 1e-5);
        let by_alpha_s = recall_order(geometry(1.0, 1.0, [1.0, 1.0, 16.0]));
        assert_eq!(labels(&by_alpha_s), "HES");
        assert!((by_alpha_s[2].1 - 2.4).abs() < 1e-5);
        // And alpha_e pulls the Euclidean node from last to first.
        let by_alpha_e = recall_order(geometry(1.0, 1.0, [1.0, 0.0625, 1.0]));
        assert_eq!(labels(&by_alpha_e), "ESH");
        assert!((by_alpha_e[0].1 - 0.25).abs() < 1e-5);

        // Radius: d_S = R * 0.6 = 1.8 sends the sphere node last.
        let by_radius = recall_order(geometry(1.0, 3.0, [1.0, 1.0, 1.0]));
        assert_eq!(labels(&by_radius), "HES");
        assert!((by_radius[2].1 - 1.8).abs() < 1e-5);
        // Curvature 4: d_H = artanh(0.8) = 1.0986 sends the hyperbolic node last.
        let by_curvature = recall_order(geometry(4.0, 1.0, [1.0, 1.0, 1.0]));
        assert_eq!(labels(&by_curvature), "SEH");
        assert!((by_curvature[2].1 - 0.8_f32.atanh()).abs() < 1e-5);
    }

    /// Curvature fixes the domain: the same coordinate is a node of one graph
    /// and refused by another, at insert and at query.
    #[test]
    fn graph_curvature_decides_which_coordinates_are_in_domain() {
        let mut coord = MixedCurvatureCoord::origin();
        coord.hyperbolic[0] = 0.8;
        let origin = MixedCurvatureCoord::origin();
        let unit = LodGraph::new();
        unit.add_node(node_with("in", 1, coord)).unwrap();
        assert!(unit.two_stage_recall(&coord, &[0; 4], 1, 0.0).is_ok());

        // Curvature 4 is a ball of radius 0.5.
        let tight = LodGraph::with_geometry(geometry(4.0, 1.0, [1.0; 3])).unwrap();
        assert_eq!(
            tight.add_node(node_with("out", 1, coord)).unwrap_err(),
            LodError::Geometry(crate::manifold::Reject::DomainViolation)
        );
        tight.add_node(node_with("origin", 2, origin)).unwrap();
        assert!(tight.two_stage_recall(&coord, &[0; 4], 1, 0.0).is_err());

        // Curvature 0.25 is a ball of radius 2: a norm of 1.5 is a valid node.
        coord.hyperbolic[0] = 1.5;
        let wide = LodGraph::with_geometry(geometry(0.25, 1.0, [1.0; 3])).unwrap();
        wide.add_node(node_with("far", 1, coord)).unwrap();
        assert!(unit.add_node(node_with("far", 3, coord)).is_err());

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-60, 1e60] {
            for params in [
                geometry(bad, 1.0, [1.0; 3]),
                geometry(1.0, bad, [1.0; 3]),
                geometry(1.0, 1.0, [bad, 1.0, 1.0]),
                geometry(1.0, 1.0, [1.0, bad, 1.0]),
                geometry(1.0, 1.0, [1.0, 1.0, bad]),
            ] {
                assert!(
                    LodGraph::with_geometry(params).is_err(),
                    "{params:?} accepted"
                );
            }
        }
    }

    #[test]
    fn add_node_refuses_inconsistent_confidence_fields() {
        let graph = LodGraph::new();
        let mut split = node("split", 1);
        split.confidence = 0.9;
        assert!(matches!(
            graph.add_node(split),
            Err(LodError::InvalidNode(_))
        ));
        for prior in [-0.1, 1.1, f32::NAN] {
            assert!(matches!(
                graph.add_node(node("p", 2).with_prior(prior)),
                Err(LodError::InvalidNode(_))
            ));
        }
        let mut marked = node("marked", 3);
        marked.refuted = true;
        assert!(matches!(
            graph.add_node(marked),
            Err(LodError::InvalidNode(_))
        ));
        assert_eq!(graph.node_count(), 0);
        // An inserted Falsified node is refuted evidence: pinned, revoked, retractable.
        let f = graph
            .add_node(node("f", 4).with_status(EpistemicStatus::Falsified))
            .unwrap();
        let n = graph.get_node(f).unwrap();
        assert!(n.refuted && n.confidence == 0.0 && n.prior == 0.5);
        assert!(graph.is_revoked(4));
        graph.retract_falsification(f).unwrap();
        assert!(!graph.is_revoked(4));
        assert_eq!(graph.get_node(f).unwrap().confidence, 0.5);
    }

    #[test]
    fn test_graph_ppr_query() {
        let graph = LodGraph::new();
        let id0 = graph.add_node(node("root", 1)).unwrap();
        let id1 = graph.add_node(node("child_a", 2)).unwrap();
        let id2 = graph.add_node(node("child_b", 3)).unwrap();

        graph
            .add_edge(id0, id1, EdgeType::CausalTransition, 1.0)
            .unwrap();
        graph
            .add_edge(id1, id2, EdgeType::CausalTransition, 1.0)
            .unwrap();
        graph.flush_edges_to_csr().unwrap();

        let ppr = graph.query_ppr(&[(id0, 1.0)], 0.15, 200, 1e-5).unwrap();
        assert_eq!(ppr.ranked.len(), 3);
        assert_eq!(ppr.ranked[0].0, id0);
        assert!(ppr.converged);
    }

    #[test]
    fn ppr_covers_nodes_added_after_the_last_flush() {
        let graph = LodGraph::new();
        let a = graph.add_node(node("a", 1)).unwrap();
        let b = graph.add_node(node("b", 2)).unwrap();
        graph.add_edge(a, b, EdgeType::Semantic, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();
        let late = graph.add_node(node("late", 3)).unwrap();
        assert_eq!(graph.csr_snapshot().num_nodes(), 2);
        let ppr = graph.query_ppr(&[(late, 1.0)], 0.15, 20, 1e-6).unwrap();
        assert_eq!(ppr.ranked[0], (late, 1.0));
    }

    #[test]
    fn csr_from_edges_keeps_invariants_and_refuses_out_of_range() {
        let g = CsrGraph::default();
        assert_eq!(g.num_nodes(), 0);
        assert_eq!(g.row_ptrs(), &[0]);

        let edge = |source, target| BufferedEdge {
            source,
            target,
            edge_type: EdgeType::Semantic,
            weight: 0.5,
            ticket: 1,
        };
        let g = CsrGraph::from_edges(2, &[edge(0, 1)]).unwrap();
        let g = g.merged(4, &[]).unwrap();
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.row_ptrs(), &[0, 1, 1, 1, 1]);
        assert_eq!(g.col_indices(), &[1]);
        assert_eq!(g.edge_weights(), &[0.5]);
        assert_eq!(g.edge_types(), &[EdgeType::Semantic]);
        assert_eq!(g.neighbors(3).count(), 0);
        assert!(matches!(
            CsrGraph::from_edges(2, &[edge(0, 5)]),
            Err(LodError::InvalidEdge(_))
        ));
        assert!(matches!(g.merged(3, &[]), Err(LodError::CsrInvariant(_))));
    }

    /// The incremental merge must equal a from-scratch build of all edges.
    #[test]
    fn incremental_merge_equals_full_rebuild() {
        let edge = |source, target, ticket| BufferedEdge {
            source,
            target,
            edge_type: EdgeType::DependsOn,
            weight: ticket as f32,
            ticket,
        };
        let first = [edge(0, 1, 1), edge(2, 0, 2), edge(0, 2, 3)];
        let second = [edge(1, 2, 4), edge(0, 3, 5), edge(3, 3, 6)];
        let incremental = CsrGraph::from_edges(3, &first)
            .unwrap()
            .merged(4, &second)
            .unwrap();
        let all: Vec<_> = first.iter().chain(&second).copied().collect();
        let full = CsrGraph::from_edges(4, &all).unwrap();
        assert_eq!(incremental.row_ptrs(), full.row_ptrs());
        assert_eq!(incremental.col_indices(), full.col_indices());
        assert_eq!(incremental.edge_weights(), full.edge_weights());
        assert_eq!(incremental.edge_types(), full.edge_types());
    }

    /// Flush drains the buffer: its length returns to zero after every flush
    /// instead of growing with the whole edge history.
    #[test]
    fn flush_drains_the_buffer_and_keeps_every_edge_once() {
        let graph = LodGraph::new();
        let ids: Vec<u32> = (0..10)
            .map(|i| graph.add_node(node("n", 100 + i)).unwrap())
            .collect();
        for round in 0..5u32 {
            for k in 0..20u32 {
                let (s, t) = (ids[(k % 10) as usize], ids[((k + round) % 10) as usize]);
                graph.add_edge(s, t, EdgeType::Semantic, 1.0).unwrap();
            }
            assert_eq!(graph.pending_edge_count(), 20);
            let report = graph.flush_edges_to_csr().unwrap();
            assert_eq!(report.merged_edges, 20);
            assert_eq!(report.pending_edges, 0);
            assert_eq!(graph.pending_edge_count(), 0);
            assert_eq!(report.csr_edges, 20 * (round as usize + 1));
        }
        let empty = graph.flush_edges_to_csr().unwrap();
        assert_eq!(empty.merged_edges, 0);
        assert_eq!(empty.csr_edges, 100);
    }

    #[test]
    fn rollback_restores_nodes_edges_csr_and_revocations() {
        let graph = LodGraph::new();
        let a = graph.add_node(node("a", 1)).unwrap();
        let b = graph.add_node(node("b", 2)).unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();
        graph.add_edge(b, a, EdgeType::Semantic, 1.0).unwrap();
        let checkpoint = graph.create_checkpoint();
        let csr_before = graph.csr_snapshot();

        let c = graph.add_node(node("c", 3)).unwrap();
        graph.add_edge(a, c, EdgeType::DependsOn, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();
        graph.falsify_node(a).unwrap();
        evolve(&graph, 0.2, 0.8);
        graph.add_privilege(9, 1).unwrap();
        assert!(graph.is_revoked(1) && graph.is_revoked(2) && graph.is_revoked(3));

        graph.rollback_checkpoint(&checkpoint).unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.node_for_entity(3), None);
        assert_eq!(graph.pending_edge_count(), 1);
        assert!(Arc::ptr_eq(&graph.csr_snapshot(), &csr_before));
        assert!(!graph.is_revoked(1) && !graph.is_revoked(2));
        assert!(!graph.has_privilege(9, 1));
        let restored = graph.get_node(a).unwrap();
        assert_eq!(restored.status, EpistemicStatus::Hypothesized);
        assert!(!restored.refuted && restored.confidence == 0.5);
        assert_eq!(graph.get_node(b).unwrap().confidence, 0.5);
        // Entity 3 is free again, and the restored pending edge flushes once.
        graph.add_node(node("c2", 3)).unwrap();
        assert_eq!(graph.flush_edges_to_csr().unwrap().csr_edges, 2);
    }

    #[test]
    fn report_checkpoint_undoes_the_evolution_and_an_earlier_one_the_evidence() {
        let graph = LodGraph::new();
        let a = graph
            .add_node(node("a", 1).with_status(EpistemicStatus::Validated))
            .unwrap();
        let b = graph
            .add_node(node("b", 2).with_status(EpistemicStatus::Validated))
            .unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        assert_eq!(sorted_deps(&graph), vec![(1, 2)]);
        let before = graph.create_checkpoint();

        graph.falsify_node(a).unwrap();
        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!(report.revoked_entities, vec![2]);
        assert!(graph.is_revoked(1) && graph.is_revoked(2));
        assert!(sorted_deps(&graph).is_empty());

        // The report's checkpoint is the state the evolution started from.
        graph.rollback_checkpoint(&report.checkpoint).unwrap();
        assert!(graph.is_revoked(1) && !graph.is_revoked(2));
        assert_eq!(status(&graph, b), EpistemicStatus::Validated);
        assert_eq!(graph.get_node(b).unwrap().confidence, 0.5);
        assert!(graph.get_node(a).unwrap().refuted);

        graph.rollback_checkpoint(&before).unwrap();
        assert!(!graph.is_revoked(1) && !graph.is_revoked(2));
        let restored = graph.get_node(a).unwrap();
        assert_eq!(restored.status, EpistemicStatus::Validated);
        assert!(!restored.refuted);
        assert_eq!(sorted_deps(&graph), vec![(1, 2)]);
    }

    #[test]
    fn rollback_refuses_foreign_and_discarded_checkpoints() {
        let graph = LodGraph::new();
        let other = LodGraph::new();
        assert!(matches!(
            graph.rollback_checkpoint(&other.create_checkpoint()),
            Err(LodError::CheckpointRejected(_))
        ));

        let early = graph.create_checkpoint();
        graph.add_node(node("a", 1)).unwrap();
        let late = graph.create_checkpoint();
        graph.rollback_checkpoint(&early).unwrap();
        // `late` describes a state that rollback discarded, even once the node
        // count grows back.
        graph.add_node(node("b", 2)).unwrap();
        assert!(matches!(
            graph.rollback_checkpoint(&late),
            Err(LodError::CheckpointRejected(_))
        ));
        // The earlier checkpoint is still an ancestor and stays valid.
        graph.rollback_checkpoint(&early).unwrap();
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn transact_rolls_back_a_failed_write() {
        let graph = LodGraph::new();
        graph.add_node(node("keep", 1)).unwrap();
        let err = graph
            .transact(|g| {
                let a = g.add_node(node("tmp", 2))?;
                g.add_edge(a, 0, EdgeType::Semantic, 1.0)?;
                g.flush_edges_to_csr()?;
                g.add_edge(a, 42, EdgeType::Semantic, 1.0)
            })
            .unwrap_err();
        assert!(matches!(err, LodError::InvalidEdge(_)));
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.node_for_entity(2), None);
        assert_eq!(graph.csr_snapshot().num_edges(), 0);
        assert_eq!(graph.pending_edge_count(), 0);
    }

    /// Writers, flushes and transactional rollbacks run concurrently. Every
    /// surviving edge must be exactly once in CSR or buffer, never both.
    #[test]
    fn concurrent_add_flush_and_rollback_never_duplicate_edges() {
        let graph = Arc::new(LodGraph::new());
        for i in 0..8 {
            graph.add_node(node("n", i)).unwrap();
        }
        let mut handles = Vec::new();
        for t in 0..4u32 {
            let g = Arc::clone(&graph);
            handles.push(std::thread::spawn(move || {
                for k in 0..200u32 {
                    let _ = g.transact(|g| {
                        g.add_edge(t, k % 8, EdgeType::Semantic, 1.0)?;
                        if k % 3 == 0 {
                            return Err(LodError::InvalidQuery("forced rollback".into()));
                        }
                        Ok(())
                    });
                }
            }));
        }
        let g = Arc::clone(&graph);
        handles.push(std::thread::spawn(move || {
            for _ in 0..200 {
                match g.flush_edges_to_csr() {
                    Ok(_) | Err(LodError::FlushConflict) => {}
                    Err(e) => panic!("flush failed: {e}"),
                }
            }
        }));
        for h in handles {
            h.join().unwrap();
        }
        graph.flush_edges_to_csr().unwrap();
        assert_eq!(graph.pending_edge_count(), 0);
        // 4 writers x 200 attempts, 67 of each forced to roll back.
        assert_eq!(graph.csr_snapshot().num_edges(), 4 * (200 - 67));
        graph.csr_snapshot().validate().unwrap();
    }

    /// A unit-ball coordinate at normalized depth `rho` along axis `axis`.
    fn at_depth(rho: f64, axis: usize) -> MixedCurvatureCoord {
        let mut h = [0.0_f32; 4];
        h[axis] = (rho / 2.0).tanh() as f32;
        MixedCurvatureCoord::new(h, [1.0, 0.0, 0.0, 0.0], [0.0; 8]).unwrap()
    }

    /// Depth at the center of `band` on the unit ball.
    fn band_depth(band: LodBand) -> f64 {
        let w = crate::node::band_scale_width();
        crate::node::max_chart_depth() - (band as u8 as f64 + 0.5) * w
    }

    /// Three atomic members near the boundary; returns their ids.
    fn atomic_members(graph: &LodGraph) -> Vec<u32> {
        (0..3)
            .map(|k| {
                let coord = at_depth(band_depth(LodBand::Lod0Atomic), k);
                graph
                    .add_node(LodNode::new(
                        0,
                        LodBand::Lod0Atomic,
                        coord,
                        "m",
                        100 + k as u64,
                    ))
                    .unwrap()
            })
            .collect()
    }

    #[test]
    fn coarse_grain_cluster_links_members_to_a_coarser_summary_in_the_csr() {
        let graph = LodGraph::new();
        let members = atomic_members(&graph);
        let summary_coord = at_depth(band_depth(LodBand::Lod1Cluster), 0);
        let summary = graph
            .coarse_grain_cluster(&members, 900, summary_coord, [7; 4])
            .unwrap();

        let node = graph.get_node(summary).unwrap();
        assert_eq!(node.band, LodBand::Lod1Cluster);
        assert_eq!(node.entity_id, 900);
        assert_eq!(node.hdc_fingerprint, [7; 4]);
        assert_eq!(graph.pending_edge_count(), 0, "the cluster is flushed");
        let csr = graph.csr_snapshot();
        csr.validate().unwrap();
        for &m in &members {
            let links: Vec<_> = csr.neighbors(m).collect();
            assert_eq!(links, vec![(summary, EdgeType::CoarseGrain, 1.0)]);
            assert_eq!(graph.get_node(m).unwrap().parent_id, Some(summary));
        }
        // Diffusion from one member reaches the summary through the new edge.
        let ranking = graph
            .query_ppr(&[(members[0], 1.0)], 0.15, 100, 1e-6)
            .unwrap();
        let score = |id| ranking.ranked.iter().find(|r| r.0 == id).unwrap().1;
        assert!(score(summary) > 0.0);
        assert_eq!(score(members[1]), 0.0);

        // A second level: the Lod1 summary coarse-grains into a Lod3 root.
        let root = graph
            .coarse_grain_cluster(&[summary], 901, MixedCurvatureCoord::origin(), [0; 4])
            .unwrap();
        assert_eq!(graph.get_node(root).unwrap().band, LodBand::Lod3Systemic);
        assert_eq!(graph.csr_snapshot().num_edges(), 4);
    }

    #[test]
    fn coarse_grain_cluster_refuses_and_changes_nothing() {
        let graph = LodGraph::new();
        let members = atomic_members(&graph);
        let coarse = at_depth(band_depth(LodBand::Lod2Milestone), 1);
        let before = (graph.node_count(), graph.csr_snapshot().num_edges());
        let refusals = [
            graph.coarse_grain_cluster(&[], 900, coarse, [0; 4]),
            graph.coarse_grain_cluster(&[members[0], members[0]], 900, coarse, [0; 4]),
            graph.coarse_grain_cluster(&[members[0], 77], 900, coarse, [0; 4]),
            // Summary no coarser than its members.
            graph.coarse_grain_cluster(
                &members,
                900,
                at_depth(band_depth(LodBand::Lod0Atomic), 3),
                [0; 4],
            ),
            // Entity already has a node.
            graph.coarse_grain_cluster(&members, 100, coarse, [0; 4]),
        ];
        for r in &refusals {
            assert!(r.is_err(), "{r:?}");
        }
        graph.falsify_node(members[2]).unwrap();
        assert!(matches!(
            graph.coarse_grain_cluster(&members, 900, coarse, [0; 4]),
            Err(LodError::InvalidNode(_))
        ));
        assert_eq!(
            (graph.node_count(), graph.csr_snapshot().num_edges()),
            before
        );
        assert_eq!(graph.pending_edge_count(), 0);
        assert!(members
            .iter()
            .all(|&m| graph.get_node(m).unwrap().parent_id.is_none()));

        // A member already in a cluster cannot join a second one.
        let s = graph
            .coarse_grain_cluster(&members[..2], 900, coarse, [0; 4])
            .unwrap();
        assert!(graph.get_node(s).is_some());
        assert!(matches!(
            graph.coarse_grain_cluster(&members[..1], 901, MixedCurvatureCoord::origin(), [0; 4]),
            Err(LodError::InvalidNode(_))
        ));
    }

    #[test]
    fn coarse_grain_edges_must_point_to_a_coarser_band() {
        let graph = LodGraph::new();
        let fine = graph.add_node(node("fine", 1)).unwrap();
        let coarse = graph
            .add_node(LodNode::new(
                0,
                LodBand::Lod2Milestone,
                MixedCurvatureCoord::origin(),
                "coarse",
                2,
            ))
            .unwrap();
        let peer = graph.add_node(node("peer", 3)).unwrap();
        assert!(graph
            .add_edge(coarse, fine, EdgeType::CoarseGrain, 1.0)
            .is_err());
        assert!(graph
            .add_edge(fine, peer, EdgeType::CoarseGrain, 1.0)
            .is_err());
        assert!(graph
            .add_edge(fine, coarse, EdgeType::CoarseGrain, 1.0)
            .is_ok());
        // Other edge types carry no band order.
        assert!(graph
            .add_edge(coarse, fine, EdgeType::Semantic, 1.0)
            .is_ok());
    }

    #[test]
    fn zoom_node_keeps_the_coarse_grain_order_and_rolls_back() {
        let graph = LodGraph::new();
        let members = atomic_members(&graph);
        let summary = graph
            .coarse_grain_cluster(
                &members,
                900,
                at_depth(band_depth(LodBand::Lod1Cluster), 0),
                [0; 4],
            )
            .unwrap();

        // Members may not reach their summary's band; the summary may not sink to theirs.
        assert!(matches!(
            graph.zoom_node(members[0], ZoomDirection::Out),
            Err(LodError::InvalidStateTransition(_))
        ));
        assert!(matches!(
            graph.zoom_node(summary, ZoomDirection::In),
            Err(LodError::InvalidStateTransition(_))
        ));
        assert!(matches!(
            graph.zoom_node(members[0], ZoomDirection::In),
            Err(LodError::SpineBreatheOutOfBounds { .. })
        ));
        assert!(matches!(
            graph.zoom_node(99, ZoomDirection::In),
            Err(LodError::NodeNotFound(99))
        ));

        let checkpoint = graph.create_checkpoint();
        assert_eq!(
            graph.zoom_node(summary, ZoomDirection::Out).unwrap(),
            LodBand::Lod2Milestone
        );
        assert_eq!(
            graph.zoom_node(members[0], ZoomDirection::Out).unwrap(),
            LodBand::Lod1Cluster
        );
        assert_eq!(
            graph.zoom_node(summary, ZoomDirection::Out).unwrap(),
            LodBand::Lod3Systemic
        );
        assert!(graph.zoom_node(summary, ZoomDirection::Out).is_err());

        graph.rollback_checkpoint(&checkpoint).unwrap();
        assert_eq!(graph.get_node(summary).unwrap().band, LodBand::Lod1Cluster);
        assert_eq!(
            graph.get_node(members[0]).unwrap().band,
            LodBand::Lod0Atomic
        );

        // Rolling back past the cluster also restores the members' parents.
        let fresh = LodGraph::new();
        let members = atomic_members(&fresh);
        let before = fresh.create_checkpoint();
        fresh
            .coarse_grain_cluster(&members, 900, MixedCurvatureCoord::origin(), [0; 4])
            .unwrap();
        fresh.rollback_checkpoint(&before).unwrap();
        assert_eq!(fresh.node_count(), 3);
        assert_eq!(fresh.csr_snapshot().num_edges(), 0);
        assert!(members
            .iter()
            .all(|&m| fresh.get_node(m).unwrap().parent_id.is_none()));
    }

    #[test]
    fn migrate_band_to_coord_steps_to_the_coordinate_band() {
        let graph = LodGraph::new();
        // Inserted at Lod0 with a coordinate at the origin, which implies Lod3.
        let id = graph.add_node(node("root", 1)).unwrap();
        assert_eq!(
            graph.migrate_band_to_coord(id).unwrap(),
            (LodBand::Lod0Atomic, LodBand::Lod3Systemic)
        );
        assert_eq!(
            graph.migrate_band_to_coord(id).unwrap(),
            (LodBand::Lod3Systemic, LodBand::Lod3Systemic)
        );

        // A member stored at Lod0 whose coordinate implies Lod3 cannot migrate
        // to or above its Lod1 summary; the refusal changes nothing.
        let member = graph.add_node(node("member", 2)).unwrap();
        let summary = graph
            .coarse_grain_cluster(
                &[member],
                900,
                at_depth(band_depth(LodBand::Lod1Cluster), 0),
                [0; 4],
            )
            .unwrap();
        assert!(matches!(
            graph.migrate_band_to_coord(member),
            Err(LodError::InvalidStateTransition(_))
        ));
        assert_eq!(graph.get_node(member).unwrap().band, LodBand::Lod0Atomic);
        assert_eq!(
            graph.migrate_band_to_coord(summary).unwrap(),
            (LodBand::Lod1Cluster, LodBand::Lod1Cluster)
        );
    }

    #[test]
    fn a_summary_loses_confidence_when_its_members_are_refuted() {
        let graph = LodGraph::new();
        let members = atomic_members(&graph);
        let summary = graph
            .coarse_grain_cluster(&members, 900, MixedCurvatureCoord::origin(), [0; 4])
            .unwrap();
        let before = evolve(&graph, 0.2, 0.8);
        assert_eq!(before.dependency_edges, 3);
        let c0 = graph.get_node(summary).unwrap().confidence;
        assert!((c0 - 0.5).abs() < 1e-5);
        for &m in &members {
            graph.falsify_node(m).unwrap();
        }
        evolve(&graph, 0.2, 0.8);
        let summary_node = graph.get_node(summary).unwrap();
        // c = (1 - beta) 0.5 + beta * 0 = 0.075, below theta_lo.
        assert!((summary_node.confidence - 0.075).abs() < 1e-5);
        assert_eq!(summary_node.status, EpistemicStatus::Falsified);
    }

    /// A text node through the graph's own projector, carrying its text.
    fn text_node(graph: &LodGraph, text: &str, entity: u64) -> LodNode {
        let (coord, hdc) = graph.project_text(text).unwrap();
        LodNode::new(0, LodBand::Lod0Atomic, coord, text, entity)
            .with_hdc_fingerprint(hdc)
            .with_payload(text, Some(format!("doc://kb/{entity}")), 1_000 + entity)
            .unwrap()
    }

    const PUMP: &str = "the reactor coolant pump failed during the night shift";
    const LOG: &str = "maintenance ticket 4411 replaced seal kit on unit two";
    const BUDGET: &str = "quarterly marketing budget for the new espresso brand";

    #[test]
    fn hybrid_text_search_returns_anchor_payload_and_diffused_neighbor() {
        let graph = LodGraph::new();
        let pump = graph.add_node(text_node(&graph, PUMP, 1)).unwrap();
        let log = graph.add_node(text_node(&graph, LOG, 2)).unwrap();
        let budget = graph.add_node(text_node(&graph, BUDGET, 3)).unwrap();
        // Same text as the query, but refuted: must never come back.
        let refuted = graph
            .add_node(
                text_node(&graph, "coolant pump failed during night shift", 4)
                    .with_status(EpistemicStatus::Falsified),
            )
            .unwrap();
        graph
            .add_edge(pump, log, EdgeType::CausalTransition, 1.0)
            .unwrap();
        graph
            .add_edge(refuted, budget, EdgeType::Semantic, 1.0)
            .unwrap();
        graph.flush_edges_to_csr().unwrap();

        let result = graph
            .hybrid_rag_search_query(
                Some("coolant pump failed during night shift"),
                None,
                1,
                0.0,
                0.15,
                200,
            )
            .unwrap();
        assert_eq!(result.anchors.len(), 1);
        assert_eq!(result.anchors[0].0, pump);
        assert_eq!(result.stage1_candidates, 3);
        let ids: Vec<u32> = result.hits.iter().map(|h| h.node_id).collect();
        // The budget node is live but no anchor reaches it: score 0, left out.
        assert_eq!(ids, vec![pump, log], "{result:?}");
        let top = &result.hits[0];
        assert_eq!(top.payload.as_deref(), Some(PUMP));
        assert_eq!(top.source_uri.as_deref(), Some("doc://kb/1"));
        assert_eq!(top.timestamp_ns, 1_001);
        assert_eq!(
            top.payload_digest,
            *blake3::hash(PUMP.as_bytes()).as_bytes()
        );
        assert!(top.anchor_distance.is_some());
        let neighbor = &result.hits[1];
        assert_eq!(neighbor.anchor_distance, None);
        assert_eq!(neighbor.payload.as_deref(), Some(LOG));
        assert!(neighbor.ppr_score > 0.0 && neighbor.ppr_score < top.ppr_score);
        let diffusion = result.diffusion.unwrap();
        assert!(diffusion.converged, "{diffusion:?}");
        assert_eq!(diffusion.tolerance, HYBRID_PPR_TOLERANCE);
    }

    #[test]
    fn hybrid_search_equals_recall_then_ppr_from_the_same_anchors() {
        let graph = LodGraph::new();
        let ids: Vec<u32> = [PUMP, LOG, BUDGET]
            .iter()
            .enumerate()
            .map(|(i, t)| graph.add_node(text_node(&graph, t, 10 + i as u64)).unwrap())
            .collect();
        graph
            .add_edge(ids[0], ids[1], EdgeType::DependsOn, 2.0)
            .unwrap();
        graph
            .add_edge(ids[1], ids[2], EdgeType::Semantic, 1.0)
            .unwrap();
        graph.flush_edges_to_csr().unwrap();
        let (coord, hdc) = graph.project_text("reactor pump seal failed").unwrap();

        let result = graph
            .hybrid_rag_search(&coord, &hdc, 2, 0.0, 0.2, 300)
            .unwrap();
        let recall = graph.two_stage_recall(&coord, &hdc, 2, 0.0).unwrap();
        let max_distance = recall.iter().map(|a| a.1).fold(0.0_f32, f32::max);
        assert!(max_distance > 0.0);
        let normalized: Vec<_> = recall
            .iter()
            .map(|&(id, d)| (id, d / max_distance))
            .collect();
        assert_eq!(result.anchors, normalized);
        let seeds: Vec<(u32, f32)> = normalized
            .iter()
            .map(|&(id, d)| (id, 1.0 / (1.0 + d)))
            .collect();
        let ppr = graph
            .query_ppr(&seeds, 0.2, 300, HYBRID_PPR_TOLERANCE)
            .unwrap();
        for hit in &result.hits {
            let expected = ppr.ranked.iter().find(|r| r.0 == hit.node_id).unwrap().1;
            assert_eq!(hit.ppr_score, expected);
        }
        assert!(result
            .hits
            .windows(2)
            .all(|w| w[0].ppr_score >= w[1].ppr_score));
    }

    #[test]
    fn hybrid_search_fails_closed_on_bad_input_and_is_empty_on_an_empty_graph() {
        let graph = LodGraph::new();
        let empty = graph
            .hybrid_rag_search_query(Some(PUMP), None, 3, 0.0, 0.15, 50)
            .unwrap();
        assert!(empty.hits.is_empty() && empty.anchors.is_empty());
        assert_eq!(empty.diffusion, None);

        graph.add_node(text_node(&graph, PUMP, 1)).unwrap();
        for text in ["", "   ", "?!"] {
            assert!(matches!(
                graph.hybrid_rag_search_query(Some(text), None, 3, 0.0, 0.15, 50),
                Err(LodError::EmptyInput(_))
            ));
        }
        for (alpha, iters) in [(0.0, 50), (1.0, 50), (f32::NAN, 50), (0.15, 0)] {
            assert!(matches!(
                graph.hybrid_rag_search_query(Some(PUMP), None, 3, 0.0, alpha, iters),
                Err(LodError::InvalidQuery(_))
            ));
        }
        assert!(graph
            .hybrid_rag_search_query(Some(PUMP), None, 0, 0.0, 0.15, 50)
            .is_err());
        assert!(graph
            .hybrid_rag_search_query(Some(PUMP), None, 1, -1.0, 0.15, 50)
            .is_err());
    }

    #[test]
    fn insert_refuses_a_payload_whose_digest_was_tampered() {
        let graph = LodGraph::new();
        let mut n = text_node(&graph, PUMP, 1);
        n.payload = Some("the pump is fine".into());
        assert!(matches!(
            graph.add_node(n),
            Err(LodError::InvalidPayload(_))
        ));
        assert_eq!(graph.node_count(), 0);
        let mut n = node("bare", 2);
        n.source_uri = Some("doc://x".into());
        assert!(graph.add_node(n).is_err());
    }

    #[test]
    fn rollback_drops_the_payload_nodes_it_removes() {
        let graph = LodGraph::new();
        graph.add_node(text_node(&graph, PUMP, 1)).unwrap();
        let checkpoint = graph.create_checkpoint();
        graph.add_node(text_node(&graph, LOG, 2)).unwrap();
        graph.rollback_checkpoint(&checkpoint).unwrap();
        let result = graph
            .hybrid_rag_search_query(Some(LOG), None, 2, 0.0, 0.15, 50)
            .unwrap();
        assert!(result
            .hits
            .iter()
            .all(|h| h.payload.as_deref() != Some(LOG)));
        assert_eq!(graph.node_for_entity(2), None);
    }

    // ------------------------------------------------- aliases and embeddings

    use crate::projection::test_vectors;

    /// Thirteen texts with no token of the queries below except where noted, so
    /// a `top_k` of 1 or 2 leaves most of them outside the Stage 1 pool.
    const DISTRACTORS: [&str; 13] = [
        // These two share a word with "valve closure".
        "the relief valve was replaced last week",
        "closure of the quarterly accounts is due friday",
        "the reactor coolant pump failed during the night shift",
        "maintenance ticket 4411 replaced seal kit on unit two",
        "quarterly marketing budget for the new espresso brand",
        "the night shift supervisor signed the handover log",
        "spare bearings are stored in warehouse three",
        "the canteen menu changes every monday",
        "calibration of the flow meter is overdue",
        "the forklift battery needs charging",
        "fire drill scheduled for the second floor",
        "new safety boots arrive next month",
        "the turbine hall lighting was upgraded",
    ];

    const CLOSE_MAIN_VALVE: &str = "关闭主阀";
    const HANDWHEEL: &str = "turn the handwheel clockwise until the stem stops";

    fn with_distractors() -> LodGraph {
        let graph = LodGraph::new();
        for (i, text) in DISTRACTORS.iter().enumerate() {
            graph
                .add_node(text_node(&graph, text, 100 + i as u64))
                .unwrap();
        }
        graph
    }

    fn hit_entities(result: &HybridRagResult) -> Vec<u64> {
        result.hits.iter().map(|h| h.entity_id).collect()
    }

    #[test]
    fn an_alias_recalls_a_translation_the_lexical_projection_misses() {
        let query = "valve closure";
        // Control: the Chinese node has no alias. It shares no n-gram with the
        // query, falls outside the Stage 1 pool of 4 and is not a hit.
        let control = with_distractors();
        control
            .add_node(text_node(&control, CLOSE_MAIN_VALVE, 1))
            .unwrap();
        let missed = control
            .hybrid_rag_search_query(Some(query), None, 1, 0.0, 0.15, 100)
            .unwrap();
        assert_eq!(missed.searchable_nodes, 14);
        assert_eq!(missed.stage1_candidates, 4);
        assert!(!hit_entities(&missed).contains(&1), "{missed:?}");
        // The lexical winner is a text that shares a word with the query.
        assert!([100, 101].contains(&missed.hits[0].entity_id), "{missed:?}");

        // Same graph, same query, the node now carries the English alias.
        let graph = with_distractors();
        let target = graph
            .add_node(
                text_node(&graph, CLOSE_MAIN_VALVE, 1).with_aliases(["主阀关断", "Valve closure"]),
            )
            .unwrap();
        let result = graph
            .hybrid_rag_search_query(Some(query), None, 1, 0.0, 0.15, 100)
            .unwrap();
        assert_eq!(result.anchors.len(), 1);
        assert_eq!(result.anchors[0].0, target);
        let top = &result.hits[0];
        assert_eq!(top.entity_id, 1);
        assert_eq!(top.anchor_match, Some(AnchorMatch::Alias(1)));
        assert!(top.anchor_distance.unwrap() < 1e-6);
        assert_eq!(top.payload.as_deref(), Some(CLOSE_MAIN_VALVE));
        assert_eq!(top.aliases, ["主阀关断", "Valve closure"]);
        // The node's own text still finds it, by its own coordinate.
        let own = graph
            .hybrid_rag_search_query(Some(CLOSE_MAIN_VALVE), None, 1, 0.0, 0.15, 100)
            .unwrap();
        assert_eq!(own.hits[0].anchor_match, Some(AnchorMatch::Primary));
        // Recall takes the same closest anchor.
        let (coord, hdc) = graph.project_text(query).unwrap();
        let recall = graph.two_stage_recall(&coord, &hdc, 1, 0.0).unwrap();
        assert_eq!(recall, result.anchors);
    }

    #[test]
    fn diffusion_recalls_synonyms_along_alias_links_and_validates_edges() {
        // Control: no shared alias, no edge. The Chinese query anchors its own
        // node and nothing else about the valve comes back.
        let control = with_distractors();
        control
            .add_node(text_node(&control, CLOSE_MAIN_VALVE, 1))
            .unwrap();
        control.add_node(text_node(&control, HANDWHEEL, 2)).unwrap();
        control.flush_edges_to_csr().unwrap();
        let missed = control
            .hybrid_rag_search_query(Some(CLOSE_MAIN_VALVE), None, 1, 0.0, 0.15, 200)
            .unwrap();
        assert_eq!(hit_entities(&missed), vec![1], "{missed:?}");

        let graph = with_distractors();
        let zh = graph
            .add_node(text_node(&graph, CLOSE_MAIN_VALVE, 1).with_aliases(["valve closure"]))
            .unwrap();
        assert_eq!(graph.pending_edge_count(), 0);
        // Same alias in another case and spacing: linked both ways at insert.
        let en = graph
            .add_node(text_node(&graph, HANDWHEEL, 2).with_aliases(["  Valve   CLOSURE "]))
            .unwrap();
        assert_eq!(graph.pending_edge_count(), 2);
        let proof = graph
            .add_node(text_node(
                &graph,
                "leak test passed at forty bar after isolation",
                3,
            ))
            .unwrap();
        graph.add_edge(zh, proof, EdgeType::Validates, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();
        let csr = graph.csr_snapshot();
        assert!(csr
            .neighbors(zh)
            .any(|(v, t, w)| v == en && t == EdgeType::Semantic && w == ALIAS_LINK_WEIGHT));
        assert!(csr
            .neighbors(en)
            .any(|(v, t, _)| v == zh && t == EdgeType::Semantic));

        let result = graph
            .hybrid_rag_search_query(Some(CLOSE_MAIN_VALVE), None, 1, 0.0, 0.15, 200)
            .unwrap();
        assert_eq!(result.anchors, vec![(zh, result.anchors[0].1)]);
        assert_eq!(result.stage1_candidates, 4);
        // top_k 1 admits one diffusion-only node; ask for 2 to see both.
        assert_eq!(result.hits.len(), 2);
        let result = graph
            .hybrid_rag_search_query(Some(CLOSE_MAIN_VALVE), None, 2, 0.0, 0.15, 200)
            .unwrap();
        let ids = hit_entities(&result);
        assert_eq!(ids[0], 1, "{result:?}");
        for synonym in [2, 3] {
            let hit = result
                .hits
                .iter()
                .find(|h| h.entity_id == synonym)
                .unwrap_or_else(|| panic!("entity {synonym} not recalled: {ids:?}"));
            assert_eq!(hit.anchor_distance, None);
            assert_eq!(hit.anchor_match, None);
            assert!(hit.ppr_score > 0.0);
        }
        assert!(result.diffusion.unwrap().converged);
    }

    #[test]
    fn alias_rules_are_enforced_and_nothing_is_inserted_on_refusal() {
        let graph = LodGraph::new();
        let base = || text_node(&graph, HANDWHEEL, 1);
        let too_many: Vec<String> = (0..=crate::node::MAX_ALIASES)
            .map(|i| format!("name {i}"))
            .collect();
        assert!(matches!(
            graph.add_node(base().with_aliases(too_many)),
            Err(LodError::InvalidNode(_))
        ));
        let long = "x".repeat(crate::node::MAX_ALIAS_BYTES + 1);
        assert!(matches!(
            graph.add_node(base().with_aliases([long])),
            Err(LodError::InvalidNode(_))
        ));
        for blank in ["", "  ", "?!"] {
            assert!(matches!(
                graph.add_node(base().with_aliases([blank])),
                Err(LodError::EmptyInput(_))
            ));
        }
        assert!(matches!(
            graph.add_node(base().with_aliases(["shut off", "Shut  OFF"])),
            Err(LodError::InvalidNode(_))
        ));
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.pending_edge_count(), 0);

        // Anchors a caller puts on the node are replaced by the graph's own.
        let mut forged = base().with_aliases(["shut off"]);
        forged.alias_anchors = vec![
            ChartAnchor {
                coord: MixedCurvatureCoord::origin(),
                hdc_fingerprint: [u64::MAX; 4],
            };
            3
        ];
        forged.embedding_anchor = forged.alias_anchors.first().copied();
        let id = graph.add_node(forged).unwrap();
        let stored = graph.get_node(id).unwrap();
        let (coord, hdc_fingerprint) = graph.project_text("shut off").unwrap();
        assert_eq!(
            stored.alias_anchors,
            vec![ChartAnchor {
                coord,
                hdc_fingerprint
            }]
        );
        assert_eq!(stored.embedding_anchor, None);
    }

    #[test]
    fn one_alias_links_at_most_the_holder_cap() {
        let graph = LodGraph::new();
        for i in 0..MAX_ALIAS_HOLDERS as u64 {
            graph
                .add_node(node("n", i).with_aliases(["shared tag"]))
                .unwrap();
        }
        // Holder k links to the k earlier ones, both ways.
        let n = MAX_ALIAS_HOLDERS;
        assert_eq!(graph.pending_edge_count(), n * (n - 1));
        assert!(matches!(
            graph.add_node(node("n", 999).with_aliases(["Shared Tag"])),
            Err(LodError::InvalidNode(_))
        ));
        assert_eq!(graph.node_count(), n);
        assert_eq!(graph.pending_edge_count(), n * (n - 1));
    }

    #[test]
    fn rollback_forgets_aliases_links_and_the_embedding_dimension() {
        let graph = LodGraph::new();
        graph
            .add_node(node("kept", 1).with_aliases(["tag"]))
            .unwrap();
        let checkpoint = graph.create_checkpoint();
        graph
            .add_node(
                node("dropped", 2)
                    .with_aliases(["tag", "other"])
                    .with_embedding(test_vectors::random(1, 128)),
            )
            .unwrap();
        assert_eq!(graph.pending_edge_count(), 2);
        graph.rollback_checkpoint(&checkpoint).unwrap();
        assert_eq!(graph.pending_edge_count(), 0);
        // "other" has no holder again and "tag" has one: one pair of links.
        graph
            .add_node(node("again", 3).with_aliases(["other"]))
            .unwrap();
        assert_eq!(graph.pending_edge_count(), 0);
        graph
            .add_node(node("again", 4).with_aliases(["tag"]))
            .unwrap();
        assert_eq!(graph.pending_edge_count(), 2);
        // The 128-dimension embedding is gone, so another dimension is accepted.
        graph
            .add_node(node("vec", 5).with_embedding(test_vectors::random(2, 256)))
            .unwrap();
    }

    /// A graph of 13 embedding-placed distractors, a target near `query` and a
    /// text-only procedure node the target points to.
    fn embedded_graph(query: &[f32], link: bool) -> (LodGraph, u32, u32) {
        let graph = LodGraph::new();
        for i in 0..13_u64 {
            graph
                .add_node(
                    node("distractor", 100 + i)
                        .with_embedding(test_vectors::random(900 + i, 256))
                        .placed_by_embedding(),
                )
                .unwrap();
        }
        let target = graph
            .add_node(
                node("冷却液泄漏", 1)
                    .with_embedding(test_vectors::at_cosine(query, 0.9, 77))
                    .placed_by_embedding(),
            )
            .unwrap();
        let procedure = graph.add_node(text_node(&graph, HANDWHEEL, 2)).unwrap();
        if link {
            graph
                .add_edge(target, procedure, EdgeType::Semantic, 1.0)
                .unwrap();
        }
        graph.flush_edges_to_csr().unwrap();
        (graph, target, procedure)
    }

    #[test]
    fn a_vector_query_anchors_by_embedding_and_diffuses_to_text_nodes() {
        let query = test_vectors::random(42, 256);
        let (graph, target, procedure) = embedded_graph(&query, true);
        let result = graph
            .hybrid_rag_search_query(None, Some(&query), 1, 0.0, 0.15, 200)
            .unwrap();
        // The text-only node has no embedding: 14 of 15 nodes are searchable.
        assert_eq!(result.searchable_nodes, 14);
        assert_eq!(result.stage1_candidates, 4);
        assert_eq!(result.anchors.len(), 1);
        assert_eq!(result.anchors[0].0, target);
        let ids: Vec<u32> = result.hits.iter().map(|h| h.node_id).collect();
        assert_eq!(ids, vec![target, procedure], "{result:?}");
        assert_eq!(result.hits[0].anchor_match, Some(AnchorMatch::Embedding));
        assert_eq!(result.hits[1].anchor_distance, None);
        assert_eq!(result.hits[1].payload.as_deref(), Some(HANDWHEEL));
        assert!(result.hits[1].ppr_score > 0.0);
        // A single non-exact anchor is its track's maximum, so normalizes to 1.
        let (qc, _) = graph.project_dense(&query).unwrap();
        let stored = graph.get_node(target).unwrap();
        assert_eq!(stored.placement, Placement::Embedding);
        assert_eq!(Some(stored.coord), stored.embedding_anchor.map(|a| a.coord));
        assert!(graph.distance(&stored.coord, &qc).unwrap() > 0.0);
        assert_eq!(result.anchors[0].1, 1.0);

        // Control: without the edge the procedure node is not recalled.
        let (control, target, _) = embedded_graph(&query, false);
        let missed = control
            .hybrid_rag_search_query(None, Some(&query), 1, 0.0, 0.15, 200)
            .unwrap();
        let ids: Vec<u32> = missed.hits.iter().map(|h| h.node_id).collect();
        assert_eq!(ids, vec![target]);
    }

    #[test]
    fn text_and_vector_tracks_never_cross_and_combine_in_one_query() {
        let query = test_vectors::random(42, 256);
        let (graph, target, procedure) = embedded_graph(&query, false);
        // A text query sees only the one node with a chart anchor.
        let text_only = graph
            .hybrid_rag_search_query(Some("handwheel clockwise"), None, 3, 0.0, 0.15, 100)
            .unwrap();
        assert_eq!(text_only.searchable_nodes, 1);
        assert_eq!(text_only.anchors.len(), 1);
        assert_eq!(text_only.anchors[0].0, procedure);
        // The same holds for a coordinate query and for recall.
        let (coord, hdc) = graph.project_text("handwheel clockwise").unwrap();
        assert_eq!(
            graph.two_stage_recall(&coord, &hdc, 3, 0.0).unwrap().len(),
            1
        );
        // Both tracks in one query: each gives its own top anchor, and the
        // partly matching text is not crowded out by the dense track.
        let both = graph
            .hybrid_rag_search_query(Some("handwheel clockwise"), Some(&query), 1, 0.0, 0.15, 100)
            .unwrap();
        assert_eq!(both.searchable_nodes, 15);
        assert_eq!(both.stage1_candidates, 1 + 4);
        assert_eq!(both.anchors.len(), 2);
        assert!(both.anchors[0].1 <= both.anchors[1].1);
        let matched: HashMap<u32, AnchorMatch> = both
            .hits
            .iter()
            .filter_map(|h| h.anchor_match.map(|m| (h.node_id, m)))
            .collect();
        assert_eq!(matched.len(), 2);
        assert_eq!(matched.get(&target), Some(&AnchorMatch::Embedding));
        assert_eq!(matched.get(&procedure), Some(&AnchorMatch::Primary));
    }

    #[test]
    fn embeddings_and_vector_queries_fail_closed() {
        let graph = LodGraph::new();
        let vector = test_vectors::random(1, 128);
        // No node carries an embedding: a vector has nothing to be compared with.
        graph.add_node(text_node(&graph, PUMP, 1)).unwrap();
        assert!(matches!(
            graph.hybrid_rag_search_query(None, Some(&vector), 1, 0.0, 0.15, 50),
            Err(LodError::InvalidQuery(_))
        ));
        assert!(matches!(
            graph.hybrid_rag_search_query(None, None, 1, 0.0, 0.15, 50),
            Err(LodError::InvalidQuery(_))
        ));
        assert!(matches!(
            graph.add_node(node("no vector", 2).placed_by_embedding()),
            Err(LodError::InvalidNode(_))
        ));
        let mut broken = vector.clone();
        broken[0] = f32::NAN;
        assert!(graph
            .add_node(node("nan", 2).with_embedding(broken.clone()))
            .is_err());
        assert!(graph
            .add_node(node("short", 2).with_embedding(vec![1.0; 4]))
            .is_err());
        assert_eq!(graph.node_count(), 1);

        graph
            .add_node(node("vec", 2).with_embedding(vector.clone()))
            .unwrap();
        // The first embedding fixes the dimension for nodes and for queries.
        assert!(matches!(
            graph.add_node(node("other dim", 3).with_embedding(test_vectors::random(2, 256))),
            Err(LodError::InvalidNode(_))
        ));
        assert!(matches!(
            graph.hybrid_rag_search_query(
                None,
                Some(&test_vectors::random(2, 256)),
                1,
                0.0,
                0.15,
                50
            ),
            Err(LodError::InvalidQuery(_))
        ));
        assert!(graph
            .hybrid_rag_search_query(None, Some(&broken), 1, 0.0, 0.15, 50)
            .is_err());
        let ok = graph
            .hybrid_rag_search_query(None, Some(&vector), 1, 0.0, 0.15, 50)
            .unwrap();
        assert_eq!(ok.hits[0].entity_id, 2);
        assert!(ok.hits[0].anchor_distance.unwrap() < 1e-6);
        // A node with an embedding but a chart placement keeps its own
        // coordinate and answers both kinds of query.
        assert_eq!(graph.get_node(1).unwrap().placement, Placement::Chart);
        assert_eq!(ok.searchable_nodes, 1);
    }

    // ---------------------------------------------------------------- SCC blocks

    fn evolve_signed(
        graph: &LodGraph,
        beta: f32,
        gamma: f32,
    ) -> Result<FixedPointReport, LodError> {
        graph.evolve_signed_epistemic_fixed_point_within(
            beta,
            gamma,
            TOL,
            0.2,
            0.8,
            MAX_FIXED_POINT_STEPS,
        )
    }

    /// Tarjan on a hand-drawn graph: components exact, every component listed
    /// after every component its arcs reach.
    #[test]
    fn tarjan_lists_components_after_everything_they_reach() {
        // Arcs: 0 -> 1, 1 -> 2, 2 -> 1, 2 -> 3, 3 -> 3, 4 -> 0, 4 -> 5, 5 -> 4.
        let arcs: [&[u32]; 6] = [&[1], &[2], &[1, 3], &[3], &[0, 5], &[4]];
        let mut offsets = vec![0];
        let mut targets = Vec::new();
        for row in arcs {
            targets.extend_from_slice(row);
            offsets.push(targets.len());
        }
        let condensation = tarjan_scc(&offsets, &targets);
        let components: Vec<Vec<u32>> = (0..condensation.component_count())
            .map(|k| {
                let mut c = condensation.component(k).to_vec();
                c.sort_unstable();
                c
            })
            .collect();
        assert_eq!(components, vec![vec![3], vec![1, 2], vec![0], vec![4, 5]]);

        // A chain of 200 000 arcs: iterative, so no stack overflow.
        let n = 200_000;
        let offsets: Vec<usize> = (0..=n).map(|i| i.min(n - 1)).collect();
        let targets: Vec<u32> = (1..n as u32).collect();
        let condensation = tarjan_scc(&offsets, &targets);
        assert_eq!(condensation.component_count(), n);
        assert_eq!(condensation.component(0), &[n as u32 - 1]);
        assert_eq!(condensation.component(n - 1), &[0]);
    }

    /// A DAG (an axiom feeding a chain and a diamond): every component is
    /// trivial, each node takes exactly one evaluation, there is no iteration,
    /// and each value is the exact fixed point.
    #[test]
    fn dag_blocks_are_trivial_and_take_one_evaluation_each() {
        let graph = LodGraph::new();
        let x = graph
            .add_node(node("x", 1).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        let a = graph.add_node(node("a", 2).with_prior(0.4)).unwrap();
        let b = graph.add_node(node("b", 3).with_prior(0.6)).unwrap();
        let c = graph.add_node(node("c", 4).with_prior(0.2)).unwrap();
        let d = graph.add_node(node("d", 5).with_prior(0.9)).unwrap();
        graph.add_edge(x, a, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        graph
            .add_edge(a, c, EdgeType::CausalTransition, 1.0)
            .unwrap();
        graph.add_edge(b, d, EdgeType::DependsOn, 3.0).unwrap();
        graph.add_edge(c, d, EdgeType::DependsOn, 1.0).unwrap();

        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!(
            (
                report.scc_count,
                report.trivial_scc_count,
                report.cyclic_scc_count,
                report.max_scc_size
            ),
            (5, 5, 0, 1)
        );
        assert_eq!(report.node_updates, 5);
        assert_eq!((report.iterations, report.k_max), (0, 0));
        assert_eq!((report.contraction, report.error_bound), (0.0, 0.0));

        let beta = f64::from(BETA);
        let k = 1.0 - beta;
        let ea = k * 0.4_f32 as f64 + beta;
        let eb = k * 0.6_f32 as f64 + beta * ea;
        let ec = k * 0.2_f32 as f64 + beta * ea;
        let ed = k * 0.9_f32 as f64 + beta * (0.75 * eb + 0.25 * ec);
        let got = confidences(&graph);
        for (id, exact) in [(a, ea), (b, eb), (c, ec), (d, ed)] {
            assert_eq!(got[id as usize], exact as f32, "node {id}");
        }
    }

    /// Only the cycle iterates: an axiom feeds the loop {a, b}, whose output
    /// feeds d. d is evaluated once, from the converged loop, and every node
    /// satisfies the global equation within the reported bound.
    #[test]
    fn cyclic_block_iterates_alone_and_feeds_its_dependents_once() {
        let graph = LodGraph::new();
        let x = graph
            .add_node(node("x", 1).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        let a = graph.add_node(node("a", 2).with_prior(0.3)).unwrap();
        let b = graph.add_node(node("b", 3).with_prior(0.7)).unwrap();
        let d = graph.add_node(node("d", 4).with_prior(0.5)).unwrap();
        let s = graph.add_node(node("s", 5).with_prior(0.5)).unwrap();
        graph.add_edge(x, a, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(b, a, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(b, d, EdgeType::DependsOn, 1.0).unwrap();
        // A self-loop is a cycle of one node.
        graph.add_edge(s, s, EdgeType::DependsOn, 1.0).unwrap();

        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!(
            (
                report.scc_count,
                report.trivial_scc_count,
                report.cyclic_scc_count,
                report.max_scc_size
            ),
            (4, 2, 2, 2)
        );
        assert!(report.iterations >= 1 && report.iterations <= report.k_max);
        assert!(report.residual < f64::from(TOL));
        // The loop {a, b}: row a is half internal, row b wholly: q = beta.
        assert_eq!(report.contraction, f64::from(BETA));

        let beta = f64::from(BETA);
        let k = 1.0 - beta;
        let c: Vec<f64> = confidences(&graph).iter().map(|&v| f64::from(v)).collect();
        let t = [
            1.0,
            k * 0.3_f32 as f64 + beta * 0.5 * (c[x as usize] + c[b as usize]),
            k * 0.7_f32 as f64 + beta * c[a as usize],
            k * 0.5 + beta * c[b as usize],
            k * 0.5 + beta * c[s as usize],
        ];
        for v in 0..5 {
            // f32 storage adds its rounding to the iteration's own bound.
            assert!(
                (c[v] - t[v]).abs() <= report.error_bound + 1e-6,
                "node {v}: {} vs T = {}",
                c[v],
                t[v]
            );
        }
        // The self-loop's fixed point is its prior.
        assert!((c[s as usize] - 0.5).abs() <= report.error_bound + 1e-6);
    }

    /// Axiom s supports t (prior 0.9); axiom f falsifies t; u depends on t.
    fn falsified_world() -> (LodGraph, [u32; 3]) {
        let graph = LodGraph::new();
        let s = graph
            .add_node(node("s", 1).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        let f = graph
            .add_node(node("f", 2).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        let t = graph.add_node(node("t", 3).with_prior(0.9)).unwrap();
        let u = graph.add_node(node("u", 4).with_prior(0.9)).unwrap();
        graph.add_edge(s, t, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(f, t, EdgeType::Falsifies, 2.0).unwrap();
        graph.add_edge(t, u, EdgeType::DependsOn, 1.0).unwrap();
        (graph, [t, u, f])
    }

    /// `t = (1 - beta) 0.9 + beta max(0, 1 - gamma)`: validated without the
    /// falsifier, falsified and revoked at gamma 1, held by hysteresis at gamma
    /// 0.5, validated and reinstated again at gamma 0.
    #[test]
    fn falsifies_edge_presses_its_target_to_falsified_and_revokes_it() {
        let (graph, [t, u, _]) = falsified_world();
        let beta = f64::from(BETA);
        let k = 1.0 - beta;
        let expect_t = |gamma: f64| k * 0.9_f32 as f64 + beta * (1.0 - gamma).max(0.0);

        let off = evolve_signed(&graph, BETA, 0.0).unwrap();
        assert_eq!((off.dependency_edges, off.falsification_edges), (2, 0));
        assert_eq!(status(&graph, t), EpistemicStatus::Validated);
        assert_eq!(graph.get_node(t).unwrap().confidence, expect_t(0.0) as f32);

        let on = evolve_signed(&graph, BETA, 1.0).unwrap();
        assert_eq!((on.dependency_edges, on.falsification_edges), (2, 1));
        assert_eq!(on.gamma, 1.0);
        let ct = graph.get_node(t).unwrap().confidence;
        assert_eq!(ct, expect_t(1.0) as f32);
        assert!(ct < 0.2, "{ct}");
        assert_eq!(status(&graph, t), EpistemicStatus::Falsified);
        assert!(graph.is_revoked(3));
        assert!(on.revoked_entities.contains(&3));
        assert!(on.transitions.iter().any(|tr| tr.node == t
            && tr.from == EpistemicStatus::Validated
            && tr.to == EpistemicStatus::Falsified));
        // The dependent sinks with it, into the band where hysteresis holds its
        // status: it is damped, not cascaded.
        let cu = graph.get_node(u).unwrap().confidence;
        assert_eq!(cu, (k * 0.9_f32 as f64 + beta * f64::from(ct)) as f32);
        assert!((0.2..0.8).contains(&cu), "{cu}");
        assert_eq!(status(&graph, u), EpistemicStatus::Validated);

        // Between the thresholds the falsified status is kept.
        evolve_signed(&graph, BETA, 0.5).unwrap();
        let ct = graph.get_node(t).unwrap().confidence;
        assert!((0.2..=0.8).contains(&ct), "{ct}");
        assert_eq!(status(&graph, t), EpistemicStatus::Falsified);
        assert!(graph.is_revoked(3));

        let back = evolve_signed(&graph, BETA, 0.0).unwrap();
        assert_eq!(status(&graph, t), EpistemicStatus::Validated);
        assert!(!graph.is_revoked(3));
        assert!(back.reinstated_entities.contains(&3));
    }

    /// A doubtful falsifier presses less: the penalty is its confidence.
    #[test]
    fn falsifier_penalty_scales_with_the_falsifier_confidence() {
        let graph = LodGraph::new();
        let f = graph.add_node(node("f", 1).with_prior(0.25)).unwrap();
        let t = graph.add_node(node("t", 2).with_prior(0.8)).unwrap();
        graph.add_edge(f, t, EdgeType::Falsifies, 1.0).unwrap();
        evolve(&graph, 0.2, 0.9);
        // t has no support edge, so its own prior supports it:
        // (1 - beta) 0.8 + beta max(0, 0.8 - 0.25).
        let beta = f64::from(BETA);
        let exact = (1.0 - beta) * 0.8_f32 as f64 + beta * (0.8_f32 as f64 - 0.25);
        assert!((f64::from(graph.get_node(t).unwrap().confidence) - exact).abs() < 1e-6);
        assert_eq!(graph.get_node(f).unwrap().confidence, 0.25);
    }

    /// gamma = 0 leaves `Falsifies` edges out of the graph: the result is the
    /// same bits as the same graph without them, block structure included.
    #[test]
    fn zero_gamma_is_the_unsigned_evolution_bit_for_bit() {
        let build = |with_falsifiers: bool| {
            let graph = LodGraph::new();
            let x = graph
                .add_node(node("x", 1).with_status(EpistemicStatus::Axiomatic))
                .unwrap();
            let ids: Vec<u32> = [0.3_f32, 0.6, 0.45, 0.8]
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    graph
                        .add_node(node("n", 10 + i as u64).with_prior(p))
                        .unwrap()
                })
                .collect();
            // x carries 9 of row n0's 10 support weight, so the falsifier that
            // closes the cycle below leaves it admissible (q = 1.1 beta < 1).
            graph.add_edge(x, ids[0], EdgeType::DependsOn, 9.0).unwrap();
            graph
                .add_edge(ids[0], ids[1], EdgeType::DependsOn, 2.0)
                .unwrap();
            graph
                .add_edge(ids[1], ids[0], EdgeType::CausalTransition, 1.0)
                .unwrap();
            graph
                .add_edge(ids[1], ids[2], EdgeType::DependsOn, 1.0)
                .unwrap();
            if with_falsifiers {
                // One would join ids[3] to the loop, one close a new cycle.
                graph
                    .add_edge(ids[3], ids[0], EdgeType::Falsifies, 5.0)
                    .unwrap();
                graph
                    .add_edge(ids[2], ids[3], EdgeType::Falsifies, 1.0)
                    .unwrap();
            }
            graph
        };
        let (plain, signed) = (build(false), build(true));
        let r_plain = evolve_signed(&plain, BETA, 1.0).unwrap();
        let r_zero = evolve_signed(&signed, BETA, 0.0).unwrap();
        assert_eq!(confidences(&plain), confidences(&signed));
        let shape = |r: &FixedPointReport| {
            (
                r.scc_count,
                r.trivial_scc_count,
                r.cyclic_scc_count,
                r.max_scc_size,
                r.falsification_edges,
                r.iterations,
                r.node_updates,
            )
        };
        assert_eq!(shape(&r_plain), shape(&r_zero));
        assert_eq!(r_plain.error_bound, r_zero.error_bound);
        // With gamma on, the falsifiers enter and close the cycle
        // n0 -> n1 -> n2 -| n3 -| n0. Row n0 then weighs 1/10 + gamma inside it:
        // q = 1.6 beta at gamma 1.5, refused at beta 0.85; at gamma 1 it is
        // 1.1 beta, solved at beta 0.5. (A loop refused at gamma 1, beta 0.85
        // is refused by admission and cannot be built: see
        // `admission_refuses_a_non_contractive_cycle_and_keeps_the_graph`.)
        match evolve_signed(&build(true), BETA, 1.5).unwrap_err() {
            LodError::FixedPointNotContractive { block_size, .. } => assert_eq!(block_size, 4),
            other => panic!("expected FixedPointNotContractive, got {other}"),
        }
        let r_on = evolve_signed(&build(true), 0.5, 1.0).unwrap();
        assert_eq!(r_on.falsification_edges, 2);
        assert_eq!((r_plain.max_scc_size, r_on.max_scc_size), (2, 4));
        assert!(
            (r_on.contraction - 0.55).abs() < 1e-12,
            "{}",
            r_on.contraction
        );
    }

    /// A falsifier inside a cycle: row a is `P+ = b` and `P- = b`, Lipschitz
    /// `beta (1 + gamma)`, 1.7 at the admission parameters. The edge that closes
    /// it is refused and the graph keeps the two edges it had. The same edge is
    /// admitted once another falsifier dilutes it to a tenth of row a's `P-`:
    /// `q = 1.1 beta`. On that admitted graph an evolution at gamma 2
    /// (`q = 1.2 beta`) is still refused and commits nothing; at gamma 1 it
    /// solves to the exact fixed point.
    #[test]
    fn admission_refuses_a_non_contractive_cycle_and_keeps_the_graph() {
        let graph = LodGraph::new();
        let a = graph.add_node(node("a", 1).with_prior(0.7)).unwrap();
        let b = graph.add_node(node("b", 2).with_prior(0.4)).unwrap();
        graph.add_edge(b, a, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        let before = (confidences(&graph), status(&graph, a), status(&graph, b));
        match graph.add_edge(b, a, EdgeType::Falsifies, 1.0).unwrap_err() {
            LodError::FixedPointNotContractive {
                block_size,
                contraction,
                beta,
                gamma,
            } => {
                assert_eq!(block_size, 2);
                assert!((contraction - 2.0 * f64::from(ADMISSION_BETA)).abs() < 1e-12);
                assert_eq!((beta, gamma), (0.85_f32 as f64, 1.0));
            }
            other => panic!("expected FixedPointNotContractive, got {other}"),
        }
        assert_eq!(graph.pending_edge_count(), 2);
        assert_eq!(
            (confidences(&graph), status(&graph, a), status(&graph, b)),
            before
        );
        assert!(!graph.is_revoked(1) && !graph.is_revoked(2));

        // A falsifier from outside the loop takes 9/10 of row a's `P-`. As a
        // batch, the closing edge alone is refused and so is the whole batch;
        // with the diluting edge in it, the batch is admitted, tickets in order.
        let e = graph.add_node(node("e", 3).with_prior(0.0)).unwrap();
        assert!(matches!(
            graph.add_edges(&[
                (e, b, EdgeType::DependsOn, 1.0),
                (b, a, EdgeType::Falsifies, 1.0)
            ]),
            Err(LodError::FixedPointNotContractive { block_size: 2, .. })
        ));
        assert_eq!(graph.pending_edge_count(), 2);
        let tickets = graph
            .add_edges(&[
                (e, a, EdgeType::Falsifies, 9.0),
                (b, a, EdgeType::Falsifies, 1.0),
            ])
            .unwrap();
        assert_eq!(tickets.len(), 2);
        assert!(tickets[0] < tickets[1]);
        assert_eq!(graph.pending_edge_count(), 4);
        let before = (confidences(&graph), status(&graph, a), status(&graph, b));
        match evolve_signed(&graph, BETA, 2.0).unwrap_err() {
            LodError::FixedPointNotContractive {
                block_size,
                contraction,
                ..
            } => {
                assert_eq!(block_size, 2);
                assert!((contraction - 1.2 * f64::from(BETA)).abs() < 1e-12);
            }
            other => panic!("expected FixedPointNotContractive, got {other}"),
        }
        assert_eq!(
            (confidences(&graph), status(&graph, a), status(&graph, b)),
            before
        );

        let report = evolve_signed(&graph, BETA, 1.0).unwrap();
        assert!((report.contraction - 1.1 * f64::from(BETA)).abs() < 1e-12);
        assert!(report.adapted_blocks.is_empty());
        // e has prior 0 and no support: it stays at 0 and presses nothing.
        // a = k 0.7 + beta (b - b / 10), b = k 0.4 + beta a.
        let beta = f64::from(BETA);
        let k = 1.0 - beta;
        let (pa, pb) = (f64::from(0.7_f32), f64::from(0.4_f32));
        let ea = k * (pa + 0.9 * beta * pb) / (1.0 - 0.9 * beta * beta);
        let eb = k * pb + beta * ea;
        let got = confidences(&graph);
        assert_eq!(got[e as usize], 0.0);
        assert!((f64::from(got[a as usize]) - ea).abs() <= report.error_bound + 1e-6);
        assert!((f64::from(got[b as usize]) - eb).abs() <= report.error_bound + 1e-6);
    }

    /// Unpinning a refuted node gives it back its incoming edges. When they
    /// close a cycle admission refuses, the retraction is refused and the node
    /// stays refuted.
    #[test]
    fn retraction_that_would_close_a_non_contractive_cycle_is_refused() {
        let graph = LodGraph::new();
        let a = graph.add_node(node("a", 1).with_prior(0.7)).unwrap();
        let b = graph.add_node(node("b", 2).with_prior(0.4)).unwrap();
        graph.falsify_node(a).unwrap();
        // Edges into a refuted node do not enter the iteration: all admitted.
        graph.add_edge(b, a, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(b, a, EdgeType::Falsifies, 1.0).unwrap();
        assert!(matches!(
            graph.retract_falsification(a),
            Err(LodError::FixedPointNotContractive { block_size: 2, .. })
        ));
        let node_a = graph.get_node(a).unwrap();
        assert!(node_a.refuted);
        assert_eq!(node_a.status, EpistemicStatus::Falsified);
        assert_eq!(node_a.confidence, 0.0);
        assert!(graph.is_revoked(1));
        evolve_signed(&graph, BETA, 1.0).unwrap();
    }

    /// Fixture only: append edges to the pending buffer without admission, the
    /// state a graph would be in if the admission invariant were broken. No
    /// production path writes the buffer this way.
    fn inject_unadmitted(graph: &LodGraph, edges: &[(u32, u32, EdgeType, f32)]) {
        let mut st = graph.state.write();
        for &(source, target, edge_type, weight) in edges {
            let ticket = graph.ticket_counter.fetch_add(1, Ordering::Relaxed);
            st.edge_buffer.push(BufferedEdge {
                source,
                target,
                edge_type,
                weight,
                ticket,
            });
        }
    }

    /// Reflection on a graph holding a cycle that is not a contraction at its
    /// parameters: the cycle's internal falsifier gain is lowered and reported,
    /// the evolution converges, the evidence and its edge are committed and
    /// the action is revoked. The plain evolution on the same graph is refused.
    #[test]
    fn reflection_adapts_the_gain_of_a_non_contractive_cycle() {
        let graph = LodGraph::new();
        let a = graph.add_node(node("a", 7).with_prior(0.9)).unwrap();
        let b = graph.add_node(node("b", 8).with_prior(0.9)).unwrap();
        let s = graph
            .add_node(node("s", 9).with_status(EpistemicStatus::Axiomatic))
            .unwrap();
        inject_unadmitted(
            &graph,
            &[
                (b, a, EdgeType::DependsOn, 1.0),
                (a, b, EdgeType::DependsOn, 1.0),
                (b, a, EdgeType::Falsifies, 1.0),
                (a, b, EdgeType::Falsifies, 1.0),
                (s, b, EdgeType::DependsOn, 1.0),
            ],
        );
        assert!(matches!(
            evolve_signed(&graph, ADMISSION_BETA, ADMISSION_GAMMA),
            Err(LodError::FixedPointNotContractive { block_size: 2, .. })
        ));

        let reflection = graph
            .reflect_failure(7, "model terminal diagnostic", 1)
            .unwrap();
        assert_eq!(reflection.target, a);
        let evidence = graph.get_node(reflection.evidence).unwrap();
        assert_eq!(
            evidence.payload.as_deref(),
            Some("model terminal diagnostic")
        );
        // Nothing was flushed: the observation's edge is in the pending buffer.
        assert!(graph
            .state
            .read()
            .edge_buffer
            .iter()
            .any(|e| e.source == reflection.evidence
                && e.target == a
                && e.edge_type == EdgeType::Falsifies
                && e.weight == 1.0));
        assert_eq!(reflection.revocation, ReflectionRevocation::Evolution);
        assert!(graph.is_revoked(7));
        assert!(status(&graph, a).is_falsified());

        // Row a: support b (1), falsifiers b (1/2) and the evidence (1/2, from
        // outside). Row b: support a (1/2) and s (1/2), falsifier a (1).
        // a_max = 1, so q* = (1 + beta) / 2 and the binding row is a:
        // g = (q* / beta - 1) / (1/2) = (1 - beta) / beta.
        let report = &reflection.evolution;
        assert_eq!(report.adapted_blocks.len(), 1);
        let adapted = &report.adapted_blocks[0];
        let beta = f64::from(ADMISSION_BETA);
        let mut nodes = adapted.nodes.clone();
        nodes.sort_unstable();
        assert_eq!(nodes, vec![a, b]);
        assert_eq!(adapted.requested_gamma, 1.0);
        assert!((adapted.requested_contraction - 1.5 * beta).abs() < 1e-12);
        assert!((adapted.applied_gamma - (1.0 - beta) / beta).abs() < 1e-12);
        assert!((adapted.contraction - 0.5 * (1.0 + beta)).abs() < 1e-12);
        assert!(adapted.contraction < 1.0);
        assert_eq!(report.contraction, adapted.contraction);

        // The committed values are a fixed point of the adapted map.
        let g = adapted.applied_gamma;
        let k = 1.0 - beta;
        let c = confidences(&graph);
        let (ca, cb, ce) = (
            f64::from(c[a as usize]),
            f64::from(c[b as usize]),
            f64::from(c[reflection.evidence as usize]),
        );
        let ta = k * 0.9 + beta * (cb - (g * 0.5 * cb + 1.0 * 0.5 * ce)).max(0.0);
        let tb = k * 0.9 + beta * (0.5 * ca + 0.5 - g * ca).max(0.0);
        assert!((ca - ta).abs() <= report.error_bound + 1e-5, "{ca} vs {ta}");
        assert!((cb - tb).abs() <= report.error_bound + 1e-5, "{cb} vs {tb}");
        assert_eq!(f64::from(reflection.target_confidence), ca);
    }

    /// A cycle that contracts (`q = 1 - 1e-5`) but too slowly for the step
    /// budget, injected past admission: the plain evolution runs out of steps;
    /// the reflection lowers the block's internal falsifier gain, reports it
    /// and converges.
    #[test]
    fn reflection_adapts_a_cycle_too_slow_for_the_step_budget() {
        let graph = LodGraph::new();
        let p = |e: u64, prior: f32| graph.add_node(node("n", e).with_prior(prior)).unwrap();
        let (a, b, c, d) = (p(1, 1.0), p(2, 1.0), p(3, 1.0 - 6.0e-5), p(4, 1.0 - 6.0e-5));
        let zero = p(5, 0.0);
        p(99, 0.9);
        let mut edges = vec![
            (b, a, EdgeType::DependsOn, 1.0),
            (a, b, EdgeType::DependsOn, 1.0),
            (d, c, EdgeType::DependsOn, 1.0),
            (c, d, EdgeType::DependsOn, 1.0),
        ];
        for (source, target) in [(c, a), (d, b), (a, c), (b, d)] {
            edges.push((zero, target, EdgeType::Falsifies, 8235412.0));
            edges.push((source, target, EdgeType::Falsifies, 1764588.0));
        }
        assert!(matches!(
            graph.add_edges(&edges),
            Err(LodError::FixedPointTooSlow { block_size: 4, .. })
        ));
        inject_unadmitted(&graph, &edges);
        let before = confidences(&graph);
        let refused = graph.evolve_signed_epistemic_fixed_point_within(
            ADMISSION_BETA,
            ADMISSION_GAMMA,
            REFLECTION_TOLERANCE,
            REFLECTION_THETA_LO,
            REFLECTION_THETA_HI,
            MAX_FIXED_POINT_STEPS,
        );
        assert!(
            matches!(refused, Err(LodError::FixedPointDiverged { iterations, .. })
                if iterations == MAX_FIXED_POINT_STEPS),
            "{refused:?}"
        );
        assert_eq!(confidences(&graph), before);

        let reflection = graph.reflect_failure(99, "observation", 1).unwrap();
        assert!(graph.is_revoked(99));
        let adapted = &reflection.evolution.adapted_blocks;
        assert_eq!(adapted.len(), 1);
        assert_eq!(adapted[0].nodes.len(), 4);
        assert!(adapted[0].requested_contraction < 1.0);
        assert!(adapted[0].requested_contraction > 0.9999);
        assert!(adapted[0].applied_gamma < 1.0);
        assert!(adapted[0].contraction <= 0.5 * (1.0 + f64::from(ADMISSION_BETA)) + 1e-12);
        assert!(reflection.evolution.iterations < MAX_FIXED_POINT_STEPS);
    }

    /// A non-finite value in either kind of block is a refusal, never a value.
    #[test]
    fn non_finite_values_fail_closed_in_both_block_kinds() {
        // Node 0 alone, nodes 1 and 2 a loop reading node 0.
        let rows = SignedRows {
            offsets: vec![0, 0, 2, 3],
            sources: vec![0, 2, 1],
            probabilities: vec![0.5, 0.5, 1.0],
            falsifies: vec![false; 3],
            supported: vec![false, true, true],
        };
        for prior in [
            vec![f64::NAN, 0.5, 0.5],
            vec![0.5, f64::INFINITY, 0.5],
            vec![0.5, 0.5, f64::NAN],
        ] {
            assert!(
                matches!(
                    solve_by_blocks(
                        &rows,
                        &prior,
                        0.85,
                        1.0,
                        1e-6,
                        100,
                        OnNonContractive::Refuse
                    ),
                    Err(LodError::FixedPointDiverged { .. })
                ),
                "{prior:?}"
            );
        }
        let ok = solve_by_blocks(
            &rows,
            &[0.5, 0.5, 0.5],
            0.85,
            1.0,
            1e-6,
            100,
            OnNonContractive::Refuse,
        )
        .unwrap();
        assert_eq!((ok.trivial_scc_count, ok.cyclic_scc_count), (1, 1));
    }
}
