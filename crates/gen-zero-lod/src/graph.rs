//! gen-zero-lod dynamic graph topology, epistemic lifecycle and causal cascade pruning.
//!
//! Readers traverse an immutable CSR snapshot (`ArcSwap<CsrGraph>`) without locks.
//! Writers append ticketed edges to a pending buffer. `flush_edges_to_csr` merges
//! only the pending edges into a new snapshot and drains them from the buffer, so
//! the buffer never holds committed history.
//!
//! `create_checkpoint` / `rollback_checkpoint` restore the whole mutable graph state
//! atomically. `cascade_prune_and_rollback` falsifies a premise and its dependents,
//! retracts their validated dependencies, and returns the pre-prune checkpoint so
//! the prune itself can be undone. Nothing here touches a search tree: the planner's
//! MCTS is sequential and has no virtual loss (see `gen-zero-planner/src/config.rs`).

use crate::error::LodError;
use crate::manifold::MixedCurvatureCoord;
use crate::node::{hdc_hamming_distance_256, EpistemicStatus, LodNode};
use crate::ppr::compute_ppr_csr;
use arc_swap::ArcSwap;
use gen_zero_core::GraphFactProvider;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
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
#[derive(Clone, Debug)]
pub struct CsrGraph {
    num_nodes: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<u32>,
    edge_weights: Vec<f32>,
    edge_types: Vec<EdgeType>,
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
        if self.row_offsets.len() != self.num_nodes + 1 {
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

/// Mutable graph state. One lock guards all of it, so a checkpoint or a rollback
/// reads or restores a single consistent state.
///
/// Lock order: `txn_lock` -> `flush_lock` -> `state`. The CSR snapshot is stored
/// only while `state` is write-locked; methods that must see it consistent with
/// the nodes load it while holding `state`.
#[derive(Default)]
struct GraphState {
    nodes: Vec<LodNode>,
    entity_index: HashMap<u64, u32>,
    /// Edges not yet merged into the CSR snapshot, in ticket order.
    edge_buffer: Vec<BufferedEdge>,
    revocations: HashSet<u64>,
    privileges: HashMap<u64, Vec<u32>>,
    validated_deps: HashSet<(u64, u64)>,
    /// Bumped by every CSR store and every rollback. A flush built against an
    /// older generation is discarded.
    generation: u64,
    /// Sorted, disjoint checkpoint sequence ranges `(after, upto]` whose state a
    /// rollback discarded. Checkpoints in them can no longer be restored.
    discarded: Vec<(u64, u64)>,
}

/// Everything a rollback restores: the node count, each node's status and
/// confidence (the only node fields any method mutates), the CSR snapshot
/// reference, pending edges, revocations, privileges and validated dependencies.
/// The edge ticket counter is never rewound, so tickets stay unique.
///
/// Memory is O(nodes + pending edges + revocations + dependencies) per checkpoint.
#[derive(Clone, Debug)]
pub struct GraphCheckpoint {
    graph_id: u64,
    seq: u64,
    node_states: Vec<(EpistemicStatus, f32)>,
    csr: Arc<CsrGraph>,
    edge_buffer: Vec<BufferedEdge>,
    revocations: HashSet<u64>,
    privileges: HashMap<u64, Vec<u32>>,
    validated_deps: HashSet<(u64, u64)>,
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

/// Result of one [`LodGraph::cascade_prune_and_rollback`].
#[derive(Clone, Debug)]
pub struct PruneOutcome {
    /// The falsified root, then every dependent falsified with it, in BFS order.
    pub pruned: Vec<u32>,
    /// Entity ids revoked by this prune, parallel to `pruned`.
    pub revoked_entities: Vec<u64>,
    /// Validated dependencies retracted because one end was pruned.
    pub retracted_dependencies: usize,
    /// State just before the prune, taken under the same lock. Pass it to
    /// [`LodGraph::rollback_checkpoint`] to undo the prune.
    pub checkpoint: GraphCheckpoint,
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

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// Dynamic LodGraph integrating multi-scale nodes, ticketed edge buffer,
/// lock-free CSR snapshots, and epistemic governance.
pub struct LodGraph {
    /// Identity checked by `rollback_checkpoint`, so a checkpoint cannot be
    /// restored into another graph.
    graph_id: u64,
    state: RwLock<GraphState>,
    csr_snapshot: ArcSwap<CsrGraph>,
    ticket_counter: AtomicU64,
    checkpoint_seq: AtomicU64,
    /// Serializes flush builds against each other and against rollbacks.
    flush_lock: Mutex<()>,
    /// Serializes [`LodGraph::transact`] writers.
    txn_lock: Mutex<()>,
}

impl Default for LodGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl LodGraph {
    /// Initialize an empty LodGraph.
    pub fn new() -> Self {
        Self {
            graph_id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            state: RwLock::new(GraphState::default()),
            csr_snapshot: ArcSwap::from_pointee(CsrGraph::empty()),
            ticket_counter: AtomicU64::new(1),
            checkpoint_seq: AtomicU64::new(0),
            flush_lock: Mutex::new(()),
            txn_lock: Mutex::new(()),
        }
    }

    /// Insert a node and return its id. Refuses a coordinate the recall metric
    /// would reject, a confidence outside [0, 1], an unknown parent, and an
    /// entity id that already has a node: the gate addresses nodes by entity.
    pub fn add_node(&self, mut node: LodNode) -> Result<u32, LodError> {
        node.coord
            .product_distance(&MixedCurvatureCoord::origin())?;
        if !(node.confidence.is_finite() && (0.0..=1.0).contains(&node.confidence)) {
            return Err(LodError::InvalidNode(format!(
                "confidence {} must lie in [0, 1]",
                node.confidence
            )));
        }
        let mut st = self.state.write();
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
        st.entity_index.insert(node.entity_id, id);
        st.nodes.push(node);
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
    /// nonnegative. The critical section is O(1): no history is copied.
    pub fn add_edge(
        &self,
        source: u32,
        target: u32,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<u64, LodError> {
        let mut st = self.state.write();
        check_edge(st.nodes.len(), source, target, weight)?;
        // Taken under the lock so buffer order is ticket order.
        let ticket = self.ticket_counter.fetch_add(1, Ordering::Relaxed);
        st.edge_buffer.push(BufferedEdge {
            source,
            target,
            edge_type,
            weight,
            ticket,
        });
        if matches!(edge_type, EdgeType::DependsOn | EdgeType::Validates) {
            let (u, v) = (&st.nodes[source as usize], &st.nodes[target as usize]);
            if u.status.is_active_truth() && v.status.is_active_truth() {
                let dep = (u.entity_id, v.entity_id);
                st.validated_deps.insert(dep);
            }
        }
        Ok(ticket)
    }

    /// Transition a node from Hypothesized to Validated upon environment proof or intervention.
    /// Backfills validated dependencies on both outgoing and incoming
    /// `DependsOn` / `Validates` edges, in the CSR snapshot and the pending buffer.
    pub fn validate_node(&self, node_id: u32) -> Result<(), LodError> {
        let mut guard = self.state.write();
        let st = &mut *guard;
        let idx = node_id as usize;
        if idx >= st.nodes.len() {
            return Err(LodError::NodeNotFound(node_id));
        }
        if st.nodes[idx].status == EpistemicStatus::Falsified {
            return Err(LodError::InvalidStateTransition(
                "Cannot validate an already falsified node".into(),
            ));
        }

        st.nodes[idx].status = EpistemicStatus::Validated;
        st.nodes[idx].confidence = 1.0;
        let node_entity = st.nodes[idx].entity_id;
        st.revocations.remove(&node_entity);

        let dependency = |t: EdgeType| matches!(t, EdgeType::DependsOn | EdgeType::Validates);
        let active = |id: u32| {
            st.nodes
                .get(id as usize)
                .filter(|n| n.status.is_active_truth())
                .map(|n| n.entity_id)
        };
        let snapshot = self.csr_snapshot.load();
        let mut new_deps = Vec::new();
        for (nbr, edge_type, _) in snapshot.neighbors(node_id) {
            if dependency(edge_type) {
                if let Some(target) = active(nbr) {
                    new_deps.push((node_entity, target));
                }
            }
        }
        // Incoming CSR edges: the CSR indexes rows by source only, so this is an
        // O(E) scan. After a flush the buffer is small and nearly every edge lives here.
        for src in 0..snapshot.num_nodes() as u32 {
            for (nbr, edge_type, _) in snapshot.neighbors(src) {
                if nbr == node_id && dependency(edge_type) {
                    if let Some(source) = active(src) {
                        new_deps.push((source, node_entity));
                    }
                }
            }
        }
        for edge in &st.edge_buffer {
            if !dependency(edge.edge_type) {
                continue;
            }
            if edge.source == node_id {
                if let Some(target) = active(edge.target) {
                    new_deps.push((node_entity, target));
                }
            }
            if edge.target == node_id {
                if let Some(source) = active(edge.source) {
                    new_deps.push((source, node_entity));
                }
            }
        }
        st.validated_deps.extend(new_deps);
        Ok(())
    }

    /// Merge the pending edges into a new CSR snapshot and drain them from the
    /// buffer. The new snapshot is the current snapshot plus the pending delta,
    /// grown to the current node count, and is validated before it is published.
    ///
    /// On any error the old snapshot and the whole buffer are kept. Edges added
    /// while the snapshot builds stay pending. A rollback during the build makes
    /// this return `FlushConflict` without committing.
    pub fn flush_edges_to_csr(&self) -> Result<FlushReport, LodError> {
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
    /// keeping the `4 * top_k` closest.
    /// Stage 2: Mixed-curvature product geodesic rerank of those candidates, with a
    /// Corrective RAG (CRAG) margin: when the top two are closer than
    /// `crag_margin`, the 1-hop CSR neighbors of the top one join the rerank.
    ///
    /// Excludes falsified and revoked nodes. `top_k` must be at least 1 and
    /// `crag_margin` finite and nonnegative; an out-of-domain coordinate is an error.
    pub fn two_stage_recall(
        &self,
        query_coord: &MixedCurvatureCoord,
        query_hdc: &[u64; 4],
        top_k: usize,
        crag_margin: f32,
    ) -> Result<Vec<(u32, f32)>, LodError> {
        if top_k == 0 {
            return Err(LodError::InvalidQuery("top_k must be at least 1".into()));
        }
        if !(crag_margin.is_finite() && crag_margin >= 0.0) {
            return Err(LodError::InvalidQuery(format!(
                "crag_margin must be finite and nonnegative, got {crag_margin}"
            )));
        }
        query_coord.product_distance(&MixedCurvatureCoord::origin())?;
        let st = self.state.read();
        let (nodes, revs) = (&st.nodes, &st.revocations);

        let mut candidates: Vec<(u32, u32)> = nodes
            .iter()
            .filter(|n| !n.status.is_falsified() && !revs.contains(&n.entity_id))
            .map(|n| {
                (
                    n.id,
                    hdc_hamming_distance_256(&n.hdc_fingerprint, query_hdc),
                )
            })
            .collect();

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_pool_size = top_k.saturating_mul(4).min(candidates.len());
        if candidate_pool_size < candidates.len() {
            // O(N) linear selection instead of O(N log N) full sort
            candidates.select_nth_unstable_by_key(candidate_pool_size, |c| c.1);
            candidates.truncate(candidate_pool_size);
        }

        // An out-of-domain coordinate aborts the recall instead of being skipped.
        let mut reranked: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        for &(id, _) in &candidates {
            reranked.push((id, nodes[id as usize].coord.product_distance(query_coord)?));
        }
        reranked.sort_by(|a, b| a.1.total_cmp(&b.1));

        if reranked.len() >= 2 && (reranked[1].1 - reranked[0].1).abs() < crag_margin {
            let snapshot = self.csr_snapshot.load();
            let top1_id = reranked[0].0;
            for (nbr, _, _) in snapshot.neighbors(top1_id) {
                if reranked.iter().any(|(id, _)| *id == nbr) {
                    continue;
                }
                let nbr_node = &nodes[nbr as usize];
                if !nbr_node.status.is_falsified() && !revs.contains(&nbr_node.entity_id) {
                    reranked.push((nbr, nbr_node.coord.product_distance(query_coord)?));
                }
            }
            reranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        }

        reranked.truncate(top_k);
        Ok(reranked)
    }

    /// Pearl causal subtree pruning with belief rollback.
    ///
    /// When environment feedback or formal proof falsifies an assumption:
    /// 1. Mark `falsified_node_id` as `Falsified`.
    /// 2. Walk every downstream `DependsOn` / `CausalTransition` edge, in the CSR
    ///    snapshot and the pending buffer, and falsify each non-axiomatic target.
    /// 3. Revoke the entity of every pruned node.
    /// 4. Retract every validated dependency that touches a pruned entity.
    ///
    /// The returned [`PruneOutcome::checkpoint`] is the state just before step 1,
    /// captured under the same write lock, so the whole prune can be undone with
    /// [`Self::rollback_checkpoint`]. An unknown node or an axiomatic root is an error.
    pub fn cascade_prune_and_rollback(
        &self,
        falsified_node_id: u32,
    ) -> Result<PruneOutcome, LodError> {
        let mut guard = self.state.write();
        let root = falsified_node_id as usize;
        if root >= guard.nodes.len() {
            return Err(LodError::NodeNotFound(falsified_node_id));
        }
        if guard.nodes[root].status == EpistemicStatus::Axiomatic {
            return Err(LodError::InvalidStateTransition(format!(
                "node {falsified_node_id} is axiomatic and cannot be pruned"
            )));
        }
        let checkpoint = self.capture(&guard);
        let st = &mut *guard;

        let mut pruned = vec![falsified_node_id];
        let mut queue = VecDeque::from([falsified_node_id]);
        st.nodes[root].status = EpistemicStatus::Falsified;

        let snapshot = self.csr_snapshot.load();
        let propagates =
            |t: EdgeType| matches!(t, EdgeType::DependsOn | EdgeType::CausalTransition);
        while let Some(curr) = queue.pop_front() {
            let csr_targets = snapshot
                .neighbors(curr)
                .filter(|(_, t, _)| propagates(*t))
                .map(|(v, _, _)| v);
            let buffered_targets = st
                .edge_buffer
                .iter()
                .filter(|e| e.source == curr && propagates(e.edge_type))
                .map(|e| e.target);
            let targets: Vec<u32> = csr_targets.chain(buffered_targets).collect();
            for nbr in targets {
                let node = &mut st.nodes[nbr as usize];
                if !matches!(
                    node.status,
                    EpistemicStatus::Axiomatic | EpistemicStatus::Falsified
                ) {
                    node.status = EpistemicStatus::Falsified;
                    pruned.push(nbr);
                    queue.push_back(nbr);
                }
            }
        }

        let revoked_entities: Vec<u64> = pruned
            .iter()
            .map(|&id| st.nodes[id as usize].entity_id)
            .collect();
        st.revocations.extend(revoked_entities.iter().copied());
        let pruned_entities: HashSet<u64> = revoked_entities.iter().copied().collect();
        let before = st.validated_deps.len();
        st.validated_deps
            .retain(|(u, v)| !pruned_entities.contains(u) && !pruned_entities.contains(v));
        let retracted_dependencies = before - st.validated_deps.len();

        Ok(PruneOutcome {
            pruned,
            revoked_entities,
            retracted_dependencies,
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
            seq: self.checkpoint_seq.fetch_add(1, Ordering::Relaxed) + 1,
            node_states: st.nodes.iter().map(|n| (n.status, n.confidence)).collect(),
            csr: self.csr_snapshot.load_full(),
            edge_buffer: st.edge_buffer.clone(),
            revocations: st.revocations.clone(),
            privileges: st.privileges.clone(),
            validated_deps: st.validated_deps.clone(),
        }
    }

    /// Restore `checkpoint` atomically: nodes added after it are removed (their
    /// ids become free again), statuses, CSR snapshot, pending edges,
    /// revocations, privileges and validated dependencies return to its values.
    ///
    /// Every write since the checkpoint is discarded, including writes by other
    /// threads; use [`Self::transact`] to keep writers serialized. Refused: a
    /// checkpoint of another graph, and one taken after an earlier checkpoint that
    /// has since been restored (its state no longer exists).
    pub fn rollback_checkpoint(&self, checkpoint: &GraphCheckpoint) -> Result<(), LodError> {
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

        st.nodes.truncate(keep);
        st.entity_index.retain(|_, id| (*id as usize) < keep);
        for (node, &(status, confidence)) in st.nodes.iter_mut().zip(&checkpoint.node_states) {
            node.status = status;
            node.confidence = confidence;
        }
        st.edge_buffer = checkpoint.edge_buffer.clone();
        st.revocations = checkpoint.revocations.clone();
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

    /// Run `f` as one write transaction: writers through `transact` are
    /// serialized, and an error from `f` rolls the graph back to its state before
    /// `f`. Not reentrant: calling `transact` inside `f` deadlocks.
    pub fn transact<T>(
        &self,
        f: impl FnOnce(&LodGraph) -> Result<T, LodError>,
    ) -> Result<T, LodError> {
        let _txn = self.txn_lock.lock();
        let checkpoint = self.create_checkpoint();
        match f(self) {
            Ok(value) => Ok(value),
            Err(error) => match self.rollback_checkpoint(&checkpoint) {
                Ok(()) => Err(error),
                Err(rollback) => Err(LodError::CheckpointRejected(format!(
                    "transaction failed ({error}) and its rollback was refused: {rollback}"
                ))),
            },
        }
    }

    /// Register a privilege bitflag for an agent.
    pub fn add_privilege(&self, agent_id: u64, privilege: u32) {
        let mut st = self.state.write();
        let list = st.privileges.entry(agent_id).or_default();
        if !list.contains(&privilege) {
            list.push(privilege);
        }
    }

    /// Explicitly revoke an entity ID in the cognitive graph.
    pub fn revoke_entity(&self, entity_id: u64) {
        self.state.write().revocations.insert(entity_id);
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

    fn node(label: &str, entity: u64) -> LodNode {
        LodNode::new(
            0,
            LodBand::Lod0Atomic,
            MixedCurvatureCoord::origin(),
            label,
            entity,
        )
    }

    fn pruned(graph: &LodGraph, id: u32) -> Vec<u32> {
        graph.cascade_prune_and_rollback(id).unwrap().pruned
    }

    #[test]
    fn axiom_survives_direct_and_cascading_prune() {
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
            graph.cascade_prune_and_rollback(axiom),
            Err(LodError::InvalidStateTransition(_))
        ));
        assert_eq!(pruned(&graph, source), vec![source]);
        assert_eq!(
            graph.get_node(axiom).unwrap().status,
            EpistemicStatus::Axiomatic
        );
        assert!(!graph.is_revoked(11));
        assert_eq!(
            graph.cascade_prune_and_rollback(99).unwrap_err(),
            LodError::NodeNotFound(99)
        );
    }

    #[test]
    fn ppr_omits_falsified_and_revoked_nodes() {
        let graph = LodGraph::new();
        let live = graph.add_node(node("live", 20)).unwrap();
        let false_node = graph
            .add_node(node("false", 21).with_status(EpistemicStatus::Falsified))
            .unwrap();
        let revoked = graph.add_node(node("revoked", 22)).unwrap();
        graph.revoke_entity(22);
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

        // Falsifying node 1 cascades to node 2.
        let outcome = graph.cascade_prune_and_rollback(id1).unwrap();
        assert_eq!(outcome.pruned, vec![id1, id2]);
        assert_eq!(outcome.revoked_entities, vec![1002, 1003]);

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
        graph.cascade_prune_and_rollback(id0).unwrap();
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
    fn test_validate_node_and_prune_propagation() {
        let graph = LodGraph::new();
        let id0 = graph.add_node(node("hypo_0", 3001)).unwrap();
        let id1 = graph.add_node(node("hypo_1", 3002)).unwrap();
        let id2 = graph.add_node(node("hypo_2", 3003)).unwrap();

        graph.add_edge(id0, id1, EdgeType::DependsOn, 1.0).unwrap();
        graph.add_edge(id1, id2, EdgeType::DependsOn, 1.0).unwrap();

        graph.validate_node(id0).unwrap();
        graph.validate_node(id1).unwrap();

        let deps: Vec<_> = graph.active_validated_dependencies().collect();
        assert!(deps.contains(&(3001, 3002)));

        // Falsifying node 0 cascades through validated node 1 to hypothesized node 2.
        let outcome = graph.cascade_prune_and_rollback(id0).unwrap();
        assert_eq!(outcome.pruned, vec![id0, id1, id2]);
        assert_eq!(outcome.retracted_dependencies, 1);
        assert!(graph.is_revoked(3001));
        assert!(graph.is_revoked(3002));
        assert!(graph.is_revoked(3003));
        assert!(graph.active_validated_dependencies().next().is_none());
    }

    /// Regression: the CSR pass used to read outgoing edges only, so a flushed
    /// edge INTO the validated node never became a validated dependency.
    #[test]
    fn validate_node_backfills_incoming_csr_edges() {
        let graph = LodGraph::new();
        let src = graph
            .add_node(node("src", 5001).with_status(EpistemicStatus::Validated))
            .unwrap();
        let dst = graph.add_node(node("dst", 5002)).unwrap();
        graph.add_edge(src, dst, EdgeType::DependsOn, 1.0).unwrap();
        graph.flush_edges_to_csr().unwrap();
        assert_eq!(graph.pending_edge_count(), 0);
        assert!(graph.active_validated_dependencies().next().is_none());

        graph.validate_node(dst).unwrap();
        let deps: Vec<_> = graph.active_validated_dependencies().collect();
        assert_eq!(deps, vec![(5001, 5002)]);
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
        graph.cascade_prune_and_rollback(a).unwrap();
        graph.add_privilege(9, 1);
        assert!(graph.is_revoked(1) && graph.is_revoked(2) && graph.is_revoked(3));

        graph.rollback_checkpoint(&checkpoint).unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.node_for_entity(3), None);
        assert_eq!(graph.pending_edge_count(), 1);
        assert!(Arc::ptr_eq(&graph.csr_snapshot(), &csr_before));
        assert!(!graph.is_revoked(1) && !graph.is_revoked(2));
        assert!(!graph.has_privilege(9, 1));
        assert_eq!(
            graph.get_node(a).unwrap().status,
            EpistemicStatus::Hypothesized
        );
        // Entity 3 is free again, and the restored pending edge flushes once.
        graph.add_node(node("c2", 3)).unwrap();
        assert_eq!(graph.flush_edges_to_csr().unwrap().csr_edges, 2);
    }

    #[test]
    fn prune_outcome_checkpoint_undoes_the_prune() {
        let graph = LodGraph::new();
        let a = graph
            .add_node(node("a", 1).with_status(EpistemicStatus::Validated))
            .unwrap();
        let b = graph
            .add_node(node("b", 2).with_status(EpistemicStatus::Validated))
            .unwrap();
        graph.add_edge(a, b, EdgeType::DependsOn, 1.0).unwrap();
        let outcome = graph.cascade_prune_and_rollback(a).unwrap();
        assert_eq!(outcome.pruned, vec![a, b]);
        assert!(graph.active_validated_dependencies().next().is_none());

        graph.rollback_checkpoint(&outcome.checkpoint).unwrap();
        assert!(!graph.is_revoked(1) && !graph.is_revoked(2));
        assert_eq!(
            graph.get_node(b).unwrap().status,
            EpistemicStatus::Validated
        );
        let deps: Vec<_> = graph.active_validated_dependencies().collect();
        assert_eq!(deps, vec![(1, 2)]);
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
}
