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
//! Confidence is the fixed point of `c = (1 - beta) pi + beta P c` over the
//! `DependsOn` / `CausalTransition` / `CoarseGrain` edges
//! (`evolve_epistemic_fixed_point`), a
//! `beta`-contraction in the max norm, so cycles converge to one answer. Evidence
//! enters through `falsify_node` and leaves through `retract_falsification`; the
//! next evolution moves every dependent accordingly, in either direction.
//!
//! `create_checkpoint` / `rollback_checkpoint` restore the whole mutable graph state
//! atomically. Nothing here touches a search tree: the planner's MCTS is sequential
//! and has no virtual loss (see `gen-zero-planner/src/config.rs`).

use crate::error::LodError;
use crate::manifold::{Epochs, GeometryParams, MixedCurvatureCoord, ProductManifold, Version};
use crate::node::{hdc_hamming_distance_256, EpistemicStatus, LodBand, LodNode, ZoomDirection};
use crate::ppr::compute_ppr_csr;
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
    /// Entities revoked through [`LodGraph::revoke_entity`]. A subset of
    /// `revocations` that no confidence evolution ever lifts.
    manual_revocations: HashSet<u64>,
    privileges: HashMap<u64, Vec<u32>>,
    validated_deps: HashSet<(u64, u64)>,
    /// Bumped by every CSR store and every rollback. A flush built against an
    /// older generation is discarded.
    generation: u64,
    /// Sorted, disjoint checkpoint sequence ranges `(after, upto]` whose state a
    /// rollback discarded. Checkpoints in them can no longer be restored.
    discarded: Vec<(u64, u64)>,
}

/// The node fields a graph method may change after insert. Label, coordinate,
/// fingerprint, entity and prior are fixed at insert.
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
/// CSR snapshot reference, pending edges, revocations, privileges and validated
/// dependencies.
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

/// Most steps one confidence evolution may take. A request whose bound
/// `k_max` is larger fails with [`LodError::FixedPointDiverged`] once it has used
/// this many.
pub const MAX_FIXED_POINT_STEPS: usize = 100_000;

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

/// Result of one [`LodGraph::evolve_epistemic_fixed_point`].
#[derive(Clone, Debug)]
pub struct FixedPointReport {
    pub beta: f32,
    pub tolerance: f32,
    pub theta_lo: f32,
    pub theta_hi: f32,
    /// Nodes in the iteration.
    pub nodes: usize,
    /// Nodes held at their prior by evidence: axioms at 1, refuted nodes at 0.
    pub pinned: usize,
    /// Dependency edges (`DependsOn` / `CausalTransition` / `CoarseGrain`, positive weight, into
    /// an unpinned node) the transition matrix was built from.
    pub dependency_edges: usize,
    /// The `k` of the first iterate with `||c^{k+1} - c^k||_inf < tolerance`.
    /// Never above `k_max`.
    pub iterations: usize,
    /// `ceil(ln(tolerance (1 - beta) / ||c^1 - c^0||_inf) / ln beta)`, or 0 when
    /// the first step is already below the tolerance.
    pub k_max: usize,
    /// `||c^1 - c^0||_inf`.
    pub initial_delta: f64,
    /// `||c^{k+1} - c^k||_inf` at the stop.
    pub residual: f64,
    /// A-posteriori bound on the distance from the committed confidences to the
    /// exact fixed point: `beta / (1 - beta) * residual`.
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
    /// State just before the evolution, taken under the same lock. Pass it to
    /// [`LodGraph::rollback_checkpoint`] to undo the evolution.
    pub checkpoint: GraphCheckpoint,
}

/// `P` of the confidence iteration, stored by dependent (row). A row with no
/// entries is the identity row: the node keeps its prior.
struct DependencyRows {
    offsets: Vec<usize>,
    sources: Vec<u32>,
    probabilities: Vec<f64>,
}

/// Outcome of [`iterate_to_fixed_point`].
struct FixedPointRun {
    confidences: Vec<f64>,
    iterations: usize,
    k_max: usize,
    initial_delta: f64,
    residual: f64,
}

/// One application of `T(c) = (1 - beta) prior + beta P c` into `next`. Returns
/// `||next - current||_inf`, or `None` when a value is not finite.
fn apply_step(
    rows: &DependencyRows,
    prior: &[f64],
    beta: f64,
    current: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    let mut delta = 0.0_f64;
    for (i, out) in next.iter_mut().enumerate() {
        let (start, end) = (rows.offsets[i], rows.offsets[i + 1]);
        *out = if start == end {
            // Identity row, written so that `current == prior` is reproduced exactly.
            current[i] + (1.0 - beta) * (prior[i] - current[i])
        } else {
            let inherited: f64 = rows.sources[start..end]
                .iter()
                .zip(&rows.probabilities[start..end])
                .map(|(&u, p)| p * current[u as usize])
                .sum();
            (1.0 - beta) * prior[i] + beta * inherited
        };
        let step = (*out - current[i]).abs();
        if !(out.is_finite() && step.is_finite()) {
            return None;
        }
        delta = delta.max(step);
    }
    Some(delta)
}

/// Iterate `c^{k+1} = (1 - beta) prior + beta P c^k` from `start` until
/// `||c^{k+1} - c^k||_inf < tolerance`.
///
/// `P` is row-stochastic, so the map contracts the max norm by `beta` and
/// `||c^{k+1} - c^k|| <= beta^k ||c^1 - c^0||`. The stop is therefore reached by
/// `k_max = ceil(ln(tolerance (1 - beta) / ||c^1 - c^0||) / ln beta)`. A run that
/// has not stopped at `min(k_max, max_steps)`, or meets a non-finite value, is
/// [`LodError::FixedPointDiverged`]: no partial vector is returned.
fn iterate_to_fixed_point(
    rows: &DependencyRows,
    prior: &[f64],
    start: &[f64],
    beta: f64,
    tolerance: f64,
    max_steps: usize,
) -> Result<FixedPointRun, LodError> {
    let mut current = start.to_vec();
    let mut next = vec![0.0; current.len()];
    let diverged = |iterations, k_max, residual| LodError::FixedPointDiverged {
        iterations,
        k_max,
        residual,
        tolerance,
    };
    let initial_delta = apply_step(rows, prior, beta, &current, &mut next)
        .ok_or_else(|| diverged(0, 0, f64::NAN))?;
    let k_max = if initial_delta < tolerance {
        0
    } else {
        // Both logarithms are negative, so the ratio is positive. The cast
        // saturates; a bound that large can only fail against `max_steps`.
        ((tolerance * (1.0 - beta) / initial_delta).ln() / beta.ln()).ceil() as usize
    };
    let limit = k_max.min(max_steps);
    let (mut iterations, mut residual) = (0, initial_delta);
    while residual >= tolerance {
        if iterations == limit {
            return Err(diverged(iterations, k_max, residual));
        }
        std::mem::swap(&mut current, &mut next);
        iterations += 1;
        residual = apply_step(rows, prior, beta, &current, &mut next)
            .ok_or_else(|| diverged(iterations, k_max, f64::NAN))?;
    }
    Ok(FixedPointRun {
        confidences: next,
        iterations,
        k_max,
        initial_delta,
        residual,
    })
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

/// Row-normalise the positive-weight confidence edges into each unpinned node.
fn dependency_rows(
    nodes: &[LodNode],
    snapshot: &CsrGraph,
    pending: &[BufferedEdge],
) -> DependencyRows {
    let n = nodes.len();
    let inherits = |&(_, target, edge_type, weight): &(u32, u32, EdgeType, f32)| {
        carries_confidence(edge_type) && weight > 0.0 && !is_pinned(&nodes[target as usize])
    };
    let mut offsets = vec![0usize; n + 1];
    let mut totals = vec![0.0_f64; n];
    for (_, target, _, weight) in all_edges(snapshot, pending).filter(inherits) {
        offsets[target as usize + 1] += 1;
        totals[target as usize] += f64::from(weight);
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets[..n].to_vec();
    let mut sources = vec![0u32; offsets[n]];
    let mut probabilities = vec![0.0_f64; offsets[n]];
    for (source, target, _, weight) in all_edges(snapshot, pending).filter(inherits) {
        let at = &mut cursor[target as usize];
        sources[*at] = source;
        probabilities[*at] = f64::from(weight) / totals[target as usize];
        *at += 1;
    }
    DependencyRows {
        offsets,
        sources,
        probabilities,
    }
}

/// The validated dependencies the current statuses and edges imply.
fn validated_dependencies(
    nodes: &[LodNode],
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
            state: RwLock::new(GraphState::default()),
            csr_snapshot: ArcSwap::from_pointee(CsrGraph::empty()),
            ticket_counter: AtomicU64::new(1),
            checkpoint_seq: AtomicU64::new(0),
            flush_lock: Mutex::new(()),
            txn_lock: Mutex::new(()),
        })
    }

    /// The geometry every distance of this graph uses.
    pub fn geometry(&self) -> GeometryParams {
        self.manifold.params()
    }

    /// Distance between two coordinates under this graph's geometry.
    fn distance(&self, a: &MixedCurvatureCoord, b: &MixedCurvatureCoord) -> Result<f32, LodError> {
        let (alphas, c, r) = self.metric;
        a.product_distance_with_params(b, alphas, c, r)
    }

    /// Insert a node and return its id. Refuses a coordinate outside this
    /// graph's geometry, a prior outside [0, 1], a posterior that differs from
    /// the prior, a refutation mark on a node that is not `Falsified`, an unknown
    /// parent, and an entity id that already has a node: the gate addresses
    /// nodes by entity.
    ///
    /// An `Axiomatic` node gets confidence 1. A `Falsified` node is refuted by
    /// evidence: confidence 0, entity revoked.
    pub fn add_node(&self, node: LodNode) -> Result<u32, LodError> {
        let mut st = self.state.write();
        self.insert_node(&mut st, node)
    }

    /// [`Self::add_node`] under a held write lock.
    fn insert_node(&self, st: &mut GraphState, mut node: LodNode) -> Result<u32, LodError> {
        node.coord.to_point(&self.manifold)?;
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
    /// nonnegative. A `CoarseGrain` edge must run from a finer band to a strictly
    /// coarser one. The critical section is O(1): no history is copied.
    pub fn add_edge(
        &self,
        source: u32,
        target: u32,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<u64, LodError> {
        let mut st = self.state.write();
        self.push_edge(&mut st, source, target, edge_type, weight)
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
        // Taken under the lock so buffer order is ticket order.
        let ticket = self.ticket_counter.fetch_add(1, Ordering::Relaxed);
        st.edge_buffer.push(BufferedEdge {
            source,
            target,
            edge_type,
            weight,
            ticket,
        });
        if states_dependency(edge_type) {
            let (u, v) = (&st.nodes[source as usize], &st.nodes[target as usize]);
            if u.status.is_active_truth() && v.status.is_active_truth() {
                let dep = (u.entity_id, v.entity_id);
                st.validated_deps.insert(dep);
            }
        }
        Ok(ticket)
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

    /// Move one node one band up or down with [`LodBand::zoom_out`] /
    /// [`LodBand::zoom_in`] and return the new band. Refused past `Lod3Systemic`
    /// or below `Lod0Atomic`, and when the move would break the coarse-grain
    /// order: every member of the node (source of a `CoarseGrain` edge into it)
    /// must stay strictly finer and every summary it belongs to strictly
    /// coarser. The coordinate is not moved.
    pub fn zoom_node(&self, node_id: u32, direction: ZoomDirection) -> Result<LodBand, LodError> {
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
    /// geometry and an entity that already has a node. A flush failure is
    /// returned after the summary and its pending edges are inserted; run the
    /// call inside [`Self::transact`] to roll that back.
    pub fn coarse_grain_cluster(
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
        self.flush_edges_to_csr()?;
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
    /// Stage 2: product geodesic rerank of those candidates under this graph's
    /// geometry (curvature, radius and the three metric weights), with a
    /// Corrective RAG (CRAG) margin: when the top two are closer than
    /// `crag_margin`, the 1-hop CSR neighbors of the top one join the rerank.
    ///
    /// Excludes falsified and revoked nodes. `top_k` must be at least 1 and
    /// `crag_margin` finite and nonnegative; a coordinate outside this graph's
    /// geometry is an error.
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
        query_coord.to_point(&self.manifold)?;
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
            reranked.push((id, self.distance(&nodes[id as usize].coord, query_coord)?));
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
                    reranked.push((nbr, self.distance(&nbr_node.coord, query_coord)?));
                }
            }
            reranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        }

        reranked.truncate(top_k);
        Ok(reranked)
    }

    /// Record direct evidence against a node: it becomes `Falsified` and refuted,
    /// its confidence is pinned to 0, its entity is revoked and every validated
    /// dependency touching it is retracted. Dependents move at the next
    /// [`Self::evolve_epistemic_fixed_point`]. Repeating the call changes nothing.
    /// Returns the number of validated dependencies retracted. An unknown node or
    /// an axiom is an error.
    pub fn falsify_node(&self, node_id: u32) -> Result<usize, LodError> {
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
    /// A node that is not refuted is an error.
    pub fn retract_falsification(&self, node_id: u32) -> Result<(), LodError> {
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
        node.refuted = false;
        node.status = EpistemicStatus::Hypothesized;
        node.confidence = node.prior;
        let entity = node.entity_id;
        if !st.manual_revocations.contains(&entity) {
            st.revocations.remove(&entity);
        }
        Ok(())
    }

    /// Evolve every node's confidence to the fixed point of
    ///
    /// `c^{k+1} = (1 - beta) pi + beta P c^k`, `0 < beta < 1`,
    ///
    /// and move statuses by hysteresis on the result.
    ///
    /// - `P[v][u]` is the weight of the `DependsOn` / `CausalTransition` /
    ///   `CoarseGrain` edges
    ///   `u -> v` (`v` depends on `u`) divided by the total such weight into `v`,
    ///   over the CSR snapshot and the pending buffer. A node with no positive
    ///   weight coming in, an axiom and a refuted node get the identity row, so
    ///   `P` is row-stochastic and the map contracts the max norm by `beta`:
    ///   cycles converge to the one fixed point.
    /// - `pi` is 1 for an axiom, 0 for a refuted node (both held exactly) and the
    ///   node's prior otherwise. The iteration starts at `pi`, so the result
    ///   depends only on priors, evidence, edges and the four arguments: undoing
    ///   a change of evidence and evolving again reproduces the earlier
    ///   confidences bit for bit.
    /// - The stop `||c^{k+1} - c^k||_inf < tolerance` is reached by
    ///   `k_max = ceil(ln(tolerance (1 - beta) / ||c^1 - c^0||_inf) / ln beta)`.
    ///   If it is not, or `k_max` exceeds [`MAX_FIXED_POINT_STEPS`] and the run
    ///   uses them all, the result is [`LodError::FixedPointDiverged`] and the
    ///   graph is unchanged.
    /// - Hysteresis: confidence below `theta_lo` makes a node `Falsified` and
    ///   revokes its entity; above `theta_hi` makes it `Validated` and lifts that
    ///   revocation (never one made by [`Self::revoke_entity`]); in between the
    ///   status is kept. Validated dependencies are rebuilt from the new statuses.
    ///
    /// Arguments outside `0 < beta < 1`, `tolerance > 0`,
    /// `0 < theta_lo < theta_hi < 1` are `InvalidQuery`.
    pub fn evolve_epistemic_fixed_point(
        &self,
        beta: f32,
        tolerance: f32,
        theta_lo: f32,
        theta_hi: f32,
    ) -> Result<FixedPointReport, LodError> {
        self.evolve_epistemic_fixed_point_within(
            beta,
            tolerance,
            theta_lo,
            theta_hi,
            MAX_FIXED_POINT_STEPS,
        )
    }

    /// [`Self::evolve_epistemic_fixed_point`] with a caller step budget: the run
    /// may take `min(k_max, max_steps, MAX_FIXED_POINT_STEPS)` steps.
    pub fn evolve_epistemic_fixed_point_within(
        &self,
        beta: f32,
        tolerance: f32,
        theta_lo: f32,
        theta_hi: f32,
        max_steps: usize,
    ) -> Result<FixedPointReport, LodError> {
        let bad = |detail: String| Err(LodError::InvalidQuery(detail));
        if !(beta.is_finite() && beta > 0.0 && beta < 1.0) {
            return bad(format!("beta must satisfy 0 < beta < 1, got {beta}"));
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
        let checkpoint = self.capture(&guard);
        let snapshot = self.csr_snapshot.load_full();
        let st = &mut *guard;

        let rows = dependency_rows(&st.nodes, &snapshot, &st.edge_buffer);
        let prior: Vec<f64> = st
            .nodes
            .iter()
            .map(|n| match (n.status, n.refuted) {
                (EpistemicStatus::Axiomatic, _) => 1.0,
                (_, true) => 0.0,
                _ => f64::from(n.prior),
            })
            .collect();
        let run = iterate_to_fixed_point(
            &rows,
            &prior,
            &prior,
            f64::from(beta),
            f64::from(tolerance),
            max_steps.min(MAX_FIXED_POINT_STEPS),
        )?;

        // Converged: commit. Nothing above this line changed the graph.
        let mut transitions = Vec::new();
        let mut revoked_entities = Vec::new();
        let mut reinstated_entities = Vec::new();
        let mut pinned = 0;
        for (node, &c) in st.nodes.iter_mut().zip(&run.confidences) {
            let confidence = c.clamp(0.0, 1.0) as f32;
            node.confidence = confidence;
            if is_pinned(node) {
                pinned += 1;
                continue;
            }
            let from = node.status;
            let to = if confidence < theta_lo {
                EpistemicStatus::Falsified
            } else if confidence > theta_hi {
                EpistemicStatus::Validated
            } else {
                from
            };
            if to == from {
                continue;
            }
            node.status = to;
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

        let beta64 = f64::from(beta);
        Ok(FixedPointReport {
            beta,
            tolerance,
            theta_lo,
            theta_hi,
            nodes: st.nodes.len(),
            pinned,
            dependency_edges: rows.sources.len(),
            iterations: run.iterations,
            k_max: run.k_max,
            initial_delta: run.initial_delta,
            residual: run.residual,
            error_bound: beta64 / (1.0 - beta64) * run.residual,
            transitions,
            revoked_entities,
            reinstated_entities,
            retracted_dependencies,
            added_dependencies,
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
            node_states: st.nodes.iter().map(NodeMutable::of).collect(),
            csr: self.csr_snapshot.load_full(),
            edge_buffer: st.edge_buffer.clone(),
            revocations: st.revocations.clone(),
            manual_revocations: st.manual_revocations.clone(),
            privileges: st.privileges.clone(),
            validated_deps: st.validated_deps.clone(),
        }
    }

    /// Restore `checkpoint` atomically: nodes added after it are removed (their
    /// ids become free again), statuses, confidences, refutation marks, bands,
    /// parents, CSR
    /// snapshot, pending edges, revocations, privileges and validated dependencies return to its values.
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
        for (node, state) in st.nodes.iter_mut().zip(&checkpoint.node_states) {
            state.restore(node);
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

    /// Explicitly revoke an entity ID in the cognitive graph. No confidence
    /// evolution and no evidence retraction lifts this revocation.
    pub fn revoke_entity(&self, entity_id: u64) {
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
        // Edges of other types carry no confidence.
        graph.add_edge(b, a, EdgeType::Semantic, 50.0).unwrap();
        graph.add_edge(c, b, EdgeType::Falsifies, 50.0).unwrap();

        let report = evolve(&graph, 0.2, 0.8);
        assert_eq!((report.nodes, report.pinned), (4, 1));
        assert_eq!(report.dependency_edges, 4);

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
        let runs: Vec<FixedPointRun> = {
            let st = graph.state.read();
            let rows = dependency_rows(&st.nodes, &graph.csr_snapshot(), &st.edge_buffer);
            [
                prior.clone(),
                vec![0.0; 3],
                vec![1.0; 3],
                vec![1.0, 0.0, 0.37],
            ]
            .iter()
            .map(|start| {
                iterate_to_fixed_point(&rows, &prior, start, beta, tol, usize::MAX).unwrap()
            })
            .collect()
        };
        let bound = 2.0 * tol * beta / (1.0 - beta);
        for run in &runs {
            assert!(run.iterations <= run.k_max);
            let c = &run.confidences;
            for i in 0..3 {
                assert!((c[i] - runs[0].confidences[i]).abs() <= bound);
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
            .evolve_epistemic_fixed_point_within(BETA, TOL, 0.3, 0.6, 3)
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
        graph.revoke_entity(5);
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
        graph.add_privilege(9, 1);
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
}
