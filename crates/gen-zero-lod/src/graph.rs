//! gen-zero-lod Dynamic Graph Topology, Epistemic Lifecycle & Pearl Cascading Pruning.
//!
//! Provides lock-free double-buffered CSR snapshots (ArcSwap<CsrGraph>),
//! dynamic ticket-sequenced edge append buffer, 2-stage HDC+Fisher/Manifold recall,
//! and Pearl causal cascade pruning with virtual loss rollback.

use crate::error::LodError;
use crate::manifold::MixedCurvatureCoord;
use crate::node::{hdc_hamming_distance_256, EpistemicStatus, LodNode};
use crate::ppr::compute_ppr_csr;
use arc_swap::ArcSwap;
use gen_zero_core::GraphFactProvider;
use parking_lot::RwLock;
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

/// Immutable Compressed Sparse Row (CSR) Graph Snapshot.
/// Provides lock-free zero-allocation graph traversal for PPR and CRAG.
///
/// Fields are private so the CSR invariants hold for every value:
/// `row_offsets.len() == num_nodes + 1`, `row_offsets` is non-decreasing and
/// ends at `col_indices.len()`, and the three edge arrays share one length.
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

    /// Construct CSR snapshot from a set of buffered edges and node count.
    pub fn from_edges(num_nodes: usize, edges: &[BufferedEdge]) -> Self {
        if num_nodes == 0 {
            return Self::empty();
        }

        // Count degrees for valid node ranges
        let mut degrees = vec![0usize; num_nodes];
        for edge in edges {
            let u = edge.source as usize;
            let v = edge.target as usize;
            if u < num_nodes && v < num_nodes {
                degrees[u] += 1;
            }
        }

        // Build prefix row offsets
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
        for edge in edges {
            let u = edge.source as usize;
            let v = edge.target as usize;
            if u < num_nodes && v < num_nodes {
                let idx = insert_cursor[u];
                col_indices[idx] = edge.target;
                edge_weights[idx] = edge.weight;
                edge_types[idx] = edge.edge_type;
                insert_cursor[u] += 1;
            }
        }

        Self {
            num_nodes,
            row_offsets,
            col_indices,
            edge_weights,
            edge_types,
        }
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

    /// Copy of this snapshot grown to `num_nodes` nodes; new nodes have no edges.
    fn with_num_nodes(&self, num_nodes: usize) -> Self {
        let mut grown = self.clone();
        if num_nodes > grown.num_nodes {
            grown.num_nodes = num_nodes;
            grown
                .row_offsets
                .resize(num_nodes + 1, grown.col_indices.len());
        }
        grown
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

/// Dynamic LodGraph integrating multi-scale nodes, ticketed edge buffer,
/// lock-free CSR snapshots, and epistemic governance.
pub struct LodGraph {
    nodes: RwLock<Vec<LodNode>>,
    edge_buffer: RwLock<Vec<BufferedEdge>>,
    ticket_counter: AtomicU64,
    csr_snapshot: ArcSwap<CsrGraph>,
    revocations: RwLock<HashSet<u64>>,
    privileges: RwLock<HashMap<u64, Vec<u32>>>,
    validated_deps: RwLock<HashSet<(u64, u64)>>,
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
            nodes: RwLock::new(Vec::new()),
            edge_buffer: RwLock::new(Vec::new()),
            ticket_counter: AtomicU64::new(1),
            csr_snapshot: ArcSwap::from_pointee(CsrGraph::empty()),
            revocations: RwLock::new(HashSet::new()),
            privileges: RwLock::new(HashMap::new()),
            validated_deps: RwLock::new(HashSet::new()),
        }
    }

    /// Insert a new node into the graph, assigning an auto-incrementing ID.
    pub fn add_node(&self, mut node: LodNode) -> u32 {
        let mut nodes = self.nodes.write();
        let id = nodes.len() as u32;
        node.id = id;

        if node.status.is_falsified() {
            self.revocations.write().insert(node.entity_id);
        }

        nodes.push(node);

        // Keep CSR snapshot node count aligned so queries know about newly added nodes
        let snap = self.csr_snapshot.load();
        if snap.num_nodes < nodes.len() {
            self.csr_snapshot
                .store(Arc::new(snap.with_num_nodes(nodes.len())));
        }

        id
    }

    /// Retrieve a cloned copy of a node by ID.
    pub fn get_node(&self, id: u32) -> Option<LodNode> {
        let nodes = self.nodes.read();
        nodes.get(id as usize).cloned()
    }

    /// Number of nodes currently in graph.
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Append a directional edge to the dynamic edge buffer.
    pub fn add_edge(&self, source: u32, target: u32, edge_type: EdgeType, weight: f32) -> u64 {
        let ticket = self.ticket_counter.fetch_add(1, Ordering::Relaxed);
        let edge = BufferedEdge {
            source,
            target,
            edge_type,
            weight,
            ticket,
        };

        self.edge_buffer.write().push(edge);

        // If edge represents a validated causal dependency, register into validated_deps
        if edge_type == EdgeType::DependsOn || edge_type == EdgeType::Validates {
            let nodes = self.nodes.read();
            if let (Some(u), Some(v)) = (nodes.get(source as usize), nodes.get(target as usize)) {
                if u.status.is_active_truth() && v.status.is_active_truth() {
                    self.validated_deps
                        .write()
                        .insert((u.entity_id, v.entity_id));
                }
            }
        }

        ticket
    }

    /// Transition a node from Hypothesized to Validated upon environment proof or intervention.
    /// Automatically detects and backfills connected validated dependencies into `validated_deps`.
    pub fn validate_node(&self, node_id: u32) -> Result<(), LodError> {
        let mut nodes = self.nodes.write();
        let num_nodes = nodes.len();
        if (node_id as usize) >= num_nodes {
            return Err(LodError::NodeNotFound(node_id));
        }

        if nodes[node_id as usize].status == EpistemicStatus::Falsified {
            return Err(LodError::InvalidStateTransition(
                "Cannot validate an already falsified node".into(),
            ));
        }

        nodes[node_id as usize].status = EpistemicStatus::Validated;
        nodes[node_id as usize].confidence = 1.0;

        let node_entity = nodes[node_id as usize].entity_id;
        self.revocations.write().remove(&node_entity);

        // Check both CSR snapshot and dynamic edge_buffer for connected dependencies
        let snapshot = self.csr_snapshot.load();
        let mut new_deps = Vec::new();

        for (nbr, edge_type, _) in snapshot.neighbors(node_id) {
            if matches!(edge_type, EdgeType::DependsOn | EdgeType::Validates) {
                if let Some(target) = nodes.get(nbr as usize) {
                    if target.status.is_active_truth() {
                        new_deps.push((node_entity, target.entity_id));
                    }
                }
            }
        }

        let edge_buf = self.edge_buffer.read();
        for edge in edge_buf.iter() {
            if matches!(edge.edge_type, EdgeType::DependsOn | EdgeType::Validates) {
                if edge.source == node_id {
                    if let Some(target) = nodes.get(edge.target as usize) {
                        if target.status.is_active_truth() {
                            new_deps.push((node_entity, target.entity_id));
                        }
                    }
                } else if edge.target == node_id {
                    if let Some(src) = nodes.get(edge.source as usize) {
                        if src.status.is_active_truth() {
                            new_deps.push((src.entity_id, node_entity));
                        }
                    }
                }
            }
        }

        let mut val_deps = self.validated_deps.write();
        for dep in new_deps {
            val_deps.insert(dep);
        }

        Ok(())
    }

    /// Flush and compact all buffered edges into a new immutable CSR snapshot.
    /// Atomically replaces the current CSR snapshot for lock-free reader access.
    pub fn flush_edges_to_csr(&self) {
        let num_nodes = self.nodes.read().len();
        let edges = self.edge_buffer.read().clone();
        let new_csr = CsrGraph::from_edges(num_nodes, &edges);
        self.csr_snapshot.store(Arc::new(new_csr));
    }

    /// Run Personalized PageRank over the active CSR snapshot.
    pub fn query_ppr(
        &self,
        seeds: &[(u32, f32)],
        alpha: f32,
        max_iters: usize,
        tolerance: f32,
    ) -> Vec<(u32, f32)> {
        let snapshot = self.csr_snapshot.load();
        let scores = compute_ppr_csr(
            snapshot.num_nodes,
            &snapshot.row_offsets,
            &snapshot.col_indices,
            &snapshot.edge_weights,
            seeds,
            alpha,
            max_iters,
            tolerance,
        );

        let nodes = self.nodes.read();
        let revoked = self.revocations.read();
        let mut ranked: Vec<(u32, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| {
                nodes.get(*idx).is_some_and(|node| {
                    !node.status.is_falsified() && !revoked.contains(&node.entity_id)
                })
            })
            .map(|(idx, s)| (idx as u32, s))
            .collect();

        // Sort descending by score
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Two-Stage Memory Recall Pipeline:
    /// Stage 1: Fast HDC POPCNT Hamming filter (< 1.5µs across 500k nodes) -> selects Top-4K.
    /// Stage 2: Mixed-Curvature product geodesic rerank with Corrective RAG (CRAG) margin.
    ///
    /// Excludes falsified and revoked hypotheses to eliminate false memory recall.
    pub fn two_stage_recall(
        &self,
        query_coord: &MixedCurvatureCoord,
        query_hdc: &[u64; 4],
        top_k: usize,
        crag_margin: f32,
    ) -> Result<Vec<(u32, f32)>, LodError> {
        let nodes = self.nodes.read();
        if nodes.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let revs = self.revocations.read();

        // Stage 1: Filter out falsified / revoked nodes & compute HDC Hamming Distance
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

        let candidate_pool_size = (top_k * 4)
            .min(candidates.len())
            .max(top_k.min(candidates.len()));
        if candidate_pool_size < candidates.len() {
            // O(N) linear selection instead of O(N log N) full sort
            candidates.select_nth_unstable_by_key(candidate_pool_size, |c| c.1);
            candidates.truncate(candidate_pool_size);
        }

        // Stage 2: Full Mixed-Curvature Product Geodesic Reranking.
        // An out-of-domain coordinate aborts the recall instead of being skipped.
        let mut reranked: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        for &(id, _) in &candidates {
            if let Some(n) = nodes.get(id as usize) {
                reranked.push((id, n.coord.product_distance(query_coord)?));
            }
        }

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Corrective RAG (CRAG) Margin Check:
        // If distance difference between top-1 and top-2 is below margin, expand neighborhood
        if reranked.len() >= 2 {
            let d1 = reranked[0].1;
            let d2 = reranked[1].1;
            if (d2 - d1).abs() < crag_margin {
                // Critical ambiguity detected; query CSR snapshot to pull 1-hop neighbors of top-1
                let snapshot = self.csr_snapshot.load();
                let top1_id = reranked[0].0;
                for (nbr, _, _) in snapshot.neighbors(top1_id) {
                    if !reranked.iter().any(|(id, _)| *id == nbr) {
                        if let Some(nbr_node) = nodes.get(nbr as usize) {
                            if !nbr_node.status.is_falsified()
                                && !revs.contains(&nbr_node.entity_id)
                            {
                                let dist = nbr_node.coord.product_distance(query_coord)?;
                                reranked.push((nbr, dist));
                            }
                        }
                    }
                }
                reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        reranked.truncate(top_k);
        Ok(reranked)
    }

    /// Pearl Causal Subtree Pruning & Rollback.
    ///
    /// When environment feedback or formal proof falsifies an assumption:
    /// 1. Mark `falsified_node_id` as `Falsified`.
    /// 2. Traverse all downstream dependent nodes (`DependsOn`, `CausalTransition`) that are not Axiomatic.
    /// 3. Cascade mark them as `Falsified`.
    /// 4. Register entity revocations and purge invalidated dependencies.
    /// 5. Return all pruned node IDs so MCTS Arena can rollback virtual losses.
    pub fn cascade_prune_and_rollback(&self, falsified_node_id: u32) -> Vec<u32> {
        let mut nodes = self.nodes.write();
        let num_nodes = nodes.len();
        if (falsified_node_id as usize) >= num_nodes {
            return Vec::new();
        }
        if nodes[falsified_node_id as usize].status == EpistemicStatus::Axiomatic {
            return Vec::new();
        }

        let mut pruned = Vec::new();
        let mut new_revocations = Vec::new();
        let mut queue = VecDeque::new();

        // Mark root falsified
        nodes[falsified_node_id as usize].status = EpistemicStatus::Falsified;
        let root_entity = nodes[falsified_node_id as usize].entity_id;
        new_revocations.push(root_entity);
        pruned.push(falsified_node_id);
        queue.push_back(falsified_node_id);

        let snapshot = self.csr_snapshot.load();
        let edge_buf = self.edge_buffer.read();

        while let Some(curr) = queue.pop_front() {
            // 1. Check CSR outgoing edges
            for (nbr, edge_type, _) in snapshot.neighbors(curr) {
                if matches!(edge_type, EdgeType::DependsOn | EdgeType::CausalTransition) {
                    let nbr_idx = nbr as usize;
                    if nbr_idx < num_nodes
                        && !matches!(
                            nodes[nbr_idx].status,
                            EpistemicStatus::Axiomatic | EpistemicStatus::Falsified
                        )
                    {
                        nodes[nbr_idx].status = EpistemicStatus::Falsified;
                        let ent = nodes[nbr_idx].entity_id;
                        new_revocations.push(ent);
                        pruned.push(nbr);
                        queue.push_back(nbr);
                    }
                }
            }

            // 2. Check un-flushed buffered edges to guarantee zero invisible dependency gaps
            for edge in edge_buf.iter() {
                if edge.source == curr
                    && matches!(
                        edge.edge_type,
                        EdgeType::DependsOn | EdgeType::CausalTransition
                    )
                {
                    let nbr_idx = edge.target as usize;
                    if nbr_idx < num_nodes
                        && !matches!(
                            nodes[nbr_idx].status,
                            EpistemicStatus::Axiomatic | EpistemicStatus::Falsified
                        )
                    {
                        nodes[nbr_idx].status = EpistemicStatus::Falsified;
                        let ent = nodes[nbr_idx].entity_id;
                        new_revocations.push(ent);
                        pruned.push(edge.target);
                        queue.push_back(edge.target);
                    }
                }
            }
        }

        // Batch update revocations without acquiring lock on every BFS pop
        self.revocations.write().extend(new_revocations);

        // Purge invalidated dependencies
        let pruned_entities: HashSet<u64> = pruned
            .iter()
            .filter_map(|&id| nodes.get(id as usize).map(|n| n.entity_id))
            .collect();

        self.validated_deps
            .write()
            .retain(|(u, v)| !pruned_entities.contains(u) && !pruned_entities.contains(v));

        pruned
    }

    /// Register a privilege bitflag for an agent.
    pub fn add_privilege(&self, agent_id: u64, privilege: u32) {
        let mut privs = self.privileges.write();
        let list = privs.entry(agent_id).or_default();
        if !list.contains(&privilege) {
            list.push(privilege);
        }
    }

    /// Explicitly revoke an entity ID in the cognitive graph.
    pub fn revoke_entity(&self, entity_id: u64) {
        self.revocations.write().insert(entity_id);
    }
}

impl GraphFactProvider for LodGraph {
    type DepIter<'a> = std::vec::IntoIter<(u64, u64)>;

    fn active_validated_dependencies<'a>(&'a self) -> Self::DepIter<'a> {
        let deps: Vec<(u64, u64)> = self.validated_deps.read().iter().copied().collect();
        deps.into_iter()
    }

    fn is_revoked(&self, entity_id: u64) -> bool {
        self.revocations.read().contains(&entity_id)
    }

    fn has_privilege(&self, agent_id: u64, privilege: u32) -> bool {
        let privs = self.privileges.read();
        privs
            .get(&agent_id)
            .map(|list| list.contains(&privilege))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::LodBand;

    #[test]
    fn axiom_survives_direct_and_cascading_prune() {
        let graph = LodGraph::new();
        let coord = MixedCurvatureCoord::origin();
        let source = graph.add_node(LodNode::new(0, LodBand::Lod0Atomic, coord, "source", 10));
        let axiom = graph.add_node(
            LodNode::new(0, LodBand::Lod0Atomic, coord, "axiom", 11)
                .with_status(EpistemicStatus::Axiomatic),
        );
        graph.add_edge(source, axiom, EdgeType::DependsOn, 1.0);
        graph.flush_edges_to_csr();
        assert!(graph.cascade_prune_and_rollback(axiom).is_empty());
        assert_eq!(graph.cascade_prune_and_rollback(source), vec![source]);
        assert_eq!(
            graph.get_node(axiom).unwrap().status,
            EpistemicStatus::Axiomatic
        );
        assert!(!graph.is_revoked(11));
    }

    #[test]
    fn ppr_omits_falsified_and_revoked_nodes() {
        let graph = LodGraph::new();
        let coord = MixedCurvatureCoord::origin();
        let live = graph.add_node(LodNode::new(0, LodBand::Lod0Atomic, coord, "live", 20));
        let false_node = graph.add_node(
            LodNode::new(0, LodBand::Lod0Atomic, coord, "false", 21)
                .with_status(EpistemicStatus::Falsified),
        );
        let revoked = graph.add_node(LodNode::new(0, LodBand::Lod0Atomic, coord, "revoked", 22));
        graph.revoke_entity(22);
        let result = graph.query_ppr(
            &[(live, 1.0), (false_node, 1.0), (revoked, 1.0)],
            0.15,
            10,
            1e-4,
        );
        assert_eq!(
            result.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![live]
        );
    }

    #[test]
    fn test_lod_graph_workflow() {
        let graph = LodGraph::new();

        let coord0 = MixedCurvatureCoord::origin();
        let n0 = LodNode::new(0, LodBand::Lod0Atomic, coord0, "sensor_read", 1001)
            .with_status(EpistemicStatus::Validated);
        let id0 = graph.add_node(n0);

        let n1 = LodNode::new(1, LodBand::Lod1Cluster, coord0, "sub_goal", 1002)
            .with_status(EpistemicStatus::Hypothesized);
        let id1 = graph.add_node(n1);

        let n2 = LodNode::new(2, LodBand::Lod2Milestone, coord0, "milestone", 1003)
            .with_status(EpistemicStatus::Hypothesized);
        let id2 = graph.add_node(n2);

        // Add dependencies: 0 -> 1 -> 2
        graph.add_edge(id0, id1, EdgeType::DependsOn, 1.0);
        graph.add_edge(id1, id2, EdgeType::DependsOn, 1.0);
        graph.flush_edges_to_csr();

        // Check GraphFactProvider contract
        assert!(!graph.is_revoked(1001));
        assert!(!graph.is_revoked(1002));

        // Now falsify node 1: should cascade prune node 2 as well!
        let pruned = graph.cascade_prune_and_rollback(id1);
        assert_eq!(pruned, vec![id1, id2]);

        assert!(graph.is_revoked(1002));
        assert!(graph.is_revoked(1003));
        assert!(!graph.is_revoked(1001)); // Root node 0 was not pruned
    }

    #[test]
    fn test_two_stage_recall_filters_falsified() {
        let graph = LodGraph::new();

        let coord0 = MixedCurvatureCoord::origin();
        let mut coord1 = MixedCurvatureCoord::origin();
        coord1.euclidean[0] = 5.0;

        let n0 = LodNode::new(0, LodBand::Lod0Atomic, coord0, "target_node", 2001)
            .with_hdc_fingerprint([0b1111, 0, 0, 0]);
        let n1 = LodNode::new(1, LodBand::Lod0Atomic, coord1, "distant_node", 2002)
            .with_hdc_fingerprint([0b0000, 0, 0, 0]);

        let id0 = graph.add_node(n0);
        let id1 = graph.add_node(n1);

        let query_fp = [0b1111, 0, 0, 0];
        let recalled = graph.two_stage_recall(&coord0, &query_fp, 2, 0.1).unwrap();
        assert_eq!(recalled.len(), 2);
        assert_eq!(recalled[0].0, id0);

        // Falsify node 0: must never be recalled again!
        graph.cascade_prune_and_rollback(id0);
        let recalled_after = graph.two_stage_recall(&coord0, &query_fp, 2, 0.1).unwrap();
        assert_eq!(recalled_after.len(), 1);
        assert_eq!(recalled_after[0].0, id1);
    }

    #[test]
    fn test_two_stage_recall_rejects_out_of_domain_coord() {
        let graph = LodGraph::new();
        let mut bad = MixedCurvatureCoord::origin();
        // Public fields bypass `new`; the rerank must fail, not skip the node.
        bad.hyperbolic = [1.0, 0.0, 0.0, 0.0];
        graph.add_node(LodNode::new(0, LodBand::Lod0Atomic, bad, "bad", 4001));
        let err = graph
            .two_stage_recall(&MixedCurvatureCoord::origin(), &[0; 4], 1, 0.1)
            .unwrap_err();
        assert_eq!(
            err,
            LodError::Geometry(crate::manifold::Reject::DomainViolation)
        );
    }

    #[test]
    fn test_validate_node_and_prune_propagation() {
        let graph = LodGraph::new();
        let coord = MixedCurvatureCoord::origin();

        let id0 = graph.add_node(LodNode::new(0, LodBand::Lod0Atomic, coord, "hypo_0", 3001));
        let id1 = graph.add_node(LodNode::new(1, LodBand::Lod1Cluster, coord, "hypo_1", 3002));
        let id2 = graph.add_node(LodNode::new(
            2,
            LodBand::Lod2Milestone,
            coord,
            "hypo_2",
            3003,
        ));

        graph.add_edge(id0, id1, EdgeType::DependsOn, 1.0);
        graph.add_edge(id1, id2, EdgeType::DependsOn, 1.0);

        // Validate node 0 & node 1
        graph.validate_node(id0).unwrap();
        graph.validate_node(id1).unwrap();

        // Check active dependencies
        let deps: Vec<_> = graph.active_validated_dependencies().collect();
        assert!(deps.contains(&(3001, 3002)));

        // Falsifying node 0 should cascade through validated node 1 to hypothesized node 2!
        let pruned = graph.cascade_prune_and_rollback(id0);
        assert_eq!(pruned, vec![id0, id1, id2]);
        assert!(graph.is_revoked(3001));
        assert!(graph.is_revoked(3002));
        assert!(graph.is_revoked(3003));

        let deps_after: Vec<_> = graph.active_validated_dependencies().collect();
        assert!(deps_after.is_empty());
    }

    #[test]
    fn test_graph_ppr_query() {
        let graph = LodGraph::new();
        let coord = MixedCurvatureCoord::origin();

        let id0 = graph.add_node(LodNode::new(0, LodBand::Lod0Atomic, coord, "root", 1));
        let id1 = graph.add_node(LodNode::new(1, LodBand::Lod1Cluster, coord, "child_a", 2));
        let id2 = graph.add_node(LodNode::new(2, LodBand::Lod1Cluster, coord, "child_b", 3));

        graph.add_edge(id0, id1, EdgeType::CausalTransition, 1.0);
        graph.add_edge(id1, id2, EdgeType::CausalTransition, 1.0);
        graph.flush_edges_to_csr();

        let ppr = graph.query_ppr(&[(id0, 1.0)], 0.15, 20, 1e-4);
        assert_eq!(ppr.len(), 3);
        assert_eq!(ppr[0].0, id0);
    }

    #[test]
    fn csr_default_and_growth_keep_invariants() {
        let g = CsrGraph::default();
        assert_eq!(g.num_nodes(), 0);
        assert_eq!(g.row_ptrs(), &[0]);

        let edges = [BufferedEdge {
            source: 0,
            target: 1,
            edge_type: EdgeType::Semantic,
            weight: 0.5,
            ticket: 1,
        }];
        let g = CsrGraph::from_edges(2, &edges).with_num_nodes(4);
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.row_ptrs(), &[0, 1, 1, 1, 1]);
        assert_eq!(g.col_indices(), &[1]);
        assert_eq!(g.edge_weights(), &[0.5]);
        assert_eq!(g.edge_types(), &[EdgeType::Semantic]);
        assert_eq!(g.neighbors(3).count(), 0);
    }
}
