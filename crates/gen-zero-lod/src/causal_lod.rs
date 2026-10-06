//! Causal AND/OR DAG on the four Lod bands, and the closure potential the
//! causal-triad sampler steers by.
//!
//! The planner owns the DAG (`gen_zero_planner::triad::CausalDag`); this crate
//! sits below the planner, so the DAG reaches it through the bitmask view
//! [`CausalMaskSource`].
//!
//! What acts on a decision: exactly two bitmask computations over the causal
//! topology. Neither uses any geometry.
//!
//! 1. Mask derivation for the fail-closed pre-check
//!    ([`CausalLod::check_feasible`], run before any plan is sampled). The
//!    checkpoints are the actions every plan to the target must complete
//!    (removing one makes the target unreachable); the hazard barriers are the
//!    policy-blocked actions in the goal cone. The target must be reachable
//!    with the barriers removed, and `time_used + sum(cost of checkpoints and
//!    target)` is a lower bound on the finish time, so above the budget the
//!    request is refused with that reason instead of sampling into an empty
//!    gate.
//! 2. The closure potential [`CausalPotential`]: `Phi(s) = -(g(s) + h(s))`
//!    with `h` the greedy feasible closure cost to the target. The triad
//!    sampler biases every draw by `alpha * dPhi`. This is the only
//!    implementation of the closure; it is heap-free (fixed arrays), both to
//!    build and to query. Its effect is measured by one paired test,
//!    `energy_paired_convergence_alpha2_vs_native_flow_k4` in
//!    `gen-zero-planner/tests/causal_triad_tests.rs`: 40 fixture graphs, K = 4
//!    samples, one shard, same seeds for alpha 2 and alpha 0, sign test on the
//!    discordant pairs. That is the whole evidence; nothing beyond that
//!    fixture set is claimed.
//!
//! The band layout [`CausalLod`] records the same structure on a fresh
//! [`LodGraph`]:
//! - `Lod0Atomic`: one node per action. An AND parent gives a `DependsOn` edge
//!   (a hard prerequisite), an OR parent a `CausalTransition` edge (one
//!   enabling alternative).
//! - `Lod1Cluster`: one node per local precondition closure: the parent set of
//!   an OR action (`OrAlternatives`) and of an AND action with two or more
//!   parents (`AndPreconditions`). Members link up by `CoarseGrain`.
//! - `Lod2Milestone`: one node per checkpoint and per hazard barrier (item 1).
//! - `Lod3Systemic`: the target with its budget. Every cone cluster, every
//!   milestone and the target action coarse-grain into it.
//!
//! Observable metric only: `d_H`. Placement in the Poincare ball (the graph's
//! `H^4` block) follows the band rule of [`band_from_scale`]: Lod3 sits nearest
//! the origin, Lod0 nearest the boundary. Inside Lod0 an action sits deeper the
//! more causal hops separate it from the target, so
//! [`CausalLod::goal_distance`], the hyperbolic geodesic distance `d_H` from an
//! action to the Lod3 node, grows with causal depth. The triad writes `d_H` of
//! the committed plan into its report (`goal_distance_used_in_choice: false`)
//! and nothing else reads it: it takes no part in the pre-check, sampling,
//! gating or ranking, and no evaluation shows it would help if it did.

use crate::error::LodError;
use crate::graph::{EdgeType, LodGraph};
use crate::manifold::{kernel, MixedCurvatureCoord};
use crate::node::{band_scale_width, max_chart_depth, LodBand, LodNode};
use thiserror::Error;

/// Bitmask width: a causal DAG holds at most this many actions.
pub const MAX_CAUSAL_ATOMS: usize = 64;

/// Entity ids of the synthetic nodes (clusters, milestones, systemic) start
/// here, above every `u32` action id, so they can never collide with an action.
pub const SYNTHETIC_ENTITY_BASE: u64 = 1 << 40;

const ENTITY_CLUSTER: u64 = SYNTHETIC_ENTITY_BASE;
const ENTITY_CHECKPOINT: u64 = SYNTHETIC_ENTITY_BASE + (1 << 32);
const ENTITY_BARRIER: u64 = SYNTHETIC_ENTITY_BASE + (2 << 32);
const ENTITY_SYSTEMIC: u64 = SYNTHETIC_ENTITY_BASE + (3 << 32);

/// Read-only bitmask view of a validated AND/OR causal DAG. Node `i`'s parents
/// are the set bits of `atom_parent_mask(i)`. The implementer guarantees
/// acyclicity and that `topo_order` lists every node once, parents first;
/// [`CausalPotential::new`] and [`CausalLod::build`] still check the shape and
/// refuse a view that breaks it.
pub trait CausalMaskSource {
    fn atom_count(&self) -> usize;
    fn target_atom(&self) -> usize;
    fn budget(&self) -> u32;
    fn topo_order(&self) -> &[usize];
    fn atom_parent_mask(&self, i: usize) -> u64;
    fn atom_is_or(&self, i: usize) -> bool;
    /// Nominal time cost, >= 1.
    fn atom_cost(&self, i: usize) -> u32;
    /// Immediate value on completion; finite.
    fn atom_value(&self, i: usize) -> f64;
    /// Action id, used as the Lod0 node's entity id.
    fn atom_action(&self, i: usize) -> u32;
}

#[derive(Error, Debug, Clone, PartialEq)]
pub enum CausalLodError {
    #[error("invalid causal DAG view: {0}")]
    Invalid(String),
    #[error("causal Lod graph refused: {0}")]
    Graph(#[from] LodError),
    #[error(
        "causal target {target} is unreachable once the hazard barriers {barriers:?} are removed"
    )]
    TargetUnreachable { target: u32, barriers: Vec<u32> },
    #[error(
        "budget {budget} cannot cover the mandatory checkpoints: time_used {time_used} + \
         floor {floor} (checkpoints {checkpoints:?} and the target) exceeds it"
    )]
    BudgetInsufficient {
        budget: u32,
        time_used: u32,
        floor: u64,
        checkpoints: Vec<u32>,
    },
}

/// Ascending set-bit positions of a mask.
#[derive(Clone)]
pub struct BitIter(pub u64);

impl Iterator for BitIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let i = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1;
        Some(i)
    }
}

fn bit(i: usize) -> u64 {
    1_u64 << i
}

/// Shape checks shared by both builders. Heap-free on success.
fn check_view<S: CausalMaskSource + ?Sized>(src: &S) -> Result<(), CausalLodError> {
    let n = src.atom_count();
    if n == 0 || n > MAX_CAUSAL_ATOMS {
        return Err(CausalLodError::Invalid(format!(
            "{n} actions; 1..={MAX_CAUSAL_ATOMS} are accepted"
        )));
    }
    if src.target_atom() >= n {
        return Err(CausalLodError::Invalid(format!(
            "target index {} outside the {n}-action DAG",
            src.target_atom()
        )));
    }
    let all = if n == MAX_CAUSAL_ATOMS {
        u64::MAX
    } else {
        bit(n) - 1
    };
    let topo = src.topo_order();
    if topo.len() != n {
        return Err(CausalLodError::Invalid(format!(
            "topological order lists {} of {n} actions",
            topo.len()
        )));
    }
    let mut seen = 0_u64;
    for &v in topo {
        if v >= n || seen & bit(v) != 0 {
            return Err(CausalLodError::Invalid(format!(
                "topological order repeats or overflows at index {v}"
            )));
        }
        let pm = src.atom_parent_mask(v);
        if pm & !all != 0 || pm & !seen != 0 {
            return Err(CausalLodError::Invalid(format!(
                "action index {v} has a parent outside the DAG or after it in the order"
            )));
        }
        if src.atom_cost(v) < 1 || !src.atom_value(v).is_finite() {
            return Err(CausalLodError::Invalid(format!(
                "action index {v} needs cost >= 1 and a finite value"
            )));
        }
        seen |= bit(v);
    }
    Ok(())
}

/// Causal potential `Phi(s) = -(paid + h(done))` over one DAG, with per-node
/// cost `c[a] = cost[a] - value[a]`: the loss the triad gate's arbiter ranks by
/// (it maximises `net_reward = sum(value) - time`).
///
/// `closure(done)` is the cost of one feasible completion set: the target plus,
/// for an AND node every not-done parent's set, for an OR node with no done
/// parent the parent set of least cost (ties: smaller bitmask). This is the
/// Lod1 reading of the DAG: an `AndPreconditions` cluster costs the union of
/// its members, an `OrAlternatives` cluster its cheapest member. Port of the
/// research `CausalBounds.closure`. All state lives in fixed-size arrays:
/// neither construction nor [`Self::closure`] touches the heap.
#[derive(Clone, Debug)]
pub struct CausalPotential {
    /// Goal-cone nodes in topological order; only `order[..len]` is used.
    order: [u8; MAX_CAUSAL_ATOMS],
    len: usize,
    step: [f64; MAX_CAUSAL_ATOMS],
    parent_mask: [u64; MAX_CAUSAL_ATOMS],
    or_mask: u64,
    cone: u64,
    target: usize,
}

impl CausalPotential {
    pub fn new<S: CausalMaskSource + ?Sized>(src: &S) -> Result<Self, CausalLodError> {
        check_view(src)?;
        // Goal cone by one reverse-topological sweep over the parent masks.
        let topo = src.topo_order();
        let mut cone = bit(src.target_atom());
        for &v in topo.iter().rev() {
            if cone & bit(v) != 0 {
                cone |= src.atom_parent_mask(v);
            }
        }
        let mut pot = Self {
            order: [0; MAX_CAUSAL_ATOMS],
            len: 0,
            step: [0.0; MAX_CAUSAL_ATOMS],
            parent_mask: [0; MAX_CAUSAL_ATOMS],
            or_mask: 0,
            cone,
            target: src.target_atom(),
        };
        for i in 0..src.atom_count() {
            // Both finite (checked above), so is the difference.
            pot.step[i] = f64::from(src.atom_cost(i)) - src.atom_value(i);
            pot.parent_mask[i] = src.atom_parent_mask(i);
            pot.or_mask |= u64::from(src.atom_is_or(i)) << i;
        }
        // Any topological order works: each node's set depends on its parents only.
        for &v in topo {
            if cone & bit(v) != 0 {
                pot.order[pot.len] = v as u8;
                pot.len += 1;
            }
        }
        Ok(pot)
    }

    /// The target and its ancestors, as a bitmask.
    pub fn cone(&self) -> u64 {
        self.cone
    }

    /// Per-node cost `cost[a] - value[a]`.
    pub fn step_cost(&self, a: usize) -> f64 {
        self.step[a]
    }

    fn mask_cost(&self, m: u64) -> f64 {
        BitIter(m).map(|i| self.step[i]).sum()
    }

    /// Greedy feasible closure cost from done-set `done` to the target. Bits
    /// outside the goal cone never change the result.
    pub fn closure(&self, done: u64) -> f64 {
        let mut set = [0_u64; MAX_CAUSAL_ATOMS];
        let mut set_cost = [0.0_f64; MAX_CAUSAL_ATOMS];
        for &v in &self.order[..self.len] {
            let v = usize::from(v);
            if done & bit(v) != 0 {
                continue; // set[v] and set_cost[v] stay 0
            }
            let pm = self.parent_mask[v];
            let mut s = bit(v);
            if self.or_mask & bit(v) != 0 {
                if pm != 0 && pm & done == 0 {
                    // Cheapest parent set; ties by the smaller bitmask, as in
                    // the research `min(..., key=(mask_cost, mask))`.
                    let mut best: Option<usize> = None;
                    for p in BitIter(pm) {
                        let better = best.is_none_or(|b| {
                            set_cost[p]
                                .total_cmp(&set_cost[b])
                                .then(set[p].cmp(&set[b]))
                                .is_lt()
                        });
                        if better {
                            best = Some(p);
                        }
                    }
                    if let Some(b) = best {
                        s |= set[b];
                    }
                }
            } else {
                for p in BitIter(pm) {
                    s |= set[p];
                }
            }
            set[v] = s;
            set_cost[v] = self.mask_cost(s);
        }
        set_cost[self.target]
    }

    /// `Phi(s') - Phi(s)` for completing node `a` from done-set `done`.
    pub fn delta_phi(&self, done: u64, a: usize) -> f64 {
        -(self.step[a] + self.closure(done | bit(a)) - self.closure(done))
    }

    /// `alpha * dPhi(s, a)` for every allowed node into `out[a]`, sharing the
    /// one `h(s)` evaluation across them. A non-finite bias is an error, never
    /// clamped.
    pub fn bias_into(
        &self,
        done: u64,
        allowed: u64,
        alpha: f64,
        out: &mut [f64; MAX_CAUSAL_ATOMS],
    ) -> Result<(), CausalLodError> {
        let h0 = self.closure(done);
        for a in BitIter(allowed) {
            let b = alpha * -(self.step[a] + self.closure(done | bit(a)) - h0);
            if !b.is_finite() {
                return Err(CausalLodError::Invalid(format!(
                    "energy bias of node {a} is not finite ({b})"
                )));
            }
            out[a] = b;
        }
        Ok(())
    }
}

/// Live planning context the Lod2 layer is computed under.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CausalLodContext {
    /// Already-completed actions.
    pub done: u64,
    pub time_used: u32,
    /// Actions forbidden at every step (policy hard stops).
    pub blocked: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ClusterKind {
    /// Parents of an OR action: one completed member enables it.
    OrAlternatives,
    /// Parents of an AND action with two or more parents: all must complete.
    AndPreconditions,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalCluster {
    pub kind: ClusterKind,
    /// Action index whose precondition this cluster is.
    pub owner: usize,
    pub members: u64,
    pub node: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MilestoneKind {
    /// Every plan to the target completes this action.
    Checkpoint,
    /// A policy-blocked action in the goal cone.
    HazardBarrier,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalMilestone {
    pub kind: MilestoneKind,
    pub atom: usize,
    pub node: u32,
}

/// Counts of what [`CausalLod::build`] placed, for the decision report.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalLodSummary {
    pub atoms: usize,
    pub or_clusters: usize,
    pub and_clusters: usize,
    pub checkpoints: Vec<u32>,
    pub hazard_barriers: Vec<u32>,
    /// `sum(cost)` of the not-done checkpoints and the target: the least
    /// nominal time any plan still needs.
    pub mandatory_floor: u64,
    pub graph_nodes: usize,
    pub graph_edges: usize,
}

/// A causal DAG laid out on the four Lod bands of its own [`LodGraph`]. The
/// graph is private to one planning call; it is never merged into a live
/// knowledge graph (whose entity ids are action ids too, and whose
/// `planning_prior` walks `CausalTransition` edges).
pub struct CausalLod {
    graph: LodGraph,
    atom_node: Vec<u32>,
    atom_action: Vec<u32>,
    atom_cost: Vec<u32>,
    clusters: Vec<CausalCluster>,
    milestones: Vec<CausalMilestone>,
    systemic: u32,
    target: usize,
    budget: u32,
    cone: u64,
    ctx: CausalLodContext,
    /// `false` when the target cannot be reached around the barriers.
    reachable_without_barriers: bool,
    checkpoints: u64,
    barriers: u64,
}

/// Monotone closure: every action completable from `done` without touching
/// `excluded`. O(n) passes of O(n) each; n <= 64.
fn reach<S: CausalMaskSource + ?Sized>(src: &S, done: u64, excluded: u64) -> u64 {
    let mut have = done;
    loop {
        let mut grew = false;
        for v in 0..src.atom_count() {
            if have & bit(v) != 0 || excluded & bit(v) != 0 {
                continue;
            }
            let pm = src.atom_parent_mask(v);
            let ok = if pm == 0 {
                true
            } else if src.atom_is_or(v) {
                have & pm != 0
            } else {
                have & pm == pm
            };
            if ok {
                have |= bit(v);
                grew = true;
            }
        }
        if !grew {
            return have;
        }
    }
}

/// Coordinate at coarse-graining scale `scale` along unit direction `dir`, on
/// the ball of curvature `-c`. Inverse of `scale_from_depth`.
fn coord_at_scale(scale: f64, dir: [f64; 4], c: f64) -> Result<MixedCurvatureCoord, LodError> {
    let rho = (max_chart_depth() - scale).max(0.0);
    let r = (rho / 2.0).tanh() / c.sqrt();
    let h = dir.map(|d| (d * r) as f32);
    MixedCurvatureCoord::with_curvature(h, [1.0, 0.0, 0.0, 0.0], [0.0; 8], c as f32)
}

/// Deterministic unit direction of an entity (splitmix64 of its id). Only the
/// angular spread depends on it; the band and the causal depth set the radius.
fn direction(entity: u64) -> [f64; 4] {
    let mut x = entity;
    let mut next = || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    };
    let v = [next(), next(), next(), next()];
    let n = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    if n < 1e-6 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        v.map(|a| a / n)
    }
}

impl CausalLod {
    /// Lay the DAG out on a fresh [`LodGraph`] under the live context. Every
    /// node's band is checked against the band its coordinate implies; a
    /// mismatch is refused. Feasibility is not judged here; see
    /// [`Self::check_feasible`].
    pub fn build<S: CausalMaskSource + ?Sized>(
        src: &S,
        ctx: CausalLodContext,
    ) -> Result<Self, CausalLodError> {
        check_view(src)?;
        let n = src.atom_count();
        let all = if n == MAX_CAUSAL_ATOMS {
            u64::MAX
        } else {
            bit(n) - 1
        };
        if (ctx.done | ctx.blocked) & !all != 0 {
            return Err(CausalLodError::Invalid(format!(
                "context masks reference actions outside the {n}-action DAG"
            )));
        }
        let target = src.target_atom();
        let topo = src.topo_order();
        let mut cone = bit(target);
        for &v in topo.iter().rev() {
            if cone & bit(v) != 0 {
                cone |= src.atom_parent_mask(v);
            }
        }

        // Lod2: hazard barriers and checkpoints under the live context.
        let barriers = ctx.blocked & cone & !ctx.done;
        let base = reach(src, ctx.done, ctx.blocked);
        let reachable = base & bit(target) != 0;
        let mut checkpoints = 0_u64;
        if reachable {
            for v in BitIter(cone & !ctx.done & !bit(target)) {
                if reach(src, ctx.done, ctx.blocked | bit(v)) & bit(target) == 0 {
                    checkpoints |= bit(v);
                }
            }
        }

        // Longest causal hop count from each cone action to the target.
        let mut hops = [0_u32; MAX_CAUSAL_ATOMS];
        for &c in topo.iter().rev() {
            if cone & bit(c) == 0 {
                continue;
            }
            for p in BitIter(src.atom_parent_mask(c)) {
                hops[p] = hops[p].max(hops[c] + 1);
            }
        }
        let max_hops = BitIter(cone).map(|v| hops[v]).max().unwrap_or(0);

        let graph = LodGraph::new();
        let c = graph.geometry().curvature;
        let w = band_scale_width();
        let place = |band: LodBand,
                     scale: f64,
                     dir: [f64; 4],
                     label: String,
                     entity: u64|
         -> Result<u32, CausalLodError> {
            let coord = coord_at_scale(scale, dir, c)?;
            let node = LodNode::new(0, band, coord, label, entity);
            let implied = node.derive_band_from_coord(&graph.geometry())?;
            if implied != band {
                return Err(CausalLodError::Invalid(format!(
                    "entity {entity} placed for {band:?} but its coordinate implies {implied:?}"
                )));
            }
            Ok(graph.add_node(node)?)
        };

        // Lod3 first: the systemic target-and-budget node.
        let target_dir = direction(u64::from(src.atom_action(target)));
        let systemic = place(
            LodBand::Lod3Systemic,
            3.5 * w,
            target_dir,
            format!(
                "goal: action {} within budget {}",
                src.atom_action(target),
                src.budget()
            ),
            ENTITY_SYSTEMIC,
        )?;

        // Lod0: one node per action, deeper the further from the target.
        let mut atom_node = Vec::with_capacity(n);
        for (v, &hop) in hops.iter().enumerate().take(n) {
            let depth = if cone & bit(v) != 0 {
                hop
            } else {
                max_hops + 1
            };
            // Strictly inside (0, w): the deepest action stays off the boundary floor.
            let scale = w * (1.0 - f64::from(depth + 1) / f64::from(max_hops + 3));
            let action = src.atom_action(v);
            let dir = if v == target {
                target_dir
            } else {
                direction(u64::from(action))
            };
            atom_node.push(place(
                LodBand::Lod0Atomic,
                scale,
                dir,
                format!("action {action}"),
                u64::from(action),
            )?);
        }

        let mut edges: Vec<(u32, u32, EdgeType, f32)> = Vec::new();
        for v in 0..n {
            let ty = if src.atom_is_or(v) {
                EdgeType::CausalTransition
            } else {
                EdgeType::DependsOn
            };
            for p in BitIter(src.atom_parent_mask(v)) {
                edges.push((atom_node[p], atom_node[v], ty, 1.0));
            }
        }
        edges.push((atom_node[target], systemic, EdgeType::CoarseGrain, 1.0));

        // Lod1: precondition clusters.
        let mut clusters = Vec::new();
        for v in 0..n {
            let pm = src.atom_parent_mask(v);
            let kind = if src.atom_is_or(v) && pm != 0 {
                ClusterKind::OrAlternatives
            } else if !src.atom_is_or(v) && pm.count_ones() >= 2 {
                ClusterKind::AndPreconditions
            } else {
                continue;
            };
            let node = place(
                LodBand::Lod1Cluster,
                1.5 * w,
                direction(u64::from(src.atom_action(v))),
                format!("{kind:?} of action {}", src.atom_action(v)),
                ENTITY_CLUSTER + v as u64,
            )?;
            for p in BitIter(pm) {
                edges.push((atom_node[p], node, EdgeType::CoarseGrain, 1.0));
            }
            if cone & bit(v) != 0 {
                edges.push((node, systemic, EdgeType::CoarseGrain, 1.0));
            }
            clusters.push(CausalCluster {
                kind,
                owner: v,
                members: pm,
                node,
            });
        }

        // Lod2: checkpoints and hazard barriers.
        let mut milestones = Vec::new();
        for (kind, mask, base_entity) in [
            (MilestoneKind::Checkpoint, checkpoints, ENTITY_CHECKPOINT),
            (MilestoneKind::HazardBarrier, barriers, ENTITY_BARRIER),
        ] {
            for v in BitIter(mask) {
                let node = place(
                    LodBand::Lod2Milestone,
                    2.5 * w,
                    direction(u64::from(src.atom_action(v))),
                    format!("{kind:?}: action {}", src.atom_action(v)),
                    base_entity + v as u64,
                )?;
                edges.push((atom_node[v], node, EdgeType::CoarseGrain, 1.0));
                edges.push((node, systemic, EdgeType::CoarseGrain, 1.0));
                milestones.push(CausalMilestone {
                    kind,
                    atom: v,
                    node,
                });
            }
        }
        graph.add_edges(&edges)?;
        graph.flush_edges_to_csr()?;

        Ok(Self {
            graph,
            atom_node,
            atom_action: (0..n).map(|v| src.atom_action(v)).collect(),
            atom_cost: (0..n).map(|v| src.atom_cost(v)).collect(),
            clusters,
            milestones,
            systemic,
            target,
            budget: src.budget(),
            cone,
            ctx,
            reachable_without_barriers: reachable,
            checkpoints,
            barriers,
        })
    }

    pub fn graph(&self) -> &LodGraph {
        &self.graph
    }

    /// Graph node of action index `atom`.
    pub fn atom_node(&self, atom: usize) -> Option<u32> {
        self.atom_node.get(atom).copied()
    }

    pub fn systemic_node(&self) -> u32 {
        self.systemic
    }

    pub fn clusters(&self) -> &[CausalCluster] {
        &self.clusters
    }

    pub fn milestones(&self) -> &[CausalMilestone] {
        &self.milestones
    }

    pub fn cone(&self) -> u64 {
        self.cone
    }

    /// Bitmask of the checkpoints (target excluded).
    pub fn checkpoints(&self) -> u64 {
        self.checkpoints
    }

    pub fn hazard_barriers(&self) -> u64 {
        self.barriers
    }

    /// `sum(cost)` of the not-done checkpoints and the target.
    pub fn mandatory_floor(&self) -> u64 {
        BitIter(self.checkpoints | bit(self.target))
            .map(|v| u64::from(self.atom_cost[v]))
            .sum()
    }

    fn actions(&self, mask: u64) -> Vec<u32> {
        BitIter(mask).map(|v| self.atom_action[v]).collect()
    }

    /// Fail-closed pre-check: the target is reachable around the hazard
    /// barriers, and `time_used + mandatory_floor <= budget`.
    pub fn check_feasible(&self) -> Result<(), CausalLodError> {
        if !self.reachable_without_barriers {
            return Err(CausalLodError::TargetUnreachable {
                target: self.atom_action[self.target],
                barriers: self.actions(self.barriers),
            });
        }
        let floor = self.mandatory_floor();
        if u64::from(self.ctx.time_used) + floor > u64::from(self.budget) {
            return Err(CausalLodError::BudgetInsufficient {
                budget: self.budget,
                time_used: self.ctx.time_used,
                floor,
                checkpoints: self.actions(self.checkpoints),
            });
        }
        Ok(())
    }

    /// Hyperbolic geodesic distance `d_H` from action `atom`'s node to the Lod3
    /// systemic node, on the graph's ball. Descriptive; see the module doc.
    pub fn goal_distance(&self, atom: usize) -> Result<f64, CausalLodError> {
        let id = self.atom_node(atom).ok_or_else(|| {
            CausalLodError::Invalid(format!("action index {atom} is not in the DAG"))
        })?;
        let node = self.graph.get_node(id).ok_or(LodError::NodeNotFound(id))?;
        let goal = self
            .graph
            .get_node(self.systemic)
            .ok_or(LodError::NodeNotFound(self.systemic))?;
        let widen = |h: [f32; 4]| h.map(f64::from);
        Ok(kernel::hyperbolic_distance(
            self.graph.geometry().curvature,
            &widen(node.coord.hyperbolic),
            &widen(goal.coord.hyperbolic),
        )
        .map_err(LodError::from)?)
    }

    pub fn summary(&self) -> CausalLodSummary {
        let count = |k| self.clusters.iter().filter(|c| c.kind == k).count();
        CausalLodSummary {
            atoms: self.atom_node.len(),
            or_clusters: count(ClusterKind::OrAlternatives),
            and_clusters: count(ClusterKind::AndPreconditions),
            checkpoints: self.actions(self.checkpoints),
            hazard_barriers: self.actions(self.barriers),
            mandatory_floor: self.mandatory_floor(),
            graph_nodes: self.graph.node_count(),
            graph_edges: self.graph.csr_snapshot().num_edges(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::band_from_scale;

    /// Plain DAG for the tests: parents as index lists.
    struct Dag {
        parents: Vec<u64>,
        is_or: Vec<bool>,
        cost: Vec<u32>,
        value: Vec<f64>,
        topo: Vec<usize>,
        target: usize,
        budget: u32,
    }

    impl CausalMaskSource for Dag {
        fn atom_count(&self) -> usize {
            self.parents.len()
        }
        fn target_atom(&self) -> usize {
            self.target
        }
        fn budget(&self) -> u32 {
            self.budget
        }
        fn topo_order(&self) -> &[usize] {
            &self.topo
        }
        fn atom_parent_mask(&self, i: usize) -> u64 {
            self.parents[i]
        }
        fn atom_is_or(&self, i: usize) -> bool {
            self.is_or[i]
        }
        fn atom_cost(&self, i: usize) -> u32 {
            self.cost[i]
        }
        fn atom_value(&self, i: usize) -> f64 {
            self.value[i]
        }
        fn atom_action(&self, i: usize) -> u32 {
            100 + i as u32
        }
    }

    fn mask(ix: &[usize]) -> u64 {
        ix.iter().fold(0, |m, &i| m | bit(i))
    }

    /// 0 fetch(2) -> 1 build(3) -> 2 test(2) -\
    /// 0 fetch    -> 3 lint(1) ---------------+-> 4 deploy (AND, target)
    /// 5 package: OR over {1, 3}, outside the cone. 6 docs: free source.
    /// 7 sign: OR over {2, 3}, outside the cone.
    fn release(budget: u32) -> Dag {
        Dag {
            parents: vec![
                0,
                mask(&[0]),
                mask(&[1]),
                mask(&[0]),
                mask(&[2, 3]),
                mask(&[1, 3]),
                0,
                mask(&[2, 3]),
            ],
            is_or: vec![false, false, false, false, false, true, false, true],
            cost: vec![2, 3, 2, 1, 1, 1, 1, 1],
            value: vec![0.0; 8],
            topo: vec![0, 1, 3, 2, 4, 5, 6, 7],
            target: 4,
            budget,
        }
    }

    /// Target 3 is OR over 0 and 1; 2 is an AND child of 0 outside the cone.
    fn or_target() -> Dag {
        Dag {
            parents: vec![0, 0, mask(&[0]), mask(&[0, 1])],
            is_or: vec![false, false, false, true],
            cost: vec![4, 1, 1, 1],
            value: vec![0.0, 0.0, 0.0, 0.0],
            topo: vec![0, 1, 2, 3],
            target: 3,
            budget: 10,
        }
    }

    fn edges_of(lod: &CausalLod) -> Vec<(u32, u32, EdgeType)> {
        let csr = lod.graph().csr_snapshot();
        (0..csr.num_nodes() as u32)
            .flat_map(|u| {
                csr.neighbors(u)
                    .map(move |(v, t, _)| (u, v, t))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn every_action_lands_in_lod0_and_each_layer_in_its_band() {
        let d = release(20);
        let lod = CausalLod::build(&d, CausalLodContext::default()).unwrap();
        let g = lod.graph();
        for v in 0..8 {
            let node = g.get_node(lod.atom_node(v).unwrap()).unwrap();
            assert_eq!(node.band, LodBand::Lod0Atomic, "action {v}");
            assert_eq!(node.entity_id, 100 + v as u64);
            assert_eq!(
                node.derive_band_from_coord(&g.geometry()).unwrap(),
                node.band
            );
        }
        let sys = g.get_node(lod.systemic_node()).unwrap();
        assert_eq!(sys.band, LodBand::Lod3Systemic);
        assert!(sys.entity_id >= SYNTHETIC_ENTITY_BASE);
        for c in lod.clusters() {
            let node = g.get_node(c.node).unwrap();
            assert_eq!(node.band, LodBand::Lod1Cluster);
            assert_eq!(
                node.derive_band_from_coord(&g.geometry()).unwrap(),
                node.band
            );
        }
        for m in lod.milestones() {
            let node = g.get_node(m.node).unwrap();
            assert_eq!(node.band, LodBand::Lod2Milestone);
            assert_eq!(
                node.derive_band_from_coord(&g.geometry()).unwrap(),
                node.band
            );
        }
        // The placement scales sit inside the band intervals of band_from_scale.
        let w = band_scale_width() as f32;
        assert_eq!(band_from_scale(1.5 * w).unwrap(), LodBand::Lod1Cluster);
        assert_eq!(band_from_scale(2.5 * w).unwrap(), LodBand::Lod2Milestone);
        assert_eq!(band_from_scale(3.5 * w).unwrap(), LodBand::Lod3Systemic);
    }

    #[test]
    fn and_parents_depend_on_and_or_parents_are_causal_transitions() {
        let d = release(20);
        let lod = CausalLod::build(&d, CausalLodContext::default()).unwrap();
        let edges = edges_of(&lod);
        let a = |v| lod.atom_node(v).unwrap();
        // AND: build depends on fetch, deploy depends on test and lint.
        for (p, c) in [(0, 1), (1, 2), (0, 3), (2, 4), (3, 4)] {
            assert!(
                edges.contains(&(a(p), a(c), EdgeType::DependsOn)),
                "{p}->{c}"
            );
        }
        // OR: package and sign are enabled by transitions.
        for (p, c) in [(1, 5), (3, 5), (2, 7), (3, 7)] {
            assert!(
                edges.contains(&(a(p), a(c), EdgeType::CausalTransition)),
                "{p}->{c}"
            );
        }
        let atom_atom = edges
            .iter()
            .filter(|(_, _, t)| matches!(t, EdgeType::DependsOn | EdgeType::CausalTransition))
            .count();
        assert_eq!(atom_atom, 9);
        // The target coarse-grains into the systemic node.
        assert!(edges.contains(&(a(4), lod.systemic_node(), EdgeType::CoarseGrain)));
    }

    #[test]
    fn clusters_cover_or_parents_and_multi_parent_and_preconditions() {
        let d = release(20);
        let lod = CausalLod::build(&d, CausalLodContext::default()).unwrap();
        let edges = edges_of(&lod);
        let kinds: Vec<(ClusterKind, usize, u64)> = lod
            .clusters()
            .iter()
            .map(|c| (c.kind, c.owner, c.members))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (ClusterKind::AndPreconditions, 4, mask(&[2, 3])),
                (ClusterKind::OrAlternatives, 5, mask(&[1, 3])),
                (ClusterKind::OrAlternatives, 7, mask(&[2, 3])),
            ]
        );
        for c in lod.clusters() {
            for m in BitIter(c.members) {
                let e = (lod.atom_node(m).unwrap(), c.node, EdgeType::CoarseGrain);
                assert!(edges.contains(&e), "member {m} of {:?}", c.kind);
            }
            // Only a cone cluster coarse-grains into the goal.
            let up = (c.node, lod.systemic_node(), EdgeType::CoarseGrain);
            assert_eq!(edges.contains(&up), lod.cone() & bit(c.owner) != 0);
        }
        let s = lod.summary();
        assert_eq!((s.atoms, s.or_clusters, s.and_clusters), (8, 2, 1));
        assert_eq!(s.graph_nodes, lod.graph().node_count());
        assert_eq!(s.graph_edges, edges.len());
    }

    #[test]
    fn checkpoints_are_exactly_the_actions_every_plan_needs() {
        // release: every plan runs fetch, build, test, lint before deploy.
        let d = release(20);
        let lod = CausalLod::build(&d, CausalLodContext::default()).unwrap();
        assert_eq!(lod.checkpoints(), mask(&[0, 1, 2, 3]));
        assert_eq!(lod.mandatory_floor(), 2 + 3 + 2 + 1 + 1);
        // OR target: neither branch is mandatory.
        let o = or_target();
        let lod = CausalLod::build(&o, CausalLodContext::default()).unwrap();
        assert_eq!(lod.checkpoints(), 0);
        assert_eq!(lod.mandatory_floor(), 1);
        // Blocking branch 1 makes branch 0 mandatory and a barrier appears.
        let ctx = CausalLodContext {
            blocked: mask(&[1]),
            ..CausalLodContext::default()
        };
        let lod = CausalLod::build(&o, ctx).unwrap();
        assert_eq!(lod.checkpoints(), mask(&[0]));
        assert_eq!(lod.hazard_barriers(), mask(&[1]));
        let kinds: Vec<_> = lod.milestones().iter().map(|m| (m.kind, m.atom)).collect();
        assert_eq!(
            kinds,
            vec![
                (MilestoneKind::Checkpoint, 0),
                (MilestoneKind::HazardBarrier, 1)
            ]
        );
        // Done checkpoints leave the floor.
        let ctx = CausalLodContext {
            done: mask(&[0, 1]),
            time_used: 5,
            blocked: 0,
        };
        let lod = CausalLod::build(&d, ctx).unwrap();
        assert_eq!(lod.checkpoints(), mask(&[2, 3]));
        assert_eq!(lod.mandatory_floor(), 2 + 1 + 1);
    }

    #[test]
    fn feasibility_refuses_unreachable_targets_and_short_budgets() {
        let d = release(9);
        CausalLod::build(&d, CausalLodContext::default())
            .unwrap()
            .check_feasible()
            .unwrap();
        // One unit short of the floor of 9.
        let short = release(8);
        match CausalLod::build(&short, CausalLodContext::default())
            .unwrap()
            .check_feasible()
        {
            Err(CausalLodError::BudgetInsufficient {
                floor, checkpoints, ..
            }) => {
                assert_eq!(floor, 9);
                assert_eq!(checkpoints, vec![100, 101, 102, 103]);
            }
            other => panic!("expected BudgetInsufficient, got {other:?}"),
        }
        // time_used counts against the same budget.
        let ctx = CausalLodContext {
            time_used: 1,
            ..CausalLodContext::default()
        };
        assert!(matches!(
            CausalLod::build(&d, ctx).unwrap().check_feasible(),
            Err(CausalLodError::BudgetInsufficient { .. })
        ));
        // Both OR branches blocked: the target is unreachable.
        let o = or_target();
        let ctx = CausalLodContext {
            blocked: mask(&[0, 1]),
            ..CausalLodContext::default()
        };
        match CausalLod::build(&o, ctx).unwrap().check_feasible() {
            Err(CausalLodError::TargetUnreachable { target, barriers }) => {
                assert_eq!(target, 103);
                assert_eq!(barriers, vec![100, 101]);
            }
            other => panic!("expected TargetUnreachable, got {other:?}"),
        }
    }

    #[test]
    fn goal_distance_grows_with_causal_depth() {
        let d = release(20);
        let lod = CausalLod::build(&d, CausalLodContext::default()).unwrap();
        let dist = |v| lod.goal_distance(v).unwrap();
        // Hop counts to deploy: deploy 0, test/lint 1, build 2, fetch 3.
        // The target shares the goal's direction, so only depth separates them.
        assert!(dist(4) < dist(2), "{} {}", dist(4), dist(2));
        // Same direction comparison is not guaranteed for the others, but the
        // radius is: compare normalized depths.
        let depth = |v| {
            let n = lod.graph().get_node(lod.atom_node(v).unwrap()).unwrap();
            crate::node::normalized_depth(&n.coord.hyperbolic, 1.0).unwrap()
        };
        assert!(depth(4) < depth(2) && depth(2) < depth(1) && depth(1) < depth(0));
        // Off-cone actions sit deepest.
        assert!(depth(6) > depth(0));
        // d_H to the goal is at least the radial gap (triangle inequality).
        let goal_depth = {
            let n = lod.graph().get_node(lod.systemic_node()).unwrap();
            crate::node::normalized_depth(&n.coord.hyperbolic, 1.0).unwrap()
        };
        for v in 0..8 {
            assert!(dist(v) + 1e-9 >= depth(v) - goal_depth, "action {v}");
        }
        assert!(lod.goal_distance(99).is_err());
    }

    #[test]
    fn a_64_action_chain_places_its_deepest_action_inside_the_ball() {
        // Longest possible hop count: 0 -> 1 -> ... -> 63, target 63.
        let n = MAX_CAUSAL_ATOMS;
        let d = Dag {
            parents: (0..n)
                .map(|i| if i == 0 { 0 } else { bit(i - 1) })
                .collect(),
            is_or: vec![false; n],
            cost: vec![1; n],
            value: vec![0.0; n],
            topo: (0..n).collect(),
            target: n - 1,
            budget: 64,
        };
        let lod = CausalLod::build(&d, CausalLodContext::default()).unwrap();
        lod.check_feasible().unwrap();
        assert_eq!(lod.mandatory_floor(), 64);
        assert_eq!(lod.checkpoints().count_ones(), 63);
        let g = lod.graph();
        let depth = |v: usize| {
            let node = g.get_node(lod.atom_node(v).unwrap()).unwrap();
            assert_eq!(
                node.derive_band_from_coord(&g.geometry()).unwrap(),
                LodBand::Lod0Atomic
            );
            crate::node::normalized_depth(&node.coord.hyperbolic, 1.0).unwrap()
        };
        // Strictly ordered by hops all the way down, and all below the floor depth.
        for v in 1..n {
            assert!(depth(v - 1) > depth(v), "action {v}");
        }
        assert!(depth(0) < max_chart_depth());
        assert!(lod.goal_distance(0).unwrap().is_finite());
        // One budget unit short of the floor is refused.
        let short = Dag { budget: 63, ..d };
        assert!(matches!(
            CausalLod::build(&short, CausalLodContext::default())
                .unwrap()
                .check_feasible(),
            Err(CausalLodError::BudgetInsufficient { floor: 64, .. })
        ));
    }

    #[test]
    fn synthetic_entities_never_collide_with_actions() {
        // u32::MAX as an action id still sits below the synthetic range.
        assert!(u64::from(u32::MAX) < SYNTHETIC_ENTITY_BASE);
        let d = release(20);
        let lod = CausalLod::build(&d, CausalLodContext::default()).unwrap();
        let g = lod.graph();
        let mut ids: Vec<u64> = (0..g.node_count() as u32)
            .map(|i| g.get_node(i).unwrap().entity_id)
            .collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n);
    }

    #[test]
    fn malformed_views_are_refused() {
        let mut d = release(20);
        d.topo = vec![1, 0, 3, 2, 4, 5, 6, 7]; // build before fetch
        assert!(matches!(
            CausalPotential::new(&d),
            Err(CausalLodError::Invalid(_))
        ));
        let mut d = release(20);
        d.target = 8;
        assert!(CausalLod::build(&d, CausalLodContext::default()).is_err());
        let d = release(20);
        let ctx = CausalLodContext {
            blocked: bit(9),
            ..CausalLodContext::default()
        };
        assert!(matches!(
            CausalLod::build(&d, ctx),
            Err(CausalLodError::Invalid(_))
        ));
    }

    #[test]
    fn closure_matches_hand_computed_costs() {
        let d = release(20);
        let pot = CausalPotential::new(&d).unwrap();
        assert_eq!(pot.cone(), mask(&[0, 1, 2, 3, 4]));
        // Nothing done: fetch+build+test+lint+deploy = 2+3+2+1+1.
        assert_eq!(pot.closure(0), 9.0);
        // fetch and build done: test+lint+deploy.
        assert_eq!(pot.closure(mask(&[0, 1])), 4.0);
        // Off-cone bits change nothing.
        assert_eq!(pot.closure(mask(&[5, 6, 7])), 9.0);
        // Completing fetch pays 2 and lowers h by 2: dPhi = 0.
        assert_eq!(pot.delta_phi(0, 0), 0.0);
        // OR target picks the cheaper branch: 1 (cost 1) over 0 (cost 4).
        let o = or_target();
        let pot = CausalPotential::new(&o).unwrap();
        assert_eq!(pot.closure(0), 2.0);
        // Taking the expensive branch costs 4 and leaves h = 1: dPhi = -(4 + 1 - 2).
        assert_eq!(pot.delta_phi(0, 0), -3.0);
        let mut out = [0.0; MAX_CAUSAL_ATOMS];
        pot.bias_into(0, mask(&[0, 1]), 2.0, &mut out).unwrap();
        assert_eq!((out[0], out[1]), (-6.0, 0.0));
    }
}
