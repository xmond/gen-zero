//! Causal DAG: the structural input of the causal-triad pipeline.
//!
//! Port of the gen-zero-research repository's `python/gen_zero/planner/triad/dag.py`. Nodes are the decide
//! candidates in request order, so node index `i` is `candidates[i]`. Each node
//! carries an AND/OR precondition rule over its parents, an integer nominal
//! cost (>= 1) and an immediate value. The DAG is refused, never repaired, when
//! any invariant breaks: unknown action, self-parent, duplicate edge, missing
//! cost, non-finite value, out-of-range target, budget < 1 or == `u32::MAX`,
//! or a cycle.
//!
//! Index sets are `u64` bitmasks, so a DAG holds at most [`MAX_CAUSAL_ATOMS`]
//! nodes. The production `decide` caps candidates at 16, far below that.
//!
//! The DAG is handed to `gen-zero-lod` through [`CausalMaskSource`]: the
//! closure potential and the four-band layout live there, below the planner.
//!
//! Not to be confused with [`crate::causal_dag`], the exact Dijkstra planner of
//! the `causal_plan` verb: that one takes float costs, AND and OR groups on the
//! same node, and up to 65 536 nodes, and returns the single optimal order.
//! This one is the triad's sampled-plan input over the decide candidates.

use crate::error::PlannerError;
use gen_zero_core::ActionId;
use gen_zero_lod::{CausalMaskSource, MAX_CAUSAL_ATOMS};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Maximum nominal budget the triad accepts. Keeps exact-scoring scratch bounded.
pub const MAX_TRIAD_BUDGET: u32 = 1 << 24;

/// One action of the DAG.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalNode {
    pub action: ActionId,
    /// Nominal time cost, >= 1.
    pub cost: u32,
    /// Immediate value on first completion. Finite.
    pub value: f64,
    /// `true`: one done parent suffices (OR). `false`: all parents must be done (AND).
    pub is_or: bool,
}

/// `parent` must complete before `child` may run (subject to the child's AND/OR rule).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CausalEdge {
    pub parent: ActionId,
    pub child: ActionId,
}

/// Wire form of a DAG, keyed by action id. Same contract as the Python
/// `state["causal_dag"]` dict: `cost` may be spelled `costs` and `value` may
/// be spelled `values`. Giving both spellings is a duplicate-field error, not a
/// silent preference. Unknown fields are refused. Serialized with the
/// canonical spellings (`cost`, `value`), so a serialized spec reads back.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CausalDagSpec {
    /// child -> parents.
    #[serde(default)]
    pub parents: BTreeMap<u32, Vec<u32>>,
    /// Actions with an OR rule. Absent means AND.
    #[serde(default)]
    pub is_or: BTreeMap<u32, bool>,
    /// Required for every candidate.
    #[serde(alias = "costs")]
    pub cost: BTreeMap<u32, u32>,
    /// Absent means 0.0.
    #[serde(default, alias = "values")]
    pub value: BTreeMap<u32, f64>,
    pub target: u32,
    pub budget: u32,
}

/// A validated AND/OR causal DAG over integer node indices.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalDag {
    nodes: Vec<CausalNode>,
    parents: Vec<Vec<usize>>,
    parent_mask: Vec<u64>,
    topo: Vec<usize>,
    target: usize,
    budget: u32,
}

fn invalid(detail: impl std::fmt::Display) -> PlannerError {
    PlannerError::InvalidInput(format!("invalid causal DAG: {detail}"))
}

impl CausalDag {
    /// Build and validate a DAG from explicit nodes and edges.
    pub fn new(
        nodes: Vec<CausalNode>,
        edges: &[CausalEdge],
        target: ActionId,
        budget: u32,
    ) -> Result<Self, PlannerError> {
        let n = nodes.len();
        if n == 0 {
            return Err(invalid("no actions"));
        }
        // Bitmask width: a larger DAG is refused instead of silently truncated.
        if n > MAX_CAUSAL_ATOMS {
            return Err(invalid(format!(
                "{n} actions exceed the cap of {MAX_CAUSAL_ATOMS}"
            )));
        }
        let mut index = BTreeMap::new();
        for (i, node) in nodes.iter().enumerate() {
            if index.insert(node.action.0, i).is_some() {
                return Err(invalid(format!("duplicate action {}", node.action.0)));
            }
            if node.cost < 1 {
                return Err(invalid(format!(
                    "cost of action {} is {}, must be >= 1",
                    node.action.0, node.cost
                )));
            }
            if !node.value.is_finite() {
                return Err(invalid(format!(
                    "value of action {} is not finite ({})",
                    node.action.0, node.value
                )));
            }
        }
        let lookup = |a: ActionId, role: &str| {
            index
                .get(&a.0)
                .copied()
                .ok_or_else(|| invalid(format!("{role} references unknown action {}", a.0)))
        };
        let target = lookup(target, "target")?;
        if !(1..=MAX_TRIAD_BUDGET).contains(&budget) {
            return Err(invalid(format!(
                "budget {budget} must lie in 1..={MAX_TRIAD_BUDGET}"
            )));
        }
        let mut parents = vec![Vec::new(); n];
        let mut parent_mask = vec![0_u64; n];
        for edge in edges {
            let p = lookup(edge.parent, "edge parent")?;
            let c = lookup(edge.child, "edge child")?;
            if p == c {
                return Err(invalid(format!(
                    "action {} lists itself as its own parent",
                    edge.child.0
                )));
            }
            if parent_mask[c] & (1 << p) != 0 {
                return Err(invalid(format!(
                    "duplicate edge {} -> {}",
                    edge.parent.0, edge.child.0
                )));
            }
            parent_mask[c] |= 1 << p;
            parents[c].push(p);
        }
        for ps in &mut parents {
            ps.sort_unstable();
        }
        let topo = topological_order(&parents).ok_or_else(|| invalid("the graph has a cycle"))?;
        Ok(Self {
            nodes,
            parents,
            parent_mask,
            topo,
            target,
            budget,
        })
    }

    /// Build from the wire spec. `candidates` fixes the node order, so a chosen
    /// index path maps straight back to actions.
    pub fn from_spec(candidates: &[ActionId], spec: &CausalDagSpec) -> Result<Self, PlannerError> {
        let known = |id: u32| candidates.iter().any(|a| a.0 == id);
        for (field, keys) in [
            ("parents", spec.parents.keys().copied().collect::<Vec<_>>()),
            ("is_or", spec.is_or.keys().copied().collect()),
            ("cost", spec.cost.keys().copied().collect()),
            ("value", spec.value.keys().copied().collect()),
        ] {
            if let Some(id) = keys.into_iter().find(|&id| !known(id)) {
                return Err(invalid(format!(
                    "{field} references action {id}, which is not a candidate"
                )));
            }
        }
        let nodes = candidates
            .iter()
            .map(|&action| {
                let cost = *spec.cost.get(&action.0).ok_or_else(|| {
                    invalid(format!("cost is missing an entry for action {}", action.0))
                })?;
                Ok(CausalNode {
                    action,
                    cost,
                    value: spec.value.get(&action.0).copied().unwrap_or(0.0),
                    is_or: spec.is_or.get(&action.0).copied().unwrap_or(false),
                })
            })
            .collect::<Result<Vec<_>, PlannerError>>()?;
        let edges: Vec<CausalEdge> = spec
            .parents
            .iter()
            .flat_map(|(&child, ps)| {
                ps.iter().map(move |&parent| CausalEdge {
                    parent: ActionId(parent),
                    child: ActionId(child),
                })
            })
            .collect();
        Self::new(nodes, &edges, ActionId(spec.target), spec.budget)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[CausalNode] {
        &self.nodes
    }

    pub fn node(&self, i: usize) -> &CausalNode {
        &self.nodes[i]
    }

    pub fn parents(&self, i: usize) -> &[usize] {
        &self.parents[i]
    }

    /// Bit `p` is set when node `p` is a parent of node `i`.
    pub fn parent_mask(&self, i: usize) -> u64 {
        self.parent_mask[i]
    }

    pub fn target(&self) -> usize {
        self.target
    }

    pub fn budget(&self) -> u32 {
        self.budget
    }

    /// Node index of `action`, if it is in the DAG.
    pub fn index_of(&self, action: ActionId) -> Option<usize> {
        self.nodes.iter().position(|n| n.action == action)
    }

    /// One valid topological order (Kahn, smallest ready index first).
    pub fn topological_order(&self) -> &[usize] {
        &self.topo
    }

    /// Whether `order` places every parent before each of its children.
    /// Only the nodes present in `order` are checked against each other.
    pub fn respects_topology(&self, order: &[usize]) -> bool {
        let mut pos = vec![usize::MAX; self.len()];
        for (k, &v) in order.iter().enumerate() {
            match pos.get_mut(v) {
                Some(slot) if *slot == usize::MAX => *slot = k,
                _ => return false,
            }
        }
        order.iter().all(|&c| {
            self.parents[c]
                .iter()
                .all(|&p| pos[p] == usize::MAX || pos[p] < pos[c])
        })
    }

    /// The target and every ancestor of it, as a bitmask.
    pub fn goal_cone(&self) -> u64 {
        let mut cone = 1_u64 << self.target;
        let mut stack = vec![self.target];
        while let Some(v) = stack.pop() {
            for &p in &self.parents[v] {
                if cone & (1 << p) == 0 {
                    cone |= 1 << p;
                    stack.push(p);
                }
            }
        }
        cone
    }

    /// Bitmask of the nodes whose action is in `actions`. Unknown actions are refused.
    pub fn mask_of(&self, actions: &[ActionId]) -> Result<u64, PlannerError> {
        actions.iter().try_fold(0_u64, |m, &a| {
            self.index_of(a)
                .map(|i| m | (1 << i))
                .ok_or_else(|| invalid(format!("action {} is not in the DAG", a.0)))
        })
    }
}

impl CausalMaskSource for CausalDag {
    fn atom_count(&self) -> usize {
        self.nodes.len()
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
        self.parent_mask[i]
    }
    fn atom_is_or(&self, i: usize) -> bool {
        self.nodes[i].is_or
    }
    fn atom_cost(&self, i: usize) -> u32 {
        self.nodes[i].cost
    }
    fn atom_value(&self, i: usize) -> f64 {
        self.nodes[i].value
    }
    fn atom_action(&self, i: usize) -> u32 {
        self.nodes[i].action.0
    }
}

/// Kahn's algorithm over parent lists. `None` on a cycle.
fn topological_order(parents: &[Vec<usize>]) -> Option<Vec<usize>> {
    let n = parents.len();
    let mut indegree: Vec<usize> = parents.iter().map(Vec::len).collect();
    let mut children = vec![Vec::new(); n];
    for (c, ps) in parents.iter().enumerate() {
        for &p in ps {
            children[p].push(c);
        }
    }
    let mut ready: VecDeque<usize> = (0..n).filter(|&v| indegree[v] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(v) = ready.pop_front() {
        order.push(v);
        for &c in &children[v] {
            indegree[c] -= 1;
            if indegree[c] == 0 {
                ready.push_back(c);
            }
        }
    }
    (order.len() == n).then_some(order)
}
