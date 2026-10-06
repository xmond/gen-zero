//! Exact cost-optimal plan on a symbolic AND/OR causal DAG.
//!
//! A node can be completed once every `and_parents` member is complete and
//! every `or_parents` group has at least one complete member. Completing a
//! node costs `cost`. The plan is the cheapest completion order that ends
//! with `target` complete.
//!
//! The search is Dijkstra over completion sets, so it is exact, not a
//! heuristic. Before it runs, two polynomial steps check and shrink the
//! problem:
//!
//! 1. The monotone closure from `initial_completed`: completing a node never
//!    disables another, so the closure is exactly the set of nodes that can
//!    ever be completed. A target outside it is `UnreachableGoal`, decided
//!    without any search. Because of this, a search that exhausts afterwards
//!    is attributed to the budget, never guessed. The closure is one worklist
//!    pass over a reverse-adjacency graph (parent to the children it can
//!    unlock): every reverse edge is visited once and every OR group closes
//!    once, so it is O(V+E) regardless of the order `nodes` is listed in,
//!    not a repeated full scan.
//! 2. The goal cone: ancestors of `target` that are in the closure and not
//!    already complete. Nothing outside it can lower the cost of reaching
//!    `target`, so it is dropped. The search walks `2^cone` states at most,
//!    so a cone above [`MAX_CONE_NODES`] is refused as `TargetTooComplex`.
//!
//! Acyclicity is not required. In an acyclic graph every node is reachable
//! (each OR group is non-empty), so `UnreachableGoal` comes only from a
//! prerequisite cycle that no node outside it can enter, a deadlock. A cycle
//! that can be entered from outside is planned through normally.
//!
//! The search itself also stops at [`MAX_EXPANDED_STATES`] with
//! `StateLimitExceeded`, and at a wall-clock [`PLAN_DEADLINE`] with
//! `DeadlineExceeded`. The deadline is checked once per expansion inside the
//! Dijkstra loop, not during validation, the closure pass, or the cone walk,
//! because `spawn_blocking` cannot be cancelled from the outside: an outer
//! `tokio` timeout would abandon the future but leave the search running and
//! the CPU permit held, so the deadline has to live in the loop itself.
//! Neither cap ever degrades to an approximate plan.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Largest goal cone (nodes still to complete) the exact search accepts.
pub const MAX_CONE_NODES: usize = 32;
/// Largest number of completion sets the search expands before it refuses.
pub const MAX_EXPANDED_STATES: usize = 200_000;
/// Largest DAG a request may describe; bounds validation and closure work.
pub const MAX_DAG_NODES: usize = 65_536;
/// Wall-clock budget for the exact search. Checked inside the Dijkstra loop
/// because `spawn_blocking` cannot be cancelled, so an outer timeout would
/// not free the CPU slot; see the module doc.
pub const PLAN_DEADLINE: Duration = Duration::from_millis(500);

/// One node of the causal DAG.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalNode {
    pub id: u32,
    /// Cost of completing this node; finite and non-negative.
    pub cost: f64,
    /// Every one of these must be complete first.
    #[serde(default)]
    pub and_parents: Vec<u32>,
    /// Each group needs at least one complete member first.
    #[serde(default)]
    pub or_parents: Vec<Vec<u32>>,
}

/// A planning problem: reach `target` from `initial_completed` within `budget`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalDagRequest {
    pub nodes: Vec<CausalNode>,
    pub target: u32,
    #[serde(default)]
    pub initial_completed: Vec<u32>,
    /// Largest accepted total cost; finite and non-negative.
    pub budget: f64,
}

/// The cheapest plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDagPlan {
    /// Node ids in completion order; ends with the target. Empty when the
    /// target is already in `initial_completed`.
    pub path: Vec<u32>,
    /// Left-to-right sum of the costs along `path`.
    pub total_cost: f64,
    /// Completion sets popped and expanded by the search.
    pub expanded_states: usize,
    /// Nodes in the pruned goal cone that the search worked over.
    pub cone_nodes: usize,
}

#[derive(Error, Debug, Clone, PartialEq)]
pub enum CausalDagError {
    #[error("invalid causal DAG: {0}")]
    InvalidInput(String),
    #[error("goal cone of target {target} has {cone_nodes} nodes; the exact search accepts at most {max}")]
    TargetTooComplex {
        target: u32,
        cone_nodes: usize,
        max: usize,
    },
    #[error("target {target} cannot be completed from the initial set")]
    UnreachableGoal { target: u32 },
    #[error("target {target} is reachable but every plan costs more than the budget {budget}")]
    BudgetExceeded { target: u32, budget: f64 },
    #[error("exact search stopped after {expanded} expanded states (limit {max})")]
    StateLimitExceeded { expanded: usize, max: usize },
    #[error("exact search stopped after {elapsed_ms} ms (deadline {limit_ms} ms)")]
    DeadlineExceeded { elapsed_ms: u64, limit_ms: u64 },
}

/// Heap entry ordered so `BinaryHeap` pops the cheapest set first.
struct Entry {
    cost: f64,
    mask: u64,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.mask.cmp(&self.mask))
    }
}

/// Requirement of one cone node, in cone-local bits. A parent already
/// complete is dropped; an OR group with a complete member is dropped whole.
struct Need {
    and_mask: u64,
    or_masks: Vec<u64>,
}

fn invalid(detail: impl Into<String>) -> CausalDagError {
    CausalDagError::InvalidInput(detail.into())
}

fn validate(req: &CausalDagRequest) -> Result<HashMap<u32, usize>, CausalDagError> {
    if req.nodes.is_empty() {
        return Err(invalid("`nodes` is empty"));
    }
    if req.nodes.len() > MAX_DAG_NODES {
        return Err(invalid(format!(
            "{} nodes; at most {MAX_DAG_NODES} are accepted",
            req.nodes.len()
        )));
    }
    if !req.budget.is_finite() || req.budget < 0.0 {
        return Err(invalid(format!(
            "budget {} must be finite and >= 0",
            req.budget
        )));
    }
    let mut index = HashMap::with_capacity(req.nodes.len());
    for (i, node) in req.nodes.iter().enumerate() {
        if index.insert(node.id, i).is_some() {
            return Err(invalid(format!("duplicate node id {}", node.id)));
        }
        if !node.cost.is_finite() || node.cost < 0.0 {
            return Err(invalid(format!(
                "node {} cost {} must be finite and >= 0",
                node.id, node.cost
            )));
        }
    }
    for node in &req.nodes {
        for group in &node.or_parents {
            if group.is_empty() {
                return Err(invalid(format!(
                    "node {} has an empty or_parents group, which no plan can satisfy",
                    node.id
                )));
            }
        }
        let parents = node
            .and_parents
            .iter()
            .chain(node.or_parents.iter().flatten());
        for parent in parents {
            if !index.contains_key(parent) {
                return Err(invalid(format!(
                    "node {} names parent {parent}, which is not in `nodes`",
                    node.id
                )));
            }
            if *parent == node.id {
                return Err(invalid(format!("node {} is its own parent", node.id)));
            }
        }
    }
    if !index.contains_key(&req.target) {
        return Err(invalid(format!("target {} is not in `nodes`", req.target)));
    }
    let mut seen = HashSet::with_capacity(req.initial_completed.len());
    for id in &req.initial_completed {
        if !index.contains_key(id) {
            return Err(invalid(format!(
                "initial_completed names {id}, which is not in `nodes`"
            )));
        }
        if !seen.insert(*id) {
            return Err(invalid(format!("initial_completed names {id} twice")));
        }
    }
    Ok(index)
}

/// One reverse edge: completing the parent counts down one of the child's
/// requirements. `Or(g)` names the child's OR-group index so a repeated
/// member, or any other member of the same group finishing first, is only
/// counted once.
enum RevEdge {
    And,
    Or(usize),
}

/// Nodes that can ever be completed, starting from `done`.
///
/// Worklist propagation: each reverse edge is visited once and each OR group
/// closes once, so the cost is O(V+E) whatever order `nodes` is listed in.
fn closure(req: &CausalDagRequest, index: &HashMap<u32, usize>, mut done: Vec<bool>) -> Vec<bool> {
    let n = req.nodes.len();
    let mut rev: Vec<Vec<(usize, RevEdge)>> = (0..n).map(|_| Vec::new()).collect();
    let mut and_left = vec![0usize; n];
    let mut or_left = vec![0usize; n];
    let mut or_done: Vec<Vec<bool>> = Vec::with_capacity(n);
    for (i, node) in req.nodes.iter().enumerate() {
        and_left[i] = node.and_parents.len();
        or_left[i] = node.or_parents.len();
        or_done.push(vec![false; node.or_parents.len()]);
        for p in &node.and_parents {
            rev[index[p]].push((i, RevEdge::And));
        }
        for (g, group) in node.or_parents.iter().enumerate() {
            for p in group {
                rev[index[p]].push((i, RevEdge::Or(g)));
            }
        }
    }

    let mut queue = Vec::with_capacity(n);
    for i in 0..n {
        if done[i] || (and_left[i] == 0 && or_left[i] == 0) {
            done[i] = true;
            queue.push(i);
        }
    }
    let mut head = 0;
    while head < queue.len() {
        let i = queue[head];
        head += 1;
        for (child, edge) in &rev[i] {
            let child = *child;
            if done[child] {
                continue;
            }
            match edge {
                RevEdge::And => and_left[child] -= 1,
                RevEdge::Or(g) => {
                    if !or_done[child][*g] {
                        or_done[child][*g] = true;
                        or_left[child] -= 1;
                    }
                }
            }
            if and_left[child] == 0 && or_left[child] == 0 {
                done[child] = true;
                queue.push(child);
            }
        }
    }
    done
}

/// Cheapest plan that completes `req.target`, or a typed refusal.
pub fn exact_causal_plan(req: &CausalDagRequest) -> Result<CausalDagPlan, CausalDagError> {
    plan_within(req, PLAN_DEADLINE)
}

/// Implementation behind [`exact_causal_plan`], parameterized on the deadline
/// so tests can swap it: a long deadline lets a state-limit test run to
/// completion, and `Duration::ZERO` proves `DeadlineExceeded` fires before
/// the search does any work.
fn plan_within(req: &CausalDagRequest, limit: Duration) -> Result<CausalDagPlan, CausalDagError> {
    let start = Instant::now();
    let index = validate(req)?;
    let n = req.nodes.len();
    let mut initial = vec![false; n];
    for id in &req.initial_completed {
        initial[index[id]] = true;
    }
    let target_g = index[&req.target];
    if initial[target_g] {
        return Ok(CausalDagPlan {
            path: Vec::new(),
            total_cost: 0.0,
            expanded_states: 0,
            cone_nodes: 0,
        });
    }

    let reachable = closure(req, &index, initial.clone());
    if !reachable[target_g] {
        return Err(CausalDagError::UnreachableGoal { target: req.target });
    }

    // Goal cone: reachable, not yet complete ancestors of the target. The walk
    // stops at complete nodes, whose own ancestors no longer matter, at
    // unreachable ones, which no plan can use, and at OR groups the initial
    // set already satisfies.
    let mut in_cone = vec![false; n];
    let mut cone = Vec::new();
    let mut stack = vec![target_g];
    in_cone[target_g] = true;
    while let Some(gi) = stack.pop() {
        cone.push(gi);
        let node = &req.nodes[gi];
        let open_groups = node
            .or_parents
            .iter()
            .filter(|group| !group.iter().any(|p| initial[index[p]]));
        for p in node.and_parents.iter().chain(open_groups.flatten()) {
            let pg = index[p];
            if !in_cone[pg] && !initial[pg] && reachable[pg] {
                in_cone[pg] = true;
                stack.push(pg);
            }
        }
    }
    let local: HashMap<usize, usize> = cone.iter().enumerate().map(|(l, &g)| (g, l)).collect();
    if cone.len() > MAX_CONE_NODES {
        return Err(CausalDagError::TargetTooComplex {
            target: req.target,
            cone_nodes: cone.len(),
            max: MAX_CONE_NODES,
        });
    }

    let bit = |g: usize| 1u64 << local[&g];
    let needs: Vec<Need> = cone
        .iter()
        .map(|&gi| {
            let node = &req.nodes[gi];
            let mut and_mask = 0;
            for p in &node.and_parents {
                let pg = index[p];
                if !initial[pg] {
                    // Every non-initial AND parent is reachable (the node is)
                    // and an ancestor of the target, so it is in the cone.
                    and_mask |= bit(pg);
                }
            }
            let or_masks = node
                .or_parents
                .iter()
                .filter(|group| !group.iter().any(|p| initial[index[p]]))
                .map(|group| {
                    group
                        .iter()
                        .filter(|p| local.contains_key(&index[p]))
                        .fold(0u64, |m, p| m | bit(index[p]))
                })
                .collect();
            Need { and_mask, or_masks }
        })
        .collect();
    let target_bit = bit(target_g);

    let mut dist: HashMap<u64, f64> = HashMap::new();
    let mut prev: HashMap<u64, (u64, usize)> = HashMap::new();
    let mut heap = BinaryHeap::new();
    dist.insert(0, 0.0);
    heap.push(Entry { cost: 0.0, mask: 0 });
    let mut expanded = 0usize;
    while let Some(Entry { cost, mask }) = heap.pop() {
        if dist.get(&mask).is_some_and(|&d| d < cost) {
            continue;
        }
        if mask & target_bit != 0 {
            let mut steps = Vec::new();
            let mut at = mask;
            while let Some(&(from, li)) = prev.get(&at) {
                steps.push(li);
                at = from;
            }
            steps.reverse();
            let path: Vec<u32> = steps.iter().map(|&li| req.nodes[cone[li]].id).collect();
            let total_cost = steps
                .iter()
                .fold(0.0, |acc, &li| acc + req.nodes[cone[li]].cost);
            return Ok(CausalDagPlan {
                path,
                total_cost,
                expanded_states: expanded,
                cone_nodes: cone.len(),
            });
        }
        if expanded == MAX_EXPANDED_STATES {
            return Err(CausalDagError::StateLimitExceeded {
                expanded,
                max: MAX_EXPANDED_STATES,
            });
        }
        let elapsed = start.elapsed();
        if elapsed > limit {
            return Err(CausalDagError::DeadlineExceeded {
                elapsed_ms: elapsed.as_millis() as u64,
                limit_ms: limit.as_millis() as u64,
            });
        }
        expanded += 1;
        for (li, need) in needs.iter().enumerate() {
            let b = 1u64 << li;
            if mask & b != 0
                || mask & need.and_mask != need.and_mask
                || need.or_masks.iter().any(|g| mask & g == 0)
            {
                continue;
            }
            let next = mask | b;
            let next_cost = cost + req.nodes[cone[li]].cost;
            if next_cost > req.budget {
                continue;
            }
            if dist.get(&next).is_none_or(|&d| next_cost < d) {
                dist.insert(next, next_cost);
                prev.insert(next, (mask, li));
                heap.push(Entry {
                    cost: next_cost,
                    mask: next,
                });
            }
        }
    }
    Err(CausalDagError::BudgetExceeded {
        target: req.target,
        budget: req.budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u32, cost: f64, and_parents: &[u32], or_parents: &[&[u32]]) -> CausalNode {
        CausalNode {
            id,
            cost,
            and_parents: and_parents.to_vec(),
            or_parents: or_parents.iter().map(|g| g.to_vec()).collect(),
        }
    }

    fn request(
        nodes: Vec<CausalNode>,
        target: u32,
        initial: &[u32],
        budget: f64,
    ) -> CausalDagRequest {
        CausalDagRequest {
            nodes,
            target,
            initial_completed: initial.to_vec(),
            budget,
        }
    }

    #[test]
    fn linear_chain_completes_every_node_in_order() {
        let req = request(
            vec![
                node(1, 1.0, &[], &[]),
                node(2, 2.0, &[1], &[]),
                node(3, 3.0, &[2], &[]),
            ],
            3,
            &[],
            10.0,
        );
        let plan = exact_causal_plan(&req).unwrap();
        assert_eq!(plan.path, vec![1, 2, 3]);
        assert_eq!(plan.total_cost, 6.0);
        assert_eq!(plan.cone_nodes, 3);
    }

    #[test]
    fn initial_completed_nodes_are_not_paid_again() {
        let req = request(
            vec![
                node(1, 1.0, &[], &[]),
                node(2, 2.0, &[1], &[]),
                node(3, 3.0, &[2], &[]),
            ],
            3,
            &[1],
            10.0,
        );
        let plan = exact_causal_plan(&req).unwrap();
        assert_eq!(plan.path, vec![2, 3]);
        assert_eq!(plan.total_cost, 5.0);
    }

    #[test]
    fn or_group_takes_the_cheaper_branch_and_and_parents_are_all_paid() {
        // target 10 needs 4 AND one of {5, 6}. 5 needs 1 and 2 (cost 1+1+1),
        // 6 needs 3 (cost 10+1). 4 is shared.
        let req = request(
            vec![
                node(1, 1.0, &[], &[]),
                node(2, 1.0, &[], &[]),
                node(3, 10.0, &[], &[]),
                node(4, 2.0, &[], &[]),
                node(5, 1.0, &[1, 2], &[]),
                node(6, 1.0, &[3], &[]),
                node(10, 1.0, &[4], &[&[5, 6]]),
            ],
            10,
            &[],
            100.0,
        );
        let plan = exact_causal_plan(&req).unwrap();
        assert_eq!(plan.total_cost, 6.0);
        assert!(!plan.path.contains(&3) && !plan.path.contains(&6));
        assert_eq!(*plan.path.last().unwrap(), 10);
        let pos = |id| plan.path.iter().position(|&x| x == id).unwrap();
        assert!(pos(1) < pos(5) && pos(2) < pos(5) && pos(5) < pos(10) && pos(4) < pos(10));
    }

    #[test]
    fn shared_ancestor_is_paid_once_where_a_tree_sum_would_double_count() {
        // 10 needs 5 and 6; both need the expensive 1. Exact cost 8+1+1+1 = 11,
        // not the tree sum 8+1 + 8+1 + 1 = 19.
        let req = request(
            vec![
                node(1, 8.0, &[], &[]),
                node(5, 1.0, &[1], &[]),
                node(6, 1.0, &[1], &[]),
                node(10, 1.0, &[5, 6], &[]),
            ],
            10,
            &[],
            11.0,
        );
        assert_eq!(exact_causal_plan(&req).unwrap().total_cost, 11.0);
    }

    #[test]
    fn or_group_satisfied_by_initial_set_costs_nothing() {
        let req = request(
            vec![
                node(1, 5.0, &[], &[]),
                node(2, 5.0, &[], &[]),
                node(3, 1.0, &[], &[&[1, 2]]),
            ],
            3,
            &[2],
            1.0,
        );
        let plan = exact_causal_plan(&req).unwrap();
        assert_eq!(plan.path, vec![3]);
        assert_eq!(plan.cone_nodes, 1);
    }

    #[test]
    fn over_budget_is_refused_and_exact_budget_is_accepted() {
        let nodes = vec![node(1, 2.0, &[], &[]), node(2, 3.0, &[1], &[])];
        let err = exact_causal_plan(&request(nodes.clone(), 2, &[], 4.0)).unwrap_err();
        assert_eq!(
            err,
            CausalDagError::BudgetExceeded {
                target: 2,
                budget: 4.0
            }
        );
        let plan = exact_causal_plan(&request(nodes, 2, &[], 5.0)).unwrap();
        assert_eq!(plan.total_cost, 5.0);
    }

    #[test]
    fn deadlocked_prerequisite_is_unreachable_without_search() {
        // 3 needs 2, 2 needs 1 AND 4. 4 needs 5 and 5 needs 4: a prerequisite
        // cycle nothing outside it can enter.
        let req = request(
            vec![
                node(1, 1.0, &[], &[]),
                node(4, 1.0, &[], &[&[5]]),
                node(5, 1.0, &[4], &[]),
                node(2, 1.0, &[1, 4], &[]),
                node(3, 1.0, &[2], &[]),
            ],
            3,
            &[],
            100.0,
        );
        assert_eq!(
            exact_causal_plan(&req).unwrap_err(),
            CausalDagError::UnreachableGoal { target: 3 }
        );
    }

    #[test]
    fn goal_cone_drops_unrelated_nodes() {
        // 3-node chain plus 200 unrelated nodes. Without pruning the cone cap
        // would refuse; with it the search sees 3 nodes.
        let mut nodes = vec![
            node(0, 1.0, &[], &[]),
            node(1, 1.0, &[0], &[]),
            node(2, 1.0, &[1], &[]),
        ];
        // Descendants of the target and free-standing roots: none can help.
        for id in 100..300 {
            let parent = if id == 100 { 2 } else { id - 1 };
            nodes.push(node(
                id,
                0.5,
                if id % 2 == 0 {
                    &[]
                } else {
                    std::slice::from_ref(&parent)
                },
                &[],
            ));
        }
        let req = request(nodes, 2, &[], 10.0);
        let plan = exact_causal_plan(&req).unwrap();
        assert_eq!(plan.path, vec![0, 1, 2]);
        assert_eq!(plan.cone_nodes, 3);
        assert_eq!(plan.expanded_states, 3);
    }

    #[test]
    fn cone_stops_at_completed_nodes() {
        // 40 ancestors behind node 50, but 50 is already complete.
        let mut nodes: Vec<CausalNode> = (0..40).map(|i| node(i, 1.0, &[], &[])).collect();
        let all: Vec<u32> = (0..40).collect();
        nodes.push(node(50, 1.0, &all, &[]));
        nodes.push(node(51, 1.0, &[50], &[]));
        let plan = exact_causal_plan(&request(nodes, 51, &[50], 10.0)).unwrap();
        assert_eq!(plan.path, vec![51]);
        assert_eq!(plan.cone_nodes, 1);
    }

    #[test]
    fn cone_above_cap_is_too_complex() {
        let mut nodes: Vec<CausalNode> = (0..33).map(|i| node(i, 1.0, &[], &[])).collect();
        let all: Vec<u32> = (0..33).collect();
        nodes.push(node(99, 1.0, &all, &[]));
        assert_eq!(
            exact_causal_plan(&request(nodes, 99, &[], 1e9)).unwrap_err(),
            CausalDagError::TargetTooComplex {
                target: 99,
                cone_nodes: 34,
                max: MAX_CONE_NODES
            }
        );
    }

    #[test]
    fn wide_independent_cone_hits_the_state_limit() {
        // 31 independent roots, all AND parents of the target: the subset
        // lattice has 2^31 states, far past the expansion limit. This run
        // takes a few seconds in a debug build, well past PLAN_DEADLINE, so
        // it must use a deadline long enough to let the state limit win.
        let mut nodes: Vec<CausalNode> = (0..31).map(|i| node(i, 1.0, &[], &[])).collect();
        let all: Vec<u32> = (0..31).collect();
        nodes.push(node(99, 1.0, &all, &[]));
        assert_eq!(
            plan_within(&request(nodes, 99, &[], 1e9), Duration::from_secs(600)).unwrap_err(),
            CausalDagError::StateLimitExceeded {
                expanded: MAX_EXPANDED_STATES,
                max: MAX_EXPANDED_STATES
            }
        );
    }

    #[test]
    fn zero_deadline_refuses_before_any_expansion() {
        // Same wide-cone request as above, but with no time budget at all:
        // the very first loop iteration must see an elapsed deadline.
        let mut nodes: Vec<CausalNode> = (0..31).map(|i| node(i, 1.0, &[], &[])).collect();
        let all: Vec<u32> = (0..31).collect();
        nodes.push(node(99, 1.0, &all, &[]));
        let err = plan_within(&request(nodes, 99, &[], 1e9), Duration::ZERO).unwrap_err();
        assert!(
            matches!(err, CausalDagError::DeadlineExceeded { limit_ms: 0, .. }),
            "{err}"
        );
    }

    #[test]
    fn reverse_listed_65536_chain_closes_in_linear_time() {
        // Chain i needs i-1, node 0 a free root, but `nodes` lists node 65535
        // first and node 0 last. The old repeated full-scan fixpoint grew the
        // closure by exactly one node per pass when walked backwards like
        // this: ~65536 passes over 65536 nodes, about 4e9 node checks. The
        // worklist closure does not care which order `nodes` lists them in.
        const N: u32 = 65_536;
        let nodes: Vec<CausalNode> = (0..N)
            .rev()
            .map(|i| {
                if i == 0 {
                    node(i, 1.0, &[], &[])
                } else {
                    node(i, 1.0, &[i - 1], &[])
                }
            })
            .collect();
        let start = Instant::now();
        let plan = exact_causal_plan(&request(nodes, 4, &[], 10.0)).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(plan.path, vec![0, 1, 2, 3, 4]);
        assert_eq!(plan.cone_nodes, 5);
        assert!(elapsed < Duration::from_secs(1), "took {elapsed:?}");
    }

    #[test]
    fn reverse_listed_65536_chain_deep_target_is_refused_fast() {
        // Same reverse-listed chain, but the target is the far end: the
        // whole chain is in the cone, well past MAX_CONE_NODES. Refused
        // without running the exponential search, and fast despite the
        // closure pass over all 65536 nodes.
        const N: u32 = 65_536;
        let nodes: Vec<CausalNode> = (0..N)
            .rev()
            .map(|i| {
                if i == 0 {
                    node(i, 1.0, &[], &[])
                } else {
                    node(i, 1.0, &[i - 1], &[])
                }
            })
            .collect();
        let start = Instant::now();
        let err = exact_causal_plan(&request(nodes, N - 1, &[], 1e9)).unwrap_err();
        let elapsed = start.elapsed();
        assert_eq!(
            err,
            CausalDagError::TargetTooComplex {
                target: N - 1,
                cone_nodes: N as usize,
                max: MAX_CONE_NODES
            }
        );
        assert!(elapsed < Duration::from_secs(1), "took {elapsed:?}");
    }

    #[test]
    fn or_branch_behind_a_deadlock_does_not_inflate_the_cone() {
        // T needs one of {A, B}. A is a free root. B needs C (AND), and C
        // sits in a 101-node AND cycle (C -> D1 -> D2 -> ... -> D100 -> C)
        // that nothing outside it can enter, so B, C and the whole D chain
        // are unreachable. The cheap branch through A must win, and the
        // deadlocked branch must not inflate the cone.
        const A: u32 = 1;
        const B: u32 = 2;
        const C: u32 = 3;
        const TARGET: u32 = 1000;
        let mut nodes = vec![
            node(A, 1.0, &[], &[]),
            node(B, 1.0, &[C], &[]),
            node(C, 1.0, &[4], &[]), // C needs D1
        ];
        for k in 1..=99u32 {
            let this = 3 + k; // D_k: ids 4..=102
            let next = 3 + k + 1; // D_{k+1}: ids 5..=103
            nodes.push(node(this, 1.0, &[next], &[])); // D_k needs D_{k+1}
        }
        nodes.push(node(103, 1.0, &[C], &[])); // D100 needs C, closing the cycle
        nodes.push(node(TARGET, 1.0, &[], &[&[A, B]]));
        let plan = exact_causal_plan(&request(nodes, TARGET, &[], 10.0)).unwrap();
        assert_eq!(plan.path, vec![A, TARGET]);
        assert_eq!(plan.cone_nodes, 2);
    }

    #[test]
    fn target_already_complete_is_an_empty_plan() {
        let plan = exact_causal_plan(&request(vec![node(1, 4.0, &[], &[])], 1, &[1], 0.0)).unwrap();
        assert!(plan.path.is_empty());
        assert_eq!(plan.total_cost, 0.0);
    }

    #[test]
    fn enterable_cycle_is_planned_through_its_entry() {
        // 1 and 2 need each other, but 1 can also start from 3.
        let req = request(
            vec![
                node(3, 1.0, &[], &[]),
                node(1, 1.0, &[], &[&[2, 3]]),
                node(2, 1.0, &[], &[&[1]]),
                node(9, 1.0, &[2], &[]),
            ],
            9,
            &[],
            100.0,
        );
        let plan = exact_causal_plan(&req).unwrap();
        assert_eq!(plan.path, vec![3, 1, 2, 9]);
        assert_eq!(plan.total_cost, 4.0);
    }

    #[test]
    fn malformed_inputs_are_refused() {
        let ok = || {
            request(
                vec![node(1, 1.0, &[], &[]), node(2, 1.0, &[1], &[])],
                2,
                &[],
                5.0,
            )
        };
        let mut cases: Vec<(&str, CausalDagRequest)> = Vec::new();
        let mut r = ok();
        r.nodes.clear();
        cases.push(("empty", r));
        let mut r = ok();
        r.nodes.push(node(1, 1.0, &[], &[]));
        cases.push(("duplicate", r));
        let mut r = ok();
        r.nodes[0].cost = -1.0;
        cases.push(("negative cost", r));
        let mut r = ok();
        r.nodes[0].cost = f64::INFINITY;
        cases.push(("infinite cost", r));
        let mut r = ok();
        r.budget = f64::NAN;
        cases.push(("nan budget", r));
        let mut r = ok();
        r.nodes[1].and_parents = vec![7];
        cases.push(("unknown parent", r));
        let mut r = ok();
        r.nodes[1].or_parents = vec![vec![]];
        cases.push(("empty or group", r));
        let mut r = ok();
        r.nodes[1].and_parents = vec![2];
        cases.push(("self parent", r));
        let mut r = ok();
        r.target = 9;
        cases.push(("unknown target", r));
        let mut r = ok();
        r.initial_completed = vec![1, 1];
        cases.push(("duplicate initial", r));
        let mut r = ok();
        r.initial_completed = vec![8];
        cases.push(("unknown initial", r));
        for (name, req) in cases {
            assert!(
                matches!(
                    exact_causal_plan(&req),
                    Err(CausalDagError::InvalidInput(_))
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn exact_search_matches_brute_force_on_small_random_dags() {
        // Enumerate every completion order of up to 7 nodes and compare.
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..200 {
            let n = rng.gen_range(2..=7u32);
            let mut nodes = Vec::new();
            for id in 0..n {
                let mut and_p = Vec::new();
                let mut or_p = Vec::new();
                for p in 0..id {
                    match rng.gen_range(0..4) {
                        0 => and_p.push(p),
                        1 => or_p.push(p),
                        _ => {}
                    }
                }
                let groups = if or_p.is_empty() { vec![] } else { vec![or_p] };
                nodes.push(CausalNode {
                    id,
                    cost: rng.gen_range(0..5) as f64,
                    and_parents: and_p,
                    or_parents: groups,
                });
            }
            let req = request(nodes, n - 1, &[], 1e9);
            let best = brute_force(&req, 0, 0.0);
            match exact_causal_plan(&req) {
                Ok(plan) => assert_eq!(Some(plan.total_cost), best, "{req:?}"),
                Err(e) => panic!("{e} on {req:?}"),
            }
        }
    }

    fn brute_force(req: &CausalDagRequest, mask: u32, cost: f64) -> Option<f64> {
        let target = req.target as usize;
        if mask & (1 << target) != 0 {
            return Some(cost);
        }
        let done = |id: &u32| mask & (1 << id) != 0;
        let mut best: Option<f64> = None;
        for node in &req.nodes {
            let id = node.id as usize;
            if mask & (1 << id) != 0
                || !node.and_parents.iter().all(done)
                || !node.or_parents.iter().all(|g| g.iter().any(done))
            {
                continue;
            }
            if let Some(c) = brute_force(req, mask | (1 << id), cost + node.cost) {
                best = Some(best.map_or(c, |b: f64| b.min(c)));
            }
        }
        best
    }

    #[test]
    fn request_rejects_unknown_fields() {
        let text = r#"{"nodes":[{"id":1,"cost":1}],"target":1,"budget":1,"heuristic":"greedy"}"#;
        assert!(serde_json::from_str::<CausalDagRequest>(text).is_err());
    }
}
