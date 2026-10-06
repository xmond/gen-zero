//! The 6 Orthogonal Planning Engines of Gen-Zero.
//!
//! 1. MctsEngine: Finite-horizon PUCT tree search with discounted backups
//! 2. AStarEngine: Multi-step graph search with a zero admissible heuristic
//! 3. MpcCemEngine: Horizon rolling + Cross-Entropy Method
//! 4. ManifoldGFlowNetEngine: Exact depth-one reward-proportional flow sampling
//! 5. CfrNashEngine: Two-player normal-form cumulative regret matching
//! 6. CpSatFormalEngine: Bounded finite-candidate feasibility/optimality enumeration
//!
//! Fail-closed contract: a world-model `Err` aborts the plan with
//! `PlannerError::Core`. No engine turns a failed transition into a skipped
//! action, a zero reward, or a default choice. An engine that finds no action
//! allowed by the PolicyGate returns `NoFeasibleAction`, never `actions[0]`.

use crate::error::PlannerError;
use gen_zero_core::{
    ActionId, CoreError, FullLatent, LocalActionFrame, NormalizedEntropy, WorldModelDynamics,
};
use gen_zero_gate::{GateVerdict, PolicyGate, PolicyTier};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

/// Finite, JSON-safe value charged when a dynamics mask leaves no legal action.
///
/// This is intentionally much larger than ordinary planner rewards while still
/// remaining finite in `f32`, so a dead end cannot look like a zero-cost normal
/// terminal state and serialized plans never contain `-Infinity`/`NaN`.
pub const DEAD_END_PENALTY: f32 = -1.0e30;
const DEAD_END_PENALTY_F64: f64 = -1.0e30;

/// Reject a latent before it enters an engine's numerical scoring path.
///
/// Checking the coordinates alone is insufficient: a finite latent can still
/// have an overflowing f32 norm, which would otherwise turn a later distance
/// or cost into infinity.
fn validate_state(state: &FullLatent, label: &str) -> Result<(), PlannerError> {
    if let Some(i) = state.as_slice().iter().position(|x| !x.is_finite()) {
        return Err(PlannerError::DivergentState(format!(
            "{label} coordinate {i} is not finite"
        )));
    }
    let norm = state.l2_norm();
    if !norm.is_finite() {
        return Err(PlannerError::DivergentState(format!(
            "{label} L2 norm overflows ({norm})"
        )));
    }
    Ok(())
}

/// Validate every numerical output, including successors unused by depth-one rankers.
/// Call immediately after `step`, before scoring or storing model output.
fn validate_transition(
    next_state: &FullLatent,
    reward: f32,
    label: &str,
) -> Result<(), PlannerError> {
    validate_state(next_state, &format!("{label} successor state"))?;
    if !reward.is_finite() {
        return Err(PlannerError::DivergentState(format!(
            "{label} reward {reward} is not finite"
        )));
    }
    Ok(())
}

fn validate_score(value: f32, label: &str) -> Result<(), PlannerError> {
    if !value.is_finite() {
        return Err(PlannerError::DivergentState(format!(
            "{label} {value} is not finite"
        )));
    }
    Ok(())
}

/// A policy is over distinct actions, not duplicate slots in a local frame.
fn validate_actions(actions: &[ActionId]) -> Result<(), PlannerError> {
    if actions.is_empty() {
        return Err(PlannerError::NoFeasibleAction);
    }
    for (index, action) in actions.iter().enumerate() {
        if actions[..index].contains(action) {
            return Err(PlannerError::InvalidInput(format!(
                "duplicate candidate action {}",
                action.0
            )));
        }
    }
    Ok(())
}

/// Shannon entropy of a normalized categorical policy, with the exact 0 ln 0 = 0
/// convention. f64 accumulation preserves small positive probability terms.
pub(crate) fn shannon_entropy(probabilities: &[f32]) -> NormalizedEntropy {
    if probabilities.len() <= 1 {
        return NormalizedEntropy::ZERO;
    }
    let entropy: f64 = probabilities
        .iter()
        .copied()
        .filter(|&p| p > 0.0)
        .map(|p| {
            let p = f64::from(p);
            -p * p.ln()
        })
        .sum();
    NormalizedEntropy((entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0) as f32)
}

/// Cooperative search deadline. Synchronous model calls cannot be preempted here;
/// `ProductionPipeline::decide` supplies the bounded caller-side wait.
pub struct SearchBudget<'a> {
    pub deadline: Option<std::time::Instant>,
    pub(crate) started: std::time::Instant,
    pub(crate) publish: Option<&'a (dyn Fn(ActionId, &'static str) + Sync)>,
    /// Optional caller-owned marker for a dead end observed during search.
    /// Keeping this out of the engine return tuple preserves router compatibility
    /// while allowing `ProductionPipeline` to expose the observation.
    pub(crate) dead_end: Option<&'a AtomicBool>,
}

impl Default for SearchBudget<'_> {
    fn default() -> Self {
        Self::new(None)
    }
}

impl<'a> SearchBudget<'a> {
    pub fn new(deadline: Option<std::time::Instant>) -> Self {
        Self {
            deadline,
            started: std::time::Instant::now(),
            publish: None,
            dead_end: None,
        }
    }

    /// Attach a caller-owned marker that engines set whenever dynamics masking
    /// produces an empty action set below the root.
    pub fn with_dead_end_marker(mut self, marker: &'a AtomicBool) -> Self {
        self.dead_end = Some(marker);
        self
    }

    /// Record a dynamics dead end without changing the search result shape.
    pub(crate) fn mark_dead_end(&self) {
        if let Some(marker) = self.dead_end {
            marker.store(true, AtomicOrdering::Release);
        }
    }

    /// Read the caller-owned dead-end observation, if one was attached.
    pub fn has_dead_end(&self) -> bool {
        self.dead_end
            .is_some_and(|marker| marker.load(AtomicOrdering::Acquire))
    }

    pub fn check(&self) -> Result<(), PlannerError> {
        if self
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            Err(PlannerError::TimeoutExceeded(
                self.started.elapsed().as_secs_f64() * 1000.0,
            ))
        } else {
            Ok(())
        }
    }

    fn candidate(&self, action: ActionId, engine: &'static str) -> Result<(), PlannerError> {
        self.check()?;
        if let Some(publish) = self.publish {
            publish(action, engine);
        }
        Ok(())
    }
}

/// Search result with explicit mask-dead-end diagnostics.
#[derive(Clone, Debug)]
pub struct PlanReport {
    pub action: ActionId,
    pub entropy: NormalizedEntropy,
    /// Any explored branch reached a masked dead end, not necessarily the winner.
    pub has_dead_end: bool,
}

/// Core interface implemented by all 6 orthogonal planning engines.
pub trait PlanningEngine: Send + Sync {
    fn name(&self) -> &'static str;

    fn plan(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        self.plan_until(state, actions, world_model, gate, &SearchBudget::default())
    }

    fn plan_report(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
    ) -> Result<PlanReport, PlannerError> {
        let marker = AtomicBool::new(false);
        let budget = SearchBudget::default().with_dead_end_marker(&marker);
        let (action, entropy) = self.plan_until(state, actions, world_model, gate, &budget)?;
        Ok(PlanReport {
            action,
            entropy,
            has_dead_end: budget.has_dead_end(),
        })
    }

    fn plan_until(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        // Tuple callers have no place for degradation metadata. Refuse a
        // degraded result explicitly unless the caller installed an observer.
        if budget.dead_end.is_none() {
            let marker = AtomicBool::new(false);
            let observed = SearchBudget {
                deadline: budget.deadline,
                started: budget.started,
                publish: budget.publish,
                dead_end: Some(&marker),
            };
            let result = self.plan_until(state, actions, world_model, gate, &observed)?;
            if observed.has_dead_end() {
                return Err(PlannerError::DeadEndRequiresReport);
            }
            return Ok(result);
        }
        if budget.deadline.is_none() {
            return self.search(state, actions, world_model, gate, budget);
        }
        let incumbent = std::sync::Mutex::new(None);
        let publish = |action, engine| {
            if budget.check().is_ok() {
                if let Some(callback) = budget.publish {
                    callback(action, engine);
                }
                if budget.check().is_ok() {
                    *incumbent.lock().expect("local incumbent") = Some(action);
                }
            }
        };
        let search = SearchBudget {
            deadline: budget.deadline,
            started: budget.started,
            publish: Some(&publish),
            dead_end: budget.dead_end,
        };
        match self.search(state, actions, world_model, gate, &search) {
            Err(error @ PlannerError::TimeoutExceeded(_)) => incumbent
                .into_inner()
                .expect("local incumbent")
                .map(|action| (action, NormalizedEntropy(1.0)))
                .ok_or(error),
            result => result,
        }
    }

    /// Search implementation; callers should use `plan` or `plan_until`.
    fn search(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError>;
}

// =========================================================================
// 1. MctsEngine (finite-horizon PUCT tree search)
// =========================================================================

/// Sequential PUCT search with one persistent edge expansion per simulation,
/// followed by a balanced deterministic rollout to `horizon` (or `done`).
/// Each node caches its model successor: this assumes deterministic dynamics.
/// The candidate frame is filtered by dynamics at each node and rollout state.
/// Backups store discounted return-to-go, including the node's incoming reward.
/// Leaves at the horizon/terminal have zero continuation value; no learned
/// value function or optimality guarantee is implied. Root visits define the
/// returned action and normalized Shannon entropy.
#[derive(Clone)]
pub struct MctsEngine {
    pub max_simulations: usize,
    pub c_puct: f32,
    /// Maximum persistent nodes, including the root. Exhaustion is an error.
    pub arena_capacity: usize,
    pub horizon: usize,
    pub discount: f32,
}

impl Default for MctsEngine {
    fn default() -> Self {
        Self {
            max_simulations: 128,
            c_puct: 1.414,
            arena_capacity: 1024,
            horizon: 4,
            discount: 0.99,
        }
    }
}

/// Owned state arena for sequential search; indices remain stable as it grows.
struct MctsNode {
    action: Option<ActionId>,
    reward: f64,
    done: bool,
    dead_end: bool,
    children: smallvec::SmallVec<[usize; 16]>,
    visits: usize,
    value_sum: f64,
}

impl MctsNode {
    fn new(action: Option<ActionId>, reward: f32, done: bool) -> Self {
        Self {
            action,
            reward: f64::from(reward),
            done,
            dead_end: false,
            children: smallvec::SmallVec::new(),
            visits: 0,
            value_sum: 0.0,
        }
    }
}

/// Back up each node's return-to-go; the root has no incoming edge.
fn mcts_backup(
    nodes: &mut [MctsNode],
    path: &[usize],
    mut value: f64,
    discount: f64,
) -> Result<(), PlannerError> {
    for &idx in path.iter().rev() {
        let node = &mut nodes[idx];
        if idx != 0 {
            value = node.reward + discount * value;
        }
        node.value_sum += value;
        if !node.value_sum.is_finite() {
            return Err(PlannerError::DivergentState("MCTS backup overflow".into()));
        }
        node.visits += 1;
    }
    Ok(())
}

impl PlanningEngine for MctsEngine {
    fn name(&self) -> &'static str {
        "MctsEngine"
    }

    fn search(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        validate_state(state, "MCTS input state")?;
        if self.max_simulations == 0
            || self.horizon == 0
            || !self.c_puct.is_finite()
            || self.c_puct < 0.0
            || !self.discount.is_finite()
            || !(0.0..=1.0).contains(&self.discount)
        {
            return Err(PlannerError::InvalidInput(
                "MCTS requires positive simulations/horizon, finite nonnegative c_puct and discount in [0, 1]".into(),
            ));
        }
        let valid_actions: Vec<_> = actions
            .actions()
            .iter()
            .copied()
            .filter(|&act| {
                gate.evaluate_basic(act, NormalizedEntropy::ZERO)
                    .map(|v| v.tier != PolicyTier::Tier3HardStop)
                    .unwrap_or(false)
            })
            .collect();
        let root_actions = world_model.allowed_actions(state, &valid_actions)?;
        if root_actions.is_empty() {
            budget.mark_dead_end();
            return Err(PlannerError::NoFeasibleAction);
        }
        if self.arena_capacity == 0 {
            return Err(PlannerError::ArenaCapacityExceeded { capacity: 0 });
        }
        let reserve = self
            .arena_capacity
            .min(self.max_simulations.saturating_add(1))
            .min(4096);
        let mut nodes = Vec::with_capacity(reserve);
        let mut states = Vec::with_capacity(reserve.min(64));
        nodes.push(MctsNode::new(None, 0.0, false));
        states.push(state.clone());
        let mut path = Vec::with_capacity(self.horizon.saturating_add(1));
        let gamma = f64::from(self.discount);
        for simulation in 0..self.max_simulations {
            budget.check()?;
            path.clear();
            path.push(0);
            let mut current = 0;
            let mut depth = 0;
            let mut encountered_dead_end = nodes[current].dead_end;
            // Selection descends through fully expanded nodes. Expand the first
            // missing action at the selected node, never a root-state replay.
            while depth < self.horizon && !nodes[current].done && !nodes[current].dead_end {
                budget.check()?;
                let node_actions = world_model.allowed_actions(&states[current], &valid_actions)?;
                if node_actions.is_empty() {
                    nodes[current].dead_end = true;
                    encountered_dead_end = true;
                    budget.mark_dead_end();
                    break;
                }
                if let Some(action) = node_actions.iter().copied().find(|a| {
                    !nodes[current]
                        .children
                        .iter()
                        .any(|&i| nodes[i].action == Some(*a))
                }) {
                    if nodes.len() >= self.arena_capacity {
                        return Err(PlannerError::ArenaCapacityExceeded {
                            capacity: self.arena_capacity,
                        });
                    }
                    budget.check()?;
                    let (next, reward, done) = world_model.step(&states[current], action)?;
                    budget.check()?;
                    validate_transition(&next, reward, "MCTS expansion")?;
                    let child = nodes.len();
                    nodes.push(MctsNode::new(Some(action), reward, done));
                    states.push(next);
                    nodes[current].children.push(child);
                    current = child;
                    path.push(current);
                    depth += 1;
                    break;
                }
                let parent = &nodes[current];
                let scale = f64::from(self.c_puct) / node_actions.len() as f64
                    * ((parent.visits + 1) as f64).sqrt();
                if parent.children.len() > 16 {
                    return Err(PlannerError::InvalidInput(
                        "MCTS action frame exceeds 16 children".into(),
                    ));
                }
                // Score contiguous lanes first, then select the earliest maximum.
                let mut scores = [0.0_f64; 16];
                for (lane, &child) in parent.children.iter().enumerate() {
                    let node = &nodes[child];
                    let visits = node.visits as f64;
                    scores[lane] = node.value_sum / visits + scale / (1.0 + visits);
                }
                let mut best = parent.children[0];
                let mut best_score = f64::NEG_INFINITY;
                for (lane, &child) in parent.children.iter().enumerate() {
                    if scores[lane] > best_score {
                        best_score = scores[lane];
                        best = child;
                    }
                }
                current = best;
                path.push(current);
                depth += 1;
            }
            // Evaluate the unexpanded tail. Rotate rollout actions across
            // simulations/depths; this is a heuristic policy, not a value oracle.
            encountered_dead_end |= nodes[current].dead_end;
            let mut rollout_state = states[current].clone();
            let mut done = nodes[current].done || nodes[current].dead_end;
            let mut continuation = if encountered_dead_end {
                DEAD_END_PENALTY_F64
            } else {
                0.0
            };
            let mut weight = 1.0_f64;
            while depth < self.horizon && !done {
                budget.check()?;
                let rollout_actions =
                    world_model.allowed_actions(&rollout_state, &valid_actions)?;
                if rollout_actions.is_empty() {
                    // A masked state is a real terminal failure for search. Keep
                    // the penalty finite and attach it at the current rollout
                    // discount; never turn this into a zero-cost normal leaf.
                    budget.mark_dead_end();
                    continuation += weight * DEAD_END_PENALTY_F64;
                    break;
                }
                let action = rollout_actions[(simulation % rollout_actions.len()
                    + depth % rollout_actions.len())
                    % rollout_actions.len()];
                budget.check()?;
                let (next, reward, terminal) = world_model.step(&rollout_state, action)?;
                budget.check()?;
                validate_transition(&next, reward, "MCTS rollout")?;
                continuation += weight * f64::from(reward);
                weight *= gamma;
                rollout_state = next;
                done = terminal;
                depth += 1;
            }
            mcts_backup(&mut nodes, &path, continuation, gamma)?;
            if let Some(&root_child) = path.get(1) {
                if !nodes[root_child].done {
                    budget.candidate(nodes[root_child].action.expect("root edge"), self.name())?;
                }
            }
        }
        if nodes[0].children.is_empty() {
            // A stateful dynamics mask may have closed the root after the
            // initial preflight. The marker above records that dead end; do not
            // index an empty root or fabricate an action.
            return Err(PlannerError::NoFeasibleAction);
        }
        let children = &nodes[0].children;
        let mut best = children[0];
        let mut probabilities: Vec<_> = children
            .iter()
            .map(|&idx| {
                if nodes[idx].visits > nodes[best].visits {
                    best = idx;
                }
                nodes[idx].visits as f32 / nodes[0].visits as f32
            })
            .collect();
        probabilities.resize(root_actions.len(), 0.0);
        Ok((
            nodes[best].action.expect("root children carry actions"),
            NormalizedEntropy::from_probabilities(&probabilities),
        ))
    }
}

// =========================================================================
// 2. AStarEngine (Multi-step graph search with a zero admissible heuristic)
// =========================================================================

#[derive(Clone, Copy, PartialEq)]
struct AStarItem {
    f: f64,
    node: usize,
}
impl Eq for AStarItem {}
impl Ord for AStarItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .total_cmp(&self.f)
            .then_with(|| other.node.cmp(&self.node))
    }
}
impl PartialOrd for AStarItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// An explicit goal; model `done` alone is never interpreted as success.
#[derive(Clone)]
pub enum AStarGoal {
    Predicate(fn(&FullLatent) -> bool),
    WithinDistance {
        target: Box<FullLatent>,
        tolerance: f32,
    },
}
impl AStarGoal {
    fn reached(&self, state: &FullLatent) -> bool {
        match self {
            Self::Predicate(predicate) => predicate(state),
            Self::WithinDistance { target, tolerance } => {
                state
                    .as_slice()
                    .iter()
                    .zip(target.as_slice())
                    .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    <= f64::from(*tolerance)
            }
        }
    }
}

/// Complete minimum-cost path, available only after a goal is popped.
#[derive(Debug)]
pub struct AStarPath {
    pub actions: Vec<ActionId>,
    pub cost: f64,
    pub expanded: usize,
}

struct AStarNode {
    state: FullLatent,
    g: f64,
    parent: Option<(usize, ActionId)>,
    terminal: bool,
}

// Exact coordinates avoid merging distinct states. Normalize signed zero.
fn astar_key(state: &FullLatent, terminal: bool) -> (Vec<u32>, bool) {
    (
        state
            .as_slice()
            .iter()
            .map(|&x| if x == 0.0 { 0 } else { x.to_bits() })
            .collect(),
        terminal,
    )
}

/// A* with h=0 (Dijkstra): admissible without assuming a bound between latent
/// distance and reward. Edge cost is `1 + max(-reward, 0) + weight * 0.05 * distance`.
/// Positive rewards are not negative edges. Distance is a displacement penalty,
/// not calibrated model uncertainty. Optimality assumes deterministic Markov
/// dynamics, a pure goal predicate, and deterministic state-dependent action masks.
/// Exact state keys prevent cycles; near-equal floating states remain distinct.
/// No goal is configured by default: callers must explicitly define success.
/// Exhaustion/budget limits return errors, never an unproven partial-path action.
#[derive(Clone)]
pub struct AStarEngine {
    pub uncertainty_penalty_weight: f32,
    /// Retained for existing configuration migration; validated but no longer used to invent
    /// a policy distribution from single-step costs.
    pub boltzmann_temperature: f32,
    pub goal: Option<AStarGoal>,
    pub max_expansions: usize,
}
impl Default for AStarEngine {
    fn default() -> Self {
        Self {
            uncertainty_penalty_weight: 0.5,
            boltzmann_temperature: ASTAR_BOLTZMANN_TEMPERATURE,
            goal: None,
            max_expansions: 1024,
        }
    }
}
impl AStarEngine {
    pub fn search(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
    ) -> Result<AStarPath, PlannerError> {
        self.search_until(state, actions, world_model, gate, &SearchBudget::default())
    }

    fn search_until(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<AStarPath, PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        validate_state(state, "A* input state")?;
        validate_score(self.uncertainty_penalty_weight, "A* penalty weight")?;
        if self.uncertainty_penalty_weight < 0.0
            || self.max_expansions == 0
            || !self.boltzmann_temperature.is_finite()
            || self.boltzmann_temperature <= 0.0
        {
            return Err(PlannerError::InvalidInput(
                "invalid A* weight, temperature or expansion budget".into(),
            ));
        }
        let allowed: Vec<_> = actions
            .actions()
            .iter()
            .copied()
            .filter(|&action| {
                gate.evaluate_basic(action, NormalizedEntropy::ZERO)
                    .is_ok_and(|v| v.tier != PolicyTier::Tier3HardStop)
            })
            .collect();
        if world_model.allowed_actions(state, &allowed)?.is_empty() {
            return Err(PlannerError::NoFeasibleAction);
        }
        let goal = self.goal.as_ref().ok_or(PlannerError::MissingSearchGoal)?;
        if let AStarGoal::WithinDistance { target, tolerance } = goal {
            validate_state(target, "A* goal")?;
            if !tolerance.is_finite() || *tolerance < 0.0 {
                return Err(PlannerError::InvalidInput(
                    "A* goal tolerance must be finite and nonnegative".into(),
                ));
            }
        }
        let mut nodes = vec![AStarNode {
            state: state.clone(),
            g: 0.0,
            parent: None,
            terminal: false,
        }];
        let mut frontier = BinaryHeap::from([AStarItem { f: 0.0, node: 0 }]);
        let mut best = HashMap::from([(astar_key(state, false), 0.0)]);
        let mut closed = HashSet::new();
        let mut expanded = 0;
        while let Some(item) = frontier.pop() {
            budget.check()?;
            let node = &nodes[item.node];
            let key = astar_key(&node.state, node.terminal);
            if best.get(&key) != Some(&node.g) || closed.contains(&key) {
                continue;
            }
            if goal.reached(&node.state) {
                let cost = node.g;
                let mut path = Vec::new();
                let mut cursor = item.node;
                while let Some((parent, action)) = nodes[cursor].parent {
                    path.push(action);
                    cursor = parent;
                }
                path.reverse();
                if let Some(&first) = path.first() {
                    let mut first_node = item.node;
                    while let Some((parent, _)) = nodes[first_node].parent {
                        if parent == 0 {
                            break;
                        }
                        first_node = parent;
                    }
                    if !nodes[first_node].terminal {
                        budget.candidate(first, self.name())?;
                    }
                }
                return Ok(AStarPath {
                    actions: path,
                    cost,
                    expanded,
                });
            }
            closed.insert(key);
            if node.terminal {
                continue;
            }
            if expanded == self.max_expansions {
                return Err(PlannerError::SearchBudgetExceeded { expanded });
            }
            expanded += 1;
            let origin = node.state.clone();
            let g = node.g;
            let node_actions = world_model.allowed_actions(&origin, &allowed)?;
            if node_actions.is_empty() {
                continue; // DeadEnd: already closed, never enqueued again.
            }
            for &action in &node_actions {
                budget.check()?;
                budget.check()?;
                let (next, reward, terminal) = world_model.step(&origin, action)?;
                budget.check()?;
                validate_transition(&next, reward, "A* transition")?;
                let distance = gen_zero_core::l2_distance_f32(origin.as_slice(), next.as_slice());
                validate_score(distance, "A* distance")?;
                let penalty = self.uncertainty_penalty_weight * (0.05 * distance);
                validate_score(penalty, "A* displacement penalty")?;
                let next_g = g + 1.0 + f64::from((-reward).max(0.0)) + f64::from(penalty);
                if !next_g.is_finite() {
                    return Err(PlannerError::DivergentState("A* path cost overflow".into()));
                }
                let key = astar_key(&next, terminal);
                if closed.contains(&key) || best.get(&key).is_some_and(|&old| old <= next_g) {
                    continue;
                }
                best.insert(key, next_g);
                let index = nodes.len();
                nodes.push(AStarNode {
                    state: next,
                    g: next_g,
                    parent: Some((item.node, action)),
                    terminal,
                });
                // h=0 is a consistent lower bound for the nonnegative edge costs.
                frontier.push(AStarItem {
                    f: next_g + 0.0,
                    node: index,
                });
            }
        }
        Err(PlannerError::SearchGoalUnreachable)
    }
}
impl PlanningEngine for AStarEngine {
    fn name(&self) -> &'static str {
        "AStarEngine"
    }
    fn search(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        let path = self.search_until(state, actions, world_model, gate, budget)?;
        let action = path
            .actions
            .first()
            .copied()
            .ok_or(PlannerError::SearchAlreadyAtGoal)?;
        let mut first_step_costs = Vec::new();
        for act in world_model.allowed_actions(state, actions.actions())? {
            if let Ok(verdict) = gate.evaluate_basic(act, NormalizedEntropy::ZERO) {
                if verdict.tier != PolicyTier::Tier3HardStop {
                    let (next, reward, _) = world_model.step(state, act)?;
                    validate_transition(&next, reward, "A* entropy")?;
                    let step_cost = (1.0 - reward).max(0.0);
                    first_step_costs.push(step_cost);
                }
            }
        }
        let entropy = astar_boltzmann_entropy(&first_step_costs, self.boltzmann_temperature);
        Ok((action, entropy))
    }
}

fn astar_boltzmann_entropy(costs: &[f32], temperature: f32) -> NormalizedEntropy {
    if costs.len() <= 1 {
        return NormalizedEntropy::ZERO;
    }
    let min_cost = costs.iter().copied().fold(f32::INFINITY, f32::min);
    let mut exps: Vec<f32> = costs
        .iter()
        .map(|&c| (-((c - min_cost) / temperature)).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 && sum.is_finite() {
        for p in &mut exps {
            *p /= sum;
        }
        shannon_entropy(&exps)
    } else {
        NormalizedEntropy(1.0)
    }
}

const ASTAR_BOLTZMANN_TEMPERATURE: f32 = 0.1;

// =========================================================================
// 3. MpcCemEngine (Cross-Entropy Method / Horizon Rolling)
// =========================================================================

/// Native CEM parameterizations. Gaussian variance is diagonal variance, not
/// standard deviation. Continuous execution requires a continuous world-model
/// and gate adapter; the ActionId-based PlanningEngine only uses Categorical.
#[derive(Clone, Debug, PartialEq)]
pub enum CemDistribution {
    Categorical {
        probabilities: Vec<Vec<f32>>,
    },
    DiagonalGaussian {
        means: Vec<Vec<f32>>,
        variances: Vec<Vec<f32>>,
    },
}

impl CemDistribution {
    /// Construct H independent D-dimensional Gaussian parameter vectors.
    pub fn diagonal_gaussian(
        means: Vec<Vec<f32>>,
        variances: Vec<Vec<f32>>,
    ) -> Result<Self, PlannerError> {
        let dimension = means.first().map_or(0, Vec::len);
        if dimension == 0
            || means.len() != variances.len()
            || means
                .iter()
                .any(|row| row.len() != dimension || row.iter().any(|v| !v.is_finite()))
            || variances
                .iter()
                .any(|row| row.len() != dimension || row.iter().any(|v| !v.is_finite() || *v < 0.0))
        {
            return Err(PlannerError::InvalidInput(
                "invalid H x D diagonal Gaussian parameters".into(),
            ));
        }
        Ok(Self::DiagonalGaussian { means, variances })
    }
}

/// Inspectable optimization result. Only executed prefixes of terminal
/// trajectories contribute to distribution updates.
#[derive(Clone, Debug)]
pub struct CemPlan {
    pub actions: Vec<ActionId>,
    pub score: f32,
    pub distribution: CemDistribution,
    /// Whether any sampled trajectory reached a state with no legal dynamics
    /// action. The score already includes [`DEAD_END_PENALTY`] for each such
    /// trajectory; this flag makes the degradation explicit to callers.
    pub has_dead_end: bool,
}

#[derive(Clone)]
pub struct MpcCemEngine {
    pub num_samples: usize,
    pub num_elites: usize,
    pub horizon: usize,
    pub num_iterations: usize,
    /// Geometric reward discount in [0, 1].
    pub gamma: f32,
}

impl Default for MpcCemEngine {
    fn default() -> Self {
        Self {
            num_samples: 32,
            num_elites: 8,
            horizon: 4,
            num_iterations: 3,
            gamma: 0.95,
        }
    }
}

impl MpcCemEngine {
    pub fn optimize(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
    ) -> Result<CemPlan, PlannerError> {
        self.optimize_until(state, actions, world_model, gate, &SearchBudget::default())
    }

    fn optimize_until(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<CemPlan, PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        validate_state(state, "CEM input state")?;
        if self.num_samples == 0
            || self.num_elites == 0
            || self.horizon == 0
            || self.num_iterations == 0
            || !self.gamma.is_finite()
            || !(0.0..=1.0).contains(&self.gamma)
        {
            return Err(PlannerError::InvalidInput(
                "CEM counts must be positive and gamma finite in [0, 1]".into(),
            ));
        }
        let action_slice = actions.actions();
        if action_slice.is_empty() {
            return Err(PlannerError::NoFeasibleAction);
        }
        let num_actions = action_slice.len();
        let allowed: Vec<bool> = action_slice
            .iter()
            .map(|&a| {
                gate.evaluate_basic(a, NormalizedEntropy::ZERO)
                    .is_ok_and(|v| v.tier != PolicyTier::Tier3HardStop)
            })
            .collect();
        let allowed_count = allowed.iter().filter(|&&v| v).count();
        if allowed_count == 0 {
            return Err(PlannerError::NoFeasibleAction);
        }
        let initial: Vec<f32> = allowed
            .iter()
            .map(|&v| if v { 1.0 / allowed_count as f32 } else { 0.0 })
            .collect();
        let candidates: Vec<_> = action_slice
            .iter()
            .zip(&allowed)
            .filter_map(|(&a, &ok)| ok.then_some(a))
            .collect();
        let root_actions = world_model.allowed_actions(state, &candidates)?;
        if root_actions.is_empty() {
            budget.mark_dead_end();
            return Err(PlannerError::NoFeasibleAction);
        }
        let mut probs = vec![initial; self.horizon];
        for (p, action) in probs[0].iter_mut().zip(action_slice) {
            *p = if root_actions.contains(action) {
                1.0 / root_actions.len() as f32
            } else {
                0.0
            };
        }
        let mut best: Option<(Vec<usize>, f32)> = None;
        let mut has_dead_end = false;
        for iteration in 0..self.num_iterations {
            budget.check()?;
            let mut trajectories = Vec::new();
            'trajectory: for s_idx in 0..self.num_samples {
                budget.check()?;
                let mut curr_state = state.clone();
                let mut sequence = Vec::new();
                let mut total_reward = 0.0;
                let mut discount = 1.0;
                let mut first_terminal = false;
                for (step, row) in probs.iter().enumerate() {
                    budget.check()?;
                    let u = if step == 0 {
                        (s_idx as f32 + 0.5) / self.num_samples as f32
                    } else {
                        cem_uniform(iteration, s_idx, step)
                    };
                    let legal = world_model.allowed_actions(&curr_state, &candidates)?;
                    if legal.is_empty() {
                        // A state with no legal masked action is a failed
                        // trajectory. Keep its executed prefix for diagnostics,
                        // but make it strictly costly and report the observation.
                        budget.mark_dead_end();
                        has_dead_end = true;
                        total_reward += discount * DEAD_END_PENALTY;
                        validate_score(total_reward, "CEM dead-end penalty")?;
                        break;
                    }
                    // Preserve the original sampler exactly when no candidates
                    // are removed; otherwise condition on the current legal set.
                    let index = if legal == candidates {
                        sample_categorical(row, u)
                    } else {
                        let mut conditioned: Vec<_> = row
                            .iter()
                            .zip(action_slice)
                            .map(|(&p, a)| if legal.contains(a) { p } else { 0.0 })
                            .collect();
                        let mass: f32 = conditioned.iter().sum();
                        if !mass.is_finite() || mass <= 0.0 {
                            return Err(PlannerError::InvalidInput(
                                "CEM mask has no probability mass".into(),
                            ));
                        }
                        for p in &mut conditioned {
                            *p /= mass;
                        }
                        sample_categorical(&conditioned, u)
                    };
                    let action = action_slice[index];
                    if !gate
                        .evaluate_basic(action, NormalizedEntropy::ZERO)
                        .map(|v| v.tier != PolicyTier::Tier3HardStop)
                        .unwrap_or(false)
                    {
                        continue 'trajectory;
                    }
                    // Model and numerical errors abort the entire plan.
                    budget.check()?;
                    let (next, reward, done) = world_model.step(&curr_state, action)?;
                    budget.check()?;
                    validate_transition(&next, reward, "CEM transition")?;
                    total_reward += discount * reward;
                    validate_score(total_reward, "CEM accumulated reward")?;
                    if step == 0 {
                        first_terminal = done;
                    }
                    sequence.push(index);
                    curr_state = next;
                    discount *= self.gamma;
                    if done {
                        break;
                    }
                }
                if sequence.is_empty() {
                    // A stateful mask may close the root between preflight and
                    // sampling. The trajectory has no executable first action,
                    // so it cannot become an incumbent or a returned plan.
                    continue 'trajectory;
                }
                if !first_terminal {
                    budget.candidate(action_slice[sequence[0]], self.name())?;
                }
                trajectories.push((sequence, total_reward));
            }
            if trajectories.is_empty() {
                return Err(PlannerError::NoFeasibleAction);
            }
            trajectories.sort_by(|a, b| b.1.total_cmp(&a.1));
            let elites = &trajectories[..self.num_elites.min(trajectories.len())];
            if best.as_ref().is_none_or(|b| elites[0].1 > b.1) {
                best = Some(elites[0].clone());
            }
            for (step, row) in probs.iter_mut().enumerate() {
                let mut counts = vec![0.0; num_actions];
                let mut observed = 0;
                for (sequence, _) in elites {
                    if let Some(&index) = sequence.get(step) {
                        counts[index] += 1.0;
                        observed += 1;
                    }
                }
                if observed > 0 {
                    for (p, count) in row.iter_mut().zip(counts) {
                        *p = CEM_SMOOTHING * count / observed as f32 + (1.0 - CEM_SMOOTHING) * *p;
                    }
                }
            }
        }
        let (sequence, score) = best.ok_or(PlannerError::NoFeasibleAction)?;
        Ok(CemPlan {
            actions: sequence.into_iter().map(|i| action_slice[i]).collect(),
            score,
            distribution: CemDistribution::Categorical {
                probabilities: probs,
            },
            has_dead_end,
        })
    }
}

impl PlanningEngine for MpcCemEngine {
    fn name(&self) -> &'static str {
        "MpcCemEngine"
    }

    fn search(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        let result = self.optimize_until(state, actions, world_model, gate, budget)?;
        let CemDistribution::Categorical { probabilities } = result.distribution else {
            return Err(PlannerError::InvalidInput(
                "continuous execution is unsupported by ActionId dynamics".into(),
            ));
        };
        Ok((
            result.actions[0],
            NormalizedEntropy::from_probabilities(&probabilities[0]),
        ))
    }
}

/// Weight of the elite frequencies in the smoothed CEM update. Keeping part of
/// the prior stops a single iteration from zeroing out an action for good.
const CEM_SMOOTHING: f32 = 0.7;

/// Inverse-CDF sample from a categorical distribution at quantile `u` in `[0, 1)`.
fn sample_categorical(probs: &[f32], u: f32) -> usize {
    let mut cumulative = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i;
        }
    }
    // Rounding residue must never select a blocked, zero-mass branch.
    probs
        .iter()
        .rposition(|&p| p > 0.0)
        .expect("validated categorical mass")
}

/// Deterministic pseudo-uniform in `[0, 1)` for a horizon step (SplitMix64 hash),
/// so rollouts are reproducible without a `rand` dependency.
fn cem_uniform(iteration: usize, sample: usize, step: usize) -> f32 {
    let mut z = (iteration as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((sample as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((step as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 24 bits give an exactly representable f32 in [0, 1).
    (z >> 40) as f32 / (1u64 << 24) as f32
}

// =========================================================================

// 4. Exact depth-one reward-proportional flow sampling
// =========================================================================

/// Compatibility name: an exact flow sampler on a root-to-candidate star DAG.
/// No learned manifold, multi-step expansion, or trajectory-balance training is
/// claimed. Each candidate is a leaf of this bounded planning graph, regardless
/// of whether the world model reports environment termination.
/// `plan` treats world-model reward as LOG target flow: R(a) = exp(reward(a)).
/// Use `flow_distribution` for explicit nonnegative unnormalized target flows.
#[derive(Clone)]
pub struct ManifoldGFlowNetEngine;

#[derive(Clone, Debug)]
pub struct FlowDistribution {
    pub actions: Vec<ActionId>,
    pub probabilities: Vec<f64>,
}

impl FlowDistribution {
    pub fn sample(&self, rng: &mut impl rand::Rng) -> ActionId {
        self.actions[crate::game::sample(&self.probabilities, rng)]
    }
}

fn unique_actions(actions: &[ActionId]) -> Result<(), PlannerError> {
    let mut seen = std::collections::HashSet::new();
    if actions.iter().any(|a| !seen.insert(a.0)) {
        return Err(PlannerError::InvalidInput(
            "duplicate action IDs distort probability mass".into(),
        ));
    }
    Ok(())
}

impl ManifoldGFlowNetEngine {
    /// Exact flow conservation on the star: F(root) = sum R(leaf),
    /// P_F(leaf | root) = R(leaf)/F(root), P_B(root | leaf) = 1.
    /// Blocked branches have no flow. Zero total mass is an error, not uniform.
    pub fn flow_distribution(
        &self,
        actions: &LocalActionFrame<'_>,
        target_flows: &[f64],
        gate: &PolicyGate,
    ) -> Result<FlowDistribution, PlannerError> {
        if actions.actions().len() != target_flows.len() {
            return Err(PlannerError::DimensionMismatch {
                expected: actions.actions().len(),
                actual: target_flows.len(),
            });
        }
        unique_actions(actions.actions())?;
        if target_flows.iter().any(|f| !f.is_finite() || *f < 0.0) {
            return Err(PlannerError::InvalidInput(
                "target flows must be finite and nonnegative".into(),
            ));
        }
        let mut feasible = Vec::new();
        let mut flows = Vec::new();
        for (&action, &flow) in actions.actions().iter().zip(target_flows) {
            if gate
                .evaluate_basic(action, NormalizedEntropy::ZERO)
                .map(|v| v.tier != PolicyTier::Tier3HardStop)
                .unwrap_or(false)
            {
                feasible.push(action);
                flows.push(flow);
            }
        }
        if feasible.is_empty() {
            return Err(PlannerError::NoFeasibleAction);
        }
        let max = flows.iter().copied().fold(0.0, f64::max);
        if max == 0.0 {
            return Err(PlannerError::InvalidInput("zero total target flow".into()));
        }
        let weights: Vec<f64> = flows.iter().map(|f| f / max).collect();
        let sum: f64 = weights.iter().sum();
        let probabilities: Vec<f64> = weights.iter().map(|w| w / sum).collect();
        if flows
            .iter()
            .zip(&probabilities)
            .any(|(f, p)| *f > 0.0 && *p == 0.0)
        {
            return Err(PlannerError::DivergentState(
                "target flow normalization underflow".into(),
            ));
        }
        Ok(FlowDistribution {
            actions: feasible,
            probabilities,
        })
    }

    pub fn plan_with_rng(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        rng: &mut impl rand::Rng,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        self.plan_with_rng_until(
            state,
            actions,
            world_model,
            gate,
            &SearchBudget::default(),
            rng,
        )
    }

    fn plan_with_rng_until(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
        rng: &mut impl rand::Rng,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        validate_state(state, "GFlowNet input state")?;
        unique_actions(actions.actions())?;
        let mut feasible = Vec::new();
        let mut log_flows = Vec::new();
        for action in world_model.allowed_actions(state, actions.actions())? {
            budget.check()?;
            if gate
                .evaluate_basic(action, NormalizedEntropy::ZERO)
                .map(|v| v.tier != PolicyTier::Tier3HardStop)
                .unwrap_or(false)
            {
                budget.check()?;
                let (next, reward, done) = world_model.step(state, action)?;
                budget.check()?;
                validate_transition(&next, reward, "GFlowNet transition")?;
                if !done {
                    budget.candidate(action, self.name())?;
                }
                feasible.push(action);
                log_flows.push(f64::from(reward));
            }
        }
        if feasible.is_empty() {
            return Err(PlannerError::NoFeasibleAction);
        }
        let max = log_flows.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let flows: Vec<f64> = log_flows.iter().map(|f| (f - max).exp()).collect();
        if flows.contains(&0.0) {
            return Err(PlannerError::DivergentState(
                "log target flow exponentiation underflow".into(),
            ));
        }
        let names = vec!["flow leaf"; feasible.len()];
        let frame = LocalActionFrame::new(&names, &feasible).ok_or_else(|| {
            PlannerError::InvalidInput("flow action frame cannot be represented".into())
        })?;
        let distribution = self.flow_distribution(&frame, &flows, gate)?;
        let probabilities: Vec<f32> = distribution
            .probabilities
            .iter()
            .map(|p| *p as f32)
            .collect();
        Ok((
            distribution.sample(rng),
            NormalizedEntropy::from_probabilities(&probabilities),
        ))
    }
}

impl PlanningEngine for ManifoldGFlowNetEngine {
    fn name(&self) -> &'static str {
        "ManifoldGFlowNetEngine"
    }
    fn search(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        self.plan_with_rng_until(
            state,
            actions,
            world_model,
            gate,
            budget,
            &mut rand::thread_rng(),
        )
    }
}

// =========================================================================
// 5. Two-player normal-form cumulative regret matching
// =========================================================================
pub use crate::game::{GameSolution, NormalFormGame};

/// Compatibility name for regret matching at one information set per player.
/// `solve_game` requires an explicit two-player payoff matrix. Zero-sum average
/// strategies have a measured best-response gap; a finite budget alone is NOT
/// a claim of convergence. General-sum marginals need not be Nash equilibria.
#[derive(Clone)]
pub struct CfrNashEngine;

impl CfrNashEngine {
    pub fn solve_game(
        &self,
        game: &NormalFormGame,
        iterations: usize,
    ) -> Result<GameSolution, PlannerError> {
        crate::game::solve(game, iterations)
    }
}

impl PlanningEngine for CfrNashEngine {
    fn name(&self) -> &'static str {
        "CfrNashEngine"
    }

    /// Legacy single-agent adapter: the world-model trait has no opponent.
    /// Explicitly solve a degenerate one-column zero-sum game for 4096 rounds,
    /// then choose the modal row action (first action breaks ties). This does
    /// not infer opponent behavior. Use `solve_game` and `sample_row` for actual
    /// two-player mixed-strategy play with an injected/seeded random generator.
    fn search(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        validate_state(state, "CFR input state")?;
        unique_actions(actions.actions())?;
        let mut feasible = Vec::new();
        let mut payoffs = Vec::new();
        for action in world_model.allowed_actions(state, actions.actions())? {
            budget.check()?;
            if gate
                .evaluate_basic(action, NormalizedEntropy::ZERO)
                .map(|v| v.tier != PolicyTier::Tier3HardStop)
                .unwrap_or(false)
            {
                budget.check()?;
                let (next, reward, done) = world_model.step(state, action)?;
                budget.check()?;
                validate_transition(&next, reward, "CFR transition")?;
                if !done {
                    budget.candidate(action, self.name())?;
                }
                feasible.push(action);
                payoffs.push(vec![f64::from(reward)]);
            }
        }
        if feasible.is_empty() {
            return Err(PlannerError::NoFeasibleAction);
        }
        let opponent = payoffs.iter().map(|r| vec![-r[0]]).collect();
        let game = NormalFormGame::new(payoffs, opponent)?;
        let solution = crate::game::solve_until(&game, 4096, budget)?;
        let mut best = 0;
        for i in 1..feasible.len() {
            if solution.row_strategy[i] > solution.row_strategy[best] {
                best = i;
            }
        }
        let probabilities: Vec<f32> = solution.row_strategy.iter().map(|p| *p as f32).collect();
        Ok((
            feasible[best],
            NormalizedEntropy::from_probabilities(&probabilities),
        ))
    }
}

// =========================================================================

// 6. Bounded finite-candidate enumeration, NOT a general CP-SAT backend
// =========================================================================

/// Compatibility name: gate-filtered finite-candidate objective enumeration.
/// There is no SAT encoding, branch-and-bound tree, or external CP-SAT solver.
#[derive(Clone)]
pub struct CpSatFormalEngine;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SolveStatus {
    Optimal,
    Feasible,
    Infeasible,
    Timeout,
}

#[derive(Clone, Debug)]
pub struct FormalSolveOptions {
    /// Deterministic budget of world-model evaluations.
    pub max_evaluations: usize,
    /// Cooperative deadline: a synchronous world-model call cannot be preempted.
    pub time_limit: Option<std::time::Duration>,
    pub stop_after_first: bool,
}
impl Default for FormalSolveOptions {
    fn default() -> Self {
        Self {
            max_evaluations: usize::MAX,
            time_limit: None,
            stop_after_first: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FormalSolveResult {
    /// OPTIMAL only certifies the supplied finite action set and one-step reward.
    /// TIMEOUT can contain an incumbent, but never certifies infeasibility/optimality.
    pub status: SolveStatus,
    pub action: Option<ActionId>,
    pub objective: Option<f32>,
    pub evaluated: usize,
}

impl CpSatFormalEngine {
    /// Gate errors are hard stops, preserving a verdict for every candidate.
    pub fn filter_verdicts(
        &self,
        actions: &LocalActionFrame<'_>,
        gate: &PolicyGate,
    ) -> smallvec::SmallVec<[GateVerdict; 16]> {
        actions
            .actions()
            .iter()
            .map(|&action| {
                gate.evaluate_basic(action, NormalizedEntropy::ZERO)
                    .unwrap_or_else(|error| GateVerdict {
                        tier: PolicyTier::Tier3HardStop,
                        action,
                        certified_state: None,
                        violated_rules: Default::default(),
                        reason: format!("gate evaluation error treated as hard stop: {error}"),
                    })
            })
            .collect()
    }
    pub fn filter_feasible(
        &self,
        actions: &LocalActionFrame<'_>,
        gate: &PolicyGate,
    ) -> smallvec::SmallVec<[ActionId; 16]> {
        self.filter_verdicts(actions, gate)
            .into_iter()
            .filter(|v| v.tier != PolicyTier::Tier3HardStop)
            .map(|v| v.action)
            .collect()
    }

    pub fn solve(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        options: &FormalSolveOptions,
    ) -> Result<FormalSolveResult, PlannerError> {
        self.solve_until(
            state,
            actions,
            world_model,
            gate,
            &SearchBudget::default(),
            options,
        )
    }

    fn solve_until(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
        options: &FormalSolveOptions,
    ) -> Result<FormalSolveResult, PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        let started = std::time::Instant::now();
        let expired = || {
            options
                .time_limit
                .is_some_and(|limit| started.elapsed() >= limit)
        };
        validate_state(state, "CP-SAT input state")?;
        let mut result = FormalSolveResult {
            status: SolveStatus::Infeasible,
            action: None,
            objective: None,
            evaluated: 0,
        };
        for action in world_model.allowed_actions(state, actions.actions())? {
            budget.check()?;
            if expired() {
                result.status = SolveStatus::Timeout;
                return Ok(result);
            }
            // An evaluation error is not a proof of infeasibility.
            let verdict = gate
                .evaluate_basic(action, NormalizedEntropy::ZERO)
                .map_err(|e| PlannerError::InvalidInput(format!("gate evaluation failed: {e}")))?;
            if verdict.tier == PolicyTier::Tier3HardStop {
                continue;
            }
            if result.evaluated >= options.max_evaluations || expired() {
                result.status = SolveStatus::Timeout;
                return Ok(result);
            }
            budget.check()?;
            let (next, reward, done) = world_model.step(state, action)?;
            budget.check()?;
            validate_transition(&next, reward, &format!("CP-SAT action {}", action.0))?;
            if !done {
                budget.candidate(action, self.name())?;
            }
            result.evaluated += 1;
            if result.objective.is_none_or(|best| reward > best) {
                result.action = Some(action);
                result.objective = Some(reward);
            }
            if expired() {
                result.status = SolveStatus::Timeout;
                return Ok(result);
            }
            if options.stop_after_first {
                result.status = SolveStatus::Feasible;
                return Ok(result);
            }
        }
        result.status = if expired() {
            SolveStatus::Timeout
        } else if result.action.is_some() {
            SolveStatus::Optimal
        } else {
            SolveStatus::Infeasible
        };
        Ok(result)
    }
}

impl PlanningEngine for CpSatFormalEngine {
    fn name(&self) -> &'static str {
        "CpSatFormalEngine"
    }
    fn search(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        budget.check()?;
        validate_actions(actions.actions())?;
        let result = self.solve_until(
            state,
            actions,
            world_model,
            gate,
            budget,
            &FormalSolveOptions::default(),
        )?;
        match result.status {
            SolveStatus::Optimal | SolveStatus::Feasible => result
                .action
                .map(|a| (a, NormalizedEntropy::ZERO))
                .ok_or(PlannerError::NoFeasibleAction),
            SolveStatus::Infeasible => Err(PlannerError::NoFeasibleAction),
            SolveStatus::Timeout => Err(PlannerError::TimeoutExceeded(0.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcts_backup_updates_every_ancestor_with_its_own_discounted_return() {
        let mut nodes = vec![
            MctsNode::new(None, 0.0, false),
            MctsNode::new(Some(ActionId(0)), 1.0, false),
            MctsNode::new(Some(ActionId(1)), 10.0, false),
            MctsNode::new(Some(ActionId(2)), -7.0, false),
        ];
        // Leaf tail value 4 -> leaf 12 -> parent 7 -> root 7.
        mcts_backup(&mut nodes, &[0, 1, 2], 4.0, 0.5).unwrap();
        assert_eq!(
            nodes.iter().map(|n| n.visits).collect::<Vec<_>>(),
            [1, 1, 1, 0]
        );
        assert_eq!(
            nodes.iter().map(|n| n.value_sum).collect::<Vec<_>>(),
            [7.0, 7.0, 12.0, 0.0]
        );
        mcts_backup(&mut nodes, &[0, 1, 2], 0.0, 0.5).unwrap();
        assert_eq!(
            nodes.iter().map(|n| n.visits).collect::<Vec<_>>(),
            [2, 2, 2, 0]
        );
        assert_eq!(
            nodes.iter().map(|n| n.value_sum).collect::<Vec<_>>(),
            [13.0, 13.0, 22.0, 0.0]
        );
    }

    struct DummyDynamics;
    impl WorldModelDynamics for DummyDynamics {
        type Error = CoreError;
        fn step(
            &self,
            state: &FullLatent,
            action: ActionId,
        ) -> Result<(FullLatent, f32, bool), Self::Error> {
            let mut next = state.clone();
            next.as_mut_slice()[0] += action.0 as f32;
            let reward = (action.0 as f32) * 1.5;
            Ok((next, reward, false))
        }
        fn step_batch(
            &self,
            states: &[FullLatent],
            actions: &[ActionId],
            next_states: &mut [FullLatent],
            rewards: &mut [f32],
            dones: &mut [bool],
        ) -> Result<(), Self::Error> {
            for i in 0..states.len() {
                let (ns, r, d) = self.step(&states[i], actions[i])?;
                next_states[i] = ns;
                rewards[i] = r;
                dones[i] = d;
            }
            Ok(())
        }
    }

    struct PayoffDynamics {
        payoffs: [f32; 3],
        calls: std::sync::Mutex<Vec<ActionId>>,
    }

    impl WorldModelDynamics for PayoffDynamics {
        type Error = CoreError;

        fn step(
            &self,
            state: &FullLatent,
            action: ActionId,
        ) -> Result<(FullLatent, f32, bool), Self::Error> {
            self.calls.lock().unwrap().push(action);
            Ok((state.clone(), self.payoffs[action.0 as usize], false))
        }

        fn step_batch(
            &self,
            _states: &[FullLatent],
            _actions: &[ActionId],
            _next_states: &mut [FullLatent],
            _rewards: &mut [f32],
            _dones: &mut [bool],
        ) -> Result<(), Self::Error> {
            unreachable!("these rankers use single transitions")
        }
    }

    fn payoff_model(payoffs: [f32; 3]) -> PayoffDynamics {
        PayoffDynamics {
            payoffs,
            calls: Default::default(),
        }
    }

    fn block_action(action: ActionId) -> PolicyGate {
        let mut gate = PolicyGate::default();
        gate.add_constraint(gen_zero_gate::LinearConstraint::prohibit(
            gen_zero_gate::RuleId(1),
            "blocked",
            action,
        ));
        gate
    }

    #[test]
    fn gflow_entropy_matches_softmax_and_handles_ties_and_singletons() {
        let state = FullLatent::zeros();
        let gate = PolicyGate::default();
        let actions = [ActionId(0), ActionId(1)];
        let frame = LocalActionFrame::new(&["a", "b"], &actions).unwrap();
        let (_, entropy) = ManifoldGFlowNetEngine
            .plan(
                &state,
                &frame,
                &payoff_model([0.0, 3.0_f32.ln(), 0.0]),
                &gate,
            )
            .unwrap();
        let expected = -(0.25_f32 * 0.25_f32.ln() + 0.75_f32 * 0.75_f32.ln()) / 2.0_f32.ln();
        assert!((entropy.value() - expected).abs() < 1e-6);
        let (_, tied) = ManifoldGFlowNetEngine
            .plan(&state, &frame, &payoff_model([1000.0; 3]), &gate)
            .unwrap();
        assert!((tied.value() - 1.0).abs() < 1e-6);
        let (action, single) = ManifoldGFlowNetEngine
            .plan(
                &state,
                &frame,
                &payoff_model([0.0; 3]),
                &block_action(ActionId(1)),
            )
            .unwrap();
        assert_eq!(action, ActionId(0));
        assert_eq!(single.value(), 0.0);
    }

    #[test]
    fn cfr_average_strategy_concentrates_on_best_payoff_and_preserves_ties() {
        let state = FullLatent::zeros();
        let gate = PolicyGate::default();
        let actions = [ActionId(0), ActionId(1), ActionId(2)];
        let frame = LocalActionFrame::new(&["a", "b", "c"], &actions).unwrap();
        // Multi-round average play concentrates on the dominant action.
        let (action, entropy) = CfrNashEngine
            .plan(&state, &frame, &payoff_model([-3.0, 1.0, 2.0]), &gate)
            .unwrap();
        assert_eq!(action, ActionId(2));
        assert!(entropy.value() > 0.0 && entropy.value() < 0.02);
        let (action, entropy) = CfrNashEngine
            .plan(&state, &frame, &payoff_model([7.0; 3]), &gate)
            .unwrap();
        assert_eq!(action, ActionId(0));
        assert!((entropy.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blocked_candidates_do_not_change_cfr_baseline_or_entropy() {
        let state = FullLatent::zeros();
        let actions = [ActionId(0), ActionId(1), ActionId(2)];
        let full = LocalActionFrame::new(&["a", "b", "blocked"], &actions).unwrap();
        let legal = LocalActionFrame::new(&["a", "b"], &actions[..2]).unwrap();
        let model = payoff_model([1.0, 2.0, 1000.0]);
        let expected = CfrNashEngine
            .plan(&state, &legal, &model, &PolicyGate::default())
            .unwrap();
        let actual = CfrNashEngine
            .plan(&state, &full, &model, &block_action(ActionId(2)))
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.0, ActionId(1));
        assert!(actual.1.value() < 0.01);
        assert!(!model.calls.lock().unwrap().contains(&ActionId(2)));
        let (action, entropy) = CfrNashEngine
            .plan(&state, &legal, &model, &block_action(ActionId(1)))
            .unwrap();
        assert_eq!((action, entropy), (ActionId(0), NormalizedEntropy::ZERO));
    }

    #[test]
    fn cem_excludes_blocked_actions_from_every_rollout_and_entropy() {
        let actions = [ActionId(0), ActionId(1)];
        let frame = LocalActionFrame::new(&["allowed", "blocked"], &actions).unwrap();
        let engine = MpcCemEngine {
            num_samples: 2,
            num_elites: 1,
            horizon: 8,
            num_iterations: 2,
            ..MpcCemEngine::default()
        };
        let model = payoff_model([1.0, 1000.0, 0.0]);
        let result = engine
            .plan(
                &FullLatent::zeros(),
                &frame,
                &model,
                &block_action(ActionId(1)),
            )
            .unwrap();
        assert_eq!(result, (ActionId(0), NormalizedEntropy::ZERO));
        let calls = model.calls.lock().unwrap();
        assert_eq!(calls.len(), 32);
        assert!(calls.iter().all(|&action| action == ActionId(0)));
    }

    #[test]
    fn mcts_visits_unexplored_actions_even_with_large_rewards() {
        let actions = [ActionId(0), ActionId(1), ActionId(2)];
        let frame = LocalActionFrame::new(&["a", "b", "c"], &actions).unwrap();
        let model = payoff_model([2e6, 3e6, 4e6]);
        let (_, entropy) = MctsEngine {
            max_simulations: 3,
            horizon: 1,
            ..MctsEngine::default()
        }
        .plan(&FullLatent::zeros(), &frame, &model, &PolicyGate::default())
        .unwrap();
        assert_eq!(*model.calls.lock().unwrap(), actions);
        assert!((entropy.value() - 1.0).abs() < 1e-6);
    }

    struct SequenceDynamics {
        discount_case: bool,
        calls: std::sync::Mutex<Vec<(usize, ActionId)>>,
    }

    impl WorldModelDynamics for SequenceDynamics {
        type Error = CoreError;

        fn step(
            &self,
            state: &FullLatent,
            action: ActionId,
        ) -> Result<(FullLatent, f32, bool), CoreError> {
            let depth = state.as_slice()[0] as usize;
            self.calls.lock().unwrap().push((depth, action));
            let mut next = state.clone();
            next.as_mut_slice()[0] += 1.0;
            if self.discount_case {
                // Immediate 8.5 beats 10 received two steps later: 0.9^2 * 10 = 8.1.
                assert!(depth < 3, "must not step past terminal");
                if depth == 0 && action == ActionId(1) {
                    next.as_mut_slice()[0] = 3.0;
                    return Ok((next, 8.5, true));
                }
                Ok((next, if depth == 2 { 10.0 } else { 0.0 }, depth == 2))
            } else {
                assert!(depth < 2, "must not step past terminal");
                if depth == 0 {
                    next.as_mut_slice()[1] = action.0 as f32;
                    Ok((next, 0.0, false))
                } else {
                    let reward = if state.as_slice()[1] == 0.0 && action == ActionId(1) {
                        10.0
                    } else {
                        -10.0
                    };
                    Ok((next, reward, true))
                }
            }
        }

        fn step_batch(
            &self,
            _: &[FullLatent],
            _: &[ActionId],
            _: &mut [FullLatent],
            _: &mut [f32],
            _: &mut [bool],
        ) -> Result<(), CoreError> {
            unreachable!()
        }
    }

    #[test]
    fn cem_learns_different_actions_at_different_positions() {
        let actions = [ActionId(0), ActionId(1)];
        let frame = LocalActionFrame::new(&["a", "b"], &actions).unwrap();
        let model = SequenceDynamics {
            discount_case: false,
            calls: Default::default(),
        };
        let engine = MpcCemEngine {
            num_samples: 64,
            num_elites: 8,
            horizon: 8,
            num_iterations: 4,
            gamma: 0.9,
        };
        let (action, _) = engine
            .plan(&FullLatent::zeros(), &frame, &model, &PolicyGate::default())
            .unwrap();
        assert_eq!(action, ActionId(0));
        let calls = model.calls.lock().unwrap();
        assert_eq!(calls.len(), 4 * 64 * 2); // Terminal after two, despite horizon eight.
        let final_iteration = &calls[3 * 64 * 2..];
        assert!(
            final_iteration
                .iter()
                .filter(|&&(depth, act)| depth == 0 && act == ActionId(0))
                .count()
                > 55
        );
        assert!(
            final_iteration
                .iter()
                .filter(|&&(depth, act)| depth == 1 && act == ActionId(1))
                .count()
                > 55
        );
    }

    #[test]
    fn cem_discounts_by_depth_and_stops_at_terminal() {
        let actions = [ActionId(0), ActionId(1)];
        let frame = LocalActionFrame::new(&["a", "b"], &actions).unwrap();
        let model = SequenceDynamics {
            discount_case: true,
            calls: Default::default(),
        };
        let engine = MpcCemEngine {
            num_samples: 16,
            num_elites: 4,
            horizon: 8,
            num_iterations: 2,
            gamma: 0.9,
        };
        let (action, _) = engine
            .plan(&FullLatent::zeros(), &frame, &model, &PolicyGate::default())
            .unwrap();
        assert_eq!(action, ActionId(1));
    }

    #[test]
    fn test_all_six_engines() {
        let wm = DummyDynamics;
        let gate = PolicyGate::default();
        let state = FullLatent::default();
        let candidate_names = ["act1", "act2", "act3"];
        let raw_acts = [ActionId(1), ActionId(2), ActionId(3)];
        let actions = LocalActionFrame::new(&candidate_names, &raw_acts).unwrap();

        let mcts = MctsEngine::default();
        let (a1, _) = mcts.plan(&state, &actions, &wm, &gate).unwrap();
        assert_eq!(a1, ActionId(3));

        let astar = AStarEngine {
            goal: Some(AStarGoal::Predicate(|s| s.as_slice()[0] >= 3.0)),
            uncertainty_penalty_weight: 0.0,
            ..Default::default()
        };
        let (a2, _) = astar.plan(&state, &actions, &wm, &gate).unwrap();
        assert_eq!(a2, ActionId(3));

        let mpc = MpcCemEngine::default();
        let (a3, _) = mpc.plan(&state, &actions, &wm, &gate).unwrap();
        assert_eq!(a3, ActionId(3));

        let gflow = ManifoldGFlowNetEngine;
        let (a4, _) = gflow.plan(&state, &actions, &wm, &gate).unwrap();
        assert!(raw_acts.contains(&a4));

        let cfr = CfrNashEngine;
        let (a5, _) = cfr.plan(&state, &actions, &wm, &gate).unwrap();
        assert_eq!(a5, ActionId(3));

        let cpsat = CpSatFormalEngine;
        let (a6, _) = cpsat.plan(&state, &actions, &wm, &gate).unwrap();
        assert_eq!(a6, ActionId(3));
    }

    #[test]
    fn sample_categorical_follows_cumulative_mass() {
        let probs = [0.25, 0.5, 0.25];
        assert_eq!(sample_categorical(&probs, 0.0), 0);
        assert_eq!(sample_categorical(&probs, 0.3), 1);
        assert_eq!(sample_categorical(&probs, 0.8), 2);
        assert_eq!(sample_categorical(&[0.0, 1.0], 0.0), 1);
        // Mass summing just below 1.0 still maps high quantiles to the last action.
        assert_eq!(sample_categorical(&[0.5, 0.4999], 0.99995), 1);
        assert_eq!(sample_categorical(&[0.5, 0.4999, 0.0], 0.99995), 1);
    }

    #[test]
    fn cem_uniform_is_deterministic_and_in_unit_interval() {
        for it in 0..3 {
            for s in 0..32 {
                for h in 1..4 {
                    let u = cem_uniform(it, s, h);
                    assert!((0.0..1.0).contains(&u));
                    assert_eq!(u, cem_uniform(it, s, h));
                }
            }
        }
        assert_ne!(cem_uniform(0, 1, 1), cem_uniform(0, 1, 2));
    }
}
