//! Production pipeline: `simulate`, `what_if`, `audit_action` and multi-mode `decide`
//! over one world model and one PolicyGate.
//!
//! One rollout loop serves every method. Terminal semantics:
//!
//! - `done`: the transition ended the episode. The pipeline knows nothing else about
//!   a terminal, so every `done` counts as a hazard (fail-closed). For
//!   `LatentDynamicsWorldModel` this is also exact: its only terminal is divergence
//!   past `DONE_NORM`.
//! - `survival_horizon`: steps completed before the first hazard.
//! - `terminated_early`: the rollout hit `done` (death) at or before its horizon,
//!   same as the Python `rollout`.
//!
//! Safety readings come from [`WorldModelDynamics::safety_estimate`]. A model without
//! one yields `safe_prob: None`; `audit_action` then refuses with
//! `MissingSafetyEstimate` instead of scoring risk as zero.
//!
//! Gate contract: a gate error is treated as `Tier3HardStop`. Hard-stopped actions are
//! pruned before any engine runs and reported with their rule ids. `decide` also
//! excludes candidates whose first predicted transition is terminal; this is an
//! immediate-hazard screen, not a certificate of safety for future trajectories.

use crate::config::{resolve_deadline, PlannerConfig};
use crate::engine::SearchBudget;
use crate::engine::{
    AStarEngine, CfrNashEngine, CpSatFormalEngine, ManifoldGFlowNetEngine, MctsEngine,
    MpcCemEngine, PlanningEngine,
};
use crate::error::PlannerError;
use crate::router::{DynamicKMoERouter, RoutingTier};
use arc_swap::ArcSwapOption;
use gen_zero_core::{
    ActionId, CoreError, FullLatent, LocalActionFrame, NormalizedEntropy, WorldModelDynamics,
};
use gen_zero_gate::{PolicyGate, PolicyTier};
use gen_zero_lod::{EpistemicStatus, LodGraph};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::Instant;

/// Search cannot discard a terminal hazard it observes after candidate screening.
/// This adapter is confined to decide; simulate keeps counterfactual diagnostics.
struct HazardCheckedDynamics<'a>(&'a dyn WorldModelDynamics<Error = CoreError>);

impl WorldModelDynamics for HazardCheckedDynamics<'_> {
    type Error = CoreError;

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), CoreError> {
        let transition = self.0.step(state, action)?;
        if transition.2 {
            return Err(CoreError::WorldModel(
                "terminal hazard during decision search".into(),
            ));
        }
        Ok(transition)
    }

    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next_states: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), CoreError> {
        if [actions.len(), next_states.len(), rewards.len(), dones.len()]
            .iter()
            .any(|&len| len != states.len())
        {
            return Err(CoreError::WorldModel(
                "decision batch length mismatch".into(),
            ));
        }
        for i in 0..states.len() {
            let (next, reward, done) = self.step(&states[i], actions[i])?;
            next_states[i] = next;
            rewards[i] = reward;
            dones[i] = done;
        }
        Ok(())
    }
}

/// Longest rollout any method accepts. Each step stores a 4 KiB latent.
pub const MAX_HORIZON: usize = 128;
/// Most candidates `what_if` expands; each one is a full rollout with a greedy
/// continuation, so cost grows as candidates² × horizon.
pub const MAX_WHAT_IF_CANDIDATES: usize = 64;
/// `LocalActionFrame` holds at most 16 actions.
pub const MAX_DECIDE_CANDIDATES: usize = 16;
/// Default `risk_score` at or above which a hazard-free audit is only a warning.
pub const DEFAULT_WARN_RISK: f32 = 0.3;

pub const POLICY_FIXED_PLAN: &str = "fixed_plan";
pub const POLICY_GREEDY: &str = "greedy_one_step_over_candidates";
pub const POLICY_REPEAT: &str = "repeat_audited_action";

/// One imagined transition.
#[derive(Clone, Debug)]
pub struct SimStep {
    /// 1-based.
    pub step_idx: usize,
    pub action: ActionId,
    /// State after the transition.
    pub state: FullLatent,
    pub reward: f32,
    pub safe_prob: Option<f32>,
    pub done: bool,
    pub hazard: bool,
    /// Gate tier of `action` (hard constraints only, entropy 0).
    pub gate_tier: PolicyTier,
}

/// A finished rollout.
#[derive(Clone, Debug)]
pub struct Rollout {
    pub steps: Vec<SimStep>,
    pub survival_horizon: usize,
    pub cumulative_return: f32,
    pub terminated_early: bool,
    pub termination_step: Option<usize>,
    pub first_hazard_step: Option<usize>,
    pub min_safe_prob: Option<f32>,
    /// Steps that carried a safety estimate.
    pub safety_coverage: usize,
    /// `Some(true)` only if every estimate was calibrated; `None` without estimates.
    pub safety_calibrated: Option<bool>,
    pub safety_sources: Vec<&'static str>,
    pub continuation_policy: &'static str,
    pub final_state: FullLatent,
}

impl Rollout {
    pub fn steps_simulated(&self) -> usize {
        self.steps.len()
    }

    /// Whether every imagined action passed policy (independent of model hazards).
    pub fn policy_allowed(&self) -> bool {
        self.steps
            .iter()
            .all(|step| step.gate_tier != PolicyTier::Tier3HardStop)
    }

    /// Whether the model predicted no terminal hazard; not a policy approval.
    pub fn hazard_free(&self) -> bool {
        self.first_hazard_step.is_none()
    }

    pub fn is_safe(&self) -> bool {
        self.policy_allowed() && self.hazard_free()
    }
}

/// An action removed by policy or a predicted terminal hazard.
#[derive(Clone, Debug, PartialEq)]
pub struct PrunedAction {
    pub action: ActionId,
    pub tier: PolicyTier,
    pub violated_rules: Vec<u32>,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct CandidateOutcome {
    pub action: ActionId,
    pub rollout: Rollout,
}

#[derive(Clone, Debug)]
pub struct WhatIfReport {
    /// Gate-allowed candidates, in request order.
    pub outcomes: Vec<CandidateOutcome>,
    pub best_candidate: ActionId,
    /// Safe before trapped, then longer survival, then higher return.
    pub safety_ranking: Vec<ActionId>,
    pub traps_detected: Vec<ActionId>,
    pub all_candidates_trapped: bool,
    pub gate_blocked: Vec<PrunedAction>,
    pub horizon: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AuditVerdict {
    Approved,
    WarnHazard,
    RejectLethal,
}

impl AuditVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Approved => "Approved",
            Self::WarnHazard => "WarnHazard",
            Self::RejectLethal => "RejectLethal",
        }
    }
}

#[derive(Clone, Debug)]
pub struct AuditReport {
    pub action: ActionId,
    pub verdict: AuditVerdict,
    /// `1 - min safe_prob` over the rollout; 1.0 for a gate hard stop.
    pub risk_score: f32,
    pub reasons: Vec<String>,
    pub gate_tier: PolicyTier,
    pub is_safe: bool,
    pub survival_horizon: usize,
    pub first_hazard_step: Option<usize>,
    /// Continuation actions the gate removed.
    pub continuation_pruned: Vec<PrunedAction>,
    /// `None` when the gate hard-stopped the action: it is never imagined.
    pub trajectory: Option<Rollout>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DecideMode {
    /// Dynamic K-MoE router: reflex, pipeline or committee by entropy and gate tier.
    Auto,
    /// Finite-horizon multi-step PUCT search.
    Mcts,
    /// Multi-step categorical cross-entropy trajectory optimization.
    MpcCem,
    /// Multi-step A* graph search with an explicit goal.
    AStar,
    /// Exact one-step flow sampling with exp(world-model reward) target mass.
    /// This does not train a manifold or multi-step trajectory-balance model.
    ManifoldGFlowNet,
    /// Single-agent, one-column regret-matching adapter. No opponent is inferred.
    /// Explicit two-player games use CfrNashEngine::solve_game instead.
    CfrNash,
    /// Gated one-step best reward (`CpSatFormalEngine`). Never the ungated first action.
    Reflex,
}

impl DecideMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Mcts => "mcts",
            Self::MpcCem => "mpc_cem",
            Self::AStar => "astar",
            Self::ManifoldGFlowNet => "manifold_gflownet",
            Self::CfrNash => "cfr_nash",
            Self::Reflex => "reflex",
        }
    }
}

impl FromStr for DecideMode {
    type Err = PlannerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "mcts" => Ok(Self::Mcts),
            "mpc_cem" => Ok(Self::MpcCem),
            "astar" => Ok(Self::AStar),
            "manifold_gflownet" | "gflownet" => Ok(Self::ManifoldGFlowNet),
            "cfr_nash" | "cfr" => Ok(Self::CfrNash),
            "reflex" => Ok(Self::Reflex),
            other => Err(PlannerError::UnknownMode(other.to_string())),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecideRequest<'a> {
    /// Earliest of request/config absolute and relative limits is enforced.
    pub deadline: Option<Instant>,
    pub budget_ms: Option<f64>,
    pub state: &'a FullLatent,
    pub candidates: &'a [ActionId],
    /// Actions currently executing; multiplicity is significant for quotas.
    pub active_context: Vec<ActionId>,
    pub mode: DecideMode,
    /// Perceptual entropy in `[0, 1]`: drives auto routing and the reported gate tier.
    pub entropy: NormalizedEntropy,
    pub return_trajectory: bool,
    /// Rollout length when `return_trajectory` is set.
    pub horizon: usize,
}

/// PPR restart probability, iteration cap, L1 tolerance and fact count used for
/// [`Decision::graph_context`].
const GRAPH_CONTEXT_ALPHA: f32 = 0.15;
const GRAPH_CONTEXT_MAX_ITERS: usize = 100;
const GRAPH_CONTEXT_TOLERANCE: f32 = 1e-6;
const GRAPH_CONTEXT_TOP: usize = 8;

/// One live graph node near the chosen action.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphFact {
    pub entity_id: u64,
    pub label: String,
    pub status: EpistemicStatus,
    /// Personalized PageRank mass from the seeds.
    pub score: f32,
}

/// Topological facts around the chosen action, from Personalized PageRank over
/// the live graph. Advisory only: it is computed after the choice and no engine,
/// gate check or tier reads it.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphContext {
    /// Diffusion seeded at the entity nodes of the chosen action and of the
    /// active-context actions that have one. `facts` holds the highest-scoring
    /// non-seed live nodes with nonzero mass.
    Diffused {
        seed_entities: Vec<u64>,
        facts: Vec<GraphFact>,
        iterations: usize,
        converged: bool,
    },
    /// No diffusion ran; `reason` says why.
    Unavailable { reason: String },
}

#[derive(Clone, Debug)]
pub struct Decision {
    /// Search was cut short; entropy is conservatively unknown (ONE), and no
    /// optional trajectory is claimed complete. Gate certification is not a
    /// calibrated guarantee of real-world safety.
    pub timed_out: bool,
    pub action: ActionId,
    /// Entropy of the engine's own decision distribution.
    pub entropy: NormalizedEntropy,
    pub mode: DecideMode,
    pub engine: &'static str,
    /// Set for a completed auto decision only; an anytime member is not consensus.
    pub routing_tier: Option<RoutingTier>,
    /// Gate tier of the chosen action at the request entropy.
    pub gate_tier: PolicyTier,
    pub requires_confirmation: bool,
    /// Whether any candidate predicted an immediate terminal hazard.
    pub hazard_detected: bool,
    /// Candidates excluded because their first transition returned `done=true`.
    pub hazardous_actions: Vec<ActionId>,
    pub feasible: Vec<ActionId>,
    pub pruned: Vec<PrunedAction>,
    pub trajectory: Option<Rollout>,
    /// Advisory graph neighborhood of `action`; see [`GraphContext`].
    pub graph_context: GraphContext,
}

/// Unified simulate / what-if / audit / decide entry for Rust production callers.
#[derive(Clone)]
pub struct ProductionPipeline {
    config_deadline: Option<Instant>,
    config_budget_ms: Option<f64>,
    in_flight: Arc<AtomicBool>,
    deadline_worker: Option<mpsc::SyncSender<DeadlineJob>>,
    world_model: Arc<dyn WorldModelDynamics<Error = CoreError>>,
    gate: Arc<PolicyGate>,
    graph: Arc<LodGraph>,
    router: DynamicKMoERouter,
    mcts: MctsEngine,
    mpc: MpcCemEngine,
    astar: AStarEngine,
    manifold_gflownet: ManifoldGFlowNetEngine,
    cfr_nash: CfrNashEngine,
    cpsat: CpSatFormalEngine,
}

impl ProductionPipeline {
    /// Configure explicit A* success for direct and routed searches.
    pub fn with_astar_goal(mut self, goal: crate::AStarGoal) -> Self {
        self.astar.goal = Some(goal.clone());
        self.router = self.router.with_astar_goal(goal);
        self
    }

    pub fn new(
        world_model: Arc<dyn WorldModelDynamics<Error = CoreError>>,
        gate: Arc<PolicyGate>,
    ) -> Self {
        Self {
            config_deadline: None,
            config_budget_ms: None,
            in_flight: Arc::new(AtomicBool::new(false)),
            deadline_worker: start_deadline_worker(),
            world_model,
            gate,
            graph: Arc::new(LodGraph::new()),
            router: DynamicKMoERouter::default(),
            mcts: MctsEngine::default(),
            mpc: MpcCemEngine::default(),
            astar: AStarEngine::default(),
            manifold_gflownet: ManifoldGFlowNetEngine,
            cfr_nash: CfrNashEngine,
            cpsat: CpSatFormalEngine,
        }
    }

    /// Share the live cognitive graph used for action revocation checks.
    pub fn with_graph(mut self, graph: Arc<LodGraph>) -> Self {
        self.graph = graph;
        self
    }

    /// Like [`Self::new`], but every engine's tunable hyperparameters come from
    /// `config` instead of each engine's own `::default()`. Both the router's
    /// internal engines (used by `decide(Auto)`) and the pipeline's own engine
    /// instances (used by `decide(Mcts | MpcCem | AStar)`) are built from the
    /// same config, so the effective knob is the same regardless of mode.
    /// Refuses a config that would degrade silently (see
    /// [`PlannerConfig::validate`]) instead of building a pipeline that quietly
    /// never searches.
    pub fn new_with_config(
        world_model: Arc<dyn WorldModelDynamics<Error = CoreError>>,
        gate: Arc<PolicyGate>,
        config: PlannerConfig,
    ) -> Result<Self, PlannerError> {
        config.validate()?;
        Ok(Self {
            config_deadline: config.deadline,
            config_budget_ms: config.budget_ms,
            in_flight: Arc::new(AtomicBool::new(false)),
            deadline_worker: start_deadline_worker(),
            world_model,
            gate,
            graph: Arc::new(LodGraph::new()),
            router: DynamicKMoERouter::from_config(&config),
            mcts: MctsEngine {
                max_simulations: config.mcts_max_simulations,
                c_puct: config.mcts_c_puct,
                horizon: config.mcts_horizon,
                discount: config.mcts_discount,
                ..MctsEngine::default()
            },
            mpc: MpcCemEngine {
                num_samples: config.cem_num_samples,
                horizon: config.cem_horizon,
                gamma: config.cem_gamma,
                ..MpcCemEngine::default()
            },
            astar: AStarEngine {
                uncertainty_penalty_weight: config.astar_uncertainty_penalty_weight,
                ..AStarEngine::default()
            },
            manifold_gflownet: ManifoldGFlowNetEngine,
            cfr_nash: CfrNashEngine,
            cpsat: CpSatFormalEngine,
        })
    }

    /// Roll a fixed action sequence forward. `horizon` defaults to `actions.len()`;
    /// a longer horizon is refused rather than padded with invented actions.
    /// A hard-stopped action aborts before its world-model transition.
    pub fn simulate(
        &self,
        state: &FullLatent,
        actions: &[ActionId],
        horizon: Option<usize>,
    ) -> Result<Rollout, PlannerError> {
        validate_state(state)?;
        if actions.is_empty() {
            return Err(PlannerError::InvalidInput(
                "simulate needs a non-empty action sequence".into(),
            ));
        }
        let horizon = horizon.unwrap_or(actions.len());
        validate_horizon(horizon)?;
        if horizon > actions.len() {
            return Err(PlannerError::InvalidInput(format!(
                "horizon {horizon} exceeds the {} given actions",
                actions.len()
            )));
        }
        self.rollout(
            state,
            horizon,
            POLICY_FIXED_PLAN,
            &SearchBudget::default(),
            |step_idx, _| Ok(actions[step_idx - 1]),
        )
    }

    /// Each gate-allowed candidate as the first move, then a greedy one-step
    /// continuation over the allowed candidates.
    pub fn what_if(
        &self,
        state: &FullLatent,
        candidates: &[ActionId],
        horizon: usize,
    ) -> Result<WhatIfReport, PlannerError> {
        validate_state(state)?;
        validate_horizon(horizon)?;
        validate_candidates(candidates, MAX_WHAT_IF_CANDIDATES)?;
        let (allowed, gate_blocked) = self.prune(candidates);
        if allowed.is_empty() {
            return Err(PlannerError::NoFeasibleAction);
        }

        let mut outcomes = Vec::with_capacity(allowed.len());
        for &first in &allowed {
            let rollout = self.greedy_rollout(
                state,
                first,
                &allowed,
                horizon,
                POLICY_GREEDY,
                &SearchBudget::default(),
            )?;
            outcomes.push(CandidateOutcome {
                action: first,
                rollout,
            });
        }

        let mut order: Vec<usize> = (0..outcomes.len()).collect();
        // Stable sort keeps request order on ties.
        order.sort_by(|&a, &b| {
            let (ra, rb) = (&outcomes[a].rollout, &outcomes[b].rollout);
            rb.is_safe()
                .cmp(&ra.is_safe())
                .then(rb.survival_horizon.cmp(&ra.survival_horizon))
                .then(rb.cumulative_return.total_cmp(&ra.cumulative_return))
        });
        let safety_ranking: Vec<ActionId> = order.iter().map(|&i| outcomes[i].action).collect();
        let traps_detected: Vec<ActionId> = outcomes
            .iter()
            .filter(|o| !o.rollout.is_safe())
            .map(|o| o.action)
            .collect();

        Ok(WhatIfReport {
            best_candidate: safety_ranking[0],
            all_candidates_trapped: traps_detected.len() == outcomes.len(),
            safety_ranking,
            traps_detected,
            outcomes,
            gate_blocked,
            horizon,
        })
    }

    /// Shadow risk review of `action`. The gate is checked first: a hard stop is
    /// `RejectLethal` without imagining the action. Otherwise the action is rolled
    /// forward; later steps repeat it, or follow a greedy policy over the
    /// gate-allowed `continuation` actions.
    ///
    /// Verdict: any hazard → `RejectLethal`; else `risk_score >= warn_risk` or a
    /// confirm-tier action → `WarnHazard`; else `Approved`.
    pub fn audit_action(
        &self,
        state: &FullLatent,
        action: ActionId,
        horizon: usize,
        continuation: Option<&[ActionId]>,
        warn_risk: f32,
    ) -> Result<AuditReport, PlannerError> {
        validate_state(state)?;
        validate_horizon(horizon)?;
        if !(warn_risk > 0.0 && warn_risk <= 1.0) {
            return Err(PlannerError::InvalidInput(format!(
                "warn_risk must lie in (0, 1], got {warn_risk}"
            )));
        }

        let (gate_tier, rules, gate_reason) = self.gate_check(action);
        if gate_tier == PolicyTier::Tier3HardStop {
            return Ok(AuditReport {
                action,
                verdict: AuditVerdict::RejectLethal,
                risk_score: 1.0,
                reasons: vec![format!(
                    "PolicyGate hard stop (rules {rules:?}): {gate_reason}"
                )],
                gate_tier,
                is_safe: false,
                survival_horizon: 0,
                first_hazard_step: None,
                continuation_pruned: Vec::new(),
                trajectory: None,
            });
        }

        let (continuation_set, continuation_pruned, policy) = match continuation {
            None => (vec![action], Vec::new(), POLICY_REPEAT),
            Some(list) => {
                validate_candidates(list, MAX_WHAT_IF_CANDIDATES)?;
                let (allowed, pruned) = self.prune(list);
                if allowed.is_empty() {
                    return Err(PlannerError::NoFeasibleAction);
                }
                (allowed, pruned, POLICY_GREEDY)
            }
        };

        let roll = self.greedy_rollout(
            state,
            action,
            &continuation_set,
            horizon,
            policy,
            &SearchBudget::default(),
        )?;
        if roll.safety_coverage != roll.steps_simulated() {
            return Err(PlannerError::MissingSafetyEstimate {
                covered: roll.safety_coverage,
                steps: roll.steps_simulated(),
            });
        }
        let min_safe = roll
            .min_safe_prob
            .ok_or(PlannerError::MissingSafetyEstimate {
                covered: 0,
                steps: roll.steps_simulated(),
            })?;
        let risk_score = (1.0 - min_safe).clamp(0.0, 1.0);

        let mut reasons = Vec::new();
        let verdict = if let Some(step) = roll.first_hazard_step {
            let who = if step == 1 {
                "the audited action itself".to_string()
            } else {
                format!("the {policy} continuation")
            };
            reasons.push(format!(
                "hazard at step {step} of {horizon} ({who}); survives {} step(s)",
                roll.survival_horizon
            ));
            AuditVerdict::RejectLethal
        } else if risk_score >= warn_risk {
            reasons.push(format!(
                "no hazard in {} step(s), but lowest safe_prob {min_safe:.4} gives \
                 risk_score {risk_score:.4} >= warn_risk {warn_risk}",
                roll.steps_simulated()
            ));
            AuditVerdict::WarnHazard
        } else if matches!(
            gate_tier,
            PolicyTier::Tier1Confirm | PolicyTier::Tier2Escalate
        ) {
            reasons.push(format!("PolicyGate requires confirmation: {gate_reason}"));
            AuditVerdict::WarnHazard
        } else {
            reasons.push(format!(
                "no hazard in {} step(s) of horizon {horizon}; risk_score {risk_score:.4} \
                 < warn_risk {warn_risk}",
                roll.steps_simulated()
            ));
            AuditVerdict::Approved
        };
        if roll.safety_calibrated != Some(true) {
            reasons.push(format!(
                "safety estimate is uncalibrated ({}): a model-internal margin, not an \
                 observed failure rate",
                roll.safety_sources.join(",")
            ));
        }
        if !continuation_pruned.is_empty() {
            let ids: Vec<u32> = continuation_pruned.iter().map(|p| p.action.0).collect();
            reasons.push(format!("PolicyGate removed continuation actions {ids:?}"));
        }

        Ok(AuditReport {
            action,
            verdict,
            risk_score,
            reasons,
            gate_tier,
            is_safe: roll.is_safe(),
            survival_horizon: roll.survival_horizon,
            first_hazard_step: roll.first_hazard_step,
            continuation_pruned,
            trajectory: Some(roll),
        })
    }

    /// Multi-mode decision. The gate prunes hard-stopped candidates before any
    /// engine runs; an engine answer outside the pruned frame is refused.
    ///
    /// With a deadline/budget, the caller stops waiting at that deadline and
    /// returns a previously certified candidate or `TimeoutExceeded`. The
    /// returned candidate has `timed_out = true` and no optional trajectory.
    /// `None` for both request and config limits retains unbounded behavior.
    ///
    /// This is a wall-clock wait bound, NOT a hard real-time scheduling guarantee.
    /// OS scheduling and allocation can overrun it. Worker startup occurs during
    /// pipeline construction, before any decision budget begins. A synchronous
    /// model call cannot be killed safely; it may finish in the background. Until
    /// it exits, another budgeted call (including through clones) gets `PlannerBusy`.
    /// Search checkpoints prevent further model steps after expiration.
    pub fn decide(&self, req: &DecideRequest<'_>) -> Result<Decision, PlannerError> {
        let start = Instant::now();
        let configured = resolve_deadline(self.config_deadline, self.config_budget_ms, start)?;
        let requested = resolve_deadline(req.deadline, req.budget_ms, start)?;
        let deadline = match (configured, requested) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let Some(deadline) = deadline else {
            return self.decide_inner(req, &SearchBudget::default(), None);
        };
        let timeout = || PlannerError::TimeoutExceeded(start.elapsed().as_secs_f64() * 1000.0);
        if Instant::now() >= deadline {
            return Err(timeout());
        }
        // Never queue another uninterruptible model call behind a timed-out one.
        if self
            .in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(PlannerError::PlannerBusy);
        }
        let permit = InFlight(self.in_flight.clone());
        // Bound request copying before moving borrowed data into the worker.
        validate_candidates(req.candidates, MAX_DECIDE_CANDIDATES)?;
        let pipeline = self.clone();
        let state = req.state.clone();
        let candidates = req.candidates.to_vec();
        let active_context = req.active_context.clone();
        let (mode, entropy, return_trajectory, horizon) =
            (req.mode, req.entropy, req.return_trajectory, req.horizon);
        let incumbent = Arc::new(ArcSwapOption::empty());
        let worker_incumbent = incumbent.clone();
        let (tx, rx) = mpsc::channel();
        let worker = self.deadline_worker.as_ref().ok_or_else(|| {
            PlannerError::ConvergenceFailure("deadline worker unavailable".into())
        })?;
        worker
            .try_send(Box::new(move || {
                let _permit = permit;
                let request = DecideRequest {
                    active_context,
                    state: &state,
                    candidates: &candidates,
                    mode,
                    entropy,
                    return_trajectory,
                    horizon,
                    deadline: Some(deadline),
                    budget_ms: None,
                };
                let budget = SearchBudget {
                    deadline: Some(deadline),
                    started: start,
                    publish: None,
                };
                let result = pipeline.decide_inner(&request, &budget, Some(&worker_incumbent));
                let finished = Instant::now();
                drop(_permit);
                let _ = tx.send((finished, result));
            }))
            .map_err(|_| {
                PlannerError::ConvergenceFailure("deadline worker queue unavailable".into())
            })?;
        // A timed condvar wait can wake noticeably after its target. Spend the
        // final 200us actively checking the deadline; longer budgets sleep first.
        // This consumes a CPU while polling and still cannot prevent OS preemption.
        let received = loop {
            match rx.try_recv() {
                Ok(result) => break Some(result),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(PlannerError::ConvergenceFailure(
                        "deadline worker disconnected".into(),
                    ));
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break None;
            }
            let active_window = std::time::Duration::from_micros(200);
            if remaining > active_window {
                match rx.recv_timeout(remaining - active_window) {
                    Ok(result) => break Some(result),
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err(PlannerError::ConvergenceFailure(
                            "deadline worker disconnected".into(),
                        ));
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            } else {
                std::hint::spin_loop();
            }
        };
        if let Some((finished, result)) = received {
            // A real model/numeric error completed before cutoff is never replaced
            // by an incumbent, even when delivery itself was delayed.
            if finished < deadline && !matches!(result, Err(PlannerError::TimeoutExceeded(_))) {
                return result;
            }
        }
        // Atomic snapshot: publication cannot hide a previously certified candidate
        // behind a contended mutex. Never join the blocked dynamics call.
        match incumbent.swap(None) {
            Some(decision) => Arc::try_unwrap(decision).map_err(|_| {
                PlannerError::ConvergenceFailure("incumbent unexpectedly shared".into())
            }),
            None => Err(timeout()),
        }
    }

    /// Explicit context snapshot; preserves request deadlines and budgets.
    pub fn decide_with_context(
        &self,
        req: &DecideRequest<'_>,
        active_context: &[ActionId],
    ) -> Result<Decision, PlannerError> {
        let mut request = req.clone();
        request.active_context = active_context.to_vec();
        self.decide(&request)
    }

    fn decide_inner(
        &self,
        req: &DecideRequest<'_>,
        budget: &SearchBudget<'_>,
        incumbent: Option<&ArcSwapOption<Decision>>,
    ) -> Result<Decision, PlannerError> {
        budget.check()?;
        validate_state(req.state)?;
        validate_candidates(req.candidates, MAX_DECIDE_CANDIDATES)?;
        let h = req.entropy.0;
        if !(0.0..=1.0).contains(&h) {
            return Err(PlannerError::InvalidInput(format!(
                "entropy must lie in [0, 1], got {h}"
            )));
        }
        if req.return_trajectory {
            validate_horizon(req.horizon)?;
        }

        let (allowed, mut pruned) =
            self.prune_with_context(req.candidates, &req.active_context, req.entropy);
        // A terminal transition is a hazard regardless of its reward. Screen all
        // candidates before dispatch so every engine receives the same safe frame.
        let mut feasible = Vec::with_capacity(allowed.len());
        let mut hazardous_actions = Vec::new();
        for action in allowed {
            budget.check()?;
            let (next, reward, done) = self.world_model.step(req.state, action)?;
            budget.check()?;
            validate_state(&next)?;
            if !reward.is_finite() {
                return Err(PlannerError::DivergentState(format!(
                    "world model returned reward {reward} while screening action {}",
                    action.0
                )));
            }
            if done {
                hazardous_actions.push(action);
                pruned.push(PrunedAction {
                    action,
                    tier: PolicyTier::Tier3HardStop,
                    violated_rules: Vec::new(),
                    reason: "model predicted terminal hazard".into(),
                });
            } else {
                feasible.push(action);
            }
        }
        if feasible.is_empty() {
            return Err(PlannerError::NoFeasibleAction);
        }
        let names: Vec<String> = feasible.iter().map(|a| format!("action_{}", a.0)).collect();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let frame = LocalActionFrame::new(&name_refs, &feasible).ok_or_else(|| {
            PlannerError::InvalidInput(format!(
                "{} feasible actions exceed the frame capacity {MAX_DECIDE_CANDIDATES}",
                feasible.len()
            ))
        })?;

        let publish = |action: ActionId, engine: &'static str| {
            if incumbent.is_none() || !feasible.contains(&action) {
                return;
            }
            // Full gate evaluation includes the live graph and request entropy.
            let Ok(verdict) = self.gate.evaluate_with_context(
                action,
                &req.active_context,
                None,
                req.entropy,
                Some(self.graph.as_ref()),
                None,
            ) else {
                return;
            };
            if verdict.tier == PolicyTier::Tier3HardStop {
                return;
            }
            if let Some(slot) = incumbent {
                let decision = Decision {
                    timed_out: true,
                    hazard_detected: !hazardous_actions.is_empty(),
                    hazardous_actions: hazardous_actions.clone(),
                    action,
                    entropy: NormalizedEntropy(1.0),
                    mode: req.mode,
                    engine,
                    routing_tier: None,
                    gate_tier: verdict.tier,
                    requires_confirmation: matches!(
                        verdict.tier,
                        PolicyTier::Tier1Confirm | PolicyTier::Tier2Escalate
                    ),
                    feasible: feasible.clone(),
                    pruned: pruned.clone(),
                    trajectory: None,
                    graph_context: GraphContext::Unavailable {
                        reason: "timed-out incumbent: graph context not computed".into(),
                    },
                };
                let decision = Arc::new(decision);
                if budget.check().is_ok() {
                    slot.store(Some(decision));
                }
            }
        };
        let search = SearchBudget {
            deadline: budget.deadline,
            started: budget.started,
            publish: Some(&publish),
        };
        let checked_model = HazardCheckedDynamics(self.world_model.as_ref());
        let wm = &checked_model;
        let gate = self.gate.as_ref();
        let (action, entropy, engine, routing_tier) = match req.mode {
            DecideMode::Auto => {
                let (a, e, tier) = self.router.dispatch_until(
                    req.state,
                    &frame,
                    req.entropy,
                    wm,
                    gate,
                    &search,
                )?;
                (a, e, "DynamicKMoERouter", Some(tier))
            }
            DecideMode::Mcts => run(&self.mcts, req.state, &frame, wm, gate, &search)?,
            DecideMode::MpcCem => run(&self.mpc, req.state, &frame, wm, gate, &search)?,
            DecideMode::AStar => run(&self.astar, req.state, &frame, wm, gate, &search)?,
            DecideMode::ManifoldGFlowNet => run(
                &self.manifold_gflownet,
                req.state,
                &frame,
                wm,
                gate,
                &search,
            )?,
            DecideMode::CfrNash => run(&self.cfr_nash, req.state, &frame, wm, gate, &search)?,
            DecideMode::Reflex => run(&self.cpsat, req.state, &frame, wm, gate, &search)?,
        };
        if !feasible.contains(&action) {
            return Err(PlannerError::ConvergenceFailure(format!(
                "{engine} chose action {} outside the gated frame",
                action.0
            )));
        }

        let gate_tier = self
            .gate
            .evaluate_with_context(
                action,
                &req.active_context,
                None,
                req.entropy,
                Some(self.graph.as_ref()),
                None,
            )
            .map_or(PolicyTier::Tier3HardStop, |v| v.tier);
        if gate_tier == PolicyTier::Tier3HardStop {
            return Err(PlannerError::NoFeasibleAction);
        }

        budget.check()?;
        // Revalidate the selected first transition: model calls may be stateful/stochastic.
        let (next, reward, done) = self.world_model.step(req.state, action)?;
        budget.check()?;
        validate_state(&next)?;
        if !reward.is_finite() {
            return Err(PlannerError::DivergentState(
                "nonfinite selected reward".into(),
            ));
        }
        if done {
            return Err(PlannerError::NoFeasibleAction);
        }

        publish(action, engine);

        let trajectory = if req.return_trajectory {
            Some(self.greedy_rollout(
                req.state,
                action,
                &feasible,
                req.horizon,
                POLICY_GREEDY,
                budget,
            )?)
        } else {
            None
        };

        let graph_context = self.graph_context(action, &req.active_context);
        budget.check()?;
        Ok(Decision {
            timed_out: false,
            action,
            entropy,
            mode: req.mode,
            engine,
            routing_tier,
            gate_tier,
            requires_confirmation: matches!(
                gate_tier,
                PolicyTier::Tier1Confirm | PolicyTier::Tier2Escalate
            ),
            hazard_detected: !hazardous_actions.is_empty(),
            hazardous_actions,
            feasible,
            pruned,
            trajectory,
            graph_context,
        })
    }

    /// PPR over the live graph from the chosen action's entity node (entity id ==
    /// action id, the key the gate's revocation check uses) plus the
    /// active-context actions that have nodes. A PPR error is reported in the
    /// context, never swallowed.
    fn graph_context(&self, action: ActionId, active_context: &[ActionId]) -> GraphContext {
        let entity = u64::from(action.0);
        let Some(root) = self.graph.node_for_entity(entity) else {
            return GraphContext::Unavailable {
                reason: format!("action {} has no node in the live graph", action.0),
            };
        };
        let mut seed_entities = vec![entity];
        let mut seeds = vec![(root, 1.0)];
        for ctx in active_context {
            let e = u64::from(ctx.0);
            if seed_entities.contains(&e) {
                continue;
            }
            if let Some(node) = self.graph.node_for_entity(e) {
                seed_entities.push(e);
                seeds.push((node, 1.0));
            }
        }
        let ranking = match self.graph.query_ppr(
            &seeds,
            GRAPH_CONTEXT_ALPHA,
            GRAPH_CONTEXT_MAX_ITERS,
            GRAPH_CONTEXT_TOLERANCE,
        ) {
            Ok(r) => r,
            Err(e) => {
                return GraphContext::Unavailable {
                    reason: format!("graph PPR failed: {e}"),
                }
            }
        };
        let facts = ranking
            .ranked
            .iter()
            .filter(|(id, score)| *score > 0.0 && !seeds.iter().any(|(s, _)| s == id))
            .filter_map(|&(id, score)| {
                self.graph.get_node(id).map(|n| GraphFact {
                    entity_id: n.entity_id,
                    label: n.label,
                    status: n.status,
                    score,
                })
            })
            .take(GRAPH_CONTEXT_TOP)
            .collect();
        GraphContext::Diffused {
            seed_entities,
            facts,
            iterations: ranking.iterations,
            converged: ranking.converged,
        }
    }

    /// Gate tier, violated rule ids and reason at entropy 0 (hard constraints and
    /// confirm lists only). A gate error is a hard stop.
    fn gate_check(&self, action: ActionId) -> (PolicyTier, Vec<u32>, String) {
        self.gate_check_with_context(action, &[], NormalizedEntropy::ZERO)
    }

    fn gate_check_with_context(
        &self,
        action: ActionId,
        active_context: &[ActionId],
        entropy: NormalizedEntropy,
    ) -> (PolicyTier, Vec<u32>, String) {
        match self.gate.evaluate_with_context(
            action,
            active_context,
            None,
            entropy,
            Some(self.graph.as_ref()),
            None,
        ) {
            Ok(v) => (
                v.tier,
                v.violated_rules.iter().map(|r| r.0).collect(),
                v.reason,
            ),
            Err(e) => (
                PolicyTier::Tier3HardStop,
                Vec::new(),
                format!("gate error treated as hard stop: {e}"),
            ),
        }
    }

    /// Split into (allowed, pruned), keeping request order.
    fn prune(&self, candidates: &[ActionId]) -> (Vec<ActionId>, Vec<PrunedAction>) {
        self.prune_with_context(candidates, &[], NormalizedEntropy::ZERO)
    }

    fn prune_with_context(
        &self,
        candidates: &[ActionId],
        active_context: &[ActionId],
        entropy: NormalizedEntropy,
    ) -> (Vec<ActionId>, Vec<PrunedAction>) {
        let mut allowed = Vec::with_capacity(candidates.len());
        let mut pruned = Vec::new();
        for &a in candidates {
            let (tier, violated_rules, reason) =
                self.gate_check_with_context(a, active_context, entropy);
            if tier == PolicyTier::Tier3HardStop {
                pruned.push(PrunedAction {
                    action: a,
                    tier,
                    violated_rules,
                    reason,
                });
            } else {
                allowed.push(a);
            }
        }
        (allowed, pruned)
    }

    /// Step 1 plays `first`; later steps pick the one-step best in `set`, ranked
    /// non-hazard first, then higher safe_prob, then higher reward. Ties keep `set`
    /// order.
    fn greedy_rollout(
        &self,
        state: &FullLatent,
        first: ActionId,
        set: &[ActionId],
        horizon: usize,
        policy: &'static str,
        budget: &SearchBudget<'_>,
    ) -> Result<Rollout, PlannerError> {
        self.rollout(state, horizon, policy, budget, |step_idx, current| {
            if step_idx == 1 {
                return Ok(first);
            }
            let mut best: Option<(ActionId, (bool, f32, f32))> = None;
            for &a in set {
                if self.gate_check(a).0 == PolicyTier::Tier3HardStop {
                    continue;
                }
                budget.check()?;
                let (next, r, d) = self.world_model.step(current, a)?;
                budget.check()?;
                if !r.is_finite() {
                    return Err(PlannerError::DivergentState(format!(
                        "world model returned reward {r} while scoring continuation action {}",
                        a.0
                    )));
                }
                validate_state(&next)?;
                let safe = self
                    .world_model
                    .safety_estimate(&next, r, d)
                    .map_or(0.0, |e| e.safe_prob);
                if !(0.0..=1.0).contains(&safe) {
                    return Err(PlannerError::InvalidSafetyEstimate(safe));
                }
                let key = (!d, safe, r);
                // Lexicographic, false < true; strict so ties keep `set` order.
                let better = match &best {
                    None => true,
                    Some((_, k)) => key.partial_cmp(k) == Some(std::cmp::Ordering::Greater),
                };
                if better {
                    best = Some((a, key));
                }
            }
            best.map(|(a, _)| a).ok_or(PlannerError::NoFeasibleAction)
        })
    }

    fn rollout(
        &self,
        state: &FullLatent,
        horizon: usize,
        policy_name: &'static str,
        budget: &SearchBudget<'_>,
        mut policy: impl FnMut(usize, &FullLatent) -> Result<ActionId, PlannerError>,
    ) -> Result<Rollout, PlannerError> {
        let mut steps = Vec::with_capacity(horizon);
        let mut current = state.clone();
        let mut cumulative = 0.0_f32;
        let mut termination_step = None;
        let mut first_hazard_step = None;
        let mut min_safe: Option<f32> = None;
        let mut coverage = 0;
        let mut all_calibrated = true;
        let mut sources: Vec<&'static str> = Vec::new();

        for step_idx in 1..=horizon {
            budget.check()?;
            let action = policy(step_idx, &current)?;
            let gate_tier = self.gate_check(action).0;
            if gate_tier == PolicyTier::Tier3HardStop {
                return Err(PlannerError::NoFeasibleAction);
            }
            budget.check()?;
            let (next, reward, done) = self.world_model.step(&current, action)?;
            budget.check()?;
            if !reward.is_finite() {
                return Err(PlannerError::DivergentState(format!(
                    "world model returned reward {reward} at step {step_idx}"
                )));
            }
            validate_state(&next)?;
            let estimate = self.world_model.safety_estimate(&next, reward, done);
            let safe_prob = match estimate {
                Some(e) => {
                    if !(0.0..=1.0).contains(&e.safe_prob) {
                        return Err(PlannerError::InvalidSafetyEstimate(e.safe_prob));
                    }
                    coverage += 1;
                    all_calibrated &= e.calibrated;
                    if !sources.contains(&e.source) {
                        sources.push(e.source);
                    }
                    min_safe = Some(min_safe.map_or(e.safe_prob, |m: f32| m.min(e.safe_prob)));
                    Some(e.safe_prob)
                }
                None => None,
            };
            cumulative += reward;
            if !cumulative.is_finite() {
                return Err(PlannerError::DivergentState(format!(
                    "cumulative return {cumulative} is not finite at step {step_idx}"
                )));
            }
            let hazard = done;
            if hazard && first_hazard_step.is_none() {
                first_hazard_step = Some(step_idx);
            }
            steps.push(SimStep {
                step_idx,
                action,
                state: next.clone(),
                reward,
                safe_prob,
                done,
                hazard,
                gate_tier,
            });
            current = next;
            if done {
                termination_step = Some(step_idx);
                break;
            }
        }

        let survival_horizon = first_hazard_step.map_or(steps.len(), |s| s - 1);
        Ok(Rollout {
            survival_horizon,
            cumulative_return: cumulative,
            terminated_early: termination_step.is_some(),
            termination_step,
            first_hazard_step,
            min_safe_prob: min_safe,
            safety_coverage: coverage,
            safety_calibrated: (coverage > 0).then_some(all_calibrated),
            safety_sources: sources,
            continuation_policy: policy_name,
            steps,
            final_state: current,
        })
    }
}

fn run(
    engine: &dyn PlanningEngine,
    state: &FullLatent,
    frame: &LocalActionFrame<'_>,
    wm: &dyn WorldModelDynamics<Error = CoreError>,
    gate: &PolicyGate,
    budget: &SearchBudget<'_>,
) -> Result<
    (
        ActionId,
        NormalizedEntropy,
        &'static str,
        Option<RoutingTier>,
    ),
    PlannerError,
> {
    let (a, e) = engine.plan_until(state, frame, wm, gate, budget)?;
    Ok((a, e, engine.name(), None))
}

/// A non-finite coordinate, or a norm that overflows f32, is a diverged state.
fn validate_state(state: &FullLatent) -> Result<(), PlannerError> {
    if let Some(i) = state.as_slice().iter().position(|x| !x.is_finite()) {
        return Err(PlannerError::DivergentState(format!(
            "state coordinate {i} is not finite"
        )));
    }
    let norm = state.l2_norm();
    if !norm.is_finite() {
        return Err(PlannerError::DivergentState(format!(
            "state L2 norm overflows ({norm})"
        )));
    }
    Ok(())
}

fn validate_horizon(horizon: usize) -> Result<(), PlannerError> {
    if horizon == 0 || horizon > MAX_HORIZON {
        return Err(PlannerError::InvalidHorizon {
            horizon,
            max: MAX_HORIZON,
        });
    }
    Ok(())
}

fn validate_candidates(candidates: &[ActionId], max: usize) -> Result<(), PlannerError> {
    if candidates.is_empty() {
        return Err(PlannerError::NoFeasibleAction);
    }
    if candidates.len() > max {
        return Err(PlannerError::InvalidInput(format!(
            "{} candidates exceed the cap of {max}",
            candidates.len()
        )));
    }
    for (i, a) in candidates.iter().enumerate() {
        if candidates[..i].contains(a) {
            return Err(PlannerError::InvalidInput(format!(
                "duplicate candidate action {}",
                a.0
            )));
        }
    }
    Ok(())
}

/// Held by the worker, including after its caller has stopped waiting.
struct InFlight(Arc<AtomicBool>);
impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

// Eager readiness moves thread startup out of the decision budget. The bounded
// queue is shared by pipeline clones; jobs also hold the in-flight permit. There
// is no per-request thread creation and no model work during initialization.
type DeadlineJob = Box<dyn FnOnce() + Send + 'static>;
fn start_deadline_worker() -> Option<mpsc::SyncSender<DeadlineJob>> {
    let (tx, rx) = mpsc::sync_channel::<DeadlineJob>(1);
    let (ready_tx, ready_rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("planner-deadline".into())
        .spawn(move || {
            let _ = ready_tx.send(());
            while let Ok(job) = rx.recv() {
                job();
            }
        })
        .ok()?;
    ready_rx.recv().ok()?;
    Some(tx)
}
