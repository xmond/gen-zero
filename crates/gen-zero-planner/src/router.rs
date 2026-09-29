//! Dynamic K-MoE Router guiding execution across K=1, 2, 3.
//!
//! K=1 (Reflex): Goal-directed A* search for clear, low-entropy states (H <= 0.2).
//! K=2 (Pipeline): PolicyGate filter followed by multi-step PUCT for moderate uncertainty (0.2 < H <= 0.7).
//! K=3 (Committee): Multi-engine consensus committee voting for high-entropy / adversarial states (H > 0.7).

use crate::config::PlannerConfig;
use crate::engine::SearchBudget;
use crate::engine::{
    shannon_entropy, AStarEngine, CpSatFormalEngine, MctsEngine, MpcCemEngine, PlanningEngine,
};
use crate::error::PlannerError;
use gen_zero_core::{
    ActionId, CoreError, FullLatent, LocalActionFrame, NormalizedEntropy, WorldModelDynamics,
};
use gen_zero_gate::{PolicyGate, PolicyTier};

/// Routing mode for planning deliberation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RoutingTier {
    /// K=1: Goal-directed A* search
    K1Reflex,
    /// K=2: PolicyGate filtering followed by multi-step PUCT
    K2Pipeline,
    /// K=3: Consensus committee voting across multiple engines
    K3Committee,
}

/// Dynamic K-MoE Router.
#[derive(Clone)]
pub struct DynamicKMoERouter {
    pub entropy_threshold_low: f32,
    pub entropy_threshold_high: f32,
    mcts: MctsEngine,
    astar: AStarEngine,
    cpsat: CpSatFormalEngine,
    mpc: MpcCemEngine,
}

impl Default for DynamicKMoERouter {
    fn default() -> Self {
        Self {
            entropy_threshold_low: 0.20,
            entropy_threshold_high: 0.70,
            mcts: MctsEngine::default(),
            astar: AStarEngine::default(),
            cpsat: CpSatFormalEngine,
            mpc: MpcCemEngine::default(),
        }
    }
}

fn validate_entropy(entropy: NormalizedEntropy) -> Result<(), PlannerError> {
    if !entropy.0.is_finite() || !(0.0..=1.0).contains(&entropy.0) {
        return Err(PlannerError::InvalidInput(format!(
            "entropy must be finite and lie in [0, 1], got {}",
            entropy.0
        )));
    }
    Ok(())
}

impl DynamicKMoERouter {
    /// Configure explicit A* success for direct and routed searches.
    pub fn with_astar_goal(mut self, goal: crate::AStarGoal) -> Self {
        self.astar.goal = Some(goal);
        self
    }

    /// Build a router whose thresholds and member engines all come from `config`.
    /// Caller must call `config.validate()` first: this never re-checks it.
    pub fn from_config(config: &PlannerConfig) -> Self {
        Self {
            entropy_threshold_low: config.router_entropy_threshold_low,
            entropy_threshold_high: config.router_entropy_threshold_high,
            mcts: MctsEngine {
                max_simulations: config.mcts_max_simulations,
                c_puct: config.mcts_c_puct,
                horizon: config.mcts_horizon,
                discount: config.mcts_discount,
                ..MctsEngine::default()
            },
            astar: AStarEngine {
                uncertainty_penalty_weight: config.astar_uncertainty_penalty_weight,
                ..AStarEngine::default()
            },
            cpsat: CpSatFormalEngine,
            mpc: MpcCemEngine {
                num_samples: config.cem_num_samples,
                horizon: config.cem_horizon,
                gamma: config.cem_gamma,
                ..MpcCemEngine::default()
            },
        }
    }

    /// Classify target routing tier based on perceptual entropy and gate tier.
    ///
    /// Routing is fail-closed: malformed entropy, malformed thresholds, and a
    /// hard-stop verdict are refused instead of being translated into a more
    /// permissive planning tier.
    pub fn classify_tier(
        &self,
        entropy: NormalizedEntropy,
        gate_tier: PolicyTier,
    ) -> Result<RoutingTier, PlannerError> {
        validate_entropy(entropy)?;
        if !self.entropy_threshold_low.is_finite()
            || !self.entropy_threshold_high.is_finite()
            || !(0.0..=1.0).contains(&self.entropy_threshold_low)
            || !(0.0..=1.0).contains(&self.entropy_threshold_high)
            || self.entropy_threshold_low > self.entropy_threshold_high
        {
            return Err(PlannerError::InvalidInput(
                "router entropy thresholds must be finite, in [0, 1], and low <= high".into(),
            ));
        }

        if gate_tier == PolicyTier::Tier3HardStop {
            return Err(PlannerError::NoFeasibleAction);
        }

        if entropy.0 <= self.entropy_threshold_low && gate_tier == PolicyTier::Tier0Proceed {
            Ok(RoutingTier::K1Reflex)
        } else if entropy.0 <= self.entropy_threshold_high {
            Ok(RoutingTier::K2Pipeline)
        } else {
            Ok(RoutingTier::K3Committee)
        }
    }

    /// Dispatch planning deliberation through the dynamic tier.
    pub fn dispatch(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        entropy: NormalizedEntropy,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        self.dispatch_with_tier(state, actions, entropy, world_model, gate)
            .map(|(act, h, _)| (act, h))
    }

    /// [`Self::dispatch`], also returning the tier that ran.
    ///
    /// Gate evaluation is performed at the request entropy before dispatching
    /// to any engine. A malformed entropy value, gate error, or hard-stop
    /// verdict refuses the request; no hard-stopped request gets a chance to
    /// reach an engine's legacy zero-entropy feasibility check.
    pub fn dispatch_with_tier(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        entropy: NormalizedEntropy,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
    ) -> Result<(ActionId, NormalizedEntropy, RoutingTier), PlannerError> {
        self.dispatch_until(
            state,
            actions,
            entropy,
            world_model,
            gate,
            &SearchBudget::default(),
        )
    }

    pub(crate) fn dispatch_until(
        &self,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        entropy: NormalizedEntropy,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy, RoutingTier), PlannerError> {
        budget.check()?;
        let action_slice = actions.actions();
        if action_slice.is_empty() {
            return Err(PlannerError::NoFeasibleAction);
        }

        // Validate before invoking the gate. PolicyGate itself returns a
        // hard-stop verdict for malformed entropy, but allowing that verdict
        // to flow into the old K2 fallback would let engines re-check at H=0.
        validate_entropy(entropy)?;

        let mut worst_tier = PolicyTier::Tier0Proceed;
        for &act in action_slice {
            let verdict = gate.evaluate_basic(act, entropy).map_err(|error| {
                PlannerError::InvalidInput(format!(
                    "policy gate evaluation failed for action {}: {error}",
                    act.0
                ))
            })?;
            if verdict.tier == PolicyTier::Tier3HardStop {
                return Err(PlannerError::NoFeasibleAction);
            }
            worst_tier = worst_tier.max(verdict.tier);
        }

        let tier = self.classify_tier(entropy, worst_tier)?;
        self.run_tier(tier, state, actions, world_model, gate, budget)
            .map(|(act, h)| (act, h, tier))
    }

    fn run_tier(
        &self,
        tier: RoutingTier,
        state: &FullLatent,
        actions: &LocalActionFrame<'_>,
        world_model: &dyn WorldModelDynamics<Error = CoreError>,
        gate: &PolicyGate,
        budget: &SearchBudget<'_>,
    ) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
        budget.check()?;
        match tier {
            RoutingTier::K1Reflex => {
                // K=1: Goal-directed A* (an explicit goal is required)
                self.astar
                    .plan_until(state, actions, world_model, gate, budget)
            }
            RoutingTier::K2Pipeline => {
                // K=2: PolicyGate filtering followed by multi-step PUCT ranking
                let feasible_acts = self.cpsat.filter_feasible(actions, gate);
                if feasible_acts.is_empty() {
                    return Err(PlannerError::NoFeasibleAction);
                }

                // If feasible subset is strictly smaller, construct a filtered action frame
                if let Some(filtered_frame) = actions.filter_by_actions(&feasible_acts) {
                    self.mcts
                        .plan_until(state, &filtered_frame, world_model, gate, budget)
                } else {
                    self.mcts
                        .plan_until(state, actions, world_model, gate, budget)
                }
            }
            RoutingTier::K3Committee => {
                // K=3: Consensus committee: MCTS, MPC-CEM, and A* vote
                // Fixed stack array tracking votes for up to 16 actions (0-heap alloc)
                let acts = actions.actions();
                let mut votes = [0usize; 16];

                let vote_for = |act: ActionId, weight: usize, votes: &mut [usize; 16]| {
                    if let Some(idx) = actions.to_local(act) {
                        if idx < 16 {
                            votes[idx] += weight;
                        }
                    }
                };

                // Any member error aborts the committee. Dropping a failed member
                // would let the survivors outvote a world-model fault unseen.
                let (act_mcts, _) =
                    self.mcts
                        .plan_until(state, actions, world_model, gate, budget)?;
                vote_for(act_mcts, 2, &mut votes); // Higher weight for MCTS
                let (act_mpc, _) =
                    self.mpc
                        .plan_until(state, actions, world_model, gate, budget)?;
                vote_for(act_mpc, 1, &mut votes);
                let (act_astar, _) =
                    self.astar
                        .plan_until(state, actions, world_model, gate, budget)?;
                vote_for(act_astar, 1, &mut votes);

                // Pick action with highest consensus vote. Report normalized vote
                // entropy as committee disagreement, not calibrated confidence.
                let mut max_vote = 0;
                let mut best_act = None;
                for (idx, &v) in votes.iter().enumerate().take(acts.len()) {
                    if v > max_vote {
                        max_vote = v;
                        best_act = Some(acts[idx]);
                    }
                }

                // No vote landed in the frame: refuse rather than hand back the
                // unvetted `acts[0]`.
                best_act
                    .map(|winner| {
                        let total = votes.iter().sum::<usize>() as f32;
                        let probabilities: Vec<f32> = votes[..acts.len()]
                            .iter()
                            .map(|&v| v as f32 / total)
                            .collect();
                        (winner, shannon_entropy(&probabilities))
                    })
                    .ok_or(PlannerError::NoFeasibleAction)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PanicDynamics;

    impl WorldModelDynamics for PanicDynamics {
        type Error = CoreError;

        fn step(
            &self,
            _state: &FullLatent,
            _action: ActionId,
        ) -> Result<(FullLatent, f32, bool), Self::Error> {
            panic!("router dispatched to an engine after a preflight refusal");
        }

        fn step_batch(
            &self,
            _states: &[FullLatent],
            _actions: &[ActionId],
            _next_states: &mut [FullLatent],
            _rewards: &mut [f32],
            _dones: &mut [bool],
        ) -> Result<(), Self::Error> {
            panic!("router dispatched to an engine after a preflight refusal");
        }
    }

    fn test_frame() -> LocalActionFrame<'static> {
        let actions = [ActionId(1)];
        LocalActionFrame::new(&NAMES, &actions).unwrap()
    }

    static NAMES: [&str; 1] = ["first"];

    #[test]
    fn test_router_tier_classification() {
        let router = DynamicKMoERouter::default();

        let low_entropy = NormalizedEntropy(0.1);
        assert_eq!(
            router
                .classify_tier(low_entropy, PolicyTier::Tier0Proceed)
                .unwrap(),
            RoutingTier::K1Reflex
        );

        let mid_entropy = NormalizedEntropy(0.5);
        assert_eq!(
            router
                .classify_tier(mid_entropy, PolicyTier::Tier0Proceed)
                .unwrap(),
            RoutingTier::K2Pipeline
        );

        let high_entropy = NormalizedEntropy(0.85);
        assert_eq!(
            router
                .classify_tier(high_entropy, PolicyTier::Tier0Proceed)
                .unwrap(),
            RoutingTier::K3Committee
        );
    }

    #[test]
    fn classification_refuses_malformed_entropy_and_hard_stop() {
        let router = DynamicKMoERouter::default();

        for entropy in [
            NormalizedEntropy(f32::NAN),
            NormalizedEntropy(f32::INFINITY),
            NormalizedEntropy(-0.1),
            NormalizedEntropy(1.1),
        ] {
            assert!(matches!(
                router.classify_tier(entropy, PolicyTier::Tier0Proceed),
                Err(PlannerError::InvalidInput(_))
            ));
        }
        assert_eq!(
            router.classify_tier(NormalizedEntropy::ZERO, PolicyTier::Tier3HardStop),
            Err(PlannerError::NoFeasibleAction)
        );
    }

    #[test]
    fn dispatch_refuses_invalid_entropy_before_any_engine_call() {
        let router = DynamicKMoERouter::default();
        let actions = test_frame();
        let result = router.dispatch(
            &FullLatent::zeros(),
            &actions,
            NormalizedEntropy(f32::NAN),
            &PanicDynamics,
            &PolicyGate::default(),
        );

        assert!(matches!(result, Err(PlannerError::InvalidInput(_))));
    }

    #[test]
    fn dispatch_refuses_gate_hard_stop_before_any_engine_call() {
        let router = DynamicKMoERouter::default();
        let actions = test_frame();
        let mut gate = PolicyGate::default();
        gate.add_constraint(gen_zero_gate::LinearConstraint::prohibit(
            gen_zero_gate::RuleId(1),
            "blocked",
            ActionId(1),
        ));
        let result = router.dispatch(
            &FullLatent::zeros(),
            &actions,
            NormalizedEntropy::ZERO,
            &PanicDynamics,
            &gate,
        );

        assert_eq!(result, Err(PlannerError::NoFeasibleAction));
    }
}
