//! Proof that `PlannerConfig`'s knobs reach the engines and change real
//! decisions: `mcts_c_puct` for `MctsEngine`, `astar_uncertainty_penalty_weight`
//! for `AStarEngine`. Both models below are deterministic (no RNG anywhere in
//! the engines under test), so every assertion is exact, not statistical.

use gen_zero_core::{ActionId, CoreError, FullLatent, NormalizedEntropy, WorldModelDynamics};
use gen_zero_gate::PolicyGate;
use gen_zero_planner::{
    DecideMode, DecideRequest, PlannerConfig, PlannerError, ProductionPipeline,
};
use std::sync::Arc;

/// Fixed reward per action id; the transition never changes the state and never
/// terminates. Isolates the engine's own scoring math from world-model effects.
struct FixedRewardModel {
    rewards: [(u32, f32); 3],
}

impl WorldModelDynamics for FixedRewardModel {
    type Error = CoreError;

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), CoreError> {
        let reward = self
            .rewards
            .iter()
            .find(|(id, _)| *id == action.0)
            .map(|(_, r)| *r)
            .unwrap_or(0.0);
        Ok((state.clone(), reward, false))
    }

    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next_states: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), CoreError> {
        for i in 0..states.len() {
            let (n, r, d) = self.step(&states[i], actions[i])?;
            next_states[i] = n;
            rewards[i] = r;
            dones[i] = d;
        }
        Ok(())
    }
}

const ACTS: [ActionId; 3] = [ActionId(1), ActionId(2), ActionId(3)];

fn decide_req<'a>(state: &'a FullLatent, mode: DecideMode) -> DecideRequest<'a> {
    DecideRequest {
        active_context: Vec::new(),
        deadline: None,
        budget_ms: None,
        state,
        candidates: &ACTS,
        mode,
        entropy: NormalizedEntropy(0.5),
        return_trajectory: false,
        horizon: 4,
        causal_triad: None,
    }
}

/// Action 2 has the strictly highest reward. With a small `c_puct`, PUCT's
/// exploration term stays small relative to the reward gap, so the 5-simulation
/// search commits its extra visits to the true best action.
#[test]
fn low_c_puct_converges_to_the_highest_reward_action() {
    let model = FixedRewardModel {
        rewards: [(1, 2.9), (2, 3.0), (3, 2.9)],
    };
    let config = PlannerConfig {
        mcts_max_simulations: 5,
        mcts_c_puct: 0.1,
        ..PlannerConfig::default()
    };
    let pipeline = ProductionPipeline::new_with_config(
        Arc::new(model),
        Arc::new(PolicyGate::default()),
        config,
    )
    .unwrap();
    let state = FullLatent::zeros();
    let decision = pipeline
        .decide(&decide_req(&state, DecideMode::Mcts))
        .unwrap();
    assert_eq!(
        decision.action,
        ActionId(2),
        "low c_puct must pick the highest-reward action"
    );
}

/// Same rewards, same 5 simulations, only `mcts_c_puct` raised. The larger
/// exploration bonus outweighs the (small) reward gap on the deciding
/// simulation, so the visit-count leader flips to a lower-reward, less-visited
/// action. This is the exact mechanism `mcts_c_puct` exists to control: proof
/// the knob is load-bearing, not decorative.
#[test]
fn high_c_puct_flips_the_decision_away_from_the_reward_optimum() {
    let model = FixedRewardModel {
        rewards: [(1, 2.9), (2, 3.0), (3, 2.9)],
    };
    let config = PlannerConfig {
        mcts_max_simulations: 5,
        mcts_c_puct: 1.0,
        ..PlannerConfig::default()
    };
    let pipeline = ProductionPipeline::new_with_config(
        Arc::new(model),
        Arc::new(PolicyGate::default()),
        config,
    )
    .unwrap();
    let state = FullLatent::zeros();
    let decision = pipeline
        .decide(&decide_req(&state, DecideMode::Mcts))
        .unwrap();
    assert_eq!(
        decision.action,
        ActionId(1),
        "high c_puct must overweight exploration and abandon the reward optimum"
    );
}

/// A world model whose two actions trade reward for state displacement:
/// action 1 has a larger negative reward and moves only the goal coordinate;
/// action 2 has a smaller negative reward and also moves 10 units laterally.
struct RewardVsDisplacementModel;

impl WorldModelDynamics for RewardVsDisplacementModel {
    type Error = CoreError;

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), CoreError> {
        let mut next = state.clone();
        let (reward, delta) = match action.0 {
            1 => (-1.2, 0.0),
            2 => (-1.0, 10.0),
            other => panic!("unexpected action {other} in this test model"),
        };
        next.as_mut_slice()[0] += delta;
        next.as_mut_slice()[1] = 1.0;
        Ok((next, reward, false))
    }

    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next_states: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), CoreError> {
        for i in 0..states.len() {
            let (n, r, d) = self.step(&states[i], actions[i])?;
            next_states[i] = n;
            rewards[i] = r;
            dones[i] = d;
        }
        Ok(())
    }
}

const DISPLACEMENT_ACTS: [ActionId; 2] = [ActionId(1), ActionId(2)];

/// With no displacement penalty, edge costs are 2.0 for action 2 and 2.2
/// for action 1. Both reach the explicitly configured goal.
#[test]
fn zero_uncertainty_weight_picks_the_higher_reward_despite_displacement() {
    let config = PlannerConfig {
        astar_uncertainty_penalty_weight: 0.0,
        ..PlannerConfig::default()
    };
    let pipeline = ProductionPipeline::new_with_config(
        Arc::new(RewardVsDisplacementModel),
        Arc::new(PolicyGate::default()),
        config,
    )
    .unwrap();
    let pipeline = pipeline.with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
        s.as_slice()[1] == 1.0
    }));
    let state = FullLatent::zeros();
    let decision = pipeline
        .decide(&DecideRequest {
            active_context: Vec::new(),
            deadline: None,
            budget_ms: None,
            state: &state,
            candidates: &DISPLACEMENT_ACTS,
            mode: DecideMode::AStar,
            entropy: NormalizedEntropy(0.5),
            return_trajectory: false,
            horizon: 4,
            causal_triad: None,
        })
        .unwrap();
    assert_eq!(decision.action, ActionId(2));
}

/// Weight 1 adds about 0.5025 to action 2 and 0.05 to action 1,
/// making the calmer action cheaper (2.25 versus about 2.5025).
#[test]
fn high_uncertainty_weight_picks_the_calmer_lower_reward_action() {
    let config = PlannerConfig {
        astar_uncertainty_penalty_weight: 1.0,
        ..PlannerConfig::default()
    };
    let pipeline = ProductionPipeline::new_with_config(
        Arc::new(RewardVsDisplacementModel),
        Arc::new(PolicyGate::default()),
        config,
    )
    .unwrap();
    let pipeline = pipeline.with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
        s.as_slice()[1] == 1.0
    }));
    let state = FullLatent::zeros();
    let decision = pipeline
        .decide(&DecideRequest {
            active_context: Vec::new(),
            deadline: None,
            budget_ms: None,
            state: &state,
            candidates: &DISPLACEMENT_ACTS,
            mode: DecideMode::AStar,
            entropy: NormalizedEntropy(0.5),
            return_trajectory: false,
            horizon: 4,
            causal_triad: None,
        })
        .unwrap();
    assert_eq!(decision.action, ActionId(1));
}

/// `new_with_config` must fail closed on a config that would silently degrade
/// the search (e.g. zero simulations never runs one), not build a pipeline that
/// looks fine and then never searches.
#[test]
fn new_with_config_refuses_an_invalid_config() {
    let bad = PlannerConfig {
        mcts_max_simulations: 0,
        ..PlannerConfig::default()
    };
    let result = ProductionPipeline::new_with_config(
        Arc::new(RewardVsDisplacementModel),
        Arc::new(PolicyGate::default()),
        bad,
    );
    assert!(matches!(result, Err(PlannerError::InvalidInput(_))));
}

/// The two stateless engines are still part of the configured pipeline: callers
/// select them on the request while the constructor validates and installs the
/// same config used by the other explicit modes.
#[test]
fn configured_pipeline_selects_gflownet_and_cfr_nash_from_the_request() {
    let pipeline = ProductionPipeline::new_with_config(
        Arc::new(FixedRewardModel {
            rewards: [(1, 1.0), (2, 2.0), (3, 3.0)],
        }),
        Arc::new(PolicyGate::default()),
        PlannerConfig::default(),
    )
    .unwrap();
    let state = FullLatent::zeros();

    for (mode, engine) in [
        (DecideMode::ManifoldGFlowNet, "ManifoldGFlowNetEngine"),
        (DecideMode::CfrNash, "CfrNashEngine"),
    ] {
        let decision = pipeline.decide(&decide_req(&state, mode)).unwrap();
        assert_eq!(decision.mode, mode);
        assert_eq!(decision.engine, engine);
        if mode == DecideMode::ManifoldGFlowNet {
            assert!([ActionId(1), ActionId(2), ActionId(3)].contains(&decision.action));
        } else {
            assert_eq!(decision.action, ActionId(3));
        }
    }
}
