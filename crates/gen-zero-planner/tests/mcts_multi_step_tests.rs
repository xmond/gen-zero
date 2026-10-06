//! Integration coverage for finite-horizon MCTS.
//!
//! The model below encodes the rollout stage in latent coordinate zero and the
//! root branch in coordinate one.  That makes a call made from a cached
//! successor observable without exposing the engine's private tree.

use gen_zero_core::{
    ActionId, CoreError, FullLatent, LocalActionFrame, NormalizedEntropy, WorldModelDynamics,
};
use gen_zero_gate::{LinearConstraint, PolicyGate, RuleId};
use gen_zero_planner::{
    DecideMode, DecideRequest, MctsEngine, PlannerConfig, PlannerError, PlanningEngine,
    ProductionPipeline,
};
use std::sync::{Arc, Mutex};

const ACTIONS: [ActionId; 2] = [ActionId(1), ActionId(2)];
const ACTION_NAMES: [&str; 2] = ["A", "B"];

#[derive(Clone, Copy, Debug)]
enum Scenario {
    /// A is best at depth one; B is best after a delayed payoff.
    Delayed,
    /// A terminates immediately.  Stepping its successor would incur a large
    /// penalty and is therefore easy to detect.
    Terminal,
    /// Every transition is live and records the depth at which it was called.
    Horizon,
    /// B has a high payoff only after taking its second action at depth two.
    DeepChoice,
    /// The first transition succeeds, but every deeper transition fails.
    DeepError,
    /// The first transition returns a non-finite successor or reward.
    NonFiniteSuccessor,
    NonFiniteReward,
}

#[derive(Clone, Copy, Debug)]
struct TransitionCall {
    stage: usize,
    branch: u32,
    action: ActionId,
}

struct TestModel {
    scenario: Scenario,
    calls: Arc<Mutex<Vec<TransitionCall>>>,
}

impl TestModel {
    fn new(scenario: Scenario) -> Self {
        Self {
            scenario,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn calls(&self) -> Vec<TransitionCall> {
        self.calls.lock().expect("call log lock poisoned").clone()
    }
}

impl WorldModelDynamics for TestModel {
    type Error = CoreError;

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), Self::Error> {
        let stage = state.as_slice()[0] as usize;
        let branch = state.as_slice()[1] as u32;
        self.calls
            .lock()
            .expect("call log lock poisoned")
            .push(TransitionCall {
                stage,
                branch,
                action,
            });

        if matches!(self.scenario, Scenario::DeepError) && stage >= 1 {
            return Err(CoreError::WorldModel("deep transition failed".into()));
        }

        let (reward, done) = match self.scenario {
            Scenario::Delayed => match (stage, branch, action.0) {
                (0, _, 1) => (1.0, false),
                (0, _, 2) => (0.1, false),
                (_, 1, _) => (-10.0, false),
                (1, 2, _) => (100.0, false),
                _ => (0.0, false),
            },
            Scenario::Terminal => match (stage, branch, action.0) {
                (0, _, 1) => (1.0, true),
                (0, _, 2) => (0.0, false),
                (1, 1, _) => (-100.0, false),
                _ => (0.0, false),
            },
            Scenario::DeepChoice => match (stage, branch, action.0) {
                (0, _, 1) => (1.0, false),
                (0, _, 2) => (0.1, false),
                (1, 2, 2) => (100.0, false),
                _ => (0.0, false),
            },
            Scenario::Horizon | Scenario::DeepError => (0.0, false),
            Scenario::NonFiniteSuccessor | Scenario::NonFiniteReward => (0.0, false),
        };

        let mut next = state.clone();
        next.as_mut_slice()[0] = stage as f32 + 1.0;
        next.as_mut_slice()[1] = if stage == 0 {
            action.0 as f32
        } else {
            branch as f32
        };
        if matches!(self.scenario, Scenario::NonFiniteSuccessor) && stage == 0 {
            next.as_mut_slice()[2] = f32::NAN;
        }
        let reward = if matches!(self.scenario, Scenario::NonFiniteReward) && stage == 0 {
            f32::NAN
        } else {
            reward
        };
        Ok((next, reward, done))
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
            let (next, reward, done) = self.step(&states[i], actions[i])?;
            next_states[i] = next;
            rewards[i] = reward;
            dones[i] = done;
        }
        Ok(())
    }
}

fn frame(actions: &'static [ActionId]) -> LocalActionFrame<'static> {
    LocalActionFrame::new(&ACTION_NAMES[..actions.len()], actions).expect("test frame fits")
}

fn blocked_gate(actions: &[ActionId]) -> PolicyGate {
    let mut gate = PolicyGate::default();
    for (index, &action) in actions.iter().enumerate() {
        gate.add_constraint(LinearConstraint::prohibit(
            RuleId(index as u32 + 1),
            "blocked by test",
            action,
        ));
    }
    gate
}

fn mcts_request<'a>(state: &'a FullLatent) -> DecideRequest<'a> {
    DecideRequest {
        active_context: Vec::new(),
        deadline: None,
        budget_ms: None,

        state,
        candidates: &ACTIONS,
        mode: DecideMode::Mcts,
        entropy: NormalizedEntropy::ZERO,
        return_trajectory: false,
        horizon: 4,
        causal_triad: None,
    }
}

#[test]
fn mcts_defaults_use_a_four_step_horizon_and_099_discount() {
    let engine = MctsEngine::default();
    assert_eq!(engine.horizon, 4);
    assert!((engine.discount - 0.99).abs() < f32::EPSILON);
}

#[test]
fn horizon_one_prefers_immediate_reward() {
    let model = TestModel::new(Scenario::Delayed);
    let (action, _) = MctsEngine {
        max_simulations: 32,
        horizon: 1,
        ..MctsEngine::default()
    }
    .plan(
        &FullLatent::zeros(),
        &frame(&ACTIONS),
        &model,
        &PolicyGate::default(),
    )
    .expect("one-step MCTS should produce an action");

    assert_eq!(action, ActionId(1));
    assert!(model.calls().iter().all(|call| call.stage == 0));
}

#[test]
fn default_horizon_uses_delayed_successor_value() {
    let model = TestModel::new(Scenario::Delayed);
    let (action, _) = MctsEngine::default()
        .plan(
            &FullLatent::zeros(),
            &frame(&ACTIONS),
            &model,
            &PolicyGate::default(),
        )
        .expect("finite delayed environment should plan");

    assert_eq!(action, ActionId(2));
    let calls = model.calls();
    assert!(
        calls.iter().any(|call| call.stage == 1),
        "default horizon must evaluate at least one real successor state"
    );
    assert!(
        calls
            .iter()
            .filter(|call| call.stage == 1)
            .all(|call| call.branch == 1 || call.branch == 2),
        "deeper transitions must retain the root branch in the successor state"
    );
}

#[test]
fn descendant_action_can_change_the_root_decision() {
    let model = TestModel::new(Scenario::DeepChoice);
    let (action, _) = MctsEngine {
        max_simulations: 96,
        horizon: 2,
        c_puct: 1.0,
        ..MctsEngine::default()
    }
    .plan(
        &FullLatent::zeros(),
        &frame(&ACTIONS),
        &model,
        &PolicyGate::default(),
    )
    .expect("deep-choice environment should plan");

    assert_eq!(
        action,
        ActionId(2),
        "the high-value descendant of B must outweigh A's immediate reward"
    );
    let calls = model.calls();
    assert!(calls
        .iter()
        .any(|call| { call.stage == 1 && call.branch == 2 && call.action == ActionId(2) }));
    assert!(calls
        .iter()
        .any(|call| { call.stage == 1 && call.branch == 2 && call.action == ActionId(1) }));
}

#[test]
fn rollout_respects_horizon_and_uses_successor_states() {
    let model = TestModel::new(Scenario::Horizon);
    MctsEngine {
        max_simulations: 12,
        horizon: 2,
        ..MctsEngine::default()
    }
    .plan(
        &FullLatent::zeros(),
        &frame(&ACTIONS),
        &model,
        &PolicyGate::default(),
    )
    .expect("finite horizon environment should plan");

    let calls = model.calls();
    assert!(calls.iter().any(|call| call.stage == 1));
    // Two root expansions, two initial rollout steps, four child expansions.
    // Later simulations traverse the cached grandchildren without replaying
    // either root or descendant transitions (flat root rollouts make 24 calls).
    assert_eq!(calls.iter().filter(|call| call.stage == 0).count(), 2);
    assert_eq!(calls.len(), 8, "persistent tree edges must be reused");
    for branch in [1, 2] {
        for action in ACTIONS {
            assert!(calls
                .iter()
                .any(|c| c.stage == 1 && c.branch == branch && c.action == action));
        }
    }
    assert!(
        calls.iter().all(|call| call.stage < 2),
        "horizon two must never step a third transition: {calls:?}"
    );
}

#[test]
fn terminal_successors_are_not_stepped_again() {
    let model = TestModel::new(Scenario::Terminal);
    let (action, _) = MctsEngine::default()
        .plan(
            &FullLatent::zeros(),
            &frame(&ACTIONS),
            &model,
            &PolicyGate::default(),
        )
        .expect("terminal environment should plan");

    assert_eq!(action, ActionId(1));
    assert!(!model
        .calls()
        .iter()
        .any(|call| call.branch == 1 && call.stage > 0));
}

#[test]
fn deeper_world_model_errors_abort_the_search() {
    let model = TestModel::new(Scenario::DeepError);
    let result = MctsEngine {
        max_simulations: 4,
        horizon: 2,
        ..MctsEngine::default()
    }
    .plan(
        &FullLatent::zeros(),
        &frame(&ACTIONS),
        &model,
        &PolicyGate::default(),
    );

    assert_eq!(
        result,
        Err(PlannerError::Core(CoreError::WorldModel(
            "deep transition failed".into(),
        )))
    );
    assert!(model.calls().iter().any(|call| call.stage == 1));
}

#[test]
fn nonfinite_inputs_and_transitions_fail_closed() {
    let mut invalid_state = FullLatent::zeros();
    invalid_state.as_mut_slice()[0] = f32::NAN;
    let input_result = MctsEngine::default().plan(
        &invalid_state,
        &frame(&ACTIONS),
        &TestModel::new(Scenario::Horizon),
        &PolicyGate::default(),
    );
    assert!(matches!(input_result, Err(PlannerError::DivergentState(_))));

    for scenario in [Scenario::NonFiniteSuccessor, Scenario::NonFiniteReward] {
        let result = MctsEngine {
            max_simulations: 2,
            horizon: 2,
            ..MctsEngine::default()
        }
        .plan(
            &FullLatent::zeros(),
            &frame(&ACTIONS),
            &TestModel::new(scenario),
            &PolicyGate::default(),
        );
        assert!(
            matches!(result, Err(PlannerError::DivergentState(_))),
            "{scenario:?}: {result:?}"
        );
    }
}

#[test]
fn zero_and_nonfinite_mcts_parameters_are_rejected() {
    let engines = [
        MctsEngine {
            max_simulations: 0,
            ..MctsEngine::default()
        },
        MctsEngine {
            horizon: 0,
            ..MctsEngine::default()
        },
        MctsEngine {
            discount: -0.1,
            ..MctsEngine::default()
        },
        MctsEngine {
            discount: 1.1,
            ..MctsEngine::default()
        },
        MctsEngine {
            discount: f32::NAN,
            ..MctsEngine::default()
        },
        MctsEngine {
            c_puct: -0.1,
            ..MctsEngine::default()
        },
        MctsEngine {
            c_puct: f32::NAN,
            ..MctsEngine::default()
        },
    ];
    for engine in engines {
        let result = engine.plan(
            &FullLatent::zeros(),
            &frame(&ACTIONS),
            &TestModel::new(Scenario::Horizon),
            &PolicyGate::default(),
        );
        assert!(matches!(result, Err(PlannerError::InvalidInput(_))));
    }
}

#[test]
fn arena_exhaustion_is_an_error() {
    let result = MctsEngine {
        max_simulations: 2,
        arena_capacity: 2,
        horizon: 2,
        ..MctsEngine::default()
    }
    .plan(
        &FullLatent::zeros(),
        &frame(&ACTIONS),
        &TestModel::new(Scenario::Horizon),
        &PolicyGate::default(),
    );

    assert_eq!(
        result,
        Err(PlannerError::ArenaCapacityExceeded { capacity: 2 })
    );
}

#[test]
fn blocked_actions_are_never_stepped_and_all_blocked_fails_closed() {
    let model = TestModel::new(Scenario::Delayed);
    let (action, _) = MctsEngine {
        max_simulations: 16,
        horizon: 2,
        ..MctsEngine::default()
    }
    .plan(
        &FullLatent::zeros(),
        &frame(&ACTIONS),
        &model,
        &blocked_gate(&[ActionId(2)]),
    )
    .expect("one allowed action should remain feasible");
    assert_eq!(action, ActionId(1));
    assert!(model.calls().iter().all(|call| call.action == ActionId(1)));

    let blocked = TestModel::new(Scenario::Horizon);
    let result = MctsEngine::default().plan(
        &FullLatent::zeros(),
        &frame(&ACTIONS),
        &blocked,
        &blocked_gate(&ACTIONS),
    );
    assert_eq!(result, Err(PlannerError::NoFeasibleAction));
    assert!(blocked.calls().is_empty());
}

#[test]
fn planner_config_wires_mcts_horizon_and_discount() {
    let defaults = PlannerConfig::default();
    assert_eq!(defaults.mcts_horizon, 4);
    assert!((defaults.mcts_discount - 0.99).abs() < f32::EPSILON);

    let state = FullLatent::zeros();
    let default_pipeline = ProductionPipeline::new_with_config(
        Arc::new(TestModel::new(Scenario::Delayed)),
        Arc::new(PolicyGate::default()),
        defaults,
    )
    .expect("default MCTS config should be accepted");
    assert_eq!(
        default_pipeline
            .decide(&mcts_request(&state))
            .unwrap()
            .action,
        ActionId(2)
    );

    let short_horizon = PlannerConfig {
        mcts_horizon: 1,
        mcts_discount: 0.5,
        ..PlannerConfig::default()
    };
    let short_pipeline = ProductionPipeline::new_with_config(
        Arc::new(TestModel::new(Scenario::Delayed)),
        Arc::new(PolicyGate::default()),
        short_horizon,
    )
    .expect("custom MCTS config should be accepted");
    assert_eq!(
        short_pipeline.decide(&mcts_request(&state)).unwrap().action,
        ActionId(1)
    );
}

#[test]
fn zero_discount_in_pipeline_removes_delayed_reward_preference() {
    let pipeline = ProductionPipeline::new_with_config(
        Arc::new(TestModel::new(Scenario::Delayed)),
        Arc::new(PolicyGate::default()),
        PlannerConfig {
            mcts_discount: 0.0,
            ..PlannerConfig::default()
        },
    )
    .unwrap();
    assert_eq!(
        pipeline
            .decide(&mcts_request(&FullLatent::zeros()))
            .unwrap()
            .action,
        ActionId(1)
    );
}
