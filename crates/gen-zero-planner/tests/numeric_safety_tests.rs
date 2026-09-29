use gen_zero_core::{ActionId, CoreError, FullLatent, LocalActionFrame, WorldModelDynamics};
use gen_zero_gate::PolicyGate;
use gen_zero_planner::{
    AStarEngine, CfrNashEngine, CpSatFormalEngine, ManifoldGFlowNetEngine, MctsEngine,
    MpcCemEngine, PlannerError, PlanningEngine,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct NumericModel {
    coordinate: usize,
    state_value: f32,
    reward: f32,
    fail_at: usize,
    done: bool,
    calls: AtomicUsize,
}

impl WorldModelDynamics for NumericModel {
    type Error = CoreError;

    fn step(&self, state: &FullLatent, _: ActionId) -> Result<(FullLatent, f32, bool), CoreError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let mut next = state.clone();
        if call >= self.fail_at {
            next.as_mut_slice()[self.coordinate] = self.state_value;
            Ok((next, self.reward, self.done))
        } else {
            Ok((next, 1.0, false))
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
        panic!("engines must use the tested single-step path")
    }
}

fn engines() -> [Box<dyn PlanningEngine>; 6] {
    [
        Box::new(MctsEngine::default()),
        Box::new(AStarEngine {
            goal: Some(gen_zero_planner::AStarGoal::Predicate(|_| false)),
            ..AStarEngine::default()
        }),
        Box::new(MpcCemEngine::default()),
        Box::new(ManifoldGFlowNetEngine),
        Box::new(CfrNashEngine),
        Box::new(CpSatFormalEngine),
    ]
}

fn rejects(state_value: f32, reward: f32, coordinates: &[usize]) {
    let state = FullLatent::zeros();
    let actions = LocalActionFrame::new(&["a", "b"], &[ActionId(0), ActionId(1)]).unwrap();
    for engine in engines() {
        for &coordinate in coordinates {
            // The second CEM call exercises its separate horizon transition path.
            for fail_at in [1, 2] {
                for done in [false, true] {
                    let model = NumericModel {
                        coordinate,
                        state_value,
                        reward,
                        fail_at,
                        done,
                        calls: AtomicUsize::new(0),
                    };
                    let result = engine.plan(&state, &actions, &model, &PolicyGate::default());
                    assert!(matches!(result, Err(PlannerError::DivergentState(_))),
                        "{} coordinate={coordinate} state={state_value} reward={reward} fail_at={fail_at} done={done}: {result:?}", engine.name());
                    assert_eq!(
                        model.calls.load(Ordering::SeqCst),
                        fail_at,
                        "{} must stop at the first invalid transition",
                        engine.name()
                    );
                }
            }
        }
    }
}

#[test]
fn all_engines_reject_combined_nan_state_and_reward() {
    rejects(f32::NAN, f32::NAN, &[0]);
}

#[test]
fn all_engines_reject_each_nonfinite_successor_coordinate_with_finite_reward() {
    let coordinates: Vec<_> = (0..FullLatent::zeros().as_slice().len()).collect();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        rejects(value, 1.0, &coordinates);
    }
}

#[test]
fn all_engines_reject_nonfinite_reward_with_finite_successor() {
    for reward in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        rejects(0.0, reward, &[0]);
    }
}

#[test]
fn mcts_wide_accumulation_keeps_finite_f32_max_rewards_valid() {
    let model = NumericModel {
        coordinate: 0,
        state_value: 0.0,
        reward: f32::MAX,
        fail_at: 1,
        done: false,
        calls: AtomicUsize::new(0),
    };
    let actions = LocalActionFrame::new(&["a"], &[ActionId(0)]).unwrap();
    let result = MctsEngine::default().plan(
        &FullLatent::zeros(),
        &actions,
        &model,
        &PolicyGate::default(),
    );
    let (action, entropy) = result.expect("f64 return accumulation remains finite");
    assert_eq!(action, ActionId(0));
    assert!(entropy.0.is_finite());
    assert!(model.calls.load(Ordering::SeqCst) > 2);
}
