//! Regression tests for state-dependent dynamics dead ends.
//!
//! The fixture is a small deterministic model used only to make the planner's
//! ranking observable: action 1 enters a masked state with no legal successor,
//! while action 2 continues through a negative-reward live path. A zero-cost
//! dead end would incorrectly outrank the live path.

use gen_zero_core::{ActionId, CoreError, FullLatent, LocalActionFrame, WorldModelDynamics};
use gen_zero_gate::PolicyGate;
use gen_zero_planner::engine::{SearchBudget, DEAD_END_PENALTY};
use gen_zero_planner::{CemPlan, MctsEngine, MpcCemEngine, PlanningEngine};
use std::sync::atomic::{AtomicBool, Ordering};

const ACTIONS: [ActionId; 2] = [ActionId(1), ActionId(2)];
const NAMES: [&str; 2] = ["dead_end", "live"];

/// A deterministic mask fixture, not a production model or a claim of learned
/// dynamics. The root has two choices; only action 2 has a legal continuation.
struct NegativeRewardDeadEnd(f32);

impl WorldModelDynamics for NegativeRewardDeadEnd {
    type Error = CoreError;

    fn allowed_actions(
        &self,
        state: &FullLatent,
        candidates: &[ActionId],
    ) -> Result<Vec<ActionId>, Self::Error> {
        let stage = state.as_slice()[0] as usize;
        let branch = state.as_slice()[1] as u32;
        match (stage, branch) {
            (0, _) => Ok(candidates.to_vec()),
            (_, 1) => Ok(Vec::new()),
            (_, 2) => Ok(candidates
                .iter()
                .copied()
                .filter(|action| action.0 == 2)
                .collect()),
            _ => Ok(Vec::new()),
        }
    }

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), Self::Error> {
        let stage = state.as_slice()[0] as usize;
        let mut next = state.clone();
        next.as_mut_slice()[0] = stage as f32 + 1.0;
        if stage == 0 {
            next.as_mut_slice()[1] = action.0 as f32;
        }
        // The live branch is deliberately negative: the old zero-cost dead-end
        // behavior made action 1 appear better than this valid path.
        let reward = if stage == 0 { 0.0 } else { self.0 };
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
            let (next, reward, done) = self.step(&states[i], actions[i])?;
            next_states[i] = next;
            rewards[i] = reward;
            dones[i] = done;
        }
        Ok(())
    }
}

/// Every root action reaches a masked state. This isolates the numeric penalty
/// from branch ranking so the returned plan score can be checked directly.
struct AlwaysDeadAfterRoot;

impl WorldModelDynamics for AlwaysDeadAfterRoot {
    type Error = CoreError;

    fn allowed_actions(
        &self,
        state: &FullLatent,
        candidates: &[ActionId],
    ) -> Result<Vec<ActionId>, Self::Error> {
        if state.as_slice()[0] == 0.0 {
            Ok(candidates.to_vec())
        } else {
            Ok(Vec::new())
        }
    }

    fn step(
        &self,
        state: &FullLatent,
        _action: ActionId,
    ) -> Result<(FullLatent, f32, bool), Self::Error> {
        let mut next = state.clone();
        next.as_mut_slice()[0] = 1.0;
        Ok((next, 0.0, false))
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

fn frame() -> LocalActionFrame<'static> {
    LocalActionFrame::new(&NAMES, &ACTIONS).expect("test frame fits")
}

#[test]
fn mcts_penalizes_deep_dead_end_and_reports_marker() {
    let marker = AtomicBool::new(false);
    let budget = SearchBudget::new(None).with_dead_end_marker(&marker);
    let (action, _) = MctsEngine {
        max_simulations: 32,
        horizon: 2,
        ..MctsEngine::default()
    }
    .plan_until(
        &FullLatent::zeros(),
        &frame(),
        &NegativeRewardDeadEnd(-1.0),
        &PolicyGate::default(),
        &budget,
    )
    .expect("live branch remains feasible");

    assert_eq!(action, ActionId(2));
    assert!(marker.load(Ordering::Acquire));
    assert!(DEAD_END_PENALTY.is_finite());
    assert!(DEAD_END_PENALTY < 0.0);
}

#[test]
fn cem_penalizes_deep_dead_end_and_exposes_plan_metadata() {
    let engine = MpcCemEngine {
        num_samples: 16,
        num_elites: 1,
        horizon: 2,
        num_iterations: 1,
        gamma: 1.0,
    };
    let result: CemPlan = engine
        .optimize(
            &FullLatent::zeros(),
            &frame(),
            &NegativeRewardDeadEnd(-1.0),
            &PolicyGate::default(),
        )
        .expect("live branch remains feasible");

    assert_eq!(result.actions.first().copied(), Some(ActionId(2)));
    assert!(result.has_dead_end);
    assert!(result.score.is_finite());
    assert!(result.score < 0.0);

    let marker = AtomicBool::new(false);
    let budget = SearchBudget::new(None).with_dead_end_marker(&marker);
    let (action, _) = engine
        .plan_until(
            &FullLatent::zeros(),
            &frame(),
            &NegativeRewardDeadEnd(-1.0),
            &PolicyGate::default(),
            &budget,
        )
        .expect("live branch remains feasible");
    assert_eq!(action, ActionId(2));
    assert!(marker.load(Ordering::Acquire));
}

#[test]
fn cem_score_contains_finite_penalty_when_every_candidate_dead_ends() {
    let engine = MpcCemEngine {
        num_samples: 8,
        num_elites: 1,
        horizon: 2,
        num_iterations: 1,
        gamma: 1.0,
    };
    let result = engine
        .optimize(
            &FullLatent::zeros(),
            &frame(),
            &AlwaysDeadAfterRoot,
            &PolicyGate::default(),
        )
        .expect("root actions are feasible before the dead end");

    assert!(result.has_dead_end);
    assert_eq!(result.score, DEAD_END_PENALTY);
    assert!(result.score.is_finite());
}

#[test]
fn direct_rust_tuple_api_refuses_to_hide_dead_end_metadata() {
    let engines: Vec<Box<dyn PlanningEngine>> = vec![
        Box::new(MctsEngine {
            horizon: 2,
            ..MctsEngine::default()
        }),
        Box::new(MpcCemEngine {
            horizon: 2,
            ..MpcCemEngine::default()
        }),
    ];
    for engine in engines {
        let state = FullLatent::zeros();
        let gate = PolicyGate::default();
        assert_eq!(
            engine.plan(&state, &frame(), &AlwaysDeadAfterRoot, &gate),
            Err(gen_zero_planner::PlannerError::DeadEndRequiresReport)
        );
        let report = engine
            .plan_report(&state, &frame(), &AlwaysDeadAfterRoot, &gate)
            .unwrap();
        assert!(report.has_dead_end);
    }
}

#[test]
fn mcts_revisited_cached_dead_end_keeps_its_penalty() {
    // A comparable live cost and large exploration term force repeated visits
    // to both branches. Clearing the cached dead-end cost reverses this choice.
    let engine = MctsEngine {
        max_simulations: 128,
        horizon: 2,
        discount: 1.0,
        c_puct: 1.0e30,
        ..MctsEngine::default()
    };
    let report = engine
        .plan_report(
            &FullLatent::zeros(),
            &frame(),
            &NegativeRewardDeadEnd(-9.0e29),
            &PolicyGate::default(),
        )
        .unwrap();
    assert!(report.has_dead_end);
    assert_eq!(report.action, ActionId(2));
}
