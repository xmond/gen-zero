//! Integration coverage for the public game-theory, flow-sampling, and formal
//! solve APIs.
//!
//! These tests intentionally exercise distributions and solver status values,
//! rather than only checking the action selected by a single call.  A
//! deterministic arg-max implementation can therefore not satisfy them.

use gen_zero_core::{ActionId, CoreError, FullLatent, LocalActionFrame, WorldModelDynamics};
use gen_zero_gate::{LinearConstraint, PolicyGate, RuleId};
use gen_zero_planner::engine::{
    CfrNashEngine, CpSatFormalEngine, FormalSolveOptions, ManifoldGFlowNetEngine, NormalFormGame,
    SolveStatus,
};
use gen_zero_planner::PlannerError;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const ACTION_NAMES: [&str; 3] = ["a", "b", "c"];
const ACTIONS: [ActionId; 3] = [ActionId(0), ActionId(1), ActionId(2)];

fn frame(actions: &[ActionId]) -> LocalActionFrame<'static> {
    LocalActionFrame::new(&ACTION_NAMES[..actions.len()], actions)
        .expect("test action frame should fit")
}

fn forbid(gate: &mut PolicyGate, action: ActionId, rule_id: u32) {
    gate.add_constraint(LinearConstraint::prohibit(
        RuleId(rule_id),
        "blocked by integration test",
        action,
    ));
}

fn assert_probability_vector(probabilities: &[f64]) {
    assert!(!probabilities.is_empty());
    assert!(probabilities.iter().all(|p| p.is_finite() && *p >= 0.0));
    let total: f64 = probabilities.iter().sum();
    assert!((total - 1.0).abs() < 1e-10, "probabilities sum to {total}");
}

fn assert_normalized_strategy(strategy: &[f64]) {
    assert!(!strategy.is_empty());
    assert!(strategy.iter().all(|p| p.is_finite() && *p >= 0.0));
    let total: f64 = strategy.iter().sum();
    assert!((total - 1.0).abs() < 1e-8, "strategy sums to {total}");
}

fn rps_game() -> NormalFormGame {
    NormalFormGame::new(
        vec![
            vec![0.0, -1.0, 1.0],
            vec![1.0, 0.0, -1.0],
            vec![-1.0, 1.0, 0.0],
        ],
        vec![
            vec![0.0, 1.0, -1.0],
            vec![-1.0, 0.0, 1.0],
            vec![1.0, -1.0, 0.0],
        ],
    )
    .expect("RPS is a valid zero-sum game")
}

#[test]
fn cfr_converges_to_rps_and_samples_each_action() {
    let solution = CfrNashEngine
        .solve_game(&rps_game(), 20_000)
        .expect("RPS should solve");

    assert_eq!(solution.iterations, 20_000);
    assert_normalized_strategy(&solution.row_strategy);
    assert_normalized_strategy(&solution.column_strategy);
    for probability in solution
        .row_strategy
        .iter()
        .chain(solution.column_strategy.iter())
    {
        assert!(
            (probability - 1.0 / 3.0).abs() < 1e-8,
            "probability {probability}"
        );
    }
    assert!(
        solution
            .zero_sum_gap
            .expect("zero-sum games report a gap")
            .abs()
            < 1e-8
    );

    // The same seed must produce the same mixed-strategy draw sequence, and
    // the sequence must actually visit every action.
    let mut first_rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut second_rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut row_counts = [0usize; 3];
    for _ in 0..6_000 {
        let row = solution.sample_row(&mut first_rng);
        let repeated = solution.sample_row(&mut second_rng);
        assert_eq!(row, repeated, "same seed should reproduce row draws");
        assert!(row < 3);
        row_counts[row] += 1;
    }
    assert!(row_counts
        .iter()
        .all(|&count| (1_850..=2_150).contains(&count)));
    let mut column_rng = StdRng::seed_from_u64(0xBAD5EED);
    let mut column_counts = [0usize; 3];
    for _ in 0..6_000 {
        let column = solution.sample_column(&mut column_rng);
        assert!(column < 3);
        column_counts[column] += 1;
    }
    assert!(column_counts
        .iter()
        .all(|&count| (1_850..=2_150).contains(&count)));
    eprintln!(
        "RPS row={:?} column={:?} zero_sum_gap={:?} row_samples={row_counts:?} column_samples={column_counts:?}",
        solution.row_strategy, solution.column_strategy, solution.zero_sum_gap
    );
}

#[test]
fn cfr_converges_on_an_asymmetric_zero_sum_game() {
    // The unique mixed equilibrium is (0.4, 0.6) for both players.  A solver
    // that merely returns its uniform initialization stays at (0.5, 0.5).
    let game = NormalFormGame::new(
        vec![vec![2.0, -1.0], vec![-1.0, 1.0]],
        vec![vec![-2.0, 1.0], vec![1.0, -1.0]],
    )
    .unwrap();
    let one_round_gap = CfrNashEngine
        .solve_game(&game, 1)
        .unwrap()
        .zero_sum_gap
        .unwrap();
    let solution = CfrNashEngine.solve_game(&game, 40_000).unwrap();

    assert_normalized_strategy(&solution.row_strategy);
    assert_normalized_strategy(&solution.column_strategy);
    for strategy in [&solution.row_strategy, &solution.column_strategy] {
        assert!((strategy[0] - 0.4).abs() < 0.02, "{strategy:?}");
        assert!((strategy[1] - 0.6).abs() < 0.02, "{strategy:?}");
    }
    let gap = solution.zero_sum_gap.unwrap();
    assert!(gap < 0.03, "converged gap {gap}");
    assert!(
        gap < one_round_gap * 0.1,
        "one-round gap {one_round_gap}, converged gap {gap}"
    );
    eprintln!(
        "asymmetric row={:?} column={:?} one_round_gap={one_round_gap} zero_sum_gap={gap}",
        solution.row_strategy, solution.column_strategy
    );
}

#[test]
fn cfr_marks_general_sum_games_and_rejects_malformed_games() {
    let general_sum = NormalFormGame::new(
        vec![vec![3.0, 0.0], vec![0.0, 2.0]],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();
    let solution = CfrNashEngine.solve_game(&general_sum, 200).unwrap();
    assert_eq!(solution.zero_sum_gap, None);
    assert_normalized_strategy(&solution.row_strategy);
    assert_normalized_strategy(&solution.column_strategy);

    assert!(NormalFormGame::new(Vec::new(), Vec::new()).is_err());
    assert!(NormalFormGame::new(vec![vec![1.0, 2.0]], vec![vec![1.0]]).is_err());
    assert!(NormalFormGame::new(
        vec![vec![1.0, 2.0], vec![3.0]],
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
    )
    .is_err());
    assert!(
        NormalFormGame::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![vec![0.0, 0.0]],).is_err()
    );
    assert!(NormalFormGame::new(vec![vec![f64::NAN]], vec![vec![0.0]],).is_err());
    assert!(NormalFormGame::new(vec![vec![f64::INFINITY]], vec![vec![0.0]],).is_err());
    assert!(CfrNashEngine.solve_game(&general_sum, 0).is_err());
}

#[test]
fn gflow_preserves_direct_flow_proportions_and_excludes_blocked_actions() {
    let engine = ManifoldGFlowNetEngine;
    let all_actions = frame(&ACTIONS);
    let open_gate = PolicyGate::default();
    let distribution = engine
        .flow_distribution(&all_actions, &[1.0, 3.0, 6.0], &open_gate)
        .unwrap();

    assert_eq!(distribution.actions.as_slice(), ACTIONS.as_slice());
    assert_probability_vector(&distribution.probabilities);
    for (actual, expected) in distribution.probabilities.iter().zip([0.1, 0.3, 0.6]) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "{actual} versus {expected}"
        );
    }

    let mut rng = StdRng::seed_from_u64(0x1234_5678);
    let mut counts = [0usize; 3];
    for _ in 0..20_000 {
        let action = distribution.sample(&mut rng);
        assert!(action.0 < 3);
        counts[action.0 as usize] += 1;
    }
    for (&count, expected) in counts.iter().zip([0.1, 0.3, 0.6]) {
        let observed = count as f64 / 20_000.0;
        assert!((observed - expected).abs() < 0.03, "{counts:?}");
    }

    let zero_mass = engine
        .flow_distribution(&all_actions, &[0.0, 1.0, 2.0], &open_gate)
        .unwrap();
    assert_eq!(zero_mass.probabilities[0], 0.0);
    let mut zero_mass_rng = StdRng::seed_from_u64(0x5EE0);
    for _ in 0..5_000 {
        assert_ne!(zero_mass.sample(&mut zero_mass_rng), ActionId(0));
    }

    let mut blocked_gate = PolicyGate::default();
    forbid(&mut blocked_gate, ActionId(1), 21);
    // The blocked flow is deliberately much larger.  It must not be allowed
    // to steal probability mass before normalization.
    let filtered = engine
        .flow_distribution(&all_actions, &[1.0, 100.0, 6.0], &blocked_gate)
        .unwrap();
    assert_eq!(filtered.actions, vec![ActionId(0), ActionId(2)]);
    assert_probability_vector(&filtered.probabilities);
    assert!((filtered.probabilities[0] - 1.0 / 7.0).abs() < 1e-12);
    assert!((filtered.probabilities[1] - 6.0 / 7.0).abs() < 1e-12);

    let mut rng = StdRng::seed_from_u64(0xB10C_BE01);
    for _ in 0..20_000 {
        assert_ne!(filtered.sample(&mut rng), ActionId(1));
    }
}

#[test]
fn gflow_rejects_invalid_flows_and_duplicate_actions() {
    let engine = ManifoldGFlowNetEngine;
    let actions = frame(&ACTIONS);
    let gate = PolicyGate::default();

    for flows in [
        vec![0.0, 0.0, 0.0],
        vec![1.0, -1.0, 1.0],
        vec![1.0, f64::NAN, 1.0],
        vec![1.0, f64::INFINITY, 1.0],
        vec![f64::MIN_POSITIVE, f64::MAX, 1.0],
    ] {
        assert!(
            engine.flow_distribution(&actions, &flows, &gate).is_err(),
            "invalid flows {flows:?} were accepted"
        );
    }
    assert!(engine
        .flow_distribution(&actions, &[1.0, 2.0], &gate)
        .is_err());

    let duplicate_ids = [ActionId(0), ActionId(0), ActionId(2)];
    let duplicate_frame = frame(&duplicate_ids);
    assert!(engine
        .flow_distribution(&duplicate_frame, &[1.0, 2.0, 3.0], &gate)
        .is_err());

    let mut all_blocked = PolicyGate::default();
    for (i, &action) in ACTIONS.iter().enumerate() {
        forbid(&mut all_blocked, action, 30 + i as u32);
    }
    assert!(engine
        .flow_distribution(&actions, &[1.0, 2.0, 3.0], &all_blocked)
        .is_err());
}

#[test]
fn gflow_plan_samples_exp_rewards_without_a_distance_penalty() {
    let state = FullLatent::zeros();
    let actions = frame(&ACTIONS);
    let gate = PolicyGate::default();
    let rewards = [1.0_f32.ln(), 3.0_f32.ln(), 6.0_f32.ln()];
    // Make the first action's successor very far away.  Its probability must
    // still be proportional to exp(reward), because distance is not a flow
    // term in this API.
    let model = RewardModel::new(rewards.to_vec(), vec![100.0, 0.0, 0.0]);

    let mut rng = StdRng::seed_from_u64(0xFACE_FEED);
    let mut counts = [0usize; 3];
    for _ in 0..20_000 {
        let (action, entropy) = ManifoldGFlowNetEngine
            .plan_with_rng(&state, &actions, &model, &gate, &mut rng)
            .unwrap();
        assert!(entropy.value().is_finite());
        assert!((0.0..=1.0).contains(&entropy.value()));
        counts[action.0 as usize] += 1;
    }
    for (&count, expected) in counts.iter().zip([0.1, 0.3, 0.6]) {
        let observed = count as f64 / 20_000.0;
        assert!((observed - expected).abs() < 0.03, "{counts:?}");
    }
    eprintln!("GFlow plan exp-reward samples={counts:?}");

    let invalid_successor = RewardModel::new(vec![1.0, 2.0, 3.0], vec![f32::INFINITY, 0.0, 0.0]);
    let mut invalid_rng = StdRng::seed_from_u64(0xBAD);
    let error = ManifoldGFlowNetEngine.plan_with_rng(
        &state,
        &actions,
        &invalid_successor,
        &gate,
        &mut invalid_rng,
    );
    assert!(matches!(error, Err(PlannerError::DivergentState(_))));
}

fn solve_options(max_evaluations: usize, stop_after_first: bool) -> FormalSolveOptions {
    FormalSolveOptions {
        max_evaluations,
        time_limit: None,
        stop_after_first,
    }
}

#[test]
fn cpsat_reports_optimal_feasible_timeout_and_infeasible_statuses() {
    let state = FullLatent::zeros();
    let actions = frame(&ACTIONS);
    let model = RewardModel::new(vec![1.0, 3.0, 2.0], vec![0.0, 0.0, 0.0]);
    let gate = PolicyGate::default();

    let optimal = CpSatFormalEngine
        .solve(
            &state,
            &actions,
            &model,
            &gate,
            &FormalSolveOptions::default(),
        )
        .unwrap();
    assert_eq!(optimal.status, SolveStatus::Optimal);
    assert_eq!(optimal.action, Some(ActionId(1)));
    assert_eq!(optimal.objective, Some(3.0));
    assert_eq!(optimal.evaluated, 3);

    let early = CpSatFormalEngine
        .solve(
            &state,
            &actions,
            &model,
            &gate,
            &solve_options(usize::MAX, true),
        )
        .unwrap();
    assert_eq!(early.status, SolveStatus::Feasible);
    assert_eq!(early.action, Some(ActionId(0)));
    assert_eq!(early.objective, Some(1.0));
    assert_eq!(early.evaluated, 1);

    let partial = CpSatFormalEngine
        .solve(&state, &actions, &model, &gate, &solve_options(2, false))
        .unwrap();
    assert_eq!(partial.status, SolveStatus::Timeout);
    assert_eq!(partial.action, Some(ActionId(1)));
    assert_eq!(partial.objective, Some(3.0));
    assert_eq!(partial.evaluated, 2);

    let zero_budget = CpSatFormalEngine
        .solve(&state, &actions, &model, &gate, &solve_options(0, false))
        .unwrap();
    assert_eq!(zero_budget.status, SolveStatus::Timeout);
    assert_eq!(zero_budget.action, None);
    assert_eq!(zero_budget.objective, None);
    assert_eq!(zero_budget.evaluated, 0);

    let mut all_blocked = PolicyGate::default();
    for (i, &action) in ACTIONS.iter().enumerate() {
        forbid(&mut all_blocked, action, 50 + i as u32);
    }
    let infeasible = CpSatFormalEngine
        .solve(
            &state,
            &actions,
            &model,
            &all_blocked,
            &FormalSolveOptions::default(),
        )
        .unwrap();
    assert_eq!(infeasible.status, SolveStatus::Infeasible);
    assert_eq!(infeasible.action, None);
    assert_eq!(infeasible.objective, None);
}

#[test]
fn cpsat_zero_deadline_is_timeout_and_model_errors_are_not_hidden() {
    let state = FullLatent::zeros();
    let actions = frame(&ACTIONS);
    let gate = PolicyGate::default();
    let model = RewardModel::new(vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.0]);

    let timeout = CpSatFormalEngine
        .solve(
            &state,
            &actions,
            &model,
            &gate,
            &FormalSolveOptions {
                max_evaluations: usize::MAX,
                time_limit: Some(Duration::ZERO),
                stop_after_first: false,
            },
        )
        .unwrap();
    assert_eq!(timeout.status, SolveStatus::Timeout);
    assert_eq!(timeout.evaluated, 0);
    assert_eq!(timeout.action, None);
    assert_eq!(timeout.objective, None);
    assert!(model.calls.lock().unwrap().is_empty());

    let failing = RewardModel::with_failure(vec![1.0, 2.0, 3.0], ActionId(0));
    let error = CpSatFormalEngine.solve(
        &state,
        &actions,
        &failing,
        &gate,
        &FormalSolveOptions::default(),
    );
    assert!(matches!(
        error,
        Err(PlannerError::Core(CoreError::WorldModel(_)))
    ));

    let nonfinite = RewardModel::new(vec![f32::NAN, 2.0, 3.0], vec![0.0, 0.0, 0.0]);
    let error = CpSatFormalEngine.solve(
        &state,
        &actions,
        &nonfinite,
        &gate,
        &FormalSolveOptions::default(),
    );
    assert!(matches!(error, Err(PlannerError::DivergentState(_))));

    let invalid_successor = RewardModel::new(vec![1.0, 2.0, 3.0], vec![f32::INFINITY, 0.0, 0.0]);
    let error = CpSatFormalEngine.solve(
        &state,
        &actions,
        &invalid_successor,
        &gate,
        &FormalSolveOptions::default(),
    );
    assert!(matches!(error, Err(PlannerError::DivergentState(_))));
}

#[derive(Clone)]
struct RewardModel {
    rewards: Vec<f32>,
    displacements: Vec<f32>,
    failure: Option<ActionId>,
    calls: Arc<Mutex<Vec<ActionId>>>,
}

impl RewardModel {
    fn new(rewards: Vec<f32>, displacements: Vec<f32>) -> Self {
        Self {
            rewards,
            displacements,
            failure: None,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn with_failure(rewards: Vec<f32>, action: ActionId) -> Self {
        Self {
            failure: Some(action),
            ..Self::new(rewards, vec![0.0; 3])
        }
    }
}

impl WorldModelDynamics for RewardModel {
    type Error = CoreError;

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), Self::Error> {
        if self.failure == Some(action) {
            return Err(CoreError::WorldModel(format!(
                "integration model failed for action {}",
                action.0
            )));
        }
        let index = action.0 as usize;
        let reward = self
            .rewards
            .get(index)
            .copied()
            .ok_or_else(|| CoreError::WorldModel(format!("unknown action {}", action.0)))?;
        let displacement = self.displacements.get(index).copied().unwrap_or(0.0);
        let mut next = state.clone();
        next.as_mut_slice()[0] += displacement;
        self.calls.lock().unwrap().push(action);
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
