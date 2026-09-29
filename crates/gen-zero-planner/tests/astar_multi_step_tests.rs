use gen_zero_core::{ActionId, CoreError, FullLatent, LocalActionFrame, WorldModelDynamics};
use gen_zero_gate::{LinearConstraint, PolicyGate, RuleId};
use gen_zero_planner::{AStarEngine, AStarGoal, PlannerError, PlanningEngine};
use std::sync::Mutex;

fn state(id: u32) -> FullLatent {
    let mut state = FullLatent::zeros();
    state.as_mut_slice()[0] = id as f32;
    state
}

// Graph maze: the tempting east move (0) leads to a closed pocket 1.
// Going north (1) bypasses the wall: 0 -> 2 -> 3 -> 4, cost 3.
// A second route 0 -> 5 -> 4 costs 6 despite using fewer edges.
// Unlisted actions hit a wall (self-loop). Goals are not model terminals.
struct Maze {
    calls: Mutex<Vec<(u32, u32)>>,
    edges: Vec<(u32, u32, u32, f32, bool)>,
    fail_at: Option<u32>,
}
impl Maze {
    fn new() -> Self {
        Self {
            calls: Mutex::new(vec![]),
            fail_at: None,
            edges: vec![
                (0, 0, 1, 5.0, false),
                (0, 1, 2, 0.0, false),
                (2, 1, 3, 0.0, false),
                (3, 0, 4, 0.0, false),
                (0, 2, 5, -4.0, false),
                (5, 0, 4, 0.0, false),
                (2, 0, 0, 0.0, false), // explicit cycle
            ],
        }
    }
}
impl WorldModelDynamics for Maze {
    type Error = CoreError;
    fn step(&self, s: &FullLatent, a: ActionId) -> Result<(FullLatent, f32, bool), CoreError> {
        let id = s.as_slice()[0] as u32;
        self.calls.lock().unwrap().push((id, a.0));
        if self.fail_at == Some(id) {
            // Real model error rather than a synthetic planner-side rejection.
            return Err(CoreError::WorldModel("deep transition failed".into()));
        }
        let edge = self
            .edges
            .iter()
            .find(|&&(from, action, _, _, _)| from == id && action == a.0);
        Ok(match edge {
            Some(&(_, _, to, reward, done)) => (state(to), reward, done),
            None => (s.clone(), 0.0, false),
        })
    }
    fn step_batch(
        &self,
        _: &[FullLatent],
        _: &[ActionId],
        _: &mut [FullLatent],
        _: &mut [f32],
        _: &mut [bool],
    ) -> Result<(), CoreError> {
        unreachable!("A* uses state-dependent single transitions")
    }
}
const ACTIONS: [ActionId; 3] = [ActionId(0), ActionId(1), ActionId(2)];
fn frame() -> LocalActionFrame<'static> {
    LocalActionFrame::new(&["east", "north", "detour"], &ACTIONS).unwrap()
}
fn engine() -> AStarEngine {
    AStarEngine {
        goal: Some(AStarGoal::WithinDistance {
            target: Box::new(state(4)),
            tolerance: 0.0,
        }),
        uncertainty_penalty_weight: 0.0,
        max_expansions: 20,
        ..Default::default()
    }
}

#[test]
fn three_step_obstacle_path_beats_one_step_ranker_and_expensive_shortcut() {
    let maze = Maze::new();
    let gate = PolicyGate::default();
    let frame = frame();
    let e = engine();
    // Reproduce the old ranker's actual cost: it chooses the dead-end action.
    let greedy = ACTIONS
        .iter()
        .copied()
        .min_by(|&a, &b| {
            let score = |a| {
                let (next, r, _) = maze.step(&state(0), a).unwrap();
                -r + e.uncertainty_penalty_weight * 0.05 * next.l2_norm() + 0.1
            };
            score(a).total_cmp(&score(b))
        })
        .unwrap();
    assert_eq!(greedy, ActionId(0));
    assert!(matches!(
        e.search(&state(1), &frame, &maze, &gate),
        Err(PlannerError::SearchGoalUnreachable)
    ));
    maze.calls.lock().unwrap().clear();
    let path = e.search(&state(0), &frame, &maze, &gate).unwrap();
    assert_eq!(path.actions, vec![ActionId(1), ActionId(1), ActionId(0)]);
    assert_eq!(path.cost, 3.0);
    assert_eq!(path.expanded, 4); // root, dead-end, north, beyond wall
    assert!(maze.calls.lock().unwrap().contains(&(3, 0)));
    let mut position = state(0);
    for &a in &path.actions {
        position = maze.step(&position, a).unwrap().0;
    }
    assert_eq!(position.as_slice()[0], 4.0);
    assert_eq!(
        e.plan(&state(0), &frame, &maze, &gate).unwrap().0,
        ActionId(1)
    );
}

#[test]
fn cheaper_duplicate_replaces_parent_and_stale_frontier_is_ignored() {
    let mut maze = Maze::new();
    maze.edges = vec![
        (0, 0, 2, -8.0, false),
        (0, 1, 1, 0.0, false),
        (1, 0, 2, 0.0, false),
        (2, 0, 4, -10.0, false),
    ];
    let path = engine()
        .search(&state(0), &frame(), &maze, &PolicyGate::default())
        .unwrap();
    assert_eq!(path.actions, vec![ActionId(1), ActionId(0), ActionId(0)]);
    assert_eq!(path.cost, 13.0);
    assert_eq!(
        maze.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|&&(id, _)| id == 2)
            .count(),
        3
    );
}

#[test]
fn cycles_and_budget_exhaustion_fail_closed() {
    let mut e = engine();
    e.max_expansions = 2;
    assert_eq!(
        e.plan(&state(0), &frame(), &Maze::new(), &PolicyGate::default()),
        Err(PlannerError::SearchBudgetExceeded { expanded: 2 })
    );
    e.max_expansions = 4; // goal popped exactly at budget is valid
    assert!(e
        .search(&state(0), &frame(), &Maze::new(), &PolicyGate::default())
        .is_ok());
    assert_eq!(
        e.plan(&state(1), &frame(), &Maze::new(), &PolicyGate::default()),
        Err(PlannerError::SearchGoalUnreachable)
    );
}

#[test]
fn terminal_failure_is_not_a_goal_or_expandable() {
    let mut maze = Maze::new();
    maze.edges = vec![(0, 0, 1, 0.0, true), (1, 0, 4, 0.0, false)];
    assert_eq!(
        engine().plan(&state(0), &frame(), &maze, &PolicyGate::default()),
        Err(PlannerError::SearchGoalUnreachable)
    );
    assert!(maze.calls.lock().unwrap().iter().all(|&(id, _)| id == 0));
}

#[test]
fn deep_model_errors_and_blocked_actions_do_not_become_fallbacks() {
    let mut maze = Maze::new();
    maze.fail_at = Some(2);
    assert!(matches!(
        engine().plan(&state(0), &frame(), &maze, &PolicyGate::default()),
        Err(PlannerError::Core(_))
    ));
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::prohibit(
        RuleId(1),
        "no north",
        ActionId(1),
    ));
    let maze = Maze::new();
    let path = engine().search(&state(0), &frame(), &maze, &gate).unwrap();
    assert_eq!(path.actions, vec![ActionId(2), ActionId(0)]);
    assert_eq!(path.cost, 6.0);
    assert!(maze.calls.lock().unwrap().iter().all(|&(_, a)| a != 1));
}

#[test]
fn explicit_goal_and_valid_parameters_are_required() {
    let maze = Maze::new();
    let gate = PolicyGate::default();
    assert_eq!(
        AStarEngine::default().plan(&state(0), &frame(), &maze, &gate),
        Err(PlannerError::MissingSearchGoal)
    );
    assert!(maze.calls.lock().unwrap().is_empty());
    let mut e = engine();
    e.goal = Some(AStarGoal::Predicate(|s| s.as_slice()[0] == 4.0));
    assert_eq!(
        e.search(&state(4), &frame(), &maze, &gate)
            .unwrap()
            .actions
            .len(),
        0
    );
    assert_eq!(
        e.plan(&state(4), &frame(), &maze, &gate),
        Err(PlannerError::SearchAlreadyAtGoal)
    );
    for tolerance in [-1.0, f32::NAN, f32::INFINITY] {
        e.goal = Some(AStarGoal::WithinDistance {
            target: Box::new(state(4)),
            tolerance,
        });
        assert!(matches!(
            e.plan(&state(0), &frame(), &maze, &gate),
            Err(PlannerError::InvalidInput(_))
        ));
    }
}

#[test]
fn goal_is_accepted_on_pop_not_when_first_generated() {
    let mut maze = Maze::new();
    maze.edges = vec![
        (0, 0, 4, -9.0, false),
        (0, 1, 2, 0.0, false),
        (2, 1, 3, 0.0, false),
        (3, 0, 4, 0.0, true),
    ];
    let path = engine()
        .search(&state(0), &frame(), &maze, &PolicyGate::default())
        .unwrap();
    assert_eq!(path.cost, 3.0);
    assert_eq!(path.actions, vec![ActionId(1), ActionId(1), ActionId(0)]);
}

#[test]
fn terminal_and_nonterminal_arrivals_are_not_merged() {
    let mut maze = Maze::new();
    maze.edges = vec![
        (0, 0, 2, 0.0, true),
        (0, 1, 2, -1.0, false),
        (2, 0, 4, 0.0, false),
    ];
    let path = engine()
        .search(&state(0), &frame(), &maze, &PolicyGate::default())
        .unwrap();
    assert_eq!(path.actions, vec![ActionId(1), ActionId(0)]);
    assert_eq!(path.cost, 3.0);
}

#[test]
fn distance_threshold_and_invalid_cost_settings_are_checked() {
    let mut e = engine();
    let mut target = state(4);
    target.as_mut_slice()[0] += 0.25;
    e.goal = Some(AStarGoal::WithinDistance {
        target: Box::new(target),
        tolerance: 0.25,
    });
    assert_eq!(
        e.search(&state(0), &frame(), &Maze::new(), &PolicyGate::default())
            .unwrap()
            .cost,
        3.0
    );
    e.uncertainty_penalty_weight = -1.0;
    assert!(matches!(
        e.search(&state(0), &frame(), &Maze::new(), &PolicyGate::default()),
        Err(PlannerError::InvalidInput(_))
    ));
    e.uncertainty_penalty_weight = 0.0;
    e.max_expansions = 0;
    assert!(matches!(
        e.search(&state(0), &frame(), &Maze::new(), &PolicyGate::default()),
        Err(PlannerError::InvalidInput(_))
    ));
}

#[test]
fn pipeline_does_not_silently_restore_the_old_ranker_without_a_goal() {
    use gen_zero_core::NormalizedEntropy;
    use gen_zero_planner::{DecideMode, DecideRequest, ProductionPipeline};
    use std::sync::Arc;
    let pipeline = ProductionPipeline::new(Arc::new(Maze::new()), Arc::new(PolicyGate::default()));
    let s = state(0);
    let request = DecideRequest {
        active_context: Vec::new(),
        deadline: None,
        budget_ms: None,

        state: &s,
        candidates: &ACTIONS,
        mode: DecideMode::AStar,
        entropy: NormalizedEntropy::ZERO,
        return_trajectory: false,
        horizon: 3,
    };
    assert!(matches!(
        pipeline.decide(&request),
        Err(PlannerError::MissingSearchGoal)
    ));
    let configured = pipeline.with_astar_goal(AStarGoal::Predicate(|s| s.as_slice()[0] == 4.0));
    assert_eq!(configured.decide(&request).unwrap().action, ActionId(1));
}
