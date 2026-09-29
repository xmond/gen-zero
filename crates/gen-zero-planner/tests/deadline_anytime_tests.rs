//! These are wall-clock regression tests, not a real-time scheduling proof.
//! A <3ms bound deliberately distinguishes a 2ms caller timeout from waiting
//! for the requested 3ms synchronous step. Scheduling overruns fail visibly.
use gen_zero_core::{ActionId, CoreError, FullLatent, NormalizedEntropy, WorldModelDynamics};
use gen_zero_gate::{LinearConstraint, PolicyGate, PolicyTier, RuleId};
use gen_zero_planner::{
    AStarGoal, DecideMode, DecideRequest, PlannerConfig, PlannerError, ProductionPipeline,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};
use std::time::{Duration, Instant};

static TIMING: Mutex<()> = Mutex::new(());
const MODES: [DecideMode; 3] = [DecideMode::Mcts, DecideMode::MpcCem, DecideMode::AStar];
const ACTIONS: [ActionId; 2] = [ActionId(1), ActionId(2)];

struct SlowModel {
    calls: AtomicUsize,
    /// Calls before this prefix are immediate; the next call is a real
    /// synchronous model stall.  A prefix, instead of a single fast call,
    /// accounts for candidate screening and complete engine work separately.
    fast_calls: usize,
    sleep: Duration,
}
impl WorldModelDynamics for SlowModel {
    type Error = CoreError;
    fn step(&self, state: &FullLatent, _: ActionId) -> Result<(FullLatent, f32, bool), CoreError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call >= self.fast_calls {
            std::thread::sleep(self.sleep);
        }
        // One deterministic state transition gives the pipeline's configured
        // A* predicate a real, complete one-step goal path.  The transition is
        // always nonterminal, so the screen cannot certify an engine result.
        let mut next = state.clone();
        next.as_mut_slice()[0] += 1.0;
        Ok((next, 1.0, false))
    }
    fn step_batch(
        &self,
        _: &[FullLatent],
        _: &[ActionId],
        _: &mut [FullLatent],
        _: &mut [f32],
        _: &mut [bool],
    ) -> Result<(), CoreError> {
        panic!("deadline search must use checked single steps")
    }
}
fn request(state: &FullLatent, mode: DecideMode) -> DecideRequest<'_> {
    DecideRequest {
        active_context: Vec::new(),

        state,
        candidates: &ACTIONS,
        mode,
        entropy: NormalizedEntropy::ZERO,
        return_trajectory: false,
        horizon: 4,
        deadline: None,
        budget_ms: Some(2.0),
    }
}
fn one_step_goal(state: &FullLatent) -> bool {
    state.as_slice()[0] >= 1.0
}
fn pipeline_with_config(
    fast_calls: usize,
    config: PlannerConfig,
) -> (ProductionPipeline, Arc<SlowModel>) {
    let model = Arc::new(SlowModel {
        calls: AtomicUsize::new(0),
        fast_calls,
        sleep: Duration::from_millis(3),
    });
    let p =
        ProductionPipeline::new_with_config(model.clone(), Arc::new(PolicyGate::default()), config)
            .unwrap()
            .with_astar_goal(AStarGoal::Predicate(one_step_goal));
    (p, model)
}
fn pipeline(fast_calls: usize, horizon: usize) -> (ProductionPipeline, Arc<SlowModel>) {
    pipeline_with_config(
        fast_calls,
        PlannerConfig {
            mcts_horizon: horizon,
            cem_horizon: horizon,
            ..Default::default()
        },
    )
}

#[test]
fn three_ms_step_does_not_hold_caller_past_three_ms() {
    let mut model_calls = 0;
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    for mode in MODES {
        let (p, model) = pipeline(0, 4);
        let state = FullLatent::zeros();
        let start = Instant::now();
        let result = p.decide(&request(&state, mode));
        let elapsed = start.elapsed();
        eprintln!("slow 3ms {mode:?}: {elapsed:?}, result={result:?}");
        assert!(matches!(result, Err(PlannerError::TimeoutExceeded(_))));
        assert!(elapsed >= Duration::from_millis(2));
        assert!(
            elapsed < Duration::from_millis(15),
            "2ms deadline overrun: {elapsed:?}"
        );
        // Give the outstanding call time to return; it must not start step 2.
        std::thread::sleep(Duration::from_millis(5));
        let calls = model.calls.load(Ordering::SeqCst);
        // The deadline may expire before the worker is scheduled. That must
        // fail closed without calling the model, rather than force a late step.
        assert!(calls <= 1, "worker started another step after expiry");
        model_calls += calls;
    }
    assert!(model_calls > 0, "no slow model was actually exercised");
}

#[test]
fn completed_certified_candidate_survives_timeout_in_each_engine() {
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    for mode in [
        DecideMode::Mcts,
        DecideMode::MpcCem,
        DecideMode::AStar,
        DecideMode::ManifoldGFlowNet,
        DecideMode::CfrNash,
        DecideMode::Reflex,
    ] {
        let (p, model) = pipeline(2, 1);
        let state = FullLatent::zeros();
        let mut req = request(&state, mode);
        // One screened candidate plus one complete engine evaluation leaves
        // the third model call for the real slow operation.  A trajectory is
        // requested so one-step engines also remain in the timeout path after
        // publishing their certified candidate.
        req.candidates = &ACTIONS[..1];
        req.return_trajectory = true;
        let start = Instant::now();
        let decision = p.decide(&req).unwrap();
        let elapsed = start.elapsed();
        eprintln!(
            "anytime {mode:?}: {elapsed:?}, action={:?}",
            decision.action
        );
        assert!(elapsed < Duration::from_millis(15), "{elapsed:?}");
        assert_eq!(decision.action, ACTIONS[0]);
        assert_eq!(decision.gate_tier, PolicyTier::Tier0Proceed);
        assert!(decision.timed_out);
        assert_eq!(decision.entropy, NormalizedEntropy(1.0));
        assert!(decision.trajectory.is_none());
        let calls_at_return = model.calls.load(Ordering::SeqCst);
        assert!(
            calls_at_return >= 2,
            "screen and complete search were not both exercised: {calls_at_return}"
        );
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(model.calls.load(Ordering::SeqCst), calls_at_return);
    }
}

#[test]
fn incomplete_cem_trajectory_is_not_an_incumbent() {
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    let (p, _) = pipeline(1, 4);
    let state = FullLatent::zeros();
    let mut req = request(&state, DecideMode::MpcCem);
    req.candidates = &ACTIONS[..1];
    assert!(matches!(
        p.decide(&req),
        Err(PlannerError::TimeoutExceeded(_))
    ));
}

#[test]
fn expired_and_invalid_budgets_do_not_call_model() {
    let (p, model) = pipeline(0, 4);
    let state = FullLatent::zeros();
    let mut req = request(&state, DecideMode::Mcts);
    req.deadline = Some(Instant::now());
    assert!(matches!(
        p.decide(&req),
        Err(PlannerError::TimeoutExceeded(_))
    ));
    req.deadline = None;
    for budget in [f64::NAN, f64::INFINITY, -1.0, f64::MAX] {
        req.budget_ms = Some(budget);
        assert!(matches!(p.decide(&req), Err(PlannerError::InvalidInput(_))));
        assert!(PlannerConfig {
            budget_ms: Some(budget),
            ..Default::default()
        }
        .validate()
        .is_err());
    }
    req.budget_ms = Some(0.0);
    assert!(matches!(
        p.decide(&req),
        Err(PlannerError::TimeoutExceeded(_))
    ));
    assert_eq!(model.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn blocked_action_is_never_promoted_to_incumbent() {
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    let mut gate = PolicyGate::default();
    for action in ACTIONS {
        gate.add_constraint(LinearConstraint::prohibit(
            RuleId(action.0),
            "blocked",
            action,
        ));
    }
    let model = Arc::new(SlowModel {
        calls: AtomicUsize::new(0),
        fast_calls: usize::MAX,
        sleep: Duration::from_millis(3),
    });
    let p = ProductionPipeline::new(model.clone(), Arc::new(gate));
    for mode in MODES {
        let result = p.decide(&request(&FullLatent::zeros(), mode));
        assert!(
            matches!(
                result,
                Err(PlannerError::NoFeasibleAction)
                    | Err(PlannerError::TimeoutExceeded(_))
                    | Err(PlannerError::PlannerBusy)
            ),
            "blocked candidate escaped refusal: {result:?}"
        );
    }
    assert_eq!(model.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn budgeted_active_context_mutex_prunes_before_screening_model_call() {
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    let model = Arc::new(SlowModel {
        calls: AtomicUsize::new(0),
        fast_calls: usize::MAX,
        sleep: Duration::from_millis(3),
    });
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::mutex(
        RuleId(700),
        "active_mutex",
        ActionId(1),
        ActionId(2),
    ));
    let p = ProductionPipeline::new_with_config(
        model.clone(),
        Arc::new(gate),
        PlannerConfig {
            mcts_horizon: 1,
            cem_horizon: 1,
            ..Default::default()
        },
    )
    .unwrap()
    .with_astar_goal(AStarGoal::Predicate(one_step_goal));
    let state = FullLatent::zeros();
    let mut req = request(&state, DecideMode::Mcts);
    req.candidates = &ACTIONS[..1];
    req.active_context = vec![ActionId(2)];
    assert_eq!(p.decide(&req).unwrap_err(), PlannerError::NoFeasibleAction);
    assert_eq!(
        model.calls.load(Ordering::SeqCst),
        0,
        "a mutex-blocked budgeted candidate reached screening/model"
    );
}

struct BlockedModel {
    release: Arc<(Mutex<bool>, Condvar)>,
    entered: std::sync::mpsc::Sender<()>,
}
impl WorldModelDynamics for BlockedModel {
    type Error = CoreError;
    fn step(&self, state: &FullLatent, _: ActionId) -> Result<(FullLatent, f32, bool), CoreError> {
        self.entered.send(()).unwrap();
        let (lock, cv) = &*self.release;
        let _guard = cv
            .wait_while(lock.lock().unwrap(), |released| !*released)
            .unwrap();
        Ok((state.clone(), 1.0, false))
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
struct Release(Arc<(Mutex<bool>, Condvar)>);
impl Drop for Release {
    fn drop(&mut self) {
        *self.0 .0.lock().unwrap() = true;
        self.0 .1.notify_all();
    }
}
#[test]
fn caller_returns_while_model_is_still_blocked_and_reentry_fails_closed() {
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    let release = Release(Arc::new((Mutex::new(false), Condvar::new())));
    let (tx, rx) = std::sync::mpsc::channel();
    let p = ProductionPipeline::new(
        Arc::new(BlockedModel {
            release: release.0.clone(),
            entered: tx,
        }),
        Arc::new(PolicyGate::default()),
    );
    let caller_pipeline = p.clone();
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    let caller = std::thread::spawn(move || {
        let state = FullLatent::zeros();
        let req = request(&state, DecideMode::Mcts);
        let start = Instant::now();
        let result = caller_pipeline.decide(&req);
        let _ = result_tx.send((result, start.elapsed()));
    });
    // A regression that joins the blocked worker fails instead of hanging this test.
    let (result, elapsed) = result_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(matches!(result, Err(PlannerError::TimeoutExceeded(_))));
    assert!(elapsed < Duration::from_millis(15));
    caller.join().unwrap();
    let state = FullLatent::zeros();
    let req = request(&state, DecideMode::Mcts);
    rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert_eq!(p.decide(&req).unwrap_err(), PlannerError::PlannerBusy);
    assert_eq!(
        p.clone().decide(&req).unwrap_err(),
        PlannerError::PlannerBusy
    );
    // Dropping release unblocks the worker only AFTER the caller has returned.
}

#[test]
fn config_budget_cannot_be_extended_by_request() {
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    let (p, _) = pipeline_with_config(
        0,
        PlannerConfig {
            budget_ms: Some(2.0),
            ..Default::default()
        },
    );
    let state = FullLatent::zeros();
    let mut req = request(&state, DecideMode::Mcts);
    req.budget_ms = Some(100.0);
    let start = Instant::now();
    assert!(matches!(
        p.decide(&req),
        Err(PlannerError::TimeoutExceeded(_))
    ));
    assert!(
        start.elapsed() < Duration::from_millis(20),
        "elapsed={:?}",
        start.elapsed()
    );
}

#[test]
fn full_graph_revocation_cannot_be_bypassed_by_anytime() {
    use gen_zero_lod::{EpistemicStatus, LodBand, LodGraph, LodNode, MixedCurvatureCoord};
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    let graph = Arc::new(LodGraph::new());
    graph.add_node(
        LodNode::new(
            0,
            LodBand::Lod0Atomic,
            MixedCurvatureCoord::origin(),
            "revoked",
            1,
        )
        .with_status(EpistemicStatus::Falsified),
    );
    let state = FullLatent::zeros();
    for mode in MODES {
        let (p, _) = pipeline(usize::MAX, 1);
        let p = p.with_graph(graph.clone());
        let mut req = request(&state, mode);
        // Revocation is the subject here; remove the deadline so a fast
        // unbounded engine can finish its configured search after screening.
        req.budget_ms = None;
        let decision = p.decide(&req).unwrap();
        assert_eq!(decision.action, ActionId(2));
        assert!(!decision.feasible.contains(&ActionId(1)));
    }
}

#[test]
fn early_numeric_error_is_not_hidden_by_an_incumbent() {
    struct InvalidSecond(AtomicUsize);
    impl WorldModelDynamics for InvalidSecond {
        type Error = CoreError;
        fn step(
            &self,
            state: &FullLatent,
            _: ActionId,
        ) -> Result<(FullLatent, f32, bool), CoreError> {
            let n = self.0.fetch_add(1, Ordering::SeqCst);
            Ok((state.clone(), if n == 0 { 1.0 } else { f32::NAN }, false))
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
    for mode in MODES {
        let p = ProductionPipeline::new_with_config(
            Arc::new(InvalidSecond(AtomicUsize::new(0))),
            Arc::new(PolicyGate::default()),
            PlannerConfig {
                cem_horizon: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let state = FullLatent::zeros();
        let mut req = request(&state, mode);
        req.budget_ms = Some(100.0);
        assert!(matches!(
            p.decide(&req),
            Err(PlannerError::DivergentState(_))
        ));
    }
}

#[test]
fn trajectory_timeout_returns_explicitly_truncated_decision() {
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    let (p, model) = pipeline(2, 1);
    let state = FullLatent::zeros();
    let mut req = request(&state, DecideMode::AStar);
    req.candidates = &ACTIONS[..1];
    req.return_trajectory = true;
    let result = p.decide(&req).unwrap();
    assert!(result.timed_out);
    assert!(result.trajectory.is_none());
    assert_eq!(result.action, ActionId(1));
    let calls_at_return = model.calls.load(Ordering::SeqCst);
    assert!(calls_at_return >= 3);
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(model.calls.load(Ordering::SeqCst), calls_at_return);
}

#[test]
fn auto_and_other_modes_obey_the_same_caller_deadline() {
    let _serial = TIMING.lock().unwrap_or_else(|poison| poison.into_inner());
    for (mode, entropy) in [
        (DecideMode::Auto, 0.0),
        (DecideMode::Auto, 0.5),
        (DecideMode::Auto, 0.9),
        (DecideMode::Reflex, 0.0),
        (DecideMode::ManifoldGFlowNet, 0.0),
        (DecideMode::CfrNash, 0.0),
    ] {
        let (p, model) = pipeline(0, 4);
        let state = FullLatent::zeros();
        let mut req = request(&state, mode);
        req.entropy = NormalizedEntropy(entropy);
        let start = Instant::now();
        let result = p.decide(&req);
        eprintln!(
            "deadline {mode:?}, entropy={entropy}: {:?}",
            start.elapsed()
        );
        assert!(
            matches!(result, Err(PlannerError::TimeoutExceeded(_))),
            "{mode:?}: {result:?}"
        );
        assert!(
            start.elapsed() < Duration::from_millis(15),
            "elapsed={:?}",
            start.elapsed()
        );
        std::thread::sleep(Duration::from_millis(5));
        assert!(model.calls.load(Ordering::SeqCst) <= 1);
    }
}

#[test]
fn config_rejects_serialized_process_local_deadline() {
    assert!(serde_json::from_str::<PlannerConfig>(r#"{"deadline": 42}"#).is_err());
    let config: PlannerConfig = serde_json::from_str(r#"{"budget_ms":2.0}"#).unwrap();
    assert_eq!(config.budget_ms, Some(2.0));
    assert!(config.validate().is_ok());
}
