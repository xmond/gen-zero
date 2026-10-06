//! Audit §5.3 F01–F09: adversarial models return Ok with invalid values on purpose.
//! These are software boundary regressions, not proofs of real-world safety.
use gen_zero_core::{
    ActionId, CoreError, FullLatent, LocalActionFrame, NormalizedEntropy, WorldModelDynamics,
};
use gen_zero_gate::{Budget, LinearConstraint, PolicyGate, PolicyTier, RuleId, SheafProblem};
use gen_zero_lod::LodGraph;
use gen_zero_planner::{
    AStarEngine, CfrNashEngine, CpSatFormalEngine, DecideMode, DecideRequest, DynamicKMoERouter,
    ManifoldGFlowNetEngine, MctsEngine, MpcCemEngine, PlannerError, PlanningEngine,
    ProductionPipeline,
};
use nalgebra::{DMatrix, DVector};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

const BAD: ActionId = ActionId(1);
const GOOD: ActionId = ActionId(2);
const MODES: [DecideMode; 7] = [
    DecideMode::Auto,
    DecideMode::Mcts,
    DecideMode::MpcCem,
    DecideMode::AStar,
    DecideMode::ManifoldGFlowNet,
    DecideMode::CfrNash,
    DecideMode::Reflex,
];

#[derive(Clone, Copy)]
enum Fault {
    None,
    Reward(f32),
    State(f32),
    Hazard,
    LateHazard,
    SearchHazard,
}
struct Model {
    fault: Fault,
    calls: Arc<AtomicUsize>,
}
impl Model {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}
impl WorldModelDynamics for Model {
    type Error = CoreError;
    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), CoreError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let mut next = state.clone();
        next.as_mut_slice()[0] += 1.0;
        let mut reward = if action == BAD { 10.0 } else { 0.0 };
        let mut done = false;
        if action == BAD {
            match self.fault {
                Fault::Reward(value) => reward = value,
                Fault::State(value) => next.as_mut_slice()[1023] = value,
                Fault::Hazard => done = true,
                Fault::LateHazard => done = call >= 3,
                Fault::SearchHazard => done = call == 2,
                Fault::None => {}
            }
        }
        Ok((next, reward, done))
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
            let (next, reward, done) = self.step(&states[i], actions[i])?;
            next_states[i] = next;
            rewards[i] = reward;
            dones[i] = done;
        }
        Ok(())
    }
}
fn engines() -> Vec<Box<dyn PlanningEngine>> {
    vec![
        Box::new(MctsEngine::default()),
        Box::new(AStarEngine {
            goal: Some(gen_zero_planner::AStarGoal::Predicate(|s| {
                s.as_slice()[0] >= 1.0
            })),
            ..Default::default()
        }),
        Box::new(MpcCemEngine::default()),
        Box::new(ManifoldGFlowNetEngine),
        Box::new(CfrNashEngine),
        Box::new(CpSatFormalEngine),
    ]
}
fn request<'a>(
    state: &'a FullLatent,
    candidates: &'a [ActionId],
    mode: DecideMode,
) -> DecideRequest<'a> {
    DecideRequest {
        active_context: Vec::new(),
        deadline: None,
        budget_ms: None,
        state,
        candidates,
        mode,
        entropy: NormalizedEntropy(0.5),
        return_trajectory: false,
        horizon: 2,
        causal_triad: None,
    }
}
fn blocked_gate() -> PolicyGate {
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::prohibit(RuleId(8), "forbidden", BAD));
    gate
}

#[test]
fn f01_large_coefficients_cannot_wrap_or_panic_through_gate() {
    let rule = LinearConstraint {
        rule_id: RuleId(1),
        rule_name: "overflow",
        terms: [(BAD, i32::MAX), (BAD, i32::MAX)].into_iter().collect(),
        rhs: 0,
    };
    assert!(!rule.is_satisfied_by_single_action(BAD));
    assert!(!rule.is_satisfied_by_bundle(&[BAD]));
    assert!(!rule.is_satisfied_with_context(BAD, &[BAD, BAD]));
    assert!(rule.is_satisfied_by_single_action(GOOD));
    let mut gate = PolicyGate::default();
    gate.add_constraint(rule);
    let verdict = gate.evaluate_basic(BAD, NormalizedEntropy::ZERO).unwrap();
    assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
    assert_eq!(verdict.violated_rules.as_slice(), &[RuleId(1)]);
    assert_eq!(
        gate.evaluate_basic(GOOD, NormalizedEntropy::ZERO)
            .unwrap()
            .tier,
        PolicyTier::Tier0Proceed
    );
}

#[test]
fn f02_all_six_engines_reject_nan_reward_without_fallback() {
    let frame = LocalActionFrame::new(&["bad"], &[BAD]).unwrap();
    for engine in engines() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let result = engine.plan(
                &FullLatent::zeros(),
                &frame,
                &Model::new(Fault::Reward(value)),
                &PolicyGate::default(),
            );
            assert!(
                matches!(result, Err(PlannerError::DivergentState(_))),
                "{}: {result:?}",
                engine.name()
            );
        }
        assert!(
            engine
                .plan(
                    &FullLatent::zeros(),
                    &frame,
                    &Model::new(Fault::None),
                    &PolicyGate::default()
                )
                .is_ok(),
            "{} rejects valid model",
            engine.name()
        );
    }
}

#[test]
fn f03_mcts_and_cfr_reject_nonfinite_successor_even_with_finite_reward() {
    let frame = LocalActionFrame::new(&["bad"], &[BAD]).unwrap();
    for engine in [
        Box::new(MctsEngine::default()) as Box<dyn PlanningEngine>,
        Box::new(CfrNashEngine),
    ] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let result = engine.plan(
                &FullLatent::zeros(),
                &frame,
                &Model::new(Fault::State(value)),
                &PolicyGate::default(),
            );
            assert!(
                matches!(result, Err(PlannerError::DivergentState(_))),
                "{}: {result:?}",
                engine.name()
            );
        }
    }
}

#[test]
fn f04_occupied_mutex_context_hard_stops_gate_and_pipeline() {
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::mutex(RuleId(4), "occupied", BAD, GOOD));
    assert_eq!(
        gate.evaluate_basic(BAD, NormalizedEntropy::ZERO)
            .unwrap()
            .tier,
        PolicyTier::Tier0Proceed
    );
    let verdict = gate
        .evaluate_with_context::<LodGraph>(BAD, &[GOOD], None, NormalizedEntropy::ZERO, None, None)
        .unwrap();
    assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
    assert_eq!(verdict.violated_rules.as_slice(), &[RuleId(4)]);
    let model = Model::new(Fault::None);
    let calls = model.calls.clone();
    let pipeline = ProductionPipeline::new(Arc::new(model), Arc::new(gate));
    for mode in MODES {
        assert_eq!(
            pipeline
                .decide_with_context(&request(&FullLatent::zeros(), &[BAD], mode), &[GOOD])
                .unwrap_err(),
            PlannerError::NoFeasibleAction
        );
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "occupied action reached dynamics"
    );
    assert_eq!(
        pipeline
            .decide(&request(&FullLatent::zeros(), &[BAD], DecideMode::Reflex))
            .unwrap()
            .action,
        BAD
    );
}

#[test]
fn f05_registered_topology_requires_certificate_before_model_execution() {
    let problem = SheafProblem::new(
        DMatrix::from_row_slice(1, 2, &[1.0, -1.0]),
        DVector::from_row_slice(&[1.0]),
        DVector::from_row_slice(&[0.0]),
        vec![],
        vec![],
        1,
    )
    .unwrap();
    let mut gate = PolicyGate::default();
    gate.require_heat_certificate(BAD, problem, Budget::default());
    assert_eq!(
        gate.evaluate_basic(BAD, NormalizedEntropy::ZERO)
            .unwrap()
            .tier,
        PolicyTier::Tier3HardStop
    );
    let model = Model::new(Fault::None);
    let calls = model.calls.clone();
    let pipeline = ProductionPipeline::new(Arc::new(model), Arc::new(gate));
    for mode in MODES {
        assert_eq!(
            pipeline
                .decide(&request(&FullLatent::zeros(), &[BAD], mode))
                .unwrap_err(),
            PlannerError::NoFeasibleAction
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        pipeline
            .decide(&request(&FullLatent::zeros(), &[GOOD], DecideMode::Reflex))
            .unwrap()
            .action,
        GOOD
    );
}

#[test]
fn f06_decide_excludes_high_reward_terminal_hazard_in_every_mode() {
    let pipeline = ProductionPipeline::new(
        Arc::new(Model::new(Fault::Hazard)),
        Arc::new(PolicyGate::default()),
    );
    let pipeline = pipeline.with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
        s.as_slice()[0] >= 1.0
    }));
    for mode in MODES {
        let decision = pipeline
            .decide(&request(&FullLatent::zeros(), &[BAD, GOOD], mode))
            .unwrap();
        assert_eq!(decision.action, GOOD, "{mode:?}");
        assert_eq!(decision.feasible, vec![GOOD]);
        assert!(decision.pruned.iter().any(|p| p.action == BAD
            && p.source == gen_zero_planner::PruneSource::ModelHazard
            && p.tier.is_none()));
        assert_eq!(
            pipeline
                .decide(&request(&FullLatent::zeros(), &[BAD], mode))
                .unwrap_err(),
            PlannerError::NoFeasibleAction
        );
    }
}

#[test]
fn f07_simulate_separates_policy_allowed_from_hazard_free() {
    for (fault, gate, policy_allowed, hazard_free) in [
        (Fault::None, PolicyGate::default(), true, true),
        (Fault::None, blocked_gate(), false, true),
        (Fault::Hazard, PolicyGate::default(), true, false),
        (Fault::Hazard, blocked_gate(), false, false),
    ] {
        let pipeline = ProductionPipeline::new(Arc::new(Model::new(fault)), Arc::new(gate));
        if !policy_allowed {
            assert!(matches!(
                pipeline.simulate(&FullLatent::zeros(), &[BAD], None),
                Err(PlannerError::NoFeasibleAction)
            ));
            continue;
        }
        let rollout = pipeline
            .simulate(&FullLatent::zeros(), &[BAD], None)
            .unwrap();
        assert_eq!(
            rollout.steps_simulated(),
            1,
            "counterfactual simulation must be explicit"
        );
        assert_eq!(rollout.policy_allowed(), policy_allowed);
        assert_eq!(rollout.hazard_free(), hazard_free);
        assert_eq!(rollout.is_safe(), policy_allowed && hazard_free);
    }
}

#[test]
fn f08_reflex_never_returns_unvetted_first_action() {
    let pipeline =
        ProductionPipeline::new(Arc::new(Model::new(Fault::None)), Arc::new(blocked_gate()));
    assert_eq!(
        pipeline
            .decide(&request(
                &FullLatent::zeros(),
                &[BAD, GOOD],
                DecideMode::Reflex
            ))
            .unwrap()
            .action,
        GOOD
    );
    assert_eq!(
        pipeline
            .decide(&request(&FullLatent::zeros(), &[BAD], DecideMode::Reflex))
            .unwrap_err(),
        PlannerError::NoFeasibleAction
    );
}

#[test]
fn f09_router_rejects_invalid_entropy_before_dispatch() {
    let router =
        DynamicKMoERouter::default().with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
            s.as_slice()[0] >= 1.0
        }));
    let model = Model::new(Fault::None);
    let frame = LocalActionFrame::new(&["bad", "good"], &[BAD, GOOD]).unwrap();
    let gate = PolicyGate::default();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.5] {
        assert_eq!(
            gate.evaluate_basic(BAD, NormalizedEntropy(value))
                .unwrap()
                .tier,
            PolicyTier::Tier3HardStop
        );
        assert!(matches!(
            router.dispatch(
                &FullLatent::zeros(),
                &frame,
                NormalizedEntropy(value),
                &model,
                &gate
            ),
            Err(PlannerError::InvalidInput(_))
        ));
        assert!(matches!(
            router.dispatch_with_tier(
                &FullLatent::zeros(),
                &frame,
                NormalizedEntropy(value),
                &model,
                &gate
            ),
            Err(PlannerError::InvalidInput(_))
        ));
    }
    assert_eq!(
        model.calls.load(Ordering::SeqCst),
        0,
        "invalid entropy reached an engine"
    );
    for value in [0.0, 0.5, 1.0] {
        assert!(router
            .dispatch(
                &FullLatent::zeros(),
                &frame,
                NormalizedEntropy(value),
                &model,
                &gate
            )
            .is_ok());
    }
}

#[test]
fn f06_changed_prediction_at_final_check_is_rejected_without_fallback() {
    // Reflex performs candidate screening, engine scoring, then final validation.
    let model = Model::new(Fault::LateHazard);
    let calls = model.calls.clone();
    let pipeline = ProductionPipeline::new(Arc::new(model), Arc::new(PolicyGate::default()));
    assert_eq!(
        pipeline
            .decide(&request(&FullLatent::zeros(), &[BAD], DecideMode::Reflex))
            .unwrap_err(),
        PlannerError::NoFeasibleAction
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn f06_search_cannot_ignore_a_transient_hazard() {
    for mode in MODES {
        let pipeline = ProductionPipeline::new(
            Arc::new(Model::new(Fault::SearchHazard)),
            Arc::new(PolicyGate::default()),
        );
        let pipeline = pipeline.with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
            s.as_slice()[0] >= 1.0
        }));
        let result = pipeline.decide(&request(&FullLatent::zeros(), &[BAD], mode));
        assert!(
            matches!(result, Err(PlannerError::Core(CoreError::WorldModel(_)))),
            "{mode:?}: {result:?}"
        );
    }
}
