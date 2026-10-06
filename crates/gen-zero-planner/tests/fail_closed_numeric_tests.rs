use gen_zero_core::{ActionId, CoreError, FullLatent, LocalActionFrame, WorldModelDynamics};
use gen_zero_gate::{LinearConstraint, PolicyGate, PolicyTier, RuleId};
use gen_zero_planner::{
    AStarEngine, CfrNashEngine, CpSatFormalEngine, DecideMode, DecideRequest,
    ManifoldGFlowNetEngine, MctsEngine, MpcCemEngine, PlannerError, PlanningEngine,
    ProductionPipeline,
};
use std::sync::Arc;

const ACTIONS: [ActionId; 3] = [ActionId(1), ActionId(2), ActionId(3)];
const NAMES: [&str; 3] = ["one", "two", "three"];

fn numeric_astar() -> AStarEngine {
    AStarEngine {
        goal: Some(gen_zero_planner::AStarGoal::Predicate(|s| {
            s.as_slice()[1] >= 1.0
        })),
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
enum NumericFault {
    None,
    CoreError,
    SuccessorNan,
    SuccessorInf,
    SuccessorNegInf,
    RewardNan,
    RewardInf,
    RewardNegInf,
    CostOverflow,
    RewardMax,
}

struct NumericModel {
    fault: NumericFault,
}

impl NumericModel {
    fn step_one(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), CoreError> {
        if matches!(self.fault, NumericFault::CoreError) {
            return Err(CoreError::WorldModel("transition failed".into()));
        }
        let mut next = state.clone();
        next.as_mut_slice()[1] += 1.0;
        let reward = match self.fault {
            NumericFault::RewardNan if action == ActionId(1) => f32::NAN,
            NumericFault::RewardInf if action == ActionId(1) => f32::INFINITY,
            NumericFault::RewardNegInf if action == ActionId(1) => f32::NEG_INFINITY,
            NumericFault::RewardMax => f32::MAX,
            _ => action.0 as f32,
        };
        match self.fault {
            NumericFault::SuccessorNan if action == ActionId(1) => {
                next.as_mut_slice()[0] = f32::NAN;
            }
            NumericFault::SuccessorInf if action == ActionId(1) => {
                next.as_mut_slice()[0] = f32::INFINITY;
            }
            NumericFault::SuccessorNegInf if action == ActionId(1) => {
                next.as_mut_slice()[0] = f32::NEG_INFINITY;
            }
            NumericFault::CostOverflow if action == ActionId(1) => {
                next.as_mut_slice()[0] = 100.0;
            }
            _ => {}
        }
        Ok((next, reward, false))
    }
}

impl WorldModelDynamics for NumericModel {
    type Error = CoreError;

    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), Self::Error> {
        self.step_one(state, action)
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
            let (next, reward, done) = self.step_one(&states[i], actions[i])?;
            next_states[i] = next;
            rewards[i] = reward;
            dones[i] = done;
        }
        Ok(())
    }
}

fn frame(actions: &[ActionId]) -> LocalActionFrame<'static> {
    assert_eq!(actions.len(), NAMES.len());
    LocalActionFrame::new(&NAMES, actions).expect("test frame fits")
}

fn assert_divergent(result: Result<(ActionId, gen_zero_core::NormalizedEntropy), PlannerError>) {
    let is_divergent = matches!(&result, Err(PlannerError::DivergentState(_)));
    assert!(is_divergent, "{result:?}");
}

fn decide_request<'a>(state: &'a FullLatent, candidates: &'a [ActionId]) -> DecideRequest<'a> {
    DecideRequest {
        active_context: Vec::new(),
        deadline: None,
        budget_ms: None,
        state,
        candidates,
        mode: DecideMode::AStar,
        entropy: gen_zero_core::NormalizedEntropy::ZERO,
        return_trajectory: false,
        horizon: 2,
        causal_triad: None,
    }
}

#[test]
fn astar_rejects_nonfinite_input_successor_reward_and_costs() {
    let gate = PolicyGate::default();
    let actions = frame(&ACTIONS);

    let mut nan_state = FullLatent::zeros();
    nan_state.as_mut_slice()[0] = f32::NAN;
    assert_divergent(numeric_astar().plan(
        &nan_state,
        &actions,
        &NumericModel {
            fault: NumericFault::None,
        },
        &gate,
    ));

    for fault in [
        NumericFault::SuccessorNan,
        NumericFault::SuccessorInf,
        NumericFault::SuccessorNegInf,
        NumericFault::RewardNan,
        NumericFault::RewardInf,
        NumericFault::RewardNegInf,
    ] {
        assert_divergent(numeric_astar().plan(
            &FullLatent::zeros(),
            &actions,
            &NumericModel { fault },
            &gate,
        ));
    }

    // Finite transition values can still overflow the weighted cost.
    assert_divergent(
        AStarEngine {
            uncertainty_penalty_weight: f32::MAX,
            ..numeric_astar()
        }
        .plan(
            &FullLatent::zeros(),
            &actions,
            &NumericModel {
                fault: NumericFault::CostOverflow,
            },
            &gate,
        ),
    );
}

#[test]
fn cpsat_rejects_nonfinite_values_before_nan_can_win_best_action() {
    let actions = frame(&ACTIONS);
    let gate = PolicyGate::default();

    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut state = FullLatent::zeros();
        state.as_mut_slice()[0] = value;
        assert_divergent(CpSatFormalEngine.plan(
            &state,
            &actions,
            &NumericModel {
                fault: NumericFault::None,
            },
            &gate,
        ));
    }

    // Action 1 is visited first. A NaN must fail closed instead of becoming the
    // initial best action through `best_act.is_none()`.
    assert_divergent(CpSatFormalEngine.plan(
        &FullLatent::zeros(),
        &actions,
        &NumericModel {
            fault: NumericFault::RewardNan,
        },
        &gate,
    ));
    for fault in [
        NumericFault::SuccessorNan,
        NumericFault::SuccessorInf,
        NumericFault::SuccessorNegInf,
        NumericFault::RewardInf,
        NumericFault::RewardNegInf,
    ] {
        assert_divergent(CpSatFormalEngine.plan(
            &FullLatent::zeros(),
            &actions,
            &NumericModel { fault },
            &gate,
        ));
    }
}

#[test]
fn cem_zero_parameters_are_rejected_before_sampling() {
    let actions = frame(&ACTIONS);
    let gate = PolicyGate::default();
    for engine in [
        MpcCemEngine {
            num_elites: 0,
            ..MpcCemEngine::default()
        },
        MpcCemEngine {
            num_iterations: 0,
            ..MpcCemEngine::default()
        },
        MpcCemEngine {
            num_samples: 0,
            ..MpcCemEngine::default()
        },
        MpcCemEngine {
            horizon: 0,
            ..MpcCemEngine::default()
        },
    ] {
        assert!(matches!(
            engine.plan(
                &FullLatent::zeros(),
                &actions,
                &NumericModel {
                    fault: NumericFault::None,
                },
                &gate,
            ),
            Err(PlannerError::InvalidInput(_))
        ));
    }
}

#[test]
fn cpsat_filter_verdicts_preserves_blocked_candidates() {
    let actions = frame(&ACTIONS);
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::prohibit(
        RuleId(7),
        "blocked",
        ActionId(2),
    ));

    let verdicts = CpSatFormalEngine.filter_verdicts(&actions, &gate);
    assert_eq!(verdicts.len(), 3);
    assert_eq!(
        verdicts.iter().map(|v| v.action).collect::<Vec<_>>(),
        ACTIONS
    );
    assert_eq!(verdicts[1].tier, PolicyTier::Tier3HardStop);
    assert_eq!(verdicts[1].violated_rules.as_slice(), &[RuleId(7)]);
    assert_eq!(
        CpSatFormalEngine
            .filter_feasible(&actions, &gate)
            .as_slice(),
        &[ActionId(1), ActionId(3)]
    );
}

#[test]
fn pipeline_rejects_nonfinite_astar_transition_and_return_overflow() {
    let actions = [ActionId(1)];
    let state = FullLatent::zeros();

    let pipeline = ProductionPipeline::new(
        Arc::new(NumericModel {
            fault: NumericFault::SuccessorNan,
        }),
        Arc::new(PolicyGate::default()),
    );
    let pipeline = pipeline.with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
        s.as_slice()[1] >= 1.0
    }));
    assert!(matches!(
        pipeline.decide(&decide_request(&state, &actions)),
        Err(PlannerError::DivergentState(_))
    ));

    let overflow_pipeline = ProductionPipeline::new(
        Arc::new(NumericModel {
            fault: NumericFault::RewardMax,
        }),
        Arc::new(PolicyGate::default()),
    );
    let overflow_pipeline =
        overflow_pipeline.with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
            s.as_slice()[1] >= 1.0
        }));
    let mut trajectory_request = decide_request(&state, &actions);
    trajectory_request.return_trajectory = true;
    assert!(matches!(
        overflow_pipeline.decide(&trajectory_request),
        Err(PlannerError::DivergentState(_))
    ));
}

#[test]
fn validated_engines_preserve_world_model_errors() {
    let actions = frame(&ACTIONS);
    let engines: [&dyn PlanningEngine; 6] = [
        &numeric_astar(),
        &CpSatFormalEngine,
        &MpcCemEngine::default(),
        &MctsEngine::default(),
        &CfrNashEngine,
        &ManifoldGFlowNetEngine,
    ];
    for engine in engines {
        assert_eq!(
            engine.plan(
                &FullLatent::zeros(),
                &actions,
                &NumericModel {
                    fault: NumericFault::CoreError
                },
                &PolicyGate::default()
            ),
            Err(PlannerError::Core(CoreError::WorldModel(
                "transition failed".into()
            )))
        );
    }
}

#[test]
fn cem_rejects_nonfinite_transitions_and_accumulated_reward() {
    // All trajectories diverge: exclusion must never return an unvetted fallback.
    let actions = LocalActionFrame::new(&NAMES[..1], &ACTIONS[..1]).unwrap();
    for fault in [
        NumericFault::RewardNan,
        NumericFault::RewardInf,
        NumericFault::SuccessorNan,
        NumericFault::RewardMax,
    ] {
        assert_divergent(MpcCemEngine::default().plan(
            &FullLatent::zeros(),
            &actions,
            &NumericModel { fault },
            &PolicyGate::default(),
        ));
    }
}

#[test]
fn mcts_cfr_and_gflow_reject_invalid_inputs_and_transitions() {
    let actions = frame(&ACTIONS);
    let engines: [&dyn PlanningEngine; 3] = [
        &MctsEngine::default(),
        &CfrNashEngine,
        &ManifoldGFlowNetEngine,
    ];
    for engine in engines {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let mut state = FullLatent::zeros();
            state.as_mut_slice()[0] = value;
            assert_divergent(engine.plan(
                &state,
                &actions,
                &NumericModel {
                    fault: NumericFault::None,
                },
                &PolicyGate::default(),
            ));
        }
        for fault in [
            NumericFault::SuccessorNan,
            NumericFault::SuccessorInf,
            NumericFault::SuccessorNegInf,
            NumericFault::RewardNan,
            NumericFault::RewardInf,
            NumericFault::RewardNegInf,
        ] {
            assert_divergent(engine.plan(
                &FullLatent::zeros(),
                &actions,
                &NumericModel { fault },
                &PolicyGate::default(),
            ));
        }
    }
}

#[test]
fn mcts_rejects_invalid_configuration_and_uses_finite_wide_backups() {
    let actions = frame(&ACTIONS);
    for engine in [
        MctsEngine {
            max_simulations: 0,
            ..MctsEngine::default()
        },
        MctsEngine {
            c_puct: f32::NAN,
            ..MctsEngine::default()
        },
        MctsEngine {
            c_puct: -1.0,
            ..MctsEngine::default()
        },
        MctsEngine {
            c_puct: f32::INFINITY,
            ..MctsEngine::default()
        },
    ] {
        assert!(matches!(
            engine.plan(
                &FullLatent::zeros(),
                &actions,
                &NumericModel {
                    fault: NumericFault::None
                },
                &PolicyGate::default()
            ),
            Err(PlannerError::InvalidInput(_))
        ));
    }
    // P5 stores return sums in f64, so finite f32::MAX rewards need not overflow.
    assert!(MctsEngine::default()
        .plan(
            &FullLatent::zeros(),
            &actions,
            &NumericModel {
                fault: NumericFault::RewardMax,
            },
            &PolicyGate::default(),
        )
        .is_ok());
}

#[test]
fn rollout_blocks_later_action_before_evaluating_its_invalid_transition() {
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::prohibit(
        RuleId(42),
        "blocked",
        ActionId(1),
    ));
    let pipeline = ProductionPipeline::new(
        Arc::new(NumericModel {
            fault: NumericFault::RewardNan,
        }),
        Arc::new(gate),
    );
    // The allowed first step is finite. Calling the blocked second step would
    // produce DivergentState instead of the gate's NoFeasibleAction.
    assert!(matches!(
        pipeline.simulate(&FullLatent::zeros(), &[ActionId(2), ActionId(1)], None),
        Err(PlannerError::NoFeasibleAction)
    ));
}

#[test]
fn unanimous_committee_reports_zero_vote_entropy_despite_uncertain_input() {
    let pipeline = ProductionPipeline::new(
        Arc::new(NumericModel {
            fault: NumericFault::None,
        }),
        Arc::new(PolicyGate::default()),
    )
    .with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
        s.as_slice()[1] >= 1.0
    }));
    let state = FullLatent::zeros();
    let mut request = decide_request(&state, &[ActionId(2)]);
    request.mode = DecideMode::Auto;
    request.entropy = gen_zero_core::NormalizedEntropy::ONE;
    let decision = pipeline.decide(&request).unwrap();
    assert_eq!(
        decision.routing_tier,
        Some(gen_zero_planner::RoutingTier::K3Committee)
    );
    assert_eq!(decision.entropy, gen_zero_core::NormalizedEntropy::ZERO);
}

#[test]
fn duplicate_actions_cannot_inflate_policy_mass_or_entropy() {
    let actions = frame(&[ActionId(1), ActionId(1), ActionId(2)]);
    let engines: [&dyn PlanningEngine; 6] = [
        &AStarEngine::default(),
        &MctsEngine::default(),
        &MpcCemEngine::default(),
        &ManifoldGFlowNetEngine,
        &CfrNashEngine,
        &CpSatFormalEngine,
    ];
    for engine in engines {
        assert!(matches!(
            engine.plan(
                &FullLatent::zeros(),
                &actions,
                &NumericModel {
                    fault: NumericFault::None
                },
                &PolicyGate::default()
            ),
            Err(PlannerError::InvalidInput(_))
        ));
    }
}

#[test]
fn direct_router_rejects_invalid_entropy_instead_of_rerouting() {
    let actions = frame(&ACTIONS);
    let router = gen_zero_planner::DynamicKMoERouter::default();
    for entropy in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(matches!(
            router.dispatch(
                &FullLatent::zeros(),
                &actions,
                gen_zero_core::NormalizedEntropy(entropy),
                &NumericModel {
                    fault: NumericFault::None
                },
                &PolicyGate::default()
            ),
            Err(PlannerError::InvalidInput(_))
        ));
    }
}
