use gen_zero_core::{ActionId, FullLatent, NormalizedEntropy};
use gen_zero_gate::{LinearConstraint, PolicyGate, RuleId};
use gen_zero_planner::triad::{
    shortest_first_order, CausalDag, CausalDagSpec, CausalGate, DisturbanceModel, GateContext,
    RobustObjective, RobustSlackSelector, ScoreScratch,
};
use gen_zero_planner::{
    CausalTriadRequest, DecideMode, DecideRequest, PlannerError, ProductionPipeline,
};
use gen_zero_worldmodel::LatentDynamicsWorldModel;
use serde_json::json;
use std::sync::Arc;

#[test]
fn huge_finite_prior_retains_probability_mass_and_moments() {
    let model = DisturbanceModel::new(vec![0.5, 0.5], None, 0, f64::MAX, "boundary").unwrap();
    let (post, bad) = model.posterior(&[0, 1, 1]);
    assert_eq!(bad, 0);
    assert_eq!(post.extra_pmf(), &[0.5, 0.5]);
    assert_eq!(post.moments().mean, 0.5);
    assert_eq!(post.moments().variance, 0.25);
    assert_eq!(post.moments().kappa, Some(f64::MAX));
}

#[test]
fn forged_verdict_and_repeated_target_are_refused() {
    let spec: CausalDagSpec = serde_json::from_value(
        json!({"cost":{"1":1,"2":1},"parents":{"2":[1]},"target":2,"budget":4}),
    )
    .unwrap();
    let dag = CausalDag::from_spec(&[ActionId(1), ActionId(2)], &spec).unwrap();
    let gate = CausalGate::new(
        &dag,
        GateContext {
            done: 0,
            time_used: 0,
            blocked: 0,
            blocked_first: 0,
        },
    );
    let mut verdict = gate.check(&[0, 1]);
    verdict.net_reward = 100.0;
    let selector = RobustSlackSelector::new(
        DisturbanceModel::new(vec![1.0], None, 0, 1.0, "boundary").unwrap(),
        RobustObjective::PSuccess,
        true,
    );
    assert!(selector
        .select(&gate, &[verdict], &mut ScoreScratch::default())
        .is_err());
    assert!(shortest_first_order(&gate, &[0, 1, 1], &mut Vec::new()).is_err());
}

#[test]
fn robust_probe_cannot_move_a_policy_blocked_action_into_the_plan() {
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::prohibit(
        RuleId(17),
        "blocked",
        ActionId(1),
    ));
    let pipeline = ProductionPipeline::new(
        Arc::new(LatentDynamicsWorldModel::default()),
        Arc::new(gate),
    );
    let state = FullLatent::zeros();
    let request = DecideRequest {
        state:&state,candidates:&[ActionId(1),ActionId(2)],mode:DecideMode::CausalTriad,
        entropy:NormalizedEntropy(0.2),active_context:Vec::new(),deadline:None,budget_ms:None,
        return_trajectory:false,horizon:2,
        causal_triad:Some(CausalTriadRequest {
            dag:serde_json::from_value(json!({"cost":{"1":1,"2":1},"parents":{"2":[1]},"target":2,"budget":4})).unwrap(),
            options:serde_json::from_value(json!({"seed":3,"n_samples":16,"robust":{"model":{"extra_pmf":[1.0],"fatigue_frac":0.5,"fatigue_extra":1},"probe":true}})).unwrap(),
        }),
    };
    assert!(matches!(
        pipeline.decide(&request),
        Err(PlannerError::CausalGateEmpty { .. } | PlannerError::CausalInfeasible(_))
    ));
}

#[test]
fn root_state_mask_constrains_probe_but_allows_later_execution() {
    use gen_zero_planner::{MaskedDynamics, StateActionMask};
    // Domain precondition over the real residual dynamics: only action 2 is
    // admissible at the zero state; a real transition unlocks the other actions.
    struct RequiresNonzeroState;
    impl StateActionMask for RequiresNonzeroState {
        fn allowed_actions(&self, state: &[f32], ids: &[u32]) -> Vec<u32> {
            ids.iter()
                .copied()
                .filter(|&id| id == 2 || state.iter().any(|&v| v != 0.0))
                .collect()
        }
    }
    for mode in [DecideMode::CausalTriad, DecideMode::TournamentTriad] {
        let pipeline = ProductionPipeline::new(
            Arc::new(MaskedDynamics::new(
                LatentDynamicsWorldModel::default(),
                Arc::new(RequiresNonzeroState),
            )),
            Arc::new(PolicyGate::default()),
        );
        let state = FullLatent::zeros();
        let mut request = DecideRequest {
            state:&state,candidates:&[ActionId(1),ActionId(2),ActionId(3)],mode,
            entropy:NormalizedEntropy(0.2),active_context:Vec::new(),deadline:None,budget_ms:None,
            return_trajectory:true,horizon:3,
            causal_triad:Some(CausalTriadRequest {
                dag:serde_json::from_value(json!({"cost":{"1":1,"2":3,"3":1},"parents":{"3":[1,2]},"target":3,"budget":10})).unwrap(),
                options:serde_json::from_value(json!({"seed":3,"n_samples":16,"robust":{"model":{"extra_pmf":[1.0],"fatigue_frac":0.5,"fatigue_extra":1},"probe":true}})).unwrap(),
            }),
        };
        let decision = pipeline.decide(&request).unwrap();
        let report = decision.triad.unwrap();
        assert_eq!(
            report.chosen_path,
            vec![ActionId(2), ActionId(1), ActionId(3)]
        );
        assert_eq!(report.robust.unwrap().probe_action, Some(ActionId(2)));
        assert_eq!(decision.trajectory.unwrap().steps.len(), 3);
        request.deadline = Some(std::time::Instant::now());
        assert!(matches!(
            pipeline.decide(&request),
            Err(PlannerError::TimeoutExceeded(_))
        ));
    }
}
