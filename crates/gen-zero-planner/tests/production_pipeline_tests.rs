//! Integration tests: `ProductionPipeline` over the real `LatentDynamicsWorldModel`
//! and a real `PolicyGate`.
//!
//! The trap state is built from the model's own action encoding
//! (`delta_i = 0.05 * sin(0.05 i + 0.17 a)`): a latent aligned with action 0 is pushed
//! past `DONE_NORM` by action 0 and pulled back by action 18. Each trap test first
//! checks that geometry with a direct `model.step`, so a change to the model fails
//! the precondition loudly instead of passing by accident.

use gen_zero_core::{ActionId, CoreError, FullLatent, NormalizedEntropy, WorldModelDynamics};
use gen_zero_gate::{Budget, LinearConstraint, PolicyGate, PolicyTier, RuleId, SheafProblem};
use gen_zero_lod::{EpistemicStatus, LodBand, LodGraph, LodNode, MixedCurvatureCoord};
use gen_zero_planner::{
    AuditVerdict, DecideMode, DecideRequest, PlannerError, ProductionPipeline, RoutingTier,
    DEFAULT_WARN_RISK, MAX_HORIZON, MAX_WHAT_IF_CANDIDATES,
};
use gen_zero_worldmodel::{LatentDynamicsWorldModel, DONE_NORM, SAFETY_SOURCE_NORM_MARGIN};
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;

const LETHAL: ActionId = ActionId(0);
const ESCAPE: ActionId = ActionId(18);
const ACTS: [ActionId; 3] = [ActionId(1), ActionId(2), ActionId(3)];
const ALL_MODES: [DecideMode; 7] = [
    DecideMode::Auto,
    DecideMode::Mcts,
    DecideMode::MpcCem,
    DecideMode::AStar,
    DecideMode::ManifoldGFlowNet,
    DecideMode::CfrNash,
    DecideMode::Reflex,
];

fn pipeline(gate: PolicyGate) -> ProductionPipeline {
    ProductionPipeline::new(
        Arc::new(LatentDynamicsWorldModel::default()),
        Arc::new(gate),
    )
    .with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
        s.l2_norm() > 0.0
    }))
}

#[test]
fn graph_revocation_prunes_candidate_before_engine_dispatch() {
    let graph = Arc::new(LodGraph::new());
    graph.add_node(
        LodNode::new(
            0,
            LodBand::Lod0Atomic,
            MixedCurvatureCoord::origin(),
            "revoked",
            2,
        )
        .with_status(EpistemicStatus::Falsified),
    );
    let p = pipeline(PolicyGate::default()).with_graph(graph);
    let state = FullLatent::zeros();
    assert_eq!(
        p.decide(&decide_req(&state, &[ActionId(2)], DecideMode::Reflex))
            .unwrap_err(),
        PlannerError::NoFeasibleAction
    );
}

#[test]
fn heat_certificate_failure_prunes_high_risk_action() {
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
    gate.register_confirm_action(ActionId(2));
    gate.require_heat_certificate(ActionId(2), problem, Budget::default());
    let state = FullLatent::zeros();
    assert_eq!(
        pipeline(gate)
            .decide(&decide_req(&state, &[ActionId(2)], DecideMode::Reflex))
            .unwrap_err(),
        PlannerError::NoFeasibleAction
    );
}

fn prohibiting(actions: &[ActionId]) -> PolicyGate {
    let mut gate = PolicyGate::default();
    for (i, &a) in actions.iter().enumerate() {
        gate.add_constraint(LinearConstraint::prohibit(
            RuleId(900 + i as u32),
            "test_prohibit",
            a,
        ));
    }
    gate
}

/// Latent aligned with action 0's perturbation, scaled so action 0 lands at norm 101.
fn trap_state() -> FullLatent {
    let mut u = FullLatent::zeros();
    for (i, x) in u.as_mut_slice().iter_mut().enumerate() {
        *x = (i as f32 * 0.05).sin();
    }
    let c = (101.0 / u.l2_norm() - 0.05) / 0.95;
    u.as_mut_slice().iter_mut().for_each(|x| *x *= c);
    let model = LatentDynamicsWorldModel::default();
    let (_, _, lethal_done) = model.step(&u, LETHAL).unwrap();
    let (escape_next, _, escape_done) = model.step(&u, ESCAPE).unwrap();
    assert!(lethal_done, "precondition: action 0 must cross DONE_NORM");
    assert!(
        !escape_done && escape_next.l2_norm() < DONE_NORM,
        "precondition: action 18 must stay inside"
    );
    u
}

fn nan_state() -> FullLatent {
    FullLatent {
        values: [f32::NAN; 1024],
    }
}

/// Finite coordinates whose squared norm overflows f32.
fn overflow_state() -> FullLatent {
    FullLatent {
        values: [1e30; 1024],
    }
}

/// Every state already past the boundary: the first step is terminal.
fn doomed_state() -> FullLatent {
    FullLatent {
        values: [3.4; 1024],
    }
}

fn decide_req<'a>(
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
        horizon: 4,
    }
}

// ---------------------------------------------------------------------------
// simulate
// ---------------------------------------------------------------------------

#[test]
fn simulate_replays_the_plan_through_the_real_world_model() {
    let p = pipeline(PolicyGate::default());
    let s0 = FullLatent::zeros();
    let roll = p.simulate(&s0, &ACTS, None).unwrap();

    assert_eq!(roll.steps_simulated(), 3);
    assert_eq!(roll.survival_horizon, 3);
    assert!(!roll.terminated_early && roll.is_safe());
    assert_eq!(roll.termination_step, None);
    assert_eq!(roll.continuation_policy, "fixed_plan");

    // Every step equals the direct model transition chained by hand.
    let model = LatentDynamicsWorldModel::default();
    let mut s = s0;
    let mut total = 0.0;
    for (k, step) in roll.steps.iter().enumerate() {
        let (next, r, d) = model.step(&s, ACTS[k]).unwrap();
        assert_eq!(step.step_idx, k + 1);
        assert_eq!(step.action, ACTS[k]);
        assert_eq!(step.state.as_slice(), next.as_slice());
        assert_eq!(step.reward, r);
        assert_eq!(step.done, d);
        assert!(!step.hazard);
        assert_eq!(step.gate_tier, PolicyTier::Tier0Proceed);
        let sp = step.safe_prob.unwrap();
        assert!((sp - (1.0 - next.l2_norm() / DONE_NORM)).abs() < 1e-6);
        total += r;
        s = next;
    }
    assert!((roll.cumulative_return - total).abs() < 1e-6);
    assert_eq!(roll.final_state.as_slice(), s.as_slice());
    assert_eq!(roll.safety_coverage, 3);
    assert_eq!(roll.safety_calibrated, Some(false));
    assert_eq!(roll.safety_sources, vec![SAFETY_SOURCE_NORM_MARGIN]);
}

#[test]
fn simulate_truncates_to_a_shorter_horizon() {
    let p = pipeline(PolicyGate::default());
    let roll = p.simulate(&FullLatent::zeros(), &ACTS, Some(2)).unwrap();
    assert_eq!(roll.steps_simulated(), 2);
    assert_eq!(roll.survival_horizon, 2);
}

#[test]
fn simulate_stops_at_death_and_reports_it() {
    let p = pipeline(PolicyGate::default());
    let roll = p.simulate(&doomed_state(), &ACTS, None).unwrap();
    assert_eq!(roll.steps_simulated(), 1);
    assert!(roll.steps[0].done && roll.steps[0].hazard);
    assert_eq!(roll.steps[0].safe_prob, Some(0.0));
    assert_eq!(roll.survival_horizon, 0);
    assert!(roll.terminated_early);
    assert_eq!(roll.termination_step, Some(1));
    assert_eq!(roll.first_hazard_step, Some(1));
    assert!(!roll.is_safe());
}

#[test]
fn simulate_rejects_gate_blocked_steps() {
    let p = pipeline(prohibiting(&[ActionId(2)]));
    assert_eq!(
        p.simulate(&FullLatent::zeros(), &ACTS, None).unwrap_err(),
        PlannerError::NoFeasibleAction
    );
}

#[test]
fn simulate_refuses_bad_horizons_and_empty_plans() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    assert!(matches!(
        p.simulate(&s, &ACTS, Some(4)),
        Err(PlannerError::InvalidInput(_))
    ));
    assert!(matches!(
        p.simulate(&s, &ACTS, Some(0)),
        Err(PlannerError::InvalidHorizon { horizon: 0, .. })
    ));
    let long = vec![ActionId(1); MAX_HORIZON + 1];
    assert!(matches!(
        p.simulate(&s, &long, None),
        Err(PlannerError::InvalidHorizon { .. })
    ));
    assert!(matches!(
        p.simulate(&s, &[], None),
        Err(PlannerError::InvalidInput(_))
    ));
}

// ---------------------------------------------------------------------------
// what_if
// ---------------------------------------------------------------------------

#[test]
fn what_if_detects_the_trap_and_picks_the_escape() {
    let p = pipeline(PolicyGate::default());
    let report = p.what_if(&trap_state(), &[LETHAL, ESCAPE], 5).unwrap();

    assert_eq!(report.best_candidate, ESCAPE);
    assert_eq!(report.traps_detected, vec![LETHAL]);
    assert_eq!(report.safety_ranking, vec![ESCAPE, LETHAL]);
    assert!(!report.all_candidates_trapped);
    assert!(report.gate_blocked.is_empty());
    assert_eq!(report.horizon, 5);

    let lethal = &report.outcomes[0];
    assert_eq!(lethal.action, LETHAL);
    assert_eq!(lethal.rollout.survival_horizon, 0);
    assert!(lethal.rollout.terminated_early);
    assert_eq!(lethal.rollout.first_hazard_step, Some(1));

    let escape = &report.outcomes[1];
    assert!(escape.rollout.is_safe());
    assert_eq!(escape.rollout.survival_horizon, 5);
    assert!(!escape.rollout.terminated_early);
    // Greedy continuation must not walk back into the trap.
    assert!(escape.rollout.steps.iter().all(|s| s.action == ESCAPE));
    assert_eq!(
        escape.rollout.continuation_policy,
        "greedy_one_step_over_candidates"
    );
}

#[test]
fn what_if_flags_every_candidate_when_all_are_trapped() {
    let p = pipeline(PolicyGate::default());
    let report = p.what_if(&doomed_state(), &ACTS, 3).unwrap();
    assert!(report.all_candidates_trapped);
    assert_eq!(report.traps_detected, ACTS.to_vec());
}

#[test]
fn what_if_excludes_gate_blocked_candidates() {
    let p = pipeline(prohibiting(&[ESCAPE]));
    let report = p.what_if(&trap_state(), &[LETHAL, ESCAPE], 3).unwrap();
    assert_eq!(report.outcomes.len(), 1);
    assert_eq!(report.best_candidate, LETHAL);
    assert!(report.all_candidates_trapped);
    assert_eq!(report.gate_blocked.len(), 1);
    assert_eq!(report.gate_blocked[0].action, ESCAPE);
    assert_eq!(report.gate_blocked[0].violated_rules, vec![900]);
    assert_eq!(report.gate_blocked[0].tier, PolicyTier::Tier3HardStop);
}

#[test]
fn what_if_fails_closed_on_empty_blocked_or_malformed_candidates() {
    let s = FullLatent::zeros();
    assert_eq!(
        pipeline(prohibiting(&ACTS))
            .what_if(&s, &ACTS, 3)
            .unwrap_err(),
        PlannerError::NoFeasibleAction
    );
    let p = pipeline(PolicyGate::default());
    assert_eq!(
        p.what_if(&s, &[], 3).unwrap_err(),
        PlannerError::NoFeasibleAction
    );
    assert!(matches!(
        p.what_if(&s, &[ActionId(1), ActionId(1)], 3),
        Err(PlannerError::InvalidInput(_))
    ));
    let many: Vec<ActionId> = (0..=MAX_WHAT_IF_CANDIDATES as u32).map(ActionId).collect();
    assert!(matches!(
        p.what_if(&s, &many, 3),
        Err(PlannerError::InvalidInput(_))
    ));
}

// ---------------------------------------------------------------------------
// audit_action
// ---------------------------------------------------------------------------

#[test]
fn audit_approves_a_calm_action_and_says_the_estimate_is_uncalibrated() {
    let p = pipeline(PolicyGate::default());
    let report = p
        .audit_action(
            &FullLatent::zeros(),
            ActionId(1),
            5,
            None,
            DEFAULT_WARN_RISK,
        )
        .unwrap();
    assert_eq!(report.verdict, AuditVerdict::Approved);
    assert_eq!(report.verdict.as_str(), "Approved");
    assert!(report.risk_score > 0.0 && report.risk_score < DEFAULT_WARN_RISK);
    assert!(report.is_safe);
    assert_eq!(report.survival_horizon, 5);
    let roll = report.trajectory.as_ref().unwrap();
    assert!(roll.steps.iter().all(|s| s.action == ActionId(1)));
    assert_eq!(roll.continuation_policy, "repeat_audited_action");
    assert!((report.risk_score - (1.0 - roll.min_safe_prob.unwrap())).abs() < 1e-6);
    assert!(report.reasons.iter().any(|r| r.contains("uncalibrated")));
}

#[test]
fn audit_rejects_the_trap_action_as_lethal() {
    let p = pipeline(PolicyGate::default());
    let report = p
        .audit_action(&trap_state(), LETHAL, 5, None, DEFAULT_WARN_RISK)
        .unwrap();
    assert_eq!(report.verdict, AuditVerdict::RejectLethal);
    assert_eq!(report.first_hazard_step, Some(1));
    assert_eq!(report.survival_horizon, 0);
    assert_eq!(report.risk_score, 1.0);
    assert!(report.reasons[0].contains("audited action itself"));
}

#[test]
fn audit_warns_when_the_state_hugs_the_boundary() {
    // Escape survives, but its first step sits at norm ~98.75: risk ~0.99.
    let p = pipeline(PolicyGate::default());
    let report = p
        .audit_action(&trap_state(), ESCAPE, 5, None, DEFAULT_WARN_RISK)
        .unwrap();
    assert_eq!(report.verdict, AuditVerdict::WarnHazard);
    assert!(report.is_safe);
    assert!(report.risk_score >= DEFAULT_WARN_RISK);
}

#[test]
fn audit_rejects_a_gate_hard_stop_without_imagining_it() {
    let p = pipeline(prohibiting(&[ActionId(1)]));
    let report = p
        .audit_action(
            &FullLatent::zeros(),
            ActionId(1),
            5,
            None,
            DEFAULT_WARN_RISK,
        )
        .unwrap();
    assert_eq!(report.verdict, AuditVerdict::RejectLethal);
    assert_eq!(report.gate_tier, PolicyTier::Tier3HardStop);
    assert_eq!(report.risk_score, 1.0);
    assert!(report.trajectory.is_none());
    assert!(report.reasons[0].contains("900"));
}

#[test]
fn audit_warns_on_a_confirm_tier_action() {
    let mut gate = PolicyGate::default();
    gate.register_confirm_action(ActionId(1));
    let report = pipeline(gate)
        .audit_action(
            &FullLatent::zeros(),
            ActionId(1),
            3,
            None,
            DEFAULT_WARN_RISK,
        )
        .unwrap();
    assert_eq!(report.verdict, AuditVerdict::WarnHazard);
    assert_eq!(report.gate_tier, PolicyTier::Tier1Confirm);
}

#[test]
fn audit_continuation_follows_the_greedy_policy_and_drops_blocked_actions() {
    let p = pipeline(prohibiting(&[ActionId(3)]));
    let report = p
        .audit_action(
            &trap_state(),
            ESCAPE,
            4,
            Some(&[LETHAL, ESCAPE, ActionId(3)]),
            1.0,
        )
        .unwrap();
    assert_eq!(report.continuation_pruned.len(), 1);
    assert_eq!(report.continuation_pruned[0].action, ActionId(3));
    let roll = report.trajectory.unwrap();
    assert_eq!(roll.continuation_policy, "greedy_one_step_over_candidates");
    assert!(roll.steps.iter().all(|s| s.action != ActionId(3)));
    assert!(roll.is_safe());

    let all_blocked = p.audit_action(
        &FullLatent::zeros(),
        ActionId(1),
        4,
        Some(&[ActionId(3)]),
        0.3,
    );
    assert_eq!(all_blocked.unwrap_err(), PlannerError::NoFeasibleAction);
}

#[test]
fn audit_refuses_invalid_warn_risk() {
    let p = pipeline(PolicyGate::default());
    for bad in [0.0, -0.1, 1.5, f32::NAN] {
        assert!(matches!(
            p.audit_action(&FullLatent::zeros(), ActionId(1), 3, None, bad),
            Err(PlannerError::InvalidInput(_))
        ));
    }
}

/// A simulator that exposes transitions but no safety reading, like an external
/// caller-supplied environment. Audit must refuse to score it, not treat it as safe.
struct NoSafetyModel(LatentDynamicsWorldModel);

impl WorldModelDynamics for NoSafetyModel {
    type Error = CoreError;
    fn step(&self, s: &FullLatent, a: ActionId) -> Result<(FullLatent, f32, bool), CoreError> {
        self.0.step(s, a)
    }
    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next_states: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), CoreError> {
        self.0
            .step_batch(states, actions, next_states, rewards, dones)
    }
}

#[test]
fn audit_refuses_a_model_without_safety_estimates() {
    let p = ProductionPipeline::new(
        Arc::new(NoSafetyModel(LatentDynamicsWorldModel::default())),
        Arc::new(PolicyGate::default()),
    );
    let err = p
        .audit_action(
            &FullLatent::zeros(),
            ActionId(1),
            3,
            None,
            DEFAULT_WARN_RISK,
        )
        .unwrap_err();
    assert_eq!(
        err,
        PlannerError::MissingSafetyEstimate {
            covered: 0,
            steps: 3
        }
    );
    // simulate still works and says plainly that no estimate exists.
    let roll = p.simulate(&FullLatent::zeros(), &ACTS, None).unwrap();
    assert_eq!(roll.min_safe_prob, None);
    assert_eq!(roll.safety_calibrated, None);
    assert!(roll.steps.iter().all(|s| s.safe_prob.is_none()));
}

// ---------------------------------------------------------------------------
// decide
// ---------------------------------------------------------------------------

#[test]
fn decide_runs_every_mode_on_its_own_engine() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    let expected = [
        (DecideMode::Auto, "DynamicKMoERouter"),
        (DecideMode::Mcts, "MctsEngine"),
        (DecideMode::MpcCem, "MpcCemEngine"),
        (DecideMode::AStar, "AStarEngine"),
        (DecideMode::ManifoldGFlowNet, "ManifoldGFlowNetEngine"),
        (DecideMode::CfrNash, "CfrNashEngine"),
        (DecideMode::Reflex, "CpSatFormalEngine"),
    ];
    for (mode, engine) in expected {
        let d = p.decide(&decide_req(&s, &ACTS, mode)).unwrap();
        assert_eq!(d.engine, engine, "{mode:?}");
        assert_eq!(d.mode, mode);
        assert!(ACTS.contains(&d.action));
        assert!((0.0..=1.0).contains(&d.entropy.0));
        assert_eq!(d.feasible, ACTS.to_vec());
        assert!(d.pruned.is_empty());
        assert!(d.trajectory.is_none());
        assert_eq!(d.routing_tier.is_some(), mode == DecideMode::Auto);
        assert!(!d.requires_confirmation);
    }
}

#[test]
fn decide_mode_parses_all_names_and_refuses_others() {
    for m in ALL_MODES {
        assert_eq!(m.as_str().parse::<DecideMode>().unwrap(), m);
    }
    assert_eq!(
        "gflownet".parse::<DecideMode>().unwrap(),
        DecideMode::ManifoldGFlowNet
    );
    assert_eq!("cfr".parse::<DecideMode>().unwrap(), DecideMode::CfrNash);
    assert_eq!(
        "greedy".parse::<DecideMode>().unwrap_err(),
        PlannerError::UnknownMode("greedy".into())
    );
}

#[test]
fn decide_auto_routes_by_entropy() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    for (h, tier) in [
        (0.1, RoutingTier::K1Reflex),
        (0.5, RoutingTier::K2Pipeline),
        (0.85, RoutingTier::K3Committee),
    ] {
        let mut req = decide_req(&s, &ACTS, DecideMode::Auto);
        req.entropy = NormalizedEntropy(h);
        assert_eq!(
            p.decide(&req).unwrap().routing_tier,
            Some(tier),
            "entropy {h}"
        );
    }
}

#[test]
fn decide_prunes_hard_stops_before_any_engine_runs() {
    // The gate removes action 3; no engine may return it.
    let p = pipeline(prohibiting(&[ActionId(3)]));
    let s = FullLatent::zeros();
    for mode in ALL_MODES {
        let d = p.decide(&decide_req(&s, &ACTS, mode)).unwrap();
        assert_ne!(d.action, ActionId(3), "{mode:?}");
        assert_eq!(d.feasible, vec![ActionId(1), ActionId(2)]);
        assert_eq!(d.pruned.len(), 1);
        assert_eq!(d.pruned[0].action, ActionId(3));
        assert_eq!(d.pruned[0].violated_rules, vec![900]);
    }
}

#[test]
fn decide_fails_closed_when_no_action_is_legal() {
    let p = pipeline(prohibiting(&ACTS));
    let s = FullLatent::zeros();
    for mode in ALL_MODES {
        assert_eq!(
            p.decide(&decide_req(&s, &ACTS, mode)).unwrap_err(),
            PlannerError::NoFeasibleAction,
            "{mode:?}"
        );
    }
    let open = pipeline(PolicyGate::default());
    assert_eq!(
        open.decide(&decide_req(&s, &[], DecideMode::Auto))
            .unwrap_err(),
        PlannerError::NoFeasibleAction
    );
}

#[test]
fn decide_returns_the_chosen_actions_trajectory_on_request() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    for mode in ALL_MODES {
        let mut req = decide_req(&s, &ACTS, mode);
        req.return_trajectory = true;
        req.horizon = 6;
        let d = p.decide(&req).unwrap();
        let roll = d.trajectory.expect("trajectory requested");
        assert_eq!(roll.steps_simulated(), 6);
        assert_eq!(roll.steps[0].action, d.action, "{mode:?}");
        assert!(roll.steps.iter().all(|st| d.feasible.contains(&st.action)));
    }
}

#[test]
fn decide_reports_a_confirm_tier_choice() {
    let mut gate = PolicyGate::default();
    gate.register_confirm_action(ActionId(2));
    let p = pipeline(gate);
    let s = FullLatent::zeros();
    let d = p
        .decide(&decide_req(&s, &[ActionId(2)], DecideMode::Reflex))
        .unwrap();
    assert!(d.requires_confirmation);
    assert_eq!(d.gate_tier, PolicyTier::Tier1Confirm);
}

#[test]
fn decide_refuses_malformed_requests() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    let dup = [ActionId(1), ActionId(1)];
    assert!(matches!(
        p.decide(&decide_req(&s, &dup, DecideMode::Auto)),
        Err(PlannerError::InvalidInput(_))
    ));
    let many: Vec<ActionId> = (0..17).map(ActionId).collect();
    assert!(matches!(
        p.decide(&decide_req(&s, &many, DecideMode::Auto)),
        Err(PlannerError::InvalidInput(_))
    ));
    let mut req = decide_req(&s, &ACTS, DecideMode::Auto);
    req.entropy = NormalizedEntropy(1.5);
    assert!(matches!(p.decide(&req), Err(PlannerError::InvalidInput(_))));
    let mut req = decide_req(&s, &ACTS, DecideMode::Auto);
    req.return_trajectory = true;
    req.horizon = 0;
    assert!(matches!(
        p.decide(&req),
        Err(PlannerError::InvalidHorizon { .. })
    ));
}

// ---------------------------------------------------------------------------
// Divergent-state interception, shared by every method
// ---------------------------------------------------------------------------

#[test]
fn every_method_refuses_a_divergent_state() {
    let p = pipeline(PolicyGate::default());
    for bad in [nan_state(), overflow_state()] {
        assert!(matches!(
            p.simulate(&bad, &ACTS, None),
            Err(PlannerError::DivergentState(_))
        ));
        assert!(matches!(
            p.what_if(&bad, &ACTS, 3),
            Err(PlannerError::DivergentState(_))
        ));
        assert!(matches!(
            p.audit_action(&bad, ActionId(1), 3, None, DEFAULT_WARN_RISK),
            Err(PlannerError::DivergentState(_))
        ));
        for mode in ALL_MODES {
            assert!(matches!(
                p.decide(&decide_req(&bad, &ACTS, mode)),
                Err(PlannerError::DivergentState(_))
            ));
        }
    }
}

#[test]
fn decide_preserves_mutex_and_quota_context_in_every_mode() {
    let state = FullLatent::zeros();
    let candidates = [ActionId(2), ActionId(3)];
    for rule in [
        LinearConstraint::mutex(RuleId(71), "mutex", ActionId(1), ActionId(2)),
        LinearConstraint::quota(RuleId(71), "quota", &[ActionId(1), ActionId(2)], 2).unwrap(),
    ] {
        let active_context = if rule.rhs == 2 {
            vec![ActionId(1), ActionId(1)]
        } else {
            vec![ActionId(1)]
        };
        let mut gate = PolicyGate::default();
        gate.add_constraint(rule);
        let p = pipeline(gate);
        for mode in ALL_MODES {
            let mut req = decide_req(&state, &candidates, mode);
            req.active_context = active_context.clone();
            let decision = p.decide(&req).unwrap();
            assert_eq!(decision.action, ActionId(3), "{mode:?}");
            assert_eq!(decision.feasible, vec![ActionId(3)]);
            assert_eq!(decision.pruned[0].action, ActionId(2));
            assert_eq!(decision.pruned[0].violated_rules, vec![71]);
            req.candidates = &candidates[..1];
            assert_eq!(p.decide(&req).unwrap_err(), PlannerError::NoFeasibleAction);
        }
    }
}

struct HighRewardHazard;
impl WorldModelDynamics for HighRewardHazard {
    type Error = CoreError;
    fn step(
        &self,
        state: &FullLatent,
        action: ActionId,
    ) -> Result<(FullLatent, f32, bool), CoreError> {
        let mut next = state.clone();
        next.as_mut_slice()[0] += 1.0;
        Ok((
            next,
            if action == LETHAL { 1000.0 } else { 1.0 },
            action == LETHAL,
        ))
    }
    fn step_batch(
        &self,
        states: &[FullLatent],
        actions: &[ActionId],
        next: &mut [FullLatent],
        rewards: &mut [f32],
        dones: &mut [bool],
    ) -> Result<(), CoreError> {
        for i in 0..states.len() {
            (next[i], rewards[i], dones[i]) = self.step(&states[i], actions[i])?;
        }
        Ok(())
    }
}

#[test]
fn decide_excludes_high_reward_hazard_in_every_mode_without_trajectory() {
    let p = ProductionPipeline::new(Arc::new(HighRewardHazard), Arc::new(PolicyGate::default()))
        .with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
            s.as_slice()[0] >= 1.0
        }));
    let state = FullLatent::zeros();
    for mode in ALL_MODES {
        let decision = p
            .decide(&decide_req(&state, &[LETHAL, ESCAPE], mode))
            .unwrap();
        assert_eq!(decision.action, ESCAPE, "{mode:?}");
        assert!(decision.hazard_detected);
        assert_eq!(decision.hazardous_actions, vec![LETHAL]);
        assert_eq!(decision.feasible, vec![ESCAPE]);
        assert!(decision.trajectory.is_none());
        assert_eq!(
            p.decide(&decide_req(&state, &[LETHAL], mode)).unwrap_err(),
            PlannerError::NoFeasibleAction
        );
    }
}

#[test]
fn decide_excludes_real_worldmodel_divergence_in_every_mode() {
    let state = trap_state();
    let (target, _, _) = LatentDynamicsWorldModel::default()
        .step(&state, ESCAPE)
        .unwrap();
    let p = pipeline(PolicyGate::default()).with_astar_goal(
        gen_zero_planner::AStarGoal::WithinDistance {
            target: Box::new(target),
            tolerance: 1e-5,
        },
    );
    for mode in ALL_MODES {
        let decision = p
            .decide(&decide_req(&state, &[LETHAL, ESCAPE], mode))
            .unwrap();
        assert_eq!(decision.action, ESCAPE);
        assert!(decision.hazard_detected);
        assert_eq!(decision.hazardous_actions, vec![LETHAL]);
    }
}
