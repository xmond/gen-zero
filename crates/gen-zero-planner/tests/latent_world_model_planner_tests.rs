//! Integration tests: the real `LatentDynamicsWorldModel` driven through
//! `DynamicKMoERouter::dispatch` on every routing tier.
//!
//! Before the `From<WorldModelError> for CoreError` bridge this did not compile
//! (E0271: dispatch requires `Error = CoreError`). The failure tests pin the
//! fail-closed contract: a world-model error or an all-blocked frame must come
//! back as `Err`, never as a chosen action.

use gen_zero_core::{
    ActionId, FullLatent, LocalActionFrame, NormalizedEntropy, WorldModelDynamics,
};
use gen_zero_gate::{LinearConstraint, PolicyGate, RuleId};
use gen_zero_planner::{
    AStarEngine, CfrNashEngine, CpSatFormalEngine, DynamicKMoERouter, ManifoldGFlowNetEngine,
    MctsEngine, MpcCemEngine, PlannerError, PlanningEngine,
};
use gen_zero_worldmodel::LatentDynamicsWorldModel;

/// Entropies that route to K1 reflex, K2 pipeline and K3 committee.
const TIER_ENTROPIES: [f32; 3] = [0.1, 0.5, 0.85];

const NAMES: [&str; 3] = ["act1", "act2", "act3"];
const ACTS: [ActionId; 3] = [ActionId(1), ActionId(2), ActionId(3)];

fn plan(
    router: &DynamicKMoERouter,
    state: &FullLatent,
    frame: &LocalActionFrame<'_>,
    gate: PolicyGate,
    entropy: f32,
) -> Result<(ActionId, NormalizedEntropy), PlannerError> {
    router.dispatch(
        state,
        frame,
        NormalizedEntropy(entropy),
        &LatentDynamicsWorldModel::default(),
        &gate,
    )
}

fn nan_state() -> FullLatent {
    FullLatent {
        values: [f32::NAN; 1024],
    }
}

fn gate_blocking_all() -> PolicyGate {
    let mut gate = PolicyGate::default();
    for (i, &act) in ACTS.iter().enumerate() {
        gate.add_constraint(LinearConstraint::prohibit(
            RuleId(i as u32 + 1),
            "blocked",
            act,
        ));
    }
    gate
}

#[test]
fn latent_world_model_plans_through_router_on_every_tier() {
    let router =
        DynamicKMoERouter::default().with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
            s.l2_norm() > 0.0
        }));
    let frame = LocalActionFrame::new(&NAMES, &ACTS).unwrap();
    let state = FullLatent::zeros();
    for entropy in TIER_ENTROPIES {
        let (act, _) = plan(&router, &state, &frame, PolicyGate::default(), entropy)
            .unwrap_or_else(|e| panic!("entropy {entropy}: {e}"));
        assert!(
            ACTS.contains(&act),
            "entropy {entropy}: {act:?} not in frame"
        );
    }
}

#[test]
fn nonfinite_input_is_refused_on_every_tier() {
    let router =
        DynamicKMoERouter::default().with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
            s.l2_norm() > 0.0
        }));
    let frame = LocalActionFrame::new(&NAMES, &ACTS).unwrap();
    let state = nan_state();
    for entropy in TIER_ENTROPIES {
        let out = plan(&router, &state, &frame, PolicyGate::default(), entropy);
        assert!(
            matches!(
                out,
                Err(PlannerError::Core(_) | PlannerError::DivergentState(_))
            ),
            "entropy {entropy}: expected numerical rejection, got {out:?}"
        );
    }
}

#[test]
fn nonfinite_input_is_refused_by_every_engine() {
    let wm = LatentDynamicsWorldModel::default();
    let gate = PolicyGate::default();
    let frame = LocalActionFrame::new(&NAMES, &ACTS).unwrap();
    let state = nan_state();
    let engines: [&dyn PlanningEngine; 6] = [
        &MctsEngine::default(),
        &AStarEngine {
            goal: Some(gen_zero_planner::AStarGoal::Predicate(|s| {
                s.l2_norm() > 0.0
            })),
            ..Default::default()
        },
        &MpcCemEngine::default(),
        &ManifoldGFlowNetEngine,
        &CfrNashEngine,
        &CpSatFormalEngine,
    ];
    for engine in engines {
        let out = engine.plan(&state, &frame, &wm, &gate);
        assert!(
            matches!(
                out,
                Err(PlannerError::Core(_) | PlannerError::DivergentState(_))
            ),
            "{}: expected numerical rejection, got {out:?}",
            engine.name()
        );
    }
}

#[test]
fn all_blocked_frame_is_refused_by_every_engine_and_tier() {
    let wm = LatentDynamicsWorldModel::default();
    let gate = gate_blocking_all();
    let frame = LocalActionFrame::new(&NAMES, &ACTS).unwrap();
    let state = FullLatent::zeros();
    let engines: [&dyn PlanningEngine; 6] = [
        &MctsEngine::default(),
        &AStarEngine {
            goal: Some(gen_zero_planner::AStarGoal::Predicate(|s| {
                s.l2_norm() > 0.0
            })),
            ..Default::default()
        },
        &MpcCemEngine::default(),
        &ManifoldGFlowNetEngine,
        &CfrNashEngine,
        &CpSatFormalEngine,
    ];
    for engine in engines {
        let out = engine.plan(&state, &frame, &wm, &gate);
        assert_eq!(
            out,
            Err(PlannerError::NoFeasibleAction),
            "{}",
            engine.name()
        );
    }

    let router =
        DynamicKMoERouter::default().with_astar_goal(gen_zero_planner::AStarGoal::Predicate(|s| {
            s.l2_norm() > 0.0
        }));
    for entropy in TIER_ENTROPIES {
        let out = plan(&router, &state, &frame, gate_blocking_all(), entropy);
        assert_eq!(
            out,
            Err(PlannerError::NoFeasibleAction),
            "entropy {entropy}"
        );
    }
}

#[test]
fn engines_never_pick_a_blocked_action() {
    let wm = LatentDynamicsWorldModel::default();
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::prohibit(
        RuleId(1),
        "blocked",
        ActionId(1),
    ));
    let frame = LocalActionFrame::new(&NAMES, &ACTS).unwrap();
    let state = FullLatent::zeros();
    let engines: [&dyn PlanningEngine; 6] = [
        &MctsEngine::default(),
        &AStarEngine {
            goal: Some(gen_zero_planner::AStarGoal::Predicate(|s| {
                s.l2_norm() > 0.0
            })),
            ..Default::default()
        },
        &MpcCemEngine::default(),
        &ManifoldGFlowNetEngine,
        &CfrNashEngine,
        &CpSatFormalEngine,
    ];
    for engine in engines {
        let (act, _) = engine.plan(&state, &frame, &wm, &gate).unwrap();
        assert_ne!(
            act,
            ActionId(1),
            "{} picked the blocked action",
            engine.name()
        );
    }
    // Sanity: the model itself is healthy on this state.
    assert!(wm.step(&state, ActionId(1)).is_ok());
}
