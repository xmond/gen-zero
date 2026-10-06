//! Causal triad: DAG validation, on-trajectory pruner, deterministic gate,
//! geodesic sampler, and the `DecideMode::CausalTriad` production route over the
//! real `LatentDynamicsWorldModel` and a real `PolicyGate`.

use gen_zero_core::{ActionId, FullLatent, NormalizedEntropy, WorldModelDynamics};
use gen_zero_gate::{LinearConstraint, PolicyGate, RuleId};
use gen_zero_planner::triad::{
    arbiter_cmp, CausalDag, CausalDagSpec, CausalEdge, CausalGate, CausalNode, CausalPruner,
    EmptyActionSet, GateContext, GateReason, GeodesicFlowConfig, GeodesicFlowSampler, SamplerStart,
    TriadRunOptions,
};
use gen_zero_planner::{
    CausalTriadRequest, DecideMode, DecideRequest, PlannerError, ProductionPipeline,
};
use gen_zero_worldmodel::LatentDynamicsWorldModel;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;
use std::sync::Arc;

// Release workflow DAG. Node index == position in CANDS.
//   1 fetch(2) -> 2 build(3) -> 3 test(2) --\
//   1 fetch    -> 4 lint(1) ----------------+--> 5 deploy(1, AND, target)
//   2 build | 4 lint -> 7 package(1, OR) (not a deploy ancestor)
//   6 docs(1) is a free-standing source.
const CANDS: [ActionId; 7] = [
    ActionId(1),
    ActionId(2),
    ActionId(3),
    ActionId(4),
    ActionId(5),
    ActionId(6),
    ActionId(7),
];

fn spec_value() -> serde_json::Value {
    json!({
        "parents": {"2": [1], "3": [2], "4": [1], "5": [3, 4], "7": [2, 4]},
        "is_or": {"7": true},
        "cost": {"1": 2, "2": 3, "3": 2, "4": 1, "5": 1, "6": 1, "7": 1},
        "target": 5,
        "budget": 20
    })
}

fn spec() -> CausalDagSpec {
    serde_json::from_value(spec_value()).unwrap()
}

fn dag() -> CausalDag {
    CausalDag::from_spec(&CANDS, &spec()).unwrap()
}

fn idx(d: &CausalDag, a: u32) -> usize {
    d.index_of(ActionId(a)).unwrap()
}

fn path(d: &CausalDag, ids: &[u32]) -> Vec<usize> {
    ids.iter().map(|&a| idx(d, a)).collect()
}

fn ctx() -> GateContext {
    GateContext {
        done: 0,
        time_used: 0,
        blocked: 0,
        blocked_first: 0,
    }
}

fn invalid_msg(r: Result<CausalDag, PlannerError>) -> String {
    match r {
        Err(PlannerError::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

fn node(a: u32, cost: u32) -> CausalNode {
    CausalNode {
        action: ActionId(a),
        cost,
        value: 0.0,
        is_or: false,
    }
}

fn edge(p: u32, c: u32) -> CausalEdge {
    CausalEdge {
        parent: ActionId(p),
        child: ActionId(c),
    }
}

// ---------------------------------------------------------------------------
// DAG
// ---------------------------------------------------------------------------

#[test]
fn dag_builds_from_spec_with_a_valid_topological_order() {
    let d = dag();
    assert_eq!(d.len(), 7);
    assert_eq!(d.target(), idx(&d, 5));
    assert_eq!(d.budget(), 20);
    assert_eq!(d.parents(idx(&d, 5)), &[idx(&d, 3), idx(&d, 4)]);
    assert!(d.node(idx(&d, 7)).is_or);
    let topo = d.topological_order().to_vec();
    assert_eq!(topo.len(), 7);
    assert!(d.respects_topology(&topo));
    // deploy before its parent test breaks the order.
    assert!(!d.respects_topology(&path(&d, &[1, 2, 5, 3])));
    // A repeated node is not an order.
    assert!(!d.respects_topology(&path(&d, &[1, 1])));
    // Goal cone: deploy and its ancestors only; docs and package are outside.
    let cone = d.goal_cone();
    for a in [1, 2, 3, 4, 5] {
        assert_ne!(cone & (1 << idx(&d, a)), 0, "{a} in cone");
    }
    for a in [6, 7] {
        assert_eq!(cone & (1 << idx(&d, a)), 0, "{a} outside cone");
    }
}

#[test]
fn dag_refuses_cycles_fail_closed() {
    // 1 -> 2 -> 3 -> 1
    let nodes = vec![node(1, 1), node(2, 1), node(3, 1)];
    let m = invalid_msg(CausalDag::new(
        nodes,
        &[edge(1, 2), edge(2, 3), edge(3, 1)],
        ActionId(3),
        10,
    ));
    assert!(m.contains("cycle"), "{m}");

    // Same cycle through the wire spec.
    let mut v = spec_value();
    v["parents"]["1"] = json!([5]);
    let s: CausalDagSpec = serde_json::from_value(v).unwrap();
    let m = invalid_msg(CausalDag::from_spec(&CANDS, &s));
    assert!(m.contains("cycle"), "{m}");
}

#[test]
fn dag_refuses_every_structural_violation() {
    let cases: Vec<(serde_json::Value, &str)> = vec![
        (json!({"2": [2]}), "own parent"),
        (json!({"2": [99]}), "unknown action 99"),
        (json!({"99": [1]}), "not a candidate"),
    ];
    for (parents, want) in cases {
        let mut v = spec_value();
        v["parents"] = parents;
        let s: CausalDagSpec = serde_json::from_value(v).unwrap();
        let m = invalid_msg(CausalDag::from_spec(&CANDS, &s));
        assert!(m.contains(want), "{want}: {m}");
    }

    let mut v = spec_value();
    v["cost"].as_object_mut().unwrap().remove("6");
    let s: CausalDagSpec = serde_json::from_value(v).unwrap();
    assert!(invalid_msg(CausalDag::from_spec(&CANDS, &s)).contains("missing an entry for action 6"));

    let mut v = spec_value();
    v["cost"]["3"] = json!(0);
    let s: CausalDagSpec = serde_json::from_value(v).unwrap();
    assert!(invalid_msg(CausalDag::from_spec(&CANDS, &s)).contains("must be >= 1"));

    let mut v = spec_value();
    v["target"] = json!(42);
    let s: CausalDagSpec = serde_json::from_value(v).unwrap();
    assert!(invalid_msg(CausalDag::from_spec(&CANDS, &s)).contains("target"));

    let mut v = spec_value();
    v["budget"] = json!(0);
    let s: CausalDagSpec = serde_json::from_value(v).unwrap();
    assert!(invalid_msg(CausalDag::from_spec(&CANDS, &s)).contains("budget"));

    // Budgets above MAX_TRIAD_BUDGET are refused.
    let mut v = spec_value();
    v["budget"] = json!(u32::MAX);
    let s: CausalDagSpec = serde_json::from_value(v).unwrap();
    let m = invalid_msg(CausalDag::from_spec(&CANDS, &s));
    assert!(m.contains("must lie in 1..="), "{m}");
    assert!(CausalDag::new(vec![node(1, 1)], &[], ActionId(1), gen_zero_planner::triad::MAX_TRIAD_BUDGET).is_ok());

    let mut bad_value = node(1, 1);
    bad_value.value = f64::NAN;
    assert!(invalid_msg(CausalDag::new(vec![bad_value], &[], ActionId(1), 5)).contains("finite"));

    let m = invalid_msg(CausalDag::new(
        vec![node(1, 1), node(2, 1)],
        &[edge(1, 2), edge(1, 2)],
        ActionId(2),
        5,
    ));
    assert!(m.contains("duplicate edge"), "{m}");

    let m = invalid_msg(CausalDag::new(
        vec![node(1, 1), node(1, 1)],
        &[],
        ActionId(1),
        5,
    ));
    assert!(m.contains("duplicate action"), "{m}");

    let many: Vec<CausalNode> = (0..65).map(|a| node(a, 1)).collect();
    assert!(invalid_msg(CausalDag::new(many, &[], ActionId(0), 5)).contains("cap of 64"));
}

#[test]
fn spec_accepts_costs_and_values_aliases_and_refuses_both_spellings() {
    let mut v = spec_value();
    let cost = v.as_object_mut().unwrap().remove("cost").unwrap();
    v["costs"] = cost;
    v["values"] = json!({"5": 4.5});
    let s: CausalDagSpec = serde_json::from_value(v).unwrap();
    let d = CausalDag::from_spec(&CANDS, &s).unwrap();
    assert_eq!(d.node(idx(&d, 2)).cost, 3);
    assert_eq!(d.node(idx(&d, 5)).value, 4.5);
    assert_eq!(d.node(idx(&d, 1)).value, 0.0);

    let mut both = spec_value();
    both["costs"] = both["cost"].clone();
    let e = serde_json::from_value::<CausalDagSpec>(both).unwrap_err();
    assert!(e.to_string().contains("duplicate field"), "{e}");

    let mut unknown = spec_value();
    unknown["costz"] = json!({});
    assert!(serde_json::from_value::<CausalDagSpec>(unknown).is_err());
}

// ---------------------------------------------------------------------------
// Pruner
// ---------------------------------------------------------------------------

#[test]
fn pruner_filters_actions_whose_preconditions_are_unmet() {
    let d = dag();
    let p = CausalPruner::new(&d, false);
    let bit = |a| 1_u64 << idx(&d, a);
    // Nothing done: only the sources fetch and docs.
    assert_eq!(p.admissible_mask(0), bit(1) | bit(6));
    // fetch done: build and lint open; fetch itself is done.
    assert_eq!(p.admissible_mask(bit(1)), bit(2) | bit(4) | bit(6));
    // AND: deploy needs test AND lint.
    let partial = bit(1) | bit(2) | bit(3);
    assert!(!p.is_admissible(partial, idx(&d, 5)));
    assert!(p.is_admissible(partial | bit(4), idx(&d, 5)));
    // OR: package needs build OR lint.
    assert!(p.is_admissible(bit(1) | bit(4), idx(&d, 7)));
    assert!(p.is_admissible(bit(1) | bit(2), idx(&d, 7)));
    assert!(!p.is_admissible(bit(1), idx(&d, 7)));
}

#[test]
fn pruner_applies_policy_blocks_and_the_goal_cone() {
    let d = dag();
    let bit = |a| 1_u64 << idx(&d, a);
    let cone = CausalPruner::new(&d, true);
    // Goal cone drops docs (outside) but keeps fetch.
    let r = cone.prune(0, 0).unwrap();
    assert_eq!(r.allowed, bit(1));
    assert_eq!((r.n_admissible, r.n_unblocked, r.n_allowed), (2, 2, 1));
    // Package is admissible but outside the cone.
    let r = cone.prune(bit(1) | bit(4), 0).unwrap();
    assert_eq!(r.allowed, bit(2));
    // Policy block on build.
    let r = cone.prune(bit(1), bit(2)).unwrap();
    assert_eq!(r.allowed, bit(4));
    // Fail-closed: blocking fetch leaves nothing in the cone.
    assert_eq!(
        cone.prune(0, bit(1)).unwrap_err(),
        EmptyActionSet {
            done: 0,
            n_admissible: 2
        }
    );
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

#[test]
fn gate_verdicts_cover_every_reason() {
    let d = dag();
    let g = CausalGate::new(&d, ctx());

    let ok = g.check(&path(&d, &[1, 2, 3, 4, 5, 6]));
    assert!(ok.ok);
    assert_eq!(ok.reason, GateReason::Ok);
    assert_eq!(
        ok.path,
        path(&d, &[1, 2, 3, 4, 5]),
        "cut right after the target"
    );
    assert_eq!(ok.nominal_time, 9);
    assert_eq!(ok.net_reward, -9.0);
    assert_eq!(ok.first_bad_step, None);

    let v = g.check(&path(&d, &[1, 3]));
    assert_eq!(
        (v.reason, v.first_bad_step),
        (GateReason::Violation, Some(1))
    );
    // Deploy before lint: AND precondition unmet.
    let v = g.check(&path(&d, &[1, 2, 3, 5]));
    assert_eq!(
        (v.reason, v.first_bad_step),
        (GateReason::Violation, Some(3))
    );
    // Repeating a done action.
    let v = g.check(&path(&d, &[1, 1]));
    assert_eq!(v.reason, GateReason::Violation);
    // Out-of-range index.
    assert_eq!(g.check(&[99]).reason, GateReason::Violation);

    let v = g.check(&path(&d, &[1, 2, 3]));
    assert_eq!((v.reason, v.nominal_time), (GateReason::NoTarget, 7));

    let tight = CausalGate::new(
        &d,
        GateContext {
            time_used: 12,
            ..ctx()
        },
    );
    let v = tight.check(&path(&d, &[1, 2, 3, 4, 5]));
    assert_eq!((v.reason, v.nominal_time), (GateReason::OverBudget, 21));
}

#[test]
fn gate_rejects_a_clock_overflow_instead_of_saturating() {
    // 1 -> 2 (target). time_used + cost(1) fits; + cost(2) overflows u32.
    let d = CausalDag::new(
        vec![node(1, 1), node(2, u32::MAX)],
        &[edge(1, 2)],
        ActionId(2),
        1000,
    )
    .unwrap();
    let g = CausalGate::new(
        &d,
        GateContext {
            time_used: 10,
            ..ctx()
        },
    );
    let v = g.check(&[0, 1]);
    assert!(!v.ok);
    assert_eq!(
        (v.reason, v.first_bad_step, v.nominal_time),
        (GateReason::TimeOverflow, Some(1), 11),
        "the clock before the overflowing step is reported, never a saturated u32::MAX"
    );
    assert_eq!(v.reason.as_str(), "time_overflow");
    assert_eq!(v.net_reward, f64::NEG_INFINITY);
}

#[test]
fn gate_rejects_a_non_finite_reward_sum() {
    // Each value is finite (the DAG accepts it); their sum is +inf.
    let big = |a| CausalNode {
        value: f64::MAX,
        ..node(a, 1)
    };
    let d = CausalDag::new(vec![big(1), big(2)], &[edge(1, 2)], ActionId(2), 10).unwrap();
    let g = CausalGate::new(&d, ctx());
    let v = g.check(&[0, 1]);
    assert!(!v.ok);
    assert_eq!(
        (v.reason, v.first_bad_step),
        (GateReason::RewardNonFinite, Some(1))
    );
    assert_eq!(v.reason.as_str(), "reward_non_finite");
    // Nothing passes, so nothing non-finite can reach the arbiter's sort key.
    let (best, verdicts) = g.select(&[vec![0, 1]], None).unwrap();
    assert!(best.is_none());
    assert_eq!(verdicts[0].reason, GateReason::RewardNonFinite);

    // A single f64::MAX step stays finite and passes.
    let one = CausalDag::new(vec![big(1)], &[], ActionId(1), 10).unwrap();
    let v = CausalGate::new(&one, ctx()).check(&[0]);
    assert!(v.ok && v.net_reward.is_finite(), "{v:?}");
}

#[test]
fn gate_blocks_policy_actions_at_every_step_and_hazards_at_step_one_only() {
    let d = dag();
    let bit = |a| 1_u64 << idx(&d, a);
    let blocked = CausalGate::new(
        &d,
        GateContext {
            blocked: bit(4),
            ..ctx()
        },
    );
    let v = blocked.check(&path(&d, &[1, 2, 3, 4, 5]));
    assert_eq!((v.reason, v.first_bad_step), (GateReason::Blocked, Some(3)));

    let first = CausalGate::new(
        &d,
        GateContext {
            done: bit(1),
            blocked_first: bit(4),
            ..ctx()
        },
    );
    assert_eq!(
        first.check(&path(&d, &[4, 2, 3, 5])).reason,
        GateReason::Blocked
    );
    // The same action later in the plan is fine.
    assert!(first.check(&path(&d, &[2, 3, 4, 5])).ok);
}

#[test]
fn gate_is_deterministic_and_ranks_by_net_reward_then_time() {
    let d = dag();
    let g = CausalGate::new(&d, ctx());
    let p = path(&d, &[1, 4, 2, 3, 5]);
    let first = g.check(&p);
    for _ in 0..10_000 {
        assert_eq!(g.check(&p), first);
    }

    // All values 0: rank by nominal time (Python parity). Both orders take 9,
    // so the lexicographically smaller index path wins.
    let paths = vec![
        path(&d, &[1, 4, 2, 3, 5]),
        path(&d, &[1, 2, 3, 4, 5]),
        path(&d, &[1, 2, 3]),
        path(&d, &[1, 2, 3, 4, 7, 5]),
    ];
    let (best, verdicts) = g.select(&paths, None).unwrap();
    let best = best.unwrap();
    assert_eq!(best.path, path(&d, &[1, 2, 3, 4, 5]));
    assert_eq!(verdicts.len(), 4);
    assert_eq!(verdicts[2].reason, GateReason::NoTarget);
    // The detour through package costs one more unit.
    assert_eq!(verdicts[3].nominal_time, 10);
    assert!(arbiter_cmp(&verdicts[1], &verdicts[3]).is_lt());

    // Give package a value of 5: the detour now nets more and wins.
    let mut v = spec_value();
    v["value"] = json!({"7": 5.0});
    let valued = CausalDag::from_spec(&CANDS, &serde_json::from_value(v).unwrap()).unwrap();
    let (best, _) = CausalGate::new(&valued, ctx()).select(&paths, None).unwrap();
    let best = best.unwrap();
    assert_eq!(best.path, path(&d, &[1, 2, 3, 4, 7, 5]));
    assert_eq!(best.net_reward, 5.0 - 10.0);

    // Nothing passes: no best, never an unchecked fallback.
    let (none, _) = g.select(&[path(&d, &[1, 3]), vec![]], None).unwrap();
    assert!(none.is_none());
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

#[test]
fn sampler_proposes_only_admissible_unblocked_plans_and_is_seed_reproducible() {
    let d = dag();
    let bit = |a| 1_u64 << idx(&d, a);
    let s = GeodesicFlowSampler::new(&d, GeodesicFlowConfig::default()).unwrap();
    let start = SamplerStart {
        done: 0,
        time_used: 0,
        blocked: bit(6),
        blocked_first: 0,
    };
    let run = |seed| {
        s.sample(&mut StdRng::seed_from_u64(seed), 500, d.len(), &start, None)
            .unwrap()
    };
    let a = run(7);
    let b = run(7);
    assert_eq!(a.paths, b.paths, "same seed, same plans");
    assert_ne!(a.paths, run(8).paths, "different seed, different plans");

    let g = CausalGate::new(
        &d,
        GateContext {
            blocked: bit(6),
            ..ctx()
        },
    );
    let mut distinct = std::collections::BTreeSet::new();
    for p in &a.paths {
        let v = g.check(&p.actions);
        assert!(
            !matches!(v.reason, GateReason::Violation | GateReason::Blocked),
            "sampler proposed an inadmissible plan {:?}: {:?}",
            p.actions,
            v.reason
        );
        assert!(p.log_pf <= 0.0 && p.log_pf.is_finite());
        // With a 20-unit budget every cone-only plan reaches deploy in time.
        assert_eq!(v.reason, GateReason::Ok, "{:?}", p.actions);
        distinct.insert(p.actions.clone());
    }
    // Two orders exist (lint before or after build/test chains); the flow finds several.
    assert!(distinct.len() >= 2, "{distinct:?}");
}

#[test]
fn sampler_reports_clock_overrun_and_dead_ends() {
    let d = dag();
    let s = GeodesicFlowSampler::new(&d, GeodesicFlowConfig::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    // 15 of 20 units spent: the plan runs out of clock before deploy.
    let late = SamplerStart {
        done: 0,
        time_used: 15,
        blocked: 0,
        blocked_first: 0,
    };
    let set = s.sample(&mut rng, 20, d.len(), &late, None).unwrap();
    let g = CausalGate::new(
        &d,
        GateContext {
            time_used: 15,
            ..ctx()
        },
    );
    assert!(set
        .paths
        .iter()
        .all(|p| g.check(&p.actions).reason == GateReason::NoTarget));

    // Fetch blocked: the very first step dead-ends, nothing is invented.
    let dead = SamplerStart {
        blocked: 1 << idx(&d, 1),
        ..late
    };
    let set = s.sample(&mut rng, 5, d.len(), &dead, None).unwrap();
    assert_eq!(set.dead_ends, 5);
    assert!(set.paths.iter().all(|p| p.actions.is_empty()));

    // An expired deadline stops sampling with a timeout, not a partial set.
    let past = std::time::Instant::now() - std::time::Duration::from_millis(1);
    assert!(matches!(
        s.sample(&mut rng, 5, d.len(), &late, Some(past)),
        Err(PlannerError::TimeoutExceeded(_))
    ));
}

// ---------------------------------------------------------------------------
// Production route: DecideMode::CausalTriad
// ---------------------------------------------------------------------------

fn pipeline(gate: PolicyGate) -> ProductionPipeline {
    ProductionPipeline::new(
        Arc::new(LatentDynamicsWorldModel::default()),
        Arc::new(gate),
    )
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

fn triad_req<'a>(
    state: &'a FullLatent,
    candidates: &'a [ActionId],
    mode: DecideMode,
    triad: Option<CausalTriadRequest>,
) -> DecideRequest<'a> {
    DecideRequest {
        active_context: Vec::new(),
        deadline: None,
        budget_ms: None,
        state,
        candidates,
        mode,
        entropy: NormalizedEntropy(0.2),
        return_trajectory: false,
        horizon: 4,
        causal_triad: triad,
    }
}

fn triad_input(options: TriadRunOptions) -> Option<CausalTriadRequest> {
    Some(CausalTriadRequest {
        dag: spec(),
        options,
    })
}

#[test]
fn decide_causal_triad_routes_to_the_triad_and_commits_a_gate_checked_plan() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    let opts = TriadRunOptions {
        seed: Some(11),
        ..TriadRunOptions::default()
    };
    let d = p
        .decide(&triad_req(
            &s,
            &CANDS,
            DecideMode::CausalTriad,
            triad_input(opts.clone()),
        ))
        .unwrap();
    assert_eq!(d.mode, DecideMode::CausalTriad);
    assert_eq!(d.engine, "CausalTriadPipeline");
    let r = d.triad.as_ref().expect("triad report");
    assert_eq!(d.action, ActionId(1), "fetch is the only causal first step");
    assert_eq!(r.chosen_path.first(), Some(&d.action));
    assert_eq!(r.chosen_path.last(), Some(&ActionId(5)));
    assert_eq!(r.nominal_time, 9);
    assert_eq!((r.shards, r.threads_spawned, r.n_samples), (1, 0, 64));
    assert_eq!(r.gate_reasons.values().sum::<usize>(), 64);
    assert!(r.n_pass > 0);
    // Only one causal first step: the decision distribution has no spread.
    assert_eq!(d.entropy, NormalizedEntropy(0.0));

    // The committed plan passes an independent gate check.
    let dag = dag();
    let idx_path: Vec<usize> = r
        .chosen_path
        .iter()
        .map(|&a| dag.index_of(a).unwrap())
        .collect();
    assert!(CausalGate::new(&dag, ctx()).check(&idx_path).ok);
    assert!(dag.respects_topology(&idx_path));

    // Same seed through the deadline worker gives the same plan.
    let mut budgeted = triad_req(&s, &CANDS, DecideMode::CausalTriad, triad_input(opts));
    budgeted.budget_ms = Some(5_000.0);
    let again = p.decide(&budgeted).unwrap();
    assert_eq!(again.triad.unwrap().chosen_path, r.chosen_path);
}

#[test]
fn decide_causal_triad_mid_plan_uses_done_and_time_used() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    let opts = TriadRunOptions {
        done: vec![1, 2],
        time_used: 5,
        seed: Some(3),
        ..TriadRunOptions::default()
    };
    let d = p
        .decide(&triad_req(
            &s,
            &CANDS,
            DecideMode::CausalTriad,
            triad_input(opts),
        ))
        .unwrap();
    let r = d.triad.unwrap();
    assert!(!r.chosen_path.contains(&ActionId(1)) && !r.chosen_path.contains(&ActionId(2)));
    assert_eq!(r.nominal_time, 9);
    // test and lint are both open now: two causal first steps.
    assert!([ActionId(3), ActionId(4)].contains(&d.action));
}

#[test]
fn decide_causal_triad_never_plans_through_a_policy_hard_stop() {
    // Target 3 is OR over a cheap branch 1 and a valuable branch 2.
    // Net reward: via 2 is 10 - 4 = 6, via 1 is 0 - 2 = -2.
    let cands = [ActionId(1), ActionId(2), ActionId(3)];
    let dag: CausalDagSpec = serde_json::from_value(json!({
        "parents": {"3": [1, 2]},
        "is_or": {"3": true},
        "cost": {"1": 1, "2": 3, "3": 1},
        "value": {"2": 10.0},
        "target": 3,
        "budget": 10
    }))
    .unwrap();
    let s = FullLatent::zeros();
    let run = |gate| {
        pipeline(gate).decide(&triad_req(
            &s,
            &cands,
            DecideMode::CausalTriad,
            Some(CausalTriadRequest {
                dag: dag.clone(),
                options: TriadRunOptions {
                    seed: Some(5),
                    n_samples: Some(128),
                    ..TriadRunOptions::default()
                },
            }),
        ))
    };
    let open = run(PolicyGate::default()).unwrap();
    let r = open.triad.unwrap();
    assert_eq!(r.chosen_path, vec![ActionId(2), ActionId(3)]);
    assert_eq!(r.net_reward, 6.0);

    let gated = run(prohibiting(&[ActionId(2)])).unwrap();
    let r = gated.triad.unwrap();
    assert_eq!(r.chosen_path, vec![ActionId(1), ActionId(3)]);
    assert_eq!(gated.pruned[0].action, ActionId(2));

    // Forbid both branches: no plan can pass. The Lod pre-check sees both
    // hazard barriers and refuses before sampling, naming them.
    match run(prohibiting(&[ActionId(1), ActionId(2)])) {
        Err(PlannerError::CausalInfeasible(m)) => {
            assert!(m.contains("unreachable"), "{m}");
            assert!(m.contains("[1, 2]"), "{m}");
        }
        other => panic!("expected CausalInfeasible, got {other:?}"),
    }
}

#[test]
fn decide_causal_triad_refuses_missing_or_misrouted_input() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    for mode in [DecideMode::CausalTriad, DecideMode::TournamentTriad] {
        let e = p.decide(&triad_req(&s, &CANDS, mode, None)).unwrap_err();
        assert!(
            matches!(&e, PlannerError::InvalidInput(m) if m.contains("requires a causal_triad")),
            "{e}"
        );
    }
    let e = p
        .decide(&triad_req(
            &s,
            &CANDS,
            DecideMode::Reflex,
            triad_input(TriadRunOptions::default()),
        ))
        .unwrap_err();
    assert!(
        matches!(&e, PlannerError::InvalidInput(m) if m.contains("only read by")),
        "{e}"
    );

    let e = p
        .decide(&triad_req(
            &s,
            &CANDS,
            DecideMode::CausalTriad,
            triad_input(TriadRunOptions {
                shards: Some(4),
                ..TriadRunOptions::default()
            }),
        ))
        .unwrap_err();
    assert!(
        matches!(&e, PlannerError::InvalidInput(m) if m.contains("one shard")),
        "{e}"
    );

    let e = p
        .decide(&triad_req(
            &s,
            &CANDS,
            DecideMode::CausalTriad,
            triad_input(TriadRunOptions {
                done: vec![1, 2, 3, 4, 5],
                ..TriadRunOptions::default()
            }),
        ))
        .unwrap_err();
    assert!(
        matches!(&e, PlannerError::InvalidInput(m) if m.contains("already done")),
        "{e}"
    );

    assert_eq!(
        "triad".parse::<DecideMode>().unwrap(),
        DecideMode::CausalTriad
    );
    assert_eq!(
        "tournament_triad".parse::<DecideMode>().unwrap().as_str(),
        "tournament_triad"
    );
}

/// Latent aligned with action 0's perturbation so action 0 crosses DONE_NORM
/// (same construction as production_pipeline_tests).
fn trap_state() -> FullLatent {
    let mut u = FullLatent::zeros();
    for (i, x) in u.as_mut_slice().iter_mut().enumerate() {
        *x = (i as f32 * 0.05).sin();
    }
    let c = (101.0 / u.l2_norm() - 0.05) / 0.95;
    u.as_mut_slice().iter_mut().for_each(|x| *x *= c);
    let (_, _, lethal_done) = LatentDynamicsWorldModel::default()
        .step(&u, ActionId(0))
        .unwrap();
    assert!(lethal_done, "precondition: action 0 must cross DONE_NORM");
    u
}

#[test]
fn decide_causal_triad_blocks_an_immediate_hazard_as_first_step_and_replays_the_plan() {
    // Two sources 0 and 18 both enable target 30 (OR). Action 0 is lethal now.
    let cands = [ActionId(0), ActionId(18), ActionId(30)];
    let dag: CausalDagSpec = serde_json::from_value(json!({
        "parents": {"30": [0, 18]},
        "is_or": {"30": true},
        "cost": {"0": 1, "18": 3, "30": 1},
        "target": 30,
        "budget": 10
    }))
    .unwrap();
    let s = trap_state();
    let p = pipeline(PolicyGate::default());
    let mut req = triad_req(
        &s,
        &cands,
        DecideMode::CausalTriad,
        Some(CausalTriadRequest {
            dag,
            options: TriadRunOptions {
                seed: Some(2),
                ..TriadRunOptions::default()
            },
        }),
    );
    req.return_trajectory = true;
    let d = p.decide(&req).unwrap();
    // From the trap state both 0 and the target 30 cross DONE_NORM on step 1.
    assert_eq!(d.hazardous_actions, vec![ActionId(0), ActionId(30)]);
    // The cheaper source 0 is a step-1 hazard; the triad must start with 18.
    // 30 is only blocked as a first step, so it may still close the plan.
    assert_eq!(d.action, ActionId(18));
    let r = d.triad.as_ref().unwrap();
    assert_eq!(r.chosen_path, vec![ActionId(18), ActionId(30)]);
    assert!(r.gate_reasons.get("blocked").copied().unwrap_or(0) == 0);
    // The trajectory is the committed plan replayed on the world model.
    let t = d.trajectory.unwrap();
    assert_eq!(t.continuation_policy, "fixed_plan");
    let acts: Vec<ActionId> = t.steps.iter().map(|s| s.action).collect();
    assert_eq!(acts, r.chosen_path);
}

// ------------------------------------------------ energy-steered sampler
//
// Shared golden fixture with the Python tests, generated from the research
// `CausalBounds.closure` by
// gen-zero-research docs/research/gflownet-causal-triad/src/make_energy_potential_golden.py.

use gen_zero_lod::CausalPotential;
use gen_zero_planner::triad::{
    TournamentTriadPipeline, TriadProblem, DEFAULT_ENERGY_ALPHA, MAX_ENERGY_ALPHA,
};

fn golden() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/energy_potential_golden.json")).unwrap()
}

/// Fixture DAG over `ActionId(i)` for node `i`. `zero_values` sets every value
/// to 0, which makes the Rust net cost `cost - value` equal the research time
/// cost, so closures can be compared with the research reference exactly.
fn fixture_dag(d: &serde_json::Value, zero_values: bool) -> CausalDag {
    let n = d["n"].as_u64().unwrap() as usize;
    let nodes: Vec<CausalNode> = (0..n)
        .map(|i| CausalNode {
            action: ActionId(i as u32),
            cost: d["cost"][i].as_u64().unwrap() as u32,
            value: if zero_values {
                0.0
            } else {
                d["value"][i].as_f64().unwrap()
            },
            is_or: d["is_or"][i].as_bool().unwrap(),
        })
        .collect();
    let mut edges = Vec::new();
    for c in 0..n {
        for p in d["parents"][c].as_array().unwrap() {
            edges.push(CausalEdge {
                parent: ActionId(p.as_u64().unwrap() as u32),
                child: ActionId(c as u32),
            });
        }
    }
    let target = ActionId(d["target"].as_u64().unwrap() as u32);
    CausalDag::new(nodes, &edges, target, d["budget"].as_u64().unwrap() as u32).unwrap()
}

/// Best `(net_reward, nominal_time)` the arbiter could pick among plans the
/// sampler can propose (goal-cone nodes, target last, within budget). Both
/// numbers depend only on the done-set, so a BFS over cone subsets is exact.
fn arbiter_best(dag: &CausalDag) -> Option<(f64, u32)> {
    let cone = dag.goal_cone();
    let t = dag.target();
    let admissible = |done: u64, v: usize| {
        let pm = dag.parent_mask(v);
        done & (1 << v) == 0
            && (pm == 0
                || if dag.node(v).is_or {
                    pm & done != 0
                } else {
                    pm & done == pm
                })
    };
    let cost_of = |m: u64| {
        (0..dag.len())
            .filter(|&i| m & (1 << i) != 0)
            .map(|i| dag.node(i).cost)
            .sum::<u32>()
    };
    let value_of = |m: u64| {
        (0..dag.len())
            .filter(|&i| m & (1 << i) != 0)
            .map(|i| dag.node(i).value)
            .sum::<f64>()
    };
    let mut seen = std::collections::HashSet::from([0_u64]);
    let mut queue = vec![0_u64];
    let mut best: Option<(f64, u32)> = None;
    while let Some(m) = queue.pop() {
        let time = cost_of(m);
        if admissible(m, t) && time + dag.node(t).cost <= dag.budget() {
            let full = m | (1 << t);
            let cand = (value_of(full) - f64::from(cost_of(full)), cost_of(full));
            if best.is_none_or(|b| cand.0 > b.0 || (cand.0 == b.0 && cand.1 < b.1)) {
                best = Some(cand);
            }
        }
        for v in 0..dag.len() {
            if v != t
                && cone & (1 << v) != 0
                && admissible(m, v)
                && time + dag.node(v).cost <= dag.budget()
            {
                let next = m | (1 << v);
                if seen.insert(next) {
                    queue.push(next);
                }
            }
        }
    }
    best
}

#[test]
fn energy_closure_and_delta_phi_match_the_research_reference() {
    let g = golden();
    let (mut n_closure, mut n_delta) = (0, 0);
    for entry in g["dags"].as_array().unwrap() {
        let pot = CausalPotential::new(&fixture_dag(&entry["dag"], true)).unwrap();
        for pair in entry["closure"].as_array().unwrap() {
            let (done, want) = (pair[0].as_u64().unwrap(), pair[1].as_f64().unwrap());
            assert_eq!(pot.closure(done), want, "closure({done:#x})");
            n_closure += 1;
        }
        for t in entry["delta_phi"].as_array().unwrap() {
            let (done, a, want) = (
                t[0].as_u64().unwrap(),
                t[1].as_u64().unwrap() as usize,
                t[2].as_f64().unwrap(),
            );
            assert_eq!(pot.delta_phi(done, a), want, "dphi({done:#x}, {a})");
            n_delta += 1;
        }
    }
    // Same counts the Python test checks: the fixture really was exercised.
    assert_eq!((n_closure, n_delta), (7376, 1858));
}

#[test]
fn energy_alpha_zero_reproduces_the_pre_steering_sampler() {
    // Captured from the Rust sampler before energy steering existed.
    let a0: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/energy_alpha0_golden_rs.json")).unwrap();
    let g = golden();
    let runs = a0["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 6);
    let cfg = GeodesicFlowConfig {
        energy_alpha: 0.0,
        ..GeodesicFlowConfig::default()
    };
    for r in runs {
        let dag = fixture_dag(
            &g["dags"][r["dag_index"].as_u64().unwrap() as usize]["dag"],
            false,
        );
        let sampler = GeodesicFlowSampler::new(&dag, cfg).unwrap();
        let mut rng = StdRng::seed_from_u64(r["seed"].as_u64().unwrap());
        let set = sampler
            .sample(
                &mut rng,
                r["n_samples"].as_u64().unwrap() as usize,
                dag.len(),
                &SamplerStart {
                    done: 0,
                    time_used: 0,
                    blocked: 0,
                    blocked_first: 0,
                },
                None,
            )
            .unwrap();
        for (k, path) in set.paths.iter().enumerate() {
            let want: Vec<usize> = r["paths"][k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(path.actions, want);
            // Exact bits: serde_json's default float parser can be 1 ulp off.
            let want = r["log_pf_bits"][k].as_u64().unwrap();
            assert_eq!(path.log_pf.to_bits(), want, "log_pf {:?}", path.log_pf);
        }
    }
}

#[test]
fn energy_alpha_defaults_to_two_and_out_of_range_is_refused() {
    assert_eq!(DEFAULT_ENERGY_ALPHA, 2.0);
    assert_eq!(GeodesicFlowConfig::default().energy_alpha, 2.0);
    let t = TournamentTriadPipeline::new(1, 1).unwrap();
    assert_eq!(t.energy_alpha(), 2.0);
    for bad in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.5,
        MAX_ENERGY_ALPHA + 1.0,
    ] {
        let cfg = GeodesicFlowConfig {
            energy_alpha: bad,
            ..GeodesicFlowConfig::default()
        };
        assert!(cfg.validate().is_err(), "{bad}");
        assert!(t.clone().with_energy_alpha(bad).is_err(), "{bad}");
    }
    assert_eq!(
        t.with_energy_alpha(MAX_ENERGY_ALPHA)
            .unwrap()
            .energy_alpha(),
        MAX_ENERGY_ALPHA
    );
}

#[test]
fn energy_potential_uses_the_arbiter_net_cost() {
    // Target 3 is an OR over a slow but valuable branch (1: cost 4, value 9)
    // and a fast worthless one (2: cost 2). The arbiter ranks by net reward,
    // so branch 1 is the better plan and the potential must say so.
    let nodes = vec![
        CausalNode {
            action: ActionId(0),
            cost: 1,
            value: 0.0,
            is_or: false,
        },
        CausalNode {
            action: ActionId(1),
            cost: 4,
            value: 9.0,
            is_or: false,
        },
        CausalNode {
            action: ActionId(2),
            cost: 2,
            value: 0.0,
            is_or: false,
        },
        CausalNode {
            action: ActionId(3),
            cost: 1,
            value: 0.0,
            is_or: true,
        },
    ];
    let e = |p, c| CausalEdge {
        parent: ActionId(p),
        child: ActionId(c),
    };
    let dag = CausalDag::new(
        nodes,
        &[e(0, 1), e(0, 2), e(1, 3), e(2, 3)],
        ActionId(3),
        20,
    )
    .unwrap();
    let pot = CausalPotential::new(&dag).unwrap();
    assert_eq!(pot.step_cost(1), -5.0);
    // h({0}) = c1 + c3 = -4; after either branch h = c3 = 1.
    assert_eq!(pot.closure(0b0001), -4.0);
    assert_eq!(pot.delta_phi(0b0001, 1), 0.0);
    assert_eq!(pot.delta_phi(0b0001, 2), -7.0);
    // Every steered plan takes the valuable branch: exp(-2 * 7) = 8.3e-7.
    let r = TournamentTriadPipeline::new(1, 1)
        .unwrap()
        .plan(&TriadProblem::new(&dag), 64, Some(5), None)
        .unwrap();
    assert_eq!(r.chosen_path, vec![ActionId(0), ActionId(1), ActionId(3)]);
    assert_eq!(r.net_reward, 3.0);
    assert_eq!(r.energy_alpha, 2.0);
}

#[test]
fn energy_paired_convergence_alpha2_vs_native_flow_k4() {
    // Same 40 research graphs as the Python test (budget 1.25 x the time
    // optimum, real node values), same seeds for both arms, one shard, K = 4.
    // Quality is judged against the arbiter-best plan, not the time optimum.
    let g = golden();
    let dags = g["convergence_dags"].as_array().unwrap();
    assert_eq!(dags.len(), 40);
    let native = TournamentTriadPipeline::new(1, 1)
        .unwrap()
        .with_energy_alpha(0.0)
        .unwrap();
    let steered = TournamentTriadPipeline::new(1, 1).unwrap();
    let (mut both, mut only_native, mut only_steered) = (0_u64, 0_u64, 0_u64);
    let (mut opt_native, mut opt_steered) = (0, 0);
    for (i, d) in dags.iter().enumerate() {
        let dag = fixture_dag(d, false);
        let best = arbiter_best(&dag).expect("every fixture graph has an on-time plan");
        let run = |e: &TournamentTriadPipeline| match e.plan(
            &TriadProblem::new(&dag),
            4,
            Some(1000 + i as u64),
            None,
        ) {
            Ok(r) => {
                assert!(
                    r.net_reward <= best.0 + 1e-9,
                    "beat the exhaustive best: {r:?}"
                );
                Some((r.net_reward - best.0).abs() < 1e-9 && r.nominal_time == best.1)
            }
            Err(PlannerError::CausalGateEmpty { .. }) => None,
            Err(e) => panic!("{e}"),
        };
        let (a, b) = (run(&native), run(&steered));
        match (a.is_some(), b.is_some()) {
            (true, true) => both += 1,
            (true, false) => only_native += 1,
            (false, true) => only_steered += 1,
            (false, false) => {}
        }
        opt_native += usize::from(a == Some(true));
        opt_steered += usize::from(b == Some(true));
    }
    let disc = only_native + only_steered;
    let comb = |n: u64, k: u64| (0..k).fold(1.0_f64, |c, j| c * (n - j) as f64 / (j + 1) as f64);
    let p_value =
        (only_steered..=disc).map(|k| comb(disc, k)).sum::<f64>() / 2_f64.powi(disc as i32);
    println!(
        "[energy-convergence rs K=4] pass native={}/40 steered={}/40 discordant={only_native}:{only_steered} \
         sign-test p={p_value:.3e} arbiter-optimal native={opt_native} steered={opt_steered}",
        both + only_native,
        both + only_steered
    );
    assert_eq!(only_native, 0);
    assert!(only_steered >= 15, "{only_steered}");
    assert!(p_value < 1e-4, "{p_value}");
    assert!(
        opt_steered > opt_native + 20,
        "{opt_native} -> {opt_steered}"
    );
}

#[test]
fn decide_causal_triad_reports_and_honours_energy_alpha() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    let run = |alpha: Option<f64>| {
        p.decide(&triad_req(
            &s,
            &CANDS,
            DecideMode::CausalTriad,
            triad_input(TriadRunOptions {
                seed: Some(4),
                energy_alpha: alpha,
                ..TriadRunOptions::default()
            }),
        ))
    };
    assert_eq!(run(None).unwrap().triad.unwrap().energy_alpha, 2.0);
    let native = run(Some(0.0)).unwrap().triad.unwrap();
    assert_eq!(native.energy_alpha, 0.0);
    // The release DAG's goal cone is all AND: every plan runs the same node
    // set, so dPhi is 0 for every allowed node and the coupling changes nothing.
    assert_eq!(
        native.mean_log_pf,
        run(Some(8.0)).unwrap().triad.unwrap().mean_log_pf
    );
    for bad in [-1.0, f64::NAN, 1e6] {
        assert!(
            matches!(run(Some(bad)), Err(PlannerError::InvalidInput(_))),
            "{bad}"
        );
    }

    // With an OR choice the coupling reaches the sampler through decide:
    // 4 = OR(2: slow, value 9 | 3: fast, worthless); the arbiter wants 2.
    let cands = [ActionId(1), ActionId(2), ActionId(3), ActionId(4)];
    let or_spec: CausalDagSpec = serde_json::from_value(json!({
        "parents": {"2": [1], "3": [1], "4": [2, 3]},
        "is_or": {"4": true},
        "cost": {"1": 1, "2": 4, "3": 2, "4": 1},
        "value": {"2": 9.0},
        "target": 4,
        "budget": 20
    }))
    .unwrap();
    let run_or = |alpha: f64| {
        p.decide(&triad_req(
            &s,
            &cands,
            DecideMode::CausalTriad,
            Some(CausalTriadRequest {
                dag: or_spec.clone(),
                options: TriadRunOptions {
                    seed: Some(4),
                    n_samples: Some(64),
                    energy_alpha: Some(alpha),
                    ..TriadRunOptions::default()
                },
            }),
        ))
        .unwrap()
        .triad
        .unwrap()
    };
    let (flat, steered) = (run_or(0.0), run_or(8.0));
    assert_eq!(
        steered.chosen_path,
        vec![ActionId(1), ActionId(2), ActionId(4)]
    );
    // Steering concentrates the flow on the one good branch.
    assert!(
        steered.mean_log_pf > flat.mean_log_pf,
        "{} vs {}",
        steered.mean_log_pf,
        flat.mean_log_pf
    );
}

// ---------------------------------------------------------------------------
// Lod layout of the decide route (gen_zero_lod::causal_lod)
// ---------------------------------------------------------------------------

#[test]
fn decide_causal_triad_reports_the_lod_layout_of_the_dag() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    let d = p
        .decide(&triad_req(
            &s,
            &CANDS,
            DecideMode::CausalTriad,
            triad_input(TriadRunOptions {
                seed: Some(11),
                ..TriadRunOptions::default()
            }),
        ))
        .unwrap();
    let r = d.triad.unwrap();
    let lod = &r.lod.summary;
    // Seven actions; deploy's two AND parents and package's OR parents cluster.
    assert_eq!((lod.atoms, lod.and_clusters, lod.or_clusters), (7, 1, 1));
    // Every deploy plan runs fetch, build, test and lint: 2 + 3 + 2 + 1 + 1.
    assert_eq!(lod.checkpoints, vec![1, 2, 3, 4]);
    assert_eq!(lod.mandatory_floor, 9);
    assert!(lod.hazard_barriers.is_empty());
    // 7 atoms + 2 clusters + 4 checkpoints + 1 systemic node.
    assert_eq!(lod.graph_nodes, 14);
    // One goal distance per committed step, shrinking toward the target.
    assert_eq!(r.lod.chosen_goal_distance.len(), r.chosen_path.len());
    assert!(r
        .lod
        .chosen_goal_distance
        .iter()
        .all(|x| x.is_finite() && *x > 0.0));
    let first = r.lod.chosen_goal_distance[0];
    let last = *r.lod.chosen_goal_distance.last().unwrap();
    assert!(
        last < first,
        "fetch {first} should sit further out than deploy {last}"
    );
}

#[test]
fn decide_causal_triad_refuses_a_budget_below_the_mandatory_floor() {
    let p = pipeline(PolicyGate::default());
    let s = FullLatent::zeros();
    let mut v = spec_value();
    v["budget"] = json!(8); // the floor is 9
    let r = p.decide(&triad_req(
        &s,
        &CANDS,
        DecideMode::CausalTriad,
        Some(CausalTriadRequest {
            dag: serde_json::from_value(v).unwrap(),
            options: TriadRunOptions {
                seed: Some(1),
                ..TriadRunOptions::default()
            },
        }),
    ));
    match r {
        Err(PlannerError::CausalInfeasible(m)) => {
            assert!(m.contains("floor 9"), "{m}");
            assert!(m.contains("[1, 2, 3, 4]"), "{m}");
        }
        other => panic!("expected CausalInfeasible, got {other:?}"),
    }
    // Mid-plan: fetch and build done at t = 5 leave test + lint + deploy = 4.
    let mut v = spec_value();
    v["budget"] = json!(8);
    let tight = p.decide(&triad_req(
        &s,
        &CANDS,
        DecideMode::CausalTriad,
        Some(CausalTriadRequest {
            dag: serde_json::from_value(v).unwrap(),
            options: TriadRunOptions {
                done: vec![1, 2],
                time_used: 5,
                seed: Some(1),
                ..TriadRunOptions::default()
            },
        }),
    ));
    assert!(
        matches!(tight, Err(PlannerError::CausalInfeasible(_))),
        "{tight:?}"
    );
}

#[test]
fn triad_run_options_accepts_valid_robust_spec_and_refuses_unknown_field() {
    let opts: TriadRunOptions = serde_json::from_value(json!({
        "seed": 1,
        "robust": {
            "model": {
                "extra_pmf": [0.5, 0.5],
                "fatigue_frac": null,
                "fatigue_extra": 0
            }
        }
    }))
    .unwrap();
    assert!(opts.robust.is_some());
    let e = serde_json::from_value::<TriadRunOptions>(json!({"seed": 1, "robust": {"typo": 1}}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("unknown field `typo`") || e.contains("missing field `model`"), "{e}");
}

#[test]
fn decide_causal_triad_gate_empty_when_every_first_step_is_an_immediate_hazard() {
    // Target 30 needs source 0 (AND). From the trap state 0 and 30 are both
    // lethal on step 1; 18 is safe but outside the goal cone. The Lod
    // pre-check passes (a first-step hazard is not a permanent barrier), the
    // sampler dead-ends on every plan, and decide refuses with the histogram.
    let cands = [ActionId(0), ActionId(18), ActionId(30)];
    let dag: CausalDagSpec = serde_json::from_value(json!({
        "parents": {"30": [0]},
        "cost": {"0": 1, "18": 1, "30": 1},
        "target": 30,
        "budget": 10
    }))
    .unwrap();
    let s = trap_state();
    let r = pipeline(PolicyGate::default()).decide(&triad_req(
        &s,
        &cands,
        DecideMode::CausalTriad,
        Some(CausalTriadRequest {
            dag,
            options: TriadRunOptions {
                seed: Some(4),
                n_samples: Some(16),
                ..TriadRunOptions::default()
            },
        }),
    ));
    match r {
        Err(PlannerError::CausalGateEmpty { sampled, reasons }) => {
            assert_eq!(sampled, 16);
            assert!(reasons.contains("no_target"), "{reasons}");
        }
        other => panic!("expected CausalGateEmpty, got {other:?}"),
    }
}

/// Rejects action 1 at the all-zero state only.
struct RejectOneAtZero;

impl gen_zero_planner::StateActionMask for RejectOneAtZero {
    fn allowed_actions(&self, state: &[f32], candidate_actions: &[u32]) -> Vec<u32> {
        let at_zero = state.iter().all(|&x| x == 0.0);
        candidate_actions
            .iter()
            .copied()
            .filter(|&a| !(at_zero && a == 1))
            .collect()
    }
}

#[test]
fn decide_causal_triad_blocks_a_root_state_mask_rejection_as_first_step() {
    // Target 3 is OR over cheap 1 and dear 2; the nominal best is 1 -> 3.
    let cands = [ActionId(1), ActionId(2), ActionId(3)];
    let dag: CausalDagSpec = serde_json::from_value(json!({
        "parents": {"3": [1, 2]},
        "is_or": {"3": true},
        "cost": {"1": 1, "2": 3, "3": 1},
        "target": 3,
        "budget": 10
    }))
    .unwrap();
    let s = FullLatent::zeros();
    let run = |p: ProductionPipeline| {
        p.decide(&triad_req(
            &s,
            &cands,
            DecideMode::CausalTriad,
            Some(CausalTriadRequest {
                dag: dag.clone(),
                options: TriadRunOptions {
                    seed: Some(6),
                    n_samples: Some(64),
                    ..TriadRunOptions::default()
                },
            }),
        ))
        .unwrap()
    };
    let open = run(pipeline(PolicyGate::default()));
    assert_eq!(
        open.triad.unwrap().chosen_path,
        vec![ActionId(1), ActionId(3)]
    );

    let masked = ProductionPipeline::new(
        Arc::new(gen_zero_planner::MaskedDynamics::new(
            LatentDynamicsWorldModel::default(),
            Arc::new(RejectOneAtZero),
        )),
        Arc::new(PolicyGate::default()),
    );
    let d = run(masked);
    assert_eq!(d.pruned[0].action, ActionId(1));
    assert_eq!(
        d.pruned[0].source,
        gen_zero_planner::PruneSource::StateActionMask
    );
    // Masked at the root only: no barrier, the plan starts with the other branch.
    let r = d.triad.unwrap();
    assert!(r.lod.summary.hazard_barriers.is_empty());
    assert_eq!(r.chosen_path, vec![ActionId(2), ActionId(3)]);
    assert_eq!(d.action, ActionId(2));
}
