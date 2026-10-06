//! Robust slack arbitration and the active causal probe (`triad::robust_gate`),
//! their mount in both tournament tiers, and the production route through
//! `ProductionPipeline::decide` with `TriadRunOptions::robust`.
//!
//! Fixtures are the ones of `python/gen_zero/tests/test_probe_slack.py`. Every
//! expected probability is worked out by hand in the comment next to it, or
//! checked against an independent brute-force enumeration in this file; nothing
//! is copied from a run of the code under test.

use gen_zero_core::{ActionId, FullLatent, NormalizedEntropy};
use gen_zero_gate::PolicyGate;
use gen_zero_planner::triad::{
    choose_probe_action, score_path_into, shortest_first_order, CausalDag, CausalDagSpec,
    CausalGate, DisturbanceModel, GateContext, PathVerdict, RobustObjective, RobustSlackSelector,
    RobustSpec, ScoreScratch, TournamentTriadPipeline, TriadProblem, TriadRunOptions,
    MAX_ROBUST_WINDOW,
};
use gen_zero_planner::{
    CausalTriadRequest, DecideMode, DecideRequest, PlannerError, ProductionPipeline,
};
use gen_zero_worldmodel::LatentDynamicsWorldModel;
use serde_json::{json, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::time::Instant;

// ------------------------------------------------------ allocation counter

/// Counts heap allocations of the current thread, so parallel tests in this
/// binary do not disturb each other's counts.
struct CountingAlloc;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: forwards the caller's layout contract to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as `GlobalAlloc::realloc`, forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

// ------------------------------------------------------------- fixtures

// Long cheap chain vs. short expensive branch, OR target. Action id = index + 1.
//   s(1) -> a1 -> a2 -> a3 -> a4 -> t    nominal 6, 6 items
//   s(1) -> b(5) -> t                     nominal 1+5+1 = 7, 3 items
// Budget 8. Under COIN: chain finishes at 6 + Bin(6, .5), P(<= 8) = (1+6+15)/64;
// branch at 7 + Bin(3, .5), P(<= 8) = (1+3)/8. The nominal gate picks the chain.
const CHAIN_IDS: [ActionId; 7] = [
    ActionId(1), // s
    ActionId(2), // a1
    ActionId(3), // a2
    ActionId(4), // a3
    ActionId(5), // a4
    ActionId(6), // b
    ActionId(7), // t
];
const CHAIN_PATH: [usize; 6] = [0, 1, 2, 3, 4, 6];
const BRANCH_PATH: [usize; 3] = [0, 5, 6];

fn chain_spec(budget: u32) -> CausalDagSpec {
    serde_json::from_value(json!({
        "parents": {"2": [1], "3": [2], "4": [3], "5": [4], "6": [1], "7": [5, 6]},
        "is_or": {"7": true},
        "cost": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 5, "7": 1},
        "target": 7,
        "budget": budget
    }))
    .unwrap()
}

// Fatigue ordering: x (cost 3), y, z (cost 1) all needed by t (AND). Budget 7.
// Fatigue +1 from clock >= ceil(0.42 * 7) = 3, no other surcharge.
//   x,y,z,t: x 0->3, y 3->5, z 5->7, t 7->9   over budget
//   y,z,x,t: y 0->1, z 1->2, x 2->5, t 5->7   on time
const FAT_IDS: [ActionId; 4] = [ActionId(1), ActionId(2), ActionId(3), ActionId(4)];

fn fat_spec() -> CausalDagSpec {
    serde_json::from_value(json!({
        "parents": {"4": [1, 2, 3]},
        "cost": {"1": 3, "2": 1, "3": 1, "4": 1},
        "target": 4,
        "budget": 7
    }))
    .unwrap()
}

fn chain() -> CausalDag {
    CausalDag::from_spec(&CHAIN_IDS, &chain_spec(8)).unwrap()
}

fn fat() -> CausalDag {
    CausalDag::from_spec(&FAT_IDS, &fat_spec()).unwrap()
}

fn coin() -> DisturbanceModel {
    DisturbanceModel::new(vec![0.5, 0.5], None, 0, f64::INFINITY, "test: fair coin").unwrap()
}

fn fatigue_only() -> DisturbanceModel {
    DisturbanceModel::new(
        vec![1.0],
        Some(0.42),
        1,
        f64::INFINITY,
        "test: fatigue only",
    )
    .unwrap()
}

fn ctx0() -> GateContext {
    GateContext {
        done: 0,
        time_used: 0,
        blocked: 0,
        blocked_first: 0,
    }
}

fn sel(model: DisturbanceModel, objective: RobustObjective, reorder: bool) -> RobustSlackSelector {
    RobustSlackSelector::new(model, objective, reorder)
}

fn score_path(
    dag: &CausalDag,
    m: &DisturbanceModel,
    path: &[usize],
    t0: u32,
) -> Result<gen_zero_planner::triad::RobustScore, PlannerError> {
    score_path_into(dag, m, path, t0, &mut ScoreScratch::default())
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

// ------------------------------------------------------------- scoring

#[test]
fn finish_time_distribution_is_exact_on_the_python_fixture() {
    let dag = chain();
    let c = score_path(&dag, &coin(), &CHAIN_PATH, 0).unwrap();
    let b = score_path(&dag, &coin(), &BRANCH_PATH, 0).unwrap();
    assert!(close(c.p_success, 22.0 / 64.0), "{c:?}");
    assert!(close(b.p_success, 4.0 / 8.0), "{b:?}");
    // E[finish] = nominal + 0.5 per item, overruns included.
    assert!(close(c.expected_time, 6.0 + 3.0), "{c:?}");
    assert!(close(b.expected_time, 7.0 + 1.5), "{b:?}");
    // Branch: finish 7 w.p. 1/8 (slack 1), 8 w.p. 3/8 (slack 0).
    assert!(close(b.expected_slack, 1.0 / 8.0), "{b:?}");
}

#[test]
fn fatigue_is_tracked_through_the_clock() {
    let dag = fat();
    let m = fatigue_only();
    assert_eq!(m.fatigue_at(7).unwrap(), Some(3));
    assert_eq!(
        score_path(&dag, &m, &[0, 1, 2, 3], 0).unwrap().p_success,
        0.0
    );
    assert_eq!(
        score_path(&dag, &m, &[1, 2, 0, 3], 0).unwrap().p_success,
        1.0
    );
    // y,z,x,t from clock 0 ends at exactly 7: slack 0, E[finish] 7.
    let s = score_path(&dag, &m, &[1, 2, 0, 3], 0).unwrap();
    assert_eq!((s.expected_slack, s.expected_time), (0.0, 7.0));
    // x,y,z,t ends at 9 (deterministic): the overflow bucket must carry it exactly.
    assert_eq!(
        score_path(&dag, &m, &[0, 1, 2, 3], 0)
            .unwrap()
            .expected_time,
        9.0
    );
    // From clock 3 every step is fatigued: 3 + 2 + 2 + 4 + 2 = 13 > 7.
    let late = score_path(&dag, &m, &[1, 2, 0, 3], 3).unwrap();
    assert_eq!((late.p_success, late.expected_time), (0.0, 13.0));
}

/// Independent reference: enumerate every extra-cost outcome of every step.
fn brute_force(dag: &CausalDag, m: &DisturbanceModel, path: &[usize], t0: u32) -> (f64, f64, f64) {
    let budget = dag.budget();
    let fat_at = m.fatigue_at(budget).unwrap();
    let mut out = (0.0, 0.0, 0.0);
    let mut stack = vec![(0_usize, u64::from(t0), 1.0_f64)];
    while let Some((k, t, p)) = stack.pop() {
        if k == path.len() {
            if t <= u64::from(budget) {
                out.0 += p;
                out.1 += p * (u64::from(budget) - t) as f64;
            }
            out.2 += p * t as f64;
            continue;
        }
        let late = fat_at.is_some_and(|f| t >= f);
        let c = u64::from(dag.node(path[k]).cost)
            + if late {
                u64::from(m.fatigue_extra())
            } else {
                0
            };
        for (j, &pj) in m.extra_pmf().iter().enumerate() {
            if pj > 0.0 {
                stack.push((k + 1, t + c + j as u64, p * pj));
            }
        }
    }
    out
}

#[test]
fn exact_scorer_matches_brute_force_enumeration_including_overruns() {
    let dag = chain();
    let models = [
        coin(),
        DisturbanceModel::new(vec![0.6, 0.3, 0.1], Some(0.5), 2, 3.0, "skewed + fatigue").unwrap(),
        DisturbanceModel::new(vec![0.2, 0.0, 0.8], Some(1.0), 1, f64::INFINITY, "gap").unwrap(),
        // fatigue_extra without a rule: no surcharge anywhere.
        DisturbanceModel::new(vec![0.7, 0.3], None, 3, f64::INFINITY, "no rule").unwrap(),
    ];
    for m in &models {
        for (path, t0) in [
            (&CHAIN_PATH[..], 0),
            (&BRANCH_PATH[..], 0),
            (&CHAIN_PATH[..], 2),
            (&BRANCH_PATH[..], 7),
        ] {
            let s = score_path(&dag, m, path, t0).unwrap();
            let (p, slack, et) = brute_force(&dag, m, path, t0);
            assert!(
                (s.p_success - p).abs() < 1e-12,
                "{m:?} {path:?}@{t0}: {s:?} vs {p}"
            );
            assert!(
                (s.expected_slack - slack).abs() < 1e-9,
                "{m:?} {path:?}@{t0}: {s:?} vs {slack}"
            );
            assert!(
                (s.expected_time - et).abs() < 1e-9,
                "{m:?} {path:?}@{t0}: {s:?} vs {et}"
            );
        }
    }
}

#[test]
fn scorer_refuses_what_it_cannot_score_exactly() {
    let dag = chain();
    assert!(matches!(
        score_path(&dag, &coin(), &[0, 99], 0),
        Err(PlannerError::InvalidInput(_))
    ));
    let wide = CausalDag::from_spec(&CHAIN_IDS, &chain_spec(MAX_ROBUST_WINDOW + 1)).unwrap();
    let err = score_path(&wide, &coin(), &CHAIN_PATH, 0).unwrap_err();
    assert!(err.to_string().contains("exact-scoring window"), "{err}");
}

// -------------------------------------------------- causal reordering

#[test]
fn shortest_first_order_keeps_every_precondition_and_defers_the_expensive_item() {
    let dag = fat();
    let gate = CausalGate::new(&dag, ctx0());
    let mut out = Vec::new();
    shortest_first_order(&gate, &[0, 1, 2, 3], &mut out).unwrap();
    assert_eq!(out, vec![1, 2, 0, 3], "x (cost 3) deferred, target last");
    assert!(dag.respects_topology(&out));
    assert!(gate.check(&out).ok);

    // On the chain every item has exactly one ready successor: order unchanged.
    let dag = chain();
    let gate = CausalGate::new(&dag, ctx0());
    shortest_first_order(&gate, &CHAIN_PATH, &mut out).unwrap();
    assert_eq!(out, CHAIN_PATH.to_vec());
    // a1 without its parent s is not a gate-passing plan.
    assert!(shortest_first_order(&gate, &[1, 6], &mut out).is_err());
    // A repeated item is refused, not deduplicated.
    assert!(shortest_first_order(&gate, &[0, 0, 5, 6], &mut out).is_err());
}

#[test]
fn shortest_first_order_never_breaks_a_precondition_on_sampled_plans() {
    // Diamond with an expensive AND root: cheap leaves may never jump ahead of it.
    //   r(4) -> p(1), q(1); p,q -> u(1, AND); u, w(1) -> goal (AND); w free.
    let ids: Vec<ActionId> = (1..=6).map(ActionId).collect();
    let spec: CausalDagSpec = serde_json::from_value(json!({
        "parents": {"2": [1], "3": [1], "4": [2, 3], "6": [4, 5]},
        "cost": {"1": 4, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1},
        "target": 6, "budget": 20
    }))
    .unwrap();
    let dag = CausalDag::from_spec(&ids, &spec).unwrap();
    let problem = TriadProblem::new(&dag);
    let t = TournamentTriadPipeline::new(1, 1).unwrap();
    let specs = t.shard_plan(256, Some(7)).unwrap();
    let outcomes = t.run_tier1(&problem, &specs, None).unwrap();
    let gate = CausalGate::new(&dag, ctx0());
    let mut out = Vec::new();
    let mut n = 0;
    for v in outcomes[0].verdicts.iter().filter(|v| v.ok) {
        shortest_first_order(&gate, &v.path, &mut out).unwrap();
        assert!(dag.respects_topology(&out), "{:?} -> {out:?}", v.path);
        let mut a = v.path.clone();
        let mut b = out.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "same items");
        assert_eq!(out.last(), Some(&dag.target()));
        assert!(gate.check(&out).ok);
        // w (index 4, cost 1, free) is the cheapest ready item at the start;
        // then only the root (cost 4) is ready, since p, q, u all need it.
        assert_eq!(&out[..2], &[4, 0]);
        n += 1;
    }
    assert!(n > 10, "only {n} passing samples");
}

#[test]
fn shortest_first_order_skips_a_first_step_hazard_in_the_first_slot() {
    // y is a first-step hazard: z (next cheapest, also cost 1) goes first, then
    // y is fine as a later step. The reorder still passes the gate.
    let dag = fat();
    let gate = CausalGate::new(
        &dag,
        GateContext {
            blocked_first: 1 << 1,
            ..ctx0()
        },
    );
    let mut out = Vec::new();
    shortest_first_order(&gate, &[2, 0, 1, 3], &mut out).unwrap();
    assert_eq!(out, vec![2, 1, 0, 3]);
    assert!(gate.check(&out).ok);
}

#[test]
fn reordering_moves_the_long_item_out_of_the_fatigue_zone() {
    let dag = fat();
    let gate = CausalGate::new(&dag, ctx0());
    let v = gate.check(&[0, 1, 2, 3]);
    assert!(v.ok);
    let mut scratch = ScoreScratch::default();
    let no = sel(fatigue_only(), RobustObjective::PSuccess, false)
        .select(&gate, std::slice::from_ref(&v), &mut scratch)
        .unwrap();
    assert_eq!(no.score.p_success, 0.0);
    assert!(!no.reordered);
    let yes = sel(fatigue_only(), RobustObjective::PSuccess, true)
        .select(&gate, &[v], &mut scratch)
        .unwrap();
    assert!(yes.reordered);
    assert_eq!(yes.score.p_success, 1.0);
    assert_eq!(yes.verdict.path.last(), Some(&3));
    assert_eq!(yes.verdict.path[2], 0, "x after both cheap items");
    assert!(gate.check(&yes.verdict.path).ok);
    assert!(dag.respects_topology(&yes.verdict.path));
}

// ------------------------------------------- robust vs nominal gate

#[test]
fn robust_gate_overrides_the_nominal_choice() {
    let dag = chain();
    let gate = CausalGate::new(&dag, ctx0());
    let paths = vec![CHAIN_PATH.to_vec(), BRANCH_PATH.to_vec(), vec![1, 0, 6]];
    let (nominal, verdicts) = gate.select(&paths, None).unwrap();
    assert_eq!(nominal.unwrap().path, CHAIN_PATH.to_vec());
    assert!(!verdicts[2].ok);

    let ps = sel(coin(), RobustObjective::PSuccess, false);
    let (robust, _) = gate.select(&paths, Some(&ps)).unwrap();
    assert_eq!(robust.unwrap().path, BRANCH_PATH.to_vec());
    // expected_time objective: chain 9.0 vs branch 8.5, branch again.
    let et = sel(coin(), RobustObjective::ExpectedTime, false);
    let (robust, _) = gate.select(&paths, Some(&et)).unwrap();
    assert_eq!(robust.unwrap().path, BRANCH_PATH.to_vec());

    // A disturbance-free model has nothing to say: the nominal key decides.
    let none = DisturbanceModel::new(vec![1.0], None, 0, f64::INFINITY, "nominal").unwrap();
    let (same, _) = gate
        .select(&paths, Some(&sel(none, RobustObjective::PSuccess, false)))
        .unwrap();
    assert_eq!(same.unwrap().path, CHAIN_PATH.to_vec());

    // Nothing passes: no best, robust or not.
    let (empty, _) = gate.select(&[vec![1, 0, 6]], Some(&ps)).unwrap();
    assert!(empty.is_none());
}

#[test]
fn selector_refuses_gate_rejected_or_missing_plans() {
    let dag = chain();
    let gate = CausalGate::new(&dag, ctx0());
    let bad: PathVerdict = gate.check(&[1, 0, 6]);
    assert!(!bad.ok);
    let s = sel(coin(), RobustObjective::PSuccess, true);
    let mut scratch = ScoreScratch::default();
    assert!(s.select(&gate, &[bad], &mut scratch).is_err());
    assert!(s.select(&gate, &[], &mut scratch).is_err());
}

// ---------------------------------------------- probe and posterior

#[test]
fn probe_is_the_cheapest_first_step_legal_item_of_the_plan() {
    let dag = fat();
    let gate = CausalGate::new(&dag, ctx0());
    assert_eq!(choose_probe_action(&gate, &[0, 1, 2, 3]).unwrap(), 1);
    // t is not admissible yet.
    assert!(choose_probe_action(&gate, &[3]).is_err());
    // y is a first-step hazard: the probe falls to z, the next cheapest.
    let hazard = CausalGate::new(
        &dag,
        GateContext {
            blocked_first: 1 << 1,
            ..ctx0()
        },
    );
    assert_eq!(choose_probe_action(&hazard, &[0, 1, 2, 3]).unwrap(), 2);
    // y and z blocked at every step: only x is left.
    let blocked = CausalGate::new(
        &dag,
        GateContext {
            blocked: 0b110,
            ..ctx0()
        },
    );
    assert_eq!(choose_probe_action(&blocked, &[0, 1, 2, 3]).unwrap(), 0);
    // y already done: it is no longer a candidate.
    let done_y = CausalGate::new(
        &dag,
        GateContext {
            done: 1 << 1,
            ..ctx0()
        },
    );
    assert_eq!(choose_probe_action(&done_y, &[0, 2, 3]).unwrap(), 2);
}

#[test]
fn probe_first_plan_is_regated_and_rescored() {
    let dag = fat();
    let gate = CausalGate::new(&dag, ctx0());
    let s = sel(coin(), RobustObjective::PSuccess, false).with_probe(true);
    let mut scratch = ScoreScratch::default();
    let v = gate.check(&[0, 1, 2, 3]);
    let (probed, probe) = s.apply_probe(&gate, &v, &mut scratch).unwrap();
    assert_eq!(probe, 1);
    assert_eq!(probed.verdict.path, vec![1, 0, 2, 3]);
    assert!(probed.verdict.ok);
    // COIN, 4 items, nominal 6, budget 7: P(Bin(4, .5) <= 1) = 5/16.
    assert!(
        close(probed.score.p_success, 5.0 / 16.0),
        "{:?}",
        probed.score
    );
}

#[test]
fn posterior_is_the_dirichlet_multinomial_update() {
    let (same, bad) = coin().posterior(&[1, 1, 1]);
    assert_eq!(
        same,
        coin(),
        "kappa inf: an observation says nothing about other items"
    );
    assert_eq!(bad, 0);
    let m = DisturbanceModel::new(vec![0.5, 0.5], None, 0, 2.0, "").unwrap();
    let (post, bad) = m.posterior(&[1, 1]);
    // alpha = 2*(.5,.5) + (0,2) = (1, 3) -> (.25, .75), kappa 4.
    assert_eq!(bad, 0);
    assert!(close(post.extra_pmf()[0], 0.25) && close(post.extra_pmf()[1], 0.75));
    assert_eq!(post.kappa(), 4.0);
    let (_, bad) = m.posterior(&[5, -1, 0]);
    assert_eq!(bad, 2);
    // Residual: observed - nominal - modelled fatigue.
    let f = fatigue_only();
    assert_eq!(
        f.residual(3, 4, 3, 7).unwrap(),
        0,
        "late step: +1 is fatigue"
    );
    assert_eq!(
        f.residual(3, 4, 2, 7).unwrap(),
        1,
        "early step: +1 is extra"
    );
}

#[test]
fn fatigue_onset_error_propagates_to_residual() {
    let model = fatigue_only();
    assert!(model.fatigue_at(0).is_err());
    assert!(model.residual(1, 2, 0, 0).is_err());
}

fn obs(nominal: u32, observed: u32, before: u32) -> Value {
    json!({"nominal_cost": nominal, "observed_cost": observed, "time_before": before})
}

fn spec_json(model: Value, extra: Value) -> RobustSpec {
    let mut v = json!({"model": model});
    for (k, x) in extra.as_object().unwrap() {
        v[k] = x.clone();
    }
    serde_json::from_value(v).unwrap()
}

#[test]
fn posterior_from_residuals_flips_the_robust_choice() {
    // Prior (.9, .1), kappa 1. Chain P(Bin(6,.1) <= 2) ~ .984 beats branch
    // P(Bin(3,.1) <= 1) = .972. After four residuals of 1: alpha = (.9, 4.1),
    // pmf (.18, .82). Chain P(Bin(6,.82) <= 2) ~ .0116, branch
    // P(Bin(3,.82) <= 1) = .18^3 + 3 * .18^2 * .82 ~ .0855: the branch wins.
    let dag = chain();
    let gate = CausalGate::new(&dag, ctx0());
    let ok = [gate.check(&CHAIN_PATH), gate.check(&BRANCH_PATH)];
    let model =
        json!({"extra_pmf": [0.9, 0.1], "fatigue_frac": null, "fatigue_extra": 0, "kappa": 1.0});
    let mut scratch = ScoreScratch::default();
    let prior =
        RobustSlackSelector::from_spec(&spec_json(model.clone(), json!({"reorder": false})), 8)
            .unwrap();
    let c = prior.select(&gate, &ok, &mut scratch).unwrap();
    assert_eq!(c.verdict.path, CHAIN_PATH.to_vec());
    let post = RobustSlackSelector::from_spec(&spec_json(
        model,
        json!({"reorder": false, "observations": [obs(1, 2, 0), obs(1, 2, 1), obs(5, 6, 2), obs(1, 2, 7)]}),
    ), 8)
    .unwrap();
    assert_eq!(post.n_observed(), 4);
    assert_eq!(post.model().kappa(), 5.0);
    let c = post.select(&gate, &ok, &mut scratch).unwrap();
    assert_eq!(c.verdict.path, BRANCH_PATH.to_vec());
    let expect = 0.18_f64.powi(3) + 3.0 * 0.18_f64.powi(2) * 0.82;
    assert!((c.score.p_success - expect).abs() < 1e-12, "{:?}", c.score);
}

#[test]
fn robust_spec_parse_is_strict() {
    let good = json!({"extra_pmf": [0.7, 0.3], "fatigue_frac": 0.7, "fatigue_extra": 1, "kappa": 12.5, "provenance": "x"});
    let s = RobustSlackSelector::from_spec(&spec_json(good.clone(), json!({})), 8).unwrap();
    assert_eq!(s.objective(), RobustObjective::PSuccess);
    assert!(s.reorder() && !s.probe());
    // kappa null or absent = infinite.
    let inf = json!({"extra_pmf": [1.0], "fatigue_frac": null, "fatigue_extra": 0, "kappa": null});
    assert!(
        RobustSlackSelector::from_spec(&spec_json(inf, json!({})), 8)
            .unwrap()
            .model()
            .kappa()
            .is_infinite()
    );
    for bad in [
        json!({"model": good, "typo": 1}),
        json!({"objective": "p_success"}),
        json!({"model": {"extra_pmf": [1.0], "fatigue_extra": 0}}), // fatigue_frac missing
        json!({"model": good, "objective": "mean_vibes"}),
        json!({"model": good, "reorder": "yes"}),
    ] {
        assert!(
            serde_json::from_value::<RobustSpec>(bad.clone()).is_err(),
            "{bad}"
        );
    }
    for model in [
        json!({"extra_pmf": [0.6, 0.6], "fatigue_frac": null, "fatigue_extra": 0}),
        json!({"extra_pmf": [], "fatigue_frac": null, "fatigue_extra": 0}),
        json!({"extra_pmf": [1.0], "fatigue_frac": 0.0, "fatigue_extra": 1}),
        json!({"extra_pmf": [1.0], "fatigue_frac": 1.5, "fatigue_extra": 1}),
        json!({"extra_pmf": [1.0], "fatigue_frac": null, "fatigue_extra": 0, "kappa": 0.0}),
    ] {
        assert!(
            RobustSlackSelector::from_spec(&spec_json(model.clone(), json!({})), 8).is_err(),
            "{model}"
        );
    }
    // A residual outside the support is refused, not dropped.
    let err = RobustSlackSelector::from_spec(&spec_json(
        json!({"extra_pmf": [0.5, 0.5], "fatigue_frac": null, "fatigue_extra": 0, "kappa": 2.0}),
        json!({"observations": [obs(1, 1, 0), obs(1, 6, 1)]}),
    ), 8)
    .unwrap_err();
    assert!(
        err.to_string().contains("outside the model support"),
        "{err}"
    );
}

#[test]
fn observations_are_reduced_to_residuals_with_the_models_fatigue_rule() {
    // Fatigue-only model on budget 7: onset at clock 3, +1 per late step, no
    // other extra (support {0}). A +1 overrun on a late step is fatigue, so
    // residual 0 and accepted; the same +1 on an early step is residual 1,
    // outside the support, and refused.
    let fatigue =
        json!({"extra_pmf": [1.0], "fatigue_frac": 0.42, "fatigue_extra": 1, "kappa": 4.0});
    let late = RobustSlackSelector::from_spec(
        &spec_json(fatigue.clone(), json!({"observations": [obs(3, 4, 3)]})),
        7,
    )
    .unwrap();
    assert_eq!(late.n_observed(), 1);
    assert_eq!(late.model().kappa(), 5.0);
    let early = RobustSlackSelector::from_spec(
        &spec_json(fatigue, json!({"observations": [obs(3, 4, 2)]})),
        7,
    );
    assert!(early.is_err());
    // Unknown observation fields are refused.
    assert!(serde_json::from_value::<RobustSpec>(json!({
        "model": {"extra_pmf": [1.0], "fatigue_frac": null, "fatigue_extra": 0},
        "observations": [{"nominal_cost": 1, "observed_cost": 1, "time_before": 0, "action": 3}]
    }))
    .is_err());
}

// ------------------------------------------ tournament tiers

#[test]
fn robust_tournament_equals_the_flat_selector_over_the_sample_union() {
    // Tier 1 keeps max(top_p, REORDER_TOP) robust elites per shard; Tier 2 must
    // then pick exactly what one flat robust selection over every sampled plan
    // picks, reorders included.
    let dag = fat();
    let problem = TriadProblem::new(&dag);
    let gate = CausalGate::new(&dag, ctx0());
    for (seed, model) in [(1, fatigue_only()), (2, coin()), (3, fatigue_only())] {
        let s = sel(model, RobustObjective::PSuccess, true);
        let t = TournamentTriadPipeline::new(4, 1)
            .unwrap()
            .with_energy_alpha(0.0)
            .unwrap()
            .with_robust(Some(s.clone()));
        assert_eq!(t.elites_per_shard(), 3);
        let specs = t.shard_plan(64, Some(seed)).unwrap();
        let outcomes = t.run_tier1(&problem, &specs, None).unwrap();
        let arb = t.arbitrate(&problem, &outcomes).unwrap();
        let union: Vec<PathVerdict> = outcomes
            .iter()
            .flat_map(|o| o.verdicts.iter().filter(|v| v.ok).cloned())
            .collect();
        let flat = s
            .select(&gate, &union, &mut ScoreScratch::default())
            .unwrap();
        assert_eq!(arb.best.path, flat.verdict.path, "seed {seed}");
        let r = arb.robust.unwrap();
        assert!(close(r.p_success, flat.score.p_success), "seed {seed}");
    }
}

#[test]
fn reported_nominal_choice_is_the_true_nominal_pick_even_outside_the_robust_elites() {
    // Chain (nominal 6, P 22/64) plus three expensive branches b, c, d (nominal
    // 7, P 1/2 each). Under COIN the robust top 3 are the three branches, so a
    // nominal pick taken over the robust elites would be a branch. The report
    // must still name the chain, the gate's real nominal winner.
    let ids: Vec<ActionId> = (1..=9).map(ActionId).collect();
    let spec: CausalDagSpec = serde_json::from_value(json!({
        "parents": {"2": [1], "3": [2], "4": [3], "5": [4], "6": [1], "7": [1], "8": [1],
                    "9": [5, 6, 7, 8]},
        "is_or": {"9": true},
        "cost": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 5, "7": 5, "8": 5, "9": 1},
        "target": 9, "budget": 8
    }))
    .unwrap();
    let dag = CausalDag::from_spec(&ids, &spec).unwrap();
    let problem = TriadProblem::new(&dag);
    // Default energy-steered flow: the unsteered one rarely draws the 6-step chain.
    let t = TournamentTriadPipeline::new(1, 1)
        .unwrap()
        .with_robust(Some(sel(coin(), RobustObjective::PSuccess, false)));
    let specs = t.shard_plan(256, Some(5)).unwrap();
    let outcomes = t.run_tier1(&problem, &specs, None).unwrap();
    let elite_paths: Vec<&Vec<usize>> = outcomes[0].elites.iter().map(|v| &v.path).collect();
    let chain_path = vec![0, 1, 2, 3, 4, 8];
    let sampled_chain = outcomes[0]
        .verdicts
        .iter()
        .filter(|v| v.ok && v.path == chain_path)
        .count();
    assert!(
        sampled_chain > 0,
        "the chain was never sampled; the test would prove nothing"
    );
    assert_eq!(elite_paths.len(), 3);
    assert!(
        !elite_paths.contains(&&chain_path),
        "chain must not be a robust elite: {elite_paths:?}"
    );
    let arb = t.arbitrate(&problem, &outcomes).unwrap();
    let r = arb.robust.unwrap();
    let chain_ids: Vec<ActionId> = chain_path.iter().map(|&i| ids[i]).collect();
    assert_eq!(r.nominal_path, chain_ids);
    assert!(close(r.nominal_p_success, 22.0 / 64.0), "{r:?}");
    assert!(close(r.p_success, 0.5));
    // And it matches one flat nominal pass over the same samples.
    let (flat, _) = CausalGate::new(&dag, ctx0())
        .select(&outcomes[0].paths, None)
        .unwrap();
    assert_eq!(flat.unwrap().path, chain_path);
}

#[test]
fn nominal_tournament_is_unchanged_without_a_selector() {
    let dag = chain();
    let problem = TriadProblem::new(&dag);
    let t = TournamentTriadPipeline::new(2, 2)
        .unwrap()
        .with_energy_alpha(0.0)
        .unwrap();
    assert_eq!(t.elites_per_shard(), 2);
    let report = t.plan(&problem, 128, Some(3), None).unwrap();
    assert!(report.robust.is_none());
    assert_eq!(report.nominal_time, 6);
}

// ------------------------------------------------ production decide

fn pipeline() -> ProductionPipeline {
    ProductionPipeline::new(
        Arc::new(LatentDynamicsWorldModel::default()),
        Arc::new(PolicyGate::default()),
    )
}

fn decide(
    p: &ProductionPipeline,
    cands: &[ActionId],
    mode: DecideMode,
    dag: CausalDagSpec,
    options: TriadRunOptions,
) -> Result<gen_zero_planner::Decision, PlannerError> {
    let s = FullLatent::zeros();
    p.decide(&DecideRequest {
        active_context: Vec::new(),
        deadline: None,
        budget_ms: None,
        state: &s,
        candidates: cands,
        mode,
        entropy: NormalizedEntropy(0.2),
        return_trajectory: false,
        horizon: 4,
        causal_triad: Some(CausalTriadRequest { dag, options }),
    })
}

fn robust_opts(robust: Option<Value>) -> TriadRunOptions {
    let mut v = json!({"seed": 3, "n_samples": 128, "energy_alpha": 0.0});
    if let Some(r) = robust {
        v["robust"] = r;
    }
    serde_json::from_value(v).unwrap()
}

fn coin_json() -> Value {
    json!({"extra_pmf": [0.5, 0.5], "fatigue_frac": null, "fatigue_extra": 0, "kappa": null,
           "provenance": "test: fair coin"})
}

#[test]
fn decide_threads_the_robust_spec_through_both_triad_modes() {
    let p = pipeline();
    let ids = |v: &[usize]| -> Vec<ActionId> { v.iter().map(|&i| CHAIN_IDS[i]).collect() };
    for mode in [DecideMode::CausalTriad, DecideMode::TournamentTriad] {
        let plain = decide(&p, &CHAIN_IDS, mode, chain_spec(8), robust_opts(None)).unwrap();
        let r = plain.triad.unwrap();
        assert_eq!(r.chosen_path, ids(&CHAIN_PATH), "{mode:?}");
        assert!(r.robust.is_none());

        let robust = json!({"model": coin_json(), "objective": "p_success", "reorder": false});
        let d = decide(
            &p,
            &CHAIN_IDS,
            mode,
            chain_spec(8),
            robust_opts(Some(robust)),
        )
        .unwrap();
        let r = d.triad.unwrap();
        assert_eq!(r.chosen_path, ids(&BRANCH_PATH), "{mode:?}");
        assert_eq!(d.action, ActionId(1));
        assert_eq!(r.nominal_time, 7);
        let rb = r.robust.unwrap();
        assert_eq!(rb.objective, "p_success");
        assert!(close(rb.p_success, 0.5), "{rb:?}");
        assert_eq!(rb.nominal_path, ids(&CHAIN_PATH));
        assert!(close(rb.nominal_p_success, 22.0 / 64.0), "{rb:?}");
        assert!(!rb.reordered);
        assert_eq!((rb.probe_action, rb.posterior_kappa), (None, None));
    }
}

#[test]
fn decide_reorders_out_of_fatigue_and_commits_the_probe() {
    let p = pipeline();
    let fatigue = json!({"extra_pmf": [1.0], "fatigue_frac": 0.42, "fatigue_extra": 1});
    // Robust + reorder: y, z, x, t (P = 1), decision y.
    let d = decide(
        &p,
        &FAT_IDS,
        DecideMode::CausalTriad,
        fat_spec(),
        robust_opts(Some(json!({"model": fatigue}))),
    )
    .unwrap();
    let r = d.triad.unwrap();
    let rb = r.robust.unwrap();
    assert_eq!(rb.p_success, 1.0);
    assert_eq!(
        r.chosen_path[2],
        ActionId(1),
        "x deferred past both cheap items"
    );
    assert_eq!(rb.fatigue_at, Some(3));
    assert!(
        rb.nominal_p_success < 1.0,
        "nominal tie-break order hits fatigue: {rb:?}"
    );

    // COIN without reorder: every order ties, the gate key picks x,y,z,t. The
    // probe moves y (cheapest legal) first, so the decision changes x -> y.
    let probe = json!({"model": coin_json(), "reorder": false, "probe": true});
    let d = decide(
        &p,
        &FAT_IDS,
        DecideMode::CausalTriad,
        fat_spec(),
        robust_opts(Some(probe)),
    )
    .unwrap();
    let r = d.triad.unwrap();
    let rb = r.robust.unwrap();
    assert_eq!(rb.probe_action, Some(ActionId(2)));
    assert_eq!(d.action, ActionId(2));
    assert_eq!(
        r.chosen_path,
        vec![ActionId(2), ActionId(1), ActionId(3), ActionId(4)]
    );
    assert!(close(rb.p_success_before_probe.unwrap(), 5.0 / 16.0));
    assert!(close(rb.p_success, 5.0 / 16.0));
}

#[test]
fn decide_fails_closed_on_bad_robust_input_and_empty_gate() {
    let p = pipeline();
    // Out-of-support residual: refused, no nominal fallback.
    let robust = json!({"model": {"extra_pmf": [0.5, 0.5], "fatigue_frac": null, "fatigue_extra": 0, "kappa": 2.0},
                        "observations": [obs(1, 8, 0)]});
    let err = decide(
        &p,
        &CHAIN_IDS,
        DecideMode::CausalTriad,
        chain_spec(8),
        robust_opts(Some(robust)),
    )
    .unwrap_err();
    assert!(matches!(err, PlannerError::InvalidInput(_)), "{err}");
    // Bad pmf: refused.
    let robust = json!({"model": {"extra_pmf": [2.0], "fatigue_frac": null, "fatigue_extra": 0}});
    assert!(decide(
        &p,
        &CHAIN_IDS,
        DecideMode::TournamentTriad,
        chain_spec(8),
        robust_opts(Some(robust))
    )
    .is_err());
    // Nothing finishes nominally in 3: the robust selector never sees a plan.
    let robust = json!({"model": coin_json()});
    let err = decide(
        &p,
        &CHAIN_IDS,
        DecideMode::CausalTriad,
        chain_spec(3),
        robust_opts(Some(robust)),
    )
    .unwrap_err();
    assert!(matches!(err, PlannerError::CausalGateEmpty { .. }), "{err}");
}

// ------------------------------------------- hot-path cost

#[test]
fn hot_kernels_allocate_nothing_once_warm() {
    let dag = chain();
    let gate = CausalGate::new(&dag, ctx0());
    let model = coin();
    let mut scratch = ScoreScratch::default();
    let mut order = Vec::with_capacity(dag.len());
    // Warm-up sizes the scratch to the budget window.
    score_path_into(&dag, &model, &CHAIN_PATH, 0, &mut scratch).unwrap();
    let before = allocs();
    let mut acc = 0.0;
    for _ in 0..1000 {
        acc += score_path_into(&dag, &model, &CHAIN_PATH, 0, &mut scratch)
            .unwrap()
            .p_success;
        shortest_first_order(&gate, &CHAIN_PATH, &mut order).unwrap();
        acc += choose_probe_action(&gate, &CHAIN_PATH).unwrap() as f64;
    }
    let after = allocs();
    assert_eq!(
        after - before,
        0,
        "hot kernels allocated {} times",
        after - before
    );
    assert!(acc > 0.0);
}

/// Median ns per call over 41 batches of 1000 calls.
fn median_ns(mut f: impl FnMut()) -> f64 {
    let mut samples: Vec<f64> = (0..41)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..1000 {
                f();
            }
            t.elapsed().as_nanos() as f64 / 1000.0
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[20]
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "latency bound is a release-profile property: cargo test --release --test robust_gate_tests"
)]
fn hot_kernels_run_under_a_microsecond() {
    let dag = chain();
    let gate = CausalGate::new(&dag, ctx0());
    let model = coin();
    let fat_dag = fat();
    let fat_gate = CausalGate::new(&fat_dag, ctx0());
    let fatigue = fatigue_only();
    let mut scratch = ScoreScratch::default();
    let mut order = Vec::with_capacity(dag.len());
    let score = median_ns(|| {
        std::hint::black_box(
            score_path_into(
                &dag,
                &model,
                std::hint::black_box(&CHAIN_PATH),
                0,
                &mut scratch,
            )
            .unwrap(),
        );
    });
    let score_fat = median_ns(|| {
        std::hint::black_box(
            score_path_into(
                &fat_dag,
                &fatigue,
                std::hint::black_box(&[0, 1, 2, 3]),
                0,
                &mut scratch,
            )
            .unwrap(),
        );
    });
    let reorder = median_ns(|| {
        shortest_first_order(&fat_gate, std::hint::black_box(&[0, 1, 2, 3]), &mut order).unwrap();
        std::hint::black_box(&order);
    });
    let probe = median_ns(|| {
        std::hint::black_box(
            choose_probe_action(&gate, std::hint::black_box(&CHAIN_PATH)).unwrap(),
        );
    });
    println!(
        "median ns/call: score_chain {score:.1}, score_fatigue {score_fat:.1}, \
         shortest_first_order {reorder:.1}, choose_probe_action {probe:.1}"
    );
    for (name, ns) in [
        ("score_chain", score),
        ("score_fatigue", score_fat),
        ("shortest_first_order", reorder),
        ("choose_probe_action", probe),
    ] {
        assert!(ns < 1000.0, "{name}: {ns:.1} ns per call");
    }
}

// ------------------------------------------------ b1001n-t3-fix regressions

fn verdict_of(dag: &CausalDag, path: &[usize]) -> PathVerdict {
    let v = CausalGate::new(dag, ctx0()).check(path);
    assert!(v.ok, "{v:?}");
    v
}

#[test]
fn gate_refuses_a_clock_that_overflows_u32_instead_of_saturating() {
    // Two items of cost u32::MAX - 1: the nominal clock overflows on step 2.
    // A saturating clock would land on u32::MAX and report over_budget (or,
    // with budget u32::MAX, pass); the gate must report the overflow itself.
    let big = u32::MAX - 1;
    let spec: CausalDagSpec = serde_json::from_value(json!({
        "cost": {"1": big, "2": big}, "parents": {"2": [1]}, "target": 2, "budget": 1000
    }))
    .unwrap();
    let dag = CausalDag::from_spec(&[ActionId(1), ActionId(2)], &spec).unwrap();
    let v = CausalGate::new(&dag, ctx0()).check(&[0, 1]);
    assert!(!v.ok);
    assert_eq!(v.reason.as_str(), "time_overflow");
    assert_eq!(v.first_bad_step, Some(1));
    assert_eq!(v.nominal_time, big, "clock before the overflowing step");
}

#[test]
fn dag_refuses_a_budget_of_u32_max_and_anything_above_the_cap() {
    use gen_zero_planner::triad::{CausalEdge, CausalNode, MAX_TRIAD_BUDGET};
    let build = |budget: u32| {
        CausalDag::new(
            vec![CausalNode {
                action: ActionId(1),
                cost: 1,
                value: 0.0,
                is_or: false,
            }],
            &[] as &[CausalEdge],
            ActionId(1),
            budget,
        )
    };
    const { assert!(MAX_TRIAD_BUDGET < u32::MAX) };
    assert!(build(MAX_TRIAD_BUDGET).is_ok());
    for bad in [MAX_TRIAD_BUDGET + 1, u32::MAX] {
        let err = build(bad).unwrap_err();
        assert!(matches!(err, PlannerError::InvalidInput(_)), "{err}");
    }
}

#[test]
fn robust_order_does_not_saturate_on_huge_expected_time_or_slack() {
    // x * 1e9 of both values is past i64::MAX; an integer key saw them equal.
    let dag = fat();
    let v = verdict_of(&dag, &[0, 1, 2, 3]);
    let s = |p: f64, slack: f64, time: f64| gen_zero_planner::triad::RobustScore {
        p_success: p,
        expected_slack: slack,
        expected_time: time,
    };
    let et = sel(coin(), RobustObjective::ExpectedTime, false);
    let (fast, slow) = (s(0.5, 0.0, 1.0e10), s(0.5, 0.0, 1.0e10 + 1.0));
    assert_eq!(et.cmp((&fast, &v), (&slow, &v)), std::cmp::Ordering::Less);
    assert_eq!(
        et.cmp((&slow, &v), (&fast, &v)),
        std::cmp::Ordering::Greater
    );
    let ps = sel(coin(), RobustObjective::PSuccess, false);
    let (more, less) = (s(0.5, -1.0e10, 0.0), s(0.5, -1.0e10 - 1.0, 0.0));
    assert_eq!(ps.cmp((&more, &v), (&less, &v)), std::cmp::Ordering::Less);
    // Float noise below the quantum still ties, so the gate order decides.
    let w = verdict_of(&dag, &[1, 0, 2, 3]);
    let (a, b) = (s(0.5, 0.0, 3.0), s(0.5, 0.0, 3.0 + 1e-13));
    assert_eq!(
        et.cmp((&b, &v), (&a, &w)),
        gen_zero_planner::triad::arbiter_cmp(&v, &w)
    );
    // -0.0 and +0.0 slack are the same value.
    let (neg, pos) = (s(0.5, -1e-13, 0.0), s(0.5, 0.0, 0.0));
    assert_eq!(
        ps.cmp((&neg, &v), (&pos, &w)),
        gen_zero_planner::triad::arbiter_cmp(&v, &w)
    );
}

#[test]
fn probe_that_moves_the_plan_is_reported_as_reordered_and_counted() {
    use gen_zero_planner::triad::{ShardOutcome, ShardSpec};
    // Only x,y,z,t was sampled. The probe moves y (cheapest legal) first, so
    // the committed y,x,z,t is not a sampled plan. Distributions scored: the
    // one elite (rank), the nominal pick, the probe-first plan = 3.
    let dag = fat();
    let only = verdict_of(&dag, &[0, 1, 2, 3]);
    let outcome = ShardOutcome {
        spec: ShardSpec {
            index: 0,
            n_samples: 1,
            seed: Some(0),
        },
        paths: vec![only.path.clone()],
        log_pf: vec![0.0],
        verdicts: vec![only.clone()],
        elites: vec![only.clone()],
        nominal_best: Some(only.clone()),
        n_pass: 1,
        steps: 4,
        dead_ends: 0,
        wall_ms: 0.0,
    };
    let problem = TriadProblem::new(&dag);
    for (probe, path, reordered, n_scored) in [
        (true, vec![1, 0, 2, 3], true, 3),
        (false, vec![0, 1, 2, 3], false, 2),
    ] {
        let t = TournamentTriadPipeline::new(1, 1)
            .unwrap()
            .with_robust(Some(
                sel(coin(), RobustObjective::PSuccess, false).with_probe(probe),
            ));
        let arb = t
            .arbitrate(&problem, std::slice::from_ref(&outcome))
            .unwrap();
        let rb = arb.robust.unwrap();
        assert_eq!(arb.best.path, path);
        assert_eq!(rb.reordered, reordered, "probe = {probe}");
        assert_eq!(rb.n_scored, n_scored, "probe = {probe}");
    }
}

#[test]
fn probe_onto_a_sampled_non_elite_plan_is_not_reordered() {
    use gen_zero_planner::triad::{ShardOutcome, ShardSpec};
    // Shard 0 sampled x,y,z,t and y,x,z,t but kept only x,y,z,t as its elite.
    // The probe moves y first, so the committed y,x,z,t is a sampled plan:
    // reordered must be false. Shard 1 sampled only x,y,z,t, so on it alone
    // the same commit is outside every sampled set: reordered must be true.
    let dag = fat();
    let xyzt = verdict_of(&dag, &[0, 1, 2, 3]);
    let yxzt = verdict_of(&dag, &[1, 0, 2, 3]);
    assert!(xyzt.ok && yxzt.ok);
    let shard = |index: usize, sampled: Vec<PathVerdict>| ShardOutcome {
        spec: ShardSpec {
            index,
            n_samples: sampled.len(),
            seed: Some(index as u64),
        },
        paths: sampled.iter().map(|v| v.path.clone()).collect(),
        log_pf: vec![0.0; sampled.len()],
        n_pass: sampled.iter().filter(|v| v.ok).count(),
        verdicts: sampled,
        elites: vec![xyzt.clone()],
        nominal_best: Some(xyzt.clone()),
        steps: 4,
        dead_ends: 0,
        wall_ms: 0.0,
    };
    let both = shard(0, vec![xyzt.clone(), yxzt.clone()]);
    let only = shard(1, vec![xyzt.clone()]);
    assert!(!both.elites.iter().any(|e| e.path == yxzt.path));
    let problem = TriadProblem::new(&dag);
    let t = TournamentTriadPipeline::new(2, 1)
        .unwrap()
        .with_robust(Some(
            sel(coin(), RobustObjective::PSuccess, false).with_probe(true),
        ));
    for (outcomes, reordered) in [
        (vec![both.clone(), only.clone()], false),
        (vec![only.clone(), both.clone()], false),
        (vec![only.clone()], true),
    ] {
        let arb = t.arbitrate(&problem, &outcomes).unwrap();
        assert_eq!(arb.best.path, yxzt.path);
        let rb = arb.robust.unwrap();
        assert_eq!(
            rb.reordered,
            reordered,
            "shards {:?}",
            outcomes.iter().map(|o| o.spec.index).collect::<Vec<_>>()
        );
    }
}
