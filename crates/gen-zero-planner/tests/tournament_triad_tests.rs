//! Multi-core sharded tournament: shard plan, Tier-1 / Tier-2 consistency with a
//! flat gate over the same samples, brute-force optimality, real parallel
//! wall-clock speed-up, fail-closed paths, the `DecideMode::TournamentTriad`
//! production route, and the energy-steered flow in every shard.
//!
//! Every test takes `SERIAL` so the timing test does not compete with the other
//! tests of this binary for cores.

use gen_zero_core::{ActionId, FullLatent, NormalizedEntropy};
use gen_zero_gate::PolicyGate;
use gen_zero_planner::triad::{
    arbiter_cmp, CausalDag, CausalDagSpec, CausalEdge, CausalGate, CausalNode, CausalPruner,
    GateContext, GeodesicFlowConfig, GeodesicFlowSampler, PathVerdict, SamplerStart, ShardSpec,
    TournamentTriadPipeline, TriadProblem, TriadRunOptions,
};
use gen_zero_planner::{
    CausalTriadRequest, DecideMode, DecideRequest, PlannerError, ProductionPipeline,
};
use gen_zero_worldmodel::LatentDynamicsWorldModel;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Target 6 is OR over three branches with different cost/value trade-offs:
///   5 -> 6            t=3, net  0 - 3 = -3
///   1 -> 2 -> 6       t=4, net  3 - 4 = -1
///   3 -> 4 -> 6       t=6, net 10 - 6 =  4   (best)
fn branchy() -> CausalDag {
    let spec: CausalDagSpec = serde_json::from_value(json!({
        "parents": {"2": [1], "4": [3], "6": [2, 4, 5]},
        "is_or": {"6": true},
        "cost": {"1": 1, "2": 2, "3": 4, "4": 1, "5": 2, "6": 1},
        "value": {"2": 3.0, "3": 9.0, "4": 1.0},
        "target": 6,
        "budget": 30
    }))
    .unwrap();
    let cands: Vec<ActionId> = (1..=6).map(ActionId).collect();
    CausalDag::from_spec(&cands, &spec).unwrap()
}

/// Layered DAG: `layers` x `width` nodes, each node AND over two nodes of the
/// previous layer, target AND over the whole last layer. Every plan must run
/// nearly the full DAG, so each sample is long.
fn layered(layers: u32, width: u32) -> CausalDag {
    let id = |l: u32, k: u32| l * width + k;
    let target = layers * width;
    let mut nodes: Vec<CausalNode> = (0..=target)
        .map(|a| CausalNode {
            action: ActionId(a),
            cost: 1 + a % 3,
            value: f64::from(a % 5) * 0.25,
            is_or: false,
        })
        .collect();
    nodes[target as usize].cost = 1;
    let mut edges = Vec::new();
    for l in 1..layers {
        for k in 0..width {
            for p in [k, (k + 1) % width] {
                edges.push(CausalEdge {
                    parent: ActionId(id(l - 1, p)),
                    child: ActionId(id(l, k)),
                });
            }
        }
    }
    for k in 0..width {
        edges.push(CausalEdge {
            parent: ActionId(id(layers - 1, k)),
            child: ActionId(target),
        });
    }
    CausalDag::new(nodes, &edges, ActionId(target), 10_000).unwrap()
}

/// Every plan from the empty start that the gate accepts, by exhaustive DFS.
fn brute_force_best(dag: &CausalDag) -> PathVerdict {
    let gate = CausalGate::new(
        dag,
        GateContext {
            done: 0,
            time_used: 0,
            blocked: 0,
            blocked_first: 0,
        },
    );
    let pruner = CausalPruner::new(dag, false);
    let mut best: Option<PathVerdict> = None;
    let mut stack = vec![(0_u64, Vec::<usize>::new())];
    while let Some((done, path)) = stack.pop() {
        if path.last() == Some(&dag.target()) {
            let v = gate.check(&path);
            if v.ok && best.as_ref().is_none_or(|b| arbiter_cmp(&v, b).is_lt()) {
                best = Some(v);
            }
            continue;
        }
        for a in 0..dag.len() {
            if pruner.is_admissible(done, a) {
                let mut next = path.clone();
                next.push(a);
                stack.push((done | (1 << a), next));
            }
        }
    }
    best.expect("the DAG has a valid plan")
}

#[test]
fn shard_plan_splits_samples_evenly_and_derives_seeds() {
    let _g = serial();
    let t = TournamentTriadPipeline::new(4, 2).unwrap();
    let specs = t.shard_plan(257, Some(10)).unwrap();
    let sizes: Vec<usize> = specs.iter().map(|s| s.n_samples).collect();
    assert_eq!(sizes, vec![65, 64, 64, 64]);
    let seeds: Vec<Option<u64>> = specs.iter().map(|s| s.seed).collect();
    assert_eq!(seeds, vec![Some(40), Some(41), Some(42), Some(43)]);
    // No seed asked for, none invented.
    assert!(t
        .shard_plan(8, None)
        .unwrap()
        .iter()
        .all(|s| s.seed.is_none()));
    // Fail-closed configuration.
    assert!(matches!(
        t.shard_plan(3, None),
        Err(PlannerError::InvalidInput(_))
    ));
    assert!(matches!(
        TournamentTriadPipeline::new(0, 2),
        Err(PlannerError::InvalidInput(_))
    ));
    assert!(matches!(
        TournamentTriadPipeline::new(4, 0),
        Err(PlannerError::InvalidInput(_))
    ));
    assert!(matches!(
        TournamentTriadPipeline::new(65, 1),
        Err(PlannerError::InvalidInput(_))
    ));
}

#[test]
fn tier2_final_arbiter_equals_a_flat_gate_over_the_union_of_shard_samples() {
    let _g = serial();
    let dag = branchy();
    let problem = TriadProblem::new(&dag);
    let t = TournamentTriadPipeline::new(4, 2).unwrap();
    for seed in [1_u64, 2, 3, 99, 12345] {
        let report = t.plan(&problem, 64, Some(seed), None).unwrap();
        let specs = t.shard_plan(64, Some(seed)).unwrap();
        let outcomes = t.run_tier1(&problem, &specs, None).unwrap();
        assert_eq!(outcomes.len(), 4);
        // Tier 1: at most P elites per shard, best first, all passing.
        for o in &outcomes {
            assert!(o.elites.len() <= 2);
            assert!(o.elites.iter().all(|e| e.ok));
            assert!(o
                .elites
                .windows(2)
                .all(|w| arbiter_cmp(&w[0], &w[1]).is_lt()));
            assert_eq!(o.paths.len(), o.spec.n_samples);
        }
        // Flat gate over the union of the same samples.
        let union: Vec<Vec<usize>> = outcomes.iter().flat_map(|o| o.paths.clone()).collect();
        let gate = CausalGate::new(
            &dag,
            GateContext {
                done: 0,
                time_used: 0,
                blocked: 0,
                blocked_first: 0,
            },
        );
        let (flat, verdicts) = gate.select(&union, None).unwrap();
        let flat = flat.unwrap();
        let flat_ids: Vec<ActionId> = flat.path.iter().map(|&i| dag.node(i).action).collect();
        assert_eq!(report.chosen_path, flat_ids, "seed {seed}");
        assert_eq!(report.nominal_time, flat.nominal_time);
        assert_eq!(report.n_pass, verdicts.iter().filter(|v| v.ok).count());
        assert_eq!(report.gate_reasons.values().sum::<usize>(), 64);
        // Changing P never changes the choice.
        for p in [1, 3, 8] {
            let other = TournamentTriadPipeline::new(4, p).unwrap();
            assert_eq!(
                other
                    .plan(&problem, 64, Some(seed), None)
                    .unwrap()
                    .chosen_path,
                report.chosen_path,
                "seed {seed}, top_p {p}"
            );
        }
    }
}

#[test]
fn tournament_finds_the_brute_force_optimum() {
    let _g = serial();
    let dag = branchy();
    let best = brute_force_best(&dag);
    let best_ids: Vec<ActionId> = best.path.iter().map(|&i| dag.node(i).action).collect();
    assert_eq!(best_ids, vec![ActionId(3), ActionId(4), ActionId(6)]);
    assert_eq!(best.net_reward, 4.0);
    let t = TournamentTriadPipeline::new(4, 2).unwrap();
    for seed in [7_u64, 8, 9] {
        let r = t
            .plan(&TriadProblem::new(&dag), 512, Some(seed), None)
            .unwrap();
        assert_eq!(r.chosen_path, best_ids, "seed {seed}");
        assert_eq!(r.net_reward, 4.0);
        assert_eq!(r.shards, 4);
        assert_eq!(r.threads_spawned, 4);
        assert!(r.elites <= 8 && r.elites >= 1);
        assert_eq!(r.shard_n_pass.iter().sum::<usize>(), r.n_pass);
    }
}

#[test]
fn four_shards_on_threads_beat_the_same_shards_run_serially() {
    let _g = serial();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    if cores < 4 {
        eprintln!(
            "skipping test four_shards_on_threads_beat_the_same_shards_run_serially: requires >= 4 cores, found {cores}"
        );
        return;
    }
    let dag = layered(8, 6);
    assert_eq!(dag.len(), 49);
    let problem = TriadProblem::new(&dag);
    let t = TournamentTriadPipeline::new(4, 2).unwrap();
    let per_shard = if cfg!(debug_assertions) { 60 } else { 1200 };
    let specs: Vec<ShardSpec> = t.shard_plan(4 * per_shard, Some(2026)).unwrap();

    let mut best_serial = Duration::MAX;
    let mut best_parallel = Duration::MAX;
    let mut serial_out = Vec::new();
    let mut parallel_out = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        serial_out = specs
            .iter()
            .map(|s| t.run_shard(&problem, s, None).unwrap())
            .collect::<Vec<_>>();
        best_serial = best_serial.min(t0.elapsed());

        let t0 = Instant::now();
        parallel_out = t.run_tier1(&problem, &specs, None).unwrap();
        best_parallel = best_parallel.min(t0.elapsed());
    }
    // Same seeds, same work: identical samples and elites on threads.
    for (a, b) in serial_out.iter().zip(&parallel_out) {
        assert_eq!(a.paths, b.paths);
        assert_eq!(a.elites, b.elites);
    }
    let steps: usize = parallel_out.iter().map(|o| o.steps).sum();
    let ratio = best_parallel.as_secs_f64() / best_serial.as_secs_f64();
    println!(
        "tier1 4 shards x {per_shard} samples, {steps} sampled steps on {cores} cores: \
         serial {:.1} ms, parallel {:.1} ms, parallel/serial = {ratio:.3}, speed-up {:.2}x",
        best_serial.as_secs_f64() * 1e3,
        best_parallel.as_secs_f64() * 1e3,
        1.0 / ratio
    );
    assert!(
        best_serial >= Duration::from_millis(100),
        "workload too small to time: serial {best_serial:?}"
    );
    assert!(ratio < 0.5, "parallel/serial = {ratio:.3}, expected < 0.5");

    // And the full plan over those shards picks the flat optimum of the union.
    let report = t.plan(&problem, 4 * per_shard, Some(2026), None).unwrap();
    let union: Vec<Vec<usize>> = parallel_out.iter().flat_map(|o| o.paths.clone()).collect();
    let (flat, _) = CausalGate::new(
        &dag,
        GateContext {
            done: 0,
            time_used: 0,
            blocked: 0,
            blocked_first: 0,
        },
    )
    .select(&union, None)
    .unwrap();
    let flat_ids: Vec<ActionId> = flat
        .unwrap()
        .path
        .iter()
        .map(|&i| dag.node(i).action)
        .collect();
    assert_eq!(report.chosen_path, flat_ids);
    assert!(dag.respects_topology(
        &report
            .chosen_path
            .iter()
            .map(|&a| dag.index_of(a).unwrap())
            .collect::<Vec<_>>()
    ));
}

#[test]
fn tournament_fails_closed() {
    let _g = serial();
    let dag = branchy();
    let t = TournamentTriadPipeline::new(4, 2).unwrap();
    let mut problem = TriadProblem::new(&dag);
    let heads = (1 << dag.index_of(ActionId(1)).unwrap())
        | (1 << dag.index_of(ActionId(3)).unwrap())
        | (1 << dag.index_of(ActionId(5)).unwrap());
    // Block every branch head at every step: the Lod pre-check refuses before
    // any sample is drawn and names the barriers.
    problem.blocked = heads;
    match t.plan(&problem, 32, Some(1), None) {
        Err(PlannerError::CausalInfeasible(m)) => {
            assert!(m.contains("unreachable"), "{m}");
            assert!(m.contains("[1, 3, 5]"), "{m}");
        }
        other => panic!("expected CausalInfeasible, got {other:?}"),
    }
    // Block them as the first step only: the pre-check cannot see a first-step
    // block, so the shards sample, every one dead-ends, and the gate is empty.
    let mut first = TriadProblem::new(&dag);
    first.blocked_first = heads;
    match t.plan(&first, 32, Some(1), None) {
        Err(PlannerError::CausalGateEmpty { sampled, reasons }) => {
            assert_eq!(sampled, 32);
            assert!(reasons.contains("no_target"), "{reasons}");
        }
        other => panic!("expected CausalGateEmpty, got {other:?}"),
    }
    // Expired deadline: a timeout, never a partial pick.
    let past = Instant::now() - Duration::from_millis(1);
    assert!(matches!(
        t.plan(&TriadProblem::new(&dag), 32, Some(1), Some(past)),
        Err(PlannerError::TimeoutExceeded(_))
    ));
    // Target already done.
    let mut done = TriadProblem::new(&dag);
    done.done = 1 << dag.target();
    assert!(matches!(
        t.plan(&done, 32, None, None),
        Err(PlannerError::InvalidInput(_))
    ));
    // Over budget on arrival is infeasible (422 at the service), not malformed.
    let mut late = TriadProblem::new(&dag);
    late.time_used = 30;
    match t.plan(&late, 32, None, None) {
        Err(PlannerError::CausalInfeasible(m)) => {
            assert!(m.contains("time_used 30 leaves no budget"), "{m}")
        }
        other => panic!("expected CausalInfeasible, got {other:?}"),
    }
    // Masks outside the DAG.
    let mut wide = TriadProblem::new(&dag);
    wide.blocked = 1 << 40;
    assert!(matches!(
        t.plan(&wide, 32, None, None),
        Err(PlannerError::InvalidInput(_))
    ));
}

fn tournament_req<'a>(
    state: &'a FullLatent,
    candidates: &'a [ActionId],
    options: TriadRunOptions,
) -> DecideRequest<'a> {
    DecideRequest {
        active_context: Vec::new(),
        deadline: None,
        budget_ms: None,
        state,
        candidates,
        mode: DecideMode::TournamentTriad,
        entropy: NormalizedEntropy(0.2),
        return_trajectory: false,
        horizon: 4,
        causal_triad: Some(CausalTriadRequest {
            dag: serde_json::from_value(json!({
                "parents": {"2": [1], "4": [3], "6": [2, 4, 5]},
                "is_or": {"6": true},
                "costs": {"1": 1, "2": 2, "3": 4, "4": 1, "5": 2, "6": 1},
                "values": {"2": 3.0, "3": 9.0, "4": 1.0},
                "target": 6,
                "budget": 30
            }))
            .unwrap(),
            options,
        }),
    }
}

#[test]
fn decide_tournament_triad_runs_sharded_on_threads_in_production() {
    let _g = serial();
    let p = ProductionPipeline::new(
        Arc::new(LatentDynamicsWorldModel::default()),
        Arc::new(PolicyGate::default()),
    );
    let s = FullLatent::zeros();
    let cands: Vec<ActionId> = (1..=6).map(ActionId).collect();
    let d = p
        .decide(&tournament_req(
            &s,
            &cands,
            TriadRunOptions {
                seed: Some(77),
                ..TriadRunOptions::default()
            },
        ))
        .unwrap();
    assert_eq!(d.mode, DecideMode::TournamentTriad);
    assert_eq!(d.engine, "TournamentTriadPipeline");
    let r = d.triad.as_ref().unwrap();
    assert_eq!(
        (r.shards, r.top_p, r.n_samples, r.threads_spawned),
        (4, 2, 256, 4)
    );
    assert_eq!(
        r.shard_seeds,
        vec![Some(308), Some(309), Some(310), Some(311)]
    );
    assert_eq!(r.gate_reasons.values().sum::<usize>(), 256);
    assert_eq!(r.chosen_path, vec![ActionId(3), ActionId(4), ActionId(6)]);
    assert_eq!(d.action, ActionId(3));
    assert!(d.feasible.contains(&d.action));
    // Three first steps are causally open and the passing plans use them all.
    assert!(d.entropy.0 > 0.0 && d.entropy.0 <= 1.0, "{:?}", d.entropy);

    // Custom shard count and the deadline-worker path.
    let mut req = tournament_req(
        &s,
        &cands,
        TriadRunOptions {
            shards: Some(8),
            top_p: Some(1),
            n_samples: Some(64),
            seed: Some(1),
            ..TriadRunOptions::default()
        },
    );
    req.budget_ms = Some(5_000.0);
    let d = p.decide(&req).unwrap();
    let r = d.triad.unwrap();
    assert_eq!((r.shards, r.threads_spawned, r.elites <= 8), (8, 8, true));

    // More shards than samples is refused, not clamped.
    let e = p
        .decide(&tournament_req(
            &s,
            &cands,
            TriadRunOptions {
                shards: Some(8),
                n_samples: Some(4),
                ..TriadRunOptions::default()
            },
        ))
        .unwrap_err();
    assert!(matches!(e, PlannerError::InvalidInput(_)), "{e}");
}

#[test]
fn every_shard_samples_under_the_tournament_energy_alpha() {
    let _g = serial();
    let dag = branchy();
    let problem = TriadProblem::new(&dag);
    for alpha in [0.0, 2.0, 5.0] {
        let t = TournamentTriadPipeline::new(4, 2)
            .unwrap()
            .with_energy_alpha(alpha)
            .unwrap();
        let specs = t.shard_plan(64, Some(21)).unwrap();
        let outcomes = t.run_tier1(&problem, &specs, None).unwrap();
        let cfg = GeodesicFlowConfig {
            energy_alpha: alpha,
            ..GeodesicFlowConfig::default()
        };
        let sampler = GeodesicFlowSampler::new(&dag, cfg).unwrap();
        let start = SamplerStart {
            done: 0,
            time_used: 0,
            blocked: 0,
            blocked_first: 0,
        };
        for (o, spec) in outcomes.iter().zip(&specs) {
            let mut rng = StdRng::seed_from_u64(spec.seed.unwrap());
            let set = sampler
                .sample(&mut rng, spec.n_samples, dag.len(), &start, None)
                .unwrap();
            let paths: Vec<Vec<usize>> = set.paths.into_iter().map(|p| p.actions).collect();
            assert_eq!(o.paths, paths, "alpha {alpha}, shard {}", spec.index);
        }
        assert_eq!(
            t.plan(&problem, 64, Some(21), None).unwrap().energy_alpha,
            alpha
        );
    }
}

#[test]
fn energy_steering_finds_the_net_best_plan_with_fewer_samples() {
    // branchy's net-best plan 3 -> 4 -> 6 starts with its slowest node. The
    // net-cost potential points at it; the native flow finds it by luck.
    let _g = serial();
    let dag = branchy();
    let problem = TriadProblem::new(&dag);
    let best = vec![ActionId(3), ActionId(4), ActionId(6)];
    let hits = |alpha: f64| {
        let t = TournamentTriadPipeline::new(4, 2)
            .unwrap()
            .with_energy_alpha(alpha)
            .unwrap();
        (0..40_u64)
            .filter(|&seed| t.plan(&problem, 8, Some(seed), None).unwrap().chosen_path == best)
            .count()
    };
    let (native, steered) = (hits(0.0), hits(2.0));
    println!(
        "[energy-tournament rs K=8 x 40 seeds] net-best found native={native} steered={steered}"
    );
    assert_eq!(steered, 40);
    assert!(native < 35, "{native}");
}

#[test]
fn decide_tournament_triad_echoes_and_validates_energy_alpha() {
    let _g = serial();
    let p = ProductionPipeline::new(
        Arc::new(LatentDynamicsWorldModel::default()),
        Arc::new(PolicyGate::default()),
    );
    let s = FullLatent::zeros();
    let cands: Vec<ActionId> = (1..=6).map(ActionId).collect();
    let run = |alpha: Option<f64>| {
        p.decide(&tournament_req(
            &s,
            &cands,
            TriadRunOptions {
                seed: Some(5),
                n_samples: Some(32),
                energy_alpha: alpha,
                ..TriadRunOptions::default()
            },
        ))
    };
    assert_eq!(run(None).unwrap().triad.unwrap().energy_alpha, 2.0);
    assert_eq!(run(Some(0.0)).unwrap().triad.unwrap().energy_alpha, 0.0);
    for bad in [f64::INFINITY, -2.0, 65.0] {
        assert!(
            matches!(run(Some(bad)), Err(PlannerError::InvalidInput(_))),
            "{bad}"
        );
    }
}
