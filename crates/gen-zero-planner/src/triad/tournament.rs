//! Multi-core sharded tournament over the causal triad.
//!
//! Port of the gen-zero-research repository's `python/gen_zero/planner/triad/tournament.py`, with the shards on
//! real OS threads (`std::thread::scope`) instead of a GIL-bound pool. Every
//! shard owns its sampler, RNG, gate and output buffers; the only shared data
//! is the read-only DAG and problem, so there is no lock anywhere.
//!
//! - Tier 1: `M` shards each sample `K/M` plans with their own seed, judge every
//!   plan with the deterministic [`CausalGate`], and keep their top-`P`
//!   passing verdicts (the elites).
//! - Tier 2: the final arbiter re-runs the gate on every pooled elite (a
//!   mismatch with the shard's verdict is an internal error, never ignored) and
//!   picks the best by [`arbiter_cmp`], or under a [`RobustSlackSelector`].
//!
//! What Tier 2 does NOT buy: `arbiter_cmp` is a strict total order, so the best
//! of the pooled per-shard top-`P` (any `P >= 1`) is exactly the best over the
//! union of all shards' samples. The tournament never chooses a different plan
//! than one flat gate pass over the same samples. What sharding buys is
//! wall-clock (shards run in parallel) and seed diversity (each shard draws a
//! different slice of plan space under the same total budget `K`).
//!
//! With `shards == 1` the single shard runs on the calling thread; that is the
//! flat `CausalTriad` decide mode.
//!
//! Every shard samples under the same energy-steered flow
//! ([`GeodesicFlowConfig::energy_alpha`], default 2.0, set by
//! [`TournamentTriadPipeline::with_energy_alpha`]); the report echoes it.
//!
//! Before Tier 1, [`TournamentTriadPipeline::plan`] lays the DAG out on the
//! four Lod bands ([`CausalLod`]) and runs its fail-closed pre-check: a target
//! unreachable around the policy-blocked actions, or a budget below the cost
//! of the mandatory checkpoints, is refused as
//! [`PlannerError::CausalInfeasible`] without sampling. The report carries the
//! band layout and the hyperbolic goal distance of each committed step; the
//! goal distance is descriptive and takes no part in sampling or ranking.

use super::dag::CausalDag;
use super::gate::{arbiter_cmp, reason_histogram, CausalGate, GateContext, PathVerdict};
use super::pruner::{CausalPruner, EmptyActionSet};
use super::robust_gate::{RobustSlackSelector, ScoreScratch, REORDER_TOP};
use super::sampler::{GeodesicFlowConfig, GeodesicFlowSampler, SamplerStart};
use crate::error::PlannerError;
use gen_zero_core::ActionId;
use gen_zero_lod::{CausalLod, CausalLodContext, CausalLodError, CausalLodSummary};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::BTreeMap;
use std::time::Instant;

pub const MAX_TRIAD_SAMPLES: usize = 1 << 16;
pub const MAX_TRIAD_SHARDS: usize = 64;

/// Where planning starts: the DAG plus the live context.
#[derive(Copy, Clone, Debug)]
pub struct TriadProblem<'a> {
    pub dag: &'a CausalDag,
    /// Already-completed nodes.
    pub done: u64,
    pub time_used: u32,
    /// Nodes forbidden at every step (PolicyGate hard stops).
    pub blocked: u64,
    /// Nodes forbidden as the first step only (immediate world-model hazards).
    pub blocked_first: u64,
}

impl<'a> TriadProblem<'a> {
    pub fn new(dag: &'a CausalDag) -> Self {
        Self {
            dag,
            done: 0,
            time_used: 0,
            blocked: 0,
            blocked_first: 0,
        }
    }

    pub fn validate(&self) -> Result<(), PlannerError> {
        let n = self.dag.len();
        let all = if n == 64 { u64::MAX } else { (1_u64 << n) - 1 };
        if (self.done | self.blocked | self.blocked_first) & !all != 0 {
            return Err(PlannerError::InvalidInput(format!(
                "triad context masks reference nodes outside the {n}-node DAG"
            )));
        }
        if self.done & (1 << self.dag.target()) != 0 {
            return Err(PlannerError::InvalidInput(
                "the causal target is already done; there is nothing to plan".into(),
            ));
        }
        // A well-formed request that is over budget on arrival is infeasible
        // (422), the same contract as the Lod budget pre-check, not malformed.
        if self.time_used >= self.dag.budget() {
            return Err(PlannerError::CausalInfeasible(format!(
                "time_used {} leaves no budget (budget {})",
                self.time_used,
                self.dag.budget()
            )));
        }
        Ok(())
    }

    fn gate_context(&self) -> GateContext {
        GateContext {
            done: self.done,
            time_used: self.time_used,
            blocked: self.blocked,
            blocked_first: self.blocked_first,
        }
    }

    fn sampler_start(&self) -> SamplerStart {
        SamplerStart {
            done: self.done,
            time_used: self.time_used,
            blocked: self.blocked,
            blocked_first: self.blocked_first,
        }
    }
}

/// One shard's share of the sample budget.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ShardSpec {
    pub index: usize,
    pub n_samples: usize,
    /// `None`: the shard draws OS entropy. A seed is never invented for a caller
    /// that gave none.
    pub seed: Option<u64>,
}

/// Tier-1 result of one shard.
#[derive(Clone, Debug)]
pub struct ShardOutcome {
    pub spec: ShardSpec,
    /// Raw sampled node paths, in sample order.
    pub paths: Vec<Vec<usize>>,
    /// `log P_F` of each path under the masked flow policy, same order.
    pub log_pf: Vec<f64>,
    /// Gate verdict of each path, same order.
    pub verdicts: Vec<PathVerdict>,
    /// Top-`P` passing verdicts, best first.
    pub elites: Vec<PathVerdict>,
    /// Shard's own nominal pick, kept so the Tier-2 robust arbiter can echo it.
    pub nominal_best: Option<PathVerdict>,
    pub n_pass: usize,
    pub steps: usize,
    pub dead_ends: usize,
    pub wall_ms: f64,
}

/// The Lod layout a triad decision was planned under.
#[derive(Clone, Debug, PartialEq)]
pub struct TriadLodReport {
    pub summary: CausalLodSummary,
    /// Hyperbolic geodesic distance `d_H` from each committed step's Lod0 node
    /// to the Lod3 goal node, in plan order. Descriptive only.
    pub chosen_goal_distance: Vec<f64>,
}

/// How the robust selector arbitrated (present only with a [`RobustSpec`]).
///
/// [`RobustSpec`]: super::robust_gate::RobustSpec
#[derive(Clone, Debug, PartialEq)]
pub struct RobustReport {
    pub clock_phase: gen_zero_lod::manifold::ClockPhase,
    pub disturbance_moments: gen_zero_lod::manifold::DisturbanceMoments,
    /// `"p_success"` or `"expected_time"`.
    pub objective: &'static str,
    /// Scores of the committed plan (after reorder and probe) under the posterior model.
    pub p_success: f64,
    pub expected_slack: f64,
    pub expected_time: f64,
    /// The committed order is not one of the sampled plans.
    pub reordered: bool,
    /// Exact finish-time distributions evaluated in Tier 2: ranking and
    /// reorder search, the nominal pick, and the probe-first plan.
    pub n_scored: usize,
    /// The probe action, committed as the first step; `None` without `probe`.
    pub probe_action: Option<ActionId>,
    /// `P(finish <= budget)` of the selected plan before the probe moved first.
    pub p_success_before_probe: Option<f64>,
    /// What the nominal arbiter would have committed over the same samples,
    /// and its `P(finish <= budget)` under the same model.
    pub nominal_path: Vec<ActionId>,
    pub nominal_p_success: f64,
    /// Residuals the posterior was conditioned on.
    pub n_observed: usize,
    /// Prior and posterior Dirichlet strength; `None` = infinite.
    pub prior_kappa: Option<f64>,
    pub posterior_kappa: Option<f64>,
    pub extra_pmf: Vec<f64>,
    /// Clock at which fatigue starts; `None` without a fatigue rule.
    pub fatigue_at: Option<u64>,
}

/// Tier-2 result: the committed verdict and, with a selector, how it was chosen.
#[derive(Clone, Debug, PartialEq)]
pub struct Arbitration {
    pub best: PathVerdict,
    pub robust: Option<RobustReport>,
}

/// Everything a triad decision is based on.
#[derive(Clone, Debug, PartialEq)]
pub struct TriadReport {
    pub engine: &'static str,
    /// Causal-potential coupling every shard sampled with (0 = native flow).
    pub energy_alpha: f64,
    /// The committed plan; its first action is the decision.
    pub chosen_path: Vec<ActionId>,
    pub nominal_time: u32,
    pub net_reward: f64,
    /// Gate verdict counts over every sampled plan of every shard.
    pub gate_reasons: BTreeMap<&'static str, usize>,
    pub n_samples: usize,
    pub n_pass: usize,
    pub dead_ends: usize,
    pub sampled_steps: usize,
    /// Mean `log P_F` of the sampled plans: how concentrated the flow policy is.
    pub mean_log_pf: f64,
    /// Shannon entropy (bits) of the first action over all sampled plans.
    pub sampling_entropy_bits: f64,
    /// Entropy of the first action over passing plans, divided by ln(number of
    /// first actions the step-1 mask allows). 0 when at most one is allowed.
    pub decision_entropy: f64,
    pub shards: usize,
    pub top_p: usize,
    pub elites: usize,
    pub shard_seeds: Vec<Option<u64>>,
    pub shard_n_pass: Vec<usize>,
    pub shard_wall_ms: Vec<f64>,
    /// OS threads spawned for Tier 1 (0 when the single shard ran inline).
    pub threads_spawned: usize,
    pub tier1_ms: f64,
    pub tier2_ms: f64,
    pub plan_ms: f64,
    pub lod: TriadLodReport,
    /// Present when a robust selector arbitrated; `None` = nominal arbitration.
    pub robust: Option<RobustReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TournamentTriadPipeline {
    shards: usize,
    top_p: usize,
    flow: GeodesicFlowConfig,
    robust: Option<RobustSlackSelector>,
}

impl TournamentTriadPipeline {
    pub fn new(shards: usize, top_p: usize) -> Result<Self, PlannerError> {
        if !(1..=MAX_TRIAD_SHARDS).contains(&shards) {
            return Err(PlannerError::InvalidInput(format!(
                "shards must lie in 1..={MAX_TRIAD_SHARDS}, got {shards}"
            )));
        }
        if top_p < 1 {
            return Err(PlannerError::InvalidInput(format!(
                "top_p must be >= 1, got {top_p}"
            )));
        }
        Ok(Self {
            shards,
            top_p,
            flow: GeodesicFlowConfig::default(),
            robust: None,
        })
    }

    /// Same pipeline, ranked by `robust` in both tiers (`None`: nominal).
    pub fn with_robust(self, robust: Option<RobustSlackSelector>) -> Self {
        Self { robust, ..self }
    }

    pub fn robust(&self) -> Option<&RobustSlackSelector> {
        self.robust.as_ref()
    }

    /// Elites each shard keeps: `top_p`, raised to [`REORDER_TOP`] under a
    /// selector so the pooled elites hold the global top `REORDER_TOP`.
    pub fn elites_per_shard(&self) -> usize {
        match self.robust {
            Some(_) => self.top_p.max(REORDER_TOP),
            None => self.top_p,
        }
    }

    /// Same pipeline with causal-potential coupling `alpha` in every shard.
    /// Refused unless finite and in `[0, MAX_ENERGY_ALPHA]`.
    pub fn with_energy_alpha(self, alpha: f64) -> Result<Self, PlannerError> {
        let flow = GeodesicFlowConfig {
            energy_alpha: alpha,
            ..self.flow
        };
        flow.validate()?;
        Ok(Self { flow, ..self })
    }

    pub fn energy_alpha(&self) -> f64 {
        self.flow.energy_alpha
    }

    pub fn shards(&self) -> usize {
        self.shards
    }

    pub fn top_p(&self) -> usize {
        self.top_p
    }

    fn engine_name(&self) -> &'static str {
        if self.shards == 1 {
            "CausalTriadPipeline"
        } else {
            "TournamentTriadPipeline"
        }
    }

    /// Even split of `n_samples` (remainder one each to the first shards) and
    /// per-shard seeds `seed * shards + i`.
    pub fn shard_plan(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<Vec<ShardSpec>, PlannerError> {
        if n_samples < self.shards || n_samples > MAX_TRIAD_SAMPLES {
            return Err(PlannerError::InvalidInput(format!(
                "n_samples must lie in {}..={MAX_TRIAD_SAMPLES} for {} shard(s), got {n_samples}",
                self.shards, self.shards
            )));
        }
        let (base, rem) = (n_samples / self.shards, n_samples % self.shards);
        Ok((0..self.shards)
            .map(|i| ShardSpec {
                index: i,
                n_samples: base + usize::from(i < rem),
                seed: seed.map(|s| s.wrapping_mul(self.shards as u64).wrapping_add(i as u64)),
            })
            .collect())
    }

    /// Tier 1 for one shard. Public so a caller can run the exact same shard
    /// work serially (the parallel speed-up test does).
    pub fn run_shard(
        &self,
        problem: &TriadProblem<'_>,
        spec: &ShardSpec,
        deadline: Option<Instant>,
    ) -> Result<ShardOutcome, PlannerError> {
        let t0 = Instant::now();
        let mut rng = match spec.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let sampler = GeodesicFlowSampler::new(problem.dag, self.flow)?;
        let set = sampler.sample(
            &mut rng,
            spec.n_samples,
            problem.dag.len(),
            &problem.sampler_start(),
            deadline,
        )?;
        let gate = CausalGate::new(problem.dag, problem.gate_context());
        let (paths, log_pf): (Vec<Vec<usize>>, Vec<f64>) =
            set.paths.into_iter().map(|p| (p.actions, p.log_pf)).unzip();
        let verdicts: Vec<PathVerdict> = paths.iter().map(|p| gate.check(p)).collect();
        let elites = gate.top_passing(&verdicts, self.elites_per_shard(), self.robust.as_ref())?;
        let nominal_best = match self.robust {
            Some(_) => gate.top_passing(&verdicts, 1, None)?.into_iter().next(),
            None => elites.first().cloned(),
        };
        Ok(ShardOutcome {
            spec: *spec,
            n_pass: verdicts.iter().filter(|v| v.ok).count(),
            paths,
            log_pf,
            verdicts,
            elites,
            nominal_best,
            steps: set.steps,
            dead_ends: set.dead_ends,
            wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Run Tier 1 on all shards, in parallel when there is more than one.
    /// A failed or panicked shard fails the whole call; no shard is dropped.
    pub fn run_tier1(
        &self,
        problem: &TriadProblem<'_>,
        specs: &[ShardSpec],
        deadline: Option<Instant>,
    ) -> Result<Vec<ShardOutcome>, PlannerError> {
        if specs.len() == 1 {
            return Ok(vec![self.run_shard(problem, &specs[0], deadline)?]);
        }
        let joined: Vec<std::thread::Result<Result<ShardOutcome, PlannerError>>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = specs
                    .iter()
                    .map(|spec| scope.spawn(move || self.run_shard(problem, spec, deadline)))
                    .collect();
                // Join every handle before looking at any result, so a panic in
                // one shard is reported here instead of re-panicking the scope.
                handles.into_iter().map(|h| h.join()).collect()
            });
        joined
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                r.map_err(|_| {
                    PlannerError::ConvergenceFailure(format!("triad shard {i} panicked"))
                })?
            })
            .collect()
    }

    /// Tier 2: re-verify every pooled elite, then take the best: by
    /// `arbiter_cmp`, or under the robust selector (with reorder and probe).
    pub fn arbitrate(
        &self,
        problem: &TriadProblem<'_>,
        outcomes: &[ShardOutcome],
    ) -> Result<Arbitration, PlannerError> {
        let gate = CausalGate::new(problem.dag, problem.gate_context());
        let mut best: Option<&PathVerdict> = None;
        for elite in outcomes.iter().flat_map(|o| &o.elites) {
            let again = gate.check(&elite.path);
            if !again.ok || again != *elite {
                return Err(PlannerError::ConvergenceFailure(format!(
                    "tier-2 gate disagrees with tier-1 on elite {:?}: {:?}",
                    elite.path, again.reason
                )));
            }
            if best.is_none_or(|b| arbiter_cmp(elite, b).is_lt()) {
                best = Some(elite);
            }
        }
        let Some(nominal) = best else {
            let all: Vec<PathVerdict> = outcomes
                .iter()
                .flat_map(|o| o.verdicts.iter().cloned())
                .collect();
            return Err(PlannerError::CausalGateEmpty {
                sampled: all.len(),
                reasons: format!("{:?}", reason_histogram(&all)),
            });
        };
        let Some(sel) = &self.robust else {
            return Ok(Arbitration {
                best: nominal.clone(),
                robust: None,
            });
        };
        // The elites were ranked by the robust key, so the nominal winner may
        // not be among them; each shard's own nominal best is. arbiter_cmp is
        // a strict total order, so the best of those is the nominal pick over
        // the whole sample union.
        let mut nominal = nominal;
        for nb in outcomes.iter().filter_map(|o| o.nominal_best.as_ref()) {
            if gate.check(&nb.path) != *nb {
                return Err(PlannerError::ConvergenceFailure(format!(
                    "tier-2 gate disagrees with tier-1 on nominal best {:?}",
                    nb.path
                )));
            }
            if arbiter_cmp(nb, nominal).is_lt() {
                nominal = nb;
            }
        }
        let pooled: Vec<PathVerdict> = outcomes
            .iter()
            .flat_map(|o| o.elites.iter().cloned())
            .collect();
        let mut scratch = ScoreScratch::default();
        let choice = sel.select(&gate, &pooled, &mut scratch)?;
        let nominal_score = sel.score(&gate, &nominal.path, &mut scratch)?;
        // Distributions scored: every one `select` evaluated, the nominal pick,
        // and the probe-first plan when the probe is on.
        let mut n_scored = choice.n_scored + 1;
        let (committed, score, probe, before) = if sel.probe() {
            let (probed, probe) = sel.apply_probe(&gate, &choice.verdict, &mut scratch)?;
            n_scored += 1;
            (
                probed.verdict,
                probed.score,
                Some(probe),
                Some(choice.score.p_success),
            )
        } else {
            (choice.verdict, choice.score, None, None)
        };
        // Against every sampled path of every shard, not only the elites: a
        // reorder or probe can land on a plan a shard sampled but did not keep.
        let reordered = !outcomes
            .iter()
            .any(|o| o.paths.iter().any(|p| *p == committed.path));
        let actions = |path: &[usize]| -> Vec<ActionId> {
            path.iter().map(|&i| problem.dag.node(i).action).collect()
        };
        let finite = |k: f64| k.is_finite().then_some(k);
        let model = sel.model();
        let report = RobustReport {
            clock_phase: gen_zero_lod::manifold::ClockPhase::new(
                u64::from(problem.time_used),
                problem.dag.budget(),
                if model.fatigue_extra() > 0 {
                    model.fatigue_frac()
                } else {
                    None
                },
            )
            .map_err(|e| PlannerError::InvalidInput(format!("clock phase: {e}")))?,
            disturbance_moments: model.moments(),
            objective: sel.objective().as_str(),
            p_success: score.p_success,
            expected_slack: score.expected_slack,
            expected_time: score.expected_time,
            reordered,
            n_scored,
            probe_action: probe.map(|i| problem.dag.node(i).action),
            p_success_before_probe: before,
            nominal_path: actions(&nominal.path),
            nominal_p_success: nominal_score.p_success,
            n_observed: sel.n_observed(),
            prior_kappa: finite(sel.prior_kappa()),
            posterior_kappa: finite(model.kappa()),
            extra_pmf: model.extra_pmf().to_vec(),
            fatigue_at: model.fatigue_at(problem.dag.budget())?,
        };
        Ok(Arbitration {
            best: committed,
            robust: Some(report),
        })
    }

    /// Lay the DAG out on the Lod bands and refuse an infeasible problem
    /// before any plan is sampled.
    pub fn causal_lod(problem: &TriadProblem<'_>) -> Result<CausalLod, PlannerError> {
        let lod = CausalLod::build(
            problem.dag,
            CausalLodContext {
                done: problem.done,
                time_used: problem.time_used,
                blocked: problem.blocked,
            },
        )
        .map_err(lod_error)?;
        lod.check_feasible().map_err(lod_error)?;
        Ok(lod)
    }

    pub fn plan(
        &self,
        problem: &TriadProblem<'_>,
        n_samples: usize,
        seed: Option<u64>,
        deadline: Option<Instant>,
    ) -> Result<TriadReport, PlannerError> {
        let t0 = Instant::now();
        problem.validate()?;
        let specs = self.shard_plan(n_samples, seed)?;
        let lod = Self::causal_lod(problem)?;
        let outcomes = self.run_tier1(problem, &specs, deadline)?;
        let tier1_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let Arbitration { best, robust } = self.arbitrate(problem, &outcomes)?;
        let tier2_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let verdicts: Vec<&PathVerdict> = outcomes.iter().flat_map(|o| &o.verdicts).collect();
        let mut gate_reasons = BTreeMap::new();
        for v in &verdicts {
            *gate_reasons.entry(v.reason.as_str()).or_insert(0) += 1;
        }
        let first_all = outcomes
            .iter()
            .flat_map(|o| o.paths.iter().filter_map(|p| p.first().copied()));
        let first_pass = verdicts.iter().filter(|v| v.ok).map(|v| v.path[0]);
        let step1_support = match CausalPruner::new(problem.dag, true)
            .prune(problem.done, problem.blocked | problem.blocked_first)
        {
            Ok(r) => r.n_allowed,
            Err(EmptyActionSet { .. }) => 0,
        };
        let decision_entropy = if step1_support > 1 {
            entropy_bits(first_pass) / f64::from(step1_support).log2()
        } else {
            0.0
        };
        Ok(TriadReport {
            engine: self.engine_name(),
            energy_alpha: self.flow.energy_alpha,
            chosen_path: best
                .path
                .iter()
                .map(|&i| problem.dag.node(i).action)
                .collect(),
            nominal_time: best.nominal_time,
            net_reward: best.net_reward,
            n_samples,
            n_pass: gate_reasons.get("ok").copied().unwrap_or(0),
            gate_reasons,
            dead_ends: outcomes.iter().map(|o| o.dead_ends).sum(),
            sampled_steps: outcomes.iter().map(|o| o.steps).sum(),
            mean_log_pf: outcomes.iter().flat_map(|o| &o.log_pf).sum::<f64>() / n_samples as f64,
            sampling_entropy_bits: entropy_bits(first_all),
            decision_entropy: decision_entropy.clamp(0.0, 1.0),
            shards: self.shards,
            top_p: self.top_p,
            elites: outcomes.iter().map(|o| o.elites.len()).sum(),
            shard_seeds: specs.iter().map(|s| s.seed).collect(),
            shard_n_pass: outcomes.iter().map(|o| o.n_pass).collect(),
            shard_wall_ms: outcomes.iter().map(|o| o.wall_ms).collect(),
            threads_spawned: if self.shards > 1 { self.shards } else { 0 },
            tier1_ms,
            tier2_ms,
            plan_ms: t0.elapsed().as_secs_f64() * 1000.0,
            lod: TriadLodReport {
                summary: lod.summary(),
                chosen_goal_distance: best
                    .path
                    .iter()
                    .map(|&i| lod.goal_distance(i))
                    .collect::<Result<_, _>>()
                    .map_err(lod_error)?,
            },
            robust,
        })
    }
}

/// Infeasibility is the caller's problem (422); a malformed view is invalid
/// input; a graph refusal on a DAG that passed validation is an internal fault.
fn lod_error(e: CausalLodError) -> PlannerError {
    match e {
        CausalLodError::Invalid(m) => PlannerError::InvalidInput(m),
        CausalLodError::Graph(g) => {
            PlannerError::ConvergenceFailure(format!("causal Lod graph refused: {g}"))
        }
        infeasible @ (CausalLodError::TargetUnreachable { .. }
        | CausalLodError::BudgetInsufficient { .. }) => {
            PlannerError::CausalInfeasible(infeasible.to_string())
        }
    }
}

/// Shannon entropy in bits of the empirical distribution of `items`; 0 when empty.
fn entropy_bits(items: impl Iterator<Item = usize>) -> f64 {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    let mut total = 0_usize;
    for i in items {
        *counts.entry(i).or_insert(0) += 1;
        total += 1;
    }
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    -counts
        .values()
        .map(|&c| {
            let p = c as f64 / total;
            p * p.log2()
        })
        .sum::<f64>()
}
