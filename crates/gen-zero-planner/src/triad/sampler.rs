//! Geodesic flow sampler: stage 2 of the causal triad.
//!
//! Port of the gen-zero-research repository's `python/gen_zero/planner/triad/sampler.py` (`MaskedTriadSampler`).
//! Every DAG node gets a vertex of the production Helmert simplex ETF
//! ([`SimplexEtfFrame`]). A latent `z` lives on the unit sphere. Per step:
//!
//! 1. the [`CausalPruner`] mask gives the allowed nodes (fail-closed dead end
//!    when none survive);
//! 2. forward policy `P_F(a | z) = softmax(<z, e_a> / T)` over allowed nodes;
//! 3. sample `a`, apply the nominal transition (cost, budget clock);
//! 4. move `z` along the great circle toward `e_a` (slerp, step 0.35), then add
//!    isotropic Gaussian noise (sigma 0.03) and re-project onto the sphere.
//!
//! The hyperparameters match the Python `_GFN_KW`. This is an UNTRAINED
//! proposal distribution: there is no trajectory-balance update and no learned
//! flow. Its job is to propose diverse admissible plans; the deterministic gate
//! decides which one is valid and best.
//!
//! Energy steering (`energy_alpha`, default [`DEFAULT_ENERGY_ALPHA`] = 2.0): step
//! 2 draws from `softmax(<z, e_a> / T + alpha * dPhi(s, a))` over the allowed
//! nodes, with the causal potential `Phi(s) = -(g(s) + h(s))`: `g` the cost
//! already paid, `h` the greedy feasible closure cost to the target
//! ([`CausalPotential::closure`], in `gen-zero-lod`: the potential is computed
//! on the Lod1 reading of the DAG, AND clusters as unions and OR clusters as
//! their cheapest member). So `dPhi(s, a) = -(c[a] + h(s + a) - h(s))`.
//! `alpha = 0` skips the bias and draws exactly the native paths.
//!
//! The per-node cost `c[a]` is `cost[a] - value[a]`: the loss the Rust gate's
//! arbiter ranks by ([`super::gate::arbiter_cmp`] maximises
//! `net_reward = sum(value) - time`). The Python gate ranks by time alone, so the
//! Python sampler uses `c[a] = cost[a]`. Each side's potential measures what its
//! own gate ranks by. With every value 0 the two are the same function, and the
//! shared golden fixture (`tests/fixtures/energy_potential_golden.json`, from
//! the research `CausalBounds.closure`) checks that exactly. A time-only
//! potential here steered the tournament away from the net-best plan (the
//! `tournament_finds_the_brute_force_optimum` test caught it).
//!
//! What this is, per the controlled ablation
//! (gen-zero-research `docs/research/gflownet-causal-triad/02_evolution_ablation_report.md` lines
//! 13-17 and 239): an A*-style `g + h` heuristic bias on the proposal. The
//! measured gain belongs to the potential, not to the flow geometry (a uniform
//! proposal with the same bias tied it), on layered AND/OR synthetic graphs with
//! goal cones <= 16 nodes and budget 1.25 x the optimum, under the TIME
//! objective. The net-cost variant here is not covered by that ablation; its
//! only evidence is the paired test in `tests/causal_triad_tests.rs`. `h` can be
//! wrong in either direction, so it only biases sampling; the gate alone
//! decides admissibility, on-time and the winner.
//!
//! Start latent: deterministic in the start state (done set, clock), like the
//! Python `initialize_latent_state(str(state))`. Diversity comes from the
//! sampling RNG and the noise.

use super::dag::CausalDag;
use super::pruner::{CausalPruner, EmptyActionSet};
use crate::error::PlannerError;
use gen_zero_core::SimplexEtfFrame;
use gen_zero_lod::causal_lod::BitIter;
use gen_zero_lod::{CausalPotential, MAX_CAUSAL_ATOMS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// Coupling of the causal-potential bias. Frozen on the research dev set
/// (gen-zero-research `evolution_ablation.py` `FROZEN["alpha"]`), never tuned on test graphs.
pub const DEFAULT_ENERGY_ALPHA: f64 = 2.0;
/// Upper bound of `energy_alpha`. A negative coupling would steer away from the
/// target and a huge one overflows `exp`; both are refused, never clamped.
pub const MAX_ENERGY_ALPHA: f64 = 64.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GeodesicFlowConfig {
    pub dim: usize,
    pub temperature: f64,
    pub slerp_step: f64,
    pub noise: f64,
    /// Causal-potential coupling, in `[0, MAX_ENERGY_ALPHA]`. 0 = native flow.
    pub energy_alpha: f64,
}

impl Default for GeodesicFlowConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            temperature: 1.0,
            slerp_step: 0.35,
            noise: 0.03,
            energy_alpha: DEFAULT_ENERGY_ALPHA,
        }
    }
}

impl GeodesicFlowConfig {
    pub fn validate(&self) -> Result<(), PlannerError> {
        let ok = self.dim >= 1
            && self.temperature.is_finite()
            && self.temperature > 0.0
            && self.slerp_step.is_finite()
            && (0.0..=1.0).contains(&self.slerp_step)
            && self.noise.is_finite()
            && self.noise >= 0.0
            && self.energy_alpha.is_finite()
            && (0.0..=MAX_ENERGY_ALPHA).contains(&self.energy_alpha);
        if ok {
            Ok(())
        } else {
            Err(PlannerError::InvalidInput(format!(
                "invalid geodesic flow config {self:?}"
            )))
        }
    }
}

/// Start of every sampled plan.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SamplerStart {
    pub done: u64,
    pub time_used: u32,
    /// Forbidden at every step.
    pub blocked: u64,
    /// Forbidden as the first step only.
    pub blocked_first: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledPath {
    /// Node indices in execution order. The last one may have run out the clock.
    pub actions: Vec<usize>,
    /// `log P_F` of the sampled path under the masked (and energy-steered) policy.
    pub log_pf: f64,
    /// The pruner found no allowed action before the plan ended.
    pub dead_end: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SampleSet {
    pub paths: Vec<SampledPath>,
    pub steps: usize,
    pub dead_ends: usize,
}

pub struct GeodesicFlowSampler<'a> {
    dag: &'a CausalDag,
    pruner: CausalPruner<'a>,
    potential: CausalPotential,
    etf: SimplexEtfFrame,
    cfg: GeodesicFlowConfig,
}

impl<'a> GeodesicFlowSampler<'a> {
    pub fn new(dag: &'a CausalDag, cfg: GeodesicFlowConfig) -> Result<Self, PlannerError> {
        cfg.validate()?;
        let etf = SimplexEtfFrame::new(dag.len(), cfg.dim).map_err(|e| {
            PlannerError::InvalidInput(format!(
                "ETF frame for {} actions in dim {}: {e}",
                dag.len(),
                cfg.dim
            ))
        })?;
        Ok(Self {
            dag,
            pruner: CausalPruner::new(dag, true),
            potential: CausalPotential::new(dag)
                .map_err(|e| PlannerError::InvalidInput(e.to_string()))?,
            etf,
            cfg,
        })
    }

    /// Sample `n` plans of at most `horizon` steps. Checks `deadline` before every
    /// plan and returns `TimeoutExceeded` once it passes.
    pub fn sample(
        &self,
        rng: &mut StdRng,
        n: usize,
        horizon: usize,
        start: &SamplerStart,
        deadline: Option<Instant>,
    ) -> Result<SampleSet, PlannerError> {
        let t0 = Instant::now();
        let z0 = self.start_latent(start)?;
        let mut set = SampleSet {
            paths: Vec::with_capacity(n),
            ..SampleSet::default()
        };
        let mut logits = vec![0.0_f32; self.dag.len()];
        for _ in 0..n {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(PlannerError::TimeoutExceeded(
                    t0.elapsed().as_secs_f64() * 1000.0,
                ));
            }
            let path = self.sample_one(rng, &z0, horizon, start, &mut logits, &mut set.steps)?;
            set.dead_ends += usize::from(path.dead_end);
            set.paths.push(path);
        }
        Ok(set)
    }

    fn sample_one(
        &self,
        rng: &mut StdRng,
        z0: &[f64],
        horizon: usize,
        start: &SamplerStart,
        logits: &mut [f32],
        steps: &mut usize,
    ) -> Result<SampledPath, PlannerError> {
        let mut z = z0.to_vec();
        let mut z32 = vec![0.0_f32; z.len()];
        let mut done = start.done;
        let mut clock = start.time_used;
        let budget = self.dag.budget();
        let mut path = SampledPath {
            actions: Vec::with_capacity(horizon),
            log_pf: 0.0,
            dead_end: false,
        };
        for step in 0..horizon {
            let blocked = start.blocked | if step == 0 { start.blocked_first } else { 0 };
            let allowed = match self.pruner.prune(done, blocked) {
                Ok(r) => r,
                Err(EmptyActionSet { .. }) => {
                    path.dead_end = true;
                    break;
                }
            };
            for (dst, src) in z32.iter_mut().zip(&z) {
                *dst = *src as f32;
            }
            self.etf.project_logits(&z32, logits);
            let mut bias = [0.0_f64; MAX_CAUSAL_ATOMS];
            let bias = if self.cfg.energy_alpha > 0.0 {
                self.potential
                    .bias_into(done, allowed.allowed, self.cfg.energy_alpha, &mut bias)
                    .map_err(|e| PlannerError::DivergentState(e.to_string()))?;
                Some(&bias)
            } else {
                None
            };
            let a = self.draw(rng, allowed.allowed, logits, bias)?;
            path.log_pf += a.1;
            let a = a.0;
            let node = self.dag.node(a);
            *steps += 1;
            path.actions.push(a);
            // Nominal transition: running out the clock ends the plan. The action
            // stays in the path, as in the Python sampler, so the gate reports
            // over_budget / no_target instead of the sampler hiding it.
            if node.cost > budget.saturating_sub(clock) {
                break;
            }
            clock += node.cost;
            done |= 1 << a;
            if a == self.dag.target() {
                break;
            }
            if clock >= budget {
                break;
            }
            self.step_geodesic(rng, &mut z, a)?;
        }
        Ok(path)
    }

    /// Masked softmax draw over `logits / T` plus the energy bias when given.
    /// Returns the node and its log-probability under that steered policy.
    fn draw(
        &self,
        rng: &mut StdRng,
        allowed: u64,
        logits: &[f32],
        bias: Option<&[f64; MAX_CAUSAL_ATOMS]>,
    ) -> Result<(usize, f64), PlannerError> {
        let allowed = BitIter(allowed);
        let t = self.cfg.temperature;
        // No `+ 0.0` on the native path: alpha = 0 must stay bit-identical.
        let score = |i: usize| match bias {
            Some(b) => f64::from(logits[i]) / t + b[i],
            None => f64::from(logits[i]) / t,
        };
        let max = allowed.clone().map(score).fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = allowed.clone().map(|i| (score(i) - max).exp()).sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(PlannerError::DivergentState(format!(
                "masked forward policy has no mass (sum={total})"
            )));
        }
        let u = rng.gen::<f64>() * total;
        let mut acc = 0.0;
        let mut last = None;
        for i in allowed {
            let w = (score(i) - max).exp();
            acc += w;
            last = Some((i, w));
            if u < acc {
                return Ok((i, (w / total).ln()));
            }
        }
        // u landed on the rounding gap at the top end: take the last allowed node.
        let (i, w) = last.ok_or_else(|| {
            PlannerError::ConvergenceFailure("draw over an empty allowed set".into())
        })?;
        Ok((i, (w / total).ln()))
    }

    fn step_geodesic(&self, rng: &mut StdRng, z: &mut [f64], a: usize) -> Result<(), PlannerError> {
        let e = self.etf.candidate_vector(a).ok_or_else(|| {
            PlannerError::ConvergenceFailure(format!("no ETF vertex for node {a}"))
        })?;
        slerp_toward(z, e, self.cfg.slerp_step);
        if self.cfg.noise > 1e-7 {
            for x in z.iter_mut() {
                *x += gaussian(rng) * self.cfg.noise;
            }
        }
        normalize(z)
    }

    fn start_latent(&self, start: &SamplerStart) -> Result<Vec<f64>, PlannerError> {
        let seed = splitmix64(start.done ^ (u64::from(start.time_used) << 32).rotate_left(17));
        let mut rng = StdRng::seed_from_u64(seed);
        let mut z: Vec<f64> = (0..self.cfg.dim).map(|_| gaussian(&mut rng)).collect();
        normalize(&mut z)?;
        Ok(z)
    }
}

/// In-place spherical interpolation of unit `z` toward unit `e` by fraction `t`.
fn slerp_toward(z: &mut [f64], e: &[f32], t: f64) {
    let dot: f64 = z
        .iter()
        .zip(e)
        .map(|(a, b)| a * f64::from(*b))
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    let theta = dot.acos();
    let s = theta.sin();
    let (w0, w1) = if s < 1e-9 {
        // Parallel or antipodal: linear blend; normalize() rejects a zero result.
        (1.0 - t, t)
    } else {
        (((1.0 - t) * theta).sin() / s, (t * theta).sin() / s)
    };
    for (i, x) in z.iter_mut().enumerate() {
        let ei = e.get(i).copied().map_or(0.0, f64::from);
        *x = w0 * *x + w1 * ei;
    }
}

fn normalize(z: &mut [f64]) -> Result<(), PlannerError> {
    let norm = z.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 1e-12) {
        return Err(PlannerError::DivergentState(format!(
            "geodesic latent collapsed (norm {norm})"
        )));
    }
    z.iter_mut().for_each(|x| *x /= norm);
    Ok(())
}

/// Box-Muller standard normal (the workspace has no rand_distr).
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
