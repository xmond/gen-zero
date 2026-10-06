//! Distributionally robust slack arbitration and the active causal probe.
//!
//! Port of `python/gen_zero/planner/triad/robust_gate.py`. The causal gate
//! ranks passing plans by the NOMINAL model. On a real system each step can
//! cost more than nominal, and the overrun can depend on the clock (fatigue:
//! every step started at or after the onset costs extra). This module ranks the
//! same gate-passing plans under a disturbance model instead:
//!
//! - [`DisturbanceModel`]: per-step extra-cost pmf, fatigue onset rule
//!   (onset `ClockPhase::onset_tick`: `ceil(fatigue_frac * budget)`, exact
//!   in `u64` for 9-decimal fractions and within machine precision otherwise,
//!   so `0.07 * 100` is 7, not the 8 a raw `f64` ceil gives;
//!   `+fatigue_extra` per late step) and an
//!   instance-level Dirichlet prior strength `kappa` (`INFINITY` = none). The
//!   model is fitted offline from metered traces by the Python
//!   `fit_disturbance_model`; the fit itself is NOT ported, Rust only reads its
//!   output through [`RobustSpec`].
//! - [`score_path_into`]: the exact finish-time distribution of a fixed plan
//!   with fatigue tracked through the convolution, into caller-owned scratch.
//! - [`RobustSlackSelector`]: ranks gate-passing plans by `P(finish <= budget)`,
//!   then expected slack, then the gate's [`arbiter_cmp`] (objective
//!   `p_success`), or by `E[finish]` (objective `expected_time`). It also tries
//!   reorders of the best plans: a [`shortest_first_order`] topological order
//!   and admissible adjacent swaps (hill climbing) from both orders.
//! - [`choose_probe_action`]: the cheapest action of the committed plan that is
//!   legal as the first step, so the probe costs no extra step.
//!
//! Deliberate deviations from the Python port:
//!
//! 1. The nominal tie-break is [`arbiter_cmp`] (net reward first), the Rust
//!    gate's order. With every node value 0 it equals Python's
//!    `(nominal_time, len, path)`.
//! 2. An observed residual outside the model's support is REFUSED
//!    ([`RobustSlackSelector::from_spec`] errors). Python logs it and leaves it
//!    out of the update; this crate has no logger, and ranking under a model the
//!    observation just contradicted would be a silent degradation.
//! 3. The probe also honours the gate's `blocked` / `blocked_first` masks: the
//!    probe becomes step 1, and the Python gate has no such masks.
//! 4. The sharded tournament supports robust ranking (Python refuses it). That
//!    is sound because the robust key, with [`arbiter_cmp`] as tie-break, is a
//!    strict total order over distinct paths: each shard keeps its top
//!    `max(top_p, REORDER_TOP)` by that key, so the pooled elites contain the
//!    global top `REORDER_TOP`, which is exactly what the flat selector scores
//!    and reorders. See `tournament.rs`.
//!
//! Fail-closed: the selector only ranks plans that already passed the gate, and
//! every reordered or probe-first plan is re-checked by the same gate; a plan the
//! gate rejects is never admitted. A model or window the scorer cannot handle
//! exactly is refused, never approximated.

use super::dag::CausalDag;
use super::gate::{arbiter_cmp, CausalGate, PathVerdict};
use crate::error::PlannerError;
use gen_zero_lod::manifold::{
    valid_fatigue_frac, ClockPhase, DisturbanceMoments, FATIGUE_FRAC_QUANTUM,
};
use serde::{Deserialize, Deserializer};
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// Plans scored exactly before reordering; reordering runs on the best few.
pub const REORDER_TOP: usize = 3;
/// Hill-climbing passes of adjacent swaps per reorder seed.
pub const REORDER_MAX_PASSES: usize = 8;
/// Largest `budget - time_used` the exact scorer accepts (one f64 per tick).
pub const MAX_ROBUST_WINDOW: u32 = 1 << 16;
/// Largest extra-cost support (`extra_pmf.len()`).
pub const MAX_EXTRA_SUPPORT: usize = 256;

fn invalid(detail: impl std::fmt::Display) -> PlannerError {
    PlannerError::InvalidInput(format!("robust gate: {detail}"))
}

// ------------------------------------------------------------------ model

/// Learned per-step disturbance. Immutable; [`Self::posterior`] returns a new one.
#[derive(Clone, Debug, PartialEq)]
pub struct DisturbanceModel {
    extra_pmf: Vec<f64>,
    fatigue_frac: Option<f64>,
    fatigue_extra: u32,
    kappa: f64,
    provenance: String,
    moments: DisturbanceMoments,
}

impl DisturbanceModel {
    /// `extra_pmf[j] = P(extra == j)` before fatigue. `fatigue_frac` in
    /// `[FATIGUE_FRAC_QUANTUM, 1]` or `None` (no fatigue rule). `kappa > 0`;
    /// `f64::INFINITY` means the fit found no instance-level variation.
    pub fn new(
        extra_pmf: Vec<f64>,
        fatigue_frac: Option<f64>,
        fatigue_extra: u32,
        kappa: f64,
        provenance: impl Into<String>,
    ) -> Result<Self, PlannerError> {
        if extra_pmf.is_empty() || extra_pmf.len() > MAX_EXTRA_SUPPORT {
            return Err(invalid(format!(
                "extra_pmf must have 1..={MAX_EXTRA_SUPPORT} entries, got {}",
                extra_pmf.len()
            )));
        }
        if extra_pmf.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(invalid(format!(
                "extra_pmf entries must be finite and >= 0, got {extra_pmf:?}"
            )));
        }
        let total: f64 = extra_pmf.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(invalid(format!(
                "extra_pmf must sum to 1, got {total} ({extra_pmf:?})"
            )));
        }
        if let Some(f) = fatigue_frac {
            if !valid_fatigue_frac(f) {
                return Err(invalid(format!(
                    "fatigue_frac must be in [{FATIGUE_FRAC_QUANTUM}, 1], got {f}"
                )));
            }
        }
        if kappa.is_nan() || kappa <= 0.0 {
            return Err(invalid(format!(
                "kappa must be > 0 (inf allowed), got {kappa}"
            )));
        }
        let mean_extra: f64 = extra_pmf
            .iter()
            .enumerate()
            .map(|(j, p)| j as f64 * p)
            .sum();
        let variance = extra_pmf
            .iter()
            .enumerate()
            .map(|(j, p)| p * (j as f64 - mean_extra).powi(2))
            .sum();
        Ok(Self {
            extra_pmf,
            fatigue_frac,
            fatigue_extra,
            kappa,
            provenance: provenance.into(),
            moments: DisturbanceMoments {
                mean: mean_extra,
                variance,
                kappa: kappa.is_finite().then_some(kappa),
            },
        })
    }

    pub fn from_spec(spec: &DisturbanceModelSpec) -> Result<Self, PlannerError> {
        Self::new(
            spec.extra_pmf.clone(),
            spec.fatigue_frac,
            spec.fatigue_extra,
            spec.kappa.unwrap_or(f64::INFINITY),
            spec.provenance.clone(),
        )
    }

    pub fn moments(&self) -> DisturbanceMoments {
        self.moments
    }

    pub fn extra_pmf(&self) -> &[f64] {
        &self.extra_pmf
    }

    pub fn fatigue_frac(&self) -> Option<f64> {
        self.fatigue_frac
    }

    pub fn fatigue_extra(&self) -> u32 {
        self.fatigue_extra
    }

    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    /// Clock at which fatigue starts, or `None` when there is no fatigue rule.
    pub fn fatigue_at(&self, budget: u32) -> Result<Option<u64>, PlannerError> {
        match self.fatigue_frac {
            Some(f) if self.fatigue_extra > 0 => ClockPhase::onset_tick(budget, f)
                .map(Some)
                .map_err(|e| invalid(format!("invalid fatigue onset tick: {e}"))),
            _ => Ok(None),
        }
    }

    /// Observed extra with the model's fatigue term removed.
    pub fn residual(
        &self,
        nominal_cost: u32,
        observed_cost: u32,
        time_before: u64,
        budget: u32,
    ) -> Result<i64, PlannerError> {
        let raw = i64::from(observed_cost) - i64::from(nominal_cost);
        let late = self
            .fatigue_at(budget)?
            .is_some_and(|onset| time_before >= onset);
        let extra = if late {
            raw - i64::from(self.fatigue_extra)
        } else {
            raw
        };
        Ok(extra)
    }

    /// Dirichlet-multinomial posterior predictive after `residuals`. Returns
    /// the model and the number of residuals outside the support `0..len`,
    /// which are left out of the update; the caller decides what that means.
    /// With `kappa = inf` the posterior equals the prior: an observation then
    /// carries no information about the other items.
    #[must_use = "out-of-support residual count must be checked"]
    pub fn posterior(&self, residuals: &[i64]) -> (Self, usize) {
        let k = self.extra_pmf.len();
        let mut counts = vec![0.0_f64; k];
        let mut bad = 0;
        for &r in residuals {
            match usize::try_from(r).ok().filter(|&j| j < k) {
                Some(j) => counts[j] += 1.0,
                None => bad += 1,
            }
        }
        let n: f64 = counts.iter().sum();
        if self.kappa.is_infinite() || n == 0.0 {
            return (self.clone(), bad);
        }
        // Form weights before summing: large finite concentrations must not
        // overflow the alpha sum and silently collapse the PMF to zeros.
        let strength = self.kappa + n;
        let pmf: Vec<f64> = self
            .extra_pmf
            .iter()
            .zip(&counts)
            .map(|(p, c)| (self.kappa / strength) * p + c / strength)
            .collect();
        let mean_extra: f64 = pmf.iter().enumerate().map(|(j, p)| j as f64 * p).sum();
        let variance = pmf
            .iter()
            .enumerate()
            .map(|(j, p)| p * (j as f64 - mean_extra).powi(2))
            .sum();
        (
            Self {
                extra_pmf: pmf,
                kappa: self.kappa + n,
                moments: DisturbanceMoments {
                    mean: mean_extra,
                    variance,
                    kappa: Some(self.kappa + n),
                },
                ..self.clone()
            },
            bad,
        )
    }
}

/// Wire form of [`DisturbanceModel`], same keys as Python `to_dict()`.
/// `fatigue_frac` must be present (null = no fatigue rule); `kappa` null or
/// absent = infinite (no instance-level variation).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DisturbanceModelSpec {
    pub extra_pmf: Vec<f64>,
    #[serde(deserialize_with = "required_nullable")]
    pub fatigue_frac: Option<f64>,
    pub fatigue_extra: u32,
    #[serde(default)]
    pub kappa: Option<f64>,
    #[serde(default)]
    pub provenance: String,
}

/// Present-but-nullable: a custom `deserialize_with` turns off serde's
/// "missing Option means None", so a missing key is an error.
fn required_nullable<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    Option::<f64>::deserialize(d)
}

#[derive(Copy, Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RobustObjective {
    /// Highest `P(finish <= budget)`, then most expected slack.
    #[default]
    PSuccess,
    /// Lowest `E[finish]` (overruns not truncated).
    ExpectedTime,
}

impl RobustObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PSuccess => "p_success",
            Self::ExpectedTime => "expected_time",
        }
    }
}

fn default_true() -> bool {
    true
}

/// One completed, metered step of the running episode. A timed-out step is
/// censored (its cost is cut at the budget) and must not be sent.
#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StepObservation {
    pub nominal_cost: u32,
    pub observed_cost: u32,
    /// Clock before the step.
    pub time_before: u32,
}

/// `causal_triad.robust` of a decide request. Python's `{model, objective,
/// reorder}` plus `probe` (commit the probe action as the decision) and
/// `observations` (metered steps so far; their residuals, computed here with
/// the model's own fatigue rule, update the posterior before ranking).
/// Unknown keys are refused; nothing defaults to nominal.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RobustSpec {
    pub model: DisturbanceModelSpec,
    #[serde(default)]
    pub objective: RobustObjective,
    #[serde(default = "default_true")]
    pub reorder: bool,
    #[serde(default)]
    pub probe: bool,
    #[serde(default)]
    pub observations: Vec<StepObservation>,
}

// ---------------------------------------------------------------- scoring

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RobustScore {
    pub p_success: f64,
    /// `E[(budget - finish) * 1{finish <= budget}]`.
    pub expected_slack: f64,
    /// `E[finish]`, overruns not truncated.
    pub expected_time: f64,
}

/// Reusable buffers of [`score_path_into`]. After the first call at a given
/// window size, scoring allocates nothing.
#[derive(Clone, Debug, Default)]
pub struct ScoreScratch {
    cur: Vec<f64>,
    next: Vec<f64>,
}

/// Exact finish-time distribution of `path` started at `time_used`, reduced to
/// a [`RobustScore`]. Fatigue applies per step from the clock before that step.
///
/// Mass is kept per tick up to the budget; mass past the budget goes to one
/// overflow bucket as `(mass, sum of mass * time)`. That is exact: the fatigue
/// onset never exceeds the budget (`fatigue_frac <= 1`), so every later step of
/// an overrun is fatigued and adds exactly `cost + E[extra] + fatigue_extra` in
/// expectation.
pub fn score_path_into(
    dag: &CausalDag,
    model: &DisturbanceModel,
    path: &[usize],
    time_used: u32,
    scratch: &mut ScoreScratch,
) -> Result<RobustScore, PlannerError> {
    let budget = dag.budget();
    let window = budget.saturating_sub(time_used);
    if window > MAX_ROBUST_WINDOW {
        return Err(invalid(format!(
            "budget - time_used = {window} exceeds the exact-scoring window {MAX_ROBUST_WINDOW}"
        )));
    }
    let w = window as usize;
    let base_t = u64::from(time_used);
    if scratch.cur.len() <= w {
        scratch.cur.resize(w + 1, 0.0);
        scratch.next.resize(w + 1, 0.0);
    }
    let pmf = model.extra_pmf.as_slice();
    let fat_at = model.fatigue_at(budget)?;
    // `fatigue_at` is None when there is no rule, so no surcharge anywhere.
    let fe = if fat_at.is_some() {
        u64::from(model.fatigue_extra)
    } else {
        0
    };
    // Occupied range [lo, hi) of `cur`; nothing outside it is ever read.
    let (mut lo, mut hi) = (0_usize, 0_usize);
    let (mut over_mass, mut over_sum) = (0.0_f64, 0.0_f64);
    if time_used <= budget {
        scratch.cur[0] = 1.0;
        hi = 1;
    } else {
        over_mass = 1.0;
        over_sum = base_t as f64;
    }
    for &a in path {
        if a >= dag.len() {
            return Err(invalid(format!(
                "path index {a} is outside the {}-node DAG",
                dag.len()
            )));
        }
        let c = u64::from(dag.node(a).cost);
        over_sum += over_mass * (c as f64 + model.moments.mean + fe as f64);
        if lo >= hi {
            continue;
        }
        let step_max = c + fe + (pmf.len() as u64 - 1);
        let nlo = (lo as u64 + c).min(w as u64 + 1) as usize;
        let nhi = (hi as u64 + step_max).min(w as u64 + 1) as usize;
        scratch.next[nlo..nhi].fill(0.0);
        for i in lo..hi {
            let m = scratch.cur[i];
            if m == 0.0 {
                continue;
            }
            // Integer clock threshold: `fat_at` is ClockPhase::onset_tick, the
            // same tick ClockPhase::new uses for `fatigued`. No angle is read.
            let late = fat_at.is_some_and(|f| base_t + i as u64 >= f);
            let shift = i as u64 + c + if late { fe } else { 0 };
            for (j, &pj) in pmf.iter().enumerate() {
                if pj == 0.0 {
                    continue;
                }
                let ni = shift + j as u64;
                let mass = m * pj;
                if ni <= w as u64 {
                    scratch.next[ni as usize] += mass;
                } else {
                    over_mass += mass;
                    over_sum += mass * (base_t + ni) as f64;
                }
            }
        }
        std::mem::swap(&mut scratch.cur, &mut scratch.next);
        (lo, hi) = (nlo, nhi);
    }
    let (mut p, mut slack, mut et) = (0.0_f64, 0.0_f64, over_sum);
    for i in lo..hi {
        let m = scratch.cur[i];
        p += m;
        slack += m * (w - i) as f64;
        et += m * (base_t + i as u64) as f64;
    }
    Ok(RobustScore {
        p_success: p,
        expected_slack: slack,
        expected_time: et,
    })
}

// -------------------------------------------------------------- reordering

/// Topological order of `path`'s items that always takes the cheapest
/// admissible item next (ties: lowest index; the first item must also be legal
/// as a first step, see [`CausalGate::legal_first`]); the target stays last. Written
/// into `out` (cleared first), so a warm buffer allocates nothing. Every item
/// of a gate-passing plan stays reachable, so the order uses the same items;
/// a plan that is not admissible from the gate's `done` mask is refused.
pub fn shortest_first_order(
    gate: &CausalGate<'_>,
    path: &[usize],
    out: &mut Vec<usize>,
) -> Result<(), PlannerError> {
    let dag = gate.dag();
    let tgt = dag.target();
    out.clear();
    let mut rest = 0_u64;
    let mut has_target = false;
    let mut seen = 0_u64;
    for &a in path {
        if a >= dag.len() || seen & (1 << a) != 0 {
            return Err(invalid(format!(
                "plan {path:?} repeats an item or names one outside the DAG"
            )));
        }
        seen |= 1 << a;
        if a == tgt {
            has_target = true;
        } else {
            rest |= 1 << a;
        }
    }
    let mut done = gate.context().done;
    // The first slot skips first-step hazards (and blocked items), so the
    // reorder is not lost to the gate when the cheapest ready item is one.
    let mut first = true;
    while rest != 0 {
        let mut best: Option<(u32, usize)> = None;
        let mut bits = rest;
        while bits != 0 {
            let a = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let ok = if first {
                gate.legal_first(a)
            } else {
                gate.admissible(done, a)
            };
            if ok {
                let key = (dag.node(a).cost, a);
                if best.is_none_or(|b| key < b) {
                    best = Some(key);
                }
            }
        }
        let Some((_, a)) = best else {
            // The earliest remaining item of an admissible order always has its
            // parents done, so this means the input was not a gate-passing plan.
            return Err(invalid(format!(
                "plan {path:?} is not admissible from done mask {:#x}",
                gate.context().done
            )));
        };
        out.push(a);
        rest &= !(1 << a);
        done |= 1 << a;
        first = false;
    }
    if has_target {
        out.push(tgt);
    }
    Ok(())
}

/// Cheapest action of `plan` that is legal as the next (first) step: causally
/// admissible from the gate's `done` mask and not blocked, neither at every
/// step nor as a first step (ties: lowest index). Taken from the plan's own
/// items, so probing adds no step. A gate-passing plan's first action always
/// qualifies, so an error means the caller broke that contract.
pub fn choose_probe_action(gate: &CausalGate<'_>, plan: &[usize]) -> Result<usize, PlannerError> {
    let dag = gate.dag();
    plan.iter()
        .copied()
        .filter(|&a| a < dag.len() && gate.legal_first(a))
        .min_by_key(|&a| (dag.node(a).cost, a))
        .ok_or_else(|| {
            invalid(format!(
                "no action of plan {plan:?} is legal as the first step; the plan did not pass the gate"
            ))
        })
}

// ---------------------------------------------------------------- selector

/// A ranked plan: the robust score next to the gate verdict it belongs to.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    pub score: RobustScore,
    pub verdict: PathVerdict,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustChoice {
    /// Gate verdict of the chosen (possibly reordered) plan.
    pub verdict: PathVerdict,
    pub score: RobustScore,
    /// The chosen order is not one of the input plans.
    pub reordered: bool,
    /// Exact distributions evaluated.
    pub n_scored: usize,
}

/// `x` snapped to a grid of `1 / scale`, kept in `f64`. An `as i64` key
/// saturated once `x * scale` passed `i64::MAX` (an expected time of about
/// 9.2e9 at scale 1e9) and made unequal scores compare equal. `+ 0.0` maps the
/// `-0.0` that `round` gives for a tiny negative to `+0.0`, because `total_cmp`
/// orders `-0.0` before `+0.0`.
fn quantize(x: f64, scale: f64) -> f64 {
    (x * scale).round() + 0.0
}

/// Ranks gate-passing plans under a [`DisturbanceModel`]. Immutable; build a
/// new one when the posterior changes.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustSlackSelector {
    model: DisturbanceModel,
    objective: RobustObjective,
    reorder: bool,
    probe: bool,
    n_observed: usize,
    prior_kappa: f64,
}

impl RobustSlackSelector {
    pub fn new(model: DisturbanceModel, objective: RobustObjective, reorder: bool) -> Self {
        let prior_kappa = model.kappa;
        Self {
            model,
            objective,
            reorder,
            probe: false,
            n_observed: 0,
            prior_kappa,
        }
    }

    /// Same selector, also committing the probe action as the decision.
    pub fn with_probe(self, probe: bool) -> Self {
        Self { probe, ..self }
    }

    /// Validate the spec, then condition the model on the residuals of
    /// `spec.observations` under the DAG's `budget`. A residual outside the
    /// model's support is refused: the observation contradicts the fitted
    /// model, so ranking under it would be a silent degradation (see the
    /// module doc, deviation 2).
    pub fn from_spec(spec: &RobustSpec, budget: u32) -> Result<Self, PlannerError> {
        if budget == 0 {
            return Err(invalid("budget must be positive"));
        }
        let prior = DisturbanceModel::from_spec(&spec.model)?;
        let residuals: Result<Vec<i64>, PlannerError> = spec
            .observations
            .iter()
            .map(|o| {
                prior.residual(
                    o.nominal_cost,
                    o.observed_cost,
                    u64::from(o.time_before),
                    budget,
                )
            })
            .collect();
        let residuals = residuals?;
        let (post, bad) = prior.posterior(&residuals);
        if bad > 0 {
            return Err(invalid(format!(
                "{bad} of {} observed residuals {residuals:?} fall outside the model support \
                 0..{}; refit the disturbance model",
                residuals.len(),
                prior.extra_pmf.len() - 1
            )));
        }
        Ok(Self {
            n_observed: residuals.len(),
            prior_kappa: prior.kappa,
            ..Self::new(post, spec.objective, spec.reorder).with_probe(spec.probe)
        })
    }

    pub fn model(&self) -> &DisturbanceModel {
        &self.model
    }

    pub fn objective(&self) -> RobustObjective {
        self.objective
    }

    pub fn reorder(&self) -> bool {
        self.reorder
    }

    pub fn probe(&self) -> bool {
        self.probe
    }

    /// Residuals the posterior was conditioned on.
    pub fn n_observed(&self) -> usize {
        self.n_observed
    }

    pub fn prior_kappa(&self) -> f64 {
        self.prior_kappa
    }

    /// Strict total order of scored plans, best first. Scores are snapped in
    /// `f64` (p to 1e-12, slack and time to 1e-9) and compared with
    /// `f64::total_cmp`, so float noise cannot break a tie the gate's order
    /// should decide, and no magnitude saturates the key. Above 2^53 / scale
    /// the grid is finer than `f64` spacing and the comparison is exact.
    pub fn cmp(
        &self,
        a: (&RobustScore, &PathVerdict),
        b: (&RobustScore, &PathVerdict),
    ) -> Ordering {
        let primary = match self.objective {
            RobustObjective::PSuccess => quantize(b.0.p_success, 1e12)
                .total_cmp(&quantize(a.0.p_success, 1e12))
                .then(
                    quantize(b.0.expected_slack, 1e9).total_cmp(&quantize(a.0.expected_slack, 1e9)),
                ),
            RobustObjective::ExpectedTime => {
                quantize(a.0.expected_time, 1e9).total_cmp(&quantize(b.0.expected_time, 1e9))
            }
        };
        primary.then_with(|| arbiter_cmp(a.1, b.1))
    }

    fn better(&self, a: &Scored, b: &Scored) -> bool {
        self.cmp((&a.score, &a.verdict), (&b.score, &b.verdict))
            .is_lt()
    }

    /// Score every distinct passing verdict and sort them best first. A
    /// rejected verdict is an error, never ranked.
    pub fn rank<'v>(
        &self,
        gate: &CausalGate<'_>,
        ok: impl IntoIterator<Item = &'v PathVerdict>,
        scratch: &mut ScoreScratch,
    ) -> Result<Vec<Scored>, PlannerError> {
        let mut seen = BTreeSet::new();
        let mut scored = Vec::new();
        for v in ok {
            if !v.ok || gate.check(&v.path) != *v {
                return Err(invalid(format!(
                    "selector got plan {:?} that the causal gate rejected ({})",
                    v.path,
                    v.reason.as_str()
                )));
            }
            if seen.insert(v.path.as_slice()) {
                let score = score_path_into(
                    gate.dag(),
                    &self.model,
                    &v.path,
                    gate.context().time_used,
                    scratch,
                )?;
                scored.push(Scored {
                    score,
                    verdict: v.clone(),
                });
            }
        }
        scored.sort_by(|a, b| self.cmp((&a.score, &a.verdict), (&b.score, &b.verdict)));
        Ok(scored)
    }

    /// Best plan under the model, trying reorders of the top
    /// [`REORDER_TOP`] when `reorder` is set. Needs at least one plan.
    pub fn select(
        &self,
        gate: &CausalGate<'_>,
        ok: &[PathVerdict],
        scratch: &mut ScoreScratch,
    ) -> Result<RobustChoice, PlannerError> {
        let scored = self.rank(gate, ok, scratch)?;
        let Some(first) = scored.first() else {
            return Err(invalid("selector needs at least one gate-passing plan"));
        };
        let mut n_scored = scored.len();
        let mut best = first.clone();
        if self.reorder {
            let mut order = Vec::with_capacity(gate.dag().len());
            for s in scored.iter().take(REORDER_TOP) {
                let mut seeds = vec![s.clone()];
                shortest_first_order(gate, &s.verdict.path, &mut order)?;
                let spt = gate.check(&order);
                if spt.ok && spt.path != s.verdict.path {
                    let score = self.score(gate, &spt.path, scratch)?;
                    n_scored += 1;
                    seeds.push(Scored {
                        score,
                        verdict: spt,
                    });
                }
                for seed in seeds {
                    let (cand, n_eval) = self.hill_climb(gate, seed, scratch)?;
                    n_scored += n_eval;
                    if self.better(&cand, &best) {
                        best = cand;
                    }
                }
            }
        }
        let reordered = !scored.iter().any(|s| s.verdict.path == best.verdict.path);
        Ok(RobustChoice {
            verdict: best.verdict,
            score: best.score,
            reordered,
            n_scored,
        })
    }

    pub fn score(
        &self,
        gate: &CausalGate<'_>,
        path: &[usize],
        scratch: &mut ScoreScratch,
    ) -> Result<RobustScore, PlannerError> {
        score_path_into(
            gate.dag(),
            &self.model,
            path,
            gate.context().time_used,
            scratch,
        )
    }

    /// Adjacent swaps that stay gate-passing, accepted while they improve the
    /// key. The target stays last: the gate cuts a plan right after it.
    fn hill_climb(
        &self,
        gate: &CausalGate<'_>,
        start: Scored,
        scratch: &mut ScoreScratch,
    ) -> Result<(Scored, usize), PlannerError> {
        let mut cur = start;
        let mut n_eval = 0;
        let mut trial = Vec::with_capacity(cur.verdict.path.len());
        for _ in 0..REORDER_MAX_PASSES {
            let mut improved = false;
            for i in 0..cur.verdict.path.len().saturating_sub(2) {
                trial.clear();
                trial.extend_from_slice(&cur.verdict.path);
                trial.swap(i, i + 1);
                let tv = gate.check(&trial);
                if !tv.ok {
                    continue;
                }
                let score = self.score(gate, &tv.path, scratch)?;
                n_eval += 1;
                let cand = Scored { score, verdict: tv };
                if self.better(&cand, &cur) {
                    cur = cand;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        Ok((cur, n_eval))
    }

    /// Move the probe action to the front of `chosen` and re-check the result
    /// with the same gate. Returns the probe-first verdict, its score and the
    /// probe's node index. A probe-first plan the gate rejects is an error.
    pub fn apply_probe(
        &self,
        gate: &CausalGate<'_>,
        chosen: &PathVerdict,
        scratch: &mut ScoreScratch,
    ) -> Result<(Scored, usize), PlannerError> {
        let probe = choose_probe_action(gate, &chosen.path)?;
        let mut path = Vec::with_capacity(chosen.path.len());
        path.push(probe);
        path.extend(chosen.path.iter().copied().filter(|&a| a != probe));
        let v = gate.check(&path);
        if !v.ok || v.path != path {
            return Err(PlannerError::ConvergenceFailure(format!(
                "probe-first plan {path:?} fails the causal gate ({}); probe {probe} not committed",
                v.reason.as_str()
            )));
        }
        let score = self.score(gate, &v.path, scratch)?;
        Ok((Scored { score, verdict: v }, probe))
    }
}
