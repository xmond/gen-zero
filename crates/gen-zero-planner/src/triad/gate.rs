//! Deterministic causal gate over complete candidate plans.
//!
//! Port of the gen-zero-research repository's `python/gen_zero/planner/triad/gate.py`. A path passes only when,
//! under the nominal model of the DAG:
//!
//! 1. no step uses a blocked action (policy hard stop at every step, or a
//!    first-step hazard on step 1),
//! 2. every step is causally admissible when it is taken,
//! 3. the target is completed (the path is cut right after it), and
//! 4. the nominal completion time stays within the budget quota.
//!
//! The gate never repairs a path and never falls back to an unchecked one.
//! Arithmetic that cannot be represented is a rejection, never a clamp: a clock
//! that overflows `u32` fails with `budget_overflow`, and a reward sum that
//! leaves the finite range fails with `reward_non_finite`, so neither a
//! saturated clock nor a NaN/inf ranking key can reach the arbiter.
//!
//! Ranking (deliberate deviation from the Python port, which ranks by nominal
//! time alone and ignores `value`): passing verdicts are ordered by
//! `net_reward` descending, then nominal time, then path length, then the
//! lexicographically smaller index path. `net_reward = sum(value) - cost`,
//! one unit of value per unit of nominal time spent by the path. With every
//! value at 0 (the spec default) this is exactly the Python order. The key is a
//! strict total order, so an argmax over any partition's per-block winners
//! equals the argmax over the whole set; the tournament relies on that.

use super::dag::CausalDag;
use super::robust_gate::{RobustSlackSelector, ScoreScratch};
use crate::error::PlannerError;
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GateReason {
    Ok,
    /// A step used a policy-blocked action.
    Blocked,
    /// A step was not causally admissible (precondition unmet, repeat, unknown index).
    Violation,
    /// The path ended without completing the target.
    NoTarget,
    /// The target completed after the budget ran out.
    OverBudget,
    /// The nominal clock passed `u32::MAX`; the plan is refused, never clamped.
    TimeOverflow,
    /// The accumulated value or the net reward is not finite.
    RewardNonFinite,
}

impl GateReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Blocked => "blocked",
            Self::Violation => "violation",
            Self::NoTarget => "no_target",
            Self::OverBudget => "over_budget",
            Self::TimeOverflow => "time_overflow",
            Self::RewardNonFinite => "reward_non_finite",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathVerdict {
    pub ok: bool,
    pub reason: GateReason,
    /// Cut right after the target when it is reached; up to the bad step otherwise.
    pub path: Vec<usize>,
    /// Absolute nominal clock (includes the starting `time_used`).
    pub nominal_time: u32,
    /// `sum(value) - cost` of `path`. Only meaningful when `ok`.
    pub net_reward: f64,
    pub first_bad_step: Option<usize>,
}

/// Strict total order of passing verdicts, best first.
pub fn arbiter_cmp(a: &PathVerdict, b: &PathVerdict) -> Ordering {
    b.net_reward
        .total_cmp(&a.net_reward)
        .then(a.nominal_time.cmp(&b.nominal_time))
        .then(a.path.len().cmp(&b.path.len()))
        .then_with(|| a.path.cmp(&b.path))
}

/// Count of verdicts per reason.
pub fn reason_histogram(verdicts: &[PathVerdict]) -> BTreeMap<&'static str, usize> {
    let mut h = BTreeMap::new();
    for v in verdicts {
        *h.entry(v.reason.as_str()).or_insert(0) += 1;
    }
    h
}

/// Starting point of every plan the gate judges.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GateContext {
    /// Already-completed nodes.
    pub done: u64,
    pub time_used: u32,
    /// Nodes forbidden at every step.
    pub blocked: u64,
    /// Nodes additionally forbidden as the first step only.
    pub blocked_first: u64,
}

#[derive(Clone, Debug)]
pub struct CausalGate<'a> {
    dag: &'a CausalDag,
    ctx: GateContext,
}

impl<'a> CausalGate<'a> {
    pub fn new(dag: &'a CausalDag, ctx: GateContext) -> Self {
        Self { dag, ctx }
    }

    pub fn dag(&self) -> &'a CausalDag {
        self.dag
    }

    pub fn context(&self) -> GateContext {
        self.ctx
    }

    /// `v` is not done and its AND/OR precondition holds under `done`.
    pub fn admissible(&self, done: u64, v: usize) -> bool {
        if done & (1 << v) != 0 {
            return false;
        }
        let pm = self.dag.parent_mask(v);
        if pm == 0 {
            true
        } else if self.dag.node(v).is_or {
            done & pm != 0
        } else {
            done & pm == pm
        }
    }

    /// `v` may be taken as the first step: admissible from the context's
    /// `done` mask and blocked neither at every step nor as a first step.
    pub fn legal_first(&self, v: usize) -> bool {
        let blocked = self.ctx.blocked | self.ctx.blocked_first;
        v < self.dag.len() && blocked & (1 << v) == 0 && self.admissible(self.ctx.done, v)
    }

    pub fn check(&self, path: &[usize]) -> PathVerdict {
        let mut t = self.ctx.time_used;
        let mut done = self.ctx.done;
        let mut value = 0.0_f64;
        let target = self.dag.target();
        let bad = |reason, i: usize, t| PathVerdict {
            ok: false,
            reason,
            path: path[..=i].to_vec(),
            nominal_time: t,
            net_reward: f64::NEG_INFINITY,
            first_bad_step: Some(i),
        };
        for (i, &a) in path.iter().enumerate() {
            if a >= self.dag.len() {
                return bad(GateReason::Violation, i, t);
            }
            let blocked = self.ctx.blocked | if i == 0 { self.ctx.blocked_first } else { 0 };
            if blocked & (1 << a) != 0 {
                return bad(GateReason::Blocked, i, t);
            }
            if !self.admissible(done, a) {
                return bad(GateReason::Violation, i, t);
            }
            let node = self.dag.node(a);
            let Some(next) = t.checked_add(node.cost) else {
                return bad(GateReason::TimeOverflow, i, t);
            };
            t = next;
            value += node.value;
            if !value.is_finite() {
                return bad(GateReason::RewardNonFinite, i, t);
            }
            done |= 1 << a;
            if a == target {
                if t > self.dag.budget() {
                    return bad(GateReason::OverBudget, i, t);
                }
                let net_reward = value - f64::from(t - self.ctx.time_used);
                if !net_reward.is_finite() {
                    return bad(GateReason::RewardNonFinite, i, t);
                }
                return PathVerdict {
                    ok: true,
                    reason: GateReason::Ok,
                    path: path[..=i].to_vec(),
                    nominal_time: t,
                    net_reward,
                    first_bad_step: None,
                };
            }
        }
        PathVerdict {
            ok: false,
            reason: GateReason::NoTarget,
            path: path.to_vec(),
            nominal_time: t,
            net_reward: f64::NEG_INFINITY,
            first_bad_step: Some(path.len()),
        }
    }

    /// The best `k` distinct passing verdicts, best first: by [`arbiter_cmp`],
    /// or by the robust key when `robust` is given. Rejected verdicts are
    /// skipped here (only passing ones are ever ranked).
    pub fn top_passing(
        &self,
        verdicts: &[PathVerdict],
        k: usize,
        robust: Option<&RobustSlackSelector>,
    ) -> Result<Vec<PathVerdict>, PlannerError> {
        let passing = verdicts.iter().filter(|v| v.ok);
        if let Some(sel) = robust {
            let ranked = sel.rank(self, passing, &mut ScoreScratch::default())?;
            return Ok(ranked.into_iter().take(k).map(|s| s.verdict).collect());
        }
        let mut passing: Vec<&PathVerdict> = passing.collect();
        passing.sort_by(|a, b| arbiter_cmp(a, b));
        passing.dedup_by(|a, b| a.path == b.path);
        Ok(passing.into_iter().take(k).cloned().collect())
    }

    /// Best passing verdict plus every verdict, in input order. Ranked by
    /// [`arbiter_cmp`], or by the robust key when `robust` is given. `None` for
    /// the best when nothing passed; the caller must treat that as failure.
    pub fn select(
        &self,
        paths: &[Vec<usize>],
        robust: Option<&RobustSlackSelector>,
    ) -> Result<(Option<PathVerdict>, Vec<PathVerdict>), PlannerError> {
        let verdicts: Vec<PathVerdict> = paths.iter().map(|p| self.check(p)).collect();
        let best = self.top_passing(&verdicts, 1, robust)?.into_iter().next();
        Ok((best, verdicts))
    }
}
