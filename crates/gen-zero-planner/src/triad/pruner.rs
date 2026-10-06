//! On-trajectory causal action mask.
//!
//! Port of the gen-zero-research repository's `python/gen_zero/planner/triad/pruner.py`. Three filters, counted
//! separately:
//!
//! 1. Precondition (causal admissibility): the action is not done and its
//!    parents satisfy its AND/OR rule.
//! 2. Policy context: the action is not in the `blocked` mask (PolicyGate hard
//!    stops, or first-step world-model hazards). The Python port has no policy
//!    layer; this tier is what binds the triad to the production gate.
//! 3. Goal cone: the action is the target or a target ancestor. Anything else
//!    can never enable the target, so removing it cannot lower the best
//!    reachable plan; it only burns time.
//!
//! Fail-closed: an empty surviving set returns [`EmptyActionSet`]. The pruner
//! never falls back to "all actions".

use super::dag::CausalDag;

/// No action survives the mask.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EmptyActionSet {
    pub done: u64,
    pub n_admissible: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PruneResult {
    /// Surviving node indices as a bitmask.
    pub allowed: u64,
    pub n_all: u32,
    /// After tier 1.
    pub n_admissible: u32,
    /// After tiers 1 and 2.
    pub n_unblocked: u32,
    /// After all tiers.
    pub n_allowed: u32,
}

#[derive(Clone, Debug)]
pub struct CausalPruner<'a> {
    dag: &'a CausalDag,
    cone: u64,
    use_goal_cone: bool,
}

impl<'a> CausalPruner<'a> {
    pub fn new(dag: &'a CausalDag, use_goal_cone: bool) -> Self {
        Self {
            dag,
            cone: dag.goal_cone(),
            use_goal_cone,
        }
    }

    /// Whether node `v` may run given the done set (tier 1 only).
    pub fn is_admissible(&self, done: u64, v: usize) -> bool {
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

    /// Tier-1 mask.
    pub fn admissible_mask(&self, done: u64) -> u64 {
        (0..self.dag.len())
            .filter(|&v| self.is_admissible(done, v))
            .fold(0, |m, v| m | (1 << v))
    }

    pub fn prune(&self, done: u64, blocked: u64) -> Result<PruneResult, EmptyActionSet> {
        let admissible = self.admissible_mask(done);
        let unblocked = admissible & !blocked;
        let allowed = if self.use_goal_cone {
            unblocked & self.cone
        } else {
            unblocked
        };
        if allowed == 0 {
            return Err(EmptyActionSet {
                done,
                n_admissible: admissible.count_ones(),
            });
        }
        Ok(PruneResult {
            allowed,
            n_all: self.dag.len() as u32,
            n_admissible: admissible.count_ones(),
            n_unblocked: unblocked.count_ones(),
            n_allowed: allowed.count_ones(),
        })
    }
}
