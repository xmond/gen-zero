//! Caller-tunable hyperparameters for [`crate::ProductionPipeline`]'s search engines.
//!
//! MCTS runs sequentially, so virtual loss is not exposed. Its persistent node
//! budget remains an engine-level setting; exhaustion fails closed.

use crate::error::PlannerError;
use serde::{Deserialize, Serialize};

/// Hyperparameters injected into `ProductionPipeline`'s MCTS, MPC-CEM and A*
/// engines and the Dynamic K-MoE router. The stateless Manifold GFlowNet and
/// CFR/Nash engines are selected through [`crate::DecideRequest::mode`] and do
/// not currently expose caller-tunable hyperparameters. See
/// [`crate::ProductionPipeline::new_with_config`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlannerConfig {
    /// Absolute process-local deadline; never serialized.
    #[serde(skip)]
    pub deadline: Option<std::time::Instant>,
    /// Relative decision budget, measured from entry to `decide`.
    pub budget_ms: Option<f64>,
    /// `MctsEngine::max_simulations`: PUCT rollouts per `decide` call. More
    /// simulations converge closer to the true argmax; too few leave the pick
    /// dominated by round-robin exploration order.
    pub mcts_max_simulations: usize,
    /// `MctsEngine::c_puct`: exploration weight in the PUCT score
    /// `q + c_puct * prior * sqrt(N_parent) / (1 + N_child)`. Higher values
    /// keep spreading visits across actions instead of committing to the
    /// current best; can flip which action ends with the most visits when
    /// `max_simulations` is small.
    pub mcts_c_puct: f32,
    /// Maximum MCTS transition depth, including the root action.
    pub mcts_horizon: usize,
    /// Discount applied to future MCTS rewards.
    pub mcts_discount: f32,
    /// `MpcCemEngine::num_samples`: trajectories sampled per CEM iteration.
    pub cem_num_samples: usize,
    /// `MpcCemEngine::horizon`: steps rolled forward per sampled trajectory.
    pub cem_horizon: usize,
    /// Geometric CEM reward discount in [0, 1].
    pub cem_gamma: f32,
    /// `AStarEngine::uncertainty_penalty_weight`: weight on state-dispersion
    /// penalty in `edge_cost = 1 + max(-reward, 0) + weight * displacement * 0.05`. Higher values bias
    /// away from actions that move the state further, even at higher reward.
    pub astar_uncertainty_penalty_weight: f32,
    /// `DynamicKMoERouter::entropy_threshold_low`: at or below this perceptual
    /// entropy (and gate tier `Tier0Proceed`), `decide(Auto)` routes to
    /// `K1Reflex`.
    pub router_entropy_threshold_low: f32,
    /// `DynamicKMoERouter::entropy_threshold_high`: at or below this, `Auto`
    /// routes to `K2Pipeline`; above it, `K3Committee`.
    pub router_entropy_threshold_high: f32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            deadline: None,
            budget_ms: None,
            mcts_max_simulations: 128,
            mcts_c_puct: 1.414,
            mcts_horizon: 4,
            mcts_discount: 0.99,
            cem_num_samples: 32,
            cem_horizon: 4,
            cem_gamma: 0.95,
            astar_uncertainty_penalty_weight: 0.5,
            router_entropy_threshold_low: 0.20,
            router_entropy_threshold_high: 0.70,
        }
    }
}

impl PlannerConfig {
    /// Reject a config before it ever reaches an engine. A bad value here would
    /// otherwise degrade silently: e.g. `mcts_max_simulations: 0` never runs a
    /// single simulation, and `router_entropy_threshold_low > _high` makes
    /// `classify_tier` skip `K2Pipeline` entirely.
    pub fn validate(&self) -> Result<(), PlannerError> {
        if !self.cem_gamma.is_finite() || !(0.0..=1.0).contains(&self.cem_gamma) {
            return Err(PlannerError::InvalidInput(
                "cem_gamma must be finite in [0, 1]".into(),
            ));
        }
        resolve_deadline(self.deadline, self.budget_ms, std::time::Instant::now())?;
        if !(1..=50_000).contains(&self.mcts_max_simulations) {
            return Err(PlannerError::InvalidInput(format!(
                "mcts_max_simulations must lie in [1, 50000], got {}",
                self.mcts_max_simulations
            )));
        }
        if !(self.mcts_c_puct.is_finite() && self.mcts_c_puct >= 0.0) {
            return Err(PlannerError::InvalidInput(format!(
                "mcts_c_puct must be finite and >= 0, got {}",
                self.mcts_c_puct
            )));
        }
        if !(1..=100).contains(&self.mcts_horizon)
            || !self.mcts_discount.is_finite()
            || !(0.0..=1.0).contains(&self.mcts_discount)
        {
            return Err(PlannerError::InvalidInput(
                "mcts_horizon must lie in [1, 100] and mcts_discount in [0, 1]".into(),
            ));
        }
        if !(1..=10_000).contains(&self.cem_num_samples) {
            return Err(PlannerError::InvalidInput(format!(
                "cem_num_samples must lie in [1, 10000], got {}",
                self.cem_num_samples
            )));
        }
        if !(1..=100).contains(&self.cem_horizon) {
            return Err(PlannerError::InvalidInput(format!(
                "cem_horizon must lie in [1, 100], got {}",
                self.cem_horizon
            )));
        }
        if !(self.astar_uncertainty_penalty_weight.is_finite()
            && self.astar_uncertainty_penalty_weight >= 0.0)
        {
            return Err(PlannerError::InvalidInput(format!(
                "astar_uncertainty_penalty_weight must be finite and >= 0, got {}",
                self.astar_uncertainty_penalty_weight
            )));
        }
        if !(0.0..=1.0).contains(&self.router_entropy_threshold_low) {
            return Err(PlannerError::InvalidInput(format!(
                "router_entropy_threshold_low must lie in [0, 1], got {}",
                self.router_entropy_threshold_low
            )));
        }
        if !(0.0..=1.0).contains(&self.router_entropy_threshold_high) {
            return Err(PlannerError::InvalidInput(format!(
                "router_entropy_threshold_high must lie in [0, 1], got {}",
                self.router_entropy_threshold_high
            )));
        }
        if self.router_entropy_threshold_low > self.router_entropy_threshold_high {
            return Err(PlannerError::InvalidInput(format!(
                "router_entropy_threshold_low {} must be <= router_entropy_threshold_high {}",
                self.router_entropy_threshold_low, self.router_entropy_threshold_high
            )));
        }
        Ok(())
    }
}

/// Use the earliest absolute/relative limit. Invalid budgets fail closed.
pub(crate) fn resolve_deadline(
    deadline: Option<std::time::Instant>,
    budget_ms: Option<f64>,
    start: std::time::Instant,
) -> Result<Option<std::time::Instant>, PlannerError> {
    let relative = budget_ms
        .map(|ms| {
            if !ms.is_finite() || ms < 0.0 {
                return Err(PlannerError::InvalidInput(
                    "budget_ms must be finite and nonnegative".into(),
                ));
            }
            let duration = std::time::Duration::try_from_secs_f64(ms / 1000.0)
                .map_err(|_| PlannerError::InvalidInput("budget_ms out of range".into()))?;
            start
                .checked_add(duration)
                .ok_or_else(|| PlannerError::InvalidInput("deadline out of range".into()))
        })
        .transpose()?;
    Ok(match (deadline, relative) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_the_engines_own_defaults() {
        let c = PlannerConfig::default();
        assert_eq!(c.mcts_max_simulations, 128);
        assert_eq!(c.mcts_c_puct, 1.414);
        assert_eq!(c.mcts_horizon, crate::MctsEngine::default().horizon);
        assert_eq!(c.mcts_discount, crate::MctsEngine::default().discount);
        assert_eq!(c.cem_num_samples, 32);
        assert_eq!(c.cem_horizon, 4);
        assert_eq!(c.astar_uncertainty_penalty_weight, 0.5);
        assert_eq!(c.router_entropy_threshold_low, 0.20);
        assert_eq!(c.router_entropy_threshold_high, 0.70);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn deserializes_a_partial_object_over_defaults() {
        let c: PlannerConfig = serde_json::from_str(r#"{"mcts_c_puct": 2.0}"#).unwrap();
        assert_eq!(c.mcts_c_puct, 2.0);
        assert_eq!(c.mcts_max_simulations, 128); // untouched field keeps its default
    }

    #[test]
    fn validate_rejects_degenerate_and_inverted_values() {
        let bad = [
            PlannerConfig {
                mcts_horizon: 0,
                ..Default::default()
            },
            PlannerConfig {
                mcts_horizon: 101,
                ..Default::default()
            },
            PlannerConfig {
                mcts_discount: f32::NAN,
                ..Default::default()
            },
            PlannerConfig {
                mcts_discount: -0.1,
                ..Default::default()
            },
            PlannerConfig {
                mcts_discount: 1.1,
                ..Default::default()
            },
            PlannerConfig {
                mcts_max_simulations: 0,
                ..Default::default()
            },
            PlannerConfig {
                mcts_max_simulations: 50_001,
                ..Default::default()
            },
            PlannerConfig {
                mcts_c_puct: -1.0,
                ..Default::default()
            },
            PlannerConfig {
                mcts_c_puct: f32::NAN,
                ..Default::default()
            },
            PlannerConfig {
                cem_num_samples: 0,
                ..Default::default()
            },
            PlannerConfig {
                cem_num_samples: 10_001,
                ..Default::default()
            },
            PlannerConfig {
                cem_horizon: 0,
                ..Default::default()
            },
            PlannerConfig {
                cem_horizon: 101,
                ..Default::default()
            },
            PlannerConfig {
                astar_uncertainty_penalty_weight: -0.1,
                ..Default::default()
            },
            PlannerConfig {
                router_entropy_threshold_low: 1.1,
                ..Default::default()
            },
            PlannerConfig {
                router_entropy_threshold_high: -0.1,
                ..Default::default()
            },
            PlannerConfig {
                router_entropy_threshold_low: 0.8,
                router_entropy_threshold_high: 0.2,
                ..Default::default()
            },
        ];
        for cfg in bad {
            assert!(
                matches!(cfg.validate(), Err(PlannerError::InvalidInput(_))),
                "{cfg:?}"
            );
        }
    }
}
