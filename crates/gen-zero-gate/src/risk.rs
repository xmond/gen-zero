//! Semantic safety risk -> PolicyGate tier.
//!
//! The risk probability comes from a model that reads the request text in any
//! language (the Python few-shot classifier behind the `zero` bridge). This
//! module only maps it to a tier. It never looks at the text itself: there is
//! no keyword list here.

use crate::policy::{PolicyGate, PolicyTier};
use serde::{Deserialize, Serialize};

/// A calibrated risk judgement of one request.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticRisk {
    /// P(request is destructive, irreversible, privilege-escalating or
    /// security-bypassing), in [0, 1].
    pub p_dangerous: f32,
    /// At or above: human confirmation required.
    pub escalate_threshold: f32,
    /// At or above: fail-closed hard stop.
    pub hard_stop_threshold: f32,
}

impl PolicyGate {
    /// Tier for a semantic risk judgement.
    ///
    /// `None` means the request text could not be assessed (classifier down,
    /// invalid answer, request refused). That is fail-closed: `Tier2Escalate`,
    /// never `Tier0Proceed`. Malformed thresholds are treated the same way.
    pub fn evaluate_semantic_risk(&self, risk: Option<SemanticRisk>) -> PolicyTier {
        let Some(r) = risk else {
            return PolicyTier::Tier2Escalate;
        };
        let unit = |x: f32| x.is_finite() && (0.0..=1.0).contains(&x);
        if !unit(r.p_dangerous)
            || !unit(r.escalate_threshold)
            || !unit(r.hard_stop_threshold)
            || r.escalate_threshold > r.hard_stop_threshold
        {
            return PolicyTier::Tier2Escalate;
        }
        if r.p_dangerous >= r.hard_stop_threshold {
            PolicyTier::Tier3HardStop
        } else if r.p_dangerous >= r.escalate_threshold {
            PolicyTier::Tier2Escalate
        } else {
            PolicyTier::Tier0Proceed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn risk(p: f32) -> Option<SemanticRisk> {
        Some(SemanticRisk {
            p_dangerous: p,
            escalate_threshold: 0.5,
            hard_stop_threshold: 0.8,
        })
    }

    #[test]
    fn maps_probability_bands_to_tiers() {
        let gate = PolicyGate::default();
        assert_eq!(
            gate.evaluate_semantic_risk(risk(0.1)),
            PolicyTier::Tier0Proceed
        );
        assert_eq!(
            gate.evaluate_semantic_risk(risk(0.5)),
            PolicyTier::Tier2Escalate
        );
        assert_eq!(
            gate.evaluate_semantic_risk(risk(0.79)),
            PolicyTier::Tier2Escalate
        );
        assert_eq!(
            gate.evaluate_semantic_risk(risk(0.8)),
            PolicyTier::Tier3HardStop
        );
    }

    /// Bare `rm -rf /` scores 0.469. It proceeded under the old escalate
    /// threshold 0.5307 and must escalate under the recalibrated 0.4494
    /// (values from `risk_data/report.json`; the gate itself holds no threshold).
    #[test]
    fn bare_rm_rf_root_escalates_under_recalibrated_threshold() {
        let gate = PolicyGate::default();
        let bare = |escalate| {
            Some(SemanticRisk {
                p_dangerous: 0.4694,
                escalate_threshold: escalate,
                hard_stop_threshold: 0.7620,
            })
        };
        assert_eq!(
            gate.evaluate_semantic_risk(bare(0.5307)),
            PolicyTier::Tier0Proceed
        );
        assert_eq!(
            gate.evaluate_semantic_risk(bare(0.4494)),
            PolicyTier::Tier2Escalate
        );
    }

    #[test]
    fn unassessed_or_malformed_risk_fails_closed() {
        let gate = PolicyGate::default();
        assert_eq!(gate.evaluate_semantic_risk(None), PolicyTier::Tier2Escalate);
        assert_eq!(
            gate.evaluate_semantic_risk(risk(f32::NAN)),
            PolicyTier::Tier2Escalate
        );
        let inverted = Some(SemanticRisk {
            p_dangerous: 0.0,
            escalate_threshold: 0.9,
            hard_stop_threshold: 0.1,
        });
        assert_eq!(
            gate.evaluate_semantic_risk(inverted),
            PolicyTier::Tier2Escalate
        );
    }
}
