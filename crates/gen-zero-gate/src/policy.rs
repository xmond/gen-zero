//! gen-zero-gate Four-tier PolicyGate state machine and arbiters.

use crate::constraint::{LinearConstraint, RuleId};
use crate::error::GateError;
use crate::sheaf_gate::{AcceptedState, Budget, SheafProblem};
use gen_zero_core::{ActionId, GraphFactProvider, NormalizedEntropy};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Four-tier PolicyGate Risk Classification.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PolicyTier {
    /// Tier 0: Low-risk, high-confidence passthrough (P99 <= 80µs)
    Tier0Proceed = 0,
    /// Tier 1: High-impact irreversible operation requiring credentials or second confirmation
    Tier1Confirm = 1,
    /// Tier 2: Boundary uncertainty or epistemic gap triggering System 2 lookahead or human escalation
    Tier2Escalate = 2,
    /// Tier 3: Hard constraint violation or formal infeasibility triggering Fail-Closed fallback
    Tier3HardStop = 3,
}

/// Verdict emitted by the PolicyGate for an evaluated action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateVerdict {
    pub tier: PolicyTier,
    pub action: ActionId,
    pub violated_rules: SmallVec<[RuleId; 4]>,
    pub reason: String,
    /// Terminal state verified against every registered heat requirement.
    pub certified_state: Option<Vec<f64>>,
}

/// Four-tier Adaptive PolicyGate Engine.
#[derive(Debug, Clone)]
pub struct PolicyGate {
    constraints: Vec<LinearConstraint>,
    entropy_escalation_threshold: f32,
    confirm_actions: Vec<ActionId>,
    heat_requirements: Vec<(ActionId, SheafProblem, Budget)>,
}

impl Default for PolicyGate {
    fn default() -> Self {
        Self {
            constraints: Vec::new(),
            entropy_escalation_threshold: 0.65,
            confirm_actions: Vec::new(),
            heat_requirements: Vec::new(),
        }
    }
}

/// Special RuleId for Datalog cognitive graph revocations.
pub const REVOCATION_RULE_ID: RuleId = RuleId(u32::MAX);

/// Dummy provider for evaluation when no external fact provider is present.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullFactProvider;

impl GraphFactProvider for NullFactProvider {
    type DepIter<'a> = std::iter::Empty<(u64, u64)>;

    fn active_validated_dependencies<'a>(&'a self) -> Self::DepIter<'a> {
        std::iter::empty()
    }

    fn is_revoked(&self, _entity_id: u64) -> bool {
        false
    }

    fn has_privilege(&self, _agent_id: u64, _privilege: u32) -> bool {
        false
    }
}

impl PolicyGate {
    pub fn new(entropy_threshold: f32) -> Self {
        Self {
            constraints: Vec::new(),
            entropy_escalation_threshold: entropy_threshold,
            confirm_actions: Vec::new(),
            heat_requirements: Vec::new(),
        }
    }

    /// Add a formal linear constraint into the hard gate.
    pub fn add_constraint(&mut self, constraint: LinearConstraint) {
        self.constraints.push(constraint);
    }

    /// Register an irreversible high-impact action requiring secondary human confirmation.
    pub fn register_confirm_action(&mut self, action: ActionId) {
        self.confirm_actions.push(action);
    }

    /// Register a mandatory terminal-state requirement (no candidate or repair is stored).
    /// Multiple requirements for an action must all hold for the same supplied state.
    pub fn require_heat_certificate(
        &mut self,
        action: ActionId,
        problem: SheafProblem,
        budget: Budget,
    ) {
        self.heat_requirements.push((action, problem, budget));
    }

    /// Basic evaluation without graph provider context.
    #[inline]
    pub fn evaluate_basic(
        &self,
        action: ActionId,
        entropy: NormalizedEntropy,
    ) -> Result<GateVerdict, GateError> {
        self.evaluate::<NullFactProvider>(action, entropy, None, None)
    }

    /// Evaluate an action and an optional pre-repaired terminal certificate read-only.
    /// Registered heat requirements make the certificate mandatory; all requirements
    /// must validate the same terminal state. No relaxation occurs here.
    pub fn evaluate_with_context<F: GraphFactProvider>(
        &self,
        action: ActionId,
        active_context: &[ActionId],
        certificate: Option<&AcceptedState>,
        entropy: NormalizedEntropy,
        fact_provider: Option<&F>,
        agent_id: Option<u64>,
    ) -> Result<GateVerdict, GateError> {
        // Public entropy wrappers and constructor arguments are untrusted inputs.
        if !entropy.0.is_finite()
            || !(0.0..=1.0).contains(&entropy.0)
            || !self.entropy_escalation_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.entropy_escalation_threshold)
        {
            return Ok(GateVerdict {
                tier: PolicyTier::Tier3HardStop,
                action,
                certified_state: None,
                violated_rules: SmallVec::new(),
                reason: "Entropy and escalation threshold must be finite and in [0, 1]".into(),
            });
        }

        // Step 1: Check 0-1 ILP Hard Linear Constraints
        let mut violations = SmallVec::new();
        for rule in &self.constraints {
            // A candidate with a negative coefficient must not mask an already
            // invalid concurrent context. Empty context is not an executing bundle.
            if (!active_context.is_empty() && !rule.is_satisfied_by_active_context(active_context))
                || !rule.is_satisfied_with_context(action, active_context)
            {
                violations.push(rule.rule_id);
            }
        }

        if !violations.is_empty() {
            return Ok(GateVerdict {
                tier: PolicyTier::Tier3HardStop,
                action,
                certified_state: None,
                violated_rules: violations,
                reason: "Action or active context violates linear constraints or exceeds arithmetic bounds".into(),
            });
        }

        // Step 2: Check Datalog Fact Provider revocations
        if let Some(facts) = fact_provider {
            if facts.is_revoked(action.0 as u64)
                || agent_id.is_some_and(|aid| facts.is_revoked(aid))
            {
                violations.push(REVOCATION_RULE_ID);
                return Ok(GateVerdict {
                    tier: PolicyTier::Tier3HardStop,
                    action,
                    certified_state: None,
                    violated_rules: violations,
                    reason: "Action or Agent ID has active revocation in cognitive graph".into(),
                });
            }
        }

        // Read-only verification: repair must have completed before policy evaluation.
        let mut certified_state = None;
        for (_, problem, budget) in self
            .heat_requirements
            .iter()
            .filter(|(id, _, _)| *id == action)
        {
            let rejection = match certificate {
                None => Some("missing terminal heat-flow certificate".to_string()),
                Some(accepted) => problem
                    .verify_terminal(accepted, budget)
                    .err()
                    .map(|e| e.to_string()),
            };
            if let Some(rejection) = rejection {
                return Ok(GateVerdict {
                    tier: PolicyTier::Tier3HardStop,
                    action,
                    violated_rules: SmallVec::new(),
                    certified_state: None,
                    reason: format!("heat-flow certification rejected: {rejection}"),
                });
            }
            certified_state = certificate.map(|accepted| accepted.state.s.clone());
        }
        // Without a registered problem no certificate can be authenticated.
        if certificate.is_some() && certified_state.is_none() {
            return Ok(GateVerdict {
                tier: PolicyTier::Tier3HardStop,
                action,
                violated_rules: SmallVec::new(),
                certified_state: None,
                reason: "heat-flow certificate supplied without a registered requirement".into(),
            });
        }

        // Step 3: Check Tier 1 Irreversible Operations
        if self.confirm_actions.contains(&action) {
            return Ok(GateVerdict {
                tier: PolicyTier::Tier1Confirm,
                action,
                violated_rules: SmallVec::new(),
                certified_state,
                reason:
                    "Action is classified as high-impact irreversible; second confirmation required"
                        .into(),
            });
        }

        // Step 4: Check Tier 2 Epistemic Uncertainty Escalation
        if entropy.0 > self.entropy_escalation_threshold {
            return Ok(GateVerdict {
                tier: PolicyTier::Tier2Escalate,
                action,
                violated_rules: SmallVec::new(),
                certified_state,
                reason:
                    "Perceptual entropy exceeds escalation threshold; System 2 lookahead triggered"
                        .into(),
            });
        }

        // Step 5: Tier 0 Proceed
        Ok(GateVerdict {
            tier: PolicyTier::Tier0Proceed,
            action,
            violated_rules: SmallVec::new(),
            certified_state,
            reason: "Action satisfies formal verification invariants with high confidence".into(),
        })
    }

    /// Primary evaluation method cascading through 4 tiers:
    /// 1. 0-1 ILP Hard Constraints -> Tier 3 HardStop
    /// 2. Cognitive Graph Revocations -> Tier 3 HardStop
    /// 3. Irreversible Action List -> Tier 1 Confirm
    /// 4. High Uncertainty (H > threshold) -> Tier 2 Escalate
    /// 5. Otherwise Tier 0 Proceed.
    #[inline]
    pub fn evaluate<F: GraphFactProvider>(
        &self,
        action: ActionId,
        entropy: NormalizedEntropy,
        fact_provider: Option<&F>,
        agent_id: Option<u64>,
    ) -> Result<GateVerdict, GateError> {
        self.evaluate_with_context(action, &[], None, entropy, fact_provider, agent_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    use crate::sheaf_gate::{CandidateState, GeometryGate, LaplacianHeatFlowGate};

    fn heat_fixture() -> (PolicyGate, AcceptedState) {
        let problem = SheafProblem::new(
            DMatrix::from_row_slice(1, 2, &[1.0, -1.0]),
            DVector::from_row_slice(&[1.0]),
            DVector::from_row_slice(&[0.0]),
            vec![],
            vec![],
            7,
        )
        .unwrap();
        let initial = CandidateState {
            s: vec![1.0, 0.0],
            epoch: 7,
        };
        let accepted = LaplacianHeatFlowGate
            .certify(&problem, initial.clone(), &Budget::default())
            .unwrap();
        assert!(accepted.steps_taken > 0);
        assert_ne!(accepted.state, initial);
        let mut gate = PolicyGate::default();
        gate.require_heat_certificate(ActionId(42), problem, Budget::default());
        (gate, accepted)
    }

    fn with_certificate(gate: &PolicyGate, accepted: &AcceptedState, entropy: f32) -> GateVerdict {
        gate.evaluate_with_context::<NullFactProvider>(
            ActionId(42),
            &[],
            Some(accepted),
            NormalizedEntropy(entropy),
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn repaired_terminal_state_reaches_every_lower_tier_verdict() {
        let (mut gate, accepted) = heat_fixture();
        for (entropy, tier) in [
            (0.0, PolicyTier::Tier0Proceed),
            (0.9, PolicyTier::Tier2Escalate),
        ] {
            let verdict = with_certificate(&gate, &accepted, entropy);
            assert_eq!(verdict.tier, tier);
            assert_eq!(verdict.certified_state.as_ref(), Some(&accepted.state.s));
        }
        gate.register_confirm_action(ActionId(42));
        let verdict = with_certificate(&gate, &accepted, 0.0);
        assert_eq!(verdict.tier, PolicyTier::Tier1Confirm);
        assert_eq!(verdict.certified_state, Some(accepted.state.s));
    }

    #[test]
    fn residual_violation_is_rejected_without_repair_even_with_forged_diagnostics() {
        let (gate, mut accepted) = heat_fixture();
        accepted.state.s = vec![1.0, 0.0]; // Repairable, but not compliant now.
        accepted.residual = 0.0;
        let verdict = with_certificate(&gate, &accepted, 0.0);
        assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
        assert!(verdict.reason.contains("residual"));
        assert_eq!(verdict.certified_state, None);
        assert_eq!(accepted.state.s, vec![1.0, 0.0]);
    }

    #[test]
    fn certificate_from_looser_tolerance_is_rejected() {
        let (mut gate, accepted) = heat_fixture();
        assert!(accepted.residual > 0.0);
        gate.heat_requirements[0].2.residual_tol = accepted.residual / 2.0;
        let verdict = with_certificate(&gate, &accepted, 0.0);
        assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
        assert!(verdict.reason.contains("residual"));
        assert_eq!(verdict.certified_state, None);
    }

    #[test]
    fn malformed_certificates_fail_closed() {
        let (gate, accepted) = heat_fixture();
        for case in 0..5 {
            let mut bad = accepted.clone();
            match case {
                0 => bad.state.epoch += 1,
                1 => bad.state.s.clear(),
                2 => bad.state.s[0] = f64::NAN,
                3 => bad.residual = f64::NAN,
                _ => bad.energy += 1.0,
            }
            let verdict = with_certificate(&gate, &bad, 0.0);
            assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
            assert_eq!(verdict.certified_state, None);
        }
    }

    #[test]
    fn empty_registration_differs_from_missing_certificate() {
        let (gate, accepted) = heat_fixture();
        let empty = PolicyGate::default();
        assert_eq!(
            empty
                .evaluate_basic(ActionId(42), NormalizedEntropy::ZERO)
                .unwrap()
                .tier,
            PolicyTier::Tier0Proceed
        );
        assert_eq!(
            gate.evaluate_basic(ActionId(42), NormalizedEntropy::ZERO)
                .unwrap()
                .tier,
            PolicyTier::Tier3HardStop
        );
        assert_eq!(
            with_certificate(&empty, &accepted, 0.0).tier,
            PolicyTier::Tier3HardStop
        );
    }

    #[test]
    fn every_registered_requirement_checks_the_same_terminal_state() {
        let (mut gate, accepted) = heat_fixture();
        let other = SheafProblem::new(
            DMatrix::from_row_slice(1, 2, &[1.0, -1.0]),
            DVector::from_row_slice(&[1.0]),
            DVector::from_row_slice(&[1.0]),
            vec![],
            vec![],
            7,
        )
        .unwrap();
        gate.require_heat_certificate(ActionId(42), other, Budget::default());
        let verdict = with_certificate(&gate, &accepted, 0.0);
        assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
        assert_eq!(verdict.certified_state, None);
    }

    #[test]
    fn invalid_entropy_and_thresholds_fail_closed() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            let mut gate = PolicyGate::default();
            gate.register_confirm_action(ActionId(1));
            for action in [ActionId(1), ActionId(2)] {
                assert_eq!(
                    gate.evaluate_basic(action, NormalizedEntropy(invalid))
                        .unwrap()
                        .tier,
                    PolicyTier::Tier3HardStop
                );
                assert_eq!(
                    PolicyGate::new(invalid)
                        .evaluate_basic(action, NormalizedEntropy::ZERO)
                        .unwrap()
                        .tier,
                    PolicyTier::Tier3HardStop
                );
            }
        }
        for valid in [0.0, 1.0] {
            assert_eq!(
                PolicyGate::new(valid)
                    .evaluate_basic(ActionId(1), NormalizedEntropy(valid))
                    .unwrap()
                    .tier,
                PolicyTier::Tier0Proceed
            );
        }
    }

    #[test]
    fn configured_heat_certificate_rejects_high_risk_action() {
        let problem = SheafProblem::new(
            DMatrix::from_row_slice(1, 2, &[1.0, -1.0]),
            DVector::from_row_slice(&[1.0]),
            DVector::from_row_slice(&[0.0]),
            vec![],
            vec![],
            7,
        )
        .unwrap();
        let mut gate = PolicyGate::default();
        let action = ActionId(42);
        gate.register_confirm_action(action);
        gate.require_heat_certificate(action, problem, Budget::default());
        let verdict = gate
            .evaluate_basic(action, NormalizedEntropy::ZERO)
            .unwrap();
        assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
        assert!(verdict.reason.contains("missing terminal"));
    }

    #[test]
    fn test_policy_gate_cascade() {
        let mut gate = PolicyGate::new(0.60);
        // Add hard constraint: prohibit Action(99)
        gate.add_constraint(LinearConstraint::prohibit(
            RuleId(1),
            "ForbiddenRoot",
            ActionId(99),
        ));
        // Add confirm action: Action(50)
        gate.register_confirm_action(ActionId(50));

        // Test Tier 3: HardStop
        let verdict_hard = gate
            .evaluate_basic(ActionId(99), NormalizedEntropy(0.1))
            .unwrap();
        assert_eq!(verdict_hard.tier, PolicyTier::Tier3HardStop);
        assert_eq!(verdict_hard.violated_rules.len(), 1);

        // Test Tier 1: Confirm
        let verdict_confirm = gate
            .evaluate_basic(ActionId(50), NormalizedEntropy(0.1))
            .unwrap();
        assert_eq!(verdict_confirm.tier, PolicyTier::Tier1Confirm);

        // Test Tier 2: Escalate (entropy = 0.85 > 0.60)
        let verdict_escalate = gate
            .evaluate_basic(ActionId(10), NormalizedEntropy(0.85))
            .unwrap();
        assert_eq!(verdict_escalate.tier, PolicyTier::Tier2Escalate);

        // Test Tier 0: Proceed (entropy = 0.20 < 0.60)
        let verdict_proceed = gate
            .evaluate_basic(ActionId(10), NormalizedEntropy(0.20))
            .unwrap();
        assert_eq!(verdict_proceed.tier, PolicyTier::Tier0Proceed);
    }
}
