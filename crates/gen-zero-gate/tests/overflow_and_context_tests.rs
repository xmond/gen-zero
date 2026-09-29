use gen_zero_core::{ActionId, NormalizedEntropy};
use gen_zero_gate::policy::NullFactProvider;
use gen_zero_gate::{GateVerdict, LinearConstraint, PolicyGate, PolicyTier, RuleId};

fn constraint(terms: &[(u32, i32)], rhs: i32) -> LinearConstraint {
    LinearConstraint {
        rule_id: RuleId(7),
        rule_name: "overflow regression",
        terms: terms.iter().map(|&(a, c)| (ActionId(a), c)).collect(),
        rhs,
    }
}

fn evaluate(gate: &PolicyGate, action: u32, context: &[ActionId]) -> GateVerdict {
    gate.evaluate_with_context::<NullFactProvider>(
        ActionId(action),
        context,
        None,
        NormalizedEntropy::ZERO,
        None,
        None,
    )
    .unwrap()
}

fn assert_stop(verdict: GateVerdict) {
    assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
    assert_eq!(verdict.violated_rules.as_slice(), &[RuleId(7)]);
    assert!(verdict.certified_state.is_none());
}

#[test]
fn addition_overflow_and_underflow_fail_closed_in_all_entrypoints() {
    for coefficient in [i32::MAX, i32::MIN] {
        let rule = constraint(&[(1, coefficient), (1, coefficient)], i32::MAX);
        assert!(!rule.is_satisfied_by_single_action(ActionId(1)));
        assert!(!rule.is_satisfied_by_bundle(&[ActionId(1)]));
        let mut gate = PolicyGate::default();
        gate.add_constraint(rule);
        assert_stop(
            gate.evaluate_basic(ActionId(1), NormalizedEntropy::ZERO)
                .unwrap(),
        );
        assert_stop(evaluate(&gate, 1, &[]));
    }
}

#[test]
fn distinct_large_terms_and_multiplication_overflow_fail_closed() {
    for terms in [vec![(1, i32::MAX), (2, i32::MAX)], vec![(1, i32::MAX)]] {
        let mut gate = PolicyGate::default();
        gate.add_constraint(constraint(&terms, i32::MAX));
        let context = if terms.len() == 2 {
            ActionId(2)
        } else {
            ActionId(1)
        };
        assert_stop(evaluate(&gate, 1, &[context]));
    }
}

#[test]
fn intermediate_overflow_cannot_be_cancelled_by_negative_terms() {
    let mut gate = PolicyGate::default();
    gate.add_constraint(constraint(&[(1, i32::MAX), (1, 1), (1, -1)], i32::MAX));
    assert_stop(evaluate(&gate, 1, &[]));
}

#[test]
fn mutex_checks_candidate_and_existing_context_before_lower_tiers() {
    let mut gate = PolicyGate::default();
    gate.add_constraint(LinearConstraint::mutex(
        RuleId(7),
        "mutex",
        ActionId(1),
        ActionId(2),
    ));
    gate.register_confirm_action(ActionId(1));
    assert_stop(evaluate(&gate, 1, &[ActionId(2)]));
    assert_stop(evaluate(&gate, 3, &[ActionId(1), ActionId(2)]));
    assert_eq!(
        evaluate(&gate, 3, &[ActionId(1)]).tier,
        PolicyTier::Tier0Proceed
    );
    assert_eq!(evaluate(&gate, 1, &[]).tier, PolicyTier::Tier1Confirm);
}

#[test]
fn negative_candidate_cannot_hide_invalid_context() {
    let mut gate = PolicyGate::default();
    gate.add_constraint(constraint(&[(1, 1), (2, 1), (3, -1)], 1));
    assert_stop(evaluate(&gate, 3, &[ActionId(1), ActionId(2)]));
}

#[test]
fn invalid_numeric_inputs_stop_both_policy_entrypoints() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
        for (gate, entropy) in [
            (PolicyGate::default(), NormalizedEntropy(invalid)),
            (PolicyGate::new(invalid), NormalizedEntropy::ZERO),
        ] {
            let basic = gate.evaluate_basic(ActionId(1), entropy).unwrap();
            let contextual = gate
                .evaluate_with_context::<NullFactProvider>(
                    ActionId(1),
                    &[ActionId(2)],
                    None,
                    entropy,
                    None,
                    None,
                )
                .unwrap();
            for verdict in [basic, contextual] {
                assert_eq!(verdict.tier, PolicyTier::Tier3HardStop);
                assert!(verdict.certified_state.is_none());
            }
        }
    }
}

#[test]
fn valid_boundary_arithmetic_remains_accepted() {
    for coefficient in [i32::MIN, i32::MAX] {
        let mut gate = PolicyGate::default();
        gate.add_constraint(constraint(&[(1, coefficient)], coefficient));
        assert_eq!(evaluate(&gate, 1, &[]).tier, PolicyTier::Tier0Proceed);
    }
}

#[test]
fn repeated_context_instances_cannot_be_hidden_by_candidate() {
    let mut gate = PolicyGate::default();
    gate.add_constraint(constraint(&[(1, 1), (2, -1)], 1));
    assert_stop(evaluate(&gate, 2, &[ActionId(1), ActionId(1)]));
}
