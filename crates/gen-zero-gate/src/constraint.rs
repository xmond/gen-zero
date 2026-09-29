//! gen-zero-gate 0-1 ILP Linear Constraint compiler and representation.

use crate::error::GateError;
use arrayvec::ArrayVec;
use gen_zero_core::ActionId;
use serde::{Deserialize, Serialize};

/// Strongly-typed formal rule identifier
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RuleId(pub u32);

/// 0-1 Integer Linear Programming constraint: sum(c_i * x_i) <= rhs.
/// Uses stack-inlined ArrayVec up to 16 terms to guarantee zero heap allocations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearConstraint {
    pub rule_id: RuleId,
    pub rule_name: &'static str,
    /// Stack-inlined terms: (ActionId, coefficient c_i)
    pub terms: ArrayVec<(ActionId, i32), 16>,
    /// Upper bound rhs
    pub rhs: i32,
}

impl LinearConstraint {
    /// Create a mutual exclusion constraint between two actions: x_a + x_b <= 1
    pub fn mutex(rule_id: RuleId, rule_name: &'static str, a: ActionId, b: ActionId) -> Self {
        let mut terms = ArrayVec::new();
        terms.push((a, 1));
        terms.push((b, 1));
        Self {
            rule_id,
            rule_name,
            terms,
            rhs: 1,
        }
    }

    /// Create an absolute prohibition constraint: x_a <= 0
    pub fn prohibit(rule_id: RuleId, rule_name: &'static str, a: ActionId) -> Self {
        let mut terms = ArrayVec::new();
        terms.push((a, 1));
        Self {
            rule_id,
            rule_name,
            terms,
            rhs: 0,
        }
    }

    /// Create a quota limit constraint on a set of actions: sum(x_i) <= max_count
    pub fn quota(
        rule_id: RuleId,
        rule_name: &'static str,
        actions: &[ActionId],
        max_count: i32,
    ) -> Result<Self, GateError> {
        if actions.len() > 16 {
            return Err(GateError::CompilationError(
                "LinearConstraint exceeds maximum stack capacity of 16 terms".into(),
            ));
        }
        let mut terms = ArrayVec::new();
        for &act in actions {
            terms.push((act, 1));
        }
        Ok(Self {
            rule_id,
            rule_name,
            terms,
            rhs: max_count,
        })
    }

    /// Verify whether activating a candidate action satisfies this linear constraint.
    /// In single-action decision setting: x_a = 1 and all other x_{j != a} = 0.
    #[inline]
    pub fn is_satisfied_by_single_action(&self, action: ActionId) -> bool {
        self.is_satisfied_by_counts(|act| usize::from(act == action))
    }

    /// Verify whether a concurrent bundle of actions satisfies this linear constraint.
    pub fn is_satisfied_by_bundle(&self, active_actions: &[ActionId]) -> bool {
        self.is_satisfied_by_counts(|act| usize::from(active_actions.contains(&act)))
    }

    /// Verify a candidate plus all executing instances in active_context.
    /// Counts and arithmetic outside the i32 constraint domain fail closed.
    pub fn is_satisfied_with_context(
        &self,
        candidate: ActionId,
        active_context: &[ActionId],
    ) -> bool {
        self.is_satisfied_by_counts(|act| {
            // The iterator avoids narrowing a usize count before validation.
            active_context
                .iter()
                .copied()
                .chain(std::iter::once(candidate))
                .filter(|&a| a == act)
                .count()
        })
    }

    /// Check executing instances with the same multiplicity as candidate evaluation.
    pub(crate) fn is_satisfied_by_active_context(&self, active_context: &[ActionId]) -> bool {
        self.is_satisfied_by_counts(|act| active_context.iter().filter(|&&a| a == act).count())
    }

    fn is_satisfied_by_counts(&self, count: impl Fn(ActionId) -> usize) -> bool {
        let mut lhs = 0_i128;
        for &(act, coeff) in &self.terms {
            let Ok(count) = i128::try_from(count(act)) else {
                return false;
            };
            let Some(term) = i128::from(coeff).checked_mul(count) else {
                return false;
            };
            let Some(sum) = lhs.checked_add(term) else {
                return false;
            };
            // Wide checked arithmetic prevents machine overflow; the declared
            // i32 constraint domain must also hold at every intermediate step.
            if i32::try_from(count).is_err()
                || i32::try_from(term).is_err()
                || i32::try_from(sum).is_err()
            {
                return false;
            }
            lhs = sum;
        }
        lhs <= i128::from(self.rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_coefficients_do_not_wrap_or_panic() {
        let action = ActionId(1);
        for coefficient in [i32::MAX, i32::MIN] {
            let mut rule = LinearConstraint::prohibit(RuleId(1), "extreme", action);
            rule.terms.clear();
            for _ in 0..16 {
                rule.terms.push((action, coefficient));
            }
            let expected = false; // Out-of-domain intermediates fail closed, including underflow.
            assert_eq!(rule.is_satisfied_by_single_action(action), expected);
            assert_eq!(rule.is_satisfied_by_bundle(&[action]), expected);
            assert_eq!(
                rule.is_satisfied_with_context(action, &[action; 16]),
                expected
            );
        }
    }

    #[test]
    fn test_linear_constraints() {
        let rule_mutex =
            LinearConstraint::mutex(RuleId(1), "NoConcurrentTrading", ActionId(10), ActionId(20));
        assert!(rule_mutex.is_satisfied_by_single_action(ActionId(10)));
        assert!(rule_mutex.is_satisfied_by_single_action(ActionId(20)));
        // Mutex conflict: cannot execute 10 when 20 is already active
        assert!(!rule_mutex.is_satisfied_with_context(ActionId(10), &[ActionId(20)]));

        let rule_prohibit =
            LinearConstraint::prohibit(RuleId(2), "ForbiddenDeleteRoot", ActionId(99));
        assert!(!rule_prohibit.is_satisfied_by_single_action(ActionId(99)));
        assert!(rule_prohibit.is_satisfied_by_single_action(ActionId(1)));
    }

    #[test]
    fn constraint_evaluators_do_not_wrap_i32_accumulators() {
        let mut terms = ArrayVec::new();
        terms.push((ActionId(7), i32::MAX));
        terms.push((ActionId(7), i32::MAX));
        let rule = LinearConstraint {
            rule_id: RuleId(3),
            rule_name: "WideAccumulator",
            terms,
            rhs: 0,
        };

        assert!(!rule.is_satisfied_by_single_action(ActionId(7)));
        assert!(!rule.is_satisfied_by_bundle(&[ActionId(7)]));
        assert!(!rule.is_satisfied_with_context(ActionId(7), &[]));
        // Context multiplicity is part of the left-hand side and must remain
        // widened before coefficient multiplication as well.
        assert!(!rule.is_satisfied_with_context(ActionId(7), &[ActionId(7), ActionId(7)]));
    }
}
