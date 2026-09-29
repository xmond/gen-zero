"""Deterministic Symbolic Arithmetic Residual Verifier for GSM8K Multi-Step Reasoning.

Addresses the catastrophic 43.3% pure-latent-space accuracy on GSM8K by providing
a closed-loop deterministic arithmetic state machine that validates algebraic
conservation constraints on candidate numerical values and intermediate equation steps.

Key invariants:
- No English keyword regex, no language-specific pattern matching.
- No option-label leakage (never reads "[A]", "[B]", etc. as a signal).
- Strictly structured numeric inputs; natural-language text is never parsed.
- Fail-closed: any malformed input produces a positive residual (penalty).
- Every claim is backed by algebraic identity, not heuristic text scanning.

Core output: R_math(z, c) ∈ [0, +∞), the arithmetic residual for a candidate
numerical answer c given the intermediate step trace z.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EquationStep:
    """A single intermediate arithmetic equation extracted from a reasoning trace.

    The equation is represented as a structured triple (lhs, op, rhs) plus
    an optional expected_result.  For example, the natural-language step
    "23 + 17 = 40" becomes EquationStep(lhs=23.0, op='+', rhs=17.0,
    expected_result=40.0).

    Fields
    ------
    lhs : float
        Left-hand-side operand.
    op : str
        Operator: '+', '-', '*', '/', '^' (power), '==' (equality assertion).
    rhs : float
        Right-hand-side operand.
    expected_result : float | None
        If provided, the claimed result of lhs op rhs.  When None the step
        is treated as an unverified assertion.
    """

    lhs: float
    op: str
    rhs: float
    expected_result: Optional[float] = None

    def __post_init__(self) -> None:
        if self.op not in _SUPPORTED_OPS:
            raise ValueError(
                f"Unsupported operator {self.op!r}; must be one of {sorted(_SUPPORTED_OPS)}"
            )
        if not (math.isfinite(self.lhs) and math.isfinite(self.rhs)):
            raise ValueError(
                f"Non-finite operand: lhs={self.lhs}, rhs={self.rhs}"
            )
        if self.expected_result is not None and not math.isfinite(self.expected_result):
            raise ValueError(
                f"Non-finite expected_result: {self.expected_result}"
            )


@dataclasses.dataclass
class CandidateTrace:
    """A candidate numerical answer together with the reasoning steps that
    purportedly lead to it.

    Fields
    ------
    candidate_value : float
        The final numerical answer proposed by the model.
    steps : list of EquationStep
        Ordered sequence of intermediate arithmetic steps.
    source_tag : str | None
        Optional provenance label (e.g. 'cot-v1', 'latent-bridge').  Not used
        for verification logic — only for diagnostics.
    """

    candidate_value: float
    steps: List[EquationStep]
    source_tag: Optional[str] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.candidate_value):
            raise ValueError(
                f"Non-finite candidate_value: {self.candidate_value}"
            )


@dataclasses.dataclass
class ArithmeticResidual:
    """Result of deterministic arithmetic verification.

    Fields
    ------
    residual : float
        R_math(z, c) ∈ [0, +∞).  0.0 means all verified equations are
        algebraically consistent and the final value is reachable.
        Positive values encode the magnitude of violation.
    passed : bool
        True iff residual == 0.0 and all checks succeeded.
    step_violations : list of (int, str)
        Index and description of each violated equation step.
    final_value_mismatch : bool
        True if the final candidate_value does not equal the last step's result.
    dimension_violations : list of str
        Descriptions of dimension / unit consistency violations.
    factor_violations : list of str
        Descriptions of factorisation / divisibility violations.
    range_violations : list of str
        Descriptions of out-of-range / implausible-magnitude violations.
    diagnostics : dict
        Additional structured diagnostic information.
    """

    residual: float
    passed: bool
    step_violations: List[Tuple[int, str]] = dataclasses.field(default_factory=list)
    final_value_mismatch: bool = False
    dimension_violations: List[str] = dataclasses.field(default_factory=list)
    factor_violations: List[str] = dataclasses.field(default_factory=list)
    range_violations: List[str] = dataclasses.field(default_factory=list)
    diagnostics: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residual": round(self.residual, 6),
            "passed": self.passed,
            "step_violations": [
                {"index": idx, "description": desc}
                for idx, desc in self.step_violations
            ],
            "final_value_mismatch": self.final_value_mismatch,
            "dimension_violations": list(self.dimension_violations),
            "factor_violations": list(self.factor_violations),
            "range_violations": list(self.range_violations),
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_OPS: set = {"+", "-", "*", "/", "^", "=="}

# Residual penalty weights (additive contributions to R_math)
_PENALTY_STEP_VIOLATION: float = 1.0        # per violated equation step
_PENALTY_FINAL_MISMATCH: float = 2.0        # final value doesn't match trace
_PENALTY_DIMENSION: float = 0.5             # per dimension inconsistency
_PENALTY_FACTOR: float = 0.3                # per factorisation violation
_PENALTY_RANGE: float = 0.2                 # per range implausibility

# Tolerance for floating-point equality checks
_FP_TOLERANCE: float = 1e-9

# Plausible magnitude bounds for GSM8K-grade arithmetic (middle-school math).
# Answers outside this range are flagged as implausible but not automatically
# rejected — the residual penalty is additive.
_DEFAULT_MIN_PLAUSIBLE: float = -1e6
_DEFAULT_MAX_PLAUSIBLE: float = 1e9


# ---------------------------------------------------------------------------
# Core arithmetic engine
# ---------------------------------------------------------------------------

def _apply_op(lhs: float, op: str, rhs: float, tolerance: float = _FP_TOLERANCE) -> float:
    """Deterministically apply a binary arithmetic operator.

    Division by zero returns NaN (caught upstream as a violation). Other
    domain errors (e.g. 0 ** -1, or a negative base to a fractional power,
    which silently yields a complex number rather than raising) are the
    caller's responsibility via `_safe_apply_op`.
    """
    if op == "+":
        return lhs + rhs
    elif op == "-":
        return lhs - rhs
    elif op == "*":
        return lhs * rhs
    elif op == "/":
        if rhs == 0.0:
            return float("nan")
        return lhs / rhs
    elif op == "^":
        return lhs ** rhs
    elif op == "==":
        # `<=` matches _verify_step, so a diff exactly at the tolerance gets one verdict.
        return float(abs(lhs - rhs) <= tolerance)
    else:
        raise ValueError(f"Unknown operator: {op!r}")


def _safe_apply_op(lhs: float, op: str, rhs: float, tolerance: float = _FP_TOLERANCE) -> float:
    """Apply the operator; convert any evaluation failure into NaN.

    `(-8) ** (1/3)` doesn't raise, it silently returns a complex number, and
    `0 ** -1` raises ZeroDivisionError from inside `**`. Both are converted to
    NaN here so `math.isfinite` turns them into an ordinary step violation
    (a positive, finite residual) instead of an uncaught crash.
    """
    try:
        result = _apply_op(lhs, op, rhs, tolerance)
    except (TypeError, ZeroDivisionError, ValueError, OverflowError):
        return float("nan")
    if isinstance(result, complex):
        return float("nan")
    return result


def _is_integer_value(x: float, tol: float = _FP_TOLERANCE) -> bool:
    """Check whether a float is effectively an integer."""
    return abs(x - round(x)) <= tol


def _gcd_int(a: int, b: int) -> int:
    """Euclidean GCD of two non-negative integers."""
    while b != 0:
        a, b = b, a % b
    return a


# ---------------------------------------------------------------------------
# Dimension / unit consistency
# ---------------------------------------------------------------------------
#
# A prior version of this check flagged any addition/subtraction step where
# one operand was integer-valued and the other fractional with a magnitude
# ratio over 100x, calling it a probable "dimension mismatch". That signal is
# unsound: whether a number happens to be integer-valued is not a dimension —
# 1000 + 0.5 = 1000.5 is perfectly ordinary arithmetic on same-dimension
# quantities (e.g. dollars), not a unit conflict. There is no reliable way to
# infer physical dimension from float representation alone, so this check was
# removed rather than re-tuned. `dimension_violations` stays in the result
# shape (always empty) for API stability.


# ---------------------------------------------------------------------------
# Factorisation / divisibility checks
# ---------------------------------------------------------------------------

def _check_factorisation(
    steps: List[EquationStep],
) -> List[str]:
    """Check factorisation and divisibility consistency.

    For multiplication and division steps involving integer-valued operands,
    verify that the expected result respects integer divisibility constraints.
    For example, if step is "12 / 5 = 2.4" that's fine (rational result),
    but if step is "12 / 5 = 2" (claimed integer result) that's a violation.

    Returns list of violation descriptions.
    """
    violations: List[str] = []
    for i, step in enumerate(steps):
        if step.expected_result is None:
            continue
        if step.op not in ("*", "/"):
            continue
        lhs_int = _is_integer_value(step.lhs)
        rhs_int = _is_integer_value(step.rhs)
        res_int = _is_integer_value(step.expected_result)

        # For division of two integers claiming an integer result, verify exact divisibility
        if step.op == "/" and lhs_int and rhs_int and res_int:
            a = int(round(step.lhs))
            b = int(round(step.rhs))
            if b == 0:
                continue  # division by zero caught by step verification
            if a % b != 0:
                violations.append(
                    f"Step {i}: claimed integer result {int(round(step.expected_result))} "
                    f"for {a} / {b}, but {b} does not divide {a}"
                )
        # For multiplication of two integers claiming an integer result, verify the product
        if step.op == "*" and lhs_int and rhs_int and res_int:
            expected_prod = int(round(step.lhs)) * int(round(step.rhs))
            actual = int(round(step.expected_result))
            if expected_prod != actual:
                violations.append(
                    f"Step {i}: claimed integer product {actual} for "
                    f"{int(round(step.lhs))} * {int(round(step.rhs))}, "
                    f"but actual product is {expected_prod}"
                )
    return violations


# ---------------------------------------------------------------------------
# Range / plausibility checks
# ---------------------------------------------------------------------------

def _check_range(
    candidate_value: float,
    steps: List[EquationStep],
    min_plausible: float = _DEFAULT_MIN_PLAUSIBLE,
    max_plausible: float = _DEFAULT_MAX_PLAUSIBLE,
) -> List[str]:
    """Check that intermediate and final values fall within plausible ranges
    for GSM8K-grade arithmetic problems.

    Returns list of violation descriptions.
    """
    violations: List[str] = []
    if candidate_value < min_plausible or candidate_value > max_plausible:
        violations.append(
            f"Final candidate {candidate_value} outside plausible range "
            f"[{min_plausible}, {max_plausible}]"
        )
    for i, step in enumerate(steps):
        if step.expected_result is not None:
            if step.expected_result < min_plausible or step.expected_result > max_plausible:
                violations.append(
                    f"Step {i}: expected_result {step.expected_result} "
                    f"outside plausible range [{min_plausible}, {max_plausible}]"
                )
    return violations


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

class SymbolicArithmeticVerifier:
    """Deterministic closed-loop arithmetic verifier.

    Validates a candidate numerical answer against its claimed intermediate
    equation steps using purely algebraic identity checks.  No natural-language
    parsing, no regex, no option-label leakage.

    Usage
    -----
    >>> verifier = SymbolicArithmeticVerifier()
    >>> trace = CandidateTrace(
    ...     candidate_value=40.0,
    ...     steps=[
    ...         EquationStep(23.0, '+', 17.0, expected_result=40.0),
    ...     ],
    ... )
    >>> result = verifier.verify(trace)
    >>> result.passed
    True
    >>> result.residual
    0.0
    """

    def __init__(
        self,
        fp_tolerance: float = _FP_TOLERANCE,
        min_plausible: float = _DEFAULT_MIN_PLAUSIBLE,
        max_plausible: float = _DEFAULT_MAX_PLAUSIBLE,
    ):
        if not np.isfinite(fp_tolerance) or fp_tolerance <= 0:
            raise ValueError(f"fp_tolerance must be finite and positive, got {fp_tolerance!r}")
        self.fp_tolerance = fp_tolerance
        self.min_plausible = min_plausible
        self.max_plausible = max_plausible

    # ------------------------------------------------------------------
    # Step-level verification
    # ------------------------------------------------------------------

    def _verify_step(self, step: EquationStep, index: int) -> Optional[str]:
        """Verify a single equation step.

        Returns a violation description string, or None if the step is valid.
        """
        # Equality assertion: lhs must actually equal rhs, and if a claimed
        # expected_result is attached it must agree with that true/false
        # verdict (1.0/0.0) — an unvalidated expected_result would let a
        # step claim a true equality is false, or vice versa, without being
        # caught.
        if step.op == "==":
            equal = abs(step.lhs - step.rhs) <= self.fp_tolerance
            if not equal:
                return (
                    f"Step {index}: equality assertion failed — "
                    f"lhs={step.lhs} != rhs={step.rhs} "
                    f"(diff={abs(step.lhs - step.rhs):.3e})"
                )
            if step.expected_result is not None and abs(step.expected_result - 1.0) > self.fp_tolerance:
                return (
                    f"Step {index}: equality assertion result mismatch — "
                    f"lhs={step.lhs} == rhs={step.rhs} is true, but "
                    f"expected_result claims {step.expected_result}"
                )
            return None

        # Compute actual result; domain errors (complex results, division/power
        # errors) are converted to NaN rather than raised.
        actual = _safe_apply_op(step.lhs, step.op, step.rhs, self.fp_tolerance)

        if not math.isfinite(actual):
            return (
                f"Step {index}: non-finite result — "
                f"{step.lhs} {step.op} {step.rhs} = {actual}"
            )

        # If expected_result is provided, check it matches
        if step.expected_result is not None:
            diff = abs(actual - step.expected_result)
            if diff > self.fp_tolerance:
                return (
                    f"Step {index}: arithmetic mismatch — "
                    f"{step.lhs} {step.op} {step.rhs} = {actual}, "
                    f"but expected {step.expected_result} "
                    f"(diff={diff:.3e})"
                )

        return None

    # ------------------------------------------------------------------
    # Full trace verification
    # ------------------------------------------------------------------

    def verify(self, trace: CandidateTrace) -> ArithmeticResidual:
        """Perform complete deterministic verification of a candidate trace.

        Parameters
        ----------
        trace : CandidateTrace
            The candidate answer and its supporting equation steps.

        Returns
        -------
        ArithmeticResidual
            Structured result with residual score and detailed violation lists.
        """
        # --- Fail-closed: empty trace with a finite candidate is suspicious ---
        if len(trace.steps) == 0:
            return ArithmeticResidual(
                residual=float("inf"),
                passed=False,
                step_violations=[],
                diagnostics={
                    "reason": "empty_trace",
                    "detail": "No equation steps provided; cannot verify candidate.",
                },
            )

        # --- Step-level verification ---
        step_violations: List[Tuple[int, str]] = []
        for i, step in enumerate(trace.steps):
            violation = self._verify_step(step, i)
            if violation is not None:
                step_violations.append((i, violation))

        # --- Final value chain check ---
        # The final candidate_value should equal the last step's expected_result
        # (or its computed result if no expected_result).
        final_value_mismatch = False
        last_step = trace.steps[-1]
        if last_step.expected_result is not None:
            last_val = last_step.expected_result
        else:
            last_val = _safe_apply_op(last_step.lhs, last_step.op, last_step.rhs, self.fp_tolerance)
        if not math.isfinite(last_val):
            final_value_mismatch = True
        elif abs(trace.candidate_value - last_val) > self.fp_tolerance:
            final_value_mismatch = True

        # --- Dimension consistency: removed unsound integer/fraction ratio
        # heuristic (see module comment above); always empty now.
        dimension_violations: List[str] = []

        # --- Factorisation checks ---
        factor_violations = _check_factorisation(trace.steps)

        # --- Range checks ---
        range_violations = _check_range(
            trace.candidate_value,
            trace.steps,
            self.min_plausible,
            self.max_plausible,
        )

        # --- Compute residual ---
        residual = 0.0
        residual += len(step_violations) * _PENALTY_STEP_VIOLATION
        if final_value_mismatch:
            residual += _PENALTY_FINAL_MISMATCH
        residual += len(dimension_violations) * _PENALTY_DIMENSION
        residual += len(factor_violations) * _PENALTY_FACTOR
        residual += len(range_violations) * _PENALTY_RANGE

        passed = (
            residual == 0.0
            and not final_value_mismatch
            and len(step_violations) == 0
            and len(dimension_violations) == 0
            and len(factor_violations) == 0
            and len(range_violations) == 0
        )

        return ArithmeticResidual(
            residual=residual,
            passed=passed,
            step_violations=step_violations,
            final_value_mismatch=final_value_mismatch,
            dimension_violations=dimension_violations,
            factor_violations=factor_violations,
            range_violations=range_violations,
            diagnostics={
                "num_steps": len(trace.steps),
                "fp_tolerance": self.fp_tolerance,
                "source_tag": trace.source_tag,
            },
        )

    # ------------------------------------------------------------------
    # Batch verification
    # ------------------------------------------------------------------

    def verify_batch(
        self, traces: Sequence[CandidateTrace]
    ) -> List[ArithmeticResidual]:
        """Verify multiple candidate traces.

        Returns a list of ArithmeticResidual, one per input trace, in order.
        """
        return [self.verify(t) for t in traces]

    # ------------------------------------------------------------------
    # Penalty gradient (for potential downstream optimization)
    # ------------------------------------------------------------------

    def penalty_gradient(
        self, result: ArithmeticResidual
    ) -> Dict[str, float]:
        """Decompose the residual into per-category penalty contributions.

        Useful for diagnostic dashboards and for hooking into a training
        signal that wants to back-propagate through which specific constraint
        was violated.

        Returns
        -------
        dict with keys: step, final_mismatch, dimension, factor, range, total
        """
        grad = {
            "step": len(result.step_violations) * _PENALTY_STEP_VIOLATION,
            "final_mismatch": _PENALTY_FINAL_MISMATCH if result.final_value_mismatch else 0.0,
            "dimension": len(result.dimension_violations) * _PENALTY_DIMENSION,
            "factor": len(result.factor_violations) * _PENALTY_FACTOR,
            "range": len(result.range_violations) * _PENALTY_RANGE,
        }
        grad["total"] = sum(grad.values())
        return grad


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def trace_from_value_steps(
    candidate_value: float,
    steps_raw: List[Tuple[float, str, float, Optional[float]]],
    source_tag: Optional[str] = None,
) -> CandidateTrace:
    """Construct a CandidateTrace from a compact list-of-tuples representation.

    Parameters
    ----------
    candidate_value : float
        The final answer.
    steps_raw : list of (lhs, op, rhs, expected_result | None)
        Each tuple is one equation step.
    source_tag : str | None
        Optional provenance label.

    Returns
    -------
    CandidateTrace
    """
    steps = [
        EquationStep(lhs=l, op=op, rhs=r, expected_result=exp)
        for l, op, r, exp in steps_raw
    ]
    return CandidateTrace(
        candidate_value=candidate_value,
        steps=steps,
        source_tag=source_tag,
    )
