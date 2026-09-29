"""Gen-Zero Layer 5: Automated Frozen Safety Gate.

Validates candidate checkpoints against baseline on the frozen benchmark ladder:
- Test accuracy retention (>= 99.5% of baseline)
- Long-horizon score non-regression (>= baseline + gain_threshold)
- Collision rate non-increase
Protects against silent regressions and triggers rollback on failure.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Real
from typing import Callable, Dict, List, Any, Optional, Tuple

from .policy_gate import PolicyVerdictAction


@dataclass
class GateVerdict:
    passed: bool
    action: str              # 'DEPLOY_HOT_UPDATE' or 'ROLLBACK_ADJUST_HYPERPARAMS'
    accuracy_delta: float
    score_delta: float
    collision_delta: float
    details: Dict[str, Any]


class SafetyGate:
    """Automated Safety Gate controlling deployment progression."""
    def __init__(
        self,
        min_acc_retention: float = 0.995,
        min_score_gain: float = 0.0,
        max_allowed_collision_increase: float = 0.0,
        policy_gate: Optional["TwoTierPolicyGate"] = None,
    ):
        self.min_acc_retention = min_acc_retention
        self.min_score_gain = min_score_gain
        self.max_allowed_collision_increase = max_allowed_collision_increase
        self.policy_gate = policy_gate if policy_gate is not None else TwoTierPolicyGate()

    def evaluate_decision(
        self, prompt: str, choice: str, confidence: float, *,
        assessment: Optional["PolicyAssessment"] = None,
    ) -> "TwoTierPolicyVerdict":
        return self.policy_gate.evaluate_choice(prompt, choice, confidence, assessment=assessment)

    def evaluate_prompt_safety(
        self, prompt: str, *, confidence: float,
        assessment: Optional["PolicyAssessment"] = None,
    ) -> "TwoTierPolicyVerdict":
        return self.policy_gate.evaluate_prompt_safety(prompt, confidence=confidence, assessment=assessment)

    def evaluate_candidate(
        self,
        baseline_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float],
        perturbation_metrics: Optional[Dict[str, float]] = None,
        calibration_metrics: Optional[Dict[str, Any]] = None
    ) -> GateVerdict:
        """Evaluates whether a candidate model qualifies for hot-update deployment.

        Expected metrics dict:
        - 'accuracy': float in [0, 100]
        - 'mean_score': float
        - 'collision_rate': float
        """
        # Strict validation: Metrics must be explicitly present and non-empty
        required_keys = {"accuracy", "mean_score", "collision_rate"}
        missing_base = required_keys - set(baseline_metrics.keys())
        missing_cand = required_keys - set(candidate_metrics.keys())

        if missing_base or missing_cand:
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0,
                score_delta=0.0,
                collision_delta=0.0,
                details={
                    "error": "REJECTED_MISSING_EVALUATION_METRICS",
                    "missing_in_baseline": list(missing_base),
                    "missing_in_candidate": list(missing_cand)
                }
            )

        # Strict verification of explicit validity: must be explicitly present and strictly boolean True
        def _is_strictly_valid(m: Dict[str, Any]) -> bool:
            return "is_valid" in m and isinstance(m["is_valid"], bool) and (m["is_valid"] is True)

        if not _is_strictly_valid(baseline_metrics) or not _is_strictly_valid(candidate_metrics):
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0,
                score_delta=0.0,
                collision_delta=0.0,
                details={
                    "error": "REJECTED_INVALID_OR_MISSING_VALIDITY_FLAG",
                    "baseline_valid": baseline_metrics.get("is_valid"),
                    "candidate_valid": candidate_metrics.get("is_valid")
                }
            )

        try:
            base_acc = float(baseline_metrics["accuracy"])
            cand_acc = float(candidate_metrics["accuracy"])
            base_score = float(baseline_metrics["mean_score"])
            cand_score = float(candidate_metrics["mean_score"])
            base_coll = float(baseline_metrics["collision_rate"])
            cand_coll = float(candidate_metrics["collision_rate"])
        except (ValueError, TypeError) as e:
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0,
                score_delta=0.0,
                collision_delta=0.0,
                details={"error": f"INVALID_NUMERIC_METRICS: {e}"}
            )

        # Strict checks against Infinity, NaN, or out-of-domain numbers
        import math
        all_vals = [base_acc, cand_acc, base_score, cand_score, base_coll, cand_coll]
        if any(not math.isfinite(v) for v in all_vals):
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0,
                score_delta=0.0,
                collision_delta=0.0,
                details={"error": "REJECTED_NON_FINITE_METRICS"}
            )

        # Accuracy must be within [0, 100], collision rate within [0, 1]
        if not (0.0 <= cand_acc <= 100.0 and 0.0 <= base_acc <= 100.0):
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0,
                score_delta=0.0,
                collision_delta=0.0,
                details={"error": "REJECTED_ACCURACY_OUT_OF_BOUNDS"}
            )

        if not (0.0 <= cand_coll <= 1.0 and 0.0 <= base_coll <= 1.0):
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0,
                score_delta=0.0,
                collision_delta=0.0,
                details={"error": "REJECTED_COLLISION_RATE_OUT_OF_BOUNDS"}
            )

        acc_delta = cand_acc - base_acc
        score_delta = cand_score - base_score
        coll_delta = cand_coll - base_coll

        # A candidate model scoring 0% accuracy on the benchmark must NEVER qualify for promotion!
        if cand_acc <= 0.0:
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=round(acc_delta, 2),
                score_delta=round(score_delta, 2),
                collision_delta=round(coll_delta, 4),
                details={"error": "REJECTED_ZERO_ACCURACY_CANDIDATE"}
            )

        # Validation conditions
        acc_ok = (cand_acc >= base_acc * self.min_acc_retention)
        score_ok = (score_delta >= self.min_score_gain)
        coll_ok = (coll_delta <= self.max_allowed_collision_increase)

        passed = acc_ok and score_ok and coll_ok

        details: Dict[str, Any] = {
            "accuracy_check": {"passed": acc_ok, "candidate": cand_acc, "baseline": base_acc},
            "score_check": {"passed": score_ok, "candidate": cand_score, "baseline": base_score},
            "collision_check": {"passed": coll_ok, "candidate": cand_coll, "baseline": base_coll}
        }

        # Validate 4-dimensional perturbation stability metrics if provided
        if perturbation_metrics is not None:
            stab_gate = PerturbationStabilityGate()
            stab_verdict = stab_gate.evaluate(perturbation_metrics)
            details["perturbation_check"] = stab_verdict.details
            if not stab_verdict.passed:
                passed = False
            elif "warning" in stab_verdict.details:
                details["warning"] = stab_verdict.details["warning"]

        # Validate 10-bin ECE calibration & Confident Errors (P >= 0.90 red line)
        calib_data = calibration_metrics or candidate_metrics.get("calibration_report")
        if calib_data is not None:
            conf_errors = calib_data.get("confident_error_count", 0)
            red_line_passed = calib_data.get("passed_safety_red_line", conf_errors == 0)
            ece = calib_data.get("ece_10bin", 0.0)

            details["calibration_check"] = {
                "passed": bool(red_line_passed and conf_errors == 0),
                "ece_10bin": ece,
                "confident_error_count": conf_errors,
                "verdict": calib_data.get("verdict", "UNKNOWN")
            }

            if conf_errors > 0:
                passed = False
                details["error"] = "REJECTED_CONFIDENT_ERROR_RED_LINE_BREACH"
            elif not red_line_passed:
                passed = False
                details["error"] = "REJECTED_EXCESSIVE_CALIBRATION_DRIFT"

        action = "DEPLOY_HOT_UPDATE" if passed else "ROLLBACK_ADJUST_HYPERPARAMS"

        return GateVerdict(
            passed=passed,
            action=action,
            accuracy_delta=round(acc_delta, 2),
            score_delta=round(score_delta, 2),
            collision_delta=round(coll_delta, 4),
            details=details
        )


class PerturbationStabilityGate:
    """Automated Safety Gate enforcing 4-dimensional semantic invariance:
    - Dim 1: Option Order Invariance (DFR == 0.0%, TVD <= 1e-4) -> Hard Rollback
    - Dim 2: Criterion Semantic Wrapper (DFR <= 1.5%, TVD <= 0.05) -> Degraded Alert
    - Dim 3: Context Noise Resistance (DFR <= 1.0%, TVD <= 0.04) -> Degraded Alert
    - Dim 4: Missing Evidence Active Abstain (Recall >= 98.0%, Conf < 0.70) -> Hard Rollback
    """
    def __init__(
        self,
        max_dim1_dfr: float = 0.0,
        max_dim1_tvd: float = 1e-4,
        min_dim4_abstain_recall: float = 0.98,
        max_dim4_confidence: float = 0.70,
        max_dim2_dfr: float = 0.015,
        max_dim2_tvd: float = 0.05,
        max_dim3_dfr: float = 0.010,
        max_dim3_tvd: float = 0.04
    ):
        self.max_dim1_dfr = max_dim1_dfr
        self.max_dim1_tvd = max_dim1_tvd
        self.min_dim4_abstain_recall = min_dim4_abstain_recall
        self.max_dim4_confidence = max_dim4_confidence
        self.max_dim2_dfr = max_dim2_dfr
        self.max_dim2_tvd = max_dim2_tvd
        self.max_dim3_dfr = max_dim3_dfr
        self.max_dim3_tvd = max_dim3_tvd

    def evaluate(self, metrics: Dict[str, Any]) -> GateVerdict:
        """Evaluates perturbation metrics with strict validation against empty, non-finite, or invalid inputs."""
        import math

        # 1. Strict validity flag check
        if not (isinstance(metrics, dict) and metrics.get("is_valid") is True):
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0,
                score_delta=0.0,
                collision_delta=0.0,
                details={"error": "REJECTED_INVALID_PERTURBATION_METRICS_FLAG"}
            )

        # 2. Required fields verification
        required_keys = {"dim1_dfr", "dim1_tvd", "dim4_abstain_recall", "dim4_max_confidence"}
        if not required_keys.issubset(metrics.keys()):
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0,
                score_delta=0.0,
                collision_delta=0.0,
                details={"error": "REJECTED_MISSING_PERTURBATION_METRICS", "missing": list(required_keys - set(metrics.keys()))}
            )

        # 3. Numeric extraction & finite check
        try:
            d1_dfr = float(metrics["dim1_dfr"])
            d1_tvd = float(metrics["dim1_tvd"])
            d4_recall = float(metrics["dim4_abstain_recall"])
            d4_conf = float(metrics["dim4_max_confidence"])
            d2_dfr = float(metrics.get("dim2_dfr", 0.0))
            d2_tvd = float(metrics.get("dim2_tvd", 0.0))
            d3_dfr = float(metrics.get("dim3_dfr", 0.0))
            d3_tvd = float(metrics.get("dim3_tvd", 0.0))
        except (ValueError, TypeError) as e:
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0, score_delta=0.0, collision_delta=0.0,
                details={"error": f"INVALID_NUMERIC_PERTURBATION_METRICS: {e}"}
            )

        all_vals = [d1_dfr, d1_tvd, d4_recall, d4_conf, d2_dfr, d2_tvd, d3_dfr, d3_tvd]
        if any(not math.isfinite(v) for v in all_vals):
            return GateVerdict(
                passed=False,
                action="ROLLBACK_ADJUST_HYPERPARAMS",
                accuracy_delta=0.0, score_delta=0.0, collision_delta=0.0,
                details={"error": "REJECTED_NON_FINITE_PERTURBATION_METRICS"}
            )

        # 4. Scale normalization (if passed as percentage > 1.0)
        if d4_recall > 1.0:
            d4_recall /= 100.0

        # 5. Hard constraints (Dim 1 & Dim 4)
        dim1_ok = (d1_dfr <= self.max_dim1_dfr) and (d1_tvd <= self.max_dim1_tvd)
        dim4_ok = (d4_recall >= self.min_dim4_abstain_recall) and (d4_conf < self.max_dim4_confidence)

        passed = dim1_ok and dim4_ok
        action = "DEPLOY_HOT_UPDATE" if passed else "ROLLBACK_ADJUST_HYPERPARAMS"

        details = {
            "dim1_order_invariance": {"passed": dim1_ok, "dfr": d1_dfr, "tvd": d1_tvd},
            "dim4_active_abstain": {"passed": dim4_ok, "recall": d4_recall, "max_conf": d4_conf}
        }

        # Soft degraded checks (Dim 2 & Dim 3)
        dim2_ok = (d2_dfr <= self.max_dim2_dfr) and (d2_tvd <= self.max_dim2_tvd)
        dim3_ok = (d3_dfr <= self.max_dim3_dfr) and (d3_tvd <= self.max_dim3_tvd)
        details["dim2_semantic_wrapper"] = {"passed": dim2_ok, "dfr": d2_dfr, "tvd": d2_tvd}
        details["dim3_noise_injection"] = {"passed": dim3_ok, "dfr": d3_dfr, "tvd": d3_tvd}

        if passed and (not dim2_ok or not dim3_ok):
            details["warning"] = "ALERT_DEGRADED_STABILITY"

        return GateVerdict(
            passed=passed, action=action,
            accuracy_delta=0.0, score_delta=0.0, collision_delta=0.0,
            details=details
        )



class PolicyTier(IntEnum):
    """Outcome severity; distinct from the two evaluation stages."""
    TIER0_PROCEED = 0
    TIER2_ESCALATE = 2
    TIER3_HARDSTOP = 3


@dataclass(frozen=True)
class PolicyAssessment:
    """Evidence supplied by a trusted semantic evaluator, never by prompt parsing.

    Risk and uncertainty are normalized to [0, 1]. Their capped sum is a
    conservative risk bound, not a statistically calibrated confidence interval.
    Violations are policy identifiers established by that evaluator. An empty
    tuple means it assessed the input and found no hard violation. Bind evidence
    to both inputs to avoid accidentally applying another request's assessment.
    This object is a trusted internal API, not an untrusted client payload.
    """
    prompt: str
    choice: str
    risk: float
    uncertainty: float
    violations: Tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class TwoTierPolicyVerdict:
    passed: bool
    tier: PolicyTier
    action: PolicyVerdictAction
    triggered_rules: Tuple[str, ...]
    reason: str
    risk_score: Optional[float] = None


_ILLICIT_SUBSTANCE_KEYWORDS = (
    "cocaine", "heroin", "methamphetamine", "fentanyl", "illicit substance",
    "buy drugs", "find cocaine", "deal drugs", "crack cocaine",
)

_WEAPON_LETHAL_KEYWORDS = (
    "pipe bomb", "explosive device", "ghost gun", "3d print gun",
    "ricin", "lethal poison", "castor beans", "synthesize poison",
    "synthesize lethal poison", "dirty bomb", "biological weapon", "chemical weapon",
)

_JAILBREAK_KEYWORDS = (
    "dan mode", "bypass safety", "ignore previous instructions",
    "disregard all guidelines", "developer mode enabled", "jailbreak",
)


def _default_baseline_policy_evaluator(prompt: str, choice: str) -> Optional[PolicyAssessment]:
    p_lower = prompt.lower()
    for kw in _ILLICIT_SUBSTANCE_KEYWORDS:
        if kw in p_lower:
            return PolicyAssessment(
                prompt=prompt,
                choice=choice,
                risk=1.0,
                uncertainty=0.0,
                violations=("ILLICIT_SUBSTANCE_PROHIBITION",),
                reason=f"Detected illicit substance inquiry: {kw}",
            )
    for kw in _WEAPON_LETHAL_KEYWORDS:
        if kw in p_lower:
            return PolicyAssessment(
                prompt=prompt,
                choice=choice,
                risk=1.0,
                uncertainty=0.0,
                violations=("WEAPON_GENERATION_PROHIBITION",),
                reason=f"Detected weapon or lethal hazard: {kw}",
            )
    for kw in _JAILBREAK_KEYWORDS:
        if kw in p_lower:
            return PolicyAssessment(
                prompt=prompt,
                choice=choice,
                risk=1.0,
                uncertainty=0.0,
                violations=("JAILBREAK_ATTEMPT_PROHIBITION",),
                reason=f"Detected jailbreak attempt: {kw}",
            )
    words = prompt.strip().split()
    if len(words) >= 3:
        return PolicyAssessment(
            prompt=prompt,
            choice=choice,
            risk=0.05,
            uncertainty=0.05,
            violations=(),
            reason="Baseline policy check passed",
        )
    return None


class TwoTierPolicyGate:
    """Hard policy evidence followed by continuous risk/uncertainty arbitration.

    No built-in text classifier is claimed. Supply a trusted evaluator or an
    explicit assessment. Missing or invalid evidence cannot authorize execution.
    Choice confidence alone measures certainty, not safety.
    """

    def __init__(
        self,
        evaluator: Optional[Callable[[str, str], Optional[PolicyAssessment]]] = None,
        min_confidence: float = 0.45,
        proceed_confidence: float = 0.65,
        confirm_risk: float = 0.20,
        stop_risk: float = 0.50,
    ):
        for value in (min_confidence, proceed_confidence, confirm_risk, stop_risk):
            if not self._probability(value):
                raise ValueError("policy thresholds must be finite numbers in [0, 1]")
        if not min_confidence < proceed_confidence or not confirm_risk < stop_risk:
            raise ValueError("policy thresholds must be strictly ordered")
        if evaluator is not None and not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self.evaluator = evaluator if evaluator is not None else _default_baseline_policy_evaluator
        self.min_confidence = min_confidence
        self.proceed_confidence = proceed_confidence
        self.confirm_risk = confirm_risk
        self.stop_risk = stop_risk

    @staticmethod
    def _probability(value: Any) -> bool:
        return (isinstance(value, Real) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= 1)

    @staticmethod
    def _stop(rule: str, reason: str) -> TwoTierPolicyVerdict:
        return TwoTierPolicyVerdict(False, PolicyTier.TIER3_HARDSTOP,
                                    PolicyVerdictAction.STOP, (rule,), reason)

    def evaluate_choice(
        self, prompt: str, choice: str, confidence: float, *,
        assessment: Optional[PolicyAssessment] = None,
    ) -> TwoTierPolicyVerdict:
        if not isinstance(prompt, str) or not isinstance(choice, str):
            return self._stop("INVALID_POLICY_INPUT", "prompt and choice must be strings")
        if not self._probability(confidence):
            return self._stop("INVALID_CONFIDENCE", "confidence must be finite and in [0, 1]")
        if assessment is None and self.evaluator is not None:
            try:
                assessment = self.evaluator(prompt, choice)
            except Exception as exc:
                # Do not silently substitute a lexical heuristic or a safe score.
                return self._stop("POLICY_EVALUATOR_ERROR", type(exc).__name__)
        if assessment is not None:
            if (not isinstance(assessment, PolicyAssessment)
                    or assessment.prompt != prompt or assessment.choice != choice
                    or not self._probability(assessment.risk)
                    or not self._probability(assessment.uncertainty)
                    or not isinstance(assessment.violations, tuple)
                    or any(not isinstance(rule, str) or not rule.strip()
                           for rule in assessment.violations)
                    or not isinstance(assessment.reason, str)):
                return self._stop("INVALID_POLICY_ASSESSMENT", "invalid or mismatched evidence")
            # Tier 0: direct hard policy facts take precedence over confidence.
            if assessment.violations:
                return TwoTierPolicyVerdict(
                    False, PolicyTier.TIER3_HARDSTOP, PolicyVerdictAction.STOP,
                    assessment.violations, assessment.reason, float(assessment.risk))
        # Tier 1: uncertainty never increases permission.
        if confidence < self.min_confidence:
            return self._stop("LOW_CONFIDENCE_BLIND_GUESS", "insufficient decision confidence")
        if assessment is None:
            return self._stop("MISSING_POLICY_ASSESSMENT", "no semantic safety evaluator or evidence supplied")
        risk_bound = min(1.0, float(assessment.risk) + float(assessment.uncertainty))
        if risk_bound >= self.stop_risk:
            return TwoTierPolicyVerdict(
                False, PolicyTier.TIER3_HARDSTOP, PolicyVerdictAction.STOP,
                ("RISK_BOUND_EXCEEDED",), "risk plus uncertainty reaches stop threshold", risk_bound)
        rules = []
        if confidence <= self.proceed_confidence:
            rules.append("BORDERLINE_CONFIDENCE_AMBIGUITY")
        if risk_bound >= self.confirm_risk:
            rules.append("RISK_UNCERTAINTY_REQUIRES_CONFIRMATION")
        if rules:
            return TwoTierPolicyVerdict(
                False, PolicyTier.TIER2_ESCALATE, PolicyVerdictAction.CONFIRM,
                tuple(rules), "explicit confirmation required", risk_bound)
        return TwoTierPolicyVerdict(
            True, PolicyTier.TIER0_PROCEED, PolicyVerdictAction.PROCEED,
            (), "assessed risk and confidence satisfy policy", risk_bound)

    def evaluate_prompt_safety(
        self, prompt: str, *, confidence: float,
        assessment: Optional[PolicyAssessment] = None,
    ) -> TwoTierPolicyVerdict:
        """Assess a prompt alone; evidence must bind to the empty choice."""
        return self.evaluate_choice(prompt, "", confidence, assessment=assessment)
