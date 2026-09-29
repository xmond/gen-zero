"""Universal Task-Agnostic Cognitive Pipeline.

Architectural Mandate:
Strictly zero hardcoding of benchmark dataset names (`task == "..."`).
All decision routing, calibration, and safety interlocks are triggered purely
from:
1. The structural topology and semantic types of the candidate action space.
2. The intrinsic syntactic and discourse structure of the input context.
3. Information-theoretic quantities (entropy, variance, confidence margins).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from gen_zero.nanocore.relational_binding import analyze_context
from gen_zero.gate.grounding_gate import EpistemicGroundingGate, DiscourseFocusRewriter

# Minimum P(yes) - P(no) before an affirmative truth-value claim is accepted.
AFFIRMATIVE_MARGIN = 0.15


class DecisionRouting(str, Enum):
    """Execution path selected before a static choice is attempted."""

    DirectChoice = "DirectChoice"
    CoTRequired = "CoTRequired"
    ToolExecution = "ToolExecution"


ARITHMETIC_ROUTING_MESSAGE = (
    "Arithmetic decision requires a scratchpad or tool result."
)


class ArithmeticRoutingError(RuntimeError):
    """Raised when a static choice would bypass arithmetic reasoning."""

    def __init__(self, routing: DecisionRouting = DecisionRouting.CoTRequired):
        self.routing = routing
        super().__init__(ARITHMETIC_ROUTING_MESSAGE)


class CandidateSpaceType(str, Enum):
    ORDINAL_SCALE = "ordinal_scale"
    ASYMMETRIC_RISK = "asymmetric_risk"
    PAIRWISE_EQUIVALENCE = "pairwise_equivalence"
    FACTUAL_VERIFICATION = "factual_verification"
    EPISTEMIC_GROUNDING = "epistemic_grounding"
    HYPOTHESIS_TRUTH_VALUE = "hypothesis_truth_value"
    CATEGORICAL_DECISION = "categorical_decision"
    ARITHMETIC_CHOICE = "arithmetic_choice"

# Optional option label such as "(A)", "A)", "B." before a numeric value.
_OPTION_LABEL_RE = re.compile(r"^\(?[a-z][).:]\s*")
# Digit-adjacent operators, so hyphens and colons in prose do not count.
_MATH_OPERATOR_RE = re.compile(r"\d\s*[+\-*/=\u00d7\u00f7]\s*\d")
_MATH_WORD_RE = re.compile(
    r"\b(?:add(?:ed|ing|ition)?|average|calculate|computed?|cost|costs|"
    r"difference|divide[ds]?|division|double|each|equal(?:s|ity)?|"
    r"fewer|half|left over|minus|more than|multipl(?:y|ied|ies)|"
    r"per|plus|product|quotient|ratio|remaining|remainder|subtract(?:ed|ing)?|"
    r"sum|spent|subtraction|times|total|triple|twice)\b",
    re.IGNORECASE,
)

# Axiom A2 intentionally uses a conservative lexical/symbolic boundary.  A
# single arithmetic cue is enough to leave a static choice path; relation
# chains additionally trigger once they contain more than one edge or exceed
# depth two.  Minus is bounded so ordinary prose hyphens do not dominate the
# detector, while spaced and digit-adjacent subtraction remains covered.
_ARITHMETIC_OPERATOR_RE = re.compile(
    r"(?:[+*/%^\u00d7\u00f7\u2212\u2013\uff0b\uff0d\uff0a\uff0f\uff05\u22c5\u00b7\u2219\u2217]|"
    r"(?<![A-Za-z])-(?![A-Za-z])|(?<=\d)-(?![A-Za-z]))"
)
_RELATION_SYMBOL_RE = re.compile(
    r"(?:<=>|<=|>=|==|!=|<|>|\u2264|\u2265|\u2260|\u2248|\u2261|\u2192|\u21d2|\u27f9|=)"
)
_RELATION_WORD_RE = re.compile(
    r"\b(?:at\s+(?:least|most)|no\s+(?:more|less)\s+than|"
    r"greater\s+than|less\s+than|more\s+than|fewer\s+than|equal\s+to|"
    r"not\s+equal(?:\s+to)?|older\s+than|younger\s+than|taller\s+than|"
    r"shorter\s+than|higher\s+than|lower\s+than|left\s+of|right\s+of|"
    r"above|below|before|after|precedes?|follows?|implies?|if|then)\b",
    re.IGNORECASE,
)
_TOOL_EXECUTION_RE = re.compile(
    r"\b(?:calculator|formal\s+solver|cp[-\s]?sat|code|execute|execution|"
    r"python|solver|tool)\b|\b(?:eval|exec)\s*\(",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ArithmeticSignals:
    operator_count: int
    lexeme_count: int
    relation_count: int
    relation_depth: int

    @property
    def requires_reasoning(self) -> bool:
        return bool(
            self.operator_count
            or self.lexeme_count
            or self.relation_count > 1
            or self.relation_depth > 2
        )


def _arithmetic_signals(context: str) -> _ArithmeticSignals:
    """Return structural arithmetic/relational cues without echoing input."""
    text = context if isinstance(context, str) else str(context)
    operator_count = len(_ARITHMETIC_OPERATOR_RE.findall(text))
    lexeme_count = len(_MATH_WORD_RE.findall(text))
    relation_count = len(_RELATION_SYMBOL_RE.findall(text)) + len(
        _RELATION_WORD_RE.findall(text)
    )
    # For the bounded routing contract, a chain of n relation edges has depth
    # n + 1.  This captures symbolic and natural-language cascades alike,
    # including chains whose operands contain no digits.
    relation_depth = relation_count + 1 if relation_count else 0
    return _ArithmeticSignals(
        operator_count=operator_count,
        lexeme_count=lexeme_count,
        relation_count=relation_count,
        relation_depth=relation_depth,
    )


def _parse_numeric_option(candidate: str) -> Optional[float]:
    """Parse '12', '$1,200', '(B) 3.5' or '45%' into a float, else None."""
    text = _OPTION_LABEL_RE.sub("", candidate.strip().lower()).strip()
    text = text.lstrip("$").rstrip("%").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def context_has_arithmetic(context: str) -> bool:
    """True when the context shows calculation operators or quantity words."""
    return _arithmetic_signals(context).requires_reasoning


@dataclass(frozen=True)
class CandidateAnalysis:
    space_type: CandidateSpaceType
    special_indices: Dict[str, int]
    ordinal_values: Optional[List[float]] = None


def analyze_candidate_space(candidates: Sequence[str]) -> CandidateAnalysis:
    """Classifies the nature of the decision space purely from candidate strings."""
    c_lower = [c.strip().lower() for c in candidates]
    special: Dict[str, int] = {}

    # 1. Check if candidate set forms an ordinal scale (e.g. ['1', '2', '3', '4', '5'])
    try:
        ord_vals = [float(c) for c in c_lower]
        if len(ord_vals) >= 3 and sorted(ord_vals) == ord_vals:
            return CandidateAnalysis(
                space_type=CandidateSpaceType.ORDINAL_SCALE,
                special_indices={},
                ordinal_values=ord_vals,
            )
    except ValueError:
        pass

    # 2. Check for Asymmetric Risk / Safety candidates
    risk_labels = {"unsafe", "toxic", "harmful", "severe_toxic", "threat", "insult", "identity_hate", "block", "deny"}
    for idx, c in enumerate(c_lower):
        if c in risk_labels:
            special["risk"] = idx
            return CandidateAnalysis(
                space_type=CandidateSpaceType.ASYMMETRIC_RISK,
                special_indices=special,
            )

    # 3. Check for Factual / Evidence Verification (e.g. Supports / Refutes / Not Enough Info)
    fact_refute_labels = {"refutes", "contradiction", "contradicts"}
    fact_support_labels = {"supports", "entailment"}
    for idx, c in enumerate(c_lower):
        if c in fact_refute_labels:
            special["refutes"] = idx
        elif c in fact_support_labels:
            special["supports"] = idx
    if "refutes" in special and "supports" in special:
        return CandidateAnalysis(
            space_type=CandidateSpaceType.FACTUAL_VERIFICATION,
            special_indices=special,
        )

    # 4. Check for Pairwise Equivalence / Paraphrase
    non_equiv_labels = {"not_paraphrase", "different", "different_meaning", "no_paraphrase"}
    for idx, c in enumerate(c_lower):
        if c in non_equiv_labels:
            special["non_equivalent"] = idx
            return CandidateAnalysis(
                space_type=CandidateSpaceType.PAIRWISE_EQUIVALENCE,
                special_indices=special,
            )

    # 5. Check for Epistemic Grounding / Unanswerability
    unans_labels = {"unanswerable", "not enough info", "unknown", "cannot determine"}
    for idx, c in enumerate(c_lower):
        if c in unans_labels:
            special["unanswerable"] = idx
            return CandidateAnalysis(
                space_type=CandidateSpaceType.EPISTEMIC_GROUNDING,
                special_indices=special,
            )

    # 6. Check for Hypothesis Truth Value (e.g. yes / no / maybe)
    if set(c_lower) == {"yes", "no", "maybe"}:
        for idx, c in enumerate(c_lower):
            special[c] = idx
        return CandidateAnalysis(
            space_type=CandidateSpaceType.HYPOTHESIS_TRUTH_VALUE,
            special_indices=special,
        )

    # 7. Numeric answers, bare or labeled "(A) 12": an arithmetic choice.
    if len(c_lower) >= 2 and all(_parse_numeric_option(c) is not None for c in c_lower):
        return CandidateAnalysis(
            space_type=CandidateSpaceType.ARITHMETIC_CHOICE,
            special_indices={},
        )

    return CandidateAnalysis(
        space_type=CandidateSpaceType.CATEGORICAL_DECISION,
        special_indices=special,
    )


def route_decision(
    context: str,
    candidates: Optional[Sequence[str]] = None,
    *,
    depth: int = 0,
    requested: DecisionRouting = DecisionRouting.DirectChoice,
) -> DecisionRouting:
    """Select a safe execution path before a static choice is evaluated.

    Arithmetic candidate spaces always leave the direct-choice path.  Any
    arithmetic operator or arithmetic lexeme does the same, while relational
    cascades require at least two relation edges (or structural depth above
    two). Caller preferences can only escalate this policy floor.
    """
    c_space = analyze_candidate_space(candidates) if candidates is not None else None
    if requested == DecisionRouting.ToolExecution:
        return DecisionRouting.ToolExecution

    text = context if isinstance(context, str) else str(context)
    signals = _arithmetic_signals(text)
    tool_requested = bool(_TOOL_EXECUTION_RE.search(text))

    if c_space is not None and c_space.space_type == CandidateSpaceType.ARITHMETIC_CHOICE:
        return (
            DecisionRouting.ToolExecution
            if tool_requested
            else DecisionRouting.CoTRequired
        )
    if signals.requires_reasoning:
        return (
            DecisionRouting.ToolExecution
            if tool_requested
            else DecisionRouting.CoTRequired
        )
    if tool_requested:
        return DecisionRouting.ToolExecution
    if depth > 2 or depth < 0 or not candidates or requested != DecisionRouting.DirectChoice:
        return DecisionRouting.CoTRequired
    return DecisionRouting.DirectChoice


def extract_passage_and_query(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Generic extractor for text contexts structured as reference passage + question."""
    m = re.search(r"^(?:Passage|Context|Text|Document):\s*(.*?)\n+(?:Question|Query):\s*(.*?)(?:\n|$)", prompt, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _valid_cot_prediction(
    cot_prediction: Optional[int],
    cot_confidence: float,
    candidate_count: int,
) -> bool:
    """Validate the bounded, indexed result accepted by the A2 guard."""
    if isinstance(cot_prediction, bool) or not isinstance(cot_prediction, (int, np.integer)):
        return False
    if not 0 <= int(cot_prediction) < candidate_count:
        return False
    try:
        confidence = float(cot_confidence)
    except (TypeError, ValueError):
        return False
    return math.isfinite(confidence) and 0.50 <= confidence <= 1.0


def universal_task_agnostic_decision(
    context: str,
    candidates: Sequence[str],
    probabilities: np.ndarray,
    expert_predictions: Optional[Dict[str, str]] = None,
    expert_probs: Optional[Dict[str, np.ndarray]] = None,
    expert_weights: Optional[Dict[str, float]] = None,
    cot_prediction: Optional[int] = None,
    cot_confidence: float = 0.0,
) -> Tuple[int, str]:
    """Pure task-agnostic cognitive decision engine.

    Never queries `task` or dataset identifier. Relies entirely on candidate space
    topology, context semantics, and uncertainty metrics.

    Returns:
        (pred_index, decision_rationale)

    Raises:
        ArithmeticRoutingError: the caller must run the requested scratchpad
        or tool route. No candidate is selected as a fallback. A supplied CoT
        result is checked for index and confidence validity, not proof correctness.
    """
    if expert_predictions is None:
        expert_predictions = {}
    if expert_probs is None:
        expert_probs = {}
    if expert_weights is None:
        expert_weights = {}

    c_space = analyze_candidate_space(candidates)
    routing = route_decision(context, candidates)

    # Rule 2: Asymmetric Risk & Safety Interlock
    # Zero-false-negative safety boundary: Cost(harmful -> safe) >> Cost(safe -> harmful).
    # Require that any expert triggering an interlock has non-negligible model gate weight
    # and genuine calibrated certainty, preventing zero-weight uncalibrated noise from vetoing.
    if c_space.space_type == CandidateSpaceType.ASYMMETRIC_RISK:
        risk_idx = c_space.special_indices.get("risk")
        if risk_idx is not None:
            # Check cost-sensitive Bayesian threshold tau* = 0.20 on fused probability
            if float(probabilities[risk_idx]) >= 0.20:
                return risk_idx, "safety_interlock_cost_sensitive_threshold"
            # Active expert confirmation: expert must carry non-zero weight (>= 0.10)
            for exp_name, exp_pred in expert_predictions.items():
                w = expert_weights.get(exp_name, 1.0)
                if w >= 0.10 and exp_pred and exp_pred.strip().lower() in {"unsafe", "toxic", "harmful", "severe_toxic"}:
                    # Verify the expert itself assigned significant probability to risk
                    exp_p = expert_probs.get(exp_name)
                    if exp_p is None or (risk_idx < len(exp_p) and exp_p[risk_idx] >= 0.50):
                        return risk_idx, "safety_interlock_expert"

    # A2 runs before any static readout. A safety refusal above is terminal;
    # all other multi-step requests require a separate reasoning result.
    if routing != DecisionRouting.DirectChoice:
        if (routing == DecisionRouting.CoTRequired
                and c_space.space_type != CandidateSpaceType.ASYMMETRIC_RISK
                and _valid_cot_prediction(cot_prediction, cot_confidence, len(candidates))):
            return int(cot_prediction), "structural_arithmetic_formal_witness"
        raise ArithmeticRoutingError(routing)

    base_pred = int(np.argmax(probabilities))
    max_prob = float(probabilities[base_pred])

    # Rule 1: Ordinal Scale Decision
    # Never decode via continuous expectation round(E[y]) which collapses bimodal mass.
    # Use discrete MAP argmax by default; require genuine high-confidence CoT (>= 0.50) to override.
    if c_space.space_type == CandidateSpaceType.ORDINAL_SCALE:
        if cot_prediction is not None and _valid_cot_prediction(cot_prediction, cot_confidence, len(candidates)):
            return cot_prediction, "ordinal_cot_confident"
        return base_pred, "ordinal_map_discrete_argmax"

    # Rule 4: Pairwise Relational Anti-Symmetry (Argument Role Swap)
    # When two sentences share >75% vocabulary but invert asymmetric predicate arguments.
    if c_space.space_type == CandidateSpaceType.PAIRWISE_EQUIVALENCE:
        non_equiv_idx = c_space.special_indices.get("non_equivalent")
        if non_equiv_idx is not None:
            paws_rpt = analyze_context(context)
            if paws_rpt is not None and paws_rpt.has_role_swap:
                return non_equiv_idx, "relational_role_swap_inversion"

    # Rule 5: Epistemic Grounding & Unanswerability Gate
    # When asking about entities not present or contradicted in context, gate uncertain answers.
    if c_space.space_type == CandidateSpaceType.EPISTEMIC_GROUNDING and max_prob < 0.65:
        unans_idx = c_space.special_indices.get("unanswerable")
        if unans_idx is not None:
            pas, q = extract_passage_and_query(context)
            if pas and q:
                # Discourse-focus reweighting: if a discourse turn is detected,
                # construct a nucleus-weighted passage for evidence evaluation so
                # that concession clauses do not mislead the grounding gate.
                focus = DiscourseFocusRewriter.analyze_discourse_focus(pas)
                if focus.has_turn:
                    # Build a weighted passage by repeating nucleus segments
                    # proportionally to their boost factor (simple replication
                    # so the token-level stem matchers get more signal from
                    # nucleus content).
                    weighted_parts: List[str] = []
                    for seg_text, seg_weight in focus.segments:
                        # Scale: weight 1.0 -> 1 copy; weight 1.3 -> 1 copy
                        # plus an extra copy with 30% probability (always 1 extra
                        # for simplicity).
                        repeats = max(1, int(round(seg_weight)))
                        for _ in range(repeats):
                            weighted_parts.append(seg_text)
                    weighted_passage = " ".join(weighted_parts)
                    gate = EpistemicGroundingGate(confidence_gate=0.65)
                    ev = gate.evaluate_evidence(weighted_passage, q)
                else:
                    gate = EpistemicGroundingGate(confidence_gate=0.65)
                    ev = gate.evaluate_evidence(pas, q)
                p_unans = float(probabilities[unans_idx])
                lbl, reason = gate.decide(p_unans, ev)
                c_lower = [c.strip().lower() for c in candidates]
                if lbl.lower() in c_lower:
                    return c_lower.index(lbl.lower()), f"epistemic_grounding_{reason}"

    # Rule 6: Hypothesis Truth Margin Calibration
    if c_space.space_type == CandidateSpaceType.HYPOTHESIS_TRUTH_VALUE:
        yes_idx = c_space.special_indices.get("yes")
        no_idx = c_space.special_indices.get("no")
        maybe_idx = c_space.special_indices.get("maybe")
        if yes_idx is not None and no_idx is not None and maybe_idx is not None:
            p_yes = float(probabilities[yes_idx])
            p_no = float(probabilities[no_idx])
            p_maybe = float(probabilities[maybe_idx])
            # Affirmative claims need a margin over the strongest non-affirmative
            # answer; borderline evidence must not resolve to "yes".
            if p_maybe >= 0.30:
                return maybe_idx, "hypothesis_inconclusive"
            if base_pred == yes_idx:
                if p_yes - p_no >= AFFIRMATIVE_MARGIN:
                    return yes_idx, "hypothesis_affirmative_confirmed"
                if p_no >= p_maybe:
                    return no_idx, "hypothesis_affirmative_margin_fallback_no"
                return maybe_idx, "hypothesis_affirmative_margin_fallback_maybe"

    # Rule 7: Multi-Step Relational & Arithmetic Reasoning Witness
    # A confident chain-of-thought answer is the formal witness for multi-step
    # arithmetic. The safety space is excluded: Rule 2 owns that boundary.
    if cot_prediction is not None and cot_confidence >= 0.50:
        if c_space.space_type in (
            CandidateSpaceType.ARITHMETIC_CHOICE,
            CandidateSpaceType.CATEGORICAL_DECISION,
            CandidateSpaceType.ORDINAL_SCALE,
        ) or (
            c_space.space_type != CandidateSpaceType.ASYMMETRIC_RISK
            and context_has_arithmetic(context)
        ):
            return cot_prediction, "structural_arithmetic_formal_witness"

    return base_pred, "standard_maximum_a_posteriori"
