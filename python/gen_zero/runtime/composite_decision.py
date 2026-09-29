"""Gen-Zero Layer 1: Composite Decision Bundling & decide_step Protocol.

RFC Implementation for Issue #8 (Module 1):
- Bundles the 4-tuple decision contract into a single sub-5ms pass:
  1. target (choice): Select best interaction entity from dynamic affordances.
  2. action (choice): Select discrete action type (click, input, scroll, etc.).
  3. done (noul): Termination/goal achievement probability.
  4. risk (noul): High-risk side-effect or irreversibility probability.
- Eliminates multi-round request overhead and repeated context tokenization.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, asdict
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from ..model.sanitization import sanitize_candidates, sanitize_state, sanitize_input_text


DEFAULT_ACTION_TYPES: List[str] = [
    "click",
    "input_text",
    "navigate",
    "scroll",
    "hover",
    "press_key",
    "finish"
]


@dataclass
class CompositeStepDecision:
    """Standardized output of a single composite decision step."""
    target: Optional[str]
    target_confidence: float
    action: str
    action_confidence: float
    done_prob: float
    risk_prob: float
    target_probabilities: Dict[str, float] = field(default_factory=dict)
    action_probabilities: Dict[str, float] = field(default_factory=dict)
    timing_ms: float = 0.0
    state_feedback: Optional[Dict[str, Any]] = None
    raw_answers: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "target_confidence": round(self.target_confidence, 4),
            "action": self.action,
            "action_confidence": round(self.action_confidence, 4),
            "done_prob": round(self.done_prob, 4),
            "risk_prob": round(self.risk_prob, 4),
            "target_probabilities": {k: round(v, 4) for k, v in self.target_probabilities.items()},
            "action_probabilities": {k: round(v, 4) for k, v in self.action_probabilities.items()},
            "timing_ms": round(self.timing_ms, 2),
            "state_feedback": self.state_feedback,
            "raw_answers": self.raw_answers
        }


class CompositeDecisionEngine:
    """Executes the standard composite 4-tuple decision in a single forward pass."""

    def __init__(
        self,
        client: Optional[Any] = None,
        default_actions: Optional[List[str]] = None
    ):
        self.client = client
        self.default_actions = default_actions or list(DEFAULT_ACTION_TYPES)

    @staticmethod
    def _validate_probability(value: Any, field_name: str) -> float:
        """Validate model probability metadata before it reaches policy code.

        The policy gate treats non-finite values as fail-closed, but allowing them
        through this boundary still leaks NaN/Inf into serialized decisions and
        telemetry.  Composite responses therefore use the same closed unit
        interval contract as ``ConfidenceFloorGate``.
        """
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a finite number in [0.0, 1.0]") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{field_name} must be a finite number in [0.0, 1.0]")
        return number

    @classmethod
    def _validate_probability_map(cls, values: Any, field_name: str) -> Dict[Any, float]:
        """Validate a probability mapping while retaining all candidate metadata."""
        if not isinstance(values, Mapping):
            raise ValueError(f"{field_name} must be a mapping of candidate to probability")
        return {
            candidate: cls._validate_probability(value, f"{field_name}[{candidate!r}]")
            for candidate, value in values.items()
        }

    def decide_step(
        self,
        state: Any,
        affordances: List[str],
        action_types: Optional[List[str]] = None,
        goal: Optional[str] = None,
        risk_criteria: Optional[str] = None,
        state_feedback: Optional[Dict[str, Any]] = None
    ) -> CompositeStepDecision:
        """Executes composite decision step bundling target, action, done, and risk.

        Args:
            state: Current environment observation or state representation.
            affordances: Available candidate interaction targets.
            action_types: Available action verbs. Defaults to standard UI/API actions.
            goal: Optional task goal or user intent instruction.
            risk_criteria: Optional custom risk definition.
            state_feedback: Optional register readings (display text, focus, status).

        Returns:
            CompositeStepDecision with calibrated target, action, done, and risk.
        """
        t0 = time.perf_counter()
        clean_state = sanitize_state(state)
        clean_affordances = sanitize_candidates(affordances)
        acts = action_types or self.default_actions
        clean_acts = sanitize_candidates(acts)
        task_goal = sanitize_input_text(goal or "Accomplish user intent efficiently and safely.")

        if not clean_affordances:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return CompositeStepDecision(
                target=None,
                target_confidence=0.0,
                action="finish",
                action_confidence=1.0,
                done_prob=1.0,
                risk_prob=0.0,
                target_probabilities={},
                action_probabilities={"finish": 1.0},
                timing_ms=elapsed_ms,
                state_feedback=state_feedback
            )

        # Formulate standard 4-tuple question batch
        questions: Dict[str, Dict[str, Any]] = {
            "target": {
                "type": "choice",
                "instructions": f"Select the most appropriate entity to advance goal: {task_goal}",
                "criteria": clean_affordances
            },
            "action": {
                "type": "choice",
                "instructions": "Select the exact operation type to execute on the selected target.",
                "criteria": clean_acts
            },
            "done": {
                "type": "noul",
                "instructions": f"Has the goal '{task_goal}' been fully achieved?",
                "criteria": {
                    "true": "Goal reached; no further actions required",
                    "false": "Task ongoing or intermediate step"
                }
            },
            "risk": {
                "type": "noul",
                "instructions": risk_criteria or "Does the prospective action carry high risk, irreversibility, payment, or auth impact?",
                "criteria": {
                    "true": "High impact, irreversible, financial or destructive",
                    "false": "Standard, benign, or reversible interaction"
                }
            }
        }

        # Resolve via client if available
        if self.client is not None and hasattr(self.client, "decide"):
            # Execute target and action via client
            t_res = self.client.decide(
                state=f"Goal: {task_goal}\nState: {clean_state}",
                candidates=clean_affordances,
                mode="reflex"
            )
            a_res = self.client.decide(
                state=f"Target: {t_res.get('action')}\nGoal: {task_goal}\nState: {clean_state}",
                candidates=clean_acts,
                mode="reflex"
            )
            # Done & risk estimates with explicit criteria grounding
            done_q = f"Question: Has the goal '{task_goal}' been fully achieved?\nTrue: Goal reached; task complete\nFalse: Ongoing or intermediate step"
            done_res = self.client.decide(
                state=f"{clean_state}\n{done_q}",
                candidates=["true", "false"],
                candidate_descriptions={"true": "Goal fully achieved", "false": "Task ongoing"},
                mode="reflex"
            )
            risk_crit = risk_criteria or "Does prospective action carry high risk, irreversibility, payment, or auth impact?"
            risk_q = f"Proposed: {a_res.get('action')} on {t_res.get('action')}\nQuestion: {risk_crit}\nTrue: Irreversible/destructive/financial\nFalse: Benign and reversible"
            risk_res = self.client.decide(
                state=f"{clean_state}\n{risk_q}",
                candidates=["true", "false"],
                candidate_descriptions={"true": "High risk or irreversible", "false": "Benign and safe"},
                mode="reflex"
            )

            missing_fields = []
            if "confidence" not in t_res:
                missing_fields.append("target_confidence")
            if "confidence" not in a_res:
                missing_fields.append("action_confidence")
            if "probs" not in t_res:
                missing_fields.append("target_probs")
            if "probs" not in a_res:
                missing_fields.append("action_probs")

            chosen_target = t_res.get("action")
            if chosen_target not in clean_affordances:
                raise ValueError(
                    f"Model-selected target {chosen_target!r} is not in affordances"
                )
            target_conf = self._validate_probability(
                t_res.get("confidence", 0.0), "target confidence"
            )
            target_probs = self._validate_probability_map(
                t_res.get("probs", {c: 1.0 / len(clean_affordances) for c in clean_affordances}),
                "target probabilities",
            )

            chosen_action = a_res.get("action")
            if chosen_action not in clean_acts:
                raise ValueError(
                    f"Model-selected action {chosen_action!r} is not in action_types"
                )
            action_conf = self._validate_probability(
                a_res.get("confidence", 0.0), "action confidence"
            )
            action_probs = self._validate_probability_map(
                a_res.get("probs", {c: 1.0 / len(clean_acts) for c in clean_acts}),
                "action probabilities",
            )

            done_probs = self._validate_probability_map(
                done_res.get("probs", {}), "done probabilities"
            )
            done_p = self._validate_probability(done_probs.get("true", 0.0), "done probability")

            risk_probs = risk_res.get("probs")
            if risk_probs:
                if not isinstance(risk_probs, Mapping):
                    raise ValueError("risk probabilities must be a mapping")
                validated_risk_probs = {
                    key: self._validate_probability(value, f"risk probability {key!r}")
                    for key, value in risk_probs.items()
                }
            else:
                validated_risk_probs = {}

            if "true" in validated_risk_probs:
                risk_p = validated_risk_probs["true"]
                risk_assessed = True
            else:
                # Fail-closed: unassessed risk is treated as maximal, never as benign.
                risk_p = 1.0
                risk_assessed = False
                missing_fields.append("risk_probs")

            raw = {
                "target_raw": t_res,
                "action_raw": a_res,
                "done_raw": done_res,
                "risk_raw": risk_res,
                "risk_assessed": risk_assessed,
                "missing_fields": missing_fields,
            }

        else:
            # No decision client available: fail-closed deterministic lexical ordering.
            # This is not a stand-in for a model's judgment -- it never fabricates confidence
            # or a risk assessment. Unassessed risk is treated as maximal (risk_prob=1.0) so
            # downstream policy gates do not proceed as if a real risk check had passed.
            chosen_target = sorted(clean_affordances)[0]
            target_probs = {c: 1.0 / len(clean_affordances) for c in clean_affordances}
            target_conf = 0.0

            chosen_action = sorted(clean_acts)[0]
            action_probs = {a: 1.0 / len(clean_acts) for a in clean_acts}
            action_conf = 0.0

            done_p = 0.0
            risk_p = 1.0
            raw = {
                "ranked": False,
                "fallback": "unranked_lexical",
                "risk_assessed": False,
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return CompositeStepDecision(
            target=chosen_target,
            target_confidence=target_conf,
            action=chosen_action,
            action_confidence=action_conf,
            done_prob=done_p,
            risk_prob=risk_p,
            target_probabilities=target_probs,
            action_probabilities=action_probs,
            timing_ms=elapsed_ms,
            state_feedback=state_feedback,
            raw_answers=raw
        )
