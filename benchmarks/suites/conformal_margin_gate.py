"""Conformal margin abstain gate for binary safety verdicts (Aegis 2.0 / Spec 28 Tier-2 hand-off).

Input is the model's signed margin m = logit(unsafe) - logit(safe). The gate has three outcomes:

    |m| < theta            -> ABSTAIN   (label "TIER2_ESCALATE": goes to human review, never served)
    m >= theta             -> BLOCK     (unsafe)
    m <= -theta            -> ALLOW     (safe)

Fail-closed rules (nothing here degrades silently):
  * ABSTAIN is never a release. `GateDecision.released` is True only for ALLOW.
  * A caller-supplied `high_risk=True` flag (crime planning / violence / weapons, decided by an upstream
    category detector, never by gold labels) turns the uncertain band into BLOCK, still escalated.
  * m == 0 with theta == 0 is a tie and goes to BLOCK.
  * Non-finite margin, bad theta, out-of-range confidence, or a configured confidence floor with no
    confidence supplied all raise ValueError. The gate never guesses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence

DEFAULT_THETA = 1.0

# Aegis 2.0 taxonomy names for the categories where a missed unsafe request is unacceptable.
HIGH_RISK_CATEGORIES = frozenset({"Criminal Planning/Confessions", "Violence", "Guns and Illegal Weapons"})


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class GateDecision:
    verdict: Verdict
    margin: float
    confidence: Optional[float]
    escalate: bool
    reason: str

    @property
    def released(self) -> bool:
        """Only a confident ALLOW lets content through."""
        return self.verdict is Verdict.ALLOW

    @property
    def label(self) -> str:
        return "TIER2_ESCALATE" if self.verdict is Verdict.ABSTAIN else self.verdict.value


def is_high_risk(categories: Iterable[str]) -> bool:
    return any(c in HIGH_RISK_CATEGORIES for c in categories)


def _finite(value, what: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a real number, got {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"{what} must be finite, got {x!r}")
    return x


class ConformalMarginGate:
    def __init__(self, theta: float = DEFAULT_THETA, min_confidence: Optional[float] = None):
        self.theta = _finite(theta, "theta")
        if self.theta < 0:
            raise ValueError(f"theta must be >= 0, got {self.theta}")
        self.min_confidence = None if min_confidence is None else _finite(min_confidence, "min_confidence")
        if self.min_confidence is not None and not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {self.min_confidence}")

    def decide(self, margin: float, confidence: Optional[float] = None, high_risk: bool = False) -> GateDecision:
        m = _finite(margin, "margin")
        conf = None
        if confidence is not None:
            conf = _finite(confidence, "confidence")
            if not 0.0 <= conf <= 1.0:
                raise ValueError(f"confidence must be in [0, 1], got {conf}")
        if self.min_confidence is not None and conf is None:
            raise ValueError("confidence is required when min_confidence is configured")

        low_conf = self.min_confidence is not None and conf < self.min_confidence
        if abs(m) < self.theta or low_conf:
            if high_risk:
                return GateDecision(Verdict.BLOCK, m, conf, True, "HIGH_RISK_FAIL_CLOSED")
            return GateDecision(Verdict.ABSTAIN, m, conf, True, "LOW_CONFIDENCE" if low_conf else "MARGIN_BAND")
        if m < 0:
            return GateDecision(Verdict.ALLOW, m, conf, False, "CONFIDENT_SAFE")
        return GateDecision(Verdict.BLOCK, m, conf, False, "TIE_FAIL_CLOSED" if m == 0 else "CONFIDENT_UNSAFE")

    def decide_batch(self, margins: Sequence[float], confidences: Optional[Sequence[float]] = None,
                     high_risk: Optional[Sequence[bool]] = None) -> List[GateDecision]:
        n = len(margins)
        for name, seq in (("confidences", confidences), ("high_risk", high_risk)):
            if seq is not None and len(seq) != n:
                raise ValueError(f"{name} has {len(seq)} entries for {n} margins")
        return [self.decide(margins[i],
                            None if confidences is None else confidences[i],
                            False if high_risk is None else bool(high_risk[i]))
                for i in range(n)]
