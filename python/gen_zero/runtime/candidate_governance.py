"""Gen-Zero Layer 0: Candidate Governance, Anti-Truncation Ranking & State Decoupling.

RFC Implementation for Issue #8 (Module 4):
1. Anti-Truncation Ranking:
   - Avoids naive positional truncation that silently drops critical bottom buttons (e.g. Next, Submit, Confirm).
   - Re-ranks candidates based on functional role weights and goal semantic relevance.
   - Explicitly emits clipped: true and total_candidates metadata.
2. State Feedback vs Affordance Decoupling:
   - Clearly separates instantaneous state registers (display value, focus, URL) from actionable candidates.
3. External Input Cleaning:
   - Cleans protocol headers and prompt injection boundaries before entering tokenizer.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..model.sanitization import sanitize_candidates, sanitize_state, sanitize_input_text


ROLE_WEIGHTS: Dict[str, float] = {
    "submit": 2.5,
    "confirm": 2.5,
    "checkout": 2.5,
    "pay": 2.5,
    "save": 2.0,
    "next": 2.0,
    "continue": 2.0,
    "button": 1.8,
    "input": 1.8,
    "select": 1.5,
    "link": 1.2,
    "menu": 1.1,
    "checkbox": 1.1,
    "radio": 1.1,
    "cancel": 1.0,
    "close": 1.0,
    "back": 1.0,
}


@dataclass
class GovernedCandidatePool:
    """Sanitized, pre-ranked candidate pool guaranteed against positional truncation bias."""
    candidates: List[str]
    clipped: bool
    total_candidates: int
    retained_count: int
    state_feedback: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": self.candidates,
            "clipped": self.clipped,
            "total_candidates": self.total_candidates,
            "retained_count": self.retained_count,
            "state_feedback": self.state_feedback,
            "metadata": self.metadata
        }


class AntiTruncationRanker:
    """Pre-ranks and selects candidate affordances to avoid blind positional truncation."""

    def __init__(self, max_candidates: int = 15):
        self.max_candidates = max_candidates

    def _score_candidate(self, candidate: str, goal_tokens: Set[str]) -> float:
        cand_lower = candidate.lower()
        cand_words = set(re.findall(r"\w+", cand_lower))
        score = 1.0

        # Role weight match (token-level & delimiter match prevents substring false positives e.g. 'display' vs 'pay')
        for role, weight in ROLE_WEIGHTS.items():
            if role in cand_words or f"-{role}" in cand_lower or f"_{role}" in cand_lower or f":{role}" in cand_lower:
                score += weight

        # Goal keyword relevance
        for tok in goal_tokens:
            if len(tok) > 2 and (tok in cand_words or tok in cand_lower):
                score += 1.5

        return score

    def govern_candidates(
        self,
        raw_candidates: List[str],
        state_feedback: Optional[Dict[str, Any]] = None,
        goal: Optional[str] = None
    ) -> GovernedCandidatePool:
        """Sanitizes, ranks, and truncates candidates while strictly tracking metadata.

        Guarantees that critical functional elements (e.g. 'Submit', 'Confirm', 'Next')
        at the end of a long list are not dropped due to naive sequential slicing.
        """
        # 1. Clean and sanitize all candidate strings
        clean_cands = sanitize_candidates(raw_candidates)
        total_count = len(clean_cands)

        # 2. Extract goal tokens
        goal_tokens = set(re.findall(r"\w+", (goal or "").lower()))

        # If candidates fit within window, preserve original order
        if total_count <= self.max_candidates:
            return GovernedCandidatePool(
                candidates=clean_cands,
                clipped=False,
                total_candidates=total_count,
                retained_count=total_count,
                state_feedback=sanitize_state(state_feedback or {}),
                metadata={"ranking_applied": False}
            )

        # 3. Candidates exceed budget: compute relevance and role scores
        scored: List[Tuple[float, int, str]] = []
        for orig_idx, c in enumerate(clean_cands):
            s = self._score_candidate(c, goal_tokens)
            # Tie-break with reverse index so equal score favors earlier items
            scored.append((s, -orig_idx, c))

        # Sort descending by score
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Take top max_candidates, but restore relative order for stable visual scanning
        selected_pairs = [(abs(orig_idx), c) for _, orig_idx, c in scored[:self.max_candidates]]
        selected_pairs.sort(key=lambda x: x[0])
        final_cands = [c for _, c in selected_pairs]

        return GovernedCandidatePool(
            candidates=final_cands,
            clipped=True,
            total_candidates=total_count,
            retained_count=len(final_cands),
            state_feedback=sanitize_state(state_feedback or {}),
            metadata={
                "ranking_applied": True,
                "dropped_count": total_count - len(final_cands)
            }
        )
