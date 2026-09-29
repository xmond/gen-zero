"""RAG Multi-Aspect Battery & Anti-Sycophancy Evidence Slotting.

Implements Module 2 (Part 1) of Issue #27:
- Evaluates 4 Noul probes per retrieved passage:
  1. contains_prompt_injection -> Drop immediately from prompt context.
  2. contradicts_query_premise -> Routes to <conflicting_evidence> to counter user misconception.
  3. is_relevant -> Discards irrelevant noise.
  4. contains_answer_evidence -> Routes to <accepted_evidence>.
- Anti-Sycophancy slotting: forces downstream models to confront factual conflicts instead of blind agreement.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import dataclasses
import time
import re
import numpy as np


@dataclasses.dataclass
class PassageAspectEvaluation:
    passage_id: str
    text: str
    is_relevant: float
    contains_answer_evidence: float
    contradicts_query_premise: float
    contains_prompt_injection: float
    assigned_slot: str  # "dropped", "conflicting", "accepted", "unrelated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "text": self.text,
            "is_relevant": round(self.is_relevant, 4),
            "contains_answer_evidence": round(self.contains_answer_evidence, 4),
            "contradicts_query_premise": round(self.contradicts_query_premise, 4),
            "contains_prompt_injection": round(self.contains_prompt_injection, 4),
            "assigned_slot": self.assigned_slot,
        }


@dataclasses.dataclass
class EvidenceSlottingResult:
    accepted_evidence: List[PassageAspectEvaluation]
    conflicting_evidence: List[PassageAspectEvaluation]
    dropped_injections: List[PassageAspectEvaluation]
    unrelated_passages: List[PassageAspectEvaluation]
    formatted_prompt_block: str
    total_evaluated: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_evidence": [p.to_dict() for p in self.accepted_evidence],
            "conflicting_evidence": [p.to_dict() for p in self.conflicting_evidence],
            "dropped_injections": [p.to_dict() for p in self.dropped_injections],
            "unrelated_passages": [p.to_dict() for p in self.unrelated_passages],
            "formatted_prompt_block": self.formatted_prompt_block,
            "total_evaluated": self.total_evaluated,
            "latency_ms": round(self.latency_ms, 2),
        }


class PassageMultiAspectBattery:
    """Evaluates multi-aspect quality probes on retrieved RAG passages."""

    def __init__(
        self,
        injection_threshold: float = 0.50,
        conflict_threshold: float = 0.60,
        relevance_threshold: float = 0.45,
    ):
        self.injection_threshold = injection_threshold
        self.conflict_threshold = conflict_threshold
        self.relevance_threshold = relevance_threshold

    def evaluate_passage(
        self,
        query: str,
        passage_id: str,
        passage_text: str,
    ) -> PassageAspectEvaluation:
        """Evaluates 4 Noul probes for a single passage against the query."""
        p_clean = (passage_text or "").strip()
        p_lower = p_clean.lower()
        q_lower = (query or "").strip().lower()

        # 1. contains_prompt_injection
        injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
            r"\b(system\s+override|dan\s+mode|jailbreak|pwned)\b",
            r"assistant:\s*i\s+am\s+free",
        ]
        has_inj = any(re.search(pat, p_lower) for pat in injection_patterns)
        p_inj = 0.98 if has_inj else 0.02

        # 2. contradicts_query_premise
        # Detect contradiction indicators like "deprecated", "removed", "false", "no longer", "contrary to"
        has_contradiction = False
        contradiction_cues = ["deprecated", "removed in", "does not support", "never existed", "contrary to popular belief", "no longer valid", "is false"]
        for cue in contradiction_cues:
            if cue in p_lower:
                # If query asks about that concept, it's conflicting evidence!
                has_contradiction = True
                break
        p_conf = 0.85 if has_contradiction else 0.10

        # 3. is_relevant & contains_answer_evidence
        q_words = set(re.findall(r"[a-z0-9]+", q_lower))
        p_words = set(re.findall(r"[a-z0-9]+", p_lower))

        stop_words = {"what", "are", "the", "a", "an", "in", "on", "of", "for", "to", "and", "is", "at", "with"}
        q_content = q_words - stop_words
        if not q_content:
            q_content = q_words

        overlap = len(q_content.intersection(p_words))
        rel_score = overlap / float(len(q_content) + 1e-8) if q_content else 0.0

        p_rel = float(np.clip(0.15 + 0.85 * rel_score, 0.05, 0.98))
        p_ans = float(np.clip(p_rel * 0.95, 0.05, 0.95))

        # Slot assignment
        if p_inj >= self.injection_threshold:
            slot = "dropped"
        elif p_conf >= self.conflict_threshold and (overlap > 0 or p_rel >= self.relevance_threshold):
            slot = "conflicting"
        elif p_rel >= self.relevance_threshold:
            slot = "accepted"
        else:
            slot = "unrelated"

        return PassageAspectEvaluation(
            passage_id=passage_id,
            text=p_clean,
            is_relevant=p_rel,
            contains_answer_evidence=p_ans,
            contradicts_query_premise=p_conf,
            contains_prompt_injection=p_inj,
            assigned_slot=slot,
        )

    def process_passages(
        self,
        query: str,
        passages: List[Tuple[str, str]],
    ) -> EvidenceSlottingResult:
        """Categorizes passages into structured slots and generates anti-sycophancy prompt."""
        t0 = time.perf_counter()

        accepted = []
        conflicting = []
        dropped = []
        unrelated = []

        for pid, txt in passages:
            eval_res = self.evaluate_passage(query, pid, txt)
            if eval_res.assigned_slot == "dropped":
                dropped.append(eval_res)
            elif eval_res.assigned_slot == "conflicting":
                conflicting.append(eval_res)
            elif eval_res.assigned_slot == "accepted":
                accepted.append(eval_res)
            else:
                unrelated.append(eval_res)

        # Build formatted anti-sycophancy prompt block
        prompt_parts = []
        if conflicting:
            prompt_parts.append("<conflicting_evidence>")
            prompt_parts.append("CRITICAL: The following verified facts contradict common assumptions or query premises. You must prioritize correcting the premise:")
            for item in conflicting:
                prompt_parts.append(f"[{item.passage_id}] {item.text}")
            prompt_parts.append("</conflicting_evidence>")

        if accepted:
            prompt_parts.append("<accepted_evidence>")
            for item in accepted:
                prompt_parts.append(f"[{item.passage_id}] {item.text}")
            prompt_parts.append("</accepted_evidence>")

        formatted_block = "\n".join(prompt_parts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return EvidenceSlottingResult(
            accepted_evidence=accepted,
            conflicting_evidence=conflicting,
            dropped_injections=dropped,
            unrelated_passages=unrelated,
            formatted_prompt_block=formatted_block,
            total_evaluated=len(passages),
            latency_ms=elapsed_ms,
        )
