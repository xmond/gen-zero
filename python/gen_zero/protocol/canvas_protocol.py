"""Canvas Slot Masking Protocol (Issue #32 & RFC-032).

Defines the unified Canvas Template specification:
1. Structured placeholder schema: <|decision_canvas|> ... @{slot_spec} ... <|end_canvas|>
2. Slot typing: CHOICE (discrete set), NOUL (continuous sigmoid probability in [0, 1]),
   and SCORE (continuous scalar rating).
3. Zero-decoding fixed-token alignment for single-pass multi-slot prefill.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class CanvasSlotType(str, Enum):
    """Supported semantic slot types in the Canvas Slot Masking specification."""
    CHOICE = "CHOICE"  # Discrete multi-choice categorical distribution
    NOUL = "NOUL"      # Binary / continuous sigmoid probability in [0, 1]
    SCORE = "SCORE"    # Ordered / bounded scalar rating in [min_val, max_val]


@dataclass
class CanvasSlotSpec:
    """Specification of an individual decision placeholder slot in the Canvas."""
    name: str
    slot_type: CanvasSlotType
    candidates: Optional[List[str]] = None
    description: str = ""
    min_val: float = 0.0
    max_val: float = 1.0

    @property
    def placeholder(self) -> str:
        suffix = self.slot_type.value.lower()
        return f"@{{{self.name}_{suffix}}}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "slot_type": self.slot_type.value,
            "candidates": self.candidates,
            "description": self.description,
            "placeholder": self.placeholder,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }


@dataclass
class SlotResult:
    """Evaluated decision result for a single Canvas slot."""
    name: str
    slot_type: CanvasSlotType
    chosen_value: Any
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    entropy: float = 0.0
    mean_confidence: float = 0.0
    std_error: float = 0.0
    multi_read: bool = False
    reads_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "slot_type": self.slot_type.value,
            "chosen_value": self.chosen_value,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "entropy": round(float(self.entropy), 4),
            "mean_confidence": round(float(self.mean_confidence), 4),
            "std_error": round(float(self.std_error), 4),
            "error_bar": f"{round(self.mean_confidence, 4)} ± {round(self.std_error, 4)}",
            "multi_read": self.multi_read,
            "reads_count": self.reads_count,
        }


@dataclass
class CanvasDecisionResult:
    """Overall multi-slot structured result produced by single prefill Canvas execution."""
    slots: Dict[str, SlotResult]
    is_multi_read: bool = False
    max_entropy: float = 0.0
    timing_ms: float = 0.0
    prompt_tokens: int = 0
    raw_reads: List[Dict[str, Any]] = field(default_factory=list)

    def get_slot(self, name: str) -> Optional[SlotResult]:
        return self.slots.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "is_multi_read": self.is_multi_read,
            "max_entropy": round(float(self.max_entropy), 4),
            "timing_ms": round(float(self.timing_ms), 2),
            "prompt_tokens": self.prompt_tokens,
            "reads_executed": len(self.raw_reads) if self.raw_reads else 1,
        }


class CanvasTemplate:
    """Parser, formatter, and coordinator for the Canvas Slot Masking Protocol."""

    HEADER_TAG = "<|decision_canvas|>"
    FOOTER_TAG = "<|end_canvas|>"
    SLOT_REGEX = re.compile(r'([a-zA-Z0-9_\-]+)\s*:\s*@\{([a-zA-Z0-9_\-]+)\}')

    def __init__(self, slots: Optional[Sequence[CanvasSlotSpec]] = None):
        self.slots: List[CanvasSlotSpec] = list(slots) if slots else []
        self._slot_map: Dict[str, CanvasSlotSpec] = {s.name: s for s in self.slots}

    def add_slot(self, slot: CanvasSlotSpec) -> None:
        self.slots.append(slot)
        self._slot_map[slot.name] = slot

    def get_slot(self, name: str) -> Optional[CanvasSlotSpec]:
        return self._slot_map.get(name)

    def get_slot_names(self) -> List[str]:
        return [s.name for s in self.slots]

    def format_prompt(self, context_state: str) -> str:
        """Formats context state and appends standardized Canvas Template block."""
        canvas_lines = [self.HEADER_TAG]
        for slot in self.slots:
            canvas_lines.append(f"{slot.name}: {slot.placeholder}")
        canvas_lines.append(self.FOOTER_TAG)
        canvas_block = "\n".join(canvas_lines)

        state_clean = context_state.strip()
        return f"{state_clean}\n\n[Canvas Template]\n{canvas_block}"

    @classmethod
    def parse_canvas_text(cls, canvas_text: str) -> List[CanvasSlotSpec]:
        """Parses a Canvas text block into a list of CanvasSlotSpecs."""
        specs: List[CanvasSlotSpec] = []
        for match in cls.SLOT_REGEX.finditer(canvas_text):
            slot_name = match.group(1).strip()
            placeholder_body = match.group(2).strip().lower()

            if "noul" in placeholder_body or "prob" in placeholder_body:
                slot_type = CanvasSlotType.NOUL
                candidates = ["true", "false"]
            elif "score" in placeholder_body or "rate" in placeholder_body:
                slot_type = CanvasSlotType.SCORE
                candidates = None
            else:
                slot_type = CanvasSlotType.CHOICE
                candidates = None

            specs.append(CanvasSlotSpec(name=slot_name, slot_type=slot_type, candidates=candidates))
        return specs

    def find_slot_token_positions(self, token_sequence: Sequence[str]) -> Dict[str, int]:
        """Identifies exact token positions for zero-decoding fixed-token prefill reading."""
        positions: Dict[str, int] = {}
        for idx, token in enumerate(token_sequence):
            clean_tok = str(token).strip()
            for slot in self.slots:
                # Matches "@" or placeholder token fragments
                if slot.name not in positions:
                    if slot.placeholder in clean_tok or f"@{slot.name}" in clean_tok or f"@{slot.placeholder[2:-1]}" in clean_tok:
                        positions[slot.name] = idx
                    elif clean_tok.startswith("@") and slot.name in clean_tok:
                        positions[slot.name] = idx
        return positions

    @classmethod
    def standard_4tuple(
        cls,
        affordances: Sequence[str],
        action_types: Optional[Sequence[str]] = None,
        risk_criteria: Optional[str] = None
    ) -> "CanvasTemplate":
        """Factory creating the standardized 4-Tuple Decision Canvas (action, target, done, risk)."""
        default_actions = list(action_types) if action_types else [
            "click", "input_text", "navigate", "scroll", "hover", "press_key", "finish"
        ]
        return cls([
            CanvasSlotSpec(
                name="action",
                slot_type=CanvasSlotType.CHOICE,
                candidates=list(default_actions),
                description="Select discrete action verb to execute"
            ),
            CanvasSlotSpec(
                name="target",
                slot_type=CanvasSlotType.CHOICE,
                candidates=list(affordances),
                description="Select interaction entity from dynamic affordances"
            ),
            CanvasSlotSpec(
                name="done",
                slot_type=CanvasSlotType.NOUL,
                candidates=["true", "false"],
                description="Termination or task accomplishment probability"
            ),
            CanvasSlotSpec(
                name="risk",
                slot_type=CanvasSlotType.SCORE,
                min_val=0.0,
                max_val=1.0,
                description=risk_criteria or "Continuous security risk / side-effect severity score"
            ),
        ])
