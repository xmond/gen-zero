"""Gen-Zero Input Boundary Sanitization & Forgery Protection.

Prevents prompt injection and option boundary forgery across untrusted inputs.
Escapes structural control sequences (e.g., <|fim_*|>, <|box_*|>, <|im_start|>)
into safe Unicode non-control variants (<¦...¦>) prior to tokenization.
Guarantees that only system-injected delimiters are parsed as internal control tokens.
"""

import re
from typing import Any, Dict, List, Set, Union, Tuple, Optional

# Regular expression matching any control token delimiter <|...|>
CONTROL_TOKEN_REGEX = re.compile(r"<\|([^|>]+)\|>")

# Known critical control tokens for explicit auditing
KNOWN_CRITICAL_CONTROL_TOKENS: Set[str] = {
    "fim_prefix", "fim_middle", "fim_suffix",
    "im_start", "im_end",
    "box_start", "box_end",
    "gen_zero_suffix", "option_start", "option_end",
    "endoftext", "pad", "system", "user", "assistant"
}


def escape_control_tokens(text: str) -> str:
    """Escapes structural control token markers <|...|> into safe Unicode <¦...¦>."""
    if not isinstance(text, str):
        return text
    return CONTROL_TOKEN_REGEX.sub(r"<¦\1¦>", text)


def unescape_control_tokens(text: str) -> str:
    """Restores previously escaped <¦...¦> to <|...|> (for internal system use only)."""
    if not isinstance(text, str):
        return text
    return re.sub(r"<¦([^¦>]+)¦>", r"<|\1|>", text)


def is_boundary_forgery_attempt(text: str) -> bool:
    """Detects whether raw untrusted text contains unauthorized control token markers."""
    if not isinstance(text, str):
        return False
    return bool(CONTROL_TOKEN_REGEX.search(text))


def audit_control_tokens(text: str) -> List[str]:
    """Returns a list of all raw control token names found in untrusted text."""
    if not isinstance(text, str):
        return []
    return CONTROL_TOKEN_REGEX.findall(text)


def sanitize_input_text(text: str) -> str:
    """Universal text sanitization preventing option boundary forgery."""
    return escape_control_tokens(text)


def sanitize_candidates(candidates: List[str]) -> List[str]:
    """Sanitizes a list of candidate strings."""
    return [escape_control_tokens(c) if isinstance(c, str) else c for c in candidates]


def sanitize_state(state: Any) -> Any:
    """Recursively traverses arbitrary state structures and escapes control tokens in all strings."""
    if state is None:
        return None
    if isinstance(state, str):
        return escape_control_tokens(state)
    if isinstance(state, dict):
        return {
            (escape_control_tokens(k) if isinstance(k, str) else k): sanitize_state(v)
            for k, v in state.items()
        }
    if isinstance(state, list):
        return [sanitize_state(elem) for elem in state]
    if isinstance(state, tuple):
        return tuple(sanitize_state(elem) for elem in state)
    if isinstance(state, set):
        return {sanitize_state(elem) for elem in state}
    return state


class BoundaryForgerySanitizer:
    """Stateful sanitizer for end-to-end request pipelines."""

    def __init__(self, reject_on_forgery: bool = False):
        self.reject_on_forgery = reject_on_forgery

    def process(
        self,
        state: Any,
        candidates: List[str],
        instruction: Optional[str] = None
    ) -> Tuple[Any, List[str], Dict[str, Any]]:
        """Sanitizes state, candidates, and optional instruction, returning audit metadata.

        Raises ValueError if reject_on_forgery is True and forgery tokens are detected.
        """
        raw_state_str = str(state)
        forgery_in_state = audit_control_tokens(raw_state_str)
        
        forgery_in_cands = []
        for c in candidates:
            if isinstance(c, str):
                forgery_in_cands.extend(audit_control_tokens(c))

        forgery_in_instr = audit_control_tokens(instruction) if instruction else []

        total_forgeries = forgery_in_state + forgery_in_cands + forgery_in_instr
        critical_forgeries = [t for t in total_forgeries if t in KNOWN_CRITICAL_CONTROL_TOKENS]
        has_forgery = len(total_forgeries) > 0

        if has_forgery and self.reject_on_forgery:
            raise ValueError(
                f"Boundary forgery attempt rejected! Detected illegal control tokens: {total_forgeries} "
                f"(critical: {critical_forgeries})"
            )

        clean_state = sanitize_state(state)
        clean_cands = sanitize_candidates(candidates)
        clean_instr = sanitize_input_text(instruction) if instruction else instruction

        audit_info = {
            "forgery_detected": has_forgery,
            "forged_tokens": total_forgeries,
            "critical_forgeries": critical_forgeries,
            "escaped_count": len(total_forgeries),
            "clean_instruction": clean_instr
        }
        return clean_state, clean_cands, audit_info
