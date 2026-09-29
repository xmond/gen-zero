"""Discrete Closure Handle Dispatcher & Zero-Selector Security Guard (Issue #34 & RFC-034).

Enforces the Zero-Selector contract:
1. Candidate criteria exposed to Zero contain exclusively discrete operation verbs and integer slot IDs
   (e.g., CLICK:0, TYPE_TEXT:1, SELECT:2:opt, SCROLL_UP, WAIT, REVIEW, BLOCKED, DONE).
2. The model NEVER generates CSS/XPath selectors or executable JavaScript.
3. Actions are dispatched directly to closure-retained native element handles by slot index.
4. Attempts to inject arbitrary selector strings or executable script are blocked with HarnessSecurityException.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import re
from .types import (
    DOMElement,
    HarnessSecurityException,
    InvalidSlotError,
    WebOperationType,
    WebTargetSlot,
)


class ClosureHandlePool:
    """Retains active DOM element handles in closure memory and dispatches discrete slot actions."""

    SELECTOR_INJECTION_PATTERN = re.compile(
        r'([.#\[\]>+~]|//|xpath|document|window|eval|<script|\b(div|span|button|input|a|p)\b)',
        re.IGNORECASE
    )

    def __init__(
        self,
        nodes: Optional[Sequence[DOMElement]] = None,
        targets: Optional[Sequence[WebTargetSlot]] = None,
        scroll_handler: Optional[Callable[[str], Any]] = None
    ):
        self.nodes: List[DOMElement] = list(nodes) if nodes else []
        self.targets: List[WebTargetSlot] = list(targets) if targets else []
        self.targets_by_key: Dict[str, WebTargetSlot] = {t.criteria_key: t for t in self.targets}
        self.scroll_handler = scroll_handler

    def update_closure(
        self,
        nodes: Sequence[DOMElement],
        targets: Sequence[WebTargetSlot]
    ) -> None:
        """Updates the closure-retained nodes and target slots for a new observation cycle."""
        self.nodes = list(nodes)
        self.targets = list(targets)
        self.targets_by_key = {t.criteria_key: t for t in self.targets}

    def validate_zero_selector_safety(self, action_key: str) -> None:
        """Enforces that the action key is a recognized discrete token and not a CSS/XPath selector or script."""
        action_clean = action_key.strip()

        # If it's a known criteria key, it's valid
        if action_clean in self.targets_by_key:
            return

        # Check standard verb formats: VERB:int or VERB:int:param
        parts = action_clean.split(":")
        verb = parts[0].upper()
        if verb in ("CLICK", "TYPE_TEXT", "SELECT") and len(parts) >= 2 and parts[1].isdigit():
            return

        if verb in ("SCROLL_UP", "SCROLL_DOWN", "WAIT", "REVIEW", "BLOCKED", "DONE"):
            return

        # If the string contains CSS/XPath or script characters, trigger security exception
        if self.SELECTOR_INJECTION_PATTERN.search(action_clean):
            raise HarnessSecurityException(
                f"Security violation: received selector/script injection attempt '{action_clean}'. "
                f"Zero-Selector contract strictly forbids CSS/XPath selectors or JavaScript code."
            )

        raise InvalidSlotError(f"Action '{action_clean}' is not a valid discrete slot criteria in current observation pool")

    def parse_action(
        self,
        action_input: Union[str, int]
    ) -> Tuple[WebOperationType, Optional[int], Optional[str]]:
        """Parses a criteria string or integer slot into (operation_type, slot_id, auxiliary_param)."""
        if isinstance(action_input, int):
            slot_id = action_input
            if slot_id < 0 or slot_id >= len(self.nodes):
                raise InvalidSlotError(f"Slot ID {slot_id} out of bounds (0..{len(self.nodes) - 1})")
            node = self.nodes[slot_id]
            tag = node.tag.lower()
            role = (node.role or "").lower()
            if tag in ("input", "textarea") or role in ("textbox", "searchbox"):
                return WebOperationType.TYPE_TEXT, slot_id, None
            if tag == "select" or role == "combobox":
                return WebOperationType.SELECT, slot_id, None
            return WebOperationType.CLICK, slot_id, None

        action_str = str(action_input).strip()
        self.validate_zero_selector_safety(action_str)

        # Check sentinels
        if action_str == "WAIT":
            return WebOperationType.WAIT, None, None
        if action_str == "REVIEW":
            return WebOperationType.REVIEW, None, None
        if action_str == "BLOCKED":
            return WebOperationType.BLOCKED, None, None
        if action_str == "DONE":
            return WebOperationType.DONE, None, None
        if action_str == "SCROLL_UP":
            return WebOperationType.SCROLL_UP, None, None
        if action_str == "SCROLL_DOWN":
            return WebOperationType.SCROLL_DOWN, None, None

        # Parse verbs with slot index: VERB:slot_id[:aux]
        parts = action_str.split(":")
        verb_str = parts[0].upper()
        slot_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        aux = parts[2] if len(parts) > 2 else None

        if verb_str == "CLICK":
            return WebOperationType.CLICK, slot_id, aux
        if verb_str == "TYPE_TEXT":
            return WebOperationType.TYPE_TEXT, slot_id, aux
        if verb_str == "SELECT":
            return WebOperationType.SELECT, slot_id, aux

        raise InvalidSlotError(f"Unrecognized operation verb in '{action_str}'")

    def dispatch(
        self,
        action_input: Union[str, int],
        text_param: Optional[str] = None,
        option_param: Optional[str] = None
    ) -> Tuple[bool, Optional[DOMElement], str]:
        """Dispatches an atomic action to closure-retained native handles.

        Returns:
            Tuple of (success: bool, target_node: Optional[DOMElement], message: str)
        """
        op, slot_id, aux = self.parse_action(action_input)

        if op in (WebOperationType.WAIT, WebOperationType.REVIEW, WebOperationType.BLOCKED, WebOperationType.DONE):
            return True, None, f"Sentinel operation {op.value} invoked"

        if op in (WebOperationType.SCROLL_UP, WebOperationType.SCROLL_DOWN):
            if self.scroll_handler:
                self.scroll_handler(op.value)
            return True, None, f"Executed directional scroll {op.value}"

        if slot_id is None or slot_id < 0 or slot_id >= len(self.nodes):
            raise InvalidSlotError(f"Slot ID {slot_id} invalid or out of range")

        target_node = self.nodes[slot_id]

        if op == WebOperationType.CLICK:
            # Handle native on_click callback if registered
            if target_node.on_click is not None:
                target_node.on_click()
            # Toggle checked state if applicable
            if target_node.checked is not None:
                target_node.checked = not target_node.checked
            return True, target_node, f"Clicked element [{target_node.effective_label}] at slot {slot_id}"

        if op == WebOperationType.TYPE_TEXT:
            content_to_type = text_param if text_param is not None else (aux or "")
            target_node.value = content_to_type
            if target_node.on_fill is not None:
                target_node.on_fill(content_to_type)
            return True, target_node, f"Typed text '{content_to_type}' into [{target_node.effective_label}] at slot {slot_id}"

        if op == WebOperationType.SELECT:
            selected_option = option_param if option_param is not None else (aux or "")
            target_node.value = selected_option
            if target_node.on_select is not None:
                target_node.on_select(selected_option)
            return True, target_node, f"Selected option '{selected_option}' in [{target_node.effective_label}] at slot {slot_id}"

        return False, None, f"Unsupported operation {op}"
