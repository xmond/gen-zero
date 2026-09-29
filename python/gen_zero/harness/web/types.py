"""Web Harness Data Structures and Types (Issue #34 & RFC-034).

Defines typed contracts for zero-selector DOM closure handles, pointer penetration
hit-testing, state signatures, and two-stage discrete operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class WebOperationType(str, Enum):
    """Discrete browser operation types supported by Zero Browser Harness."""
    CLICK = "CLICK"
    TYPE_TEXT = "TYPE_TEXT"
    SELECT = "SELECT"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    WAIT = "WAIT"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"
    DONE = "DONE"


@dataclass
class BoundingBox:
    """Represents the 2D spatial layout of a DOM element or overlay."""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def cx(self) -> float:
        return self.x + self.width / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.height / 2.0

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains_point(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.right) and (self.y <= py <= self.bottom)


@dataclass
class ModalOverlay:
    """Represents a modal backdrop, sticky header, or dialog overlay covering the viewport."""
    overlay_id: str
    bounding_box: BoundingBox
    z_index: int = 100
    is_visible: bool = True
    dismiss_button_id: Optional[str] = None


@dataclass
class DOMElement:
    """Closure-retained DOM element representation holding live attributes and execution hooks."""
    element_id: str
    tag: str
    role: Optional[str] = None
    name: Optional[str] = None
    aria_label: Optional[str] = None
    text_content: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    bounding_box: BoundingBox = field(default_factory=BoundingBox)
    is_visible: bool = True
    is_connected: bool = True
    opacity: float = 1.0
    inert: bool = False
    aria_hidden: bool = False
    z_index: int = 0
    options: Optional[List[str]] = None
    value: Optional[str] = None
    checked: Optional[bool] = None
    on_click: Optional[Callable[[], Any]] = None
    on_fill: Optional[Callable[[str], Any]] = None
    on_select: Optional[Callable[[str], Any]] = None

    @property
    def effective_label(self) -> str:
        """Returns the most informative descriptive label for this element."""
        if self.aria_label and self.aria_label.strip():
            return self.aria_label.strip()
        if self.text_content and self.text_content.strip():
            return self.text_content.strip()
        if self.name and self.name.strip():
            return self.name.strip()
        if self.element_id and self.element_id.strip():
            return self.element_id.strip()
        return f"{self.tag}:{self.role or 'element'}"

    def signature_tuple(self) -> Tuple[Any, ...]:
        """Returns immutable attribute tuple for page state fingerprinting."""
        return (
            self.tag.lower(),
            (self.role or "").lower(),
            self.element_id,
            self.name or "",
            self.attributes.get("href", ""),
            self.attributes.get("aria-checked", ""),
            self.attributes.get("aria-selected", ""),
            self.checked if self.checked is not None else self.attributes.get("checked", False),
            self.value if self.value is not None else self.attributes.get("value", ""),
            self.is_connected,
        )


@dataclass
class WebTargetSlot:
    """Discrete operational slot exposed to Zero model for zero-selector dispatch."""
    slot_id: int
    operation: WebOperationType
    label: str
    role: Optional[str] = None
    current_value: Optional[str] = None
    option_value: Optional[str] = None
    criteria_key: str = ""
    description: str = ""
    bounding_box: Optional[BoundingBox] = None
    element_ref: Optional[DOMElement] = None


@dataclass
class WebObservation:
    """Structured DOM observation snapshot generated from penetration hit-test and offscreen sensing."""
    url: str
    title: str
    text_summary: str
    targets: List[WebTargetSlot]
    offscreen_above_count: int
    offscreen_below_count: int
    can_scroll_up: bool
    can_scroll_down: bool
    scroll_x: float
    scroll_y: float
    viewport_width: float
    viewport_height: float
    signature_hash: str
    timestamp: float = 0.0

    def get_slot_by_key(self, criteria_key: str) -> Optional[WebTargetSlot]:
        for target in self.targets:
            if target.criteria_key == criteria_key:
                return target
        return None

    def get_criteria_dict(self) -> Dict[str, str]:
        """Returns criteria dictionary suitable for Zero reflex decision."""
        return {target.criteria_key: target.description for target in self.targets}


@dataclass
class BrowserStepResult:
    """Structured result of executing a single discrete browser step."""
    success: bool
    action: str
    slot_id: Optional[int] = None
    operation: WebOperationType = WebOperationType.WAIT
    text_param: Optional[str] = None
    latency_ms: float = 0.0
    stale_recovered: bool = False
    observation_after: Optional[WebObservation] = None
    is_terminal: bool = False
    terminal_reason: Optional[str] = None
    visual_event_id: Optional[str] = None
    error: Optional[str] = None


class HarnessError(Exception):
    """Base exception for Zero browser harness errors."""
    pass


class StaleObservationError(HarnessError):
    """Raised when page state signature does not match observation snapshot (anti-race)."""
    def __init__(self, expected_hash: str, actual_hash: str, reason: str = "Page signature mismatch"):
        super().__init__(f"{reason}: expected={expected_hash}, actual={actual_hash}")
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.reason = reason


class OccludedElementError(HarnessError):
    """Raised when attempting to interact with an element occluded by modal overlay."""
    pass


class InvalidSlotError(HarnessError):
    """Raised when an unrecognized slot ID or action criteria key is passed."""
    pass


class HarnessSecurityException(HarnessError):
    """Raised on security violation (e.g. attempted selector injection)."""
    pass
