"""Zero Web Browser Harness & DOM Safety Package (Issue #34 & RFC-034)."""

from .types import (
    BoundingBox,
    DOMElement,
    ModalOverlay,
    WebTargetSlot,
    WebObservation,
    WebOperationType,
    BrowserStepResult,
    HarnessError,
    StaleObservationError,
    OccludedElementError,
    InvalidSlotError,
    HarnessSecurityException,
)
from .dom_observer import (
    PointerPenetrationChecker,
    OffscreenDetector,
    DOMObserver,
)
from .signature import PageSignature
from .discrete_dispatcher import ClosureHandlePool
from .text_cache import TextCache
from .visual_recorder import (
    VisualEvent,
    VisualTrajectoryRecorder,
)
from .browser_harness import ZeroBrowserHarness

__all__ = [
    "BoundingBox",
    "DOMElement",
    "ModalOverlay",
    "WebTargetSlot",
    "WebObservation",
    "WebOperationType",
    "BrowserStepResult",
    "HarnessError",
    "StaleObservationError",
    "OccludedElementError",
    "InvalidSlotError",
    "HarnessSecurityException",
    "PointerPenetrationChecker",
    "OffscreenDetector",
    "DOMObserver",
    "PageSignature",
    "ClosureHandlePool",
    "TextCache",
    "VisualEvent",
    "VisualTrajectoryRecorder",
    "ZeroBrowserHarness",
]
