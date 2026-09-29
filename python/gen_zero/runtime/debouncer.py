"""Coalesced Debouncer for Async Supervisor Runtime.

Implements Milestone 1 of Issue #20:
1. Coalescing debounce floor (default: 5.0 seconds):
   Prevents hammering the semantic evaluation engine when the worker emits rapid bursts of tool events.
2. Periodic patrol interval (default: 30.0 seconds):
   Triggers periodic assessment when the worker is executing silent, long-running tasks.
"""

import threading
import time
from typing import Optional, Dict, Any, List, Tuple
import dataclasses



@dataclasses.dataclass
class DebounceEvent:
    event_type: str
    timestamp: float
    payload: Optional[Dict[str, Any]] = None


class CoalescedDebouncer:
    """Manages event coalescing and periodic pulse scheduling for the supervisor loop."""

    def __init__(
        self,
        debounce_floor_seconds: float = 5.0,
        periodic_interval_seconds: float = 30.0,
    ):
        self.debounce_floor_seconds = debounce_floor_seconds
        self.periodic_interval_seconds = periodic_interval_seconds
        self._last_event_time: Optional[float] = None
        self._last_assessment_time: float = time.monotonic()
        self._pending_events: List[DebounceEvent] = []
        self._lock = threading.Lock()

    def record_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Records an incoming worker or environmental event."""
        with self._lock:
            now = time.monotonic()
            self._last_event_time = now
            self._pending_events.append(DebounceEvent(
                event_type=event_type,
                timestamp=now,
                payload=payload or {}
            ))

    def should_assess(self) -> Tuple[bool, str]:
        """Determines whether a supervisor assessment should be performed now.

        Returns:
            Tuple of (should_assess: bool, trigger_reason: str).
        """
        with self._lock:
            now = time.monotonic()

            # 1. Periodic pulse: has periodic interval elapsed since last assessment?
            if now - self._last_assessment_time >= self.periodic_interval_seconds:
                return True, "PERIODIC_INTERVAL_ELAPSED"

            # 2. Coalesced event debouncing: have events arrived and quieted down for debounce_floor?
            if self._pending_events and self._last_event_time is not None:
                time_since_last_event = now - self._last_event_time
                if time_since_last_event >= self.debounce_floor_seconds:
                    return True, f"DEBOUNCE_FLOOR_QUIET_{len(self._pending_events)}_EVENTS"

            return False, "COALESCING"

    def mark_assessed(self) -> List[DebounceEvent]:
        """Marks current timestamp as assessed and drains coalesced pending events."""
        with self._lock:
            now = time.monotonic()
            self._last_assessment_time = now
            self._last_event_time = None
            drained = list(self._pending_events)
            self._pending_events.clear()
            return drained

    def reset(self) -> None:
        """Resets the debouncer clock and pending events."""
        with self._lock:
            now = time.monotonic()
            self._last_event_time = None
            self._last_assessment_time = now
            self._pending_events.clear()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_events)

    @property
    def time_since_last_assessment(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_assessment_time
