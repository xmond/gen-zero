"""Visual Trajectory Recorder with Blue Cursor and Red Ripple Overlay (Issue #34 & RFC-034).

Implements visual audit trails and SVG overlay rendering:
1. Records coordinate points, slot IDs, operation verbs, and timestamps.
2. Blue follower cursor (#2563eb) and animated red expanding ripple (#ef4444).
3. Exportable JSONL trajectory log and SVG overlay for independent host-agent visual audit.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import json
import time


@dataclass
class VisualEvent:
    """Represents a discrete visual interaction event on the browser viewport."""
    event_id: str
    step_number: int
    timestamp: float
    x: float
    y: float
    operation: str
    slot_id: Optional[int] = None
    label: str = ""
    cursor_color: str = "#2563eb"
    ripple_color: str = "#ef4444"
    ripple_radius: float = 24.0
    signature_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisualTrajectoryRecorder:
    """Records visual interaction events and renders SVG/JSONL visual verification artifacts."""

    def __init__(self):
        self.events: List[VisualEvent] = []
        self._step_counter = 0

    def record(
        self,
        x: float,
        y: float,
        operation: str,
        slot_id: Optional[int] = None,
        label: str = "",
        signature_hash: str = "",
        timestamp: Optional[float] = None
    ) -> VisualEvent:
        """Records an action event with spatial coordinates."""
        self._step_counter += 1
        ts = timestamp if timestamp is not None else time.time()
        ev = VisualEvent(
            event_id=f"vev_{self._step_counter}_{int(ts * 1000)}",
            step_number=self._step_counter,
            timestamp=ts,
            x=round(float(x), 1),
            y=round(float(y), 1),
            operation=operation,
            slot_id=slot_id,
            label=label,
            signature_hash=signature_hash
        )
        self.events.append(ev)
        return ev

    def export_jsonl(self) -> str:
        """Exports the event sequence as newline-delimited JSON."""
        lines = [json.dumps(ev.to_dict(), default=str) for ev in self.events]
        return "\n".join(lines)

    def export_svg_overlay(
        self,
        viewport_w: float = 1280.0,
        viewport_h: float = 800.0
    ) -> str:
        """Generates a high-fidelity SVG overlay representing the agent's interaction trajectory."""
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {viewport_w} {viewport_h}" width="{viewport_w}" height="{viewport_h}">',
            '  <defs>',
            '    <style>',
            '      .cursor { fill: #2563eb; stroke: #ffffff; stroke-width: 2px; }',
            '      .ripple { fill: none; stroke: #ef4444; stroke-width: 3px; stroke-dasharray: 4,2; opacity: 0.85; }',
            '      .path-line { stroke: #3b82f6; stroke-width: 2px; stroke-dasharray: 5,5; fill: none; opacity: 0.6; }',
            '      .step-text { font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 11px; fill: #1e293b; font-weight: bold; }',
            '    </style>',
            '  </defs>',
            '  <g id="trajectory_layer">'
        ]

        # 1. Trajectory connecting lines
        if len(self.events) > 1:
            points = " ".join(f"{ev.x},{ev.y}" for ev in self.events)
            svg_parts.append(f'    <polyline class="path-line" points="{points}" />')

        # 2. Render each event point
        for ev in self.events:
            # Red ripple circle
            svg_parts.append(
                f'    <circle class="ripple" cx="{ev.x}" cy="{ev.y}" r="{ev.ripple_radius}" />'
            )
            # Blue cursor dot
            svg_parts.append(
                f'    <circle class="cursor" cx="{ev.x}" cy="{ev.y}" r="8" />'
            )
            # Step index inside or near cursor
            svg_parts.append(
                f'    <text class="step-text" x="{ev.x + 12}" y="{ev.y + 4}">'
                f'#{ev.step_number} {ev.operation} [{ev.label[:20]}]'
                f'</text>'
            )

        svg_parts.append('  </g>')
        svg_parts.append('</svg>')
        return "\n".join(svg_parts)

    def export_verification_report(self) -> Dict[str, Any]:
        """Exports audit summary for the done_unverified verification pipeline."""
        return {
            "total_steps": len(self.events),
            "events": [ev.to_dict() for ev in self.events],
            "operations_breakdown": {
                op: sum(1 for e in self.events if e.operation == op)
                for op in set(e.operation for e in self.events)
            }
        }

    def clear(self) -> None:
        self.events.clear()
        self._step_counter = 0
