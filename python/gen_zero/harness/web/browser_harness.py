"""Zero Web Browser Harness Orchestrator (Issue #34 & RFC-034).

Coordinates:
1. DOM observation and pointer penetration hit-testing (<300ms inner loop).
2. Closure-retained handles with Zero-Selector discrete slot mapping.
3. Multi-dimensional page state snapshot signature & assert_fresh anti-race assertions.
4. Two-stage decoupled action and argument parameter execution.
5. High-priority REVIEW / BLOCKED / DONE sentinel guardrails.
6. Visual trajectory overlay recording and done_unverified verification artifacts.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import time

from .types import (
    BoundingBox,
    BrowserStepResult,
    DOMElement,
    ModalOverlay,
    StaleObservationError,
    WebObservation,
    WebOperationType,
    WebTargetSlot,
)
from .dom_observer import DOMObserver
from .signature import PageSignature
from .discrete_dispatcher import ClosureHandlePool
from .text_cache import TextCache
from .visual_recorder import VisualTrajectoryRecorder


class ZeroBrowserHarness:
    """Production runtime harness for autonomous web agent execution."""

    def __init__(
        self,
        url: str = "about:blank",
        title: str = "Blank",
        viewport_width: float = 1280.0,
        viewport_height: float = 800.0,
        decision_client: Optional[Any] = None
    ):
        self.url = url
        self.title = title
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.decision_client = decision_client

        self.scroll_x: float = 0.0
        self.scroll_y: float = 0.0
        self.max_scroll_y: float = 2000.0

        self.elements: List[DOMElement] = []
        self.overlays: List[ModalOverlay] = []

        self.observer = DOMObserver(viewport_width, viewport_height)
        self.closure_pool = ClosureHandlePool(scroll_handler=self._handle_scroll)
        self.text_cache = TextCache()
        self.recorder = VisualTrajectoryRecorder()

        self.current_observation: Optional[WebObservation] = None
        self.latest_signature_hash: str = ""

    def _handle_scroll(self, direction: str) -> None:
        """Applies scroll offsets."""
        delta = self.viewport_height * 0.75
        if direction == "SCROLL_UP":
            self.scroll_y = max(0.0, self.scroll_y - delta)
        elif direction == "SCROLL_DOWN":
            self.scroll_y = min(self.max_scroll_y, self.scroll_y + delta)

    def set_page_state(
        self,
        url: Optional[str] = None,
        title: Optional[str] = None,
        elements: Optional[Sequence[DOMElement]] = None,
        overlays: Optional[Sequence[ModalOverlay]] = None,
        scroll_x: Optional[float] = None,
        scroll_y: Optional[float] = None,
        max_scroll_y: Optional[float] = None
    ) -> None:
        """Updates internal DOM and page state representation."""
        if url is not None:
            self.url = url
        if title is not None:
            self.title = title
        if elements is not None:
            self.elements = list(elements)
        if overlays is not None:
            self.overlays = list(overlays)
        if scroll_x is not None:
            self.scroll_x = float(scroll_x)
        if scroll_y is not None:
            self.scroll_y = float(scroll_y)
        if max_scroll_y is not None:
            self.max_scroll_y = float(max_scroll_y)

    def observe(
        self,
        elements: Optional[Sequence[DOMElement]] = None,
        overlays: Optional[Sequence[ModalOverlay]] = None,
        url: Optional[str] = None,
        title: Optional[str] = None,
        scroll_x: Optional[float] = None,
        scroll_y: Optional[float] = None,
        max_scroll_y: Optional[float] = None
    ) -> WebObservation:
        """Executes full DOM observation cycle, filtering occlusions and updating closure pool."""
        self.set_page_state(url, title, elements, overlays, scroll_x, scroll_y, max_scroll_y)

        obs, nodes = self.observer.observe(
            url=self.url,
            title=self.title,
            elements=self.elements,
            overlays=self.overlays,
            scroll_x=self.scroll_x,
            scroll_y=self.scroll_y,
            max_scroll_y=self.max_scroll_y,
            timestamp=time.time()
        )

        self.current_observation = obs
        self.latest_signature_hash = obs.signature_hash
        self.closure_pool.update_closure(nodes, obs.targets)
        return obs

    def step(
        self,
        action_choice: Union[str, int],
        text_param: Optional[str] = None,
        option_param: Optional[str] = None,
        assert_freshness: bool = True
    ) -> BrowserStepResult:
        """Executes a single discrete browser interaction step.

        Args:
            action_choice: Discrete action criteria string (e.g., 'CLICK:0', 'DONE') or slot ID.
            text_param: Explicit text argument for TYPE_TEXT.
            option_param: Explicit option argument for SELECT.
            assert_freshness: Whether to verify page signature anti-race check before execution.

        Returns:
            BrowserStepResult with execution details, freshness recovery status, and visual trace.
        """
        t0 = time.perf_counter()
        action_str = str(action_choice).strip()
        stale_recovered = False

        if self.current_observation is None:
            self.observe()

        # 1. Anti-race signature assertion
        if assert_freshness:
            try:
                PageSignature.assert_fresh(
                    expected_signature=self.latest_signature_hash,
                    current_url=self.url,
                    current_scroll_x=self.scroll_x,
                    current_scroll_y=self.scroll_y,
                    current_nodes=self.closure_pool.nodes
                )
            except StaleObservationError:
                # Anti-race recovery: Re-observe immediately
                stale_recovered = True
                self.observe()
                # Re-validate action compatibility
                try:
                    self.closure_pool.validate_zero_selector_safety(action_str)
                except Exception as e:
                    latency = (time.perf_counter() - t0) * 1000.0
                    return BrowserStepResult(
                        success=False,
                        action=action_str,
                        latency_ms=latency,
                        stale_recovered=True,
                        error=f"Stale observation recovered, action no longer valid: {str(e)}"
                    )

        # 2. Parse operation
        op, slot_id, aux = self.closure_pool.parse_action(action_str)

        # 3. Handle terminal sentinel operations
        if op == WebOperationType.DONE:
            latency = (time.perf_counter() - t0) * 1000.0
            self.recorder.record(
                x=self.viewport_width / 2.0,
                y=self.viewport_height / 2.0,
                operation="DONE",
                label="Task Complete",
                signature_hash=self.latest_signature_hash
            )
            return BrowserStepResult(
                success=True,
                action="DONE",
                operation=op,
                latency_ms=latency,
                stale_recovered=stale_recovered,
                is_terminal=True,
                terminal_reason="DONE"
            )

        if op == WebOperationType.REVIEW:
            latency = (time.perf_counter() - t0) * 1000.0
            self.recorder.record(
                x=self.viewport_width / 2.0,
                y=self.viewport_height / 2.0,
                operation="REVIEW",
                label="Host Review Required",
                signature_hash=self.latest_signature_hash
            )
            return BrowserStepResult(
                success=True,
                action="REVIEW",
                operation=op,
                latency_ms=latency,
                stale_recovered=stale_recovered,
                is_terminal=True,
                terminal_reason="REVIEW"
            )

        if op == WebOperationType.BLOCKED:
            latency = (time.perf_counter() - t0) * 1000.0
            self.recorder.record(
                x=self.viewport_width / 2.0,
                y=self.viewport_height / 2.0,
                operation="BLOCKED",
                label="Task Blocked",
                signature_hash=self.latest_signature_hash
            )
            return BrowserStepResult(
                success=True,
                action="BLOCKED",
                operation=op,
                latency_ms=latency,
                stale_recovered=stale_recovered,
                is_terminal=True,
                terminal_reason="BLOCKED"
            )

        if op == WebOperationType.WAIT:
            latency = (time.perf_counter() - t0) * 1000.0
            self.recorder.record(
                x=self.viewport_width / 2.0,
                y=self.viewport_height / 2.0,
                operation="WAIT",
                label="Wait Idle",
                signature_hash=self.latest_signature_hash
            )
            return BrowserStepResult(
                success=True,
                action="WAIT",
                operation=op,
                latency_ms=latency,
                stale_recovered=stale_recovered,
                is_terminal=False
            )

        # 4. Dispatch action to closure handle
        success, target_node, msg = self.closure_pool.dispatch(
            action_input=action_str,
            text_param=text_param,
            option_param=option_param
        )

        # 5. Visual recording
        cx = target_node.bounding_box.cx if target_node else (self.viewport_width / 2.0)
        cy = target_node.bounding_box.cy if target_node else (self.viewport_height / 2.0)
        lbl = target_node.effective_label if target_node else op.value

        v_event = self.recorder.record(
            x=cx,
            y=cy,
            operation=op.value,
            slot_id=slot_id,
            label=lbl,
            signature_hash=self.latest_signature_hash
        )

        # Re-observe to refresh DOM signature after mutation
        obs_after = self.observe()

        latency = (time.perf_counter() - t0) * 1000.0
        return BrowserStepResult(
            success=success,
            action=action_str,
            slot_id=slot_id,
            operation=op,
            text_param=text_param,
            latency_ms=latency,
            stale_recovered=stale_recovered,
            observation_after=obs_after,
            is_terminal=False,
            visual_event_id=v_event.event_id
        )

    def run_autonomous(
        self,
        goal: str,
        decision_client: Optional[Any] = None,
        max_steps: int = 15,
        step_callback: Optional[Callable[[int, BrowserStepResult], None]] = None
    ) -> Dict[str, Any]:
        """Runs an autonomous closed-loop browser session until goal completion or sentinel escalation."""
        client = decision_client or self.decision_client
        if client is None:
            raise ValueError("decision_client must be provided to run_autonomous")

        self.observe()
        history: List[BrowserStepResult] = []
        start_time = time.perf_counter()

        for step_idx in range(1, max_steps + 1):
            if not self.current_observation or not self.current_observation.targets:
                break

            criteria = self.current_observation.get_criteria_dict()
            candidates = list(criteria.keys())

            state_desc = (
                f"Goal: {goal} | URL: {self.url} | Title: {self.title} | "
                f"Scroll: ({self.scroll_x}, {self.scroll_y})"
            )

            dec_res = client.decide(
                state=state_desc,
                candidates=candidates,
                mode="reflex",
                candidate_descriptions=criteria
            )

            chosen_action = dec_res.get("action", candidates[0])

            # Two-stage argument resolution for TYPE_TEXT
            text_arg = None
            if "TYPE_TEXT" in chosen_action:
                slot_obj = self.current_observation.get_slot_by_key(chosen_action)
                if slot_obj:
                    text_arg = self.text_cache.resolve_text_parameter(
                        goal=goal,
                        slot=slot_obj,
                        signature_hash=self.latest_signature_hash
                    )

            step_res = self.step(chosen_action, text_param=text_arg)
            history.append(step_res)

            if step_callback:
                step_callback(step_idx, step_res)

            if step_res.is_terminal:
                break

        total_duration_ms = (time.perf_counter() - start_time) * 1000.0
        last_step = history[-1] if history else None

        return {
            "success": last_step.is_terminal and last_step.terminal_reason == "DONE" if last_step else False,
            "terminal_reason": last_step.terminal_reason if last_step else "MAX_STEPS_EXCEEDED",
            "steps_taken": len(history),
            "total_duration_ms": total_duration_ms,
            "mean_step_latency_ms": (total_duration_ms / len(history)) if history else 0.0,
            "stale_recoveries": sum(1 for s in history if s.stale_recovered),
            "trajectory_report": self.recorder.export_verification_report(),
            "svg_overlay": self.recorder.export_svg_overlay(self.viewport_width, self.viewport_height)
        }
