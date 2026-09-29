"""DOM Observer with Pointer Penetration Hit-Test and Offscreen Sensing (Issue #34 & RFC-034).

Provides:
1. PointerPenetrationChecker: centroid hit-testing against modal overlays and backdrops.
2. OffscreenDetector: directional sensing for above/below viewport controls to prevent blind scrolling.
3. DOMObserver: high-efficiency DOM traversal emitting zero-selector closure target slots.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import re
from .types import (
    BoundingBox,
    DOMElement,
    ModalOverlay,
    WebObservation,
    WebOperationType,
    WebTargetSlot,
)


class PointerPenetrationChecker:
    """Verifies that elements are physically accessible to pointer events (elementFromPoint simulation)."""

    @classmethod
    def check_visibility(cls, element: DOMElement) -> Tuple[bool, str]:
        """Checks styling, geometric size, and aria/inert accessibility flags."""
        if not element.is_visible:
            return False, "Element marked is_visible=False"
        if element.opacity <= 0.001:
            return False, f"Element has zero or near-zero opacity ({element.opacity})"
        if element.inert:
            return False, "Element has inert attribute"
        if element.aria_hidden:
            return False, "Element has aria-hidden='true'"
        if element.attributes.get("disabled", False):
            return False, "Element has disabled attribute"
        if not element.is_connected:
            return False, "Element is disconnected from DOM"

        bbox = element.bounding_box
        if bbox.width <= 0.0 or bbox.height <= 0.0:
            return False, f"Non-positive dimensions ({bbox.width}x{bbox.height})"

        return True, "Visible and interactive"

    @classmethod
    def receives_pointer(
        cls,
        element: DOMElement,
        overlays: Optional[Sequence[ModalOverlay]] = None
    ) -> Tuple[bool, str]:
        """Simulates document.elementFromPoint hit-testing against viewport overlays.

        Filters elements occluded by modal backdrops, cookie banners, or higher z-index sheets.
        """
        vis_ok, vis_reason = cls.check_visibility(element)
        if not vis_ok:
            return False, vis_reason

        if not overlays:
            return True, "Pointer received (no overlays)"

        cx = element.bounding_box.cx
        cy = element.bounding_box.cy

        for overlay in overlays:
            if not overlay.is_visible:
                continue

            # If element is the dismiss button of this overlay, it must receive pointer
            if overlay.dismiss_button_id and element.element_id == overlay.dismiss_button_id:
                continue

            # Check if overlay physically bounds the element centroid
            if overlay.bounding_box.contains_point(cx, cy):
                # If overlay has higher z-index than element, it occludes element
                if overlay.z_index > element.z_index:
                    return False, f"Occluded by overlay '{overlay.overlay_id}' (z-index {overlay.z_index} > {element.z_index})"

        return True, "Pointer received"


class OffscreenDetector:
    """Detects controls outside the current viewport to guide deterministic scrolling."""

    @classmethod
    def is_in_viewport(
        cls,
        bbox: BoundingBox,
        viewport_w: float = 1280.0,
        viewport_h: float = 800.0
    ) -> bool:
        """Determines if element bounding box intersects the current viewport."""
        return (
            bbox.bottom > 0.0
            and bbox.y < viewport_h
            and bbox.right > 0.0
            and bbox.x < viewport_w
        )

    @classmethod
    def is_offscreen_above(cls, bbox: BoundingBox) -> bool:
        """Determines if element lies completely above the visible viewport."""
        return bbox.bottom <= 0.0

    @classmethod
    def is_offscreen_below(cls, bbox: BoundingBox, viewport_h: float = 800.0) -> bool:
        """Determines if element lies completely below the visible viewport."""
        return bbox.y >= viewport_h


class DOMObserver:
    """Extracts DOM observation snapshots and closure-retained candidates."""

    INTERACTIVE_TAGS = {"button", "a", "input", "select", "textarea", "option"}
    INTERACTIVE_ROLES = {
        "button", "link", "checkbox", "radio", "tab", "menuitem",
        "switch", "combobox", "textbox", "searchbox", "option"
    }

    def __init__(
        self,
        viewport_width: float = 1280.0,
        viewport_height: float = 800.0
    ):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    def is_interactive_candidate(self, element: DOMElement) -> bool:
        """Determines if an element is a candidate for autonomous user interaction."""
        tag = element.tag.lower()
        role = (element.role or "").lower()

        if tag in self.INTERACTIVE_TAGS:
            return True
        if role in self.INTERACTIVE_ROLES:
            return True
        if element.on_click is not None or element.on_fill is not None or element.on_select is not None:
            return True
        if element.attributes.get("onclick") or element.attributes.get("role") in self.INTERACTIVE_ROLES:
            return True
        return False

    def observe(
        self,
        url: str,
        title: str,
        elements: Sequence[DOMElement],
        overlays: Optional[Sequence[ModalOverlay]] = None,
        scroll_x: float = 0.0,
        scroll_y: float = 0.0,
        max_scroll_y: float = 2000.0,
        page_summary: str = "",
        timestamp: float = 0.0
    ) -> Tuple[WebObservation, List[DOMElement]]:
        """Processes page DOM, performs penetration checks, and constructs observation snapshot.

        Returns:
            Tuple of (WebObservation, closure_nodes)
        """
        closure_nodes: List[DOMElement] = []
        targets: List[WebTargetSlot] = []

        offscreen_above = 0
        offscreen_below = 0

        # 1. Classify elements by viewport and pointer penetration
        for el in elements:
            if not self.is_interactive_candidate(el):
                continue

            bbox = el.bounding_box
            in_view = OffscreenDetector.is_in_viewport(bbox, self.viewport_width, self.viewport_height)

            if in_view:
                receives, _ = PointerPenetrationChecker.receives_pointer(el, overlays)
                if receives:
                    closure_nodes.append(el)
            else:
                vis_ok, _ = PointerPenetrationChecker.check_visibility(el)
                if vis_ok:
                    if OffscreenDetector.is_offscreen_above(bbox):
                        offscreen_above += 1
                    elif OffscreenDetector.is_offscreen_below(bbox, self.viewport_height):
                        offscreen_below += 1

        # 2. Build discrete target slots for in-viewport closure nodes
        slot_counter = 0
        for node in closure_nodes:
            tag = node.tag.lower()
            role = (node.role or "").lower()
            label = node.effective_label

            if tag in ("input", "textarea") or role in ("textbox", "searchbox"):
                input_type = str(node.attributes.get("type", "text")).lower()
                if input_type in ("checkbox", "radio"):
                    slot = WebTargetSlot(
                        slot_id=slot_counter,
                        operation=WebOperationType.CLICK,
                        label=label,
                        role=role or input_type,
                        current_value="checked" if node.checked else "unchecked",
                        criteria_key=f"CLICK:{slot_counter}",
                        description=f"Toggle {input_type} [{label}]",
                        bounding_box=node.bounding_box,
                        element_ref=node
                    )
                    targets.append(slot)
                    slot_counter += 1
                elif input_type in ("submit", "button", "reset"):
                    slot = WebTargetSlot(
                        slot_id=slot_counter,
                        operation=WebOperationType.CLICK,
                        label=label,
                        role=role or "button",
                        criteria_key=f"CLICK:{slot_counter}",
                        description=f"Click button [{label}]",
                        bounding_box=node.bounding_box,
                        element_ref=node
                    )
                    targets.append(slot)
                    slot_counter += 1
                else:
                    slot = WebTargetSlot(
                        slot_id=slot_counter,
                        operation=WebOperationType.TYPE_TEXT,
                        label=label,
                        role=role or "input",
                        current_value=node.value or "",
                        criteria_key=f"TYPE_TEXT:{slot_counter}",
                        description=f"Type text into input [{label}]",
                        bounding_box=node.bounding_box,
                        element_ref=node
                    )
                    targets.append(slot)
                    slot_counter += 1

            elif tag == "select" or role == "combobox":
                if node.options:
                    for opt in node.options[:4]:  # limit candidate slots per select
                        slot = WebTargetSlot(
                            slot_id=slot_counter,
                            operation=WebOperationType.SELECT,
                            label=f"{label} -> {opt}",
                            role="select",
                            option_value=opt,
                            criteria_key=f"SELECT:{slot_counter}:{opt}",
                            description=f"Select option '{opt}' on [{label}]",
                            bounding_box=node.bounding_box,
                            element_ref=node
                        )
                        targets.append(slot)
                        slot_counter += 1
                else:
                    slot = WebTargetSlot(
                        slot_id=slot_counter,
                        operation=WebOperationType.SELECT,
                        label=label,
                        role="select",
                        criteria_key=f"SELECT:{slot_counter}",
                        description=f"Open/Select dropdown [{label}]",
                        bounding_box=node.bounding_box,
                        element_ref=node
                    )
                    targets.append(slot)
                    slot_counter += 1

            else:
                # Standard click (buttons, links, menu items)
                slot = WebTargetSlot(
                    slot_id=slot_counter,
                    operation=WebOperationType.CLICK,
                    label=label,
                    role=role or tag,
                    criteria_key=f"CLICK:{slot_counter}",
                    description=f"Click [{label}]",
                    bounding_box=node.bounding_box,
                    element_ref=node
                )
                targets.append(slot)
                slot_counter += 1

        # 3. Directional scrolling conditions
        can_scroll_up = (scroll_y > 10.0) or (offscreen_above > 0)
        can_scroll_down = (offscreen_below > 0) or (scroll_y < max_scroll_y - 10.0)

        if can_scroll_up:
            targets.append(WebTargetSlot(
                slot_id=slot_counter,
                operation=WebOperationType.SCROLL_UP,
                label="Scroll Up",
                criteria_key="SCROLL_UP",
                description=f"Scroll page upwards ({offscreen_above} offscreen controls above)"
            ))
            slot_counter += 1

        if can_scroll_down:
            targets.append(WebTargetSlot(
                slot_id=slot_counter,
                operation=WebOperationType.SCROLL_DOWN,
                label="Scroll Down",
                criteria_key="SCROLL_DOWN",
                description=f"Scroll page downwards ({offscreen_below} offscreen controls below)"
            ))
            slot_counter += 1

        # 4. Mandatory sentinel guardrails
        targets.append(WebTargetSlot(
            slot_id=slot_counter,
            operation=WebOperationType.WAIT,
            label="Wait",
            criteria_key="WAIT",
            description="Wait for dynamic network requests, hydration, or animations"
        ))
        slot_counter += 1

        targets.append(WebTargetSlot(
            slot_id=slot_counter,
            operation=WebOperationType.REVIEW,
            label="Review",
            criteria_key="REVIEW",
            description="Escalate for host/human review on payments, credentials, or destructive actions"
        ))
        slot_counter += 1

        targets.append(WebTargetSlot(
            slot_id=slot_counter,
            operation=WebOperationType.BLOCKED,
            label="Blocked",
            criteria_key="BLOCKED",
            description="Report task blocked due to fatal error, CAPTCHA, or inaccessible goal"
        ))
        slot_counter += 1

        targets.append(WebTargetSlot(
            slot_id=slot_counter,
            operation=WebOperationType.DONE,
            label="Done",
            criteria_key="DONE",
            description="Declare goal successfully accomplished and verified"
        ))

        from .signature import PageSignature
        sig_hash = PageSignature.compute(url, scroll_x, scroll_y, closure_nodes)

        observation = WebObservation(
            url=url,
            title=title,
            text_summary=page_summary or f"Page: {title} | Interactive controls: {len(closure_nodes)}",
            targets=targets,
            offscreen_above_count=offscreen_above,
            offscreen_below_count=offscreen_below,
            can_scroll_up=can_scroll_up,
            can_scroll_down=can_scroll_down,
            scroll_x=scroll_x,
            scroll_y=scroll_y,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            signature_hash=sig_hash,
            timestamp=timestamp
        )

        return observation, closure_nodes
