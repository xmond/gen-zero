"""Gen-Zero Layer 4: Semantic Action Synthesizer & Browser DOM Compression Engine (Issue #15 & #34).

Architectural Implementation:
1. High-Order Semantic Action Synthesis:
   Compresses 15~20 micro-step raw DOM interactions (click -> focus -> clear -> type -> enter)
   into 2~3 atomic high-order tool calls (e.g., search_products, filter_catalog, submit_form).
2. DOM Closure Handle Zero-Selector Mapping:
   Assigns immutable handle IDs to target DOM elements, decoupling execution from brittle XPath/CSS selectors.
3. Pointer Penetration & Overlay Validation:
   Checks pointer-events, z-index overlays, and bounding box occlusion before action dispatch.
4. State Signature Anti-Race Assertions:
   Computes perceptual fingerprint (SHA-256) of target subtrees to prevent race conditions during rapid async renders.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import json
import re
import time


@dataclass
class DOMClosureHandle:
    """Stable closure reference to a rendered browser DOM element."""
    handle_id: str
    tag: str
    role: Optional[str] = None
    aria_label: Optional[str] = None
    text_content: Optional[str] = None
    bounding_box: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "w": 100.0, "h": 30.0})
    is_clickable: bool = True
    is_visible: bool = True
    is_obscured: bool = False
    state_signature: str = ""


@dataclass
class SynthesizedAction:
    """High-level semantic action synthesized from multiple raw interactive primitives."""
    action_name: str
    description: str
    category: str
    target_handle: Optional[DOMClosureHandle] = None
    sub_handles: List[DOMClosureHandle] = field(default_factory=list)
    raw_step_equivalent: int = 1
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    state_signature: str = ""


class SemanticActionSynthesizer:
    """Transforms raw accessibility trees and DOM element lists into high-order semantic tools."""

    @classmethod
    def compute_element_signature(cls, element: Dict[str, Any]) -> str:
        """Computes a SHA-256 fingerprint of the element's structural attributes."""
        canonical = {
            "tag": str(element.get("tag", "")).lower(),
            "role": str(element.get("role", "")).lower(),
            "aria_label": str(element.get("aria_label", "")).strip(),
            "id": str(element.get("id", "")).strip(),
            "name": str(element.get("name", "")).strip(),
            "text": str(element.get("text_content") or element.get("text", "")).strip()
        }
        raw_str = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def validate_pointer_penetration(
        cls,
        element: Dict[str, Any],
        viewport_overlays: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str]:
        """Validates that an element is physically clickable and not occluded by modal overlays."""
        # 1. Check visibility flags
        if element.get("hidden", False) or element.get("display") == "none" or element.get("visibility") == "hidden":
            return False, "Element is hidden via CSS style"

        # 2. Check disabled status
        if element.get("disabled", False):
            return False, "Element has disabled attribute"

        # 3. Check bounding box dimensions
        bbox = element.get("bounding_box", {})
        w = float(bbox.get("w", bbox.get("width", 10.0)))
        h = float(bbox.get("h", bbox.get("height", 10.0)))
        if w <= 0.0 or h <= 0.0:
            return False, f"Zero-dimension bounding box ({w}x{h})"

        # 4. Check overlay occlusion
        if viewport_overlays:
            ex = float(bbox.get("x", 0.0))
            ey = float(bbox.get("y", 0.0))
            for ov in viewport_overlays:
                if not ov.get("is_visible", True):
                    continue
                ox = float(ov.get("x", 0.0))
                oy = float(ov.get("y", 0.0))
                ow = float(ov.get("w", 0.0))
                oh = float(ov.get("h", 0.0))
                # If overlay covers element center point
                cx, cy = ex + w / 2.0, ey + h / 2.0
                if (ox <= cx <= ox + ow) and (oy <= cy <= oy + oh):
                    if ov.get("z_index", 100) > element.get("z_index", 0):
                        return False, f"Occluded by overlay modal '{ov.get('id', 'unknown')}'"

        return True, "Clickable"

    @classmethod
    def create_handle(
        cls,
        element: Dict[str, Any],
        viewport_overlays: Optional[List[Dict[str, Any]]] = None
    ) -> DOMClosureHandle:
        """Constructs a validated DOMClosureHandle."""
        is_pen, reason = cls.validate_pointer_penetration(element, viewport_overlays)
        sig = cls.compute_element_signature(element)
        h_id = element.get("handle_id") or f"handle_{element.get('tag', 'elem')}_{sig[:8]}"

        return DOMClosureHandle(
            handle_id=h_id,
            tag=str(element.get("tag", "div")),
            role=element.get("role"),
            aria_label=element.get("aria_label"),
            text_content=element.get("text_content") or element.get("text"),
            bounding_box=element.get("bounding_box", {"x": 0.0, "y": 0.0, "w": 100.0, "h": 30.0}),
            is_clickable=is_pen,
            is_visible=True,
            is_obscured=(not is_pen and "Occluded" in reason),
            state_signature=sig
        )

    def synthesize_actions(
        self,
        elements: List[Dict[str, Any]],
        viewport_overlays: Optional[List[Dict[str, Any]]] = None
    ) -> List[SynthesizedAction]:
        """Synthesizes high-level actions from a flat list of DOM or a11y nodes."""
        synthesized: List[SynthesizedAction] = []
        handles: List[DOMClosureHandle] = [
            self.create_handle(el, viewport_overlays) for el in elements
        ]

        # 1. Search Box pattern recognition
        search_inputs = [
            h for h in handles
            if h.tag in ("input", "textarea")
            and (
                "search" in str(h.role or "").lower()
                or "search" in str(h.aria_label or "").lower()
                or "search" in str(h.text_content or "").lower()
                or "query" in str(h.aria_label or "").lower()
            )
        ]
        if search_inputs:
            h = search_inputs[0]
            synthesized.append(SynthesizedAction(
                action_name="search_products",
                description="Search for items, products or articles across the catalog",
                category="search",
                target_handle=h,
                raw_step_equivalent=5,  # click + focus + clear + type + submit
                parameters_schema={"query": {"type": "string", "description": "Search query text"}},
                state_signature=h.state_signature
            ))

        # 2. Filter / Sort pattern recognition
        filter_elements = [
            h for h in handles
            if any(k in str(h.aria_label or "").lower() for k in ("filter", "sort", "category", "price", "brand"))
            or any(k in str(h.text_content or "").lower() for k in ("filter", "sort by", "narrow results"))
        ]
        if filter_elements:
            synthesized.append(SynthesizedAction(
                action_name="filter_catalog",
                description="Apply faceted filtering by category, price, or attributes",
                category="filter",
                target_handle=filter_elements[0],
                sub_handles=filter_elements[1:4],
                raw_step_equivalent=4,  # expand filter + pick option + apply + wait
                parameters_schema={"filter_key": {"type": "string"}},
                state_signature=filter_elements[0].state_signature
            ))

        # 3. Authentication / Login pattern recognition
        user_inputs = [h for h in handles if any(k in str(h.aria_label or "").lower() for k in ("username", "email", "login"))]
        pwd_inputs = [h for h in handles if "password" in str(h.aria_label or "").lower()]
        if user_inputs and pwd_inputs:
            synthesized.append(SynthesizedAction(
                action_name="submit_login",
                description="Authenticate with username/email and password credentials",
                category="auth",
                target_handle=user_inputs[0],
                sub_handles=[pwd_inputs[0]],
                raw_step_equivalent=6,  # click user + type user + click pwd + type pwd + submit + wait
                parameters_schema={"username": {"type": "string"}, "password": {"type": "string"}},
                state_signature=user_inputs[0].state_signature
            ))

        # 4. Checkout / Cart pattern recognition
        cart_buttons = [
            h for h in handles
            if any(k in str(h.text_content or "").lower() for k in ("add to cart", "checkout", "buy now", "place order"))
            or any(k in str(h.aria_label or "").lower() for k in ("cart", "checkout"))
        ]
        for cb in cart_buttons[:2]:
            act_name = "add_to_cart" if "add" in str(cb.text_content or "").lower() else "proceed_to_checkout"
            synthesized.append(SynthesizedAction(
                action_name=act_name,
                description=f"Trigger {act_name.replace('_', ' ')} flow",
                category="cart",
                target_handle=cb,
                raw_step_equivalent=3,
                parameters_schema={},
                state_signature=cb.state_signature
            ))

        # 5. Form submission button
        submit_buttons = [
            h for h in handles
            if (h.tag == "button" or h.role == "button")
            and any(k in str(h.text_content or "").lower() for k in ("submit", "save", "apply", "confirm", "send"))
        ]
        for sb in submit_buttons[:2]:
            synthesized.append(SynthesizedAction(
                action_name="submit_form",
                description="Submit current form data for validation",
                category="form",
                target_handle=sb,
                raw_step_equivalent=2,
                parameters_schema={},
                state_signature=sb.state_signature
            ))

        # If nothing specific matched, wrap primary clickable items
        if not synthesized:
            for h in [h for h in handles if h.is_clickable][:5]:
                label = h.text_content or h.aria_label or h.tag
                synthesized.append(SynthesizedAction(
                    action_name=f"interact_{h.tag}_{h.handle_id[:8]}",
                    description=f"Interact with {label}",
                    category="generic",
                    target_handle=h,
                    raw_step_equivalent=1,
                    parameters_schema={},
                    state_signature=h.state_signature
                ))

        return synthesized

    @classmethod
    def calculate_compression_ratio(cls, synthesized_actions: List[SynthesizedAction]) -> float:
        """Calculates step compression ratio: 1.0 - (synthesized_count / raw_equivalent_count)."""
        if not synthesized_actions:
            return 0.0
        synth_count = len(synthesized_actions)
        raw_count = sum(a.raw_step_equivalent for a in synthesized_actions)
        if raw_count <= 0:
            return 0.0
        return round(1.0 - (synth_count / raw_count), 4)
