"""Page State Snapshot Signature and Anti-Race Assertions (Issue #34 & RFC-034).

Provides:
1. Multi-dimensional page state fingerprinting (URL, scroll offsets, DOM closure node attributes).
2. assert_fresh: Pre-action anti-race assertion preventing blind mutations on stale SPA renders.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib
import json
from .types import DOMElement, StaleObservationError, WebObservation


class PageSignature:
    """Generates and asserts deterministic perceptual hashes of browser state."""

    @classmethod
    def compute(
        cls,
        url: str,
        scroll_x: float,
        scroll_y: float,
        nodes: Sequence[DOMElement]
    ) -> str:
        """Computes a 16-character SHA-256 hash of the page URL, viewport scroll, and closure node attributes."""
        canonical_state = {
            "url": str(url).strip(),
            "scroll_x": round(float(scroll_x), 1),
            "scroll_y": round(float(scroll_y), 1),
            "nodes": [node.signature_tuple() for node in nodes]
        }
        raw_json = json.dumps(canonical_state, sort_keys=True, default=str)
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def assert_fresh(
        cls,
        expected_signature: str,
        current_url: str,
        current_scroll_x: float,
        current_scroll_y: float,
        current_nodes: Sequence[DOMElement]
    ) -> bool:
        """Asserts that the current DOM and page state strictly match expected signature.

        Raises:
            StaleObservationError: If URL, scroll, node attributes, or DOM connection status changed.
        """
        # Check node connection integrity first
        for idx, node in enumerate(current_nodes):
            if not node.is_connected:
                raise StaleObservationError(
                    expected_hash=expected_signature,
                    actual_hash="DISCONNECTED",
                    reason=f"Target node at slot {idx} (id={node.element_id}) is disconnected from DOM"
                )

        current_sig = cls.compute(current_url, current_scroll_x, current_scroll_y, current_nodes)
        if current_sig != expected_signature:
            raise StaleObservationError(
                expected_hash=expected_signature,
                actual_hash=current_sig,
                reason="Page signature mismatch due to asynchronous SPA DOM/scroll re-render"
            )

        return True

    @classmethod
    def assert_fresh_observation(
        cls,
        expected_signature: str,
        current_observation: WebObservation,
        current_nodes: Sequence[DOMElement]
    ) -> bool:
        """Asserts signature freshness using a pre-computed WebObservation."""
        # Check node connection integrity first
        for idx, node in enumerate(current_nodes):
            if not node.is_connected:
                raise StaleObservationError(
                    expected_hash=expected_signature,
                    actual_hash="DISCONNECTED",
                    reason=f"Target node at slot {idx} (id={node.element_id}) is disconnected from DOM"
                )

        if current_observation.signature_hash != expected_signature:
            raise StaleObservationError(
                expected_hash=expected_signature,
                actual_hash=current_observation.signature_hash,
                reason="Page observation signature hash does not match expected state"
            )

        return True
