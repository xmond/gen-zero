"""Unit and Integration Tests for Zero Browser Harness & DOM Safety (Issue #34 & RFC-034).

Tests coverage:
1. DOM Closure Handle Zero-Selector Mapping and Injection Security.
2. Pointer Penetration Hit-Testing and Modal Backdrop Occlusion.
3. Offscreen Controls Directional Sensing and Deterministic Scrolling.
4. Page State Snapshot Fingerprinting and assert_fresh Anti-Race Assertions.
5. Two-Stage Decoupled Execution and Idempotent TextCache.
6. High-Priority Safety Sentinels (REVIEW, BLOCKED, DONE).
7. Visual Trajectory Recorder with Blue Cursor & Red Ripple Overlay.
8. End-to-End Autonomous Browser Execution Loop.
9. GenZero Client Browser Convenience Methods.
"""

import unittest
import time
import json
import xml.etree.ElementTree as ET

from gen_zero.client import GenZero
from gen_zero.harness.web import (
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
    PointerPenetrationChecker,
    OffscreenDetector,
    DOMObserver,
    PageSignature,
    ClosureHandlePool,
    TextCache,
    VisualEvent,
    VisualTrajectoryRecorder,
    ZeroBrowserHarness,
)


class TestIssue34BrowserHarness(unittest.TestCase):

    def setUp(self):
        self.client = GenZero()
        self.harness = ZeroBrowserHarness(
            url="https://store.example.com/checkout",
            title="Secure Checkout",
            viewport_width=1280.0,
            viewport_height=800.0,
            decision_client=self.client
        )

    def test_closure_retained_handles_zero_selector(self):
        """Test Module 1: Zero-Selector contract and injection prevention."""
        click_fired = [False]
        def on_btn_click():
            click_fired[0] = True

        el1 = DOMElement(
            element_id="search_box",
            tag="input",
            role="searchbox",
            bounding_box=BoundingBox(100, 50, 200, 35),
            attributes={"type": "text"}
        )
        el2 = DOMElement(
            element_id="submit_btn",
            tag="button",
            role="button",
            text_content="Search Now",
            bounding_box=BoundingBox(320, 50, 100, 35),
            on_click=on_btn_click
        )

        obs = self.harness.observe(elements=[el1, el2])
        criteria = obs.get_criteria_dict()

        # 1. Verify candidate keys are purely discrete tokens without selectors
        self.assertIn("TYPE_TEXT:0", criteria)
        self.assertIn("CLICK:1", criteria)
        for key in criteria.keys():
            self.assertNotIn("input[", key)
            self.assertNotIn("button#", key)
            self.assertNotIn("//", key)

        # 2. Verify dispatch uses native closure hook without selectors
        res = self.harness.step("CLICK:1")
        self.assertTrue(res.success)
        self.assertTrue(click_fired[0])

        # 3. Verify CSS / XPath / Script injection triggers HarnessSecurityException
        with self.assertRaises(HarnessSecurityException):
            self.harness.step("button.danger > div")

        with self.assertRaises(HarnessSecurityException):
            self.harness.step("//input[@id='password']")

        with self.assertRaises(HarnessSecurityException):
            self.harness.step("document.cookie")

    def test_pointer_penetration_hit_test(self):
        """Test Module 2: Centroid hit-testing and modal backdrop occlusion."""
        # Main page button underneath backdrop
        background_button = DOMElement(
            element_id="pay_button",
            tag="button",
            role="button",
            text_content="Pay Now",
            bounding_box=BoundingBox(500, 400, 120, 40),
            z_index=1
        )
        # Fullscreen modal backdrop overlay
        cookie_backdrop = ModalOverlay(
            overlay_id="cookie_modal_backdrop",
            bounding_box=BoundingBox(0, 0, 1280, 800),
            z_index=100,
            is_visible=True,
            dismiss_button_id="accept_cookie_btn"
        )
        # Dismiss button on the modal itself
        modal_dismiss_btn = DOMElement(
            element_id="accept_cookie_btn",
            tag="button",
            role="button",
            text_content="Accept All Cookies",
            bounding_box=BoundingBox(580, 450, 150, 40),
            z_index=101
        )

        # Direct pointer penetration check
        can_click_bg, reason_bg = PointerPenetrationChecker.receives_pointer(
            background_button, [cookie_backdrop]
        )
        self.assertFalse(can_click_bg)
        self.assertIn("Occluded by overlay", reason_bg)

        can_click_dismiss, reason_dismiss = PointerPenetrationChecker.receives_pointer(
            modal_dismiss_btn, [cookie_backdrop]
        )
        self.assertTrue(can_click_dismiss)

        # In DOMObserver, background button must NOT be added to closure candidates
        obs = self.harness.observe(
            elements=[background_button, modal_dismiss_btn],
            overlays=[cookie_backdrop]
        )
        target_labels = [t.label for t in obs.targets]
        self.assertIn("Accept All Cookies", target_labels)
        self.assertNotIn("Pay Now", target_labels)

    def test_offscreen_controls_directional_sensing(self):
        """Test Module 3: Offscreen control sensing and directional scroll activation."""
        in_view_el = DOMElement(
            element_id="field1",
            tag="input",
            bounding_box=BoundingBox(100, 200, 300, 40)
        )
        above_el = DOMElement(
            element_id="nav_bar_link",
            tag="a",
            text_content="Home",
            bounding_box=BoundingBox(50, -60, 80, 30)  # y < 0 (above viewport)
        )
        below_el1 = DOMElement(
            element_id="submit_footer",
            tag="button",
            text_content="Save Form",
            bounding_box=BoundingBox(100, 950, 150, 40)  # y > 800 (below viewport)
        )
        below_el2 = DOMElement(
            element_id="agree_checkbox",
            tag="input",
            bounding_box=BoundingBox(100, 1050, 20, 20),
            attributes={"type": "checkbox"}
        )

        obs = self.harness.observe(
            elements=[in_view_el, above_el, below_el1, below_el2],
            scroll_x=0.0,
            scroll_y=100.0,
            max_scroll_y=1500.0
        )

        self.assertEqual(obs.offscreen_above_count, 1)
        self.assertEqual(obs.offscreen_below_count, 2)
        self.assertTrue(obs.can_scroll_up)
        self.assertTrue(obs.can_scroll_down)

        criteria = obs.get_criteria_dict()
        self.assertIn("SCROLL_UP", criteria)
        self.assertIn("SCROLL_DOWN", criteria)
        self.assertIn("1 offscreen controls above", criteria["SCROLL_UP"])
        self.assertIn("2 offscreen controls below", criteria["SCROLL_DOWN"])

        # Stepping SCROLL_DOWN adjusts scroll position
        initial_scroll = self.harness.scroll_y
        step_res = self.harness.step("SCROLL_DOWN")
        self.assertTrue(step_res.success)
        self.assertGreater(self.harness.scroll_y, initial_scroll)

    def test_page_signature_anti_race_and_recovery(self):
        """Test Module 4: Page signature computation, assert_fresh, and SPA race recovery."""
        el = DOMElement(
            element_id="btn1",
            tag="button",
            text_content="Continue",
            bounding_box=BoundingBox(100, 100, 100, 40)
        )
        obs = self.harness.observe(elements=[el], url="https://app.io/step1", scroll_y=0.0)
        original_hash = obs.signature_hash
        self.assertEqual(len(original_hash), 16)

        # Freshness check passes when identical
        self.assertTrue(PageSignature.assert_fresh(
            expected_signature=original_hash,
            current_url="https://app.io/step1",
            current_scroll_x=0.0,
            current_scroll_y=0.0,
            current_nodes=[el]
        ))

        # 1. URL change breaks signature
        with self.assertRaises(StaleObservationError):
            PageSignature.assert_fresh(
                expected_signature=original_hash,
                current_url="https://app.io/step2",
                current_scroll_x=0.0,
                current_scroll_y=0.0,
                current_nodes=[el]
            )

        # 2. Disconnected node raises StaleObservationError
        el.is_connected = False
        with self.assertRaises(StaleObservationError):
            PageSignature.assert_fresh(
                expected_signature=original_hash,
                current_url="https://app.io/step1",
                current_scroll_x=0.0,
                current_scroll_y=0.0,
                current_nodes=[el]
            )
        el.is_connected = True

        # 3. In-flight mutation during harness.step triggers automatic re-observation recovery
        # Manually alter scroll_y to simulate unexpected asynchronous scroll
        self.harness.scroll_y = 350.0
        step_res = self.harness.step("CLICK:0", assert_freshness=True)
        self.assertTrue(step_res.stale_recovered)
        self.assertTrue(step_res.success)

    def test_two_stage_decoupled_execution_and_text_cache(self):
        """Test Module 5: Two-stage parameter resolution with WordSpan & TextCache."""
        text_input = DOMElement(
            element_id="search_query",
            tag="input",
            role="searchbox",
            bounding_box=BoundingBox(100, 100, 250, 40)
        )
        obs = self.harness.observe(elements=[text_input])
        slot_obj = obs.targets[0]
        self.assertEqual(slot_obj.criteria_key, "TYPE_TEXT:0")

        cache = TextCache()
        goal = "Please search for 'autonomous world model' in the catalog"

        # First resolution extracts span deterministically without autoregression (0 tokens)
        resolved_text = cache.resolve_text_parameter(
            goal=goal,
            slot=slot_obj,
            signature_hash=obs.signature_hash
        )
        self.assertEqual(resolved_text, "autonomous world model")

        # Second resolution hits cache directly
        cached_text = cache.get(obs.signature_hash, goal, slot_obj.slot_id)
        self.assertEqual(cached_text, "autonomous world model")

        # Explicit parameter override
        override_text = cache.resolve_text_parameter(
            goal=goal,
            slot=slot_obj,
            signature_hash=obs.signature_hash,
            explicit_param="force override query"
        )
        self.assertEqual(override_text, "force override query")

    def test_safety_guardrails_sentinels(self):
        """Test Module 6: REVIEW, BLOCKED, and DONE sentinel guardrails."""
        el = DOMElement(
            element_id="confirm",
            tag="button",
            text_content="Confirm",
            bounding_box=BoundingBox(100, 100, 100, 40)
        )
        self.harness.observe(elements=[el])

        # 1. REVIEW sentinel yields control to host
        review_res = self.harness.step("REVIEW")
        self.assertTrue(review_res.is_terminal)
        self.assertEqual(review_res.terminal_reason, "REVIEW")

        # 2. BLOCKED sentinel terminates on fatal obstacle
        blocked_res = self.harness.step("BLOCKED")
        self.assertTrue(blocked_res.is_terminal)
        self.assertEqual(blocked_res.terminal_reason, "BLOCKED")

        # 3. DONE sentinel declares task completion
        done_res = self.harness.step("DONE")
        self.assertTrue(done_res.is_terminal)
        self.assertEqual(done_res.terminal_reason, "DONE")

    def test_visual_trajectory_overlay_and_export(self):
        """Test Module 6: Blue cursor, red ripple, SVG overlay, and JSONL audit trail."""
        recorder = VisualTrajectoryRecorder()
        recorder.record(150.0, 200.0, "CLICK", slot_id=0, label="Search Button")
        recorder.record(150.0, 280.0, "TYPE_TEXT", slot_id=1, label="Query Box")
        recorder.record(640.0, 400.0, "DONE", label="Task Complete")

        # 1. JSONL Export
        jsonl = recorder.export_jsonl()
        lines = jsonl.strip().split("\n")
        self.assertEqual(len(lines), 3)
        ev1 = json.loads(lines[0])
        self.assertEqual(ev1["x"], 150.0)
        self.assertEqual(ev1["y"], 200.0)
        self.assertEqual(ev1["cursor_color"], "#2563eb")
        self.assertEqual(ev1["ripple_color"], "#ef4444")

        # 2. SVG Overlay Export
        svg = recorder.export_svg_overlay(1280.0, 800.0)
        self.assertIn('<svg xmlns="http://www.w3.org/2000/svg"', svg)
        self.assertIn('.cursor { fill: #2563eb;', svg)
        self.assertIn('.ripple { fill: none; stroke: #ef4444;', svg)
        self.assertIn('<polyline class="path-line"', svg)
        self.assertIn('#1 CLICK [Search Button]', svg)

        # 3. Audit Report
        report = recorder.export_verification_report()
        self.assertEqual(report["total_steps"], 3)
        self.assertEqual(report["operations_breakdown"]["CLICK"], 1)
        self.assertEqual(report["operations_breakdown"]["TYPE_TEXT"], 1)
        self.assertEqual(report["operations_breakdown"]["DONE"], 1)

    def test_end_to_end_autonomous_browser_loop(self):
        """End-to-End Test: Autonomous multi-step session with closed-loop termination."""
        query_input = DOMElement(
            element_id="search_box",
            tag="input",
            role="searchbox",
            bounding_box=BoundingBox(200, 100, 300, 40)
        )
        submit_btn = DOMElement(
            element_id="search_btn",
            tag="button",
            role="button",
            text_content="Search",
            bounding_box=BoundingBox(520, 100, 100, 40)
        )

        harness = ZeroBrowserHarness(
            url="https://books.example.com",
            title="Book Store",
            decision_client=self.client
        )
        harness.observe(elements=[query_input, submit_btn])

        # Step 1: Type query
        res1 = harness.step("TYPE_TEXT:0", text_param="Clean Architecture")
        self.assertTrue(res1.success)
        self.assertEqual(query_input.value, "Clean Architecture")

        # Step 2: Click submit
        res2 = harness.step("CLICK:1")
        self.assertTrue(res2.success)

        # Step 3: Complete task
        res3 = harness.step("DONE")
        self.assertTrue(res3.is_terminal)
        self.assertEqual(res3.terminal_reason, "DONE")

        report = harness.recorder.export_verification_report()
        self.assertEqual(report["total_steps"], 3)

    def test_client_browser_convenience_methods(self):
        """Test GenZero client convenience methods for browser automation."""
        client = GenZero()
        harness = client.create_browser_harness(url="https://demo.com", title="Demo")
        self.assertIsInstance(harness, ZeroBrowserHarness)

        btn = DOMElement(
            element_id="login_btn",
            tag="button",
            text_content="Login",
            bounding_box=BoundingBox(100, 100, 80, 30)
        )
        obs = client.observe_browser(elements=[btn], url="https://demo.com", title="Demo")
        self.assertIsInstance(obs, WebObservation)
        self.assertIn("CLICK:0", obs.get_criteria_dict())

        step_res = client.step_browser("CLICK:0")
        self.assertTrue(step_res.success)
        self.assertEqual(step_res.operation, WebOperationType.CLICK)


if __name__ == "__main__":
    unittest.main()
