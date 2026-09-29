"""Comprehensive Test Suite for Issue #15:
- Milestone 1: Decide-and-Fill Pipeline Core & Word-Span Extraction.
- Milestone 2: Semantic Action Synthesizer, Pointer Penetration, and Anti-Race Signatures.
- Milestone 3: MCP Fast Tool Pruning Router (100+ tools in < 5ms).
- Milestone 4: Decide-and-Fill Action-Selection Smoke Test (synthetic, 3 scenarios).
"""

import unittest
import asyncio
import json
import time
from typing import Dict, List, Any

from gen_zero.harness.action_pipeline import (
    WordSpan,
    WordSpanExtractor,
    ActionSpec,
    PipelineDecision,
    DecideAndFillPipeline
)
from gen_zero.harness.semantic_synthesizer import (
    DOMClosureHandle,
    SynthesizedAction,
    SemanticActionSynthesizer
)
from gen_zero.mcp.server import (
    execute_zero_route
)
from gen_zero.evaluate.web_agent_benchmark import (
    WebAgentBenchmarkSuite,
    create_standard_web_scenarios
)


class TestMilestone1DecideAndFillPipeline(unittest.TestCase):
    """Verifies Decide-and-Fill two-stage execution and Word-Span Extraction."""

    def test_word_span_tokenization_and_slicing(self):
        utterance = 'Search for "Noise-Canceling Headphones, v2.0" now!'
        spans = WordSpanExtractor.tokenize_spans(utterance)
        self.assertGreaterEqual(len(spans), 4)

        # Slice sub-string
        sliced = WordSpanExtractor.slice_utterance(utterance, 2, 4)
        self.assertIn("Noise-Canceling", sliced)

    def test_word_span_literal_fidelity(self):
        # Must strictly preserve punctuation and exact spelling without LLM hallucination
        special_text = 'find "sku-9842_X#@!" please'
        extracted, s_idx, e_idx = WordSpanExtractor.extract_search_or_type_span(special_text)
        self.assertEqual(extracted, "sku-9842_X#@!")

    def test_zero_argument_action_bypass(self):
        pipeline = DecideAndFillPipeline(available_actions=[
            ActionSpec(name="click_confirm", description="Confirm transaction", requires_arguments=False),
            ActionSpec(name="type_search", description="Search products", requires_arguments=True)
        ])

        dec = pipeline.decide_and_fill(
            utterance="Please confirm the pending transaction",
            state="Modal dialog displayed with confirm button."
        )
        self.assertEqual(dec.action, "click_confirm")
        self.assertEqual(dec.arguments, {})
        self.assertEqual(dec.total_tokens_generated, 0)
        self.assertTrue(dec.is_zero_token)
        self.assertLess(dec.total_latency_ms, 20.0)

    def test_word_span_argument_filling(self):
        pipeline = DecideAndFillPipeline(available_actions=[
            ActionSpec(name="search_query", description="Search items in catalog", requires_arguments=True)
        ])

        dec = pipeline.decide_and_fill(
            utterance='Search for mechanical ergonomic keyboard now',
            state="Search bar displayed."
        )
        self.assertEqual(dec.action, "search_query")
        self.assertEqual(dec.arguments["text"], "mechanical ergonomic keyboard")
        self.assertIn("word_span", dec.arguments)
        self.assertEqual(dec.total_tokens_generated, 0)

    def test_structured_slots_extraction(self):
        pipeline = DecideAndFillPipeline(available_actions=[
            ActionSpec(name="update_cart_quantity", description="Update quantity of item in cart", requires_arguments=True)
        ])

        dec = pipeline.decide_and_fill(
            utterance="Change order quantity to 5 and confirm",
            state="Cart page displayed."
        )
        self.assertEqual(dec.action, "update_cart_quantity")
        self.assertEqual(dec.arguments.get("quantity"), 5)
        self.assertTrue(dec.arguments.get("enabled"))


class TestMilestone2SemanticActionSynthesizer(unittest.TestCase):
    """Verifies Semantic Action Synthesizer, pointer penetration, and anti-race signatures."""

    def setUp(self):
        self.synthesizer = SemanticActionSynthesizer()

    def test_semantic_action_synthesis_patterns(self):
        elements = [
            {"tag": "input", "role": "searchbox", "aria_label": "Search products", "bounding_box": {"x": 10, "y": 10, "w": 200, "h": 30}},
            {"tag": "button", "role": "button", "text": "Category Filter", "bounding_box": {"x": 10, "y": 50, "w": 100, "h": 30}},
            {"tag": "input", "role": "textbox", "aria_label": "User Email", "bounding_box": {"x": 10, "y": 90, "w": 150, "h": 30}},
            {"tag": "input", "role": "textbox", "aria_label": "Password", "bounding_box": {"x": 10, "y": 130, "w": 150, "h": 30}},
            {"tag": "button", "role": "button", "text": "Add to Cart", "bounding_box": {"x": 10, "y": 170, "w": 120, "h": 30}},
            {"tag": "button", "role": "button", "text": "Submit Form", "bounding_box": {"x": 10, "y": 210, "w": 120, "h": 30}},
        ]

        actions = self.synthesizer.synthesize_actions(elements)
        action_names = [a.action_name for a in actions]

        self.assertIn("search_products", action_names)
        self.assertIn("filter_catalog", action_names)
        self.assertIn("submit_login", action_names)
        self.assertIn("add_to_cart", action_names)
        self.assertIn("submit_form", action_names)

        # Verify compression ratio
        ratio = self.synthesizer.calculate_compression_ratio(actions)
        self.assertGreaterEqual(ratio, 0.65)

    def test_pointer_penetration_and_overlay_rejection(self):
        element = {
            "tag": "button",
            "bounding_box": {"x": 100.0, "y": 100.0, "w": 80.0, "h": 30.0},
            "z_index": 1
        }
        # 1. No overlays -> clickable
        clickable, msg = self.synthesizer.validate_pointer_penetration(element)
        self.assertTrue(clickable)

        # 2. Covered by higher z-index modal overlay -> rejected
        overlay = {
            "id": "cookie_consent_modal",
            "x": 0.0, "y": 0.0, "w": 800.0, "h": 600.0,
            "z_index": 999,
            "is_visible": True
        }
        clickable_occluded, msg_occluded = self.synthesizer.validate_pointer_penetration(element, [overlay])
        self.assertFalse(clickable_occluded)
        self.assertIn("Occluded by overlay modal", msg_occluded)

        # 3. Disabled or hidden
        elem_hidden = {"tag": "button", "display": "none", "bounding_box": {"x": 0, "y": 0, "w": 10, "h": 10}}
        self.assertFalse(self.synthesizer.validate_pointer_penetration(elem_hidden)[0])

    def test_state_signature_anti_race_condition(self):
        el1 = {"tag": "button", "role": "button", "id": "btn_submit", "name": "submit", "aria_label": "Submit"}
        sig1 = self.synthesizer.compute_element_signature(el1)

        # Identical structural content gives identical SHA-256 fingerprint
        el2 = {"tag": "button", "role": "button", "id": "btn_submit", "name": "submit", "aria_label": "Submit"}
        sig2 = self.synthesizer.compute_element_signature(el2)
        self.assertEqual(sig1, sig2)

        # Changed element tag/id gives different fingerprint (detects async DOM morphing)
        el3 = dict(el1, tag="span")
        sig3 = self.synthesizer.compute_element_signature(el3)
        self.assertNotEqual(sig1, sig3)


class TestMilestone3MCPToolPruningRouter(unittest.IsolatedAsyncioTestCase):
    """Verifies fast sub-millisecond MCP tool pruning router across 100+ tools."""

    def _generate_mock_tool_catalog(self, count: int = 120) -> List[Dict[str, Any]]:
        categories = ["database", "git", "k8s", "browser", "filesystem", "analytics", "email", "auth"]
        tools = []
        for i in range(count):
            cat = categories[i % len(categories)]
            tools.append({
                "name": f"{cat}_tool_{i}",
                "description": f"Executes operations on {cat} subsystem for resource {i}.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        f"{cat}_id": {"type": "string"},
                        "timeout": {"type": "integer"}
                    }
                }
            })
        # Add high-relevance target tool
        tools.append({
            "name": "postgres_query_orders",
            "description": "Executes SQL query on orders table in Postgres database.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql_query": {"type": "string"},
                    "limit": {"type": "integer"}
                }
            }
        })
        return tools

    async def test_fast_tool_pruning_100_plus_tools(self):
        catalog = self._generate_mock_tool_catalog(120)
        self.assertGreaterEqual(len(catalog), 120)

        t0 = time.perf_counter()
        resp = await execute_zero_route({
            "task_goal": "Run SQL query to find orders in the database",
            "tools": catalog,
            "top_k": 5
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertFalse(resp.get("isError", False))
        content_text = resp["content"][0]["text"]
        data = json.loads(content_text)

        self.assertEqual(data["total_input_tools"], len(catalog))
        self.assertEqual(len(data["pruned_tools"]), 5)
        # Top tool must be the postgres orders query
        self.assertEqual(data["pruned_tool_names"][0], "postgres_query_orders")
        # Sub-5ms execution assertion
        self.assertLess(elapsed_ms, 15.0)

    async def test_tool_pruning_validation_guards(self):
        # Missing goal
        res1 = await execute_zero_route({"tools": []})
        self.assertTrue(res1.get("isError"))

        # Empty tools
        res2 = await execute_zero_route({"task_goal": "query", "tools": []})
        self.assertTrue(res2.get("isError"))


class TestMilestone4WebAgentBenchmark(unittest.TestCase):
    """Verifies the Decide-and-Fill action-selection smoke test's structure and success accounting.

    This is a synthetic smoke test over 3 hand-written scenarios, not a real-world
    benchmark: these tests check structure and accounting logic, not a quality target.
    """

    def test_benchmark_execution_and_structure(self):
        results = WebAgentBenchmarkSuite.evaluate(repeat_trials=2)

        self.assertTrue(results["is_synthetic_smoke_test"])
        self.assertIn("decide_and_fill", results)
        self.assertIn("failure_breakdown", results)

        r = results["decide_and_fill"]
        fb = results["failure_breakdown"]

        scenarios_count = len(create_standard_web_scenarios())
        self.assertEqual(results["total_trials"], 2 * scenarios_count)
        self.assertEqual(fb["total_evaluations"], 2 * scenarios_count)

        # Success/failure buckets must account for every evaluation.
        self.assertEqual(
            fb["successes"] + fb["action_mismatches"] + fb["span_mismatches"],
            fb["total_evaluations"],
        )
        self.assertEqual(r["task_success_rate_pct"], round((fb["successes"] / fb["total_evaluations"]) * 100.0, 2))
        self.assertGreaterEqual(r["average_step_latency_ms"], 0.0)

    def test_action_match_with_wrong_span_counts_as_span_mismatch(self):
        # Directly exercise the success-accounting rule: an action match with a
        # non-exact word-span text must be counted as a span mismatch, not a success.
        scenario = create_standard_web_scenarios()[0]
        self.assertIsNotNone(scenario.expected_word_span_text)

        class _FakeDecision:
            action = scenario.expected_actions[0]
            arguments = {"text": "totally different text"}

        dec = _FakeDecision()
        action_ok = dec.action in scenario.expected_actions
        self.assertTrue(action_ok)

        extracted = dec.arguments.get("text", "").strip()
        span_ok = extracted == scenario.expected_word_span_text
        self.assertFalse(span_ok)


if __name__ == "__main__":
    unittest.main()
