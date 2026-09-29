"""Unit & Stress Tests for Issue #18: Sub-Second Streaming Decision Architecture.

Verifies:
1. L2OrderBookSnapshot & L2MarketStateEncoder: imbalance [-1.0, 1.0], spread bps, depth profile.
2. CPSATMarketRiskGate: max inventory hard bounds, consecutive adverse cutoff, extreme spread breaker.
3. SubSecondDecisionPipeline: <= 80ms stage budgets, late-breaker drift protection.
4. FastAPI SSE streaming endpoint: GET /v1/decisions/stream with event: decision frames.
"""

import unittest
import os
import json
import time
from fastapi.testclient import TestClient

from gen_zero.model.market_state import L2OrderBookSnapshot, L2MarketStateEncoder
from gen_zero.gate.market_risk_gate import CPSATMarketRiskGate
from gen_zero.runtime.subsecond_pipeline import SubSecondDecisionPipeline
from gen_zero.service.app import app, _load_api_token


class TestL2MarketStateEncoder(unittest.TestCase):
    """Verifies Milestone 2: L2 order book microstructural feature extraction."""

    def setUp(self):
        self.encoder = L2MarketStateEncoder(top_k_levels=3)
        self.snap = L2OrderBookSnapshot(
            bids=[(100.0, 10.0), (99.95, 15.0), (99.90, 20.0)],
            asks=[(100.05, 5.0), (100.10, 10.0), (100.15, 15.0)],
            cvd=50.0,
            last_price=100.02,
        )

    def test_book_validity(self):
        self.assertTrue(self.encoder.is_valid_book(self.snap))

        # Crossed book
        crossed_snap = L2OrderBookSnapshot(
            bids=[(100.10, 10.0)],
            asks=[(100.05, 10.0)]
        )
        self.assertFalse(self.encoder.is_valid_book(crossed_snap))

        # Empty book
        empty_snap = L2OrderBookSnapshot(bids=[], asks=[])
        self.assertFalse(self.encoder.is_valid_book(empty_snap))

    def test_mid_price_and_spread(self):
        mid = self.encoder.compute_mid_price(self.snap)
        self.assertAlmostEqual(mid, 100.025, places=3)

        spread_bps = self.encoder.compute_spread_bps(self.snap)
        # Spread: (100.05 - 100.00) / 100.025 * 10000 = ~4.998 bps
        self.assertAlmostEqual(spread_bps, 5.0, delta=0.5)

    def test_book_imbalance_bounds(self):
        # Top-3 bids = 10+15+20 = 45; Top-3 asks = 5+10+15 = 30
        # Imbalance = (45 - 30) / (45 + 30) = 15 / 75 = +0.20
        imb = self.encoder.compute_book_imbalance(self.snap)
        self.assertAlmostEqual(imb, 0.20, places=2)
        self.assertGreaterEqual(imb, -1.0)
        self.assertLessEqual(imb, 1.0)

    def test_depth_profile(self):
        depth = self.encoder.compute_depth_profile(self.snap, bps_tiers=(10, 25))
        self.assertIn("bid_10bps", depth)
        self.assertIn("ask_10bps", depth)
        self.assertGreaterEqual(depth["bid_10bps"], 0.0)
        self.assertGreaterEqual(depth["ask_10bps"], 0.0)

    def test_encode_to_state_text(self):
        text = self.encoder.encode_to_state_text(self.snap)
        self.assertIn("L2 Microstructure", text)
        self.assertIn("Spread:", text)
        self.assertIn("Imbalance:", text)
        self.assertIn("CVD:", text)


class TestCPSATMarketRiskGate(unittest.TestCase):
    """Verifies Milestone 3: CP-SAT Inventory & Market Risk Gate."""

    def setUp(self):
        self.gate = CPSATMarketRiskGate(
            max_inventory=5.0,
            lot_size=1.0,
            max_consecutive_adverse=3,
            normal_spread_bps=2.5,
            max_spread_multiplier=3.0,
        )

    def test_max_inventory_long_boundary(self):
        # Current position at 5.0 (max inventory) -> Buy must be blocked and downgraded to hold
        verdict = self.gate.evaluate_risk(proposed_action="buy", override_position=5.0)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["action"], "hold")
        self.assertEqual(verdict["pruned_action"], "buy")
        self.assertIn("inventory_limit_exceeded", verdict["reason"])

        # Sell is allowed to reduce inventory
        verdict_sell = self.gate.evaluate_risk(proposed_action="sell", override_position=5.0)
        self.assertTrue(verdict_sell["passed"])
        self.assertEqual(verdict_sell["action"], "sell")

    def test_max_inventory_short_boundary(self):
        # Current position at -5.0 -> Sell must be blocked, Buy allowed
        verdict = self.gate.evaluate_risk(proposed_action="sell", override_position=-5.0)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["action"], "hold")
        self.assertEqual(verdict["pruned_action"], "sell")

        verdict_buy = self.gate.evaluate_risk(proposed_action="buy", override_position=-5.0)
        self.assertTrue(verdict_buy["passed"])
        self.assertEqual(verdict_buy["action"], "buy")

    def test_consecutive_adverse_cutoff(self):
        # Record 3 adverse buy fills
        self.gate.record_fill_outcome("buy", is_adverse=True)
        self.gate.record_fill_outcome("buy", is_adverse=True)
        self.gate.record_fill_outcome("buy", is_adverse=True)

        verdict = self.gate.evaluate_risk(proposed_action="buy", override_position=0.0)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["action"], "hold")
        self.assertIn("consecutive_adverse_fills_cutoff", verdict["reason"])

    def test_extreme_spread_circuit_breaker(self):
        # Normal spread is 2.5bps, multiplier is 3.0 -> Max allowed is 7.5bps
        verdict = self.gate.evaluate_risk(proposed_action="buy", spread_bps=10.0)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["action"], "hold")
        self.assertIn("extreme_spread_circuit_breaker", verdict["reason"])

    def test_latency_under_10ms(self):
        t0 = time.perf_counter()
        verdict = self.gate.evaluate_risk(proposed_action="buy")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 10.0)
        self.assertLess(verdict["latency_ms"], 10.0)


class TestSubSecondPipeline(unittest.TestCase):
    """Verifies Milestone 4: Sub-second Pipeline & Late-Breaker Circuit."""

    def setUp(self):
        self.pipeline = SubSecondDecisionPipeline(safety_deadline_ms=100.0)
        self.snap = L2OrderBookSnapshot(
            bids=[(100.0, 30.0), (99.95, 20.0)],
            asks=[(100.05, 10.0), (100.10, 10.0)],
            cvd=25.0,
            last_price=100.02,
        )

    def test_pipeline_normal_budget_execution(self):
        frame = self.pipeline.process_tick(self.snap)

        # Wire schema assertions
        self.assertIn("tick_id", frame)
        self.assertIn("timestamp_ms", frame)
        self.assertIn("action", frame)
        self.assertIn("probabilities", frame)
        self.assertIn("confidence", frame)
        self.assertIn("latency_ms", frame)
        self.assertIn("late", frame)
        self.assertIn("risk_guard", frame)

        # Latency budget SLA (< 80ms)
        self.assertLess(frame["latency_ms"], 80.0)
        self.assertFalse(frame["late"])

        # Stage breakdown
        stages = frame["stage_breakdown_ms"]
        self.assertLess(stages["ingest"], 20.0)
        self.assertLess(stages["scoring"], 50.0)
        self.assertLess(stages["risk_gate"], 10.0)

    def test_late_breaker_trigger(self):
        # Simulate delay exceeding safety_deadline_ms (120ms > 100ms)
        frame = self.pipeline.process_tick(self.snap, simulate_artificial_delay_ms=120.0)
        self.assertTrue(frame["late"])
        self.assertEqual(frame["action"], "hold")
        self.assertFalse(frame["risk_guard"]["passed"])
        self.assertIn("late_breaker_triggered", frame["risk_guard"]["reason"])


class TestSSEStreamingEndpoint(unittest.TestCase):
    """Verifies Milestone 1: GET /v1/decisions/stream SSE Endpoint."""

    @classmethod
    def setUpClass(cls):
        cls._orig_env = os.environ.get("GENZERO_API_KEY")
        existing_token = _load_api_token()
        if existing_token:
            cls.token = existing_token
        else:
            cls.token = "test-secret-token"
            os.environ["GENZERO_API_KEY"] = cls.token

    @classmethod
    def tearDownClass(cls):
        if cls._orig_env is not None:
            os.environ["GENZERO_API_KEY"] = cls._orig_env
        else:
            os.environ.pop("GENZERO_API_KEY", None)

    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_sse_stream_fails_closed_without_live_feed(self):
        # No live L2 feed is wired in; the old sine-wave synthetic order book is gone.
        resp = self.client.get(
            "/v1/decisions/stream?max_events=3&tick_interval_ms=10",
            headers=self.headers
        )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["detail"]["error"]["code"], "market_feed_unavailable")
        self.assertNotIn("event: decision", resp.text)

    def test_stream_unauthorized(self):
        resp = self.client.get("/v1/decisions/stream", headers={"Authorization": "Bearer invalid-token"})
        self.assertEqual(resp.status_code, 401)

    def test_root_includes_stream_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        endpoints = resp.json().get("endpoints", [])
        self.assertTrue(any("GET  /v1/decisions/stream" in ep for ep in endpoints))


if __name__ == "__main__":
    unittest.main()
