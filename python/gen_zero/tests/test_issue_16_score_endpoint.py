"""Unit & Benchmark Tests for Minimal Non-Autoregressive Score Protocol (Issue #16).

Verifies:
1. POST /v1/score and /score endpoints with FastAPI TestClient.
2. LlamaCppScoreAdapter with Thought Tag Folding, 1-token logprobs extraction,
   and fail-closed behaviour when llama-server is unreachable (no pseudo-scores).
3. Python SDK `client.score(...)` signature, type compliance, and error boundaries.
"""

import unittest
import os
import time
import math
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from fastapi.testclient import TestClient

from gen_zero.service.app import app, compute_closed_form_confidence
from gen_zero.client import GenZero
from gen_zero.gateway.llamacpp_adapter import (
    LlamaCppScoreAdapter,
    LlamaCppUnavailableError,
    compute_closed_form_confidence as adapter_compute_confidence,
)


class TestScoreEndpoint(unittest.TestCase):
    """Verifies Milestone 1: POST /v1/score HTTP endpoint behavior."""

    @classmethod
    def setUpClass(cls):
        from gen_zero.service.app import _load_api_token
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

    def test_score_endpoint_success(self):
        payload = {
            "prompt": "生产环境 CPU 达到 96%，出现连接超时告警",
            "candidates": ["扩容实例", "重启服务", "忽略日志"],
            "model": "typesafe/zero-1.13",
            "temperature": 1.0
        }
        # Without a loaded dual-head checkpoint the reflex path fails closed (503).
        os.environ.pop("LLAMACPP_BASE_URL", None)
        resp = self.client.post("/v1/score", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(
            resp.json()["detail"]["error"]["message"],
            "Model checkpoint not loaded; pseudo-random inference is disabled per fail-closed policy",
        )

        # The wire contract is exercised only with the checkpoint flag set.
        from gen_zero.service import app as service_app
        from unittest.mock import patch
        with patch.object(service_app.client, "weights_loaded_from_checkpoint", True):
            resp = self.client.post("/v1/score", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Schema assertions
        self.assertIn("choice", data)
        self.assertIn("scores", data)
        self.assertIn("probabilities", data)
        self.assertIn("confidence", data)
        self.assertIn("timing_ms", data)

        # Content assertions
        self.assertIn(data["choice"], payload["candidates"])
        self.assertEqual(len(data["scores"]), len(payload["candidates"]))
        self.assertEqual(len(data["probabilities"]), len(payload["candidates"]))

        # Probability sum check (~1.0)
        total_prob = sum(data["scores"])
        self.assertAlmostEqual(total_prob, 1.0, places=2)

        # Closed-form confidence contract: c = (p_max - 1/K) / (1 - 1/K)
        p_max = max(data["scores"])
        K = len(payload["candidates"])
        expected_conf = max(0.0, min(1.0, (p_max - 1.0 / K) / (1.0 - 1.0 / K)))
        self.assertAlmostEqual(data["confidence"], round(expected_conf, 4), places=2)

    def test_score_endpoint_alias(self):
        payload = {
            "prompt": "Database connection pool exhausted",
            "candidates": ["scale_pool", "drop_idle", "alert_oncall"]
        }
        os.environ.pop("LLAMACPP_BASE_URL", None)
        resp = self.client.post("/score", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 503)
        from gen_zero.service import app as service_app
        from unittest.mock import patch
        with patch.object(service_app.client, "weights_loaded_from_checkpoint", True):
            resp = self.client.post("/score", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("choice", data)
        self.assertIn(data["choice"], payload["candidates"])

    def test_score_endpoint_validation_errors(self):
        # Missing prompt key
        resp = self.client.post("/v1/score", json={"candidates": ["a", "b"]}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["detail"]["error"]["code"], "missing_prompt")

        # Empty prompt string
        resp = self.client.post("/v1/score", json={"prompt": "", "candidates": ["a", "b"]}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["detail"]["error"]["code"], "missing_prompt")

        # Missing candidates key
        resp = self.client.post("/v1/score", json={"prompt": "Some prompt"}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["detail"]["error"]["code"], "missing_candidates")

        # Empty candidates list
        resp = self.client.post("/v1/score", json={"prompt": "Some prompt", "candidates": []}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["detail"]["error"]["code"], "missing_candidates")

    def test_root_endpoint_includes_score(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        endpoints = data.get("endpoints", [])
        self.assertTrue(any("POST /v1/score" in ep for ep in endpoints))


class _FakeLlamaServer:
    """Local HTTP stand-in for llama-server /completion, returning a fixed real-format reply."""

    def __init__(self, body):
        payload = json.dumps(body).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self._srv = HTTPServer(("127.0.0.1", 0), Handler)

    def __enter__(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def __exit__(self, *exc):
        self._srv.shutdown()
        self._srv.server_close()


class TestLlamaCppScoreAdapter(unittest.TestCase):
    """Verifies Milestone 2: LlamaCppScoreAdapter mechanics."""

    def setUp(self):
        self.adapter = LlamaCppScoreAdapter(
            base_url="http://127.0.0.1:8080",
            timeout=0.1,
        )

    def test_thought_tag_folding(self):
        # Case 1: Unclosed <think> tag must be closed with </think>\n
        prompt1 = "<think>\nThinking about CPU load and cluster health..."
        folded1 = self.adapter.fold_thought_tags(prompt1)
        self.assertTrue(folded1.endswith("</think>\n"))
        self.assertIn("<think>", folded1)

        # Case 2: Already closed tag must be preserved without duplicate tags
        prompt2 = "<think>\nResolved thought.\n</think>\nWhat should we do?"
        folded2 = self.adapter.fold_thought_tags(prompt2)
        self.assertEqual(folded2, prompt2)
        self.assertEqual(folded2.count("</think>"), 1)

        # Case 3: Plain prompt with force folding
        prompt3 = "Analyze system load"
        folded3 = self.adapter.fold_thought_tags(prompt3, force=True)
        self.assertTrue(folded3.endswith("</think>\n"))

    def test_unreachable_server_raises_instead_of_pseudo_score(self):
        adapter = LlamaCppScoreAdapter(base_url="http://127.0.0.1:1", timeout=0.2)
        with self.assertLogs("gen_zero.gateway.llamacpp_adapter", level="ERROR"):
            with self.assertRaises(LlamaCppUnavailableError):
                adapter.score("High CPU on node-3", ["restart", "scale_up"])

    def test_no_mock_symbols_remain(self):
        self.assertFalse(hasattr(self.adapter, "_mock_offline_score"))
        self.assertFalse(hasattr(self.adapter, "enable_mock_fallback"))

    def test_real_server_reply_is_normalised(self):
        body = {"completion_probabilities": [{"probs": [
            {"tok_str": "restart", "prob": 0.6},
            {"tok_str": "ignore", "prob": 0.2},
        ]}]}
        with _FakeLlamaServer(body) as url:
            adapter = LlamaCppScoreAdapter(base_url=url, timeout=2.0)
            res = adapter.score("High CPU", ["restart", "ignore", "scale_up"])
        self.assertEqual(res["choice"], "restart")
        self.assertEqual(res["source"], "llamacpp")
        self.assertAlmostEqual(sum(res["scores"]), 1.0, places=2)

    def test_reply_without_candidate_probability_raises(self):
        body = {"completion_probabilities": [{"probs": [{"tok_str": "zzz", "prob": 0.9}]}]}
        with _FakeLlamaServer(body) as url:
            adapter = LlamaCppScoreAdapter(base_url=url, timeout=2.0)
            with self.assertLogs("gen_zero.gateway.llamacpp_adapter", level="ERROR"):
                with self.assertRaises(LlamaCppUnavailableError):
                    adapter.score("High CPU", ["restart", "ignore"])

    def test_adapter_health_check(self):
        health = self.adapter.check_health()
        self.assertIn("status", health)
        self.assertIn("base_url", health)
        self.assertNotIn("mock", health["status"])


class TestClientScoreMethod(unittest.TestCase):
    """Verifies Milestone 3: Python Client client.score(...) interface."""

    def setUp(self):
        self.client = GenZero()

    def test_client_score_reflex(self):
        prompt = "Container memory OOM error"
        candidates = ["increase_limit", "kill_pod", "do_nothing"]
        res = self.client.score(prompt=prompt, candidates=candidates)

        self.assertIn("choice", res)
        self.assertIn(res["choice"], candidates)
        self.assertEqual(len(res["scores"]), len(candidates))
        self.assertAlmostEqual(sum(res["scores"]), 1.0, places=2)
        self.assertIn("confidence", res)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertLess(res["timing_ms"], 50.0)

    def test_client_score_with_llamacpp_routing(self):
        prompt = "<think>Analyzing DB pool</think> Connection pool full"
        candidates = ["increase_pool", "close_idle"]
        self.client.llamacpp_adapter.base_url = "http://127.0.0.1:1"
        with self.assertLogs("gen_zero.gateway.llamacpp_adapter", level="ERROR"):
            with self.assertRaises(LlamaCppUnavailableError):
                self.client.score(prompt=prompt, candidates=candidates, use_llamacpp=True)

    def test_client_score_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.client.score(prompt="", candidates=["a", "b"])

        with self.assertRaises(ValueError):
            self.client.score(prompt="Valid", candidates=[])


class TestClosedFormConfidence(unittest.TestCase):
    def test_closed_form_confidence_invariants(self):
        """Validates mathematical adherence to c = (p_max - 1/K) / (1 - 1/K).

        Invariants:
        1. K = 1 -> 1.0 (trivial certainty)
        2. K = 0 -> 0.0 (empty input guard)
        3. Uniform distribution (p_max = 1/K) -> 0.0 (maximum uncertainty)
        4. One-hot distribution (p_max = 1.0) -> 1.0 (absolute certainty)
        5. Exact closed-form interpolation for intermediate values
        6. Finite bounds clamping [0.0, 1.0] and numerical stability with inf/nan
        """
        for fn in [compute_closed_form_confidence, adapter_compute_confidence]:
            # K = 1
            self.assertEqual(fn({"only": 1.0}), 1.0)
            self.assertEqual(fn({"only": 0.0}), 1.0)
            self.assertEqual(fn([0.7]), 1.0 if fn == adapter_compute_confidence else 1.0)

            # K = 0 / empty
            self.assertEqual(fn({}), 0.0)
            if fn == adapter_compute_confidence:
                self.assertEqual(fn([]), 0.0)

            # Uniform distribution -> 0.0
            for K in [2, 3, 5, 10]:
                uniform_dict = {f"k_{i}": 1.0 / K for i in range(K)}
                self.assertAlmostEqual(fn(uniform_dict), 0.0, places=5)
                if fn == adapter_compute_confidence:
                    self.assertAlmostEqual(fn([1.0 / K] * K), 0.0, places=5)

            # One-hot distribution -> 1.0
            for K in [2, 3, 5, 10]:
                one_hot = {f"k_{i}": 1.0 if i == 0 else 0.0 for i in range(K)}
                self.assertAlmostEqual(fn(one_hot), 1.0, places=5)

            # Monotonic intermediate interpolation
            # For K=4, uniform is 0.25. If p_max=0.625, c = (0.625 - 0.25) / (1 - 0.25) = 0.375 / 0.75 = 0.5
            k4_probs = {"a": 0.625, "b": 0.125, "c": 0.125, "d": 0.125}
            self.assertAlmostEqual(fn(k4_probs), 0.5, places=5)

            # Out of bounds clamping: sub-uniform (p_max < 1/K) -> 0.0
            self.assertEqual(fn({"a": 0.1, "b": 0.1}), 0.0)

            # Numerical stability: non-finite inputs
            c_inf = fn({"a": float("inf"), "b": 0.5})
            self.assertTrue(0.0 <= c_inf <= 1.0)


if __name__ == "__main__":
    unittest.main()
