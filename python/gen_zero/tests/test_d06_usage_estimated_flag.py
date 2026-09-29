"""Regression guard for D06: /v1/decisions and /v1/decide_step must keep
reporting usage.estimated == True.

Both endpoints compute "usage.total_tokens" from a len(text) // 4 byte-length
heuristic, not a real tokenizer count (see app.py's own comments next to each
`usage` dict). `estimated: True` is the only signal callers that bill on
tokens have to know this number is not measured. If that key is ever dropped
during a refactor, a billing caller would silently start trusting a fake
count -- this test exists so that drops fail CI instead of production.
"""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from gen_zero.service import app as service_app
from gen_zero.service.app import app


class TestUsageEstimatedFlag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gen_zero.service.app import _load_api_token
        cls._orig_env = os.environ.get("GENZERO_API_KEY")
        existing_token = _load_api_token()
        if existing_token:
            cls.token = existing_token
        else:
            cls.token = "d06-usage-test-token"
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

    def test_decisions_usage_is_marked_estimated(self):
        payload = {
            "state": "CPU utilization at 96%, connection timeout warnings firing.",
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "What should the on-call operator do?",
                    "criteria": ["scale_up", "restart_service", "ignore"],
                }
            },
        }
        with patch.object(service_app.client, "weights_loaded_from_checkpoint", True):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertIn("usage", data)
        self.assertIs(data["usage"]["estimated"], True)

    def test_decide_step_usage_is_marked_estimated(self):
        payload = {
            "state": "Checkout page displayed with a filled shipping form.",
            "affordances": ["submit_order", "edit_address", "cancel"],
        }
        with patch.object(service_app.client, "weights_loaded_from_checkpoint", True):
            resp = self.client.post("/v1/decide_step", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertIn("usage", data)
        self.assertIs(data["usage"]["estimated"], True)

    def test_batch_decisions_usage_is_marked_estimated(self):
        payload = {
            "states": [
                "CPU utilization at 96%, connection timeout warnings firing.",
                "Database replica lag 120s, read errors increasing.",
            ],
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "What should the on-call operator do?",
                    "criteria": ["scale_up", "restart_service", "ignore"],
                }
            },
        }
        with patch.object(service_app.client, "weights_loaded_from_checkpoint", True):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertIn("usage", data)
        self.assertIs(data["usage"]["estimated"], True)
        self.assertIn("estimate_method", data["usage"])

    def test_decisions_fails_closed_without_checkpoint(self):
        payload = {
            "state": "CPU utilization at 96%, connection timeout warnings firing.",
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "What should the on-call operator do?",
                    "criteria": ["scale_up", "restart_service", "ignore"],
                }
            },
        }
        with patch.object(service_app.client, "weights_loaded_from_checkpoint", False):
            resp = self.client.post("/v1/decisions", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["detail"]["error"]["code"], "checkpoint_not_loaded")

    def test_decide_step_fails_closed_without_checkpoint(self):
        payload = {
            "state": "Checkout page displayed with a filled shipping form.",
            "affordances": ["submit_order", "edit_address", "cancel"],
        }
        with patch.object(service_app.client, "weights_loaded_from_checkpoint", False):
            resp = self.client.post("/v1/decide_step", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["detail"]["error"]["code"], "checkpoint_not_loaded")


if __name__ == "__main__":
    unittest.main()
