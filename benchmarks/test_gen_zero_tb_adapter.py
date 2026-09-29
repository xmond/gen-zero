"""Adapter boundary checks; live service acceptance requires a separate run."""
import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_zero_tb_adapter import GenZeroAgent, OpenAIProposer, Refused, validate_decision, validate_url


class URLTests(unittest.TestCase):
    def test_private_http_endpoints(self):
        for host in ("100.102.231.124", "100.64.0.1", "100.127.255.254",
                     "192.168.1.25", "127.0.0.1", "localhost", "[::1]"):
            url = f"http://{host}:8080/v1/chat/completions"
            with self.subTest(url=url):
                self.assertEqual(validate_url(url), url)

    def test_public_or_malformed_http_refused(self):
        for url in ("http://100.63.255.255/v1", "http://100.128.0.1/v1",
                    "http://192.169.1.1/v1", "http://example.com/v1",
                    "http://100.102.231.124:bad/v1", "http://100.102.231.124:0/v1",
                    "http://100.102.231.124/v1?token=secret"):
            with self.subTest(url=url), self.assertRaises(Refused):
                validate_url(url)

    def test_missing_gate_evidence_refused(self):
        for verb in ("route", "ask", "imagine"):
            with self.subTest(verb=verb), self.assertRaises(Refused):
                validate_decision({"isError": False}, verb)

    def test_harbor_model_and_proposer_configuration(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "GENZERO_PROPOSER_URL": "http://100.102.231.124:8080/v1/chat/completions",
            "HARBOR_MODEL": "qwen3.6-35b",
        }):
            agent = GenZeroAgent(logs_dir=Path(directory) / "agent")
            proposer = OpenAIProposer(agent.proposer_url, agent.model_name)
            self.assertEqual(proposer.model, "qwen3.6-35b")
            self.assertEqual(proposer.url, os.environ["GENZERO_PROPOSER_URL"])


if __name__ == "__main__":
    unittest.main()
