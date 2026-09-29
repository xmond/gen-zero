"""Pure boundary tests. No model, HTTP service, or sandbox is mocked."""
import importlib.util
import sys
import unittest
from pathlib import Path

# The adapter lives one level up (benchmarks/), not under pytest.ini's `python` path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Harbor is an optional eval-host dependency; without it the adapter cannot import.
if importlib.util.find_spec("harbor") is None:
    raise unittest.SkipTest("harbor not installed; run these in the Harbor venv")

from gen_zero_tb_adapter import Refused, validate_decision, validate_proposals, validate_url


class BoundaryTests(unittest.TestCase):
    def test_missing_and_error_policy_evidence_is_refused(self):
        for verb in ("ask", "route", "imagine"):
            for response in ({}, {"isError": True}, {"semantic_scoring": False}):
                with self.subTest(verb=verb, response=response), self.assertRaises(Refused):
                    validate_decision(response, verb)

    def test_commands_round_trip_without_shell_rewriting(self):
        proposals = {"candidates": [
            {"kind": "bash", "command": "printf '%s\\n' '$HOME;$(id)'"},
            {"kind": "bash", "command": "pwd"},
        ]}
        self.assertEqual(validate_proposals(proposals), proposals["candidates"])

    def test_invalid_proposals_fail_closed(self):
        good = {"kind": "bash", "command": "pwd"}
        for candidates in ([], [good], [good, good], [good, {"kind": "python"}],
                           [good, {"kind": "bash", "command": "\0"}],
                           [good, {"kind": "finish", "reason": ""}],
                           [good, {"kind": "bash", "command": "ls", "safe": True}]):
            with self.subTest(candidates=candidates), self.assertRaises(Refused):
                validate_proposals({"candidates": candidates})

    def test_credentials_and_remote_plaintext_urls_refused(self):
        for url in ("http://example.com/v1", "https://key@example.com/v1",
                    "https://example.com/v1?key=secret", "file:///tmp/service"):
            with self.subTest(url=url), self.assertRaises(Refused):
                validate_url(url)
        self.assertEqual(validate_url("http://127.0.0.1:8080/v1/decisions"),
                         "http://127.0.0.1:8080/v1/decisions")



class ProposerTests(unittest.IsolatedAsyncioTestCase):
    async def test_natural_language_is_not_disguised_as_reasoning(self):
        from gen_zero_tb_adapter import DeterministicProposer
        with self.assertRaises(Refused):
            await DeterministicProposer().propose(
                {"instruction": "Find a file", "history": []}, None, None)

    async def test_explicit_plan_requires_exact_observed_postcondition(self):
        import json
        from gen_zero_tb_adapter import DeterministicProposer
        candidates = [{"kind": "bash", "command": "pwd"},
                      {"kind": "bash", "command": "printf '%s\\n' \"$PWD\""}]
        state = {"instruction": json.dumps({"steps": [
            {"candidates": candidates, "expect_stdout": "/app\n"}]}), "history": []}
        proposer = DeterministicProposer()
        self.assertEqual(await proposer.propose(state, None, None), candidates)
        state.update(history=[{}], observation={"return_code": 1, "stdout": "/app\n"})
        with self.assertRaises(Refused):
            await proposer.propose(state, None, None)
        state["observation"]["return_code"] = 0
        self.assertEqual((await proposer.propose(state, None, None))[0]["kind"], "finish")

if __name__ == "__main__":
    unittest.main()
