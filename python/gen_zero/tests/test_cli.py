"""Unit and Integration tests for Gen-Zero Unified Shell CLI."""

import io
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from gen_zero.cli import main, build_cli_parser


class TestGenZeroCLI(unittest.TestCase):
    """Verifies all subcommands of the unified gen-zero / zero Shell CLI."""

    def test_01_version_flag(self):
        """Verify gen-zero -v outputs version and returns 0."""
        with patch("sys.stdout") as mock_stdout:
            exit_code = main(["-v"])
            self.assertEqual(exit_code, 0)

    def test_02_status_subcommand(self):
        """Verify gen-zero status command."""
        exit_code = main(["status", "--no-color"])
        self.assertEqual(exit_code, 0)

    @unittest.skipUnless(
        os.environ.get("GENZERO_API_KEY"),
        "live integration test: needs GENZERO_API_KEY (no baked-in default token exists any more)",
    )
    def test_03_ask_subcommand_noul(self):
        """Verify gen-zero ask command with default noul type."""
        exit_code = main([
            "ask",
            "User requested DB backup",
            "Is the database online and accessible?",
            "--no-color"
        ])
        self.assertEqual(exit_code, 0)

    @unittest.skipUnless(
        os.environ.get("GENZERO_API_KEY"),
        "live integration test: needs GENZERO_API_KEY (no baked-in default token exists any more)",
    )
    def test_04_ask_subcommand_choice(self):
        """Verify gen-zero ask command with choice type."""
        exit_code = main([
            "ask",
            "Select deployment environment",
            "Which cluster to target?",
            "--type", "choice",
            "--choices", "staging", "production",
            "--no-color"
        ])
        self.assertEqual(exit_code, 0)

    def test_04b_ask_subcommand_abstain_shown_explicitly(self):
        """B0928 blocker 3: an ABSTAIN answer must print as ABSTAIN, never a fabricated
        0.000 noul bar / 'none' choice / 0.0 score from a `.get(key, default)` fallback."""
        fake_response = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "model": "typesafe/zero-1.13",
                    "timing_ms": 1.0,
                    "answers": {
                        "q1": {
                            "type": "noul",
                            "status": "ABSTAIN",
                            "kernel_status": "INFEASIBLE_ABSTAIN",
                            "error": "Decision kernel abstained: no admissible candidate action.",
                        }
                    },
                })
            }]
        }

        async def fake_execute_zero_ask(_req):
            return fake_response

        with patch("gen_zero.cli.execute_zero_ask", side_effect=fake_execute_zero_ask), \
             patch("sys.stdout", new=io.StringIO()) as fake_out:
            exit_code = main(["ask", "ambiguous state", "is it safe?", "--no-color"])
            self.assertEqual(exit_code, 0)
            output = fake_out.getvalue()
            self.assertIn("ABSTAIN", output)
            self.assertIn("INFEASIBLE_ABSTAIN", output)
            self.assertNotIn("0.000", output)

    def test_05_route_subcommand(self):
        """Verify gen-zero route command."""
        mock_tools = json.dumps([
            {"name": "git_commit", "description": "Commits code to git repo"},
            {"name": "bake_bread", "description": "Bakes fresh sourdough bread"},
            {"name": "git_push", "description": "Pushes local branch to remote repository"},
        ])
        exit_code = main([
            "route",
            "Push changes to GitHub repository",
            "--tools", mock_tools,
            "--top-k", "2",
            "--no-color"
        ])
        self.assertEqual(exit_code, 0)

    def test_06_imagine_subcommand(self):
        """Verify gen-zero imagine command with CP-SAT safety verification."""
        exit_code = main([
            "imagine",
            "Production microservice memory leak detected",
            "--actions", "ROLLBACK", "RESTART_POD", "SCALE_UP",
            "--horizon", "3",
            "--no-color"
        ])
        self.assertEqual(exit_code, 0)

    def test_07_stream_subcommand(self):
        """Verify gen-zero stream command."""
        exit_code = main([
            "stream",
            "telemetry_stream_frame_042",
            "--actions", "ACT_LEFT", "ACT_RIGHT", "ACT_STAY",
            "--no-color"
        ])
        self.assertEqual(exit_code, 0)

    def test_08_compact_subcommand(self):
        """Verify gen-zero compact command."""
        messages = [
            {"id": "0", "role": "system", "content": "You are a software engineer."},
            {"id": "1", "role": "tool", "name": "pwd", "content": "/tmp"},
            {"id": "2", "role": "tool", "name": "whoami", "content": "testuser"},
            {"id": "3", "role": "user", "content": "Fix the bug on line 42."}
        ]
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(messages, f)
            temp_path = f.name

        try:
            exit_code = main(["compact", temp_path, "--no-color"])
            self.assertEqual(exit_code, 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_09_grep_subcommand(self):
        """Verify gen-zero grep subcommand."""
        from unittest.mock import MagicMock, patch
        def mock_urlopen(req, *args, **kwargs):
            body = json.loads(req.data.decode("utf-8"))
            n = len(body.get("states", []))
            qids = list(body.get("questions", {}).keys())
            res = {
                "results": [{"answers": {qid: {"noul": 0.95} for qid in qids}} for _ in range(n)]
            }
            cm = MagicMock()
            cm.__enter__.return_value.read.return_value = json.dumps(res).encode("utf-8")
            return cm

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            exit_code = main([
                "grep",
                "-e", "test",
                os.path.join(os.path.dirname(__file__), "test_mcp_server.py")
            ])
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
