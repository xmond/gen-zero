"""Real helper processes and disposable Git repos: no provider/test doubles."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ADAPTER = Path(__file__).resolve().parents[1] / "gen_zero_deepswe_adapter.py"


class SandboxBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        (self.repo / "module.py").write_text("def value():\n    return 1\n")
        (self.repo / "tests").mkdir()
        (self.repo / "tests/test_value.py").write_text("def test_value():\n    assert True\n")
        for cmd in (["git", "init", "-q"], ["git", "add", "."],
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "base"]):
            subprocess.run(cmd, cwd=self.repo, check=True, capture_output=True)

    def action(self, value, expected=0):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as action_file:
            json.dump(value, action_file)
            action_file.flush()
            r = subprocess.run([sys.executable, str(ADAPTER), "--sandbox-action-file", action_file.name],
                               cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(r.returncode, expected, r.stderr + r.stdout)
        return json.loads(r.stdout) if expected == 0 else r.stderr

    def edit(self, path="module.py", old=None, content="def value():\n    return 2\n"):
        return {"action": "edit", "files": [{"path": path, "old_sha256": old, "content": content}]}

    def test_real_search_read_apply_and_git_diff(self):
        inventory = self.action({"action": "inventory"})
        self.assertIn("module.py", inventory["files"])
        matches = self.action({"action": "search", "query": "def value"})
        self.assertIn("module.py:1:", matches["matches"])
        read = self.action({"action": "read", "path": "module.py"})
        result = self.action(self.edit(old=read["sha256"]))
        self.assertIn("+    return 2", result["patch"])
        subprocess.run(["git", "diff", "--cached", "--check"], cwd=self.repo, check=True)
        self.assertEqual(subprocess.run([sys.executable, "-c", "from module import value; assert value() == 2"], cwd=self.repo).returncode, 0)

    def test_new_file_is_in_standard_diff(self):
        result = self.action(self.edit(path="new_module.py", content="VALUE = 3\n"))
        self.assertIn("new file mode", result["patch"])
        self.assertIn("+VALUE = 3", result["patch"])

    def test_large_source_does_not_use_shell_argument_transport(self):
        content = "TEXT = " + repr("x" * 150_000) + "\n"
        result = self.action(self.edit(path="large_module.py", content=content))
        self.assertIn(content, result["patch"])

    def test_stale_hash_fails_without_mutation(self):
        before = (self.repo / "module.py").read_bytes()
        self.assertIn("Stale source hash", self.action(self.edit(old="0" * 64), 1))
        self.assertEqual((self.repo / "module.py").read_bytes(), before)

    def test_all_edits_are_validated_before_writing(self):
        before = (self.repo / "module.py").read_bytes()
        value = self.edit(old=hashlib.sha256(before).hexdigest())
        value["files"].append({"path": "new_module.py", "old_sha256": None, "content": "def broken(:"})
        self.action(value, 1)
        self.assertEqual((self.repo / "module.py").read_bytes(), before)
        self.assertFalse((self.repo / "new_module.py").exists())

    def test_path_escape_and_symlink_fail(self):
        for name in ("../escape.py", "/tmp/escape.py", ".git/config"):
            self.action(self.edit(path=name), 1)
        (self.repo / "linked.py").symlink_to(self.repo / "module.py")
        self.assertIn("Symlink", self.action(self.edit(path="linked.py"), 1))

    def test_test_mutation_and_dirty_start_fail(self):
        self.assertIn("may not edit tests", self.action(self.edit(path="tests/test_value.py"), 1))
        (self.repo / "untracked.py").write_text("X = 1\n")
        self.assertIn("clean", self.action({"action": "inventory"}, 1))

    def test_empty_submission_and_unknown_action_fail(self):
        head = self.action({"action": "inventory"})["head"]
        self.assertIn("empty submission", self.action({"action": "finalize", "base": head}, 1))
        self.action({"action": "shell", "command": "true"}, 1)

    def test_missing_provider_credential_exits_without_fallback(self):
        env = dict(os.environ)
        env.pop("DEEPSWE_TEST_UNSET_CREDENTIAL", None)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as request:
            json.dump({"config": ["https://example.invalid/v1/messages", "unconfigured",
                                  "anthropic", "DEEPSWE_TEST_UNSET_CREDENTIAL"],
                       "messages": [], "output": str(self.repo / "provider")}, request)
            request.flush()
            result = subprocess.run([sys.executable, str(ADAPTER), "--proposer-request-file", request.name],
                                    env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("credential environment variable DEEPSWE_TEST_UNSET_CREDENTIAL is unset", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.repo / "provider.request.json").exists())


if __name__ == "__main__":
    unittest.main()
