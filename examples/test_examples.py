"""Smoke tests of the actual public example programs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExampleSmokeTests(unittest.TestCase):
    def test_python_examples_emit_json(self):
        # No PYTHONPATH injected here: each script must find `gen_zero` on its
        # own (via its sys.path.insert), the same way a user running
        # `python3 examples/foo.py` straight from a clone would.
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for name, required in (
            ("01_quickstart_decision.py", {"discrete", "continuous"}),
            ("02_world_model_simulation.py", {"simulation", "what_if", "audit"}),
            ("03_mpc_cem_continuous.py", {"planner", "result"}),
        ):
            with self.subTest(name=name):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "examples" / name)],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout)
                self.assertTrue(required <= value.keys())
                if name.startswith("01"):
                    self.assertIn(value["discrete"]["action"], ["north", "east", "south", "west", "ABSTAIN"])
                    self.assertEqual(len(value["continuous"]["action"]), 2)
                if name.startswith("02"):
                    self.assertIn("provenance", value["simulation"])
                    self.assertIn("traps_detected", value["what_if"])
                    self.assertIn("verdict", value["audit"])
                if name.startswith("03"):
                    self.assertEqual(value["result"]["mode"], "continuous")
                    self.assertEqual(len(value["result"]["best_trajectory"]), 4)

    def test_9b_demo_emits_json(self):
        artifact = ROOT / "artifacts" / "qwen35_9b" / "zero_rnn_set_adapter_qwen35_9b.npz"
        if not artifact.exists():
            self.skipTest(f"artifact not present at {artifact}")
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "run_9b_demo.py")],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertIn("results", value)
        self.assertIn("latency_ms", value)
        self.assertTrue(value["results"])

    def test_rust_cli_native_fails_without_binary(self):
        env = os.environ.copy()
        env["GEN_ZERO_BIN"] = str(ROOT / "examples" / "missing-gen-zero")
        result = subprocess.run(
            ["bash", str(ROOT / "examples" / "01_rust_cli_native.sh")],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("binary is missing", result.stderr)

    def test_rust_cli_native_walkthrough(self):
        binary = ROOT / "target" / "release" / "gen-zero"
        if not binary.exists():
            self.skipTest(f"binary not built at {binary} (cargo build --release -p gen-zero-cli)")
        result = subprocess.run(
            ["bash", str(ROOT / "examples" / "01_rust_cli_native.sh")],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("what-if", result.stdout)
        self.assertIn("Native Rust CLI walkthrough complete", result.stdout)


if __name__ == "__main__":
    unittest.main()
