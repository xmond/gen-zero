"""Fail-closed guarantees for the Python service layer (audit P0-1 / P1 / P2).

Every test here pins a behavior that used to be faked:
- 503 instead of seeded pseudo-random probabilities when no checkpoint is loaded
- 503 instead of a sine-wave synthetic order book on the SSE stream
- API key from GENZERO_API_KEY only, never from /tmp/zero.txt, never a baked-in default
- CP-SAT crashes are logged and reported as fallbacks, never as clean solves
- unknown co-riding probes raise instead of hashing a fake confidence
- torch scoring failures are logged and surface as degraded=True
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from gen_zero.service import app as service_app
from gen_zero.service.app import MODEL_NOT_LOADED_MESSAGE, _load_api_token, app


TOKEN = "fail-closed-test-token"


class _EnvTokenMixin:
    @classmethod
    def setUpClass(cls):
        cls._orig_env = os.environ.get("GENZERO_API_KEY")
        os.environ["GENZERO_API_KEY"] = TOKEN
        cls._orig_llama = os.environ.pop("LLAMACPP_BASE_URL", None)

    @classmethod
    def tearDownClass(cls):
        if cls._orig_env is not None:
            os.environ["GENZERO_API_KEY"] = cls._orig_env
        else:
            os.environ.pop("GENZERO_API_KEY", None)
        if cls._orig_llama is not None:
            os.environ["LLAMACPP_BASE_URL"] = cls._orig_llama

    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"Authorization": f"Bearer {TOKEN}"}


class TestModelNotLoadedGate(_EnvTokenMixin, unittest.TestCase):
    """No dual-head checkpoint => 503 with the exact policy message."""

    def setUp(self):
        super().setUp()
        self.assertFalse(
            service_app.client.weights_loaded_from_checkpoint,
            "test assumes GENZERO_DUAL_HEAD_CHECKPOINT is unset",
        )

    def _assert_503(self, resp):
        self.assertEqual(resp.status_code, 503, resp.text)
        err = resp.json()["detail"]["error"]
        self.assertEqual(err["message"], MODEL_NOT_LOADED_MESSAGE)
        self.assertEqual(err["code"], "checkpoint_not_loaded")

    def test_decisions_single_state_503(self):
        payload = {
            "state": "CPU at 96% with connection timeouts",
            "questions": {"escalate": {"type": "noul", "criteria": {"true": "page", "false": "wait"}}},
        }
        for path in ("/v1/decisions", "/api/alpha/decisions", "/decisions"):
            with self.assertLogs("gen_zero.service.app", level="ERROR"):
                self._assert_503(self.client.post(path, json=payload, headers=self.headers))

    def test_decisions_batch_states_503(self):
        payload = {
            "states": ["a", "b"],
            "questions": {"route": {"type": "choice", "criteria": {"x": "do x", "y": "do y"}}},
        }
        self._assert_503(self.client.post("/v1/decisions", json=payload, headers=self.headers))

    def test_decide_step_503(self):
        payload = {"state": "page loaded", "affordances": ["e1", "e2"], "goal": "click login"}
        self._assert_503(self.client.post("/v1/decide_step", json=payload, headers=self.headers))

    def test_score_reflex_path_503(self):
        payload = {"prompt": "Database pool exhausted", "candidates": ["scale", "alert"]}
        self._assert_503(self.client.post("/v1/score", json=payload, headers=self.headers))
        self._assert_503(self.client.post("/score", json=payload, headers=self.headers))

    def test_score_llamacpp_configured_but_unreachable_503(self):
        payload = {"prompt": "Database pool exhausted", "candidates": ["scale", "alert"]}
        with patch.dict(os.environ, {"LLAMACPP_BASE_URL": "http://127.0.0.1:1"}):
            with self.assertLogs("gen_zero.service.app", level="ERROR"):
                resp = self.client.post("/v1/score", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 503, resp.text)
        self.assertEqual(resp.json()["detail"]["error"]["code"], "llamacpp_unreachable")
        self.assertNotIn("choice", resp.text)

    def test_score_validation_still_400_before_gate(self):
        resp = self.client.post("/v1/score", json={"candidates": ["a"]}, headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    def test_gate_opens_only_when_checkpoint_flag_is_true(self):
        payload = {"prompt": "Database pool exhausted", "candidates": ["scale", "alert"]}
        with patch.object(service_app.client, "weights_loaded_from_checkpoint", True):
            resp = self.client.post("/v1/score", json=payload, headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn(resp.json()["choice"], payload["candidates"])

    def test_stream_503_no_synthetic_order_book(self):
        resp = self.client.get("/v1/decisions/stream?max_events=3", headers=self.headers)
        self.assertEqual(resp.status_code, 503)
        err = resp.json()["detail"]["error"]
        self.assertEqual(err["code"], "market_feed_unavailable")
        self.assertNotIn("event: decision", resp.text)


class TestApiTokenSources(unittest.TestCase):
    def test_service_token_env_only(self):
        with patch.dict(os.environ, {"GENZERO_API_KEY": " abc "}):
            self.assertEqual(_load_api_token(), "abc")
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True), patch(
                "builtins.open", side_effect=AssertionError("credential file must not be read")
            ):
                self.assertEqual(_load_api_token(), "")

    def test_mcp_token_env_only_and_no_default(self):
        from gen_zero.mcp.server import resolve_api_token

        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.open", side_effect=AssertionError("credential file must not be read")):
                self.assertEqual(resolve_api_token(), "")
        with patch.dict(os.environ, {"GENZERO_API_KEY": "from-env"}):
            self.assertEqual(resolve_api_token(), "from-env")

    def test_gen_grep_token_env_only(self):
        from gen_zero.scripts.gen_grep import _load_token

        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.open", side_effect=AssertionError("credential file must not be read")):
                self.assertEqual(_load_token(), "")
        self.assertEqual(_load_token("explicit"), "explicit")


def _install_crashing_ortools():
    """Fake ortools whose CpModel() raises, so the except path is reachable without OR-Tools."""
    ortools = types.ModuleType("ortools")
    sat = types.ModuleType("ortools.sat")
    python_mod = types.ModuleType("ortools.sat.python")
    cp_model = types.ModuleType("ortools.sat.python.cp_model")

    class CpModel:
        def __init__(self):
            raise RuntimeError("simulated solver crash")

    cp_model.CpModel = CpModel
    cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN = 4, 2, 0
    python_mod.cp_model = cp_model
    sat.python = python_mod
    ortools.sat = sat
    return {
        "ortools": ortools,
        "ortools.sat": sat,
        "ortools.sat.python": python_mod,
        "ortools.sat.python.cp_model": cp_model,
    }


class TestSolverExceptionsAreNotSwallowed(unittest.TestCase):
    def test_cpsat_formal_solver_reports_exception_fallback(self):
        from gen_zero.gate.cpsat_formal_solver import CPSATFormalSolver

        solver = CPSATFormalSolver(hard_timeout_ms=1000.0)
        solver._ortools_available = True
        with patch.dict(sys.modules, _install_crashing_ortools()):
            with self.assertLogs("gen_zero.gate.cpsat_formal_solver", level="ERROR") as logs:
                verdict = solver.solve_safest_optimal_action({"a": 0.2, "b": 0.7, "c": 0.1})
        self.assertTrue(verdict.fallback_used)
        self.assertTrue(verdict.solver_status.startswith("CPSAT_EXCEPTION_FALLBACK"))
        self.assertNotEqual(verdict.solver_status, "DETERMINISTIC_SAFE_SOLVED")
        self.assertEqual(verdict.selected_action, "b")
        self.assertTrue(any("simulated solver crash" in line for line in logs.output))

    def test_constraint_compiler_reports_exception_fallback(self):
        from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler

        compiler = ConstraintLinearProjectionCompiler(hard_timeout_ms=1000.0)
        with patch.dict(sys.modules, _install_crashing_ortools()):
            with self.assertLogs("gen_zero.gate.constraint_compiler", level="ERROR") as logs:
                verdict = compiler.solve_safest_action({"BUY": 0.3, "SELL": 0.6, "HOLD": 0.1})
        self.assertTrue(verdict.fallback_used)
        self.assertTrue(verdict.solver_status.startswith("CPSAT_EXCEPTION_FALLBACK"))
        self.assertNotEqual(verdict.solver_status, "0_1_ILP_FEASIBLE")
        self.assertEqual(verdict.selected_action, "HOLD")
        self.assertFalse(verdict.is_safe)
        self.assertTrue(any("simulated solver crash" in line for line in logs.output))


class TestMissingOrToolsIsExplicitFallback(unittest.TestCase):
    _blocked = {"ortools": None, "ortools.sat": None, "ortools.sat.python": None,
                "ortools.sat.python.cp_model": None}

    def test_cpsat_formal_solver_flags_missing_ortools(self):
        from gen_zero.gate.cpsat_formal_solver import CPSATFormalSolver

        with patch.dict(sys.modules, self._blocked):
            solver = CPSATFormalSolver(hard_timeout_ms=1000.0)
            with self.assertLogs("gen_zero.gate.cpsat_formal_solver", level="WARNING"):
                verdict = solver.solve_safest_optimal_action({"a": 0.2, "b": 0.7})
        self.assertTrue(verdict.fallback_used)
        self.assertEqual(verdict.solver_status, "ORTOOLS_UNAVAILABLE_FALLBACK")
        self.assertEqual(verdict.selected_action, "b")

    def test_constraint_compiler_flags_missing_ortools(self):
        from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler

        compiler = ConstraintLinearProjectionCompiler(hard_timeout_ms=1000.0)
        with patch.dict(sys.modules, self._blocked):
            with self.assertLogs("gen_zero.gate.constraint_compiler", level="WARNING"):
                verdict = compiler.solve_safest_action({"BUY": 0.3, "SELL": 0.6, "HOLD": 0.1})
        self.assertTrue(verdict.fallback_used)
        self.assertEqual(verdict.solver_status, "ORTOOLS_UNAVAILABLE_FALLBACK")
        self.assertEqual(verdict.selected_action, "HOLD")
        self.assertFalse(verdict.is_safe)


class TestCoRidingUnknownProbe(unittest.TestCase):
    def test_unknown_probe_raises_instead_of_hashing(self):
        from gen_zero.service.co_riding_adapter import (
            CoRidingAlignmentRequest,
            CoRidingProbesAdapter,
            NoulProbeSpec,
        )

        adapter = CoRidingProbesAdapter()
        req = CoRidingAlignmentRequest(
            entity_a={"id": "1"},
            entity_b={"id": "1"},
            probes=[NoulProbeSpec(name="made_up_probe", instructions="?", weight=1.0, critical=False)],
        )
        with self.assertRaisesRegex(ValueError, "Unknown co-riding probe 'made_up_probe'"):
            adapter.evaluate_co_riding(req)

    def test_default_probes_still_evaluate(self):
        from gen_zero.service.co_riding_adapter import CoRidingAlignmentRequest, CoRidingProbesAdapter

        adapter = CoRidingProbesAdapter()
        res = adapter.evaluate_co_riding(CoRidingAlignmentRequest(entity_a={"id": "1"}, entity_b={"id": "2"}))
        self.assertEqual(set(res.probes), {p.name for p in adapter.default_probes})


class TestTorchDegradationIsLoud(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gen_zero.client import GenZero

        cls.client = GenZero()

    def test_decide_marks_degraded_and_logs_when_torch_path_fails(self):
        def boom(*args, **kwargs):
            raise RuntimeError("simulated torch failure")

        with patch.object(type(self.client.model), "forward", boom):
            with self.assertLogs("gen_zero.client", level="WARNING") as logs:
                res = self.client.decide(state="some state text", candidates=["a", "b"], mode="reflex")
        self.assertTrue(res["degraded"])
        self.assertTrue(any("simulated torch failure" in line for line in logs.output))

    def test_decide_batch_marks_degraded_and_logs_when_torch_path_fails(self):
        def boom(*args, **kwargs):
            raise RuntimeError("simulated batch torch failure")

        with patch.object(type(self.client.model), "forward", boom):
            with self.assertLogs("gen_zero.client", level="WARNING") as logs:
                results = self.client.decide_batch(["s1", "s2"], candidates=["a", "b"], mode="reflex")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["degraded"] for r in results))
        self.assertTrue(any("simulated batch torch failure" in line for line in logs.output))

    def test_untrained_weights_marked_degraded_and_logged(self):
        self.assertFalse(self.client.weights_loaded_from_checkpoint)
        with self.assertLogs("gen_zero.client", level="WARNING") as logs:
            res = self.client.decide(state="some state text", candidates=["a", "b"], mode="reflex")
        self.assertTrue(res["degraded"])
        self.assertEqual(res["scorer"], "untrained_weights_fallback")
        self.assertEqual(res["degraded_reason"], "untrained_weights_fallback")
        self.assertFalse(res["weights_loaded_from_checkpoint"])
        self.assertEqual(res["expert_outputs"]["reflex"]["meta"]["scorer"], "untrained_weights_fallback")
        self.assertTrue(any("untrained random-init weights" in line for line in logs.output))

    def test_torch_failure_reason_kept_alongside_untrained_marker(self):
        def boom(*args, **kwargs):
            raise RuntimeError("simulated torch failure")

        with patch.object(type(self.client.model), "forward", boom):
            res = self.client.decide(state="some state text", candidates=["a", "b"], mode="reflex")
        self.assertEqual(
            res["degraded_reason"], "torch_reflex_exception:RuntimeError;untrained_weights_fallback"
        )

    def test_not_degraded_when_checkpoint_loaded(self):
        with patch.object(self.client, "weights_loaded_from_checkpoint", True):
            res = self.client.decide(state="some state text", candidates=["a", "b"], mode="reflex")
        self.assertFalse(res["degraded"])
        self.assertNotIn("degraded_reason", res)
        self.assertNotIn("scorer", res)
        self.assertEqual(res["expert_outputs"]["reflex"]["meta"]["scorer"], "torch_live_model")

    def test_decide_batch_torch_path_marks_untrained(self):
        with self.assertLogs("gen_zero.client", level="WARNING"):
            results = self.client.decide_batch(["s1", "s2"], candidates=["a", "b"], mode="reflex")
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertTrue(r["degraded"])
            self.assertEqual(r["scorer"], "untrained_weights_fallback")

    def test_decide_cpu_extreme_marks_untrained(self):
        # A raw string has no real embedding on this pure-CPU path (no fake
        # hash-derived pseudo-vector allowed); use a real feature vector instead.
        state_vec = [0.1] * self.client.config.hidden_dim
        res = self.client.decide_cpu_extreme(state_vec, candidates=["a", "b"])
        self.assertTrue(res["degraded"])
        self.assertEqual(res["scorer"], "untrained_weights_fallback")
        # A loaded dual-head alone is not enough: the INT8 scorer must have been re-synced from it.
        with patch.object(self.client, "weights_loaded_from_checkpoint", True):
            res = self.client.decide_cpu_extreme(state_vec, candidates=["a", "b"])
        self.assertTrue(res["degraded"])

    def test_decide_cpu_extreme_rejects_raw_string_without_embedder(self):
        with self.assertRaisesRegex(ValueError, "no real embedder"):
            self.client.decide_cpu_extreme("some state text", candidates=["a", "b"])

    def test_decide_cpu_extreme_rejects_dict_without_features(self):
        with self.assertRaisesRegex(ValueError, "no 'features' or 'embedding' key"):
            self.client.decide_cpu_extreme({"tag": "sensor_frame"}, candidates=["a", "b"])


class TestCheckpointLoadClearsUntrainedMarker(unittest.TestCase):
    def test_real_checkpoint_load_syncs_scorer_and_clears_marker(self):
        import tempfile
        import torch
        from gen_zero.client import GenZero

        client = GenZero()
        self.assertFalse(client.scorer_synced_from_checkpoint)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dual_head.pt")
            torch.save({"state_dict": client.model.state_dict()}, path)
            client.load_dual_head_checkpoint(path)
        self.assertTrue(client.weights_loaded_from_checkpoint)
        self.assertTrue(client.scorer_synced_from_checkpoint)
        res = client.decide(state="some state text", candidates=["a", "b"], mode="reflex")
        self.assertFalse(res["degraded"])
        self.assertNotIn("scorer", res)
        self.assertEqual(res["expert_outputs"]["reflex"]["meta"]["scorer"], "torch_live_model")
        state_vec = [0.1] * client.config.hidden_dim
        self.assertNotIn("degraded", client.decide_cpu_extreme(state_vec, candidates=["a", "b"]))


if __name__ == "__main__":
    unittest.main()
