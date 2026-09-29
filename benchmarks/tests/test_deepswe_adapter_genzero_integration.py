"""Real Gen-Zero gate/MCTS, a loopback contract server, and dispatch spies.

The HTTP server supplies deterministic test predictions, not a trained model.
Pier stubs provide only its base-class import surface; no gate is mocked.
"""
import asyncio
import importlib
import json
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
from benchmarks.deepswe_genzero_gate import CognitiveService, DeepSWEGate


def candidate(path="module.py", content="def value():\n    return 2\n"):
    return {"action": "edit", "files": [{"path": path, "content": content, "old_sha256": None}]}


@pytest.fixture
def server():
    calls = []
    state = {"mode": "ok"}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            response = {"request_id": request["request_id"]}
            if request["operation"] == "assess":
                response.update(confidence=.95, risk=.01, relevance=.9, syntax_valid=True)
            else:
                depth = request["state"]["remote"].get("depth", 0) + 1
                reward = .8 if "return 2" in json.dumps(request["candidate"]) else -.8
                response.update(state={"depth": depth}, reward=reward, value=reward, risk=.01, confidence=.95)
            if state["mode"] == "nan":
                response["risk"] = float("nan")
            if state["mode"] == "mismatch":
                response["request_id"] = "other"
            if state["mode"] == "irrelevant":
                response["relevance"] = .1
            if state["mode"] == "transition_nan" and request["operation"] == "transition":
                response["value"] = float("inf")
            if state["mode"] == "future_risk" and request["operation"] == "transition":
                response["risk"] = .9
            if state["mode"] == "missing":
                response.pop("confidence", None)
            body = json.dumps(response).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield CognitiveService(f"http://127.0.0.1:{http.server_port}/evaluate"), calls, state
    http.shutdown()
    http.server_close()
    thread.join()


@pytest.mark.parametrize("action", [
    candidate("tests/test_value.py"), candidate("evaluation/harness.py"),
    candidate("../escape.py"), candidate("/tmp/escape.py"), candidate(".git/config"),
    candidate(content="import shutil\nshutil.rmtree('tests')\n"),
    {"action": "command", "command": "rm -rf /app"},
    {"action": "command", "command": "sudo chmod 777 /"},
    {"action": "shell", "command": "true"},
    candidate(content="def broken(:"), candidate("module.rs"),
])
def test_dangerous_candidates_never_reach_scoring(action):
    evidence = []
    def forbidden(_):
        pytest.fail("dangerous candidate reached semantic service")
    gate = DeepSWEGate(evidence.append, forbidden)
    assert not gate.check(action, "repair value", semantic=True)
    assert gate.telemetry()["gen_zero_gate_used"] is True
    assert gate.blocked == 1
    assert evidence[-1]["verdict"]["passed"] is False
    assert evidence[-1]["verdict"]["triggered_rules"]


def test_real_http_gate_and_multistep_mcts_rank_candidates(server):
    service, calls, _ = server
    evidence = []
    gate = DeepSWEGate(evidence.append, service)
    good = candidate()
    bad_score = candidate(content="def value():\n    return 1\n")
    assert gate.select([bad_score, good], "repair value") == good
    assert sum(c["operation"] == "assess" for c in calls) == 2
    assert max(c["state"]["remote"].get("depth", 0) for c in calls if c["operation"] == "transition") >= 2
    assert evidence[-1]["plan"]["simulations"] == 16
    assert len(evidence[-1]["plan"]["imagined_trajectory"]) == 3
    assert gate.telemetry()["gen_zero_mcts_used"] is True
    assert gate.blocked == 0


@pytest.mark.parametrize("mode", ["nan", "mismatch", "irrelevant", "transition_nan", "future_risk", "missing"])
def test_invalid_service_evidence_fails_closed(server, mode):
    service, _, state = server
    state["mode"] = mode
    evidence = []
    gate = DeepSWEGate(evidence.append, service)
    assert gate.select([candidate()], "repair value") is None
    assert gate.blocked >= 1
    assert gate.telemetry()["gen_zero_mcts_used"] is False
    assert evidence[-1]["verdict"]["passed"] is False


def test_unavailable_service_no_fake_pass():
    for service in (None, CognitiveService("http://127.0.0.1:1/evaluate")):
        evidence = []
        gate = DeepSWEGate(evidence.append, service)
        assert gate.select([candidate()], "repair value") is None
        assert gate.used and gate.blocked == 1
        assert not gate.planner_used


@pytest.fixture
def adapter(monkeypatch, tmp_path, server):
    # Isolate absent optional Pier dependency without replacing adapter methods.
    base = ModuleType("pier.agents.base")
    base.BaseAgent = object
    installed = ModuleType("pier.agents.installed.base")
    installed.NonZeroAgentExitCodeError = RuntimeError
    monkeypatch.setitem(sys.modules, "pier.agents.base", base)
    monkeypatch.setitem(sys.modules, "pier.agents.installed.base", installed)
    module = importlib.import_module("benchmarks.gen_zero_deepswe_adapter")
    obj = object.__new__(module.GenZeroDeepSWEAdapter)
    obj.logs_dir = tmp_path
    obj.counter = obj.gate_counter = 0
    obj.repo_dir = "/app"
    obj.helper = "/tmp/gen-zero-deepswe-helper.py"
    obj.instruction = "repair value"
    obj.gate = DeepSWEGate(obj.record_gate, server[0])
    return obj


class Environment:
    def __init__(self):
        self.calls = []
    async def upload_file(self, *args):
        self.calls.append(("upload", args))
    async def exec(self, command, **kwargs):
        self.calls.append(("exec", command))
        return SimpleNamespace(return_code=0, stdout='{"patch":"real-dispatch-spy"}',
                               model_dump=lambda: {"return_code": 0})


def test_adapter_rejects_before_any_upload_or_execution(adapter):
    env = Environment()
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.action(env, candidate("harness.py")))
    assert env.calls == []
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.execute(env, "regression", "rm -rf /app"))
    assert env.calls == []
    telemetry = json.loads((adapter.logs_dir / "gen-zero-telemetry.json").read_text())
    assert telemetry["gen_zero_gate_used"] and telemetry["gen_zero_gate_blocked_count"] == 2


def test_adapter_legal_candidate_reaches_normal_dispatch(adapter):
    env = Environment()
    result = asyncio.run(adapter.action(env, candidate()))
    assert result["patch"] == "real-dispatch-spy"
    assert [c[0] for c in env.calls] == ["upload", "exec"]
    assert adapter.gate.planner_used


def test_adapter_service_failure_never_executes(adapter, server):
    server[2]["mode"] = "transition_nan"
    env = Environment()
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.action(env, candidate()))
    assert env.calls == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 2, None])
def test_strict_mcts_never_substitutes_heuristic(value):
    from gen_zero.world_model.imagination_planner import ImaginationMCTSPlanner
    planner = ImaginationMCTSPlanner(None, strict_evaluation=True)
    with pytest.raises(ValueError):
        planner._evaluate_latent_state({"value": 0}, lambda _: value)


def test_strict_mcts_propagates_evaluator_exception():
    from gen_zero.world_model.imagination_planner import ImaginationMCTSPlanner
    def unavailable(_):
        raise ConnectionError("evaluator unavailable")
    planner = ImaginationMCTSPlanner(None, strict_evaluation=True)
    with pytest.raises(ConnectionError):
        planner._evaluate_latent_state({"value": 0}, unavailable)


def test_planning_budget_fails_closed(server, monkeypatch):
    service, _, _ = server
    evidence = []
    gate = DeepSWEGate(evidence.append, service)
    ticks = iter([0, 61])
    monkeypatch.setattr("benchmarks.deepswe_genzero_gate.time", SimpleNamespace(monotonic=lambda: next(ticks)))
    assert gate.select([candidate()], "repair value") is None
    assert evidence[-1]["reason"] == "PLANNING_BUDGET_EXCEEDED"
    assert not gate.planner_used
