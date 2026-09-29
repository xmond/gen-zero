"""Context pruning and JSON repair for the DeepSWE adapter.

Pure functions are exercised directly. The Proposer is exercised for real through
its CLI entry point against a loopback HTTPS server (self-signed certificate,
trusted through SSL_CERT_FILE), and ``run`` is driven through that same real
subprocess. Only the Pier sandbox is a spy: no Docker here.
"""
import asyncio
import hashlib
import importlib
import json
import logging
from pathlib import Path
import ssl
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deepswe_adapter_genzero_integration import server  # noqa: E402,F401  (real gate service fixture)

ADAPTER = ROOT / "benchmarks" / "gen_zero_deepswe_adapter.py"


@pytest.fixture
def mod(monkeypatch):
    # Pier is not installed here; stub only its base-class import surface.
    base = ModuleType("pier.agents.base")
    base.BaseAgent = object
    installed = ModuleType("pier.agents.installed.base")
    installed.NonZeroAgentExitCodeError = RuntimeError
    monkeypatch.setitem(sys.modules, "pier.agents.base", base)
    monkeypatch.setitem(sys.modules, "pier.agents.installed.base", installed)
    return importlib.import_module("benchmarks.gen_zero_deepswe_adapter")


def user(payload) -> dict:
    return {"role": "user", "content": json.dumps(payload)}


def assistant(payload) -> dict:
    return {"role": "assistant", "content": json.dumps(payload)}


def read_obs(name: str, size: int) -> dict:
    source = ("0123456789abcdef" * 4 + "\n") * (size // 65)
    return {"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "total_lines": size // 65, "source": source}


def history(cycles: int, observation_size: int) -> list[dict]:
    """u0 then (assistant read, user observation) pairs, as run() builds it."""
    out = [user({"instruction": "fix items", "repository": {"files": ["items.py"]},
                 "regression_command": "pytest"})]
    for k in range(cycles):
        out.append(assistant({"action": "read", "path": f"mod{k}.py"}))
        out.append(user(read_obs(f"mod{k}.py", observation_size)))
    return out


def alternates(messages) -> bool:
    return (len(messages) % 2 == 1 and all(
        m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(messages)))


# ---- sliding window -------------------------------------------------------

def test_old_70kb_reads_condensed_first_and_recent_six_kept_verbatim(mod):
    messages = history(12, 70_000)
    # Make the newest six messages small so they can all survive verbatim.
    for i in range(len(messages) - 6, len(messages)):
        if messages[i]["role"] == "user":
            messages[i] = user(read_obs(f"recent{i}.py", 2_000))
    snapshot = json.dumps(messages)
    pruned, report = mod.prune_messages(messages)
    assert json.dumps(messages) == snapshot, "input must not be mutated"
    assert pruned[0] == messages[0]
    assert pruned[-6:] == messages[-6:]
    assert alternates(pruned) and len(pruned) == len(messages)
    assert report["changed"] and report["output_bytes"] <= mod.PROMPT_BUDGET_BYTES
    assert report["input_bytes"] > 600_000
    assert report["dropped_messages"] == 0
    middle = pruned[1:-6]
    assert any("PRUNED" in m["content"] for m in middle)
    # path and sha256 survive so a later edit can still name old_sha256
    survived = [json.loads(m["content"]) for m in middle if m["role"] == "user"]
    assert all(len(o["sha256"]) == 64 and o["path"].startswith("mod") for o in survived)
    assert all(len(o["source"]) < 1_200 for o in survived)


def test_recent_seventy_kb_reads_are_squeezed_but_newest_observation_is_whole(mod):
    messages = history(13, 70_000)
    pruned, report = mod.prune_messages(messages)
    assert report["output_bytes"] <= mod.PROMPT_BUDGET_BYTES
    assert report["approx_tokens"] <= 25_000
    assert pruned[-1] == messages[-1], "newest observation stays complete when it fits"
    assert pruned[0] == messages[0] and alternates(pruned)
    assert report["last_message_clipped"] is False


def test_short_history_under_budget_is_untouched(mod):
    messages = history(3, 500)
    pruned, report = mod.prune_messages(messages)
    assert pruned == messages and report["changed"] is False


def test_very_long_history_drops_oldest_pairs_and_stays_valid(mod):
    messages = history(400, 2_000)
    pruned, report = mod.prune_messages(messages)
    assert report["dropped_messages"] > 0 and report["dropped_messages"] % 2 == 0
    assert pruned[0] == messages[0] and pruned[-6:] == messages[-6:]
    assert alternates(pruned) and report["output_bytes"] <= mod.PROMPT_BUDGET_BYTES


def test_oversized_newest_observation_is_clipped_with_explicit_marker(mod):
    messages = history(4, 2_000)
    messages[-1] = user(read_obs("big.py", 1_000_000))
    pruned, report = mod.prune_messages(messages)
    assert report["last_message_clipped"] and report["output_bytes"] <= mod.PROMPT_BUDGET_BYTES
    body = json.loads(pruned[-1]["content"])
    assert "PRUNED" in body["source"] and "never rebuild a file" in body["source"]
    assert body["sha256"] == json.loads(messages[-1]["content"])["sha256"]


def test_unreachable_budget_raises_instead_of_sending(mod):
    messages = history(2, 100)
    messages[0] = user({"instruction": "x" * 200_000})
    with pytest.raises(ValueError, match="refusing to send"):
        mod.prune_messages(messages)


@pytest.mark.parametrize("bad", [
    [], [{"role": "user", "content": "{}"}, {"role": "assistant", "content": "{}"}],
    [{"role": "assistant", "content": "{}"}],
    [{"role": "user", "content": "{}"}, {"role": "user", "content": "{}"}, {"role": "user", "content": "{}"}],
    [{"role": "user", "content": ["not", "a", "string"]}],
])
def test_malformed_history_fails_closed(mod, bad):
    with pytest.raises(ValueError):
        mod.prune_messages(bad)


# ---- JSON repair ----------------------------------------------------------

PY_SOURCE = 'import re\n\ndef total(items):\n\tpattern = re.compile(r"\\d+")\n\treturn sum(int(x) for x in pattern.findall(items))\n'


def raw_newline_answer(source: str) -> str:
    """What a model emits: JSON whose content string holds real newlines and tabs."""
    return ('{"action":"edit","reason":"fix","files":[{"path":"items.py","old_sha256":null,'
            '"content":"' + source.replace('"', '\\"') + '"}]}')


def test_real_newlines_in_code_string_are_repaired_and_round_trip(mod):
    answer = raw_newline_answer("def f():\n    return 1\n")
    with pytest.raises(json.JSONDecodeError, match="Invalid control character"):
        json.loads(answer)
    action, repairs = mod.parse_proposer_answer(answer)
    assert repairs == ["control_chars"]
    content = action["files"][0]["content"]
    assert content == "def f():\n    return 1\n"
    compile(content, "items.py", "exec")


def test_newlines_tabs_and_lone_backslashes_together(mod):
    answer = raw_newline_answer(PY_SOURCE)
    action, repairs = mod.parse_proposer_answer(answer)
    assert repairs == ["invalid_escapes"]
    content = action["files"][0]["content"]
    assert content == PY_SOURCE
    compile(content, "items.py", "exec")


def test_valid_escapes_are_not_disturbed(mod):
    answer = json.dumps({"action": "edit", "files": [{"path": "a.py", "old_sha256": None,
                                                        "content": "s = '\\\\d\\n'\nx = 1\n"}]})
    action, repairs = mod.parse_proposer_answer(answer)
    assert repairs == [] and action == json.loads(answer)


def test_markdown_fence_and_prose_are_extracted_and_reported(mod):
    body = raw_newline_answer("x = 1\n")
    action, repairs = mod.parse_proposer_answer("Here you go:\n```json\n" + body + "\n```\n")
    assert repairs == ["extract_object+control_chars"] and action["files"][0]["content"] == "x = 1\n"


def test_unescaped_inner_quotes_are_not_guessed(mod):
    answer = ('{"action":"edit","files":[{"path":"a.py","old_sha256":null,'
              '"content":"print("hi", "there")\n"}]}')
    with pytest.raises(ValueError, match="no safe repair"):
        mod.parse_proposer_answer(answer)


def test_garbage_and_truncated_json_fail_closed(mod):
    for answer in ("", "no json here", '{"action":"edit","files":[{"path":"a.py","content":"abc'):
        with pytest.raises(ValueError, match="no safe repair"):
            mod.parse_proposer_answer(answer)


# ---- real Proposer over HTTPS, and run() wiring --------------------------

@pytest.fixture
def https_proposer(tmp_path, monkeypatch):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=127.0.0.1",
                    "-addext", "subjectAltName=IP:127.0.0.1"], check=True, capture_output=True)
    state = {"received": [], "script": lambda n, payload: "{}"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["received"].append(payload)
            text = state["script"](len(state["received"]), payload)
            body = json.dumps({"stop_reason": "end_turn",
                               "content": [{"type": "text", "text": text}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    http.socket = context.wrap_socket(http.socket, server_side=True)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    monkeypatch.setenv("DEEPSWE_TEST_KEY", "test-key")
    state["url"] = f"https://127.0.0.1:{http.server_port}/v1/messages"
    yield state
    http.shutdown()
    http.server_close()
    thread.join()


def run_cli_proposer(state, tmp_path, messages):
    request = tmp_path / "req.json"
    request.write_text(json.dumps({"config": [state["url"], "m", "anthropic", "DEEPSWE_TEST_KEY"],
                                   "messages": messages, "output": str(tmp_path / "out")}))
    return subprocess.run([sys.executable, str(ADAPTER), "--proposer-request-file", str(request)],
                          capture_output=True, text=True)


def test_real_proposer_process_accepts_multiline_code_answer(https_proposer, tmp_path):
    source = PY_SOURCE
    https_proposer["script"] = lambda n, p: raw_newline_answer(source)
    result = run_cli_proposer(https_proposer, tmp_path, [user({"instruction": "go"})])
    assert result.returncode == 0, result.stderr
    action = json.loads(result.stdout)
    assert action["files"][0]["content"] == source
    compile(action["files"][0]["content"], "items.py", "exec")
    assert json.loads((tmp_path / "out.parse.json").read_text()) == {
        "parsed": True, "repairs": ["invalid_escapes"]}
    assert "Proposer JSON repaired" in result.stderr


def test_real_proposer_process_rejects_unrepairable_answer(https_proposer, tmp_path):
    https_proposer["script"] = lambda n, p: '{"action":"edit","files":[{"content":"print("a")"}]}'
    result = run_cli_proposer(https_proposer, tmp_path, [user({"instruction": "go"})])
    assert result.returncode == 1 and result.stdout == ""
    assert "no safe repair" in result.stderr
    assert json.loads((tmp_path / "out.parse.json").read_text())["parsed"] is False


class Sandbox:
    """Pier sandbox spy: echoes the instruction and answers read/search with 70KB sources."""
    def __init__(self, instruction):
        self.instruction, self.uploads = instruction, {}

    async def upload_file(self, src, dst):
        self.uploads[str(dst)] = Path(src).read_text()

    async def exec(self, command, **kwargs):
        def done(stdout):
            return SimpleNamespace(return_code=0, stdout=stdout, stderr="",
                                   model_dump=lambda: {"return_code": 0, "stdout": stdout[:80]})
        if command.startswith("cat "):
            return done(self.instruction)
        if "--sandbox-action-file" in command:
            action = json.loads(self.uploads["/tmp/gen-zero-deepswe-action.json"])
            if action["action"] == "read":
                return done(json.dumps(read_obs(action["path"], 70_000)))
            return done(json.dumps({"return_code": 0, "matches": "module.py:1: def value()"}))
        return done("")


def test_run_prunes_prompts_sent_to_the_real_proposer(mod, server, https_proposer, tmp_path):
    instruction = "Repair value in module.py"

    def script(n, payload):
        if n == 1:
            return json.dumps({"action": "analyze", "problem": "value is wrong", "targets": ["value"]})
        return json.dumps({"action": "read", "path": f"pkg/mod{n}.py"})

    https_proposer["script"] = script
    obj = object.__new__(mod.GenZeroDeepSWEAdapter)
    obj.logs_dir, obj.counter, obj.gate_counter = tmp_path, 0, 0
    obj.repo_dir, obj.helper = "/app", "/tmp/gen-zero-deepswe-helper.py"
    obj.instruction_path = "/tmp/gen-zero-deepswe-instruction.md"
    obj.instruction, obj.validation_command, obj.max_steps = "", "true", 14
    obj.inventory = {"head": "h", "files": ["module.py"]}
    obj.logger = logging.getLogger("deepswe-test")
    obj.provider_config = (https_proposer["url"], "m", "anthropic", "DEEPSWE_TEST_KEY")
    obj.gate = mod.DeepSWEGate(obj.record_gate, server[0])
    context = SimpleNamespace(metadata=None)

    with pytest.raises(RuntimeError, match="step budget exhausted"):
        asyncio.run(obj.run(instruction, Sandbox(instruction), context))

    received = https_proposer["received"]
    assert len(received) == 14
    sizes = [len(json.dumps(r["messages"]).encode()) for r in received]
    assert max(sizes) <= mod.PROMPT_BUDGET_BYTES, sizes
    first_user = received[0]["messages"][0]
    assert all(r["messages"][0] == first_user for r in received)
    assert all(alternates(r["messages"]) for r in received)
    late = received[-1]["messages"]
    assert len(late) < 27, "oldest pairs must have been dropped"
    assert json.loads(late[-1]["content"])["source"].count("\n") > 1000, \
        "newest 70KB observation is complete"
    assert any("PRUNED" in m["content"] for m in late[1:])
    reports = sorted(tmp_path.glob("prune-*.json"))
    assert reports, "pruning must leave a report per step"
    assert all(json.loads(r.read_text())["output_bytes"] <= mod.PROMPT_BUDGET_BYTES for r in reports)
    assert json.loads((tmp_path / "failure.json").read_text())["status"] == "failed"
