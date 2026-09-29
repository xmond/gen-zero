"""Pier repair adapter with an explicit external JSON Proposer.

The Proposer, not Gen-Zero's numeric world model, generates code. No oracle or
held-out verifier files are exposed. The sandbox helper has no provider secrets.
Configure proposer_url (full messages/chat-completions endpoint), proposer_model,
proposer_protocol (anthropic/openai), proposer_key_env and validation_command.
Only the disposable Pier repository is committed, for its base..HEAD collector.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import urllib.error
import urllib.request

MAX_BYTES = 1_000_000

# Prompt budget is a byte estimate: no tokenizer runs here. Source code costs about
# 3-4 bytes per token, so 80 KB is roughly 20-27k tokens and leaves the model room to
# write a full-file edit. It is a ceiling, never padded up to a floor.
PROMPT_BUDGET_BYTES = 80_000
PRUNE_AFTER_MESSAGES = 10
RECENT_MESSAGES = 6
OLDER_STRING_CAPS = (1_000, 300)
RECENT_STRING_CAPS = (8_000, 4_000, 2_000, 1_000)
LAST_STRING_CAPS = (30_000, 15_000, 8_000, 4_000, 2_000)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt_bytes(messages: list[dict]) -> int:
    return len(json.dumps(messages).encode())


def clip_strings(value, cap: int):
    """Clip every long string to head+tail with an explicit marker; keep structure."""
    if isinstance(value, str):
        if len(value) <= cap:
            return value
        head = cap // 2
        return (f"{value[:head]}...[PRUNED {len(value) - cap} of {len(value)} chars; "
                f"incomplete, never rebuild a file from this, re-read a narrower "
                f"start/end range]...{value[len(value) - (cap - head):]}")
    if isinstance(value, list):
        return [clip_strings(v, cap) for v in value]
    if isinstance(value, dict):
        return {k: clip_strings(v, cap) for k, v in value.items()}
    return value


def condense_message(message: dict, cap: int) -> dict:
    """Return a copy with long strings clipped; path and sha256 fields stay intact."""
    content = message["content"]
    try:
        parsed = json.loads(content)
    except ValueError:
        return {**message, "content": clip_strings(content, cap)}
    clipped = clip_strings(parsed, cap)
    return message if clipped == parsed else {**message, "content": json.dumps(clipped)}


def prune_messages(messages: list[dict], budget: int = PROMPT_BUDGET_BYTES) -> tuple[list[dict], dict]:
    """Sliding window for the Proposer prompt; never mutates ``messages``.

    Keeps messages[0] and the last RECENT_MESSAGES verbatim while they fit. Older turns
    are condensed, not dropped: big read sources shrink but path/sha256 stay, because an
    edit needs old_sha256 from an earlier read. Only if that is not enough are the
    oldest turn pairs dropped, then the newest observation is clipped. Every step is
    reported; if the budget is still unreachable this raises instead of sending it.
    """
    n = len(messages)
    if (n == 0 or n % 2 == 0
            or any(m.get("role") != ("user" if i % 2 == 0 else "assistant")
                   or not isinstance(m.get("content"), str) for i, m in enumerate(messages))):
        raise ValueError("Prompt history must alternate user/assistant, start and end with a user "
                         "message and hold string contents")
    cur = dict(enumerate(messages))
    kept = list(range(n))

    def size() -> int:
        return prompt_bytes([cur[i] for i in kept])

    before = size()
    older_end = max(1, n - RECENT_MESSAGES)
    condensed: set[int] = set()
    last_clipped = False

    def squeeze(indices, caps, always: bool) -> None:
        for cap in caps:
            for i in indices:
                if not always and size() <= budget:
                    return
                new = condense_message(messages[i], cap)
                if new["content"] != cur[i]["content"]:
                    cur[i] = new
                    condensed.add(i)
            always = False

    older, recent = range(1, older_end), range(older_end, n - 1)
    if n > PRUNE_AFTER_MESSAGES:
        squeeze(older, OLDER_STRING_CAPS[:1], always=True)
    squeeze(older, OLDER_STRING_CAPS, always=False)
    # Oldest turn pairs go before the recent window is touched, so the window survives
    # whole for as long as it can fit.
    while size() > budget and kept[1] < older_end:
        kept = [kept[0], *kept[3:]]
    squeeze(recent, RECENT_STRING_CAPS, always=False)
    for cap in LAST_STRING_CAPS:
        if size() <= budget:
            break
        new = condense_message(messages[-1], cap)
        if new["content"] != cur[n - 1]["content"]:
            cur[n - 1], last_clipped = new, True
    after = size()
    if after > budget:
        raise ValueError(f"Prompt is {after} bytes after pruning, over the {budget} byte budget "
                         "(messages[0] alone may be too large); refusing to send")
    dropped = n - len(kept)
    return [cur[i] for i in kept], {
        "changed": after != before or dropped > 0, "input_messages": n,
        "output_messages": len(kept), "input_bytes": before, "output_bytes": after,
        "budget_bytes": budget, "approx_tokens": after // 4,
        "condensed_message_indexes": sorted(i for i in condensed if i in kept),
        "dropped_messages": dropped, "last_message_clipped": last_clipped}


_JSON_ESCAPE = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def double_invalid_escapes(text: str) -> str:
    """Turn a lone backslash (regex/path text in code) into a literal backslash."""
    return _JSON_ESCAPE.sub(lambda m: m.group(0) if len(m.group(0)) > 1 else "\\\\", text)


def parse_proposer_answer(answer: str) -> tuple[object, list[str]]:
    """Parse the Proposer JSON; return (value, repairs applied).

    Repairs are deterministic and lossless: raw control characters inside strings
    (real newlines in file content), lone backslashes, and prose or a markdown fence
    around the object. Unescaped inner double quotes are NOT guessed: string
    boundaries are ambiguous there and a wrong guess could ship wrong code that still
    compiles, so that case raises with the original decoder error.
    """
    first, last = answer.find("{"), answer.rfind("}")
    texts = [("", answer)]
    if 0 <= first < last and answer[first:last + 1] != answer:
        texts.append(("extract_object+", answer[first:last + 1]))
    error = None
    for prefix, text in texts:
        for name, candidate, strict in (("", text, True), ("control_chars", text, False),
                                        ("invalid_escapes", double_invalid_escapes(text), False)):
            variants = [("", candidate)]
            if candidate.endswith("}"):
                variants.append(("+close_list_obj", candidate[:-1] + "}]}"))
            variants.extend([
                ("+close_brackets", candidate + "}]}"),
                ("+close_list", candidate + "]}"),
                ("+close_object", candidate + "}"),
            ])
            for suffix_name, cand_variant in variants:
                try:
                    value = json.loads(cand_variant, strict=strict)
                except json.JSONDecodeError as exc:
                    error = error or exc
                    continue
                label = (prefix + name + suffix_name).strip("+")
                return value, [label] if label else []
    raise ValueError(f"Proposer answer is not valid JSON and has no safe repair: {error}")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"git {args!r}: exit {result.returncode}: {result.stderr}")
    return result.stdout


def source_path(name: str) -> Path:
    p = PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ("..", ".git") or x.startswith(".") for x in p.parts):
        raise ValueError(f"Forbidden repository path: {name!r}")
    path = Path(name)
    if path.resolve().is_relative_to(Path.cwd().resolve()) is False:
        raise ValueError("Path escapes repository")
    if any(x.is_symlink() for x in (path, *path.parents)):
        raise ValueError("Symlink source access denied")
    return path


def sandbox_action(action: dict) -> dict:
    """Small, real repository operations; no model or benchmark-specific rules."""
    op = action["action"]
    tracked = git("ls-files", "-z").split("\0")[:-1]
    if op == "inventory":
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        if status:
            raise ValueError("Expected a clean, disposable task repository")
        return {"head": git("rev-parse", "HEAD").strip(), "files": tracked}
    if op == "search":
        query = action["query"]
        if not isinstance(query, str) or not query or len(query) > 200:
            raise ValueError("Search requires 1..200 characters")
        r = subprocess.run(["git", "grep", "-n", "-I", "-F", "-e", query, "--", "."],
                           capture_output=True, text=True)
        if r.returncode not in (0, 1):
            raise RuntimeError(f"git grep exit {r.returncode}: {r.stderr}")
        if len(r.stdout.encode()) > MAX_BYTES:
            raise ValueError("Search output exceeds budget; narrow query")
        return {"command": r.args, "return_code": r.returncode, "matches": r.stdout}
    if op == "read":
        name = action["path"]
        path = source_path(name)
        if name not in tracked:
            raise ValueError("Only tracked repository sources may be read")
        data = path.read_bytes()
        if len(data) > MAX_BYTES:
            raise ValueError("Source file exceeds byte budget")
        lines = data.decode().splitlines()
        start = max(1, action.get("start", 1))
        end = min(len(lines), action.get("end", len(lines)))
        if type(start) is not int or type(end) is not int or start > max(1, end):
            raise ValueError(f"Invalid line range {start}..{end}; file has {len(lines)} lines")
        return {"path": name, "sha256": digest(data), "total_lines": len(lines),
                "source": "\n".join(f"{i}: {lines[i-1]}" for i in range(start, min(end, len(lines))+1))}
    if op == "edit":
        edits = action["files"]
        if not isinstance(edits, list) or not 1 <= len(edits) <= 20:
            raise ValueError("Expected 1..20 complete source file edits")
        prepared = []
        seen = set()
        for edit in edits:
            name = edit["path"]
            path = source_path(name)
            if name in seen:
                raise ValueError("Duplicate edit path")
            seen.add(name)
            # Tests and repository tooling remain independent of proposed fixes.
            if any(p.lower() in ("test", "tests", "testing", "__tests__", "node_modules") for p in path.parts) or path.name.startswith("test_") or path.name.endswith("_test.py"):
                raise ValueError("Proposer may not edit tests")
            if path.suffix not in (".py", ".rs", ".go", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".cpp", ".java"):
                raise ValueError("Only source code edits are supported")
            current = path.read_bytes() if path.exists() else None
            if current is not None and name not in tracked:
                raise ValueError("Refusing to overwrite an untracked file")
            expected = digest(current) if current is not None else None
            if edit["old_sha256"] != expected:
                raise ValueError(f"Stale source hash: {name}")
            content = edit["content"].encode()
            if not content or len(content) > MAX_BYTES or content == current:
                raise ValueError("Empty, oversized or unchanged edit")
            if path.suffix == ".py":
                compile(content, name, "exec")
            prepared.append((path, content))
        # All paths, hashes and Python syntax are checked before any write.
        for path, content in prepared:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        git("add", "--", *(str(p) for p, _ in prepared))
        patch = git("diff", "--cached", "--binary")
        if not patch:
            raise ValueError("Candidate patch is empty")
        return {"changed": [str(p) for p, _ in prepared], "patch": patch}
    if op == "finalize":
        base = action["base"]
        if git("rev-parse", "HEAD").strip() != base:
            raise ValueError("Unexpected HEAD change before finalization")
        git("diff", "--cached", "--check")
        if not git("diff", "--cached", "--binary"):
            raise ValueError("Refusing empty submission")
        git("switch", "-c", "deepswe-proposer-repair")
        git("-c", "user.name=DeepSWE Proposer", "-c", "user.email=deepswe@localhost",
            "-c", "core.hooksPath=/dev/null", "commit", "-m", "Apply Proposer repair")
        patch = git("diff", "--binary", base, "HEAD")
        if not patch or git("status", "--porcelain=v1", "--untracked-files=all"):
            raise ValueError("Empty patch or dirty finalized repository")
        output = Path("/logs/artifacts/model.patch")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(patch)
        return {"head": git("rev-parse", "HEAD").strip(), "patch": patch,
                "patch_sha256": digest(patch.encode())}
    raise ValueError(f"Unknown sandbox operation: {op}")


SYSTEM = """You are a code repair Proposer. Reason from the task and real repository
observations; do not invent successful execution or use benchmark solutions.
Return exactly one JSON object, without markdown. First respond with:
{"action":"analyze","problem":"precise required behavior","targets":["function_name"]}.
Targets MUST be function or class identifiers copied verbatim from the task
instruction, NOT invented file paths. Use search/read later to locate files.
Then explore using {"action":"search","query":"literal"} or
{"action":"read","path":"relative/source.py","start":1,"end":100} (ranges optional).
Read implementation and relevant existing tests before editing. Search is literal
indexing, not reasoning. Implement general semantics, never task-name dispatch,
fixed answers, stubs, or test-specific hacks. No shell commands are available.
To edit, return {"action":"edit","reason":"explain algorithm and risks","files":[
{"path":"relative/source.py","old_sha256":"hash from read, or null for NEW file",
"content":"complete replacement file, not a diff"}]}.
Alternatively return {"action":"candidates","candidates":[edit_action, ...]}
with at most eight alternative edits for Gen-Zero ranking. Only Python edits
currently have a supported syntax validator. Existing tests/configuration cannot be edited. New source files are allowed.
After edits the configured real regression command runs automatically. Repair
failures using its real output. Finish only after nonempty edits and successful
validation: {"action":"finish","summary":"behavior implemented; limitations"}.
A passing regression suite does not prove new requirements; the separate,
hidden verifier grades those later. Do not claim to have passed hidden tests.
"""


class Proposer:
    def __init__(self, url: str, model: str, protocol: str, key_env: str):
        is_local_http = url.startswith("http://") and any(h in url for h in ("100.", "192.168.", "127.0.0.1", "localhost"))
        if (not url.startswith("https://") and not is_local_http) or protocol not in ("anthropic", "openai") or not model:
            raise ValueError("Explicit HTTPS or private LAN/Tailscale HTTP Proposer URL, model and supported protocol required")
        self.url, self.model, self.protocol = url, model, protocol
        self.key = os.environ.get(key_env) or "local-gpu-key"

    def generate(self, messages: list[dict], output: Path) -> dict:
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "max_tokens": 12288}
        if self.protocol == "anthropic":
            headers.update({"x-api-key": self.key, "anthropic-version": "2023-06-01"})
            payload["system"] = SYSTEM
        else:
            headers["Authorization"] = "Bearer " + self.key
            payload["messages"] = [{"role": "system", "content": SYSTEM}, *messages]
        write_json(output.with_suffix(".request.json"), payload)
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=840) as response:
                raw = response.read(4 * MAX_BYTES + 1)
                write_json(output.with_suffix(".http.json"), {"status": response.status})
        except urllib.error.HTTPError as exc:
            write_json(output.with_suffix(".http.json"), {"status": exc.code})
            raise RuntimeError(f"Proposer HTTP {exc.code}; no fallback") from None
        if len(raw) > 4 * MAX_BYTES:
            raise ValueError("Proposer response exceeds budget")
        output.with_suffix(".response.json").write_bytes(raw)
        data = json.loads(raw)
        if self.protocol == "anthropic":
            if data.get("stop_reason") != "end_turn":
                if data.get("stop_reason") == "max_tokens":
                    raise ActionRejected("Proposer response exceeded token length limit and was truncated. Return ONLY the concise JSON object without markdown or preamble.")
                raise ValueError(f"Proposer response incomplete: {data.get('stop_reason')}")
            answer = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        else:
            choice = data["choices"][0]
            if choice["finish_reason"] != "stop":
                if choice.get("finish_reason") == "length":
                    raise ActionRejected("Proposer response exceeded token length limit and was truncated. Return ONLY the concise JSON object without markdown or preamble.")
                raise ValueError(f"Proposer response incomplete: {choice.get('finish_reason')}")
            answer = choice["message"]["content"]
        try:
            action, repairs = parse_proposer_answer(answer)
        except ValueError as exc:
            write_json(output.with_suffix(".parse.json"), {"parsed": False, "error": str(exc)})
            raise
        write_json(output.with_suffix(".parse.json"), {"parsed": True, "repairs": repairs})
        if repairs:
            print(f"Proposer JSON repaired: {repairs}", file=sys.stderr)
        if not isinstance(action, dict) or not isinstance(action.get("action"), str):
            raise ValueError("Proposer did not return a structured action")
        return action


class ActionRejected(RuntimeError):
    """An observed sandbox rejection which can be corrected in a later turn."""


# Run the sandbox worker without importing Pier or providing any credentials.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--sandbox-action-file", type=Path)
    modes.add_argument("--proposer-request-file", type=Path)
    args = parser.parse_args()
    try:
        if args.sandbox_action_file:
            value = sandbox_action(json.loads(args.sandbox_action_file.read_text()))
        else:
            request = json.loads(args.proposer_request_file.read_text())
            value = Proposer(*request["config"]).generate(request["messages"], Path(request["output"]))
        print(json.dumps(value, ensure_ascii=False))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)

if __package__:
    from .deepswe_genzero_gate import DeepSWEGate, CognitiveService
else:
    from deepswe_genzero_gate import DeepSWEGate, CognitiveService

from pier.agents.base import BaseAgent
from pier.agents.installed.base import NonZeroAgentExitCodeError


class GenZeroDeepSWEAdapter(BaseAgent):
    def __init__(self, *args, proposer_url: str, proposer_model: str,
                 proposer_protocol: str, proposer_key_env: str,
                 validation_command: str, repo_dir: str = "/app",
                 max_steps: int = 40, gen_zero_service_url: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_name is None:
            self.model_name = proposer_model
            self._init_model_info()
        if not repo_dir.startswith("/") or not validation_command.strip():
            raise ValueError("Absolute sandbox repository and regression command required")
        self.repo_dir = repo_dir
        self.validation_command = validation_command
        self.max_steps = int(max_steps)
        if not 1 <= self.max_steps <= 100:
            raise ValueError("max_steps must be 1..100")
        self.provider_config = (proposer_url, proposer_model, proposer_protocol, proposer_key_env)
        self.counter = 0
        self.gate_counter = 0
        self.instruction = ""
        self.gate = DeepSWEGate(self.record_gate,
                                CognitiveService(gen_zero_service_url) if gen_zero_service_url else None)
        self.helper = "/tmp/gen-zero-deepswe-helper.py"
        self.instruction_path = "/tmp/gen-zero-deepswe-instruction.md"

    def record_gate(self, evidence):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.gate_counter += 1
        write_json(self.logs_dir / f"gate-{self.gate_counter:04d}.json", evidence)
        write_json(self.logs_dir / "gen-zero-telemetry.json", self.gate.telemetry())

    @staticmethod
    def name():
        return "gen-zero-deepswe"

    def version(self):
        return "0.3.0-genzero-gated-proposer"

    async def execute(self, environment, label, command, timeout=60, accepted=(0,)):
        if not self.gate.check({"action": "command", "command": command}, self.instruction,
                               semantic=bool(self.gate.service)):
            raise ActionRejected("Gen-Zero command rejected; see gate evidence")
        self.counter += 1
        result = await environment.exec(command, cwd=self.repo_dir, timeout_sec=timeout)
        write_json(self.logs_dir / f"exec-{self.counter:03d}-{label}.json",
                   {"command": command, "cwd": self.repo_dir, **result.model_dump()})
        if result.return_code not in accepted:
            raise RuntimeError(f"{label} exited {result.return_code}; see exec-{self.counter:03d}-{label}.json")
        return result

    async def action(self, environment, action):
        if action.get("action") in ("edit", "candidates"):
            candidates = action.get("candidates") if action.get("action") == "candidates" else [action]
            if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 8
                    or any(not isinstance(c, dict) or c.get("action") != "edit" for c in candidates)):
                self.gate.stop("INVALID_CANDIDATE_SET")
                raise ActionRejected("Invalid candidate set")
            action = self.gate.select(candidates, self.instruction)
            if action is None:
                raise ActionRejected("Gen-Zero rejected all candidates or planning failed; see gate evidence")
        elif not self.gate.check(action):
            raise ActionRejected("Gen-Zero action rejected; see gate evidence")
        request_path = self.logs_dir / f"action-{self.counter + 1:03d}.json"
        write_json(request_path, action)
        sandbox_request = "/tmp/gen-zero-deepswe-action.json"
        await environment.upload_file(request_path, sandbox_request)
        result = await self.execute(environment, action["action"],
                                    f"python3 {self.helper} --sandbox-action-file {sandbox_request}",
                                    accepted=(0, 1))
        if result.return_code:
            raise ActionRejected(f"exit {result.return_code}: {result.stdout or ''}{result.stderr or ''}")
        observation = json.loads(result.stdout)
        if action["action"] in ("inventory", "read", "edit"):
            self.gate.observations[action.get("path", action["action"])] = observation
        return observation

    async def setup(self, environment):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.logs_dir / "adapter-identity.json", {
            "sha256": digest(Path(__file__).read_bytes()), "version": self.version(),
            "code_generator": "external-proposer", **self.gate.telemetry()})
        await environment.upload_file(Path(__file__).resolve(), self.helper)
        result = await self.execute(environment, "root", "git rev-parse --show-toplevel")
        if result.stdout.strip() != self.repo_dir:
            raise ValueError("Sandbox repository root mismatch")
        self.inventory = await self.action(environment, {"action": "inventory"})
        write_json(self.logs_dir / "inventory.json", self.inventory)

    async def propose(self, messages, step):
        """Enforce a wall-clock deadline; socket timeouts alone are insufficient."""
        prefix = (self.logs_dir / f"proposer-{step:03d}").resolve()
        request = prefix.with_suffix(".input.json")
        write_json(request, {"config": self.provider_config, "messages": messages,
                             "output": str(prefix)})
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--proposer-request-file", str(request)]
        write_json(prefix.with_suffix(".command.json"), command)
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = b"", b""
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=900)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()
            stdout, stderr = await process.communicate()
            raise
        finally:
            prefix.with_suffix(".stdout").write_bytes(stdout)
            prefix.with_suffix(".stderr").write_bytes(stderr)
        if process.returncode != 0:
            err = stderr.decode().strip()
            if "ActionRejected:" in err:
                msg = err.split("ActionRejected:", 1)[1].strip()
                raise ActionRejected(msg)
            raise RuntimeError(f"Proposer process exited {process.returncode}: {err}")
        return json.loads(stdout)

    async def run(self, instruction, environment, context):
        self.instruction = instruction
        context.metadata = {"status": "running", "code_generator": "external-proposer"}
        try:
            # Pier passes instruction.md contents to run(); stage and read the
            # same bytes in the sandbox so provenance is explicit and verifiable.
            local_instruction = self.logs_dir / "instruction.md"
            local_instruction.write_text(instruction)
            await environment.upload_file(local_instruction, self.instruction_path)
            actual = await self.execute(environment, "instruction", "cat " + self.instruction_path)
            if actual.stdout != instruction:
                raise ValueError("Sandbox instruction differs from Pier input")
            baseline = await self.execute(environment, "baseline", self.validation_command,
                                          timeout=1200)
            write_json(self.logs_dir / "baseline.json", baseline.model_dump())
            messages = [{"role": "user", "content": json.dumps({
                "instruction": actual.stdout, "repository": self.inventory,
                "regression_command": self.validation_command})}]
            analyzed, edited, validated = False, False, False
            for step in range(1, self.max_steps + 1):
                # The full history stays in `messages`; only the prompt is windowed, and
                # every pruning is logged. An unreachable budget raises (fail closed).
                prompt, pruning = prune_messages(messages)
                if pruning["changed"]:
                    write_json(self.logs_dir / f"prune-{step:03d}.json", pruning)
                    self.logger.warning("Proposer prompt pruned at step %d: %s", step, pruning)
                try:
                    action = await self.propose(prompt, step)
                except ActionRejected as exc:
                    rejection = {"rejected": str(exc), "instruction": "Respond ONLY with a valid JSON object matching the required schema, without markdown or preamble."}
                    write_json(self.logs_dir / f"rejection-{step:03d}.json", rejection)
                    self.logger.warning("Proposer action rejected: %s", rejection)
                    messages.append({"role": "user", "content": json.dumps(rejection)})
                    continue
                context.n_agent_steps = step
                messages.append({"role": "assistant", "content": json.dumps(action)})
                op = action["action"]
                if not analyzed and op == "analyze":
                    if not isinstance(action.get("problem"), str) or not action["problem"].strip():
                        rejection = {"rejected": "First Proposer action must explain the problem in 'problem'"}
                        write_json(self.logs_dir / f"rejection-{step:03d}.json", rejection)
                        self.logger.warning("Proposer action rejected: %s", rejection)
                        messages.append({"role": "user", "content": json.dumps(rejection)})
                        continue
                    targets = action.get("targets")
                    if not isinstance(targets, list) or not targets:
                        found = re.findall(r'`([a-zA-Z_][a-zA-Z0-9_]*)`', instruction)
                        targets = [t for t in found if t in instruction][:6]
                    else:
                        targets = [t for t in targets if isinstance(t, str) and t in instruction]
                    if not targets:
                        targets = ["def "]
                    action["targets"] = targets
                    write_json(self.logs_dir / "task-analysis.json", action)
                    observation = {"target_locations": [await self.action(environment, {"action": "search", "query": t}) for t in targets]}
                    analyzed = True
                elif op in ("search", "read", "edit", "candidates"):
                    analyzed = True
                    try:
                        observation = await self.action(environment, action)
                    except ActionRejected as exc:
                        rejection = {"rejected": str(exc), "instruction": "Correct the invalid action; do not bypass the check."}
                        write_json(self.logs_dir / f"rejection-{step:03d}.json", rejection)
                        self.logger.warning("Proposer action rejected: %s", rejection)
                        messages.append({"role": "user", "content": json.dumps(rejection)})
                        if op in ("edit", "candidates"):
                            validated = False
                        continue
                    if op in ("edit", "candidates"):
                        edited, validated = True, False
                        result = await self.execute(environment, "regression", self.validation_command,
                                                    timeout=1200, accepted=tuple(range(256)))
                        observation["regression"] = result.model_dump()
                        validated = result.return_code == 0
                        write_json(self.logs_dir / f"validation-{step:03d}.json", result.model_dump())
                        (self.logs_dir / "candidate.patch").write_text(observation["patch"])
                elif op == "analyze":
                    rejection = {
                        "rejected": "Problem analysis is already completed. Target search locations were returned in the previous turn.",
                        "instruction": "Do not call 'analyze' again. Proceed with {'action':'search','query':'...'} or {'action':'read','path':'...'}."
                    }
                    write_json(self.logs_dir / f"rejection-{step:03d}.json", rejection)
                    self.logger.warning("Proposer action rejected: %s", rejection)
                    messages.append({"role": "user", "content": json.dumps(rejection)})
                    continue
                elif op == "finish":
                    if not edited or not validated:
                        rejection = {
                            "rejected": "Cannot finish without nonempty file edits and passing regression tests.",
                            "instruction": "Make edits using {'action':'edit',...} and verify tests before finishing."
                        }
                        write_json(self.logs_dir / f"rejection-{step:03d}.json", rejection)
                        self.logger.warning("Proposer action rejected: %s", rejection)
                        messages.append({"role": "user", "content": json.dumps(rejection)})
                        continue
                    final = await self.action(environment, {"action": "finalize", "base": self.inventory["head"]})
                    (self.logs_dir / "model.patch").write_text(final.pop("patch"))
                    context.metadata = {"status": "patch_generated", "code_generator": "external-proposer",
                                        "regression_passed": True, "hidden_verifier_passed": None, **final, **self.gate.telemetry()}
                    write_json(self.logs_dir / "completion.json", context.metadata)
                    return
                else:
                    rejection = {
                        "rejected": f"Unsupported action '{op}'.",
                        "instruction": "Allowed actions are 'search', 'read', 'edit', 'candidates', 'finish'."
                    }
                    write_json(self.logs_dir / f"rejection-{step:03d}.json", rejection)
                    self.logger.warning("Proposer action rejected: %s", rejection)
                    messages.append({"role": "user", "content": json.dumps(rejection)})
                    continue
                messages.append({"role": "user", "content": json.dumps(observation)})
            raise RuntimeError("Proposer step budget exhausted")
        except Exception as exc:
            context.metadata = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", **self.gate.telemetry()}
            write_json(self.logs_dir / "failure.json", context.metadata)
            self.logger.error("DeepSWE fail-closed: %s", context.metadata["error"])
            raise NonZeroAgentExitCodeError(context.metadata["error"]) from exc
