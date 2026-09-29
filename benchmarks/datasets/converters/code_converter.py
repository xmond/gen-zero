"""APPS train code-execution prediction; HumanEval has no official train split.

Each record shows a problem statement, one APPS reference solution and one
stdin, and asks which option is exactly what that program prints. Nothing is
taken on trust from the dataset: the reference solution is executed and a case
is kept only when its output equals the official expected output. Wrong options
are real program outputs too: outputs of the same program on the problem's
other test inputs, and outputs of single-site AST mutants of the program on the
same stdin (near-miss executions, e.g. ``<`` became ``<=``).

APPS statements quote their sample tests, and the first official cases are often
those samples. A case whose stdin or stdout appears in the statement is dropped,
so kept cases are hidden tests. Texts under 2 characters are exempt from that
rule, so a wrong option is kept only when it appears in the statement exactly
when the answer does: no option can be found, or ruled out, by searching it.

Every problem ends in exactly one ``problem.*`` reason and every stdin/stdout
case of a problem that reaches the case stage ends in exactly one ``case.*``
reason, so nothing is dropped without a count.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .common import EVALUATION, exact_hash, isolated, shuffled

# APPS train holds integer literals with more than 4300 digits (one has 9131).
sys.set_int_max_str_digits(100000)

APPS_URL = "https://huggingface.co/datasets/codeparrot/apps/resolve/main/train.jsonl"

# The child sets its own limits: preexec_fn is not safe with a thread pool.
_RUNNER = (
    "import resource, runpy, sys\n"
    "mem, out = int(sys.argv[2]), int(sys.argv[3])\n"
    "resource.setrlimit(resource.RLIMIT_AS, (mem, mem))\n"
    "resource.setrlimit(resource.RLIMIT_FSIZE, (out, out))\n"
    "path = sys.argv[1]\n"
    "sys.argv = [path]\n"
    "runpy.run_path(path, run_name='__main__')\n"
)


@dataclass(frozen=True)
class AppsConfig:
    max_solutions: int = 3
    max_cases: int = 10
    max_mutants: int = 6
    max_candidates: int = 4
    max_stdin_chars: int = 2000
    max_stdout_chars: int = 1000
    solution_timeout: float = 10.0
    mutant_timeout: float = 2.0
    memory_bytes: int = 2 << 30
    output_bytes: int = 8 << 20


def normalize_output(value) -> str:
    """APPS stdout as text: list elements are lines; trailing spaces and blank tail lines do not count."""
    if isinstance(value, list):
        if not all(isinstance(line, str) for line in value):
            raise ValueError("invalid APPS output lines")
        value = "\n".join(value)
    if not isinstance(value, str):
        raise ValueError("invalid APPS output")
    lines = [line.rstrip() for line in value.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def as_stdin(value) -> str:
    if isinstance(value, list):
        if not all(isinstance(line, str) for line in value):
            raise ValueError("invalid APPS input lines")
        return "\n".join(value)
    if not isinstance(value, str):
        raise ValueError("invalid APPS input")
    return value


def run_program(code: str, stdin: str, *, timeout: float, config: AppsConfig = AppsConfig()) -> tuple[str, str]:
    """Run code as a script in a fresh interpreter. Returns (status, stdout); status is ok, error or timeout."""
    with tempfile.TemporaryDirectory() as work:
        script, out_path = os.path.join(work, "main.py"), os.path.join(work, "stdout")
        with open(script, "w", encoding="utf-8") as stream:
            stream.write(code)
        command = [sys.executable, "-I", "-X", "int_max_str_digits=0", "-c", _RUNNER,
                   script, str(config.memory_bytes), str(config.output_bytes)]
        with open(out_path, "wb") as out:
            process = subprocess.Popen(command, cwd=work, stdin=subprocess.PIPE, stdout=out,
                                       stderr=subprocess.DEVNULL, start_new_session=True,
                                       env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"})
            try:
                process.communicate(stdin.encode("utf-8"), timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return "timeout", ""
            except BrokenPipeError:
                # The program exited without reading all of stdin.
                process.wait()
        with open(out_path, "rb") as stream:
            stdout = stream.read().decode("utf-8", errors="replace")
        return ("ok" if process.returncode == 0 else "error"), stdout


_COMPARE = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BINARY = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}
_BOOLEAN = {ast.And: ast.Or, ast.Or: ast.And}


def _mutation_site(node) -> bool:
    if isinstance(node, ast.Compare):
        return type(node.ops[0]) in _COMPARE
    if isinstance(node, (ast.BinOp, ast.AugAssign)):
        return type(node.op) in _BINARY
    if isinstance(node, ast.BoolOp):
        return type(node.op) in _BOOLEAN
    return isinstance(node, ast.Constant) and type(node.value) is int


def _mutate(node) -> None:
    if isinstance(node, ast.Compare):
        node.ops[0] = _COMPARE[type(node.ops[0])]()
    elif isinstance(node, (ast.BinOp, ast.AugAssign)):
        node.op = _BINARY[type(node.op)]()
    elif isinstance(node, ast.BoolOp):
        node.op = _BOOLEAN[type(node.op)]()
    else:
        node.value += 1


def mutants(code: str, key: str, limit: int) -> list[str]:
    """Up to ``limit`` single-site mutants of code; sites are picked by a hash of key, not by position."""
    tree = ast.parse(code)
    sites = [index for index, node in enumerate(ast.walk(tree)) if _mutation_site(node)]
    sites.sort(key=lambda index: exact_hash(f"{key}:{index}"))
    original, result = ast.unparse(tree), []
    for index in sites:
        if len(result) == limit:
            break
        variant = copy.deepcopy(tree)
        _mutate(list(ast.walk(variant))[index])
        text = ast.unparse(variant)
        if text != original and text not in result:
            result.append(text)
    return result


def _parsable(code) -> bool:
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):  # Python 2 solutions, null bytes
        return False
    return True


def _load_json_field(value, empty: str, invalid: str):
    """Returns (parsed, reason); reason is None on success."""
    if isinstance(value, str):
        if not value.strip():
            return None, empty
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None, invalid
    return value, None


def _flat(text: str) -> str:
    return " ".join(text.casefold().split())


def _in_question(text: str, question: str) -> bool:
    """Whether text is quoted in the flattened question. Case and whitespace are ignored and
    word boundaries are not required: a looser match drops more, the safe side for leakage.
    Texts under 2 characters are exempt, since a single digit or letter occurs in almost
    every statement and says nothing about which option is right."""
    text = _flat(text)
    return len(text) >= 2 and text in question


def _problem(index: int, source: dict, config: AppsConfig):
    """Convert one APPS problem. Returns (records, stats)."""
    stats = Counter()

    def finish(reason):
        stats[f"problem.{reason}"] += 1
        return [], stats

    question = source.get("question")
    if not isinstance(question, str) or not question.strip():
        return finish("no_question")
    cases, reason = _load_json_field(source.get("input_output"), "io_empty", "io_invalid_json")
    if reason:
        return finish(reason)
    if not isinstance(cases, dict):
        return finish("io_invalid_json")
    inputs, outputs = cases.get("inputs"), cases.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise ValueError(f"invalid APPS input_output at {index}")
    if cases.get("fn_name"):
        # Function-call problems need a return-value rendering contract; not built yet.
        return finish("fn_call_unsupported")

    flat_question = _flat(question)
    usable = []
    for case_index, (raw_in, raw_out) in enumerate(zip(inputs, outputs)):
        stdin, stdout = as_stdin(raw_in), normalize_output(raw_out)
        if not stdout.strip():
            stats["case.empty_stdout"] += 1
        elif _in_question(stdin, flat_question) or _in_question(stdout, flat_question):
            stats["case.in_question_example"] += 1
        elif len(stdin) > config.max_stdin_chars:
            stats["case.stdin_too_long"] += 1
        elif len(stdout) > config.max_stdout_chars:
            stats["case.stdout_too_long"] += 1
        elif len(usable) == config.max_cases:
            stats["case.over_case_cap"] += 1
        else:
            usable.append((case_index, stdin, stdout))
    if not usable:
        return finish("no_usable_case")

    def drop_usable(case_reason, problem_reason):
        stats[f"case.{case_reason}"] += len(usable)
        return finish(problem_reason)

    solutions, reason = _load_json_field(source.get("solutions"), "solutions_empty", "solutions_invalid_json")
    if reason:
        return drop_usable("no_solution", reason)
    if not isinstance(solutions, list) or not solutions:
        return drop_usable("no_solution", "solutions_empty")
    parsable = [(i, code) for i, code in enumerate(solutions) if _parsable(code)]
    stats["solution.unparsable"] += len(solutions) - len(parsable)
    if not parsable:
        return drop_usable("no_solution", "no_parsable_solution")

    # Pick the reference solution that reproduces the most official outputs.
    best = None
    for solution_index, code in parsable[:config.max_solutions]:
        passed = []
        for case in usable:
            status, printed = run_program(code, case[1], timeout=config.solution_timeout, config=config)
            if status == "ok" and normalize_output(printed) == case[2]:
                passed.append(case)
                stats["solution_run.match"] += 1
            else:
                stats[f"solution_run.{'mismatch' if status == 'ok' else status}"] += 1
        if passed and (best is None or len(passed) > len(best[2])):
            best = (solution_index, code, passed)
        if len(passed) == len(usable):
            break
    if best is None:
        return drop_usable("solution_mismatch", "no_verified_solution")
    solution_index, code, verified = best
    stats["case.solution_mismatch"] += len(usable) - len(verified)

    key = str(source.get("id", index))
    variants = mutants(code, key, config.max_mutants)
    records = []
    for case_index, stdin, stdout in verified:
        pool = {}
        for other in verified:
            if other[2] != stdout:
                pool.setdefault(other[2], "other_case")
        for variant in variants:
            status, printed = run_program(variant, stdin, timeout=config.mutant_timeout, config=config)
            printed = normalize_output(printed)
            stats[f"mutant_run.{status}"] += 1
            if status == "ok" and printed.strip() and printed != stdout and len(printed) <= config.max_stdout_chars:
                pool.setdefault(printed, "mutant")
        # Otherwise "pick the option the statement quotes" (or "never pick it") beats chance.
        # Plain containment, no length threshold: a 1-character answer such as "1" is
        # usually quoted, so its wrong options must be quoted too.
        quoted = _flat(stdout) in flat_question
        mismatched = [text for text in pool if (_flat(text) in flat_question) != quoted]
        stats["distractor.question_mismatch"] += len(mismatched)
        for text in mismatched:
            del pool[text]
        if not pool:
            stats["case.no_distractor"] += 1
            continue
        record_id = f"apps-train-{index:05d}-{case_index:03d}"
        # Prefer distractors with the answer's line count, then same-stdin mutants:
        # otherwise the answer can often be picked by counting lines, with no execution.
        lines = stdout.count("\n")
        chosen = sorted(pool, key=lambda text: (text.count("\n") != lines, pool[text] != "mutant",
                                                exact_hash(f"{record_id}:{text}")))[:config.max_candidates - 1]
        records.append({
            "id": record_id,
            "task": "apps_execution_prediction",
            "context": (f"{question.strip()}\n\nPython program:\n```python\n{code.strip()}\n```\n\n"
                        f"Standard input:\n```\n{stdin}\n```\n\n"
                        "Which option is exactly what this program prints to standard output?"),
            "candidates": shuffled([stdout, *chosen], record_id),
            "ground_truth": stdout,
            "metadata": {"source": "codeparrot/apps", "split": "train", "source_index": index,
                         "apps_id": source.get("id"), "case_index": case_index,
                         "solution_index": solution_index, "difficulty": source.get("difficulty"),
                         "distractors": dict(Counter(pool[text] for text in chosen))},
        })
    if not records:
        return finish("no_distractor")
    stats["case.emitted"] += len(records)
    return records, stats + Counter({"problem.emitted": 1})


def convert_apps(rows, *, split: str, evaluation_path: Path = EVALUATION,
                 config: AppsConfig = AppsConfig(), workers: int = 1):
    """Returns (records, stats Counter). Problems run in a thread pool; output order follows source order."""
    if split != "train":
        raise ValueError("APPS conversion requires the official train split")
    rows = list(rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda item: _problem(item[0], item[1], config), enumerate(rows)))
    stats = Counter({"problem.total": len(rows)})
    records = []
    for problem_records, problem_stats in results:
        records.extend(problem_records)
        stats.update(problem_stats)
    isolation = Counter()
    kept = list(isolated(records, evaluation_path, exact=True, stats=isolation))
    stats.update({f"isolation.{name}": count for name, count in isolation.items()})
    return kept, Counter(dict(sorted(stats.items())))


def convert_humaneval(rows, *, split: str, evaluation_path: Path = EVALUATION):
    raise ValueError("HumanEval has no official train split; evaluation problems cannot enter training")


def load_official_apps(*, path: Path | None = None, evaluation_path: Path = EVALUATION,
                       config: AppsConfig = AppsConfig(), workers: int = 1):
    """Convert the published APPS train.jsonl (never test.jsonl): a local copy of it, or a download."""
    if path is None:
        with urllib.request.urlopen(APPS_URL, timeout=120) as response:
            lines = response.read().decode("utf-8").splitlines()
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return convert_apps(rows, split="train", evaluation_path=evaluation_path, config=config, workers=workers)
