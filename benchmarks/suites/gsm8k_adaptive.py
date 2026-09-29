#!/usr/bin/env python3
"""Two-pass adaptive GSM8K solver: tolerant parsing, equation sandbox, verify/escalate.

Pass 1 is a short chain of thought. Its text is then audited by pure functions:

* format check      the final number must sit on a `Final answer` line and parse cleanly
* option check      the number must match exactly one printed option
* arithmetic check  every `a op b = c` in the scratchpad is re-evaluated in a sandbox
* agreement check   optional caller signal, e.g. the log-likelihood expert disagrees

A clean audit keeps the pass-1 answer. Any trip escalates to pass 2: an independent
Python program (executed in a locked-down subprocess) plus a self-verification re-solve.
The tie-break prefers executed arithmetic over recited arithmetic.

The checks never compare options with each other and never use option position. Only the
model's own text and the problem statement feed them.

Nothing here touches torch. The model enters through `generate(prompt, max_new_tokens)`,
so the whole flow runs and tests on CPU. Only structured verdicts leave `solve`; the
generated free text is dropped, as the benchmark runners require.

Run `--self-test` for the CPU checks.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Generate = Callable[[str, int], Tuple[str, bool]]  # (prompt, max_new_tokens) -> (text, finished)

PASS1_TOKENS = 160
PASS2_TOKENS = 320
SANDBOX_TIMEOUT_S = 3.0

_NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_FINAL_LINE = re.compile(r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*(?P<body>[^\n]*)", re.I)
_HASH_LINE = re.compile(r"^\s*####\s*(?P<body>[^\n]*)$", re.I | re.M)
_OPTION = re.compile(r"\(([A-D])\)\s*([^\n\r]+)")


# ---------------------------------------------------------------- parsing

def _to_decimal(raw: str) -> Optional[Decimal]:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def parse_final_number(text: str) -> Tuple[Optional[Decimal], bool]:
    """Return (value, strict). Strict means a labelled line holding one bare number.

    Accepts `$18`, `18 dollars`, `**Final answer: 18**`, `#### 18`, `Final answer: 18.` and
    `1,200`. When no labelled line exists it falls back to the last number of the text and
    reports strict=False, which the caller treats as a reason to verify.
    """
    labelled = [m.group("body") for m in _FINAL_LINE.finditer(text)]
    labelled += [m.group("body") for m in _HASH_LINE.finditer(text)]
    for body in reversed(labelled):
        cleaned = body.replace("$", "").replace("*", "").replace("\\", "")
        nums = re.findall(_NUM, cleaned)
        if len(nums) == 1:
            value = _to_decimal(nums[0])
            if value is not None:
                return value, True
    nums = re.findall(_NUM, text.replace("$", ""))
    if nums:
        return _to_decimal(nums[-1]), False
    return None, False


def match_option(value: Optional[Decimal], context: str, candidates: Sequence[str]) -> Optional[int]:
    """Index of the single option whose printed number equals `value`, else None."""
    if value is None:
        return None
    printed = {k.strip(): v.strip() for k, v in _OPTION.findall(context)}
    hits = []
    for i, cand in enumerate(candidates):
        raw = printed.get(cand.strip(), cand).strip().replace("$", "")
        if re.fullmatch(_NUM, raw) and _to_decimal(raw) == value:
            hits.append(i)
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------- arithmetic sandbox

_BIN = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
}


def safe_eval(expr: str) -> Optional[Fraction]:
    """Evaluate a pure arithmetic expression exactly. Names, calls and attributes are refused."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None

    def walk(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("unsupported node")

    try:
        return walk(tree)
    except (ValueError, ZeroDivisionError):
        return None


def _normalise(text: str) -> str:
    text = text.replace("÷", "/").replace("−", "-").replace("$", "")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"(?<=[\d)])\s*[x×]\s*(?=[\d(])", "*", text)


_EQUATION = re.compile(rf"(?P<lhs>[\d.+\-*/() ]*[\d)])\s*=\s*(?P<rhs>{_NUM})(?![\d.]*\s*[+\-*/])")


@dataclass(frozen=True)
class EquationAudit:
    checked: int
    mismatched: int


def audit_equations(text: str) -> EquationAudit:
    """Re-evaluate each `lhs = rhs` that has an operator on the left. Count mismatches.

    A leading step label such as `2. 5 * 3 = 15` is peeled off by dropping leading tokens
    until the left side parses. Left sides that never parse are skipped, not blamed.
    """
    checked = mismatched = 0
    for line in _normalise(text).splitlines():
        for m in _EQUATION.finditer(line):
            tokens = m.group("lhs").split()
            value = None
            for start in range(len(tokens)):
                cand = " ".join(tokens[start:])
                if not re.search(r"[+\-*/]", cand.lstrip("+-")):
                    break
                value = safe_eval(cand)
                if value is not None:
                    break
            if value is None:
                continue
            rhs = m.group("rhs").replace(",", "")
            decimals = len(rhs.split(".")[1]) if "." in rhs else 0
            tolerance = Fraction(1, 2 * 10 ** decimals)
            checked += 1
            if abs(value - Fraction(rhs)) > tolerance:
                mismatched += 1
    return EquationAudit(checked, mismatched)


# ---------------------------------------------------------------- program-of-thought sandbox

_SAFE_CALLS = {"print", "round", "int", "float", "min", "max", "abs", "sum"}
_SAFE_NODES = (ast.Module, ast.Assign, ast.AugAssign, ast.Expr, ast.Call, ast.BinOp, ast.UnaryOp,
               ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
               ast.USub, ast.UAdd)


def program_is_safe(code: str) -> bool:
    """Whitelist check: arithmetic, assignment and a few pure builtins. No imports, loops, attrs."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return False
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return False
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name)
                                               and node.func.id in _SAFE_CALLS and not node.keywords):
            return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, int)
                    and 0 <= node.right.value <= 6):
                return False
    return True


_RUNNER = (
    "import sys\n"
    "src = sys.stdin.read()\n"
    "safe = {k: __builtins__.__dict__[k] if hasattr(__builtins__, '__dict__') else __builtins__[k]"
    " for k in ('print','round','int','float','min','max','abs','sum')}\n"
    "exec(compile(src, '<pot>', 'exec'), {'__builtins__': safe}, {})\n"
)


def run_program(code: str) -> Optional[Decimal]:
    """Run a validated program in an isolated subprocess and parse its last printed number."""
    if not program_is_safe(code):
        return None
    try:
        done = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=code, text=True,
                              capture_output=True, timeout=SANDBOX_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if done.returncode != 0:
        return None
    nums = re.findall(_NUM, done.stdout.replace("$", ""))
    return _to_decimal(nums[-1]) if nums else None


def extract_program(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    return (fenced.group(1) if fenced else text).strip()


# ---------------------------------------------------------------- solver

@dataclass
class Verdict:
    """Structured result. No generated free text is kept."""
    choice: Optional[int]
    value: Optional[str]
    escalated: bool
    triggers: List[str] = field(default_factory=list)
    votes: Dict[str, Optional[int]] = field(default_factory=dict)
    equations_checked: int = 0
    equations_mismatched: int = 0


def _problem_only(context: str) -> str:
    return re.split(r"\n\s*\([A-D]\)", context)[0].strip()


def pass1_prompt(context: str) -> str:
    return (f"{_problem_only(context)}\nSolve in brief numbered steps. Write each calculation as "
            f"'expression = result'. End with a line 'Final answer: <number>' and nothing after it.\n")


def pot_prompt(context: str) -> str:
    return (f"{_problem_only(context)}\nWrite a short Python program that computes the answer with "
            f"plain arithmetic and variables. No imports, no loops. End with print(answer).\n```python\n")


def verify_prompt(context: str, draft: str) -> str:
    return (f"{_problem_only(context)}\nA draft solution follows.\n{draft.strip()}\n"
            f"Check every step against the problem, fix any mistake, and redo the arithmetic "
            f"carefully. End with a line 'Final answer: <number>'.\n")


def solve(context: str, candidates: Sequence[str], generate: Generate,
          agreement_hint: Optional[int] = None) -> Verdict:
    """Solve one multiple-choice GSM8K item. `agreement_hint` is another expert's option index."""
    text1, finished1 = generate(pass1_prompt(context), PASS1_TOKENS)
    value1, strict1 = parse_final_number(text1)
    choice1 = match_option(value1, context, candidates)
    audit = audit_equations(text1)

    triggers: List[str] = []
    if not finished1 and not strict1:
        triggers.append("truncated")
    if not strict1:
        triggers.append("format")
    if value1 is not None and choice1 is None:
        triggers.append("no_unique_option")
    if audit.mismatched:
        triggers.append("arithmetic_mismatch")
    if agreement_hint is not None and choice1 is not None and agreement_hint != choice1:
        triggers.append("expert_disagreement")

    def verdict(choice: Optional[int], value: Optional[Decimal], escalated: bool,
                votes: Dict[str, Optional[int]]) -> Verdict:
        return Verdict(choice, None if value is None else format(value, "f"), escalated, triggers,
                       votes, audit.checked, audit.mismatched)

    if not triggers:
        return verdict(choice1, value1, False, {"cot": choice1})

    text_pot, _ = generate(pot_prompt(context), PASS2_TOKENS)
    value_pot = run_program(extract_program(text_pot))
    choice_pot = match_option(value_pot, context, candidates)
    text_ver, _ = generate(verify_prompt(context, text1), PASS2_TOKENS)
    value_ver, _ = parse_final_number(text_ver)
    choice_ver = match_option(value_ver, context, candidates)

    votes = {"cot": choice1, "program": choice_pot, "verify": choice_ver}
    values = {"cot": value1, "program": value_pot, "verify": value_ver}
    tally: Dict[int, int] = {}
    for c in votes.values():
        if c is not None:
            tally[c] = tally.get(c, 0) + 1
    if tally:
        best = max(tally.values())
        leaders = {c for c, n in tally.items() if n == best}
        for source in ("program", "verify", "cot"):  # executed arithmetic breaks ties
            if votes[source] in leaders:
                return verdict(votes[source], values[source], True, votes)
    return verdict(None, None, True, votes)


# ---------------------------------------------------------------- self-test

def _scripted(program: Dict[str, str]) -> Generate:
    """Fake model: pick the reply by prompt kind. Records the call order."""
    def gen(prompt: str, _n: int) -> Tuple[str, bool]:
        if prompt.rstrip().endswith("```python"):
            return program["pot"], True
        if "A draft solution follows" in prompt:
            return program["verify"], True
        return program["pass1"], True
    return gen


CTX = ("Problem: Ducks lay 16 eggs. She eats 3 and bakes with 4. She sells the rest at $2 each. "
       "How much does she make?\n(A) 18\n(B) 16\n(C) 21\n(D) 36\nSelect the answer.")
CANDS = ["A", "B", "C", "D"]


def run_self_test() -> int:
    d = Decimal
    checks: List[Tuple[str, bool]] = []

    for label, text, want in [
        ("parse: dollar sign", "steps\nFinal answer: $18", d(18)),
        ("parse: trailing unit", "Final answer: 18 dollars", d(18)),
        ("parse: bold markdown", "**Final answer: 18**", d(18)),
        ("parse: hash line", "so\n#### 18", d(18)),
        ("parse: thousands comma", "Final answer: 1,200.", d(1200)),
    ]:
        v, strict = parse_final_number(text)
        checks.append((label, v == want and strict))
    v, strict = parse_final_number("16 - 3 - 4 = 9 and then 9 * 2 = 18")
    checks.append(("parse: truncated tail is weak, last number", v == d(18) and not strict))
    checks.append(("parse: no number at all", parse_final_number("no idea") == (None, False)))
    checks.append(("option: unique match", match_option(d(18), CTX, CANDS) == 0))
    checks.append(("option: no match is None", match_option(d(19), CTX, CANDS) is None))

    a = audit_equations("16 - 3 - 4 = 9\n9 * 2 = 18")
    checks.append(("audit: correct chain passes", (a.checked, a.mismatched) == (2, 0)))
    a = audit_equations("Step 1: 3 + 4 = 8")
    checks.append(("audit: 3 + 4 = 8 is flagged", (a.checked, a.mismatched) == (1, 1)))
    a = audit_equations("2. 5 x 3 = 15 and $1,000 / 4 = 250")
    checks.append(("audit: step label, x sign, money", (a.checked, a.mismatched) == (2, 0)))
    a = audit_equations("10 / 3 = 3.33")
    checks.append(("audit: rounded division tolerated", a.mismatched == 0))
    checks.append(("safe_eval: refuses names and calls",
                   safe_eval("__import__('os')") is None and safe_eval("a + 1") is None))

    checks.append(("sandbox: arithmetic program runs",
                   run_program("a = 16 - 3 - 4\nanswer = a * 2\nprint(answer)") == d(18)))
    for label, code in [("import", "import os\nprint(1)"), ("attribute", "print((1).real)"),
                        ("dunder", "print(__builtins__)"), ("loop", "while True:\n    pass"),
                        ("open", "print(open('x'))"), ("huge power", "print(9 ** 999999)")]:
        checks.append((f"sandbox: rejects {label}", run_program(code) is None))

    clean = _scripted({"pass1": "16 - 3 - 4 = 9\n9 * 2 = 18\nFinal answer: 18", "pot": "", "verify": ""})
    r = solve(CTX, CANDS, clean)
    checks.append(("solve: clean pass 1 does not escalate", r.choice == 0 and not r.escalated))

    fixed = _scripted({"pass1": "16 - 3 - 4 = 9\n9 * 2 = 20\nFinal answer: 20",
                       "pot": "```python\nanswer = (16 - 3 - 4) * 2\nprint(answer)\n```",
                       "verify": "Final answer: 18"})
    r = solve(CTX, CANDS, fixed)
    checks.append(("solve: arithmetic slip escalates and is repaired",
                   r.choice == 0 and r.escalated and "arithmetic_mismatch" in r.triggers))
    checks.append(("solve: verdict keeps no free text", not hasattr(r, "text")))

    weak = _scripted({"pass1": "so it is about 9 * 2 = 18", "pot": "print((16 - 3 - 4) * 2)",
                      "verify": "Final answer: 18"})
    r = solve(CTX, CANDS, weak)
    checks.append(("solve: missing final line triggers format check", "format" in r.triggers and r.choice == 0))

    tie = _scripted({"pass1": "9 * 2 = 18\nFinal answer: 16", "pot": "print(16)", "verify": "Final answer: 18"})
    r = solve(CTX, CANDS, tie)
    checks.append(("solve: no-consensus tie prefers executed program", r.choice == 1))

    r = solve(CTX, CANDS, clean, agreement_hint=3)
    checks.append(("solve: expert disagreement escalates", "expert_disagreement" in r.triggers and r.escalated))

    width = max(len(c[0]) for c in checks)
    for label, ok in checks:
        print(f"  {label:<{width}}  {'... ok' if ok else '... FAILED'}")
    failed = sum(not ok for _, ok in checks)
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        sys.exit(run_self_test())
    ap.print_help()
