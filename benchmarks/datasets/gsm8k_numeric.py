#!/usr/bin/env python3
"""Exact numeric extraction and comparison for GSM8K answers (protocol §3.1).

Everything is a `fractions.Fraction`. Nothing here uses floats, so "18" == "18.0" ==
"36/2" == "$18.00" and 0.1 + 0.2 style drift cannot occur. Two answers match only when
they are the same rational number. No tolerance, no "close enough".

Accepted forms: integers, decimals, thousands separators, a leading `$` or sign, trailing
period, plain fractions `a/b`, mixed numbers `1 1/2`, LaTeX `\\frac{a}{b}` and `\\boxed{..}`.
Refused (returns None): percent signs, units, ranges, anything ambiguous. A refusal is a
miss, never a guess.

Not wired into `run_remote_eval_v6.py` (that file is self-contained by design).
Run `python3 gsm8k_numeric.py --self-test`.
"""

from __future__ import annotations

import re
import sys
from fractions import Fraction
from typing import List, Optional

_INT = r"\d{1,3}(?:,\d{3})+|\d+"
_DEC = rf"(?:{_INT})(?:\.\d+)?|\.\d+"
_SIGNED = rf"[+-]?(?:{_DEC})"
_MIXED = re.compile(rf"^([+-]?)(\d+)\s+(\d+)\s*/\s*(\d+)$")
_FRAC = re.compile(rf"^({_SIGNED})\s*/\s*({_DEC})$")
_LATEX_FRAC = re.compile(rf"^([+-]?)\\[dt]?frac\{{\s*({_DEC})\s*\}}\{{\s*({_DEC})\s*\}}$")
_PLAIN = re.compile(rf"^({_SIGNED})$")
_GOLD_LINE = re.compile(r"^[ \t]*####[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL = re.compile(r"final\s+answer\s*(?:is)?\s*[:=]?\s*([^\n]*)", re.IGNORECASE)


def _dec(raw: str) -> Fraction:
    return Fraction(raw.replace(",", ""))


def parse_number(text: Optional[str]) -> Optional[Fraction]:
    """Parse one number written as integer, decimal or fraction. None when it is not exactly one."""
    if text is None:
        return None
    s = text.strip().replace("\u2212", "-").replace("\u00a0", " ")
    s = s.strip("*").strip()
    s = s.replace("$", "").strip()
    s = s.rstrip(".").strip()          # "18." at the end of a sentence
    if not s or "%" in s:
        return None
    m = _LATEX_FRAC.match(s)
    if m:
        den = _dec(m.group(3))
        return None if den == 0 else (-1 if m.group(1) == "-" else 1) * _dec(m.group(2)) / den
    m = _MIXED.match(s)
    if m:
        den = int(m.group(4))
        if den == 0:
            return None
        val = int(m.group(2)) + Fraction(int(m.group(3)), den)
        return -val if m.group(1) == "-" else val
    m = _FRAC.match(s)
    if m:
        den = _dec(m.group(2))
        return None if den == 0 else _dec(m.group(1)) / den
    m = _PLAIN.match(s)
    if m:
        return _dec(m.group(1))
    return None


def normalize(text: Optional[str]) -> Optional[str]:
    """Canonical string of a parsed number: '18', '-7', '3/2'. None when unparsable."""
    value = parse_number(text)
    if value is None:
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def numeric_equal(a: Optional[str], b: Optional[str]) -> bool:
    """True only when both parse and the rationals are identical."""
    x, y = parse_number(a), parse_number(b)
    return x is not None and y is not None and x == y


def gold_from_raw_answer(raw_answer: str) -> Fraction:
    """Gold value from a GSM8K `answer` field: the number after the last `####`."""
    hits = _GOLD_LINE.findall(raw_answer)
    if not hits:
        raise ValueError("no '#### <n>' line in raw answer")
    value = parse_number(hits[-1])
    if value is None:
        raise ValueError(f"unparsable gold {hits[-1]!r}")
    return value


def extract_final_answer(text: str) -> Optional[Fraction]:
    """Pull the model's final number out of free text.

    Priority: last `#### n`, last `\\boxed{n}`, last `Final answer: n`. Each labelled body
    must itself be exactly one number. Unlabelled text never scores: a stray number in a
    scratchpad returns None.
    """
    candidates: List[str] = []
    candidates += _GOLD_LINE.findall(text)
    candidates += _BOXED.findall(text)
    candidates += [m.strip() for m in _FINAL.findall(text)]
    for body in reversed(candidates):
        value = parse_number(body)
        if value is not None:
            return value
    return None


def _self_test() -> int:
    F = Fraction
    cases = [
        ("18", F(18)), ("$18", F(18)), ("18.", F(18)), ("18.0", F(18)), ("18.00", F(18)),
        ("1,200", F(1200)), ("1,450,000", F(1450000)), ("-2.5", F(-5, 2)), ("+7", F(7)),
        ("3/4", F(3, 4)), ("36/2", F(18)), ("1 1/2", F(3, 2)), ("-1 1/2", F(-3, 2)),
        ("\\frac{3}{4}", F(3, 4)), ("-\\frac{1}{2}", F(-1, 2)), (".5", F(1, 2)),
        ("**18**", F(18)), ("\u22122", F(-2)), ("0.1", F(1, 10)),
    ]
    refused = ["", "abc", "18 apples", "12%", "1/0", "3-4", "1.2.3", "18,00", "\\frac{1}{0}", None]
    checks = [(f"parse {t!r}", parse_number(t) == v) for t, v in cases]
    checks += [(f"refuse {t!r}", parse_number(t) is None) for t in refused]
    checks += [
        ("0.1 + 0.2 style: '0.3' equals '3/10'", numeric_equal("0.3", "3/10")),
        ("near miss is not equal: 18 vs 18.01", not numeric_equal("18", "18.01")),
        ("unparsable is never equal", not numeric_equal("abc", "abc")),
        ("normalize 36/2", normalize("36/2") == "18"),
        ("normalize 6/4", normalize("6/4") == "3/2"),
        ("gold line", gold_from_raw_answer("so 2*9 = <<2*9=18>>18\n#### 1,200") == F(1200)),
        ("extract #### line", extract_final_answer("work 5\n#### 18") == F(18)),
        ("extract boxed", extract_final_answer("so \\boxed{3/4}") == F(3, 4)),
        ("extract Final answer", extract_final_answer("...\nFinal answer: $1,200.") == F(1200)),
        ("unlabelled last number is not scored", extract_final_answer("2 * 9 = 18") is None),
        ("labelled body with two numbers refused", extract_final_answer("Final answer: 18 or 19") is None),
    ]
    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"{len(checks) - len(bad)}/{len(checks)} PASS")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
