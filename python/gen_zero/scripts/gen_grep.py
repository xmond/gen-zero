#!/usr/bin/env python3
"""Zero-Dependency Semantic Grep CLI (gen-grep).

Uses pure Python standard library (urllib, json, sys, argparse) with 0 external dependencies.
Connects to Gen-Zero Decision Microservice (POST /v1/decisions) using batch states
and evaluates compound proposition expressions (AND, OR, NOT) with calibrated Noul probabilities.

Exit Codes:
  0: One or more matches found.
  1: No matches found.
  2: Error encountered (network, syntax, or file access).
"""

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Compact Self-Contained Boolean Logic Engine (Pure Python Stdlib)
# ---------------------------------------------------------------------------

def _clamp_prob(p: float) -> float:
    if not isinstance(p, (int, float)) or not math.isfinite(p):
        return 0.0
    return max(0.0, min(1.0, float(p)))


def p_not(p: float) -> float:
    return 1.0 - _clamp_prob(p)


def p_and(p1: float, p2: float) -> float:
    return min(_clamp_prob(p1), _clamp_prob(p2))


def p_or(p1: float, p2: float) -> float:
    return max(_clamp_prob(p1), _clamp_prob(p2))


TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(\()|'
    r'(\))|'
    r'(\bAND\b|\&\&|\&)|'
    r'(\bOR\b|\|\||\|)|'
    r'(\bNOT\b|\!|\~)|'
    r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|'
    r'([^\s\(\)\&\|\!\~\'\"]+)'
    r')',
    re.IGNORECASE
)


class BoolNode:
    def eval(self, probs: Dict[str, float]) -> float:
        raise NotImplementedError

    def variables(self) -> List[str]:
        raise NotImplementedError


class LitNode(BoolNode):
    def __init__(self, val: str):
        self.val = val.strip()

    def eval(self, probs: Dict[str, float]) -> float:
        if self.val in probs:
            return _clamp_prob(probs[self.val])
        # Case-insensitive
        v_low = self.val.lower()
        for k, v in probs.items():
            if k.lower() == v_low:
                return _clamp_prob(v)
        # Strip quotes
        s = self.val.strip("'\"")
        if s in probs:
            return _clamp_prob(probs[s])
        for k, v in probs.items():
            if k.strip("'\"").lower() == s.lower():
                return _clamp_prob(v)
        return 0.0

    def variables(self) -> List[str]:
        return [self.val.strip("'\"")]

    def __repr__(self) -> str:
        return f"Lit({self.val!r})"


class NotNode(BoolNode):
    def __init__(self, child: BoolNode):
        self.child = child

    def eval(self, probs: Dict[str, float]) -> float:
        return p_not(self.child.eval(probs))

    def variables(self) -> List[str]:
        return self.child.variables()

    def __repr__(self) -> str:
        return f"NOT({self.child})"


class AndNode(BoolNode):
    def __init__(self, children: List[BoolNode]):
        self.children = children

    def eval(self, probs: Dict[str, float]) -> float:
        if not self.children:
            return 1.0
        acc = self.children[0].eval(probs)
        for c in self.children[1:]:
            acc = p_and(acc, c.eval(probs))
        return acc

    def variables(self) -> List[str]:
        v: List[str] = []
        for c in self.children:
            v.extend(c.variables())
        return list(dict.fromkeys(v))

    def __repr__(self) -> str:
        return f"AND({', '.join(str(c) for c in self.children)})"


class OrNode(BoolNode):
    def __init__(self, children: List[BoolNode]):
        self.children = children

    def eval(self, probs: Dict[str, float]) -> float:
        if not self.children:
            return 0.0
        acc = self.children[0].eval(probs)
        for c in self.children[1:]:
            acc = p_or(acc, c.eval(probs))
        return acc

    def variables(self) -> List[str]:
        v: List[str] = []
        for c in self.children:
            v.extend(c.variables())
        return list(dict.fromkeys(v))

    def __repr__(self) -> str:
        return f"OR({', '.join(str(c) for c in self.children)})"


def parse_expression(expr: str) -> BoolNode:
    tokens: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = TOKEN_RE.match(expr, pos)
        if not m:
            pos += 1
            continue
        lp, rp, op_and, op_or, op_not, quoted, ident = m.groups()
        if lp:
            tokens.append(("LPAREN", "("))
        elif rp:
            tokens.append(("RPAREN", ")"))
        elif op_and:
            tokens.append(("AND", "AND"))
        elif op_or:
            tokens.append(("OR", "OR"))
        elif op_not:
            tokens.append(("NOT", "NOT"))
        elif quoted:
            unescaped = quoted[1:-1].replace('\\"', '"').replace("\\'", "'")
            tokens.append(("LITERAL", unescaped))
        elif ident:
            tokens.append(("LITERAL", ident))
        pos = m.end()

    if not tokens:
        raise ValueError(f"Empty expression: {expr!r}")

    idx = 0

    def cur() -> Optional[Tuple[str, str]]:
        nonlocal idx
        return tokens[idx] if idx < len(tokens) else None

    def consume(expected: Optional[str] = None) -> Tuple[str, str]:
        nonlocal idx
        tok = cur()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected and tok[0] != expected:
            raise ValueError(f"Expected {expected}, got {tok[0]}")
        idx += 1
        return tok

    def p_or_terms() -> BoolNode:
        nodes = [p_and_terms()]
        while cur() and cur()[0] == "OR":
            consume("OR")
            nodes.append(p_and_terms())
        return nodes[0] if len(nodes) == 1 else OrNode(nodes)

    def p_and_terms() -> BoolNode:
        nodes = [p_not_factor()]
        while cur() and cur()[0] == "AND":
            consume("AND")
            nodes.append(p_not_factor())
        return nodes[0] if len(nodes) == 1 else AndNode(nodes)

    def p_not_factor() -> BoolNode:
        if cur() and cur()[0] == "NOT":
            consume("NOT")
            return NotNode(p_not_factor())
        return p_primary()

    def p_primary() -> BoolNode:
        tok = cur()
        if not tok:
            raise ValueError("Unexpected end of tokens")
        if tok[0] == "LPAREN":
            consume("LPAREN")
            n = p_or_terms()
            consume("RPAREN")
            return n
        elif tok[0] == "LITERAL":
            consume("LITERAL")
            return LitNode(tok[1])
        raise ValueError(f"Unexpected token: {tok}")

    tree = p_or_terms()
    if cur() is not None:
        raise ValueError(f"Trailing token: {cur()}")
    return tree


# ---------------------------------------------------------------------------
# API Client & Pattern Bundling
# ---------------------------------------------------------------------------

def _load_token(explicit_token: Optional[str] = None) -> str:
    if explicit_token:
        return explicit_token.strip()
    return os.environ.get("GENZERO_API_KEY", "").strip()


def query_batch_decisions(
    endpoint: str,
    token: str,
    states: List[str],
    patterns: List[str],
    model: str = "typesafe/zero-1.13",
    timeout: float = 15.0
) -> Tuple[List[Dict[str, Optional[float]]], List[Tuple[int, str, str]]]:
    """Sends batch states request to Gen-Zero Decision server.

    Returns (per-line pattern -> probability, unresolved). ``unresolved`` lists
    (row_index, pattern, reason) for every (line, pattern) where the kernel abstained
    or otherwise returned no 'noul' value: the caller must surface these explicitly,
    never treat the missing value as probability 0.0 (a fabricated, confident non-match).
    """
    if not states or not patterns:
        return [{} for _ in states], []

    # Map pattern -> question_id
    questions = {}
    pattern_to_qid = {}
    for i, pat in enumerate(patterns):
        qid = f"pat_{i}"
        pattern_to_qid[pat] = qid
        questions[qid] = {
            "type": "noul",
            "instructions": f"Does the log/text state indicate: '{pat}'?",
            "criteria": {
                "true": f"Log line or text conveys or indicates '{pat}'",
                "false": f"Log line does not indicate '{pat}'"
            }
        }

    url = endpoint.rstrip("/")
    if not url.endswith("/v1/decisions") and not url.endswith("/decisions"):
        url = f"{url}/v1/decisions"

    payload = {
        "model": model,
        "state": states[0] if states else "",
        "states": states,
        "questions": questions
    }

    req_data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "gen-grep/1.0 (pure-stdlib)"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {err_msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to {url}: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Request error: {e}") from e

    results_list = data.get("results")
    if results_list is None:
        raise RuntimeError(f"Malformed server response, missing 'results': {data}")
    if len(results_list) != len(states):
        raise RuntimeError(
            f"Server returned {len(results_list)} results but expected {len(states)} rows for states"
        )

    parsed_probs: List[Dict[str, Optional[float]]] = []
    unresolved: List[Tuple[int, str, str]] = []
    for row_idx, item in enumerate(results_list):
        ans = item.get("answers", {})
        row: Dict[str, Optional[float]] = {}
        for pat, qid in pattern_to_qid.items():
            q_res = ans.get(qid, {})
            status = q_res.get("status")
            if status == "ABSTAIN" or "noul" not in q_res:
                reason = q_res.get("error") or status or "kernel returned no 'noul' result"
                unresolved.append((row_idx, pat, str(reason)))
                row[pat] = None
            else:
                nval = q_res["noul"]
                if nval is None:
                    unresolved.append((row_idx, pat, "kernel returned null noul probability"))
                    row[pat] = None
                else:
                    try:
                        fnval = float(nval)
                        if not math.isfinite(fnval) or not (0.0 <= fnval <= 1.0):
                            raise ValueError(f"out of range or non-finite: {fnval}")
                        row[pat] = fnval
                    except (TypeError, ValueError) as err:
                        unresolved.append((row_idx, pat, f"invalid noul probability ({err}): {nval!r}"))
                        row[pat] = None
        parsed_probs.append(row)

    return parsed_probs, unresolved


# ---------------------------------------------------------------------------
# CLI Argument Parser & Runner
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen-grep",
        description="Gen-Zero Zero-Dependency Semantic Grep CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cat /var/log/syslog | gen-grep -e "database deadlock" -v "timeout"
  gen-grep -a "payment failed" -a "insufficient funds" --level strict server.log
  gen-grep --expr '("database error" AND NOT "timeout") OR "out of memory"' app.log
  gen-grep -r -l -e "security vulnerability" ./src/
"""
    )

    parser.add_argument("pattern", nargs="?", default=None, help="Semantic search pattern (if -e/-a/-v not used)")
    parser.add_argument("files", nargs="*", default=[], help="Input files or directories to search")

    # Semantic pattern options
    parser.add_argument("-e", "--regexp", dest="or_patterns", action="append", default=[],
                        help="Match lines semantically matching PATTERN (OR mode)")
    parser.add_argument("-a", "--and", dest="and_patterns", action="append", default=[],
                        help="Match lines semantically matching PATTERN (AND mode)")
    parser.add_argument("-v", "--invert-match", dest="not_patterns", action="append", default=[],
                        help="Exclude lines semantically matching PATTERN (AND NOT mode)")
    parser.add_argument("--expr", dest="expression", default=None,
                        help="Complex propositional Boolean logic expression (e.g. '(A AND B) OR NOT C')")

    # Standard grep behavior options
    parser.add_argument("-r", "-R", "--recursive", action="store_true", help="Recursively search directories")
    parser.add_argument("-l", "--files-with-matches", action="store_true", help="Only print filenames of matching files")
    parser.add_argument("-n", "--line-number", action="store_true", help="Prefix each output line with its 1-based line number")
    parser.add_argument("-c", "--count", action="store_true", help="Only print count of matching lines per file")
    parser.add_argument("-i", "--ignore-case", action="store_true", help="Case-insensitive pattern matching hint")
    parser.add_argument("-A", "--after-context", type=int, default=0, help="Print NUM lines of trailing context")
    parser.add_argument("-B", "--before-context", type=int, default=0, help="Print NUM lines of leading context")
    parser.add_argument("-C", "--context", type=int, default=None, help="Print NUM lines of leading and trailing context")
    parser.add_argument("--color", choices=["always", "auto", "never"], default="auto", help="Colorize output")

    # Threshold & Microservice options
    parser.add_argument("--level", choices=["loose", "balanced", "strict"], default="balanced",
                        help="Semantic confidence level: loose (P>=0.5), balanced (P>=0.7, default), strict (P>=0.85)")
    parser.add_argument("--threshold", type=float, default=None, help="Custom numeric probability threshold in [0.0, 1.0]")
    parser.add_argument("--endpoint", default=os.environ.get("GENZERO_ENDPOINT", os.environ.get("GENZERO_URL", "http://127.0.0.1:8999")),
                        help="Gen-Zero server endpoint (default: http://127.0.0.1:8999)")
    parser.add_argument("--token", default=None, help="Bearer authorization token (default: from GENZERO_API_KEY)")
    parser.add_argument("--batch-size", type=int, default=30, help="Batch size of lines per API inference round (default: 30)")
    parser.add_argument("--model", default="typesafe/zero-1.13", help="Target decision model name")
    return parser


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command line arguments and normalizes positional files when flags are present."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.or_patterns or args.and_patterns or args.not_patterns or args.expression:
        if args.pattern is not None:
            args.files.insert(0, args.pattern)
            args.pattern = None
    return args


def resolve_threshold(level: str, custom_threshold: Optional[float]) -> float:
    if custom_threshold is not None:
        return _clamp_prob(custom_threshold)
    levels = {
        "loose": 0.50,
        "balanced": 0.70,
        "strict": 0.85
    }
    return levels.get(level.lower(), 0.70)


def build_composite_expression(
    positional_pattern: Optional[str],
    or_patterns: List[str],
    and_patterns: List[str],
    not_patterns: List[str],
    explicit_expr: Optional[str]
) -> Tuple[BoolNode, List[str]]:
    """Builds and parses compound Boolean logic AST and returns (AST, list_of_patterns)."""
    if explicit_expr:
        ast = parse_expression(explicit_expr)
        return ast, ast.variables()

    all_or = list(or_patterns)
    if positional_pattern and not all_or and not and_patterns and not not_patterns:
        all_or.append(positional_pattern)
    elif positional_pattern:
        all_or.append(positional_pattern)

    if not all_or and not and_patterns and not not_patterns:
        raise ValueError("No search patterns specified. Use -e, -a, -v, or specify pattern.")

    def _escape_pattern(pat: str) -> str:
        return pat.replace('"', '\\"')

    parts: List[str] = []
    if all_or:
        clean_or = [f'"{_escape_pattern(p)}"' for p in all_or if p]
        if len(clean_or) == 1:
            parts.append(clean_or[0])
        elif len(clean_or) > 1:
            or_str = " OR ".join(clean_or)
            parts.append(f"({or_str})")

    for a in and_patterns:
        if a:
            parts.append(f'"{_escape_pattern(a)}"')

    for n in not_patterns:
        if n:
            parts.append(f'NOT "{_escape_pattern(n)}"')

    expr_str = " AND ".join(parts)
    ast = parse_expression(expr_str)
    return ast, ast.variables()


def collect_target_files(files_arg: List[str], recursive: bool) -> List[Optional[str]]:
    """Collects list of filepaths. None represents stdin."""
    if not files_arg or files_arg == ["-"]:
        return [None]

    result: List[Optional[str]] = []
    for path in files_arg:
        if path == "-":
            result.append(None)
            continue
        if os.path.isdir(path):
            if not recursive:
                sys.stderr.write(f"gen-grep: {path}: Is a directory\n")
                continue
            for root, _, files in os.walk(path):
                for f in sorted(files):
                    result.append(os.path.join(root, f))
        elif os.path.exists(path):
            result.append(path)
        else:
            sys.stderr.write(f"gen-grep: {path}: No such file or directory\n")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_cli_args(argv)

    before_ctx = args.context if args.context is not None else args.before_context
    after_ctx = args.context if args.context is not None else args.after_context
    threshold = resolve_threshold(args.level, args.threshold)
    token = _load_token(args.token)

    # Color configuration
    use_color = (
        args.color == "always" or
        (args.color == "auto" and hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
    )

    RED = "\033[1;31m" if use_color else ""
    GREEN = "\033[32m" if use_color else ""
    CYAN = "\033[36m" if use_color else ""
    MAGENTA = "\033[1;35m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    try:
        ast, patterns = build_composite_expression(
            positional_pattern=args.pattern,
            or_patterns=args.or_patterns,
            and_patterns=args.and_patterns,
            not_patterns=args.not_patterns,
            explicit_expr=args.expression
        )
    except Exception as e:
        sys.stderr.write(f"gen-grep: error: invalid pattern or expression: {e}\n")
        return 2

    target_files = collect_target_files(args.files, args.recursive)
    if not target_files:
        return 2

    multi_files = len([f for f in target_files if f is not None]) > 1
    total_matches = 0

    for file_path in target_files:
        try:
            if file_path is None:
                lines = sys.stdin.readlines()
                display_name = "(standard input)"
            else:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                display_name = file_path
        except Exception as e:
            sys.stderr.write(f"gen-grep: {file_path}: {e}\n")
            continue

        # Strip line breaks for clean prompt states
        clean_lines = [line.rstrip("\r\n") for line in lines]
        n_lines = len(clean_lines)
        if n_lines == 0:
            if args.count:
                prefix = f"{display_name}:" if multi_files else ""
                print(f"{prefix}0")
            continue

        # Batch inference
        matched_line_indices: Set[int] = set()
        unresolved_line_indices: Set[int] = set()
        line_scores: Dict[int, float] = {}

        batch_size = max(1, min(100, args.batch_size))
        for batch_start in range(0, n_lines, batch_size):
            batch_slice = clean_lines[batch_start: batch_start + batch_size]
            try:
                batch_probs, batch_unresolved = query_batch_decisions(
                    endpoint=args.endpoint,
                    token=token,
                    states=batch_slice,
                    patterns=patterns,
                    model=args.model
                )
            except Exception as e:
                sys.stderr.write(f"gen-grep: error: {e}\n")
                return 2

            for row_idx, pat, reason in batch_unresolved:
                ln_num = batch_start + row_idx + 1
                sys.stderr.write(
                    f"gen-grep: warning: {display_name}:{ln_num}: pattern {pat!r} UNRESOLVED "
                    f"({reason}); treating as a match instead of silently assuming no match\n"
                )

            for offset, row_probs in enumerate(batch_probs):
                line_idx = batch_start + offset
                if any(p is None for p in row_probs.values()):
                    # Fail-closed for a search tool: an unresolved pattern must never be
                    # silently scored as probability 0.0 (a confident non-match). Surface it
                    # as a match candidate instead, so a human reviews it rather than the
                    # kernel's refusal being invisible.
                    unresolved_line_indices.add(line_idx)
                    matched_line_indices.add(line_idx)
                    continue
                score = ast.eval(row_probs)
                line_scores[line_idx] = score
                if score >= threshold:
                    matched_line_indices.add(line_idx)

        num_matches = len(matched_line_indices)
        total_matches += num_matches

        if unresolved_line_indices:
            # -c and -l below never print per-line content, so without this summary the
            # UNRESOLVED lines folded into num_matches would look like ordinary matches.
            sys.stderr.write(
                f"gen-grep: warning: {display_name}: {len(unresolved_line_indices)} line(s) "
                f"UNRESOLVED (kernel abstained); counted as matches, see warnings above\n"
            )

        # Mode -l: files with matches
        if args.files_with_matches:
            if num_matches > 0:
                if use_color:
                    print(f"{MAGENTA}{display_name}{RESET}")
                else:
                    print(display_name)
            continue

        # Mode -c: count
        if args.count:
            prefix = f"{CYAN}{display_name}{RESET}:" if (multi_files and use_color) else (f"{display_name}:" if multi_files else "")
            print(f"{prefix}{num_matches}")
            continue

        if num_matches == 0:
            continue

        # Print matches with context (-A, -B, -C)
        printed_indices: Set[int] = set()
        sorted_matches = sorted(matched_line_indices)
        last_printed = -1

        for m_idx in sorted_matches:
            start_ctx = max(0, m_idx - before_ctx)
            end_ctx = min(n_lines - 1, m_idx + after_ctx)

            # Print context separator '--' if gap exists
            if last_printed >= 0 and start_ctx > last_printed + 1 and (before_ctx > 0 or after_ctx > 0):
                print("--")

            for c_idx in range(start_ctx, end_ctx + 1):
                if c_idx in printed_indices:
                    continue
                printed_indices.add(c_idx)
                last_printed = c_idx

                is_match = (c_idx in matched_line_indices)
                sep = ":" if is_match else "-"

                parts_out = []
                if multi_files:
                    fn_str = f"{MAGENTA}{display_name}{RESET}" if use_color else display_name
                    parts_out.append(f"{fn_str}{sep}")
                if args.line_number:
                    ln_num = c_idx + 1
                    ln_str = f"{GREEN}{ln_num}{RESET}" if use_color else str(ln_num)
                    parts_out.append(f"{ln_str}{sep}")

                line_content = clean_lines[c_idx]
                if is_match and c_idx in unresolved_line_indices:
                    # Must render regardless of --color: piped/non-tty output (the normal
                    # grep use case) would otherwise show this exactly like a real match,
                    # hiding the fact that the kernel never actually scored this line.
                    tag = f"{RED}[UNRESOLVED]{RESET} " if use_color else "[UNRESOLVED] "
                    line_disp = f"{tag}{RED}{line_content}{RESET}" if use_color else f"{tag}{line_content}"
                elif is_match and use_color:
                    score_tag = f"{RED}[P={line_scores.get(c_idx, 1.0):.2f}]{RESET} "
                    line_disp = f"{score_tag}{RED}{line_content}{RESET}"
                else:
                    line_disp = line_content

                prefix_out = "".join(parts_out)
                print(f"{prefix_out}{line_disp}")

    return 0 if total_matches > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
