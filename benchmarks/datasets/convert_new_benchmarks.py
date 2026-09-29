#!/usr/bin/env python3
"""Standardized fixture generator for new public benchmark datasets:
1. HANS (30,000 heuristic adversarial NLI examples)
2. BIG-Bench Hard (BBH 27 algorithmic reasoning tasks)
3. Bespoke Labs causal decision & policy tasks (the eval.jsonl we hold has 324 rows: 162 base + 162 counterfactual)

Data Governance:
- Zero private filesystem path leaks (no hardcoded absolute paths in output).
- Pure standard JSONL format:
  {"id": str, "task": str, "context": str, "candidates": List[str], "ground_truth": str, "metadata": dict}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = Path(
    os.environ.get(
        "GEN_ZERO_BENCHMARK_RAW_DIR",
        BENCHMARKS_DIR / ".cache" / "raw_new_benchmarks",
    )
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "data"


HANS_CANDIDATES = ["entailment", "non-entailment"]
HANS_HEURISTICS = ("lexical_overlap", "subsequence", "constituent")
_HANS_REQUIRED = ("gold_label", "sentence1", "sentence2", "pairID", "heuristic", "subcase", "template")


def _read_hans_rows(raw_path: Path) -> List[Dict[str, str]]:
    """Parse the HANS TSV. Any malformed row raises: a silently dropped row would skew strata."""
    with open(raw_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in _HANS_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"hans TSV header lacks columns {missing}")
        rows: List[Dict[str, str]] = []
        seen: set = set()
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"hans line {line_no}: more fields than the header")
            vals = {c: (row.get(c) or "").strip() for c in _HANS_REQUIRED}
            empty = [c for c, v in vals.items() if not v]
            if empty:
                raise ValueError(f"hans line {line_no}: empty fields {empty}")
            if vals["gold_label"] not in HANS_CANDIDATES:
                raise ValueError(f"hans line {line_no}: gold_label {vals['gold_label']!r} not in {HANS_CANDIDATES}")
            if vals["heuristic"] not in HANS_HEURISTICS:
                raise ValueError(f"hans line {line_no}: heuristic {vals['heuristic']!r} not in {HANS_HEURISTICS}")
            if vals["pairID"] in seen:
                raise ValueError(f"hans line {line_no}: duplicate pairID {vals['pairID']!r}")
            seen.add(vals["pairID"])
            rows.append(vals)
    if not rows:
        raise ValueError("hans TSV has no data rows")
    return rows


def _stratified_hans_sample(rows: List[Dict[str, str]], limit: int, seed: int) -> List[Dict[str, str]]:
    """Equal share per (heuristic, gold_label) stratum, so every heuristic gets limit/3 rows
    and each heuristic stays label-balanced. Output keeps source-file order."""
    strata: Dict[tuple, List[int]] = {(h, g): [] for h in HANS_HEURISTICS for g in HANS_CANDIDATES}
    for i, r in enumerate(rows):
        strata[(r["heuristic"], r["gold_label"])].append(i)
    if limit % len(strata):
        raise ValueError(f"hans stratified limit {limit} must be a multiple of {len(strata)} "
                         f"(3 heuristics x 2 labels); refusing to round silently")
    per = limit // len(strata)
    rng = random.Random(seed)
    picked: List[int] = []
    for key, idx in strata.items():
        if len(idx) < per:
            raise ValueError(f"hans stratum {key} has {len(idx)} rows, need {per}")
        picked.extend(rng.sample(idx, per))
    return [rows[i] for i in sorted(picked)]


def convert_hans(
    raw_path: Path,
    out_path: Path,
    limit: Optional[int] = None,
    *,
    stratify: bool = True,
    seed: int = 0,
) -> int:
    """Convert the HANS TSV to six-key JSONL.

    limit=None converts all rows. With a limit, stratify=True draws an equal, seeded share per
    heuristic and label; stratify=False takes the first `limit` rows in file order (the file is
    grouped by heuristic, so that prefix is one heuristic only).
    """
    rows = _read_hans_rows(raw_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"hans limit must be positive or None (full), got {limit}")
        if limit > len(rows):
            raise ValueError(f"hans limit {limit} exceeds the {len(rows)} source rows")
        rows = _stratified_hans_sample(rows, limit, seed) if stratify else rows[:limit]

    with open(out_path, "w", encoding="utf-8") as f_out:
        for r in rows:
            context = (
                f"Premise: {r['sentence1']}\n"
                f"Hypothesis: {r['sentence2']}\n"
                f"Determine whether the premise entails the hypothesis."
            )
            record = {
                "id": f"hans-{r['pairID']}",
                "task": "hans",
                "context": context,
                "candidates": list(HANS_CANDIDATES),
                "ground_truth": r["gold_label"],
                "metadata": {
                    "heuristic": r["heuristic"],
                    "subcase": r["subcase"],
                    "template": r["template"],
                    "source": "hans",
                },
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(rows)


_BBH_OPTION_RE = re.compile(r"\(([A-Z])\)\s*([^\n\r(]+)")



_BBH_OPTIONS_MARKER = "\nOptions:\n"
_BBH_LETTERED_LINE = re.compile(r"^\(([A-Za-z0-9])\)\s*(.*)$")
_BBH_DASHED_LINE = re.compile(r"^-\s*(.+?)\s*$")

# BBH tasks whose *own* upstream design has no enumerated candidate set at all
_BBH_EXACT_MATCH_TASKS = frozenset(
    {"dyck_languages", "multistep_arithmetic_two", "object_counting", "word_sorting"}
)

# All 27 BIG-Bench Hard task names (github.com/suzgunmirac/BIG-Bench-Hard).
BBH_TASK_NAMES = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)

class BBHFormatError(ValueError):
    """Raised when a BBH example's input/target cannot be safely normalized.

    Fail-closed: a parsing gap here must never be papered over with a fake
    single-item candidate list (that would trivially satisfy
    "ground_truth in candidates" while destroying the task's discriminative
    power). Callers must fix the parser or the input, not swallow this.
    """


class BBHGroundTruthMismatch(BBHFormatError):
    """Raised when target_clean is not in the derived candidate set.

    Unlike BBHFormatError's other causes (unrecognized Options: layout,
    duplicate/empty candidates -- always a parser bug), this one case is also
    triggered by known upstream BBH annotation noise: movie_recommendation and
    ruin_names split option text on commas, so a target that is itself a
    comma-containing title (e.g. "Monsters, Inc") lands outside the letter
    options its own commas fragmented into. convert_bbh_task catches only this
    subclass, logs it loudly, and skips the record -- never silently.
    """


def _split_bbh_options(inp: str) -> Optional[List[str]]:
    """Extracts the candidate label list from a BBH "Options:" block, if present.

    Returns None if there is no Options: block at all (genuinely free-form
    upstream). Raises BBHFormatError if an Options: block exists but its lines
    don't uniformly match either the lettered "(A) text" or dashed "- text"
    style BBH actually uses -- never guesses at a third format.
    """
    idx = inp.find(_BBH_OPTIONS_MARKER)
    if idx == -1:
        return None
    tail = inp[idx + len(_BBH_OPTIONS_MARKER):]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        raise BBHFormatError(f"'Options:' marker with no option lines: {inp!r}")

    lettered = [_BBH_LETTERED_LINE.match(line) for line in lines]
    if all(lettered):
        return [m.group(1).upper() for m in lettered]

    dashed = [_BBH_DASHED_LINE.match(line) for line in lines]
    if all(dashed):
        return [m.group(1) for m in dashed]

    raise BBHFormatError(
        f"Options: block lines match neither lettered nor dashed style: {lines!r}"
    )


def _normalize_bbh_example(task_name: str, idx: int, inp: str, target: str) -> tuple[List[str], str, str]:
    """Derives (candidates, ground_truth, answer_mode) for one BBH example.

    Raises BBHGroundTruthMismatch if the target isn't in the derived candidate
    set (see that class for why this one failure mode is skip-not-abort), or
    the base BBHFormatError for anything else unrecognized (always a parser
    bug, always aborts the whole file).
    """
    cands = _split_bbh_options(inp)
    if cands is not None:
        answer_mode = "choice"
        if target.startswith("(") and target.endswith(")"):
            target_clean = target.strip("()").upper()
        else:
            target_clean = target
    elif target.lower() in ("true", "false"):
        answer_mode = "choice"
        cands = ["True", "False"]
        target_clean = "True" if target.lower() == "true" else "False"
    elif target.lower() in ("yes", "no"):
        answer_mode = "choice"
        cands = ["yes", "no"]
        target_clean = target.lower()
    elif target.lower() in ("valid", "invalid"):
        answer_mode = "choice"
        cands = ["valid", "invalid"]
        target_clean = target.lower()
    elif task_name in _BBH_EXACT_MATCH_TASKS:
        answer_mode = "exact_match"
        target_clean = target
        cands = [target]
    else:
        raise BBHFormatError(
            f"{task_name}[{idx}]: no Options: block, target {target!r} is not a "
            "known fixed binary domain, and this task is not on the free-form "
            "allow-list -- unrecognized BBH format, refusing to guess."
        )

    if target_clean not in cands:
        raise BBHGroundTruthMismatch(
            f"{task_name}[{idx}]: ground_truth {target_clean!r} not in "
            f"candidates {cands!r} (raw target={target!r})"
        )
    if answer_mode == "choice" and len(cands) < 2:
        # A "choice" record with 1 candidate is trivially "correct" no matter
        # what a downstream consumer picks -- the exact single-candidate cheat
        # this rewrite exists to remove. Seen in practice as truncated raw BBH
        # input (snarks[88]: input cuts off after "(A) The NB", option (B)
        # never printed). Same class of upstream noise as
        # BBHGroundTruthMismatch: skip and log, don't abort the whole file.
        raise BBHGroundTruthMismatch(
            f"{task_name}[{idx}]: 'choice' mode but only {len(cands)} candidate {cands!r} "
            f"(raw input truncated?): {inp!r}"
        )
    if len(set(cands)) != len(cands):
        raise BBHFormatError(f"{task_name}[{idx}]: duplicate candidates {cands!r}")
    if any(not c for c in cands):
        raise BBHFormatError(f"{task_name}[{idx}]: empty candidate in {cands!r}")
    return cands, target_clean, answer_mode


# Above this fraction of a task's examples failing ground-truth normalization
# stops looking like known upstream annotation noise (BBH's comma-in-option
# splitting bug, ~1-2 rows per 250) and starts looking like a parser bug; abort
# rather than keep silently thinning the dataset.
_BBH_MAX_SKIP_RATE = 0.02


def convert_bbh_task(
    task_file: Path,
    out_path: Path,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """Converts a single BBH task JSON file into standardized JSONL format.

    Every written record satisfies the strict 6-key schema and ground_truth in
    candidates. metadata.answer_mode records how candidates were derived:
      - "choice": candidates are the task's real enumerated option set
        (BBH's own lettered or dashed Options: block, or a fixed binary
        True/False | yes/no | valid/invalid domain for tasks that never
        print an Options: block but only ever take one of those two values).
      - "exact_match": the task has no enumerated candidate set in BBH's own
        design (dyck_languages, multistep_arithmetic_two, object_counting,
        word_sorting); candidates is the single-element [ground_truth], never
        a fabricated multi-way choice, so a downstream consumer must score
        these by exact string match, not by candidate elimination.

    Returns (written, skipped). skipped counts only BBHGroundTruthMismatch
    (known upstream annotation noise, logged loudly per record below); any
    other malformed input raises and aborts the whole file.
    """
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_name = task_file.stem
    examples = data.get("examples", [])
    count = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, ex in enumerate(examples):
            inp = ex.get("input", "").strip()
            target = str(ex.get("target", "")).strip()
            if not inp or not target:
                continue

            try:
                cands, target_clean, answer_mode = _normalize_bbh_example(task_name, idx, inp, target)
            except BBHGroundTruthMismatch as exc:
                skipped += 1
                print(f"[BBH] WARNING skipping malformed upstream record: {exc}", file=sys.stderr)
                continue

            context = f"Task: {task_name.replace('_', ' ')}\nProblem: {inp}"
            record = {
                "id": f"bbh-{task_name}-{idx:04d}",
                "task": f"bbh_{task_name}",
                "context": context,
                "candidates": cands,
                "ground_truth": target_clean,
                "metadata": {
                    "source": "bbh",
                    "task_name": task_name,
                    "answer_mode": answer_mode,
                },
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break

    if examples and skipped / len(examples) > _BBH_MAX_SKIP_RATE:
        raise BBHFormatError(
            f"{task_name}: skip rate {skipped}/{len(examples)} exceeds "
            f"{_BBH_MAX_SKIP_RATE:.0%} -- looks like a parser bug, not upstream noise"
        )
    return count, skipped



_BESPOKE_PAIR_SUFFIXES = ("-base", "-counterfactual")
_SCORE_LABEL_RE = re.compile(r"^(\d+)\s*[—:-]")

def _bespoke_label(key: str) -> str:
    # Upper-case the first letter only; str.capitalize() would lower-case codes like "R-C".
    label = key.replace("_", " ").strip()
    return label[:1].upper() + label[1:]


def _render_bespoke_value(value: Any, indent: str) -> List[str]:
    if isinstance(value, dict):
        lines: List[str] = []
        for k, v in value.items():
            sub = _render_bespoke_value(v, indent + "  ")
            if len(sub) == 1 and not isinstance(v, (dict, list)):
                lines.append(f"{indent}{_bespoke_label(str(k))}: {sub[0].strip()}")
            else:
                lines.append(f"{indent}{_bespoke_label(str(k))}:")
                lines.extend(sub)
        return lines
    if isinstance(value, list):
        lines = []
        for i, v in enumerate(value, start=1):
            if isinstance(v, dict) and set(v) == {"speaker", "text"}:
                lines.append(f"{indent}{v['speaker']}: {v['text']}")
            elif isinstance(v, (dict, list)):
                lines.append(f"{indent}{i}.")
                lines.extend(_render_bespoke_value(v, indent + "  "))
            else:
                lines.append(f"{indent}{i}. {v}")
        return lines
    return [f"{indent}{value}"]


def _render_bespoke_state(state: Any, questions: Dict[str, Any], row_id: str) -> str:
    """Render every state field. The old whitelist (context/evidence/field_note/request) silently
    dropped timeline, policy, observations, facts, findings and similar decision evidence."""
    if isinstance(state, dict):
        state = dict(state)
        # Some rows embed a verbatim copy of input.questions; it is rendered from the outer
        # copy below, so drop it here only after proving it is identical.
        if "questions" in state:
            if state.pop("questions") != questions:
                raise ValueError(f"bespoke {row_id}: state.questions differs from input.questions")
            if set(state) == {"state"}:
                state = state["state"]
    if isinstance(state, str):
        body = state.strip()
    elif isinstance(state, list):
        body = "\n".join(_render_bespoke_value(state, "")).strip()
    elif isinstance(state, dict):
        parts = []
        for key, value in state.items():
            if isinstance(value, (dict, list)):
                parts.append(f"{_bespoke_label(key)}:\n" + "\n".join(_render_bespoke_value(value, "  ")))
            else:
                parts.append(f"{_bespoke_label(key)}: {str(value).strip()}")
        body = "\n".join(parts).strip()
    else:
        raise ValueError(f"bespoke {row_id}: unsupported state type {type(state).__name__}")
    if not body:
        raise ValueError(f"bespoke {row_id}: state renders to empty context")
    return body


def _normalize_bespoke_target(raw_target: Any, decision_type: str, row_id: str) -> str:
    # noul targets are JSON booleans while their criteria keys are "true"/"false";
    # score targets are ints matched against the leading level number of each criterion.
    if isinstance(raw_target, bool):
        if decision_type != "noul":
            raise ValueError(f"bespoke {row_id}: boolean target for decision type {decision_type!r}")
        return "true" if raw_target else "false"
    if isinstance(raw_target, int):
        if decision_type != "score":
            raise ValueError(f"bespoke {row_id}: integer target for decision type {decision_type!r}")
        return str(raw_target)
    if isinstance(raw_target, str):
        return raw_target.strip()
    raise ValueError(f"bespoke {row_id}: unsupported target type {type(raw_target).__name__}")


def convert_bespoke(
    raw_path: Path,
    out_path: Path,
    limit: Optional[int] = None,
) -> int:
    """Convert Bespoke Labs eval.jsonl (causal decision / policy tasks) to six-key JSONL.

    Every malformed row raises. Only input.state and input.questions reach the context;
    evidence_certificate (it holds the audited fact states, i.e. the answer) and reference
    never do.
    """
    if limit is not None and limit <= 0:
        raise ValueError(f"bespoke limit must be positive or None (full), got {limit}")
    count = 0
    seen: set = set()
    with open(raw_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            row_id = str(item.get("id") or "").strip() or f"line {line_no}"
            inp = item.get("input") or {}
            questions = inp.get("questions") or {}
            decision = questions.get("decision") or {}
            criteria = decision.get("criteria")
            instructions = str(decision.get("instructions") or "").strip()
            decision_type = str(decision.get("type") or "").strip()
            reference = item.get("reference") or {}
            if "target" not in reference or reference["target"] is None:
                raise ValueError(f"bespoke {row_id}: missing reference.target")
            target = _normalize_bespoke_target(reference["target"], decision_type, row_id)

            if isinstance(criteria, dict):
                cands = [str(k).strip() for k in criteria.keys()]
                crit_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
            elif isinstance(criteria, list) and decision_type == "score":
                # Score targets are 0-based indices into the ordered levels. Some level lists carry
                # a leading "<n> —" label and some carry a name ("Hold—weak match: ..."), so the
                # index is the only key both share. A numeric label that disagrees with its
                # position would make the index ambiguous, so it raises.
                cands = [str(i) for i in range(len(criteria))]
                lines = []
                for i, c in enumerate(criteria):
                    c_str = str(c).strip()
                    m = _SCORE_LABEL_RE.match(c_str)
                    if m and int(m.group(1)) != i:
                        raise ValueError(f"bespoke {row_id}: score level {i} is labelled {m.group(1)}")
                    lines.append(f"- {c_str}" if m else f"- {i}: {c_str}")
                crit_text = "\n".join(lines)
            elif isinstance(criteria, list):
                cands = []
                for c in criteria:
                    c_str = str(c).strip()
                    c_key = c_str.split(" — ")[0].strip().split(":")[0].strip().split(" ")[0].strip()
                    cands.append(c_key if c_key else c_str)
                crit_text = "\n".join(f"- {c}" for c in criteria)
            else:
                raise ValueError("bespoke criteria must be a mapping or list; cannot derive candidates from the answer")

            if len(cands) < 2 or len(set(cands)) != len(cands) or target not in cands:
                raise ValueError("bespoke requires distinct candidates including the target")
            if not instructions:
                raise ValueError(f"bespoke {row_id}: missing decision instructions")
            if not str(item.get("id") or "").strip():
                raise ValueError(f"bespoke line {line_no}: missing id")
            if row_id in seen:
                raise ValueError(f"bespoke line {line_no}: duplicate id {row_id!r}")
            seen.add(row_id)

            body_text = _render_bespoke_state(inp.get("state"), questions, row_id)
            context = (
                f"{body_text}\n"
                f"Instructions: {instructions}\n"
                f"Candidate Options:\n{crit_text}"
            )
            variant = str(item.get("variant") or "")
            pair_id = row_id
            for suffix in _BESPOKE_PAIR_SUFFIXES:
                if row_id.endswith(suffix):
                    pair_id = row_id[: -len(suffix)]
            record = {
                "id": f"bespoke-{row_id}",
                "task": "bespoke",
                "context": context,
                "candidates": cands,
                "ground_truth": target,
                "metadata": {
                    "source": "bespoke_labs",
                    # The raw rows carry `domain`; it is the policy domain the rubric governs.
                    # There is no decision-horizon field in the source, so none is emitted.
                    "policy_domain": str(item.get("domain") or ""),
                    "decision_type": decision_type,
                    "variant": variant,
                    "pair_id": pair_id,
                    "family": str(item.get("family") or ""),
                    "source_family": str(item.get("source_family") or ""),
                    "split": str(item.get("split") or ""),
                    "human_reviewed": bool(reference.get("human_reviewed", False)),
                },
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    if count == 0:
        raise ValueError(f"bespoke source {raw_path.name} has no rows")
    return count


def _limit_arg(value: str) -> Optional[int]:
    """'all' (or 0) means full conversion; any other value must be a positive row count."""
    if value.strip().lower() in ("all", "0"):
        return None
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"limit must be >= 0 or 'all', got {value}")
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert new benchmark datasets to standardized format")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="raw source root (default: $GEN_ZERO_BENCHMARK_RAW_DIR or benchmarks/.cache)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hans-limit", type=_limit_arg, default=_limit_arg("102"),
                        help="rows to keep, or 'all'; stratified runs need a multiple of 6")
    parser.add_argument("--hans-stratify", action=argparse.BooleanOptionalAction, default=True,
                        help="equal share per heuristic and label (default on)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bbh-limit-per-task", type=_limit_arg, default=_limit_arg("30"))
    parser.add_argument("--bespoke-limit", type=_limit_arg, default=_limit_arg("100"))
    parser.add_argument("--only", choices=("hans", "bbh", "bespoke"), action="append",
                        help="convert only these sources; a requested source that is missing is an error")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only or ("hans", "bbh", "bespoke"))
    converted: List[str] = []
    missing: List[str] = []

    # 1. HANS
    if "hans" in wanted:
        hans_raw = args.raw_dir / "hans" / "heuristics_evaluation_set.txt"
        if hans_raw.exists():
            n = convert_hans(hans_raw, args.out_dir / "hans.jsonl", limit=args.hans_limit,
                             stratify=args.hans_stratify, seed=args.seed)
            print(f"[HANS] Converted {n} samples -> hans.jsonl")
            converted.append("hans")
        else:
            missing.append("hans/heuristics_evaluation_set.txt")

    # 2. BBH
    if "bbh" in wanted:
        bbh_raw_dir = args.raw_dir / "bbh"
        bbh_files = sorted(bbh_raw_dir.glob("*.json")) if bbh_raw_dir.exists() else []
        if bbh_files:
            total_bbh = 0
            for bf in bbh_files:
                total_bbh += convert_bbh_task(bf, args.out_dir / f"bbh_{bf.stem}.jsonl", limit=args.bbh_limit_per_task)
            print(f"[BBH] Converted {total_bbh} samples across {len(bbh_files)} tasks")
            converted.append("bbh")
        else:
            missing.append("bbh/*.json")

    # 3. Bespoke Labs
    if "bespoke" in wanted:
        bespoke_raw = args.raw_dir / "bespoke" / "eval.jsonl"
        if bespoke_raw.exists():
            n = convert_bespoke(bespoke_raw, args.out_dir / "bespoke.jsonl", limit=args.bespoke_limit)
            print(f"[BESPOKE] Converted {n} samples -> bespoke.jsonl")
            converted.append("bespoke")
        else:
            missing.append("bespoke/eval.jsonl")

    for name in missing:
        # Relative names only: the raw root is a private path and must not reach logs or artifacts.
        print(f"[SKIP] missing raw source <raw-dir>/{name}", file=sys.stderr)
    if args.only and missing:
        return 2
    if not converted:
        print("[ERROR] no source converted", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
