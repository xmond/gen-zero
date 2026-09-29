#!/usr/bin/env python3
"""Verify benchmarks/data against manifest.json, and the rebuilt sets against the v6 runner.

Checks (exit 0 only if all pass):
  1  every manifest file: SHA256 and sample count match; ids unique; six-key schema and types;
     ground_truth is one of candidates; all_benchmarks.jsonl equals the concatenation
  2  GSM8K: n, no letter above 35%, gold option equals the '#### n' gold, other options differ,
     options distinct, exact rational compare through gsm8k_numeric
  3  GSM8K through v6: match_cot_answer('#### <gold>') returns the gold letter's index
  4  PAWS: n, balanced labels, overlap metadata and bucket rule, v6 paws_sentences parses
  5  PAWS minimal pairs: parents exist, swap_order round-trips, label relations hold
CPU only. Imports run_remote_eval_v6 for helpers; that module needs numpy, not torch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gsm8k_numeric import numeric_equal, parse_number  # noqa: E402
from dataset_registry import DatasetError, load_registry, registered_tasks, allowed_files  # noqa: E402

DATA = HERE.parent / "data"
V6 = HERE.parent / "suites" / "run_remote_eval_v6.py"
KEYS = {"id": str, "task": str, "context": str, "candidates": list, "ground_truth": str, "metadata": dict}

results: List[tuple] = []


def check(name: str, ok: bool) -> None:
    results.append((name, bool(ok)))
    color = "\033[32m" if ok else "\033[31m"
    print(f"{color}{'PASS' if ok else 'FAIL'}\033[0m {name}")


def read_rows(path: Path) -> List[dict]:
    # split on \n only: str.splitlines() also splits on U+2028 and would cut a row.
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x]


def load_v6():
    spec = importlib.util.spec_from_file_location("v6", V6)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    try:
        _, datasets = load_registry(DATA)
    except DatasetError as exc:
        check(f"registry: {exc}", False)
        return 1
    all_lines: List[str] = []
    total = 0
    for task, entry in manifest["tasks"].items():
        rows = datasets[task]
        raw = (DATA / entry["file"]).read_text(encoding="utf-8")
        all_lines.extend(x for x in raw.split("\n") if x)
        total += len(rows)
    for task, rows in datasets.items():
        entry = registered_tasks(manifest)[task]
        check(f"{task}: {entry['file']} sha256={entry['sha256']}, count={len(rows)}, "
              f"six-key schema, IDs, task, ground_truth: all rows", True)
    check("manifest total_samples", total == manifest["total_samples"])
    allb = (DATA / "all_benchmarks.jsonl").read_text(encoding="utf-8")
    check("all_benchmarks.jsonl = core concatenation, sha256 matches manifest",
          [x for x in allb.split("\n") if x] == all_lines
          and hashlib.sha256(allb.encode("utf-8")).hexdigest() == manifest["all_benchmarks"]["sha256"])
    for name, meta in manifest.get("aux_files", {}).items():
        raw = (DATA / name).read_text(encoding="utf-8")
        check(f"aux {name}: sha256 and count",
              hashlib.sha256(raw.encode("utf-8")).hexdigest() == meta["sha256"]
              and len([x for x in raw.split("\n") if x]) == meta["samples_count"])
    check("all JSONL files registered", True)

    v6 = load_v6()

    # ---- GSM8K
    gsm = read_rows(DATA / "gsm8k.jsonl")
    check(f"gsm8k n={len(gsm)} >= 200", len(gsm) >= 200)
    letters = Counter(r["ground_truth"] for r in gsm)
    print("  gold letters:", dict(sorted(letters.items())))
    check("gsm8k gold letters: all four used, none above 35%",
          set(letters) == {"A", "B", "C", "D"} and max(letters.values()) <= 0.35 * len(gsm))
    check("gsm8k gold is not always 'A'", letters["A"] < len(gsm))
    good = distinct = hashline = True
    for r in gsm:
        m = r["metadata"]
        opts = dict(v6.extract_options(r["context"]))
        vals = [parse_number(opts.get(L)) for L in "ABCD"]
        gold = parse_number(m["raw_answer"].split("####")[-1])
        gi = "ABCD".index(r["ground_truth"])
        good &= all(v is not None for v in vals) and vals[gi] == gold and numeric_equal(m["gold_raw"], opts[r["ground_truth"]])
        distinct &= len(set(vals)) == 4 and sum(v == gold for v in vals) == 1
        hashline &= m["raw_answer"].rstrip().split("\n")[-1].startswith("#### ") and m["gold_numeric"] == str(gold)
        good &= m["option_values"] == [opts[L] for L in "ABCD"]
    check("gsm8k: option at gold letter equals '#### n' exactly (rational compare)", good)
    check("gsm8k: four distinct option values, exactly one equals gold", distinct)
    check("gsm8k: raw_answer ends with '#### <n>' and gold_numeric matches", hashline)
    hits = 0
    for r in gsm:
        idx = v6.match_cot_answer(f"#### {r['metadata']['gold_numeric']}", r["context"], r["candidates"])
        hits += idx is not None and r["candidates"][idx] == r["ground_truth"]
    print(f"  v6 match_cot_answer recovers gold letter: {hits}/{len(gsm)}")
    check("gsm8k via v6 match_cot_answer: all rows", hits == len(gsm))
    check("gsm8k v6 is_positional_task fires (letter labels)", all(v6.is_positional_task(r["candidates"]) for r in gsm))
    b4 = sum(r["metadata"]["n_steps"] >= 4 for r in gsm)
    print(f"  4+ step problems: {b4}")
    check("gsm8k has >= 50 problems with 4+ steps", b4 >= 50)

    # ---- PAWS
    paws = read_rows(DATA / "paws.jsonl")
    check(f"paws n={len(paws)} >= 200", len(paws) >= 200)
    labels = Counter(r["ground_truth"] for r in paws)
    check("paws labels balanced", labels["paraphrase"] == labels["not_paraphrase"] and len(labels) == 2)
    parsed = 0
    for r in paws:
        s1, s2 = v6.paws_sentences(r["context"])
        parsed += bool(s1 and s2)
    print(f"  v6 paws_sentences parses: {parsed}/{len(paws)}")
    check("paws: v6 paws_sentences parses every row", parsed == len(paws))
    th = manifest["tasks"]["paws"]["build"]["overlap_thresholds"]
    ok_bucket = True
    for r in paws:
        lo = r["metadata"]["lexical_overlap"]
        want = "low" if lo["unigram_jaccard"] < th["low_lt"] else "high" if lo["unigram_jaccard"] >= th["high_ge"] else "mid"
        ok_bucket &= lo["bucket"] == want == r["metadata"]["overlap_bucket"]
    check("paws: overlap bucket matches recorded thresholds", ok_bucket)
    cells = Counter((r["metadata"]["overlap_bucket"], r["ground_truth"]) for r in paws)
    print("  bucket x label:", dict(sorted(cells.items())))
    check("paws: every bucket has both labels with >= 30 rows", all(cells[(b, l)] >= 30 for b in ("low", "mid", "high") for l in labels))
    s1s = [v6.paws_sentences(r["context"])[0] for r in paws]
    check("paws: no shared sentence 1", len(set(s1s)) == len(s1s))

    # ---- minimal pairs
    pairs = read_rows(DATA / "aux" / "paws_minimal_pairs.jsonl")
    by_id = {r["id"]: r for r in paws}
    ok_parent = all(p["metadata"]["minimal_pair_of"] in by_id for p in pairs)
    ok_swap = ok_neg = ok_ctrl = True
    n = Counter()
    for p in pairs:
        par = by_id[p["metadata"]["minimal_pair_of"]]
        s1, s2 = v6.paws_sentences(par["context"])
        q1, q2 = v6.paws_sentences(p["context"])
        kind = p["metadata"]["perturbation"]
        n[kind] += 1
        if kind == "swap_order":
            ok_swap &= (q1, q2) == (s2, s1) and p["ground_truth"] == par["ground_truth"]
        elif kind == "negation":
            ok_neg &= q1 == s1 and q2 != s1 and q2.replace(" not", "", 1) == s1 and p["ground_truth"] == "not_paraphrase"
        elif kind == "negation_control":
            ok_ctrl &= q1 == q2 == s1 and p["ground_truth"] == "paraphrase"
    check("pairs: every parent exists and lists the kind", ok_parent and all(
        p["metadata"]["perturbation"] in by_id[p["metadata"]["minimal_pair_of"]]["metadata"]["minimal_pairs"] for p in pairs))
    check(f"pairs: swap_order exact and label preserved (n={n['swap_order']})", ok_swap and n["swap_order"] == len(paws))
    check(f"pairs: negation differs only by one inserted 'not' (n={n['negation']})", ok_neg and n["negation"] > 0)
    check(f"pairs: negation_control is identical text (n={n['negation_control']})", ok_ctrl and n["negation_control"] == n["negation"])
    check("pairs: all parse with v6 paws_sentences and use the same question line",
          all(p["context"].endswith("\nDo these two sentences have the exact same meaning?") for p in pairs))

    bad = [name for name, ok in results if not ok]
    print(f"{len(results) - len(bad)}/{len(results)} PASS")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
