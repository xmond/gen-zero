#!/usr/bin/env python3
"""Rebuild benchmarks/data/{gsm8k,paws}.jsonl for the A100 zero-token protocol (§1, §3.1, §3.2).

GSM8K  openai/gsm8k `main` test split. Stratified by solution length, seeded. The four
       options hold the gold value plus three distinct wrong values; the gold letter is
       balanced and randomly placed (never fixed to 'A'). The exact numeric gold is kept in
       `metadata.raw_answer` ('#### <n>') and `metadata.gold_numeric` (canonical rational).
PAWS   google-research-datasets/paws `labeled_final` test split. Label-balanced, stratified
       by lexical-overlap bucket, seeded. Minimal pairs go to data/aux/ (a `*.jsonl` file in
       data/ itself would be read as a new task by run_remote_eval_v6.py).

Also refreshes data/all_benchmarks.jsonl and data/manifest.json (SHA256, counts, seeds, HF
revisions, sample ids, label maps). Output has no timestamps, so a rerun is byte-identical.

    python3 rebuild_gsm8k_paws.py --cache-dir /tmp/hfraw
    python3 verify_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_numeric import gold_from_raw_answer, normalize  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 20260921                                  # protocol §1 first seed
GSM8K_N = 200                                    # protocol §3.1 preregistered subset
PAWS_N = 400                                     # 200 per label
LETTERS = ["A", "B", "C", "D"]
PAWS_CANDIDATES = ["paraphrase", "not_paraphrase"]
PAWS_QUESTION = "Do these two sentences have the exact same meaning?"

# Fixed before sampling. Unigram Jaccard over lower-cased whitespace tokens.
OVERLAP_THRESHOLDS = {"low_lt": 0.80, "high_ge": 0.90}
PAWS_MIN_PER_LABEL_BUCKET = 30
GSM8K_STEP_BUCKETS = ["0-1", "2", "3", "4", "5+"]
GSM8K_MIN_PER_BUCKET = 10

_ANNOT = re.compile(r"<<([^<>]*?)=([^<>]*?)>>")


def rng_for(*parts: Any) -> random.Random:
    """Deterministic RNG per key. str seeds hash via sha512, so PYTHONHASHSEED is irrelevant."""
    return random.Random(":".join(str(p) for p in (SEED, *parts)))


def allocate(total: int, pop: Dict[str, int], floor: int) -> Dict[str, int]:
    """Proportional allocation by largest remainder, every stratum at least `floor`."""
    keys = sorted(pop)
    alloc = {k: min(floor, pop[k]) for k in keys}
    left = total - sum(alloc.values())
    weight = {k: pop[k] - alloc[k] for k in keys}
    wsum = sum(weight.values())
    exact = {k: left * weight[k] / wsum for k in keys}
    for k in keys:
        alloc[k] += int(exact[k])
    rem = total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - int(exact[k])), k))[:rem]:
        alloc[k] += 1
    assert sum(alloc.values()) == total and all(alloc[k] <= pop[k] for k in keys)
    return alloc


# ---------------------------------------------------------------- GSM8K

def step_bucket(answer: str) -> Tuple[int, str]:
    n = len(_ANNOT.findall(answer))
    return n, ("0-1" if n <= 1 else "5+" if n >= 5 else str(n))


def operations(answer: str, question: str) -> List[str]:
    exprs = " ".join(m.group(1) for m in _ANNOT.finditer(answer))
    ops = []
    for name, sym in (("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")):
        if re.search(re.escape(sym), exprs):
            ops.append(name)
    if "%" in exprs or "%" in question or re.search(r"\bpercent", question, re.I):
        ops.append("percent")
    return ops


def _intermediates(answer: str) -> List[int]:
    out = []
    for m in _ANNOT.finditer(answer):
        v = normalize(m.group(2))
        if v is not None and "/" not in v:
            out.append(int(v))
    return out


def make_distractors(gold: int, answer: str, rng: random.Random, rank: int) -> List[Tuple[int, str]]:
    """Three distinct wrong non-negative integers with a recorded origin.

    `rank` is the wanted value-rank of the gold among the four options (0 = smallest). The
    caller draws it uniformly, so the gold is not systematically the smallest or largest
    option. That removes the tell the old `val-2, val+3, val*2` scheme had.
    """
    pool: List[Tuple[int, str]] = [(v, "intermediate_step") for v in _intermediates(answer)]
    mag = max(abs(gold), 1)
    for _ in range(4):
        d = max(1, round(mag * rng.choice([0.05, 0.1, 0.2, 0.25, 0.5])))
        pool += [(gold - d, "offset"), (gold + d, "offset")]
    for d in (1, 2, 5, 10):
        pool += [(gold - d, "small_offset"), (gold + d, "small_offset")]
    pool += [(gold * 2, "scale"), (gold * 3, "scale"), (gold * 10, "scale")]
    if gold % 2 == 0:
        pool.append((gold // 2, "scale"))
    s = str(abs(gold))
    if len(s) >= 2:
        i = rng.randrange(len(s) - 1)
        pool.append((int(s[:i] + s[i + 1] + s[i] + s[i + 2:]), "digit_swap"))
    rng.shuffle(pool)
    # Last resort, tried only after the shuffled pool: unit steps on both sides.
    pool += [(gold + sg * d, "fallback") for d in range(1, 40) for sg in (-1, 1)]

    lo: List[Tuple[int, str]] = []
    hi: List[Tuple[int, str]] = []
    seen = {gold}
    for v, kind in pool:
        if v in seen or v < 0:
            continue
        seen.add(v)
        (lo if v < gold else hi).append((v, kind))
    want_lo = min(rank, len(lo))
    want_hi = 3 - want_lo
    if want_hi > len(hi):
        want_hi = len(hi)
        want_lo = 3 - want_hi
    picked = lo[:want_lo] + hi[:want_hi]
    assert len(picked) == 3, f"cannot build distractors for {gold}"
    rng.shuffle(picked)                          # slot order must not follow value order
    return picked


def build_gsm8k(rows: List[Dict[str, str]], repo_sha: str) -> Tuple[List[dict], Dict[str, Any]]:
    by_bucket: Dict[str, List[int]] = {b: [] for b in GSM8K_STEP_BUCKETS}
    for i, r in enumerate(rows):
        by_bucket[step_bucket(r["answer"])[1]].append(i)
    pop = {b: len(v) for b, v in by_bucket.items()}
    alloc = allocate(GSM8K_N, pop, GSM8K_MIN_PER_BUCKET)
    chosen: List[int] = []
    for b in GSM8K_STEP_BUCKETS:
        chosen += rng_for("gsm8k-sample", b).sample(by_bucket[b], alloc[b])
    chosen.sort()                                # test-split order, not stratum order

    # Gold letter: balanced by construction, then shuffled. Exactly n/4 per letter.
    positions = [i % 4 for i in range(len(chosen))]
    rng_for("gsm8k-positions").shuffle(positions)
    # Gold value-rank among the options, also balanced and drawn independently of the letter.
    ranks = [i % 4 for i in range(len(chosen))]
    rng_for("gsm8k-ranks").shuffle(ranks)

    samples = []
    for k, idx in enumerate(chosen):
        r = rows[idx]
        sid = f"gsm8k-t{idx:04d}"
        gold_frac = gold_from_raw_answer(r["answer"])
        assert gold_frac.denominator == 1, f"non-integer gold in {sid}"
        gold = int(gold_frac)
        rng = rng_for("gsm8k-options", sid)
        distractors = make_distractors(gold, r["answer"], rng, ranks[k])
        order = [None] * 4                       # slot -> (value, kind)
        order[positions[k]] = (gold, "gold")
        rest = iter(distractors)
        for slot in range(4):
            if order[slot] is None:
                order[slot] = next(rest)
        opts = "\n".join(f"({L}) {v}" for L, (v, _) in zip(LETTERS, order))
        n_steps, bucket = step_bucket(r["answer"])
        gold_raw = r["answer"].split("####")[-1].strip()
        samples.append({
            "id": sid,
            "task": "gsm8k",
            "context": f"Problem: {r['question'].strip()}\n{opts}\nSelect the correct calculated final solution.",
            "candidates": list(LETTERS),
            "ground_truth": LETTERS[positions[k]],
            "metadata": {
                "raw_answer": r["answer"].strip(),
                "gold_raw": gold_raw,
                "gold_numeric": normalize(gold_raw),
                "option_values": [str(v) for v, _ in order],
                "option_kinds": [kind for _, kind in order],
                "n_steps": n_steps,
                "step_bucket": bucket,
                "operations": operations(r["answer"], r["question"]),
                "test_index": idx,
                "source": "huggingface",
                "repo": "openai/gsm8k",
                "config": "main",
                "split": "test",
                "repo_revision": repo_sha,
            },
        })
    rank_counts: Dict[str, int] = {}
    for smp in samples:
        vals = [int(v) for v in smp["metadata"]["option_values"]]
        rk = sorted(vals).index(int(smp["metadata"]["gold_numeric"]))
        rank_counts[str(rk)] = rank_counts.get(str(rk), 0) + 1
    info = {
        "population": pop, "allocation": alloc, "seed": SEED, "sample_order": "test-split index",
        "gold_letter_counts": {L: sum(s["ground_truth"] == L for s in samples) for L in LETTERS},
        "letter_assignment": "balanced (n/4 per letter), shuffled with seed",
        "distractor_scheme": "intermediate_step | offset | small_offset | scale | digit_swap | fallback; "
                             "3 distinct non-gold non-negative integers from a shuffled pool, chosen so the gold "
                             "value-rank among the options is balanced (n/4 per rank where feasible)",
        "gold_value_rank_counts": rank_counts,
        "normalization": "fractions.Fraction via benchmarks/datasets/gsm8k_numeric.py",
    }
    return samples, info


# ---------------------------------------------------------------- PAWS

def _toks(s: str) -> List[str]:
    return s.lower().split()


def overlap(s1: str, s2: str) -> Dict[str, Any]:
    a, b = _toks(s1), _toks(s2)
    ua, ub = set(a), set(b)
    uj = len(ua & ub) / len(ua | ub)
    ba, bb = set(zip(a, a[1:])), set(zip(b, b[1:]))
    bj = len(ba & bb) / max(1, len(ba | bb))
    bucket = "low" if uj < OVERLAP_THRESHOLDS["low_lt"] else "high" if uj >= OVERLAP_THRESHOLDS["high_ge"] else "mid"
    return {"unigram_jaccard": round(uj, 4), "bigram_jaccard": round(bj, 4),
            "same_bag_of_words": sorted(a) == sorted(b), "bucket": bucket}


def paws_context(s1: str, s2: str) -> str:
    return f"Sentence 1: {s1}\nSentence 2: {s2}\n{PAWS_QUESTION}"


_NEG_BLOCK = re.compile(r"\b(?:not|no|never|neither|nor|cannot)\b|n't|\bwithout\b", re.I)
_AUX = re.compile(r"\b(was|is|were|are|has|had|have)\b")


def negate(sentence: str) -> Optional[str]:
    """Insert 'not' after the first simple auxiliary. None when the rule does not apply cleanly."""
    if _NEG_BLOCK.search(sentence):
        return None
    m = _AUX.search(sentence)
    if m is None or m.start() == 0:
        return None
    return sentence[:m.end()] + " not" + sentence[m.end():]


def minimal_pairs(sid: str, s1: str, s2: str, label: str) -> List[dict]:
    """Deterministic, label-verifiable pairs derived from one PAWS row.

    swap_order       (s2, s1). Paraphrase is symmetric, so the label is unchanged. Exact.
    negation         (s1, s1 with 'not' inserted). Meaning differs by construction: not_paraphrase.
    negation_control (s1, s1). Identical text: paraphrase. Twin of `negation`; the two differ
                     only by the inserted 'not'.
    Time-adverbial moves are not generated: no rule preserves the label mechanically.
    """
    out = [{"kind": "swap_order", "s1": s2, "s2": s1, "gt": label, "relation": "label_preserved"}]
    neg = negate(s1)
    if neg is not None:
        out.append({"kind": "negation", "s1": s1, "s2": neg, "gt": "not_paraphrase", "relation": "label_fixed_by_rule"})
        out.append({"kind": "negation_control", "s1": s1, "s2": s1, "gt": "paraphrase", "relation": "label_fixed_by_rule"})
    return [{
        "is_synthetic": True,
        "id": f"{sid}:{p['kind']}",
        "task": "paws",
        "context": paws_context(p["s1"], p["s2"]),
        "candidates": list(PAWS_CANDIDATES),
        "ground_truth": p["gt"],
        "metadata": {
            "minimal_pair_of": sid,
            "perturbation": p["kind"],
            "label_relation": p["relation"],
            "generator": "rebuild_gsm8k_paws.minimal_pairs/rule_v1",
            "lexical_overlap": overlap(p["s1"], p["s2"]),
            "source": "derived", "parent_repo": "google-research-datasets/paws",
        },
    } for p in out]


def build_paws(rows: List[Dict[str, Any]], repo_sha: str) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    seen, pool = set(), []
    for r in rows:                               # one row per sentence1: no shared-sentence leakage
        s1, s2 = r["sentence1"].strip(), r["sentence2"].strip()
        if not s1 or not s2 or s1 in seen:
            continue
        seen.add(s1)
        pool.append({**r, "sentence1": s1, "sentence2": s2, "_ov": overlap(s1, s2)})

    per_label = PAWS_N // 2
    chosen: List[dict] = []
    alloc_info: Dict[str, Any] = {}
    for lab, name in ((1, "paraphrase"), (0, "not_paraphrase")):
        sub = [r for r in pool if int(r["label"]) == lab]
        by_b = {b: [r for r in sub if r["_ov"]["bucket"] == b] for b in ("low", "mid", "high")}
        pop = {b: len(v) for b, v in by_b.items()}
        alloc = allocate(per_label, pop, PAWS_MIN_PER_LABEL_BUCKET)
        alloc_info[name] = {"population": pop, "allocation": alloc}
        for b in ("low", "mid", "high"):
            chosen += rng_for("paws-sample", name, b).sample(by_b[b], alloc[b])
    chosen.sort(key=lambda r: int(r["id"]))      # PAWS id order
    # Interleave labels deterministically so any prefix of the file is not label-sorted.
    rng_for("paws-order").shuffle(chosen)

    samples, pairs = [], []
    for r in chosen:
        sid = f"paws-t{int(r['id']):05d}"
        label = "paraphrase" if int(r["label"]) == 1 else "not_paraphrase"
        mp = minimal_pairs(sid, r["sentence1"], r["sentence2"], label)
        samples.append({
            "id": sid,
            "task": "paws",
            "context": paws_context(r["sentence1"], r["sentence2"]),
            "candidates": list(PAWS_CANDIDATES),
            "ground_truth": label,
            "metadata": {
                "original_id": str(r["id"]),
                "lexical_overlap": r["_ov"],
                "overlap_bucket": r["_ov"]["bucket"],
                "minimal_pair_of": None,
                "minimal_pairs": [p["metadata"]["perturbation"] for p in mp],
                "minimal_pair_file": "aux/paws_minimal_pairs.jsonl",
                "source": "huggingface",
                "repo": "google-research-datasets/paws",
                "config": "labeled_final",
                "split": "test",
                "repo_revision": repo_sha,
            },
        })
        pairs += mp
    counts = {b: {lab: sum(1 for s in samples if s["metadata"]["overlap_bucket"] == b and s["ground_truth"] == lab)
                  for lab in PAWS_CANDIDATES} for b in ("low", "mid", "high")}
    kinds: Dict[str, int] = {}
    for p in pairs:
        kinds[p["metadata"]["perturbation"]] = kinds.get(p["metadata"]["perturbation"], 0) + 1
    info = {
        "seed": SEED, "dedupe": "one row per sentence1 in the split",
        "overlap_metric": "unigram Jaccard over lower-cased whitespace tokens",
        "overlap_thresholds": OVERLAP_THRESHOLDS, "per_label_bucket_floor": PAWS_MIN_PER_LABEL_BUCKET,
        "stratification": alloc_info, "bucket_label_counts": counts,
        "label_counts": {lab: sum(s["ground_truth"] == lab for s in samples) for lab in PAWS_CANDIDATES},
        "minimal_pair_counts": kinds,
        "minimal_pair_file": "aux/paws_minimal_pairs.jsonl",
        "minimal_pair_kinds_not_generated": ["time_adverbial_move", "subject_object_swap"],
        "sample_order": "seeded shuffle",
    }
    return samples, pairs, info


# ---------------------------------------------------------------- IO + manifest

def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hf_revision(repo: str) -> str:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo).sha or "unknown"
    except Exception as exc:                     # offline rebuild from cache is still allowed
        return f"unknown ({type(exc).__name__})"


def refresh_manifest(gsm_info: Dict[str, Any], paws_info: Dict[str, Any], revisions: Dict[str, str]) -> Dict[str, Any]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    order = list(manifest["tasks"])
    all_rows: List[str] = []
    total = 0
    for task in order:
        entry = manifest["tasks"][task]
        path = DATA_DIR / entry["file"]
        lines = [x for x in path.read_text(encoding="utf-8").split("\n") if x]
        rows = [json.loads(x) for x in lines]
        all_rows += lines
        total += len(rows)
        entry["samples_count"] = len(rows)
        entry["sha256"] = sha256_file(path)
        entry["sample_ids"] = [r["id"] for r in rows]
        entry.pop("label_map", None), entry.pop("label_counts", None)
        entry["label_map"] = {c: i for i, c in enumerate(entry["candidates"])}
        entry["label_counts"] = {c: sum(r["ground_truth"] == c for r in rows) for c in entry["candidates"]}
    (DATA_DIR / "all_benchmarks.jsonl").write_text("\n".join(all_rows) + "\n", encoding="utf-8")

    manifest["version"] = "1.1.0"
    manifest["total_samples"] = total
    manifest["schema_version"] = 1
    manifest["schema_check"] = {
        "required_keys": ["id", "task", "context", "candidates", "ground_truth", "metadata"],
        "verifier": "benchmarks/datasets/verify_data.py",
        "invariants": "ids unique; ground_truth in candidates; sha256 and counts match files",
    }
    manifest["all_benchmarks"] = {"file": "all_benchmarks.jsonl", "samples_count": total,
                                  "sha256": sha256_file(DATA_DIR / "all_benchmarks.jsonl"),
                                  "note": "concatenation of the task files in manifest order; runners skip it"}
    tasks = manifest["tasks"]
    tasks["gsm8k"].update({
        "hf_repo": "openai/gsm8k", "hf_config": "main", "hf_split": "test",
        "hf_revision": revisions["openai/gsm8k"], "seed": SEED, "build": gsm_info,
        "scoring": "letter gold for the MCQ view; numeric gold in metadata.gold_numeric "
                   "(exact rational compare, benchmarks/datasets/gsm8k_numeric.py)",
    })
    tasks["paws"].update({
        "hf_repo": "google-research-datasets/paws", "hf_config": "labeled_final", "hf_split": "test",
        "hf_revision": revisions["google-research-datasets/paws"], "seed": SEED, "build": paws_info,
    })
    aux = DATA_DIR / "aux" / "paws_minimal_pairs.jsonl"
    manifest["aux_files"] = {"aux/paws_minimal_pairs.jsonl": {
        "sha256": sha256_file(aux),
        "samples_count": sum(1 for _ in open(aux, encoding="utf-8")),
        "note": "not read by run_remote_eval_v6.py (data/*.jsonl is non-recursive)"}}
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=None, help="HF datasets cache directory")
    args = ap.parse_args()
    from datasets import load_dataset

    revisions = {repo: hf_revision(repo) for repo in ("openai/gsm8k", "google-research-datasets/paws")}
    pin = {repo: (sha if re.fullmatch(r"[0-9a-f]{40}", sha) else None) for repo, sha in revisions.items()}
    gsm_rows = [dict(r) for r in load_dataset("openai/gsm8k", "main", split="test",
                                              cache_dir=args.cache_dir, revision=pin["openai/gsm8k"])]
    paws_rows = [dict(r) for r in load_dataset("google-research-datasets/paws", "labeled_final", split="test",
                                                cache_dir=args.cache_dir,
                                                revision=pin["google-research-datasets/paws"])]
    assert len(gsm_rows) == 1319 and len(paws_rows) == 8000, "unexpected split size"

    gsm, gsm_info = build_gsm8k(gsm_rows, revisions["openai/gsm8k"])
    paws, pairs, paws_info = build_paws(paws_rows, revisions["google-research-datasets/paws"])
    write_jsonl(DATA_DIR / "gsm8k.jsonl", gsm)
    write_jsonl(DATA_DIR / "paws.jsonl", paws)
    write_jsonl(DATA_DIR / "aux" / "paws_minimal_pairs.jsonl", pairs)
    manifest = refresh_manifest(gsm_info, paws_info, revisions)
    print(f"gsm8k: {len(gsm)} rows, gold letters {gsm_info['gold_letter_counts']}")
    print(f"paws:  {len(paws)} rows, labels {paws_info['label_counts']}, buckets {paws_info['bucket_label_counts']}")
    print(f"pairs: {len(pairs)} rows, {paws_info['minimal_pair_counts']}")
    print(f"manifest total_samples={manifest['total_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
