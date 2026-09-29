#!/usr/bin/env python3
"""MMLU train-split converter into gen-zero six-field JSONL.

MMLU (`cais/mmlu`) is a zero/few-shot eval benchmark by design: every one of
its 57 per-subject configs only ships `dev` (5-shot exemplars), `validation`,
and `test` splits -- there is no per-subject train partition. The dataset's
own **`auxiliary_train`** config is the one upstream-declared `train` split
(99,842 rows), assembled by the MMLU authors from other multiple-choice exam
sources for training use. Per its own parquet data, this split's `subject`
field is always the empty string (verified: every one of the 99,842 rows) --
that is an honest upstream property, not something this converter fabricates
or infers, so it is recorded as `metadata.subject = "unspecified"` rather
than guessing a stem/humanities/social_sciences label from question text.

Converts each row into the six-field schema
{"id","task","context","candidates","ground_truth","metadata"} and drops any
row whose context collides with the frozen eval set or the calibration split
under the physical leak_gate SHA256 dedup.

No language-specific regex or keyword heuristics: candidates are built purely
from the option's ordinal position and ground_truth from the dataset's own
integer `answer` index -- never inferred from question text.

Data governance:
- Upstream: https://huggingface.co/datasets/cais/mmlu, license MIT (confirmed
  via `GET https://huggingface.co/api/datasets/cais/mmlu`).
- Raw parquet is cached outside the git repo; converted output must be written
  outside the frozen benchmarks/data/ directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leak_gate import DATA_DIR, filter_leak_free, load_excluded_hashes  # noqa: E402

MMLU_REPO = "cais/mmlu"
MMLU_CONFIG = "auxiliary_train"
MMLU_SPLIT = "train"
MMLU_TASK_NAME = "mmlu_auxiliary_train"
MMLU_LICENSE = "MIT"
MMLU_LICENSE_URL = "https://huggingface.co/datasets/cais/mmlu"
HF_USER_AGENT = "Gen-Zero-MMLU-Converter/1.0"
# Overridable so tests can point at a closed port and get a real, unmocked
# connection failure instead of hitting the live Hugging Face API.
HF_API_BASE = os.environ.get("GEN_ZERO_HF_API_BASE", "https://huggingface.co")
_LETTERS = [chr(ord("A") + i) for i in range(26)]


def default_cache_dir() -> Path:
    env_dir = os.environ.get("GEN_ZERO_EVAL_DATA_DIR")
    if env_dir:
        return Path(env_dir) / "raw_datasets"
    sibling = DATA_DIR.parents[2] / "gen-zero-eval-data" / "raw_datasets"
    if sibling.parent.exists():
        return sibling
    return DATA_DIR.parent / ".cache" / "raw_datasets"


def resolve_parquet_url(repo: str, config: str, split: str, timeout: int = 30) -> str:
    api_url = f"{HF_API_BASE}/api/datasets/{repo}/parquet"
    req = urllib.request.Request(api_url, headers={"User-Agent": HF_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        catalog = json.loads(resp.read().decode("utf-8"))
    urls = catalog.get(config, {}).get(split)
    if not urls:
        raise RuntimeError(f"no parquet url for {repo}:{config}:{split}")
    return urls[0]


def download_parquet(url: str, cache_path: Path, timeout: int = 180) -> Path:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    req = urllib.request.Request(url, headers={"User-Agent": HF_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return cache_path


def read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    """MMLU's auxiliary_train parquet nests each row under a `train` struct column."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if "train" in table.column_names:
        col = table.column("train")
        return [col[i].as_py() for i in range(table.num_rows)]
    cols = table.column_names
    return [{c: table[c][i].as_py() for c in cols} for i in range(table.num_rows)]


def mmlu_row_to_record(row: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Pure, network-free conversion of one raw cais/mmlu auxiliary_train row."""
    question = (row.get("question") or "").strip()
    choices_raw = row.get("choices") or []
    answer = row.get("answer")

    choice_texts = [str(c).strip() for c in choices_raw]
    if not question or len(choice_texts) < 2 or len(choice_texts) > len(_LETTERS):
        return None
    if any(not t for t in choice_texts) or len(set(choice_texts)) != len(choice_texts):
        return None
    if not isinstance(answer, int) or isinstance(answer, bool) or not (0 <= answer < len(choice_texts)):
        return None

    labels = _LETTERS[: len(choice_texts)]
    formatted = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, choice_texts))
    context = f"Question: {question}\n{formatted}\nSelect the single correct option."
    ground_truth = labels[answer]
    subject = str(row.get("subject") or "").strip()

    return {
        "id": f"{MMLU_TASK_NAME}-{index:06d}",
        "task": MMLU_TASK_NAME,
        "context": context,
        "candidates": labels,
        "ground_truth": ground_truth,
        "metadata": {
            "source": "huggingface",
            "repo": MMLU_REPO,
            "config": MMLU_CONFIG,
            "split": MMLU_SPLIT,
            "subject": subject or "unspecified",
            "license": MMLU_LICENSE,
            "license_url": MMLU_LICENSE_URL,
        },
    }


def convert_mmlu_split(rows: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        record = mmlu_row_to_record(row, i)
        if record is not None:
            records.append(record)
        if limit and len(records) >= limit:
            break
    return records


def fetch_and_convert(cache_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    url = resolve_parquet_url(MMLU_REPO, MMLU_CONFIG, MMLU_SPLIT)
    cache_path = cache_dir / f"mmlu_{MMLU_CONFIG}_{MMLU_SPLIT}.parquet"
    download_parquet(url, cache_path)
    rows = read_parquet_rows(cache_path)
    return convert_mmlu_split(rows, limit=limit)


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory, must be outside benchmarks/data")
    parser.add_argument("--cache-dir", type=Path, default=None, help="raw parquet cache directory")
    parser.add_argument("--limit", type=int, default=None, help="cap converted rows (smoke testing only)")
    args = parser.parse_args()

    resolved_output = args.output_dir.resolve()
    if resolved_output == DATA_DIR.resolve() or DATA_DIR.resolve() in resolved_output.parents:
        parser.error("--output-dir must be outside the frozen benchmarks/data directory")

    cache_dir = args.cache_dir or default_cache_dir()
    excluded = load_excluded_hashes()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = fetch_and_convert(cache_dir, limit=args.limit)
    kept, stats = filter_leak_free(records, excluded)
    if not kept:
        raise SystemExit(f"{MMLU_TASK_NAME}: zero rows survived the leak gate")
    out_path = args.output_dir / f"{MMLU_TASK_NAME}.jsonl"
    write_jsonl(kept, out_path)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"{MMLU_TASK_NAME}: {stats} sha256={digest} -> {out_path}")


if __name__ == "__main__":
    main()
