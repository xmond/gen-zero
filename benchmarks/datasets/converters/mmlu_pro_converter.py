#!/usr/bin/env python3
"""MMLU-Pro converter into gen-zero six-field JSONL.

MMLU-Pro (`TIGER-Lab/MMLU-Pro`) ships only two splits upstream: `test`
(12,032 rows, the main benchmark) and `validation` (70 few-shot CoT
exemplars). There is no `train` config -- unlike plain MMLU, MMLU-Pro has no
`auxiliary_train` equivalent. This converter uses the `test` split as the
distillation-training source and records `metadata.split = "test"` honestly
rather than mislabeling it "train". This is safe against self-leak because
MMLU-Pro is not one of the 13 tasks in the frozen 930-question eval manifest
(benchmarks/data/manifest.json) -- confirmed by inspection, no MMLU or
MMLU-Pro task is present there -- and the physical leak_gate below still
gates every row against that manifest plus the calibration split as a
belt-and-suspenders check, not because a known collision exists.

Converts each row into the six-field schema
{"id","task","context","candidates","ground_truth","metadata"} and drops any
row whose context collides with the frozen eval set or the calibration split
under the physical leak_gate SHA256 dedup.

No language-specific regex or keyword heuristics: candidates are built purely
from each option's ordinal position (A..J, up to 10 seen upstream) and
ground_truth from the dataset's own integer `answer_index` field -- never
inferred from question text. The 14-category taxonomy (`category`, e.g.
"biology", "law", "psychology") and the `src` provenance field are carried
through into metadata verbatim, never guessed.

Data governance:
- Upstream: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro. License is
  MIT -- confirmed via `GET https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro`
  (`cardData.license` / tag `license:mit`), not Apache-2.0.
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

MMLU_PRO_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_CONFIG = "default"
MMLU_PRO_SPLIT = "test"
MMLU_PRO_TASK_NAME = "mmlu_pro_test"
MMLU_PRO_LICENSE = "MIT"
MMLU_PRO_LICENSE_URL = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
HF_USER_AGENT = "Gen-Zero-MMLUPro-Converter/1.0"
# Overridable so tests can point at a closed port and get a real, unmocked
# connection failure instead of hitting the live Hugging Face API.
HF_API_BASE = os.environ.get("GEN_ZERO_HF_API_BASE", "https://huggingface.co")
_LETTERS = [chr(ord("A") + i) for i in range(26)]

# The 14 upstream MMLU-Pro categories, verified against the live test parquet.
MMLU_PRO_CATEGORIES = frozenset(
    {
        "biology",
        "business",
        "chemistry",
        "computer science",
        "economics",
        "engineering",
        "health",
        "history",
        "law",
        "math",
        "other",
        "philosophy",
        "physics",
        "psychology",
    }
)


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


def download_parquet(url: str, cache_path: Path, timeout: int = 120) -> Path:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    req = urllib.request.Request(url, headers={"User-Agent": HF_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return cache_path


def read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.column_names
    return [{c: table[c][i].as_py() for c in cols} for i in range(table.num_rows)]


def mmlu_pro_row_to_record(row: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """Pure, network-free conversion of one raw TIGER-Lab/MMLU-Pro row."""
    question = (row.get("question") or "").strip()
    options_raw = row.get("options") or []
    answer_index = row.get("answer_index")

    choice_texts = [str(c).strip() for c in options_raw]
    if not question or len(choice_texts) < 2 or len(choice_texts) > len(_LETTERS):
        return None
    if any(not t for t in choice_texts) or len(set(choice_texts)) != len(choice_texts):
        return None
    if not isinstance(answer_index, int) or isinstance(answer_index, bool):
        return None
    if not (0 <= answer_index < len(choice_texts)):
        return None

    labels = _LETTERS[: len(choice_texts)]
    formatted = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, choice_texts))
    context = f"Question: {question}\n{formatted}\nSelect the single correct option."
    ground_truth = labels[answer_index]

    category = str(row.get("category") or "").strip()
    src = str(row.get("src") or "").strip()
    upstream_id = str(row.get("question_id") if row.get("question_id") is not None else f"mmlu-pro-{index:05d}")

    return {
        "id": f"{MMLU_PRO_TASK_NAME}-{index:05d}",
        "task": MMLU_PRO_TASK_NAME,
        "context": context,
        "candidates": labels,
        "ground_truth": ground_truth,
        "metadata": {
            "question_id": upstream_id,
            "source": "huggingface",
            "repo": MMLU_PRO_REPO,
            "config": MMLU_PRO_CONFIG,
            "split": MMLU_PRO_SPLIT,
            "category": category or "unspecified",
            "src": src or "unspecified",
            "license": MMLU_PRO_LICENSE,
            "license_url": MMLU_PRO_LICENSE_URL,
        },
    }


def convert_mmlu_pro_split(rows: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        record = mmlu_pro_row_to_record(row, i)
        if record is not None:
            records.append(record)
        if limit and len(records) >= limit:
            break
    return records


def fetch_and_convert(cache_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Zero network calls when the parquet is already cached; otherwise fetches once.

    The URL-resolution API call must not run when the file is already on disk --
    doing it unconditionally would mean "offline" mode still depends on network
    reachability even with a full local cache, which defeats the point of caching.
    """
    cache_path = cache_dir / f"mmlu_pro_{MMLU_PRO_CONFIG}_{MMLU_PRO_SPLIT}.parquet"
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        try:
            url = resolve_parquet_url(MMLU_PRO_REPO, MMLU_PRO_CONFIG, MMLU_PRO_SPLIT)
            download_parquet(url, cache_path)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"{MMLU_PRO_TASK_NAME}: network unreachable and no cached parquet at {cache_path}; "
                "place the parquet there manually for an offline conversion"
            ) from exc
    rows = read_parquet_rows(cache_path)
    return convert_mmlu_pro_split(rows, limit=limit)


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
        raise SystemExit(f"{MMLU_PRO_TASK_NAME}: zero rows survived the leak gate")
    out_path = args.output_dir / f"{MMLU_PRO_TASK_NAME}.jsonl"
    write_jsonl(kept, out_path)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"{MMLU_PRO_TASK_NAME}: {stats} sha256={digest} -> {out_path}")


if __name__ == "__main__":
    main()
