#!/usr/bin/env python3
"""ARC-Easy / ARC-Challenge train-split converter into gen-zero six-field JSONL.

Pulls the real `allenai/ai2_arc` **train** split (never test/validation -- the
frozen 930-question eval set already draws its 30 ARC-Challenge rows from the
`test` split) from Hugging Face, converts each row into the six-field schema
{"id","task","context","candidates","ground_truth","metadata"}, and drops any
row whose context collides with the frozen eval set or the calibration split
under the physical leak_gate SHA256 dedup.

No language-specific regex or keyword heuristics: candidates and ground_truth
come directly from the dataset's own `choices.label` / `answerKey` fields,
with one normalization: 124 train rows (all New York Regents exam questions,
verified against the live parquet) ship digit labels ("1".."4") instead of
letters, and get mapped 1:1 to "A".."D" so the whole ARC train pool shares one
uniform candidate alphabet with every other converter's output.

Data governance:
- Upstream: https://huggingface.co/datasets/allenai/ai2_arc, license CC-BY-SA-4.0
  (confirmed via `GET https://huggingface.co/api/datasets/allenai/ai2_arc`).
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

ARC_REPO = "allenai/ai2_arc"
ARC_LICENSE = "CC-BY-SA-4.0"
ARC_LICENSE_URL = "https://huggingface.co/datasets/allenai/ai2_arc"
HF_USER_AGENT = "Gen-Zero-ARC-Converter/1.0"
# Overridable so tests can point at a closed port and get a real, unmocked
# connection failure instead of hitting the live Hugging Face API.
HF_API_BASE = os.environ.get("GEN_ZERO_HF_API_BASE", "https://huggingface.co")

# task name -> upstream HF config name
ARC_CONFIGS: Dict[str, str] = {
    "arc_challenge_train": "ARC-Challenge",
    "arc_easy_train": "ARC-Easy",
}

# 124 ARC-Challenge/ARC-Easy train rows (all New York Regents exam questions)
# ship digit labels ("1".."4") instead of letters -- verified directly against
# the live allenai/ai2_arc train parquet (124 rows with labels exactly
# ("1","2","3","4"), plus 1 outlier row with labels ("1","2","3")). Normalizing
# every numeric label to its letter equivalent keeps the candidate space a
# uniform ["A","B",...] across the whole ARC train pool instead of mixing two
# label alphabets, which the state-manifold scorer treats as a single space.
_LETTERS = [chr(ord("A") + i) for i in range(26)]
_DIGIT_TO_LETTER = {str(n): _LETTERS[n - 1] for n in range(1, len(_LETTERS) + 1)}


def _normalize_label(label: str) -> str:
    return _DIGIT_TO_LETTER.get(label, label)


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


def arc_row_to_record(row: Dict[str, Any], task_name: str, config: str, index: int) -> Optional[Dict[str, Any]]:
    """Pure, network-free conversion of one raw ai2_arc row into the six-field schema."""
    question = (row.get("question") or "").strip()
    answer_key = (row.get("answerKey") or "").strip()
    choices = row.get("choices") or {}
    raw_labels = [str(x).strip() for x in (choices.get("label") or [])]
    texts = [str(x).strip() for x in (choices.get("text") or [])]

    if not question or not answer_key or not raw_labels or not texts:
        return None
    if len(raw_labels) != len(texts) or len(raw_labels) < 2:
        return None
    if len(set(raw_labels)) != len(raw_labels):
        return None
    if any(not t for t in texts) or len(set(texts)) != len(texts):
        return None
    if answer_key not in raw_labels:
        return None

    labels = [_normalize_label(lbl) for lbl in raw_labels]
    answer_key = _normalize_label(answer_key)
    if len(set(labels)) != len(labels):
        return None

    formatted = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
    context = f"Question: {question}\n{formatted}\nSelect the single correct option."
    upstream_id = str(row.get("id") or f"{config}-train-{index:05d}")

    return {
        "id": f"{task_name}-{index:05d}",
        "task": task_name,
        "context": context,
        "candidates": labels,
        "ground_truth": answer_key,
        "metadata": {
            "question_id": upstream_id,
            "source": "huggingface",
            "repo": ARC_REPO,
            "config": config,
            "split": "train",
            "license": ARC_LICENSE,
            "license_url": ARC_LICENSE_URL,
            "label_space_normalized": raw_labels != labels,
        },
    }


def convert_arc_split(rows: List[Dict[str, Any]], task_name: str, config: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        record = arc_row_to_record(row, task_name, config, i)
        if record is not None:
            records.append(record)
        if limit and len(records) >= limit:
            break
    return records


def fetch_and_convert(task_name: str, config: str, cache_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Zero network calls when the parquet is already cached; otherwise fetches once.

    The URL-resolution API call must not run when the file is already on disk --
    doing it unconditionally would mean "offline" mode still depends on network
    reachability even with a full local cache, which defeats the point of caching.
    """
    cache_path = cache_dir / f"ai2_arc_{config}_train.parquet"
    if not (cache_path.exists() and cache_path.stat().st_size > 0):
        try:
            url = resolve_parquet_url(ARC_REPO, config, "train")
            download_parquet(url, cache_path)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"{task_name}: network unreachable and no cached parquet at {cache_path}; "
                "place the parquet there manually for an offline conversion"
            ) from exc
    rows = read_parquet_rows(cache_path)
    return convert_arc_split(rows, task_name, config, limit=limit)


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory, must be outside benchmarks/data")
    parser.add_argument("--cache-dir", type=Path, default=None, help="raw parquet cache directory")
    parser.add_argument("--limit", type=int, default=None, help="cap converted rows per config (smoke testing only)")
    args = parser.parse_args()

    resolved_output = args.output_dir.resolve()
    if resolved_output == DATA_DIR.resolve() or DATA_DIR.resolve() in resolved_output.parents:
        parser.error("--output-dir must be outside the frozen benchmarks/data directory")

    cache_dir = args.cache_dir or default_cache_dir()
    excluded = load_excluded_hashes()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, config in ARC_CONFIGS.items():
        records = fetch_and_convert(task_name, config, cache_dir, limit=args.limit)
        kept, stats = filter_leak_free(records, excluded)
        if not kept:
            raise SystemExit(f"{task_name}: zero rows survived the leak gate")
        out_path = args.output_dir / f"{task_name}.jsonl"
        write_jsonl(kept, out_path)
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        print(f"{task_name}: {stats} sha256={digest} -> {out_path}")


if __name__ == "__main__":
    main()
