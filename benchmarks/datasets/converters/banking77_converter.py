"""Convert the official BANKING77 train split into 77-way intent records."""
from __future__ import annotations

import json
import csv
import io
import urllib.request
from pathlib import Path

from .common import EVALUATION, isolated


def convert(rows, label_names: list[str], *, split: str, evaluation_path: Path = EVALUATION):
    if split != "train":
        raise ValueError("BANKING77 conversion requires the official train split")
    if len(label_names) != 77 or len(set(label_names)) != 77 or any(not isinstance(x, str) or not x.strip() for x in label_names):
        raise ValueError("BANKING77 requires 77 distinct nonempty official label names")
    def records():
        for index, source in enumerate(rows):
            utterance, label = source["text"], source["label"]
            if not isinstance(utterance, str) or not utterance.strip() or type(label) is not int or not 0 <= label < 77:
                raise ValueError(f"invalid BANKING77 row {index}")
            yield {"id": f"banking77-train-{index:05d}", "task": "banking77",
                   "context": utterance, "candidates": label_names.copy(),
                   "ground_truth": label_names[label],
                   "metadata": {"source": "PolyAI/banking77", "split": "train", "source_index": index}}
    return list(isolated(records(), evaluation_path))


def load_official(*, evaluation_path: Path = EVALUATION):
    """Load only PolyAI's published train.csv, with its original label strings."""
    url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read().decode("utf-8-sig")
    source = list(csv.DictReader(io.StringIO(raw)))
    if len(source) != 10003:
        raise ValueError(f"unexpected official BANKING77 train size: {len(source)}")
    names = sorted({row["category"] for row in source})
    if len(names) != 77:
        raise ValueError("official BANKING77 train must contain 77 categories")
    indexes = {name: index for index, name in enumerate(names)}
    rows = ({"text": row["text"], "label": indexes[row["category"]]} for row in source)
    return convert(rows, names, split="train", evaluation_path=evaluation_path)


def write_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
