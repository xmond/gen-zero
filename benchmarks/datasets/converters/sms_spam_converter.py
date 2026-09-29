"""Convert the official UCI SMS Spam Collection dataset into ham/spam binary records."""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from .common import EVALUATION, isolated

LABELS = ["ham", "spam"]


def parse(raw_text: str):
    """Split each line on the first tab: label, then the raw SMS text."""
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"malformed SMS Spam Collection line {line_number}")
        label, text = parts
        yield {"label": label, "text": text}


def convert(rows, *, split: str, evaluation_path: Path = EVALUATION, stats: Counter | None = None):
    if split != "train":
        raise ValueError("SMS Spam Collection conversion requires the official train split")
    def records():
        for index, source in enumerate(rows):
            label, text = source["label"], source["text"]
            if label not in LABELS or not isinstance(text, str) or not text.strip():
                raise ValueError(f"invalid SMS Spam Collection row {index}")
            yield {"id": f"sms_spam-train-{index:05d}", "task": "sms_spam",
                   "context": text, "candidates": LABELS.copy(),
                   "ground_truth": label,
                   "metadata": {"source": "uciml/sms-spam-collection-dataset", "split": "train", "source_index": index}}
    return list(isolated(records(), evaluation_path, stats=stats))


def load_official(*, evaluation_path: Path = EVALUATION, stats: Counter | None = None):
    """Load only the official UCI SMS Spam Collection zip (5,574 labeled messages)."""
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        text = archive.read("SMSSpamCollection").decode("utf-8")
    rows = list(parse(text))
    if len(rows) != 5574:
        raise ValueError(f"unexpected official SMS Spam Collection size: {len(rows)}")
    return convert(rows, split="train", evaluation_path=evaluation_path, stats=stats)


def write_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
