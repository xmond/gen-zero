#!/usr/bin/env python3
"""Gen-Zero Benchmark Dataset Fetcher & Real Fixture Generator.

Automated loader and fixture generator for the 13 competitive benchmark tasks
from independent evaluations (Bespoke Labs Nimble & S1Bench):
1. MASSIVE en-US (Intent routing across 18 scenarios)
2. MASSIVE de-DE (Multilingual intent routing across 18 scenarios)
3. VitaminC (Contrastive fact verification: SUPPORTS / REFUTES / NOT ENOUGH INFO)
4. BoolQ (Passage Yes/No answerability)
5. SQuAD 2.0 (Passage answerability binary decision)
6. PAWS (Adversarial paraphrase identification)
7. Civil Comments (Toxicity moderation)
8. Aegis 2.0 (Prompt safety & guardrail enforcement)
9. MultiNLI (Natural language inference: entailment / neutral / contradiction)
10. PubMedQA (Biomedical QA: yes / no / maybe)
11. SummEval (Summary consistency rating: 1 to 5)
12. ARC-Challenge (Science multi-choice QA: A / B / C / D)
13. GSM8K (Grade school math reasoning decision)

Standardized Output Format (JSONL):
{
    "id": str,
    "task": str,
    "context": str,
    "candidates": List[str],
    "ground_truth": str,
    "metadata": dict
}

Data Governance:
- Raw downloads and heavy caches are stored in dedicated external storage
  (e.g., ../gen-zero-eval-data/raw_datasets or $GEN_ZERO_EVAL_DATA_DIR).
- Lightweight, validated benchmark fixtures are generated in benchmarks/data/.
- Fail-closed: only real Hugging Face data is accepted. A missing pyarrow, a failed download
  or a parse failure raises RuntimeError. There is no hand-written fixture fallback.
- Zero private filesystem paths or sensitive identifiers leaked.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


# Base directory resolution (relative, clean, zero private paths)
BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BENCHMARKS_DIR.parent
DEFAULT_DATA_DIR = BENCHMARKS_DIR / "data"


def normalized_context(value: str) -> str:
    return " ".join(value.split()).strip().lower()

# External cache directory for raw heavy datasets (outside main git repo)
def get_default_external_cache_dir() -> Path:
    env_dir = os.environ.get("GEN_ZERO_EVAL_DATA_DIR")
    if env_dir:
        return Path(env_dir) / "raw_datasets"
    # Fallback to sibling directory alongside repository root
    sibling = ROOT_DIR.parent / "gen-zero-eval-data" / "raw_datasets"
    if sibling.parent.exists():
        return sibling
    return BENCHMARKS_DIR / ".cache" / "raw_datasets"


@dataclass
class DatasetSpec:
    name: str
    display_name: str
    hf_repo: str
    hf_config: str
    hf_split: str
    candidates: List[str]
    description: str
    extractor: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]


def extract_massive_en(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 18 scenarios: alarm, audio, calendar, cooking, datetime, email, general, iot, lists,
    # music, news, play, qa, recommendation, social, takeaway, transport, weather
    label = row.get("label_text") or str(row.get("label"))
    text = row.get("text", "").strip()
    if not text or not label:
        return None
    return {
        "context": f"Route the following natural language request to its target intent domain:\n\"{text}\"",
        "ground_truth": str(label).lower(),
        "metadata": {"locale": "en-US", "original_id": str(row.get("id", ""))}
    }


def extract_massive_de(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    label = row.get("label_text") or str(row.get("label"))
    text = row.get("text", "").strip()
    if not text or not label:
        return None
    return {
        "context": f"Route the following German request to its target intent domain:\n\"{text}\"",
        "ground_truth": str(label).lower(),
        "metadata": {"locale": "de-DE", "original_id": str(row.get("id", ""))}
    }


def extract_vitaminc(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    claim = row.get("claim", "").strip()
    evidence = row.get("evidence", "").strip()
    label = row.get("label", "").strip().upper()
    if not claim or not label:
        return None
    label_map = {
        "SUPPORTS": "SUPPORTS",
        "REFUTES": "REFUTES",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO",
        "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
    }
    gt = label_map.get(label, label)
    context = f"Evidence: {evidence}\nClaim: {claim}\nDetermine whether the evidence supports, refutes, or provides not enough info for the claim."
    return {
        "context": context,
        "ground_truth": gt,
        "metadata": {"unique_id": str(row.get("unique_id", "")), "wiki_revision_id": str(row.get("wiki_revision_id", ""))}
    }


def extract_boolq(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    passage = row.get("passage", "").strip()
    question = row.get("question", "").strip()
    ans = row.get("answer")
    if ans is None or not question or not passage:
        return None
    gt = "yes" if bool(ans) else "no"
    context = f"Passage: {passage}\nQuestion: {question}?\nAnswer with yes or no based strictly on the passage."
    return {
        "context": context,
        "ground_truth": gt,
        "metadata": {"question": question}
    }


def extract_squad2(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ctx = row.get("context", "").strip()
    q = row.get("question", "").strip()
    answers = row.get("answers", {})
    if not ctx or not q:
        return None
    ans_texts = answers.get("text", []) if isinstance(answers, dict) else []
    is_answerable = len(ans_texts) > 0 and len(ans_texts[0].strip()) > 0
    gt = "answerable" if is_answerable else "unanswerable"
    context = f"Passage: {ctx}\nQuestion: {q}\nDetermine whether this question can be answered definitively using only the provided passage."
    return {
        "context": context,
        "ground_truth": gt,
        "metadata": {"original_id": str(row.get("id", "")), "num_answers": len(ans_texts)}
    }


def extract_civil_comments(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = row.get("text", "").strip()
    tox = row.get("toxicity")
    if tox is None or not text:
        return None
    is_toxic = float(tox) >= 0.5
    gt = "toxic" if is_toxic else "non_toxic"
    context = f"Comment: \"{text}\"\nEvaluate the toxicity status of the comment."
    return {
        "context": context,
        "ground_truth": gt,
        "metadata": {"toxicity_score": float(tox)}
    }


def extract_aegis_safety(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = row.get("prompt", "").strip()
    label = row.get("prompt_label", "").strip()
    if not prompt or not label:
        return None
    if label.lower() not in ("safe", "unsafe"):
        raise ValueError(f"unknown Aegis prompt label: {label!r}")
    gt = label.lower()
    context = f"User Request: \"{prompt}\"\nAssess prompt safety against injection and policy violation guidelines."
    return {
        "context": context,
        "ground_truth": gt,
        "metadata": {"original_id": str(row.get("id", "")), "raw_label": label}
    }


def extract_multinli(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    premise = row.get("premise", "").strip()
    hypothesis = row.get("hypothesis", "").strip()
    label = row.get("label")
    if label is None or not premise or not hypothesis:
        return None
    # MultiNLI label mapping: 0: entailment, 1: neutral, 2: contradiction
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    if label not in mapping:
        return None
    context = f"Premise: {premise}\nHypothesis: {hypothesis}\nDetermine the relationship between the premise and hypothesis."
    return {
        "context": context,
        "ground_truth": mapping[label],
        "metadata": {"pair_id": str(row.get("pairID", "")), "prompt_id": str(row.get("promptID", ""))}
    }


def extract_pubmedqa(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question", "").strip()
    ctx_data = row.get("context", {})
    decision = row.get("final_decision", "").strip().lower()
    if not q or not decision:
        return None
    if isinstance(ctx_data, dict):
        contexts = ctx_data.get("contexts", [])
        ctx_str = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    else:
        ctx_str = str(ctx_data)
    if decision not in ("yes", "no", "maybe"):
        return None
    context = f"Abstract: {ctx_str[:1200]}\nQuestion: {q}\nDetermine the medical answer based on the abstract."
    return {
        "context": context,
        "ground_truth": decision,
        "metadata": {"pubid": str(row.get("pubid", ""))}
    }


def extract_summeval(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # mteb/summeval: text, machine_summaries (list of 16), consistency (list of 16 scores 1.0-5.0)
    summaries = row.get("machine_summaries", [])
    consistencies = row.get("consistency", [])
    src_text = row.get("text", "").strip()
    if not summaries or not consistencies or not src_text:
        return None
    # Select the first machine summary with valid score
    summary = summaries[0].strip()
    score = float(consistencies[0])
    rating = str(min(5, max(1, round(score))))
    context = f"Source Document: {src_text[:800]}\nCandidate Summary: {summary}\nRate summary factual consistency on a scale from 1 (unsupported) to 5 (fully supported)."
    return {
        "context": context,
        "ground_truth": rating,
        "metadata": {"original_id": str(row.get("id", "")), "raw_score": score}
    }


def extract_arc_challenge(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question", "").strip()
    key = row.get("answerKey", "").strip().upper()
    choices = row.get("choices", {})
    if not q or not key or not choices:
        return None
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    if not labels or not texts or len(labels) != len(texts) or key not in labels:
        return None
    formatted_choices = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
    context = f"Question: {q}\n{formatted_choices}\nSelect the single correct option."
    return {
        "context": context,
        "ground_truth": key,
        "metadata": {"question_id": str(row.get("id", ""))}
    }


MASSIVE_CANDIDATES = [
    "alarm", "audio", "calendar", "cooking", "datetime", "email",
    "general", "iot", "lists", "music", "news", "play",
    "qa", "recommendation", "social", "takeaway", "transport", "weather"
]

BENCHMARK_SPECS: Dict[str, DatasetSpec] = {
    "massive_en": DatasetSpec(
        name="massive_en",
        display_name="MASSIVE en-US (Intent Routing)",
        hf_repo="SetFit/amazon_massive_scenario_en-US",
        hf_config="default",
        hf_split="validation",
        candidates=MASSIVE_CANDIDATES,
        description="Intent domain routing across 18 household/agent scenarios (English)",
        extractor=extract_massive_en
    ),
    "massive_de": DatasetSpec(
        name="massive_de",
        display_name="MASSIVE de-DE (Multilingual Intent)",
        hf_repo="SetFit/amazon_massive_scenario_de-DE",
        hf_config="default",
        hf_split="validation",
        candidates=MASSIVE_CANDIDATES,
        description="Multilingual intent domain routing across 18 scenarios (German)",
        extractor=extract_massive_de
    ),
    "vitaminc": DatasetSpec(
        name="vitaminc",
        display_name="VitaminC (Fact Verification)",
        hf_repo="tals/vitaminc",
        hf_config="default",
        hf_split="test",
        candidates=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        description="Contrastive fact checking with subtle factual revisions",
        extractor=extract_vitaminc
    ),
    "boolq": DatasetSpec(
        name="boolq",
        display_name="BoolQ (Passage Answerability)",
        hf_repo="google/boolq",
        hf_config="default",
        hf_split="validation",
        candidates=["yes", "no"],
        description="Binary Yes/No question answerability over Wikipedia passages",
        extractor=extract_boolq
    ),
    "squad2": DatasetSpec(
        name="squad2",
        display_name="SQuAD 2.0 (Answerability Decision)",
        hf_repo="rajpurkar/squad_v2",
        hf_config="squad_v2",
        hf_split="validation",
        candidates=["answerable", "unanswerable"],
        description="Adversarial unanswerable question detection in reading comprehension",
        extractor=extract_squad2
    ),
    "paws": DatasetSpec(
        name="paws",
        display_name="PAWS (Adversarial Paraphrase)",
        hf_repo="google-research-datasets/paws",
        hf_config="labeled_final",
        hf_split="validation",
        candidates=["paraphrase", "not_paraphrase"],
        description="Adversarial paraphrase identification with high lexical overlap",
        extractor=None
    ),
    "civil_comments": DatasetSpec(
        name="civil_comments",
        display_name="Civil Comments (Toxicity Moderation)",
        hf_repo="google/civil_comments",
        hf_config="default",
        hf_split="test",
        candidates=["toxic", "non_toxic"],
        description="Online comment toxicity classification and safety filtering",
        extractor=extract_civil_comments
    ),
    "aegis_safety": DatasetSpec(
        name="aegis_safety",
        display_name="Aegis 2.0 (Prompt Guardrail)",
        hf_repo="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        hf_config="default",
        hf_split="test",
        candidates=["safe", "unsafe"],
        description="Content safety and jailbreak / prompt injection defense",
        extractor=extract_aegis_safety
    ),
    "multinli": DatasetSpec(
        name="multinli",
        display_name="MultiNLI (Natural Language Inference)",
        hf_repo="nyu-mll/multi_nli",
        hf_config="default",
        hf_split="validation_matched",
        candidates=["entailment", "neutral", "contradiction"],
        description="Multi-genre natural language entailment / contradiction inference",
        extractor=extract_multinli
    ),
    "pubmedqa": DatasetSpec(
        name="pubmedqa",
        display_name="PubMedQA (Biomedical Decision)",
        hf_repo="qiaojin/PubMedQA",
        hf_config="pqa_labeled",
        hf_split="train",
        candidates=["yes", "no", "maybe"],
        description="Biomedical literature question answering and clinical reasoning",
        extractor=extract_pubmedqa
    ),
    "summeval": DatasetSpec(
        name="summeval",
        display_name="SummEval (Summary Quality 1-5)",
        hf_repo="mteb/summeval",
        hf_config="default",
        hf_split="test",
        candidates=["1", "2", "3", "4", "5"],
        description="Multi-aspect summary factual consistency Likert grading",
        extractor=extract_summeval
    ),
    "arc_challenge": DatasetSpec(
        name="arc_challenge",
        display_name="ARC-Challenge (Science QA)",
        hf_repo="allenai/ai2_arc",
        hf_config="ARC-Challenge",
        hf_split="test",
        candidates=["A", "B", "C", "D"],
        description="Challenging grade-school science questions requiring reasoning",
        extractor=extract_arc_challenge
    ),
    "gsm8k": DatasetSpec(
        name="gsm8k",
        display_name="GSM8K (Math Multi-Step Reasoning)",
        hf_repo="openai/gsm8k",
        hf_config="main",
        hf_split="test",
        candidates=["A", "B", "C", "D"],
        description="Multi-step elementary mathematical word problem decision",
        extractor=None
    ),
}


class DatasetFetcher:
    """Manages downloading, extracting, and standardizing real benchmark datasets."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_DATA_DIR,
        raw_cache_dir: Optional[Path] = None,
        timeout_seconds: int = 15,
        user_agent: str = "Gen-Zero-Benchmark-Fetcher/1.0"
    ):
        self.output_dir = Path(output_dir)
        self.raw_cache_dir = Path(raw_cache_dir) if raw_cache_dir else get_default_external_cache_dir()
        self.timeout = timeout_seconds
        self.user_agent = user_agent
        calibration_path = DEFAULT_DATA_DIR / "calibration_clean_16.jsonl"
        if not calibration_path.exists():
            raise RuntimeError(f"calibration isolation source missing: {calibration_path}")
        self.calibration_contexts = {
            normalized_context(json.loads(line)["context"])
            for line in calibration_path.read_text(encoding="utf-8").splitlines() if line
        }
        
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_parquet_download_url(self, repo: str, config: str, split: str) -> str:
        """Resolves direct parquet file URL from Hugging Face API. Raises RuntimeError on any failure."""
        api_url = f"https://huggingface.co/api/datasets/{repo}/parquet"
        req = urllib.request.Request(api_url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as err:
            raise RuntimeError(f"Hugging Face parquet index request failed for {repo}: {err!r}") from err
        if config in data and split in data[config]:
            return data[config][split][0]
        raise RuntimeError(f"Hugging Face parquet index for {repo} has no config={config!r} split={split!r}")

    def _download_parquet(self, url: str, local_path: Path) -> None:
        """Downloads a remote parquet file to the cache atomically. Raises RuntimeError on failure."""
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        part_path = local_path.with_name(local_path.name + ".part")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout * 2) as resp:
                content = resp.read()
            with open(part_path, "wb") as f:
                f.write(content)
            os.replace(part_path, local_path)
        except Exception as err:
            part_path.unlink(missing_ok=True)
            raise RuntimeError(f"parquet download failed from {url}: {err!r}") from err

    def fetch_task_samples(
        self,
        task_name: str,
        max_samples: int = 100,
        source: str = "auto"
    ) -> List[Dict[str, Any]]:
        """Fetches and standardizes samples for a specific benchmark task."""
        if task_name not in BENCHMARK_SPECS:
            raise ValueError(f"Unknown task: {task_name}. Supported tasks: {list(BENCHMARK_SPECS.keys())}")

        if max_samples <= 0:
            raise ValueError("sample limit must be positive")
        if source not in ("auto", "huggingface"):
            raise ValueError(f"Unsupported source {source!r}; only real Hugging Face data is allowed (auto, huggingface)")
        if not HAS_PYARROW:
            raise RuntimeError(f"{task_name}: pyarrow is required to read real datasets; refusing to substitute fixtures")

        spec = BENCHMARK_SPECS[task_name]
        samples: List[Dict[str, Any]] = []

        # These two tasks have separately validated stratification/scoring protocols.
        # Their committed rows retain the numeric gold and overlap-bucket metadata.
        if task_name in ("gsm8k", "paws"):
            path = DEFAULT_DATA_DIR / f"{task_name}.jsonl"
            if not path.exists():
                raise RuntimeError(f"{task_name}: validated source file missing: {path}")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            if len(rows) < max_samples:
                raise RuntimeError(f"{task_name}: requested {max_samples}, validated source has only {len(rows)}")
            if max_samples == len(rows):
                return rows
            groups = {label: [] for label in spec.candidates}
            for row in rows:
                if row["ground_truth"] not in groups:
                    raise RuntimeError(f"{task_name}: invalid validated source label")
                groups[row["ground_truth"]].append(row)
            selected = []
            for i in range(max_samples):
                label = spec.candidates[i % len(spec.candidates)]
                index = i // len(spec.candidates)
                if index >= len(groups[label]):
                    raise RuntimeError(f"{task_name}: insufficient validated rows for balanced {max_samples}")
                selected.append(groups[label][index])
            return selected

        if spec.extractor is None:
            raise RuntimeError(f"{task_name}: validated builder unavailable")

        cache_file = self.raw_cache_dir / f"{spec.name}_{spec.hf_config}_{spec.hf_split}.parquet"
        if not (cache_file.exists() and cache_file.stat().st_size > 0):
            p_url = self._get_parquet_download_url(spec.hf_repo, spec.hf_config, spec.hf_split)
            self._download_parquet(p_url, cache_file)

        try:
            tbl = pq.read_table(cache_file)
            num_rows = len(tbl)
            col_names = tbl.column_names

            buckets = {candidate: [] for candidate in spec.candidates}
            needed = {candidate: max_samples // len(spec.candidates) + (i < max_samples % len(spec.candidates)) for i, candidate in enumerate(spec.candidates)}
            seen_contexts = set()
            for i in range(num_rows):
                row_dict = {c: tbl[c][i].as_py() for c in col_names}
                extracted_rows = []
                if task_name == "summeval":
                    # Each source document has multiple distinct rated summaries.
                    summaries = row_dict.get("machine_summaries") or []
                    scores = row_dict.get("consistency") or []
                    if len(summaries) != len(scores):
                        raise ValueError(f"summeval row {i}: summary/score length mismatch")
                    for j in range(len(summaries)):
                        variant = dict(row_dict)
                        variant["machine_summaries"] = [summaries[j]]
                        variant["consistency"] = [scores[j]]
                        item = spec.extractor(variant)
                        if item:
                            item["metadata"]["summary_index"] = j
                            extracted_rows.append(item)
                else:
                    item = spec.extractor(row_dict)
                    if item:
                        extracted_rows.append(item)
                for extracted in extracted_rows:
                    label = extracted["ground_truth"]
                    context = extracted["context"]
                    normalized = normalized_context(context)
                    if label not in buckets or normalized in seen_contexts or normalized in self.calibration_contexts:
                        continue
                    seen_contexts.add(normalized)
                    if len(buckets[label]) < needed[label]:
                        buckets[label].append(extracted)
                if all(len(buckets[label]) >= needed[label] for label in spec.candidates):
                    break
        except Exception as err:
            raise RuntimeError(f"{task_name}: parsing {cache_file} failed: {err!r}") from err

        # Round-robin labels with fixed candidate order. This yields a maximum
        # label-count difference of one, or fails if a class is exhausted.
        for index in range(max_samples):
            label = spec.candidates[index % len(spec.candidates)]
            bucket_index = index // len(spec.candidates)
            if bucket_index >= len(buckets[label]):
                counts = {key: len(value) for key, value in buckets.items()}
                raise RuntimeError(f"{task_name}: insufficient real rows for balanced {max_samples}: {counts}")
            extracted = buckets[label][bucket_index]
            samples.append({
                "id": f"{task_name}-{index + 1:04d}", "task": task_name,
                "context": extracted["context"], "candidates": spec.candidates,
                "ground_truth": label,
                "metadata": {**extracted.get("metadata", {}), "source": "huggingface",
                             "repo": spec.hf_repo, "split": spec.hf_split},
            })

        return samples

    def generate_all(
        self,
        samples_per_task: int = 100,
        source: str = "auto",
        task_limits: Optional[Dict[str, int]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Generates standardized benchmark fixtures across all 13 benchmark dimensions."""
        print(f"[*] Gen-Zero Benchmark Dataset Ingestion Pipeline")
        print(f"    Target Directory: {self.output_dir}")
        print(f"    Raw Cache Directory: {self.raw_cache_dir}")
        print(f"    PyArrow Available: {HAS_PYARROW}")
        print(f"    Mode: {source} (Target: {samples_per_task} samples/task)\n")

        if samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        previous = json.loads((DEFAULT_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
        task_limits = task_limits or {}
        if set(task_limits) - set(BENCHMARK_SPECS) or any(v <= 0 for v in task_limits.values()):
            raise ValueError("task limits must name known tasks and be positive")
        all_bytes = bytearray()
        task_summaries: Dict[str, Any] = {}
        files: Dict[str, bytes] = {}
        all_ids = set()

        for task_name, spec in BENCHMARK_SPECS.items():
            print(f"  -> Processing: {spec.display_name} ...", end=" ", flush=True)
            limit = task_limits.get(task_name, {"paws": 400, "gsm8k": 200}.get(task_name, samples_per_task))
            samples = self.fetch_task_samples(task_name, max_samples=limit, source=source)
            ids = [sample["id"] for sample in samples]
            if len(ids) != len(set(ids)) or all_ids.intersection(ids):
                raise RuntimeError(f"{task_name}: duplicate sample IDs")
            all_ids.update(ids)
            for sample in samples:
                if set(sample) != {"id", "task", "context", "candidates", "ground_truth", "metadata"} or sample["task"] != task_name or sample["ground_truth"] not in spec.candidates or sample["candidates"] != spec.candidates:
                    raise RuntimeError(f"{task_name}: invalid six-key row {sample.get('id')}")
            
            # Save task-specific JSONL file
            payload = ("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)).encode("utf-8")
            files[f"{task_name}.jsonl"] = payload
            all_bytes.extend(payload)

            task_summaries[task_name] = {
                "display_name": spec.display_name,
                "hf_repo": spec.hf_repo,
                "samples_count": len(samples),
                "candidates_count": len(spec.candidates),
                "candidates": spec.candidates,
                "file": f"{task_name}.jsonl",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sample_ids": ids,
                "label_map": {label: i for i, label in enumerate(spec.candidates)},
                "label_counts": dict(Counter(sample["ground_truth"] for sample in samples)),
            }
            if task_name in ("paws", "gsm8k"):
                for key in ("hf_config", "hf_split", "hf_revision", "seed", "build", "scoring"):
                    if key in previous["tasks"][task_name]:
                        task_summaries[task_name][key] = previous["tasks"][task_name][key]
            print(f"Done ({len(samples)} samples)")

        # Save unified combined dataset
        master_file = self.output_dir / "all_benchmarks.jsonl"
        files[master_file.name] = bytes(all_bytes)

        # Save metadata manifest
        manifest = {
            "version": "1.2.0",
            "license": "Apache-2.0",
            "benchmark_dimensions": len(BENCHMARK_SPECS),
            "total_samples": len(all_ids),
            "schema": {
                "id": "str",
                "task": "str",
                "context": "str",
                "candidates": "List[str]",
                "ground_truth": "str",
                "metadata": "dict"
            },
            "tasks": task_summaries,
            "schema_version": 1,
            "schema_check": {"required_keys": ["id", "task", "context", "candidates", "ground_truth", "metadata"], "verifier": "benchmarks/datasets/verify_data.py"},
            "all_benchmarks": {"file": master_file.name, "samples_count": len(all_ids), "sha256": hashlib.sha256(all_bytes).hexdigest(), "note": "concatenation of the task files in manifest order; runners skip it"},
        }
        aux = DEFAULT_DATA_DIR / "aux" / "paws_minimal_pairs.jsonl"
        if aux.exists() and self.output_dir == DEFAULT_DATA_DIR:
            manifest["aux_files"] = {"aux/paws_minimal_pairs.jsonl": {"sha256": hashlib.sha256(aux.read_bytes()).hexdigest(), "samples_count": sum(1 for _ in aux.open(encoding="utf-8")), "note": "not read by run_remote_eval_v6.py"}}
        manifest_file = self.output_dir / "manifest.json"
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for name, payload in files.items():
                (self.output_dir / name).write_bytes(payload)
            manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(f"\n[✓] Ingestion Complete!")
        print(f"    Total Samples Generated: {len(all_ids)}")
        print(f"    Master JSONL: {master_file}")
        print(f"    Manifest: {manifest_file}")
        return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Gen-Zero Real Dataset Fetcher & Benchmark Fixture Generator"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Target output directory for standardized benchmark JSONL files"
    )
    parser.add_argument(
        "--raw-cache-dir",
        default=None,
        help="External storage path for raw heavy downloads (avoids polluting repo)"
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=100,
        help="Number of samples per task (default: 100; validated PAWS/GSM8K retain 400/200)"
    )
    parser.add_argument("--task-limit", action="append", default=[], metavar="TASK=N",
                        help="Override one task's sample count; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate in memory without writing")
    parser.add_argument("--verify", action="store_true", help="Verify existing output files against manifest")
    parser.add_argument(
        "--source",
        choices=["auto", "huggingface"],
        default="auto",
        help="Data acquisition mode: real Hugging Face data only. Any download or parse failure raises RuntimeError; no fixture fallback exists"
    )
    args = parser.parse_args()

    if args.verify:
        verify_generated(Path(args.output_dir))
        print("PASS manifest, task files, IDs, labels, and aggregate")
        return
    task_limits = {}
    for value in args.task_limit:
        name, separator, count = value.partition("=")
        if not separator or name in task_limits:
            parser.error(f"invalid or repeated --task-limit: {value}")
        try:
            task_limits[name] = int(count)
        except ValueError:
            parser.error(f"invalid --task-limit count: {value}")

    fetcher = DatasetFetcher(
        output_dir=Path(args.output_dir),
        raw_cache_dir=Path(args.raw_cache_dir) if args.raw_cache_dir else None,
    )
    fetcher.generate_all(
        samples_per_task=args.samples_per_task,
        source=args.source,
        task_limits=task_limits,
        dry_run=args.dry_run,
    )


def verify_generated(output_dir: Path) -> None:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["tasks"]) != set(BENCHMARK_SPECS):
        raise RuntimeError("manifest task set mismatch")
    combined = bytearray()
    ids = set()
    total = 0
    for task, entry in manifest["tasks"].items():
        raw = (output_dir / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"{task}: SHA256 mismatch")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if len(rows) != entry["samples_count"] or [row["id"] for row in rows] != entry["sample_ids"]:
            raise RuntimeError(f"{task}: count or IDs mismatch")
        counts = Counter()
        for row in rows:
            if set(row) != {"id", "task", "context", "candidates", "ground_truth", "metadata"} or row["task"] != task or row["candidates"] != entry["candidates"] or row["ground_truth"] not in entry["candidates"] or row["id"] in ids:
                raise RuntimeError(f"{task}: schema, label, or duplicate ID failure")
            ids.add(row["id"])
            counts[row["ground_truth"]] += 1
        if dict(counts) != entry["label_counts"]:
            raise RuntimeError(f"{task}: label counts mismatch")
        combined.extend(raw)
        total += len(rows)
    actual = (output_dir / "all_benchmarks.jsonl").read_bytes()
    if actual != combined or hashlib.sha256(actual).hexdigest() != manifest["all_benchmarks"]["sha256"] or total != manifest["total_samples"] or total != manifest["all_benchmarks"]["samples_count"]:
        raise RuntimeError("aggregate or manifest total mismatch")


if __name__ == "__main__":
    main()
