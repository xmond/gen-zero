"""Train/test data for the 01.PNG 13-task grand challenge.

Test set: `benchmarks/data/full_13/*.jsonl`, the 3,880 records that produced
01.PNG (Bespoke Labs "Nimble vs Jev on 13 public benchmarks"), rebuilt by
`/ebs/tmp/benchmarks/scripts/build_full_13.py` from nimble's published id
manifests. Never used for training, model selection, or early stopping.

Train set: drawn from each dataset's OTHER public partition (train split, or
for single-split sources, the rows not in the test manifest), converted into
the exact same context string the test records use. The instruction line and
the candidate list are copied from the test file itself, so train and test
share one input format by construction.

Leakage gate (fail-closed, counted in the report):
  1. every test family (passage hash, premise, pubid, article, prompt hash,
     ...) is excluded from train;
  2. every train record whose normalized field text equals a test record's
     is dropped;
  3. after both, id, family and normalized-text overlap must be exactly zero.

Sampling is uniform at random (seed fixed) from the eligible pool, never
stratified by label: nimble's manifests state their selection never looked at
targets, so the natural label prior is the honest training prior.

Field splitting (`split_fields`) recovers the builder's `key: value` fields
from a test context and asserts the re-serialized string is byte-identical,
so the split is lossless, not a heuristic. Field names are the task's input
schema (published in the builder), never the label.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TEST_DIR = Path(os.environ.get("GC_TEST_DIR", str(REPO / "benchmarks" / "data" / "full_13")))
RAW = Path(os.environ.get("GC_RAW", "/ebs/tmp/benchmarks/_raw"))
SEED = 20260923

TASKS = ("massive_en", "massive_de", "multinli", "pubmedqa", "vitaminc", "boolq", "squad2",
         "paws", "civil_comments", "aegis_safety", "helpsteer2", "summeval_relevance",
         "summeval_consistency")

# Every feature file encoded before 2026-09-24 used 1000 train rows per task; the
# default keeps those caches valid. Raise it per task with GC_N_TRAIN_TASKS
# ("massive_de=3000,boolq=3000"), for all tasks with GC_N_TRAIN, or unblock a task's
# whole eligible pool with the literal "max" (GC_N_TRAIN=max or
# GC_N_TRAIN_TASKS="massive_de=max"). "0" stays a rejected input, not a max alias:
# tests already assert it fails loudly, and overloading a magic zero into "unlimited"
# would flip a loud config error into a silent multi-hour run. Sampling walks one
# fixed permutation, so a larger cap is a strict superset of a smaller one.
N_TRAIN_DEFAULT = 1000
MAX_TOKEN = "max"


def _positive_int(text: str, what: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{what} must be a positive integer, got {text!r}") from None
    if value <= 0:
        raise ValueError(f"{what} must be a positive integer, got {text!r}")
    return value


def _cap(text: str, what: str) -> Optional[int]:
    """A train-row cap: the literal "max" (unlimited, i.e. None) or a positive integer."""
    if text.strip().lower() == MAX_TOKEN:
        return None
    return _positive_int(text, what)


def n_train_for(task: str, env: Optional[Dict[str, str]] = None) -> Optional[int]:
    """Train-row cap for `task`: GC_N_TRAIN_TASKS entry, else GC_N_TRAIN, else 1000.

    None means unlimited (GC_N_TRAIN[_TASKS]=max): take every row `build_train`'s
    leakage/duplicate filters leave in the eligible pool.
    """
    env = os.environ if env is None else env
    per_task: Dict[str, Optional[int]] = {}
    for item in env.get("GC_N_TRAIN_TASKS", "").split(","):
        if not item.strip():
            continue
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or name not in TASKS:
            raise ValueError(f"GC_N_TRAIN_TASKS entry {item!r} is not <task>=<n> with a known task")
        per_task[name] = _cap(value.strip(), f"GC_N_TRAIN_TASKS[{name}]")
    if task in per_task:
        return per_task[task]
    return _cap(env.get("GC_N_TRAIN", str(N_TRAIN_DEFAULT)), "GC_N_TRAIN")


def train_spec(task: str) -> Tuple[Optional[int], str]:
    """(n_train_max, pubmedqa_extra) that a fresh build_train(task, ...) would use now."""
    return n_train_for(task), pubmedqa_extra() if task == "pubmedqa" else ""


def cached_train_spec(path: Path) -> Tuple[Optional[int], str]:
    """(n_train_max, pubmedqa_extra) a feature .npz was built with.

    Caches written before GC_N_TRAIN existed carry no n_train_max key; they always
    used 1000. An explicit null (json for Python None) means that cache was built
    unlimited.
    """
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z["info_json"]))
    raw = info.get("n_train_max", N_TRAIN_DEFAULT)
    return (None if raw is None else int(raw)), info.get("pubmedqa_extra", "")


# ------------------------------------------------------------- token budgets

def parse_task_max_tok(spec: str, head_tok: int, known_tasks: Tuple[str, ...] = TASKS) -> Dict[str, int]:
    """"boolq=1536,squad2=1024" -> {"boolq": 1536, "squad2": 1024}. Empty spec -> {}.

    Every budget must exceed `head_tok`: `truncate_ids` keeps head_tok + (max_tok - head_tok)
    tokens, and a zero tail would silently keep the WHOLE sequence (ids[-0:]).
    Unknown tasks, duplicates and non-integers raise, like GC_N_TRAIN_TASKS.
    """
    out: Dict[str, int] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or name not in known_tasks:
            raise ValueError(f"--task-max-tok entry {item!r} is not <task>=<tokens> with a known task")
        if name in out:
            raise ValueError(f"--task-max-tok names {name!r} twice")
        out[name] = _positive_int(value.strip(), f"--task-max-tok[{name}]")
        if out[name] <= head_tok:
            raise ValueError(f"--task-max-tok[{name}]={out[name]} must exceed head_tok={head_tok}")
    return out


def truncate_ids(ids: List[int], max_tok: int, head_tok: int) -> List[int]:
    """Keep the first `head_tok` and the last `max_tok - head_tok` ids when over `max_tok`."""
    if not 0 <= head_tok < max_tok:
        raise ValueError(f"need 0 <= head_tok < max_tok, got head_tok={head_tok}, max_tok={max_tok}")
    if len(ids) <= max_tok:
        return ids
    return ids[:head_tok] + ids[-(max_tok - head_tok):]


def cached_token_budget(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """(max_tok, head_tok) a feature .npz recorded in its info_json; None when absent."""
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z["info_json"]))
    return info.get("max_tok"), info.get("head_tok")

# Input schema of each task: the builder's state keys, in serialization order.
# None means the state is a bare string (civil_comments).
FIELDS: Dict[str, Optional[Tuple[str, ...]]] = {
    "massive_en": ("utterance", "locale"),
    "massive_de": ("utterance", "locale"),
    "multinli": ("premise", "hypothesis"),
    "pubmedqa": ("question", "abstract_context"),
    "vitaminc": ("evidence", "claim"),
    "boolq": ("passage", "question"),
    "squad2": ("paragraph", "question"),
    "paws": ("sentence_1", "sentence_2"),
    "civil_comments": None,
    "aegis_safety": ("user_message",),
    "helpsteer2": ("prompt", "response"),
    "summeval_relevance": ("article", "summary"),
    "summeval_consistency": ("article", "summary"),
}
# Tasks the user named for the paired cross-difference path, with the (A, B) fields.
PAIR_FIELDS = {
    "multinli": ("premise", "hypothesis"),
    "paws": ("sentence_1", "sentence_2"),
    "vitaminc": ("evidence", "claim"),
}


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def serialize(instruction: str, fields) -> str:
    """Byte-identical to build_full_13.state_to_context."""
    if isinstance(fields, str):
        body = fields
    else:
        body = "\n".join(f"{k}: {v}" for k, v in fields.items())
    return f"{instruction}\n\n{body}"


def split_fields(task: str, context: str) -> Tuple[str, object]:
    instruction, body = context.split("\n\n", 1)
    keys = FIELDS[task]
    if keys is None:
        fields: object = body
    else:
        if not body.startswith(f"{keys[0]}: "):
            raise ValueError(f"{task}: body does not start with field {keys[0]!r}")
        fields, rest = {}, body[len(keys[0]) + 2:]
        for cur, nxt in zip(keys, keys[1:]):
            marker = f"\n{nxt}: "
            pos = rest.rfind(marker) if nxt == keys[-1] else rest.find(marker)
            if pos < 0:
                raise ValueError(f"{task}: field {nxt!r} missing")
            fields[cur], rest = rest[:pos], rest[pos + len(marker):]
        fields[keys[-1]] = rest
    if serialize(instruction, fields) != context:
        raise ValueError(f"{task}: field split is not lossless")
    return instruction, fields


def field_key(fields) -> str:
    if isinstance(fields, str):
        return norm(fields)
    return "\x1f".join(norm(str(v)) for v in fields.values())


# --------------------------------------------------------------------- test

def load_test(task: str) -> List[dict]:
    path = TEST_DIR / f"{task}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    for r in rows:
        r["instruction"], r["fields"] = split_fields(task, r["context"])
        if r["ground_truth"] not in r["candidates"]:
            raise ValueError(f"{task}/{r['id']}: ground_truth not in candidates")
    return rows


def test_file_digest(task: str) -> Dict[str, object]:
    path = TEST_DIR / f"{task}.jsonl"
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "lines": data.count(b"\n")}


def test_family(task: str, r: dict) -> str:
    """Family key recomputed from the test record's own fields (never its label)."""
    f = r["fields"]
    if task == "multinli":
        return "premise:" + norm(f["premise"])            # builder: promptID = one premise
    if task == "vitaminc":
        return "evidence:" + norm(f["evidence"])          # builder: case_id = one revision
    if task == "boolq":
        return sha16(f["passage"])
    if task == "squad2":
        return sha16(f["paragraph"])
    if task == "helpsteer2":
        return "helpsteer2-" + sha16(f["prompt"])
    if task.startswith("summeval"):
        return "article:" + norm(f["article"])
    return r["id"]                                         # massive id, pubid, per-row sources


# -------------------------------------------------------------------- train

def _hf(repo: str, config: Optional[str] = None, split: str = "train"):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import Dataset, concatenate_datasets, load_dataset
    arrow_dir = os.environ.get("GC_ARROW_DIR")
    if arrow_dir:
        if repo == "qiaojin/PubMedQA" and config != "pqa_labeled":
            # Arrow files are named by repo and split only; two PubMedQA configs would collide.
            raise ValueError(f"GC_ARROW_DIR cannot tell PubMedQA/{config} from pqa_labeled; unset it")
        prefix = {
            "qiaojin/PubMedQA": "pub_med_qa",
            "nvidia/HelpSteer2": "help_steer2",
            "nvidia/Aegis-AI-Content-Safety-Dataset-2.0": "aegis-ai-content-safety-dataset-2.0",
            "rajpurkar/squad_v2": "squad_v2",
        }.get(repo, repo.rsplit("/", 1)[-1].lower())
        matches = sorted(Path(arrow_dir).glob(f"{prefix}-{split}*.arrow"))
        if not matches:
            raise FileNotFoundError(f"no {repo}/{split} Arrow files in {arrow_dir}")
        parts = [Dataset.from_file(str(path)) for path in matches]
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)


def _raw_massive(locale: str) -> List[dict]:
    with tarfile.open(RAW / "amazon-massive-dataset-1.1.tar.gz", "r:gz") as archive:
        member = [m for m in archive.getmembers() if m.name.endswith(f"/data/{locale}.jsonl")][0]
        lines = archive.extractfile(member).read().decode().split("\n")
    rows = []
    for line in lines:
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("partition") != "train" or raw.get("locale") != locale:
            continue
        utt = raw.get("utt")
        if isinstance(utt, str) and utt.strip():
            rows.append({"id": f"massive-{raw['id']}", "family": f"massive-{raw['id']}",
                         "fields": {"utterance": utt, "locale": locale}, "label": raw["scenario"]})
    return rows


def _raw_multinli() -> List[dict]:
    opts = ("entailment", "neutral", "contradiction")
    rows = []
    with zipfile.ZipFile(RAW / "multinli_1.0.zip") as zf:
        with zf.open("multinli_1.0/multinli_1.0_train.jsonl") as fh:
            for line in fh:
                raw = json.loads(line)
                if raw.get("gold_label") not in opts:
                    continue
                prem, hyp = raw["sentence1"].strip(), raw["sentence2"].strip()
                if prem and hyp:
                    rows.append({"id": "multinli-" + raw["pairID"], "family": "premise:" + norm(prem),
                                 "fields": {"premise": prem, "hypothesis": hyp}, "label": raw["gold_label"]})
    return rows


def _summeval_rows(dim: str) -> List[dict]:
    rows = []
    for raw in _hf("mteb/summeval", split="test"):
        for idx, (summ, val) in enumerate(zip(raw["machine_summaries"], raw[dim])):
            if isinstance(summ, str) and summ.strip():
                rows.append({"id": f"summeval-{dim}-{raw['id']}-{idx}", "family": "article:" + norm(raw["text"]),
                             "fields": {"article": raw["text"], "summary": summ},
                             "label_index": min(4, max(0, math.floor(val + 0.5) - 1))})
    return rows


def _convert_hf(task: str, raw: dict, i: int) -> Optional[dict]:
    if task == "pubmedqa":
        dec = (raw.get("final_decision") or "").strip().lower()
        ctx = [c for c in (raw.get("context") or {}).get("contexts", []) if isinstance(c, str) and c.strip()]
        if dec in ("yes", "no", "maybe") and ctx and (raw.get("question") or "").strip():
            # The family is the pubid, exactly as test_family() keys a test record, so
            # an artificial row about a test abstract is dropped by the family gate.
            fam = f"pubmedqa-{raw['pubid']}"
            rid = f"pubmedqa-art-{raw['pubid']}" if raw.get("_subset") == "artificial" else fam
            return {"id": rid, "family": fam, "label": dec,
                    "fields": {"question": raw["question"], "abstract_context": " ".join(ctx)}}
    elif task == "vitaminc":
        if raw["label"] in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO") and raw["claim"].strip() \
                and raw["evidence"].strip():
            return {"id": "vitaminc-" + raw["unique_id"], "family": "evidence:" + norm(raw["evidence"]),
                    "fields": {"evidence": raw["evidence"], "claim": raw["claim"]}, "label": raw["label"]}
    elif task == "boolq":
        return {"id": "boolq-" + sha16(raw["question"] + "\n" + raw["passage"]), "family": sha16(raw["passage"]),
                "label": "true" if raw["answer"] else "false",
                "fields": {"passage": raw["passage"], "question": raw["question"]}}
    elif task == "squad2":
        texts = [t for t in raw["answers"]["text"] if t.strip()]
        return {"id": "squad2-" + raw["id"], "family": sha16(raw["context"]),
                "label": "true" if texts else "false",
                "fields": {"paragraph": raw["context"], "question": raw["question"]}}
    elif task == "paws":
        rid = f"paws-train-{raw['id']}"
        return {"id": rid, "family": rid, "label": "true" if raw["label"] == 1 else "false",
                "fields": {"sentence_1": raw["sentence1"], "sentence_2": raw["sentence2"]}}
    elif task == "civil_comments":
        text, tox = (raw.get("text") or "").strip(), raw.get("toxicity")
        if text and isinstance(tox, (int, float)) and 0 <= tox <= 1:
            rid = f"civil_comments-train-{i}"
            return {"id": rid, "family": rid, "fields": text, "label": "true" if tox >= 0.5 else "false"}
    elif task == "aegis_safety":
        if raw.get("prompt_label_source") != "human" or raw.get("reconstruction_id_if_redacted") is not None:
            return None
        prompt = (raw.get("prompt") or "").strip()
        if prompt and raw.get("prompt_label") in ("safe", "unsafe"):
            rid = "aegis2-" + raw["id"]
            return {"id": rid, "family": rid, "fields": {"user_message": prompt},
                    "label": "true" if raw["prompt_label"] == "unsafe" else "false"}
    elif task == "helpsteer2":
        h = raw.get("helpfulness")
        if isinstance(h, int) and 0 <= h <= 4 and raw["prompt"].strip() and raw["response"].strip():
            return {"id": f"helpsteer2-train-{i}", "family": "helpsteer2-" + sha16(raw["prompt"]),
                    "fields": {"prompt": raw["prompt"], "response": raw["response"]}, "label_index": h}
    return None


_HF_SOURCE = {
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_labeled"),
    "vitaminc": ("tals/vitaminc", None),
    "boolq": ("google/boolq", None),
    "squad2": ("rajpurkar/squad_v2", None),
    "paws": ("google-research-datasets/paws", "labeled_final"),
    "civil_comments": ("google/civil_comments", None),
    "aegis_safety": ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None),
    "helpsteer2": ("nvidia/HelpSteer2", None),
}


def _iter_train(task: str, rng) -> Iterator[dict]:
    if task.startswith("massive") or task == "multinli" or task.startswith("summeval"):
        if task.startswith("massive"):
            rows = _raw_massive("en-US" if task == "massive_en" else "de-DE")
        elif task == "multinli":
            rows = _raw_multinli()
        else:
            rows = _summeval_rows(task.split("_", 1)[1])
        for i in rng.permutation(len(rows)):
            yield rows[int(i)]
        return
    ds = _hf(*_HF_SOURCE[task])
    for i in rng.permutation(len(ds)):
        rec = _convert_hf(task, ds[int(i)], int(i))
        if rec is not None:
            yield rec
    if task == "pubmedqa" and pubmedqa_extra() == "artificial":
        # pqa_labeled has only 750 rows outside the test set. pqa_artificial (211k,
        # heuristic yes/no labels, NO "maybe", ~93% yes) only fills rows after every
        # labeled row is taken, so the human-labeled prior is never sampled away.
        art = _hf("qiaojin/PubMedQA", "pqa_artificial")
        for i in rng.permutation(len(art)):
            rec = _convert_hf(task, dict(art[int(i)], _subset="artificial"), int(i))
            if rec is not None:
                yield rec


def pubmedqa_extra() -> str:
    value = os.environ.get("GC_PUBMEDQA_EXTRA", "").strip()
    if value not in ("", "artificial"):
        raise ValueError(f"GC_PUBMEDQA_EXTRA must be empty or 'artificial', got {value!r}")
    return value


_FROM_ENV = object()  # build_train(task, rows) with no n_max: look up n_train_for(task).


def build_train(task: str, test_rows: List[dict], n_max=_FROM_ENV) -> Tuple[List[dict], dict]:
    """n_max: row cap, or None for unlimited (every eligible row). Omit it to read
    the cap from GC_N_TRAIN[_TASKS] via n_train_for(task)."""
    if n_max is _FROM_ENV:
        n_max = n_train_for(task)
    rng = np.random.default_rng([SEED, TASKS.index(task)])
    instruction = test_rows[0]["instruction"]
    candidates = test_rows[0]["candidates"]
    if any(r["instruction"] != instruction or r["candidates"] != candidates for r in test_rows):
        raise ValueError(f"{task}: test records disagree on instruction/candidates")
    test_ids = {r["id"] for r in test_rows}
    test_fams = {test_family(task, r) for r in test_rows}
    test_keys = {field_key(r["fields"]) for r in test_rows}
    stats: Counter = Counter()
    out, seen = [], set()
    for rec in _iter_train(task, rng):
        stats["scanned"] += 1
        if rec["id"] in test_ids:
            stats["dropped_test_id"] += 1
            continue
        if rec["family"] in test_fams:
            stats["dropped_test_family"] += 1
            continue
        key = field_key(rec["fields"])
        if key in test_keys:
            stats["dropped_test_text"] += 1
            continue
        if key in seen:
            stats["dropped_train_duplicate"] += 1
            continue
        seen.add(key)
        label = candidates[rec["label_index"]] if "label_index" in rec else rec["label"]
        if label not in candidates:
            raise ValueError(f"{task}: train label {label!r} not a test candidate")
        out.append({"task": task, "id": rec["id"], "family": rec["family"], "fields": rec["fields"],
                    "instruction": instruction, "context": serialize(instruction, rec["fields"]),
                    "candidates": candidates, "ground_truth": label})
        if n_max is not None and len(out) >= n_max:
            break
    # Final gate: must be exactly zero after the filters above.
    id_overlap = len({r["id"] for r in out} & test_ids)
    text_overlap = len({field_key(r["fields"]) for r in out} & test_keys)
    fam_overlap = len({r["family"] for r in out} & test_fams)
    if id_overlap or text_overlap or fam_overlap:
        raise AssertionError(f"{task}: leakage id={id_overlap} text={text_overlap} family={fam_overlap}")
    # n_max=None (unlimited) always exhausts the pool by construction.
    pool_exhausted = True if n_max is None else len(out) < n_max
    result = dict(stats)
    result.update({"kept": len(out), "n_max": n_max, "pool_exhausted": pool_exhausted,
                   "id_overlap": id_overlap, "text_overlap": text_overlap,
                   "family_overlap": fam_overlap,
                   "label_counts": dict(Counter(r["ground_truth"][:40] for r in out))})
    return out, result
