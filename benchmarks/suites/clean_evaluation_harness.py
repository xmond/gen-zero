"""A leakage-resistant, hash-bound evaluation contract.

This module deliberately has no model-specific fallback.  A caller supplies a
predictor which receives only ``Sample.input`` from the test split.  Labels are
used solely by the harness for scoring and by ``fit_prototypes`` on calibration
data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


class ContractViolation(ValueError):
    """Raised when an evaluation invariant cannot be established."""


def _deep_freeze(val: Any) -> Any:
    if isinstance(val, (str, int, float, bool, bytes, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return tuple(_deep_freeze(x) for x in val)
    if isinstance(val, (dict, Mapping)):
        return MappingProxyType({k: _deep_freeze(v) for k, v in val.items()})
    return val


@dataclass(frozen=True)
class Sample:
    sample_id: str
    input: Any
    label: Any


@dataclass(frozen=True)
class LoadedSplit:
    samples: tuple[Sample, ...]
    sha256: str
    source: str
    raw_bytes: bytes = b""
    # Order- and formatting-insensitive digest of the (input, label) records.
    content_sha256: str = ""


@dataclass(frozen=True)
class PrototypeArtifact:
    """Frozen calibration artifact; it contains no test data or test labels."""

    prototypes: Mapping[Any, Any]
    calibration_sha256: str
    calibration_content_sha256: str = ""
    calibration_input_hashes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    correct: int
    errors: int
    accuracy: float
    coverage: float
    test_sha256: str
    calibration_sha256: str


def _input_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=dict).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_digest(records: list) -> str:
    """Hash the multiset of (input, label) pairs, ignoring ids, order and whitespace.

    A re-indented, re-ordered or re-id'd copy of the calibration file has a new
    byte-level sha256 but the same content, so it is still the same data.
    """
    canonical = sorted(
        json.dumps([r["input"], r["label"]], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for r in records
    )
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def _parse_records(raw: bytes, source: str) -> tuple[tuple[Sample, ...], str]:
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"invalid JSON in {source}") from exc
    if not isinstance(records, list):
        raise ContractViolation("split must be a JSON list")
    result = []
    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not {"id", "input", "label"} <= record.keys():
            raise ContractViolation(f"record {index} must contain id, input, and label")
        if str(record["id"]) in seen_ids:
            raise ContractViolation("duplicate sample id")
        seen_ids.add(str(record["id"]))
        result.append(Sample(str(record["id"]), _deep_freeze(record["input"]), _deep_freeze(record["label"])))
    return tuple(result), _content_digest(records)


def load_split(path: str | Path, *, expected_sha256: str | None = None) -> LoadedSplit:
    """Read one immutable byte snapshot, hash that snapshot, then parse it."""
    source = str(path)
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ContractViolation(f"cannot atomically load {source}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ContractViolation(
            f"sha256 mismatch for {source}: expected {expected_sha256}, got {digest}"
        )
    samples, content_digest = _parse_records(raw, source)
    return LoadedSplit(samples, digest, source, content_sha256=content_digest)


def fit_prototypes(calibration: LoadedSplit, *, embed: Callable[[Any], Any]) -> PrototypeArtifact:
    """Fit class means using calibration labels only, then freeze the result.

    ``embed`` is intentionally called only with calibration inputs.  The
    resulting mapping is detached from mutable caller mappings where possible.
    """
    if not calibration.samples:
        raise ContractViolation("calibration split must not be empty")
    grouped: dict[Any, list[Any]] = {}
    for sample in calibration.samples:
        value = embed(sample.input)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ContractViolation("embed must return a numeric sequence")
        grouped.setdefault(sample.label, []).append(tuple(float(x) for x in value))
    prototypes = {}
    for label, vectors in grouped.items():
        width = len(vectors[0])
        if not width or any(len(vector) != width for vector in vectors):
            raise ContractViolation("calibration embeddings must have equal non-zero width")
        prototypes[label] = tuple(sum(vector[i] for vector in vectors) / len(vectors) for i in range(width))
    return PrototypeArtifact(MappingProxyType(prototypes), calibration.sha256,
                             calibration.content_sha256,
                             frozenset(_input_hash(s.input) for s in calibration.samples))


def evaluate(
    test: LoadedSplit,
    calibration: PrototypeArtifact,
    *,
    predict: Callable[[Any], Any],
) -> EvaluationResult:
    """Score every test sample; exceptions and abstentions count as errors."""
    if not test.samples:
        raise ContractViolation("test split must not be empty")
    if test.sha256 == calibration.calibration_sha256:
        raise ContractViolation("Data contamination: test split and calibration split are identical")
    if test.content_sha256 and test.content_sha256 == calibration.calibration_content_sha256:
        raise ContractViolation(
            "Data contamination: test split and calibration split have identical content"
        )
    if any(_input_hash(s.input) in calibration.calibration_input_hashes for s in test.samples):
        raise ContractViolation("Data contamination: test inputs overlap calibration inputs")
    correct = 0
    answered = 0
    for sample in test.samples:
        try:
            # sample.input is already deep-frozen (tuples / MappingProxyType), so a
            # mutating predictor cannot alter the hash-bound snapshot; no copy needed
            # (and copy.deepcopy cannot pickle a MappingProxyType anyway).
            prediction = predict(sample.input)
            if prediction is not None:
                answered += 1
                if _deep_freeze(prediction) == sample.label:
                    correct += 1
        except Exception:  # model failure is an observed evaluation error
            pass
    total = len(test.samples)
    return EvaluationResult(total, correct, total - correct, correct / total, answered / total,
                            test.sha256, calibration.calibration_sha256)


def load_registered_dataset(data_dir: str | Path, task: str) -> LoadedSplit:
    """Load a manifest-bound benchmark task as a hash-bound evaluation split."""
    import sys
    datasets_dir = str(Path(__file__).resolve().parents[1] / "datasets")
    if datasets_dir not in sys.path:
        sys.path.insert(0, datasets_dir)
    from dataset_registry import DatasetError, load_registry

    data_dir = Path(data_dir)
    try:
        manifest, datasets = load_registry(data_dir)
        if task not in datasets:
            raise DatasetError(f"unregistered task: {task}")
        entry = next(value for group in manifest["task_groups"].values()
                     for name, value in group.items() if name == task)
    except DatasetError as exc:
        raise ContractViolation(str(exc)) from exc
    def _clean_input_meta(meta):
        # Fail-closed anti-leakage: strip raw answers and gold indices from input metadata
        # so predictors can never cheat by reading answers off sample.input
        leaked_keys = {"raw_answer", "gold_numeric", "gold", "option_kinds", "evidence_certificate", "reference"}
        return {k: v for k, v in meta.items() if k not in leaked_keys}

    samples = tuple(Sample(row["id"], _deep_freeze({
        "task": row["task"], "context": row["context"],
        "candidates": row["candidates"], "metadata": _clean_input_meta(row.get("metadata", {}))
    }), row["ground_truth"]) for row in datasets[task])
    return LoadedSplit(samples, entry["sha256"], str(data_dir / entry["file"]))
