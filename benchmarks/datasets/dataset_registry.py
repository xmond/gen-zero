"""Fail-closed manifest registry for benchmark evaluation data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

KEYS = {"id": str, "task": str, "context": str, "candidates": list,
        "ground_truth": str, "metadata": dict}


class DatasetError(ValueError):
    pass


def registered_tasks(manifest: dict, suite: str = "all") -> dict:
    groups = manifest.get("task_groups", {})
    if not isinstance(groups, dict) or "core_13" not in groups:
        raise DatasetError("manifest.task_groups.core_13 is required")
    if suite != "all" and suite not in groups:
        raise DatasetError(f"unknown suite: {suite}")
    selected = groups if suite == "all" else {suite: groups[suite]}
    tasks = {}
    for group, entries in groups.items():
        if not isinstance(entries, dict):
            raise DatasetError(f"task group {group} must be an object")
        for name, entry in entries.items():
            if name in tasks:
                raise DatasetError(f"duplicate task registration: {name}")
            tasks[name] = entry
    return tasks if suite == "all" else selected[suite]


def allowed_files(manifest: dict) -> set[str]:
    tasks = registered_tasks(manifest)
    names = [entry.get("file") for entry in tasks.values()]
    names += ["all_benchmarks.jsonl", *(manifest.get("aux_files") or {}).keys()]
    if any(not isinstance(n, str) or not n.endswith(".jsonl") or
           Path(n).is_absolute() or ".." in Path(n).parts for n in names):
        raise DatasetError("invalid registered JSONL path")
    if len(set(names)) != len(names):
        raise DatasetError("duplicate registered JSONL path")
    return set(names)


def check_file(data_dir: Path | str, task: str, entry: dict) -> list[dict]:
    data_dir = Path(data_dir)
    name = entry.get("file")
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise DatasetError(f"{task}: invalid file path")
    path = data_dir / name
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"{task}: cannot read {name}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry.get("sha256"):
        raise DatasetError(f"{name}: sha256 mismatch: expected {entry.get('sha256')}, got {digest}")
    try:
        lines = [x for x in raw.decode("utf-8").split("\n") if x]
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{name}: invalid UTF-8: {exc}") from exc
    if len(lines) != entry.get("samples_count"):
        raise DatasetError(f"{name}: count mismatch: expected {entry.get('samples_count')}, got {len(lines)}")
    rows = []
    ids = set()
    expected_ids = entry.get("sample_ids")
    for line_no, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{name}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict) or set(row) != set(KEYS) or any(
            not isinstance(row[k], typ) for k, typ in KEYS.items()
        ) or not row["id"] or not row["task"] or not row["candidates"] or any(
            not isinstance(c, str) or not c for c in row["candidates"]
        ):
            raise DatasetError(f"{name}:{line_no}: invalid six-key schema")
        if row["task"] not in entry.get("task_values", {task: len(lines)}):
            raise DatasetError(f"{name}:{line_no}: unregistered task value {row['task']}")
        if row["ground_truth"] not in row["candidates"]:
            raise DatasetError(f"{name}:{line_no}: ground_truth absent from candidates")
        if row["id"] in ids:
            raise DatasetError(f"{name}:{line_no}: duplicate id {row['id']}")
        if expected_ids is not None and (not isinstance(expected_ids, list) or
                                         line_no > len(expected_ids) or row["id"] != expected_ids[line_no - 1]):
            raise DatasetError(f"{name}:{line_no}: sample_ids mismatch")
        ids.add(row["id"])
        rows.append(row)
    if "task_values" in entry:
        from collections import Counter
        if dict(Counter(r["task"] for r in rows)) != entry["task_values"]:
            raise DatasetError(f"{name}: task_values count mismatch")
    if expected_ids is not None and len(expected_ids) != len(rows):
        raise DatasetError(f"{name}: sample_ids count mismatch")
    return rows


def load_registry(data_dir: Path | str, suite: str = "all", *, strict_files: bool = True) -> tuple[dict, dict]:
    data_dir = Path(data_dir)
    try:
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"cannot read manifest: {exc}") from exc
    tasks = registered_tasks(manifest, suite)
    if strict_files:
        actual = {p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*.jsonl")}
        unknown = actual - allowed_files(manifest)
        if unknown:
            raise DatasetError(f"unregistered JSONL files: {sorted(unknown)}")
    return manifest, {name: check_file(data_dir, name, entry) for name, entry in tasks.items()}
