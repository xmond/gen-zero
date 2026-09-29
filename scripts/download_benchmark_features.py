#!/usr/bin/env python3
"""Locate and verify the frozen features for the dual Qwen2.5-72B + LLaMA-3.1-70B
13-task manifold reproduction.

This is a **locator and verifier**, not a downloader: there is no public mirror for
these features. If a file is missing this script prints "no public mirror; contact
maintainers" and reports MISSING -- it never fabricates, fetches from an invented
URL, or silently substitutes another file.

The official benchmark uses only two frozen feature sets, both 8192-d hidden states
read straight off the model (no random projection at extraction time):

  * Qwen2.5-72B  (8192-d), 13 tasks, ~1.1 GB total
  * LLaMA-3.1-70B (8192-d), 13 tasks, ~1.1 GB total
  * ~2.2 GB combined

There is no dependency on Mistral-123B or Llama-3.1-405B features anywhere in this
path (a prior "master manifold" 3-model result that used 123B features was removed
on 2026-09-29 because the 123B feature directory was a broken symlink and the result
was never actually reproducible; see docs/manuals/closed_loop_dense_pipeline.md).

The 26 SHA256 hashes below (13 tasks x 2 models) are pinned from the run that
produced benchmarks/results/spec21_manifold_pareto_ensemble_report.json (Peak
macro accuracy 81.52%, see benchmarks/results/spec21_manifold_pareto_ensemble_report.md).
They were extracted with:

    python3 -c "
    import json
    d = json.load(open('benchmarks/results/spec21_manifold_pareto_ensemble_report.json'))
    manifest = {}
    for task, info in d['tasks'].items():
        q = l = None
        for path, h in info['feature_sha256'].items():
            if '/q/' in path: q = h
            elif '/l/' in path: l = h
        manifest[task] = {'qwen': q, 'llama': l}
    print(manifest)
    "

Usage:
    python3 scripts/download_benchmark_features.py
    python3 scripts/download_benchmark_features.py --no-hash
    python3 scripts/download_benchmark_features.py --qwen-dir DIR --llama-dir DIR --json out.json

Exit codes: 0 all 26 files verify (or, with --no-hash, are all present); 1 one or
more files are missing or fail hash verification; 2 bad arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

TASKS = (
    "massive_en", "massive_de", "multinli", "pubmedqa", "vitaminc", "boolq", "squad2",
    "paws", "civil_comments", "aegis_safety", "helpsteer2", "summeval_relevance",
    "summeval_consistency",
)

# Pinned manifest: 13 tasks x {qwen, llama} = 26 SHA256 hashes, extracted from
# benchmarks/results/spec21_manifold_pareto_ensemble_report.json (see docstring for
# the exact extraction command). This manifest is authoritative; a directory's own
# SHA256SUMS (if present) is cross-checked against it but never overrides it.
FEATURE_SHA256: Dict[str, Dict[str, str]] = {
    "aegis_safety": {"qwen": "1a549d60906f4d93039c3219277a7eaf17508f88ea0df8599f50e88d0b9d5c1b", "llama": "ef2960039e5970a5a9c7d7c8fdd3e7a287c8eaf66ade743fcbd8a2ddae989e4a"},
    "boolq": {"qwen": "0b5bb874e4905ff62574294f646d0a5eef15c8a6e7bb3518938615e6b0897f2c", "llama": "d3b0172849c785016ee76ea01dde4f3c05c448e2ae49a29be9e2c88080d0da8b"},
    "civil_comments": {"qwen": "81e12de1a1db9a3ea2033444bef54e381fe852a5a84232bebc3288d28a7057b6", "llama": "3254b02807e72215284eae762a666984bd4c729083c46d2b0a5c2af1c9824e5b"},
    "helpsteer2": {"qwen": "62568fd84eda03651a536de838025b9ae5239143b4bf6b6c7f88ad6ff421807b", "llama": "1765cda1421617ca321c52c3d9f837c67e3a35ecf474b68873c457a518d7db2a"},
    "massive_de": {"qwen": "522db7e482cf5aa65cbb3aa99a5d441e6c608f8e2633392f5e35e917766a168c", "llama": "62bb38bc9e080d95f372d1206de2c4a21376bed95fc42f25fef1f679026d7baa"},
    "massive_en": {"qwen": "6bf96fd909a0883d26d58fcec2ed3c3e6234b831f0885093bac92b6a4fb6c1a8", "llama": "bbb91c2a9a579a04e5ae97224e3be8af34ed18a0c7659f470a55477894aca7ef"},
    "multinli": {"qwen": "53fcfbee86975b7110147179d6c9a0813e881631889a7c618434a7d01eea8728", "llama": "1b8165fe5c72e6e8859d24b4f9c7d0065147c92336ae46a2d994c4e210fe4504"},
    "paws": {"qwen": "987e9db1cfbfe34305fe3615fb0f7c06d8e56c3b9f6b0d3c3d78ec4e0b7ee415", "llama": "1df188764d1e8d89e18a11d5f52209242360cd708c1aa6d08eb0b3f63e818e4a"},
    "pubmedqa": {"qwen": "aa07455d2972410bc77e0fb075e0ba606421f398a7f5595b8fb0bee3eca0b6b1", "llama": "8c2980c43d71d4409b6d7dc6c6eedc80c5a8c74a03793d5e338f5329a67f606d"},
    "squad2": {"qwen": "00d1b4ff0d09633c49cb61d66de55bd50c0e49e57e3169d950d87e2ec65b0899", "llama": "cad113956b9e1dde9faeaf7113ae9cdba0c85549fccffa22a0d74801d93ed7f4"},
    "summeval_consistency": {"qwen": "77479cf6057d9cb4cc2b4c9e6f738a3e02441c455cafe5b6b826d448763d8d23", "llama": "99300d5a00a6032981cacfda0110696f4c403607267f69d48a6e4e1a4272075d"},
    "summeval_relevance": {"qwen": "23d3e573f72d9fb931300c16ae866da6ac60da05b87c5ae2d5c68fb6949a8598", "llama": "bd9b99aa47841bf4e26bf3e91d88aba12a9dc4d9090d122649750fbf6ccff7bc"},
    "vitaminc": {"qwen": "93cc2f1482ac513e7a12f87309554fe80fea6f31126d2738224e5dc8be19bba9", "llama": "7687fc0dfdd2358a0693f36a87f078cdf406b78ef57dabe36d91680d9a3fe6f0"},
}

AUTO_QDIR = Path("/ebs/data/extracted_features/qwen72b/features")
AUTO_LDIR = Path("/ebs/data/extracted_features/llama70b")

NO_MIRROR_MESSAGE = (
    "no public mirror; contact maintainers "
    "(see benchmarks/README.md for the reproduction guide)"
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_dir(cli_value: Optional[str], env_name: str, auto_path: Path, label: str):
    """Returns (path, source_description). Priority: CLI arg > env var > auto-detect."""
    if cli_value:
        return Path(cli_value), f"{label}: --{'qwen' if 'QWEN' in env_name else 'llama'}-dir CLI argument"
    if os.environ.get(env_name):
        return Path(os.environ[env_name]), f"{label}: ${env_name}"
    return auto_path, f"{label}: auto-detected default ({auto_path})"


def read_sha256sums(dir_path: Path) -> Dict[str, str]:
    """Best-effort parse of a directory's own SHA256SUMS file (sha256sum -c format)."""
    sums_path = dir_path / "SHA256SUMS"
    if not sums_path.is_file():
        return {}
    result = {}
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts
        name = name.lstrip("*").strip()
        result[name] = digest
    return result


def check_features(qdir: Path, ldir: Path, verify_hash: bool):
    """Checks all 26 pinned files. Returns (all_ok, per_file_status list)."""
    statuses = []
    qsums = read_sha256sums(qdir)
    lsums = read_sha256sums(ldir)
    all_ok = True
    for task in TASKS:
        for model, model_dir, sums in (("qwen", qdir, qsums), ("llama", ldir, lsums)):
            expected = FEATURE_SHA256[task][model]
            path = model_dir / f"{task}.npz"
            entry = {
                "task": task, "model": model, "path": str(path),
                "expected_sha256": expected,
            }
            if not path.is_file():
                entry["status"] = "MISSING"
                entry["detail"] = NO_MIRROR_MESSAGE
                all_ok = False
                statuses.append(entry)
                continue
            entry["size_bytes"] = path.stat().st_size
            local_sum_name = path.name
            if local_sum_name in sums:
                entry["dir_sha256sums_agrees"] = (sums[local_sum_name] == expected)
            if not verify_hash:
                entry["status"] = "PRESENT (unverified)"
                entry["detail"] = "hash NOT verified (--no-hash); this is not a readiness guarantee"
                statuses.append(entry)
                continue
            actual = sha256_of(path)
            entry["actual_sha256"] = actual
            if actual == expected:
                entry["status"] = "sha256 OK"
            else:
                entry["status"] = "MISMATCH"
                all_ok = False
            statuses.append(entry)
    if verify_hash:
        # "PRESENT (unverified)" never counts as READY; under hash mode, MISSING/MISMATCH gate all_ok.
        pass
    else:
        # Under --no-hash, presence alone is never reported as READY, and the exit
        # code still reflects only whether every file is present.
        all_ok = all(s["status"] != "MISSING" for s in statuses)
    return all_ok, statuses


def print_report(qdir, qsrc, ldir, lsrc, statuses, verify_hash: bool):
    print(f"Qwen2.5-72B feature dir : {qdir}  ({qsrc})")
    print(f"LLaMA-3.1-70B feature dir: {ldir}  ({lsrc})")
    print()
    header = f"{'task':<22}{'model':<7}{'status':<22}{'size':>12}"
    print(header)
    print("-" * len(header))
    for s in statuses:
        size = s.get("size_bytes", "")
        print(f"{s['task']:<22}{s['model']:<7}{s['status']:<22}{str(size):>12}")
        if s["status"] == "MISSING":
            print(f"  -> {s['detail']}")
        if s["status"] == "MISMATCH":
            print(f"  -> expected {s['expected_sha256']} got {s.get('actual_sha256')}")
        if "dir_sha256sums_agrees" in s and not s["dir_sha256sums_agrees"]:
            print(f"  -> WARNING: directory's own SHA256SUMS disagrees with the pinned manifest for {s['path']}")
    print()


def print_success_commands():
    print("All 26 files verified against the pinned manifest. Reproduction commands:")
    print()
    print("# Full 13-task Pareto run (produces the 81.52% Peak / 81.60% control report)")
    print(f"export MASTER_QWEN_DIR={AUTO_QDIR}")
    print(f"export MASTER_LLAMA_DIR={AUTO_LDIR}")
    print("python3 benchmarks/suites/evaluate_manifold_pareto_ensemble.py --workers 3")
    print()
    print("# Dual ridge-probe ensemble run")
    print("python3 benchmarks/suites/evaluate_dual_70b_72b_ensemble.py")
    print()
    print("# One-task quick check (fast smoke test; expected peak test accuracy 89.71 for massive_en)")
    print("python3 benchmarks/suites/evaluate_manifold_pareto_ensemble.py --tasks massive_en --workers 1")
    print()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qwen-dir", default=None,
                         help="Qwen2.5-72B feature directory (default: $MASTER_QWEN_DIR or auto-detect)")
    parser.add_argument("--llama-dir", default=None,
                         help="LLaMA-3.1-70B feature directory (default: $MASTER_LLAMA_DIR or auto-detect)")
    parser.add_argument("--no-hash", action="store_true",
                         help="Fast existence-only check; hashes are NOT verified, status is 'PRESENT (unverified)', never READY")
    parser.add_argument("--json", default=None, help="Write a machine-readable status report to this path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        # argparse already printed usage/error; normalize the "bad args" exit code to 2.
        return exc.code if exc.code else 2

    qdir, qsrc = resolve_dir(args.qwen_dir, "MASTER_QWEN_DIR", AUTO_QDIR, "Qwen2.5-72B")
    ldir, lsrc = resolve_dir(args.llama_dir, "MASTER_LLAMA_DIR", AUTO_LDIR, "LLaMA-3.1-70B")

    verify_hash = not args.no_hash
    all_ok, statuses = check_features(qdir, ldir, verify_hash)
    print_report(qdir, qsrc, ldir, lsrc, statuses, verify_hash)

    if args.json:
        payload = {
            "qwen_dir": str(qdir), "qwen_dir_source": qsrc,
            "llama_dir": str(ldir), "llama_dir_source": lsrc,
            "hash_verified": verify_hash,
            "all_ok": all_ok,
            "files": statuses,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json}")

    if all_ok:
        if verify_hash:
            print_success_commands()
        else:
            print("All 26 files are present (hashes NOT verified; re-run without --no-hash to confirm integrity).")
        return 0

    print("FAILED: one or more files are missing or fail hash verification. "
          + NO_MIRROR_MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
