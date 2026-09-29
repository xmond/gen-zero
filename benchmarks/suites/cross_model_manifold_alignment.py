"""Cross-model manifold feature alignment for the Spec 22 / 23 GGUF teachers.

Reads two ``<task>.npz`` feature files produced by the Qwen-72B and LLaMA-70B
extractors, whose common schema is
``train_full, test_full, cands, train_label, train_ids, test_ids, info_json``.

This module never assumes the two files it is given are already row-aligned: several of those
extractors take ``--qwen-dir`` as OPTIONAL (no reference file existed for the first model of its
kind), so two feature files can legally exist with different row sets. Alignment is re-verified
here, independently of whatever check (if any) ran at extraction time, before any geometry is
computed -- a mismatched pairing must raise, never silently produce a number.

Two model-agnostic metrics (both supported teachers have 8192 hidden dimensions):

  * Linear CKA (Kornblith et al. 2019), which needs equal ``n_samples`` but not equal feature
    dimension. Reused unchanged from
    ``gen_zero.causal.universal_manifold_extractor.PhaseTransitionLayerExtractor.linear_cka``,
    not reimplemented here.
  * An orthogonal-Procrustes residual, generalized to unequal dims. For centered,
    Frobenius-normalized ``A`` (n x p) and ``B`` (n x q) with ``p <= q``, minimizing
    ``||A R - B||_F`` subject to ``R R^T = I_p`` (R has orthonormal ROWS) is exactly solved by
    ``R = U V^T`` from the SVD of ``A^T B``: because ``R R^T = I_p``, ``||A R||_F^2 == ||A||_F^2``
    is CONSTANT over the whole feasible set, so minimizing the residual reduces to maximizing
    ``tr(R^T A^T B)``, which is what the SVD solves. This does NOT hold in the other orientation
    (mapping the larger block down via ``R^T R = I_q`` on ``q > p`` columns): there ``||A R||_F^2``
    varies with R, so the same SVD formula only maximizes the cross term and is a heuristic upper
    bound on the true minimum, not the minimum itself (verified numerically: for a random 40x9 A
    and 40x4 B, the SVD solution gives residual 14.20, while 5000 steps of manifold gradient
    descent from that starting point find 12.76 -- a real, non-numerical-noise gap).

    So ``procrustes_residual`` always identifies the lower-dimensional of the two blocks and maps
    IT into the higher-dimensional block's space (arbitrarily picking the first argument on a tie),
    never the reverse -- this is also the only direction with a real geometric interpretation
    without loss: a lower-dimensional manifold can be isometrically embedded in a higher-dimensional
    space, but a higher-dimensional one cannot in general be embedded losslessly into fewer
    dimensions. A useful side effect: this makes ``procrustes_residual(X, Y) ==
    procrustes_residual(Y, X)``, so the direction two feature files are passed in does not change
    the reported number. ``gen_zero.nanocore.zca_whitening.orthogonal_procrustes`` additionally
    corrects a reflection sign, which is only well-defined for a SQUARE rotation and is not
    attempted here for p != q.

    Centering and Frobenius-normalizing both blocks before the residual matters: these are raw,
    unnormalized llama-server states (``embd_normalize=-1``), so a scale-naive residual would be
    dominated by the scale mismatch between two different teachers and would say nothing about
    representational shape.

    The residual is evaluated in closed form, without materializing ``R``: with both blocks
    Frobenius-normalized, ``||A R - B||_F^2 = 2 - 2 ||A^T B||_*`` (nuclear norm), and the singular
    values of the (p x q) matrix ``A^T B`` equal those of the small ``(n x n)`` matrix
    ``S_a U_a^T U_b S_b`` built from the thin SVDs ``A = U_a S_a V_a^T``, ``B = U_b S_b V_b^T``.
    That costs O(n^2 (p + q)) instead of the O(p^3) full SVD of ``A^T B`` (measured: 288 s per block
    at n = 250, p = q = 8192, versus seconds). Because the closed form subtracts two O(1) numbers,
    residuals below ~1e-7 are quantized by float64 round-off (sqrt of ~1e-16).

Controls (``controls=True`` / ``--controls``): with n samples << feature dim (here n = 250-1000
against d = 8192) both metrics are biased toward "similar" even for unrelated data, because a
linear map with 8192 free directions can fit n rows almost arbitrarily. Two controls are reported
next to every measured value so the number is never read alone:

  * a row-permutation null: the same metrics after shuffling the rows of the second model's block
    (breaks example pairing, keeps every marginal statistic), seeded and averaged;
  * a per-feature z-scored CKA: raw last-token LLM states carry a few very-large-magnitude
    "massive activation" dimensions that can dominate an unstandardized linear kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
from gen_zero.causal.universal_manifold_extractor import PhaseTransitionLayerExtractor  # noqa: E402

REQUIRED_KEYS: Tuple[str, ...] = ("train_full", "test_full", "cands", "train_label", "train_ids", "test_ids")
ID_KEYS: Tuple[str, ...] = ("train_ids", "test_ids")
FEATURE_BLOCKS: Tuple[str, ...] = ("train_full", "test_full")


def _finite_2d(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name}: must be a nonempty 2D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: contains non-finite (NaN/Inf) values")
    return array


def load_features(path: Path) -> Dict[str, np.ndarray]:
    """Load and validate one extractor's ``<task>.npz``. Refuses non-finite feature blocks and a
    file whose own feature/id/label row counts disagree with each other (a file can be internally
    broken the same way in isolation, before it is ever compared against a second file)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        data = {key: np.array(archive[key]) for key in REQUIRED_KEYS}
    for block in ("train_full", "test_full", "cands"):
        _finite_2d(data[block], f"{path}:{block}")
    for id_key in ID_KEYS:
        if data[id_key].ndim != 1:
            raise ValueError(f"{path}:{id_key}: must be 1D, got shape {data[id_key].shape}")
    if data["train_label"].shape[0] != data["train_ids"].shape[0]:
        raise ValueError(f"{path}: train_label has {data['train_label'].shape[0]} rows, "
                         f"train_ids has {data['train_ids'].shape[0]}")
    for feature_key, id_key in (("train_full", "train_ids"), ("test_full", "test_ids")):
        if data[feature_key].shape[0] != data[id_key].shape[0]:
            raise ValueError(f"{path}: {feature_key} has {data[feature_key].shape[0]} rows, "
                             f"{id_key} has {data[id_key].shape[0]}")
    return data


def verify_id_alignment(a: Dict[str, np.ndarray], a_name: str,
                        b: Dict[str, np.ndarray], b_name: str) -> None:
    """Refuse unless train_ids, test_ids and train_label are identical, id for id and label for
    label. This is the load-bearing check: everything downstream assumes row i in a's train_full
    and row i in b's train_full describe the SAME underlying example."""
    for key in ID_KEYS:
        va, vb = a[key], b[key]
        if va.shape != vb.shape:
            raise ValueError(f"{key}: shape differs, {a_name} has {va.shape}, {b_name} has {vb.shape}")
        if not np.array_equal(va, vb):
            diff = np.flatnonzero(va != vb)
            i = int(diff[0])
            raise ValueError(
                f"{key}: not identical between {a_name} and {b_name} ({diff.size} of {va.shape[0]} "
                f"positions differ); first mismatch at index {i}: {va[i]!r} != {vb[i]!r}"
            )
    la, lb = a["train_label"], b["train_label"]
    if la.shape != lb.shape or not np.array_equal(la, lb):
        raise ValueError(
            f"train_label differs between {a_name} and {b_name}: identical ids but different ground "
            "truth labels -- the two files do not describe the same task build"
        )


def procrustes_residual(X: np.ndarray, Y: np.ndarray) -> float:
    """Rectangular orthogonal-Procrustes residual between centered, Frobenius-normalized X, Y.

    X: (n, d1), Y: (n, d2), same n. Returns the residual Frobenius norm (relative to the target's
    norm) in [0, 2] after both blocks are centered and scaled to unit Frobenius norm (so identical
    scale never masks or inflates the residual). 0 = an isometry between X and Y reproduces the
    target exactly (up to reflection); values near sqrt(2) indicate no shared linear structure.

    Symmetric in (X, Y): internally always maps the LOWER-dimensional block into the
    higher-dimensional block's space (see module docstring for why the SVD solution is only the
    true minimizer in that direction), so procrustes_residual(X, Y) == procrustes_residual(Y, X).
    """
    X = _finite_2d(X, "X")
    Y = _finite_2d(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples, got {X.shape[0]} and {Y.shape[0]}")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    x_norm = float(np.linalg.norm(Xc))
    y_norm = float(np.linalg.norm(Yc))
    if x_norm <= 1e-12 or y_norm <= 1e-12:
        raise ValueError("degenerate input: near-zero variance after centering")
    Xn, Yn = Xc / x_norm, Yc / y_norm
    # source = the lower-dimensional block, target = the higher-dimensional one (tie -> X is
    # source). Only in this orientation does R R^T == I_{source_dim} hold, which is what makes
    # ||source @ R||_F constant and the SVD solution the exact minimizer (see module docstring).
    if Xn.shape[1] <= Yn.shape[1]:
        source, target = Xn, Yn
    else:
        source, target = Yn, Xn
    us, ss, _ = np.linalg.svd(source, full_matrices=False)
    ut, st, _ = np.linalg.svd(target, full_matrices=False)
    # Singular values of source^T @ target, via the small (k_s x k_t) core (module docstring).
    core = (ss[:, None] * (us.T @ ut)) * st[None, :]
    nuclear = float(np.linalg.svd(core, compute_uv=False).sum())
    return float(np.sqrt(max(2.0 - 2.0 * nuclear, 0.0)))


def standardized_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA after per-feature z-scoring, dropping constant features (std == 0).

    Guards against a handful of huge-magnitude dimensions dominating the unstandardized kernel."""
    def zscore(M: np.ndarray) -> np.ndarray:
        M = _finite_2d(M, "block")
        std = M.std(axis=0)
        keep = std > 1e-12
        if not keep.any():
            raise ValueError("degenerate input: every feature is constant")
        return (M[:, keep] - M[:, keep].mean(axis=0)) / std[keep]
    return PhaseTransitionLayerExtractor.linear_cka(zscore(X), zscore(Y))


def row_permutation_null(X: np.ndarray, Y: np.ndarray, permutations: int, seed: int) -> Dict[str, object]:
    """Both metrics after shuffling Y's rows (pairing destroyed), averaged over ``permutations``
    seeded shuffles. Identity shuffles are rejected so every draw really breaks the pairing."""
    rng = np.random.default_rng(seed)
    ckas, residuals = [], []
    for _ in range(permutations):
        order = rng.permutation(Y.shape[0])
        while np.array_equal(order, np.arange(Y.shape[0])):
            order = rng.permutation(Y.shape[0])
        Yp = Y[order]
        ckas.append(PhaseTransitionLayerExtractor.linear_cka(X, Yp))
        residuals.append(procrustes_residual(X, Yp))
    return {
        "permutations": int(permutations), "seed": int(seed),
        "linear_cka_mean": float(np.mean(ckas)), "linear_cka_max": float(np.max(ckas)),
        "procrustes_residual_mean": float(np.mean(residuals)),
        "procrustes_residual_min": float(np.min(residuals)),
    }


def align_block(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], block: str,
                controls: bool = False, null_permutations: int = 3, seed: int = 0) -> Dict[str, object]:
    """Linear CKA and Procrustes residual for one shared block ("train_full" or "test_full").

    ``controls=True`` additionally reports the z-scored CKA and the row-permutation null."""
    X, Y = a[block], b[block]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"{block}: sample counts differ ({X.shape[0]} vs {Y.shape[0]}); ids were "
                         "supposed to be verified before calling align_block")
    result: Dict[str, object] = {
        "block": block,
        "n_samples": int(X.shape[0]),
        "dim_a": int(X.shape[1]),
        "dim_b": int(Y.shape[1]),
        "linear_cka": PhaseTransitionLayerExtractor.linear_cka(X, Y),
        "procrustes_residual": procrustes_residual(X, Y),
    }
    if controls:
        result["linear_cka_zscore"] = standardized_linear_cka(X, Y)
        result["null_row_permutation"] = row_permutation_null(X, Y, null_permutations, seed)
    return result


def compare(path_a: Path, path_b: Path, blocks: Sequence[str] = FEATURE_BLOCKS,
            controls: bool = False, null_permutations: int = 3, seed: int = 0) -> Dict[str, object]:
    """Load two feature files, verify row alignment, and report CKA + Procrustes for each block."""
    a, b = load_features(path_a), load_features(path_b)
    verify_id_alignment(a, str(path_a), b, str(path_b))
    return {
        "path_a": str(path_a), "path_b": str(path_b),
        "n_train": int(a["train_ids"].shape[0]), "n_test": int(a["test_ids"].shape[0]),
        "blocks": [align_block(a, b, block, controls, null_permutations, seed) for block in blocks],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_directories(dir_a: Path, dir_b: Path, tasks: Sequence[str], blocks: Sequence[str] = FEATURE_BLOCKS,
                        controls: bool = True, null_permutations: int = 3, seed: int = 0) -> Dict[str, object]:
    """Run ``compare`` over ``<dir>/<task>.npz`` for every task. A missing file or an id mismatch
    on ANY task raises: a partial table must never be presented as the full 13-task result."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    per_task: Dict[str, object] = {}
    for task in tasks:
        path_a, path_b = dir_a / f"{task}.npz", dir_b / f"{task}.npz"
        for path in (path_a, path_b):
            if not path.is_file():
                raise FileNotFoundError(path)
        entry = compare(path_a, path_b, blocks, controls, null_permutations, seed)
        entry["sha256_a"], entry["sha256_b"] = sha256_file(path_a), sha256_file(path_b)
        per_task[task] = entry
        print(f"[{task}] " + " ".join(
            f"{blk['block']}: cka={blk['linear_cka']:.4f} procrustes={blk['procrustes_residual']:.4f}"
            for blk in entry["blocks"]), file=sys.stderr, flush=True)
    summary: Dict[str, object] = {}
    for block in blocks:
        rows = [next(b for b in per_task[t]["blocks"] if b["block"] == block) for t in tasks]
        stats = {
            "linear_cka_mean": float(np.mean([r["linear_cka"] for r in rows])),
            "procrustes_residual_mean": float(np.mean([r["procrustes_residual"] for r in rows])),
        }
        if controls:
            stats["linear_cka_zscore_mean"] = float(np.mean([r["linear_cka_zscore"] for r in rows]))
            stats["null_linear_cka_mean"] = float(np.mean([r["null_row_permutation"]["linear_cka_mean"] for r in rows]))
            stats["null_procrustes_residual_mean"] = float(
                np.mean([r["null_row_permutation"]["procrustes_residual_mean"] for r in rows]))
        summary[block] = stats
    return {"dir_a": str(dir_a), "dir_b": str(dir_b), "tasks": list(tasks), "n_tasks": len(tasks),
            "controls": bool(controls), "summary_unweighted_mean_over_tasks": summary, "per_task": per_task}


def render_markdown(report: Dict[str, object], name_a: str, name_b: str) -> str:
    """Readable per-task table of a ``compare_directories`` report."""
    controls = bool(report["controls"])
    lines = [f"# Cross-model manifold alignment: {name_a} vs {name_b}", "",
             f"Tasks: {report['n_tasks']}. A = `{report['dir_a']}`, B = `{report['dir_b']}`.", ""]
    for block in FEATURE_BLOCKS:
        lines += [f"## {block}", ""]
        head = "| task | n | CKA | Procrustes residual |"
        rule = "|---|---:|---:|---:|"
        if controls:
            head += " CKA z-scored | null CKA | null Procrustes |"
            rule += "---:|---:|---:|"
        lines += [head, rule]
        for task in report["tasks"]:
            row = next(b for b in report["per_task"][task]["blocks"] if b["block"] == block)
            cells = f"| {task} | {row['n_samples']} | {row['linear_cka']:.4f} | {row['procrustes_residual']:.4f} |"
            if controls:
                nul = row["null_row_permutation"]
                cells += (f" {row['linear_cka_zscore']:.4f} | {nul['linear_cka_mean']:.4f} |"
                          f" {nul['procrustes_residual_mean']:.4f} |")
            lines.append(cells)
        stats = report["summary_unweighted_mean_over_tasks"][block]
        mean = f"| **mean** | | {stats['linear_cka_mean']:.4f} | {stats['procrustes_residual_mean']:.4f} |"
        if controls:
            mean += (f" {stats['linear_cka_zscore_mean']:.4f} | {stats['null_linear_cka_mean']:.4f} |"
                     f" {stats['null_procrustes_residual_mean']:.4f} |")
        lines += [mean, ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path_a", type=Path, nargs="?", help="single-pair mode: first <task>.npz")
    parser.add_argument("path_b", type=Path, nargs="?", help="single-pair mode: second <task>.npz")
    parser.add_argument("--blocks", default=",".join(FEATURE_BLOCKS))
    parser.add_argument("--dir-a", type=Path, help="batch mode: directory holding <task>.npz for model A")
    parser.add_argument("--dir-b", type=Path, help="batch mode: directory holding <task>.npz for model B")
    parser.add_argument("--tasks", help="batch mode: comma-separated task names")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--controls", action="store_true", help="add z-scored CKA and row-permutation null")
    parser.add_argument("--null-permutations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    blocks = [b for b in args.blocks.split(",") if b]
    if args.dir_a or args.dir_b or args.tasks:
        if not (args.dir_a and args.dir_b and args.tasks and args.out_json):
            parser.error("batch mode needs --dir-a, --dir-b, --tasks and --out-json")
        tasks = [t for t in args.tasks.split(",") if t]
        report = compare_directories(args.dir_a, args.dir_b, tasks, blocks, args.controls,
                                     args.null_permutations, args.seed)
        report["loadavg_at_end"] = list(os.getloadavg())
        args.out_json.write_text(json.dumps(report, indent=2) + "\n")
        if args.out_md:
            args.out_md.write_text(render_markdown(report, args.name_a, args.name_b) + "\n")
        return 0
    if not (args.path_a and args.path_b):
        parser.error("give either path_a path_b, or --dir-a/--dir-b/--tasks/--out-json")
    report = compare(args.path_a, args.path_b, blocks, args.controls, args.null_permutations, args.seed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
