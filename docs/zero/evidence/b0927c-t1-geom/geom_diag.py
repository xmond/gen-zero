"""Diagnostic evidence for task b0927c-t1-geom (NOT production code, NOT imported anywhere).

Read-only geometry diagnostics on the row-aligned 13-task feature files of two dense teachers
(Qwen2.5-72B and Llama-3.1-70B, both 8192-D, llama-server last-token raw states). Answers three
measurable questions the research report needs numbers for:

  Q1  intrinsic geometry per model: participation ratio, Gavish-Donoho rank (reused from
      spec21_advanced_heads, not reimplemented), norm concentration (spherical check),
      massive-activation share, and Gromov four-point delta-hyperbolicity on raw / z-scored /
      angular distances against a Gaussian null with the SAME singular spectrum.
  Q2  is a single global orthogonal map (one gauge) enough, or does the map vary over the manifold
      (a non-flat connection)?  Held-out tests only:
        * global, fitted on train rows, scored on the test rows: orthogonal R in a shared top-r
          subspace; R plus a diagonal scaling (structure group O(r) x Diag); ridge (linear upper bound).
        * local, per anchor: R_i fitted on k_fit nearest neighbours (k_fit >= 4 r so the local
          Procrustes is overdetermined; refused otherwise), scored on k_eval further neighbours,
          compared with the global R on the SAME eval points, and with an R fitted on a RANDOM
          (non-local) subset of the same size (kills "smaller fit set" as the explanation).
        * Lie-algebra deviation of R_i from R_global: sqrt(sum theta_j^2) over principal angles,
          with its dispersion across anchors (isotropic noise vs concentrated curvature).
  Q3  cannot be tested on these files (single static vectors, no trajectory, no second layer);
      the script records that refusal explicitly instead of inventing a momentum.

Every number is written to one JSON next to the log; nothing is cached or reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))
sys.path.insert(0, str(REPO / "python"))
from spec21_advanced_heads import _gd_from_singular_values  # noqa: E402
from cross_model_manifold_alignment import load_features, verify_id_alignment, procrustes_residual  # noqa: E402

TASKS = ("massive_en", "massive_de", "multinli", "pubmedqa", "vitaminc", "boolq", "squad2",
         "paws", "civil_comments", "aegis_safety", "helpsteer2", "summeval_relevance",
         "summeval_consistency")
DIR_A = Path("/ebs/data/extracted_features/qwen72b/features")
DIR_B = Path("/ebs/data/extracted_features/llama70b")
N_SUB = 1000            # rows used for the O(n^2) tests on the two big tasks
N_TUPLES = 20000        # four-point samples for delta
LOCAL_RANKS = (16, 32, 64)
SEED = 20260927


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- Q1 helpers

def spectrum_stats(X: np.ndarray) -> dict:
    n, d = X.shape
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    gd_raw = _gd_from_singular_values(s, n, d, center=True)
    std = Xc.std(axis=0)
    keep = std > 1e-12
    Z = Xc[:, keep] / std[keep]
    s_z = np.linalg.svd(Z, compute_uv=False)
    gd_z = _gd_from_singular_values(s_z, n, int(keep.sum()), center=True)
    lam_z = s_z ** 2
    norms = np.linalg.norm(X, axis=1)
    # massive activations: share of total second moment carried by the 8 largest coordinates
    second_moment = (X ** 2).sum(axis=0)
    top8 = np.sort(second_moment)[::-1][:8]
    var_frac = lam / lam.sum()
    return {
        "n": int(n), "d": int(d),
        "participation_ratio_raw": pr,
        "participation_ratio_zscored": float(lam_z.sum() ** 2 / (lam_z ** 2).sum()),
        "gd_rank_raw": int(gd_raw["rank"]), "gd_rank_zscored": int(gd_z["rank"]),
        "top1_var_frac_raw": float(var_frac[0]), "top10_var_frac_raw": float(var_frac[:10].sum()),
        "n_components_for_90pct_var_raw": int(np.searchsorted(np.cumsum(var_frac), 0.90) + 1),
        "norm_mean": float(norms.mean()), "norm_cv": float(norms.std() / norms.mean()),
        "massive_top8_coord_share_of_second_moment": float(top8.sum() / second_moment.sum()),
        "massive_top1_coord_share_of_second_moment": float(top8[0] / second_moment.sum()),
    }, s


def pairwise_euclid(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D


def pairwise_angular(X: np.ndarray) -> np.ndarray:
    U = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    C = np.clip(U @ U.T, -1.0, 1.0)
    D = np.arccos(C)
    np.fill_diagonal(D, 0.0)
    return D


def gromov_delta(D: np.ndarray, rng: np.random.Generator, n_tuples: int) -> dict:
    """Four-point delta: for x,y,z,w let S1=d(x,y)+d(z,w), S2=d(x,z)+d(y,w), S3=d(x,w)+d(y,z);
    delta = (largest - second largest)/2. Tree metric => 0. Reported relative to the diameter."""
    n = D.shape[0]
    idx = np.array([rng.choice(n, 4, replace=False) for _ in range(n_tuples)])
    x, y, z, w = idx.T
    S = np.stack([D[x, y] + D[z, w], D[x, z] + D[y, w], D[x, w] + D[y, z]], axis=1)
    S.sort(axis=1)
    delta = 0.5 * (S[:, 2] - S[:, 1])
    diam = float(D.max())
    return {"delta_max": float(delta.max()), "delta_mean": float(delta.mean()),
            "delta_p99": float(np.quantile(delta, 0.99)), "diameter": diam,
            "delta_rel_max": float(2.0 * delta.max() / diam),
            "delta_rel_mean": float(2.0 * delta.mean() / diam)}


def gaussian_null_same_spectrum(s: np.ndarray, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Rows = Q diag(s) with Q (n x m) random orthonormal columns: identical singular values,
    Gaussian (isotropic) directions. Geometry of the real cloud beyond its spectrum shows up as a
    difference from this null."""
    m = min(len(s), n, d)
    G = rng.standard_normal((n, m))
    Q, _ = np.linalg.qr(G)
    return Q * s[:m][None, :]


def q1_for_block(X: np.ndarray, rng: np.random.Generator, name: str) -> dict:
    stats, s = spectrum_stats(X)
    sub = X if X.shape[0] <= N_SUB else X[rng.choice(X.shape[0], N_SUB, replace=False)]
    subc = sub - sub.mean(axis=0, keepdims=True)
    std = subc.std(axis=0); keep = std > 1e-12
    subz = subc[:, keep] / std[keep]
    s_sub = np.linalg.svd(subc, compute_uv=False)
    null = gaussian_null_same_spectrum(s_sub, subc.shape[0], subc.shape[1], rng)
    out = {"spectrum": stats, "delta": {}}
    for label, M in (("raw_euclid", subc), ("zscored_euclid", subz), ("angular_raw", sub),
                     ("null_gaussian_same_spectrum_euclid", null)):
        D = pairwise_angular(M) if label.startswith("angular") else pairwise_euclid(M)
        out["delta"][label] = gromov_delta(D, rng, N_TUPLES)
    # angular null: same spectrum null, angular distance
    out["delta"]["null_gaussian_same_spectrum_angular"] = gromov_delta(pairwise_angular(null), rng, N_TUPLES)
    log(f"  Q1 {name}: PR={stats['participation_ratio_raw']:.1f} GDraw={stats['gd_rank_raw']} GDz={stats['gd_rank_zscored']} "
        f"normCV={stats['norm_cv']:.3f} massive8={stats['massive_top8_coord_share_of_second_moment']:.3f} "
        f"delta_rel_mean raw={out['delta']['raw_euclid']['delta_rel_mean']:.4f} z={out['delta']['zscored_euclid']['delta_rel_mean']:.4f} "
        f"ang={out['delta']['angular_raw']['delta_rel_mean']:.4f} null={out['delta']['null_gaussian_same_spectrum_euclid']['delta_rel_mean']:.4f}")
    return out


# ----------------------------------------------------------------------------- Q2 helpers

def shared_coords(Xtr: np.ndarray, Xte: np.ndarray, r: int, zscore: bool) -> tuple[np.ndarray, np.ndarray]:
    """Top-r principal coordinates fitted on TRAIN rows only; test rows projected with the same
    mean/std/basis. Each block scaled so the train coordinates have unit total variance."""
    mu = Xtr.mean(axis=0, keepdims=True)
    Atr, Ate = Xtr - mu, Xte - mu
    if zscore:
        std = Atr.std(axis=0); std[std <= 1e-12] = 1.0
        Atr, Ate = Atr / std, Ate / std
    _, s, vt = np.linalg.svd(Atr, full_matrices=False)
    V = vt[:r].T
    Ctr, Cte = Atr @ V, Ate @ V
    scale = np.linalg.norm(Ctr) / np.sqrt(Ctr.shape[0])
    return Ctr / scale, Cte / scale


def procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(A.T @ B, full_matrices=False)
    return u @ vt


def rel_residual(A: np.ndarray, R: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A @ R - B) / max(np.linalg.norm(B), 1e-12))


def procrustes_diag(A: np.ndarray, B: np.ndarray, iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """min ||A R D - B||_F over R orthogonal, D diagonal (alternating; O(r) x Diag structure group)."""
    D = np.ones(A.shape[1])
    R = procrustes(A, B)
    for _ in range(iters):
        P = A @ R
        D = (P * B).sum(axis=0) / np.maximum((P * P).sum(axis=0), 1e-12)
        R = procrustes(A, B / np.where(np.abs(D) > 1e-12, D, 1.0))
    return R, D


def ridge(A: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    r = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(r), A.T @ B)


def q2_global(a: dict, b: dict, r: int) -> dict:
    out = {}
    for zs in (False, True):
        Atr, Ate = shared_coords(a["train_full"], a["test_full"], r, zs)
        Btr, Bte = shared_coords(b["train_full"], b["test_full"], r, zs)
        R = procrustes(Atr, Btr)
        Rd, D = procrustes_diag(Atr, Btr)
        # ridge lambda picked on a train split (no test peeking)
        n = Atr.shape[0]; cut = int(0.8 * n)
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            W = ridge(Atr[:cut], Btr[:cut], lam * n)
            res = rel_residual(Atr[cut:], W, Btr[cut:])
            if best is None or res < best[1]:
                best = (lam, res)
        W = ridge(Atr, Btr, best[0] * n)
        out["zscored" if zs else "raw"] = {
            "r": r,
            "orthogonal_train": rel_residual(Atr, R, Btr), "orthogonal_test": rel_residual(Ate, R, Bte),
            "orthogonal_plus_diag_train": float(np.linalg.norm(Atr @ Rd * D - Btr) / np.linalg.norm(Btr)),
            "orthogonal_plus_diag_test": float(np.linalg.norm(Ate @ Rd * D - Bte) / np.linalg.norm(Bte)),
            "diag_scale_spread_log10": float(np.log10(np.abs(D).max() / max(np.abs(D).min(), 1e-12))),
            "ridge_lambda": best[0], "ridge_train": rel_residual(Atr, W, Btr), "ridge_test": rel_residual(Ate, W, Bte),
            "baseline_test_zero_map": 1.0,
        }
    return out


def q2_local(a: dict, b: dict, r: int, rng: np.random.Generator) -> dict:
    k_fit, k_eval = 4 * r, r
    Atr, _ = shared_coords(a["train_full"], a["test_full"], r, True)
    Btr, _ = shared_coords(b["train_full"], b["test_full"], r, True)
    n = Atr.shape[0]
    if n > N_SUB:
        sel = rng.choice(n, N_SUB, replace=False); Atr, Btr = Atr[sel], Btr[sel]; n = N_SUB
    if k_fit < 4 * r or k_fit + k_eval + 1 > n:
        raise RuntimeError(f"local test refused: r={r}, k_fit={k_fit}, k_eval={k_eval}, n={n}")
    Rg = procrustes(Atr, Btr)
    D = pairwise_euclid(Atr)
    order = np.argsort(D, axis=1)
    loc_res, glob_res, rand_res, lie_dev = [], [], [], []
    anchors = rng.choice(n, min(n, 300), replace=False)
    for i in anchors:
        nb = order[i, 1:1 + k_fit + k_eval]
        fit, ev = nb[:k_fit], nb[k_fit:]
        muA, muB = Atr[fit].mean(axis=0), Btr[fit].mean(axis=0)
        Ri = procrustes(Atr[fit] - muA, Btr[fit] - muB)
        A_ev, B_ev = Atr[ev] - muA, Btr[ev] - muB
        loc_res.append(rel_residual(A_ev, Ri, B_ev))
        glob_res.append(rel_residual(A_ev, Rg, B_ev))
        # random (non-local) fit set of the same size; scored on the SAME locally-centred eval
        # vectors, so the three residuals share one denominator (a rotation acts on tangent vectors
        # and is translation-free once both sides are centred).
        rnd = rng.choice(np.setdiff1d(np.arange(n), ev), k_fit, replace=False)
        Rr = procrustes(Atr[rnd] - Atr[rnd].mean(axis=0), Btr[rnd] - Btr[rnd].mean(axis=0))
        rand_res.append(rel_residual(A_ev, Rr, B_ev))
        # Lie-algebra distance ||log(R_i^T R_g)||_F = sqrt(sum theta_k^2) over the rotation angles
        # (eigenvalue arguments) of the orthogonal matrix R_i^T R_g. Singular values would all be 1.
        theta = np.abs(np.angle(np.linalg.eigvals(Ri.T @ Rg)))
        lie_dev.append(float(np.sqrt((theta ** 2).sum())))
    loc, glo, rnd, lie = map(np.array, (loc_res, glob_res, rand_res, lie_dev))
    return {
        "r": r, "k_fit": k_fit, "k_eval": k_eval, "n_rows": int(n), "n_anchors": int(len(anchors)),
        "heldout_residual_local_mean": float(loc.mean()), "heldout_residual_global_mean": float(glo.mean()),
        "heldout_residual_randomsubset_mean": float(rnd.mean()),
        "local_over_global_ratio": float(loc.mean() / glo.mean()),
        "local_beats_global_frac": float((loc < glo).mean()),
        "local_beats_randomsubset_frac": float((loc < rnd).mean()),
        "lie_deviation_mean": float(lie.mean()), "lie_deviation_cv": float(lie.std() / lie.mean()),
        "lie_deviation_top10pct_share": float(np.sort(lie)[::-1][:max(1, len(lie) // 10)].sum() / lie.sum()),
        "lie_deviation_max_possible": float(np.pi * np.sqrt(r)),
        "lie_deviation_of_random_orthogonal_pair_expected": "about pi*sqrt(r/3) for Haar-random R (angles ~ uniform on [-pi, pi])",
    }


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    report = {"task_code": "b0927c-t1-geom", "head": os.popen("git -C %s rev-parse HEAD" % REPO).read().strip(),
              "dir_a": str(DIR_A), "dir_b": str(DIR_B), "seed": SEED, "n_tuples": N_TUPLES,
              "loadavg_start": list(os.getloadavg()), "tasks": {},
              "q3_refusal": "train_full/test_full are single static last-token vectors per row: no trajectory, "
                            "no second layer, no time index. A (q, p) split cannot be validated on these files; "
                            "no momentum was fabricated."}
    for task in TASKS:
        pa, pb = DIR_A / f"{task}.npz", DIR_B / f"{task}.npz"
        a, b = load_features(pa), load_features(pb)
        verify_id_alignment(a, str(pa), b, str(pb))
        log(f"[{task}] n_train={a['train_full'].shape[0]} n_test={a['test_full'].shape[0]}")
        entry = {"sha256_a": sha256(pa), "sha256_b": sha256(pb), "n_train": int(a["train_full"].shape[0]),
                 "n_test": int(a["test_full"].shape[0]),
                 "reference_full_dim_procrustes_residual_train": procrustes_residual(a["train_full"], b["train_full"])
                 if a["train_full"].shape[0] <= N_SUB else None}
        entry["q1"] = {"qwen72b": q1_for_block(a["train_full"], rng, f"{task}/qwen72b"),
                       "llama70b": q1_for_block(b["train_full"], rng, f"{task}/llama70b")}
        entry["q2_global"] = {str(r): q2_global(a, b, r) for r in (64, 256)}
        for r in (64, 256):
            g = entry["q2_global"][str(r)]["zscored"]
            log(f"  Q2 global r={r} zscored: orth test={g['orthogonal_test']:.4f} orth+diag test={g['orthogonal_plus_diag_test']:.4f} ridge test={g['ridge_test']:.4f} (train orth={g['orthogonal_train']:.4f})")
        entry["q2_local"] = {}
        for r in LOCAL_RANKS:
            try:
                res = q2_local(a, b, r, rng)
            except RuntimeError as exc:
                res = {"refused": str(exc)}
            entry["q2_local"][str(r)] = res
            if "refused" not in res:
                log(f"  Q2 local r={r}: heldout local={res['heldout_residual_local_mean']:.4f} global={res['heldout_residual_global_mean']:.4f} "
                    f"random={res['heldout_residual_randomsubset_mean']:.4f} local<global={res['local_beats_global_frac']:.2f} "
                    f"lie_dev={res['lie_deviation_mean']:.3f} cv={res['lie_deviation_cv']:.3f}")
            else:
                log(f"  Q2 local r={r}: {res['refused']}")
        report["tasks"][task] = entry
    report["seconds"] = time.time() - t0
    report["loadavg_end"] = list(os.getloadavg())
    out = HERE / "geom_diag_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {out} in {report['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
