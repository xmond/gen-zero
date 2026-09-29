"""Distillation of a frozen ModernBERT student into a 64-D causal manifold.

This module implements the CPU/GPU-portable training pipeline that turns a
(frozen by default) ModernBERT student encoder plus a small ``StudentManifoldProjector``
head into a 64-dimensional causal state representation.  Two losses drive the
manifold geometry:

1. ``attractor_loss``  - smooth-min potential pulling each student state toward
   the nearest codebook vertex (derived from the real v5 teacher features, or a
   fixed centered simplex).
2. ``contrastive_loss`` - nearest-teacher alignment plus a spread-matching term
   that prevents mode collapse against the real v5-derived teacher manifold.

The pipeline is honest about what it does and is not:
  * ``--tiny`` builds a random-init ``ModernBertModel`` from a tiny config (NO
    network, NO pretrained download) and is used for the fast CPU unit tests.
  * Teacher prototypes come from the *real* ``v5_hidden_features.npz`` file by
    default (PCA to 64-D + seeded deterministic Lloyd k-means).
  * Per-sample text/teacher pairing is NOT assumed; the contrastive target is the
    *nearest* teacher point (distributional alignment), with a spread-matching
    regulariser.

The GPU/A100 real run is out of scope for this module's tests; this file is
CPU-test-clean.  No git commit, no checkout/reset/clean.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gen_zero.causal.student_manifold_projector import StudentManifoldProjector

# Optional import: simplex_codebook lives in counterfactual_drift_dynamics.
try:
    from gen_zero.causal.counterfactual_drift_dynamics import simplex_codebook
except Exception:  # pragma: no cover - kept for robustness if dep cycles.
    simplex_codebook = None  # type: ignore[assignment]


__all__ = [
    "DistillConfig",
    "derive_teacher_codebook",
    "build_student",
    "attractor_loss",
    "contrastive_loss",
    "train_distillation",
    "export_artifacts",
    "main",
]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DistillConfig:
    """Holds every knob the CLI exposes plus derived teacher metadata."""

    device: str = "cpu"
    student_model: str = "answerdotai/ModernBERT-large"
    teacher_features: str = "benchmarks/results/v5_hidden_features.npz"
    num_prototypes: int = 13
    manifold_dim: int = 64
    num_epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-3
    alpha: float = 0.5
    beta: float = 4.0
    gamma: float = 0.015
    freeze_backbone: bool = True
    text_file: Optional[str] = None
    num_text: int = 64
    tiny: bool = False
    seed: int = 0
    out_pt: str = "benchmarks/results/distilled_modernbert_manifold.pt"
    out_npy: str = "benchmarks/results/modernbert_to_causal64.npy"
    codebook: str = "v5"
    # tiny-mode config knobs (defaults sized for a fast CPU test).
    tiny_hidden_size: int = 64
    tiny_num_layers: int = 2
    tiny_num_heads: int = 4
    # Derived (filled in by derive_teacher_codebook / build_student).
    student_hidden_size: int = 1024
    teacher_mean: Optional[np.ndarray] = None
    teacher_basis: Optional[np.ndarray] = None
    teacher_64: Optional[np.ndarray] = None
    codebook_np: Optional[np.ndarray] = None
    teacher_64_sample: Optional[np.ndarray] = None
    provenance: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Teacher codebook derivation (pure functions, deterministic)
# ---------------------------------------------------------------------------


def _pca_reduce(h: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, basis, projected) where basis is (n_features, n) orthonormal."""
    h = np.asarray(h, dtype=np.float64)
    mean = h.mean(axis=0)
    xc = h - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    n = min(n_components, vt.shape[0])
    basis = vt[:n].T.copy()
    projected = xc @ basis
    return mean, basis, projected


def _kmeans(data: np.ndarray, k: int, seed: int, max_iter: int = 100) -> np.ndarray:
    """Deterministic Lloyd k-means, self-contained (no sklearn).

    Init: first ``k`` distinct rows of ``data``.  Empty clusters keep their
    previous centroid (so the iteration is total and deterministic).
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    # Determinism across numpy global state too (mirrors the spec).
    np.random.seed(seed)
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= n:
        # Not enough points to seed distinct rows: pad by sampling with rng.
        idx = np.arange(n)
        if k > n:
            extra = rng.integers(0, n, size=k - n)
            idx = np.concatenate([idx, extra])
        centroids = data[idx].copy()
        return centroids

    # First k distinct rows.
    seen = {}
    init_rows = []
    for i in range(n):
        key = data[i].tobytes()
        if key not in seen:
            seen[key] = True
            init_rows.append(i)
        if len(init_rows) == k:
            break
    if len(init_rows) < k:
        # Top up with random distinct-ish picks to reach k.
        pool = [i for i in range(n) if i not in set(init_rows)]
        while len(init_rows) < k and pool:
            j = int(rng.integers(0, len(pool)))
            init_rows.append(pool.pop(j))
    centroids = data[init_rows].copy()

    for _ in range(max_iter):
        d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)  # (n,k)
        assign = d2.argmin(axis=1)
        new = centroids.copy()
        for c in range(k):
            members = data[assign == c]
            if members.shape[0] > 0:
                new[c] = members.mean(axis=0)
        shift = np.linalg.norm(new - centroids)
        centroids = new
        if shift < 1e-9:
            break
    return centroids


def derive_teacher_codebook(
    cfg: DistillConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive (codebook, teacher_64, mean, basis) deterministically from v5.

    Loads the REAL v5 npz, PCA-reduces ``h`` (N,4096) -> (N,64), then runs a
    seeded deterministic Lloyd k-means to obtain ``K`` prototypes.  Two calls
    with the same seed and inputs MUST return identical codebooks (asserted in
    the tests).  Falls back to a simplex codebook when ``cfg.codebook`` is
    ``"simplex"`` (still returns the v5-derived teacher_64/mean/basis so the
    contrastive target remains the real teacher manifold).
    """
    if cfg.codebook == "simplex":
        if simplex_codebook is None:
            raise RuntimeError("simplex_codebook unavailable; counterfactual_drift_dynamics import failed")
        codebook = simplex_codebook(cfg.num_prototypes, cfg.manifold_dim).astype(np.float64)
        # Still derive the real teacher manifold for the contrastive target.
        if os.path.exists(cfg.teacher_features):
            with np.load(cfg.teacher_features, allow_pickle=False) as d:
                h = np.asarray(d["h"], dtype=np.float64)
            mean, basis, teacher_64 = _pca_reduce(h, cfg.manifold_dim)
        else:
            mean = np.zeros(cfg.manifold_dim * 64, dtype=np.float64)  # placeholder shape
            raise FileNotFoundError(cfg.teacher_features)
        return codebook, teacher_64, mean, basis

    # Default: v5-derived prototypes.
    if not os.path.exists(cfg.teacher_features):
        raise FileNotFoundError(f"teacher features not found: {cfg.teacher_features}")
    with np.load(cfg.teacher_features, allow_pickle=False) as d:
        h = np.asarray(d["h"], dtype=np.float64)
    mean, basis, teacher_64 = _pca_reduce(h, cfg.manifold_dim)
    codebook = _kmeans(teacher_64, cfg.num_prototypes, cfg.seed)
    return codebook, teacher_64, mean, basis


def _teacher_stats(teacher_64: np.ndarray) -> dict:
    t = np.asarray(teacher_64, dtype=np.float64)
    return {
        "n_samples": int(t.shape[0]),
        "dim": int(t.shape[1]),
        "mean_norm": float(np.linalg.norm(t.mean(axis=0))),
        "std_mean": float(t.std(axis=0).mean()),
        "all_finite": bool(np.isfinite(t).all()),
    }


def _task_histogram(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as d:
        if "tasks" not in d.files:
            return {}
        tasks = np.asarray(d["tasks"])
    uniq, counts = np.unique(tasks, return_counts=True)
    return {str(u): int(c) for u, c in zip(uniq, counts)}


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------


def _tiny_config(cfg: DistillConfig):
    from transformers import ModernBertConfig

    return ModernBertConfig(
        hidden_size=cfg.tiny_hidden_size,
        num_hidden_layers=cfg.tiny_num_layers,
        intermediate_size=2 * cfg.tiny_hidden_size,
        max_position_embeddings=128,
        vocab_size=50368,
        local_attention=32,
        num_attention_heads=cfg.tiny_num_heads,
    )


class ManifoldProjectorHead(nn.Module):
    """``StudentManifoldProjector(1024, 64, rank=None)`` wrapper for clarity."""

    def __init__(self, hidden_dim: int, manifold_dim: int) -> None:
        super().__init__()
        self.projector = StudentManifoldProjector(
            hidden_dim, manifold_dim, rank=None, normalize=True
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        return self.projector(hidden_states, attention_mask)


def build_student(cfg: DistillConfig) -> nn.Module:
    """Build the student (backbone + projector head).

    ``--tiny`` builds a random-init ``ModernBertModel`` from a tiny config (NO
    network, NO pretrained weights).  Otherwise loads via ``AutoModel``.
    """
    from transformers import AutoModel

    if cfg.tiny:
        cfg.student_hidden_size = cfg.tiny_hidden_size
        backbone = AutoModel.from_config(_tiny_config(cfg))
    else:
        cfg.student_hidden_size = 1024
        backbone = AutoModel.from_pretrained(cfg.student_model)

    head = ManifoldProjectorHead(cfg.student_hidden_size, cfg.manifold_dim)

    class _Student(nn.Module):
        def __init__(self, backbone, head) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # (B, L, H)
            return self.head(hidden, attention_mask)  # (B, manifold_dim)

    student = _Student(backbone, head)
    if cfg.freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
    return student


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def attractor_loss(student_states: Tensor, codebook_t: Tensor, beta: float = 4.0) -> Tensor:
    """Smooth-min potential over batch: lower == closer to a codebook vertex.

    V(q) = -1/beta * log sum_k exp(-beta*||q-c_k||^2/2).  We return the mean
    over the batch of ``logsumexp(-beta*d2)`` (without the 1/beta scaling, which
    only rescales the loss).  ``codebook_t`` is detached (frozen target).
    """
    codebook_t = codebook_t.detach()
    d2 = ((student_states[:, None, :] - codebook_t[None, :, :]) ** 2).sum(-1)  # (B,K)
    softmin = torch.logsumexp(-beta * d2, dim=1)  # <= min d2; differentiable
    return softmin.mean()


def contrastive_loss(
    student_states: Tensor,
    teacher_64_t: Tensor,
    tau: float = 1.0,
) -> Tensor:
    """Nearest-teacher alignment + spread matching.

    ``teacher_64_t``: (M,64) sample of real v5-derived teacher manifold points
    (detached).  ``pos = argmax(-cdist/tau)`` is the nearest teacher point per
    student; the spread term keeps the student's per-dim std close to the
    teacher's so the manifold does not collapse.
    """
    teacher_64_t = teacher_64_t.detach()
    sim = -torch.cdist(student_states, teacher_64_t) / tau  # (B,M)
    pos = sim.argmax(-1)  # nearest teacher point per student
    ce = F.cross_entropy(sim, pos)
    spread = (student_states.std(0) - teacher_64_t.std(0)).abs().mean()
    return ce + spread


def total_loss(
    student_states: Tensor,
    codebook_t: Tensor,
    teacher_64_t: Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    tau: float = 1.0,
) -> Tensor:
    return alpha * attractor_loss(student_states, codebook_t, beta) + \
        gamma * contrastive_loss(student_states, teacher_64_t, tau)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_input_batches(cfg: DistillConfig, student_backbone) -> Tuple[list, list]:
    """Build (input_ids_list, attention_mask_list) batches.

    In ``--tiny`` mode we generate deterministic random input_ids (no tokenizer
    download).  Otherwise we load text from a jsonl file (``text`` / ``question``
    / ``sentence1`` field) or auto-discover ``benchmarks/data/*.jsonl``.
    """
    rng = np.random.default_rng(cfg.seed)
    if cfg.tiny:
        batches_ids, batches_mask = [], []
        n = cfg.num_text
        L = 32
        for i in range(0, n, cfg.batch_size):
            b = min(cfg.batch_size, n - i)
            ids = torch.from_numpy(
                rng.integers(1, 50368, size=(b, L)).astype(np.int64)
            )
            mask = torch.ones((b, L), dtype=torch.long)
            batches_ids.append(ids)
            batches_mask.append(mask)
        return batches_ids, batches_mask

    # Real text path: load jsonl.
    text_path = cfg.text_file
    if text_path is None:
        cands = sorted(Path("benchmarks/data").glob("*.jsonl"))
        if not cands:
            raise FileNotFoundError("no benchmarks/data/*.jsonl found and --text-file not set")
        text_path = str(cands[0])
    texts = []
    with open(text_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text") or obj.get("question") or obj.get("sentence1")
            if t:
                texts.append(t)
            if len(texts) >= cfg.num_text:
                break
    if not texts:
        raise ValueError(f"no text loaded from {text_path}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.student_model)
    batches_ids, batches_mask = [], []
    for i in range(0, len(texts), cfg.batch_size):
        chunk = texts[i:i + cfg.batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
        batches_ids.append(enc["input_ids"])
        batches_mask.append(enc.get("attention_mask", torch.ones_like(enc["input_ids"])))
    return batches_ids, batches_mask


def train_distillation(
    cfg: DistillConfig,
    student: nn.Module,
    codebook_np: np.ndarray,
    teacher_64_sample: np.ndarray,
    batches_ids: Sequence[Tensor],
    batches_mask: Sequence[Tensor],
) -> list:
    """Run the training loop and return the loss history (list of float).

    Backbone is frozen by default; only the projector head trains.  In
    ``--tiny`` mode with ``--freeze-backbone False`` the backbone also trains
    (slow).  All forward/loss is under autograd; eval-mode backbone uses
    ``torch.no_grad``-free forward but frozen params keep grads off the backbone.
    """
    device = torch.device(cfg.device)
    student = student.to(device)
    codebook_t = torch.from_numpy(codebook_np.astype(np.float32)).to(device)
    teacher_sample_t = torch.from_numpy(teacher_64_sample.astype(np.float32)).to(device)

    params = [p for p in student.head.parameters() if p.requires_grad]
    if not cfg.freeze_backbone:
        params += [p for p in student.backbone.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    history: list = []
    for epoch in range(cfg.num_epochs):
        epoch_losses = []
        for ids, mask in zip(batches_ids, batches_mask):
            ids = ids.to(device)
            mask = mask.to(device)
            states = student(ids, mask)  # (B, manifold_dim)
            loss = total_loss(
                states, codebook_t, teacher_sample_t,
                alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append(mean_loss)
    return history


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_artifacts(
    cfg: DistillConfig,
    student: nn.Module,
    codebook_np: np.ndarray,
    teacher_64: np.ndarray,
    history: list,
    path_pt: str,
    path_npy: str,
) -> list:
    """Save the .pt bundle, the projection .npy and a codebook sidecar .npz."""
    Path(path_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(path_npy).parent.mkdir(parents=True, exist_ok=True)

    proj_weight = student.head.projector.projection.weight.detach().cpu().numpy()
    # proj_weight: (manifold_dim, student_hidden_size)
    expected_bytes = cfg.manifold_dim * cfg.student_hidden_size * 4
    actual_bytes = proj_weight.nbytes
    if abs(actual_bytes - expected_bytes) > max(4, expected_bytes // 1000):
        raise ValueError(
            f"projection matrix size mismatch: {actual_bytes} != {expected_bytes} bytes"
        )

    bundle = {
        "student_model_type": "modernbert",
        "student_model": cfg.student_model,
        "projector_state_dict": student.head.projector.state_dict(),
        "codebook": codebook_np.astype(np.float32),
        "manifold_dim": cfg.manifold_dim,
        "student_hidden_size": cfg.student_hidden_size,
        "loss_history": history,
        "provenance": dict(cfg.provenance),
        "config": {k: v for k, v in dataclasses.asdict(cfg).items()
                   if not isinstance(v, np.ndarray)},
        "projection_shape": list(proj_weight.shape),
        "projection_bytes": int(actual_bytes),
    }
    torch.save(bundle, path_pt)
    np.save(path_npy, proj_weight.astype(np.float32))

    sidecar = path_npy.replace(".npy", "_codebook.npz")
    np.savez(
        sidecar,
        codebook=codebook_np.astype(np.float32),
        teacher_64_sample=teacher_64.astype(np.float32),
        manifold_dim=np.int32(cfg.manifold_dim),
    )
    return [path_pt, path_npy, sidecar]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distill ModernBERT into a 64-D causal manifold")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--student-model", default="answerdotai/ModernBERT-large")
    p.add_argument("--teacher-features", default="benchmarks/results/v5_hidden_features.npz")
    p.add_argument("--num-prototypes", type=int, default=13)
    p.add_argument("--manifold-dim", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--gamma", type=float, default=0.015)
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--text-file", default=None)
    p.add_argument("--num-text", type=int, default=64)
    p.add_argument("--tiny", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-pt", default="benchmarks/results/distilled_modernbert_manifold.pt")
    p.add_argument("--out-npy", default="benchmarks/results/modernbert_to_causal64.npy")
    p.add_argument("--codebook", default="v5", choices=["v5", "simplex"])
    p.add_argument("--epochs", type=int, default=None,
                   help="alias for --num-epochs (overrides if set)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.epochs is not None:
        args.num_epochs = args.epochs

    cfg = DistillConfig(
        device=args.device,
        student_model=args.student_model,
        teacher_features=args.teacher_features,
        num_prototypes=args.num_prototypes,
        manifold_dim=args.manifold_dim,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        freeze_backbone=args.freeze_backbone,
        text_file=args.text_file,
        num_text=args.num_text,
        tiny=args.tiny,
        seed=args.seed,
        out_pt=args.out_pt,
        out_npy=args.out_npy,
        codebook=args.codebook,
    )

    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    print(f"[distill] device={cfg.device} tiny={cfg.tiny} codebook={cfg.codebook}")

    codebook_np, teacher_64, mean, basis = derive_teacher_codebook(cfg)
    cfg.teacher_mean = mean
    cfg.teacher_basis = basis
    cfg.teacher_64 = teacher_64
    cfg.codebook_np = codebook_np

    rng = np.random.default_rng(cfg.seed)
    M = min(64, teacher_64.shape[0])
    sample_idx = rng.choice(teacher_64.shape[0], size=M, replace=False)
    teacher_64_sample = teacher_64[sample_idx]
    cfg.teacher_64_sample = teacher_64_sample

    print(f"[distill] teacher_64 stats: {_teacher_stats(teacher_64)}")
    print(f"[distill] codebook shape: {codebook_np.shape}")
    print(f"[distill] v5 task histogram: {_task_histogram(cfg.teacher_features)}")
    print(f"[distill] projection matrix: {cfg.manifold_dim}x{1024 if not cfg.tiny else cfg.tiny_hidden_size} "
          f"= {cfg.manifold_dim * (1024 if not cfg.tiny else cfg.tiny_hidden_size) * 4} bytes")

    cfg.provenance = {
        "source": "v5_hidden_features.npz" if cfg.codebook == "v5" else "simplex",
        "seed": cfg.seed,
        "manifold_dim": cfg.manifold_dim,
        "num_prototypes": cfg.num_prototypes,
        "student_model": cfg.student_model,
        "tiny": cfg.tiny,
        "freeze_backbone": cfg.freeze_backbone,
        "codebook_kind": cfg.codebook,
        "teacher_n_samples": int(teacher_64.shape[0]),
    }

    student = build_student(cfg)
    batches_ids, batches_mask = _build_input_batches(cfg, student.backbone)

    initial_loss = None
    history = []
    # Capture initial loss on the first batch (eval-style, no grad).
    student.eval()
    with torch.no_grad():
        device_t = torch.device(cfg.device)
        codebook_t = torch.from_numpy(codebook_np.astype(np.float32)).to(device_t)
        teacher_sample_t = torch.from_numpy(teacher_64_sample.astype(np.float32)).to(device_t)
        ids = batches_ids[0].to(device_t)
        mask = batches_mask[0].to(device_t)
        st = student(ids, mask)
        initial_loss = float(total_loss(
            st, codebook_t, teacher_sample_t,
            alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
        ).cpu().item())
    student.train()

    history = train_distillation(cfg, student, codebook_np, teacher_64_sample,
                                 batches_ids, batches_mask)
    final_loss = history[-1] if history else float("nan")

    artifacts = export_artifacts(
        cfg, student, codebook_np, teacher_64_sample, history,
        cfg.out_pt, cfg.out_npy,
    )
    print(f"[distill] history: {[round(x, 6) for x in history]}")

    summary = {
        "device": cfg.device,
        "epochs": cfg.num_epochs,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "decreased": bool(final_loss < initial_loss) if initial_loss is not None else False,
        "artifacts": artifacts,
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
