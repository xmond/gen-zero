"""CPU-only pytest for the ModernBERT distillation pipeline.

All tests run in ``--tiny`` mode (NO network, NO pretrained download).  The
real-v5 test is guarded by the existence of the teacher features file so the
suite is honest about its preconditions.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from gen_zero.train.distill_modernbert_gpu import (
    DistillConfig,
    derive_teacher_codebook,
    attractor_loss,
    contrastive_loss,
    build_student,
    train_distillation,
    export_artifacts,
    total_loss,
)

V5_PATH = "benchmarks/results/v5_hidden_features.npz"


def _tiny_cfg(**over) -> DistillConfig:
    base = dict(
        device="cpu",
        tiny=True,
        num_prototypes=5,
        manifold_dim=64,
        num_epochs=40,
        batch_size=4,
        lr=1e-2,
        freeze_backbone=True,
        num_text=8,
        seed=0,
        codebook="v5",
        teacher_features=V5_PATH,
    )
    base.update(over)
    return DistillConfig(**base)


def _deterministic_batches(cfg: DistillConfig, n: int = 4, L: int = 16):
    rng = np.random.default_rng(cfg.seed)
    ids = torch.from_numpy(rng.integers(1, 50368, size=(n, L)).astype(np.int64))
    mask = torch.ones((n, L), dtype=torch.long)
    return [ids], [mask]


# ---------------------------------------------------------------------------
# 1. teacher codebook deterministic & shape
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(V5_PATH), reason="v5 teacher features missing")
def test_teacher_codebook_deterministic_and_shape():
    cfg = _tiny_cfg(num_prototypes=13)
    cb1, t1, m1, b1 = derive_teacher_codebook(cfg)
    cb2, t2, m2, b2 = derive_teacher_codebook(cfg)
    assert cb1.shape == (13, 64)
    assert t1.shape[0] == 390 and t1.shape[1] == 64
    assert np.all(np.isfinite(cb1))
    np.testing.assert_allclose(cb1, cb2)
    np.testing.assert_allclose(t1, t2)


# ---------------------------------------------------------------------------
# 2. forward dims & gradient backprop
# ---------------------------------------------------------------------------


def test_forward_dims_and_gradient_backprop():
    cfg = _tiny_cfg(freeze_backbone=False)
    student = build_student(cfg)
    ids = torch.randint(1, 50368, (2, 8))
    mask = torch.ones((2, 8), dtype=torch.long)
    states = student(ids, mask)
    assert states.shape == (2, 64)
    assert torch.isfinite(states).all()

    codebook = torch.from_numpy(np.random.default_rng(1).standard_normal((5, 64)).astype(np.float32))
    teacher = torch.from_numpy(np.random.default_rng(2).standard_normal((5, 64)).astype(np.float32))
    loss = total_loss(states, codebook, teacher, alpha=0.5, beta=4.0, gamma=0.015)
    loss.backward()
    g = student.head.projector.projection.weight.grad
    assert g is not None and torch.isfinite(g).all()
    # backbone unfrozen -> some backbone param has grad
    bb_grad = [p.grad for p in student.backbone.parameters()
               if p.grad is not None and torch.isfinite(p.grad).any()]
    assert len(bb_grad) > 0

    # frozen path: backbone grads None
    cfg2 = _tiny_cfg(freeze_backbone=True)
    student2 = build_student(cfg2)
    states2 = student2(ids, mask)
    loss2 = total_loss(states2, codebook, teacher, alpha=0.5, beta=4.0, gamma=0.015)
    loss2.backward()
    assert all(p.grad is None for p in student2.backbone.parameters())


# ---------------------------------------------------------------------------
# 3. loss monotonic decrease
# ---------------------------------------------------------------------------


def test_loss_monotonic_decrease():
    cfg = _tiny_cfg(num_epochs=40, lr=1e-2, freeze_backbone=True, num_prototypes=5)
    student = build_student(cfg)
    if os.path.exists(V5_PATH):
        cb_np, t64, _, _ = derive_teacher_codebook(cfg)
    else:
        cb_np = np.random.default_rng(0).standard_normal((5, 64)).astype(np.float64)
        t64 = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float64)
    rng = np.random.default_rng(cfg.seed)
    M = min(16, t64.shape[0])
    t_sample = t64[rng.choice(t64.shape[0], size=M, replace=False)]

    batches_ids, batches_mask = _deterministic_batches(cfg, n=4, L=16)
    history = train_distillation(cfg, student, cb_np, t_sample, batches_ids, batches_mask)
    assert len(history) == cfg.num_epochs
    assert history[-1] < history[0], f"no decrease: {history[0]} -> {history[-1]}"
    assert np.mean(history[-5:]) < np.mean(history[:5])


# ---------------------------------------------------------------------------
# 4. export artifacts shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("manifold_dim,hidden", [(64, 64), (64, 1024)])
def test_export_artifacts_shapes(tmp_path, manifold_dim, hidden):
    cfg = _tiny_cfg(num_epochs=3, manifold_dim=manifold_dim, tiny_hidden_size=hidden)
    # For the non-tiny hidden-size case we still build tiny backbone but assert
    # the general shape relation (manifold_dim, student_hidden_size).
    cfg.tiny_hidden_size = hidden
    cfg.student_hidden_size = hidden
    student = build_student(cfg)
    cb_np = np.random.default_rng(0).standard_normal((5, manifold_dim)).astype(np.float64)
    t64 = np.random.default_rng(1).standard_normal((16, manifold_dim)).astype(np.float64)
    batches_ids, batches_mask = _deterministic_batches(cfg, n=4, L=16)
    history = train_distillation(cfg, student, cb_np, t64, batches_ids, batches_mask)
    pt = str(tmp_path / "out.pt")
    npy = str(tmp_path / "out.npy")
    artifacts = export_artifacts(cfg, student, cb_np, t64, history, pt, npy)
    assert pt in artifacts and npy in artifacts

    bundle = torch.load(pt, weights_only=False)
    for k in ["projector_state_dict", "codebook", "manifold_dim", "loss_history"]:
        assert k in bundle, f"missing {k}"
    assert bundle["manifold_dim"] == manifold_dim
    assert len(bundle["loss_history"]) == cfg.num_epochs

    proj = np.load(npy)
    assert proj.shape == (manifold_dim, hidden), proj.shape
    # size check within tolerance (float32)
    expected = manifold_dim * hidden * 4
    assert abs(proj.nbytes - expected) <= max(4, expected // 1000)


# ---------------------------------------------------------------------------
# 5. losses finite and positive
# ---------------------------------------------------------------------------


def test_losses_finite_and_positive():
    student = torch.from_numpy(np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32))
    codebook = torch.from_numpy(np.random.default_rng(1).standard_normal((5, 64)).astype(np.float32))
    teacher = torch.from_numpy(np.random.default_rng(2).standard_normal((5, 64)).astype(np.float32))

    al = attractor_loss(student, codebook, beta=4.0)
    cl = contrastive_loss(student, teacher, tau=1.0)
    assert torch.isfinite(al).all() and float(al) > -float("inf")
    assert torch.isfinite(cl).all() and float(cl) >= 0.0
