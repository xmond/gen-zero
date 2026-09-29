"""Spec 20 P1: partition anchor pooling (S2.3 path A). Covers:
  1. short-sequence degeneracy (T = 1, 5, 20): no out-of-bounds, no NaN, head==tail merge;
  2. pad exclusion: appended zero/random pad rows with mask=0 leave the output unchanged;
  3. anti-dilution: a 5-token discriminative direction inside T=1000 noise tokens is
     recovered by anchor pooling far better than uniform mean;
  4. mode shapes: convex stays D-wide, concat is 3D-wide;
  5. non-finite input defense;
  6. torch tensor input (skipped if torch is not installed) and NumPy-without-torch import.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES))

import partition_anchor_pooling as pap  # noqa: E402

Config = pap.PartitionAnchorConfig
Pooler = pap.PartitionAnchorPooler


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = Config()
    assert cfg.n_head == 64 and cfg.n_tail == 128 and cfg.n_anchor == 64
    assert cfg.tau == pytest.approx(0.1)
    assert cfg.mode == "convex"
    assert cfg.alpha == (0.2, 0.5, 0.3)


@pytest.mark.parametrize("kwargs", [
    {"n_head": 0}, {"n_head": -1}, {"n_head": 1.5},
    {"n_tail": 0}, {"n_anchor": 0},
    {"tau": 0.0}, {"tau": -0.1},
    {"mode": "mean"},
    {"alpha": (0.5, 0.5)},
    {"alpha": (0.5, 0.5, 0.5)},
    {"alpha": (-0.1, 0.6, 0.5)},
])
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_config_accepts_custom_alpha_and_concat():
    cfg = Config(mode="concat", alpha=(1 / 3, 1 / 3, 1 / 3))
    assert cfg.mode == "concat"


# ---------------------------------------------------------------------------
# Short-sequence degeneracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 5, 20])
def test_short_sequence_no_nan_no_oob(T):
    D = 16
    H = _rng(T).normal(size=(T, D)).astype(np.float32)
    cfg = Config(n_head=64, n_tail=128, n_anchor=64)
    out = Pooler(cfg).pool(H)
    assert out.shape == (D,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("T", [1, 5, 20])
def test_short_sequence_head_tail_merge_to_full_mean(T):
    D = 8
    H = _rng(T + 1).normal(size=(T, D)).astype(np.float64)
    cfg = Config(n_head=64, n_tail=128, n_anchor=64, mode="concat")
    out = Pooler(cfg).pool(H, query=H.mean(axis=0))
    p_head, p_tail, p_anchor = out[:D], out[D:2 * D], out[2 * D:]
    full_mean = H.mean(axis=0)
    assert np.allclose(p_head, full_mean, atol=1e-10)
    assert np.allclose(p_tail, full_mean, atol=1e-10)
    assert np.all(np.isfinite(p_anchor))


def test_degenerate_boundary_exact_head_plus_tail():
    D = 4
    T = 64 + 128
    H = _rng(1).normal(size=(T, D)).astype(np.float64)
    cfg = Config(n_head=64, n_tail=128, n_anchor=16, mode="concat")
    out = Pooler(cfg).pool(H, query=H.mean(axis=0))
    p_head, p_tail = out[:D], out[D:2 * D]
    full_mean = H.mean(axis=0)
    assert np.allclose(p_head, full_mean, atol=1e-10)
    assert np.allclose(p_tail, full_mean, atol=1e-10)


# ---------------------------------------------------------------------------
# Pad exclusion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pad_kind", ["zeros", "random"])
def test_pad_exclusion_matches_unpadded(pad_kind):
    D = 32
    T = 300
    rng = _rng(7)
    H = rng.normal(size=(T, D)).astype(np.float64)
    query = rng.normal(size=D)
    cfg = Config(n_head=64, n_tail=128, n_anchor=64)
    pooler = Pooler(cfg)

    out_no_pad = pooler.pool(H, attention_mask=None, query=query)

    n_pad = 50
    if pad_kind == "zeros":
        pad_rows = np.zeros((n_pad, D))
    else:
        pad_rows = rng.normal(size=(n_pad, D)) * 100.0
    H_padded = np.concatenate([H, pad_rows], axis=0)
    mask = np.concatenate([np.ones(T, dtype=np.int64), np.zeros(n_pad, dtype=np.int64)])
    out_padded = pooler.pool(H_padded, attention_mask=mask, query=query)

    assert np.allclose(out_no_pad, out_padded, atol=1e-6)


def test_pad_exclusion_leading_pad_too():
    D = 16
    T = 100
    rng = _rng(11)
    H = rng.normal(size=(T, D)).astype(np.float64)
    query = rng.normal(size=D)
    cfg = Config(n_head=8, n_tail=8, n_anchor=8)
    pooler = Pooler(cfg)

    out_no_pad = pooler.pool(H, query=query)

    n_pad = 10
    pad_rows = np.zeros((n_pad, D))
    H_padded = np.concatenate([pad_rows, H], axis=0)
    mask = np.concatenate([np.zeros(n_pad, dtype=np.int64), np.ones(T, dtype=np.int64)])
    out_padded = pooler.pool(H_padded, attention_mask=mask, query=query)

    assert np.allclose(out_no_pad, out_padded, atol=1e-6)


def test_all_pad_raises():
    D = 8
    H = _rng(1).normal(size=(10, D))
    mask = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError):
        Pooler().pool(H, attention_mask=mask)


# ---------------------------------------------------------------------------
# Anti-dilution: anchor pooling must recover a sparse discriminative direction
# ---------------------------------------------------------------------------

def test_anchor_pooling_resists_dilution_vs_uniform_mean():
    D = 64
    T = 1000
    m = 5
    rng = _rng(42)

    v = rng.normal(size=D)
    v = v / np.linalg.norm(v)
    signal_scale = 12.0

    H = rng.normal(scale=1.0, size=(T, D))
    insert_at = 500
    for i in range(m):
        H[insert_at + i] = v * signal_scale + rng.normal(scale=0.1, size=D)

    cfg = Config(n_head=64, n_tail=128, n_anchor=32, tau=0.1, mode="convex", alpha=(0.0, 0.0, 1.0))
    p_anchor = Pooler(cfg).pool(H, query=v)

    uniform_mean = H.mean(axis=0)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    cos_anchor = cos(p_anchor, v)
    cos_uniform = cos(uniform_mean, v)

    assert cos_anchor > 0.9, f"anchor pooling failed to recover v: cos={cos_anchor}"
    assert cos_anchor > cos_uniform + 0.5, (
        f"anchor pooling ({cos_anchor}) not decisively better than uniform mean ({cos_uniform})"
    )


def test_anchor_pooling_beats_uniform_without_explicit_query_scale_check():
    # Same construction, but confirm uniform mean's signal really is ~ m*v/T-scaled (weak).
    D = 64
    T = 1000
    m = 5
    rng = _rng(43)
    v = rng.normal(size=D)
    v = v / np.linalg.norm(v)
    H = rng.normal(scale=1.0, size=(T, D))
    for i in range(m):
        H[500 + i] = v * 12.0
    uniform_mean = H.mean(axis=0)
    cos_uniform = float(uniform_mean @ v / np.linalg.norm(uniform_mean))
    assert cos_uniform < 0.5


# ---------------------------------------------------------------------------
# Mode shapes
# ---------------------------------------------------------------------------

def test_convex_mode_preserves_width():
    D = 20
    H = _rng(2).normal(size=(200, D))
    out = Pooler(Config(mode="convex")).pool(H)
    assert out.shape == (D,)


def test_concat_mode_triples_width():
    D = 20
    H = _rng(3).normal(size=(200, D))
    out = Pooler(Config(mode="concat")).pool(H)
    assert out.shape == (3 * D,)


def test_convex_alpha_weighting_matches_manual_combination():
    D = 12
    T = 500
    rng = _rng(5)
    H = rng.normal(size=(T, D))
    query = rng.normal(size=D)
    alpha = (0.5, 0.3, 0.2)
    cfg_convex = Config(mode="convex", alpha=alpha, n_head=32, n_tail=32, n_anchor=16)
    cfg_concat = Config(mode="concat", alpha=alpha, n_head=32, n_tail=32, n_anchor=16)
    out_convex = Pooler(cfg_convex).pool(H, query=query)
    out_concat = Pooler(cfg_concat).pool(H, query=query)
    D_ = D
    manual = (alpha[0] * out_concat[:D_] + alpha[1] * out_concat[D_:2 * D_]
              + alpha[2] * out_concat[2 * D_:])
    assert np.allclose(out_convex, manual, atol=1e-10)


# ---------------------------------------------------------------------------
# Non-finite defense
# ---------------------------------------------------------------------------

def test_rejects_nan_in_valid_tokens():
    D = 8
    H = _rng(1).normal(size=(20, D))
    H[3, 0] = np.nan
    with pytest.raises(ValueError):
        Pooler().pool(H)


def test_ignores_nan_in_masked_out_tokens():
    D = 8
    H = _rng(1).normal(size=(20, D))
    H[15:] = np.nan
    mask = np.concatenate([np.ones(15, dtype=np.int64), np.zeros(5, dtype=np.int64)])
    out = Pooler(Config(n_head=4, n_tail=4, n_anchor=4)).pool(H, attention_mask=mask)
    assert np.all(np.isfinite(out))


def test_rejects_inf():
    D = 8
    H = _rng(1).normal(size=(20, D))
    H[0, 0] = np.inf
    with pytest.raises(ValueError):
        Pooler().pool(H)


def test_rejects_nan_query():
    D = 8
    H = _rng(1).normal(size=(20, D))
    query = np.full(D, np.nan)
    with pytest.raises(ValueError):
        Pooler().pool(H, query=query)


def test_rejects_mismatched_query_width():
    D = 8
    H = _rng(1).normal(size=(20, D))
    query = np.zeros(D + 1)
    with pytest.raises(ValueError):
        Pooler().pool(H, query=query)


def test_rejects_mismatched_mask_length():
    D = 8
    H = _rng(1).normal(size=(20, D))
    mask = np.ones(19, dtype=np.int64)
    with pytest.raises(ValueError):
        Pooler().pool(H, attention_mask=mask)


# ---------------------------------------------------------------------------
# Torch interop
# ---------------------------------------------------------------------------

def test_numpy_path_works_without_torch_import(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch intentionally blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    D = 16
    H = _rng(9).normal(size=(50, D))
    out = Pooler().pool(H)
    assert np.all(np.isfinite(out))
    assert out.shape == (D,)


def test_torch_tensor_input_matches_numpy():
    torch = pytest.importorskip("torch")
    D = 24
    T = 300
    rng = _rng(6)
    H_np = rng.normal(size=(T, D)).astype(np.float32)
    query_np = rng.normal(size=D).astype(np.float32)
    cfg = Config(n_head=32, n_tail=32, n_anchor=16)

    out_np = Pooler(cfg).pool(H_np, query=query_np)

    H_t = torch.from_numpy(H_np)
    query_t = torch.from_numpy(query_np)
    out_t = Pooler(cfg).pool(H_t, query=query_t)

    assert isinstance(out_t, torch.Tensor)
    assert out_t.dtype == H_t.dtype
    np.testing.assert_allclose(out_t.numpy(), out_np, atol=1e-5)


def test_torch_tensor_with_mask():
    torch = pytest.importorskip("torch")
    D = 16
    T = 40
    rng = _rng(13)
    H_np = rng.normal(size=(T, D)).astype(np.float32)
    mask_np = np.ones(T, dtype=np.int64)
    mask_np[-5:] = 0
    H_t = torch.from_numpy(H_np)
    mask_t = torch.from_numpy(mask_np)
    out = Pooler(Config(n_head=4, n_tail=4, n_anchor=4)).pool(H_t, attention_mask=mask_t)
    assert isinstance(out, torch.Tensor)
    assert torch.all(torch.isfinite(out))
