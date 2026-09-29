"""Tests for the Parallel RNN + Set-Attention adapter (torch trainer + NumPy runtime).

Small dims (D=32, d=16, rank=4, T=5), CPU only. The torch suite module is
imported from benchmarks/suites via sys.path, like it is deployed (standalone).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime  # noqa: E402

torch = pytest.importorskip("torch")
rsa = pytest.importorskip("rnn_set_adapter_torch")

D, DH, RANK, T = 32, 16, 4, 5


def _model(seed: int = 0, n_layers: int = 2):
    torch.manual_seed(seed)
    m = rsa.ParallelRNNSetAdapter(D, d=DH, rank=RANK, think_steps=T, n_heads=4,
                                  n_layers=n_layers, dropout=0.1)
    m.init_projection(torch.randn(200, D) * torch.linspace(0.5, 2.0, D) + 0.3)
    with torch.no_grad():  # move W_s / biases off their trivial init so parity is non-trivial
        m.W_s.add_(0.1 * torch.randn(DH, DH))
        m.U_B.mul_(3.0)
    return m.eval()


def _export(m, tmp_path, name="a.npz"):
    path = tmp_path / name
    m.export_npz(path, {"test": True})
    return path


def test_parity_torch_numpy_padded_mixed_k(tmp_path):
    m = _model()
    rt = RNNSetAdapterRuntime.from_npz(_export(m, tmp_path))
    assert rt.meta == {"test": True}
    rng = np.random.default_rng(1)
    Ks = [3, 5, 7]
    q = rng.normal(size=(3, D)).astype(np.float32)
    Cs = [rng.normal(size=(k, D)).astype(np.float32) for k in Ks]
    C = np.zeros((3, max(Ks), D), np.float32)
    mask = np.zeros((3, max(Ks)), bool)
    for i, (k, c) in enumerate(zip(Ks, Cs)):
        C[i, :k], mask[i, :k] = c, True
    with torch.no_grad():
        logits = m(torch.from_numpy(q), torch.from_numpy(C), torch.from_numpy(mask)).numpy()
    for i, k in enumerate(Ks):
        np.testing.assert_allclose(rt.score(q[i], Cs[i]), logits[i, :k], atol=1e-4, rtol=0)
        assert np.all(logits[i, k:] <= -1e8)
    assert rt.working_set_bytes() > 0


def test_permutation_equivariance(tmp_path):
    m = _model(seed=2)
    rt = RNNSetAdapterRuntime.from_npz(_export(m, tmp_path))
    rng = np.random.default_rng(3)
    q = rng.normal(size=D).astype(np.float32)
    C = rng.normal(size=(6, D)).astype(np.float32)
    perm = rng.permutation(6)
    s = rt.score(q, C)
    np.testing.assert_allclose(rt.score(q, C[perm]), s[perm], atol=1e-5, rtol=0)
    mask = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        a = m(torch.from_numpy(q)[None], torch.from_numpy(C)[None], mask)[0].numpy()
        b = m(torch.from_numpy(q)[None], torch.from_numpy(C[perm])[None], mask)[0].numpy()
    np.testing.assert_allclose(b, a[perm], atol=1e-5, rtol=0)


def test_parallel_scan_matches_sequential_loop():
    m = _model(seed=4)
    zq = torch.randn(3, DH)
    with torch.no_grad():
        A, _ = m.transition()
        Bm = m.U_B @ m.V_B.T
        u = zq @ Bm.T
        for steps in (1, 2, 5, 8):
            scan = rsa.lti_prefix_scan(A, u, steps)
            h = torch.zeros_like(zq)
            for t in range(steps):
                h = h @ A.T + u
                torch.testing.assert_close(scan[t], h, atol=1e-5, rtol=0)
        h = torch.zeros_like(zq)
        for _ in range(T):
            h = h @ A.T + u
        torch.testing.assert_close(m.think(zq), zq + h, atol=1e-5, rtol=0)


def test_stability_clamp_and_tamper(tmp_path):
    m = _model(seed=5)
    with torch.no_grad():
        m.U_A.copy_(10.0 * torch.randn(DH, RANK))
        m.V_A.copy_(10.0 * torch.randn(DH, RANK))
    path = _export(m, tmp_path)
    z = dict(np.load(path))
    raw = np.diag(1 / (1 + np.exp(-z["lam"].astype(np.float64)))) + z["U_A"].astype(np.float64) @ z["V_A"].T
    A = float(z["a_scale"]) * raw
    sigma = float(np.linalg.svd(A, compute_uv=False)[0])
    assert float(z["a_scale"]) < 1.0
    assert sigma <= m.rho_max + 1e-5 and m.rho_max < 1.0
    rt = RNNSetAdapterRuntime.from_npz(path)
    assert rt.sigma_max_A < 1.0
    assert m.spectral_radius() <= sigma + 1e-5

    z["a_scale"] = np.float32(float(z["a_scale"]) * 1.5)
    bad = tmp_path / "tampered.npz"
    np.savez(bad, **z)
    with pytest.raises(ValueError, match="a_scale"):
        RNNSetAdapterRuntime.from_npz(bad)


def test_runtime_rejects_nan(tmp_path):
    rt = RNNSetAdapterRuntime.from_npz(_export(_model(seed=6), tmp_path))
    q = np.zeros(D, np.float32)
    C = np.ones((3, D), np.float32)
    q_bad = q.copy()
    q_bad[0] = np.nan
    C_bad = C.copy()
    C_bad[1, 2] = np.inf
    with pytest.raises(ValueError):
        rt.score(q_bad, C)
    with pytest.raises(ValueError):
        rt.score(q, C_bad)


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    Ks = rng.integers(2, 9, size=n)
    offsets = np.concatenate([[0], np.cumsum(Ks)]).astype(np.int64)
    cands = rng.normal(size=(offsets[-1], D)).astype(np.float32)
    labels = np.array([rng.integers(0, k) for k in Ks], dtype=np.int64)
    q = np.stack([cands[offsets[i] + labels[i]] + 0.3 * rng.normal(size=D)
                  for i in range(n)]).astype(np.float32)
    tasks = np.array(["a" if i % 2 else "b" for i in range(n)])
    is_val = np.zeros(n, bool)
    is_val[rng.permutation(n)[: n // 4]] = True
    return rsa.AdapterData(q=q.astype(np.float16), cands=cands.astype(np.float16), offsets=offsets,
                           labels=labels, tasks=tasks, is_val=is_val), Ks


def test_train_adapter_learns_synthetic():
    data, Ks = _synthetic()
    lines = []
    model, rep = rsa.train_adapter(data, d=DH, rank=RANK, think_steps=T, n_heads=4, n_layers=1,
                                   epochs=15, lr=3e-3, batch_size=32, patience=15,
                                   device="cpu", seed=0, log=lines.append)
    required = {"n_train", "n_early_stop", "early_stop_acc", "n_val", "params", "best_epoch", "train_acc", "val_acc",
                "val_cosine_baseline_acc", "per_task", "spectral_radius_A", "sigma_max_A",
                "a_scale", "train_time_sec", "history"}
    assert required <= set(rep)
    assert set(rep["per_task"]) == {"a", "b"}
    assert all({"n_val", "val_acc", "val_cosine_baseline_acc"} <= set(v) for v in rep["per_task"].values())
    assert len(lines) == len(rep["history"]) and all("early_stop_acc" in h for h in rep["history"])
    chance = float(np.mean(1.0 / Ks[data.is_val]))
    assert rep["val_acc"] > chance + 0.3, (rep["val_acc"], chance)
    assert rep["sigma_max_A"] <= 0.95 + 1e-5
    import json
    json.dumps(rep)


def test_train_adapter_rejects_bad_input():
    data, _ = _synthetic(n=20)
    bad = rsa.AdapterData(q=data.q, cands=data.cands, offsets=data.offsets,
                          labels=data.labels + 100, tasks=data.tasks, is_val=data.is_val)
    with pytest.raises(ValueError, match="labels"):
        rsa.train_adapter(bad, d=DH, rank=RANK, device="cpu", epochs=1)
