"""Negative-contract tests for RNNSetAdapterRuntime (P1-03).

Two failure modes must fail closed rather than silently produce a candidate:
  1. Non-finite / non-positive rms_eps or rho_max at construction time.
  2. Non-finite scores (numeric overflow) reaching score()/predict(); predict()
     must never let argmax turn a NaN score into a valid candidate index.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gen_zero.causal.causal_mcts_rnn import certified_contraction
from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime


def _valid_arrays_cfg(D=16, d=8, r=2, T=6, heads=2, L=1, seed=0):
    rng = np.random.default_rng(seed)
    F = 2 * d
    g = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    arr = {"mu": 0.1 * g(D), "W_in": g(D, d) / math.sqrt(D), "lam": g(d),
           "U_A": 0.5 * g(d, r), "V_A": 0.5 * g(d, r), "U_B": 0.5 * g(d, r), "V_B": 0.5 * g(d, r),
           "W_s": np.eye(d, dtype=np.float32) + 0.2 * g(d, d),
           "wq": 0.4 * g(L, d, d), "bq": 0.05 * g(L, d), "wk": 0.4 * g(L, d, d), "bk": 0.05 * g(L, d),
           "wv": 0.4 * g(L, d, d), "bv": 0.05 * g(L, d), "wo": 0.4 * g(L, d, d), "bo": 0.05 * g(L, d),
           "w1": 0.4 * g(L, F, d), "b1": 0.05 * g(L, F), "w2": 0.4 * g(L, d, F), "b2": 0.05 * g(L, d)}
    cfg = {"in_dim": D, "d": d, "rank": r, "think_steps": T, "n_heads": heads, "n_layers": L, "ffn_dim": F}
    return arr, cfg, D


def _runtime(rms_eps=1e-6, rho_max=0.95, D=16, d=8, r=2, T=6, heads=2, L=1, seed=0):
    arr, cfg, _ = _valid_arrays_cfg(D, d, r, T, heads, L, seed)
    a_scale, _ = certified_contraction(arr["lam"], arr["U_A"], arr["V_A"], rho_max)
    return RNNSetAdapterRuntime(arr, cfg, a_scale, rho_max, rms_eps, {"test": True})


# --- 1. rms_eps / rho_max scalar bounds (gen_zero/causal/rnn_set_adapter.py:62-65) ---

@pytest.mark.parametrize("rms_eps", [float("nan"), -2.0, 0.0])
def test_init_rejects_bad_rms_eps(rms_eps):
    with pytest.raises(ValueError, match="rms_eps"):
        _runtime(rms_eps=rms_eps)


@pytest.mark.parametrize("rho_max", [float("nan"), -1.0, 0.0])
def test_init_rejects_bad_rho_max(rho_max):
    with pytest.raises(ValueError, match="rho_max"):
        _runtime(rho_max=rho_max)


def test_init_accepts_valid_scalars():
    rt = _runtime(rms_eps=1e-6, rho_max=0.95)
    assert rt.eps > 0 and np.isfinite(rt.eps)
    assert rt.rho_max > 0 and np.isfinite(rt.rho_max)


# --- 2. score()/predict() must fail closed on numeric overflow ---
# (gen_zero/causal/rnn_set_adapter.py:152-161)

def test_score_rejects_non_finite_output():
    rt = _runtime()
    D = rt.cfg["in_dim"]
    q = np.full(D, 3e38, dtype=np.float32)  # finite input, overflows inside _encode/_rms
    C = np.random.default_rng(1).standard_normal((5, D)).astype(np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        rt.score(q, C)


def test_predict_never_returns_candidate_zero_on_overflow():
    rt = _runtime()
    D = rt.cfg["in_dim"]
    q = np.full(D, 3e38, dtype=np.float32)
    C = np.random.default_rng(1).standard_normal((5, D)).astype(np.float32)
    with pytest.raises(ValueError):
        rt.predict(q, C)


def test_score_and_predict_agree_on_normal_input():
    rt = _runtime()
    D = rt.cfg["in_dim"]
    rng = np.random.default_rng(2)
    q = rng.standard_normal(D).astype(np.float32)
    C = rng.standard_normal((7, D)).astype(np.float32)
    s = rt.score(q, C)
    assert np.all(np.isfinite(s))
    assert rt.predict(q, C) == int(np.argmax(s))
