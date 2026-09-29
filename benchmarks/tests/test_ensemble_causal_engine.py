"""Tests for the late causal consensus ensemble (pure NumPy, CPU only).

Builds synthetic RNNSetAdapterRuntime checkpoints directly in NumPy (no torch
dependency, unlike test_rnn_set_adapter.py) so two models with *different*
in_dim can be built cheaply and the independence of the two spaces is
provable rather than assumed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime  # noqa: E402
from gen_zero.causal.ensemble_causal_engine import (  # noqa: E402
    load_adapters,
    score_ensemble,
    score_single,
)


def _build_adapter_npz(path, *, in_dim, d, rank=4, think_steps=3, n_heads=2, n_layers=1,
                        ffn_dim=None, seed=0, rho_max=0.95, rms_eps=1e-5):
    """Write a stability-valid RNNSetAdapterRuntime checkpoint, pure NumPy.

    a_scale is derived the same way _check_stability re-derives it (SVD of
    diag(sigmoid(lam)) + U_A V_A^T), so the checkpoint loads cleanly.
    """
    ffn_dim = ffn_dim or 2 * d
    rng = np.random.default_rng(seed)

    def randn(*shape, scale=0.1):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    lam = randn(d, scale=1.0)
    U_A, V_A = randn(d, rank, scale=0.3), randn(d, rank, scale=0.3)
    dvec = 1.0 / (1.0 + np.exp(-lam.astype(np.float64)))
    raw = np.diag(dvec) + U_A.astype(np.float64) @ V_A.astype(np.float64).T
    sigma_raw = float(np.linalg.svd(raw, compute_uv=False)[0])
    a_scale = min(1.0, rho_max / sigma_raw) if sigma_raw > 0.0 else 1.0

    arrays = {
        "mu": randn(in_dim, scale=0.2),
        "W_in": randn(in_dim, d, scale=1.0 / np.sqrt(in_dim)),
        "lam": lam,
        "U_A": U_A, "V_A": V_A,
        "U_B": randn(d, rank, scale=0.3), "V_B": randn(d, rank, scale=0.3),
        "W_s": randn(d, d, scale=0.2),
        "wq": randn(n_layers, d, d, scale=0.1), "bq": randn(n_layers, d, scale=0.05),
        "wk": randn(n_layers, d, d, scale=0.1), "bk": randn(n_layers, d, scale=0.05),
        "wv": randn(n_layers, d, d, scale=0.1), "bv": randn(n_layers, d, scale=0.05),
        "wo": randn(n_layers, d, d, scale=0.1), "bo": randn(n_layers, d, scale=0.05),
        "w1": randn(n_layers, ffn_dim, d, scale=0.1), "b1": randn(n_layers, ffn_dim, scale=0.05),
        "w2": randn(n_layers, d, ffn_dim, scale=0.1), "b2": randn(n_layers, d, scale=0.05),
    }
    config = {"in_dim": in_dim, "d": d, "rank": rank, "think_steps": think_steps,
              "n_heads": n_heads, "n_layers": n_layers, "ffn_dim": ffn_dim}
    np.savez(path, **arrays, **config, a_scale=np.float32(a_scale), rho_max=np.float32(rho_max),
             rms_eps=np.float32(rms_eps), meta_json=json.dumps({"synthetic": True}))
    return path


def _rt(tmp_path, name, **kwargs) -> RNNSetAdapterRuntime:
    path = _build_adapter_npz(tmp_path / name, **kwargs)
    return RNNSetAdapterRuntime.from_npz(path)


# ---------------------------------------------------------------------------
# score_single
# ---------------------------------------------------------------------------

def test_score_single_matches_raw_runtime(tmp_path):
    rt = _rt(tmp_path, "a.npz", in_dim=32, d=16, rank=4)
    rng = np.random.default_rng(1)
    q, C = rng.normal(size=32).astype(np.float32), rng.normal(size=(5, 32)).astype(np.float32)
    result = score_single(rt, q, C)
    np.testing.assert_allclose(result["logits"], rt.score(q, C), atol=1e-6, rtol=0)
    assert result["pred"] == int(np.argmax(rt.score(q, C)))
    assert np.isclose(result["probs"].sum(), 1.0, atol=1e-5)


def test_score_single_mask_isolates_candidates(tmp_path):
    rt = _rt(tmp_path, "a.npz", in_dim=16, d=8, rank=2)
    rng = np.random.default_rng(2)
    q = rng.normal(size=16).astype(np.float32)
    C_real = rng.normal(size=(3, 16)).astype(np.float32)
    C_padded = np.concatenate([C_real, np.full((2, 16), np.nan, np.float32)], axis=0)
    mask = np.array([True, True, True, False, False])
    result = score_single(rt, q, C_padded, mask)
    np.testing.assert_allclose(result["logits"][:3], rt.score(q, C_real), atol=1e-6, rtol=0)
    assert np.all(result["logits"][3:] == -np.inf)
    assert np.all(result["probs"][3:] == 0.0)
    assert result["pred"] < 3


def test_score_single_rejects_all_masked(tmp_path):
    rt = _rt(tmp_path, "a.npz", in_dim=8, d=4, rank=2)
    q, C = np.zeros(8, np.float32), np.zeros((3, 8), np.float32)
    with pytest.raises(ValueError, match="zero candidates"):
        score_single(rt, q, C, mask=np.zeros(3, bool))


# ---------------------------------------------------------------------------
# score_ensemble: numerical correctness of the fusion strategies
# ---------------------------------------------------------------------------

def test_ensemble_single_model_equals_score_single(tmp_path):
    """A single-entry ensemble must reduce to score_single exactly, per strategy.

    logits_sum sums raw logits (one term -> identity). log_prob_sum and
    consensus_veto (which falls back to log_prob_sum on agreement, and a
    lone model always agrees with itself) sum log-softmax, which is a
    per-vector monotonic reshaping of the logits, not the logits themselves;
    argmax is invariant under it, but the values are not, so they are
    checked against solo["log_probs"], not solo["logits"].
    """
    rt = _rt(tmp_path, "a.npz", in_dim=24, d=12, rank=3)
    rng = np.random.default_rng(3)
    q, C = rng.normal(size=24).astype(np.float32), rng.normal(size=(4, 24)).astype(np.float32)
    solo = score_single(rt, q, C)
    expected = {"logits_sum": solo["logits"], "log_prob_sum": solo["log_probs"],
                "consensus_veto": solo["log_probs"]}
    for strategy in ("logits_sum", "log_prob_sum", "consensus_veto"):
        out = score_ensemble({"m": rt}, {"m": {"query": q, "candidates": C}}, strategy)
        np.testing.assert_allclose(out["fused_logits"], expected[strategy], atol=1e-6, rtol=0)
        assert out["pred"] == solo["pred"]
        assert out["agreement"] is True


def test_two_identical_adapters_logits_sum_is_double(tmp_path):
    rt = _rt(tmp_path, "a.npz", in_dim=20, d=10, rank=3, seed=7)
    rng = np.random.default_rng(4)
    q, C = rng.normal(size=20).astype(np.float32), rng.normal(size=(5, 20)).astype(np.float32)
    single = score_single(rt, q, C)
    out = score_ensemble({"x": rt, "y": rt}, {"x": {"query": q, "candidates": C},
                                               "y": {"query": q, "candidates": C}}, "logits_sum")
    np.testing.assert_allclose(out["fused_logits"], 2.0 * single["logits"], atol=1e-5, rtol=0)
    assert out["agreement"] is True


def test_two_identical_adapters_log_prob_sum_is_double_log_softmax(tmp_path):
    rt = _rt(tmp_path, "a.npz", in_dim=20, d=10, rank=3, seed=8)
    rng = np.random.default_rng(5)
    q, C = rng.normal(size=20).astype(np.float32), rng.normal(size=(6, 20)).astype(np.float32)
    single = score_single(rt, q, C)
    out = score_ensemble({"x": rt, "y": rt}, {"x": {"query": q, "candidates": C},
                                               "y": {"query": q, "candidates": C}}, "log_prob_sum")
    np.testing.assert_allclose(out["fused_logits"], 2.0 * single["log_probs"], atol=1e-5, rtol=0)


def test_consensus_veto_agrees_uses_log_prob_sum(tmp_path):
    """Two models that predict the same class: consensus_veto == log_prob_sum's argmax."""
    rt_a = _rt(tmp_path, "a.npz", in_dim=16, d=8, rank=2, seed=10)
    rt_b = _rt(tmp_path, "b.npz", in_dim=16, d=8, rank=2, seed=10)  # identical -> always agrees
    rng = np.random.default_rng(6)
    q, C = rng.normal(size=16).astype(np.float32), rng.normal(size=(4, 16)).astype(np.float32)
    veto = score_ensemble({"a": rt_a, "b": rt_b}, {"a": {"query": q, "candidates": C},
                                                    "b": {"query": q, "candidates": C}}, "consensus_veto")
    lps = score_ensemble({"a": rt_a, "b": rt_b}, {"a": {"query": q, "candidates": C},
                                                   "b": {"query": q, "candidates": C}}, "log_prob_sum")
    assert veto["agreement"] is True
    assert veto["pred"] == lps["pred"]


def test_consensus_veto_disagrees_differs_from_logits_sum(tmp_path):
    """Construct a case where the two models disagree and check veto != naive sum in general.

    consensus_veto must be a genuinely different rule from logits_sum, not a
    relabeling of it: when the two models disagree, veto picks the single
    most-confident model's own prediction rather than summing evidence.
    """
    rt_a = _rt(tmp_path, "a.npz", in_dim=12, d=6, rank=2, seed=20)
    rt_b = _rt(tmp_path, "b.npz", in_dim=12, d=6, rank=2, seed=21)
    rng = np.random.default_rng(9)
    disagreement_found = False
    for trial in range(200):
        q = rng.normal(size=12).astype(np.float32)
        C = rng.normal(size=(4, 12)).astype(np.float32)
        out_a = score_single(rt_a, q, C)
        out_b = score_single(rt_b, q, C)
        if out_a["pred"] == out_b["pred"]:
            continue
        disagreement_found = True
        veto = score_ensemble({"a": rt_a, "b": rt_b}, {"a": {"query": q, "candidates": C},
                                                        "b": {"query": q, "candidates": C}},
                               "consensus_veto")
        assert veto["agreement"] is False
        from gen_zero.causal.ensemble_causal_engine import _prob_margin
        best = "a" if _prob_margin(out_a["probs"]) >= _prob_margin(out_b["probs"]) else "b"
        expected_pred = out_a["pred"] if best == "a" else out_b["pred"]
        assert veto["pred"] == expected_pred
        break
    assert disagreement_found, "no disagreement found in 200 trials; test is not exercising the veto branch"


# ---------------------------------------------------------------------------
# Independence: different in_dim per model, isolated failures
# ---------------------------------------------------------------------------

def test_two_models_different_in_dim_run_independently(tmp_path):
    rt_qwen = _rt(tmp_path, "qwen.npz", in_dim=64, d=16, rank=4, seed=30)
    rt_gemma = _rt(tmp_path, "gemma.npz", in_dim=96, d=24, rank=4, seed=31)
    rng = np.random.default_rng(11)
    K = 5
    q_qwen, C_qwen = rng.normal(size=64).astype(np.float32), rng.normal(size=(K, 64)).astype(np.float32)
    q_gemma, C_gemma = rng.normal(size=96).astype(np.float32), rng.normal(size=(K, 96)).astype(np.float32)
    for strategy in ("logits_sum", "log_prob_sum", "consensus_veto"):
        out = score_ensemble(
            {"qwen": rt_qwen, "gemma": rt_gemma},
            {"qwen": {"query": q_qwen, "candidates": C_qwen},
             "gemma": {"query": q_gemma, "candidates": C_gemma}},
            strategy,
        )
        assert out["fused_logits"].shape == (K,)
        assert set(out["preds"]) == {"qwen", "gemma"}


def test_mismatched_own_input_dim_raises_and_names_model(tmp_path):
    rt_qwen = _rt(tmp_path, "qwen.npz", in_dim=64, d=16, rank=4, seed=32)
    rt_gemma = _rt(tmp_path, "gemma.npz", in_dim=96, d=24, rank=4, seed=33)
    rng = np.random.default_rng(12)
    q_qwen, C_qwen = rng.normal(size=64).astype(np.float32), rng.normal(size=(4, 64)).astype(np.float32)
    wrong_q_gemma = rng.normal(size=64).astype(np.float32)  # wrong dim for gemma (needs 96)
    wrong_C_gemma = rng.normal(size=(4, 64)).astype(np.float32)
    with pytest.raises(ValueError, match="gemma"):
        score_ensemble(
            {"qwen": rt_qwen, "gemma": rt_gemma},
            {"qwen": {"query": q_qwen, "candidates": C_qwen},
             "gemma": {"query": wrong_q_gemma, "candidates": wrong_C_gemma}},
            "logits_sum",
        )


def test_mismatched_candidate_count_across_models_raises(tmp_path):
    rt_a = _rt(tmp_path, "a.npz", in_dim=16, d=8, rank=2, seed=40)
    rt_b = _rt(tmp_path, "b.npz", in_dim=16, d=8, rank=2, seed=41)
    rng = np.random.default_rng(13)
    q = rng.normal(size=16).astype(np.float32)
    with pytest.raises(ValueError, match="candidate slots"):
        score_ensemble(
            {"a": rt_a, "b": rt_b},
            {"a": {"query": q, "candidates": rng.normal(size=(4, 16)).astype(np.float32)},
             "b": {"query": q, "candidates": rng.normal(size=(6, 16)).astype(np.float32)}},
            "logits_sum",
        )


def test_unknown_fusion_strategy_rejected(tmp_path):
    rt = _rt(tmp_path, "a.npz", in_dim=8, d=4, rank=2)
    q, C = np.zeros(8, np.float32), np.ones((3, 8), np.float32)
    with pytest.raises(ValueError, match="fusion_strategy"):
        score_ensemble({"m": rt}, {"m": {"query": q, "candidates": C}}, "nonsense_strategy")


def test_load_adapters_names_failing_model(tmp_path):
    good = _build_adapter_npz(tmp_path / "good.npz", in_dim=8, d=4, rank=2)
    with pytest.raises(ValueError, match="broken"):
        load_adapters({"good": good, "broken": tmp_path / "does_not_exist.npz"})


# ---------------------------------------------------------------------------
# Permutation equivariance of the fused output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["logits_sum", "log_prob_sum", "consensus_veto"])
def test_permutation_equivariance_of_fusion(tmp_path, strategy):
    rt_a = _rt(tmp_path, "a.npz", in_dim=20, d=10, rank=3, seed=50)
    rt_b = _rt(tmp_path, "b.npz", in_dim=20, d=10, rank=3, seed=51)
    rng = np.random.default_rng(14)
    q = rng.normal(size=20).astype(np.float32)
    C = rng.normal(size=(7, 20)).astype(np.float32)
    perm = rng.permutation(7)

    out1 = score_ensemble({"a": rt_a, "b": rt_b}, {"a": {"query": q, "candidates": C},
                                                    "b": {"query": q, "candidates": C}}, strategy)
    out2 = score_ensemble({"a": rt_a, "b": rt_b}, {"a": {"query": q, "candidates": C[perm]},
                                                    "b": {"query": q, "candidates": C[perm]}}, strategy)
    np.testing.assert_allclose(out2["fused_logits"], out1["fused_logits"][perm], atol=1e-5, rtol=0)
    inv = np.argsort(perm)
    assert out2["pred"] == inv[out1["pred"]]


# ---------------------------------------------------------------------------
# Lossless full-dimension configs (no compression: in_dim == adapter_dim)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [64, 512])
def test_lossless_full_dimension_config_runs(tmp_path, dim):
    """in_dim == d (adapter_dim): the projection is square, not a bottleneck."""
    rt = _rt(tmp_path, f"lossless_{dim}.npz", in_dim=dim, d=dim, rank=8, n_layers=1, ffn_dim=dim)
    rng = np.random.default_rng(15)
    q, C = rng.normal(size=dim).astype(np.float32), rng.normal(size=(4, dim)).astype(np.float32)
    result = score_single(rt, q, C)
    assert result["logits"].shape == (4,)
    assert np.all(np.isfinite(result["logits"]))


def test_lossless_full_dimension_4096_runs_and_reports_time(tmp_path):
    """Full 4096-D lossless config (matches the real Qwen3.5-9B adapter's in_dim).

    n_layers=1 and ffn_dim=d keep the weight file under ~1 GB; this is a
    correctness + timing check, not a latency assertion (see the CPU-cost
    docstring in rnn_set_adapter.py: the set block is O(K^2*d + K*d^2), which
    at d=4096 is not microsecond-scale).
    """
    dim = 4096
    rt = _rt(tmp_path, "lossless_4096.npz", in_dim=dim, d=dim, rank=16, n_layers=1, ffn_dim=dim)
    rng = np.random.default_rng(16)
    q, C = rng.normal(size=dim).astype(np.float32), rng.normal(size=(4, dim)).astype(np.float32)
    t0 = time.perf_counter()
    result = score_single(rt, q, C)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    print(f"\n[timing] in_dim=d=4096 K=4 single score_single: {elapsed_ms:.2f} ms")
    assert result["logits"].shape == (4,)
    assert np.all(np.isfinite(result["logits"]))


# ---------------------------------------------------------------------------
# Latency: measured at a realistic (non-lossless) adapter_dim, not asserted at full dim
# ---------------------------------------------------------------------------

def test_cpu_latency_ms_scale_at_realistic_adapter_dim(tmp_path):
    """Matches the shipped qwen35_9b adapter's shape (in_dim=4096, d=256).

    The FLOP count at this shape (~1-8M multiply-adds for the projection) is
    sub-millisecond on dedicated hardware. Measured on THIS box the wall
    time is much higher and noisy (observed 300-400ms with loadavg over 100
    on 24 cores during development): this is host contention, not an O(n^2)
    bug in the adapter, and it is not safe to assert a tight ms-scale bound
    here without making the suite flaky. The bound below only catches a
    real algorithmic regression (e.g. an accidental O(dim^3) path); it does
    not certify the "1-10ms" target from a dedicated/idle machine, which
    this test cannot measure honestly on a shared host.
    """
    rt = _rt(tmp_path, "latency.npz", in_dim=4096, d=256, rank=16, n_layers=1, ffn_dim=512)
    rng = np.random.default_rng(17)
    q, C = rng.normal(size=4096).astype(np.float32), rng.normal(size=(8, 4096)).astype(np.float32)
    for _ in range(3):  # warm-up (page faults, first-call numpy dispatch)
        score_single(rt, q, C)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        score_single(rt, q, C)
        times.append((time.perf_counter() - t0) * 1e3)
    median_ms = float(np.median(times))
    print(f"\n[timing] in_dim=4096 d=256 K=8 median score_single: {median_ms:.3f} ms "
          f"(loadavg={np.round(np.array(__import__('os').getloadavg()), 1).tolist()}, "
          f"cores={__import__('os').cpu_count()})")
    assert median_ms < 5000.0, f"unexpectedly slow even accounting for host contention: {median_ms:.3f} ms"


# ---------------------------------------------------------------------------
# Real qwen35_9b adapter + real features (skipped if artifacts are absent)
# ---------------------------------------------------------------------------

_REAL_ADAPTER = REPO / "artifacts" / "qwen35_9b" / "zero_rnn_set_adapter_qwen35_9b.npz"
_REAL_FEATURES = REPO / "artifacts" / "qwen35_9b" / "parity_val200.npz"


@pytest.mark.skipif(not (_REAL_ADAPTER.is_file() and _REAL_FEATURES.is_file()),
                     reason="real qwen35_9b adapter/features artifacts not present")
def test_ensemble_single_model_matches_solo_on_real_qwen35_9b_data():
    rt = RNNSetAdapterRuntime.from_npz(_REAL_ADAPTER)
    z = np.load(_REAL_FEATURES, allow_pickle=True)
    q, cands, offsets, labels = z["q"], z["cands"], z["offsets"], z["labels"]
    correct = 0
    for i in range(len(q)):
        C = cands[offsets[i]:offsets[i + 1]]
        solo = score_single(rt, q[i], C)
        ens = score_ensemble({"qwen35_9b": rt}, {"qwen35_9b": {"query": q[i], "candidates": C}},
                              "logits_sum")
        assert ens["pred"] == solo["pred"]
        np.testing.assert_allclose(ens["fused_logits"], solo["logits"], atol=1e-6, rtol=0)
        correct += int(solo["pred"] == int(labels[i]))
    acc = correct / len(q)
    assert acc > 0.3, f"real-data sanity: accuracy {acc:.3f} implausibly low"
