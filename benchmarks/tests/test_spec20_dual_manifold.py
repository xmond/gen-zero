"""Spec 20 P4: dual-manifold orthogonal innovation projection (sota_enhanced_heads.DualManifoldHead).

Covers, per Spec 20 S3.3 / S6.2 and the P4 task sheet:
  1. fold identity: the two-GEMV folded logits equal the explicit chain (materialize z, then the top
     probe's own affine map) to atol 1e-5 in float64, with gate != 1 and lambda > 0 so the cross
     term -g W_zG R_G^T B^T P_Q^T is exercised; the production float32 path is checked separately;
  2. orthogonal innovation: Z_Q^T E_G == 0 (float64, tight) at lambda = 0, and Z_Q^T E_G == lambda B
     at lambda > 0 (S3.3: ridge residuals are NOT exactly orthogonal, and the head does not claim so);
     R_G has orthonormal columns; ranks truncate to n_fit - 1 with no zero padding;
  3. NumPy-only inference: load() + scores() run with torch import blocked, in-process and in a
     fresh interpreter;
  4. input defense: NaN/Inf, width/row mismatch, wrong K, bad labels, bad arrays, bad configs;
  5. row isolation through _fit_predict (NaN sentinel rows outside `tr` in BOTH sources);
  6. ensemble wiring: expert_specs registers the expert once with both sources, expert_query
     concatenates, parse_dual_manifold validates, save/load round-trips with and without the chain.
The pure-NumPy tests need neither torch nor sklearn (the top probe is injected); the sklearn-backed
fit tests skip without sklearn.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(SUITES))

import benchmark_sota_ensemble as bse  # noqa: E402
import sota_enhanced_heads as eh  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402

Head = eh.DualManifoldHead
Cfg = eh.DualManifoldConfig


HIDDEN = {}   # seed -> the Gemma-only binary factor of the last _sources() call with that seed


def _sources(n=90, D_Q=64, D_G=40, K=4, seed=0, pair=False, shared=0.7):
    """Two correlated sources: Gemma = shared latent through Qwen + its own innovation, which
    carries a binary factor `hidden` (independent of y) that Qwen never sees."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 12))
    y = rng.integers(0, K, size=n)
    latent[:, 0] += 1.5 * y                        # label signal in the shared latent
    innov = rng.normal(size=(n, 6))
    hidden = rng.integers(0, 2, size=n)
    innov[:, 0] += 2.0 * hidden                    # signal only Gemma carries
    HIDDEN[seed] = hidden
    X_Q = latent @ rng.normal(size=(12, D_Q)) + 0.3 * rng.normal(size=(n, D_Q)) + rng.normal(size=D_Q) * 3
    X_G = (shared * latent @ rng.normal(size=(12, D_G)) + innov @ rng.normal(size=(6, D_G))
           + 0.3 * rng.normal(size=(n, D_G)) + rng.normal(size=D_G) * 2)
    if pair:
        X_Q = np.concatenate([X_Q, rng.normal(size=(n, 2 * D_Q))], axis=1)
        X_G = np.concatenate([X_G, rng.normal(size=(n, 2 * D_G))], axis=1)
    return X_Q.astype(np.float32), X_G.astype(np.float32), y.astype(np.int64)


def _random_probe(z: np.ndarray, y: np.ndarray, K: int, seed: int = 0) -> sx.LinearProbe:
    """A `lin_full` LinearProbe on z with random weights and z's own standardization: no sklearn."""
    rng = np.random.default_rng(seed)
    kz = z.shape[1]
    arrays = {"mu_full": z.mean(0).astype(np.float32), "sd_full": (z.std(0) + 1e-6).astype(np.float32),
              "W": (rng.normal(size=(K, kz)) / np.sqrt(kz)).astype(np.float32),
              "b": rng.normal(size=K).astype(np.float32)}
    return sx.LinearProbe(arrays, {"K": K, "pair": False, "kind": "full", "pca_k": None, "in_dim": kz,
                                    "C": 1.0, "feature_dim": kz})


def _fit(cfg: Cfg, pair=False, seed=0, **src):
    X_Q, X_G, y = _sources(pair=pair, seed=seed, **src)
    K = int(y.max()) + 1
    head = Head.fit(X_Q, X_G, y, K, pair=pair, seed=seed, config=cfg,
                    probe_fit=lambda z, yy: _random_probe(z, yy, K, seed))
    return head, X_Q, X_G, y


# ------------------------------------------------------------- 1. fold identity

@pytest.mark.parametrize("pair", [False, True])
def test_folded_logits_equal_explicit_chain_float64(pair):
    cfg = Cfg(k_qwen=20, k_gemma=8, target_dim=64, regularization=0.3, gate=0.7)
    head, X_Q, X_G, y = _fit(cfg, pair=pair)
    Xq_new, Xg_new, _ = _sources(n=37, seed=9, pair=pair)
    fold = eh.fold_dual_manifold(head.u, head.gate, np.float64)
    fq, _, _ = sx.split_source(Xq_new.astype(np.float64), pair)
    fg, _, _ = sx.split_source(Xg_new.astype(np.float64), pair)
    folded = fq @ fold["W_fold_Q"].T + fg @ fold["W_fold_G"].T + fold["b_fold"]
    explicit = head.scores_unfolded(Xq_new, Xg_new)
    z = head.fused_features(Xq_new, Xg_new)
    u = head.u
    by_hand = ((z - u["z_mu"]) / u["z_sd"]) @ u["W_top"].T + u["b_top"]   # W_z z + b_z literally
    assert z.shape == (37, head.k_q + head.k_g)
    assert np.max(np.abs(folded - explicit)) < 1e-5
    assert np.max(np.abs(folded - by_hand)) < 1e-5
    assert np.max(np.abs(explicit)) > 0.5                         # the identity is not 0 == 0
    # The production float32 two-GEMV path against the float64 chain.
    got = head.scores(Xq_new, Xg_new)
    assert got.dtype == np.float32 and got.shape == (37, head.K)
    assert np.max(np.abs(got - explicit)) <= 1e-4 * (1.0 + np.max(np.abs(explicit)))
    assert head.info["export_max_abs_err"] <= 1e-4 * (1.0 + np.max(np.abs(head.scores_unfolded(X_Q, X_G))))


def test_gate_and_cross_term_matter():
    """gate scales the Gemma block AND the cross term; g = 0 must reduce to a pure Qwen-subspace head."""
    h1, X_Q, X_G, _ = _fit(Cfg(k_qwen=20, k_gemma=8, target_dim=64, regularization=0.1, gate=1.0))
    h0, _, _, _ = _fit(Cfg(k_qwen=20, k_gemma=8, target_dim=64, regularization=0.1, gate=0.0))
    assert np.all(h0.a["W_fold_G"] == 0.0)
    assert np.any(h1.a["W_fold_G"] != 0.0)
    # Cross term is present: W_fold_Q differs from the direct Qwen part W_zQ P_Q^T / sd_Q.
    u = h1.u
    W_eff = u["W_top"] / u["z_sd"]
    direct = (u["P_Q"] @ W_eff[:, :h1.k_q].T).T / u["sd_Q"]
    assert np.max(np.abs(h1.a["W_fold_Q"] - direct)) > 1e-3


# ------------------------------------------------------ 2. orthogonal innovation

def _chain_on_fit_rows(head, X_Q, X_G):
    u = head.u
    fq, _, _ = sx.split_source(X_Q.astype(np.float64), head.pair)
    fg, _, _ = sx.split_source(X_G.astype(np.float64), head.pair)
    Z_Q = ((fq - u["mu_Q"]) / u["sd_Q"]) @ u["P_Q"]
    Xgs = (fg - u["mu_G"]) / u["sd_G"]
    E_G = Xgs - Z_Q @ u["B"]
    return Z_Q, Xgs, E_G


def test_innovation_is_exactly_orthogonal_at_lambda_zero():
    head, X_Q, X_G, _ = _fit(Cfg(k_qwen=20, k_gemma=8, target_dim=64, regularization=0.0))
    Z_Q, Xgs, E_G = _chain_on_fit_rows(head, X_Q, X_G)
    X_hat = Z_Q @ head.u["B"]
    scale = np.linalg.norm(Z_Q, axis=0).max() * np.linalg.norm(Xgs, axis=0).max()
    assert np.max(np.abs(Z_Q.T @ E_G)) < 1e-9 * scale             # Z_Q^T E_G == 0 (OLS)
    assert np.max(np.abs(X_hat.T @ E_G)) < 1e-9 * scale           # hence X_hat _|_ E_G
    assert np.allclose(X_hat + E_G, Xgs)
    assert head.info["orthogonality_identity_max_abs_err"] < 1e-9 * scale
    # Whitened Z_Q: unit variance per column on the fit rows.
    n = len(Z_Q)
    assert np.allclose(Z_Q.T @ Z_Q / (n - 1), np.eye(head.k_q), atol=1e-8)


def test_ridge_residual_satisfies_ZQt_EG_equals_lambda_B_not_zero():
    lam = 0.5
    head, X_Q, X_G, _ = _fit(Cfg(k_qwen=20, k_gemma=8, target_dim=64, regularization=lam))
    Z_Q, _, E_G = _chain_on_fit_rows(head, X_Q, X_G)
    cross = Z_Q.T @ E_G
    assert np.allclose(cross, lam * head.u["B"], atol=1e-9)        # S3.3 identity
    assert np.max(np.abs(cross)) > 1e-3                            # and it is NOT zero under ridge
    assert head.info["orthogonality_identity_max_abs_err"] < 1e-9


def test_R_G_orthonormal_and_ranks_truncate_without_padding():
    # n_fit = 30 < budgets: ranks must drop to what the rows identify, never zero-padded, and the
    # n - 1 = 29 identifiable directions are split in the budgets' 3:1 ratio (22 + 7).
    head, X_Q, X_G, _ = _fit(Cfg(k_qwen=1536, k_gemma=512, target_dim=2048), n=30)
    R = head.u["R_G"]
    assert np.allclose(R.T @ R, np.eye(head.k_g), atol=1e-10)
    assert Head.rank_caps(30, 1536, 512) == (22, 7)
    assert head.k_q + head.k_g <= 29 and head.k_q <= 22 and 1 <= head.k_g <= 7
    assert head.u["P_Q"].shape == (64, head.k_q) and np.all(np.linalg.norm(head.u["P_Q"], axis=0) > 0)
    assert head.info["actual_projection_rank"] == [head.k_q, head.k_g]
    assert head.info["rank_truncated"] is True and head.info["budget_rank"] == [1536, 512]
    assert head.info["rank_cap_by_rows"] == [22, 7]
    assert all(r >= 1e-3 for r in head.info["singular_value_ratio_kept"])
    # A real-sized fold: 750 rows under the default budgets identify at most 562 + 187 directions.
    assert Head.rank_caps(750, 1536, 512) == (562, 187)
    assert Head.rank_caps(3, 1536, 512) == (1, 1)
    assert head.supervised_param_count() == head.K * (head.k_q + head.k_g + 1)
    assert head.exported_param_count() == head.K * (64 + 40 + 1)
    assert 0.0 < head.info["gemma_variance_explained_by_qwen"] < 1.0


def test_innovation_carries_signal_qwen_lacks():
    """The generator's Gemma-only factor: E_G R_G separates it, Z_Q does not (complementarity evidence)."""
    head, X_Q, X_G, y = _fit(Cfg(k_qwen=12, k_gemma=6, target_dim=64, regularization=0.0), n=300)
    z = head.fused_features(X_Q, X_G)
    zq, zg = z[:, :head.k_q], z[:, head.k_q:]
    par = HIDDEN[0].astype(bool)

    def sep(F):
        d = F[par].mean(0) - F[~par].mean(0)
        return float(np.abs(d).max() / (F.std(0).max() + 1e-12))
    assert sep(zg) > 1.0 > 3 * sep(zq)


# ------------------------------------------------------- 3. NumPy-only inference

def test_load_and_score_run_with_torch_import_blocked(tmp_path, monkeypatch):
    head, X_Q, X_G, _ = _fit(Cfg(k_qwen=16, k_gemma=8, target_dim=64))
    path = tmp_path / "dm.npz"
    head.save(path)                                       # folded only
    ref = head.scores(X_Q, X_G)
    monkeypatch.setitem(sys.modules, "torch", None)       # any `import torch` now raises ImportError
    loaded = Head.load(path)
    assert loaded.u is None
    assert np.array_equal(loaded.scores(X_Q, X_G), ref)
    assert np.array_equal(loaded.scores_concat(np.concatenate([X_Q, X_G], axis=1)), ref)
    # 1-D input goes through GEMV, not GEMM: BLAS summation order differs in the last float32 bits.
    assert np.allclose(loaded.score(np.concatenate([X_Q[0], X_G[0]]), np.zeros((head.K, 3))), ref[0], rtol=1e-5, atol=1e-6)
    with pytest.raises(ImportError):
        import torch as _t  # noqa: F401
    with pytest.raises(ValueError, match="unfolded chain was not kept"):
        loaded.fused_features(X_Q, X_G)


def test_module_import_and_inference_in_a_fresh_process_without_torch(tmp_path):
    head, X_Q, X_G, _ = _fit(Cfg(k_qwen=16, k_gemma=8, target_dim=64))
    path = tmp_path / "dm.npz"
    head.save(path, include_unfolded=True)
    np.savez(tmp_path / "x.npz", X_Q=X_Q, X_G=X_G, ref=head.scores(X_Q, X_G), ref_u=head.scores_unfolded(X_Q, X_G))
    code = ("import sys; sys.modules['torch'] = None; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "import numpy as np, sota_enhanced_heads as eh; h = eh.DualManifoldHead.load(%r); "
            "d = np.load(%r); assert np.array_equal(h.scores(d['X_Q'], d['X_G']), d['ref']); "
            "assert np.allclose(h.scores_unfolded(d['X_Q'], d['X_G']), d['ref_u']); "
            "assert 'torch' not in [m for m in sys.modules if sys.modules[m] is not None]; print('OK', h.k_q, h.k_g)"
            % (str(REPO / "python"), str(SUITES), str(path), str(tmp_path / "x.npz")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK")


def test_save_load_round_trip_with_unfolded_chain(tmp_path):
    head, X_Q, X_G, _ = _fit(Cfg(k_qwen=16, k_gemma=8, target_dim=64, gate=0.5, regularization=0.2))
    path = tmp_path / "dm_full.npz"
    head.save(path, include_unfolded=True)
    loaded = Head.load(path)
    assert loaded.cfg == head.cfg and loaded.gate == 0.5
    for k in Head.FOLD_KEYS:
        assert np.array_equal(loaded.a[k], head.a[k])
    for k in Head.UNFOLD_KEYS:
        assert np.array_equal(loaded.u[k], head.u[k])
    assert np.allclose(loaded.scores_unfolded(X_Q, X_G), head.scores_unfolded(X_Q, X_G))
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {"cfg_json", *Head.FOLD_KEYS, *(f"unfolded__{k}" for k in Head.UNFOLD_KEYS)}
    with pytest.raises(ValueError, match="cannot load"):
        eh.FoldedResidualAdapterBHead.load(path)


# ----------------------------------------------------------------- 4. defense

def test_scores_reject_bad_inputs():
    head, X_Q, X_G, _ = _fit(Cfg(k_qwen=16, k_gemma=8, target_dim=64))
    bad = X_Q.copy(); bad[3, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        head.scores(bad, X_G)
    bad = X_G.copy(); bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        head.scores(X_Q, bad)
    with pytest.raises(ValueError, match="widths"):
        head.scores(X_Q[:, :-1], X_G)
    with pytest.raises(ValueError, match="widths"):
        head.scores(X_Q, X_G[:, :-1])
    with pytest.raises(ValueError, match="rows"):
        head.scores(X_Q[:5], X_G[:4])
    with pytest.raises(ValueError, match="both be 1-D or both 2-D"):
        head.scores(X_Q[0], X_G)
    with pytest.raises(ValueError, match="concatenated"):
        head.scores_concat(X_Q)
    with pytest.raises(ValueError, match="K="):
        head.score(np.concatenate([X_Q[0], X_G[0]]), np.zeros((head.K + 1, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        head.fused_features(bad if False else np.full_like(X_Q, np.nan), X_G)


def test_fit_rejects_bad_inputs():
    X_Q, X_G, y = _sources()
    K = int(y.max()) + 1
    pf = lambda z, yy: _random_probe(z, yy, K)  # noqa: E731
    cfg = Cfg(k_qwen=8, k_gemma=4, target_dim=16)
    nan_q = X_Q.copy(); nan_q[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Head.fit(nan_q, X_G, y, K, pair=False, seed=0, config=cfg, probe_fit=pf)
    with pytest.raises(ValueError, match="share N"):
        Head.fit(X_Q, X_G[:-1], y, K, pair=False, seed=0, config=cfg, probe_fit=pf)
    with pytest.raises(ValueError, match="share N"):
        Head.fit(X_Q, X_G, y[:-1], K, pair=False, seed=0, config=cfg, probe_fit=pf)
    with pytest.raises(ValueError, match="2-D"):
        Head.fit(X_Q[0], X_G[0], y[:1], K, pair=False, seed=0, config=cfg, probe_fit=pf)
    with pytest.raises(ValueError, match=r"labels must be integers in \[0, 4\)"):
        Head.fit(X_Q, X_G, y + 1, K, pair=False, seed=0, config=cfg, probe_fit=pf)
    with pytest.raises(ValueError, match="labels must be integers"):
        Head.fit(X_Q, X_G, y.astype(np.float32), K, pair=False, seed=0, config=cfg, probe_fit=pf)
    with pytest.raises(ValueError, match="two classes"):
        Head.fit(X_Q, X_G, np.zeros_like(y), K, pair=False, seed=0, config=cfg, probe_fit=pf)
    with pytest.raises(TypeError, match="DualManifoldConfig"):
        Head.fit(X_Q, X_G, y, K, pair=False, seed=0, config=eh.AdapterBConfig(), probe_fit=pf)
    with pytest.raises(ValueError, match="pair source vector width"):
        Head.fit(X_Q, X_G, y, K, pair=True, seed=0, config=cfg, probe_fit=pf)
    # Gemma a linear image of the top-8 Qwen principal subspace: no innovation -> refuse, never
    # fake a dual head out of float noise.
    Xqs = (X_Q - X_Q.mean(0)) / (X_Q.std(0) + 1e-6)
    _, _, Vt = np.linalg.svd(Xqs.astype(np.float64), full_matrices=False)
    X_G_dep = ((Xqs @ Vt[:8].T) @ np.random.default_rng(1).normal(size=(8, 40))).astype(np.float32)
    with pytest.raises(ValueError, match="no innovation"):
        Head.fit(X_Q, X_G_dep, y, K, pair=False, seed=0, config=Cfg(k_qwen=8, k_gemma=4, target_dim=16,
                                                                    regularization=0.0), probe_fit=pf)


def test_config_validation():
    with pytest.raises(ValueError, match="exceeds target_dim"):
        Cfg(k_qwen=1536, k_gemma=513)
    for bad in ({"k_qwen": 0}, {"k_gemma": -1}, {"target_dim": 0}, {"k_qwen": True}, {"k_qwen": 2.0}):
        with pytest.raises(ValueError):
            Cfg(**bad)
    for bad in ({"regularization": -1e-3}, {"regularization": np.nan}, {"gate": np.inf}, {"gate": -0.5},
                {"rank_rtol": 0.0}, {"rank_rtol": 1.0}, {"rank_rtol": np.nan}):
        with pytest.raises(ValueError):
            Cfg(**bad)
    with pytest.raises(ValueError):
        Cfg(device="tpu")
    assert Cfg() == Cfg(k_qwen=1536, k_gemma=512, target_dim=2048, regularization=1e-4, gate=1.0, rank_rtol=1e-3,
                        device="cpu")
    assert eh.head_config({"type": "dual_manifold", "k_qwen": 256, "k_gemma": 128}) == Cfg(k_qwen=256, k_gemma=128)
    assert eh.HEADS["dual_manifold"] is Head and eh.CONFIGS["dual_manifold"] is Cfg


def test_constructor_validates_arrays_and_cfg():
    head, *_ = _fit(Cfg(k_qwen=16, k_gemma=8, target_dim=64))
    cfg, a = head.cfg, dict(head.a)
    with pytest.raises(ValueError, match="cannot load"):
        Head(a, dict(cfg, head_type="adapter_b"))
    with pytest.raises(ValueError, match="cfg missing"):
        Head(a, {k: v for k, v in cfg.items() if k != "D_G"})
    with pytest.raises(ValueError, match="arrays missing"):
        Head({k: v for k, v in a.items() if k != "b_fold"}, cfg)
    with pytest.raises(ValueError, match="has shape"):
        Head(dict(a, W_fold_G=a["W_fold_G"][:, :-1]), cfg)
    bad = a["W_fold_Q"].copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Head(dict(a, W_fold_Q=bad), cfg)
    with pytest.raises(ValueError, match="in_dim_g"):
        Head(a, dict(cfg, in_dim_g=cfg["in_dim_g"] + 1))
    with pytest.raises(ValueError, match="gate"):
        Head(a, dict(cfg, gate=-1.0))
    with pytest.raises(ValueError, match="unfolded arrays missing"):
        Head(a, cfg, {k: v for k, v in head.u.items() if k != "B"})
    with pytest.raises(ValueError, match="has shape"):
        Head(a, cfg, dict(head.u, B=head.u["B"][:-1]))


# ------------------------------------------------ 5. dispatcher: fit / isolation

def _fake_sources(n=80, seed=0):
    X_Q, X_G, y = _sources(n=n, seed=seed)
    K = int(y.max()) + 1
    cands = np.zeros((K, 8), dtype=np.float32)
    ids = np.arange(n)
    mk = lambda X: {"X_train": X, "train_full": X, "train_label": y, "cands": cands, "train_ids": ids,  # noqa: E731
                    "test_ids": ids[:3], "info": {}, "X_test": X[:3]}
    return {"qwen": mk(X_Q), "gemma": mk(X_G)}


def test_fit_predict_dual_branch_reads_both_sources_and_isolates_rows(monkeypatch):
    pytest.importorskip("sklearn")
    fs = _fake_sources()
    for s in fs.values():
        s["X_train"] = s["X_train"].copy()
        s["X_train"][:10] = np.nan                    # sentinel rows outside `tr`, in BOTH sources
    tr = np.arange(10, 80)
    spec = {"type": "dual_manifold", "sources": ["qwen", "gemma"], "k_qwen": 12, "k_gemma": 6, "target_dim": 32}
    with pytest.raises(ValueError, match="needs fs="):
        bse._fit_predict(spec, "massive_en", fs["qwen"], tr, tr[:20], None, [0, 0, 100])
    with pytest.raises(ValueError, match="Qwen source"):
        bse._fit_predict(spec, "massive_en", fs["gemma"], tr, tr[:20], None, [0, 0, 100], fs=fs)
    m, s, info = bse._fit_predict(spec, "massive_en", fs["qwen"], tr, tr[:20], None, [0, 0, 100], fs=fs)
    assert isinstance(m, Head) and s.shape == (20, 4) and np.all(np.isfinite(s))
    assert info["params"] == m.supervised_param_count() == 4 * (12 + 6 + 1)
    assert info["actual_projection_rank"] == [12, 6] and info["device"] == "cpu"
    assert np.array_equal(s, m.scores(fs["qwen"]["X_train"][tr[:20]], fs["gemma"]["X_train"][tr[:20]]))
    # Sentinel rows: scoring them must fail loudly, never silently produce numbers.
    with pytest.raises(ValueError, match="non-finite"):
        m.scores(fs["qwen"]["X_train"][:10], fs["gemma"]["X_train"][:10])
    # Real sklearn top probe: the fold identity still holds on fresh rows.
    Xq_new, Xg_new, _ = _sources(n=15, seed=5)
    assert np.max(np.abs(m.scores(Xq_new, Xg_new) - m.scores_unfolded(Xq_new, Xg_new))) <= 1e-4 * (
        1.0 + np.max(np.abs(m.scores_unfolded(Xq_new, Xg_new))))


def test_fit_predict_signature_stays_backward_compatible():
    """Existing tests call _fit_predict positionally without fs; the linear path must not change."""
    pytest.importorskip("sklearn")
    fs = _fake_sources(n=60)
    tr = np.arange(60)
    m, s, info = bse._fit_predict({"type": "linear", "kind": "full", "pca_k": None}, "massive_en", fs["qwen"], tr,
                                  tr[:5], fs["qwen"]["X_train"][:5], [0, 0, 0])
    assert isinstance(m, sx.LinearProbe) and s.shape == (5, 4)


# ------------------------------------------------------------ 6. dispatcher: wiring

def test_expert_specs_register_dual_manifold_once_with_both_sources(monkeypatch):
    fs = {"qwen": {"train_full": np.zeros((5, 16))}, "gemma": {"train_full": np.zeros((5, 8))}}
    monkeypatch.setattr(bse, "DUAL_MANIFOLD", None)
    assert not [n for n, _, _ in bse.expert_specs("massive_en", fs) if "dual" in n]
    monkeypatch.setattr(bse, "DUAL_MANIFOLD", ("qwen", "gemma"))
    specs = bse.expert_specs("massive_en", fs)
    dual = [(n, s, sp) for n, s, sp in specs if sp["type"] == "dual_manifold"]
    assert dual == [("qwen+gemma:dual_manifold", "qwen", {"type": "dual_manifold", "sources": ["qwen", "gemma"]})]
    assert eh.head_config(bse.head_spec(dual[0][2])) == Cfg()
    assert bse.safe(dual[0][0]) == "qwen+gemma__dual_manifold"
    monkeypatch.setattr(bse, "DUAL_MANIFOLD", ("qwen", "other"))
    with pytest.raises(ValueError, match="two DIFFERENT registered sources"):
        bse.expert_specs("massive_en", fs)
    monkeypatch.setattr(bse, "DUAL_MANIFOLD", ("qwen", "qwen"))
    with pytest.raises(ValueError, match="two DIFFERENT registered sources"):
        bse.expert_specs("massive_en", fs)


def test_expert_query_concatenates_only_for_dual():
    fs = _fake_sources(n=12)
    dual = {"type": "dual_manifold", "sources": ["qwen", "gemma"]}
    q = bse.expert_query(dual, "qwen", fs, "train")
    assert q.shape == (12, 64 + 40)
    assert np.array_equal(q[:, :64], fs["qwen"]["X_train"]) and np.array_equal(q[:, 64:], fs["gemma"]["X_train"])
    assert bse.expert_query(dual, "qwen", fs, "test").shape == (3, 104)
    assert bse.expert_query({"type": "linear"}, "gemma", fs, "train") is fs["gemma"]["X_train"]
    head, X_Q, X_G, _ = _fit(Cfg(k_qwen=8, k_gemma=4, target_dim=16), n=12)
    assert head.in_dim == 104                                   # what sx.EngineExpert reads
    eng = sx.EngineExpert("d", head, head.in_dim)
    assert np.array_equal(eng.score(q[0], fs["qwen"]["cands"]), head.scores(X_Q[:1], X_G[:1])[0])


def test_parse_dual_manifold():
    assert bse.parse_dual_manifold("", ["a", "b"]) is None
    assert bse.parse_dual_manifold(" a , b ", ["a", "b"]) == ("a", "b")
    for text in ("a", "a,a", "a,c", "a,b,c"):
        with pytest.raises(ValueError, match="--dual-manifold"):
            bse.parse_dual_manifold(text, ["a", "b"])


def test_cli_flag_reaches_stage_fit(monkeypatch, tmp_path):
    pytest.importorskip("torch")                          # main() imports torch for the fit stage
    for g in ("OUT_ART", "RESULTS", "PRIOR_REPORT", "ADAPTER_RANKS", "SUPCON_RANKS", "ADAPTER_B_RANKS",
              "DUAL_MANIFOLD", "N_FOLDS", "FOLD_SEED", "NESTED_SEED", "RAW_MAX_DIM", "MOE_THREADS", "ENABLE_RDA",
              "ENABLE_NYSTROM", "RNN_SOURCES", "RNN_HEADS"):
        monkeypatch.setattr(bse, g, getattr(bse, g))
    monkeypatch.setattr(bse, "SOURCES", dict(bse.SOURCES))
    monkeypatch.setattr(bse.gd, "TEST_DIR", bse.gd.TEST_DIR)
    seen = {}
    monkeypatch.setattr(bse, "stage_fit", lambda tasks, only, device: seen.update(dual=bse.DUAL_MANIFOLD))
    monkeypatch.setattr(bse.sx, "resolve_device", lambda d: "cpu")
    base = ["prog", "fit", "--tasks", "massive_en", "--source", f"q={tmp_path}", "--source", f"g={tmp_path}",
            "--out-art", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", base + ["--dual-manifold", "q,g"])
    bse.main()
    assert seen["dual"] == ("q", "g")
    monkeypatch.setattr(sys, "argv", base + ["--dual-manifold", "q,zz"])
    with pytest.raises(SystemExit, match="--dual-manifold"):
        bse.main()
