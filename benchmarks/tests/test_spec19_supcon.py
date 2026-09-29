"""Spec 19 Phase 2 direction 4: the SupCon joint-training head (sota_enhanced_heads.SupConHead).

Covers zero-init == LinearProbe (max abs err exactly 0.0), fold exactness < 1e-5 against the
unfolded torch forward, save/load bit-exact, NumPy-only inference with torch import blocked,
input / K / label / config defense, training-row isolation (NaN sentinels), the SupCon loss
itself (against a brute-force NumPy reference, its gradient at the zero init, its safe
degradation to CE) and the S6.2 pipeline wiring (expert_specs, _fit_predict, stage_fit,
strategy_complexity, load_expert).

All data is synthetic; no task files. Tests that call LinearProbe.fit need scikit-learn and
skip without it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(SUITES))

torch = pytest.importorskip("torch")
import benchmark_sota_ensemble as bse  # noqa: E402
import sota_enhanced_heads as eh  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402

torch.set_num_threads(2)


def _probe_from_arrays(D: int, K: int, seed: int, pair: bool = False) -> sx.LinearProbe:
    rng = np.random.default_rng(seed)
    arrays = {"mu_full": rng.normal(size=D).astype(np.float32),
              "sd_full": rng.uniform(0.5, 2.0, size=D).astype(np.float32),
              "W": rng.normal(size=(K, D)).astype(np.float32),
              "b": rng.normal(size=K).astype(np.float32)}
    cfg = {"K": K, "pair": pair, "kind": "full", "pca_k": None, "in_dim": 3 * D if pair else D,
           "C": 1.0, "feature_dim": D}
    return sx.LinearProbe(arrays, cfg)


def _random_trained_like(D=96, r=32, K=5, seed=0, fold=True):
    """A SupConHead with every parameter non-zero (as after training), plus its raw parameters."""
    rng = np.random.default_rng(seed)
    lp = _probe_from_arrays(D, K, seed)
    lp = sx.LinearProbe(dict(lp.a, W=(lp.a["W"] * 2.0 / np.sqrt(D)).astype(np.float32)), lp.cfg)
    p = {"W_down": rng.normal(scale=1 / np.sqrt(D), size=(r, D)), "b_down": rng.normal(scale=0.1, size=r),
         "W_up": rng.normal(scale=0.2, size=(D, r)), "b_up": rng.normal(scale=0.1, size=D),
         "W_h": lp.a["W"], "b_h": lp.a["b"]}
    p = {k: v.astype(np.float32) for k, v in p.items()}
    head = eh.SupConHead.from_unfolded(
        {"mu_full": lp.a["mu_full"], "sd_full": lp.a["sd_full"]}, p["W_down"], p["b_down"], p["W_up"],
        p["b_up"], p["W_h"], p["b_h"], K=K, pair=False, in_dim=D,
        config=eh.SupConConfig(rank=r, fold_for_inference=fold))
    return head, p, lp


def _xor_data(n, D=16, seed=0):
    """Label = sign(x0) XOR sign(x1): no linear function of x separates it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D)).astype(np.float32)
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)


# ------------------------------------------------------------ registration + config

def test_registered_under_the_supcon_key():
    assert eh.HEADS["supcon"] is eh.SupConHead and eh.CONFIGS["supcon"] is eh.SupConConfig
    assert eh.SupConHead.head_type == "supcon"
    assert eh.head_config({"type": "supcon", "rank": 32}) == eh.SupConConfig(rank=32)


def test_config_defaults_match_the_task_statement():
    c = eh.SupConConfig()
    assert (c.rank, c.lr, c.lambda_up, c.tau, c.lambda_supcon, c.feature_dropout, c.fold_for_inference,
            c.device) == (64, 1e-3, 1e-3, 0.1, 0.5, 0.1, True, "auto")


@pytest.mark.parametrize("kw", [
    {"rank": 0}, {"rank": -3}, {"rank": 1.5}, {"rank": True}, {"rank": None},
    {"lr": 0.0}, {"lr": float("nan")}, {"lr": float("inf")}, {"lr": None}, {"lr": "1e-3"},
    {"lambda_up": -1.0}, {"lambda_up": float("nan")},
    {"tau": 0.0}, {"tau": -0.1}, {"tau": float("nan")}, {"tau": float("inf")}, {"tau": None},
    {"lambda_supcon": -0.5}, {"lambda_supcon": float("nan")}, {"lambda_supcon": float("inf")},
    {"feature_dropout": -0.01}, {"feature_dropout": 1.0}, {"feature_dropout": float("nan")},
    {"feature_dropout": None},
    {"fold_for_inference": "yes"}, {"fold_for_inference": None},
    {"device": ""}, {"device": None}, {"device": "tpu"},
])
def test_config_rejects_illegal_values(kw):
    with pytest.raises(ValueError):
        eh.SupConConfig(**kw)


def test_config_accepts_the_boundaries():
    eh.SupConConfig(lambda_supcon=0.0, feature_dropout=0.0, lambda_up=0.0, device="cpu", rank=1)
    eh.SupConConfig(feature_dropout=0.999)


def test_fit_rejects_the_wrong_config_class_and_heads_reject_each_others_files(tmp_path):
    X, y = _xor_data(60, D=8)
    with pytest.raises(TypeError):
        eh.SupConHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=4))
    with pytest.raises(TypeError):
        eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.SupConConfig(rank=4))
    head, _, _ = _random_trained_like()
    head.save(tmp_path / "s.npz")
    with pytest.raises(ValueError):
        eh.DeepResidualAdapterHead.load(tmp_path / "s.npz")
    ad = eh.DeepResidualAdapterHead.from_linear_probe(_probe_from_arrays(16, 3, 0), eh.AdapterConfig(rank=4),
                                                      np.random.default_rng(0))
    ad.save(tmp_path / "a.npz")
    with pytest.raises(ValueError):
        eh.SupConHead.load(tmp_path / "a.npz")


# ------------------------------------------------ 1: zero init == linear probe

@pytest.mark.parametrize("fold", [True, False])
def test_zero_initialized_supcon_head_equals_the_linear_probe_exactly(fold):
    D, K = 256, 7
    lp = _probe_from_arrays(D, K, seed=1)
    head = eh.SupConHead.from_linear_probe(lp, eh.SupConConfig(rank=64, fold_for_inference=fold),
                                           np.random.default_rng(0))
    X = np.random.default_rng(2).normal(size=(200, D)).astype(np.float32) * 3
    assert isinstance(head, eh.SupConHead) and head.cfg["head_type"] == "supcon"
    assert np.any(head.a["W_down"] != 0)                   # the bottleneck is live, only W_up is zero
    err = float(np.max(np.abs(head.scores(X) - lp.scores(X))))
    assert err == 0.0, err


def test_fit_starts_from_the_probe_fitted_on_the_same_rows():
    pytest.importorskip("sklearn")
    X, y = _xor_data(300, seed=3)
    head = eh.SupConHead.fit(X, y, 2, pair=False, seed=0, config=eh.SupConConfig(rank=8, device="cpu"))
    assert head.info["identity_max_abs_err_at_init"] == 0.0
    assert isinstance(head, eh.SupConHead)


# ------------------------------------------------- 2: fold is an exact identity

@pytest.mark.parametrize("D,r,K", [(96, 32, 5), (512, 64, 18), (1024, 128, 3)])
def test_folded_numpy_inference_equals_the_unfolded_torch_forward(D, r, K):
    head, p, lp = _random_trained_like(D, r, K, seed=D)
    X = np.random.default_rng(5).normal(size=(64, D)).astype(np.float32)
    z = torch.as_tensor((X - lp.a["mu_full"]) / lp.a["sd_full"])
    net = {k: torch.as_tensor(v) for k, v in p.items()}
    with torch.no_grad():
        ref = eh.torch_unfolded_logits(net, z).numpy()
    err = float(np.max(np.abs(head.scores(X) - ref)))
    assert err < 1e-5, err
    assert float(np.max(np.abs(ref - lp.scores(X)))) > 1e-2   # the residual branch really contributes
    unfolded, _, _ = _random_trained_like(D, r, K, seed=D, fold=False)
    assert float(np.max(np.abs(unfolded.scores(X) - ref))) < 1e-5


def test_the_fold_formula_is_the_stated_algebra():
    head, p, _ = _random_trained_like(64, 16, 4, seed=7)
    W_h, W_up, b_up, b_h = (p[k].astype(np.float64) for k in ("W_h", "W_up", "b_up", "b_h"))
    assert np.allclose(head.a["W_fold"], W_h @ W_up, atol=1e-6)
    assert np.allclose(head.a["b_fold"], W_h @ b_up + b_h, atol=1e-6)


def test_trained_head_export_matches_its_torch_forward():
    pytest.importorskip("sklearn")
    X, y = _xor_data(600, seed=4)
    head = eh.SupConHead.fit(X, y, 2, pair=False, seed=0, config=eh.SupConConfig(rank=8, device="cpu"))
    assert head.info["export_max_abs_err_es"] < 1e-4


# ----------------------------------------------------------- 3: save / load

@pytest.mark.parametrize("fold", [True, False])
def test_save_load_roundtrip_is_bit_exact(tmp_path, fold):
    head, _, _ = _random_trained_like(fold=fold)
    head.save(tmp_path / "a.npz")
    with np.load(tmp_path / "a.npz", allow_pickle=False) as z:
        keys = set(z.files)
    want = {"W_h", "W_fold", "W_down", "b_down", "b_fold"} if fold else {"W_h", "W_up", "b_up", "b_h"}
    assert want | {"cfg_json", "mu_full", "sd_full"} <= keys
    back = eh.SupConHead.load(tmp_path / "a.npz")
    assert isinstance(back, eh.SupConHead) and back.cfg == head.cfg
    assert back.cfg["config"]["tau"] == 0.1 and back.cfg["head_type"] == "supcon"
    for k in head.a:
        assert back.a[k].dtype == np.float32 and np.array_equal(back.a[k], head.a[k]), k
    X = np.random.default_rng(6).normal(size=(50, head.in_dim)).astype(np.float32)
    assert np.array_equal(back.scores(X), head.scores(X))
    assert np.array_equal(back.score(X[0], np.zeros((head.K, 1))), head.scores(X[:1])[0])


# ---------------------------------------------------- 4: NumPy-only inference

def test_load_and_score_run_with_torch_import_blocked(tmp_path, monkeypatch):
    head, _, _ = _random_trained_like()
    head.save(tmp_path / "a.npz")
    X = np.random.default_rng(7).normal(size=(10, head.in_dim)).astype(np.float32)
    expect = head.scores(X)
    monkeypatch.setitem(sys.modules, "torch", None)      # any `import torch` now raises ImportError
    back = eh.SupConHead.load(tmp_path / "a.npz")
    out = back.scores(X)
    assert type(out) is np.ndarray and out.dtype == np.float32
    assert np.array_equal(out, expect)
    assert all(type(v) is np.ndarray for v in back.a.values())
    one = back.score(X[0], np.zeros((back.K, 1)))              # 1-D matvec: BLAS rounding, not bit-equal to the batch row
    assert np.array_equal(one, back.scores(X[0])) and float(np.max(np.abs(one - expect[0]))) < 1e-5
    with pytest.raises(ImportError):
        import torch as _t  # noqa: F401


def test_module_import_and_inference_in_a_fresh_process_without_torch(tmp_path):
    head, _, _ = _random_trained_like()
    head.save(tmp_path / "a.npz")
    X = np.random.default_rng(8).normal(size=(4, head.in_dim)).astype(np.float32)
    np.save(tmp_path / "x.npy", X)
    code = ("import sys; sys.modules['torch'] = None; sys.path.insert(0, %r); "
            "import numpy as np, sota_enhanced_heads as eh; "
            "h = eh.SupConHead.load(%r); np.save(%r, h.scores(np.load(%r))); "
            "print('torch' in sys.modules and sys.modules['torch'] is not None)"
            % (str(SUITES), str(tmp_path / "a.npz"), str(tmp_path / "o.npy"), str(tmp_path / "x.npy")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
    assert np.array_equal(np.load(tmp_path / "o.npy"), head.scores(X))


# ------------------------------------------------------------ 5: input defense

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_rejected(bad):
    head, _, _ = _random_trained_like()
    X = np.zeros((3, head.in_dim), dtype=np.float32)
    X[1, 5] = bad
    with pytest.raises(ValueError):
        head.scores(X)
    with pytest.raises(ValueError):
        head.score(X[1], np.zeros((head.K, 1)))
    with pytest.raises(ValueError):
        eh.SupConHead.fit(np.repeat(X, 20, 0), np.tile([0, 1, 0], 20), 2, pair=False, seed=0,
                          probe_fit=lambda *a: pytest.fail("probe must not be fitted"))


def test_wrong_width_wrong_k_and_bad_labels_are_rejected():
    head, _, _ = _random_trained_like()
    with pytest.raises(ValueError):
        head.scores(np.zeros((2, head.in_dim + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        head.scores(np.zeros((2, 2, head.in_dim), dtype=np.float32))
    with pytest.raises(ValueError):                                     # K + 1 candidates
        head.score(np.zeros(head.in_dim, dtype=np.float32), np.zeros((head.K + 1, 1)))
    with pytest.raises(ValueError):                                     # K - 1 candidates
        head.score(np.zeros(head.in_dim, dtype=np.float32), np.zeros((head.K - 1, 1)))
    X = np.zeros((40, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        eh.SupConHead.fit(X, np.full(40, 3), 3, pair=False, seed=0)          # label == K
    with pytest.raises(ValueError):
        eh.SupConHead.fit(X, np.full(40, -1), 3, pair=False, seed=0)         # negative label
    with pytest.raises(ValueError):
        eh.SupConHead.fit(X, np.zeros(40, dtype=np.float32), 3, pair=False, seed=0)   # non-integer dtype
    with pytest.raises(ValueError):
        eh.SupConHead.fit(X, np.zeros(39, dtype=int), 3, pair=False, seed=0)  # length mismatch
    with pytest.raises(ValueError):
        eh.SupConHead(dict(head.a, W_fold=head.a["W_fold"][:, :-1]), head.cfg)
    with pytest.raises(ValueError):
        eh.SupConHead(dict(head.a, W_h=np.where(np.arange(head.a["W_h"].size).reshape(head.a["W_h"].shape) == 3,
                                                np.nan, head.a["W_h"])), head.cfg)


# ---------------------------------------------------- 6: training data isolation

def test_training_reads_only_the_rows_passed():
    """NaN sentinel rows sit outside `tr`; _fit_predict must neither fail nor read them."""
    pytest.importorskip("sklearn")
    X, y = _xor_data(360, seed=9)
    X[::6] = np.nan
    tr = np.flatnonzero(np.arange(360) % 6 != 0)
    f = {"X_train": X, "train_label": y, "cands": np.zeros((2, 1), dtype=np.float32)}
    m, s, info = bse._fit_predict({"type": "supcon", "rank": 8}, "massive_en", f, tr, tr[:40], X[tr[:40]],
                                  [0, 0, 100])
    assert isinstance(m, eh.SupConHead)
    assert s.shape == (40, 2) and np.all(np.isfinite(s))
    assert info["n_train"] + info["n_early_stop"] == len(tr)
    assert info["supcon_batches"] >= 0 and info["params"] == m.supervised_param_count()


def test_the_warm_start_probe_never_sees_the_early_stop_rows():
    seen = {}

    def probe_fit(Xp, yp):
        seen["X"] = Xp.copy()
        return _probe_from_arrays(Xp.shape[1], 2, seed=0)

    X, y = _xor_data(200, D=8, seed=10)
    X += np.arange(200, dtype=np.float32)[:, None] * 1e-3        # make every row unique
    h = eh.SupConHead.fit(X, y, 2, pair=False, seed=0, config=eh.SupConConfig(rank=4, device="cpu"),
                          rng_key=[0, 1, 2], probe_fit=probe_fit)
    es = np.random.default_rng([0, 1, 2]).random(200) < sx.EARLY_STOP_FRACTION   # fit's first draw
    assert len(seen["X"]) == h.info["n_train"] == int((~es).sum())
    assert np.array_equal(seen["X"], X[~es])


# ---------------------------------------------------------- the SupCon loss itself

def _supcon_numpy(u1, u2, y, tau):
    """Brute-force reference: explicit loops over anchors, positives and denominators."""
    B = len(y)
    u = np.concatenate([u1, u2]).astype(np.float64)
    lab = np.concatenate([y, y])
    rec = np.concatenate([np.arange(B), np.arange(B)])
    losses = []
    for i in range(2 * B):
        others = [j for j in range(2 * B) if j != i]
        pos = [j for j in others if lab[j] == lab[i]]
        if not any(rec[j] != rec[i] for j in pos):
            continue                                            # singleton class: skipped anchor
        denom = sum(np.exp(u[i] @ u[a] / tau) for a in others)
        losses.append(-np.mean([np.log(np.exp(u[i] @ u[p] / tau) / denom) for p in pos]))
    return float(np.mean(losses)) if losses else None


def _unit(rng, n, d):
    v = rng.normal(size=(n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.mark.parametrize("tau", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("labels", [[0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 2], [0, 1, 0, 1, 0, 1]])
def test_supcon_loss_matches_the_brute_force_reference(tau, labels):
    rng = np.random.default_rng(3)
    u1, u2, y = _unit(rng, len(labels), 8), _unit(rng, len(labels), 8), np.array(labels)
    got = float(eh.supcon_loss(torch.as_tensor(u1), torch.as_tensor(u2), torch.as_tensor(y), tau))
    assert abs(got - _supcon_numpy(u1, u2, y, tau)) < 1e-5


def test_supcon_loss_skips_singleton_class_anchors():
    rng = np.random.default_rng(4)
    u1, u2, y = _unit(rng, 5, 6), _unit(rng, 5, 6), np.array([0, 0, 0, 1, 2])   # classes 1, 2 are singletons
    got = float(eh.supcon_loss(torch.as_tensor(u1), torch.as_tensor(u2), torch.as_tensor(y), 0.2))
    assert abs(got - _supcon_numpy(u1, u2, y, 0.2)) < 1e-5


def test_supcon_loss_returns_none_when_no_class_has_two_records():
    rng = np.random.default_rng(5)
    u1, u2 = _unit(rng, 4, 6), _unit(rng, 4, 6)
    assert eh.supcon_loss(torch.as_tensor(u1), torch.as_tensor(u2), torch.arange(4), 0.1) is None
    assert eh.supcon_loss(torch.as_tensor(u1[:1]), torch.as_tensor(u2[:1]), torch.zeros(1, dtype=torch.long),
                          0.1) is None


def test_supcon_loss_is_lower_for_class_clustered_embeddings():
    rng = np.random.default_rng(6)
    y = np.repeat(np.arange(3), 4)
    centers = _unit(rng, 3, 16)
    clustered = centers[y] + 0.05 * rng.normal(size=(12, 16)).astype(np.float32)
    clustered /= np.linalg.norm(clustered, axis=1, keepdims=True)
    scattered = _unit(rng, 12, 16)
    f = lambda u: float(eh.supcon_loss(torch.as_tensor(u), torch.as_tensor(u), torch.as_tensor(y), 0.1))  # noqa: E731
    assert f(clustered) < f(scattered) - 0.5


def test_supcon_gradient_reaches_the_adapter_at_the_zero_init_and_a_step_lowers_the_loss():
    """At W_up = 0 the SupCon term must already have a gradient into W_up / W_down / b_up (else the
    contrastive term could never move the projection), and none into the classifier head."""
    D, r, K, B = 24, 8, 3, 30
    lp = _probe_from_arrays(D, K, seed=2)
    init = eh.SupConHead.from_linear_probe(lp, eh.SupConConfig(rank=r, fold_for_inference=False),
                                           np.random.default_rng(0))
    net = {k: torch.nn.Parameter(torch.as_tensor(np.array(init.a[k]))) for k in
           ("W_down", "b_down", "W_up", "b_up", "W_h", "b_h")}
    rng = np.random.default_rng(1)
    y = torch.as_tensor(rng.integers(0, K, size=B))
    z = torch.as_tensor(rng.normal(size=(B, D)).astype(np.float32))
    obj = eh._SupConObjective(eh.SupConConfig(rank=r, feature_dropout=0.1, tau=0.2), seed=0, dev="cpu")
    loss = obj(net, z, y)
    loss.backward()
    assert float(net["W_up"].grad.abs().max()) > 0 and float(net["b_up"].grad.abs().max()) > 0
    assert net["W_h"].grad is None or float(net["W_h"].grad.abs().max()) == 0.0
    def cur():
        with torch.no_grad():
            u = torch.nn.functional.normalize(eh.torch_embedding(net, z), dim=-1)
            return float(eh.supcon_loss(u, u, y, 0.2))
    before = cur()
    with torch.no_grad():
        for k in ("W_down", "b_down", "W_up", "b_up"):
            net[k] -= 0.05 * net[k].grad
    after = cur()
    assert after < before, (before, after)


def test_the_two_dropout_views_differ_and_zero_dropout_gives_identical_views():
    z = torch.ones(8, 64)
    obj = eh._SupConObjective(eh.SupConConfig(feature_dropout=0.3), seed=1, dev="cpu")
    v1, v2 = obj._view(z), obj._view(z)
    assert not torch.equal(v1, v2)
    assert set(torch.unique(v1).tolist()) <= {0.0, float(np.float32(1 / 0.7))}
    assert abs(float(v1.mean()) - 1.0) < 0.15                               # inverted-dropout scaling
    obj0 = eh._SupConObjective(eh.SupConConfig(feature_dropout=0.0), seed=1, dev="cpu")
    assert torch.equal(obj0._view(z), z)


def test_all_singleton_batches_degrade_to_cross_entropy_without_error():
    """Every batch label distinct: the SupCon term is skipped, counted, and the loss is exactly 0."""
    D, K = 12, 6
    lp = _probe_from_arrays(D, K, seed=3)
    init = eh.SupConHead.from_linear_probe(lp, eh.SupConConfig(rank=4, fold_for_inference=False),
                                           np.random.default_rng(0))
    net = {k: torch.nn.Parameter(torch.as_tensor(np.array(init.a[k]))) for k in
           ("W_down", "b_down", "W_up", "b_up", "W_h", "b_h")}
    obj = eh._SupConObjective(eh.SupConConfig(rank=4), seed=0, dev="cpu")
    out = obj(net, torch.randn(K, D), torch.arange(K))
    assert float(out) == 0.0 and obj.info()["supcon_degraded_batches"] == 1
    assert obj.info()["supcon_mean_loss"] is None


def test_fit_with_supcon_off_or_degenerate_still_trains():
    pytest.importorskip("sklearn")
    X, y = _xor_data(300, D=8, seed=16)
    off = eh.SupConHead.fit(X, y, 2, pair=False, seed=0,
                            config=eh.SupConConfig(rank=8, lambda_supcon=0.0, device="cpu"))
    assert off.info["supcon_degraded_batches"] == off.info["supcon_batches"] > 0
    Xu = np.random.default_rng(1).normal(size=(120, 8)).astype(np.float32)
    yu = np.arange(120) % 60                                           # 60 classes, 2 rows each at best
    many = eh.SupConHead.fit(Xu, yu, 60, pair=False, seed=0,
                             config=eh.SupConConfig(rank=4, device="cpu"),
                             probe_fit=lambda Xp, yp: _probe_from_arrays(8, 60, 0))
    assert np.all(np.isfinite(many.scores(Xu)))


# ------------------------------------------------ parameter count, learning, determinism

@pytest.mark.parametrize("D,r,K", [(1792, 64, 18), (96, 32, 5)])
def test_supervised_param_count_equals_the_adapter_count(D, r, K):
    lp = _probe_from_arrays(D, K, seed=0)
    head = eh.SupConHead.from_linear_probe(lp, eh.SupConConfig(rank=r), np.random.default_rng(0))
    assert head.supervised_param_count() == 2 * D * r + D + r + K * (D + 1)     # SupCon adds no parameter


def test_pair_source_uses_the_full_context_third_only():
    D, K = 12, 3
    lp = _probe_from_arrays(D, K, seed=2, pair=True)
    head = eh.SupConHead.from_linear_probe(lp, eh.SupConConfig(rank=4), np.random.default_rng(0))
    assert head.in_dim == 3 * D
    X = np.random.default_rng(3).normal(size=(5, 3 * D)).astype(np.float32)
    X2 = X.copy()
    X2[:, D:] = 99.0
    assert np.array_equal(head.scores(X), head.scores(X2))


def test_supcon_head_learns_an_interaction_the_linear_probe_cannot():
    pytest.importorskip("sklearn")
    X, y = _xor_data(3000, seed=0)
    tr, te = np.arange(2400), np.arange(2400, 3000)
    head = eh.SupConHead.fit(X[tr], y[tr], 2, pair=False, seed=0, config=eh.SupConConfig(rank=16, device="cpu"))
    lp = sx.LinearProbe.fit(X[tr], y[tr], 2, pair=False, kind="full", pca_k=None, seed=0)
    acc_s = float(np.mean(head.scores(X[te]).argmax(1) == y[te]))
    acc_l = float(np.mean(lp.scores(X[te]).argmax(1) == y[te]))
    assert acc_l < 0.6 and acc_s > 0.85, (acc_l, acc_s)
    assert head.info["best_epoch"] > 0 and head.info["supcon_mean_loss"] is not None
    assert head.info["supcon_degraded_batches"] < head.info["supcon_batches"]


def test_fit_is_deterministic_for_a_fixed_seed():
    pytest.importorskip("sklearn")
    X, y = _xor_data(500, D=10, seed=17)
    cfg = eh.SupConConfig(rank=8, device="cpu")
    a = eh.SupConHead.fit(X, y, 2, pair=False, seed=3, config=cfg)
    b = eh.SupConHead.fit(X, y, 2, pair=False, seed=3, config=cfg)
    assert all(np.array_equal(a.a[k], b.a[k]) for k in a.a)


def test_no_improvement_returns_the_linear_probe_itself():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=300)
    head = eh.SupConHead.fit(X, y, 2, pair=False, seed=0, config=eh.SupConConfig(rank=4, device="cpu"))
    if head.info["best_epoch"] == 0:
        assert not np.any(head.a["W_fold"])
    else:
        assert head.info["early_stop_acc"] > head.info["early_stop_acc_at_init"] or \
            head.info["early_stop_ce"] < head.info["early_stop_ce_at_init"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_fit_returns_a_numpy_only_head():
    pytest.importorskip("sklearn")
    X, y = _xor_data(600, D=10, seed=18)
    h = eh.SupConHead.fit(X, y, 2, pair=False, seed=0, config=eh.SupConConfig(rank=8, device="cuda"))
    assert h.info["device"] == "cuda" and all(type(v) is np.ndarray for v in h.a.values())


# ------------------------------------- 7: pipeline wiring (expert_specs ... load_expert)

def test_expert_specs_register_supcon_only_when_enabled(monkeypatch):
    fs = {"enc": {"train_full": np.zeros((5, 16))}}
    monkeypatch.setattr(bse, "ADAPTER_RANKS", ())
    monkeypatch.setattr(bse, "SUPCON_RANKS", ())
    assert not [n for n, _, _ in bse.expert_specs("massive_en", fs) if "supcon" in n]
    monkeypatch.setattr(bse, "SUPCON_RANKS", (32, 64, 128))
    specs = {n: (s, sp) for n, s, sp in bse.expert_specs("massive_en", fs)}
    for r in (32, 64, 128):
        assert specs[f"enc:supcon_r{r}"] == ("enc", {"type": "supcon", "rank": r})
        assert eh.head_config(specs[f"enc:supcon_r{r}"][1]) == eh.SupConConfig(rank=r)
    assert not [n for n in specs if "adapter" in n]                       # independent of the adapter flag
    assert bse.parse_supcon_ranks("", True) == (32, 64, 128)
    assert bse.parse_supcon_ranks("16, 32", False) == (16, 32)
    assert bse.parse_supcon_ranks("", False) == ()
    for bad in ("64,64", "0", "-2"):
        with pytest.raises(ValueError, match="--supcon-ranks"):
            bse.parse_supcon_ranks(bad, False)


def test_cli_flags_are_registered_and_set_the_module_state(monkeypatch, tmp_path):
    monkeypatch.setattr(bse, "SUPCON_RANKS", ())
    seen = {}
    monkeypatch.setattr(bse, "stage_combine", lambda *a, **k: seen.setdefault("ranks", bse.SUPCON_RANKS))
    monkeypatch.setattr(bse, "load_sources", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["x", "combine", "--source", f"s={tmp_path}", "--tasks", "massive_en",
                                      "--enable-supcon", "--out-art", str(tmp_path), "--results-dir", str(tmp_path)])
    try:
        bse.main()
    except SystemExit:
        pass                                                              # later stages need real data
    assert bse.SUPCON_RANKS == (32, 64, 128)
    monkeypatch.setattr(sys, "argv", ["x", "combine", "--source", f"s={tmp_path}", "--tasks", "massive_en",
                                      "--supcon-ranks", "16,48", "--out-art", str(tmp_path)])
    try:
        bse.main()
    except SystemExit:
        pass
    assert bse.SUPCON_RANKS == (16, 48)


def test_five_fold_fit_feeds_the_complexity_order():
    pytest.importorskip("sklearn")
    X, y = _xor_data(400, D=10, seed=12)
    f = {"X_train": X, "train_label": y, "cands": np.zeros((2, 1), dtype=np.float32)}
    folds = bse.fold_ids("massive_en", len(y))
    oof, infos = np.zeros((len(y), 2), dtype=np.float32), []
    for k in range(bse.N_FOLDS):
        tr, ho = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        _, s, info = bse._fit_predict({"type": "supcon", "rank": 8}, "massive_en", f, tr, ho, X[ho], [0, 0, 100 + k])
        oof[ho] = s
        infos.append(info)
    assert np.all(np.isfinite(oof)) and all(i["params"] == 2 * 10 * 8 + 10 + 8 + 2 * 11 for i in infos)
    specs = {"e:supcon_r8": {"type": "supcon", "rank": 8},
             "e:lin_full": {"type": "linear", "kind": "full", "pca_k": None}, "e:fractal": {"type": "fractal"}}
    meta = {"e:supcon_r8": {"folds": infos}, "e:lin_full": {"folds": [{"feature_dim": 10}] * 5},
            "e:fractal": {"folds": [{}] * 5}}
    cx = {st: bse.strategy_complexity(st, specs, meta, 2, {}) for st in
          ["single:prior", "single:e:lin_full", "single:e:supcon_r8", "single:e:fractal", "pool_greedy"]}
    assert cx["single:e:supcon_r8"] == (0, max(i["params"] for i in infos))
    assert cx["single:e:fractal"] < cx["single:prior"] < cx["single:e:lin_full"] < cx["single:e:supcon_r8"] \
        < cx["pool_greedy"]
    json.dumps(infos, default=float)                                       # fit info must serialize for the OOF meta


def _write_source(root: Path, task: str, X, y, Xte, K):
    (root / "features").mkdir(parents=True)
    np.savez(root / "features" / f"{task}.npz", train_full=X, test_full=Xte,
             train_ids=np.arange(len(y)), test_ids=np.arange(len(Xte)), train_label=y,
             cands=np.eye(K, 4, dtype=np.float32), info_json=np.array(json.dumps({"encoder": "synthetic"})))


def test_stage_fit_combine_and_load_expert_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    X, y = _xor_data(500, D=10, seed=13)
    Xte, _ = _xor_data(40, D=10, seed=14)
    _write_source(tmp_path / "src", "massive_en", X, y, Xte, 2)
    monkeypatch.setattr(bse, "SOURCES", {"syn": tmp_path / "src"})
    monkeypatch.setattr(bse, "OUT_ART", tmp_path / "art")
    monkeypatch.setattr(bse, "ADAPTER_RANKS", ())
    monkeypatch.setattr(bse, "SUPCON_RANKS", (8,))
    monkeypatch.setattr(bse, "RNN_SOURCES", [])
    bse.stage_fit(["massive_en"], ["supcon_r8", "syn:lin_full"])
    model = tmp_path / "art" / "models" / "massive_en" / "syn__supcon_r8.npz"
    assert model.exists()
    with np.load(tmp_path / "art" / "oof" / "massive_en" / "syn__supcon_r8.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
    assert meta["spec"] == {"type": "supcon", "rank": 8} and len(meta["folds"]) == bse.N_FOLDS
    assert all(fo["params"] == 2 * 10 * 8 + 10 + 8 + 2 * 11 for fo in meta["folds"])
    bse.stage_combine(["massive_en"], learned=())
    comb = json.loads((tmp_path / "art" / "combine" / "massive_en.json").read_text())
    assert "single:syn:supcon_r8" in comb["nested_cv_acc"] and "single:prior" in comb["nested_cv_acc"]
    # Wiring only. On this 500-row XOR set the SupCon head does NOT beat lin_full (measured ~54% vs 54%;
    # it needs ~4000 rows, see test_supcon_head_learns_an_interaction...), so no accuracy claim is made
    # here; the selector must simply pick one of the compared strategies.
    assert 0.0 < comb["nested_cv_acc"]["single:syn:supcon_r8"] < 100.0
    assert comb["chosen"] in comb["nested_cv_acc"]
    fs = bse.load_sources("massive_en")
    src, spec = {n: (s, sp) for n, s, sp in bse.expert_specs("massive_en", fs)}["syn:supcon_r8"]
    core = bse.load_expert("massive_en", "syn:supcon_r8", src, spec, fs)
    assert isinstance(core, eh.SupConHead)
    assert core.score(Xte[0], fs["syn"]["cands"]).shape == (2,)
