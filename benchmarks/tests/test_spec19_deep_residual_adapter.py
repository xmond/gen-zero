"""Spec 19 Phase 1: the folded deep residual adapter (sota_enhanced_heads) and its pipeline wiring.

Covers the S7 per-head test list (zero-init == LinearProbe, fold exactness < 1e-5, save/load
bit-exact, non-finite input and wrong K rejected, training reads only the rows passed), the
S2.1 parameter formula, NumPy-only inference, the S6.2 integration points of
benchmark_sota_ensemble (expert_specs, _fit_predict, stage_fit save, strategy_complexity,
load_expert) and the S6.5 gate-2 `single:prior` strategy inside the 1-SE selector.

All data is synthetic (Gaussian vectors, labels from a stated rule); no task files.
Tests that call LinearProbe.fit need scikit-learn and skip without it; the ones that only
need a probe build it from arrays.
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
    """A `lin_full` LinearProbe with random (not fitted) weights: no sklearn needed."""
    rng = np.random.default_rng(seed)
    arrays = {"mu_full": rng.normal(size=D).astype(np.float32),
              "sd_full": rng.uniform(0.5, 2.0, size=D).astype(np.float32),
              "W": rng.normal(size=(K, D)).astype(np.float32),
              "b": rng.normal(size=K).astype(np.float32)}
    cfg = {"K": K, "pair": pair, "kind": "full", "pca_k": None, "in_dim": 3 * D if pair else D,
           "C": 1.0, "feature_dim": D}
    return sx.LinearProbe(arrays, cfg)


def _random_trained_like(D=96, r=32, K=5, seed=0, fold=True, head_scale=None):
    """An adapter with every parameter non-zero (as after training), plus its raw parameters.

    head_scale sets the std of W_h entries; the default 2/sqrt(D) gives logits of a few units,
    the scale of a fitted L2 logistic head. head_scale=1.0 gives logits in the hundreds."""
    rng = np.random.default_rng(seed)
    lp = _probe_from_arrays(D, K, seed)
    scale = 2.0 / np.sqrt(D) if head_scale is None else head_scale
    lp = sx.LinearProbe(dict(lp.a, W=(lp.a["W"] * scale).astype(np.float32)), lp.cfg)
    p = {"W_down": rng.normal(scale=1 / np.sqrt(D), size=(r, D)), "b_down": rng.normal(scale=0.1, size=r),
         "W_up": rng.normal(scale=0.2, size=(D, r)), "b_up": rng.normal(scale=0.1, size=D),
         "W_h": lp.a["W"].astype(np.float64), "b_h": lp.a["b"].astype(np.float64)}
    p = {k: v.astype(np.float32) for k, v in p.items()}
    head = eh.DeepResidualAdapterHead.from_unfolded(
        {"mu_full": lp.a["mu_full"], "sd_full": lp.a["sd_full"]}, p["W_down"], p["b_down"], p["W_up"],
        p["b_up"], p["W_h"], p["b_h"], K=K, pair=False, in_dim=D,
        config=eh.AdapterConfig(rank=r, fold_for_inference=fold))
    return head, p, lp


def _xor_data(n, D=16, seed=0):
    """Label = sign(x0) XOR sign(x1): no linear function of x separates it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D)).astype(np.float32)
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)


# ------------------------------------------------ test 1: zero init == linear probe

@pytest.mark.parametrize("fold", [True, False])
def test_zero_initialized_adapter_equals_the_linear_probe_exactly(fold):
    D, K = 256, 7
    lp = _probe_from_arrays(D, K, seed=1)
    head = eh.DeepResidualAdapterHead.from_linear_probe(
        lp, eh.AdapterConfig(rank=64, fold_for_inference=fold), np.random.default_rng(0))
    X = np.random.default_rng(2).normal(size=(200, D)).astype(np.float32) * 3
    assert np.any(head.a["W_down"] != 0)                   # the bottleneck is live, only W_up is zero
    err = float(np.max(np.abs(head.scores(X) - lp.scores(X))))
    assert err < 1e-6, err
    assert err == 0.0                                      # same float32 operands: bit-identical


def test_fit_starts_from_the_probe_fitted_on_the_same_rows():
    pytest.importorskip("sklearn")
    X, y = _xor_data(300, seed=3)
    head = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=8))
    assert head.info["identity_max_abs_err_at_init"] == 0.0


# ------------------------------------------------- test 2: fold is an exact identity

def test_gelu_matches_torch_tanh_gelu():
    z = np.linspace(-8, 8, 4001, dtype=np.float32)
    ref = torch.nn.functional.gelu(torch.as_tensor(z), approximate="tanh").numpy()
    assert float(np.max(np.abs(eh.gelu_tanh(z) - ref))) < 1e-6


@pytest.mark.parametrize("D,r,K", [(96, 32, 5), (512, 64, 18), (1024, 128, 3)])
def test_folded_numpy_inference_equals_the_unfolded_torch_forward(D, r, K):
    head, p, lp = _random_trained_like(D, r, K, seed=D)
    X = np.random.default_rng(5).normal(size=(64, D)).astype(np.float32)
    z = torch.as_tensor((X - lp.a["mu_full"]) / lp.a["sd_full"])
    net = {k: torch.as_tensor(v) for k, v in p.items()}
    with torch.no_grad():
        ref = eh.torch_unfolded_logits(net, z).numpy()
    folded = head.scores(X)
    err = float(np.max(np.abs(folded - ref)))
    assert err < 1e-5, err
    assert float(np.max(np.abs(ref - lp.scores(X)))) > 1e-2   # the residual branch really contributes
    unfolded, _, _ = _random_trained_like(D, r, K, seed=D, fold=False)
    assert float(np.max(np.abs(unfolded.scores(X) - ref))) < 1e-5


@pytest.mark.parametrize("D,r,K", [(512, 64, 18), (1024, 128, 3)])
def test_fold_error_at_large_logits_is_float32_rounding_not_a_formula_error(D, r, K):
    """With W_h ~ N(0, 1) the logits reach the hundreds and float32 rounding alone exceeds 1e-5
    in BOTH paths. Against a float64 unfolded forward, the folded NumPy head must be at least as
    accurate as torch's own float32 unfolded forward, and within float32 relative precision."""
    head, p, lp = _random_trained_like(D, r, K, seed=D, head_scale=1.0)
    X = np.random.default_rng(5).normal(size=(64, D)).astype(np.float32)
    z = (X - lp.a["mu_full"]) / lp.a["sd_full"]
    with torch.no_grad():
        truth = eh.torch_unfolded_logits({k: torch.as_tensor(v.astype(np.float64)) for k, v in p.items()},
                                          torch.as_tensor(z.astype(np.float64))).numpy()
        t32 = eh.torch_unfolded_logits({k: torch.as_tensor(v) for k, v in p.items()}, torch.as_tensor(z)).numpy()
    err_fold, err_t32 = float(np.max(np.abs(head.scores(X) - truth))), float(np.max(np.abs(t32 - truth)))
    assert float(np.max(np.abs(truth))) > 30
    assert err_fold <= 1.5 * err_t32, (err_fold, err_t32)
    assert err_fold / float(np.max(np.abs(truth))) < 1e-6


def test_trained_head_export_matches_its_torch_forward():
    pytest.importorskip("sklearn")
    X, y = _xor_data(400, seed=4)
    head = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=8))
    assert head.info["export_max_abs_err_es"] < 1e-5
    assert set(head.a) == set(eh.DeepResidualAdapterHead.FOLDED_KEYS)   # W_up itself is not exported


# ----------------------------------------------------------- test 3: save / load

@pytest.mark.parametrize("fold", [True, False])
def test_save_load_roundtrip_is_bit_exact(tmp_path, fold):
    head, _, _ = _random_trained_like(fold=fold)
    head.save(tmp_path / "a.npz")
    with np.load(tmp_path / "a.npz", allow_pickle=False) as z:
        keys = set(z.files)
    want = {"W_h", "W_fold", "W_down", "b_down", "b_fold"} if fold else {"W_h", "W_up", "b_up", "b_h"}
    assert want | {"cfg_json", "mu_full", "sd_full"} <= keys
    back = eh.DeepResidualAdapterHead.load(tmp_path / "a.npz")
    assert back.cfg == head.cfg
    for k in head.a:
        assert back.a[k].dtype == np.float32 and np.array_equal(back.a[k], head.a[k]), k
    X = np.random.default_rng(6).normal(size=(50, head.in_dim)).astype(np.float32)
    assert np.array_equal(back.scores(X), head.scores(X))
    assert np.array_equal(back.score(X[0], np.zeros((head.K, 1))), head.scores(X[:1])[0])


# ------------------------------------------------------- test 4: NumPy-only inference

def test_load_and_score_run_with_torch_import_blocked(tmp_path, monkeypatch):
    head, _, _ = _random_trained_like()
    head.save(tmp_path / "a.npz")
    X = np.random.default_rng(7).normal(size=(10, head.in_dim)).astype(np.float32)
    expect = head.scores(X)
    monkeypatch.setitem(sys.modules, "torch", None)      # any `import torch` now raises ImportError
    back = eh.DeepResidualAdapterHead.load(tmp_path / "a.npz")
    out = back.scores(X)
    assert type(out) is np.ndarray and out.dtype == np.float32
    assert np.array_equal(out, expect)
    assert all(type(v) is np.ndarray for v in back.a.values())
    with pytest.raises(ImportError):
        import torch as _t  # noqa: F401


def test_module_import_and_inference_in_a_fresh_process_without_torch(tmp_path):
    head, _, _ = _random_trained_like()
    head.save(tmp_path / "a.npz")
    X = np.random.default_rng(8).normal(size=(4, head.in_dim)).astype(np.float32)
    np.save(tmp_path / "x.npy", X)
    code = ("import sys; sys.modules['torch'] = None; sys.path.insert(0, %r); "
            "import numpy as np, sota_enhanced_heads as eh; "
            "h = eh.DeepResidualAdapterHead.load(%r); np.save(%r, h.scores(np.load(%r))); "
            "print('torch' in sys.modules and sys.modules['torch'] is not None)"
            % (str(SUITES), str(tmp_path / "a.npz"), str(tmp_path / "o.npy"), str(tmp_path / "x.npy")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
    assert np.array_equal(np.load(tmp_path / "o.npy"), head.scores(X))


# ----------------------------------------------------------- test 5: input defense

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
        eh.DeepResidualAdapterHead.fit(np.repeat(X, 20, 0), np.tile([0, 1, 0], 20), 2, pair=False, seed=0,
                                       probe_fit=lambda *a: pytest.fail("probe must not be fitted"))


def test_wrong_width_wrong_k_and_bad_labels_are_rejected():
    head, _, _ = _random_trained_like()
    with pytest.raises(ValueError):
        head.scores(np.zeros((2, head.in_dim + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        head.scores(np.zeros((2, 2, head.in_dim), dtype=np.float32))
    with pytest.raises(ValueError):
        head.score(np.zeros(head.in_dim, dtype=np.float32), np.zeros((head.K + 1, 1)))
    X = np.zeros((40, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        eh.DeepResidualAdapterHead.fit(X, np.full(40, 3), 3, pair=False, seed=0)       # label == K
    with pytest.raises(ValueError):
        eh.DeepResidualAdapterHead.fit(X, np.zeros(39, dtype=int), 3, pair=False, seed=0)
    with pytest.raises(ValueError):
        eh.AdapterConfig(rank=0)
    with pytest.raises(ValueError):
        eh.AdapterConfig(spectral_cap=-1.0)
    with pytest.raises(ValueError):
        eh.DeepResidualAdapterHead(dict(head.a, W_fold=head.a["W_fold"][:, :-1]), head.cfg)


@pytest.mark.parametrize("bad_lr", [0.0, -1e-3, float("nan"), float("inf")])
def test_adapter_config_rejects_non_finite_or_non_positive_lr(bad_lr):
    with pytest.raises(ValueError):
        eh.AdapterConfig(lr=bad_lr)


def test_adapter_config_accepts_a_custom_lr_and_defaults_to_1em3():
    assert eh.AdapterConfig().lr == 1e-3
    assert eh.AdapterConfig(rank=64, lr=5e-4).lr == 5e-4


def test_training_reads_only_the_rows_passed():
    """S7: NaN sentinel rows sit outside `tr`; _fit_predict must neither fail nor read them."""
    pytest.importorskip("sklearn")
    X, y = _xor_data(360, seed=9)
    X[::6] = np.nan
    tr = np.flatnonzero(np.arange(360) % 6 != 0)
    f = {"X_train": X, "train_label": y, "cands": np.zeros((2, 1), dtype=np.float32)}
    m, s, info = bse._fit_predict({"type": "adapter", "rank": 8}, "massive_en", f, tr, tr[:40], X[tr[:40]],
                                  [0, 0, 100])
    assert s.shape == (40, 2) and np.all(np.isfinite(s))
    assert info["n_train"] + info["n_early_stop"] == len(tr)


def test_the_warm_start_probe_never_sees_the_early_stop_rows():
    seen = {}

    def probe_fit(Xp, yp):
        seen["X"] = Xp.copy()
        return _probe_from_arrays(Xp.shape[1], 2, seed=0)

    X, y = _xor_data(200, D=8, seed=10)
    X += np.arange(200, dtype=np.float32)[:, None] * 1e-3        # make every row unique
    h = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=4),
                                       rng_key=[0, 1, 2], probe_fit=probe_fit)
    es = np.random.default_rng([0, 1, 2]).random(200) < sx.EARLY_STOP_FRACTION   # fit's first draw
    assert len(seen["X"]) == h.info["n_train"] == int((~es).sum())
    assert np.array_equal(seen["X"], X[~es])


# ------------------------------------------------------ test 6: parameter count

@pytest.mark.parametrize("D,r,K", [(1792, 64, 18), (5120, 64, 18), (96, 32, 5)])
def test_supervised_param_count_matches_the_spec_formula(D, r, K):
    lp = _probe_from_arrays(D, K, seed=0)
    head = eh.DeepResidualAdapterHead.from_linear_probe(lp, eh.AdapterConfig(rank=r), np.random.default_rng(0))
    n = head.supervised_param_count()
    assert n == 2 * D * r + D + r + K * (D + 1)
    # the parameters the torch model actually trains: W_down, b_down, W_up, b_up, W_h, b_h
    assert n == r * D + r + D * r + D + K * D + K


def test_spec_table_values():
    """Spec 19 S2.1 table: adapter part 2Dr + D + r, head part K(D+1)."""
    for D, r, adapter, head in [(1792, 64, 231_232, 32_274), (5120, 64, 660_544, 92_178),
                                (5120, 512, 5_248_512, 92_178), (8192, 512, 8_397_312, 147_474)]:
        assert 2 * D * r + D + r == adapter and 18 * (D + 1) == head


def test_pair_source_uses_the_full_context_third_only():
    D, K = 12, 3
    lp = _probe_from_arrays(D, K, seed=2, pair=True)
    head = eh.DeepResidualAdapterHead.from_linear_probe(lp, eh.AdapterConfig(rank=4), np.random.default_rng(0))
    assert head.in_dim == 3 * D and head.supervised_param_count() == 2 * D * 4 + D + 4 + K * (D + 1)
    X = np.random.default_rng(3).normal(size=(5, 3 * D)).astype(np.float32)
    X2 = X.copy()
    X2[:, D:] = 99.0
    assert np.array_equal(head.scores(X), head.scores(X2))


# -------------------------------------------- nonlinear capacity and regularization

def test_adapter_learns_an_interaction_the_linear_probe_cannot():
    """XOR of two coordinates: the linear probe stays at chance, the adapter must not."""
    pytest.importorskip("sklearn")
    X, y = _xor_data(3000, seed=0)
    tr, te = np.arange(2400), np.arange(2400, 3000)
    head = eh.DeepResidualAdapterHead.fit(X[tr], y[tr], 2, pair=False, seed=0, config=eh.AdapterConfig(rank=16))
    lp = sx.LinearProbe.fit(X[tr], y[tr], 2, pair=False, kind="full", pca_k=None, seed=0)
    acc_a = float(np.mean(head.scores(X[te]).argmax(1) == y[te]))
    acc_l = float(np.mean(lp.scores(X[te]).argmax(1) == y[te]))
    assert acc_l < 0.6 and acc_a > 0.85, (acc_l, acc_a)
    assert head.info["best_epoch"] > 0


def test_spectral_cap_bounds_w_up():
    pytest.importorskip("sklearn")
    X, y = _xor_data(1500, seed=1)
    head = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0,
                                          config=eh.AdapterConfig(rank=16, spectral_cap=0.5))
    assert head.info["w_up_spectral"] <= 0.5 + 1e-5


def test_no_improvement_returns_the_linear_probe_itself():
    """Pure-noise labels: if no epoch beats epoch 0 on the held-out slice, W_fold stays 0."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=300)
    head = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=4))
    if head.info["best_epoch"] == 0:
        assert not np.any(head.a["W_fold"])
    else:
        assert head.info["early_stop_acc"] > head.info["early_stop_acc_at_init"] or \
            head.info["early_stop_ce"] < head.info["early_stop_ce_at_init"]


# -------------------------------------- test 7: pipeline folds, complexity and 1-SE

def test_expert_specs_register_adapters_only_when_enabled(monkeypatch):
    fs = {"enc": {"train_full": np.zeros((5, 16))}}
    monkeypatch.setattr(bse, "ADAPTER_RANKS", ())
    assert not [n for n, _, _ in bse.expert_specs("massive_en", fs) if "adapter" in n]
    monkeypatch.setattr(bse, "ADAPTER_RANKS", (32, 64, 128))
    specs = {n: sp for n, _, sp in bse.expert_specs("massive_en", fs)}
    for r in (32, 64, 128):
        assert specs[f"enc:adapter_r{r}"] == {"type": "adapter", "rank": r}
        assert eh.head_config(specs[f"enc:adapter_r{r}"]) == eh.AdapterConfig(rank=r)
    assert bse.parse_adapter_ranks("", True) == (32, 64, 128)
    assert bse.parse_adapter_ranks("", False) == ()
    with pytest.raises(ValueError):
        bse.parse_adapter_ranks("64,64", False)


def test_task_adapter_overrides_apply_the_tuned_lr_only_to_their_task_and_rank(monkeypatch):
    fs = {"enc": {"train_full": np.zeros((5, 16))}}
    monkeypatch.setattr(bse, "ADAPTER_RANKS", (32, 64, 128))
    massive_de_specs = {n: sp for n, _, sp in bse.expert_specs("massive_de", fs)}
    assert massive_de_specs["enc:adapter_r64"] == {"type": "adapter", "rank": 64, "lr": 5e-4}
    assert massive_de_specs["enc:adapter_r32"] == {"type": "adapter", "rank": 32}     # untuned rank: no override
    pubmedqa_specs = {n: sp for n, _, sp in bse.expert_specs("pubmedqa", fs)}
    assert pubmedqa_specs["enc:adapter_r128"] == {"type": "adapter", "rank": 128, "lr": 1e-3}
    assert pubmedqa_specs["enc:adapter_r64"] == {"type": "adapter", "rank": 64}        # untuned rank: no override
    other_specs = {n: sp for n, _, sp in bse.expert_specs("massive_en", fs)}
    assert other_specs["enc:adapter_r64"] == {"type": "adapter", "rank": 64}           # unrelated task: no override
    # head_config() transparently threads the override lr into AdapterConfig
    assert eh.head_config(massive_de_specs["enc:adapter_r64"]) == eh.AdapterConfig(rank=64, lr=5e-4)
    assert eh.head_config(pubmedqa_specs["enc:adapter_r128"]) == eh.AdapterConfig(rank=128, lr=1e-3)


def test_five_fold_fit_feeds_the_complexity_order():
    pytest.importorskip("sklearn")
    X, y = _xor_data(400, D=10, seed=12)
    f = {"X_train": X, "train_label": y, "cands": np.zeros((2, 1), dtype=np.float32)}
    folds = bse.fold_ids("massive_en", len(y))
    oof, infos = np.zeros((len(y), 2), dtype=np.float32), []
    for k in range(bse.N_FOLDS):
        tr, ho = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        _, s, info = bse._fit_predict({"type": "adapter", "rank": 8}, "massive_en", f, tr, ho, X[ho], [0, 0, 100 + k])
        oof[ho] = s
        infos.append(info)
    assert np.all(np.isfinite(oof)) and all(i["params"] == 2 * 10 * 8 + 10 + 8 + 2 * 11 for i in infos)
    assert all(i["n_train"] + i["n_early_stop"] == int((folds != k).sum()) for k, i in enumerate(infos))
    specs = {"e:adapter_r8": {"type": "adapter", "rank": 8},
             "e:lin_full": {"type": "linear", "kind": "full", "pca_k": None}, "e:fractal": {"type": "fractal"}}
    meta = {"e:adapter_r8": {"folds": infos}, "e:lin_full": {"folds": [{"feature_dim": 10}] * 5},
            "e:fractal": {"folds": [{}] * 5}}
    cx = {st: bse.strategy_complexity(st, specs, meta, 2, {}) for st in
          ["single:prior", "single:e:lin_full", "single:e:adapter_r8", "single:e:fractal", "pool_greedy"]}
    assert cx["single:prior"] == (0, 1)                                 # (0, K-1)
    assert cx["single:e:adapter_r8"] == (0, max(i["params"] for i in infos))
    assert cx["single:e:fractal"] < cx["single:prior"] < cx["single:e:lin_full"] < cx["single:e:adapter_r8"] \
        < cx["pool_greedy"]


def _folds(*pcts):
    return np.array(pcts, dtype=float) / 100.0


def test_one_se_prefers_the_linear_probe_unless_the_adapter_clears_one_se():
    strategies = ["single:prior", "single:e:lin_full", "single:e:adapter_r64"]
    cx = {"single:prior": (0, 17), "single:e:lin_full": (0, 18 * 5121), "single:e:adapter_r64": (0, 752_722)}
    near = {"single:prior": _folds(40, 41, 40, 42, 41), "single:e:lin_full": _folds(84, 85, 84, 86, 84),
            "single:e:adapter_r64": _folds(84, 86, 84, 86, 85)}                # mean 85.0, SE 0.447
    assert bse.select_one_se(strategies, near, cx)[0] == "single:e:lin_full"
    far = dict(near, **{"single:e:adapter_r64": _folds(90, 91, 90, 91, 90)})
    chosen, info = bse.select_one_se(strategies, far, cx)
    assert chosen == "single:e:adapter_r64" and info["band"] == ["single:e:adapter_r64"]
    tie = dict(near, **{"single:prior": _folds(84, 85, 84, 86, 84)})          # prior inside the band
    assert bse.select_one_se(strategies, tie, cx)[0] == "single:prior"


def test_prior_strategy_fits_on_its_rows_and_replays_the_majority_class():
    y = np.array([0, 1, 1, 2, 1, 0, 2, 2, 2, 2])
    S = {"e": np.zeros((10, 3))}
    Xq = {"e": np.zeros((10, 4))}
    fit = bse.fit_strategy("single:prior", S, Xq, y, np.arange(5), {"e": 1.0})
    assert fit["prior"] == pytest.approx([0.2, 0.6, 0.2])
    assert np.array_equal(bse.replay_eval("single:prior", fit, S, Xq, np.arange(5, 10), 3), [1] * 5)
    assert bse.n_experts_run("single:prior", {"weights": {}, "temperatures": {}}, 7) == 0
    fit_all = bse.fit_strategy("single:prior", S, Xq, y, np.arange(10), {"e": 1.0})
    assert int(np.argmax(fit_all["prior"])) == 2


def _write_source(root: Path, task: str, X, y, Xte, K):
    (root / "features").mkdir(parents=True)
    np.savez(root / "features" / f"{task}.npz", train_full=X, test_full=Xte,
             train_ids=np.arange(len(y)), test_ids=np.arange(len(Xte)), train_label=y,
             cands=np.eye(K, 4, dtype=np.float32), info_json=np.array(json.dumps({"encoder": "synthetic"})))


def test_stage_fit_combine_and_load_expert_end_to_end(tmp_path, monkeypatch):
    """The four S6.2 wiring points on a synthetic source: fit (OOF + full-train save),
    complexity from the saved fold params, combine with single:prior, load_expert."""
    pytest.importorskip("sklearn")
    X, y = _xor_data(500, D=10, seed=13)
    Xte, _ = _xor_data(40, D=10, seed=14)
    _write_source(tmp_path / "src", "massive_en", X, y, Xte, 2)
    monkeypatch.setattr(bse, "SOURCES", {"syn": tmp_path / "src"})
    monkeypatch.setattr(bse, "OUT_ART", tmp_path / "art")
    monkeypatch.setattr(bse, "ADAPTER_RANKS", (8,))
    monkeypatch.setattr(bse, "RNN_SOURCES", [])
    bse.stage_fit(["massive_en"], ["adapter_r8", "syn:lin_full"])
    model = tmp_path / "art" / "models" / "massive_en" / "syn__adapter_r8.npz"
    assert model.exists()
    with np.load(tmp_path / "art" / "oof" / "massive_en" / "syn__adapter_r8.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
    assert meta["spec"] == {"type": "adapter", "rank": 8} and len(meta["folds"]) == bse.N_FOLDS
    assert all(fo["params"] == 2 * 10 * 8 + 10 + 8 + 2 * 11 for fo in meta["folds"])
    bse.stage_combine(["massive_en"], learned=())
    comb = json.loads((tmp_path / "art" / "combine" / "massive_en.json").read_text())
    assert "single:prior" in comb["nested_cv_acc"] and "single:syn:adapter_r8" in comb["nested_cv_acc"]
    assert comb["selection"]["complexity"].get("single:prior", [0, 1]) == [0, 1]
    # XOR: lin_full and prior sit near 50%; the adapter must be clear of both by > 1 SE
    assert comb["nested_cv_acc"]["single:syn:adapter_r8"] > comb["nested_cv_acc"]["single:syn:lin_full"] + 10
    assert comb["chosen"] == "single:syn:adapter_r8"
    fs = bse.load_sources("massive_en")
    spec = {n: (s, sp) for n, s, sp in bse.expert_specs("massive_en", fs)}["syn:adapter_r8"]
    core = bse.load_expert("massive_en", "syn:adapter_r8", spec[0], spec[1], fs)
    assert isinstance(core, eh.DeepResidualAdapterHead)
    assert core.score(Xte[0], fs["syn"]["cands"]).shape == (2,)


def test_cold_start_head_trains_and_keeps_absent_classes_unreachable():
    """warm_start_linear=False: head starts at zero (logits = absent-class bias only), K=3 with
    class 2 absent from the rows; the trained head must learn and never predict class 2."""
    pytest.importorskip("sklearn")
    X, y = _xor_data(1500, D=8, seed=15)
    head = eh.DeepResidualAdapterHead.fit(X, y, 3, pair=False, seed=0,
                                          config=eh.AdapterConfig(rank=16, warm_start_linear=False))
    assert head.info["early_stop_acc_at_init"] < 0.6 < head.info["early_stop_acc"]
    assert not np.any(head.scores(X).argmax(1) == 2)
