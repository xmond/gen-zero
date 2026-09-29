"""Spec 20 P3: Formulation B folded residual adapter (sota_enhanced_heads.FoldedResidualAdapterBHead).

Covers, per Spec 20 S4.2 / S4.3 and the P3 task sheet:
  1. zero-init identity: with C == 0 the head's logits equal the frozen linear baseline f0 bit for bit;
  2. objective monotonicity: every accepted step has J_t <= J_{t-1}, J_final <= J_0, and J_0 is the
     baseline cross-entropy recomputed independently here;
  3. NumPy-only inference: load() + scores() run with torch import blocked, in-process and in a
     fresh interpreter;
  4. input defense: NaN/Inf, wrong width, wrong K, bad labels, bad arrays, bad configs are rejected;
  5. row isolation: NaN sentinel rows outside `tr` are neither read nor fatal through _fit_predict;
  6. ensemble wiring: expert_specs / parse flags / argparse register adapter_b only when enabled.
The pure-NumPy tests do not need torch; the training tests and the CLI test skip without torch / sklearn.
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

import benchmark_sota_ensemble as bse  # noqa: E402
import sota_enhanced_heads as eh  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402

Head = eh.FoldedResidualAdapterBHead


def _needs_training_backend():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    torch.set_num_threads(2)
    return torch


def _probe_from_arrays(D: int, K: int, seed: int, pair: bool = False) -> sx.LinearProbe:
    """A `lin_full` LinearProbe with random (not fitted) weights: no sklearn needed."""
    rng = np.random.default_rng(seed)
    arrays = {"mu_full": rng.normal(size=D).astype(np.float32),
              "sd_full": rng.uniform(0.5, 2.0, size=D).astype(np.float32),
              "W": (rng.normal(size=(K, D)) * 2.0 / np.sqrt(D)).astype(np.float32),
              "b": rng.normal(size=K).astype(np.float32)}
    cfg = {"K": K, "pair": pair, "kind": "full", "pca_k": None, "in_dim": 3 * D if pair else D,
           "C": 1.0, "feature_dim": D}
    return sx.LinearProbe(arrays, cfg)


def _random_trained_like(D=96, r=32, K=5, seed=0, pair=False):
    """A Formulation B head with every parameter non-zero (as after training), plus the raw params."""
    rng = np.random.default_rng(seed)
    lp = _probe_from_arrays(D, K, seed, pair=pair)
    p = {"C": rng.normal(scale=0.3, size=(K, r)), "U": rng.normal(scale=1 / np.sqrt(D), size=(r, D)),
         "a": rng.normal(scale=0.1, size=r)}
    p = {k: v.astype(np.float32) for k, v in p.items()}
    head = Head.from_arrays(lp, p["C"], p["U"], p["a"], config=eh.AdapterBConfig(rank=r))
    return head, p, lp


def _xor_data(n, D=16, seed=0):
    """Label = sign(x0) XOR sign(x1): no linear function of x separates it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D)).astype(np.float32)
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)


def _mean_ce_f64(logits: np.ndarray, y: np.ndarray) -> float:
    lg = np.asarray(logits, dtype=np.float64)
    lg = lg - lg.max(1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    return float(-np.mean(logp[np.arange(len(y)), y]))


# ------------------------------------------------ 1. zero init == linear baseline, bit exact

@pytest.mark.parametrize("D,r,K,pair", [(16, 4, 2, False), (96, 32, 5, False), (40, 8, 3, True)])
def test_zero_c_gives_the_linear_baseline_logits_exactly(D, r, K, pair):
    lp = _probe_from_arrays(D, K, 3, pair=pair)
    rng = np.random.default_rng(4)
    U = rng.normal(size=(r, D)).astype(np.float32)          # non-zero U and a: the residual must still vanish
    a = rng.normal(size=r).astype(np.float32)
    head = Head.from_arrays(lp, np.zeros((K, r), np.float32), U, a, config=eh.AdapterBConfig(rank=r))
    X = rng.normal(size=(50, head.in_dim)).astype(np.float32) * 10.0
    err = float(np.max(np.abs(head.scores(X) - lp.scores(X))))
    assert err == 0.0
    assert np.array_equal(head.scores(X[0]), lp.scores(X[0]))


def test_fit_starts_exactly_at_the_baseline_and_records_it():
    _needs_training_backend()
    X, y = _xor_data(240, seed=1)
    head = Head.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=4, max_epochs=1))
    assert head.info["identity_max_abs_err_at_init"] == 0.0
    assert head.info["early_stop_scope"] == "none" and head.info["n_train"] == 240


# ------------------------------------------------------- 2. objective monotonicity

def test_objective_trace_is_non_increasing_and_starts_at_the_baseline_ce():
    _needs_training_backend()
    X, y = _xor_data(600, seed=0)
    cfg = eh.AdapterBConfig(rank=8, max_epochs=12)
    head = Head.fit(X, y, 2, pair=False, seed=0, config=cfg)
    info = head.info
    trace = np.asarray(info["objective_trace"], dtype=np.float64)
    assert trace.ndim == 1 and len(trace) >= 2 and np.all(np.isfinite(trace))
    assert np.all(np.diff(trace) <= 0.0)
    assert trace[-1] <= trace[0]
    assert info["J0"] == trace[0] and info["J_final"] == trace[-1]
    # J_0 is the frozen baseline's cross-entropy, recomputed here from the exported W0, b0.
    z = (X - head.a["mu_full"]) / head.a["sd_full"]
    base = z.astype(np.float64) @ head.a["W0"].astype(np.float64).T + head.a["b0"].astype(np.float64)
    assert trace[0] == pytest.approx(_mean_ce_f64(base, y), rel=1e-9, abs=1e-9)
    # Every accepted step in the log is a non-increase; rejected steps never enter the trace.
    accepted = [s for s in info["step_log"] if s["accepted"]]
    assert len(accepted) == len(trace) - 1 == info["accepted_steps"]
    for s in accepted:
        assert s["J_after"] <= s["J_before"]
    assert info["optimizer_status"] in ("converged", "max_epochs", "stalled", "no_step_accepted")
    # The exported NumPy head reproduces the final float64 objective's CE (up to float32 rounding).
    ce_np = _mean_ce_f64(head.scores(X), y)
    assert ce_np == pytest.approx(info["train_ce_final"], abs=1e-4)
    assert info["train_ce_final"] <= info["J_final"] <= info["train_ce_baseline"]


def test_training_lowers_the_objective_on_xor_where_the_baseline_cannot():
    _needs_training_backend()
    X, y = _xor_data(600, seed=0)
    head = Head.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=8, max_epochs=30))
    trace = head.info["objective_trace"]
    assert trace[-1] < trace[0] - 0.1                      # 0.69 -> ~0.18 on this seed
    acc = float(np.mean(head.scores(X).argmax(1) == y))
    base = (X - head.a["mu_full"]) / head.a["sd_full"] @ head.a["W0"].T + head.a["b0"]
    assert acc > float(np.mean(base.argmax(1) == y)) + 0.3


def test_pca_init_is_unit_variance_label_free_and_bounded_by_the_available_directions():
    X, y = _xor_data(300, seed=2)
    z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U0 = Head.pca_init(z, 6)
    assert U0.shape == (6, 16)
    proj = (z - z.mean(0)).astype(np.float64) @ U0.T
    assert np.allclose(proj.var(0, ddof=1), 1.0, atol=1e-6)  # unit-variance projections on the fit rows
    with pytest.raises(ValueError, match="exceeds"):
        Head.pca_init(z, 17)
    with pytest.raises(ValueError, match="exceeds"):
        Head.pca_init(z[:5], 5)


# ------------------------------------------------------- 3. NumPy-only inference

def test_load_and_score_run_with_torch_import_blocked(tmp_path, monkeypatch):
    head, _, _ = _random_trained_like()
    head.save(tmp_path / "b.npz")
    X = np.random.default_rng(7).normal(size=(10, head.in_dim)).astype(np.float32)
    expect = head.scores(X)
    monkeypatch.setitem(sys.modules, "torch", None)      # any `import torch` now raises ImportError
    back = Head.load(tmp_path / "b.npz")
    out = back.scores(X)
    assert type(out) is np.ndarray and out.dtype == np.float32 and out.shape == (10, head.K)
    assert np.array_equal(out, expect)
    assert all(type(v) is np.ndarray for v in back.a.values())
    with pytest.raises(ImportError):
        import torch as _t  # noqa: F401


def test_module_import_and_inference_in_a_fresh_process_without_torch(tmp_path):
    head, _, _ = _random_trained_like()
    head.save(tmp_path / "b.npz")
    X = np.random.default_rng(8).normal(size=(4, head.in_dim)).astype(np.float32)
    np.save(tmp_path / "x.npy", X)
    code = ("import sys; sys.modules['torch'] = None; sys.path.insert(0, %r); "
            "import numpy as np, sota_enhanced_heads as eh; "
            "h = eh.FoldedResidualAdapterBHead.load(%r); np.save(%r, h.scores(np.load(%r))); "
            "print('torch' in sys.modules and sys.modules['torch'] is not None)"
            % (str(SUITES), str(tmp_path / "b.npz"), str(tmp_path / "o.npy"), str(tmp_path / "x.npy")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
    assert np.array_equal(np.load(tmp_path / "o.npy"), head.scores(X))


def test_scores_follow_the_folded_formula_in_float64():
    head, p, lp = _random_trained_like()
    X = np.random.default_rng(9).normal(size=(30, head.in_dim)).astype(np.float32)
    z = ((X - lp.a["mu_full"]) / lp.a["sd_full"]).astype(np.float64)
    ref = (z @ lp.a["W"].astype(np.float64).T + lp.a["b"]
           + eh.gelu_tanh(z @ p["U"].astype(np.float64).T + p["a"]) @ p["C"].astype(np.float64).T)
    assert np.max(np.abs(head.scores(X) - ref)) <= 1e-4 * (1.0 + np.max(np.abs(ref)))


def test_save_load_roundtrip_is_bit_exact(tmp_path):
    head, _, _ = _random_trained_like()
    head.save(tmp_path / "b.npz")
    back = Head.load(tmp_path / "b.npz")
    assert back.cfg == head.cfg
    for k in Head.ARRAY_KEYS:
        assert np.array_equal(back.a[k], head.a[k]) and back.a[k].dtype == np.float32
    with np.load(tmp_path / "b.npz", allow_pickle=False) as z:
        assert set(z.files) == set(Head.ARRAY_KEYS) | {"cfg_json"}
        assert json.loads(str(z["cfg_json"]))["head_type"] == "adapter_b"


def test_trained_head_export_matches_the_torch_forward_and_survives_reload(tmp_path):
    _needs_training_backend()
    X, y = _xor_data(300, seed=5)
    head = Head.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=4, max_epochs=3))
    assert head.info["export_max_abs_err"] <= 1e-4 * (1.0 + float(np.max(np.abs(head.scores(X)))))
    head.save(tmp_path / "t.npz")
    assert np.array_equal(Head.load(tmp_path / "t.npz").scores(X), head.scores(X))


# ----------------------------------------------------------- 4. input defense

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_rejected_at_inference_and_at_fit(bad):
    head, _, _ = _random_trained_like()
    X = np.zeros((3, head.in_dim), np.float32)
    X[1, 5] = bad
    with pytest.raises(ValueError, match="non-finite"):
        head.scores(X)
    with pytest.raises(ValueError, match="non-finite"):
        head.score(X[1], np.zeros((head.K, 1)))
    Xt, yt = _xor_data(40)
    Xt[3, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        Head.fit(Xt, yt, 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=2))


def test_wrong_width_wrong_k_and_bad_labels_are_rejected():
    head, _, _ = _random_trained_like(D=96, K=5)
    with pytest.raises(ValueError, match="expects"):
        head.scores(np.zeros((2, 95), np.float32))
    with pytest.raises(ValueError, match="expects"):
        head.scores(np.zeros((2, 3, 96), np.float32))
    with pytest.raises(ValueError, match="K=5"):
        head.score(np.zeros(96, np.float32), np.zeros((4, 1)))
    X, y = _xor_data(40)
    with pytest.raises(ValueError, match="labels"):
        Head.fit(X, y.astype(np.float32), 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=2))
    with pytest.raises(ValueError, match="labels"):
        Head.fit(X, y + 1, 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=2))
    with pytest.raises(ValueError, match="labels"):
        Head.fit(X, y - 1, 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=2))
    with pytest.raises(ValueError, match="N == len"):
        Head.fit(X, y[:-1], 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=2))
    with pytest.raises(ValueError, match="two classes"):
        Head.fit(X, np.zeros(40, np.int64), 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=2))
    with pytest.raises(TypeError):
        Head.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=2))


def test_malformed_arrays_and_foreign_heads_are_rejected(tmp_path):
    head, p, lp = _random_trained_like(D=96, r=32, K=5)
    good = dict(head.a)
    with pytest.raises(ValueError, match="missing"):
        Head({k: v for k, v in good.items() if k != "C"}, head.cfg)
    with pytest.raises(ValueError, match="shape"):
        Head(dict(good, C=np.zeros((5, 31), np.float32)), head.cfg)
    with pytest.raises(ValueError, match="shape"):
        Head(dict(good, U=np.zeros((31, 96), np.float32)), head.cfg)
    bad = good["U"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Head(dict(good, U=bad), head.cfg)
    with pytest.raises(ValueError, match="in_dim"):
        Head(good, dict(head.cfg, in_dim=97))
    with pytest.raises(ValueError, match="cannot load"):
        Head(good, dict(head.cfg, head_type="adapter"))
    with pytest.raises(ValueError, match="adapter_b needs"):
        Head.from_arrays(sx.LinearProbe(dict(lp.a, pca_mu=np.zeros(96, np.float32)), dict(lp.cfg, pca_k=8)),
                         p["C"], p["U"], p["a"], config=eh.AdapterBConfig(rank=32))
    other = _probe_from_arrays(96, 5, 1)
    other.save(tmp_path / "lp.npz")
    with pytest.raises(ValueError):
        Head.load(tmp_path / "lp.npz")


@pytest.mark.parametrize("kw", [{"rank": 0}, {"rank": True}, {"rank": 2.0}, {"lambda_C": -1e-3},
                                {"lambda_C": float("nan")}, {"lambda_theta": float("inf")}, {"lr": 0.0},
                                {"lr": -1.0}, {"lr": float("nan")}, {"max_epochs": 0}, {"max_epochs": 1.5},
                                {"device": "tpu"}, {"max_backtracks": 0}, {"rel_tol": 0.0}])
def test_adapter_b_config_rejects_bad_fields(kw):
    with pytest.raises(ValueError):
        eh.AdapterBConfig(**kw)


def test_adapter_b_config_defaults_match_the_task_sheet():
    c = eh.AdapterBConfig()
    assert (c.rank, c.lambda_C, c.lambda_theta, c.lr, c.max_epochs, c.device) == (64, 1e-3, 1e-3, 1e-3, 50, "cpu")
    assert eh.AdapterBConfig(device="cuda").device == "cuda" and eh.AdapterBConfig(device="auto").device == "auto"
    assert eh.head_config({"type": "adapter_b", "rank": 32}) == eh.AdapterBConfig(rank=32)


# --------------------------------------------------- 5. row isolation and parameter count

def test_training_reads_only_the_rows_passed():
    """NaN sentinel rows sit outside `tr`; _fit_predict must neither fail nor read them."""
    _needs_training_backend()
    X, y = _xor_data(360, seed=9)
    X[::6] = np.nan
    tr = np.flatnonzero(np.arange(360) % 6 != 0)
    f = {"X_train": X, "train_label": y, "cands": np.zeros((2, 1), dtype=np.float32)}
    m, s, info = bse._fit_predict({"type": "adapter_b", "rank": 8, "max_epochs": 2}, "massive_en", f, tr, tr[:40],
                                  X[tr[:40]], [0, 0, 100])
    assert isinstance(m, Head)
    assert s.shape == (40, 2) and np.all(np.isfinite(s))
    assert info["n_train"] == len(tr) and info["n_early_stop"] == 0
    assert info["params"] == m.supervised_param_count()
    assert all(b <= a for a, b in zip(info["objective_trace"], info["objective_trace"][1:]))


@pytest.mark.parametrize("D,r,K", [(16, 4, 2), (8192, 64, 18), (300, 32, 7)])
def test_supervised_param_count_matches_the_spec_formula(D, r, K):
    lp = _probe_from_arrays(D, K, 0)
    head = Head.from_arrays(lp, np.zeros((K, r)), np.zeros((r, D)), np.zeros(r), config=eh.AdapterBConfig(rank=r))
    assert head.supervised_param_count() == K * (D + 1) + r * (D + 1) + K * r
    assert head.supervised_param_count() == sum(head.a[k].size for k in ("W0", "b0", "C", "U", "a"))


def test_spec_table_values():
    """Spec 20 S4.3: D8192, K18, r64 -> 525504 new parameters plus 147474 baseline parameters."""
    D, K, r = 8192, 18, 64
    lp = _probe_from_arrays(D, K, 0)
    head = Head.from_arrays(lp, np.zeros((K, r)), np.zeros((r, D)), np.zeros(r), config=eh.AdapterBConfig(rank=r))
    assert r * (D + 1) + K * r == 525504 and K * (D + 1) == 147474
    assert head.supervised_param_count() == 525504 + 147474


def test_pair_source_uses_the_full_context_third_only():
    head, _, _ = _random_trained_like(D=40, r=8, K=3, pair=True)
    assert head.in_dim == 120
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 120)).astype(np.float32)
    X2 = X.copy()
    X2[:, 40:] = rng.normal(size=(6, 80))
    assert np.array_equal(head.scores(X), head.scores(X2))


# --------------------------------------------------------- 6. ensemble wiring

def test_expert_specs_register_adapter_b_only_when_enabled(monkeypatch):
    fs = {"enc": {"train_full": np.zeros((5, 16))}}
    monkeypatch.setattr(bse, "ADAPTER_B_RANKS", ())
    assert not [n for n, _, _ in bse.expert_specs("massive_en", fs) if "adapter_b" in n]
    monkeypatch.setattr(bse, "ADAPTER_B_RANKS", (32, 64))
    specs = {n: sp for n, _, sp in bse.expert_specs("massive_en", fs)}
    for r in (32, 64):
        assert specs[f"enc:adapter_b_r{r}"] == {"type": "adapter_b", "rank": r}
        assert eh.head_config(specs[f"enc:adapter_b_r{r}"]) == eh.AdapterBConfig(rank=r)
    assert "enc:adapter_r32" not in specs                 # Formulation A stays off
    assert bse.parse_adapter_b_ranks("", True) == (32, 64)
    assert bse.parse_adapter_b_ranks("", False) == ()
    assert bse.parse_adapter_b_ranks("128", True) == (128,)
    with pytest.raises(ValueError):
        bse.parse_adapter_b_ranks("64,64", False)


def test_registry_and_dispatch_paths_know_adapter_b():
    assert eh.HEADS["adapter_b"] is Head and eh.CONFIGS["adapter_b"] is eh.AdapterBConfig
    assert Head.head_type == "adapter_b" and Head.CONFIG_CLS is eh.AdapterBConfig
    # strategy_complexity reads the per-fold params of any eh.HEADS type
    tier, params = bse.strategy_complexity("single:enc:adapter_b_r4", {"enc:adapter_b_r4": {"type": "adapter_b", "rank": 4}},
                                           {"enc:adapter_b_r4": {"folds": [{"params": 11}, {"params": 13}]}}, 2, {})
    assert (tier, params) == (0, 13)


def test_load_expert_reads_a_saved_adapter_b(tmp_path, monkeypatch):
    head, _, _ = _random_trained_like(D=16, r=4, K=2)
    monkeypatch.setattr(bse, "OUT_ART", tmp_path)
    (tmp_path / "models" / "massive_en").mkdir(parents=True)
    head.save(tmp_path / "models" / "massive_en" / (bse.safe("enc:adapter_b_r4") + ".npz"))
    back = bse.load_expert("massive_en", "enc:adapter_b_r4", "enc", {"type": "adapter_b", "rank": 4}, {})
    X = np.random.default_rng(1).normal(size=(3, 16)).astype(np.float32)
    assert isinstance(back, Head) and np.array_equal(back.scores(X), head.scores(X))


def test_cli_flags_parse(monkeypatch, tmp_path):
    pytest.importorskip("torch")                          # main() imports torch for the fit stage
    for g in ("OUT_ART", "RESULTS", "PRIOR_REPORT", "ADAPTER_RANKS", "SUPCON_RANKS", "ADAPTER_B_RANKS",
              "N_FOLDS", "FOLD_SEED", "NESTED_SEED", "RAW_MAX_DIM", "MOE_THREADS", "ENABLE_RDA",
              "ENABLE_NYSTROM", "RNN_SOURCES", "RNN_HEADS"):
        monkeypatch.setattr(bse, g, getattr(bse, g))      # main() rebinds these; restore them after
    monkeypatch.setattr(bse, "SOURCES", dict(bse.SOURCES))
    monkeypatch.setattr(bse.gd, "TEST_DIR", bse.gd.TEST_DIR)
    seen = {}

    def fake_stage_fit(tasks, only, device):
        seen["ranks"] = bse.ADAPTER_B_RANKS

    monkeypatch.setattr(bse, "stage_fit", fake_stage_fit)
    monkeypatch.setattr(bse.sx, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(sys, "argv", ["prog", "fit", "--source", f"enc={tmp_path}", "--enable-adapter-b",
                                      "--tasks", "massive_en", "--out-art", str(tmp_path)])
    bse.main()
    assert seen["ranks"] == (32, 64)
    monkeypatch.setattr(sys, "argv", ["prog", "fit", "--source", f"enc={tmp_path}", "--adapter-b-ranks", "16,128",
                                      "--tasks", "massive_en", "--out-art", str(tmp_path)])
    bse.main()
    assert seen["ranks"] == (16, 128)


# ------------------------------------------------- 7. the acceptance gate must actually fire

def test_a_c_step_that_raises_the_objective_is_rejected_and_rescued_by_backtracking(monkeypatch):
    """Replace L-BFGS by a solver that perturbs C at random: J would go up, the gate must refuse it."""
    torch = _needs_training_backend()
    real = torch.optim.LBFGS

    class BadLBFGS(real):
        def step(self, closure):
            closure()
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        p.add_(torch.randn_like(p) * 50.0)
            return None

    monkeypatch.setattr(torch.optim, "LBFGS", BadLBFGS)
    X, y = _xor_data(300, seed=3)
    head = Head.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterBConfig(rank=4, max_epochs=3))
    info = head.info
    assert info["lbfgs_rejections"] == 3
    c_steps = [s for s in info["step_log"] if s["step"] == "C"]
    assert all(s["method"] == "backtracking_gd" and s["accepted"] for s in c_steps)
    trace = info["objective_trace"]
    assert all(b <= a for a, b in zip(trace, trace[1:])) and trace[-1] <= trace[0]
    assert np.all(np.isfinite(head.a["C"])) and float(np.abs(head.a["C"]).max()) < 50.0


def test_a_theta_step_that_breaks_the_upper_bound_is_rejected_and_leaves_the_trace_untouched():
    _needs_training_backend()
    X, y = _xor_data(300, seed=3)
    cfg = eh.AdapterBConfig(rank=4, lr=1e9, max_backtracks=1, max_epochs=1)
    head = Head.fit(X, y, 2, pair=False, seed=0, config=cfg)
    info = head.info
    theta = [s for s in info["step_log"] if s["step"] == "theta"]
    assert len(theta) == 1 and theta[0]["accepted"] is False and theta[0]["J_after"] is None
    assert info["theta_backtrack_failures"] == 1 and info["rejected_steps"] == 1
    trace = info["objective_trace"]
    assert len(trace) == 1 + info["accepted_steps"] == 2          # only the C step entered the trace
    assert trace[-1] <= trace[0]
    # theta stayed at theta_0: the exported U is exactly the PCA init on these rows
    z = (X - head.a["mu_full"]) / head.a["sd_full"]
    assert np.allclose(head.a["U"], Head.pca_init(z, 4).astype(np.float32), atol=1e-6)
    assert np.array_equal(head.a["a"], np.zeros(4, np.float32))
    assert info["theta_dist_from_init"] == 0.0
