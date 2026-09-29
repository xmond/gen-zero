"""GPU option for the offline fit pipeline: device resolution, CPU no-regression, GPU code paths.

The rule under test: fitting may use a GPU, inference never does. Whatever device a head or a
probe was fitted on, what comes back is float32 NumPy, the .npz layout is unchanged and score()
runs with torch unimportable.

What runs where (read this before trusting a green run):
  * Real CUDA tests are marked `needs_cuda` and SKIP on a box without a GPU. On the dev box
    they are skipped; they only mean something on the A100 server.
  * `fake_cuda` pretends CUDA exists but keeps every tensor on the host: it makes
    torch.cuda.is_available() True and redirects the `device="cuda"` argument of torch.as_tensor /
    Module.to to the CPU, recording each request. That proves the code ASKS for the device on
    every training tensor and that the results are device-plumbing-neutral; it does NOT prove
    that CUDA kernels give the same numbers. Only `needs_cuda` tests can.
  * The torch L-BFGS solver is CPU-testable (it is plain torch), so its agreement with
    scikit-learn is measured for real here.

All data is synthetic; no task files. Tests that call LinearProbe.fit need scikit-learn.
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(SUITES))

torch = pytest.importorskip("torch")
import benchmark_sota_ensemble as bse  # noqa: E402
import grand_challenge_data as gd  # noqa: E402
import sota_enhanced_heads as eh  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402

torch.set_num_threads(2)

HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device on this box")
no_cuda_box = pytest.mark.skipif(HAS_CUDA, reason="asserts behaviour on a box WITHOUT CUDA")


# ----------------------------------------------------------------------------- helpers

def _xor_data(n, D=16, seed=0):
    """Label = sign(x0) XOR sign(x1): no linear function separates it, so the adapter has work to do."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D)).astype(np.float32)
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)


def _blob_data(n, D, K, seed=0, noise=3.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D))
    y = (X @ rng.normal(size=(K, D)).T + rng.gumbel(size=(n, K)) * noise).argmax(1)
    return X, y


def _fixed_probe(X, y, K, pair=False):
    """A `lin_full` probe with fixed, non-degenerate weights and no sklearn: the same object for
    both devices, so a CPU/GPU comparison isolates the training loop."""
    rng = np.random.default_rng(3)
    D = X.shape[1] // 3 if pair else X.shape[1]
    full = X[:, :D]
    arrays = {"mu_full": full.mean(0).astype(np.float32), "sd_full": (full.std(0) + 1e-6).astype(np.float32),
              "W": (rng.normal(size=(K, D)) * 0.1).astype(np.float32), "b": np.zeros(K, dtype=np.float32)}
    cfg = {"K": K, "pair": pair, "kind": "full", "pca_k": None, "in_dim": X.shape[1], "C": 1.0, "feature_dim": D}
    return sx.LinearProbe(arrays, cfg)


class _Requests:
    """What the code asked torch to put on the (fake) GPU."""

    def __init__(self):
        self.tensors, self.modules = [], []

    def all_cuda(self):
        return all(d == "cuda" for d in self.tensors + self.modules)


@pytest.fixture
def fake_cuda(monkeypatch):
    """CUDA 'exists', every tensor stays on the host, every request for it is recorded."""
    req = _Requests()
    real_as_tensor, real_module_to = torch.as_tensor, torch.nn.Module.to

    def as_tensor(data, *a, **kw):
        dev = kw.get("device")
        if dev is not None and torch.device(dev).type == "cuda":
            req.tensors.append(str(dev))
            kw = dict(kw, device="cpu")
        return real_as_tensor(data, *a, **kw)

    def module_to(self, *a, **kw):
        if a and isinstance(a[0], (str, torch.device)):
            self._fake_device = torch.device(a[0]).type      # where the module would live on a real GPU
            if self._fake_device == "cuda":
                req.modules.append(str(a[0]))
                a = ("cpu",) + a[1:]
        return real_module_to(self, *a, **kw)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "as_tensor", as_tensor)
    monkeypatch.setattr(torch.nn.Module, "to", module_to)
    return req


def _assert_numpy_head(head):
    assert isinstance(head, eh.DeepResidualAdapterHead)
    for k in ("W_h", "W_fold", "W_down", "b_down", "b_fold", "mu_full", "sd_full"):
        v = head.a[k]
        assert isinstance(v, np.ndarray) and type(v) is np.ndarray and v.dtype == np.float32, k
    assert "W_up" not in head.a and "b_h" not in head.a          # folded export, as before


def _score_in_fresh_process_without_torch(npz: Path, loader: str, X: np.ndarray, tmp: Path) -> np.ndarray:
    """A pure-CPU consumer: a new interpreter with torch unimportable loads the .npz and scores."""
    np.save(tmp / "x.npy", X)
    code = ("import sys; sys.modules['torch'] = None; sys.path.insert(0, %r); "
            "import numpy as np, sota_enhanced_heads as eh, sota_ensemble_experts as sx; "
            "m = %s.load(%r); np.save(%r, m.scores(np.load(%r)))"
            % (str(SUITES), loader, str(npz), str(tmp / "o.npy"), str(tmp / "x.npy")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    return np.load(tmp / "o.npy")


# ------------------------------------------------------------ 1. device resolution

def test_resolve_cpu_is_cpu_even_when_cuda_exists(monkeypatch):
    monkeypatch.setattr(sx, "cuda_available", lambda: True)
    assert sx.resolve_device("cpu") == "cpu"


def test_resolve_auto_without_cuda_is_cpu(monkeypatch):
    monkeypatch.setattr(sx, "cuda_available", lambda: False)
    assert sx.resolve_device("auto") == "cpu"


def test_resolve_auto_with_cuda_is_cuda(monkeypatch):
    monkeypatch.setattr(sx, "cuda_available", lambda: True)
    assert sx.resolve_device("auto") == "cuda"
    assert sx.resolve_device("cuda") == "cuda"
    assert sx.resolve_device("cuda:1") == "cuda:1"


def test_resolve_explicit_cuda_without_cuda_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(sx, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="is_available"):
        sx.resolve_device("cuda")
    with pytest.raises(RuntimeError):
        sx.resolve_device("cuda:0")


@pytest.mark.parametrize("bad", ["gpu", "CUDA", "cuda:", "cuda:x", "", "cpu:0", None, 0])
def test_resolve_rejects_unknown_names(bad):
    with pytest.raises(ValueError, match="device must be"):
        sx.resolve_device(bad)


def test_cuda_available_is_false_when_torch_is_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert sx.cuda_available() is False


def test_adapter_config_device_defaults_to_auto_and_is_validated():
    assert eh.AdapterConfig().device == "auto"
    assert eh.AdapterConfig(device="cpu").device == "cpu"
    with pytest.raises(ValueError, match="device must be"):
        eh.AdapterConfig(device="tpu")


def test_head_config_reads_device_from_a_pipeline_spec():
    assert eh.head_config({"type": "adapter", "rank": 8, "device": "cpu"}).device == "cpu"


# ------------------------------------------- 2. CPU mode: no deviation from the baseline

def test_linear_probe_cpu_default_is_exactly_the_direct_sklearn_fit():
    """device defaults to 'cpu' and then the fit IS scikit-learn's LogisticRegressionCV, same args."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    X, y = _blob_data(160, 24, 3, seed=1)
    lp = sx.LinearProbe.fit(X, y, 3, pair=False, kind="full", pca_k=None, seed=5, n_jobs=1)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    clf = LogisticRegressionCV(Cs=sx.cs_grid(None), cv=StratifiedKFold(4, shuffle=True, random_state=5),
                               scoring="neg_log_loss", max_iter=3000, n_jobs=1).fit(Z, y)
    assert np.array_equal(lp.a["W"], clf.coef_.astype(np.float32))
    assert np.array_equal(lp.a["b"], clf.intercept_.astype(np.float32))
    assert lp.cfg["C"] == float(np.ravel(clf.C_)[0])
    assert lp.fit_info == {"device": "cpu", "solver": "sklearn-lbfgs"}
    explicit = sx.LinearProbe.fit(X, y, 3, pair=False, kind="full", pca_k=None, seed=5, n_jobs=1, device="cpu")
    assert np.array_equal(explicit.a["W"], lp.a["W"]) and np.array_equal(explicit.a["b"], lp.a["b"])


@no_cuda_box
def test_linear_probe_auto_without_cuda_is_the_cpu_fit():
    pytest.importorskip("sklearn")
    X, y = _blob_data(120, 16, 3, seed=2)
    a = sx.LinearProbe.fit(X, y, 3, pair=False, kind="full", pca_k=None, seed=1, n_jobs=1, device="auto")
    c = sx.LinearProbe.fit(X, y, 3, pair=False, kind="full", pca_k=None, seed=1, n_jobs=1, device="cpu")
    assert a.fit_info["solver"] == "sklearn-lbfgs"
    assert np.array_equal(a.a["W"], c.a["W"]) and np.array_equal(a.a["b"], c.a["b"])


@no_cuda_box
def test_adapter_auto_without_cuda_is_bit_identical_to_cpu():
    X, y = _xor_data(240, seed=4)
    probe = lambda Xp, yp: _fixed_probe(Xp, yp, 2)  # noqa: E731
    a = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=8),
                                       probe_fit=probe)                      # config.device == "auto"
    c = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=8),
                                       probe_fit=probe, device="cpu")
    assert a.info["device"] == c.info["device"] == "cpu"
    for k in a.a:
        assert np.array_equal(a.a[k], c.a[k]), k


def test_adapter_saved_config_and_npz_keys_do_not_record_the_device(tmp_path):
    X, y = _xor_data(200, seed=5)
    head = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=8, device="cpu"),
                                          probe_fit=lambda Xp, yp: _fixed_probe(Xp, yp, 2))
    assert "device" not in head.cfg["config"]
    head.save(tmp_path / "h.npz")
    with np.load(tmp_path / "h.npz", allow_pickle=False) as z:
        assert sorted(z.files) == sorted(["cfg_json", *eh.DeepResidualAdapterHead.FOLDED_KEYS])


def test_train_rnn_cpu_is_deterministic_and_returns_a_cpu_model(monkeypatch):
    monkeypatch.setattr(sx, "EPOCHS", 2)
    rng = np.random.default_rng(6)
    q, cands, y = rng.normal(size=(80, 256)).astype(np.float32), rng.normal(size=(4, 256)).astype(np.float32), rng.integers(0, 4, 80)
    m1, i1 = sx.train_rnn("baseline", (q,), y, cands, [0, 1], device="cpu")
    m2, i2 = sx.train_rnn("baseline", (q,), y, cands, [0, 1], device="cpu")
    assert i1["device"] == "cpu"
    assert all(p.device.type == "cpu" for p in m1.parameters())
    for (k, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k


# --------------------------------- 3. the torch L-BFGS solver vs scikit-learn (real numbers)

@pytest.mark.parametrize("K,n,D,seed", [(2, 240, 30, 0), (4, 300, 50, 1), (3, 150, 200, 2)])
def test_torch_solver_picks_the_same_C_and_lands_on_the_sklearn_optimum(K, n, D, seed):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    X, y = _blob_data(n, D, K, seed=seed)
    Cs = sx.cs_grid(None)
    cv = StratifiedKFold(4, shuffle=True, random_state=1)
    clf = LogisticRegressionCV(Cs=Cs, cv=cv, scoring="neg_log_loss", max_iter=3000).fit(X, y)
    classes, coef, icpt, best_C, info = sx._logreg_cv_torch(X, y, Cs, cv, "cpu")
    assert np.array_equal(classes, clf.classes_)
    assert best_C == float(np.ravel(clf.C_)[0])
    assert coef.shape == clf.coef_.shape and icpt.shape == clf.intercept_.shape       # binary: one row, as sklearn
    assert info["max_abs_grad"] < sx.LOGREG_GRAD_OK
    # Against a tightly converged sklearn solution at that C, torch is at least as close as sklearn's own
    # default-tol answer, and both are tiny.
    exact = LogisticRegression(C=best_C, tol=1e-12, max_iter=20000).fit(X, y)
    err_torch = max(np.abs(coef - exact.coef_).max(), np.abs(icpt - exact.intercept_).max())
    err_sk = np.abs(clf.coef_ - exact.coef_).max()
    assert err_torch < 5e-4, err_torch
    assert err_torch <= err_sk + 2e-4, (err_torch, err_sk)


def test_torch_solver_gradient_check_on_a_hand_sized_problem():
    """Independent of sklearn: at the solution the penalized-loss gradient is ~0 in float64."""
    X, y = _blob_data(90, 8, 3, seed=7)
    Ft, yt = torch.as_tensor(X), torch.as_tensor(y)
    W, b, info = sx._logreg_lbfgs(Ft, yt, 3, C=0.5)
    W, b = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(Ft @ W.T + b, yt) + 0.5 / (0.5 * 90) * W.pow(2).sum()
    loss.backward()
    assert float(max(W.grad.abs().max(), b.grad.abs().max())) < 1e-5


def test_torch_solver_rejects_a_single_class():
    pytest.importorskip("sklearn")
    from sklearn.model_selection import StratifiedKFold
    X = np.random.default_rng(0).normal(size=(20, 4))
    with pytest.raises(ValueError, match="at least 2 classes"):
        sx._logreg_cv_torch(X, np.zeros(20, dtype=int), [1.0], StratifiedKFold(2), "cpu")


# ------------------------------------ 4. GPU code paths under fake_cuda (host tensors)

def test_linear_probe_gpu_branch_exports_float32_numpy_and_matches_the_cpu_probe(fake_cuda):
    pytest.importorskip("sklearn")
    X, y = _blob_data(200, 20, 4, seed=3)
    gpu = sx.LinearProbe.fit(X, y, 4, pair=False, kind="full", pca_k=None, seed=2, n_jobs=1, device="cuda")
    cpu = sx.LinearProbe.fit(X, y, 4, pair=False, kind="full", pca_k=None, seed=2, n_jobs=1, device="cpu")
    assert gpu.fit_info["solver"] == "torch-lbfgs" and gpu.fit_info["device"] == "cuda"
    assert fake_cuda.tensors and fake_cuda.all_cuda()
    for k in ("W", "b", "mu_full", "sd_full"):
        assert type(gpu.a[k]) is np.ndarray and gpu.a[k].dtype == np.float32, k
    assert sorted(gpu.a) == sorted(cpu.a) and gpu.cfg.keys() == cpu.cfg.keys()        # same layout
    assert gpu.cfg["C"] == cpu.cfg["C"]
    assert np.max(np.abs(gpu.scores(X) - cpu.scores(X))) < 5e-3
    assert np.array_equal(gpu.scores(X).argmax(1), cpu.scores(X).argmax(1))


def test_linear_probe_gpu_branch_keeps_absent_classes_unreachable_and_binary_layout(fake_cuda):
    """K=5 but only classes {0, 3} occur: binary logit [0, z] lands on rows 0 and 3, the rest sit far below."""
    pytest.importorskip("sklearn")
    X, y2 = _blob_data(160, 12, 2, seed=4)
    y = np.where(y2 == 0, 0, 3)
    gpu = sx.LinearProbe.fit(X, y, 5, pair=False, kind="full", pca_k=None, seed=0, n_jobs=1, device="cuda")
    cpu = sx.LinearProbe.fit(X, y, 5, pair=False, kind="full", pca_k=None, seed=0, n_jobs=1, device="cpu")
    for absent in (1, 2, 4):
        assert np.all(gpu.a["W"][absent] == 0) and gpu.a["b"][absent] < -1e3
    s = gpu.scores(X)
    assert set(np.unique(s.argmax(1))) <= {0, 3}
    assert np.array_equal(s.argmax(1), cpu.scores(X).argmax(1))


def test_linear_probe_gpu_npz_loads_and_scores_in_a_process_without_torch(fake_cuda, tmp_path):
    pytest.importorskip("sklearn")
    X, y = _blob_data(150, 14, 3, seed=5)
    lp = sx.LinearProbe.fit(X, y, 3, pair=False, kind="full", pca_k=None, seed=0, n_jobs=1, device="cuda")
    lp.save(tmp_path / "lp.npz")
    got = _score_in_fresh_process_without_torch(tmp_path / "lp.npz", "sx.LinearProbe", X.astype(np.float32), tmp_path)
    assert np.array_equal(got, lp.scores(X.astype(np.float32)))


def test_cuda_requested_without_cuda_raises_in_every_fit_entry_point(monkeypatch):
    monkeypatch.setattr(sx, "cuda_available", lambda: False)
    X, y = _xor_data(60)
    with pytest.raises(RuntimeError):
        sx.LinearProbe.fit(X, y, 2, pair=False, kind="full", pca_k=None, seed=0, device="cuda")
    with pytest.raises(RuntimeError):
        eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=4),
                                       probe_fit=lambda Xp, yp: _fixed_probe(Xp, yp, 2), device="cuda")
    with pytest.raises(RuntimeError):
        sx.train_rnn("baseline", (np.zeros((4, 256), dtype=np.float32),), np.zeros(4, dtype=int),
                     np.zeros((2, 256), dtype=np.float32), [0, 0], device="cuda")


def test_adapter_gpu_branch_puts_every_training_tensor_on_the_device_and_exports_numpy(fake_cuda, tmp_path):
    X, y = _xor_data(300, seed=6)
    probe = lambda Xp, yp: _fixed_probe(Xp, yp, 2)  # noqa: E731
    gpu = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=8),
                                         probe_fit=probe, device="cuda")
    assert gpu.info["device"] == "cuda"
    # 6 parameters + standardized features + labels + early-stop index + one permutation per epoch.
    assert len(fake_cuda.tensors) >= 9 + gpu.info["epochs_run"], fake_cuda.tensors
    assert fake_cuda.all_cuda()
    _assert_numpy_head(gpu)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cpu = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=8),
                                             probe_fit=probe, device="cpu")
    # Same arithmetic on the same host tensors: the device plumbing must not move a single bit.
    for k in cpu.a:
        assert np.array_equal(gpu.a[k], cpu.a[k]), k
    assert gpu.info["best_epoch"] == cpu.info["best_epoch"]
    # A pure-CPU consumer loads the saved file and scores identically.
    gpu.save(tmp_path / "h.npz")
    Xs = X[:20]
    got = _score_in_fresh_process_without_torch(tmp_path / "h.npz", "eh.DeepResidualAdapterHead", Xs, tmp_path)
    assert type(got) is np.ndarray and np.array_equal(got, gpu.scores(Xs))
    assert np.max(np.abs(got - cpu.scores(Xs))) < 1e-4


def test_adapter_default_probe_is_fitted_on_the_same_device(fake_cuda, monkeypatch):
    pytest.importorskip("sklearn")
    seen = []
    orig = sx.LinearProbe.fit
    monkeypatch.setattr(sx.LinearProbe, "fit",
                        classmethod(lambda cls, *a, **kw: (seen.append(kw.get("device")), orig(*a, **kw))[1]))
    X, y = _xor_data(200, seed=7)
    head = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=eh.AdapterConfig(rank=4),
                                          device="cuda")
    assert seen == ["cuda"]
    _assert_numpy_head(head)


def test_train_rnn_gpu_branch_trains_on_the_device_and_hands_back_a_cpu_model(fake_cuda, monkeypatch, tmp_path):
    monkeypatch.setattr(sx, "EPOCHS", 2)
    rng = np.random.default_rng(8)
    q, cands, y = rng.normal(size=(80, 256)).astype(np.float32), rng.normal(size=(4, 256)).astype(np.float32), rng.integers(0, 4, 80)
    gpu, info = sx.train_rnn("baseline", (q,), y, cands, [0, 1], device="cuda")
    assert info["device"] == "cuda" and fake_cuda.modules == ["cuda"]
    assert len(fake_cuda.tensors) >= 4 + info["best_epoch"]          # q, C, y, early-stop idx, perms
    assert fake_cuda.all_cuda()
    assert all(p.device.type == "cpu" for p in gpu.parameters()) and not gpu.training
    assert gpu._fake_device == "cpu"          # it went to the device for training and was moved back for export
    cpu, _ = sx.train_rnn("baseline", (q,), y, cands, [0, 1], device="cpu")
    for (k, a), (_, b) in zip(gpu.state_dict().items(), cpu.state_dict().items()):
        assert torch.equal(a, b), k
    # Export and the NumPy runtime work on the returned model.
    gpu.export_npz(tmp_path / "rnn.npz", {"task": "t", "config": "baseline"})
    rt = sx.RNNRuntimeExpertCore("baseline", tmp_path / "rnn.npz", pair=False)
    ref = sx.rnn_scores(gpu, sx.rnn_query("baseline", q[:5], False), cands)
    out = np.stack([rt.score(x, cands) for x in q[:5]])
    assert isinstance(out, np.ndarray) and np.max(np.abs(out - ref)) < 1e-3


# ------------------------------------------- 5. pipeline wiring: _fit_predict / stage_fit / CLI

def test_fit_predict_forwards_device_to_every_fit_kind(monkeypatch):
    seen = {}

    class _M:
        info: dict = {}
        cfg = {"C": 1.0, "feature_dim": 3}

        def scores(self, X):
            return np.zeros((len(X), 2), dtype=np.float32)

        def supervised_param_count(self):
            return 0

    def spy(name):
        def fit(*a, **kw):
            seen[name] = kw.get("device")
            return _M()
        return fit

    monkeypatch.setattr(sx.LinearProbe, "fit", classmethod(lambda cls, *a, **kw: spy("linear")(*a, **kw)))
    monkeypatch.setattr(eh.DeepResidualAdapterHead, "fit", classmethod(lambda cls, *a, **kw: spy("adapter")(*a, **kw)))
    monkeypatch.setattr(sx, "train_rnn", lambda *a, **kw: (seen.__setitem__("rnn", kw.get("device")) or (object(), {})))
    monkeypatch.setattr(sx, "rnn_scores", lambda *a, **kw: np.zeros((2, 2), dtype=np.float32))
    X = np.zeros((6, 3), dtype=np.float32)
    f = {"X_train": X, "train_label": np.array([0, 1] * 3), "cands": np.zeros((2, 3), dtype=np.float32)}
    tr = np.arange(6)
    for spec in ({"type": "linear", "kind": "full", "pca_k": None}, {"type": "adapter", "rank": 4},
                 {"type": "rnn", "config": "baseline"}):
        bse._fit_predict(spec, "massive_en", f, tr, tr[:2], X[:2], [0, 0, 0], device="cuda")
    assert seen == {"linear": "cuda", "adapter": "cuda", "rnn": "cuda"}
    seen.clear()
    bse._fit_predict({"type": "linear", "kind": "full", "pca_k": None}, "massive_en", f, tr, tr[:2], X[:2], [0, 0, 0])
    assert seen == {"linear": "cpu"}                    # callers that never heard of device stay on the CPU


def _run_main(monkeypatch, capsys, argv):
    """Run bse.main() with every module global it rewrites restored afterwards; stage_fit is recorded."""
    for name in ("OUT_ART", "RESULTS", "RNN_SOURCES", "RNN_HEADS", "PRIOR_REPORT", "ADAPTER_RANKS", "N_FOLDS",
                 "FOLD_SEED", "NESTED_SEED", "RAW_MAX_DIM", "MOE_THREADS"):
        monkeypatch.setattr(bse, name, getattr(bse, name))
    monkeypatch.setattr(bse, "SOURCES", dict(bse.SOURCES))
    monkeypatch.setattr(gd, "TEST_DIR", gd.TEST_DIR)
    calls = []
    monkeypatch.setattr(bse, "stage_fit", lambda tasks, only, device="cpu": calls.append(device))
    monkeypatch.setattr(sys, "argv", ["benchmark_sota_ensemble.py", "fit", "--source", "s=/nonexistent",
                                      "--tasks", gd.TASKS[0], "--torch-threads", "2", *argv])
    bse.main()
    return calls, capsys.readouterr().out


@no_cuda_box
def test_cli_auto_is_cpu_without_cuda_and_logs_it(monkeypatch, capsys):
    calls, out = _run_main(monkeypatch, capsys, [])                     # default --device auto
    assert calls == ["cpu"]
    assert "[device] Fitting pipeline running on: cpu (CUDA available: False)" in out


def test_cli_explicit_cpu_stays_on_cpu_even_with_cuda(monkeypatch, capsys, fake_cuda):
    calls, out = _run_main(monkeypatch, capsys, ["--device", "cpu"])
    assert calls == ["cpu"]
    assert "[device] Fitting pipeline running on: cpu (CUDA available: True)" in out


def test_cli_auto_with_cuda_resolves_to_cuda_and_logs_it(monkeypatch, capsys, fake_cuda):
    calls, out = _run_main(monkeypatch, capsys, ["--device", "auto"])
    assert calls == ["cuda"]
    assert "[device] Fitting pipeline running on: cuda (CUDA available: True)" in out


@no_cuda_box
def test_cli_cuda_without_cuda_exits_with_a_clear_message(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, capsys, ["--device", "cuda"])
    assert "cuda" in str(e.value) and "is_available" in str(e.value)


def test_cli_rejects_an_unknown_device(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, capsys, ["--device", "tpu"])
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# ----------------------------------------- 6. real CUDA (skipped where there is no GPU)

@needs_cuda
def test_real_cuda_adapter_scores_match_the_cpu_fit_within_float32_rounding(tmp_path):
    """Same rows, same seed, same warm-start probe, a short run: only the device differs."""
    sx_epochs = sx.EPOCHS
    sx.EPOCHS = 3
    try:
        X, y = _xor_data(400, D=32, seed=11)
        probe = lambda Xp, yp: _fixed_probe(Xp, yp, 2)  # noqa: E731
        cfg = eh.AdapterConfig(rank=8)
        gpu = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=cfg, probe_fit=probe, device="cuda")
        cpu = eh.DeepResidualAdapterHead.fit(X, y, 2, pair=False, seed=0, config=cfg, probe_fit=probe, device="cpu")
    finally:
        sx.EPOCHS = sx_epochs
    _assert_numpy_head(gpu)
    ref = cpu.scores(X)
    assert np.max(np.abs(gpu.scores(X) - ref)) < 1e-4 * (1.0 + np.max(np.abs(ref)))
    gpu.save(tmp_path / "g.npz")
    got = _score_in_fresh_process_without_torch(tmp_path / "g.npz", "eh.DeepResidualAdapterHead", X[:16], tmp_path)
    assert np.array_equal(got, gpu.scores(X[:16]))


@needs_cuda
def test_real_cuda_linear_probe_matches_sklearn_within_solver_tolerance():
    pytest.importorskip("sklearn")
    X, y = _blob_data(400, 40, 4, seed=12)
    gpu = sx.LinearProbe.fit(X, y, 4, pair=False, kind="full", pca_k=None, seed=1, device="cuda")
    cpu = sx.LinearProbe.fit(X, y, 4, pair=False, kind="full", pca_k=None, seed=1, n_jobs=1, device="cpu")
    assert gpu.fit_info["solver"] == "torch-lbfgs" and gpu.cfg["C"] == cpu.cfg["C"]
    assert np.max(np.abs(gpu.scores(X) - cpu.scores(X))) < 5e-3


@needs_cuda
def test_real_cuda_rnn_trains_on_the_gpu_and_returns_a_cpu_model(monkeypatch):
    monkeypatch.setattr(sx, "EPOCHS", 2)
    rng = np.random.default_rng(13)
    q, cands, y = rng.normal(size=(80, 256)).astype(np.float32), rng.normal(size=(4, 256)).astype(np.float32), rng.integers(0, 4, 80)
    model, info = sx.train_rnn("baseline", (q,), y, cands, [0, 1], device="cuda")
    assert info["device"] == "cuda" and all(p.device.type == "cpu" for p in model.parameters())
    assert np.all(np.isfinite(sx.rnn_scores(model, (q,), cands)))
