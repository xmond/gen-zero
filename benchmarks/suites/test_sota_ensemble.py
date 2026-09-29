"""Unit tests for sota_ensemble_experts / benchmark_sota_ensemble (synthetic data only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "python"))

import benchmark_sota_ensemble as bse  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402
from gen_zero.causal.ensemble_causal_engine import score_ensemble  # noqa: E402


def _pair_data(n=400, d=6, seed=0):
    """Label = whether field A and field B are the same point (a pure cross-field signal)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, d))
    y = rng.integers(0, 2, size=n)
    b = np.where(y[:, None] == 1, a + 0.1 * rng.normal(size=(n, d)), rng.normal(size=(n, d)))
    full = rng.normal(size=(n, d))                     # whole-context vector carries no signal here
    return np.concatenate([full, a, b], axis=1).astype(np.float32), y


def test_linear_probe_numpy_matches_its_own_fit_and_learns_cross_features():
    X, y = _pair_data()
    tr, te = np.arange(300), np.arange(300, 400)
    full = sx.LinearProbe.fit(X[tr], y[tr], 2, pair=True, kind="full", pca_k=None, seed=0)
    pair = sx.LinearProbe.fit(X[tr], y[tr], 2, pair=True, kind="pair", pca_k=None, seed=0)
    acc_full = np.mean(full.scores(X[te]).argmax(1) == y[te])
    acc_pair = np.mean(pair.scores(X[te]).argmax(1) == y[te])
    assert acc_pair > 0.9 > acc_full + 0.2, (acc_full, acc_pair)


def test_linear_probe_save_load_roundtrip(tmp_path):
    X, y = _pair_data()
    m = sx.LinearProbe.fit(X, y, 2, pair=True, kind="hybrid", pca_k=16, seed=0)
    m.save(tmp_path / "m.npz")
    m2 = sx.LinearProbe.load(tmp_path / "m.npz")
    np.testing.assert_allclose(m.scores(X[:20]), m2.scores(X[:20]), rtol=1e-5, atol=1e-5)
    C = np.zeros((2, 1))
    np.testing.assert_allclose(m2.score(X[0], C), m.scores(X[:1])[0], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        m2.score(X[0], np.zeros((3, 1)))


def test_absent_class_is_never_predicted():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)                      # classes 0 and 1 only, K = 3
    m = sx.LinearProbe.fit(X, y, 3, pair=False, kind="full", pca_k=None, seed=0)
    assert (m.scores(X).argmax(1) != 2).all()


def test_calibrated_pool_through_score_ensemble_equals_direct_formula():
    rng = np.random.default_rng(2)
    S = {"a": rng.normal(size=(50, 4)), "b": rng.normal(size=(50, 4))}
    T, w = {"a": 0.7, "b": 2.0}, {"a": 0.25, "b": 0.75}
    rep = {n: sx.ReplayExpert(n, S[n], 3) for n in S}
    C = np.zeros((4, 1), dtype=np.float32)
    for i in range(50):
        for r in rep.values():
            r.cursor = i
        ad = {n: sx.CalibratedExpert(rep[n], T[n], w[n]) for n in S}
        res = score_ensemble(ad, {n: {"query": np.zeros(3), "candidates": C} for n in S}, "logits_sum")
        direct = sum(w[n] * sx.log_softmax(S[n][i] / T[n]) for n in S)
        np.testing.assert_allclose(res["fused_logits"], direct, rtol=1e-5, atol=1e-5)


def test_greedy_weights_prefer_the_informative_expert():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=300)
    good = np.eye(3)[y] * 3 + rng.normal(size=(300, 3))
    noise = rng.normal(size=(300, 3))
    w = sx.greedy_pool_weights({"good": sx.log_softmax(good), "noise": sx.log_softmax(noise)}, y)
    assert w.get("good", 0) > 0.8


def test_strategies_run_through_engines_on_replayed_scores():
    rng = np.random.default_rng(4)
    n, K = 200, 3
    y = rng.integers(0, K, size=n)
    S = {"good": np.eye(K)[y] * 2 + rng.normal(size=(n, K)), "noise": rng.normal(size=(n, K))}
    Xq = {k: rng.normal(size=(n, 8)).astype(np.float32) for k in S}
    rows, ev = np.arange(150), np.arange(150, 200)
    T = bse.fit_temperatures(S, y, rows)
    for st in ["single:good", "pool_all", "pool_greedy", "veto_greedy", "moe_dense", "moe_consensus"]:
        fit = bse.fit_strategy(st, S, Xq, y, rows, T)
        pred = bse.replay_eval(st, fit, S, Xq, ev, K)
        assert pred.shape == (50,) and set(pred) <= set(range(K))
        if st in ("single:good", "pool_greedy", "moe_dense"):
            assert np.mean(pred == y[ev]) > 0.7, st


# ------------------------------------------------------- fractal arbitration

def _orth(n, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q


def test_fractal_core_is_orthogonal_and_permutation_equivariant():
    rng = np.random.default_rng(10)
    q, C = rng.normal(size=40), rng.normal(size=(4, 40))
    s = sx.FractalArbitrationExpert.core_scores(q, C)
    Q = _orth(40, rng)
    np.testing.assert_allclose(sx.FractalArbitrationExpert.core_scores(Q @ q, C @ Q.T), s, atol=1e-9)
    perm = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(sx.FractalArbitrationExpert.core_scores(q, C[perm]), s[perm], atol=1e-9)
    assert np.all(np.isfinite(s)) and abs(s.sum()) < 1e-9          # centered: a pure relative residual


def test_fractal_projection_is_exact_against_the_unprojected_engine():
    """Running CCRE in span(q0, C) must equal running it in the full space."""
    from gen_zero.causal.continuous_causal_reasoning_expert import continuous_causal_reasoning_expert
    rng = np.random.default_rng(11)
    q, C = rng.normal(size=12), rng.normal(size=(3, 12))
    full = continuous_causal_reasoning_expert(q, C, domain_prototype=None, seed=0).scores
    np.testing.assert_allclose(sx.FractalArbitrationExpert.core_scores(q, C), full - full.mean(), atol=1e-8)


def test_fractal_score_matches_the_nearest_candidate():
    """Confirms CCRE fix: the candidate the query sits on gets the HIGHEST score (not lowest)."""
    from gen_zero.causal.continuous_causal_reasoning_expert import continuous_causal_reasoning_expert
    rng = np.random.default_rng(12)
    C = rng.normal(size=(4, 30))
    for k in range(4):
        q = C[k] + 0.05 * rng.normal(size=30)
        assert int(np.argmax(sx.FractalArbitrationExpert.core_scores(q, C))) == k
    res = continuous_causal_reasoning_expert(C[0] + 0.3 * rng.normal(size=30), C, domain_prototype=None)
    for tr in res.traces:                         # the race actually ran and cut the pool to one branch
        assert tr.race_steps >= 1 and tr.race_alive_history[-1] == 1 and np.isfinite(tr.macro_survivor_cost)


def test_fractal_expert_fit_score_save_load(tmp_path):
    X, y = _pair_data(n=60, d=8)
    rng = np.random.default_rng(13)
    cands = rng.normal(size=(2, 8)).astype(np.float32)
    m = sx.FractalArbitrationExpert.fit(X, pair=True)
    S = m.scores(X[:10], cands)
    assert S.shape == (10, 2) and np.all(np.isfinite(S)) and np.allclose(S.sum(1), 0, atol=1e-5)
    assert m.scores(X[:0], cands).shape == (0, 2)
    m.save(tmp_path / "f.npz")
    np.testing.assert_allclose(sx.FractalArbitrationExpert.load(tmp_path / "f.npz").scores(X[:10], cands), S)
    with pytest.raises(ValueError):
        m.score(X[0, :8], cands)                   # pair expert fed a non-pair vector


def test_fractal_gate_fires_on_ties_only_and_the_residual_can_flip_the_choice():
    fit = {"beta": 1.0, "tau_m": 0.1, "tau_h": 1.01}
    resid = np.array([0.5, -0.5])
    tie = np.log(np.array([0.48, 0.52]))
    wide = np.log(np.array([0.1, 0.9]))
    assert sx.fractal_gate(np.stack([tie, wide]), 0.1, 1.01).tolist() == [True, False]
    assert int(np.argmax(tie)) == 1 and int(np.argmax(sx.fractal_arbitrate(tie, resid, fit))) == 0
    np.testing.assert_array_equal(sx.fractal_arbitrate(wide, resid, fit), wide)
    np.testing.assert_array_equal(sx.fractal_arbitrate(tie, resid, dict(fit, beta=0.0)), tie)


def test_fit_fractal_gate_uses_an_informative_residual_and_ignores_noise():
    rng = np.random.default_rng(14)
    n = 400
    y = rng.integers(0, 2, size=n)
    margin = np.where(rng.random(n) < 0.3, 0.02, 2.0)              # 30% near-ties
    sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    z = np.where(margin > 1, np.where(y == 1, 1.0, -1.0), sign) * margin   # ties are coin flips
    base = sx.log_softmax(np.stack([-z / 2, z / 2], 1))
    informative = np.stack([np.where(y == 0, 1.0, -1.0), np.where(y == 1, 1.0, -1.0)], 1)
    good = sx.fit_fractal_gate(base, informative, y)
    assert good["beta"] > 0 and good["fit_acc"] > np.mean(base.argmax(1) == y) + 0.05
    flipped = sx.fit_fractal_gate(base, -informative, y)
    assert flipped["beta"] < 0 and abs(flipped["fit_acc"] - good["fit_acc"]) < 1e-12
    noise = sx.fit_fractal_gate(base, rng.normal(size=(n, 2)), y)
    base_acc = np.mean(base.argmax(1) == y)
    assert noise["fit_acc"] >= base_acc and (noise["beta"] == 0.0 or noise["fit_acc"] - base_acc < 0.06)


def _fractal_scores(n=240, K=3, seed=15):
    """A good expert that is unsure on a third of the rows, and a label-free row that knows those rows."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    hard = rng.random(n) < 0.35
    good = np.where(hard[:, None], 0.05 * rng.normal(size=(n, K)), np.eye(K)[y] * 3 + 0.3 * rng.normal(size=(n, K)))
    # Informative on the hard rows only, a random wrong-ish vote elsewhere: the gate has to matter.
    vote = np.where(hard, y, rng.integers(0, K, size=n))
    frac = np.eye(K)[vote] - 1.0 / K + 0.2 * rng.normal(size=(n, K))
    S = {"src:good": good, "src:noise": rng.normal(size=(n, K)), "src" + sx.FRACTAL_SUFFIX: frac}
    Xq = {k: rng.normal(size=(n, 8)).astype(np.float32) for k in S}
    return S, Xq, y


def test_fractal_strategies_run_through_engines_on_replayed_scores():
    S, Xq, y = _fractal_scores()
    K = 3
    rows, ev = np.arange(180), np.arange(180, 240)
    T = bse.fit_temperatures(S, y, rows)
    base_fit = bse.fit_strategy("pool_greedy", {k: v for k, v in S.items() if not k.endswith(sx.FRACTAL_SUFFIX)},
                                Xq, y, rows, T)
    base_acc = np.mean(bse.replay_eval("pool_greedy", base_fit, {k: S[k] for k in base_fit["weights"]},
                                       Xq, ev, K) == y[ev])
    for st in ("fractal_tiebreak", "moe_fractal_consensus"):
        fit = bse.fit_strategy(st, S, Xq, y, rows, T)
        fr = fit["fractal"]
        assert fr["names"] == ["src" + sx.FRACTAL_SUFFIX] and fr["beta"] > 0 and 0 < fr["fit_gated_frac"] < 1
        assert all(not n.endswith(sx.FRACTAL_SUFFIX) for n in (fit["weights"] or fit["router"].names))
        pred = bse.replay_eval(st, fit, S, Xq, ev, K)
        assert pred.shape == (60,) and set(pred) <= set(range(K))
        if st == "fractal_tiebreak":
            assert np.mean(pred == y[ev]) > base_acc + 0.1, (np.mean(pred == y[ev]), base_acc)


def test_fractal_strategy_without_a_fractal_expert_raises():
    S, Xq, y = _fractal_scores()
    S = {k: v for k, v in S.items() if not k.endswith(sx.FRACTAL_SUFFIX)}
    T = bse.fit_temperatures(S, y, np.arange(100))
    for st in bse.FRACTAL_STRATEGIES:
        with pytest.raises(ValueError, match="fractal"):
            bse.fit_strategy(st, S, Xq, y, np.arange(100), T)


def test_fractal_fit_survives_the_combine_json_and_eval_reconstruction():
    """stage_combine writes rec; stage_eval rebuilds fit from exactly those keys. Same predictions both ways."""
    import json
    S, Xq, y = _fractal_scores()
    rows, ev = np.arange(180), np.arange(180, 240)
    T = bse.fit_temperatures(S, y, rows)
    fit = bse.fit_strategy("fractal_tiebreak", S, Xq, y, rows, T)
    rec = json.loads(json.dumps({"temperatures": fit["T"], "weights": fit["weights"], "fractal": fit["fractal"]}))
    rebuilt = {"T": rec["temperatures"], "weights": rec["weights"], "fractal": rec["fractal"]}
    np.testing.assert_array_equal(bse.replay_eval("fractal_tiebreak", fit, S, Xq, ev, 3),
                                  bse.replay_eval("fractal_tiebreak", rebuilt, S, Xq, ev, 3))
    comb = dict(rec, chosen="fractal_tiebreak")
    assert bse.n_experts_run("fractal_tiebreak", comb, 99) == len(rec["weights"]) + 1


def test_strategies_and_expert_specs_register_the_new_experts(monkeypatch):
    assert bse.STRATEGIES_LEARNED[-2:] == ("fractal_tiebreak", "moe_fractal_consensus")
    monkeypatch.setattr(bse, "RNN_SOURCES", ["enc"])
    fs = {"enc": {"train_full": np.zeros((5, 16))}}
    specs = {n: sp for n, _, sp in bse.expert_specs("boolq", fs)}
    assert specs["enc:fractal"] == {"type": "fractal"}
    assert specs["enc:baseline_mcts"] == {"type": "mcts", "config": "baseline"}


def test_causal_mcts_expert_scores_a_real_exported_head(tmp_path):
    torch = pytest.importorskip("torch")
    from rnn_set_adapter_torch import ParallelRNNSetAdapter
    torch.manual_seed(0)
    m = ParallelRNNSetAdapter(24, d=16, rank=4, think_steps=3, n_heads=2, n_layers=1, dropout=0.0).eval()
    m.export_npz(tmp_path / "h.npz", {"test": True})
    rng = np.random.default_rng(16)
    x = rng.normal(size=72).astype(np.float32)           # pair source: [full; a; b]
    C = rng.normal(size=(3, 24)).astype(np.float32)
    core = sx.CausalMCTSExpertCore(tmp_path / "h.npz", pair=True)
    s = core.score(x, C)
    assert s.shape == (3,) and np.all(np.isfinite(s)) and abs(np.exp(s).sum() - 1) < 1e-4
    d = core.engine.classify(x[:24], C)
    np.testing.assert_allclose(s, np.log(d.probs), atol=1e-6)
    assert d.mode == "fast" and 0.0 <= d.cf_sensitivity <= 1.0


def test_legacy_strategies_never_see_the_fractal_row():
    S, Xq, y = _fractal_scores()
    rows = np.arange(180)
    T = bse.fit_temperatures(S, y, rows)
    for st in ("pool_all", "pool_greedy", "veto_greedy"):
        assert all(not n.endswith(sx.FRACTAL_SUFFIX) for n in bse.fit_strategy(st, S, Xq, y, rows, T)["weights"])
    for st in ("moe_dense", "moe_consensus"):
        assert all(not n.endswith(sx.FRACTAL_SUFFIX) for n in bse.fit_strategy(st, S, Xq, y, rows, T)["router"].names)
    # beta = 0 reproduces pool_greedy exactly, so the combine tie rule can pick the simpler entry
    fit = bse.fit_strategy("fractal_tiebreak", S, Xq, y, rows, T)
    ev = np.arange(180, 240)
    np.testing.assert_array_equal(bse.replay_eval("fractal_tiebreak", dict(fit, fractal=dict(fit["fractal"], beta=0.0)),
                                                  S, Xq, ev, 3),
                                  bse.replay_eval("pool_greedy", bse.fit_strategy("pool_greedy", S, Xq, y, rows, T),
                                                  S, Xq, ev, 3))


# ------------------------------------------------- 1-SE selection, greedy stop

def test_greedy_early_stop_never_adds_an_expert_that_does_not_lower_nll():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, size=400)
    good = sx.log_softmax(np.eye(3)[y] * 3 + rng.normal(size=(400, 3)))
    L = {"good": good, "noise_a": sx.log_softmax(rng.normal(size=(400, 3))),
         "noise_b": sx.log_softmax(rng.normal(size=(400, 3)))}
    w = sx.greedy_pool_weights(L, y)
    assert set(w) == {"good"} and w["good"] == 1.0, w
    # the stop test is the pool NLL itself: every kept round lowered it by more than tol
    w2 = sx.greedy_pool_weights({"a": good, "b": good + 0.0}, y)
    assert abs(sum(w2.values()) - 1.0) < 1e-12 and len(w2) == 1, w2


def test_greedy_early_stop_still_mixes_two_complementary_experts():
    rng = np.random.default_rng(6)
    n = 600
    y = rng.integers(0, 2, size=n)
    sig = np.where(y == 1, 1.0, -1.0)
    a = np.stack([np.zeros(n), sig + rng.normal(size=n) * 1.5], 1)      # two independent noisy views
    b = np.stack([np.zeros(n), sig + rng.normal(size=n) * 1.5], 1)
    w = sx.greedy_pool_weights({"a": sx.log_softmax(a), "b": sx.log_softmax(b)}, y)
    assert set(w) == {"a", "b"} and abs(sum(w.values()) - 1.0) < 1e-12, w


def _folds(*accs):
    return np.array(accs, dtype=float) / 100


def test_one_se_falls_back_to_the_single_expert_when_the_fusion_gain_is_inside_one_se():
    strategies = ["single:big", "single:small", "pool_greedy", "fractal_tiebreak"]
    fa = {"single:big": _folds(84, 86, 84, 86, 85), "single:small": _folds(84, 85, 84, 86, 84),
          "pool_greedy": _folds(82, 88, 83, 87, 86.5), "fractal_tiebreak": _folds(80, 82, 81, 80, 82)}
    cx = {"single:big": (0, 15363), "single:small": (0, 387), "pool_greedy": (1, 0), "fractal_tiebreak": (4, 0)}
    chosen, info = bse.select_one_se(strategies, fa, cx)
    se = np.std(fa["pool_greedy"], ddof=1) / np.sqrt(5)
    assert info["top"] == "pool_greedy" and abs(info["se"] - round(100 * se, 3)) < 1e-9
    assert set(info["band"]) == {"single:big", "single:small", "pool_greedy"}
    assert chosen == "single:small"                     # least complex inside the band


def test_one_se_keeps_the_fusion_when_it_wins_by_more_than_one_se():
    strategies = ["single:a", "pool_greedy", "fractal_tiebreak"]
    fa = {"single:a": _folds(70, 71, 70, 72, 71), "pool_greedy": _folds(80, 81, 80, 81, 80),
          "fractal_tiebreak": _folds(80, 81, 80, 81, 81)}
    cx = {"single:a": (0, 99), "pool_greedy": (1, 0), "fractal_tiebreak": (4, 0)}
    chosen, info = bse.select_one_se(strategies, fa, cx)
    assert info["top"] == "fractal_tiebreak" and chosen == "pool_greedy"   # simpler fusion inside the band
    assert "single:a" not in info["band"]


def test_one_se_with_zero_fold_variance_is_the_old_argmax_with_earlier_entry_ties():
    strategies = ["single:a", "single:b", "pool_greedy"]
    fa = {"single:a": _folds(80, 80, 80, 80, 80), "single:b": _folds(80, 80, 80, 80, 80),
          "pool_greedy": _folds(81, 81, 81, 81, 81)}
    cx = {"single:a": (0, 10), "single:b": (0, 10), "pool_greedy": (1, 0)}
    assert bse.select_one_se(strategies, fa, cx)[0] == "pool_greedy"
    fa["pool_greedy"] = _folds(80, 80, 80, 80, 80)
    assert bse.select_one_se(strategies, fa, cx)[0] == "single:a"


def test_strategy_complexity_orders_singles_before_fusions_and_counts_supervised_params():
    specs = {"s:lin_full": {"type": "linear"}, "s:lin_full_pca": {"type": "linear"},
             "s:fractal": {"type": "fractal"}}
    meta = {"s:lin_full": {"folds": [{"feature_dim": 896}] * 5},
            "s:lin_full_pca": {"folds": [{"feature_dim": 128}] * 5}}
    cx = {st: bse.strategy_complexity(st, specs, meta, 3, {}) for st in
          ["single:s:lin_full", "single:s:lin_full_pca", "single:s:fractal"] + list(bse.STRATEGIES_LEARNED)}
    assert cx["single:s:lin_full"] == (0, 3 * 897) and cx["single:s:lin_full_pca"] == (0, 3 * 129)
    assert cx["single:s:fractal"] == (0, 0)
    assert max(v for k, v in cx.items() if k.startswith("single:")) < min(
        v for k, v in cx.items() if not k.startswith("single:"))
    assert set(bse.LEARNED_TIER) == set(bse.STRATEGIES_LEARNED)
