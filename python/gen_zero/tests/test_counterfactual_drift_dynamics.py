"""Tests for the calibrated counterfactual-drift dynamics (numeric, label-free inference).

These are mathematical checks on synthetic vectors.  They are NOT evidence about
natural-language accuracy; that lives in benchmarks/results_track_c/cpu_dynamics_clean_eval.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gen_zero.causal.counterfactual_drift_dynamics import (
    ARRAY_KEYS,
    CalibrationSplitError,
    CandidateAttractorPotential,
    CounterfactualDriftDynamics,
    fit_counterfactual_drift_dynamics,
    simplex_codebook,
)
from gen_zero.causal.continuous_causal_reasoning_expert import (
    STATUS_PRUNED,
    continuous_causal_reasoning_expert,
)

ENC = "synthetic-identity"


def _problem(seed: int, n: int = 40, p: int = 64, k: int = 3, cf_signal: float = 0.0):
    """x carries a class signal; c carries an independent class signal of strength cf_signal."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    dirs_x = rng.normal(size=(k, p))
    dirs_c = rng.normal(size=(k, p))
    x = rng.normal(size=(n, p)) + 0.8 * dirs_x[y]
    c = rng.normal(size=(n, p)) + cf_signal * dirs_c[y]
    ids = [f"s{seed}-{i}" for i in range(n)]
    return x, c, y, ids, k


def _fit(seed=0, **kw):
    x, c, y, ids, k = _problem(seed, **{k_: v for k_, v in kw.items() if k_ in ("n", "p", "k", "cf_signal")})
    dyn, info = fit_counterfactual_drift_dynamics(
        x, c, y, sample_ids=ids, source="synthetic-test", split="calibration", encoder_id=ENC,
        n_classes=k, **{k_: v for k_, v in kw.items() if k_ not in ("n", "p", "k", "cf_signal")})
    return dyn, info, (x, c, y, ids, k)


def test_fit_rejects_evaluation_splits():
    x, c, y, ids, k = _problem(1)
    for split in ("test", "validation", "validation_matched", ""):
        with pytest.raises(CalibrationSplitError):
            fit_counterfactual_drift_dynamics(x, c, y, sample_ids=ids, source="s", split=split,
                                              encoder_id=ENC, n_classes=k)


def test_certificate_rejects_non_contractive_A():
    dyn, _, _ = _fit(2)
    bad = CounterfactualDriftDynamics.__new__(CounterfactualDriftDynamics)
    for name in ARRAY_KEYS:
        setattr(bad, name, getattr(dyn, name).copy())
    bad.provenance = dict(dyn.provenance)
    bad.A = np.eye(dyn.dim) * 1.2
    with pytest.raises(ValueError, match="not contractive"):
        bad.certify()


def test_recurrence_converges_to_closed_form_fixed_point():
    dyn, _, (x, c, y, ids, k) = _fit(3)
    res = dyn.infer(x[0], c[0], langevin=False, expert=False, relax_steps=64)
    assert res.relaxation_residual < 1e-9
    assert np.allclose(res.relaxed_state, res.fixed_point, atol=1e-9)


def test_npz_roundtrip_and_schema_checks(tmp_path: Path):
    dyn, _, (x, c, y, ids, k) = _fit(4)
    path = tmp_path / "drift.npz"
    dyn.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"metadata", *ARRAY_KEYS}
        meta = json.loads(str(data["metadata"]))
        assert meta["provenance"]["split"] == "calibration"
        assert all(data[k_].dtype == np.float32 for k_ in ARRAY_KEYS)
    back = CounterfactualDriftDynamics.load(path, encoder_id=ENC)
    a = dyn.infer(x[1], c[1], langevin=False, expert=False)
    b = back.infer(x[1], c[1], langevin=False, expert=False)
    assert np.allclose(a.fixed_point, b.fixed_point, atol=1e-4)
    with pytest.raises(ValueError, match="encoder"):
        CounterfactualDriftDynamics.load(path, encoder_id="other")
    # A corrupted / extended schema must raise, never fall back.
    arrays = {k_: getattr(dyn, k_).astype(np.float32) for k_ in ARRAY_KEYS}
    with open(tmp_path / "extra.npz", "wb") as fh:
        np.savez(fh, metadata=json.dumps({"version": 1}), extra=np.zeros(1, np.float32), **arrays)
    with pytest.raises(ValueError, match="schema"):
        CounterfactualDriftDynamics.load(tmp_path / "extra.npz", encoder_id=ENC)
    # An artifact whose provenance claims a test split is refused at load.
    meta = {"version": 1, "provenance": {**dyn.provenance, "split": "test"},
            "shapes": {k_: list(v.shape) for k_, v in arrays.items()}}
    with open(tmp_path / "leak.npz", "wb") as fh:
        np.savez(fh, metadata=json.dumps(meta), **arrays)
    with pytest.raises(CalibrationSplitError):
        CounterfactualDriftDynamics.load(tmp_path / "leak.npz", encoder_id=ENC)


def test_counterfactual_drift_is_not_decorative():
    """When only c carries the class signal, W_c must be what makes predictions right."""
    rng = np.random.default_rng(5)
    n, p, k = 60, 64, 3
    y = rng.integers(0, k, n)
    x = rng.normal(size=(n, p))                       # x: pure noise
    dirs_c = rng.normal(size=(k, p))
    c = rng.normal(size=(n, p)) + 1.5 * dirs_c[y]      # c: the only signal
    ids = [f"cf{i}" for i in range(n)]
    with_cf, _ = fit_counterfactual_drift_dynamics(x[:40], c[:40], y[:40], sample_ids=ids[:40],
                                                   source="s", split="calibration", encoder_id=ENC,
                                                   n_classes=k, use_counterfactual=True)
    without, _ = fit_counterfactual_drift_dynamics(x[:40], c[:40], y[:40], sample_ids=ids[:40],
                                                   source="s", split="calibration", encoder_id=ENC,
                                                   n_classes=k, use_counterfactual=False)
    assert np.all(without.W_c == 0.0)
    acc_with = np.mean([with_cf.infer(x[i], c[i], langevin=False, expert=False).prediction == y[i]
                        for i in range(40, n)])
    acc_without = np.mean([without.infer(x[i], c[i], use_counterfactual=False, langevin=False,
                                         expert=False).prediction == y[i] for i in range(40, n)])
    assert acc_with > 0.8, acc_with
    assert acc_without < 0.6, acc_without


def test_langevin_layer_runs_and_reports_convergence():
    dyn, _, (x, c, y, ids, k) = _fit(6)
    res = dyn.infer(x[0], c[0], expert=False)
    assert res.langevin is not None
    assert res.langevin.steps > 0
    assert res.langevin.converged, (res.langevin.final_residual, res.langevin.final_violation)
    # Langevin stays in the basin the readout chose (a sanity check, not an accuracy claim).
    pot = CandidateAttractorPotential(dyn.codebook)
    assert pot.nearest(res.langevin.q) == pot.nearest(res.fixed_point)


def test_attractor_potential_gradient_matches_finite_difference():
    pot = CandidateAttractorPotential(simplex_codebook(3, 8), beta=4.0)
    rng = np.random.default_rng(7)
    q = rng.normal(size=8)
    out, scratch = np.zeros(8), np.zeros(8)
    g = pot.grad_into(q, out, scratch).copy()
    eps = 1e-6
    fd = np.array([(pot.value(q + eps * e) - pot.value(q - eps * e)) / (2 * eps) for e in np.eye(8)])
    assert np.allclose(g, fd, atol=1e-5)
    with pytest.raises(ValueError):
        pot.grad_into(q, q, scratch)


def test_expert_keeps_readout_decision_when_state_is_at_a_vertex():
    """Regression for the absolute-state projection defect fixed by G(s-q0)=0."""
    cb = simplex_codebook(3, 16)
    rng = np.random.default_rng(8)
    hits = 0
    for i in range(30):
        truth = i % 3
        q0 = cb[truth] + 0.05 * rng.normal(size=16)
        res = continuous_causal_reasoning_expert(q0, cb, seed=0)
        hits += int(np.argmax(res.scores) == truth)
    assert hits >= 24, f"expert agreed with nearest vertex on only {hits}/30"


def test_expert_does_not_invert_decision_at_a_vertex():
    """The nearby candidate must remain reachable under displacement constraints."""
    cb = simplex_codebook(2, 16)
    q0 = cb[0] + 0.05 * np.random.default_rng(9).normal(size=16)
    res = continuous_causal_reasoning_expert(q0, cb, seed=0)
    assert int(np.argmax(res.scores)) == 0
    assert res.traces[0].status != STATUS_PRUNED
    assert res.traces[0].micro_residual_to_target < res.traces[1].micro_residual_to_target
