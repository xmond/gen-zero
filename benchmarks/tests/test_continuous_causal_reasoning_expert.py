"""End-to-end wiring tests for `continuous_causal_reasoning_expert`.

Covers two layers:

1. The engine composition itself (`gen_zero.causal.continuous_causal_reasoning_expert`),
   run directly on synthetic representation vectors: macro race, micro hard forcing, and
   the counterfactual judge each demonstrably fire, and the repulsion ablation switch
   proves the macro/micro layers are not a decorative re-derivation of plain cosine
   similarity (Finding F4 from design review).
2. The wiring into `run_remote_eval_v6.analyze()`: with `SampleRecord.h_candidates`
   populated (the only way the real A100 pipeline populates it, via
   `A100Engine.unified_forward_and_score`), the expert is genuinely selected by
   `probe_mode == 'none'` for a domain (`mcq`) that previously had only `ar_loglik`,
   produces legal calibrated probabilities, and never reads `ground_truth`/labels.

No GPU is available in this environment (`derive_is_mock()` is True here), so every test
below is the pure-numpy path: real `SampleRecord`/`analyze()` objects with fabricated but
finite representation vectors, exactly the pattern `run_remote_eval_v6.py`'s own
`_synthetic_records` / `run_self_test` already use for GPU-free CPU checks.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gen_zero.causal.continuous_causal_reasoning_expert import (
    STATUS_CONVERGED,
    STATUS_PRUNED,
    continuous_causal_reasoning_expert,
)

SPEC = importlib.util.spec_from_file_location(
    "run_remote_eval_v6",
    Path(__file__).resolve().parents[1] / "suites" / "run_remote_eval_v6.py",
)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

SampleRecord = runner.SampleRecord
AnalysisConfig = runner.AnalysisConfig
analyze = runner.analyze
domain_of = runner.domain_of
DOMAIN_EXPERTS = runner.DOMAIN_EXPERTS


# ---------------------------------------------------------------------------
# 1. Direct engine-composition tests
# ---------------------------------------------------------------------------

def test_module_source_never_references_labels():
    """Static audit: the expert's own source contains no ground-truth/label token."""
    import gen_zero.causal.continuous_causal_reasoning_expert as mod
    text = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in (".ground_truth", "['ground_truth']", '["ground_truth"]', ".y_global", "y_global["):
        assert banned not in text, f"expert source must never reference {banned!r}"


def test_engines_actually_run_and_converge():
    rng = np.random.default_rng(0)
    dim = 16
    q0 = rng.normal(size=dim); q0 /= np.linalg.norm(q0)
    cands = rng.normal(size=(4, dim))
    cands /= np.linalg.norm(cands, axis=1, keepdims=True)
    proto = rng.normal(size=dim); proto /= np.linalg.norm(proto)

    result = continuous_causal_reasoning_expert(q0, cands, domain_prototype=proto)
    assert result.scores.shape == (4,)
    assert np.all(np.isfinite(result.scores))
    assert len(result.traces) == 4
    for t in result.traces:
        assert t.status in (STATUS_CONVERGED, STATUS_PRUNED)
        # `race_steps >= 1` is direct evidence BifurcatedFractalEngine.race actually looped,
        # not a stub returning immediately.
        assert t.race_steps >= 1
        assert len(t.race_alive_history) >= 2
        assert t.race_alive_history[0] >= t.race_alive_history[-1]  # branches only get pruned
    # At least one candidate must reach the micro layer (else the whole test is vacuous).
    assert any(t.status == STATUS_CONVERGED for t in result.traces)


def test_judge_layer_only_penalizes_never_rewards():
    rng = np.random.default_rng(3)
    dim = 12
    q0 = rng.normal(size=dim); q0 /= np.linalg.norm(q0)
    cands = rng.normal(size=(3, dim))
    cands /= np.linalg.norm(cands, axis=1, keepdims=True)
    proto = rng.normal(size=dim); proto /= np.linalg.norm(proto)

    with_judge = continuous_causal_reasoning_expert(q0, cands, domain_prototype=proto)
    without_judge = continuous_causal_reasoning_expert(q0, cands, domain_prototype=None)
    for t in with_judge.traces:
        assert t.judge_penalty >= 0.0
    # Judge only subtracts: every candidate's judged score is <= its unjudged score.
    assert np.all(with_judge.scores <= without_judge.scores + 1e-9)


def test_repulsion_ablation_is_not_decorative():
    """Finding F4: if disabling repulsion doesn't move the output, the dynamics are cosmetic."""
    dim = 8
    q0 = np.zeros(dim); q0[0] = 0.01
    c0 = np.zeros(dim); c0[0] = 1.0
    c1 = np.zeros(dim); c1[0] = 0.98; c1[1] = 0.05   # near-collinear with c0 from q0
    c2 = np.zeros(dim); c2[2] = 1.0                   # distinct, orthogonal candidate
    cands = np.stack([c0, c1, c2])

    on = continuous_causal_reasoning_expert(q0, cands, enable_repulsion=True, seed=1)
    off = continuous_causal_reasoning_expert(q0, cands, enable_repulsion=False, seed=1)

    assert not np.allclose(on.scores, off.scores, atol=1e-3)
    # With repulsion on, the two near-collinear candidates must be treated worse (relative
    # to the distinct candidate 2) than with repulsion off -- the whole point of the macro
    # race is to break exactly this kind of degenerate/collinear ambiguity.
    on_gap = on.scores[2] - max(on.scores[0], on.scores[1])
    off_gap = off.scores[2] - max(off.scores[0], off.scores[1])
    assert on_gap > off_gap


def test_all_candidates_can_be_pruned_and_score_stays_finite():
    dim = 6
    q0 = np.zeros(dim)
    # Every candidate points in the same direction from q0: maximal mutual repulsion.
    c = np.ones(dim) / np.sqrt(dim)
    cands = np.stack([c * (1.0 + 1e-3 * i) for i in range(4)])
    result = continuous_causal_reasoning_expert(q0, cands, seed=2)
    assert np.all(np.isfinite(result.scores))
    assert all(t.status in (STATUS_CONVERGED, STATUS_PRUNED) for t in result.traces)


def test_single_candidate_has_no_repulsors_and_identity_micro_projection():
    """K=1: no competitor exists, so `ConditionSet.empty` (identity projector) must be used."""
    dim = 8
    rng = np.random.default_rng(5)
    q0 = rng.normal(size=dim)
    c0 = rng.normal(size=dim)
    result = continuous_causal_reasoning_expert(q0, c0[None, :])
    assert result.scores.shape == (1,)
    assert result.traces[0].status == STATUS_CONVERGED
    assert np.isfinite(result.traces[0].micro_residual_to_target)


@pytest.mark.parametrize("dim,seed", [(6, 11), (13, 29), (24, 47)])
def test_micro_residual_matches_affine_distance_and_is_translation_invariant(dim, seed):
    """Check the geometric contract independently of labels or decision accuracy."""
    rng = np.random.default_rng(seed)
    q0 = rng.normal(size=dim)
    cands = rng.normal(size=(4, dim))
    shift = rng.normal(size=dim) * 5
    result = continuous_causal_reasoning_expert(q0, cands, micro_steps=200)
    translated = continuous_causal_reasoning_expert(q0 + shift, cands + shift, micro_steps=200)
    for i, trace in enumerate(result.traces):
        # An orthonormal basis gives the distance to the affine constraint
        # plane through q0 independently of ConditionSet's pseudoinverse.
        directions = np.delete(cands, i, axis=0) - q0
        basis, _ = np.linalg.qr(directions.T)
        distance = np.linalg.norm(basis.T @ (cands[i] - q0))
        assert trace.micro_residual_to_target == pytest.approx(distance, abs=1e-10)
    np.testing.assert_allclose(result.scores, translated.scores, atol=1e-10, rtol=1e-10)


def test_candidate_at_prompt_is_a_feasible_fixed_point():
    q0 = np.array([1.0, -2.0, 3.0])
    cands = np.stack([q0, q0 + [1.0, 0.0, 0.0], q0 + [0.0, 1.0, 0.0]])
    result = continuous_causal_reasoning_expert(q0, cands)
    assert result.traces[0].micro_residual_to_target == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Wiring through run_remote_eval_v6.analyze()
# ---------------------------------------------------------------------------

def _synthetic_records_with_candidates(rng: np.random.Generator, task: str, cands, per: int = 24,
                                       d: int = 48):
    """Mirrors run_remote_eval_v6._synthetic_records but also fills h_candidates, exactly
    the field only A100Engine.unified_forward_and_score populates in the real pipeline."""
    k = len(cands)
    common = rng.normal(size=d) * 5.0
    class_dir = rng.normal(size=(k, d)) * 1.5
    labels = rng.integers(0, k, size=per)
    recs = []
    for j in range(per):
        y = int(labels[j])
        h = common + class_dir[y] + rng.normal(size=d) * 0.3
        # Each candidate's terminal vector is pulled toward its own class direction, the
        # correct one most strongly -- exactly what a real forward pass over each candidate
        # continuation should look like (candidate text steers the terminal hidden state).
        h_cands = np.stack([
            common + class_dir[c] + rng.normal(size=d) * 0.3 + (2.0 if c == y else 0.0) * class_dir[c] / np.linalg.norm(class_dir[c])
            for c in range(k)
        ])
        rec = SampleRecord(sid=f"{task}-{j:03d}", task=task, context="", candidates=list(cands),
                           ground_truth=cands[y], h=h, forward_ms=1.0)
        rec.ar_scores = rng.normal(size=k) * 0.1  # deliberately weak/uninformative AR signal
        rec.h_candidates = h_cands
        recs.append(rec)
    return recs, labels


def test_expert_is_selected_for_mcq_domain_previously_ar_loglik_only():
    assert domain_of("arc_challenge") == "mcq"
    assert DOMAIN_EXPERTS["mcq"] == ["ar_loglik", "continuous_causal_reasoning"]

    rng = np.random.default_rng(42)
    recs, _ = _synthetic_records_with_candidates(rng, "arc_challenge", ["A", "B", "C", "D"])
    out = analyze(recs, AnalysisConfig(probe_mode="none"))

    assert all("continuous_causal_reasoning" in r["expert_pred"] for r in out["results"])
    for r in out["results"]:
        p = r["probs"]
        assert p.shape == (4,)
        assert np.all(np.isfinite(p))
        assert abs(float(p.sum()) - 1.0) < 1e-6


def test_without_h_candidates_expert_does_not_appear_old_behavior_preserved():
    """Regression guard: records without h_candidates (every pre-existing self-test record)
    must behave exactly as before -- only ar_loglik active for mcq."""
    rng = np.random.default_rng(9)
    recs, _ = _synthetic_records_with_candidates(rng, "arc_challenge", ["A", "B", "C", "D"])
    for r in recs:
        r.h_candidates = None
    out = analyze(recs, AnalysisConfig(probe_mode="none"))
    assert all(set(r["expert_pred"]) == {"ar_loglik"} for r in out["results"])


def test_label_invariance_garbage_and_shuffled_ground_truth():
    rng = np.random.default_rng(17)
    recs, _ = _synthetic_records_with_candidates(rng, "arc_challenge", ["A", "B", "C", "D"])
    out_ref = analyze(recs, AnalysisConfig(probe_mode="none"))

    garbage = copy.deepcopy(recs)
    for r in garbage:
        r.ground_truth = "NOT_A_REAL_CANDIDATE_LABEL"
    out_garbage = analyze(garbage, AnalysisConfig(probe_mode="none"))

    shuffled = copy.deepcopy(recs)
    for r in shuffled:
        idx = runner.truth_index(r.candidates, r.ground_truth)
        r.ground_truth = r.candidates[(idx + 1) % len(r.candidates)]
    out_shuffled = analyze(shuffled, AnalysisConfig(probe_mode="none"))

    for a, b, c in zip(out_ref["results"], out_garbage["results"], out_shuffled["results"]):
        assert np.allclose(a["probs"], b["probs"])
        assert np.allclose(a["probs"], c["probs"])
        assert np.allclose(
            a["weights"].get("continuous_causal_reasoning", 0.0),
            b["weights"].get("continuous_causal_reasoning", 0.0),
        )


def test_reasoning_expert_beats_uninformative_ar_scores():
    """The synthetic AR signal is deliberately near-uniform noise; the fused decision must
    still beat chance because the geometry-driven expert is genuinely doing work."""
    rng = np.random.default_rng(123)
    recs, labels = _synthetic_records_with_candidates(rng, "arc_challenge", ["A", "B", "C", "D"], per=60)
    out = analyze(recs, AnalysisConfig(probe_mode="none"))
    correct = [r["pred"] == r["y"] for r in out["results"]]
    acc = float(np.mean(correct))
    assert acc > 0.4, f"fused accuracy {acc:.3f} should clear chance (0.25) by a wide margin"


def test_h_candidates_shape_mismatch_fails_closed():
    rng = np.random.default_rng(1)
    recs, _ = _synthetic_records_with_candidates(rng, "arc_challenge", ["A", "B", "C", "D"])
    recs[0].h_candidates = recs[0].h_candidates[:2]  # wrong K
    with pytest.raises(ValueError):
        analyze(recs, AnalysisConfig(probe_mode="none"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
