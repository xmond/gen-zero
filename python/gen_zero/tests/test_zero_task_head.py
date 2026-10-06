"""Zero task head: bilinear scorer over the label-free manifold.

Mathematical calibration tests use synthetic latent-variable data (clearly
labelled as such), never natural-language evidence. The one real-weight
integration test reuses the checked-in Zero backbone/manifold and is skipped
with an explicit reason when either is absent, matching test_zero_runtime.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gen_zero.causal.zero_runtime import (
    MANIFOLD_DIM,
    ZeroStandaloneRuntime,
    build_int8_artifact,
    find_local_snapshot,
)
from gen_zero.causal.zero_task_head import ZeroTaskHead

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "benchmarks" / "artifacts" / "zero" / "zero_int8_v2.safetensors"
MANIFOLD = REPO / "benchmarks" / "artifacts" / "zero" / "zero_manifold_v1.npz"
CALIBRATION = REPO / "benchmarks" / "data" / "calibration_clean_16.jsonl"


def numeric_problem(seed, n, dim, k):
    """n records, each with k candidate directions; class 0 is always the true positive."""
    rng = np.random.default_rng(seed)
    prompts = rng.normal(size=(n, dim)).astype(np.float32)
    prompts /= np.linalg.norm(prompts, axis=1, keepdims=True)
    candidates = rng.normal(size=(n, k, dim)).astype(np.float32)
    # Make the first candidate track the prompt direction (a learnable signal);
    # the rest stay random noise directions.
    candidates[:, 0, :] = 0.7 * prompts + 0.3 * candidates[:, 0, :]
    candidates /= np.linalg.norm(candidates, axis=2, keepdims=True)
    positive = np.zeros(n, dtype=np.int64)
    return prompts, candidates, positive


def accuracy(head, prompts, candidates, positive):
    correct = 0
    for z0, zc, pos in zip(prompts, candidates, positive):
        correct += int(np.argmax(head.score(z0, zc)) == pos)
    return correct / len(positive)


def test_save_load_roundtrip_and_encoder_check(tmp_path):
    dim, k = 16, 5
    eval_prompts, eval_candidates, eval_positive = numeric_problem(1, 200, dim, k)
    # Static weight: identity plus a small fixed perturbation.
    rng = np.random.default_rng(7)
    weight = (np.eye(dim) + 0.01 * rng.normal(size=(dim, dim))).astype(np.float32)
    head = ZeroTaskHead(weight=weight,
                        provenance=dict(version=1, dim=dim, encoder_id="numeric",
                                        manifold_sha256="x", source="x", split="calibration"))
    assert 0.0 <= accuracy(head, eval_prompts, eval_candidates, eval_positive) <= 1.0

    path = tmp_path / "head.npz"
    head.save(path)
    restored = ZeroTaskHead.load(path, encoder_id="numeric")
    np.testing.assert_array_equal(restored.weight, head.weight)
    np.testing.assert_allclose(restored.score(eval_prompts[0], eval_candidates[0]),
                               head.score(eval_prompts[0], eval_candidates[0]))
    with pytest.raises(ValueError, match="encoder"):
        ZeroTaskHead.load(path, encoder_id="different")


def test_load_rejects_eval_split_provenance(tmp_path):
    dim = 4
    bad = ZeroTaskHead(weight=np.eye(dim, dtype=np.float32),
                       provenance=dict(version=1, dim=dim, encoder_id="numeric",
                                       manifold_sha256="x", source="x", split="test"))
    path = tmp_path / "bad-split-head.npz"
    with open(path, "wb") as stream:
        np.savez(stream, metadata=json.dumps(bad.provenance), weight=bad.weight)
    with pytest.raises(ValueError, match="split"):
        ZeroTaskHead.load(path, encoder_id="numeric")


def test_score_is_order_invariant_per_candidate():
    dim = 8
    rng = np.random.default_rng(3)
    head = ZeroTaskHead(weight=rng.normal(size=(dim, dim)).astype(np.float32),
                        provenance=dict(version=1, dim=dim, encoder_id="numeric",
                                        manifold_sha256="x", source="x", split="calibration"))
    z0 = rng.normal(size=dim).astype(np.float32)
    zc = rng.normal(size=(5, dim)).astype(np.float32)
    scores = head.score(z0, zc)
    perm = [3, 1, 4, 0, 2]
    scores_perm = head.score(z0, zc[perm])
    np.testing.assert_allclose(scores_perm, scores[perm])


def test_score_rejects_non_finite_and_shape_mismatch():
    dim = 8
    head = ZeroTaskHead(weight=np.eye(dim, dtype=np.float32),
                        provenance=dict(version=1, dim=dim, encoder_id="numeric",
                                        manifold_sha256="x", source="x", split="calibration"))
    with pytest.raises(ValueError):
        head.score(np.zeros(dim - 1, dtype=np.float32), np.zeros((2, dim), dtype=np.float32))
    bad = np.zeros((2, dim), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        head.score(np.zeros(dim, dtype=np.float32), bad)


# --------------------------------------------------------------------------
# Real-weight integration: decide() actually takes the head branch
# --------------------------------------------------------------------------

def _snapshot():
    try:
        return find_local_snapshot()
    except FileNotFoundError as error:
        pytest.skip(f"real Zero backbone weights are not available locally: {error}")


def test_decide_uses_task_head_when_loaded(tmp_path):
    _snapshot()
    if not ARTIFACT.exists() or not MANIFOLD.exists() or not CALIBRATION.exists():
        pytest.skip("Zero int8 artifact, manifold or calibration split not available locally")
    baseline = ZeroStandaloneRuntime(int8_artifact=ARTIFACT, manifold_path=MANIFOLD)
    with open(CALIBRATION, encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    without_head = baseline.decide(record["context"], record["candidates"])
    assert without_head.task_head_used is False

    # A deterministic, clearly-synthetic head at the runtime's own encoder_id and
    # manifold dimension -- not claimed to be semantically trained, only used to
    # prove decide() actually reads self.task_head instead of ignoring it.
    import hashlib
    manifold_sha256 = hashlib.sha256(MANIFOLD.read_bytes()).hexdigest()
    rng = np.random.default_rng(7)
    head = ZeroTaskHead(
        weight=rng.normal(size=(MANIFOLD_DIM, MANIFOLD_DIM)).astype(np.float32),
        provenance=dict(version=1, dim=MANIFOLD_DIM, encoder_id=baseline.encoder_id,
                        manifold_sha256=manifold_sha256, source="integration-test-only",
                        split="calibration"))
    head_path = tmp_path / "head.npz"
    head.save(head_path)

    with_head = ZeroStandaloneRuntime(int8_artifact=ARTIFACT, manifold_path=MANIFOLD,
                                      task_head_path=head_path)
    decision = with_head.decide(record["context"], record["candidates"])
    assert decision.task_head_used is True
    assert decision.scores.shape == without_head.scores.shape
    assert not np.allclose(decision.scores, without_head.scores)
    expected = head.score(decision.prompt_state, decision.candidate_states)
    np.testing.assert_allclose(decision.scores, expected)

    with pytest.raises(ValueError, match="encoder"):
        ZeroTaskHead.load(head_path, encoder_id="wrong-encoder")

