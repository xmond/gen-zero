"""Fail-closed regressions and real CPU optimizer verification (no model mocks)."""
import math

import pytest
import torch

from gen_zero.model.dual_head import GenZeroDualHeadModel
from gen_zero.train import invariant_distiller as module
from gen_zero.train.replay_buffer import StabilityReplayBuffer

REQUIRED = "Torch and an active optimizer are required for invariant distillation training step"


@pytest.fixture
def distiller():
    torch.manual_seed(19)
    model = GenZeroDualHeadModel(hidden_dim=16, embed_dim=8, num_layers=1, num_heads=2)
    return module.InvariantCausalDistiller(model, StabilityReplayBuffer(), device="cpu")


@pytest.fixture
def batch():
    return [dict(type="choice", leaf_tokens=[[1, 2], [3, 4]],
                 candidate_ids=["a", "b"], soft_target={"a": 0.8, "b": 0.2},
                 value_target=1.0, environment_id=env) for env in ("one", "two")]


@pytest.mark.parametrize("batch", [[], [{}]])
def test_no_optimizer_raises(batch):
    distiller = module.InvariantCausalDistiller(object(), StabilityReplayBuffer())
    with pytest.raises(RuntimeError, match=f"^{REQUIRED}$"):
        distiller.train_step(batch)


@pytest.mark.parametrize("batch", [[], [{}]])
def test_unavailable_torch_raises(distiller, monkeypatch, batch):
    # Inject dependency unavailability, not fabricated training outputs.
    monkeypatch.setattr(module, "HAS_TORCH", False)
    with pytest.raises(RuntimeError, match=f"^{REQUIRED}$"):
        distiller.train_step(batch)


def test_real_training_updates_parameters(distiller, batch):
    before = [p.detach().clone() for p in distiller.model.parameters()]
    metrics = distiller.train_step(batch)
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["num_environments"] == 2
    assert metrics["loss"] == pytest.approx(
        metrics["erm_loss"] + distiller.irm_lambda * metrics["irm_penalty"], abs=0.0002)
    assert any(not torch.equal(old, new) for old, new in zip(before, distiller.model.parameters()))
    assert distiller.optimizer.state


def test_empty_batches_rejected(distiller):
    with pytest.raises(ValueError, match="Training batch"):
        distiller.train_step([])
    with pytest.raises(ValueError, match="Environment batch"):
        distiller.compute_environment_loss([])
    with pytest.raises(ValueError, match="positive"):
        distiller.run_iteration(steps=0)


def test_invalid_samples_rejected(distiller):
    with pytest.raises(ValueError, match="leaf_tokens"):
        distiller.train_step([{"candidate_ids": ["a"]}])


def test_forward_error_propagates(distiller, batch):
    del batch[0]["type"]  # The real model rejects a malformed forward input.
    with pytest.raises(KeyError, match="type"):
        distiller.train_step(batch)
    assert not distiller.optimizer.state


def test_optimizer_error_propagates(distiller, batch):
    def reject_step(optimizer, args, kwargs):
        raise RuntimeError("optimizer step rejected by hook")
    handle = distiller.optimizer.register_step_pre_hook(reject_step)
    try:
        with pytest.raises(RuntimeError, match="optimizer step rejected by hook"):
            distiller.train_step(batch)
    finally:
        handle.remove()
    assert not distiller.optimizer.state


def test_nonfinite_loss_rejected(distiller, batch):
    batch[0]["value_target"] = float("nan")
    with pytest.raises(RuntimeError, match="not finite"):
        distiller.train_step(batch)
    assert not distiller.optimizer.state


def test_backward_error_propagates(distiller, batch):
    def reject_gradient(gradient):
        raise RuntimeError("backward rejected by hook")
    handle = distiller.model.scalar.weight.register_hook(reject_gradient)
    try:
        with pytest.raises(RuntimeError, match="backward rejected by hook"):
            distiller.train_step(batch)
    finally:
        handle.remove()
    assert not distiller.optimizer.state
