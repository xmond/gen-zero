"""Focused regressions for model tensor placement and precision consistency."""

import numpy as np
import pytest
import torch

from gen_zero.nanocore.bidirectional_slot_attention import BidirectionalNanoCore
from gen_zero.world_model.neural_dynamics import (
    NeuralDynamicsWorldModel,
    joint_loss,
    make_synthetic_transitions,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_neural_dynamics_step_matches_parameter_dtype(dtype):
    model = NeuralDynamicsWorldModel(state_dim=3, action_dim=2, hidden_dim=8, action_vocab=["left", "right"])
    model.to(dtype=dtype)
    model._weights_loaded = True

    next_state, reward, done = model.step(
        np.asarray([0.1, -0.2, 0.3], dtype=np.float64),
        np.asarray([1.0, 0.0], dtype=np.float64),
    )

    assert next_state.shape == (3,)
    assert next_state.dtype == np.float32
    assert np.all(np.isfinite(next_state))
    assert 0.0 <= reward <= 1.0
    assert done == (reward < model.done_threshold)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_neural_dynamics_step_returns_cpu_numpy_after_cuda_inference():
    model = NeuralDynamicsWorldModel(state_dim=3, action_dim=2, hidden_dim=8).to(
        device="cuda", dtype=torch.float64
    )
    model._weights_loaded = True

    next_state, reward, done = model.step(
        np.zeros(3, dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert isinstance(next_state, np.ndarray)
    assert next_state.dtype == np.float32
    assert next_state.shape == (3,)
    assert np.all(np.isfinite(next_state))
    assert 0.0 <= reward <= 1.0
    assert done == (reward < model.done_threshold)


def test_neural_dynamics_loss_aligns_float32_batch_to_float64_model():
    data = make_synthetic_transitions(8, state_dim=3, action_vocab=["left", "right"], seed=2)
    model = NeuralDynamicsWorldModel(state_dim=3, action_dim=2, hidden_dim=8).double()

    loss, state_error, bce = joint_loss(model, data.as_tensors(), bce_weight=1.0)

    assert loss.dtype == torch.float64
    assert state_error.dtype == torch.float64
    assert bce.dtype == torch.float64
    assert torch.isfinite(loss)


def test_nanocore_output_heads_follow_attention_dtype_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = BidirectionalNanoCore(embed_dim=8, num_heads=2, device=device)
    core.attn.to(dtype=torch.float64)

    context = torch.randn(1, 2, 8, dtype=torch.float32)
    slots = torch.randn(1, 4, 8, dtype=torch.float32)
    result = core.forward_slots(context, slots)

    attention_parameter = next(core.attn.parameters())
    for head in (core.action_head, core.target_head, core.done_head, core.risk_head):
        assert head.weight.device == attention_parameter.device
        assert head.weight.dtype == attention_parameter.dtype
    assert result["action_logits"].dtype == torch.float64
    assert result["updated_slots"].dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_latent_transition_step_and_batch_follow_parameters(dtype):
    from gen_zero.world_model.latent_dynamics import LatentTransitionModel

    model = LatentTransitionModel(latent_dim=4, action_dim=3, hidden_dim=8)
    model.net.to(dtype=dtype)
    model.device = "meta"  # Configuration may be stale after moving the network.
    state = torch.ones(4, dtype=torch.float32)
    actions = [0, [1.0], torch.tensor([0.0, 1.0], dtype=torch.float32)]
    batch = model.step_batch(state, actions)
    for action, (batched_state, reward, variance) in zip(actions, batch):
        single_state, single_reward, single_variance = model.step(state, action)
        assert single_state.dtype == batched_state.dtype == dtype
        assert single_state.device == next(model.net.parameters()).device
        torch.testing.assert_close(single_state, batched_state)
        assert reward == pytest.approx(single_reward)
        assert variance == pytest.approx(single_variance)
        assert torch.isfinite(single_state).all()
    shock, delta = model.compute_causal_shock(state, 0, state.double())
    assert np.isfinite(shock)
    assert delta.dtype == dtype


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_latent_transition_rejects_nonfinite_state_and_action(value):
    from gen_zero.world_model.latent_dynamics import LatentTransitionModel

    model = LatentTransitionModel(latent_dim=4, action_dim=3, hidden_dim=8)
    state = torch.ones(4)
    for call in (lambda s, a: model.step(s, a), lambda s, a: model.step_batch(s, [a])):
        with pytest.raises(ValueError, match="non-finite"):
            call(torch.full((4,), value), 0)
        with pytest.raises(ValueError, match="non-finite"):
            call(state, [value])


def test_latent_transition_rejects_nonfinite_network_output():
    from gen_zero.world_model.latent_dynamics import LatentTransitionModel

    model = LatentTransitionModel(latent_dim=4, action_dim=3, hidden_dim=8)
    with torch.no_grad():
        next(model.net.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.step(torch.ones(4), 0)
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.step_batch(torch.ones(4), [0, 1])


def test_pipeline_device_resolution_preserves_index_and_mps(monkeypatch):
    from gen_zero.world_model.latent_dynamics import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_torch_device("cuda:1") == "cuda:1"
    assert resolve_torch_device(torch.device("cuda:0")) == "cuda:0"
    assert resolve_torch_device("mps") == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_torch_device("cuda:1") == "cpu"


def test_latent_transition_rejects_ambiguous_batches():
    from gen_zero.world_model.latent_dynamics import LatentTransitionModel

    model = LatentTransitionModel(latent_dim=4, action_dim=3, hidden_dim=8)
    with pytest.raises(ValueError, match="one latent state"):
        model.step(torch.ones(2, 4), 0)
    with pytest.raises(ValueError, match="one vector"):
        model.step_batch(torch.ones(4), [torch.ones(2, 3)])
