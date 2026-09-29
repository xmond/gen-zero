import pytest
import torch

from gen_zero.causal.student_manifold_projector import (
    CausalStateDynamics,
    StudentCausalDecisionPipeline,
    StudentManifoldProjector,
)


@pytest.mark.parametrize("rank", [None, 8])
def test_projection_shapes_gradients_and_mask_pooling(rank):
    torch.manual_seed(7)
    module = StudentManifoldProjector(896, 64, rank=rank)
    hidden = torch.randn(2, 5, 896, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 0, 1, 0]])
    result = module(hidden, mask)
    assert result.shape == (2, 64)
    result.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    pooled = module.pool(hidden.detach(), mask)
    torch.testing.assert_close(pooled[0], hidden.detach()[0, 2])
    torch.testing.assert_close(pooled[1], hidden.detach()[1, 3])


def test_invalid_and_nonfinite_inputs_fail_closed():
    module = StudentManifoldProjector()
    with pytest.raises(ValueError, match="shape"):
        module(torch.ones(2, 64))
    bad = torch.ones(1, 2, 896)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module(bad)
    with pytest.raises(ValueError, match="at least one"):
        module(torch.ones(1, 2, 896), torch.zeros(1, 2))


def test_dynamics_changes_state_and_backpropagates():
    dynamics = CausalStateDynamics(64, steps=3)
    state = torch.randn(4, 64, requires_grad=True)
    final = dynamics(state)
    assert final.shape == state.shape
    assert not torch.equal(final, state)
    final.sum().backward()
    assert torch.isfinite(state.grad).all()


def test_pipeline_refuses_uncalibrated_semantic_decision():
    pipeline = StudentCausalDecisionPipeline(StudentManifoldProjector(rank=8), 3)
    hidden = torch.randn(2, 4, 896)
    output = pipeline(hidden)
    assert output.initial_state.shape == output.final_state.shape == (2, 64)
    assert output.logits.shape == (2, 3)
    assert output.decision is None
    with pytest.raises(RuntimeError, match="not calibrated"):
        pipeline.decide(hidden)
    diagnostic = pipeline(hidden, allow_uncalibrated=True)
    assert diagnostic.decision.shape == (2,)


def test_parameter_count_low_rank_is_smaller():
    dense = StudentManifoldProjector(rank=None)
    low_rank = StudentManifoldProjector(rank=8)
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert count(low_rank) < count(dense)
