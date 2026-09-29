import pytest
import torch

from gen_zero.causal.modernbert_manifold_projector import (
    LyapunovAttractor,
    ModernBertManifoldProjector,
)


def test_mean_pooling_projection_shape_and_gradients():
    torch.manual_seed(11)
    bridge = ModernBertManifoldProjector(pooling="mean")
    hidden = torch.randn(2, 4, 1024, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])
    output = bridge(hidden, mask)
    assert output.target.shape == output.state.shape == (2, 64)
    assert output.converged
    torch.testing.assert_close(output.pooled[0], hidden.detach()[0, :2].mean(0))
    output.state.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_cls_pooling_uses_first_unmasked_token_without_language_assumptions():
    bridge = ModernBertManifoldProjector(pooling="cls")
    hidden = torch.randn(2, 3, 1024)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    pooled = bridge.pool(hidden, mask)
    torch.testing.assert_close(pooled[0], hidden[0, 0])
    torch.testing.assert_close(pooled[1], hidden[1, 1])


def test_lyapunov_energy_contracts_by_exact_declared_factor():
    dynamics = LyapunovAttractor(64, contraction=0.25)
    target = torch.randn(3, 64)
    state = torch.randn(3, 64)
    before = (state - target).square().sum(dim=1)
    after_state = dynamics.step(state, target)
    after = (after_state - target).square().sum(dim=1)
    torch.testing.assert_close(after, before * 0.25**2, rtol=2e-5, atol=1e-6)


def test_projection_is_exact_64_by_1024_and_payload_is_reported():
    bridge = ModernBertManifoldProjector()
    assert bridge.projection.weight.shape == (64, 1024)
    expected = sum(p.numel() * p.element_size() for p in bridge.parameters())
    assert bridge.projection_payload_bytes == expected


def test_fail_closed_validation_and_calibration_provenance():
    with pytest.raises(ValueError, match="provenance"):
        ModernBertManifoldProjector(calibrated=True)
    bridge = ModernBertManifoldProjector()
    with pytest.raises(ValueError, match="unmasked"):
        bridge(torch.ones(1, 2, 1024), torch.zeros(1, 2))
    bad = torch.ones(1, 1024)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        bridge(bad)

