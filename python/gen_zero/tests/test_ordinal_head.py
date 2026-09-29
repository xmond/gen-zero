import torch

from gen_zero.model.ordinal_head import ProportionalOddsHead, wasserstein1_loss


def test_wasserstein_penalizes_distant_rank_errors_more_than_adjacent():
    target = torch.tensor([5])
    adjacent = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    distant = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    assert wasserstein1_loss(adjacent, target).item() < wasserstein1_loss(distant, target).item()


def test_proportional_odds_outputs_ordered_distribution_and_expected_score():
    head = ProportionalOddsHead(hidden_dim=3)
    output = head(torch.ones(2, 3))
    assert torch.all(output["cumulative_probabilities"][:, :-1] >= output["cumulative_probabilities"][:, 1:])
    assert torch.allclose(output["probabilities"].sum(-1), torch.ones(2), atol=1e-6)
    assert torch.all((output["expected_score"] >= 1) & (output["expected_score"] <= 5))
