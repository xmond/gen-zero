"""Ordinal regression for SummEval's 1--5 quality ratings.

The proportional-odds parameterization predicts the four logits
``P(Y > k)``.  Their cumulative probabilities define a valid five-class
distribution and make the ordering of the ratings explicit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def wasserstein1_loss(probabilities: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Return ordinal Wasserstein-1 distance to the target rating.

    For a target point mass, W1 is the sum of absolute differences between
    the predicted and target CDFs.  Consequently moving probability mass by
    one rating costs less than moving it several ratings.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch, num_classes >= 2]")
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("targets must have shape [batch]")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        targets = targets.long()
    if torch.any((targets < 1) | (targets > probabilities.shape[1])):
        raise ValueError("targets must be 1-indexed ratings in [1, num_classes]")

    cdf = probabilities.cumsum(dim=-1)[..., :-1]
    dev = probabilities.device
    targets = targets.to(dev)
    ranks = torch.arange(1, probabilities.shape[1], device=dev)
    target_cdf = (ranks.unsqueeze(0) >= targets.unsqueeze(1)).to(probabilities.dtype)
    loss = (cdf - target_cdf).abs().sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


MIN_CUTPOINT_GAP = 1e-4


def ordinal_nll_loss(probabilities: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Negative log-likelihood of the 1-indexed target rating (ordinal cross entropy)."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch, num_classes >= 2]")
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("targets must have shape [batch]")
    targets = targets.long().to(probabilities.device)
    if torch.any((targets < 1) | (targets > probabilities.shape[1])):
        raise ValueError("targets must be 1-indexed ratings in [1, num_classes]")
    picked = probabilities.gather(1, (targets - 1).unsqueeze(1)).squeeze(1)
    # The floor only guards log(0) on underflow; it never changes valid mass.
    loss = -torch.log(picked.clamp_min(torch.finfo(probabilities.dtype).tiny))
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


class ProportionalOddsHead(nn.Module):
    """Cumulative-link head with strictly increasing cutpoints.

    ``b_0 = first_cutpoint``, ``b_k = b_{k-1} + softplus(raw_k) + 1e-4``.
    Any parameter value keeps ``b_0 < b_1 < ...``, so ``P(Y > k)`` is
    non-increasing and every class probability is non-negative.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 5) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.score = nn.Linear(hidden_dim, 1)
        n_gaps = num_classes - 2
        self.first_cutpoint = nn.Parameter(torch.tensor(-(num_classes - 2) / 2.0))
        # softplus(0.5413) ~= 1.0: cutpoints start about one unit apart.
        self.raw_cutpoint_deltas = nn.Parameter(torch.full((n_gaps,), 0.5413))

    @property
    def thresholds(self) -> Tensor:
        """The K-1 strictly increasing cutpoints."""
        gaps = F.softplus(self.raw_cutpoint_deltas) + MIN_CUTPOINT_GAP
        return torch.cat((self.first_cutpoint.reshape(1), self.first_cutpoint + gaps.cumsum(0)))

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        score = self.score(hidden)
        # P(Y > k) = sigmoid(score - b_k); b increasing so this is non-increasing in k.
        cumulative = torch.sigmoid(score - self.thresholds)
        ones = torch.ones_like(cumulative[..., :1])
        zeros = torch.zeros_like(cumulative[..., :1])
        upper = torch.cat((ones, cumulative), dim=-1)
        lower = torch.cat((cumulative, zeros), dim=-1)
        probabilities = upper - lower
        ranks = torch.arange(1, self.num_classes + 1, device=hidden.device, dtype=probabilities.dtype)
        expected = (probabilities * ranks).sum(-1, keepdim=True)
        return {"score": score, "cumulative_probabilities": cumulative, "probabilities": probabilities, "expected_score": expected}

    def predict(self, hidden: Tensor) -> Tensor:
        """Median rating (1-indexed): first class whose CDF reaches 0.5."""
        cdf = self(hidden)["probabilities"].cumsum(dim=-1)
        return (cdf < 0.5).sum(dim=-1) + 1

    def expected_score(self, hidden: Tensor) -> Tensor:
        """E[Y] = sum_k k * P(Y=k), shape [batch]."""
        return self(hidden)["expected_score"].squeeze(-1)

    def nll_loss(self, hidden: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
        return ordinal_nll_loss(self(hidden)["probabilities"], targets, reduction)

    def loss(self, hidden: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
        return wasserstein1_loss(self(hidden)["probabilities"], targets, reduction)


# Descriptive aliases for callers using the terminology from the task.
OrdinalHead = ProportionalOddsHead
wasserstein_loss = wasserstein1_loss
nll_loss = ordinal_nll_loss
