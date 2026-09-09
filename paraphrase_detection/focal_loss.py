import math

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary or multilabel classification logits."""

    def __init__(self, gamma=2.0):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be a finite, non-negative number")
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        probability_of_true_class = torch.exp(-bce_loss)
        focal_loss = (
            (1.0 - probability_of_true_class) ** self.gamma * bce_loss
        )
        return focal_loss.mean()


def create_focal_loss(gamma=2.0):
    """Create binary focal loss with the requested focusing parameter."""
    return BinaryFocalLoss(gamma=gamma)
