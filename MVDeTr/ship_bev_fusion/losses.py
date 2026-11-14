"""Losses for BEV mask fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * target, dim=dims)
    denominator = torch.sum(probs + target, dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + float(dice_weight) * dice_loss_with_logits(logits, target)
