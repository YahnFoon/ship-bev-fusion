"""Segmentation metrics for BEV masks."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping

import torch


def mask_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> Dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).bool()
    truth = target.bool()
    tp = torch.logical_and(pred, truth).sum().item()
    fp = torch.logical_and(pred, ~truth).sum().item()
    fn = torch.logical_and(~pred, truth).sum().item()
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def mean_metrics(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    totals = defaultdict(float)
    count = 0
    for row in rows:
        count += 1
        for key in ("iou", "dice", "precision", "recall"):
            totals[key] += float(row[key])
    return {key: totals[key] / max(count, 1) for key in ("iou", "dice", "precision", "recall")}
