"""Visualization helpers for BEV fusion outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np


def save_comparison_figure(
    output_path: Path,
    input_bev_masks: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    title: str,
    metrics: Mapping[str, float],
    camera_ids: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [_panel(mask, f"cam {camera_id} BEV") for mask, camera_id in zip(input_bev_masks, camera_ids)]
    panels.extend([_panel(pred_mask, "prediction"), _panel(target_mask, "221 truth"), _error_panel(pred_mask, target_mask, "error overlay")])
    row = np.concatenate(panels, axis=1)
    header = np.full((54, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    metric_text = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    cv2.putText(header, metric_text, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), np.concatenate([header, row], axis=0))


def save_raw_view_diagnostic_figure(
    output_path: Path,
    *,
    raw_rgb: np.ndarray,
    raw_masks: np.ndarray,
    roi_rgb: np.ndarray,
    projected_features: np.ndarray,
    fused_features: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    title: str,
    metrics: Mapping[str, float],
    camera_ids: Sequence[str],
    input_mode: Optional[str] = None,
    aligned_features: Optional[np.ndarray] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(_image_row(raw_rgb, camera_ids, prefix="rgb"))
    rows.append(_mask_row(raw_masks, camera_ids, prefix="mask"))
    rows.append(_image_row(roi_rgb, camera_ids, prefix="roi"))
    rows.append(_mask_row(_feature_mean_per_view(projected_features), camera_ids, prefix="proj"))
    if aligned_features is not None:
        rows.append(_mask_row(_feature_mean_per_view(aligned_features), camera_ids, prefix="align"))
    bottom = [
        _panel(_feature_mean(fused_features), "fused"),
        _panel(pred_mask, "prediction"),
        _panel(target_mask, "GT"),
        _error_panel(pred_mask, target_mask, "diff"),
    ]
    while len(bottom) < len(camera_ids):
        bottom.append(np.zeros_like(bottom[0]))
    rows.append(np.concatenate(bottom[: len(camera_ids)], axis=1))
    body = np.concatenate(rows, axis=0)
    header = np.full((64, body.shape[1], 3), 255, dtype=np.uint8)
    mode_text = f" input={input_mode}" if input_mode else ""
    cv2.putText(header, f"{title}{mode_text}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    metric_text = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    cv2.putText(header, metric_text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), np.concatenate([header, body], axis=0))


def _panel(mask: np.ndarray, label: str, size: int = 256) -> np.ndarray:
    mask = cv2.resize(_to_uint8(mask), (size, size), interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(color, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return color


def _image_panel(image: np.ndarray, label: str, size: int = 256) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = image.transpose(1, 2, 0)
    if image.ndim == 2:
        return _panel(image, label, size=size)
    if image.max() <= 1.0:
        image = (image.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return image


def _image_row(images: np.ndarray, camera_ids: Sequence[str], prefix: str, size: int = 256) -> np.ndarray:
    return np.concatenate([_image_panel(image, f"{prefix} {camera_id}", size=size) for image, camera_id in zip(images, camera_ids)], axis=1)


def _mask_row(masks: np.ndarray, camera_ids: Sequence[str], prefix: str, size: int = 256) -> np.ndarray:
    return np.concatenate([_panel(mask, f"{prefix} {camera_id}", size=size) for mask, camera_id in zip(masks, camera_ids)], axis=1)


def _feature_mean_per_view(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim == 4:
        return np.stack([_normalize_feature(item) for item in features.mean(axis=1)], axis=0)
    if features.ndim == 3:
        return np.stack([_normalize_feature(item) for item in features], axis=0)
    return _normalize_feature(np.squeeze(features))


def _feature_mean(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim == 3:
        return _normalize_feature(features.mean(axis=0))
    return _normalize_feature(np.squeeze(features))


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.size == 0:
        return feature
    finite = np.isfinite(feature)
    if not finite.any():
        return np.zeros_like(feature, dtype=np.float32)
    lo = float(feature[finite].min())
    hi = float(feature[finite].max())
    if hi <= lo + 1e-12:
        return np.zeros_like(feature, dtype=np.float32)
    return ((feature - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)


def _error_panel(pred: np.ndarray, target: np.ndarray, label: str, size: int = 256) -> np.ndarray:
    pred_bool = _to_uint8(pred) > 127
    target_bool = _to_uint8(target) > 127
    rgb = np.zeros((*pred_bool.shape, 3), dtype=np.uint8)
    rgb[np.logical_and(pred_bool, target_bool)] = (255, 255, 255)
    rgb[np.logical_and(pred_bool, ~target_bool)] = (0, 0, 255)
    rgb[np.logical_and(~pred_bool, target_bool)] = (0, 255, 255)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.putText(rgb, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return rgb


def _to_uint8(mask: np.ndarray) -> np.ndarray:
    mask = np.squeeze(mask)
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255
    if mask.size == 0:
        return mask.astype(np.uint8)
    if mask.max() <= 1.0:
        return (mask.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    return mask.astype(np.uint8)
