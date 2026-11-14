#!/usr/bin/env python3
"""Stage-2 joint training: Mask2Former weak masks -> BEV DeformTrans fusion."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import register_all_modules
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MVDETR_ROOT = PROJECT_ROOT / "MVDeTr"
sys.path.insert(0, str(MVDETR_ROOT))

if os.environ.get("INDOOR_MASK2FORMER_MSDA_BACKEND", "pytorch").strip().lower() in {
    "torch",
    "pytorch",
    "fallback",
    "grid_sample",
}:
    import scripts.patch_mmcv_msda_to_pytorch  # noqa: F401,E402

from ship_bev_fusion.config import load_config  # noqa: E402
from ship_bev_fusion.dataset import binarize  # noqa: E402
from ship_bev_fusion.losses import dice_loss_with_logits  # noqa: E402
from ship_bev_fusion.metrics import mask_metrics, mean_metrics  # noqa: E402
from ship_bev_fusion.lightweight import WeightMaskController, apply_compression_recipe  # noqa: E402
from ship_bev_fusion.mvdetr_dataset import ShipMVDeTrMaskDataset, VideoFrameStore  # noqa: E402
from ship_bev_fusion.mvdetr_model import MVDeTrShipMask  # noqa: E402
from ship_bev_fusion.calibration import undistort_mask, warp_mask_to_bev  # noqa: E402
from ship_bev_fusion.runtime import (  # noqa: E402
    MODELOPT_BASE_FUSION_KEY,
    MODELOPT_BASE_MASK2FORMER_KEY,
    MODELOPT_STATE_DICT_TRANSFORM,
    is_modelopt_state_dict_checkpoint,
    restore_modelopt_state_dict_models,
)


IMAGENET_BGR_MEAN = torch.tensor([103.53, 116.28, 123.675], dtype=torch.float32).view(3, 1, 1)
IMAGENET_BGR_STD = torch.tensor([57.375, 57.12, 58.395], dtype=torch.float32).view(3, 1, 1)
IMAGENET_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
STAGE2_INPUT_MODES = ("bev_soft_mask", "rgb_mask_overlay")
INITIALIZATION_DATASET_COMPATIBILITY_KEYS = (
    "input_camera_ids",
    "target_camera_id",
    "plane_z_mm",
    "bev_size",
    "input_size",
    "target_homography_image_size",
    "target_homography_direction",
    "target_homography_space",
    "assume_world_xy_is_target_image",
    "target_world_bounds_margin",
    "cache_camera_ids",
    "input_mode",
)


def resolve_fusion_config_for_initialization(
    configured_fusion_cfg: Mapping[str, Any],
    initialization_checkpoint: Optional[Mapping[str, Any]],
    *,
    stage2_input_mode: str,
    roi_background_weight: float,
) -> Dict[str, Any]:
    """Keep runtime data paths while inheriting the initialized model architecture."""
    configured = copy.deepcopy(dict(configured_fusion_cfg))
    requested_stage2 = {
        "input_mode": str(stage2_input_mode),
        "roi_background_weight": float(roi_background_weight),
    }
    if initialization_checkpoint is None:
        configured["stage2"] = requested_stage2
        return configured

    checkpoint_cfg_value = initialization_checkpoint.get("fusion_config")
    if not isinstance(checkpoint_cfg_value, Mapping):
        if is_modelopt_state_dict_checkpoint(initialization_checkpoint):
            raise ValueError("ModelOpt initialization checkpoint is missing fusion_config")
        configured["stage2"] = requested_stage2
        return configured

    checkpoint_cfg = copy.deepcopy(dict(checkpoint_cfg_value))
    if not isinstance(checkpoint_cfg.get("model"), Mapping):
        raise ValueError("Initialization checkpoint fusion_config is missing model architecture")

    checkpoint_stage2 = dict(checkpoint_cfg.get("stage2", {}))
    checkpoint_mode = checkpoint_stage2.get("input_mode")
    if checkpoint_mode is not None and str(checkpoint_mode) != requested_stage2["input_mode"]:
        raise ValueError(
            "Initialization checkpoint stage2 input mode mismatch: "
            f"checkpoint={checkpoint_mode!r} requested={requested_stage2['input_mode']!r}"
        )
    checkpoint_background = checkpoint_stage2.get("roi_background_weight")
    if checkpoint_background is not None and not math.isclose(
        float(checkpoint_background),
        requested_stage2["roi_background_weight"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Initialization checkpoint ROI background weight mismatch: "
            f"checkpoint={float(checkpoint_background)} "
            f"requested={requested_stage2['roi_background_weight']}"
        )

    configured_dataset = copy.deepcopy(dict(configured.get("dataset", {})))
    checkpoint_dataset = dict(checkpoint_cfg.get("dataset", {}))
    mismatches = []
    for key in INITIALIZATION_DATASET_COMPATIBILITY_KEYS:
        if key in configured_dataset and key in checkpoint_dataset:
            if configured_dataset[key] != checkpoint_dataset[key]:
                mismatches.append(
                    f"{key}: checkpoint={checkpoint_dataset[key]!r} configured={configured_dataset[key]!r}"
                )
    if mismatches:
        raise ValueError(
            "Initialization checkpoint dataset geometry is incompatible with the configured dataset: "
            + "; ".join(mismatches)
        )

    checkpoint_cfg["dataset"] = configured_dataset
    checkpoint_cfg["stage2"] = requested_stage2
    if "output_dir" in configured:
        checkpoint_cfg["output_dir"] = configured["output_dir"]
    return checkpoint_cfg


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_stage2_provenance(
    output_dir: Path,
    args: argparse.Namespace,
    fusion_cfg: Mapping[str, Any],
) -> None:
    dataset_cfg = dict(fusion_cfg.get("dataset", {}))
    manifest_path = Path(dataset_cfg["split_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = Path(dataset_cfg["root"]) / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Locked split manifest does not exist: {manifest_path}")
    git_commit = "source-snapshot-see-protocol.json"
    resolved = {
        "mmdet_config": str(args.mmdet_config.resolve()),
        "mask2former_checkpoint": str(args.mask2former_checkpoint.resolve()),
        "initial_stage2_checkpoint": (
            str(args.initial_stage2_checkpoint.resolve()) if args.initial_stage2_checkpoint else None
        ),
        "initial_stage2_checkpoint_sha256": (
            sha256_file(args.initial_stage2_checkpoint) if args.initial_stage2_checkpoint else None
        ),
        "epochs": int(args.epochs),
        "micro_batch_size": int(args.batch_size),
        "grad_accumulation_steps": int(args.grad_accumulation_steps),
        "effective_batch_size": int(args.batch_size) * int(args.grad_accumulation_steps),
        "num_workers": int(args.num_workers),
        "lr_fusion": float(args.lr_fusion),
        "lr_mask2former": float(args.lr_mask2former),
        "amp": False,
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "bev_only": bool(args.bev_only or float(args.mask2former_loss_weight) <= 0.0),
        "stage2_input_mode": str(args.stage2_input_mode),
        "roi_background_weight": float(args.roi_background_weight),
        "use_rgb_cache": bool(args.use_rgb_cache),
        "fusion_config": dict(fusion_cfg),
        "git_commit": git_commit,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output_dir / "run_config_resolved.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
    (output_dir / "manifest_sha256.txt").write_text(resolved["manifest_sha256"] + "\n", encoding="utf-8")
    (output_dir / "split_manifest_used.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output_dir / "command_history.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )


def report_msda_backends(*, require_expected: bool = False) -> Dict[str, str]:
    import mmcv.ops.multi_scale_deform_attn as mmcv_msda
    from multiview_detector.models.ops.functions import ms_deform_attn_func as fusion_msda

    mmcv_function = mmcv_msda.MultiScaleDeformableAttnFunction
    fusion_function = fusion_msda.MSDeformAttnFunction
    mask2former_backend = os.environ.get(
        "INDOOR_MASK2FORMER_MSDA_BACKEND", "pytorch"
    ).strip().lower()
    details = {
        "mask2former": f"{mmcv_function.__module__}.{mmcv_function.__name__}",
        "mask2former_backend_env": mask2former_backend,
        "deform_trans": (
            "pytorch_grid_sample"
            if fusion_function is None
            else f"{fusion_function.__module__}.{fusion_function.__name__}"
        ),
        "deform_trans_backend_env": os.environ.get("MVDETR_MS_DEFORM_ATTN_BACKEND", "auto"),
    }
    print(
        "MSDeformAttn backends: "
        f"Mask2Former={details['mask2former']} "
        f"INDOOR_MASK2FORMER_MSDA_BACKEND={details['mask2former_backend_env']} "
        f"DeformTrans={details['deform_trans']} "
        f"MVDETR_MS_DEFORM_ATTN_BACKEND={details['deform_trans_backend_env']}",
        flush=True,
    )
    if require_expected:
        if mask2former_backend in {"torch", "pytorch", "fallback", "grid_sample"}:
            if not getattr(mmcv_msda, "_ship_patch_force_pytorch_msda", False):
                raise RuntimeError(
                    "Mask2Former must use the MMCV PyTorch fallback, but the "
                    "process-level patch is not active."
                )
        elif not details["mask2former"].startswith("mmcv.ops.multi_scale_deform_attn."):
            raise RuntimeError(
                "Mask2Former must use MMCV native MultiScaleDeformableAttention; "
                f"got {details['mask2former']}"
            )
        if details["deform_trans_backend_env"].strip().lower() not in {"custom", "project"}:
            raise RuntimeError(
                "DeformTrans must use the project custom MSDeformAttn backend; "
                f"got MVDETR_MS_DEFORM_ATTN_BACKEND={details['deform_trans_backend_env']}"
            )
        if not details["deform_trans"].startswith(
            "multiview_detector.models.ops.functions.ms_deform_attn_func."
        ):
            raise RuntimeError(
                "DeformTrans did not load the project custom MSDeformAttn function; "
                f"got {details['deform_trans']}"
            )
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmdet-config", type=Path, required=True)
    parser.add_argument("--mask2former-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-config", type=Path, required=True)
    parser.add_argument("--mask-dir", default=None, help="Override fusion-config dataset.mask_dir, relative to dataset.root unless absolute.")
    parser.add_argument("--target-mask-dir", default=None, help="Override fusion-config dataset.target_mask_dir.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate this many micro-batches before each optimizer step.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-fusion", type=float, default=2e-4)
    parser.add_argument(
        "--lr-mask2former",
        type=float,
        default=2e-7,
        help="Conservative online fine-tuning LR for the pretrained Mask2Former.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask2former-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--bev-only",
        action="store_true",
        help="Train Mask2Former+DeformTrans with BEV loss only; weak-mask instance loss is skipped.",
    )
    parser.add_argument(
        "--stage2-input-mode",
        choices=STAGE2_INPUT_MODES,
        default="bev_soft_mask",
        help=(
            "Fusion input construction. 'bev_soft_mask' is the original pipeline "
            "that projects Mask2Former soft masks to BEV. 'rgb_mask_overlay' keeps "
            "image-plane RGB and uses the online soft mask as an ROI overlay before "
            "image-ResNet + DeformTrans fusion."
        ),
    )
    parser.add_argument(
        "--roi-background-weight",
        type=float,
        default=0.25,
        help=(
            "For --stage2-input-mode rgb_mask_overlay, keep this fraction of the "
            "background RGB intensity outside the soft-mask ROI. 0 means hard "
            "masking; 1 disables ROI dimming."
        ),
    )
    parser.add_argument(
        "--use-rgb-cache",
        action="store_true",
        help="Read normalized RGB/target/affine from the fusion config cache instead of decoding videos and weak masks.",
    )
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument(
        "--boundary-iou-weight",
        type=float,
        default=0.0,
        help="Optional weight for a differentiable BEV boundary IoU loss.",
    )
    parser.add_argument(
        "--boundary-kernel-size",
        type=int,
        default=5,
        help="Odd max-pool kernel size used to extract soft BEV boundaries.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--max-frames-per-condition", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initial-stage2-checkpoint",
        type=Path,
        default=None,
        help="Optional prepared UP/NM24/SP/LR checkpoint used before recovery training.",
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Keep one full Stage-2 checkpoint per epoch under output_dir/checkpoints/.",
    )
    return parser.parse_args()


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 + 1), float(y1 + 1)]


def resize_binary_mask(mask: np.ndarray, size: Tuple[int, int], threshold: int) -> np.ndarray:
    input_w, input_h = size
    resized = cv2.resize(mask, (int(input_w), int(input_h)), interpolation=cv2.INTER_NEAREST)
    return (binarize(resized, int(threshold)) > 0).astype(np.uint8)


class Stage2AlignedDataset(Dataset):
    """Read aligned four-view RGB, weak masks, target BEV, and projection affine."""

    def __init__(
        self,
        split: str,
        fusion_cfg: Mapping[str, Any],
        max_frames_per_condition: Optional[int] = None,
        *,
        bev_only: bool = False,
        use_rgb_cache: bool = False,
    ) -> None:
        dataset_cfg = dict(fusion_cfg["dataset"])
        dataset_cfg["use_cache"] = bool(use_rgb_cache)
        dataset_cfg["input_mode"] = "image_rgb"
        if max_frames_per_condition is not None:
            dataset_cfg["max_frames_per_condition"] = int(max_frames_per_condition)
        self.base_dataset = ShipMVDeTrMaskDataset(split=split, **dataset_cfg)
        self.records = self.base_dataset.records
        self.split = split
        self.bev_only = bool(bev_only)
        self.use_rgb_cache = bool(use_rgb_cache)
        if self.use_rgb_cache:
            self.video_store = None
            return
        if self.base_dataset.video_root is None:
            raise ValueError("Stage2 dataset requires video_dir in fusion config")
        self.video_store = self.base_dataset.video_store or VideoFrameStore(
            self.base_dataset.video_root,
            self.base_dataset.input_camera_ids,
            per_condition=self.base_dataset.per_condition,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.use_rgb_cache:
            cached = self.base_dataset[index]
            inputs = cached["inputs"].float()
            if inputs.ndim != 4 or inputs.shape[1] != 3:
                raise ValueError(f"RGB cache must have shape [N,3,H,W], got {tuple(inputs.shape)}")


            mean = inputs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = inputs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            rgb01 = (inputs * std + mean).clamp(0.0, 1.0)
            bgr255 = rgb01[:, [2, 1, 0], :, :] * 255.0
            n, _, h, w = bgr255.shape
            weak_masks = torch.zeros((n, 1, h, w), dtype=torch.float32)
            return {
                "mmdet_inputs": bgr255,
                "rgb_inputs": inputs,
                "weak_masks": weak_masks,
                "target": cached["target"].float(),
                "affine": cached["affine"].float(),
                "condition": cached["condition"],
                "frame_index": int(cached["frame_index"]),
            }
        record = self.records[index]
        input_w, input_h = self.base_dataset.input_size
        mmdet_inputs: List[torch.Tensor] = []
        rgb_inputs: List[torch.Tensor] = []
        weak_masks: List[torch.Tensor] = []
        affines: List[np.ndarray] = []
        for camera_id in self.base_dataset.input_camera_ids:
            params = self.base_dataset.base.camera_parameters[camera_id]
            frame_rgb = self.video_store.read_rgb(
                camera_id,
                record.frame_index,
                record.condition if self.base_dataset.per_condition else None,
            )
            frame_rgb = undistort_mask(frame_rgb, params)
            orig_h, orig_w = frame_rgb.shape[:2]
            interpolation = cv2.INTER_AREA if input_w < orig_w or input_h < orig_h else cv2.INTER_LINEAR
            resized_rgb = cv2.resize(frame_rgb, (input_w, input_h), interpolation=interpolation)
            rgb01_np = resized_rgb.astype(np.float32) / 255.0
            rgb_norm = (rgb01_np - np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)) / np.asarray(
                [0.229, 0.224, 0.225],
                dtype=np.float32,
            ).reshape(1, 1, 3)
            resized_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
            mmdet_inputs.append(torch.from_numpy(resized_bgr.transpose(2, 0, 1).copy()))
            rgb_inputs.append(torch.from_numpy(rgb_norm.transpose(2, 0, 1).copy()))
            if self.bev_only:
                weak_masks.append(torch.zeros((1, input_h, input_w), dtype=torch.float32))
            else:
                if self.base_dataset.per_condition:
                    mask = self.base_dataset.mask_stores[camera_id].get_mask(record.condition, record.frame_index)
                else:
                    mask = self.base_dataset.mask_stores[camera_id].get_mask(record.frame_index)
                mask = undistort_mask(mask, params)
                weak = resize_binary_mask(mask, self.base_dataset.input_size, self.base_dataset.threshold)
                weak_masks.append(torch.from_numpy(weak[None, ...].astype(np.float32)))
            affines.append(
                np.array(
                    [[input_w / orig_w, 0.0, 0.0], [0.0, input_h / orig_h, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
            )
        if self.base_dataset.per_condition:
            target = self.base_dataset.mask_stores[self.base_dataset.target_camera_id].get_mask(
                record.condition,
                record.frame_index,
            )
        else:
            target = self.base_dataset.mask_stores[self.base_dataset.target_camera_id].get_mask(record.frame_index)
        target_bev = warp_mask_to_bev(target, self.base_dataset.base.target_image_to_bev_mat, self.base_dataset.bev_size)
        return {
            "mmdet_inputs": torch.stack(mmdet_inputs, dim=0),
            "rgb_inputs": torch.stack(rgb_inputs, dim=0),
            "weak_masks": torch.stack(weak_masks, dim=0),
            "target": torch.from_numpy(binarize(target_bev, self.base_dataset.threshold)[None, ...].astype(np.float32)),
            "affine": torch.from_numpy(np.stack(affines, axis=0).astype(np.float32)),
            "condition": record.condition,
            "frame_index": int(record.frame_index),
        }


def collate_stage2(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "mmdet_inputs": torch.stack([item["mmdet_inputs"] for item in batch], dim=0),
        "rgb_inputs": torch.stack([item["rgb_inputs"] for item in batch], dim=0),
        "weak_masks": torch.stack([item["weak_masks"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "affine": torch.stack([item["affine"] for item in batch], dim=0),
        "condition": [str(item["condition"]) for item in batch],
        "frame_index": torch.tensor([int(item["frame_index"]) for item in batch], dtype=torch.long),
    }


def stage2_worker_init(_: int) -> None:


    cv2.setNumThreads(0)


def resolve_mmdet_config(config_path: Path) -> Path:
    return config_path


def build_detector(config_path: Path, checkpoint_path: Path) -> torch.nn.Module:
    register_all_modules(init_default_scope=True)
    config_path = resolve_mmdet_config(config_path)
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)


    if hasattr(model, "data_preprocessor") and hasattr(model.data_preprocessor, "batch_augments"):
        model.data_preprocessor.batch_augments = None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"loaded Mask2Former checkpoint: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model


def make_data_samples(weak_masks: torch.Tensor) -> List[DetDataSample]:

    b, n, _, h, w = weak_masks.shape
    samples: List[DetDataSample] = []
    weak_np = weak_masks.detach().cpu().numpy()
    for batch_idx in range(b):
        for cam_idx in range(n):
            mask = (weak_np[batch_idx, cam_idx, 0] > 0.5).astype(np.uint8)
            bbox = bbox_from_mask(mask)
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "img_id": int(batch_idx * n + cam_idx),
                    "img_shape": (int(h), int(w)),
                    "ori_shape": (int(h), int(w)),
                }
            )
            instances = InstanceData()
            if bbox is None:
                instances.bboxes = torch.zeros((0, 4), dtype=torch.float32)
                instances.labels = torch.zeros((0,), dtype=torch.long)
                instances.masks = BitmapMasks(np.zeros((0, h, w), dtype=np.uint8), int(h), int(w))
            else:
                instances.bboxes = torch.tensor([bbox], dtype=torch.float32)
                instances.labels = torch.zeros((1,), dtype=torch.long)
                instances.masks = BitmapMasks(mask[None, ...], int(h), int(w))
            sample.gt_instances = instances
            samples.append(sample)
    return samples


def make_prediction_samples(batch_size: int, num_cameras: int, height: int, width: int) -> List[DetDataSample]:
    samples: List[DetDataSample] = []
    for batch_idx in range(int(batch_size)):
        for cam_idx in range(int(num_cameras)):
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "img_id": int(batch_idx * num_cameras + cam_idx),
                    "img_shape": (int(height), int(width)),
                    "ori_shape": (int(height), int(width)),
                }
            )
            samples.append(sample)
    return samples


def sum_loss_dict(losses: Mapping[str, Any]) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for value in losses.values():
        if isinstance(value, torch.Tensor):
            values.append(value.mean())
        elif isinstance(value, (list, tuple)):
            values.extend(item.mean() for item in value if isinstance(item, torch.Tensor))
    if not values:
        raise RuntimeError("Mask2Former loss dict has no tensor losses")
    return sum(values)


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float, dice_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=torch.tensor(float(pos_weight), device=logits.device),
    )
    return bce + float(dice_weight) * dice_loss_with_logits(logits, target)


def soft_boundary_map(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = int(kernel_size)
    if kernel <= 1:
        return torch.zeros_like(mask)
    if kernel % 2 == 0:
        raise ValueError(f"boundary kernel size must be odd, got {kernel}")
    pad = kernel // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_iou_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, kernel_size: int, eps: float = 1e-6) -> torch.Tensor:
    pred_boundary = soft_boundary_map(logits.sigmoid(), int(kernel_size))
    target_boundary = soft_boundary_map(target.float(), int(kernel_size))
    dims = tuple(range(1, pred_boundary.ndim))
    intersection = (pred_boundary * target_boundary).sum(dim=dims)
    union = pred_boundary.sum(dim=dims) + target_boundary.sum(dim=dims) - intersection
    return (1.0 - (intersection + float(eps)) / (union + float(eps))).mean()


def preprocess_with_mmdet(mask_model: torch.nn.Module, flat_inputs: torch.Tensor, samples: List[DetDataSample], training: bool) -> Dict[str, Any]:
    data = {"inputs": [img for img in flat_inputs], "data_samples": samples}
    return mask_model.data_preprocessor(data, training)


def soft_masks_from_mask2former(mask_model: torch.nn.Module, processed: Mapping[str, Any]) -> torch.Tensor:
    feats = mask_model.extract_feat(processed["inputs"])
    mask_cls, mask_pred = mask_model.panoptic_head.predict(feats, processed["data_samples"])

    scores = mask_cls.softmax(dim=-1)[..., 0]
    soft = torch.einsum("bq,bqhw->bhw", scores, mask_pred.sigmoid()).clamp(0.0, 1.0)
    cropped: List[torch.Tensor] = []
    for idx, sample in enumerate(processed["data_samples"]):
        img_h, img_w = sample.metainfo["img_shape"]
        cropped.append(soft[idx, : int(img_h), : int(img_w)])
    return torch.stack(cropped, dim=0).unsqueeze(1)


def project_soft_masks_to_bev(dataset: Stage2AlignedDataset, soft_masks: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    b, n, _, h, w = soft_masks.shape
    world_h, world_w = dataset.base_dataset.Rworld_shape
    mats = torch.stack(
        [
            torch.from_numpy(dataset.base_dataset.base.input_image_to_bev_mats[str(camera_id)]).float()
            for camera_id in dataset.base_dataset.input_camera_ids
        ],
        dim=0,
    ).to(device=soft_masks.device, dtype=soft_masks.dtype)
    inv_affine = torch.inverse(affine.to(device=soft_masks.device, dtype=soft_masks.dtype).view(b * n, 3, 3))
    h_img_to_bev = mats.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * n, 3, 3) @ inv_affine
    flat = soft_masks.reshape(b * n, 1, h, w)
    with torch.amp.autocast("cuda", enabled=False):
        bev = warp_perspective(flat.float(), h_img_to_bev.float(), (world_h, world_w), align_corners=False)
    return bev.view(b, n, 1, world_h, world_w).to(dtype=soft_masks.dtype)


def stage2_input_settings_from_config(fusion_cfg: Mapping[str, Any]) -> Tuple[str, float]:
    stage2_cfg = dict(fusion_cfg.get("stage2", {}))
    mode = str(stage2_cfg.get("input_mode", "bev_soft_mask"))
    if mode not in STAGE2_INPUT_MODES:
        raise ValueError(f"stage2.input_mode must be one of {STAGE2_INPUT_MODES}, got {mode!r}")
    return mode, float(stage2_cfg.get("roi_background_weight", 0.25))


def fusion_model_overrides_for_stage2_input(stage2_input_mode: str) -> Dict[str, Any]:
    if stage2_input_mode == "bev_soft_mask":
        return {
            "feature_mode": "bev_mask_preprojected_direct",
            "input_channels": 1,
            "world_feat_arch": "deform_trans",
            "deform_trans_reference_mode": "camera",
        }
    if stage2_input_mode == "rgb_mask_overlay":
        return {
            "feature_mode": "image_resnet",
            "input_channels": 3,
            "world_feat_arch": "deform_trans",
            "deform_trans_reference_mode": "camera",
        }
    raise ValueError(f"Unsupported stage2 input mode: {stage2_input_mode}")


def build_stage2_fusion_model(
    dataset: Stage2AlignedDataset,
    fusion_cfg: Mapping[str, Any],
    stage2_input_mode: str,
) -> MVDeTrShipMask:
    fusion_model_cfg = dict(fusion_cfg.get("model", {}))
    fusion_model_cfg.update(fusion_model_overrides_for_stage2_input(stage2_input_mode))
    fusion_model_cfg.setdefault("deform_trans_n_points", 12)
    fusion_model_cfg.setdefault("deform_trans_hidden_dim", 128)
    fusion_model_cfg.setdefault("deform_trans_n_heads", 8)
    fusion_model_cfg.setdefault("deform_trans_ffn_dim", 512)
    fusion_model_cfg.setdefault("deform_trans_num_layers", 3)
    fusion_model_cfg.setdefault("deform_trans_stride", 2)
    fusion_model_cfg.setdefault("bottleneck_dim", 128)
    return MVDeTrShipMask(dataset.base_dataset, **fusion_model_cfg)


def mmdet_bgr255_to_rgb_norm(mmdet_inputs: torch.Tensor) -> torch.Tensor:
    rgb01 = (mmdet_inputs[:, :, [2, 1, 0], :, :] / 255.0).clamp(0.0, 1.0)
    mean = IMAGENET_RGB_MEAN.to(device=rgb01.device, dtype=rgb01.dtype).view(1, 1, 3, 1, 1)
    std = IMAGENET_RGB_STD.to(device=rgb01.device, dtype=rgb01.dtype).view(1, 1, 3, 1, 1)
    return (rgb01 - mean) / std


def soft_mask_rgb_overlay(
    rgb_norm: torch.Tensor,
    soft_masks: torch.Tensor,
    background_weight: float,
) -> torch.Tensor:
    keep = float(background_weight)
    if not 0.0 <= keep <= 1.0:
        raise ValueError(f"roi_background_weight must be in [0,1], got {keep}")
    mean = IMAGENET_RGB_MEAN.to(device=rgb_norm.device, dtype=rgb_norm.dtype).view(1, 1, 3, 1, 1)
    std = IMAGENET_RGB_STD.to(device=rgb_norm.device, dtype=rgb_norm.dtype).view(1, 1, 3, 1, 1)
    rgb01 = (rgb_norm * std + mean).clamp(0.0, 1.0)
    roi_weight = keep + (1.0 - keep) * soft_masks.to(device=rgb_norm.device, dtype=rgb_norm.dtype)
    overlay01 = (rgb01 * roi_weight).clamp(0.0, 1.0)
    return (overlay01 - mean) / std


def stage2_fusion_inputs_from_soft_masks(
    *,
    dataset: Stage2AlignedDataset,
    batch: Mapping[str, Any],
    mmdet_inputs: torch.Tensor,
    soft_masks: torch.Tensor,
    affine: torch.Tensor,
    stage2_input_mode: str,
    roi_background_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if stage2_input_mode == "bev_soft_mask":
        bev_masks = project_soft_masks_to_bev(dataset, soft_masks, affine)
        b, n = bev_masks.shape[:2]
        identity_affine = torch.eye(3, device=bev_masks.device, dtype=bev_masks.dtype).view(1, 1, 3, 3).expand(b, n, -1, -1)
        return bev_masks, identity_affine, {"soft_masks": soft_masks, "bev_masks": bev_masks}
    if stage2_input_mode == "rgb_mask_overlay":
        if "rgb_inputs" in batch:
            rgb_norm = batch["rgb_inputs"].to(device=mmdet_inputs.device, dtype=mmdet_inputs.dtype, non_blocking=True)
        else:
            rgb_norm = mmdet_bgr255_to_rgb_norm(mmdet_inputs)
        overlay = soft_mask_rgb_overlay(rgb_norm, soft_masks, float(roi_background_weight))
        return overlay, affine, {"soft_masks": soft_masks, "roi_rgb_inputs": overlay}
    raise ValueError(f"Unsupported stage2 input mode: {stage2_input_mode}")


def forward_joint(
    *,
    mask_model: torch.nn.Module,
    fusion_model: MVDeTrShipMask,
    dataset: Stage2AlignedDataset,
    batch: Mapping[str, Any],
    device: torch.device,
    training: bool,
    bev_only: bool = False,
    stage2_input_mode: str = "bev_soft_mask",
    roi_background_weight: float = 0.25,
    return_intermediates: bool = False,
    return_soft_masks: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    mmdet_inputs = batch["mmdet_inputs"].to(device=device, non_blocking=True)
    affine = batch["affine"].to(device=device, non_blocking=True)
    b, n, c, h, w = mmdet_inputs.shape
    flat_inputs = mmdet_inputs.view(b * n, c, h, w)
    if bev_only:
        samples = make_prediction_samples(b, n, h, w)
    else:
        samples = make_data_samples(batch["weak_masks"])
    processed = preprocess_with_mmdet(mask_model, flat_inputs, samples, training=training and not bev_only)
    if training and not bev_only:
        feats = mask_model.extract_feat(processed["inputs"])
        mask_losses = mask_model.panoptic_head.loss(feats, processed["data_samples"])
        mask_loss = sum_loss_dict(mask_losses)
        mask_cls, mask_pred = mask_model.panoptic_head.predict(feats, processed["data_samples"])
        scores = mask_cls.softmax(dim=-1)[..., 0]
        soft_flat = torch.einsum("bq,bqhw->bhw", scores, mask_pred.sigmoid()).clamp(0.0, 1.0)
        cropped = []
        for idx, sample in enumerate(processed["data_samples"]):
            img_h, img_w = sample.metainfo["img_shape"]
            cropped.append(soft_flat[idx, : int(img_h), : int(img_w)])
        soft_flat = torch.stack(cropped, dim=0).unsqueeze(1)
    else:
        mask_loss = torch.zeros((), device=device)
        soft_flat = soft_masks_from_mask2former(mask_model, processed)
    soft_masks = soft_flat.view(b, n, 1, h, w)
    fusion_inputs, fusion_affine, input_intermediates = stage2_fusion_inputs_from_soft_masks(
        dataset=dataset,
        batch=batch,
        mmdet_inputs=mmdet_inputs,
        soft_masks=soft_masks,
        affine=affine,
        stage2_input_mode=stage2_input_mode,
        roi_background_weight=roi_background_weight,
    )
    aux: Dict[str, torch.Tensor] = {"mask2former": mask_loss}
    if return_soft_masks:
        aux["soft_masks"] = soft_masks
    if return_intermediates:
        fusion_outputs = fusion_model.forward_with_intermediates(fusion_inputs, fusion_affine)
        logits = fusion_outputs["logits"]
        aux.update(input_intermediates)
        aux.update(
            {
                "projected_features": fusion_outputs["projected_features"],
                "fused_features": fusion_outputs["fused_features"],
            }
        )
    else:
        logits = fusion_model(fusion_inputs, fusion_affine)
    return logits, aux, fusion_inputs


@torch.no_grad()
def evaluate(
    mask_model: torch.nn.Module,
    fusion_model: MVDeTrShipMask,
    dataset: Stage2AlignedDataset,
    loader: DataLoader,
    device: torch.device,
    *,
    stage2_input_mode: str,
    roi_background_weight: float,
) -> Dict[str, float]:
    mask_model.eval()
    fusion_model.eval()
    rows = []
    for batch in loader:
        target = batch["target"].to(device=device, non_blocking=True)
        logits, _, _ = forward_joint(
            mask_model=mask_model,
            fusion_model=fusion_model,
            dataset=dataset,
            batch=batch,
            device=device,
            training=False,
            bev_only=True,
            stage2_input_mode=stage2_input_mode,
            roi_background_weight=roi_background_weight,
        )
        rows.append(mask_metrics(logits, target))
    return mean_metrics(rows)


def train_one_epoch(
    *,
    mask_model: torch.nn.Module,
    fusion_model: MVDeTrShipMask,
    dataset: Stage2AlignedDataset,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    weight_mask_controller: Optional[WeightMaskController] = None,
) -> float:
    freeze_mask2former = bool(getattr(args, "freeze_mask2former", False))
    if freeze_mask2former:
        mask_model.eval()
    else:
        mask_model.train()
    fusion_model.train()
    total = 0.0
    count = 0
    accumulation_steps = int(args.grad_accumulation_steps)
    if accumulation_steps < 1:
        raise ValueError("--grad-accumulation-steps must be >= 1")
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader, start=1):
        gate_monitor_interval = int(getattr(args, "gate_monitor_interval", 0))
        monitor_gate = (
            str(args.stage2_input_mode) == "rgb_mask_overlay"
            and gate_monitor_interval > 0
            and (batch_idx == 1 or batch_idx % gate_monitor_interval == 0 or batch_idx == len(loader))
        )
        target = batch["target"].to(device=device, non_blocking=True)
        logits, aux_losses, _ = forward_joint(
            mask_model=mask_model,
            fusion_model=fusion_model,
            dataset=dataset,
            batch=batch,
            device=device,
            training=True,
            bev_only=bool(args.bev_only or float(args.mask2former_loss_weight) <= 0.0),
            stage2_input_mode=str(args.stage2_input_mode),
            roi_background_weight=float(args.roi_background_weight),
            return_soft_masks=monitor_gate,
        )
        seg_loss = segmentation_loss(logits, target, float(args.pos_weight), float(args.dice_weight))
        if float(args.boundary_iou_weight) > 0:
            boundary_loss = boundary_iou_loss_with_logits(logits, target, int(args.boundary_kernel_size))
        else:
            boundary_loss = torch.zeros((), device=device)
        bev_loss = seg_loss + float(args.boundary_iou_weight) * boundary_loss
        loss = bev_loss + float(args.mask2former_loss_weight) * aux_losses["mask2former"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={batch_idx}: {float(loss.detach().cpu())}")
        (loss / float(accumulation_steps)).backward()
        should_step = batch_idx % accumulation_steps == 0 or batch_idx == len(loader)
        if should_step:
            if weight_mask_controller is not None:
                weight_mask_controller.mask_gradients()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in list(mask_model.parameters()) + list(fusion_model.parameters())
                        if parameter.requires_grad
                    ],
                    float(args.grad_clip),
                )
            optimizer.step()
            if weight_mask_controller is not None:
                weight_mask_controller.apply()
            optimizer.zero_grad(set_to_none=True)
        batch_size = int(target.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        count += batch_size
        if monitor_gate or (
            int(args.log_interval) > 0
            and (batch_idx % int(args.log_interval) == 0 or batch_idx == len(loader))
        ):
            boundary_log = (
                f" seg={float(seg_loss.detach().cpu()):.5f} "
                f"boundary={float(boundary_loss.detach().cpu()):.5f}"
                if float(args.boundary_iou_weight) > 0
                else ""
            )
            gate_log = ""
            if "soft_masks" in aux_losses:
                gates = aux_losses["soft_masks"].detach().float()
                per_view_max = gates.flatten(start_dim=2).max(dim=2).values.cpu().flatten().tolist()
                gate_log = " gate_max_per_view=[" + ",".join(f"{value:.6e}" for value in per_view_max) + "]"
            print(
                f"epoch={epoch:03d} batch={batch_idx}/{len(loader)} "
                f"loss={float(loss.detach().cpu()):.5f} bev={float(bev_loss.detach().cpu()):.5f} "
                f"m2f={float(aux_losses['mask2former'].detach().cpu()):.5f}{boundary_log}{gate_log}",
                flush=True,
            )
    return total / max(count, 1)


def make_optimizer(mask_model: torch.nn.Module, fusion_model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": mask_model.parameters(), "lr": float(args.lr_mask2former)},
            {"params": fusion_model.parameters(), "lr": float(args.lr_fusion)},
        ],
        weight_decay=float(args.weight_decay),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    mask_model: torch.nn.Module,
    fusion_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Mapping[str, Any],
    val_metrics: Mapping[str, float],
    compression_recipe: Optional[Mapping[str, Any]] = None,
    compression_masks: Optional[Mapping[str, torch.Tensor]] = None,
    compression_report: Optional[Mapping[str, Any]] = None,
    modelopt_state: Optional[Mapping[str, Any]] = None,
    modelopt_base_mask2former: Optional[torch.nn.Module] = None,
    modelopt_base_fusion_model: Optional[torch.nn.Module] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "mask2former": mask_model.state_dict(),
        "fusion_model": fusion_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "fusion_config": dict(cfg),
        "val_metrics": dict(val_metrics),
    }
    if compression_recipe:
        payload["compression_recipe"] = dict(compression_recipe)
    if compression_masks:
        payload["compression_masks"] = {name: mask.detach().cpu() for name, mask in compression_masks.items()}
    if compression_report:
        payload["compression_report"] = dict(compression_report)
    transform = str((compression_recipe or {}).get("transform", ""))
    if transform == MODELOPT_STATE_DICT_TRANSFORM:
        if modelopt_state is None:
            raise ValueError("QAT checkpoint save requires modelopt_state")
        payload["modelopt_state"] = dict(modelopt_state)
        if (modelopt_base_mask2former is None) != (modelopt_base_fusion_model is None):
            raise ValueError("QAT checkpoint has an incomplete serialized base structure")
        if modelopt_base_mask2former is not None:
            payload[MODELOPT_BASE_MASK2FORMER_KEY] = modelopt_base_mask2former
            payload[MODELOPT_BASE_FUSION_KEY] = modelopt_base_fusion_model
    elif transform == "serialized_structured":
        payload["serialized_mask2former"] = mask_model
        payload["serialized_fusion_model"] = fusion_model
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_history_rows(history_path: Path) -> List[Dict[str, str]]:
    if not history_path.exists():
        return []
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_history_rows(history_path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = history_path.with_name(f".{history_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(temporary_path, history_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def history_epoch(row: Mapping[str, Any]) -> Optional[int]:
    try:
        return int(row.get("epoch", ""))
    except (TypeError, ValueError):
        return None


def qualified_named_parameters(mask_model: torch.nn.Module, fusion_model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    result = {f"mask2former.{name}": parameter for name, parameter in mask_model.named_parameters()}
    result.update({f"fusion_model.{name}": parameter for name, parameter in fusion_model.named_parameters()})
    return result


def load_lightweight_initialization(
    checkpoint: Mapping[str, Any],
    mask_model: torch.nn.Module,
    fusion_model: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, Optional[WeightMaskController]]:
    recipe = dict(checkpoint.get("compression_recipe", {}))
    if is_modelopt_state_dict_checkpoint(checkpoint):
        mask_model, fusion_model = restore_modelopt_state_dict_models(
            checkpoint,
            mask_model,
            fusion_model,
            device,
        )
    elif str(recipe.get("transform", "")) == "serialized_structured":
        if "serialized_mask2former" not in checkpoint or "serialized_fusion_model" not in checkpoint:
            raise ValueError("Structured checkpoint is missing serialized model objects")
        mask_model = checkpoint["serialized_mask2former"].to(device)
        fusion_model = checkpoint["serialized_fusion_model"].to(device)
    else:
        apply_compression_recipe({"mask2former": mask_model, "fusion_model": fusion_model}, recipe)
        missing, unexpected = mask_model.load_state_dict(checkpoint["mask2former"], strict=False)
        if missing or unexpected:
            print(
                f"lightweight Mask2Former load: missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        fusion_model.load_state_dict(checkpoint["fusion_model"], strict=True)
    masks = dict(checkpoint.get("compression_masks", {}))
    controller = None
    if masks:
        controller = WeightMaskController(qualified_named_parameters(mask_model, fusion_model), masks)
        controller.apply()
        print(f"activated permanent compression masks: tensors={len(masks)}", flush=True)
    return mask_model, fusion_model, controller


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage-2 training")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    report_msda_backends(
        require_expected=os.environ.get("INDOOR_STAGE2_REQUIRE_EXPECTED_MSDA", "0") == "1"
    )
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configured_fusion_cfg = load_config(args.fusion_config)
    last_path = args.output_dir / "stage2_last_checkpoint.pt"
    initialization_path = last_path if args.resume and last_path.exists() else args.initial_stage2_checkpoint
    initialization_checkpoint = None
    if initialization_path is not None:
        initialization_checkpoint = torch.load(initialization_path, map_location="cpu")
    fusion_cfg = resolve_fusion_config_for_initialization(
        configured_fusion_cfg,
        initialization_checkpoint,
        stage2_input_mode=str(args.stage2_input_mode),
        roi_background_weight=float(args.roi_background_weight),
    )
    dataset_cfg = dict(fusion_cfg.get("dataset", {}))
    if args.mask_dir is not None:
        dataset_cfg["mask_dir"] = str(args.mask_dir)
    if args.target_mask_dir is not None:
        dataset_cfg["target_mask_dir"] = str(args.target_mask_dir)
    elif args.mask_dir is not None:
        dataset_cfg["target_mask_dir"] = str(args.mask_dir)
    fusion_cfg["dataset"] = dataset_cfg
    write_stage2_provenance(args.output_dir, args, fusion_cfg)
    bev_only = bool(args.bev_only or float(args.mask2former_loss_weight) <= 0.0)
    print(
        f"stage2 mode: bev_only={bev_only} use_rgb_cache={bool(args.use_rgb_cache)} "
        f"mask2former_loss_weight={float(args.mask2former_loss_weight)} "
        f"stage2_input_mode={args.stage2_input_mode} "
        f"roi_background_weight={float(args.roi_background_weight):.3f} "
        f"micro_batch={int(args.batch_size)} "
        f"grad_accumulation_steps={int(args.grad_accumulation_steps)} "
        f"effective_batch={int(args.batch_size) * int(args.grad_accumulation_steps)}",
        flush=True,
    )
    train_set = Stage2AlignedDataset(
        "train",
        fusion_cfg,
        max_frames_per_condition=args.max_frames_per_condition,
        bev_only=bev_only,
        use_rgb_cache=bool(args.use_rgb_cache),
    )
    val_set = Stage2AlignedDataset(
        "val",
        fusion_cfg,
        max_frames_per_condition=args.max_frames_per_condition,
        bev_only=True,
        use_rgb_cache=bool(args.use_rgb_cache),
    )
    train_loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": int(args.num_workers),
        "pin_memory": True,
        "collate_fn": collate_stage2,
        "generator": torch.Generator().manual_seed(int(args.seed)),
    }
    if int(args.num_workers) > 0:
        train_loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
                "worker_init_fn": stage2_worker_init,
            }
        )
    train_loader = DataLoader(train_set, **train_loader_kwargs)
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_stage2,
    )
    mask_model = build_detector(args.mmdet_config, args.mask2former_checkpoint).to(device)
    fusion_model = build_stage2_fusion_model(train_set, fusion_cfg, str(args.stage2_input_mode)).to(device)
    weight_mask_controller: Optional[WeightMaskController] = None
    compression_recipe: Dict[str, Any] = {}
    compression_report: Dict[str, Any] = {}
    modelopt_state: Optional[Mapping[str, Any]] = None
    modelopt_base_mask2former: Optional[torch.nn.Module] = None
    modelopt_base_fusion_model: Optional[torch.nn.Module] = None
    if initialization_checkpoint is not None:
        mask_model, fusion_model, weight_mask_controller = load_lightweight_initialization(
            initialization_checkpoint,
            mask_model,
            fusion_model,
            device,
        )
        compression_recipe = dict(initialization_checkpoint.get("compression_recipe", {}))
        compression_report = dict(initialization_checkpoint.get("compression_report", {}))
        if is_modelopt_state_dict_checkpoint(initialization_checkpoint):
            modelopt_state = initialization_checkpoint.get("modelopt_state")
            if modelopt_state is None:
                raise ValueError("QAT initialization is missing modelopt_state")
            modelopt_base_mask2former = initialization_checkpoint.get(MODELOPT_BASE_MASK2FORMER_KEY)
            modelopt_base_fusion_model = initialization_checkpoint.get(MODELOPT_BASE_FUSION_KEY)
            if (modelopt_base_mask2former is None) != (modelopt_base_fusion_model is None):
                raise ValueError("QAT initialization has an incomplete serialized base structure")
        print(f"loaded Stage2 initialization: {initialization_path}", flush=True)
    optimizer = make_optimizer(mask_model, fusion_model, args)
    print(
        "Stage2 optimizer: "
        f"fusion_lr={float(args.lr_fusion):.8g} "
        f"mask2former_lr={float(args.lr_mask2former):.8g} "
        f"weight_decay={float(args.weight_decay):.8g}",
        flush=True,
    )
    history_path = args.output_dir / "stage2_training_history.csv"
    fieldnames = ["epoch", "train_loss", "val_iou", "val_dice", "val_precision", "val_recall"]
    start_epoch = 1
    best_dice = -math.inf
    best_path = args.output_dir / "stage2_best_checkpoint.pt"
    epoch_checkpoint_dir = args.output_dir / "checkpoints"
    if args.resume and last_path.exists():
        ckpt = initialization_checkpoint or torch.load(last_path, map_location="cpu")
        optimizer.load_state_dict(ckpt["optimizer"])
        checkpoint_epoch = int(ckpt.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        best_dice = float(ckpt.get("val_metrics", {}).get("dice", -math.inf))
        if history_path.exists():
            history_rows = read_history_rows(history_path)
            kept_rows = [
                row
                for row in history_rows
                if (history_epoch(row) is not None and int(history_epoch(row)) <= checkpoint_epoch)
            ]
            if len(kept_rows) != len(history_rows):
                write_history_rows(history_path, fieldnames, kept_rows)
                print(
                    f"trimmed stage2 history from {len(history_rows)} to {len(kept_rows)} rows "
                    f"to match checkpoint epoch {checkpoint_epoch}",
                    flush=True,
                )
            for row in kept_rows:
                try:
                    best_dice = max(best_dice, float(row.get("val_dice", -math.inf)))
                except (TypeError, ValueError):
                    continue
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_dice = max(
                best_dice,
                float(best_checkpoint.get("val_metrics", {}).get("dice", -math.inf)),
            )
            del best_checkpoint
        print(
            f"resuming stage2 from checkpoint epoch {checkpoint_epoch}; "
            f"next epoch {start_epoch}",
            flush=True,
        )
    else:
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_loss = train_one_epoch(
            mask_model=mask_model,
            fusion_model=fusion_model,
            dataset=train_set,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
            weight_mask_controller=weight_mask_controller,
        )
        val_metrics = evaluate(
            mask_model,
            fusion_model,
            val_set,
            val_loader,
            device,
            stage2_input_mode=str(args.stage2_input_mode),
            roi_background_weight=float(args.roi_background_weight),
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_iou={val_metrics['iou']:.6f} val_dice={val_metrics['dice']:.6f}",
            flush=True,
        )
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, **{f"val_{key}": val for key, val in val_metrics.items()}})
        save_checkpoint(
            last_path,
            epoch=epoch,
            mask_model=mask_model,
            fusion_model=fusion_model,
            optimizer=optimizer,
            cfg=fusion_cfg,
            val_metrics=val_metrics,
            compression_recipe=compression_recipe,
            compression_masks=(weight_mask_controller.state_dict() if weight_mask_controller is not None else None),
            compression_report=compression_report,
            modelopt_state=modelopt_state,
            modelopt_base_mask2former=modelopt_base_mask2former,
            modelopt_base_fusion_model=modelopt_base_fusion_model,
        )
        if bool(args.save_epoch_checkpoints):
            save_checkpoint(
                epoch_checkpoint_dir / f"stage2_epoch_{epoch:03d}_checkpoint.pt",
                epoch=epoch,
                mask_model=mask_model,
                fusion_model=fusion_model,
                optimizer=optimizer,
                cfg=fusion_cfg,
                val_metrics=val_metrics,
                compression_recipe=compression_recipe,
                compression_masks=(weight_mask_controller.state_dict() if weight_mask_controller is not None else None),
                compression_report=compression_report,
                modelopt_state=modelopt_state,
                modelopt_base_mask2former=modelopt_base_mask2former,
                modelopt_base_fusion_model=modelopt_base_fusion_model,
            )
        if float(val_metrics["dice"]) > best_dice:
            best_dice = float(val_metrics["dice"])
            save_checkpoint(
                best_path,
                epoch=epoch,
                mask_model=mask_model,
                fusion_model=fusion_model,
                optimizer=optimizer,
                cfg=fusion_cfg,
                val_metrics=val_metrics,
                compression_recipe=compression_recipe,
                compression_masks=(weight_mask_controller.state_dict() if weight_mask_controller is not None else None),
                compression_report=compression_report,
                modelopt_state=modelopt_state,
                modelopt_base_mask2former=modelopt_base_mask2former,
                modelopt_base_fusion_model=modelopt_base_fusion_model,
            )


if __name__ == "__main__":
    main()
