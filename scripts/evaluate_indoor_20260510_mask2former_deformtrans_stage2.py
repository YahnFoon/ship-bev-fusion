#!/usr/bin/env python3
"""Evaluate stage-2 Mask2Former + DeformTrans checkpoints and render diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MVDETR_ROOT = PROJECT_ROOT / "MVDeTr"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MVDETR_ROOT))

from scripts.train_indoor_20260510_mask2former_deformtrans_stage2 import (  # noqa: E402
    Stage2AlignedDataset,
    build_stage2_fusion_model,
    collate_stage2,
    make_prediction_samples,
    preprocess_with_mmdet,
    report_msda_backends,
    soft_masks_from_mask2former,
    stage2_fusion_inputs_from_soft_masks,
    stage2_input_settings_from_config,
)
from ship_bev_fusion.config import require_openmmlab_env  # noqa: E402
from ship_bev_fusion.metrics import mask_metrics, mean_metrics  # noqa: E402
from ship_bev_fusion.lightweight import apply_compression_recipe  # noqa: E402
from ship_bev_fusion.mvdetr_dataset import ShipMVDeTrMaskDataset  # noqa: E402
from ship_bev_fusion.mvdetr_model import MVDeTrShipMask  # noqa: E402
from ship_bev_fusion.runtime import (  # noqa: E402
    is_modelopt_state_dict_checkpoint,
    rebind_fusion_projection_geometry,
    restore_modelopt_state_dict_models,
)
from ship_bev_fusion.visualize import save_raw_view_diagnostic_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--mmdet-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--video-dir", default="Video/waterLevel_9/waterlevel_9_synced_conditions")
    parser.add_argument("--mask-dir", default="Video/waterLevel_9/multiview_dino_sam2_masks")
    parser.add_argument("--target-mask-dir", default="Video/waterLevel_9/multiview_dino_sam2_masks")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Locked split manifest to use instead of the checkpoint's legacy split settings.",
    )
    parser.add_argument(
        "--cache-dir",
        default="Video/waterLevel_9/multi_view_fusion_outputs/mvdetr_raw_view_rgb_deform_trans_4cam_aligned_masks_z81p5_512x288_cache/rgb_cache",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-rgb-cache",
        action="store_true",
        help="Decode synchronized RGB videos directly instead of reading the Stage2 RGB cache.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-diagnostics", type=int, default=16)
    parser.add_argument("--best-diagnostics", type=int, default=4)
    parser.add_argument("--worst-condition-count", type=int, default=4)
    parser.add_argument("--run-name", default="best")
    return parser.parse_args()


def build_detector_from_config(config_path: Path) -> torch.nn.Module:
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    if hasattr(model, "data_preprocessor") and hasattr(model.data_preprocessor, "batch_augments"):
        model.data_preprocessor.batch_augments = None
    return model


def build_fusion_config(checkpoint: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    fusion_cfg = dict(checkpoint.get("fusion_config", {}))
    dataset_cfg = dict(fusion_cfg.get("dataset", {}))
    dataset_cfg["root"] = str(args.data_root)
    dataset_cfg["video_dir"] = str(args.video_dir)
    dataset_cfg["mask_dir"] = str(args.mask_dir)
    dataset_cfg["target_mask_dir"] = str(args.target_mask_dir)
    dataset_cfg["cache_dir"] = str(args.cache_dir)
    if args.split_manifest is not None:
        dataset_cfg["split_mode"] = "locked_manifest"
        dataset_cfg["split_manifest"] = str(args.split_manifest)
        dataset_cfg.pop("split_manifest_path", None)
    fusion_cfg["dataset"] = dataset_cfg
    return fusion_cfg


def build_models(
    checkpoint: Mapping[str, Any],
    dataset: Stage2AlignedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, MVDeTrShipMask]:
    mask_model = build_detector_from_config(args.mmdet_config).to(device)
    fusion_cfg = dict(checkpoint.get("fusion_config", {}))
    stage2_input_mode, roi_background_weight = stage2_input_settings_from_config(fusion_cfg)
    print(
        f"stage2 checkpoint input_mode={stage2_input_mode} "
        f"roi_background_weight={roi_background_weight:.3f}",
        flush=True,
    )
    fusion_model = build_stage2_fusion_model(dataset, fusion_cfg, stage2_input_mode).to(device)

    def finalize_fusion_model(model: MVDeTrShipMask) -> MVDeTrShipMask:
        model._stage2_input_mode = stage2_input_mode
        model._stage2_roi_background_weight = roi_background_weight
        model_cfg = dict(fusion_cfg.get("model", {}))
        target_z_mm = float(model_cfg.get("z_mm", getattr(model, "_projection_z_mm", 81.5)))
        rebind_fusion_projection_geometry(model, dataset, target_z_mm)
        print(f"bound stage2 projection dataset and height: z_mm={target_z_mm:.3f}", flush=True)
        return model

    if is_modelopt_state_dict_checkpoint(checkpoint):
        mask_model, fusion_model = restore_modelopt_state_dict_models(
            checkpoint,
            mask_model,
            fusion_model,
            device,
        )
        fusion_model = finalize_fusion_model(fusion_model)
        print("restored ModelOpt structure and component state dicts", flush=True)
        return mask_model, fusion_model
    if "serialized_mask2former" in checkpoint or "serialized_fusion_model" in checkpoint:
        if "serialized_mask2former" not in checkpoint or "serialized_fusion_model" not in checkpoint:
            raise ValueError("Structured checkpoint must contain both serialized model objects")
        mask_model = checkpoint["serialized_mask2former"].to(device)
        fusion_model = checkpoint["serialized_fusion_model"].to(device)
        fusion_model = finalize_fusion_model(fusion_model)
        return mask_model, fusion_model
    compression_recipe = checkpoint.get("compression_recipe")
    if compression_recipe:
        records = apply_compression_recipe(
            {"mask2former": mask_model, "fusion_model": fusion_model},
            compression_recipe,
        )
        print(
            f"rebuilt lightweight structure: transform={compression_recipe.get('transform')} "
            f"replaced_layers={len(records)}",
            flush=True,
        )
    missing, unexpected = mask_model.load_state_dict(checkpoint["mask2former"], strict=False)
    print(f"loaded stage2 mask2former weights: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    fusion_model.load_state_dict(checkpoint["fusion_model"], strict=True)
    return mask_model, finalize_fusion_model(fusion_model)


@torch.no_grad()
def forward_stage2(
    *,
    mask_model: torch.nn.Module,
    fusion_model: MVDeTrShipMask,
    dataset: Stage2AlignedDataset,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    mmdet_inputs = batch["mmdet_inputs"].to(device=device, non_blocking=True)
    affine = batch["affine"].to(device=device, non_blocking=True)
    b, n, c, h, w = mmdet_inputs.shape
    flat_inputs = mmdet_inputs.view(b * n, c, h, w)
    samples = make_prediction_samples(b, n, h, w)
    processed = preprocess_with_mmdet(mask_model, flat_inputs, samples, training=False)
    soft_flat = soft_masks_from_mask2former(mask_model, processed)
    soft_masks = soft_flat.view(b, n, 1, h, w)
    stage2_input_mode = str(getattr(fusion_model, "_stage2_input_mode", "bev_soft_mask"))
    roi_background_weight = float(getattr(fusion_model, "_stage2_roi_background_weight", 0.25))
    fusion_inputs, fusion_affine, extras = stage2_fusion_inputs_from_soft_masks(
        dataset=dataset,
        batch=batch,
        mmdet_inputs=mmdet_inputs,
        soft_masks=soft_masks,
        affine=affine,
        stage2_input_mode=stage2_input_mode,
        roi_background_weight=roi_background_weight,
    )
    outputs = fusion_model.forward_with_intermediates(fusion_inputs, fusion_affine)
    outputs["soft_masks"] = soft_masks
    outputs.update(extras)
    return outputs


def evaluate_split(
    *,
    mask_model: torch.nn.Module,
    fusion_model: MVDeTrShipMask,
    dataset: Stage2AlignedDataset,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    mask_model.eval()
    fusion_model.eval()
    frame_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Mapping[str, float]]] = defaultdict(list)
    started_at = time.monotonic()
    total_batches = len(loader)
    log_interval = max(1, min(25, total_batches // 20))
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            target = batch["target"].to(device=device, non_blocking=True)
            outputs = forward_stage2(mask_model=mask_model, fusion_model=fusion_model, dataset=dataset, batch=batch, device=device)
            logits = outputs["logits"]
            for sample_idx in range(logits.shape[0]):
                metrics = mask_metrics(
                    logits[sample_idx : sample_idx + 1],
                    target[sample_idx : sample_idx + 1],
                    threshold=threshold,
                )
                condition = str(batch["condition"][sample_idx])
                frame_index = int(batch["frame_index"][sample_idx].item())
                row = {"condition": condition, "frame_index": frame_index, **metrics}
                frame_rows.append(row)
                grouped[condition].append(metrics)
            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
                elapsed = time.monotonic() - started_at
                batches_per_second = batch_index / max(elapsed, 1e-6)
                eta_seconds = (total_batches - batch_index) / max(batches_per_second, 1e-6)
                print(
                    f"evaluation progress: {batch_index}/{total_batches} batches "
                    f"({100.0 * batch_index / max(total_batches, 1):.1f}%) "
                    f"frames={len(frame_rows)}/{len(dataset)} "
                    f"rate={batches_per_second:.3f} batch/s "
                    f"eta={eta_seconds / 60.0:.1f} min",
                    flush=True,
                )
    condition_rows: List[Dict[str, Any]] = []
    for condition, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        summary = mean_metrics(rows)
        condition_rows.append({"condition": condition, "num_frames": len(rows), **summary})
    aggregate = mean_metrics(condition_rows)
    condition_rows.append({"condition": "aggregate", "num_frames": sum(int(row["num_frames"]) for row in condition_rows), **aggregate})
    return frame_rows, condition_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_diagnostic_frames(
    frame_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    *,
    max_worst: int,
    max_best: int,
    worst_condition_count: int,
) -> List[Tuple[str, int, str]]:
    selected: List[Tuple[str, int, str]] = []
    sorted_frames = sorted(frame_rows, key=lambda row: (float(row["dice"]), str(row["condition"]), int(row["frame_index"])))
    selected.extend((str(row["condition"]), int(row["frame_index"]), f"overall_worst_{rank:02d}") for rank, row in enumerate(sorted_frames[:max_worst], start=1))
    if max_best > 0:
        best_frames = sorted(frame_rows, key=lambda row: (float(row["dice"]), str(row["condition"]), int(row["frame_index"])), reverse=True)
        selected.extend((str(row["condition"]), int(row["frame_index"]), f"overall_best_{rank:02d}") for rank, row in enumerate(best_frames[:max_best], start=1))
    ranked_conditions = [row for row in condition_rows if str(row["condition"]) != "aggregate"]
    ranked_conditions.sort(key=lambda row: float(row["dice"]))
    for condition_row in ranked_conditions[:worst_condition_count]:
        condition = str(condition_row["condition"])
        cond_frames = [row for row in frame_rows if str(row["condition"]) == condition]
        if not cond_frames:
            continue
        worst_frame = min(cond_frames, key=lambda row: float(row["dice"]))
        best_frame = max(cond_frames, key=lambda row: float(row["dice"]))
        selected.append((condition, int(worst_frame["frame_index"]), f"condition_{int(condition):02d}_worst"))
        selected.append((condition, int(best_frame["frame_index"]), f"condition_{int(condition):02d}_best"))
    deduped: List[Tuple[str, int, str]] = []
    seen = set()
    for condition, frame_index, label in selected:
        key = (condition, frame_index)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((condition, frame_index, label))
    return deduped


@torch.no_grad()
def render_diagnostics(
    *,
    mask_model: torch.nn.Module,
    fusion_model: MVDeTrShipMask,
    eval_dataset: Stage2AlignedDataset,
    diagnostic_dataset: ShipMVDeTrMaskDataset,
    selections: Sequence[Tuple[str, int, str]],
    output_dir: Path,
    device: torch.device,
    threshold: float,
) -> List[Dict[str, Any]]:
    eval_lookup = {(str(record.condition), int(record.frame_index)): idx for idx, record in enumerate(eval_dataset.records)}
    record_lookup = {(str(record.condition), int(record.frame_index)): record for record in diagnostic_dataset.records}
    rows: List[Dict[str, Any]] = []
    for condition, frame_index, label in selections:
        eval_index = eval_lookup.get((condition, frame_index))
        record = record_lookup.get((condition, frame_index))
        if eval_index is None or record is None:
            continue
        batch = collate_stage2([eval_dataset[eval_index]])
        target = batch["target"].to(device=device)
        outputs = forward_stage2(mask_model=mask_model, fusion_model=fusion_model, dataset=eval_dataset, batch=batch, device=device)
        logits = outputs["logits"]
        metrics = mask_metrics(logits, target, threshold=threshold)
        raw_views = diagnostic_dataset.build_raw_view_diagnostics(record)
        pred_mask = (torch.sigmoid(logits[0, 0]).cpu().numpy() >= float(threshold)).astype("float32")
        target_mask = batch["target"][0, 0].numpy()
        condition_dir = output_dir / "frames" / f"condition_{int(condition):02d}"
        figure_path = condition_dir / f"{label}_frame_{frame_index:05d}.png"
        save_raw_view_diagnostic_figure(
            figure_path,
            raw_rgb=raw_views["rgb"],
            raw_masks=raw_views["mask"],
            roi_rgb=raw_views["roi_rgb"],
            projected_features=outputs["projected_features"][0].detach().float().cpu().numpy(),
            aligned_features=outputs["aligned_features"][0].detach().float().cpu().numpy(),
            fused_features=outputs["fused_features"][0].detach().float().cpu().numpy(),
            pred_mask=pred_mask,
            target_mask=target_mask,
            title=f"Stage2 {condition} frame {frame_index}",
            metrics=metrics,
            camera_ids=diagnostic_dataset.input_camera_ids,
            input_mode="rgb->mask2former->bev->deform_trans",
        )
        rows.append({"condition": condition, "frame_index": frame_index, "label": label, "path": str(figure_path), **metrics})
    return rows


def main() -> None:
    args = parse_args()
    require_openmmlab_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    report_msda_backends(
        require_expected=os.environ.get("INDOOR_STAGE2_REQUIRE_EXPECTED_MSDA", "0") == "1"
    )
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu")
    fusion_cfg = build_fusion_config(checkpoint, args)
    eval_dataset = Stage2AlignedDataset(
        args.split,
        fusion_cfg,
        bev_only=True,
        use_rgb_cache=not bool(args.no_rgb_cache),
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_stage2,
    )
    diagnostic_cfg = dict(fusion_cfg["dataset"])
    diagnostic_cfg["use_cache"] = False
    diagnostic_cfg["input_mode"] = "image_rgb"
    diagnostic_dataset = ShipMVDeTrMaskDataset(split=args.split, **diagnostic_cfg)

    mask_model, fusion_model = build_models(checkpoint, eval_dataset, args, device)
    frame_rows, condition_rows = evaluate_split(
        mask_model=mask_model,
        fusion_model=fusion_model,
        dataset=eval_dataset,
        loader=loader,
        device=device,
        threshold=float(args.threshold),
    )

    run_root = args.output_dir / f"evaluation_{args.run_name}" / args.split
    write_csv(run_root / "frame_metrics.csv", frame_rows, ["condition", "frame_index", "iou", "dice", "precision", "recall"])
    write_csv(run_root / "condition_metrics.csv", condition_rows, ["condition", "num_frames", "iou", "dice", "precision", "recall"])
    aggregate_row = next((row for row in condition_rows if str(row["condition"]) == "aggregate"), None)
    if aggregate_row is not None:
        (run_root / "aggregate_metrics.json").write_text(json.dumps(aggregate_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    selections = select_diagnostic_frames(
        frame_rows,
        condition_rows,
        max_worst=int(args.max_diagnostics),
        max_best=int(args.best_diagnostics),
        worst_condition_count=int(args.worst_condition_count),
    )
    diagnostic_rows = render_diagnostics(
        mask_model=mask_model,
        fusion_model=fusion_model,
        eval_dataset=eval_dataset,
        diagnostic_dataset=diagnostic_dataset,
        selections=selections,
        output_dir=run_root,
        device=device,
        threshold=float(args.threshold),
    )
    write_csv(run_root / "diagnostic_manifest.csv", diagnostic_rows, ["condition", "frame_index", "label", "path", "iou", "dice", "precision", "recall"])
    print(f"Wrote evaluation under: {run_root}", flush=True)


if __name__ == "__main__":
    main()
