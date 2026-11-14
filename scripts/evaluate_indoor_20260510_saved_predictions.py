#!/usr/bin/env python3
'Evaluate saved Indoor_20260510 BEV predictions with paper metrics.'

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import ndimage
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MVDETR_ROOT = PROJECT_ROOT / "MVDeTr"
for path in (PROJECT_ROOT, MVDETR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ship_bev_fusion.mvdetr_dataset import ShipMVDeTrBase  # noqa: E402


DEFAULT_MAIN_ROOT = Path("data/indoor/paper_clean_rerun_v1/main_bev")
DEFAULT_DATA_ROOT = Path("data/indoor")
METRIC_KEYS = (
    "iou",
    "dice",
    "precision",
    "recall",
    "boundary_iou_20mm",
    "contour_f1_20mm",
    "hd95_mm",
    "assd_mm",
    "chamfer_distance_mm",
    "centroid_error_mm",
    "clearance_abs_error_mm",
    "clearance_abs_radial_error_mm",
    "clearance_radial_error_mm",
    "clearance_abs_tangential_error_mm",
    "heading_error_deg",
)
COCO_KEYS = (
    "coco_mAP",
    "coco_mAP_50",
    "coco_mAP_75",
    "coco_AP_s",
    "coco_AP_m",
    "coco_AP_l",
)


@dataclass(frozen=True)
class MethodRoot:
    name: str
    root: Path
    target_root: Optional[Path] = None


@dataclass
class SamplePrediction:
    method: str
    split: str
    condition: str
    frame_index: int
    pred_mask: np.ndarray
    target_mask: np.ndarray
    score: float
    logits: Optional[np.ndarray] = None


@dataclass
class FrameState:
    method: str
    split: str
    condition: str
    frame_index: int
    valid: bool
    center_xy: Optional[np.ndarray]
    polygon: Optional[Any]
    heading_deg: float = math.nan
    clearance_mm: float = math.nan


@dataclass
class PierGeometry:
    polygon: Polygon
    gt_world_polygon: Polygon
    left_edge_center: np.ndarray
    outside_normal: np.ndarray
    left_edge_axis: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method spec NAME=PATH. If omitted, known method directories under --main-root are used.",
    )
    parser.add_argument(
        "--target-root",
        action="append",
        default=[],
        help=(
            "Optional target/evaluation root override NAME=PATH. "
            "Predictions are read from --method PATH, but targets and evaluation geometry are read from PATH."
        ),
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=("train", "val", "test"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance-mm", type=float, default=20.0)
    parser.add_argument("--world-unit", choices=("m", "cm", "mm"), default="mm")
    parser.add_argument("--pier-length-mm", type=float, default=300.0)
    parser.add_argument("--pier-width-mm", type=float, default=80.0)
    parser.add_argument("--pier-outside-gap-mm", type=float, default=100.0)
    parser.add_argument("--heading-window", type=int, default=2)
    parser.add_argument("--heading-min-displacement-mm", type=float, default=20.0)
    parser.add_argument("--max-frames-per-split", type=int, default=None, help="Debug limiter; do not use for final paper tables.")
    return parser.parse_args()


def parse_target_roots(args: argparse.Namespace) -> Dict[str, Path]:
    roots: Dict[str, Path] = {}
    for spec in args.target_root:
        if "=" not in spec:
            raise ValueError(f"--target-root must be NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        roots[name.strip()] = Path(path).expanduser()
    return roots


def parse_method_roots(args: argparse.Namespace) -> List[MethodRoot]:
    target_roots = parse_target_roots(args)
    if args.method:
        roots = []
        for spec in args.method:
            if "=" not in spec:
                raise ValueError(f"--method must be NAME=PATH, got: {spec}")
            name, path = spec.split("=", 1)
            method_name = name.strip()
            roots.append(MethodRoot(method_name, Path(path).expanduser(), target_roots.get(method_name)))
        return roots
    candidates = [
        MethodRoot("geometry", args.main_root / "geometry_baselines"),
        MethodRoot("direct_bev_mask_deformtrans", args.main_root / "direct_bev_mask_deformtrans"),
        MethodRoot("direct_bev_rgb_deformtrans", args.main_root / "direct_bev_rgb_deformtrans"),
        MethodRoot("mvdetr_mask_deformtrans", args.main_root / "mvdetr_mask_deformtrans"),
        MethodRoot("mvdetr_rgb_first_clean", args.main_root / "mvdetr_rgb_deformtrans_cachev20260604"),
        MethodRoot("mvdetr_rgb_rerun_seed_default", args.main_root / "mvdetr_rgb_deformtrans_cachev20260604_rerun_seed_default"),
        MethodRoot("ours_m2f_deformtrans_lr1e4", args.main_root / "ours_mask2former_deformtrans/stage2_m2f_lr1e4_nomargin_cachev20260604"),
        MethodRoot("ours_m2f_deformtrans_lr2e4", args.main_root / "ours_mask2former_deformtrans/stage2_m2f_lr2e4_nomargin_cachev20260604"),
        MethodRoot("mask2former_deformtrans_lr2e7", args.main_root / "ours_mask2former_deformtrans/stage2_m2f_lr2e7_nomargin_cachev20260604"),
        MethodRoot("mask2former_deformtrans_lr2e5", args.main_root / "ours_mask2former_deformtrans/stage2_m2f_lr2e5_nomargin_cachev20260604"),
    ]
    return [
        MethodRoot(item.name, item.root, target_roots.get(item.name))
        for item in candidates
        if item.root.exists()
    ]


def condition_key(value: str) -> Tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10**9, text)


def csv_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf"
        return value
    return value


def json_value(value: Any) -> Any:
    value = csv_value(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = [key for key in preferred if key in keys] + sorted(keys - set(preferred))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_value) + "\n", encoding="utf-8")


def load_mapping(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_config(method_root: Path) -> Dict[str, Any]:
    for name in ("run_config_resolved.json", "run_config_resolved.yaml", "run_config_resolved.yml"):
        path = method_root / name
        if path.exists():
            return load_mapping(path)
    raise FileNotFoundError(f"No run_config_resolved.json/yaml under {method_root}")


def evaluation_config_root(root: MethodRoot) -> Path:
    if root.target_root is not None:
        try:
            find_run_config(root.target_root)
        except FileNotFoundError:
            pass
        else:
            return root.target_root
    return root.root


def dataset_cfg_from_run_config(method_root: Path, data_root_override: Path) -> Dict[str, Any]:
    cfg = find_run_config(method_root)
    dataset_cfg = dict(cfg.get("dataset", {}))
    if not dataset_cfg and "config" in cfg:
        config_path = Path(str(cfg["config"]))
        if config_path.exists():
            dataset_cfg = dict(load_mapping(config_path).get("dataset", {}))
    configured_root = Path(str(dataset_cfg.get("root", ""))) if dataset_cfg.get("root") else None
    if data_root_override.exists():
        dataset_cfg["root"] = str(data_root_override)
    elif configured_root is not None and configured_root.exists():
        dataset_cfg["root"] = str(configured_root)
    else:
        raise FileNotFoundError(f"Cannot resolve dataset root for {method_root}")
    return dataset_cfg


def build_base(dataset_cfg: Mapping[str, Any]) -> ShipMVDeTrBase:
    return ShipMVDeTrBase(
        root=Path(str(dataset_cfg["root"])),
        input_camera_ids=tuple(str(v) for v in dataset_cfg.get("input_camera_ids", ("221", "222", "223", "224"))),
        target_camera_id=str(dataset_cfg.get("target_camera_id", "GT")),
        camera_parameters_dir=str(dataset_cfg.get("camera_parameters_dir", ".")),
        target_homography_image_size=tuple(int(v) for v in dataset_cfg.get("target_homography_image_size", (1920, 1080))),
        bev_size=tuple(int(v) for v in dataset_cfg.get("bev_size", (512, 288))),
        plane_z_mm=float(dataset_cfg.get("plane_z_mm", 81.5)),
        world_to_bev_homography_file=dataset_cfg.get("world_to_bev_homography_file"),
        target_homography_direction=str(dataset_cfg.get("target_homography_direction", "image_to_bev")),
        target_homography_space=str(dataset_cfg.get("target_homography_space", "world")),
        target_homography_output_size=tuple(int(v) for v in dataset_cfg["target_homography_output_size"])
        if dataset_cfg.get("target_homography_output_size")
        else None,
        assume_world_xy_is_target_image=bool(dataset_cfg.get("assume_world_xy_is_target_image", False)),
        target_world_bounds_margin=float(dataset_cfg.get("target_world_bounds_margin", 0.0)),
        bev_projection_mode=str(dataset_cfg.get("bev_projection_mode", "target_homography")),
    )


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def logit_threshold(threshold: float) -> float:
    if abs(float(threshold) - 0.5) < 1e-12:
        return 0.0
    threshold = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    return float(np.log(threshold / (1.0 - threshold)))


def score_from_logits(logits: np.ndarray, pred_mask: np.ndarray) -> float:
    prob = sigmoid_np(logits)
    mask = np.asarray(pred_mask).astype(bool)
    if mask.any():
        return float(prob[mask].mean())
    return float(prob.max(initial=0.0))


def decode_rle(segmentation: Mapping[str, Any], height: int, width: int) -> np.ndarray:
    rle = dict(segmentation)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    mask = mask_utils.decode(rle)
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def frame_index_from_coco_image(image: Mapping[str, Any]) -> int:
    value = image.get("frame_index")
    if value is not None:
        return int(value)
    stem = Path(str(image.get("file_name", "0"))).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(np.asarray(mask).astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    rle["size"] = [int(v) for v in rle["size"]]
    return rle


def coco_masks_by_key(path: Path) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in data.get("images", [])}
    masks: Dict[Tuple[str, int], np.ndarray] = {}
    scores: Dict[Tuple[str, int], float] = {}
    for ann in data.get("annotations", []):
        image = images_by_id[int(ann["image_id"])]
        condition = str(image.get("condition", ""))
        frame_index = frame_index_from_coco_image(image)
        key = (condition, frame_index)
        height, width = int(image["height"]), int(image["width"])
        mask = decode_rle(ann["segmentation"], height, width)
        if key in masks:
            masks[key] |= mask
            scores[key] = max(scores[key], float(ann.get("score", 1.0)))
        else:
            masks[key] = mask
            scores[key] = float(ann.get("score", 1.0))
    return masks, scores


def coco_annotations_by_key(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    images_by_id = {int(image["id"]): image for image in data.get("images", [])}
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for ann in data.get("annotations", []):
        image = images_by_id[int(ann["image_id"])]
        condition = str(image.get("condition", ""))
        frame_index = frame_index_from_coco_image(image)
        key = (condition, frame_index)
        score = float(ann.get("score", 1.0))
        current = out.get(key)
        if current is None or score >= float(current.get("score", -math.inf)):
            out[key] = {
                "segmentation": ann["segmentation"],
                "height": int(image["height"]),
                "width": int(image["width"]),
                "score": score,
            }
    return out


def target_cache_dir(root: Path, split: str) -> Path:
    candidates = (
        root / "targets" / split / "target_masks_uint8",
        root / split / "target_masks_uint8",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing target cache; checked: {rendered}")


def load_targets(method_root: Path, split: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    target_root = target_cache_dir(method_root, split)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(target_root.glob("*.npz"), key=lambda p: condition_key(p.stem)):
        payload = np.load(path)
        frames = np.asarray(payload["frame_indices"], dtype=np.int32)
        masks = np.asarray(payload["masks"]).astype(bool)
        out[path.stem] = (frames, masks)
    return out


def iter_learned_predictions(
    method: str,
    method_root: Path,
    target_root: Path,
    split: str,
    threshold: float,
) -> Iterable[SamplePrediction]:
    logits_root = method_root / "predictions" / split / "pred_logits_fp16"
    if not logits_root.exists():
        return
    targets = load_targets(target_root, split)
    thresh = logit_threshold(threshold)
    for condition, (target_frames, target_masks) in targets.items():
        logits_path = logits_root / f"{condition}.npz"
        if not logits_path.exists():
            continue
        payload = np.load(logits_path)
        pred_frames = np.asarray(payload["frame_indices"], dtype=np.int32)
        logits = np.asarray(payload["logits"], dtype=np.float32)
        by_frame = {int(frame): idx for idx, frame in enumerate(pred_frames.tolist())}
        for target_idx, frame in enumerate(target_frames.tolist()):
            pred_idx = by_frame.get(int(frame))
            if pred_idx is None:
                continue
            logit = logits[pred_idx]
            pred = logit >= thresh
            yield SamplePrediction(
                method=method,
                split=split,
                condition=str(condition),
                frame_index=int(frame),
                pred_mask=pred,
                target_mask=target_masks[target_idx].astype(bool),
                score=score_from_logits(logit, pred),
                logits=logit,
            )


def iter_geometry_predictions(
    method_root: Path,
    target_root: Path,
    split: str,
    method_prefix: str = "",
) -> Iterable[SamplePrediction]:
    pred_root = method_root / "predictions" / split
    if not pred_root.exists():
        return
    targets = load_targets(target_root, split)
    coco_paths = sorted(pred_root.glob("*_masks_coco.json"))
    for coco_path in coco_paths:
        base_method = coco_path.name[: -len("_masks_coco.json")]
        method = f"{method_prefix}_{base_method}" if method_prefix else base_method
        anns_by_key = coco_annotations_by_key(coco_path)
        for condition, (frames, target_masks) in targets.items():
            for idx, frame in enumerate(frames.tolist()):
                key = (str(condition), int(frame))
                ann = anns_by_key.get(key)
                if ann is None:
                    pred = np.zeros_like(target_masks[idx], dtype=bool)
                    score = 1.0
                else:
                    pred = decode_rle(ann["segmentation"], int(ann["height"]), int(ann["width"]))
                    score = float(ann.get("score", 1.0))
                yield SamplePrediction(
                    method=method,
                    split=split,
                    condition=str(condition),
                    frame_index=int(frame),
                    pred_mask=pred.astype(bool),
                    target_mask=target_masks[idx].astype(bool),
                    score=score,
                )


def iter_method_predictions(root: MethodRoot, split: str, threshold: float) -> Iterable[SamplePrediction]:
    target_root = root.target_root or root.root
    logits_root = root.root / "predictions" / split / "pred_logits_fp16"
    if logits_root.exists():
        yield from iter_learned_predictions(root.name, root.root, target_root, split, threshold)
    else:
        geometry_prefix = "" if root.name == "geometry" else root.name
        yield from iter_geometry_predictions(root.root, target_root, split, geometry_prefix)


def basic_mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    eps = 1e-6
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2.0 * tp / (2.0 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def surface_mask(mask: np.ndarray) -> np.ndarray:
    mask_b = np.asarray(mask).astype(bool)
    if not mask_b.any():
        return mask_b
    eroded = ndimage.binary_erosion(mask_b, structure=ndimage.generate_binary_structure(mask_b.ndim, 1), border_value=0)
    return np.logical_and(mask_b, ~eroded)


def boundary_band(mask: np.ndarray, tolerance_px: int) -> np.ndarray:
    edge = surface_mask(mask)
    if not edge.any() or tolerance_px <= 0:
        return edge
    return ndimage.binary_dilation(edge, iterations=int(tolerance_px))


def boundary_iou(pred: np.ndarray, gt: np.ndarray, tolerance_px: int) -> float:
    pred_band = boundary_band(pred, tolerance_px)
    gt_band = boundary_band(gt, tolerance_px)
    union = np.logical_or(pred_band, gt_band).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_band, gt_band).sum() / union)


def contour_f1(pred: np.ndarray, gt: np.ndarray, sampling_mm: Tuple[float, float], tolerance_mm: float) -> float:
    pred_edge = surface_mask(pred)
    gt_edge = surface_mask(gt)
    if not pred_edge.any() and not gt_edge.any():
        return 1.0
    if not pred_edge.any() or not gt_edge.any():
        return 0.0
    dist_to_gt = ndimage.distance_transform_edt(~gt_edge, sampling=sampling_mm)
    dist_to_pred = ndimage.distance_transform_edt(~pred_edge, sampling=sampling_mm)
    precision = float(np.logical_and(pred_edge, dist_to_gt <= tolerance_mm).sum()) / max(float(pred_edge.sum()), 1.0)
    recall = float(np.logical_and(gt_edge, dist_to_pred <= tolerance_mm).sum()) / max(float(gt_edge.sum()), 1.0)
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def surface_distances_mm(pred: np.ndarray, gt: np.ndarray, sampling_mm: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    pred_surf = surface_mask(pred)
    gt_surf = surface_mask(gt)
    pred_area = bool(np.asarray(pred).astype(bool).any())
    gt_area = bool(np.asarray(gt).astype(bool).any())
    if not pred_area and not gt_area:
        empty = np.asarray([0.0], dtype=np.float64)
        return empty, empty
    if not pred_area or not gt_area:
        inf = np.asarray([math.inf], dtype=np.float64)
        return inf, inf
    dist_to_gt = ndimage.distance_transform_edt(~gt_surf, sampling=sampling_mm)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surf, sampling=sampling_mm)
    return dist_to_gt[pred_surf].astype(np.float64), dist_to_pred[gt_surf].astype(np.float64)


def boundary_distance_metrics(pred: np.ndarray, gt: np.ndarray, sampling_mm: Tuple[float, float], tolerance_mm: float) -> Dict[str, float]:
    pred_to_gt, gt_to_pred = surface_distances_mm(pred, gt, sampling_mm)
    all_dist = np.concatenate([pred_to_gt, gt_to_pred])
    if any(math.isinf(float(v)) for v in all_dist):
        assd = hd95 = chamfer = math.inf
    else:
        assd = float(all_dist.mean())
        hd95 = float(np.percentile(all_dist, 95.0))
        chamfer = float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0)
    tolerance_px = int(math.ceil(float(tolerance_mm) / max(min(sampling_mm), 1e-9)))
    return {
        "boundary_iou_20mm": boundary_iou(pred, gt, tolerance_px),
        "contour_f1_20mm": contour_f1(pred, gt, sampling_mm, tolerance_mm),
        "hd95_mm": hd95,
        "assd_mm": assd,
        "chamfer_distance_mm": chamfer,
    }


def transform_points(h: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    homo = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts), dtype=np.float64)])
    out = (np.asarray(h, dtype=np.float64) @ homo.T).T
    denom = out[:, 2:3]
    denom[np.abs(denom) < 1e-12] = np.nan
    return out[:, :2] / denom


def world_unit_to_mm(unit: str) -> float:
    return {"mm": 1.0, "cm": 10.0, "m": 1000.0}[str(unit)]


def scaled_worldcoord_mat(world_from_grid: np.ndarray, world_unit: str) -> np.ndarray:
    mat = np.asarray(world_from_grid, dtype=np.float64).copy()
    mat[:2, :] *= world_unit_to_mm(world_unit)
    return mat


def centroid_world(mask: np.ndarray, world_from_grid: np.ndarray) -> Tuple[bool, float, float, float, float]:
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return False, math.nan, math.nan, math.nan, math.nan
    x_px = float(xs.mean())
    y_px = float(ys.mean())
    world = transform_points(world_from_grid, np.array([[x_px, y_px]], dtype=np.float64))[0]
    return True, x_px, y_px, float(world[0]), float(world[1])


def geometry_parts(geom: Any) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [polygon for child in geom.geoms for polygon in geometry_parts(child)]
    return []


def clean_polygon(geom: Any) -> Optional[Any]:
    if geom is None or geom.is_empty:
        return None
    geom = make_valid(geom)
    parts = list(geometry_parts(geom))
    if not parts:
        return None
    merged = unary_union(parts)
    if merged.is_empty:
        return None
    return merged


def mask_to_world_polygon(mask: np.ndarray, world_from_grid: np.ndarray, min_area_px: float = 1.0) -> Optional[Any]:
    mask_u8 = np.asarray(mask).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[Polygon] = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < float(min_area_px):
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        world_points = transform_points(world_from_grid, points)
        polygon = clean_polygon(Polygon(world_points))
        if polygon is not None:
            polygons.extend(list(geometry_parts(polygon)))
    if not polygons:
        return None
    return clean_polygon(unary_union(polygons))


def unit_vector(vec: np.ndarray, name: str) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError(f"Degenerate vector for {name}: {vec}")
    return vec / norm


def compute_pier_geometry(
    base: ShipMVDeTrBase,
    dataset_cfg: Mapping[str, Any],
    args: argparse.Namespace,
    world_from_grid_mm: np.ndarray,
) -> PierGeometry:
    image_w, image_h = tuple(int(v) for v in dataset_cfg.get("target_homography_image_size", (1920, 1080)))
    target_image_to_world = world_from_grid_mm @ base.target_image_to_bev_mat
    corners_img = np.asarray(
        [[0.0, 0.0], [image_w - 1.0, 0.0], [image_w - 1.0, image_h - 1.0], [0.0, image_h - 1.0]],
        dtype=np.float64,
    )
    corners_world = transform_points(target_image_to_world, corners_img)
    top_left, _, _, bottom_left = corners_world
    left_edge_center = (top_left + bottom_left) / 2.0
    world_center = corners_world.mean(axis=0)
    outside_normal = unit_vector(left_edge_center - world_center, "GT left-edge outside normal")
    left_edge_axis = unit_vector(bottom_left - top_left, "GT left-edge axis")
    left_edge_axis = left_edge_axis - outside_normal * float(np.dot(left_edge_axis, outside_normal))
    left_edge_axis = unit_vector(left_edge_axis, "GT left-edge tangent")

    pier_center = left_edge_center + outside_normal * (float(args.pier_length_mm) / 2.0 + float(args.pier_outside_gap_mm))
    long_axis = outside_normal
    short_axis = left_edge_axis
    corners = [
        pier_center + long_axis * args.pier_length_mm / 2.0 + short_axis * args.pier_width_mm / 2.0,
        pier_center + long_axis * args.pier_length_mm / 2.0 - short_axis * args.pier_width_mm / 2.0,
        pier_center - long_axis * args.pier_length_mm / 2.0 - short_axis * args.pier_width_mm / 2.0,
        pier_center - long_axis * args.pier_length_mm / 2.0 + short_axis * args.pier_width_mm / 2.0,
    ]
    return PierGeometry(
        polygon=Polygon(corners),
        gt_world_polygon=Polygon(corners_world),
        left_edge_center=left_edge_center,
        outside_normal=outside_normal,
        left_edge_axis=left_edge_axis,
    )


def nearest_clearance_vector(ship_geom: Any, pier_geom: Any) -> Optional[np.ndarray]:
    if ship_geom is None or pier_geom is None or ship_geom.is_empty or pier_geom.is_empty:
        return None
    try:
        ship_point, pier_point = nearest_points(ship_geom, pier_geom)
    except Exception:
        return None
    vec = np.array([ship_point.x - pier_point.x, ship_point.y - pier_point.y], dtype=np.float64)
    return vec if np.isfinite(vec).all() else None


def radial_tangential_error(pred_geom: Any, gt_geom: Any, pier_geom: Any) -> Tuple[float, float]:
    pred_vec = nearest_clearance_vector(pred_geom, pier_geom)
    gt_vec = nearest_clearance_vector(gt_geom, pier_geom)
    if pred_vec is None or gt_vec is None:
        return math.nan, math.nan
    gt_norm = float(np.linalg.norm(gt_vec))
    if not math.isfinite(gt_norm) or gt_norm <= 1e-9:
        return math.nan, math.nan
    radial_axis = gt_vec / gt_norm
    tangential_axis = np.array([-radial_axis[1], radial_axis[0]], dtype=np.float64)
    err_vec = pred_vec - gt_vec
    return float(np.dot(err_vec, radial_axis)), float(np.dot(err_vec, tangential_axis))


def wrap_to_180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def heading_error(pred_deg: float, gt_deg: float) -> float:
    if not math.isfinite(float(pred_deg)) or not math.isfinite(float(gt_deg)):
        return math.nan
    return abs(wrap_to_180(float(pred_deg) - float(gt_deg)))


def motion_heading(centers: Sequence[Optional[np.ndarray]], idx: int, min_displacement_mm: float, window: int) -> float:
    current = centers[idx]
    if current is None or idx <= 0:
        return math.nan
    best_vec = None
    best_norm = 0.0
    for radius in range(1, max(int(window), 1) + 1):
        prev_idx = idx - radius
        if prev_idx < 0:
            break
        prev = centers[prev_idx]
        if prev is None:
            continue
        vec = current - prev
        norm = float(np.linalg.norm(vec))
        if norm > best_norm:
            best_vec = vec
            best_norm = norm
        if norm >= float(min_displacement_mm):
            break
    if best_vec is None or best_norm <= 1e-9:
        return math.nan
    return wrap_to_180(math.degrees(math.atan2(float(best_vec[1]), float(best_vec[0]))))


def trajectory_equivalent_distance(gt_points: np.ndarray, pred_points: np.ndarray) -> Dict[str, float]:
    if len(gt_points) < 2 or len(pred_points) < 2:
        return {"trajectory_valid": 0.0, "trajectory_truth_length_mm": math.nan, "trajectory_area_mm2": math.nan, "equivalent_distance_mm": math.nan}
    truth_length = float(np.linalg.norm(gt_points[1:] - gt_points[:-1], axis=1).sum())
    if truth_length <= 0:
        return {"trajectory_valid": 0.0, "trajectory_truth_length_mm": truth_length, "trajectory_area_mm2": math.nan, "equivalent_distance_mm": math.nan}
    try:
        polygon = make_valid(Polygon(gt_points.tolist() + pred_points.tolist()[::-1]))
        area = float(polygon.area)
        return {
            "trajectory_valid": 1.0,
            "trajectory_truth_length_mm": truth_length,
            "trajectory_area_mm2": area,
            "equivalent_distance_mm": area / truth_length,
        }
    except Exception:
        return {"trajectory_valid": 0.0, "trajectory_truth_length_mm": truth_length, "trajectory_area_mm2": math.nan, "equivalent_distance_mm": math.nan}


def make_coco_gt_and_pred(samples: Sequence[SamplePrediction]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    images: List[Dict[str, Any]] = []
    gt_annotations: List[Dict[str, Any]] = []
    pred_annotations: List[Dict[str, Any]] = []
    gt_ann_id = 1
    pred_ann_id = 1
    for image_id, sample in enumerate(samples, start=1):
        height, width = sample.target_mask.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": f"{sample.condition}/{sample.frame_index:06d}.png",
                "width": int(width),
                "height": int(height),
                "condition": sample.condition,
                "frame_index": int(sample.frame_index),
            }
        )
        gt_mask = sample.target_mask.astype(np.uint8)
        if gt_mask.any():
            rle = encode_binary_mask(gt_mask)
            gt_annotations.append(
                {
                    "id": gt_ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                    "bbox": [float(v) for v in mask_utils.toBbox(rle)],
                    "iscrowd": 0,
                }
            )
            gt_ann_id += 1
        pred_mask = sample.pred_mask.astype(np.uint8)
        if pred_mask.any():
            rle = encode_binary_mask(pred_mask)
            pred_annotations.append(
                {
                    "id": pred_ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                    "bbox": [float(v) for v in mask_utils.toBbox(rle)],
                    "score": float(sample.score),
                }
            )
            pred_ann_id += 1
    gt_coco = {"info": {}, "images": images, "annotations": gt_annotations, "categories": [{"id": 1, "name": "ship"}]}
    return gt_coco, pred_annotations


def coco_ap_metrics(samples: Sequence[SamplePrediction]) -> Dict[str, float]:
    gt_coco, pred_annotations = make_coco_gt_and_pred(samples)
    if not gt_coco["annotations"]:
        return {key: math.nan for key in COCO_KEYS}
    if not pred_annotations:
        return {key: 0.0 for key in COCO_KEYS}
    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(pred_annotations)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    def stat(index: int) -> float:
        value = float(stats[index])
        return math.nan if value < 0 else value

    return {
        "coco_mAP": stat(0),
        "coco_mAP_50": stat(1),
        "coco_mAP_75": stat(2),
        "coco_AP_s": stat(3),
        "coco_AP_m": stat(4),
        "coco_AP_l": stat(5),
    }


def coco_ap_metrics_from_payload(payload: Mapping[str, Any]) -> Dict[str, float]:
    gt_coco = {
        "info": {},
        "images": list(payload.get("images", [])),
        "annotations": list(payload.get("gt_annotations", [])),
        "categories": [{"id": 1, "name": "ship"}],
    }
    pred_annotations = list(payload.get("pred_annotations", []))
    if not gt_coco["annotations"]:
        return {key: math.nan for key in COCO_KEYS}
    if not pred_annotations:
        return {key: 0.0 for key in COCO_KEYS}
    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(pred_annotations)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats

    def stat(index: int) -> float:
        value = float(stats[index])
        return math.nan if value < 0 else value

    return {
        "coco_mAP": stat(0),
        "coco_mAP_50": stat(1),
        "coco_mAP_75": stat(2),
        "coco_AP_s": stat(3),
        "coco_AP_m": stat(4),
        "coco_AP_l": stat(5),
    }


def append_ap_payload(
    payloads: MutableMapping[Tuple[str, str], Dict[str, Any]],
    sample: SamplePrediction,
) -> None:
    key = (sample.method, sample.split)
    payload = payloads.setdefault(
        key,
        {"images": [], "gt_annotations": [], "pred_annotations": [], "next_image_id": 1, "next_gt_ann_id": 1, "next_pred_ann_id": 1},
    )
    image_id = int(payload["next_image_id"])
    payload["next_image_id"] = image_id + 1
    height, width = sample.target_mask.shape[:2]
    payload["images"].append(
        {
            "id": image_id,
            "file_name": f"{sample.condition}/{sample.frame_index:06d}.png",
            "width": int(width),
            "height": int(height),
            "condition": sample.condition,
            "frame_index": int(sample.frame_index),
        }
    )
    gt_mask = sample.target_mask.astype(np.uint8)
    if gt_mask.any():
        rle = encode_binary_mask(gt_mask)
        ann_id = int(payload["next_gt_ann_id"])
        payload["next_gt_ann_id"] = ann_id + 1
        payload["gt_annotations"].append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": float(mask_utils.area(rle)),
                "bbox": [float(v) for v in mask_utils.toBbox(rle)],
                "iscrowd": 0,
            }
        )
    pred_mask = sample.pred_mask.astype(np.uint8)
    if pred_mask.any():
        rle = encode_binary_mask(pred_mask)
        ann_id = int(payload["next_pred_ann_id"])
        payload["next_pred_ann_id"] = ann_id + 1
        payload["pred_annotations"].append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": float(mask_utils.area(rle)),
                "bbox": [float(v) for v in mask_utils.toBbox(rle)],
                "score": float(sample.score),
            }
        )


def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            vals.append(value)
    return np.asarray(vals, dtype=np.float64)


def mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else math.nan


def p95_or_nan(values: np.ndarray) -> float:
    return float(np.percentile(values, 95.0)) if len(values) else math.nan


def evaluate_samples(
    samples: Iterable[SamplePrediction],
    *,
    base: ShipMVDeTrBase,
    dataset_cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    world_from_grid = scaled_worldcoord_mat(base.worldcoord_from_worldgrid_mat, args.world_unit)
    row_step_mm = float(np.linalg.norm(world_from_grid[:2, 1]))
    col_step_mm = float(np.linalg.norm(world_from_grid[:2, 0]))
    sampling_mm = (row_step_mm, col_step_mm)
    pier = compute_pier_geometry(base, dataset_cfg, args, world_from_grid)

    frame_rows: List[Dict[str, Any]] = []
    states: List[FrameState] = []
    gt_states: List[FrameState] = []
    ap_payloads: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for sample in samples:
        append_ap_payload(ap_payloads, sample)
        basic = basic_mask_metrics(sample.pred_mask, sample.target_mask)
        boundary = boundary_distance_metrics(sample.pred_mask, sample.target_mask, sampling_mm, args.boundary_tolerance_mm)
        pred_valid, pred_x_px, pred_y_px, pred_x_mm, pred_y_mm = centroid_world(sample.pred_mask, world_from_grid)
        gt_valid, gt_x_px, gt_y_px, gt_x_mm, gt_y_mm = centroid_world(sample.target_mask, world_from_grid)
        centroid_error = math.hypot(pred_x_mm - gt_x_mm, pred_y_mm - gt_y_mm) if pred_valid and gt_valid else math.inf if gt_valid else math.nan
        pred_polygon = mask_to_world_polygon(sample.pred_mask, world_from_grid)
        gt_polygon = mask_to_world_polygon(sample.target_mask, world_from_grid)
        pred_clearance = float(pred_polygon.distance(pier.polygon)) if pred_polygon is not None else math.nan
        gt_clearance = float(gt_polygon.distance(pier.polygon)) if gt_polygon is not None else math.nan
        clearance_error = pred_clearance - gt_clearance if math.isfinite(pred_clearance) and math.isfinite(gt_clearance) else math.nan
        radial_error, tangential_error = radial_tangential_error(pred_polygon, gt_polygon, pier.polygon)
        row = {
            "method": sample.method,
            "split": sample.split,
            "condition": sample.condition,
            "frame_index": sample.frame_index,
            "score": float(sample.score),
            **basic,
            **boundary,
            "pred_centroid_x_px": pred_x_px,
            "pred_centroid_y_px": pred_y_px,
            "gt_centroid_x_px": gt_x_px,
            "gt_centroid_y_px": gt_y_px,
            "pred_world_x_mm": pred_x_mm,
            "pred_world_y_mm": pred_y_mm,
            "gt_world_x_mm": gt_x_mm,
            "gt_world_y_mm": gt_y_mm,
            "centroid_error_mm": centroid_error,
            "pred_valid": float(pred_valid),
            "gt_valid": float(gt_valid),
            "pred_clearance_mm": pred_clearance,
            "gt_clearance_mm": gt_clearance,
            "clearance_error_mm": clearance_error,
            "clearance_abs_error_mm": abs(clearance_error) if math.isfinite(clearance_error) else math.nan,
            "clearance_radial_error_mm": radial_error,
            "clearance_abs_radial_error_mm": abs(radial_error) if math.isfinite(radial_error) else math.nan,
            "clearance_tangential_error_mm": tangential_error,
            "clearance_abs_tangential_error_mm": abs(tangential_error) if math.isfinite(tangential_error) else math.nan,
        }
        frame_rows.append(row)
        states.append(
            FrameState(
                method=sample.method,
                split=sample.split,
                condition=sample.condition,
                frame_index=sample.frame_index,
                valid=pred_valid,
                center_xy=np.asarray([pred_x_mm, pred_y_mm], dtype=np.float64) if pred_valid else None,
                polygon=None,
                clearance_mm=pred_clearance,
            )
        )
        gt_states.append(
            FrameState(
                method="GT",
                split=sample.split,
                condition=sample.condition,
                frame_index=sample.frame_index,
                valid=gt_valid,
                center_xy=np.asarray([gt_x_mm, gt_y_mm], dtype=np.float64) if gt_valid else None,
                polygon=None,
                clearance_mm=gt_clearance,
            )
        )

    add_headings_and_heading_errors(frame_rows, states, gt_states, args)
    condition_rows = build_condition_rows(frame_rows, states, gt_states)
    method_rows = build_method_rows(frame_rows, condition_rows, ap_payloads)
    return frame_rows, condition_rows, method_rows


def add_headings_and_heading_errors(
    frame_rows: List[Dict[str, Any]],
    states: List[FrameState],
    gt_states: List[FrameState],
    args: argparse.Namespace,
) -> None:
    unique_gt_by_key = {
        (state.split, state.condition, int(state.frame_index)): state
        for state in gt_states
    }
    unique_gt_states = list(unique_gt_by_key.values())
    all_states = states + unique_gt_states
    grouped: Dict[Tuple[str, str, str], List[FrameState]] = defaultdict(list)
    for state in all_states:
        grouped[(state.method, state.split, state.condition)].append(state)
    for group in grouped.values():
        group.sort(key=lambda item: int(item.frame_index))
        centers = [state.center_xy if state.valid else None for state in group]
        for idx, state in enumerate(group):
            state.heading_deg = motion_heading(centers, idx, args.heading_min_displacement_mm, args.heading_window)
    pred_by_key = {(state.method, state.split, state.condition, state.frame_index): state for state in states}
    gt_by_key = {(state.split, state.condition, state.frame_index): state for state in unique_gt_states}
    for row in frame_rows:
        pred = pred_by_key[(str(row["method"]), str(row["split"]), str(row["condition"]), int(row["frame_index"]))]
        gt = gt_by_key[(str(row["split"]), str(row["condition"]), int(row["frame_index"]))]
        row["pred_heading_deg"] = pred.heading_deg
        row["gt_heading_deg"] = gt.heading_deg
        row["heading_error_deg"] = heading_error(pred.heading_deg, gt.heading_deg)
        row["heading_valid"] = float(math.isfinite(float(row["heading_error_deg"])))


def build_condition_rows(
    frame_rows: Sequence[Mapping[str, Any]],
    states: Sequence[FrameState],
    gt_states: Sequence[FrameState],
) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_key[(str(row["method"]), str(row["split"]), str(row["condition"]))].append(row)
    pred_points: Dict[Tuple[str, str, str], Dict[int, np.ndarray]] = defaultdict(dict)
    gt_points: Dict[Tuple[str, str], Dict[int, np.ndarray]] = defaultdict(dict)
    for state in states:
        if state.valid and state.center_xy is not None:
            pred_points[(state.method, state.split, state.condition)][int(state.frame_index)] = state.center_xy
    for state in gt_states:
        if state.valid and state.center_xy is not None:
            gt_points[(state.split, state.condition)][int(state.frame_index)] = state.center_xy
    rows: List[Dict[str, Any]] = []
    for key, group in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1], condition_key(item[0][2]))):
        method, split, condition = key
        out: Dict[str, Any] = {"method": method, "split": split, "condition": condition, "num_frames": len(group)}
        for metric in METRIC_KEYS:
            vals = finite_values(group, metric)
            out[f"mean_{metric}"] = mean_or_nan(vals)
            out[f"p95_{metric}"] = p95_or_nan(vals)
        common_frames = sorted(set(pred_points[key]) & set(gt_points[(split, condition)]))
        pred_arr = np.asarray([pred_points[key][frame] for frame in common_frames], dtype=np.float64)
        gt_arr = np.asarray([gt_points[(split, condition)][frame] for frame in common_frames], dtype=np.float64)
        out["valid_trajectory_frames"] = len(common_frames)
        out.update(trajectory_equivalent_distance(gt_arr, pred_arr))
        rows.append(out)
    return rows


def build_method_rows(
    frame_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    ap_payloads: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    frame_by_key: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    cond_by_key: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        frame_by_key[(str(row["method"]), str(row["split"]))].append(row)
    for row in condition_rows:
        cond_by_key[(str(row["method"]), str(row["split"]))].append(row)
    rows: List[Dict[str, Any]] = []
    for key, frames in sorted(frame_by_key.items()):
        method, split = key
        conditions = cond_by_key.get(key, [])
        out: Dict[str, Any] = {
            "method": method,
            "split": split,
            "num_frames": len(frames),
            "num_conditions": len(conditions),
        }
        for metric in METRIC_KEYS:
            frame_vals = finite_values(frames, metric)
            cond_mean_vals = finite_values(conditions, f"mean_{metric}")
            out[f"frame_mean_{metric}"] = mean_or_nan(frame_vals)
            out[f"frame_p95_{metric}"] = p95_or_nan(frame_vals)
            out[f"macro_mean_{metric}"] = mean_or_nan(cond_mean_vals)
            out[f"macro_p95_mean_{metric}"] = p95_or_nan(cond_mean_vals)
        equiv = finite_values(conditions, "equivalent_distance_mm")
        out["macro_mean_equivalent_distance_mm"] = mean_or_nan(equiv)
        out["macro_p95_equivalent_distance_mm"] = p95_or_nan(equiv)
        out.update(coco_ap_metrics_from_payload(ap_payloads.get(key, {})))
        out["mAP"] = out.get("coco_mAP", math.nan)
        out["mAP@0.5"] = out.get("coco_mAP_50", math.nan)
        out["mAP@0.5:0.95"] = out.get("coco_mAP", math.nan)
        out["AP50"] = out.get("coco_mAP_50", math.nan)
        out["AP75"] = out.get("coco_mAP_75", math.nan)
        rows.append(out)
    return rows


def evaluate_method_root(root: MethodRoot, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    evaluation_root = evaluation_config_root(root)
    dataset_cfg = dataset_cfg_from_run_config(evaluation_root, args.data_root)
    base = build_base(dataset_cfg)

    def split_samples(split: str) -> Iterable[SamplePrediction]:
        count = 0
        iterator = iter_method_predictions(root, split, args.threshold)
        if args.max_frames_per_split is not None:
            iterator = itertools.islice(iterator, int(args.max_frames_per_split))
        for sample in iterator:
            count += 1
            yield sample
        print(f"[evaluate] {root.name} split={split} samples={count}", flush=True)

    samples = itertools.chain.from_iterable(split_samples(split) for split in args.splits)
    return evaluate_samples(samples, base=base, dataset_cfg=dataset_cfg, args=args)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.main_root / "evaluation_metrics"
    method_roots = parse_method_roots(args)
    all_frame_rows: List[Dict[str, Any]] = []
    all_condition_rows: List[Dict[str, Any]] = []
    all_method_rows: List[Dict[str, Any]] = []
    run_metadata = {
        "main_root": str(args.main_root),
        "output_dir": str(out_dir),
        "data_root": str(args.data_root),
        "splits": list(args.splits),
        "threshold": float(args.threshold),
        "world_unit": str(args.world_unit),
        "boundary_tolerance_mm": float(args.boundary_tolerance_mm),
        "pier_length_mm": float(args.pier_length_mm),
        "pier_width_mm": float(args.pier_width_mm),
        "pier_outside_gap_mm": float(args.pier_outside_gap_mm),
        "heading_window": int(args.heading_window),
        "heading_min_displacement_mm": float(args.heading_min_displacement_mm),
        "methods": [
            {
                "name": item.name,
                "root": str(item.root),
                "target_root": str(item.target_root) if item.target_root is not None else None,
            }
            for item in method_roots
        ],
    }
    for method_root in method_roots:
        frame_rows, condition_rows, method_rows = evaluate_method_root(method_root, args)
        all_frame_rows.extend(frame_rows)
        all_condition_rows.extend(condition_rows)
        all_method_rows.extend(method_rows)

    preferred_frame = [
        "method",
        "split",
        "condition",
        "frame_index",
        "iou",
        "dice",
        "precision",
        "recall",
        "boundary_iou_20mm",
        "contour_f1_20mm",
        "hd95_mm",
        "assd_mm",
        "chamfer_distance_mm",
        "centroid_error_mm",
        "clearance_abs_error_mm",
        "clearance_abs_radial_error_mm",
        "clearance_radial_error_mm",
        "clearance_abs_tangential_error_mm",
        "heading_error_deg",
        "score",
    ]
    preferred_condition = ["method", "split", "condition", "num_frames", "valid_trajectory_frames", "equivalent_distance_mm"]
    preferred_method = [
        "method",
        "split",
        "num_frames",
        "num_conditions",
        "macro_mean_iou",
        "macro_mean_dice",
        "macro_mean_precision",
        "macro_mean_recall",
        "coco_mAP",
        "coco_mAP_50",
        "coco_mAP_75",
        "coco_AP_s",
        "coco_AP_m",
        "coco_AP_l",
        "macro_mean_boundary_iou_20mm",
        "macro_mean_contour_f1_20mm",
        "macro_mean_hd95_mm",
        "macro_mean_assd_mm",
        "macro_mean_chamfer_distance_mm",
        "macro_mean_centroid_error_mm",
        "frame_p95_centroid_error_mm",
        "macro_mean_equivalent_distance_mm",
        "macro_p95_equivalent_distance_mm",
        "macro_mean_clearance_abs_error_mm",
        "frame_p95_clearance_abs_error_mm",
        "macro_mean_clearance_abs_radial_error_mm",
        "frame_p95_clearance_abs_radial_error_mm",
        "macro_mean_clearance_radial_error_mm",
        "frame_p95_clearance_radial_error_mm",
        "macro_mean_clearance_abs_tangential_error_mm",
        "frame_p95_clearance_abs_tangential_error_mm",
        "macro_mean_heading_error_deg",
        "frame_p95_heading_error_deg",
    ]
    write_csv(out_dir / "frame_level_metrics.csv", all_frame_rows, preferred_frame)
    write_csv(out_dir / "condition_level_metrics.csv", all_condition_rows, preferred_condition)
    write_csv(out_dir / "method_summary_metrics.csv", all_method_rows, preferred_method)
    write_json(out_dir / "run_config_resolved.json", run_metadata)
    print(f"Wrote evaluation metrics under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
