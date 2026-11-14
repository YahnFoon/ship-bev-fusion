'MVDeTr-style dataset for ship BEV mask segmentation.'

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .calibration import (
    build_old_batch_concat_homographies,
    build_homographies,
    load_camera_parameters,
    load_matrix_file,
    plane_world_to_image_homography,
    undistort_mask,
    warp_mask_to_bev,
)
from .dataset import (
    CocoMaskStore,
    FrameRecord,
    PerConditionCocoMaskStore,
    binarize,
    build_records_from_split_manifest,
    build_records_from_sources,
    canonical_camera_id,
    condition_dir_name,
    condition_sort_key,
    condition_token,
    find_coco_mask_json,
    flatten_split_records,
    maybe_write_split_manifest,
    resolve_under_root,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def slice_cache_by_camera_ids(
    *,
    inputs: np.ndarray,
    affine: np.ndarray,
    input_camera_ids: Sequence[str],
    cache_camera_ids: Optional[Sequence[str]],
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select cached camera tensors by explicit camera ID order."""
    expected_camera_ids = tuple(canonical_camera_id(camera_id) for camera_id in input_camera_ids)
    if inputs.ndim < 1 or affine.ndim < 1:
        raise ValueError(f"Cache arrays must include a camera dimension: {cache_path}")
    if cache_camera_ids is None:
        expected = len(expected_camera_ids)
        if inputs.shape[0] != expected or affine.shape[0] != expected:
            raise ValueError(
                f"Cache camera dimension mismatch for {cache_path}: "
                f"inputs={inputs.shape[0]} affine={affine.shape[0]} expected={expected}. "
                "Set dataset.cache_camera_ids when reusing a superset cache."
            )
        return inputs, affine

    cached_camera_ids = tuple(canonical_camera_id(camera_id) for camera_id in cache_camera_ids)
    if len(set(cached_camera_ids)) != len(cached_camera_ids):
        raise ValueError(f"dataset.cache_camera_ids contains duplicates: {cached_camera_ids}")
    if inputs.shape[0] != len(cached_camera_ids) or affine.shape[0] != len(cached_camera_ids):
        raise ValueError(
            f"Cache camera dimension mismatch for {cache_path}: "
            f"inputs={inputs.shape[0]} affine={affine.shape[0]} "
            f"cache_camera_ids={len(cached_camera_ids)}"
        )
    cache_index = {camera_id: idx for idx, camera_id in enumerate(cached_camera_ids)}
    missing = [camera_id for camera_id in expected_camera_ids if camera_id not in cache_index]
    if missing:
        raise ValueError(
            f"dataset.input_camera_ids requests cameras not present in cache_camera_ids: "
            f"missing={missing} cache_camera_ids={cached_camera_ids}"
        )
    indices = [cache_index[camera_id] for camera_id in expected_camera_ids]
    return inputs[indices], affine[indices]


def build_records_from_cache_dir(
    cache_dir: Path,
    *,
    split_ratios: Tuple[float, float, float],
    max_frames_per_condition: Optional[int],
    split_mode: str,
    split_seed: int,
    split_manifest: Optional[str | Path] = None,
    root: Optional[Path] = None,
) -> Dict[str, List[FrameRecord]]:
    condition_frames: Dict[str, List[int]] = {}
    for condition_dir in sorted(Path(cache_dir).iterdir(), key=lambda path: condition_sort_key(path.name)):
        if not condition_dir.is_dir():
            continue
        frames = sorted(int(path.stem) for path in condition_dir.glob("*.npz") if path.stem.isdigit())
        if frames:
            condition_frames[condition_dir.name] = frames
    if not condition_frames:
        raise ValueError(f"No cache records found under {cache_dir}")
    if split_manifest:
        manifest_path = resolve_under_root(Path(root) if root is not None else Path("."), split_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        return build_records_from_split_manifest(manifest_path, condition_frames, max_frames_per_condition)
    if max_frames_per_condition is not None:
        condition_frames = {
            condition: frames[: int(max_frames_per_condition)]
            for condition, frames in condition_frames.items()
        }

    train_ratio, val_ratio, test_ratio = split_ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    splits: Dict[str, List[FrameRecord]] = {"train": [], "val": [], "test": []}

    if split_mode == "condition_random":
        conditions = sorted(condition_frames, key=condition_sort_key)
        rng = random.Random(int(split_seed))
        rng.shuffle(conditions)
        n = len(conditions)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        else:
            n_train = max(1, n - 1)
            n_val = 0
        split_conditions = {
            "train": conditions[:n_train],
            "val": conditions[n_train : n_train + n_val],
            "test": conditions[n_train + n_val :],
        }
        for split_name, split_condition_names in split_conditions.items():
            for condition in sorted(split_condition_names, key=condition_sort_key):
                splits[split_name].extend(
                    FrameRecord(str(condition), int(frame), split_name)
                    for frame in condition_frames[condition]
                )
        return splits

    if split_mode in {"frame", "frame_within_condition"}:
        records = [
            FrameRecord(str(condition), int(frame), "")
            for condition in sorted(condition_frames, key=condition_sort_key)
            for frame in condition_frames[condition]
        ]
        n = len(records)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        for split_name, split_records in {
            "train": records[:n_train],
            "val": records[n_train : n_train + n_val],
            "test": records[n_train + n_val :],
        }.items():
            splits[split_name] = [
                FrameRecord(record.condition, record.frame_index, split_name)
                for record in split_records
            ]
        return splits

    raise ValueError(f"Unsupported split_mode for cache records: {split_mode}")


def find_video_path(video_root: Path, camera_id: str, condition: Optional[str] = None) -> Path:
    camera_id = canonical_camera_id(camera_id)
    if condition is None:
        candidates = [
            video_root / f"camera_{camera_id}_synced_clean.mp4",
            video_root / f"camera_{camera_id}.mp4",
            video_root / f"{camera_id}_synced_clean.mp4",
            video_root / f"{camera_id}.mp4",
        ]
    else:
        token = condition_token(condition)
        cond_dir = condition_dir_name(condition)
        candidates = [
            video_root / camera_id / f"camera_{camera_id}_{token}.mp4",
            video_root / camera_id / f"camera_{camera_id}_{cond_dir}.mp4",
            video_root / camera_id / f"{camera_id}_{token}.mp4",
            video_root / camera_id / cond_dir / f"camera_{camera_id}_{token}.mp4",
            video_root / cond_dir / camera_id / f"camera_{camera_id}_{token}.mp4",
        ]
    for path in candidates:
        if path.exists():
            return path
    detail = f" condition {condition}" if condition is not None else ""
    raise FileNotFoundError(f"Could not find video for camera {camera_id}{detail} in {video_root}")


class VideoFrameStore:
    """Lazy random-access reader for synchronized RGB videos."""

    def __init__(
        self,
        video_root: Path,
        camera_ids: Sequence[str],
        per_condition: bool = False,
        max_open_captures: int = 8,
    ) -> None:
        self.video_root = Path(video_root)
        self.per_condition = bool(per_condition)
        self.paths = {} if self.per_condition else {str(camera_id): find_video_path(self.video_root, str(camera_id)) for camera_id in camera_ids}
        self.max_open_captures = max(1, int(max_open_captures))
        self._caps: "OrderedDict[Tuple[str, Optional[str]], cv2.VideoCapture]" = OrderedDict()

    def __getstate__(self) -> Dict[str, object]:
        return {
            "video_root": self.video_root,
            "per_condition": self.per_condition,
            "paths": self.paths,
            "max_open_captures": self.max_open_captures,
            "_caps": OrderedDict(),
        }

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def _capture(self, camera_id: str, condition: Optional[str] = None) -> cv2.VideoCapture:
        camera_id = canonical_camera_id(camera_id)
        key = (camera_id, condition_token(condition) if condition is not None else None)
        cap = self._caps.get(key)
        if cap is None:
            path = find_video_path(self.video_root, camera_id, condition) if self.per_condition else self.paths[camera_id]
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {path}")
            self._caps[key] = cap
            while len(self._caps) > self.max_open_captures:
                old_key, old_cap = self._caps.popitem(last=False)
                if old_key != key:
                    old_cap.release()
        return cap

    def read_rgb(self, camera_id: str, frame_index: int, condition: Optional[str] = None) -> np.ndarray:
        cap = self._capture(camera_id, condition)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise EOFError(f"Failed to read frame {frame_index} for camera={camera_id}, condition={condition}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def normalize_rgb_chw(
    image_rgb: np.ndarray,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    mean_np = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return ((image - mean_np) / std_np).transpose(2, 0, 1).astype(np.float32)


def roi_rgb_from_mask(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    input_size: Tuple[int, int],
    threshold: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_w, input_h = tuple(int(v) for v in input_size)
    orig_h, orig_w = frame_rgb.shape[:2]
    interpolation = cv2.INTER_AREA if input_w < orig_w or input_h < orig_h else cv2.INTER_LINEAR
    resized_rgb = cv2.resize(frame_rgb, (input_w, input_h), interpolation=interpolation)
    resized_mask = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
    mask_binary = binarize(resized_mask, threshold)[..., None].astype(np.float32)
    mean_np = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    image = resized_rgb.astype(np.float32) / 255.0
    roi = image * mask_binary + mean_np * (1.0 - mask_binary)
    std_np = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = ((roi - mean_np) / std_np).transpose(2, 0, 1).astype(np.float32)
    roi_uint8 = (roi * 255.0).clip(0, 255).astype(np.uint8)
    return normalized, resized_rgb, mask_binary[..., 0], roi_uint8


@dataclass
class ShipMVDeTrBase:
    root: Path
    input_camera_ids: Sequence[str]
    target_camera_id: str
    camera_parameters_dir: str
    target_homography_image_size: Tuple[int, int]
    bev_size: Tuple[int, int]
    plane_z_mm: float
    world_to_bev_homography_file: Optional[str] = None
    target_homography_direction: str = "image_to_bev"
    target_homography_space: str = "world"
    target_homography_output_size: Optional[Tuple[int, int]] = None
    assume_world_xy_is_target_image: bool = False
    target_world_bounds_margin: float = 0.0
    bev_projection_mode: str = "target_homography"
    old_bev_canvas: Tuple[int, int] = (2600, 1600)
    old_world_origin: Tuple[float, float] = (350.0, 250.0)
    old_bev_resolution: float = 1.0
    old_homography_invert: bool = True

    def __post_init__(self) -> None:
        self.__name__ = "ShipPool"
        self.root = Path(self.root)
        self.num_cam = len(self.input_camera_ids)
        self.indexing = "xy"
        self.worldcoord_unit = 1.0
        self.world_indexing_from_xy_mat = np.eye(3, dtype=np.float64)
        self.img_xy_from_xy_mat = np.eye(3, dtype=np.float64)
        self.target_camera_id = canonical_camera_id(self.target_camera_id)
        self.camera_parameters_root = resolve_under_root(self.root, self.camera_parameters_dir)
        if self.world_to_bev_homography_file:
            world_to_bev_path = Path(self.world_to_bev_homography_file)
            if not world_to_bev_path.is_absolute():
                world_to_bev_path = self.root / world_to_bev_path
            self.world_to_bev_mat = load_matrix_file(world_to_bev_path)
            self.target_image_to_bev_mat = np.eye(3, dtype=np.float64)
        elif self.bev_projection_mode == "old_batch_concat":
            bundle = build_old_batch_concat_homographies(
                self.camera_parameters_root,
                self.input_camera_ids,
                self.target_camera_id,
                plane_z=float(self.plane_z_mm),
                bev_size=self.bev_size,
                old_bev_canvas=tuple(int(v) for v in self.old_bev_canvas),
                old_world_origin=tuple(float(v) for v in self.old_world_origin),
                old_bev_resolution=float(self.old_bev_resolution),
                old_homography_invert=bool(self.old_homography_invert),
            )
            self.world_to_bev_mat = bundle.world_to_bev
            self.target_image_to_bev_mat = bundle.target_image_to_bev
        else:
            if self.bev_projection_mode != "target_homography":
                raise ValueError("bev_projection_mode must be 'target_homography' or 'old_batch_concat'")
            bundle = build_homographies(
                self.camera_parameters_root,
                self.input_camera_ids,
                self.target_camera_id,
                plane_z=float(self.plane_z_mm),
                bev_size=self.bev_size,
                target_homography_direction=self.target_homography_direction,
                target_homography_output_size=self.target_homography_output_size,
                target_homography_space=self.target_homography_space,
                target_homography_image_size=self.target_homography_image_size,
                target_world_bounds_margin=float(self.target_world_bounds_margin),
                assume_world_xy_is_target_image=self.assume_world_xy_is_target_image,
            )
            self.world_to_bev_mat = bundle.world_to_bev
            self.target_image_to_bev_mat = bundle.target_image_to_bev
        self.worldcoord_from_worldgrid_mat = np.linalg.inv(self.world_to_bev_mat)
        bev_w, bev_h = self.bev_size
        self.worldgrid_shape = [bev_h, bev_w]
        self.intrinsic_matrices = []
        self.extrinsic_matrices = []
        self.camera_parameters = {}
        self.input_image_to_bev_mats = {}
        for camera_id in self.input_camera_ids:
            params = load_camera_parameters(self.camera_parameters_root, camera_id)
            self.camera_parameters[str(camera_id)] = params
            self.intrinsic_matrices.append(params.intrinsic)
            self.extrinsic_matrices.append(np.column_stack([params.rotation, params.translation.reshape(3, 1)]))
            h_world_to_img = plane_world_to_image_homography(params, float(self.plane_z_mm))
            h_img_to_bev = self.world_to_bev_mat @ np.linalg.inv(h_world_to_img)
            if abs(h_img_to_bev[2, 2]) > 1e-12:
                h_img_to_bev = h_img_to_bev / h_img_to_bev[2, 2]
            self.input_image_to_bev_mats[str(camera_id)] = h_img_to_bev.astype(np.float64)


class ShipMVDeTrMaskDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        input_camera_ids: Sequence[str] = ("222", "223", "224"),
        target_camera_id: str = "221",
        video_dir: Optional[str] = None,
        mask_dir: str = "mask",
        camera_parameters_dir: str = "camera_parameters",
        intervals_csv: str = "synced_clean_intervals.csv",
        plane_z_mm: float = 93.0,
        bev_size: Tuple[int, int] = (512, 288),
        input_size: Tuple[int, int] = (512, 288),
        target_homography_image_size: Tuple[int, int] = (2560, 1440),
        world_to_bev_homography_file: Optional[str] = None,
        target_homography_direction: str = "image_to_bev",
        target_homography_space: str = "world",
        target_homography_output_size: Optional[Tuple[int, int]] = None,
        assume_world_xy_is_target_image: bool = False,
        target_world_bounds_margin: float = 0.0,
        bev_projection_mode: str = "target_homography",
        old_bev_canvas: Tuple[int, int] = (2600, 1600),
        old_world_origin: Tuple[float, float] = (350.0, 250.0),
        old_bev_resolution: float = 1.0,
        old_homography_invert: bool = True,
        target_mode: str = "camera_homography",
        input_mode: str = "image_mask",
        bev_channel_stack: bool = False,
        bev_mask_dir: str = "bev_mask",
        bev_mask_json: Optional[str] = None,
        per_condition: bool = False,
        target_mask_dir: Optional[str] = None,
        split_mode: str = "frame",
        split_seed: int = 20260510,
        split_manifest: Optional[str] = None,
        write_split_manifest: Optional[str] = None,
        split_manifest_path: Optional[str] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        threshold: int = 127,
        max_frames_per_condition: Optional[int] = None,
        cache_dir: Optional[str] = None,
        cache_camera_ids: Optional[Sequence[str]] = None,
        use_cache: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.input_camera_ids = tuple(str(c) for c in input_camera_ids)
        self.target_camera_id = canonical_camera_id(target_camera_id)
        self.per_condition = bool(per_condition)
        self.video_root = resolve_under_root(self.root, video_dir) if video_dir else None
        self.mask_root = resolve_under_root(self.root, mask_dir)
        self.target_mask_root = resolve_under_root(self.root, target_mask_dir) if target_mask_dir else self.mask_root
        self.target_mode = str(target_mode)
        if self.target_mode not in {"camera_homography", "direct_bev_mask"}:
            raise ValueError("target_mode must be 'camera_homography' or 'direct_bev_mask'")
        self.input_mode = str(input_mode)
        if self.input_mode not in {"image_mask", "preprojected_bev_mask", "preprojected_bev_rgb", "roi_rgb", "image_rgb"}:
            raise ValueError(
                "input_mode must be 'image_mask', 'preprojected_bev_mask', "
                "'preprojected_bev_rgb', 'roi_rgb', or 'image_rgb'"
            )
        if self.input_mode in {"roi_rgb", "image_rgb", "preprojected_bev_rgb"} and self.video_root is None:
            raise ValueError(f"input_mode='{self.input_mode}' requires dataset.video_dir")
        self.bev_channel_stack = bool(bev_channel_stack)
        if self.bev_channel_stack and self.input_mode != "preprojected_bev_mask":
            raise ValueError("bev_channel_stack requires input_mode='preprojected_bev_mask'")
        self.bev_mask_root = resolve_under_root(self.root, bev_mask_dir)
        self.bev_mask_json = bev_mask_json
        self.threshold = int(threshold)
        self.bev_size = tuple(int(v) for v in bev_size)
        self.input_size = tuple(int(v) for v in input_size)
        self.cache_dir = self.root / cache_dir if cache_dir else None
        self.cache_camera_ids = (
            tuple(canonical_camera_id(camera_id) for camera_id in cache_camera_ids)
            if cache_camera_ids is not None
            else None
        )
        self.use_cache = bool(use_cache)
        self.base = ShipMVDeTrBase(
            root=self.root,
            input_camera_ids=self.input_camera_ids,
            target_camera_id=self.target_camera_id,
            camera_parameters_dir=camera_parameters_dir,
            target_homography_image_size=tuple(int(v) for v in target_homography_image_size),
            bev_size=self.bev_size,
            plane_z_mm=float(plane_z_mm),
            world_to_bev_homography_file=world_to_bev_homography_file,
            target_homography_direction=target_homography_direction,
            target_homography_space=target_homography_space,
            target_homography_output_size=None if target_homography_output_size is None else tuple(int(v) for v in target_homography_output_size),
            assume_world_xy_is_target_image=assume_world_xy_is_target_image,
            target_world_bounds_margin=float(target_world_bounds_margin),
            bev_projection_mode=str(bev_projection_mode),
            old_bev_canvas=tuple(int(v) for v in old_bev_canvas),
            old_world_origin=tuple(float(v) for v in old_world_origin),
            old_bev_resolution=float(old_bev_resolution),
            old_homography_invert=bool(old_homography_invert),
        )
        self.num_input_cameras = self.base.num_cam
        self.num_cam = 1 if self.bev_channel_stack else self.base.num_cam
        self.img_reduce = 8
        self.world_reduce = 1
        input_w, input_h = self.input_size
        bev_w, bev_h = self.bev_size
        self.img_shape = [input_h, input_w]
        self.worldgrid_shape = [bev_h, bev_w]
        self.Rimg_shape = [input_h // self.img_reduce, input_w // self.img_reduce]
        self.Rworld_shape = self.worldgrid_shape
        if self.use_cache and self.cache_dir is not None and self.cache_dir.exists() and any(self.cache_dir.glob("*/*.npz")):
            record_splits = build_records_from_cache_dir(
                self.cache_dir,
                split_ratios=split_ratios,
                max_frames_per_condition=max_frames_per_condition,
                split_mode=split_mode,
                split_seed=split_seed,
                split_manifest=split_manifest or split_manifest_path,
                root=self.root,
            )
        else:
            record_splits = build_records_from_sources(
                root=self.root,
                split_ratios=split_ratios,
                max_frames_per_condition=max_frames_per_condition,
                intervals_csv=intervals_csv,
                split_mode=split_mode,
                split_seed=split_seed,
                per_condition=self.per_condition,
                condition_mask_root=self.target_mask_root,
                condition_camera_id=self.target_camera_id,
                split_manifest=split_manifest or split_manifest_path,
            )
        manifest_output = write_split_manifest
        if manifest_output is None and split_manifest is None and split_manifest_path:
            legacy_path = resolve_under_root(self.root, split_manifest_path)
            manifest_output = None if legacy_path.exists() else split_manifest_path
        maybe_write_split_manifest(record_splits, manifest_output, self.root)
        self.records = flatten_split_records(record_splits, split)
        if not self.records:
            raise ValueError(f"No records found for split '{split}'")
        self.mask_stores = {}
        self.bev_mask_store = None
        self.video_store: Optional[VideoFrameStore] = None
        if not self.use_cache:
            if self.per_condition:
                self.mask_stores = {
                    camera_id: PerConditionCocoMaskStore(self.mask_root, camera_id)
                    for camera_id in self.input_camera_ids
                }
            else:
                self.mask_stores = {
                    camera_id: CocoMaskStore(find_coco_mask_json(self.mask_root, camera_id))
                    for camera_id in self.input_camera_ids
                }
            if self.target_mode == "camera_homography":
                if self.per_condition:
                    self.mask_stores[self.target_camera_id] = PerConditionCocoMaskStore(self.target_mask_root, self.target_camera_id)
                else:
                    self.mask_stores[self.target_camera_id] = CocoMaskStore(find_coco_mask_json(self.mask_root, self.target_camera_id))
            else:
                bev_json = self.root / self.bev_mask_json if self.bev_mask_json else find_coco_mask_json(self.bev_mask_root, "bev")
                self.bev_mask_store = CocoMaskStore(bev_json)
            if self.input_mode in {"roi_rgb", "image_rgb", "preprojected_bev_rgb"}:
                self.video_store = VideoFrameStore(self.video_root, self.input_camera_ids, per_condition=self.per_condition)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        cached = self._load_cache(record)
        if cached is not None:
            return cached
        inputs, target, affine = self.build_sample_numpy(record)
        return {
            "inputs": torch.from_numpy(inputs),
            "target": torch.from_numpy(target),
            "affine": torch.from_numpy(affine),
            "condition": record.condition,
            "frame_index": record.frame_index,
        }

    def _load_cache(self, record: FrameRecord) -> Optional[Dict[str, object]]:
        if not self.use_cache or self.cache_dir is None:
            return None
        path = self.cache_dir / str(record.condition) / f"{record.frame_index:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"MVDeTr cache missing: {path}. Run tools/precompute_ship_mvdetr_cache.py first.")
        data = np.load(path)
        inputs, affine = slice_cache_by_camera_ids(
            inputs=data["inputs"],
            affine=data["affine"],
            input_camera_ids=self.input_camera_ids,
            cache_camera_ids=self.cache_camera_ids,
            cache_path=path,
        )
        return {
            "inputs": torch.from_numpy(inputs.astype(np.float32)),
            "target": torch.from_numpy(data["target"].astype(np.float32)),
            "affine": torch.from_numpy(affine.astype(np.float32)),
            "condition": record.condition,
            "frame_index": record.frame_index,
        }

    def build_sample_numpy(self, record: FrameRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_w, input_h = self.input_size
        inputs = []
        affines = []
        for camera_id in self.input_camera_ids:
            params = self.base.camera_parameters[camera_id]
            if self.input_mode == "image_rgb":
                if self.video_store is None:
                    raise RuntimeError("input_mode='image_rgb' requires use_cache=False while building samples")
                frame = self.video_store.read_rgb(camera_id, record.frame_index, record.condition if self.per_condition else None)
                frame = undistort_mask(frame, params)
                orig_h, orig_w = frame.shape[:2]
                input_w, input_h = self.input_size
                interpolation = cv2.INTER_AREA if input_w < orig_w or input_h < orig_h else cv2.INTER_LINEAR
                resized_rgb = cv2.resize(frame, (input_w, input_h), interpolation=interpolation)
                inputs.append(normalize_rgb_chw(resized_rgb))
                affines.append(np.array([[input_w / orig_w, 0.0, 0.0], [0.0, input_h / orig_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
                continue
            if self.input_mode == "preprojected_bev_rgb":
                if self.video_store is None:
                    raise RuntimeError("input_mode='preprojected_bev_rgb' requires use_cache=False while building samples")
                frame = self.video_store.read_rgb(camera_id, record.frame_index, record.condition if self.per_condition else None)
                frame = undistort_mask(frame, params)
                mean_np = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
                std_np = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
                normalized_rgb = (frame.astype(np.float32) / 255.0 - mean_np) / std_np
                bev_rgb = cv2.warpPerspective(
                    normalized_rgb,
                    self.base.input_image_to_bev_mats[camera_id],
                    tuple(int(v) for v in self.bev_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0.0, 0.0, 0.0),
                )
                inputs.append(bev_rgb.transpose(2, 0, 1).astype(np.float32))
                affines.append(np.eye(3, dtype=np.float32))
                continue

            if self.per_condition:
                mask = self.mask_stores[camera_id].get_mask(record.condition, record.frame_index)
            else:
                mask = self.mask_stores[camera_id].get_mask(record.frame_index)
            mask = undistort_mask(mask, params)
            if self.input_mode == "preprojected_bev_mask":
                bev_mask = warp_mask_to_bev(mask, self.base.input_image_to_bev_mats[camera_id], self.bev_size)
                inputs.append(binarize(bev_mask, self.threshold)[None, ...])
                affines.append(np.eye(3, dtype=np.float32))
            elif self.input_mode == "roi_rgb":
                if self.video_store is None:
                    raise RuntimeError("input_mode='roi_rgb' requires use_cache=False while building samples")
                frame = self.video_store.read_rgb(camera_id, record.frame_index, record.condition if self.per_condition else None)
                frame = undistort_mask(frame, params)
                orig_h, orig_w = frame.shape[:2]
                roi_chw, _, _, _ = roi_rgb_from_mask(frame, mask, self.input_size, self.threshold)
                inputs.append(roi_chw)
                affines.append(np.array([[input_w / orig_w, 0.0, 0.0], [0.0, input_h / orig_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
            else:
                orig_h, orig_w = mask.shape[:2]
                resized = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
                inputs.append(binarize(resized, self.threshold)[None, ...])
                affines.append(np.array([[input_w / orig_w, 0.0, 0.0], [0.0, input_h / orig_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
        if self.target_mode == "camera_homography":
            if self.per_condition:
                target = self.mask_stores[self.target_camera_id].get_mask(record.condition, record.frame_index)
            else:
                target = self.mask_stores[self.target_camera_id].get_mask(record.frame_index)
            target_bev = warp_mask_to_bev(target, self.base.target_image_to_bev_mat, self.bev_size)
        else:
            if self.bev_mask_store is None:
                raise RuntimeError("direct_bev_mask target mode requires use_cache=False or precomputed target cache")
            target_bev = self.bev_mask_store.get_mask(record.frame_index)
            bev_w, bev_h = self.bev_size
            if target_bev.shape[:2] != (bev_h, bev_w):
                target_bev = cv2.resize(target_bev, (bev_w, bev_h), interpolation=cv2.INTER_NEAREST)
        if self.bev_channel_stack:
            input_array = np.concatenate(inputs, axis=0)[None, ...]
            affine_array = np.eye(3, dtype=np.float32)[None, ...]
        else:
            input_array = np.stack(inputs, axis=0)
            affine_array = np.stack(affines, axis=0)
        return (
            input_array.astype(np.float32),
            binarize(target_bev, self.threshold)[None, ...].astype(np.float32),
            affine_array.astype(np.float32),
        )

    def build_raw_view_diagnostics(self, record: FrameRecord) -> Dict[str, np.ndarray]:
        if self.video_root is None:
            raise RuntimeError("Raw-view diagnostics require dataset.video_dir")
        video_store = self.video_store or VideoFrameStore(self.video_root, self.input_camera_ids, per_condition=self.per_condition)
        temporary_store = self.video_store is None
        try:
            rgbs = []
            masks = []
            rois = []
            for camera_id in self.input_camera_ids:
                if self.per_condition:
                    mask = self.mask_stores[camera_id].get_mask(record.condition, record.frame_index)
                else:
                    mask = self.mask_stores[camera_id].get_mask(record.frame_index)
                params = self.base.camera_parameters[camera_id]
                mask = undistort_mask(mask, params)
                frame = video_store.read_rgb(camera_id, record.frame_index, record.condition if self.per_condition else None)
                frame = undistort_mask(frame, params)
                _, resized_rgb, resized_mask, roi_uint8 = roi_rgb_from_mask(frame, mask, self.input_size, self.threshold)
                rgbs.append(resized_rgb)
                masks.append(resized_mask.astype(np.float32))
                rois.append(roi_uint8)
            return {
                "rgb": np.stack(rgbs, axis=0),
                "mask": np.stack(masks, axis=0),
                "roi_rgb": np.stack(rois, axis=0),
            }
        finally:
            if temporary_store:
                video_store.close()
