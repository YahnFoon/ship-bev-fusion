"""RGB-video dataset for direct MVDeTr ship BEV mask segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .calibration import undistort_mask, warp_mask_to_bev
from .dataset import (
    CocoMaskStore,
    FrameRecord,
    PerConditionCocoMaskStore,
    binarize,
    build_records_from_sources,
    canonical_camera_id,
    condition_dir_name,
    condition_token,
    find_coco_mask_json,
    flatten_split_records,
    resolve_under_root,
)
from .mvdetr_dataset import ShipMVDeTrBase, slice_cache_by_camera_ids


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def iter_mvdetr_records(
    root: Path,
    intervals_csv: str,
    split: str,
    split_ratios: Sequence[float],
    max_frames_per_condition: Optional[int] = None,
    per_condition: bool = False,
    target_mask_dir: Optional[str] = None,
    split_mode: str = "frame",
    split_seed: int = 20260510,
    split_manifest: Optional[str] = None,
) -> List[FrameRecord]:
    condition_mask_root = resolve_under_root(root, target_mask_dir) if target_mask_dir else None
    records_by_split = build_records_from_sources(
        root=Path(root),
        intervals_csv=intervals_csv,
        split_ratios=tuple(float(v) for v in split_ratios),
        max_frames_per_condition=max_frames_per_condition,
        split_mode=split_mode,
        split_seed=split_seed,
        per_condition=per_condition,
        condition_mask_root=condition_mask_root,
        condition_camera_id="GT",
        split_manifest=split_manifest,
    )
    return flatten_split_records(records_by_split, split)


class VideoFrameStore:
    """Lazy random-access reader for synchronized camera videos."""

    def __init__(self, video_root: Path, camera_ids: Sequence[str], per_condition: bool = False) -> None:
        self.video_root = Path(video_root)
        self.per_condition = bool(per_condition)
        self.paths = {} if self.per_condition else {str(camera_id): find_video_path(self.video_root, str(camera_id)) for camera_id in camera_ids}
        self._caps: Dict[Tuple[str, Optional[str]], cv2.VideoCapture] = {}

    def __getstate__(self) -> Mapping[str, object]:
        return {"video_root": self.video_root, "per_condition": self.per_condition, "paths": self.paths, "_caps": {}}

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def _capture(self, camera_id: str, condition: Optional[str] = None) -> cv2.VideoCapture:
        camera_id = canonical_camera_id(camera_id)
        key = (camera_id, condition_token(condition) if condition is not None else None)
        cap = self._caps.get(key)
        if cap is None:
            if len(self._caps) >= 8:
                self._caps.pop(next(iter(self._caps))).release()
            path = find_video_path(self.video_root, camera_id, condition) if self.per_condition else self.paths[camera_id]
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {path}")
            self._caps[key] = cap
        return cap

    def read_rgb(self, camera_id: str, frame_index: int, condition: Optional[str] = None) -> np.ndarray:
        cap = self._capture(camera_id, condition)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise EOFError(f"Failed to read frame {frame_index} for camera={camera_id}, condition={condition}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_frame_to_chw(
    frame_rgb: np.ndarray,
    input_size: Tuple[int, int],
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    input_w, input_h = tuple(int(v) for v in input_size)
    orig_h, orig_w = frame_rgb.shape[:2]
    interpolation = cv2.INTER_AREA if input_w < orig_w or input_h < orig_h else cv2.INTER_LINEAR
    resized = cv2.resize(frame_rgb, (input_w, input_h), interpolation=interpolation)
    image = resized.astype(np.float32) / 255.0
    mean_np = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return ((image - mean_np) / std_np).transpose(2, 0, 1).astype(np.float32)


class ShipMVDeTrRGBDataset(Dataset):
    """Direct RGB multi-view input with the same BEV mask target as mask fusion."""

    def __init__(
        self,
        root: Path,
        split: str,
        input_camera_ids: Sequence[str] = ("222", "223", "224"),
        target_camera_id: str = "221",
        video_dir: str = "video",
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
        split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        threshold: int = 127,
        normalize_mean: Sequence[float] = IMAGENET_MEAN,
        normalize_std: Sequence[float] = IMAGENET_STD,
        max_frames_per_condition: Optional[int] = None,
        cache_dir: Optional[str] = None,
        cache_camera_ids: Optional[Sequence[str]] = None,
        use_cache: bool = False,
        per_condition: bool = False,
        target_mask_dir: Optional[str] = None,
        split_mode: str = "frame",
        split_seed: int = 20260510,
        split_manifest: Optional[str] = None,
        input_mode: str = "image_rgb",
    ) -> None:
        if str(input_mode) != "image_rgb":
            raise ValueError("ShipMVDeTrRGBDataset only supports input_mode='image_rgb'")
        self.root = Path(root)
        self.split = split
        self.input_camera_ids = tuple(str(c) for c in input_camera_ids)
        self.target_camera_id = canonical_camera_id(target_camera_id)
        self.per_condition = bool(per_condition)
        self.video_root = resolve_under_root(self.root, video_dir)
        self.mask_root = resolve_under_root(self.root, mask_dir)
        self.target_mask_root = resolve_under_root(self.root, target_mask_dir) if target_mask_dir else self.mask_root
        self.threshold = int(threshold)
        self.bev_size = tuple(int(v) for v in bev_size)
        self.input_size = tuple(int(v) for v in input_size)
        self.normalize_mean = tuple(float(v) for v in normalize_mean)
        self.normalize_std = tuple(float(v) for v in normalize_std)
        self.cache_dir = resolve_under_root(self.root, cache_dir) if cache_dir else None
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
        )
        self.num_cam = self.base.num_cam
        self.img_reduce = 8
        self.world_reduce = 1
        input_w, input_h = self.input_size
        bev_w, bev_h = self.bev_size
        self.img_shape = [input_h, input_w]
        self.worldgrid_shape = [bev_h, bev_w]
        self.Rimg_shape = [input_h // self.img_reduce, input_w // self.img_reduce]
        self.Rworld_shape = self.worldgrid_shape
        self.records = iter_mvdetr_records(
            self.root,
            intervals_csv,
            split,
            split_ratios,
            max_frames_per_condition,
            per_condition=self.per_condition,
            target_mask_dir=str(self.target_mask_root),
            split_mode=split_mode,
            split_seed=split_seed,
            split_manifest=split_manifest,
        )
        if not self.records:
            raise ValueError(f"No records found for split '{split}'")
        self.video_store: Optional[VideoFrameStore] = None
        self.target_mask_store: Optional[CocoMaskStore] = None
        if not self.use_cache:
            self.video_store = VideoFrameStore(self.video_root, self.input_camera_ids, per_condition=self.per_condition)
            if self.per_condition:
                self.target_mask_store = PerConditionCocoMaskStore(self.target_mask_root, self.target_camera_id)
            else:
                self.target_mask_store = CocoMaskStore(find_coco_mask_json(self.mask_root, self.target_camera_id))

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
            raise FileNotFoundError(f"RGB MVDeTr cache missing: {path}. Run tools/precompute_ship_mvdetr_rgb_cache.py first.")
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
        if self.video_store is None or self.target_mask_store is None:
            raise RuntimeError("build_sample_numpy requires use_cache=False")
        input_w, input_h = self.input_size
        inputs = []
        affines = []
        for camera_id in self.input_camera_ids:
            frame = self.video_store.read_rgb(camera_id, record.frame_index, record.condition if self.per_condition else None)
            params = self.base.camera_parameters[camera_id]
            frame = undistort_mask(frame, params)
            orig_h, orig_w = frame.shape[:2]
            inputs.append(rgb_frame_to_chw(frame, self.input_size, self.normalize_mean, self.normalize_std))
            affines.append(np.array([[input_w / orig_w, 0.0, 0.0], [0.0, input_h / orig_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
        if self.per_condition:
            target = self.target_mask_store.get_mask(record.condition, record.frame_index)
        else:
            target = self.target_mask_store.get_mask(record.frame_index)
        target_bev = warp_mask_to_bev(target, self.base.target_image_to_bev_mat, self.bev_size)
        return (
            np.stack(inputs, axis=0).astype(np.float32),
            binarize(target_bev, self.threshold)[None, ...].astype(np.float32),
            np.stack(affines, axis=0).astype(np.float32),
        )
