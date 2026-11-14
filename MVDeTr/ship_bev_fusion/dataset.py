"""Dataset for synchronized ship-pool BEV mask fusion."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from .calibration import build_homographies, load_camera_parameters, load_matrix_file, undistort_mask, warp_mask_to_bev


@dataclass(frozen=True)
class FrameRecord:
    condition: str
    frame_index: int
    split: str


class ShipPoolMaskDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        input_camera_ids: Sequence[str] = ("222", "223", "224"),
        target_camera_id: str = "221",
        mask_dir: str = "mask",
        camera_parameters_dir: str = "camera_parameters",
        intervals_csv: str = "synced_clean_intervals.csv",
        plane_z_mm: float = 93.0,
        bev_size: Tuple[int, int] = (1024, 1024),
        target_homography_direction: str = "image_to_bev",
        target_homography_output_size: Optional[Tuple[int, int]] = None,
        target_homography_space: str = "bev",
        target_homography_image_size: Optional[Tuple[int, int]] = None,
        world_to_bev_homography_file: Optional[str] = None,
        assume_world_xy_is_target_image: bool = True,
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
        bev_cache_dir: Optional[str] = None,
        use_bev_cache: bool = False,
        mask_cache_size: int = 16,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.input_camera_ids = tuple(str(c) for c in input_camera_ids)
        self.target_camera_id = canonical_camera_id(target_camera_id)
        self.per_condition = bool(per_condition)
        self.mask_root = resolve_under_root(self.root, mask_dir)
        self.target_mask_root = resolve_under_root(self.root, target_mask_dir) if target_mask_dir else self.mask_root
        self.camera_parameters_root = resolve_under_root(self.root, camera_parameters_dir)
        self.threshold = int(threshold)
        self.bev_size = tuple(int(v) for v in bev_size)
        self.bev_cache_dir = self.root / bev_cache_dir if bev_cache_dir else None
        self.use_bev_cache = bool(use_bev_cache)
        self.mask_cache_size = int(mask_cache_size)
        self._cache: "OrderedDict[Tuple[str, int], np.ndarray]" = OrderedDict()
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
            source = self.target_mask_root if self.per_condition else self.root / str(intervals_csv)
            raise ValueError(f"No records found for split '{split}' from {source}")
        if self.use_bev_cache:
            self.homographies = None
            self.camera_parameters = {}
            self.mask_stores = {}
            return
        world_to_bev = load_matrix_file(self.root / world_to_bev_homography_file) if world_to_bev_homography_file else None
        self.homographies = build_homographies(
            self.camera_parameters_root,
            self.input_camera_ids,
            self.target_camera_id,
            plane_z=float(plane_z_mm),
            bev_size=self.bev_size,
            target_homography_direction=target_homography_direction,
            target_homography_output_size=target_homography_output_size,
            target_homography_space=target_homography_space,
            target_homography_image_size=target_homography_image_size,
            world_to_bev_override=world_to_bev,
            assume_world_xy_is_target_image=assume_world_xy_is_target_image,
        )
        self.camera_parameters = {c: load_camera_parameters(self.camera_parameters_root, c) for c in self.input_camera_ids}
        if self.per_condition:
            self.mask_stores = {
                camera_id: PerConditionCocoMaskStore(self.mask_root, camera_id)
                for camera_id in self.input_camera_ids
            }
            self.mask_stores[self.target_camera_id] = PerConditionCocoMaskStore(self.target_mask_root, self.target_camera_id)
        else:
            self.mask_stores = {
                camera_id: CocoMaskStore(find_coco_mask_json(self.mask_root, camera_id))
                for camera_id in (*self.input_camera_ids, self.target_camera_id)
            }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        cached = self._load_bev_cache(record)
        if cached is not None:
            return cached
        inputs, target = self.build_bev_sample_numpy(record)
        return {
            "inputs": torch.from_numpy(inputs),
            "target": torch.from_numpy(target),
            "condition": record.condition,
            "frame_index": record.frame_index,
            "split": record.split,
        }

    def _load_bev_cache(self, record: FrameRecord) -> Optional[Dict[str, object]]:
        if not self.use_bev_cache or self.bev_cache_dir is None:
            return None
        path = self.bev_cache_dir / str(record.condition) / f"{record.frame_index:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"BEV cache missing: {path}. Run tools/precompute_ship_bev_cache.py first.")
        data = np.load(path)
        return {
            "inputs": torch.from_numpy(data["inputs"].astype(np.float32)),
            "target": torch.from_numpy(data["target"].astype(np.float32)),
            "condition": record.condition,
            "frame_index": record.frame_index,
            "split": record.split,
        }

    def build_bev_sample_numpy(self, record: FrameRecord) -> Tuple[np.ndarray, np.ndarray]:
        side_masks = []
        for camera_id in self.input_camera_ids:
            mask = self._get_mask(camera_id, record)
            mask = undistort_mask(mask, self.camera_parameters[camera_id])
            bev = warp_mask_to_bev(mask, self.homographies.image_to_bev[camera_id], self.bev_size)
            side_masks.append(binarize(bev, self.threshold))
        target = self._get_mask(self.target_camera_id, record)
        target_bev = warp_mask_to_bev(target, self.homographies.target_image_to_bev, self.bev_size)
        return np.stack(side_masks, axis=0).astype(np.float32), binarize(target_bev, self.threshold)[None, ...].astype(np.float32)

    def _get_mask(self, camera_id: str, record: FrameRecord) -> np.ndarray:
        key = (camera_id, str(record.condition), int(record.frame_index))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        store = self.mask_stores[camera_id]
        if self.per_condition:
            mask = store.get_mask(record.condition, record.frame_index)
        else:
            mask = store.get_mask(record.frame_index)
        if self.mask_cache_size > 0:
            self._cache[key] = mask
            self._cache.move_to_end(key)
            while len(self._cache) > self.mask_cache_size:
                self._cache.popitem(last=False)
        return mask


def parse_intervals_csv(path: Path) -> Dict[str, Tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    intervals: Dict[str, Tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        headers = {h.lower().strip(): h for h in reader.fieldnames}
        condition_key = _first_existing(headers, ("condition", "case", "scene", "sequence", "work_condition", "工况"))
        start_key = _first_existing(headers, ("start_frame", "start", "begin", "begin_frame", "起始帧", "起始"))
        end_key = _first_existing(headers, ("end_frame", "end", "stop", "finish_frame", "截至帧", "结束帧", "结束"))
        if start_key is None or end_key is None:
            raise ValueError(f"{path} must include start/end frame columns. Headers: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            condition = row.get(condition_key, "").strip() if condition_key else f"condition_{row_number - 1:03d}"
            start = int(float(row[start_key]))
            end = int(float(row[end_key]))
            if end < start:
                start, end = end, start
            intervals[condition or f"condition_{row_number - 1:03d}"] = (start, end)
    return intervals


def build_split_records(
    intervals: Mapping[str, Tuple[int, int]],
    split_ratios: Tuple[float, float, float],
    max_frames_per_condition: Optional[int] = None,
) -> Dict[str, List[FrameRecord]]:
    train_ratio, val_ratio, test_ratio = split_ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    splits = {"train": [], "val": [], "test": []}
    for condition, (start, end) in intervals.items():
        frames = list(range(int(start), int(end) + 1))
        if max_frames_per_condition is not None:
            frames = frames[: int(max_frames_per_condition)]
        n = len(frames)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        else:
            n_train = max(1, n - 1)
            n_val = 0
        for split, split_frames in (
            ("train", frames[:n_train]),
            ("val", frames[n_train : n_train + n_val]),
            ("test", frames[n_train + n_val :]),
        ):
            splits[split].extend(FrameRecord(condition, frame, split) for frame in split_frames)
    return splits


def build_condition_split_records(
    intervals: Mapping[str, Tuple[int, int]],
    split_ratios: Tuple[float, float, float],
    max_frames_per_condition: Optional[int] = None,
    split_seed: int = 20260510,
) -> Dict[str, List[FrameRecord]]:
    train_ratio, val_ratio, test_ratio = split_ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    conditions = sorted(intervals.keys(), key=condition_sort_key)
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
    splits = {"train": [], "val": [], "test": []}
    for split, split_condition_names in split_conditions.items():
        for condition in sorted(split_condition_names, key=condition_sort_key):
            start, end = intervals[condition]
            frames = list(range(int(start), int(end) + 1))
            if max_frames_per_condition is not None:
                frames = frames[: int(max_frames_per_condition)]
            splits[split].extend(FrameRecord(str(condition), int(frame), split) for frame in frames)
    return splits


def build_records_from_sources(
    root: Path,
    split_ratios: Tuple[float, float, float],
    max_frames_per_condition: Optional[int] = None,
    intervals_csv: Optional[str] = "synced_clean_intervals.csv",
    split_mode: str = "frame",
    split_seed: int = 20260510,
    per_condition: bool = False,
    condition_mask_root: Optional[Path] = None,
    condition_camera_id: str = "GT",
    split_manifest: Optional[str | Path] = None,
) -> Dict[str, List[FrameRecord]]:
    if per_condition:
        if condition_mask_root is None:
            raise ValueError("condition_mask_root is required for per_condition datasets")
        intervals = discover_per_condition_intervals(condition_mask_root, condition_camera_id)
    else:
        if intervals_csv is None:
            raise ValueError("intervals_csv is required when per_condition=False")
        intervals = parse_intervals_csv(Path(root) / str(intervals_csv))
    if split_manifest:
        manifest_path = resolve_under_root(Path(root), split_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        condition_frames = {
            str(condition): list(range(int(start), int(end) + 1))
            for condition, (start, end) in intervals.items()
        }
        return build_records_from_split_manifest(manifest_path, condition_frames, max_frames_per_condition)
    if split_mode == "locked_manifest":
        raise ValueError("split_mode='locked_manifest' requires dataset.split_manifest")
    if split_mode == "condition_random":
        return build_condition_split_records(intervals, split_ratios, max_frames_per_condition, split_seed)
    if split_mode in {"frame", "frame_within_condition"}:
        return build_split_records(intervals, split_ratios, max_frames_per_condition)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def build_records_from_split_manifest(
    manifest_path: Path,
    condition_frames: Mapping[str, Sequence[int]],
    max_frames_per_condition: Optional[int] = None,
) -> Dict[str, List[FrameRecord]]:
    'Build records from a locked condition/frame split manifest.'
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split_data = data.get("splits", data)
    condition_by_token = {condition_token(condition): str(condition) for condition in condition_frames}
    records_by_split: Dict[str, List[FrameRecord]] = {"train": [], "val": [], "test": []}
    seen_conditions: Dict[str, str] = {}

    for split in ("train", "val", "test"):
        raw_entry = split_data.get(split, {})
        for condition, frames in iter_manifest_conditions(raw_entry):
            token = condition_token(str(condition))
            canonical = condition_by_token.get(token)
            if canonical is None:
                raise KeyError(f"Condition {condition!r} from {manifest_path} is not available")
            previous = seen_conditions.get(token)
            if previous is not None and previous != split:
                raise ValueError(f"Condition {condition!r} appears in both {previous!r} and {split!r}")
            seen_conditions[token] = split
            available = [int(frame) for frame in condition_frames[canonical]]
            selected = [int(frame) for frame in (frames if frames is not None else available)]
            if max_frames_per_condition is not None:
                selected = selected[: int(max_frames_per_condition)]
            missing = sorted(set(selected) - set(available))
            if missing:
                preview = ", ".join(str(frame) for frame in missing[:10])
                raise KeyError(f"Manifest {manifest_path} requests missing frames for condition {condition}: {preview}")
            records_by_split[split].extend(FrameRecord(canonical, int(frame), split) for frame in selected)
    return records_by_split


def iter_manifest_conditions(raw_entry: object) -> Iterable[Tuple[str, Optional[List[int]]]]:
    if raw_entry is None:
        return
    if isinstance(raw_entry, Mapping):
        records = raw_entry.get("records")
        if records is not None:
            for item in records:
                yield manifest_record_to_condition_frames(item)
            return
        frames_by_condition = raw_entry.get("frames_by_condition")
        if isinstance(frames_by_condition, Mapping):
            for condition, frames in frames_by_condition.items():
                yield str(condition), [int(frame) for frame in frames]
            return
        conditions = raw_entry.get("conditions", [])
    else:
        conditions = raw_entry
    for item in conditions or []:
        yield manifest_record_to_condition_frames(item)


def manifest_record_to_condition_frames(item: object) -> Tuple[str, Optional[List[int]]]:
    if isinstance(item, Mapping):
        condition = item.get("condition", item.get("id", item.get("name")))
        if condition is None:
            raise ValueError(f"Manifest record is missing condition: {item}")
        if "frames" in item and item["frames"] is not None:
            return str(condition), [int(frame) for frame in item["frames"]]
        if "frame_indices" in item and item["frame_indices"] is not None:
            return str(condition), [int(frame) for frame in item["frame_indices"]]
        if "frame" in item and item["frame"] is not None:
            return str(condition), [int(item["frame"])]
        return str(condition), None
    return str(item), None


def flatten_split_records(records_by_split: Mapping[str, List[FrameRecord]], split: str) -> List[FrameRecord]:
    if split == "all":
        records: List[FrameRecord] = []
        for split_name in ("train", "val", "test"):
            records.extend(FrameRecord(r.condition, r.frame_index, "all") for r in records_by_split.get(split_name, []))
        return records
    return list(records_by_split.get(split, []))


def maybe_write_split_manifest(records_by_split: Mapping[str, List[FrameRecord]], manifest_path: Optional[str], root: Path) -> None:
    if not manifest_path:
        return
    path = resolve_under_root(root, manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for split, records in records_by_split.items():
        conditions = sorted({str(record.condition) for record in records}, key=condition_sort_key)
        frames_by_condition = {
            condition: [
                int(record.frame_index)
                for record in sorted(records, key=lambda item: (condition_sort_key(item.condition), int(item.frame_index)))
                if str(record.condition) == condition
            ]
            for condition in conditions
        }
        data[split] = {
            "conditions": conditions,
            "frames_by_condition": frames_by_condition,
            "num_conditions": len(conditions),
            "num_frames": len(records),
        }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class CocoMaskStore:
    def __init__(self, path: Path) -> None:
        import json

        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.images_by_id = {int(img["id"]): img for img in data.get("images", [])}
        self.image_id_by_frame = {}
        for image_id, img in self.images_by_id.items():
            stem = Path(img.get("file_name", str(image_id))).stem
            try:
                frame = int(stem)
            except ValueError:
                frame = image_id
            self.image_id_by_frame[frame] = image_id
        self.annotations_by_image: Dict[int, List[Mapping[str, object]]] = {}
        for ann in data.get("annotations", []):
            self.annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    def frame_indices(self) -> List[int]:
        return sorted(int(frame) for frame in self.image_id_by_frame)

    def get_mask(self, frame_index: int) -> np.ndarray:
        image_id = self.image_id_by_frame.get(int(frame_index), int(frame_index))
        image = self.images_by_id.get(image_id)
        if image is None:
            raise FileNotFoundError(f"Frame {frame_index} not found in {self.path}")
        height, width = int(image["height"]), int(image["width"])
        out = np.zeros((height, width), dtype=np.uint8)
        for ann in self.annotations_by_image.get(image_id, []):
            segmentation = ann.get("segmentation")
            if segmentation is None:
                continue
            if isinstance(segmentation, list):
                rles = mask_utils.frPyObjects(segmentation, height, width)
                decoded = mask_utils.decode(rles)
                if decoded.ndim == 3:
                    decoded = np.any(decoded, axis=2)
            else:
                rle = dict(segmentation)
                counts = rle.get("counts")
                if isinstance(counts, str):
                    rle["counts"] = counts.encode("ascii")
                decoded = mask_utils.decode(rle)
            out[np.asarray(decoded).astype(bool)] = 255
        return out


class PerConditionCocoMaskStore:
    def __init__(self, root: Path, camera_id: str, prefer_edited: bool = True) -> None:
        self.root = Path(root)
        self.camera_id = canonical_camera_id(camera_id)
        self.prefer_edited = bool(prefer_edited)
        self._stores: Dict[str, CocoMaskStore] = {}
        self._paths: Dict[str, Path] = {}

    def get_mask(self, condition: str, frame_index: int) -> np.ndarray:
        return self._store(condition).get_mask(frame_index)

    def frame_indices(self, condition: str) -> List[int]:
        return self._store(condition).frame_indices()

    def mask_path(self, condition: str) -> Path:
        key = condition_token(condition)
        if key not in self._paths:
            self._paths[key] = find_per_condition_coco_mask_json(self.root, self.camera_id, condition, self.prefer_edited)
        return self._paths[key]

    def _store(self, condition: str) -> CocoMaskStore:
        key = condition_token(condition)
        if key not in self._stores:
            self._stores[key] = CocoMaskStore(self.mask_path(condition))
        return self._stores[key]


def find_coco_mask_json(mask_root: Path, camera_id: str) -> Path:
    camera_id = canonical_camera_id(camera_id)
    candidates = [
        mask_root / f"camera_{camera_id}_synced_clean_coco_masks.json",
        mask_root / f"camera_{camera_id}_coco_masks.json",
        mask_root / f"camera_{camera_id}.json",
        mask_root / f"{camera_id}_coco_masks.json",
        mask_root / f"{camera_id}_masks.json",
        mask_root / f"{camera_id}_synced_clean_coco_masks.json",
        mask_root / f"{camera_id}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find COCO mask JSON for camera {camera_id} in {mask_root}")


def find_per_condition_coco_mask_json(mask_root: Path, camera_id: str, condition: str, prefer_edited: bool = True) -> Path:
    camera_id = canonical_camera_id(camera_id)
    token = condition_token(condition)
    condition_dir = condition_dir_name(condition)
    base_dirs = [
        mask_root / f"view_{camera_id}" / condition_dir / "sam2",
        mask_root / f"view_{camera_id}" / condition_dir,
        mask_root / condition_dir / "sam2",
        mask_root / condition_dir,
    ]
    stems = [
        f"camera_{camera_id}_{token}_coco_masks",
        f"camera_{camera_id}_{condition_dir}_coco_masks",
        f"{camera_id}_{token}_coco_masks",
        f"{camera_id}_coco_masks",
    ]
    suffixes = ["_edited.json", ".json"] if prefer_edited else [".json", "_edited.json"]
    for directory in base_dirs:
        for stem in stems:
            for suffix in suffixes:
                path = directory / f"{stem}{suffix}"
                if path.exists():
                    return path
    raise FileNotFoundError(f"Could not find per-condition COCO mask JSON: camera={camera_id}, condition={condition}, root={mask_root}")


def discover_per_condition_intervals(mask_root: Path, camera_id: str) -> Dict[str, Tuple[int, int]]:
    mask_root = Path(mask_root)
    intervals: Dict[str, Tuple[int, int]] = {}
    condition_dirs = sorted(mask_root.glob("condition_*"), key=lambda p: condition_sort_key(p.name))
    if not condition_dirs:
        condition_dirs = sorted((p for p in mask_root.rglob("condition_*") if p.is_dir()), key=lambda p: condition_sort_key(p.name))
    for condition_path in condition_dirs:
        condition = condition_token(condition_path.name)
        try:
            store = CocoMaskStore(find_per_condition_coco_mask_json(mask_root, camera_id, condition))
        except FileNotFoundError:
            continue
        frames = store.frame_indices()
        if frames:
            intervals[condition] = (min(frames), max(frames))
    if not intervals:
        raise FileNotFoundError(f"No per-condition mask JSON files found for camera {camera_id} under {mask_root}")
    return intervals


def canonical_camera_id(camera_id: str) -> str:
    camera_id = str(camera_id)
    return "GT" if camera_id.lower() == "gt" else camera_id


def condition_token(condition: str) -> str:
    condition = str(condition)
    match = re.search(r"(\d+)$", condition)
    return str(int(match.group(1))) if match else condition


def condition_dir_name(condition: str) -> str:
    condition = str(condition)
    return condition if condition.startswith("condition_") else f"condition_{condition_token(condition)}"


def condition_sort_key(condition: str) -> Tuple[int, object]:
    token = condition_token(condition)
    return (0, int(token)) if str(token).isdigit() else (1, str(condition))


def resolve_under_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def find_mask_file(mask_root: Path, camera_id: str, condition: str, frame_index: int) -> Path:
    camera_tokens = (f"camera_{camera_id}", f"cam_{camera_id}", str(camera_id))
    frame_tokens = (f"{frame_index:06d}", f"{frame_index:05d}", f"{frame_index:04d}", str(frame_index))
    suffixes = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    direct_dirs = []
    for cam in camera_tokens:
        direct_dirs.extend([mask_root / condition / cam, mask_root / cam / condition, mask_root / condition, mask_root / cam])
    for directory in direct_dirs:
        for token in frame_tokens:
            for suffix in suffixes:
                for name in (
                    f"{token}{suffix}",
                    f"frame_{token}{suffix}",
                    f"{condition}_{token}{suffix}",
                    f"camera_{camera_id}_{token}{suffix}",
                    f"cam_{camera_id}_{token}{suffix}",
                    f"{camera_id}_{token}{suffix}",
                ):
                    path = directory / name
                    if path.exists():
                        return path
    pattern = re.compile(rf"(?:^|[^0-9]){frame_index}(?:[^0-9]|$)")
    for cam in camera_tokens:
        for root in (mask_root / cam / condition, mask_root / condition / cam, mask_root / cam, mask_root / condition):
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.suffix.lower() in suffixes and pattern.search(path.stem):
                    return path
    raise FileNotFoundError(f"Mask not found: camera={camera_id}, condition={condition}, frame={frame_index}")


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask image: {path}")
    return mask


def binarize(mask: np.ndarray, threshold: int) -> np.ndarray:
    return (mask > threshold).astype(np.float32)


def _first_existing(headers: Mapping[str, str], names: Iterable[str]) -> Optional[str]:
    for name in names:
        key = headers.get(name.lower())
        if key is not None:
            return key
    return None
