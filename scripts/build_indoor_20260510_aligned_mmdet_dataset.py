#!/usr/bin/env python3
'Build an MMDetection COCO dataset from Indoor 20260510 aligned condition data.'

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2


DEFAULT_DATA_ROOT = Path("data/indoor")
DEFAULT_VIDEO_REL = Path("Video/waterLevel_9/waterlevel_9_synced_conditions")
DEFAULT_MASK_REL = Path("Video/waterLevel_9/multiview_dino_sam2_masks_zero_valid")
DEFAULT_SPLIT_MANIFEST_REL = Path("splits/water9.json")
DEFAULT_CAMERAS = ("221", "222", "223", "224")


@dataclass(frozen=True)
class FrameRecord:
    condition: str
    frame_index: int
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_REL)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_REL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera-ids", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST_REL)
    parser.add_argument("--split-ratios", nargs=3, type=float, default=[0.7, 0.1, 0.2])
    parser.add_argument("--split-seed", type=int, default=20260510)
    parser.add_argument("--max-frames-per-condition", type=int, default=None)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--category-name", default="ship")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--skip-frame-extract", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
        help="Only build the requested splits; defaults to all splits.",
    )
    return parser.parse_args()


def resolve_under_root(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def condition_token(value: str) -> str:
    match = re.search(r"(\d+)$", str(value))
    return str(int(match.group(1))) if match else str(value)


def condition_dir_name(value: str) -> str:
    return f"condition_{condition_token(value)}"


def condition_sort_key(value: str) -> Tuple[int, object]:
    token = condition_token(value)
    return (0, int(token)) if token.isdigit() else (1, token)


def discover_conditions(mask_root: Path, camera_ids: Sequence[str]) -> List[str]:
    conditions: set[str] = set()
    for camera_id in camera_ids:
        view_dir = mask_root / f"view_{camera_id}"
        for path in view_dir.glob("condition_*"):
            if path.is_dir():
                conditions.add(condition_token(path.name))
    if not conditions:
        raise FileNotFoundError(f"No condition_* directories found under {mask_root}")
    return sorted(conditions, key=condition_sort_key)


def find_mask_json(mask_root: Path, camera_id: str, condition: str) -> Path:
    token = condition_token(condition)
    candidates = [
        mask_root / f"view_{camera_id}" / f"condition_{token}" / "sam2" / f"camera_{camera_id}_{token}_coco_masks.json",
        mask_root / f"view_{camera_id}" / f"condition_{token}" / f"camera_{camera_id}_{token}_coco_masks.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing mask JSON for camera={camera_id} condition={token}: {candidates[0]}")


def find_video(video_root: Path, camera_id: str, condition: str) -> Path:
    token = condition_token(condition)
    candidates = [
        video_root / camera_id / f"camera_{camera_id}_{token}.mp4",
        video_root / camera_id / f"camera_{camera_id}_condition_{token}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing video for camera={camera_id} condition={token}: {candidates[0]}")


def frame_from_image(image: Mapping[str, Any], fallback_id: int) -> int:
    for key in ("frame_index", "frame_id"):
        if key in image:
            return int(image[key])
    stem = Path(str(image.get("file_name", fallback_id))).stem
    try:
        return int(stem)
    except ValueError:
        return int(fallback_id)


def load_condition_mask_data(mask_root: Path, camera_id: str, condition: str) -> Dict[str, Any]:
    path = find_mask_json(mask_root, camera_id, condition)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images_by_id = {int(image["id"]): image for image in data.get("images", [])}
    image_id_by_frame = {
        frame_from_image(image, int(image_id)): int(image_id)
        for image_id, image in images_by_id.items()
    }
    anns_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        anns_by_image_id.setdefault(image_id, []).append(ann)
    return {
        "path": str(path),
        "images_by_id": images_by_id,
        "image_id_by_frame": image_id_by_frame,
        "anns_by_image_id": anns_by_image_id,
    }


def valid_annotation(ann: Mapping[str, Any]) -> bool:
    bbox = ann.get("bbox", [])
    if len(bbox) != 4:
        return False
    if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
        return False
    if float(ann.get("area", 0.0)) <= 0.0:
        return False
    return bool(ann.get("segmentation"))


def load_manifest(path: Path) -> Optional[Dict[str, Dict[str, Optional[List[int]]]]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    split_data = data.get("splits", data)
    out: Dict[str, Dict[str, Optional[List[int]]]] = {}
    for split in ("train", "val", "test"):
        value = split_data.get(split, {})
        out[split] = parse_manifest_split_entry(value)
    if not any(out.values()):
        raise ValueError(f"Split manifest has no conditions: {path}")
    return out


def parse_manifest_split_entry(value: Any) -> Dict[str, Optional[List[int]]]:
    selected: Dict[str, Optional[List[int]]] = {}
    if isinstance(value, Mapping):
        frames_by_condition = value.get("frames_by_condition")
        if isinstance(frames_by_condition, Mapping):
            for condition, frames in frames_by_condition.items():
                selected[condition_token(str(condition))] = [int(frame) for frame in frames]
        records = value.get("records")
        if records is not None:
            for item in records:
                condition, frames = parse_manifest_condition_item(item)
                selected[condition_token(condition)] = frames
            return selected
        conditions = value.get("conditions", [])
    else:
        conditions = value
    for item in conditions or []:
        condition, frames = parse_manifest_condition_item(item)
        selected.setdefault(condition_token(condition), frames)
    return selected


def parse_manifest_condition_item(item: Any) -> Tuple[str, Optional[List[int]]]:
    if isinstance(item, Mapping):
        condition = item.get("condition", item.get("id", item.get("name")))
        if condition is None:
            raise ValueError(f"Manifest condition entry is missing condition: {item}")
        if item.get("frames") is not None:
            return str(condition), [int(frame) for frame in item["frames"]]
        if item.get("frame_indices") is not None:
            return str(condition), [int(frame) for frame in item["frame_indices"]]
        if item.get("frame") is not None:
            return str(condition), [int(item["frame"])]
        return str(condition), None
    return str(item), None


def make_condition_split(
    conditions: Sequence[str],
    split_ratios: Sequence[float],
    split_seed: int,
) -> Dict[str, Dict[str, Optional[List[int]]]]:
    rng = random.Random(int(split_seed))
    shuffled = list(conditions)
    rng.shuffle(shuffled)
    train_ratio, val_ratio, test_ratio = [float(v) for v in split_ratios]
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)
    else:
        n_train = max(1, n - 1)
        n_val = 0
    return {
        "train": {condition: None for condition in sorted(shuffled[:n_train], key=condition_sort_key)},
        "val": {condition: None for condition in sorted(shuffled[n_train : n_train + n_val], key=condition_sort_key)},
        "test": {condition: None for condition in sorted(shuffled[n_train + n_val :], key=condition_sort_key)},
    }


def condition_frames(mask_root: Path, camera_ids: Sequence[str], condition: str) -> List[int]:
    frame_sets: List[set[int]] = []
    for camera_id in camera_ids:
        data = load_condition_mask_data(mask_root, camera_id, condition)
        frame_sets.append(set(data["image_id_by_frame"].keys()))
    common = set.intersection(*frame_sets) if frame_sets else set()
    return sorted(common)


def build_records(
    mask_root: Path,
    camera_ids: Sequence[str],
    split_conditions: Mapping[str, Mapping[str, Optional[Sequence[int]]]],
    max_frames_per_condition: Optional[int],
) -> Dict[str, List[FrameRecord]]:
    out: Dict[str, List[FrameRecord]] = {split: [] for split in split_conditions}
    for split, condition_selection in split_conditions.items():
        for condition in sorted(condition_selection, key=condition_sort_key):
            common_frames = condition_frames(mask_root, camera_ids, condition)
            requested = condition_selection[condition]
            frames = list(common_frames if requested is None else [int(frame) for frame in requested])
            missing = sorted(set(frames) - set(common_frames))
            if missing:
                preview = ", ".join(str(frame) for frame in missing[:10])
                raise KeyError(f"Split manifest requests missing frames for condition {condition}: {preview}")
            if max_frames_per_condition is not None:
                frames = frames[: int(max_frames_per_condition)]
            out[split].extend(FrameRecord(condition_token(condition), int(frame), split) for frame in frames)
    return out


class VideoFrameReader:
    """Small sequential-reader cache for per-condition videos."""

    def __init__(self, max_open: int = 8) -> None:
        self.max_open = max(1, int(max_open))
        self._caps: "OrderedDict[Path, cv2.VideoCapture]" = OrderedDict()
        self._positions: Dict[Path, int] = {}

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._positions.clear()

    def read(self, video_path: Path, frame_index: int) -> Any:
        frame_index = int(frame_index)
        cap = self._caps.get(video_path)
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            self._caps[video_path] = cap
            self._positions[video_path] = 0
            while len(self._caps) > self.max_open:
                old_path, old_cap = self._caps.popitem(last=False)
                old_cap.release()
                self._positions.pop(old_path, None)
        else:
            self._caps.move_to_end(video_path)
        current = int(self._positions.get(video_path, 0))
        if frame_index < current:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            current = frame_index
        while current < frame_index:
            ok, _ = cap.read()
            if not ok:
                raise EOFError(f"Failed to seek frame {frame_index} from {video_path}")
            current += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            raise EOFError(f"Failed to read frame {frame_index} from {video_path}")
        self._positions[video_path] = frame_index + 1
        return frame


def write_dataset(
    *,
    output_root: Path,
    video_root: Path,
    mask_root: Path,
    camera_ids: Sequence[str],
    records_by_split: Mapping[str, Sequence[FrameRecord]],
    category_id: int,
    category_name: str,
    jpeg_quality: int,
    overwrite_images: bool,
    skip_frame_extract: bool,
) -> None:
    images_root = output_root / "images"
    annotations_root = output_root / "annotations"
    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    for split, records in records_by_split.items():
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        image_id = 0
        ann_id = 0
        mask_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        video_cache: Dict[Tuple[str, str], Path] = {}
        frame_reader = VideoFrameReader()
        dropped_annotations = 0
        try:
            for record_idx, record in enumerate(records, start=1):
                for camera_id in camera_ids:
                    key = (str(camera_id), str(record.condition))
                    mask_data = mask_cache.get(key)
                    if mask_data is None:
                        mask_data = load_condition_mask_data(mask_root, str(camera_id), record.condition)
                        mask_cache[key] = mask_data
                    source_image_id = mask_data["image_id_by_frame"].get(int(record.frame_index))
                    if source_image_id is None:
                        raise FileNotFoundError(
                            f"Frame {record.frame_index} missing in {mask_data['path']} "
                            f"for camera={camera_id} condition={record.condition}"
                        )
                    source_image = mask_data["images_by_id"][int(source_image_id)]
                    width, height = int(source_image["width"]), int(source_image["height"])
                    rel_image = (
                        Path(condition_dir_name(record.condition))
                        / f"camera_{camera_id}"
                        / f"{int(record.frame_index):06d}.jpg"
                    )
                    out_image = images_root / rel_image
                    if not skip_frame_extract and (overwrite_images or not out_image.exists()):
                        video = video_cache.get(key)
                        if video is None:
                            video = find_video(video_root, str(camera_id), record.condition)
                            video_cache[key] = video
                        out_image.parent.mkdir(parents=True, exist_ok=True)
                        frame = frame_reader.read(video, int(record.frame_index))
                        ok = cv2.imwrite(str(out_image), frame, params)
                        if not ok:
                            raise ValueError(f"Failed to write image: {out_image}")
                        height, width = frame.shape[:2]
                    images.append(
                        {
                            "id": int(image_id),
                            "file_name": str(rel_image).replace("\\", "/"),
                            "width": int(width),
                            "height": int(height),
                            "camera_id": str(camera_id),
                            "condition": condition_token(record.condition),
                            "frame_index": int(record.frame_index),
                        }
                    )
                    for ann in mask_data["anns_by_image_id"].get(int(source_image_id), []):
                        if not valid_annotation(ann):
                            dropped_annotations += 1
                            continue
                        annotations.append(
                            {
                                "id": int(ann_id),
                                "image_id": int(image_id),
                                "category_id": int(category_id),
                                "bbox": [float(v) for v in ann.get("bbox", [])],
                                "segmentation": ann["segmentation"],
                                "area": float(ann.get("area", 0.0)),
                                "iscrowd": int(ann.get("iscrowd", 0)),
                            }
                        )
                        ann_id += 1
                    image_id += 1
                if record_idx % 500 == 0 or record_idx == len(records):
                    print(f"{split}: processed {record_idx}/{len(records)} frame records", flush=True)
        finally:
            frame_reader.close()

        out_json = annotations_root / f"{split}.json"
        data = {
            "info": {
                "description": "Indoor 20260510 aligned weak ship masks",
                "split": split,
                "camera_ids": [str(camera_id) for camera_id in camera_ids],
                "mask_root": str(mask_root),
                "video_root": str(video_root),
            },
            "images": images,
            "annotations": annotations,
            "categories": [{"id": int(category_id), "name": str(category_name), "supercategory": str(category_name)}],
        }
        out_json.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        summary_rows.append(
            {
                "split": split,
                "images": len(images),
                "annotations": len(annotations),
                "dropped_degenerate_annotations": dropped_annotations,
                "path": str(out_json),
            }
        )
        print(
            f"{split}: {len(images)} images, {len(annotations)} annotations, "
            f"dropped_degenerate={dropped_annotations} -> {out_json}",
            flush=True,
        )

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "images", "annotations", "dropped_degenerate_annotations", "path"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    video_root = resolve_under_root(data_root, args.video_dir).resolve()
    mask_root = resolve_under_root(data_root, args.mask_dir).resolve()
    output_root = args.output_root.resolve()
    split_manifest = resolve_under_root(data_root, args.split_manifest).resolve()
    camera_ids = [str(camera_id) for camera_id in args.camera_ids]

    output_root.mkdir(parents=True, exist_ok=True)
    conditions = discover_conditions(mask_root, camera_ids)
    split_conditions = load_manifest(split_manifest) or make_condition_split(conditions, args.split_ratios, args.split_seed)
    selected_splits = list(dict.fromkeys(args.splits))
    selected_conditions = {split: split_conditions[split] for split in selected_splits}
    records_by_split = build_records(mask_root, camera_ids, selected_conditions, args.max_frames_per_condition)
    manifest_out = output_root / "condition_split_manifest_used.json"
    manifest_out.write_text(
        json.dumps(
            {
                split: {
                    "conditions": list(split_conditions.get(split, [])),
                    "frames_by_condition": {
                        condition: list(frames) if frames is not None else None
                        for condition, frames in split_conditions.get(split, {}).items()
                    },
                    "num_frames": len(records_by_split.get(split, [])),
                }
                for split in selected_splits
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_dataset(
        output_root=output_root,
        video_root=video_root,
        mask_root=mask_root,
        camera_ids=camera_ids,
        records_by_split=records_by_split,
        category_id=int(args.category_id),
        category_name=str(args.category_name),
        jpeg_quality=int(args.jpeg_quality),
        overwrite_images=bool(args.overwrite_images),
        skip_frame_extract=bool(args.skip_frame_extract),
    )
    print(f"MMDetection aligned dataset ready: {output_root}", flush=True)


if __name__ == "__main__":
    main()
