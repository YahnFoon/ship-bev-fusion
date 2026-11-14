"""Streaming COCO masks, FP16 logits, and aligned target archives."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from pycocotools import mask as mask_utils

def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    rle["size"] = [int(v) for v in rle["size"]]
    return rle

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))

def score_from_logits(logits_hw: np.ndarray, pred_mask: np.ndarray) -> float:
    """Return a COCO-style confidence score for a single thresholded BEV mask."""
    prob = sigmoid_np(logits_hw)
    mask = np.asarray(pred_mask).astype(bool)
    if mask.any():
        return float(prob[mask].mean())
    return float(prob.max(initial=0.0))

class StreamingSplitWriter:
    """Write logits/targets per condition instead of holding the full split."""

    def __init__(
        self,
        *,
        split: str,
        output_dir: Path,
        threshold: float,
        run_name: str,
        mode: str,
        shared_target_root: Optional[Path] = None,
    ) -> None:
        self.split = split
        self.output_dir = output_dir
        self.threshold = float(threshold)
        self.run_name = run_name
        self.mode = mode
        self.pred_root = output_dir / "predictions" / split
        self.logits_root = self.pred_root / "pred_logits_fp16"
        self.target_root = (
            Path(shared_target_root) / split / "target_masks_uint8"
            if shared_target_root is not None
            else output_dir / "targets" / split / "target_masks_uint8"
        )
        self.logits_root.mkdir(parents=True, exist_ok=True)
        self.target_root.mkdir(parents=True, exist_ok=True)
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.manifest_rows: List[Dict[str, Any]] = []
        self._bucket: Optional[Dict[str, List[np.ndarray]]] = None
        self._condition: Optional[str] = None
        self._flushed_conditions: set[str] = set()
        self._ann_id = 1
        self._image_id = 1
        self._height: Optional[int] = None
        self._width: Optional[int] = None

    @property
    def num_manifest_rows(self) -> int:
        bucket_count = 0 if self._bucket is None else len(self._bucket["frames"])
        return len(self.manifest_rows) + bucket_count

    def append(self, *, condition: str, frame_index: int, logits_hw: np.ndarray, target_hw: np.ndarray) -> None:
        condition = str(condition)
        if self._condition is None:
            self._condition = condition
            self._bucket = {"frames": [], "logits": [], "targets": []}
        elif condition != self._condition:
            self.flush_condition()
            if condition in self._flushed_conditions:
                raise ValueError(
                    f"Condition {condition} appeared again after being flushed; "
                    "streaming export expects records sorted by condition."
                )
            self._condition = condition
            self._bucket = {"frames": [], "logits": [], "targets": []}
        if self._bucket is None:
            raise RuntimeError("Streaming writer bucket was not initialized")
        self._bucket["frames"].append(np.asarray(int(frame_index), dtype=np.int32))
        self._bucket["logits"].append(logits_hw.astype(np.float16, copy=False))
        self._bucket["targets"].append((target_hw > 0.5).astype(np.uint8, copy=False))

    def flush_condition(self) -> None:
        if self._condition is None or self._bucket is None:
            return
        condition = self._condition
        bucket = self._bucket
        order = np.argsort(np.asarray(bucket["frames"], dtype=np.int32))
        frames = np.asarray(bucket["frames"], dtype=np.int32)[order]
        logits = np.stack(bucket["logits"], axis=0)[order]
        targets = np.stack(bucket["targets"], axis=0)[order]
        if logits.ndim != 3:
            raise ValueError(f"Expected logits [N,H,W] for condition={condition}, got {logits.shape}")
        self._height, self._width = int(logits.shape[1]), int(logits.shape[2])
        logits_path = self.logits_root / f"{condition}.npz"
        target_path = self.target_root / f"{condition}.npz"
        np.savez_compressed(logits_path, frame_indices=frames, logits=logits)
        if target_path.exists():
            with np.load(target_path, allow_pickle=False) as cached:
                cached_frames = np.asarray(cached["frame_indices"], dtype=np.int32)
                cached_targets = np.asarray(cached["masks"], dtype=np.uint8)
            if not np.array_equal(cached_frames, frames) or not np.array_equal(cached_targets, targets):
                raise ValueError(f"Shared target cache mismatch: {target_path}")
        else:
            temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, frame_indices=frames, masks=targets)
                os.replace(temporary, target_path)
            finally:
                temporary.unlink(missing_ok=True)

        logit_thr = 0.0 if self.threshold == 0.5 else float(np.log(self.threshold / max(1.0 - self.threshold, 1e-12)))
        for row_idx, frame_index in enumerate(frames.tolist()):
            pred_mask = (logits[row_idx] >= logit_thr).astype(np.uint8)
            self.images.append(
                {
                    "id": self._image_id,
                    "file_name": f"{condition}/{int(frame_index):06d}.png",
                    "width": self._width,
                    "height": self._height,
                    "condition": str(condition),
                    "frame_index": int(frame_index),
                }
            )
            rle = encode_binary_mask(pred_mask)
            area = float(mask_utils.area(rle))
            if area > 0:
                score = score_from_logits(logits[row_idx].astype(np.float32), pred_mask)
                self.annotations.append(
                    {
                        "id": self._ann_id,
                        "image_id": self._image_id,
                        "category_id": 1,
                        "segmentation": rle,
                        "area": area,
                        "bbox": [float(v) for v in mask_utils.toBbox(rle)],
                        "score": score,
                        "iscrowd": 0,
                    }
                )
                self._ann_id += 1
            self.manifest_rows.append(
                {
                    "split": self.split,
                    "condition": condition,
                    "frame_index": int(frame_index),
                    "image_id": self._image_id,
                    "npz_index": row_idx,
                    "logits_npz": str(logits_path),
                    "target_npz": str(target_path),
                }
            )
            self._image_id += 1
        self._flushed_conditions.add(condition)
        self._condition = None
        self._bucket = None

    def finalize(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.flush_condition()
        coco_info = {
            "mode": self.mode,
            "run_name": self.run_name,
            "split": self.split,
            "threshold": float(self.threshold),
            "logit_threshold": 0.0 if self.threshold == 0.5 else float(np.log(self.threshold / max(1.0 - self.threshold, 1e-12))),
            "score": "mean sigmoid probability inside the thresholded predicted mask",
            "height": self._height,
            "width": self._width,
        }
        coco = {
            "info": coco_info,
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": 1, "name": "ship"}],
        }
        write_json(self.pred_root / "pred_masks_thr0p5_coco.json", coco)
        write_csv(
            self.pred_root / "manifest.csv",
            self.manifest_rows,
            ["split", "condition", "frame_index", "image_id", "npz_index", "logits_npz", "target_npz"],
        )
        return self.manifest_rows, coco_info
