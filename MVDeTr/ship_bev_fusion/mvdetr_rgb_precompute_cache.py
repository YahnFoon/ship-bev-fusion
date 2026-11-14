"""Precompute resized RGB inputs and BEV targets for direct MVDeTr training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .mvdetr_rgb_dataset import ShipMVDeTrRGBDataset, resolve_under_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute direct-RGB MVDeTr cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--limit-frames-per-condition", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset_cfg = dict(cfg["dataset"])
    cache_value = dataset_cfg.get("cache_dir") or "outputs/mvdetr_rgb_vs_mask_pipeline/rgb_cache_512x288"
    dataset_cfg["cache_dir"] = cache_value
    dataset_cfg["use_cache"] = False
    if args.limit_frames_per_condition is not None:
        dataset_cfg["max_frames_per_condition"] = args.limit_frames_per_condition
    cache_dir = resolve_under_root(Path(dataset_cfg["root"]), cache_value)
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in args.splits:
        dataset = None
        try:
            dataset = ShipMVDeTrRGBDataset(split=split, **dataset_cfg)
        except ValueError as exc:
            if "No records found" not in str(exc):
                raise
            print(f"{split}: no records, skipped", flush=True)
            continue
        try:
            last_condition = None
            for idx, record in enumerate(dataset.records, start=1):
                if last_condition is not None and str(record.condition) != last_condition and dataset.video_store is not None:
                    dataset.video_store.close()
                last_condition = str(record.condition)
                out_dir = cache_dir / str(record.condition)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{record.frame_index:06d}.npz"
                if args.force or not out_path.exists():
                    inputs, target, affine = dataset.build_sample_numpy(record)
                    np.savez_compressed(
                        out_path,
                        inputs=inputs.astype(np.float16),
                        target=target.astype(np.float16),
                        affine=affine.astype(np.float32),
                    )
                total += 1
                if idx % 50 == 0 or idx >= len(dataset.records):
                    print(f"{split}: cached {idx}/{len(dataset.records)}", flush=True)
        finally:
            if dataset is not None and dataset.video_store is not None:
                dataset.video_store.close()
    print(f"RGB MVDeTr cache ready: {cache_dir} ({total} split records)")


if __name__ == "__main__":
    main()
