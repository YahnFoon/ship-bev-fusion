"""Config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    dataset_cfg = cfg.get("dataset")
    if isinstance(dataset_cfg, dict) and "root" in dataset_cfg:
        root = Path(dataset_cfg["root"])
        if not root.is_absolute():
            dataset_cfg["root"] = str((Path(__file__).resolve().parents[2] / root).resolve())
    if "output_dir" in cfg:
        output_dir = Path(cfg["output_dir"])
        if not output_dir.is_absolute():
            cfg["output_dir"] = str((Path(__file__).resolve().parents[2] / output_dir).resolve())
    return cfg


def require_openmmlab_env(*, allow_deploy: bool = False) -> None:
    import importlib.util
    missing = [name for name in ("torch", "mmcv", "mmengine", "mmdet")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing dependencies: " + ", ".join(missing) + ". See docs/INSTALL.md.")
