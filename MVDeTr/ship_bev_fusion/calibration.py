"""Calibration and homography helpers for ship BEV mask fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np


Matrix = np.ndarray


@dataclass(frozen=True)
class CameraParameters:
    camera_id: str
    intrinsic: Matrix
    rotation: Matrix
    translation: Matrix
    distortion: Optional[Matrix] = None
    image_size: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class HomographyBundle:
    image_to_bev: Dict[str, Matrix]
    target_image_to_bev: Matrix
    world_to_bev: Matrix
    bev_size: Tuple[int, int]


def load_camera_parameters(camera_parameters_dir: Path, camera_id: str) -> CameraParameters:
    camera_parameters_dir = Path(camera_parameters_dir)
    candidates = _camera_file_candidates(camera_parameters_dir, camera_id)
    for path in candidates:
        if path.exists():
            return _parse_camera_parameters(_load_parameter_file(path), camera_id, source=path)
    split = _load_split_camera_parameters(camera_parameters_dir, camera_id)
    if split is not None:
        return _parse_camera_parameters(split, camera_id, source=camera_parameters_dir)
    for path in _global_parameter_candidates(camera_parameters_dir):
        nested = _extract_nested_camera(_load_parameter_file(path), camera_id)
        if nested is not None:
            return _parse_camera_parameters(nested, camera_id, source=path)
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find parameters for camera {camera_id}. Searched:\n{searched}")


def load_homography(camera_parameters_dir: Path, camera_id: str) -> Matrix:
    camera_parameters_dir = Path(camera_parameters_dir)
    candidates = []
    for path in (
        camera_parameters_dir / "Homography" / "homography.json",
        camera_parameters_dir / "homography.json",
    ):
        candidates.append(path)
    for stem in (
        f"camera_{camera_id}_homography",
        f"camera_{camera_id}_H",
        f"camera_{camera_id}",
        f"{camera_id}_homography",
        f"{camera_id}_H",
        f"{camera_id}",
        f"homography_{camera_id}",
        f"H_{camera_id}",
    ):
        for suffix in (".npy", ".npz", ".json", ".yaml", ".yml", ".xml", ".txt", ".csv"):
            candidates.append(camera_parameters_dir / f"{stem}{suffix}")
    for path in candidates:
        if not path.exists():
            continue
        raw = _load_parameter_file(path)
        matrix = _extract_matrix(raw, ("homography", "H", "h", "matrix", "image_to_bev", "M"), required=False)
        if matrix is None and isinstance(raw, np.ndarray):
            matrix = raw
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape == (3, 3):
                return _normalize_homography(matrix)
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find a 3x3 homography for camera {camera_id}. Searched:\n{searched}")


def load_matrix_file(path: Path) -> Matrix:
    raw = _load_parameter_file(Path(path))
    matrix = _extract_matrix(raw, ("homography", "H", "h", "matrix", "image_to_bev", "M"), required=False)
    if matrix is None and isinstance(raw, np.ndarray):
        matrix = raw
    if matrix is None:
        raise ValueError(f"Could not find a matrix in {path}")
    return np.asarray(matrix, dtype=np.float64)


def plane_world_to_image_homography(camera: CameraParameters, plane_z: float) -> Matrix:
    r = camera.rotation
    t = camera.translation.reshape(3)
    h = np.column_stack((r[:, 0], r[:, 1], r[:, 2] * float(plane_z) + t))
    return _normalize_homography(camera.intrinsic @ h)


def build_homographies(
    camera_parameters_dir: Path,
    input_camera_ids: Iterable[str],
    target_camera_id: str,
    plane_z: float,
    bev_size: Tuple[int, int],
    target_homography_direction: str = "image_to_bev",
    target_homography_output_size: Optional[Tuple[int, int]] = None,
    target_homography_space: str = "bev",
    target_homography_image_size: Optional[Tuple[int, int]] = None,
    target_world_bounds_margin: float = 0.0,
    world_to_bev_override: Optional[Matrix] = None,
    assume_world_xy_is_target_image: bool = True,
) -> HomographyBundle:
    input_camera_ids = [str(c) for c in input_camera_ids]
    target_camera_id = str(target_camera_id)
    h_target = load_homography(camera_parameters_dir, target_camera_id)
    if target_homography_direction == "bev_to_image":
        h_target = np.linalg.inv(h_target)
    elif target_homography_direction != "image_to_bev":
        raise ValueError("target_homography_direction must be 'image_to_bev' or 'bev_to_image'")
    h_target = _normalize_homography(h_target)
    if target_homography_space == "world":
        if target_homography_image_size is None:
            raise ValueError("target_homography_image_size is required when target_homography_space='world'")
        h_world_to_bev = _world_to_bev_from_target_homography(
            h_target,
            target_homography_image_size,
            bev_size,
            bounds_margin=float(target_world_bounds_margin),
        )
        h_target = _normalize_homography(h_world_to_bev @ h_target)
    elif target_homography_space != "bev":
        raise ValueError("target_homography_space must be 'bev' or 'world'")
    elif target_homography_output_size is not None:
        ref_w, ref_h = target_homography_output_size
        bev_w, bev_h = bev_size
        scale = np.array([[bev_w / ref_w, 0.0, 0.0], [0.0, bev_h / ref_h, 0.0], [0.0, 0.0, 1.0]])
        h_target = _normalize_homography(scale @ h_target)

    if target_homography_space == "world":
        pass
    elif world_to_bev_override is not None:
        h_world_to_bev = _normalize_homography(np.asarray(world_to_bev_override, dtype=np.float64))
    else:
        try:
            target_params = load_camera_parameters(camera_parameters_dir, target_camera_id)
        except FileNotFoundError:
            if not assume_world_xy_is_target_image:
                raise
            h_world_to_bev = h_target
        else:
            h_world_to_target_img = plane_world_to_image_homography(target_params, plane_z)
            h_world_to_bev = _normalize_homography(h_target @ h_world_to_target_img)

    image_to_bev: Dict[str, Matrix] = {}
    for camera_id in input_camera_ids:
        params = load_camera_parameters(camera_parameters_dir, camera_id)
        h_world_to_img = plane_world_to_image_homography(params, plane_z)
        image_to_bev[camera_id] = _normalize_homography(h_world_to_bev @ np.linalg.inv(h_world_to_img))
    return HomographyBundle(
        image_to_bev=image_to_bev,
        target_image_to_bev=h_target,
        world_to_bev=h_world_to_bev,
        bev_size=bev_size,
    )


def build_old_batch_concat_homographies(
    camera_parameters_dir: Path,
    input_camera_ids: Iterable[str],
    target_camera_id: str,
    plane_z: float,
    bev_size: Tuple[int, int],
    old_bev_canvas: Tuple[int, int] = (2600, 1600),
    old_world_origin: Tuple[float, float] = (350.0, 250.0),
    old_bev_resolution: float = 1.0,
    old_homography_invert: bool = True,
) -> HomographyBundle:
    'Reproduce the Mask2BEV batch-concat projection, then resize to model BEV.'
    input_camera_ids = [str(c) for c in input_camera_ids]
    target_camera_id = str(target_camera_id)
    canvas_w, canvas_h = (int(old_bev_canvas[0]), int(old_bev_canvas[1]))
    bev_w, bev_h = (int(bev_size[0]), int(bev_size[1]))
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError(f"old_bev_canvas must be positive, got {old_bev_canvas}")
    if bev_w <= 0 or bev_h <= 0:
        raise ValueError(f"bev_size must be positive, got {bev_size}")

    old_world_to_canvas = np.array(
        [
            [float(old_bev_resolution), 0.0, float(old_world_origin[0])],
            [0.0, float(old_bev_resolution), float(old_world_origin[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    canvas_to_model = np.array(
        [[bev_w / canvas_w, 0.0, 0.0], [0.0, bev_h / canvas_h, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    h_world_to_bev = _normalize_homography(canvas_to_model @ old_world_to_canvas)

    h_target = load_homography(camera_parameters_dir, target_camera_id)
    if old_homography_invert:
        h_target = np.linalg.inv(h_target)
    h_target = _normalize_homography(h_world_to_bev @ h_target)

    image_to_bev: Dict[str, Matrix] = {}
    for camera_id in input_camera_ids:
        params = load_camera_parameters(camera_parameters_dir, camera_id)
        h_world_to_img = plane_world_to_image_homography(params, plane_z)
        image_to_bev[camera_id] = _normalize_homography(h_world_to_bev @ np.linalg.inv(h_world_to_img))

    return HomographyBundle(
        image_to_bev=image_to_bev,
        target_image_to_bev=h_target,
        world_to_bev=h_world_to_bev,
        bev_size=bev_size,
    )


def _world_to_bev_from_target_homography(
    target_image_to_world: Matrix,
    target_image_size: Tuple[int, int],
    bev_size: Tuple[int, int],
    bounds_margin: float = 0.0,
) -> Matrix:
    image_w, image_h = target_image_size
    bev_w, bev_h = bev_size
    corners = np.array(
        [[0.0, 0.0, 1.0], [image_w - 1.0, 0.0, 1.0], [image_w - 1.0, image_h - 1.0, 1.0], [0.0, image_h - 1.0, 1.0]]
    ).T
    world = target_image_to_world @ corners
    world_xy = world[:2] / world[2:]
    min_xy = world_xy.min(axis=1)
    max_xy = world_xy.max(axis=1)
    span = max_xy - min_xy
    if np.any(span <= 1e-9):
        raise ValueError(f"Degenerate target homography world bounds: min={min_xy}, max={max_xy}")
    if bounds_margin < 0:
        raise ValueError(f"bounds_margin must be non-negative, got {bounds_margin}")
    if bounds_margin > 0:
        padding = span * float(bounds_margin)
        min_xy = min_xy - padding
        max_xy = max_xy + padding
        span = max_xy - min_xy
    sx = (bev_w - 1.0) / span[0]
    sy = (bev_h - 1.0) / span[1]


    return np.array([[sx, 0.0, -min_xy[0] * sx], [0.0, -sy, max_xy[1] * sy], [0.0, 0.0, 1.0]], dtype=np.float64)


def undistort_mask(mask: Matrix, camera: CameraParameters) -> Matrix:
    if camera.distortion is None or not np.any(np.abs(camera.distortion) > 1e-12):
        return mask
    return cv2.undistort(mask, camera.intrinsic, camera.distortion)


def warp_mask_to_bev(mask: Matrix, homography: Matrix, bev_size: Tuple[int, int]) -> Matrix:
    return cv2.warpPerspective(
        mask,
        homography,
        tuple(int(v) for v in bev_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _camera_file_candidates(root: Path, camera_id: str) -> List[Path]:
    names = [
        f"camera_{camera_id}",
        f"camera_{camera_id}_params",
        f"camera_{camera_id}_parameters",
        str(camera_id),
        f"cam_{camera_id}",
        f"Camera_{camera_id}",
    ]
    candidates: List[Path] = []
    for base in (root, root / f"camera_{camera_id}", root / str(camera_id)):
        for name in names:
            for suffix in (".npz", ".npy", ".json", ".yaml", ".yml", ".xml", ".txt", ".csv"):
                candidates.append(base / f"{name}{suffix}")
    return candidates


def _load_split_camera_parameters(root: Path, camera_id: str) -> Optional[Dict[str, Matrix]]:
    camera_id = "GT" if str(camera_id).lower() == "gt" else str(camera_id)
    intrinsic_files = [
        root / "Intrinsic" / f"camera_{camera_id}_intrinsic_params.json",
        root / "Intrinsic" / f"camera_{camera_id}_intrinsics.json",
        root / "Intrinsic" / f"{camera_id}_intrinsic_params.json",
        root / f"camera_{camera_id}_intrinsic_params.npz",
        root / f"camera_{camera_id}_intrinsic_params.json",
        root / f"camera_{camera_id}_intrinsics.npz",
        root / f"camera_{camera_id}_intrinsics.json",
        root / f"{camera_id}_intrinsic_params.npz",
        root / f"{camera_id}_intrinsic_params.json",
    ]
    extrinsic_files = [
        root / "Extrinsic" / f"camera_{camera_id}_extrinsic_params.json",
        root / "Extrinsic" / f"camera_{camera_id}_extrinsics.json",
        root / "Extrinsic" / f"{camera_id}_extrinsic_params.json",
        root / f"camera_{camera_id}_extrinsic_params.npz",
        root / f"camera_{camera_id}_extrinsic_params.json",
        root / f"camera_{camera_id}_extrinsics.npz",
        root / f"camera_{camera_id}_extrinsics.json",
        root / f"{camera_id}_extrinsic_params.npz",
        root / f"{camera_id}_extrinsic_params.json",
    ]
    intrinsic_path = next((p for p in intrinsic_files if p.exists()), None)
    extrinsic_path = next((p for p in extrinsic_files if p.exists()), None)
    if intrinsic_path is None or extrinsic_path is None:
        return None
    intrinsic = _load_parameter_file(intrinsic_path)
    extrinsic = _load_parameter_file(extrinsic_path)
    if not isinstance(intrinsic, Mapping) or not isinstance(extrinsic, Mapping):
        return None
    merged: Dict[str, Matrix] = {}
    merged.update(intrinsic)
    merged.update(extrinsic)
    return merged


def _global_parameter_candidates(root: Path) -> List[Path]:
    candidates = []
    for name in ("camera_parameters", "cameras", "calibration", "params"):
        for suffix in (".npz", ".json", ".yaml", ".yml", ".xml"):
            candidates.append(root / f"{name}{suffix}")
    return [path for path in candidates if path.exists()]


def _load_parameter_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    if suffix == ".json":
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in (".yaml", ".yml"):
        import yaml

        try:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            return _load_opencv_filestorage(path)
    if suffix == ".xml":
        return _load_opencv_filestorage(path)
    if suffix in (".txt", ".csv"):
        return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    raise ValueError(f"Unsupported camera parameter file: {path}")


def _load_opencv_filestorage(path: Path) -> Dict[str, Matrix]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"OpenCV FileStorage failed to open: {path}")
    keys = (
        "K",
        "intrinsic",
        "intrinsics",
        "camera_matrix",
        "mtx",
        "distortion",
        "dist",
        "dist_coeffs",
        "distCoeffs",
        "D",
        "extrinsic",
        "extrinsics",
        "Tcw",
        "world_to_camera",
        "R",
        "rotation",
        "rotation_matrix",
        "t",
        "T",
        "translation",
        "tvec",
        "rvec",
        "homography",
        "H",
        "image_to_bev",
        "M",
    )
    data: Dict[str, Matrix] = {}
    for key in keys:
        node = fs.getNode(key)
        if not node.empty():
            data[key] = node.mat()
    fs.release()
    if not data:
        raise ValueError(f"No recognized OpenCV FileStorage keys in {path}")
    return data


def _parse_camera_parameters(raw: Any, camera_id: str, source: Path) -> CameraParameters:
    if isinstance(raw, np.ndarray) and raw.shape == (3, 3):
        raise ValueError(f"{source} only contains a 3x3 matrix; expected intrinsics and extrinsics.")
    mapping = raw.item() if isinstance(raw, np.ndarray) and raw.shape == () else raw
    if not isinstance(mapping, Mapping):
        raise ValueError(f"Unsupported camera parameter structure in {source}")

    intrinsic = _extract_matrix(mapping, ("K", "intrinsic", "intrinsics", "camera_matrix", "mtx"), required=True)
    distortion = _extract_matrix(mapping, ("distortion", "dist", "dist_coeffs", "distCoeffs", "D", "k"), required=False)
    extrinsic = _extract_matrix(mapping, ("extrinsic", "extrinsics", "extrinsic_matrix", "Tcw", "world_to_camera"), required=False)
    if extrinsic is not None:
        extrinsic = np.asarray(extrinsic, dtype=np.float64)
        if extrinsic.shape == (4, 4):
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
        elif extrinsic.shape == (3, 4):
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
        else:
            raise ValueError(f"Extrinsic in {source} must be 3x4 or 4x4, got {extrinsic.shape}")
    else:
        rotation = _extract_matrix(mapping, ("R", "rotation", "rot", "rotation_matrix"), required=False)
        translation = _extract_matrix(mapping, ("t", "T", "translation", "trans", "tvec"), required=False)
        rvec = _extract_matrix(mapping, ("rvec", "rotation_vector"), required=False)
        if rotation is None and rvec is not None:
            rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
        if rotation is None or translation is None:
            raise ValueError(f"{source} must contain either extrinsic or R/t parameters.")

    return CameraParameters(
        camera_id=str(camera_id),
        intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        rotation=np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        translation=np.asarray(translation, dtype=np.float64).reshape(3),
        distortion=None if distortion is None else np.asarray(distortion, dtype=np.float64).reshape(-1),
        image_size=_extract_image_size(mapping),
    )


def _extract_matrix(mapping: Any, names: Iterable[str], required: bool) -> Optional[Matrix]:
    if isinstance(mapping, np.ndarray):
        return mapping
    if not isinstance(mapping, Mapping):
        if required:
            raise ValueError(f"Expected mapping while looking for {tuple(names)}")
        return None
    lowered = {str(k).lower(): k for k in mapping.keys()}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None and mapping[key] is not None:
            return np.asarray(mapping[key], dtype=np.float64)
    if required:
        raise KeyError(f"Missing required parameter. Tried keys: {tuple(names)}")
    return None


def _extract_nested_camera(raw: Any, camera_id: str) -> Optional[Any]:
    if not isinstance(raw, Mapping):
        return None
    lowered = {str(k).lower(): k for k in raw.keys()}
    for name in (str(camera_id), f"camera_{camera_id}", f"cam_{camera_id}", f"Camera_{camera_id}"):
        key = lowered.get(name.lower())
        if key is not None:
            return raw[key]
    return None


def _extract_image_size(mapping: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    lowered = {str(k).lower(): k for k in mapping.keys()}
    for name in ("image_size", "imagesize", "imgsize", "size", "resolution"):
        key = lowered.get(name)
        if key is not None:
            values = np.asarray(mapping[key]).reshape(-1)
            if len(values) >= 2:


                if name in {"imagesize", "imgsize"}:
                    return int(values[1]), int(values[0])
                return int(values[0]), int(values[1])
    width_key = lowered.get("width")
    height_key = lowered.get("height")
    if width_key is not None and height_key is not None:
        return int(mapping[width_key]), int(mapping[height_key])
    return None


def _normalize_homography(h: Matrix) -> Matrix:
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError(f"Homography contains non-finite values:\n{h}")
    scale = h[2, 2] if abs(h[2, 2]) > 1e-12 else np.linalg.norm(h)
    if abs(scale) < 1e-12:
        raise ValueError(f"Degenerate homography:\n{h}")
    return h / scale
