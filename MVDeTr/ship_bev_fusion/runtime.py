"""Runtime for RGB-to-BEV inference and model export."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
import torch


RGBInput = Union[np.ndarray, torch.Tensor]
MODELOPT_STATE_DICT_TRANSFORM = "modelopt_state_dict"
MODELOPT_BASE_MASK2FORMER_KEY = "modelopt_base_serialized_mask2former"
MODELOPT_BASE_FUSION_KEY = "modelopt_base_serialized_fusion_model"


class EndToEndBEVModule(torch.nn.Module):
    """Tensor-only RGB-to-BEV graph suitable for ONNX/TensorRT export."""

    def __init__(
        self,
        mask_model: torch.nn.Module,
        fusion_model: torch.nn.Module,
        *,
        background_weight: float = 0.25,
        calibration_image_size: tuple = (2560, 1440),
    ) -> None:
        super().__init__()
        self.mask_model = mask_model
        self.fusion_model = fusion_model
        self.background_weight = float(background_weight)
        self.calibration_image_size = (int(calibration_image_size[0]), int(calibration_image_size[1]))
        self.register_buffer("mask_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("mask_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward_with_intermediates(
        self,
        rgb01: torch.Tensor,
        base_projection_mats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if rgb01.ndim != 5 or rgb01.shape[2] != 3:
            raise ValueError(f"rgb01 must have shape [B,N,3,H,W], got {tuple(rgb01.shape)}")
        b, n, _, h, w = rgb01.shape
        flat_rgb = rgb01.reshape(b * n, 3, h, w)
        processed_inputs = (flat_rgb * 255.0 - self.mask_mean) / self.mask_std
        features = self.mask_model.extract_feat(processed_inputs)
        all_mask_cls, all_mask_pred = self.mask_model.panoptic_head(features, [])
        mask_cls = all_mask_cls[-1]
        mask_pred = torch.nn.functional.interpolate(
            all_mask_pred[-1],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        scores = mask_cls.softmax(dim=-1)[..., 0]
        soft_flat = torch.einsum("bq,bqhw->bhw", scores, mask_pred.sigmoid()).clamp(0.0, 1.0)
        soft_masks = soft_flat[:, None].reshape(b, n, 1, h, w)

        roi_weight = self.background_weight + (1.0 - self.background_weight) * soft_masks
        overlay = (rgb01 * roi_weight - self.rgb_mean) / self.rgb_std
        calibration_w, calibration_h = self.calibration_image_size
        affine_single = rgb01.new_tensor(
            [[w / calibration_w, 0.0, 0.0], [0.0, h / calibration_h, 0.0], [0.0, 0.0, 1.0]]
        )
        affine = affine_single.view(1, 1, 3, 3).expand(b, n, -1, -1)
        fusion = self.fusion_model.forward_with_projection_mats(overlay, affine, base_projection_mats)
        fusion["soft_masks"] = soft_masks
        fusion["roi_rgb_inputs"] = overlay
        return fusion

    def forward(self, rgb01: torch.Tensor, base_projection_mats: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(rgb01, base_projection_mats)["logits"]


def is_modelopt_state_dict_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    recipe = checkpoint.get("compression_recipe", {})
    return (
        isinstance(recipe, Mapping)
        and str(recipe.get("transform", "")) == MODELOPT_STATE_DICT_TRANSFORM
    )


def rebind_fusion_projection_geometry(
    fusion_model: torch.nn.Module,
    dataset: Any,
    z_mm: float,
) -> None:
    """Bind a restored fusion model to the active dataset calibration and height."""
    projection_dataset = getattr(dataset, "base_dataset", dataset)
    if not hasattr(fusion_model, "_projection_matrices_for_height"):
        raise TypeError("Restored fusion model cannot rebuild projection matrices")
    if not hasattr(fusion_model, "proj_mats"):
        raise TypeError("Restored fusion model has no projection matrices")

    target_z_mm = float(z_mm)
    fusion_model._projection_dataset = projection_dataset
    projection_mats = fusion_model._projection_matrices_for_height(target_z_mm)
    current_projection_mats = fusion_model.proj_mats
    fusion_model.proj_mats = projection_mats.to(
        device=current_projection_mats.device,
        dtype=current_projection_mats.dtype,
    )
    fusion_model._projection_z_mm = target_z_mm


def restore_modelopt_state_dict_models(
    checkpoint: Mapping[str, Any],
    mask_model: torch.nn.Module,
    fusion_model: torch.nn.Module,
    device: Union[str, torch.device],
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Restore ModelOpt's dynamic structure before loading ordinary state dicts."""
    if "modelopt_state" not in checkpoint:
        raise ValueError("ModelOpt checkpoint is missing modelopt_state")
    if "mask2former" not in checkpoint or "fusion_model" not in checkpoint:
        raise ValueError("ModelOpt checkpoint is missing component state dicts")

    import modelopt.torch.opt as mto

    resolved_device = torch.device(device)
    has_base_mask = MODELOPT_BASE_MASK2FORMER_KEY in checkpoint
    has_base_fusion = MODELOPT_BASE_FUSION_KEY in checkpoint
    if has_base_mask != has_base_fusion:
        raise ValueError("ModelOpt checkpoint has an incomplete serialized base structure")
    if has_base_mask:
        mask_model = copy.deepcopy(checkpoint[MODELOPT_BASE_MASK2FORMER_KEY]).to(resolved_device)
        fusion_model = copy.deepcopy(checkpoint[MODELOPT_BASE_FUSION_KEY]).to(resolved_device)
    wrapper = EndToEndBEVModule(mask_model, fusion_model).to(resolved_device)
    wrapper = mto.restore_from_modelopt_state(wrapper, checkpoint["modelopt_state"])
    missing, unexpected = wrapper.mask_model.load_state_dict(checkpoint["mask2former"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ModelOpt Mask2Former state mismatch after structure restoration: "
            f"missing={missing} unexpected={unexpected}"
        )
    wrapper.fusion_model.load_state_dict(checkpoint["fusion_model"], strict=True)
    return wrapper.mask_model.to(resolved_device), wrapper.fusion_model.to(resolved_device)


class EndToEndBEVRuntime:
    """Expose Mask2Former and DeformTrans as one deployment-facing model."""

    def __init__(
        self,
        mask_model: torch.nn.Module,
        fusion_model: torch.nn.Module,
        *,
        device: Union[str, torch.device] = "cuda",
        background_weight: float = 0.25,
        calibration_image_size: tuple = (2560, 1440),
    ) -> None:
        self.device = torch.device(device)
        self.mask_model = mask_model.to(self.device).eval()
        self.fusion_model = fusion_model.to(self.device).eval()
        self.background_weight = float(background_weight)
        self.calibration_image_size = (int(calibration_image_size[0]), int(calibration_image_size[1]))
        self.module = EndToEndBEVModule(
            self.mask_model,
            self.fusion_model,
            background_weight=self.background_weight,
            calibration_image_size=self.calibration_image_size,
        ).to(self.device).eval()

    def _rgb01(self, rgb_views: RGBInput) -> torch.Tensor:
        tensor = torch.as_tensor(rgb_views)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 5 or tensor.shape[2] != 3:
            raise ValueError(f"rgb_views must have shape [B,N,3,H,W] or [N,3,H,W], got {tuple(tensor.shape)}")
        tensor = tensor.to(device=self.device, dtype=torch.float32, non_blocking=True)
        if float(tensor.detach().amax()) > 1.5:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    def _forward(self, rgb_views: RGBInput, water_level_mm: float, *, profile: bool = False) -> Dict[str, torch.Tensor]:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if profile else None
        if events is not None:
            events[0].record()
        rgb01 = self._rgb01(rgb_views)
        b = rgb01.shape[0]
        base_projection_mats = self.fusion_model._projection_matrices_for_height(float(water_level_mm))
        base_projection_mats = base_projection_mats.to(device=self.device, dtype=rgb01.dtype)
        base_projection_mats = base_projection_mats.unsqueeze(0).expand(b, -1, -1, -1)
        if events is not None:
            events[1].record()
        fusion = self.module.forward_with_intermediates(rgb01, base_projection_mats)
        if events is not None:
            events[3].record()
        if events is not None:
            events[4].record()
            events[4].synchronize()
            fusion["_timings_ms"] = {
                "projection_matrix_setup": events[0].elapsed_time(events[1]),
                "complete_rgb_to_bev": events[1].elapsed_time(events[3]),
                "output_bookkeeping": events[3].elapsed_time(events[4]),
                "total": events[0].elapsed_time(events[4]),
            }
        return fusion

    @torch.inference_mode()
    def predict(
        self,
        rgb_views: RGBInput,
        water_level_mm: float,
        *,
        return_cpu: bool = False,
    ) -> torch.Tensor:
        logits = self._forward(rgb_views, water_level_mm)["logits"]
        return logits.cpu() if return_cpu else logits

    @torch.inference_mode()
    def predict_with_intermediates(self, rgb_views: RGBInput, water_level_mm: float) -> Mapping[str, torch.Tensor]:
        return self._forward(rgb_views, water_level_mm)

    @torch.inference_mode()
    def profile_stages(self, rgb_views: RGBInput, water_level_mm: float) -> Mapping[str, float]:
        return self._forward(rgb_views, water_level_mm, profile=True)["_timings_ms"]


class TensorRTEngineRunner:
    """Run the exported two-input engine with CUDA torch tensors and no CPU copy."""

    def __init__(self, engine_path: Union[str, "Path"], rgb: torch.Tensor, projection_mats: torch.Tensor) -> None:
        import tensorrt as trt

        engine_path = Path(engine_path)
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        for name, tensor in {"rgb_views": rgb, "base_projection_mats": projection_mats}.items():
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                raise KeyError(f"TensorRT engine has no input named {name!r}")
            accepted = self.context.set_input_shape(name, tuple(tensor.shape))
            if accepted is False:
                raise RuntimeError(f"TensorRT rejected shape {tuple(tensor.shape)} for {name}")
        output_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        ]
        if output_names != ["bev_logits"]:
            raise RuntimeError(f"Expected one bev_logits output, got {output_names}")
        output_shape = tuple(self.context.get_tensor_shape("bev_logits"))
        output_dtype = torch.from_numpy(
            np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype("bev_logits")))
        ).dtype
        self.output = torch.empty(output_shape, dtype=output_dtype, device=rgb.device)

    def predict(self, rgb: torch.Tensor, projection_mats: torch.Tensor) -> torch.Tensor:
        tensors = {"rgb_views": rgb, "base_projection_mats": projection_mats, "bev_logits": self.output}
        for name, tensor in tensors.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        stream = torch.cuda.current_stream(rgb.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.output
