'Patch MMCV MultiScaleDeformableAttention to use its PyTorch fallback.'

from __future__ import annotations

import warnings


def patch_mmcv_ms_deform_attn_to_pytorch() -> None:
    import mmcv.ops.multi_scale_deform_attn as msda

    if getattr(msda, "_ship_patch_force_pytorch_msda", False):
        return

    class TorchFallbackMultiScaleDeformableAttnFunction:
        @staticmethod
        def apply(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        ):
            del value_level_start_index, im2col_step
            return msda.multi_scale_deformable_attn_pytorch(
                value,
                value_spatial_shapes,
                sampling_locations,
                attention_weights,
            )

    msda.MultiScaleDeformableAttnFunction = TorchFallbackMultiScaleDeformableAttnFunction
    msda._ship_patch_force_pytorch_msda = True
    warnings.warn(
        "Patched mmcv.ops MultiScaleDeformableAttention to use the PyTorch fallback "
        "for this process.",
        RuntimeWarning,
    )


patch_mmcv_ms_deform_attn_to_pytorch()
