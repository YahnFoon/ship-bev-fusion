# Modified for Ship BEV Fusion: PyTorch attention fallback and export support.
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Imported via https://github.com/hou-yz/MVDeTr (66cae1538206).
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def _onnx_constant(graph, value, *, dtype=torch.long):
    return graph.op("Constant", value_t=torch.as_tensor(value, dtype=dtype))


def _onnx_shape_dim(graph, value, index):
    return graph.op(
        "Gather",
        graph.op("Shape", value),
        _onnx_constant(graph, index),
        axis_i=0,
    )


def _onnx_shape_vector(graph, *dimensions):
    values = []
    for dimension in dimensions:
        if isinstance(dimension, int):
            values.append(_onnx_constant(graph, [dimension]))
        else:
            values.append(
                graph.op("Unsqueeze", dimension, _onnx_constant(graph, [0]))
            )
    return graph.op("Concat", *values, axis_i=0)


def ms_deform_attn_onnx_symbolic(
        graph, value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights, im2col_step):
    """Decompose the project CUDA op into standard fixed-shape ONNX operators."""
    del value_level_start_index, im2col_step
    from torch.onnx import symbolic_helper

    spatial_shapes = symbolic_helper._maybe_get_const(value_spatial_shapes, "t")
    if isinstance(spatial_shapes, torch.Tensor):
        spatial_shapes = spatial_shapes.detach().cpu().tolist()
    if not isinstance(spatial_shapes, (list, tuple)):
        raise RuntimeError(
            "MSDeformAttn ONNX export spatial-shape inspection failed: "
            f"spatial_shapes_type={type(spatial_shapes).__name__}, "
            f"spatial_shapes={spatial_shapes}"
        )

    batch = _onnx_shape_dim(graph, value, 0)
    num_heads = _onnx_shape_dim(graph, value, 2)
    channels_per_head = _onnx_shape_dim(graph, value, 3)
    num_queries = _onnx_shape_dim(graph, sampling_locations, 1)
    num_points = _onnx_shape_dim(graph, sampling_locations, 4)
    batch_heads = graph.op("Mul", batch, num_heads)
    heads_channels = graph.op("Mul", num_heads, channels_per_head)
    level_points = graph.op(
        "Mul", _onnx_constant(graph, len(spatial_shapes)), num_points
    )

    sampled_levels = []
    start = 0
    for level, (height, width) in enumerate(spatial_shapes):
        height, width = int(height), int(width)
        end = start + height * width
        value_level = graph.op(
            "Slice",
            value,
            _onnx_constant(graph, [start]),
            _onnx_constant(graph, [end]),
            _onnx_constant(graph, [1]),
        )
        value_level = graph.op(
            "Reshape",
            value_level,
            _onnx_shape_vector(graph, batch, height * width, heads_channels),
        )
        value_level = graph.op("Transpose", value_level, perm_i=[0, 2, 1])
        value_level = graph.op(
            "Reshape",
            value_level,
            _onnx_shape_vector(
                graph, batch_heads, channels_per_head, height, width
            ),
        )

        sampling_grid = graph.op(
            "Gather",
            sampling_locations,
            _onnx_constant(graph, level),
            axis_i=3,
        )
        sampling_grid = graph.op("Transpose", sampling_grid, perm_i=[0, 2, 1, 3, 4])
        sampling_grid = graph.op(
            "Reshape",
            sampling_grid,
            _onnx_shape_vector(graph, batch_heads, num_queries, num_points, 2),
        )
        sampling_grid = graph.op(
            "Sub",
            graph.op("Mul", sampling_grid, _onnx_constant(graph, 2.0, dtype=torch.float32)),
            _onnx_constant(graph, 1.0, dtype=torch.float32),
        )
        sampled = graph.op(
            "GridSample",
            value_level,
            sampling_grid,
            align_corners_i=0,
            mode_s="bilinear",
            padding_mode_s="zeros",
        )
        sampled_levels.append(
            graph.op("Unsqueeze", sampled, _onnx_constant(graph, [-2]))
        )
        start = end

    sampled = graph.op("Concat", *sampled_levels, axis_i=-2)
    sampled = graph.op(
        "Reshape",
        sampled,
        _onnx_shape_vector(
            graph, batch_heads, channels_per_head, num_queries, level_points
        ),
    )
    weights = graph.op("Transpose", attention_weights, perm_i=[0, 2, 1, 3, 4])
    weights = graph.op(
        "Reshape",
        weights,
        _onnx_shape_vector(graph, batch_heads, 1, num_queries, level_points),
    )
    output = graph.op("Mul", sampled, weights)
    output = graph.op(
        "ReduceSum", output, _onnx_constant(graph, [-1]), keepdims_i=0
    )
    output = graph.op(
        "Reshape",
        output,
        _onnx_shape_vector(graph, batch, heads_channels, num_queries),
    )
    return graph.op("Transpose", output, perm_i=[0, 2, 1])

_BACKEND = os.environ.get("MVDETR_MS_DEFORM_ATTN_BACKEND", "auto").strip().lower()
_USE_MMCV = _BACKEND not in {"custom", "project", "torch", "pytorch", "grid_sample"}

if _BACKEND in {"torch", "pytorch", "grid_sample"}:
    MSDeformAttnFunction = None
else:
    try:
        if not _USE_MMCV:
            raise ImportError(f"mmcv backend disabled by MVDETR_MS_DEFORM_ATTN_BACKEND={_BACKEND}")
        from mmcv.ops.multi_scale_deform_attn import (
            MultiScaleDeformableAttnFunction as MSDeformAttnFunction,
        )
    except Exception:
        _ops_dir = Path(__file__).resolve().parents[1]
        if str(_ops_dir) not in sys.path:
            sys.path.insert(0, str(_ops_dir))
        import MultiScaleDeformableAttention as MSDA

        class MSDeformAttnFunction(Function):
            @staticmethod
            def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
                ctx.im2col_step = im2col_step
                output = MSDA.ms_deform_attn_forward(
                    value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
                ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
                return output

            @staticmethod
            def symbolic(graph, value, value_spatial_shapes, value_level_start_index,
                         sampling_locations, attention_weights, im2col_step):
                return ms_deform_attn_onnx_symbolic(
                    graph,
                    value,
                    value_spatial_shapes,
                    value_level_start_index,
                    sampling_locations,
                    attention_weights,
                    im2col_step,
                )

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
                grad_value, grad_sampling_loc, grad_attn_weight = \
                    MSDA.ms_deform_attn_backward(
                        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output.contiguous(), ctx.im2col_step)

                return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    spatial_shapes = [(int(H_), int(W_)) for H_, W_ in value_spatial_shapes.tolist()]
    value_list = value.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
