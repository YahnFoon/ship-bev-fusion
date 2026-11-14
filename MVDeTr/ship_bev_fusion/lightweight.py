"""Compression utilities for the Indoor 20260510 deployment experiments."""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import nn


DEFAULT_PROTECTED_TOKENS = (
    "sampling_offsets",
    "attention_weights",
    "class_embed",
    "mask_embed",
    "query_embed",
    "query_feat",
    "world_mask",
    "conv_seg",
)


@dataclass(frozen=True)
class LayerCompressionRecord:
    name: str
    module_type: str
    original_parameters: int
    compressed_parameters: int
    detail: str


@dataclass(frozen=True)
class ModelStatistics:
    parameters: int
    nonzero_parameters: int
    trainable_parameters: int
    zero_fraction: float
    fp32_weight_mib: float


def model_statistics(modules: Mapping[str, nn.Module]) -> ModelStatistics:
    parameters = 0
    nonzero = 0
    trainable = 0
    seen: set[int] = set()
    for module in modules.values():
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters += int(parameter.numel())
            nonzero += int(torch.count_nonzero(parameter.detach()).item())
            if parameter.requires_grad:
                trainable += int(parameter.numel())
    zero_fraction = 1.0 - float(nonzero) / max(float(parameters), 1.0)
    return ModelStatistics(
        parameters=parameters,
        nonzero_parameters=nonzero,
        trainable_parameters=trainable,
        zero_fraction=zero_fraction,
        fp32_weight_mib=float(parameters * 4) / (1024.0**2),
    )


def statistics_dict(modules: Mapping[str, nn.Module]) -> Dict[str, Any]:
    return asdict(model_statistics(modules))


def _is_eligible_module(name: str, module: nn.Module, protected_tokens: Sequence[str]) -> bool:
    if any(token in name for token in protected_tokens):
        return False
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return False
    if module.weight is None or module.weight.numel() < 256:
        return False
    if isinstance(module, nn.Conv2d) and module.groups != 1:
        return False
    return True


def eligible_weight_parameters(
    modules: Mapping[str, nn.Module],
    protected_tokens: Sequence[str] = DEFAULT_PROTECTED_TOKENS,
) -> List[Tuple[str, nn.Parameter]]:
    result: List[Tuple[str, nn.Parameter]] = []
    for root_name, root in modules.items():
        for name, module in root.named_modules():
            qualified = f"{root_name}.{name}" if name else root_name
            if _is_eligible_module(qualified, module, protected_tokens):
                result.append((f"{qualified}.weight", module.weight))
    return result


class WeightMaskController:
    """Keep pruning masks active during recovery training."""

    def __init__(self, named_parameters: Mapping[str, nn.Parameter], masks: Mapping[str, torch.Tensor]) -> None:
        missing = sorted(set(masks) - set(named_parameters))
        if missing:
            raise KeyError(f"Pruning masks reference missing parameters: {missing[:5]}")
        self.named_parameters = dict(named_parameters)
        self.masks = {name: mask.detach().to(device="cpu", dtype=torch.bool) for name, mask in masks.items()}

    def apply(self) -> None:
        with torch.no_grad():
            for name, mask in self.masks.items():
                parameter = self.named_parameters[name]
                parameter.mul_(mask.to(device=parameter.device, dtype=parameter.dtype))

    def mask_gradients(self) -> None:
        for name, mask in self.masks.items():
            parameter = self.named_parameters[name]
            if parameter.grad is not None:
                parameter.grad.mul_(mask.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: mask.clone() for name, mask in self.masks.items()}


def global_magnitude_pruning(
    modules: Mapping[str, nn.Module],
    sparsity: float,
    protected_tokens: Sequence[str] = DEFAULT_PROTECTED_TOKENS,
) -> Tuple[WeightMaskController, Dict[str, Any]]:
    if not 0.0 < float(sparsity) < 1.0:
        raise ValueError(f"sparsity must be in (0,1), got {sparsity}")
    eligible = eligible_weight_parameters(modules, protected_tokens)
    if not eligible:
        raise RuntimeError("No eligible Conv2d/Linear weights found for magnitude pruning")
    scores = torch.cat([parameter.detach().abs().flatten().cpu() for _, parameter in eligible])
    prune_count = min(max(int(math.floor(scores.numel() * float(sparsity))), 1), scores.numel() - 1)
    threshold = torch.kthvalue(scores, prune_count).values
    masks: Dict[str, torch.Tensor] = {}
    named_parameters: Dict[str, nn.Parameter] = {}
    for name, parameter in eligible:
        masks[name] = parameter.detach().abs().cpu() > threshold
        named_parameters[name] = parameter
    controller = WeightMaskController(named_parameters, masks)
    controller.apply()
    eligible_total = sum(mask.numel() for mask in masks.values())
    eligible_nonzero = sum(int(mask.sum().item()) for mask in masks.values())
    report = {
        "method": "global_magnitude",
        "requested_sparsity": float(sparsity),
        "eligible_parameters": int(eligible_total),
        "remaining_eligible_parameters": int(eligible_nonzero),
        "actual_eligible_sparsity": 1.0 - float(eligible_nonzero) / max(float(eligible_total), 1.0),
        "threshold": float(threshold.item()),
        "masked_tensors": len(masks),
    }
    return controller, report


def _two_of_four_mask(weight: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if weight.ndim == 2:
        arranged = weight.detach().abs()
        original_shape = arranged.shape
        if original_shape[1] % 4 != 0:
            return torch.ones_like(weight, dtype=torch.bool), 0
        groups = arranged.reshape(original_shape[0], original_shape[1] // 4, 4)
        keep = groups.topk(k=2, dim=-1, largest=True).indices
        mask_grouped = torch.zeros_like(groups, dtype=torch.bool)
        mask_grouped.scatter_(-1, keep, True)
        return mask_grouped.reshape(original_shape), int(weight.numel())
    if weight.ndim == 4:
        out_channels, in_channels, kh, kw = weight.shape
        if in_channels % 4 != 0:
            return torch.ones_like(weight, dtype=torch.bool), 0
        arranged = weight.detach().abs().permute(0, 2, 3, 1).contiguous()
        groups = arranged.reshape(out_channels, kh, kw, in_channels // 4, 4)
        keep = groups.topk(k=2, dim=-1, largest=True).indices
        mask_grouped = torch.zeros_like(groups, dtype=torch.bool)
        mask_grouped.scatter_(-1, keep, True)
        mask = mask_grouped.reshape(out_channels, kh, kw, in_channels).permute(0, 3, 1, 2).contiguous()
        return mask, int(weight.numel())
    return torch.ones_like(weight, dtype=torch.bool), 0


def two_of_four_pruning(
    modules: Mapping[str, nn.Module],
    protected_tokens: Sequence[str] = DEFAULT_PROTECTED_TOKENS,
) -> Tuple[WeightMaskController, Dict[str, Any]]:
    eligible = eligible_weight_parameters(modules, protected_tokens)
    masks: Dict[str, torch.Tensor] = {}
    named_parameters: Dict[str, nn.Parameter] = {}
    eligible_count = 0
    for name, parameter in eligible:
        mask, count = _two_of_four_mask(parameter)
        if count == 0:
            continue
        masks[name] = mask.cpu()
        named_parameters[name] = parameter
        eligible_count += count
    if not masks:
        raise RuntimeError("No weights satisfy the TensorRT 2:4 input-axis requirements")
    controller = WeightMaskController(named_parameters, masks)
    controller.apply()
    nonzero = sum(int(mask.sum().item()) for mask in masks.values())
    return controller, {
        "method": "two_of_four",
        "eligible_parameters": int(eligible_count),
        "remaining_eligible_parameters": int(nonzero),
        "actual_eligible_sparsity": 1.0 - float(nonzero) / max(float(eligible_count), 1.0),
        "masked_tensors": len(masks),
    }


class FactorizedLinear(nn.Module):
    def __init__(self, first: nn.Linear, second: nn.Linear) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(inputs))


class FactorizedConv2d(nn.Module):
    def __init__(self, spatial: nn.Conv2d, pointwise: nn.Conv2d) -> None:
        super().__init__()
        self.spatial = spatial
        self.pointwise = pointwise

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.spatial(inputs))


def _rounded_rank(raw_rank: float, maximum: int, round_to: int) -> int:
    if maximum <= 1:
        return 1
    rounded = int(round(float(raw_rank) / float(round_to)) * int(round_to))
    return min(max(rounded, 1 if maximum < round_to else round_to), maximum - 1)


def factorize_linear(module: nn.Linear, rank: int) -> FactorizedLinear:
    weight = module.weight.detach().float().cpu()
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    rank = min(int(rank), int(s.numel()))
    root = s[:rank].sqrt()
    first = nn.Linear(module.in_features, rank, bias=False, device=module.weight.device, dtype=module.weight.dtype)
    second = nn.Linear(rank, module.out_features, bias=module.bias is not None, device=module.weight.device, dtype=module.weight.dtype)
    with torch.no_grad():
        first.weight.copy_((root[:, None] * vh[:rank]).to(first.weight))
        second.weight.copy_((u[:, :rank] * root[None, :]).to(second.weight))
        if module.bias is not None:
            second.bias.copy_(module.bias.detach())
    return FactorizedLinear(first, second)


def factorize_conv2d(module: nn.Conv2d, rank: int) -> FactorizedConv2d:
    if module.groups != 1:
        raise ValueError("Only groups=1 Conv2d can be factorized")
    out_channels, in_channels, kh, kw = module.weight.shape
    matrix = module.weight.detach().float().cpu().reshape(out_channels, in_channels * kh * kw)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = min(int(rank), int(s.numel()))
    root = s[:rank].sqrt()
    spatial = nn.Conv2d(
        in_channels,
        rank,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=1,
        bias=False,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    pointwise = nn.Conv2d(
        rank,
        out_channels,
        kernel_size=1,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    with torch.no_grad():
        spatial.weight.copy_((root[:, None] * vh[:rank]).reshape(rank, in_channels, kh, kw).to(spatial.weight))
        pointwise.weight.copy_((u[:, :rank] * root[None, :]).reshape(out_channels, rank, 1, 1).to(pointwise.weight))
        if module.bias is not None:
            pointwise.bias.copy_(module.bias.detach())
    return FactorizedConv2d(spatial, pointwise)


def _parent_and_leaf(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, leaf: str, replacement: nn.Module) -> None:
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def apply_low_rank_factorization(
    modules: MutableMapping[str, nn.Module],
    rank_ratio: float,
    *,
    minimum_reduction: float = 0.0,
    round_to: int = 8,
    protected_tokens: Sequence[str] = DEFAULT_PROTECTED_TOKENS,
) -> List[LayerCompressionRecord]:
    if not 0.0 < float(rank_ratio) < 1.0:
        raise ValueError(f"rank_ratio must be in (0,1), got {rank_ratio}")
    records: List[LayerCompressionRecord] = []
    for root_name, root in modules.items():
        candidates = list(root.named_modules())
        for name, module in candidates:
            if not name or not _is_eligible_module(f"{root_name}.{name}", module, protected_tokens):
                continue
            if isinstance(module, nn.Linear):
                maximum = min(module.in_features, module.out_features)
                rank = _rounded_rank(maximum * float(rank_ratio), maximum, int(round_to))
                original = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                compressed = module.in_features * rank + rank * module.out_features
                compressed += module.out_features if module.bias is not None else 0
            else:
                out_channels, in_channels, kh, kw = module.weight.shape
                maximum = min(out_channels, in_channels * kh * kw)
                rank = _rounded_rank(maximum * float(rank_ratio), maximum, int(round_to))
                original = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
                compressed = rank * in_channels * kh * kw + out_channels * rank
                compressed += out_channels if module.bias is not None else 0
            if compressed >= original:
                continue
            reduction = 1.0 - float(compressed) / float(original)
            if reduction < float(minimum_reduction):
                continue
            replacement = (
                factorize_linear(module, rank)
                if isinstance(module, nn.Linear)
                else factorize_conv2d(module, rank)
            )
            parent, leaf = _parent_and_leaf(root, name)
            _set_child(parent, leaf, replacement)
            records.append(
                LayerCompressionRecord(
                    name=f"{root_name}.{name}",
                    module_type=type(module).__name__,
                    original_parameters=int(original),
                    compressed_parameters=int(compressed),
                    detail=f"rank={rank} ratio={float(rank_ratio):.3f} reduction={reduction:.4f}",
                )
            )
    if not records:
        raise RuntimeError("No layer met the requested low-rank reduction criterion")
    return records


def records_as_dicts(records: Iterable[LayerCompressionRecord]) -> List[Dict[str, Any]]:
    return [asdict(record) for record in records]


def apply_compression_recipe(
    modules: MutableMapping[str, nn.Module],
    recipe: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebuild deterministic structural transforms before loading a state dict."""
    if not recipe:
        return []
    transform = str(recipe.get("transform", "none"))
    if transform in {"none", "global_magnitude", "two_of_four"}:
        return []
    if transform == "svd":
        records = apply_low_rank_factorization(
            modules,
            rank_ratio=float(recipe["rank_ratio"]),
            minimum_reduction=float(recipe.get("minimum_reduction", 0.0)),
            round_to=int(recipe.get("round_to", 8)),
            protected_tokens=tuple(recipe.get("protected_tokens", DEFAULT_PROTECTED_TOKENS)),
        )
        return records_as_dicts(records)
    raise ValueError(f"Unsupported deterministic compression recipe: {transform}")


def _infer_single_unknown_concat_size(channel_sizes: Sequence[Optional[int]], output_size: int) -> List[int]:
    """Recover one unknown concat branch from the known output width."""
    unknown = [index for index, size in enumerate(channel_sizes) if size is None]
    if len(unknown) != 1:
        raise RuntimeError(
            "Torch-Pruning concat compatibility requires exactly one unknown branch, "
            f"got {len(unknown)} in {list(channel_sizes)}"
        )
    known_total = sum(int(size) for size in channel_sizes if size is not None)
    inferred = int(output_size) - known_total
    if inferred <= 0:
        raise RuntimeError(
            "Torch-Pruning concat compatibility inferred a non-positive branch width: "
            f"output={output_size}, known_total={known_total}, sizes={list(channel_sizes)}"
        )
    repaired = [int(size) if size is not None else inferred for size in channel_sizes]
    if sum(repaired) != int(output_size):
        raise RuntimeError(
            f"Repaired concat widths {repaired} do not sum to output width {output_size}"
        )
    return repaired


def _is_permuted_token_axis_concat(cat_node: Any) -> bool:
    """Identify [batch, tokens, channels] concatenation after feature permutation."""
    saved_dim = getattr(cat_node.grad_fn, "_saved_dim", None)
    input_grad_fns = [type(node.grad_fn).__name__ for node in cat_node.inputs]
    return (
        saved_dim == 1
        and len(input_grad_fns) > 1
        and all(name.startswith("PermuteBackward") for name in input_grad_fns)
    )


@contextmanager
def _torch_pruning_concat_compatibility(tp: Any) -> Iterable[List[Dict[str, Any]]]:
    """Work around Torch-Pruning 1.6.0 summing concat widths before its None guard."""
    index_mapping = tp.dependency.index_mapping
    original = index_mapping.update_concat_index_mapping
    recoveries: List[Dict[str, Any]] = []

    def guarded(graph: Any, cat_node: Any) -> Any:
        if getattr(cat_node.module, "_indoor_token_axis_identity", False):
            return None
        try:
            return original(graph, cat_node)
        except TypeError:
            sizes = getattr(cat_node.module, "concat_sizes", None)
            if not sizes or not any(size is None for size in sizes):
                raise
            output_size = index_mapping.shape_infer.infer_in_channels_recursively(
                graph,
                cat_node,
                recursive_depth=[0],
            )
            if output_size is None:
                raise RuntimeError(
                    f"Torch-Pruning could not infer the output width for concat node {cat_node.name}"
                )
            if sum(size is None for size in sizes) > 1 and _is_permuted_token_axis_concat(cat_node):
                cat_node.module.concat_sizes = None
                cat_node.module._indoor_token_axis_identity = True
                cat_node.enable_index_mapping = False
                recoveries.append(
                    {
                        "mode": "permuted_token_axis_identity",
                        "node": str(cat_node.name),
                        "original_sizes": [None if size is None else int(size) for size in sizes],
                        "output_size": int(output_size),
                        "concat_dim": 1,
                    }
                )
                return None
            try:
                repaired = _infer_single_unknown_concat_size(sizes, int(output_size))
            except RuntimeError as exc:
                input_nodes = [
                    {
                        "name": str(node.name),
                        "type": str(node.type),
                        "module": str(node.module),
                    }
                    for node in cat_node.inputs
                ]
                output_nodes = [
                    {
                        "name": str(node.name),
                        "type": str(node.type),
                        "module": str(node.module),
                    }
                    for node in cat_node.outputs
                ]
                saved_dim = getattr(cat_node.grad_fn, "_saved_dim", None)
                raise RuntimeError(
                    "Torch-Pruning concat width recovery is ambiguous: "
                    f"node={cat_node.name}, saved_dim={saved_dim}, output_size={output_size}, "
                    f"sizes={list(sizes)}, inputs={input_nodes}, outputs={output_nodes}"
                ) from exc
            cat_node.module.concat_sizes = repaired
            recoveries.append(
                {
                    "mode": "single_unknown_width_inferred",
                    "node": str(cat_node.name),
                    "original_sizes": [None if size is None else int(size) for size in sizes],
                    "repaired_sizes": repaired,
                    "output_size": int(output_size),
                }
            )
            return original(graph, cat_node)

    index_mapping.update_concat_index_mapping = guarded
    try:
        yield recoveries
    finally:
        index_mapping.update_concat_index_mapping = original


def _call_with_example_inputs(model: nn.Module, example_inputs: Any) -> Any:
    if isinstance(example_inputs, Mapping):
        return model(**example_inputs)
    if isinstance(example_inputs, tuple):
        return model(*example_inputs)
    return model(example_inputs)


def _count_torch_pruning_macs(tp: Any, model: nn.Module, example_inputs: Any) -> float:
    """Run Torch-Pruning's module MAC counter with kwargs-aware MHA hooks."""
    op_counter = tp.utils.op_counter
    was_training = model.training
    counter = op_counter.add_flops_counting_methods(model)
    handles: List[Any] = []
    started = False

    def mha_hook(module: nn.MultiheadAttention, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> None:
        query = kwargs.get("query", args[0] if len(args) > 0 else None)
        key = kwargs.get("key", args[1] if len(args) > 1 else query)
        value = kwargs.get("value", args[2] if len(args) > 2 else key)
        if query is None or key is None or value is None:
            raise RuntimeError("MultiheadAttention MAC hook could not resolve query/key/value")
        op_counter.multihead_attention_counter_hook(module, (query, key, value), output)

    model.eval()
    try:
        counter.start_flops_count(
            ost=sys.stdout,
            verbose=False,
            ignore_list=[nn.MultiheadAttention],
        )
        started = True
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                handles.append(module.register_forward_hook(mha_hook, with_kwargs=True))
        with torch.no_grad():
            _call_with_example_inputs(model, example_inputs)
        macs = float(counter.compute_average_flops_cost()[0])
    finally:
        for handle in handles:
            handle.remove()
        if started:
            counter.stop_flops_count()
        op_counter.CUSTOM_MODULES_MAPPING = {}
        model.train(was_training)
    if macs <= 0:
        raise RuntimeError(f"Torch-Pruning MAC counter returned {macs}")
    return macs


def _validate_pruned_forward(model: nn.Module, example_inputs: Any) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = _call_with_example_inputs(model, example_inputs)
    finally:
        model.train(was_training)
    tensors = [value for value in torch.utils._pytree.tree_leaves(output) if isinstance(value, torch.Tensor)]
    if not tensors:
        raise RuntimeError("Structured-pruned model forward produced no tensor outputs")
    nonfinite = sum(int((~torch.isfinite(value)).sum().item()) for value in tensors)
    if nonfinite:
        raise RuntimeError(f"Structured-pruned model forward produced {nonfinite} non-finite values")
    return {
        "tensor_count": len(tensors),
        "output_shapes": [list(value.shape) for value in tensors],
        "nonfinite_values": 0,
    }


def apply_structured_pruning(
    model: nn.Module,
    example_inputs: Any,
    pruning_ratio: float,
    *,
    ignored_layers: Optional[Sequence[nn.Module]] = None,
    round_to: int = 8,
) -> Dict[str, Any]:
    """Physically prune coupled Conv/Linear dimensions with Torch-Pruning."""
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError("Structured pruning requires torch-pruning in indoor_bev_deploy") from exc
    if not 0.0 < float(pruning_ratio) < 1.0:
        raise ValueError(f"pruning_ratio must be in (0,1), got {pruning_ratio}")
    importance = tp.importance.TaylorImportance()
    macs_before: Optional[float] = None
    macs_after: Optional[float] = None
    mac_error: Optional[str] = None
    try:
        macs_before = _count_torch_pruning_macs(tp, model, example_inputs)
    except Exception as exc:
        mac_error = f"before: {type(exc).__name__}: {exc}"
    with _torch_pruning_concat_compatibility(tp) as concat_recoveries:
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_ratio=float(pruning_ratio),
            global_pruning=True,
            isomorphic=True,
            ignored_layers=list(ignored_layers or []),
            round_to=int(round_to),
        )
        before_params = sum(parameter.numel() for parameter in model.parameters())
        pruner.step()
    after_params = sum(parameter.numel() for parameter in model.parameters())
    forward_validation = _validate_pruned_forward(model, example_inputs)
    try:
        macs_after = _count_torch_pruning_macs(tp, model, example_inputs)
    except Exception as exc:
        message = f"after: {type(exc).__name__}: {exc}"
        mac_error = f"{mac_error}; {message}" if mac_error else message
    return {
        "method": "torch_pruning_taylor",
        "requested_pruning_ratio": float(pruning_ratio),
        "parameters_before": int(before_params),
        "parameters_after": int(after_params),
        "actual_parameter_reduction": 1.0 - float(after_params) / max(float(before_params), 1.0),
        "macs_before": macs_before,
        "macs_after": macs_after,
        "actual_mac_reduction": (
            1.0 - float(macs_after) / max(float(macs_before), 1.0)
            if macs_before is not None and macs_after is not None
            else None
        ),
        "mac_count_error": mac_error,
        "mac_count_method": "torch_pruning_module_hooks_with_kwargs_aware_multihead_attention",
        "round_to": int(round_to),
        "torch_pruning_concat_recoveries": concat_recoveries,
        "forward_validation": forward_validation,
    }
