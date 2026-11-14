"""MVDeTr adaptation for ship BEV mask segmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective

try:
    from multiview_detector.models.conv_world_feat import ConvWorldFeat, DeformConvWorldFeat
    from multiview_detector.models.mvdetr import create_reference_map, output_head
    from multiview_detector.models.resnet import resnet18
    from multiview_detector.models.trans_world_feat import (
        DeformTransWorldFeat, DeformTransWorldFeat_aio, TransformerWorldFeat,
    )
    from multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat
except ModuleNotFoundError as exc:
    missing = exc.name or ''
    if missing == 'multiview_detector' or missing.startswith('multiview_detector.'):
        raise ImportError(
            'The paper fusion model requires the MVDeTr runtime files listed in docs/INSTALL.md.'
        ) from exc
    raise



def inverse_3x3(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Exportable batched 3x3 inverse using the analytic adjugate."""
    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    g, h, i = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    cof00 = e * i - f * h
    cof01 = -(d * i - f * g)
    cof02 = d * h - e * g
    cof10 = -(b * i - c * h)
    cof11 = a * i - c * g
    cof12 = -(a * h - b * g)
    cof20 = b * f - c * e
    cof21 = -(a * f - c * d)
    cof22 = a * e - b * d
    determinant = a * cof00 + b * cof01 + c * cof02
    safe = torch.where(determinant.abs() > float(eps), determinant, torch.full_like(determinant, float(eps)))
    adjugate = torch.stack(
        (cof00, cof10, cof20, cof01, cof11, cof21, cof02, cof12, cof22),
        dim=-1,
    ).reshape(matrix.shape)
    return adjugate / safe[..., None, None]


def create_bev_identity_reference_map(
    rworld_shape: Sequence[int],
    num_levels: int,
    n_points: int,
    downsample: int = 2,
) -> torch.Tensor:
    """Reference map for inputs that are already aligned to the BEV grid."""
    world_h, world_w = (int(rworld_shape[0]), int(rworld_shape[1]))
    h, w = world_h // int(downsample), world_w // int(downsample)
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, h - 0.5, h, dtype=torch.float32),
        torch.linspace(0.5, w - 0.5, w, dtype=torch.float32),
    )
    ref = torch.stack((ref_x.reshape(-1) / w, ref_y.reshape(-1) / h), -1)
    return ref[:, None, None, :].repeat(1, int(num_levels), int(n_points), 1)


def _norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        dilation: int = 1,
        stride: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                stride=stride,
                bias=False,
            ),
            nn.GroupNorm(_norm_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBEVBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3, dilation=dilation),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class StrongBEVMaskEncoder(nn.Module):
    """Encode preprojected BEV masks before MVDeTr world feature fusion."""

    def __init__(self, input_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_channels = max(32, out_channels // 2)
        self.stem = ConvNormAct(input_channels, hidden_channels, 3)
        self.context = nn.Sequential(
            ResidualBEVBlock(hidden_channels, dilation=1, dropout=dropout),
            ResidualBEVBlock(hidden_channels, dilation=2, dropout=dropout),
            ResidualBEVBlock(hidden_channels, dilation=4, dropout=dropout),
            ResidualBEVBlock(hidden_channels, dilation=1, dropout=dropout),
        )
        self.project = ConvNormAct(hidden_channels, out_channels, 1)
        self.refine = nn.Sequential(
            ResidualBEVBlock(out_channels, dilation=1, dropout=dropout),
            ResidualBEVBlock(out_channels, dilation=2, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.context(x)
        x = self.project(x)
        return self.refine(x)


class PyramidBEVMaskEncoder(nn.Module):
    """Efficient FPN-style BEV mask encoder with image-resnet-aligned output channels."""

    def __init__(self, input_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_channels = max(16, out_channels // 8)
        mid_channels = max(32, out_channels // 4)
        high_channels = max(64, out_channels // 2)
        self.stem = nn.Sequential(
            ConvNormAct(input_channels, hidden_channels, 3),
            ResidualBEVBlock(hidden_channels, dilation=1, dropout=dropout),
        )
        self.down1 = ConvNormAct(hidden_channels, mid_channels, 3, stride=2)
        self.enc1 = ResidualBEVBlock(mid_channels, dilation=1, dropout=dropout)
        self.down2 = ConvNormAct(mid_channels, high_channels, 3, stride=2)
        self.enc2 = nn.Sequential(
            ResidualBEVBlock(high_channels, dilation=1, dropout=dropout),
            ResidualBEVBlock(high_channels, dilation=2, dropout=dropout),
        )
        self.lat2 = nn.Conv2d(high_channels, mid_channels, 1, bias=False)
        self.lat1 = nn.Conv2d(mid_channels, mid_channels, 1, bias=False)
        self.fuse1 = ResidualBEVBlock(mid_channels, dilation=1, dropout=dropout)
        self.fuse0 = nn.Sequential(
            ConvNormAct(hidden_channels + mid_channels, mid_channels, 3),
            ResidualBEVBlock(mid_channels, dilation=1, dropout=dropout),
        )
        self.project = nn.Sequential(
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            ConvNormAct(mid_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.enc1(self.down1(x0))
        x2 = self.enc2(self.down2(x1))
        p1 = self.lat1(x1) + torch.nn.functional.interpolate(
            self.lat2(x2),
            size=x1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.fuse1(p1)
        p0 = torch.nn.functional.interpolate(p1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        p0 = self.fuse0(torch.cat([x0, p0], dim=1))
        return self.project(p0)


class PerCameraSimilarityAligner(nn.Module):
    """Learn a small BEV-space similarity residual for each camera level."""

    def __init__(
        self,
        num_cameras: int,
        rworld_shape: Sequence[int],
        *,
        max_translate_px: float = 40.0,
        max_scale_delta: float = 0.10,
        max_rotation_deg: float = 5.0,
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if int(num_cameras) < 1:
            raise ValueError("PerCameraSimilarityAligner requires at least one camera")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be one of 'zeros', 'border', or 'reflection'")
        self.num_cameras = int(num_cameras)
        self.rworld_shape = (int(rworld_shape[0]), int(rworld_shape[1]))
        self.max_translate_px = float(max_translate_px)
        self.max_scale_delta = float(max_scale_delta)
        self.max_rotation_rad = float(max_rotation_deg) * float(np.pi) / 180.0
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)

        self.raw_params = nn.Parameter(torch.zeros(self.num_cameras, 5))

    @classmethod
    def from_config(cls, num_cameras: int, rworld_shape: Sequence[int], config: Mapping[str, Any]) -> "PerCameraSimilarityAligner":
        if str(config.get("type", "per_camera_similarity")) != "per_camera_similarity":
            raise ValueError("feature_alignment.type must be 'per_camera_similarity'")
        return cls(
            num_cameras,
            rworld_shape,
            max_translate_px=float(config.get("max_translate_px", 40.0)),
            max_scale_delta=float(config.get("max_scale_delta", 0.10)),
            max_rotation_deg=float(config.get("max_rotation_deg", 5.0)),
            padding_mode=str(config.get("padding_mode", "zeros")),
            align_corners=bool(config.get("align_corners", False)),
        )

    def bounded_parameters(self) -> Mapping[str, torch.Tensor]:
        bounded = torch.tanh(self.raw_params)
        scale_delta = bounded[:, 2:4] * self.max_scale_delta
        return {
            "dx_px": bounded[:, 0] * self.max_translate_px,
            "dy_px": bounded[:, 1] * self.max_translate_px,
            "scale_x": 1.0 + scale_delta[:, 0],
            "scale_y": 1.0 + scale_delta[:, 1],
            "rotation_rad": bounded[:, 4] * self.max_rotation_rad,
            "normalized": bounded,
        }

    def regularization_loss(self) -> torch.Tensor:
        params = self.bounded_parameters()["normalized"]
        return params.pow(2).mean()

    @torch.no_grad()
    def parameter_rows(self, camera_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        params = self.bounded_parameters()
        camera_names = list(camera_ids or [str(index) for index in range(self.num_cameras)])
        rows: List[Dict[str, Any]] = []
        for index in range(self.num_cameras):
            rotation_deg = float(params["rotation_rad"][index].detach().cpu()) * 180.0 / float(np.pi)
            rows.append(
                {
                    "camera_index": index,
                    "camera_id": camera_names[index] if index < len(camera_names) else str(index),
                    "dx_px": float(params["dx_px"][index].detach().cpu()),
                    "dy_px": float(params["dy_px"][index].detach().cpu()),
                    "scale_x": float(params["scale_x"][index].detach().cpu()),
                    "scale_y": float(params["scale_y"][index].detach().cpu()),
                    "rotation_deg": rotation_deg,
                }
            )
        return rows

    def _theta(self, *, device: torch.device, dtype: torch.dtype, height: int, width: int) -> torch.Tensor:
        params = self.bounded_parameters()
        dx = params["dx_px"].to(device=device, dtype=dtype)
        dy = params["dy_px"].to(device=device, dtype=dtype)
        sx = params["scale_x"].to(device=device, dtype=dtype).clamp_min(1e-4)
        sy = params["scale_y"].to(device=device, dtype=dtype).clamp_min(1e-4)
        rot = params["rotation_rad"].to(device=device, dtype=dtype)
        cos = torch.cos(rot)
        sin = torch.sin(rot)


        inv_a00 = cos / sx
        inv_a01 = sin / sx
        inv_a10 = -sin / sy
        inv_a11 = cos / sy
        x_scale = 2.0 / float(width if not self.align_corners else max(width - 1, 1))
        y_scale = 2.0 / float(height if not self.align_corners else max(height - 1, 1))
        tx = dx * x_scale
        ty = dy * y_scale
        theta = torch.zeros(self.num_cameras, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = inv_a00
        theta[:, 0, 1] = inv_a01
        theta[:, 1, 0] = inv_a10
        theta[:, 1, 1] = inv_a11
        theta[:, 0, 2] = -(inv_a00 * tx + inv_a01 * ty)
        theta[:, 1, 2] = -(inv_a10 * tx + inv_a11 * ty)
        return theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"PerCameraSimilarityAligner expects [B,N,C,H,W], got {tuple(x.shape)}")
        b, n, c, h, w = x.shape
        if int(n) != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} camera levels, got {n}")
        theta = self._theta(device=x.device, dtype=x.dtype, height=int(h), width=int(w))
        theta = theta.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * n, 2, 3)
        flat = x.reshape(b * n, c, h, w)
        grid = F.affine_grid(theta, flat.shape, align_corners=self.align_corners)
        aligned = F.grid_sample(
            flat,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return aligned.view(b, n, c, h, w)


class MVDeTrShipMask(nn.Module):
    """MVDeTr feature projection/fusion with a BEV mask head."""

    def __init__(
        self,
        dataset,
        z_mm: float = 93.0,
        arch: str = "resnet18",
        world_feat_arch: str = "conv",
        bottleneck_dim: int = 32,
        outfeat_dim: int = 32,
        dropout: float = 0.0,
        pretrained: bool = False,
        input_channels: int = 1,
        feature_mode: str = "image_resnet",
        bev_lift_dim: int = 0,
        deform_trans_n_points: int = 4,
        deform_trans_hidden_dim: int = 128,
        deform_trans_n_heads: int = 8,
        deform_trans_ffn_dim: int = 512,
        deform_trans_num_layers: int = 3,
        deform_trans_stride: int = 2,
        deform_trans_reference_mode: str = "camera",
        feature_alignment: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.Rimg_shape, self.Rworld_shape = dataset.Rimg_shape, dataset.Rworld_shape
        self.img_reduce = dataset.img_reduce
        self.feature_mode = str(feature_mode)
        self._projection_dataset = dataset
        self._projection_z_mm = float(z_mm)
        valid_feature_modes = {
            "image_resnet",
            "bev_mask_lift",
            "bev_mask_preprojected",
            "bev_mask_preprojected_strong",
            "bev_mask_preprojected_fpn",
            "bev_mask_preprojected_direct",
        }
        if self.feature_mode not in valid_feature_modes:
            raise ValueError(f"feature_mode must be one of {sorted(valid_feature_modes)}")
        self.proj_mats = self._projection_matrices_for_height(float(z_mm))
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, int(self.Rworld_shape[0]), dtype=torch.float32),
            torch.linspace(-1.0, 1.0, int(self.Rworld_shape[1]), dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer(
            "_deployment_bev_normalized_grid",
            torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0),
            persistent=False,
        )
        self._deployment_export_warp = False

        if self.feature_mode == "image_resnet":
            if arch != "resnet18":
                raise ValueError("MVDeTrShipMask currently supports arch='resnet18'")
            backbone = resnet18(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
            input_channels = int(input_channels)
            if input_channels != 3:
                old_conv1 = backbone.conv1
                new_conv1 = nn.Conv2d(input_channels, old_conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
                if pretrained:
                    with torch.no_grad():
                        mean_weight = old_conv1.weight.mean(dim=1, keepdim=True)
                        new_conv1.weight.copy_(mean_weight.repeat(1, input_channels, 1, 1))
                backbone.conv1 = new_conv1
            self.base = nn.Sequential(*list(backbone.children())[:-2])
            base_dim = 512

            if bottleneck_dim:
                self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, bottleneck_dim, 1), nn.Dropout2d(dropout))
                base_dim = bottleneck_dim
            else:
                self.bottleneck = nn.Identity()
            self.bev_lift = None
            self.camera_embedding = None
        else:
            self.base = nn.Identity()
            self.bottleneck = nn.Identity()
            if self.feature_mode == "bev_mask_preprojected_direct":
                base_dim = int(input_channels)
                self.bev_lift = nn.Identity()
                self.camera_embedding = None
            elif self.feature_mode in {"bev_mask_preprojected_strong", "bev_mask_preprojected_fpn"}:
                image_feature_dim = int(bottleneck_dim) if int(bottleneck_dim or 0) > 0 else 512
                if int(bev_lift_dim or 0) not in (0, image_feature_dim):
                    raise ValueError(
                        f"{self.feature_mode} keeps the world_feat input dimension aligned with "
                        f"image_resnet ({image_feature_dim}); got bev_lift_dim={bev_lift_dim}"
                    )
                base_dim = image_feature_dim
                if self.feature_mode == "bev_mask_preprojected_strong":
                    self.bev_lift = StrongBEVMaskEncoder(int(input_channels), base_dim, dropout=dropout)
                else:
                    self.bev_lift = PyramidBEVMaskEncoder(int(input_channels), base_dim, dropout=dropout)
                self.camera_embedding = nn.Parameter(torch.zeros(dataset.num_cam, base_dim, 1, 1))
            else:
                base_dim = int(bev_lift_dim or bottleneck_dim or 128)
                self.bev_lift = nn.Sequential(
                    nn.Conv2d(int(input_channels), base_dim, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_dim, base_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                )
                self.camera_embedding = None

        world_feat_out_dim = base_dim
        if world_feat_arch == "conv":
            self.world_feat = ConvWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim)
        elif world_feat_arch == "trans":
            self.world_feat = TransformerWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim)
        elif world_feat_arch == "deform_conv":
            self.world_feat = DeformConvWorldFeat(dataset.num_cam, dataset.Rworld_shape, base_dim)
        elif world_feat_arch == "deform_trans":
            n_points = int(deform_trans_n_points)
            reference_mode = str(deform_trans_reference_mode)
            if reference_mode == "camera":
                reference_points = create_reference_map(dataset, n_points).repeat([dataset.num_cam, 1, 1, 1])
            elif reference_mode == "bev_identity":
                reference_points = create_bev_identity_reference_map(
                    dataset.Rworld_shape,
                    dataset.num_cam,
                    n_points,
                    downsample=int(deform_trans_stride),
                ).repeat([dataset.num_cam, 1, 1, 1])
            else:
                raise ValueError("deform_trans_reference_mode must be 'camera' or 'bev_identity'")
            hidden_dim = int(deform_trans_hidden_dim)
            self.world_feat = DeformTransWorldFeat(
                dataset.num_cam,
                dataset.Rworld_shape,
                base_dim,
                hidden_dim=hidden_dim,
                nhead=int(deform_trans_n_heads),
                dim_feedforward=int(deform_trans_ffn_dim),
                n_points=n_points,
                stride=int(deform_trans_stride),
                reference_points=reference_points,
                num_layers=int(deform_trans_num_layers),
            )
            world_feat_out_dim = hidden_dim
        elif world_feat_arch == "aio":
            self.world_feat = DeformTransWorldFeat_aio(dataset.num_cam, dataset.Rworld_shape, base_dim)
            world_feat_out_dim = 128
        else:
            raise ValueError(f"Unknown world_feat_arch: {world_feat_arch}")

        alignment_cfg = dict(feature_alignment or {})
        if bool(alignment_cfg.get("enabled", False)):
            self.feature_aligner = PerCameraSimilarityAligner.from_config(
                dataset.num_cam,
                dataset.Rworld_shape,
                alignment_cfg,
            )
        else:
            self.feature_aligner = None
        self.camera_ids = tuple(
            str(camera_id)
            for camera_id in getattr(dataset, "input_camera_ids", getattr(dataset, "camera_ids", range(dataset.num_cam)))
        )
        self.world_mask = output_head(world_feat_out_dim, outfeat_dim, 1)

    def _projection_matrices_for_height(self, z_mm: float) -> torch.Tensor:
        dataset = self._projection_dataset
        world_zoom_mat = np.diag([dataset.world_reduce, dataset.world_reduce, 1])
        rworldgrid_from_worldcoord_mat = np.linalg.inv(
            dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat
        )
        matrices = []
        for cam in range(dataset.num_cam):
            worldcoord_from_imgcoord = get_worldcoord_from_imgcoord_mat(
                dataset.base.intrinsic_matrices[cam],
                dataset.base.extrinsic_matrices[cam],
                float(z_mm) / dataset.base.worldcoord_unit,
            )
            matrices.append(torch.from_numpy(rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord))
        return torch.stack(matrices).float()

    @torch.no_grad()
    def set_projection_height(self, z_mm: float) -> None:
        """Update only water-level-conditioned geometry; learned weights are unchanged."""
        target = float(z_mm)
        if abs(target - self._projection_z_mm) < 1e-9:
            return
        device = self.proj_mats.device
        self.proj_mats = self._projection_matrices_for_height(target).to(device=device)
        self._projection_z_mm = target

    def feature_alignment_regularization(self) -> torch.Tensor:
        if self.feature_aligner is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.feature_aligner.regularization_loss()

    @torch.no_grad()
    def feature_alignment_rows(self) -> List[Dict[str, Any]]:
        if self.feature_aligner is None:
            return []
        return self.feature_aligner.parameter_rows(self.camera_ids)

    def _projection_mats(
        self,
        affine_mats: torch.Tensor,
        *,
        source_reduce: float,
        batch_size: int,
        num_cameras: int,
        base_projection_mats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inv_affine = inverse_3x3(affine_mats.view(batch_size * num_cameras, 3, 3))
        imgcoord_from_grid = inv_affine.to(affine_mats.device) @ torch.diag(
            torch.tensor([float(source_reduce), float(source_reduce), 1.0], device=affine_mats.device)
        ).view(1, 3, 3).repeat(batch_size * num_cameras, 1, 1)
        if base_projection_mats is None:
            base_projection_mats = self.proj_mats
        if base_projection_mats.ndim == 3:
            base_projection_mats = base_projection_mats.unsqueeze(0).expand(batch_size, -1, -1, -1)
        expected = (batch_size, num_cameras, 3, 3)
        if tuple(base_projection_mats.shape) != expected:
            raise ValueError(
                f"base_projection_mats must have shape {expected}, got {tuple(base_projection_mats.shape)}"
            )
        return base_projection_mats.to(affine_mats.device).reshape(batch_size * num_cameras, 3, 3) @ imgcoord_from_grid

    def _warp_perspective_exportable(self, source: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        batch, _, source_h, source_w = source.shape
        output_h, output_w = self.Rworld_shape
        src_normalize = matrix.new_tensor(
            [[2.0 / max(source_w - 1, 1), 0.0, -1.0], [0.0, 2.0 / max(source_h - 1, 1), -1.0], [0.0, 0.0, 1.0]]
        ).unsqueeze(0)
        dst_normalize = matrix.new_tensor(
            [[2.0 / max(output_w - 1, 1), 0.0, -1.0], [0.0, 2.0 / max(output_h - 1, 1), -1.0], [0.0, 0.0, 1.0]]
        ).unsqueeze(0)
        normalized_forward = dst_normalize @ matrix @ inverse_3x3(src_normalize)
        normalized_inverse = inverse_3x3(normalized_forward)
        grid = self._deployment_bev_normalized_grid.to(device=source.device, dtype=source.dtype)
        grid = grid.expand(batch, -1, -1, -1)
        homogeneous = torch.nn.functional.pad(grid, (0, 1), mode="constant", value=1.0)
        source_homogeneous = homogeneous.reshape(batch, -1, 3) @ normalized_inverse.transpose(-1, -2)
        z = source_homogeneous[..., 2:3]
        scale = torch.where(z.abs() > 1e-8, 1.0 / (z + 1e-8), torch.ones_like(z))
        sample_grid = (source_homogeneous[..., :2] * scale).reshape(batch, output_h, output_w, 2)
        return torch.nn.functional.grid_sample(
            source,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def _project_camera_features(
        self,
        imgs: torch.Tensor,
        affine_mats: torch.Tensor,
        base_projection_mats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, c, h, w = imgs.shape
        world_h, world_w = self.Rworld_shape
        flat_imgs = imgs.view(b * n, c, h, w)
        if self.feature_mode == "image_resnet":
            proj_mats = self._projection_mats(
                affine_mats,
                source_reduce=self.img_reduce,
                batch_size=b,
                num_cameras=n,
                base_projection_mats=base_projection_mats,
            )
            imgs_feat = self.bottleneck(self.base(flat_imgs))
            _, feat_c, _, _ = imgs_feat.shape
            with torch.amp.autocast("cuda", enabled=False):
                if self._deployment_export_warp:
                    projected = self._warp_perspective_exportable(imgs_feat.float(), proj_mats.float())
                else:
                    projected = warp_perspective(
                        imgs_feat.float(),
                        proj_mats.float(),
                        self.Rworld_shape,
                        align_corners=False,
                    )
                world_feat = projected.view(b, n, feat_c, world_h, world_w)
        elif self.feature_mode == "bev_mask_lift":
            proj_mats = self._projection_mats(
                affine_mats,
                source_reduce=1.0,
                batch_size=b,
                num_cameras=n,
                base_projection_mats=base_projection_mats,
            )
            with torch.amp.autocast("cuda", enabled=False):
                if self._deployment_export_warp:
                    world_masks = self._warp_perspective_exportable(flat_imgs.float(), proj_mats.float())
                else:
                    world_masks = warp_perspective(
                        flat_imgs.float(),
                        proj_mats.float(),
                        self.Rworld_shape,
                        align_corners=False,
                    )
            world_feat = self.bev_lift(world_masks).view(b, n, -1, world_h, world_w)
        else:
            if (h, w) != (world_h, world_w):
                raise ValueError(
                    f"{self.feature_mode} expects BEV-sized inputs {(world_h, world_w)}, got {(h, w)}"
                )
            if self.feature_mode == "bev_mask_preprojected_direct":
                world_feat = flat_imgs.float().view(b, n, c, world_h, world_w)
            else:
                world_feat = self.bev_lift(flat_imgs.float()).view(b, n, -1, world_h, world_w)
            if self.camera_embedding is not None:
                world_feat = world_feat + self.camera_embedding[:n].view(1, n, -1, 1, 1).to(world_feat.device)
        return world_feat

    def forward_with_intermediates(self, imgs: torch.Tensor, affine_mats: torch.Tensor) -> dict:
        world_feat = self._project_camera_features(imgs, affine_mats)
        return self._fuse_projected_features(world_feat)

    def forward_with_projection_mats(
        self,
        imgs: torch.Tensor,
        affine_mats: torch.Tensor,
        base_projection_mats: torch.Tensor,
    ) -> dict:
        """Run fusion with explicit water-level-conditioned projection matrices."""
        world_feat = self._project_camera_features(imgs, affine_mats, base_projection_mats)
        return self._fuse_projected_features(world_feat)

    def _fuse_projected_features(self, world_feat: torch.Tensor) -> dict:
        aligned_feat = self.feature_aligner(world_feat) if self.feature_aligner is not None else world_feat
        fused = self.world_feat(aligned_feat)
        logits = self.world_mask(fused)
        outputs = {
            "logits": logits,
            "projected_features": world_feat,
            "aligned_features": aligned_feat,
            "fused_features": fused,
        }
        if self.feature_aligner is not None:
            outputs["alignment_params"] = self.feature_aligner.bounded_parameters()
        return outputs

    def forward(self, imgs: torch.Tensor, affine_mats: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(imgs, affine_mats)["logits"]
