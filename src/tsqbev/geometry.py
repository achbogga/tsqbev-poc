"""Projection, backprojection, and sparse sampling geometry.

References:
- DETR3D 3D-to-2D projection:
  https://proceedings.mlr.press/v164/wang22b/wang22b.pdf
- PETR position-aware features:
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870523.pdf
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def make_homogeneous(points_xyz: Tensor) -> Tensor:
    """Append a homogeneous coordinate to xyz points."""

    ones = torch.ones(*points_xyz.shape[:-1], 1, device=points_xyz.device, dtype=points_xyz.dtype)
    return torch.cat((points_xyz, ones), dim=-1)


def transform_points(points_xyz: Tensor, transform: Tensor) -> Tensor:
    """Transform points by a 4x4 matrix with broadcasting over leading dimensions."""

    homogeneous = make_homogeneous(points_xyz)
    transformed = homogeneous @ transform.transpose(-1, -2)
    return transformed[..., :3]


def project_points(
    points_xyz: Tensor, intrinsics: Tensor, extrinsics: Tensor, eps: float = 1e-6
) -> Tensor:
    """Project ego-frame points into image space."""

    camera_points = transform_points(points_xyz, extrinsics)
    depths = camera_points[..., 2:3].clamp_min(eps)
    pixels_h = camera_points @ intrinsics.transpose(-1, -2)
    uv = pixels_h[..., :2] / depths
    return torch.cat((uv, camera_points[..., 2:3]), dim=-1)


def normalize_grid(uv: Tensor, height: int, width: int) -> Tensor:
    """Convert pixel coordinates into grid_sample coordinates."""

    width_scale = torch.clamp(torch.as_tensor(width - 1, device=uv.device, dtype=uv.dtype), min=1.0)
    height_scale = torch.clamp(
        torch.as_tensor(height - 1, device=uv.device, dtype=uv.dtype), min=1.0
    )
    x = (uv[..., 0] / width_scale) * 2.0 - 1.0
    y = (uv[..., 1] / height_scale) * 2.0 - 1.0
    return torch.stack((x, y), dim=-1)


def box_centers_xy(boxes_xyxy: Tensor) -> Tensor:
    """Convert xyxy boxes to center coordinates."""

    return torch.stack(
        (
            (boxes_xyxy[..., 0] + boxes_xyxy[..., 2]) * 0.5,
            (boxes_xyxy[..., 1] + boxes_xyxy[..., 3]) * 0.5,
        ),
        dim=-1,
    )


def ray_points_from_proposals(
    boxes_xyxy: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor,
    depths: Tensor,
) -> Tensor:
    """Backproject 2D proposal centers along depth bins into ego space."""

    centers = box_centers_xy(boxes_xyxy)
    batch, views, proposals = centers.shape[:3]
    rays = torch.cat(
        (
            centers,
            torch.ones(batch, views, proposals, 1, device=centers.device, dtype=centers.dtype),
        ),
        dim=-1,
    )
    intrinsics_inv = torch.inverse(intrinsics)
    rays_camera = rays @ intrinsics_inv.transpose(-1, -2)
    rays_camera = rays_camera.unsqueeze(-2) * depths.view(1, 1, 1, -1, 1)

    camera_to_ego = torch.inverse(extrinsics)
    rays_h = make_homogeneous(rays_camera)
    ego_points = rays_h @ camera_to_ego.unsqueeze(-3).transpose(-1, -2)
    return ego_points[..., :3]
