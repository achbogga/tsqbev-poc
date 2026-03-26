"""Lightweight LiDAR pillarization and sparse seed encoding.

References:
- Sparse4D efficient sparse aggregation:
  https://arxiv.org/pdf/2211.10581
- CMT multimodal LiDAR-camera fusion:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_Cross_Modal_Transformer_Towards_Fast_and_Robust_3D_Object_Detection_ICCV_2023_paper.pdf
- BEVFusion multimodal robustness tradeoffs:
  https://arxiv.org/abs/2205.13542
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tsqbev.config import ModelConfig, PillarConfig

Tensor = torch.Tensor


@dataclass(slots=True)
class PillarBatch:
    """Flattened per-batch pillar aggregation results."""

    batch_indices: Tensor
    centers_xyz: Tensor
    features: Tensor
    scores: Tensor


def pillarize_points(points: Tensor, mask: Tensor, cfg: PillarConfig) -> PillarBatch:
    """Aggregate raw LiDAR points into a compact set of pillars."""

    batch_indices: list[Tensor] = []
    centers_xyz: list[Tensor] = []
    features: list[Tensor] = []
    scores: list[Tensor] = []

    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    z_min, z_max = cfg.z_range
    size_x, size_y = cfg.pillar_size_xy

    for batch_index in range(points.shape[0]):
        valid_points = points[batch_index][mask[batch_index]]
        if valid_points.numel() == 0:
            continue

        xyz = valid_points[:, :3]
        intensity = valid_points[:, 3:4]
        valid = (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] <= x_max)
            & (xyz[:, 1] >= y_min)
            & (xyz[:, 1] <= y_max)
            & (xyz[:, 2] >= z_min)
            & (xyz[:, 2] <= z_max)
        )
        xyz = xyz[valid]
        intensity = intensity[valid]
        if xyz.numel() == 0:
            continue

        grid_x = torch.floor((xyz[:, 0] - x_min) / size_x).to(torch.int64)
        grid_y = torch.floor((xyz[:, 1] - y_min) / size_y).to(torch.int64)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        unique_grid, inverse = torch.unique(grid, dim=0, return_inverse=True)

        pillar_centers = torch.zeros(
            unique_grid.shape[0], 3, device=points.device, dtype=points.dtype
        )
        pillar_features = torch.zeros(
            unique_grid.shape[0], 4, device=points.device, dtype=points.dtype
        )
        pillar_counts = torch.zeros(unique_grid.shape[0], device=points.device, dtype=points.dtype)

        pillar_centers.index_add_(0, inverse, xyz)
        pillar_features.index_add_(0, inverse, torch.cat((xyz, intensity), dim=-1))
        pillar_counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=points.dtype))

        pillar_centers = pillar_centers / pillar_counts.unsqueeze(-1)
        pillar_features = pillar_features / pillar_counts.unsqueeze(-1)

        batch_indices.append(
            torch.full((unique_grid.shape[0],), batch_index, device=points.device, dtype=torch.long)
        )
        centers_xyz.append(pillar_centers)
        features.append(pillar_features)
        scores.append(pillar_counts)

    if not batch_indices:
        empty_long = torch.empty(0, dtype=torch.long, device=points.device)
        empty_float = torch.empty(0, 3, dtype=points.dtype, device=points.device)
        empty_feat = torch.empty(0, 4, dtype=points.dtype, device=points.device)
        empty_score = torch.empty(0, dtype=points.dtype, device=points.device)
        return PillarBatch(empty_long, empty_float, empty_feat, empty_score)

    return PillarBatch(
        batch_indices=torch.cat(batch_indices, dim=0),
        centers_xyz=torch.cat(centers_xyz, dim=0),
        features=torch.cat(features, dim=0),
        scores=torch.cat(scores, dim=0),
    )


class LidarSeedEncoder(nn.Module):
    """Encode pillars into top-k LiDAR seed queries."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Sequential(
            nn.Linear(5, config.model_dim),
            nn.ReLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )

    def forward(self, points: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return LiDAR seed embeddings, refs, and scores."""

        pillars = pillarize_points(points, mask, self.config.pillar)
        batch = points.shape[0]
        device = points.device
        dtype = points.dtype
        queries = torch.zeros(
            batch, self.config.q_lidar, self.config.model_dim, device=device, dtype=dtype
        )
        refs = torch.zeros(batch, self.config.q_lidar, 3, device=device, dtype=dtype)
        scores = torch.zeros(batch, self.config.q_lidar, device=device, dtype=dtype)

        if pillars.features.numel() == 0:
            return queries, refs, scores

        embedded = self.embed(torch.cat((pillars.features, pillars.scores.unsqueeze(-1)), dim=-1))
        for batch_index in range(batch):
            indices = torch.nonzero(pillars.batch_indices == batch_index, as_tuple=False).squeeze(
                -1
            )
            if indices.numel() == 0:
                continue
            batch_scores = pillars.scores[indices]
            order = torch.argsort(batch_scores, descending=True)
            top = indices[order[: self.config.q_lidar]]
            count = top.numel()
            queries[batch_index, :count] = embedded[top]
            refs[batch_index, :count] = pillars.centers_xyz[top]
            scores[batch_index, :count] = pillars.scores[top]
        return queries, refs, scores
