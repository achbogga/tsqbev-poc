"""Hybrid query initialization and routing.

References:
- PETR / PETRv2 proposal-guided queries:
  https://github.com/megvii-research/PETR
- DETR3D query sampling:
  https://proceedings.mlr.press/v164/wang22b/wang22b.pdf
- RaCFormer query-centric multimodal fusion:
  https://openaccess.thecvf.com/content/CVPR2025/html/Chu_RaCFormer_Towards_High-Quality_3D_Object_Detection_via_Query-based_Radar-Camera_Fusion_CVPR_2025_paper.html
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from tsqbev.config import ModelConfig
from tsqbev.contracts import CameraProposals, QuerySeedBank
from tsqbev.geometry import ray_points_from_proposals

Tensor = torch.Tensor


class ProposalRayInitializer(nn.Module):
    """Turn 2D proposals into query embeddings and 3D reference points."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_mlp = nn.Sequential(
            nn.Linear(5, config.model_dim),
            nn.ReLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.depth_head = nn.Linear(config.model_dim, config.num_depth_bins)
        depth_bins = torch.linspace(5.0, 60.0, config.num_depth_bins)
        self.register_buffer("depth_bins", depth_bins)

    def forward(
        self,
        proposals: CameraProposals,
        intrinsics: Tensor,
        extrinsics: Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return proposal query embeddings, refs, and scores."""

        centers = torch.stack(
            (
                (proposals.boxes_xyxy[..., 0] + proposals.boxes_xyxy[..., 2]) * 0.5 / image_width,
                (proposals.boxes_xyxy[..., 1] + proposals.boxes_xyxy[..., 3]) * 0.5 / image_height,
                (proposals.boxes_xyxy[..., 2] - proposals.boxes_xyxy[..., 0]) / image_width,
                (proposals.boxes_xyxy[..., 3] - proposals.boxes_xyxy[..., 1]) / image_height,
                proposals.scores,
            ),
            dim=-1,
        )
        embeddings = self.feature_mlp(centers)
        depth_logits = self.depth_head(embeddings)
        depth_weights = depth_logits.softmax(dim=-1)
        depth_bins = cast(Tensor, self.depth_bins)
        ray_points = ray_points_from_proposals(
            proposals.boxes_xyxy,
            intrinsics,
            extrinsics,
            depth_bins,
        )
        refs = (depth_weights.unsqueeze(-1) * ray_points).sum(dim=-2)

        batch, views, proposals_per_view = proposals.scores.shape
        flat_embeddings = embeddings.reshape(
            batch, views * proposals_per_view, self.config.model_dim
        )
        flat_refs = refs.reshape(batch, views * proposals_per_view, 3)
        flat_scores = proposals.scores.reshape(batch, views * proposals_per_view)

        top_scores, top_indices = torch.topk(
            flat_scores,
            k=min(self.config.q_2d, flat_scores.shape[-1]),
            dim=-1,
        )
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
        gathered_embeddings = torch.gather(flat_embeddings, 1, gather_index)
        gathered_refs = torch.gather(flat_refs, 1, top_indices.unsqueeze(-1).expand(-1, -1, 3))

        if gathered_embeddings.shape[1] < self.config.q_2d:
            pad = self.config.q_2d - gathered_embeddings.shape[1]
            gathered_embeddings = torch.cat(
                (
                    gathered_embeddings,
                    torch.zeros(
                        batch,
                        pad,
                        self.config.model_dim,
                        device=embeddings.device,
                        dtype=embeddings.dtype,
                    ),
                ),
                dim=1,
            )
            gathered_refs = torch.cat(
                (gathered_refs, torch.zeros(batch, pad, 3, device=refs.device, dtype=refs.dtype)),
                dim=1,
            )
            top_scores = torch.cat(
                (
                    top_scores,
                    torch.zeros(batch, pad, device=top_scores.device, dtype=top_scores.dtype),
                ),
                dim=1,
            )

        return gathered_embeddings, gathered_refs, top_scores


class LearnedGlobalSeeds(nn.Module):
    """Small learned global query bank for unmatched cases."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Parameter(torch.randn(config.q_global, config.model_dim) * 0.02)
        self.refs = nn.Parameter(torch.randn(config.q_global, 3) * 0.5)
        self.scores = nn.Parameter(torch.zeros(config.q_global))

    def forward(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        queries = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        refs = self.refs.unsqueeze(0).expand(batch_size, -1, -1)
        scores = self.scores.sigmoid().unsqueeze(0).expand(batch_size, -1)
        return queries, refs, scores


class TriSourceQueryRouter(nn.Module):
    """Route LiDAR, proposal, and global queries into a compact sparse bank."""

    SOURCE_LIDAR = 0
    SOURCE_PROPOSAL = 1
    SOURCE_GLOBAL = 2

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.source_embeddings = nn.Parameter(torch.randn(3, config.model_dim) * 0.02)
        self.score_head = nn.Sequential(
            nn.Linear(config.model_dim + 1, config.model_dim),
            nn.ReLU(),
            nn.Linear(config.model_dim, 1),
        )

    def forward(
        self,
        lidar_queries: Tensor,
        lidar_refs: Tensor,
        lidar_scores: Tensor,
        proposal_queries: Tensor,
        proposal_refs: Tensor,
        proposal_scores: Tensor,
        global_queries: Tensor,
        global_refs: Tensor,
        global_scores: Tensor,
    ) -> QuerySeedBank:
        """Fuse and route the three seed sources."""

        query_groups = [
            lidar_queries + self.source_embeddings[self.SOURCE_LIDAR],
            proposal_queries + self.source_embeddings[self.SOURCE_PROPOSAL],
            global_queries + self.source_embeddings[self.SOURCE_GLOBAL],
        ]
        ref_groups = [lidar_refs, proposal_refs, global_refs]
        score_groups = [lidar_scores, proposal_scores, global_scores]
        source_ids = [
            torch.full(
                lidar_scores.shape, self.SOURCE_LIDAR, device=lidar_scores.device, dtype=torch.long
            ),
            torch.full(
                proposal_scores.shape,
                self.SOURCE_PROPOSAL,
                device=proposal_scores.device,
                dtype=torch.long,
            ),
            torch.full(
                global_scores.shape,
                self.SOURCE_GLOBAL,
                device=global_scores.device,
                dtype=torch.long,
            ),
        ]

        all_queries = torch.cat(query_groups, dim=1)
        all_refs = torch.cat(ref_groups, dim=1)
        all_scores = torch.cat(score_groups, dim=1)
        all_source_ids = torch.cat(source_ids, dim=1)
        keep_logits = self.score_head(
            torch.cat((all_queries, all_scores.unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        keep_scores = keep_logits + all_scores

        keep_count = self.config.max_object_queries
        top_values, top_indices = torch.topk(keep_scores, k=keep_count, dim=-1)

        query_index = top_indices.unsqueeze(-1).expand(-1, -1, self.config.model_dim)
        refs_index = top_indices.unsqueeze(-1).expand(-1, -1, 3)
        gathered_queries = torch.gather(all_queries, 1, query_index)
        gathered_refs = torch.gather(all_refs, 1, refs_index)
        gathered_scores = torch.gather(all_scores, 1, top_indices)
        gathered_sources = torch.gather(all_source_ids, 1, top_indices)
        gathered_logits = torch.gather(keep_logits, 1, top_indices)
        return QuerySeedBank(
            embeddings=gathered_queries,
            refs_xyz=gathered_refs,
            scores=gathered_scores,
            source_ids=gathered_sources,
            keep_logits=gathered_logits,
        )
