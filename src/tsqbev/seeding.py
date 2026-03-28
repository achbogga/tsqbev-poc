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

    def _normalize_source_scores(
        self,
        scores: Tensor,
        *,
        source_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Normalize source confidences before routing.

        Repo note:
        The public mini nuScenes diagnosis showed that raw LiDAR pillar counts can be
        orders of magnitude larger than proposal confidences. The router therefore
        normalizes each source independently before cross-source ranking so the final
        sparse bank remains truly multimodal.
        """

        if source_id == self.SOURCE_LIDAR:
            source_scores = torch.log1p(scores)
            valid_mask = scores > 0
        elif source_id == self.SOURCE_GLOBAL:
            source_scores = scores
            valid_mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            source_scores = scores
            valid_mask = scores > 0

        normalized = torch.zeros_like(source_scores)
        for batch_index in range(source_scores.shape[0]):
            valid = valid_mask[batch_index]
            if not bool(valid.any()):
                continue
            values = source_scores[batch_index, valid]
            min_value = values.min()
            max_value = values.max()
            if float((max_value - min_value).abs()) < 1e-6:
                normalized[batch_index, valid] = 1.0
                continue
            normalized[batch_index, valid] = (values - min_value) / (max_value - min_value)
        return normalized, valid_mask

    def _source_keep_counts(self) -> list[int]:
        budgets = torch.tensor(
            [self.config.q_lidar, self.config.q_2d, self.config.q_global],
            dtype=torch.float32,
        )
        raw = budgets / budgets.sum() * float(self.config.max_object_queries)
        counts = torch.floor(raw).to(torch.int64)
        remainder = int(self.config.max_object_queries - int(counts.sum()))
        if remainder > 0:
            fractional = raw - counts.float()
            order = torch.argsort(fractional, descending=True)
            for source_index in order[:remainder]:
                counts[int(source_index)] += 1
        return [int(value) for value in counts.tolist()]

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

        lidar_scores, lidar_valid = self._normalize_source_scores(
            lidar_scores,
            source_id=self.SOURCE_LIDAR,
        )
        proposal_scores, proposal_valid = self._normalize_source_scores(
            proposal_scores,
            source_id=self.SOURCE_PROPOSAL,
        )
        global_scores, global_valid = self._normalize_source_scores(
            global_scores,
            source_id=self.SOURCE_GLOBAL,
        )

        query_groups = [
            lidar_queries + self.source_embeddings[self.SOURCE_LIDAR],
            proposal_queries + self.source_embeddings[self.SOURCE_PROPOSAL],
            global_queries + self.source_embeddings[self.SOURCE_GLOBAL],
        ]
        ref_groups = [lidar_refs, proposal_refs, global_refs]
        score_groups = [lidar_scores, proposal_scores, global_scores]
        valid_groups = [lidar_valid, proposal_valid, global_valid]
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
        all_valid = torch.cat(valid_groups, dim=1)
        all_source_ids = torch.cat(source_ids, dim=1)
        keep_logits = self.score_head(
            torch.cat((all_queries, all_scores.unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        keep_scores = keep_logits + all_scores

        keep_count = self.config.max_object_queries
        source_keep_counts = self._source_keep_counts()
        source_offsets = [
            0,
            lidar_queries.shape[1],
            lidar_queries.shape[1] + proposal_queries.shape[1],
        ]
        top_indices_rows: list[Tensor] = []

        for batch_index in range(all_queries.shape[0]):
            selected: list[int] = []
            selected_mask = torch.zeros(
                all_queries.shape[1],
                device=all_queries.device,
                dtype=torch.bool,
            )
            for source_index, (offset, _group_scores, group_valid, keep_for_source) in enumerate(
                zip(source_offsets, score_groups, valid_groups, source_keep_counts, strict=True)
            ):
                del source_index
                valid_indices = torch.nonzero(group_valid[batch_index], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0 or keep_for_source <= 0:
                    continue
                source_keep = min(int(valid_indices.numel()), keep_for_source)
                source_scores = keep_scores[batch_index, offset + valid_indices]
                source_order = torch.argsort(source_scores, descending=True)
                chosen = valid_indices[source_order[:source_keep]] + offset
                selected.extend(int(index) for index in chosen.tolist())
                selected_mask[chosen] = True

            if len(selected) < keep_count:
                remaining_valid = torch.nonzero(
                    all_valid[batch_index] & ~selected_mask,
                    as_tuple=False,
                )
                remaining_valid = remaining_valid.squeeze(-1)
                if remaining_valid.numel() > 0:
                    remaining_scores = keep_scores[batch_index, remaining_valid]
                    fill_count = min(keep_count - len(selected), int(remaining_valid.numel()))
                    remaining_order = torch.argsort(remaining_scores, descending=True)
                    fill = remaining_valid[remaining_order[:fill_count]]
                    selected.extend(int(index) for index in fill.tolist())
                    selected_mask[fill] = True

            if len(selected) < keep_count:
                remaining_any = torch.nonzero(~selected_mask, as_tuple=False).squeeze(-1)
                fill_count = min(keep_count - len(selected), int(remaining_any.numel()))
                if fill_count > 0:
                    remaining_scores = keep_scores[batch_index, remaining_any]
                    remaining_order = torch.argsort(remaining_scores, descending=True)
                    fill = remaining_any[remaining_order[:fill_count]]
                    selected.extend(int(index) for index in fill.tolist())

            selected_tensor = torch.tensor(selected, device=all_queries.device, dtype=torch.long)
            selected_scores = keep_scores[batch_index, selected_tensor]
            selected_order = torch.argsort(selected_scores, descending=True)
            top_indices_rows.append(selected_tensor[selected_order[:keep_count]])

        top_indices = torch.stack(top_indices_rows, dim=0)
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
