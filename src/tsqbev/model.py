"""Minimal multimodal temporal sparse-query BEV model.

References:
- DETR3D sparse 3D-to-2D sampling:
  https://proceedings.mlr.press/v164/wang22b/wang22b.pdf
- PETRv2 temporal and multitask queries:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf
- StreamPETR persistent temporal state:
  https://arxiv.org/abs/2303.11926
- Sparse4D efficient sparse keypoint sampling:
  https://arxiv.org/pdf/2211.10581
- PersFormer lane-centric reasoning:
  https://arxiv.org/abs/2203.11089
- torchvision pretrained model API:
  https://pytorch.org/vision/stable/models.html
- MobileNetV3:
  https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf
- EfficientNet:
  https://proceedings.mlr.press/v97/tan19a/tan19a.pdf
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)

from tsqbev.config import ModelConfig
from tsqbev.contracts import (
    CameraProposals,
    MapPriorBatch,
    QuerySeedBank,
    SceneBatch,
    TeacherTargets,
    TemporalState,
)
from tsqbev.geometry import normalize_grid, project_points
from tsqbev.lidar import LidarSeedEncoder
from tsqbev.seeding import LearnedGlobalSeeds, ProposalRayInitializer, TriSourceQueryRouter
from tsqbev.teacher_seed import TeacherSeedEncoder, select_teacher_seed_indices

Tensor = torch.Tensor

_CENTER_OFFSET_RADIUS_M = 8.0
_MIN_BOX_SIZE_M = 0.1
_VELOCITY_RADIUS_MPS = 25.0
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _checkpoint_enabled(enabled: bool, *tensors: Tensor) -> bool:
    return enabled and torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def _resize_to_multiple(images: Tensor, multiple: int) -> Tensor:
    height, width = images.shape[-2:]
    target_height = max(multiple, ((height + multiple - 1) // multiple) * multiple)
    target_width = max(multiple, ((width + multiple - 1) // multiple) * multiple)
    if target_height == height and target_width == width:
        return images
    return F.interpolate(
        images,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


class TinyImageBackbone(nn.Module):
    """Two-scale CNN backbone for multi-view image features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.deep = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj8 = nn.Conv2d(128, config.model_dim, kernel_size=1)
        self.proj16 = nn.Conv2d(256, config.model_dim, kernel_size=1)

    def forward(self, images: Tensor) -> list[Tensor]:
        batch, views = images.shape[:2]
        x = images.reshape(batch * views, *images.shape[2:])
        f8 = self.stem(x)
        f16 = self.deep(f8)
        f8 = self.proj8(f8).reshape(batch, views, self.config.model_dim, f8.shape[-2], f8.shape[-1])
        f16 = self.proj16(f16).reshape(
            batch, views, self.config.model_dim, f16.shape[-2], f16.shape[-1]
        )
        return [f8, f16]


class TorchvisionImageBackbone(nn.Module):
    """Two-scale image backbone backed by official torchvision weights."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.freeze_backbone = config.freeze_image_backbone
        if config.image_backbone == "mobilenet_v3_large":
            weights = (
                MobileNet_V3_Large_Weights.DEFAULT if config.pretrained_image_backbone else None
            )
            self.extractor = mobilenet_v3_large(weights=weights).features
            self.low_stage_index = 4
            self.high_stage_index = 12
            low_channels = 40
            high_channels = 112
        elif config.image_backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained_image_backbone else None
            self.extractor = efficientnet_b0(weights=weights).features
            self.low_stage_index = 3
            self.high_stage_index = 5
            low_channels = 40
            high_channels = 112
        else:  # pragma: no cover - guarded by config validation and factory selection.
            raise ValueError(f"unsupported torchvision image backbone: {config.image_backbone}")
        if self.freeze_backbone:
            self.extractor.requires_grad_(False)
            self.extractor.eval()
        self.proj8 = nn.Conv2d(low_channels, config.model_dim, kernel_size=1)
        self.proj16 = nn.Conv2d(high_channels, config.model_dim, kernel_size=1)

    def _extract_multiscale(self, images: Tensor) -> tuple[Tensor, Tensor]:
        low: Tensor | None = None
        high: Tensor | None = None
        x = images
        for index, layer in enumerate(self.extractor):
            x = layer(x)
            if index == self.low_stage_index:
                low = x
            if index == self.high_stage_index:
                high = x
                break
        if low is None or high is None:
            raise RuntimeError(
                "torchvision backbone did not emit the requested multiscale features"
            )
        return low, high

    def train(self, mode: bool = True) -> TorchvisionImageBackbone:
        super().train(mode)
        if self.freeze_backbone:
            self.extractor.eval()
        return self

    def forward(self, images: Tensor) -> list[Tensor]:
        batch, views = images.shape[:2]
        x = images.reshape(batch * views, *images.shape[2:])
        if self.freeze_backbone:
            with torch.no_grad():
                f8, f16 = self._extract_multiscale(x)
        else:
            f8, f16 = self._extract_multiscale(x)
        f8 = self.proj8(f8).reshape(batch, views, self.config.model_dim, f8.shape[-2], f8.shape[-1])
        f16 = self.proj16(f16).reshape(
            batch, views, self.config.model_dim, f16.shape[-2], f16.shape[-1]
        )
        return [f8, f16]


class DINOv2ProjectorBackbone(nn.Module):
    """Projected multiscale camera backbone from official DINOv2 intermediate features.

    References:
    - DINOv2 official repo and hub loaders:
      https://github.com/facebookresearch/dinov2
    - BEVFormer v2 perspective supervision:
      https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.freeze_backbone = config.freeze_image_backbone
        self.activation_checkpointing = config.activation_checkpointing
        repo_root = Path(config.foundation_repo_root or "/home/achbogga/projects/dinov2")
        if not repo_root.exists():
            raise FileNotFoundError(
                f"DINOv2 repo root `{repo_root}` does not exist; clone the official repo first"
            )
        self.extractor = torch.hub.load(
            str(repo_root),
            config.image_backbone,
            source="local",
            pretrained=config.pretrained_image_backbone,
        )
        self.layers = list(config.foundation_intermediate_layers)
        embed_dim = int(self.extractor.embed_dim)
        self.low_proj = nn.Conv2d(embed_dim, config.model_dim, kernel_size=1)
        self.high_proj = nn.Conv2d(embed_dim, config.model_dim, kernel_size=1)
        self.high_downsample = nn.Sequential(
            nn.Conv2d(config.model_dim, config.model_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )
        if self.freeze_backbone:
            self.extractor.requires_grad_(False)
            self.extractor.eval()

    def train(self, mode: bool = True) -> DINOv2ProjectorBackbone:
        super().train(mode)
        if self.freeze_backbone:
            self.extractor.eval()
        return self

    def _extract_multiscale(self, images: Tensor) -> tuple[Tensor, Tensor]:
        image_mean = cast(Tensor, self.image_mean).to(dtype=images.dtype, device=images.device)
        image_std = cast(Tensor, self.image_std).to(dtype=images.dtype, device=images.device)
        normalized = (images - image_mean) / image_std
        resized = _resize_to_multiple(normalized, self.config.foundation_patch_multiple)
        outputs = self.extractor.get_intermediate_layers(
            resized,
            n=self.layers,
            reshape=True,
            norm=True,
        )
        if len(outputs) != 2:
            raise RuntimeError(
                "DINOv2 backbone must return exactly two intermediate feature maps for TSQBEV"
            )
        low, high = outputs
        return low, high

    def forward(self, images: Tensor) -> list[Tensor]:
        batch, views = images.shape[:2]
        x = images.reshape(batch * views, *images.shape[2:])
        if self.freeze_backbone:
            with torch.no_grad():
                low, high = self._extract_multiscale(x)
        elif _checkpoint_enabled(self.activation_checkpointing, x):
            low, high = checkpoint(self._extract_multiscale, x, use_reentrant=False)
        else:
            low, high = self._extract_multiscale(x)
        low = self.low_proj(low)
        high = self.high_proj(high)
        high = self.high_downsample(high)
        low = low.reshape(batch, views, self.config.model_dim, low.shape[-2], low.shape[-1])
        high = high.reshape(batch, views, self.config.model_dim, high.shape[-2], high.shape[-1])
        return [low, high]


def build_image_backbone(config: ModelConfig) -> nn.Module:
    """Construct the configured image backbone."""

    if config.image_backbone == "tiny":
        return TinyImageBackbone(config)
    if config.image_backbone.startswith("dinov2_"):
        return DINOv2ProjectorBackbone(config)
    return TorchvisionImageBackbone(config)


class ProposalHead(nn.Module):
    """Lightweight top-k 2D proposal generator from image features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.score = nn.Conv2d(config.model_dim, 1, kernel_size=1)

    def forward(self, f8: Tensor, image_height: int, image_width: int) -> CameraProposals:
        batch, views = f8.shape[:2]
        scores_map = torch.sigmoid(self.score(f8.reshape(batch * views, *f8.shape[2:])))
        scores_map = scores_map.reshape(batch, views, scores_map.shape[-2], scores_map.shape[-1])
        flat_scores = scores_map.flatten(2)
        top_values, top_indices = torch.topk(flat_scores, k=self.config.proposals_per_view, dim=-1)
        feat_h, feat_w = scores_map.shape[-2:]
        ys = (top_indices // feat_w).float() * (image_height / feat_h)
        xs = (top_indices % feat_w).float() * (image_width / feat_w)
        half = self.config.proposal_box_size_px * 0.5
        boxes = torch.stack((xs - half, ys - half, xs + half, ys + half), dim=-1)
        return CameraProposals(boxes_xyxy=boxes, scores=top_values)


class CrossViewSparseSampler(nn.Module):
    """Project sparse 3D references and sample camera features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        keypoints = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [config.sample_radius_m, 0.0, 0.0],
                [0.0, config.sample_radius_m, 0.0],
                [-config.sample_radius_m, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("keypoints", keypoints[: config.sample_keypoints])

    def forward(
        self, features: list[Tensor], refs_xyz: Tensor, intrinsics: Tensor, extrinsics: Tensor
    ) -> Tensor:
        batch, queries = refs_xyz.shape[:2]
        sampled = torch.zeros(
            batch, queries, features[0].shape[2], device=refs_xyz.device, dtype=refs_xyz.dtype
        )
        for feature_map in features[: self.config.feature_levels]:
            _, views, channels, height, width = feature_map.shape
            keypoints = cast(Tensor, self.keypoints)
            refs = refs_xyz.unsqueeze(-2) + keypoints.view(1, 1, self.config.sample_keypoints, 3)
            refs = refs.reshape(batch, queries * self.config.sample_keypoints, 3)
            for batch_index in range(batch):
                batch_sum = torch.zeros(
                    queries, channels, device=refs_xyz.device, dtype=refs_xyz.dtype
                )
                batch_count = torch.zeros(queries, 1, device=refs_xyz.device, dtype=refs_xyz.dtype)
                for view_index in range(views):
                    uvz = project_points(
                        refs[batch_index],
                        intrinsics[batch_index, view_index],
                        extrinsics[batch_index, view_index],
                    )
                    valid = uvz[:, 2] > 0.0
                    grid = normalize_grid(uvz[:, :2], height=height, width=width)
                    grid = grid.view(1, queries * self.config.sample_keypoints, 1, 2)
                    sampled_map = F.grid_sample(
                        feature_map[batch_index, view_index].unsqueeze(0),
                        grid,
                        align_corners=True,
                    )
                    sampled_map = sampled_map.squeeze(0).squeeze(-1).transpose(0, 1)
                    sampled_map = sampled_map.view(queries, self.config.sample_keypoints, channels)
                    valid = valid.view(queries, self.config.sample_keypoints, 1).float()
                    batch_sum += (sampled_map * valid).sum(dim=1)
                    batch_count += valid.sum(dim=1)
                batch_count = batch_count.clamp_min(1.0)
                sampled[batch_index] += batch_sum / batch_count
        sampled = sampled / float(self.config.feature_levels)
        return sampled


class QueryFusionBlock(nn.Module):
    """Minimal query-update block using sampled camera features."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, queries: Tensor, sampled_features: Tensor) -> Tensor:
        updated = self.update(torch.cat((queries, sampled_features), dim=-1))
        return queries + updated


class TemporalStateUpdater(nn.Module):
    """Sparse temporal fusion for persistent queries."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.model_dim * 2, config.model_dim)

    def forward(
        self, queries: Tensor, refs_xyz: Tensor, state: TemporalState | None
    ) -> tuple[Tensor, Tensor]:
        if state is None:
            return queries, refs_xyz
        gate = torch.sigmoid(self.gate(torch.cat((queries, state.object_queries), dim=-1)))
        fused_queries = gate * queries + (1.0 - gate) * state.object_queries
        fused_refs = (
            self.config.temporal_momentum * refs_xyz
            + (1.0 - self.config.temporal_momentum) * state.object_refs
        )
        return fused_queries, fused_refs


class ObjectHead(nn.Module):
    """Minimal object head with bounded center refinement."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cls = nn.Linear(config.model_dim, config.num_object_classes)
        self.objectness = nn.Linear(config.model_dim, 1)
        self.center_offset = nn.Linear(config.model_dim, 3)
        self.size = nn.Linear(config.model_dim, 3)
        self.yaw = nn.Linear(config.model_dim, 2)
        self.velocity = nn.Linear(config.model_dim, 2)
        self.anchor_objectness_scale = nn.Parameter(torch.tensor(1.0))
        self.anchor_class_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        queries: Tensor,
        refs_xyz: Tensor,
        prior_labels: Tensor | None = None,
        prior_scores: Tensor | None = None,
        prior_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        objectness_logits = self.objectness(queries).squeeze(-1)
        class_logits = self.cls(queries)
        if (
            prior_labels is not None
            and prior_scores is not None
            and prior_valid_mask is not None
            and bool(prior_valid_mask.any())
        ):
            clipped_prior = prior_scores.clamp(0.05, 0.95)
            prior_logits = torch.logit(clipped_prior)
            prior_mask = prior_valid_mask.to(dtype=queries.dtype)
            objectness_logits = (
                objectness_logits
                + self.anchor_objectness_scale * prior_logits * prior_mask
            )
            one_hot = F.one_hot(
                prior_labels.to(torch.long),
                num_classes=class_logits.shape[-1],
            ).to(dtype=class_logits.dtype)
            class_logits = class_logits + (
                self.anchor_class_scale
                * prior_logits.unsqueeze(-1)
                * one_hot
                * prior_mask.unsqueeze(-1)
            )

        center_delta = torch.tanh(self.center_offset(queries)) * _CENTER_OFFSET_RADIUS_M
        centers = refs_xyz + center_delta
        sizes = F.softplus(self.size(queries)) + _MIN_BOX_SIZE_M
        yaw_vec = F.normalize(self.yaw(queries), dim=-1, eps=1e-6)
        yaw = torch.atan2(yaw_vec[..., 0], yaw_vec[..., 1])
        velocity = torch.tanh(self.velocity(queries)) * _VELOCITY_RADIUS_MPS

        boxes = torch.cat((centers, sizes, yaw.unsqueeze(-1), velocity), dim=-1)
        return objectness_logits, class_logits, boxes


class LaneHead(nn.Module):
    """Camera-dominant lane head with optional LiDAR ground hint and map priors."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.map_projector = nn.Linear(config.map_input_dim, config.model_dim)
        self.ground_projector = nn.Linear(config.model_dim, config.model_dim)
        self.query_bank = nn.Parameter(torch.randn(config.lane_queries, config.model_dim) * 0.02)
        self.attn = SDPACrossAttention(
            config.model_dim,
            num_heads=4,
            backend=config.attention_backend,
            activation_checkpointing=config.activation_checkpointing,
        )
        self.lane_logits = nn.Linear(config.model_dim, 1)
        self.lane_points = nn.Linear(config.model_dim, config.lane_points * 3)

    def forward(
        self, features: list[Tensor], object_queries: Tensor, map_priors: MapPriorBatch | None
    ) -> tuple[Tensor, Tensor]:
        batch = features[0].shape[0]
        camera_tokens = features[0].mean(dim=(-2, -1))
        ground_token = self.ground_projector(object_queries.mean(dim=1, keepdim=True))
        memory = torch.cat((camera_tokens, ground_token), dim=1)
        if map_priors is not None:
            projected_map = self.map_projector(map_priors.tokens)
            memory = torch.cat((memory, projected_map), dim=1)
        lane_queries = self.query_bank.unsqueeze(0).expand(batch, -1, -1)
        attended = self.attn(lane_queries, memory)
        lane_logits = self.lane_logits(attended).squeeze(-1)
        lane_polylines = self.lane_points(attended).view(
            batch, self.config.lane_queries, self.config.lane_points, 3
        )
        return lane_logits, lane_polylines


class SDPACrossAttention(nn.Module):
    """Cross-attention with explicit SDPA backend selection.

    References:
    - PyTorch scaled dot product attention:
      https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    - Sparse4D multi-view sparse attention motivation:
      https://arxiv.org/pdf/2211.10581
    """

    def __init__(
        self,
        model_dim: int,
        *,
        num_heads: int,
        backend: str,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.backend = backend
        self.activation_checkpointing = activation_checkpointing
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def _sdpa_context(self, device: torch.device) -> AbstractContextManager[object]:
        if self.backend == "auto" or device.type != "cuda":
            return nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except Exception:  # pragma: no cover - depends on torch build.
            return nullcontext()
        if self.backend == "flash":
            major, _minor = torch.cuda.get_device_capability(device)
            if major < 8:
                return nullcontext()
        backend_map = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }
        selected = backend_map.get(self.backend)
        if selected is None:
            return nullcontext()
        return sdpa_kernel(backends=[selected])

    def _forward_impl(self, queries: Tensor, context: Tensor) -> Tensor:
        batch, query_count, _ = queries.shape
        key_count = context.shape[1]
        q = self.q_proj(queries).view(batch, query_count, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(batch, key_count, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(batch, key_count, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with self._sdpa_context(queries.device):
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, query_count, self.model_dim)
        return self.out_proj(attended)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        if _checkpoint_enabled(self.activation_checkpointing, queries, context):
            return checkpoint(self._forward_impl, queries, context, use_reentrant=False)
        return self._forward_impl(queries, context)


class TSQBEVCore(nn.Module):
    """Export-friendly core that consumes prepared LiDAR seeds."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = build_image_backbone(config)
        self.proposal_head = ProposalHead(config)
        self.proposal_init = ProposalRayInitializer(config)
        self.global_seeds = LearnedGlobalSeeds(config)
        self.router = TriSourceQueryRouter(config)
        self.sampler = CrossViewSparseSampler(config)
        self.fusion = QueryFusionBlock(config.model_dim)
        self.temporal = TemporalStateUpdater(config)
        self.object_head = ObjectHead(config)
        self.lane_head = LaneHead(config)

    def forward(
        self,
        images: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        lidar_queries: Tensor,
        lidar_refs: Tensor,
        lidar_scores: Tensor,
        lidar_prior_labels: Tensor | None = None,
        lidar_prior_scores: Tensor | None = None,
        lidar_prior_valid_mask: Tensor | None = None,
        camera_proposals: CameraProposals | None = None,
        proposal_queries: Tensor | None = None,
        proposal_refs: Tensor | None = None,
        proposal_scores: Tensor | None = None,
        map_priors: MapPriorBatch | None = None,
        state: TemporalState | None = None,
    ) -> dict[str, Tensor | QuerySeedBank | TemporalState]:
        features = self.backbone(images)
        image_height, image_width = images.shape[-2:]
        if proposal_queries is None or proposal_refs is None or proposal_scores is None:
            proposals = (
                camera_proposals
                if camera_proposals is not None
                else self.proposal_head(features[0], image_height, image_width)
            )
            proposal_queries, proposal_refs, proposal_scores = self.proposal_init(
                proposals,
                intrinsics,
                extrinsics,
                image_height=image_height,
                image_width=image_width,
            )
        global_queries, global_refs, global_scores = self.global_seeds(images.shape[0])
        seed_bank = self.router(
            lidar_queries,
            lidar_refs,
            lidar_scores,
            lidar_prior_labels,
            lidar_prior_scores,
            lidar_prior_valid_mask,
            proposal_queries,
            proposal_refs,
            proposal_scores,
            global_queries,
            global_refs,
            global_scores,
        )
        sampled = self.sampler(features, seed_bank.refs_xyz, intrinsics, extrinsics)
        fused_queries = self.fusion(seed_bank.embeddings, sampled)
        fused_queries, fused_refs = self.temporal(fused_queries, seed_bank.refs_xyz, state)
        objectness_logits, object_logits, object_boxes = self.object_head(
            fused_queries,
            fused_refs,
            prior_labels=seed_bank.prior_labels,
            prior_scores=seed_bank.prior_scores,
            prior_valid_mask=seed_bank.prior_valid_mask,
        )
        lane_logits, lane_polylines = self.lane_head(features, fused_queries, map_priors)
        temporal_state = TemporalState(object_queries=fused_queries, object_refs=fused_refs)
        return {
            "objectness_logits": objectness_logits,
            "object_logits": object_logits,
            "object_boxes": object_boxes,
            "lane_logits": lane_logits,
            "lane_polylines": lane_polylines,
            "seed_bank": seed_bank,
            "temporal_state": temporal_state,
        }


class TSQBEVModel(nn.Module):
    """Top-level model that preprocesses raw LiDAR points before calling the core."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.lidar_encoder = LidarSeedEncoder(config)
        self.teacher_seed_encoder = (
            TeacherSeedEncoder(config) if config.teacher_seed_mode == "replace_lidar" else None
        )
        self.core = TSQBEVCore(config)

    def _replace_lidar_refs_from_teacher(
        self,
        lidar_refs: Tensor,
        lidar_scores: Tensor,
        teacher_targets: TeacherTargets | None,
    ) -> tuple[Tensor, Tensor]:
        if (
            teacher_targets is None
            or teacher_targets.object_boxes is None
            or teacher_targets.object_scores is None
        ):
            return lidar_refs, lidar_scores

        refs = lidar_refs.clone()
        scores = lidar_scores.clone()
        valid_mask = (
            teacher_targets.valid_mask
            if teacher_targets.valid_mask is not None
            else torch.ones_like(teacher_targets.object_scores, dtype=torch.bool)
        )
        for batch_index in range(refs.shape[0]):
            valid = valid_mask[batch_index]
            if not bool(valid.any()):
                continue
            teacher_boxes = teacher_targets.object_boxes[batch_index][valid]
            teacher_labels = (
                teacher_targets.object_labels[batch_index][valid]
                if teacher_targets.object_labels is not None
                else None
            )
            teacher_scores = teacher_targets.object_scores[batch_index][valid]
            if teacher_labels is None:
                keep = torch.argsort(teacher_scores, descending=True)[: self.config.q_lidar]
            else:
                keep = select_teacher_seed_indices(
                    teacher_labels,
                    teacher_scores,
                    max_keep=self.config.q_lidar,
                    mode=self.config.teacher_seed_selection_mode,
                )
            count = int(keep.numel())
            refs[batch_index, :count] = teacher_boxes[keep, :3]
            scores[batch_index, :count] = teacher_scores[keep]
        return refs, scores

    def forward(
        self, batch: SceneBatch, state: TemporalState | None = None
    ) -> dict[str, Tensor | QuerySeedBank | TemporalState]:
        batch.validate()
        teacher_seed_bank = None
        teacher_prior_labels = None
        teacher_prior_scores = None
        teacher_prior_valid_mask = None
        if (
            self.config.teacher_seed_mode == "replace_lidar"
            and self.teacher_seed_encoder is not None
        ):
            encoded = self.teacher_seed_encoder.encode_with_priors(batch.teacher_targets)
            if encoded is not None:
                (
                    teacher_queries,
                    teacher_refs,
                    teacher_scores,
                    teacher_prior_labels,
                    teacher_prior_scores,
                    teacher_prior_valid_mask,
                ) = encoded
                teacher_seed_bank = (teacher_queries, teacher_refs, teacher_scores)
        if teacher_seed_bank is None:
            lidar_queries, lidar_refs, lidar_scores = self.lidar_encoder(
                batch.lidar_points,
                batch.lidar_mask,
            )
        else:
            lidar_queries, lidar_refs, lidar_scores = teacher_seed_bank
        if self.config.teacher_seed_mode == "replace_lidar_refs":
            lidar_refs, lidar_scores = self._replace_lidar_refs_from_teacher(
                lidar_refs,
                lidar_scores,
                batch.teacher_targets,
            )
        return self.core(
            images=batch.images,
            intrinsics=batch.intrinsics,
            extrinsics=batch.extrinsics,
            lidar_queries=lidar_queries,
            lidar_refs=lidar_refs,
            lidar_scores=lidar_scores,
            lidar_prior_labels=teacher_prior_labels,
            lidar_prior_scores=teacher_prior_scores,
            lidar_prior_valid_mask=teacher_prior_valid_mask,
            camera_proposals=batch.camera_proposals,
            map_priors=batch.map_priors,
            state=state,
        )
