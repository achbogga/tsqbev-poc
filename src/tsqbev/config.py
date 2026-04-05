"""Typed configuration models for the minimal TSQBEV proof of concept.

References:
- Sparse4D: https://arxiv.org/pdf/2211.10581
- PETRv2: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf
- HotBEV: https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class PillarConfig(BaseModel):
    """Configuration for lightweight LiDAR pillarization."""

    x_range: tuple[float, float] = (-54.0, 54.0)
    y_range: tuple[float, float] = (-54.0, 54.0)
    z_range: tuple[float, float] = (-5.0, 5.0)
    pillar_size_xy: tuple[float, float] = (0.6, 0.6)
    q_lidar: int = 256


class ModelConfig(BaseModel):
    """Configuration for the multimodal temporal sparse-query model."""

    image_channels: int = 3
    image_backbone: Literal[
        "tiny",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "dinov2_vits14_reg",
        "dinov2_vitb14_reg",
        "dinov3_vits16",
        "dinov3_vitb16",
    ] = "tiny"
    pretrained_image_backbone: bool = False
    freeze_image_backbone: bool = False
    foundation_repo_root: str | None = None
    foundation_weights: str | None = None
    foundation_intermediate_layers: tuple[int, int] = (8, 11)
    foundation_patch_multiple: int = 14
    activation_checkpointing: bool = False
    attention_backend: Literal["auto", "math", "flash", "efficient", "cudnn"] = "auto"
    auto_vram_fit: bool = False
    sam2_repo_root: str | None = None
    sam2_model_cfg: str | None = None
    sam2_checkpoint: str | None = None
    sam2_region_prior_mode: Literal["off", "proposal_boxes"] = "off"
    sam2_region_prior_weight: float = 0.0
    router_mode: Literal["tri_source", "anchor_first"] = "tri_source"
    views: int = 6
    model_dim: int = 256
    num_object_classes: int = 10
    q_lidar: int = 256
    q_2d: int = 192
    q_global: int = 64
    max_object_queries: int = 384
    lane_queries: int = 128
    lane_points: int = 20
    proposals_per_view: int = 32
    num_depth_bins: int = 8
    sample_keypoints: int = 4
    feature_levels: int = 2
    temporal_frames: int = 2
    temporal_momentum: float = 0.5
    proposal_box_size_px: float = 32.0
    sample_radius_m: float = 1.25
    map_input_dim: int = 256
    dropout_lidar_probability: float = 0.2
    teacher_seed_mode: Literal["off", "replace_lidar", "replace_lidar_refs"] = "off"
    teacher_seed_selection_mode: Literal["score_topk", "class_balanced_round_robin"] = "score_topk"
    ranking_mode: Literal["class_times_objectness", "quality_class_only"] = (
        "class_times_objectness"
    )
    anchor_first_min_proposal: int = 0
    anchor_first_min_global: int = 0
    pillar: PillarConfig = Field(default_factory=PillarConfig)

    @model_validator(mode="after")
    def _validate_query_budget(self) -> Self:
        total_seed_queries = self.q_lidar + self.q_2d + self.q_global
        if total_seed_queries < self.max_object_queries:
            raise ValueError("total seed queries must be at least max_object_queries")
        if not 0.0 <= self.temporal_momentum <= 1.0:
            raise ValueError("temporal_momentum must be in [0, 1]")
        if not 0.0 <= self.dropout_lidar_probability <= 1.0:
            raise ValueError("dropout_lidar_probability must be in [0, 1]")
        if self.sample_keypoints <= 0:
            raise ValueError("sample_keypoints must be positive")
        if self.feature_levels not in {1, 2}:
            raise ValueError("feature_levels must be 1 or 2 for the minimal POC")
        if self.image_backbone == "tiny" and self.pretrained_image_backbone:
            raise ValueError("the tiny fallback backbone does not have pretrained weights")
        if self.sam2_region_prior_weight < 0.0:
            raise ValueError("sam2_region_prior_weight must be non-negative")
        if self.sam2_region_prior_mode == "off" and self.sam2_region_prior_weight > 0.0:
            raise ValueError("sam2_region_prior_weight requires a non-off sam2_region_prior_mode")
        if self.foundation_patch_multiple <= 0:
            raise ValueError("foundation_patch_multiple must be positive")
        low_layer, high_layer = self.foundation_intermediate_layers
        if low_layer < 0 or high_layer < 0:
            raise ValueError("foundation_intermediate_layers must be non-negative")
        if high_layer <= low_layer:
            raise ValueError("foundation_intermediate_layers must be strictly increasing")
        if self.anchor_first_min_proposal < 0 or self.anchor_first_min_global < 0:
            raise ValueError("anchor_first source reserves must be non-negative")
        if self.anchor_first_min_proposal > self.q_2d:
            raise ValueError("anchor_first proposal reserve cannot exceed q_2d")
        if self.anchor_first_min_global > self.q_global:
            raise ValueError("anchor_first global reserve cannot exceed q_global")
        if self.anchor_first_min_proposal + self.anchor_first_min_global > self.max_object_queries:
            raise ValueError("anchor_first source reserves cannot exceed max_object_queries")
        return self

    @classmethod
    def small(cls) -> ModelConfig:
        """Return a light config for tests and smoke runs."""

        return cls(
            image_backbone="tiny",
            pretrained_image_backbone=False,
            freeze_image_backbone=False,
            model_dim=64,
            q_lidar=24,
            q_2d=16,
            q_global=8,
            max_object_queries=24,
            lane_queries=12,
            lane_points=8,
            proposals_per_view=8,
            num_depth_bins=4,
            map_input_dim=64,
            pillar=PillarConfig(q_lidar=24),
        )

    @classmethod
    def rtx5000_nuscenes_baseline(cls) -> ModelConfig:
        """Return a conservative pretrained baseline tuned for local RTX 5000 runs.

        References:
        - torchvision pretrained model API:
          https://pytorch.org/vision/stable/models.html
        - MobileNetV3:
          https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf
        """

        return cls(
            image_backbone="mobilenet_v3_large",
            pretrained_image_backbone=True,
            freeze_image_backbone=True,
            model_dim=128,
            q_lidar=96,
            q_2d=64,
            q_global=32,
            max_object_queries=96,
            lane_queries=32,
            lane_points=12,
            proposals_per_view=16,
            num_depth_bins=6,
            map_input_dim=128,
            pillar=PillarConfig(q_lidar=96),
        )

    @classmethod
    def rtx5000_nuscenes_teacher_bootstrap(cls) -> ModelConfig:
        """Return the local baseline with external teacher seeds enabled."""

        return cls.rtx5000_nuscenes_baseline().model_copy(
            update={
                "teacher_seed_mode": "replace_lidar",
                "teacher_seed_selection_mode": "class_balanced_round_robin",
                "router_mode": "anchor_first",
                "ranking_mode": "quality_class_only",
            }
        )

    @classmethod
    def rtx5000_nuscenes_dinov2_teacher(cls) -> ModelConfig:
        """Return the DINOv2-projected teacher-seeded baseline for local RTX 5000 runs.

        References:
        - DINOv2 model card:
          https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
        - BEVFormer v2 perspective supervision:
          https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf
        """

        return cls(
            image_backbone="dinov2_vits14_reg",
            pretrained_image_backbone=True,
            freeze_image_backbone=True,
            foundation_repo_root="/home/achbogga/projects/dinov2",
            foundation_intermediate_layers=(8, 11),
            foundation_patch_multiple=14,
            activation_checkpointing=False,
            attention_backend="auto",
            router_mode="anchor_first",
            teacher_seed_mode="replace_lidar",
            teacher_seed_selection_mode="class_balanced_round_robin",
            ranking_mode="quality_class_only",
            model_dim=128,
            q_lidar=96,
            q_2d=80,
            q_global=32,
            max_object_queries=112,
            lane_queries=32,
            lane_points=12,
            proposals_per_view=24,
            num_depth_bins=6,
            map_input_dim=128,
            pillar=PillarConfig(q_lidar=96),
        )

    @classmethod
    def rtx5000_nuscenes_dinov3_teacher(cls) -> ModelConfig:
        """Return the DINOv3-projected teacher-seeded baseline for local RTX 5000 runs.

        References:
        - DINOv3 official repo and model card:
          https://github.com/facebookresearch/dinov3
        - BEVFormer v2 perspective supervision:
          https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf
        """

        return cls(
            image_backbone="dinov3_vits16",
            pretrained_image_backbone=True,
            freeze_image_backbone=True,
            foundation_repo_root="/home/achbogga/projects/dinov3",
            foundation_intermediate_layers=(8, 11),
            foundation_patch_multiple=16,
            activation_checkpointing=True,
            attention_backend="auto",
            auto_vram_fit=True,
            router_mode="anchor_first",
            teacher_seed_mode="replace_lidar",
            teacher_seed_selection_mode="class_balanced_round_robin",
            ranking_mode="quality_class_only",
            model_dim=128,
            q_lidar=96,
            q_2d=80,
            q_global=32,
            max_object_queries=112,
            lane_queries=32,
            lane_points=12,
            proposals_per_view=24,
            num_depth_bins=6,
            map_input_dim=128,
            pillar=PillarConfig(q_lidar=96),
            sam2_region_prior_mode="proposal_boxes",
            sam2_region_prior_weight=0.05,
        )

    @classmethod
    def rtx5000_nuscenes_query_boost(cls) -> ModelConfig:
        """Return the current best local mini recipe configuration.

        This matches the promoted query-boost exploitation recipe from the bounded
        `v1.0-mini` loop: proposal-heavy frozen MobileNetV3 with a small extra
        query budget.
        """

        baseline = cls.rtx5000_nuscenes_baseline()
        return baseline.model_copy(
            update={
                "q_lidar": 64,
                "q_2d": 112,
                "q_global": 32,
                "max_object_queries": 112,
                "proposals_per_view": 32,
                "pillar": baseline.pillar.model_copy(update={"q_lidar": 64}),
            }
        )

    @classmethod
    def rtx5000_nuscenes_teacher_quality_plus(cls) -> ModelConfig:
        """Return the current best MobileNet teacher-quality winner-line config.

        This matches the selected recipe structure from the `v28` continuation frontier.
        """

        baseline = cls.rtx5000_nuscenes_baseline()
        return baseline.model_copy(
            update={
                "pretrained_image_backbone": True,
                "freeze_image_backbone": False,
                "router_mode": "anchor_first",
                "teacher_seed_mode": "replace_lidar",
                "teacher_seed_selection_mode": "class_balanced_round_robin",
                "ranking_mode": "quality_class_only",
                "q_lidar": 96,
                "q_2d": 80,
                "q_global": 32,
                "max_object_queries": 112,
                "proposals_per_view": 24,
                "pillar": baseline.pillar.model_copy(update={"q_lidar": 96}),
            }
        )


class LatencyPredictorConfig(BaseModel):
    """Simple latency-predictor coefficients and gates."""

    b0: float = 8.0
    b1: float = 0.45
    b2: float = 0.03
    b3: float = 0.002
    b4: float = 4.0
    b5: float = 0.08
    production_p95_ms: float = 100.0
    stretch_p95_ms: float = 50.0
