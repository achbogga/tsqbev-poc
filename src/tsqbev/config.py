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
    image_backbone: Literal["tiny", "mobilenet_v3_large", "efficientnet_b0"] = "tiny"
    pretrained_image_backbone: bool = False
    freeze_image_backbone: bool = False
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
    teacher_seed_mode: Literal["off", "replace_lidar"] = "off"
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
            update={"teacher_seed_mode": "replace_lidar"}
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
