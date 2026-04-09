from __future__ import annotations

from tsqbev.config import ModelConfig


def test_small_config_preserves_default_detection_class_count() -> None:
    assert ModelConfig.small().num_object_classes == ModelConfig().num_object_classes


def test_rtx5000_baseline_uses_pretrained_mobilenet() -> None:
    config = ModelConfig.rtx5000_nuscenes_baseline()
    assert config.image_backbone == "mobilenet_v3_large"
    assert config.pretrained_image_backbone is True
    assert config.freeze_image_backbone is True


def test_rtx5000_teacher_bootstrap_enables_teacher_seed_mode() -> None:
    config = ModelConfig.rtx5000_nuscenes_teacher_bootstrap()
    assert config.image_backbone == "mobilenet_v3_large"
    assert config.teacher_seed_mode == "replace_lidar"
    assert config.teacher_seed_selection_mode == "class_balanced_round_robin"
    assert config.router_mode == "anchor_first"


def test_rtx5000_query_boost_matches_current_best_mini_direction() -> None:
    config = ModelConfig.rtx5000_nuscenes_query_boost()
    assert config.image_backbone == "mobilenet_v3_large"
    assert config.freeze_image_backbone is True
    assert config.q_lidar == 64
    assert config.q_2d == 112
    assert config.max_object_queries == 112


def test_rtx5000_dinov2_teacher_preset_uses_foundation_backbone() -> None:
    config = ModelConfig.rtx5000_nuscenes_dinov2_teacher()
    assert config.image_backbone == "dinov2_vits14_reg"
    assert config.pretrained_image_backbone is True
    assert config.freeze_image_backbone is True
    assert config.foundation_repo_root == "/home/achbogga/projects/dinov2"
    assert config.ranking_mode == "quality_class_only"


def test_rtx5000_teacher_quality_plus_matches_winner_line_shape() -> None:
    config = ModelConfig.rtx5000_nuscenes_teacher_quality_plus()
    assert config.image_backbone == "mobilenet_v3_large"
    assert config.freeze_image_backbone is False
    assert config.teacher_seed_mode == "replace_lidar"
    assert config.q_lidar == 96
    assert config.q_2d == 80
    assert config.max_object_queries == 112


def test_rtx5000_bridge_teacher_preset_stays_lightweight() -> None:
    config = ModelConfig.rtx5000_nuscenes_bridge_teacher()
    assert config.image_backbone == "mobilenet_v3_large"
    assert config.fusion_style == "gated_latent_cross_attn"
    assert config.teacher_seed_mode == "off"
    assert config.latent_bridge_slots == 8
