from __future__ import annotations

from tsqbev.config import ModelConfig


def test_small_config_preserves_default_detection_class_count() -> None:
    assert ModelConfig.small().num_object_classes == ModelConfig().num_object_classes


def test_rtx5000_baseline_uses_pretrained_mobilenet() -> None:
    config = ModelConfig.rtx5000_nuscenes_baseline()
    assert config.image_backbone == "mobilenet_v3_large"
    assert config.pretrained_image_backbone is True
    assert config.freeze_image_backbone is True
