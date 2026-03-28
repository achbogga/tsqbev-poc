from __future__ import annotations

from tsqbev.config import ModelConfig


def test_small_config_preserves_default_detection_class_count() -> None:
    assert ModelConfig.small().num_object_classes == ModelConfig().num_object_classes
