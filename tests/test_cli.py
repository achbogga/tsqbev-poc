from __future__ import annotations

import argparse
from pathlib import Path

from tsqbev.cli import _resolve_config, _resolve_teacher_provider_config
from tsqbev.research import _canonical_command
from tsqbev.teacher_backends import TeacherProviderConfig


def test_resolve_teacher_provider_config_infers_cache_kind_from_cache_dir() -> None:
    args = argparse.Namespace(
        teacher_kind=None,
        teacher_cache_dir=Path("/tmp/teacher_cache"),
        teacher_checkpoint=None,
    )

    config = _resolve_teacher_provider_config(args)

    assert config is not None
    assert config.kind == "cache"
    assert config.cache_dir == "/tmp/teacher_cache"


def test_canonical_command_includes_teacher_provider_flags() -> None:
    command = _canonical_command(
        dataroot=Path("/tmp/nuscenes"),
        artifact_dir=Path("/tmp/artifacts/research_v1"),
        device="cuda",
        max_experiments=5,
        teacher_provider_config=TeacherProviderConfig(
            kind="cache",
            cache_dir="/tmp/teacher_cache",
        ),
    )

    assert "--teacher-kind cache" in command
    assert "--teacher-cache-dir /tmp/teacher_cache" in command


def test_resolve_config_supports_dinov2_teacher_preset() -> None:
    args = argparse.Namespace(
        preset="rtx5000-nuscenes-dinov2-teacher",
        image_backbone=None,
        pretrained_image_backbone=None,
        freeze_image_backbone=None,
        foundation_repo_root=None,
        activation_checkpointing=None,
        attention_backend=None,
        teacher_seed_mode=None,
        teacher_seed_selection_mode=None,
    )

    config = _resolve_config(args)

    assert config.image_backbone == "dinov2_vits14_reg"
    assert config.freeze_image_backbone is True
    assert config.ranking_mode == "quality_class_only"


def test_resolve_config_supports_teacher_quality_plus_preset() -> None:
    args = argparse.Namespace(
        preset="rtx5000-nuscenes-teacher-quality-plus",
        image_backbone=None,
        pretrained_image_backbone=None,
        freeze_image_backbone=None,
        foundation_repo_root=None,
        activation_checkpointing=None,
        attention_backend=None,
        teacher_seed_mode=None,
        teacher_seed_selection_mode=None,
    )

    config = _resolve_config(args)

    assert config.image_backbone == "mobilenet_v3_large"
    assert config.freeze_image_backbone is False
    assert config.teacher_seed_mode == "replace_lidar"
