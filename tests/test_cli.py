from __future__ import annotations

import argparse
from pathlib import Path

from tsqbev.cli import _resolve_teacher_provider_config
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
