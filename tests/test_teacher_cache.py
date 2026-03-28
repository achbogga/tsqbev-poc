from __future__ import annotations

import torch

from tsqbev.contracts import TeacherTargets
from tsqbev.teacher_backends import (
    CachedTeacherProvider,
    TeacherProviderConfig,
    build_teacher_provider,
    teacher_key_from_metadata,
)
from tsqbev.teacher_cache import TeacherCacheStore


def _make_teacher_targets() -> TeacherTargets:
    return TeacherTargets(
        object_features=torch.randn(1, 4, 8),
        object_boxes=torch.randn(1, 4, 9),
        object_labels=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        object_scores=torch.tensor([[0.9, 0.7, 0.1, 0.5]], dtype=torch.float32),
        lane_features=torch.randn(1, 2, 8),
        router_logits=torch.randn(1, 4),
        valid_mask=torch.tensor([[True, True, False, True]]),
    )


def test_teacher_cache_round_trip(tmp_path) -> None:
    store = TeacherCacheStore(tmp_path)
    targets = _make_teacher_targets()
    store.save("sample-1", backend="cache", targets=targets, metadata={"split": "mini_val"})

    record = store.load("sample-1")
    assert record is not None
    assert record.key == "sample-1"
    assert record.backend == "cache"
    assert record.metadata == {"split": "mini_val"}
    assert torch.equal(record.targets.valid_mask, targets.valid_mask)
    assert torch.allclose(record.targets.object_features, targets.object_features)
    assert torch.allclose(record.targets.object_boxes, targets.object_boxes)
    assert torch.equal(record.targets.object_labels, targets.object_labels)
    assert torch.allclose(record.targets.object_scores, targets.object_scores)


def test_cached_teacher_provider_loads_targets(tmp_path) -> None:
    store = TeacherCacheStore(tmp_path)
    targets = _make_teacher_targets()
    store.save("sample-1", backend="cache", targets=targets)

    provider = CachedTeacherProvider(str(tmp_path))
    loaded = provider.load_targets({"sample_token": "sample-1"})
    assert loaded is not None
    assert torch.allclose(loaded.object_boxes, targets.object_boxes)


def test_teacher_provider_factory_builds_cache_provider(tmp_path) -> None:
    provider = build_teacher_provider(
        TeacherProviderConfig(kind="cache", cache_dir=str(tmp_path))
    )
    assert isinstance(provider, CachedTeacherProvider)


def test_teacher_key_prefers_sample_token() -> None:
    assert teacher_key_from_metadata({"sample_token": "abc", "file_path": "x/y"}) == "abc"
