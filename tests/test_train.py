from __future__ import annotations

from tsqbev.datasets import SceneExample
from tsqbev.synthetic import make_synthetic_batch
from tsqbev.teacher_backends import TeacherProviderConfig
from tsqbev.teacher_cache import TeacherCacheStore
from tsqbev.train import maybe_attach_teacher_targets, resolve_nuscenes_splits


class _SingleExampleDataset:
    def __init__(self, example: SceneExample) -> None:
        self.example = example

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> SceneExample:
        assert index == 0
        return self.example


def test_resolve_nuscenes_splits_defaults_to_mini() -> None:
    assert resolve_nuscenes_splits("v1.0-mini", None, None) == ("mini_train", "mini_val")


def test_resolve_nuscenes_splits_defaults_to_trainval() -> None:
    assert resolve_nuscenes_splits("v1.0-trainval", None, None) == ("train", "val")


def test_resolve_nuscenes_splits_preserves_explicit_values() -> None:
    assert resolve_nuscenes_splits("v1.0-mini", "custom_train", "custom_val") == (
        "custom_train",
        "custom_val",
    )


def test_maybe_attach_teacher_targets_wraps_dataset_when_cache_is_configured(
    small_config,
    tmp_path,
) -> None:
    batch = make_synthetic_batch(small_config, batch_size=1, with_teacher=True)
    example = SceneExample(scene=batch, metadata={"sample_token": "sample-1"})
    dataset = _SingleExampleDataset(example)
    assert batch.teacher_targets is not None
    TeacherCacheStore(tmp_path).save(
        "sample-1",
        backend="cache",
        targets=batch.teacher_targets,
    )

    wrapped = maybe_attach_teacher_targets(
        dataset,
        TeacherProviderConfig(kind="cache", cache_dir=str(tmp_path)),
    )
    loaded = wrapped[0]
    assert loaded.scene.teacher_targets is not None
