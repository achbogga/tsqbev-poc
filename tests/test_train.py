from __future__ import annotations

from pathlib import Path

from tsqbev.datasets import SceneExample
from tsqbev.synthetic import make_synthetic_batch
from tsqbev.teacher_backends import TeacherProviderConfig
from tsqbev.teacher_cache import TeacherCacheStore
from tsqbev.train import fit_nuscenes, maybe_attach_teacher_targets, resolve_nuscenes_splits


class _SingleExampleDataset:
    def __init__(self, example: SceneExample) -> None:
        self.example = example

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> SceneExample:
        assert index == 0
        return self.example


class _RepeatedDataset:
    def __init__(self, example: SceneExample, length: int) -> None:
        self.example = example
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> SceneExample:
        assert 0 <= index < self.length
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


def test_fit_nuscenes_respects_max_train_steps(
    monkeypatch,
    small_config,
    tmp_path: Path,
) -> None:
    batch = make_synthetic_batch(small_config, batch_size=1, with_teacher=False)
    example = SceneExample(scene=batch, metadata={"sample_token": "sample-1"})

    def fake_nuscenes_dataset(**kwargs: object) -> _RepeatedDataset:
        del kwargs
        return _RepeatedDataset(example, length=4)

    monkeypatch.setattr("tsqbev.train.NuScenesDataset", fake_nuscenes_dataset)

    result = fit_nuscenes(
        dataroot=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        config=small_config,
        version="v1.0-mini",
        train_split="mini_train",
        val_split="mini_val",
        epochs=4,
        max_train_steps=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        log_every_steps=None,
    )

    assert result["train_steps"] == 1
    assert result["epochs"] == 1


def test_fit_nuscenes_passes_explicit_sample_tokens(
    monkeypatch,
    small_config,
    tmp_path: Path,
) -> None:
    batch = make_synthetic_batch(small_config, batch_size=1, with_teacher=False)
    example = SceneExample(scene=batch, metadata={"sample_token": "sample-1"})
    seen_tokens: list[list[str] | None] = []

    def fake_nuscenes_dataset(**kwargs: object) -> _RepeatedDataset:
        sample_tokens = kwargs.get("sample_tokens")
        assert sample_tokens is None or isinstance(sample_tokens, list)
        seen_tokens.append(sample_tokens)
        return _RepeatedDataset(example, length=1)

    monkeypatch.setattr("tsqbev.train.NuScenesDataset", fake_nuscenes_dataset)

    result = fit_nuscenes(
        dataroot=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        config=small_config,
        version="v1.0-mini",
        train_split="mini_train",
        val_split="mini_train",
        train_sample_tokens=["tok-a", "tok-b"],
        val_sample_tokens=["tok-a", "tok-b"],
        epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        log_every_steps=None,
    )

    assert result["epochs"] == 1
    assert seen_tokens == [["tok-a", "tok-b"], ["tok-a", "tok-b"]]


def test_fit_nuscenes_warm_starts_from_init_checkpoint(
    monkeypatch,
    small_config,
    tmp_path: Path,
) -> None:
    batch = make_synthetic_batch(small_config, batch_size=1, with_teacher=False)
    example = SceneExample(scene=batch, metadata={"sample_token": "sample-1"})
    loaded: list[str] = []

    def fake_nuscenes_dataset(**kwargs: object) -> _RepeatedDataset:
        del kwargs
        return _RepeatedDataset(example, length=1)

    monkeypatch.setattr("tsqbev.train.NuScenesDataset", fake_nuscenes_dataset)
    monkeypatch.setattr(
        "tsqbev.train.load_weights_into_model_from_checkpoint",
        lambda model, checkpoint_path, map_location="cpu": loaded.append(str(checkpoint_path)),
    )
    init_checkpoint = tmp_path / "init.pt"

    result = fit_nuscenes(
        dataroot=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        config=small_config,
        version="v1.0-mini",
        train_split="mini_train",
        val_split="mini_val",
        epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        log_every_steps=None,
        init_checkpoint=init_checkpoint,
    )

    assert result["epochs"] == 1
    assert loaded == [str(init_checkpoint)]


def test_fit_nuscenes_early_stops_on_val_plateau(
    monkeypatch,
    small_config,
    tmp_path: Path,
) -> None:
    batch = make_synthetic_batch(small_config, batch_size=1, with_teacher=False)
    example = SceneExample(scene=batch, metadata={"sample_token": "sample-1"})

    def fake_nuscenes_dataset(**kwargs: object) -> _RepeatedDataset:
        del kwargs
        return _RepeatedDataset(example, length=2)

    train_calls: list[int] = []
    val_totals = iter([10.0, 8.0, 8.03, 8.04, 8.05])

    def fake_train_epoch(**kwargs: object) -> dict[str, float]:
        del kwargs
        train_calls.append(1)
        return {"total": 12.0}

    def fake_eval_epoch(**kwargs: object) -> dict[str, float]:
        del kwargs
        return {"total": next(val_totals)}

    monkeypatch.setattr("tsqbev.train.NuScenesDataset", fake_nuscenes_dataset)
    monkeypatch.setattr("tsqbev.train._train_epoch", fake_train_epoch)
    monkeypatch.setattr("tsqbev.train._eval_epoch", fake_eval_epoch)

    result = fit_nuscenes(
        dataroot=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        config=small_config,
        version="v1.0-mini",
        train_split="mini_train",
        val_split="mini_val",
        epochs=5,
        batch_size=1,
        num_workers=0,
        device="cpu",
        log_every_steps=None,
        keep_best_checkpoint=True,
        early_stop_patience=2,
        early_stop_min_delta=0.05,
        early_stop_min_epochs=2,
    )

    assert result["epochs"] == 4
    assert result["selected_epoch"] == 2
    assert result["best_epoch"] == 2
    assert result["early_stop_triggered"] is True
    assert "plateau" in str(result["early_stop_reason"])
    assert len(train_calls) == 4
