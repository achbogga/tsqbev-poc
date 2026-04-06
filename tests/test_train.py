from __future__ import annotations

import json
from pathlib import Path

import torch

from tsqbev.datasets import SceneExample
from tsqbev.synthetic import make_synthetic_batch
from tsqbev.teacher_backends import TeacherProviderConfig
from tsqbev.teacher_cache import TeacherCacheStore
from tsqbev.train import (
    _catastrophic_nuscenes_official_failure,
    _joint_official_metrics_better,
    _make_detection_criterion,
    _nuscenes_official_metrics_better,
    _run_joint_public_official_eval,
    _teacher_anchor_schedule_value,
    fit_nuscenes,
    fit_openlane,
    maybe_attach_teacher_targets,
    resolve_nuscenes_splits,
    set_global_seed,
)


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


def test_make_detection_criterion_exposes_teacher_anchor_weights() -> None:
    criterion = _make_detection_criterion(
        loss_mode="baseline",
        hard_negative_ratio=3,
        hard_negative_cap=96,
        teacher_anchor_class_weight=2.0,
        teacher_anchor_quality_class_weight=0.0,
        teacher_anchor_objectness_weight=1.5,
        teacher_region_objectness_weight=0.25,
        teacher_region_radius_m=6.0,
    )
    assert criterion.teacher_anchor_class_weight == 2.0
    assert criterion.teacher_anchor_objectness_weight == 1.5
    assert criterion.teacher_region_objectness_weight == 0.25
    assert criterion.teacher_region_radius_m == 6.0


def test_teacher_anchor_schedule_value_stays_constant_without_decay() -> None:
    assert (
        _teacher_anchor_schedule_value(
            epoch=8,
            initial_weight=0.5,
            final_weight=0.5,
            bootstrap_epochs=4,
            decay_epochs=8,
        )
        == 0.5
    )


def test_teacher_anchor_schedule_value_decays_after_bootstrap() -> None:
    assert _teacher_anchor_schedule_value(
        epoch=4,
        initial_weight=0.5,
        final_weight=0.1,
        bootstrap_epochs=4,
        decay_epochs=8,
    ) == 0.5
    assert round(
        _teacher_anchor_schedule_value(
            epoch=8,
            initial_weight=0.5,
            final_weight=0.1,
            bootstrap_epochs=4,
            decay_epochs=8,
        ),
        6,
    ) == 0.3
    assert _teacher_anchor_schedule_value(
        epoch=12,
        initial_weight=0.5,
        final_weight=0.1,
        bootstrap_epochs=4,
        decay_epochs=8,
    ) == 0.1


def test_set_global_seed_makes_torch_sampling_repeatable() -> None:
    set_global_seed(1337)
    first = torch.randn(4)
    set_global_seed(1337)
    second = torch.randn(4)
    assert torch.equal(first, second)


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


def test_nuscenes_official_metric_prefers_sane_higher_nds() -> None:
    weak = {
        "nuscenes_eval_ok": 1.0,
        "nuscenes_export_sanity_ok": 1.0,
        "nuscenes_nds": 0.12,
        "nuscenes_map": 0.11,
        "nuscenes_boxes_per_sample_mean": 60.0,
        "nuscenes_ego_translation_norm_p99": 55.0,
        "nuscenes_max_box_size_m": 18.0,
    }
    strong = {
        "nuscenes_eval_ok": 1.0,
        "nuscenes_export_sanity_ok": 1.0,
        "nuscenes_nds": 0.16,
        "nuscenes_map": 0.14,
        "nuscenes_boxes_per_sample_mean": 40.0,
        "nuscenes_ego_translation_norm_p99": 45.0,
        "nuscenes_max_box_size_m": 12.0,
    }

    assert _nuscenes_official_metrics_better(strong, weak) is True


def test_catastrophic_nuscenes_official_failure_detects_zero_metric_export_collapse() -> None:
    assert (
        _catastrophic_nuscenes_official_failure(
            {
                "nuscenes_eval_ok": 1.0,
                "nuscenes_export_sanity_ok": 0.0,
                "nuscenes_nds": 0.0,
                "nuscenes_map": 0.0,
                "nuscenes_max_box_size_m": 1106.7,
                "nuscenes_score_mean": 0.9998,
                "nuscenes_ego_translation_norm_p99": 25.18,
            }
        )
        is True
    )
    assert (
        _catastrophic_nuscenes_official_failure(
            {
                "nuscenes_eval_ok": 1.0,
                "nuscenes_export_sanity_ok": 1.0,
                "nuscenes_nds": 0.18,
                "nuscenes_map": 0.16,
                "nuscenes_max_box_size_m": 8.0,
                "nuscenes_score_mean": 0.42,
                "nuscenes_ego_translation_norm_p99": 6.0,
            }
        )
        is False
    )


def test_fit_nuscenes_selects_best_official_checkpoint(
    monkeypatch,
    small_config,
    tmp_path: Path,
) -> None:
    batch = make_synthetic_batch(small_config, batch_size=1, with_teacher=False)
    example = SceneExample(scene=batch, metadata={"sample_token": "sample-1"})

    def fake_nuscenes_dataset(**kwargs: object) -> _RepeatedDataset:
        del kwargs
        return _RepeatedDataset(example, length=1)

    def fake_train_epoch(**kwargs: object) -> dict[str, float]:
        del kwargs
        return {"total": 12.0}

    val_totals = iter([9.0, 8.5, 8.6, 8.7])

    def fake_eval_epoch(**kwargs: object) -> dict[str, float]:
        del kwargs
        return {"total": next(val_totals)}

    official_scores = {
        2: {
            "nuscenes_eval_ok": 1.0,
            "nuscenes_export_sanity_ok": 1.0,
            "nuscenes_nds": 0.13,
            "nuscenes_map": 0.12,
            "nuscenes_boxes_per_sample_mean": 42.0,
            "nuscenes_ego_translation_norm_p99": 48.0,
            "nuscenes_max_box_size_m": 15.0,
            "nuscenes_score_mean": 0.42,
        },
        4: {
            "nuscenes_eval_ok": 1.0,
            "nuscenes_export_sanity_ok": 1.0,
            "nuscenes_nds": 0.18,
            "nuscenes_map": 0.16,
            "nuscenes_boxes_per_sample_mean": 38.0,
            "nuscenes_ego_translation_norm_p99": 44.0,
            "nuscenes_max_box_size_m": 12.0,
            "nuscenes_score_mean": 0.39,
        },
    }

    def fake_run_official_eval(**kwargs: object) -> dict[str, float]:
        epoch = kwargs.get("epoch")
        assert isinstance(epoch, int)
        return official_scores[epoch]

    monkeypatch.setattr("tsqbev.train.NuScenesDataset", fake_nuscenes_dataset)
    monkeypatch.setattr("tsqbev.train._train_epoch", fake_train_epoch)
    monkeypatch.setattr("tsqbev.train._eval_epoch", fake_eval_epoch)
    monkeypatch.setattr("tsqbev.train._run_nuscenes_official_eval", fake_run_official_eval)

    result = fit_nuscenes(
        dataroot=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        config=small_config,
        version="v1.0-mini",
        train_split="mini_train",
        val_split="mini_val",
        epochs=4,
        batch_size=1,
        num_workers=0,
        device="cpu",
        log_every_steps=None,
        keep_best_checkpoint=True,
        official_eval_every_epochs=2,
    )

    best_official = Path(str(result["best_official_checkpoint_path"]))
    assert best_official.exists()
    assert Path(str(result["checkpoint_path"])) == best_official
    assert result["best_official_epoch"] == 4
    assert result["best_official_metrics"] == official_scores[4]


def test_fit_nuscenes_stops_on_catastrophic_official_eval(
    monkeypatch,
    small_config,
    tmp_path: Path,
) -> None:
    batch = make_synthetic_batch(small_config, batch_size=1, with_teacher=False)
    example = SceneExample(scene=batch, metadata={"sample_token": "sample-1"})

    def fake_nuscenes_dataset(**kwargs: object) -> _RepeatedDataset:
        del kwargs
        return _RepeatedDataset(example, length=1)

    def fake_train_epoch(**kwargs: object) -> dict[str, float]:
        del kwargs
        return {"total": 12.0}

    def fake_eval_epoch(**kwargs: object) -> dict[str, float]:
        del kwargs
        return {"total": 8.5}

    def fake_run_official_eval(**kwargs: object) -> dict[str, float]:
        del kwargs
        return {
            "nuscenes_eval_ok": 1.0,
            "nuscenes_export_sanity_ok": 0.0,
            "nuscenes_nds": 0.0,
            "nuscenes_map": 0.0,
            "nuscenes_boxes_per_sample_mean": 40.0,
            "nuscenes_ego_translation_norm_p99": 25.2,
            "nuscenes_max_box_size_m": 1106.7,
            "nuscenes_score_mean": 0.9998,
        }

    monkeypatch.setattr("tsqbev.train.NuScenesDataset", fake_nuscenes_dataset)
    monkeypatch.setattr("tsqbev.train._train_epoch", fake_train_epoch)
    monkeypatch.setattr("tsqbev.train._eval_epoch", fake_eval_epoch)
    monkeypatch.setattr("tsqbev.train._run_nuscenes_official_eval", fake_run_official_eval)

    result = fit_nuscenes(
        dataroot=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        config=small_config,
        version="v1.0-mini",
        train_split="mini_train",
        val_split="mini_val",
        epochs=8,
        batch_size=1,
        num_workers=0,
        device="cpu",
        log_every_steps=None,
        keep_best_checkpoint=True,
        official_eval_every_epochs=1,
    )

    assert result["epochs"] == 1
    assert result["early_stop_triggered"] is True
    assert "catastrophic official-eval failure" in str(result["early_stop_reason"])


def test_fit_openlane_supports_warm_start_and_max_train_steps(
    monkeypatch,
    small_config,
    tmp_path: Path,
) -> None:
    batch = make_synthetic_batch(small_config, batch_size=1, with_teacher=False)
    example = SceneExample(scene=batch, metadata={"file_path": "segment-001/frame-001.jpg"})
    loaded: list[str] = []

    def fake_openlane_dataset(**kwargs: object) -> _RepeatedDataset:
        del kwargs
        return _RepeatedDataset(example, length=4)

    monkeypatch.setattr("tsqbev.train.OpenLaneDataset", fake_openlane_dataset)
    monkeypatch.setattr(
        "tsqbev.train.load_weights_into_model_from_checkpoint",
        lambda model, checkpoint_path, map_location="cpu": loaded.append(str(checkpoint_path)),
    )

    init_checkpoint = tmp_path / "lane-init.pt"
    result = fit_openlane(
        dataroot=tmp_path,
        artifact_dir=tmp_path / "openlane-artifacts",
        config=small_config,
        epochs=3,
        max_train_steps=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        log_every_steps=None,
        init_checkpoint=init_checkpoint,
        augmentation_mode="moderate",
    )

    assert loaded == [str(init_checkpoint)]
    assert result["train_steps"] == 1
    assert result["augmentation_mode"] == "moderate"


def test_joint_public_official_eval_degrades_gracefully_on_openlane_failure(
    monkeypatch,
    small_config,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("tsqbev.train.TSQBEVModel", lambda config: object())
    monkeypatch.setattr(
        "tsqbev.train.load_weights_into_model_from_checkpoint",
        lambda model, checkpoint_path, map_location="cpu": None,
    )

    nuscenes_pred_path = tmp_path / "nuscenes.json"
    nuscenes_pred_path.write_text("{}")
    lane_pred_dir = tmp_path / "lane_preds"
    lane_pred_dir.mkdir()
    lane_test_list = tmp_path / "validation_test_list.txt"
    lane_test_list.write_text("sample\n")

    monkeypatch.setattr(
        "tsqbev.train.export_nuscenes_predictions",
        lambda **kwargs: nuscenes_pred_path,
    )
    monkeypatch.setattr(
        "tsqbev.train.evaluate_nuscenes_predictions",
        lambda **kwargs: {"nd_score": 0.25, "mean_ap": 0.2},
    )
    monkeypatch.setattr(
        "tsqbev.train.export_sanity_diagnostics",
        lambda *args, **kwargs: {
            "sanity_ok": 1.0,
            "boxes_per_sample_mean": 12.0,
            "ego_translation_norm_p99": 24.0,
            "max_box_size_m": 4.5,
            "score_mean": 0.75,
        },
    )
    monkeypatch.setattr(
        "tsqbev.train.export_openlane_predictions",
        lambda **kwargs: lane_pred_dir,
    )
    monkeypatch.setattr(
        "tsqbev.train.write_openlane_test_list",
        lambda **kwargs: lane_test_list,
    )

    def _raise_openlane_failure(**kwargs: object) -> dict[str, float]:
        raise RuntimeError("openlane evaluator missing dependency")

    monkeypatch.setattr(
        "tsqbev.train.evaluate_openlane_predictions",
        _raise_openlane_failure,
    )

    metrics = _run_joint_public_official_eval(
        checkpoint_path=tmp_path / "checkpoint.pt",
        model_config=small_config,
        device="cpu",
        artifact_root=tmp_path / "artifacts",
        epoch=5,
        nuscenes_root=tmp_path / "nuscenes",
        nuscenes_version="v1.0-mini",
        nuscenes_split="mini_val",
        openlane_root=tmp_path / "openlane",
        openlane_subset="lane3d_300",
        openlane_repo_root=tmp_path / "OpenLane",
        score_threshold=0.2,
        top_k=40,
    )

    assert metrics["nuscenes_eval_ok"] == 1.0
    assert metrics["nuscenes_nds"] == 0.25
    assert metrics["nuscenes_export_sanity_ok"] == 1.0
    assert metrics["openlane_eval_ok"] == 0.0
    assert metrics["openlane_f_score"] == 0.0

    summary = json.loads(
        (tmp_path / "artifacts" / "official_eval" / "epoch_005" / "summary.json").read_text()
    )
    assert summary["metrics"]["nuscenes_map"] == 0.2
    assert "openlane" in summary["errors"]


def test_joint_official_metrics_prefer_sane_official_detection() -> None:
    broken = {
        "nuscenes_eval_ok": 1.0,
        "nuscenes_export_sanity_ok": 0.0,
        "nuscenes_nds": 0.40,
        "nuscenes_map": 0.35,
        "openlane_eval_ok": 1.0,
        "openlane_f_score": 0.70,
        "openlane_precision": 0.72,
        "openlane_recall": 0.68,
    }
    sane = {
        "nuscenes_eval_ok": 1.0,
        "nuscenes_export_sanity_ok": 1.0,
        "nuscenes_nds": 0.20,
        "nuscenes_map": 0.18,
        "openlane_eval_ok": 1.0,
        "openlane_f_score": 0.65,
        "openlane_precision": 0.66,
        "openlane_recall": 0.64,
    }
    assert _joint_official_metrics_better(sane, broken) is True
