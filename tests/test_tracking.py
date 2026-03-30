from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from tsqbev.config import ModelConfig
from tsqbev.tracking import (
    DEFAULT_WANDB_ENTITY,
    TrackingMetadata,
    project_name_for_experiment,
    start_experiment_tracking,
    tracking_enabled_from_env,
)


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, Any], int | None]] = []
        self.summary: dict[str, Any] = {}
        self.exit_code: int | None = None

    def log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        self.logged.append((payload, step))

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


def test_project_name_groups_hyperparameter_tuning_within_architecture_family() -> None:
    baseline = ModelConfig.rtx5000_nuscenes_baseline()
    query_boost = ModelConfig.rtx5000_nuscenes_query_boost()
    efficientnet = baseline.model_copy(update={"image_backbone": "efficientnet_b0"})
    teacher_seeded = baseline.model_copy(update={"teacher_seed_mode": "replace_lidar"})

    assert (
        project_name_for_experiment("nuScenes", baseline)
        == project_name_for_experiment("nuScenes", query_boost)
    )
    assert (
        project_name_for_experiment("nuScenes", baseline)
        != project_name_for_experiment("nuScenes", efficientnet)
    )
    assert (
        project_name_for_experiment("nuScenes", baseline)
        != project_name_for_experiment("nuScenes", teacher_seeded)
    )


def test_tracking_enabled_from_env_defaults_off_under_pytest(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TSQBEV_WANDB", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "secret")

    assert tracking_enabled_from_env() is True

    monkeypatch.setenv("TSQBEV_WANDB", "1")
    assert tracking_enabled_from_env() is True


def test_start_experiment_tracking_initializes_wandb_when_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    fake_run = _FakeRun()
    init_calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> _FakeRun:
        init_calls.append(kwargs)
        return fake_run

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TSQBEV_WANDB", "1")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))

    config = ModelConfig.rtx5000_nuscenes_baseline()
    tracker = start_experiment_tracking(
        artifact_dir=tmp_path / "run",
        config=config,
        metadata=TrackingMetadata(
            suite="research",
            dataset="nuscenes",
            job_type="research-loop",
            run_name="candidate-a",
            group="research-v1",
            tags=("research-loop", "mobilenet_v3_large"),
            extra_config={"recipe": "candidate-a"},
        ),
        config_payload={"model": config.model_dump(), "train": {"epochs": 4}},
    )

    assert tracker.enabled is True
    assert tracker.entity == DEFAULT_WANDB_ENTITY
    assert tracker.project == project_name_for_experiment("nuscenes", config)
    assert init_calls[0]["entity"] == DEFAULT_WANDB_ENTITY
    assert init_calls[0]["project"] == project_name_for_experiment("nuscenes", config)
    assert init_calls[0]["group"] == "research-v1"

    tracker.log({"metric": 1.5}, step=3)
    tracker.summary({"checkpoint": tmp_path / "run" / "checkpoint.pt"})
    tracker.finish(status="completed")

    assert fake_run.logged == [({"metric": 1.5}, 3)]
    assert fake_run.summary["checkpoint"].endswith("checkpoint.pt")
    assert fake_run.summary["run_status"] == "completed"
    assert fake_run.exit_code == 0
