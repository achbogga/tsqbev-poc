"""Lazy experiment tracking for tsqbev experiments.

References:
- W&B run overview:
  https://docs.wandb.ai/guides/runs
- W&B Run API:
  https://docs.wandb.ai/ref/python/experiments/run/
- W&B logging guide:
  https://docs.wandb.ai/guides/track/log
- W&B summary metrics:
  https://docs.wandb.ai/guides/track/log/log-summary
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tsqbev.config import ModelConfig

DEFAULT_WANDB_ENTITY = "achbogga-track"
DEFAULT_PROJECT_PREFIX = "tsqbev"


def _slug(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        model_dump = value.model_dump
        if callable(model_dump):
            return _jsonable(model_dump())
    return str(value)


def _architecture_family(config: ModelConfig) -> str:
    parts = [f"v{config.views}", _slug(config.image_backbone)]
    if config.teacher_seed_mode != "off":
        parts.append("teacher-seed")
    return "-".join(parts)


def project_name_for_experiment(dataset: str, config: ModelConfig) -> str:
    """Keep tuning runs together within one architecture family."""

    return f"{DEFAULT_PROJECT_PREFIX}-{_slug(dataset)}-{_architecture_family(config)}"


def tracking_enabled_from_env() -> bool:
    toggle = os.getenv("TSQBEV_WANDB")
    if toggle is not None and toggle.strip().lower() in {"0", "false", "off", "no"}:
        return False
    if toggle is not None and toggle.strip().lower() in {"1", "true", "on", "yes"}:
        return True
    mode = os.getenv("WANDB_MODE", "").strip().lower()
    if mode == "disabled":
        return False
    return bool(os.getenv("WANDB_API_KEY"))


@dataclass(slots=True)
class TrackingMetadata:
    suite: str
    dataset: str
    job_type: str
    run_name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """Small wrapper that degrades to a no-op when W&B is unavailable."""

    def __init__(
        self,
        *,
        run: Any | None,
        enabled: bool,
        entity: str,
        project: str,
        artifact_dir: Path,
    ) -> None:
        self._run = run
        self.enabled = enabled
        self.entity = entity
        self.project = project
        self.artifact_dir = artifact_dir

    def log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._run.log(_jsonable(payload), step=step)
        except Exception as exc:
            print(f"[wandb] log failed, disabling tracking: {exc!r}", flush=True)
            self.enabled = False
            self._run = None

    def summary(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            for key, value in _jsonable(payload).items():
                self._run.summary[key] = value
        except Exception as exc:
            print(f"[wandb] summary update failed, disabling tracking: {exc!r}", flush=True)
            self.enabled = False
            self._run = None

    def finish(self, *, status: str) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._run.summary["run_status"] = status
            self._run.finish(exit_code=0 if status == "completed" else 1)
        except Exception as exc:
            print(f"[wandb] finish failed: {exc!r}", flush=True)
        finally:
            self._run = None


def start_experiment_tracking(
    *,
    artifact_dir: str | Path,
    config: ModelConfig,
    metadata: TrackingMetadata,
    config_payload: dict[str, Any],
) -> ExperimentTracker:
    """Start W&B tracking when credentials are present; otherwise return a no-op tracker."""

    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    entity = os.getenv("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)
    project = project_name_for_experiment(metadata.dataset, config)
    if not tracking_enabled_from_env():
        return ExperimentTracker(
            run=None,
            enabled=False,
            entity=entity,
            project=project,
            artifact_dir=artifact_root,
        )
    try:
        import wandb
    except ImportError:
        print("[wandb] package not installed; continuing without tracking", flush=True)
        return ExperimentTracker(
            run=None,
            enabled=False,
            entity=entity,
            project=project,
            artifact_dir=artifact_root,
        )

    init_config = {
        "suite": metadata.suite,
        "dataset": metadata.dataset,
        "job_type": metadata.job_type,
        "artifact_dir": str(artifact_root),
        "hostname": socket.gethostname(),
        **_jsonable(config_payload),
        **_jsonable(metadata.extra_config),
    }
    try:
        run = wandb.init(
            entity=entity,
            project=project,
            job_type=metadata.job_type,
            name=metadata.run_name,
            group=metadata.group,
            tags=list(metadata.tags),
            notes=metadata.notes,
            dir=str(artifact_root),
            config=init_config,
        )
        return ExperimentTracker(
            run=run,
            enabled=run is not None,
            entity=entity,
            project=project,
            artifact_dir=artifact_root,
        )
    except Exception as exc:
        print(f"[wandb] init failed; continuing without tracking: {exc!r}", flush=True)
        return ExperimentTracker(
            run=None,
            enabled=False,
            entity=entity,
            project=project,
            artifact_dir=artifact_root,
        )
