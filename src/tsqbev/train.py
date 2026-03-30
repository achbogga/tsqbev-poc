"""Real-data training loops for public baseline runs.

References:
- PETRv2 multitask sparse-query optimization:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf
- BEVDistill student-training framing:
  https://arxiv.org/abs/2211.09386
"""

from __future__ import annotations

import json
import time
from collections.abc import Sized
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

from tsqbev.checkpoints import load_weights_into_model_from_checkpoint, save_model_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.datasets import NuScenesDataset, OpenLaneDataset, collate_scene_examples
from tsqbev.losses import DetectionSetCriterion, MultitaskCriterion
from tsqbev.model import TSQBEVModel
from tsqbev.runtime import move_batch, resolve_device
from tsqbev.teacher_backends import TeacherProviderConfig, build_teacher_provider
from tsqbev.teacher_dataset import TeacherAugmentedDataset
from tsqbev.tracking import ExperimentTracker, TrackingMetadata, start_experiment_tracking


def _format_metrics(metrics: dict[str, float]) -> str:
    visible = []
    for name, value in metrics.items():
        if name != "total" and abs(value) < 1e-8:
            continue
        visible.append(f"{name}={value:.4f}")
    return ", ".join(visible)


def _to_float_metrics(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def _average_history(history: list[dict[str, float]]) -> dict[str, float]:
    keys = history[0].keys()
    return {key: sum(item[key] for item in history) / float(len(history)) for key in keys}


def _subset_if_requested(dataset: Dataset[Any], max_samples: int | None) -> Dataset[Any]:
    sized_dataset = cast(Sized, dataset)
    if max_samples is None or max_samples >= len(sized_dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def maybe_attach_teacher_targets(
    dataset: Dataset[Any],
    teacher_provider_config: TeacherProviderConfig | None,
) -> Dataset[Any]:
    """Wrap a dataset with an optional external teacher cache/provider."""

    if teacher_provider_config is None:
        return dataset
    provider = build_teacher_provider(teacher_provider_config)
    return TeacherAugmentedDataset(dataset, provider)


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _default_tracking_group(artifact_dir: Path, dataset: str) -> str:
    return f"{dataset}-{artifact_dir.parent.name}"


def _make_scheduler(
    optimizer: AdamW,
    *,
    epochs: int,
    optimizer_schedule: Literal["cosine", "constant"],
) -> CosineAnnealingLR | LambdaLR:
    if optimizer_schedule == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    return CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def _make_detection_criterion(
    *,
    loss_mode: Literal["baseline", "focal_hardneg"],
    hard_negative_ratio: int,
    hard_negative_cap: int,
) -> DetectionSetCriterion:
    return DetectionSetCriterion(
        loss_mode=loss_mode,
        hard_negative_ratio=hard_negative_ratio,
        hard_negative_cap=hard_negative_cap,
    )


def resolve_nuscenes_splits(
    version: str,
    train_split: str | None,
    val_split: str | None,
) -> tuple[str, str]:
    """Resolve the appropriate nuScenes splits for the requested version."""

    if train_split is not None and val_split is not None:
        return train_split, val_split
    if version == "v1.0-mini":
        return train_split or "mini_train", val_split or "mini_val"
    return train_split or "train", val_split or "val"


def _train_epoch(
    model: TSQBEVModel,
    loader: DataLoader,
    criterion: MultitaskCriterion,
    optimizer: AdamW,
    grad_accum_steps: int,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
    epoch: int,
    log_every_steps: int | None,
    max_steps: int | None = None,
    grad_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, float]] = []
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32

    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    for step, (batch, _) in enumerate(loader, start=1):
        if max_steps is not None and step > max_steps:
            break
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            outputs = model(batch)
            losses = criterion(outputs, batch)
        scaled_loss = losses["total"] / float(grad_accum_steps)
        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if step % grad_accum_steps == 0:
            if amp_enabled:
                scaler.unscale_(optimizer)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step_metrics = _to_float_metrics(losses)
        history.append(step_metrics)
        should_log = (
            log_every_steps is not None
            and (step == 1 or step % log_every_steps == 0 or step == total_steps)
        )
        if should_log:
            print(
                f"[train] epoch={epoch} step={step}/{total_steps} {_format_metrics(step_metrics)}",
                flush=True,
            )

    if len(loader) % grad_accum_steps != 0:
        if amp_enabled:
            scaler.unscale_(optimizer)
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return _average_history(history)


@torch.no_grad()
def _eval_epoch(
    model: TSQBEVModel,
    loader: DataLoader,
    criterion: MultitaskCriterion,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    history: list[dict[str, float]] = []
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
    for batch, _ in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            outputs = model(batch)
            losses = criterion(outputs, batch)
        history.append(_to_float_metrics(losses))
    avg = _average_history(history)
    print(f"[val] epoch={epoch} {_format_metrics(avg)}", flush=True)
    return avg


def _write_history(artifact_dir: Path, history: list[dict[str, object]]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "history.json").write_text(json.dumps(history, indent=2))


def _make_loader(
    dataset: Dataset,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
) -> DataLoader:
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_scene_examples,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_scene_examples,
        pin_memory=torch.cuda.is_available(),
    )


def fit_nuscenes(
    dataroot: str | Path,
    artifact_dir: str | Path,
    config: ModelConfig | None = None,
    version: str = "v1.0-trainval",
    train_split: str | None = None,
    val_split: str | None = None,
    train_sample_tokens: list[str] | None = None,
    val_sample_tokens: list[str] | None = None,
    epochs: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 8,
    num_workers: int = 4,
    batch_size: int = 1,
    device: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_train_steps: int | None = None,
    teacher_provider_config: TeacherProviderConfig | None = None,
    init_checkpoint: str | Path | None = None,
    use_amp: bool = False,
    log_every_steps: int | None = 100,
    optimizer_schedule: Literal["cosine", "constant"] = "cosine",
    grad_clip_norm: float | None = 1.0,
    keep_best_checkpoint: bool = False,
    loss_mode: Literal["baseline", "focal_hardneg"] = "baseline",
    hard_negative_ratio: int = 3,
    hard_negative_cap: int = 96,
    tracker: ExperimentTracker | None = None,
    tracking_metadata: TrackingMetadata | None = None,
) -> dict[str, object]:
    """Fit the public object-detection baseline on nuScenes train/val."""

    artifact_root = Path(artifact_dir)
    resolved_device = resolve_device(device)
    owns_tracker = tracker is None
    active_tracker = tracker
    status = "failed"
    try:
        if resolved_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        amp_enabled = bool(use_amp) and resolved_device.type == "cuda"
        model_config = config if config is not None else ModelConfig()
        resolved_train_split, resolved_val_split = resolve_nuscenes_splits(
            version=version,
            train_split=train_split,
            val_split=val_split,
        )
        print(
            f"[setup] loading nuScenes train split={resolved_train_split} from {dataroot}",
            flush=True,
        )
        start_time = time.perf_counter()
        train_dataset = _subset_if_requested(
            NuScenesDataset(
                dataroot=dataroot,
                version=version,
                split=resolved_train_split,
                sample_tokens=train_sample_tokens,
            ),
            max_train_samples,
        )
        train_dataset = maybe_attach_teacher_targets(train_dataset, teacher_provider_config)
        print(
            f"[setup] loaded train split in {time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        print(
            f"[setup] loading nuScenes val split={resolved_val_split} from {dataroot}",
            flush=True,
        )
        start_time = time.perf_counter()
        val_dataset = _subset_if_requested(
            NuScenesDataset(
                dataroot=dataroot,
                version=version,
                split=resolved_val_split,
                sample_tokens=val_sample_tokens,
            ),
            max_val_samples,
        )
        val_dataset = maybe_attach_teacher_targets(val_dataset, teacher_provider_config)
        print(
            f"[setup] loaded val split in {time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        train_loader = _make_loader(
            train_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        val_loader = _make_loader(
            val_dataset,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
        )

        train_sample_count = len(cast(Sized, train_dataset))
        val_sample_count = len(cast(Sized, val_dataset))
        if active_tracker is None:
            metadata = tracking_metadata or TrackingMetadata(
                suite="train",
                dataset="nuscenes",
                job_type="train-nuscenes",
                run_name=artifact_root.name,
                group=_default_tracking_group(artifact_root, "nuscenes"),
                tags=(
                    version,
                    resolved_train_split,
                    resolved_val_split,
                    model_config.image_backbone,
                    model_config.teacher_seed_mode,
                ),
                extra_config={
                    "train_split": resolved_train_split,
                    "val_split": resolved_val_split,
                    "train_samples": train_sample_count,
                    "val_samples": val_sample_count,
                    "teacher_provider": (
                        teacher_provider_config.kind
                        if teacher_provider_config is not None
                        else None
                    ),
                },
            )
            active_tracker = start_experiment_tracking(
                artifact_dir=artifact_root,
                config=model_config,
                metadata=metadata,
                config_payload={
                    "model": model_config.model_dump(),
                    "train": {
                        "epochs": epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "grad_accum_steps": grad_accum_steps,
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "max_train_steps": max_train_steps,
                        "max_train_samples": max_train_samples,
                        "max_val_samples": max_val_samples,
                        "init_checkpoint": (
                            str(init_checkpoint) if init_checkpoint is not None else None
                        ),
                        "optimizer_schedule": optimizer_schedule,
                        "grad_clip_norm": grad_clip_norm,
                        "keep_best_checkpoint": keep_best_checkpoint,
                        "loss_mode": loss_mode,
                        "hard_negative_ratio": hard_negative_ratio,
                        "hard_negative_cap": hard_negative_cap,
                    },
                },
            )

        print(
            f"[setup] building model backbone={model_config.image_backbone} "
            f"pretrained_backbone={model_config.pretrained_image_backbone} "
            f"freeze_backbone={model_config.freeze_image_backbone} "
            f"teacher_seed_mode={model_config.teacher_seed_mode} "
            f"teacher_provider="
            f"{teacher_provider_config.kind if teacher_provider_config is not None else 'none'}",
            flush=True,
        )
        model = TSQBEVModel(model_config).to(resolved_device)
        if init_checkpoint is not None:
            load_weights_into_model_from_checkpoint(
                model,
                init_checkpoint,
                map_location=resolved_device,
            )
            print(f"[setup] warm-started model from {init_checkpoint}", flush=True)
        criterion = MultitaskCriterion(
            detection=_make_detection_criterion(
                loss_mode=loss_mode,
                hard_negative_ratio=hard_negative_ratio,
                hard_negative_cap=hard_negative_cap,
            )
        )
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = _make_scheduler(
            optimizer,
            epochs=epochs,
            optimizer_schedule=optimizer_schedule,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        history: list[dict[str, object]] = []
        checkpoint_last_path = artifact_root / "checkpoint_last.pt"
        checkpoint_best_path = artifact_root / "checkpoint_best.pt"
        best_epoch = 0
        best_val_total = float("inf")
        best_train_metrics: dict[str, float] | None = None
        best_val_metrics: dict[str, float] | None = None
        train_steps_completed = 0
        epochs_run = 0
        for epoch in range(1, epochs + 1):
            epoch_max_steps = None
            if max_train_steps is not None:
                remaining_steps = max_train_steps - train_steps_completed
                if remaining_steps <= 0:
                    break
                epoch_max_steps = min(len(train_loader), remaining_steps)
            print(
                f"[train] epoch={epoch}/{epochs} device={resolved_device.type} "
                f"train_samples={train_sample_count} val_samples={val_sample_count} "
                f"batch_size={batch_size} grad_accum_steps={grad_accum_steps} "
                f"max_train_steps={max_train_steps} "
                f"backbone={model_config.image_backbone} "
                f"pretrained_backbone={model_config.pretrained_image_backbone} "
                f"freeze_backbone={model_config.freeze_image_backbone}",
                flush=True,
            )
            train_metrics = _train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                grad_accum_steps=grad_accum_steps,
                device=resolved_device,
                amp_enabled=amp_enabled,
                scaler=scaler,
                epoch=epoch,
                log_every_steps=log_every_steps,
                max_steps=epoch_max_steps,
                grad_clip_norm=grad_clip_norm,
            )
            val_metrics = _eval_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=resolved_device,
                amp_enabled=amp_enabled,
                epoch=epoch,
            )
            scheduler.step()
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            _write_history(artifact_root, history)
            save_model_checkpoint(
                model,
                model_config,
                checkpoint_last_path,
                epoch=epoch,
                history=history,
            )
            if float(val_metrics["total"]) <= best_val_total:
                best_val_total = float(val_metrics["total"])
                best_epoch = epoch
                best_train_metrics = dict(train_metrics)
                best_val_metrics = dict(val_metrics)
                save_model_checkpoint(
                    model,
                    model_config,
                    checkpoint_best_path,
                    epoch=epoch,
                    history=history,
                )
            if active_tracker is not None:
                active_tracker.log(
                    {
                        "epoch": epoch,
                        **_prefixed_metrics("train", train_metrics),
                        **_prefixed_metrics("val", val_metrics),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "train_steps_completed": train_steps_completed
                        + (len(train_loader) if epoch_max_steps is None else epoch_max_steps),
                        "best_epoch": best_epoch,
                        "best_val_total": best_val_total,
                    },
                    step=epoch,
                )
            print(
                f"[epoch] completed epoch={epoch} train=({_format_metrics(train_metrics)}) "
                f"val=({_format_metrics(val_metrics)})",
                flush=True,
            )
            epochs_run = epoch
            train_steps_completed += (
                len(train_loader) if epoch_max_steps is None else epoch_max_steps
            )
            if max_train_steps is not None and train_steps_completed >= max_train_steps:
                break

        if not history:
            raise RuntimeError("fit_nuscenes completed without any train/val history")

        selected_checkpoint_path = (
            checkpoint_best_path
            if keep_best_checkpoint and checkpoint_best_path.exists()
            else checkpoint_last_path
        )
        selected_epoch = (
            best_epoch
            if selected_checkpoint_path == checkpoint_best_path
            else epochs_run
        )
        selected_train_metrics = (
            best_train_metrics
            if selected_checkpoint_path == checkpoint_best_path and best_train_metrics is not None
            else cast(dict[str, float], history[-1]["train"])
        )
        selected_val_metrics = (
            best_val_metrics
            if selected_checkpoint_path == checkpoint_best_path and best_val_metrics is not None
            else cast(dict[str, float], history[-1]["val"])
        )

        result = {
            "device": resolved_device.type,
            "amp_enabled": amp_enabled,
            "epochs": epochs_run,
            "max_train_steps": max_train_steps,
            "train_steps": train_steps_completed,
            "version": version,
            "train_split": resolved_train_split,
            "val_split": resolved_val_split,
            "artifact_dir": str(artifact_root),
            "checkpoint_path": str(selected_checkpoint_path),
            "selected_checkpoint_path": str(selected_checkpoint_path),
            "selected_epoch": selected_epoch,
            "selected_train": selected_train_metrics,
            "selected_val": selected_val_metrics,
            "last_checkpoint_path": str(checkpoint_last_path),
            "best_checkpoint_path": str(checkpoint_best_path),
            "best_epoch": best_epoch,
            "best_train": best_train_metrics,
            "best_val": best_val_metrics,
            "teacher_seed_mode": model_config.teacher_seed_mode,
            "train_samples": train_sample_count,
            "val_samples": val_sample_count,
            "last_train": history[-1]["train"],
            "last_val": history[-1]["val"],
            "teacher_provider": (
                teacher_provider_config.kind if teacher_provider_config is not None else None
            ),
            "history": history,
        }
        if active_tracker is not None:
            active_tracker.summary(
                {
                    "artifact_dir": str(artifact_root),
                    "checkpoint_path": str(selected_checkpoint_path),
                    "epochs": epochs_run,
                    "train_steps": train_steps_completed,
                    "train_samples": train_sample_count,
                    "val_samples": val_sample_count,
                    "best_epoch": best_epoch,
                    "best_val_total": best_val_total,
                    **_prefixed_metrics(
                        "final_train",
                        cast(dict[str, float], history[-1]["train"]),
                    ),
                    **_prefixed_metrics(
                        "final_val",
                        cast(dict[str, float], history[-1]["val"]),
                    ),
                    **_prefixed_metrics("selected_train", selected_train_metrics),
                    **_prefixed_metrics("selected_val", selected_val_metrics),
                }
            )
        status = "completed"
        return result
    finally:
        if owns_tracker and active_tracker is not None:
            active_tracker.finish(status=status)


def fit_openlane(
    dataroot: str | Path,
    artifact_dir: str | Path,
    config: ModelConfig | None = None,
    train_split: str = "training",
    val_split: str = "validation",
    subset: str = "lane3d_300",
    epochs: int = 6,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 8,
    num_workers: int = 4,
    batch_size: int = 1,
    device: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    use_amp: bool = False,
    log_every_steps: int | None = 100,
    tracker: ExperimentTracker | None = None,
    tracking_metadata: TrackingMetadata | None = None,
) -> dict[str, object]:
    """Fit the public lane baseline on OpenLane V1."""

    artifact_root = Path(artifact_dir)
    resolved_device = resolve_device(device)
    owns_tracker = tracker is None
    active_tracker = tracker
    status = "failed"
    try:
        if resolved_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        amp_enabled = bool(use_amp) and resolved_device.type == "cuda"
        model_config = config if config is not None else ModelConfig(views=1)
        print(f"[setup] loading OpenLane train split={train_split} from {dataroot}", flush=True)
        start_time = time.perf_counter()
        train_dataset = _subset_if_requested(
            OpenLaneDataset(
                dataroot=dataroot,
                split=train_split,
                subset=subset,
                lane_points=model_config.lane_points,
            ),
            max_train_samples,
        )
        print(
            f"[setup] loaded train split in {time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        print(f"[setup] loading OpenLane val split={val_split} from {dataroot}", flush=True)
        start_time = time.perf_counter()
        val_dataset = _subset_if_requested(
            OpenLaneDataset(
                dataroot=dataroot,
                split=val_split,
                subset=subset,
                lane_points=model_config.lane_points,
            ),
            max_val_samples,
        )
        print(
            f"[setup] loaded val split in {time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        train_loader = _make_loader(
            train_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        val_loader = _make_loader(
            val_dataset,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
        )

        train_sample_count = len(cast(Sized, train_dataset))
        val_sample_count = len(cast(Sized, val_dataset))
        if active_tracker is None:
            metadata = tracking_metadata or TrackingMetadata(
                suite="train",
                dataset="openlane",
                job_type="train-openlane",
                run_name=artifact_root.name,
                group=_default_tracking_group(artifact_root, "openlane"),
                tags=(subset, train_split, val_split, model_config.image_backbone),
                extra_config={
                    "train_split": train_split,
                    "val_split": val_split,
                    "subset": subset,
                    "train_samples": train_sample_count,
                    "val_samples": val_sample_count,
                },
            )
            active_tracker = start_experiment_tracking(
                artifact_dir=artifact_root,
                config=model_config,
                metadata=metadata,
                config_payload={
                    "model": model_config.model_dump(),
                    "train": {
                        "epochs": epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "grad_accum_steps": grad_accum_steps,
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                        "max_train_samples": max_train_samples,
                        "max_val_samples": max_val_samples,
                    },
                },
            )

        print(
            f"[setup] building model backbone={model_config.image_backbone} "
            f"pretrained_backbone={model_config.pretrained_image_backbone} "
            f"freeze_backbone={model_config.freeze_image_backbone}",
            flush=True,
        )
        model = TSQBEVModel(model_config).to(resolved_device)
        criterion = MultitaskCriterion()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        history: list[dict[str, object]] = []
        checkpoint_path = artifact_root / "checkpoint_last.pt"
        for epoch in range(1, epochs + 1):
            print(
                f"[train] epoch={epoch}/{epochs} device={resolved_device.type} "
                f"train_samples={train_sample_count} val_samples={val_sample_count}",
                flush=True,
            )
            train_metrics = _train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                grad_accum_steps=grad_accum_steps,
                device=resolved_device,
                amp_enabled=amp_enabled,
                scaler=scaler,
                epoch=epoch,
                log_every_steps=log_every_steps,
            )
            val_metrics = _eval_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=resolved_device,
                amp_enabled=amp_enabled,
                epoch=epoch,
            )
            scheduler.step()
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            _write_history(artifact_root, history)
            save_model_checkpoint(
                model,
                model_config,
                checkpoint_path,
                epoch=epoch,
                history=history,
            )
            if active_tracker is not None:
                active_tracker.log(
                    {
                        "epoch": epoch,
                        **_prefixed_metrics("train", train_metrics),
                        **_prefixed_metrics("val", val_metrics),
                        "learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=epoch,
                )
            print(
                f"[epoch] completed epoch={epoch} train=({_format_metrics(train_metrics)}) "
                f"val=({_format_metrics(val_metrics)})",
                flush=True,
            )

        result = {
            "device": resolved_device.type,
            "amp_enabled": amp_enabled,
            "epochs": epochs,
            "artifact_dir": str(artifact_root),
            "checkpoint_path": str(checkpoint_path),
            "train_samples": train_sample_count,
            "val_samples": val_sample_count,
            "last_train": history[-1]["train"],
            "last_val": history[-1]["val"],
        }
        if active_tracker is not None:
            active_tracker.summary(
                {
                    "artifact_dir": str(artifact_root),
                    "checkpoint_path": str(checkpoint_path),
                    "epochs": epochs,
                    "train_samples": train_sample_count,
                    "val_samples": val_sample_count,
                    **_prefixed_metrics(
                        "final_train",
                        cast(dict[str, float], history[-1]["train"]),
                    ),
                    **_prefixed_metrics(
                        "final_val",
                        cast(dict[str, float], history[-1]["val"]),
                    ),
                }
            )
        status = "completed"
        return result
    finally:
        if owns_tracker and active_tracker is not None:
            active_tracker.finish(status=status)
