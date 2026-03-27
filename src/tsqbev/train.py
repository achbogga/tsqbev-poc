"""Real-data training loops for public baseline runs.

References:
- PETRv2 multitask sparse-query optimization:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf
- BEVDistill student-training framing:
  https://arxiv.org/abs/2211.09386
"""

from __future__ import annotations

import json
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

from tsqbev.checkpoints import save_model_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.datasets import NuScenesDataset, OpenLaneDataset, collate_single_scene_example
from tsqbev.losses import MultitaskCriterion
from tsqbev.model import TSQBEVModel
from tsqbev.runtime import move_batch, resolve_device


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


def _train_epoch(
    model: TSQBEVModel,
    loader: DataLoader,
    criterion: MultitaskCriterion,
    optimizer: AdamW,
    grad_accum_steps: int,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, float]] = []
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32

    for step, (batch, _) in enumerate(loader, start=1):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        history.append(_to_float_metrics(losses))

    if len(loader) % grad_accum_steps != 0:
        if amp_enabled:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    return _average_history(history)


def _write_history(artifact_dir: Path, history: list[dict[str, object]]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "history.json").write_text(json.dumps(history, indent=2))


def _make_loader(dataset: Dataset, shuffle: bool, num_workers: int) -> DataLoader:
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_single_scene_example,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_single_scene_example,
        pin_memory=torch.cuda.is_available(),
    )


def fit_nuscenes(
    dataroot: str | Path,
    artifact_dir: str | Path,
    config: ModelConfig | None = None,
    version: str = "v1.0-trainval",
    train_split: str = "train",
    val_split: str = "val",
    epochs: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 8,
    num_workers: int = 4,
    device: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    use_amp: bool = False,
) -> dict[str, object]:
    """Fit the public object-detection baseline on nuScenes train/val."""

    resolved_device = resolve_device(device)
    amp_enabled = bool(use_amp) and resolved_device.type == "cuda"
    model_config = config if config is not None else ModelConfig()
    train_dataset = _subset_if_requested(
        NuScenesDataset(dataroot=dataroot, version=version, split=train_split),
        max_train_samples,
    )
    val_dataset = _subset_if_requested(
        NuScenesDataset(dataroot=dataroot, version=version, split=val_split),
        max_val_samples,
    )
    train_loader = _make_loader(train_dataset, shuffle=True, num_workers=num_workers)
    val_loader = _make_loader(val_dataset, shuffle=False, num_workers=num_workers)

    model = TSQBEVModel(model_config).to(resolved_device)
    criterion = MultitaskCriterion()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, object]] = []
    checkpoint_path = Path(artifact_dir) / "checkpoint_last.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = _train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            grad_accum_steps=grad_accum_steps,
            device=resolved_device,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        val_metrics = _eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=resolved_device,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        _write_history(Path(artifact_dir), history)
        save_model_checkpoint(
            model,
            model_config,
            checkpoint_path,
            epoch=epoch,
            history=history,
        )

    return {
        "device": resolved_device.type,
        "amp_enabled": amp_enabled,
        "epochs": epochs,
        "artifact_dir": str(artifact_dir),
        "checkpoint_path": str(checkpoint_path),
        "train_samples": len(cast(Sized, train_dataset)),
        "val_samples": len(cast(Sized, val_dataset)),
        "last_train": history[-1]["train"],
        "last_val": history[-1]["val"],
    }


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
    device: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    use_amp: bool = False,
) -> dict[str, object]:
    """Fit the public lane baseline on OpenLane V1."""

    resolved_device = resolve_device(device)
    amp_enabled = bool(use_amp) and resolved_device.type == "cuda"
    model_config = config if config is not None else ModelConfig(views=1)
    train_dataset = _subset_if_requested(
        OpenLaneDataset(
            dataroot=dataroot,
            split=train_split,
            subset=subset,
            lane_points=model_config.lane_points,
        ),
        max_train_samples,
    )
    val_dataset = _subset_if_requested(
        OpenLaneDataset(
            dataroot=dataroot,
            split=val_split,
            subset=subset,
            lane_points=model_config.lane_points,
        ),
        max_val_samples,
    )
    train_loader = _make_loader(train_dataset, shuffle=True, num_workers=num_workers)
    val_loader = _make_loader(val_dataset, shuffle=False, num_workers=num_workers)

    model = TSQBEVModel(model_config).to(resolved_device)
    criterion = MultitaskCriterion()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, object]] = []
    checkpoint_path = Path(artifact_dir) / "checkpoint_last.pt"
    for epoch in range(1, epochs + 1):
        train_metrics = _train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            grad_accum_steps=grad_accum_steps,
            device=resolved_device,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        val_metrics = _eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=resolved_device,
            amp_enabled=amp_enabled,
        )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        _write_history(Path(artifact_dir), history)
        save_model_checkpoint(
            model,
            model_config,
            checkpoint_path,
            epoch=epoch,
            history=history,
        )

    return {
        "device": resolved_device.type,
        "amp_enabled": amp_enabled,
        "epochs": epochs,
        "artifact_dir": str(artifact_dir),
        "checkpoint_path": str(checkpoint_path),
        "train_samples": len(cast(Sized, train_dataset)),
        "val_samples": len(cast(Sized, val_dataset)),
        "last_train": history[-1]["train"],
        "last_val": history[-1]["val"],
    }
