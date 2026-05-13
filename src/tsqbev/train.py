"""Real-data training loops for public baseline runs.

References:
- PETRv2 multitask sparse-query optimization:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf
- BEVDistill student-training framing:
  https://arxiv.org/abs/2211.09386
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sized
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

from tsqbev.checkpoints import load_weights_into_model_from_checkpoint, save_model_checkpoint
from tsqbev.config import ModelConfig
from tsqbev.datasets import NuScenesDataset, OpenLaneDataset, collate_scene_examples
from tsqbev.eval_nuscenes import (
    evaluate_nuscenes_predictions,
    export_nuscenes_predictions,
    export_sanity_diagnostics,
)
from tsqbev.eval_openlane import (
    evaluate_openlane_predictions,
    export_openlane_predictions,
    write_openlane_test_list,
)
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


def set_global_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch RNGs for repeatable bounded runs."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _teacher_anchor_schedule_value(
    *,
    epoch: int,
    initial_weight: float,
    final_weight: float,
    bootstrap_epochs: int,
    decay_epochs: int,
) -> float:
    if decay_epochs <= 0 or abs(final_weight - initial_weight) < 1e-8:
        return initial_weight
    if epoch <= bootstrap_epochs:
        return initial_weight
    decay_step = epoch - bootstrap_epochs
    if decay_step >= decay_epochs:
        return final_weight
    progress = decay_step / float(decay_epochs)
    return initial_weight + (final_weight - initial_weight) * progress


def _make_detection_criterion(
    *,
    loss_mode: Literal["baseline", "focal_hardneg", "quality_focal"],
    hard_negative_ratio: int,
    hard_negative_cap: int,
    teacher_anchor_class_weight: float,
    teacher_anchor_quality_class_weight: float,
    teacher_anchor_objectness_weight: float,
    teacher_region_objectness_weight: float = 0.0,
    teacher_region_class_weight: float = 0.0,
    teacher_region_radius_m: float = 4.0,
) -> DetectionSetCriterion:
    return DetectionSetCriterion(
        loss_mode=loss_mode,
        hard_negative_ratio=hard_negative_ratio,
        hard_negative_cap=hard_negative_cap,
        teacher_anchor_class_weight=teacher_anchor_class_weight,
        teacher_anchor_quality_class_weight=teacher_anchor_quality_class_weight,
        teacher_anchor_objectness_weight=teacher_anchor_objectness_weight,
        teacher_region_objectness_weight=teacher_region_objectness_weight,
        teacher_region_class_weight=teacher_region_class_weight,
        teacher_region_radius_m=teacher_region_radius_m,
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


def _scale_metrics(metrics: dict[str, float], scale: float) -> dict[str, float]:
    return {name: value * scale for name, value in metrics.items()}


def _merge_metric_histories(
    left: list[dict[str, float]],
    right: list[dict[str, float]],
) -> dict[str, float]:
    if not left and not right:
        raise ValueError("cannot merge empty metric histories")
    merged: dict[str, list[float]] = {}
    for history in (left, right):
        for row in history:
            for key, value in row.items():
                merged.setdefault(key, []).append(value)
    return {key: sum(values) / float(len(values)) for key, values in merged.items()}


def _write_history(artifact_dir: Path, history: list[dict[str, object]]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "history.json").write_text(json.dumps(history, indent=2))


def _make_loader(
    dataset: Dataset,
    shuffle: bool,
    num_workers: int,
    batch_size: int,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
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
            generator=generator,
            worker_init_fn=_seed_worker,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_scene_examples,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _make_round_robin_plan(*named_lengths: tuple[str, int]) -> list[str]:
    remaining = {name: length for name, length in named_lengths if length > 0}
    plan: list[str] = []
    while remaining:
        for name in list(remaining.keys()):
            plan.append(name)
            remaining[name] -= 1
            if remaining[name] <= 0:
                del remaining[name]
    return plan


def _lane_batches_per_epoch(
    *,
    detection_batches: int,
    lane_batches: int,
    lane_batch_multiplier: float,
) -> int:
    if detection_batches <= 0 or lane_batches <= 0:
        return 0
    capped = int(round(float(detection_batches) * lane_batch_multiplier))
    return max(1, min(lane_batches, capped))


def _train_joint_epoch(
    *,
    model: TSQBEVModel,
    detection_loader: DataLoader,
    lane_loader: DataLoader,
    criterion: MultitaskCriterion,
    optimizer: AdamW,
    grad_accum_steps: int,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
    epoch: int,
    log_every_steps: int | None,
    lane_loss_scale: float,
    lane_batch_multiplier: float,
    grad_clip_norm: float | None = 1.0,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    detection_iter = iter(detection_loader)
    lane_iter = iter(lane_loader)
    detection_history: list[dict[str, float]] = []
    lane_history: list[dict[str, float]] = []
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
    lane_batches_per_epoch = _lane_batches_per_epoch(
        detection_batches=len(detection_loader),
        lane_batches=len(lane_loader),
        lane_batch_multiplier=lane_batch_multiplier,
    )
    plan = _make_round_robin_plan(
        ("detection", len(detection_loader)),
        ("lane", lane_batches_per_epoch),
    )
    total_steps = len(plan)
    optimizer_steps = 0

    for step_index, task_name in enumerate(plan, start=1):
        batch, _ = next(detection_iter if task_name == "detection" else lane_iter)
        batch = move_batch(batch, device)
        loss_scale = 1.0 if task_name == "detection" else lane_loss_scale
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            outputs = model(batch)
            losses = criterion(outputs, batch)
        scaled_total = losses["total"] * loss_scale
        scaled_loss = scaled_total / float(grad_accum_steps)
        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        step_metrics = _to_float_metrics(losses)
        step_metrics["total"] *= loss_scale
        if task_name == "detection":
            detection_history.append(step_metrics)
        else:
            lane_history.append(step_metrics)

        if step_index % grad_accum_steps == 0:
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
            optimizer_steps += 1

        should_log = (
            log_every_steps is not None
            and (step_index == 1 or step_index % log_every_steps == 0 or step_index == total_steps)
        )
        if should_log:
            print(
                f"[joint-train] epoch={epoch} step={step_index}/{total_steps} "
                f"task={task_name} {_format_metrics(step_metrics)}",
                flush=True,
            )

    if total_steps % grad_accum_steps != 0:
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
        optimizer_steps += 1

    return (
        _merge_metric_histories(detection_history, []),
        _merge_metric_histories(lane_history, []),
        {
            "optimizer_steps": float(optimizer_steps),
            "lane_loss_scale": lane_loss_scale,
            "lane_batch_multiplier": lane_batch_multiplier,
            "lane_batches_per_epoch": float(lane_batches_per_epoch),
            "total_batches": float(total_steps),
        },
    )


@torch.no_grad()
def _eval_joint_epoch(
    *,
    model: TSQBEVModel,
    detection_loader: DataLoader,
    lane_loader: DataLoader,
    criterion: MultitaskCriterion,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
    lane_loss_scale: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    detection_metrics = _eval_epoch(
        model=model,
        loader=detection_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        epoch=epoch,
    )
    lane_metrics = _eval_epoch(
        model=model,
        loader=lane_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        epoch=epoch,
    )
    selection = {
        "detection_total": detection_metrics["total"],
        "lane_total": lane_metrics["total"],
        "joint_total": detection_metrics["total"] + lane_metrics["total"] * lane_loss_scale,
        "lane_loss_scale": lane_loss_scale,
    }
    return detection_metrics, lane_metrics, selection


def _run_joint_public_official_eval(
    *,
    checkpoint_path: Path,
    model_config: ModelConfig,
    device: str | None,
    artifact_root: Path,
    epoch: int,
    nuscenes_root: str | Path,
    nuscenes_version: str,
    nuscenes_split: str,
    openlane_root: str | Path,
    openlane_subset: str,
    openlane_repo_root: str | Path,
    score_threshold: float,
    top_k: int,
) -> dict[str, float]:
    eval_model = TSQBEVModel(model_config)
    resolved_device = resolve_device(device)
    load_weights_into_model_from_checkpoint(
        eval_model,
        checkpoint_path,
        map_location=resolved_device,
    )

    eval_root = artifact_root / "official_eval" / f"epoch_{epoch:03d}"
    det_output_dir = eval_root / "nuscenes"
    lane_output_dir = eval_root / "openlane"
    eval_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        "nuscenes_nds": 0.0,
        "nuscenes_map": 0.0,
        "nuscenes_eval_ok": 0.0,
        "nuscenes_export_sanity_ok": 0.0,
        "nuscenes_boxes_per_sample_mean": 0.0,
        "nuscenes_ego_translation_norm_p99": 0.0,
        "nuscenes_max_box_size_m": 0.0,
        "nuscenes_score_mean": 0.0,
        "openlane_f_score": 0.0,
        "openlane_recall": 0.0,
        "openlane_precision": 0.0,
        "openlane_eval_ok": 0.0,
    }
    errors: dict[str, str] = {}
    summary_payload: dict[str, object] = {"metrics": metrics, "errors": errors}

    try:
        det_result_path = export_nuscenes_predictions(
            model=eval_model,
            dataroot=nuscenes_root,
            version=nuscenes_version,
            split=nuscenes_split,
            output_path=det_output_dir / "predictions.json",
            score_threshold=score_threshold,
            top_k=top_k,
            device=device,
        )
        det_metrics = evaluate_nuscenes_predictions(
            dataroot=nuscenes_root,
            version=nuscenes_version,
            split=nuscenes_split,
            result_path=det_result_path,
            output_dir=det_output_dir / "metrics",
        )
        sanity = export_sanity_diagnostics(
            det_result_path,
            dataroot=nuscenes_root,
            version=nuscenes_version,
        )
        metrics["nuscenes_nds"] = float(cast(float, det_metrics.get("nd_score", 0.0)))
        metrics["nuscenes_map"] = float(cast(float, det_metrics.get("mean_ap", 0.0)))
        metrics["nuscenes_eval_ok"] = 1.0
        metrics["nuscenes_export_sanity_ok"] = float(sanity.get("sanity_ok", 0.0))
        metrics["nuscenes_boxes_per_sample_mean"] = float(
            sanity.get("boxes_per_sample_mean", 0.0)
        )
        metrics["nuscenes_ego_translation_norm_p99"] = float(
            sanity.get("ego_translation_norm_p99", 0.0)
        )
        metrics["nuscenes_max_box_size_m"] = float(sanity.get("max_box_size_m", 0.0))
        metrics["nuscenes_score_mean"] = float(sanity.get("score_mean", 0.0))
        summary_payload["nuscenes_sanity"] = sanity
    except Exception as exc:
        errors["nuscenes"] = repr(exc)

    try:
        lane_pred_dir = export_openlane_predictions(
            model=eval_model,
            dataroot=openlane_root,
            output_dir=lane_output_dir / "predictions",
            split="validation",
            subset=openlane_subset,
            score_threshold=0.5,
            max_lanes=64,
            device=device,
        )
        lane_test_list = write_openlane_test_list(
            dataroot=openlane_root,
            output_path=lane_output_dir / "validation_test_list.txt",
            split="validation",
            subset=openlane_subset,
        )
        lane_metrics = evaluate_openlane_predictions(
            openlane_repo_root=openlane_repo_root,
            dataset_dir=Path(openlane_root) / openlane_subset,
            pred_dir=lane_pred_dir,
            test_list=lane_test_list,
        )
        metrics["openlane_f_score"] = float(lane_metrics.get("f_score", 0.0))
        metrics["openlane_recall"] = float(lane_metrics.get("recall", 0.0))
        metrics["openlane_precision"] = float(lane_metrics.get("precision", 0.0))
        metrics["openlane_eval_ok"] = 1.0
    except Exception as exc:
        errors["openlane"] = repr(exc)

    (eval_root / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    return metrics


def _run_nuscenes_official_eval(
    *,
    checkpoint_path: Path,
    model_config: ModelConfig,
    device: str | None,
    artifact_root: Path,
    epoch: int,
    nuscenes_root: str | Path,
    nuscenes_version: str,
    nuscenes_split: str,
    score_threshold: float,
    top_k: int,
) -> dict[str, float]:
    eval_model = TSQBEVModel(model_config)
    resolved_device = resolve_device(device)
    load_weights_into_model_from_checkpoint(
        eval_model,
        checkpoint_path,
        map_location=resolved_device,
    )

    eval_root = artifact_root / "official_eval" / f"epoch_{epoch:03d}"
    det_output_dir = eval_root / "nuscenes"
    eval_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        "nuscenes_nds": 0.0,
        "nuscenes_map": 0.0,
        "nuscenes_eval_ok": 0.0,
        "nuscenes_export_sanity_ok": 0.0,
        "nuscenes_boxes_per_sample_mean": 0.0,
        "nuscenes_ego_translation_norm_p99": 0.0,
        "nuscenes_max_box_size_m": 0.0,
        "nuscenes_score_mean": 0.0,
    }
    errors: dict[str, str] = {}
    summary_payload: dict[str, object] = {"metrics": metrics, "errors": errors}

    try:
        det_result_path = export_nuscenes_predictions(
            model=eval_model,
            dataroot=nuscenes_root,
            version=nuscenes_version,
            split=nuscenes_split,
            output_path=det_output_dir / "predictions.json",
            score_threshold=score_threshold,
            top_k=top_k,
            device=device,
        )
        det_metrics = evaluate_nuscenes_predictions(
            dataroot=nuscenes_root,
            version=nuscenes_version,
            split=nuscenes_split,
            result_path=det_result_path,
            output_dir=det_output_dir / "metrics",
        )
        sanity = export_sanity_diagnostics(
            det_result_path,
            dataroot=nuscenes_root,
            version=nuscenes_version,
        )
        metrics["nuscenes_nds"] = float(cast(float, det_metrics.get("nd_score", 0.0)))
        metrics["nuscenes_map"] = float(cast(float, det_metrics.get("mean_ap", 0.0)))
        metrics["nuscenes_eval_ok"] = 1.0
        metrics["nuscenes_export_sanity_ok"] = float(sanity.get("sanity_ok", 0.0))
        metrics["nuscenes_boxes_per_sample_mean"] = float(
            sanity.get("boxes_per_sample_mean", 0.0)
        )
        metrics["nuscenes_ego_translation_norm_p99"] = float(
            sanity.get("ego_translation_norm_p99", 0.0)
        )
        metrics["nuscenes_max_box_size_m"] = float(sanity.get("max_box_size_m", 0.0))
        metrics["nuscenes_score_mean"] = float(sanity.get("score_mean", 0.0))
        summary_payload["nuscenes_sanity"] = sanity
    except Exception as exc:
        errors["nuscenes"] = repr(exc)

    (eval_root / "summary.json").write_text(json.dumps(summary_payload, indent=2))
    return metrics


def _nuscenes_official_metric_key(metrics: dict[str, float] | None) -> tuple[float, ...]:
    if metrics is None:
        return (0.0,) * 7
    return (
        float(metrics.get("nuscenes_eval_ok", 0.0)),
        float(metrics.get("nuscenes_export_sanity_ok", 0.0)),
        float(metrics.get("nuscenes_nds", 0.0)),
        float(metrics.get("nuscenes_map", 0.0)),
        -float(metrics.get("nuscenes_boxes_per_sample_mean", float("inf"))),
        -float(metrics.get("nuscenes_ego_translation_norm_p99", float("inf"))),
        -float(metrics.get("nuscenes_max_box_size_m", float("inf"))),
    )


def _nuscenes_official_metrics_better(
    candidate: dict[str, float] | None,
    incumbent: dict[str, float] | None,
) -> bool:
    return _nuscenes_official_metric_key(candidate) > _nuscenes_official_metric_key(incumbent)


def _joint_official_metric_key(metrics: dict[str, float] | None) -> tuple[float, ...]:
    if metrics is None:
        return (0.0,) * 8
    return (
        float(metrics.get("nuscenes_eval_ok", 0.0)),
        float(metrics.get("nuscenes_export_sanity_ok", 0.0)),
        float(metrics.get("nuscenes_nds", 0.0)),
        float(metrics.get("nuscenes_map", 0.0)),
        float(metrics.get("openlane_eval_ok", 0.0)),
        float(metrics.get("openlane_f_score", 0.0)),
        float(metrics.get("openlane_precision", 0.0)),
        float(metrics.get("openlane_recall", 0.0)),
    )


def _joint_official_metrics_better(
    candidate: dict[str, float] | None,
    incumbent: dict[str, float] | None,
) -> bool:
    return _joint_official_metric_key(candidate) > _joint_official_metric_key(incumbent)


def _catastrophic_nuscenes_official_failure(metrics: dict[str, float] | None) -> bool:
    if metrics is None:
        return False
    eval_ok = float(metrics.get("nuscenes_eval_ok", 0.0)) >= 1.0
    if not eval_ok:
        return False
    nds = float(metrics.get("nuscenes_nds", 0.0))
    mean_ap = float(metrics.get("nuscenes_map", 0.0))
    sanity_ok = float(metrics.get("nuscenes_export_sanity_ok", 0.0))
    max_box_size_m = float(metrics.get("nuscenes_max_box_size_m", 0.0))
    score_mean = float(metrics.get("nuscenes_score_mean", 0.0))
    ego_translation_p99 = float(metrics.get("nuscenes_ego_translation_norm_p99", 0.0))
    return (
        nds <= 1e-8
        and mean_ap <= 1e-8
        and (
            sanity_ok <= 0.0
            or max_box_size_m > 100.0
            or score_mean > 0.99
            or ego_translation_p99 > 20.0
        )
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
    seed: int | None = None,
    teacher_provider_config: TeacherProviderConfig | None = None,
    init_checkpoint: str | Path | None = None,
    use_amp: bool = False,
    log_every_steps: int | None = 100,
    optimizer_schedule: Literal["cosine", "constant"] = "cosine",
    grad_clip_norm: float | None = 1.0,
    keep_best_checkpoint: bool = False,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    early_stop_min_epochs: int = 0,
    official_eval_every_epochs: int | None = None,
    official_eval_score_threshold: float = 0.20,
    official_eval_top_k: int = 40,
    augmentation_mode: Literal["off", "moderate", "strong"] = "off",
    loss_mode: Literal["baseline", "focal_hardneg", "quality_focal"] = "baseline",
    hard_negative_ratio: int = 3,
    hard_negative_cap: int = 96,
    teacher_anchor_class_weight: float = 0.5,
    teacher_anchor_quality_class_weight: float = 0.0,
    teacher_anchor_objectness_weight: float = 0.5,
    teacher_region_objectness_weight: float = 0.0,
    teacher_region_class_weight: float = 0.0,
    teacher_region_radius_m: float = 4.0,
    teacher_anchor_final_class_weight: float | None = None,
    teacher_anchor_final_objectness_weight: float | None = None,
    teacher_anchor_bootstrap_epochs: int = 0,
    teacher_anchor_decay_epochs: int = 0,
    enable_teacher_distillation: bool = True,
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
        set_global_seed(seed)
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
                augmentation_mode=augmentation_mode,
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
                augmentation_mode="off",
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
                        "seed": seed,
                        "init_checkpoint": (
                            str(init_checkpoint) if init_checkpoint is not None else None
                        ),
                        "optimizer_schedule": optimizer_schedule,
                        "grad_clip_norm": grad_clip_norm,
                        "keep_best_checkpoint": keep_best_checkpoint,
                        "early_stop_patience": early_stop_patience,
                        "early_stop_min_delta": early_stop_min_delta,
                        "early_stop_min_epochs": early_stop_min_epochs,
                        "official_eval_every_epochs": official_eval_every_epochs,
                        "official_eval_score_threshold": official_eval_score_threshold,
                        "official_eval_top_k": official_eval_top_k,
                        "augmentation_mode": augmentation_mode,
                        "loss_mode": loss_mode,
                        "hard_negative_ratio": hard_negative_ratio,
                        "hard_negative_cap": hard_negative_cap,
                        "teacher_anchor_class_weight": teacher_anchor_class_weight,
                        "teacher_anchor_quality_class_weight": (
                            teacher_anchor_quality_class_weight
                        ),
                        "teacher_anchor_objectness_weight": teacher_anchor_objectness_weight,
                        "teacher_region_objectness_weight": teacher_region_objectness_weight,
                        "teacher_region_class_weight": teacher_region_class_weight,
                        "teacher_region_radius_m": teacher_region_radius_m,
                        "teacher_anchor_final_class_weight": teacher_anchor_final_class_weight,
                        "teacher_anchor_final_objectness_weight": (
                            teacher_anchor_final_objectness_weight
                        ),
                        "teacher_anchor_bootstrap_epochs": teacher_anchor_bootstrap_epochs,
                        "teacher_anchor_decay_epochs": teacher_anchor_decay_epochs,
                        "enable_teacher_distillation": enable_teacher_distillation,
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
                teacher_anchor_class_weight=teacher_anchor_class_weight,
                teacher_anchor_quality_class_weight=teacher_anchor_quality_class_weight,
                teacher_anchor_objectness_weight=teacher_anchor_objectness_weight,
                teacher_region_objectness_weight=teacher_region_objectness_weight,
                teacher_region_class_weight=teacher_region_class_weight,
                teacher_region_radius_m=teacher_region_radius_m,
            ),
            enable_distillation=enable_teacher_distillation,
        )
        final_teacher_anchor_class_weight = (
            teacher_anchor_class_weight
            if teacher_anchor_final_class_weight is None
            else teacher_anchor_final_class_weight
        )
        final_teacher_anchor_objectness_weight = (
            teacher_anchor_objectness_weight
            if teacher_anchor_final_objectness_weight is None
            else teacher_anchor_final_objectness_weight
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
        checkpoint_best_official_path = artifact_root / "checkpoint_best_official.pt"
        best_epoch = 0
        best_val_total = float("inf")
        best_train_metrics: dict[str, float] | None = None
        best_val_metrics: dict[str, float] | None = None
        best_official_epoch = 0
        best_official_metrics: dict[str, float] | None = None
        latest_official_metrics: dict[str, float] | None = None
        epochs_without_improvement = 0
        early_stop_triggered = False
        early_stop_reason: str | None = None
        train_steps_completed = 0
        epochs_run = 0
        for epoch in range(1, epochs + 1):
            epoch_max_steps = None
            if max_train_steps is not None:
                remaining_steps = max_train_steps - train_steps_completed
                if remaining_steps <= 0:
                    break
                epoch_max_steps = min(len(train_loader), remaining_steps)
            current_teacher_anchor_class_weight = _teacher_anchor_schedule_value(
                epoch=epoch,
                initial_weight=teacher_anchor_class_weight,
                final_weight=final_teacher_anchor_class_weight,
                bootstrap_epochs=teacher_anchor_bootstrap_epochs,
                decay_epochs=teacher_anchor_decay_epochs,
            )
            current_teacher_anchor_objectness_weight = _teacher_anchor_schedule_value(
                epoch=epoch,
                initial_weight=teacher_anchor_objectness_weight,
                final_weight=final_teacher_anchor_objectness_weight,
                bootstrap_epochs=teacher_anchor_bootstrap_epochs,
                decay_epochs=teacher_anchor_decay_epochs,
            )
            criterion.detection.set_teacher_anchor_weights(
                class_weight=current_teacher_anchor_class_weight,
                objectness_weight=current_teacher_anchor_objectness_weight,
            )
            print(
                f"[train] epoch={epoch}/{epochs} device={resolved_device.type} "
                f"train_samples={train_sample_count} val_samples={val_sample_count} "
                f"batch_size={batch_size} grad_accum_steps={grad_accum_steps} "
                f"max_train_steps={max_train_steps} "
                f"backbone={model_config.image_backbone} "
                f"pretrained_backbone={model_config.pretrained_image_backbone} "
                f"freeze_backbone={model_config.freeze_image_backbone} "
                f"teacher_anchor_cls_w={current_teacher_anchor_class_weight:.3f} "
                f"teacher_anchor_qcls_w={teacher_anchor_quality_class_weight:.3f} "
                f"teacher_anchor_obj_w={current_teacher_anchor_objectness_weight:.3f} "
                f"teacher_region_obj_w={teacher_region_objectness_weight:.3f} "
                f"teacher_region_cls_w={teacher_region_class_weight:.3f} "
                f"augment={augmentation_mode}",
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
            current_val_total = float(val_metrics["total"])
            if current_val_total < (best_val_total - early_stop_min_delta):
                best_val_total = current_val_total
                best_epoch = epoch
                best_train_metrics = dict(train_metrics)
                best_val_metrics = dict(val_metrics)
                epochs_without_improvement = 0
                save_model_checkpoint(
                    model,
                    model_config,
                    checkpoint_best_path,
                    epoch=epoch,
                    history=history,
                )
            else:
                epochs_without_improvement += 1
            should_run_official_eval = (
                official_eval_every_epochs is not None
                and official_eval_every_epochs > 0
                and epoch % official_eval_every_epochs == 0
            )
            if should_run_official_eval:
                eval_checkpoint_path = (
                    checkpoint_best_path
                    if keep_best_checkpoint and checkpoint_best_path.exists()
                    else checkpoint_last_path
                )
                latest_official_metrics = _run_nuscenes_official_eval(
                    checkpoint_path=eval_checkpoint_path,
                    model_config=model_config,
                    device=device,
                    artifact_root=artifact_root,
                    epoch=epoch,
                    nuscenes_root=dataroot,
                    nuscenes_version=version,
                    nuscenes_split=resolved_val_split,
                    score_threshold=official_eval_score_threshold,
                    top_k=official_eval_top_k,
                )
                history[-1]["official_eval"] = latest_official_metrics
                _write_history(artifact_root, history)
                if _nuscenes_official_metrics_better(
                    latest_official_metrics,
                    best_official_metrics,
                ):
                    best_official_metrics = dict(latest_official_metrics)
                    best_official_epoch = epoch
                    save_model_checkpoint(
                        model,
                        model_config,
                        checkpoint_best_official_path,
                        epoch=epoch,
                        history=history,
                    )
                if active_tracker is not None:
                    active_tracker.log({"epoch": epoch, **latest_official_metrics}, step=epoch)
                print(
                    "[official-eval] "
                    f"epoch={epoch} "
                    f"nds={latest_official_metrics['nuscenes_nds']:.4f} "
                    f"map={latest_official_metrics['nuscenes_map']:.4f} "
                    f"sanity_ok={latest_official_metrics['nuscenes_export_sanity_ok']:.0f}",
                    flush=True,
                )
                if _catastrophic_nuscenes_official_failure(latest_official_metrics):
                    epochs_run = epoch
                    train_steps_completed += (
                        len(train_loader) if epoch_max_steps is None else epoch_max_steps
                    )
                    early_stop_triggered = True
                    early_stop_reason = (
                        "catastrophic official-eval failure: "
                        f"epoch={epoch} nds={latest_official_metrics['nuscenes_nds']:.4f} "
                        f"map={latest_official_metrics['nuscenes_map']:.4f} "
                        f"sanity_ok={latest_official_metrics['nuscenes_export_sanity_ok']:.0f} "
                        f"max_box_size_m={latest_official_metrics['nuscenes_max_box_size_m']:.2f} "
                        f"score_mean={latest_official_metrics['nuscenes_score_mean']:.4f}"
                    )
                    print(f"[early-stop] {early_stop_reason}", flush=True)
                    break
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
                        "epochs_without_improvement": epochs_without_improvement,
                        "teacher_anchor_class_weight": current_teacher_anchor_class_weight,
                        "teacher_anchor_quality_class_weight": (
                            teacher_anchor_quality_class_weight
                        ),
                        "teacher_anchor_objectness_weight": (
                            current_teacher_anchor_objectness_weight
                        ),
                        "teacher_region_objectness_weight": teacher_region_objectness_weight,
                        "teacher_region_class_weight": teacher_region_class_weight,
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
            if (
                early_stop_patience is not None
                and epoch >= early_stop_min_epochs
                and epochs_without_improvement >= early_stop_patience
            ):
                early_stop_triggered = True
                early_stop_reason = (
                    "plateau: "
                    f"best_val_total={best_val_total:.4f} at epoch={best_epoch}, "
                    f"no improvement > {early_stop_min_delta:.4f} for "
                    f"{epochs_without_improvement} epoch(s)"
                )
                print(f"[early-stop] {early_stop_reason}", flush=True)
                break
            if max_train_steps is not None and train_steps_completed >= max_train_steps:
                break

        if not history:
            raise RuntimeError("fit_nuscenes completed without any train/val history")

        if checkpoint_best_official_path.exists():
            selected_checkpoint_path = checkpoint_best_official_path
        else:
            selected_checkpoint_path = (
                checkpoint_best_path
                if keep_best_checkpoint and checkpoint_best_path.exists()
                else checkpoint_last_path
            )
        selected_epoch = (
            best_epoch
            if selected_checkpoint_path in {checkpoint_best_path, checkpoint_best_official_path}
            else epochs_run
        )
        selected_train_metrics = (
            best_train_metrics
            if (
                selected_checkpoint_path in {checkpoint_best_path, checkpoint_best_official_path}
                and best_train_metrics is not None
            )
            else cast(dict[str, float], history[-1]["train"])
        )
        selected_val_metrics = (
            best_val_metrics
            if (
                selected_checkpoint_path in {checkpoint_best_path, checkpoint_best_official_path}
                and best_val_metrics is not None
            )
            else cast(dict[str, float], history[-1]["val"])
        )

        result = {
            "device": resolved_device.type,
            "amp_enabled": amp_enabled,
            "epochs": epochs_run,
            "max_train_steps": max_train_steps,
            "seed": seed,
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
            "best_official_checkpoint_path": (
                str(checkpoint_best_official_path)
                if checkpoint_best_official_path.exists()
                else None
            ),
            "best_epoch": best_epoch,
            "best_train": best_train_metrics,
            "best_val": best_val_metrics,
            "best_official_epoch": best_official_epoch,
            "best_official_metrics": best_official_metrics,
            "teacher_seed_mode": model_config.teacher_seed_mode,
            "train_samples": train_sample_count,
            "val_samples": val_sample_count,
            "last_train": history[-1]["train"],
            "last_val": history[-1]["val"],
            "teacher_provider": (
                teacher_provider_config.kind if teacher_provider_config is not None else None
            ),
            "enable_teacher_distillation": enable_teacher_distillation,
            "augmentation_mode": augmentation_mode,
            "teacher_anchor_class_weight": teacher_anchor_class_weight,
            "teacher_anchor_quality_class_weight": teacher_anchor_quality_class_weight,
            "teacher_anchor_objectness_weight": teacher_anchor_objectness_weight,
            "teacher_region_objectness_weight": teacher_region_objectness_weight,
            "teacher_region_class_weight": teacher_region_class_weight,
            "teacher_region_radius_m": teacher_region_radius_m,
            "teacher_anchor_final_class_weight": final_teacher_anchor_class_weight,
            "teacher_anchor_final_objectness_weight": final_teacher_anchor_objectness_weight,
            "teacher_anchor_bootstrap_epochs": teacher_anchor_bootstrap_epochs,
            "teacher_anchor_decay_epochs": teacher_anchor_decay_epochs,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "early_stop_min_epochs": early_stop_min_epochs,
            "official_eval_every_epochs": official_eval_every_epochs,
            "official_eval_score_threshold": official_eval_score_threshold,
            "official_eval_top_k": official_eval_top_k,
            "latest_official_eval": latest_official_metrics,
            "early_stop_triggered": early_stop_triggered,
            "early_stop_reason": early_stop_reason,
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
                    "best_official_epoch": best_official_epoch,
                    "best_val_total": best_val_total,
                    "official_eval_every_epochs": official_eval_every_epochs,
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
                    **(best_official_metrics or latest_official_metrics or {}),
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
    max_train_steps: int | None = None,
    seed: int | None = None,
    init_checkpoint: str | Path | None = None,
    use_amp: bool = False,
    log_every_steps: int | None = 100,
    augmentation_mode: Literal["off", "moderate", "strong"] = "off",
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
        set_global_seed(seed)
        amp_enabled = bool(use_amp) and resolved_device.type == "cuda"
        base_config = config if config is not None else ModelConfig(views=1)
        model_config = base_config.model_copy(update={"views": 1})
        print(f"[setup] loading OpenLane train split={train_split} from {dataroot}", flush=True)
        start_time = time.perf_counter()
        train_dataset = _subset_if_requested(
            OpenLaneDataset(
                dataroot=dataroot,
                split=train_split,
                subset=subset,
                lane_points=model_config.lane_points,
                augmentation_mode=augmentation_mode,
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
                augmentation_mode="off",
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
            seed=seed,
        )
        val_loader = _make_loader(
            val_dataset,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=None if seed is None else seed + 1,
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
                        "seed": seed,
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
                        "augmentation_mode": augmentation_mode,
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
        if init_checkpoint is not None:
            load_weights_into_model_from_checkpoint(
                model,
                init_checkpoint,
                map_location=resolved_device,
            )
            print(f"[setup] warm-started model from {init_checkpoint}", flush=True)
        criterion = MultitaskCriterion()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        history: list[dict[str, object]] = []
        checkpoint_path = artifact_root / "checkpoint_last.pt"
        train_steps_completed = 0
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
                f"max_train_steps={max_train_steps}",
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
                        "train_steps_completed": train_steps_completed
                        + (len(train_loader) if epoch_max_steps is None else epoch_max_steps),
                    },
                    step=epoch,
                )
            print(
                f"[epoch] completed epoch={epoch} train=({_format_metrics(train_metrics)}) "
                f"val=({_format_metrics(val_metrics)})",
                flush=True,
            )
            train_steps_completed += (
                len(train_loader) if epoch_max_steps is None else epoch_max_steps
            )
            if max_train_steps is not None and train_steps_completed >= max_train_steps:
                break

        result = {
            "device": resolved_device.type,
            "amp_enabled": amp_enabled,
            "seed": seed,
            "epochs": epochs,
            "train_steps": train_steps_completed,
            "artifact_dir": str(artifact_root),
            "checkpoint_path": str(checkpoint_path),
            "augmentation_mode": augmentation_mode,
            "max_train_steps": max_train_steps,
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


def fit_joint_public(
    nuscenes_root: str | Path,
    openlane_root: str | Path,
    artifact_dir: str | Path,
    *,
    config: ModelConfig | None = None,
    nuscenes_version: str = "v1.0-mini",
    nuscenes_train_split: str | None = None,
    nuscenes_val_split: str | None = None,
    openlane_subset: str = "lane3d_300",
    epochs: int = 36,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    grad_accum_steps: int = 2,
    num_workers: int = 4,
    batch_size: int = 1,
    device: str | None = None,
    max_nuscenes_train_samples: int | None = None,
    max_nuscenes_val_samples: int | None = None,
    max_openlane_train_samples: int | None = None,
    max_openlane_val_samples: int | None = None,
    seed: int | None = None,
    teacher_provider_config: TeacherProviderConfig | None = None,
    init_checkpoint: str | Path | None = None,
    use_amp: bool = False,
    log_every_steps: int | None = 100,
    optimizer_schedule: Literal["cosine", "constant"] = "constant",
    grad_clip_norm: float | None = 5.0,
    keep_best_checkpoint: bool = True,
    early_stop_patience: int | None = 6,
    early_stop_min_delta: float = 0.01,
    early_stop_min_epochs: int = 6,
    augmentation_mode: Literal["off", "moderate", "strong"] = "off",
    loss_mode: Literal["baseline", "focal_hardneg", "quality_focal"] = "quality_focal",
    hard_negative_ratio: int = 3,
    hard_negative_cap: int = 96,
    teacher_anchor_class_weight: float = 0.5,
    teacher_anchor_quality_class_weight: float = 0.45,
    teacher_anchor_objectness_weight: float = 0.5,
    teacher_region_objectness_weight: float = 0.12,
    teacher_region_class_weight: float = 0.12,
    teacher_region_radius_m: float = 4.0,
    enable_teacher_distillation: bool = True,
    lane_loss_scale: float = 0.05,
    lane_batch_multiplier: float = 1.0,
    official_eval_every_epochs: int | None = None,
    official_eval_score_threshold: float = 0.20,
    official_eval_top_k: int = 40,
    openlane_repo_root: str | Path = "/home/achbogga/projects/OpenLane",
    tracker: ExperimentTracker | None = None,
    tracking_metadata: TrackingMetadata | None = None,
) -> dict[str, object]:
    """Alternating public multitask training on nuScenes detection and OpenLane lane data."""

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
        set_global_seed(seed)
        amp_enabled = bool(use_amp) and resolved_device.type == "cuda"
        model_config = config if config is not None else ModelConfig()
        det_train_split, det_val_split = resolve_nuscenes_splits(
            version=nuscenes_version,
            train_split=nuscenes_train_split,
            val_split=nuscenes_val_split,
        )

        print(f"[setup] loading joint nuScenes train split={det_train_split}", flush=True)
        det_train = _subset_if_requested(
            NuScenesDataset(
                dataroot=nuscenes_root,
                version=nuscenes_version,
                split=det_train_split,
                augmentation_mode=augmentation_mode,
            ),
            max_nuscenes_train_samples,
        )
        det_train = maybe_attach_teacher_targets(det_train, teacher_provider_config)
        print(f"[setup] loading joint nuScenes val split={det_val_split}", flush=True)
        det_val = _subset_if_requested(
            NuScenesDataset(
                dataroot=nuscenes_root,
                version=nuscenes_version,
                split=det_val_split,
                augmentation_mode="off",
            ),
            max_nuscenes_val_samples,
        )
        det_val = maybe_attach_teacher_targets(det_val, teacher_provider_config)
        print(
            f"[setup] loading joint OpenLane train split=training subset={openlane_subset}",
            flush=True,
        )
        lane_train = _subset_if_requested(
            OpenLaneDataset(
                dataroot=openlane_root,
                split="training",
                subset=openlane_subset,
                lane_points=model_config.lane_points,
                augmentation_mode=augmentation_mode,
            ),
            max_openlane_train_samples,
        )
        print(
            f"[setup] loading joint OpenLane val split=validation subset={openlane_subset}",
            flush=True,
        )
        lane_val = _subset_if_requested(
            OpenLaneDataset(
                dataroot=openlane_root,
                split="validation",
                subset=openlane_subset,
                lane_points=model_config.lane_points,
                augmentation_mode="off",
            ),
            max_openlane_val_samples,
        )

        det_train_loader = _make_loader(
            det_train,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=seed,
        )
        det_val_loader = _make_loader(
            det_val,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=None if seed is None else seed + 1,
        )
        lane_train_loader = _make_loader(
            lane_train,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=None if seed is None else seed + 2,
        )
        lane_val_loader = _make_loader(
            lane_val,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
            seed=None if seed is None else seed + 3,
        )

        if len(det_train_loader) == 0 or len(lane_train_loader) == 0:
            raise RuntimeError(
                "joint training requires non-empty nuScenes and OpenLane train loaders"
            )

        if active_tracker is None:
            metadata = tracking_metadata or TrackingMetadata(
                suite="train",
                dataset="joint-public",
                job_type="train-joint-public",
                run_name=artifact_root.name,
                group=_default_tracking_group(artifact_root, "joint-public"),
                tags=("nuscenes", "openlane", model_config.image_backbone, openlane_subset),
                extra_config={
                    "nuscenes_train_split": det_train_split,
                    "nuscenes_val_split": det_val_split,
                    "openlane_subset": openlane_subset,
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
                        "seed": seed,
                        "init_checkpoint": (
                            str(init_checkpoint) if init_checkpoint is not None else None
                        ),
                        "optimizer_schedule": optimizer_schedule,
                        "grad_clip_norm": grad_clip_norm,
                        "keep_best_checkpoint": keep_best_checkpoint,
                        "early_stop_patience": early_stop_patience,
                        "early_stop_min_delta": early_stop_min_delta,
                        "early_stop_min_epochs": early_stop_min_epochs,
                        "augmentation_mode": augmentation_mode,
                        "loss_mode": loss_mode,
                        "hard_negative_ratio": hard_negative_ratio,
                        "hard_negative_cap": hard_negative_cap,
                        "teacher_anchor_class_weight": teacher_anchor_class_weight,
                        "teacher_anchor_quality_class_weight": teacher_anchor_quality_class_weight,
                        "teacher_anchor_objectness_weight": teacher_anchor_objectness_weight,
                        "teacher_region_objectness_weight": teacher_region_objectness_weight,
                        "teacher_region_class_weight": teacher_region_class_weight,
                        "teacher_region_radius_m": teacher_region_radius_m,
                        "enable_teacher_distillation": enable_teacher_distillation,
                        "lane_loss_scale": lane_loss_scale,
                        "lane_batch_multiplier": lane_batch_multiplier,
                        "official_eval_every_epochs": official_eval_every_epochs,
                        "official_eval_score_threshold": official_eval_score_threshold,
                        "official_eval_top_k": official_eval_top_k,
                    },
                },
            )

        print(
            f"[setup] building joint model backbone={model_config.image_backbone} "
            f"pretrained_backbone={model_config.pretrained_image_backbone} "
            f"freeze_backbone={model_config.freeze_image_backbone}",
            flush=True,
        )
        model = TSQBEVModel(model_config).to(resolved_device)
        if init_checkpoint is not None:
            load_weights_into_model_from_checkpoint(
                model,
                init_checkpoint,
                map_location=resolved_device,
            )
            print(f"[setup] warm-started joint model from {init_checkpoint}", flush=True)
        criterion = MultitaskCriterion(
            detection=_make_detection_criterion(
                loss_mode=loss_mode,
                hard_negative_ratio=hard_negative_ratio,
                hard_negative_cap=hard_negative_cap,
                teacher_anchor_class_weight=teacher_anchor_class_weight,
                teacher_anchor_quality_class_weight=teacher_anchor_quality_class_weight,
                teacher_anchor_objectness_weight=teacher_anchor_objectness_weight,
                teacher_region_objectness_weight=teacher_region_objectness_weight,
                teacher_region_class_weight=teacher_region_class_weight,
                teacher_region_radius_m=teacher_region_radius_m,
            ),
            enable_distillation=enable_teacher_distillation,
        )
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = _make_scheduler(
            optimizer,
            epochs=epochs,
            optimizer_schedule=optimizer_schedule,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        checkpoint_last_path = artifact_root / "checkpoint_last.pt"
        checkpoint_best_path = artifact_root / "checkpoint_best.pt"
        history: list[dict[str, object]] = []
        best_epoch = 0
        best_detection_total = float("inf")
        best_lane_total = float("inf")
        best_official_epoch = 0
        best_official_metrics: dict[str, float] | None = None
        epochs_without_improvement = 0
        early_stop_triggered = False
        early_stop_reason: str | None = None
        latest_official_metrics: dict[str, float] | None = None
        checkpoint_best_official_path = artifact_root / "checkpoint_best_official.pt"

        for epoch in range(1, epochs + 1):
            lane_batches_per_epoch = _lane_batches_per_epoch(
                detection_batches=len(det_train_loader),
                lane_batches=len(lane_train_loader),
                lane_batch_multiplier=lane_batch_multiplier,
            )
            print(
                f"[joint-train] epoch={epoch}/{epochs} device={resolved_device.type} "
                f"detection_batches={len(det_train_loader)} "
                f"lane_batches_full={len(lane_train_loader)} "
                f"lane_batches_epoch={lane_batches_per_epoch} "
                f"lane_batch_multiplier={lane_batch_multiplier:.3f} "
                f"lane_loss_scale={lane_loss_scale:.3f}",
                flush=True,
            )
            det_train_metrics, lane_train_metrics, train_aux = _train_joint_epoch(
                model=model,
                detection_loader=det_train_loader,
                lane_loader=lane_train_loader,
                criterion=criterion,
                optimizer=optimizer,
                grad_accum_steps=grad_accum_steps,
                device=resolved_device,
                amp_enabled=amp_enabled,
                scaler=scaler,
                epoch=epoch,
                log_every_steps=log_every_steps,
                lane_loss_scale=lane_loss_scale,
                lane_batch_multiplier=lane_batch_multiplier,
                grad_clip_norm=grad_clip_norm,
            )
            det_val_metrics, lane_val_metrics, selection = _eval_joint_epoch(
                model=model,
                detection_loader=det_val_loader,
                lane_loader=lane_val_loader,
                criterion=criterion,
                device=resolved_device,
                amp_enabled=amp_enabled,
                epoch=epoch,
                lane_loss_scale=lane_loss_scale,
            )
            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "train_detection": det_train_metrics,
                    "train_lane": lane_train_metrics,
                    "train_aux": train_aux,
                    "val_detection": det_val_metrics,
                    "val_lane": lane_val_metrics,
                    "selection": selection,
                }
            )
            _write_history(artifact_root, history)
            save_model_checkpoint(
                model,
                model_config,
                checkpoint_last_path,
                epoch=epoch,
                history=history,
            )

            current_det_total = float(det_val_metrics["total"])
            current_lane_total = float(lane_val_metrics["total"])
            improved = (
                current_det_total < (best_detection_total - early_stop_min_delta)
                or (
                    abs(current_det_total - best_detection_total) <= early_stop_min_delta
                    and current_lane_total < best_lane_total
                )
            )
            if improved:
                best_detection_total = current_det_total
                best_lane_total = current_lane_total
                best_epoch = epoch
                epochs_without_improvement = 0
                save_model_checkpoint(
                    model,
                    model_config,
                    checkpoint_best_path,
                    epoch=epoch,
                    history=history,
                )
            else:
                epochs_without_improvement += 1

            if active_tracker is not None:
                active_tracker.log(
                    {
                        "epoch": epoch,
                        **_prefixed_metrics("train_detection", det_train_metrics),
                        **_prefixed_metrics("train_lane", lane_train_metrics),
                        **_prefixed_metrics("val_detection", det_val_metrics),
                        **_prefixed_metrics("val_lane", lane_val_metrics),
                        **_prefixed_metrics("selection", selection),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "best_epoch": best_epoch,
                        "best_detection_total": best_detection_total,
                        "best_lane_total": best_lane_total,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                    step=epoch,
                )

            should_run_official_eval = (
                official_eval_every_epochs is not None
                and official_eval_every_epochs > 0
                and epoch % official_eval_every_epochs == 0
            )
            if should_run_official_eval:
                selected_checkpoint_path = (
                    checkpoint_best_path
                    if keep_best_checkpoint and checkpoint_best_path.exists()
                    else checkpoint_last_path
                )
                latest_official_metrics = _run_joint_public_official_eval(
                    checkpoint_path=selected_checkpoint_path,
                    model_config=model_config,
                    device=device,
                    artifact_root=artifact_root,
                    epoch=epoch,
                    nuscenes_root=nuscenes_root,
                    nuscenes_version=nuscenes_version,
                    nuscenes_split=det_val_split,
                    openlane_root=openlane_root,
                    openlane_subset=openlane_subset,
                    openlane_repo_root=openlane_repo_root,
                    score_threshold=official_eval_score_threshold,
                    top_k=official_eval_top_k,
                )
                history[-1]["official_eval"] = latest_official_metrics
                _write_history(artifact_root, history)
                if _joint_official_metrics_better(latest_official_metrics, best_official_metrics):
                    best_official_metrics = dict(latest_official_metrics)
                    best_official_epoch = epoch
                    save_model_checkpoint(
                        model,
                        model_config,
                        checkpoint_best_official_path,
                        epoch=epoch,
                        history=history,
                    )
                if active_tracker is not None:
                    active_tracker.log(
                        {"epoch": epoch, **latest_official_metrics},
                        step=epoch,
                    )
                print(
                    "[joint-official-eval] "
                    f"epoch={epoch} "
                    f"nuscenes_nds={latest_official_metrics['nuscenes_nds']:.4f} "
                    f"nuscenes_map={latest_official_metrics['nuscenes_map']:.4f} "
                    f"sanity_ok={latest_official_metrics['nuscenes_export_sanity_ok']:.0f} "
                    f"openlane_f={latest_official_metrics['openlane_f_score']:.4f} "
                    f"openlane_recall={latest_official_metrics['openlane_recall']:.4f} "
                    f"openlane_precision={latest_official_metrics['openlane_precision']:.4f}",
                    flush=True,
                )
                if _catastrophic_nuscenes_official_failure(latest_official_metrics):
                    early_stop_triggered = True
                    early_stop_reason = (
                        "catastrophic joint official-eval failure: "
                        f"epoch={epoch} nds={latest_official_metrics['nuscenes_nds']:.4f} "
                        f"map={latest_official_metrics['nuscenes_map']:.4f} "
                        f"sanity_ok={latest_official_metrics['nuscenes_export_sanity_ok']:.0f} "
                        f"max_box_size_m={latest_official_metrics['nuscenes_max_box_size_m']:.2f} "
                        f"score_mean={latest_official_metrics['nuscenes_score_mean']:.4f}"
                    )
                    print(f"[early-stop] {early_stop_reason}", flush=True)
                    break

            print(
                f"[joint-epoch] completed epoch={epoch} "
                f"train_det=({_format_metrics(det_train_metrics)}) "
                f"train_lane=({_format_metrics(lane_train_metrics)}) "
                f"val_det=({_format_metrics(det_val_metrics)}) "
                f"val_lane=({_format_metrics(lane_val_metrics)})",
                flush=True,
            )
            if (
                early_stop_patience is not None
                and epoch >= early_stop_min_epochs
                and epochs_without_improvement >= early_stop_patience
            ):
                early_stop_triggered = True
                early_stop_reason = (
                    "joint plateau: "
                    f"best_detection_total={best_detection_total:.4f} at epoch={best_epoch}, "
                    f"no detection improvement > {early_stop_min_delta:.4f} for "
                    f"{epochs_without_improvement} epoch(s)"
                )
                print(f"[early-stop] {early_stop_reason}", flush=True)
                break

        if checkpoint_best_official_path.exists():
            selected_checkpoint_path = checkpoint_best_official_path
        else:
            selected_checkpoint_path = (
                checkpoint_best_path
                if keep_best_checkpoint and checkpoint_best_path.exists()
                else checkpoint_last_path
            )
        result = {
            "device": resolved_device.type,
            "amp_enabled": amp_enabled,
            "epochs": len(history),
            "artifact_dir": str(artifact_root),
            "checkpoint_path": str(selected_checkpoint_path),
            "selected_checkpoint_path": str(selected_checkpoint_path),
            "last_checkpoint_path": str(checkpoint_last_path),
            "best_checkpoint_path": str(checkpoint_best_path),
            "best_official_checkpoint_path": (
                str(checkpoint_best_official_path)
                if checkpoint_best_official_path.exists()
                else None
            ),
            "best_epoch": best_epoch,
            "best_detection_total": best_detection_total,
            "best_lane_total": best_lane_total,
            "best_official_epoch": best_official_epoch,
            "best_official_metrics": best_official_metrics,
            "early_stop_triggered": early_stop_triggered,
            "early_stop_reason": early_stop_reason,
            "lane_loss_scale": lane_loss_scale,
            "lane_batch_multiplier": lane_batch_multiplier,
            "official_eval_every_epochs": official_eval_every_epochs,
            "latest_official_eval": latest_official_metrics,
            "last": history[-1],
            "history": history,
        }
        if active_tracker is not None:
            active_tracker.summary(
                {
                    "artifact_dir": str(artifact_root),
                    "checkpoint_path": str(selected_checkpoint_path),
                    "epochs": len(history),
                    "best_epoch": best_epoch,
                    "best_detection_total": best_detection_total,
                    "best_lane_total": best_lane_total,
                    "best_official_epoch": best_official_epoch,
                    "lane_loss_scale": lane_loss_scale,
                    "lane_batch_multiplier": lane_batch_multiplier,
                    "official_eval_every_epochs": official_eval_every_epochs,
                    **(best_official_metrics or latest_official_metrics or {}),
                }
            )
        status = "completed"
        return result
    finally:
        if owns_tracker and active_tracker is not None:
            active_tracker.finish(status=status)
