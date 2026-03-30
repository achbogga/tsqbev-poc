"""Manual runtime harnesses for training, evaluation, and benchmarking.

References:
- PETRv2 multitask training framing:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf
- BEVDistill teacher-guided optimization:
  https://arxiv.org/abs/2211.09386
- HotBEV deployment-minded measurement discipline:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf
"""

from __future__ import annotations

import time
from typing import cast

import torch

from tsqbev.config import ModelConfig
from tsqbev.contracts import QuerySeedBank, SceneBatch, TemporalState
from tsqbev.distill import DistillationObjective
from tsqbev.model import TSQBEVModel
from tsqbev.synthetic import make_synthetic_batch

Tensor = torch.Tensor


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve an execution device for manual runtime harnesses."""

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch: SceneBatch, device: torch.device) -> SceneBatch:
    """Move a scene batch and its optional targets to a device."""

    camera_proposals = batch.camera_proposals
    if camera_proposals is not None:
        camera_proposals = type(camera_proposals)(
            boxes_xyxy=camera_proposals.boxes_xyxy.to(device),
            scores=camera_proposals.scores.to(device),
        )

    od_targets = batch.od_targets
    if od_targets is not None:
        od_targets = type(od_targets)(
            boxes_3d=od_targets.boxes_3d.to(device),
            labels=od_targets.labels.to(device),
            valid_mask=od_targets.valid_mask.to(device),
        )

    lane_targets = batch.lane_targets
    if lane_targets is not None:
        lane_targets = type(lane_targets)(
            polylines=lane_targets.polylines.to(device),
            valid_mask=lane_targets.valid_mask.to(device),
        )

    map_priors = batch.map_priors
    if map_priors is not None:
        map_priors = type(map_priors)(
            tokens=map_priors.tokens.to(device),
            coords_xy=map_priors.coords_xy.to(device),
            valid_mask=map_priors.valid_mask.to(device),
        )

    teacher_targets = batch.teacher_targets
    if teacher_targets is not None:
        teacher_targets = type(teacher_targets)(
            object_features=teacher_targets.object_features.to(device)
            if teacher_targets.object_features is not None
            else None,
            object_boxes=teacher_targets.object_boxes.to(device)
            if teacher_targets.object_boxes is not None
            else None,
            object_labels=teacher_targets.object_labels.to(device)
            if teacher_targets.object_labels is not None
            else None,
            object_scores=teacher_targets.object_scores.to(device)
            if teacher_targets.object_scores is not None
            else None,
            lane_features=teacher_targets.lane_features.to(device)
            if teacher_targets.lane_features is not None
            else None,
            router_logits=teacher_targets.router_logits.to(device)
            if teacher_targets.router_logits is not None
            else None,
            valid_mask=teacher_targets.valid_mask.to(device)
            if teacher_targets.valid_mask is not None
            else None,
        )

    moved = SceneBatch(
        images=batch.images.to(device),
        lidar_points=batch.lidar_points.to(device),
        lidar_mask=batch.lidar_mask.to(device),
        intrinsics=batch.intrinsics.to(device),
        extrinsics=batch.extrinsics.to(device),
        ego_pose=batch.ego_pose.to(device),
        time_delta_s=batch.time_delta_s.to(device),
        camera_proposals=camera_proposals,
        od_targets=od_targets,
        lane_targets=lane_targets,
        map_priors=map_priors,
        teacher_targets=teacher_targets,
    )
    moved.validate()
    return moved


def _match_count(pred: Tensor, target: Tensor) -> int:
    return min(pred.shape[1], target.shape[1])


def compute_training_losses(
    outputs: dict[str, object],
    batch: SceneBatch,
    distillation: DistillationObjective | None = None,
) -> dict[str, Tensor]:
    """Compute compact synthetic training losses for the manual harness."""

    object_logits = outputs["object_logits"]
    object_boxes = outputs["object_boxes"]
    lane_logits = outputs["lane_logits"]
    lane_polylines = outputs["lane_polylines"]
    temporal_state = cast(TemporalState, outputs["temporal_state"])
    seed_bank = cast(QuerySeedBank, outputs["seed_bank"])

    assert isinstance(object_logits, torch.Tensor)
    assert isinstance(object_boxes, torch.Tensor)
    assert isinstance(lane_logits, torch.Tensor)
    assert isinstance(lane_polylines, torch.Tensor)

    losses: dict[str, Tensor] = {}
    total = object_logits.new_tensor(0.0)

    if batch.od_targets is not None:
        valid_mask = batch.od_targets.valid_mask
        object_count = min(
            object_boxes.shape[1],
            int(valid_mask.sum(dim=1).min().item()),
        )
        if object_count > 0:
            cls_loss = torch.nn.functional.cross_entropy(
                object_logits[:, :object_count].reshape(-1, object_logits.shape[-1]),
                batch.od_targets.labels[:, :object_count].reshape(-1),
            )
            box_loss = torch.nn.functional.smooth_l1_loss(
                object_boxes[:, :object_count],
                batch.od_targets.boxes_3d[:, :object_count],
            )
        else:
            cls_loss = object_logits.sum() * 0.0
            box_loss = object_boxes.sum() * 0.0
        losses["object_cls"] = cls_loss
        losses["object_box"] = box_loss
        total = total + cls_loss + box_loss

    if batch.lane_targets is not None:
        lane_logits_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            lane_logits,
            batch.lane_targets.valid_mask.float(),
        )
        lane_shape_loss = torch.nn.functional.smooth_l1_loss(
            lane_polylines,
            batch.lane_targets.polylines,
        )
        losses["lane_logits"] = lane_logits_loss
        losses["lane_shape"] = lane_shape_loss
        total = total + lane_logits_loss + lane_shape_loss

    if distillation is None:
        distillation = DistillationObjective()
    kd_losses = distillation(
        object_logits=object_logits,
        object_queries=temporal_state.object_queries,
        object_boxes=object_boxes,
        seed_bank=seed_bank,
        teacher=batch.teacher_targets,
    )
    losses.update(kd_losses)
    total = total + kd_losses["kd_total"]
    losses["total"] = total
    return losses


def run_train_step(
    config: ModelConfig,
    batch_size: int = 2,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, float | str]:
    """Run one manual synthetic train step and return scalar losses."""

    resolved_device = resolve_device(device)
    batch = move_batch(make_synthetic_batch(config, batch_size=batch_size), resolved_device)
    model = TSQBEVModel(config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    outputs = model(batch)
    losses = compute_training_losses(outputs, batch)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    optimizer.step()

    return {
        "device": resolved_device.type,
        "total": float(losses["total"].detach().cpu()),
        "object_cls": float(losses.get("object_cls", losses["total"]).detach().cpu()),
        "object_box": float(losses.get("object_box", losses["total"]).detach().cpu()),
        "lane_logits": float(losses.get("lane_logits", losses["total"]).detach().cpu()),
        "lane_shape": float(losses.get("lane_shape", losses["total"]).detach().cpu()),
        "kd_total": float(losses["kd_total"].detach().cpu()),
    }


def run_eval_step(
    config: ModelConfig,
    batch_size: int = 2,
    device: str | None = None,
) -> dict[str, float | tuple[int, ...]]:
    """Run one manual synthetic eval step and return summary metrics."""

    resolved_device = resolve_device(device)
    batch = move_batch(make_synthetic_batch(config, batch_size=batch_size), resolved_device)
    model = TSQBEVModel(config).to(resolved_device)
    model.eval()

    with torch.no_grad():
        outputs = model(batch)
        losses = compute_training_losses(outputs, batch)

    object_logits = outputs["object_logits"]
    lane_polylines = outputs["lane_polylines"]
    assert isinstance(object_logits, torch.Tensor)
    assert isinstance(lane_polylines, torch.Tensor)

    return {
        "total": float(losses["total"].detach().cpu()),
        "object_logits_shape": tuple(object_logits.shape),
        "lane_polylines_shape": tuple(lane_polylines.shape),
    }


def benchmark_forward(
    config: ModelConfig,
    steps: int = 5,
    warmup: int = 2,
    batch_size: int = 1,
    device: str | None = None,
    image_height: int = 96,
    image_width: int = 160,
) -> dict[str, float | str]:
    """Measure synthetic forward latency on CPU or CUDA."""

    resolved_device = resolve_device(device)
    batch = move_batch(
        make_synthetic_batch(
            config,
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        ),
        resolved_device,
    )
    model = TSQBEVModel(config).to(resolved_device)
    model.eval()

    for _ in range(warmup):
        with torch.no_grad():
            model(batch)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)

    elapsed_ms: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        with torch.no_grad():
            model(batch)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    samples = sorted(elapsed_ms)
    p50_index = len(samples) // 2
    p95_index = min(len(samples) - 1, max(0, int(len(samples) * 0.95) - 1))
    return {
        "device": resolved_device.type,
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": samples[p50_index],
        "p95_ms": samples[p95_index],
        "steps": float(steps),
    }
