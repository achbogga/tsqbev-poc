"""Distillation objectives for the minimal proof of concept.

References:
- BEVDistill:
  https://arxiv.org/abs/2211.09386
- StreamPETR temporal consistency motivation:
  https://arxiv.org/abs/2303.11926
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tsqbev.contracts import QuerySeedBank, TeacherTargets

Tensor = torch.Tensor


def _zero_with_grad(tensor: Tensor) -> Tensor:
    return tensor.sum() * 0.0


def _masked_mean(tensor: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return tensor.mean()
    weight = mask.float()
    while weight.ndim < tensor.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(tensor)
    return (tensor * weight).sum() / weight.sum().clamp_min(1.0)


def _teacher_selection_indices(
    teacher: TeacherTargets,
    target_count: int,
) -> tuple[Tensor, Tensor]:
    scores = teacher.object_scores
    if scores is None:
        raise ValueError("teacher object_scores are required for top-k alignment")
    mask = teacher.valid_mask
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if mask.shape != scores.shape:
        raise ValueError("teacher valid_mask/object_scores shape mismatch")

    batch_size, teacher_count = scores.shape
    indices = torch.zeros(batch_size, target_count, dtype=torch.long, device=scores.device)
    keep_mask = torch.zeros(batch_size, target_count, dtype=torch.bool, device=scores.device)
    position_bias = torch.linspace(
        0.0,
        -1e-6 * max(teacher_count - 1, 0),
        steps=max(teacher_count, 1),
        device=scores.device,
        dtype=scores.dtype,
    )
    for batch_index in range(batch_size):
        valid = mask[batch_index]
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            continue
        keep = min(target_count, valid_count)
        ranking = scores[batch_index] + position_bias
        ranking = ranking.masked_fill(~valid, float("-inf"))
        top_indices = torch.topk(ranking, k=keep, dim=0).indices
        indices[batch_index, :keep] = top_indices
        keep_mask[batch_index, :keep] = True
    return indices, keep_mask


def _gather_teacher_rows(
    tensor: Tensor | None,
    indices: Tensor,
    keep_mask: Tensor,
) -> Tensor | None:
    if tensor is None:
        return None
    if int(indices.max().item()) >= tensor.shape[1]:
        return None
    aligned = torch.gather(
        tensor,
        dim=1,
        index=indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1]),
    )
    aligned = aligned * keep_mask.unsqueeze(-1).to(dtype=aligned.dtype)
    return aligned


def _gather_teacher_vector(
    tensor: Tensor | None,
    indices: Tensor,
    keep_mask: Tensor,
) -> Tensor | None:
    if tensor is None:
        return None
    if int(indices.max().item()) >= tensor.shape[1]:
        return None
    aligned = torch.gather(tensor, dim=1, index=indices)
    fill_value = 0 if tensor.dtype in (torch.int32, torch.int64, torch.long) else 0.0
    return aligned.masked_fill(~keep_mask, fill_value)


def _align_teacher_targets(
    object_queries: Tensor,
    object_boxes: Tensor,
    teacher: TeacherTargets,
) -> tuple[TeacherTargets, Tensor]:
    target_count = int(object_boxes.shape[1])
    indices, keep_mask = _teacher_selection_indices(teacher, target_count)
    aligned = TeacherTargets(
        object_features=_gather_teacher_rows(teacher.object_features, indices, keep_mask),
        object_boxes=_gather_teacher_rows(teacher.object_boxes, indices, keep_mask),
        object_labels=_gather_teacher_vector(teacher.object_labels, indices, keep_mask),
        object_scores=_gather_teacher_vector(teacher.object_scores, indices, keep_mask),
        lane_features=teacher.lane_features,
        router_logits=teacher.router_logits,
        valid_mask=keep_mask,
    )
    if (
        aligned.object_features is not None
        and aligned.object_features.shape[:2] != object_queries.shape[:2]
    ):
        aligned.object_features = None
    if (
        aligned.object_boxes is not None
        and aligned.object_boxes.shape[:2] != object_boxes.shape[:2]
    ):
        aligned.object_boxes = None
    if aligned.router_logits is not None and aligned.router_logits.shape != keep_mask.shape:
        aligned.router_logits = None
    return aligned, keep_mask


class DistillationObjective(nn.Module):
    """Compute optional distillation losses from teacher cache tensors."""

    def __init__(
        self,
        feature_weight: float = 1.0,
        box_weight: float = 1.0,
        router_weight: float = 0.5,
        class_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_weight = feature_weight
        self.box_weight = box_weight
        self.router_weight = router_weight
        self.class_weight = class_weight

    def forward(
        self,
        object_logits: Tensor,
        object_queries: Tensor,
        object_boxes: Tensor,
        seed_bank: QuerySeedBank,
        teacher: TeacherTargets | None,
    ) -> dict[str, Tensor]:
        zero = _zero_with_grad(object_queries)
        if teacher is None:
            return {
                "kd_total": zero,
                "kd_features": zero,
                "kd_boxes": zero,
                "kd_router": zero,
                "kd_logits": zero,
            }

        aligned_teacher, mask = _align_teacher_targets(object_queries, object_boxes, teacher)
        feature_loss = zero
        if aligned_teacher.object_features is not None:
            feature_loss = (
                _masked_mean((object_queries - aligned_teacher.object_features) ** 2, mask)
                * self.feature_weight
            )

        box_loss = zero
        if aligned_teacher.object_boxes is not None:
            box_loss = (
                _masked_mean(
                    F.smooth_l1_loss(object_boxes, aligned_teacher.object_boxes, reduction="none"),
                    mask.unsqueeze(-1),
                )
                * self.box_weight
            )

        logits_loss = zero
        if (
            aligned_teacher.object_labels is not None
            and aligned_teacher.object_scores is not None
            and mask.any()
        ):
            target_classes = torch.zeros_like(object_logits)
            score_weights = torch.zeros_like(aligned_teacher.object_scores)
            for batch_index in range(object_logits.shape[0]):
                valid = mask[batch_index]
                if not valid.any():
                    continue
                labels = aligned_teacher.object_labels[batch_index][valid]
                scores = aligned_teacher.object_scores[batch_index][valid].clamp(0.0, 1.0)
                query_indices = torch.arange(
                    labels.shape[0], device=labels.device, dtype=torch.long
                )
                target_classes[batch_index, query_indices, labels] = 1.0
                score_weights[batch_index, query_indices] = scores
            logits_loss = (
                _masked_mean(
                    F.binary_cross_entropy_with_logits(
                        object_logits,
                        target_classes,
                        reduction="none",
                    ),
                    score_weights.unsqueeze(-1),
                )
                * self.class_weight
            )

        router_loss = zero
        if aligned_teacher.router_logits is not None:
            router_loss = (
                F.mse_loss(seed_bank.keep_logits, aligned_teacher.router_logits)
                * self.router_weight
            )

        total = feature_loss + box_loss + router_loss + logits_loss
        return {
            "kd_total": total,
            "kd_features": feature_loss,
            "kd_boxes": box_loss,
            "kd_router": router_loss,
            "kd_logits": logits_loss,
        }
