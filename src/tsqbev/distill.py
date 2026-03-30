"""Distillation objectives for the minimal proof of concept.

References:
- BEVDistill:
  https://arxiv.org/abs/2211.09386
- StreamPETR temporal consistency motivation:
  https://arxiv.org/abs/2303.11926
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsqbev.contracts import QuerySeedBank, TeacherTargets

Tensor = torch.Tensor

try:  # pragma: no cover - SciPy is exercised in real data runs.
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


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


def _greedy_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    remaining_rows = set(range(cost.shape[0]))
    remaining_cols = set(range(cost.shape[1]))
    while remaining_rows and remaining_cols:
        best_row = -1
        best_col = -1
        best_cost = float("inf")
        for row in remaining_rows:
            for col in remaining_cols:
                current = float(cost[row, col])
                if current < best_cost:
                    best_row = row
                    best_col = col
                    best_cost = current
        rows.append(best_row)
        cols.append(best_col)
        remaining_rows.remove(best_row)
        remaining_cols.remove(best_col)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _linear_sum_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    cost_np = cost.detach().cpu().numpy()
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_np)
    else:  # pragma: no cover - only used when SciPy is absent.
        row_ind, col_ind = _greedy_assignment(cost_np)
    device = cost.device
    return (
        torch.from_numpy(np.asarray(row_ind, dtype=np.int64)).to(device=device),
        torch.from_numpy(np.asarray(col_ind, dtype=np.int64)).to(device=device),
    )


def _teacher_selection_indices(
    query_refs: Tensor,
    object_logits: Tensor,
    teacher: TeacherTargets,
) -> tuple[Tensor, Tensor]:
    scores = teacher.object_scores
    boxes = teacher.object_boxes
    if scores is None or boxes is None:
        raise ValueError("teacher object_boxes/object_scores are required for geometry alignment")
    mask = teacher.valid_mask
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if mask.shape != scores.shape:
        raise ValueError("teacher valid_mask/object_scores shape mismatch")

    batch_size, target_count = query_refs.shape[:2]
    indices = torch.zeros(batch_size, target_count, dtype=torch.long, device=query_refs.device)
    keep_mask = torch.zeros(batch_size, target_count, dtype=torch.bool, device=query_refs.device)

    for batch_index in range(batch_size):
        valid = mask[batch_index]
        if not bool(valid.any()):
            continue

        teacher_positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        teacher_boxes = boxes[batch_index][valid]
        teacher_scores = scores[batch_index][valid].clamp(0.0, 1.0)
        geo_cost = torch.cdist(query_refs[batch_index], teacher_boxes[:, :3], p=1)
        total_cost = geo_cost + 0.1 * (1.0 - teacher_scores).unsqueeze(0)

        if teacher.object_labels is not None:
            class_probs = object_logits[batch_index].sigmoid()
            teacher_labels = teacher.object_labels[batch_index][valid]
            class_cost = 1.0 - class_probs[:, teacher_labels]
            total_cost = total_cost + 0.5 * class_cost

        pred_index, teacher_index = _linear_sum_assignment(total_cost)
        if pred_index.numel() == 0:
            continue
        indices[batch_index, pred_index] = teacher_positions[teacher_index]
        keep_mask[batch_index, pred_index] = True
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
    query_refs: Tensor,
    object_logits: Tensor,
    object_queries: Tensor,
    object_boxes: Tensor,
    teacher: TeacherTargets,
) -> tuple[TeacherTargets, Tensor]:
    target_count = int(object_boxes.shape[1])
    if target_count == 0:
        empty_mask = torch.zeros(
            object_boxes.shape[:2], dtype=torch.bool, device=object_boxes.device
        )
        return teacher, empty_mask
    indices, keep_mask = _teacher_selection_indices(query_refs, object_logits, teacher)
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

        aligned_teacher, mask = _align_teacher_targets(
            seed_bank.refs_xyz,
            object_logits,
            object_queries,
            object_boxes,
            teacher,
        )
        if (
            aligned_teacher.object_features is not None
            and aligned_teacher.object_features.shape != object_queries.shape
        ):
            aligned_teacher.object_features = None
        if (
            aligned_teacher.object_boxes is not None
            and aligned_teacher.object_boxes.shape != object_boxes.shape
        ):
            aligned_teacher.object_boxes = None
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
                query_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
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
