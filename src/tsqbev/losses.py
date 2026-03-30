"""Set-based training losses for real dataset runs.

References:
- DETR-style bipartite matching objective:
  https://arxiv.org/abs/2005.12872
- PETRv2 temporal sparse-query supervision:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf
- BEVDistill teacher-guided losses:
  https://arxiv.org/abs/2211.09386
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsqbev.contracts import QuerySeedBank, SceneBatch, TemporalState
from tsqbev.distill import DistillationObjective

try:  # pragma: no cover - SciPy is exercised in real data runs.
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

Tensor = torch.Tensor


def _zero_with_grad(tensor: Tensor) -> Tensor:
    return tensor.sum() * 0.0


def _sigmoid_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Elementwise sigmoid focal loss.

    Reference:
    - RetinaNet focal loss:
      https://arxiv.org/abs/1708.02002
    """

    probs = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * ((1.0 - pt).clamp_min(1e-6) ** gamma) * ce


def _angle_difference(pred_yaw: Tensor, target_yaw: Tensor) -> Tensor:
    return torch.atan2(
        torch.sin(pred_yaw.unsqueeze(-1) - target_yaw.unsqueeze(0)),
        torch.cos(pred_yaw.unsqueeze(-1) - target_yaw.unsqueeze(0)),
    ).abs()


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


class DetectionSetCriterion(nn.Module):
    """DETR-style sparse detection matching for object queries."""

    def __init__(
        self,
        class_weight: float = 2.0,
        center_weight: float = 2.0,
        size_weight: float = 1.0,
        yaw_weight: float = 0.5,
        velocity_weight: float = 0.5,
        reference_weight: float = 0.25,
        loss_mode: str = "baseline",
        hard_negative_ratio: int = 3,
        hard_negative_cap: int = 96,
    ) -> None:
        super().__init__()
        self.class_weight = class_weight
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.yaw_weight = yaw_weight
        self.velocity_weight = velocity_weight
        self.reference_weight = reference_weight
        self.loss_mode = loss_mode
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_negative_cap = hard_negative_cap

    def _select_query_mask(
        self,
        class_logits: Tensor,
        objectness_logits: Tensor | None,
        matched_mask: Tensor,
    ) -> Tensor:
        if self.loss_mode != "focal_hardneg":
            return torch.ones_like(matched_mask, dtype=torch.bool)

        positive_count = int(matched_mask.sum().item())
        negative_budget = min(
            self.hard_negative_cap,
            max(32, self.hard_negative_ratio * positive_count),
        )
        query_mask = matched_mask.clone()
        unmatched_mask = ~matched_mask
        unmatched_count = int(unmatched_mask.sum().item())
        if unmatched_count == 0 or negative_budget <= 0:
            return query_mask

        class_hardness = class_logits.sigmoid().max(dim=-1).values
        if objectness_logits is None:
            hardness = class_hardness
        else:
            hardness = torch.maximum(class_hardness, objectness_logits.sigmoid())

        unmatched_indices = unmatched_mask.nonzero(as_tuple=False).flatten()
        keep_count = min(negative_budget, unmatched_count)
        if keep_count <= 0:
            return query_mask
        hardest = torch.topk(hardness[unmatched_indices], k=keep_count, sorted=False).indices
        query_mask[unmatched_indices[hardest]] = True
        return query_mask

    def _classification_loss(
        self,
        object_logits: Tensor,
        target_classes: Tensor,
        query_mask: Tensor,
        normalizer: float,
    ) -> Tensor:
        if self.loss_mode == "focal_hardneg":
            loss = _sigmoid_focal_loss_with_logits(object_logits, target_classes)
        else:
            loss = F.binary_cross_entropy_with_logits(
                object_logits,
                target_classes,
                reduction="none",
            )
        masked = loss * query_mask.unsqueeze(-1).float()
        return masked.sum() / normalizer

    def _objectness_loss(
        self,
        objectness_logits: Tensor,
        target_objectness: Tensor,
        query_mask: Tensor,
        normalizer: float,
    ) -> Tensor:
        if self.loss_mode == "focal_hardneg":
            loss = _sigmoid_focal_loss_with_logits(objectness_logits, target_objectness)
        else:
            loss = F.binary_cross_entropy_with_logits(
                objectness_logits,
                target_objectness,
                reduction="none",
            )
        return (loss * query_mask.float()).sum() / normalizer

    def forward(
        self,
        object_logits: Tensor,
        object_boxes: Tensor,
        batch: SceneBatch,
        objectness_logits: Tensor | None = None,
        reference_points: Tensor | None = None,
    ) -> dict[str, Tensor]:
        object_logits = object_logits.float()
        object_boxes = object_boxes.float()
        zero = _zero_with_grad(object_logits)
        if batch.od_targets is None:
            return {
                "objectness": zero,
                "object_cls": zero,
                "object_box": zero,
                "object_ref": zero,
            }

        batch_size, queries, _classes = object_logits.shape
        target_classes = torch.zeros_like(object_logits)
        target_objectness = (
            torch.zeros_like(objectness_logits.float())
            if objectness_logits is not None
            else None
        )
        query_selection = torch.ones(
            batch_size,
            queries,
            dtype=torch.bool,
            device=object_logits.device,
        )
        total_box = zero
        total_gt = 0

        for batch_index in range(batch_size):
            valid = batch.od_targets.valid_mask[batch_index]
            target_boxes = batch.od_targets.boxes_3d[batch_index][valid]
            target_labels = batch.od_targets.labels[batch_index][valid]
            if target_boxes.numel() == 0:
                continue

            class_probs = object_logits[batch_index].sigmoid()
            if objectness_logits is not None:
                class_probs = (
                    class_probs * objectness_logits[batch_index].float().sigmoid().unsqueeze(-1)
                )
            class_cost = 1.0 - class_probs[:, target_labels]
            center_cost = torch.cdist(
                object_boxes[batch_index, :, :3], target_boxes[:, :3], p=1
            )
            size_cost = torch.cdist(
                object_boxes[batch_index, :, 3:6].abs(), target_boxes[:, 3:6].abs(), p=1
            )
            yaw_cost = _angle_difference(
                object_boxes[batch_index, :, 6], target_boxes[:, 6]
            )
            velocity_cost = torch.cdist(
                object_boxes[batch_index, :, 7:9], target_boxes[:, 7:9], p=1
            )
            total_cost = (
                self.class_weight * class_cost
                + self.center_weight * center_cost
                + self.size_weight * size_cost
                + self.yaw_weight * yaw_cost
                + self.velocity_weight * velocity_cost
            )
            pred_index, target_index = _linear_sum_assignment(total_cost)
            if pred_index.numel() == 0:
                continue
            target_classes[batch_index, pred_index, target_labels[target_index]] = 1.0
            if target_objectness is not None:
                target_objectness[batch_index, pred_index] = 1.0
            total_box = total_box + F.smooth_l1_loss(
                object_boxes[batch_index, pred_index], target_boxes[target_index], reduction="sum"
            )
            total_gt += int(target_index.numel())
        if target_objectness is not None:
            for batch_index in range(batch_size):
                matched_mask = target_objectness[batch_index] > 0.5
                query_selection[batch_index] = self._select_query_mask(
                    object_logits[batch_index],
                    objectness_logits[batch_index].float()
                    if objectness_logits is not None
                    else None,
                    matched_mask,
                )
        elif self.loss_mode == "focal_hardneg":
            for batch_index in range(batch_size):
                query_selection[batch_index] = self._select_query_mask(
                    object_logits[batch_index],
                    None,
                    torch.zeros(queries, dtype=torch.bool, device=object_logits.device),
                )

        normalizer = max(total_gt, 1)
        objectness_loss = zero
        if objectness_logits is not None:
            target_tensor = (
                target_objectness
                if target_objectness is not None
                else torch.zeros_like(objectness_logits.float())
            )
            objectness_loss = self._objectness_loss(
                objectness_logits.float(),
                target_tensor,
                query_selection,
                float(normalizer),
            )
        reference_loss = zero
        if reference_points is not None and target_objectness is not None:
            unmatched = (target_objectness < 0.5).float()
            if float(unmatched.sum()) > 0.0:
                reference_loss = (
                    (
                        F.smooth_l1_loss(
                            object_boxes[..., :3],
                            reference_points.float(),
                            reduction="none",
                        ).sum(dim=-1)
                        * unmatched
                    ).sum()
                    / unmatched.sum().clamp_min(1.0)
                ) * self.reference_weight
        cls_loss = self._classification_loss(
            object_logits,
            target_classes,
            query_selection,
            float(normalizer),
        )
        box_loss = total_box / float(normalizer)
        return {
            "objectness": objectness_loss,
            "object_cls": cls_loss,
            "object_box": box_loss,
            "object_ref": reference_loss,
        }


class LaneSetCriterion(nn.Module):
    """Sparse lane matching loss for OpenLane-style polylines."""

    def __init__(self, geometry_weight: float = 4.0) -> None:
        super().__init__()
        self.geometry_weight = geometry_weight

    def forward(
        self, lane_logits: Tensor, lane_polylines: Tensor, batch: SceneBatch
    ) -> dict[str, Tensor]:
        lane_logits = lane_logits.float()
        lane_polylines = lane_polylines.float()
        zero = _zero_with_grad(lane_logits)
        if batch.lane_targets is None:
            return {"lane_logits": zero, "lane_shape": zero}

        batch_size, queries = lane_logits.shape
        target_valid = torch.zeros_like(lane_logits)
        total_shape = zero
        total_gt = 0

        for batch_index in range(batch_size):
            valid = batch.lane_targets.valid_mask[batch_index]
            target_lanes = batch.lane_targets.polylines[batch_index][valid]
            if target_lanes.numel() == 0:
                continue
            pred_lanes = lane_polylines[batch_index]
            cost = (
                pred_lanes.unsqueeze(1) - target_lanes.unsqueeze(0)
            ).abs().mean(dim=(-1, -2)) * self.geometry_weight
            pred_index, target_index = _linear_sum_assignment(cost)
            if pred_index.numel() == 0:
                continue
            target_valid[batch_index, pred_index] = 1.0
            total_shape = total_shape + F.smooth_l1_loss(
                pred_lanes[pred_index], target_lanes[target_index], reduction="sum"
            )
            total_gt += int(target_index.numel())

        normalizer = max(total_gt, 1)
        logit_loss = F.binary_cross_entropy_with_logits(
            lane_logits, target_valid, reduction="sum"
        ) / float(normalizer)
        shape_loss = total_shape / float(normalizer)
        return {"lane_logits": logit_loss, "lane_shape": shape_loss}


class MultitaskCriterion(nn.Module):
    """Combine detection, lane, and optional distillation losses."""

    def __init__(
        self,
        detection: DetectionSetCriterion | None = None,
        lane: LaneSetCriterion | None = None,
        distillation: DistillationObjective | None = None,
        enable_distillation: bool = True,
    ) -> None:
        super().__init__()
        self.detection = detection if detection is not None else DetectionSetCriterion()
        self.lane = lane if lane is not None else LaneSetCriterion()
        self.distillation = distillation if distillation is not None else DistillationObjective()
        self.enable_distillation = enable_distillation

    def forward(self, outputs: dict[str, object], batch: SceneBatch) -> dict[str, Tensor]:
        object_logits = cast(Tensor, outputs["object_logits"])
        object_boxes = cast(Tensor, outputs["object_boxes"])
        lane_logits = cast(Tensor, outputs["lane_logits"])
        lane_polylines = cast(Tensor, outputs["lane_polylines"])
        objectness_logits = cast(Tensor | None, outputs.get("objectness_logits"))
        temporal_state = cast(TemporalState, outputs["temporal_state"])
        seed_bank = cast(QuerySeedBank, outputs["seed_bank"])

        losses = {}
        losses.update(
            self.detection(
                object_logits,
                object_boxes,
                batch,
                objectness_logits=objectness_logits,
                reference_points=temporal_state.object_refs,
            )
        )
        losses.update(self.lane(lane_logits, lane_polylines, batch))
        losses.update(
            self.distillation(
                object_logits=object_logits,
                object_queries=temporal_state.object_queries,
                object_boxes=object_boxes,
                seed_bank=seed_bank,
                teacher=batch.teacher_targets if self.enable_distillation else None,
            )
        )
        total = _zero_with_grad(object_logits) + _zero_with_grad(lane_logits)
        for key, value in losses.items():
            if key != "kd_total":
                total = total + value
        losses["total"] = total
        return losses
