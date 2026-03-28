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


class DistillationObjective(nn.Module):
    """Compute optional distillation losses from teacher cache tensors."""

    def __init__(
        self, feature_weight: float = 1.0, box_weight: float = 1.0, router_weight: float = 0.5
    ) -> None:
        super().__init__()
        self.feature_weight = feature_weight
        self.box_weight = box_weight
        self.router_weight = router_weight

    def forward(
        self,
        object_queries: Tensor,
        object_boxes: Tensor,
        seed_bank: QuerySeedBank,
        teacher: TeacherTargets | None,
    ) -> dict[str, Tensor]:
        zero = _zero_with_grad(object_queries)
        if teacher is None:
            return {"kd_total": zero, "kd_features": zero, "kd_boxes": zero, "kd_router": zero}

        mask = teacher.valid_mask
        feature_loss = zero
        if teacher.object_features is not None:
            feature_loss = (
                _masked_mean((object_queries - teacher.object_features) ** 2, mask)
                * self.feature_weight
            )

        box_loss = zero
        if teacher.object_boxes is not None:
            box_loss = (
                _masked_mean(
                    F.smooth_l1_loss(object_boxes, teacher.object_boxes, reduction="none"),
                    mask.unsqueeze(-1) if mask is not None else None,
                )
                * self.box_weight
            )

        router_loss = zero
        if teacher.router_logits is not None:
            router_loss = (
                F.mse_loss(seed_bank.keep_logits, teacher.router_logits) * self.router_weight
            )

        total = feature_loss + box_loss + router_loss
        return {
            "kd_total": total,
            "kd_features": feature_loss,
            "kd_boxes": box_loss,
            "kd_router": router_loss,
        }
