"""Bridge cached external teacher detections into sparse LiDAR seed queries.

References:
- CenterPoint:
  https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf
- OpenPCDet model zoo:
  https://github.com/open-mmlab/OpenPCDet
- BEVDistill:
  https://arxiv.org/abs/2211.09386
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tsqbev.config import ModelConfig
from tsqbev.contracts import TeacherTargets

Tensor = torch.Tensor


def select_teacher_seed_indices(
    labels: Tensor,
    scores: Tensor,
    *,
    max_keep: int,
    mode: str,
) -> Tensor:
    """Select cached teacher detections for seed replacement."""

    order = torch.argsort(scores, descending=True)
    keep_count = min(int(max_keep), int(order.numel()))
    if keep_count <= 0:
        return order[:0]
    if mode == "score_topk":
        return order[:keep_count]
    if mode != "class_balanced_round_robin":
        raise ValueError(f"unsupported teacher seed selection mode: {mode}")

    class_buckets: dict[int, list[int]] = {}
    for index in order.tolist():
        label = int(labels[index])
        class_buckets.setdefault(label, []).append(int(index))
    class_order = sorted(
        class_buckets,
        key=lambda label: float(scores[class_buckets[label][0]]),
        reverse=True,
    )
    selected: list[int] = []
    cursors = {label: 0 for label in class_order}
    while len(selected) < keep_count:
        progressed = False
        for label in class_order:
            cursor = cursors[label]
            bucket = class_buckets[label]
            if cursor >= len(bucket):
                continue
            selected.append(bucket[cursor])
            cursors[label] = cursor + 1
            progressed = True
            if len(selected) >= keep_count:
                break
        if not progressed:
            break
    return torch.tensor(selected, device=labels.device, dtype=torch.long)


class TeacherSeedEncoder(nn.Module):
    """Project cached teacher detections into the LiDAR seed interface.

    Repo note:
    This is an explicit bootstrap bridge for evaluating the tsqbev student with a
    stronger external LiDAR detector. The teacher stays outside the core runtime;
    the student consumes cached teacher boxes, labels, and scores as sparse seeds.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = 9 + 1 + config.num_object_classes
        self.project = nn.Sequential(
            nn.Linear(input_dim, config.model_dim),
            nn.ReLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )

    def encode_with_priors(
        self,
        teacher: TeacherTargets | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None:
        if teacher is None:
            return None
        if (
            teacher.object_boxes is None
            or teacher.object_labels is None
            or teacher.object_scores is None
        ):
            return None

        batch = teacher.object_boxes.shape[0]
        device = teacher.object_boxes.device
        dtype = teacher.object_boxes.dtype
        queries = torch.zeros(
            batch,
            self.config.q_lidar,
            self.config.model_dim,
            device=device,
            dtype=dtype,
        )
        refs = torch.zeros(batch, self.config.q_lidar, 3, device=device, dtype=dtype)
        scores = torch.zeros(batch, self.config.q_lidar, device=device, dtype=dtype)
        prior_labels = torch.zeros(batch, self.config.q_lidar, device=device, dtype=torch.long)
        prior_scores = torch.zeros(batch, self.config.q_lidar, device=device, dtype=dtype)
        prior_valid_mask = torch.zeros(batch, self.config.q_lidar, device=device, dtype=torch.bool)
        valid_mask = (
            teacher.valid_mask
            if teacher.valid_mask is not None
            else torch.ones_like(teacher.object_scores, dtype=torch.bool)
        )

        for batch_index in range(batch):
            valid = valid_mask[batch_index]
            if not bool(valid.any()):
                continue
            batch_boxes = teacher.object_boxes[batch_index][valid]
            batch_labels = teacher.object_labels[batch_index][valid]
            batch_scores = teacher.object_scores[batch_index][valid]
            keep = select_teacher_seed_indices(
                batch_labels,
                batch_scores,
                max_keep=self.config.q_lidar,
                mode=self.config.teacher_seed_selection_mode,
            )
            batch_boxes = batch_boxes[keep]
            batch_labels = batch_labels[keep]
            batch_scores = batch_scores[keep]

            one_hot = F.one_hot(
                batch_labels.to(torch.long),
                num_classes=self.config.num_object_classes,
            ).to(dtype=dtype)
            features = torch.cat((batch_boxes, batch_scores.unsqueeze(-1), one_hot), dim=-1)
            projected = self.project(features)
            count = projected.shape[0]
            queries[batch_index, :count] = projected
            refs[batch_index, :count] = batch_boxes[:, :3]
            scores[batch_index, :count] = batch_scores
            prior_labels[batch_index, :count] = batch_labels
            prior_scores[batch_index, :count] = batch_scores
            prior_valid_mask[batch_index, :count] = True

        return queries, refs, scores, prior_labels, prior_scores, prior_valid_mask

    def forward(self, teacher: TeacherTargets | None) -> tuple[Tensor, Tensor, Tensor] | None:
        encoded = self.encode_with_priors(teacher)
        if encoded is None:
            return None
        queries, refs, scores, _prior_labels, _prior_scores, _prior_valid_mask = encoded
        return queries, refs, scores
