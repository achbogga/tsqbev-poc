"""Core typed contracts for multimodal inputs, targets, and temporal state.

References:
- PETRv2 multitask query framing:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf
- StreamPETR persistent query state:
  https://arxiv.org/abs/2303.11926
- BEVDistill teacher-target design motivation:
  https://arxiv.org/abs/2211.09386
- MapTR map-token framing:
  https://arxiv.org/abs/2208.14437
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


def _check_rank(name: str, tensor: Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


@dataclass(slots=True)
class CameraProposals:
    """2D proposal boxes and scores per camera view."""

    boxes_xyxy: Tensor
    scores: Tensor

    def validate(self, batch_size: int, views: int) -> None:
        _check_rank("boxes_xyxy", self.boxes_xyxy, 4)
        _check_rank("scores", self.scores, 3)
        if self.boxes_xyxy.shape[:2] != (batch_size, views):
            raise ValueError("proposal boxes batch/view shape mismatch")
        if self.scores.shape[:2] != (batch_size, views):
            raise ValueError("proposal scores batch/view shape mismatch")
        if self.boxes_xyxy.shape[-1] != 4:
            raise ValueError("proposal boxes must end with 4 coordinates")


@dataclass(slots=True)
class ObjectTargets:
    """Sparse object-level supervision."""

    boxes_3d: Tensor
    labels: Tensor
    valid_mask: Tensor

    def validate(self, batch_size: int) -> None:
        _check_rank("boxes_3d", self.boxes_3d, 3)
        _check_rank("labels", self.labels, 2)
        _check_rank("valid_mask", self.valid_mask, 2)
        if self.boxes_3d.shape[0] != batch_size:
            raise ValueError("object targets batch mismatch")
        if self.boxes_3d.shape[-1] != 9:
            raise ValueError("boxes_3d must use 9 parameters")
        if self.labels.shape != self.valid_mask.shape:
            raise ValueError("object target labels and valid_mask shape mismatch")


@dataclass(slots=True)
class LaneTargets:
    """Sparse lane polyline supervision."""

    polylines: Tensor
    valid_mask: Tensor

    def validate(self, batch_size: int) -> None:
        _check_rank("polylines", self.polylines, 4)
        _check_rank("valid_mask", self.valid_mask, 2)
        if self.polylines.shape[0] != batch_size:
            raise ValueError("lane targets batch mismatch")
        if self.polylines.shape[-1] != 3:
            raise ValueError("lane polyline points must be 3D")


@dataclass(slots=True)
class MapPriorBatch:
    """Optional HD-map priors represented as tokens."""

    tokens: Tensor
    coords_xy: Tensor
    valid_mask: Tensor

    def validate(self, batch_size: int) -> None:
        _check_rank("map tokens", self.tokens, 3)
        _check_rank("map coords_xy", self.coords_xy, 3)
        _check_rank("map valid_mask", self.valid_mask, 2)
        if self.tokens.shape[0] != batch_size:
            raise ValueError("map prior batch mismatch")
        if self.coords_xy.shape[-1] != 2:
            raise ValueError("map coords must be xy")


@dataclass(slots=True)
class TeacherTargets:
    """Optional teacher cache tensors used for distillation."""

    object_features: Tensor | None = None
    object_boxes: Tensor | None = None
    object_labels: Tensor | None = None
    object_scores: Tensor | None = None
    lane_features: Tensor | None = None
    router_logits: Tensor | None = None
    valid_mask: Tensor | None = None

    def validate(self, batch_size: int) -> None:
        if self.object_features is not None:
            _check_rank("teacher object_features", self.object_features, 3)
            if self.object_features.shape[0] != batch_size:
                raise ValueError("teacher object_features batch mismatch")
        if self.object_boxes is not None:
            _check_rank("teacher object_boxes", self.object_boxes, 3)
            if self.object_boxes.shape[0] != batch_size:
                raise ValueError("teacher object_boxes batch mismatch")
            if self.object_boxes.shape[-1] != 9:
                raise ValueError("teacher object_boxes must use 9 parameters")
        if self.object_labels is not None:
            _check_rank("teacher object_labels", self.object_labels, 2)
            if self.object_labels.shape[0] != batch_size:
                raise ValueError("teacher object_labels batch mismatch")
        if self.object_scores is not None:
            _check_rank("teacher object_scores", self.object_scores, 2)
            if self.object_scores.shape[0] != batch_size:
                raise ValueError("teacher object_scores batch mismatch")
        if self.router_logits is not None:
            _check_rank("teacher router_logits", self.router_logits, 2)
            if self.router_logits.shape[0] != batch_size:
                raise ValueError("teacher router_logits batch mismatch")
        if self.valid_mask is not None:
            _check_rank("teacher valid_mask", self.valid_mask, 2)
            if self.valid_mask.shape[0] != batch_size:
                raise ValueError("teacher valid_mask batch mismatch")
        if self.object_boxes is not None and self.object_labels is not None:
            if self.object_boxes.shape[:2] != self.object_labels.shape:
                raise ValueError("teacher object_boxes/object_labels shape mismatch")
        if self.object_boxes is not None and self.object_scores is not None:
            if self.object_boxes.shape[:2] != self.object_scores.shape:
                raise ValueError("teacher object_boxes/object_scores shape mismatch")
        if self.object_boxes is not None and self.valid_mask is not None:
            if self.object_boxes.shape[:2] != self.valid_mask.shape:
                raise ValueError("teacher object_boxes/valid_mask shape mismatch")


@dataclass(slots=True)
class QuerySeedBank:
    """Routed sparse object queries and metadata."""

    embeddings: Tensor
    refs_xyz: Tensor
    scores: Tensor
    source_ids: Tensor
    keep_logits: Tensor

    def validate(self, batch_size: int) -> None:
        _check_rank("embeddings", self.embeddings, 3)
        _check_rank("refs_xyz", self.refs_xyz, 3)
        _check_rank("scores", self.scores, 2)
        _check_rank("source_ids", self.source_ids, 2)
        _check_rank("keep_logits", self.keep_logits, 2)
        if self.embeddings.shape[0] != batch_size:
            raise ValueError("seed bank batch mismatch")
        if self.refs_xyz.shape[-1] != 3:
            raise ValueError("refs_xyz must contain xyz")


@dataclass(slots=True)
class TemporalState:
    """Persistent sparse state carried between frames."""

    object_queries: Tensor
    object_refs: Tensor
    lane_queries: Tensor | None = None

    def validate(self, batch_size: int) -> None:
        _check_rank("object_queries", self.object_queries, 3)
        _check_rank("object_refs", self.object_refs, 3)
        if self.object_queries.shape[0] != batch_size:
            raise ValueError("temporal state batch mismatch")
        if self.object_refs.shape[-1] != 3:
            raise ValueError("temporal refs must be xyz")
        if self.lane_queries is not None:
            _check_rank("lane_queries", self.lane_queries, 3)


@dataclass(slots=True)
class SceneBatch:
    """Canonical multimodal batch for TSQBEV."""

    images: Tensor
    lidar_points: Tensor
    lidar_mask: Tensor
    intrinsics: Tensor
    extrinsics: Tensor
    ego_pose: Tensor
    time_delta_s: Tensor
    camera_proposals: CameraProposals | None = None
    od_targets: ObjectTargets | None = None
    lane_targets: LaneTargets | None = None
    map_priors: MapPriorBatch | None = None
    teacher_targets: TeacherTargets | None = None

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

    @property
    def views(self) -> int:
        return int(self.images.shape[1])

    def validate(self) -> None:
        _check_rank("images", self.images, 5)
        _check_rank("lidar_points", self.lidar_points, 3)
        _check_rank("lidar_mask", self.lidar_mask, 2)
        _check_rank("intrinsics", self.intrinsics, 4)
        _check_rank("extrinsics", self.extrinsics, 4)
        _check_rank("ego_pose", self.ego_pose, 3)
        _check_rank("time_delta_s", self.time_delta_s, 1)
        if self.images.shape[2] != 3:
            raise ValueError("images must be RGB")
        if self.lidar_points.shape[-1] != 4:
            raise ValueError("lidar_points must end with xyz + intensity")
        if self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError("intrinsics must be 3x3")
        if self.extrinsics.shape[-2:] != (4, 4):
            raise ValueError("extrinsics must be 4x4")
        if self.ego_pose.shape[-2:] != (4, 4):
            raise ValueError("ego_pose must be 4x4")
        if self.lidar_points.shape[:2] != self.lidar_mask.shape:
            raise ValueError("lidar_points and lidar_mask shape mismatch")
        if self.images.shape[0] != self.time_delta_s.shape[0]:
            raise ValueError("time_delta_s batch mismatch")
        if self.camera_proposals is not None:
            self.camera_proposals.validate(self.batch_size, self.views)
        if self.od_targets is not None:
            self.od_targets.validate(self.batch_size)
        if self.lane_targets is not None:
            self.lane_targets.validate(self.batch_size)
        if self.map_priors is not None:
            self.map_priors.validate(self.batch_size)
        if self.teacher_targets is not None:
            self.teacher_targets.validate(self.batch_size)
