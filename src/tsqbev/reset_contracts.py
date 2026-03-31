"""Dense-BEV contracts for the reset architecture migration.

References:
- BEVFusion unified BEV representation:
  https://github.com/mit-han-lab/bevfusion
- CenterPoint dense detection head:
  https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf
- MapTR vectorized map prediction:
  https://arxiv.org/abs/2208.14437
- BEVDet temporal camera BEV lifting:
  https://github.com/HuangJunJie2017/BEVDet
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


def _require_rank(name: str, tensor: Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


@dataclass(frozen=True, slots=True)
class BevGridSpec:
    """Canonical dense BEV grid metadata."""

    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    cell_size_m: float
    channels: int
    temporal_frames: int = 1

    @property
    def width(self) -> int:
        return int(round((self.x_range_m[1] - self.x_range_m[0]) / self.cell_size_m))

    @property
    def height(self) -> int:
        return int(round((self.y_range_m[1] - self.y_range_m[0]) / self.cell_size_m))


@dataclass(slots=True)
class BevFeatureBatch:
    """Dense BEV features shared between detection and vector-map heads."""

    bev: Tensor
    grid: BevGridSpec
    valid_mask: Tensor | None = None

    def validate(self, batch_size: int | None = None) -> None:
        _require_rank("bev", self.bev, 4)
        if batch_size is not None and self.bev.shape[0] != batch_size:
            raise ValueError("bev batch mismatch")
        if self.bev.shape[1] != self.grid.channels:
            raise ValueError("bev channels must match grid spec")
        if self.bev.shape[2:] != (self.grid.height, self.grid.width):
            raise ValueError("bev spatial shape must match grid spec")
        if self.valid_mask is not None:
            _require_rank("valid_mask", self.valid_mask, 3)
            if self.valid_mask.shape[0] != self.bev.shape[0]:
                raise ValueError("valid_mask batch mismatch")
            if self.valid_mask.shape[1:] != self.bev.shape[2:]:
                raise ValueError("valid_mask spatial shape mismatch")


@dataclass(slots=True)
class TemporalBevState:
    """Dense temporal state aligned in BEV rather than sparse query space."""

    memory: Tensor
    ego_motion: Tensor | None = None

    def validate(self, batch_size: int | None = None) -> None:
        _require_rank("memory", self.memory, 4)
        if batch_size is not None and self.memory.shape[0] != batch_size:
            raise ValueError("temporal memory batch mismatch")
        if self.ego_motion is not None:
            _require_rank("ego_motion", self.ego_motion, 3)
            if self.ego_motion.shape[-2:] != (4, 4):
                raise ValueError("ego_motion must be 4x4")


@dataclass(slots=True)
class DenseDetectionOutputs:
    """Dense BEV detection outputs in CenterHead-style format."""

    heatmap: Tensor
    box_regression: Tensor
    velocity: Tensor | None = None

    def validate(self, batch_size: int) -> None:
        _require_rank("heatmap", self.heatmap, 4)
        _require_rank("box_regression", self.box_regression, 4)
        if self.heatmap.shape[0] != batch_size or self.box_regression.shape[0] != batch_size:
            raise ValueError("detection output batch mismatch")
        if self.heatmap.shape[-2:] != self.box_regression.shape[-2:]:
            raise ValueError("heatmap and box_regression spatial shape mismatch")
        if self.velocity is not None:
            _require_rank("velocity", self.velocity, 4)
            if self.velocity.shape[0] != batch_size:
                raise ValueError("velocity batch mismatch")
            if self.velocity.shape[-2:] != self.heatmap.shape[-2:]:
                raise ValueError("velocity spatial shape mismatch")


@dataclass(slots=True)
class VectorMapOutputs:
    """Vectorized lane/map predictions on top of shared BEV features."""

    polylines: Tensor
    labels: Tensor
    scores: Tensor
    valid_mask: Tensor

    def validate(self, batch_size: int) -> None:
        _require_rank("polylines", self.polylines, 4)
        _require_rank("labels", self.labels, 2)
        _require_rank("scores", self.scores, 2)
        _require_rank("valid_mask", self.valid_mask, 2)
        if self.polylines.shape[0] != batch_size:
            raise ValueError("polylines batch mismatch")
        if self.polylines.shape[-1] != 3:
            raise ValueError("polyline points must be xyz")
        if self.labels.shape != self.scores.shape or self.labels.shape != self.valid_mask.shape:
            raise ValueError("vector output tensors must share [batch, queries] shape")
