"""Synthetic fixtures used by tests and smoke runs.

References:
- The synthetic path exists to keep the repo functional before real dataset wiring.
- Contracts are still grounded in the cited papers and official repos in docs/reference-matrix.md.
"""

from __future__ import annotations

import torch

from tsqbev.config import ModelConfig
from tsqbev.contracts import LaneTargets, MapPriorBatch, ObjectTargets, SceneBatch, TeacherTargets


def make_synthetic_batch(
    config: ModelConfig,
    batch_size: int = 2,
    image_height: int = 96,
    image_width: int = 160,
    max_points: int = 256,
    max_objects: int = 12,
    with_map: bool = True,
    with_teacher: bool = True,
) -> SceneBatch:
    """Create a compact synthetic batch that satisfies all contracts."""

    images = torch.randn(batch_size, config.views, 3, image_height, image_width)
    lidar_points = torch.randn(batch_size, max_points, 4)
    lidar_mask = torch.ones(batch_size, max_points, dtype=torch.bool)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, config.views, 1, 1)
    intrinsics[..., 0, 0] = 128.0
    intrinsics[..., 1, 1] = 128.0
    intrinsics[..., 0, 2] = image_width / 2.0
    intrinsics[..., 1, 2] = image_height / 2.0
    extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, config.views, 1, 1)
    ego_pose = torch.eye(4).view(1, 4, 4).repeat(batch_size, 1, 1)
    time_delta_s = torch.full((batch_size,), 0.1)

    od_targets = ObjectTargets(
        boxes_3d=torch.randn(batch_size, max_objects, 9),
        labels=torch.randint(0, config.num_object_classes, (batch_size, max_objects)),
        valid_mask=torch.ones(batch_size, max_objects, dtype=torch.bool),
    )
    lane_targets = LaneTargets(
        polylines=torch.randn(batch_size, config.lane_queries, config.lane_points, 3),
        valid_mask=torch.ones(batch_size, config.lane_queries, dtype=torch.bool),
    )

    map_priors = None
    if with_map:
        map_priors = MapPriorBatch(
            tokens=torch.randn(batch_size, 16, config.map_input_dim),
            coords_xy=torch.randn(batch_size, 16, 2),
            valid_mask=torch.ones(batch_size, 16, dtype=torch.bool),
        )

    teacher_targets = None
    if with_teacher:
        teacher_targets = TeacherTargets(
            object_features=torch.randn(batch_size, config.max_object_queries, config.model_dim),
            object_boxes=torch.randn(batch_size, config.max_object_queries, 9),
            router_logits=torch.randn(batch_size, config.max_object_queries),
            valid_mask=torch.ones(batch_size, config.max_object_queries),
        )

    batch = SceneBatch(
        images=images,
        lidar_points=lidar_points,
        lidar_mask=lidar_mask,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        ego_pose=ego_pose,
        time_delta_s=time_delta_s,
        od_targets=od_targets,
        lane_targets=lane_targets,
        map_priors=map_priors,
        teacher_targets=teacher_targets,
    )
    batch.validate()
    return batch
