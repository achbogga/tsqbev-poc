"""Thin public and Torc-inspired adapter contracts.

References:
- OpenLane dataset:
  https://github.com/OpenDriveLab/OpenLane
- MapTR public code:
  https://github.com/hustvl/MapTR
- Torc internal reference path used for thin compatibility:
  torc_ml/projects/scene_modeling/scene_modeling/data/datasets
"""

from __future__ import annotations

from collections.abc import Mapping


def assert_required_keys(sample: Mapping[str, object], required_keys: tuple[str, ...]) -> None:
    """Raise a clear error if a sample is missing required keys."""

    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise KeyError(f"missing required keys: {missing}")


class NuScenesODAdapter:
    """Minimal OD adapter contract for public nuScenes-style samples."""

    required_keys = (
        "images",
        "lidar_points",
        "intrinsics",
        "extrinsics",
        "ego_pose",
        "boxes_3d",
        "labels",
    )

    def validate(self, sample: Mapping[str, object]) -> None:
        assert_required_keys(sample, self.required_keys)


class OpenLaneAdapter:
    """Minimal lane adapter contract for public OpenLane-style samples."""

    required_keys = (
        "images",
        "intrinsics",
        "extrinsics",
        "ego_pose",
        "lane_polylines",
        "lane_valid_mask",
    )

    def validate(self, sample: Mapping[str, object]) -> None:
        assert_required_keys(sample, self.required_keys)


class MapTRNuScenesAdapter:
    """Minimal map prior adapter contract."""

    required_keys = ("map_tokens", "map_coords_xy", "map_valid_mask")

    def validate(self, sample: Mapping[str, object]) -> None:
        assert_required_keys(sample, self.required_keys)


class TorcThinAdapter:
    """Thin schema mirror for Torc-style column naming."""

    required_columns = (
        "general_timestamp",
        "lidar_path",
        "annotations_3d_cuboid",
        "img_051_intrinsic",
        "img_051_odom_extrinsic",
        "img_053_intrinsic",
        "img_053_odom_extrinsic",
    )

    def validate(self, sample: Mapping[str, object]) -> None:
        assert_required_keys(sample, self.required_columns)
