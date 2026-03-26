from __future__ import annotations

from tsqbev.adapters import (
    MapTRNuScenesAdapter,
    NuScenesODAdapter,
    OpenLaneAdapter,
    TorcThinAdapter,
)


def test_nuscenes_adapter_validate() -> None:
    adapter = NuScenesODAdapter()
    adapter.validate(
        {
            "images": object(),
            "lidar_points": object(),
            "intrinsics": object(),
            "extrinsics": object(),
            "ego_pose": object(),
            "boxes_3d": object(),
            "labels": object(),
        }
    )


def test_openlane_adapter_validate() -> None:
    adapter = OpenLaneAdapter()
    adapter.validate(
        {
            "images": object(),
            "intrinsics": object(),
            "extrinsics": object(),
            "ego_pose": object(),
            "lane_polylines": object(),
            "lane_valid_mask": object(),
        }
    )


def test_maptr_adapter_validate() -> None:
    adapter = MapTRNuScenesAdapter()
    adapter.validate(
        {
            "map_tokens": object(),
            "map_coords_xy": object(),
            "map_valid_mask": object(),
        }
    )


def test_torc_adapter_validate() -> None:
    adapter = TorcThinAdapter()
    adapter.validate(
        {
            "general_timestamp": object(),
            "lidar_path": object(),
            "annotations_3d_cuboid": object(),
            "img_051_intrinsic": object(),
            "img_051_odom_extrinsic": object(),
            "img_053_intrinsic": object(),
            "img_053_odom_extrinsic": object(),
        }
    )
