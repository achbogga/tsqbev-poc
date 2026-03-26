from __future__ import annotations

from tsqbev.adapters import (
    MapTRNuScenesAdapter,
    NuScenesODAdapter,
    OpenLaneAdapter,
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
