"""Dataset-root validation helpers for public baselines.

References:
- nuScenes official download structure and metadata requirements:
  https://github.com/nutonomy/nuscenes-devkit
- OpenLane official folder structure:
  https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md
"""

from __future__ import annotations

from pathlib import Path


def check_nuscenes_root(dataroot: str | Path) -> dict[str, object]:
    """Return a simple readiness report for a nuScenes dataroot."""

    root = Path(dataroot)
    checks = {
        "root_exists": root.exists(),
        "metadata_dir": (root / "v1.0-trainval").exists(),
        "scene_json": (root / "v1.0-trainval" / "scene.json").exists(),
        "sample_data_json": (root / "v1.0-trainval" / "sample_data.json").exists(),
        "calibrated_sensor_json": (root / "v1.0-trainval" / "calibrated_sensor.json").exists(),
        "ego_pose_json": (root / "v1.0-trainval" / "ego_pose.json").exists(),
        "sample_images": (root / "samples" / "CAM_FRONT").exists(),
        "sample_lidar": (root / "samples" / "LIDAR_TOP").exists(),
        "maps": (root / "maps").exists(),
    }
    ready = all(bool(value) for value in checks.values())
    return {"dataset": "nuScenes", "root": str(root), "ready": ready, "checks": checks}


def check_openlane_root(dataroot: str | Path) -> dict[str, object]:
    """Return a simple readiness report for an OpenLane dataroot."""

    root = Path(dataroot)
    checks = {
        "root_exists": root.exists(),
        "images_training": (root / "images" / "training").exists(),
        "images_validation": (root / "images" / "validation").exists(),
        "lane3d_300_training": (root / "lane3d_300" / "training").exists(),
        "lane3d_300_validation": (root / "lane3d_300" / "validation").exists(),
        "lane3d_1000_training": (root / "lane3d_1000" / "training").exists(),
        "lane3d_1000_validation": (root / "lane3d_1000" / "validation").exists(),
    }
    ready = (
        bool(checks["root_exists"])
        and bool(checks["images_training"])
        and bool(checks["images_validation"])
        and (
            bool(checks["lane3d_300_training"])
            and bool(checks["lane3d_300_validation"])
            or bool(checks["lane3d_1000_training"])
            and bool(checks["lane3d_1000_validation"])
        )
    )
    return {"dataset": "OpenLane", "root": str(root), "ready": ready, "checks": checks}
