"""Dataset label and sensor constants.

References:
- nuScenes official detection task class list:
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md
- nuScenes official camera channel names:
  https://github.com/nutonomy/nuscenes-devkit
"""

from __future__ import annotations

NUSCENES_DETECTION_NAMES: tuple[str, ...] = (
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
)

NUSCENES_DETECTION_NAME_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(NUSCENES_DETECTION_NAMES)
}

NUSCENES_CAMERA_NAMES: tuple[str, ...] = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

OPENLANE_DEFAULT_CATEGORY = 1
