"""Minimal quaternion helpers for dataset transforms and export.

References:
- nuScenes uses quaternion sensor and pose records throughout the official devkit:
  https://github.com/nutonomy/nuscenes-devkit
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def rotation_matrix_from_quaternion(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Convert a unit quaternion in ``(w, x, y, z)`` order into a 3x3 matrix."""

    w, x, y, z = quaternion_wxyz
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_from_quaternion(
    quaternion_wxyz: Sequence[float], translation_xyz: Sequence[float]
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from quaternion + translation."""

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_matrix_from_quaternion(quaternion_wxyz)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=np.float32)
    return transform


def yaw_from_quaternion(quaternion_wxyz: Sequence[float]) -> float:
    """Extract the z-axis yaw from a quaternion in ``(w, x, y, z)`` order."""

    w, x, y, z = quaternion_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    """Return a z-axis rotation quaternion in ``(w, x, y, z)`` order."""

    half = yaw_rad * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def rotate_xy(vector_xy: Sequence[float], yaw_rad: float) -> np.ndarray:
    """Rotate a planar vector by ``yaw_rad``."""

    x, y = vector_xy
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array((c * x - s * y, s * x + c * y), dtype=np.float32)


def yaw_from_rotation_matrix(rotation: np.ndarray) -> float:
    """Extract the planar yaw from the top-left 3x3 rotation matrix."""

    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
