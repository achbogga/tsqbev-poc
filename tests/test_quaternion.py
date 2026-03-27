from __future__ import annotations

import math

from tsqbev.quaternion import (
    quaternion_from_yaw,
    rotation_matrix_from_quaternion,
    wrap_angle,
    yaw_from_quaternion,
    yaw_from_rotation_matrix,
)


def test_quaternion_yaw_round_trip() -> None:
    yaw = 0.73
    quaternion = quaternion_from_yaw(yaw)
    rotation = rotation_matrix_from_quaternion(quaternion)
    assert math.isclose(yaw_from_quaternion(quaternion), yaw, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(yaw_from_rotation_matrix(rotation), yaw, rel_tol=1e-6, abs_tol=1e-6)


def test_wrap_angle_bounds() -> None:
    wrapped = wrap_angle(4.0 * math.pi + 0.25)
    assert -math.pi <= wrapped < math.pi
    assert math.isclose(wrapped, 0.25, rel_tol=1e-6, abs_tol=1e-6)
