from __future__ import annotations

import torch

from tsqbev.geometry import project_points, ray_points_from_proposals


def test_project_points_identity_camera() -> None:
    points = torch.tensor([[0.0, 0.0, 5.0], [2.0, 4.0, 2.0]])
    intrinsics = torch.eye(3)
    extrinsics = torch.eye(4)
    uvz = project_points(points, intrinsics, extrinsics)
    assert torch.allclose(uvz[0], torch.tensor([0.0, 0.0, 5.0]))
    assert torch.allclose(uvz[1], torch.tensor([1.0, 2.0, 2.0]))


def test_ray_points_from_proposals_identity_camera() -> None:
    boxes = torch.tensor([[[[0.0, 0.0, 0.0, 0.0]]]])
    intrinsics = torch.eye(3).view(1, 1, 3, 3)
    extrinsics = torch.eye(4).view(1, 1, 4, 4)
    depths = torch.tensor([5.0, 10.0])
    ego_points = ray_points_from_proposals(boxes, intrinsics, extrinsics, depths)
    assert ego_points.shape == (1, 1, 1, 2, 3)
    assert torch.allclose(ego_points[0, 0, 0, 0], torch.tensor([0.0, 0.0, 5.0]))
    assert torch.allclose(ego_points[0, 0, 0, 1], torch.tensor([0.0, 0.0, 10.0]))
