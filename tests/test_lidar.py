from __future__ import annotations

import torch

from tsqbev.config import ModelConfig
from tsqbev.lidar import LidarSeedEncoder, pillarize_points


def test_pillarize_points_merges_points_in_same_cell(small_config: ModelConfig) -> None:
    points = torch.tensor(
        [
            [
                [0.1, 0.1, 0.0, 1.0],
                [0.2, 0.2, 0.5, 2.0],
                [2.0, 2.0, 0.0, 3.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, True]])
    pillars = pillarize_points(points, mask, small_config.pillar)
    assert pillars.centers_xyz.shape[0] == 2


def test_lidar_seed_encoder_returns_fixed_budget(small_config: ModelConfig) -> None:
    points = torch.randn(2, 64, 4)
    mask = torch.ones(2, 64, dtype=torch.bool)
    encoder = LidarSeedEncoder(small_config)
    queries, refs, scores = encoder(points, mask)
    assert queries.shape == (2, small_config.q_lidar, small_config.model_dim)
    assert refs.shape == (2, small_config.q_lidar, 3)
    assert scores.shape == (2, small_config.q_lidar)
