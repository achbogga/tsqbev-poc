from __future__ import annotations

from dataclasses import replace

import torch

from tsqbev.model import TSQBEVModel
from tsqbev.teacher_seed import TeacherSeedEncoder


def test_teacher_seed_encoder_projects_cached_teacher_boxes(small_config, synthetic_batch) -> None:
    assert synthetic_batch.teacher_targets is not None
    encoder = TeacherSeedEncoder(small_config)
    encoded = encoder(synthetic_batch.teacher_targets)
    assert encoded is not None
    queries, refs, scores = encoded
    assert queries.shape == (
        synthetic_batch.batch_size,
        small_config.q_lidar,
        small_config.model_dim,
    )
    assert refs.shape == (synthetic_batch.batch_size, small_config.q_lidar, 3)
    assert scores.shape == (synthetic_batch.batch_size, small_config.q_lidar)


def test_model_can_replace_lidar_with_teacher_seeds(small_config, synthetic_batch) -> None:
    assert synthetic_batch.teacher_targets is not None
    config = small_config.model_copy(update={"teacher_seed_mode": "replace_lidar"})
    model = TSQBEVModel(config)
    zero_lidar_batch = replace(
        synthetic_batch,
        lidar_points=torch.zeros_like(synthetic_batch.lidar_points),
        lidar_mask=torch.zeros_like(synthetic_batch.lidar_mask),
    )

    outputs = model(zero_lidar_batch)
    assert outputs["object_logits"].shape[1] == small_config.max_object_queries
