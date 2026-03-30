from __future__ import annotations

from dataclasses import replace

import torch

from tsqbev.datasets import SceneExample, collate_scene_examples
from tsqbev.model import TSQBEVModel
from tsqbev.teacher_seed import TeacherSeedEncoder


def _slice_optional_teacher_field(field, index: int):
    if field is None:
        return None
    return field[index : index + 1]


def _single_example_from_batch(batch, index: int) -> SceneExample:
    assert batch.teacher_targets is not None
    teacher_targets = replace(
        batch.teacher_targets,
        object_features=_slice_optional_teacher_field(batch.teacher_targets.object_features, index),
        object_boxes=_slice_optional_teacher_field(batch.teacher_targets.object_boxes, index),
        object_labels=_slice_optional_teacher_field(batch.teacher_targets.object_labels, index),
        object_scores=_slice_optional_teacher_field(batch.teacher_targets.object_scores, index),
        lane_features=_slice_optional_teacher_field(batch.teacher_targets.lane_features, index),
        router_logits=_slice_optional_teacher_field(batch.teacher_targets.router_logits, index),
        valid_mask=_slice_optional_teacher_field(batch.teacher_targets.valid_mask, index),
    )
    return SceneExample(
        scene=replace(
            batch,
            images=batch.images[index : index + 1],
            lidar_points=batch.lidar_points[index : index + 1],
            lidar_mask=batch.lidar_mask[index : index + 1],
            intrinsics=batch.intrinsics[index : index + 1],
            extrinsics=batch.extrinsics[index : index + 1],
            ego_pose=batch.ego_pose[index : index + 1],
            time_delta_s=batch.time_delta_s[index : index + 1],
            camera_proposals=None,
            od_targets=None,
            lane_targets=None,
            map_priors=None,
            teacher_targets=teacher_targets,
        ),
        metadata={"sample_token": f"sample-{index}"},
    )


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


def test_model_can_replace_lidar_with_teacher_seeds_after_batch_collation(
    small_config,
    synthetic_batch,
) -> None:
    examples = [
        _single_example_from_batch(synthetic_batch, 0),
        _single_example_from_batch(synthetic_batch, 1),
    ]
    batch, _ = collate_scene_examples(examples)
    config = small_config.model_copy(update={"teacher_seed_mode": "replace_lidar"})
    model = TSQBEVModel(config)
    zero_lidar_batch = replace(
        batch,
        lidar_points=torch.zeros_like(batch.lidar_points),
        lidar_mask=torch.zeros_like(batch.lidar_mask),
    )

    outputs = model(zero_lidar_batch)
    assert outputs["object_logits"].shape[0] == batch.batch_size


def test_model_can_replace_lidar_refs_with_teacher_centers(
    small_config,
    synthetic_batch,
) -> None:
    assert synthetic_batch.teacher_targets is not None
    config = small_config.model_copy(update={"teacher_seed_mode": "replace_lidar_refs"})
    model = TSQBEVModel(config)
    _lidar_queries, lidar_refs, lidar_scores = model.lidar_encoder(
        torch.zeros_like(synthetic_batch.lidar_points),
        torch.zeros_like(synthetic_batch.lidar_mask),
    )

    replaced_refs, replaced_scores = model._replace_lidar_refs_from_teacher(
        lidar_refs,
        lidar_scores,
        synthetic_batch.teacher_targets,
    )
    teacher_scores = synthetic_batch.teacher_targets.object_scores
    teacher_boxes = synthetic_batch.teacher_targets.object_boxes
    assert teacher_scores is not None
    assert teacher_boxes is not None
    teacher_order = torch.argsort(teacher_scores[0], descending=True)[: small_config.q_lidar]

    assert torch.allclose(
        replaced_refs[0, : teacher_order.numel()],
        teacher_boxes[0, teacher_order, :3],
    )
    assert torch.allclose(
        replaced_scores[0, : teacher_order.numel()],
        teacher_scores[0, teacher_order],
    )
