from __future__ import annotations

import torch

from tsqbev.contracts import LaneTargets, ObjectTargets, SceneBatch
from tsqbev.losses import DetectionSetCriterion, LaneSetCriterion, MultitaskCriterion
from tsqbev.model import TSQBEVModel


def test_detection_set_criterion_prefers_exact_match(small_config, synthetic_batch) -> None:
    criterion = DetectionSetCriterion()
    target_boxes = synthetic_batch.od_targets.boxes_3d[:, :2].clone()
    target_labels = synthetic_batch.od_targets.labels[:, :2].clone()

    logits = torch.full(
        (
            synthetic_batch.batch_size,
            small_config.max_object_queries,
            small_config.num_object_classes,
        ),
        -8.0,
    )
    batch_indices = torch.arange(synthetic_batch.batch_size)
    logits[batch_indices, 0, target_labels[:, 0]] = 8.0
    logits[batch_indices, 1, target_labels[:, 1]] = 8.0
    boxes = torch.zeros(synthetic_batch.batch_size, small_config.max_object_queries, 9)
    boxes[:, 0] = target_boxes[:, 0]
    boxes[:, 1] = target_boxes[:, 1]

    batch = SceneBatch(
        images=synthetic_batch.images,
        lidar_points=synthetic_batch.lidar_points,
        lidar_mask=synthetic_batch.lidar_mask,
        intrinsics=synthetic_batch.intrinsics,
        extrinsics=synthetic_batch.extrinsics,
        ego_pose=synthetic_batch.ego_pose,
        time_delta_s=synthetic_batch.time_delta_s,
        od_targets=ObjectTargets(
            boxes_3d=target_boxes,
            labels=target_labels,
            valid_mask=torch.ones_like(target_labels, dtype=torch.bool),
        ),
    )
    objectness = torch.full((synthetic_batch.batch_size, small_config.max_object_queries), -8.0)
    objectness[batch_indices, :2] = 8.0
    losses = criterion(logits, boxes, batch, objectness_logits=objectness)
    assert losses["objectness"] < 5e-2
    assert losses["object_box"] < 1e-5
    assert losses["object_cls"] < 5e-2


def test_quality_focal_objectness_penalizes_poorly_localized_matches(
    small_config, synthetic_batch
) -> None:
    criterion = DetectionSetCriterion(loss_mode="quality_focal")
    target_boxes = synthetic_batch.od_targets.boxes_3d[:, :1].clone()
    target_labels = synthetic_batch.od_targets.labels[:, :1].clone()
    batch_indices = torch.arange(synthetic_batch.batch_size)

    logits = torch.full(
        (
            synthetic_batch.batch_size,
            small_config.max_object_queries,
            small_config.num_object_classes,
        ),
        -20.0,
    )
    logits[batch_indices, 0, target_labels[:, 0]] = 8.0
    objectness = torch.full((synthetic_batch.batch_size, small_config.max_object_queries), -8.0)
    objectness[:, 0] = 8.0

    exact_boxes = torch.full(
        (synthetic_batch.batch_size, small_config.max_object_queries, 9),
        100.0,
    )
    exact_boxes[:, 0] = target_boxes[:, 0]
    shifted_boxes = exact_boxes.clone()
    shifted_boxes[:, 0, 0] = shifted_boxes[:, 0, 0] + 6.0

    batch = SceneBatch(
        images=synthetic_batch.images,
        lidar_points=synthetic_batch.lidar_points,
        lidar_mask=synthetic_batch.lidar_mask,
        intrinsics=synthetic_batch.intrinsics,
        extrinsics=synthetic_batch.extrinsics,
        ego_pose=synthetic_batch.ego_pose,
        time_delta_s=synthetic_batch.time_delta_s,
        od_targets=ObjectTargets(
            boxes_3d=target_boxes,
            labels=target_labels,
            valid_mask=torch.ones_like(target_labels, dtype=torch.bool),
        ),
    )

    exact_losses = criterion(logits, exact_boxes, batch, objectness_logits=objectness)
    shifted_losses = criterion(logits, shifted_boxes, batch, objectness_logits=objectness)

    assert shifted_losses["objectness"] > exact_losses["objectness"]


def test_quality_focal_keeps_low_quality_matches_out_of_unmatched_penalty(
    small_config, synthetic_batch
) -> None:
    criterion = DetectionSetCriterion(loss_mode="quality_focal")
    target_boxes = synthetic_batch.od_targets.boxes_3d[:, :1].clone()
    target_labels = synthetic_batch.od_targets.labels[:, :1].clone()
    batch_indices = torch.arange(synthetic_batch.batch_size)

    logits = torch.full(
        (
            synthetic_batch.batch_size,
            small_config.max_object_queries,
            small_config.num_object_classes,
        ),
        -20.0,
    )
    logits[batch_indices, 0, target_labels[:, 0]] = 8.0
    objectness = torch.full((synthetic_batch.batch_size, small_config.max_object_queries), -8.0)
    objectness[:, 0] = 8.0

    exact_boxes = torch.full(
        (synthetic_batch.batch_size, small_config.max_object_queries, 9),
        100.0,
    )
    exact_boxes[:, 0] = target_boxes[:, 0]
    shifted_boxes = exact_boxes.clone()
    shifted_boxes[:, 0, 0] = shifted_boxes[:, 0, 0] + 6.0
    reference_points = exact_boxes[..., :3].clone()

    batch = SceneBatch(
        images=synthetic_batch.images,
        lidar_points=synthetic_batch.lidar_points,
        lidar_mask=synthetic_batch.lidar_mask,
        intrinsics=synthetic_batch.intrinsics,
        extrinsics=synthetic_batch.extrinsics,
        ego_pose=synthetic_batch.ego_pose,
        time_delta_s=synthetic_batch.time_delta_s,
        od_targets=ObjectTargets(
            boxes_3d=target_boxes,
            labels=target_labels,
            valid_mask=torch.ones_like(target_labels, dtype=torch.bool),
        ),
    )

    exact_losses = criterion(
        logits,
        exact_boxes,
        batch,
        objectness_logits=objectness,
        reference_points=reference_points,
    )
    shifted_losses = criterion(
        logits,
        shifted_boxes,
        batch,
        objectness_logits=objectness,
        reference_points=reference_points,
    )

    assert shifted_losses["object_ref"] < 1e-6
    assert exact_losses["object_ref"] < 1e-6


def test_lane_set_criterion_prefers_exact_match(small_config, synthetic_batch) -> None:
    criterion = LaneSetCriterion()
    target_polylines = synthetic_batch.lane_targets.polylines[:, :2].clone()
    target_mask = synthetic_batch.lane_targets.valid_mask[:, :2].clone()
    logits = torch.full((synthetic_batch.batch_size, small_config.lane_queries), -8.0)
    logits[:, :2] = 8.0
    polylines = torch.zeros(
        synthetic_batch.batch_size,
        small_config.lane_queries,
        small_config.lane_points,
        3,
    )
    polylines[:, :2] = target_polylines

    batch = SceneBatch(
        images=synthetic_batch.images,
        lidar_points=synthetic_batch.lidar_points,
        lidar_mask=synthetic_batch.lidar_mask,
        intrinsics=synthetic_batch.intrinsics,
        extrinsics=synthetic_batch.extrinsics,
        ego_pose=synthetic_batch.ego_pose,
        time_delta_s=synthetic_batch.time_delta_s,
        lane_targets=LaneTargets(polylines=target_polylines, valid_mask=target_mask),
    )
    losses = criterion(logits, polylines, batch)
    assert losses["lane_shape"] < 1e-5
    assert losses["lane_logits"] < 3e-3


def test_multitask_criterion_keeps_grad_when_batch_has_no_supervision(
    small_config, synthetic_batch
) -> None:
    model = TSQBEVModel(small_config)
    criterion = MultitaskCriterion()
    batch = SceneBatch(
        images=synthetic_batch.images,
        lidar_points=synthetic_batch.lidar_points,
        lidar_mask=synthetic_batch.lidar_mask,
        intrinsics=synthetic_batch.intrinsics,
        extrinsics=synthetic_batch.extrinsics,
        ego_pose=synthetic_batch.ego_pose,
        time_delta_s=synthetic_batch.time_delta_s,
        camera_proposals=synthetic_batch.camera_proposals,
    )
    outputs = model(batch)
    losses = criterion(outputs, batch)
    assert losses["total"].requires_grad
    losses["total"].backward()


def test_multitask_criterion_can_disable_teacher_distillation(
    small_config, synthetic_batch
) -> None:
    model = TSQBEVModel(small_config.model_copy(update={"teacher_seed_mode": "replace_lidar"}))
    criterion = MultitaskCriterion(enable_distillation=False)
    outputs = model(synthetic_batch)
    losses = criterion(outputs, synthetic_batch)
    assert float(losses["kd_total"]) == 0.0


def test_detection_set_criterion_adds_teacher_anchor_prior_losses(
    small_config,
    synthetic_batch,
) -> None:
    model = TSQBEVModel(small_config.model_copy(update={"teacher_seed_mode": "replace_lidar"}))
    outputs = model(synthetic_batch)
    criterion = DetectionSetCriterion()
    losses = criterion(
        outputs["object_logits"],
        outputs["object_boxes"],
        synthetic_batch,
        objectness_logits=outputs["objectness_logits"],
        reference_points=outputs["temporal_state"].object_refs,
        teacher_prior_labels=outputs["seed_bank"].prior_labels,
        teacher_prior_scores=outputs["seed_bank"].prior_scores,
        teacher_prior_valid_mask=outputs["seed_bank"].prior_valid_mask,
    )
    assert losses["object_teacher_anchor_cls"] > 0.0
    assert losses["object_teacher_anchor_obj"] > 0.0
