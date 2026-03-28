from __future__ import annotations

from dataclasses import replace

from tsqbev.config import ModelConfig
from tsqbev.model import TSQBEVModel


def test_model_forward_shapes(small_config, synthetic_batch) -> None:
    model = TSQBEVModel(small_config)
    outputs = model(synthetic_batch)
    assert outputs["object_logits"].shape == (
        synthetic_batch.batch_size,
        small_config.max_object_queries,
        small_config.num_object_classes,
    )
    assert outputs["object_boxes"].shape == (
        synthetic_batch.batch_size,
        small_config.max_object_queries,
        9,
    )
    assert outputs["lane_logits"].shape == (
        synthetic_batch.batch_size,
        small_config.lane_queries,
    )
    assert outputs["lane_polylines"].shape == (
        synthetic_batch.batch_size,
        small_config.lane_queries,
        small_config.lane_points,
        3,
    )


def test_model_temporal_state_round_trip(small_config, synthetic_batch) -> None:
    model = TSQBEVModel(small_config)
    first = model(synthetic_batch)
    second = model(synthetic_batch, state=first["temporal_state"])
    assert (
        second["temporal_state"].object_queries.shape
        == first["temporal_state"].object_queries.shape
    )


def test_model_supports_torchvision_backbone(synthetic_batch) -> None:
    config = ModelConfig.small().model_copy(
        update={
            "image_backbone": "mobilenet_v3_large",
            "pretrained_image_backbone": False,
            "freeze_image_backbone": True,
        }
    )
    model = TSQBEVModel(config)
    batch = replace(
        synthetic_batch,
        images=synthetic_batch.images[:1],
        lidar_points=synthetic_batch.lidar_points[:1],
        lidar_mask=synthetic_batch.lidar_mask[:1],
        intrinsics=synthetic_batch.intrinsics[:1],
        extrinsics=synthetic_batch.extrinsics[:1],
        ego_pose=synthetic_batch.ego_pose[:1],
        time_delta_s=synthetic_batch.time_delta_s[:1],
        od_targets=None,
        lane_targets=None,
        map_priors=replace(
            synthetic_batch.map_priors,
            tokens=synthetic_batch.map_priors.tokens[:1],
            coords_xy=synthetic_batch.map_priors.coords_xy[:1],
            valid_mask=synthetic_batch.map_priors.valid_mask[:1],
        )
        if synthetic_batch.map_priors is not None
        else None,
    )
    outputs = model(batch)
    assert outputs["object_logits"].shape[0] == 1
