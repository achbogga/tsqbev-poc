from __future__ import annotations

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
