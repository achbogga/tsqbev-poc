from __future__ import annotations

import pytest
import torch

from tsqbev.reset_contracts import (
    BevFeatureBatch,
    BevGridSpec,
    DenseDetectionOutputs,
    TemporalBevState,
    VectorMapOutputs,
)


def test_bev_feature_batch_validation_passes() -> None:
    grid = BevGridSpec(x_range_m=(-2.0, 2.0), y_range_m=(-2.0, 2.0), cell_size_m=1.0, channels=8)
    features = BevFeatureBatch(
        bev=torch.zeros(2, 8, 4, 4),
        grid=grid,
        valid_mask=torch.ones(2, 4, 4, dtype=torch.bool),
    )

    features.validate(batch_size=2)


def test_bev_feature_batch_validation_rejects_shape_mismatch() -> None:
    grid = BevGridSpec(x_range_m=(-2.0, 2.0), y_range_m=(-2.0, 2.0), cell_size_m=1.0, channels=4)
    features = BevFeatureBatch(bev=torch.zeros(1, 5, 4, 4), grid=grid)

    with pytest.raises(ValueError, match="channels"):
        features.validate(batch_size=1)


def test_temporal_bev_state_validates_ego_motion() -> None:
    state = TemporalBevState(memory=torch.zeros(1, 8, 4, 4), ego_motion=torch.eye(4).view(1, 4, 4))

    state.validate(batch_size=1)


def test_detection_outputs_validate_spatial_alignment() -> None:
    outputs = DenseDetectionOutputs(
        heatmap=torch.zeros(1, 10, 8, 8),
        box_regression=torch.zeros(1, 9, 8, 8),
        velocity=torch.zeros(1, 2, 8, 8),
    )

    outputs.validate(batch_size=1)


def test_vector_map_outputs_reject_invalid_polyline_shape() -> None:
    outputs = VectorMapOutputs(
        polylines=torch.zeros(1, 3, 8, 2),
        labels=torch.zeros(1, 3, dtype=torch.long),
        scores=torch.zeros(1, 3),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="xyz"):
        outputs.validate(batch_size=1)
