from __future__ import annotations

from tsqbev.contracts import CameraProposals


def test_scene_batch_validate(synthetic_batch) -> None:
    synthetic_batch.validate()


def test_camera_proposals_validate(synthetic_batch) -> None:
    proposals = CameraProposals(
        boxes_xyxy=synthetic_batch.images.new_zeros(
            synthetic_batch.batch_size,
            synthetic_batch.views,
            4,
            4,
        ),
        scores=synthetic_batch.images.new_zeros(
            synthetic_batch.batch_size,
            synthetic_batch.views,
            4,
        ),
    )
    proposals.validate(synthetic_batch.batch_size, synthetic_batch.views)
