from __future__ import annotations

import json
from dataclasses import replace

from PIL import Image

from tsqbev.datasets import OpenLaneDataset, SceneExample, collate_scene_examples


def _slice_optional_teacher_field(field, index: int):
    if field is None:
        return None
    return field[index : index + 1]


def _single_example_from_batch(batch, index: int) -> SceneExample:
    teacher_targets = None
    if batch.teacher_targets is not None:
        teacher_targets = replace(
            batch.teacher_targets,
            object_features=_slice_optional_teacher_field(
                batch.teacher_targets.object_features,
                index,
            ),
            object_boxes=_slice_optional_teacher_field(batch.teacher_targets.object_boxes, index),
            object_labels=_slice_optional_teacher_field(batch.teacher_targets.object_labels, index),
            object_scores=_slice_optional_teacher_field(batch.teacher_targets.object_scores, index),
            lane_features=_slice_optional_teacher_field(batch.teacher_targets.lane_features, index),
            router_logits=_slice_optional_teacher_field(batch.teacher_targets.router_logits, index),
            valid_mask=_slice_optional_teacher_field(batch.teacher_targets.valid_mask, index),
        )
    example_batch = replace(
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
    )
    return SceneExample(scene=example_batch, metadata={"sample_token": f"sample-{index}"})


def test_openlane_dataset_reads_public_layout(tmp_path) -> None:
    image_rel = "validation/segment-001/frame-001.jpg"
    image_path = tmp_path / "images" / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 160), color=(16, 32, 64)).save(image_path)

    annotation_path = tmp_path / "lane3d_300" / "validation" / "segment-001" / "frame-001.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation = {
        "intrinsic": [[100.0, 0.0, 160.0], [0.0, 100.0, 80.0], [0.0, 0.0, 1.0]],
        "extrinsic": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "file_path": image_rel,
        "lane_lines": [
            {
                "xyz": [[0.0, 0.2, 0.4, 0.6], [5.0, 10.0, 15.0, 20.0], [0.0, 0.0, 0.1, 0.1]],
                "visibility": [1.0, 1.0, 1.0, 1.0],
                "category": 1,
                "attribute": 0,
                "track_id": 0,
            }
        ],
    }
    annotation_path.write_text(json.dumps(annotation))

    dataset = OpenLaneDataset(tmp_path, split="validation", subset="lane3d_300", lane_points=8)
    example = dataset[0]
    assert example.scene.images.shape == (1, 1, 3, 256, 704)
    assert example.scene.lane_targets is not None
    assert example.scene.lane_targets.polylines.shape == (1, 1, 8, 3)
    assert example.metadata["file_path"] == image_rel


def test_collate_scene_examples_pads_and_batches_openlane(tmp_path) -> None:
    for frame_index, lane_count in enumerate((0, 2), start=1):
        image_rel = f"validation/segment-001/frame-{frame_index:03d}.jpg"
        image_path = tmp_path / "images" / image_rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=(frame_index, 0, 0)).save(image_path)
        annotation_path = (
            tmp_path / "lane3d_300" / "validation" / "segment-001" / f"frame-{frame_index:03d}.json"
        )
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        lane_lines = []
        for lane_id in range(lane_count):
            lane_lines.append(
                {
                    "xyz": [
                        [0.0 + lane_id, 0.2 + lane_id, 0.4 + lane_id],
                        [5.0, 10.0, 15.0],
                        [0.0, 0.0, 0.1],
                    ],
                    "visibility": [1.0, 1.0, 1.0],
                    "category": 1,
                    "attribute": 0,
                    "track_id": lane_id,
                }
            )
        annotation_path.write_text(
            json.dumps(
                {
                    "intrinsic": [[10.0, 0.0, 32.0], [0.0, 10.0, 32.0], [0.0, 0.0, 1.0]],
                    "extrinsic": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "file_path": image_rel,
                    "lane_lines": lane_lines,
                }
            )
        )
    dataset = OpenLaneDataset(tmp_path, split="validation", subset="lane3d_300")
    batch, metadata = collate_scene_examples([dataset[0], dataset[1]])
    assert batch.images.shape[0] == 2
    assert batch.lane_targets is not None
    assert batch.lane_targets.polylines.shape[0] == 2
    assert batch.lane_targets.valid_mask.shape == (2, 2)
    assert batch.lane_targets.valid_mask[0].sum().item() == 0
    assert batch.lane_targets.valid_mask[1].sum().item() == 2
    assert len(metadata) == 2


def test_collate_scene_examples_preserves_teacher_targets(synthetic_batch) -> None:
    assert synthetic_batch.teacher_targets is not None
    examples = [
        _single_example_from_batch(synthetic_batch, 0),
        _single_example_from_batch(synthetic_batch, 1),
    ]

    batch, metadata = collate_scene_examples(examples)

    assert batch.teacher_targets is not None
    assert batch.teacher_targets.object_boxes is not None
    assert batch.teacher_targets.object_boxes.shape[0] == 2
    assert batch.teacher_targets.valid_mask is not None
    assert batch.teacher_targets.valid_mask.shape[0] == 2
    assert len(metadata) == 2
