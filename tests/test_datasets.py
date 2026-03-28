from __future__ import annotations

import json

from PIL import Image

from tsqbev.datasets import OpenLaneDataset, collate_scene_examples


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
