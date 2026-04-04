from __future__ import annotations

import json

from tsqbev.eval_openlane import write_openlane_test_list


def test_write_openlane_test_list_uses_annotation_file_paths(tmp_path) -> None:
    dataroot = tmp_path / "OpenLane"
    lane_root = dataroot / "lane3d_300" / "validation" / "segment-001"
    lane_root.mkdir(parents=True)
    annotation = {
        "file_path": "validation/segment-001/frame-0001.jpg",
        "intrinsic": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "extrinsic": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "lane_lines": [],
    }
    (lane_root / "frame-0001.json").write_text(json.dumps(annotation))

    output_path = dataroot / "validation_test_list.txt"
    result = write_openlane_test_list(
        dataroot=dataroot,
        output_path=output_path,
        split="validation",
        subset="lane3d_300",
    )

    assert result == output_path
    assert output_path.read_text() == "validation/segment-001/frame-0001.jpg\n"
