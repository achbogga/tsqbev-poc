from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tsqbev.quaternion import quaternion_from_yaw
from tsqbev.teacher_cache import TeacherCacheStore
from tsqbev.teacher_import import (
    cache_nuscenes_detection_results,
    detections_to_teacher_targets,
)


def test_detections_to_teacher_targets_converts_standard_nuscenes_rows() -> None:
    targets = detections_to_teacher_targets(
        [
            {
                "translation": [1.0, 2.0, 3.0],
                "size": [4.0, 5.0, 6.0],
                "rotation": quaternion_from_yaw(0.5),
                "velocity": [1.0, 0.0],
                "detection_name": "car",
                "detection_score": 0.9,
            }
        ],
        global_to_ego=np.eye(4, dtype=np.float32),
        ego_yaw=0.0,
    )

    assert targets.object_boxes is not None
    assert targets.object_labels is not None
    assert targets.object_scores is not None
    assert targets.valid_mask is not None
    assert tuple(targets.object_boxes.shape) == (1, 1, 9)
    assert float(targets.object_boxes[0, 0, 0]) == 1.0
    assert float(targets.object_boxes[0, 0, 6]) == 0.5
    assert int(targets.object_labels[0, 0]) == 3
    assert float(targets.object_scores[0, 0]) == pytest.approx(0.9)
    assert bool(targets.valid_mask[0, 0]) is True


def test_cache_nuscenes_detection_results_writes_teacher_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "detections.json"
    result_path.write_text(
        json.dumps(
            {
                "meta": {},
                "results": {
                    "sample-1": [
                        {
                            "translation": [1.0, 0.0, 0.0],
                            "size": [4.0, 2.0, 1.5],
                            "rotation": quaternion_from_yaw(0.0),
                            "velocity": [0.0, 0.0],
                            "detection_name": "car",
                            "detection_score": 0.8,
                        }
                    ]
                },
            }
        )
    )

    class _FakeNuScenes:
        def __init__(self, version: str, dataroot: str, verbose: bool) -> None:
            del version, dataroot, verbose

        def get(self, table: str, token: str) -> dict[str, object]:
            if table == "sample":
                return {"data": {"LIDAR_TOP": "lidar-sd"}}
            if table == "sample_data":
                return {"ego_pose_token": "ego-pose"}
            if table == "ego_pose":
                return {"rotation": quaternion_from_yaw(0.0), "translation": [0.0, 0.0, 0.0]}
            raise KeyError((table, token))

    monkeypatch.setattr("tsqbev.teacher_import._load_nuscenes", lambda: _FakeNuScenes)

    cache_dir = tmp_path / "cache"
    summary = cache_nuscenes_detection_results(
        dataroot=tmp_path,
        version="v1.0-mini",
        result_path=result_path,
        cache_dir=cache_dir,
    )

    assert summary["stored_records"] == 1
    record = TeacherCacheStore(cache_dir).load("sample-1")
    assert record is not None
    assert record.backend == "nuscenes-result-json"
    assert record.targets.object_scores is not None
    assert float(record.targets.object_scores[0, 0]) == pytest.approx(0.8)
