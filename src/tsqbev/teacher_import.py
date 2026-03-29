"""Import external nuScenes detection JSON into the teacher cache format.

References:
- nuScenes detection submission format:
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md
- OpenPCDet model zoo:
  https://github.com/open-mmlab/OpenPCDet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tsqbev.contracts import TeacherTargets
from tsqbev.labels import NUSCENES_DETECTION_NAME_TO_INDEX
from tsqbev.quaternion import rotate_xy, transform_from_quaternion, wrap_angle, yaw_from_quaternion
from tsqbev.teacher_cache import TeacherCacheStore


def _load_nuscenes() -> Any:
    from nuscenes.nuscenes import NuScenes

    return NuScenes


def detections_to_teacher_targets(
    detections: list[dict[str, Any]],
    *,
    global_to_ego: np.ndarray,
    ego_yaw: float,
    top_k: int = 300,
) -> TeacherTargets:
    """Convert standard nuScenes detection JSON rows into cached teacher targets."""

    ranked = sorted(
        detections,
        key=lambda row: float(row.get("detection_score", 0.0)),
        reverse=True,
    )[:top_k]
    if not ranked:
        empty_boxes = torch.zeros(1, 0, 9, dtype=torch.float32)
        empty_labels = torch.zeros(1, 0, dtype=torch.long)
        empty_scores = torch.zeros(1, 0, dtype=torch.float32)
        empty_mask = torch.zeros(1, 0, dtype=torch.bool)
        return TeacherTargets(
            object_boxes=empty_boxes,
            object_labels=empty_labels,
            object_scores=empty_scores,
            valid_mask=empty_mask,
        )

    boxes: list[list[float]] = []
    labels: list[int] = []
    scores: list[float] = []
    for detection in ranked:
        detection_name = str(detection["detection_name"])
        center_global = np.asarray(detection["translation"], dtype=np.float32)
        center_ego = (np.append(center_global, 1.0) @ global_to_ego.T)[:3]
        size = np.abs(np.asarray(detection["size"], dtype=np.float32))
        yaw_global = yaw_from_quaternion(detection["rotation"])
        velocity_global = np.asarray(detection.get("velocity", [0.0, 0.0]), dtype=np.float32)
        velocity_ego = rotate_xy(velocity_global.tolist(), -ego_yaw)
        boxes.append(
            [
                float(center_ego[0]),
                float(center_ego[1]),
                float(center_ego[2]),
                float(size[0]),
                float(size[1]),
                float(size[2]),
                float(wrap_angle(yaw_global - ego_yaw)),
                float(velocity_ego[0]),
                float(velocity_ego[1]),
            ]
        )
        labels.append(int(NUSCENES_DETECTION_NAME_TO_INDEX[detection_name]))
        scores.append(float(detection["detection_score"]))

    return TeacherTargets(
        object_boxes=torch.tensor([boxes], dtype=torch.float32),
        object_labels=torch.tensor([labels], dtype=torch.long),
        object_scores=torch.tensor([scores], dtype=torch.float32),
        valid_mask=torch.ones(1, len(boxes), dtype=torch.bool),
    )


def cache_nuscenes_detection_results(
    dataroot: str | Path,
    version: str,
    result_path: str | Path,
    cache_dir: str | Path,
    *,
    top_k: int = 300,
    backend: str = "nuscenes-result-json",
) -> dict[str, Any]:
    """Convert standard nuScenes detections into per-sample teacher cache records."""

    result_path = Path(result_path)
    payload = json.loads(result_path.read_text())
    if "results" not in payload or not isinstance(payload["results"], dict):
        raise ValueError("result JSON must contain a `results` dictionary")

    NuScenes = _load_nuscenes()
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    store = TeacherCacheStore(cache_dir)
    stored = 0

    for sample_token, detections_payload in payload["results"].items():
        detections = list(detections_payload)
        sample = nusc.get("sample", sample_token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_to_global = transform_from_quaternion(ego_pose["rotation"], ego_pose["translation"])
        global_to_ego = np.linalg.inv(ego_to_global)
        ego_yaw = yaw_from_quaternion(ego_pose["rotation"])
        targets = detections_to_teacher_targets(
            detections,
            global_to_ego=global_to_ego,
            ego_yaw=ego_yaw,
            top_k=top_k,
        )
        store.save(
            sample_token,
            backend=backend,
            targets=targets,
            metadata={
                "sample_token": sample_token,
                "source_result_path": str(result_path),
                "num_detections": len(detections),
                "top_k": top_k,
            },
        )
        stored += 1

    summary = {
        "status": "completed",
        "backend": backend,
        "result_path": str(result_path),
        "cache_dir": str(cache_dir),
        "stored_records": stored,
        "top_k": top_k,
    }
    summary_path = Path(cache_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
