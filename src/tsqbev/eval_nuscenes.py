"""nuScenes prediction export and official local evaluation helpers.

References:
- nuScenes official detection result format:
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md
- nuScenes official local evaluation entrypoint:
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tsqbev.datasets import NuScenesDataset, collate_scene_examples
from tsqbev.labels import NUSCENES_DETECTION_NAMES
from tsqbev.model import TSQBEVModel
from tsqbev.quaternion import quaternion_from_yaw, rotate_xy, yaw_from_rotation_matrix
from tsqbev.runtime import move_batch, resolve_device
from tsqbev.teacher_backends import TeacherProviderConfig, build_teacher_provider
from tsqbev.teacher_dataset import TeacherAugmentedDataset


def _load_nuscenes_eval() -> tuple[Any, Any, Any, Any]:
    try:
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import DetectionEval
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:  # pragma: no cover - exercised by real runs.
        raise RuntimeError(
            "nuScenes evaluation requires `uv sync --extra data` to install nuscenes-devkit"
        ) from exc
    return NuScenes, DetectionEval, config_factory, None


@torch.no_grad()
def export_nuscenes_predictions(
    model: TSQBEVModel,
    dataroot: str | Path,
    version: str,
    split: str,
    output_path: str | Path,
    score_threshold: float = 0.25,
    top_k: int = 300,
    device: str | None = None,
    teacher_provider_config: TeacherProviderConfig | None = None,
) -> Path:
    """Write a nuScenes detection submission JSON for local validation."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    dataset: Dataset[Any] = NuScenesDataset(dataroot=dataroot, version=version, split=split)
    if teacher_provider_config is not None:
        dataset = TeacherAugmentedDataset(
            dataset,
            build_teacher_provider(teacher_provider_config),
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_scene_examples,
        pin_memory=torch.cuda.is_available(),
    )

    model = model.to(resolved_device).eval()
    results: dict[str, list[dict[str, object]]] = {}

    for batch, metadata_list in loader:
        batch = move_batch(batch, resolved_device)
        outputs = model(batch)
        logits = outputs["object_logits"]
        boxes = outputs["object_boxes"]
        assert isinstance(logits, torch.Tensor)
        assert isinstance(boxes, torch.Tensor)

        metadata = metadata_list[0]
        sample_token = str(metadata["sample_token"])
        ego_to_global = np.asarray(metadata["ego_to_global"], dtype=np.float32)
        ego_yaw = yaw_from_rotation_matrix(ego_to_global[:3, :3])

        class_scores, class_ids = logits[0].sigmoid().max(dim=-1)
        order = torch.argsort(class_scores, descending=True)
        sample_results: list[dict[str, object]] = []
        for query_index in order[:top_k]:
            score = float(class_scores[query_index])
            if score < score_threshold:
                continue
            raw_box = boxes[0, query_index].detach().cpu().numpy().astype(np.float32)
            center_global = (np.append(raw_box[:3], 1.0) @ ego_to_global.T)[:3]
            size = np.maximum(np.abs(raw_box[3:6]), 0.1)
            yaw_global = ego_yaw + float(raw_box[6])
            velocity_global = rotate_xy(raw_box[7:9].tolist(), ego_yaw)

            sample_results.append(
                {
                    "sample_token": sample_token,
                    "translation": center_global.tolist(),
                    "size": size.tolist(),
                    "rotation": list(quaternion_from_yaw(yaw_global)),
                    "velocity": velocity_global.tolist(),
                    "detection_name": NUSCENES_DETECTION_NAMES[int(class_ids[query_index])],
                    "detection_score": score,
                    "attribute_name": "",
                }
            )
        results[sample_token] = sample_results

    payload = {
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def evaluate_nuscenes_predictions(
    dataroot: str | Path,
    version: str,
    split: str,
    result_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run the official nuScenes local validation metrics on a result JSON."""

    NuScenes, DetectionEval, config_factory, _ = _load_nuscenes_eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    evaluator = DetectionEval(
        nusc=nusc,
        config=config_factory("detection_cvpr_2019"),
        result_path=str(result_path),
        eval_set=split,
        output_dir=str(output_dir),
        verbose=False,
    )
    return evaluator.main(plot_examples=0, render_curves=False)
