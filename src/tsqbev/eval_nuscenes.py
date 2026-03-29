"""nuScenes prediction export and official local evaluation helpers.

References:
- nuScenes official detection result format:
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md
- nuScenes official local evaluation entrypoint:
  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py
"""

from __future__ import annotations

import json
import time
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
    sample_tokens: list[str] | None = None,
    teacher_provider_config: TeacherProviderConfig | None = None,
) -> Path:
    """Write a nuScenes detection submission JSON for local validation."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    dataset: Dataset[Any] = NuScenesDataset(
        dataroot=dataroot,
        version=version,
        split=split,
        sample_tokens=sample_tokens,
    )
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
    sample_tokens: list[str] | None = None,
) -> dict[str, object]:
    """Run the official nuScenes local validation metrics on a result JSON."""

    NuScenes, DetectionEval, config_factory, _ = _load_nuscenes_eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    if sample_tokens is not None:
        return _evaluate_nuscenes_predictions_subset(
            nusc=nusc,
            result_path=result_path,
            output_dir=output_dir,
            sample_tokens=sample_tokens,
        )
    evaluator = DetectionEval(
        nusc=nusc,
        config=config_factory("detection_cvpr_2019"),
        result_path=str(result_path),
        eval_set=split,
        output_dir=str(output_dir),
        verbose=False,
    )
    return evaluator.main(plot_examples=0, render_curves=False)


def _evaluate_nuscenes_predictions_subset(
    nusc: Any,
    result_path: str | Path,
    output_dir: Path,
    sample_tokens: list[str],
) -> dict[str, object]:
    """Evaluate an exact token subset using the official nuScenes metric stack.

    References:
    - DetectionEval implementation:
      https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py
    - common loader helpers:
      https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/common/loaders.py
    """

    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import (
        add_center_dist,
        filter_eval_boxes,
        load_gt_of_sample_tokens,
        load_prediction_of_sample_tokens,
    )
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import (
        DetectionBox,
        DetectionMetricDataList,
        DetectionMetrics,
    )

    cfg = config_factory("detection_cvpr_2019")
    pred_boxes, meta = load_prediction_of_sample_tokens(
        str(result_path),
        cfg.max_boxes_per_sample,
        DetectionBox,
        sample_tokens=sample_tokens,
        verbose=False,
    )
    gt_boxes = load_gt_of_sample_tokens(nusc, sample_tokens, DetectionBox, verbose=False)
    if set(pred_boxes.sample_tokens) != set(gt_boxes.sample_tokens):
        raise ValueError("subset prediction tokens do not match subset ground-truth tokens")

    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=False)

    start_time = time.time()
    metric_data_list = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            metric_data = accumulate(
                gt_boxes,
                pred_boxes,
                class_name,
                cfg.dist_fcn_callable,
                dist_th,
            )
            metric_data_list.set(class_name, dist_th, metric_data)

    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            metrics.add_label_ap(
                class_name,
                dist_th,
                calc_ap(metric_data, cfg.min_recall, cfg.min_precision),
            )
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            if class_name in ["traffic_cone"] and metric_name in [
                "attr_err",
                "vel_err",
                "orient_err",
            ]:
                tp = np.nan
            elif class_name in ["barrier"] and metric_name in ["attr_err", "vel_err"]:
                tp = np.nan
            else:
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)
    metrics.add_runtime(time.time() - start_time)

    metrics_summary = metrics.serialize()
    metrics_summary["meta"] = meta.copy()
    metrics_summary["evaluated_sample_tokens"] = list(sample_tokens)
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))
    (output_dir / "metrics_details.json").write_text(
        json.dumps(metric_data_list.serialize(), indent=2)
    )
    return metrics_summary
