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
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tsqbev.datasets import NuScenesDataset, collate_scene_examples
from tsqbev.labels import NUSCENES_DETECTION_NAMES
from tsqbev.model import TSQBEVModel
from tsqbev.quaternion import (
    quaternion_from_yaw,
    rotate_xy,
    rotation_matrix_from_quaternion,
    yaw_from_rotation_matrix,
)
from tsqbev.runtime import move_batch, resolve_device
from tsqbev.teacher_backends import TeacherProviderConfig, build_teacher_provider
from tsqbev.teacher_dataset import TeacherAugmentedDataset


def _car_ap_4m(evaluation: dict[str, object]) -> float:
    label_aps = evaluation.get("label_aps", {})
    if not isinstance(label_aps, dict):
        return 0.0
    car_aps = label_aps.get("car", {})
    if not isinstance(car_aps, dict):
        return 0.0
    value = 0.0
    for key in ("4.0", 4.0, 4):
        if key in car_aps:
            value = car_aps[key]
            break
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _geometry_gate_pass(geometry: dict[str, object] | None) -> bool:
    if geometry is None:
        return False
    try:
        boxes_mean = geometry.get("boxes_per_sample_mean", float("inf"))
        boxes_p95 = geometry.get("boxes_per_sample_p95", float("inf"))
        norm_p99 = geometry.get("ego_translation_norm_p99", float("inf"))
        norm_max = geometry.get("ego_translation_norm_max", float("inf"))
        return (
            float(cast(float | int | str, boxes_mean)) <= 40.0
            and float(cast(float | int | str, boxes_p95)) <= 60.0
            and float(cast(float | int | str, norm_p99)) <= 120.0
            and float(cast(float | int | str, norm_max)) <= 150.0
        )
    except (TypeError, ValueError):
        return False


def _is_better_calibration(
    candidate: dict[str, object],
    current_best: dict[str, object] | None,
) -> bool:
    if current_best is None:
        return True
    candidate_eval = candidate["evaluation"]
    current_eval = current_best["evaluation"]
    assert isinstance(candidate_eval, dict)
    assert isinstance(current_eval, dict)
    candidate_geometry = candidate.get("prediction_geometry")
    current_geometry = current_best.get("prediction_geometry")
    candidate_geometry_pass = _geometry_gate_pass(
        candidate_geometry if isinstance(candidate_geometry, dict) else None
    )
    current_geometry_pass = _geometry_gate_pass(
        current_geometry if isinstance(current_geometry, dict) else None
    )
    candidate_key = (
        candidate_geometry_pass,
        float(candidate_eval.get("nd_score", float("-inf"))),
        float(candidate_eval.get("mean_ap", float("-inf"))),
        _car_ap_4m(candidate_eval),
        -float(cast(float, candidate["score_threshold"])),
        -float(cast(int, candidate["top_k"])),
    )
    current_key = (
        current_geometry_pass,
        float(current_eval.get("nd_score", float("-inf"))),
        float(current_eval.get("mean_ap", float("-inf"))),
        _car_ap_4m(current_eval),
        -float(cast(float, current_best["score_threshold"])),
        -float(cast(int, current_best["top_k"])),
    )
    return candidate_key > current_key


def _rank_detection_queries(
    class_logits: torch.Tensor,
    objectness_logits: torch.Tensor | None,
    *,
    ranking_mode: str = "class_times_objectness",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return objectness-aware per-query scores and class ids."""

    class_scores, class_ids = class_logits.sigmoid().max(dim=-1)
    if ranking_mode == "quality_class_only":
        return class_scores, class_ids
    if objectness_logits is None:
        objectness_scores = torch.ones_like(class_scores)
    else:
        objectness_scores = objectness_logits.sigmoid()
    return class_scores * objectness_scores, class_ids


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def prediction_geometry_diagnostics(
    result_path: str | Path,
    *,
    dataroot: str | Path,
    version: str,
) -> dict[str, float]:
    """Measure exported result geometry in the ego frame.

    Result JSON translations are stored in the nuScenes global frame. Sanity
    thresholds should be measured in the ego frame instead.
    """

    payload = json.loads(Path(result_path).read_text())
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("nuScenes result JSON must contain a results dictionary")

    box_counts: list[float] = []
    ego_translation_norms: list[float] = []

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    ego_transform_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sample_token, sample_predictions in results.items():
        if not isinstance(sample_predictions, list):
            continue
        box_counts.append(float(len(sample_predictions)))
        if sample_token not in ego_transform_cache:
            sample = nusc.get("sample", sample_token)
            lidar_token = sample["data"]["LIDAR_TOP"]
            sample_data = nusc.get("sample_data", lidar_token)
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
            rotation = rotation_matrix_from_quaternion(ego_pose["rotation"])
            translation = np.asarray(ego_pose["translation"], dtype=np.float32)
            rotation_t = rotation.T
            ego_transform_cache[sample_token] = (
                rotation_t,
                -(rotation_t @ translation),
            )
        rotation_t, translation_t = ego_transform_cache[sample_token]
        for prediction in sample_predictions:
            if not isinstance(prediction, dict):
                continue
            translation = prediction.get("translation", [0.0, 0.0, 0.0])
            if not isinstance(translation, list) or len(translation) != 3:
                continue
            global_xyz = np.asarray(
                [float(component) for component in translation],
                dtype=np.float32,
            )
            ego_xyz = rotation_t @ global_xyz + translation_t
            ego_translation_norms.append(float(np.linalg.norm(ego_xyz)))

    return {
        "sample_count": float(len(results)),
        "boxes_per_sample_mean": float(sum(box_counts) / max(len(box_counts), 1)),
        "boxes_per_sample_p95": _percentile(box_counts, 95.0),
        "boxes_per_sample_max": float(max(box_counts) if box_counts else 0.0),
        "ego_translation_norm_mean": float(
            sum(ego_translation_norms) / max(len(ego_translation_norms), 1)
        ),
        "ego_translation_norm_p95": _percentile(ego_translation_norms, 95.0),
        "ego_translation_norm_p99": _percentile(ego_translation_norms, 99.0),
        "ego_translation_norm_max": float(
            max(ego_translation_norms) if ego_translation_norms else 0.0
        ),
    }


def export_sanity_diagnostics(
    result_path: str | Path,
    *,
    dataroot: str | Path,
    version: str,
) -> dict[str, float]:
    """Measure whether an exported prediction file is numerically sane enough to trust."""

    payload = json.loads(Path(result_path).read_text())
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("nuScenes result JSON must contain a results dictionary")

    geometry = prediction_geometry_diagnostics(result_path, dataroot=dataroot, version=version)
    max_sizes: list[float] = []
    score_values: list[float] = []
    saturated_scores = 0
    total_scores = 0
    for sample_predictions in results.values():
        if not isinstance(sample_predictions, list):
            continue
        for prediction in sample_predictions:
            if not isinstance(prediction, dict):
                continue
            size = prediction.get("size", [0.0, 0.0, 0.0])
            if isinstance(size, list) and len(size) == 3:
                max_sizes.append(max(float(size[0]), float(size[1]), float(size[2])))
            score = float(prediction.get("detection_score", 0.0))
            score_values.append(score)
            total_scores += 1
            if score >= 0.995:
                saturated_scores += 1

    size_sorted = sorted(max_sizes)
    size_p95 = _percentile(max_sizes, 95.0)
    score_p95 = _percentile(score_values, 95.0)
    score_mean = float(sum(score_values) / max(len(score_values), 1))
    saturated_fraction = float(saturated_scores / max(total_scores, 1))
    max_size = float(size_sorted[-1] if size_sorted else 0.0)

    sanity_ok = float(
        geometry["boxes_per_sample_mean"] <= 60.0
        and geometry["boxes_per_sample_p95"] <= 80.0
        and geometry["ego_translation_norm_p99"] <= 120.0
        and geometry["ego_translation_norm_max"] <= 150.0
        and max_size <= 50.0
        and size_p95 <= 20.0
        and score_mean <= 0.98
        and score_p95 <= 0.999
        and saturated_fraction <= 0.5
    )
    return {
        **geometry,
        "max_box_size_m": max_size,
        "box_size_p95_m": size_p95,
        "score_mean": score_mean,
        "score_p95": score_p95,
        "saturated_score_fraction": saturated_fraction,
        "sanity_ok": sanity_ok,
    }


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
    ranking_mode = getattr(model.config, "ranking_mode", "class_times_objectness")
    results: dict[str, list[dict[str, object]]] = {}

    for batch, metadata_list in loader:
        batch = move_batch(batch, resolved_device)
        outputs = model(batch)
        logits = outputs["object_logits"]
        boxes = outputs["object_boxes"]
        objectness_logits = outputs.get("objectness_logits")
        assert isinstance(logits, torch.Tensor)
        assert isinstance(boxes, torch.Tensor)
        if objectness_logits is not None and not isinstance(objectness_logits, torch.Tensor):
            raise TypeError("objectness_logits must be a tensor when present")

        metadata = metadata_list[0]
        sample_token = str(metadata["sample_token"])
        ego_to_global = np.asarray(metadata["ego_to_global"], dtype=np.float32)
        ego_yaw = yaw_from_rotation_matrix(ego_to_global[:3, :3])

        combined_scores, class_ids = _rank_detection_queries(
            logits[0],
            objectness_logits[0] if isinstance(objectness_logits, torch.Tensor) else None,
            ranking_mode=ranking_mode,
        )
        order = torch.argsort(combined_scores, descending=True)
        sample_results: list[dict[str, object]] = []
        for query_index in order[:top_k]:
            score = float(combined_scores[query_index])
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


def export_and_evaluate_nuscenes_grid(
    model: TSQBEVModel,
    *,
    dataroot: str | Path,
    version: str,
    split: str,
    output_dir: str | Path,
    score_threshold_candidates: list[float] | tuple[float, ...],
    top_k_candidates: list[int] | tuple[int, ...],
    device: str | None = None,
    sample_tokens: list[str] | None = None,
    teacher_provider_config: TeacherProviderConfig | None = None,
) -> dict[str, object]:
    """Export/evaluate a bounded threshold-topk grid and select the best candidate."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, object]] = []
    best_candidate: dict[str, object] | None = None

    unique_thresholds = sorted({float(value) for value in score_threshold_candidates})
    unique_top_k = sorted({int(value) for value in top_k_candidates})
    for score_threshold in unique_thresholds:
        for top_k in unique_top_k:
            candidate_slug = f"s{score_threshold:.2f}_k{top_k}"
            prediction_path = root / f"predictions_{candidate_slug}.json"
            export_nuscenes_predictions(
                model=model,
                dataroot=dataroot,
                version=version,
                split=split,
                output_path=prediction_path,
                score_threshold=score_threshold,
                top_k=top_k,
                device=device,
                sample_tokens=sample_tokens,
                teacher_provider_config=teacher_provider_config,
            )
            evaluation = evaluate_nuscenes_predictions(
                dataroot=dataroot,
                version=version,
                split=split,
                result_path=prediction_path,
                output_dir=root / f"eval_{candidate_slug}",
                sample_tokens=sample_tokens,
            )
            candidate = {
                "score_threshold": score_threshold,
                "top_k": top_k,
                "prediction_path": str(prediction_path),
                "evaluation": evaluation,
                "car_ap_4m": _car_ap_4m(evaluation),
                "prediction_geometry": prediction_geometry_diagnostics(
                    prediction_path,
                    dataroot=dataroot,
                    version=version,
                ),
            }
            candidates.append(candidate)
            if _is_better_calibration(candidate, best_candidate):
                best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("nuScenes calibration grid did not produce any candidate evaluation")

    payload: dict[str, object] = {
        "selected": best_candidate,
        "candidates": candidates,
    }
    (root / "calibration_summary.json").write_text(json.dumps(payload, indent=2))
    return payload
