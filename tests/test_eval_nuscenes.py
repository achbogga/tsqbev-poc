from __future__ import annotations

import torch

from tsqbev.eval_nuscenes import _car_ap_4m, _is_better_calibration, _rank_detection_queries


def test_objectness_aware_ranking_prefers_high_objectness_queries() -> None:
    class_logits = torch.tensor(
        [
            [8.0, -8.0],
            [0.0, 0.0],
            [4.0, -4.0],
        ]
    )
    objectness_logits = torch.tensor([-8.0, 8.0, 0.0])

    combined_scores, class_ids = _rank_detection_queries(class_logits, objectness_logits)
    order = torch.argsort(combined_scores, descending=True)

    assert int(class_ids[0]) == 0
    assert int(class_ids[1]) == 0
    assert int(class_ids[2]) == 0
    assert int(order[0]) == 1
    assert int(order[1]) == 2
    assert int(order[-1]) == 0
    assert combined_scores[1] > combined_scores[2] > combined_scores[0]


def test_quality_class_ranking_ignores_objectness_product() -> None:
    class_logits = torch.tensor(
        [
            [8.0, -8.0],
            [0.0, 0.0],
            [4.0, -4.0],
        ]
    )
    objectness_logits = torch.tensor([-8.0, 8.0, 0.0])

    combined_scores, class_ids = _rank_detection_queries(
        class_logits,
        objectness_logits,
        ranking_mode="quality_class_only",
    )
    order = torch.argsort(combined_scores, descending=True)

    assert int(class_ids[0]) == 0
    assert int(order[0]) == 0
    assert int(order[1]) == 2
    assert int(order[-1]) == 1


def test_car_ap_4m_accepts_numeric_distance_keys() -> None:
    evaluation = {"label_aps": {"car": {4.0: 0.42}}}
    assert _car_ap_4m(evaluation) == 0.42


def test_calibration_prefers_geometry_safe_candidate_over_slightly_higher_nds() -> None:
    geometry_safe = {
        "score_threshold": 0.05,
        "top_k": 32,
        "evaluation": {"nd_score": 0.14, "mean_ap": 0.18, "label_aps": {"car": {"4.0": 0.53}}},
        "prediction_geometry": {
            "boxes_per_sample_mean": 32.0,
            "boxes_per_sample_p95": 32.0,
            "ego_translation_norm_p99": 52.0,
            "ego_translation_norm_max": 71.0,
        },
    }
    overproducing = {
        "score_threshold": 0.05,
        "top_k": 112,
        "evaluation": {"nd_score": 0.149, "mean_ap": 0.185, "label_aps": {"car": {"4.0": 0.60}}},
        "prediction_geometry": {
            "boxes_per_sample_mean": 77.6,
            "boxes_per_sample_p95": 92.0,
            "ego_translation_norm_p99": 55.0,
            "ego_translation_norm_max": 71.4,
        },
    }

    assert _is_better_calibration(geometry_safe, overproducing) is True
    assert _is_better_calibration(overproducing, geometry_safe) is False
