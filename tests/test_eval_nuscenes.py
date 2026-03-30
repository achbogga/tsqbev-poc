from __future__ import annotations

import torch

from tsqbev.eval_nuscenes import _rank_detection_queries


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
