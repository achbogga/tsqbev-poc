from __future__ import annotations

import torch

from tsqbev.contracts import QuerySeedBank, TeacherTargets
from tsqbev.distill import DistillationObjective, _align_teacher_targets


def test_geometry_aware_teacher_alignment_prefers_nearest_student_queries() -> None:
    query_refs = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    object_logits = torch.zeros(1, 2, 3)
    object_queries = torch.zeros(1, 2, 2)
    object_boxes = torch.zeros(1, 2, 9)
    teacher = TeacherTargets(
        object_features=torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]),
        object_boxes=torch.tensor(
            [
                [
                    [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        object_labels=torch.tensor([[0, 1]]),
        object_scores=torch.tensor([[0.95, 0.10]]),
        valid_mask=torch.tensor([[True, True]]),
    )

    aligned, mask = _align_teacher_targets(
        query_refs,
        object_logits,
        object_queries,
        object_boxes,
        teacher,
    )

    assert mask.tolist() == [[True, True]]
    assert aligned.object_boxes is not None
    assert torch.allclose(aligned.object_boxes[0, 0, :3], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(aligned.object_boxes[0, 1, :3], torch.tensor([10.0, 0.0, 0.0]))
    assert aligned.object_features is not None
    assert torch.allclose(aligned.object_features[0, 0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(aligned.object_features[0, 1], torch.tensor([1.0, 1.0]))


def test_distillation_objective_uses_geometry_aligned_teacher_rows() -> None:
    query_refs = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    object_logits = torch.zeros(1, 2, 3)
    object_queries = torch.zeros(1, 2, 2)
    object_boxes = torch.zeros(1, 2, 9)
    seed_bank = QuerySeedBank(
        embeddings=torch.zeros(1, 2, 4),
        refs_xyz=query_refs,
        scores=torch.ones(1, 2),
        source_ids=torch.zeros(1, 2, dtype=torch.long),
        keep_logits=torch.zeros(1, 2),
    )
    teacher = TeacherTargets(
        object_features=torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]),
        object_boxes=torch.tensor(
            [
                [
                    [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        object_labels=torch.tensor([[0, 1]]),
        object_scores=torch.tensor([[0.95, 0.10]]),
        valid_mask=torch.tensor([[True, True]]),
    )

    objective = DistillationObjective()
    losses = objective(
        object_logits=object_logits,
        object_queries=object_queries,
        object_boxes=object_boxes,
        seed_bank=seed_bank,
        teacher=teacher,
    )

    assert losses["kd_total"].item() > 0.0
    assert losses["kd_features"].item() > 0.0
    assert losses["kd_boxes"].item() > 0.0


def test_distillation_logits_follow_aligned_query_positions() -> None:
    query_refs = torch.tensor([[[0.0, 0.0, 0.0], [99.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    object_logits = torch.full((1, 3, 3), -8.0)
    object_logits[0, 0, 0] = 8.0
    object_logits[0, 2, 1] = 8.0
    object_queries = torch.zeros(1, 3, 2)
    object_boxes = torch.zeros(1, 3, 9)
    seed_bank = QuerySeedBank(
        embeddings=torch.zeros(1, 3, 4),
        refs_xyz=query_refs,
        scores=torch.ones(1, 3),
        source_ids=torch.zeros(1, 3, dtype=torch.long),
        keep_logits=torch.zeros(1, 3),
    )
    teacher = TeacherTargets(
        object_features=None,
        object_boxes=torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        object_labels=torch.tensor([[0, 1]]),
        object_scores=torch.tensor([[0.95, 0.90]]),
        valid_mask=torch.tensor([[True, True]]),
    )

    objective = DistillationObjective(feature_weight=0.0, box_weight=0.0, router_weight=0.0)
    losses = objective(
        object_logits=object_logits,
        object_queries=object_queries,
        object_boxes=object_boxes,
        seed_bank=seed_bank,
        teacher=teacher,
    )

    assert losses["kd_logits"].item() < 1e-2
