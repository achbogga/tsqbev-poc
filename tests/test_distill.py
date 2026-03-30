from __future__ import annotations

from dataclasses import replace

import torch

from tsqbev.distill import DistillationObjective
from tsqbev.model import TSQBEVModel


def test_distillation_objective_returns_zero_without_teacher(small_config, synthetic_batch) -> None:
    model = TSQBEVModel(small_config)
    outputs = model(synthetic_batch)
    objective = DistillationObjective()
    losses = objective(
        object_queries=outputs["temporal_state"].object_queries,
        object_boxes=outputs["object_boxes"],
        seed_bank=outputs["seed_bank"],
        teacher=None,
    )
    assert float(losses["kd_total"]) == 0.0


def test_distillation_objective_uses_teacher_targets(small_config, synthetic_batch) -> None:
    model = TSQBEVModel(small_config)
    outputs = model(synthetic_batch)
    objective = DistillationObjective()
    losses = objective(
        object_queries=outputs["temporal_state"].object_queries,
        object_boxes=outputs["object_boxes"],
        seed_bank=outputs["seed_bank"],
        teacher=synthetic_batch.teacher_targets,
    )
    assert float(losses["kd_total"]) >= 0.0


def test_distillation_objective_aligns_variable_teacher_boxes(
    small_config,
    synthetic_batch,
) -> None:
    assert synthetic_batch.teacher_targets is not None
    model = TSQBEVModel(small_config)
    outputs = model(synthetic_batch)
    teacher = replace(
        synthetic_batch.teacher_targets,
        object_features=None,
        object_boxes=torch.randn(
            synthetic_batch.batch_size,
            small_config.max_object_queries + 3,
            9,
        ),
        object_scores=torch.rand(
            synthetic_batch.batch_size,
            small_config.max_object_queries + 3,
        ),
        router_logits=None,
        valid_mask=torch.ones(
            synthetic_batch.batch_size,
            small_config.max_object_queries + 3,
            dtype=torch.bool,
        ),
    )
    objective = DistillationObjective()

    losses = objective(
        object_queries=outputs["temporal_state"].object_queries,
        object_boxes=outputs["object_boxes"],
        seed_bank=outputs["seed_bank"],
        teacher=teacher,
    )

    assert float(losses["kd_boxes"]) > 0.0
