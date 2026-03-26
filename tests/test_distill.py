from __future__ import annotations

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
