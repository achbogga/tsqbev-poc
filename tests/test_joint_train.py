from __future__ import annotations

from tsqbev.train import _lane_batches_per_epoch, _make_round_robin_plan


def test_make_round_robin_plan_alternates_tasks_until_exhausted() -> None:
    plan = _make_round_robin_plan(("detection", 3), ("lane", 2))
    assert plan == ["detection", "lane", "detection", "lane", "detection"]


def test_make_round_robin_plan_skips_empty_sources() -> None:
    plan = _make_round_robin_plan(("detection", 0), ("lane", 2))
    assert plan == ["lane", "lane"]


def test_lane_batches_per_epoch_caps_lane_loader_by_detection_budget() -> None:
    lane_batches = _lane_batches_per_epoch(
        detection_batches=323,
        lane_batches=47533,
        lane_batch_multiplier=1.0,
    )
    assert lane_batches == 323


def test_lane_batches_per_epoch_respects_fractional_multiplier() -> None:
    lane_batches = _lane_batches_per_epoch(
        detection_batches=10,
        lane_batches=100,
        lane_batch_multiplier=0.5,
    )
    assert lane_batches == 5
