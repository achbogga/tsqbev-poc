from __future__ import annotations

from tsqbev.train import _make_round_robin_plan


def test_make_round_robin_plan_alternates_tasks_until_exhausted() -> None:
    plan = _make_round_robin_plan(("detection", 3), ("lane", 2))
    assert plan == ["detection", "lane", "detection", "lane", "detection"]


def test_make_round_robin_plan_skips_empty_sources() -> None:
    plan = _make_round_robin_plan(("detection", 0), ("lane", 2))
    assert plan == ["lane", "lane"]
