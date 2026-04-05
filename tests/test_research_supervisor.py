from __future__ import annotations

from tsqbev.research_supervisor import (
    _as_str_list,
    _critic_decision_from_planner,
    _heuristic_planner,
)


def test_heuristic_planner_detects_joint_metric_collapse() -> None:
    brief = {
        "current_state": [
            "joint_public_v3 collapsed at epoch 5 with NDS 0.0000 and mAP 0.0000",
        ]
    }
    decision = _heuristic_planner(brief)
    assert decision.active_bottleneck == "joint-metric-collapse"
    assert decision.force_priority_only is True
    assert "teacher_bag" in decision.suppress_tags


def test_critic_fallback_approves_low_token_burn_plan() -> None:
    brief = {"current_state": ["best frontier is v28 detection control"]}
    planner = _heuristic_planner(brief)
    critic = _critic_decision_from_planner(brief, planner)
    assert critic.approved is True
    assert critic.supervisor_policy["force_priority_only"] is True


def test_as_str_list_normalizes_scalar_strings() -> None:
    assert _as_str_list("reject this plan") == ["reject this plan"]
    assert _as_str_list(["a", 1]) == ["a", "1"]
