from __future__ import annotations

from pathlib import Path

from tsqbev.research_supervisor import (
    SupervisorState,
    _active_checklist_item_for_phase,
    _as_str_list,
    _critic_decision_from_planner,
    _heuristic_planner,
    _render_supervisor_report,
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


def test_active_checklist_item_maps_joint_collapse_to_multitask_reset() -> None:
    planner = _heuristic_planner(
        {"current_state": ["joint_public_v3 collapsed with NDS 0.0000 and mAP 0.0000"]}
    )
    item = _active_checklist_item_for_phase("launching_bounded_loop", planner_decision=planner)
    assert "Multitask Reset" in item


def test_render_supervisor_report_includes_phase_and_planner_fields(tmp_path: Path) -> None:
    repo_tmp = Path("/home/achbogga/projects/tsqbev-poc/tests/.tmp")
    state = SupervisorState(
        status="running",
        generated_at_utc="2026-04-05T18:00:00+00:00",
        repo_sha="abc123",
        current_branch="main",
        dataset_root="/data",
        artifact_root="/artifacts",
        attempted_invocations=1,
        completed_invocations=0,
        last_invocation_dir="/artifacts/invocation_001",
        last_selected_recipe=None,
        last_nds=None,
        last_map=None,
        last_publish_status=None,
        last_publish_message=None,
        memory_mode="server",
        memory_embedder="fastembed",
        planner_provider="openai",
        critic_provider="openai",
        notes=["alive"],
        active_phase="launching_bounded_loop",
        active_checklist_item=(
            "Run Queue / Launch the next validated detection run from the fixed control plane."
        ),
        planner_bottleneck="quality-vs-calibration-boundary",
        planner_objective="improve nds",
        planner_decision_path="/artifacts/invocation_001/planner_decision.json",
        critic_decision_path="/artifacts/invocation_001/critic_decision.json",
    )
    report = _render_supervisor_report(
        state,
        latest_brief_path=repo_tmp / "current.md",
        ledger_path=repo_tmp / "ledger.jsonl",
        stop_path=repo_tmp / "STOP",
    )
    assert "active phase" in report
    assert "planner bottleneck" in report
    assert "launching_bounded_loop" in report
